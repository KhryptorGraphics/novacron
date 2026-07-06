package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

// KVMDriverEnhanced implements the VMDriver interface for KVM-based VMs
type KVMDriverEnhanced struct {
	qemuBinaryPath string
	vmBasePath     string
	vms            map[string]*KVMVMInfo
	vmLock         sync.RWMutex
}

// KVMVMInfo stores information about a KVM VM
type KVMVMInfo struct {
	ID          string
	Config      VMConfig
	Process     *os.Process
	PID         int
	State       State
	DiskPath    string
	ConfigPath  string
	MonitorPath string
	VNCPort     int
	StartTime   time.Time
	StoppedTime *time.Time

	// Migration fields (empty/false for normal VMs). RuntimeDir decouples the
	// per-process sockets/pidfile/console from DiskPath so a migration dest can
	// share the source's disk+UEFI vars while keeping its own runtime dir.
	// ShareStorage adds file.locking=off to the writable disk and UEFI vars so
	// source and dest can open the same files (memory-only migration over
	// shared storage). IncomingURI, when set, launches qemu with -incoming.
	RuntimeDir   string
	ShareStorage bool
	IncomingURI  string

	// NUMA, when non-nil, makes buildQEMUArgs emit a -numa topology at launch.
	// QEMU fixes NUMA at machine init, so ConfigureNUMA stores it here for the
	// next boot (and refuses a running VM) rather than pretending live reconfig.
	// ConfigureNUMA also JSON-encodes the topology into Config.Tags["numa.topology"]
	// and persists config.json, so it survives a driver restart between
	// ConfigureNUMA and Start (adopt/reload rehydrates Config, not this field);
	// buildQEMUArgs falls back to that tag via effectiveNUMA when this field is nil.
	NUMA *NUMATopology
}

// NewKVMDriver creates a new KVM driver (main entry point)
func NewKVMDriver(config map[string]interface{}) (VMDriver, error) {
	qemuPath := ""
	if path, ok := config["qemu_path"].(string); ok {
		qemuPath = path
	}
	// Explicit qemu_path wins; otherwise pick the binary for the configured
	// target arch, falling back to the host arch inside newKVMDriverEnhanced.
	if qemuPath == "" {
		if arch, ok := config["arch"].(string); ok && arch != "" {
			qemuPath = defaultQEMUBinary(arch)
		}
	}

	vmBasePath := "/var/lib/novacron/vms"
	if path, ok := config["vm_path"].(string); ok && path != "" {
		vmBasePath = path
	}
	if path, ok := config["base_path"].(string); ok && path != "" {
		vmBasePath = path
	}

	return newKVMDriverEnhanced(qemuPath, vmBasePath)
}

// NewKVMDriverEnhanced creates a new enhanced KVM driver
func NewKVMDriverEnhanced(qemuPath string) (VMDriver, error) {
	return newKVMDriverEnhanced(qemuPath, "/var/lib/novacron/vms")
}

// defaultQEMUBinary returns the qemu-system binary name for the target
// architecture, defaulting to the host architecture (runtime.GOARCH) when arch
// is empty. This is what makes the driver resolve to qemu-system-aarch64 on
// arm64 hosts instead of the previously hardcoded x86_64 binary.
func defaultQEMUBinary(arch string) string {
	if arch == "" {
		arch = runtime.GOARCH
	}
	switch arch {
	case "arm64", "aarch64":
		return "qemu-system-aarch64"
	case "amd64", "x86_64", "x86", "386":
		return "qemu-system-x86_64"
	default:
		return "qemu-system-" + arch
	}
}

// kvmAccessible reports whether /dev/kvm is usable for hardware acceleration.
// When false, callers fall back to TCG so qemu still launches (e.g. when the
// running user is not in the kvm group).
func kvmAccessible() bool {
	f, err := os.OpenFile("/dev/kvm", os.O_RDWR, 0)
	if err != nil {
		return false
	}
	_ = f.Close()
	return true
}

// freeVNCPort probes for a free VNC port (display :N == 5900+N). qemu binds this
// at Start; a taken port makes qemu exit while Start looks successful, so we
// bind-test instead of the old `5900+len(vms)` guess (which collided with any
// external VNC on :0). ponytail: a tiny probe->bind TOCTOU race remains; if it
// is lost, launchVM's liveness confirm detects the dead qemu and retries with a
// fresh port (see isPortBindError), so a lost race self-heals rather than
// silently leaving the VM StateRunning-on-dead-qemu.
func freeVNCPort() int {
	for port := 5900; port < 6000; port++ {
		ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
		if err != nil {
			continue
		}
		_ = ln.Close()
		return port
	}
	return 5900 // fall back; qemu will surface the bind error
}

func newKVMDriverEnhanced(qemuPath, vmBasePath string) (VMDriver, error) {
	if qemuPath == "" {
		qemuPath = defaultQEMUBinary("")
	}
	if vmBasePath == "" {
		vmBasePath = "/var/lib/novacron/vms"
	}

	if !filepath.IsAbs(qemuPath) {
		resolvedPath, err := exec.LookPath(qemuPath)
		if err != nil {
			return nil, fmt.Errorf("QEMU binary %q not found in PATH: %w", qemuPath, err)
		}
		qemuPath = resolvedPath
	}

	// Check if QEMU binary exists
	if _, err := os.Stat(qemuPath); err != nil {
		return nil, fmt.Errorf("QEMU binary not found at %s: %w", qemuPath, err)
	}

	if err := os.MkdirAll(vmBasePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create VM base directory: %w", err)
	}

	d := &KVMDriverEnhanced{
		qemuBinaryPath: qemuPath,
		vmBasePath:     vmBasePath,
		vms:            make(map[string]*KVMVMInfo),
	}
	// Re-adopt any qemu processes that outlived a previous driver instance
	// (e.g. a server restart) so their in-memory state is not orphaned.
	d.adoptRunningVMs()
	return d, nil
}

// adoptRunningVMs repopulates d.vms from qemu processes that survived a restart
// of this driver. For each VM directory under vmBasePath it reads qemu.pid and,
// if that process is still alive, reconstructs a StateRunning KVMVMInfo so the
// restarted driver manages the live qemu instead of losing track of it.
// ponytail: called only from construction (single goroutine) so it takes no lock.
// ponytail: adopted VMs get no monitorVM goroutine, so an adopted qemu that dies
// on its own is only noticed on the next Stop/GetStatus — fine for restart re-sync.
func (d *KVMDriverEnhanced) adoptRunningVMs() {
	entries, err := os.ReadDir(d.vmBasePath)
	if err != nil {
		return
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		vmID := e.Name()
		vmDir := filepath.Join(d.vmBasePath, vmID)

		pidData, err := os.ReadFile(filepath.Join(vmDir, "qemu.pid"))
		if err != nil {
			continue // no pidfile: VM was never started or already cleaned up
		}
		pid, err := strconv.Atoi(strings.TrimSpace(string(pidData)))
		if err != nil || pid <= 0 {
			continue
		}
		if syscall.Kill(pid, 0) != nil {
			continue // process is dead
		}
		// Guard against a recycled PID: after a host reboot the pidfile can name a
		// live process that is NOT our qemu (PID reuse). Only adopt when the
		// process cmdline is actually this VM's qemu, else a later Stop would
		// signal an unrelated process.
		if cmdline, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid)); err != nil || !containsQEMUAndVMID(string(cmdline), vmID) {
			log.Printf("Skipping adoption of PID %d for VM %s: not this VM's qemu (stale/recycled pidfile)", pid, vmID)
			continue
		}

		proc, _ := os.FindProcess(pid) // never errors on Unix
		vmInfo := &KVMVMInfo{
			ID:          vmID,
			Process:     proc,
			PID:         pid,
			State:       StateRunning,
			DiskPath:    filepath.Join(vmDir, "disk.qcow2"),
			ConfigPath:  filepath.Join(vmDir, "config.json"),
			MonitorPath: filepath.Join(vmDir, "monitor.sock"),
			StartTime:   time.Now(), // real start time is unknown; approximate
		}
		if data, err := os.ReadFile(vmInfo.ConfigPath); err == nil {
			_ = json.Unmarshal(data, &vmInfo.Config)
		}
		d.vms[vmID] = vmInfo
		// Adopted qemu is not our child, so cmd.Wait is unavailable; poll instead
		// so an adopted VM that dies is noticed (previously only seen on next op).
		go d.monitorAdoptedVM(vmID, pid)
		log.Printf("Adopted running KVM VM %s (PID %d) on driver restart", vmID, pid)
	}
}

// monitorAdoptedVM polls an adopted qemu (a non-child process, so cmd.Wait can't
// be used) and marks the VM stopped once its process exits.
func (d *KVMDriverEnhanced) monitorAdoptedVM(vmID string, pid int) {
	for {
		time.Sleep(3 * time.Second)
		if syscall.Kill(pid, 0) == nil {
			continue // still alive
		}
		d.vmLock.Lock()
		if vmInfo, ok := d.vms[vmID]; ok && vmInfo.PID == pid && vmInfo.State == StateRunning {
			now := time.Now()
			vmInfo.State = StateStopped
			vmInfo.StoppedTime = &now
			vmInfo.Process = nil
			vmInfo.PID = 0
			log.Printf("Adopted KVM VM %s (PID %d) exited; marked stopped", vmID, pid)
		}
		d.vmLock.Unlock()
		return
	}
}

// Create creates a new KVM VM
func (d *KVMDriverEnhanced) Create(ctx context.Context, config VMConfig) (string, error) {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	vmID := config.ID
	if vmID == "" {
		return "", fmt.Errorf("VM ID is required")
	}

	log.Printf("Creating KVM VM %s (%s)", config.Name, vmID)

	// Create VM directory
	vmDir := filepath.Join(d.vmBasePath, vmID)
	if err := os.MkdirAll(vmDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create VM directory: %w", err)
	}

	// Roll back the freshly-created VM dir on any failure before the VM is
	// registered in d.vms, so a partial create (disk-image or config-save
	// failure) leaves no orphaned STORAGE_PATH/vms/<id>/ with no manager entry.
	created := false
	defer func() {
		if !created {
			os.RemoveAll(vmDir)
		}
	}()

	// Create disk image
	diskPath := filepath.Join(vmDir, "disk.qcow2")
	if err := d.createDiskImage(ctx, config, diskPath); err != nil {
		return "", err
	}

	// Create VM info
	vmInfo := &KVMVMInfo{
		ID:          vmID,
		Config:      config,
		State:       StateCreated,
		DiskPath:    diskPath,
		ConfigPath:  filepath.Join(vmDir, "config.json"),
		MonitorPath: filepath.Join(vmDir, "monitor.sock"),
		VNCPort:     freeVNCPort(), // probed free display; refreshed at Start
		StartTime:   time.Now(),
	}

	// Save config
	if err := d.saveVMConfig(vmInfo); err != nil {
		return "", fmt.Errorf("failed to save VM config: %w", err)
	}

	d.vms[vmID] = vmInfo
	created = true

	log.Printf("Created KVM VM %s with disk %s", vmID, diskPath)
	return vmID, nil
}

// Start starts a KVM VM
func (d *KVMDriverEnhanced) Start(ctx context.Context, vmID string) error {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vmInfo.State == StateRunning {
		return fmt.Errorf("VM %s is already running", vmID)
	}

	log.Printf("Starting KVM VM %s", vmID)

	// Refresh the VNC display to a currently-free port right before qemu binds.
	vmInfo.VNCPort = freeVNCPort()

	return d.launchVM(vmID, vmInfo)
}

// launchVM builds the QEMU command for vmInfo and starts the process. The
// caller must hold d.vmLock. Shared by Start and the migration entry points
// (StartMigrationSource / StartIncoming) so the dest mirrors the source args.
func (d *KVMDriverEnhanced) launchVM(vmID string, vmInfo *KVMVMInfo) error {
	const maxAttempts = 3
	var lastErr error
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		sockDir := d.runtimeDir(vmInfo)
		qmpSock := filepath.Join(sockDir, "qmp.sock")
		_ = os.Remove(qmpSock) // drop any stale socket from a prior dead qemu

		cmd := exec.Command(d.qemuBinaryPath, d.buildQEMUArgs(vmInfo)...)
		cmd.Dir = sockDir
		// Capture qemu's stderr; otherwise a dead-on-arrival qemu is a silent
		// exit-status-1 (its error goes to /dev/null). The child inherits its own
		// dup of the fd at Start, so closing our copy afterward is safe.
		if f, err := os.Create(filepath.Join(cmd.Dir, "qemu-stderr.log")); err == nil {
			cmd.Stderr = f
			defer f.Close()
		}

		if err := cmd.Start(); err != nil {
			return fmt.Errorf("failed to start QEMU: %w", err)
		}
		vmInfo.Process = cmd.Process
		vmInfo.PID = cmd.Process.Pid
		go d.monitorVM(vmID, cmd)

		// Liveness confirm before declaring StateRunning. A qemu that dies on
		// arrival (bad args, missing firmware, or a VNC-port clash from the
		// freeVNCPort probe->bind TOCTOU) still returns success from cmd.Start();
		// only a live qemu opens its QMP socket. This replaces the old optimistic
		// StateRunning that masked dead-on-arrival launches.
		if err := waitQMPUp(qmpSock, 3*time.Second); err == nil {
			vmInfo.State = StateRunning
			vmInfo.StartTime = time.Now()
			log.Printf("Started KVM VM %s with PID %d", vmID, vmInfo.PID)
			return nil
		} else {
			lastErr = err
			stderr := tailQEMUStderr(cmd.Dir)
			// A lost VNC-port race is recoverable: pick a fresh display and retry.
			if attempt < maxAttempts && isPortBindError(stderr) {
				log.Printf("KVM VM %s qemu failed to bind (attempt %d/%d), retrying with fresh VNC port: %s",
					vmID, attempt, maxAttempts, firstLine(stderr))
				vmInfo.VNCPort = freeVNCPort()
				continue
			}
			now := time.Now()
			vmInfo.State = StateFailed
			vmInfo.StoppedTime = &now
			return fmt.Errorf("QEMU for VM %s failed to come up: %w; stderr: %s", vmID, err, stderr)
		}
	}
	return fmt.Errorf("QEMU for VM %s failed to come up after %d attempts: %w", vmID, maxAttempts, lastErr)
}

// waitQMPUp waits until qemu has opened its QMP socket (proof it launched and is
// alive) or the timeout elapses. A successful connect is enough; it does not
// negotiate, and closes immediately so the real QMP clients connect cleanly.
func waitQMPUp(qmpSock string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		if c, err := net.DialTimeout("unix", qmpSock, 500*time.Millisecond); err == nil {
			c.Close()
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("QMP socket %s not up within %s", qmpSock, timeout)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// tailQEMUStderr returns the tail of a VM's captured qemu stderr for diagnostics.
func tailQEMUStderr(dir string) string {
	data, err := os.ReadFile(filepath.Join(dir, "qemu-stderr.log"))
	if err != nil {
		return "(no stderr)"
	}
	if len(data) > 600 {
		data = data[len(data)-600:]
	}
	return strings.TrimSpace(string(data))
}

// isPortBindError reports whether qemu stderr indicates a socket/port bind
// failure (e.g. a VNC display already taken), which a fresh port can recover.
func isPortBindError(stderr string) bool {
	s := strings.ToLower(stderr)
	return strings.Contains(s, "could not bind") ||
		strings.Contains(s, "failed to bind") ||
		strings.Contains(s, "address already in use") ||
		(strings.Contains(s, "vnc") && strings.Contains(s, "bind"))
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// Stop stops a KVM VM
func (d *KVMDriverEnhanced) Stop(ctx context.Context, vmID string) error {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vmInfo.State != StateRunning {
		return fmt.Errorf("VM %s is not running", vmID)
	}

	log.Printf("Stopping KVM VM %s", vmID)

	// Terminate via the shared poll-based path (reliable for adopted VMs).
	if err := d.stopVMInternal(vmInfo); err != nil {
		return err
	}

	log.Printf("Stopped KVM VM %s", vmID)
	return nil
}

// Delete deletes a KVM VM
func (d *KVMDriverEnhanced) Delete(ctx context.Context, vmID string) error {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	// Stop the VM if it's running
	if vmInfo.State == StateRunning {
		if err := d.stopVMInternal(vmInfo); err != nil {
			log.Printf("Warning: Failed to stop VM %s before deletion: %v", vmID, err)
		}
	}

	log.Printf("Deleting KVM VM %s", vmID)

	// Remove VM directory
	vmDir := filepath.Dir(vmInfo.DiskPath)
	if err := os.RemoveAll(vmDir); err != nil {
		log.Printf("Warning: Failed to remove VM directory %s: %v", vmDir, err)
	}

	delete(d.vms, vmID)

	log.Printf("Deleted KVM VM %s", vmID)
	return nil
}

// GetStatus returns the status of a VM
func (d *KVMDriverEnhanced) GetStatus(ctx context.Context, vmID string) (VMState, error) {
	d.vmLock.RLock()
	defer d.vmLock.RUnlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return VMState(""), fmt.Errorf("VM %s not found", vmID)
	}

	return VMState(vmInfo.State), nil
}

// GetInfo returns information about a VM
func (d *KVMDriverEnhanced) GetInfo(ctx context.Context, vmID string) (*VMInfo, error) {
	d.vmLock.RLock()
	defer d.vmLock.RUnlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}

	info := &VMInfo{
		ID:        vmInfo.ID,
		Name:      vmInfo.Config.Name,
		State:     vmInfo.State,
		PID:       vmInfo.PID,
		CPUShares: vmInfo.Config.CPUShares,
		MemoryMB:  vmInfo.Config.MemoryMB,
		CreatedAt: vmInfo.StartTime,
		StartedAt: &vmInfo.StartTime,
		StoppedAt: vmInfo.StoppedTime,
		Tags:      vmInfo.Config.Tags,
		NetworkID: vmInfo.Config.NetworkID,
		RootFS:    vmInfo.DiskPath,
	}

	return info, nil
}

// GetMetrics returns performance metrics for a VM
func (d *KVMDriverEnhanced) GetMetrics(ctx context.Context, vmID string) (*VMInfo, error) {
	// For now, return the same as GetInfo
	// In a real implementation, this would collect real-time metrics
	return d.GetInfo(ctx, vmID)
}

// ListVMs returns a list of all VMs
func (d *KVMDriverEnhanced) ListVMs(ctx context.Context) ([]VMInfo, error) {
	d.vmLock.RLock()
	defer d.vmLock.RUnlock()

	vms := make([]VMInfo, 0, len(d.vms))
	for _, vmInfo := range d.vms {
		info := VMInfo{
			ID:        vmInfo.ID,
			Name:      vmInfo.Config.Name,
			State:     vmInfo.State,
			PID:       vmInfo.PID,
			CPUShares: vmInfo.Config.CPUShares,
			MemoryMB:  vmInfo.Config.MemoryMB,
			CreatedAt: vmInfo.StartTime,
			StartedAt: &vmInfo.StartTime,
			StoppedAt: vmInfo.StoppedTime,
			Tags:      vmInfo.Config.Tags,
			NetworkID: vmInfo.Config.NetworkID,
			RootFS:    vmInfo.DiskPath,
		}
		vms = append(vms, info)
	}

	return vms, nil
}

// Optional operation support
func (d *KVMDriverEnhanced) SupportsPause() bool    { return true }
func (d *KVMDriverEnhanced) SupportsResume() bool   { return true }
func (d *KVMDriverEnhanced) SupportsSnapshot() bool { return true }
func (d *KVMDriverEnhanced) SupportsMigrate() bool  { return true }

// Pause pauses a VM
func (d *KVMDriverEnhanced) Pause(ctx context.Context, vmID string) error {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vmInfo.State != StateRunning {
		return fmt.Errorf("VM %s is not running", vmID)
	}

	// Send SIGSTOP to pause the QEMU process
	if err := vmInfo.Process.Signal(os.Signal(syscall.SIGSTOP)); err != nil {
		return fmt.Errorf("failed to pause VM: %w", err)
	}

	vmInfo.State = StatePaused
	log.Printf("Paused KVM VM %s", vmID)
	return nil
}

// Resume resumes a paused VM
func (d *KVMDriverEnhanced) Resume(ctx context.Context, vmID string) error {
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vmInfo.State != StatePaused {
		return fmt.Errorf("VM %s is not paused", vmID)
	}

	// Send SIGCONT to resume the QEMU process
	if err := vmInfo.Process.Signal(os.Signal(syscall.SIGCONT)); err != nil {
		return fmt.Errorf("failed to resume VM: %w", err)
	}

	vmInfo.State = StateRunning
	log.Printf("Resumed KVM VM %s", vmID)
	return nil
}

// Snapshot creates a snapshot of a VM
func (d *KVMDriverEnhanced) Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error) {
	d.vmLock.RLock()
	defer d.vmLock.RUnlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return "", fmt.Errorf("VM %s not found", vmID)
	}

	snapshotID := fmt.Sprintf("%s-%s-%d", vmID, name, time.Now().Unix())

	// Create snapshot using qemu-img
	cmd := exec.CommandContext(ctx, "qemu-img", "snapshot", "-c", snapshotID, vmInfo.DiskPath)
	if output, err := cmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("failed to create snapshot: %w, output: %s", err, string(output))
	}

	log.Printf("Created snapshot %s for VM %s", snapshotID, vmID)
	return snapshotID, nil
}

// Migrate, StartMigrationSource, StartIncoming and the QMP client live in
// driver_kvm_migrate.go.

// Private helper methods

func (d *KVMDriverEnhanced) buildQEMUArgs(vmInfo *KVMVMInfo) []string {
	// Machine type is arch-specific: aarch64 uses "virt", x86 uses "pc".
	machine := "pc"
	if strings.Contains(d.qemuBinaryPath, "aarch64") {
		machine = "virt"
	}
	// -cpu host only works under KVM. Under TCG use a concrete model: on aarch64
	// "max"/"cortex-a57" fault stock kernels here (Synchronous Exception at the
	// EFI stub); cortex-a72 boots stock cloud images (verified with cirros).
	accel, cpu := "tcg", "max"
	if machine == "virt" {
		cpu = "cortex-a72"
	}
	if kvmAccessible() {
		accel, cpu = "kvm", "host"
	}
	mem := vmInfo.Config.MemoryMB
	if mem <= 0 {
		mem = 128 // qemu rejects -m 0
	}
	// CPUShares is a scheduling weight (NewVM defaults it to 1024), not a vCPU
	// count. Passing it verbatim yields e.g. -smp 1024, which qemu rejects
	// ("max CPUs supported by machine 'virt' is 512") and the VM never boots.
	// Clamp to the host's logical CPUs so a default create still starts.
	// ponytail: host-core cap; add a real vCPU field to VMConfig if overcommit is ever needed.
	cpus := vmInfo.Config.CPUShares
	if cpus <= 0 {
		cpus = 1 // qemu rejects -smp 0
	} else if maxCPUs := runtime.NumCPU(); cpus > maxCPUs {
		cpus = maxCPUs
	}

	// vCPU/memory hotplug headroom is OPT-IN via Config.Tags so a default VM
	// emits the EXACT same -smp/-m as before (Gate-1 boot / Gate-2 migration must
	// stay byte-for-byte unaffected). Absent/zero tags -> smpArg/memArg are just
	// the plain counts. "hotplug.maxvcpus" > cpus adds ,maxcpus=N (query-
	// hotpluggable-cpus + device_add a vCPU later); "hotplug.maxmem_mb" > mem adds
	// ,slots=S,maxmem=M (object_add memory-backend-ram + device_add pc-dimm later).
	smpArg := strconv.Itoa(cpus)
	if maxVCPUs := tagInt(vmInfo.Config.Tags, "hotplug.maxvcpus"); maxVCPUs > cpus {
		smpArg = fmt.Sprintf("%d,maxcpus=%d", cpus, maxVCPUs)
	}
	memArg := strconv.Itoa(mem)
	if maxMem := tagInt(vmInfo.Config.Tags, "hotplug.maxmem_mb"); maxMem > mem {
		slots := tagInt(vmInfo.Config.Tags, "hotplug.mem_slots")
		if slots <= 0 {
			slots = 2 // ponytail: default 2 DIMM slots when only maxmem is requested
		}
		memArg = fmt.Sprintf("%d,slots=%d,maxmem=%dM", mem, slots, maxMem)
	}

	// ponytail: no -daemonize — keeps qemu as a tracked child so cmd.Wait()
	// in monitorVM and driver.Stop reflect the real process; -pidfile still
	// lets a restarted server rediscover the running qemu.
	// Per-process runtime dir (sockets/console/pidfile). Defaults to the disk's
	// dir; a migration dest overrides it so it can share the source disk while
	// keeping its own sockets. lock disables the qcow2/pflash write-lock for
	// shared-storage migration so source and dest can open the same files.
	sockDir := d.runtimeDir(vmInfo)
	lock := ""
	if vmInfo.ShareStorage {
		lock = ",file.locking=off"
	}

	args := []string{
		"-machine", machine + ",accel=" + accel,
		"-cpu", cpu,
		"-m", memArg,
		"-smp", smpArg,
		"-netdev", "user,id=net0",
		"-device", "virtio-net-pci,netdev=net0",
		"-vnc", fmt.Sprintf(":%d", vmInfo.VNCPort-5900),
		"-monitor", fmt.Sprintf("unix:%s,server,nowait", vmInfo.MonitorPath),
		// Dedicated QMP socket for programmatic control (migration handshake).
		"-qmp", fmt.Sprintf("unix:%s,server,nowait", filepath.Join(sockDir, "qmp.sock")),
		// Capture the guest serial console so boot is observable.
		"-serial", "file:" + filepath.Join(sockDir, "console.log"),
		"-pidfile", filepath.Join(sockDir, "qemu.pid"),
	}

	// NUMA topology (opt-in; set via ConfigureNUMA before Start). Emitted at
	// launch because QEMU fixes NUMA at machine init. Nil topology -> no args, so
	// a default VM is unaffected. effectiveNUMA falls back to the topology persisted
	// in Config.Tags so it survives a driver restart between ConfigureNUMA and Start.
	args = append(args, numaArgs(effectiveNUMA(vmInfo))...)

	// IOThreads (opt-in via Config.Tags["iothreads"]=N). Emit N iothread objects
	// (iothread0..iothreadN-1) so query-iothreads returns them and
	// ConfigureCPUPinning can actually pin their host threads; the primary disk
	// below runs its virtio-blk dataplane on iothread0. Absent/zero tag -> no args
	// AND the disk device string is unchanged, so a default VM (and any migration
	// cutover, which never sets this tag) is byte-for-byte unaffected.
	iothreads := tagInt(vmInfo.Config.Tags, "iothreads")
	args = append(args, iothreadArgs(iothreads)...)

	// PCIe hotplug slots: the aarch64 "virt" root bus (pcie.0) refuses
	// hot-plugging, so pre-provision a few pcie-root-ports for HotPlugDevice to
	// attach disks/NICs onto. x86 "pc" (i440fx) hot-plugs onto pci.0 natively and
	// needs none, so this is virt-only (pcie-root-port is invalid on i440fx).
	// ponytail: hotplugRootPortCount fixed ports -> that many concurrent
	// hot-plugs; raise it (or device_add ports on demand) if more are needed.
	if machine == "virt" {
		for i := 0; i < hotplugRootPortCount; i++ {
			args = append(args, "-device",
				fmt.Sprintf("pcie-root-port,id=%s%d,chassis=%d", hotplugRootPortPrefix, i, i))
		}
	}

	// Primary boot disk as a named -blockdev (not -drive if=virtio) so ANY running
	// VM can be drive-mirror'd (block-migration source) and NBD-exported (dest) by
	// node name -- block migration must work on normally-created VMs, and the
	// source/dest device topology must match for the RAM stream. file.locking=off
	// (shared-storage migration) rides the file child. Boot with this form is
	// verified on arm64 virt+UEFI and x86 KVM.
	// Attach the primary disk to iothread0 ONLY when iothreads were requested; the
	// -device string is otherwise byte-for-byte identical to before (migration
	// cutover depends on the exact source/dest device topology).
	blkDev := "virtio-blk-pci,drive=" + kvmMigDiskNode
	if iothreads > 0 {
		blkDev += ",iothread=iothread0"
	}
	args = append(args,
		"-blockdev", fmt.Sprintf("node-name=%s,driver=qcow2,file.driver=file,file.filename=%s%s", kvmMigDiskNode, vmInfo.DiskPath, lock),
		"-device", blkDev)

	// aarch64 "virt" needs UEFI firmware (pflash) to boot a disk image.
	if machine == "virt" {
		if code, vars, ok := d.ensureUEFI(filepath.Dir(vmInfo.DiskPath)); ok {
			args = append(args,
				"-drive", "if=pflash,format=raw,unit=0,file="+code+",readonly=on",
				"-drive", "if=pflash,format=raw,unit=1,file="+vars+lock,
			)
		}
	}

	if vmInfo.Config.CloudInitISO != "" {
		args = append(
			args,
			"-drive",
			fmt.Sprintf("file=%s,format=raw,media=cdrom,readonly=on", vmInfo.Config.CloudInitISO),
		)
	}

	// Add memory balloon
	args = append(args, "-device", "virtio-balloon-pci")

	// Add virtio-rng for entropy
	args = append(args, "-device", "virtio-rng-pci")

	// Migration destination: start paused waiting for the incoming stream.
	if vmInfo.IncomingURI != "" {
		args = append(args, "-incoming", vmInfo.IncomingURI)
	}

	return args
}

// tagInt reads an integer opt-in flag from a VM's Config.Tags map, returning 0
// when the key is absent or unparseable. Tags is used (rather than a new VMConfig
// field) because VMConfig lives in another file this driver does not own.
func tagInt(tags map[string]string, key string) int {
	if tags == nil {
		return 0
	}
	n, err := strconv.Atoi(strings.TrimSpace(tags[key]))
	if err != nil {
		return 0
	}
	return n
}

// numaArgs builds the -numa launch args for a topology, one memory-backend-ram +
// -numa node per NUMANode. Returns nil for a nil/empty topology (default VM).
// ponytail: the caller's node memory must sum to the VM's -m size and its node
// cpus must fall within -smp; QEMU rejects a mismatch at launch (surfaced by
// launchVM's stderr capture) rather than silently booting wrong.
func numaArgs(topo *NUMATopology) []string {
	if topo == nil || len(topo.Nodes) == 0 {
		return nil
	}
	var out []string
	for _, n := range topo.Nodes {
		memMB := n.MemoryMB
		if memMB <= 0 {
			memMB = 128 // ponytail: qemu rejects a zero-sized memory-backend
		}
		backend := fmt.Sprintf("numaram%d", n.ID)
		out = append(out, "-object", fmt.Sprintf("memory-backend-ram,id=%s,size=%dM", backend, memMB))
		node := fmt.Sprintf("node,nodeid=%d,memdev=%s", n.ID, backend)
		if n.CPUs != "" {
			node += ",cpus=" + n.CPUs
		}
		out = append(out, "-numa", node)
	}
	return out
}

// numaTopologyTag is the Config.Tags key under which ConfigureNUMA JSON-encodes
// the NUMA topology so it persists in config.json across a driver restart (Tags
// is used because VMConfig lives in a file this driver does not own).
const numaTopologyTag = "numa.topology"

// effectiveNUMA returns the topology to emit at launch: the in-memory NUMA field
// if set, else the one persisted in Config.Tags["numa.topology"] by ConfigureNUMA.
// The tag fallback is what makes NUMA survive a driver restart between
// ConfigureNUMA and Start (adopt/reload rehydrates Config, not the NUMA field).
// Returns nil when neither is present, so a default VM emits no -numa.
func effectiveNUMA(vmInfo *KVMVMInfo) *NUMATopology {
	if vmInfo.NUMA != nil {
		return vmInfo.NUMA
	}
	return decodeNUMATag(vmInfo.Config.Tags)
}

// decodeNUMATag decodes the JSON topology persisted under numaTopologyTag, or nil
// when the tag is absent/empty/unparseable (logged) or has no nodes.
func decodeNUMATag(tags map[string]string) *NUMATopology {
	raw := tags[numaTopologyTag] // nil map read is "" — no need to nil-check
	if raw == "" {
		return nil
	}
	var topo NUMATopology
	if err := json.Unmarshal([]byte(raw), &topo); err != nil {
		log.Printf("ignoring unparseable %s tag: %v", numaTopologyTag, err)
		return nil
	}
	if len(topo.Nodes) == 0 {
		return nil
	}
	return &topo
}

// iothreadArgs emits N `-object iothread` objects with ids iothread0..iothreadN-1
// so query-iothreads returns them (see ConfigureCPUPinning / pinIOThreads) and the
// primary disk can run its dataplane on iothread0. Opt-in via Config.Tags
// ["iothreads"]; n<=0 (default) -> nil, so a default VM's args are unchanged.
func iothreadArgs(n int) []string {
	if n <= 0 {
		return nil
	}
	out := make([]string, 0, n*2)
	for i := 0; i < n; i++ {
		out = append(out, "-object", fmt.Sprintf("iothread,id=iothread%d", i))
	}
	return out
}

func (d *KVMDriverEnhanced) createDiskImage(ctx context.Context, config VMConfig, diskPath string) error {
	baseImagePath := config.Image
	if baseImagePath == "" {
		baseImagePath = config.RootFS
	}

	if baseImagePath != "" {
		// Resolve http(s) image refs to a cached local file (fetched once).
		resolved, err := d.resolveBootImage(ctx, baseImagePath)
		if err != nil {
			return err
		}
		baseImagePath = resolved

		// Copy the base into a fresh qcow2 boot disk. convert handles any input
		// format (raw or qcow2) and leaves no backing-file dependency.
		// ponytail: full copy, not COW backing — fine for small cloud images;
		// switch to `-b <base> -F <fmt>` if base images get large.
		createCmd := exec.CommandContext(ctx, "qemu-img", "convert", "-O", "qcow2", baseImagePath, diskPath)
		if output, err := createCmd.CombinedOutput(); err != nil {
			return fmt.Errorf("failed to create disk image from base image: %w, output: %s", err, string(output))
		}

		if config.DiskSizeGB > 0 {
			resizeCmd := exec.CommandContext(ctx, "qemu-img", "resize", diskPath, fmt.Sprintf("%dG", config.DiskSizeGB))
			if output, err := resizeCmd.CombinedOutput(); err != nil {
				return fmt.Errorf("failed to resize boot disk: %w, output: %s", err, string(output))
			}
		}

		return nil
	}

	diskSizeMB := 8192 // Default 8GB
	if config.DiskSizeGB > 0 {
		diskSizeMB = config.DiskSizeGB * 1024
	} else if config.MemoryMB > 0 {
		diskSizeMB = config.MemoryMB * 10 // 10x memory size for disk
	}

	createCmd := exec.CommandContext(ctx, "qemu-img", "create",
		"-f", "qcow2",
		diskPath,
		fmt.Sprintf("%dM", diskSizeMB))

	if output, err := createCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to create disk image: %w, output: %s", err, string(output))
	}

	return nil
}

// imageCacheDir is a writable sibling of the VM base dir holding fetched boot
// images and the padded UEFI firmware.
func (d *KVMDriverEnhanced) imageCacheDir() string {
	return filepath.Join(filepath.Dir(d.vmBasePath), "images")
}

// resolveBootImage returns a local path for image. If image is an http(s) URL it
// is downloaded once into the image cache and reused thereafter.
func (d *KVMDriverEnhanced) resolveBootImage(ctx context.Context, image string) (string, error) {
	if !strings.HasPrefix(image, "http://") && !strings.HasPrefix(image, "https://") {
		return image, nil // local path
	}
	cacheDir := d.imageCacheDir()
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create image cache: %w", err)
	}
	dest := filepath.Join(cacheDir, filepath.Base(image))
	if fi, err := os.Stat(dest); err == nil && fi.Size() > 0 {
		return dest, nil // cache hit
	}
	if err := downloadFile(ctx, image, dest); err != nil {
		return "", fmt.Errorf("failed to fetch boot image %s: %w", image, err)
	}
	log.Printf("Fetched boot image %s -> %s", image, dest)
	return dest, nil
}

func downloadFile(ctx context.Context, url, dest string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %s", resp.Status)
	}
	tmp := dest + ".part"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if _, err := io.Copy(f, resp.Body); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dest)
}

// aarch64 "virt" pflash banks are 64 MiB each; firmware must be padded to fit.
const uefiFlashSize = 64 * 1024 * 1024

// edk2CodePaths are the common locations of the aarch64 UEFI code image.
var edk2CodePaths = []string{
	"/usr/share/qemu-efi-aarch64/QEMU_EFI.fd",
	"/usr/share/AAVMF/AAVMF_CODE.fd",
	"/usr/share/edk2/aarch64/QEMU_EFI.fd",
	"/usr/share/edk2/aarch64/QEMU_EFI-silent.fd",
}

// ensureUEFI returns readonly CODE and per-VM writable VARS pflash images for an
// aarch64 guest. ok=false (firmware absent) means the caller should skip pflash;
// the VM still spawns but won't UEFI-boot.
func (d *KVMDriverEnhanced) ensureUEFI(vmDir string) (code, vars string, ok bool) {
	var src string
	for _, p := range edk2CodePaths {
		if _, err := os.Stat(p); err == nil {
			src = p
			break
		}
	}
	if src == "" {
		return "", "", false
	}

	cacheDir := d.imageCacheDir()
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return "", "", false
	}
	code = filepath.Join(cacheDir, "edk2-aarch64-code-64m.fd")
	if fi, err := os.Stat(code); err != nil || fi.Size() != uefiFlashSize {
		if err := padFileTo(src, code, uefiFlashSize); err != nil {
			log.Printf("UEFI code pad failed: %v", err)
			return "", "", false
		}
	}

	vars = filepath.Join(vmDir, "efivars.fd")
	if fi, err := os.Stat(vars); err != nil || fi.Size() != uefiFlashSize {
		f, err := os.Create(vars)
		if err != nil {
			return "", "", false
		}
		if err := f.Truncate(uefiFlashSize); err != nil {
			f.Close()
			return "", "", false
		}
		f.Close()
	}
	return code, vars, true
}

// padFileTo copies src to dst and zero-pads it to size bytes.
func padFileTo(src, dst string, size int64) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	if int64(len(data)) > size {
		return fmt.Errorf("firmware %s (%d bytes) larger than flash size %d", src, len(data), size)
	}
	tmp := dst + ".part"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	if err := os.Truncate(tmp, size); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, dst)
}

func (d *KVMDriverEnhanced) saveVMConfig(vmInfo *KVMVMInfo) error {
	data, err := json.MarshalIndent(vmInfo.Config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(vmInfo.ConfigPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

func (d *KVMDriverEnhanced) monitorVM(vmID string, cmd *exec.Cmd) {
	err := cmd.Wait()

	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	if vmInfo, exists := d.vms[vmID]; exists {
		now := time.Now()
		vmInfo.StoppedTime = &now

		if err != nil {
			log.Printf("KVM VM %s exited with error: %v", vmID, err)
			vmInfo.State = StateFailed
		} else {
			log.Printf("KVM VM %s exited normally", vmID)
			vmInfo.State = StateStopped
		}

		vmInfo.Process = nil
		vmInfo.PID = 0
	}
}

func (d *KVMDriverEnhanced) stopVMInternal(vmInfo *KVMVMInfo) error {
	pid := vmInfo.PID
	if pid <= 0 && vmInfo.Process != nil {
		pid = vmInfo.Process.Pid
	}
	if pid <= 0 {
		markStopped(vmInfo)
		return nil
	}

	// Confirm termination by polling /proc/<pid>, NOT Process.Wait(): a VM
	// re-adopted after a restart (Process set via os.FindProcess) is not our
	// child, so Wait() returns ECHILD immediately and falsely reports "exited"
	// without confirming death or escalating -- leaving a SIGTERM-ignoring qemu
	// alive while Delete forgets it (orphan). SIGTERM, then escalate to SIGKILL.
	// ponytail: kill by PID without a cmdline re-check -- same PID-reuse window
	// as the prior Process.Signal path; add containsQEMUAndVMID if it matters.
	_ = syscall.Kill(pid, syscall.SIGTERM)
	if !awaitProcessGone(pid, stopGracePeriod) {
		_ = syscall.Kill(pid, syscall.SIGKILL)
		if !awaitProcessGone(pid, 5*time.Second) {
			return fmt.Errorf("VM process %d still alive after SIGKILL", pid)
		}
	}

	markStopped(vmInfo)
	return nil
}

// stopGracePeriod is how long to wait for a graceful SIGTERM exit before
// escalating to SIGKILL. A package var so tests can shorten it.
var stopGracePeriod = 15 * time.Second

// markStopped resets a VM's runtime fields once its process is confirmed gone.
func markStopped(vmInfo *KVMVMInfo) {
	now := time.Now()
	vmInfo.State = StateStopped
	vmInfo.StoppedTime = &now
	vmInfo.Process = nil
	vmInfo.PID = 0
}

// processAlive reports whether a PID is live via /proc -- works for any
// process, unlike Process.Wait() which only reaps our own children.
func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	_, err := os.Stat(fmt.Sprintf("/proc/%d", pid))
	return err == nil
}

// awaitProcessGone polls until the PID is gone or the timeout elapses.
func awaitProcessGone(pid int, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		if !processAlive(pid) {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// SupportsLiveMigration returns whether the driver supports live migration.
// Implemented via QEMU's own migrate command over QMP (see driver_kvm_migrate.go).
func (d *KVMDriverEnhanced) SupportsLiveMigration() bool {
	return true
}

// SupportsHotPlug returns whether the driver supports hot-plugging devices.
// Implemented for disk + network via QMP blockdev-add/netdev_add + device_add,
// and for vCPU (device_add into a query-hotpluggable-cpus slot) + memory
// (object-add memory-backend-ram + device_add pc-dimm) when the VM was launched
// with the opt-in "hotplug.maxvcpus" / "hotplug.maxmem_mb" headroom tags. Actual
// vCPU/DIMM hotplug also depends on the guest machine model (e.g. aarch64 "virt"
// on QEMU 8.2 does not support vCPU hot-plug).
func (d *KVMDriverEnhanced) SupportsHotPlug() bool {
	return true
}

// SupportsGPUPassthrough returns whether the driver supports GPU passthrough
func (d *KVMDriverEnhanced) SupportsGPUPassthrough() bool {
	return false // Not implemented yet
}

// SupportsSRIOV returns whether the driver supports SR-IOV
func (d *KVMDriverEnhanced) SupportsSRIOV() bool {
	return false // Not implemented yet
}

// SupportsNUMA returns whether the driver supports NUMA configuration. True: a
// NUMA topology set via ConfigureNUMA before Start is emitted at launch (-numa),
// verified booting a 2-node guest on aarch64 "virt". Live re-topology is not
// possible (QEMU fixes NUMA at machine init) and ConfigureNUMA says so.
func (d *KVMDriverEnhanced) SupportsNUMA() bool {
	return true
}

// GetCapabilities returns the capabilities of the KVM driver
func (d *KVMDriverEnhanced) GetCapabilities(ctx context.Context) (*HypervisorCapabilities, error) {
	return &HypervisorCapabilities{
		Type:                   VMTypeKVM,
		Version:                "QEMU/KVM",
		SupportsPause:          d.SupportsPause(),
		SupportsResume:         d.SupportsResume(),
		SupportsSnapshot:       d.SupportsSnapshot(),
		SupportsMigrate:        d.SupportsMigrate(),
		SupportsLiveMigration:  d.SupportsLiveMigration(),
		SupportsHotPlug:        d.SupportsHotPlug(),
		SupportsGPUPassthrough: d.SupportsGPUPassthrough(),
		SupportsSRIOV:          d.SupportsSRIOV(),
		SupportsNUMA:           d.SupportsNUMA(),
		MaxVCPUs:               256,
		MaxMemoryMB:            1024 * 1024, // 1TB
		SupportedFeatures:      []string{"kvm", "qemu", "x86_64"},
		HardwareExtensions:     []string{"vmx", "svm"},
	}, nil
}

// GetHypervisorInfo returns information about the KVM hypervisor
func (d *KVMDriverEnhanced) GetHypervisorInfo(ctx context.Context) (*HypervisorInfo, error) {
	capabilities, err := d.GetCapabilities(ctx)
	if err != nil {
		return nil, err
	}

	return &HypervisorInfo{
		Type:           VMTypeKVM,
		Version:        "QEMU/KVM",
		ConnectionURI:  "qemu:///system",
		Hostname:       "localhost",
		CPUModel:       "host",
		CPUCores:       8,    // Default
		MemoryMB:       8192, // Default 8GB
		Virtualization: "KVM",
		IOMMUEnabled:   false,
		NUMANodes:      1,
		GPUDevices:     []GPUDevice{},
		NetworkDevices: []NetworkDevice{},
		StorageDevices: []StorageDevice{},
		ActiveVMs:      len(d.vms),
		Capabilities:   capabilities,
		Metadata: map[string]interface{}{
			"qemu_path": d.qemuBinaryPath,
			"base_path": d.vmBasePath,
		},
	}, nil
}

// hotplug PCIe-root-port provisioning: buildQEMUArgs pre-adds this many
// pcie-root-ports (ids "hprp0".."hprpN-1") on the aarch64 "virt" machine so
// PCIe devices can be hot-plugged onto them (the root bus pcie.0 itself refuses
// hotplug). HotPlugDevice attaches onto the first free one.
const (
	hotplugRootPortPrefix = "hprp"
	hotplugRootPortCount  = 4
)

// HotPlugDevice hot-plugs a disk or network device into a running VM over QMP.
//   - disk:    blockdev-add a qcow2/raw node (path/file from device.Parameters)
//     then device_add virtio-blk-pci; the block node is rolled back if
//     device_add fails.
//   - network: netdev_add (type "user" unless device.Parameters["type"] says
//     otherwise) then device_add virtio-net-pci.
//
// On the "virt" machine the device lands on a free pcie-root-port; on x86 "pc"
// it lands on the natively-hotpluggable pci.0. The VM must be running.
func (d *KVMDriverEnhanced) HotPlugDevice(ctx context.Context, vmID string, device *DeviceConfig) error {
	if device == nil {
		return fmt.Errorf("nil device config")
	}
	if device.Name == "" {
		return fmt.Errorf("device name required (used as the qdev id for hot-unplug)")
	}

	// Look up + state-check under the lock, then release it before QMP I/O
	// (mirrors ConfigureCPUPinning / migrateWithStats).
	d.vmLock.RLock()
	vmInfo, ok := d.vms[vmID]
	d.vmLock.RUnlock()
	if !ok {
		return fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.State != StateRunning {
		return fmt.Errorf("VM %s is not running (state %s); cannot hot-plug", vmID, vmInfo.State)
	}

	sock := filepath.Join(d.runtimeDir(vmInfo), "qmp.sock")
	q, err := qmpDial(sock, 10*time.Second)
	if err != nil {
		return fmt.Errorf("connect QMP %s: %w", sock, err)
	}
	defer q.Close()

	switch device.Type {
	case "disk":
		return hotPlugDisk(q, device)
	case "network":
		return hotPlugNetwork(q, device)
	case "cpu":
		return hotPlugCPU(q, device)
	case "memory":
		return hotPlugMemory(q, device)
	default:
		return fmt.Errorf("unsupported hot-plug device type %q", device.Type)
	}
}

// hotPlugDisk blockdev-adds the backing file then attaches a virtio-blk-pci
// device; the orphan block node is deleted if the device attach fails.
func hotPlugDisk(q *qmpConn, device *DeviceConfig) error {
	path := paramString(device.Parameters, "path", "file")
	if path == "" {
		return fmt.Errorf("disk hot-plug requires a \"path\" or \"file\" parameter")
	}
	format := paramString(device.Parameters, "format")
	if format == "" {
		format = "qcow2" // ponytail: qcow2 default; pass "format":"raw" for raw images
	}
	node := "blk-" + device.Name
	if _, err := q.execute("blockdev-add", map[string]interface{}{
		"node-name": node,
		"driver":    format,
		"file":      map[string]interface{}{"driver": "file", "filename": path},
	}); err != nil {
		return fmt.Errorf("blockdev-add %s: %w", path, err)
	}
	dev := map[string]interface{}{"driver": "virtio-blk-pci", "drive": node, "id": device.Name}
	if bus := resolveHotplugBus(q, device); bus != "" {
		dev["bus"] = bus
	}
	if _, err := q.execute("device_add", dev); err != nil {
		// Roll back the now-orphaned block node so its name is reusable on retry.
		_, _ = q.execute("blockdev-del", map[string]interface{}{"node-name": node})
		return fmt.Errorf("device_add virtio-blk-pci id=%s: %w", device.Name, err)
	}
	return nil
}

// hotPlugNetwork adds a netdev backend then attaches a virtio-net-pci device;
// the netdev is removed if the device attach fails.
func hotPlugNetwork(q *qmpConn, device *DeviceConfig) error {
	netType := paramString(device.Parameters, "type")
	if netType == "" {
		netType = "user" // ponytail: SLIRP user-net default; "tap"/etc. via params
	}
	id := "net-" + device.Name
	if _, err := q.execute("netdev_add", map[string]interface{}{"type": netType, "id": id}); err != nil {
		return fmt.Errorf("netdev_add %s: %w", netType, err)
	}
	dev := map[string]interface{}{"driver": "virtio-net-pci", "netdev": id, "id": device.Name}
	if bus := resolveHotplugBus(q, device); bus != "" {
		dev["bus"] = bus
	}
	if _, err := q.execute("device_add", dev); err != nil {
		_, _ = q.execute("netdev_del", map[string]interface{}{"id": id})
		return fmt.Errorf("device_add virtio-net-pci id=%s: %w", device.Name, err)
	}
	return nil
}

// hotPlugCPU hot-plugs a vCPU into the first free query-hotpluggable-cpus slot.
// It reads the machine's pluggable-CPU layout, picks a slot with no "qom-path"
// (unplugged), and device_add's device.Name onto it using the slot's own type
// and topology props (socket/core/thread/cluster ids), which are arch-specific.
// Requires launch-time headroom (-smp ...,maxcpus=N, opt-in via the
// "hotplug.maxvcpus" tag) so a free slot exists. Machines that cannot hot-plug
// CPUs (e.g. aarch64 "virt" on QEMU 8.2) make query-hotpluggable-cpus itself
// error, which is surfaced verbatim.
func hotPlugCPU(q *qmpConn, device *DeviceConfig) error {
	raw, err := q.execute("query-hotpluggable-cpus", nil)
	if err != nil {
		return fmt.Errorf("query-hotpluggable-cpus (this machine may not support vCPU hot-plug): %w", err)
	}
	var slots []struct {
		Type    string                 `json:"type"`
		QOMPath *string                `json:"qom-path"` // present == already plugged
		Props   map[string]interface{} `json:"props"`
	}
	if err := json.Unmarshal(raw, &slots); err != nil {
		return fmt.Errorf("parse query-hotpluggable-cpus: %w", err)
	}
	for _, s := range slots {
		if s.QOMPath != nil {
			continue // slot occupied
		}
		add := map[string]interface{}{"driver": s.Type, "id": device.Name}
		for k, v := range s.Props { // socket-id/core-id/thread-id[/cluster-id]
			add[k] = v
		}
		if _, err := q.execute("device_add", add); err != nil {
			return fmt.Errorf("device_add cpu %s id=%s: %w", s.Type, device.Name, err)
		}
		return nil
	}
	return fmt.Errorf("no free vCPU slot (all maxcpus in use); raise the \"hotplug.maxvcpus\" headroom")
}

// hotPlugMemory hot-plugs a pc-dimm backed by an anonymous memory-backend-ram of
// "size_mb" MiB. Requires launch-time headroom (-m ...,slots=S,maxmem=M, opt-in
// via the "hotplug.maxmem_mb" tag). The backend object is rolled back if the
// device attach fails. NOTE: on aarch64 "virt" this needs UEFI firmware loaded
// (the driver's ensureUEFI) so the machine has an acpi-ged for memory hotplug.
func hotPlugMemory(q *qmpConn, device *DeviceConfig) error {
	sizeMB := paramInt(device.Parameters, "size_mb", "size")
	if sizeMB <= 0 {
		return fmt.Errorf("memory hot-plug requires a positive \"size_mb\" parameter")
	}
	backend := "mem-" + device.Name
	if _, err := q.execute("object-add", map[string]interface{}{
		"qom-type": "memory-backend-ram",
		"id":       backend,
		"size":     int64(sizeMB) * 1024 * 1024,
	}); err != nil {
		return fmt.Errorf("object-add memory-backend-ram (%dMiB): %w", sizeMB, err)
	}
	if _, err := q.execute("device_add", map[string]interface{}{
		"driver": "pc-dimm",
		"id":     device.Name,
		"memdev": backend,
	}); err != nil {
		// Roll back the now-orphaned backend so its id is reusable on retry.
		_, _ = q.execute("object-del", map[string]interface{}{"id": backend})
		return fmt.Errorf("device_add pc-dimm id=%s: %w", device.Name, err)
	}
	return nil
}

// paramInt returns the first parameter present under keys as an int, coercing the
// JSON number (float64) / int / string forms; 0 if absent or unparseable.
func paramInt(params map[string]interface{}, keys ...string) int {
	for _, k := range keys {
		v, ok := params[k]
		if !ok {
			continue
		}
		switch n := v.(type) {
		case float64:
			return int(n)
		case int:
			return n
		case int64:
			return int(n)
		case string:
			if i, err := strconv.Atoi(strings.TrimSpace(n)); err == nil {
				return i
			}
		}
	}
	return 0
}

// HotUnplugDevice removes a hot-plugged device by its qdev id over QMP. It
// issues device_del, waits briefly for the guest to release the PCIe device
// (device_del is asynchronous), then best-effort deletes the backing block node
// / netdev HotPlugDevice created (named "blk-<id>" / "net-<id>"). The VM must be
// running.
func (d *KVMDriverEnhanced) HotUnplugDevice(ctx context.Context, vmID string, deviceID string) error {
	if deviceID == "" {
		return fmt.Errorf("device id required")
	}
	d.vmLock.RLock()
	vmInfo, ok := d.vms[vmID]
	d.vmLock.RUnlock()
	if !ok {
		return fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.State != StateRunning {
		return fmt.Errorf("VM %s is not running (state %s); cannot hot-unplug", vmID, vmInfo.State)
	}

	sock := filepath.Join(d.runtimeDir(vmInfo), "qmp.sock")
	q, err := qmpDial(sock, 10*time.Second)
	if err != nil {
		return fmt.Errorf("connect QMP %s: %w", sock, err)
	}
	defer q.Close()

	if _, err := q.execute("device_del", map[string]interface{}{"id": deviceID}); err != nil {
		return fmt.Errorf("device_del %s: %w", deviceID, err)
	}
	// device_del is guest-driven and asynchronous; wait for the qdev id to leave
	// query-pci so the backend can be freed without an "in use" error.
	awaitDeviceGone(q, deviceID, 10*time.Second)
	// Best-effort backend cleanup: only one of these matches (disk vs nic); the
	// other simply errors because that name never existed. A device that never
	// released leaves its node to be reaped when the VM stops.
	_, _ = q.execute("blockdev-del", map[string]interface{}{"node-name": "blk-" + deviceID})
	_, _ = q.execute("netdev_del", map[string]interface{}{"id": "net-" + deviceID})
	return nil
}

// resolveHotplugBus returns the QMP bus= for a hot-plugged PCIe device. An
// explicit device.Bus wins; otherwise the first free pcie-root-port is used. It
// returns "" when no root port exists (x86 "pc" hot-plugs onto pci.0 natively)
// or none is free (device_add then fails with a clear "no hotplug bus" error).
func resolveHotplugBus(q *qmpConn, device *DeviceConfig) string {
	if device.Bus != "" {
		return device.Bus
	}
	for _, dv := range queryPCITopo(q) {
		if strings.HasPrefix(dv.qdevID, hotplugRootPortPrefix) && len(dv.children) == 0 {
			return dv.qdevID
		}
	}
	return ""
}

// awaitDeviceGone polls query-pci until no device carries qdevID, or timeout.
// A query error is treated as "gone" (stop waiting) -- the subsequent best-effort
// cleanup is harmless either way.
func awaitDeviceGone(q *qmpConn, qdevID string, timeout time.Duration) {
	deadline := time.Now().Add(timeout)
	for {
		present := false
		for _, dv := range queryPCITopo(q) {
			if dv.qdevID == qdevID {
				present = true
				break
			}
			for _, c := range dv.children {
				if c == qdevID {
					present = true
					break
				}
			}
		}
		if !present || time.Now().After(deadline) {
			return
		}
		time.Sleep(200 * time.Millisecond)
	}
}

// pciTopoDev is one PCI device from query-pci plus, for a pci_bridge (root port),
// the qdev ids of the devices behind it.
type pciTopoDev struct {
	qdevID   string
	children []string
}

// queryPCITopo flattens query-pci into the root-bus devices and, for each
// pci_bridge, its downstream qdev ids. Returns nil on any error (callers treat
// nil as "nothing found").
func queryPCITopo(q *qmpConn) []pciTopoDev {
	raw, err := q.execute("query-pci", nil)
	if err != nil {
		return nil
	}
	var buses []struct {
		Devices []struct {
			QdevID    string `json:"qdev_id"`
			PCIBridge *struct {
				Devices []struct {
					QdevID string `json:"qdev_id"`
				} `json:"devices"`
			} `json:"pci_bridge"`
		} `json:"devices"`
	}
	if err := json.Unmarshal(raw, &buses); err != nil {
		return nil
	}
	var out []pciTopoDev
	for _, b := range buses {
		for _, dv := range b.Devices {
			pd := pciTopoDev{qdevID: dv.QdevID}
			if dv.PCIBridge != nil {
				for _, k := range dv.PCIBridge.Devices {
					pd.children = append(pd.children, k.QdevID)
				}
			}
			out = append(out, pd)
		}
	}
	return out
}

// paramString returns the first parameter present under keys as a non-empty
// string, else "".
func paramString(params map[string]interface{}, keys ...string) string {
	for _, k := range keys {
		if v, ok := params[k]; ok {
			if s, ok := v.(string); ok && s != "" {
				return s
			}
		}
	}
	return ""
}

// ConfigureCPUPinning pins a running VM's host threads to physical CPUs via
// sched_setaffinity(2). It maps each guest vCPU to its host thread-id over QMP
// (query-cpus-fast), then applies the requested cpuset to that thread; it can
// also pin the emulator (main qemu process) and any iothreads. The VM must be
// running (a paused/created VM has no vCPU threads to pin).
func (d *KVMDriverEnhanced) ConfigureCPUPinning(ctx context.Context, vmID string, pinning *CPUPinningConfig) error {
	if pinning == nil {
		return fmt.Errorf("nil CPU pinning config")
	}

	// Look up + state-check under the lock, then release it: QMP and affinity
	// syscalls must not run while holding vmLock (mirrors migrateWithStats).
	d.vmLock.RLock()
	vmInfo, ok := d.vms[vmID]
	d.vmLock.RUnlock()
	if !ok {
		return fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.State != StateRunning {
		return fmt.Errorf("VM %s is not running (state %s); cannot pin vCPUs", vmID, vmInfo.State)
	}

	sock := filepath.Join(d.runtimeDir(vmInfo), "qmp.sock")
	q, err := qmpDial(sock, 10*time.Second)
	if err != nil {
		return fmt.Errorf("connect QMP %s: %w", sock, err)
	}
	defer q.Close()

	// Map guest vCPU index -> host thread-id (the kernel TID of that vCPU thread).
	raw, err := q.execute("query-cpus-fast", nil)
	if err != nil {
		return fmt.Errorf("query-cpus-fast: %w", err)
	}
	var cpus []struct {
		CPUIndex int `json:"cpu-index"`
		ThreadID int `json:"thread-id"`
	}
	if err := json.Unmarshal(raw, &cpus); err != nil {
		return fmt.Errorf("parse query-cpus-fast: %w", err)
	}
	vcpuThread := make(map[int]int, len(cpus))
	for _, c := range cpus {
		vcpuThread[c.CPUIndex] = c.ThreadID
	}

	// Pin each requested vCPU to its cpuset.
	for _, vp := range pinning.VCPUs {
		tid, ok := vcpuThread[vp.VCPU]
		if !ok {
			return fmt.Errorf("vcpu %d not present (guest has %d vcpus)", vp.VCPU, len(cpus))
		}
		if err := pinThreadToCPUSet(tid, vp.CPUSet); err != nil {
			return fmt.Errorf("pin vcpu %d (thread %d) to %q: %w", vp.VCPU, tid, vp.CPUSet, err)
		}
	}

	// Pin the emulator (the main qemu process) if requested.
	if pinning.EmulatorPin != "" {
		if vmInfo.PID <= 0 {
			return fmt.Errorf("emulator pin requested but VM %s has no PID", vmID)
		}
		if err := pinThreadToCPUSet(vmInfo.PID, pinning.EmulatorPin); err != nil {
			return fmt.Errorf("pin emulator (pid %d) to %q: %w", vmInfo.PID, pinning.EmulatorPin, err)
		}
	}

	// IOThreads: pinned by matching QEMU's "iothread<N>" id from query-iothreads.
	// buildQEMUArgs emits N iothread objects when a VM opts in via
	// Config.Tags["iothreads"]=N, so query-iothreads returns them and these pins
	// take effect. A VM launched WITHOUT that tag has no iothreads, so requested
	// pins are skipped (logged, not errored) rather than failing the call.
	if len(pinning.IOThreads) > 0 {
		if err := pinIOThreads(q, vmID, pinning.IOThreads); err != nil {
			return err
		}
	}

	return nil
}

// pinThreadToCPUSet sets the CPU affinity of one host thread (or process) named
// by tid to the cpus in a Linux cpulist ("0-3,8,9"). tid is a kernel thread id
// (QEMU's QMP thread-id) or a pid; Linux sched_setaffinity treats both alike.
func pinThreadToCPUSet(tid int, cpuset string) error {
	cpus := parseCPUList(cpuset) // reused from hardware_virtualization.go
	if len(cpus) == 0 {
		return fmt.Errorf("empty or unparseable cpuset %q", cpuset)
	}
	var set unix.CPUSet
	set.Zero()
	for _, c := range cpus {
		set.Set(c) // ponytail: CPUSet.Set silently ignores c>=1024; a bad cpuset then yields EINVAL below
	}
	if set.Count() == 0 {
		return fmt.Errorf("cpuset %q selected no representable cpus", cpuset)
	}
	if err := unix.SchedSetaffinity(tid, &set); err != nil {
		return fmt.Errorf("sched_setaffinity(%d): %w", tid, err)
	}
	return nil
}

// pinIOThreads pins requested iothreads to their cpusets, matched by QEMU's
// "iothread<N>" id convention against query-iothreads. iothreads absent from the
// running VM are skipped (logged), not treated as errors.
func pinIOThreads(q *qmpConn, vmID string, reqs []IOThreadPinning) error {
	raw, err := q.execute("query-iothreads", nil)
	if err != nil {
		return fmt.Errorf("query-iothreads: %w", err)
	}
	var iothreads []struct {
		ID       string `json:"id"`
		ThreadID int    `json:"thread-id"`
	}
	if err := json.Unmarshal(raw, &iothreads); err != nil {
		return fmt.Errorf("parse query-iothreads: %w", err)
	}
	byID := make(map[string]int, len(iothreads))
	for _, it := range iothreads {
		byID[it.ID] = it.ThreadID
	}
	for _, req := range reqs {
		id := fmt.Sprintf("iothread%d", req.IOThread)
		tid, ok := byID[id]
		if !ok {
			log.Printf("VM %s: iothread %s not present, skipping pin to %q", vmID, id, req.CPUSet)
			continue
		}
		if err := pinThreadToCPUSet(tid, req.CPUSet); err != nil {
			return fmt.Errorf("pin %s (thread %d) to %q: %w", id, tid, req.CPUSet, err)
		}
	}
	return nil
}

// ConfigureNUMA sets a VM's NUMA topology. QEMU fixes NUMA at machine init and
// CANNOT re-topologize a running guest, so this stores the topology for the next
// boot (buildQEMUArgs emits -object memory-backend-ram + -numa node per node) and
// refuses a running VM with a clear error rather than pretending live reconfig
// works. Configure it before Start (the topology is applied when launchVM runs).
func (d *KVMDriverEnhanced) ConfigureNUMA(ctx context.Context, vmID string, topology *NUMATopology) error {
	if topology == nil || len(topology.Nodes) == 0 {
		return fmt.Errorf("nil or empty NUMA topology")
	}
	d.vmLock.Lock()
	defer d.vmLock.Unlock()

	vmInfo, ok := d.vms[vmID]
	if !ok {
		return fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.State == StateRunning {
		return fmt.Errorf("VM %s is running; NUMA topology is fixed at QEMU machine init and cannot be changed on a live VM — stop it and call ConfigureNUMA before Start (buildQEMUArgs applies it at launch)", vmID)
	}

	// Persist the topology into Config.Tags (JSON) and out to config.json so it
	// survives a driver restart between ConfigureNUMA and Start — adopt/reload
	// rehydrates Config from config.json but not the in-memory NUMA field, and
	// buildQEMUArgs reads the tag (effectiveNUMA) when that field is nil. The
	// in-memory field is kept too as the same-process fast path.
	encoded, err := json.Marshal(topology)
	if err != nil {
		return fmt.Errorf("encode NUMA topology: %w", err)
	}
	if vmInfo.Config.Tags == nil {
		vmInfo.Config.Tags = map[string]string{}
	}
	vmInfo.Config.Tags[numaTopologyTag] = string(encoded)
	vmInfo.NUMA = topology
	if err := d.saveVMConfig(vmInfo); err != nil {
		return fmt.Errorf("persist NUMA topology to config.json: %w", err)
	}
	log.Printf("Stored %d-node NUMA topology for VM %s (persisted to config.json; applied at next launch)", len(topology.Nodes), vmID)
	return nil
}

// GetProcessPID returns the hypervisor process PID for a given VM.
// For KVM/QEMU VMs, this reads from the PID file or falls back to process scanning.
func (d *KVMDriverEnhanced) GetProcessPID(vmID string) int {
	d.vmLock.RLock()
	defer d.vmLock.RUnlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return 0
	}

	// If we have a cached PID and the VM is running, return it
	if vmInfo.PID > 0 && vmInfo.State == StateRunning {
		return vmInfo.PID
	}

	// Try to read from PID file
	pidFilePath := filepath.Join(filepath.Dir(vmInfo.DiskPath), "qemu.pid")
	pidData, err := os.ReadFile(pidFilePath)
	if err == nil {
		pid, err := strconv.Atoi(string(pidData[:len(pidData)-1])) // Remove newline
		if err == nil && pid > 0 {
			// Verify process exists
			if _, err := os.Stat(fmt.Sprintf("/proc/%d", pid)); err == nil {
				return pid
			}
		}
	}

	// Fallback: try /var/run/libvirt/qemu pattern (for libvirt-managed VMs)
	libvirtPidPath := fmt.Sprintf("/var/run/libvirt/qemu/%s.pid", vmID)
	pidData, err = os.ReadFile(libvirtPidPath)
	if err == nil {
		pid, err := strconv.Atoi(string(pidData[:len(pidData)-1]))
		if err == nil && pid > 0 {
			if _, err := os.Stat(fmt.Sprintf("/proc/%d", pid)); err == nil {
				return pid
			}
		}
	}

	// Final fallback: scan /proc for qemu process with matching VM ID
	return d.findQEMUProcessByVMID(vmID)
}

// findQEMUProcessByVMID scans /proc to find a QEMU process associated with the given VM ID
func (d *KVMDriverEnhanced) findQEMUProcessByVMID(vmID string) int {
	// Read /proc directory
	procDir, err := os.Open("/proc")
	if err != nil {
		return 0
	}
	defer procDir.Close()

	entries, err := procDir.Readdirnames(-1)
	if err != nil {
		return 0
	}

	for _, entry := range entries {
		// Check if entry is a PID (numeric)
		pid, err := strconv.Atoi(entry)
		if err != nil {
			continue
		}

		// Read cmdline for this process
		cmdlinePath := fmt.Sprintf("/proc/%d/cmdline", pid)
		cmdlineData, err := os.ReadFile(cmdlinePath)
		if err != nil {
			continue
		}

		cmdline := string(cmdlineData)

		// Check if it's a QEMU process with our VM ID
		if containsQEMUAndVMID(cmdline, vmID) {
			return pid
		}
	}

	return 0
}

// containsQEMUAndVMID checks if a cmdline contains both qemu and the VM ID
func containsQEMUAndVMID(cmdline, vmID string) bool {
	hasQEMU := false
	hasVMID := false

	// cmdline has null-separated arguments
	for i := 0; i < len(cmdline); i++ {
		// Find next null or end of string
		j := i
		for j < len(cmdline) && cmdline[j] != 0 {
			j++
		}

		arg := cmdline[i:j]

		// Check for qemu in the argument
		if !hasQEMU {
			for k := 0; k <= len(arg)-4; k++ {
				if arg[k:k+4] == "qemu" {
					hasQEMU = true
					break
				}
			}
		}

		// Check for VM ID in the argument
		if !hasVMID && len(arg) >= len(vmID) {
			for k := 0; k <= len(arg)-len(vmID); k++ {
				if arg[k:k+len(vmID)] == vmID {
					hasVMID = true
					break
				}
			}
		}

		if hasQEMU && hasVMID {
			return true
		}

		i = j
	}

	return false
}

// GetNamespacePath returns the PID namespace path for a given VM's QEMU process.
// This is used for guest namespace eBPF injection.
func (d *KVMDriverEnhanced) GetNamespacePath(vmID string) string {
	pid := d.GetProcessPID(vmID)
	if pid <= 0 {
		return ""
	}
	return fmt.Sprintf("/proc/%d/ns/pid", pid)
}
