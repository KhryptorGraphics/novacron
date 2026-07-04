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
// external VNC on :0). ponytail: tiny probe->bind TOCTOU race; acceptable for a
// display port, revisit only if concurrent Starts contend.
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
		log.Printf("Adopted running KVM VM %s (PID %d) on driver restart", vmID, pid)
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

	// Build QEMU command
	args := d.buildQEMUArgs(vmInfo)

	// Start QEMU process
	cmd := exec.Command(d.qemuBinaryPath, args...)
	cmd.Dir = filepath.Dir(vmInfo.DiskPath)

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start QEMU: %w", err)
	}

	vmInfo.Process = cmd.Process
	vmInfo.PID = cmd.Process.Pid
	vmInfo.State = StateRunning
	vmInfo.StartTime = time.Now()

	// Monitor the process
	go d.monitorVM(vmID, cmd)

	log.Printf("Started KVM VM %s with PID %d", vmID, vmInfo.PID)
	return nil
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

	// Send SIGTERM to QEMU process
	if err := vmInfo.Process.Signal(os.Interrupt); err != nil {
		log.Printf("Failed to send SIGTERM to VM %s: %v, trying SIGKILL", vmID, err)
		if err := vmInfo.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill VM process: %w", err)
		}
	}

	// Wait for process to exit with timeout
	done := make(chan error, 1)
	go func() {
		_, err := vmInfo.Process.Wait()
		done <- err
	}()

	select {
	case err := <-done:
		if err != nil {
			log.Printf("VM %s process exited with error: %v", vmID, err)
		}
	case <-time.After(30 * time.Second):
		log.Printf("VM %s did not exit within timeout, force killing", vmID)
		vmInfo.Process.Kill()
		<-done
	}

	now := time.Now()
	vmInfo.State = StateStopped
	vmInfo.StoppedTime = &now
	vmInfo.Process = nil
	vmInfo.PID = 0

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
func (d *KVMDriverEnhanced) SupportsMigrate() bool  { return false }

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

// Migrate is not implemented for this basic driver
func (d *KVMDriverEnhanced) Migrate(ctx context.Context, vmID, target string, params map[string]string) error {
	return fmt.Errorf("migration not supported by this driver")
}

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

	// ponytail: no -daemonize — keeps qemu as a tracked child so cmd.Wait()
	// in monitorVM and driver.Stop reflect the real process; -pidfile still
	// lets a restarted server rediscover the running qemu.
	args := []string{
		"-machine", machine + ",accel=" + accel,
		"-cpu", cpu,
		"-m", strconv.Itoa(mem),
		"-smp", strconv.Itoa(cpus),
		"-drive", fmt.Sprintf("file=%s,format=qcow2,if=virtio", vmInfo.DiskPath),
		"-netdev", "user,id=net0",
		"-device", "virtio-net-pci,netdev=net0",
		"-vnc", fmt.Sprintf(":%d", vmInfo.VNCPort-5900),
		"-monitor", fmt.Sprintf("unix:%s,server,nowait", vmInfo.MonitorPath),
		// Capture the guest serial console so boot is observable.
		"-serial", "file:" + filepath.Join(filepath.Dir(vmInfo.DiskPath), "console.log"),
		"-pidfile", filepath.Join(filepath.Dir(vmInfo.DiskPath), "qemu.pid"),
	}

	// aarch64 "virt" needs UEFI firmware (pflash) to boot a disk image.
	if machine == "virt" {
		if code, vars, ok := d.ensureUEFI(filepath.Dir(vmInfo.DiskPath)); ok {
			args = append(args,
				"-drive", "if=pflash,format=raw,unit=0,file="+code+",readonly=on",
				"-drive", "if=pflash,format=raw,unit=1,file="+vars,
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

	return args
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
	if vmInfo.Process == nil {
		return nil
	}

	// Send SIGTERM
	if err := vmInfo.Process.Signal(os.Interrupt); err != nil {
		// Try SIGKILL if SIGTERM fails
		if err := vmInfo.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill VM process: %w", err)
		}
	}

	// Wait with timeout
	done := make(chan error, 1)
	go func() {
		_, err := vmInfo.Process.Wait()
		done <- err
	}()

	select {
	case <-done:
		// Process exited
	case <-time.After(30 * time.Second):
		// Force kill if timeout
		vmInfo.Process.Kill()
		<-done
	}

	now := time.Now()
	vmInfo.State = StateStopped
	vmInfo.StoppedTime = &now
	vmInfo.Process = nil
	vmInfo.PID = 0

	return nil
}

// SupportsLiveMigration returns whether the driver supports live migration
func (d *KVMDriverEnhanced) SupportsLiveMigration() bool {
	return false // Not implemented yet
}

// SupportsHotPlug returns whether the driver supports hot-plugging devices
func (d *KVMDriverEnhanced) SupportsHotPlug() bool {
	return false // Not implemented yet
}

// SupportsGPUPassthrough returns whether the driver supports GPU passthrough
func (d *KVMDriverEnhanced) SupportsGPUPassthrough() bool {
	return false // Not implemented yet
}

// SupportsSRIOV returns whether the driver supports SR-IOV
func (d *KVMDriverEnhanced) SupportsSRIOV() bool {
	return false // Not implemented yet
}

// SupportsNUMA returns whether the driver supports NUMA configuration
func (d *KVMDriverEnhanced) SupportsNUMA() bool {
	return false // Not implemented yet
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

// HotPlugDevice hot-plugs a device (not implemented yet)
func (d *KVMDriverEnhanced) HotPlugDevice(ctx context.Context, vmID string, device *DeviceConfig) error {
	return fmt.Errorf("hot-plug not implemented for KVM driver")
}

// HotUnplugDevice hot-unplugs a device (not implemented yet)
func (d *KVMDriverEnhanced) HotUnplugDevice(ctx context.Context, vmID string, deviceID string) error {
	return fmt.Errorf("hot-unplug not implemented for KVM driver")
}

// ConfigureCPUPinning configures CPU pinning (not implemented yet)
func (d *KVMDriverEnhanced) ConfigureCPUPinning(ctx context.Context, vmID string, pinning *CPUPinningConfig) error {
	return fmt.Errorf("CPU pinning not implemented for KVM driver")
}

// ConfigureNUMA configures NUMA topology (not implemented yet)
func (d *KVMDriverEnhanced) ConfigureNUMA(ctx context.Context, vmID string, topology *NUMATopology) error {
	return fmt.Errorf("NUMA configuration not implemented for KVM driver")
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
