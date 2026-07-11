package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/google/uuid"
)

// processIDUnsafe matches any character not allowed in a VM id. We validate the
// id before joining it onto the driver base path, so a caller-supplied id can
// never escape basePath via "../" or absolute paths.
var processIDUnsafe = regexp.MustCompile(`[^a-zA-Z0-9._-]`)

const defaultProcessBasePath = "/var/lib/novacron/processes"

// ProcessDriver implements VMDriver by running each "VM" as a native OS process
// via os/exec — no hypervisor or container runtime involved. It mirrors the
// shell-out shape of ContainerDriver, but manages processes directly instead of
// delegating to docker.
//
// Per-VM state lives on disk under basePath/<vmID>/ (metadata.json + vm.pid) so
// that Start (which only receives a vmID) can recover the command to run, and so
// status/list survive a manager restart.
//
// ponytail: liveness is a pid-file + kill(pid,0) check. Known ceiling — a pid
// can be reused by an unrelated process after a hard restart; acceptable for a
// process driver. Upgrade path: record a start timestamp and cross-check
// /proc/<pid>/stat starttime if reuse ever bites.
type ProcessDriver struct {
	nodeID   string
	basePath string
}

// processMetadata is the persisted description of a process VM.
type processMetadata struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	Command   string            `json:"command"`
	Args      []string          `json:"args"`
	Env       map[string]string `json:"env"`
	WorkDir   string            `json:"work_dir"`
	CPUShares int               `json:"cpu_shares"`
	MemoryMB  int               `json:"memory_mb"`
	CreatedAt time.Time         `json:"created_at"`
}

// NewProcessDriver creates a real process driver. Recognised config keys:
// "node_id" (string) and "base_path" (string, defaults to
// /var/lib/novacron/processes).
func NewProcessDriver(config map[string]interface{}) (VMDriver, error) {
	nodeID, _ := config["node_id"].(string)
	basePath, _ := config["base_path"].(string)
	if strings.TrimSpace(basePath) == "" {
		basePath = defaultProcessBasePath
	}
	if err := makeDirectoryIfNotExists(basePath); err != nil {
		return nil, fmt.Errorf("failed to create process base path %s: %w", basePath, err)
	}
	return &ProcessDriver{nodeID: nodeID, basePath: basePath}, nil
}

func (d *ProcessDriver) vmDir(vmID string) string   { return filepath.Join(d.basePath, vmID) }
func (d *ProcessDriver) pidFile(vmID string) string { return filepath.Join(d.vmDir(vmID), "vm.pid") }
func (d *ProcessDriver) metaFile(vmID string) string {
	return filepath.Join(d.vmDir(vmID), "metadata.json")
}

func validProcessID(vmID string) bool {
	return vmID != "" && !processIDUnsafe.MatchString(vmID)
}

// Create records a process VM on disk. Like `docker create`, it does not start
// anything — the VM lands in the stopped state and Start spawns it.
func (d *ProcessDriver) Create(ctx context.Context, config VMConfig) (string, error) {
	if strings.TrimSpace(config.Command) == "" {
		return "", fmt.Errorf("invalid config: process VM requires a command")
	}

	vmID := config.ID
	if vmID == "" {
		vmID = uuid.New().String()
	}
	if !validProcessID(vmID) {
		return "", fmt.Errorf("invalid vm id %q: must match [a-zA-Z0-9._-]", vmID)
	}

	dir := d.vmDir(vmID)
	if err := makeDirectoryIfNotExists(dir); err != nil {
		return "", fmt.Errorf("failed to create vm dir for %s: %w", vmID, err)
	}

	meta := processMetadata{
		ID:        vmID,
		Name:      config.Name,
		Command:   config.Command,
		Args:      config.Args,
		Env:       config.Env,
		WorkDir:   config.WorkDir,
		CPUShares: config.CPUShares,
		MemoryMB:  config.MemoryMB,
		CreatedAt: time.Now(),
	}
	if err := d.writeMetadata(vmID, meta); err != nil {
		_ = os.RemoveAll(dir)
		return "", err
	}

	log.Printf("Created process VM %s (command: %s)", vmID, config.Command)
	return vmID, nil
}

// Start spawns the process recorded by Create. It is idempotent: starting an
// already-running VM is a no-op.
func (d *ProcessDriver) Start(ctx context.Context, vmID string) error {
	if !validProcessID(vmID) {
		return fmt.Errorf("invalid vm id %q", vmID)
	}
	if st, _ := d.GetStatus(ctx, vmID); st == StateRunning {
		return nil
	}

	meta, err := d.readMetadata(vmID)
	if err != nil {
		return err
	}

	stdout, err := os.OpenFile(filepath.Join(d.vmDir(vmID), "stdout.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open stdout log for %s: %w", vmID, err)
	}
	defer stdout.Close()
	stderr, err := os.OpenFile(filepath.Join(d.vmDir(vmID), "stderr.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open stderr log for %s: %w", vmID, err)
	}
	defer stderr.Close()

	// ponytail: exec.Command, not CommandContext — the process must outlive the
	// request ctx passed to Start. Its lifetime is bounded by Stop/Delete.
	cmd := exec.Command(meta.Command, meta.Args...)
	cmd.Dir = meta.WorkDir
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	cmd.Env = processEnv(meta.Env)
	// Own process group so Stop can signal the whole tree (kill -pgid).
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start process VM %s: %w", vmID, err)
	}
	pid := cmd.Process.Pid
	if err := os.WriteFile(d.pidFile(vmID), []byte(strconv.Itoa(pid)), 0644); err != nil {
		_ = cmd.Process.Kill()
		return fmt.Errorf("failed to write pid file for %s: %w", vmID, err)
	}
	// Reap the child when it exits so it doesn't linger as a zombie. The pid
	// file remains the source of truth for status; a stale entry is detected by
	// the liveness check in GetStatus.
	go func() { _ = cmd.Wait() }()

	log.Printf("Started process VM %s (pid %d)", vmID, pid)
	return nil
}

// Stop terminates the process (SIGTERM, then SIGKILL after a grace period) and
// removes the pid file. It is idempotent.
func (d *ProcessDriver) Stop(ctx context.Context, vmID string) error {
	pid, ok := d.readPID(vmID)
	if !ok {
		return nil // nothing recorded as running
	}

	if processAlive(pid) {
		signalProcess(pid, syscall.SIGTERM)
		if !awaitProcessGone(pid, 5*time.Second) {
			signalProcess(pid, syscall.SIGKILL)
			awaitProcessGone(pid, 2*time.Second)
		}
	}

	_ = os.Remove(d.pidFile(vmID))
	log.Printf("Stopped process VM %s (pid %d)", vmID, pid)
	return nil
}

// Delete stops the process and removes all of its on-disk state.
func (d *ProcessDriver) Delete(ctx context.Context, vmID string) error {
	if !validProcessID(vmID) {
		return fmt.Errorf("invalid vm id %q", vmID)
	}
	_ = d.Stop(ctx, vmID)
	if err := os.RemoveAll(d.vmDir(vmID)); err != nil {
		return fmt.Errorf("failed to delete process VM %s: %w", vmID, err)
	}
	log.Printf("Deleted process VM %s", vmID)
	return nil
}

// GetStatus reports the current lifecycle state of a process VM.
func (d *ProcessDriver) GetStatus(ctx context.Context, vmID string) (State, error) {
	if _, err := os.Stat(d.metaFile(vmID)); err != nil {
		if os.IsNotExist(err) {
			return StateUnknown, fmt.Errorf("process VM %s not found", vmID)
		}
		return StateUnknown, err
	}

	pid, ok := d.readPID(vmID)
	if !ok {
		return StateStopped, nil // created but not started, or already stopped
	}
	if processAlive(pid) {
		return StateRunning, nil
	}
	// Stale pid file: the process exited on its own. Clean it up.
	_ = os.Remove(d.pidFile(vmID))
	return StateStopped, nil
}

// GetInfo returns metadata + current state for a process VM.
func (d *ProcessDriver) GetInfo(ctx context.Context, vmID string) (*VMInfo, error) {
	meta, err := d.readMetadata(vmID)
	if err != nil {
		return nil, err
	}
	status, err := d.GetStatus(ctx, vmID)
	if err != nil {
		return nil, err
	}

	info := &VMInfo{
		ID:        vmID,
		Name:      meta.Name,
		State:     status,
		CPUShares: meta.CPUShares,
		MemoryMB:  meta.MemoryMB,
		CreatedAt: meta.CreatedAt,
	}
	if pid, ok := d.readPID(vmID); ok && status == StateRunning {
		info.PID = pid
	}
	return info, nil
}

// GetMetrics returns the same payload as GetInfo (process VMs have no separate
// metrics source), matching the container driver's behaviour.
func (d *ProcessDriver) GetMetrics(ctx context.Context, vmID string) (*VMInfo, error) {
	return d.GetInfo(ctx, vmID)
}

// ListVMs enumerates every process VM recorded under the base path.
func (d *ProcessDriver) ListVMs(ctx context.Context) ([]VMInfo, error) {
	entries, err := os.ReadDir(d.basePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("failed to list process VMs: %w", err)
	}

	var vms []VMInfo
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		info, err := d.GetInfo(ctx, e.Name())
		if err != nil {
			continue // skip dirs without valid metadata
		}
		vms = append(vms, *info)
	}
	return vms, nil
}

func (d *ProcessDriver) writeMetadata(vmID string, meta processMetadata) error {
	data, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata for %s: %w", vmID, err)
	}
	if err := os.WriteFile(d.metaFile(vmID), data, 0644); err != nil {
		return fmt.Errorf("failed to write metadata for %s: %w", vmID, err)
	}
	return nil
}

func (d *ProcessDriver) readMetadata(vmID string) (processMetadata, error) {
	var meta processMetadata
	data, err := os.ReadFile(d.metaFile(vmID))
	if err != nil {
		if os.IsNotExist(err) {
			return meta, fmt.Errorf("process VM %s not found", vmID)
		}
		return meta, fmt.Errorf("failed to read metadata for %s: %w", vmID, err)
	}
	if err := json.Unmarshal(data, &meta); err != nil {
		return meta, fmt.Errorf("failed to parse metadata for %s: %w", vmID, err)
	}
	return meta, nil
}

func (d *ProcessDriver) readPID(vmID string) (int, bool) {
	data, err := os.ReadFile(d.pidFile(vmID))
	if err != nil {
		return 0, false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || pid <= 0 {
		return 0, false
	}
	return pid, true
}

// processEnv builds the child environment: the parent env plus any overrides.
// Returns nil (inherit parent env) when no overrides are set.
func processEnv(env map[string]string) []string {
	if len(env) == 0 {
		return nil
	}
	out := os.Environ()
	for k, v := range env {
		out = append(out, k+"="+v)
	}
	return out
}

// signalProcess sends sig to the whole process group, falling back to the bare
// pid if the group signal fails (e.g. Setpgid didn't take).
func signalProcess(pid int, sig syscall.Signal) {
	if err := syscall.Kill(-pid, sig); err != nil {
		_ = syscall.Kill(pid, sig)
	}
}

// --- Capability reporting -------------------------------------------------

func (d *ProcessDriver) SupportsPause() bool          { return false }
func (d *ProcessDriver) SupportsResume() bool         { return false }
func (d *ProcessDriver) SupportsSnapshot() bool       { return false }
func (d *ProcessDriver) SupportsMigrate() bool        { return false }
func (d *ProcessDriver) SupportsLiveMigration() bool  { return false }
func (d *ProcessDriver) SupportsHotPlug() bool        { return false }
func (d *ProcessDriver) SupportsGPUPassthrough() bool { return false }
func (d *ProcessDriver) SupportsSRIOV() bool          { return false }
func (d *ProcessDriver) SupportsNUMA() bool           { return false }

func (d *ProcessDriver) GetCapabilities(ctx context.Context) (*HypervisorCapabilities, error) {
	return &HypervisorCapabilities{
		Type:               VMTypeProcess,
		Version:            "1.0.0",
		MaxVCPUs:           1,
		MaxMemoryMB:        0, // no enforced ceiling; bounded by the host
		SupportedFeatures:  []string{"process", "exec"},
		HardwareExtensions: []string{},
	}, nil
}

func (d *ProcessDriver) GetHypervisorInfo(ctx context.Context) (*HypervisorInfo, error) {
	capabilities, err := d.GetCapabilities(ctx)
	if err != nil {
		return nil, err
	}
	return &HypervisorInfo{
		Type:           VMTypeProcess,
		Version:        "1.0.0",
		Virtualization: "None (native process)",
		NUMANodes:      1,
		GPUDevices:     []GPUDevice{},
		NetworkDevices: []NetworkDevice{},
		StorageDevices: []StorageDevice{},
		Capabilities:   capabilities,
		Metadata: map[string]interface{}{
			"driver":    "process",
			"node_id":   d.nodeID,
			"base_path": d.basePath,
		},
	}, nil
}

// --- Unsupported operations (honest errors) -------------------------------

func (d *ProcessDriver) Pause(ctx context.Context, vmID string) error {
	return ErrOperationNotSupported
}

func (d *ProcessDriver) Resume(ctx context.Context, vmID string) error {
	return ErrOperationNotSupported
}

func (d *ProcessDriver) Snapshot(ctx context.Context, vmID, name string, params map[string]string) (string, error) {
	return "", ErrOperationNotSupported
}

func (d *ProcessDriver) Migrate(ctx context.Context, vmID, target string, params map[string]string) error {
	return ErrOperationNotSupported
}

func (d *ProcessDriver) HotPlugDevice(ctx context.Context, vmID string, device *DeviceConfig) error {
	return ErrOperationNotSupported
}

func (d *ProcessDriver) HotUnplugDevice(ctx context.Context, vmID string, deviceID string) error {
	return ErrOperationNotSupported
}

func (d *ProcessDriver) ConfigureCPUPinning(ctx context.Context, vmID string, pinning *CPUPinningConfig) error {
	return ErrOperationNotSupported
}

func (d *ProcessDriver) ConfigureNUMA(ctx context.Context, vmID string, topology *NUMATopology) error {
	return ErrOperationNotSupported
}
