package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	// "strings" // Currently unused
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
	qemuPath := "/usr/bin/qemu-system-x86_64" // Default
	if path, ok := config["qemu_path"].(string); ok {
		qemuPath = path
	}

	return NewKVMDriverEnhanced(qemuPath)
}

// NewKVMDriverEnhanced creates a new enhanced KVM driver
func NewKVMDriverEnhanced(qemuPath string) (VMDriver, error) {
	if qemuPath == "" {
		qemuPath = "/usr/bin/qemu-system-x86_64"
	}

	// Check if QEMU binary exists
	if _, err := os.Stat(qemuPath); err != nil {
		return nil, fmt.Errorf("QEMU binary not found at %s: %w", qemuPath, err)
	}

	vmBasePath := "/var/lib/novacron/vms"
	if err := os.MkdirAll(vmBasePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create VM base directory: %w", err)
	}

	return &KVMDriverEnhanced{
		qemuBinaryPath: qemuPath,
		vmBasePath:     vmBasePath,
		vms:            make(map[string]*KVMVMInfo),
	}, nil
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
	diskSizeMB := 8192 // Default 8GB
	if config.MemoryMB > 0 {
		diskSizeMB = config.MemoryMB * 10 // 10x memory size for disk
	}

	createCmd := exec.CommandContext(ctx, "qemu-img", "create",
		"-f", "qcow2",
		diskPath,
		fmt.Sprintf("%dM", diskSizeMB))

	if output, err := createCmd.CombinedOutput(); err != nil {
		return "", fmt.Errorf("failed to create disk image: %w, output: %s", err, string(output))
	}

	// Create VM info
	vmInfo := &KVMVMInfo{
		ID:          vmID,
		Config:      config,
		State:       StateCreated,
		DiskPath:    diskPath,
		ConfigPath:  filepath.Join(vmDir, "config.json"),
		MonitorPath: filepath.Join(vmDir, "monitor.sock"),
		VNCPort:     5900 + len(d.vms), // Simple VNC port allocation
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
	args := []string{
		"-machine", "pc-i440fx-2.8,accel=kvm",
		"-cpu", "host",
		"-m", strconv.Itoa(vmInfo.Config.MemoryMB),
		"-smp", strconv.Itoa(vmInfo.Config.CPUShares),
		"-drive", fmt.Sprintf("file=%s,format=qcow2,if=virtio", vmInfo.DiskPath),
		"-netdev", "user,id=net0",
		"-device", "virtio-net-pci,netdev=net0",
		"-vnc", fmt.Sprintf(":%d", vmInfo.VNCPort-5900),
		"-monitor", fmt.Sprintf("unix:%s,server,nowait", vmInfo.MonitorPath),
		"-daemonize",
		"-pidfile", filepath.Join(filepath.Dir(vmInfo.DiskPath), "qemu.pid"),
	}

	// Add memory balloon
	args = append(args, "-device", "virtio-balloon-pci")

	// Add virtio-rng for entropy
	args = append(args, "-device", "virtio-rng-pci")

	return args
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
		Version:               "QEMU/KVM",
		SupportsPause:         d.SupportsPause(),
		SupportsResume:        d.SupportsResume(),
		SupportsSnapshot:      d.SupportsSnapshot(),
		SupportsMigrate:       d.SupportsMigrate(),
		SupportsLiveMigration: d.SupportsLiveMigration(),
		SupportsHotPlug:       d.SupportsHotPlug(),
		SupportsGPUPassthrough: d.SupportsGPUPassthrough(),
		SupportsSRIOV:         d.SupportsSRIOV(),
		SupportsNUMA:          d.SupportsNUMA(),
		MaxVCPUs:              256,
		MaxMemoryMB:           1024 * 1024, // 1TB
		SupportedFeatures:     []string{"kvm", "qemu", "x86_64"},
		HardwareExtensions:    []string{"vmx", "svm"},
	}, nil
}

// GetHypervisorInfo returns information about the KVM hypervisor
func (d *KVMDriverEnhanced) GetHypervisorInfo(ctx context.Context) (*HypervisorInfo, error) {
	capabilities, err := d.GetCapabilities(ctx)
	if err != nil {
		return nil, err
	}

	return &HypervisorInfo{
		Type:            VMTypeKVM,
		Version:         "QEMU/KVM",
		ConnectionURI:   "qemu:///system",
		Hostname:        "localhost",
		CPUModel:        "host",
		CPUCores:        8,  // Default
		MemoryMB:        8192, // Default 8GB
		Virtualization:  "KVM",
		IOMMUEnabled:    false,
		NUMANodes:       1,
		GPUDevices:      []GPUDevice{},
		NetworkDevices:  []NetworkDevice{},
		StorageDevices:  []StorageDevice{},
		ActiveVMs:       len(d.vms),
		Capabilities:    capabilities,
		Metadata:        map[string]interface{}{
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
