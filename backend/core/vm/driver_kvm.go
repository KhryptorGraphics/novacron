package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

// KVMDriver implements the VMDriver interface for KVM-based VMs
type KVMDriver struct {
	nodeID         string
	qemuBinaryPath string
	vmBasePath     string
	vms            map[string]*KVMInfo
	vmLock         sync.RWMutex
}

// KVMInfo stores information about a KVM VM
type KVMInfo struct {
	ID         string
	Spec       VMSpec
	Process    *os.Process
	PID        int
	Status     VMState
	DiskPath   string
	ConfigPath string
	SocketPath string
	StartTime  time.Time
}

// NewKVMDriver creates a new KVM driver
func NewKVMDriver(nodeID, qemuBinaryPath, vmBasePath string) *KVMDriver {
	// Create VM base directory if it doesn't exist
	if err := os.MkdirAll(vmBasePath, 0755); err != nil {
		log.Printf("Warning: Failed to create VM base directory %s: %v", vmBasePath, err)
	}
	
	return &KVMDriver{
		nodeID:         nodeID,
		qemuBinaryPath: qemuBinaryPath,
		vmBasePath:     vmBasePath,
		vms:            make(map[string]*KVMInfo),
	}
}

// Create creates a new KVM VM
func (d *KVMDriver) Create(ctx context.Context, spec VMSpec) (string, error) {
	log.Printf("Creating KVM VM with image %s", spec.Image)
	
	// Generate a unique VM ID
	vmID := fmt.Sprintf("novacron-kvm-%s", strconv.FormatInt(time.Now().UnixNano(), 16))
	
	// Create VM directory
	vmDir := filepath.Join(d.vmBasePath, vmID)
	if err := os.MkdirAll(vmDir, 0755); err != nil {
		return "", fmt.Errorf("failed to create VM directory: %w", err)
	}
	
	// Create disk image
	diskPath := filepath.Join(vmDir, "disk.qcow2")
	
	// Check if the image is a URL or local path
	if strings.HasPrefix(spec.Image, "http://") || strings.HasPrefix(spec.Image, "https://") {
		// For cloud images or downloadable images, we would download it
		// This is simplified for demonstration
		return "", fmt.Errorf("downloading images not implemented yet")
	} else if _, err := os.Stat(spec.Image); err == nil {
		// If the image is a local path, create a backing file
		createCmd := exec.CommandContext(ctx, "qemu-img", "create", 
			"-f", "qcow2", 
			"-o", fmt.Sprintf("backing_file=%s", spec.Image),
			diskPath, 
			fmt.Sprintf("%dM", spec.DiskMB))
		
		output, err := createCmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("failed to create disk image: %w, output: %s", err, string(output))
		}
	} else {
		// Create a new disk image
		createCmd := exec.CommandContext(ctx, "qemu-img", "create", 
			"-f", "qcow2", 
			diskPath, 
			fmt.Sprintf("%dM", spec.DiskMB))
		
		output, err := createCmd.CombinedOutput()
		if err != nil {
			return "", fmt.Errorf("failed to create disk image: %w, output: %s", err, string(output))
		}
	}
	
	// Create QEMU monitor socket path
	socketPath := filepath.Join(vmDir, "qmp.sock")
	
	// Create VM config file (for storing metadata)
	configPath := filepath.Join(vmDir, "vm.json")
	configData, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal VM config: %w", err)
	}
	
	if err := ioutil.WriteFile(configPath, configData, 0644); err != nil {
		return "", fmt.Errorf("failed to write VM config: %w", err)
	}
	
	// Store VM info
	d.vmLock.Lock()
	d.vms[vmID] = &KVMInfo{
		ID:         vmID,
		Spec:       spec,
		Status:     VMStateStopped,
		DiskPath:   diskPath,
		ConfigPath: configPath,
		SocketPath: socketPath,
	}
	d.vmLock.Unlock()
	
	log.Printf("Created KVM VM %s with disk %s", vmID, diskPath)
	return vmID, nil
}

// Start starts a KVM VM
func (d *KVMDriver) Start(ctx context.Context, vmID string) error {
	log.Printf("Starting KVM VM %s", vmID)
	
	// Get VM info
	d.vmLock.RLock()
	vmInfo, exists := d.vms[vmID]
	d.vmLock.RUnlock()
	
	if !exists {
		return fmt.Errorf("KVM VM %s not found", vmID)
	}
	
	// Check if VM is already running
	if vmInfo.Status == VMStateRunning && vmInfo.Process != nil {
		// Check if process is still alive
		if err := vmInfo.Process.Signal(syscall.Signal(0)); err == nil {
			// Process is still running
			return nil
		}
	}
	
	// Prepare QEMU command
	qemuArgs := []string{
		// Basic VM settings
		"-name", vmID,
		"-machine", "accel=kvm",
		"-cpu", "host",
		"-m", fmt.Sprintf("%d", vmInfo.Spec.MemoryMB),
		"-smp", fmt.Sprintf("%d", vmInfo.Spec.VCPU),
		
		// Disk
		"-drive", fmt.Sprintf("file=%s,if=virtio,media=disk", vmInfo.DiskPath),
		
		// Network (using virtio)
		"-netdev", "user,id=net0",
		"-device", "virtio-net-pci,netdev=net0",
		
		// QMP monitor socket for control
		"-qmp", fmt.Sprintf("unix:%s,server,nowait", vmInfo.SocketPath),
		
		// Daemonize
		"-daemonize",
	}
	
	// Add additional devices based on VM spec
	// In a real implementation, this would include:
	// - Additional disks
	// - Network configurations
	// - PCI passthrough
	// - etc.
	
	// Start QEMU process
	cmd := exec.CommandContext(ctx, d.qemuBinaryPath, qemuArgs...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		log.Printf("Failed to start KVM VM %s: %v, output: %s", vmID, err, string(output))
		return fmt.Errorf("failed to start KVM VM: %w", err)
	}
	
	// Get PID of the process
	// In a real implementation, we would parse the PID from QEMU output
	// or from a pidfile. For simplicity, we'll use a function to find the PID.
	pid, err := findQEMUProcessForVM(vmID)
	if err != nil {
		log.Printf("Warning: Started KVM VM %s but couldn't determine PID: %v", vmID, err)
	}
	
	// Update VM info
	d.vmLock.Lock()
	vmInfo.Status = VMStateRunning
	vmInfo.StartTime = time.Now()
	if pid > 0 {
		vmInfo.PID = pid
		// Get process handle
		process, err := os.FindProcess(pid)
		if err == nil {
			vmInfo.Process = process
		}
	}
	d.vmLock.Unlock()
	
	log.Printf("Started KVM VM %s with PID %d", vmID, pid)
	return nil
}

// Stop stops a KVM VM
func (d *KVMDriver) Stop(ctx context.Context, vmID string) error {
	log.Printf("Stopping KVM VM %s", vmID)
	
	// Get VM info
	d.vmLock.RLock()
	vmInfo, exists := d.vms[vmID]
	d.vmLock.RUnlock()
	
	if !exists {
		return fmt.Errorf("KVM VM %s not found", vmID)
	}
	
	// Check if VM is already stopped
	if vmInfo.Status == VMStateStopped || vmInfo.Process == nil {
		return nil
	}
	
	// Try to use QMP to shutdown gracefully
	success := false
	if vmInfo.SocketPath != "" {
		if err := qmpShutdown(vmInfo.SocketPath); err == nil {
			success = true
		} else {
			log.Printf("Failed to gracefully shutdown VM %s: %v", vmID, err)
		}
	}
	
	// If graceful shutdown failed, terminate the process
	if !success && vmInfo.Process != nil {
		if err := vmInfo.Process.Signal(syscall.SIGTERM); err != nil {
			log.Printf("Failed to send SIGTERM to VM %s: %v", vmID, err)
			
			// Try harder with SIGKILL
			if err := vmInfo.Process.Kill(); err != nil {
				return fmt.Errorf("failed to kill VM process: %w", err)
			}
		}
	}
	
	// Wait for the process to exit
	timeout := time.After(10 * time.Second)
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-timeout:
			// Force kill if timeout
			if vmInfo.Process != nil {
				_ = vmInfo.Process.Kill()
			}
			// Update VM status
			d.vmLock.Lock()
			vmInfo.Status = VMStateStopped
			vmInfo.Process = nil
			d.vmLock.Unlock()
			return nil
		case <-ticker.C:
			// Check if process has exited
			if vmInfo.Process != nil {
				if err := vmInfo.Process.Signal(syscall.Signal(0)); err != nil {
					// Process has exited
					d.vmLock.Lock()
					vmInfo.Status = VMStateStopped
					vmInfo.Process = nil
					d.vmLock.Unlock()
					return nil
				}
			} else {
				// No process reference
				d.vmLock.Lock()
				vmInfo.Status = VMStateStopped
				d.vmLock.Unlock()
				return nil
			}
		}
	}
}

// Delete deletes a KVM VM
func (d *KVMDriver) Delete(ctx context.Context, vmID string) error {
	log.Printf("Deleting KVM VM %s", vmID)
	
	// Stop the VM first
	if err := d.Stop(ctx, vmID); err != nil {
		log.Printf("Warning: Failed to stop VM %s before deletion: %v", vmID, err)
	}
	
	// Get VM info
	d.vmLock.RLock()
	vmInfo, exists := d.vms[vmID]
	d.vmLock.RUnlock()
	
	if !exists {
		return fmt.Errorf("KVM VM %s not found", vmID)
	}
	
	// Remove VM directory
	vmDir := filepath.Join(d.vmBasePath, vmID)
	if err := os.RemoveAll(vmDir); err != nil {
		log.Printf("Warning: Failed to remove VM directory %s: %v", vmDir, err)
	}
	
	// Remove from VM map
	d.vmLock.Lock()
	delete(d.vms, vmID)
	d.vmLock.Unlock()
	
	log.Printf("Deleted KVM VM %s", vmID)
	return nil
}

// GetStatus gets the status of a KVM VM
func (d *KVMDriver) GetStatus(ctx context.Context, vmID string) (VMState, error) {
	// Get VM info
	d.vmLock.RLock()
	vmInfo, exists := d.vms[vmID]
	d.vmLock.RUnlock()
	
	if !exists {
		return VMStateUnknown, fmt.Errorf("KVM VM %s not found", vmID)
	}
	
	// If there's no process reference, it's stopped
	if vmInfo.Process == nil {
		return VMStateStopped, nil
	}
	
	// Check if process is still alive
	if err := vmInfo.Process.Signal(syscall.Signal(0)); err != nil {
		// Process is not running
		d.vmLock.Lock()
		vmInfo.Status = VMStateStopped
		vmInfo.Process = nil
		d.vmLock.Unlock()
		return VMStateStopped, nil
	}
	
	// Process is running, check QMP status if available
	if vmInfo.SocketPath != "" {
		status, err := qmpGetStatus(vmInfo.SocketPath)
		if err == nil {
			return status, nil
		}
	}
	
	// Return the cached status
	return vmInfo.Status, nil
}

// GetInfo gets information about a KVM VM
func (d *KVMDriver) GetInfo(ctx context.Context, vmID string) (*VM, error) {
	// Get VM info
	d.vmLock.RLock()
	vmInfo, exists := d.vms[vmID]
	d.vmLock.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("KVM VM %s not found", vmID)
	}
	
	// Get current status
	status, err := d.GetStatus(ctx, vmID)
	if err != nil {
		return nil, err
	}
	
	// Create basic VM info
	vm := &VM{
		ID:        vmID,
		Name:      vmID,
		Spec:      vmInfo.Spec,
		State:     status,
		NodeID:    d.nodeID,
		UpdatedAt: time.Now(),
	}
	
	// Add process info if running
	if status == VMStateRunning && vmInfo.PID > 0 {
		vm.ProcessInfo = VMProcessInfo{
			PID:           vmInfo.PID,
			StartTime:     vmInfo.StartTime,
			LastUpdatedAt: time.Now(),
		}
		
		// Get resource usage via QMP or system calls
		// For simplicity, we'll use a basic approach
		if usage, err := getProcessResourceUsage(vmInfo.PID); err == nil {
			vm.ProcessInfo.CPUUsagePercent = usage.CPU
			vm.ProcessInfo.MemoryUsageMB = usage.MemoryMB
			vm.ProcessInfo.ThreadCount = usage.Threads
		}
	}
	
	// In a real implementation, we would add:
	// - Network info from QEMU
	// - Storage info from QEMU
	// - Additional metrics
	
	return vm, nil
}

// Helper function to find QEMU process ID for a VM
func findQEMUProcessForVM(vmID string) (int, error) {
	// In a real implementation, this would use proper methods to identify the process
	// For simplicity, we'll use ps and grep
	cmd := exec.Command("ps", "-eo", "pid,command")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("failed to list processes: %w", err)
	}
	
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		if strings.Contains(line, "qemu") && strings.Contains(line, vmID) {
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			
			pid, err := strconv.Atoi(fields[0])
			if err != nil {
				continue
			}
			
			return pid, nil
		}
	}
	
	return 0, fmt.Errorf("could not find QEMU process for VM %s", vmID)
}

// Resource usage information
type resourceUsage struct {
	CPU      float64
	MemoryMB int
	Threads  int
}

// Helper function to get process resource usage
func getProcessResourceUsage(pid int) (resourceUsage, error) {
	usage := resourceUsage{}
	
	// In a real implementation, this would use cgroups, proc filesystem, or OS-specific APIs
	// For simplicity, we'll use ps
	cmd := exec.Command("ps", "-p", strconv.Itoa(pid), "-o", "%cpu,%mem,vsz,nlwp")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return usage, fmt.Errorf("failed to get resource usage: %w", err)
	}
	
	lines := strings.Split(string(output), "\n")
	if len(lines) < 2 {
		return usage, fmt.Errorf("unexpected output format")
	}
	
	fields := strings.Fields(lines[1])
	if len(fields) < 4 {
		return usage, fmt.Errorf("unexpected output format")
	}
	
	// Parse CPU percentage
	cpuPct, err := strconv.ParseFloat(fields[0], 64)
	if err == nil {
		usage.CPU = cpuPct
	}
	
	// Parse memory (from VSZ in KB to MB)
	memKB, err := strconv.ParseFloat(fields[2], 64)
	if err == nil {
		usage.MemoryMB = int(memKB / 1024)
	}
	
	// Parse thread count
	threads, err := strconv.Atoi(fields[3])
	if err == nil {
		usage.Threads = threads
	}
	
	return usage, nil
}

// Helper function to shutdown VM via QMP
func qmpShutdown(socketPath string) error {
	// In a real implementation, this would use a QMP client library
	// For simplicity, we'll use a simplified approach with socat
	
	// First, send capabilities negotiation
	capCmd := fmt.Sprintf("echo '{\"execute\": \"qmp_capabilities\"}' | socat - UNIX-CONNECT:%s", socketPath)
	cmd := exec.Command("bash", "-c", capCmd)
	if _, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to negotiate QMP capabilities: %w", err)
	}
	
	// Then send shutdown command
	shutdownCmd := fmt.Sprintf("echo '{\"execute\": \"system_powerdown\"}' | socat - UNIX-CONNECT:%s", socketPath)
	cmd = exec.Command("bash", "-c", shutdownCmd)
	if _, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to send shutdown command: %w", err)
	}
	
	return nil
}

// Helper function to get VM status via QMP
func qmpGetStatus(socketPath string) (VMState, error) {
	// In a real implementation, this would use a QMP client library
	// For simplicity, we'll return a default state
	return VMStateRunning, nil
}
