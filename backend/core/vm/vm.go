package vm

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"github.com/google/uuid"
)

// State represents the current state of a VM
type State string

const (
	// StateCreated means the VM has been created but not started
	StateCreated State = "created"
	// StateRunning means the VM is currently running
	StateRunning State = "running"
	// StateStopped means the VM has been stopped
	StateStopped State = "stopped"
	// StateFailed means the VM has failed to start or has crashed
	StateFailed State = "failed"
)

// VMConfig holds configuration for a VM
type VMConfig struct {
	ID        string            `yaml:"id" json:"id"`
	Name      string            `yaml:"name" json:"name"`
	Command   string            `yaml:"command" json:"command"`
	Args      []string          `yaml:"args" json:"args"`
	CPUShares int               `yaml:"cpu_shares" json:"cpu_shares"`
	MemoryMB  int               `yaml:"memory_mb" json:"memory_mb"`
	RootFS    string            `yaml:"rootfs" json:"rootfs"`
	Mounts    []Mount           `yaml:"mounts" json:"mounts"`
	Env       map[string]string `yaml:"env" json:"env"`
	NetworkID string            `yaml:"network_id" json:"network_id"`
	WorkDir   string            `yaml:"work_dir" json:"work_dir"`
	Tags      map[string]string `yaml:"tags" json:"tags"`
}

// Mount represents a filesystem mount for a VM
type Mount struct {
	Source string `yaml:"source" json:"source"`
	Target string `yaml:"target" json:"target"`
	Type   string `yaml:"type" json:"type"`
	Flags  int    `yaml:"flags" json:"flags"`
	Data   string `yaml:"data" json:"data"`
}

// VMInfo contains runtime information about a VM
type VMInfo struct {
	ID           string            `json:"id"`
	Name         string            `json:"name"`
	State        State             `json:"state"`
	PID          int               `json:"pid"`
	CPUShares    int               `json:"cpu_shares"`
	MemoryMB     int               `json:"memory_mb"`
	CPUUsage     float64           `json:"cpu_usage"`
	MemoryUsage  int64             `json:"memory_usage"`
	NetworkSent  int64             `json:"network_sent"`
	NetworkRecv  int64             `json:"network_recv"`
	CreatedAt    time.Time         `json:"created_at"`
	StartedAt    *time.Time        `json:"started_at"`
	StoppedAt    *time.Time        `json:"stopped_at"`
	Tags         map[string]string `json:"tags"`
	NetworkID    string            `json:"network_id"`
	IPAddress    string            `json:"ip_address"`
	RootFS       string            `json:"rootfs"`
	ErrorMessage string            `json:"error_message,omitempty"`
}

// VM represents a virtual machine
type VM struct {
	config     VMConfig
	state      State
	pid        int
	cmd        *exec.Cmd
	mutex      sync.RWMutex
	createdAt  time.Time
	startedAt  *time.Time
	stoppedAt  *time.Time
	cgroupPath string
	netns      string
	ipAddress  string
	statsLock  sync.RWMutex
	stats      VMStats
}

// VMStats holds runtime statistics for a VM
type VMStats struct {
	CPUUsage    float64
	MemoryUsage int64
	NetworkSent int64
	NetworkRecv int64
	LastUpdated time.Time
}

// NewVM creates a new VM instance
func NewVM(config VMConfig) (*VM, error) {
	if config.Name == "" {
		return nil, fmt.Errorf("VM name must be specified")
	}

	if config.Command == "" {
		return nil, fmt.Errorf("VM command must be specified")
	}

	if config.ID == "" {
		config.ID = uuid.New().String()
	}

	// Default resource limits if not specified
	if config.CPUShares == 0 {
		config.CPUShares = 1024 // Default CPU shares
	}
	if config.MemoryMB == 0 {
		config.MemoryMB = 512 // Default 512MB of memory
	}

	vm := &VM{
		config:    config,
		state:     StateCreated,
		createdAt: time.Now(),
		stats: VMStats{
			LastUpdated: time.Now(),
		},
	}

	return vm, nil
}

// ID returns the VM's ID
func (vm *VM) ID() string {
	return vm.config.ID
}

// Name returns the VM's name
func (vm *VM) Name() string {
	return vm.config.Name
}

// State returns the VM's current state
func (vm *VM) State() State {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.state
}

// IsRunning returns true if the VM is running
func (vm *VM) IsRunning() bool {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.state == StateRunning
}

// Start starts the VM
func (vm *VM) Start() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state == StateRunning {
		return fmt.Errorf("VM is already running")
	}

	// Set up cgroups
	if err := vm.setupCgroups(); err != nil {
		return fmt.Errorf("failed to set up cgroups: %w", err)
	}

	// Set up namespaces
	if err := vm.setupNamespaces(); err != nil {
		return fmt.Errorf("failed to set up namespaces: %w", err)
	}

	// Set up command
	vm.cmd = exec.Command(vm.config.Command, vm.config.Args...)

	// Set up process namespaces
	vm.cmd.SysProcAttr = &syscall.SysProcAttr{
		Cloneflags: syscall.CLONE_NEWUTS | syscall.CLONE_NEWPID | syscall.CLONE_NEWNS | syscall.CLONE_NEWNET | syscall.CLONE_NEWIPC,
	}

	// Set up environment variables
	if vm.config.Env != nil {
		env := os.Environ()
		for k, v := range vm.config.Env {
			env = append(env, fmt.Sprintf("%s=%s", k, v))
		}
		vm.cmd.Env = env
	}

	// Set up working directory
	if vm.config.WorkDir != "" {
		vm.cmd.Dir = vm.config.WorkDir
	}

	// Start the process
	if err := vm.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start VM process: %w", err)
	}

	vm.pid = vm.cmd.Process.Pid

	// Move process to appropriate cgroups
	if err := vm.assignToCgroups(); err != nil {
		return fmt.Errorf("failed to assign process to cgroups: %w", err)
	}

	// Set up network if needed
	if vm.config.NetworkID != "" {
		if err := vm.setupNetwork(); err != nil {
			return fmt.Errorf("failed to set up network: %w", err)
		}
	}

	now := time.Now()
	vm.startedAt = &now
	vm.state = StateRunning

	// Monitor the process
	go vm.monitor()

	log.Printf("Started VM %s (%s) with PID %d", vm.Name(), vm.ID(), vm.pid)
	return nil
}

// Stop stops the VM
func (vm *VM) Stop() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state != StateRunning {
		return fmt.Errorf("VM is not running")
	}

	// Send SIGTERM signal
	if err := vm.cmd.Process.Signal(syscall.SIGTERM); err != nil {
		log.Printf("Failed to send SIGTERM to VM %s: %v, will try SIGKILL", vm.ID(), err)
		
		// If SIGTERM fails, try SIGKILL
		if err := vm.cmd.Process.Kill(); err != nil {
			return fmt.Errorf("failed to kill VM: %w", err)
		}
	}

	// Wait for the process to exit
	done := make(chan error)
	go func() {
		done <- vm.cmd.Wait()
	}()

	// Wait for process to exit or timeout
	select {
	case err := <-done:
		if err != nil {
			log.Printf("VM %s process exited with error: %v", vm.ID(), err)
		}
	case <-time.After(10 * time.Second):
		// If it doesn't exit within timeout, force kill
		if err := vm.cmd.Process.Kill(); err != nil {
			log.Printf("Failed to force kill VM %s: %v", vm.ID(), err)
		}
		<-done // Wait for the process to be killed
	}

	now := time.Now()
	vm.stoppedAt = &now
	vm.state = StateStopped

	// Clean up cgroups and namespaces
	if err := vm.cleanupCgroups(); err != nil {
		log.Printf("Failed to clean up cgroups for VM %s: %v", vm.ID(), err)
	}

	if err := vm.cleanupNamespaces(); err != nil {
		log.Printf("Failed to clean up namespaces for VM %s: %v", vm.ID(), err)
	}

	log.Printf("Stopped VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// Cleanup cleans up VM resources
func (vm *VM) Cleanup() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	if vm.state == StateRunning {
		return fmt.Errorf("cannot clean up a running VM")
	}

	// Clean up any remaining resources
	if err := vm.cleanupCgroups(); err != nil {
		log.Printf("Failed to clean up cgroups for VM %s: %v", vm.ID(), err)
	}

	if err := vm.cleanupNamespaces(); err != nil {
		log.Printf("Failed to clean up namespaces for VM %s: %v", vm.ID(), err)
	}

	// Clean up any storage used by the VM
	// This will be implemented based on specific storage requirements

	log.Printf("Cleaned up VM %s (%s)", vm.Name(), vm.ID())
	return nil
}

// GetInfo returns information about the VM
func (vm *VM) GetInfo() VMInfo {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	vm.statsLock.RLock()
	defer vm.statsLock.RUnlock()

	info := VMInfo{
		ID:          vm.config.ID,
		Name:        vm.config.Name,
		State:       vm.state,
		PID:         vm.pid,
		CPUShares:   vm.config.CPUShares,
		MemoryMB:    vm.config.MemoryMB,
		CPUUsage:    vm.stats.CPUUsage,
		MemoryUsage: vm.stats.MemoryUsage,
		NetworkSent: vm.stats.NetworkSent,
		NetworkRecv: vm.stats.NetworkRecv,
		CreatedAt:   vm.createdAt,
		StartedAt:   vm.startedAt,
		StoppedAt:   vm.stoppedAt,
		Tags:        vm.config.Tags,
		NetworkID:   vm.config.NetworkID,
		IPAddress:   vm.ipAddress,
		RootFS:      vm.config.RootFS,
	}

	return info
}

// Private methods

func (vm *VM) setupCgroups() error {
	// Create cgroup path
	vm.cgroupPath = filepath.Join("/sys/fs/cgroup", "novacron", vm.config.ID)
	
	// In a full implementation, we would create cgroups for CPU, memory, etc.
	// For now, we'll just log and return success
	log.Printf("Set up cgroups for VM %s at %s", vm.ID(), vm.cgroupPath)
	return nil
}

func (vm *VM) setupNamespaces() error {
	// In a full implementation, we would set up all required namespaces
	// For now, we'll just log and return success
	log.Printf("Set up namespaces for VM %s", vm.ID())
	return nil
}

func (vm *VM) assignToCgroups() error {
	// In a full implementation, we would assign the process to created cgroups
	// For now, we'll just log and return success
	log.Printf("Assigned VM %s (PID %d) to cgroups", vm.ID(), vm.pid)
	return nil
}

func (vm *VM) setupNetwork() error {
	// In a full implementation, we would set up the network namespace
	// For now, we'll just log and return success
	vm.ipAddress = "10.0.0.1" // Placeholder
	log.Printf("Set up network for VM %s with IP %s", vm.ID(), vm.ipAddress)
	return nil
}

func (vm *VM) cleanupCgroups() error {
	// In a full implementation, we would remove cgroups
	// For now, we'll just log and return success
	log.Printf("Cleaned up cgroups for VM %s", vm.ID())
	return nil
}

func (vm *VM) cleanupNamespaces() error {
	// In a full implementation, we would clean up namespaces
	// For now, we'll just log and return success
	log.Printf("Cleaned up namespaces for VM %s", vm.ID())
	return nil
}

func (vm *VM) monitor() {
	// In a full implementation, this would monitor the VM and collect stats
	for {
		// Wait for process to exit
		if err := vm.cmd.Wait(); err != nil {
			vm.mutex.Lock()
			if vm.state == StateRunning {
				log.Printf("VM %s process exited unexpectedly: %v", vm.ID(), err)
				vm.state = StateFailed
			}
			vm.mutex.Unlock()
			return
		}

		// If we get here, the process exited normally
		vm.mutex.Lock()
		if vm.state == StateRunning {
			now := time.Now()
			vm.stoppedAt = &now
			vm.state = StateStopped
			log.Printf("VM %s process exited normally", vm.ID())
		}
		vm.mutex.Unlock()
		return
	}
}

func (vm *VM) collectStats() {
	// In a full implementation, this would collect resource usage stats
	// For now, we'll just update the last updated time
	vm.statsLock.Lock()
	defer vm.statsLock.Unlock()
	vm.stats.LastUpdated = time.Now()
}
