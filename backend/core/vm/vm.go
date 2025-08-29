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
	// StateUnknown represents an unknown VM state
	StateUnknown State = "unknown"

	// StateCreated means the VM has been created but not started
	StateCreated State = "created"

	// StateCreating represents a VM that is being created
	StateCreating State = "creating"

	// StateProvisioning represents a VM that is being provisioned
	StateProvisioning State = "provisioning"

	// StateRunning means the VM is currently running
	StateRunning State = "running"

	// StateStopped means the VM has been stopped
	StateStopped State = "stopped"

	// StatePaused represents a paused VM
	StatePaused State = "paused"

	// StatePausing represents a VM that is being paused
	StatePausing State = "pausing"

	// StateResuming represents a VM that is being resumed
	StateResuming State = "resuming"

	// StateRestarting represents a VM that is being restarted
	StateRestarting State = "restarting"

	// StateDeleting represents a VM that is being deleted
	StateDeleting State = "deleting"

	// StateMigrating represents a VM that is being migrated
	StateMigrating State = "migrating"

	// StateFailed means the VM has failed to start or has crashed
	StateFailed State = "failed"

	// Legacy state constants for compatibility
	VMStateError     = StateFailed
	VMStateRunning   = StateRunning
	VMStateDeleting  = StateDeleting
	VMStateCreating  = StateCreating
	VMStateMigrating = StateMigrating
	VMStateSuspended = StatePaused
	VMStatePaused    = StatePaused
	VMStateStopped   = StateStopped
)

// VMConfig holds configuration for a VM
type VMConfig struct {
	ID                      string                           `yaml:"id" json:"id"`
	Name                    string                           `yaml:"name" json:"name"`
	Command                 string                           `yaml:"command" json:"command"`
	Args                    []string                         `yaml:"args" json:"args"`
	CPUShares               int                              `yaml:"cpu_shares" json:"cpu_shares"`
	MemoryMB                int                              `yaml:"memory_mb" json:"memory_mb"`
	DiskSizeGB              int                              `yaml:"disk_size_gb" json:"disk_size_gb"`
	RootFS                  string                           `yaml:"rootfs" json:"rootfs"`
	Mounts                  []Mount                          `yaml:"mounts" json:"mounts"`
	Env                     map[string]string                `yaml:"env" json:"env"`
	NetworkID               string                           `yaml:"network_id" json:"network_id"`
	WorkDir                 string                           `yaml:"work_dir" json:"work_dir"`
	Tags                    map[string]string                `yaml:"tags" json:"tags"`
	PredictivePrefetching   *PredictivePrefetchingConfig     `yaml:"predictive_prefetching,omitempty" json:"predictive_prefetching,omitempty"`
}

// PredictivePrefetchingConfig configures AI-driven predictive prefetching for VM migrations
type PredictivePrefetchingConfig struct {
	Enabled                bool              `yaml:"enabled" json:"enabled"`
	PredictionAccuracy     float64           `yaml:"prediction_accuracy" json:"prediction_accuracy"`         // Target accuracy (default: 0.85)
	MaxCacheSize           int64             `yaml:"max_cache_size" json:"max_cache_size"`                   // Max cache size in bytes
	PredictionLatencyMs    int64             `yaml:"prediction_latency_ms" json:"prediction_latency_ms"`     // Max prediction latency in ms
	ModelType              string            `yaml:"model_type" json:"model_type"`                           // "neural_network", "random_forest", etc.
	TrainingDataSize       int64             `yaml:"training_data_size" json:"training_data_size"`           // Max training samples to retain
	ContinuousLearning     bool              `yaml:"continuous_learning" json:"continuous_learning"`         // Enable continuous model training
	PrefetchAheadTime      string            `yaml:"prefetch_ahead_time" json:"prefetch_ahead_time"`         // How far ahead to prefetch (e.g., "5m")
	AIModelConfig          map[string]string `yaml:"ai_model_config,omitempty" json:"ai_model_config,omitempty"` // Model-specific configuration
}

// DefaultPredictivePrefetchingConfig returns default configuration for predictive prefetching
func DefaultPredictivePrefetchingConfig() *PredictivePrefetchingConfig {
	return &PredictivePrefetchingConfig{
		Enabled:                true,
		PredictionAccuracy:     TARGET_PREDICTION_ACCURACY,    // 0.85
		MaxCacheSize:           1024 * 1024 * 1024,            // 1GB
		PredictionLatencyMs:    TARGET_PREDICTION_LATENCY_MS,  // 10ms
		ModelType:              "neural_network",
		TrainingDataSize:       100000,                        // 100k samples
		ContinuousLearning:     true,
		PrefetchAheadTime:      "5m",
		AIModelConfig: map[string]string{
			"learning_rate":    "0.001",
			"batch_size":       "32",
			"epochs":          "100",
			"hidden_layers":   "128,64",
			"activation":      "relu",
		},
	}
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
	// New fields for enhanced VM management
	nodeID      string
	resourceID  string
	updatedAt   time.Time
	processInfo VMProcessInfo
}

// VMStats holds runtime statistics for a VM
type VMStats struct {
	CPUUsage    float64
	MemoryUsage int64
	NetworkSent int64
	NetworkRecv int64
	LastUpdated time.Time
}

// VMProcessInfo holds process information for a VM
type VMProcessInfo struct {
	PID             int
	PPID            int
	Command         string
	Args            []string
	StartTime       time.Time
	CPUTime         time.Duration
	MemoryRSS       int64
	MemoryVSZ       int64
	CPUUsagePercent float64
	MemoryUsageMB   int64
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

// SetState sets the VM's state
func (vm *VM) SetState(state State) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.state = state
	vm.updatedAt = time.Now()
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

// GetNodeID returns the node ID
func (vm *VM) GetNodeID() string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.nodeID
}

// NodeID returns the node ID (compatibility method)
func (vm *VM) NodeID() string {
	return vm.GetNodeID()
}

// GetResourceID returns the resource ID
func (vm *VM) GetResourceID() string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.resourceID
}

// ResourceID returns the resource ID (compatibility method)
func (vm *VM) ResourceID() string {
	return vm.GetResourceID()
}

// GetUpdatedAt returns the last update time
func (vm *VM) GetUpdatedAt() time.Time {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.updatedAt
}

// UpdatedAt returns the last update time (compatibility method)
func (vm *VM) UpdatedAt() time.Time {
	return vm.GetUpdatedAt()
}

// GetProcessInfo returns the process information
func (vm *VM) GetProcessInfo() VMProcessInfo {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.processInfo
}

// ProcessInfo returns the process information (compatibility method)
func (vm *VM) ProcessInfo() VMProcessInfo {
	return vm.GetProcessInfo()
}

// SetNodeID sets the node ID
func (vm *VM) SetNodeID(nodeID string) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.nodeID = nodeID
	vm.updatedAt = time.Now()
}

// SetResourceID sets the resource ID
func (vm *VM) SetResourceID(resourceID string) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.resourceID = resourceID
	vm.updatedAt = time.Now()
}

// SetProcessInfo sets the process information
func (vm *VM) SetProcessInfo(processInfo VMProcessInfo) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.processInfo = processInfo
	vm.updatedAt = time.Now()
}

// SetUpdatedAt sets the updated time
func (vm *VM) SetUpdatedAt(t time.Time) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.updatedAt = t
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

// SetStartedAt sets the started time
func (vm *VM) SetStartedAt(t time.Time) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.startedAt = &t
	vm.updatedAt = time.Now()
}

// Config returns the VM configuration
func (vm *VM) Config() VMConfig {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config
}

// ErrorMessage returns the VM's error message
func (vm *VM) ErrorMessage() string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	// Return empty string for now, can be extended with actual error field
	return ""
}

// CreatedAt returns when the VM was created
func (vm *VM) CreatedAt() time.Time {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.createdAt
}

// GetCreatedAt returns when the VM was created
func (vm *VM) GetCreatedAt() time.Time {
	return vm.CreatedAt()
}

// GetCommand returns the VM's command
func (vm *VM) GetCommand() string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.Command
}

// GetArgs returns the VM's arguments
func (vm *VM) GetArgs() []string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.Args
}

// GetCPUShares returns the VM's CPU shares
func (vm *VM) GetCPUShares() int {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.CPUShares
}

// GetMemoryMB returns the VM's memory in MB
func (vm *VM) GetMemoryMB() int {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.MemoryMB
}

// GetDiskSizeGB returns the VM's disk size in GB
func (vm *VM) GetDiskSizeGB() int {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.DiskSizeGB
}

// GetTags returns the VM's tags
func (vm *VM) GetTags() map[string]string {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	return vm.config.Tags
}

// GetStats returns the VM's statistics
func (vm *VM) GetStats() VMStats {
	vm.statsLock.RLock()
	defer vm.statsLock.RUnlock()
	return vm.stats
}

// SetName sets the VM's name
func (vm *VM) SetName(name string) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.Name = name
	vm.updatedAt = time.Now()
}

// SetCPUShares sets the VM's CPU shares
func (vm *VM) SetCPUShares(cpuShares int) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.CPUShares = cpuShares
	vm.updatedAt = time.Now()
}

// SetMemoryMB sets the VM's memory in MB
func (vm *VM) SetMemoryMB(memoryMB int) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.MemoryMB = memoryMB
	vm.updatedAt = time.Now()
}

// SetDiskSizeGB sets the VM's disk size in GB
func (vm *VM) SetDiskSizeGB(diskSizeGB int) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.DiskSizeGB = diskSizeGB
	vm.updatedAt = time.Now()
}

// SetTags sets the VM's tags
func (vm *VM) SetTags(tags map[string]string) {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.config.Tags = tags
	vm.updatedAt = time.Now()
}

// Delete deletes the VM (stub implementation)
func (vm *VM) Delete() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.state = StateDeleting
	// Stub implementation - would actually delete VM resources
	return nil
}

// ResumeFromState resumes VM from a saved state (stub implementation)
func (vm *VM) ResumeFromState(statePath string) error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	vm.state = StateRunning
	// Stub implementation - would actually resume VM from state
	return nil
}

// GetDiskPaths returns the disk paths for the VM (stub implementation)
func (vm *VM) GetDiskPaths() ([]string, error) {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	// Stub implementation - would return actual disk paths
	return []string{"/var/lib/novacron/vms/" + vm.config.ID + "/disk.qcow2"}, nil
}

// GetMemoryStatePath returns the memory state path for the VM (stub implementation)
func (vm *VM) GetMemoryStatePath() (string, error) {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	// Stub implementation - would return actual memory state path
	return "/var/lib/novacron/vms/" + vm.config.ID + "/memory.state", nil
}

// GetMemoryDeltaPath returns the memory delta path for the VM (stub implementation)
func (vm *VM) GetMemoryDeltaPath(iteration int) (string, error) {
	vm.mutex.RLock()
	defer vm.mutex.RUnlock()
	// Stub implementation - would return actual memory delta path
	return "/var/lib/novacron/vms/" + vm.config.ID + "/memory.delta." + fmt.Sprintf("%d", iteration), nil
}

// Suspend suspends the VM (stub implementation)
func (vm *VM) Suspend() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	if vm.state != StateRunning {
		return fmt.Errorf("VM is not running")
	}
	vm.state = StatePaused
	// Stub implementation - would actually suspend VM
	return nil
}

// Resume resumes the VM (stub implementation)
func (vm *VM) Resume() error {
	vm.mutex.Lock()
	defer vm.mutex.Unlock()
	if vm.state != StatePaused && vm.state != VMStateSuspended {
		return fmt.Errorf("VM is not suspended or paused")
	}
	vm.state = StateRunning
	// Stub implementation - would actually resume VM
	return nil
}
