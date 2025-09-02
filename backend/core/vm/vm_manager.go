package vm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/uuid"
)

// VMManager manages virtual machines across different drivers
type VMManager struct {
	drivers        map[VMType]VMDriver
	vms            map[string]*VM
	vmsMutex       sync.RWMutex
	driverFactory  VMDriverFactory
	scheduler      VMScheduler
	eventListeners []VMManagerEventListener
	eventMutex     sync.RWMutex
	vmCache        map[string]VMInfo
	mutex          sync.RWMutex
	ctx            context.Context
	cancel         context.CancelFunc
}

// VMManagerConfig contains configuration for the VM manager
type VMManagerConfig struct {
	DefaultDriver   VMType                           `yaml:"default_driver"`
	Drivers         map[VMType]VMDriverConfigManager `yaml:"drivers"`
	Scheduler       VMSchedulerConfig                `yaml:"scheduler"`
	UpdateInterval  time.Duration                    `yaml:"update_interval"`
	CleanupInterval time.Duration                    `yaml:"cleanup_interval"`
	DefaultVMType   VMType                           `yaml:"default_vm_type"`
	RetentionPeriod time.Duration                    `yaml:"retention_period"`
}

// VMDriverConfigLegacy contains legacy driver-specific configuration (use driver_factory.go VMDriverConfig instead)
type VMDriverConfigLegacy struct {
	Enabled bool
	Config  map[string]interface{}
}

// VMDriverConfigManager contains driver-specific configuration for VM manager
type VMDriverConfigManager struct {
	Enabled bool
	Config  map[string]interface{}
}

// VMSchedulerConfig contains scheduler configuration
type VMSchedulerConfig struct {
	Type   string
	Config map[string]interface{}
}

// VMManagerEventListener defines the interface for VM event listeners
type VMManagerEventListener interface {
	OnVMEvent(event VMEvent)
}

// VMEvent represents an event that occurred on a VM (defined here to avoid import cycles)
// Note: This may conflict with vm_events.go - use conditional compilation or merge definitions

// NewVMManager creates a new VM manager instance
func NewVMManager(config VMManagerConfig) (*VMManager, error) {
	ctx, cancel := context.WithCancel(context.Background())

	manager := &VMManager{
		drivers:        make(map[VMType]VMDriver),
		vms:            make(map[string]*VM),
		driverFactory:  nil, // Will be set later
		eventListeners: make([]VMManagerEventListener, 0),
		vmCache:        make(map[string]VMInfo),
		ctx:            ctx,
		cancel:         cancel,
	}

	// Initialize driver factory with default config
	driverConfig := DefaultVMDriverConfig("default-node")
	manager.driverFactory = NewVMDriverFactory(driverConfig)

	// Initialize drivers based on config
	for driverType, driverConfig := range config.Drivers {
		if !driverConfig.Enabled {
			continue
		}

		driver, err := manager.createDriverForType(driverType, driverConfig.Config)
		if err != nil {
			log.Printf("Failed to create driver %s: %v", driverType, err)
			continue
		}

		manager.drivers[driverType] = driver
		log.Printf("Initialized driver: %s", driverType)
	}

	// Initialize scheduler if configured
	if config.Scheduler.Type != "" {
		schedulerConfig := SchedulerConfig{
			Policy:                 SchedulerPolicyRoundRobin,
			EnableResourceChecking: true,
			EnableAntiAffinity:     false,
			EnableNodeLabels:       false,
			MaxVMsPerNode:          10,
		}
		scheduler := NewVMScheduler(schedulerConfig)
		manager.scheduler = *scheduler
	}

	return manager, nil
}

// createDriver creates a driver instance based on type and config
func createDriver(driverType VMType, config map[string]interface{}) (VMDriver, error) {
	switch driverType {
	case VMTypeKVM:
		uri, ok := config["uri"].(string)
		if !ok {
			uri = "qemu:///system" // Default KVM URI
		}
		return NewKVMDriverStub(map[string]interface{}{"uri": uri})
	case VMTypeContainer:
		return NewContainerDriverStub(config)
	case VMTypeContainerd:
		return NewContainerdDriver(config)
	default:
		return nil, fmt.Errorf("unsupported driver type: %s", driverType)
	}
}

// createScheduler creates a scheduler instance based on config
func createScheduler(config VMSchedulerConfig) (VMScheduler, error) {
	// For now, return a basic scheduler
	// This can be expanded to support different scheduler types
	scheduler := NewVMScheduler(SchedulerConfig{
		Policy: SchedulerPolicyRoundRobin,
	})
	return *scheduler, nil
}

// emitEvent emits a VM event to all registered listeners
func (m *VMManager) emitEvent(event VMEvent) {
	m.eventMutex.RLock()
	listeners := make([]VMManagerEventListener, len(m.eventListeners))
	copy(listeners, m.eventListeners)
	m.eventMutex.RUnlock()

	for _, listener := range listeners {
		go func(l VMManagerEventListener) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("VMManager: Event listener panicked: %v", r)
				}
			}()
			l.OnVMEvent(event)
		}(listener)
	}
}

// Start starts the VM manager
func (m *VMManager) Start() error { // Assuming m is *VMManager defined elsewhere
	log.Println("Starting VM manager")

	// Start the update loop
	go m.updateLoop()

	// Start the cleanup loop
	go m.cleanupLoop()

	return nil
}

// Stop stops the VM manager
func (m *VMManager) Stop() error {
	log.Println("Stopping VM manager")

	m.cancel() // Assuming cancel is part of the VMManager struct defined elsewhere

	return nil
}

// AddEventListener adds an event listener
func (m *VMManager) AddEventListener(listener VMManagerEventListener) { // Assuming VMManagerEventListener is defined elsewhere
	m.eventMutex.Lock() // Assuming eventMutex is part of VMManager struct
	defer m.eventMutex.Unlock()

	m.eventListeners = append(m.eventListeners, listener) // Assuming eventListeners is part of VMManager struct
}

// RemoveEventListener removes an event listener
func (m *VMManager) RemoveEventListener(listener VMManagerEventListener) {
	m.eventMutex.Lock()
	defer m.eventMutex.Unlock()

	for i, l := range m.eventListeners {
		if fmt.Sprintf("%p", l) == fmt.Sprintf("%p", listener) {
			m.eventListeners = append(m.eventListeners[:i], m.eventListeners[i+1:]...)
			break
		}
	}
}

// GetVM returns a VM by ID
func (m *VMManager) GetVM(vmID string) (*VM, error) { // Assuming VM is defined elsewhere
	m.vmsMutex.RLock() // Assuming vmsMutex is part of VMManager struct
	defer m.vmsMutex.RUnlock()

	vm, exists := m.vms[vmID] // Assuming vms map is part of VMManager struct
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}

	return vm, nil
}

// ListVMs returns all VMs
func (m *VMManager) ListVMs() map[string]*VM {
	m.vmsMutex.RLock()
	defer m.vmsMutex.RUnlock()

	result := make(map[string]*VM, len(m.vms))
	for id, vm := range m.vms {
		result[id] = vm
	}

	return result
}

// ListVMsByState returns all VMs with a specific state
func (m *VMManager) ListVMsByState(state VMState) []*VM { // Assuming VMState is defined elsewhere
	m.vmsMutex.RLock()
	defer m.vmsMutex.RUnlock()

	result := make([]*VM, 0)
	for _, vm := range m.vms {
		// Corrected comparison: Use vm.State() method
		if vm.State() == state { // Use method call
			result = append(result, vm)
		}
	}

	return result
}

// ListVMsByNode returns all VMs on a specific node
func (m *VMManager) ListVMsByNode(nodeID string) []*VM {
	m.vmsMutex.RLock()
	defer m.vmsMutex.RUnlock()

	result := make([]*VM, 0)
	for _, vm := range m.vms {
		// Corrected comparison: Use vm.NodeID() method (assuming it exists)
		if vm.NodeID() == nodeID { // Use method call
			result = append(result, vm)
		}
	}

	return result
}

// CountVMsByState counts VMs by state
func (m *VMManager) CountVMsByState() map[VMState]int {
	m.vmsMutex.RLock()
	defer m.vmsMutex.RUnlock()

	result := make(map[VMState]int)
	for _, vm := range m.vms {
		result[vm.State()]++ // Use method call
	}

	return result
}

// getDriver gets a driver for a VM type
func (m *VMManager) getDriver(config VMConfig) (VMDriver, error) { // Takes VMConfig now
	if m.driverFactory == nil { // Assuming driverFactory is part of VMManager struct
		return nil, errors.New("no driver factory configured")
	}

	return m.driverFactory(config) // Pass config
}


// GetDriverForConfig is an exported helper to obtain a driver for a VM config
func (m *VMManager) GetDriverForConfig(config VMConfig) (VMDriver, error) {
	return m.getDriver(config)
}

// checkDriverAvailability checks if a driver is available for a VM type
func (m *VMManager) checkDriverAvailability(config VMConfig) error { // Takes VMConfig now
	_, err := m.getDriver(config)
	return err
}

// listAvailableVMTypes lists all available VM types
func (m *VMManager) listAvailableVMTypes() []VMType { // Assuming VMType is defined elsewhere
	// Define the standard VM types to check
	standardTypes := []VMType{
		VMTypeContainer,
		VMTypeContainerd,
		VMTypeKVM,
		VMTypeProcess,
	}

	// Check which ones have working drivers
	availableTypes := make([]VMType, 0)
	for _, vmType := range standardTypes {
		// Need a dummy config to check driver availability
		dummyConfig := VMConfig{ /* Populate minimally if needed */ }
		// This check might need refinement depending on how driverFactory works
		if m.checkDriverAvailability(dummyConfig) == nil {
			availableTypes = append(availableTypes, vmType)
		}
	}

	return availableTypes
}

// createDriverForType creates a driver for the specified type
func (m *VMManager) createDriverForType(vmType VMType, config map[string]interface{}) (VMDriver, error) {
	switch vmType {
	case VMTypeKVM:
		return NewCoreStubDriver(config)
	case VMTypeContainer:
		return NewContainerDriverStub(config)
	case VMTypeContainerd:
		return NewContainerdDriver(config)
	case VMTypeProcess:
		return NewProcessDriverStub(config)
	default:
		return nil, fmt.Errorf("unsupported driver type: %s", vmType)
	}
}

// NewKVMDriverStub creates a stub KVM driver
func NewKVMDriverStub(config map[string]interface{}) (VMDriver, error) {
	return NewContainerdDriver(config)
}

// NewContainerDriverStub creates a stub container driver
func NewContainerDriverStub(config map[string]interface{}) (VMDriver, error) {
	return NewContainerdDriver(config)
}

// NewProcessDriverStub creates a stub process driver
func NewProcessDriverStub(config map[string]interface{}) (VMDriver, error) {
	return NewContainerdDriver(config)
}

// updateLoop runs the VM update loop
func (m *VMManager) updateLoop() {
	// Placeholder for update loop implementation
	log.Println("VM Manager update loop started")
}

// cleanupLoop runs the VM cleanup loop
func (m *VMManager) cleanupLoop() {
	// Placeholder for cleanup loop implementation
	log.Println("VM Manager cleanup loop started")
}

// AddVM adds a VM to the manager
func (m *VMManager) AddVM(vm *VM) {
	m.vmsMutex.Lock()
	defer m.vmsMutex.Unlock()
	config := vm.Config()
	m.vms[config.ID] = vm
}

// RemoveVM removes a VM from the manager
func (m *VMManager) RemoveVM(vmID string) {
	m.vmsMutex.Lock()
	defer m.vmsMutex.Unlock()
	delete(m.vms, vmID)
}

// GetVMFromCache gets a VM by ID from cache
func (m *VMManager) GetVMFromCache(vmID string) (*VM, bool) {
	m.vmsMutex.RLock()
	defer m.vmsMutex.RUnlock()
	vm, exists := m.vms[vmID]
	return vm, exists
}

// Shutdown gracefully shuts down the VM manager
func (m *VMManager) Shutdown() {
	log.Println("Shutting down VM Manager...")

	// Cancel context to stop all operations
	m.cancel()
	log.Println("VM Manager shutdown complete")
}

// Missing methods expected by API packages

// UpdateVM updates a VM's configuration
func (m *VMManager) UpdateVM(ctx context.Context, vmID string, req UpdateVMRequest) (*VM, error) {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("VM %s not found: %w", vmID, err)
	}

	// Update VM configuration
	vm.mutex.Lock()
	defer vm.mutex.Unlock()

	// Update name if provided
	if req.Name != nil {
		vm.config.Name = *req.Name
	}

	// Update CPU if provided
	if req.CPU != nil {
		vm.config.CPUShares = *req.CPU
	}

	// Update memory if provided
	if req.Memory != nil {
		vm.config.MemoryMB = int(*req.Memory)
	}

	// Update disk if provided
	if req.Disk != nil {
		vm.config.DiskSizeGB = int(*req.Disk)
	}

	// Emit update event
	m.emitEvent(VMEvent{
		Type:      VMEventUpdated,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
	})

	return vm, nil
}

// MigrateVM migrates a VM to another node
func (m *VMManager) MigrateVM(ctx context.Context, vmID string, targetNode string, live bool) (*MigrationResult, error) {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("VM %s not found: %w", vmID, err)
	}

	// Check if VM is in a migratable state
	if vm.State() != StateRunning && vm.State() != StateStopped {
		return nil, fmt.Errorf("VM %s is in state %s, cannot migrate", vmID, vm.State())
	}

	// Migration ID would be generated by the driver if needed

	// Create migration result using the structure from predictive_prefetching.go
	result := &MigrationResult{
		Success:         false, // Will be set to true when complete
		Duration:        0,     // Will be calculated
		DataTransferred: 0,     // Will be updated by driver
		CacheHits:       0,
		CacheMisses:     0,
	}

	// Update VM state to migrating
	vm.mutex.Lock()
	vm.state = StateMigrating
	vm.mutex.Unlock()

	// Emit migration started event
	m.emitEvent(VMEvent{
		Type:      VMEventMigrating,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
		Message:   fmt.Sprintf("Migration started to node %s", targetNode),
	})

	// Start migration asynchronously
	go func() {
		startTime := time.Now()
		if err := m.performMigration(ctx, vm, targetNode, live); err != nil {
			log.Printf("Migration failed for VM %s: %v", vmID, err)
			result.Success = false
			result.Duration = time.Since(startTime)
			// Revert VM state
			vm.mutex.Lock()
			vm.state = StateRunning
			vm.mutex.Unlock()
		} else {
			result.Success = true
			result.Duration = time.Since(startTime)
			// Update VM node ID
			vm.mutex.Lock()
			vm.state = StateRunning
			// vm.nodeID = targetNode // Would set if VM has nodeID field
			vm.mutex.Unlock()
		}
	}()

	return result, nil
}

// CreateSnapshot creates a snapshot of a VM
func (m *VMManager) CreateSnapshot(ctx context.Context, vmID string, name string, description string) (*SnapshotResult, error) {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("VM %s not found: %w", vmID, err)
	}

	// Generate snapshot ID
	snapshotID := uuid.New().String()

	// Create snapshot result
	result := &SnapshotResult{
		SnapshotID:  snapshotID,
		VMID:        vmID,
		Name:        name,
		Description: description,
		Status:      "creating",
		CreatedAt:   time.Now(),
	}

	// Emit snapshot started event
	m.emitEvent(VMEvent{
		Type:      VMEventSnapshot,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
		Message:   fmt.Sprintf("Snapshot %s creation started", name),
	})

	// Create snapshot asynchronously
	go func() {
		if err := m.performSnapshot(ctx, vm, snapshotID, name, description); err != nil {
			log.Printf("Snapshot creation failed for VM %s: %v", vmID, err)
			result.Status = "failed"
			result.Error = err.Error()
		} else {
			result.Status = "completed"
			result.CompletedAt = time.Now()
		}
	}()

	return result, nil
}

// ListMigrations returns list of VM migrations
func (m *VMManager) ListMigrations(ctx context.Context) ([]*Migration, error) {
	// Stub implementation - return empty list for now
	return []*Migration{}, nil
}

// GetVMStatus returns status of a VM
func (m *VMManager) GetVMStatus(ctx context.Context, vmID string) (*VMStatus, error) {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return nil, err
	}

	return &VMStatus{
		VMID:   vmID,
		Status: string(vm.State()),
		CPU:    vm.config.CPUShares,
		Memory: int64(vm.config.MemoryMB),
	}, nil
}

// GetVMMetrics returns metrics for a VM
func (m *VMManager) GetVMMetrics(ctx context.Context, vmID string) (*VMMetrics, error) {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return nil, err
	}

	// Get current metrics - this is a placeholder implementation
	return &VMMetrics{
		VMID:      vmID,
		NodeID:    vm.NodeID(),
		Timestamp: time.Now(),
		CPU:       CPUMetrics{UsagePercent: 50.0}, // 50% placeholder
		Memory:    MemoryMetrics{UsedBytes: int64(vm.config.MemoryMB * 1024 * 1024 / 2)}, // 50% of allocated memory
		Disk:      make(map[string]DiskMetrics),
		Network:   map[string]NetMetrics{
			"eth0": {RxBytes: 1024 * 1024, TxBytes: 512 * 1024}, // Placeholder data
		},
	}, nil
}

// DefaultVMManagerConfig returns a default VM manager configuration
func DefaultVMManagerConfig() VMManagerConfig {
	return VMManagerConfig{
		DefaultDriver:   VMTypeKVM,
		Drivers:         make(map[VMType]VMDriverConfigManager),
		UpdateInterval:  30 * time.Second,
		CleanupInterval: 5 * time.Minute,
		DefaultVMType:   VMTypeKVM,
		RetentionPeriod: 24 * time.Hour,
		Scheduler: VMSchedulerConfig{
			Type:   "round-robin",
			Config: make(map[string]interface{}),
		},
	}
}

// Migration represents a VM migration
type Migration struct {
	ID         string `json:"id"`
	VMID       string `json:"vm_id"`
	SourceNode string `json:"source_node"`
	TargetNode string `json:"target_node"`
	Status     string `json:"status"`
	Progress   int    `json:"progress"`
}

// VMStatus represents VM status information
type VMStatus struct {
	VMID   string `json:"vm_id"`
	Status string `json:"status"`
	CPU    int    `json:"cpu"`
	Memory int64  `json:"memory"`
}

// UpdateVMRequest represents a request to update VM configuration
type UpdateVMRequest struct {
	Name   *string `json:"name,omitempty"`
	CPU    *int    `json:"cpu,omitempty"`
	Memory *int64  `json:"memory,omitempty"`
	Disk   *int64  `json:"disk,omitempty"`
}


// SnapshotResult represents the result of a VM snapshot
type SnapshotResult struct {
	SnapshotID  string    `json:"snapshot_id"`
	VMID        string    `json:"vm_id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	CompletedAt time.Time `json:"completed_at,omitempty"`
	Size        int64     `json:"size,omitempty"`
	Error       string    `json:"error,omitempty"`
}

// performMigration performs the actual migration logic
func (m *VMManager) performMigration(ctx context.Context, vm *VM, targetNode string, live bool) error {
	// Get driver for the VM
	driver, err := m.getDriver(vm.config)
	if err != nil {
		return fmt.Errorf("failed to get VM driver: %w", err)
	}

	// Check if driver supports migration
	if !driver.SupportsMigrate() {
		return fmt.Errorf("VM driver does not support migration")
	}

	// Prepare migration parameters
	params := map[string]string{
		"target_node":     targetNode,
		"migration_type":  "live",
	}
	if !live {
		params["migration_type"] = "offline"
	}

	// Perform migration via driver
	err = driver.Migrate(ctx, vm.ID(), targetNode, params)
	if err != nil {
		return fmt.Errorf("migration failed: %w", err)
	}

	// Emit migration completed event
	m.emitEvent(VMEvent{
		Type:      VMEventMigrated,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    targetNode,
		Message:   fmt.Sprintf("VM migrated successfully to node %s", targetNode),
	})

	return nil
}

// performSnapshot performs the actual snapshot creation
func (m *VMManager) performSnapshot(ctx context.Context, vm *VM, snapshotID, name, description string) error {
	// Get driver for the VM
	driver, err := m.getDriver(vm.config)
	if err != nil {
		return fmt.Errorf("failed to get VM driver: %w", err)
	}

	// Check if driver supports snapshots
	if !driver.SupportsSnapshot() {
		return fmt.Errorf("VM driver does not support snapshots")
	}

	// Prepare snapshot parameters
	params := map[string]string{
		"name":        name,
		"description": description,
		"snapshot_id": snapshotID,
	}

	// Create snapshot via driver
	_, err = driver.Snapshot(ctx, vm.ID(), name, params)
	if err != nil {
		return fmt.Errorf("snapshot creation failed: %w", err)
	}

	// Emit snapshot completed event
	m.emitEvent(VMEvent{
		Type:      VMEventSnapshot,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
		Message:   fmt.Sprintf("Snapshot %s created successfully", name),
	})

	return nil
}

// VMRequirements represents resource requirements for a VM
type VMRequirements struct {
	CPU    int `json:"cpu"`
	Memory int `json:"memory"`
	Disk   int `json:"disk"`
}

// selectNodeForVM selects the best node for a VM based on requirements
func (m *VMManager) selectNodeForVM(spec *VMRequirements) (string, error) {
	// If we have a scheduler configured, use it
	if len(m.scheduler.nodes) > 0 {
		// Create a temporary VM for scheduling
		tempVM := &VM{
			config: VMConfig{
				CPUShares:  spec.CPU,
				MemoryMB:   spec.Memory,
				DiskSizeGB: spec.Disk,
			},
		}
		ctx := context.Background()
		return m.scheduler.ScheduleVM(ctx, tempVM)
	}

	// Fallback to default node selection logic
	availableNodes := []string{"node-1", "node-2", "node-3"}
	
	// Simple round-robin selection for now
	// In production, this would check actual resource availability
	for _, nodeID := range availableNodes {
		// Check if node has enough resources (simplified)
		log.Printf("Checking node %s for VM requirements: CPU=%d, Memory=%d, Disk=%d",
			nodeID, spec.CPU, spec.Memory, spec.Disk)
		
		// For now, assume first node always has capacity
		return nodeID, nil
	}
	
	return "", errors.New("no available nodes found")
}

