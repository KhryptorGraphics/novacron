package vm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"
)

// VMManager manages virtual machines across different drivers
type VMManager struct {
	drivers        map[VMType]VMDriver
	vms            map[string]*VM
	vmsMutex       sync.RWMutex
	driverFactory  VMDriverFactory
	scheduler      *VMScheduler
	eventListeners []VMManagerEventListener
	eventMutex     sync.RWMutex
	vmCache        map[string]VMInfo
	mutex          sync.RWMutex
	ctx            context.Context
	cancel         context.CancelFunc
	
	// Resource accounting fields
	allocatedCPU      int
	allocatedMemoryMB int64
	resourceMutex     sync.RWMutex
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
		manager.scheduler = scheduler
	}

	return manager, nil
}

// createDriver creates a driver instance based on type and config  
// Delegates to createDriverForType to avoid duplication

// createScheduler creates a scheduler instance based on config
func createScheduler(config VMSchedulerConfig) (*VMScheduler, error) {
	// For now, return a basic scheduler
	// This can be expanded to support different scheduler types
	scheduler := NewVMScheduler(SchedulerConfig{
		Policy: SchedulerPolicyRoundRobin,
	})
	return scheduler, nil
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
		return nil, fmt.Errorf("VM %s not found: %w", vmID, ErrVMNotFound)
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
		// Create config with correct Type field for driver check
		config := VMConfig{Type: vmType}
		// Check if driver is available for this specific type
		if m.checkDriverAvailability(config) == nil {
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
	return nil, fmt.Errorf("KVM driver not yet implemented")
}

// NewContainerDriverStub creates a stub container driver
func NewContainerDriverStub(config map[string]interface{}) (VMDriver, error) {
	return nil, fmt.Errorf("Container driver not yet implemented")
}

// NewProcessDriverStub creates a stub process driver
func NewProcessDriverStub(config map[string]interface{}) (VMDriver, error) {
	return nil, fmt.Errorf("Process driver not yet implemented")
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

// ListMigrations returns list of VM migrations
func (m *VMManager) ListMigrations(ctx context.Context) ([]*Migration, error) {
	// Stub implementation - return empty list for now
	return []*Migration{}, nil
}

// GetVMStatus returns status of a VM
func (m *VMManager) GetVMStatus(ctx context.Context, vmID string) (*VMStatus, error) {
	// Stub implementation
	return &VMStatus{
		VMID:   vmID,
		Status: "unknown",
	}, nil
}

// GetVMMetrics returns metrics for a VM
func (m *VMManager) GetVMMetrics(ctx context.Context, vmID string) (*VMMetrics, error) {
	// Get the VM first to ensure it exists
	vm, err := m.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("VM not found: %w", err)
	}
	
	// Get the driver for this VM
	driver, err := m.getDriver(vm.Config())
	if err != nil {
		return nil, fmt.Errorf("failed to get driver for metrics: %w", err)
	}
	
	// Call driver's GetMetrics which returns *VMInfo
	vmInfo, err := driver.GetMetrics(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get metrics: %w", err)
	}
	
	// Convert VMInfo to VMMetrics - populate available fields
	metrics := &VMMetrics{
		VMID:      vmID,
		NodeID:    vm.NodeID(),
		Timestamp: time.Now(),
		CPU: CPUMetrics{
			Cores:        vm.config.CPUShares,
			UsagePercent: 0.0, // Placeholder - containerd doesn't provide this yet
		},
		Memory: MemoryMetrics{
			UsedBytes:  int64(vm.config.MemoryMB) * 1024 * 1024, // Convert MB to bytes
			TotalBytes: int64(vm.config.MemoryMB) * 1024 * 1024,
			UsagePercent: 0.0, // Placeholder
		},
		Disk:    make(map[string]DiskMetrics),
		Network: make(map[string]NetMetrics),
	}
	
	// Add VMInfo state if available
	if vmInfo != nil {
		metrics.Labels = map[string]string{
			"state": string(vmInfo.State),
			"image": vmInfo.Image,
		}
	}
	
	return metrics, nil
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

// Note: VMMetrics is defined in vm_metrics.go to avoid duplication

// UpdateVM updates VM configuration for stopped VMs
func (m *VMManager) UpdateVM(ctx context.Context, vmID string, updateSpec VMUpdateSpec) error {
	// Get VM
	vm, err := m.GetVM(vmID)
	if err != nil {
		return &VMError{Code: "VM_NOT_FOUND", Message: fmt.Sprintf("VM %s not found", vmID), Cause: ErrVMNotFound}
	}

	// Check VM state - only allow updates on stopped VMs
	if vm.State() != StateStopped {
		return &VMError{
			Code:    "INVALID_STATE", 
			Message: fmt.Sprintf("VM %s is in state %s, can only update stopped VMs", vmID, vm.State()),
			Cause:   ErrInvalidVMState,
		}
	}

	// Get current config for resource accounting
	oldConfig := vm.Config()
	oldCPU := oldConfig.CPUShares
	oldMemory := int64(oldConfig.MemoryMB)
	
	// Input validation before applying updates
	if updateSpec.CPU != nil && *updateSpec.CPU <= 0 {
		return &VMError{Code: "INVALID_ARGUMENT", Message: "CPU must be > 0", Cause: fmt.Errorf("invalid argument")}
	}
	if updateSpec.Memory != nil && *updateSpec.Memory <= 0 {
		return &VMError{Code: "INVALID_ARGUMENT", Message: "Memory must be > 0", Cause: fmt.Errorf("invalid argument")}
	}
	if updateSpec.Disk != nil {
		newDiskSize := int(*updateSpec.Disk)
		if newDiskSize < oldConfig.DiskSizeGB {
			return &VMError{Code: "INVALID_ARGUMENT", Message: "Disk shrinking not allowed", Cause: fmt.Errorf("invalid argument")}
		}
		if newDiskSize <= 0 {
			return &VMError{Code: "INVALID_ARGUMENT", Message: "Disk size must be positive", Cause: fmt.Errorf("invalid argument")}
		}
		if newDiskSize > 10240 { // 10TB maximum disk size limit
			return &VMError{Code: "INVALID_ARGUMENT", Message: "Disk size exceeds maximum limit (10TB)", Cause: fmt.Errorf("invalid argument")}
		}
		// Ensure disk expansion is reasonable (no more than 100x increase at once)
		if newDiskSize > oldConfig.DiskSizeGB*100 {
			return &VMError{Code: "INVALID_ARGUMENT", Message: "Disk size increase too large (max 100x current size)", Cause: fmt.Errorf("invalid argument")}
		}
	}

	// Name validation - check for duplicate names if name is being updated
	if updateSpec.Name != nil {
		newName := *updateSpec.Name
		if newName == "" {
			return &VMError{Code: "INVALID_ARGUMENT", Message: "VM name cannot be empty", Cause: fmt.Errorf("invalid argument")}
		}
		
		// Check if another VM already has this name
		m.vmsMutex.RLock()
		for id, existingVM := range m.vms {
			if id != vmID && existingVM.Name() == newName {
				m.vmsMutex.RUnlock()
				return &VMError{
					Code:    "NAME_CONFLICT", 
					Message: fmt.Sprintf("VM name '%s' is already in use by VM %s", newName, id),
					Cause:   fmt.Errorf("duplicate name"),
				}
			}
		}
		m.vmsMutex.RUnlock()
	}

	// Apply updates using encapsulated method
	err = vm.ApplyUpdateSpec(updateSpec)
	if err != nil {
		return &VMError{Code: "UPDATE_FAILED", Message: fmt.Sprintf("Failed to apply update: %v", err), Cause: err}
	}

	// Get new config for resource accounting
	newConfig := vm.Config()
	newCPU := newConfig.CPUShares
	newMemory := int64(newConfig.MemoryMB)

	// Update resource counters with clamping
	m.resourceMutex.Lock()
	if updateSpec.CPU != nil {
		deltaCPU := newCPU - oldCPU
		m.allocatedCPU += deltaCPU
		// Clamp to prevent negative totals
		if m.allocatedCPU < 0 {
			m.allocatedCPU = 0
		}
	}
	if updateSpec.Memory != nil {
		deltaMemory := newMemory - oldMemory
		m.allocatedMemoryMB += deltaMemory
		// Clamp to prevent negative totals
		if m.allocatedMemoryMB < 0 {
			m.allocatedMemoryMB = 0
		}
	}
	m.resourceMutex.Unlock()

	// Emit update event
	m.emitEvent(VMEvent{
		Type:      VMEventUpdated,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
		Message:   "VM configuration updated",
	})

	log.Printf("Updated VM %s configuration", vmID)
	return nil
}

// MigrateVM migrates a VM to another node
func (m *VMManager) MigrateVM(ctx context.Context, vmID string, targetNode string, options map[string]string) error {
	// Get VM
	vm, err := m.GetVM(vmID)
	if err != nil {
		return &VMError{Code: "VM_NOT_FOUND", Message: fmt.Sprintf("VM %s not found", vmID), Cause: ErrVMNotFound}
	}

	// Get driver for VM
	driver, err := m.getDriverForVM(vm)
	if err != nil {
		return &VMError{Code: "DRIVER_ERROR", Message: "Failed to get VM driver", Cause: err}
	}

	// Check if driver supports migration
	if !driver.SupportsMigrate() {
		return &VMError{
			Code:    "OPERATION_NOT_SUPPORTED", 
			Message: "VM driver does not support migration",
			Cause:   ErrOperationNotSupported,
		}
	}

	// Convert options to driver format
	params := make(map[string]string)
	params["target_node"] = targetNode
	for k, v := range options {
		params[k] = v
	}

	// Call internal migration method
	response, err := m.migrateVM(ctx, vm, driver, params)
	if err != nil {
		return err
	}

	if !response.Success {
		return &VMError{
			Code:    "MIGRATION_FAILED", 
			Message: response.ErrorMessage,
		}
	}

	return nil
}

// CreateSnapshot creates a VM snapshot
func (m *VMManager) CreateSnapshot(ctx context.Context, vmID string, snapshotName string, options map[string]string) (string, error) {
	// Get VM
	vm, err := m.GetVM(vmID)
	if err != nil {
		return "", &VMError{Code: "VM_NOT_FOUND", Message: fmt.Sprintf("VM %s not found", vmID), Cause: ErrVMNotFound}
	}

	// Get driver for VM
	driver, err := m.getDriverForVM(vm)
	if err != nil {
		return "", &VMError{Code: "DRIVER_ERROR", Message: "Failed to get VM driver", Cause: err}
	}

	// Check if driver supports snapshots
	if !driver.SupportsSnapshot() {
		return "", &VMError{
			Code:    "OPERATION_NOT_SUPPORTED", 
			Message: "VM driver does not support snapshots",
			Cause:   ErrOperationNotSupported,
		}
	}

	// Prepare parameters
	params := make(map[string]string)
	params["name"] = snapshotName
	for k, v := range options {
		params[k] = v
	}

	// Call internal snapshot method
	response, err := m.snapshotVM(ctx, vm, driver, params)
	if err != nil {
		return "", err
	}

	if !response.Success {
		return "", &VMError{
			Code:    "SNAPSHOT_FAILED", 
			Message: response.ErrorMessage,
		}
	}

	// Extract snapshot ID from response data
	if snapshotID, ok := response.Data["snapshot_id"]; ok {
		return snapshotID, nil
	}

	return snapshotName, nil
}

// Helper method to get driver for VM (already exists but ensuring it's available)
func (m *VMManager) getDriverForVM(vm *VM) (VMDriver, error) {
	return m.getDriver(vm.config)
}

// GetCurrentAllocations returns the current resource allocations
func (m *VMManager) GetCurrentAllocations() (cpu int, memoryMB int64) {
	m.resourceMutex.RLock()
	allocatedCPU := m.allocatedCPU
	allocatedMemoryMB := m.allocatedMemoryMB
	m.resourceMutex.RUnlock()
	
	// Add safety checks to prevent negative totals - clamp locally
	if allocatedCPU < 0 {
		log.Printf("Warning: Negative CPU allocation detected: %d, clamping to 0", allocatedCPU)
		allocatedCPU = 0
	}
	if allocatedMemoryMB < 0 {
		log.Printf("Warning: Negative memory allocation detected: %d, clamping to 0", allocatedMemoryMB)
		allocatedMemoryMB = 0
	}
	
	return allocatedCPU, allocatedMemoryMB
}
