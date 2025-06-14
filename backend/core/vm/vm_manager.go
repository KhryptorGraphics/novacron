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
	drivers       map[VMType]VMDriver
	scheduler     VMScheduler
	eventListeners []VMManagerEventListener
	eventMutex    sync.RWMutex
	vmCache       map[string]VMInfo
	mutex         sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
}

// VMManagerConfig contains configuration for the VM manager
type VMManagerConfig struct {
	DefaultDriver VMType
	Drivers       map[VMType]VMDriverConfig
	Scheduler     VMSchedulerConfig
}

// VMDriverConfigLegacy contains legacy driver-specific configuration (use driver_factory.go VMDriverConfig instead)
type VMDriverConfigLegacy struct {
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

// VMEvent, VMEventType and related constants are defined in vm_events.go to avoid duplication

// NewVMManager creates a new VM manager instance
func NewVMManager(config VMManagerConfig) (*VMManager, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	manager := &VMManager{
		drivers:        make(map[VMType]VMDriver),
		eventListeners: make([]VMManagerEventListener, 0),
		vmCache:        make(map[string]VMInfo),
		ctx:            ctx,
		cancel:         cancel,
	}

	// Initialize drivers based on config
	for driverType, driverConfig := range config.Drivers {
		if !driverConfig.Enabled {
			continue
		}

		driver, err := createDriver(driverType, driverConfig.Config)
		if err != nil {
			log.Printf("Failed to create driver %s: %v", driverType, err)
			continue
		}

		manager.drivers[driverType] = driver
		log.Printf("Initialized driver: %s", driverType)
	}

	// Initialize scheduler if configured
	if config.Scheduler.Type != "" {
		scheduler, err := createScheduler(config.Scheduler)
		if err != nil {
			log.Printf("Failed to create scheduler: %v", err)
		} else {
			manager.scheduler = scheduler
		}
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
		return NewKVMDriver(uri)
	case VMTypeContainer:
		return NewContainerDriver(config)
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
	return NewBasicScheduler(), nil
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

