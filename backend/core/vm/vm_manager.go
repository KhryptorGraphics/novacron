package vm

import (
	"errors"
	"fmt"
	"log"
	"time"
)

// NOTE: VMManager struct definition, NewVMManager function, VMManagerConfig,
// VMDriverFactory, VMManagerEventListener are likely defined in vm_types.go or another central file.
// Removing duplicate definitions from here.

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

// emitEvent emits an event to all registered listeners
func (m *VMManager) emitEvent(event VMEvent) { // Assuming VMEvent is defined elsewhere
	m.eventMutex.RLock()
	defer m.eventMutex.RUnlock()

	// Make a copy of the listeners to avoid blocking during event processing
	listeners := make([]VMManagerEventListener, len(m.eventListeners))
	copy(listeners, m.eventListeners)

	// Process events asynchronously
	for _, listener := range listeners {
		go func(l VMManagerEventListener, e VMEvent) {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Recovered from panic in event listener: %v", r)
				}
			}()

			l(e)
		}(listener, event)
	}

	// Log the event
	// Corrected: Use methods to access VM fields within the event
	log.Printf("VM event: type=%s, vm=%s, node=%s, message=%s",
		event.Type, event.VM.ID(), event.NodeID, event.Message) // Use VM.ID()
}

// updateLoop periodically updates the status of VMs
func (m *VMManager) updateLoop() {
	ticker := time.NewTicker(m.config.UpdateInterval) // Assuming config is part of VMManager struct
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done(): // Assuming ctx is part of VMManager struct
			return
		case <-ticker.C:
			m.updateVMs()
		}
	}
}

// cleanupLoop periodically cleans up expired VMs
func (m *VMManager) cleanupLoop() {
	ticker := time.NewTicker(m.config.CleanupInterval) // Assuming config is part of VMManager struct
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.cleanupVMs()
		}
	}
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

// --- Internal methods to be implemented ---

func (m *VMManager) updateVMs() {
	// TODO: Implement logic to update VM statuses and metrics
	log.Println("VMManager: Running updateVMs loop (implementation pending)")
}

func (m *VMManager) cleanupVMs() {
	// TODO: Implement logic to clean up VMs marked for deletion or expired
	log.Println("VMManager: Running cleanupVMs loop (implementation pending)")
}

// Note: Actual VMManager struct and NewVMManager function should be defined in vm_types.go
// The methods below are defined here but operate on the VMManager type assumed to be defined elsewhere.
