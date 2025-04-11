package vm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
)

// VMManager manages virtual machines
type VMManager struct {
	config         VMManagerConfig
	vms            map[string]*VM
	vmsMutex       sync.RWMutex
	scheduler      *scheduler.Scheduler
	driverFactory  VMDriverFactory
	eventListeners []VMManagerEventListener
	eventMutex     sync.RWMutex
	ctx            context.Context
	cancel         context.CancelFunc
}

// NewVMManager creates a new VM manager
func NewVMManager(config VMManagerConfig, sch *scheduler.Scheduler, driverFactory VMDriverFactory) *VMManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &VMManager{
		config:        config,
		vms:           make(map[string]*VM),
		scheduler:     sch,
		driverFactory: driverFactory,
		ctx:           ctx,
		cancel:        cancel,
	}
}

// Start starts the VM manager
func (m *VMManager) Start() error {
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

	m.cancel()

	return nil
}

// AddEventListener adds an event listener
func (m *VMManager) AddEventListener(listener VMManagerEventListener) {
	m.eventMutex.Lock()
	defer m.eventMutex.Unlock()

	m.eventListeners = append(m.eventListeners, listener)
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
func (m *VMManager) GetVM(vmID string) (*VM, error) {
	m.vmsMutex.RLock()
	defer m.vmsMutex.RUnlock()

	vm, exists := m.vms[vmID]
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
func (m *VMManager) ListVMsByState(state VMState) []*VM {
	m.vmsMutex.RLock()
	defer m.vmsMutex.RUnlock()

	result := make([]*VM, 0)
	for _, vm := range m.vms {
		if vm.State == state {
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
		if vm.NodeID == nodeID {
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
		result[vm.State]++
	}

	return result
}

// emitEvent emits an event to all registered listeners
func (m *VMManager) emitEvent(event VMEvent) {
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
	log.Printf("VM event: type=%s, vm=%s, node=%s, message=%s",
		event.Type, event.VM.ID, event.NodeID, event.Message)
}

// updateLoop periodically updates the status of VMs
func (m *VMManager) updateLoop() {
	ticker := time.NewTicker(m.config.UpdateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.updateVMs()
		}
	}
}

// cleanupLoop periodically cleans up expired VMs
func (m *VMManager) cleanupLoop() {
	ticker := time.NewTicker(m.config.CleanupInterval)
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
func (m *VMManager) getDriver(vmType VMType) (VMDriver, error) {
	if m.driverFactory == nil {
		return nil, errors.New("no driver factory configured")
	}

	return m.driverFactory(vmType)
}

// checkDriverAvailability checks if a driver is available for a VM type
func (m *VMManager) checkDriverAvailability(vmType VMType) error {
	_, err := m.getDriver(vmType)
	return err
}

// listAvailableVMTypes lists all available VM types
func (m *VMManager) listAvailableVMTypes() []VMType {
	// Define the standard VM types to check
	standardTypes := []VMType{
		VMTypeContainer,
		VMTypeKVM,
		VMTypeProcess,
	}

	// Check which ones have working drivers
	availableTypes := make([]VMType, 0)
	for _, vmType := range standardTypes {
		if m.checkDriverAvailability(vmType) == nil {
			availableTypes = append(availableTypes, vmType)
		}
	}

	return availableTypes
}
