package vm

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// VMManagerFixed is a cleaned up version of VMManager with proper VM lifecycle operations
type VMManagerFixed struct {
	drivers        map[VMType]VMDriver
	vms            map[string]*VM
	vmCache        map[string]VMInfo
	eventListeners []VMManagerEventListener
	
	mutex      sync.RWMutex
	eventMutex sync.RWMutex
	
	ctx    context.Context
	cancel context.CancelFunc
	
	config VMManagerConfig
	nodeID string
}

// NewVMManagerFixed creates a new fixed VM manager
func NewVMManagerFixed(config VMManagerConfig, nodeID string) (*VMManagerFixed, error) {
	ctx, cancel := context.WithCancel(context.Background())
	
	manager := &VMManagerFixed{
		drivers:        make(map[VMType]VMDriver),
		vms:            make(map[string]*VM),
		vmCache:        make(map[string]VMInfo),
		eventListeners: make([]VMManagerEventListener, 0),
		ctx:            ctx,
		cancel:         cancel,
		config:         config,
		nodeID:         nodeID,
	}

	return manager, nil
}

// Start starts the VM manager
func (m *VMManagerFixed) Start() error {
	log.Println("Starting VM manager")

	// Start background processes
	go m.updateLoop()
	go m.cleanupLoop()

	return nil
}

// Stop stops the VM manager
func (m *VMManagerFixed) Stop() error {
	log.Println("Stopping VM manager")
	m.cancel()
	return nil
}

// RegisterDriver registers a VM driver
func (m *VMManagerFixed) RegisterDriver(vmType VMType, driver VMDriver) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	
	m.drivers[vmType] = driver
	log.Printf("Registered VM driver: %s", vmType)
}

// CreateVM creates a new VM
func (m *VMManagerFixed) CreateVM(ctx context.Context, config VMConfig) (*VM, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	// Create the VM instance
	vm, err := NewVM(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create VM: %w", err)
	}

	// Set node information
	vm.SetNodeID(m.nodeID)
	
	// Store the VM
	m.vms[vm.ID()] = vm
	m.vmCache[vm.ID()] = vm.GetInfo()

	// Emit creation event
	m.emitEvent(VMEvent{
		Type:      VMEventCreated,
		VMID:      vm.ID(),
		VMName:    vm.Name(),
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"node_id": m.nodeID,
		},
	})

	log.Printf("Created VM %s (%s)", vm.Name(), vm.ID())
	return vm, nil
}

// StartVM starts a VM
func (m *VMManagerFixed) StartVM(ctx context.Context, vmID string) error {
	m.mutex.RLock()
	vm, exists := m.vms[vmID]
	m.mutex.RUnlock()

	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	// Start the VM
	if err := vm.Start(); err != nil {
		m.emitEvent(VMEvent{
			Type:      VMEventFailed,
			VMID:      vmID,
			VMName:    vm.Name(),
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"operation": "start",
				"error":     err.Error(),
			},
		})
		return fmt.Errorf("failed to start VM: %w", err)
	}

	// Update cache
	m.mutex.Lock()
	m.vmCache[vmID] = vm.GetInfo()
	m.mutex.Unlock()

	// Emit started event
	m.emitEvent(VMEvent{
		Type:      VMEventStarted,
		VMID:      vmID,
		VMName:    vm.Name(),
		Timestamp: time.Now(),
	})

	return nil
}

// StopVM stops a VM
func (m *VMManagerFixed) StopVM(ctx context.Context, vmID string) error {
	m.mutex.RLock()
	vm, exists := m.vms[vmID]
	m.mutex.RUnlock()

	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	// Stop the VM
	if err := vm.Stop(); err != nil {
		m.emitEvent(VMEvent{
			Type:      VMEventFailed,
			VMID:      vmID,
			VMName:    vm.Name(),
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"operation": "stop",
				"error":     err.Error(),
			},
		})
		return fmt.Errorf("failed to stop VM: %w", err)
	}

	// Update cache
	m.mutex.Lock()
	m.vmCache[vmID] = vm.GetInfo()
	m.mutex.Unlock()

	// Emit stopped event
	m.emitEvent(VMEvent{
		Type:      VMEventStopped,
		VMID:      vmID,
		VMName:    vm.Name(),
		Timestamp: time.Now(),
	})

	return nil
}

// RebootVM reboots a VM
func (m *VMManagerFixed) RebootVM(ctx context.Context, vmID string) error {
	m.mutex.RLock()
	vm, exists := m.vms[vmID]
	m.mutex.RUnlock()

	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	// Stop and start the VM
	vm.SetState(StateRestarting)
	
	if err := vm.Stop(); err != nil {
		return fmt.Errorf("failed to stop VM for reboot: %w", err)
	}

	// Wait a moment before starting
	time.Sleep(1 * time.Second)

	if err := vm.Start(); err != nil {
		return fmt.Errorf("failed to start VM after reboot: %w", err)
	}

	// Update cache
	m.mutex.Lock()
	m.vmCache[vmID] = vm.GetInfo()
	m.mutex.Unlock()

	// Emit restarted event
	m.emitEvent(VMEvent{
		Type:      VMEventRestarted,
		VMID:      vmID,
		VMName:    vm.Name(),
		Timestamp: time.Now(),
	})

	return nil
}

// PauseVM pauses a VM (if supported)
func (m *VMManagerFixed) PauseVM(ctx context.Context, vmID string) error {
	m.mutex.RLock()
	vm, exists := m.vms[vmID]
	m.mutex.RUnlock()

	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vm.State() != StateRunning {
		return fmt.Errorf("VM %s is not running", vmID)
	}

	// For now, implement basic pause by changing state
	// In a full implementation, this would send SIGSTOP to the process
	vm.SetState(StatePaused)

	// Update cache
	m.mutex.Lock()
	m.vmCache[vmID] = vm.GetInfo()
	m.mutex.Unlock()

	// Emit paused event
	m.emitEvent(VMEvent{
		Type:      VMEventPaused,
		VMID:      vmID,
		VMName:    vm.Name(),
		Timestamp: time.Now(),
	})

	log.Printf("Paused VM %s", vmID)
	return nil
}

// ResumeVM resumes a paused VM
func (m *VMManagerFixed) ResumeVM(ctx context.Context, vmID string) error {
	m.mutex.RLock()
	vm, exists := m.vms[vmID]
	m.mutex.RUnlock()

	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	if vm.State() != StatePaused {
		return fmt.Errorf("VM %s is not paused", vmID)
	}

	// For now, implement basic resume by changing state
	// In a full implementation, this would send SIGCONT to the process
	vm.SetState(StateRunning)

	// Update cache
	m.mutex.Lock()
	m.vmCache[vmID] = vm.GetInfo()
	m.mutex.Unlock()

	// Emit resumed event
	m.emitEvent(VMEvent{
		Type:      VMEventResumed,
		VMID:      vmID,
		VMName:    vm.Name(),
		Timestamp: time.Now(),
	})

	log.Printf("Resumed VM %s", vmID)
	return nil
}

// DeleteVM deletes a VM
func (m *VMManagerFixed) DeleteVM(ctx context.Context, vmID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return fmt.Errorf("VM %s not found", vmID)
	}

	// Stop VM if it's running
	if vm.State() == StateRunning {
		if err := vm.Stop(); err != nil {
			log.Printf("Warning: Failed to stop VM %s before deletion: %v", vmID, err)
		}
	}

	// Clean up VM resources
	if err := vm.Cleanup(); err != nil {
		log.Printf("Warning: Failed to cleanup VM %s resources: %v", vmID, err)
	}

	// Remove from maps
	delete(m.vms, vmID)
	delete(m.vmCache, vmID)

	// Emit deleted event
	m.emitEvent(VMEvent{
		Type:      VMEventDeleted,
		VMID:      vmID,
		VMName:    vm.Name(),
		Timestamp: time.Now(),
	})

	log.Printf("Deleted VM %s", vmID)
	return nil
}

// GetVM returns a VM by ID
func (m *VMManagerFixed) GetVM(vmID string) (*VM, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	vm, exists := m.vms[vmID]
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}

	return vm, nil
}

// ListVMs returns all VMs
func (m *VMManagerFixed) ListVMs() map[string]*VM {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	result := make(map[string]*VM, len(m.vms))
	for id, vm := range m.vms {
		result[id] = vm
	}

	return result
}

// ListVMsByState returns VMs filtered by state
func (m *VMManagerFixed) ListVMsByState(state State) []*VM {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	result := make([]*VM, 0)
	for _, vm := range m.vms {
		if vm.State() == state {
			result = append(result, vm)
		}
	}

	return result
}

// GetVMInfo returns VM information from cache
func (m *VMManagerFixed) GetVMInfo(vmID string) (VMInfo, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	info, exists := m.vmCache[vmID]
	if !exists {
		return VMInfo{}, fmt.Errorf("VM %s not found", vmID)
	}

	return info, nil
}

// AddEventListener adds an event listener
func (m *VMManagerFixed) AddEventListener(listener VMManagerEventListener) {
	m.eventMutex.Lock()
	defer m.eventMutex.Unlock()

	m.eventListeners = append(m.eventListeners, listener)
}

// RemoveEventListener removes an event listener
func (m *VMManagerFixed) RemoveEventListener(listener VMManagerEventListener) {
	m.eventMutex.Lock()
	defer m.eventMutex.Unlock()

	for i, l := range m.eventListeners {
		if fmt.Sprintf("%p", l) == fmt.Sprintf("%p", listener) {
			m.eventListeners = append(m.eventListeners[:i], m.eventListeners[i+1:]...)
			break
		}
	}
}

// emitEvent emits an event to all listeners
func (m *VMManagerFixed) emitEvent(event VMEvent) {
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
			l(event)
		}(listener)
	}
}

// updateLoop periodically updates VM status
func (m *VMManagerFixed) updateLoop() {
	ticker := time.NewTicker(m.config.UpdateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			log.Println("VMManager: Update loop stopped")
			return
		case <-ticker.C:
			m.updateVMs()
		}
	}
}

// cleanupLoop periodically cleans up failed/expired VMs
func (m *VMManagerFixed) cleanupLoop() {
	ticker := time.NewTicker(m.config.CleanupInterval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			log.Println("VMManager: Cleanup loop stopped")
			return
		case <-ticker.C:
			m.cleanupExpiredVMs()
		}
	}
}

// updateVMs updates the status of all VMs
func (m *VMManagerFixed) updateVMs() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	for vmID, vm := range m.vms {
		// Update cache with latest VM info
		m.vmCache[vmID] = vm.GetInfo()

		// Collect stats if VM is running
		if vm.State() == StateRunning {
			vm.collectStats()
		}
	}
}

// cleanupExpiredVMs cleans up VMs that have been in failed state for too long
func (m *VMManagerFixed) cleanupExpiredVMs() {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	now := time.Now()
	retentionPeriod := m.config.RetentionPeriod

	toDelete := make([]string, 0)
	
	for vmID, vm := range m.vms {
		info := vm.GetInfo()
		
		// Clean up VMs that have been stopped or failed for too long
		if (info.State == StateStopped || info.State == StateFailed) && 
		   info.StoppedAt != nil && 
		   now.Sub(*info.StoppedAt) > retentionPeriod {
			toDelete = append(toDelete, vmID)
		}
	}

	// Delete expired VMs
	for _, vmID := range toDelete {
		vm := m.vms[vmID]
		log.Printf("Cleaning up expired VM %s", vmID)
		
		if err := vm.Cleanup(); err != nil {
			log.Printf("Failed to cleanup VM %s: %v", vmID, err)
		}
		
		delete(m.vms, vmID)
		delete(m.vmCache, vmID)
		
		m.emitEvent(VMEvent{
			Type:      VMEventDeleted,
			VMID:      vmID,
			VMName:    vm.Name(),
			Timestamp: time.Now(),
			Data: map[string]interface{}{
				"reason": "expired",
			},
		})
	}
}