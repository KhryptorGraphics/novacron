package vm

import (
	"context"
	"log"
	"sync"
	"time"
)

// updateVMs updates the status of all VMs
func (m *VMManager) updateVMs() {
	// Make a copy of the VMs to avoid holding the lock too long
	m.mutex.RLock()
	vmIDs := make([]string, 0, len(m.vmCache))
	for id := range m.vmCache {
		vmIDs = append(vmIDs, id)
	}
	m.mutex.RUnlock()
	
	// Update each VM in parallel with a limited number of workers
	const maxWorkers = 5
	wg := sync.WaitGroup{}
	semaphore := make(chan struct{}, maxWorkers)
	
	for _, id := range vmIDs {
		wg.Add(1)
		semaphore <- struct{}{}
		
		go func(vmID string) {
			defer func() {
				<-semaphore
				wg.Done()
			}()
			
			m.updateVMStatus(vmID)
		}(id)
	}
	
	wg.Wait()
}

// updateVMStatus updates the status of a single VM
func (m *VMManager) updateVMStatus(vmID string) {
	// Get the VM
	vm, err := m.GetVM(vmID)
	if err != nil {
		log.Printf("Failed to get VM %s: %v", vmID, err)
		return
	}
	
	// Skip VMs in transitional states
	if vm.State() == StateCreating || 
		vm.State() == StateDeleting || 
		vm.State() == StateRestarting ||
		vm.State() == StateMigrating {
		return
	}
	
	// Get the VM driver
	driver, err := m.getDriver(vm.config)
	if err != nil {
		log.Printf("Failed to get driver for VM %s: %v", vmID, err)
		return
	}
	
	// Get the current status
	status, err := driver.GetStatus(context.Background(), vmID)
	if err != nil {
		log.Printf("Failed to get status for VM %s: %v", vmID, err)
		
		// Mark as error if we can't get the status multiple times
		m.mutex.Lock()
		if vmInfo, exists := m.vmCache[vmID]; exists {
			vmInfo.ErrorMessage = "Failed to get VM status"
			m.vmCache[vmID] = vmInfo
		}
		m.mutex.Unlock()
		
		return
	}
	
	// Get additional info if available
	vmInfo, err := driver.GetInfo(context.Background(), vmID)
	if err != nil {
		log.Printf("Failed to get info for VM %s: %v", vmID, err)
	}
	
	// Update the VM state if it changed
	m.mutex.Lock()
	
	oldState := vm.State()
	if oldState != status {
		vm.SetState(status)
		
		// Update timestamps based on state changes
		if status == StateRunning && oldState != StateRunning {
			// Note: VM struct doesn't expose StartedAt directly, would need setter
			log.Printf("VM %s transitioned to running state", vmID)
		} else if status == StateStopped && oldState != StateStopped {
			// Note: VM struct doesn't expose StoppedAt directly, would need setter
			log.Printf("VM %s transitioned to stopped state", vmID)
		}
	}
	
	// Update VM cache with latest info
	if vmInfo != nil {
		// Update the cache with the latest VM info
		m.vmCache[vmID] = *vmInfo
	}
	
	// Reset error state if it's now running
	if status == StateRunning {
		if vmCacheInfo, exists := m.vmCache[vmID]; exists && vmCacheInfo.ErrorMessage != "" {
			vmCacheInfo.ErrorMessage = ""
			m.vmCache[vmID] = vmCacheInfo
		}
	}
	
	m.mutex.Unlock()
	
	// Emit event if state changed
	if oldState != status {
		m.emitEvent(VMEvent{
			Type:      VMEventUpdated,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID(),
			Message:   "VM state changed",
		})
		
		log.Printf("VM %s state changed from %s to %s", vm.ID(), oldState, status)
	}
}

// cleanupVMs cleans up expired and deleted VMs
func (m *VMManager) cleanupVMs() {
	now := time.Now()
	
	// Find VMs to clean up
	m.mutex.Lock()
	vmIDs := make([]string, 0)
	for vmID, vmInfo := range m.vmCache {
		if vmInfo.State == StateDeleting {
			vmIDs = append(vmIDs, vmID)
		}
	}
	m.mutex.Unlock()
	
	for _, vmID := range vmIDs {
		// Handle VMs in deleting state
		// Get VM info from cache
		m.mutex.RLock()
		vmInfo, exists := m.vmCache[vmID]
		m.mutex.RUnlock()
		
		if !exists {
			continue
		}
		
		// If it's been in deleting state for too long, remove it from the map
		if now.Sub(vmInfo.CreatedAt) > 10*time.Minute {
			m.mutex.Lock()
			delete(m.vmCache, vmID)
			m.mutex.Unlock()
			log.Printf("Removed deleted VM %s from memory", vmID)
		}
	}
	
	// Check for VMs with errors that need auto-recovery
	m.mutex.RLock()
	errorVMIDs := make([]string, 0)
	for vmID, vmInfo := range m.vmCache {
		if vmInfo.State == StateFailed {
			errorVMIDs = append(errorVMIDs, vmID)
		}
	}
	m.mutex.RUnlock()
	
	for _, vmID := range errorVMIDs {
		m.mutex.RLock()
		vmInfo, exists := m.vmCache[vmID]
		m.mutex.RUnlock()
		
		if exists {
			// Attempt to recover VMs that have been in error state for over 5 minutes
			if now.Sub(vmInfo.CreatedAt) > 5*time.Minute {
				go m.tryRecoverVM(vmID)
			}
		}
	}
	
	// Check for resource leaks - VMs in creating/restarting state for too long
	m.mutex.RLock()
	stuckVMIDs := make([]string, 0)
	for vmID, vmInfo := range m.vmCache {
		if (vmInfo.State == StateCreating || vmInfo.State == StateRestarting) && 
			now.Sub(vmInfo.CreatedAt) > 15*time.Minute {
			stuckVMIDs = append(stuckVMIDs, vmID)
		}
	}
	m.mutex.RUnlock()
	
	for _, vmID := range stuckVMIDs {
		// Mark as error
		m.mutex.Lock()
		if vmInfo, exists := m.vmCache[vmID]; exists {
			vmInfo.State = StateFailed
			vmInfo.ErrorMessage = "Operation timed out"
			m.vmCache[vmID] = vmInfo
		}
		m.mutex.Unlock()
		
		// Emit event for the timed out VM
		if vm, err := m.GetVM(vmID); err == nil {
			m.emitEvent(VMEvent{
				Type:      VMEventError,
				VM:        *vm,
				Timestamp: now,
				NodeID:    vm.NodeID(),
				Message:   "VM operation timed out",
			})
		}
		
		log.Printf("VM %s operation timed out, marked as error", vmID)
	}
}

// tryRecoverVM attempts to recover a VM from an error state
func (m *VMManager) tryRecoverVM(vmID string) {
	log.Printf("Attempting to recover VM %s", vmID)
	
	// Get the VM
	m.vmsMutex.RLock()
	vm, exists := m.vms[vmID]
	m.vmsMutex.RUnlock()
	
	if !exists || vm.State() != StateFailed {
		return
	}
	
	// Get the VM driver
	driver, err := m.driverFactory(vm.Config())
	if err != nil {
		log.Printf("Failed to get driver for VM %s during recovery: %v", vmID, err)
		return
	}
	
	// Try to restart the VM
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	
	// Try to stop then start
	_ = driver.Stop(ctx, vmID) // Ignore error, it might already be stopped
	
	// Try to start
	err = driver.Start(ctx, vmID)
	if err != nil {
		log.Printf("Failed to recover VM %s: %v", vmID, err)
		return
	}
	
	// Update the VM state
	m.vmsMutex.Lock()
	vm.SetState(StateRunning)
	vm.SetStartedAt(time.Now())
	m.vmsMutex.Unlock()
	
	// Emit recovered event
	m.emitEvent(VMEvent{
		Type:      VMEventStarted,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
		Message:   "VM recovered from error state",
	})
	
	log.Printf("Successfully recovered VM %s", vmID)
}

// checkResources checks if resource allocations match VM states
func (m *VMManager) checkResources() {
	// Get all active resource allocations
	allocations := m.scheduler.GetActiveAllocations()
	
	// Check each VM has a corresponding allocation
	m.vmsMutex.RLock()
	for id, vm := range m.vms {
		if vm.State() == StateDeleting {
			continue
		}
		
		if vm.ResourceID() == "" {
			log.Printf("Warning: VM %s has no resource ID", id)
			continue
		}
		
		found := false
		for _, alloc := range allocations {
			if alloc.RequestID == vm.ResourceID() {
				found = true
				break
			}
		}
		
		if !found {
			log.Printf("Warning: VM %s has resource ID %s but no active allocation", 
				id, vm.ResourceID())
		}
	}
	m.vmsMutex.RUnlock()
}

// RunHealthCheck performs a comprehensive health check of all VMs
func (m *VMManager) RunHealthCheck() map[string]string {
	results := make(map[string]string)
	
	// Make a copy of the VMs
	m.vmsMutex.RLock()
	vms := make(map[string]*VM, len(m.vms))
	for id, vm := range m.vms {
		vms[id] = vm
	}
	m.vmsMutex.RUnlock()
	
	// Check each VM
	for id, vm := range vms {
		if vm.State() == VMStateError {
			results[id] = "UNHEALTHY: VM in error state: " + vm.ErrorMessage()
		} else if vm.State() == VMStateCreating && time.Since(vm.CreatedAt()) > 10*time.Minute {
			results[id] = "WARNING: VM stuck in creating state"
		} else if vm.State() == VMStateDeleting && time.Since(vm.UpdatedAt()) > 10*time.Minute {
			results[id] = "WARNING: VM stuck in deleting state"
		} else if vm.State() == StateRestarting && time.Since(vm.UpdatedAt()) > 5*time.Minute {
			results[id] = "WARNING: VM stuck in restarting state"
		} else if vm.State() == VMStateMigrating && time.Since(vm.UpdatedAt()) > 15*time.Minute {
			results[id] = "WARNING: VM stuck in migrating state"
		} else {
			results[id] = "HEALTHY"
		}
	}
	
	return results
}
