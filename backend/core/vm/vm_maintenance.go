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
	m.vmsMutex.RLock()
	vmIDs := make([]string, 0, len(m.vms))
	for id := range m.vms {
		vmIDs = append(vmIDs, id)
	}
	m.vmsMutex.RUnlock()
	
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
	m.vmsMutex.RLock()
	vm, exists := m.vms[vmID]
	m.vmsMutex.RUnlock()
	
	if !exists {
		return
	}
	
	// Skip VMs in transitional states
	if vm.State == VMStateCreating || 
		vm.State == VMStateDeleting || 
		vm.State == VMStateRestarting ||
		vm.State == VMStateMigrating {
		return
	}
	
	// Get the VM driver
	driver, err := m.driverFactory(vm.Spec.Type)
	if err != nil {
		log.Printf("Failed to get driver for VM %s: %v", vmID, err)
		return
	}
	
	// Get the current status
	status, err := driver.GetStatus(context.Background(), vmID)
	if err != nil {
		log.Printf("Failed to get status for VM %s: %v", vmID, err)
		
		// Mark as error if we can't get the status multiple times
		m.vmsMutex.Lock()
		if vm.ErrorMessage == "" {
			vm.ErrorMessage = "Failed to get VM status"
		}
		m.vmsMutex.Unlock()
		
		return
	}
	
	// Get additional info if available
	vmInfo, err := driver.GetInfo(context.Background(), vmID)
	if err != nil {
		log.Printf("Failed to get info for VM %s: %v", vmID, err)
	}
	
	// Update the VM state if it changed
	m.vmsMutex.Lock()
	
	oldState := vm.State
	if oldState != status {
		vm.State = status
		vm.UpdatedAt = time.Now()
		
		// Update timestamps based on state changes
		if status == VMStateRunning && oldState != VMStateRunning {
			vm.StartedAt = time.Now()
		} else if status == VMStateStopped && oldState != VMStateStopped {
			vm.StoppedAt = time.Now()
		}
	}
	
	// Update VM info if available
	if vmInfo != nil {
		// Update network info
		if len(vmInfo.NetworkInfo) > 0 {
			vm.NetworkInfo = vmInfo.NetworkInfo
		}
		
		// Update storage info
		if len(vmInfo.StorageInfo) > 0 {
			vm.StorageInfo = vmInfo.StorageInfo
		}
		
		// Update process info
		if vmInfo.ProcessInfo.PID > 0 {
			vm.ProcessInfo = vmInfo.ProcessInfo
		}
	}
	
	// Reset error state if it's now running
	if status == VMStateRunning {
		vm.ErrorMessage = ""
	}
	
	m.vmsMutex.Unlock()
	
	// Emit event if state changed
	if oldState != status {
		m.emitEvent(VMEvent{
			Type:      VMEventUpdated,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID,
			Message:   "VM state changed",
		})
		
		log.Printf("VM %s state changed from %s to %s", vm.ID, oldState, status)
	}
}

// cleanupVMs cleans up expired and deleted VMs
func (m *VMManager) cleanupVMs() {
	now := time.Now()
	
	// Find VMs to clean up
	m.vmsMutex.Lock()
	
	for id, vm := range m.vms {
		// Handle VMs in deleting state
		if vm.State == VMStateDeleting {
			// If it's been in deleting state for too long, remove it from the map
			if now.Sub(vm.UpdatedAt) > 10*time.Minute {
				delete(m.vms, id)
				log.Printf("Removed deleted VM %s from memory", id)
			}
			continue
		}
		
		// Check for VMs with errors that need auto-recovery
		if vm.State == VMStateError {
			// Attempt to recover VMs that have been in error state for over 5 minutes
			if now.Sub(vm.UpdatedAt) > 5*time.Minute {
				go m.tryRecoverVM(id)
			}
		}
		
		// Check for resource leaks - VMs in creating/restarting state for too long
		if (vm.State == VMStateCreating || vm.State == VMStateRestarting) && 
			now.Sub(vm.UpdatedAt) > 15*time.Minute {
			
			// Mark as error
			vm.State = VMStateError
			vm.ErrorMessage = "Operation timed out"
			vm.UpdatedAt = now
			
			// Emit event
			m.emitEvent(VMEvent{
				Type:      VMEventError,
				VM:        *vm,
				Timestamp: now,
				NodeID:    vm.NodeID,
				Message:   "Operation timed out",
			})
			
			log.Printf("VM %s operation timed out", id)
		}
	}
	
	m.vmsMutex.Unlock()
}

// tryRecoverVM attempts to recover a VM from an error state
func (m *VMManager) tryRecoverVM(vmID string) {
	log.Printf("Attempting to recover VM %s", vmID)
	
	// Get the VM
	m.vmsMutex.RLock()
	vm, exists := m.vms[vmID]
	m.vmsMutex.RUnlock()
	
	if !exists || vm.State != VMStateError {
		return
	}
	
	// Get the VM driver
	driver, err := m.driverFactory(vm.Spec.Type)
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
	vm.State = VMStateRunning
	vm.ErrorMessage = ""
	vm.UpdatedAt = time.Now()
	vm.StartedAt = time.Now()
	m.vmsMutex.Unlock()
	
	// Emit recovered event
	m.emitEvent(VMEvent{
		Type:      VMEventStarted,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID,
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
		if vm.State == VMStateDeleting {
			continue
		}
		
		if vm.ResourceID == "" {
			log.Printf("Warning: VM %s has no resource ID", id)
			continue
		}
		
		found := false
		for _, alloc := range allocations {
			if alloc.RequestID == vm.ResourceID {
				found = true
				break
			}
		}
		
		if !found {
			log.Printf("Warning: VM %s has resource ID %s but no active allocation", 
				id, vm.ResourceID)
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
		if vm.State == VMStateError {
			results[id] = "UNHEALTHY: VM in error state: " + vm.ErrorMessage
		} else if vm.State == VMStateCreating && time.Since(vm.CreatedAt) > 10*time.Minute {
			results[id] = "WARNING: VM stuck in creating state"
		} else if vm.State == VMStateDeleting && time.Since(vm.UpdatedAt) > 10*time.Minute {
			results[id] = "WARNING: VM stuck in deleting state"
		} else if vm.State == VMStateRestarting && time.Since(vm.UpdatedAt) > 5*time.Minute {
			results[id] = "WARNING: VM stuck in restarting state"
		} else if vm.State == VMStateMigrating && time.Since(vm.UpdatedAt) > 15*time.Minute {
			results[id] = "WARNING: VM stuck in migrating state"
		} else {
			results[id] = "HEALTHY"
		}
	}
	
	return results
}
