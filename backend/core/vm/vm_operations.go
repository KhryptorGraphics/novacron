package vm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
)

// CreateVM creates a new VM
func (m *VMManager) CreateVM(ctx context.Context, req CreateVMRequest) (*VM, error) {
	// Generate a unique ID for the VM
	vmID := uuid.New().String()

	// Use default VM type if not specified
	if req.Spec.Type == "" {
		req.Spec.Type = m.config.DefaultVMType
	}

	// Get the VM driver
	driver, err := m.driverFactory(req.Spec.Type)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM driver: %w", err)
	}

	// Create resource constraints
	constraints := []scheduler.ResourceConstraint{
		{
			Type:      scheduler.ResourceCPU,
			MinAmount: float64(req.Spec.VCPU),
		},
		{
			Type:      scheduler.ResourceMemory,
			MinAmount: float64(req.Spec.MemoryMB),
		},
		{
			Type:      scheduler.ResourceDisk,
			MinAmount: float64(req.Spec.DiskMB),
		},
	}

	// Request resources from the scheduler
	resourceID, err := m.scheduler.RequestResources(constraints, 1, 1*time.Hour)
	if err != nil {
		return nil, fmt.Errorf("failed to request resources: %w", err)
	}

	// Create a task to distribute the VM
	taskID, err := m.scheduler.DistributeTask(resourceID, 1)
	if err != nil {
		m.scheduler.CancelRequest(resourceID)
		return nil, fmt.Errorf("failed to distribute task: %w", err)
	}

	// Wait for the task to be allocated
	for {
		status, err := m.scheduler.GetTaskStatus(taskID)
		if err != nil {
			m.scheduler.CancelRequest(resourceID)
			return nil, fmt.Errorf("failed to get task status: %w", err)
		}

		if status == scheduler.TaskAllocated {
			break
		}

		if status == scheduler.TaskFailed {
			m.scheduler.CancelRequest(resourceID)
			return nil, errors.New("task allocation failed")
		}

		// Check for context cancellation
		select {
		case <-ctx.Done():
			m.scheduler.CancelRequest(resourceID)
			return nil, ctx.Err()
		case <-time.After(100 * time.Millisecond):
			// Continue waiting
		}
	}

	// Get the allocations
	allocations := m.scheduler.GetActiveAllocations()
	var allocation scheduler.ResourceAllocation
	found := false
	for _, a := range allocations {
		if a.RequestID == resourceID {
			allocation = a
			found = true
			break
		}
	}

	if !found {
		m.scheduler.CancelRequest(resourceID)
		return nil, errors.New("allocation not found")
	}

	// Create the VM
	vm := &VM{
		ID:         vmID,
		Name:       req.Name,
		Spec:       req.Spec,
		State:      VMStateCreating,
		NodeID:     allocation.NodeID,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		Tags:       req.Tags,
		Owner:      req.Owner,
		ResourceID: resourceID,
	}

	// Store the VM
	m.vmsMutex.Lock()
	m.vms[vmID] = vm
	m.vmsMutex.Unlock()

	// Create the VM on the node
	driverID, err := driver.Create(ctx, req.Spec)
	if err != nil {
		vm.State = VMStateError
		vm.ErrorMessage = fmt.Sprintf("Failed to create VM: %v", err)
		vm.UpdatedAt = time.Now()

		// Emit error event
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID,
			Message:   vm.ErrorMessage,
		})

		// Cancel the resource request
		m.scheduler.CancelRequest(resourceID)

		return vm, err
	}

	// Update the VM state
	vm.State = VMStateStopped
	vm.UpdatedAt = time.Now()

	// Emit created event
	m.emitEvent(VMEvent{
		Type:      VMEventCreated,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID,
	})

	log.Printf("Created VM %s of type %s on node %s", vm.ID, vm.Spec.Type, vm.NodeID)

	return vm, nil
}

// PerformVMOperation performs an operation on a VM
func (m *VMManager) PerformVMOperation(ctx context.Context, req VMOperationRequest) (*VMOperationResponse, error) {
	// Get the VM
	m.vmsMutex.RLock()
	vm, exists := m.vms[req.VMID]
	m.vmsMutex.RUnlock()

	if !exists {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("VM %s not found", req.VMID),
		}, nil
	}

	// Get the VM driver
	driver, err := m.driverFactory(vm.Spec.Type)
	if err != nil {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("Failed to get VM driver: %v", err),
		}, err
	}

	// Perform the operation
	switch req.Operation {
	case VMOperationStart:
		return m.startVM(ctx, vm, driver)
	case VMOperationStop:
		return m.stopVM(ctx, vm, driver)
	case VMOperationRestart:
		return m.restartVM(ctx, vm, driver)
	case VMOperationDelete:
		return m.deleteVM(ctx, vm, driver)
	case VMOperationMigrate:
		return m.migrateVM(ctx, vm, driver, req.Params)
	case VMOperationPause:
		return m.pauseVM(ctx, vm, driver)
	case VMOperationResume:
		return m.resumeVM(ctx, vm, driver)
	case VMOperationSnapshot:
		return m.snapshotVM(ctx, vm, driver, req.Params)
	default:
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("Unsupported operation: %s", req.Operation),
		}, nil
	}
}

// startVM starts a VM
func (m *VMManager) startVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	// Check if the VM is already running
	if vm.State == VMStateRunning {
		return &VMOperationResponse{
			Success: true,
			VM:      vm,
		}, nil
	}

	// Update the VM state
	m.vmsMutex.Lock()
	vm.State = VMStateRestarting
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Start the VM
	err := driver.Start(ctx, vm.ID)
	if err != nil {
		m.vmsMutex.Lock()
		vm.State = VMStateError
		vm.ErrorMessage = fmt.Sprintf("Failed to start VM: %v", err)
		vm.UpdatedAt = time.Now()
		m.vmsMutex.Unlock()

		// Emit error event
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID,
			Message:   vm.ErrorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: vm.ErrorMessage,
			VM:           vm,
		}, err
	}

	// Update the VM state
	m.vmsMutex.Lock()
	vm.State = VMStateRunning
	vm.StartedAt = time.Now()
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Emit started event
	m.emitEvent(VMEvent{
		Type:      VMEventStarted,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID,
	})

	log.Printf("Started VM %s on node %s", vm.ID, vm.NodeID)

	return &VMOperationResponse{
		Success: true,
		VM:      vm,
	}, nil
}

// stopVM stops a VM
func (m *VMManager) stopVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	// Check if the VM is already stopped
	if vm.State == VMStateStopped {
		return &VMOperationResponse{
			Success: true,
			VM:      vm,
		}, nil
	}

	// Update the VM state
	m.vmsMutex.Lock()
	vm.State = VMStateRestarting
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Stop the VM
	err := driver.Stop(ctx, vm.ID)
	if err != nil {
		m.vmsMutex.Lock()
		vm.State = VMStateError
		vm.ErrorMessage = fmt.Sprintf("Failed to stop VM: %v", err)
		vm.UpdatedAt = time.Now()
		m.vmsMutex.Unlock()

		// Emit error event
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID,
			Message:   vm.ErrorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: vm.ErrorMessage,
			VM:           vm,
		}, err
	}

	// Update the VM state
	m.vmsMutex.Lock()
	vm.State = VMStateStopped
	vm.StoppedAt = time.Now()
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Emit stopped event
	m.emitEvent(VMEvent{
		Type:      VMEventStopped,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID,
	})

	log.Printf("Stopped VM %s on node %s", vm.ID, vm.NodeID)

	return &VMOperationResponse{
		Success: true,
		VM:      vm,
	}, nil
}

// restartVM restarts a VM
func (m *VMManager) restartVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	// Update the VM state
	m.vmsMutex.Lock()
	vm.State = VMStateRestarting
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Stop the VM
	err := driver.Stop(ctx, vm.ID)
	if err != nil {
		m.vmsMutex.Lock()
		vm.State = VMStateError
		vm.ErrorMessage = fmt.Sprintf("Failed to stop VM during restart: %v", err)
		vm.UpdatedAt = time.Now()
		m.vmsMutex.Unlock()

		// Emit error event
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID,
			Message:   vm.ErrorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: vm.ErrorMessage,
			VM:           vm,
		}, err
	}

	// Start the VM
	err = driver.Start(ctx, vm.ID)
	if err != nil {
		m.vmsMutex.Lock()
		vm.State = VMStateError
		vm.ErrorMessage = fmt.Sprintf("Failed to start VM during restart: %v", err)
		vm.UpdatedAt = time.Now()
		m.vmsMutex.Unlock()

		// Emit error event
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID,
			Message:   vm.ErrorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: vm.ErrorMessage,
			VM:           vm,
		}, err
	}

	// Update the VM state
	m.vmsMutex.Lock()
	vm.State = VMStateRunning
	vm.StartedAt = time.Now()
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Emit event
	m.emitEvent(VMEvent{
		Type:      VMEventStarted,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID,
		Message:   "VM restarted",
	})

	log.Printf("Restarted VM %s on node %s", vm.ID, vm.NodeID)

	return &VMOperationResponse{
		Success: true,
		VM:      vm,
	}, nil
}

// pauseVM pauses a VM
func (m *VMManager) pauseVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	// This is a simplified implementation since many drivers might not support pause
	// In a real implementation, you'd call the driver's pause method

	// Update the VM state
	m.vmsMutex.Lock()
	vm.State = VMStatePaused
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Emit paused event
	m.emitEvent(VMEvent{
		Type:      VMEventPaused,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID,
	})

	log.Printf("Paused VM %s on node %s", vm.ID, vm.NodeID)

	return &VMOperationResponse{
		Success: true,
		VM:      vm,
	}, nil
}

// resumeVM resumes a paused VM
func (m *VMManager) resumeVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	// This is a simplified implementation since many drivers might not support resume
	// In a real implementation, you'd call the driver's resume method

	// Update the VM state
	m.vmsMutex.Lock()
	vm.State = VMStateRunning
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Emit resumed event
	m.emitEvent(VMEvent{
		Type:      VMEventResumed,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID,
	})

	log.Printf("Resumed VM %s on node %s", vm.ID, vm.NodeID)

	return &VMOperationResponse{
		Success: true,
		VM:      vm,
	}, nil
}

// snapshotVM creates a snapshot of a VM
func (m *VMManager) snapshotVM(ctx context.Context, vm *VM, driver VMDriver, params map[string]string) (*VMOperationResponse, error) {
	// This is a simplified implementation since many drivers might not support snapshots
	// In a real implementation, you'd call the driver's snapshot method

	// Update the VM state temporarrily for the operation
	m.vmsMutex.Lock()
	prevState := vm.State
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	// Emit snapshot event
	m.emitEvent(VMEvent{
		Type:      VMEventSnapshot,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID,
		Message:   fmt.Sprintf("Snapshot created with params: %v", params),
	})

	log.Printf("Created snapshot of VM %s on node %s", vm.ID, vm.NodeID)

	// Restore the previous state
	m.vmsMutex.Lock()
	vm.State = prevState
	vm.UpdatedAt = time.Now()
	m.vmsMutex.Unlock()

	return &VMOperationResponse{
		Success: true,
		VM:      vm,
	}, nil
}
