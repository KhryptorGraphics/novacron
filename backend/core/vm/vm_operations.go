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

// Assuming CreateVMRequest has fields like Name string, Spec VMConfig, Tags map[string]string, Owner string
// Assuming VMConfig has fields VCPU, MemoryMB, DiskMB (adjust if different)

// CreateVM creates a new VM
func (m *VMManager) CreateVM(ctx context.Context, req CreateVMRequest) (*VM, error) {
	// Generate a unique ID for the VM if not provided in Spec
	if req.Spec.ID == "" {
		req.Spec.ID = uuid.New().String()
	}
	vmID := req.Spec.ID

	// Use default VM type if not specified - Assuming VMConfig has Type field
	// if req.Spec.Type == "" {
	// 	req.Spec.Type = m.config.DefaultVMType
	// }

	// Get the VM driver - Assuming driverFactory takes VMConfig or relevant field
	driver, err := m.driverFactory(req.Spec) // Pass the whole spec? Or just type? Needs driverFactory definition
	if err != nil {
		return nil, fmt.Errorf("failed to get VM driver: %w", err)
	}

	// Create resource constraints using fields from VMConfig
	constraints := []scheduler.ResourceConstraint{
		{
			Type:      scheduler.ResourceCPU,
			MinAmount: float64(req.Spec.CPUShares), // Use CPUShares
		},
		{
			Type:      scheduler.ResourceMemory,
			MinAmount: float64(req.Spec.MemoryMB), // Use MemoryMB
		},
		// Disk constraint might need more info than just RootFS size
		// {
		// 	Type:      scheduler.ResourceDisk,
		// 	MinAmount: float64(req.Spec.DiskMB), // Assuming DiskMB exists
		// },
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

	// Create the VM object using the constructor
	vm, err := NewVM(req.Spec) // Pass the VMConfig spec
	if err != nil {
		m.scheduler.CancelRequest(resourceID)
		return nil, fmt.Errorf("failed to initialize VM object: %w", err)
	}
	// Set internal fields not handled by constructor (if any)
	// vm.nodeID = allocation.NodeID // Assuming nodeID is internal
	// vm.resourceID = resourceID // Assuming resourceID is internal
	// vm.owner = req.Owner // Assuming owner is internal

	// Set initial state via method if available, otherwise internal (carefully)
	vm.mutex.Lock()
	vm.state = StateCreating // Directly setting internal state
	vm.mutex.Unlock()

	// Store the VM
	m.vmsMutex.Lock()
	m.vms[vmID] = vm
	m.vmsMutex.Unlock()

	// Create the VM on the node using the config from the VM object
	// Assuming driver.Create takes VMConfig
	_, err = driver.Create(ctx, vm.config) // driverID declared and not used error fixed
	if err != nil {
		vm.mutex.Lock()
		vm.state = StateFailed // Use correct state constant
		// vm.errorMessage = fmt.Sprintf("Failed to create VM: %v", err) // Assuming internal field
		vm.mutex.Unlock()

		// Emit error event
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm, // Pass the VM object
			Timestamp: time.Now(),
			NodeID:    allocation.NodeID, // Use nodeID from allocation
			Message:   fmt.Sprintf("Failed to create VM: %v", err),
		})

		// Cancel the resource request
		m.scheduler.CancelRequest(resourceID)

		return vm, err
	}

	// Update the VM state
	vm.mutex.Lock()
	vm.state = StateStopped // Use correct state constant
	// vm.updatedAt = time.Now() // Assuming internal field
	vm.mutex.Unlock()

	// Emit created event
	m.emitEvent(VMEvent{
		Type:      VMEventCreated,
		VM:        *vm, // Pass the VM object
		Timestamp: time.Now(),
		NodeID:    allocation.NodeID, // Use nodeID from allocation
	})

	log.Printf("Created VM %s of type %s on node %s", vm.ID(), vm.config.Command, allocation.NodeID) // Use methods and allocation nodeID

	return vm, nil // Return the created VM object
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

	// Get the VM driver - Assuming driverFactory takes VMConfig
	driver, err := m.driverFactory(vm.config)
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
	if vm.State() == StateRunning {
		return &VMOperationResponse{Success: true, VM: vm}, nil
	}
	if vm.State() != StateStopped && vm.State() != StateCreated && vm.State() != StateFailed {
		return &VMOperationResponse{Success: false, ErrorMessage: fmt.Sprintf("VM %s is in state %s, cannot start", vm.ID(), vm.State())}, nil
	}

	// Call the VM's own Start method which handles internal state
	err := vm.Start() // Use the VM's Start method
	if err != nil {
		// State should be updated to Failed within vm.Start() if it fails
		errorMessage := fmt.Sprintf("Failed to start VM: %v", err)
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID(), // Assuming NodeID method exists or get from config/internal state
			Message:   errorMessage,
		})
		return &VMOperationResponse{Success: false, ErrorMessage: errorMessage, VM: vm}, err
	}

	// Emit started event (VM's Start method might already do this)
	m.emitEvent(VMEvent{
		Type:      VMEventStarted,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(), // Assuming NodeID method exists
	})

	log.Printf("Started VM %s on node %s", vm.ID(), vm.NodeID())

	return &VMOperationResponse{Success: true, VM: vm}, nil
}

// stopVM stops a VM
func (m *VMManager) stopVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	if vm.State() == StateStopped {
		return &VMOperationResponse{Success: true, VM: vm}, nil
	}
	if vm.State() != StateRunning {
		return &VMOperationResponse{Success: false, ErrorMessage: fmt.Sprintf("VM %s is in state %s, cannot stop", vm.ID(), vm.State())}, nil
	}

	// Call the VM's own Stop method
	err := vm.Stop() // Use the VM's Stop method
	if err != nil {
		// State should be updated within vm.Stop() if it fails
		errorMessage := fmt.Sprintf("Failed to stop VM: %v", err)
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID(), // Assuming NodeID method exists
			Message:   errorMessage,
		})
		return &VMOperationResponse{Success: false, ErrorMessage: errorMessage, VM: vm}, err
	}

	// Emit stopped event (VM's Stop method might already do this)
	m.emitEvent(VMEvent{
		Type:      VMEventStopped,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(), // Assuming NodeID method exists
	})

	log.Printf("Stopped VM %s on node %s", vm.ID(), vm.NodeID())

	return &VMOperationResponse{Success: true, VM: vm}, nil
}

// restartVM restarts a VM
func (m *VMManager) restartVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	// Update the VM state internally (optional intermediate state like Restarting)
	vm.mutex.Lock()
	// vm.state = StateRestarting // If such state exists
	// vm.updatedAt = time.Now()
	vm.mutex.Unlock()

	// Stop the VM
	stopResp, err := m.stopVM(ctx, vm, driver)
	if err != nil || !stopResp.Success {
		// Error already handled and emitted by stopVM
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("Failed to stop VM during restart: %s", stopResp.ErrorMessage),
			VM:           vm,
		}, err
	}

	// Start the VM
	startResp, err := m.startVM(ctx, vm, driver)
	if err != nil || !startResp.Success {
		// Error already handled and emitted by startVM
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("Failed to start VM during restart: %s", startResp.ErrorMessage),
			VM:           vm,
		}, err
	}

	// Emit restarted event
	// TODO: Define VMEventRestarted if needed, or rely on Stop/Start events
	// m.emitEvent(VMEvent{
	// 	Type:      VMEventRestarted,
	// 	VM:        *vm,
	// 	Timestamp: time.Now(),
	// 	NodeID:    vm.NodeID(),
	// 	Message:   "VM restarted successfully",
	// })

	log.Printf("Restarted VM %s on node %s", vm.ID(), vm.NodeID())

	return &VMOperationResponse{Success: true, VM: vm}, nil
}

// pauseVM pauses a VM (Placeholder - Requires VMDriver interface update and VM state)
func (m *VMManager) pauseVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	log.Printf("Pause operation requested for VM %s (not fully implemented)", vm.ID())
	// TODO: Implement Pause functionality
	// 1. Check if driver supports Pause
	// 2. Call driver.Pause(ctx, vm.ID())
	// 3. Update vm internal state to StatePaused (if defined)
	// 4. Emit VMEventPaused
	return &VMOperationResponse{Success: false, ErrorMessage: "Pause operation not implemented"}, nil
}

// resumeVM resumes a paused VM (Placeholder - Requires VMDriver interface update and VM state)
func (m *VMManager) resumeVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	log.Printf("Resume operation requested for VM %s (not fully implemented)", vm.ID())
	// TODO: Implement Resume functionality
	// 1. Check if driver supports Resume
	// 2. Check if VM is in a pausable state (e.g., StatePaused if defined)
	// 3. Call driver.Resume(ctx, vm.ID())
	// 4. Update vm internal state to StateRunning
	// 5. Emit VMEventResumed
	return &VMOperationResponse{Success: false, ErrorMessage: "Resume operation not implemented"}, nil
}

// snapshotVM creates a snapshot of a VM (Placeholder - Requires VMDriver interface update)
func (m *VMManager) snapshotVM(ctx context.Context, vm *VM, driver VMDriver, params map[string]string) (*VMOperationResponse, error) {
	log.Printf("Snapshot operation requested for VM %s (not fully implemented)", vm.ID())
	// TODO: Implement Snapshot functionality
	// 1. Check if driver supports Snapshot
	// 2. Call driver.Snapshot(ctx, vm.ID(), params)
	// 3. Emit VMEventSnapshot
	// 4. Return snapshot details in Data field of response
	return &VMOperationResponse{Success: false, ErrorMessage: "Snapshot operation not implemented"}, nil
}

// deleteVM deletes a VM (Placeholder - Requires VMDriver interface update)
func (m *VMManager) deleteVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	log.Printf("Delete operation requested for VM %s (not fully implemented)", vm.ID())
	// TODO: Implement Delete functionality
	// 1. Ensure VM is stopped (call stopVM if needed)
	// 2. Call driver.Delete(ctx, vm.ID())
	// 3. Clean up resources (scheduler, storage)
	// 4. Remove VM from m.vms map
	// 5. Emit VMEventDeleted
	return &VMOperationResponse{Success: false, ErrorMessage: "Delete operation not implemented"}, nil
}

// migrateVM migrates a VM (Placeholder - Requires VMDriver interface update and migration logic)
func (m *VMManager) migrateVM(ctx context.Context, vm *VM, driver VMDriver, params map[string]string) (*VMOperationResponse, error) {
	log.Printf("Migrate operation requested for VM %s (not fully implemented)", vm.ID())
	// TODO: Implement Migrate functionality
	// 1. Check if driver supports Migrate
	// 2. Determine target node/hypervisor
	// 3. Call driver.Migrate(ctx, vm.ID(), target, params)
	// 4. Update VM's NodeID and internal state
	// 5. Emit VMEventMigrated
	return &VMOperationResponse{Success: false, ErrorMessage: "Migrate operation not implemented"}, nil
}
