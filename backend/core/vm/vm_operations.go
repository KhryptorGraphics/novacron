package vm

import (
	"context"
	"errors"
	"fmt"
	"log"
	"time"

	"github.com/google/uuid"
	// "github.com/khryptorgraphics/novacron/backend/core/scheduler" // Temporarily commented out for testing
)

// VM operation constants
const (
	VMOperationStart    = "start"
	VMOperationStop     = "stop"
	VMOperationRestart  = "restart"
	VMOperationDelete   = "delete"
	VMOperationMigrate  = "migrate"
	VMOperationPause    = "pause"
	VMOperationResume   = "resume"
	VMOperationSnapshot = "snapshot"
	VMOperationRestore  = "restore"
	VMOperationClone    = "clone"
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

	// Use default VM type if not specified
	if req.Spec.Type == "" {
		req.Spec.Type = VMTypeContainerd // Use containerd as default
	}

	// Get the VM driver - Use consistent getDriver method
	driver, err := m.getDriver(req.Spec)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM driver: %w", err)
	}

	// Create resource constraints using fields from VMConfig
	// TODO: Re-enable scheduler integration when scheduler package is available
	// constraints := []scheduler.ResourceConstraint{
	//	{
	//		Type:      scheduler.ResourceCPU,
	//		MinAmount: float64(req.Spec.CPUShares), // Use CPUShares
	//	},
	//	{
	//		Type:      scheduler.ResourceMemory,
	//		MinAmount: float64(req.Spec.MemoryMB), // Use MemoryMB
	//	},
	// }

	// Request resources from the scheduler
	// TODO: Re-enable scheduler integration when scheduler package is available
	// resourceID, err := m.scheduler.RequestResources(constraints, 1, 1*time.Hour)
	// if err != nil {
	//	return nil, fmt.Errorf("failed to request resources: %w", err)
	// }

	// Create a task to distribute the VM
	// taskID, err := m.scheduler.DistributeTask(resourceID, 1)
	// if err != nil {
	//	m.scheduler.CancelRequest(resourceID)
	//	return nil, fmt.Errorf("failed to distribute task: %w", err)
	// }

	// Wait for the task to be allocated
	// for {
	//	status, err := m.scheduler.GetTaskStatus(taskID)
	//	if err != nil {
	//		// TODO: Re-enable when scheduler is available
	// m.scheduler.CancelRequest(resourceID)
	//		return nil, fmt.Errorf("failed to get task status: %w", err)
	//	}

	//	if status == scheduler.TaskAllocated {
	//		break
	//	}

	//	if status == scheduler.TaskFailed {
	//		// TODO: Re-enable when scheduler is available
	// m.scheduler.CancelRequest(resourceID)
	//		return nil, errors.New("task allocation failed")
	// }

	//	// Check for context cancellation
	//	select {
	//	case <-ctx.Done():
	//		// TODO: Re-enable when scheduler is available
	//		// m.scheduler.CancelRequest(resourceID)
	//		return nil, ctx.Err()
	//	case <-time.After(100 * time.Millisecond):
	//		// Continue waiting
	//	}
	// }

	// Get the allocations
	// TODO: Re-enable when scheduler is available
	// allocations := m.scheduler.GetActiveAllocations()
	// var allocation scheduler.ResourceAllocation
	// found := false
	// for _, a := range allocations {
	//	if a.RequestID == resourceID {
	//		allocation = a
	//		found = true
	//		break
	//	}
	// }

	// TODO: Skip scheduler check for now
	found := true // Temporary fix
	if !found {
		// TODO: Re-enable when scheduler is available
		// m.scheduler.CancelRequest(resourceID)
		return nil, errors.New("allocation not found")
	}

	// Create the VM object using the constructor
	vm, err := NewVM(req.Spec) // Pass the VMConfig spec
	if err != nil {
		// TODO: Re-enable when scheduler is available
		// m.scheduler.CancelRequest(resourceID)
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
			NodeID:    vm.NodeID(), // Use actual VM NodeID
			Message:   fmt.Sprintf("Failed to create VM: %v", err),
		})

		// Cancel the resource request
		// TODO: Re-enable when scheduler is available
		// m.scheduler.CancelRequest(resourceID)

		return vm, err
	}

	// Store the VM only after successful creation
	m.vmsMutex.Lock()
	m.vms[vmID] = vm
	m.vmsMutex.Unlock()

	// Update the VM state
	vm.mutex.Lock()
	vm.state = StateStopped // Use correct state constant
	// vm.updatedAt = time.Now() // Assuming internal field
	vm.mutex.Unlock()

	// Track resource allocation after successful VM creation
	m.resourceMutex.Lock()
	m.allocatedCPU += req.Spec.CPUShares
	m.allocatedMemoryMB += int64(req.Spec.MemoryMB)
	// Clamp negative values at mutation time
	if m.allocatedCPU < 0 {
		m.allocatedCPU = 0
	}
	if m.allocatedMemoryMB < 0 {
		m.allocatedMemoryMB = 0
	}
	m.resourceMutex.Unlock()
	
	log.Printf("Allocated resources - CPU: %d, Memory: %dMB for VM %s", req.Spec.CPUShares, req.Spec.MemoryMB, vmID)

	// Emit created event
	m.emitEvent(VMEvent{
		Type:      VMEventCreated,
		VM:        *vm, // Pass the VM object
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(), // Use actual VM NodeID
	})

	log.Printf("Created VM %s of type %s on node %s", vm.ID(), vm.config.Command, vm.NodeID())

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

	// Get the VM driver - Use consistent getDriver method
	driver, err := m.getDriver(vm.config)
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
		// Convert params from map[string]interface{} to map[string]string
		stringParams := make(map[string]string)
		for k, v := range req.Params {
			if str, ok := v.(string); ok {
				stringParams[k] = str
			} else {
				stringParams[k] = fmt.Sprintf("%v", v)
			}
		}
		return m.migrateVM(ctx, vm, driver, stringParams)
	case VMOperationPause:
		return m.pauseVM(ctx, vm, driver)
	case VMOperationResume:
		return m.resumeVM(ctx, vm, driver)
	case VMOperationSnapshot:
		// Convert params from map[string]interface{} to map[string]string
		stringParams := make(map[string]string)
		for k, v := range req.Params {
			if str, ok := v.(string); ok {
				stringParams[k] = str
			} else {
				stringParams[k] = fmt.Sprintf("%v", v)
			}
		}
		return m.snapshotVM(ctx, vm, driver, stringParams)
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

	// Use driver to start in core mode; update state
	if err := driver.Start(ctx, vm.ID()); err != nil {
		errorMessage := fmt.Sprintf("Failed to start VM: %v", err)
		m.emitEvent(VMEvent{Type: VMEventError, VM: *vm, Timestamp: time.Now(), NodeID: vm.NodeID(), Message: errorMessage})
		return &VMOperationResponse{Success: false, ErrorMessage: errorMessage, VM: vm}, err
	}
	vm.SetState(StateRunning)
	m.emitEvent(VMEvent{Type: VMEventStarted, VM: *vm, Timestamp: time.Now(), NodeID: vm.NodeID()})
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

	// Use driver to stop in core mode; update state
	if err := driver.Stop(ctx, vm.ID()); err != nil {
		errorMessage := fmt.Sprintf("Failed to stop VM: %v", err)
		m.emitEvent(VMEvent{Type: VMEventError, VM: *vm, Timestamp: time.Now(), NodeID: vm.NodeID(), Message: errorMessage})
		return &VMOperationResponse{Success: false, ErrorMessage: errorMessage, VM: vm}, err
	}
	vm.SetState(StateStopped)
	m.emitEvent(VMEvent{Type: VMEventStopped, VM: *vm, Timestamp: time.Now(), NodeID: vm.NodeID()})
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

// pauseVM pauses a VM
func (m *VMManager) pauseVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	log.Printf("Pause operation requested for VM %s", vm.ID())

	// Check if VM is in a pausable state
	if vm.State() != StateRunning {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("VM %s is in state %s, cannot pause", vm.ID(), vm.State()),
			VM:           vm,
		}, nil
	}

	// Check if driver supports pause
	if !driver.SupportsPause() {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("VM driver does not support pause operation"),
			VM:           vm,
		}, nil
	}

	// Update VM state to pausing
	vm.mutex.Lock()
	vm.state = StatePausing
	vm.mutex.Unlock()

	// Call driver to pause the VM
	err := driver.Pause(ctx, vm.ID())
	if err != nil {
		// Update state back to running on failure
		vm.mutex.Lock()
		vm.state = StateRunning
		vm.mutex.Unlock()

		errorMessage := fmt.Sprintf("Failed to pause VM: %v", err)
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID(),
			Message:   errorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: errorMessage,
			VM:           vm,
		}, err
	}

	// Update VM state to paused
	vm.mutex.Lock()
	vm.state = StatePaused
	vm.mutex.Unlock()

	// Emit paused event
	m.emitEvent(VMEvent{
		Type:      VMEventPaused,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
	})

	log.Printf("Paused VM %s on node %s", vm.ID(), vm.NodeID())

	return &VMOperationResponse{Success: true, VM: vm}, nil
}

// resumeVM resumes a paused VM
func (m *VMManager) resumeVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	log.Printf("Resume operation requested for VM %s", vm.ID())

	// Check if VM is in a resumable state
	if vm.State() != StatePaused {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("VM %s is in state %s, cannot resume", vm.ID(), vm.State()),
			VM:           vm,
		}, nil
	}

	// Check if driver supports resume
	if !driver.SupportsResume() {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("VM driver does not support resume operation"),
			VM:           vm,
		}, nil
	}

	// Update VM state to resuming
	vm.mutex.Lock()
	vm.state = StateResuming
	vm.mutex.Unlock()

	// Call driver to resume the VM
	err := driver.Resume(ctx, vm.ID())
	if err != nil {
		// Update state back to paused on failure
		vm.mutex.Lock()
		vm.state = StatePaused
		vm.mutex.Unlock()

		errorMessage := fmt.Sprintf("Failed to resume VM: %v", err)
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID(),
			Message:   errorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: errorMessage,
			VM:           vm,
		}, err
	}

	// Update VM state to running
	vm.mutex.Lock()
	vm.state = StateRunning
	vm.mutex.Unlock()

	// Emit resumed event
	m.emitEvent(VMEvent{
		Type:      VMEventResumed,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
	})

	log.Printf("Resumed VM %s on node %s", vm.ID(), vm.NodeID())

	return &VMOperationResponse{Success: true, VM: vm}, nil
}

// snapshotVM creates a snapshot of a VM
func (m *VMManager) snapshotVM(ctx context.Context, vm *VM, driver VMDriver, params map[string]string) (*VMOperationResponse, error) {
	log.Printf("Snapshot operation requested for VM %s", vm.ID())

	// Check if driver supports snapshot
	if !driver.SupportsSnapshot() {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("VM driver does not support snapshot operation"),
			VM:           vm,
		}, nil
	}

	// Get snapshot name from params or generate one
	snapshotName := params["name"]
	if snapshotName == "" {
		// Generate a name based on timestamp
		snapshotName = fmt.Sprintf("%s-snapshot-%s", vm.ID(), time.Now().Format("20060102-150405"))
	}

	// Call driver to create snapshot
	snapshotID, err := driver.Snapshot(ctx, vm.ID(), snapshotName, params)
	if err != nil {
		errorMessage := fmt.Sprintf("Failed to create snapshot: %v", err)
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID(),
			Message:   errorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: errorMessage,
			VM:           vm,
		}, err
	}

	// Emit snapshot event
	m.emitEvent(VMEvent{
		Type:      VMEventSnapshot,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
		Message:   fmt.Sprintf("Created snapshot %s", snapshotName),
	})

	log.Printf("Created snapshot %s for VM %s", snapshotName, vm.ID())

	// Return success with snapshot details
	return &VMOperationResponse{
		Success: true,
		VM:      vm,
		Data: map[string]string{
			"snapshot_id":   snapshotID,
			"snapshot_name": snapshotName,
			"created_at":    time.Now().Format(time.RFC3339),
		},
	}, nil
}

// deleteVM deletes a VM
func (m *VMManager) deleteVM(ctx context.Context, vm *VM, driver VMDriver) (*VMOperationResponse, error) {
	log.Printf("Delete operation requested for VM %s", vm.ID())

	// If VM is running, stop it first
	if vm.State() == StateRunning || vm.State() == StatePaused {
		log.Printf("Stopping VM %s before deletion", vm.ID())
		stopResp, err := m.stopVM(ctx, vm, driver)
		if err != nil || !stopResp.Success {
			errorMessage := fmt.Sprintf("Failed to stop VM before deletion: %v", err)
			if err == nil && !stopResp.Success {
				errorMessage = fmt.Sprintf("Failed to stop VM before deletion: %s", stopResp.ErrorMessage)
			}

			m.emitEvent(VMEvent{
				Type:      VMEventError,
				VM:        *vm,
				Timestamp: time.Now(),
				NodeID:    vm.NodeID(),
				Message:   errorMessage,
			})

			return &VMOperationResponse{
				Success:      false,
				ErrorMessage: errorMessage,
				VM:           vm,
			}, err
		}
	}

	// Update VM state to deleting
	vm.mutex.Lock()
	vm.state = StateDeleting
	vm.mutex.Unlock()

	// Call driver to delete the VM
	err := driver.Delete(ctx, vm.ID())
	if err != nil {
		// Update state to failed on error
		vm.mutex.Lock()
		vm.state = StateFailed
		vm.mutex.Unlock()

		errorMessage := fmt.Sprintf("Failed to delete VM: %v", err)
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID(),
			Message:   errorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: errorMessage,
			VM:           vm,
		}, err
	}

	// Clean up resources
	// If VM has a resource allocation, release it
	if vm.resourceID != "" {
		// TODO: Re-enable when scheduler is available
		// m.scheduler.CancelRequest(vm.resourceID)
	}

	// Emit deleted event
	m.emitEvent(VMEvent{
		Type:      VMEventDeleted,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.NodeID(),
	})

	// Deallocate resources before removing VM
	config := vm.Config()
	m.resourceMutex.Lock()
	m.allocatedCPU -= config.CPUShares
	m.allocatedMemoryMB -= int64(config.MemoryMB)
	// Clamp negative values at mutation time
	if m.allocatedCPU < 0 {
		m.allocatedCPU = 0
	}
	if m.allocatedMemoryMB < 0 {
		m.allocatedMemoryMB = 0
	}
	m.resourceMutex.Unlock()
	
	log.Printf("Deallocated resources - CPU: %d, Memory: %dMB for VM %s", config.CPUShares, config.MemoryMB, vm.ID())

	// Remove VM from manager's map
	m.vmsMutex.Lock()
	delete(m.vms, vm.ID())
	m.vmsMutex.Unlock()

	log.Printf("Deleted VM %s", vm.ID())

	return &VMOperationResponse{
		Success: true,
		Data: map[string]string{
			"vm_id":  vm.ID(),
			"status": "deleted",
		},
	}, nil
}

// migrateVM migrates a VM to another node
func (m *VMManager) migrateVM(ctx context.Context, vm *VM, driver VMDriver, params map[string]string) (*VMOperationResponse, error) {
	log.Printf("Migrate operation requested for VM %s", vm.ID())

	// Check if driver supports migration
	if !driver.SupportsMigrate() {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("VM driver does not support migration"),
			VM:           vm,
		}, nil
	}

	// Get target node from params
	targetNode := params["target_node"]
	if targetNode == "" {
		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: "Target node not specified for migration",
			VM:           vm,
		}, nil
	}

	// Get migration type from params (default to "live" if supported)
	migrationType := params["migration_type"]
	if migrationType == "" {
		migrationType = "live"
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
		Message:   fmt.Sprintf("Starting %s migration to node %s", migrationType, targetNode),
	})

	// Call driver to migrate the VM
	err := driver.Migrate(ctx, vm.ID(), targetNode, params)
	if err != nil {
		// Update state back to previous state on failure
		vm.mutex.Lock()
		vm.state = StateRunning // Assume it was running before migration
		vm.mutex.Unlock()

		errorMessage := fmt.Sprintf("Failed to migrate VM: %v", err)
		m.emitEvent(VMEvent{
			Type:      VMEventError,
			VM:        *vm,
			Timestamp: time.Now(),
			NodeID:    vm.NodeID(),
			Message:   errorMessage,
		})

		return &VMOperationResponse{
			Success:      false,
			ErrorMessage: errorMessage,
			VM:           vm,
		}, err
	}

	// Update VM's node ID
	oldNodeID := vm.NodeID()
	vm.SetNodeID(targetNode) // Update VM's node ID to target node
	vm.SetState(StateRunning) // Use the proper setter method

	// Emit migrated event
	m.emitEvent(VMEvent{
		Type:      VMEventMigrated,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    targetNode,
		Message:   fmt.Sprintf("VM migrated from node %s to %s", oldNodeID, targetNode),
	})

	log.Printf("Migrated VM %s from node %s to %s", vm.ID(), oldNodeID, targetNode)

	return &VMOperationResponse{
		Success: true,
		VM:      vm,
		Data: map[string]string{
			"source_node":     oldNodeID,
			"target_node":     targetNode,
			"migration_type":  migrationType,
			"completion_time": time.Now().Format(time.RFC3339),
		},
	}, nil
}

// Public API methods for external calls

// StartVM starts a VM by ID
func (m *VMManager) StartVM(ctx context.Context, vmID string) error {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return err
	}

	driver, err := m.getDriverForVM(vm)
	if err != nil {
		return err
	}

	response, err := m.startVM(ctx, vm, driver)
	if err != nil {
		return err
	}

	if !response.Success {
		return fmt.Errorf("failed to start VM: %s", response.ErrorMessage)
	}

	return nil
}

// StopVM stops a VM by ID
func (m *VMManager) StopVM(ctx context.Context, vmID string) error {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return err
	}

	driver, err := m.getDriverForVM(vm)
	if err != nil {
		return err
	}

	response, err := m.stopVM(ctx, vm, driver)
	if err != nil {
		return err
	}

	if !response.Success {
		return fmt.Errorf("failed to stop VM: %s", response.ErrorMessage)
	}

	return nil
}

// RestartVM restarts a VM by ID
func (m *VMManager) RestartVM(ctx context.Context, vmID string) error {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return err
	}

	driver, err := m.getDriverForVM(vm)
	if err != nil {
		return err
	}

	response, err := m.restartVM(ctx, vm, driver)
	if err != nil {
		return err
	}

	if !response.Success {
		return fmt.Errorf("failed to restart VM: %s", response.ErrorMessage)
	}

	return nil
}

// PauseVM pauses a VM by ID
func (m *VMManager) PauseVM(ctx context.Context, vmID string) error {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return err
	}

	driver, err := m.getDriverForVM(vm)
	if err != nil {
		return err
	}

	response, err := m.pauseVM(ctx, vm, driver)
	if err != nil {
		return err
	}

	if !response.Success {
		return fmt.Errorf("failed to pause VM: %s", response.ErrorMessage)
	}

	return nil
}

// ResumeVM resumes a VM by ID
func (m *VMManager) ResumeVM(ctx context.Context, vmID string) error {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return err
	}

	driver, err := m.getDriverForVM(vm)
	if err != nil {
		return err
	}

	response, err := m.resumeVM(ctx, vm, driver)
	if err != nil {
		return err
	}

	if !response.Success {
		return fmt.Errorf("failed to resume VM: %s", response.ErrorMessage)
	}

	return nil
}

// DeleteVM deletes a VM by ID
func (m *VMManager) DeleteVM(ctx context.Context, vmID string) error {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return err
	}

	driver, err := m.getDriverForVM(vm)
	if err != nil {
		return err
	}

	response, err := m.deleteVM(ctx, vm, driver)
	if err != nil {
		return err
	}

	if !response.Success {
		return fmt.Errorf("failed to delete VM: %s", response.ErrorMessage)
	}

	return nil
}

// Removed duplicate getDriverForVM - using the one from vm_manager.go
