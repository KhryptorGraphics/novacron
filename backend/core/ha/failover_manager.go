package ha

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// FailoverState represents the state of a failover operation
type FailoverState string

const (
	// FailoverStateInitiated indicates the failover has been initiated
	FailoverStateInitiated FailoverState = "initiated"

	// FailoverStateInProgress indicates the failover is in progress
	FailoverStateInProgress FailoverState = "in_progress"

	// FailoverStateSucceeded indicates the failover has succeeded
	FailoverStateSucceeded FailoverState = "succeeded"

	// FailoverStateFailed indicates the failover has failed
	FailoverStateFailed FailoverState = "failed"

	// FailoverStateCancelled indicates the failover has been cancelled
	FailoverStateCancelled FailoverState = "cancelled"

	// FailoverStateAborted indicates the failover has been aborted
	FailoverStateAborted FailoverState = "aborted"
)

// FailoverOperation represents a failover operation
type FailoverOperation struct {
	// ID is the unique identifier for this operation
	ID string

	// GroupID is the ID of the availability group
	GroupID string

	// OldPrimaryID is the ID of the old primary service
	OldPrimaryID string

	// NewPrimaryID is the ID of the new primary service
	NewPrimaryID string

	// State is the current state of the operation
	State FailoverState

	// Reason is the reason for the failover
	Reason string

	// StartTime is when the operation started
	StartTime time.Time

	// EndTime is when the operation completed
	EndTime time.Time

	// Timeout is the timeout for the operation
	Timeout time.Duration

	// Manual indicates if this is a manual failover
	Manual bool

	// Steps are the steps in the failover operation
	Steps []*FailoverStep

	// CurrentStep is the index of the current step
	CurrentStep int

	// Error is the error message if the operation failed
	Error string

	// Result is the result of the operation
	Result *FailoverResult
}

// FailoverStepState represents the state of a failover step
type FailoverStepState string

const (
	// StepStatePending indicates the step is pending
	StepStatePending FailoverStepState = "pending"

	// StepStateInProgress indicates the step is in progress
	StepStateInProgress FailoverStepState = "in_progress"

	// StepStateSucceeded indicates the step has succeeded
	StepStateSucceeded FailoverStepState = "succeeded"

	// StepStateFailed indicates the step has failed
	StepStateFailed FailoverStepState = "failed"

	// StepStateSkipped indicates the step has been skipped
	StepStateSkipped FailoverStepState = "skipped"
)

// FailoverStep represents a step in a failover operation
type FailoverStep struct {
	// ID is the unique identifier for this step
	ID string

	// Name is the name of the step
	Name string

	// Description is a human-readable description of the step
	Description string

	// State is the current state of the step
	State FailoverStepState

	// StartTime is when the step started
	StartTime time.Time

	// EndTime is when the step completed
	EndTime time.Time

	// Error is the error message if the step failed
	Error string

	// Action is the action to perform for this step
	Action func(context.Context, *FailoverOperation) error

	// Rollback is the action to perform to roll back this step
	Rollback func(context.Context, *FailoverOperation) error

	// CanRollback indicates if this step can be rolled back
	CanRollback bool

	// CanSkip indicates if this step can be skipped
	CanSkip bool
}

// FailoverResult represents the result of a failover operation
type FailoverResult struct {
	// Success indicates if the operation was successful
	Success bool

	// Duration is the duration of the operation
	Duration time.Duration

	// NewPrimaryID is the ID of the new primary service
	NewPrimaryID string

	// SuccessfulSteps is the number of successful steps
	SuccessfulSteps int

	// TotalSteps is the total number of steps
	TotalSteps int

	// Errors contains error messages if any steps failed
	Errors []string
}

// FailoverManagerConfig contains configuration for the failover manager
type FailoverManagerConfig struct {
	// DefaultTimeout is the default timeout for failover operations
	DefaultTimeout time.Duration

	// RetryInterval is the interval between retries
	RetryInterval time.Duration

	// MaxRetries is the maximum number of retries
	MaxRetries int

	// RollbackEnabled indicates if rollback is enabled
	RollbackEnabled bool

	// FailoverPolicy is the policy for selecting a new primary
	FailoverPolicy FailoverPolicy

	// FailoverHistorySize is the size of the failover history
	FailoverHistorySize int
}

// DefaultFailoverManagerConfig returns a default configuration
func DefaultFailoverManagerConfig() FailoverManagerConfig {
	return FailoverManagerConfig{
		DefaultTimeout:      5 * time.Minute,
		RetryInterval:       10 * time.Second,
		MaxRetries:          3,
		RollbackEnabled:     true,
		FailoverPolicy:      FailoverPolicyPriority,
		FailoverHistorySize: 100,
	}
}

// FailoverPolicy defines the policy for selecting a new primary
type FailoverPolicy string

const (
	// FailoverPolicyPriority selects the service with the highest priority
	FailoverPolicyPriority FailoverPolicy = "priority"

	// FailoverPolicyRoundRobin selects the next service in a round-robin fashion
	FailoverPolicyRoundRobin FailoverPolicy = "round_robin"

	// FailoverPolicyLeastRecent selects the service that was primary least recently
	FailoverPolicyLeastRecent FailoverPolicy = "least_recent"

	// FailoverPolicyMostHealthy selects the healthiest service
	FailoverPolicyMostHealthy FailoverPolicy = "most_healthy"

	// FailoverPolicyCustom uses a custom selection function
	FailoverPolicyCustom FailoverPolicy = "custom"
)

// FailoverManager manages failover operations
type FailoverManager struct {
	config FailoverManagerConfig

	// availabilityManager is the availability manager
	availabilityManager *AvailabilityManager

	// operations maps operation IDs to failover operations
	operations     map[string]*FailoverOperation
	operationMutex sync.RWMutex

	// history contains recent failover operations
	history      []*FailoverOperation
	historyMutex sync.RWMutex

	// operationCallbacks maps operation IDs to callbacks
	operationCallbacks     map[string][]func(*FailoverOperation)
	operationCallbackMutex sync.RWMutex

	// customSelectFunction is a custom function for selecting a new primary
	customSelectFunction func(*AvailabilityGroup, string) (string, error)

	ctx    context.Context
	cancel context.CancelFunc
}

// NewFailoverManager creates a new failover manager
func NewFailoverManager(config FailoverManagerConfig, availabilityManager *AvailabilityManager) *FailoverManager {
	ctx, cancel := context.WithCancel(context.Background())

	return &FailoverManager{
		config:              config,
		availabilityManager: availabilityManager,
		operations:          make(map[string]*FailoverOperation),
		history:             make([]*FailoverOperation, 0, config.FailoverHistorySize),
		operationCallbacks:  make(map[string][]func(*FailoverOperation)),
		ctx:                 ctx,
		cancel:              cancel,
	}
}

// Start starts the failover manager
func (m *FailoverManager) Start() error {
	log.Println("Starting failover manager")

	// Start monitoring for potential failovers
	go m.monitorFailovers()

	return nil
}

// Stop stops the failover manager
func (m *FailoverManager) Stop() error {
	log.Println("Stopping failover manager")

	m.cancel()

	return nil
}

// monitorFailovers monitors for potential failovers
func (m *FailoverManager) monitorFailovers() {
	// The availability manager will notify us of potential failovers via callbacks
	// For now, we'll just monitor active operations
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.processActiveOperations()
		}
	}
}

// processActiveOperations processes active failover operations
func (m *FailoverManager) processActiveOperations() {
	m.operationMutex.RLock()
	operationIDs := make([]string, 0, len(m.operations))
	for id, op := range m.operations {
		if op.State == FailoverStateInitiated || op.State == FailoverStateInProgress {
			operationIDs = append(operationIDs, id)
		}
	}
	m.operationMutex.RUnlock()

	for _, id := range operationIDs {
		m.processOperation(id)
	}
}

// processOperation processes a failover operation
func (m *FailoverManager) processOperation(operationID string) {
	m.operationMutex.Lock()
	operation, exists := m.operations[operationID]
	if !exists || (operation.State != FailoverStateInitiated && operation.State != FailoverStateInProgress) {
		m.operationMutex.Unlock()
		return
	}

	// Check for timeout
	if !operation.EndTime.IsZero() && time.Now().After(operation.EndTime) {
		operation.State = FailoverStateFailed
		operation.Error = "operation timed out"
		m.operationMutex.Unlock()
		m.notifyOperationUpdate(operation)
		return
	}

	// Get the current step
	if operation.CurrentStep >= len(operation.Steps) {
		// All steps completed successfully
		operation.State = FailoverStateSucceeded
		operation.Result = m.createSuccessResult(operation)
		m.operationMutex.Unlock()
		m.notifyOperationUpdate(operation)
		m.addToHistory(operation)
		return
	}

	step := operation.Steps[operation.CurrentStep]
	m.operationMutex.Unlock()

	// Execute the step
	switch step.State {
	case StepStatePending:
		m.startStep(operation, step)
	case StepStateInProgress:
		// Step is already in progress, nothing to do
	case StepStateSucceeded:
		// Move to the next step
		m.operationMutex.Lock()
		operation.CurrentStep++
		m.operationMutex.Unlock()
		m.notifyOperationUpdate(operation)
	case StepStateFailed:
		// Step failed, check if we should rollback
		m.handleStepFailure(operation, step)
	case StepStateSkipped:
		// Move to the next step
		m.operationMutex.Lock()
		operation.CurrentStep++
		m.operationMutex.Unlock()
		m.notifyOperationUpdate(operation)
	}
}

// startStep starts a failover step
func (m *FailoverManager) startStep(operation *FailoverOperation, step *FailoverStep) {
	log.Printf("Starting failover step %s for operation %s", step.Name, operation.ID)

	// Update step state
	step.State = StepStateInProgress
	step.StartTime = time.Now()
	m.notifyOperationUpdate(operation)

	// Execute the step in a goroutine
	go func() {
		stepCtx, cancel := context.WithTimeout(m.ctx, operation.Timeout)
		defer cancel()

		err := step.Action(stepCtx, operation)

		m.operationMutex.Lock()
		defer m.operationMutex.Unlock()

		// Check if the operation is still active
		if operation.State != FailoverStateInitiated && operation.State != FailoverStateInProgress {
			return
		}

		// Update step state
		step.EndTime = time.Now()
		if err != nil {
			step.State = StepStateFailed
			step.Error = err.Error()
			log.Printf("Failover step %s failed: %v", step.Name, err)
		} else {
			step.State = StepStateSucceeded
			log.Printf("Failover step %s completed successfully", step.Name)
		}

		m.notifyOperationUpdate(operation)
	}()
}

// handleStepFailure handles a failed failover step
func (m *FailoverManager) handleStepFailure(operation *FailoverOperation, step *FailoverStep) {
	log.Printf("Handling failure of step %s for operation %s", step.Name, operation.ID)

	// If the step can be skipped, skip it and continue
	if step.CanSkip {
		log.Printf("Skipping failed step %s", step.Name)
		step.State = StepStateSkipped
		m.operationMutex.Lock()
		operation.CurrentStep++
		m.operationMutex.Unlock()
		m.notifyOperationUpdate(operation)
		return
	}

	// If rollback is enabled and the step can be rolled back, rollback the step
	if m.config.RollbackEnabled && step.CanRollback && step.Rollback != nil {
		log.Printf("Rolling back step %s", step.Name)
		go func() {
			rollbackCtx, cancel := context.WithTimeout(m.ctx, operation.Timeout)
			defer cancel()

			err := step.Rollback(rollbackCtx, operation)
			if err != nil {
				log.Printf("Rollback of step %s failed: %v", step.Name, err)
			} else {
				log.Printf("Rollback of step %s completed successfully", step.Name)
			}

			// Mark the operation as failed
			m.operationMutex.Lock()
			operation.State = FailoverStateFailed
			operation.Error = fmt.Sprintf("step %s failed: %s", step.Name, step.Error)
			m.operationMutex.Unlock()
			m.notifyOperationUpdate(operation)
			m.addToHistory(operation)
		}()
		return
	}

	// Can't skip or rollback, mark the operation as failed
	m.operationMutex.Lock()
	operation.State = FailoverStateFailed
	operation.Error = fmt.Sprintf("step %s failed: %s", step.Name, step.Error)
	m.operationMutex.Unlock()
	m.notifyOperationUpdate(operation)
	m.addToHistory(operation)
}

// createSuccessResult creates a success result for a failover operation
func (m *FailoverManager) createSuccessResult(operation *FailoverOperation) *FailoverResult {
	successfulSteps := 0
	for _, step := range operation.Steps {
		if step.State == StepStateSucceeded {
			successfulSteps++
		}
	}

	return &FailoverResult{
		Success:         true,
		Duration:        time.Since(operation.StartTime),
		NewPrimaryID:    operation.NewPrimaryID,
		SuccessfulSteps: successfulSteps,
		TotalSteps:      len(operation.Steps),
		Errors:          []string{},
	}
}

// notifyOperationUpdate notifies listeners of an operation update
func (m *FailoverManager) notifyOperationUpdate(operation *FailoverOperation) {
	m.operationCallbackMutex.RLock()
	callbacks, exists := m.operationCallbacks[operation.ID]
	m.operationCallbackMutex.RUnlock()

	if !exists {
		return
	}

	for _, callback := range callbacks {
		go callback(operation)
	}
}

// addToHistory adds an operation to the history
func (m *FailoverManager) addToHistory(operation *FailoverOperation) {
	m.historyMutex.Lock()
	defer m.historyMutex.Unlock()

	// Add to history
	m.history = append(m.history, operation)

	// Trim history if it's too large
	if len(m.history) > m.config.FailoverHistorySize {
		m.history = m.history[len(m.history)-m.config.FailoverHistorySize:]
	}
}

// InitiateFailover initiates a failover operation
func (m *FailoverManager) InitiateFailover(groupID string, newPrimaryID string, reason string, manual bool) (string, error) {
	// Get the availability group
	group, err := m.availabilityManager.GetAvailabilityGroup(groupID)
	if err != nil {
		return "", fmt.Errorf("failed to get availability group: %w", err)
	}

	// Find the current primary
	var oldPrimaryID string
	for _, service := range group.Services {
		if service.Role == ServiceRolePrimary {
			oldPrimaryID = service.ID
			break
		}
	}

	if oldPrimaryID == "" {
		return "", fmt.Errorf("no primary service found in group %s", groupID)
	}

	if oldPrimaryID == newPrimaryID {
		return "", fmt.Errorf("new primary is already the primary: %s", newPrimaryID)
	}

	// Check if the new primary exists in the group
	var newPrimaryExists bool
	for _, service := range group.Services {
		if service.ID == newPrimaryID {
			newPrimaryExists = true
			break
		}
	}

	if !newPrimaryExists {
		return "", fmt.Errorf("new primary %s not found in group %s", newPrimaryID, groupID)
	}

	// Generate a unique ID for the operation
	operationID := fmt.Sprintf("failover-%d", time.Now().UnixNano())

	// Create the failover operation
	operation := &FailoverOperation{
		ID:           operationID,
		GroupID:      groupID,
		OldPrimaryID: oldPrimaryID,
		NewPrimaryID: newPrimaryID,
		State:        FailoverStateInitiated,
		Reason:       reason,
		StartTime:    time.Now(),
		Timeout:      m.config.DefaultTimeout,
		Manual:       manual,
		Steps:        m.createFailoverSteps(groupID, oldPrimaryID, newPrimaryID),
		CurrentStep:  0,
	}

	// Store the operation
	m.operationMutex.Lock()
	m.operations[operationID] = operation
	m.operationMutex.Unlock()

	log.Printf("Initiated failover operation %s: %s -> %s", operationID, oldPrimaryID, newPrimaryID)

	return operationID, nil
}

// createFailoverSteps creates the steps for a failover operation
func (m *FailoverManager) createFailoverSteps(groupID, oldPrimaryID, newPrimaryID string) []*FailoverStep {
	steps := []*FailoverStep{
		{
			ID:          "validate",
			Name:        "Validate Failover",
			Description: "Validate that failover is possible",
			State:       StepStatePending,
			Action: func(ctx context.Context, op *FailoverOperation) error {
				// Check if the old primary is still primary
				service, err := m.availabilityManager.GetService(oldPrimaryID)
				if err != nil {
					return fmt.Errorf("failed to get old primary service: %w", err)
				}
				if service.Role != ServiceRolePrimary {
					return fmt.Errorf("old primary %s is no longer primary", oldPrimaryID)
				}

				// Check if the new primary is available
				service, err = m.availabilityManager.GetService(newPrimaryID)
				if err != nil {
					return fmt.Errorf("failed to get new primary service: %w", err)
				}
				if service.State != ServiceStateRunning {
					return fmt.Errorf("new primary %s is not running", newPrimaryID)
				}

				return nil
			},
			CanRollback: false,
			CanSkip:     false,
		},
		{
			ID:          "prepare_new_primary",
			Name:        "Prepare New Primary",
			Description: "Prepare the new primary service for promotion",
			State:       StepStatePending,
			Action: func(ctx context.Context, op *FailoverOperation) error {
				// In a real implementation, this would prepare the new primary
				// For example, ensure it's up to date with the old primary
				log.Printf("Preparing new primary %s", newPrimaryID)
				return nil
			},
			Rollback: func(ctx context.Context, op *FailoverOperation) error {
				log.Printf("Rolling back preparation of new primary %s", newPrimaryID)
				return nil
			},
			CanRollback: true,
			CanSkip:     false,
		},
		{
			ID:          "demote_old_primary",
			Name:        "Demote Old Primary",
			Description: "Demote the old primary service",
			State:       StepStatePending,
			Action: func(ctx context.Context, op *FailoverOperation) error {
				// In a real implementation, this would demote the old primary
				log.Printf("Demoting old primary %s", oldPrimaryID)

				// Update the role in the availability manager
				service, err := m.availabilityManager.GetService(oldPrimaryID)
				if err != nil {
					return fmt.Errorf("failed to get old primary service: %w", err)
				}
				service.Role = ServiceRoleSecondary

				return nil
			},
			Rollback: func(ctx context.Context, op *FailoverOperation) error {
				log.Printf("Rolling back demotion of old primary %s", oldPrimaryID)

				// Restore the role in the availability manager
				service, err := m.availabilityManager.GetService(oldPrimaryID)
				if err != nil {
					return fmt.Errorf("failed to get old primary service: %w", err)
				}
				service.Role = ServiceRolePrimary

				return nil
			},
			CanRollback: true,
			CanSkip:     false,
		},
		{
			ID:          "promote_new_primary",
			Name:        "Promote New Primary",
			Description: "Promote the new primary service",
			State:       StepStatePending,
			Action: func(ctx context.Context, op *FailoverOperation) error {
				// In a real implementation, this would promote the new primary
				log.Printf("Promoting new primary %s", newPrimaryID)

				// Update the role in the availability manager
				service, err := m.availabilityManager.GetService(newPrimaryID)
				if err != nil {
					return fmt.Errorf("failed to get new primary service: %w", err)
				}
				service.Role = ServiceRolePrimary

				return nil
			},
			Rollback: func(ctx context.Context, op *FailoverOperation) error {
				log.Printf("Rolling back promotion of new primary %s", newPrimaryID)

				// Restore the role in the availability manager
				service, err := m.availabilityManager.GetService(newPrimaryID)
				if err != nil {
					return fmt.Errorf("failed to get new primary service: %w", err)
				}
				service.Role = ServiceRoleSecondary

				return nil
			},
			CanRollback: true,
			CanSkip:     false,
		},
		{
			ID:          "update_secondary_services",
			Name:        "Update Secondary Services",
			Description: "Update secondary services to follow the new primary",
			State:       StepStatePending,
			Action: func(ctx context.Context, op *FailoverOperation) error {
				// In a real implementation, this would update secondary services
				log.Printf("Updating secondary services to follow new primary %s", newPrimaryID)
				return nil
			},
			Rollback: func(ctx context.Context, op *FailoverOperation) error {
				log.Printf("Rolling back update of secondary services")
				return nil
			},
			CanRollback: true,
			CanSkip:     true,
		},
		{
			ID:          "verify_failover",
			Name:        "Verify Failover",
			Description: "Verify that failover was successful",
			State:       StepStatePending,
			Action: func(ctx context.Context, op *FailoverOperation) error {
				// In a real implementation, this would verify the failover
				log.Printf("Verifying failover to new primary %s", newPrimaryID)

				// Check if the new primary has the primary role
				service, err := m.availabilityManager.GetService(newPrimaryID)
				if err != nil {
					return fmt.Errorf("failed to get new primary service: %w", err)
				}
				if service.Role != ServiceRolePrimary {
					return fmt.Errorf("new primary %s does not have primary role", newPrimaryID)
				}

				return nil
			},
			CanRollback: false,
			CanSkip:     false,
		},
	}

	return steps
}

// GetOperation gets a failover operation by ID
func (m *FailoverManager) GetOperation(operationID string) (*FailoverOperation, error) {
	m.operationMutex.RLock()
	defer m.operationMutex.RUnlock()

	operation, exists := m.operations[operationID]
	if !exists {
		return nil, fmt.Errorf("operation not found: %s", operationID)
	}

	return operation, nil
}

// GetOperationHistory gets the failover operation history
func (m *FailoverManager) GetOperationHistory() []*FailoverOperation {
	m.historyMutex.RLock()
	defer m.historyMutex.RUnlock()

	// Return a copy of the history
	history := make([]*FailoverOperation, len(m.history))
	copy(history, m.history)

	return history
}

// CancelOperation cancels a failover operation
func (m *FailoverManager) CancelOperation(operationID string) error {
	m.operationMutex.Lock()
	defer m.operationMutex.Unlock()

	operation, exists := m.operations[operationID]
	if !exists {
		return fmt.Errorf("operation not found: %s", operationID)
	}

	if operation.State != FailoverStateInitiated && operation.State != FailoverStateInProgress {
		return fmt.Errorf("operation cannot be cancelled: %s", operation.State)
	}

	operation.State = FailoverStateCancelled
	log.Printf("Cancelled failover operation %s", operationID)

	return nil
}

// RegisterOperationCallback registers a callback for operation updates
func (m *FailoverManager) RegisterOperationCallback(operationID string, callback func(*FailoverOperation)) {
	m.operationCallbackMutex.Lock()
	defer m.operationCallbackMutex.Unlock()

	if callbacks, exists := m.operationCallbacks[operationID]; exists {
		m.operationCallbacks[operationID] = append(callbacks, callback)
	} else {
		m.operationCallbacks[operationID] = []func(*FailoverOperation){callback}
	}
}

// SelectNewPrimary selects a new primary service based on the failover policy
func (m *FailoverManager) SelectNewPrimary(groupID string, excludeID string) (string, error) {
	// Get the availability group
	group, err := m.availabilityManager.GetAvailabilityGroup(groupID)
	if err != nil {
		return "", fmt.Errorf("failed to get availability group: %w", err)
	}

	// Select a new primary based on the failover policy
	switch m.config.FailoverPolicy {
	case FailoverPolicyPriority:
		return m.selectByPriority(group, excludeID)
	case FailoverPolicyRoundRobin:
		return m.selectByRoundRobin(group, excludeID)
	case FailoverPolicyLeastRecent:
		return m.selectByLeastRecent(group, excludeID)
	case FailoverPolicyMostHealthy:
		return m.selectByMostHealthy(group, excludeID)
	case FailoverPolicyCustom:
		if m.customSelectFunction == nil {
			return "", fmt.Errorf("custom select function not set")
		}
		return m.customSelectFunction(group, excludeID)
	default:
		return m.selectByPriority(group, excludeID)
	}
}

// selectByPriority selects a new primary based on priority
func (m *FailoverManager) selectByPriority(group *AvailabilityGroup, excludeID string) (string, error) {
	var highestPriority int = -1
	var selectedID string

	for _, service := range group.Services {
		if service.ID == excludeID {
			continue
		}

		// Only consider running services
		if service.State != ServiceStateRunning && service.State != ServiceStateDegraded {
			continue
		}

		if service.Priority > highestPriority {
			highestPriority = service.Priority
			selectedID = service.ID
		}
	}

	if selectedID == "" {
		return "", fmt.Errorf("no suitable service found for failover")
	}

	return selectedID, nil
}

// selectByRoundRobin selects a new primary in a round-robin fashion
func (m *FailoverManager) selectByRoundRobin(group *AvailabilityGroup, excludeID string) (string, error) {
	// Get the current primary
	var currentPrimaryIdx int = -1
	var runningServices []*ServiceInstance

	for _, service := range group.Services {
		// Only consider running services
		if service.State != ServiceStateRunning && service.State != ServiceStateDegraded {
			continue
		}
		if service.ID != excludeID {
			runningServices = append(runningServices, service)
		}
		if service.Role == ServiceRolePrimary {
			currentPrimaryIdx = len(runningServices) - 1
		}
	}

	if len(runningServices) == 0 {
		return "", fmt.Errorf("no suitable service found for failover")
	}

	// Select the next service in a round-robin fashion
	nextIdx := (currentPrimaryIdx + 1) % len(runningServices)
	return runningServices[nextIdx].ID, nil
}

// selectByLeastRecent selects the service that was primary least recently
func (m *FailoverManager) selectByLeastRecent(group *AvailabilityGroup, excludeID string) (string, error) {
	// This would require tracking when each service was last primary
	// For now, we'll just use priority as a fallback
	return m.selectByPriority(group, excludeID)
}

// selectByMostHealthy selects the healthiest service
func (m *FailoverManager) selectByMostHealthy(group *AvailabilityGroup, excludeID string) (string, error) {
	// In a real implementation, this would evaluate the health of each service
	// For now, we'll just select the first running service that's not excluded
	for _, service := range group.Services {
		if service.ID == excludeID {
			continue
		}

		// Only consider running services
		if service.State == ServiceStateRunning {
			return service.ID, nil
		}
	}

	// Fallback to any service in a degraded state
	for _, service := range group.Services {
		if service.ID == excludeID {
			continue
		}

		if service.State == ServiceStateDegraded {
			return service.ID, nil
		}
	}

	return "", fmt.Errorf("no suitable service found for failover")
}

// SetCustomSelectFunction sets a custom function for selecting a new primary
func (m *FailoverManager) SetCustomSelectFunction(fn func(*AvailabilityGroup, string) (string, error)) {
	m.customSelectFunction = fn
}

// GetFailoverPolicy gets the current failover policy
func (m *FailoverManager) GetFailoverPolicy() FailoverPolicy {
	return m.config.FailoverPolicy
}

// SetFailoverPolicy sets the failover policy
func (m *FailoverManager) SetFailoverPolicy(policy FailoverPolicy) {
	m.config.FailoverPolicy = policy
}
