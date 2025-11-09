package runbooks

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// RunbookEngine executes automated runbooks
type RunbookEngine struct {
	runbooks   map[string]*Runbook
	executions map[string]*RunbookExecution
	mu         sync.RWMutex
}

// Runbook defines an automated recovery procedure
type Runbook struct {
	ID          string
	Name        string
	Description string
	Scenario    string
	Steps       []RunbookStep
	Metadata    map[string]string
}

// RunbookStep represents a single step
type RunbookStep struct {
	ID               string
	Name             string
	Description      string
	Action           StepAction
	RequiresApproval bool
	AutoRollback     bool
	Timeout          time.Duration
	OnFailure        string // "abort", "continue", "retry"
	MaxRetries       int
}

// StepAction is a function that executes a step
type StepAction func(ctx context.Context, params map[string]interface{}) error

// RunbookExecution tracks execution
type RunbookExecution struct {
	ID              string
	RunbookID       string
	StartedAt       time.Time
	CompletedAt     time.Time
	Status          string
	CurrentStep     int
	Steps           []ExecutedStep
	Params          map[string]interface{}
	ApprovalsPending int
	ExecutedBy      string
	AuditLog        []AuditEntry
}

// ExecutedStep tracks an executed step
type ExecutedStep struct {
	StepID      string
	Status      string
	StartedAt   time.Time
	CompletedAt time.Time
	Error       error
	RetryCount  int
}

// AuditEntry logs actions
type AuditEntry struct {
	Timestamp time.Time
	Action    string
	User      string
	Details   map[string]interface{}
}

// NewRunbookEngine creates a new runbook engine
func NewRunbookEngine() *RunbookEngine {
	engine := &RunbookEngine{
		runbooks:   make(map[string]*Runbook),
		executions: make(map[string]*RunbookExecution),
	}

	// Register standard runbooks
	engine.registerStandardRunbooks()

	return engine
}

// registerStandardRunbooks registers built-in runbooks
func (re *RunbookEngine) registerStandardRunbooks() {
	re.RegisterRunbook(RegionFailureRunbook())
	re.RegisterRunbook(DataCorruptionRunbook())
	re.RegisterRunbook(NetworkPartitionRunbook())
	re.RegisterRunbook(SecurityIncidentRunbook())
}

// RegisterRunbook registers a new runbook
func (re *RunbookEngine) RegisterRunbook(runbook *Runbook) {
	re.mu.Lock()
	defer re.mu.Unlock()

	re.runbooks[runbook.ID] = runbook
	log.Printf("Registered runbook: %s - %s", runbook.ID, runbook.Name)
}

// ExecuteRunbook executes a runbook
func (re *RunbookEngine) ExecuteRunbook(ctx context.Context, runbookID string, params map[string]interface{}) (*RunbookExecution, error) {
	re.mu.RLock()
	runbook, exists := re.runbooks[runbookID]
	re.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("runbook not found: %s", runbookID)
	}

	execution := &RunbookExecution{
		ID:         fmt.Sprintf("exec-%s-%d", runbookID, time.Now().Unix()),
		RunbookID:  runbookID,
		StartedAt:  time.Now(),
		Status:     "running",
		Params:     params,
		Steps:      make([]ExecutedStep, len(runbook.Steps)),
		AuditLog:   make([]AuditEntry, 0),
		ExecutedBy: "system",
	}

	re.mu.Lock()
	re.executions[execution.ID] = execution
	re.mu.Unlock()

	// Log execution start
	execution.AuditLog = append(execution.AuditLog, AuditEntry{
		Timestamp: time.Now(),
		Action:    "runbook_started",
		User:      execution.ExecutedBy,
		Details: map[string]interface{}{
			"runbook": runbookID,
			"params":  params,
		},
	})

	// Execute steps
	go re.executeSteps(ctx, execution, runbook)

	return execution, nil
}

// executeSteps executes all runbook steps
func (re *RunbookEngine) executeSteps(ctx context.Context, execution *RunbookExecution, runbook *Runbook) {
	for i, step := range runbook.Steps {
		execution.CurrentStep = i

		execStep := &execution.Steps[i]
		execStep.StepID = step.ID
		execStep.Status = "running"
		execStep.StartedAt = time.Now()

		log.Printf("Executing step %d: %s", i+1, step.Name)

		// Check for approval if required
		if step.RequiresApproval {
			execution.ApprovalsPending++
			log.Printf("Step %s requires approval", step.Name)
			// In production, wait for approval via API/UI
			time.Sleep(1 * time.Second)
			execution.ApprovalsPending--
		}

		// Execute step with timeout
		stepCtx, cancel := context.WithTimeout(ctx, step.Timeout)

		err := re.executeStepWithRetry(stepCtx, step, execution.Params)
		cancel()

		execStep.CompletedAt = time.Now()

		if err != nil {
			execStep.Status = "failed"
			execStep.Error = err

			log.Printf("Step %s failed: %v", step.Name, err)

			// Handle failure
			switch step.OnFailure {
			case "abort":
				execution.Status = "failed"
				execution.CompletedAt = time.Now()
				return
			case "retry":
				// Already retried in executeStepWithRetry
				if execStep.RetryCount >= step.MaxRetries {
					execution.Status = "failed"
					execution.CompletedAt = time.Now()
					return
				}
			case "continue":
				log.Printf("Continuing despite failure in step: %s", step.Name)
			}
		} else {
			execStep.Status = "completed"
			log.Printf("Step %s completed successfully", step.Name)
		}

		// Log step completion
		execution.AuditLog = append(execution.AuditLog, AuditEntry{
			Timestamp: time.Now(),
			Action:    "step_completed",
			User:      execution.ExecutedBy,
			Details: map[string]interface{}{
				"step":   step.Name,
				"status": execStep.Status,
			},
		})
	}

	execution.Status = "completed"
	execution.CompletedAt = time.Now()

	log.Printf("Runbook execution completed: %s", execution.ID)
}

// executeStepWithRetry executes a step with retry logic
func (re *RunbookEngine) executeStepWithRetry(ctx context.Context, step RunbookStep, params map[string]interface{}) error {
	var lastErr error

	for attempt := 0; attempt <= step.MaxRetries; attempt++ {
		if attempt > 0 {
			log.Printf("Retrying step %s (attempt %d/%d)", step.Name, attempt, step.MaxRetries)
			time.Sleep(time.Duration(attempt) * 5 * time.Second)
		}

		err := step.Action(ctx, params)
		if err == nil {
			return nil
		}

		lastErr = err
	}

	return fmt.Errorf("step failed after %d retries: %w", step.MaxRetries, lastErr)
}

// GetExecution returns execution status
func (re *RunbookEngine) GetExecution(executionID string) (*RunbookExecution, error) {
	re.mu.RLock()
	defer re.mu.RUnlock()

	exec, exists := re.executions[executionID]
	if !exists {
		return nil, fmt.Errorf("execution not found: %s", executionID)
	}

	return exec, nil
}

// ListRunbooks returns all registered runbooks
func (re *RunbookEngine) ListRunbooks() []*Runbook {
	re.mu.RLock()
	defer re.mu.RUnlock()

	runbooks := make([]*Runbook, 0, len(re.runbooks))
	for _, rb := range re.runbooks {
		runbooks = append(runbooks, rb)
	}

	return runbooks
}
