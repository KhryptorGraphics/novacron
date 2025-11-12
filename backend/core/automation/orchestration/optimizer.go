package orchestration

import (
	"context"
	"fmt"
	"sort"
	"time"

	"go.uber.org/zap"
)

// WorkflowOptimizer optimizes workflow structure and execution
type WorkflowOptimizer struct {
	logger *zap.Logger
}

// ExecutionStage represents a stage in the execution plan
type ExecutionStage struct {
	ID       string          `json:"id"`
	Steps    []*WorkflowStep `json:"steps"`
	Parallel bool            `json:"parallel"`
	Order    int             `json:"order"`
}

// ExecutionPlan represents an optimized execution plan
type ExecutionPlan struct {
	WorkflowID    string            `json:"workflow_id"`
	Stages        []*ExecutionStage `json:"stages"`
	EstimatedTime time.Duration     `json:"estimated_time"`
	CreatedAt     time.Time         `json:"created_at"`
}

// Optimization represents a suggested optimization
type Optimization struct {
	Type        string  `json:"type"`
	Description string  `json:"description"`
	Impact      float64 `json:"impact"`
	Priority    int     `json:"priority"`
}

// NewWorkflowOptimizer creates a new workflow optimizer
func NewWorkflowOptimizer(logger *zap.Logger) *WorkflowOptimizer {
	return &WorkflowOptimizer{
		logger: logger,
	}
}

// OptimizeWorkflow optimizes a workflow structure
func (o *WorkflowOptimizer) OptimizeWorkflow(ctx context.Context, workflow *Workflow) (*Workflow, error) {
	optimized := *workflow

	// 1. Identify parallelization opportunities
	parallelGroups := o.identifyParallelSteps(workflow.Steps)

	// 2. Optimize step ordering for minimal latency
	optimized.Steps = o.optimizeStepOrder(workflow.Steps)

	// 3. Add intelligent caching
	o.addCachingHints(&optimized)

	// 4. Calculate optimization score
	optimized.OptimizationScore = o.calculateOptimizationScore(&optimized)

	o.logger.Info("Workflow optimized",
		zap.String("workflow_id", workflow.ID),
		zap.Int("parallel_groups", len(parallelGroups)),
		zap.Float64("score", optimized.OptimizationScore))

	return &optimized, nil
}

// identifyParallelSteps identifies steps that can run in parallel
func (o *WorkflowOptimizer) identifyParallelSteps(steps []*WorkflowStep) [][]string {
	var groups [][]string

	// Build dependency graph
	graph := make(map[string][]string)
	for _, step := range steps {
		graph[step.ID] = step.Dependencies
	}

	// Find independent steps
	independent := make(map[string]bool)
	for _, step := range steps {
		if len(step.Dependencies) == 0 {
			independent[step.ID] = true
		}
	}

	// Group independent steps
	if len(independent) > 1 {
		group := make([]string, 0, len(independent))
		for id := range independent {
			group = append(group, id)
		}
		groups = append(groups, group)
	}

	return groups
}

// optimizeStepOrder reorders steps for optimal execution
func (o *WorkflowOptimizer) optimizeStepOrder(steps []*WorkflowStep) []*WorkflowStep {
	// Topological sort based on dependencies
	sorted := make([]*WorkflowStep, 0, len(steps))
	visited := make(map[string]bool)
	temp := make(map[string]bool)

	stepMap := make(map[string]*WorkflowStep)
	for _, step := range steps {
		stepMap[step.ID] = step
	}

	var visit func(string) error
	visit = func(id string) error {
		if temp[id] {
			return fmt.Errorf("circular dependency detected")
		}
		if visited[id] {
			return nil
		}

		temp[id] = true
		step := stepMap[id]

		for _, depID := range step.Dependencies {
			if err := visit(depID); err != nil {
				return err
			}
		}

		temp[id] = false
		visited[id] = true
		sorted = append(sorted, step)

		return nil
	}

	for _, step := range steps {
		if !visited[step.ID] {
			if err := visit(step.ID); err != nil {
				o.logger.Error("Failed to optimize step order", zap.Error(err))
				return steps // Return original on error
			}
		}
	}

	return sorted
}

// addCachingHints adds caching hints to steps
func (o *WorkflowOptimizer) addCachingHints(workflow *Workflow) {
	for _, step := range workflow.Steps {
		// Add cache hints for idempotent operations
		if o.isIdempotent(step) {
			if step.Inputs == nil {
				step.Inputs = make(map[string]interface{})
			}
			step.Inputs["_cache_enabled"] = true
		}
	}
}

// isIdempotent checks if a step is idempotent
func (o *WorkflowOptimizer) isIdempotent(step *WorkflowStep) bool {
	// Simple heuristic - can be enhanced
	idempotentTypes := map[StepType]bool{
		StepTypeAPI:  true,
		StepTypeTask: true,
	}
	return idempotentTypes[step.Type]
}

// calculateOptimizationScore calculates an optimization score
func (o *WorkflowOptimizer) calculateOptimizationScore(workflow *Workflow) float64 {
	score := 100.0

	// Factors that reduce score
	for _, step := range workflow.Steps {
		// No retry policy
		if step.RetryPolicy == nil {
			score -= 5.0
		}

		// No timeout
		if step.Timeout == 0 {
			score -= 3.0
		}

		// Long dependency chains
		if len(step.Dependencies) > 3 {
			score -= 2.0
		}
	}

	// Factors that increase score
	parallelCount := 0
	for _, step := range workflow.Steps {
		if step.Parallel {
			parallelCount++
		}
	}
	score += float64(parallelCount) * 2.0

	if score < 0 {
		score = 0
	}
	if score > 100 {
		score = 100
	}

	return score
}

// DependencyScheduler creates execution plans with dependency resolution
type DependencyScheduler struct {
	logger *zap.Logger
}

// NewDependencyScheduler creates a new dependency scheduler
func NewDependencyScheduler(logger *zap.Logger) *DependencyScheduler {
	return &DependencyScheduler{
		logger: logger,
	}
}

// CreateExecutionPlan creates an optimized execution plan
func (s *DependencyScheduler) CreateExecutionPlan(ctx context.Context, workflow *Workflow) (*ExecutionPlan, error) {
	plan := &ExecutionPlan{
		WorkflowID: workflow.ID,
		CreatedAt:  time.Now(),
	}

	// Build dependency graph
	graph := s.buildDependencyGraph(workflow.Steps)

	// Perform topological sort and group by levels
	levels, err := s.topologicalSort(graph)
	if err != nil {
		return nil, err
	}

	// Create stages from levels
	for i, level := range levels {
		stage := &ExecutionStage{
			ID:       fmt.Sprintf("stage-%d", i),
			Steps:    level,
			Parallel: len(level) > 1,
			Order:    i,
		}
		plan.Stages = append(plan.Stages, stage)
	}

	// Estimate execution time
	plan.EstimatedTime = s.estimateExecutionTime(plan.Stages)

	s.logger.Info("Execution plan created",
		zap.String("workflow_id", workflow.ID),
		zap.Int("stages", len(plan.Stages)),
		zap.Duration("estimated_time", plan.EstimatedTime))

	return plan, nil
}

// buildDependencyGraph builds a dependency graph
func (s *DependencyScheduler) buildDependencyGraph(steps []*WorkflowStep) map[string]*WorkflowStep {
	graph := make(map[string]*WorkflowStep)
	for _, step := range steps {
		graph[step.ID] = step
	}
	return graph
}

// topologicalSort performs topological sort and groups by levels
func (s *DependencyScheduler) topologicalSort(graph map[string]*WorkflowStep) ([][]*WorkflowStep, error) {
	var levels [][]*WorkflowStep

	inDegree := make(map[string]int)
	for id, step := range graph {
		if _, exists := inDegree[id]; !exists {
			inDegree[id] = 0
		}
		for _, depID := range step.Dependencies {
			inDegree[depID]++
		}
	}

	// Process nodes level by level
	for len(inDegree) > 0 {
		var currentLevel []*WorkflowStep

		// Find nodes with no incoming edges
		for id, degree := range inDegree {
			if degree == 0 {
				currentLevel = append(currentLevel, graph[id])
			}
		}

		if len(currentLevel) == 0 {
			return nil, fmt.Errorf("circular dependency detected")
		}

		// Remove processed nodes and update degrees
		for _, step := range currentLevel {
			delete(inDegree, step.ID)
			for _, depID := range step.Dependencies {
				if degree, exists := inDegree[depID]; exists {
					inDegree[depID] = degree - 1
				}
			}
		}

		levels = append(levels, currentLevel)
	}

	return levels, nil
}

// estimateExecutionTime estimates total execution time
func (s *DependencyScheduler) estimateExecutionTime(stages []*ExecutionStage) time.Duration {
	var total time.Duration

	for _, stage := range stages {
		var stageTime time.Duration

		if stage.Parallel {
			// For parallel execution, take the maximum timeout
			for _, step := range stage.Steps {
				if step.Timeout > stageTime {
					stageTime = step.Timeout
				}
			}
		} else {
			// For sequential execution, sum all timeouts
			for _, step := range stage.Steps {
				stageTime += step.Timeout
			}
		}

		total += stageTime
	}

	return total
}

// LearningEngine learns from workflow executions
type LearningEngine struct {
	logger      *zap.Logger
	history     []*WorkflowExecution
	patterns    map[string]*ExecutionPattern
	mu          sync.RWMutex
}

// ExecutionPattern represents learned execution patterns
type ExecutionPattern struct {
	WorkflowID      string        `json:"workflow_id"`
	AverageDuration time.Duration `json:"average_duration"`
	SuccessRate     float64       `json:"success_rate"`
	CommonFailures  []string      `json:"common_failures"`
	OptimalSteps    []*WorkflowStep `json:"optimal_steps"`
	Timestamp       time.Time     `json:"timestamp"`
}

// NewLearningEngine creates a new learning engine
func NewLearningEngine(logger *zap.Logger) *LearningEngine {
	return &LearningEngine{
		logger:   logger,
		history:  make([]*WorkflowExecution, 0),
		patterns: make(map[string]*ExecutionPattern),
	}
}

// RecordExecution records a workflow execution for learning
func (l *LearningEngine) RecordExecution(ctx context.Context, workflow *Workflow, execution *WorkflowExecution) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.history = append(l.history, execution)

	// Update patterns
	pattern := l.patterns[workflow.ID]
	if pattern == nil {
		pattern = &ExecutionPattern{
			WorkflowID:     workflow.ID,
			CommonFailures: make([]string, 0),
		}
		l.patterns[workflow.ID] = pattern
	}

	// Update statistics
	l.updatePattern(pattern, execution)

	l.logger.Info("Recorded execution for learning",
		zap.String("workflow_id", workflow.ID),
		zap.Float64("success_rate", pattern.SuccessRate))
}

// SuggestOptimizations suggests optimizations based on learned patterns
func (l *LearningEngine) SuggestOptimizations(ctx context.Context, workflow *Workflow, execution *WorkflowExecution) []*Optimization {
	l.mu.RLock()
	defer l.mu.RUnlock()

	optimizations := make([]*Optimization, 0)

	pattern := l.patterns[workflow.ID]
	if pattern == nil {
		return optimizations
	}

	// Suggest based on historical data
	if pattern.SuccessRate < 0.95 {
		optimizations = append(optimizations, &Optimization{
			Type:        "retry_policy",
			Description: "Add retry policies to improve success rate",
			Impact:      (0.95 - pattern.SuccessRate) * 100,
			Priority:    1,
		})
	}

	// Sort by priority
	sort.Slice(optimizations, func(i, j int) bool {
		return optimizations[i].Priority < optimizations[j].Priority
	})

	return optimizations
}

// updatePattern updates execution patterns
func (l *LearningEngine) updatePattern(pattern *ExecutionPattern, execution *WorkflowExecution) {
	// Update average duration
	if pattern.AverageDuration == 0 {
		pattern.AverageDuration = execution.Metrics.TotalDuration
	} else {
		pattern.AverageDuration = (pattern.AverageDuration + execution.Metrics.TotalDuration) / 2
	}

	// Update success rate (simple moving average)
	successValue := 0.0
	if execution.Status == ExecutionStatusCompleted {
		successValue = 1.0
	}

	if pattern.SuccessRate == 0 {
		pattern.SuccessRate = successValue
	} else {
		pattern.SuccessRate = (pattern.SuccessRate*0.9 + successValue*0.1)
	}

	// Record common failures
	if execution.Error != "" {
		pattern.CommonFailures = append(pattern.CommonFailures, execution.Error)
		if len(pattern.CommonFailures) > 10 {
			pattern.CommonFailures = pattern.CommonFailures[1:]
		}
	}

	pattern.Timestamp = time.Now()
}
