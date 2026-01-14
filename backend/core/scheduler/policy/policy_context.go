package policy

import (
	"fmt"
	"sync"
)

// EvaluationContext is the base context for expression evaluation
type EvaluationContext struct {
	// VM represents the virtual machine being placed or migrated
	VM map[string]interface{}

	// CandidateNode represents a candidate node for placement
	CandidateNode map[string]interface{}

	// SourceNode represents the source node in a migration
	SourceNode map[string]interface{}

	// Variables for expression evaluation
	Variables map[string]interface{}

	// Score accumulates scoring adjustments
	Score float64

	// ScoreComponents tracks individual score adjustments
	ScoreComponents map[string]float64

	// Filtered indicates if the candidate should be filtered out
	Filtered bool

	// FilterReason explains why the candidate was filtered
	FilterReason string

	// Metrics data for evaluation
	Metrics map[string]interface{}

	// mutex for concurrent access
	mu sync.RWMutex
}

// NewEvaluationContext creates a new evaluation context
func NewEvaluationContext() *EvaluationContext {
	return &EvaluationContext{
		Variables:       make(map[string]interface{}),
		Score:           0,
		ScoreComponents: make(map[string]float64),
		Filtered:        false,
		FilterReason:    "",
		Metrics:         make(map[string]interface{}),
	}
}

// AddScore adds to the total score with a reason
func (ctx *EvaluationContext) AddScore(score float64, reason string) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	ctx.Score += score
	ctx.ScoreComponents[reason] = score
}

// SetFiltered marks a candidate as filtered out
func (ctx *EvaluationContext) SetFiltered(filtered bool, reason string) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	ctx.Filtered = filtered
	ctx.FilterReason = reason
}

// GetScore returns the current score
func (ctx *EvaluationContext) GetScore() float64 {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	return ctx.Score
}

// IsFiltered checks if the candidate is filtered out
func (ctx *EvaluationContext) IsFiltered() (bool, string) {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	return ctx.Filtered, ctx.FilterReason
}

// SetVariable sets a variable for expression evaluation
func (ctx *EvaluationContext) SetVariable(name string, value interface{}) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	ctx.Variables[name] = value
}

// GetVariable gets a variable value
func (ctx *EvaluationContext) GetVariable(name string) (interface{}, bool) {
	ctx.mu.RLock()
	defer ctx.mu.RUnlock()

	value, exists := ctx.Variables[name]
	return value, exists
}

// InitializeForVM initializes the context for a VM and candidate node evaluation
func (ctx *EvaluationContext) InitializeForVM(vm, sourceNode, candidateNode map[string]interface{}) {
	ctx.mu.Lock()
	defer ctx.mu.Unlock()

	ctx.VM = vm
	ctx.SourceNode = sourceNode
	ctx.CandidateNode = candidateNode
	ctx.Score = 0
	ctx.Filtered = false
	ctx.FilterReason = ""
	// Don't clear Variables as they might contain parameters
}

// GetParameterValue retrieves a parameter value from the context
func GetParameterValue(ctx *EvaluationContext, name string) (interface{}, bool) {
	paramKey := fmt.Sprintf("param.%s", name)
	value, found := ctx.GetVariable(paramKey)
	return value, found
}

// PolicyEvaluationContext extends EvaluationContext with policy-specific functionality
type PolicyEvaluationContext struct {
	// Parameters contains parameter values for the policy
	Parameters map[string]interface{}

	// Custom attributes that can be set during evaluation
	Attributes map[string]interface{}

	// Base context for delegation
	*EvaluationContext
}

// NewPolicyEvaluationContext creates a new policy evaluation context
func NewPolicyEvaluationContext(vm, sourceNode, candidateNode map[string]interface{}) *PolicyEvaluationContext {
	base := NewEvaluationContext()
	base.InitializeForVM(vm, sourceNode, candidateNode)

	return &PolicyEvaluationContext{
		EvaluationContext: base,
		Parameters:        make(map[string]interface{}),
		Attributes:        make(map[string]interface{}),
	}
}

// SetAttribute sets a custom attribute in the evaluation context
func (ctx *PolicyEvaluationContext) SetAttribute(key string, value interface{}) {
	ctx.EvaluationContext.mu.Lock()
	defer ctx.EvaluationContext.mu.Unlock()

	ctx.Attributes[key] = value
}

// GetAttribute gets a custom attribute from the evaluation context
func (ctx *PolicyEvaluationContext) GetAttribute(key string) (interface{}, bool) {
	ctx.EvaluationContext.mu.RLock()
	defer ctx.EvaluationContext.mu.RUnlock()

	val, ok := ctx.Attributes[key]
	return val, ok
}

// SetParameter sets a policy parameter
func (ctx *PolicyEvaluationContext) SetParameter(name string, value interface{}) {
	ctx.EvaluationContext.mu.Lock()
	defer ctx.EvaluationContext.mu.Unlock()

	ctx.Parameters[name] = value
	ctx.Variables[fmt.Sprintf("param.%s", name)] = value
}

// GetParameter gets a policy parameter
func (ctx *PolicyEvaluationContext) GetParameter(name string) (interface{}, bool) {
	ctx.EvaluationContext.mu.RLock()
	defer ctx.EvaluationContext.mu.RUnlock()

	val, ok := ctx.Parameters[name]
	return val, ok
}
