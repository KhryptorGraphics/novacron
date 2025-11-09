package conflict

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	policyApplications = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_policy_applications_total",
		Help: "Total number of policy applications",
	}, []string{"policy_name", "result"})

	policyEvaluationLatency = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "dwcp_policy_evaluation_latency_ms",
		Help:    "Policy evaluation latency in milliseconds",
		Buckets: []float64{0.1, 0.5, 1, 5, 10},
	})
)

// ResolutionPolicy defines conflict resolution behavior
type ResolutionPolicy struct {
	Name             string
	DefaultStrategy  StrategyType
	FieldStrategies  map[string]StrategyType
	CustomRules      []ResolutionRule
	EscalationRules  []EscalationRule
	MaxAutoRetries   int
	ManualThreshold  float64
	TimeoutDuration  time.Duration
	EnableFallback   bool
	FallbackStrategy StrategyType
}

// ResolutionRule defines conditional strategy selection
type ResolutionRule struct {
	Name      string
	Condition ConditionFunc
	Strategy  StrategyType
	Priority  int
}

// EscalationRule defines when to escalate conflicts
type EscalationRule struct {
	Name      string
	Condition ConditionFunc
	Action    EscalationAction
	Priority  int
}

// ConditionFunc evaluates if a rule applies
type ConditionFunc func(*Conflict) bool

// EscalationAction defines what to do on escalation
type EscalationAction int

const (
	EscalationActionNotify EscalationAction = iota
	EscalationActionBlock
	EscalationActionManual
	EscalationActionRollback
)

func (ea EscalationAction) String() string {
	return [...]string{"Notify", "Block", "Manual", "Rollback"}[ea]
}

// PolicyManager manages resolution policies
type PolicyManager struct {
	mu               sync.RWMutex
	policies         map[string]*ResolutionPolicy
	defaultPolicy    *ResolutionPolicy
	strategyRegistry *StrategyRegistry
	detector         *ConflictDetector
	mergeEngine      *MergeEngine
}

// NewPolicyManager creates a new policy manager
func NewPolicyManager(detector *ConflictDetector, registry *StrategyRegistry, mergeEngine *MergeEngine) *PolicyManager {
	pm := &PolicyManager{
		policies:         make(map[string]*ResolutionPolicy),
		strategyRegistry: registry,
		detector:         detector,
		mergeEngine:      mergeEngine,
	}

	// Create default policy
	pm.defaultPolicy = &ResolutionPolicy{
		Name:             "default",
		DefaultStrategy:  StrategyLastWriteWins,
		FieldStrategies:  make(map[string]StrategyType),
		CustomRules:      make([]ResolutionRule, 0),
		EscalationRules:  make([]EscalationRule, 0),
		MaxAutoRetries:   3,
		ManualThreshold:  0.7,
		TimeoutDuration:  30 * time.Second,
		EnableFallback:   true,
		FallbackStrategy: StrategyLastWriteWins,
	}

	pm.registerDefaultRules()
	return pm
}

// registerDefaultRules registers common resolution rules
func (pm *PolicyManager) registerDefaultRules() {
	// High severity conflicts require manual intervention
	pm.defaultPolicy.EscalationRules = append(pm.defaultPolicy.EscalationRules, EscalationRule{
		Name: "high_severity_manual",
		Condition: func(c *Conflict) bool {
			return c.Severity >= SeverityHigh
		},
		Action:   EscalationActionManual,
		Priority: 100,
	})

	// Invariant violations trigger rollback
	pm.defaultPolicy.EscalationRules = append(pm.defaultPolicy.EscalationRules, EscalationRule{
		Name: "invariant_violation_rollback",
		Condition: func(c *Conflict) bool {
			return c.Type == ConflictTypeInvariantViolation
		},
		Action:   EscalationActionRollback,
		Priority: 200,
	})

	// Concurrent updates use multi-value register
	pm.defaultPolicy.CustomRules = append(pm.defaultPolicy.CustomRules, ResolutionRule{
		Name: "concurrent_multivalue",
		Condition: func(c *Conflict) bool {
			return c.Type == ConflictTypeConcurrentUpdate && c.Severity <= SeverityMedium
		},
		Strategy: StrategyMultiValueRegister,
		Priority: 50,
	})

	// Causal violations use consensus
	pm.defaultPolicy.CustomRules = append(pm.defaultPolicy.CustomRules, ResolutionRule{
		Name: "causal_consensus",
		Condition: func(c *Conflict) bool {
			return c.Type == ConflictTypeCausalViolation
		},
		Strategy: StrategyConsensusVote,
		Priority: 60,
	})
}

// RegisterPolicy registers a new policy
func (pm *PolicyManager) RegisterPolicy(policy *ResolutionPolicy) {
	pm.mu.Lock()
	defer pm.mu.Unlock()
	pm.policies[policy.Name] = policy
}

// GetPolicy retrieves a policy by name
func (pm *PolicyManager) GetPolicy(name string) (*ResolutionPolicy, bool) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	policy, exists := pm.policies[name]
	return policy, exists
}

// ResolveConflict resolves a conflict using the appropriate policy
func (pm *PolicyManager) ResolveConflict(ctx context.Context, conflict *Conflict, policyName string) (*ResolutionResult, error) {
	start := time.Now()
	defer func() {
		policyEvaluationLatency.Observe(float64(time.Since(start).Milliseconds()))
	}()

	// Get policy
	policy := pm.defaultPolicy
	if policyName != "" {
		if p, exists := pm.GetPolicy(policyName); exists {
			policy = p
		}
	}

	// Check escalation rules first
	for _, rule := range policy.EscalationRules {
		if rule.Condition(conflict) {
			return pm.handleEscalation(ctx, conflict, rule, policy)
		}
	}

	// Apply custom rules
	strategy := policy.DefaultStrategy
	for _, rule := range policy.CustomRules {
		if rule.Condition(conflict) {
			strategy = rule.Strategy
			break
		}
	}

	// Check field-specific strategies
	if len(conflict.AffectedFields) > 0 {
		if fieldStrategy, exists := policy.FieldStrategies[conflict.AffectedFields[0]]; exists {
			strategy = fieldStrategy
		}
	}

	// Check complexity threshold
	if conflict.ComplexityScore > policy.ManualThreshold {
		strategy = StrategyManualIntervention
	}

	// Execute resolution with retries
	var lastErr error
	for attempt := 0; attempt <= policy.MaxAutoRetries; attempt++ {
		result, err := pm.executeStrategy(ctx, conflict, strategy, policy)
		if err == nil {
			policyApplications.WithLabelValues(policy.Name, "success").Inc()
			return result, nil
		}
		lastErr = err

		// Wait before retry
		if attempt < policy.MaxAutoRetries {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Duration(attempt+1) * 100 * time.Millisecond):
			}
		}
	}

	// Fallback strategy if enabled
	if policy.EnableFallback && strategy != policy.FallbackStrategy {
		result, err := pm.executeStrategy(ctx, conflict, policy.FallbackStrategy, policy)
		if err == nil {
			policyApplications.WithLabelValues(policy.Name, "fallback_success").Inc()
			return result, nil
		}
	}

	policyApplications.WithLabelValues(policy.Name, "failure").Inc()
	return nil, fmt.Errorf("failed to resolve conflict after %d attempts: %w", policy.MaxAutoRetries, lastErr)
}

// executeStrategy executes a specific resolution strategy
func (pm *PolicyManager) executeStrategy(ctx context.Context, conflict *Conflict, strategyType StrategyType, policy *ResolutionPolicy) (*ResolutionResult, error) {
	strategy, exists := pm.strategyRegistry.GetStrategy(strategyType)
	if !exists {
		return nil, fmt.Errorf("strategy %s not found", strategyType)
	}

	if !strategy.CanResolve(conflict) {
		return nil, fmt.Errorf("strategy %s cannot resolve this conflict", strategyType)
	}

	// Apply timeout
	ctxWithTimeout, cancel := context.WithTimeout(ctx, policy.TimeoutDuration)
	defer cancel()

	return strategy.Resolve(ctxWithTimeout, conflict)
}

// handleEscalation handles conflict escalation
func (pm *PolicyManager) handleEscalation(ctx context.Context, conflict *Conflict, rule EscalationRule, policy *ResolutionPolicy) (*ResolutionResult, error) {
	switch rule.Action {
	case EscalationActionNotify:
		// Log notification
		return &ResolutionResult{
			Success:   false,
			Strategy:  StrategyManualIntervention,
			Message:   fmt.Sprintf("Escalation: %s", rule.Name),
			Timestamp: time.Now(),
		}, nil

	case EscalationActionBlock:
		return nil, fmt.Errorf("conflict blocked by escalation rule: %s", rule.Name)

	case EscalationActionManual:
		return pm.executeStrategy(ctx, conflict, StrategyManualIntervention, policy)

	case EscalationActionRollback:
		return pm.executeStrategy(ctx, conflict, StrategyAutomaticRollback, policy)

	default:
		return nil, fmt.Errorf("unknown escalation action: %v", rule.Action)
	}
}

// PolicyBuilder helps build policies fluently
type PolicyBuilder struct {
	policy *ResolutionPolicy
}

// NewPolicyBuilder creates a new policy builder
func NewPolicyBuilder(name string) *PolicyBuilder {
	return &PolicyBuilder{
		policy: &ResolutionPolicy{
			Name:             name,
			DefaultStrategy:  StrategyLastWriteWins,
			FieldStrategies:  make(map[string]StrategyType),
			CustomRules:      make([]ResolutionRule, 0),
			EscalationRules:  make([]EscalationRule, 0),
			MaxAutoRetries:   3,
			ManualThreshold:  0.7,
			TimeoutDuration:  30 * time.Second,
			EnableFallback:   true,
			FallbackStrategy: StrategyLastWriteWins,
		},
	}
}

// WithDefaultStrategy sets the default strategy
func (pb *PolicyBuilder) WithDefaultStrategy(strategy StrategyType) *PolicyBuilder {
	pb.policy.DefaultStrategy = strategy
	return pb
}

// WithFieldStrategy sets a field-specific strategy
func (pb *PolicyBuilder) WithFieldStrategy(field string, strategy StrategyType) *PolicyBuilder {
	pb.policy.FieldStrategies[field] = strategy
	return pb
}

// WithRule adds a custom resolution rule
func (pb *PolicyBuilder) WithRule(name string, condition ConditionFunc, strategy StrategyType, priority int) *PolicyBuilder {
	pb.policy.CustomRules = append(pb.policy.CustomRules, ResolutionRule{
		Name:      name,
		Condition: condition,
		Strategy:  strategy,
		Priority:  priority,
	})
	return pb
}

// WithEscalation adds an escalation rule
func (pb *PolicyBuilder) WithEscalation(name string, condition ConditionFunc, action EscalationAction, priority int) *PolicyBuilder {
	pb.policy.EscalationRules = append(pb.policy.EscalationRules, EscalationRule{
		Name:      name,
		Condition: condition,
		Action:    action,
		Priority:  priority,
	})
	return pb
}

// WithMaxRetries sets maximum retry attempts
func (pb *PolicyBuilder) WithMaxRetries(retries int) *PolicyBuilder {
	pb.policy.MaxAutoRetries = retries
	return pb
}

// WithManualThreshold sets complexity threshold for manual intervention
func (pb *PolicyBuilder) WithManualThreshold(threshold float64) *PolicyBuilder {
	pb.policy.ManualThreshold = threshold
	return pb
}

// WithTimeout sets resolution timeout
func (pb *PolicyBuilder) WithTimeout(duration time.Duration) *PolicyBuilder {
	pb.policy.TimeoutDuration = duration
	return pb
}

// WithFallback enables fallback strategy
func (pb *PolicyBuilder) WithFallback(strategy StrategyType) *PolicyBuilder {
	pb.policy.EnableFallback = true
	pb.policy.FallbackStrategy = strategy
	return pb
}

// Build returns the constructed policy
func (pb *PolicyBuilder) Build() *ResolutionPolicy {
	return pb.policy
}

// Common condition functions
func IsSeverity(severity ConflictSeverity) ConditionFunc {
	return func(c *Conflict) bool {
		return c.Severity == severity
	}
}

func IsType(conflictType ConflictType) ConditionFunc {
	return func(c *Conflict) bool {
		return c.Type == conflictType
	}
}

func HasField(field string) ConditionFunc {
	return func(c *Conflict) bool {
		for _, f := range c.AffectedFields {
			if f == field {
				return true
			}
		}
		return false
	}
}

func ComplexityAbove(threshold float64) ConditionFunc {
	return func(c *Conflict) bool {
		return c.ComplexityScore > threshold
	}
}

func ComplexityBelow(threshold float64) ConditionFunc {
	return func(c *Conflict) bool {
		return c.ComplexityScore <= threshold
	}
}

func And(conditions ...ConditionFunc) ConditionFunc {
	return func(c *Conflict) bool {
		for _, cond := range conditions {
			if !cond(c) {
				return false
			}
		}
		return true
	}
}

func Or(conditions ...ConditionFunc) ConditionFunc {
	return func(c *Conflict) bool {
		for _, cond := range conditions {
			if cond(c) {
				return true
			}
		}
		return false
	}
}

func Not(condition ConditionFunc) ConditionFunc {
	return func(c *Conflict) bool {
		return !condition(c)
	}
}
