// Package zerotrust implements zero-trust security architecture
package zerotrust

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// TrustLevel represents the trust level of an entity
type TrustLevel int

const (
	TrustNone TrustLevel = iota
	TrustLow
	TrustMedium
	TrustHigh
	TrustCritical
)

// TrustContext represents the context for trust evaluation
type TrustContext struct {
	UserID       string
	DeviceID     string
	Location     string
	IPAddress    string
	Timestamp    time.Time
	ResourceType string
	ResourceID   string
	Action       string
	Metadata     map[string]interface{}
}

// TrustDecision represents a trust decision
type TrustDecision struct {
	Allowed       bool
	TrustLevel    TrustLevel
	TrustScore    float64
	Reason        string
	Conditions    []string
	ExpiresAt     time.Time
	RequiresMFA   bool
	RequiresStepUp bool
}

// TrustPolicy represents a zero-trust policy
type TrustPolicy struct {
	ID                string
	Name              string
	Enabled           bool
	Priority          int
	Conditions        []PolicyCondition
	Actions           []PolicyAction
	RequiresMFA       bool
	RequiresStepUp    bool
	MaxTrustDuration  time.Duration
	ContinuousVerify  bool
}

// PolicyCondition represents a policy condition
type PolicyCondition struct {
	Type     string // "user", "device", "location", "time", "resource", "behavior"
	Operator string // "equals", "contains", "matches", "in", "not_in"
	Value    interface{}
}

// PolicyAction represents a policy action
type PolicyAction struct {
	Type   string // "allow", "deny", "require_mfa", "require_stepup", "alert"
	Params map[string]interface{}
}

// Engine implements the zero-trust architecture engine
type Engine struct {
	policies          map[string]*TrustPolicy
	trustCache        map[string]*TrustDecision
	behaviorAnalyzer  BehaviorAnalyzer
	contextEvaluator  ContextEvaluator
	mu                sync.RWMutex
	continuousVerify  bool
	verifyInterval    time.Duration
	microSegmentation bool
	leastPrivilege    bool
}

// BehaviorAnalyzer analyzes user/device behavior
type BehaviorAnalyzer interface {
	AnalyzeBehavior(ctx context.Context, trustCtx *TrustContext) (float64, error)
	GetBaseline(entityID string) (*BehaviorBaseline, error)
	UpdateBaseline(entityID string, behavior *BehaviorData) error
}

// BehaviorBaseline represents normal behavior baseline
type BehaviorBaseline struct {
	EntityID         string
	TypicalLocations []string
	TypicalDevices   []string
	TypicalHours     []int
	TypicalActions   map[string]int
	LastUpdated      time.Time
}

// BehaviorData represents behavior data
type BehaviorData struct {
	Location  string
	Device    string
	Hour      int
	Actions   []string
	Timestamp time.Time
}

// ContextEvaluator evaluates trust context
type ContextEvaluator interface {
	EvaluateContext(ctx context.Context, trustCtx *TrustContext) (float64, error)
	ValidateDevice(deviceID string) (bool, error)
	ValidateLocation(location string, allowed []string) (bool, error)
}

// NewEngine creates a new zero-trust engine
func NewEngine() *Engine {
	return &Engine{
		policies:          make(map[string]*TrustPolicy),
		trustCache:        make(map[string]*TrustDecision),
		continuousVerify:  true,
		verifyInterval:    15 * time.Minute,
		microSegmentation: true,
		leastPrivilege:    true,
	}
}

// SetBehaviorAnalyzer sets the behavior analyzer
func (e *Engine) SetBehaviorAnalyzer(analyzer BehaviorAnalyzer) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.behaviorAnalyzer = analyzer
}

// SetContextEvaluator sets the context evaluator
func (e *Engine) SetContextEvaluator(evaluator ContextEvaluator) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.contextEvaluator = evaluator
}

// AddPolicy adds a zero-trust policy
func (e *Engine) AddPolicy(policy *TrustPolicy) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if policy.ID == "" {
		return fmt.Errorf("policy ID is required")
	}

	e.policies[policy.ID] = policy
	return nil
}

// RemovePolicy removes a policy
func (e *Engine) RemovePolicy(policyID string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.policies, policyID)
}

// Evaluate evaluates trust and makes access decision
func (e *Engine) Evaluate(ctx context.Context, trustCtx *TrustContext) (*TrustDecision, error) {
	// Check cache first
	cacheKey := e.getCacheKey(trustCtx)
	if decision := e.getCachedDecision(cacheKey); decision != nil {
		if time.Now().Before(decision.ExpiresAt) {
			return decision, nil
		}
	}

	// Evaluate context
	contextScore := 1.0
	if e.contextEvaluator != nil {
		score, err := e.contextEvaluator.EvaluateContext(ctx, trustCtx)
		if err != nil {
			return nil, fmt.Errorf("context evaluation failed: %w", err)
		}
		contextScore = score
	}

	// Analyze behavior
	behaviorScore := 1.0
	if e.behaviorAnalyzer != nil {
		score, err := e.behaviorAnalyzer.AnalyzeBehavior(ctx, trustCtx)
		if err != nil {
			return nil, fmt.Errorf("behavior analysis failed: %w", err)
		}
		behaviorScore = score
	}

	// Evaluate policies
	decision := &TrustDecision{
		Allowed:    false,
		TrustLevel: TrustNone,
		TrustScore: 0,
		ExpiresAt:  time.Now().Add(e.verifyInterval),
	}

	// Get applicable policies
	policies := e.getApplicablePolicies(trustCtx)

	for _, policy := range policies {
		if !policy.Enabled {
			continue
		}

		// Check all conditions
		conditionsMet := true
		for _, condition := range policy.Conditions {
			if !e.evaluateCondition(condition, trustCtx) {
				conditionsMet = false
				break
			}
		}

		if conditionsMet {
			// Apply actions
			for _, action := range policy.Actions {
				e.applyAction(action, decision)
			}

			// Set requirements from policy
			if policy.RequiresMFA {
				decision.RequiresMFA = true
			}
			if policy.RequiresStepUp {
				decision.RequiresStepUp = true
			}

			// Set expiration
			if policy.MaxTrustDuration > 0 {
				expiresAt := time.Now().Add(policy.MaxTrustDuration)
				if expiresAt.Before(decision.ExpiresAt) {
					decision.ExpiresAt = expiresAt
				}
			}
		}
	}

	// Calculate final trust score
	decision.TrustScore = (contextScore + behaviorScore) / 2.0

	// Determine trust level
	switch {
	case decision.TrustScore >= 0.9:
		decision.TrustLevel = TrustCritical
	case decision.TrustScore >= 0.7:
		decision.TrustLevel = TrustHigh
	case decision.TrustScore >= 0.5:
		decision.TrustLevel = TrustMedium
	case decision.TrustScore >= 0.3:
		decision.TrustLevel = TrustLow
	default:
		decision.TrustLevel = TrustNone
	}

	// Apply least privilege
	if e.leastPrivilege && decision.TrustLevel < TrustMedium {
		decision.Allowed = false
		decision.Reason = "Insufficient trust level for least privilege access"
	}

	// Cache decision
	e.cacheDecision(cacheKey, decision)

	return decision, nil
}

// ContinuousVerification performs continuous verification of trust
func (e *Engine) ContinuousVerification(ctx context.Context) {
	ticker := time.NewTicker(e.verifyInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			e.verifyActiveSessions(ctx)
		}
	}
}

// verifyActiveSessions verifies all active sessions
func (e *Engine) verifyActiveSessions(ctx context.Context) {
	e.mu.Lock()
	defer e.mu.Unlock()

	now := time.Now()
	for key, decision := range e.trustCache {
		if now.After(decision.ExpiresAt) {
			delete(e.trustCache, key)
		}
	}
}

// getApplicablePolicies returns policies applicable to the trust context
func (e *Engine) getApplicablePolicies(trustCtx *TrustContext) []*TrustPolicy {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var applicable []*TrustPolicy
	for _, policy := range e.policies {
		applicable = append(applicable, policy)
	}

	// Sort by priority (higher priority first)
	for i := 0; i < len(applicable)-1; i++ {
		for j := i + 1; j < len(applicable); j++ {
			if applicable[i].Priority < applicable[j].Priority {
				applicable[i], applicable[j] = applicable[j], applicable[i]
			}
		}
	}

	return applicable
}

// evaluateCondition evaluates a single policy condition
func (e *Engine) evaluateCondition(condition PolicyCondition, trustCtx *TrustContext) bool {
	var actualValue interface{}

	switch condition.Type {
	case "user":
		actualValue = trustCtx.UserID
	case "device":
		actualValue = trustCtx.DeviceID
	case "location":
		actualValue = trustCtx.Location
	case "resource":
		actualValue = trustCtx.ResourceID
	case "action":
		actualValue = trustCtx.Action
	default:
		return false
	}

	switch condition.Operator {
	case "equals":
		return actualValue == condition.Value
	case "contains":
		str, ok := actualValue.(string)
		if !ok {
			return false
		}
		substr, ok := condition.Value.(string)
		if !ok {
			return false
		}
		return contains(str, substr)
	case "in":
		list, ok := condition.Value.([]string)
		if !ok {
			return false
		}
		str, ok := actualValue.(string)
		if !ok {
			return false
		}
		return inList(str, list)
	case "not_in":
		list, ok := condition.Value.([]string)
		if !ok {
			return false
		}
		str, ok := actualValue.(string)
		if !ok {
			return false
		}
		return !inList(str, list)
	default:
		return false
	}
}

// applyAction applies a policy action to the decision
func (e *Engine) applyAction(action PolicyAction, decision *TrustDecision) {
	switch action.Type {
	case "allow":
		decision.Allowed = true
		decision.Reason = "Allowed by policy"
	case "deny":
		decision.Allowed = false
		decision.Reason = "Denied by policy"
	case "require_mfa":
		decision.RequiresMFA = true
		decision.Conditions = append(decision.Conditions, "MFA required")
	case "require_stepup":
		decision.RequiresStepUp = true
		decision.Conditions = append(decision.Conditions, "Step-up authentication required")
	case "alert":
		decision.Conditions = append(decision.Conditions, "Security alert generated")
	}
}

// getCacheKey generates a cache key from trust context
func (e *Engine) getCacheKey(trustCtx *TrustContext) string {
	h := sha256.New()
	h.Write([]byte(trustCtx.UserID))
	h.Write([]byte(trustCtx.DeviceID))
	h.Write([]byte(trustCtx.ResourceID))
	h.Write([]byte(trustCtx.Action))
	return hex.EncodeToString(h.Sum(nil))
}

// getCachedDecision retrieves a cached decision
func (e *Engine) getCachedDecision(key string) *TrustDecision {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.trustCache[key]
}

// cacheDecision caches a trust decision
func (e *Engine) cacheDecision(key string, decision *TrustDecision) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.trustCache[key] = decision
}

// Helper functions

func contains(s, substr string) bool {
	return len(s) >= len(substr) && s[:len(substr)] == substr
}

func inList(s string, list []string) bool {
	for _, item := range list {
		if s == item {
			return true
		}
	}
	return false
}

// GetMetrics returns zero-trust metrics
func (e *Engine) GetMetrics() map[string]interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return map[string]interface{}{
		"total_policies":     len(e.policies),
		"cached_decisions":   len(e.trustCache),
		"continuous_verify":  e.continuousVerify,
		"verify_interval_ms": e.verifyInterval.Milliseconds(),
		"micro_segmentation": e.microSegmentation,
		"least_privilege":    e.leastPrivilege,
	}
}
