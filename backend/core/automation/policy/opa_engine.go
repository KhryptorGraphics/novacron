// Package policy provides advanced policy engine with OPA integration
package policy

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/open-policy-agent/opa/rego"
	"github.com/sirupsen/logrus"
)

// PolicyEngine manages policy evaluation and enforcement
type PolicyEngine struct {
	policies   map[string]*Policy
	modules    map[string]*rego.PreparedEvalQuery
	enforcer   *PolicyEnforcer
	validator  *PolicyValidator
	logger     *logrus.Logger
	mu         sync.RWMutex
	ctx        context.Context
	cancel     context.CancelFunc
}

// Policy represents a policy definition
type Policy struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Version     string                 `json:"version"`
	Type        PolicyType             `json:"type"`
	Scope       PolicyScope            `json:"scope"`
	Rule        string                 `json:"rule"`          // Rego policy
	Parameters  map[string]interface{} `json:"parameters"`
	Priority    int                    `json:"priority"`
	Enabled     bool                   `json:"enabled"`
	EnforcementMode EnforcementMode   `json:"enforcement_mode"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Tags        []string               `json:"tags"`
}

// PolicyType defines policy types
type PolicyType string

const (
	PolicyTypeAccess      PolicyType = "access"
	PolicyTypeResource    PolicyType = "resource"
	PolicyTypeCompliance  PolicyType = "compliance"
	PolicyTypeSecurity    PolicyType = "security"`
	PolicyTypeQuota       PolicyType = "quota"
	PolicyTypeNetwork     PolicyType = "network"
	PolicyTypeData        PolicyType = "data"
)

// PolicyScope defines policy scope
type PolicyScope string

const (
	PolicyScopeGlobal     PolicyScope = "global"
	PolicyScopeOrganization PolicyScope = "organization"
	PolicyScopeProject    PolicyScope = "project"
	PolicyScopeResource   PolicyScope = "resource"
)

// EnforcementMode defines how policy is enforced
type EnforcementMode string

const (
	EnforcementModeBlock  EnforcementMode = "block"
	EnforcementModeWarn   EnforcementMode = "warn"
	EnforcementModeAudit  EnforcementMode = "audit"
)

// EvaluationRequest represents a policy evaluation request
type EvaluationRequest struct {
	PolicyID    string                 `json:"policy_id"`
	Action      string                 `json:"action"`
	Resource    map[string]interface{} `json:"resource"`
	Subject     map[string]interface{} `json:"subject"`
	Environment map[string]interface{} `json:"environment"`
	Context     map[string]interface{} `json:"context"`
}

// EvaluationResult represents policy evaluation result
type EvaluationResult struct {
	PolicyID    string                 `json:"policy_id"`
	Allowed     bool                   `json:"allowed"`
	Decision    string                 `json:"decision"`
	Violations  []Violation            `json:"violations,omitempty"`
	Reasons     []string               `json:"reasons"`
	Timestamp   time.Time              `json:"timestamp"`
	EvalTime    time.Duration          `json:"eval_time"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// Violation represents a policy violation
type Violation struct {
	Code        string    `json:"code"`
	Message     string    `json:"message"`
	Severity    string    `json:"severity"`
	Field       string    `json:"field,omitempty"`
	Remediation string    `json:"remediation,omitempty"`
}

// PolicyEnforcer enforces policy decisions
type PolicyEnforcer struct {
	engine *PolicyEngine
	logger *logrus.Logger
	stats  *EnforcementStats
	mu     sync.RWMutex
}

// EnforcementStats tracks enforcement statistics
type EnforcementStats struct {
	TotalEvaluations int64
	Allowed          int64
	Denied           int64
	Warned           int64
	AverageEvalTime  time.Duration
}

// PolicyValidator validates policies
type PolicyValidator struct {
	logger *logrus.Logger
}

// NewPolicyEngine creates a new policy engine
func NewPolicyEngine(logger *logrus.Logger) *PolicyEngine {
	ctx, cancel := context.WithCancel(context.Background())

	pe := &PolicyEngine{
		policies: make(map[string]*Policy),
		modules:  make(map[string]*rego.PreparedEvalQuery),
		logger:   logger,
		ctx:      ctx,
		cancel:   cancel,
	}

	pe.enforcer = NewPolicyEnforcer(pe, logger)
	pe.validator = NewPolicyValidator(logger)

	return pe
}

// RegisterPolicy registers a new policy
func (pe *PolicyEngine) RegisterPolicy(policy *Policy) error {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	// Validate policy
	if err := pe.validator.Validate(policy); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}

	// Compile Rego policy
	query, err := pe.compilePolicy(policy)
	if err != nil {
		return fmt.Errorf("policy compilation failed: %w", err)
	}

	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()

	pe.policies[policy.ID] = policy
	pe.modules[policy.ID] = query

	pe.logger.WithFields(logrus.Fields{
		"policy_id": policy.ID,
		"type":      policy.Type,
		"scope":     policy.Scope,
	}).Info("Policy registered")

	return nil
}

// EvaluatePolicy evaluates a policy against input
func (pe *PolicyEngine) EvaluatePolicy(ctx context.Context, req *EvaluationRequest) (*EvaluationResult, error) {
	start := time.Now()

	pe.mu.RLock()
	policy, exists := pe.policies[req.PolicyID]
	query, hasQuery := pe.modules[req.PolicyID]
	pe.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("policy not found: %s", req.PolicyID)
	}

	if !policy.Enabled {
		return &EvaluationResult{
			PolicyID:  req.PolicyID,
			Allowed:   true,
			Decision:  "policy_disabled",
			Timestamp: time.Now(),
			EvalTime:  time.Since(start),
		}, nil
	}

	if !hasQuery {
		return nil, fmt.Errorf("policy not compiled: %s", req.PolicyID)
	}

	// Prepare input for Rego evaluation
	input := map[string]interface{}{
		"action":      req.Action,
		"resource":    req.Resource,
		"subject":     req.Subject,
		"environment": req.Environment,
		"context":     req.Context,
		"parameters":  policy.Parameters,
	}

	// Evaluate policy
	results, err := query.Eval(ctx, rego.EvalInput(input))
	if err != nil {
		return nil, fmt.Errorf("policy evaluation error: %w", err)
	}

	// Parse results
	result := pe.parseEvaluationResults(policy, results)
	result.PolicyID = req.PolicyID
	result.Timestamp = time.Now()
	result.EvalTime = time.Since(start)

	// Update enforcement stats
	pe.enforcer.RecordEvaluation(result)

	pe.logger.WithFields(logrus.Fields{
		"policy_id": req.PolicyID,
		"allowed":   result.Allowed,
		"eval_time": result.EvalTime,
	}).Debug("Policy evaluated")

	return result, nil
}

// EvaluateAll evaluates all relevant policies
func (pe *PolicyEngine) EvaluateAll(ctx context.Context, req *EvaluationRequest) ([]*EvaluationResult, error) {
	pe.mu.RLock()
	relevantPolicies := pe.getRelevantPolicies(req)
	pe.mu.RUnlock()

	results := make([]*EvaluationResult, 0, len(relevantPolicies))

	for _, policy := range relevantPolicies {
		evalReq := &EvaluationRequest{
			PolicyID:    policy.ID,
			Action:      req.Action,
			Resource:    req.Resource,
			Subject:     req.Subject,
			Environment: req.Environment,
			Context:     req.Context,
		}

		result, err := pe.EvaluatePolicy(ctx, evalReq)
		if err != nil {
			pe.logger.WithError(err).Warn("Policy evaluation failed")
			continue
		}

		results = append(results, result)
	}

	return results, nil
}

// EnforcePolicy enforces policy decisions
func (pe *PolicyEngine) EnforcePolicy(ctx context.Context, req *EvaluationRequest) error {
	result, err := pe.EvaluatePolicy(ctx, req)
	if err != nil {
		return err
	}

	return pe.enforcer.Enforce(result)
}

// compilePolicy compiles Rego policy
func (pe *PolicyEngine) compilePolicy(policy *Policy) (*rego.PreparedEvalQuery, error) {
	// Create Rego query
	r := rego.New(
		rego.Query("data.novacron.allow"),
		rego.Module(policy.ID, policy.Rule),
	)

	// Prepare query for better performance
	query, err := r.PrepareForEval(pe.ctx)
	if err != nil {
		return nil, err
	}

	return &query, nil
}

// parseEvaluationResults parses OPA evaluation results
func (pe *PolicyEngine) parseEvaluationResults(policy *Policy, results rego.ResultSet) *EvaluationResult {
	result := &EvaluationResult{
		PolicyID:   policy.ID,
		Allowed:    false,
		Violations: make([]Violation, 0),
		Reasons:    make([]string, 0),
		Metadata:   make(map[string]interface{}),
	}

	if len(results) == 0 {
		result.Decision = "deny"
		result.Reasons = append(result.Reasons, "No matching policy rules")
		return result
	}

	// Check if allowed
	if len(results) > 0 && len(results[0].Expressions) > 0 {
		if allowed, ok := results[0].Expressions[0].Value.(bool); ok && allowed {
			result.Allowed = true
			result.Decision = "allow"
			result.Reasons = append(result.Reasons, "Policy requirements satisfied")
		} else {
			result.Decision = "deny"

			// Extract violations if present
			if violations, ok := results[0].Bindings["violations"]; ok {
				if violationList, ok := violations.([]interface{}); ok {
					for _, v := range violationList {
						if vMap, ok := v.(map[string]interface{}); ok {
							violation := Violation{
								Code:     getString(vMap, "code"),
								Message:  getString(vMap, "message"),
								Severity: getString(vMap, "severity"),
								Field:    getString(vMap, "field"),
								Remediation: getString(vMap, "remediation"),
							}
							result.Violations = append(result.Violations, violation)
							result.Reasons = append(result.Reasons, violation.Message)
						}
					}
				}
			}

			if len(result.Reasons) == 0 {
				result.Reasons = append(result.Reasons, "Policy requirements not met")
			}
		}
	}

	return result
}

func (pe *PolicyEngine) getRelevantPolicies(req *EvaluationRequest) []*Policy {
	policies := make([]*Policy, 0)

	for _, policy := range pe.policies {
		if policy.Enabled {
			policies = append(policies, policy)
		}
	}

	// Sort by priority (higher first)
	// Sort implementation would go here

	return policies
}

func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

// NewPolicyEnforcer creates a new policy enforcer
func NewPolicyEnforcer(engine *PolicyEngine, logger *logrus.Logger) *PolicyEnforcer {
	return &PolicyEnforcer{
		engine: engine,
		logger: logger,
		stats:  &EnforcementStats{},
	}
}

// Enforce enforces a policy decision
func (pe *PolicyEnforcer) Enforce(result *EvaluationResult) error {
	pe.mu.RLock()
	policy := pe.engine.policies[result.PolicyID]
	pe.mu.RUnlock()

	if policy == nil {
		return fmt.Errorf("policy not found: %s", result.PolicyID)
	}

	switch policy.EnforcementMode {
	case EnforcementModeBlock:
		if !result.Allowed {
			return fmt.Errorf("policy violation: %s - %s",
				policy.Name, result.Reasons[0])
		}

	case EnforcementModeWarn:
		if !result.Allowed {
			pe.logger.WithFields(logrus.Fields{
				"policy_id": result.PolicyID,
				"reasons":   result.Reasons,
			}).Warn("Policy violation (warning mode)")
		}

	case EnforcementModeAudit:
		// Just log for audit purposes
		pe.logger.WithFields(logrus.Fields{
			"policy_id": result.PolicyID,
			"allowed":   result.Allowed,
			"reasons":   result.Reasons,
		}).Info("Policy evaluation result (audit mode)")
	}

	return nil
}

// RecordEvaluation records evaluation statistics
func (pe *PolicyEnforcer) RecordEvaluation(result *EvaluationResult) {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	pe.stats.TotalEvaluations++
	if result.Allowed {
		pe.stats.Allowed++
	} else {
		pe.stats.Denied++
	}

	// Update average eval time
	pe.stats.AverageEvalTime = (pe.stats.AverageEvalTime +
		result.EvalTime) / 2
}

// GetStats returns enforcement statistics
func (pe *PolicyEnforcer) GetStats() *EnforcementStats {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	stats := *pe.stats
	return &stats
}

// NewPolicyValidator creates a new policy validator
func NewPolicyValidator(logger *logrus.Logger) *PolicyValidator {
	return &PolicyValidator{
		logger: logger,
	}
}

// Validate validates a policy
func (pv *PolicyValidator) Validate(policy *Policy) error {
	if policy.ID == "" {
		return fmt.Errorf("policy ID is required")
	}

	if policy.Name == "" {
		return fmt.Errorf("policy name is required")
	}

	if policy.Rule == "" {
		return fmt.Errorf("policy rule is required")
	}

	// Validate Rego syntax
	if err := pv.validateRego(policy.Rule); err != nil {
		return fmt.Errorf("invalid Rego policy: %w", err)
	}

	return nil
}

// validateRego validates Rego syntax
func (pv *PolicyValidator) validateRego(rule string) error {
	// Try to parse and compile the Rego policy
	r := rego.New(
		rego.Query("data.novacron.allow"),
		rego.Module("test", rule),
	)

	_, err := r.PrepareForEval(context.Background())
	return err
}

// ExportMetrics exports policy engine metrics
func (pe *PolicyEngine) ExportMetrics() map[string]interface{} {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	stats := pe.enforcer.GetStats()

	return map[string]interface{}{
		"total_policies":      len(pe.policies),
		"enabled_policies":    pe.countEnabledPolicies(),
		"total_evaluations":   stats.TotalEvaluations,
		"allowed":             stats.Allowed,
		"denied":              stats.Denied,
		"average_eval_time_ms": stats.AverageEvalTime.Milliseconds(),
	}
}

func (pe *PolicyEngine) countEnabledPolicies() int {
	count := 0
	for _, policy := range pe.policies {
		if policy.Enabled {
			count++
		}
	}
	return count
}

// Stop gracefully stops the policy engine
func (pe *PolicyEngine) Stop() {
	pe.logger.Info("Stopping policy engine")
	pe.cancel()
}
