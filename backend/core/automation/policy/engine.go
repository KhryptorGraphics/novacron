// Package policy provides advanced policy-as-code capabilities with OPA integration
package policy

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// PolicyEngine manages policies with GitOps integration
type PolicyEngine struct {
	policies       map[string]*Policy
	enforcer       *PolicyEnforcer
	simulator      *PolicySimulator
	analyzer       *ConflictAnalyzer
	versionControl *PolicyVersionControl
	gitOps         *GitOpsManager
	logger         *zap.Logger
	mu             sync.RWMutex
}

// Policy represents a policy definition
type Policy struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Version     string                 `json:"version"`
	Type        PolicyType             `json:"type"`
	Scope       PolicyScope            `json:"scope"`
	Rules       []PolicyRule           `json:"rules"`
	Enforcement EnforcementMode        `json:"enforcement"`
	Parameters  map[string]interface{} `json:"parameters"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Author      string                 `json:"author"`
	Enabled     bool                   `json:"enabled"`
}

// PolicyType defines the type of policy
type PolicyType string

const (
	PolicyTypeAccess     PolicyType = "access_control"
	PolicyTypeResource   PolicyType = "resource_quota"
	PolicyTypeSecurity   PolicyType = "security"
	PolicyTypeCompliance PolicyType = "compliance"
	PolicyTypeCost       PolicyType = "cost_control"
	PolicyTypeQuality    PolicyType = "quality"
	PolicyTypeOperation  PolicyType = "operational"
)

// PolicyScope defines where the policy applies
type PolicyScope struct {
	Level      string   `json:"level"`       // global, namespace, resource
	Targets    []string `json:"targets"`     // Specific targets
	Exclusions []string `json:"exclusions"`  // Exclusions
}

// PolicyRule represents a single policy rule
type PolicyRule struct {
	ID          string                 `json:"id"`
	Description string                 `json:"description"`
	Condition   string                 `json:"condition"`   // OPA Rego expression
	Action      PolicyAction           `json:"action"`
	Severity    string                 `json:"severity"`    // critical, high, medium, low
	Parameters  map[string]interface{} `json:"parameters"`
}

// PolicyAction defines what happens when a rule is triggered
type PolicyAction struct {
	Type       string                 `json:"type"`       // deny, allow, warn, remediate
	Message    string                 `json:"message"`
	Remediation *RemediationAction    `json:"remediation,omitempty"`
	Metadata   map[string]interface{} `json:"metadata"`
}

// RemediationAction defines automatic remediation
type RemediationAction struct {
	Type       string                 `json:"type"`
	Steps      []RemediationStep      `json:"steps"`
	Automatic  bool                   `json:"automatic"`
	Timeout    time.Duration          `json:"timeout"`
}

// RemediationStep represents a remediation step
type RemediationStep struct {
	Action     string                 `json:"action"`
	Parameters map[string]interface{} `json:"parameters"`
	Order      int                    `json:"order"`
}

// EnforcementMode defines how strictly policies are enforced
type EnforcementMode string

const (
	EnforcementStrict    EnforcementMode = "strict"     // Block violations
	EnforcementAdvisory  EnforcementMode = "advisory"   // Warn only
	EnforcementGradual   EnforcementMode = "gradual"    // Gradually enforce
	EnforcementMonitor   EnforcementMode = "monitor"    // Monitor only
)

// PolicyEvaluation represents a policy evaluation result
type PolicyEvaluation struct {
	PolicyID    string                 `json:"policy_id"`
	RuleID      string                 `json:"rule_id"`
	Decision    PolicyDecision         `json:"decision"`
	Reason      string                 `json:"reason"`
	Severity    string                 `json:"severity"`
	Remediation *RemediationAction     `json:"remediation,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
	Timestamp   time.Time              `json:"timestamp"`
}

// PolicyDecision represents the outcome of a policy evaluation
type PolicyDecision string

const (
	DecisionAllow  PolicyDecision = "allow"
	DecisionDeny   PolicyDecision = "deny"
	DecisionWarn   PolicyDecision = "warn"
)

// NewPolicyEngine creates a new policy engine
func NewPolicyEngine(logger *zap.Logger) *PolicyEngine {
	engine := &PolicyEngine{
		policies: make(map[string]*Policy),
		logger:   logger,
	}

	engine.enforcer = NewPolicyEnforcer(logger)
	engine.simulator = NewPolicySimulator(logger)
	engine.analyzer = NewConflictAnalyzer(logger)
	engine.versionControl = NewPolicyVersionControl(logger)
	engine.gitOps = NewGitOpsManager(logger)

	return engine
}

// RegisterPolicy registers a new policy
func (e *PolicyEngine) RegisterPolicy(ctx context.Context, policy *Policy) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Validate policy
	if err := e.validatePolicy(policy); err != nil {
		return fmt.Errorf("invalid policy: %w", err)
	}

	// Check for conflicts
	conflicts := e.analyzer.DetectConflicts(ctx, policy, e.getAllPolicies())
	if len(conflicts) > 0 {
		return fmt.Errorf("policy conflicts detected: %v", conflicts)
	}

	// Version the policy
	if err := e.versionControl.VersionPolicy(ctx, policy); err != nil {
		return fmt.Errorf("failed to version policy: %w", err)
	}

	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()

	e.policies[policy.ID] = policy

	e.logger.Info("Policy registered",
		zap.String("id", policy.ID),
		zap.String("name", policy.Name),
		zap.String("version", policy.Version))

	return nil
}

// EvaluatePolicy evaluates a policy against input data
func (e *PolicyEngine) EvaluatePolicy(ctx context.Context, policyID string, input map[string]interface{}) ([]*PolicyEvaluation, error) {
	e.mu.RLock()
	policy, exists := e.policies[policyID]
	e.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("policy not found: %s", policyID)
	}

	if !policy.Enabled {
		return nil, fmt.Errorf("policy is disabled: %s", policyID)
	}

	evaluations := make([]*PolicyEvaluation, 0)

	for _, rule := range policy.Rules {
		evaluation := e.evaluateRule(ctx, policy, rule, input)
		if evaluation != nil {
			evaluations = append(evaluations, evaluation)
		}
	}

	e.logger.Info("Policy evaluated",
		zap.String("policy_id", policyID),
		zap.Int("evaluations", len(evaluations)))

	return evaluations, nil
}

// evaluateRule evaluates a single policy rule
func (e *PolicyEngine) evaluateRule(ctx context.Context, policy *Policy, rule PolicyRule, input map[string]interface{}) *PolicyEvaluation {
	// Evaluate condition (simplified - would use OPA in production)
	matches := e.evaluateCondition(ctx, rule.Condition, input)

	if !matches {
		return nil
	}

	decision := e.determineDecision(rule.Action.Type, policy.Enforcement)

	evaluation := &PolicyEvaluation{
		PolicyID:  policy.ID,
		RuleID:    rule.ID,
		Decision:  decision,
		Reason:    rule.Action.Message,
		Severity:  rule.Severity,
		Metadata:  rule.Action.Metadata,
		Timestamp: time.Now(),
	}

	// Add remediation if applicable
	if rule.Action.Remediation != nil && rule.Action.Remediation.Automatic {
		evaluation.Remediation = rule.Action.Remediation
	}

	return evaluation
}

// evaluateCondition evaluates a policy condition
func (e *PolicyEngine) evaluateCondition(ctx context.Context, condition string, input map[string]interface{}) bool {
	// Simplified evaluation - would use OPA Rego in production
	// This is a placeholder for actual OPA integration

	// For demonstration, check for common patterns
	if condition == "resource.cpu > 8" {
		if cpu, ok := input["cpu"].(float64); ok {
			return cpu > 8
		}
	}

	if condition == "cost.monthly > 1000" {
		if cost, ok := input["monthly_cost"].(float64); ok {
			return cost > 1000
		}
	}

	// Default to false for unknown conditions
	return false
}

// determineDecision determines the final decision based on action and enforcement mode
func (e *PolicyEngine) determineDecision(actionType string, enforcement EnforcementMode) PolicyDecision {
	switch enforcement {
	case EnforcementStrict:
		if actionType == "deny" {
			return DecisionDeny
		}
		return DecisionAllow

	case EnforcementAdvisory:
		if actionType == "deny" {
			return DecisionWarn
		}
		return DecisionAllow

	case EnforcementGradual:
		// Gradual enforcement starts as warning and becomes strict over time
		// This would check policy age and transition
		return DecisionWarn

	case EnforcementMonitor:
		return DecisionAllow

	default:
		return DecisionAllow
	}
}

// EnforcePolicy enforces a policy decision
func (e *PolicyEngine) EnforcePolicy(ctx context.Context, evaluations []*PolicyEvaluation) error {
	for _, eval := range evaluations {
		if err := e.enforcer.Enforce(ctx, eval); err != nil {
			return fmt.Errorf("enforcement failed for policy %s: %w", eval.PolicyID, err)
		}

		// Auto-remediate if configured
		if eval.Remediation != nil && eval.Remediation.Automatic {
			if err := e.enforcer.Remediate(ctx, eval); err != nil {
				e.logger.Error("Automatic remediation failed",
					zap.String("policy_id", eval.PolicyID),
					zap.Error(err))
			}
		}
	}

	return nil
}

// SimulatePolicy simulates policy effects without enforcement
func (e *PolicyEngine) SimulatePolicy(ctx context.Context, policy *Policy, inputs []map[string]interface{}) (*SimulationResult, error) {
	return e.simulator.Simulate(ctx, policy, inputs)
}

// AnalyzeConflicts analyzes potential policy conflicts
func (e *PolicyEngine) AnalyzeConflicts(ctx context.Context) ([]*PolicyConflict, error) {
	e.mu.RLock()
	policies := e.getAllPolicies()
	e.mu.RUnlock()

	return e.analyzer.AnalyzeAll(ctx, policies)
}

// SyncToGit syncs policies to Git repository
func (e *PolicyEngine) SyncToGit(ctx context.Context) error {
	e.mu.RLock()
	policies := e.getAllPolicies()
	e.mu.RUnlock()

	return e.gitOps.Sync(ctx, policies)
}

// LoadFromGit loads policies from Git repository
func (e *PolicyEngine) LoadFromGit(ctx context.Context) error {
	policies, err := e.gitOps.Load(ctx)
	if err != nil {
		return err
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	for _, policy := range policies {
		e.policies[policy.ID] = policy
	}

	e.logger.Info("Policies loaded from Git",
		zap.Int("count", len(policies)))

	return nil
}

// Helper functions

func (e *PolicyEngine) validatePolicy(policy *Policy) error {
	if policy.ID == "" {
		return fmt.Errorf("policy ID is required")
	}

	if policy.Name == "" {
		return fmt.Errorf("policy name is required")
	}

	if len(policy.Rules) == 0 {
		return fmt.Errorf("policy must have at least one rule")
	}

	return nil
}

func (e *PolicyEngine) getAllPolicies() []*Policy {
	policies := make([]*Policy, 0, len(e.policies))
	for _, policy := range e.policies {
		policies = append(policies, policy)
	}
	return policies
}

// GetPolicy retrieves a policy by ID
func (e *PolicyEngine) GetPolicy(ctx context.Context, policyID string) (*Policy, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	policy, exists := e.policies[policyID]
	if !exists {
		return nil, fmt.Errorf("policy not found: %s", policyID)
	}

	return policy, nil
}

// ListPolicies lists all policies
func (e *PolicyEngine) ListPolicies(ctx context.Context, filters map[string]interface{}) ([]*Policy, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	policies := make([]*Policy, 0)

	for _, policy := range e.policies {
		if e.matchesFilters(policy, filters) {
			policies = append(policies, policy)
		}
	}

	return policies, nil
}

func (e *PolicyEngine) matchesFilters(policy *Policy, filters map[string]interface{}) bool {
	if policyType, ok := filters["type"].(string); ok {
		if string(policy.Type) != policyType {
			return false
		}
	}

	if enabled, ok := filters["enabled"].(bool); ok {
		if policy.Enabled != enabled {
			return false
		}
	}

	return true
}

// UpdatePolicy updates an existing policy
func (e *PolicyEngine) UpdatePolicy(ctx context.Context, policy *Policy) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.policies[policy.ID]; !exists {
		return fmt.Errorf("policy not found: %s", policy.ID)
	}

	// Create new version
	if err := e.versionControl.VersionPolicy(ctx, policy); err != nil {
		return fmt.Errorf("failed to version policy: %w", err)
	}

	policy.UpdatedAt = time.Now()
	e.policies[policy.ID] = policy

	e.logger.Info("Policy updated",
		zap.String("id", policy.ID),
		zap.String("version", policy.Version))

	return nil
}

// DeletePolicy deletes a policy
func (e *PolicyEngine) DeletePolicy(ctx context.Context, policyID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.policies[policyID]; !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	delete(e.policies, policyID)

	e.logger.Info("Policy deleted",
		zap.String("id", policyID))

	return nil
}

// RollbackPolicy rolls back a policy to a previous version
func (e *PolicyEngine) RollbackPolicy(ctx context.Context, policyID string, version string) error {
	policy, err := e.versionControl.GetVersion(ctx, policyID, version)
	if err != nil {
		return fmt.Errorf("failed to get policy version: %w", err)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.policies[policyID] = policy

	e.logger.Info("Policy rolled back",
		zap.String("id", policyID),
		zap.String("version", version))

	return nil
}
