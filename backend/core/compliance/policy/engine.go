// Package policy implements policy-as-code enforcement engine
package policy

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"novacron/backend/core/compliance"
)

// Engine is the policy enforcement engine
type Engine struct {
	policies       map[string]*compliance.Policy
	violations     map[string][]*compliance.PolicyViolation
	evaluationCache map[string]*CachedDecision
	mu             sync.RWMutex

	// OPA integration (would use actual OPA client in production)
	opaEnabled bool
}

// CachedDecision represents a cached policy decision
type CachedDecision struct {
	Decision  *compliance.PolicyDecision
	CachedAt  time.Time
	ExpiresAt time.Time
}

// NewEngine creates a new policy engine
func NewEngine() *Engine {
	engine := &Engine{
		policies:        make(map[string]*compliance.Policy),
		violations:      make(map[string][]*compliance.PolicyViolation),
		evaluationCache: make(map[string]*CachedDecision),
		opaEnabled:      false, // Enable when OPA is configured
	}

	// Register default policies
	engine.registerDefaultPolicies()

	return engine
}

// registerDefaultPolicies registers common security and compliance policies
func (e *Engine) registerDefaultPolicies() {
	policies := []*compliance.Policy{
		// Access Control Policies
		{
			ID:          "pol-access-mfa-required",
			Name:        "MFA Required for Production Access",
			Description: "All production system access requires multi-factor authentication",
			Type:        compliance.PolicyTypeAccess,
			Enabled:     true,
			Priority:    1,
			Scope: compliance.PolicyScope{
				Resources:  []string{"vm:production/*", "api:production/*"},
				Actions:    []string{"access", "login", "connect"},
				Principals: []string{"*"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"mfa_verified": false,
						"environment":  "production",
					},
					Description: "Deny access if MFA not verified",
				},
			},
		},
		{
			ID:          "pol-access-least-privilege",
			Name:        "Principle of Least Privilege",
			Description: "Users can only access resources explicitly granted to their role",
			Type:        compliance.PolicyTypeAccess,
			Enabled:     true,
			Priority:    2,
			Scope: compliance.PolicyScope{
				Resources:  []string{"*"},
				Actions:    []string{"*"},
				Principals: []string{"*"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"explicit_grant": false,
					},
					Description: "Deny by default unless explicitly granted",
				},
			},
		},
		{
			ID:          "pol-access-business-hours",
			Name:        "Non-Business Hours Access Restriction",
			Description: "Restrict sensitive operations to business hours unless approved",
			Type:        compliance.PolicyTypeAccess,
			Enabled:     true,
			Priority:    3,
			Scope: compliance.PolicyScope{
				Resources:  []string{"database:production/*", "vm:production/*"},
				Actions:    []string{"delete", "modify", "drop"},
				Principals: []string{"*"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"time_check":       "outside_business_hours",
						"emergency_access": false,
					},
					Description: "Deny destructive operations outside business hours without approval",
				},
			},
		},

		// Data Protection Policies
		{
			ID:          "pol-data-encryption-required",
			Name:        "Encryption Required for Sensitive Data",
			Description: "All sensitive data must be encrypted at rest and in transit",
			Type:        compliance.PolicyTypeData,
			Enabled:     true,
			Priority:    1,
			Scope: compliance.PolicyScope{
				Resources: []string{"*:sensitive/*", "*:confidential/*"},
				Actions:   []string{"create", "store", "transmit"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"encryption_enabled": false,
						"data_classification": []string{"sensitive", "confidential", "phi", "pii"},
					},
					Description: "Deny storage/transmission without encryption",
				},
			},
		},
		{
			ID:          "pol-data-retention",
			Name:        "Data Retention Policy",
			Description: "Enforce data retention and deletion schedules",
			Type:        compliance.PolicyTypeData,
			Enabled:     true,
			Priority:    2,
			Scope: compliance.PolicyScope{
				Resources: []string{"*"},
				Actions:   []string{"create", "store"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"retention_policy_defined": false,
					},
					Description: "Require retention policy for all data storage",
				},
			},
		},
		{
			ID:          "pol-data-cross-border",
			Name:        "Cross-Border Data Transfer Restrictions",
			Description: "Restrict data transfers to approved countries (GDPR compliance)",
			Type:        compliance.PolicyTypeData,
			Enabled:     true,
			Priority:    1,
			Scope: compliance.PolicyScope{
				Resources: []string{"*:pii/*", "*:phi/*"},
				Actions:   []string{"transfer", "replicate"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"destination_country": []string{"not_in_approved_list"},
						"adequacy_decision":   false,
						"sccs_in_place":       false,
					},
					Description: "Deny international transfers without proper safeguards",
				},
			},
		},

		// Network Security Policies
		{
			ID:          "pol-network-zero-trust",
			Name:        "Zero Trust Network Access",
			Description: "All network connections must be authenticated and authorized",
			Type:        compliance.PolicyTypeNetwork,
			Enabled:     true,
			Priority:    1,
			Scope: compliance.PolicyScope{
				Resources: []string{"network:*"},
				Actions:   []string{"connect", "access"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"authenticated": false,
					},
					Description: "Deny unauthenticated network access",
				},
				{
					ID:     "rule-2",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"tls_version": []string{"1.0", "1.1"},
					},
					Description: "Deny outdated TLS versions",
				},
			},
		},
		{
			ID:          "pol-network-segmentation",
			Name:        "Network Segmentation Enforcement",
			Description: "Enforce network segmentation between environments",
			Type:        compliance.PolicyTypeNetwork,
			Enabled:     true,
			Priority:    2,
			Scope: compliance.PolicyScope{
				Resources: []string{"network:production/*"},
				Actions:   []string{"connect"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"source_network": []string{"development", "test"},
						"approved_path":  false,
					},
					Description: "Deny direct connections from dev/test to production",
				},
			},
		},

		// Compliance Policies
		{
			ID:          "pol-compliance-soc2-audit-log",
			Name:        "SOC2 Audit Log Retention",
			Description: "Retain audit logs for minimum 1 year (SOC2 requirement)",
			Type:        compliance.PolicyTypeCompliance,
			Enabled:     true,
			Priority:    1,
			Scope: compliance.PolicyScope{
				Resources: []string{"logs:audit/*"},
				Actions:   []string{"delete", "purge"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"log_age_days": []string{"<365"},
					},
					Description: "Deny deletion of audit logs less than 1 year old",
				},
			},
		},
		{
			ID:          "pol-compliance-hipaa-phi-access",
			Name:        "HIPAA PHI Access Logging",
			Description: "All PHI access must be logged with purpose",
			Type:        compliance.PolicyTypeCompliance,
			Enabled:     true,
			Priority:    1,
			Scope: compliance.PolicyScope{
				Resources: []string{"*:phi/*"},
				Actions:   []string{"read", "access", "view"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"access_purpose": "",
						"logging_enabled": false,
					},
					Description: "Deny PHI access without purpose and logging",
				},
			},
		},

		// Governance Policies
		{
			ID:          "pol-gov-resource-tagging",
			Name:        "Mandatory Resource Tagging",
			Description: "All resources must have required tags",
			Type:        compliance.PolicyTypeGovernance,
			Enabled:     true,
			Priority:    2,
			Scope: compliance.PolicyScope{
				Resources: []string{"vm:*", "volume:*", "network:*"},
				Actions:   []string{"create"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"missing_tags": []string{"owner", "project", "environment", "cost_center"},
					},
					Description: "Deny resource creation without required tags",
				},
			},
		},
		{
			ID:          "pol-gov-cost-control",
			Name:        "Cost Control and Budget Enforcement",
			Description: "Enforce budget limits and cost controls",
			Type:        compliance.PolicyTypeGovernance,
			Enabled:     true,
			Priority:    3,
			Scope: compliance.PolicyScope{
				Resources: []string{"*"},
				Actions:   []string{"create", "scale"},
			},
			Rules: []compliance.PolicyRule{
				{
					ID:     "rule-1",
					Effect: compliance.PolicyEffectDeny,
					Conditions: map[string]interface{}{
						"budget_exceeded": true,
						"approval":        false,
					},
					Description: "Deny resource creation when budget exceeded without approval",
				},
			},
		},
	}

	for _, policy := range policies {
		policy.CreatedAt = time.Now()
		policy.UpdatedAt = time.Now()
		policy.Version = 1
		policy.Metadata = make(map[string]string)
		e.policies[policy.ID] = policy
	}
}

// CreatePolicy creates a new policy
func (e *Engine) CreatePolicy(policy *compliance.Policy) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Validate policy
	if err := e.validatePolicy(policy); err != nil {
		return fmt.Errorf("invalid policy: %w", err)
	}

	policy.ID = generateID("pol")
	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()
	policy.Version = 1

	e.policies[policy.ID] = policy

	// Clear cache when policies change
	e.evaluationCache = make(map[string]*CachedDecision)

	return nil
}

// UpdatePolicy updates an existing policy
func (e *Engine) UpdatePolicy(id string, policy *compliance.Policy) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	existing, ok := e.policies[id]
	if !ok {
		return fmt.Errorf("policy not found: %s", id)
	}

	if err := e.validatePolicy(policy); err != nil {
		return fmt.Errorf("invalid policy: %w", err)
	}

	policy.ID = id
	policy.CreatedAt = existing.CreatedAt
	policy.UpdatedAt = time.Now()
	policy.Version = existing.Version + 1

	e.policies[id] = policy

	// Clear cache
	e.evaluationCache = make(map[string]*CachedDecision)

	return nil
}

// DeletePolicy deletes a policy
func (e *Engine) DeletePolicy(id string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, ok := e.policies[id]; !ok {
		return fmt.Errorf("policy not found: %s", id)
	}

	delete(e.policies, id)

	// Clear cache
	e.evaluationCache = make(map[string]*CachedDecision)

	return nil
}

// GetPolicy retrieves a policy
func (e *Engine) GetPolicy(id string) (*compliance.Policy, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	policy, ok := e.policies[id]
	if !ok {
		return nil, fmt.Errorf("policy not found: %s", id)
	}

	return policy, nil
}

// ListPolicies returns all policies
func (e *Engine) ListPolicies() []*compliance.Policy {
	e.mu.RLock()
	defer e.mu.RUnlock()

	policies := make([]*compliance.Policy, 0, len(e.policies))
	for _, policy := range e.policies {
		policies = append(policies, policy)
	}

	return policies
}

// Evaluate evaluates a policy request
func (e *Engine) Evaluate(ctx context.Context, request *compliance.PolicyRequest) (*compliance.PolicyDecision, error) {
	// Check cache first
	cacheKey := e.generateCacheKey(request)
	if cached, ok := e.getCachedDecision(cacheKey); ok {
		return cached, nil
	}

	e.mu.RLock()
	defer e.mu.RUnlock()

	decision := &compliance.PolicyDecision{
		Allowed:         true,
		Decision:        compliance.PolicyEffectAllow,
		Reasons:         []string{},
		MatchedPolicies: []string{},
		EvaluatedAt:     time.Now(),
		Metadata:        make(map[string]string),
	}

	// Evaluate all applicable policies
	for _, policy := range e.policies {
		if !policy.Enabled {
			continue
		}

		if e.policyApplies(policy, request) {
			decision.MatchedPolicies = append(decision.MatchedPolicies, policy.ID)

			for _, rule := range policy.Rules {
				if e.ruleMatches(rule, request) {
					if rule.Effect == compliance.PolicyEffectDeny {
						decision.Allowed = false
						decision.Decision = compliance.PolicyEffectDeny
						decision.Reasons = append(decision.Reasons, fmt.Sprintf("Policy %s denied: %s", policy.Name, rule.Description))

						// Log violation
						e.logViolation(policy, request)

						// Deny takes precedence - return immediately
						return decision, nil
					}
				}
			}
		}
	}

	// If no explicit deny, check for explicit allow (depending on policy type)
	if len(decision.MatchedPolicies) == 0 {
		// No policies matched - default deny for security
		decision.Allowed = false
		decision.Decision = compliance.PolicyEffectDeny
		decision.Reasons = append(decision.Reasons, "No explicit policy grants access (default deny)")
	}

	// Cache decision
	e.cacheDecision(cacheKey, decision, 5*time.Minute)

	return decision, nil
}

// EvaluateBatch evaluates multiple requests
func (e *Engine) EvaluateBatch(ctx context.Context, requests []*compliance.PolicyRequest) ([]*compliance.PolicyDecision, error) {
	decisions := make([]*compliance.PolicyDecision, len(requests))

	for i, request := range requests {
		decision, err := e.Evaluate(ctx, request)
		if err != nil {
			return nil, err
		}
		decisions[i] = decision
	}

	return decisions, nil
}

// TestPolicy tests a policy against test cases
func (e *Engine) TestPolicy(policy *compliance.Policy, testCases []compliance.PolicyTestCase) ([]compliance.PolicyTestResult, error) {
	results := make([]compliance.PolicyTestResult, len(testCases))

	// Temporarily add policy for testing
	tempID := generateID("test-pol")
	policy.ID = tempID
	policy.Enabled = true

	e.mu.Lock()
	e.policies[tempID] = policy
	e.mu.Unlock()

	defer func() {
		e.mu.Lock()
		delete(e.policies, tempID)
		e.mu.Unlock()
	}()

	for i, testCase := range testCases {
		decision, err := e.Evaluate(context.Background(), &testCase.Request)
		if err != nil {
			return nil, err
		}

		passed := decision.Decision == testCase.Expected
		results[i] = compliance.PolicyTestResult{
			TestCase: testCase,
			Passed:   passed,
			Actual:   decision.Decision,
			Message:  strings.Join(decision.Reasons, "; "),
		}
	}

	return results, nil
}

// ValidatePolicy validates policy syntax and logic
func (e *Engine) validatePolicy(policy *compliance.Policy) error {
	if policy.Name == "" {
		return fmt.Errorf("policy name required")
	}

	if len(policy.Rules) == 0 {
		return fmt.Errorf("policy must have at least one rule")
	}

	for _, rule := range policy.Rules {
		if rule.Effect != compliance.PolicyEffectAllow && rule.Effect != compliance.PolicyEffectDeny {
			return fmt.Errorf("invalid rule effect: %s", rule.Effect)
		}
	}

	return nil
}

// EnablePolicy enables a policy
func (e *Engine) EnablePolicy(id string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	policy, ok := e.policies[id]
	if !ok {
		return fmt.Errorf("policy not found: %s", id)
	}

	policy.Enabled = true
	e.evaluationCache = make(map[string]*CachedDecision)

	return nil
}

// DisablePolicy disables a policy
func (e *Engine) DisablePolicy(id string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	policy, ok := e.policies[id]
	if !ok {
		return fmt.Errorf("policy not found: %s", id)
	}

	policy.Enabled = false
	e.evaluationCache = make(map[string]*CachedDecision)

	return nil
}

// GetPolicyViolations returns policy violations
func (e *Engine) GetPolicyViolations(policyID string, since time.Time) ([]*compliance.PolicyViolation, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	violations, ok := e.violations[policyID]
	if !ok {
		return []*compliance.PolicyViolation{}, nil
	}

	filtered := make([]*compliance.PolicyViolation, 0)
	for _, v := range violations {
		if v.Timestamp.After(since) {
			filtered = append(filtered, v)
		}
	}

	return filtered, nil
}

// Helper functions

func (e *Engine) policyApplies(policy *compliance.Policy, request *compliance.PolicyRequest) bool {
	// Check if resource matches
	if !e.matchesPattern(policy.Scope.Resources, request.Resource) {
		return false
	}

	// Check if action matches
	if !e.matchesPattern(policy.Scope.Actions, request.Action) {
		return false
	}

	// Check if principal matches
	if !e.matchesPattern(policy.Scope.Principals, request.Principal) {
		return false
	}

	return true
}

func (e *Engine) matchesPattern(patterns []string, value string) bool {
	for _, pattern := range patterns {
		if pattern == "*" {
			return true
		}
		if strings.HasSuffix(pattern, "*") {
			prefix := strings.TrimSuffix(pattern, "*")
			if strings.HasPrefix(value, prefix) {
				return true
			}
		} else if pattern == value {
			return true
		}
	}
	return false
}

func (e *Engine) ruleMatches(rule compliance.PolicyRule, request *compliance.PolicyRequest) bool {
	// Evaluate rule conditions against request context
	for key, value := range rule.Conditions {
		contextValue, ok := request.Context[key]
		if !ok {
			// Context doesn't have required key - rule doesn't match
			return false
		}

		// Simple equality check (in production would support operators like >, <, in, etc.)
		if contextValue != value {
			return false
		}
	}

	return true
}

func (e *Engine) logViolation(policy *compliance.Policy, request *compliance.PolicyRequest) {
	violation := &compliance.PolicyViolation{
		ID:        generateID("violation"),
		PolicyID:  policy.ID,
		Request:   *request,
		Timestamp: time.Now(),
		Severity:  e.determineSeverity(policy),
		Details:   fmt.Sprintf("Policy %s violated by %s", policy.Name, request.Principal),
	}

	// Store violation (in production would write to persistent storage)
	if e.violations[policy.ID] == nil {
		e.violations[policy.ID] = []*compliance.PolicyViolation{}
	}
	e.violations[policy.ID] = append(e.violations[policy.ID], violation)
}

func (e *Engine) determineSeverity(policy *compliance.Policy) string {
	if policy.Priority == 1 {
		return "critical"
	} else if policy.Priority == 2 {
		return "high"
	} else if policy.Priority == 3 {
		return "medium"
	}
	return "low"
}

func (e *Engine) generateCacheKey(request *compliance.PolicyRequest) string {
	data, _ := json.Marshal(request)
	return fmt.Sprintf("%x", data)
}

func (e *Engine) getCachedDecision(key string) (*compliance.PolicyDecision, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	cached, ok := e.evaluationCache[key]
	if !ok {
		return nil, false
	}

	if time.Now().After(cached.ExpiresAt) {
		return nil, false
	}

	return cached.Decision, true
}

func (e *Engine) cacheDecision(key string, decision *compliance.PolicyDecision, ttl time.Duration) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.evaluationCache[key] = &CachedDecision{
		Decision:  decision,
		CachedAt:  time.Now(),
		ExpiresAt: time.Now().Add(ttl),
	}
}

func generateID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}
