// Package policies implements security policy engine with OPA integration
package policies

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// PolicyLanguage represents the policy language
type PolicyLanguage string

const (
	LanguageRego PolicyLanguage = "rego" // Open Policy Agent
	LanguageJSON PolicyLanguage = "json"
	LanguageYAML PolicyLanguage = "yaml"
)

// PolicyType represents the type of security policy
type PolicyType string

const (
	PolicyTypeAccess       PolicyType = "access"
	PolicyTypeData         PolicyType = "data"
	PolicyTypeNetwork      PolicyType = "network"
	PolicyTypeCompliance   PolicyType = "compliance"
	PolicyTypeEncryption   PolicyType = "encryption"
)

// Policy represents a security policy
type Policy struct {
	ID          string
	Name        string
	Type        PolicyType
	Language    PolicyLanguage
	Content     string
	Version     int
	Enabled     bool
	Priority    int
	CreatedAt   time.Time
	UpdatedAt   time.Time
	Metadata    map[string]interface{}
}

// PolicyDecision represents a policy evaluation decision
type PolicyDecision struct {
	PolicyID   string
	Allowed    bool
	Reason     string
	Violations []string
	Metadata   map[string]interface{}
	Timestamp  time.Time
}

// DataClassification represents data classification level
type DataClassification string

const (
	ClassificationPublic       DataClassification = "public"
	ClassificationInternal     DataClassification = "internal"
	ClassificationConfidential DataClassification = "confidential"
	ClassificationRestricted   DataClassification = "restricted"
)

// ComplianceFramework represents a compliance framework
type ComplianceFramework string

const (
	FrameworkGDPR    ComplianceFramework = "gdpr"
	FrameworkHIPAA   ComplianceFramework = "hipaa"
	FrameworkPCIDSS  ComplianceFramework = "pci_dss"
	FrameworkSOC2    ComplianceFramework = "soc2"
	FrameworkISO27001 ComplianceFramework = "iso27001"
)

// Engine implements security policy engine
type Engine struct {
	policies              map[string]*Policy
	opaEnabled            bool
	opaEndpoint           string
	dataClassification    bool
	encryptionRequired    bool
	networkSegmentation   bool
	complianceFrameworks  []ComplianceFramework
	policyCache           map[string]*PolicyDecision
	mu                    sync.RWMutex
	totalEvaluations      int64
	allowedEvaluations    int64
	deniedEvaluations     int64
}

// EvaluationContext represents the context for policy evaluation
type EvaluationContext struct {
	Subject    string
	Action     string
	Resource   string
	Data       map[string]interface{}
	Timestamp  time.Time
}

// NewEngine creates a new policy engine
func NewEngine(opaEnabled bool, opaEndpoint string) *Engine {
	return &Engine{
		policies:              make(map[string]*Policy),
		opaEnabled:            opaEnabled,
		opaEndpoint:           opaEndpoint,
		dataClassification:    true,
		encryptionRequired:    true,
		networkSegmentation:   true,
		complianceFrameworks:  []ComplianceFramework{FrameworkSOC2, FrameworkGDPR},
		policyCache:           make(map[string]*PolicyDecision),
	}
}

// AddPolicy adds a security policy
func (e *Engine) AddPolicy(policy *Policy) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if policy.ID == "" {
		policy.ID = generatePolicyID()
	}

	if policy.Version == 0 {
		policy.Version = 1
	}

	policy.UpdatedAt = time.Now()
	if policy.CreatedAt.IsZero() {
		policy.CreatedAt = time.Now()
	}

	// Validate policy syntax
	if err := e.validatePolicy(policy); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}

	e.policies[policy.ID] = policy
	return nil
}

// validatePolicy validates policy syntax
func (e *Engine) validatePolicy(policy *Policy) error {
	if policy.Content == "" {
		return fmt.Errorf("policy content is empty")
	}

	switch policy.Language {
	case LanguageRego:
		// Validate Rego syntax (simplified)
		if len(policy.Content) < 10 {
			return fmt.Errorf("invalid Rego policy")
		}
	case LanguageJSON:
		// Validate JSON syntax
		var js map[string]interface{}
		if err := json.Unmarshal([]byte(policy.Content), &js); err != nil {
			return fmt.Errorf("invalid JSON policy: %w", err)
		}
	case LanguageYAML:
		// Validate YAML syntax (simplified)
		if len(policy.Content) < 5 {
			return fmt.Errorf("invalid YAML policy")
		}
	default:
		return fmt.Errorf("unsupported policy language: %s", policy.Language)
	}

	return nil
}

// UpdatePolicy updates an existing policy
func (e *Engine) UpdatePolicy(policyID string, updates map[string]interface{}) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	policy, exists := e.policies[policyID]
	if !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	// Create new version
	newPolicy := *policy
	newPolicy.Version++
	newPolicy.UpdatedAt = time.Now()

	// Apply updates
	if content, ok := updates["content"].(string); ok {
		newPolicy.Content = content
	}
	if enabled, ok := updates["enabled"].(bool); ok {
		newPolicy.Enabled = enabled
	}
	if priority, ok := updates["priority"].(int); ok {
		newPolicy.Priority = priority
	}

	// Validate updated policy
	if err := e.validatePolicy(&newPolicy); err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}

	e.policies[policyID] = &newPolicy

	// Clear cache for this policy
	e.clearPolicyCache(policyID)

	return nil
}

// DeletePolicy deletes a policy
func (e *Engine) DeletePolicy(policyID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, exists := e.policies[policyID]; !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	delete(e.policies, policyID)
	e.clearPolicyCache(policyID)

	return nil
}

// Evaluate evaluates policies against a context
func (e *Engine) Evaluate(ctx context.Context, evalCtx *EvaluationContext) (*PolicyDecision, error) {
	// Check cache first
	cacheKey := e.getCacheKey(evalCtx)
	if decision := e.getCachedDecision(cacheKey); decision != nil {
		return decision, nil
	}

	e.mu.Lock()
	e.totalEvaluations++
	e.mu.Unlock()

	decision := &PolicyDecision{
		Allowed:    true,
		Violations: make([]string, 0),
		Metadata:   make(map[string]interface{}),
		Timestamp:  time.Now(),
	}

	// Get applicable policies
	policies := e.getApplicablePolicies(evalCtx)

	// Evaluate each policy
	for _, policy := range policies {
		if !policy.Enabled {
			continue
		}

		policyDecision, err := e.evaluatePolicy(ctx, policy, evalCtx)
		if err != nil {
			return nil, fmt.Errorf("policy evaluation failed: %w", err)
		}

		if !policyDecision.Allowed {
			decision.Allowed = false
			decision.Violations = append(decision.Violations, policyDecision.Violations...)
			decision.PolicyID = policy.ID
		}

		// Merge metadata
		for k, v := range policyDecision.Metadata {
			decision.Metadata[k] = v
		}
	}

	// Apply data classification policies
	if e.dataClassification {
		classificationDecision := e.evaluateDataClassification(evalCtx)
		if !classificationDecision.Allowed {
			decision.Allowed = false
			decision.Violations = append(decision.Violations, classificationDecision.Violations...)
		}
	}

	// Apply encryption policies
	if e.encryptionRequired {
		encryptionDecision := e.evaluateEncryptionPolicy(evalCtx)
		if !encryptionDecision.Allowed {
			decision.Allowed = false
			decision.Violations = append(decision.Violations, encryptionDecision.Violations...)
		}
	}

	// Apply network segmentation policies
	if e.networkSegmentation {
		networkDecision := e.evaluateNetworkPolicy(evalCtx)
		if !networkDecision.Allowed {
			decision.Allowed = false
			decision.Violations = append(decision.Violations, networkDecision.Violations...)
		}
	}

	// Apply compliance framework policies
	for _, framework := range e.complianceFrameworks {
		complianceDecision := e.evaluateCompliancePolicy(framework, evalCtx)
		if !complianceDecision.Allowed {
			decision.Allowed = false
			decision.Violations = append(decision.Violations, complianceDecision.Violations...)
		}
	}

	// Update metrics
	e.mu.Lock()
	if decision.Allowed {
		e.allowedEvaluations++
	} else {
		e.deniedEvaluations++
	}
	e.mu.Unlock()

	// Cache decision
	e.cacheDecision(cacheKey, decision)

	return decision, nil
}

// evaluatePolicy evaluates a single policy
func (e *Engine) evaluatePolicy(ctx context.Context, policy *Policy, evalCtx *EvaluationContext) (*PolicyDecision, error) {
	decision := &PolicyDecision{
		PolicyID:   policy.ID,
		Allowed:    true,
		Violations: make([]string, 0),
		Metadata:   make(map[string]interface{}),
		Timestamp:  time.Now(),
	}

	// If OPA enabled, use OPA for evaluation
	if e.opaEnabled && policy.Language == LanguageRego {
		return e.evaluateOPAPolicy(ctx, policy, evalCtx)
	}

	// Simple rule-based evaluation for JSON/YAML policies
	var rules map[string]interface{}
	if err := json.Unmarshal([]byte(policy.Content), &rules); err != nil {
		// If not JSON, assume simple allow/deny
		if policy.Content == "deny" {
			decision.Allowed = false
			decision.Violations = append(decision.Violations, "Policy denies access")
		}
		return decision, nil
	}

	// Evaluate rules
	if allow, ok := rules["allow"].(bool); ok {
		decision.Allowed = allow
		if !allow {
			decision.Violations = append(decision.Violations, "Policy explicitly denies access")
		}
	}

	return decision, nil
}

// evaluateOPAPolicy evaluates a Rego policy using OPA
func (e *Engine) evaluateOPAPolicy(ctx context.Context, policy *Policy, evalCtx *EvaluationContext) (*PolicyDecision, error) {
	// Simplified OPA evaluation (in production, would use actual OPA SDK)
	decision := &PolicyDecision{
		PolicyID:   policy.ID,
		Allowed:    true,
		Violations: make([]string, 0),
		Metadata:   make(map[string]interface{}),
		Timestamp:  time.Now(),
	}

	// Simulate OPA decision
	decision.Metadata["opa_endpoint"] = e.opaEndpoint
	decision.Metadata["policy_language"] = "rego"

	return decision, nil
}

// evaluateDataClassification evaluates data classification policies
func (e *Engine) evaluateDataClassification(evalCtx *EvaluationContext) *PolicyDecision {
	decision := &PolicyDecision{
		Allowed:    true,
		Violations: make([]string, 0),
		Metadata:   make(map[string]interface{}),
		Timestamp:  time.Now(),
	}

	// Check data classification
	if classification, ok := evalCtx.Data["classification"].(string); ok {
		switch DataClassification(classification) {
		case ClassificationRestricted:
			// Restricted data requires special handling
			if evalCtx.Action != "read" && evalCtx.Action != "write" {
				decision.Allowed = false
				decision.Violations = append(decision.Violations, "Restricted data access violation")
			}
		case ClassificationConfidential:
			// Confidential data requires authorization
			if evalCtx.Subject == "" {
				decision.Allowed = false
				decision.Violations = append(decision.Violations, "Confidential data requires authentication")
			}
		}
	}

	return decision
}

// evaluateEncryptionPolicy evaluates encryption requirements
func (e *Engine) evaluateEncryptionPolicy(evalCtx *EvaluationContext) *PolicyDecision {
	decision := &PolicyDecision{
		Allowed:    true,
		Violations: make([]string, 0),
		Metadata:   make(map[string]interface{}),
		Timestamp:  time.Now(),
	}

	// Check if data is encrypted when required
	if encrypted, ok := evalCtx.Data["encrypted"].(bool); ok {
		if !encrypted && e.encryptionRequired {
			decision.Allowed = false
			decision.Violations = append(decision.Violations, "Encryption required but not enabled")
		}
	}

	return decision
}

// evaluateNetworkPolicy evaluates network segmentation policies
func (e *Engine) evaluateNetworkPolicy(evalCtx *EvaluationContext) *PolicyDecision {
	decision := &PolicyDecision{
		Allowed:    true,
		Violations: make([]string, 0),
		Metadata:   make(map[string]interface{}),
		Timestamp:  time.Now(),
	}

	// Check network segmentation
	if sourceNet, ok := evalCtx.Data["source_network"].(string); ok {
		if destNet, ok := evalCtx.Data["dest_network"].(string); ok {
			// Simplified network policy (in production, would check actual network policies)
			if sourceNet != destNet && e.networkSegmentation {
				decision.Metadata["cross_network"] = true
			}
		}
	}

	return decision
}

// evaluateCompliancePolicy evaluates compliance framework policies
func (e *Engine) evaluateCompliancePolicy(framework ComplianceFramework, evalCtx *EvaluationContext) *PolicyDecision {
	decision := &PolicyDecision{
		Allowed:    true,
		Violations: make([]string, 0),
		Metadata:   make(map[string]interface{}),
		Timestamp:  time.Now(),
	}

	decision.Metadata["compliance_framework"] = framework

	switch framework {
	case FrameworkGDPR:
		// GDPR requires data subject consent
		if consent, ok := evalCtx.Data["gdpr_consent"].(bool); ok {
			if !consent && evalCtx.Action == "process" {
				decision.Allowed = false
				decision.Violations = append(decision.Violations, "GDPR consent required")
			}
		}

	case FrameworkHIPAA:
		// HIPAA requires PHI protection
		if phi, ok := evalCtx.Data["phi"].(bool); ok {
			if phi {
				if encrypted, ok := evalCtx.Data["encrypted"].(bool); !ok || !encrypted {
					decision.Allowed = false
					decision.Violations = append(decision.Violations, "HIPAA requires PHI encryption")
				}
			}
		}

	case FrameworkPCIDSS:
		// PCI DSS requires cardholder data protection
		if cardData, ok := evalCtx.Data["cardholder_data"].(bool); ok {
			if cardData {
				if tokenized, ok := evalCtx.Data["tokenized"].(bool); !ok || !tokenized {
					decision.Allowed = false
					decision.Violations = append(decision.Violations, "PCI DSS requires cardholder data tokenization")
				}
			}
		}
	}

	return decision
}

// getApplicablePolicies returns policies applicable to the context
func (e *Engine) getApplicablePolicies(evalCtx *EvaluationContext) []*Policy {
	e.mu.RLock()
	defer e.mu.RUnlock()

	var applicable []*Policy
	for _, policy := range e.policies {
		if e.isPolicyApplicable(policy, evalCtx) {
			applicable = append(applicable, policy)
		}
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

// isPolicyApplicable checks if a policy is applicable to the context
func (e *Engine) isPolicyApplicable(policy *Policy, evalCtx *EvaluationContext) bool {
	// Simple applicability check (can be enhanced with more sophisticated matching)
	return policy.Enabled
}

// getCacheKey generates a cache key for policy decision
func (e *Engine) getCacheKey(evalCtx *EvaluationContext) string {
	h := sha256.New()
	h.Write([]byte(evalCtx.Subject))
	h.Write([]byte(evalCtx.Action))
	h.Write([]byte(evalCtx.Resource))
	return hex.EncodeToString(h.Sum(nil))
}

// getCachedDecision retrieves a cached decision
func (e *Engine) getCachedDecision(key string) *PolicyDecision {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.policyCache[key]
}

// cacheDecision caches a policy decision
func (e *Engine) cacheDecision(key string, decision *PolicyDecision) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Simple cache with size limit
	if len(e.policyCache) > 10000 {
		// Clear half of cache
		count := 0
		for k := range e.policyCache {
			delete(e.policyCache, k)
			count++
			if count > 5000 {
				break
			}
		}
	}

	e.policyCache[key] = decision
}

// clearPolicyCache clears cache for a specific policy
func (e *Engine) clearPolicyCache(policyID string) {
	// Clear entire cache (simplified - could be more selective)
	e.policyCache = make(map[string]*PolicyDecision)
}

// GetMetrics returns policy engine metrics
func (e *Engine) GetMetrics() map[string]interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	enabledPolicies := 0
	for _, policy := range e.policies {
		if policy.Enabled {
			enabledPolicies++
		}
	}

	allowRate := 0.0
	if e.totalEvaluations > 0 {
		allowRate = float64(e.allowedEvaluations) / float64(e.totalEvaluations)
	}

	return map[string]interface{}{
		"total_policies":         len(e.policies),
		"enabled_policies":       enabledPolicies,
		"total_evaluations":      e.totalEvaluations,
		"allowed_evaluations":    e.allowedEvaluations,
		"denied_evaluations":     e.deniedEvaluations,
		"allow_rate":             allowRate,
		"opa_enabled":            e.opaEnabled,
		"data_classification":    e.dataClassification,
		"encryption_required":    e.encryptionRequired,
		"network_segmentation":   e.networkSegmentation,
		"compliance_frameworks":  e.complianceFrameworks,
		"cache_size":             len(e.policyCache),
	}
}

// Helper functions

func generatePolicyID() string {
	b := make([]byte, 16)
	// Use crypto/rand for secure random
	_, _ = rand.Read(b)
	return fmt.Sprintf("policy-%s", hex.EncodeToString(b))
}
