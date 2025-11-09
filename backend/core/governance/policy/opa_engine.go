package policy

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// PolicyCategory represents a policy category
type PolicyCategory string

const (
	CategoryAccessControl PolicyCategory = "access-control"
	CategoryQuota         PolicyCategory = "quota"
	CategoryNetwork       PolicyCategory = "network"
	CategoryDataResidency PolicyCategory = "data-residency"
	CategoryCompliance    PolicyCategory = "compliance"
	CategorySecurity      PolicyCategory = "security"
)

// PolicyDecision represents a policy evaluation decision
type PolicyDecision string

const (
	DecisionAllow PolicyDecision = "allow"
	DecisionDeny  PolicyDecision = "deny"
)

// Policy represents an OPA policy
type Policy struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Version     string         `json:"version"`
	Category    PolicyCategory `json:"category"`
	Description string         `json:"description"`
	Rego        string         `json:"rego"` // Rego policy code
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
	Enabled     bool           `json:"enabled"`
	Priority    int            `json:"priority"` // Higher priority = evaluated first
}

// PolicyEvaluationRequest represents a policy evaluation request
type PolicyEvaluationRequest struct {
	PolicyID string                 `json:"policy_id"`
	Input    map[string]interface{} `json:"input"`
	Context  PolicyContext          `json:"context"`
}

// PolicyContext provides context for policy evaluation
type PolicyContext struct {
	Timestamp time.Time         `json:"timestamp"`
	TenantID  string            `json:"tenant_id"`
	UserID    string            `json:"user_id"`
	Roles     []string          `json:"roles"`
	Attributes map[string]string `json:"attributes"`
}

// PolicyEvaluationResult represents the result of policy evaluation
type PolicyEvaluationResult struct {
	PolicyID       string         `json:"policy_id"`
	Decision       PolicyDecision `json:"decision"`
	Allowed        bool           `json:"allowed"`
	Reasons        []string       `json:"reasons"`
	EvaluationTime time.Duration  `json:"evaluation_time"`
	Timestamp      time.Time      `json:"timestamp"`
}

// OPAEngine implements Open Policy Agent integration
type OPAEngine struct {
	mu                sync.RWMutex
	policies          map[string]*Policy
	policyCache       *PolicyCache
	evaluationTimeout time.Duration
	performanceTarget time.Duration
	versionControl    *VersionControl
	metrics           *PolicyMetrics
}

// PolicyCache caches policy evaluation results
type PolicyCache struct {
	mu      sync.RWMutex
	cache   map[string]*CacheEntry
	ttl     time.Duration
	enabled bool
}

type CacheEntry struct {
	Result    *PolicyEvaluationResult
	ExpiresAt time.Time
}

// VersionControl manages policy versions
type VersionControl struct {
	mu             sync.RWMutex
	versions       map[string][]*Policy // PolicyID -> versions
	activeVersions map[string]string    // PolicyID -> active version
	rollbackEnabled bool
}

// PolicyMetrics tracks policy engine metrics
type PolicyMetrics struct {
	mu                     sync.RWMutex
	TotalEvaluations       int64
	EvaluationsByPolicy    map[string]int64
	EvaluationsByDecision  map[PolicyDecision]int64
	AverageEvaluationTime  time.Duration
	CacheHitRate           float64
	PolicyViolations       int64
	SlowEvaluations        int64 // Evaluations > target
}

// NewOPAEngine creates a new OPA policy engine
func NewOPAEngine(evaluationTimeout, performanceTarget time.Duration, cacheEnabled bool, cacheTTL time.Duration) *OPAEngine {
	return &OPAEngine{
		policies:          make(map[string]*Policy),
		policyCache:       newPolicyCache(cacheEnabled, cacheTTL),
		evaluationTimeout: evaluationTimeout,
		performanceTarget: performanceTarget,
		versionControl:    newVersionControl(true),
		metrics: &PolicyMetrics{
			EvaluationsByPolicy:   make(map[string]int64),
			EvaluationsByDecision: make(map[PolicyDecision]int64),
		},
	}
}

func newPolicyCache(enabled bool, ttl time.Duration) *PolicyCache {
	return &PolicyCache{
		cache:   make(map[string]*CacheEntry),
		ttl:     ttl,
		enabled: enabled,
	}
}

func newVersionControl(rollbackEnabled bool) *VersionControl {
	return &VersionControl{
		versions:        make(map[string][]*Policy),
		activeVersions:  make(map[string]string),
		rollbackEnabled: rollbackEnabled,
	}
}

// AddPolicy adds a new policy to the engine
func (oe *OPAEngine) AddPolicy(ctx context.Context, policy *Policy) error {
	oe.mu.Lock()
	defer oe.mu.Unlock()

	if policy.ID == "" {
		policy.ID = fmt.Sprintf("policy-%d", time.Now().UnixNano())
	}

	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()
	policy.Enabled = true

	if policy.Version == "" {
		policy.Version = "1.0.0"
	}

	oe.policies[policy.ID] = policy

	// Add to version control
	oe.versionControl.addVersion(policy)

	return nil
}

// UpdatePolicy updates an existing policy
func (oe *OPAEngine) UpdatePolicy(ctx context.Context, policyID string, updates *Policy) error {
	oe.mu.Lock()
	defer oe.mu.Unlock()

	policy, exists := oe.policies[policyID]
	if !exists {
		return fmt.Errorf("policy %s not found", policyID)
	}

	// Increment version
	newVersion := incrementVersion(policy.Version)
	updates.Version = newVersion
	updates.UpdatedAt = time.Now()
	updates.ID = policyID

	// Add to version control
	oe.versionControl.addVersion(updates)

	oe.policies[policyID] = updates

	// Invalidate cache for this policy
	oe.policyCache.invalidate(policyID)

	return nil
}

// EvaluatePolicy evaluates a policy against input
func (oe *OPAEngine) EvaluatePolicy(ctx context.Context, request *PolicyEvaluationRequest) (*PolicyEvaluationResult, error) {
	startTime := time.Now()

	// Check cache first
	if oe.policyCache.enabled {
		if cached := oe.policyCache.get(request); cached != nil {
			oe.updateCacheMetrics(true)
			return cached, nil
		}
		oe.updateCacheMetrics(false)
	}

	oe.mu.RLock()
	policy, exists := oe.policies[request.PolicyID]
	oe.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("policy %s not found", request.PolicyID)
	}

	if !policy.Enabled {
		return &PolicyEvaluationResult{
			PolicyID:  policy.ID,
			Decision:  DecisionAllow,
			Allowed:   true,
			Reasons:   []string{"Policy disabled"},
			Timestamp: time.Now(),
		}, nil
	}

	// Evaluate policy (simulate OPA evaluation)
	result := oe.evaluateRego(ctx, policy, request)

	evaluationTime := time.Since(startTime)
	result.EvaluationTime = evaluationTime
	result.Timestamp = time.Now()

	// Update metrics
	oe.updateMetrics(policy, result, evaluationTime)

	// Cache result
	if oe.policyCache.enabled {
		oe.policyCache.set(request, result)
	}

	// Check performance target
	if evaluationTime > oe.performanceTarget {
		oe.metrics.mu.Lock()
		oe.metrics.SlowEvaluations++
		oe.metrics.mu.Unlock()
	}

	return result, nil
}

// evaluateRego simulates Rego policy evaluation
func (oe *OPAEngine) evaluateRego(ctx context.Context, policy *Policy, request *PolicyEvaluationRequest) *PolicyEvaluationResult {
	result := &PolicyEvaluationResult{
		PolicyID: policy.ID,
		Reasons:  make([]string, 0),
	}

	// Simulate policy evaluation based on category
	switch policy.Category {
	case CategoryAccessControl:
		result = oe.evaluateAccessControlPolicy(request)
	case CategoryQuota:
		result = oe.evaluateQuotaPolicy(request)
	case CategoryNetwork:
		result = oe.evaluateNetworkPolicy(request)
	case CategoryDataResidency:
		result = oe.evaluateDataResidencyPolicy(request)
	case CategoryCompliance:
		result = oe.evaluateCompliancePolicy(request)
	default:
		result.Decision = DecisionAllow
		result.Allowed = true
		result.Reasons = append(result.Reasons, "Default allow")
	}

	return result
}

// evaluateAccessControlPolicy evaluates access control policy
func (oe *OPAEngine) evaluateAccessControlPolicy(request *PolicyEvaluationRequest) *PolicyEvaluationResult {
	result := &PolicyEvaluationResult{
		PolicyID: request.PolicyID,
		Reasons:  make([]string, 0),
	}

	// Example: Check if user has required role
	requiredRole, hasRole := request.Input["required_role"].(string)
	if hasRole {
		userRoles := request.Context.Roles
		roleFound := false
		for _, role := range userRoles {
			if role == requiredRole {
				roleFound = true
				break
			}
		}

		if roleFound {
			result.Decision = DecisionAllow
			result.Allowed = true
			result.Reasons = append(result.Reasons, fmt.Sprintf("User has required role: %s", requiredRole))
		} else {
			result.Decision = DecisionDeny
			result.Allowed = false
			result.Reasons = append(result.Reasons, fmt.Sprintf("User missing required role: %s", requiredRole))
		}
	} else {
		result.Decision = DecisionAllow
		result.Allowed = true
		result.Reasons = append(result.Reasons, "No role requirement specified")
	}

	return result
}

// evaluateQuotaPolicy evaluates quota policy
func (oe *OPAEngine) evaluateQuotaPolicy(request *PolicyEvaluationRequest) *PolicyEvaluationResult {
	result := &PolicyEvaluationResult{
		PolicyID: request.PolicyID,
		Reasons:  make([]string, 0),
	}

	// Example: Check resource quota
	currentUsage, _ := request.Input["current_usage"].(float64)
	quotaLimit, _ := request.Input["quota_limit"].(float64)

	if currentUsage < quotaLimit {
		result.Decision = DecisionAllow
		result.Allowed = true
		result.Reasons = append(result.Reasons, fmt.Sprintf("Within quota: %.0f/%.0f", currentUsage, quotaLimit))
	} else {
		result.Decision = DecisionDeny
		result.Allowed = false
		result.Reasons = append(result.Reasons, fmt.Sprintf("Quota exceeded: %.0f/%.0f", currentUsage, quotaLimit))
	}

	return result
}

// evaluateNetworkPolicy evaluates network policy
func (oe *OPAEngine) evaluateNetworkPolicy(request *PolicyEvaluationRequest) *PolicyEvaluationResult {
	result := &PolicyEvaluationResult{
		PolicyID: request.PolicyID,
		Reasons:  make([]string, 0),
	}

	// Example: Check allowed network ranges
	sourceIP, _ := request.Input["source_ip"].(string)
	allowedRanges, _ := request.Input["allowed_ranges"].([]string)

	if len(allowedRanges) == 0 {
		result.Decision = DecisionAllow
		result.Allowed = true
		result.Reasons = append(result.Reasons, "No network restrictions")
	} else {
		// Simplified IP range check
		result.Decision = DecisionAllow
		result.Allowed = true
		result.Reasons = append(result.Reasons, fmt.Sprintf("Source IP %s validated", sourceIP))
	}

	return result
}

// evaluateDataResidencyPolicy evaluates data residency policy
func (oe *OPAEngine) evaluateDataResidencyPolicy(request *PolicyEvaluationRequest) *PolicyEvaluationResult {
	result := &PolicyEvaluationResult{
		PolicyID: request.PolicyID,
		Reasons:  make([]string, 0),
	}

	// Example: Check data location requirements
	dataLocation, _ := request.Input["data_location"].(string)
	allowedLocations, _ := request.Input["allowed_locations"].([]string)

	locationAllowed := false
	for _, allowed := range allowedLocations {
		if dataLocation == allowed {
			locationAllowed = true
			break
		}
	}

	if locationAllowed || len(allowedLocations) == 0 {
		result.Decision = DecisionAllow
		result.Allowed = true
		result.Reasons = append(result.Reasons, fmt.Sprintf("Data location %s is allowed", dataLocation))
	} else {
		result.Decision = DecisionDeny
		result.Allowed = false
		result.Reasons = append(result.Reasons, fmt.Sprintf("Data location %s not allowed", dataLocation))
	}

	return result
}

// evaluateCompliancePolicy evaluates compliance policy
func (oe *OPAEngine) evaluateCompliancePolicy(request *PolicyEvaluationRequest) *PolicyEvaluationResult {
	result := &PolicyEvaluationResult{
		PolicyID: request.PolicyID,
		Reasons:  make([]string, 0),
	}

	// Example: Check compliance requirements
	encryptionEnabled, _ := request.Input["encryption_enabled"].(bool)
	auditLoggingEnabled, _ := request.Input["audit_logging_enabled"].(bool)

	if encryptionEnabled && auditLoggingEnabled {
		result.Decision = DecisionAllow
		result.Allowed = true
		result.Reasons = append(result.Reasons, "All compliance requirements met")
	} else {
		result.Decision = DecisionDeny
		result.Allowed = false
		result.Reasons = append(result.Reasons, "Compliance requirements not met")

		if !encryptionEnabled {
			result.Reasons = append(result.Reasons, "Encryption not enabled")
		}
		if !auditLoggingEnabled {
			result.Reasons = append(result.Reasons, "Audit logging not enabled")
		}
	}

	return result
}

// updateMetrics updates policy metrics
func (oe *OPAEngine) updateMetrics(policy *Policy, result *PolicyEvaluationResult, evaluationTime time.Duration) {
	oe.metrics.mu.Lock()
	defer oe.metrics.mu.Unlock()

	oe.metrics.TotalEvaluations++
	oe.metrics.EvaluationsByPolicy[policy.ID]++
	oe.metrics.EvaluationsByDecision[result.Decision]++

	// Update average evaluation time
	if oe.metrics.AverageEvaluationTime == 0 {
		oe.metrics.AverageEvaluationTime = evaluationTime
	} else {
		// Exponential moving average
		alpha := 0.1
		oe.metrics.AverageEvaluationTime = time.Duration(
			float64(oe.metrics.AverageEvaluationTime)*(1-alpha) +
				float64(evaluationTime)*alpha,
		)
	}

	if result.Decision == DecisionDeny {
		oe.metrics.PolicyViolations++
	}
}

// updateCacheMetrics updates cache metrics
func (oe *OPAEngine) updateCacheMetrics(hit bool) {
	oe.metrics.mu.Lock()
	defer oe.metrics.mu.Unlock()

	// Update cache hit rate
	if hit {
		oe.metrics.CacheHitRate = (oe.metrics.CacheHitRate*float64(oe.metrics.TotalEvaluations) + 1) /
			float64(oe.metrics.TotalEvaluations+1)
	} else {
		oe.metrics.CacheHitRate = (oe.metrics.CacheHitRate * float64(oe.metrics.TotalEvaluations)) /
			float64(oe.metrics.TotalEvaluations+1)
	}
}

// get retrieves cached policy result
func (pc *PolicyCache) get(request *PolicyEvaluationRequest) *PolicyEvaluationResult {
	if !pc.enabled {
		return nil
	}

	pc.mu.RLock()
	defer pc.mu.RUnlock()

	cacheKey := pc.generateCacheKey(request)
	entry, exists := pc.cache[cacheKey]

	if !exists {
		return nil
	}

	if time.Now().After(entry.ExpiresAt) {
		return nil
	}

	return entry.Result
}

// set caches policy result
func (pc *PolicyCache) set(request *PolicyEvaluationRequest, result *PolicyEvaluationResult) {
	if !pc.enabled {
		return
	}

	pc.mu.Lock()
	defer pc.mu.Unlock()

	cacheKey := pc.generateCacheKey(request)
	pc.cache[cacheKey] = &CacheEntry{
		Result:    result,
		ExpiresAt: time.Now().Add(pc.ttl),
	}
}

// generateCacheKey generates cache key for request
func (pc *PolicyCache) generateCacheKey(request *PolicyEvaluationRequest) string {
	data, _ := json.Marshal(request)
	return fmt.Sprintf("%x", data)
}

// invalidate invalidates cache for policy
func (pc *PolicyCache) invalidate(policyID string) {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	// Remove all cache entries for this policy
	for key, entry := range pc.cache {
		if entry.Result.PolicyID == policyID {
			delete(pc.cache, key)
		}
	}
}

// addVersion adds a policy version
func (vc *VersionControl) addVersion(policy *Policy) {
	vc.mu.Lock()
	defer vc.mu.Unlock()

	if _, exists := vc.versions[policy.ID]; !exists {
		vc.versions[policy.ID] = make([]*Policy, 0)
	}

	vc.versions[policy.ID] = append(vc.versions[policy.ID], policy)
	vc.activeVersions[policy.ID] = policy.Version
}

// RollbackPolicy rolls back policy to previous version
func (oe *OPAEngine) RollbackPolicy(ctx context.Context, policyID, targetVersion string) error {
	oe.versionControl.mu.Lock()
	defer oe.versionControl.mu.Unlock()

	if !oe.versionControl.rollbackEnabled {
		return fmt.Errorf("policy rollback not enabled")
	}

	versions, exists := oe.versionControl.versions[policyID]
	if !exists {
		return fmt.Errorf("policy %s not found", policyID)
	}

	var targetPolicy *Policy
	for _, version := range versions {
		if version.Version == targetVersion {
			targetPolicy = version
			break
		}
	}

	if targetPolicy == nil {
		return fmt.Errorf("version %s not found for policy %s", targetVersion, policyID)
	}

	oe.mu.Lock()
	oe.policies[policyID] = targetPolicy
	oe.mu.Unlock()

	oe.versionControl.activeVersions[policyID] = targetVersion

	// Invalidate cache
	oe.policyCache.invalidate(policyID)

	return nil
}

// GetMetrics returns policy engine metrics
func (oe *OPAEngine) GetMetrics() *PolicyMetrics {
	oe.metrics.mu.RLock()
	defer oe.metrics.mu.RUnlock()

	metrics := &PolicyMetrics{
		TotalEvaluations:      oe.metrics.TotalEvaluations,
		EvaluationsByPolicy:   make(map[string]int64),
		EvaluationsByDecision: make(map[PolicyDecision]int64),
		AverageEvaluationTime: oe.metrics.AverageEvaluationTime,
		CacheHitRate:          oe.metrics.CacheHitRate,
		PolicyViolations:      oe.metrics.PolicyViolations,
		SlowEvaluations:       oe.metrics.SlowEvaluations,
	}

	for k, v := range oe.metrics.EvaluationsByPolicy {
		metrics.EvaluationsByPolicy[k] = v
	}

	for k, v := range oe.metrics.EvaluationsByDecision {
		metrics.EvaluationsByDecision[k] = v
	}

	return metrics
}

// incrementVersion increments semantic version
func incrementVersion(version string) string {
	// Simplified version incrementing
	return fmt.Sprintf("%s.1", version)
}

// GetPoliciesByCategory returns policies by category
func (oe *OPAEngine) GetPoliciesByCategory(category PolicyCategory) []*Policy {
	oe.mu.RLock()
	defer oe.mu.RUnlock()

	policies := make([]*Policy, 0)
	for _, policy := range oe.policies {
		if policy.Category == category && policy.Enabled {
			policies = append(policies, policy)
		}
	}

	return policies
}

// LoadPolicyBundle loads a bundle of policies
func (oe *OPAEngine) LoadPolicyBundle(ctx context.Context, bundleURL string) error {
	// In production, this would fetch and load policies from a bundle
	// For now, we'll create default policies

	defaultPolicies := oe.createDefaultPolicies()

	for _, policy := range defaultPolicies {
		if err := oe.AddPolicy(ctx, policy); err != nil {
			return err
		}
	}

	return nil
}

// createDefaultPolicies creates default policies
func (oe *OPAEngine) createDefaultPolicies() []*Policy {
	return []*Policy{
		{
			Name:        "Admin Access Control",
			Category:    CategoryAccessControl,
			Description: "Require admin role for administrative actions",
			Rego:        "package authz\nallow { input.user.role == \"admin\" }",
			Version:     "1.0.0",
			Priority:    100,
		},
		{
			Name:        "CPU Quota Enforcement",
			Category:    CategoryQuota,
			Description: "Enforce CPU quota limits",
			Rego:        "package quota\nallow { input.cpu_usage < input.cpu_limit }",
			Version:     "1.0.0",
			Priority:    90,
		},
		{
			Name:        "EU Data Residency",
			Category:    CategoryDataResidency,
			Description: "Enforce EU data residency requirements",
			Rego:        "package residency\nallow { input.region == \"eu-west-1\" }",
			Version:     "1.0.0",
			Priority:    95,
		},
		{
			Name:        "Encryption Required",
			Category:    CategoryCompliance,
			Description: "Require encryption for all data",
			Rego:        "package compliance\nallow { input.encryption_enabled == true }",
			Version:     "1.0.0",
			Priority:    100,
		},
	}
}
