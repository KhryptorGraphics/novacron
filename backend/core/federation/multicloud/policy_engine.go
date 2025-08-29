package multicloud

import (
	"context"
	"fmt"
	"time"
)

// MultiCloudPolicyEngine manages policies across multiple cloud providers
type MultiCloudPolicyEngine struct {
	policies map[string]*MultiCloudPolicy
}

// NewMultiCloudPolicyEngine creates a new multi-cloud policy engine
func NewMultiCloudPolicyEngine() *MultiCloudPolicyEngine {
	return &MultiCloudPolicyEngine{
		policies: make(map[string]*MultiCloudPolicy),
	}
}

// SetPolicy sets a multi-cloud policy
func (e *MultiCloudPolicyEngine) SetPolicy(ctx context.Context, policy *MultiCloudPolicy) error {
	if policy.ID == "" {
		return fmt.Errorf("policy ID cannot be empty")
	}

	policy.UpdatedAt = time.Now()
	e.policies[policy.ID] = policy
	return nil
}

// GetPolicy retrieves a policy by ID
func (e *MultiCloudPolicyEngine) GetPolicy(policyID string) (*MultiCloudPolicy, error) {
	policy, exists := e.policies[policyID]
	if !exists {
		return nil, fmt.Errorf("policy %s not found", policyID)
	}
	return policy, nil
}

// ListPolicies lists all policies
func (e *MultiCloudPolicyEngine) ListPolicies() []*MultiCloudPolicy {
	var policies []*MultiCloudPolicy
	for _, policy := range e.policies {
		policies = append(policies, policy)
	}
	return policies
}

// EvaluateVMCreation evaluates policies for VM creation
func (e *MultiCloudPolicyEngine) EvaluateVMCreation(ctx context.Context, providerID string, request *UnifiedVMRequest) error {
	for _, policy := range e.policies {
		if policy.Enabled && e.policyAppliesForVMCreation(policy, providerID, request) {
			if err := e.evaluateVMCreationPolicy(ctx, policy, providerID, request); err != nil {
				return fmt.Errorf("policy %s violation: %v", policy.Name, err)
			}
		}
	}
	return nil
}

// EvaluateMigration evaluates policies for migration
func (e *MultiCloudPolicyEngine) EvaluateMigration(ctx context.Context, request *CrossCloudMigrationRequest) error {
	for _, policy := range e.policies {
		if policy.Enabled && e.policyAppliesForMigration(policy, request) {
			if err := e.evaluateMigrationPolicy(ctx, policy, request); err != nil {
				return fmt.Errorf("policy %s violation: %v", policy.Name, err)
			}
		}
	}
	return nil
}

// Helper methods

func (e *MultiCloudPolicyEngine) policyAppliesForVMCreation(policy *MultiCloudPolicy, providerID string, request *UnifiedVMRequest) bool {
	// Check if policy applies to this provider
	if len(policy.ApplicableProviders) > 0 {
		found := false
		for _, p := range policy.ApplicableProviders {
			if p == providerID {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check if policy applies to this region
	if len(policy.ApplicableRegions) > 0 {
		found := false
		for _, r := range policy.ApplicableRegions {
			if r == request.Region {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check policy type
	return policy.Type == "vm_creation" || policy.Type == "all"
}

func (e *MultiCloudPolicyEngine) policyAppliesForMigration(policy *MultiCloudPolicy, request *CrossCloudMigrationRequest) bool {
	// Check if policy applies to source or destination provider
	if len(policy.ApplicableProviders) > 0 {
		sourceFound := false
		destFound := false
		for _, p := range policy.ApplicableProviders {
			if p == request.SourceProviderID {
				sourceFound = true
			}
			if p == request.DestinationProviderID {
				destFound = true
			}
		}
		if !sourceFound && !destFound {
			return false
		}
	}

	// Check policy type
	return policy.Type == "migration" || policy.Type == "all"
}

func (e *MultiCloudPolicyEngine) evaluateVMCreationPolicy(ctx context.Context, policy *MultiCloudPolicy, providerID string, request *UnifiedVMRequest) error {
	for _, rule := range policy.Rules {
		if err := e.evaluateVMCreationRule(ctx, rule, providerID, request); err != nil {
			if rule.Enforcement == "hard" {
				return err
			}
			// For soft enforcement, log warning but continue
			fmt.Printf("Warning: policy rule %s violated: %v\n", rule.Name, err)
		}
	}
	return nil
}

func (e *MultiCloudPolicyEngine) evaluateMigrationPolicy(ctx context.Context, policy *MultiCloudPolicy, request *CrossCloudMigrationRequest) error {
	for _, rule := range policy.Rules {
		if err := e.evaluateMigrationRule(ctx, rule, request); err != nil {
			if rule.Enforcement == "hard" {
				return err
			}
			fmt.Printf("Warning: migration policy rule %s violated: %v\n", rule.Name, err)
		}
	}
	return nil
}

func (e *MultiCloudPolicyEngine) evaluateVMCreationRule(ctx context.Context, rule MultiCloudPolicyRule, providerID string, request *UnifiedVMRequest) error {
	switch rule.Type {
	case "resource_limits":
		return e.evaluateResourceLimitsRule(rule, request)
	case "cost_control":
		return e.evaluateCostControlRule(rule, request)
	case "security_requirements":
		return e.evaluateSecurityRequirementsRule(rule, request)
	case "provider_preference":
		return e.evaluateProviderPreferenceRule(rule, providerID)
	case "workload_placement":
		return e.evaluateWorkloadPlacementRule(rule, providerID, request)
	default:
		return fmt.Errorf("unknown rule type: %s", rule.Type)
	}
}

func (e *MultiCloudPolicyEngine) evaluateMigrationRule(ctx context.Context, rule MultiCloudPolicyRule, request *CrossCloudMigrationRequest) error {
	switch rule.Type {
	case "migration_approval":
		return e.evaluateMigrationApprovalRule(rule, request)
	case "cost_impact":
		return e.evaluateCostImpactRule(rule, request)
	case "security_compliance":
		return e.evaluateSecurityComplianceRule(rule, request)
	case "business_hours":
		return e.evaluateBusinessHoursRule(rule, request)
	default:
		return fmt.Errorf("unknown migration rule type: %s", rule.Type)
	}
}

func (e *MultiCloudPolicyEngine) evaluateResourceLimitsRule(rule MultiCloudPolicyRule, request *UnifiedVMRequest) error {
	if maxCPU, ok := rule.Parameters["max_cpu"].(int); ok {
		if request.CPU > maxCPU {
			return fmt.Errorf("CPU request %d exceeds policy limit %d", request.CPU, maxCPU)
		}
	}

	if maxMemory, ok := rule.Parameters["max_memory"].(int64); ok {
		if request.Memory > maxMemory {
			return fmt.Errorf("memory request %d exceeds policy limit %d", request.Memory, maxMemory)
		}
	}

	return nil
}

func (e *MultiCloudPolicyEngine) evaluateCostControlRule(rule MultiCloudPolicyRule, request *UnifiedVMRequest) error {
	if maxHourlyCost, ok := rule.Parameters["max_hourly_cost"].(float64); ok {
		// This would need actual cost calculation based on instance type and provider
		// For now, use a simple estimate
		estimatedCost := float64(request.CPU) * 0.05 + float64(request.Memory)/1024 * 0.01
		if estimatedCost > maxHourlyCost {
			return fmt.Errorf("estimated hourly cost %.2f exceeds policy limit %.2f", estimatedCost, maxHourlyCost)
		}
	}

	return nil
}

func (e *MultiCloudPolicyEngine) evaluateSecurityRequirementsRule(rule MultiCloudPolicyRule, request *UnifiedVMRequest) error {
	if encryptionRequired, ok := rule.Parameters["encryption_required"].(bool); ok && encryptionRequired {
		// Check if encryption is configured
		if encrypted, ok := request.CustomOptions["encrypted"].(bool); !ok || !encrypted {
			return fmt.Errorf("encryption is required but not configured")
		}
	}

	if minSecurityGroups, ok := rule.Parameters["min_security_groups"].(int); ok {
		if len(request.SecurityGroups) < minSecurityGroups {
			return fmt.Errorf("minimum %d security groups required, got %d", minSecurityGroups, len(request.SecurityGroups))
		}
	}

	return nil
}

func (e *MultiCloudPolicyEngine) evaluateProviderPreferenceRule(rule MultiCloudPolicyRule, providerID string) error {
	if preferredProviders, ok := rule.Parameters["preferred_providers"].([]string); ok {
		for _, preferred := range preferredProviders {
			if preferred == providerID {
				return nil // Provider is in preferred list
			}
		}
	}

	if blacklistedProviders, ok := rule.Parameters["blacklisted_providers"].([]string); ok {
		for _, blacklisted := range blacklistedProviders {
			if blacklisted == providerID {
				return fmt.Errorf("provider %s is blacklisted", providerID)
			}
		}
	}

	return nil
}

func (e *MultiCloudPolicyEngine) evaluateWorkloadPlacementRule(rule MultiCloudPolicyRule, providerID string, request *UnifiedVMRequest) error {
	// Evaluate workload placement based on workload characteristics
	if workloadType, ok := request.Tags["workload_type"]; ok {
		if preferredProviders, ok := rule.Parameters[workloadType].([]string); ok {
			for _, preferred := range preferredProviders {
				if preferred == providerID {
					return nil
				}
			}
			return fmt.Errorf("provider %s not preferred for workload type %s", providerID, workloadType)
		}
	}

	return nil
}

func (e *MultiCloudPolicyEngine) evaluateMigrationApprovalRule(rule MultiCloudPolicyRule, request *CrossCloudMigrationRequest) error {
	if approvalRequired, ok := rule.Parameters["approval_required"].(bool); ok && approvalRequired {
		// Check if migration has approval
		if approval, ok := request.Options["approval_id"]; !ok || approval == "" {
			return fmt.Errorf("migration approval required but not provided")
		}
	}

	return nil
}

func (e *MultiCloudPolicyEngine) evaluateCostImpactRule(rule MultiCloudPolicyRule, request *CrossCloudMigrationRequest) error {
	if maxCostIncrease, ok := rule.Parameters["max_cost_increase_percent"].(float64); ok {
		// This would need actual cost calculation
		// For now, assume migration might increase cost by 10%
		estimatedIncrease := 10.0
		if estimatedIncrease > maxCostIncrease {
			return fmt.Errorf("estimated cost increase %.1f%% exceeds limit %.1f%%", estimatedIncrease, maxCostIncrease)
		}
	}

	return nil
}

func (e *MultiCloudPolicyEngine) evaluateSecurityComplianceRule(rule MultiCloudPolicyRule, request *CrossCloudMigrationRequest) error {
	if maintainCompliance, ok := rule.Parameters["maintain_compliance"].(bool); ok && maintainCompliance {
		// This would check that destination provider maintains same compliance level
		// For now, just pass
	}

	return nil
}

func (e *MultiCloudPolicyEngine) evaluateBusinessHoursRule(rule MultiCloudPolicyRule, request *CrossCloudMigrationRequest) error {
	if businessHoursOnly, ok := rule.Parameters["business_hours_only"].(bool); ok && businessHoursOnly {
		now := time.Now()
		hour := now.Hour()
		
		// Business hours: 9 AM to 5 PM
		if hour < 9 || hour > 17 {
			return fmt.Errorf("migration outside business hours not allowed")
		}

		// No weekends
		if now.Weekday() == time.Saturday || now.Weekday() == time.Sunday {
			return fmt.Errorf("migration on weekends not allowed")
		}
	}

	return nil
}

// Policy types

// MultiCloudPolicy represents a multi-cloud policy
type MultiCloudPolicy struct {
	ID                  string                  `json:"id"`
	Name                string                  `json:"name"`
	Description         string                  `json:"description"`
	Type                string                  `json:"type"` // vm_creation, migration, all
	Enabled             bool                    `json:"enabled"`
	Priority            int                     `json:"priority"`
	Rules               []MultiCloudPolicyRule  `json:"rules"`
	ApplicableProviders []string                `json:"applicable_providers,omitempty"`
	ApplicableRegions   []string                `json:"applicable_regions,omitempty"`
	CreatedAt           time.Time               `json:"created_at"`
	UpdatedAt           time.Time               `json:"updated_at"`
}

// MultiCloudPolicyRule represents a rule within a multi-cloud policy
type MultiCloudPolicyRule struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Enforcement string                 `json:"enforcement"` // hard, soft
}

// ResourceManager handles resource tracking across providers
type ResourceManager struct {
	registry   *ProviderRegistry
	vmTracking map[string]string // vm_id -> provider_id
}

// NewResourceManager creates a new resource manager
func NewResourceManager(registry *ProviderRegistry) *ResourceManager {
	return &ResourceManager{
		registry:   registry,
		vmTracking: make(map[string]string),
	}
}

// TrackVMCreation tracks a newly created VM
func (r *ResourceManager) TrackVMCreation(providerID string, vm *VMInstance) error {
	r.vmTracking[vm.ID] = providerID
	return nil
}

// FindVMProvider finds which provider a VM belongs to
func (r *ResourceManager) FindVMProvider(vmID string) (string, error) {
	if providerID, exists := r.vmTracking[vmID]; exists {
		return providerID, nil
	}
	
	// If not in tracking, search all providers
	providers := r.registry.ListProviders()
	for providerID, provider := range providers {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		
		_, err := provider.GetVM(ctx, vmID)
		if err == nil {
			r.vmTracking[vmID] = providerID
			return providerID, nil
		}
	}
	
	return "", fmt.Errorf("VM %s not found in any provider", vmID)
}

// GetUtilization gets resource utilization across all providers
func (r *ResourceManager) GetUtilization(ctx context.Context) (*MultiCloudResourceUtilization, error) {
	utilization := &MultiCloudResourceUtilization{
		ByProvider: make(map[string]*ResourceUsage),
		ByRegion:   make(map[string]*ResourceUsage),
		LastUpdated: time.Now(),
	}

	providers := r.registry.ListProviders()
	
	for providerID, provider := range providers {
		usage, err := provider.GetResourceUsage(ctx)
		if err != nil {
			continue
		}

		utilization.ByProvider[providerID] = usage
		utilization.TotalVMs += usage.UsedVMs
		utilization.TotalCPU += usage.UsedCPU
		utilization.TotalMemory += usage.UsedMemory
		utilization.TotalStorage += usage.UsedStorage
		utilization.TotalCost += usage.TotalCost

		// Aggregate by region (simplified)
		regions := provider.GetRegions()
		if len(regions) > 0 {
			region := regions[0] // Use first region as default
			if existing, exists := utilization.ByRegion[region]; exists {
				existing.UsedVMs += usage.UsedVMs
				existing.UsedCPU += usage.UsedCPU
				existing.UsedMemory += usage.UsedMemory
				existing.UsedStorage += usage.UsedStorage
				existing.TotalCost += usage.TotalCost
			} else {
				utilization.ByRegion[region] = usage
			}
		}
	}

	// Generate recommendations
	utilization.Recommendations = r.generateUtilizationRecommendations(utilization)

	return utilization, nil
}

func (r *ResourceManager) generateUtilizationRecommendations(utilization *MultiCloudResourceUtilization) []ResourceOptimizationRecommendation {
	var recommendations []ResourceOptimizationRecommendation

	// Check for underutilized providers
	for providerID, usage := range utilization.ByProvider {
		if usage.UsedVMs < 5 && usage.TotalCost > 100 {
			recommendations = append(recommendations, ResourceOptimizationRecommendation{
				Type:        "consolidation",
				Resource:    providerID,
				Provider:    providerID,
				Description: fmt.Sprintf("Low VM count (%d) with significant cost. Consider consolidating workloads.", usage.UsedVMs),
				Potential:   "cost savings",
				Confidence:  0.8,
				Impact:      "medium",
			})
		}
	}

	return recommendations
}