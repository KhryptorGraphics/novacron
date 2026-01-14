package multicloud

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// ComplianceEngine manages compliance across multiple cloud providers
type ComplianceEngine struct {
	registry *ProviderRegistry
	policies map[string]*CompliancePolicy
}

// NewComplianceEngine creates a new compliance engine
func NewComplianceEngine(registry *ProviderRegistry) *ComplianceEngine {
	return &ComplianceEngine{
		registry: registry,
		policies: make(map[string]*CompliancePolicy),
	}
}

// SetCompliancePolicy sets a compliance policy
func (c *ComplianceEngine) SetCompliancePolicy(policy *CompliancePolicy) error {
	if policy.ID == "" {
		return fmt.Errorf("policy ID cannot be empty")
	}

	c.policies[policy.ID] = policy
	return nil
}

// ValidateVMRequest validates a VM creation request against compliance policies
func (c *ComplianceEngine) ValidateVMRequest(ctx context.Context, providerID string, request *UnifiedVMRequest) error {
	provider, err := c.registry.GetProvider(providerID)
	if err != nil {
		return err
	}

	// Check data residency requirements
	if len(request.DataResidencyRegions) > 0 {
		allowed := false
		for _, allowedRegion := range request.DataResidencyRegions {
			if request.Region == allowedRegion {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("region %s is not allowed for data residency", request.Region)
		}
	}

	// Check compliance requirements
	for _, requirement := range request.ComplianceRequirements {
		if err := c.validateComplianceRequirement(ctx, providerID, provider, requirement); err != nil {
			return fmt.Errorf("compliance requirement %s not met: %v", requirement, err)
		}
	}

	// Apply compliance policies
	for _, policy := range c.policies {
		if policy.Enabled && c.policyApplies(policy, providerID, request) {
			if err := c.validatePolicy(ctx, policy, providerID, provider, request); err != nil {
				return fmt.Errorf("compliance policy %s violated: %v", policy.Name, err)
			}
		}
	}

	return nil
}

// ValidateMigration validates a cross-cloud migration against compliance policies
func (c *ComplianceEngine) ValidateMigration(ctx context.Context, request *CrossCloudMigrationRequest) error {
	// Get source and destination providers
	sourceProvider, err := c.registry.GetProvider(request.SourceProviderID)
	if err != nil {
		return fmt.Errorf("source provider not found: %v", err)
	}

	destProvider, err := c.registry.GetProvider(request.DestinationProviderID)
	if err != nil {
		return fmt.Errorf("destination provider not found: %v", err)
	}

	// Check compliance status of both providers
	sourceCompliance, err := sourceProvider.GetComplianceStatus(ctx)
	if err != nil {
		return fmt.Errorf("failed to get source provider compliance: %v", err)
	}

	destCompliance, err := destProvider.GetComplianceStatus(ctx)
	if err != nil {
		return fmt.Errorf("failed to get destination provider compliance: %v", err)
	}

	// Validate migration doesn't violate compliance
	if err := c.validateMigrationCompliance(sourceCompliance, destCompliance); err != nil {
		return err
	}

	// Check data residency restrictions
	if err := c.validateDataResidencyForMigration(ctx, request, sourceProvider, destProvider); err != nil {
		return err
	}

	// Apply migration-specific compliance policies
	for _, policy := range c.policies {
		if policy.Enabled && policy.Type == "migration" {
			if err := c.validateMigrationPolicy(ctx, policy, request); err != nil {
				return fmt.Errorf("migration compliance policy %s violated: %v", policy.Name, err)
			}
		}
	}

	return nil
}

// GenerateReport generates a compliance report across all providers
func (c *ComplianceEngine) GenerateReport(ctx context.Context, frameworks []string) (*MultiCloudComplianceReport, error) {
	report := &MultiCloudComplianceReport{
		ReportID:    fmt.Sprintf("compliance-report-%d", time.Now().Unix()),
		GeneratedAt: time.Now(),
		Frameworks:  frameworks,
		ByProvider:  make(map[string]*ProviderComplianceReport),
	}

	providers := c.registry.ListProviders()

	// Get compliance status from each provider
	for providerID, provider := range providers {
		complianceStatus, err := provider.GetComplianceStatus(ctx)
		if err != nil {
			fmt.Printf("Failed to get compliance status from provider %s: %v\n", providerID, err)
			continue
		}

		providerReport := &ProviderComplianceReport{
			ProviderID:      providerID,
			ProviderType:    string(provider.GetProviderType()),
			OverallScore:    complianceStatus.OverallScore,
			ComplianceStatus: complianceStatus,
			Violations:      c.analyzeViolations(complianceStatus.PolicyViolations),
		}

		// Filter frameworks if specified
		if len(frameworks) > 0 {
			filteredCompliances := []ComplianceFramework{}
			for _, compliance := range complianceStatus.Compliances {
				for _, framework := range frameworks {
					if strings.EqualFold(compliance.Name, framework) {
						filteredCompliances = append(filteredCompliances, compliance)
						break
					}
				}
			}
			complianceStatus.Compliances = filteredCompliances
		}

		report.ByProvider[providerID] = providerReport
		report.OverallScore += complianceStatus.OverallScore
	}

	// Calculate average overall score
	if len(report.ByProvider) > 0 {
		report.OverallScore /= float64(len(report.ByProvider))
	}

	// Generate recommendations
	report.Recommendations = c.generateComplianceRecommendations(report)

	// Generate summary
	report.Summary = c.generateComplianceSummary(report)

	return report, nil
}

// GetComplianceDashboard returns compliance dashboard data
func (c *ComplianceEngine) GetComplianceDashboard(ctx context.Context) (*ComplianceDashboard, error) {
	dashboard := &ComplianceDashboard{
		LastUpdated: time.Now(),
		ByProvider:  make(map[string]*ProviderComplianceMetrics),
	}

	providers := c.registry.ListProviders()

	var totalScore float64
	var totalViolations int

	for providerID, provider := range providers {
		complianceStatus, err := provider.GetComplianceStatus(ctx)
		if err != nil {
			continue
		}

		metrics := &ProviderComplianceMetrics{
			ProviderID:     providerID,
			OverallScore:   complianceStatus.OverallScore,
			ViolationCount: len(complianceStatus.PolicyViolations),
			FrameworkScores: make(map[string]float64),
		}

		for _, framework := range complianceStatus.Compliances {
			metrics.FrameworkScores[framework.Name] = framework.Score
		}

		dashboard.ByProvider[providerID] = metrics
		totalScore += complianceStatus.OverallScore
		totalViolations += len(complianceStatus.PolicyViolations)
	}

	if len(providers) > 0 {
		dashboard.OverallScore = totalScore / float64(len(providers))
	}
	dashboard.TotalViolations = totalViolations

	return dashboard, nil
}

// Helper methods

func (c *ComplianceEngine) validateComplianceRequirement(ctx context.Context, providerID string, provider CloudProvider, requirement string) error {
	complianceStatus, err := provider.GetComplianceStatus(ctx)
	if err != nil {
		return err
	}

	// Check if provider meets the compliance requirement
	for _, framework := range complianceStatus.Compliances {
		if strings.EqualFold(framework.Name, requirement) {
			if framework.Status != "compliant" {
				return fmt.Errorf("provider does not meet %s compliance", requirement)
			}
			return nil
		}
	}

	// Check certifications
	for _, cert := range complianceStatus.Certifications {
		if strings.EqualFold(cert, requirement) {
			return nil
		}
	}

	return fmt.Errorf("compliance requirement %s not found", requirement)
}

func (c *ComplianceEngine) policyApplies(policy *CompliancePolicy, providerID string, request *UnifiedVMRequest) bool {
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

	return true
}

func (c *ComplianceEngine) validatePolicy(ctx context.Context, policy *CompliancePolicy, providerID string, provider CloudProvider, request *UnifiedVMRequest) error {
	for _, rule := range policy.Rules {
		if err := c.validatePolicyRule(ctx, rule, providerID, provider, request); err != nil {
			return err
		}
	}
	return nil
}

func (c *ComplianceEngine) validatePolicyRule(ctx context.Context, rule CompliancePolicyRule, providerID string, provider CloudProvider, request *UnifiedVMRequest) error {
	switch rule.Type {
	case "data_residency":
		return c.validateDataResidencyRule(rule, request)
	case "encryption":
		return c.validateEncryptionRule(ctx, rule, provider, request)
	case "access_control":
		return c.validateAccessControlRule(ctx, rule, provider, request)
	case "network_security":
		return c.validateNetworkSecurityRule(ctx, rule, provider, request)
	default:
		return fmt.Errorf("unknown rule type: %s", rule.Type)
	}
}

func (c *ComplianceEngine) validateDataResidencyRule(rule CompliancePolicyRule, request *UnifiedVMRequest) error {
	if allowedRegions, ok := rule.Parameters["allowed_regions"].([]string); ok {
		for _, region := range allowedRegions {
			if region == request.Region {
				return nil
			}
		}
		return fmt.Errorf("region %s not allowed by data residency rule", request.Region)
	}
	return nil
}

func (c *ComplianceEngine) validateEncryptionRule(ctx context.Context, rule CompliancePolicyRule, provider CloudProvider, request *UnifiedVMRequest) error {
	// This would check if encryption is properly configured
	// For now, just check if encryption is required and enabled
	if required, ok := rule.Parameters["required"].(bool); ok && required {
		// Check if provider supports encryption
		capabilities := provider.GetCapabilities()
		for _, cap := range capabilities {
			if cap == CapabilityKeyManagement {
				return nil // Provider supports encryption
			}
		}
		return fmt.Errorf("encryption required but provider does not support key management")
	}
	return nil
}

func (c *ComplianceEngine) validateAccessControlRule(ctx context.Context, rule CompliancePolicyRule, provider CloudProvider, request *UnifiedVMRequest) error {
	// This would validate access control configurations
	// For now, just check if IAM is supported when required
	if required, ok := rule.Parameters["iam_required"].(bool); ok && required {
		capabilities := provider.GetCapabilities()
		for _, cap := range capabilities {
			if cap == CapabilityIdentityManagement {
				return nil
			}
		}
		return fmt.Errorf("IAM required but provider does not support identity management")
	}
	return nil
}

func (c *ComplianceEngine) validateNetworkSecurityRule(ctx context.Context, rule CompliancePolicyRule, provider CloudProvider, request *UnifiedVMRequest) error {
	// This would validate network security configurations
	// For now, check if network ACLs are required
	if required, ok := rule.Parameters["network_acls_required"].(bool); ok && required {
		capabilities := provider.GetCapabilities()
		for _, cap := range capabilities {
			if cap == CapabilityNetworkACLs {
				return nil
			}
		}
		return fmt.Errorf("network ACLs required but provider does not support them")
	}
	return nil
}

func (c *ComplianceEngine) validateMigrationCompliance(source, dest *ComplianceStatus) error {
	// Check if destination meets or exceeds source compliance
	if dest.OverallScore < source.OverallScore-10 { // Allow 10 point tolerance
		return fmt.Errorf("destination provider has significantly lower compliance score")
	}

	// Check if critical frameworks are maintained
	sourceFrameworks := make(map[string]ComplianceFramework)
	for _, framework := range source.Compliances {
		sourceFrameworks[framework.Name] = framework
	}

	for _, destFramework := range dest.Compliances {
		if sourceFramework, exists := sourceFrameworks[destFramework.Name]; exists {
			if destFramework.Status == "non-compliant" && sourceFramework.Status == "compliant" {
				return fmt.Errorf("destination provider loses compliance for framework: %s", destFramework.Name)
			}
		}
	}

	return nil
}

func (c *ComplianceEngine) validateDataResidencyForMigration(ctx context.Context, request *CrossCloudMigrationRequest, source, dest CloudProvider) error {
	// Check if migration violates data residency
	sourceRegions := source.GetRegions()
	destRegions := dest.GetRegions()

	// Simplified check: ensure destination region is in allowed regions
	if request.DestinationRegion != "" {
		found := false
		for _, region := range destRegions {
			if region == request.DestinationRegion {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("destination region %s not supported by provider", request.DestinationRegion)
		}
	}

	// Check for cross-border data transfer restrictions
	// This would need more sophisticated region-to-country mapping
	if c.isCrossBorderTransfer(sourceRegions, destRegions) {
		// Check if cross-border transfer is allowed by policies
		for _, policy := range c.policies {
			if policy.Type == "data_residency" && !c.allowsCrossBorderTransfer(policy) {
				return fmt.Errorf("cross-border data transfer not allowed by compliance policy")
			}
		}
	}

	return nil
}

func (c *ComplianceEngine) validateMigrationPolicy(ctx context.Context, policy *CompliancePolicy, request *CrossCloudMigrationRequest) error {
	// Validate migration-specific policies
	for _, rule := range policy.Rules {
		if rule.Type == "migration_approval" {
			if required, ok := rule.Parameters["approval_required"].(bool); ok && required {
				// Check if migration has been approved
				if approval, ok := rule.Parameters["approval_status"].(string); !ok || approval != "approved" {
					return fmt.Errorf("migration requires approval but is not approved")
				}
			}
		}
	}
	return nil
}

func (c *ComplianceEngine) analyzeViolations(violations []PolicyViolation) []ComplianceViolationAnalysis {
	var analyses []ComplianceViolationAnalysis

	for _, violation := range violations {
		analysis := ComplianceViolationAnalysis{
			ViolationID: violation.ID,
			Severity:    violation.Severity,
			Impact:      c.assessViolationImpact(violation),
			Remediation: c.suggestRemediation(violation),
		}
		analyses = append(analyses, analysis)
	}

	return analyses
}

func (c *ComplianceEngine) assessViolationImpact(violation PolicyViolation) string {
	switch violation.Severity {
	case "critical":
		return "high"
	case "high":
		return "medium"
	default:
		return "low"
	}
}

func (c *ComplianceEngine) suggestRemediation(violation PolicyViolation) string {
	// This would provide specific remediation steps based on violation type
	return fmt.Sprintf("Address policy violation: %s", violation.Description)
}

func (c *ComplianceEngine) generateComplianceRecommendations(report *MultiCloudComplianceReport) []ComplianceRecommendation {
	var recommendations []ComplianceRecommendation

	// Generate recommendations based on violations and scores
	for providerID, providerReport := range report.ByProvider {
		if providerReport.OverallScore < 80 {
			recommendations = append(recommendations, ComplianceRecommendation{
				Type:        "improve_compliance",
				Provider:    providerID,
				Priority:    "high",
				Description: fmt.Sprintf("Provider %s has low compliance score (%.1f). Consider reviewing and updating security controls.", providerID, providerReport.OverallScore),
			})
		}

		if len(providerReport.Violations) > 0 {
			recommendations = append(recommendations, ComplianceRecommendation{
				Type:        "resolve_violations",
				Provider:    providerID,
				Priority:    "high",
				Description: fmt.Sprintf("Provider %s has %d active violations. Immediate attention required.", providerID, len(providerReport.Violations)),
			})
		}
	}

	return recommendations
}

func (c *ComplianceEngine) generateComplianceSummary(report *MultiCloudComplianceReport) ComplianceSummary {
	summary := ComplianceSummary{
		TotalProviders:   len(report.ByProvider),
		OverallScore:     report.OverallScore,
	}

	for _, providerReport := range report.ByProvider {
		if providerReport.OverallScore >= 90 {
			summary.CompliantProviders++
		} else if providerReport.OverallScore >= 70 {
			summary.PartiallyCompliantProviders++
		} else {
			summary.NonCompliantProviders++
		}

		summary.TotalViolations += len(providerReport.Violations)
	}

	return summary
}

func (c *ComplianceEngine) isCrossBorderTransfer(sourceRegions, destRegions []string) bool {
	// Simplified implementation - would need proper region-to-country mapping
	// For now, assume different region prefixes indicate different countries
	if len(sourceRegions) > 0 && len(destRegions) > 0 {
		sourcePrefix := strings.Split(sourceRegions[0], "-")[0]
		destPrefix := strings.Split(destRegions[0], "-")[0]
		return sourcePrefix != destPrefix
	}
	return false
}

func (c *ComplianceEngine) allowsCrossBorderTransfer(policy *CompliancePolicy) bool {
	for _, rule := range policy.Rules {
		if rule.Type == "data_residency" {
			if allowed, ok := rule.Parameters["cross_border_transfer"].(bool); ok {
				return allowed
			}
		}
	}
	return false // Default to not allowing cross-border transfer
}

// Compliance types

// CompliancePolicy represents a compliance policy
type CompliancePolicy struct {
	ID                  string                  `json:"id"`
	Name                string                  `json:"name"`
	Description         string                  `json:"description"`
	Type                string                  `json:"type"` // data_residency, encryption, access_control, etc.
	Enabled             bool                    `json:"enabled"`
	Rules               []CompliancePolicyRule  `json:"rules"`
	ApplicableProviders []string                `json:"applicable_providers,omitempty"`
	ApplicableRegions   []string                `json:"applicable_regions,omitempty"`
	CreatedAt           time.Time               `json:"created_at"`
	UpdatedAt           time.Time               `json:"updated_at"`
}

// CompliancePolicyRule represents a rule within a compliance policy
type CompliancePolicyRule struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Severity    string                 `json:"severity"` // low, medium, high, critical
}

// MultiCloudComplianceReport represents a compliance report across providers
type MultiCloudComplianceReport struct {
	ReportID        string                           `json:"report_id"`
	GeneratedAt     time.Time                        `json:"generated_at"`
	Frameworks      []string                         `json:"frameworks"`
	OverallScore    float64                          `json:"overall_score"`
	ByProvider      map[string]*ProviderComplianceReport `json:"by_provider"`
	Summary         ComplianceSummary                `json:"summary"`
	Recommendations []ComplianceRecommendation       `json:"recommendations"`
}

// ProviderComplianceReport represents compliance report for a provider
type ProviderComplianceReport struct {
	ProviderID       string                        `json:"provider_id"`
	ProviderType     string                        `json:"provider_type"`
	OverallScore     float64                       `json:"overall_score"`
	ComplianceStatus *ComplianceStatus             `json:"compliance_status"`
	Violations       []ComplianceViolationAnalysis `json:"violations"`
}

// ComplianceViolationAnalysis represents analysis of a compliance violation
type ComplianceViolationAnalysis struct {
	ViolationID string `json:"violation_id"`
	Severity    string `json:"severity"`
	Impact      string `json:"impact"`
	Remediation string `json:"remediation"`
}

// ComplianceSummary represents a summary of compliance across providers
type ComplianceSummary struct {
	TotalProviders                int `json:"total_providers"`
	CompliantProviders            int `json:"compliant_providers"`
	PartiallyCompliantProviders   int `json:"partially_compliant_providers"`
	NonCompliantProviders         int `json:"non_compliant_providers"`
	TotalViolations               int `json:"total_violations"`
	OverallScore                  float64 `json:"overall_score"`
}

// ComplianceRecommendation represents a compliance recommendation
type ComplianceRecommendation struct {
	Type        string `json:"type"`
	Provider    string `json:"provider,omitempty"`
	Priority    string `json:"priority"`
	Description string `json:"description"`
}

// ComplianceDashboard represents compliance dashboard data
type ComplianceDashboard struct {
	OverallScore    float64                            `json:"overall_score"`
	TotalViolations int                                `json:"total_violations"`
	ByProvider      map[string]*ProviderComplianceMetrics `json:"by_provider"`
	LastUpdated     time.Time                          `json:"last_updated"`
}

// ProviderComplianceMetrics represents compliance metrics for a provider
type ProviderComplianceMetrics struct {
	ProviderID      string             `json:"provider_id"`
	OverallScore    float64            `json:"overall_score"`
	ViolationCount  int                `json:"violation_count"`
	FrameworkScores map[string]float64 `json:"framework_scores"`
}