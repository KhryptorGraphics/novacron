package edge

import (
	"context"
	"fmt"
	"time"
)

// EdgePolicyManager manages edge computing policies
type EdgePolicyManager struct {
	config           *EdgeConfig
	dataResidency    map[string]*DataResidencyPolicy
	latencySLAs      map[string]*LatencySLA
	costPolicies     map[string]*CostPolicy
	securityPolicies map[string]*SecurityPolicy
	complianceRules  map[string]*ComplianceRule
}

// DataResidencyPolicy defines data residency requirements
type DataResidencyPolicy struct {
	PolicyID        string    `json:"policy_id"`
	Name            string    `json:"name"`
	AllowedRegions  []string  `json:"allowed_regions"`
	ExcludedRegions []string  `json:"excluded_regions"`
	AllowedCountries []string `json:"allowed_countries"`
	DataClassification string `json:"data_classification"` // "public", "internal", "confidential", "restricted"
	EncryptionRequired bool  `json:"encryption_required"`
	Regulation      string    `json:"regulation"` // "GDPR", "CCPA", "HIPAA", etc.
	CreatedAt       time.Time `json:"created_at"`
}

// LatencySLA defines latency Service Level Agreement
type LatencySLA struct {
	SLAID           string        `json:"sla_id"`
	Name            string        `json:"name"`
	TargetLatencyMs int           `json:"target_latency_ms"`
	MaxLatencyMs    int           `json:"max_latency_ms"`
	Percentile      float64       `json:"percentile"` // 99.0 = p99
	Penalty         float64       `json:"penalty"`    // Cost penalty for violation
	MeasurementWindow time.Duration `json:"measurement_window"`
	ViolationThreshold int         `json:"violation_threshold"` // Consecutive violations before action
	CreatedAt       time.Time     `json:"created_at"`
}

// CostPolicy defines cost optimization policies
type CostPolicy struct {
	PolicyID        string    `json:"policy_id"`
	Name            string    `json:"name"`
	MaxCostPerHour  float64   `json:"max_cost_per_hour"`
	MaxCostPerVM    float64   `json:"max_cost_per_vm"`
	PreferLowCost   bool      `json:"prefer_low_cost"`
	OptimizationGoal string   `json:"optimization_goal"` // "cost", "performance", "balanced"
	AutoScaleDown   bool      `json:"auto_scale_down"`
	IdleTimeout     time.Duration `json:"idle_timeout"`
	CreatedAt       time.Time `json:"created_at"`
}

// SecurityPolicy defines security policies for edge
type SecurityPolicy struct {
	PolicyID             string    `json:"policy_id"`
	Name                 string    `json:"name"`
	RequireTLS           bool      `json:"require_tls"`
	MinTLSVersion        string    `json:"min_tls_version"` // "1.2", "1.3"
	RequireMutualTLS     bool      `json:"require_mutual_tls"`
	AllowedCipherSuites  []string  `json:"allowed_cipher_suites"`
	NetworkIsolation     bool      `json:"network_isolation"`
	FirewallRules        []FirewallRule `json:"firewall_rules"`
	IntrusionDetection   bool      `json:"intrusion_detection"`
	AuditLogging         bool      `json:"audit_logging"`
	VulnerabilityScanning bool     `json:"vulnerability_scanning"`
	CreatedAt            time.Time `json:"created_at"`
}

// FirewallRule defines a firewall rule
type FirewallRule struct {
	RuleID      string   `json:"rule_id"`
	Action      string   `json:"action"` // "allow", "deny"
	Protocol    string   `json:"protocol"`
	SourceCIDR  string   `json:"source_cidr"`
	DestCIDR    string   `json:"dest_cidr"`
	Ports       []int    `json:"ports"`
	Priority    int      `json:"priority"`
}

// ComplianceRule defines compliance validation rules
type ComplianceRule struct {
	RuleID          string    `json:"rule_id"`
	Name            string    `json:"name"`
	Framework       string    `json:"framework"` // "GDPR", "HIPAA", "SOC2", "ISO27001"
	Description     string    `json:"description"`
	ValidationFunc  string    `json:"validation_func"` // Name of validation function
	Severity        string    `json:"severity"` // "critical", "high", "medium", "low"
	AutoRemediate   bool      `json:"auto_remediate"`
	CreatedAt       time.Time `json:"created_at"`
}

// NewEdgePolicyManager creates a new policy manager
func NewEdgePolicyManager(config *EdgeConfig) *EdgePolicyManager {
	return &EdgePolicyManager{
		config:           config,
		dataResidency:    make(map[string]*DataResidencyPolicy),
		latencySLAs:      make(map[string]*LatencySLA),
		costPolicies:     make(map[string]*CostPolicy),
		securityPolicies: make(map[string]*SecurityPolicy),
		complianceRules:  make(map[string]*ComplianceRule),
	}
}

// ValidateDataResidency validates data residency compliance
func (epm *EdgePolicyManager) ValidateDataResidency(ctx context.Context, nodeID string, policyID string) error {
	policy, exists := epm.dataResidency[policyID]
	if !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	// Get node location (this would come from discovery service)
	// For now, simulate
	nodeCountry := "US"
	nodeRegion := "us-west-2"

	// Check if country is allowed
	if len(policy.AllowedCountries) > 0 {
		allowed := false
		for _, country := range policy.AllowedCountries {
			if country == nodeCountry {
				allowed = true
				break
			}
		}
		if !allowed {
			return ErrDataResidencyViolation
		}
	}

	// Check if region is allowed
	if len(policy.AllowedRegions) > 0 {
		allowed := false
		for _, region := range policy.AllowedRegions {
			if region == nodeRegion {
				allowed = true
				break
			}
		}
		if !allowed {
			return ErrDataResidencyViolation
		}
	}

	// Check excluded regions
	for _, region := range policy.ExcludedRegions {
		if region == nodeRegion {
			return ErrDataResidencyViolation
		}
	}

	// Special handling for GDPR
	if policy.Regulation == "GDPR" {
		if !epm.isEURegion(nodeCountry) {
			return ErrDataResidencyViolation
		}
	}

	return nil
}

// isEURegion checks if a country is in the EU
func (epm *EdgePolicyManager) isEURegion(country string) bool {
	euCountries := []string{"DE", "FR", "IT", "ES", "NL", "BE", "AT", "IE", "FI", "SE", "PL", "PT", "GR", "CZ", "RO", "HU", "DK", "BG", "SK", "HR", "SI", "LT", "LV", "EE", "CY", "LU", "MT"}
	for _, eu := range euCountries {
		if country == eu {
			return true
		}
	}
	return false
}

// ValidateLatencySLA validates latency SLA compliance
func (epm *EdgePolicyManager) ValidateLatencySLA(ctx context.Context, nodeID string, slaID string, actualLatencyMs float64) error {
	sla, exists := epm.latencySLAs[slaID]
	if !exists {
		return fmt.Errorf("SLA not found: %s", slaID)
	}

	if actualLatencyMs > float64(sla.MaxLatencyMs) {
		return ErrLatencySLAViolation
	}

	return nil
}

// ValidateCostPolicy validates cost policy compliance
func (epm *EdgePolicyManager) ValidateCostPolicy(ctx context.Context, nodeID string, policyID string, costPerHour float64) error {
	policy, exists := epm.costPolicies[policyID]
	if !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	if costPerHour > policy.MaxCostPerHour {
		return fmt.Errorf("cost exceeds policy limit: %.2f > %.2f", costPerHour, policy.MaxCostPerHour)
	}

	return nil
}

// ValidateSecurityPolicy validates security policy compliance
func (epm *EdgePolicyManager) ValidateSecurityPolicy(ctx context.Context, nodeID string, policyID string) error {
	policy, exists := epm.securityPolicies[policyID]
	if !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	// In production, this would:
	// 1. Check TLS configuration
	// 2. Verify network isolation
	// 3. Validate firewall rules are applied
	// 4. Check IDS/IPS status
	// 5. Verify audit logging is enabled

	if policy.RequireTLS && !epm.config.RequireTLS {
		return fmt.Errorf("TLS required by policy but not configured")
	}

	return nil
}

// ValidateCompliance validates all compliance rules
func (epm *EdgePolicyManager) ValidateCompliance(ctx context.Context, nodeID string) ([]ComplianceViolation, error) {
	violations := make([]ComplianceViolation, 0)

	for _, rule := range epm.complianceRules {
		if err := epm.validateComplianceRule(ctx, nodeID, rule); err != nil {
			violations = append(violations, ComplianceViolation{
				RuleID:      rule.RuleID,
				RuleName:    rule.Name,
				Framework:   rule.Framework,
				Severity:    rule.Severity,
				Description: err.Error(),
				DetectedAt:  time.Now(),
			})
		}
	}

	return violations, nil
}

// ComplianceViolation represents a compliance violation
type ComplianceViolation struct {
	RuleID      string    `json:"rule_id"`
	RuleName    string    `json:"rule_name"`
	Framework   string    `json:"framework"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	DetectedAt  time.Time `json:"detected_at"`
}

// validateComplianceRule validates a single compliance rule
func (epm *EdgePolicyManager) validateComplianceRule(ctx context.Context, nodeID string, rule *ComplianceRule) error {
	// In production, call appropriate validation function based on rule.ValidationFunc
	// For now, return nil (compliant)
	return nil
}

// AddDataResidencyPolicy adds a data residency policy
func (epm *EdgePolicyManager) AddDataResidencyPolicy(policy *DataResidencyPolicy) {
	if policy.PolicyID == "" {
		policy.PolicyID = fmt.Sprintf("dr-policy-%d", time.Now().UnixNano())
	}
	policy.CreatedAt = time.Now()
	epm.dataResidency[policy.PolicyID] = policy
}

// AddLatencySLA adds a latency SLA
func (epm *EdgePolicyManager) AddLatencySLA(sla *LatencySLA) {
	if sla.SLAID == "" {
		sla.SLAID = fmt.Sprintf("sla-%d", time.Now().UnixNano())
	}
	sla.CreatedAt = time.Now()
	epm.latencySLAs[sla.SLAID] = sla
}

// AddCostPolicy adds a cost policy
func (epm *EdgePolicyManager) AddCostPolicy(policy *CostPolicy) {
	if policy.PolicyID == "" {
		policy.PolicyID = fmt.Sprintf("cost-policy-%d", time.Now().UnixNano())
	}
	policy.CreatedAt = time.Now()
	epm.costPolicies[policy.PolicyID] = policy
}

// AddSecurityPolicy adds a security policy
func (epm *EdgePolicyManager) AddSecurityPolicy(policy *SecurityPolicy) {
	if policy.PolicyID == "" {
		policy.PolicyID = fmt.Sprintf("sec-policy-%d", time.Now().UnixNano())
	}
	policy.CreatedAt = time.Now()
	epm.securityPolicies[policy.PolicyID] = policy
}

// AddComplianceRule adds a compliance rule
func (epm *EdgePolicyManager) AddComplianceRule(rule *ComplianceRule) {
	if rule.RuleID == "" {
		rule.RuleID = fmt.Sprintf("rule-%d", time.Now().UnixNano())
	}
	rule.CreatedAt = time.Now()
	epm.complianceRules[rule.RuleID] = rule
}

// InitializeDefaultPolicies initializes default policies
func (epm *EdgePolicyManager) InitializeDefaultPolicies() {
	// Default GDPR policy
	epm.AddDataResidencyPolicy(&DataResidencyPolicy{
		Name:               "GDPR Compliance",
		AllowedCountries:   []string{"DE", "FR", "IT", "ES", "NL", "BE", "AT", "IE", "FI", "SE"},
		DataClassification: "confidential",
		EncryptionRequired: true,
		Regulation:         "GDPR",
	})

	// Default latency SLA
	epm.AddLatencySLA(&LatencySLA{
		Name:               "Standard Edge SLA",
		TargetLatencyMs:    50,
		MaxLatencyMs:       100,
		Percentile:         99.0,
		MeasurementWindow:  5 * time.Minute,
		ViolationThreshold: 3,
	})

	// Default cost policy
	epm.AddCostPolicy(&CostPolicy{
		Name:             "Standard Cost Optimization",
		MaxCostPerHour:   1.0,
		MaxCostPerVM:     0.5,
		OptimizationGoal: "balanced",
		AutoScaleDown:    true,
		IdleTimeout:      30 * time.Minute,
	})

	// Default security policy
	epm.AddSecurityPolicy(&SecurityPolicy{
		Name:                 "Standard Edge Security",
		RequireTLS:           true,
		MinTLSVersion:        "1.3",
		NetworkIsolation:     true,
		IntrusionDetection:   true,
		AuditLogging:         true,
		VulnerabilityScanning: true,
	})
}

// GetPolicyReport generates a policy compliance report
func (epm *EdgePolicyManager) GetPolicyReport(ctx context.Context) (*PolicyReport, error) {
	return &PolicyReport{
		DataResidencyPolicies: len(epm.dataResidency),
		LatencySLAs:           len(epm.latencySLAs),
		CostPolicies:          len(epm.costPolicies),
		SecurityPolicies:      len(epm.securityPolicies),
		ComplianceRules:       len(epm.complianceRules),
		GeneratedAt:           time.Now(),
	}, nil
}

// PolicyReport represents a policy compliance report
type PolicyReport struct {
	DataResidencyPolicies int       `json:"data_residency_policies"`
	LatencySLAs           int       `json:"latency_slas"`
	CostPolicies          int       `json:"cost_policies"`
	SecurityPolicies      int       `json:"security_policies"`
	ComplianceRules       int       `json:"compliance_rules"`
	GeneratedAt           time.Time `json:"generated_at"`
}
