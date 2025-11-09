package governance

import (
	"time"
)

// GovernanceConfig defines comprehensive governance configuration
type GovernanceConfig struct {
	Compliance      ComplianceConfig      `json:"compliance" yaml:"compliance"`
	AuditLogging    AuditConfig          `json:"audit_logging" yaml:"audit_logging"`
	PolicyEngine    PolicyConfig         `json:"policy_engine" yaml:"policy_engine"`
	AccessControl   AccessConfig         `json:"access_control" yaml:"access_control"`
	MultiTenancy    TenancyConfig        `json:"multi_tenancy" yaml:"multi_tenancy"`
	QuotaManagement QuotaConfig          `json:"quota_management" yaml:"quota_management"`
	Chargeback      BillingConfig        `json:"chargeback" yaml:"chargeback"`
	SLAManagement   SLAConfig            `json:"sla_management" yaml:"sla_management"`
	Workflow        WorkflowConfig       `json:"workflow" yaml:"workflow"`
	Reporting       ReportingConfig      `json:"reporting" yaml:"reporting"`
	Classification  ClassificationConfig `json:"classification" yaml:"classification"`
	Secrets         SecretsConfig        `json:"secrets" yaml:"secrets"`
	Metrics         MetricsConfig        `json:"metrics" yaml:"metrics"`
}

// ComplianceConfig defines compliance automation settings
type ComplianceConfig struct {
	EnabledStandards   []string `json:"enabled_standards" yaml:"enabled_standards"` // soc2, iso27001, hipaa, pci-dss, fedramp, gdpr
	AuditFrequency     string   `json:"audit_frequency" yaml:"audit_frequency"`     // continuous, daily, weekly
	ReportingEnabled   bool     `json:"reporting_enabled" yaml:"reporting_enabled"`
	AutoRemediation    bool     `json:"auto_remediation" yaml:"auto_remediation"`
	ComplianceTarget   float64  `json:"compliance_target" yaml:"compliance_target"` // 0.95 = 95%
	EvidenceCollection bool     `json:"evidence_collection" yaml:"evidence_collection"`
	ControlFramework   string   `json:"control_framework" yaml:"control_framework"` // nist, cis, cobit
}

// AuditConfig defines audit logging configuration
type AuditConfig struct {
	Enabled           bool          `json:"enabled" yaml:"enabled"`
	ImmutableStorage  bool          `json:"immutable_storage" yaml:"immutable_storage"`
	RetentionPeriod   time.Duration `json:"retention_period" yaml:"retention_period"` // 7 years for compliance
	LogAllActions     bool          `json:"log_all_actions" yaml:"log_all_actions"`
	LogAPIcalls       bool          `json:"log_api_calls" yaml:"log_api_calls"`
	LogUserActivity   bool          `json:"log_user_activity" yaml:"log_user_activity"`
	LogConfigChanges  bool          `json:"log_config_changes" yaml:"log_config_changes"`
	LogAccessAttempts bool          `json:"log_access_attempts" yaml:"log_access_attempts"`
	ForensicsEnabled  bool          `json:"forensics_enabled" yaml:"forensics_enabled"`
	SearchableIndex   bool          `json:"searchable_index" yaml:"searchable_index"`
	TamperProtection  bool          `json:"tamper_protection" yaml:"tamper_protection"`
	BackupEnabled     bool          `json:"backup_enabled" yaml:"backup_enabled"`
	BackupDestination string        `json:"backup_destination" yaml:"backup_destination"`
}

// PolicyConfig defines OPA policy engine configuration
type PolicyConfig struct {
	Enabled              bool          `json:"enabled" yaml:"enabled"`
	OPAEndpoint          string        `json:"opa_endpoint" yaml:"opa_endpoint"`
	PolicyBundleURL      string        `json:"policy_bundle_url" yaml:"policy_bundle_url"`
	EvaluationTimeout    time.Duration `json:"evaluation_timeout" yaml:"evaluation_timeout"`
	CacheEnabled         bool          `json:"cache_enabled" yaml:"cache_enabled"`
	CacheTTL             time.Duration `json:"cache_ttl" yaml:"cache_ttl"`
	PolicyCategories     []string      `json:"policy_categories" yaml:"policy_categories"`
	VersioningEnabled    bool          `json:"versioning_enabled" yaml:"versioning_enabled"`
	RollbackEnabled      bool          `json:"rollback_enabled" yaml:"rollback_enabled"`
	AuditDecisions       bool          `json:"audit_decisions" yaml:"audit_decisions"`
	PerformanceTarget    time.Duration `json:"performance_target" yaml:"performance_target"` // <10ms
}

// AccessConfig defines RBAC/ABAC configuration
type AccessConfig struct {
	RBACEnabled            bool     `json:"rbac_enabled" yaml:"rbac_enabled"`
	ABACEnabled            bool     `json:"abac_enabled" yaml:"abac_enabled"`
	LeastPrivilege         bool     `json:"least_privilege" yaml:"least_privilege"`
	SeparationOfDuties     bool     `json:"separation_of_duties" yaml:"separation_of_duties"`
	RoleHierarchyEnabled   bool     `json:"role_hierarchy_enabled" yaml:"role_hierarchy_enabled"`
	DynamicAccessDecisions bool     `json:"dynamic_access_decisions" yaml:"dynamic_access_decisions"`
	ContextualAttributes   []string `json:"contextual_attributes" yaml:"contextual_attributes"`
	AuditAccessDecisions   bool     `json:"audit_access_decisions" yaml:"audit_access_decisions"`
}

// TenancyConfig defines multi-tenancy configuration
type TenancyConfig struct {
	Enabled               bool    `json:"enabled" yaml:"enabled"`
	HardIsolation         bool    `json:"hard_isolation" yaml:"hard_isolation"`
	TenantQuotas          bool    `json:"tenant_quotas" yaml:"tenant_quotas"`
	TenantPolicies        bool    `json:"tenant_policies" yaml:"tenant_policies"`
	CrossTenantValidation bool    `json:"cross_tenant_validation" yaml:"cross_tenant_validation"`
	BillingPerTenant      bool    `json:"billing_per_tenant" yaml:"billing_per_tenant"`
	OverheadTarget        float64 `json:"overhead_target" yaml:"overhead_target"` // <3%
	ResourceTagging       bool    `json:"resource_tagging" yaml:"resource_tagging"`
}

// QuotaConfig defines quota management configuration
type QuotaConfig struct {
	Enabled               bool     `json:"enabled" yaml:"enabled"`
	EnforcementMode       string   `json:"enforcement_mode" yaml:"enforcement_mode"` // hard, soft, warning
	QuotaTypes            []string `json:"quota_types" yaml:"quota_types"`           // cpu, memory, storage, network
	AlertThresholds       []int    `json:"alert_thresholds" yaml:"alert_thresholds"` // 80, 90, 100
	RequestWorkflowEnabled bool    `json:"request_workflow_enabled" yaml:"request_workflow_enabled"`
	AutoScalingEnabled    bool     `json:"auto_scaling_enabled" yaml:"auto_scaling_enabled"`
	NotificationsEnabled  bool     `json:"notifications_enabled" yaml:"notifications_enabled"`
}

// BillingConfig defines chargeback/showback configuration
type BillingConfig struct {
	Enabled            bool     `json:"enabled" yaml:"enabled"`
	Mode               string   `json:"mode" yaml:"mode"` // chargeback, showback
	UsageBasedBilling  bool     `json:"usage_based_billing" yaml:"usage_based_billing"`
	CostAllocationTags []string `json:"cost_allocation_tags" yaml:"cost_allocation_tags"`
	ReportingPeriod    string   `json:"reporting_period" yaml:"reporting_period"` // daily, weekly, monthly
	BudgetManagement   bool     `json:"budget_management" yaml:"budget_management"`
	CostAlertsEnabled  bool     `json:"cost_alerts_enabled" yaml:"cost_alerts_enabled"`
	InvoiceGeneration  bool     `json:"invoice_generation" yaml:"invoice_generation"`
}

// SLAConfig defines SLA management configuration
type SLAConfig struct {
	Enabled              bool          `json:"enabled" yaml:"enabled"`
	AvailabilityTarget   float64       `json:"availability_target" yaml:"availability_target"`     // 99.95%
	LatencyTargetP95     time.Duration `json:"latency_target_p95" yaml:"latency_target_p95"`       // <100ms
	ThroughputTarget     int64         `json:"throughput_target" yaml:"throughput_target"`         // req/sec
	ErrorRateTarget      float64       `json:"error_rate_target" yaml:"error_rate_target"`         // <0.1%
	MeasurementWindow    time.Duration `json:"measurement_window" yaml:"measurement_window"`       // 1 month
	ViolationDetection   bool          `json:"violation_detection" yaml:"violation_detection"`
	AutoNotifications    bool          `json:"auto_notifications" yaml:"auto_notifications"`
	ErrorBudgetEnabled   bool          `json:"error_budget_enabled" yaml:"error_budget_enabled"`
	DashboardEnabled     bool          `json:"dashboard_enabled" yaml:"dashboard_enabled"`
}

// WorkflowConfig defines workflow automation configuration
type WorkflowConfig struct {
	Enabled               bool          `json:"enabled" yaml:"enabled"`
	ApprovalWorkflows     bool          `json:"approval_workflows" yaml:"approval_workflows"`
	MultiStageApprovals   bool          `json:"multi_stage_approvals" yaml:"multi_stage_approvals"`
	ConditionalRouting    bool          `json:"conditional_routing" yaml:"conditional_routing"`
	TimeoutHandling       bool          `json:"timeout_handling" yaml:"timeout_handling"`
	DefaultTimeout        time.Duration `json:"default_timeout" yaml:"default_timeout"`
	NotificationChannels  []string      `json:"notification_channels" yaml:"notification_channels"` // email, slack, pagerduty
	WorkflowTemplates     bool          `json:"workflow_templates" yaml:"workflow_templates"`
	AuditTrailEnabled     bool          `json:"audit_trail_enabled" yaml:"audit_trail_enabled"`
}

// ReportingConfig defines compliance reporting configuration
type ReportingConfig struct {
	Enabled              bool     `json:"enabled" yaml:"enabled"`
	AutomatedReports     bool     `json:"automated_reports" yaml:"automated_reports"`
	ReportFormats        []string `json:"report_formats" yaml:"report_formats"` // pdf, html, json
	EvidenceCollection   bool     `json:"evidence_collection" yaml:"evidence_collection"`
	GapAnalysis          bool     `json:"gap_analysis" yaml:"gap_analysis"`
	RemediationTracking  bool     `json:"remediation_tracking" yaml:"remediation_tracking"`
	ExecutiveSummaries   bool     `json:"executive_summaries" yaml:"executive_summaries"`
	ScheduledReports     bool     `json:"scheduled_reports" yaml:"scheduled_reports"`
	ReportingFrequency   string   `json:"reporting_frequency" yaml:"reporting_frequency"` // daily, weekly, monthly
}

// ClassificationConfig defines data classification configuration
type ClassificationConfig struct {
	Enabled              bool     `json:"enabled" yaml:"enabled"`
	AutoClassification   bool     `json:"auto_classification" yaml:"auto_classification"`
	SensitivityLevels    []string `json:"sensitivity_levels" yaml:"sensitivity_levels"` // public, internal, confidential, restricted
	EncryptionByLevel    bool     `json:"encryption_by_level" yaml:"encryption_by_level"`
	AccessPoliciesByLevel bool    `json:"access_policies_by_level" yaml:"access_policies_by_level"`
	DLPEnabled           bool     `json:"dlp_enabled" yaml:"dlp_enabled"` // Data Loss Prevention
	ContentScanning      bool     `json:"content_scanning" yaml:"content_scanning"`
}

// SecretsConfig defines secret management configuration
type SecretsConfig struct {
	Enabled           bool          `json:"enabled" yaml:"enabled"`
	Provider          string        `json:"provider" yaml:"provider"` // vault, aws-secrets-manager, azure-key-vault
	VaultAddress      string        `json:"vault_address" yaml:"vault_address"`
	AutoRotation      bool          `json:"auto_rotation" yaml:"auto_rotation"`
	RotationPeriod    time.Duration `json:"rotation_period" yaml:"rotation_period"`
	AuditSecretAccess bool          `json:"audit_secret_access" yaml:"audit_secret_access"`
	EncryptionAtRest  bool          `json:"encryption_at_rest" yaml:"encryption_at_rest"`
	EncryptionInTransit bool        `json:"encryption_in_transit" yaml:"encryption_in_transit"`
}

// MetricsConfig defines governance metrics configuration
type MetricsConfig struct {
	Enabled                bool `json:"enabled" yaml:"enabled"`
	ComplianceScore        bool `json:"compliance_score" yaml:"compliance_score"`
	PolicyViolationCount   bool `json:"policy_violation_count" yaml:"policy_violation_count"`
	AuditLogVolume         bool `json:"audit_log_volume" yaml:"audit_log_volume"`
	SLACompliancePercentage bool `json:"sla_compliance_percentage" yaml:"sla_compliance_percentage"`
	QuotaUtilization       bool `json:"quota_utilization" yaml:"quota_utilization"`
	CostByTenant           bool `json:"cost_by_tenant" yaml:"cost_by_tenant"`
	WorkflowCompletionRate bool `json:"workflow_completion_rate" yaml:"workflow_completion_rate"`
}

// DefaultGovernanceConfig returns production-ready governance configuration
func DefaultGovernanceConfig() *GovernanceConfig {
	return &GovernanceConfig{
		Compliance: ComplianceConfig{
			EnabledStandards:   []string{"soc2", "iso27001", "gdpr"},
			AuditFrequency:     "continuous",
			ReportingEnabled:   true,
			AutoRemediation:    true,
			ComplianceTarget:   0.95,
			EvidenceCollection: true,
			ControlFramework:   "nist",
		},
		AuditLogging: AuditConfig{
			Enabled:           true,
			ImmutableStorage:  true,
			RetentionPeriod:   7 * 365 * 24 * time.Hour, // 7 years
			LogAllActions:     true,
			LogAPIcalls:       true,
			LogUserActivity:   true,
			LogConfigChanges:  true,
			LogAccessAttempts: true,
			ForensicsEnabled:  true,
			SearchableIndex:   true,
			TamperProtection:  true,
			BackupEnabled:     true,
		},
		PolicyEngine: PolicyConfig{
			Enabled:           true,
			EvaluationTimeout: 10 * time.Millisecond,
			CacheEnabled:      true,
			CacheTTL:          5 * time.Minute,
			PolicyCategories:  []string{"access", "quota", "network", "data-residency", "compliance"},
			VersioningEnabled: true,
			RollbackEnabled:   true,
			AuditDecisions:    true,
			PerformanceTarget: 5 * time.Millisecond,
		},
		AccessControl: AccessConfig{
			RBACEnabled:            true,
			ABACEnabled:            true,
			LeastPrivilege:         true,
			SeparationOfDuties:     true,
			RoleHierarchyEnabled:   true,
			DynamicAccessDecisions: true,
			ContextualAttributes:   []string{"time", "location", "device", "risk-score"},
			AuditAccessDecisions:   true,
		},
		MultiTenancy: TenancyConfig{
			Enabled:               true,
			HardIsolation:         true,
			TenantQuotas:          true,
			TenantPolicies:        true,
			CrossTenantValidation: true,
			BillingPerTenant:      true,
			OverheadTarget:        0.03, // 3%
			ResourceTagging:       true,
		},
		QuotaManagement: QuotaConfig{
			Enabled:                true,
			EnforcementMode:        "hard",
			QuotaTypes:             []string{"cpu", "memory", "storage", "network", "vms"},
			AlertThresholds:        []int{80, 90, 100},
			RequestWorkflowEnabled: true,
			AutoScalingEnabled:     false, // Requires approval
			NotificationsEnabled:   true,
		},
		Chargeback: BillingConfig{
			Enabled:            true,
			Mode:               "chargeback",
			UsageBasedBilling:  true,
			CostAllocationTags: []string{"tenant", "project", "environment", "owner"},
			ReportingPeriod:    "monthly",
			BudgetManagement:   true,
			CostAlertsEnabled:  true,
			InvoiceGeneration:  true,
		},
		SLAManagement: SLAConfig{
			Enabled:            true,
			AvailabilityTarget: 0.9995, // 99.95%
			LatencyTargetP95:   100 * time.Millisecond,
			ThroughputTarget:   10000, // 10k req/sec
			ErrorRateTarget:    0.001, // 0.1%
			MeasurementWindow:  30 * 24 * time.Hour,
			ViolationDetection: true,
			AutoNotifications:  true,
			ErrorBudgetEnabled: true,
			DashboardEnabled:   true,
		},
		Workflow: WorkflowConfig{
			Enabled:              true,
			ApprovalWorkflows:    true,
			MultiStageApprovals:  true,
			ConditionalRouting:   true,
			TimeoutHandling:      true,
			DefaultTimeout:       24 * time.Hour,
			NotificationChannels: []string{"email", "slack"},
			WorkflowTemplates:    true,
			AuditTrailEnabled:    true,
		},
		Reporting: ReportingConfig{
			Enabled:             true,
			AutomatedReports:    true,
			ReportFormats:       []string{"pdf", "html"},
			EvidenceCollection:  true,
			GapAnalysis:         true,
			RemediationTracking: true,
			ExecutiveSummaries:  true,
			ScheduledReports:    true,
			ReportingFrequency:  "monthly",
		},
		Classification: ClassificationConfig{
			Enabled:               true,
			AutoClassification:    true,
			SensitivityLevels:     []string{"public", "internal", "confidential", "restricted"},
			EncryptionByLevel:     true,
			AccessPoliciesByLevel: true,
			DLPEnabled:            true,
			ContentScanning:       true,
		},
		Secrets: SecretsConfig{
			Enabled:             true,
			Provider:            "vault",
			AutoRotation:        true,
			RotationPeriod:      90 * 24 * time.Hour, // 90 days
			AuditSecretAccess:   true,
			EncryptionAtRest:    true,
			EncryptionInTransit: true,
		},
		Metrics: MetricsConfig{
			Enabled:                 true,
			ComplianceScore:         true,
			PolicyViolationCount:    true,
			AuditLogVolume:          true,
			SLACompliancePercentage: true,
			QuotaUtilization:        true,
			CostByTenant:            true,
			WorkflowCompletionRate:  true,
		},
	}
}
