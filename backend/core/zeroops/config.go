package zeroops

import (
	"time"
)

// ZeroOpsConfig configures the zero-ops automation system
type ZeroOpsConfig struct {
	// Core automation settings
	EnableFullAutomation   bool          `json:"enable_full_automation" yaml:"enable_full_automation"`
	HumanApproval          bool          `json:"human_approval" yaml:"human_approval"`
	MaxAutomatedCost       int64         `json:"max_automated_cost" yaml:"max_automated_cost"` // $/hour
	SafetyConstraints      SafetyConfig  `json:"safety_constraints" yaml:"safety_constraints"`
	ChaosEngineeringDaily  bool          `json:"chaos_engineering_daily" yaml:"chaos_engineering_daily"`
	AlertOnlyP0            bool          `json:"alert_only_p0" yaml:"alert_only_p0"`

	// Performance targets
	TargetMTTD             time.Duration `json:"target_mttd" yaml:"target_mttd"`
	TargetMTTR             time.Duration `json:"target_mttr" yaml:"target_mttr"`
	TargetAutomationRate   float64       `json:"target_automation_rate" yaml:"target_automation_rate"`
	MaxFalseAlertRate      float64       `json:"max_false_alert_rate" yaml:"max_false_alert_rate"`

	// Scaling configuration
	ScalingConfig          ScalingConfig `json:"scaling_config" yaml:"scaling_config"`

	// Budget configuration
	BudgetConfig           BudgetConfig  `json:"budget_config" yaml:"budget_config"`

	// Alerting configuration
	AlertingConfig         AlertConfig   `json:"alerting_config" yaml:"alerting_config"`
}

// SafetyConfig defines safety constraints for autonomous operations
type SafetyConfig struct {
	RequireApprovalAbove   int64         `json:"require_approval_above" yaml:"require_approval_above"` // $
	MaxVMsAutoProvisioned  int           `json:"max_vms_auto_provisioned" yaml:"max_vms_auto_provisioned"`
	MaxDataDeleted         int64         `json:"max_data_deleted" yaml:"max_data_deleted"` // bytes
	RateLimitActions       int           `json:"rate_limit_actions" yaml:"rate_limit_actions"` // per minute
	MaxScaleUpPercent      float64       `json:"max_scale_up_percent" yaml:"max_scale_up_percent"`
	MaxScaleDownPercent    float64       `json:"max_scale_down_percent" yaml:"max_scale_down_percent"`
	RequireMultiApproval   bool          `json:"require_multi_approval" yaml:"require_multi_approval"`
	BusinessHoursOnly      bool          `json:"business_hours_only" yaml:"business_hours_only"`
	CanaryRegionsFirst     bool          `json:"canary_regions_first" yaml:"canary_regions_first"`
}

// ScalingConfig defines autonomous scaling parameters
type ScalingConfig struct {
	PredictionWindowMinutes int           `json:"prediction_window_minutes" yaml:"prediction_window_minutes"`
	MinPredictionAccuracy   float64       `json:"min_prediction_accuracy" yaml:"min_prediction_accuracy"`
	ScaleUpThreshold        float64       `json:"scale_up_threshold" yaml:"scale_up_threshold"`
	ScaleDownThreshold      float64       `json:"scale_down_threshold" yaml:"scale_down_threshold"`
	ScaleToZeroIdleMinutes  int           `json:"scale_to_zero_idle_minutes" yaml:"scale_to_zero_idle_minutes"`
	ScaleFromZeroMaxSeconds int           `json:"scale_from_zero_max_seconds" yaml:"scale_from_zero_max_seconds"`
	CostOptimizationWeight  float64       `json:"cost_optimization_weight" yaml:"cost_optimization_weight"`
	PerformanceWeight       float64       `json:"performance_weight" yaml:"performance_weight"`
}

// BudgetConfig defines autonomous budget management
type BudgetConfig struct {
	MonthlyBudget          int64         `json:"monthly_budget" yaml:"monthly_budget"`
	AlertThreshold         float64       `json:"alert_threshold" yaml:"alert_threshold"`
	EnforceHardLimit       bool          `json:"enforce_hard_limit" yaml:"enforce_hard_limit"`
	AutoScaleDownAtPercent float64       `json:"auto_scale_down_at_percent" yaml:"auto_scale_down_at_percent"`
	ForecastDays           int           `json:"forecast_days" yaml:"forecast_days"`
	CostAnomalyThreshold   float64       `json:"cost_anomaly_threshold" yaml:"cost_anomaly_threshold"`
}

// AlertConfig defines intelligent alerting configuration
type AlertConfig struct {
	MLSuppressionEnabled   bool          `json:"ml_suppression_enabled" yaml:"ml_suppression_enabled"`
	CorrelationWindow      time.Duration `json:"correlation_window" yaml:"correlation_window"`
	MinAlertSeverity       string        `json:"min_alert_severity" yaml:"min_alert_severity"` // P0, P1, P2, P3, P4
	AutoRemediateBeforeAlert bool        `json:"auto_remediate_before_alert" yaml:"auto_remediate_before_alert"`
	MaxAlertsPerHour       int           `json:"max_alerts_per_hour" yaml:"max_alerts_per_hour"`
	FalsePositiveThreshold float64       `json:"false_positive_threshold" yaml:"false_positive_threshold"`
}

// DefaultZeroOpsConfig returns production-ready zero-ops configuration
func DefaultZeroOpsConfig() *ZeroOpsConfig {
	return &ZeroOpsConfig{
		EnableFullAutomation:   true,
		HumanApproval:          false,
		MaxAutomatedCost:       10000, // $10,000/hour
		ChaosEngineeringDaily:  true,
		AlertOnlyP0:            false, // Alert P0 and P1
		TargetMTTD:             10 * time.Second,
		TargetMTTR:             60 * time.Second,
		TargetAutomationRate:   0.999, // 99.9%
		MaxFalseAlertRate:      0.0001, // 0.01%

		SafetyConstraints: SafetyConfig{
			RequireApprovalAbove:   1000, // $1,000
			MaxVMsAutoProvisioned:  1000,
			MaxDataDeleted:         1024 * 1024 * 1024 * 1024, // 1TB
			RateLimitActions:       100, // 100 actions/min
			MaxScaleUpPercent:      200, // 200% max scale up
			MaxScaleDownPercent:    50,  // 50% max scale down
			RequireMultiApproval:   true,
			BusinessHoursOnly:      false,
			CanaryRegionsFirst:     true,
		},

		ScalingConfig: ScalingConfig{
			PredictionWindowMinutes: 15,
			MinPredictionAccuracy:   0.90,
			ScaleUpThreshold:        0.70,
			ScaleDownThreshold:      0.30,
			ScaleToZeroIdleMinutes:  60,
			ScaleFromZeroMaxSeconds: 30,
			CostOptimizationWeight:  0.40,
			PerformanceWeight:       0.60,
		},

		BudgetConfig: BudgetConfig{
			MonthlyBudget:          1000000, // $1M/month
			AlertThreshold:         0.80,    // Alert at 80%
			EnforceHardLimit:       true,
			AutoScaleDownAtPercent: 0.90,    // Scale down at 90%
			ForecastDays:           30,
			CostAnomalyThreshold:   0.50,    // 50% spike
		},

		AlertingConfig: AlertConfig{
			MLSuppressionEnabled:     true,
			CorrelationWindow:        5 * time.Minute,
			MinAlertSeverity:         "P1",
			AutoRemediateBeforeAlert: true,
			MaxAlertsPerHour:         10,
			FalsePositiveThreshold:   0.01,
		},
	}
}

// DecisionContext provides context for autonomous decision making
type DecisionContext struct {
	Timestamp          time.Time              `json:"timestamp"`
	DecisionType       string                 `json:"decision_type"`
	Confidence         float64                `json:"confidence"`
	EstimatedCost      float64                `json:"estimated_cost"`
	EstimatedImpact    string                 `json:"estimated_impact"`
	Alternatives       []Alternative          `json:"alternatives"`
	RequiresApproval   bool                   `json:"requires_approval"`
	ApprovalReason     string                 `json:"approval_reason,omitempty"`
	AutomatedAction    bool                   `json:"automated_action"`
	Metadata           map[string]interface{} `json:"metadata"`
}

// Alternative represents an alternative decision option
type Alternative struct {
	Name            string                 `json:"name"`
	Confidence      float64                `json:"confidence"`
	EstimatedCost   float64                `json:"estimated_cost"`
	EstimatedImpact string                 `json:"estimated_impact"`
	Pros            []string               `json:"pros"`
	Cons            []string               `json:"cons"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// AutomationMetrics tracks zero-ops performance metrics
type AutomationMetrics struct {
	Timestamp               time.Time `json:"timestamp"`
	HumanInterventionRate   float64   `json:"human_intervention_rate"`
	AutomationSuccessRate   float64   `json:"automation_success_rate"`
	AverageMTTD             float64   `json:"average_mttd_seconds"`
	AverageMTTR             float64   `json:"average_mttr_seconds"`
	CostOptimizationSavings float64   `json:"cost_optimization_savings"`
	Availability            float64   `json:"availability"`
	ChangeSuccessRate       float64   `json:"change_success_rate"`
	FalseAlertRate          float64   `json:"false_alert_rate"`
	TotalDecisions          int64     `json:"total_decisions"`
	AutomatedDecisions      int64     `json:"automated_decisions"`
	ManualDecisions         int64     `json:"manual_decisions"`
}

// IncidentSeverity defines incident priority levels
type IncidentSeverity string

const (
	SeverityP0 IncidentSeverity = "P0" // Catastrophic - total system failure
	SeverityP1 IncidentSeverity = "P1" // Critical - major service degradation
	SeverityP2 IncidentSeverity = "P2" // High - partial service impact
	SeverityP3 IncidentSeverity = "P3" // Medium - minor service impact
	SeverityP4 IncidentSeverity = "P4" // Low - no service impact
)

// ActionType defines types of autonomous actions
type ActionType string

const (
	ActionScaleUp          ActionType = "scale_up"
	ActionScaleDown        ActionType = "scale_down"
	ActionProvision        ActionType = "provision"
	ActionDeprovision      ActionType = "deprovision"
	ActionRestart          ActionType = "restart"
	ActionPatch            ActionType = "patch"
	ActionRollback         ActionType = "rollback"
	ActionFailover         ActionType = "failover"
	ActionThrottle         ActionType = "throttle"
	ActionBlockTraffic     ActionType = "block_traffic"
	ActionAlertEscalate    ActionType = "alert_escalate"
)
