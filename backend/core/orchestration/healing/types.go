package healing

import (
	"time"
)

// HealingController defines the interface for self-healing operations
type HealingController interface {
	// StartMonitoring begins health monitoring
	StartMonitoring() error
	
	// StopMonitoring stops all monitoring activities
	StopMonitoring() error
	
	// RegisterTarget registers a target for health monitoring
	RegisterTarget(target *HealingTarget) error
	
	// UnregisterTarget removes a target from monitoring
	UnregisterTarget(targetID string) error
	
	// GetHealthStatus gets the current health status of a target
	GetHealthStatus(targetID string) (*HealthStatus, error)
	
	// TriggerHealing manually triggers healing for a target
	TriggerHealing(targetID string, reason string) (*HealingDecision, error)
}

// FailureDetector defines interface for failure detection algorithms
type FailureDetector interface {
	// AddSample adds a new health sample
	AddSample(targetID string, sample *HealthSample) error
	
	// IsHealthy determines if a target is healthy based on samples
	IsHealthy(targetID string) (*HealthAssessment, error)
	
	// GetHealthScore gets the current health score for a target
	GetHealthScore(targetID string) (float64, error)
	
	// Configure configures the failure detector parameters
	Configure(config *FailureDetectorConfig) error
}

// RecoveryStrategy defines interface for recovery strategies
type RecoveryStrategy interface {
	// GetName returns the strategy name
	GetName() string
	
	// CanRecover determines if this strategy can handle the failure
	CanRecover(failure *FailureInfo) bool
	
	// Recover executes the recovery action
	Recover(failure *FailureInfo, target *HealingTarget) (*RecoveryResult, error)
	
	// GetPriority returns the strategy priority (higher = more preferred)
	GetPriority() int
	
	// EstimateTime estimates recovery time
	EstimateTime(failure *FailureInfo) time.Duration
}

// HealingTarget represents a target for self-healing
type HealingTarget struct {
	ID                string                 `json:"id"`
	Type              TargetType             `json:"type"`
	Name              string                 `json:"name"`
	Enabled           bool                   `json:"enabled"`
	HealthCheckConfig *HealthCheckConfig     `json:"health_check_config"`
	RecoveryConfig    *RecoveryConfig        `json:"recovery_config"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt         time.Time              `json:"created_at"`
	UpdatedAt         time.Time              `json:"updated_at"`
}

// TargetType represents the type of healing target
type TargetType string

const (
	TargetTypeVM      TargetType = "vm"
	TargetTypeNode    TargetType = "node"
	TargetTypeService TargetType = "service"
	TargetTypeCluster TargetType = "cluster"
)

// HealthCheckConfig defines how to check target health
type HealthCheckConfig struct {
	Interval              time.Duration          `json:"interval"`
	Timeout               time.Duration          `json:"timeout"`
	HealthyThreshold      int                    `json:"healthy_threshold"`
	UnhealthyThreshold    int                    `json:"unhealthy_threshold"`
	FailureThreshold      int                    `json:"failure_threshold"`
	CheckType             HealthCheckType        `json:"check_type"`
	CheckParameters       map[string]interface{} `json:"check_parameters,omitempty"`
	EnableProactiveChecks bool                   `json:"enable_proactive_checks"`
}

// HealthCheckType represents the type of health check
type HealthCheckType string

const (
	HealthCheckTypeHTTP     HealthCheckType = "http"
	HealthCheckTypeTCP      HealthCheckType = "tcp"
	HealthCheckTypePing     HealthCheckType = "ping"
	HealthCheckTypeCustom   HealthCheckType = "custom"
	HealthCheckTypeMetrics  HealthCheckType = "metrics"
)

// RecoveryConfig defines recovery behavior
type RecoveryConfig struct {
	EnableAutoRecovery    bool                   `json:"enable_auto_recovery"`
	MaxRecoveryAttempts   int                    `json:"max_recovery_attempts"`
	RecoveryTimeout       time.Duration          `json:"recovery_timeout"`
	BackoffStrategy       BackoffStrategy        `json:"backoff_strategy"`
	PreferredStrategies   []string               `json:"preferred_strategies,omitempty"`
	FallbackStrategies    []string               `json:"fallback_strategies,omitempty"`
	NotificationConfig    *NotificationConfig    `json:"notification_config,omitempty"`
	CustomParameters      map[string]interface{} `json:"custom_parameters,omitempty"`
}

// BackoffStrategy represents the backoff strategy for recovery attempts
type BackoffStrategy string

const (
	BackoffFixed       BackoffStrategy = "fixed"
	BackoffExponential BackoffStrategy = "exponential"
	BackoffLinear      BackoffStrategy = "linear"
)

// NotificationConfig defines notification settings
type NotificationConfig struct {
	Enabled       bool     `json:"enabled"`
	Channels      []string `json:"channels"`
	OnFailure     bool     `json:"on_failure"`
	OnRecovery    bool     `json:"on_recovery"`
	OnAttempt     bool     `json:"on_attempt"`
	IncludeEvents bool     `json:"include_events"`
}

// HealthSample represents a health check sample
type HealthSample struct {
	TargetID      string                 `json:"target_id"`
	Timestamp     time.Time              `json:"timestamp"`
	Healthy       bool                   `json:"healthy"`
	ResponseTime  time.Duration          `json:"response_time"`
	ErrorMessage  string                 `json:"error_message,omitempty"`
	Metrics       map[string]float64     `json:"metrics,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// HealthStatus represents the current health status of a target
type HealthStatus struct {
	TargetID           string        `json:"target_id"`
	Healthy            bool          `json:"healthy"`
	HealthScore        float64       `json:"health_score"`
	LastCheckTime      time.Time     `json:"last_check_time"`
	ConsecutiveFailures int          `json:"consecutive_failures"`
	ConsecutiveSuccess  int          `json:"consecutive_success"`
	UptimePercentage   float64       `json:"uptime_percentage"`
	FailureReason      string        `json:"failure_reason,omitempty"`
	RecoveryStatus     *RecoveryStatus `json:"recovery_status,omitempty"`
}

// HealthAssessment represents a health assessment result
type HealthAssessment struct {
	TargetID     string                 `json:"target_id"`
	Healthy      bool                   `json:"healthy"`
	HealthScore  float64                `json:"health_score"`
	Confidence   float64                `json:"confidence"`
	Reasons      []string               `json:"reasons,omitempty"`
	Timestamp    time.Time              `json:"timestamp"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// FailureInfo represents information about a detected failure
type FailureInfo struct {
	TargetID        string                 `json:"target_id"`
	FailureType     FailureType            `json:"failure_type"`
	Severity        FailureSeverity        `json:"severity"`
	Description     string                 `json:"description"`
	DetectedAt      time.Time              `json:"detected_at"`
	LastOccurrence  time.Time              `json:"last_occurrence"`
	OccurrenceCount int                    `json:"occurrence_count"`
	Symptoms        []string               `json:"symptoms,omitempty"`
	Context         map[string]interface{} `json:"context,omitempty"`
}

// FailureType represents the type of failure
type FailureType string

const (
	FailureTypeUnresponsive FailureType = "unresponsive"
	FailureTypeHighLatency  FailureType = "high_latency"
	FailureTypeHighError    FailureType = "high_error_rate"
	FailureTypeResourceExhaustion FailureType = "resource_exhaustion"
	FailureTypeNetworkIssue FailureType = "network_issue"
	FailureTypeServiceDown  FailureType = "service_down"
	FailureTypeCustom       FailureType = "custom"
)

// FailureSeverity represents the severity of a failure
type FailureSeverity string

const (
	SeverityLow      FailureSeverity = "low"
	SeverityMedium   FailureSeverity = "medium"
	SeverityHigh     FailureSeverity = "high"
	SeverityCritical FailureSeverity = "critical"
)

// HealingDecision represents a decision to heal a target
type HealingDecision struct {
	ID               string                 `json:"id"`
	TargetID         string                 `json:"target_id"`
	DecisionTime     time.Time              `json:"decision_time"`
	FailureInfo      *FailureInfo           `json:"failure_info"`
	Strategy         string                 `json:"strategy"`
	Actions          []HealingAction        `json:"actions"`
	EstimatedTime    time.Duration          `json:"estimated_time"`
	Confidence       float64                `json:"confidence"`
	Reason           string                 `json:"reason"`
	Status           HealingDecisionStatus  `json:"status"`
	Result           *HealingResult         `json:"result,omitempty"`
	Metadata         map[string]interface{} `json:"metadata,omitempty"`
}

// HealingDecisionStatus represents the status of a healing decision
type HealingDecisionStatus string

const (
	HealingStatusPending    HealingDecisionStatus = "pending"
	HealingStatusExecuting  HealingDecisionStatus = "executing"
	HealingStatusSuccessful HealingDecisionStatus = "successful"
	HealingStatusFailed     HealingDecisionStatus = "failed"
	HealingStatusCancelled  HealingDecisionStatus = "cancelled"
)

// HealingAction represents a specific healing action
type HealingAction struct {
	Type       HealingActionType      `json:"type"`
	Target     string                 `json:"target"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Timeout    time.Duration          `json:"timeout,omitempty"`
	DependsOn  []string               `json:"depends_on,omitempty"`
}

// HealingActionType represents the type of healing action
type HealingActionType string

const (
	ActionRestart      HealingActionType = "restart"
	ActionMigrate      HealingActionType = "migrate"
	ActionScale        HealingActionType = "scale"
	ActionFailover     HealingActionType = "failover"
	ActionQuarantine   HealingActionType = "quarantine"
	ActionRollback     HealingActionType = "rollback"
	ActionNotify       HealingActionType = "notify"
	ActionCustomScript HealingActionType = "custom_script"
)

// RecoveryResult represents the result of a recovery action
type RecoveryResult struct {
	Success       bool          `json:"success"`
	Message       string        `json:"message"`
	Duration      time.Duration `json:"duration"`
	ActionsExecuted []string    `json:"actions_executed"`
	Errors        []string      `json:"errors,omitempty"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// RecoveryStatus represents the current recovery status
type RecoveryStatus struct {
	InProgress        bool      `json:"in_progress"`
	CurrentStrategy   string    `json:"current_strategy,omitempty"`
	Attempts          int       `json:"attempts"`
	MaxAttempts       int       `json:"max_attempts"`
	LastAttemptTime   time.Time `json:"last_attempt_time,omitempty"`
	LastAttemptResult string    `json:"last_attempt_result,omitempty"`
}

// HealingResult represents the overall result of a healing operation
type HealingResult struct {
	Success           bool                   `json:"success"`
	TotalTime         time.Duration          `json:"total_time"`
	StrategiesAttempted []string             `json:"strategies_attempted"`
	FinalStrategy     string                 `json:"final_strategy,omitempty"`
	RecoveryResults   []*RecoveryResult      `json:"recovery_results"`
	FinalHealthStatus *HealthStatus          `json:"final_health_status,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

// FailureDetectorConfig represents configuration for failure detection
type FailureDetectorConfig struct {
	Algorithm             DetectionAlgorithm `json:"algorithm"`
	SampleWindowSize      int                `json:"sample_window_size"`
	MinSamplesForDecision int                `json:"min_samples_for_decision"`
	HealthyThreshold      float64            `json:"healthy_threshold"`
	UnhealthyThreshold    float64            `json:"unhealthy_threshold"`
	SuspicionThreshold    float64            `json:"suspicion_threshold"`
	PhiThreshold          float64            `json:"phi_threshold,omitempty"`        // For Phi Accrual
	AcceptableHeartbeat   time.Duration      `json:"acceptable_heartbeat,omitempty"` // For Phi Accrual
	Parameters            map[string]interface{} `json:"parameters,omitempty"`
}

// DetectionAlgorithm represents the failure detection algorithm
type DetectionAlgorithm string

const (
	AlgorithmSimpleThreshold DetectionAlgorithm = "simple_threshold"
	AlgorithmPhiAccrual      DetectionAlgorithm = "phi_accrual"
	AlgorithmAdaptive        DetectionAlgorithm = "adaptive"
	AlgorithmConsensus       DetectionAlgorithm = "consensus"
)

// HealingEvent represents an event in the healing system
type HealingEvent struct {
	Type          EventType              `json:"type"`
	TargetID      string                 `json:"target_id"`
	FailureInfo   *FailureInfo           `json:"failure_info,omitempty"`
	Decision      *HealingDecision       `json:"decision,omitempty"`
	Result        *HealingResult         `json:"result,omitempty"`
	HealthStatus  *HealthStatus          `json:"health_status,omitempty"`
	Timestamp     time.Time              `json:"timestamp"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// EventType represents the type of healing event
type EventType string

const (
	EventTypeFailureDetected  EventType = "healing.failure_detected"
	EventTypeRecoveryStarted  EventType = "healing.recovery_started"
	EventTypeRecoveryCompleted EventType = "healing.recovery_completed"
	EventTypeHealthRestored   EventType = "healing.health_restored"
	EventTypeHealthDegraded   EventType = "healing.health_degraded"
	EventTypeStrategyFailed   EventType = "healing.strategy_failed"
	EventTypeTargetRegistered EventType = "healing.target_registered"
)