package quotas

import (
	"context"
	"time"
)

// QuotaLevel represents the level at which a quota applies
type QuotaLevel string

const (
	// QuotaLevelSystem applies at the system level (global)
	QuotaLevelSystem QuotaLevel = "system"
	
	// QuotaLevelTenant applies at the tenant level
	QuotaLevelTenant QuotaLevel = "tenant"
	
	// QuotaLevelUser applies at the user level
	QuotaLevelUser QuotaLevel = "user"
	
	// QuotaLevelProject applies at the project level
	QuotaLevelProject QuotaLevel = "project"
)

// ResourceType represents the type of resource being managed
type ResourceType string

const (
	// Compute Resources
	ResourceTypeCPU          ResourceType = "cpu"
	ResourceTypeMemory       ResourceType = "memory"
	ResourceTypeVCPUs        ResourceType = "vcpus"
	ResourceTypeInstances    ResourceType = "instances"
	ResourceTypeGPU          ResourceType = "gpu"
	
	// Storage Resources
	ResourceTypeStorage      ResourceType = "storage"
	ResourceTypeVolumes      ResourceType = "volumes"
	ResourceTypeSnapshots    ResourceType = "snapshots"
	ResourceTypeBackups      ResourceType = "backups"
	
	// Network Resources
	ResourceTypeBandwidthIn  ResourceType = "bandwidth_in"
	ResourceTypeBandwidthOut ResourceType = "bandwidth_out"
	ResourceTypeConnections  ResourceType = "connections"
	ResourceTypeLoadBalancers ResourceType = "load_balancers"
	ResourceTypeFloatingIPs  ResourceType = "floating_ips"
	
	// API Resources
	ResourceTypeAPIRequests  ResourceType = "api_requests"
	ResourceTypeAPIRate      ResourceType = "api_rate"
	
	// Cost Resources
	ResourceTypeCost         ResourceType = "cost"
	ResourceTypeBudget       ResourceType = "budget"
)

// LimitType represents different types of limits
type LimitType string

const (
	// LimitTypeHard represents a hard limit that cannot be exceeded
	LimitTypeHard LimitType = "hard"
	
	// LimitTypeSoft represents a soft limit that can be exceeded with warnings
	LimitTypeSoft LimitType = "soft"
	
	// LimitTypeBurst represents a burst limit for temporary overages
	LimitTypeBurst LimitType = "burst"
)

// QuotaStatus represents the current status of a quota
type QuotaStatus string

const (
	// QuotaStatusActive indicates the quota is active and enforced
	QuotaStatusActive QuotaStatus = "active"
	
	// QuotaStatusSuspended indicates the quota is temporarily suspended
	QuotaStatusSuspended QuotaStatus = "suspended"
	
	// QuotaStatusExceeded indicates the quota has been exceeded
	QuotaStatusExceeded QuotaStatus = "exceeded"
	
	// QuotaStatusExpired indicates the quota has expired
	QuotaStatusExpired QuotaStatus = "expired"
)

// Quota represents a resource quota
type Quota struct {
	// Basic identification
	ID        string     `json:"id"`
	Name      string     `json:"name"`
	
	// Hierarchy and targeting
	Level    QuotaLevel `json:"level"`
	EntityID string     `json:"entity_id"` // ID of tenant, user, project, etc.
	ParentID string     `json:"parent_id,omitempty"` // For hierarchical quotas
	
	// Resource specification
	ResourceType ResourceType `json:"resource_type"`
	LimitType    LimitType    `json:"limit_type"`
	
	// Limits and usage
	Limit       int64 `json:"limit"`        // Primary limit
	BurstLimit  int64 `json:"burst_limit"`  // Temporary burst capacity
	Used        int64 `json:"used"`         // Current usage
	Reserved    int64 `json:"reserved"`     // Reserved for future use
	Available   int64 `json:"available"`    // Computed: Limit - Used - Reserved
	
	// Status and priority
	Status   QuotaStatus `json:"status"`
	Priority int         `json:"priority"` // Higher number = higher priority
	
	// Time-based controls
	StartTime *time.Time `json:"start_time,omitempty"` // When quota becomes active
	EndTime   *time.Time `json:"end_time,omitempty"`   // When quota expires
	
	// Cost management
	CostPerUnit      float64 `json:"cost_per_unit,omitempty"`      // Cost per resource unit
	BudgetLimit      float64 `json:"budget_limit,omitempty"`       // Budget-based limit
	CurrentCost      float64 `json:"current_cost,omitempty"`       // Current cost
	
	// Enforcement behavior
	EnforcementAction   EnforcementAction `json:"enforcement_action"`
	GracePeriod        time.Duration     `json:"grace_period,omitempty"`        // Grace period before enforcement
	NotificationThreshold float64        `json:"notification_threshold,omitempty"` // % threshold for notifications
	
	// Rate limiting (for API quotas)
	RateLimit     int64         `json:"rate_limit,omitempty"`     // Requests per time window
	TimeWindow    time.Duration `json:"time_window,omitempty"`    // Time window for rate limiting
	
	// Policy references
	PolicyID    string   `json:"policy_id,omitempty"`    // Associated policy
	Tags        []string `json:"tags,omitempty"`         // Searchable tags
	
	// Metadata and compliance
	Metadata              map[string]interface{}   `json:"metadata,omitempty"`
	ComplianceRequirements []ComplianceRequirement `json:"compliance_requirements,omitempty"`
	
	// Alerting configuration
	Alerts []QuotaAlert `json:"alerts,omitempty"`
	
	// Scheduling information
	ScheduledActions []ScheduledAction `json:"scheduled_actions,omitempty"`
	
	// Timestamps
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
	
	// Auto-scaling configuration
	AutoScaling *AutoScalingConfig `json:"auto_scaling,omitempty"`
}

// EnforcementAction represents actions taken on quota violations
type EnforcementAction string

const (
	// EnforcementActionDeny blocks the resource request
	EnforcementActionDeny EnforcementAction = "deny"
	
	// EnforcementActionWarn allows but sends warnings
	EnforcementActionWarn EnforcementAction = "warn"
	
	// EnforcementActionThrottle rate-limits the requests
	EnforcementActionThrottle EnforcementAction = "throttle"
	
	// EnforcementActionScale triggers auto-scaling
	EnforcementActionScale EnforcementAction = "scale"
	
	// EnforcementActionCustom uses custom handler
	EnforcementActionCustom EnforcementAction = "custom"
)

// AutoScalingConfig defines auto-scaling behavior
type AutoScalingConfig struct {
	// Whether auto-scaling is enabled
	Enabled bool `json:"enabled"`
	
	// Threshold percentage for scaling up
	ScaleUpThreshold float64 `json:"scale_up_threshold"`
	
	// Scale up factor (multiplier)
	ScaleUpFactor float64 `json:"scale_up_factor"`
	
	// Maximum limit after auto-scaling
	MaxLimit int64 `json:"max_limit"`
	
	// Cooldown period between scaling events
	Cooldown time.Duration `json:"cooldown"`
	
	// Cost impact assessment
	CostImpact float64 `json:"cost_impact,omitempty"`
}

// QuotaAlert defines alerting configuration
type QuotaAlert struct {
	// Alert ID
	ID string `json:"id"`
	
	// Threshold percentage (0-100)
	Threshold float64 `json:"threshold"`
	
	// Alert message template
	MessageTemplate string `json:"message_template"`
	
	// Notification channels
	Channels []string `json:"channels"`
	
	// Alert severity
	Severity AlertSeverity `json:"severity"`
	
	// Rate limiting for alerts
	RateLimit time.Duration `json:"rate_limit,omitempty"`
	
	// Whether the alert is enabled
	Enabled bool `json:"enabled"`
}

// AlertSeverity represents alert severity levels
type AlertSeverity string

const (
	// AlertSeverityInfo for informational alerts
	AlertSeverityInfo AlertSeverity = "info"
	
	// AlertSeverityWarning for warning alerts
	AlertSeverityWarning AlertSeverity = "warning"
	
	// AlertSeverityCritical for critical alerts
	AlertSeverityCritical AlertSeverity = "critical"
	
	// AlertSeverityEmergency for emergency alerts
	AlertSeverityEmergency AlertSeverity = "emergency"
)

// UsageRecord represents resource usage tracking
type UsageRecord struct {
	// Unique identifier
	ID string `json:"id"`
	
	// Quota ID this usage applies to
	QuotaID string `json:"quota_id"`
	
	// Entity consuming the resource
	EntityID string `json:"entity_id"`
	
	// Resource type
	ResourceType ResourceType `json:"resource_type"`
	
	// Usage amounts
	Amount int64 `json:"amount"` // Total amount used
	Delta  int64 `json:"delta"`  // Change in usage (+/-)
	
	// Timestamp
	Timestamp time.Time `json:"timestamp"`
	
	// Source of the usage
	Source string `json:"source"` // "vm_creation", "api_call", etc.
	
	// Additional context
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	
	// Cost information
	Cost     float64 `json:"cost,omitempty"`
	Currency string  `json:"currency,omitempty"`
}

// ResourceReservation represents a resource reservation
type ResourceReservation struct {
	// Unique identifier
	ID string `json:"id"`
	
	// Entity making the reservation
	EntityID string `json:"entity_id"`
	
	// Resource being reserved
	ResourceType ResourceType `json:"resource_type"`
	Amount       int64        `json:"amount"`
	
	// Time range for the reservation
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	
	// Purpose and metadata
	Purpose     string                 `json:"purpose,omitempty"`
	Description string                 `json:"description,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	
	// Status
	Status ReservationStatus `json:"status"`
	
	// Timestamps
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// ReservationStatus represents reservation status
type ReservationStatus string

const (
	ReservationStatusPending   ReservationStatus = "pending"
	ReservationStatusActive    ReservationStatus = "active"
	ReservationStatusCompleted ReservationStatus = "completed"
	ReservationStatusCancelled ReservationStatus = "cancelled"
)

// QuotaCheckResult represents the result of a quota check
type QuotaCheckResult struct {
	// Whether the request is allowed
	Allowed bool `json:"allowed"`
	
	// Quota that applies to this check
	Quota *Quota `json:"quota"`
	
	// Reason if denied
	Reason string `json:"reason,omitempty"`
	
	// Current utilization percentage
	Utilization float64 `json:"utilization"`
	
	// Available capacity
	Available int64 `json:"available"`
	
	// Whether using burst capacity
	UsingBurst bool `json:"using_burst,omitempty"`
	
	// Time until quota resets (for rate limits)
	ResetTime *time.Time `json:"reset_time,omitempty"`
	
	// Recommendations
	Recommendations []string `json:"recommendations,omitempty"`
}

// QuotaFilter represents filtering criteria for quota queries
type QuotaFilter struct {
	EntityID     string       `json:"entity_id,omitempty"`
	ResourceType ResourceType `json:"resource_type,omitempty"`
	Status       QuotaStatus  `json:"status,omitempty"`
	Level        QuotaLevel   `json:"level,omitempty"`
	Tags         []string     `json:"tags,omitempty"`
}

// ReservationFilter represents filtering criteria for reservation queries
type ReservationFilter struct {
	EntityID     string       `json:"entity_id,omitempty"`
	ResourceType ResourceType `json:"resource_type,omitempty"`
	Status       ReservationStatus `json:"status,omitempty"`
	StartAfter   *time.Time   `json:"start_after,omitempty"`
	EndBefore    *time.Time   `json:"end_before,omitempty"`
}

// QuotaUtilization represents quota utilization information
type QuotaUtilization struct {
	// Entity ID
	EntityID string `json:"entity_id"`
	
	// Per-resource utilization
	ResourceUtilization map[ResourceType]*ResourceUtilization `json:"resource_utilization"`
	
	// Top consumers
	TopConsumers []ResourceConsumer `json:"top_consumers"`
	
	// Aggregate cost information
	TotalCost            float64 `json:"total_cost"`
	ProjectedMonthlyCost float64 `json:"projected_monthly_cost"`
	
	// Timestamp
	Timestamp time.Time `json:"timestamp"`
}

// ResourceUtilization represents utilization for a specific resource
type ResourceUtilization struct {
	ResourceType ResourceType `json:"resource_type"`
	Limit        int64        `json:"limit"`
	Used         int64        `json:"used"`
	Reserved     int64        `json:"reserved"`
	Available    int64        `json:"available"`
	Utilization  float64      `json:"utilization"` // Percentage
}

// ResourceConsumer represents resource consumption information
type ResourceConsumer struct {
	EntityID     string       `json:"entity_id"`
	ResourceType ResourceType `json:"resource_type"`
	Amount       int64        `json:"amount"`
	Cost         float64      `json:"cost"`
	Percentage   float64      `json:"percentage"`
}

// CostAnalysis represents cost analysis results
type CostAnalysis struct {
	// Entity ID
	EntityID string `json:"entity_id"`
	
	// Analysis period
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
	
	// Cost breakdown
	TotalCost     float64                      `json:"total_cost"`
	ResourceCosts map[ResourceType]float64     `json:"resource_costs"`
	CostTrends    []*CostTrend                `json:"cost_trends"`
	
	// Recommendations
	Optimizations []*CostOptimization `json:"optimizations"`
	
	// Projections
	ProjectedMonthlyCost float64 `json:"projected_monthly_cost"`
	ProjectedYearlyCost  float64 `json:"projected_yearly_cost"`
}

// CostTrend represents cost trend information
type CostTrend struct {
	Timestamp time.Time `json:"timestamp"`
	Cost      float64   `json:"cost"`
	Change    float64   `json:"change"` // Percentage change from previous period
}

// CostOptimization represents a cost optimization recommendation
type CostOptimization struct {
	Type        string  `json:"type"`        // "downsize", "reserved", "spot", etc.
	Description string  `json:"description"`
	Savings     float64 `json:"savings"`     // Projected savings
	Confidence  float64 `json:"confidence"`  // Confidence level (0-1)
	Action      string  `json:"action"`      // Recommended action
}

// TimeRange represents a time range for queries and analytics
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// QuotaTemplate represents a template for creating multiple quotas
type QuotaTemplate struct {
	// Basic identification
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	
	// Template category
	Category string `json:"category"` // "service-tier", "workload-type", etc.
	
	// Service tier (if applicable)
	ServiceTier ServiceTier `json:"service_tier,omitempty"`
	
	// Quota definitions in the template
	Quotas []QuotaDefinition `json:"quotas"`
	
	// Cost information
	MonthlyCost float64 `json:"monthly_cost,omitempty"`
	
	// Template metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`
	
	// Timestamps
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// QuotaDefinition represents a quota definition within a template
type QuotaDefinition struct {
	// Resource type
	ResourceType ResourceType `json:"resource_type"`
	
	// Limit type
	LimitType LimitType `json:"limit_type"`
	
	// Limits
	Limit      int64 `json:"limit"`
	BurstLimit int64 `json:"burst_limit,omitempty"`
	
	// Priority and enforcement
	Priority          int               `json:"priority"`
	EnforcementAction EnforcementAction `json:"enforcement_action"`
	
	// Cost per unit
	CostPerUnit float64 `json:"cost_per_unit,omitempty"`
}

// ServiceTier represents different service tiers
type ServiceTier string

const (
	ServiceTierFree       ServiceTier = "free"
	ServiceTierDeveloper  ServiceTier = "developer"
	ServiceTierStartup    ServiceTier = "startup"
	ServiceTierGrowth     ServiceTier = "growth"
	ServiceTierEnterprise ServiceTier = "enterprise"
)

// QuotaService defines the interface for quota management operations
type QuotaService interface {
	// Core quota operations
	CreateQuota(ctx context.Context, quota *Quota) error
	GetQuota(ctx context.Context, quotaID string) (*Quota, error)
	UpdateQuota(ctx context.Context, quota *Quota) error
	DeleteQuota(ctx context.Context, quotaID string) error
	ListQuotas(ctx context.Context, filter QuotaFilter) ([]*Quota, error)
	
	// Quota checking and enforcement
	CheckQuota(ctx context.Context, entityID string, resourceType ResourceType, amount int64) (*QuotaCheckResult, error)
	ConsumeResource(ctx context.Context, usage *UsageRecord) error
	ReleaseResource(ctx context.Context, entityID string, resourceType ResourceType, amount int64) error
	
	// Resource reservations
	ReserveResource(ctx context.Context, reservation *ResourceReservation) error
	CancelReservation(ctx context.Context, reservationID string) error
	
	// Analytics and reporting
	GetQuotaUtilization(ctx context.Context, entityID string) (*QuotaUtilization, error)
}

// ScheduledAction represents a scheduled action on a quota
type ScheduledAction struct {
	ID          string                 `json:"id"`
	Action      ScheduledActionType    `json:"action"`
	ScheduledAt time.Time              `json:"scheduled_at"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
	Status      ScheduledActionStatus  `json:"status"`
	ExecutedAt  *time.Time            `json:"executed_at,omitempty"`
}

// ScheduledActionType represents types of scheduled actions
type ScheduledActionType string

const (
	ScheduledActionIncrease  ScheduledActionType = "increase"
	ScheduledActionDecrease  ScheduledActionType = "decrease"
	ScheduledActionSuspend   ScheduledActionType = "suspend"
	ScheduledActionActivate  ScheduledActionType = "activate"
	ScheduledActionDelete    ScheduledActionType = "delete"
	ScheduledActionNotify    ScheduledActionType = "notify"
)

// ScheduledActionStatus represents the status of scheduled actions
type ScheduledActionStatus string

const (
	ScheduledActionStatusPending   ScheduledActionStatus = "pending"
	ScheduledActionStatusExecuted  ScheduledActionStatus = "executed"
	ScheduledActionStatusFailed    ScheduledActionStatus = "failed"
	ScheduledActionStatusCancelled ScheduledActionStatus = "cancelled"
)

// PolicyRule represents a quota policy rule
type PolicyRule struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	
	// Conditions
	Conditions []PolicyCondition `json:"conditions"`
	
	// Actions to take when conditions match
	Actions []PolicyAction `json:"actions"`
	
	// Priority (higher number = higher priority)
	Priority int `json:"priority"`
	
	// Whether the rule is enabled
	Enabled bool `json:"enabled"`
	
	// Rule metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// PolicyCondition represents a condition in a policy rule
type PolicyCondition struct {
	Field    string      `json:"field"`    // "utilization", "cost", "time", etc.
	Operator string      `json:"operator"` // "gt", "lt", "eq", "contains", etc.
	Value    interface{} `json:"value"`
}

// PolicyAction represents an action in a policy rule
type PolicyAction struct {
	Type       string                 `json:"type"`       // "increase", "alert", "suspend", etc.
	Parameters map[string]interface{} `json:"parameters"`
}

// EnforcementRule represents quota enforcement configuration
type EnforcementRule struct {
	ID            string                 `json:"id"`
	ResourceType  ResourceType          `json:"resource_type"`
	Level         QuotaLevel            `json:"level"`
	Action        EnforcementAction     `json:"action"`
	Threshold     float64               `json:"threshold"`
	Enabled       bool                  `json:"enabled"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// ComplianceRequirement represents compliance requirements
type ComplianceRequirement struct {
	Type        string `json:"type"`        // "gdpr", "hipaa", "sox", etc.
	Requirement string `json:"requirement"` // Specific requirement
	Enforced    bool   `json:"enforced"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}