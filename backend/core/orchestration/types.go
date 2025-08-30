package orchestration

import (
	"context"
	"time"
)

// OrchestrationEngine represents the main orchestration controller
type OrchestrationEngine interface {
	// Start begins orchestration operations
	Start(ctx context.Context) error
	
	// Stop gracefully shuts down the orchestration engine
	Stop(ctx context.Context) error
	
	// GetStatus returns the current status of the orchestration engine
	GetStatus() EngineStatus
	
	// RegisterPolicy registers a new orchestration policy
	RegisterPolicy(policy *OrchestrationPolicy) error
	
	// UnregisterPolicy removes an orchestration policy
	UnregisterPolicy(policyID string) error
}

// EngineStatus represents the current state of the orchestration engine
type EngineStatus struct {
	State         EngineState           `json:"state"`
	StartTime     time.Time             `json:"start_time"`
	ActivePolicies int                 `json:"active_policies"`
	EventsProcessed uint64             `json:"events_processed"`
	Metrics       map[string]interface{} `json:"metrics"`
}

// EngineState represents the current state of the orchestration engine
type EngineState string

const (
	EngineStateStarting EngineState = "starting"
	EngineStateRunning  EngineState = "running"
	EngineStateStopping EngineState = "stopping"
	EngineStateStopped  EngineState = "stopped"
	EngineStateError    EngineState = "error"
)

// OrchestrationPolicy represents a high-level orchestration policy
type OrchestrationPolicy struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Selector    PolicySelector         `json:"selector"`
	Rules       []PolicyRule           `json:"rules"`
	Priority    int                    `json:"priority"`
	Enabled     bool                   `json:"enabled"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// PolicySelector defines which VMs this policy applies to
type PolicySelector struct {
	Tags        map[string]string `json:"tags,omitempty"`
	Labels      map[string]string `json:"labels,omitempty"`
	VMTypes     []string          `json:"vm_types,omitempty"`
	Namespaces  []string          `json:"namespaces,omitempty"`
	Expression  string            `json:"expression,omitempty"` // CEL expression
}

// PolicyRule defines a specific rule within a policy
type PolicyRule struct {
	Type        PolicyRuleType         `json:"type"`
	Parameters  map[string]interface{} `json:"parameters"`
	Conditions  []RuleCondition        `json:"conditions,omitempty"`
	Actions     []RuleAction           `json:"actions"`
	Priority    int                    `json:"priority"`
	Enabled     bool                   `json:"enabled"`
}

// PolicyRuleType defines the type of policy rule
type PolicyRuleType string

const (
	RuleTypePlacement   PolicyRuleType = "placement"
	RuleTypeAutoScaling PolicyRuleType = "autoscaling"
	RuleTypeHealing     PolicyRuleType = "healing"
	RuleTypeLoadBalance PolicyRuleType = "loadbalance"
	RuleTypeSecurity    PolicyRuleType = "security"
	RuleTypeCompliance  PolicyRuleType = "compliance"
)

// RuleCondition defines when a rule should be applied
type RuleCondition struct {
	Type       ConditionType          `json:"type"`
	Operator   ConditionOperator      `json:"operator"`
	Value      interface{}            `json:"value"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// ConditionType defines the type of condition
type ConditionType string

const (
	ConditionTypeCPUUtilization    ConditionType = "cpu_utilization"
	ConditionTypeMemoryUtilization ConditionType = "memory_utilization"
	ConditionTypeNetworkTraffic    ConditionType = "network_traffic"
	ConditionTypeDiskIOPS          ConditionType = "disk_iops"
	ConditionTypeNodeHealth        ConditionType = "node_health"
	ConditionTypeTimeOfDay         ConditionType = "time_of_day"
	ConditionTypeCustomMetric      ConditionType = "custom_metric"
)

// ConditionOperator defines how the condition is evaluated
type ConditionOperator string

const (
	OperatorEquals              ConditionOperator = "eq"
	OperatorNotEquals           ConditionOperator = "ne"
	OperatorGreaterThan         ConditionOperator = "gt"
	OperatorGreaterThanOrEqual  ConditionOperator = "gte"
	OperatorLessThan            ConditionOperator = "lt"
	OperatorLessThanOrEqual     ConditionOperator = "lte"
	OperatorIn                  ConditionOperator = "in"
	OperatorNotIn               ConditionOperator = "not_in"
	OperatorContains            ConditionOperator = "contains"
	OperatorNotContains         ConditionOperator = "not_contains"
)

// RuleAction defines what action should be taken when rule conditions are met
type RuleAction struct {
	Type       ActionType             `json:"type"`
	Parameters map[string]interface{} `json:"parameters"`
	Timeout    time.Duration          `json:"timeout,omitempty"`
	Retries    int                    `json:"retries,omitempty"`
}

// ActionType defines the type of action to take
type ActionType string

const (
	ActionTypeScale       ActionType = "scale"
	ActionTypeMigrate     ActionType = "migrate"
	ActionTypeRestart     ActionType = "restart"
	ActionTypeAlert       ActionType = "alert"
	ActionTypeWebhook     ActionType = "webhook"
	ActionTypeSchedule    ActionType = "schedule"
	ActionTypeQuarantine  ActionType = "quarantine"
	ActionTypeOptimize    ActionType = "optimize"
)

// OrchestrationEvent represents an event in the orchestration system
type OrchestrationEvent struct {
	ID          string                 `json:"id"`
	Type        EventType              `json:"type"`
	Source      string                 `json:"source"`
	Target      string                 `json:"target,omitempty"`
	Timestamp   time.Time              `json:"timestamp"`
	Data        map[string]interface{} `json:"data"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	Priority    EventPriority          `json:"priority"`
	TTL         time.Duration          `json:"ttl,omitempty"`
}

// EventType defines the type of orchestration event
type EventType string

const (
	EventTypeVMCreated         EventType = "vm.created"
	EventTypeVMStarted         EventType = "vm.started"
	EventTypeVMStopped         EventType = "vm.stopped"
	EventTypeVMDeleted         EventType = "vm.deleted"
	EventTypeVMMetrics         EventType = "vm.metrics"
	EventTypeNodeMetrics       EventType = "node.metrics"
	EventTypeNodeFailure       EventType = "node.failure"
	EventTypeNodeRecovered     EventType = "node.recovered"
	EventTypeScalingTriggered  EventType = "scaling.triggered"
	EventTypeHealingTriggered  EventType = "healing.triggered"
	EventTypePolicyUpdated     EventType = "policy.updated"
	EventTypeOrchestrationLog  EventType = "orchestration.log"
)

// EventPriority defines the priority level of an event
type EventPriority int

const (
	PriorityLow      EventPriority = 1
	PriorityNormal   EventPriority = 3
	PriorityHigh     EventPriority = 5
	PriorityCritical EventPriority = 10
)

// DecisionContext provides context for orchestration decisions
type DecisionContext struct {
	RequestID   string                 `json:"request_id"`
	Timestamp   time.Time              `json:"timestamp"`
	Trigger     string                 `json:"trigger"`
	VMID        string                 `json:"vm_id,omitempty"`
	NodeID      string                 `json:"node_id,omitempty"`
	PolicyIDs   []string               `json:"policy_ids"`
	Metrics     map[string]interface{} `json:"metrics,omitempty"`
	Constraints []Constraint           `json:"constraints,omitempty"`
}

// Constraint represents a constraint for orchestration decisions
type Constraint struct {
	Type        ConstraintType         `json:"type"`
	Enforcement ConstraintEnforcement  `json:"enforcement"`
	Parameters  map[string]interface{} `json:"parameters"`
	Weight      float64                `json:"weight,omitempty"`
}

// ConstraintType defines the type of constraint
type ConstraintType string

const (
	ConstraintTypeAffinity       ConstraintType = "affinity"
	ConstraintTypeAntiAffinity   ConstraintType = "anti_affinity"
	ConstraintTypeResourceLimit  ConstraintType = "resource_limit"
	ConstraintTypeNetworkLatency ConstraintType = "network_latency"
	ConstraintTypeCompliance     ConstraintType = "compliance"
	ConstraintTypeCost           ConstraintType = "cost"
	ConstraintTypeAvailability   ConstraintType = "availability"
)

// ConstraintEnforcement defines how strictly a constraint should be enforced
type ConstraintEnforcement string

const (
	EnforcementHard       ConstraintEnforcement = "hard"       // Must be satisfied
	EnforcementSoft       ConstraintEnforcement = "soft"       // Preferred but can be violated
	EnforcementPreferred  ConstraintEnforcement = "preferred"  // Weighted preference
)

// OrchestrationDecision represents a decision made by the orchestration engine
type OrchestrationDecision struct {
	ID            string                 `json:"id"`
	DecisionType  DecisionType           `json:"decision_type"`
	Context       DecisionContext        `json:"context"`
	Recommendation string                 `json:"recommendation"`
	Score         float64                `json:"score"`
	Confidence    float64                `json:"confidence"`
	Explanation   string                 `json:"explanation"`
	Alternatives  []Alternative          `json:"alternatives,omitempty"`
	Actions       []DecisionAction       `json:"actions"`
	Timestamp     time.Time              `json:"timestamp"`
	ExecutedAt    *time.Time             `json:"executed_at,omitempty"`
	Status        DecisionStatus         `json:"status"`
	Metadata      map[string]interface{} `json:"metadata,omitempty"`
}

// DecisionType defines the type of orchestration decision
type DecisionType string

const (
	DecisionTypePlacement  DecisionType = "placement"
	DecisionTypeScaling    DecisionType = "scaling"
	DecisionTypeHealing    DecisionType = "healing"
	DecisionTypeMigration  DecisionType = "migration"
	DecisionTypeOptimization DecisionType = "optimization"
)

// DecisionStatus represents the status of a decision
type DecisionStatus string

const (
	DecisionStatusPending   DecisionStatus = "pending"
	DecisionStatusExecuted  DecisionStatus = "executed"
	DecisionStatusFailed    DecisionStatus = "failed"
	DecisionStatusCancelled DecisionStatus = "cancelled"
)

// Alternative represents an alternative decision option
type Alternative struct {
	Description string                 `json:"description"`
	Score       float64                `json:"score"`
	Confidence  float64                `json:"confidence"`
	Tradeoffs   []string               `json:"tradeoffs,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// DecisionAction represents an action to be taken as part of a decision
type DecisionAction struct {
	Type       ActionType             `json:"type"`
	Target     string                 `json:"target"`
	Parameters map[string]interface{} `json:"parameters"`
	Timeout    time.Duration          `json:"timeout,omitempty"`
	DependsOn  []string               `json:"depends_on,omitempty"`
}

// MetricsSnapshot represents a point-in-time view of system metrics
type MetricsSnapshot struct {
	Timestamp    time.Time              `json:"timestamp"`
	VMMetrics    map[string]VMMetrics   `json:"vm_metrics"`
	NodeMetrics  map[string]NodeMetrics `json:"node_metrics"`
	ClusterMetrics ClusterMetrics       `json:"cluster_metrics"`
}

// VMMetrics represents metrics for a single VM
type VMMetrics struct {
	VMID           string  `json:"vm_id"`
	CPUUtilization float64 `json:"cpu_utilization"`
	MemoryUsage    int64   `json:"memory_usage"`
	MemoryLimit    int64   `json:"memory_limit"`
	NetworkInBytes int64   `json:"network_in_bytes"`
	NetworkOutBytes int64  `json:"network_out_bytes"`
	DiskReadIOPS   int64   `json:"disk_read_iops"`
	DiskWriteIOPS  int64   `json:"disk_write_iops"`
	State          string  `json:"state"`
	Healthy        bool    `json:"healthy"`
}

// NodeMetrics represents metrics for a single node
type NodeMetrics struct {
	NodeID               string  `json:"node_id"`
	CPUUtilization       float64 `json:"cpu_utilization"`
	MemoryUtilization    float64 `json:"memory_utilization"`
	DiskUtilization      float64 `json:"disk_utilization"`
	NetworkUtilization   float64 `json:"network_utilization"`
	ActiveVMs            int     `json:"active_vms"`
	AvailableCPU         int     `json:"available_cpu"`
	AvailableMemoryMB    int64   `json:"available_memory_mb"`
	LoadAverage          float64 `json:"load_average"`
	Healthy              bool    `json:"healthy"`
}

// ClusterMetrics represents overall cluster metrics
type ClusterMetrics struct {
	TotalNodes           int     `json:"total_nodes"`
	HealthyNodes         int     `json:"healthy_nodes"`
	TotalVMs             int     `json:"total_vms"`
	RunningVMs           int     `json:"running_vms"`
	OverallCPUUtilization float64 `json:"overall_cpu_utilization"`
	OverallMemoryUtilization float64 `json:"overall_memory_utilization"`
	ResourceEfficiency   float64 `json:"resource_efficiency"`
}