package policy

import (
	"time"
)

// PolicyEngine defines the interface for policy management and evaluation
type PolicyEngine interface {
	// CreatePolicy creates a new orchestration policy
	CreatePolicy(policy *OrchestrationPolicy) error
	
	// UpdatePolicy updates an existing orchestration policy
	UpdatePolicy(policyID string, policy *OrchestrationPolicy) error
	
	// DeletePolicy deletes an orchestration policy
	DeletePolicy(policyID string) error
	
	// GetPolicy gets an orchestration policy by ID
	GetPolicy(policyID string) (*OrchestrationPolicy, error)
	
	// ListPolicies lists all orchestration policies
	ListPolicies(filter *PolicyFilter) ([]*OrchestrationPolicy, error)
	
	// EvaluatePolicy evaluates a policy against the given context
	EvaluatePolicy(policyID string, context *PolicyEvaluationContext) (*PolicyEvaluationResult, error)
	
	// EvaluateAllPolicies evaluates all applicable policies against the given context
	EvaluateAllPolicies(context *PolicyEvaluationContext) ([]*PolicyEvaluationResult, error)
	
	// ValidatePolicy validates a policy's syntax and rules
	ValidatePolicy(policy *OrchestrationPolicy) (*PolicyValidationResult, error)
}

// PolicyParser defines the interface for parsing policy DSL
type PolicyParser interface {
	// ParsePolicy parses a policy from DSL string
	ParsePolicy(dsl string) (*OrchestrationPolicy, error)
	
	// ParseRule parses a single rule from DSL string
	ParseRule(dsl string) (*PolicyRule, error)
	
	// ParseCondition parses a condition from DSL string
	ParseCondition(dsl string) (*RuleCondition, error)
	
	// ValidateSyntax validates the syntax of policy DSL
	ValidateSyntax(dsl string) (*SyntaxValidationResult, error)
}

// PolicyEvaluator defines the interface for policy evaluation
type PolicyEvaluator interface {
	// EvaluateRule evaluates a single policy rule
	EvaluateRule(rule *PolicyRule, context *PolicyEvaluationContext) (*RuleEvaluationResult, error)
	
	// EvaluateCondition evaluates a single condition
	EvaluateCondition(condition *RuleCondition, context *PolicyEvaluationContext) (bool, error)
	
	// EvaluateExpression evaluates a CEL expression
	EvaluateExpression(expression string, context *PolicyEvaluationContext) (interface{}, error)
}

// OrchestrationPolicy represents a comprehensive orchestration policy
type OrchestrationPolicy struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	Version         string                 `json:"version"`
	Namespace       string                 `json:"namespace"`
	Enabled         bool                   `json:"enabled"`
	Priority        int                    `json:"priority"`
	Selector        *PolicySelector        `json:"selector"`
	Rules           []*PolicyRule          `json:"rules"`
	DSL             string                 `json:"dsl,omitempty"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	CreatedBy       string                 `json:"created_by"`
	Tags            []string               `json:"tags,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
}

// PolicySelector defines which resources this policy applies to
type PolicySelector struct {
	MatchLabels    map[string]string      `json:"match_labels,omitempty"`
	MatchTags      map[string]string      `json:"match_tags,omitempty"`
	ResourceTypes  []ResourceType         `json:"resource_types,omitempty"`
	Namespaces     []string               `json:"namespaces,omitempty"`
	CELExpression  string                 `json:"cel_expression,omitempty"`
	MatchAny       bool                   `json:"match_any,omitempty"` // OR vs AND logic
}

// ResourceType represents the type of resource the policy applies to
type ResourceType string

const (
	ResourceTypeVM      ResourceType = "vm"
	ResourceTypeNode    ResourceType = "node"
	ResourceTypeService ResourceType = "service"
	ResourceTypeCluster ResourceType = "cluster"
	ResourceTypeAll     ResourceType = "*"
)

// PolicyRule defines a specific rule within a policy
type PolicyRule struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Type        PolicyRuleType         `json:"type"`
	Priority    int                    `json:"priority"`
	Enabled     bool                   `json:"enabled"`
	Conditions  []*RuleCondition       `json:"conditions,omitempty"`
	Actions     []*RuleAction          `json:"actions"`
	Schedule    *RuleSchedule          `json:"schedule,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
	DSL         string                 `json:"dsl,omitempty"`
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
	RuleTypeResource    PolicyRuleType = "resource"
	RuleTypeNetwork     PolicyRuleType = "network"
	RuleTypeScheduling  PolicyRuleType = "scheduling"
	RuleTypeCustom      PolicyRuleType = "custom"
)

// RuleCondition defines when a rule should be applied
type RuleCondition struct {
	ID             string                 `json:"id"`
	Type           ConditionType          `json:"type"`
	Field          string                 `json:"field"`
	Operator       ConditionOperator      `json:"operator"`
	Value          interface{}            `json:"value"`
	Values         []interface{}          `json:"values,omitempty"`
	CELExpression  string                 `json:"cel_expression,omitempty"`
	TimeWindow     *TimeWindow            `json:"time_window,omitempty"`
	Parameters     map[string]interface{} `json:"parameters,omitempty"`
}

// ConditionType defines the type of condition
type ConditionType string

const (
	ConditionTypeMetric      ConditionType = "metric"
	ConditionTypeResource    ConditionType = "resource"
	ConditionTypeTime        ConditionType = "time"
	ConditionTypeEvent       ConditionType = "event"
	ConditionTypeLabel       ConditionType = "label"
	ConditionTypeTag         ConditionType = "tag"
	ConditionTypeState       ConditionType = "state"
	ConditionTypeCustom      ConditionType = "custom"
	ConditionTypeCEL         ConditionType = "cel"
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
	OperatorMatches             ConditionOperator = "matches"       // Regex
	OperatorNotMatches          ConditionOperator = "not_matches"   // Regex
	OperatorExists              ConditionOperator = "exists"
	OperatorNotExists           ConditionOperator = "not_exists"
)

// TimeWindow defines a time-based condition window
type TimeWindow struct {
	Duration   time.Duration `json:"duration"`
	StartTime  *time.Time    `json:"start_time,omitempty"`
	EndTime    *time.Time    `json:"end_time,omitempty"`
	DaysOfWeek []int         `json:"days_of_week,omitempty"` // 0=Sunday, 1=Monday, etc.
	HourRange  *HourRange    `json:"hour_range,omitempty"`
}

// HourRange defines a range of hours
type HourRange struct {
	Start int `json:"start"` // 0-23
	End   int `json:"end"`   // 0-23
}

// RuleAction defines what action should be taken when rule conditions are met
type RuleAction struct {
	ID          string                 `json:"id"`
	Type        ActionType             `json:"type"`
	Target      string                 `json:"target,omitempty"`
	Parameters  map[string]interface{} `json:"parameters"`
	Timeout     time.Duration          `json:"timeout,omitempty"`
	Retries     int                    `json:"retries,omitempty"`
	OnFailure   string                 `json:"on_failure,omitempty"` // Action to take on failure
	Async       bool                   `json:"async,omitempty"`
}

// ActionType defines the type of action to take
type ActionType string

const (
	ActionTypeScale       ActionType = "scale"
	ActionTypeMigrate     ActionType = "migrate"
	ActionTypeRestart     ActionType = "restart"
	ActionTypeStop        ActionType = "stop"
	ActionTypeStart       ActionType = "start"
	ActionTypeAlert       ActionType = "alert"
	ActionTypeWebhook     ActionType = "webhook"
	ActionTypeSchedule    ActionType = "schedule"
	ActionTypeQuarantine  ActionType = "quarantine"
	ActionTypeOptimize    ActionType = "optimize"
	ActionTypeLog         ActionType = "log"
	ActionTypeEmail       ActionType = "email"
	ActionTypeSlack       ActionType = "slack"
	ActionTypeCustom      ActionType = "custom"
)

// RuleSchedule defines when a rule should be evaluated
type RuleSchedule struct {
	CronExpression string     `json:"cron_expression,omitempty"`
	Interval       *time.Duration `json:"interval,omitempty"`
	StartTime      *time.Time `json:"start_time,omitempty"`
	EndTime        *time.Time `json:"end_time,omitempty"`
	Timezone       string     `json:"timezone,omitempty"`
	Enabled        bool       `json:"enabled"`
}

// PolicyEvaluationContext provides context for policy evaluation
type PolicyEvaluationContext struct {
	RequestID     string                 `json:"request_id"`
	Timestamp     time.Time              `json:"timestamp"`
	ResourceType  ResourceType           `json:"resource_type"`
	ResourceID    string                 `json:"resource_id"`
	Labels        map[string]string      `json:"labels,omitempty"`
	Tags          map[string]string      `json:"tags,omitempty"`
	Metrics       map[string]float64     `json:"metrics,omitempty"`
	Attributes    map[string]interface{} `json:"attributes,omitempty"`
	EventType     string                 `json:"event_type,omitempty"`
	EventData     map[string]interface{} `json:"event_data,omitempty"`
	Namespace     string                 `json:"namespace,omitempty"`
	User          string                 `json:"user,omitempty"`
	RequestSource string                 `json:"request_source,omitempty"`
}

// PolicyEvaluationResult represents the result of policy evaluation
type PolicyEvaluationResult struct {
	PolicyID      string                    `json:"policy_id"`
	PolicyName    string                    `json:"policy_name"`
	Matched       bool                      `json:"matched"`
	Score         float64                   `json:"score"`
	Confidence    float64                   `json:"confidence"`
	RuleResults   []*RuleEvaluationResult   `json:"rule_results"`
	Actions       []*RecommendedAction      `json:"actions,omitempty"`
	Explanation   string                    `json:"explanation"`
	EvaluatedAt   time.Time                 `json:"evaluated_at"`
	Duration      time.Duration             `json:"duration"`
	Metadata      map[string]interface{}    `json:"metadata,omitempty"`
}

// RuleEvaluationResult represents the result of a single rule evaluation
type RuleEvaluationResult struct {
	RuleID            string                    `json:"rule_id"`
	RuleName          string                    `json:"rule_name"`
	RuleType          PolicyRuleType            `json:"rule_type"`
	Matched           bool                      `json:"matched"`
	Score             float64                   `json:"score"`
	ConditionResults  []*ConditionResult        `json:"condition_results"`
	Actions           []*RecommendedAction      `json:"actions,omitempty"`
	Explanation       string                    `json:"explanation"`
	EvaluatedAt       time.Time                 `json:"evaluated_at"`
	Duration          time.Duration             `json:"duration"`
}

// ConditionResult represents the result of a condition evaluation
type ConditionResult struct {
	ConditionID string      `json:"condition_id"`
	Type        ConditionType `json:"type"`
	Field       string      `json:"field"`
	Operator    ConditionOperator `json:"operator"`
	Expected    interface{} `json:"expected"`
	Actual      interface{} `json:"actual"`
	Matched     bool        `json:"matched"`
	Explanation string      `json:"explanation"`
}

// RecommendedAction represents an action recommended by policy evaluation
type RecommendedAction struct {
	Type        ActionType             `json:"type"`
	Priority    int                    `json:"priority"`
	Target      string                 `json:"target"`
	Parameters  map[string]interface{} `json:"parameters"`
	Confidence  float64                `json:"confidence"`
	Explanation string                 `json:"explanation"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// PolicyValidationResult represents the result of policy validation
type PolicyValidationResult struct {
	Valid     bool                    `json:"valid"`
	Errors    []ValidationError       `json:"errors,omitempty"`
	Warnings  []ValidationWarning     `json:"warnings,omitempty"`
	Metadata  map[string]interface{}  `json:"metadata,omitempty"`
}

// ValidationError represents a policy validation error
type ValidationError struct {
	Field   string `json:"field"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Line    int    `json:"line,omitempty"`
	Column  int    `json:"column,omitempty"`
}

// ValidationWarning represents a policy validation warning
type ValidationWarning struct {
	Field   string `json:"field"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Line    int    `json:"line,omitempty"`
	Column  int    `json:"column,omitempty"`
}

// SyntaxValidationResult represents the result of DSL syntax validation
type SyntaxValidationResult struct {
	Valid    bool           `json:"valid"`
	Errors   []SyntaxError  `json:"errors,omitempty"`
	Tokens   []Token        `json:"tokens,omitempty"`
	AST      interface{}    `json:"ast,omitempty"`
}

// SyntaxError represents a DSL syntax error
type SyntaxError struct {
	Message string `json:"message"`
	Line    int    `json:"line"`
	Column  int    `json:"column"`
	Token   string `json:"token,omitempty"`
}

// Token represents a parsed DSL token
type Token struct {
	Type     TokenType `json:"type"`
	Value    string    `json:"value"`
	Line     int       `json:"line"`
	Column   int       `json:"column"`
}

// TokenType represents the type of DSL token
type TokenType string

const (
	TokenTypeKeyword    TokenType = "keyword"
	TokenTypeIdentifier TokenType = "identifier"
	TokenTypeString     TokenType = "string"
	TokenTypeNumber     TokenType = "number"
	TokenTypeOperator   TokenType = "operator"
	TokenTypeDelimiter  TokenType = "delimiter"
	TokenTypeComment    TokenType = "comment"
)

// PolicyFilter defines filters for listing policies
type PolicyFilter struct {
	Namespace    string            `json:"namespace,omitempty"`
	Tags         []string          `json:"tags,omitempty"`
	Labels       map[string]string `json:"labels,omitempty"`
	Type         PolicyRuleType    `json:"type,omitempty"`
	Enabled      *bool             `json:"enabled,omitempty"`
	CreatedBy    string            `json:"created_by,omitempty"`
	CreatedAfter *time.Time        `json:"created_after,omitempty"`
	CreatedBefore *time.Time       `json:"created_before,omitempty"`
	Limit        int               `json:"limit,omitempty"`
	Offset       int               `json:"offset,omitempty"`
	SortBy       string            `json:"sort_by,omitempty"`
	SortOrder    string            `json:"sort_order,omitempty"`
}

// PolicyEvent represents a policy-related event
type PolicyEvent struct {
	Type         EventType              `json:"type"`
	PolicyID     string                 `json:"policy_id"`
	RuleID       string                 `json:"rule_id,omitempty"`
	ResourceID   string                 `json:"resource_id,omitempty"`
	Context      *PolicyEvaluationContext `json:"context,omitempty"`
	Result       *PolicyEvaluationResult `json:"result,omitempty"`
	Action       *RecommendedAction     `json:"action,omitempty"`
	Timestamp    time.Time              `json:"timestamp"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// EventType represents the type of policy event
type EventType string

const (
	EventTypePolicyCreated      EventType = "policy.created"
	EventTypePolicyUpdated      EventType = "policy.updated"
	EventTypePolicyDeleted      EventType = "policy.deleted"
	EventTypePolicyEvaluated    EventType = "policy.evaluated"
	EventTypeRuleMatched        EventType = "policy.rule_matched"
	EventTypeActionRecommended  EventType = "policy.action_recommended"
	EventTypePolicyViolation    EventType = "policy.violation"
)

// PolicyDSL defines the structure for policy DSL
type PolicyDSL struct {
	Version string         `json:"version"`
	Kind    string         `json:"kind"`
	Spec    *PolicyDSLSpec `json:"spec"`
}

// PolicyDSLSpec defines the specification part of policy DSL
type PolicyDSLSpec struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Selector    map[string]interface{} `json:"selector"`
	Rules       []map[string]interface{} `json:"rules"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
}