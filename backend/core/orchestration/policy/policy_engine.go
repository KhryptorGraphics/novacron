package policy

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
)

// DefaultPolicyEngine implements the PolicyEngine interface
type DefaultPolicyEngine struct {
	mu          sync.RWMutex
	logger      *logrus.Logger
	eventBus    events.EventBus
	parser      PolicyParser
	evaluator   PolicyEvaluator
	
	// State
	policies    map[string]*OrchestrationPolicy // policyID -> policy
	
	// Metrics
	evaluationsCount uint64
	lastEvaluationTime time.Time
	
	// Context for lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
}

// NewDefaultPolicyEngine creates a new policy engine
func NewDefaultPolicyEngine(logger *logrus.Logger, eventBus events.EventBus) *DefaultPolicyEngine {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &DefaultPolicyEngine{
		logger:    logger,
		eventBus:  eventBus,
		parser:    NewDefaultPolicyParser(),
		evaluator: NewDefaultPolicyEvaluator(logger),
		policies:  make(map[string]*OrchestrationPolicy),
		ctx:       ctx,
		cancel:    cancel,
	}
}

// CreatePolicy creates a new orchestration policy
func (pe *DefaultPolicyEngine) CreatePolicy(policy *OrchestrationPolicy) error {
	if policy == nil {
		return fmt.Errorf("policy cannot be nil")
	}

	// Generate ID if not provided
	if policy.ID == "" {
		policy.ID = uuid.New().String()
	}

	// Validate policy
	validationResult, err := pe.ValidatePolicy(policy)
	if err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}

	if !validationResult.Valid {
		return fmt.Errorf("policy validation failed: %v", validationResult.Errors)
	}

	pe.mu.Lock()
	defer pe.mu.Unlock()

	// Check if policy with same ID already exists
	if _, exists := pe.policies[policy.ID]; exists {
		return fmt.Errorf("policy with ID %s already exists", policy.ID)
	}

	// Set timestamps
	now := time.Now()
	policy.CreatedAt = now
	policy.UpdatedAt = now

	// Store policy
	pe.policies[policy.ID] = policy

	pe.logger.WithFields(logrus.Fields{
		"policy_id":   policy.ID,
		"policy_name": policy.Name,
		"namespace":   policy.Namespace,
		"rules_count": len(policy.Rules),
		"enabled":     policy.Enabled,
	}).Info("Policy created")

	// Publish policy creation event
	if err := pe.publishEvent(EventTypePolicyCreated, policy.ID, "", nil, nil, nil); err != nil {
		pe.logger.WithError(err).Error("Failed to publish policy creation event")
	}

	return nil
}

// UpdatePolicy updates an existing orchestration policy
func (pe *DefaultPolicyEngine) UpdatePolicy(policyID string, policy *OrchestrationPolicy) error {
	if policyID == "" {
		return fmt.Errorf("policy ID cannot be empty")
	}

	if policy == nil {
		return fmt.Errorf("policy cannot be nil")
	}

	// Validate policy
	validationResult, err := pe.ValidatePolicy(policy)
	if err != nil {
		return fmt.Errorf("policy validation failed: %w", err)
	}

	if !validationResult.Valid {
		return fmt.Errorf("policy validation failed: %v", validationResult.Errors)
	}

	pe.mu.Lock()
	defer pe.mu.Unlock()

	// Check if policy exists
	existingPolicy, exists := pe.policies[policyID]
	if !exists {
		return fmt.Errorf("policy with ID %s not found", policyID)
	}

	// Preserve creation time and ID
	policy.ID = policyID
	policy.CreatedAt = existingPolicy.CreatedAt
	policy.UpdatedAt = time.Now()

	// Update policy
	pe.policies[policyID] = policy

	pe.logger.WithFields(logrus.Fields{
		"policy_id":   policy.ID,
		"policy_name": policy.Name,
		"namespace":   policy.Namespace,
		"rules_count": len(policy.Rules),
		"enabled":     policy.Enabled,
	}).Info("Policy updated")

	// Publish policy update event
	if err := pe.publishEvent(EventTypePolicyUpdated, policy.ID, "", nil, nil, nil); err != nil {
		pe.logger.WithError(err).Error("Failed to publish policy update event")
	}

	return nil
}

// DeletePolicy deletes an orchestration policy
func (pe *DefaultPolicyEngine) DeletePolicy(policyID string) error {
	if policyID == "" {
		return fmt.Errorf("policy ID cannot be empty")
	}

	pe.mu.Lock()
	defer pe.mu.Unlock()

	// Check if policy exists
	policy, exists := pe.policies[policyID]
	if !exists {
		return fmt.Errorf("policy with ID %s not found", policyID)
	}

	// Delete policy
	delete(pe.policies, policyID)

	pe.logger.WithFields(logrus.Fields{
		"policy_id":   policyID,
		"policy_name": policy.Name,
	}).Info("Policy deleted")

	// Publish policy deletion event
	if err := pe.publishEvent(EventTypePolicyDeleted, policyID, "", nil, nil, nil); err != nil {
		pe.logger.WithError(err).Error("Failed to publish policy deletion event")
	}

	return nil
}

// GetPolicy gets an orchestration policy by ID
func (pe *DefaultPolicyEngine) GetPolicy(policyID string) (*OrchestrationPolicy, error) {
	if policyID == "" {
		return nil, fmt.Errorf("policy ID cannot be empty")
	}

	pe.mu.RLock()
	defer pe.mu.RUnlock()

	policy, exists := pe.policies[policyID]
	if !exists {
		return nil, fmt.Errorf("policy with ID %s not found", policyID)
	}

	// Return a copy to avoid race conditions
	policyCopy := *policy
	policyCopy.Rules = make([]*PolicyRule, len(policy.Rules))
	for i, rule := range policy.Rules {
		ruleCopy := *rule
		policyCopy.Rules[i] = &ruleCopy
	}

	return &policyCopy, nil
}

// ListPolicies lists all orchestration policies
func (pe *DefaultPolicyEngine) ListPolicies(filter *PolicyFilter) ([]*OrchestrationPolicy, error) {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	var policies []*OrchestrationPolicy

	// Collect all policies
	for _, policy := range pe.policies {
		// Apply filters
		if filter != nil && !pe.policyMatchesFilter(policy, filter) {
			continue
		}

		// Create copy
		policyCopy := *policy
		policies = append(policies, &policyCopy)
	}

	// Sort policies
	pe.sortPolicies(policies, filter)

	// Apply limit and offset
	if filter != nil {
		policies = pe.applyPagination(policies, filter)
	}

	return policies, nil
}

// EvaluatePolicy evaluates a policy against the given context
func (pe *DefaultPolicyEngine) EvaluatePolicy(policyID string, context *PolicyEvaluationContext) (*PolicyEvaluationResult, error) {
	if policyID == "" {
		return nil, fmt.Errorf("policy ID cannot be empty")
	}

	if context == nil {
		return nil, fmt.Errorf("evaluation context cannot be nil")
	}

	pe.mu.RLock()
	policy, exists := pe.policies[policyID]
	pe.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("policy with ID %s not found", policyID)
	}

	return pe.evaluateSinglePolicy(policy, context)
}

// EvaluateAllPolicies evaluates all applicable policies against the given context
func (pe *DefaultPolicyEngine) EvaluateAllPolicies(context *PolicyEvaluationContext) ([]*PolicyEvaluationResult, error) {
	if context == nil {
		return nil, fmt.Errorf("evaluation context cannot be nil")
	}

	pe.mu.RLock()
	policies := make([]*OrchestrationPolicy, 0, len(pe.policies))
	for _, policy := range pe.policies {
		if policy.Enabled && pe.policyApplicable(policy, context) {
			policyCopy := *policy
			policies = append(policies, &policyCopy)
		}
	}
	pe.mu.RUnlock()

	// Sort policies by priority (highest first)
	sort.Slice(policies, func(i, j int) bool {
		return policies[i].Priority > policies[j].Priority
	})

	var results []*PolicyEvaluationResult

	// Evaluate each applicable policy
	for _, policy := range policies {
		result, err := pe.evaluateSinglePolicy(policy, context)
		if err != nil {
			pe.logger.WithError(err).WithField("policy_id", policy.ID).Error("Policy evaluation failed")
			continue
		}

		results = append(results, result)

		// Publish evaluation event if policy matched
		if result.Matched {
			if err := pe.publishEvent(EventTypePolicyEvaluated, policy.ID, "", context, result, nil); err != nil {
				pe.logger.WithError(err).Error("Failed to publish policy evaluation event")
			}
		}
	}

	// Update metrics
	pe.mu.Lock()
	pe.evaluationsCount++
	pe.lastEvaluationTime = time.Now()
	pe.mu.Unlock()

	return results, nil
}

// ValidatePolicy validates a policy's syntax and rules
func (pe *DefaultPolicyEngine) ValidatePolicy(policy *OrchestrationPolicy) (*PolicyValidationResult, error) {
	result := &PolicyValidationResult{
		Valid:    true,
		Errors:   []ValidationError{},
		Warnings: []ValidationWarning{},
	}

	// Basic structure validation
	if policy.Name == "" {
		result.Valid = false
		result.Errors = append(result.Errors, ValidationError{
			Field:   "name",
			Code:    "required",
			Message: "Policy name is required",
		})
	}

	if len(policy.Rules) == 0 {
		result.Warnings = append(result.Warnings, ValidationWarning{
			Field:   "rules",
			Code:    "empty",
			Message: "Policy has no rules",
		})
	}

	// Validate each rule
	for i, rule := range policy.Rules {
		if err := pe.validateRule(rule, result, i); err != nil {
			pe.logger.WithError(err).WithField("rule_id", rule.ID).Error("Rule validation failed")
		}
	}

	// Validate DSL if provided
	if policy.DSL != "" {
		syntaxResult, err := pe.parser.ValidateSyntax(policy.DSL)
		if err != nil {
			result.Valid = false
			result.Errors = append(result.Errors, ValidationError{
				Field:   "dsl",
				Code:    "syntax_validation_failed",
				Message: fmt.Sprintf("DSL syntax validation failed: %s", err.Error()),
			})
		} else if !syntaxResult.Valid {
			result.Valid = false
			for _, syntaxError := range syntaxResult.Errors {
				result.Errors = append(result.Errors, ValidationError{
					Field:   "dsl",
					Code:    "syntax_error",
					Message: syntaxError.Message,
					Line:    syntaxError.Line,
					Column:  syntaxError.Column,
				})
			}
		}
	}

	return result, nil
}

// GetStatus returns the current status of the policy engine
func (pe *DefaultPolicyEngine) GetStatus() *PolicyEngineStatus {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	enabledPolicies := 0
	totalRules := 0

	for _, policy := range pe.policies {
		if policy.Enabled {
			enabledPolicies++
		}
		totalRules += len(policy.Rules)
	}

	return &PolicyEngineStatus{
		TotalPolicies:      len(pe.policies),
		EnabledPolicies:    enabledPolicies,
		TotalRules:         totalRules,
		EvaluationsCount:   pe.evaluationsCount,
		LastEvaluationTime: pe.lastEvaluationTime,
	}
}

// PolicyEngineStatus represents the status of the policy engine
type PolicyEngineStatus struct {
	TotalPolicies      int       `json:"total_policies"`
	EnabledPolicies    int       `json:"enabled_policies"`
	TotalRules         int       `json:"total_rules"`
	EvaluationsCount   uint64    `json:"evaluations_count"`
	LastEvaluationTime time.Time `json:"last_evaluation_time"`
}

// Private methods

func (pe *DefaultPolicyEngine) evaluateSinglePolicy(policy *OrchestrationPolicy, context *PolicyEvaluationContext) (*PolicyEvaluationResult, error) {
	startTime := time.Now()

	result := &PolicyEvaluationResult{
		PolicyID:    policy.ID,
		PolicyName:  policy.Name,
		Matched:     false,
		Score:       0.0,
		Confidence:  1.0,
		RuleResults: []*RuleEvaluationResult{},
		Actions:     []*RecommendedAction{},
		EvaluatedAt: startTime,
	}

	pe.logger.WithFields(logrus.Fields{
		"policy_id":   policy.ID,
		"policy_name": policy.Name,
		"context":     context.RequestID,
	}).Debug("Evaluating policy")

	// Check if policy is enabled
	if !policy.Enabled {
		result.Explanation = "Policy is disabled"
		result.Duration = time.Since(startTime)
		return result, nil
	}

	// Evaluate all rules
	matchedRules := 0
	totalScore := 0.0

	for _, rule := range policy.Rules {
		ruleResult, err := pe.evaluator.EvaluateRule(rule, context)
		if err != nil {
			pe.logger.WithError(err).WithField("rule_id", rule.ID).Error("Rule evaluation failed")
			continue
		}

		result.RuleResults = append(result.RuleResults, ruleResult)

		if ruleResult.Matched {
			matchedRules++
			totalScore += ruleResult.Score

			// Add rule actions to policy actions
			for _, action := range ruleResult.Actions {
				result.Actions = append(result.Actions, action)
			}

			// Publish rule matched event
			if err := pe.publishEvent(EventTypeRuleMatched, policy.ID, rule.ID, context, result, nil); err != nil {
				pe.logger.WithError(err).Error("Failed to publish rule matched event")
			}
		}
	}

	// Determine if policy matches
	if len(policy.Rules) > 0 {
		result.Score = totalScore / float64(len(policy.Rules))
		result.Matched = matchedRules > 0 // Policy matches if any rule matches
	}

	// Sort actions by priority
	sort.Slice(result.Actions, func(i, j int) bool {
		return result.Actions[i].Priority > result.Actions[j].Priority
	})

	// Generate explanation
	if result.Matched {
		result.Explanation = fmt.Sprintf("Policy '%s' matched with %d/%d rules (score: %.2f)", 
			policy.Name, matchedRules, len(policy.Rules), result.Score)
	} else {
		result.Explanation = fmt.Sprintf("Policy '%s' did not match (0/%d rules)", 
			policy.Name, len(policy.Rules))
	}

	result.Duration = time.Since(startTime)

	pe.logger.WithFields(logrus.Fields{
		"policy_id":     policy.ID,
		"matched":       result.Matched,
		"score":         result.Score,
		"matched_rules": matchedRules,
		"total_rules":   len(policy.Rules),
		"actions_count": len(result.Actions),
		"duration":      result.Duration,
	}).Debug("Policy evaluation completed")

	return result, nil
}

func (pe *DefaultPolicyEngine) policyApplicable(policy *OrchestrationPolicy, context *PolicyEvaluationContext) bool {
	if policy.Selector == nil {
		return true // No selector means applies to all
	}

	selector := policy.Selector

	// Check namespace
	if len(selector.Namespaces) > 0 {
		found := false
		for _, ns := range selector.Namespaces {
			if ns == context.Namespace {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check resource types
	if len(selector.ResourceTypes) > 0 {
		found := false
		for _, rt := range selector.ResourceTypes {
			if rt == ResourceTypeAll || rt == context.ResourceType {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	// Check match labels
	if len(selector.MatchLabels) > 0 {
		if context.Labels == nil {
			return false
		}

		for key, expectedValue := range selector.MatchLabels {
			if actualValue, exists := context.Labels[key]; !exists || actualValue != expectedValue {
				if !selector.MatchAny {
					return false
				}
			} else if selector.MatchAny {
				return true
			}
		}

		if selector.MatchAny && len(selector.MatchLabels) > 0 {
			// MatchAny was true but no labels matched
			return false
		}
	}

	// Check match tags
	if len(selector.MatchTags) > 0 {
		if context.Tags == nil {
			return false
		}

		for key, expectedValue := range selector.MatchTags {
			if actualValue, exists := context.Tags[key]; !exists || actualValue != expectedValue {
				if !selector.MatchAny {
					return false
				}
			} else if selector.MatchAny {
				return true
			}
		}

		if selector.MatchAny && len(selector.MatchTags) > 0 {
			return false
		}
	}

	// Check CEL expression
	if selector.CELExpression != "" {
		result, err := pe.evaluator.EvaluateExpression(selector.CELExpression, context)
		if err != nil {
			pe.logger.WithError(err).Error("CEL expression evaluation failed")
			return false
		}

		if boolResult, ok := result.(bool); ok {
			return boolResult
		}

		return false
	}

	return true
}

func (pe *DefaultPolicyEngine) policyMatchesFilter(policy *OrchestrationPolicy, filter *PolicyFilter) bool {
	// Check namespace
	if filter.Namespace != "" && policy.Namespace != filter.Namespace {
		return false
	}

	// Check enabled status
	if filter.Enabled != nil && policy.Enabled != *filter.Enabled {
		return false
	}

	// Check created by
	if filter.CreatedBy != "" && policy.CreatedBy != filter.CreatedBy {
		return false
	}

	// Check created after
	if filter.CreatedAfter != nil && policy.CreatedAt.Before(*filter.CreatedAfter) {
		return false
	}

	// Check created before
	if filter.CreatedBefore != nil && policy.CreatedAt.After(*filter.CreatedBefore) {
		return false
	}

	// Check tags
	if len(filter.Tags) > 0 {
		policyTags := policy.Tags
		for _, requiredTag := range filter.Tags {
			found := false
			for _, policyTag := range policyTags {
				if policyTag == requiredTag {
					found = true
					break
				}
			}
			if !found {
				return false
			}
		}
	}

	// Check rule type
	if filter.Type != "" {
		hasType := false
		for _, rule := range policy.Rules {
			if rule.Type == filter.Type {
				hasType = true
				break
			}
		}
		if !hasType {
			return false
		}
	}

	return true
}

func (pe *DefaultPolicyEngine) sortPolicies(policies []*OrchestrationPolicy, filter *PolicyFilter) {
	if filter == nil || filter.SortBy == "" {
		// Default sort by priority descending, then by name
		sort.Slice(policies, func(i, j int) bool {
			if policies[i].Priority != policies[j].Priority {
				return policies[i].Priority > policies[j].Priority
			}
			return policies[i].Name < policies[j].Name
		})
		return
	}

	sortAsc := filter.SortOrder != "desc"

	switch filter.SortBy {
	case "name":
		sort.Slice(policies, func(i, j int) bool {
			if sortAsc {
				return policies[i].Name < policies[j].Name
			}
			return policies[i].Name > policies[j].Name
		})
	case "priority":
		sort.Slice(policies, func(i, j int) bool {
			if sortAsc {
				return policies[i].Priority < policies[j].Priority
			}
			return policies[i].Priority > policies[j].Priority
		})
	case "created_at":
		sort.Slice(policies, func(i, j int) bool {
			if sortAsc {
				return policies[i].CreatedAt.Before(policies[j].CreatedAt)
			}
			return policies[i].CreatedAt.After(policies[j].CreatedAt)
		})
	case "updated_at":
		sort.Slice(policies, func(i, j int) bool {
			if sortAsc {
				return policies[i].UpdatedAt.Before(policies[j].UpdatedAt)
			}
			return policies[i].UpdatedAt.After(policies[j].UpdatedAt)
		})
	}
}

func (pe *DefaultPolicyEngine) applyPagination(policies []*OrchestrationPolicy, filter *PolicyFilter) []*OrchestrationPolicy {
	if filter.Offset > 0 {
		if filter.Offset >= len(policies) {
			return []*OrchestrationPolicy{}
		}
		policies = policies[filter.Offset:]
	}

	if filter.Limit > 0 && filter.Limit < len(policies) {
		policies = policies[:filter.Limit]
	}

	return policies
}

func (pe *DefaultPolicyEngine) validateRule(rule *PolicyRule, result *PolicyValidationResult, ruleIndex int) error {
	if rule.Name == "" {
		result.Valid = false
		result.Errors = append(result.Errors, ValidationError{
			Field:   fmt.Sprintf("rules[%d].name", ruleIndex),
			Code:    "required",
			Message: "Rule name is required",
		})
	}

	if len(rule.Actions) == 0 {
		result.Warnings = append(result.Warnings, ValidationWarning{
			Field:   fmt.Sprintf("rules[%d].actions", ruleIndex),
			Code:    "empty",
			Message: "Rule has no actions",
		})
	}

	// Validate conditions
	for i, condition := range rule.Conditions {
		if condition.Field == "" && condition.CELExpression == "" {
			result.Valid = false
			result.Errors = append(result.Errors, ValidationError{
				Field:   fmt.Sprintf("rules[%d].conditions[%d]", ruleIndex, i),
				Code:    "invalid",
				Message: "Condition must have either field or CEL expression",
			})
		}
	}

	return nil
}

func (pe *DefaultPolicyEngine) publishEvent(eventType EventType, policyID, ruleID string, context *PolicyEvaluationContext, result *PolicyEvaluationResult, action *RecommendedAction) error {
	if pe.eventBus == nil {
		return nil
	}

	event := &events.OrchestrationEvent{
		Type:      events.EventType(eventType),
		Source:    "policy-engine",
		Target:    policyID,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"event_type": eventType,
			"policy_id":  policyID,
		},
		Priority: events.PriorityNormal,
	}

	if ruleID != "" {
		event.Data["rule_id"] = ruleID
	}
	if context != nil {
		event.Data["context"] = context
	}
	if result != nil {
		event.Data["result"] = result
	}
	if action != nil {
		event.Data["action"] = action
	}

	return pe.eventBus.Publish(pe.ctx, event)
}