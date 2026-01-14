package policy

import (
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// DefaultPolicyEvaluator implements the PolicyEvaluator interface
type DefaultPolicyEvaluator struct {
	logger *logrus.Logger
}

// NewDefaultPolicyEvaluator creates a new policy evaluator
func NewDefaultPolicyEvaluator(logger *logrus.Logger) *DefaultPolicyEvaluator {
	return &DefaultPolicyEvaluator{
		logger: logger,
	}
}

// EvaluateRule evaluates a single policy rule
func (e *DefaultPolicyEvaluator) EvaluateRule(rule *PolicyRule, context *PolicyEvaluationContext) (*RuleEvaluationResult, error) {
	startTime := time.Now()

	result := &RuleEvaluationResult{
		RuleID:           rule.ID,
		RuleName:         rule.Name,
		RuleType:         rule.Type,
		Matched:          true,
		Score:            0.0,
		ConditionResults: []*ConditionResult{},
		Actions:          []*RecommendedAction{},
		EvaluatedAt:      startTime,
	}

	e.logger.WithFields(logrus.Fields{
		"rule_id":   rule.ID,
		"rule_name": rule.Name,
		"rule_type": rule.Type,
		"context":   context.RequestID,
	}).Debug("Evaluating policy rule")

	// Check if rule is enabled
	if !rule.Enabled {
		result.Matched = false
		result.Explanation = "Rule is disabled"
		result.Duration = time.Since(startTime)
		return result, nil
	}

	// Check rule schedule if specified
	if rule.Schedule != nil && !e.isRuleScheduleActive(rule.Schedule) {
		result.Matched = false
		result.Explanation = "Rule schedule is not active"
		result.Duration = time.Since(startTime)
		return result, nil
	}

	// Evaluate all conditions
	conditionMatches := 0
	totalConditions := len(rule.Conditions)

	for _, condition := range rule.Conditions {
		conditionResult, err := e.evaluateConditionWithResult(condition, context)
		if err != nil {
			e.logger.WithError(err).WithField("condition_id", condition.ID).Error("Failed to evaluate condition")
			conditionResult = &ConditionResult{
				ConditionID: condition.ID,
				Type:        condition.Type,
				Field:       condition.Field,
				Operator:    condition.Operator,
				Expected:    condition.Value,
				Matched:     false,
				Explanation: fmt.Sprintf("Evaluation error: %s", err.Error()),
			}
		}

		result.ConditionResults = append(result.ConditionResults, conditionResult)

		if conditionResult.Matched {
			conditionMatches++
		}
	}

	// Determine if rule matches (all conditions must match by default)
	if totalConditions == 0 {
		// No conditions means rule always matches
		result.Matched = true
		result.Score = 1.0
	} else if conditionMatches == totalConditions {
		// All conditions match
		result.Matched = true
		result.Score = 1.0
	} else {
		// Some conditions don't match
		result.Matched = false
		result.Score = float64(conditionMatches) / float64(totalConditions)
	}

	// Generate recommended actions if rule matches
	if result.Matched {
		for _, action := range rule.Actions {
			recommendedAction := &RecommendedAction{
				Type:        action.Type,
				Priority:    rule.Priority,
				Target:      action.Target,
				Parameters:  action.Parameters,
				Confidence:  result.Score,
				Explanation: fmt.Sprintf("Action recommended by rule '%s'", rule.Name),
			}
			result.Actions = append(result.Actions, recommendedAction)
		}

		result.Explanation = fmt.Sprintf("Rule '%s' matched with score %.2f", rule.Name, result.Score)
	} else {
		result.Explanation = fmt.Sprintf("Rule '%s' did not match (%d/%d conditions)", rule.Name, conditionMatches, totalConditions)
	}

	result.Duration = time.Since(startTime)

	e.logger.WithFields(logrus.Fields{
		"rule_id":    rule.ID,
		"matched":    result.Matched,
		"score":      result.Score,
		"conditions": fmt.Sprintf("%d/%d", conditionMatches, totalConditions),
		"duration":   result.Duration,
	}).Debug("Rule evaluation completed")

	return result, nil
}

// EvaluateCondition evaluates a single condition
func (e *DefaultPolicyEvaluator) EvaluateCondition(condition *RuleCondition, context *PolicyEvaluationContext) (bool, error) {
	result, err := e.evaluateConditionWithResult(condition, context)
	if err != nil {
		return false, err
	}
	return result.Matched, nil
}

// EvaluateExpression evaluates a CEL expression
func (e *DefaultPolicyEvaluator) EvaluateExpression(expression string, context *PolicyEvaluationContext) (interface{}, error) {
	// Simplified CEL expression evaluator
	// In a real implementation, you would use the CEL library
	
	// Replace context variables in the expression
	processedExpression := e.replaceContextVariables(expression, context)
	
	// Simple expression evaluation
	return e.evaluateSimpleExpression(processedExpression)
}

// Private methods

func (e *DefaultPolicyEvaluator) evaluateConditionWithResult(condition *RuleCondition, context *PolicyEvaluationContext) (*ConditionResult, error) {
	result := &ConditionResult{
		ConditionID: condition.ID,
		Type:        condition.Type,
		Field:       condition.Field,
		Operator:    condition.Operator,
		Expected:    condition.Value,
		Matched:     false,
	}

	// Handle CEL expressions
	if condition.CELExpression != "" {
		return e.evaluateCELCondition(condition, context)
	}

	// Get actual value from context
	actualValue, err := e.getContextValue(condition.Field, context)
	if err != nil {
		result.Explanation = fmt.Sprintf("Failed to get field value: %s", err.Error())
		return result, nil
	}

	result.Actual = actualValue

	// Evaluate condition based on operator
	matched, explanation, err := e.evaluateOperator(condition.Operator, actualValue, condition.Value, condition.Values)
	if err != nil {
		result.Explanation = fmt.Sprintf("Operator evaluation failed: %s", err.Error())
		return result, err
	}

	result.Matched = matched
	result.Explanation = explanation

	e.logger.WithFields(logrus.Fields{
		"condition_id": condition.ID,
		"field":        condition.Field,
		"operator":     condition.Operator,
		"expected":     condition.Value,
		"actual":       actualValue,
		"matched":      matched,
	}).Debug("Condition evaluated")

	return result, nil
}

func (e *DefaultPolicyEvaluator) evaluateCELCondition(condition *RuleCondition, context *PolicyEvaluationContext) (*ConditionResult, error) {
	result := &ConditionResult{
		ConditionID: condition.ID,
		Type:        condition.Type,
		Field:       condition.Field,
		Operator:    condition.Operator,
		Expected:    true, // CEL expressions should evaluate to true
	}

	// Evaluate CEL expression
	expressionResult, err := e.EvaluateExpression(condition.CELExpression, context)
	if err != nil {
		result.Explanation = fmt.Sprintf("CEL expression evaluation failed: %s", err.Error())
		return result, err
	}

	result.Actual = expressionResult

	// Check if result is boolean true
	if boolResult, ok := expressionResult.(bool); ok {
		result.Matched = boolResult
		if boolResult {
			result.Explanation = "CEL expression evaluated to true"
		} else {
			result.Explanation = "CEL expression evaluated to false"
		}
	} else {
		result.Matched = false
		result.Explanation = fmt.Sprintf("CEL expression returned non-boolean value: %v", expressionResult)
	}

	return result, nil
}

func (e *DefaultPolicyEvaluator) getContextValue(field string, context *PolicyEvaluationContext) (interface{}, error) {
	// Handle dotted field notation (e.g., "labels.env", "metrics.cpu_usage")
	if strings.Contains(field, ".") {
		return e.getNestedValue(field, context)
	}

	// Handle direct field access
	switch field {
	case "resource_type":
		return string(context.ResourceType), nil
	case "resource_id":
		return context.ResourceID, nil
	case "namespace":
		return context.Namespace, nil
	case "user":
		return context.User, nil
	case "event_type":
		return context.EventType, nil
	case "timestamp":
		return context.Timestamp, nil
	default:
		return nil, fmt.Errorf("unknown field: %s", field)
	}
}

func (e *DefaultPolicyEvaluator) getNestedValue(field string, context *PolicyEvaluationContext) (interface{}, error) {
	parts := strings.SplitN(field, ".", 2)
	if len(parts) != 2 {
		return nil, fmt.Errorf("invalid nested field format: %s", field)
	}

	prefix := parts[0]
	key := parts[1]

	switch prefix {
	case "labels":
		if context.Labels != nil {
			if value, exists := context.Labels[key]; exists {
				return value, nil
			}
		}
		return nil, fmt.Errorf("label not found: %s", key)

	case "tags":
		if context.Tags != nil {
			if value, exists := context.Tags[key]; exists {
				return value, nil
			}
		}
		return nil, fmt.Errorf("tag not found: %s", key)

	case "metrics":
		if context.Metrics != nil {
			if value, exists := context.Metrics[key]; exists {
				return value, nil
			}
		}
		return nil, fmt.Errorf("metric not found: %s", key)

	case "attributes":
		if context.Attributes != nil {
			if value, exists := context.Attributes[key]; exists {
				return value, nil
			}
		}
		return nil, fmt.Errorf("attribute not found: %s", key)

	case "event_data":
		if context.EventData != nil {
			if value, exists := context.EventData[key]; exists {
				return value, nil
			}
		}
		return nil, fmt.Errorf("event data not found: %s", key)

	default:
		return nil, fmt.Errorf("unknown nested field prefix: %s", prefix)
	}
}

func (e *DefaultPolicyEvaluator) evaluateOperator(operator ConditionOperator, actual, expected interface{}, expectedValues []interface{}) (bool, string, error) {
	switch operator {
	case OperatorEquals:
		matched := e.compareValues(actual, expected) == 0
		return matched, e.formatComparison(operator, actual, expected, matched), nil

	case OperatorNotEquals:
		matched := e.compareValues(actual, expected) != 0
		return matched, e.formatComparison(operator, actual, expected, matched), nil

	case OperatorGreaterThan:
		matched := e.compareValues(actual, expected) > 0
		return matched, e.formatComparison(operator, actual, expected, matched), nil

	case OperatorGreaterThanOrEqual:
		matched := e.compareValues(actual, expected) >= 0
		return matched, e.formatComparison(operator, actual, expected, matched), nil

	case OperatorLessThan:
		matched := e.compareValues(actual, expected) < 0
		return matched, e.formatComparison(operator, actual, expected, matched), nil

	case OperatorLessThanOrEqual:
		matched := e.compareValues(actual, expected) <= 0
		return matched, e.formatComparison(operator, actual, expected, matched), nil

	case OperatorIn:
		matched := e.valueInList(actual, expectedValues)
		return matched, e.formatListComparison(operator, actual, expectedValues, matched), nil

	case OperatorNotIn:
		matched := !e.valueInList(actual, expectedValues)
		return matched, e.formatListComparison(operator, actual, expectedValues, matched), nil

	case OperatorContains:
		matched := e.stringContains(actual, expected)
		return matched, e.formatStringComparison(operator, actual, expected, matched), nil

	case OperatorNotContains:
		matched := !e.stringContains(actual, expected)
		return matched, e.formatStringComparison(operator, actual, expected, matched), nil

	case OperatorMatches:
		matched, err := e.regexMatches(actual, expected)
		return matched, e.formatRegexComparison(operator, actual, expected, matched), err

	case OperatorNotMatches:
		matched, err := e.regexMatches(actual, expected)
		if err != nil {
			return false, "", err
		}
		matched = !matched
		return matched, e.formatRegexComparison(operator, actual, expected, matched), nil

	case OperatorExists:
		matched := actual != nil
		return matched, e.formatExistenceComparison(operator, actual, matched), nil

	case OperatorNotExists:
		matched := actual == nil
		return matched, e.formatExistenceComparison(operator, actual, matched), nil

	default:
		return false, "", fmt.Errorf("unsupported operator: %s", operator)
	}
}

func (e *DefaultPolicyEvaluator) compareValues(a, b interface{}) int {
	// Convert both values to comparable types
	aVal := e.convertToComparable(a)
	bVal := e.convertToComparable(b)

	switch aTyped := aVal.(type) {
	case float64:
		if bTyped, ok := bVal.(float64); ok {
			if aTyped < bTyped {
				return -1
			} else if aTyped > bTyped {
				return 1
			}
			return 0
		}
	case string:
		if bTyped, ok := bVal.(string); ok {
			return strings.Compare(aTyped, bTyped)
		}
	case bool:
		if bTyped, ok := bVal.(bool); ok {
			if aTyped == bTyped {
				return 0
			} else if aTyped && !bTyped {
				return 1
			}
			return -1
		}
	}

	// Fallback to reflect-based comparison
	if reflect.DeepEqual(aVal, bVal) {
		return 0
	}

	// If types don't match, convert to strings and compare
	aStr := fmt.Sprintf("%v", aVal)
	bStr := fmt.Sprintf("%v", bVal)
	return strings.Compare(aStr, bStr)
}

func (e *DefaultPolicyEvaluator) convertToComparable(value interface{}) interface{} {
	if value == nil {
		return nil
	}

	switch v := value.(type) {
	case int:
		return float64(v)
	case int32:
		return float64(v)
	case int64:
		return float64(v)
	case float32:
		return float64(v)
	case float64:
		return v
	case string:
		// Try to convert string to number if it looks like a number
		if floatVal, err := strconv.ParseFloat(v, 64); err == nil {
			return floatVal
		}
		return v
	case bool:
		return v
	default:
		return fmt.Sprintf("%v", v)
	}
}

func (e *DefaultPolicyEvaluator) valueInList(value interface{}, list []interface{}) bool {
	if list == nil {
		return false
	}

	for _, item := range list {
		if e.compareValues(value, item) == 0 {
			return true
		}
	}

	return false
}

func (e *DefaultPolicyEvaluator) stringContains(haystack, needle interface{}) bool {
	haystackStr := fmt.Sprintf("%v", haystack)
	needleStr := fmt.Sprintf("%v", needle)
	return strings.Contains(haystackStr, needleStr)
}

func (e *DefaultPolicyEvaluator) regexMatches(value, pattern interface{}) (bool, error) {
	valueStr := fmt.Sprintf("%v", value)
	patternStr := fmt.Sprintf("%v", pattern)

	regex, err := regexp.Compile(patternStr)
	if err != nil {
		return false, fmt.Errorf("invalid regex pattern '%s': %w", patternStr, err)
	}

	return regex.MatchString(valueStr), nil
}

func (e *DefaultPolicyEvaluator) isRuleScheduleActive(schedule *RuleSchedule) bool {
	if !schedule.Enabled {
		return false
	}

	now := time.Now()

	// Check time bounds
	if schedule.StartTime != nil && now.Before(*schedule.StartTime) {
		return false
	}

	if schedule.EndTime != nil && now.After(*schedule.EndTime) {
		return false
	}

	// TODO: Implement cron expression evaluation
	// For now, if cron expression is specified, assume it's always active
	if schedule.CronExpression != "" {
		return true
	}

	// If no specific schedule constraints, it's active
	return true
}

func (e *DefaultPolicyEvaluator) replaceContextVariables(expression string, context *PolicyEvaluationContext) string {
	// Simple variable replacement
	replacements := map[string]string{
		"${resource_type}": string(context.ResourceType),
		"${resource_id}":   context.ResourceID,
		"${namespace}":     context.Namespace,
		"${user}":          context.User,
		"${event_type}":    context.EventType,
	}

	result := expression
	for placeholder, value := range replacements {
		result = strings.ReplaceAll(result, placeholder, value)
	}

	// Replace nested variables (simplified)
	if context.Labels != nil {
		for key, value := range context.Labels {
			placeholder := fmt.Sprintf("${labels.%s}", key)
			result = strings.ReplaceAll(result, placeholder, value)
		}
	}

	if context.Metrics != nil {
		for key, value := range context.Metrics {
			placeholder := fmt.Sprintf("${metrics.%s}", key)
			result = strings.ReplaceAll(result, placeholder, fmt.Sprintf("%f", value))
		}
	}

	return result
}

func (e *DefaultPolicyEvaluator) evaluateSimpleExpression(expression string) (interface{}, error) {
	// Simplified expression evaluator
	// In a real implementation, you would use a proper expression parser
	
	expression = strings.TrimSpace(expression)

	// Handle boolean literals
	if expression == "true" {
		return true, nil
	}
	if expression == "false" {
		return false, nil
	}

	// Handle numeric literals
	if floatVal, err := strconv.ParseFloat(expression, 64); err == nil {
		return floatVal, nil
	}

	// Handle string literals
	if strings.HasPrefix(expression, `"`) && strings.HasSuffix(expression, `"`) {
		return strings.Trim(expression, `"`), nil
	}

	// Handle simple comparisons
	if strings.Contains(expression, ">") {
		return e.evaluateComparison(expression, ">")
	}
	if strings.Contains(expression, "<") {
		return e.evaluateComparison(expression, "<")
	}
	if strings.Contains(expression, "==") {
		return e.evaluateComparison(expression, "==")
	}

	// Default to string value
	return expression, nil
}

func (e *DefaultPolicyEvaluator) evaluateComparison(expression, operator string) (bool, error) {
	parts := strings.SplitN(expression, operator, 2)
	if len(parts) != 2 {
		return false, fmt.Errorf("invalid comparison expression: %s", expression)
	}

	left := strings.TrimSpace(parts[0])
	right := strings.TrimSpace(parts[1])

	// Convert to numbers if possible
	leftVal, leftErr := strconv.ParseFloat(left, 64)
	rightVal, rightErr := strconv.ParseFloat(right, 64)

	if leftErr == nil && rightErr == nil {
		// Numeric comparison
		switch operator {
		case ">":
			return leftVal > rightVal, nil
		case "<":
			return leftVal < rightVal, nil
		case "==":
			return math.Abs(leftVal-rightVal) < 1e-9, nil
		}
	} else {
		// String comparison
		switch operator {
		case ">":
			return left > right, nil
		case "<":
			return left < right, nil
		case "==":
			return left == right, nil
		}
	}

	return false, fmt.Errorf("unsupported comparison operator: %s", operator)
}

// Formatting methods for explanations

func (e *DefaultPolicyEvaluator) formatComparison(operator ConditionOperator, actual, expected interface{}, matched bool) string {
	status := "✓"
	if !matched {
		status = "✗"
	}
	return fmt.Sprintf("%s %v %s %v", status, actual, operator, expected)
}

func (e *DefaultPolicyEvaluator) formatListComparison(operator ConditionOperator, actual interface{}, expectedList []interface{}, matched bool) string {
	status := "✓"
	if !matched {
		status = "✗"
	}
	return fmt.Sprintf("%s %v %s %v", status, actual, operator, expectedList)
}

func (e *DefaultPolicyEvaluator) formatStringComparison(operator ConditionOperator, actual, expected interface{}, matched bool) string {
	status := "✓"
	if !matched {
		status = "✗"
	}
	return fmt.Sprintf("%s '%v' %s '%v'", status, actual, operator, expected)
}

func (e *DefaultPolicyEvaluator) formatRegexComparison(operator ConditionOperator, actual, pattern interface{}, matched bool) string {
	status := "✓"
	if !matched {
		status = "✗"
	}
	return fmt.Sprintf("%s '%v' %s /%v/", status, actual, operator, pattern)
}

func (e *DefaultPolicyEvaluator) formatExistenceComparison(operator ConditionOperator, actual interface{}, matched bool) string {
	status := "✓"
	if !matched {
		status = "✗"
	}
	return fmt.Sprintf("%s value %s (actual: %v)", status, operator, actual)
}