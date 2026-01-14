package constraints

import (
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"
)

// ConstraintOperatorType defines the type of operator in a constraint expression
type ConstraintOperatorType string

const (
	// Comparison operators
	OperatorEqual            ConstraintOperatorType = "eq"
	OperatorNotEqual         ConstraintOperatorType = "neq"
	OperatorLessThan         ConstraintOperatorType = "lt"
	OperatorLessThanEqual    ConstraintOperatorType = "lte"
	OperatorGreaterThan      ConstraintOperatorType = "gt"
	OperatorGreaterThanEqual ConstraintOperatorType = "gte"
	OperatorIn               ConstraintOperatorType = "in"
	OperatorNotIn            ConstraintOperatorType = "not_in"
	OperatorContains         ConstraintOperatorType = "contains"
	OperatorNotContains      ConstraintOperatorType = "not_contains"
	OperatorMatches          ConstraintOperatorType = "matches"
	OperatorNotMatches       ConstraintOperatorType = "not_matches"
	OperatorExists           ConstraintOperatorType = "exists"
	OperatorNotExists        ConstraintOperatorType = "not_exists"

	// Logical operators
	OperatorAnd ConstraintOperatorType = "and"
	OperatorOr  ConstraintOperatorType = "or"
	OperatorNot ConstraintOperatorType = "not"
)

// ConstraintValueType defines the type of a constraint value
type ConstraintValueType string

const (
	ValueTypeString     ConstraintValueType = "string"
	ValueTypeNumber     ConstraintValueType = "number"
	ValueTypeBoolean    ConstraintValueType = "boolean"
	ValueTypeArray      ConstraintValueType = "array"
	ValueTypeObject     ConstraintValueType = "object"
	ValueTypeExpression ConstraintValueType = "expression"
	ValueTypeReference  ConstraintValueType = "reference"
	ValueTypeNull       ConstraintValueType = "null"
)

// ConstraintValue represents a value in a constraint expression
type ConstraintValue struct {
	// Type is the type of the value
	Type ConstraintValueType `json:"type"`

	// StringValue is used when Type is ValueTypeString
	StringValue string `json:"stringValue,omitempty"`

	// NumberValue is used when Type is ValueTypeNumber
	NumberValue float64 `json:"numberValue,omitempty"`

	// BooleanValue is used when Type is ValueTypeBoolean
	BooleanValue bool `json:"booleanValue,omitempty"`

	// ArrayValue is used when Type is ValueTypeArray
	ArrayValue []ConstraintValue `json:"arrayValue,omitempty"`

	// ObjectValue is used when Type is ValueTypeObject
	ObjectValue map[string]ConstraintValue `json:"objectValue,omitempty"`

	// ExpressionValue is used when Type is ValueTypeExpression
	ExpressionValue *ConstraintExpression `json:"expressionValue,omitempty"`

	// ReferenceValue is used when Type is ValueTypeReference
	ReferenceValue string `json:"referenceValue,omitempty"`
}

// ConstraintExpression represents a constraint expression
type ConstraintExpression struct {
	// Operator is the operator in the expression
	Operator ConstraintOperatorType `json:"operator"`

	// Left is the left operand
	Left *ConstraintValue `json:"left,omitempty"`

	// Right is the right operand
	Right *ConstraintValue `json:"right,omitempty"`

	// Arguments are the arguments for the operator (for logical operators)
	Arguments []*ConstraintExpression `json:"arguments,omitempty"`

	// Weight is the weight of this expression in the overall constraint (0.0-1.0)
	Weight float64 `json:"weight,omitempty"`
}

// ConstraintSolverContext contains the context for constraint solving
type ConstraintSolverContext struct {
	// EntityAttributes contains entity attributes used for constraint evaluation
	EntityAttributes map[string]map[string]interface{}

	// NodeAttributes contains node attributes used for constraint evaluation
	NodeAttributes map[string]map[string]interface{}

	// EntityPlacements contains current entity placements (entity ID -> node ID)
	EntityPlacements map[string]string

	// CustomFunctions contains custom functions that can be used in constraints
	CustomFunctions map[string]func(args ...interface{}) (interface{}, error)
}

// ConstraintSolverResult represents the result of constraint evaluation
type ConstraintSolverResult struct {
	// Satisfied indicates if the constraint is satisfied
	Satisfied bool

	// Score is the score for the constraint (0.0-1.0)
	Score float64

	// Reason is a human-readable explanation of the result
	Reason string

	// SubResults contains results for sub-expressions
	SubResults []ConstraintSolverResult
}

// ConstraintSolver solves constraints
type ConstraintSolver struct {
	// context is the context for constraint solving
	context *ConstraintSolverContext
}

// NewConstraintSolver creates a new constraint solver
func NewConstraintSolver(context *ConstraintSolverContext) *ConstraintSolver {
	if context == nil {
		context = &ConstraintSolverContext{
			EntityAttributes: make(map[string]map[string]interface{}),
			NodeAttributes:   make(map[string]map[string]interface{}),
			EntityPlacements: make(map[string]string),
			CustomFunctions:  make(map[string]func(args ...interface{}) (interface{}, error)),
		}
	}

	return &ConstraintSolver{
		context: context,
	}
}

// Evaluate evaluates a constraint expression
func (s *ConstraintSolver) Evaluate(expr *ConstraintExpression) (*ConstraintSolverResult, error) {
	if expr == nil {
		return &ConstraintSolverResult{
			Satisfied: true,
			Score:     1.0,
			Reason:    "Empty expression is always satisfied",
		}, nil
	}

	result := &ConstraintSolverResult{
		SubResults: make([]ConstraintSolverResult, 0),
	}

	// Handle logical operators
	switch expr.Operator {
	case OperatorAnd:
		return s.evaluateAnd(expr, result)
	case OperatorOr:
		return s.evaluateOr(expr, result)
	case OperatorNot:
		return s.evaluateNot(expr, result)
	}

	// Handle comparison operators
	if expr.Left == nil || expr.Right == nil {
		return nil, fmt.Errorf("left and right operands must be provided for operator %s", expr.Operator)
	}

	leftValue, err := s.resolveValue(expr.Left)
	if err != nil {
		return nil, fmt.Errorf("error resolving left operand: %v", err)
	}

	rightValue, err := s.resolveValue(expr.Right)
	if err != nil {
		return nil, fmt.Errorf("error resolving right operand: %v", err)
	}

	// Evaluate the expression based on the operator
	return s.evaluateComparison(expr.Operator, leftValue, rightValue)
}

// evaluateComparison evaluates a comparison expression
func (s *ConstraintSolver) evaluateComparison(
	operator ConstraintOperatorType,
	left interface{},
	right interface{},
) (*ConstraintSolverResult, error) {
	switch operator {
	case OperatorEqual:
		return s.evaluateEqual(left, right)
	case OperatorNotEqual:
		result, err := s.evaluateEqual(left, right)
		if err != nil {
			return nil, err
		}
		result.Satisfied = !result.Satisfied
		result.Score = 1.0 - result.Score
		result.Reason = fmt.Sprintf("%v not equal to %v", left, right)
		return result, nil
	case OperatorLessThan:
		return s.evaluateLessThan(left, right)
	case OperatorLessThanEqual:
		return s.evaluateLessThanEqual(left, right)
	case OperatorGreaterThan:
		return s.evaluateGreaterThan(left, right)
	case OperatorGreaterThanEqual:
		return s.evaluateGreaterThanEqual(left, right)
	case OperatorIn:
		return s.evaluateIn(left, right)
	case OperatorNotIn:
		result, err := s.evaluateIn(left, right)
		if err != nil {
			return nil, err
		}
		result.Satisfied = !result.Satisfied
		result.Score = 1.0 - result.Score
		result.Reason = fmt.Sprintf("%v not in %v", left, right)
		return result, nil
	case OperatorContains:
		return s.evaluateContains(left, right)
	case OperatorNotContains:
		result, err := s.evaluateContains(left, right)
		if err != nil {
			return nil, err
		}
		result.Satisfied = !result.Satisfied
		result.Score = 1.0 - result.Score
		result.Reason = fmt.Sprintf("%v does not contain %v", left, right)
		return result, nil
	case OperatorExists:
		return s.evaluateExists(left)
	case OperatorNotExists:
		result, err := s.evaluateExists(left)
		if err != nil {
			return nil, err
		}
		result.Satisfied = !result.Satisfied
		result.Score = 1.0 - result.Score
		result.Reason = fmt.Sprintf("%v does not exist", left)
		return result, nil
	default:
		return nil, fmt.Errorf("unsupported operator: %s", operator)
	}
}

// evaluateAnd evaluates an AND expression
func (s *ConstraintSolver) evaluateAnd(expr *ConstraintExpression, result *ConstraintSolverResult) (*ConstraintSolverResult, error) {
	if len(expr.Arguments) == 0 {
		result.Satisfied = true
		result.Score = 1.0
		result.Reason = "Empty AND expression is always satisfied"
		return result, nil
	}

	// All arguments must be satisfied for AND
	totalScore := 0.0
	allSatisfied := true

	for _, arg := range expr.Arguments {
		subResult, err := s.Evaluate(arg)
		if err != nil {
			return nil, err
		}

		result.SubResults = append(result.SubResults, *subResult)
		totalScore += subResult.Score

		if !subResult.Satisfied {
			allSatisfied = false
		}
	}

	// Calculate average score
	avgScore := totalScore / float64(len(expr.Arguments))

	result.Satisfied = allSatisfied
	result.Score = avgScore
	if allSatisfied {
		result.Reason = "All conditions satisfied"
	} else {
		result.Reason = "Not all conditions satisfied"
	}

	return result, nil
}

// evaluateOr evaluates an OR expression
func (s *ConstraintSolver) evaluateOr(expr *ConstraintExpression, result *ConstraintSolverResult) (*ConstraintSolverResult, error) {
	if len(expr.Arguments) == 0 {
		result.Satisfied = false
		result.Score = 0.0
		result.Reason = "Empty OR expression is never satisfied"
		return result, nil
	}

	// Any argument must be satisfied for OR
	maxScore := 0.0
	anySatisfied := false

	for _, arg := range expr.Arguments {
		subResult, err := s.Evaluate(arg)
		if err != nil {
			return nil, err
		}

		result.SubResults = append(result.SubResults, *subResult)
		if subResult.Score > maxScore {
			maxScore = subResult.Score
		}

		if subResult.Satisfied {
			anySatisfied = true
		}
	}

	result.Satisfied = anySatisfied
	result.Score = maxScore
	if anySatisfied {
		result.Reason = "At least one condition satisfied"
	} else {
		result.Reason = "No conditions satisfied"
	}

	return result, nil
}

// evaluateNot evaluates a NOT expression
func (s *ConstraintSolver) evaluateNot(expr *ConstraintExpression, result *ConstraintSolverResult) (*ConstraintSolverResult, error) {
	if len(expr.Arguments) != 1 {
		return nil, fmt.Errorf("NOT operator requires exactly one argument, got %d", len(expr.Arguments))
	}

	subResult, err := s.Evaluate(expr.Arguments[0])
	if err != nil {
		return nil, err
	}

	result.SubResults = append(result.SubResults, *subResult)
	result.Satisfied = !subResult.Satisfied
	result.Score = 1.0 - subResult.Score
	result.Reason = fmt.Sprintf("NOT (%s)", subResult.Reason)

	return result, nil
}

// evaluateEqual evaluates an equality expression
func (s *ConstraintSolver) evaluateEqual(left interface{}, right interface{}) (*ConstraintSolverResult, error) {
	result := &ConstraintSolverResult{}

	// Handle nil values
	if left == nil && right == nil {
		result.Satisfied = true
		result.Score = 1.0
		result.Reason = "Both values are nil"
		return result, nil
	}
	if left == nil || right == nil {
		result.Satisfied = false
		result.Score = 0.0
		result.Reason = "One value is nil, the other is not"
		return result, nil
	}

	// Convert to comparable types if needed
	left, right, err := s.convertToComparableTypes(left, right)
	if err != nil {
		return nil, err
	}

	// Check equality based on type
	leftType := reflect.TypeOf(left)
	rightType := reflect.TypeOf(right)

	if leftType != rightType {
		result.Satisfied = false
		result.Score = 0.0
		result.Reason = fmt.Sprintf("Types differ: %v vs %v", leftType, rightType)
		return result, nil
	}

	result.Satisfied = reflect.DeepEqual(left, right)
	if result.Satisfied {
		result.Score = 1.0
		result.Reason = fmt.Sprintf("%v equals %v", left, right)
	} else {
		result.Score = 0.0
		result.Reason = fmt.Sprintf("%v does not equal %v", left, right)
	}

	return result, nil
}

// evaluateLessThan evaluates a less-than expression
func (s *ConstraintSolver) evaluateLessThan(left interface{}, right interface{}) (*ConstraintSolverResult, error) {
	result := &ConstraintSolverResult{}

	// Convert to numeric values
	leftNum, rightNum, err := s.convertToNumeric(left, right)
	if err != nil {
		return nil, err
	}

	result.Satisfied = leftNum < rightNum
	if result.Satisfied {
		// Calculate a score based on how far below the threshold we are
		if rightNum != 0 {
			ratio := leftNum / rightNum
			if ratio < 0 {
				result.Score = 1.0 // Negative value is always less than positive
			} else {
				result.Score = 1.0 - ratio // Closer to threshold = lower score
			}
		} else {
			result.Score = 1.0 // Any negative value is less than zero
		}
		result.Reason = fmt.Sprintf("%v < %v", left, right)
	} else {
		result.Score = 0.0
		result.Reason = fmt.Sprintf("%v not < %v", left, right)
	}

	return result, nil
}

// evaluateLessThanEqual evaluates a less-than-or-equal expression
func (s *ConstraintSolver) evaluateLessThanEqual(left interface{}, right interface{}) (*ConstraintSolverResult, error) {
	result := &ConstraintSolverResult{}

	// Convert to numeric values
	leftNum, rightNum, err := s.convertToNumeric(left, right)
	if err != nil {
		return nil, err
	}

	result.Satisfied = leftNum <= rightNum
	if result.Satisfied {
		if leftNum == rightNum {
			result.Score = 1.0
			result.Reason = fmt.Sprintf("%v = %v", left, right)
		} else {
			// Calculate a score based on how far below the threshold we are
			if rightNum != 0 {
				ratio := leftNum / rightNum
				if ratio < 0 {
					result.Score = 1.0 // Negative value is always less than positive
				} else {
					result.Score = 1.0 - ratio // Closer to threshold = lower score
				}
			} else {
				result.Score = 1.0 // Any negative value is less than zero
			}
			result.Reason = fmt.Sprintf("%v <= %v", left, right)
		}
	} else {
		result.Score = 0.0
		result.Reason = fmt.Sprintf("%v not <= %v", left, right)
	}

	return result, nil
}

// evaluateGreaterThan evaluates a greater-than expression
func (s *ConstraintSolver) evaluateGreaterThan(left interface{}, right interface{}) (*ConstraintSolverResult, error) {
	result := &ConstraintSolverResult{}

	// Convert to numeric values
	leftNum, rightNum, err := s.convertToNumeric(left, right)
	if err != nil {
		return nil, err
	}

	result.Satisfied = leftNum > rightNum
	if result.Satisfied {
		// Calculate a score based on how far above the threshold we are
		if rightNum != 0 {
			ratio := rightNum / leftNum
			if ratio < 0 {
				result.Score = 1.0 // Positive value is always greater than negative
			} else {
				result.Score = 1.0 - ratio // Further above threshold = higher score
			}
		} else {
			result.Score = math.Min(leftNum/100.0, 1.0) // Normalize the score
		}
		result.Reason = fmt.Sprintf("%v > %v", left, right)
	} else {
		result.Score = 0.0
		result.Reason = fmt.Sprintf("%v not > %v", left, right)
	}

	return result, nil
}

// evaluateGreaterThanEqual evaluates a greater-than-or-equal expression
func (s *ConstraintSolver) evaluateGreaterThanEqual(left interface{}, right interface{}) (*ConstraintSolverResult, error) {
	result := &ConstraintSolverResult{}

	// Convert to numeric values
	leftNum, rightNum, err := s.convertToNumeric(left, right)
	if err != nil {
		return nil, err
	}

	result.Satisfied = leftNum >= rightNum
	if result.Satisfied {
		if leftNum == rightNum {
			result.Score = 1.0
			result.Reason = fmt.Sprintf("%v = %v", left, right)
		} else {
			// Calculate a score based on how far above the threshold we are
			if rightNum != 0 {
				ratio := rightNum / leftNum
				if ratio < 0 {
					result.Score = 1.0 // Positive value is always greater than negative
				} else {
					result.Score = 1.0 - ratio // Further above threshold = higher score
				}
			} else {
				result.Score = math.Min(leftNum/100.0, 1.0) // Normalize the score
			}
			result.Reason = fmt.Sprintf("%v >= %v", left, right)
		}
	} else {
		result.Score = 0.0
		result.Reason = fmt.Sprintf("%v not >= %v", left, right)
	}

	return result, nil
}

// evaluateIn evaluates an IN expression
func (s *ConstraintSolver) evaluateIn(left interface{}, right interface{}) (*ConstraintSolverResult, error) {
	result := &ConstraintSolverResult{}

	// Right operand must be an array or map
	rightValue := reflect.ValueOf(right)
	if rightValue.Kind() != reflect.Slice && rightValue.Kind() != reflect.Map {
		return nil, fmt.Errorf("right operand must be a slice or map for IN operator, got %T", right)
	}

	// Check if left is in right
	found := false
	if rightValue.Kind() == reflect.Slice {
		for i := 0; i < rightValue.Len(); i++ {
			item := rightValue.Index(i).Interface()
			equal, err := s.evaluateEqual(left, item)
			if err != nil {
				return nil, err
			}
			if equal.Satisfied {
				found = true
				break
			}
		}
	} else if rightValue.Kind() == reflect.Map {
		// For maps, check if the key exists
		leftStr, ok := left.(string)
		if !ok {
			return nil, fmt.Errorf("left operand must be a string for IN operator with map, got %T", left)
		}
		for _, key := range rightValue.MapKeys() {
			keyStr := key.String()
			if keyStr == leftStr {
				found = true
				break
			}
		}
	}

	result.Satisfied = found
	if found {
		result.Score = 1.0
		result.Reason = fmt.Sprintf("%v is in the collection", left)
	} else {
		result.Score = 0.0
		result.Reason = fmt.Sprintf("%v is not in the collection", left)
	}

	return result, nil
}

// evaluateContains evaluates a CONTAINS expression
func (s *ConstraintSolver) evaluateContains(left interface{}, right interface{}) (*ConstraintSolverResult, error) {
	result := &ConstraintSolverResult{}

	// Left operand must be a string, array, or map
	leftValue := reflect.ValueOf(left)
	if leftValue.Kind() != reflect.String && leftValue.Kind() != reflect.Slice && leftValue.Kind() != reflect.Map {
		return nil, fmt.Errorf("left operand must be a string, slice, or map for CONTAINS operator, got %T", left)
	}

	// Check if left contains right
	found := false
	if leftValue.Kind() == reflect.String {
		// For strings, check if the string contains the substring
		leftStr := left.(string)
		rightStr, ok := right.(string)
		if !ok {
			return nil, fmt.Errorf("right operand must be a string for CONTAINS operator with string, got %T", right)
		}
		found = strings.Contains(leftStr, rightStr)
	} else if leftValue.Kind() == reflect.Slice {
		// For slices, check if any element equals the right operand
		for i := 0; i < leftValue.Len(); i++ {
			item := leftValue.Index(i).Interface()
			equal, err := s.evaluateEqual(item, right)
			if err != nil {
				return nil, err
			}
			if equal.Satisfied {
				found = true
				break
			}
		}
	} else if leftValue.Kind() == reflect.Map {
		// For maps, check if the key exists
		rightStr, ok := right.(string)
		if !ok {
			return nil, fmt.Errorf("right operand must be a string for CONTAINS operator with map, got %T", right)
		}
		for _, key := range leftValue.MapKeys() {
			keyStr := key.String()
			if keyStr == rightStr {
				found = true
				break
			}
		}
	}

	result.Satisfied = found
	if found {
		result.Score = 1.0
		result.Reason = fmt.Sprintf("Collection contains %v", right)
	} else {
		result.Score = 0.0
		result.Reason = fmt.Sprintf("Collection does not contain %v", right)
	}

	return result, nil
}

// evaluateExists evaluates an EXISTS expression
func (s *ConstraintSolver) evaluateExists(value interface{}) (*ConstraintSolverResult, error) {
	result := &ConstraintSolverResult{}

	// nil values don't exist
	if value == nil {
		result.Satisfied = false
		result.Score = 0.0
		result.Reason = "Value is nil"
		return result, nil
	}

	// Empty strings, slices, and maps are considered non-existent
	valueReflect := reflect.ValueOf(value)
	if valueReflect.Kind() == reflect.String && valueReflect.String() == "" {
		result.Satisfied = false
		result.Score = 0.0
		result.Reason = "String is empty"
		return result, nil
	}
	if valueReflect.Kind() == reflect.Slice && valueReflect.Len() == 0 {
		result.Satisfied = false
		result.Score = 0.0
		result.Reason = "Slice is empty"
		return result, nil
	}
	if valueReflect.Kind() == reflect.Map && valueReflect.Len() == 0 {
		result.Satisfied = false
		result.Score = 0.0
		result.Reason = "Map is empty"
		return result, nil
	}

	// All other values exist
	result.Satisfied = true
	result.Score = 1.0
	result.Reason = "Value exists"
	return result, nil
}

// resolveValue resolves a constraint value to its actual value
func (s *ConstraintSolver) resolveValue(value *ConstraintValue) (interface{}, error) {
	if value == nil {
		return nil, nil
	}

	switch value.Type {
	case ValueTypeString:
		return value.StringValue, nil
	case ValueTypeNumber:
		return value.NumberValue, nil
	case ValueTypeBoolean:
		return value.BooleanValue, nil
	case ValueTypeArray:
		array := make([]interface{}, len(value.ArrayValue))
		for i, item := range value.ArrayValue {
			resolvedItem, err := s.resolveValue(&item)
			if err != nil {
				return nil, err
			}
			array[i] = resolvedItem
		}
		return array, nil
	case ValueTypeObject:
		object := make(map[string]interface{})
		for key, item := range value.ObjectValue {
			resolvedItem, err := s.resolveValue(&item)
			if err != nil {
				return nil, err
			}
			object[key] = resolvedItem
		}
		return object, nil
	case ValueTypeExpression:
		result, err := s.Evaluate(value.ExpressionValue)
		if err != nil {
			return nil, err
		}
		return result.Satisfied, nil
	case ValueTypeReference:
		return s.resolveReference(value.ReferenceValue)
	case ValueTypeNull:
		return nil, nil
	default:
		return nil, fmt.Errorf("unsupported value type: %s", value.Type)
	}
}

// resolveReference resolves a reference to an entity or node attribute
func (s *ConstraintSolver) resolveReference(reference string) (interface{}, error) {
	parts := strings.Split(reference, ".")
	if len(parts) < 2 {
		return nil, fmt.Errorf("invalid reference format: %s", reference)
	}

	// Get the entity or node ID
	scope := parts[0]
	id := parts[1]
	attributePath := parts[2:]

	// Handle special references
	if scope == "function" {
		// Function reference
		if len(attributePath) == 0 {
			return nil, fmt.Errorf("function name not specified in reference: %s", reference)
		}
		functionName := attributePath[0]
		function, exists := s.context.CustomFunctions[functionName]
		if !exists {
			return nil, fmt.Errorf("function not found: %s", functionName)
		}

		// Prepare function arguments
		args := make([]interface{}, 0)
		for i := 1; i < len(attributePath); i++ {
			// Try to resolve arguments as references
			arg, err := s.resolveReference(attributePath[i])
			if err != nil {
				// If not a valid reference, use the argument as is
				arg = attributePath[i]
			}
			args = append(args, arg)
		}

		// Call the function
		return function(args...)
	} else if scope == "placement" {
		// Placement reference
		if id == "current" && len(attributePath) > 0 {
			// Get the current placement of an entity
			entityID := attributePath[0]
			nodeID, exists := s.context.EntityPlacements[entityID]
			if !exists {
				return nil, fmt.Errorf("entity not placed: %s", entityID)
			}
			return nodeID, nil
		}
		return nil, fmt.Errorf("invalid placement reference: %s", reference)
	}

	// Regular entity or node attribute reference
	var attributes map[string]interface{}
	var exists bool

	if scope == "entity" {
		attributes, exists = s.context.EntityAttributes[id]
	} else if scope == "node" {
		attributes, exists = s.context.NodeAttributes[id]
	} else {
		return nil, fmt.Errorf("invalid reference scope: %s", scope)
	}

	if !exists {
		return nil, fmt.Errorf("%s not found: %s", scope, id)
	}

	// Navigate the attribute path
	value := interface{}(attributes)
	for _, pathPart := range attributePath {
		// Convert value to map
		valueMap, ok := value.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("cannot navigate path %s: not a map", pathPart)
		}

		// Get the next value in the path
		value, exists = valueMap[pathPart]
		if !exists {
			return nil, fmt.Errorf("attribute not found: %s", pathPart)
		}
	}

	return value, nil
}

// convertToComparableTypes converts two values to comparable types if possible
func (s *ConstraintSolver) convertToComparableTypes(left interface{}, right interface{}) (interface{}, interface{}, error) {
	// If types are already the same, no conversion needed
	leftType := reflect.TypeOf(left)
	rightType := reflect.TypeOf(right)
	if leftType == rightType {
		return left, right, nil
	}

	// Try to convert strings to numbers
	leftStr, leftIsStr := left.(string)
	rightStr, rightIsStr := right.(string)

	// String and number
	if leftIsStr && rightType != nil && (rightType.Kind() == reflect.Int ||
		rightType.Kind() == reflect.Int64 || rightType.Kind() == reflect.Float64) {
		// Convert left string to number
		leftNum, err := stringToNumber(leftStr)
		if err == nil {
			return leftNum, right, nil
		}
	}
	if rightIsStr && leftType != nil && (leftType.Kind() == reflect.Int ||
		leftType.Kind() == reflect.Int64 || leftType.Kind() == reflect.Float64) {
		// Convert right string to number
		rightNum, err := stringToNumber(rightStr)
		if err == nil {
			return left, rightNum, nil
		}
	}

	// String and bool
	if leftIsStr && rightType != nil && rightType.Kind() == reflect.Bool {
		// Convert left string to bool
		leftBool, err := stringToBool(leftStr)
		if err == nil {
			return leftBool, right, nil
		}
	}
	if rightIsStr && leftType != nil && leftType.Kind() == reflect.Bool {
		// Convert right string to bool
		rightBool, err := stringToBool(rightStr)
		if err == nil {
			return left, rightBool, nil
		}
	}

	// No conversion possible
	return left, right, nil
}

// convertToNumeric converts two values to numeric types if possible
func (s *ConstraintSolver) convertToNumeric(left interface{}, right interface{}) (float64, float64, error) {
	// Convert left to float64
	var leftNum float64
	switch v := left.(type) {
	case int:
		leftNum = float64(v)
	case int64:
		leftNum = float64(v)
	case float64:
		leftNum = v
	case string:
		num, err := stringToNumber(v)
		if err != nil {
			return 0, 0, fmt.Errorf("left operand is not a number: %v", left)
		}
		leftNum = num
	default:
		return 0, 0, fmt.Errorf("left operand is not a number: %v", left)
	}

	// Convert right to float64
	var rightNum float64
	switch v := right.(type) {
	case int:
		rightNum = float64(v)
	case int64:
		rightNum = float64(v)
	case float64:
		rightNum = v
	case string:
		num, err := stringToNumber(v)
		if err != nil {
			return 0, 0, fmt.Errorf("right operand is not a number: %v", right)
		}
		rightNum = num
	default:
		return 0, 0, fmt.Errorf("right operand is not a number: %v", right)
	}

	return leftNum, rightNum, nil
}

// stringToNumber converts a string to a float64
func stringToNumber(s string) (float64, error) {
	return strconv.ParseFloat(s, 64)
}

// stringToBool converts a string to a bool
func stringToBool(s string) (bool, error) {
	switch strings.ToLower(s) {
	case "true", "yes", "y", "1":
		return true, nil
	case "false", "no", "n", "0":
		return false, nil
	default:
		return false, fmt.Errorf("cannot convert string to bool: %s", s)
	}
}
