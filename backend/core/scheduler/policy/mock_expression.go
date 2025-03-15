package policy

import (
	"fmt"
)

// MockExpression is a simple implementation of Expression for testing
type MockExpression struct {
	Value interface{}
}

// Evaluate implements Expression.Evaluate for MockExpression
func (e *MockExpression) Evaluate(ctx *EvaluationContext) (interface{}, error) {
	return e.Value, nil
}

// String implements Expression.String for MockExpression
func (e *MockExpression) String() string {
	return fmt.Sprintf("%v", e.Value)
}

// CompileMock is a helper for creating mock expressions
func (c *DefaultExpressionCompiler) CompileMock(expr string) (Expression, error) {
	return &MockExpression{Value: expr}, nil
}
