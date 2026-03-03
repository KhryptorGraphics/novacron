package policy

import (
	"testing"
)

func TestMockExpression(t *testing.T) {
	// Create a mock expression
	expr := &MockExpression{Value: 42}

	// Test String() method
	if expr.String() != "42" {
		t.Errorf("Expected String() to return '42', got '%s'", expr.String())
	}

	// Test Evaluate() method
	ctx := NewEvaluationContext()
	result, err := expr.Evaluate(ctx)

	if err != nil {
		t.Errorf("Unexpected error from Evaluate(): %v", err)
	}

	if result != 42 {
		t.Errorf("Expected Evaluate() to return 42, got %v", result)
	}
}

func TestCompileMock(t *testing.T) {
	// Create a compiler
	compiler := &DefaultExpressionCompiler{}

	// Compile a mock expression
	expr, err := compiler.CompileMock("test expression")

	if err != nil {
		t.Errorf("Unexpected error from CompileMock(): %v", err)
	}

	// Check that it's a MockExpression
	mockExpr, ok := expr.(*MockExpression)
	if !ok {
		t.Errorf("Expected CompileMock() to return *MockExpression, got %T", expr)
	}

	// Check the value
	if mockExpr.Value != "test expression" {
		t.Errorf("Expected MockExpression.Value to be 'test expression', got '%v'", mockExpr.Value)
	}
}
