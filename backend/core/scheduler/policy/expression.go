package policy

import (
	"fmt"
)

// Expression represents a policy expression that can be evaluated
type Expression interface {
	// Evaluate evaluates the expression in the given context and returns the result
	Evaluate(ctx *EvaluationContext) (interface{}, error)

	// String returns a string representation of the expression
	String() string
}

// ExpressionCompiler compiles string expressions into Expression objects
type ExpressionCompiler interface {
	// Compile compiles a string expression into an Expression object
	Compile(expr string) (Expression, error)
	
	// CompileExpression is an alias for Compile for backward compatibility
	CompileExpression(expr string) (Expression, error)
}

// DefaultExpressionCompiler implements a simple expression compiler
type DefaultExpressionCompiler struct{}

// Compile compiles a string expression into an Expression object
func (c *DefaultExpressionCompiler) Compile(expr string) (Expression, error) {
	// For simplicity in this example, we're using the mock implementation
	return c.CompileMock(expr)
}

// CompileExpression is an alias for Compile for backward compatibility
func (c *DefaultExpressionCompiler) CompileExpression(expr string) (Expression, error) {
	return c.Compile(expr)
}

// ExpressionEvaluator evaluates policy expressions
type ExpressionEvaluator struct {
	compiler ExpressionCompiler
}

// NewExpressionEvaluator creates a new expression evaluator
func NewExpressionEvaluator() *ExpressionEvaluator {
	return &ExpressionEvaluator{
		compiler: &DefaultExpressionCompiler{},
	}
}

// Evaluate evaluates a string expression in the given context
func (e *ExpressionEvaluator) Evaluate(expr string, ctx *EvaluationContext) (interface{}, error) {
	compiled, err := e.compiler.Compile(expr)
	if err != nil {
		return nil, fmt.Errorf("compile error: %v", err)
	}

	result, err := compiled.Evaluate(ctx)
	if err != nil {
		return nil, fmt.Errorf("evaluation error: %v", err)
	}

	return result, nil
}

// EvaluateAsBool evaluates a string expression and converts the result to a boolean
func (e *ExpressionEvaluator) EvaluateAsBool(expr string, ctx *EvaluationContext) (bool, error) {
	result, err := e.Evaluate(expr, ctx)
	if err != nil {
		return false, err
	}

	switch v := result.(type) {
	case bool:
		return v, nil
	case int:
		return v != 0, nil
	case float64:
		return v != 0, nil
	case string:
		return v != "", nil
	default:
		return false, fmt.Errorf("cannot convert %T to bool", result)
	}
}

// EvaluateAsFloat evaluates a string expression and converts the result to a float64
func (e *ExpressionEvaluator) EvaluateAsFloat(expr string, ctx *EvaluationContext) (float64, error) {
	result, err := e.Evaluate(expr, ctx)
	if err != nil {
		return 0, err
	}

	switch v := result.(type) {
	case float64:
		return v, nil
	case int:
		return float64(v), nil
	case bool:
		if v {
			return 1, nil
		}
		return 0, nil
	default:
		return 0, fmt.Errorf("cannot convert %T to float64", result)
	}
}

// LiteralExpression represents a literal value expression
type LiteralExpression struct {
	Value interface{}
}

// Evaluate implements Expression.Evaluate for LiteralExpression
func (e *LiteralExpression) Evaluate(ctx *EvaluationContext) (interface{}, error) {
	return e.Value, nil
}

// String implements Expression.String for LiteralExpression
func (e *LiteralExpression) String() string {
	return fmt.Sprintf("%v", e.Value)
}
