package test_basic

import (
	"fmt"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/policy"
)

// RunTest executes a basic test of the policy package
func RunTest() {
	fmt.Println("Testing policy package basic functionality...")

	// Create a simple mock expression
	expr := &policy.MockExpression{Value: true}
	fmt.Println("Mock expression created:", expr.String())

	// Create a basic evaluation context
	ctx := policy.NewEvaluationContext()
	ctx.SetVariable("test", "value")
	value, exists := ctx.GetVariable("test")
	fmt.Printf("Variable 'test' exists: %v, value: %v\n", exists, value)

	// Create a basic policy evaluation context
	policyCtx := policy.NewPolicyEvaluationContext(
		map[string]interface{}{"id": "vm-001"},
		map[string]interface{}{"id": "node-001"},
		map[string]interface{}{"id": "node-002"},
	)
	policyCtx.SetParameter("test-param", 123)
	paramValue, exists := policyCtx.GetParameter("test-param")
	fmt.Printf("Parameter 'test-param' exists: %v, value: %v\n", exists, paramValue)

	// Create a policy engine
	engine := policy.NewPolicyEngine()
	fmt.Println("Policy engine created successfully")

	fmt.Println("Basic test completed successfully!")
}
