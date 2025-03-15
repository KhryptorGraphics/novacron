# Policy Subsystem

This package contains the VM scheduling policy framework for novacron. It provides a flexible way to define and enforce policies for VM placement, migration, and resource allocation.

## Components

- **EvaluationContext**: Base context for policy evaluation (policy_context.go)
- **PolicyEvaluationContext**: Extended context with policy-specific functionality (policy_context.go)
- **Expression**: Interface for policy expressions that can be evaluated (expression.go)
- **PolicyEngine**: Manages and evaluates scheduling policies (policy_engine.go)
- **PolicyParser**: Parses policy definitions in DSL format (policy_language.go)

## Usage Example

```go
// Create a policy engine
engine := policy.NewPolicyEngine()

// Parse a policy definition
policyDef := `policy "high-performance-gpu" {
    type = "placement"
    description = "Prioritize GPU workloads on high-performance nodes"
    
    rules {
        rule "require-gpu-capability" {
            hard_constraint = true
            when { 
                vm.labels["workload-type"] == "gpu" 
            }
            then {
                filter "Node does not have GPU capability"
            }
        }
    }
}`

// Parse the policy
parser := policy.NewPolicyParser()
gpuPolicy, err := parser.ParsePolicy(policyDef, "admin")
if err != nil {
    log.Fatalf("Error parsing policy: %v", err)
}

// Register the policy
if err := engine.RegisterPolicy(gpuPolicy); err != nil {
    log.Fatalf("Error registering policy: %v", err)
}

// Activate the policy with specific configuration
config := &policy.PolicyConfiguration{
    PolicyID: gpuPolicy.ID,
    Priority: 100,
    ParameterValues: map[string]interface{}{
        "gpu_score_weight": 15.0,
    },
}
if err := engine.ActivatePolicy(gpuPolicy.ID, config); err != nil {
    log.Fatalf("Error activating policy: %v", err)
}

// Evaluate policies for a VM against a set of candidate nodes
filteredNodes, err := engine.EvaluatePlacementPolicies(context.Background(), vm, candidateNodes)
if err != nil {
    log.Fatalf("Error evaluating policies: %v", err)
}
```

## Running Examples

Examples are available in the `backend/examples/policy` directory. To run them, make sure your Go module setup correctly references the novacron packages.

```
cd backend/examples/policy
go run policy_examples.go
```

For module issues, ensure the replace directive in `backend/examples/policy/go.mod` is pointing to the correct location:

```
replace github.com/khryptorgraphics/novacron => ../../..
