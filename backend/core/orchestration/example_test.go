package orchestration_test

import (
	"context"
	"testing"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/placement"
)

// Example demonstrates how to use the orchestration engine
func ExampleOrchestrationEngine() {
	logger := logrus.New()
	logger.SetLevel(logrus.InfoLevel)

	// Create orchestration engine
	engine := orchestration.NewDefaultOrchestrationEngine(logger)

	// Create a sample policy
	policy := &orchestration.OrchestrationPolicy{
		ID:          "production-policy",
		Name:        "Production Workload Policy",
		Description: "Policy for production workloads with high availability requirements",
		Selector: orchestration.PolicySelector{
			Labels: map[string]string{
				"environment": "production",
			},
		},
		Rules: []orchestration.PolicyRule{
			{
				Type:    orchestration.RuleTypePlacement,
				Priority: 10,
				Enabled:  true,
				Parameters: map[string]interface{}{
					"max_cpu_utilization":    0.7,
					"max_memory_utilization": 0.8,
					"min_health":             0.9,
				},
				Actions: []orchestration.RuleAction{
					{
						Type: orchestration.ActionTypeSchedule,
						Parameters: map[string]interface{}{
							"strategy": "balanced",
						},
					},
				},
			},
		},
		Priority:  10,
		Enabled:   true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Register the policy
	err := engine.RegisterPolicy(policy)
	if err != nil {
		logger.WithError(err).Error("Failed to register policy")
		return
	}

	// Create a VM specification for placement
	vmSpec := placement.VMSpec{
		CPU:     4,
		Memory:  8192, // 8GB
		Storage: 100,  // 100GB
		Labels: map[string]string{
			"environment": "production",
			"app":         "web-server",
			"vm_id":       "vm-web-001",
		},
	}

	// Make placement decision
	decision, err := engine.MakeVMPlacementDecision(context.Background(), vmSpec, placement.StrategyBalanced)
	if err != nil {
		logger.WithError(err).Error("Failed to make placement decision")
		return
	}

	logger.WithFields(logrus.Fields{
		"vm_id":         decision.Context.VMID,
		"selected_node": decision.Actions[0].Target,
		"score":         decision.Score,
		"confidence":    decision.Confidence,
	}).Info("Placement decision made")
}

func TestOrchestrationEngine_Basic(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel) // Reduce noise in tests

	engine := orchestration.NewDefaultOrchestrationEngine(logger)

	// Test engine status
	status := engine.GetStatus()
	assert.Equal(t, orchestration.EngineStateStopped, status.State)
	assert.Equal(t, 0, status.ActivePolicies)

	// Test policy registration
	policy := &orchestration.OrchestrationPolicy{
		ID:          "test-policy",
		Name:        "Test Policy",
		Description: "A test policy",
		Selector: orchestration.PolicySelector{
			Labels: map[string]string{"test": "true"},
		},
		Rules: []orchestration.PolicyRule{
			{
				Type:     orchestration.RuleTypePlacement,
				Priority: 1,
				Enabled:  true,
				Parameters: map[string]interface{}{
					"max_cpu_utilization": 0.8,
				},
			},
		},
		Priority:  1,
		Enabled:   true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	err := engine.RegisterPolicy(policy)
	require.NoError(t, err)

	// Check that policy was registered
	status = engine.GetStatus()
	assert.Equal(t, 1, status.ActivePolicies)

	// Test policy unregistration
	err = engine.UnregisterPolicy("test-policy")
	require.NoError(t, err)

	status = engine.GetStatus()
	assert.Equal(t, 0, status.ActivePolicies)
}

func TestOrchestrationEngine_PlacementDecision(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	engine := orchestration.NewDefaultOrchestrationEngine(logger)

	// Test VM placement decision
	vmSpec := placement.VMSpec{
		CPU:     2,
		Memory:  4096,
		Storage: 50,
		Labels: map[string]string{
			"vm_id": "test-vm-001",
			"app":   "test-app",
		},
	}

	decision, err := engine.MakeVMPlacementDecision(context.Background(), vmSpec, placement.StrategyBalanced)
	require.NoError(t, err)
	assert.NotNil(t, decision)

	assert.Equal(t, orchestration.DecisionTypePlacement, decision.DecisionType)
	assert.NotEmpty(t, decision.Recommendation)
	assert.Greater(t, decision.Score, 0.0)
	assert.Greater(t, decision.Confidence, 0.0)
	assert.Len(t, decision.Actions, 1)
	assert.Equal(t, orchestration.ActionTypeSchedule, decision.Actions[0].Type)
}

func TestOrchestrationEngine_PolicyMatching(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.ErrorLevel)

	engine := orchestration.NewDefaultOrchestrationEngine(logger)

	// Register a policy that matches specific labels
	policy := &orchestration.OrchestrationPolicy{
		ID:          "gpu-policy",
		Name:        "GPU Workload Policy",
		Description: "Policy for GPU-intensive workloads",
		Selector: orchestration.PolicySelector{
			Labels: map[string]string{
				"workload": "gpu",
			},
		},
		Rules: []orchestration.PolicyRule{
			{
				Type:     orchestration.RuleTypePlacement,
				Priority: 5,
				Enabled:  true,
				Parameters: map[string]interface{}{
					"requires_gpu": true,
				},
			},
		},
		Priority:  5,
		Enabled:   true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	err := engine.RegisterPolicy(policy)
	require.NoError(t, err)

	// Test VM spec that matches the policy
	vmSpec := placement.VMSpec{
		CPU:     8,
		Memory:  16384,
		Storage: 200,
		GPU: &placement.GPURequirements{
			Count:    1,
			Model:    "nvidia-tesla-v100",
			MemoryMB: 16384,
		},
		Labels: map[string]string{
			"vm_id":    "gpu-vm-001",
			"workload": "gpu",
			"app":      "ml-training",
		},
	}

	// This will fail because mock nodes don't have GPUs, but we can test the policy matching
	_, err = engine.MakeVMPlacementDecision(context.Background(), vmSpec, placement.StrategyPerformance)
	assert.Error(t, err) // Expected to fail due to no GPU nodes
	assert.Contains(t, err.Error(), "no feasible nodes found")
}