package placement

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/sirupsen/logrus"
)

func TestDefaultPlacementEngine_PlaceVM(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.DebugLevel)
	
	engine := NewDefaultPlacementEngine(logger)
	
	request := &PlacementRequest{
		VMID: "test-vm-1",
		VMSpec: VMSpec{
			CPU:     2,
			Memory:  4096,
			Storage: 50,
		},
		Strategy: StrategyBalanced,
	}
	
	ctx := context.Background()
	decision, err := engine.PlaceVM(ctx, request)
	
	require.NoError(t, err)
	assert.NotNil(t, decision)
	assert.Equal(t, "test-vm-1", decision.VMID)
	assert.NotEmpty(t, decision.SelectedNode)
	assert.Greater(t, decision.Score, 0.0)
	assert.Greater(t, decision.Confidence, 0.0)
	assert.NotEmpty(t, decision.Explanation)
}

func TestDefaultPlacementEngine_PlaceVM_InsufficientResources(t *testing.T) {
	logger := logrus.New()
	logger.SetLevel(logrus.DebugLevel)
	
	engine := NewDefaultPlacementEngine(logger)
	
	// Request more resources than any node has
	request := &PlacementRequest{
		VMID: "test-vm-large",
		VMSpec: VMSpec{
			CPU:     100, // More than available
			Memory:  1000000,
			Storage: 10000,
		},
		Strategy: StrategyBalanced,
	}
	
	ctx := context.Background()
	_, err := engine.PlaceVM(ctx, request)
	
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no feasible nodes found")
}

func TestDefaultPlacementEngine_ValidateConstraints(t *testing.T) {
	logger := logrus.New()
	engine := NewDefaultPlacementEngine(logger)
	
	t.Run("valid constraints", func(t *testing.T) {
		request := &PlacementRequest{
			VMID: "test-vm",
			VMSpec: VMSpec{
				CPU:     1,
				Memory:  1024,
				Storage: 20,
			},
			Constraints: []Constraint{
				{
					Type:        ConstraintTypeResourceLimit,
					Enforcement: EnforcementSoft,
					Parameters:  map[string]interface{}{"max_cpu_utilization": 0.8},
				},
			},
		}
		
		err := engine.ValidateConstraints(context.Background(), request)
		assert.NoError(t, err)
	})
	
	t.Run("invalid constraint parameters", func(t *testing.T) {
		request := &PlacementRequest{
			VMID: "test-vm",
			VMSpec: VMSpec{
				CPU:     1,
				Memory:  1024,
				Storage: 20,
			},
			Constraints: []Constraint{
				{
					Type:        ConstraintTypeResourceLimit,
					Enforcement: EnforcementHard,
					Parameters:  nil, // Missing parameters
				},
			},
		}
		
		err := engine.ValidateConstraints(context.Background(), request)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "missing parameters")
	})
}

func TestDefaultPlacementEngine_AlgorithmSwitching(t *testing.T) {
	logger := logrus.New()
	engine := NewDefaultPlacementEngine(logger)
	
	// Default should be bin packing
	assert.Equal(t, AlgorithmBinPacking, engine.GetAlgorithm())
	
	// Switch to load balancing
	engine.SetAlgorithm(AlgorithmLoadBalancing)
	assert.Equal(t, AlgorithmLoadBalancing, engine.GetAlgorithm())
	
	// Switch to heuristic
	engine.SetAlgorithm(AlgorithmHeuristic)
	assert.Equal(t, AlgorithmHeuristic, engine.GetAlgorithm())
}

func TestBinPackingScorer_ScoreNodes(t *testing.T) {
	logger := logrus.New()
	scorer := NewBinPackingScorer(logger)
	
	nodes := []*Node{
		{
			ID:    "node-high-util",
			State: NodeStateReady,
			Capacity: NodeCapacity{
				CPU:    8,
				Memory: 16384,
				Storage: 500,
			},
			Available: NodeCapacity{
				CPU:    2, // High utilization
				Memory: 4096,
				Storage: 100,
			},
			Health: 0.95,
		},
		{
			ID:    "node-low-util",
			State: NodeStateReady,
			Capacity: NodeCapacity{
				CPU:    16,
				Memory: 32768,
				Storage: 1000,
			},
			Available: NodeCapacity{
				CPU:    14, // Low utilization
				Memory: 28672,
				Storage: 800,
			},
			Health: 0.98,
		},
	}
	
	request := &PlacementRequest{
		VMID: "test-vm",
		VMSpec: VMSpec{
			CPU:     2,
			Memory:  4096,
			Storage: 100,
		},
		Strategy: StrategyConsolidated,
	}
	
	scores, err := scorer.ScoreNodes(context.Background(), request, nodes)
	require.NoError(t, err)
	assert.Len(t, scores, 2)
	
	// Both should be feasible
	assert.True(t, scores[0].Feasible)
	assert.True(t, scores[1].Feasible)
	
	// For bin packing, higher utilization node should score better
	// (assuming the high-util node can still fit the VM)
	highUtilScore := scores[0].Score
	lowUtilScore := scores[1].Score
	
	// Verify scores are reasonable
	assert.Greater(t, highUtilScore, 0.0)
	assert.Greater(t, lowUtilScore, 0.0)
}

func TestLoadBalancingScorer_ScoreNodes(t *testing.T) {
	logger := logrus.New()
	scorer := NewLoadBalancingScorer(logger)
	
	nodes := []*Node{
		{
			ID:    "node-high-util",
			State: NodeStateReady,
			Capacity: NodeCapacity{
				CPU:    8,
				Memory: 16384,
				Storage: 500,
			},
			Available: NodeCapacity{
				CPU:    2, // High utilization
				Memory: 4096,
				Storage: 100,
			},
			Health: 0.95,
		},
		{
			ID:    "node-low-util",
			State: NodeStateReady,
			Capacity: NodeCapacity{
				CPU:    16,
				Memory: 32768,
				Storage: 1000,
			},
			Available: NodeCapacity{
				CPU:    14, // Low utilization
				Memory: 28672,
				Storage: 800,
			},
			Health: 0.98,
		},
	}
	
	request := &PlacementRequest{
		VMID: "test-vm",
		VMSpec: VMSpec{
			CPU:     2,
			Memory:  4096,
			Storage: 100,
		},
		Strategy: StrategyBalanced,
	}
	
	scores, err := scorer.ScoreNodes(context.Background(), request, nodes)
	require.NoError(t, err)
	assert.Len(t, scores, 2)
	
	// Both should be feasible
	assert.True(t, scores[0].Feasible)
	assert.True(t, scores[1].Feasible)
	
	// For load balancing, lower utilization node should score better
	highUtilScore := scores[0].Score
	lowUtilScore := scores[1].Score
	
	assert.Greater(t, lowUtilScore, highUtilScore, "Load balancing should prefer lower utilization")
}

func TestHeuristicScorer_ScoreNodes(t *testing.T) {
	logger := logrus.New()
	scorer := NewHeuristicScorer(logger)
	
	nodes := []*Node{
		{
			ID:    "node-expensive",
			State: NodeStateReady,
			Capacity: NodeCapacity{
				CPU:    8,
				Memory: 16384,
				Storage: 500,
			},
			Available: NodeCapacity{
				CPU:    6,
				Memory: 12288,
				Storage: 400,
			},
			Health: 0.95,
			Cost:   0.5, // High cost
		},
		{
			ID:    "node-cheap",
			State: NodeStateReady,
			Capacity: NodeCapacity{
				CPU:    8,
				Memory: 16384,
				Storage: 500,
			},
			Available: NodeCapacity{
				CPU:    6,
				Memory: 12288,
				Storage: 400,
			},
			Health: 0.95,
			Cost:   0.1, // Low cost
		},
	}
	
	request := &PlacementRequest{
		VMID: "test-vm",
		VMSpec: VMSpec{
			CPU:     2,
			Memory:  4096,
			Storage: 100,
		},
		Strategy: StrategyCostOptimized,
	}
	
	scores, err := scorer.ScoreNodes(context.Background(), request, nodes)
	require.NoError(t, err)
	assert.Len(t, scores, 2)
	
	// Both should be feasible
	assert.True(t, scores[0].Feasible)
	assert.True(t, scores[1].Feasible)
	
	// For cost-optimized heuristic, cheaper node should score better
	expensiveScore := scores[0].Score
	cheapScore := scores[1].Score
	
	assert.Greater(t, cheapScore, expensiveScore, "Heuristic with cost optimization should prefer cheaper nodes")
}

func TestPlacementRequest_WithConstraints(t *testing.T) {
	logger := logrus.New()
	engine := NewDefaultPlacementEngine(logger)
	
	request := &PlacementRequest{
		VMID: "test-vm-constrained",
		VMSpec: VMSpec{
			CPU:     2,
			Memory:  4096,
			Storage: 50,
		},
		Constraints: []Constraint{
			{
				Type:        ConstraintTypeResourceLimit,
				Enforcement: EnforcementSoft,
				Parameters: map[string]interface{}{
					"max_cpu_utilization": 0.7,
				},
				Weight: 1.0,
			},
			{
				Type:        ConstraintTypeAvailability,
				Enforcement: EnforcementHard,
				Parameters: map[string]interface{}{
					"min_health": 0.9,
				},
			},
		},
		Strategy: StrategyBalanced,
	}
	
	ctx := context.Background()
	decision, err := engine.PlaceVM(ctx, request)
	
	require.NoError(t, err)
	assert.NotNil(t, decision)
	assert.Equal(t, "test-vm-constrained", decision.VMID)
	
	// Verify metadata includes constraint information
	assert.Contains(t, decision.Metadata, "algorithm")
	assert.Contains(t, decision.Metadata, "strategy")
}

func TestPlacementRequest_WithPreferences(t *testing.T) {
	logger := logrus.New()
	engine := NewDefaultPlacementEngine(logger)
	
	request := &PlacementRequest{
		VMID: "test-vm-preferences",
		VMSpec: VMSpec{
			CPU:     1,
			Memory:  2048,
			Storage: 25,
		},
		Preferences: []Preference{
			{
				Type:   PreferenceTypeLowCost,
				Weight: 2.0,
			},
			{
				Type:   PreferenceTypeHighBandwidth,
				Weight: 1.0,
			},
		},
		Strategy: StrategyCostOptimized,
	}
	
	ctx := context.Background()
	decision, err := engine.PlaceVM(ctx, request)
	
	require.NoError(t, err)
	assert.NotNil(t, decision)
	assert.Equal(t, "test-vm-preferences", decision.VMID)
}

func TestPlacementRequest_WithGPURequirements(t *testing.T) {
	logger := logrus.New()
	engine := NewDefaultPlacementEngine(logger)
	
	request := &PlacementRequest{
		VMID: "test-vm-gpu",
		VMSpec: VMSpec{
			CPU:     4,
			Memory:  8192,
			Storage: 100,
			GPU: &GPURequirements{
				Count:    1,
				Model:    "nvidia-tesla-v100",
				MemoryMB: 16384,
			},
		},
		Strategy: StrategyPerformance,
	}
	
	ctx := context.Background()
	_, err := engine.PlaceVM(ctx, request)
	
	// Should fail since mock nodes don't have GPUs
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no feasible nodes found")
}

func TestNodeScore_Breakdown(t *testing.T) {
	logger := logrus.New()
	scorer := NewBinPackingScorer(logger)
	
	node := &Node{
		ID:    "test-node",
		State: NodeStateReady,
		Capacity: NodeCapacity{
			CPU:    8,
			Memory: 16384,
			Storage: 500,
		},
		Available: NodeCapacity{
			CPU:    4,
			Memory: 8192,
			Storage: 250,
		},
		Health: 0.95,
		Cost:   0.15,
	}
	
	request := &PlacementRequest{
		VMID: "test-vm",
		VMSpec: VMSpec{
			CPU:     2,
			Memory:  4096,
			Storage: 100,
		},
	}
	
	scores, err := scorer.ScoreNodes(context.Background(), request, []*Node{node})
	require.NoError(t, err)
	assert.Len(t, scores, 1)
	
	score := scores[0]
	assert.True(t, score.Feasible)
	
	// Verify breakdown components
	breakdown := score.Breakdown
	assert.Greater(t, breakdown.ResourceScore, 0.0)
	assert.Equal(t, 0.95, breakdown.HealthScore)
	assert.Greater(t, breakdown.ConstraintScore, 0.0)
	assert.Greater(t, breakdown.CostScore, 0.0)
}

func TestPlacementDecision_Alternatives(t *testing.T) {
	logger := logrus.New()
	engine := NewDefaultPlacementEngine(logger)
	
	request := &PlacementRequest{
		VMID: "test-vm-alternatives",
		VMSpec: VMSpec{
			CPU:     1,
			Memory:  1024,
			Storage: 20,
		},
		Strategy: StrategyBalanced,
	}
	
	ctx := context.Background()
	decision, err := engine.PlaceVM(ctx, request)
	
	require.NoError(t, err)
	assert.NotNil(t, decision)
	
	// Should have alternatives since we have multiple mock nodes
	assert.NotEmpty(t, decision.Alternatives)
	
	// Each alternative should have required fields
	for _, alt := range decision.Alternatives {
		assert.NotEmpty(t, alt.NodeID)
		assert.Greater(t, alt.Score, 0.0)
		assert.NotEmpty(t, alt.Explanation)
	}
}

func TestPlacementEngine_Confidence(t *testing.T) {
	logger := logrus.New()
	engine := NewDefaultPlacementEngine(logger)
	
	request := &PlacementRequest{
		VMID: "test-vm-confidence",
		VMSpec: VMSpec{
			CPU:     1,
			Memory:  1024,
			Storage: 20,
		},
		Strategy: StrategyBalanced,
	}
	
	ctx := context.Background()
	decision, err := engine.PlaceVM(ctx, request)
	
	require.NoError(t, err)
	assert.NotNil(t, decision)
	
	// Confidence should be between 0.5 and 1.0 for multiple feasible nodes
	assert.GreaterOrEqual(t, decision.Confidence, 0.5)
	assert.LessOrEqual(t, decision.Confidence, 1.0)
}

func TestNode_HealthCheck(t *testing.T) {
	nodes := []*Node{
		{
			ID:          "healthy-node",
			State:       NodeStateReady,
			Health:      0.95,
			Capacity: NodeCapacity{
				CPU:     8,
				Memory:  16384,
				Storage: 500,
			},
			Available: NodeCapacity{
				CPU:     6,
				Memory:  12288,
				Storage: 400,
			},
			LastUpdated: time.Now(),
		},
		{
			ID:          "unhealthy-node",
			State:       NodeStateNotReady,
			Health:      0.3,
			Capacity: NodeCapacity{
				CPU:     8,
				Memory:  16384,
				Storage: 500,
			},
			Available: NodeCapacity{
				CPU:     6,
				Memory:  12288,
				Storage: 400,
			},
			LastUpdated: time.Now().Add(-time.Hour),
		},
		{
			ID:          "maintenance-node",
			State:       NodeStateMaintenance,
			Health:      0.0,
			Capacity: NodeCapacity{
				CPU:     8,
				Memory:  16384,
				Storage: 500,
			},
			Available: NodeCapacity{
				CPU:     6,
				Memory:  12288,
				Storage: 400,
			},
			LastUpdated: time.Now(),
		},
	}
	
	logger := logrus.New()
	scorer := NewBinPackingScorer(logger)
	
	request := &PlacementRequest{
		VMID: "test-vm",
		VMSpec: VMSpec{
			CPU:     1,
			Memory:  1024,
			Storage: 20,
		},
	}
	
	scores, err := scorer.ScoreNodes(context.Background(), request, nodes)
	require.NoError(t, err)
	assert.Len(t, scores, 3)
	
	// Only healthy node should be feasible
	assert.True(t, scores[0].Feasible, "Healthy node should be feasible")
	assert.False(t, scores[1].Feasible, "Unhealthy node should not be feasible")
	assert.False(t, scores[2].Feasible, "Maintenance node should not be feasible")
}