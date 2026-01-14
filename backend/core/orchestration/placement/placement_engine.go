package placement

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/sirupsen/logrus"
)

// PlacementEngine defines the interface for VM placement decisions
type PlacementEngine interface {
	// PlaceVM determines the best node for VM placement
	PlaceVM(ctx context.Context, request *PlacementRequest) (*PlacementDecision, error)
	
	// EvaluateNodes scores nodes for a given VM placement request
	EvaluateNodes(ctx context.Context, request *PlacementRequest, nodes []*Node) ([]*NodeScore, error)
	
	// GetAlgorithm returns the current placement algorithm
	GetAlgorithm() PlacementAlgorithm
	
	// SetAlgorithm changes the placement algorithm
	SetAlgorithm(algorithm PlacementAlgorithm)
	
	// ValidateConstraints checks if constraints can be satisfied
	ValidateConstraints(ctx context.Context, request *PlacementRequest) error
}

// PlacementRequest represents a request to place a VM
type PlacementRequest struct {
	VMID         string                         `json:"vm_id"`
	VMSpec       VMSpec                         `json:"vm_spec"`
	Constraints  []Constraint     `json:"constraints"`
	Preferences  []Preference                   `json:"preferences"`
	Strategy     PlacementStrategy              `json:"strategy"`
	ExcludeNodes []string                       `json:"exclude_nodes,omitempty"`
	Context      map[string]interface{}         `json:"context,omitempty"`
}

// VMSpec defines the resource requirements for a VM
type VMSpec struct {
	CPU        int                    `json:"cpu_cores"`
	Memory     int64                  `json:"memory_mb"`
	Storage    int64                  `json:"storage_gb"`
	Network    NetworkRequirements    `json:"network,omitempty"`
	GPU        *GPURequirements       `json:"gpu,omitempty"`
	Labels     map[string]string      `json:"labels,omitempty"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// NetworkRequirements defines network requirements for a VM
type NetworkRequirements struct {
	BandwidthMbps  int      `json:"bandwidth_mbps,omitempty"`
	LatencyMs      int      `json:"latency_ms,omitempty"`
	Networks       []string `json:"networks,omitempty"`
	PublicIP       bool     `json:"public_ip,omitempty"`
}

// GPURequirements defines GPU requirements for a VM
type GPURequirements struct {
	Count     int      `json:"count"`
	Model     string   `json:"model,omitempty"`
	MemoryMB  int64    `json:"memory_mb,omitempty"`
	Shared    bool     `json:"shared,omitempty"`
}

// Preference defines a soft constraint with weight
type Preference struct {
	Type       PreferenceType         `json:"type"`
	Weight     float64                `json:"weight"`
	Parameters map[string]interface{} `json:"parameters"`
}

// PreferenceType defines the type of preference
type PreferenceType string

const (
	PreferenceTypeLowLatency    PreferenceType = "low_latency"
	PreferenceTypeHighBandwidth PreferenceType = "high_bandwidth"
	PreferenceTypeLowCost       PreferenceType = "low_cost"
	PreferenceTypeLocalStorage  PreferenceType = "local_storage"
	PreferenceTypeGPUOptimized  PreferenceType = "gpu_optimized"
)

// PlacementStrategy defines the overall placement strategy
type PlacementStrategy string

const (
	StrategyBalanced     PlacementStrategy = "balanced"
	StrategyConsolidated PlacementStrategy = "consolidated"
	StrategyPerformance  PlacementStrategy = "performance"
	StrategyEfficiency   PlacementStrategy = "efficiency"
	StrategyCostOptimized PlacementStrategy = "cost_optimized"
)

// PlacementDecision represents the result of a placement decision
type PlacementDecision struct {
	RequestID    string                 `json:"request_id"`
	VMID         string                 `json:"vm_id"`
	SelectedNode string                 `json:"selected_node"`
	Score        float64                `json:"score"`
	Confidence   float64                `json:"confidence"`
	Explanation  string                 `json:"explanation"`
	Alternatives []Alternative          `json:"alternatives"`
	Timestamp    time.Time              `json:"timestamp"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// Alternative represents an alternative placement option
type Alternative struct {
	NodeID      string  `json:"node_id"`
	Score       float64 `json:"score"`
	Explanation string  `json:"explanation"`
}

// Node represents a compute node for placement decisions
type Node struct {
	ID           string                 `json:"id"`
	Name         string                 `json:"name"`
	Region       string                 `json:"region,omitempty"`
	Zone         string                 `json:"zone,omitempty"`
	Rack         string                 `json:"rack,omitempty"`
	Capacity     NodeCapacity           `json:"capacity"`
	Available    NodeCapacity           `json:"available"`
	State        NodeState              `json:"state"`
	Health       float64                `json:"health"`       // 0.0 to 1.0
	Cost         float64                `json:"cost,omitempty"` // Cost per hour
	Labels       map[string]string      `json:"labels,omitempty"`
	Taints       []Taint                `json:"taints,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
	LastUpdated  time.Time              `json:"last_updated"`
}

// NodeCapacity defines the capacity of a node
type NodeCapacity struct {
	CPU          int   `json:"cpu_cores"`
	Memory       int64 `json:"memory_mb"`
	Storage      int64 `json:"storage_gb"`
	NetworkMbps  int   `json:"network_mbps"`
	GPUs         []GPU `json:"gpus,omitempty"`
}

// GPU represents a GPU device
type GPU struct {
	ID       string `json:"id"`
	Model    string `json:"model"`
	MemoryMB int64  `json:"memory_mb"`
	InUse    bool   `json:"in_use"`
}

// NodeState represents the state of a node
type NodeState string

const (
	NodeStateReady       NodeState = "ready"
	NodeStateNotReady    NodeState = "not_ready"
	NodeStateMaintenance NodeState = "maintenance"
	NodeStateTerminating NodeState = "terminating"
)

// Taint represents a node taint
type Taint struct {
	Key    string      `json:"key"`
	Value  string      `json:"value"`
	Effect TaintEffect `json:"effect"`
}

// TaintEffect defines the effect of a taint
type TaintEffect string

const (
	TaintEffectNoSchedule       TaintEffect = "NoSchedule"
	TaintEffectPreferNoSchedule TaintEffect = "PreferNoSchedule"
	TaintEffectNoExecute        TaintEffect = "NoExecute"
)

// NodeScore represents a node with its placement score
type NodeScore struct {
	Node        *Node   `json:"node"`
	Score       float64 `json:"score"`
	Breakdown   ScoreBreakdown `json:"breakdown"`
	Feasible    bool    `json:"feasible"`
	Reason      string  `json:"reason,omitempty"`
}

// ScoreBreakdown provides detailed scoring information
type ScoreBreakdown struct {
	ResourceScore    float64 `json:"resource_score"`
	ConstraintScore  float64 `json:"constraint_score"`
	PreferenceScore  float64 `json:"preference_score"`
	HealthScore      float64 `json:"health_score"`
	CostScore        float64 `json:"cost_score,omitempty"`
	LocationScore    float64 `json:"location_score,omitempty"`
}

// PlacementAlgorithm defines the placement algorithm type
type PlacementAlgorithm string

const (
	AlgorithmBinPacking     PlacementAlgorithm = "bin_packing"
	AlgorithmLoadBalancing  PlacementAlgorithm = "load_balancing"
	AlgorithmConstraintSolver PlacementAlgorithm = "constraint_solver"
	AlgorithmHeuristic      PlacementAlgorithm = "heuristic"
)

// DefaultPlacementEngine implements the PlacementEngine interface
type DefaultPlacementEngine struct {
	algorithm PlacementAlgorithm
	logger    *logrus.Logger
	scorers   map[PlacementAlgorithm]NodeScorer
}

// NodeScorer defines the interface for scoring nodes
type NodeScorer interface {
	ScoreNodes(ctx context.Context, request *PlacementRequest, nodes []*Node) ([]*NodeScore, error)
}

// NewDefaultPlacementEngine creates a new default placement engine
func NewDefaultPlacementEngine(logger *logrus.Logger) *DefaultPlacementEngine {
	engine := &DefaultPlacementEngine{
		algorithm: AlgorithmBinPacking,
		logger:    logger,
		scorers:   make(map[PlacementAlgorithm]NodeScorer),
	}

	// Register default scorers
	engine.scorers[AlgorithmBinPacking] = NewBinPackingScorer(logger)
	engine.scorers[AlgorithmLoadBalancing] = NewLoadBalancingScorer(logger)
	engine.scorers[AlgorithmHeuristic] = NewHeuristicScorer(logger)

	return engine
}

// PlaceVM implements PlacementEngine interface
func (e *DefaultPlacementEngine) PlaceVM(ctx context.Context, request *PlacementRequest) (*PlacementDecision, error) {
	if request == nil {
		return nil, fmt.Errorf("placement request is nil")
	}

	e.logger.WithFields(logrus.Fields{
		"vm_id":     request.VMID,
		"algorithm": e.algorithm,
		"strategy":  request.Strategy,
	}).Info("Evaluating VM placement")

	// Validate constraints first
	if err := e.ValidateConstraints(ctx, request); err != nil {
		return nil, fmt.Errorf("constraint validation failed: %w", err)
	}

	// Get available nodes (this would normally come from node manager)
	nodes, err := e.getAvailableNodes(ctx, request)
	if err != nil {
		return nil, fmt.Errorf("failed to get available nodes: %w", err)
	}

	if len(nodes) == 0 {
		return nil, fmt.Errorf("no available nodes found")
	}

	// Score nodes using the current algorithm
	scores, err := e.EvaluateNodes(ctx, request, nodes)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate nodes: %w", err)
	}

	// Filter out infeasible nodes
	feasibleScores := make([]*NodeScore, 0)
	for _, score := range scores {
		if score.Feasible {
			feasibleScores = append(feasibleScores, score)
		}
	}

	if len(feasibleScores) == 0 {
		return nil, fmt.Errorf("no feasible nodes found for VM placement")
	}

	// Sort by score (highest first)
	sort.Slice(feasibleScores, func(i, j int) bool {
		return feasibleScores[i].Score > feasibleScores[j].Score
	})

	// Select the best node
	bestScore := feasibleScores[0]
	
	// Build alternatives
	alternatives := make([]Alternative, 0)
	maxAlternatives := 3
	for i := 1; i < len(feasibleScores) && i <= maxAlternatives; i++ {
		score := feasibleScores[i]
		alternatives = append(alternatives, Alternative{
			NodeID:      score.Node.ID,
			Score:       score.Score,
			Explanation: e.generateExplanation(score),
		})
	}

	// Create placement decision
	decision := &PlacementDecision{
		RequestID:    fmt.Sprintf("placement-%d", time.Now().UnixNano()),
		VMID:         request.VMID,
		SelectedNode: bestScore.Node.ID,
		Score:        bestScore.Score,
		Confidence:   e.calculateConfidence(bestScore, feasibleScores),
		Explanation:  e.generateExplanation(bestScore),
		Alternatives: alternatives,
		Timestamp:    time.Now(),
		Metadata: map[string]interface{}{
			"algorithm":      e.algorithm,
			"strategy":       request.Strategy,
			"nodes_evaluated": len(nodes),
			"feasible_nodes": len(feasibleScores),
		},
	}

	e.logger.WithFields(logrus.Fields{
		"vm_id":       request.VMID,
		"selected_node": bestScore.Node.ID,
		"score":       bestScore.Score,
		"confidence":  decision.Confidence,
	}).Info("VM placement decision made")

	return decision, nil
}

// EvaluateNodes implements PlacementEngine interface
func (e *DefaultPlacementEngine) EvaluateNodes(ctx context.Context, request *PlacementRequest, nodes []*Node) ([]*NodeScore, error) {
	scorer, exists := e.scorers[e.algorithm]
	if !exists {
		return nil, fmt.Errorf("no scorer found for algorithm: %s", e.algorithm)
	}

	return scorer.ScoreNodes(ctx, request, nodes)
}

// GetAlgorithm implements PlacementEngine interface
func (e *DefaultPlacementEngine) GetAlgorithm() PlacementAlgorithm {
	return e.algorithm
}

// SetAlgorithm implements PlacementEngine interface
func (e *DefaultPlacementEngine) SetAlgorithm(algorithm PlacementAlgorithm) {
	e.algorithm = algorithm
	e.logger.WithField("algorithm", algorithm).Info("Placement algorithm changed")
}

// ValidateConstraints implements PlacementEngine interface
func (e *DefaultPlacementEngine) ValidateConstraints(ctx context.Context, request *PlacementRequest) error {
	for _, constraint := range request.Constraints {
		if err := e.validateConstraint(constraint); err != nil {
			return fmt.Errorf("invalid constraint %s: %w", constraint.Type, err)
		}
	}
	return nil
}

// validateConstraint validates a single constraint
func (e *DefaultPlacementEngine) validateConstraint(constraint Constraint) error {
	switch constraint.Type {
	case ConstraintTypeResourceLimit:
		// Validate resource limit parameters
		if constraint.Parameters == nil {
			return fmt.Errorf("resource limit constraint missing parameters")
		}
	case ConstraintTypeAffinity:
		// Validate affinity parameters
		if constraint.Parameters == nil {
			return fmt.Errorf("affinity constraint missing parameters")
		}
	case ConstraintTypeAntiAffinity:
		// Validate anti-affinity parameters
		if constraint.Parameters == nil {
			return fmt.Errorf("anti-affinity constraint missing parameters")
		}
	}
	return nil
}

// getAvailableNodes retrieves available nodes (mock implementation)
func (e *DefaultPlacementEngine) getAvailableNodes(ctx context.Context, request *PlacementRequest) ([]*Node, error) {
	// In a real implementation, this would query the node manager
	// For now, return mock nodes for testing
	return []*Node{
		{
			ID:     "node-1",
			Name:   "Node 1",
			Region: "us-east-1",
			Zone:   "us-east-1a",
			Capacity: NodeCapacity{
				CPU:         16,
				Memory:      32768,
				Storage:     1000,
				NetworkMbps: 10000,
			},
			Available: NodeCapacity{
				CPU:         12,
				Memory:      24576,
				Storage:     800,
				NetworkMbps: 8000,
			},
			State:       NodeStateReady,
			Health:      0.95,
			Cost:        0.10,
			LastUpdated: time.Now(),
		},
		{
			ID:     "node-2",
			Name:   "Node 2",
			Region: "us-east-1",
			Zone:   "us-east-1b",
			Capacity: NodeCapacity{
				CPU:         32,
				Memory:      65536,
				Storage:     2000,
				NetworkMbps: 20000,
			},
			Available: NodeCapacity{
				CPU:         28,
				Memory:      49152,
				Storage:     1500,
				NetworkMbps: 15000,
			},
			State:       NodeStateReady,
			Health:      0.98,
			Cost:        0.20,
			LastUpdated: time.Now(),
		},
	}, nil
}

// calculateConfidence calculates confidence in the placement decision
func (e *DefaultPlacementEngine) calculateConfidence(bestScore *NodeScore, allScores []*NodeScore) float64 {
	if len(allScores) == 1 {
		return 1.0 // Only one option, full confidence
	}

	// Calculate confidence based on score gap to second best
	secondBest := allScores[1]
	scoreGap := bestScore.Score - secondBest.Score
	
	// Normalize confidence to 0.5-1.0 range
	confidence := 0.5 + (scoreGap * 0.5)
	
	// Cap at 1.0
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

// generateExplanation generates a human-readable explanation for the placement decision
func (e *DefaultPlacementEngine) generateExplanation(score *NodeScore) string {
	if !score.Feasible {
		return fmt.Sprintf("Node %s is not feasible: %s", score.Node.ID, score.Reason)
	}

	return fmt.Sprintf("Node %s selected with score %.2f (resource: %.2f, health: %.2f)",
		score.Node.ID, score.Score, score.Breakdown.ResourceScore, score.Breakdown.HealthScore)
}