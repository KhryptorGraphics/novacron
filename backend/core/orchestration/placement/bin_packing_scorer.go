package placement

import (
	"context"
	"math"

	"github.com/sirupsen/logrus"
)

// BinPackingScorer implements bin-packing algorithm for VM placement
type BinPackingScorer struct {
	logger *logrus.Logger
}

// NewBinPackingScorer creates a new bin-packing scorer
func NewBinPackingScorer(logger *logrus.Logger) *BinPackingScorer {
	return &BinPackingScorer{
		logger: logger,
	}
}

// ScoreNodes implements NodeScorer interface using bin-packing algorithm
func (b *BinPackingScorer) ScoreNodes(ctx context.Context, request *PlacementRequest, nodes []*Node) ([]*NodeScore, error) {
	scores := make([]*NodeScore, len(nodes))

	for i, node := range nodes {
		score := &NodeScore{
			Node:     node,
			Feasible: true,
		}

		// Check basic feasibility first
		if !b.isFeasible(request, node) {
			score.Feasible = false
			score.Reason = "Insufficient resources or node not ready"
			score.Score = 0.0
			scores[i] = score
			continue
		}

		// Apply constraint filtering
		if !b.satisfiesConstraints(request, node) {
			score.Feasible = false
			score.Reason = "Constraint violations"
			score.Score = 0.0
			scores[i] = score
			continue
		}

		// Calculate bin-packing score
		breakdown := b.calculateBinPackingScore(request, node)
		score.Breakdown = breakdown
		score.Score = b.aggregateScore(breakdown)

		scores[i] = score
	}

	return scores, nil
}

// isFeasible checks if a node can accommodate the VM
func (b *BinPackingScorer) isFeasible(request *PlacementRequest, node *Node) bool {
	// Check node state
	if node.State != NodeStateReady {
		return false
	}

	// Check basic resources
	if node.Available.CPU < request.VMSpec.CPU {
		return false
	}

	if node.Available.Memory < request.VMSpec.Memory {
		return false
	}

	if node.Available.Storage < request.VMSpec.Storage {
		return false
	}

	// Check GPU requirements
	if request.VMSpec.GPU != nil {
		availableGPUs := 0
		for _, gpu := range node.Available.GPUs {
			if !gpu.InUse {
				if request.VMSpec.GPU.Model == "" || gpu.Model == request.VMSpec.GPU.Model {
					if request.VMSpec.GPU.MemoryMB == 0 || gpu.MemoryMB >= request.VMSpec.GPU.MemoryMB {
						availableGPUs++
					}
				}
			}
		}
		if availableGPUs < request.VMSpec.GPU.Count {
			return false
		}
	}

	// Check network requirements
	if request.VMSpec.Network.BandwidthMbps > 0 && node.Available.NetworkMbps < request.VMSpec.Network.BandwidthMbps {
		return false
	}

	return true
}

// satisfiesConstraints checks if node satisfies hard constraints
func (b *BinPackingScorer) satisfiesConstraints(request *PlacementRequest, node *Node) bool {
	for _, constraint := range request.Constraints {
		if constraint.Enforcement == EnforcementHard {
			if !b.satisfiesConstraint(constraint, node, request) {
				return false
			}
		}
	}
	return true
}

// satisfiesConstraint checks if a specific constraint is satisfied
func (b *BinPackingScorer) satisfiesConstraint(constraint Constraint, node *Node, request *PlacementRequest) bool {
	switch constraint.Type {
	case ConstraintTypeAffinity:
		return b.checkAffinityConstraint(constraint, node)
	case ConstraintTypeAntiAffinity:
		return b.checkAntiAffinityConstraint(constraint, node)
	case ConstraintTypeResourceLimit:
		return b.checkResourceLimitConstraint(constraint, node, request)
	case ConstraintTypeAvailability:
		return b.checkAvailabilityConstraint(constraint, node)
	default:
		b.logger.WithField("constraint_type", constraint.Type).Warn("Unknown constraint type")
		return true // Unknown constraints are ignored
	}
}

// checkAffinityConstraint checks node affinity constraints
func (b *BinPackingScorer) checkAffinityConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	// Check label selectors
	if labels, ok := constraint.Parameters["labels"].(map[string]interface{}); ok {
		for key, value := range labels {
			if nodeValue, exists := node.Labels[key]; !exists || nodeValue != value.(string) {
				return false
			}
		}
	}

	// Check zone affinity
	if zone, ok := constraint.Parameters["zone"].(string); ok {
		if node.Zone != zone {
			return false
		}
	}

	// Check region affinity
	if region, ok := constraint.Parameters["region"].(string); ok {
		if node.Region != region {
			return false
		}
	}

	return true
}

// checkAntiAffinityConstraint checks node anti-affinity constraints
func (b *BinPackingScorer) checkAntiAffinityConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	// Check label selectors (opposite of affinity)
	if labels, ok := constraint.Parameters["labels"].(map[string]interface{}); ok {
		for key, value := range labels {
			if nodeValue, exists := node.Labels[key]; exists && nodeValue == value.(string) {
				return false // Node has the label we want to avoid
			}
		}
	}

	return true
}

// checkResourceLimitConstraint checks resource limit constraints
func (b *BinPackingScorer) checkResourceLimitConstraint(constraint Constraint, node *Node, request *PlacementRequest) bool {
	if constraint.Parameters == nil {
		return true
	}

	// Check CPU utilization limit
	if maxCPUUtil, ok := constraint.Parameters["max_cpu_utilization"].(float64); ok {
		currentUtil := float64(node.Capacity.CPU-node.Available.CPU) / float64(node.Capacity.CPU)
		futureUtil := float64(node.Capacity.CPU-node.Available.CPU+request.VMSpec.CPU) / float64(node.Capacity.CPU)
		if futureUtil > maxCPUUtil {
			return false
		}
		_ = currentUtil // Avoid unused variable
	}

	// Check memory utilization limit
	if maxMemUtil, ok := constraint.Parameters["max_memory_utilization"].(float64); ok {
		futureUtil := float64(node.Capacity.Memory-node.Available.Memory+request.VMSpec.Memory) / float64(node.Capacity.Memory)
		if futureUtil > maxMemUtil {
			return false
		}
	}

	return true
}

// checkAvailabilityConstraint checks availability constraints
func (b *BinPackingScorer) checkAvailabilityConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	// Check minimum health threshold
	if minHealth, ok := constraint.Parameters["min_health"].(float64); ok {
		if node.Health < minHealth {
			return false
		}
	}

	return true
}

// calculateBinPackingScore calculates the bin-packing score for a node
func (b *BinPackingScorer) calculateBinPackingScore(request *PlacementRequest, node *Node) ScoreBreakdown {
	breakdown := ScoreBreakdown{}

	// Resource utilization score (bin-packing favors higher utilization)
	breakdown.ResourceScore = b.calculateResourceUtilizationScore(request, node)

	// Health score
	breakdown.HealthScore = node.Health

	// Constraint score (how well it satisfies soft constraints)
	breakdown.ConstraintScore = b.calculateConstraintScore(request, node)

	// Preference score
	breakdown.PreferenceScore = b.calculatePreferenceScore(request, node)

	// Cost score (lower cost is better)
	if node.Cost > 0 {
		breakdown.CostScore = 1.0 / (1.0 + node.Cost) // Inverse relationship
	} else {
		breakdown.CostScore = 1.0
	}

	return breakdown
}

// calculateResourceUtilizationScore calculates how well the VM fits the node (bin-packing)
func (b *BinPackingScorer) calculateResourceUtilizationScore(request *PlacementRequest, node *Node) float64 {
	// Calculate utilization after placement for each resource
	cpuUtil := float64(node.Capacity.CPU-node.Available.CPU+request.VMSpec.CPU) / float64(node.Capacity.CPU)
	memUtil := float64(node.Capacity.Memory-node.Available.Memory+request.VMSpec.Memory) / float64(node.Capacity.Memory)
	storageUtil := float64(node.Capacity.Storage-node.Available.Storage+request.VMSpec.Storage) / float64(node.Capacity.Storage)

	// Bin-packing favors higher utilization (tighter fit)
	// But penalizes over-utilization
	cpuScore := b.calculateUtilizationScore(cpuUtil)
	memScore := b.calculateUtilizationScore(memUtil)
	storageScore := b.calculateUtilizationScore(storageUtil)

	// Weighted average (memory and CPU are most important)
	return (cpuScore*0.4 + memScore*0.4 + storageScore*0.2)
}

// calculateUtilizationScore calculates a score based on resource utilization
func (b *BinPackingScorer) calculateUtilizationScore(utilization float64) float64 {
	if utilization > 1.0 {
		return 0.0 // Over-utilization gets zero score
	}

	// Optimal utilization range is 0.7-0.9
	// Score peaks at 0.8 utilization
	optimalUtil := 0.8
	if utilization <= optimalUtil {
		return utilization / optimalUtil
	} else {
		// Penalty for high utilization
		overage := utilization - optimalUtil
		penalty := overage * 2.0 // Steep penalty
		return math.Max(0.0, 1.0-penalty)
	}
}

// calculateConstraintScore calculates how well the node satisfies soft constraints
func (b *BinPackingScorer) calculateConstraintScore(request *PlacementRequest, node *Node) float64 {
	if len(request.Constraints) == 0 {
		return 1.0
	}

	totalWeight := 0.0
	weightedScore := 0.0

	for _, constraint := range request.Constraints {
		if constraint.Enforcement != EnforcementHard {
			weight := constraint.Weight
			if weight <= 0 {
				weight = 1.0
			}

			score := 0.0
			if b.satisfiesConstraint(constraint, node, request) {
				score = 1.0
			}

			totalWeight += weight
			weightedScore += score * weight
		}
	}

	if totalWeight == 0 {
		return 1.0
	}

	return weightedScore / totalWeight
}

// calculatePreferenceScore calculates how well the node matches preferences
func (b *BinPackingScorer) calculatePreferenceScore(request *PlacementRequest, node *Node) float64 {
	if len(request.Preferences) == 0 {
		return 1.0
	}

	totalWeight := 0.0
	weightedScore := 0.0

	for _, preference := range request.Preferences {
		weight := preference.Weight
		if weight <= 0 {
			weight = 1.0
		}

		score := b.calculatePreferenceTypeScore(preference, node)

		totalWeight += weight
		weightedScore += score * weight
	}

	if totalWeight == 0 {
		return 1.0
	}

	return weightedScore / totalWeight
}

// calculatePreferenceTypeScore calculates score for a specific preference type
func (b *BinPackingScorer) calculatePreferenceTypeScore(preference Preference, node *Node) float64 {
	switch preference.Type {
	case PreferenceTypeLowLatency:
		// Prefer nodes in the same zone/region
		if zone, ok := preference.Parameters["zone"].(string); ok {
			if node.Zone == zone {
				return 1.0
			}
		}
		return 0.5
	case PreferenceTypeHighBandwidth:
		// Prefer nodes with high network capacity
		maxBandwidth := 20000.0 // Example max bandwidth
		return float64(node.Available.NetworkMbps) / maxBandwidth
	case PreferenceTypeLowCost:
		// Prefer nodes with lower cost
		if node.Cost <= 0 {
			return 1.0
		}
		maxCost := 1.0 // Example max cost per hour
		return math.Max(0.0, 1.0-(node.Cost/maxCost))
	case PreferenceTypeLocalStorage:
		// Prefer nodes with high local storage
		utilization := float64(node.Capacity.Storage-node.Available.Storage) / float64(node.Capacity.Storage)
		return 1.0 - utilization // Lower utilization is better for local storage
	case PreferenceTypeGPUOptimized:
		// Prefer nodes with available GPUs
		if len(node.Available.GPUs) > 0 {
			return 1.0
		}
		return 0.0
	default:
		return 1.0
	}
}

// aggregateScore combines all score components into final score
func (b *BinPackingScorer) aggregateScore(breakdown ScoreBreakdown) float64 {
	// Weights for bin-packing strategy
	weights := map[string]float64{
		"resource":    0.4, // Most important for bin-packing
		"health":      0.3,
		"constraint":  0.2,
		"preference":  0.1,
		"cost":        0.0, // Cost is less important in bin-packing
	}

	score := weights["resource"]*breakdown.ResourceScore +
		weights["health"]*breakdown.HealthScore +
		weights["constraint"]*breakdown.ConstraintScore +
		weights["preference"]*breakdown.PreferenceScore +
		weights["cost"]*breakdown.CostScore

	return math.Max(0.0, math.Min(1.0, score))
}