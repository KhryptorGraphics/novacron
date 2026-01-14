package placement

import (
	"context"
	"math"

	"github.com/sirupsen/logrus"
)

// HeuristicScorer implements multi-criteria heuristic algorithm for VM placement
type HeuristicScorer struct {
	logger *logrus.Logger
}

// NewHeuristicScorer creates a new heuristic scorer
func NewHeuristicScorer(logger *logrus.Logger) *HeuristicScorer {
	return &HeuristicScorer{
		logger: logger,
	}
}

// ScoreNodes implements NodeScorer interface using heuristic algorithm
func (h *HeuristicScorer) ScoreNodes(ctx context.Context, request *PlacementRequest, nodes []*Node) ([]*NodeScore, error) {
	scores := make([]*NodeScore, len(nodes))

	for i, node := range nodes {
		score := &NodeScore{
			Node:     node,
			Feasible: true,
		}

		// Check basic feasibility first
		if !h.isFeasible(request, node) {
			score.Feasible = false
			score.Reason = "Insufficient resources or node not ready"
			score.Score = 0.0
			scores[i] = score
			continue
		}

		// Apply constraint filtering
		if !h.satisfiesConstraints(request, node) {
			score.Feasible = false
			score.Reason = "Constraint violations"
			score.Score = 0.0
			scores[i] = score
			continue
		}

		// Calculate heuristic score
		breakdown := h.calculateHeuristicScore(request, node)
		score.Breakdown = breakdown
		score.Score = h.aggregateScore(request, breakdown)

		scores[i] = score
	}

	return scores, nil
}

// isFeasible checks if a node can accommodate the VM (same as other scorers)
func (h *HeuristicScorer) isFeasible(request *PlacementRequest, node *Node) bool {
	if node.State != NodeStateReady {
		return false
	}

	if node.Available.CPU < request.VMSpec.CPU ||
		node.Available.Memory < request.VMSpec.Memory ||
		node.Available.Storage < request.VMSpec.Storage {
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
func (h *HeuristicScorer) satisfiesConstraints(request *PlacementRequest, node *Node) bool {
	for _, constraint := range request.Constraints {
		if constraint.Enforcement == EnforcementHard {
			if !h.satisfiesConstraint(constraint, node, request) {
				return false
			}
		}
	}
	return true
}

// satisfiesConstraint checks if a specific constraint is satisfied
func (h *HeuristicScorer) satisfiesConstraint(constraint Constraint, node *Node, request *PlacementRequest) bool {
	switch constraint.Type {
	case ConstraintTypeAffinity:
		return h.checkAffinityConstraint(constraint, node)
	case ConstraintTypeAntiAffinity:
		return h.checkAntiAffinityConstraint(constraint, node)
	case ConstraintTypeResourceLimit:
		return h.checkResourceLimitConstraint(constraint, node, request)
	case ConstraintTypeAvailability:
		return h.checkAvailabilityConstraint(constraint, node)
	case ConstraintTypeNetworkLatency:
		return h.checkNetworkLatencyConstraint(constraint, node)
	case ConstraintTypeCost:
		return h.checkCostConstraint(constraint, node)
	default:
		h.logger.WithField("constraint_type", constraint.Type).Warn("Unknown constraint type")
		return true
	}
}

// checkAffinityConstraint checks node affinity constraints
func (h *HeuristicScorer) checkAffinityConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	if labels, ok := constraint.Parameters["labels"].(map[string]interface{}); ok {
		for key, value := range labels {
			if nodeValue, exists := node.Labels[key]; !exists || nodeValue != value.(string) {
				return false
			}
		}
	}

	if zone, ok := constraint.Parameters["zone"].(string); ok && node.Zone != zone {
		return false
	}

	if region, ok := constraint.Parameters["region"].(string); ok && node.Region != region {
		return false
	}

	return true
}

// checkAntiAffinityConstraint checks node anti-affinity constraints
func (h *HeuristicScorer) checkAntiAffinityConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	if labels, ok := constraint.Parameters["labels"].(map[string]interface{}); ok {
		for key, value := range labels {
			if nodeValue, exists := node.Labels[key]; exists && nodeValue == value.(string) {
				return false
			}
		}
	}

	return true
}

// checkResourceLimitConstraint checks resource limit constraints
func (h *HeuristicScorer) checkResourceLimitConstraint(constraint Constraint, node *Node, request *PlacementRequest) bool {
	if constraint.Parameters == nil {
		return true
	}

	if maxCPUUtil, ok := constraint.Parameters["max_cpu_utilization"].(float64); ok {
		futureUtil := float64(node.Capacity.CPU-node.Available.CPU+request.VMSpec.CPU) / float64(node.Capacity.CPU)
		if futureUtil > maxCPUUtil {
			return false
		}
	}

	if maxMemUtil, ok := constraint.Parameters["max_memory_utilization"].(float64); ok {
		futureUtil := float64(node.Capacity.Memory-node.Available.Memory+request.VMSpec.Memory) / float64(node.Capacity.Memory)
		if futureUtil > maxMemUtil {
			return false
		}
	}

	return true
}

// checkAvailabilityConstraint checks availability constraints
func (h *HeuristicScorer) checkAvailabilityConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	if minHealth, ok := constraint.Parameters["min_health"].(float64); ok {
		if node.Health < minHealth {
			return false
		}
	}

	return true
}

// checkNetworkLatencyConstraint checks network latency constraints
func (h *HeuristicScorer) checkNetworkLatencyConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	// This would require network topology information
	// For now, assume constraint is satisfied
	return true
}

// checkCostConstraint checks cost constraints
func (h *HeuristicScorer) checkCostConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	if maxCost, ok := constraint.Parameters["max_cost_per_hour"].(float64); ok {
		if node.Cost > maxCost {
			return false
		}
	}

	return true
}

// calculateHeuristicScore calculates comprehensive heuristic score
func (h *HeuristicScorer) calculateHeuristicScore(request *PlacementRequest, node *Node) ScoreBreakdown {
	breakdown := ScoreBreakdown{}

	// Resource efficiency score
	breakdown.ResourceScore = h.calculateResourceEfficiencyScore(request, node)

	// Health and reliability score
	breakdown.HealthScore = h.calculateReliabilityScore(node)

	// Constraint satisfaction score
	breakdown.ConstraintScore = h.calculateConstraintScore(request, node)

	// Preference satisfaction score
	breakdown.PreferenceScore = h.calculatePreferenceScore(request, node)

	// Cost efficiency score
	breakdown.CostScore = h.calculateCostEfficiencyScore(request, node)

	// Location and topology score
	breakdown.LocationScore = h.calculateLocationScore(request, node)

	return breakdown
}

// calculateResourceEfficiencyScore calculates how efficiently resources will be used
func (h *HeuristicScorer) calculateResourceEfficiencyScore(request *PlacementRequest, node *Node) float64 {
	// Calculate current and future utilization
	cpuUtilAfter := float64(node.Capacity.CPU-node.Available.CPU+request.VMSpec.CPU) / float64(node.Capacity.CPU)
	memUtilAfter := float64(node.Capacity.Memory-node.Available.Memory+request.VMSpec.Memory) / float64(node.Capacity.Memory)
	storageUtilAfter := float64(node.Capacity.Storage-node.Available.Storage+request.VMSpec.Storage) / float64(node.Capacity.Storage)

	// Optimal utilization range
	optimalMin, optimalMax := 0.3, 0.8

	cpuScore := h.calculateOptimalUtilizationScore(cpuUtilAfter, optimalMin, optimalMax)
	memScore := h.calculateOptimalUtilizationScore(memUtilAfter, optimalMin, optimalMax)
	storageScore := h.calculateOptimalUtilizationScore(storageUtilAfter, optimalMin, optimalMax)

	// Check resource balance
	balance := h.calculateResourceBalance(cpuUtilAfter, memUtilAfter, storageUtilAfter)

	// Weighted combination
	return (cpuScore*0.35 + memScore*0.35 + storageScore*0.2 + balance*0.1)
}

// calculateOptimalUtilizationScore calculates score based on optimal utilization range
func (h *HeuristicScorer) calculateOptimalUtilizationScore(utilization, optimalMin, optimalMax float64) float64 {
	if utilization > 1.0 {
		return 0.0 // Over-utilization
	}

	if utilization >= optimalMin && utilization <= optimalMax {
		return 1.0 // In optimal range
	}

	if utilization < optimalMin {
		// Under-utilization penalty (but not as severe)
		return 0.5 + (utilization/optimalMin)*0.5
	}

	// Over optimal range but under capacity
	overUtilization := (utilization - optimalMax) / (1.0 - optimalMax)
	return math.Max(0.0, 1.0-overUtilization*2.0)
}

// calculateResourceBalance calculates resource balance score
func (h *HeuristicScorer) calculateResourceBalance(cpuUtil, memUtil, storageUtil float64) float64 {
	mean := (cpuUtil + memUtil + storageUtil) / 3.0
	variance := ((cpuUtil-mean)*(cpuUtil-mean) + (memUtil-mean)*(memUtil-mean) + (storageUtil-mean)*(storageUtil-mean)) / 3.0
	return 1.0 / (1.0 + variance*5.0)
}

// calculateReliabilityScore calculates comprehensive reliability score
func (h *HeuristicScorer) calculateReliabilityScore(node *Node) float64 {
	healthScore := node.Health

	// Consider node age/stability (would require historical data)
	// For now, use health as primary indicator
	stabilityScore := healthScore // Placeholder

	// Check for taints that might indicate issues
	taintPenalty := 0.0
	for _, taint := range node.Taints {
		if taint.Effect == TaintEffectNoSchedule || taint.Effect == TaintEffectPreferNoSchedule {
			taintPenalty += 0.1
		}
	}

	reliabilityScore := (healthScore*0.7 + stabilityScore*0.3) - taintPenalty
	return math.Max(0.0, reliabilityScore)
}

// calculateConstraintScore calculates comprehensive constraint satisfaction
func (h *HeuristicScorer) calculateConstraintScore(request *PlacementRequest, node *Node) float64 {
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

			var score float64
			if constraint.Enforcement == EnforcementSoft {
				// Soft constraints: partial satisfaction possible
				score = h.calculateSoftConstraintScore(constraint, node, request)
			} else {
				// Preferred constraints: binary satisfaction
				if h.satisfiesConstraint(constraint, node, request) {
					score = 1.0
				} else {
					score = 0.0
				}
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

// calculateSoftConstraintScore calculates partial satisfaction for soft constraints
func (h *HeuristicScorer) calculateSoftConstraintScore(constraint Constraint, node *Node, request *PlacementRequest) float64 {
	// For soft constraints, allow partial satisfaction
	switch constraint.Type {
	case ConstraintTypeResourceLimit:
		return h.calculateSoftResourceLimitScore(constraint, node, request)
	case ConstraintTypeNetworkLatency:
		return h.calculateSoftLatencyScore(constraint, node)
	case ConstraintTypeCost:
		return h.calculateSoftCostScore(constraint, node)
	default:
		// For other constraint types, use binary satisfaction
		if h.satisfiesConstraint(constraint, node, request) {
			return 1.0
		}
		return 0.0
	}
}

// calculateSoftResourceLimitScore calculates soft resource limit satisfaction
func (h *HeuristicScorer) calculateSoftResourceLimitScore(constraint Constraint, node *Node, request *PlacementRequest) float64 {
	if constraint.Parameters == nil {
		return 1.0
	}

	score := 1.0

	if maxCPUUtil, ok := constraint.Parameters["max_cpu_utilization"].(float64); ok {
		futureUtil := float64(node.Capacity.CPU-node.Available.CPU+request.VMSpec.CPU) / float64(node.Capacity.CPU)
		if futureUtil > maxCPUUtil {
			// Gradual penalty for exceeding limit
			overage := (futureUtil - maxCPUUtil) / (1.0 - maxCPUUtil)
			score *= math.Max(0.0, 1.0-overage)
		}
	}

	return score
}

// calculateSoftLatencyScore calculates soft latency constraint satisfaction
func (h *HeuristicScorer) calculateSoftLatencyScore(constraint Constraint, node *Node) float64 {
	// Placeholder - would require network topology data
	return 1.0
}

// calculateSoftCostScore calculates soft cost constraint satisfaction
func (h *HeuristicScorer) calculateSoftCostScore(constraint Constraint, node *Node) float64 {
	if constraint.Parameters == nil {
		return 1.0
	}

	if maxCost, ok := constraint.Parameters["max_cost_per_hour"].(float64); ok {
		if node.Cost <= maxCost {
			return 1.0
		}
		// Gradual penalty for exceeding cost
		overage := (node.Cost - maxCost) / maxCost
		return math.Max(0.0, 1.0-overage)
	}

	return 1.0
}

// calculatePreferenceScore calculates comprehensive preference satisfaction
func (h *HeuristicScorer) calculatePreferenceScore(request *PlacementRequest, node *Node) float64 {
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

		score := h.calculatePreferenceTypeScore(preference, node)

		totalWeight += weight
		weightedScore += score * weight
	}

	if totalWeight == 0 {
		return 1.0
	}

	return weightedScore / totalWeight
}

// calculatePreferenceTypeScore calculates score for specific preference types
func (h *HeuristicScorer) calculatePreferenceTypeScore(preference Preference, node *Node) float64 {
	switch preference.Type {
	case PreferenceTypeLowLatency:
		// Consider zone/region proximity
		if targetZone, ok := preference.Parameters["preferred_zone"].(string); ok {
			if node.Zone == targetZone {
				return 1.0
			}
			return 0.5 // Different zone penalty
		}
		return 1.0
	case PreferenceTypeHighBandwidth:
		// Score based on available network capacity
		utilization := float64(node.Capacity.NetworkMbps-node.Available.NetworkMbps) / float64(node.Capacity.NetworkMbps)
		return 1.0 - utilization
	case PreferenceTypeLowCost:
		if node.Cost <= 0 {
			return 1.0
		}
		maxCost := 1.0 // Reference max cost
		return math.Max(0.0, 1.0-(node.Cost/maxCost))
	case PreferenceTypeLocalStorage:
		utilization := float64(node.Capacity.Storage-node.Available.Storage) / float64(node.Capacity.Storage)
		return 1.0 - utilization
	case PreferenceTypeGPUOptimized:
		if len(node.Available.GPUs) == 0 {
			return 0.0
		}
		availableGPUs := 0
		for _, gpu := range node.Available.GPUs {
			if !gpu.InUse {
				availableGPUs++
			}
		}
		return float64(availableGPUs) / float64(len(node.Available.GPUs))
	default:
		return 1.0
	}
}

// calculateCostEfficiencyScore calculates cost efficiency score
func (h *HeuristicScorer) calculateCostEfficiencyScore(request *PlacementRequest, node *Node) float64 {
	if node.Cost <= 0 {
		return 1.0
	}

	// Calculate cost per resource unit
	costPerCPU := node.Cost / float64(node.Capacity.CPU)
	costPerMemGB := node.Cost / (float64(node.Capacity.Memory) / 1024.0)

	// Compare with baseline costs (would be configurable)
	baselineCostPerCPU := 0.01
	baselineCostPerMemGB := 0.005

	cpuCostEfficiency := math.Min(1.0, baselineCostPerCPU/costPerCPU)
	memCostEfficiency := math.Min(1.0, baselineCostPerMemGB/costPerMemGB)

	return (cpuCostEfficiency + memCostEfficiency) / 2.0
}

// calculateLocationScore calculates location-based score
func (h *HeuristicScorer) calculateLocationScore(request *PlacementRequest, node *Node) float64 {
	score := 1.0

	// Prefer certain regions/zones based on context
	if context := request.Context; context != nil {
		if preferredRegion, ok := context["preferred_region"].(string); ok {
			if node.Region == preferredRegion {
				score += 0.1
			}
		}

		if preferredZone, ok := context["preferred_zone"].(string); ok {
			if node.Zone == preferredZone {
				score += 0.1
			}
		}
	}

	return math.Min(1.0, score)
}

// aggregateScore combines all score components with strategy-specific weights
func (h *HeuristicScorer) aggregateScore(request *PlacementRequest, breakdown ScoreBreakdown) float64 {
	// Dynamic weights based on placement strategy
	weights := h.getStrategyWeights(request.Strategy)

	score := weights["resource"]*breakdown.ResourceScore +
		weights["health"]*breakdown.HealthScore +
		weights["constraint"]*breakdown.ConstraintScore +
		weights["preference"]*breakdown.PreferenceScore +
		weights["cost"]*breakdown.CostScore +
		weights["location"]*breakdown.LocationScore

	return math.Max(0.0, math.Min(1.0, score))
}

// getStrategyWeights returns weights based on placement strategy
func (h *HeuristicScorer) getStrategyWeights(strategy PlacementStrategy) map[string]float64 {
	switch strategy {
	case StrategyPerformance:
		return map[string]float64{
			"resource":    0.3,
			"health":      0.25,
			"constraint":  0.2,
			"preference":  0.15,
			"cost":        0.05,
			"location":    0.05,
		}
	case StrategyCostOptimized:
		return map[string]float64{
			"resource":    0.2,
			"health":      0.15,
			"constraint":  0.15,
			"preference":  0.1,
			"cost":        0.3,
			"location":    0.1,
		}
	case StrategyBalanced:
		return map[string]float64{
			"resource":    0.25,
			"health":      0.2,
			"constraint":  0.2,
			"preference":  0.15,
			"cost":        0.1,
			"location":    0.1,
		}
	default: // StrategyEfficiency and others
		return map[string]float64{
			"resource":    0.35,
			"health":      0.2,
			"constraint":  0.2,
			"preference":  0.15,
			"cost":        0.05,
			"location":    0.05,
		}
	}
}