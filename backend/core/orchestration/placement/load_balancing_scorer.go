package placement

import (
	"context"
	"math"

	"github.com/sirupsen/logrus"
)

// LoadBalancingScorer implements load-balancing algorithm for VM placement
type LoadBalancingScorer struct {
	logger *logrus.Logger
}

// NewLoadBalancingScorer creates a new load-balancing scorer
func NewLoadBalancingScorer(logger *logrus.Logger) *LoadBalancingScorer {
	return &LoadBalancingScorer{
		logger: logger,
	}
}

// ScoreNodes implements NodeScorer interface using load-balancing algorithm
func (l *LoadBalancingScorer) ScoreNodes(ctx context.Context, request *PlacementRequest, nodes []*Node) ([]*NodeScore, error) {
	scores := make([]*NodeScore, len(nodes))

	for i, node := range nodes {
		score := &NodeScore{
			Node:     node,
			Feasible: true,
		}

		// Check basic feasibility first
		if !l.isFeasible(request, node) {
			score.Feasible = false
			score.Reason = "Insufficient resources or node not ready"
			score.Score = 0.0
			scores[i] = score
			continue
		}

		// Apply constraint filtering
		if !l.satisfiesConstraints(request, node) {
			score.Feasible = false
			score.Reason = "Constraint violations"
			score.Score = 0.0
			scores[i] = score
			continue
		}

		// Calculate load-balancing score
		breakdown := l.calculateLoadBalancingScore(request, node)
		score.Breakdown = breakdown
		score.Score = l.aggregateScore(breakdown)

		scores[i] = score
	}

	return scores, nil
}

// isFeasible checks if a node can accommodate the VM
func (l *LoadBalancingScorer) isFeasible(request *PlacementRequest, node *Node) bool {
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
func (l *LoadBalancingScorer) satisfiesConstraints(request *PlacementRequest, node *Node) bool {
	for _, constraint := range request.Constraints {
		if constraint.Enforcement == EnforcementHard {
			if !l.satisfiesConstraint(constraint, node, request) {
				return false
			}
		}
	}
	return true
}

// satisfiesConstraint checks if a specific constraint is satisfied
func (l *LoadBalancingScorer) satisfiesConstraint(constraint Constraint, node *Node, request *PlacementRequest) bool {
	switch constraint.Type {
	case ConstraintTypeAffinity:
		return l.checkAffinityConstraint(constraint, node)
	case ConstraintTypeAntiAffinity:
		return l.checkAntiAffinityConstraint(constraint, node)
	case ConstraintTypeResourceLimit:
		return l.checkResourceLimitConstraint(constraint, node, request)
	case ConstraintTypeAvailability:
		return l.checkAvailabilityConstraint(constraint, node)
	default:
		l.logger.WithField("constraint_type", constraint.Type).Warn("Unknown constraint type")
		return true
	}
}

// checkAffinityConstraint checks node affinity constraints
func (l *LoadBalancingScorer) checkAffinityConstraint(constraint Constraint, node *Node) bool {
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

	return true
}

// checkAntiAffinityConstraint checks node anti-affinity constraints
func (l *LoadBalancingScorer) checkAntiAffinityConstraint(constraint Constraint, node *Node) bool {
	if constraint.Parameters == nil {
		return true
	}

	// Check label selectors (opposite of affinity)
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
func (l *LoadBalancingScorer) checkResourceLimitConstraint(constraint Constraint, node *Node, request *PlacementRequest) bool {
	if constraint.Parameters == nil {
		return true
	}

	// Check CPU utilization limit
	if maxCPUUtil, ok := constraint.Parameters["max_cpu_utilization"].(float64); ok {
		futureUtil := float64(node.Capacity.CPU-node.Available.CPU+request.VMSpec.CPU) / float64(node.Capacity.CPU)
		if futureUtil > maxCPUUtil {
			return false
		}
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
func (l *LoadBalancingScorer) checkAvailabilityConstraint(constraint Constraint, node *Node) bool {
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

// calculateLoadBalancingScore calculates the load-balancing score for a node
func (l *LoadBalancingScorer) calculateLoadBalancingScore(request *PlacementRequest, node *Node) ScoreBreakdown {
	breakdown := ScoreBreakdown{}

	// Resource balance score (load-balancing favors lower utilization)
	breakdown.ResourceScore = l.calculateResourceBalanceScore(request, node)

	// Health score
	breakdown.HealthScore = node.Health

	// Constraint score
	breakdown.ConstraintScore = l.calculateConstraintScore(request, node)

	// Preference score
	breakdown.PreferenceScore = l.calculatePreferenceScore(request, node)

	// Cost score
	if node.Cost > 0 {
		breakdown.CostScore = 1.0 / (1.0 + node.Cost)
	} else {
		breakdown.CostScore = 1.0
	}

	return breakdown
}

// calculateResourceBalanceScore calculates how balanced the resources will be after placement
func (l *LoadBalancingScorer) calculateResourceBalanceScore(request *PlacementRequest, node *Node) float64 {
	// Calculate current utilization
	currentCPUUtil := float64(node.Capacity.CPU-node.Available.CPU) / float64(node.Capacity.CPU)
	currentMemUtil := float64(node.Capacity.Memory-node.Available.Memory) / float64(node.Capacity.Memory)
	currentStorageUtil := float64(node.Capacity.Storage-node.Available.Storage) / float64(node.Capacity.Storage)

	// Calculate utilization after placement
	futureCPUUtil := float64(node.Capacity.CPU-node.Available.CPU+request.VMSpec.CPU) / float64(node.Capacity.CPU)
	futureMemUtil := float64(node.Capacity.Memory-node.Available.Memory+request.VMSpec.Memory) / float64(node.Capacity.Memory)
	futureStorageUtil := float64(node.Capacity.Storage-node.Available.Storage+request.VMSpec.Storage) / float64(node.Capacity.Storage)

	// Load balancing prefers nodes with lower current utilization
	cpuScore := l.calculateBalanceScore(currentCPUUtil, futureCPUUtil)
	memScore := l.calculateBalanceScore(currentMemUtil, futureMemUtil)
	storageScore := l.calculateBalanceScore(currentStorageUtil, futureStorageUtil)

	// Check for resource balance - prefer nodes where all resources are utilized equally
	futureBalance := l.calculateResourceBalance(futureCPUUtil, futureMemUtil, futureStorageUtil)

	// Weighted average with balance consideration
	resourceScore := (cpuScore*0.4 + memScore*0.4 + storageScore*0.2)
	return (resourceScore * 0.7) + (futureBalance * 0.3)
}

// calculateBalanceScore calculates score based on current and future utilization
func (l *LoadBalancingScorer) calculateBalanceScore(currentUtil, futureUtil float64) float64 {
	if futureUtil > 1.0 {
		return 0.0 // Over-utilization
	}

	// Prefer nodes with lower current utilization for load balancing
	currentScore := 1.0 - currentUtil

	// But don't waste resources - moderate future utilization is good
	var futureScore float64
	if futureUtil <= 0.7 {
		futureScore = 1.0
	} else if futureUtil <= 0.9 {
		futureScore = 1.0 - ((futureUtil-0.7)*2.5) // Gradual decrease
	} else {
		futureScore = 1.0 - ((futureUtil-0.7)*5.0) // Steeper decrease
		if futureScore < 0 {
			futureScore = 0
		}
	}

	// Balance current and future considerations
	return (currentScore * 0.6) + (futureScore * 0.4)
}

// calculateResourceBalance calculates how balanced the resource utilization is
func (l *LoadBalancingScorer) calculateResourceBalance(cpuUtil, memUtil, storageUtil float64) float64 {
	// Calculate variance in utilization across different resources
	mean := (cpuUtil + memUtil + storageUtil) / 3.0
	variance := ((cpuUtil-mean)*(cpuUtil-mean) + (memUtil-mean)*(memUtil-mean) + (storageUtil-mean)*(storageUtil-mean)) / 3.0
	
	// Convert variance to balance score (lower variance = better balance)
	balance := 1.0 / (1.0 + variance*10.0) // Scale variance for meaningful impact
	
	return balance
}

// calculateConstraintScore calculates how well the node satisfies soft constraints
func (l *LoadBalancingScorer) calculateConstraintScore(request *PlacementRequest, node *Node) float64 {
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
			if l.satisfiesConstraint(constraint, node, request) {
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
func (l *LoadBalancingScorer) calculatePreferenceScore(request *PlacementRequest, node *Node) float64 {
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

		score := l.calculatePreferenceTypeScore(preference, node)

		totalWeight += weight
		weightedScore += score * weight
	}

	if totalWeight == 0 {
		return 1.0
	}

	return weightedScore / totalWeight
}

// calculatePreferenceTypeScore calculates score for a specific preference type
func (l *LoadBalancingScorer) calculatePreferenceTypeScore(preference Preference, node *Node) float64 {
	switch preference.Type {
	case PreferenceTypeLowLatency:
		// For load balancing, distribute across zones for better availability
		// This is different from bin-packing which might prefer same zone
		return 1.0 // Equal preference for all zones in load balancing
	case PreferenceTypeHighBandwidth:
		// Prefer nodes with high available network capacity
		utilization := float64(node.Capacity.NetworkMbps-node.Available.NetworkMbps) / float64(node.Capacity.NetworkMbps)
		return 1.0 - utilization // Lower utilization is better
	case PreferenceTypeLowCost:
		// Prefer nodes with lower cost
		if node.Cost <= 0 {
			return 1.0
		}
		maxCost := 1.0
		return math.Max(0.0, 1.0-(node.Cost/maxCost))
	case PreferenceTypeLocalStorage:
		// Prefer nodes with available local storage
		utilization := float64(node.Capacity.Storage-node.Available.Storage) / float64(node.Capacity.Storage)
		return 1.0 - utilization
	case PreferenceTypeGPUOptimized:
		// Prefer nodes with available GPUs
		availableGPUs := 0
		for _, gpu := range node.Available.GPUs {
			if !gpu.InUse {
				availableGPUs++
			}
		}
		if len(node.Available.GPUs) == 0 {
			return 0.0
		}
		return float64(availableGPUs) / float64(len(node.Available.GPUs))
	default:
		return 1.0
	}
}

// aggregateScore combines all score components into final score
func (l *LoadBalancingScorer) aggregateScore(breakdown ScoreBreakdown) float64 {
	// Weights for load-balancing strategy
	weights := map[string]float64{
		"resource":    0.5, // Most important for load balancing
		"health":      0.2,
		"constraint":  0.15,
		"preference":  0.1,
		"cost":        0.05,
	}

	score := weights["resource"]*breakdown.ResourceScore +
		weights["health"]*breakdown.HealthScore +
		weights["constraint"]*breakdown.ConstraintScore +
		weights["preference"]*breakdown.PreferenceScore +
		weights["cost"]*breakdown.CostScore

	return math.Max(0.0, math.Min(1.0, score))
}