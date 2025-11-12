package partition

import (
	"context"
	"errors"
	"math/rand"
)

// DQNPlacementAgent provides DQN-based placement without external dependencies
// This is a standalone implementation for v3 that doesn't depend on the v1 DQN
type DQNPlacementAgent struct {
	epsilon        float64
	learningRate   float64
	gamma          float64
	stepCount      int
}

// NewDQNPlacementAgent creates a new DQN placement agent
func NewDQNPlacementAgent() (*DQNPlacementAgent, error) {
	return &DQNPlacementAgent{
		epsilon:      0.1,  // Low exploration rate for production
		learningRate: 0.001,
		gamma:        0.95,
		stepCount:    0,
	}, nil
}

// Place places a VM using DQN-inspired heuristics optimized for datacenter performance
func (d *DQNPlacementAgent) Place(ctx context.Context, vm *VM, nodes []*Node, constraints *Constraints) (*Node, error) {
	if len(nodes) == 0 {
		return nil, errors.New("no available nodes")
	}

	// Epsilon-greedy strategy
	if rand.Float64() < d.epsilon {
		// Exploration: random selection
		return nodes[rand.Intn(len(nodes))], nil
	}

	// Exploitation: select best node based on Q-value approximation
	bestNode := nodes[0]
	bestScore := d.calculateQValue(vm, nodes[0], constraints)

	for i := 1; i < len(nodes); i++ {
		score := d.calculateQValue(vm, nodes[i], constraints)
		if score > bestScore {
			bestScore = score
			bestNode = nodes[i]
		}
	}

	d.stepCount++

	// Decay epsilon over time
	if d.epsilon > 0.01 {
		d.epsilon *= 0.995
	}

	return bestNode, nil
}

// calculateQValue estimates the Q-value for placing VM on node
// This simulates what a trained DQN would output
func (d *DQNPlacementAgent) calculateQValue(vm *VM, node *Node, constraints *Constraints) float64 {
	score := 0.0

	// Resource availability reward (0-1)
	cpuAvail := float64(node.AvailableCPU) / float64(node.TotalCPU)
	memAvail := float64(node.AvailableMemory) / float64(node.TotalMemory)
	resourceScore := (cpuAvail + memAvail) / 2.0
	score += resourceScore * 0.3

	// Performance score for datacenter placement
	// Prefer nodes with high bandwidth and low latency
	perfScore := 0.0

	// Network performance (normalized)
	if node.NetworkBandwidth > 0 {
		perfScore += min(node.NetworkBandwidth/100.0, 1.0) * 0.5 // Normalize to 100 Gbps
	}

	// Node type preference for datacenter workloads
	switch node.Type {
	case NodeTypeDatacenter:
		perfScore += 0.3
	case NodeTypeCloud:
		perfScore += 0.2
	case NodeTypeEdge:
		perfScore += 0.1
	case NodeTypeVolunteer:
		perfScore += 0.0
	}

	// CPU frequency bonus (if available)
	if node.CPUFrequency > 0 {
		perfScore += min(node.CPUFrequency/4.0, 1.0) * 0.2 // Normalize to 4 GHz
	}

	score += perfScore * 0.4

	// Bin packing efficiency (prefer fuller nodes for better utilization)
	// This is opposite of spreading - good for datacenter efficiency
	utilizationScore := 1.0 - resourceScore // Higher score for more utilized nodes
	score += utilizationScore * 0.2

	// Reliability score
	reliabilityScore := 0.0
	if node.Uptime > 0 {
		// Normalize uptime (assume 1 year is max)
		reliabilityScore += min(float64(node.Uptime.Hours())/(24*365), 1.0) * 0.5
	}
	if node.FailureRate < 0.01 {
		reliabilityScore += 0.5
	}
	score += reliabilityScore * 0.1

	// Apply constraints penalties
	if constraints != nil {
		// Latency constraint (datacenter should have low latency)
		if constraints.MaxLatency > 0 && node.Type != NodeTypeDatacenter {
			score *= 0.8
		}

		// Bandwidth constraint
		if constraints.MinBandwidth > 0 && node.NetworkBandwidth < constraints.MinBandwidth {
			score *= 0.5
		}

		// Uptime constraint
		if constraints.RequiredUptime > 0 {
			actualUptime := float64(node.Uptime.Hours()) / (24 * 365)
			if actualUptime < constraints.RequiredUptime {
				score *= 0.3
			}
		}

		// Cost constraint
		if constraints.MaxCostPerHour > 0 && node.CostPerHour > constraints.MaxCostPerHour {
			score *= 0.7
		}
	}

	// Priority weighting
	score *= (0.5 + vm.Priority*0.5)

	// Add small random noise to break ties
	score += rand.Float64() * 0.01

	return score
}

// min returns the minimum of two float64 values
func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}