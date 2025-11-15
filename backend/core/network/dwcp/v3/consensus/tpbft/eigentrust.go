// Package tpbft implements Trust-based PBFT with EigenTrust reputation system
package tpbft

import (
	"math"
	"sync"
	"time"
)

// EigenTrust implements the EigenTrust reputation algorithm for distributed trust computation
type EigenTrust struct {
	mu              sync.RWMutex
	trustScores     map[string]float64            // Global trust scores (normalized)
	localTrust      map[string]map[string]float64 // Local trust matrix: from -> to -> score
	preTrust        map[string]float64            // Pre-trusted peers (bootstrap)
	interactions    map[string]int                // Total interactions per node
	convergenceIter int                           // Iterations for convergence
	alpha           float64                       // Pre-trust weight (0.1-0.3)
	epsilon         float64                       // Convergence threshold
	lastUpdate      time.Time
}

// NewEigenTrust creates a new EigenTrust instance with default parameters
func NewEigenTrust() *EigenTrust {
	return &EigenTrust{
		trustScores:     make(map[string]float64),
		localTrust:      make(map[string]map[string]float64),
		preTrust:        make(map[string]float64),
		interactions:    make(map[string]int),
		convergenceIter: 10,  // Typical: 5-15 iterations
		alpha:           0.2, // 20% pre-trust weight
		epsilon:         0.01, // 1% convergence threshold
		lastUpdate:      time.Now(),
	}
}

// UpdateLocalTrust updates the local trust from one node to another based on interaction outcome
// score should be in [0, 1] range where 1 = fully trustworthy, 0 = completely untrustworthy
func (e *EigenTrust) UpdateLocalTrust(from, to string, score float64) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Initialize maps if needed
	if e.localTrust[from] == nil {
		e.localTrust[from] = make(map[string]float64)
	}

	// Update local trust with exponential moving average
	currentTrust := e.localTrust[from][to]
	e.localTrust[from][to] = 0.7*currentTrust + 0.3*score // 70% old, 30% new
	e.interactions[from]++
	e.interactions[to]++
}

// RecordSuccessfulInteraction records a positive interaction (e.g., correct consensus vote)
func (e *EigenTrust) RecordSuccessfulInteraction(from, to string) {
	e.UpdateLocalTrust(from, to, 1.0)
}

// RecordFailedInteraction records a negative interaction (e.g., Byzantine behavior)
func (e *EigenTrust) RecordFailedInteraction(from, to string) {
	e.UpdateLocalTrust(from, to, 0.0)
}

// ComputeGlobalTrust calculates global trust scores using iterative power method
// Implements: T = (C^T)^n * p
// where C is normalized local trust matrix, p is pre-trust vector
func (e *EigenTrust) ComputeGlobalTrust() {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Get all nodes
	nodes := e.getAllNodes()
	if len(nodes) == 0 {
		return
	}

	// Normalize local trust matrix (each row sums to 1)
	normalized := e.normalizeLocalTrust(nodes)

	// Initialize trust vector with equal distribution or pre-trust
	current := make(map[string]float64)
	for _, node := range nodes {
		if preTrust, exists := e.preTrust[node]; exists {
			current[node] = preTrust
		} else {
			current[node] = 1.0 / float64(len(nodes))
		}
	}

	// Power iteration to convergence
	for iter := 0; iter < e.convergenceIter; iter++ {
		next := make(map[string]float64)

		// Matrix multiplication: next = C^T * current
		for _, to := range nodes {
			score := 0.0
			for _, from := range nodes {
				if trust, exists := normalized[from][to]; exists {
					score += trust * current[from]
				}
			}

			// Mix with pre-trust: (1-α) * score + α * pre-trust
			if preTrust, exists := e.preTrust[to]; exists {
				next[to] = (1-e.alpha)*score + e.alpha*preTrust
			} else {
				next[to] = score
			}
		}

		// Check convergence
		if e.hasConverged(current, next) {
			break
		}
		current = next
	}

	// Store final trust scores
	e.trustScores = current
	e.lastUpdate = time.Now()
}

// normalizeLocalTrust creates a column-stochastic matrix (columns sum to 1)
func (e *EigenTrust) normalizeLocalTrust(nodes []string) map[string]map[string]float64 {
	normalized := make(map[string]map[string]float64)

	for _, from := range nodes {
		normalized[from] = make(map[string]float64)
		sum := 0.0

		// Calculate sum of local trust from this node
		for _, to := range nodes {
			if trust, exists := e.localTrust[from][to]; exists {
				sum += trust
			}
		}

		// Normalize (if sum > 0)
		if sum > 0 {
			for _, to := range nodes {
				if trust, exists := e.localTrust[from][to]; exists {
					normalized[from][to] = trust / sum
				}
			}
		} else {
			// No outgoing trust - distribute equally
			for _, to := range nodes {
				normalized[from][to] = 1.0 / float64(len(nodes))
			}
		}
	}

	return normalized
}

// hasConverged checks if trust scores have converged within epsilon
func (e *EigenTrust) hasConverged(prev, current map[string]float64) bool {
	for node := range current {
		diff := math.Abs(current[node] - prev[node])
		if diff > e.epsilon {
			return false
		}
	}
	return true
}

// GetTrustScore returns the global trust score for a node
func (e *EigenTrust) GetTrustScore(nodeID string) float64 {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.trustScores[nodeID]
}

// GetTopNodes returns the N most trusted nodes (for committee selection)
func (e *EigenTrust) GetTopNodes(n int) []string {
	e.mu.RLock()
	defer e.mu.RUnlock()

	type nodeTrust struct {
		node  string
		trust float64
	}

	// Create sorted list
	nodes := make([]nodeTrust, 0, len(e.trustScores))
	for node, trust := range e.trustScores {
		nodes = append(nodes, nodeTrust{node, trust})
	}

	// Sort by trust descending
	for i := 0; i < len(nodes)-1; i++ {
		for j := i + 1; j < len(nodes); j++ {
			if nodes[j].trust > nodes[i].trust {
				nodes[i], nodes[j] = nodes[j], nodes[i]
			}
		}
	}

	// Return top N
	result := make([]string, 0, n)
	limit := n
	if limit > len(nodes) {
		limit = len(nodes)
	}
	for i := 0; i < limit; i++ {
		result = append(result, nodes[i].node)
	}

	return result
}

// SetPreTrust sets pre-trusted nodes (bootstrap trust)
func (e *EigenTrust) SetPreTrust(nodeID string, trust float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.preTrust[nodeID] = trust
}

// getAllNodes returns all nodes that have participated in trust calculations
func (e *EigenTrust) getAllNodes() []string {
	nodeSet := make(map[string]bool)

	// Collect from local trust matrix
	for from := range e.localTrust {
		nodeSet[from] = true
		for to := range e.localTrust[from] {
			nodeSet[to] = true
		}
	}

	// Collect from pre-trust
	for node := range e.preTrust {
		nodeSet[node] = true
	}

	// Convert to slice
	nodes := make([]string, 0, len(nodeSet))
	for node := range nodeSet {
		nodes = append(nodes, node)
	}
	return nodes
}

// GetStats returns statistics about the trust system
func (e *EigenTrust) GetStats() map[string]interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	avgTrust := 0.0
	minTrust := 1.0
	maxTrust := 0.0

	for _, trust := range e.trustScores {
		avgTrust += trust
		if trust < minTrust {
			minTrust = trust
		}
		if trust > maxTrust {
			maxTrust = trust
		}
	}

	if len(e.trustScores) > 0 {
		avgTrust /= float64(len(e.trustScores))
	}

	return map[string]interface{}{
		"total_nodes":       len(e.trustScores),
		"avg_trust":         avgTrust,
		"min_trust":         minTrust,
		"max_trust":         maxTrust,
		"total_interactions": e.sumInteractions(),
		"last_update":       e.lastUpdate,
	}
}

// sumInteractions calculates total interactions
func (e *EigenTrust) sumInteractions() int {
	total := 0
	for _, count := range e.interactions {
		total += count
	}
	return total
}
