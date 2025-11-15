// Package probft implements probabilistic quorum calculations for Byzantine fault tolerance
package probft

import (
	"fmt"
	"math"
)

// QuorumConfig holds configuration for probabilistic quorum calculations
type QuorumConfig struct {
	TotalNodes      int     // Total number of nodes in the network
	ByzantineNodes  int     // Maximum number of Byzantine (faulty) nodes
	SecurityParam   float64 // Security parameter for probabilistic guarantees
	ConfidenceLevel float64 // Desired confidence level (0.0-1.0)
}

// QuorumResult contains the calculated quorum size and related metrics
type QuorumResult struct {
	QuorumSize         int     // Minimum quorum size needed
	ByzantineTolerance float64 // Percentage of Byzantine nodes tolerated
	SafetyMargin       int     // Additional nodes beyond minimum
	IsValid            bool    // Whether configuration is Byzantine-safe
}

// CalculateQuorum computes probabilistic quorum size
// Formula: q = ⌈√n⌉ for basic probabilistic BFT
func CalculateQuorum(n int) int {
	if n <= 0 {
		return 0
	}
	return int(math.Ceil(math.Sqrt(float64(n))))
}

// CalculateClassicalQuorum computes classical BFT quorum (⌈(n+f)/2⌉ + 1)
func CalculateClassicalQuorum(n, f int) int {
	if n <= 0 {
		return 0
	}
	return ((n + f) / 2) + 1
}

// CalculatePBFTQuorum computes PBFT-style quorum (2f + 1)
func CalculatePBFTQuorum(f int) int {
	return 2*f + 1
}

// IsByzantineTolerant checks if configuration can tolerate f Byzantine nodes
// For Byzantine tolerance: n ≥ 3f + 1
func IsByzantineTolerant(n, f int) bool {
	if n <= 0 || f < 0 {
		return false
	}
	return n >= 3*f+1
}

// CalculateMaxByzantineNodes returns maximum Byzantine nodes for given total
// Formula: f_max = ⌊(n-1)/3⌋ for 33% Byzantine tolerance
func CalculateMaxByzantineNodes(n int) int {
	if n < 4 {
		return 0
	}
	return (n - 1) / 3
}

// CalculateByzantineTolerance returns the Byzantine tolerance percentage
func CalculateByzantineTolerance(n, f int) float64 {
	if n <= 0 {
		return 0.0
	}
	return (float64(f) / float64(n)) * 100.0
}

// ValidateQuorumConfig validates the quorum configuration
func ValidateQuorumConfig(config QuorumConfig) error {
	if config.TotalNodes <= 0 {
		return fmt.Errorf("total nodes must be positive, got %d", config.TotalNodes)
	}

	if config.ByzantineNodes < 0 {
		return fmt.Errorf("byzantine nodes cannot be negative, got %d", config.ByzantineNodes)
	}

	if config.ByzantineNodes >= config.TotalNodes {
		return fmt.Errorf("byzantine nodes (%d) must be less than total nodes (%d)",
			config.ByzantineNodes, config.TotalNodes)
	}

	if !IsByzantineTolerant(config.TotalNodes, config.ByzantineNodes) {
		return fmt.Errorf("configuration is not Byzantine tolerant: n=%d, f=%d (requires n ≥ 3f+1)",
			config.TotalNodes, config.ByzantineNodes)
	}

	if config.ConfidenceLevel < 0.0 || config.ConfidenceLevel > 1.0 {
		return fmt.Errorf("confidence level must be between 0.0 and 1.0, got %.2f",
			config.ConfidenceLevel)
	}

	return nil
}

// CalculateProbabilisticQuorum calculates quorum with probabilistic guarantees
func CalculateProbabilisticQuorum(config QuorumConfig) (*QuorumResult, error) {
	if err := ValidateQuorumConfig(config); err != nil {
		return nil, err
	}

	n := config.TotalNodes
	f := config.ByzantineNodes

	// Base probabilistic quorum
	baseQuorum := CalculateQuorum(n)

	// Classical BFT quorum for comparison
	classicalQuorum := CalculateClassicalQuorum(n, f)

	// Use the larger of the two for safety
	quorumSize := baseQuorum
	if classicalQuorum > baseQuorum {
		quorumSize = classicalQuorum
	}

	// Add security margin based on confidence level
	securityMargin := int(math.Ceil(float64(f) * config.ConfidenceLevel))
	quorumSize += securityMargin

	// Ensure quorum doesn't exceed total nodes
	if quorumSize > n {
		quorumSize = n
	}

	tolerance := CalculateByzantineTolerance(n, f)

	return &QuorumResult{
		QuorumSize:         quorumSize,
		ByzantineTolerance: tolerance,
		SafetyMargin:       securityMargin,
		IsValid:            IsByzantineTolerant(n, f),
	}, nil
}

// CalculateMinimumNodes calculates minimum nodes needed for desired Byzantine tolerance
func CalculateMinimumNodes(maxByzantineNodes int) int {
	return 3*maxByzantineNodes + 1
}

// QuorumIntersection checks if two quorums must intersect in honest nodes
// For Byzantine safety, any two quorums must intersect in at least f+1 nodes
func QuorumIntersection(n, f, quorumSize int) bool {
	// Two quorums of size q must intersect in at least f+1 honest nodes
	// This requires: 2q - n ≥ f + 1
	minIntersection := 2*quorumSize - n
	return minIntersection >= f+1
}

// ValidateQuorumIntersection validates that quorum size ensures safety
func ValidateQuorumIntersection(config QuorumConfig, quorumSize int) error {
	if !QuorumIntersection(config.TotalNodes, config.ByzantineNodes, quorumSize) {
		return fmt.Errorf("quorum size %d insufficient for Byzantine safety (n=%d, f=%d)",
			quorumSize, config.TotalNodes, config.ByzantineNodes)
	}
	return nil
}

// AdaptiveQuorum calculates adaptive quorum based on network conditions
type NetworkCondition struct {
	ActiveNodes    int     // Currently active nodes
	Latency        float64 // Average network latency (ms)
	FailureRate    float64 // Node failure rate (0.0-1.0)
	ByzantineRatio float64 // Estimated Byzantine ratio (0.0-1.0)
}

// CalculateAdaptiveQuorum adjusts quorum size based on network conditions
func CalculateAdaptiveQuorum(config QuorumConfig, condition NetworkCondition) (*QuorumResult, error) {
	if err := ValidateQuorumConfig(config); err != nil {
		return nil, err
	}

	// Start with base probabilistic quorum
	result, err := CalculateProbabilisticQuorum(config)
	if err != nil {
		return nil, err
	}

	// Adjust for network conditions
	adjustmentFactor := 1.0

	// Increase quorum if high latency (reduces availability, increases safety)
	if condition.Latency > 100 { // High latency threshold
		adjustmentFactor += 0.1
	}

	// Increase quorum if high failure rate
	if condition.FailureRate > 0.1 { // 10% failure threshold
		adjustmentFactor += condition.FailureRate
	}

	// Increase quorum if high Byzantine ratio suspected
	if condition.ByzantineRatio > 0.2 { // 20% Byzantine threshold
		adjustmentFactor += condition.ByzantineRatio
	}

	// Apply adjustment
	adjustedQuorum := int(math.Ceil(float64(result.QuorumSize) * adjustmentFactor))

	// Ensure we don't exceed active nodes
	if adjustedQuorum > condition.ActiveNodes {
		adjustedQuorum = condition.ActiveNodes
	}

	result.QuorumSize = adjustedQuorum

	return result, nil
}
