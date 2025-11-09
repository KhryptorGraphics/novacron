package multiregion

import (
	"fmt"
	"sync"
	"time"
)

// OptimizationAlgorithm defines traffic optimization strategy
type OptimizationAlgorithm int

const (
	AlgorithmECMP  OptimizationAlgorithm = iota // Equal-Cost Multi-Path
	AlgorithmWECMP                               // Weighted ECMP
	AlgorithmTE                                  // Traffic Engineering (MPLS-TE style)
)

// TrafficFlow represents a network traffic flow
type TrafficFlow struct {
	ID          string
	Source      string
	Destination string
	Size        int64 // Bytes
	Priority    int
	QoS         QoSClass
	Deadline    time.Time
	CreatedAt   time.Time
}

// QoSClass defines Quality of Service class
type QoSClass int

const (
	QoSBestEffort QoSClass = iota
	QoSBulk
	QoSInteractive
	QoSRealtime
	QoSCritical
)

// PathOptimizer optimizes path selection
type PathOptimizer struct {
	algorithm OptimizationAlgorithm
	engine    *RoutingEngine
}

// TrafficEngineer manages traffic distribution
type TrafficEngineer struct {
	topology  *GlobalTopology
	optimizer *PathOptimizer
	flows     map[string]*TrafficFlow
	stats     *TrafficStats
	mu        sync.RWMutex
}

// TrafficStats tracks traffic statistics
type TrafficStats struct {
	TotalFlows     uint64
	ActiveFlows    uint64
	BytesTransmitted uint64
	PacketsTransmitted uint64
	AverageLatency time.Duration
	mu             sync.RWMutex
}

// FlowDistribution represents how a flow is distributed across paths
type FlowDistribution struct {
	FlowID      string
	Allocations []PathAllocation
}

// PathAllocation represents traffic allocation to a path
type PathAllocation struct {
	Path       *Route
	Percentage float64
	Bandwidth  int64
}

// NewTrafficEngineer creates a new traffic engineer
func NewTrafficEngineer(topology *GlobalTopology, algorithm OptimizationAlgorithm) *TrafficEngineer {
	engine := NewRoutingEngine(topology, StrategyBalanced)

	return &TrafficEngineer{
		topology: topology,
		optimizer: &PathOptimizer{
			algorithm: algorithm,
			engine:    engine,
		},
		flows: make(map[string]*TrafficFlow),
		stats: &TrafficStats{},
	}
}

// DistributeTraffic distributes a traffic flow across optimal paths
func (te *TrafficEngineer) DistributeTraffic(flow *TrafficFlow) error {
	// Store flow
	te.mu.Lock()
	te.flows[flow.ID] = flow
	te.stats.TotalFlows++
	te.stats.ActiveFlows++
	te.mu.Unlock()

	var err error
	switch te.optimizer.algorithm {
	case AlgorithmECMP:
		err = te.distributeECMP(flow)
	case AlgorithmWECMP:
		err = te.distributeWECMP(flow)
	case AlgorithmTE:
		err = te.optimizeTE(flow)
	default:
		err = fmt.Errorf("unknown algorithm: %v", te.optimizer.algorithm)
	}

	if err != nil {
		te.mu.Lock()
		te.stats.ActiveFlows--
		te.mu.Unlock()
		return err
	}

	return nil
}

// distributeECMP distributes traffic equally across equal-cost paths
func (te *TrafficEngineer) distributeECMP(flow *TrafficFlow) error {
	paths, err := te.optimizer.engine.FindEqualCostPaths(flow.Source, flow.Destination)
	if err != nil {
		return fmt.Errorf("failed to find paths: %w", err)
	}

	if len(paths) == 0 {
		return fmt.Errorf("no paths available")
	}

	// Distribute equally
	sharePerPath := flow.Size / int64(len(paths))

	for _, path := range paths {
		if err := te.sendTraffic(path, sharePerPath); err != nil {
			return err
		}
	}

	return nil
}

// distributeWECMP distributes traffic with weighted distribution based on capacity
func (te *TrafficEngineer) distributeWECMP(flow *TrafficFlow) error {
	paths, err := te.optimizer.engine.FindEqualCostPaths(flow.Source, flow.Destination)
	if err != nil {
		return fmt.Errorf("failed to find paths: %w", err)
	}

	if len(paths) == 0 {
		return fmt.Errorf("no paths available")
	}

	// Compute weights based on available bandwidth
	weights := make([]float64, len(paths))
	totalWeight := 0.0

	for i, path := range paths {
		availableBW := te.getPathAvailableBandwidth(path)
		weights[i] = float64(availableBW)
		totalWeight += weights[i]
	}

	// Distribute flow proportionally
	for i, path := range paths {
		if totalWeight == 0 {
			return fmt.Errorf("no available bandwidth")
		}
		flowShare := flow.Size * int64(weights[i]) / int64(totalWeight)
		if err := te.sendTraffic(path, flowShare); err != nil {
			return err
		}
	}

	return nil
}

// optimizeTE performs traffic engineering optimization
func (te *TrafficEngineer) optimizeTE(flow *TrafficFlow) error {
	// For TE, we consider QoS requirements and optimize accordingly
	var strategy RoutingStrategy

	switch flow.QoS {
	case QoSRealtime, QoSCritical:
		strategy = StrategyLatency
	case QoSBulk:
		strategy = StrategyCost
	case QoSInteractive:
		strategy = StrategyBalanced
	default:
		strategy = StrategyBandwidth
	}

	// Create custom engine with appropriate strategy
	engine := NewRoutingEngine(te.topology, strategy)
	route, err := engine.ComputeRoute(flow.Source, flow.Destination)
	if err != nil {
		return err
	}

	// Reserve bandwidth if needed
	if flow.QoS >= QoSInteractive {
		requiredBW := te.estimateBandwidth(flow)
		if err := te.reserveBandwidth(route, requiredBW); err != nil {
			return fmt.Errorf("bandwidth reservation failed: %w", err)
		}
	}

	return te.sendTraffic(route, flow.Size)
}

// sendTraffic sends traffic along a specific path
func (te *TrafficEngineer) sendTraffic(path *Route, size int64) error {
	// Update link utilization
	for _, linkID := range path.Links {
		link, err := te.topology.GetLink(linkID)
		if err != nil {
			return err
		}

		link.mu.Lock()
		// Simulate traffic transmission
		if link.Metrics != nil {
			link.Metrics.BytesSent += uint64(size)
			link.Metrics.PacketsSent++
		}
		link.mu.Unlock()
	}

	// Update stats
	te.stats.mu.Lock()
	te.stats.BytesTransmitted += uint64(size)
	te.stats.PacketsTransmitted++
	te.stats.mu.Unlock()

	return nil
}

// getPathAvailableBandwidth calculates available bandwidth on a path (bottleneck)
func (te *TrafficEngineer) getPathAvailableBandwidth(path *Route) int64 {
	minAvailable := int64(^uint64(0) >> 1) // Max int64

	for _, linkID := range path.Links {
		link, err := te.topology.GetLink(linkID)
		if err != nil {
			continue
		}

		link.mu.RLock()
		available := link.Bandwidth * (100 - int64(link.Utilization)) / 100
		link.mu.RUnlock()

		if available < minAvailable {
			minAvailable = available
		}
	}

	return minAvailable
}

// estimateBandwidth estimates required bandwidth for a flow
func (te *TrafficEngineer) estimateBandwidth(flow *TrafficFlow) int64 {
	// Simple estimation: assume transmission over 1 second
	// In production, this would be more sophisticated
	bitsPerSecond := flow.Size * 8
	mbps := bitsPerSecond / (1024 * 1024)

	// Add 20% overhead for protocol headers, retransmissions, etc.
	return mbps * 120 / 100
}

// reserveBandwidth reserves bandwidth on a path
func (te *TrafficEngineer) reserveBandwidth(path *Route, bandwidth int64) error {
	// Check if bandwidth is available on all links
	for _, linkID := range path.Links {
		link, err := te.topology.GetLink(linkID)
		if err != nil {
			return err
		}

		link.mu.RLock()
		available := link.Bandwidth * (100 - int64(link.Utilization)) / 100
		link.mu.RUnlock()

		if available < bandwidth {
			return fmt.Errorf("insufficient bandwidth on link %s", linkID)
		}
	}

	// Reserve bandwidth by increasing utilization
	for _, linkID := range path.Links {
		link, err := te.topology.GetLink(linkID)
		if err != nil {
			return err
		}

		link.mu.Lock()
		utilizationIncrease := float64(bandwidth) / float64(link.Bandwidth) * 100.0
		link.Utilization += utilizationIncrease
		link.mu.Unlock()
	}

	return nil
}

// GetFlowDistribution returns the distribution plan for a flow
func (te *TrafficEngineer) GetFlowDistribution(flowID string) (*FlowDistribution, error) {
	te.mu.RLock()
	flow, exists := te.flows[flowID]
	te.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("flow %s not found", flowID)
	}

	// Compute distribution based on algorithm
	paths, err := te.optimizer.engine.FindEqualCostPaths(flow.Source, flow.Destination)
	if err != nil {
		return nil, err
	}

	distribution := &FlowDistribution{
		FlowID:      flowID,
		Allocations: make([]PathAllocation, 0, len(paths)),
	}

	switch te.optimizer.algorithm {
	case AlgorithmECMP:
		// Equal distribution
		percentage := 100.0 / float64(len(paths))
		bandwidth := flow.Size / int64(len(paths))
		for _, path := range paths {
			distribution.Allocations = append(distribution.Allocations, PathAllocation{
				Path:       path,
				Percentage: percentage,
				Bandwidth:  bandwidth,
			})
		}

	case AlgorithmWECMP:
		// Weighted distribution
		weights := make([]float64, len(paths))
		totalWeight := 0.0
		for i, path := range paths {
			weights[i] = float64(te.getPathAvailableBandwidth(path))
			totalWeight += weights[i]
		}

		for i, path := range paths {
			percentage := weights[i] / totalWeight * 100.0
			bandwidth := int64(float64(flow.Size) * weights[i] / totalWeight)
			distribution.Allocations = append(distribution.Allocations, PathAllocation{
				Path:       path,
				Percentage: percentage,
				Bandwidth:  bandwidth,
			})
		}
	}

	return distribution, nil
}

// CompleteFlow marks a flow as complete
func (te *TrafficEngineer) CompleteFlow(flowID string) error {
	te.mu.Lock()
	defer te.mu.Unlock()

	if _, exists := te.flows[flowID]; !exists {
		return fmt.Errorf("flow %s not found", flowID)
	}

	delete(te.flows, flowID)
	te.stats.ActiveFlows--

	return nil
}

// GetStats returns traffic statistics
func (te *TrafficEngineer) GetStats() *TrafficStats {
	te.stats.mu.RLock()
	defer te.stats.mu.RUnlock()

	// Return a copy
	return &TrafficStats{
		TotalFlows:         te.stats.TotalFlows,
		ActiveFlows:        te.stats.ActiveFlows,
		BytesTransmitted:   te.stats.BytesTransmitted,
		PacketsTransmitted: te.stats.PacketsTransmitted,
		AverageLatency:     te.stats.AverageLatency,
	}
}

// OptimizeGlobalTraffic performs global traffic optimization
func (te *TrafficEngineer) OptimizeGlobalTraffic() error {
	te.mu.RLock()
	flows := make([]*TrafficFlow, 0, len(te.flows))
	for _, flow := range te.flows {
		flows = append(flows, flow)
	}
	te.mu.RUnlock()

	// Sort flows by priority (highest first)
	// In production, use proper sorting

	// Redistribute flows based on current network state
	for _, flow := range flows {
		// Re-optimize each flow
		if err := te.DistributeTraffic(flow); err != nil {
			// Log error but continue with other flows
			continue
		}
	}

	return nil
}

// String methods

func (qos QoSClass) String() string {
	switch qos {
	case QoSBestEffort:
		return "BestEffort"
	case QoSBulk:
		return "Bulk"
	case QoSInteractive:
		return "Interactive"
	case QoSRealtime:
		return "Realtime"
	case QoSCritical:
		return "Critical"
	default:
		return "Unknown"
	}
}

func (algo OptimizationAlgorithm) String() string {
	switch algo {
	case AlgorithmECMP:
		return "ECMP"
	case AlgorithmWECMP:
		return "WECMP"
	case AlgorithmTE:
		return "TrafficEngineering"
	default:
		return "Unknown"
	}
}
