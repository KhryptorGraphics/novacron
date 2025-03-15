package scheduler

import (
	"context"
	"fmt"
	"log"
	"math"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/network"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler/workload"
)

// VMCommunication represents communication between two VMs
type VMCommunication struct {
	// SourceVMID is the source VM ID
	SourceVMID string

	// DestinationVMID is the destination VM ID
	DestinationVMID string

	// Bandwidth is the bandwidth used in Mbps
	Bandwidth float64

	// PacketRate is the packets per second
	PacketRate float64

	// LastUpdated is when this communication was last updated
	LastUpdated time.Time

	// Latency is the required latency in milliseconds (0 means no requirement)
	Latency float64

	// Priority is the priority of this communication (higher is more important)
	Priority float64
}

// NetworkAwareSchedulerConfig extends ResourceAwareSchedulerConfig with network settings
type NetworkAwareSchedulerConfig struct {
	// Base scheduler configuration
	ResourceAwareSchedulerConfig

	// NetworkAwarenessWeight is the weight of network awareness in scheduling (0.0-1.0)
	NetworkAwarenessWeight float64

	// VMCommunicationWeight is the weight of VM communication in network scores
	VMCommunicationWeight float64

	// BandwidthWeight is the weight of bandwidth in network scores
	BandwidthWeight float64

	// LatencyWeight is the weight of latency in network scores
	LatencyWeight float64

	// ZoneProximityWeight is the weight of zone proximity in network scores
	ZoneProximityWeight float64

	// TopologyUpdateInterval is how often to update the network topology data
	TopologyUpdateInterval time.Duration

	// CommunicationTimeWindow is the window to consider for VM communication
	CommunicationTimeWindow time.Duration
}

// DefaultNetworkAwareSchedulerConfig returns a default configuration
func DefaultNetworkAwareSchedulerConfig() NetworkAwareSchedulerConfig {
	baseConfig := DefaultResourceAwareSchedulerConfig()

	return NetworkAwareSchedulerConfig{
		ResourceAwareSchedulerConfig: baseConfig,
		NetworkAwarenessWeight:       0.3,
		VMCommunicationWeight:        0.4,
		BandwidthWeight:              0.3,
		LatencyWeight:                0.3,
		ZoneProximityWeight:          0.2,
		TopologyUpdateInterval:       5 * time.Minute,
		CommunicationTimeWindow:      30 * time.Minute,
	}
}

// NetworkAwareScheduler extends ResourceAwareScheduler with network awareness
type NetworkAwareScheduler struct {
	*ResourceAwareScheduler
	config NetworkAwareSchedulerConfig

	// networkTopology is the network topology of the cluster
	networkTopology *network.NetworkTopology

	// vmCommunications maps VM pairs to their communication statistics
	vmCommunications     map[string]*VMCommunication
	vmCommunicationMutex sync.RWMutex

	// vmLocationCache maps VM IDs to their host node IDs
	vmLocationCache      map[string]string
	vmLocationCacheMutex sync.RWMutex

	// vmAffinityGroups maps group IDs to VM IDs with communication affinity
	vmAffinityGroups     map[string][]string
	vmAffinityGroupMutex sync.RWMutex

	topologyUpdateTicker *time.Ticker
	topologyUpdateDone   chan struct{}
}

// NewNetworkAwareScheduler creates a new network-aware scheduler
func NewNetworkAwareScheduler(
	config NetworkAwareSchedulerConfig,
	baseScheduler *Scheduler,
	workloadAnalyzer *workload.WorkloadAnalyzer,
	networkTopology *network.NetworkTopology,
) *NetworkAwareScheduler {
	// Create base resource-aware scheduler
	resourceScheduler := NewResourceAwareScheduler(
		config.ResourceAwareSchedulerConfig,
		baseScheduler,
		workloadAnalyzer,
		nil, // We'll initialize the migration cost estimator separately
	)

	scheduler := &NetworkAwareScheduler{
		ResourceAwareScheduler: resourceScheduler,
		config:                 config,
		networkTopology:        networkTopology,
		vmCommunications:       make(map[string]*VMCommunication),
		vmLocationCache:        make(map[string]string),
		vmAffinityGroups:       make(map[string][]string),
		topologyUpdateDone:     make(chan struct{}),
	}

	return scheduler
}

// Start starts the network-aware scheduler
func (s *NetworkAwareScheduler) Start() error {
	// Start the base scheduler
	err := s.ResourceAwareScheduler.Start()
	if err != nil {
		return fmt.Errorf("failed to start base scheduler: %w", err)
	}

	// Start the topology update loop
	s.topologyUpdateTicker = time.NewTicker(s.config.TopologyUpdateInterval)
	go s.topologyUpdateLoop()

	log.Println("Network-aware scheduler started")
	return nil
}

// Stop stops the network-aware scheduler
func (s *NetworkAwareScheduler) Stop() error {
	// Stop the topology update loop
	if s.topologyUpdateTicker != nil {
		s.topologyUpdateTicker.Stop()
		close(s.topologyUpdateDone)
	}

	// Stop the base scheduler
	err := s.ResourceAwareScheduler.Stop()
	if err != nil {
		return fmt.Errorf("failed to stop base scheduler: %w", err)
	}

	log.Println("Network-aware scheduler stopped")
	return nil
}

// topologyUpdateLoop updates the network topology data periodically
func (s *NetworkAwareScheduler) topologyUpdateLoop() {
	for {
		select {
		case <-s.topologyUpdateTicker.C:
			s.updateNetworkTopologyData()
		case <-s.topologyUpdateDone:
			return
		}
	}
}

// updateNetworkTopologyData updates network topology data
func (s *NetworkAwareScheduler) updateNetworkTopologyData() {
	// In a real implementation, this would collect data from monitoring systems
	// For now, just update link utilization based on VM communication
	s.updateLinkUtilizationFromVMCommunication()
}

// updateLinkUtilizationFromVMCommunication updates link utilization based on VM communication
func (s *NetworkAwareScheduler) updateLinkUtilizationFromVMCommunication() {
	s.vmCommunicationMutex.RLock()
	defer s.vmCommunicationMutex.RUnlock()

	s.vmLocationCacheMutex.RLock()
	defer s.vmLocationCacheMutex.RUnlock()

	// Map to accumulate utilization between node pairs
	nodeUtilization := make(map[string]float64)

	for _, comm := range s.vmCommunications {
		// Skip old entries
		if time.Since(comm.LastUpdated) > s.config.CommunicationTimeWindow {
			continue
		}

		// Get node IDs for the communicating VMs
		sourceNodeID, sourceExists := s.vmLocationCache[comm.SourceVMID]
		destNodeID, destExists := s.vmLocationCache[comm.DestinationVMID]

		if !sourceExists || !destExists {
			continue
		}

		// If VMs are on different nodes, record the utilization
		if sourceNodeID != destNodeID {
			key := fmt.Sprintf("%s:%s", sourceNodeID, destNodeID)
			// Use bandwidth as approximation of utilization
			// In a real system, would need to compare with link capacity
			nodeUtilization[key] += comm.Bandwidth / 1000.0 // Normalize to Gbps
		}
	}

	// Update the network topology with the utilization data
	for key, utilization := range nodeUtilization {
		// Parse the key to get source and destination
		var sourceID, destID string
		fmt.Sscanf(key, "%s:%s", &sourceID, &destID)

		// Try to update the link
		err := s.networkTopology.UpdateLinkUtilization(sourceID, destID, utilization)
		if err != nil {
			// Link might not exist in topology, can create it if needed
			log.Printf("Warning: Could not update link utilization for %s -> %s: %v", sourceID, destID, err)
		}
	}
}

// TrackVMCommunication tracks communication between two VMs
func (s *NetworkAwareScheduler) TrackVMCommunication(sourceVMID, destVMID string, bandwidth, packetRate float64) {
	s.vmCommunicationMutex.Lock()
	defer s.vmCommunicationMutex.Unlock()

	// Create a unique key for the VM pair
	key := getVMPairKey(sourceVMID, destVMID)

	// Update existing or create new entry
	if comm, exists := s.vmCommunications[key]; exists {
		// Update existing communication
		comm.Bandwidth = bandwidth
		comm.PacketRate = packetRate
		comm.LastUpdated = time.Now()
	} else {
		// Create new communication entry
		s.vmCommunications[key] = &VMCommunication{
			SourceVMID:      sourceVMID,
			DestinationVMID: destVMID,
			Bandwidth:       bandwidth,
			PacketRate:      packetRate,
			LastUpdated:     time.Now(),
			Priority:        0.5, // Default priority
		}
	}
}

// SetVMLatencyRequirement sets a latency requirement for VM communication
func (s *NetworkAwareScheduler) SetVMLatencyRequirement(sourceVMID, destVMID string, latency float64) {
	s.vmCommunicationMutex.Lock()
	defer s.vmCommunicationMutex.Unlock()

	// Create a unique key for the VM pair
	key := getVMPairKey(sourceVMID, destVMID)

	// Update existing or create new entry
	if comm, exists := s.vmCommunications[key]; exists {
		// Update existing communication
		comm.Latency = latency
		comm.LastUpdated = time.Now()
	} else {
		// Create new communication entry with just the latency requirement
		s.vmCommunications[key] = &VMCommunication{
			SourceVMID:      sourceVMID,
			DestinationVMID: destVMID,
			Latency:         latency,
			LastUpdated:     time.Now(),
			Priority:        0.5, // Default priority
		}
	}
}

// UpdateVMLocation updates the location of a VM
func (s *NetworkAwareScheduler) UpdateVMLocation(vmID, nodeID string) {
	// Update in base scheduler
	s.ResourceAwareScheduler.UpdateVMPlacement(vmID, nodeID)

	// Update our location cache
	s.vmLocationCacheMutex.Lock()
	defer s.vmLocationCacheMutex.Unlock()
	s.vmLocationCache[vmID] = nodeID
}

// CreateVMAffinityGroup creates a VM affinity group
func (s *NetworkAwareScheduler) CreateVMAffinityGroup(groupID string, vmIDs []string) error {
	s.vmAffinityGroupMutex.Lock()
	defer s.vmAffinityGroupMutex.Unlock()

	if _, exists := s.vmAffinityGroups[groupID]; exists {
		return fmt.Errorf("affinity group already exists: %s", groupID)
	}

	s.vmAffinityGroups[groupID] = vmIDs
	return nil
}

// getVMPairKey generates a unique key for a VM pair
func getVMPairKey(vm1, vm2 string) string {
	// Ensure consistent ordering of VM IDs
	if vm1 < vm2 {
		return fmt.Sprintf("%s:%s", vm1, vm2)
	}
	return fmt.Sprintf("%s:%s", vm2, vm1)
}

// Override scoreNode to include network topology awareness
func (s *NetworkAwareScheduler) scoreNode(
	ctx context.Context,
	request *PlacementRequest,
	nodeID string,
	currentNodeID string,
	vmProfile *workload.WorkloadProfile,
	constraints []PlacementConstraint,
) (float64, map[string]float64, []PlacementConstraint, []PlacementConstraint, error) {
	// Get base scoring from resource-aware scheduler
	baseScore, componentScores, satisfied, violated, err := s.ResourceAwareScheduler.scoreNode(
		ctx, request, nodeID, currentNodeID, vmProfile, constraints)
	if err != nil {
		return 0, nil, nil, nil, err
	}

	// Add network topology scoring
	networkScore := s.scoreNetworkTopology(request.VMID, nodeID)
	componentScores["network_topology"] = networkScore

	// Adjust overall score to include network awareness
	if s.config.NetworkAwarenessWeight > 0 {
		// Recalculate weighted score to include network component
		totalScore := 0.0
		totalWeight := 0.0

		// Add all component scores with their weights
		for component, score := range componentScores {
			var weight float64
			if component == "network_topology" {
				weight = s.config.NetworkAwarenessWeight
			} else {
				weight = s.config.SchedulingWeights[component]
			}

			if weight > 0 {
				totalScore += score * weight
				totalWeight += weight
			}
		}

		// Normalize score
		if totalWeight > 0 {
			return totalScore / totalWeight, componentScores, satisfied, violated, nil
		}
	}

	return baseScore, componentScores, satisfied, violated, nil
}

// scoreNetworkTopology scores a node based on network topology considerations
func (s *NetworkAwareScheduler) scoreNetworkTopology(vmID, nodeID string) float64 {
	// Compose score from multiple factors
	var scores []float64
	var weights []float64

	// 1. Score based on communicating VMs (VMs this VM communicates with)
	commScore, hasComm := s.scoreCommunicatingVMs(vmID, nodeID)
	if hasComm {
		scores = append(scores, commScore)
		weights = append(weights, s.config.VMCommunicationWeight)
	}

	// 2. Score based on bandwidth availability
	bwScore, hasBW := s.scoreBandwidthAvailability(nodeID)
	if hasBW {
		scores = append(scores, bwScore)
		weights = append(weights, s.config.BandwidthWeight)
	}

	// 3. Score based on latency characteristics
	latencyScore, hasLatency := s.scoreLatencyCharacteristics(vmID, nodeID)
	if hasLatency {
		scores = append(scores, latencyScore)
		weights = append(weights, s.config.LatencyWeight)
	}

	// 4. Score based on zone proximity for availability
	zoneScore := s.scoreZoneProximity(vmID, nodeID)
	scores = append(scores, zoneScore)
	weights = append(weights, s.config.ZoneProximityWeight)

	// If no scores were calculated, return neutral score
	if len(scores) == 0 {
		return 0.5
	}

	// Calculate weighted score
	totalScore := 0.0
	totalWeight := 0.0
	for i := range scores {
		totalScore += scores[i] * weights[i]
		totalWeight += weights[i]
	}

	if totalWeight > 0 {
		return totalScore / totalWeight
	}

	return 0.5 // Neutral score
}

// scoreCommunicatingVMs scores a node based on the location of VMs that communicate with the given VM
func (s *NetworkAwareScheduler) scoreCommunicatingVMs(vmID, nodeID string) (float64, bool) {
	s.vmCommunicationMutex.RLock()
	s.vmLocationCacheMutex.RLock()
	defer s.vmCommunicationMutex.RUnlock()
	defer s.vmLocationCacheMutex.RUnlock()

	// Find all VMs that communicate with this VM
	communicatingVMs := make(map[string]float64)
	totalBandwidth := 0.0

	for _, comm := range s.vmCommunications {
		// Skip old entries
		if time.Since(comm.LastUpdated) > s.config.CommunicationTimeWindow {
			continue
		}

		if comm.SourceVMID == vmID {
			communicatingVMs[comm.DestinationVMID] = comm.Bandwidth
			totalBandwidth += comm.Bandwidth
		} else if comm.DestinationVMID == vmID {
			communicatingVMs[comm.SourceVMID] = comm.Bandwidth
			totalBandwidth += comm.Bandwidth
		}
	}

	if len(communicatingVMs) == 0 || totalBandwidth == 0 {
		return 0.0, false
	}

	// Calculate score based on network costs to communicating VMs
	weightedCosts := 0.0

	for commVM, bandwidth := range communicatingVMs {
		// Get location of communicating VM
		commNodeID, exists := s.vmLocationCache[commVM]
		if !exists {
			continue
		}

		// Get network cost between potential node and communicating VM's node
		cost, err := s.networkTopology.GetNetworkCost(nodeID, commNodeID)
		if err != nil {
			// If error, assume worst cost
			cost = 1.0
		}

		// Weight cost by bandwidth proportion
		bandwidthProportion := bandwidth / totalBandwidth
		weightedCosts += cost * bandwidthProportion
	}

	// Convert cost to score (lower cost = higher score)
	score := 1.0 - weightedCosts
	return score, true
}

// scoreBandwidthAvailability scores a node based on bandwidth availability
func (s *NetworkAwareScheduler) scoreBandwidthAvailability(nodeID string) (float64, bool) {
	// Collect all links connected to this node
	links := s.networkTopology.GetAllLinks()
	if len(links) == 0 {
		return 0.0, false
	}

	nodeLinks := make([]*network.NetworkLink, 0)
	for _, link := range links {
		if link.SourceID == nodeID || link.DestinationID == nodeID {
			nodeLinks = append(nodeLinks, link)
		}
	}

	if len(nodeLinks) == 0 {
		return 0.0, false
	}

	// Calculate average available bandwidth (1.0 - utilization) * bandwidth
	availableBandwidth := 0.0
	for _, link := range nodeLinks {
		availableBandwidth += (1.0 - link.Utilization) * link.Bandwidth
	}
	avgAvailableBandwidth := availableBandwidth / float64(len(nodeLinks))

	// Normalize score: higher available bandwidth is better
	// Assuming 10 Gbps as reference for normalization
	score := math.Min(avgAvailableBandwidth/10000.0, 1.0)
	return score, true
}

// scoreLatencyCharacteristics scores a node based on latency characteristics
func (s *NetworkAwareScheduler) scoreLatencyCharacteristics(vmID, nodeID string) (float64, bool) {
	s.vmCommunicationMutex.RLock()
	s.vmLocationCacheMutex.RLock()
	defer s.vmCommunicationMutex.RUnlock()
	defer s.vmLocationCacheMutex.RUnlock()

	// Find all VMs that have latency requirements with this VM
	latencyReqs := make(map[string]float64)

	for _, comm := range s.vmCommunications {
		// Skip if no latency requirement
		if comm.Latency <= 0 {
			continue
		}

		// Skip old entries
		if time.Since(comm.LastUpdated) > s.config.CommunicationTimeWindow {
			continue
		}

		if comm.SourceVMID == vmID {
			latencyReqs[comm.DestinationVMID] = comm.Latency
		} else if comm.DestinationVMID == vmID {
			latencyReqs[comm.SourceVMID] = comm.Latency
		}
	}

	if len(latencyReqs) == 0 {
		return 0.0, false
	}

	// Check if latency requirements can be met
	satisfiedReqs := 0
	totalReqs := len(latencyReqs)

	for commVM, reqLatency := range latencyReqs {
		// Get location of communicating VM
		commNodeID, exists := s.vmLocationCache[commVM]
		if !exists {
			continue
		}

		// Get link between nodes
		link, err := s.networkTopology.GetLink(nodeID, commNodeID)
		if err != nil {
			// If no direct link, check network cost as approximation
			cost, err := s.networkTopology.GetNetworkCost(nodeID, commNodeID)
			if err == nil && cost < 0.5 { // Lower cost generally means lower latency
				satisfiedReqs++
			}
			continue
		}

		// Check if latency requirement is satisfied
		if link.Latency <= reqLatency {
			satisfiedReqs++
		}
	}

	// Calculate score based on satisfied requirements
	return float64(satisfiedReqs) / float64(totalReqs), true
}

// scoreZoneProximity scores a node based on zone proximity for availability
func (s *NetworkAwareScheduler) scoreZoneProximity(vmID, nodeID string) float64 {
	// 1. Check if VM is part of an affinity group
	s.vmAffinityGroupMutex.RLock()
	inAffinityGroup := false
	affinityGroupVMs := make([]string, 0)

	for _, vms := range s.vmAffinityGroups {
		for _, id := range vms {
			if id == vmID {
				inAffinityGroup = true
				affinityGroupVMs = vms
				break
			}
		}
		if inAffinityGroup {
			break
		}
	}
	s.vmAffinityGroupMutex.RUnlock()

	if !inAffinityGroup {
		// If not in affinity group, use zone diversity for availability
		// Try to distribute VMs across zones for better availability
		zones, err := s.getZoneDistribution()
		if err != nil {
			return 0.5 // Neutral score on error
		}

		// Get zone of the node
		node, err := s.networkTopology.GetNode(nodeID)
		if err != nil {
			return 0.5 // Neutral score if can't get node
		}

		nodeZone := node.Location.Zone

		// Calculate proportion of VMs in this zone
		totalVMs := 0
		for _, count := range zones {
			totalVMs += count
		}

		if totalVMs == 0 {
			return 0.8 // Slightly prefer empty zones
		}

		zoneVMs := zones[nodeZone]
		zoneProportion := float64(zoneVMs) / float64(totalVMs)

		// Lower score for zones with more VMs (encourage distribution)
		return 1.0 - zoneProportion
	} else {
		// For affinity groups, check if other VMs in the group are in the same zone
		// Higher score if in the same zone as other VMs in the affinity group
		s.vmLocationCacheMutex.RLock()
		defer s.vmLocationCacheMutex.RUnlock()

		// Get zone of the node
		node, err := s.networkTopology.GetNode(nodeID)
		if err != nil {
			return 0.5 // Neutral score if can't get node
		}

		nodeZone := node.Location.Zone

		// Count VMs in the same zone
		sameZoneCount := 0
		totalVMsWithLocation := 0

		for _, groupVMID := range affinityGroupVMs {
			if groupVMID == vmID {
				continue // Skip this VM
			}

			vmNodeID, exists := s.vmLocationCache[groupVMID]
			if !exists {
				continue
			}

			totalVMsWithLocation++

			vmNode, err := s.networkTopology.GetNode(vmNodeID)
			if err != nil {
				continue
			}

			if vmNode.Location.Zone == nodeZone {
				sameZoneCount++
			}
		}

		if totalVMsWithLocation == 0 {
			return 0.5 // Neutral score if no other VMs have location
		}

		// Higher score if more VMs in the same zone (encourage affinity)
		return float64(sameZoneCount) / float64(totalVMsWithLocation)
	}
}

// getZoneDistribution returns the count of VMs in each zone
func (s *NetworkAwareScheduler) getZoneDistribution() (map[string]int, error) {
	zones := make(map[string]int)

	s.vmLocationCacheMutex.RLock()
	defer s.vmLocationCacheMutex.RUnlock()

	for _, nodeID := range s.vmLocationCache {
		node, err := s.networkTopology.GetNode(nodeID)
		if err != nil {
			continue
		}

		zones[node.Location.Zone]++
	}

	return zones, nil
}

// RequestPlacement overrides the base method to include network awareness
func (s *NetworkAwareScheduler) RequestPlacement(vmID string, policy PlacementPolicy, constraints []PlacementConstraint, resources map[string]float64, priority int) (string, error) {
	// Handle network-aware placement policy
	if policy == PolicyNetworkAware {
		// Add network-specific constraints if needed
		// ...

		// Ensure network awareness is given proper weight
		s.config.NetworkAwarenessWeight = 0.6 // Increase weight for network-aware policy
	}

	// Call base implementation
	return s.ResourceAwareScheduler.RequestPlacement(vmID, policy, constraints, resources, priority)
}
