package migration

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/scheduler/workload"
)

// CostType represents a type of migration cost
type CostType string

// Cost types
const (
	CostTypeDowntime      CostType = "downtime"
	CostTypeBandwidth     CostType = "bandwidth"
	CostTypePerformance   CostType = "performance"
	CostTypeResourceWaste CostType = "resource_waste"
	CostTypeTotal         CostType = "total"
)

// MigrationCost represents the estimated cost of a migration
type MigrationCost struct {
	// SourceNodeID is the ID of the source node
	SourceNodeID string

	// DestNodeID is the ID of the destination node
	DestNodeID string

	// VMID is the ID of the VM being migrated
	VMID string

	// EstimatedDowntimeMs is the estimated downtime in milliseconds
	EstimatedDowntimeMs int64

	// EstimatedBandwidthMB is the estimated bandwidth usage in MB
	EstimatedBandwidthMB float64

	// EstimatedDurationMs is the estimated total migration duration in milliseconds
	EstimatedDurationMs int64

	// PerformanceImpact is the estimated performance impact (0-1)
	// 0 means no impact, 1 means maximum impact
	PerformanceImpact float64

	// ResourceEfficiency is how efficiently resources are used (0-1)
	// 1 means perfect efficiency, 0 means complete waste
	ResourceEfficiency float64

	// TotalCostScore is the combined cost score (lower is better)
	TotalCostScore float64

	// ScoreComponents maps cost types to their individual scores
	ScoreComponents map[CostType]float64

	// EstimatedAt is when this cost was estimated
	EstimatedAt time.Time

	// Confidence is how confident we are in this estimate (0-1)
	Confidence float64
}

// NetworkLink represents a network connection between two nodes
type NetworkLink struct {
	// SourceNodeID is the ID of the source node
	SourceNodeID string

	// DestNodeID is the ID of the destination node
	DestNodeID string

	// BandwidthMbps is the available bandwidth in Mbps
	BandwidthMbps float64

	// LatencyMs is the latency in milliseconds
	LatencyMs float64

	// PacketLossRate is the packet loss rate (0-1)
	PacketLossRate float64

	// IsWAN indicates if this is a WAN link
	IsWAN bool

	// MeasuredAt is when these metrics were measured
	MeasuredAt time.Time
}

// NodeInfo represents information about a node
type NodeInfo struct {
	// NodeID is the unique identifier for the node
	NodeID string

	// Available indicates if the node is available
	Available bool

	// Resources maps resource types to their capacities and usages
	Resources map[string]*NodeResource

	// LastUpdated is when this information was last updated
	LastUpdated time.Time
}

// NodeResource represents a resource on a node
type NodeResource struct {
	// Type is the type of resource
	Type string

	// Capacity is the total capacity
	Capacity float64

	// Used is the amount currently in use
	Used float64
}

// VMInfo represents information about a VM
type VMInfo struct {
	// VMID is the unique identifier for the VM
	VMID string

	// CurrentNodeID is the ID of the node where the VM is currently running
	CurrentNodeID string

	// SizeMB is the size of the VM in MB
	SizeMB int64

	// MemoryMB is the amount of memory allocated to the VM in MB
	MemoryMB int64

	// CPUCount is the number of vCPUs allocated to the VM
	CPUCount int

	// DiskSizeMB is the size of the VM's disk in MB
	DiskSizeMB int64

	// WorkloadProfile is the profile of the VM's workload
	WorkloadProfile *workload.WorkloadProfile

	// DirtyPageRate is the rate at which memory pages are modified (pages/sec)
	// Used for live migration cost estimation
	DirtyPageRate float64

	// LastUpdated is when this information was last updated
	LastUpdated time.Time
}

// MigrationCostEstimatorConfig contains configuration for the migration cost estimator
type MigrationCostEstimatorConfig struct {
	// DefaultWANBandwidthMbps is the default bandwidth for WAN links
	DefaultWANBandwidthMbps float64

	// DefaultLANBandwidthMbps is the default bandwidth for LAN links
	DefaultLANBandwidthMbps float64

	// DefaultWANLatencyMs is the default latency for WAN links
	DefaultWANLatencyMs float64

	// DefaultLANLatencyMs is the default latency for LAN links
	DefaultLANLatencyMs float64

	// CostWeights maps cost types to their weights in the total cost calculation
	CostWeights map[CostType]float64

	// MaxAcceptableDowntimeMs is the maximum acceptable downtime in milliseconds
	MaxAcceptableDowntimeMs int64

	// NetworkProbingInterval is how often to probe network links
	NetworkProbingInterval time.Duration

	// CostCacheTTL is how long to cache cost estimates
	CostCacheTTL time.Duration

	// MaxConcurrentEstimations is the maximum number of concurrent estimations
	MaxConcurrentEstimations int
}

// DefaultMigrationCostEstimatorConfig returns a default configuration
func DefaultMigrationCostEstimatorConfig() MigrationCostEstimatorConfig {
	return MigrationCostEstimatorConfig{
		DefaultWANBandwidthMbps: 100,
		DefaultLANBandwidthMbps: 1000,
		DefaultWANLatencyMs:     50,
		DefaultLANLatencyMs:     1,
		CostWeights: map[CostType]float64{
			CostTypeDowntime:      0.4,
			CostTypeBandwidth:     0.2,
			CostTypePerformance:   0.2,
			CostTypeResourceWaste: 0.2,
		},
		MaxAcceptableDowntimeMs:  5000,
		NetworkProbingInterval:   10 * time.Minute,
		CostCacheTTL:             5 * time.Minute,
		MaxConcurrentEstimations: 10,
	}
}

// MigrationCostEstimator estimates the cost of migrating VMs
type MigrationCostEstimator struct {
	config MigrationCostEstimatorConfig

	// networkLinks maps source+dest node IDs to network link information
	networkLinks     map[string]*NetworkLink
	networkLinkMutex sync.RWMutex

	// nodes maps node IDs to node information
	nodes     map[string]*NodeInfo
	nodeMutex sync.RWMutex

	// vms maps VM IDs to VM information
	vms     map[string]*VMInfo
	vmMutex sync.RWMutex

	// cachedCosts caches migration cost estimates
	cachedCosts     map[string]*MigrationCost
	cachedCostMutex sync.RWMutex

	// workloadAnalyzer is used to get VM workload profiles
	workloadAnalyzer *workload.WorkloadAnalyzer

	// estimationSemaphore limits concurrent estimations
	estimationSemaphore chan struct{}

	ctx    context.Context
	cancel context.CancelFunc
}

// NewMigrationCostEstimator creates a new migration cost estimator
func NewMigrationCostEstimator(config MigrationCostEstimatorConfig, analyzer *workload.WorkloadAnalyzer) *MigrationCostEstimator {
	ctx, cancel := context.WithCancel(context.Background())

	return &MigrationCostEstimator{
		config:              config,
		networkLinks:        make(map[string]*NetworkLink),
		nodes:               make(map[string]*NodeInfo),
		vms:                 make(map[string]*VMInfo),
		cachedCosts:         make(map[string]*MigrationCost),
		workloadAnalyzer:    analyzer,
		estimationSemaphore: make(chan struct{}, config.MaxConcurrentEstimations),
		ctx:                 ctx,
		cancel:              cancel,
	}
}

// Start starts the migration cost estimator
func (e *MigrationCostEstimator) Start() error {
	log.Println("Starting migration cost estimator")

	// Start the network probing loop
	go e.networkProbingLoop()

	// Start the cache cleanup loop
	go e.cacheCleanupLoop()

	return nil
}

// Stop stops the migration cost estimator
func (e *MigrationCostEstimator) Stop() error {
	log.Println("Stopping migration cost estimator")

	e.cancel()

	return nil
}

// networkProbingLoop periodically probes network links
func (e *MigrationCostEstimator) networkProbingLoop() {
	ticker := time.NewTicker(e.config.NetworkProbingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			e.probeNetworkLinks()
		}
	}
}

// cacheCleanupLoop periodically cleans up expired cache entries
func (e *MigrationCostEstimator) cacheCleanupLoop() {
	ticker := time.NewTicker(e.config.CostCacheTTL)
	defer ticker.Stop()

	for {
		select {
		case <-e.ctx.Done():
			return
		case <-ticker.C:
			e.cleanupCache()
		}
	}
}

// probeNetworkLinks probes network links between nodes
func (e *MigrationCostEstimator) probeNetworkLinks() {
	// In a real implementation, this would perform actual network probing
	// For now, we'll just update the timestamps

	e.networkLinkMutex.Lock()
	defer e.networkLinkMutex.Unlock()

	now := time.Now()
	for _, link := range e.networkLinks {
		link.MeasuredAt = now
	}
}

// cleanupCache removes expired cache entries
func (e *MigrationCostEstimator) cleanupCache() {
	e.cachedCostMutex.Lock()
	defer e.cachedCostMutex.Unlock()

	cutoff := time.Now().Add(-e.config.CostCacheTTL)
	for key, cost := range e.cachedCosts {
		if cost.EstimatedAt.Before(cutoff) {
			delete(e.cachedCosts, key)
		}
	}
}

// UpdateNetworkLink updates information about a network link
func (e *MigrationCostEstimator) UpdateNetworkLink(link *NetworkLink) {
	e.networkLinkMutex.Lock()
	defer e.networkLinkMutex.Unlock()

	key := fmt.Sprintf("%s|%s", link.SourceNodeID, link.DestNodeID)
	e.networkLinks[key] = link
}

// UpdateNodeInfo updates information about a node
func (e *MigrationCostEstimator) UpdateNodeInfo(node *NodeInfo) {
	e.nodeMutex.Lock()
	defer e.nodeMutex.Unlock()

	e.nodes[node.NodeID] = node
}

// UpdateVMInfo updates information about a VM
func (e *MigrationCostEstimator) UpdateVMInfo(vm *VMInfo) {
	e.vmMutex.Lock()
	defer e.vmMutex.Unlock()

	e.vms[vm.VMID] = vm

	// Clear any cached costs for this VM
	e.cachedCostMutex.Lock()
	defer e.cachedCostMutex.Unlock()

	for key := range e.cachedCosts {
		if key == vm.VMID || key == fmt.Sprintf("%s|", vm.VMID) {
			delete(e.cachedCosts, key)
		}
	}
}

// EstimateMigrationCost estimates the cost of migrating a VM from its current node to a destination node
func (e *MigrationCostEstimator) EstimateMigrationCost(ctx context.Context, vmID string, destNodeID string) (*MigrationCost, error) {
	// Check for a cached estimate
	cacheKey := fmt.Sprintf("%s|%s", vmID, destNodeID)

	e.cachedCostMutex.RLock()
	cachedCost, exists := e.cachedCosts[cacheKey]
	e.cachedCostMutex.RUnlock()

	if exists && time.Since(cachedCost.EstimatedAt) < e.config.CostCacheTTL {
		return cachedCost, nil
	}

	// Acquire a semaphore slot for this estimation
	select {
	case e.estimationSemaphore <- struct{}{}:
		// Got a slot
		defer func() {
			<-e.estimationSemaphore
		}()
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Get VM info
	e.vmMutex.RLock()
	vm, exists := e.vms[vmID]
	e.vmMutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no information for VM %s", vmID)
	}

	// Get source node info
	e.nodeMutex.RLock()
	sourceNode, exists := e.nodes[vm.CurrentNodeID]
	e.nodeMutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no information for source node %s", vm.CurrentNodeID)
	}

	// Get destination node info
	e.nodeMutex.RLock()
	destNode, exists := e.nodes[destNodeID]
	e.nodeMutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no information for destination node %s", destNodeID)
	}

	// Get network link info
	e.networkLinkMutex.RLock()
	key := fmt.Sprintf("%s|%s", vm.CurrentNodeID, destNodeID)
	link, exists := e.networkLinks[key]
	e.networkLinkMutex.RUnlock()

	if !exists {
		// Create a default link based on whether this is WAN or LAN
		isWAN := false // In a real implementation, determine this based on node locations
		bandwidth := e.config.DefaultLANBandwidthMbps
		latency := e.config.DefaultLANLatencyMs
		if isWAN {
			bandwidth = e.config.DefaultWANBandwidthMbps
			latency = e.config.DefaultWANLatencyMs
		}

		link = &NetworkLink{
			SourceNodeID:   vm.CurrentNodeID,
			DestNodeID:     destNodeID,
			BandwidthMbps:  bandwidth,
			LatencyMs:      latency,
			PacketLossRate: 0,
			IsWAN:          isWAN,
			MeasuredAt:     time.Now(),
		}
	}

	// Calculate migration cost
	cost, err := e.calculateMigrationCost(vm, sourceNode, destNode, link)
	if err != nil {
		return nil, err
	}

	// Cache the result
	e.cachedCostMutex.Lock()
	e.cachedCosts[cacheKey] = cost
	e.cachedCostMutex.Unlock()

	return cost, nil
}

// calculateMigrationCost calculates the cost of migrating a VM
func (e *MigrationCostEstimator) calculateMigrationCost(
	vm *VMInfo,
	sourceNode *NodeInfo,
	destNode *NodeInfo,
	link *NetworkLink,
) (*MigrationCost, error) {
	// This is a simplified cost model
	// In a real implementation, this would be much more sophisticated

	// Calculate bandwidth usage (simplified model)
	// For live migration, we need to consider dirty page rate
	bandwidthMB := float64(vm.SizeMB)
	if vm.WorkloadProfile != nil && vm.WorkloadProfile.DominantWorkloadType == workload.WorkloadTypeMemoryIntensive {
		// Memory-intensive workloads might have higher dirty page rates
		bandwidthMB *= 1.5
	}

	// Calculate bandwidth-based migration time
	// Convert Mbps to MBps (divide by 8)
	bandwidthMBps := link.BandwidthMbps / 8
	transferTimeMs := int64(bandwidthMB / bandwidthMBps * 1000)

	// Calculate downtime based on workload and network
	// This is a very simplified model
	var downtimeMs int64
	if link.IsWAN {
		// WAN migrations typically have longer downtime
		downtimeMs = 1000 + int64(link.LatencyMs*10)
	} else {
		// LAN migrations have shorter downtime
		downtimeMs = 200 + int64(link.LatencyMs*5)
	}

	// Adjust for workload type
	if vm.WorkloadProfile != nil {
		switch vm.WorkloadProfile.DominantWorkloadType {
		case workload.WorkloadTypeCPUIntensive:
			// CPU-intensive workloads might have less state to transfer
			downtimeMs = int64(float64(downtimeMs) * 0.8)
		case workload.WorkloadTypeMemoryIntensive:
			// Memory-intensive workloads might have more state to transfer
			downtimeMs = int64(float64(downtimeMs) * 1.5)
		case workload.WorkloadTypeIOIntensive:
			// I/O-intensive workloads might have more disk state
			downtimeMs = int64(float64(downtimeMs) * 1.2)
		}
	}

	// Calculate performance impact
	// Performance impact depends on workload type and resource availability
	performanceImpact := 0.2 // Base impact

	// Adjust based on workload type
	if vm.WorkloadProfile != nil {
		switch vm.WorkloadProfile.DominantWorkloadType {
		case workload.WorkloadTypeCPUIntensive:
			// CPU workloads are more sensitive to CPU availability
			cpuAvailability := getResourceAvailability(destNode, "cpu")
			performanceImpact *= (1.5 - cpuAvailability)
		case workload.WorkloadTypeMemoryIntensive:
			// Memory workloads are more sensitive to memory availability
			memoryAvailability := getResourceAvailability(destNode, "memory")
			performanceImpact *= (1.5 - memoryAvailability)
		case workload.WorkloadTypeIOIntensive:
			// I/O workloads are more sensitive to disk availability
			diskAvailability := getResourceAvailability(destNode, "disk")
			performanceImpact *= (1.5 - diskAvailability)
		case workload.WorkloadTypeNetworkIntensive:
			// Network workloads are more sensitive to network availability
			networkAvailability := getResourceAvailability(destNode, "network")
			performanceImpact *= (1.5 - networkAvailability)
		}
	}

	// Calculate resource efficiency
	// This measures how efficiently resources are used after migration
	resourceEfficiency := calculateResourceEfficiency(vm, destNode)

	// Calculate individual cost scores (lower is better)
	downtimeScore := float64(downtimeMs) / float64(e.config.MaxAcceptableDowntimeMs)
	if downtimeScore > 1.0 {
		downtimeScore = 1.0
	}

	// Bandwidth score (higher bandwidth usage = higher cost)
	bandwidthScore := bandwidthMB / 10000.0 // Normalize to 0-1 range
	if bandwidthScore > 1.0 {
		bandwidthScore = 1.0
	}

	// Performance score (higher impact = higher cost)
	performanceScore := performanceImpact

	// Resource waste score (lower efficiency = higher waste)
	resourceWasteScore := 1.0 - resourceEfficiency

	// Calculate total cost score (weighted sum)
	totalScore := 0.0
	scoreComponents := make(map[CostType]float64)

	scoreComponents[CostTypeDowntime] = downtimeScore
	scoreComponents[CostTypeBandwidth] = bandwidthScore
	scoreComponents[CostTypePerformance] = performanceScore
	scoreComponents[CostTypeResourceWaste] = resourceWasteScore

	for costType, score := range scoreComponents {
		weight := e.config.CostWeights[costType]
		totalScore += score * weight
	}

	// Create the cost object
	cost := &MigrationCost{
		SourceNodeID:         vm.CurrentNodeID,
		DestNodeID:           destNode.NodeID,
		VMID:                 vm.VMID,
		EstimatedDowntimeMs:  downtimeMs,
		EstimatedBandwidthMB: bandwidthMB,
		EstimatedDurationMs:  transferTimeMs,
		PerformanceImpact:    performanceImpact,
		ResourceEfficiency:   resourceEfficiency,
		TotalCostScore:       totalScore,
		ScoreComponents:      scoreComponents,
		EstimatedAt:          time.Now(),
		Confidence:           0.7, // Fixed confidence for now
	}

	return cost, nil
}

// getResourceAvailability calculates the availability of a resource on a node
func getResourceAvailability(node *NodeInfo, resourceType string) float64 {
	if node == nil || node.Resources == nil {
		return 0.5 // Default if no information is available
	}

	resource, exists := node.Resources[resourceType]
	if !exists {
		return 0.5 // Default if no information is available
	}

	if resource.Capacity <= 0 {
		return 0 // No capacity
	}

	return 1.0 - (resource.Used / resource.Capacity)
}

// calculateResourceEfficiency calculates how efficiently resources are used
func calculateResourceEfficiency(vm *VMInfo, node *NodeInfo) float64 {
	if vm == nil || node == nil || node.Resources == nil {
		return 0.5 // Default if no information is available
	}

	// Calculate efficiency based on how well the VM's requirements match the node's available resources
	// This is a simplified model

	var cpuEfficiency, memoryEfficiency, diskEfficiency float64

	// CPU efficiency
	cpuResource, exists := node.Resources["cpu"]
	if exists && cpuResource.Capacity > 0 {
		cpuAvailable := cpuResource.Capacity - cpuResource.Used
		cpuNeeded := float64(vm.CPUCount)
		if cpuNeeded <= cpuAvailable {
			// VM fits, calculate how efficiently it uses the resource
			cpuEfficiency = math.Min(1.0, cpuNeeded/cpuAvailable)
		} else {
			// VM doesn't fit, efficiency is poor
			cpuEfficiency = 0.1
		}
	} else {
		cpuEfficiency = 0.5 // Default
	}

	// Memory efficiency
	memoryResource, exists := node.Resources["memory"]
	if exists && memoryResource.Capacity > 0 {
		memoryAvailable := memoryResource.Capacity - memoryResource.Used
		memoryNeeded := float64(vm.MemoryMB)
		if memoryNeeded <= memoryAvailable {
			// VM fits, calculate how efficiently it uses the resource
			memoryEfficiency = math.Min(1.0, memoryNeeded/memoryAvailable)
		} else {
			// VM doesn't fit, efficiency is poor
			memoryEfficiency = 0.1
		}
	} else {
		memoryEfficiency = 0.5 // Default
	}

	// Disk efficiency
	diskResource, exists := node.Resources["disk"]
	if exists && diskResource.Capacity > 0 {
		diskAvailable := diskResource.Capacity - diskResource.Used
		diskNeeded := float64(vm.DiskSizeMB)
		if diskNeeded <= diskAvailable {
			// VM fits, calculate how efficiently it uses the resource
			diskEfficiency = math.Min(1.0, diskNeeded/diskAvailable)
		} else {
			// VM doesn't fit, efficiency is poor
			diskEfficiency = 0.1
		}
	} else {
		diskEfficiency = 0.5 // Default
	}

	// Weight different resources based on the VM's workload profile
	var cpuWeight, memoryWeight, diskWeight float64
	if vm.WorkloadProfile != nil {
		switch vm.WorkloadProfile.DominantWorkloadType {
		case workload.WorkloadTypeCPUIntensive:
			cpuWeight = 0.6
			memoryWeight = 0.2
			diskWeight = 0.2
		case workload.WorkloadTypeMemoryIntensive:
			cpuWeight = 0.2
			memoryWeight = 0.6
			diskWeight = 0.2
		case workload.WorkloadTypeIOIntensive:
			cpuWeight = 0.2
			memoryWeight = 0.2
			diskWeight = 0.6
		default: // Balanced
			cpuWeight = 0.33
			memoryWeight = 0.33
			diskWeight = 0.34
		}
	} else {
		// Default weights
		cpuWeight = 0.33
		memoryWeight = 0.33
		diskWeight = 0.34
	}

	// Calculate weighted efficiency
	totalEfficiency := cpuEfficiency*cpuWeight + memoryEfficiency*memoryWeight + diskEfficiency*diskWeight

	return totalEfficiency
}

// FindBestMigrationTarget finds the best node to migrate a VM to
func (e *MigrationCostEstimator) FindBestMigrationTarget(ctx context.Context, vmID string, candidateNodeIDs []string) (string, error) {
	if len(candidateNodeIDs) == 0 {
		return "", fmt.Errorf("no candidate nodes provided")
	}

	type nodeCost struct {
		nodeID string
		cost   *MigrationCost
	}

	// Estimate cost for each candidate node
	costs := make([]nodeCost, 0, len(candidateNodeIDs))

	for _, nodeID := range candidateNodeIDs {
		cost, err := e.EstimateMigrationCost(ctx, vmID, nodeID)
		if err != nil {
			log.Printf("Error estimating migration cost to node %s: %v", nodeID, err)
			continue
		}

		costs = append(costs, nodeCost{
			nodeID: nodeID,
			cost:   cost,
		})
	}

	if len(costs) == 0 {
		return "", fmt.Errorf("could not estimate costs for any candidate nodes")
	}

	// Find the node with the lowest cost
	bestNode := costs[0].nodeID
	bestCost := costs[0].cost.TotalCostScore

	for i := 1; i < len(costs); i++ {
		if costs[i].cost.TotalCostScore < bestCost {
			bestNode = costs[i].nodeID
			bestCost = costs[i].cost.TotalCostScore
		}
	}

	return bestNode, nil
}

// EstimateBulkMigrationCost estimates the cost of migrating multiple VMs
func (e *MigrationCostEstimator) EstimateBulkMigrationCost(ctx context.Context, vmIDs []string, destNodeID string) (*MigrationCost, error) {
	if len(vmIDs) == 0 {
		return nil, fmt.Errorf("no VMs provided")
	}

	// Estimate individual costs
	var totalDowntimeMs int64
	var totalBandwidthMB float64
	var totalDurationMs int64
	var totalPerformanceImpact float64
	var totalResourceEfficiency float64

	successfulEstimates := 0
	for _, vmID := range vmIDs {
		cost, err := e.EstimateMigrationCost(ctx, vmID, destNodeID)
		if err != nil {
			log.Printf("Error estimating migration cost for VM %s: %v", vmID, err)
			continue
		}

		// Accumulate metrics
		totalDowntimeMs += cost.EstimatedDowntimeMs
		totalBandwidthMB += cost.EstimatedBandwidthMB
		totalDurationMs += cost.EstimatedDurationMs
		totalPerformanceImpact += cost.PerformanceImpact
		totalResourceEfficiency += cost.ResourceEfficiency

		successfulEstimates++
	}

	if successfulEstimates == 0 {
		return nil, fmt.Errorf("could not estimate costs for any VMs")
	}

	// Calculate averages
	avgDowntimeMs := totalDowntimeMs / int64(successfulEstimates)
	avgBandwidthMB := totalBandwidthMB / float64(successfulEstimates)
	avgDurationMs := totalDurationMs / int64(successfulEstimates)
	avgPerformanceImpact := totalPerformanceImpact / float64(successfulEstimates)
	avgResourceEfficiency := totalResourceEfficiency / float64(successfulEstimates)

	// Calculate score components
	downtimeScore := float64(avgDowntimeMs) / float64(e.config.MaxAcceptableDowntimeMs)
	if downtimeScore > 1.0 {
		downtimeScore = 1.0
	}

	bandwidthScore := avgBandwidthMB / 10000.0
	if bandwidthScore > 1.0 {
		bandwidthScore = 1.0
	}

	performanceScore := avgPerformanceImpact
	resourceWasteScore := 1.0 - avgResourceEfficiency

	scoreComponents := map[CostType]float64{
		CostTypeDowntime:      downtimeScore,
		CostTypeBandwidth:     bandwidthScore,
		CostTypePerformance:   performanceScore,
		CostTypeResourceWaste: resourceWasteScore,
	}

	// Calculate total score
	totalScore := 0.0
	for costType, score := range scoreComponents {
		weight := e.config.CostWeights[costType]
		totalScore += score * weight
	}

	// Create bulk cost object
	cost := &MigrationCost{
		SourceNodeID:         "multiple",
		DestNodeID:           destNodeID,
		VMID:                 "bulk",
		EstimatedDowntimeMs:  avgDowntimeMs,
		EstimatedBandwidthMB: totalBandwidthMB,
		EstimatedDurationMs:  avgDurationMs,
		PerformanceImpact:    avgPerformanceImpact,
		ResourceEfficiency:   avgResourceEfficiency,
		TotalCostScore:       totalScore,
		ScoreComponents:      scoreComponents,
		EstimatedAt:          time.Now(),
		Confidence:           0.6, // Lower confidence for bulk estimates
	}

	return cost, nil
}

// GetMigrationRecommendations gets recommendations for VM migrations
func (e *MigrationCostEstimator) GetMigrationRecommendations(ctx context.Context, vmID string) ([]MigrationRecommendation, error) {
	// Get VM info
	e.vmMutex.RLock()
	vm, exists := e.vms[vmID]
	e.vmMutex.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no information for VM %s", vmID)
	}

	// Get all available nodes
	e.nodeMutex.RLock()
	candidateNodes := make([]string, 0, len(e.nodes))
	for nodeID, node := range e.nodes {
		if node.Available && nodeID != vm.CurrentNodeID {
			candidateNodes = append(candidateNodes, nodeID)
		}
	}
	e.nodeMutex.RUnlock()

	if len(candidateNodes) == 0 {
		return nil, fmt.Errorf("no available destination nodes for VM %s", vmID)
	}

	// Get cost estimates for each node
	recommendations := make([]MigrationRecommendation, 0, len(candidateNodes))
	for _, nodeID := range candidateNodes {
		cost, err := e.EstimateMigrationCost(ctx, vmID, nodeID)
		if err != nil {
			log.Printf("Error estimating migration cost to node %s: %v", nodeID, err)
			continue
		}

		recommendations = append(recommendations, MigrationRecommendation{
			DestNodeID:   nodeID,
			Cost:         cost,
			Reasoning:    generateReasoningForRecommendation(cost),
			Alternatives: nil, // Will be populated later
		})
	}

	if len(recommendations) == 0 {
		return nil, fmt.Errorf("could not generate any recommendations for VM %s", vmID)
	}

	// Sort recommendations by cost (lowest first)
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].Cost.TotalCostScore < recommendations[j].Cost.TotalCostScore
	})

	// For the top recommendations, find alternatives
	for i := 0; i < min(3, len(recommendations)); i++ {
		recommendations[i].Alternatives = findAlternativeNodes(recommendations, i)
	}

	return recommendations, nil
}

// MigrationRecommendation represents a recommendation for VM migration
type MigrationRecommendation struct {
	// DestNodeID is the recommended destination node
	DestNodeID string

	// Cost is the estimated cost of the migration
	Cost *MigrationCost

	// Reasoning is a human-readable explanation of the recommendation
	Reasoning string

	// Alternatives are alternative nodes that could be used
	Alternatives []string
}

// findAlternativeNodes finds alternative nodes to the recommendation at the given index
func findAlternativeNodes(recommendations []MigrationRecommendation, index int) []string {
	if index >= len(recommendations) {
		return nil
	}

	// Get the recommendation's cost score
	baseCost := recommendations[index].Cost.TotalCostScore

	// Find nodes with similar costs (within 20%)
	alternatives := make([]string, 0)
	for i, rec := range recommendations {
		if i != index {
			costDiff := rec.Cost.TotalCostScore - baseCost
			if costDiff > 0 && costDiff/baseCost <= 0.2 {
				alternatives = append(alternatives, rec.DestNodeID)
			}
		}
	}

	return alternatives
}

// generateReasoningForRecommendation generates a human-readable explanation for a recommendation
func generateReasoningForRecommendation(cost *MigrationCost) string {
	// Find the dominant factor in the cost
	var dominantFactor CostType
	var dominantScore float64

	for costType, score := range cost.ScoreComponents {
		if score > dominantScore {
			dominantFactor = costType
			dominantScore = score
		}
	}

	// Generate reasoning based on the dominant factor
	switch dominantFactor {
	case CostTypeDowntime:
		return fmt.Sprintf("This node provides the lowest estimated downtime of %d ms", cost.EstimatedDowntimeMs)
	case CostTypeBandwidth:
		return fmt.Sprintf("This node requires the least bandwidth usage (%.1f MB)", cost.EstimatedBandwidthMB)
	case CostTypePerformance:
		return fmt.Sprintf("This node minimizes performance impact (%.1f%%)", cost.PerformanceImpact*100)
	case CostTypeResourceWaste:
		return fmt.Sprintf("This node provides optimal resource utilization efficiency (%.1f%%)", cost.ResourceEfficiency*100)
	default:
		return fmt.Sprintf("This node provides the best overall migration experience with a cost score of %.2f", cost.TotalCostScore)
	}
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
