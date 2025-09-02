package federation

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	resourcePoolCapacity = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "novacron_resource_pool_capacity",
		Help: "Total capacity in the resource pool",
	}, []string{"resource_type"})

	resourcePoolUsage = promauto.NewGaugeVec(prometheus.GaugeOpts{
		Name: "novacron_resource_pool_usage",
		Help: "Current usage in the resource pool",
	}, []string{"resource_type"})

	resourceAllocations = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_resource_allocations_total",
		Help: "Total number of resource allocations",
	}, []string{"status"})
)

// ResourcePool manages resource allocation across the federation
type ResourcePool struct {
	config           *FederationConfig
	nodeInventories  map[string]*ResourceInventory
	allocations      map[string]*ResourceAllocation
	pendingRequests  []*ResourceRequest
	inventoryMu      sync.RWMutex
	allocationsMu    sync.RWMutex
	requestsMu       sync.RWMutex
	scheduler        *ResourceScheduler
	optimizer        *ResourceOptimizer
	logger           Logger
	stopCh           chan struct{}
	isRunning        bool
}

// NewResourcePool creates a new resource pool
func NewResourcePool(config *FederationConfig, logger Logger) (*ResourcePool, error) {
	rp := &ResourcePool{
		config:          config,
		nodeInventories: make(map[string]*ResourceInventory),
		allocations:     make(map[string]*ResourceAllocation),
		pendingRequests: make([]*ResourceRequest, 0),
		logger:          logger,
		stopCh:          make(chan struct{}),
	}

	// Initialize scheduler
	rp.scheduler = NewResourceScheduler(config, logger)

	// Initialize optimizer
	rp.optimizer = NewResourceOptimizer(config, logger)

	return rp, nil
}

// AllocateResources allocates resources for a request
func (rp *ResourcePool) AllocateResources(ctx context.Context, request *ResourceRequest) (*ResourceAllocation, error) {
	if request == nil {
		return nil, fmt.Errorf("request is nil")
	}

	rp.logger.Info("Processing resource allocation request",
		"request_id", request.ID,
		"cpu", request.CPUCores,
		"memory", request.MemoryGB,
		"storage", request.StorageGB,
	)

	// Find suitable node(s)
	nodes := rp.findSuitableNodes(request)
	if len(nodes) == 0 {
		// Queue request if no resources available
		rp.queueRequest(request)
		resourceAllocations.WithLabelValues("queued").Inc()
		return nil, fmt.Errorf("no suitable nodes available, request queued")
	}

	// Select best node using scheduling algorithm
	selectedNode := rp.scheduler.SelectNode(nodes, request)
	if selectedNode == "" {
		rp.queueRequest(request)
		resourceAllocations.WithLabelValues("failed").Inc()
		return nil, fmt.Errorf("scheduling failed, request queued")
	}

	// Create allocation
	allocation := &ResourceAllocation{
		ID:            generateRequestID(),
		RequestID:     request.ID,
		NodeID:        selectedNode,
		CPU:          int(request.CPUCores),
		MemoryGB:     int(request.MemoryGB),
		StorageGB:    int(request.StorageGB),
		Status:       "active",
		AllocatedAt:  time.Now(),
		ExpiresAt:     time.Now().Add(request.Duration),
		Metadata:      make(map[string]interface{}),
	}

	// Update inventory
	if err := rp.updateNodeInventory(selectedNode, request, true); err != nil {
		resourceAllocations.WithLabelValues("failed").Inc()
		return nil, fmt.Errorf("failed to update inventory: %w", err)
	}

	// Store allocation
	rp.allocationsMu.Lock()
	rp.allocations[allocation.ID] = allocation
	rp.allocationsMu.Unlock()

	// Update metrics
	resourceAllocations.WithLabelValues("success").Inc()
	rp.updateMetrics()

	rp.logger.Info("Resources allocated successfully",
		"allocation_id", allocation.ID,
		"node_id", selectedNode,
	)

	return allocation, nil
}

// ReleaseResources releases allocated resources
func (rp *ResourcePool) ReleaseResources(ctx context.Context, allocationID string) error {
	rp.allocationsMu.Lock()
	allocation, exists := rp.allocations[allocationID]
	if !exists {
		rp.allocationsMu.Unlock()
		return fmt.Errorf("allocation not found: %s", allocationID)
	}

	// Mark as released
	allocation.Status = "released"
	delete(rp.allocations, allocationID)
	rp.allocationsMu.Unlock()

	// Create a request to represent the resources being freed
	request := &ResourceRequest{
		CPUCores:  float64(allocation.CPU),
		MemoryGB:  float64(allocation.MemoryGB),
		StorageGB: float64(allocation.StorageGB),
	}

	// Update inventory
	if err := rp.updateNodeInventory(allocation.NodeID, request, false); err != nil {
		rp.logger.Error("Failed to update inventory on release", "error", err)
	}

	// Process pending requests
	go rp.processPendingRequests(ctx)

	// Update metrics
	rp.updateMetrics()

	rp.logger.Info("Resources released", "allocation_id", allocationID)

	return nil
}

// GetAllocations returns all allocations
func (rp *ResourcePool) GetAllocations(ctx context.Context) ([]*ResourceAllocation, error) {
	rp.allocationsMu.RLock()
	defer rp.allocationsMu.RUnlock()

	allocations := make([]*ResourceAllocation, 0, len(rp.allocations))
	for _, allocation := range rp.allocations {
		allocations = append(allocations, allocation)
	}

	return allocations, nil
}

// GetAvailableResources returns available resources across all nodes
func (rp *ResourcePool) GetAvailableResources(ctx context.Context) (*ResourceInventory, error) {
	rp.inventoryMu.RLock()
	defer rp.inventoryMu.RUnlock()

	total := &ResourceInventory{}

	for _, inventory := range rp.nodeInventories {
		total.TotalCPU += inventory.TotalCPU
		total.UsedCPU += inventory.UsedCPU
		total.TotalMemory += inventory.TotalMemory
		total.UsedMemory += inventory.UsedMemory
		total.TotalStorage += inventory.TotalStorage
		total.UsedStorage += inventory.UsedStorage
		total.VMs += inventory.VMs
		total.Containers += inventory.Containers
		total.NetworkPools += inventory.NetworkPools
	}

	return total, nil
}

// UpdateResourceInventory updates the resource inventory for a node
func (rp *ResourcePool) UpdateResourceInventory(ctx context.Context, nodeID string, inventory *ResourceInventory) error {
	if inventory == nil {
		return fmt.Errorf("inventory is nil")
	}

	rp.inventoryMu.Lock()
	rp.nodeInventories[nodeID] = inventory
	rp.inventoryMu.Unlock()

	// Update metrics
	rp.updateMetrics()

	// Process pending requests if resources became available
	go rp.processPendingRequests(ctx)

	return nil
}

// Internal methods

func (rp *ResourcePool) findSuitableNodes(request *ResourceRequest) []string {
	rp.inventoryMu.RLock()
	defer rp.inventoryMu.RUnlock()

	var suitableNodes []string

	for nodeID, inventory := range rp.nodeInventories {
		// Check if node has enough resources
		availableCPU := inventory.TotalCPU - inventory.UsedCPU
		availableMemory := float64(inventory.TotalMemory-inventory.UsedMemory) / (1024 * 1024 * 1024)
		availableStorage := float64(inventory.TotalStorage-inventory.UsedStorage) / (1024 * 1024 * 1024)

		if availableCPU >= request.CPUCores &&
			availableMemory >= request.MemoryGB &&
			availableStorage >= request.StorageGB {

			// Check constraints
			if rp.checkConstraints(nodeID, request.Constraints) {
				suitableNodes = append(suitableNodes, nodeID)
			}
		}
	}

	return suitableNodes
}

func (rp *ResourcePool) checkConstraints(nodeID string, constraints map[string]string) bool {
	if constraints == nil {
		return true
	}

	// In production, would check actual node attributes
	// For now, always return true
	return true
}

func (rp *ResourcePool) updateNodeInventory(nodeID string, request *ResourceRequest, allocate bool) error {
	rp.inventoryMu.Lock()
	defer rp.inventoryMu.Unlock()

	inventory, exists := rp.nodeInventories[nodeID]
	if !exists {
		return fmt.Errorf("node inventory not found: %s", nodeID)
	}

	if allocate {
		// Allocate resources
		inventory.UsedCPU += request.CPUCores
		inventory.UsedMemory += int64(request.MemoryGB * 1024 * 1024 * 1024)
		inventory.UsedStorage += int64(request.StorageGB * 1024 * 1024 * 1024)
	} else {
		// Release resources
		inventory.UsedCPU -= request.CPUCores
		inventory.UsedMemory -= int64(request.MemoryGB * 1024 * 1024 * 1024)
		inventory.UsedStorage -= int64(request.StorageGB * 1024 * 1024 * 1024)

		// Ensure non-negative
		if inventory.UsedCPU < 0 {
			inventory.UsedCPU = 0
		}
		if inventory.UsedMemory < 0 {
			inventory.UsedMemory = 0
		}
		if inventory.UsedStorage < 0 {
			inventory.UsedStorage = 0
		}
	}

	return nil
}

func (rp *ResourcePool) queueRequest(request *ResourceRequest) {
	rp.requestsMu.Lock()
	defer rp.requestsMu.Unlock()

	rp.pendingRequests = append(rp.pendingRequests, request)

	// Sort by priority
	sort.Slice(rp.pendingRequests, func(i, j int) bool {
		return rp.pendingRequests[i].Priority > rp.pendingRequests[j].Priority
	})
}

func (rp *ResourcePool) processPendingRequests(ctx context.Context) {
	rp.requestsMu.Lock()
	defer rp.requestsMu.Unlock()

	processed := make([]*ResourceRequest, 0)

	for _, request := range rp.pendingRequests {
		// Try to allocate
		allocation, err := rp.AllocateResources(ctx, request)
		if err == nil && allocation != nil {
			processed = append(processed, request)
			rp.logger.Info("Pending request processed", "request_id", request.ID)
		}
	}

	// Remove processed requests
	if len(processed) > 0 {
		newPending := make([]*ResourceRequest, 0)
		for _, request := range rp.pendingRequests {
			found := false
			for _, proc := range processed {
				if request.ID == proc.ID {
					found = true
					break
				}
			}
			if !found {
				newPending = append(newPending, request)
			}
		}
		rp.pendingRequests = newPending
	}
}

func (rp *ResourcePool) updateMetrics() {
	rp.inventoryMu.RLock()
	defer rp.inventoryMu.RUnlock()

	totalCPU := float64(0)
	usedCPU := float64(0)
	totalMemory := float64(0)
	usedMemory := float64(0)
	totalStorage := float64(0)
	usedStorage := float64(0)

	for _, inventory := range rp.nodeInventories {
		totalCPU += inventory.TotalCPU
		usedCPU += inventory.UsedCPU
		totalMemory += float64(inventory.TotalMemory)
		usedMemory += float64(inventory.UsedMemory)
		totalStorage += float64(inventory.TotalStorage)
		usedStorage += float64(inventory.UsedStorage)
	}

	resourcePoolCapacity.WithLabelValues("cpu").Set(totalCPU)
	resourcePoolCapacity.WithLabelValues("memory").Set(totalMemory)
	resourcePoolCapacity.WithLabelValues("storage").Set(totalStorage)

	resourcePoolUsage.WithLabelValues("cpu").Set(usedCPU)
	resourcePoolUsage.WithLabelValues("memory").Set(usedMemory)
	resourcePoolUsage.WithLabelValues("storage").Set(usedStorage)
}

// ResourceScheduler implements scheduling algorithms
type ResourceScheduler struct {
	config    *FederationConfig
	logger    Logger
	algorithm SchedulingAlgorithm
}

// SchedulingAlgorithm represents a scheduling algorithm
type SchedulingAlgorithm string

const (
	AlgorithmRoundRobin    SchedulingAlgorithm = "round_robin"
	AlgorithmLeastLoaded   SchedulingAlgorithm = "least_loaded"
	AlgorithmBestFit       SchedulingAlgorithm = "best_fit"
	AlgorithmSpreadEvenly  SchedulingAlgorithm = "spread_evenly"
	AlgorithmAffinity      SchedulingAlgorithm = "affinity"
)

// NewResourceScheduler creates a new resource scheduler
func NewResourceScheduler(config *FederationConfig, logger Logger) *ResourceScheduler {
	return &ResourceScheduler{
		config:    config,
		logger:    logger,
		algorithm: AlgorithmLeastLoaded, // Default algorithm
	}
}

// SelectNode selects the best node for a resource request
func (rs *ResourceScheduler) SelectNode(nodes []string, request *ResourceRequest) string {
	if len(nodes) == 0 {
		return ""
	}

	switch rs.algorithm {
	case AlgorithmRoundRobin:
		return rs.roundRobinSelect(nodes)
	case AlgorithmLeastLoaded:
		return rs.leastLoadedSelect(nodes, request)
	case AlgorithmBestFit:
		return rs.bestFitSelect(nodes, request)
	case AlgorithmSpreadEvenly:
		return rs.spreadEvenlySelect(nodes)
	case AlgorithmAffinity:
		return rs.affinitySelect(nodes, request)
	default:
		return nodes[0]
	}
}

func (rs *ResourceScheduler) roundRobinSelect(nodes []string) string {
	// Simple round-robin selection
	// In production, would maintain state for true round-robin
	return nodes[0]
}

func (rs *ResourceScheduler) leastLoadedSelect(nodes []string, request *ResourceRequest) string {
	// Select node with lowest resource utilization
	// In production, would calculate actual utilization
	return nodes[0]
}

func (rs *ResourceScheduler) bestFitSelect(nodes []string, request *ResourceRequest) string {
	// Select node that best fits the request (minimal waste)
	// In production, would calculate fit score
	return nodes[0]
}

func (rs *ResourceScheduler) spreadEvenlySelect(nodes []string) string {
	// Spread load evenly across all nodes
	// In production, would track allocations per node
	return nodes[len(nodes)/2]
}

func (rs *ResourceScheduler) affinitySelect(nodes []string, request *ResourceRequest) string {
	// Select based on affinity/anti-affinity rules
	// In production, would check constraints
	return nodes[0]
}

// ResourceOptimizer optimizes resource allocation
type ResourceOptimizer struct {
	config *FederationConfig
	logger Logger
}

// NewResourceOptimizer creates a new resource optimizer
func NewResourceOptimizer(config *FederationConfig, logger Logger) *ResourceOptimizer {
	return &ResourceOptimizer{
		config: config,
		logger: logger,
	}
}

// OptimizeAllocations optimizes resource allocations
func (ro *ResourceOptimizer) OptimizeAllocations(allocations []*ResourceAllocation) []*ResourceAllocation {
	// Implement bin packing or other optimization algorithms
	// For now, return as-is
	return allocations
}

// CalculateFragmentation calculates resource fragmentation
func (ro *ResourceOptimizer) CalculateFragmentation(inventory *ResourceInventory) float64 {
	if inventory.TotalCPU == 0 {
		return 0
	}

	// Simple fragmentation calculation
	cpuFragmentation := (inventory.TotalCPU - inventory.UsedCPU) / inventory.TotalCPU
	memFragmentation := float64(inventory.TotalMemory-inventory.UsedMemory) / float64(inventory.TotalMemory)
	storageFragmentation := float64(inventory.TotalStorage-inventory.UsedStorage) / float64(inventory.TotalStorage)

	// Average fragmentation
	return (cpuFragmentation + memFragmentation + storageFragmentation) / 3
}

// RebalanceResources rebalances resources across nodes
func (ro *ResourceOptimizer) RebalanceResources(nodes map[string]*ResourceInventory) map[string][]*ResourceAllocation {
	// Implement resource rebalancing algorithm
	// For now, return empty migrations
	return make(map[string][]*ResourceAllocation)
}

// PredictResourceDemand predicts future resource demand
func (ro *ResourceOptimizer) PredictResourceDemand(history []*ResourceRequest) *ResourceRequest {
	if len(history) == 0 {
		return nil
	}

	// Simple prediction based on average
	totalCPU := float64(0)
	totalMemory := float64(0)
	totalStorage := float64(0)

	for _, req := range history {
		totalCPU += req.CPUCores
		totalMemory += req.MemoryGB
		totalStorage += req.StorageGB
	}

	n := float64(len(history))

	return &ResourceRequest{
		CPUCores:  totalCPU / n,
		MemoryGB:  totalMemory / n,
		StorageGB: totalStorage / n,
	}
}

// CalculateEfficiency calculates resource efficiency
func (ro *ResourceOptimizer) CalculateEfficiency(inventory *ResourceInventory) float64 {
	if inventory.TotalCPU == 0 {
		return 0
	}

	cpuEfficiency := inventory.UsedCPU / inventory.TotalCPU
	memEfficiency := float64(inventory.UsedMemory) / float64(inventory.TotalMemory)
	storageEfficiency := float64(inventory.UsedStorage) / float64(inventory.TotalStorage)

	// Weighted average (CPU is more important)
	return (cpuEfficiency*0.5 + memEfficiency*0.3 + storageEfficiency*0.2)
}

// RecommendScaling recommends scaling actions
func (ro *ResourceOptimizer) RecommendScaling(efficiency float64, pending int) string {
	if efficiency > 0.9 && pending > 0 {
		return "scale_up"
	} else if efficiency < 0.3 && pending == 0 {
		return "scale_down"
	}
	return "maintain"
}

// CalculateCost calculates resource cost
func (ro *ResourceOptimizer) CalculateCost(allocation *ResourceAllocation) float64 {
	// Simple cost model
	cpuCost := float64(allocation.CPU) * 0.05      // $0.05 per CPU core per hour
	memCost := float64(allocation.MemoryGB) * 0.01      // $0.01 per GB per hour
	storageCost := float64(allocation.StorageGB) * 0.001 // $0.001 per GB per hour

	duration := time.Since(allocation.AllocatedAt).Hours()

	return (cpuCost + memCost + storageCost) * duration
}

// FindOptimalPlacement finds optimal placement for a workload
func (ro *ResourceOptimizer) FindOptimalPlacement(request *ResourceRequest, nodes []*Node) string {
	if len(nodes) == 0 {
		return ""
	}

	type nodeScore struct {
		nodeID string
		score  float64
	}

	scores := make([]nodeScore, 0, len(nodes))

	for _, node := range nodes {
		score := ro.calculatePlacementScore(node, request)
		scores = append(scores, nodeScore{
			nodeID: node.ID,
			score:  score,
		})
	}

	// Sort by score (highest first)
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].score > scores[j].score
	})

	if len(scores) > 0 {
		return scores[0].nodeID
	}

	return ""
}

func (ro *ResourceOptimizer) calculatePlacementScore(node *Node, request *ResourceRequest) float64 {
	// Calculate placement score based on multiple factors
	score := float64(100)

	// Resource availability
	availableCPU := node.Capabilities.Resources.TotalCPU - node.Capabilities.Resources.UsedCPU
	availableMemory := float64(node.Capabilities.Resources.TotalMemory-node.Capabilities.Resources.UsedMemory) / (1024 * 1024 * 1024)

	if availableCPU < request.CPUCores || availableMemory < request.MemoryGB {
		return 0 // Cannot satisfy request
	}

	// Preference for less loaded nodes
	cpuUtilization := node.Capabilities.Resources.UsedCPU / node.Capabilities.Resources.TotalCPU
	score *= (1 - cpuUtilization)

	// Network proximity (in production, would use actual latency)
	score *= 0.9

	// Node health
	if node.State == NodeStateActive {
		score *= 1.0
	} else if node.State == NodeStateUnhealthy {
		score *= 0.5
	}

	return math.Max(0, math.Min(100, score))
}