package vm

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	// "time" // Currently unused
)

// SchedulerPolicy represents a VM scheduling policy
type SchedulerPolicy string

const (
	// SchedulerPolicyRoundRobin represents a round-robin scheduling policy
	SchedulerPolicyRoundRobin SchedulerPolicy = "round_robin"

	// SchedulerPolicyBinPacking represents a bin-packing scheduling policy
	SchedulerPolicyBinPacking SchedulerPolicy = "bin_packing"

	// SchedulerPolicySpreadOut represents a spread-out scheduling policy
	SchedulerPolicySpreadOut SchedulerPolicy = "spread_out"

	// SchedulerPolicyCustom represents a custom scheduling policy
	SchedulerPolicyCustom SchedulerPolicy = "custom"
)

// NodeResourceInfo represents resource information for a node
type NodeResourceInfo struct {
	NodeID             string            `json:"node_id"`
	TotalCPU           int               `json:"total_cpu"`
	UsedCPU            int               `json:"used_cpu"`
	TotalMemoryMB      int               `json:"total_memory_mb"`
	UsedMemoryMB       int               `json:"used_memory_mb"`
	TotalDiskGB        int               `json:"total_disk_gb"`
	UsedDiskGB         int               `json:"used_disk_gb"`
	CPUUsagePercent    float64           `json:"cpu_usage_percent"`
	MemoryUsagePercent float64           `json:"memory_usage_percent"`
	DiskUsagePercent   float64           `json:"disk_usage_percent"`
	VMCount            int               `json:"vm_count"`
	Status             string            `json:"status"`
	Labels             map[string]string `json:"labels,omitempty"`
}

// SchedulerConfig represents VM scheduler configuration
type SchedulerConfig struct {
	Policy                 SchedulerPolicy `json:"policy"`
	EnableResourceChecking bool            `json:"enable_resource_checking"`
	EnableAntiAffinity     bool            `json:"enable_anti_affinity"`
	EnableNodeLabels       bool            `json:"enable_node_labels"`
	MaxVMsPerNode          int             `json:"max_vms_per_node"`
	MaxCPUOvercommit       float64         `json:"max_cpu_overcommit"`
	MaxMemoryOvercommit    float64         `json:"max_memory_overcommit"`
}

// VMScheduler schedules VMs on nodes
type VMScheduler struct {
	config           SchedulerConfig
	nodes            map[string]*NodeResourceInfo
	nodesMutex       sync.RWMutex
	allocations      map[string]ResourceAllocation
	allocationsMutex sync.RWMutex
	customScheduler  func(vm *VM, nodes []*NodeResourceInfo) (string, error)
}

// NewVMScheduler creates a new VM scheduler
func NewVMScheduler(config SchedulerConfig) *VMScheduler {
	return &VMScheduler{
		config:      config,
		nodes:       make(map[string]*NodeResourceInfo),
		allocations: make(map[string]ResourceAllocation),
	}
}

// RegisterNode registers a node with the scheduler
func (s *VMScheduler) RegisterNode(nodeInfo *NodeResourceInfo) error {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()

	// Check if node already exists
	if _, exists := s.nodes[nodeInfo.NodeID]; exists {
		return fmt.Errorf("node %s is already registered", nodeInfo.NodeID)
	}

	// Register node
	s.nodes[nodeInfo.NodeID] = nodeInfo

	log.Printf("Registered node %s with scheduler", nodeInfo.NodeID)

	return nil
}

// UpdateNode updates a node's resource information
func (s *VMScheduler) UpdateNode(nodeInfo *NodeResourceInfo) error {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()

	// Check if node exists
	if _, exists := s.nodes[nodeInfo.NodeID]; !exists {
		return fmt.Errorf("node %s is not registered", nodeInfo.NodeID)
	}

	// Update node
	s.nodes[nodeInfo.NodeID] = nodeInfo

	return nil
}

// UnregisterNode unregisters a node from the scheduler
func (s *VMScheduler) UnregisterNode(nodeID string) error {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()

	// Check if node exists
	if _, exists := s.nodes[nodeID]; !exists {
		return fmt.Errorf("node %s is not registered", nodeID)
	}

	// Unregister node
	delete(s.nodes, nodeID)

	log.Printf("Unregistered node %s from scheduler", nodeID)

	return nil
}

// GetNode returns a node's resource information
func (s *VMScheduler) GetNode(nodeID string) (*NodeResourceInfo, error) {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()

	// Check if node exists
	node, exists := s.nodes[nodeID]
	if !exists {
		return nil, fmt.Errorf("node %s is not registered", nodeID)
	}

	return node, nil
}

// ResourceAllocation represents a resource allocation for a VM
type ResourceAllocation struct {
	VMID      string
	NodeID    string
	CPUCores  int
	MemoryMB  int
	DiskGB    int
	RequestID string
}

// GetActiveAllocations returns all active resource allocations
func (s *VMScheduler) GetActiveAllocations() map[string]ResourceAllocation {
	s.allocationsMutex.RLock()
	defer s.allocationsMutex.RUnlock()

	allocations := make(map[string]ResourceAllocation, len(s.allocations))
	for vmID, allocation := range s.allocations {
		allocations[vmID] = allocation
	}

	return allocations
}

// ListNodes returns all registered nodes
func (s *VMScheduler) ListNodes() []*NodeResourceInfo {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()

	// Create a copy of the nodes
	nodes := make([]*NodeResourceInfo, 0, len(s.nodes))
	for _, node := range s.nodes {
		nodes = append(nodes, node)
	}

	return nodes
}

// SetCustomScheduler sets a custom scheduler function
func (s *VMScheduler) SetCustomScheduler(scheduler func(vm *VM, nodes []*NodeResourceInfo) (string, error)) {
	s.customScheduler = scheduler
}

// ScheduleVM schedules a VM on a node
func (s *VMScheduler) ScheduleVM(ctx context.Context, vm *VM) (string, error) {
	s.nodesMutex.RLock()
	defer s.nodesMutex.RUnlock()

	// Check if there are any nodes
	if len(s.nodes) == 0 {
		return "", fmt.Errorf("no nodes available for scheduling")
	}

	// Get a list of nodes
	nodes := make([]*NodeResourceInfo, 0, len(s.nodes))
	for _, node := range s.nodes {
		// Skip nodes that are not available
		if node.Status != "available" {
			continue
		}

		nodes = append(nodes, node)
	}

	// Check if there are any available nodes
	if len(nodes) == 0 {
		return "", fmt.Errorf("no available nodes for scheduling")
	}

	nodes, err := applyPlacementConstraints(vm, nodes)
	if err != nil {
		return "", err
	}

	// Use the appropriate scheduling policy
	switch s.config.Policy {
	case SchedulerPolicyRoundRobin:
		return s.scheduleRoundRobin(vm, nodes)
	case SchedulerPolicyBinPacking:
		return s.scheduleBinPacking(vm, nodes)
	case SchedulerPolicySpreadOut:
		return s.scheduleSpreadOut(vm, nodes)
	case SchedulerPolicyCustom:
		if s.customScheduler != nil {
			return s.customScheduler(vm, nodes)
		}
		return "", fmt.Errorf("custom scheduler not set")
	default:
		return "", fmt.Errorf("unknown scheduling policy: %s", s.config.Policy)
	}
}

// scheduleRoundRobin schedules a VM using round-robin policy
func (s *VMScheduler) scheduleRoundRobin(vm *VM, nodes []*NodeResourceInfo) (string, error) {
	// Sort nodes by VM count
	sort.Slice(nodes, func(i, j int) bool {
		return nodes[i].VMCount < nodes[j].VMCount
	})

	// Check resource constraints if enabled
	if s.config.EnableResourceChecking {
		for _, node := range nodes {
			// Check if the node has enough resources
			if s.hasEnoughResources(node, vm) {
				return node.NodeID, nil
			}
		}

		return "", fmt.Errorf("no node has enough resources for the VM")
	}

	// If resource checking is disabled, just return the node with the fewest VMs
	return nodes[0].NodeID, nil
}

// scheduleBinPacking schedules a VM using bin-packing policy
func (s *VMScheduler) scheduleBinPacking(vm *VM, nodes []*NodeResourceInfo) (string, error) {
	// Sort nodes by resource usage (highest first)
	sort.Slice(nodes, func(i, j int) bool {
		// Calculate a combined resource usage score
		scoreI := nodes[i].CPUUsagePercent*0.5 + nodes[i].MemoryUsagePercent*0.5
		scoreJ := nodes[j].CPUUsagePercent*0.5 + nodes[j].MemoryUsagePercent*0.5
		return scoreI > scoreJ
	})

	// Check resource constraints if enabled
	if s.config.EnableResourceChecking {
		for _, node := range nodes {
			// Check if the node has enough resources
			if s.hasEnoughResources(node, vm) {
				return node.NodeID, nil
			}
		}

		return "", fmt.Errorf("no node has enough resources for the VM")
	}

	// If resource checking is disabled, just return the node with the highest resource usage
	return nodes[0].NodeID, nil
}

// scheduleSpreadOut schedules a VM using spread-out policy
func (s *VMScheduler) scheduleSpreadOut(vm *VM, nodes []*NodeResourceInfo) (string, error) {
	// Sort nodes by resource usage (lowest first)
	sort.Slice(nodes, func(i, j int) bool {
		// Calculate a combined resource usage score
		scoreI := nodes[i].CPUUsagePercent*0.5 + nodes[i].MemoryUsagePercent*0.5
		scoreJ := nodes[j].CPUUsagePercent*0.5 + nodes[j].MemoryUsagePercent*0.5
		return scoreI < scoreJ
	})

	// Check resource constraints if enabled
	if s.config.EnableResourceChecking {
		for _, node := range nodes {
			// Check if the node has enough resources
			if s.hasEnoughResources(node, vm) {
				return node.NodeID, nil
			}
		}

		return "", fmt.Errorf("no node has enough resources for the VM")
	}

	// If resource checking is disabled, just return the node with the lowest resource usage
	return nodes[0].NodeID, nil
}

func applyPlacementConstraints(vm *VM, nodes []*NodeResourceInfo) ([]*NodeResourceInfo, error) {
	if vm == nil || vm.config.Placement == nil {
		return nodes, nil
	}

	excluded := normalizeNodeIDSet(vm.config.Placement.ExcludedNodes)
	filteredNodes := make([]*NodeResourceInfo, 0, len(nodes))
	for _, node := range nodes {
		if _, skip := excluded[node.NodeID]; skip {
			continue
		}
		filteredNodes = append(filteredNodes, node)
	}

	if len(filteredNodes) == 0 {
		return nil, fmt.Errorf("no available nodes remain after applying placement exclusions")
	}

	preferred := normalizeNodeIDSet(vm.config.Placement.PreferredNodes)
	if len(preferred) == 0 {
		return filteredNodes, nil
	}

	preferredNodes := make([]*NodeResourceInfo, 0, len(filteredNodes))
	for _, node := range filteredNodes {
		if _, ok := preferred[node.NodeID]; ok {
			preferredNodes = append(preferredNodes, node)
		}
	}

	if len(preferredNodes) > 0 {
		return preferredNodes, nil
	}

	// Preferred nodes are a soft constraint; fall back to the remaining eligible nodes.
	return filteredNodes, nil
}

func normalizeNodeIDSet(nodeIDs []string) map[string]struct{} {
	if len(nodeIDs) == 0 {
		return nil
	}

	normalized := make(map[string]struct{}, len(nodeIDs))
	for _, nodeID := range nodeIDs {
		nodeID = strings.TrimSpace(nodeID)
		if nodeID == "" {
			continue
		}
		normalized[nodeID] = struct{}{}
	}

	if len(normalized) == 0 {
		return nil
	}

	return normalized
}

// hasEnoughResources checks if a node has enough resources for a VM
func (s *VMScheduler) hasEnoughResources(node *NodeResourceInfo, vm *VM) bool {
	// Check if the node has reached the maximum number of VMs
	if s.config.MaxVMsPerNode > 0 && node.VMCount >= s.config.MaxVMsPerNode {
		return false
	}

	// Calculate required resources
	requiredCPU := cpuAllocationForVM(vm)
	requiredMemoryMB := vm.config.MemoryMB

	// Check CPU
	availableCPU := node.TotalCPU - node.UsedCPU
	if s.config.MaxCPUOvercommit > 1.0 {
		availableCPU = int(float64(node.TotalCPU)*s.config.MaxCPUOvercommit) - node.UsedCPU
	}

	if requiredCPU > availableCPU {
		return false
	}

	// Check memory
	availableMemoryMB := node.TotalMemoryMB - node.UsedMemoryMB
	if s.config.MaxMemoryOvercommit > 1.0 {
		availableMemoryMB = int(float64(node.TotalMemoryMB)*s.config.MaxMemoryOvercommit) - node.UsedMemoryMB
	}

	if requiredMemoryMB > availableMemoryMB {
		return false
	}

	// Check node labels if enabled
	if s.config.EnableNodeLabels && vm.config.Tags != nil {
		// Check if the node has all required labels
		for key, value := range vm.config.Tags {
			if nodeValue, exists := node.Labels[key]; !exists || nodeValue != value {
				return false
			}
		}
	}

	return true
}

// ReserveResources reserves resources on a node for a VM
func (s *VMScheduler) ReserveResources(nodeID string, vm *VM) error {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()

	// Check if node exists
	node, exists := s.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node %s is not registered", nodeID)
	}

	// Recheck capacity at reservation time to avoid stale scheduling decisions.
	if s.config.EnableResourceChecking && !s.hasEnoughResources(node, vm) {
		return fmt.Errorf("node %s no longer has enough resources for VM %s", nodeID, vm.ID())
	}

	// Calculate required resources
	requiredCPU := cpuAllocationForVM(vm)
	requiredMemoryMB := vm.config.MemoryMB
	requiredDiskGB := diskAllocationForVM(vm)

	// Update node resources
	node.UsedCPU += requiredCPU
	node.UsedMemoryMB += requiredMemoryMB
	node.UsedDiskGB += requiredDiskGB
	node.VMCount++

	updateNodeUsage(node)

	allocation := ResourceAllocation{
		VMID:      vm.ID(),
		NodeID:    nodeID,
		CPUCores:  requiredCPU,
		MemoryMB:  requiredMemoryMB,
		DiskGB:    requiredDiskGB,
		RequestID: firstNonEmpty(vm.ResourceID(), vm.ID()),
	}

	s.allocationsMutex.Lock()
	s.allocations[vm.ID()] = allocation
	s.allocationsMutex.Unlock()

	log.Printf(
		"Reserved resources on node %s for VM %s: CPU=%d, Memory=%dMB, Disk=%dGB",
		nodeID,
		vm.ID(),
		requiredCPU,
		requiredMemoryMB,
		requiredDiskGB,
	)

	return nil
}

// ReleaseResources releases resources on a node for a VM
func (s *VMScheduler) ReleaseResources(nodeID string, vm *VM) error {
	s.nodesMutex.Lock()
	defer s.nodesMutex.Unlock()

	// Check if node exists
	node, exists := s.nodes[nodeID]
	if !exists {
		return fmt.Errorf("node %s is not registered", nodeID)
	}

	requiredCPU := cpuAllocationForVM(vm)
	requiredMemoryMB := vm.config.MemoryMB
	requiredDiskGB := diskAllocationForVM(vm)

	s.allocationsMutex.Lock()
	if allocation, exists := s.allocations[vm.ID()]; exists {
		requiredCPU = allocation.CPUCores
		requiredMemoryMB = allocation.MemoryMB
		requiredDiskGB = allocation.DiskGB
		nodeID = allocation.NodeID
		delete(s.allocations, vm.ID())
		if allocationNode, allocationExists := s.nodes[nodeID]; allocationExists {
			node = allocationNode
		}
	}
	s.allocationsMutex.Unlock()

	// Update node resources
	node.UsedCPU -= requiredCPU
	if node.UsedCPU < 0 {
		node.UsedCPU = 0
	}

	node.UsedMemoryMB -= requiredMemoryMB
	if node.UsedMemoryMB < 0 {
		node.UsedMemoryMB = 0
	}
	node.UsedDiskGB -= requiredDiskGB
	if node.UsedDiskGB < 0 {
		node.UsedDiskGB = 0
	}

	node.VMCount--
	if node.VMCount < 0 {
		node.VMCount = 0
	}

	updateNodeUsage(node)

	log.Printf(
		"Released resources on node %s for VM %s: CPU=%d, Memory=%dMB, Disk=%dGB",
		nodeID,
		vm.ID(),
		requiredCPU,
		requiredMemoryMB,
		requiredDiskGB,
	)

	return nil
}

func updateNodeUsage(node *NodeResourceInfo) {
	if node.TotalCPU > 0 {
		node.CPUUsagePercent = float64(node.UsedCPU) / float64(node.TotalCPU) * 100
	} else {
		node.CPUUsagePercent = 0
	}

	if node.TotalMemoryMB > 0 {
		node.MemoryUsagePercent = float64(node.UsedMemoryMB) / float64(node.TotalMemoryMB) * 100
	} else {
		node.MemoryUsagePercent = 0
	}

	if node.TotalDiskGB > 0 {
		node.DiskUsagePercent = float64(node.UsedDiskGB) / float64(node.TotalDiskGB) * 100
	} else {
		node.DiskUsagePercent = 0
	}
}

func cpuAllocationForVM(vm *VM) int {
	if vm == nil || vm.config.CPUShares <= 0 {
		return 0
	}

	// KVM configs in this repo often use Linux-style CPU shares where 1024 ~= 1 vCPU.
	if vm.config.CPUShares > 128 {
		cores := vm.config.CPUShares / 1024
		if vm.config.CPUShares%1024 != 0 {
			cores++
		}
		if cores < 1 {
			return 1
		}
		return cores
	}

	return vm.config.CPUShares
}

func diskAllocationForVM(vm *VM) int {
	if vm == nil || vm.config.DiskSizeGB < 0 {
		return 0
	}
	return vm.config.DiskSizeGB
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
