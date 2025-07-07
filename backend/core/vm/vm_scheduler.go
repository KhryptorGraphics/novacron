package vm

import (
	"context"
	"fmt"
	"log"
	"sort"
	"sync"
	"time"
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
	NodeID            string  `json:"node_id"`
	TotalCPU          int     `json:"total_cpu"`
	UsedCPU           int     `json:"used_cpu"`
	TotalMemoryMB     int     `json:"total_memory_mb"`
	UsedMemoryMB      int     `json:"used_memory_mb"`
	TotalDiskGB       int     `json:"total_disk_gb"`
	UsedDiskGB        int     `json:"used_disk_gb"`
	CPUUsagePercent   float64 `json:"cpu_usage_percent"`
	MemoryUsagePercent float64 `json:"memory_usage_percent"`
	DiskUsagePercent   float64 `json:"disk_usage_percent"`
	VMCount           int     `json:"vm_count"`
	Status            string  `json:"status"`
	Labels            map[string]string `json:"labels,omitempty"`
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
	config         SchedulerConfig
	nodes          map[string]*NodeResourceInfo
	nodesMutex     sync.RWMutex
	customScheduler func(vm *VM, nodes []*NodeResourceInfo) (string, error)
}

// NewVMScheduler creates a new VM scheduler
func NewVMScheduler(config SchedulerConfig) *VMScheduler {
	return &VMScheduler{
		config: config,
		nodes:  make(map[string]*NodeResourceInfo),
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
	// For now, return an empty map - this should be implemented based on actual allocation tracking
	// In a real implementation, this would track active VM allocations
	return make(map[string]ResourceAllocation)
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

// hasEnoughResources checks if a node has enough resources for a VM
func (s *VMScheduler) hasEnoughResources(node *NodeResourceInfo, vm *VM) bool {
	// Check if the node has reached the maximum number of VMs
	if s.config.MaxVMsPerNode > 0 && node.VMCount >= s.config.MaxVMsPerNode {
		return false
	}
	
	// Calculate required resources
	requiredCPU := vm.config.CPUShares
	requiredMemoryMB := vm.config.MemoryMB
	
	// Check CPU
	availableCPU := node.TotalCPU - node.UsedCPU
	if s.config.MaxCPUOvercommit > 1.0 {
		availableCPU = int(float64(node.TotalCPU) * s.config.MaxCPUOvercommit) - node.UsedCPU
	}
	
	if requiredCPU > availableCPU {
		return false
	}
	
	// Check memory
	availableMemoryMB := node.TotalMemoryMB - node.UsedMemoryMB
	if s.config.MaxMemoryOvercommit > 1.0 {
		availableMemoryMB = int(float64(node.TotalMemoryMB) * s.config.MaxMemoryOvercommit) - node.UsedMemoryMB
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
	
	// Calculate required resources
	requiredCPU := vm.config.CPUShares
	requiredMemoryMB := vm.config.MemoryMB
	
	// Update node resources
	node.UsedCPU += requiredCPU
	node.UsedMemoryMB += requiredMemoryMB
	node.VMCount++
	
	// Update usage percentages
	node.CPUUsagePercent = float64(node.UsedCPU) / float64(node.TotalCPU) * 100
	node.MemoryUsagePercent = float64(node.UsedMemoryMB) / float64(node.TotalMemoryMB) * 100
	
	log.Printf("Reserved resources on node %s for VM %s: CPU=%d, Memory=%dMB", nodeID, vm.ID(), requiredCPU, requiredMemoryMB)
	
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
	
	// Calculate required resources
	requiredCPU := vm.config.CPUShares
	requiredMemoryMB := vm.config.MemoryMB
	
	// Update node resources
	node.UsedCPU -= requiredCPU
	if node.UsedCPU < 0 {
		node.UsedCPU = 0
	}
	
	node.UsedMemoryMB -= requiredMemoryMB
	if node.UsedMemoryMB < 0 {
		node.UsedMemoryMB = 0
	}
	
	node.VMCount--
	if node.VMCount < 0 {
		node.VMCount = 0
	}
	
	// Update usage percentages
	node.CPUUsagePercent = float64(node.UsedCPU) / float64(node.TotalCPU) * 100
	node.MemoryUsagePercent = float64(node.UsedMemoryMB) / float64(node.TotalMemoryMB) * 100
	
	log.Printf("Released resources on node %s for VM %s: CPU=%d, Memory=%dMB", nodeID, vm.ID(), requiredCPU, requiredMemoryMB)
	
	return nil
}
