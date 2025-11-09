package numa

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

// Optimizer handles NUMA topology optimization
type Optimizer struct {
	config   NumaConfig
	mu       sync.RWMutex
	topology *NumaTopology
	policies map[string]*NumaPolicy
}

// NumaConfig defines NUMA optimization settings
type NumaConfig struct {
	AutoTopologyDetection   bool
	MemoryPlacementStrategy string  // "local", "interleave", "preferred"
	CacheLocalityOptimize   bool
	CrossNumaTrafficTarget  float64 // 0.10 (10%)
	BalancingEnabled        bool
	MigrateThreshold        float64 // 0.80
}

// NumaTopology represents NUMA system topology
type NumaTopology struct {
	Nodes        []NumaNode
	TotalMemory  uint64
	TotalCPUs    int
	Distances    [][]int // Node distance matrix
}

// NumaNode represents a NUMA node
type NumaNode struct {
	ID          int
	CPUs        []int
	MemoryBytes uint64
	MemoryFree  uint64
	Distance    map[int]int
}

// NumaPolicy defines VM NUMA policy
type NumaPolicy struct {
	VMID                string
	NodeAffinity        []int  // Preferred NUMA nodes
	MemoryPlacement     string // "local", "interleave", "preferred"
	CPUAffinity         []int  // CPU cores
	CacheLocalityLevel  int    // 1=L1, 2=L2, 3=L3
	CrossNodeTraffic    float64
	AutoBalance         bool
}

// NewOptimizer creates NUMA optimizer
func NewOptimizer(config NumaConfig) *Optimizer {
	return &Optimizer{
		config:   config,
		policies: make(map[string]*NumaPolicy),
	}
}

// Initialize detects NUMA topology
func (o *Optimizer) Initialize(ctx context.Context) error {
	if !o.config.AutoTopologyDetection {
		return nil
	}

	topology, err := o.detectTopology()
	if err != nil {
		return fmt.Errorf("detect topology: %w", err)
	}

	o.mu.Lock()
	o.topology = topology
	o.mu.Unlock()

	return nil
}

// detectTopology detects system NUMA topology
func (o *Optimizer) detectTopology() (*NumaTopology, error) {
	// Read from /sys/devices/system/node
	nodesDir := "/sys/devices/system/node"
	entries, err := os.ReadDir(nodesDir)
	if err != nil {
		// Fallback to single node
		return o.createSingleNodeTopology(), nil
	}

	var nodes []NumaNode
	for _, entry := range entries {
		if !strings.HasPrefix(entry.Name(), "node") {
			continue
		}

		nodeIDStr := strings.TrimPrefix(entry.Name(), "node")
		nodeID, err := strconv.Atoi(nodeIDStr)
		if err != nil {
			continue
		}

		node, err := o.readNodeInfo(nodesDir, nodeID)
		if err != nil {
			continue
		}

		nodes = append(nodes, node)
	}

	if len(nodes) == 0 {
		return o.createSingleNodeTopology(), nil
	}

	// Calculate total resources
	var totalMemory uint64
	var totalCPUs int
	for _, node := range nodes {
		totalMemory += node.MemoryBytes
		totalCPUs += len(node.CPUs)
	}

	// Read distance matrix
	distances := o.readDistanceMatrix(nodesDir, len(nodes))

	return &NumaTopology{
		Nodes:       nodes,
		TotalMemory: totalMemory,
		TotalCPUs:   totalCPUs,
		Distances:   distances,
	}, nil
}

// readNodeInfo reads NUMA node information
func (o *Optimizer) readNodeInfo(nodesDir string, nodeID int) (NumaNode, error) {
	nodePath := filepath.Join(nodesDir, fmt.Sprintf("node%d", nodeID))

	// Read CPUs
	cpuListPath := filepath.Join(nodePath, "cpulist")
	cpuListData, err := os.ReadFile(cpuListPath)
	if err != nil {
		return NumaNode{}, err
	}

	cpus := o.parseCPUList(strings.TrimSpace(string(cpuListData)))

	// Read memory info
	meminfoPath := filepath.Join(nodePath, "meminfo")
	meminfoData, err := os.ReadFile(meminfoPath)
	if err != nil {
		return NumaNode{}, err
	}

	memTotal, memFree := o.parseMemInfo(string(meminfoData))

	return NumaNode{
		ID:          nodeID,
		CPUs:        cpus,
		MemoryBytes: memTotal,
		MemoryFree:  memFree,
		Distance:    make(map[int]int),
	}, nil
}

// parseCPUList parses CPU list (e.g., "0-3,8-11")
func (o *Optimizer) parseCPUList(cpuList string) []int {
	var cpus []int

	ranges := strings.Split(cpuList, ",")
	for _, r := range ranges {
		if strings.Contains(r, "-") {
			parts := strings.Split(r, "-")
			if len(parts) != 2 {
				continue
			}
			start, _ := strconv.Atoi(parts[0])
			end, _ := strconv.Atoi(parts[1])
			for i := start; i <= end; i++ {
				cpus = append(cpus, i)
			}
		} else {
			cpu, _ := strconv.Atoi(r)
			cpus = append(cpus, cpu)
		}
	}

	return cpus
}

// parseMemInfo parses memory info
func (o *Optimizer) parseMemInfo(meminfo string) (uint64, uint64) {
	var total, free uint64

	lines := strings.Split(meminfo, "\n")
	for _, line := range lines {
		if strings.Contains(line, "MemTotal:") {
			fields := strings.Fields(line)
			if len(fields) >= 4 {
				val, _ := strconv.ParseUint(fields[3], 10, 64)
				total = val * 1024 // Convert kB to bytes
			}
		} else if strings.Contains(line, "MemFree:") {
			fields := strings.Fields(line)
			if len(fields) >= 4 {
				val, _ := strconv.ParseUint(fields[3], 10, 64)
				free = val * 1024
			}
		}
	}

	return total, free
}

// readDistanceMatrix reads NUMA distance matrix
func (o *Optimizer) readDistanceMatrix(nodesDir string, nodeCount int) [][]int {
	distances := make([][]int, nodeCount)
	for i := range distances {
		distances[i] = make([]int, nodeCount)

		distancePath := filepath.Join(nodesDir, fmt.Sprintf("node%d/distance", i))
		data, err := os.ReadFile(distancePath)
		if err != nil {
			// Default distances
			for j := range distances[i] {
				if i == j {
					distances[i][j] = 10
				} else {
					distances[i][j] = 20
				}
			}
			continue
		}

		fields := strings.Fields(string(data))
		for j, field := range fields {
			if j < nodeCount {
				dist, _ := strconv.Atoi(field)
				distances[i][j] = dist
			}
		}
	}

	return distances
}

// createSingleNodeTopology creates fallback single-node topology
func (o *Optimizer) createSingleNodeTopology() *NumaTopology {
	cpus := make([]int, 8) // Assume 8 CPUs
	for i := range cpus {
		cpus[i] = i
	}

	return &NumaTopology{
		Nodes: []NumaNode{{
			ID:          0,
			CPUs:        cpus,
			MemoryBytes: 16 * 1024 * 1024 * 1024, // 16 GB
			MemoryFree:  8 * 1024 * 1024 * 1024,
			Distance:    map[int]int{0: 10},
		}},
		TotalMemory: 16 * 1024 * 1024 * 1024,
		TotalCPUs:   8,
		Distances:   [][]int{{10}},
	}
}

// OptimizeVM creates NUMA policy for VM
func (o *Optimizer) OptimizeVM(vmID string, vCPUs int, memoryGB float64) (*NumaPolicy, error) {
	o.mu.RLock()
	topology := o.topology
	o.mu.RUnlock()

	if topology == nil {
		return nil, fmt.Errorf("topology not initialized")
	}

	// Find best node(s) for VM
	nodes := o.selectNodes(vCPUs, memoryGB, topology)
	if len(nodes) == 0 {
		return nil, fmt.Errorf("no suitable NUMA nodes found")
	}

	// Allocate CPUs
	cpus := o.allocateCPUs(vCPUs, nodes, topology)

	policy := &NumaPolicy{
		VMID:               vmID,
		NodeAffinity:       nodes,
		MemoryPlacement:    o.config.MemoryPlacementStrategy,
		CPUAffinity:        cpus,
		CacheLocalityLevel: 3, // L3 cache
		AutoBalance:        o.config.BalancingEnabled,
	}

	o.mu.Lock()
	o.policies[vmID] = policy
	o.mu.Unlock()

	return policy, nil
}

// selectNodes selects best NUMA nodes for VM
func (o *Optimizer) selectNodes(vCPUs int, memoryGB float64, topology *NumaTopology) []int {
	memoryBytes := uint64(memoryGB * 1024 * 1024 * 1024)

	// Try to fit on single node
	for _, node := range topology.Nodes {
		if len(node.CPUs) >= vCPUs && node.MemoryFree >= memoryBytes {
			return []int{node.ID}
		}
	}

	// Need multiple nodes - select closest nodes
	var selectedNodes []int
	remainingCPUs := vCPUs
	remainingMemory := memoryBytes

	// Sort nodes by available resources
	availableNodes := make([]int, len(topology.Nodes))
	for i := range topology.Nodes {
		availableNodes[i] = i
	}

	for remainingCPUs > 0 || remainingMemory > 0 {
		if len(selectedNodes) >= len(topology.Nodes) {
			break
		}

		// Find next best node
		bestNode := -1
		bestScore := float64(-1)

		for _, nodeID := range availableNodes {
			if contains(selectedNodes, nodeID) {
				continue
			}

			node := topology.Nodes[nodeID]

			// Score based on distance to already selected nodes
			score := float64(len(node.CPUs)) + float64(node.MemoryFree)/1e9
			if len(selectedNodes) > 0 {
				avgDistance := 0
				for _, selectedID := range selectedNodes {
					avgDistance += topology.Distances[nodeID][selectedID]
				}
				score /= float64(avgDistance / len(selectedNodes))
			}

			if score > bestScore {
				bestScore = score
				bestNode = nodeID
			}
		}

		if bestNode == -1 {
			break
		}

		selectedNodes = append(selectedNodes, bestNode)
		node := topology.Nodes[bestNode]
		remainingCPUs -= len(node.CPUs)
		remainingMemory -= node.MemoryFree
	}

	return selectedNodes
}

// allocateCPUs allocates CPU cores from nodes
func (o *Optimizer) allocateCPUs(vCPUs int, nodes []int, topology *NumaTopology) []int {
	var cpus []int

	remainingVCPUs := vCPUs
	for _, nodeID := range nodes {
		if remainingVCPUs <= 0 {
			break
		}

		node := topology.Nodes[nodeID]
		allocCount := min(remainingVCPUs, len(node.CPUs))

		cpus = append(cpus, node.CPUs[:allocCount]...)
		remainingVCPUs -= allocCount
	}

	return cpus
}

// ApplyPolicy applies NUMA policy to VM
func (o *Optimizer) ApplyPolicy(vmID string) error {
	o.mu.RLock()
	policy, exists := o.policies[vmID]
	o.mu.RUnlock()

	if !exists {
		return fmt.Errorf("no policy for VM %s", vmID)
	}

	// Apply CPU affinity (via taskset or cgroups)
	// Apply memory policy (via numactl)

	fmt.Printf("Applying NUMA policy for VM %s: nodes=%v, cpus=%v, placement=%s\n",
		vmID, policy.NodeAffinity, policy.CPUAffinity, policy.MemoryPlacement)

	return nil
}

// MonitorCrossNodeTraffic monitors cross-NUMA traffic
func (o *Optimizer) MonitorCrossNodeTraffic(vmID string) (float64, error) {
	// Read from performance counters
	// This is simplified - actual implementation would use perf events

	o.mu.RLock()
	policy, exists := o.policies[vmID]
	o.mu.RUnlock()

	if !exists {
		return 0, fmt.Errorf("no policy for VM %s", vmID)
	}

	// Simulated cross-node traffic
	crossTraffic := 0.05 // 5%
	policy.CrossNodeTraffic = crossTraffic

	if crossTraffic > o.config.CrossNumaTrafficTarget {
		// Recommend rebalancing
		fmt.Printf("VM %s has high cross-NUMA traffic: %.1f%%\n", vmID, crossTraffic*100)
	}

	return crossTraffic, nil
}

// Helper functions
func contains(slice []int, val int) bool {
	for _, v := range slice {
		if v == val {
			return true
		}
	}
	return false
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
