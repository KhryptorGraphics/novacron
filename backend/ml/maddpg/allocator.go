package maddpg

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"
)

// ResourceType represents different types of resources
type ResourceType string

const (
	CPU       ResourceType = "cpu"
	Memory    ResourceType = "memory"
	Bandwidth ResourceType = "bandwidth"
	Storage   ResourceType = "storage"
)

// Node represents a compute node in the distributed system
type Node struct {
	ID                 int     `json:"id"`
	CPUCapacity        float64 `json:"cpu_capacity"`
	MemoryCapacity     float64 `json:"memory_capacity"`
	BandwidthCapacity  float64 `json:"bandwidth_capacity"`
	StorageCapacity    float64 `json:"storage_capacity"`
	CPUUsage           float64 `json:"cpu_usage"`
	MemoryUsage        float64 `json:"memory_usage"`
	BandwidthUsage     float64 `json:"bandwidth_usage"`
	StorageUsage       float64 `json:"storage_usage"`
}

// GetObservation returns normalized observation vector for this node
func (n *Node) GetObservation() []float64 {
	return []float64{
		n.CPUUsage / max(n.CPUCapacity, 1e-6),
		n.MemoryUsage / max(n.MemoryCapacity, 1e-6),
		n.BandwidthUsage / max(n.BandwidthCapacity, 1e-6),
		n.StorageUsage / max(n.StorageCapacity, 1e-6),
		(n.CPUCapacity - n.CPUUsage) / max(n.CPUCapacity, 1e-6),
		(n.MemoryCapacity - n.MemoryUsage) / max(n.MemoryCapacity, 1e-6),
		(n.BandwidthCapacity - n.BandwidthUsage) / max(n.BandwidthCapacity, 1e-6),
		(n.StorageCapacity - n.StorageUsage) / max(n.StorageCapacity, 1e-6),
	}
}

// Workload represents a workload to be allocated
type Workload struct {
	ID                  int     `json:"id"`
	CPURequirement      float64 `json:"cpu_requirement"`
	MemoryRequirement   float64 `json:"memory_requirement"`
	BandwidthRequirement float64 `json:"bandwidth_requirement"`
	StorageRequirement  float64 `json:"storage_requirement"`
	Priority            float64 `json:"priority"`
	SLADeadline         float64 `json:"sla_deadline"`
}

// Allocation represents a resource allocation decision
type Allocation struct {
	WorkloadID int     `json:"workload_id"`
	NodeID     int     `json:"node_id"`
	CPUAlloc   float64 `json:"cpu_alloc"`
	MemAlloc   float64 `json:"mem_alloc"`
	BWAlloc    float64 `json:"bw_alloc"`
	StorageAlloc float64 `json:"storage_alloc"`
	Timestamp  time.Time `json:"timestamp"`
}

// MADDPGModel represents the trained MADDPG model
type MADDPGModel struct {
	ModelPath   string
	NumAgents   int
	StateDim    int
	ActionDim   int
	pythonPath  string
	mu          sync.RWMutex
	initialized bool
}

// NewMADDPGModel creates a new MADDPG model instance
func NewMADDPGModel(modelPath string, numAgents int) (*MADDPGModel, error) {
	// Check if model exists
	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("model path does not exist: %s", modelPath)
	}

	// Find Python interpreter
	pythonPath, err := exec.LookPath("python3")
	if err != nil {
		pythonPath, err = exec.LookPath("python")
		if err != nil {
			return nil, fmt.Errorf("python interpreter not found")
		}
	}

	model := &MADDPGModel{
		ModelPath:   modelPath,
		NumAgents:   numAgents,
		StateDim:    8,  // From environment observation space
		ActionDim:   4,  // From environment action space
		pythonPath:  pythonPath,
		initialized: false,
	}

	// Initialize model
	if err := model.initialize(); err != nil {
		return nil, err
	}

	return model, nil
}

// initialize loads the MADDPG model
func (m *MADDPGModel) initialize() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Verify model files exist
	for i := 0; i < m.NumAgents; i++ {
		agentPath := filepath.Join(m.ModelPath, fmt.Sprintf("agent_%d.pt", i))
		if _, err := os.Stat(agentPath); os.IsNotExist(err) {
			return fmt.Errorf("agent %d model not found: %s", i, agentPath)
		}
	}

	m.initialized = true
	return nil
}

// Predict runs inference using the trained MADDPG model
func (m *MADDPGModel) Predict(states [][]float64) ([][]float64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.initialized {
		return nil, fmt.Errorf("model not initialized")
	}

	// Create temporary file for states
	statesData, err := json.Marshal(map[string]interface{}{
		"states": states,
	})
	if err != nil {
		return nil, err
	}

	tmpFile, err := os.CreateTemp("", "maddpg_states_*.json")
	if err != nil {
		return nil, err
	}
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.Write(statesData); err != nil {
		return nil, err
	}
	tmpFile.Close()

	// Run Python inference script
	scriptPath := filepath.Join(filepath.Dir(m.ModelPath), "inference.py")
	cmd := exec.Command(m.pythonPath, scriptPath, m.ModelPath, tmpFile.Name())

	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("inference failed: %v", err)
	}

	// Parse output
	var result struct {
		Actions [][]float64 `json:"actions"`
	}
	if err := json.Unmarshal(output, &result); err != nil {
		return nil, err
	}

	return result.Actions, nil
}

// ResourceAllocator uses MADDPG for intelligent resource allocation
type ResourceAllocator struct {
	model           *MADDPGModel
	nodes           []*Node
	mu              sync.RWMutex
	allocationHistory []Allocation
	metrics         AllocationMetrics
}

// AllocationMetrics tracks allocation performance
type AllocationMetrics struct {
	TotalAllocations  int       `json:"total_allocations"`
	SuccessfulAllocs  int       `json:"successful_allocs"`
	FailedAllocs      int       `json:"failed_allocs"`
	SLAViolations     int       `json:"sla_violations"`
	AvgUtilization    float64   `json:"avg_utilization"`
	LastUpdate        time.Time `json:"last_update"`
}

// NewResourceAllocator creates a new MADDPG-based resource allocator
func NewResourceAllocator(modelPath string, nodes []*Node) (*ResourceAllocator, error) {
	model, err := NewMADDPGModel(modelPath, len(nodes))
	if err != nil {
		return nil, err
	}

	return &ResourceAllocator{
		model:             model,
		nodes:             nodes,
		allocationHistory: make([]Allocation, 0),
		metrics: AllocationMetrics{
			LastUpdate: time.Now(),
		},
	}, nil
}

// AllocateResources allocates workloads to nodes using trained MADDPG model
func (a *ResourceAllocator) AllocateResources(workloads []Workload) ([]Allocation, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if len(workloads) == 0 {
		return []Allocation{}, nil
	}

	// Get current node states
	states := make([][]float64, len(a.nodes))
	for i, node := range a.nodes {
		states[i] = node.GetObservation()
	}

	// Get actions from MADDPG model
	actions, err := a.model.Predict(states)
	if err != nil {
		return nil, fmt.Errorf("model prediction failed: %v", err)
	}

	// Allocate workloads based on actions
	allocations := make([]Allocation, 0)

	for _, workload := range workloads {
		bestNode := -1
		bestScore := -1.0

		// Find best node for this workload
		for nodeIdx, node := range a.nodes {
			action := actions[nodeIdx]

			// Calculate available resources based on action
			cpuAvail := action[0] * (node.CPUCapacity - node.CPUUsage)
			memAvail := action[1] * (node.MemoryCapacity - node.MemoryUsage)
			bwAvail := action[2] * (node.BandwidthCapacity - node.BandwidthUsage)
			storageAvail := action[3] * (node.StorageCapacity - node.StorageUsage)

			// Check if node can satisfy workload
			canAllocate := cpuAvail >= workload.CPURequirement &&
				memAvail >= workload.MemoryRequirement &&
				bwAvail >= workload.BandwidthRequirement &&
				storageAvail >= workload.StorageRequirement

			if canAllocate {
				// Score based on resource efficiency (lower waste is better)
				wasteScore := (cpuAvail - workload.CPURequirement) / max(node.CPUCapacity, 1e-6) +
					(memAvail - workload.MemoryRequirement) / max(node.MemoryCapacity, 1e-6) +
					(bwAvail - workload.BandwidthRequirement) / max(node.BandwidthCapacity, 1e-6) +
					(storageAvail - workload.StorageRequirement) / max(node.StorageCapacity, 1e-6)
				wasteScore /= 4.0

				// Lower waste is better
				score := 1.0 - wasteScore

				if bestNode == -1 || score > bestScore {
					bestNode = nodeIdx
					bestScore = score
				}
			}
		}

		// Allocate to best node
		if bestNode != -1 {
			node := a.nodes[bestNode]

			// Update node usage
			node.CPUUsage += workload.CPURequirement
			node.MemoryUsage += workload.MemoryRequirement
			node.BandwidthUsage += workload.BandwidthRequirement
			node.StorageUsage += workload.StorageRequirement

			// Record allocation
			allocation := Allocation{
				WorkloadID:   workload.ID,
				NodeID:       node.ID,
				CPUAlloc:     workload.CPURequirement,
				MemAlloc:     workload.MemoryRequirement,
				BWAlloc:      workload.BandwidthRequirement,
				StorageAlloc: workload.StorageRequirement,
				Timestamp:    time.Now(),
			}
			allocations = append(allocations, allocation)
			a.allocationHistory = append(a.allocationHistory, allocation)

			a.metrics.SuccessfulAllocs++
		} else {
			// Failed to allocate
			a.metrics.FailedAllocs++
			a.metrics.SLAViolations++
		}

		a.metrics.TotalAllocations++
	}

	// Update metrics
	a.updateMetrics()

	return allocations, nil
}

// updateMetrics calculates current allocation metrics
func (a *ResourceAllocator) updateMetrics() {
	// Calculate average utilization
	totalUtil := 0.0
	for _, node := range a.nodes {
		cpuUtil := node.CPUUsage / max(node.CPUCapacity, 1e-6)
		totalUtil += cpuUtil
	}
	a.metrics.AvgUtilization = totalUtil / float64(len(a.nodes))
	a.metrics.LastUpdate = time.Now()
}

// GetMetrics returns current allocation metrics
func (a *ResourceAllocator) GetMetrics() AllocationMetrics {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.metrics
}

// GetAllocationHistory returns allocation history
func (a *ResourceAllocator) GetAllocationHistory(limit int) []Allocation {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if limit <= 0 || limit > len(a.allocationHistory) {
		limit = len(a.allocationHistory)
	}

	history := make([]Allocation, limit)
	start := len(a.allocationHistory) - limit
	copy(history, a.allocationHistory[start:])

	return history
}

// PerformanceReport generates performance comparison report
func (a *ResourceAllocator) PerformanceReport() map[string]interface{} {
	a.mu.RLock()
	defer a.mu.RUnlock()

	successRate := 0.0
	if a.metrics.TotalAllocations > 0 {
		successRate = float64(a.metrics.SuccessfulAllocs) / float64(a.metrics.TotalAllocations)
	}

	slaViolationRate := 0.0
	if a.metrics.TotalAllocations > 0 {
		slaViolationRate = float64(a.metrics.SLAViolations) / float64(a.metrics.TotalAllocations)
	}

	return map[string]interface{}{
		"total_allocations":   a.metrics.TotalAllocations,
		"successful_allocs":   a.metrics.SuccessfulAllocs,
		"failed_allocs":       a.metrics.FailedAllocs,
		"success_rate":        successRate,
		"sla_violations":      a.metrics.SLAViolations,
		"sla_violation_rate":  slaViolationRate,
		"avg_utilization":     a.metrics.AvgUtilization,
		"last_update":         a.metrics.LastUpdate,
		"num_nodes":           len(a.nodes),
		"model_path":          a.model.ModelPath,
	}
}

// Helper function
func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}
