package sync

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/yourusername/novacron/backend/core/network/dwcp/sync/crdt"
)

// ClusterMetadata manages conflict-free cluster metadata using CRDTs
type ClusterMetadata struct {
	nodeID      string
	vmStates    *crdt.ORMap     // VM ID -> VM State (OR-Map of LWW-Registers)
	nodeStatus  *crdt.ORMap     // Node ID -> Node Status
	assignments *crdt.ORMap     // VM ID -> Node ID (VM assignments)
	counters    *crdt.PNCounter // Various cluster counters
	resources   *crdt.ORMap     // Resource allocations
	mu          sync.RWMutex
}

// VMState represents the state of a virtual machine
type VMState struct {
	ID           string                 `json:"id"`
	Status       string                 `json:"status"` // running, stopped, migrating, etc.
	NodeID       string                 `json:"node_id"`
	CPUCores     int                    `json:"cpu_cores"`
	MemoryMB     int                    `json:"memory_mb"`
	DiskGB       int                    `json:"disk_gb"`
	IPAddress    string                 `json:"ip_address"`
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// NodeStatus represents the status of a cluster node
type NodeStatus struct {
	ID              string    `json:"id"`
	Region          string    `json:"region"`
	Status          string    `json:"status"` // active, inactive, failed
	CPUUsage        float64   `json:"cpu_usage"`
	MemoryUsage     float64   `json:"memory_usage"`
	DiskUsage       float64   `json:"disk_usage"`
	VMCount         int       `json:"vm_count"`
	LastHeartbeat   time.Time `json:"last_heartbeat"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// NewClusterMetadata creates a new cluster metadata manager
func NewClusterMetadata(nodeID string) *ClusterMetadata {
	return &ClusterMetadata{
		nodeID:      nodeID,
		vmStates:    crdt.NewORMap(nodeID),
		nodeStatus:  crdt.NewORMap(nodeID),
		assignments: crdt.NewORMap(nodeID),
		counters:    crdt.NewPNCounter(nodeID),
		resources:   crdt.NewORMap(nodeID),
	}
}

// UpdateVMState updates the state of a VM using LWW-Register
func (cm *ClusterMetadata) UpdateVMState(vmID string, state VMState) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	state.UpdatedAt = time.Now()

	// Convert state to interface{} for LWW-Register
	stateData, err := json.Marshal(state)
	if err != nil {
		return err
	}

	// Use LWW-Register for VM state
	cm.vmStates.SetLWW(vmID, string(stateData))

	return nil
}

// GetVMState retrieves the state of a VM
func (cm *ClusterMetadata) GetVMState(vmID string) (*VMState, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	value, exists := cm.vmStates.GetLWW(vmID)
	if !exists {
		return nil, &MetadataError{Message: "VM not found"}
	}

	stateData, ok := value.(string)
	if !ok {
		return nil, &MetadataError{Message: "invalid VM state format"}
	}

	var state VMState
	if err := json.Unmarshal([]byte(stateData), &state); err != nil {
		return nil, err
	}

	return &state, nil
}

// ListVMs returns all VM states
func (cm *ClusterMetadata) ListVMs() ([]VMState, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	vmIDs := cm.vmStates.Keys()
	vms := make([]VMState, 0, len(vmIDs))

	for _, vmID := range vmIDs {
		value, exists := cm.vmStates.GetLWW(vmID)
		if !exists {
			continue
		}

		stateData, ok := value.(string)
		if !ok {
			continue
		}

		var state VMState
		if err := json.Unmarshal([]byte(stateData), &state); err != nil {
			continue
		}

		vms = append(vms, state)
	}

	return vms, nil
}

// DeleteVM removes a VM from the metadata
func (cm *ClusterMetadata) DeleteVM(vmID string) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.vmStates.Remove(vmID)
	cm.assignments.Remove(vmID)
}

// UpdateNodeStatus updates the status of a node
func (cm *ClusterMetadata) UpdateNodeStatus(nodeID string, status NodeStatus) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	status.UpdatedAt = time.Now()

	statusData, err := json.Marshal(status)
	if err != nil {
		return err
	}

	cm.nodeStatus.SetLWW(nodeID, string(statusData))

	return nil
}

// GetNodeStatus retrieves the status of a node
func (cm *ClusterMetadata) GetNodeStatus(nodeID string) (*NodeStatus, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	value, exists := cm.nodeStatus.GetLWW(nodeID)
	if !exists {
		return nil, &MetadataError{Message: "node not found"}
	}

	statusData, ok := value.(string)
	if !ok {
		return nil, &MetadataError{Message: "invalid node status format"}
	}

	var status NodeStatus
	if err := json.Unmarshal([]byte(statusData), &status); err != nil {
		return nil, err
	}

	return &status, nil
}

// ListNodes returns all node statuses
func (cm *ClusterMetadata) ListNodes() ([]NodeStatus, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	nodeIDs := cm.nodeStatus.Keys()
	nodes := make([]NodeStatus, 0, len(nodeIDs))

	for _, nodeID := range nodeIDs {
		value, exists := cm.nodeStatus.GetLWW(nodeID)
		if !exists {
			continue
		}

		statusData, ok := value.(string)
		if !ok {
			continue
		}

		var status NodeStatus
		if err := json.Unmarshal([]byte(statusData), &status); err != nil {
			continue
		}

		nodes = append(nodes, status)
	}

	return nodes, nil
}

// AssignVM assigns a VM to a node
func (cm *ClusterMetadata) AssignVM(vmID, nodeID string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.assignments.SetLWW(vmID, nodeID)

	return nil
}

// GetVMAssignment retrieves the node assignment for a VM
func (cm *ClusterMetadata) GetVMAssignment(vmID string) (string, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	value, exists := cm.assignments.GetLWW(vmID)
	if !exists {
		return "", &MetadataError{Message: "VM not assigned"}
	}

	nodeID, ok := value.(string)
	if !ok {
		return "", &MetadataError{Message: "invalid assignment format"}
	}

	return nodeID, nil
}

// IncrementCounter increments a cluster counter
func (cm *ClusterMetadata) IncrementCounter(delta int64) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	cm.counters.Increment(delta)
}

// GetCounterValue returns the current counter value
func (cm *ClusterMetadata) GetCounterValue() int64 {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	value := cm.counters.Value()
	return value.(int64)
}

// AllocateResource allocates a resource using OR-Map
func (cm *ClusterMetadata) AllocateResource(resourceID string, allocation interface{}) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	allocationData, err := json.Marshal(allocation)
	if err != nil {
		return err
	}

	cm.resources.SetLWW(resourceID, string(allocationData))

	return nil
}

// GetResourceAllocation retrieves a resource allocation
func (cm *ClusterMetadata) GetResourceAllocation(resourceID string) (interface{}, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	value, exists := cm.resources.GetLWW(resourceID)
	if !exists {
		return nil, &MetadataError{Message: "resource not found"}
	}

	return value, nil
}

// Merge merges cluster metadata from another node
func (cm *ClusterMetadata) Merge(other *ClusterMetadata) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Merge VM states
	if err := cm.vmStates.Merge(other.vmStates); err != nil {
		return err
	}

	// Merge node statuses
	if err := cm.nodeStatus.Merge(other.nodeStatus); err != nil {
		return err
	}

	// Merge assignments
	if err := cm.assignments.Merge(other.assignments); err != nil {
		return err
	}

	// Merge counters
	if err := cm.counters.Merge(other.counters); err != nil {
		return err
	}

	// Merge resources
	if err := cm.resources.Merge(other.resources); err != nil {
		return err
	}

	return nil
}

// Marshal serializes the cluster metadata
func (cm *ClusterMetadata) Marshal() ([]byte, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	vmStatesData, err := cm.vmStates.Marshal()
	if err != nil {
		return nil, err
	}

	nodeStatusData, err := cm.nodeStatus.Marshal()
	if err != nil {
		return nil, err
	}

	assignmentsData, err := cm.assignments.Marshal()
	if err != nil {
		return nil, err
	}

	countersData, err := cm.counters.Marshal()
	if err != nil {
		return nil, err
	}

	resourcesData, err := cm.resources.Marshal()
	if err != nil {
		return nil, err
	}

	data := struct {
		NodeID      string          `json:"node_id"`
		VMStates    json.RawMessage `json:"vm_states"`
		NodeStatus  json.RawMessage `json:"node_status"`
		Assignments json.RawMessage `json:"assignments"`
		Counters    json.RawMessage `json:"counters"`
		Resources   json.RawMessage `json:"resources"`
	}{
		NodeID:      cm.nodeID,
		VMStates:    vmStatesData,
		NodeStatus:  nodeStatusData,
		Assignments: assignmentsData,
		Counters:    countersData,
		Resources:   resourcesData,
	}

	return json.Marshal(data)
}

// Unmarshal deserializes the cluster metadata
func (cm *ClusterMetadata) Unmarshal(data []byte) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	var parsed struct {
		NodeID      string          `json:"node_id"`
		VMStates    json.RawMessage `json:"vm_states"`
		NodeStatus  json.RawMessage `json:"node_status"`
		Assignments json.RawMessage `json:"assignments"`
		Counters    json.RawMessage `json:"counters"`
		Resources   json.RawMessage `json:"resources"`
	}

	if err := json.Unmarshal(data, &parsed); err != nil {
		return err
	}

	cm.nodeID = parsed.NodeID

	// Unmarshal CRDTs
	cm.vmStates = crdt.NewORMap(cm.nodeID)
	if err := cm.vmStates.Unmarshal(parsed.VMStates); err != nil {
		return err
	}

	cm.nodeStatus = crdt.NewORMap(cm.nodeID)
	if err := cm.nodeStatus.Unmarshal(parsed.NodeStatus); err != nil {
		return err
	}

	cm.assignments = crdt.NewORMap(cm.nodeID)
	if err := cm.assignments.Unmarshal(parsed.Assignments); err != nil {
		return err
	}

	cm.counters = crdt.NewPNCounter(cm.nodeID)
	if err := cm.counters.Unmarshal(parsed.Counters); err != nil {
		return err
	}

	cm.resources = crdt.NewORMap(cm.nodeID)
	if err := cm.resources.Unmarshal(parsed.Resources); err != nil {
		return err
	}

	return nil
}

// GetStats returns cluster statistics
func (cm *ClusterMetadata) GetStats() ClusterStats {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	return ClusterStats{
		TotalVMs:         cm.vmStates.Size(),
		TotalNodes:       cm.nodeStatus.Size(),
		TotalAssignments: cm.assignments.Size(),
		CounterValue:     cm.GetCounterValue(),
		ResourcesCount:   cm.resources.Size(),
	}
}

// ClusterStats represents cluster statistics
type ClusterStats struct {
	TotalVMs         int   `json:"total_vms"`
	TotalNodes       int   `json:"total_nodes"`
	TotalAssignments int   `json:"total_assignments"`
	CounterValue     int64 `json:"counter_value"`
	ResourcesCount   int   `json:"resources_count"`
}

// MetadataError represents a metadata error
type MetadataError struct {
	Message string
}

func (e *MetadataError) Error() string {
	return e.Message
}
