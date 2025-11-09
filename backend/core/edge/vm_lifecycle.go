package edge

import (
	"context"
	"fmt"
	"time"
)

// EdgeVMLifecycle manages VM lifecycle at the edge
type EdgeVMLifecycle struct {
	config      *EdgeConfig
	discovery   *EdgeDiscovery
	placement   *PlacementEngine
	coordinator *EdgeCloudCoordinator
	vms         map[string]*EdgeVM
}

// EdgeVM represents a VM deployed at the edge
type EdgeVM struct {
	VMID         string        `json:"vm_id"`
	Name         string        `json:"name"`
	EdgeNodeID   string        `json:"edge_node_id"`
	State        VMState       `json:"state"`
	Resources    VMResources   `json:"resources"`
	Image        string        `json:"image"`
	StartTime    time.Time     `json:"start_time"`
	ProvisionTime time.Duration `json:"provision_time"`
	LastStateChange time.Time  `json:"last_state_change"`
	Metadata     map[string]string `json:"metadata"`
}

// VMState represents VM state
type VMState string

const (
	VMStateProvisioning VMState = "provisioning"
	VMStateRunning      VMState = "running"
	VMStateSuspended    VMState = "suspended"
	VMStateMigrating    VMState = "migrating"
	VMStateTerminating  VMState = "terminating"
	VMStateTerminated   VMState = "terminated"
	VMStateFailed       VMState = "failed"
)

// VMResources represents VM resource allocation
type VMResources struct {
	CPUCores      int   `json:"cpu_cores"`
	MemoryMB      int64 `json:"memory_mb"`
	StorageGB     int64 `json:"storage_gb"`
	BandwidthMbps int   `json:"bandwidth_mbps"`
	GPUCount      int   `json:"gpu_count"`
}

// NewEdgeVMLifecycle creates a new VM lifecycle manager
func NewEdgeVMLifecycle(config *EdgeConfig, discovery *EdgeDiscovery, placement *PlacementEngine, coordinator *EdgeCloudCoordinator) *EdgeVMLifecycle {
	return &EdgeVMLifecycle{
		config:      config,
		discovery:   discovery,
		placement:   placement,
		coordinator: coordinator,
		vms:         make(map[string]*EdgeVM),
	}
}

// ProvisionVM provisions a new VM at the edge with rapid provisioning
func (evl *EdgeVMLifecycle) ProvisionVM(ctx context.Context, req *ProvisionRequest) (*EdgeVM, error) {
	startTime := time.Now()

	// Find optimal placement
	placementReq := &PlacementRequest{
		VMID:         req.VMID,
		UserLocation: req.UserLocation,
		Requirements: req.Requirements,
		Constraints:  req.Constraints,
	}

	decision, err := evl.placement.PlaceVM(ctx, placementReq)
	if err != nil {
		return nil, err
	}

	// Create VM object
	vm := &EdgeVM{
		VMID:            req.VMID,
		Name:            req.Name,
		EdgeNodeID:      decision.EdgeNodeID,
		State:           VMStateProvisioning,
		Resources:       VMResources{
			CPUCores:      req.Requirements.CPUCores,
			MemoryMB:      req.Requirements.MemoryMB,
			StorageGB:     req.Requirements.StorageGB,
			BandwidthMbps: req.Requirements.BandwidthMbps,
		},
		Image:           req.Image,
		StartTime:       startTime,
		LastStateChange: startTime,
		Metadata:        req.Metadata,
	}

	// Provision on edge node
	if err := evl.provisionOnNode(ctx, vm); err != nil {
		vm.State = VMStateFailed
		return nil, err
	}

	// Update provision time
	vm.ProvisionTime = time.Since(startTime)
	vm.State = VMStateRunning
	vm.LastStateChange = time.Now()

	evl.vms[vm.VMID] = vm

	// Check if provision time meets target
	if vm.ProvisionTime > evl.config.TargetProvisionTime {
		// Log warning
	}

	return vm, nil
}

// ProvisionRequest represents a VM provision request
type ProvisionRequest struct {
	VMID          string                   `json:"vm_id"`
	Name          string                   `json:"name"`
	Image         string                   `json:"image"`
	Requirements  PlacementRequirements    `json:"requirements"`
	Constraints   PlacementConstraints     `json:"constraints"`
	UserLocation  *GeoLocation             `json:"user_location"`
	Metadata      map[string]string        `json:"metadata"`
}

// provisionOnNode provisions VM on the selected edge node
func (evl *EdgeVMLifecycle) provisionOnNode(ctx context.Context, vm *EdgeVM) error {
	// In production, this would:
	// 1. Pull VM image (if not cached)
	// 2. Allocate resources (CPU, memory, storage)
	// 3. Setup networking (IP allocation, routing)
	// 4. Configure security (firewall, SELinux)
	// 5. Start VM
	// 6. Wait for healthy state

	// Simulate rapid provisioning
	time.Sleep(2 * time.Second)

	return nil
}

// SuspendVM suspends a running VM for resource management
func (evl *EdgeVMLifecycle) SuspendVM(ctx context.Context, vmID string) error {
	vm, exists := evl.vms[vmID]
	if !exists {
		return fmt.Errorf("VM not found: %s", vmID)
	}

	if vm.State != VMStateRunning {
		return fmt.Errorf("VM not in running state: %s", vm.State)
	}

	// Suspend VM
	// In production:
	// 1. Pause VM execution
	// 2. Snapshot memory to disk
	// 3. Free CPU and memory resources
	// 4. Keep storage allocated

	vm.State = VMStateSuspended
	vm.LastStateChange = time.Now()

	return nil
}

// ResumeVM resumes a suspended VM
func (evl *EdgeVMLifecycle) ResumeVM(ctx context.Context, vmID string) error {
	vm, exists := evl.vms[vmID]
	if !exists {
		return fmt.Errorf("VM not found: %s", vmID)
	}

	if vm.State != VMStateSuspended {
		return fmt.Errorf("VM not in suspended state: %s", vm.State)
	}

	// Resume VM
	// In production:
	// 1. Restore memory from snapshot
	// 2. Re-allocate CPU and memory
	// 3. Resume execution
	// 4. Restore network state

	vm.State = VMStateRunning
	vm.LastStateChange = time.Now()

	return nil
}

// MigrateVM migrates VM between edge and cloud
func (evl *EdgeVMLifecycle) MigrateVM(ctx context.Context, vmID string, targetNodeID string) error {
	vm, exists := evl.vms[vmID]
	if !exists {
		return fmt.Errorf("VM not found: %s", vmID)
	}

	if vm.State != VMStateRunning && vm.State != VMStateSuspended {
		return fmt.Errorf("VM not in migratable state: %s", vm.State)
	}

	oldState := vm.State
	vm.State = VMStateMigrating
	vm.LastStateChange = time.Now()

	// Create migration request
	migReq := &MigrationRequest{
		VMID:          vmID,
		SourceNodeID:  vm.EdgeNodeID,
		TargetNodeID:  targetNodeID,
		MigrationType: MigrationTypeLive,
		MaxDowntime:   evl.config.TargetMigrationTime,
		Priority:      5,
		Reason:        "user_initiated",
	}

	// Execute migration
	status, err := evl.coordinator.MigrateVM(ctx, migReq)
	if err != nil {
		vm.State = oldState
		vm.LastStateChange = time.Now()
		return err
	}

	// Wait for migration to complete
	for status.State == MigrationStateRunning || status.State == MigrationStatePending {
		time.Sleep(100 * time.Millisecond)
		status, _ = evl.coordinator.GetMigrationStatus(vmID)
	}

	if status.State == MigrationStateCompleted {
		vm.EdgeNodeID = targetNodeID
		vm.State = VMStateRunning
		vm.LastStateChange = time.Now()
		return nil
	}

	vm.State = oldState
	vm.LastStateChange = time.Now()
	return ErrMigrationFailed
}

// TerminateVM terminates a VM
func (evl *EdgeVMLifecycle) TerminateVM(ctx context.Context, vmID string) error {
	vm, exists := evl.vms[vmID]
	if !exists {
		return fmt.Errorf("VM not found: %s", vmID)
	}

	vm.State = VMStateTerminating
	vm.LastStateChange = time.Now()

	// Terminate VM
	// In production:
	// 1. Gracefully shutdown VM
	// 2. Release resources
	// 3. Cleanup networking
	// 4. Delete storage (if not persistent)
	// 5. Update edge node state

	vm.State = VMStateTerminated
	vm.LastStateChange = time.Now()

	delete(evl.vms, vmID)

	return nil
}

// AutoScale implements auto-scaling at the edge
func (evl *EdgeVMLifecycle) AutoScale(ctx context.Context) error {
	// Get all edge nodes
	nodes := evl.discovery.GetHealthyNodes()

	for _, node := range nodes {
		// Check if node is overloaded
		if node.Resources.UtilizationPercent > 90.0 {
			// Scale out: Provision additional capacity
			// Or migrate some VMs to less loaded nodes
		} else if node.Resources.UtilizationPercent < 20.0 {
			// Scale in: Consider consolidating VMs
			// Suspend idle VMs
		}
	}

	return nil
}

// GracefulDegradation implements graceful degradation
func (evl *EdgeVMLifecycle) GracefulDegradation(ctx context.Context, nodeID string) error {
	// When edge node fails or degrades:
	// 1. Identify affected VMs
	// 2. Prioritize critical VMs
	// 3. Migrate high-priority VMs to cloud
	// 4. Suspend low-priority VMs
	// 5. Queue deferred operations

	return nil
}

// GetVMStatus retrieves VM status
func (evl *EdgeVMLifecycle) GetVMStatus(vmID string) (*VMStatus, error) {
	vm, exists := evl.vms[vmID]
	if !exists {
		return nil, fmt.Errorf("VM not found: %s", vmID)
	}

	return &VMStatus{
		VMID:         vm.VMID,
		State:        vm.State,
		EdgeNodeID:   vm.EdgeNodeID,
		UptimeSeconds: int64(time.Since(vm.StartTime).Seconds()),
		CPUUsage:     45.2, // Would come from actual metrics
		MemoryUsage:  62.8,
		NetworkTxMB:  1024,
		NetworkRxMB:  512,
		Timestamp:    time.Now(),
	}, nil
}

// VMStatus represents VM runtime status
type VMStatus struct {
	VMID          string    `json:"vm_id"`
	State         VMState   `json:"state"`
	EdgeNodeID    string    `json:"edge_node_id"`
	UptimeSeconds int64     `json:"uptime_seconds"`
	CPUUsage      float64   `json:"cpu_usage"`
	MemoryUsage   float64   `json:"memory_usage"`
	NetworkTxMB   int64     `json:"network_tx_mb"`
	NetworkRxMB   int64     `json:"network_rx_mb"`
	Timestamp     time.Time `json:"timestamp"`
}

// ListVMs lists all VMs
func (evl *EdgeVMLifecycle) ListVMs() []*EdgeVM {
	vms := make([]*EdgeVM, 0, len(evl.vms))
	for _, vm := range evl.vms {
		vms = append(vms, vm)
	}
	return vms
}

// GetLifecycleMetrics retrieves lifecycle metrics
func (evl *EdgeVMLifecycle) GetLifecycleMetrics() (*LifecycleMetrics, error) {
	metrics := &LifecycleMetrics{
		TotalVMs:       len(evl.vms),
		RunningVMs:     0,
		SuspendedVMs:   0,
		MigratingVMs:   0,
		AvgProvisionTime: 0,
		Timestamp:      time.Now(),
	}

	var totalProvisionTime time.Duration
	for _, vm := range evl.vms {
		switch vm.State {
		case VMStateRunning:
			metrics.RunningVMs++
		case VMStateSuspended:
			metrics.SuspendedVMs++
		case VMStateMigrating:
			metrics.MigratingVMs++
		}
		totalProvisionTime += vm.ProvisionTime
	}

	if metrics.TotalVMs > 0 {
		metrics.AvgProvisionTime = totalProvisionTime / time.Duration(metrics.TotalVMs)
	}

	return metrics, nil
}

// LifecycleMetrics represents VM lifecycle metrics
type LifecycleMetrics struct {
	TotalVMs         int           `json:"total_vms"`
	RunningVMs       int           `json:"running_vms"`
	SuspendedVMs     int           `json:"suspended_vms"`
	MigratingVMs     int           `json:"migrating_vms"`
	AvgProvisionTime time.Duration `json:"avg_provision_time"`
	Timestamp        time.Time     `json:"timestamp"`
}
