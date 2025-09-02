package graphql

import (
	"context"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage/tiering"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// Resolver is the root GraphQL resolver
type Resolver struct {
	vmManager      *vm.VMManager
	storageManager *tiering.TierManager
	subscriptions  *SubscriptionManager
}

// NewResolver creates a new GraphQL resolver
func NewResolver(vmManager *vm.VMManager, storageManager *tiering.TierManager) *Resolver {
	return &Resolver{
		vmManager:      vmManager,
		storageManager: storageManager,
		subscriptions:  NewSubscriptionManager(),
	}
}

// VM Resolvers

// VMs returns all VMs with optional pagination
func (r *Resolver) VMs(ctx context.Context, args struct{ Pagination *PaginationInput }) ([]*VM, error) {
	vmsMap := r.vmManager.ListVMs()
	
	// Convert map to slice
	vms := make([]*vm.VM, 0, len(vmsMap))
	for _, vmInstance := range vmsMap {
		vms = append(vms, vmInstance)
	}
	
	// Apply pagination if provided
	if args.Pagination != nil {
		start := args.Pagination.Page * args.Pagination.PageSize
		end := start + args.Pagination.PageSize
		
		if start > len(vms) {
			return []*VM{}, nil
		}
		if end > len(vms) {
			end = len(vms)
		}
		
		vms = vms[start:end]
	}
	
	// Convert to GraphQL types
	result := make([]*VM, len(vms))
	for i, vm := range vms {
		result[i] = convertVM(vm)
	}
	
	return result, nil
}

// VM returns a specific VM by ID
func (r *Resolver) VM(ctx context.Context, args struct{ ID string }) (*VM, error) {
	vm, err := r.vmManager.GetVM(args.ID)
	if err != nil {
		return nil, err
	}
	
	return convertVM(vm), nil
}

// CreateVM creates a new VM
func (r *Resolver) CreateVM(ctx context.Context, args struct{ Input CreateVMInput }) (*VM, error) {
	createReq := vm.CreateVMRequest{
		Name: args.Input.Name,
		Spec: vm.VMConfig{
			Name:       args.Input.Name,
			CPUShares:  args.Input.CPU,
			MemoryMB:   args.Input.Memory,
			DiskSizeGB: args.Input.Disk,
			Command:    "/bin/bash",
		},
		Tags: make(map[string]string),
	}
	
	vmInstance, err := r.vmManager.CreateVM(ctx, createReq)
	if err != nil {
		return nil, err
	}
	
	result := convertVM(vmInstance)
	
	// Publish event
	r.subscriptions.PublishVMStateChange(result)
	
	return result, nil
}

// UpdateVM updates a VM configuration
func (r *Resolver) UpdateVM(ctx context.Context, args struct {
	ID    string
	Input UpdateVMInput
}) (*VM, error) {
	// TODO: Implement UpdateVM method on VMManager
	// For now, return placeholder response
	vmInstance, err := r.vmManager.GetVM(args.ID)
	if err != nil {
		return nil, err
	}
	
	result := convertVM(vmInstance)
	
	// Publish event
	r.subscriptions.PublishVMStateChange(result)
	
	return result, nil
}

// DeleteVM deletes a VM
func (r *Resolver) DeleteVM(ctx context.Context, args struct{ ID string }) (bool, error) {
	err := r.vmManager.DeleteVM(ctx, args.ID)
	if err != nil {
		return false, err
	}
	
	return true, nil
}

// StartVM starts a VM
func (r *Resolver) StartVM(ctx context.Context, args struct{ ID string }) (*VM, error) {
	err := r.vmManager.StartVM(ctx, args.ID)
	if err != nil {
		return nil, err
	}
	
	vm, err := r.vmManager.GetVM(args.ID)
	if err != nil {
		return nil, err
	}
	
	result := convertVM(vm)
	
	// Publish event
	r.subscriptions.PublishVMStateChange(result)
	
	return result, nil
}

// StopVM stops a VM
func (r *Resolver) StopVM(ctx context.Context, args struct{ ID string }) (*VM, error) {
	err := r.vmManager.StopVM(ctx, args.ID)
	if err != nil {
		return nil, err
	}
	
	vm, err := r.vmManager.GetVM(args.ID)
	if err != nil {
		return nil, err
	}
	
	result := convertVM(vm)
	
	// Publish event
	r.subscriptions.PublishVMStateChange(result)
	
	return result, nil
}

// MigrateVM migrates a VM to another host
func (r *Resolver) MigrateVM(ctx context.Context, args struct {
	ID    string
	Input MigrateVMInput
}) (*Migration, error) {
	migrationID := fmt.Sprintf("migration-%d", time.Now().Unix())
	
	migration := &Migration{
		ID:         migrationID,
		VMID:       args.ID,
		SourceHost: "current-host", // Get from VM
		TargetHost: args.Input.TargetHost,
		Type:       "LIVE",
		Status:     "IN_PROGRESS",
		Progress:   0.0,
		StartedAt:  time.Now(),
	}
	
	// Start migration in background
	go func() {
		// TODO: Implement MigrateVM method on VMManager
		// For now, simulate migration completion
		time.Sleep(1 * time.Second)
		migration.Status = "COMPLETED"
		migration.CompletedAt = &time.Time{}
		*migration.CompletedAt = time.Now()
		migration.Progress = 100.0
		
		// Publish completion
		r.subscriptions.PublishMigrationProgress(migration)
	}()
	
	// Publish start
	r.subscriptions.PublishMigrationProgress(migration)
	
	return migration, nil
}

// Storage Resolvers

// Volumes returns all storage volumes
func (r *Resolver) Volumes(ctx context.Context, args struct{ Pagination *PaginationInput }) ([]*StorageVolume, error) {
	// TODO: Implement volume listing through TierManager
	volumes := []map[string]interface{}{}
	
	// Apply pagination if provided
	if args.Pagination != nil {
		start := args.Pagination.Page * args.Pagination.PageSize
		end := start + args.Pagination.PageSize
		
		if start > len(volumes) {
			return []*StorageVolume{}, nil
		}
		if end > len(volumes) {
			end = len(volumes)
		}
		
		volumes = volumes[start:end]
	}
	
	// Convert to GraphQL types
	result := make([]*StorageVolume, len(volumes))
	for i, vol := range volumes {
		result[i] = convertVolume(vol)
	}
	
	return result, nil
}

// CreateVolume creates a new storage volume
func (r *Resolver) CreateVolume(ctx context.Context, args struct{ Input CreateVolumeInput }) (*StorageVolume, error) {
	// TODO: Implement volume creation through TierManager
	volume := map[string]interface{}{
		"id":   fmt.Sprintf("vol-%d", time.Now().Unix()),
		"name": args.Input.Name,
		"size": args.Input.Size,
		"tier": args.Input.Tier,
	}
	
	return convertVolume(volume), nil
}

// ChangeVolumeTier changes the storage tier of a volume
func (r *Resolver) ChangeVolumeTier(ctx context.Context, args struct {
	ID   string
	Tier string
}) (*StorageVolume, error) {
	// TODO: Implement tier migration through TierManager
	volume := map[string]interface{}{
		"id":   args.ID,
		"name": "sample-volume",
		"size": 100,
		"tier": args.Tier,
	}
	
	return convertVolume(volume), nil
}

// Cluster Resolvers

// Nodes returns all cluster nodes
func (r *Resolver) Nodes(ctx context.Context) ([]*Node, error) {
	// Implementation would fetch from cluster membership
	nodes := []*Node{
		{
			ID:       "node-1",
			Address:  "192.168.1.100",
			State:    "HEALTHY",
			IsLeader: true,
			CPU:      8,
			Memory:   32768,
			Disk:     1048576,
			VMCount:  5,
		},
		{
			ID:       "node-2",
			Address:  "192.168.1.101",
			State:    "HEALTHY",
			IsLeader: false,
			CPU:      8,
			Memory:   32768,
			Disk:     1048576,
			VMCount:  3,
		},
	}
	
	return nodes, nil
}

// ClusterStatus returns the current cluster status
func (r *Resolver) ClusterStatus(ctx context.Context) (*ClusterStatus, error) {
	status := &ClusterStatus{
		Healthy:      true,
		TotalNodes:   3,
		HealthyNodes: 2,
		HasQuorum:    true,
	}
	
	return status, nil
}

// Monitoring Resolvers

// SystemMetrics returns system metrics
func (r *Resolver) SystemMetrics(ctx context.Context, args struct{ Range *TimeRangeInput }) ([]*SystemMetrics, error) {
	// Generate sample metrics
	metrics := []*SystemMetrics{
		{
			CPU: &CPUMetrics{
				Usage:   65.5,
				Cores:   8,
				LoadAvg: []float64{2.5, 2.8, 3.1},
			},
			Memory: &MemoryMetrics{
				Total:     32768,
				Used:      20000,
				Free:      12768,
				Cached:    5000,
				Available: 17768,
			},
			Disk: &DiskMetrics{
				Total: 1048576,
				Used:  524288,
				Free:  524288,
			},
			Network: &NetworkMetrics{
				BytesIn:    1048576,
				BytesOut:   524288,
				PacketsIn:  10000,
				PacketsOut: 5000,
			},
			Timestamp: time.Now(),
		},
	}
	
	return metrics, nil
}

// Alerts returns system alerts
func (r *Resolver) Alerts(ctx context.Context, args struct {
	Severity     *string
	Acknowledged *bool
}) ([]*Alert, error) {
	alerts := []*Alert{
		{
			ID:           "alert-1",
			Severity:     "WARNING",
			Message:      "High memory usage on node-3",
			Source:       "node-3",
			Timestamp:    time.Now().Add(-10 * time.Minute),
			Acknowledged: false,
		},
	}
	
	// Filter by severity if specified
	if args.Severity != nil {
		filtered := []*Alert{}
		for _, alert := range alerts {
			if alert.Severity == *args.Severity {
				filtered = append(filtered, alert)
			}
		}
		alerts = filtered
	}
	
	// Filter by acknowledged status if specified
	if args.Acknowledged != nil {
		filtered := []*Alert{}
		for _, alert := range alerts {
			if alert.Acknowledged == *args.Acknowledged {
				filtered = append(filtered, alert)
			}
		}
		alerts = filtered
	}
	
	return alerts, nil
}

// Events returns system events
func (r *Resolver) Events(ctx context.Context, args struct{ Limit *int }) ([]*Event, error) {
	limit := 100
	if args.Limit != nil {
		limit = *args.Limit
	}
	
	events := []*Event{
		{
			ID:        "event-1",
			Type:      "vm.created",
			Message:   "VM web-server-1 created",
			Source:    "api",
			Timestamp: time.Now().Add(-5 * time.Minute),
		},
		{
			ID:        "event-2",
			Type:      "vm.migrated",
			Message:   "VM database-1 migrated",
			Source:    "scheduler",
			Timestamp: time.Now().Add(-15 * time.Minute),
		},
	}
	
	if len(events) > limit {
		events = events[:limit]
	}
	
	return events, nil
}

// Subscription Resolvers

// VMStateChanged subscription for VM state changes
func (r *Resolver) VMStateChanged(ctx context.Context, args struct{ VMID *string }) (<-chan *VM, error) {
	ch := make(chan *VM)
	
	r.subscriptions.SubscribeVMStateChange(func(vm *VM) {
		if args.VMID == nil || vm.ID == *args.VMID {
			select {
			case ch <- vm:
			case <-ctx.Done():
			}
		}
	})
	
	go func() {
		<-ctx.Done()
		close(ch)
	}()
	
	return ch, nil
}

// NewAlert subscription for new alerts
func (r *Resolver) NewAlert(ctx context.Context) (<-chan *Alert, error) {
	ch := make(chan *Alert)
	
	r.subscriptions.SubscribeNewAlert(func(alert *Alert) {
		select {
		case ch <- alert:
		case <-ctx.Done():
		}
	})
	
	go func() {
		<-ctx.Done()
		close(ch)
	}()
	
	return ch, nil
}

// Helper functions to convert between internal and GraphQL types

func convertVM(vmInstance *vm.VM) *VM {
	info := vmInstance.GetInfo()
	return &VM{
		ID:        vmInstance.ID(),
		Name:      vmInstance.Name(),
		State:     string(vmInstance.State()),
		CPU:       info.CPUShares,
		Memory:    info.MemoryMB,
		Disk:      0,    // Not available in VMInfo
		Image:     "",   // Not available from VMInfo
		Host:      "",   // NodeID not available in VMInfo
		IPAddress: "",   // Not available from VMInfo
		CreatedAt: info.CreatedAt,
		UpdatedAt: info.CreatedAt, // Use CreatedAt as placeholder
	}
}

func convertVolume(vol map[string]interface{}) *StorageVolume {
	return &StorageVolume{
		ID:        getString(vol, "id"),
		Name:      getString(vol, "name"),
		Size:      getInt(vol, "size"),
		Tier:      getString(vol, "tier"),
		CreatedAt: time.Now(), // placeholder
		UpdatedAt: time.Now(), // placeholder
	}
}

// Helper functions for map conversion
func getString(m map[string]interface{}, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

func getInt(m map[string]interface{}, key string) int {
	if v, ok := m[key].(int); ok {
		return v
	}
	if v, ok := m[key].(int64); ok {
		return int(v)
	}
	return 0
}