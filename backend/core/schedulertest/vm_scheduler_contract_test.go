package schedulertest

import (
	"context"
	"testing"

	corevm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestSchedulerTracksAllocationsAndReleases(t *testing.T) {
	scheduler := corevm.NewVMScheduler(corevm.SchedulerConfig{
		Policy:                 corevm.SchedulerPolicyRoundRobin,
		EnableResourceChecking: true,
		MaxVMsPerNode:          10,
	})

	err := scheduler.RegisterNode(&corevm.NodeResourceInfo{
		NodeID:        "node-1",
		TotalCPU:      8,
		TotalMemoryMB: 16384,
		TotalDiskGB:   200,
		Status:        "available",
	})
	if err != nil {
		t.Fatalf("register node: %v", err)
	}

	vm, err := corevm.NewVM(corevm.VMConfig{
		ID:         "vm-1",
		Name:       "ubuntu-24-04",
		Type:       corevm.VMTypeKVM,
		CPUShares:  1024,
		MemoryMB:   2048,
		DiskSizeGB: 20,
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}
	vm.SetResourceID(vm.ID())

	if err := scheduler.ReserveResources("node-1", vm); err != nil {
		t.Fatalf("reserve resources: %v", err)
	}

	allocations := scheduler.GetActiveAllocations()
	allocation, ok := allocations[vm.ID()]
	if !ok {
		t.Fatalf("expected allocation for %s", vm.ID())
	}
	if allocation.NodeID != "node-1" {
		t.Fatalf("expected node-1 allocation, got %s", allocation.NodeID)
	}
	if allocation.CPUCores != 1 {
		t.Fatalf("expected 1 allocated CPU core, got %d", allocation.CPUCores)
	}
	if allocation.MemoryMB != 2048 {
		t.Fatalf("expected 2048MB allocated memory, got %d", allocation.MemoryMB)
	}
	if allocation.DiskGB != 20 {
		t.Fatalf("expected 20GB allocated disk, got %d", allocation.DiskGB)
	}

	node, err := scheduler.GetNode("node-1")
	if err != nil {
		t.Fatalf("get node: %v", err)
	}
	if node.UsedCPU != 1 || node.UsedMemoryMB != 2048 || node.UsedDiskGB != 20 || node.VMCount != 1 {
		t.Fatalf("unexpected node usage after reserve: %+v", node)
	}

	if err := scheduler.ReleaseResources("node-1", vm); err != nil {
		t.Fatalf("release resources: %v", err)
	}

	if got := scheduler.GetActiveAllocations(); len(got) != 0 {
		t.Fatalf("expected allocations to be empty after release, got %d entries", len(got))
	}

	node, err = scheduler.GetNode("node-1")
	if err != nil {
		t.Fatalf("get node after release: %v", err)
	}
	if node.UsedCPU != 0 || node.UsedMemoryMB != 0 || node.UsedDiskGB != 0 || node.VMCount != 0 {
		t.Fatalf("unexpected node usage after release: %+v", node)
	}
}

func TestSchedulerUsesNormalizedCPUForPlacement(t *testing.T) {
	scheduler := corevm.NewVMScheduler(corevm.SchedulerConfig{
		Policy:                 corevm.SchedulerPolicyRoundRobin,
		EnableResourceChecking: true,
		MaxVMsPerNode:          10,
	})

	for _, node := range []*corevm.NodeResourceInfo{
		{
			NodeID:        "full-node",
			TotalCPU:      1,
			UsedCPU:       1,
			TotalMemoryMB: 4096,
			TotalDiskGB:   100,
			Status:        "available",
		},
		{
			NodeID:        "open-node",
			TotalCPU:      4,
			UsedCPU:       0,
			TotalMemoryMB: 8192,
			TotalDiskGB:   100,
			Status:        "available",
		},
	} {
		if err := scheduler.RegisterNode(node); err != nil {
			t.Fatalf("register node %s: %v", node.NodeID, err)
		}
	}

	vm, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-normalized",
		Name:      "vm-normalized",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  1024,
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}

	nodeID, err := scheduler.ScheduleVM(context.Background(), vm)
	if err != nil {
		t.Fatalf("schedule vm: %v", err)
	}
	if nodeID != "open-node" {
		t.Fatalf("expected open-node, got %s", nodeID)
	}
}

func TestSchedulerRejectsReservationsWhenCapacityIsGone(t *testing.T) {
	scheduler := corevm.NewVMScheduler(corevm.SchedulerConfig{
		Policy:                 corevm.SchedulerPolicyRoundRobin,
		EnableResourceChecking: true,
		MaxVMsPerNode:          10,
	})

	if err := scheduler.RegisterNode(&corevm.NodeResourceInfo{
		NodeID:        "node-1",
		TotalCPU:      1,
		TotalMemoryMB: 1024,
		TotalDiskGB:   50,
		Status:        "available",
	}); err != nil {
		t.Fatalf("register node: %v", err)
	}

	first, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-1",
		Name:      "vm-1",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  1024,
	})
	if err != nil {
		t.Fatalf("new first vm: %v", err)
	}
	first.SetResourceID(first.ID())

	if err := scheduler.ReserveResources("node-1", first); err != nil {
		t.Fatalf("reserve first vm: %v", err)
	}

	second, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-2",
		Name:      "vm-2",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  512,
	})
	if err != nil {
		t.Fatalf("new second vm: %v", err)
	}
	second.SetResourceID(second.ID())

	if err := scheduler.ReserveResources("node-1", second); err == nil {
		t.Fatalf("expected second reservation to fail once node capacity is exhausted")
	}

	if got := scheduler.GetActiveAllocations(); len(got) != 1 {
		t.Fatalf("expected one surviving allocation after failed reserve, got %d", len(got))
	}
}

func TestSchedulerHonorsPreferredNodesWhenAvailable(t *testing.T) {
	scheduler := corevm.NewVMScheduler(corevm.SchedulerConfig{
		Policy:                 corevm.SchedulerPolicyRoundRobin,
		EnableResourceChecking: true,
		MaxVMsPerNode:          10,
	})

	for _, node := range []*corevm.NodeResourceInfo{
		{
			NodeID:        "node-a",
			TotalCPU:      4,
			TotalMemoryMB: 8192,
			TotalDiskGB:   100,
			Status:        "available",
		},
		{
			NodeID:        "node-b",
			TotalCPU:      4,
			TotalMemoryMB: 8192,
			TotalDiskGB:   100,
			Status:        "available",
		},
	} {
		if err := scheduler.RegisterNode(node); err != nil {
			t.Fatalf("register node %s: %v", node.NodeID, err)
		}
	}

	vm, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-preferred",
		Name:      "vm-preferred",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  1024,
		Placement: &corevm.VMPlacementSpec{
			PreferredNodes: []string{"node-b"},
		},
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}

	nodeID, err := scheduler.ScheduleVM(context.Background(), vm)
	if err != nil {
		t.Fatalf("schedule vm: %v", err)
	}
	if nodeID != "node-b" {
		t.Fatalf("expected preferred node node-b, got %s", nodeID)
	}
}

func TestSchedulerFallsBackWhenPreferredNodesUnavailable(t *testing.T) {
	scheduler := corevm.NewVMScheduler(corevm.SchedulerConfig{
		Policy:                 corevm.SchedulerPolicyRoundRobin,
		EnableResourceChecking: true,
		MaxVMsPerNode:          10,
	})

	if err := scheduler.RegisterNode(&corevm.NodeResourceInfo{
		NodeID:        "node-a",
		TotalCPU:      4,
		TotalMemoryMB: 8192,
		TotalDiskGB:   100,
		Status:        "available",
	}); err != nil {
		t.Fatalf("register node: %v", err)
	}

	vm, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-fallback",
		Name:      "vm-fallback",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  1024,
		Placement: &corevm.VMPlacementSpec{
			PreferredNodes: []string{"missing-node"},
		},
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}

	nodeID, err := scheduler.ScheduleVM(context.Background(), vm)
	if err != nil {
		t.Fatalf("schedule vm: %v", err)
	}
	if nodeID != "node-a" {
		t.Fatalf("expected fallback node node-a, got %s", nodeID)
	}
}

func TestSchedulerRejectsExcludedNodes(t *testing.T) {
	scheduler := corevm.NewVMScheduler(corevm.SchedulerConfig{
		Policy:                 corevm.SchedulerPolicyRoundRobin,
		EnableResourceChecking: true,
		MaxVMsPerNode:          10,
	})

	if err := scheduler.RegisterNode(&corevm.NodeResourceInfo{
		NodeID:        "node-a",
		TotalCPU:      4,
		TotalMemoryMB: 8192,
		TotalDiskGB:   100,
		Status:        "available",
	}); err != nil {
		t.Fatalf("register node: %v", err)
	}

	vm, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-excluded",
		Name:      "vm-excluded",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  1024,
		Placement: &corevm.VMPlacementSpec{
			ExcludedNodes: []string{"node-a"},
		},
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}

	if _, err := scheduler.ScheduleVM(context.Background(), vm); err == nil {
		t.Fatalf("expected scheduling to fail when the only available node is excluded")
	}
}

func TestVMManagerAdmissionRejectsAggregateCPUExhaustion(t *testing.T) {
	manager, err := corevm.NewVMManager(corevm.DefaultVMManagerConfig())
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}

	if err := manager.RegisterSchedulerNode(&corevm.NodeResourceInfo{
		NodeID:        "node-a",
		TotalCPU:      1,
		UsedCPU:       1,
		TotalMemoryMB: 4096,
		UsedMemoryMB:  0,
		TotalDiskGB:   100,
		UsedDiskGB:    0,
		Status:        "available",
	}); err != nil {
		t.Fatalf("register scheduler node: %v", err)
	}

	vm, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-cpu-exhausted",
		Name:      "vm-cpu-exhausted",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  512,
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}

	if err := manager.CanAdmitVM(vm); err == nil {
		t.Fatalf("expected admission control to reject CPU-exhausted inventory")
	}
}

func TestVMManagerAdmissionRejectsPlacementExclusions(t *testing.T) {
	manager, err := corevm.NewVMManager(corevm.DefaultVMManagerConfig())
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}

	if err := manager.RegisterSchedulerNode(&corevm.NodeResourceInfo{
		NodeID:        "node-a",
		TotalCPU:      4,
		TotalMemoryMB: 4096,
		TotalDiskGB:   100,
		Status:        "available",
	}); err != nil {
		t.Fatalf("register scheduler node: %v", err)
	}

	vm, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-placement-excluded",
		Name:      "vm-placement-excluded",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  512,
		Placement: &corevm.VMPlacementSpec{
			ExcludedNodes: []string{"node-a"},
		},
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}

	if err := manager.CanAdmitVM(vm); err == nil {
		t.Fatalf("expected admission control to reject when placement exclusions remove all eligible nodes")
	}
}

func TestSchedulerUsesPlacementPolicyMappingPerVM(t *testing.T) {
	scheduler := corevm.NewVMScheduler(corevm.SchedulerConfig{
		Policy:                 corevm.SchedulerPolicyRoundRobin,
		EnableResourceChecking: true,
		MaxVMsPerNode:          10,
	})

	for _, node := range []*corevm.NodeResourceInfo{
		{
			NodeID:             "busy-node",
			TotalCPU:           8,
			UsedCPU:            6,
			TotalMemoryMB:      16384,
			UsedMemoryMB:       12288,
			TotalDiskGB:        200,
			UsedDiskGB:         50,
			CPUUsagePercent:    75,
			MemoryUsagePercent: 75,
			Status:             "available",
		},
		{
			NodeID:             "open-node",
			TotalCPU:           8,
			UsedCPU:            1,
			TotalMemoryMB:      16384,
			UsedMemoryMB:       2048,
			TotalDiskGB:        200,
			UsedDiskGB:         20,
			CPUUsagePercent:    12.5,
			MemoryUsagePercent: 12.5,
			Status:             "available",
		},
	} {
		if err := scheduler.RegisterNode(node); err != nil {
			t.Fatalf("register node %s: %v", node.NodeID, err)
		}
	}

	vm, err := corevm.NewVM(corevm.VMConfig{
		ID:        "vm-performance",
		Name:      "vm-performance",
		Type:      corevm.VMTypeKVM,
		CPUShares: 1024,
		MemoryMB:  1024,
		Placement: &corevm.VMPlacementSpec{
			Policy: "performance",
		},
	})
	if err != nil {
		t.Fatalf("new vm: %v", err)
	}

	nodeID, err := scheduler.ScheduleVM(context.Background(), vm)
	if err != nil {
		t.Fatalf("schedule vm: %v", err)
	}
	if nodeID != "open-node" {
		t.Fatalf("expected performance placement to pick open-node, got %s", nodeID)
	}
}

func TestVMManagerCreateVMRejectsTenantQuotaExceeded(t *testing.T) {
	t.Setenv("NOVACRON_ALLOW_STUB_KVM", "1")

	config := corevm.DefaultVMManagerConfig()
	config.TenantQuota.Default = corevm.TenantQuotaLimits{
		MaxVMs:      1,
		MaxCPUUnits: 1,
		MaxMemoryMB: 1024,
	}
	config.Drivers[corevm.VMTypeKVM] = corevm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"node_id":   "quota-node",
			"qemu_path": "missing-qemu-for-stub-test",
			"vm_path":   t.TempDir(),
		},
	}

	manager, err := corevm.NewVMManager(config)
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}
	defer manager.Stop()

	if err := manager.RegisterSchedulerNode(&corevm.NodeResourceInfo{
		NodeID:        "quota-node",
		TotalCPU:      4,
		TotalMemoryMB: 8192,
		TotalDiskGB:   100,
		Status:        "available",
	}); err != nil {
		t.Fatalf("register scheduler node: %v", err)
	}

	firstReq := corevm.CreateVMRequest{
		Name: "tenant-first",
		Spec: corevm.VMConfig{
			Name:       "tenant-first",
			Type:       corevm.VMTypeKVM,
			OwnerID:    "owner-a",
			TenantID:   "tenant-a",
			CPUShares:  1024,
			MemoryMB:   512,
			DiskSizeGB: 10,
		},
	}
	if _, err := manager.CreateVM(context.Background(), firstReq); err != nil {
		t.Fatalf("create first vm: %v", err)
	}

	secondReq := corevm.CreateVMRequest{
		Name: "tenant-second",
		Spec: corevm.VMConfig{
			Name:       "tenant-second",
			Type:       corevm.VMTypeKVM,
			OwnerID:    "owner-a",
			TenantID:   "tenant-a",
			CPUShares:  1024,
			MemoryMB:   512,
			DiskSizeGB: 10,
		},
	}
	if _, err := manager.CreateVM(context.Background(), secondReq); err == nil {
		t.Fatalf("expected second create to fail on tenant quota")
	} else if vmErr, ok := err.(*corevm.VMError); !ok || vmErr.Code != "QUOTA_EXCEEDED" {
		t.Fatalf("expected QUOTA_EXCEEDED, got %v", err)
	}
}
