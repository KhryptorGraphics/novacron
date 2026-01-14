package vm

import (
	"context"
	"testing"
	"time"
)

// TestMockHypervisor tests the mock hypervisor implementation
func TestMockHypervisor(t *testing.T) {
	mock := NewMockHypervisor("test-node-1", "test-mock")
	ctx := context.Background()

	t.Run("BasicLifecycle", func(t *testing.T) {
		config := VMConfig{
			ID:       "test-vm-1",
			Name:     "Test VM 1",
			CPUShares: 2,
			MemoryMB: 512,
			Tags:     map[string]string{"test": "mock"},
		}

		// Create VM
		vmID, err := mock.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create mock VM: %v", err)
		}

		if vmID != config.ID {
			t.Errorf("Expected VM ID %s, got %s", config.ID, vmID)
		}

		// Check status
		status, err := mock.GetStatus(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM status: %v", err)
		}

		if status != StateCreated {
			t.Errorf("Expected state %s, got %s", StateCreated, status)
		}

		// Start VM
		if err := mock.Start(ctx, vmID); err != nil {
			t.Errorf("Failed to start mock VM: %v", err)
		}

		// Check running status
		status, err = mock.GetStatus(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM status after start: %v", err)
		}

		if status != StateRunning {
			t.Errorf("Expected state %s, got %s", StateRunning, status)
		}

		// Get VM info
		info, err := mock.GetInfo(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM info: %v", err)
		}

		if info.ID != vmID {
			t.Errorf("Expected VM ID %s, got %s", vmID, info.ID)
		}

		if info.State != StateRunning {
			t.Errorf("Expected state %s, got %s", StateRunning, info.State)
		}

		// Stop VM
		if err := mock.Stop(ctx, vmID); err != nil {
			t.Errorf("Failed to stop mock VM: %v", err)
		}

		// Delete VM
		if err := mock.Delete(ctx, vmID); err != nil {
			t.Errorf("Failed to delete mock VM: %v", err)
		}

		// Verify deletion
		_, err = mock.GetStatus(ctx, vmID)
		if err == nil {
			t.Error("Expected error when getting status of deleted VM")
		}
	})

	t.Run("PauseResume", func(t *testing.T) {
		if !mock.SupportsPause() || !mock.SupportsResume() {
			t.Skip("Mock doesn't support pause/resume")
		}

		config := VMConfig{
			ID:       "test-vm-pause",
			Name:     "Test VM Pause",
			CPUShares: 1,
			MemoryMB: 256,
		}

		vmID, err := mock.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create mock VM: %v", err)
		}
		defer mock.Delete(ctx, vmID)

		// Start VM
		if err := mock.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start mock VM: %v", err)
		}

		// Pause VM
		if err := mock.Pause(ctx, vmID); err != nil {
			t.Errorf("Failed to pause mock VM: %v", err)
		}

		// Check paused status
		status, err := mock.GetStatus(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM status: %v", err)
		}

		if status != StatePaused {
			t.Errorf("Expected state %s, got %s", StatePaused, status)
		}

		// Resume VM
		if err := mock.Resume(ctx, vmID); err != nil {
			t.Errorf("Failed to resume mock VM: %v", err)
		}

		// Check running status
		status, err = mock.GetStatus(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM status: %v", err)
		}

		if status != StateRunning {
			t.Errorf("Expected state %s, got %s", StateRunning, status)
		}
	})

	t.Run("Snapshot", func(t *testing.T) {
		if !mock.SupportsSnapshot() {
			t.Skip("Mock doesn't support snapshots")
		}

		config := VMConfig{
			ID:       "test-vm-snapshot",
			Name:     "Test VM Snapshot",
			CPUShares: 1,
			MemoryMB: 256,
		}

		vmID, err := mock.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create mock VM: %v", err)
		}
		defer mock.Delete(ctx, vmID)

		// Start VM
		if err := mock.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start mock VM: %v", err)
		}

		// Create snapshot
		snapshotID, err := mock.Snapshot(ctx, vmID, "test-snapshot", nil)
		if err != nil {
			t.Errorf("Failed to create snapshot: %v", err)
		}

		if snapshotID == "" {
			t.Error("Snapshot ID should not be empty")
		}

		t.Logf("Created snapshot: %s", snapshotID)
	})

	t.Run("ListVMs", func(t *testing.T) {
		// Get initial count
		initialVMs, err := mock.ListVMs(ctx)
		if err != nil {
			t.Fatalf("Failed to list VMs: %v", err)
		}
		initialCount := len(initialVMs)

		// Create test VMs
		var vmIDs []string
		for i := 0; i < 3; i++ {
			config := VMConfig{
				ID:       fmt.Sprintf("test-vm-list-%d", i),
				Name:     fmt.Sprintf("Test VM List %d", i),
				CPUShares: 1,
				MemoryMB: 256,
			}

			vmID, err := mock.Create(ctx, config)
			if err != nil {
				t.Errorf("Failed to create VM %d: %v", i, err)
				continue
			}
			vmIDs = append(vmIDs, vmID)
		}

		// List VMs again
		vms, err := mock.ListVMs(ctx)
		if err != nil {
			t.Errorf("Failed to list VMs: %v", err)
		}

		if len(vms) != initialCount+len(vmIDs) {
			t.Errorf("Expected %d VMs, got %d", initialCount+len(vmIDs), len(vms))
		}

		// Cleanup
		for _, vmID := range vmIDs {
			mock.Delete(ctx, vmID)
		}
	})

	t.Run("ErrorHandling", func(t *testing.T) {
		nonExistentID := "non-existent-vm"

		// Test operations on non-existent VM
		if _, err := mock.GetStatus(ctx, nonExistentID); err == nil {
			t.Error("Expected error when getting status of non-existent VM")
		}

		if err := mock.Start(ctx, nonExistentID); err == nil {
			t.Error("Expected error when starting non-existent VM")
		}

		if err := mock.Stop(ctx, nonExistentID); err == nil {
			t.Error("Expected error when stopping non-existent VM")
		}

		if err := mock.Delete(ctx, nonExistentID); err == nil {
			t.Error("Expected error when deleting non-existent VM")
		}
	})

	t.Run("FailureInjection", func(t *testing.T) {
		// Configure mock for failures
		unreliableMock := NewMockHypervisor("test-node-2", "test-unreliable")
		unreliableMock.Configure(
			MockFailureConfig{
				CreateFailureRate: 1.0, // 100% failure rate
			},
			MockLatencyConfig{},
			MockCapabilities{},
		)

		config := VMConfig{
			ID:       "test-vm-failure",
			Name:     "Test VM Failure",
			CPUShares: 1,
			MemoryMB: 256,
		}

		// This should fail due to 100% failure rate
		_, err := unreliableMock.Create(ctx, config)
		if err == nil {
			t.Error("Expected create operation to fail due to failure injection")
		}

		t.Logf("Expected failure occurred: %v", err)
	})

	t.Run("PerformanceTesting", func(t *testing.T) {
		const numVMs = 10
		start := time.Now()

		var vmIDs []string
		for i := 0; i < numVMs; i++ {
			config := VMConfig{
				ID:       fmt.Sprintf("perf-vm-%d", i),
				Name:     fmt.Sprintf("Performance VM %d", i),
				CPUShares: 1,
				MemoryMB: 128,
			}

			vmID, err := mock.Create(ctx, config)
			if err != nil {
				t.Errorf("Failed to create VM %d: %v", i, err)
				continue
			}
			vmIDs = append(vmIDs, vmID)
		}

		creationTime := time.Since(start)
		avgTime := creationTime / time.Duration(len(vmIDs))
		
		t.Logf("Created %d VMs in %v (avg: %v per VM)", len(vmIDs), creationTime, avgTime)

		// Cleanup
		for _, vmID := range vmIDs {
			mock.Delete(ctx, vmID)
		}
	})

	// Reset mock state
	mock.Reset()
}

// TestHypervisorTestSuite tests the generic test suite framework
func TestHypervisorTestSuite(t *testing.T) {
	mock := NewMockHypervisor("test-node", "test-suite-mock")
	suite := NewHypervisorTestSuite(t, mock, VMTypeKVM)
	defer suite.Cleanup()

	ctx := context.Background()

	// Run all tests in the suite
	if err := suite.RunAllTests(ctx); err != nil {
		t.Errorf("Test suite failed: %v", err)
	}

	t.Logf("Test suite completed successfully")
}

// TestDriverFactory tests the VM driver factory
func TestDriverFactory(t *testing.T) {
	config := DefaultVMDriverConfig("test-factory-node")
	factory := NewVMDriverFactory(config)

	t.Run("KVMDriver", func(t *testing.T) {
		vmConfig := VMConfig{
			ID:   "test-kvm-vm",
			Name: "Test KVM VM",
			Tags: map[string]string{"vm_type": "kvm"},
		}

		// This will likely fail due to missing QEMU, but should test the factory logic
		driver, err := factory(vmConfig)
		if err != nil {
			t.Logf("Expected failure for KVM driver (QEMU not available): %v", err)
		} else {
			t.Logf("KVM driver created successfully")
			if driver == nil {
				t.Error("Driver should not be nil")
			}
		}
	})

	t.Run("ContainerDriver", func(t *testing.T) {
		vmConfig := VMConfig{
			ID:   "test-container-vm",
			Name: "Test Container VM",
			Tags: map[string]string{"vm_type": "container"},
		}

		driver, err := factory(vmConfig)
		if err != nil {
			t.Logf("Container driver creation failed (Docker not available): %v", err)
		} else {
			t.Logf("Container driver created successfully")
			if driver == nil {
				t.Error("Driver should not be nil")
			}
		}
	})

	t.Run("UnsupportedType", func(t *testing.T) {
		vmConfig := VMConfig{
			ID:   "test-unsupported-vm",
			Name: "Test Unsupported VM",
			Tags: map[string]string{"vm_type": "unsupported"},
		}

		_, err := factory(vmConfig)
		if err == nil {
			t.Error("Expected error for unsupported VM type")
		} else {
			t.Logf("Expected error for unsupported type: %v", err)
		}
	})
}

// TestVMDriverManager tests the VM driver manager
func TestVMDriverManager(t *testing.T) {
	config := DefaultVMDriverConfig("test-manager-node")
	manager := NewVMDriverManager(config)
	defer manager.Close()

	t.Run("SupportedTypes", func(t *testing.T) {
		types := manager.ListSupportedTypes()
		expectedTypes := []VMType{
			VMTypeContainer,
			VMTypeContainerd,
			VMTypeKataContainers,
			VMTypeKVM,
			VMTypeProcess,
		}

		if len(types) != len(expectedTypes) {
			t.Errorf("Expected %d supported types, got %d", len(expectedTypes), len(types))
		}

		for _, expectedType := range expectedTypes {
			found := false
			for _, actualType := range types {
				if actualType == expectedType {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("Expected type %s not found in supported types", expectedType)
			}
		}
	})
}

// BenchmarkMockHypervisor benchmarks the mock hypervisor performance
func BenchmarkMockHypervisor(b *testing.B) {
	mock := NewMockHypervisor("bench-node", "bench-mock")
	ctx := context.Background()

	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		config := VMConfig{
			ID:       fmt.Sprintf("bench-vm-%d", i),
			Name:     fmt.Sprintf("Benchmark VM %d", i),
			CPUShares: 1,
			MemoryMB: 128,
		}

		vmID, err := mock.Create(ctx, config)
		if err != nil {
			b.Errorf("Failed to create VM: %v", err)
			continue
		}

		// Clean up immediately
		mock.Delete(ctx, vmID)
	}
}

// BenchmarkMockHypervisorWithLatency benchmarks with realistic latency
func BenchmarkMockHypervisorWithLatency(b *testing.B) {
	mock := NewMockHypervisor("bench-latency-node", "bench-latency-mock")
	mock.Configure(
		MockFailureConfig{},
		MockLatencyConfig{
			CreateLatency: 100 * time.Millisecond,
			StartLatency:  1 * time.Second,
			StopLatency:   500 * time.Millisecond,
		},
		MockCapabilities{},
	)
	
	ctx := context.Background()

	b.ResetTimer()
	
	for i := 0; i < b.N; i++ {
		config := VMConfig{
			ID:       fmt.Sprintf("bench-latency-vm-%d", i),
			Name:     fmt.Sprintf("Benchmark Latency VM %d", i),
			CPUShares: 1,
			MemoryMB: 128,
		}

		vmID, err := mock.Create(ctx, config)
		if err != nil {
			b.Errorf("Failed to create VM: %v", err)
			continue
		}

		// Clean up immediately
		mock.Delete(ctx, vmID)
	}
}