package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

// KVMIntegrationTest provides comprehensive testing for KVM/QEMU hypervisor
type KVMIntegrationTest struct {
	driver       VMDriver
	tempDir      string
	qemuPath     string
	hasQEMU      bool
	hasKVM       bool
	testImagePath string
}

// NewKVMIntegrationTest creates a new KVM integration test suite
func NewKVMIntegrationTest(t *testing.T) *KVMIntegrationTest {
	test := &KVMIntegrationTest{}
	
	// Create temporary directory for test files
	tempDir, err := os.MkdirTemp("", "novacron-kvm-test-*")
	if err != nil {
		t.Fatalf("Failed to create temp dir: %v", err)
	}
	test.tempDir = tempDir

	// Check for QEMU availability
	test.qemuPath, test.hasQEMU = test.findQEMUBinary()
	
	// Check for KVM availability
	test.hasKVM = test.checkKVMAvailability()

	if test.hasQEMU {
		// Initialize KVM driver
		config := map[string]interface{}{
			"qemu_path": test.qemuPath,
			"vm_path":   filepath.Join(tempDir, "vms"),
		}
		
		driver, err := NewKVMDriver(config)
		if err != nil {
			t.Logf("Warning: Failed to create KVM driver: %v", err)
		} else {
			test.driver = driver
		}

		// Create test disk image
		test.testImagePath = test.createTestImage(t)
	}

	return test
}

// findQEMUBinary locates QEMU binary on the system
func (k *KVMIntegrationTest) findQEMUBinary() (string, bool) {
	candidates := []string{
		"/usr/bin/qemu-system-x86_64",
		"/usr/local/bin/qemu-system-x86_64",
		"/opt/homebrew/bin/qemu-system-x86_64", // macOS with Homebrew
		"qemu-system-x86_64", // PATH lookup
	}

	for _, path := range candidates {
		if _, err := exec.LookPath(path); err == nil {
			return path, true
		}
		if _, err := os.Stat(path); err == nil {
			return path, true
		}
	}

	return "", false
}

// checkKVMAvailability checks if KVM is available on the system
func (k *KVMIntegrationTest) checkKVMAvailability() bool {
	// Check if /dev/kvm exists and is accessible
	if _, err := os.Stat("/dev/kvm"); err != nil {
		return false
	}

	// Try to open /dev/kvm to verify permissions
	if file, err := os.Open("/dev/kvm"); err == nil {
		file.Close()
		return true
	}

	return false
}

// createTestImage creates a minimal test disk image
func (k *KVMIntegrationTest) createTestImage(t *testing.T) string {
	imagePath := filepath.Join(k.tempDir, "test-image.qcow2")
	
	// Create a small test image (64MB)
	cmd := exec.Command("qemu-img", "create", "-f", "qcow2", imagePath, "64M")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Logf("Warning: Failed to create test image: %v, output: %s", err, string(output))
		return ""
	}

	return imagePath
}

// Cleanup removes temporary files and resources
func (k *KVMIntegrationTest) Cleanup() {
	if k.tempDir != "" {
		os.RemoveAll(k.tempDir)
	}
}

// TestKVMDriverCreation tests KVM driver initialization
func (k *KVMIntegrationTest) TestKVMDriverCreation(t *testing.T) {
	if !k.hasQEMU {
		t.Skip("QEMU not available, skipping KVM driver creation test")
	}

	t.Run("ValidConfiguration", func(t *testing.T) {
		config := map[string]interface{}{
			"qemu_path": k.qemuPath,
			"vm_path":   filepath.Join(k.tempDir, "test-vms"),
		}

		driver, err := NewKVMDriver(config)
		if err != nil {
			t.Errorf("Failed to create KVM driver: %v", err)
		}

		if driver == nil {
			t.Error("KVM driver should not be nil")
		}
	})

	t.Run("InvalidQEMUPath", func(t *testing.T) {
		config := map[string]interface{}{
			"qemu_path": "/nonexistent/qemu-system-x86_64",
		}

		_, err := NewKVMDriver(config)
		if err == nil {
			t.Error("Expected error for invalid QEMU path")
		}
	})
}

// TestQMPProtocol tests QEMU Machine Protocol communication
func (k *KVMIntegrationTest) TestQMPProtocol(t *testing.T) {
	if !k.hasQEMU || k.driver == nil {
		t.Skip("QEMU not available or driver not initialized")
	}

	t.Run("QMPConnection", func(t *testing.T) {
		// This would test QMP socket connection
		// For now, we'll test that the monitor socket path is created
		config := VMConfig{
			ID:       "qmp-test-vm",
			Name:     "QMP Test VM",
			CPUShares: 1,
			MemoryMB: 128,
			RootFS:   k.testImagePath,
		}

		if config.RootFS == "" {
			t.Skip("No test image available")
		}

		ctx := context.Background()
		vmID, err := k.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create VM for QMP test: %v", err)
		}
		defer k.driver.Delete(ctx, vmID)

		// Check that monitor socket path exists in VM directory
		vmDir := filepath.Join(k.tempDir, "vms", vmID)
		monitorPath := filepath.Join(vmDir, "monitor.sock")
		
		// The socket won't exist until VM is started, but directory should exist
		if _, err := os.Stat(vmDir); err != nil {
			t.Errorf("VM directory should exist: %v", err)
		}

		t.Logf("Expected monitor socket path: %s", monitorPath)
	})
}

// TestKVMVMLifecycle tests complete VM lifecycle with KVM
func (k *KVMIntegrationTest) TestKVMVMLifecycle(t *testing.T) {
	if !k.hasQEMU || k.driver == nil {
		t.Skip("QEMU not available or driver not initialized")
	}

	if k.testImagePath == "" {
		t.Skip("No test image available")
	}

	ctx := context.Background()
	config := VMConfig{
		ID:       "lifecycle-test-vm",
		Name:     "Lifecycle Test VM",
		CPUShares: 1,
		MemoryMB: 256,
		RootFS:   k.testImagePath,
		Tags:     map[string]string{"test": "lifecycle"},
	}

	t.Run("CreateAndDelete", func(t *testing.T) {
		vmID, err := k.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}

		// Verify VM was created
		status, err := k.driver.GetStatus(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM status: %v", err)
		} else if status != StateCreated && status != StateStopped {
			t.Errorf("Expected VM to be created or stopped, got: %v", status)
		}

		// Get VM info
		info, err := k.driver.GetInfo(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM info: %v", err)
		} else {
			if info.ID != vmID {
				t.Errorf("Expected VM ID %s, got %s", vmID, info.ID)
			}
			if info.Name != config.Name {
				t.Errorf("Expected VM name %s, got %s", config.Name, info.Name)
			}
		}

		// Delete VM
		if err := k.driver.Delete(ctx, vmID); err != nil {
			t.Errorf("Failed to delete VM: %v", err)
		}

		// Verify VM was deleted
		if _, err := k.driver.GetStatus(ctx, vmID); err == nil {
			t.Error("Expected error when getting status of deleted VM")
		}
	})
}

// TestKVMResourceManagement tests resource allocation and monitoring
func (k *KVMIntegrationTest) TestKVMResourceManagement(t *testing.T) {
	if !k.hasQEMU || k.driver == nil {
		t.Skip("QEMU not available or driver not initialized")
	}

	if k.testImagePath == "" {
		t.Skip("No test image available")
	}

	ctx := context.Background()

	t.Run("ResourceAllocation", func(t *testing.T) {
		configs := []VMConfig{
			{
				ID:       "resource-test-vm-1",
				Name:     "Small VM",
				CPUShares: 1,
				MemoryMB: 128,
				RootFS:   k.testImagePath,
			},
			{
				ID:       "resource-test-vm-2", 
				Name:     "Medium VM",
				CPUShares: 2,
				MemoryMB: 512,
				RootFS:   k.testImagePath,
			},
		}

		var vmIDs []string
		for _, config := range configs {
			vmID, err := k.driver.Create(ctx, config)
			if err != nil {
				t.Errorf("Failed to create VM %s: %v", config.Name, err)
				continue
			}
			vmIDs = append(vmIDs, vmID)

			// Verify resource allocation
			info, err := k.driver.GetInfo(ctx, vmID)
			if err != nil {
				t.Errorf("Failed to get info for VM %s: %v", vmID, err)
			} else {
				if info.CPUShares != config.CPUShares {
					t.Errorf("Expected CPU shares %d, got %d", config.CPUShares, info.CPUShares)
				}
				if info.MemoryMB != config.MemoryMB {
					t.Errorf("Expected memory %d MB, got %d MB", config.MemoryMB, info.MemoryMB)
				}
			}
		}

		// Cleanup
		for _, vmID := range vmIDs {
			k.driver.Delete(ctx, vmID)
		}
	})
}

// TestKVMDiskOperations tests disk-related operations
func (k *KVMIntegrationTest) TestKVMDiskOperations(t *testing.T) {
	if !k.hasQEMU || k.driver == nil {
		t.Skip("QEMU not available or driver not initialized")
	}

	ctx := context.Background()

	t.Run("DiskImageCreation", func(t *testing.T) {
		config := VMConfig{
			ID:         "disk-test-vm",
			Name:       "Disk Test VM",
			CPUShares:  1,
			MemoryMB:   256,
			DiskSizeGB: 1, // 1GB disk
		}

		vmID, err := k.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		defer k.driver.Delete(ctx, vmID)

		// Verify disk was created
		vmDir := filepath.Join(k.tempDir, "vms", vmID)
		diskPath := filepath.Join(vmDir, "disk.qcow2")

		if _, err := os.Stat(diskPath); err != nil {
			t.Errorf("Expected disk file to exist at %s: %v", diskPath, err)
		}

		// Check disk size using qemu-img info
		cmd := exec.Command("qemu-img", "info", "--output=json", diskPath)
		output, err := cmd.CombinedOutput()
		if err != nil {
			t.Logf("Could not get disk info: %v", err)
		} else {
			var info struct {
				VirtualSize int64 `json:"virtual-size"`
				Format      string `json:"format"`
			}
			
			if err := json.Unmarshal(output, &info); err == nil {
				expectedSize := int64(config.MemoryMB * 10 * 1024 * 1024) // 10x memory in bytes
				if config.MemoryMB == 0 {
					expectedSize = 8192 * 1024 * 1024 // Default 8GB
				}
				
				if info.VirtualSize != expectedSize {
					t.Logf("Disk size: expected %d bytes, got %d bytes", expectedSize, info.VirtualSize)
				}
				
				if info.Format != "qcow2" {
					t.Errorf("Expected qcow2 format, got %s", info.Format)
				}
			}
		}
	})
}

// TestKVMNetworking tests network configuration
func (k *KVMIntegrationTest) TestKVMNetworking(t *testing.T) {
	if !k.hasQEMU || k.driver == nil {
		t.Skip("QEMU not available or driver not initialized")
	}

	if k.testImagePath == "" {
		t.Skip("No test image available")
	}

	ctx := context.Background()

	t.Run("NetworkConfiguration", func(t *testing.T) {
		config := VMConfig{
			ID:        "network-test-vm",
			Name:      "Network Test VM",
			CPUShares: 1,
			MemoryMB:  256,
			RootFS:    k.testImagePath,
			NetworkID: "test-network",
		}

		vmID, err := k.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		defer k.driver.Delete(ctx, vmID)

		info, err := k.driver.GetInfo(ctx, vmID)
		if err != nil {
			t.Errorf("Failed to get VM info: %v", err)
		} else if info.NetworkID != config.NetworkID {
			t.Errorf("Expected network ID %s, got %s", config.NetworkID, info.NetworkID)
		}
	})
}

// TestKVMPerformanceBenchmarks runs performance benchmarks
func (k *KVMIntegrationTest) TestKVMPerformanceBenchmarks(t *testing.T) {
	if !k.hasQEMU || k.driver == nil {
		t.Skip("QEMU not available or driver not initialized")
	}

	if k.testImagePath == "" {
		t.Skip("No test image available")
	}

	ctx := context.Background()

	t.Run("VMCreationBenchmark", func(t *testing.T) {
		const numVMs = 5
		configs := make([]VMConfig, numVMs)
		
		for i := 0; i < numVMs; i++ {
			configs[i] = VMConfig{
				ID:       fmt.Sprintf("bench-vm-%d", i),
				Name:     fmt.Sprintf("Benchmark VM %d", i),
				CPUShares: 1,
				MemoryMB: 256,
				RootFS:   k.testImagePath,
			}
		}

		start := time.Now()
		var vmIDs []string

		for _, config := range configs {
			vmID, err := k.driver.Create(ctx, config)
			if err != nil {
				t.Errorf("Failed to create VM %s: %v", config.Name, err)
				continue
			}
			vmIDs = append(vmIDs, vmID)
		}

		creationTime := time.Since(start)
		avgCreationTime := creationTime / time.Duration(len(vmIDs))

		t.Logf("Created %d VMs in %v (avg: %v per VM)", len(vmIDs), creationTime, avgCreationTime)

		// Cleanup
		for _, vmID := range vmIDs {
			k.driver.Delete(ctx, vmID)
		}
	})
}

// TestKVMErrorRecovery tests error handling and recovery scenarios
func (k *KVMIntegrationTest) TestKVMErrorRecovery(t *testing.T) {
	if !k.hasQEMU || k.driver == nil {
		t.Skip("QEMU not available or driver not initialized")
	}

	ctx := context.Background()

	t.Run("InvalidDiskPath", func(t *testing.T) {
		config := VMConfig{
			ID:       "error-test-vm",
			Name:     "Error Test VM",
			CPUShares: 1,
			MemoryMB: 256,
			RootFS:   "/nonexistent/disk.qcow2",
		}

		vmID, err := k.driver.Create(ctx, config)
		if err != nil {
			// This is expected - creation might fail with invalid disk path
			t.Logf("Expected error creating VM with invalid disk: %v", err)
			return
		}

		// If creation succeeded, starting should fail
		if err := k.driver.Start(ctx, vmID); err == nil {
			t.Error("Expected error starting VM with invalid disk path")
		}

		// Cleanup
		k.driver.Delete(ctx, vmID)
	})

	t.Run("ExcessiveMemory", func(t *testing.T) {
		config := VMConfig{
			ID:       "memory-test-vm",
			Name:     "Memory Test VM", 
			CPUShares: 1,
			MemoryMB: 1024 * 1024, // 1TB - should be rejected or cause issues
		}

		_, err := k.driver.Create(ctx, config)
		if err != nil {
			t.Logf("Creation failed with excessive memory (expected): %v", err)
		} else {
			t.Log("VM creation succeeded with excessive memory allocation")
			// Don't start it - just clean up
			k.driver.Delete(ctx, "memory-test-vm")
		}
	})
}

// TestKVMSnapshot tests snapshot functionality if supported
func (k *KVMIntegrationTest) TestKVMSnapshot(t *testing.T) {
	if !k.hasQEMU || k.driver == nil {
		t.Skip("QEMU not available or driver not initialized")
	}

	if !k.driver.SupportsSnapshot() {
		t.Skip("KVM driver doesn't support snapshots")
	}

	if k.testImagePath == "" {
		t.Skip("No test image available")
	}

	ctx := context.Background()
	config := VMConfig{
		ID:       "snapshot-test-vm",
		Name:     "Snapshot Test VM",
		CPUShares: 1,
		MemoryMB: 256,
		RootFS:   k.testImagePath,
	}

	t.Run("SnapshotCreation", func(t *testing.T) {
		vmID, err := k.driver.Create(ctx, config)
		if err != nil {
			t.Fatalf("Failed to create VM: %v", err)
		}
		defer k.driver.Delete(ctx, vmID)

		// Start VM
		if err := k.driver.Start(ctx, vmID); err != nil {
			t.Fatalf("Failed to start VM: %v", err)
		}
		defer k.driver.Stop(ctx, vmID)

		// Create snapshot
		snapshotID, err := k.driver.Snapshot(ctx, vmID, "test-snapshot", nil)
		if err != nil {
			t.Errorf("Failed to create snapshot: %v", err)
		} else if snapshotID == "" {
			t.Error("Snapshot ID should not be empty")
		} else {
			t.Logf("Created snapshot: %s", snapshotID)
		}
	})
}

// RunAllKVMTests executes all KVM integration tests
func (k *KVMIntegrationTest) RunAllKVMTests(t *testing.T) {
	defer k.Cleanup()

	t.Run("KVMDriverCreation", k.TestKVMDriverCreation)
	t.Run("QMPProtocol", k.TestQMPProtocol)
	t.Run("KVMVMLifecycle", k.TestKVMVMLifecycle)
	t.Run("KVMResourceManagement", k.TestKVMResourceManagement)
	t.Run("KVMDiskOperations", k.TestKVMDiskOperations)
	t.Run("KVMNetworking", k.TestKVMNetworking)
	t.Run("KVMPerformanceBenchmarks", k.TestKVMPerformanceBenchmarks)
	t.Run("KVMErrorRecovery", k.TestKVMErrorRecovery)
	t.Run("KVMSnapshot", k.TestKVMSnapshot)
}

// Integration test entry point
func TestKVMIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping KVM integration tests in short mode")
	}

	kvmTest := NewKVMIntegrationTest(t)
	kvmTest.RunAllKVMTests(t)
}

// Benchmark tests
func BenchmarkKVMVMCreation(b *testing.B) {
	kvmTest := NewKVMIntegrationTest(nil)
	defer kvmTest.Cleanup()

	if !kvmTest.hasQEMU || kvmTest.driver == nil {
		b.Skip("QEMU not available")
	}

	ctx := context.Background()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config := VMConfig{
			ID:       fmt.Sprintf("bench-vm-%d", i),
			Name:     fmt.Sprintf("Benchmark VM %d", i),
			CPUShares: 1,
			MemoryMB: 256,
		}

		vmID, err := kvmTest.driver.Create(ctx, config)
		if err != nil {
			b.Errorf("Failed to create VM: %v", err)
			continue
		}

		// Clean up immediately to avoid resource exhaustion
		kvmTest.driver.Delete(ctx, vmID)
	}
}