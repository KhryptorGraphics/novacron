package vm

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

func TestVMStorageIntegration(t *testing.T) {
	// Create storage service
	storageConfig := storage.DefaultStorageConfig()
	storageService := storage.NewBaseStorageService(storageConfig)
	
	err := storageService.Start()
	if err != nil {
		t.Fatalf("Failed to start storage service: %v", err)
	}
	defer storageService.Stop()

	// Create VM storage manager
	storageManager := NewVMStorageManager(storageService)

	// Create VM manager
	vmConfig := DefaultVMManagerConfig()
	vmManager, err := NewVMManagerFixed(vmConfig, "test-node-storage")
	if err != nil {
		t.Fatalf("Failed to create VM manager: %v", err)
	}

	err = vmManager.Start()
	if err != nil {
		t.Fatalf("Failed to start VM manager: %v", err)
	}
	defer vmManager.Stop()

	ctx := context.Background()

	// Test creating VM with storage
	vmCfg := VMConfig{
		ID:        "storage-test-vm",
		Name:      "storage-test",
		Command:   "/bin/sleep",
		Args:      []string{"30"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := vmManager.CreateVMWithStorage(ctx, vmCfg, storageManager, 10, 5) // 10GB boot, 5GB data
	if err != nil {
		t.Fatalf("Failed to create VM with storage: %v", err)
	}

	// Verify VM was created
	if vm.ID() != vmCfg.ID {
		t.Errorf("Expected VM ID %s, got %s", vmCfg.ID, vm.ID())
	}

	// Test getting VM volumes
	volumes, err := storageManager.GetVMVolumes(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to get VM volumes: %v", err)
	}

	if len(volumes) < 1 {
		t.Error("Expected at least 1 volume (boot volume)")
	}

	// Test getting boot volume
	bootVolume, err := storageManager.GetBootVolume(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to get boot volume: %v", err)
	}

	if bootVolume.Metadata["volume_type"] != "boot" {
		t.Error("Expected boot volume to have volume_type=boot")
	}

	if bootVolume.Metadata["vm_id"] != vm.ID() {
		t.Errorf("Expected boot volume vm_id to be %s, got %s", vm.ID(), bootVolume.Metadata["vm_id"])
	}

	// Test getting storage stats
	stats, err := storageManager.GetVMStorageStats(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to get VM storage stats: %v", err)
	}

	if stats.VMID != vm.ID() {
		t.Errorf("Expected stats VM ID %s, got %s", vm.ID(), stats.VMID)
	}

	if stats.VolumeCount == 0 {
		t.Error("Expected at least 1 volume in stats")
	}

	if stats.TotalSizeGB <= 0 {
		t.Error("Expected total size to be greater than 0")
	}

	// Test creating VM snapshot
	snapshots, err := storageManager.CreateVMSnapshot(ctx, vm.ID(), "test-snapshot")
	if err != nil {
		t.Fatalf("Failed to create VM snapshot: %v", err)
	}

	if len(snapshots) == 0 {
		t.Error("Expected at least one snapshot to be created")
	}

	// Test VM storage methods
	vmVolumes, err := vm.GetStorageVolumes(ctx, storageManager)
	if err != nil {
		t.Fatalf("Failed to get storage volumes through VM: %v", err)
	}

	if len(vmVolumes) != len(volumes) {
		t.Errorf("Expected %d volumes through VM method, got %d", len(volumes), len(vmVolumes))
	}

	// Test getting storage stats through VM
	vmStats, err := vm.GetStorageStats(ctx, storageManager)
	if err != nil {
		t.Fatalf("Failed to get storage stats through VM: %v", err)
	}

	if vmStats.VMID != vm.ID() {
		t.Errorf("Expected VM stats ID %s, got %s", vm.ID(), vmStats.VMID)
	}

	// Test deleting VM with storage
	err = vmManager.DeleteVMWithStorage(ctx, vm.ID(), storageManager)
	if err != nil {
		t.Fatalf("Failed to delete VM with storage: %v", err)
	}

	// Verify VM and volumes were deleted
	_, err = vmManager.GetVM(vm.ID())
	if err == nil {
		t.Error("Expected error when getting deleted VM")
	}

	remainingVolumes, err := storageManager.GetVMVolumes(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to check remaining volumes: %v", err)
	}

	if len(remainingVolumes) != 0 {
		t.Errorf("Expected 0 remaining volumes, got %d", len(remainingVolumes))
	}
}

func TestVMStorageVolumeOperations(t *testing.T) {
	// Create storage service
	storageConfig := storage.DefaultStorageConfig()
	storageService := storage.NewBaseStorageService(storageConfig)
	
	err := storageService.Start()
	if err != nil {
		t.Fatalf("Failed to start storage service: %v", err)
	}
	defer storageService.Stop()

	// Create VM storage manager
	storageManager := NewVMStorageManager(storageService)

	ctx := context.Background()

	// Create a test VM (without using the manager)
	vmConfig := VMConfig{
		ID:        "volume-ops-test-vm",
		Name:      "volume-ops-test",
		Command:   "/bin/sleep",
		Args:      []string{"10"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(vmConfig)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Test creating individual volumes
	bootVolume, err := storageManager.CreateBootVolume(ctx, vm.ID(), vm.Name(), 8)
	if err != nil {
		t.Fatalf("Failed to create boot volume: %v", err)
	}

	dataVolume, err := storageManager.CreateDataVolume(ctx, vm.ID(), vm.Name(), "appdata", 4)
	if err != nil {
		t.Fatalf("Failed to create data volume: %v", err)
	}

	// Test attaching volumes
	err = storageManager.AttachVolume(ctx, vm.ID(), bootVolume.ID)
	if err != nil {
		t.Fatalf("Failed to attach boot volume: %v", err)
	}

	err = storageManager.AttachVolume(ctx, vm.ID(), dataVolume.ID)
	if err != nil {
		t.Fatalf("Failed to attach data volume: %v", err)
	}

	// Verify volumes are attached
	attachedVolumes, err := storageManager.GetVMVolumes(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to get attached volumes: %v", err)
	}

	if len(attachedVolumes) != 2 {
		t.Errorf("Expected 2 attached volumes, got %d", len(attachedVolumes))
	}

	// Test detaching volumes
	err = storageManager.DetachVolume(ctx, vm.ID(), dataVolume.ID)
	if err != nil {
		t.Fatalf("Failed to detach data volume: %v", err)
	}

	// Test volume access through VM methods
	err = vm.AttachStorageVolume(ctx, storageManager, dataVolume.ID)
	if err != nil {
		t.Fatalf("Failed to attach volume through VM method: %v", err)
	}

	err = vm.DetachStorageVolume(ctx, storageManager, dataVolume.ID)
	if err != nil {
		t.Fatalf("Failed to detach volume through VM method: %v", err)
	}

	// Test creating snapshot through VM
	snapshots, err := vm.CreateStorageSnapshot(ctx, storageManager, "vm-method-snapshot")
	if err != nil {
		t.Fatalf("Failed to create snapshot through VM method: %v", err)
	}

	if len(snapshots) == 0 {
		t.Error("Expected at least one snapshot")
	}

	// Clean up
	err = storageManager.DeleteVMVolumes(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to delete VM volumes: %v", err)
	}
}

func TestVMStorageStatsAccuracy(t *testing.T) {
	// Create storage service
	storageConfig := storage.DefaultStorageConfig()
	storageService := storage.NewBaseStorageService(storageConfig)
	
	err := storageService.Start()
	if err != nil {
		t.Fatalf("Failed to start storage service: %v", err)
	}
	defer storageService.Stop()

	// Create VM storage manager
	storageManager := NewVMStorageManager(storageService)

	ctx := context.Background()

	// Create VM
	vmConfig := VMConfig{
		ID:        "stats-test-vm",
		Name:      "stats-test",
		Command:   "/bin/sleep",
		Args:      []string{"10"},
		CPUShares: 1024,
		MemoryMB:  512,
		RootFS:    "/tmp",
	}

	vm, err := NewVM(vmConfig)
	if err != nil {
		t.Fatalf("Failed to create VM: %v", err)
	}

	// Create multiple volumes with known sizes
	volumes := []struct {
		name   string
		sizeGB int
		vtype  string
	}{
		{"boot", 10, "boot"},
		{"data1", 5, "data"},
		{"data2", 3, "data"},
	}

	expectedTotalGB := float64(0)
	for _, vol := range volumes {
		expectedTotalGB += float64(vol.sizeGB)
		
		if vol.vtype == "boot" {
			_, err := storageManager.CreateBootVolume(ctx, vm.ID(), vm.Name(), vol.sizeGB)
			if err != nil {
				t.Fatalf("Failed to create boot volume: %v", err)
			}
		} else {
			_, err := storageManager.CreateDataVolume(ctx, vm.ID(), vm.Name(), vol.name, vol.sizeGB)
			if err != nil {
				t.Fatalf("Failed to create data volume %s: %v", vol.name, err)
			}
		}
	}

	// Wait for volumes to be created
	time.Sleep(100 * time.Millisecond)

	// Get storage stats
	stats, err := storageManager.GetVMStorageStats(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to get storage stats: %v", err)
	}

	// Verify stats accuracy
	if stats.VolumeCount != len(volumes) {
		t.Errorf("Expected %d volumes in stats, got %d", len(volumes), stats.VolumeCount)
	}

	if stats.TotalSizeGB < expectedTotalGB*0.9 || stats.TotalSizeGB > expectedTotalGB*1.1 {
		t.Errorf("Expected total size around %.1fGB, got %.1fGB", expectedTotalGB, stats.TotalSizeGB)
	}

	// Verify individual volume stats
	if len(stats.VolumeStats) != len(volumes) {
		t.Errorf("Expected %d volume stats, got %d", len(volumes), len(stats.VolumeStats))
	}

	// Check for boot volume
	foundBoot := false
	for _, volStat := range stats.VolumeStats {
		if volStat.Bootable {
			foundBoot = true
			if volStat.SizeGB < 9 || volStat.SizeGB > 11 { // 10GB ± 1GB tolerance
				t.Errorf("Expected boot volume size around 10GB, got %.1fGB", volStat.SizeGB)
			}
			break
		}
	}

	if !foundBoot {
		t.Error("Expected to find a bootable volume in stats")
	}

	// Clean up
	err = storageManager.DeleteVMVolumes(ctx, vm.ID())
	if err != nil {
		t.Fatalf("Failed to delete VM volumes: %v", err)
	}
}