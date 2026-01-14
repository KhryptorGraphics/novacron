package verification

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
)

// TestBackupVMIDAssociation verifies that VM IDs are properly handled
// and not confused with Job IDs
func TestBackupVMIDAssociation(t *testing.T) {
	// Create a backup manager
	manager := backup.NewBackupManager()

	// Test 1: Verify BackupInfo has VMID field
	backupInfo := backup.BackupInfo{
		ID:    "backup-1",
		JobID: "job-1",
		VMID:  "vm-123", // This should be independent of JobID
	}

	if backupInfo.VMID == backupInfo.JobID {
		t.Errorf("VMID should not equal JobID by default")
	}

	// Test 2: Create a test backup with metadata
	testBackup := &backup.Backup{
		ID:    "backup-2",
		JobID: "job-2",
		Metadata: map[string]string{
			"vm_id": "vm-456",
		},
		StartedAt: time.Now(),
		TenantID:  "test-tenant",
	}

	// Add backup for testing
	manager.AddBackupForTest(testBackup)

	// Test 3: ListAllBackups should extract VM ID from metadata
	ctx := context.Background()
	backups, err := manager.ListAllBackups(ctx)
	if err != nil {
		t.Fatalf("Failed to list backups: %v", err)
	}

	if len(backups) != 1 {
		t.Fatalf("Expected 1 backup, got %d", len(backups))
	}

	// Verify VMID is extracted from metadata, not JobID
	if backups[0].VMID != "vm-456" {
		t.Errorf("Expected VMID to be 'vm-456', got '%s'", backups[0].VMID)
	}

	if backups[0].VMID == backups[0].JobID {
		t.Errorf("VMID should not be equal to JobID")
	}

	// Test 4: GetBackupManifest should return proper VMID
	manifest, err := manager.GetBackupManifest("backup-2")
	if err != nil {
		t.Fatalf("Failed to get backup manifest: %v", err)
	}

	if manifest.VMID != "vm-456" {
		t.Errorf("Expected manifest VMID to be 'vm-456', got '%s'", manifest.VMID)
	}

	// Test 5: InitializeCBT and GetCBTStats should work
	tracker, err := manager.InitializeCBT("vm-789", 1024*1024*1024)
	if err != nil {
		t.Fatalf("Failed to initialize CBT: %v", err)
	}

	if tracker.VMID() != "vm-789" {
		t.Errorf("Expected tracker VMID to be 'vm-789', got '%s'", tracker.VMID())
	}

	stats, err := manager.GetCBTStats("vm-789")
	if err != nil {
		t.Fatalf("Failed to get CBT stats: %v", err)
	}

	if stats["vm_id"] != "vm-789" {
		t.Errorf("Expected stats vm_id to be 'vm-789', got '%v'", stats["vm_id"])
	}

	fmt.Println("✅ All VM ID association tests passed!")
}