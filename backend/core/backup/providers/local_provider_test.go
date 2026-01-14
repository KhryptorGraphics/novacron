package providers

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
)

func TestLocalStorageProvider(t *testing.T) {
	// Create a temporary directory for testing
	baseDir, err := os.MkdirTemp("", "novacron-test-backups")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(baseDir)

	// Create the provider
	provider, err := NewLocalStorageProvider("test-provider", "Test Provider", baseDir)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Test provider properties
	if provider.ID() != "test-provider" {
		t.Errorf("Expected provider ID to be 'test-provider', got '%s'", provider.ID())
	}
	if provider.Name() != "Test Provider" {
		t.Errorf("Expected provider name to be 'Test Provider', got '%s'", provider.Name())
	}
	if provider.Type() != backup.LocalStorage {
		t.Errorf("Expected provider type to be LocalStorage, got '%s'", provider.Type())
	}

	// Test backup creation
	backupJob := &backup.BackupJob{
		ID:          "job-1",
		Name:        "Test Job",
		Description: "Test backup job",
		Type:        backup.FullBackup,
		Targets: []*backup.BackupTarget{
			{
				ID:         "target-1",
				Name:       "Test VM",
				Type:       "vm",
				ResourceID: "vm-123",
				Metadata: map[string]string{
					"os": "linux",
				},
			},
		},
		TenantID: "tenant-1",
	}

	// Create a backup
	ctx := context.Background()
	b, err := provider.CreateBackup(ctx, backupJob)
	if err != nil {
		t.Fatalf("Failed to create backup: %v", err)
	}

	// Verify backup properties
	if b.JobID != "job-1" {
		t.Errorf("Expected backup JobID to be 'job-1', got '%s'", b.JobID)
	}
	if b.Type != backup.FullBackup {
		t.Errorf("Expected backup Type to be FullBackup, got '%s'", b.Type)
	}
	if b.State != backup.BackupCompleted {
		t.Errorf("Expected backup State to be BackupCompleted, got '%s'", b.State)
	}
	if b.TenantID != "tenant-1" {
		t.Errorf("Expected backup TenantID to be 'tenant-1', got '%s'", b.TenantID)
	}

	// Verify backup directory structure
	backupDir := b.StorageLocation
	if _, err := os.Stat(backupDir); os.IsNotExist(err) {
		t.Errorf("Backup directory does not exist: %s", backupDir)
	}

	// Verify target directory
	targetDir := filepath.Join(backupDir, "target-1")
	if _, err := os.Stat(targetDir); os.IsNotExist(err) {
		t.Errorf("Target directory does not exist: %s", targetDir)
	}

	// Verify metadata file
	metadataFile := filepath.Join(targetDir, "metadata.json")
	if _, err := os.Stat(metadataFile); os.IsNotExist(err) {
		t.Errorf("Metadata file does not exist: %s", metadataFile)
	}

	// Test getting the backup
	retreivedBackup, err := provider.GetBackup(ctx, b.ID)
	if err != nil {
		t.Fatalf("Failed to get backup: %v", err)
	}
	if retreivedBackup.ID != b.ID {
		t.Errorf("Expected retrieved backup ID to be '%s', got '%s'", b.ID, retreivedBackup.ID)
	}

	// Test listing backups
	backups, err := provider.ListBackups(ctx, map[string]interface{}{
		"tenant_id": "tenant-1",
	})
	if err != nil {
		t.Fatalf("Failed to list backups: %v", err)
	}
	if len(backups) != 1 {
		t.Errorf("Expected 1 backup, got %d", len(backups))
	}

	// Test backup validation
	err = provider.ValidateBackup(ctx, b.ID)
	if err != nil {
		t.Errorf("Backup validation failed: %v", err)
	}

	// Test backup session
	session, err := provider.NewBackupSession(b, "target-2")
	if err != nil {
		t.Fatalf("Failed to create backup session: %v", err)
	}

	// Write a file to the backup
	err = session.WriteFile("test.txt", []byte("test content"))
	if err != nil {
		t.Fatalf("Failed to write file to backup: %v", err)
	}

	// Verify the file was written
	testFile := filepath.Join(b.StorageLocation, "target-2", "test.txt")
	if _, err := os.Stat(testFile); os.IsNotExist(err) {
		t.Errorf("Test file does not exist: %s", testFile)
	}

	// Complete the backup session
	err = session.Complete()
	if err != nil {
		t.Fatalf("Failed to complete backup session: %v", err)
	}

	// Test restore
	restoreJob := &backup.RestoreJob{
		ID:       "restore-1",
		BackupID: b.ID,
		Targets: []*backup.RestoreTarget{
			{
				SourceID:      "target-1",
				DestinationID: "vm-456",
				Type:          "vm",
				State:         backup.RestorePending,
			},
		},
		TenantID: "tenant-1",
	}

	err = provider.RestoreBackup(ctx, restoreJob)
	if err != nil {
		t.Fatalf("Failed to restore backup: %v", err)
	}

	if restoreJob.Targets[0].State != backup.RestoreCompleted {
		t.Errorf("Expected restore target state to be RestoreCompleted, got '%s'", restoreJob.Targets[0].State)
	}

	// Test deleting the backup
	err = provider.DeleteBackup(ctx, b.ID)
	if err != nil {
		t.Fatalf("Failed to delete backup: %v", err)
	}

	// Verify the backup was deleted
	_, err = provider.GetBackup(ctx, b.ID)
	if err == nil {
		t.Errorf("Expected error when getting deleted backup, got nil")
	}
}

func TestCalculateBackupSize(t *testing.T) {
	// Create a temporary directory for testing
	baseDir, err := os.MkdirTemp("", "novacron-test-size")
	if err != nil {
		t.Fatalf("Failed to create temp directory: %v", err)
	}
	defer os.RemoveAll(baseDir)

	// Create the provider
	provider, err := NewLocalStorageProvider("test-size", "Test Size", baseDir)
	if err != nil {
		t.Fatalf("Failed to create provider: %v", err)
	}

	// Create a test directory
	testDir := filepath.Join(baseDir, "test-dir")
	if err := os.MkdirAll(testDir, 0755); err != nil {
		t.Fatalf("Failed to create test directory: %v", err)
	}

	// Create some test files
	file1 := filepath.Join(testDir, "file1.txt")
	if err := os.WriteFile(file1, []byte("file1 content"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	file2 := filepath.Join(testDir, "file2.txt")
	if err := os.WriteFile(file2, []byte("file2 longer content"), 0644); err != nil {
		t.Fatalf("Failed to create test file: %v", err)
	}

	// Test calculateBackupSize
	size := provider.calculateBackupSize(testDir)
	expectedSize := int64(len("file1 content") + len("file2 longer content"))
	if size != expectedSize {
		t.Errorf("Expected backup size to be %d, got %d", expectedSize, size)
	}
}
