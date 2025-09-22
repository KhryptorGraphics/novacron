package integration

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/mock"
)

// MockBackupProvider is a mock implementation of BackupProvider for testing
type MockBackupProvider struct {
	mock.Mock
}

func (m *MockBackupProvider) ID() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBackupProvider) Name() string {
	args := m.Called()
	return args.String(0)
}

func (m *MockBackupProvider) Type() backup.StorageType {
	args := m.Called()
	return args.Get(0).(backup.StorageType)
}

func (m *MockBackupProvider) CreateBackup(ctx context.Context, job *backup.BackupJob) (*backup.Backup, error) {
	args := m.Called(ctx, job)
	return args.Get(0).(*backup.Backup), args.Error(1)
}

func (m *MockBackupProvider) DeleteBackup(ctx context.Context, backupID string) error {
	args := m.Called(ctx, backupID)
	return args.Error(0)
}

func (m *MockBackupProvider) RestoreBackup(ctx context.Context, job *backup.RestoreJob) error {
	args := m.Called(ctx, job)
	return args.Error(0)
}

func (m *MockBackupProvider) ListBackups(ctx context.Context, filter map[string]interface{}) ([]*backup.Backup, error) {
	args := m.Called(ctx, filter)
	return args.Get(0).([]*backup.Backup), args.Error(1)
}

func (m *MockBackupProvider) GetBackup(ctx context.Context, backupID string) (*backup.Backup, error) {
	args := m.Called(ctx, backupID)
	return args.Get(0).(*backup.Backup), args.Error(1)
}

func (m *MockBackupProvider) ValidateBackup(ctx context.Context, backupID string) error {
	args := m.Called(ctx, backupID)
	return args.Error(0)
}

// TestBackupDeletion tests the newly implemented backup deletion functionality
func TestBackupDeletion(t *testing.T) {
	manager := backup.NewBackupManager()
	mockProvider := new(MockBackupProvider)
	
	// Setup mock provider
	mockProvider.On("ID").Return("mock-provider")
	mockProvider.On("Name").Return("Mock Provider")
	mockProvider.On("Type").Return(backup.LocalStorage)
	
	err := manager.RegisterProvider(mockProvider)
	require.NoError(t, err)
	
	ctx := context.Background()
	
	t.Run("SuccessfulDeletion", func(t *testing.T) {
		backupID := "test-backup-1"
		tenantID := "tenant-1"
		
		// Create a test backup
		testBackup := &backup.Backup{
			ID:        backupID,
			JobID:     "job-1",
			Type:      backup.FullBackup,
			State:     backup.BackupCompleted,
			Size:      1024,
			StartedAt: time.Now().Add(-1 * time.Hour),
			CompletedAt: time.Now().Add(-30 * time.Minute),
			TenantID:  tenantID,
		}
		
		// Manually add backup to manager (simulating existing backup)
		manager.AddBackupForTest(testBackup)
		
		// Setup mock expectations
		mockProvider.On("GetBackup", ctx, backupID).Return(testBackup, nil)
		mockProvider.On("DeleteBackup", ctx, backupID).Return(nil)
		
		// Test deletion
		err := manager.DeleteBackup(ctx, backupID)
		assert.NoError(t, err)
		
		// Verify backup was removed from manager
		_, err = manager.GetBackup(backupID)
		assert.Error(t, err)
		
		// Verify tenant backup list was updated (use public API to check)
		backupsForTenant, err := manager.ListBackups(tenantID, "")
		assert.NoError(t, err)
		for _, b := range backupsForTenant {
			assert.NotEqual(t, backupID, b.ID)
		}
		
		mockProvider.AssertExpectations(t)
	})
	
	t.Run("BackupNotFound", func(t *testing.T) {
		err := manager.DeleteBackup(ctx, "nonexistent-backup")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "does not exist")
	})
	
	t.Run("ProviderDeletionFailure", func(t *testing.T) {
		backupID := "test-backup-2"
		tenantID := "tenant-2"
		
		testBackup := &backup.Backup{
			ID:       backupID,
			JobID:    "job-2",
			Type:     backup.IncrementalBackup,
			State:    backup.BackupCompleted,
			TenantID: tenantID,
		}
		
		manager.AddBackupForTest(testBackup)
		
		// Setup mock to fail deletion
		mockProvider.On("GetBackup", ctx, backupID).Return(testBackup, nil)
		mockProvider.On("DeleteBackup", ctx, backupID).Return(fmt.Errorf("provider deletion failed"))
		
		err := manager.DeleteBackup(ctx, backupID)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed to delete backup data")
		
		// Verify backup still exists in manager
		_, err = manager.GetBackup(backupID)
		assert.NoError(t, err)
		
		mockProvider.AssertExpectations(t)
	})
	
	t.Run("BackupChainCleanup", func(t *testing.T) {
		parentBackupID := "parent-backup"
		childBackupID := "child-backup"
		tenantID := "tenant-3"
		
		parentBackup := &backup.Backup{
			ID:       parentBackupID,
			JobID:    "job-3",
			Type:     backup.FullBackup,
			State:    backup.BackupCompleted,
			TenantID: tenantID,
		}
		
		childBackup := &backup.Backup{
			ID:       childBackupID,
			JobID:    "job-3",
			Type:     backup.IncrementalBackup,
			State:    backup.BackupCompleted,
			ParentID: parentBackupID,
			TenantID: tenantID,
		}
		
		manager.AddBackupForTest(parentBackup)
		manager.AddBackupForTest(childBackup)
		
		// Setup mock expectations for parent deletion
		mockProvider.On("GetBackup", ctx, parentBackupID).Return(parentBackup, nil)
		mockProvider.On("DeleteBackup", ctx, parentBackupID).Return(nil)
		
		// Delete parent backup (should update child backup chain)
		err := manager.DeleteBackup(ctx, parentBackupID)
		assert.NoError(t, err)
		
		// Verify parent was deleted
		_, err = manager.GetBackup(parentBackupID)
		assert.Error(t, err)
		
		// Verify child backup chain was updated (parent should be empty now)
		remainingChild, err := manager.GetBackup(childBackupID)
		require.NoError(t, err)
		assert.Empty(t, remainingChild.ParentID)
		
		mockProvider.AssertExpectations(t)
	})
}

// TestBackupVerification tests the newly implemented backup verification functionality
func TestBackupVerification(t *testing.T) {
	manager := backup.NewBackupManager()
	mockProvider := new(MockBackupProvider)
	
	mockProvider.On("ID").Return("mock-provider")
	mockProvider.On("Name").Return("Mock Provider")
	mockProvider.On("Type").Return(backup.LocalStorage)
	
	err := manager.RegisterProvider(mockProvider)
	require.NoError(t, err)
	
	ctx := context.Background()
	
	t.Run("ValidBackup", func(t *testing.T) {
		backupID := "valid-backup"
		testBackup := &backup.Backup{
			ID:       backupID,
			JobID:    "job-1",
			Type:     backup.FullBackup,
			State:    backup.BackupCompleted,
			TenantID: "tenant-1",
		}
		
		manager.AddBackupForTest(testBackup)
		
		// Setup successful validation
		mockProvider.On("GetBackup", ctx, backupID).Return(testBackup, nil)
		mockProvider.On("ValidateBackup", ctx, backupID).Return(nil)
		
		result, err := manager.VerifyBackup(ctx, backupID)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, backupID, result.BackupID)
		assert.Equal(t, "valid", result.Status)
		assert.Equal(t, 1, result.CheckedItems)
		assert.Empty(t, result.ErrorsFound)
		
		mockProvider.AssertExpectations(t)
	})
	
	t.Run("CorruptedBackup", func(t *testing.T) {
		backupID := "corrupted-backup"
		testBackup := &backup.Backup{
			ID:       backupID,
			JobID:    "job-2",
			Type:     backup.FullBackup,
			State:    backup.BackupCompleted,
			TenantID: "tenant-1",
		}
		
		manager.AddBackupForTest(testBackup)
		
		// Setup failed validation
		validationErr := fmt.Errorf("checksum mismatch detected")
		mockProvider.On("GetBackup", ctx, backupID).Return(testBackup, nil)
		mockProvider.On("ValidateBackup", ctx, backupID).Return(validationErr)
		
		result, err := manager.VerifyBackup(ctx, backupID)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, backupID, result.BackupID)
		assert.Equal(t, "corrupted", result.Status)
		assert.Len(t, result.ErrorsFound, 1)
		assert.Contains(t, result.ErrorsFound[0], "checksum mismatch")
		
		mockProvider.AssertExpectations(t)
	})
	
	t.Run("IncrementalBackupChainVerification", func(t *testing.T) {
		parentID := "parent-backup"
		childID := "child-backup"
		
		parentBackup := &backup.Backup{
			ID:       parentID,
			JobID:    "job-3",
			Type:     backup.FullBackup,
			State:    backup.BackupCompleted,
			TenantID: "tenant-1",
		}
		
		childBackup := &backup.Backup{
			ID:       childID,
			JobID:    "job-3",
			Type:     backup.IncrementalBackup,
			State:    backup.BackupCompleted,
			ParentID: parentID,
			TenantID: "tenant-1",
		}
		
		manager.AddBackupForTest(parentBackup)
		manager.AddBackupForTest(childBackup)
		
		// Setup validation for child backup
		mockProvider.On("GetBackup", ctx, childID).Return(childBackup, nil)
		mockProvider.On("ValidateBackup", ctx, childID).Return(nil)
		
		result, err := manager.VerifyBackup(ctx, childID)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, childID, result.BackupID)
		assert.Equal(t, "valid", result.Status)
		assert.Equal(t, 2, result.CheckedItems) // Basic validation + chain verification
		assert.Contains(t, result.Details, "chain_verified")
		
		mockProvider.AssertExpectations(t)
	})
	
	t.Run("BrokenBackupChain", func(t *testing.T) {
		orphanID := "orphan-backup"
		
		orphanBackup := &backup.Backup{
			ID:       orphanID,
			JobID:    "job-4",
			Type:     backup.IncrementalBackup,
			State:    backup.BackupCompleted,
			ParentID: "missing-parent",
			TenantID: "tenant-1",
		}
		
		manager.AddBackupForTest(orphanBackup)
		
		// Setup validation
		mockProvider.On("GetBackup", ctx, orphanID).Return(orphanBackup, nil)
		mockProvider.On("ValidateBackup", ctx, orphanID).Return(nil)
		
		result, err := manager.VerifyBackup(ctx, orphanID)
		assert.NoError(t, err)
		assert.NotNil(t, result)
		assert.Equal(t, orphanID, result.BackupID)
		assert.Equal(t, "incomplete", result.Status)
		assert.Contains(t, result.ErrorsFound[0], "Chain verification failed")
		
		mockProvider.AssertExpectations(t)
	})
	
	t.Run("BackupNotFound", func(t *testing.T) {
		_, err := manager.VerifyBackup(ctx, "nonexistent-backup")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "does not exist")
	})
}

// TestListAllBackups tests the newly implemented list all backups functionality
func TestListAllBackups(t *testing.T) {
	manager := backup.NewBackupManager()
	ctx := context.Background()
	
	// Create test backups for multiple tenants
	tenant1Backups := []*backup.Backup{
		{
			ID:       "backup-1-1",
			JobID:    "job-1",
			Type:     backup.FullBackup,
			State:    backup.BackupCompleted,
			Size:     1024,
			StartedAt: time.Now().Add(-2 * time.Hour),
			CompletedAt: time.Now().Add(-90 * time.Minute),
			TenantID: "tenant-1",
		},
		{
			ID:       "backup-1-2",
			JobID:    "job-1",
			Type:     backup.IncrementalBackup,
			State:    backup.BackupCompleted,
			Size:     512,
			StartedAt: time.Now().Add(-1 * time.Hour),
			CompletedAt: time.Now().Add(-30 * time.Minute),
			TenantID: "tenant-1",
		},
	}
	
	tenant2Backups := []*backup.Backup{
		{
			ID:       "backup-2-1",
			JobID:    "job-2",
			Type:     backup.FullBackup,
			State:    backup.BackupCompleted,
			Size:     2048,
			StartedAt: time.Now().Add(-3 * time.Hour),
			CompletedAt: time.Now().Add(-2 * time.Hour),
			TenantID: "tenant-2",
		},
	}
	
	// Add backups to manager
	for _, b := range tenant1Backups {
		manager.AddBackupForTest(b)
	}
	for _, b := range tenant2Backups {
		manager.AddBackupForTest(b)
	}
	
	t.Run("ListAllBackupsSuccess", func(t *testing.T) {
		allBackups, err := manager.ListAllBackups(ctx)
		assert.NoError(t, err)
		assert.Len(t, allBackups, 3)
		
		// Verify all backups are included
		backupIDs := make([]string, len(allBackups))
		for i, backup := range allBackups {
			backupIDs[i] = backup.ID
		}
		
		assert.Contains(t, backupIDs, "backup-1-1")
		assert.Contains(t, backupIDs, "backup-1-2")
		assert.Contains(t, backupIDs, "backup-2-1")
	})
	
	t.Run("BackupInfoFields", func(t *testing.T) {
		allBackups, err := manager.ListAllBackups(ctx)
		assert.NoError(t, err)
		
		// Find a specific backup and verify its fields
		var foundBackup *backup.BackupInfo
		for _, b := range allBackups {
			if b.ID == "backup-1-1" {
				foundBackup = &b
				break
			}
		}
		
		require.NotNil(t, foundBackup)
		assert.Equal(t, "backup-1-1", foundBackup.ID)
		assert.Equal(t, "job-1", foundBackup.JobID)
		assert.Equal(t, backup.FullBackup, foundBackup.Type)
		assert.Equal(t, backup.BackupCompleted, foundBackup.State)
		assert.Equal(t, int64(1024), foundBackup.Size)
		assert.Equal(t, "tenant-1", foundBackup.TenantID)
		assert.NotNil(t, foundBackup.CompletedAt)
	})
	
	t.Run("EmptyBackupList", func(t *testing.T) {
		emptyManager := backup.NewBackupManager()
		allBackups, err := emptyManager.ListAllBackups(ctx)
		assert.NoError(t, err)
		assert.Empty(t, allBackups)
	})
}

// TestBackupAPIIntegration tests the backup API handlers
func TestBackupAPIIntegration(t *testing.T) {
	// This would test the API handlers, but since we don't have the full backup manager
	// with IncrementalBackupManager, this is a placeholder for API integration tests
	
	t.Run("DeleteBackupAPI", func(t *testing.T) {
		// TODO: Implement API integration tests once the backup API is fully integrated
		// This would test the DELETE /api/v1/backup/backups/{backup_id} endpoint
		t.Skip("API integration tests require full backup system setup")
	})
	
	t.Run("VerifyBackupAPI", func(t *testing.T) {
		// TODO: Implement API integration tests
		// This would test the POST /api/v1/backup/backups/{backup_id}/verify endpoint
		t.Skip("API integration tests require full backup system setup")
	})
	
	t.Run("ListAllBackupsAPI", func(t *testing.T) {
		// TODO: Implement API integration tests
		// This would test the GET /api/v1/backup/backups endpoint without vm_id filter
		t.Skip("API integration tests require full backup system setup")
	})
}

// TestBackupConcurrency tests concurrent backup operations
func TestBackupConcurrency(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping concurrent backup operations test in short mode")
	}
	
	manager := backup.NewBackupManager()
	mockProvider := new(MockBackupProvider)
	
	mockProvider.On("ID").Return("concurrent-provider")
	mockProvider.On("Name").Return("Concurrent Provider")
	mockProvider.On("Type").Return(backup.LocalStorage)
	
	err := manager.RegisterProvider(mockProvider)
	require.NoError(t, err)
	
	ctx := context.Background()
	numBackups := 10
	
	// Create multiple backups
	backupIDs := make([]string, numBackups)
	for i := 0; i < numBackups; i++ {
		backupID := fmt.Sprintf("concurrent-backup-%d", i)
		backupIDs[i] = backupID
		
		testBackup := &backup.Backup{
			ID:       backupID,
			JobID:    fmt.Sprintf("job-%d", i),
			Type:     backup.FullBackup,
			State:    backup.BackupCompleted,
			TenantID: "concurrent-tenant",
		}
		
		manager.AddBackupForTest(testBackup)
	}
	
	t.Run("ConcurrentVerification", func(t *testing.T) {
		// Setup mock expectations for all verifications
		for _, backupID := range backupIDs {
			backup, _ := manager.GetBackup(backupID)
			mockProvider.On("GetBackup", ctx, backupID).Return(backup, nil)
			mockProvider.On("ValidateBackup", ctx, backupID).Return(nil)
		}
		
		// Run verifications concurrently
		results := make(chan error, numBackups)
		for _, backupID := range backupIDs {
			go func(id string) {
				_, err := manager.VerifyBackup(ctx, id)
				results <- err
			}(backupID)
		}
		
		// Wait for all verifications
		for i := 0; i < numBackups; i++ {
			err := <-results
			assert.NoError(t, err)
		}
		
		mockProvider.AssertExpectations(t)
	})
	
	t.Run("ConcurrentDeletion", func(t *testing.T) {
		// Setup mock expectations for all deletions
		for _, backupID := range backupIDs {
			backup, _ := manager.GetBackup(backupID)
			mockProvider.On("GetBackup", ctx, backupID).Return(backup, nil)
			mockProvider.On("DeleteBackup", ctx, backupID).Return(nil)
		}
		
		// Run deletions concurrently
		results := make(chan error, numBackups)
		for _, backupID := range backupIDs {
			go func(id string) {
				results <- manager.DeleteBackup(ctx, id)
			}(backupID)
		}
		
		// Wait for all deletions
		for i := 0; i < numBackups; i++ {
			err := <-results
			assert.NoError(t, err)
		}
		
		// Verify all backups were deleted
		for _, backupID := range backupIDs {
			_, err := manager.GetBackup(backupID)
			assert.Error(t, err)
		}
		
		mockProvider.AssertExpectations(t)
	})
}

// TestBackupChainManagement tests backup chain management during deletions
func TestBackupChainManagement(t *testing.T) {
	manager := backup.NewBackupManager()
	
	// Create a backup chain: Full -> Inc1 -> Inc2
	fullBackup := &backup.Backup{
		ID:       "full-backup",
		JobID:    "chain-job",
		Type:     backup.FullBackup,
		State:    backup.BackupCompleted,
		TenantID: "chain-tenant",
	}
	
	inc1Backup := &backup.Backup{
		ID:       "inc1-backup",
		JobID:    "chain-job",
		Type:     backup.IncrementalBackup,
		State:    backup.BackupCompleted,
		ParentID: "full-backup",
		TenantID: "chain-tenant",
	}
	
	inc2Backup := &backup.Backup{
		ID:       "inc2-backup",
		JobID:    "chain-job",
		Type:     backup.IncrementalBackup,
		State:    backup.BackupCompleted,
		ParentID: "inc1-backup",
		TenantID: "chain-tenant",
	}
	
	// Add backups to manager
	manager.AddBackupForTest(fullBackup)
	manager.AddBackupForTest(inc1Backup)
	manager.AddBackupForTest(inc2Backup)
	
	t.Run("DeleteMiddleBackupUpdatesChain", func(t *testing.T) {
		mockProvider := new(MockBackupProvider)
		mockProvider.On("ID").Return("chain-provider")
		mockProvider.On("Name").Return("Chain Provider")
		mockProvider.On("Type").Return(backup.LocalStorage)
		
		err := manager.RegisterProvider(mockProvider)
		require.NoError(t, err)
		
		// Setup mock for inc1 deletion
		mockProvider.On("GetBackup", mock.Anything, "inc1-backup").Return(inc1Backup, nil)
		mockProvider.On("DeleteBackup", mock.Anything, "inc1-backup").Return(nil)
		
		// Delete middle backup (inc1)
		ctx := context.Background()
		err = manager.DeleteBackup(ctx, "inc1-backup")
		assert.NoError(t, err)
		
		// Verify inc2 now points to full backup
		updatedInc2, err := manager.GetBackup("inc2-backup")
		require.NoError(t, err)
		assert.Equal(t, "full-backup", updatedInc2.ParentID)
		
		// Verify inc1 was deleted
		_, err = manager.GetBackup("inc1-backup")
		assert.Error(t, err)
		
		mockProvider.AssertExpectations(t)
	})
}