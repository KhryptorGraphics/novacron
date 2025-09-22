package integration

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
)

// MockFailingBackupProvider is a mock that fails on delete operations
type MockFailingBackupProvider struct {
	mock.Mock
}

func (m *MockFailingBackupProvider) CreateBackup(ctx context.Context, backup *backup.Backup) error {
	args := m.Called(ctx, backup)
	return args.Error(0)
}

func (m *MockFailingBackupProvider) GetBackup(ctx context.Context, backupID string) (*backup.Backup, error) {
	args := m.Called(ctx, backupID)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*backup.Backup), args.Error(1)
}

func (m *MockFailingBackupProvider) ListBackups(ctx context.Context) ([]*backup.Backup, error) {
	args := m.Called(ctx)
	return args.Get(0).([]*backup.Backup), args.Error(1)
}

func (m *MockFailingBackupProvider) DeleteBackup(ctx context.Context, backupID string) error {
	args := m.Called(ctx, backupID)
	return args.Error(0)
}

func (m *MockFailingBackupProvider) UpdateBackup(ctx context.Context, backup *backup.Backup) error {
	args := m.Called(ctx, backup)
	return args.Error(0)
}

func (m *MockFailingBackupProvider) GetStorageUsage(ctx context.Context) (int64, error) {
	args := m.Called(ctx)
	return args.Get(0).(int64), args.Error(1)
}

func (m *MockFailingBackupProvider) ValidateBackup(ctx context.Context, backupID string) error {
	args := m.Called(ctx, backupID)
	return args.Error(0)
}

func (m *MockFailingBackupProvider) RestoreBackup(ctx context.Context, backupID string, targetPath string) error {
	args := m.Called(ctx, backupID, targetPath)
	return args.Error(0)
}

func (m *MockFailingBackupProvider) GetProviderType() string {
	return "mock"
}

func (m *MockFailingBackupProvider) GetCapabilities() []string {
	return []string{"backup", "restore", "delete"}
}

func TestBackupDeletionRollback(t *testing.T) {
	ctx := context.Background()
	
	// Create backup manager
	manager := backup.NewBackupManager()
	
	// Create mock provider that will fail on delete
	mockProvider := new(MockFailingBackupProvider)
	manager.RegisterProvider("mock", mockProvider)
	
	// Create parent backup
	parentBackup := &backup.Backup{
		ID:        "parent-backup",
		VMID:      "vm-123",
		TenantID:  "tenant-456",
		Type:      backup.FullBackup,
		State:     backup.BackupCompleted,
		CreatedAt: time.Now().Add(-2 * time.Hour),
		Size:      1024 * 1024 * 100, // 100MB
	}
	
	// Create child backup that references parent
	childBackup := &backup.Backup{
		ID:        "child-backup",
		VMID:      "vm-123",
		TenantID:  "tenant-456",
		Type:      backup.IncrementalBackup,
		State:     backup.BackupCompleted,
		ParentID:  "parent-backup",
		CreatedAt: time.Now().Add(-1 * time.Hour),
		Size:      1024 * 1024 * 10, // 10MB
	}
	
	// Create another child backup
	childBackup2 := &backup.Backup{
		ID:        "child-backup-2",
		VMID:      "vm-123",
		TenantID:  "tenant-456",
		Type:      backup.IncrementalBackup,
		State:     backup.BackupCompleted,
		ParentID:  "parent-backup",
		CreatedAt: time.Now(),
		Size:      1024 * 1024 * 5, // 5MB
	}
	
	// Register backups with manager
	err := manager.CreateBackup(ctx, parentBackup)
	assert.NoError(t, err)
	
	err = manager.CreateBackup(ctx, childBackup)
	assert.NoError(t, err)
	
	err = manager.CreateBackup(ctx, childBackup2)
	assert.NoError(t, err)
	
	// Set up mock to return backups when queried
	mockProvider.On("GetBackup", ctx, "parent-backup").Return(parentBackup, nil)
	mockProvider.On("GetBackup", ctx, "child-backup").Return(childBackup, nil)
	mockProvider.On("GetBackup", ctx, "child-backup-2").Return(childBackup2, nil)
	
	t.Run("Provider delete failure should not modify chain", func(t *testing.T) {
		// Configure mock to fail on delete
		mockProvider.On("DeleteBackup", ctx, "parent-backup").Return(errors.New("provider delete failed"))
		
		// Attempt to delete parent backup
		err := manager.DeleteBackup(ctx, "parent-backup")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "provider delete failed")
		
		// Verify that child backups still reference the parent
		child1, err := manager.GetBackup("child-backup")
		assert.NoError(t, err)
		assert.Equal(t, "parent-backup", child1.ParentID)
		
		child2, err := manager.GetBackup("child-backup-2")
		assert.NoError(t, err)
		assert.Equal(t, "parent-backup", child2.ParentID)
		
		// Parent should still exist
		parent, err := manager.GetBackup("parent-backup")
		assert.NoError(t, err)
		assert.NotNil(t, parent)
	})
	
	t.Run("Successful delete should update chain", func(t *testing.T) {
		// Create a new isolated test case
		manager2 := backup.NewBackupManager()
		mockProvider2 := new(MockFailingBackupProvider)
		manager2.RegisterProvider("mock", mockProvider2)
		
		// Create backups
		parent := &backup.Backup{
			ID:        "parent-2",
			VMID:      "vm-456",
			TenantID:  "tenant-789",
			Type:      backup.FullBackup,
			State:     backup.BackupCompleted,
			CreatedAt: time.Now().Add(-2 * time.Hour),
		}
		
		child := &backup.Backup{
			ID:        "child-3",
			VMID:      "vm-456",
			TenantID:  "tenant-789",
			Type:      backup.IncrementalBackup,
			State:     backup.BackupCompleted,
			ParentID:  "parent-2",
			CreatedAt: time.Now().Add(-1 * time.Hour),
		}
		
		grandchild := &backup.Backup{
			ID:        "grandchild-1",
			VMID:      "vm-456",
			TenantID:  "tenant-789",
			Type:      backup.IncrementalBackup,
			State:     backup.BackupCompleted,
			ParentID:  "child-3",
			CreatedAt: time.Now(),
		}
		
		// Register backups
		manager2.CreateBackup(ctx, parent)
		manager2.CreateBackup(ctx, child)
		manager2.CreateBackup(ctx, grandchild)
		
		// Configure mock to succeed on delete
		mockProvider2.On("GetBackup", ctx, "child-3").Return(child, nil)
		mockProvider2.On("DeleteBackup", ctx, "child-3").Return(nil)
		
		// Delete middle backup (child)
		err := manager2.DeleteBackup(ctx, "child-3")
		assert.NoError(t, err)
		
		// Verify grandchild now references the parent directly
		gc, err := manager2.GetBackup("grandchild-1")
		assert.NoError(t, err)
		assert.Equal(t, "parent-2", gc.ParentID, "Grandchild should now reference the parent after middle backup deletion")
		
		// Child should be deleted
		_, err = manager2.GetBackup("child-3")
		assert.Error(t, err)
	})
}

// TestBackupChainIntegrity tests that backup chains maintain integrity
func TestBackupChainIntegrity(t *testing.T) {
	ctx := context.Background()
	manager := backup.NewBackupManager()
	
	// Create a chain of backups
	fullBackup := &backup.Backup{
		ID:        "full-1",
		VMID:      "vm-chain",
		TenantID:  "tenant-chain",
		Type:      backup.FullBackup,
		State:     backup.BackupCompleted,
		CreatedAt: time.Now().Add(-4 * time.Hour),
	}
	
	inc1 := &backup.Backup{
		ID:        "inc-1",
		VMID:      "vm-chain",
		TenantID:  "tenant-chain",
		Type:      backup.IncrementalBackup,
		State:     backup.BackupCompleted,
		ParentID:  "full-1",
		CreatedAt: time.Now().Add(-3 * time.Hour),
	}
	
	inc2 := &backup.Backup{
		ID:        "inc-2",
		VMID:      "vm-chain",
		TenantID:  "tenant-chain",
		Type:      backup.IncrementalBackup,
		State:     backup.BackupCompleted,
		ParentID:  "inc-1",
		CreatedAt: time.Now().Add(-2 * time.Hour),
	}
	
	inc3 := &backup.Backup{
		ID:        "inc-3",
		VMID:      "vm-chain",
		TenantID:  "tenant-chain",
		Type:      backup.IncrementalBackup,
		State:     backup.BackupCompleted,
		ParentID:  "inc-2",
		CreatedAt: time.Now().Add(-1 * time.Hour),
	}
	
	// Create all backups
	assert.NoError(t, manager.CreateBackup(ctx, fullBackup))
	assert.NoError(t, manager.CreateBackup(ctx, inc1))
	assert.NoError(t, manager.CreateBackup(ctx, inc2))
	assert.NoError(t, manager.CreateBackup(ctx, inc3))
	
	// Verify initial chain
	b3, _ := manager.GetBackup("inc-3")
	assert.Equal(t, "inc-2", b3.ParentID)
	
	b2, _ := manager.GetBackup("inc-2")
	assert.Equal(t, "inc-1", b2.ParentID)
	
	b1, _ := manager.GetBackup("inc-1")
	assert.Equal(t, "full-1", b1.ParentID)
	
	// The chain should be maintained through operations
	// This test verifies the chain structure is preserved
}