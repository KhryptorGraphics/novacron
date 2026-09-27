package integration

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptedBackupProvider is a backup.BackupProvider whose CreateBackup returns a
// backup pre-registered per job ID and whose DeleteBackup can be made to fail.
type scriptedBackupProvider struct {
	mu        sync.Mutex
	byJob     map[string]*backup.Backup
	backups   map[string]*backup.Backup
	deleteErr map[string]error
	deleted   []string
}

func newScriptedBackupProvider() *scriptedBackupProvider {
	return &scriptedBackupProvider{
		byJob:     make(map[string]*backup.Backup),
		backups:   make(map[string]*backup.Backup),
		deleteErr: make(map[string]error),
	}
}

func (p *scriptedBackupProvider) ID() string               { return "scripted" }
func (p *scriptedBackupProvider) Name() string             { return "Scripted Provider" }
func (p *scriptedBackupProvider) Type() backup.StorageType { return backup.LocalStorage }

func (p *scriptedBackupProvider) CreateBackup(ctx context.Context, job *backup.BackupJob) (*backup.Backup, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	b, ok := p.byJob[job.ID]
	if !ok {
		return nil, fmt.Errorf("no scripted backup for job %s", job.ID)
	}
	b.JobID = job.ID
	b.TenantID = job.TenantID
	p.backups[b.ID] = b
	return b, nil
}

func (p *scriptedBackupProvider) DeleteBackup(ctx context.Context, backupID string) error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if err := p.deleteErr[backupID]; err != nil {
		return err
	}
	delete(p.backups, backupID)
	p.deleted = append(p.deleted, backupID)
	return nil
}

func (p *scriptedBackupProvider) RestoreBackup(ctx context.Context, job *backup.RestoreJob) error {
	return nil
}

func (p *scriptedBackupProvider) ListBackups(ctx context.Context, filter map[string]interface{}) ([]*backup.Backup, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([]*backup.Backup, 0, len(p.backups))
	for _, b := range p.backups {
		out = append(out, b)
	}
	return out, nil
}

func (p *scriptedBackupProvider) GetBackup(ctx context.Context, backupID string) (*backup.Backup, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	b, ok := p.backups[backupID]
	if !ok {
		return nil, fmt.Errorf("backup %s not found", backupID)
	}
	return b, nil
}

func (p *scriptedBackupProvider) ValidateBackup(ctx context.Context, backupID string) error {
	return nil
}

// runScriptedBackup registers a job whose provider run yields the given backup
// and executes it through the manager, so the backup enters the manager's
// registry the same way production backups do.
func runScriptedBackup(t *testing.T, ctx context.Context, manager *backup.BackupManager, provider *scriptedBackupProvider, tenantID string, b *backup.Backup) {
	t.Helper()
	jobID := "job-" + b.ID
	provider.mu.Lock()
	provider.byJob[jobID] = b
	provider.mu.Unlock()

	require.NoError(t, manager.CreateBackupJob(&backup.BackupJob{
		ID:       jobID,
		Name:     jobID,
		Type:     b.Type,
		Enabled:  true,
		TenantID: tenantID,
		Storage:  &backup.StorageConfig{Type: backup.LocalStorage},
	}))
	got, err := manager.RunBackupJob(ctx, jobID)
	require.NoError(t, err)
	require.Equal(t, b.ID, got.ID)
}

func TestBackupDeletionRollback(t *testing.T) {
	ctx := context.Background()
	now := time.Now()

	t.Run("Provider delete failure should not modify chain", func(t *testing.T) {
		manager := backup.NewBackupManager()
		provider := newScriptedBackupProvider()
		require.NoError(t, manager.RegisterProvider(provider))

		parent := &backup.Backup{ID: "parent-backup", VMID: "vm-123", Type: backup.FullBackup, State: backup.BackupCompleted, StartedAt: now.Add(-2 * time.Hour), Size: 100 << 20}
		child1 := &backup.Backup{ID: "child-backup", VMID: "vm-123", Type: backup.IncrementalBackup, State: backup.BackupCompleted, ParentID: "parent-backup", StartedAt: now.Add(-time.Hour), Size: 10 << 20}
		child2 := &backup.Backup{ID: "child-backup-2", VMID: "vm-123", Type: backup.IncrementalBackup, State: backup.BackupCompleted, ParentID: "parent-backup", StartedAt: now, Size: 5 << 20}
		for _, b := range []*backup.Backup{parent, child1, child2} {
			runScriptedBackup(t, ctx, manager, provider, "tenant-456", b)
		}

		provider.deleteErr["parent-backup"] = errors.New("provider delete failed")

		err := manager.DeleteBackup(ctx, "parent-backup")
		require.Error(t, err)
		assert.Contains(t, err.Error(), "provider delete failed")

		// Children still reference the parent and the parent still exists.
		for _, id := range []string{"child-backup", "child-backup-2"} {
			c, err := manager.GetBackup(id)
			require.NoError(t, err)
			assert.Equal(t, "parent-backup", c.ParentID)
		}
		p, err := manager.GetBackup("parent-backup")
		require.NoError(t, err)
		assert.NotNil(t, p)
		assert.Empty(t, provider.deleted, "provider must not record a delete that failed")
	})

	t.Run("Successful delete should update chain", func(t *testing.T) {
		manager := backup.NewBackupManager()
		provider := newScriptedBackupProvider()
		require.NoError(t, manager.RegisterProvider(provider))

		parent := &backup.Backup{ID: "parent-2", VMID: "vm-456", Type: backup.FullBackup, State: backup.BackupCompleted, StartedAt: now.Add(-2 * time.Hour)}
		child := &backup.Backup{ID: "child-3", VMID: "vm-456", Type: backup.IncrementalBackup, State: backup.BackupCompleted, ParentID: "parent-2", StartedAt: now.Add(-time.Hour)}
		grandchild := &backup.Backup{ID: "grandchild-1", VMID: "vm-456", Type: backup.IncrementalBackup, State: backup.BackupCompleted, ParentID: "child-3", StartedAt: now}
		for _, b := range []*backup.Backup{parent, child, grandchild} {
			runScriptedBackup(t, ctx, manager, provider, "tenant-789", b)
		}

		// Delete the middle backup.
		require.NoError(t, manager.DeleteBackup(ctx, "child-3"))

		gc, err := manager.GetBackup("grandchild-1")
		require.NoError(t, err)
		assert.Equal(t, "parent-2", gc.ParentID, "grandchild should reference the parent after the middle backup is deleted")

		_, err = manager.GetBackup("child-3")
		assert.Error(t, err, "deleted backup must no longer be resolvable")
		assert.Equal(t, []string{"child-3"}, provider.deleted)

		tenantBackups, err := manager.ListBackups("tenant-789", "")
		require.NoError(t, err)
		ids := make([]string, 0, len(tenantBackups))
		for _, b := range tenantBackups {
			ids = append(ids, b.ID)
		}
		assert.ElementsMatch(t, []string{"parent-2", "grandchild-1"}, ids)
	})
}

// TestBackupChainIntegrity verifies a full→incremental chain keeps its parent
// links when registered through the manager.
func TestBackupChainIntegrity(t *testing.T) {
	ctx := context.Background()
	manager := backup.NewBackupManager()
	provider := newScriptedBackupProvider()
	require.NoError(t, manager.RegisterProvider(provider))

	now := time.Now()
	chain := []*backup.Backup{
		{ID: "full-1", VMID: "vm-chain", Type: backup.FullBackup, State: backup.BackupCompleted, StartedAt: now.Add(-4 * time.Hour)},
		{ID: "inc-1", VMID: "vm-chain", Type: backup.IncrementalBackup, State: backup.BackupCompleted, ParentID: "full-1", StartedAt: now.Add(-3 * time.Hour)},
		{ID: "inc-2", VMID: "vm-chain", Type: backup.IncrementalBackup, State: backup.BackupCompleted, ParentID: "inc-1", StartedAt: now.Add(-2 * time.Hour)},
		{ID: "inc-3", VMID: "vm-chain", Type: backup.IncrementalBackup, State: backup.BackupCompleted, ParentID: "inc-2", StartedAt: now.Add(-time.Hour)},
	}
	for _, b := range chain {
		runScriptedBackup(t, ctx, manager, provider, "tenant-chain", b)
	}

	expectedParent := map[string]string{"full-1": "", "inc-1": "full-1", "inc-2": "inc-1", "inc-3": "inc-2"}
	for id, parent := range expectedParent {
		b, err := manager.GetBackup(id)
		require.NoError(t, err)
		assert.Equal(t, parent, b.ParentID, "parent link of %s", id)
	}

	// Deleting the head of the chain re-parents its only child to the root.
	require.NoError(t, manager.DeleteBackup(ctx, "inc-1"))
	b2, err := manager.GetBackup("inc-2")
	require.NoError(t, err)
	assert.Equal(t, "full-1", b2.ParentID)
	b3, err := manager.GetBackup("inc-3")
	require.NoError(t, err)
	assert.Equal(t, "inc-2", b3.ParentID)
}
