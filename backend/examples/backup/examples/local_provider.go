package examples

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
	"github.com/khryptorgraphics/novacron/backend/core/backup/providers"
)

// RunLocalProviderExample demonstrates the use of the local storage provider
func RunLocalProviderExample() {
	// Create the backup manager
	manager := backup.NewBackupManager()

	// Create and register a local storage provider
	baseDir := filepath.Join(os.TempDir(), "novacron-backups")
	fmt.Printf("Using backup directory: %s\n", baseDir)

	localProvider, err := providers.NewLocalStorageProvider("local-fs", "Local Filesystem", baseDir)
	if err != nil {
		log.Fatalf("Failed to create local storage provider: %v", err)
	}

	// Register the provider with the backup manager
	if err := manager.RegisterProvider(localProvider); err != nil {
		log.Fatalf("Failed to register provider: %v", err)
	}

	// Start the backup manager
	if err := manager.Start(); err != nil {
		log.Fatalf("Failed to start backup manager: %v", err)
	}

	// Create a backup job
	job := &backup.BackupJob{
		ID:          "job-1",
		Name:        "Example Backup Job",
		Description: "A test backup job using the local storage provider",
		Type:        backup.FullBackup,
		Targets: []*backup.BackupTarget{
			{
				ID:         "target-1",
				Name:       "Test VM",
				Type:       "vm",
				ResourceID: "vm-123",
				TenantID:   "tenant-1",
				Metadata: map[string]string{
					"os":   "linux",
					"size": "10GB",
				},
			},
		},
		Storage: &backup.StorageConfig{
			Type:             backup.LocalStorage,
			Encryption:       true,
			EncryptionKeyID:  "key-1",
			Compression:      true,
			CompressionLevel: 6,
			Config: map[string]interface{}{
				"path": baseDir,
			},
		},
		Schedule: &backup.Schedule{
			Type:       "manual",
			Expression: "manual",
			TimeZone:   "UTC",
		},
		Retention: &backup.RetentionPolicy{
			KeepLast:  5,
			KeepDaily: 7,
		},
		Enabled:  true,
		TenantID: "tenant-1",
	}

	// Create the job
	if err := manager.CreateBackupJob(job); err != nil {
		log.Fatalf("Failed to create backup job: %v", err)
	}

	fmt.Printf("Created backup job: %s\n", job.ID)

	// Run the job
	ctx := context.Background()
	b, err := manager.RunBackupJob(ctx, job.ID)
	if err != nil {
		log.Fatalf("Failed to run backup job: %v", err)
	}

	fmt.Printf("Backup completed: %s (State: %s, Size: %d bytes)\n", b.ID, b.State, b.Size)

	// List backups
	backups, err := manager.ListBackups("tenant-1", "")
	if err != nil {
		log.Fatalf("Failed to list backups: %v", err)
	}

	fmt.Printf("Found %d backups\n", len(backups))
	for i, backup := range backups {
		fmt.Printf("%d. Backup ID: %s, Job: %s, State: %s, Started: %s\n",
			i+1, backup.ID, backup.JobID, backup.State, backup.StartedAt.Format(time.RFC3339))
	}

	// Create a restore job for the latest backup
	if len(backups) > 0 {
		latestBackup := backups[len(backups)-1]
		restoreJob := &backup.RestoreJob{
			ID:       "restore-1",
			Name:     "Example Restore Job",
			BackupID: latestBackup.ID,
			Targets: []*backup.RestoreTarget{
				{
					SourceID:      "target-1",
					DestinationID: "vm-456",
					Type:          "vm",
					State:         backup.RestorePending,
				},
			},
			State:    backup.RestorePending,
			TenantID: "tenant-1",
			Options: &backup.RestoreOptions{
				OverwriteExisting:     true,
				RestorePermissions:    true,
				ValidateBeforeRestore: true,
			},
		}

		// Create and run the restore job
		if err := manager.CreateRestoreJob(ctx, restoreJob); err != nil {
			log.Fatalf("Failed to create restore job: %v", err)
		}

		fmt.Printf("Restore completed for backup %s\n", latestBackup.ID)
	}

	// Stop the backup manager
	if err := manager.Stop(); err != nil {
		log.Fatalf("Failed to stop backup manager: %v", err)
	}

	fmt.Println("Example completed successfully")
}
