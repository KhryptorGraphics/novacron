# Backup and Disaster Recovery

This package provides comprehensive backup and disaster recovery capabilities for NovaCron, including scheduled backups, retention policies, and point-in-time recovery options.

## Features

- **Multiple Backup Types**: Full, incremental, and differential backups
- **Scheduling**: Flexible scheduling with cron-style expressions
- **Retention Policies**: Policy-based retention with daily, weekly, monthly options
- **Multiple Storage Backends**: Local, object storage, block storage, and file storage
- **Encryption**: Secure backups with AES encryption
- **Compression**: Reduce storage requirements with configurable compression
- **Multi-tenant**: Full tenant isolation for backups and restores
- **Point-in-Time Recovery**: Restore to specific points in time
- **Validation**: Verify backup integrity before restores

## Components

### Backup Manager

The `BackupManager` is the central component that manages backup and restore operations:

- Creating and scheduling backup jobs
- Managing backup targets and storage configurations
- Executing backup and restore operations
- Applying retention policies
- Tracking backup metadata and status

### Backup Providers

Backup providers implement the `BackupProvider` interface to support different storage backends:

- **Local Storage Provider**: For local file system backups
- **Object Storage Provider**: For S3-compatible storage
- **Block Storage Provider**: For volume-based backups
- **File Storage Provider**: For NFS and SMB-based backup storage

### Scheduler

The scheduler component manages automated backup execution:

- Tracking scheduled backup jobs
- Executing backups at the scheduled time
- Handling job dependencies and conflicts
- Managing job queues and priorities

## Usage Examples

### Creating a Backup Job

```go
// Create a backup manager
manager := backup.NewBackupManager()

// Register providers
manager.RegisterProvider(NewLocalStorageProvider())
manager.RegisterProvider(NewObjectStorageProvider())

// Define backup targets
targets := []*backup.BackupTarget{
    {
        ID:         "vm-1",
        Name:       "Web Server VM",
        Type:       "vm",
        ResourceID: "vm-12345",
        TenantID:   "tenant-1",
    },
}

// Create backup job
job := &backup.BackupJob{
    ID:          "daily-backup-1",
    Name:        "Daily VM Backup",
    Description: "Daily backup of critical VMs",
    Type:        backup.FullBackup,
    Targets:     targets,
    Storage: &backup.StorageConfig{
        Type:            backup.ObjectStorage,
        Config:          map[string]interface{}{"bucket": "vm-backups"},
        Encryption:      true,
        EncryptionKeyID: "encryption-key-1",
        Compression:     true,
        CompressionLevel: 6,
    },
    Schedule: &backup.Schedule{
        Type:       "cron",
        Expression: "0 2 * * *", // Daily at 2 AM
        TimeZone:   "UTC",
    },
    Retention: &backup.RetentionPolicy{
        KeepLast:    30,
        KeepDaily:   7,
        KeepWeekly:  4,
        KeepMonthly: 3,
    },
    Enabled:  true,
    TenantID: "tenant-1",
}

// Add job to manager
if err := manager.CreateBackupJob(job); err != nil {
    log.Fatalf("Failed to create backup job: %v", err)
}

// Start the backup manager (and scheduler)
if err := manager.Start(); err != nil {
    log.Fatalf("Failed to start backup manager: %v", err)
}
```

### Running a Backup Manually

```go
// Run a backup job immediately
backup, err := manager.RunBackupJob(context.Background(), "daily-backup-1")
if err != nil {
    log.Fatalf("Failed to run backup: %v", err)
}

log.Printf("Backup created with ID: %s", backup.ID)
```

### Restoring from a Backup

```go
// Create a restore job
restoreJob := &backup.RestoreJob{
    ID:       "restore-1",
    Name:     "VM Restore",
    BackupID: "backup-12345",
    Targets: []*backup.RestoreTarget{
        {
            SourceID:      "vm-1",
            DestinationID: "vm-5678",
            Type:          "vm",
        },
    },
    Options: &backup.RestoreOptions{
        OverwriteExisting:      false,
        RestorePermissions:     true,
        ValidateBeforeRestore:  true,
    },
    TenantID: "tenant-1",
}

// Start restore
if err := manager.CreateRestoreJob(context.Background(), restoreJob); err != nil {
    log.Fatalf("Failed to create restore job: %v", err)
}

// Check restore status
status, err := manager.GetRestoreJob("restore-1")
if err != nil {
    log.Fatalf("Failed to get restore status: %v", err)
}

log.Printf("Restore status: %s", status.State)
```

## Integration Points

The backup system integrates with other NovaCron components:

- **VM Manager**: For VM-level backups and restores
- **Storage Manager**: For volume and storage management
- **Network Manager**: For network configuration during restores
- **Security Manager**: For encryption key management and access control

## Performance Considerations

The backup system is designed for performance and efficiency:

1. **Block-level Deduplication**: Only storing unique blocks to save space
2. **Parallel Processing**: Multiple simultaneous backup and restore operations
3. **Bandwidth Throttling**: Control backup impact on network performance
4. **Incremental Backups**: Only backing up changed data after initial full backup
5. **Stream Processing**: Streaming large files to avoid high memory usage

## Disaster Recovery Planning

Beyond simple backups, the system supports comprehensive disaster recovery:

1. **Recovery Point Objectives (RPO)**: Define acceptable data loss
2. **Recovery Time Objectives (RTO)**: Define acceptable downtime
3. **Disaster Recovery Testing**: Regular automated testing of recovery processes
4. **Multi-site Replication**: Replicating backups across multiple locations
5. **Failover Planning**: Automated recovery procedures for rapid recovery

## Future Enhancements

- **Application-aware Backups**: Coordinate with applications for consistent backups
- **Database Integration**: Specialized handling for database backups
- **Continuous Data Protection**: Real-time backup for critical systems
- **Automated Recovery Testing**: Regularly test restores without affecting production
- **Cross-Cloud Recovery**: Restore to different cloud providers
