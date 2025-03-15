# Snapshot and Restore Capabilities

This package provides comprehensive VM snapshot and restore capabilities for NovaCron, including snapshot management, scheduling, consistency groups, and fast restore operations.

## Features

- **Multiple Snapshot Types**: Memory snapshots, disk-only snapshots, and application-consistent snapshots
- **Scheduled Snapshots**: Automate snapshots with flexible scheduling options
- **Retention Policies**: Define policies for automatic snapshot cleanup
- **Consistency Groups**: Create consistent snapshots across multiple VMs 
- **Fast Restore**: Quickly restore VMs from snapshots
- **Multi-tenant**: Full tenant isolation for snapshots
- **Incremental Snapshots**: Efficient storage with incremental snapshots
- **Quiesced Snapshots**: Application-consistent snapshots with VM quiescing

## Components

### Snapshot Manager

The `SnapshotManager` is the central component that manages snapshots:

- Creating and managing snapshots
- Scheduling automated snapshots
- Managing consistency groups
- Handling restore operations
- Applying retention policies

### Snapshot Types

The system supports multiple snapshot types:

- **Memory Snapshots**: Include VM memory state for quick recovery
- **Disk Snapshots**: Disk-only snapshots for storage efficiency
- **Application-Consistent Snapshots**: Coordinated with applications for data consistency

### Snapshot Scheduler

The scheduler component manages automated snapshot execution:

- Tracking scheduled snapshots
- Executing snapshots at the scheduled time
- Managing job retention policies
- Handling failure recovery

### Consistency Groups

Consistency groups ensure multiple VMs are snapshotted at the same logical point in time:

- Group definition with multiple VMs
- Atomic snapshot creation across VMs
- Coordinated restore capabilities
- Application-level consistency

## Usage Examples

### Creating a Snapshot

```go
// Create a snapshot manager
manager := snapshot.NewSnapshotManager()

// Register a provider
manager.RegisterProvider(NewKVMSnapshotProvider())

// Create a VM snapshot
snapshot, err := manager.CreateSnapshot(
    "vm-1234",
    "daily-backup",
    "Daily backup snapshot",
    snapshot.MemorySnapshot,
    map[string]interface{}{
        "quiesce": true,
    },
    "tenant-1",
    "admin",
)
if err != nil {
    log.Fatalf("Failed to create snapshot: %v", err)
}

log.Printf("Created snapshot with ID: %s", snapshot.ID)
```

### Creating a Scheduled Snapshot

```go
// Create a scheduled snapshot
scheduled := &snapshot.ScheduledSnapshot{
    ID:             "daily-vm-backup",
    Name:           "Daily VM Backup",
    Description:    "Daily backup of production VMs",
    VMID:           "vm-1234",
    Type:           snapshot.DiskSnapshot,
    QuiesceVM:      true,
    MemorySnapshot: false,
    Schedule: &snapshot.SnapshotSchedule{
        Type:       "cron",
        Expression: "0 2 * * *", // Daily at 2 AM
        TimeZone:   "UTC",
    },
    Retention: &snapshot.SnapshotRetention{
        MaxSnapshots: 7,
        RetentionType: "simple",
    },
    Enabled:  true,
    TenantID: "tenant-1",
}

// Add scheduled snapshot
if err := manager.CreateScheduledSnapshot(scheduled); err != nil {
    log.Fatalf("Failed to create scheduled snapshot: %v", err)
}

// Start snapshot manager to begin scheduling
if err := manager.Start(); err != nil {
    log.Fatalf("Failed to start snapshot manager: %v", err)
}
```

### Creating a Consistency Group

```go
// Create a consistency group
group := &snapshot.SnapshotConsistencyGroup{
    ID:          "db-cluster",
    Name:        "Database Cluster",
    Description: "Consistency group for database cluster VMs",
    VMIDs:       []string{"vm-db-1", "vm-db-2", "vm-db-3"},
    TenantID:    "tenant-1",
    Tags:        []string{"database", "production"},
}

// Add consistency group
if err := manager.CreateConsistencyGroup(group); err != nil {
    log.Fatalf("Failed to create consistency group: %v", err)
}

// Create consistency group snapshot
snapshots, err := manager.CreateConsistencyGroupSnapshot("db-cluster", "admin")
if err != nil {
    log.Fatalf("Failed to create consistency group snapshot: %v", err)
}

log.Printf("Created %d snapshots in consistency group", len(snapshots))
```

### Restoring from a Snapshot

```go
// Create a restore request
request := &snapshot.SnapshotRestoreRequest{
    ID:         "restore-1",
    SnapshotID: "snapshot-12345",
    VMID:       "vm-1234",
    CreatedBy:  "admin",
    TenantID:   "tenant-1",
    Options: &snapshot.SnapshotRestoreOptions{
        RestoreMemory:        true,
        PowerOnAfterRestore:  true,
        TargetResourcePool:   "resource-pool-1",
        TargetDatastore:      "datastore-1",
        NetworkMapping: map[string]string{
            "network-1": "network-2",
        },
    },
}

// Create restore request
if err := manager.CreateRestoreRequest(request); err != nil {
    log.Fatalf("Failed to create restore request: %v", err)
}

// Start restore
if err := manager.StartRestore(request.ID); err != nil {
    log.Fatalf("Failed to start restore: %v", err)
}

// Check restore status
status, err := manager.GetRestoreRequest(request.ID)
if err != nil {
    log.Fatalf("Failed to get restore status: %v", err)
}

log.Printf("Restore status: %s", status.Status)
```

## Integration Points

The snapshot system integrates with other NovaCron components:

- **VM Manager**: For VM state management during snapshots and restores
- **Storage Manager**: For efficient storage of snapshots
- **Network Manager**: For network mapping during restores
- **Security Manager**: For access control to snapshots
- **Backup Manager**: For backup integration with snapshots

## Performance Considerations

The snapshot system is designed for performance and reliability:

1. **Incremental Snapshots**: Efficient storage by capturing only changed blocks
2. **Quiescing Support**: Application consistency with minimal downtime
3. **Parallel Processing**: Multiple simultaneous snapshot operations
4. **Optimized Restore**: Fast restore from snapshots with memory state
5. **Efficient Scheduling**: Smart scheduling to avoid performance impacts

## Future Enhancements

- **Snapshot Replication**: Replicate snapshots to remote sites for disaster recovery
- **Pre/Post Snapshot Scripts**: Execute scripts before and after snapshots
- **Snapshot Verification**: Automatically verify snapshot integrity
- **Application-Aware Snapshots**: Enhanced integration with specific applications
- **Enhanced Differencing**: More efficient storage with advanced deduplication
- **Policy-Based Management**: Define complex policies for snapshot management
