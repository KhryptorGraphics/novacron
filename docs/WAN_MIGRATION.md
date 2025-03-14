# WAN Migration in NovaCron

This document explains how to use the WAN-optimized migration capabilities in NovaCron for efficient VM migrations over wide-area networks.

## Overview

NovaCron provides specialized components for optimizing VM migrations over WAN links where:
- Bandwidth may be limited
- Latency is higher than in LAN environments
- Connection stability might be an issue
- Network costs may be significant

The WAN migration optimizations focus on three key areas:
1. **Data reduction**: Compression and delta sync to minimize transferred data
2. **Network optimization**: Adaptive bandwidth usage, parallel transfers, and retries
3. **Performance monitoring**: Detailed statistics for analyzing migration efficiency

## Key Components

### WANMigrationOptimizer

The `WANMigrationOptimizer` provides high-level optimization for WAN transfers:

```go
// Create with default configuration
config := vm.DefaultWANMigrationConfig()

// Customize settings as needed
config.CompressionLevel = 6         // Higher compression for WAN
config.EnableDeltaSync = true       // Enable delta sync
config.ChunkSizeKB = 256            // 256KB chunks
config.Parallelism = 4              // 4 parallel transfers
config.MaxBandwidthMbps = 100       // 100 Mbps limit
config.QoSPriority = 5              // Higher priority for migration traffic

// Create the optimizer
optimizer := vm.NewWANMigrationOptimizer(config)
```

### DeltaSyncManager

The `DeltaSyncManager` performs efficient delta synchronization of VM disks, only transferring changed blocks:

```go
// Create with default config
syncConfig := vm.DefaultDeltaSyncConfig()
syncManager := vm.NewDeltaSyncManager(syncConfig)
defer syncManager.Close()

// Can integrate with the WAN migration optimizer
syncManager.IntegrateWithWANMigrationOptimizer(optimizer)

// Synchronize a VM disk
err := syncManager.SyncFile(ctx, sourceDiskPath, destDiskPath)
if err != nil {
    log.Fatalf("Sync failed: %v", err)
}

// Get stats about the transfer
stats := syncManager.GetStats()
log.Printf("Bytes saved: %d (%.1f%%)", stats.BytesSaved, stats.BytesSavedPercent)
```

## Network Adaptation

The system can dynamically adapt to network conditions:

```go
// Tune the optimizer for specific network conditions
optimizer.TuneForNetwork(
    50.0,               // Bandwidth in Mbps
    100*time.Millisecond, // Latency
    0.01,               // Packet loss as fraction (1%)
)
```

## Migration Options

When performing a migration, get optimized options from the optimizer:

```go
// Get migration options optimized for WAN
migrationOpts := optimizer.GetOptimizedMigrationOptions("live")

// Use these options with the migration manager
migrationID, err := migrationManager.StartMigration(vmID, destNodeID, migrationOpts)
```

## Statistics and Monitoring

After migration, detailed statistics are available:

```go
// Get optimizer stats
stats := optimizer.GetStats()

// Access various metrics
log.Printf("Average bandwidth: %.2f Mbps", stats.AverageBandwidthMbps)
log.Printf("Total bytes transferred: %d", stats.BytesTransferred)
log.Printf("Compression savings: %d bytes", stats.CompressionSavingsBytes)
log.Printf("Delta sync savings: %d bytes", stats.DeltaSyncSavingsBytes)
log.Printf("Total downtime: %d ms", stats.TotalDowntimeMs)
```

## Example Usage

See `backend/examples/wan_migration_example.go` for a complete example of using the WAN migration optimization components.

## Recommendations

1. **High compression for low bandwidth**: Use higher compression levels (6-9) when bandwidth is limited
2. **Adjust chunk size based on latency**: Lower chunk sizes for high-latency links, larger for higher-bandwidth, low-latency links
3. **Enable delta sync**: Almost always beneficial, especially for incremental migrations
4. **Use parallel transfers**: Especially helpful with packet loss or high-latency connections
5. **Adjust QoS priority**: Higher priority for critical migrations to ensure consistent performance
