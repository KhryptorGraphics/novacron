# Distributed Storage for NovaCron

This package implements a distributed storage layer for the NovaCron virtualization platform, providing high availability, fault tolerance, and horizontal scalability.

## Overview

The distributed storage system extends NovaCron's base storage manager with sharding, replication, and self-healing capabilities. It distributes volume data across multiple storage nodes, ensuring no single point of failure.

## Features

### Sharding and Replication

- **Data Sharding**: Volumes are automatically split into configurable-size shards
- **Flexible Replication**: Each shard is replicated across multiple nodes with configurable replication factor
- **Synchronous/Asynchronous Replication**: Support for both modes with tunable consistency guarantees

### Placement Strategies

The system includes multiple placement strategies to optimize for different scenarios:

- **Random**: Simple distribution for development environments
- **Balanced**: Evenly distributes shards across all nodes for optimal resource utilization
- **Locality-Aware**: Places related shards on the same nodes to optimize access patterns
- **Zone-Aware**: Ensures replicas are distributed across availability zones for disaster recovery

### Self-Healing

- **Automatic Monitoring**: Periodic health checks identify under-replicated or damaged shards
- **Automatic Healing**: Self-repair system restores replication factor when nodes fail
- **Automatic Rebalancing**: Redistributes data when nodes are added or removed

### Consistency Models

Configurable consistency guarantees for different workloads:

- **Eventual Consistency**: Maximizes performance with eventual guarantees
- **Strong Consistency**: Ensures all readers see the latest write
- **Causal Consistency**: Preserves causal relationships between operations

## Usage

### Creating a Distributed Storage Service

```go
// Create a base storage manager
baseManager := CreateStorageManager()

// Create a distributed storage configuration
config := DefaultDistributedStorageConfig()
config.RootDir = "/var/lib/novacron/distributed"
config.ShardSize = 64 * 1024 * 1024  // 64 MB shards
config.DefaultReplicationFactor = 3
config.SynchronousReplication = true
config.ConsistencyProtocol = "strong"

// Create the distributed storage service
service, err := NewDistributedStorageService(baseManager, config)
if err != nil {
    log.Fatalf("Failed to create distributed storage service: %v", err)
}

// Start the service
if err := service.Start(); err != nil {
    log.Fatalf("Failed to start distributed storage service: %v", err)
}
```

### Adding Storage Nodes

```go
// Add a storage node
node := NodeInfo{
    ID:        "node1",
    Name:      "Storage Node 1",
    Role:      "storage",
    Address:   "192.168.1.10",
    Port:      8000,
    Available: true,
}
service.AddNode(node)
```

### Creating a Distributed Volume

```go
// Create a volume specification
spec := VolumeSpec{
    Name:   "my-distributed-volume",
    SizeMB: 1024,
    Type:   VolumeTypeCeph,
    Options: map[string]string{
        "description": "My distributed volume",
    },
}

// Create a distributed volume with replication factor 3
ctx := context.Background()
volume, err := service.CreateDistributedVolume(ctx, spec, 3)
if err != nil {
    log.Fatalf("Failed to create distributed volume: %v", err)
}
```

### Reading and Writing Data

```go
// Write data to a shard
data := []byte("Hello, distributed world!")
err = service.WriteShard(ctx, volume.ID, 0, data)
if err != nil {
    log.Fatalf("Failed to write to shard: %v", err)
}

// Read data from a shard
readData, err := service.ReadShard(ctx, volume.ID, 0)
if err != nil {
    log.Fatalf("Failed to read from shard: %v", err)
}
```

### Repairing a Volume

```go
// Trigger manual repair
err = service.RepairVolume(ctx, volume.ID)
if err != nil {
    log.Fatalf("Failed to repair volume: %v", err)
}
```

### Rebalancing a Volume

```go
// Trigger manual rebalancing
err = service.RebalanceVolume(ctx, volume.ID)
if err != nil {
    log.Fatalf("Failed to rebalance volume: %v", err)
}
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `RootDir` | Root directory for storage | `/var/lib/novacron/distributed` |
| `MaxCapacity` | Maximum storage capacity in bytes (0 = unlimited) | `0` |
| `DefaultReplicationFactor` | Default number of replicas | `3` |
| `DefaultEncryption` | Whether to enable encryption by default | `false` |
| `EnableSharding` | Whether to enable data sharding | `true` |
| `ShardSize` | Size of shards in bytes | `64 MB` |
| `ConsistencyProtocol` | Consistency protocol (eventual, strong, causal) | `eventual` |
| `HealingQueryCount` | Number of nodes to query for healing | `5` |
| `HealthCheckInterval` | Interval for health checks | `1 minute` |
| `HealingInterval` | Interval for data healing | `1 hour` |
| `SynchronousReplication` | Whether to enable synchronous replication | `false` |

## Architecture

The distributed storage system consists of the following key components:

1. **DistributedStorageService**: Main service that manages storage nodes and volumes
2. **DistributedVolume**: Extended volume object with sharding and replication metadata
3. **NodeInfo**: Information about storage nodes in the cluster
4. **ShardPlacementStrategy**: Interface for different shard placement algorithms
5. **Health Checking**: Background services for monitoring and repairing volumes

## Implementation Notes

### Shard Placement

Shard placement is a crucial aspect of the system's performance and reliability. The current implementation includes multiple strategies:

- **Random**: Simplest approach, randomly assigns shards to nodes
- **Balanced**: Ensures even distribution of shards across nodes
- **Locality-Aware**: Places related shards on the same node to optimize access patterns
- **Zone-Aware**: Distributes replicas across different availability zones

### Data Healing

The system includes a comprehensive self-healing mechanism:

1. **Health Checks**: Periodic checks identify damaged or under-replicated shards
2. **Replica Selection**: Identifies the best available replica of a shard
3. **Repair**: Creates new replicas until the target replication factor is achieved
4. **Verification**: Confirms the health of repaired shards

### Replication Modes

The system supports two replication modes:

- **Synchronous**: Write operations only succeed when data is replicated to all nodes
- **Asynchronous**: Write operations succeed after writing to one node, with background replication

## Implemented Enhancements

### End-to-End Encryption

The distributed storage system now includes robust end-to-end encryption capabilities:

- **Multiple Encryption Algorithms**: Support for AES-256, AES-192, and AES-128
- **Multiple Encryption Modes**: Support for GCM (authenticated), CBC, and CTR modes
- **Volume-Specific Keys**: Each volume has a unique encryption key derived from the master key
- **Transparent Operation**: Encryption and decryption happen automatically during reads and writes
- **Key Verification**: Includes verification data to detect incorrect keys
- **Size-Based Policy**: Option to only encrypt data above a configurable size threshold

#### Configuration Options

```go
config := DefaultDistributedStorageConfig()
config.EncryptionConfig.Algorithm = encryption.EncryptionAES256
config.EncryptionConfig.Mode = encryption.EncryptionModeGCM
config.EncryptionConfig.MasterKey = "your-secure-master-key"
config.EncryptionConfig.MinSizeBytes = 1024  // Only encrypt data >= 1KB
config.EncryptionConfig.IncludeVerification = true
```

#### Security Features

- **Per-Volume Isolation**: Even with the same master key, each volume uses a different derived key
- **Authenticated Encryption**: When using GCM mode, data integrity is automatically verified
- **Defense in Depth**: Combined with network security and permissions, provides multiple security layers

### Data Compression

The distributed storage system includes built-in data compression capabilities that significantly reduce storage requirements while maintaining data integrity:

- **Multiple Compression Algorithms**: Support for Gzip, Zlib, and LZW algorithms with configurable compression levels
- **Intelligent Auto-detection**: Automatically detects if data is compressible to avoid wasting CPU cycles on incompressible data
- **Per-shard Control**: Each shard can use different compression algorithms based on data characteristics
- **Compression Metadata**: Tracks compression ratio, original size, and algorithm used for each shard
- **Transparent Access**: Higher-level components can read and write data normally without knowledge of compression

#### Configuration Options

```go
config := DefaultDistributedStorageConfig()
config.CompressionConfig.Algorithm = compression.CompressionGzip
config.CompressionConfig.Level = compression.CompressionBest
config.CompressionConfig.MinSizeBytes = 4 * 1024        // 4KB
config.CompressionConfig.MaxSizeBytes = 1024 * 1024 * 32 // 32MB
config.CompressionConfig.AutoDetect = true
```

#### Compression Statistics

The system tracks detailed compression statistics:

```go
// Get compression information for a shard
shard := volume.DistInfo.Shards[0]
fmt.Printf("Compression algorithm: %s\n", shard.CompressionAlgorithm)
fmt.Printf("Original size: %d bytes\n", shard.OriginalSize)
fmt.Printf("Compressed size: %d bytes\n", shard.Size)
fmt.Printf("Compression ratio: %.2f\n", shard.CompressionRatio)
```

## Future Enhancements

1. **Deduplication**: Implement block-level deduplication across volumes
2. **Dynamic Shard Sizes**: Adjust shard sizes based on access patterns
3. **Tiered Storage**: Automatically move hot/cold data between different storage tiers
