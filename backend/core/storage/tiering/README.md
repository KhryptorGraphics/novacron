# Storage Tiering System

The Storage Tiering System automatically migrates data between different storage tiers based on access patterns, cost optimization, and other configurable policies. This enables NovaCron to efficiently utilize various storage technologies to optimize for both performance and cost.

## Architecture

The storage tiering system is built on the following components:

1. **Tier Manager**: Coordinates all tiering operations and manages the lifecycle of data across tiers
2. **Tiering Policies**: Define rules for when and where to move data
3. **Metadata Store**: Tracks volume locations and statistics across tiers
4. **Volume Statistics**: Collects and analyzes usage patterns for intelligent placement

## Storage Tiers

The system defines three primary storage tiers:

1. **Hot Tier**: Highest performance, typically using SSD or in-memory storage
   - Best for frequently accessed data
   - Highest cost per GB
   - Optimized for low latency and high IOPS

2. **Warm Tier**: Medium performance, typically using HDD or network storage
   - Suitable for occasionally accessed data
   - Moderate cost per GB
   - Balanced performance and cost

3. **Cold Tier**: Lowest performance, typically using object storage or archive solutions
   - Ideal for rarely accessed data
   - Lowest cost per GB
   - Optimized for durability and cost efficiency

## Default Tiering Policies

The system comes with two built-in policies:

1. **Data Aging Policy**: Moves data based on access frequency
   - Frequently accessed data (>1 access/day) stays in hot tier
   - Occasionally accessed data (>1 access/week) moves to warm tier
   - Rarely accessed data (<1 access/week) moves to cold tier

2. **Cost Optimization Policy**: Balances performance needs with storage costs
   - Large, infrequently accessed volumes moved to colder tiers
   - Small, frequently accessed volumes kept in hot tier
   - Very large, rarely accessed volumes (>1TB) moved to cold tier

## Using the Tiering System

### Setup

```go
// Create a tier manager
tierManager := tiering.NewTierManager()

// Add storage tiers
tierManager.AddTier(tiering.TierHot, hotStorageDriver, "SSD-Storage", 0.20, 1000)  // $0.20/GB/month, 1TB capacity
tierManager.AddTier(tiering.TierWarm, warmStorageDriver, "HDD-Storage", 0.08, 5000)  // $0.08/GB/month, 5TB capacity
tierManager.AddTier(tiering.TierCold, coldStorageDriver, "Object-Storage", 0.01, 0)  // $0.01/GB/month, unlimited capacity

// Add default policies
tierManager.CreateDefaultAgingPolicy()
tierManager.CreateCostOptimizationPolicy()

// Initialize the manager
tierManager.Initialize()

// Start background worker for automatic tiering (runs every 4 hours)
tierManager.StartBackgroundWorker(4 * time.Hour)
```

### Custom Policies

You can create custom policies for specialized workload requirements:

```go
// Create a policy for high-performance VMs
tierManager.AddPolicy("HighPerformanceVMs", func(stats *tiering.VolumeStats) (bool, tiering.TierLevel) {
    // If the volume name has a "high-perf" prefix, always keep in hot tier
    if strings.HasPrefix(stats.Name, "high-perf") {
        return stats.CurrentTier != tiering.TierHot, tiering.TierHot
    }
    
    // Otherwise, no specific placement
    return false, stats.CurrentTier
}, 200) // Higher priority than default policies
```

### Manual Operations

For specific volumes, you can override automatic tiering:

```go
// Pin a volume to a specific tier
tierManager.PinVolume("critical-database", tiering.TierHot)

// Unpin a volume to allow automatic tiering again
tierManager.UnpinVolume("archived-logs")

// Record access (normally done automatically within storage operations)
tierManager.RecordVolumeAccess("frequently-accessed-volume")
```

## Monitoring

Get statistics about tier usage:

```go
stats := tierManager.GetTierStats()
for tierLevel, tierStats := range stats {
    fmt.Printf("Tier %d (%s):\n", tierLevel, tierStats["name"])
    fmt.Printf("  - Volumes: %d\n", tierStats["volume_count"])
    fmt.Printf("  - Usage: %.2f/%.2f GB\n", tierStats["current_usage_gb"], tierStats["max_capacity_gb"])
    fmt.Printf("  - Cost: $%.2f/GB/month\n", tierStats["cost_per_gb_month"])
}
```

## Integration with Storage Drivers

The tiering system works with any storage driver that implements the StorageDriver interface, which includes methods for:
- Volume operations (create, delete, resize)
- Data operations (read, write)
- Metadata operations (get info, list volumes)

This allows NovaCron to use a wide variety of storage backends while providing consistent tiering capabilities.

## Implementation Details

- Thread-safe design with mutex protection for concurrent access
- Context-based cancellation for graceful shutdown
- Automatic metadata management for tracking volumes across tiers
- Efficient data transfer between tiers with streaming interfaces
- Configurable policies with priority-based evaluation
