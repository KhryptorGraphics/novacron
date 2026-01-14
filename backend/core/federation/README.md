# Multi-Cluster Federation

The federation package provides the capability to create, manage, and operate across multiple NovaCron clusters, enabling distributed resource management, cross-cluster operations, and high availability across geographic regions.

## Features

- **Multiple Federation Modes**: Support for hierarchical, mesh, and hybrid federation models
- **Dynamic Cluster Discovery**: Automatic cluster discovery and health monitoring
- **Cross-Cluster Resource Sharing**: Intelligent resource pooling and allocation across clusters
- **Cross-Cluster VM Migration**: Seamless VM migration between federated clusters
- **Federation Policies**: Configurable policies for resource sharing, authorization, and operation limits
- **Federated Resource Pools**: Create and manage resource pools that span multiple clusters
- **Geographic Awareness**: Location-based optimization for resource placement and management
- **Multi-Tenant Support**: Full tenant isolation in federated environments
- **Cross-Cluster Operations**: Unified operations across multiple clusters
- **Secure Communication**: Encrypted cluster-to-cluster communication

## Components

### Federation Manager

The `FederationManager` is the central component that manages federation:

- Creating and managing cluster federation
- Monitoring cluster health
- Managing federation policies
- Creating and managing federated resource pools
- Coordinating cross-cluster operations

### Cluster Health Checker

The `ClusterHealthChecker` monitors the health of clusters in the federation:

- Periodic health checks of all clusters
- Detection of disconnected or degraded clusters
- Automatic state updates based on health status
- Heartbeat tracking for each cluster

### Cross-Cluster Communication

The `CrossClusterCommunication` handles secure communication between clusters:

- Message-based communication
- Reliable message delivery with retries
- Message queuing and priority
- Message expiration and delivery status tracking

### Resource Sharing

The `ResourceSharing` manages resource sharing between clusters:

- Resource usage tracking per cluster
- Resource allocation and balancing
- Resource sharing policies
- Resource borrowing and lending between clusters

### Cross-Cluster Migration

The `CrossClusterMigration` handles VM migration between clusters:

- Coordinated migration planning
- Migration progress tracking
- Migration state management
- Error handling and recovery

## Usage Examples

### Creating a Federation

```go
// Create a federation manager for the local cluster
manager := federation.NewFederationManager(
    "cluster-1",
    federation.PrimaryCluster,
    federation.MeshMode,
)

// Start the federation manager
if err := manager.Start(); err != nil {
    log.Fatalf("Failed to start federation manager: %v", err)
}

// Add remote clusters to the federation
remoteCluster := &federation.Cluster{
    ID:        "cluster-2",
    Name:      "Data Center 2",
    Endpoint:  "https://dc2.example.com:8443",
    Role:      federation.PeerCluster,
    Resources: &federation.ClusterResources{
        TotalCPU:       1024,
        TotalMemoryGB:  4096,
        TotalStorageGB: 102400,
        NodeCount:      64,
    },
    LocationInfo: &federation.ClusterLocation{
        Region:   "us-west",
        Zone:     "us-west-2a",
        DataCenter: "dc-2",
    },
}

if err := manager.AddCluster(remoteCluster); err != nil {
    log.Fatalf("Failed to add remote cluster: %v", err)
}
```

### Creating Federation Policies

```go
// Create a resource sharing policy
policy := &federation.FederationPolicy{
    ID:          "resource-sharing-policy-1",
    Name:        "High Performance Computing Policy",
    Description: "Policy for sharing HPC resources between clusters",
    ResourceSharingRules: map[string]interface{}{
        "max_cpu_share_percent":     50,
        "max_memory_share_percent":  40,
        "max_storage_share_percent": 30,
        "priority_clusters":         []string{"cluster-1", "cluster-3"},
    },
    MigrationRules: map[string]interface{}{
        "allow_live_migration":    true,
        "max_concurrent_migrations": 5,
        "bandwidth_limit_mbps":    1000,
    },
    Priority: 10,
    Enabled:  true,
    AppliesTo: []string{"cluster-1", "cluster-2", "cluster-3"},
}

if err := manager.CreateFederationPolicy(policy); err != nil {
    log.Fatalf("Failed to create federation policy: %v", err)
}
```

### Creating a Federated Resource Pool

```go
// Create a federated resource pool across multiple clusters
pool := &federation.FederatedResourcePool{
    ID:          "hpc-pool-1",
    Name:        "HPC Resource Pool",
    Description: "High-performance computing resource pool",
    ClusterAllocations: map[string]*federation.ResourceAllocation{
        "cluster-1": {
            CPU:       256,
            MemoryGB:  1024,
            StorageGB: 10240,
            Priority:  10,
        },
        "cluster-2": {
            CPU:       128,
            MemoryGB:  512,
            StorageGB: 5120,
            Priority:  5,
        },
    },
    PolicyID: "resource-sharing-policy-1",
    TenantID: "tenant-1",
}

if err := manager.CreateResourcePool(pool); err != nil {
    log.Fatalf("Failed to create federated resource pool: %v", err)
}
```

### Cross-Cluster VM Migration

```go
// Create a cross-cluster migration job
job := &federation.MigrationJob{
    ID:                    "migration-job-1",
    VMID:                  "vm-1234",
    SourceClusterID:       "cluster-1",
    DestinationClusterID:  "cluster-2",
    Options: map[string]interface{}{
        "migration_type":      "live",
        "bandwidth_limit_mbps": 1000,
        "compression_enabled":  true,
        "compression_level":    7,
    },
}

// Start the migration
if err := manager.crossClusterMigration.StartMigration(job); err != nil {
    log.Fatalf("Failed to start cross-cluster migration: %v", err)
}

// Check migration status
updatedJob, err := manager.crossClusterMigration.GetMigrationJob("migration-job-1")
if err != nil {
    log.Fatalf("Failed to get migration job: %v", err)
}

log.Printf("Migration progress: %d%%, state: %s", updatedJob.Progress, updatedJob.State)
```

### Resource Sharing Between Clusters

```go
// Share resources from one cluster to another
resources := map[string]int{
    "cpu":       64,
    "memory_gb": 256,
}

if err := manager.resourceSharing.ShareResources("cluster-1", "cluster-3", resources); err != nil {
    log.Fatalf("Failed to share resources: %v", err)
}

// Get resource usage for a cluster
usage, err := manager.resourceSharing.GetClusterResourceUsage("cluster-1")
if err != nil {
    log.Fatalf("Failed to get resource usage: %v", err)
}

log.Printf("CPU allocated: %d, used: %d, shared: %d", 
    usage.AllocatedResources["cpu"],
    usage.UsedResources["cpu"],
    usage.SharedResources["cpu"])
```

## Integration Points

The federation system integrates with other NovaCron components:

- **VM Manager**: For VM operations across clusters
- **Storage Manager**: For storage management across clusters
- **Network Manager**: For network configuration in cross-cluster operations
- **Scheduler**: For workload scheduling across federated clusters
- **Security Manager**: For secure cross-cluster communication
- **Monitoring System**: For federated monitoring and alerting

## Best Practices

1. **Cluster Deployment**: Deploy clusters in different geographic regions for disaster recovery
2. **Network Connectivity**: Ensure high-bandwidth, low-latency connections between federated clusters
3. **Policy Configuration**: Carefully configure federation policies based on resource availability
4. **Resource Allocation**: Allocate resources based on workload patterns and priorities
5. **Migration Planning**: Plan cross-cluster migrations during low-utilization periods
6. **Monitoring**: Implement comprehensive monitoring of federation status

## Advanced Features

- **Global Scheduler**: Schedule workloads across all federated clusters based on resource availability
- **Disaster Recovery**: Automatically migrate workloads from failed clusters
- **Geographic Load Balancing**: Distribute workloads based on geographic proximity to users
- **Follow-the-Sun Scheduling**: Migrate workloads to follow daylight hours across time zones
- **Data Locality Awareness**: Place workloads near their data sources
- **Latency-Based Routing**: Route workloads based on network latency metrics
