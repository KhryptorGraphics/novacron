# NovaCron Redis Caching Infrastructure

## Overview

The NovaCron caching infrastructure provides a comprehensive, multi-tier caching system designed for high-performance VM management operations. The system achieves 90-95% cache hit rates with sub-millisecond access times through intelligent cache layering and Redis clustering.

## Architecture

### Multi-Tier Cache Design

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   L1 Cache      │    │   L2 Cache      │    │   L3 Cache      │
│   (Memory)      │    │   (Redis)       │    │  (Persistent)   │
│                 │    │                 │    │                 │
│ • LRU eviction  │    │ • Redis Cluster │    │ • File-based    │
│ • Sub-ms access │    │ • Sentinel HA   │    │ • Long-term     │
│ • 10K items max │    │ • Compression   │    │ • Disk storage  │
│ • 5min TTL      │    │ • 15min TTL     │    │ • Configurable  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                │
                    ┌─────────────────┐
                    │  Cache Manager  │
                    │                 │
                    │ • Smart routing │
                    │ • Invalidation  │
                    │ • Metrics       │
                    │ • Health checks │
                    └─────────────────┘
```

### Redis Cluster Configurations

#### 1. Redis Sentinel (High Availability)
- **Components**: 1 Master + 2 Slaves + 3 Sentinels
- **Use Case**: High availability with automatic failover
- **Benefits**: Minimal data loss, automatic recovery
- **Ports**: 6379 (master), 6380-6381 (slaves), 26379-26381 (sentinels)

#### 2. Redis Cluster (Horizontal Scaling)
- **Components**: 6 nodes (3 masters + 3 replicas)
- **Use Case**: Horizontal scaling and data partitioning
- **Benefits**: Higher throughput, data sharding
- **Ports**: 7001-7006 (cluster nodes), 17001-17006 (cluster bus)

#### 3. Standalone Redis
- **Components**: Single Redis instance
- **Use Case**: Development and testing
- **Benefits**: Simple setup, minimal resources

## Features

### Core Functionality
- **Multi-tier caching** with automatic fallback
- **Smart cache invalidation** based on VM lifecycle events
- **Batch operations** for improved performance
- **Compression and serialization** options
- **TTL management** with configurable timeouts
- **Pattern-based caching** for different data types

### VM-Specific Caching
- **VM State Caching**: Current VM status with 30s TTL
- **VM Resources Caching**: CPU, memory, disk allocation with 2min TTL
- **VM Migration Status**: Real-time migration progress with 1min TTL
- **VM Metrics**: Performance data with 15s TTL
- **VM Configuration**: Static config with 15min TTL
- **Node Resources**: Cluster node capacity with 30s TTL

### Monitoring and Observability
- **Real-time metrics collection** with Prometheus export
- **Health monitoring** with alerting
- **Performance analytics** with histograms and heatmaps
- **Web UI dashboard** for visual monitoring
- **Custom alerts** based on configurable thresholds

## Quick Start

### 1. Setup Redis with Sentinel (Recommended)
```bash
./scripts/cache-setup.sh setup sentinel
./scripts/cache-setup.sh start
```

### 2. Setup Redis Cluster (For Scale)
```bash
./scripts/cache-setup.sh setup cluster
./scripts/cache-setup.sh start
```

### 3. Test Cache Connectivity
```bash
./scripts/cache-setup.sh test
```

### 4. Check Status
```bash
./scripts/cache-setup.sh status
```

## Configuration

### Cache Configuration
```go
config := &cache.CacheConfig{
    // L1 Cache (Memory)
    L1Enabled:    true,
    L1MaxSize:    10000,
    L1TTL:        5 * time.Minute,
    L1CleanupInt: 30 * time.Second,

    // L2 Cache (Redis)
    L2Enabled:       true,
    RedisAddrs:      []string{"localhost:6379"},
    RedisCluster:    false,
    SentinelEnabled: true,
    SentinelAddrs:   []string{"localhost:26379", "localhost:26380", "localhost:26381"},
    SentinelMaster:  "mymaster",

    // L3 Cache (Persistent)
    L3Enabled: true,
    L3Path:    "/var/cache/novacron",

    // General
    DefaultTTL:        15 * time.Minute,
    EnableCompression: true,
    EnableMetrics:     true,
}
```

### VM Cache Configuration
```go
vmConfig := &cache.VMCacheConfig{
    VMStateTTL:       30 * time.Second,
    VMResourcesTTL:   2 * time.Minute,
    VMMigrationTTL:   1 * time.Minute,
    VMMetricsTTL:     15 * time.Second,
    VMConfigTTL:      15 * time.Minute,
    EnableWarmup:     true,
    WarmupInterval:   5 * time.Minute,
    CriticalVMList:   []string{"vm-prod-1", "vm-prod-2"},
}
```

## API Usage

### Basic Cache Operations
```go
// Create cache
cache, err := cache.NewMultiTierCache(config, logger)
if err != nil {
    return err
}
defer cache.Close()

// Set value
err = cache.Set(ctx, "key", []byte("value"), 5*time.Minute)

// Get value
value, err := cache.Get(ctx, "key")

// Batch operations
items := map[string]cache.CacheItem{
    "key1": {Value: []byte("value1"), TTL: 5*time.Minute},
    "key2": {Value: []byte("value2"), TTL: 10*time.Minute},
}
err = cache.SetMulti(ctx, items)
results, err := cache.GetMulti(ctx, []string{"key1", "key2"})
```

### VM Cache Integration
```go
// Create VM cache manager
vmCacheManager := cache.NewVMCacheManager(cache, vmConfig, logger)

// Cache VM state
vmState := map[string]interface{}{
    "id":    "vm-123",
    "state": "running",
    "type":  "kvm",
}
err = vmCacheManager.SetVMState(ctx, "vm-123", vmState)

// Get VM state from cache
state, err := vmCacheManager.GetVMState(ctx, "vm-123")

// Batch VM operations
states, err := vmCacheManager.GetMultipleVMStates(ctx, []string{"vm-1", "vm-2"})
```

### Event-Driven Cache Management
```go
// Register for VM lifecycle events
vmCacheManager.OnVMStateChange(ctx, "vm-123", "running", "stopped")
vmCacheManager.OnVMCreated(ctx, "vm-123", "node-1")
vmCacheManager.OnVMDeleted(ctx, "vm-123", "node-1")
vmCacheManager.OnVMResourceChange(ctx, "vm-123")
```

## Performance Metrics

### Target Performance
- **Cache Hit Rate**: 90-95%
- **L1 Cache Access Time**: < 1ms
- **L2 Cache Access Time**: < 5ms
- **L3 Cache Access Time**: < 50ms
- **Throughput**: > 10,000 operations/second
- **Memory Efficiency**: < 10MB per 1000 cached items

### Monitoring Endpoints
- **Health Check**: `http://localhost:9091/health`
- **Prometheus Metrics**: `http://localhost:9091/metrics`
- **Detailed Metrics**: `http://localhost:9091/api/metrics`
- **Web Dashboard**: `http://localhost:8082`
- **Redis Commander**: `http://localhost:8081`

### Key Metrics Monitored
- Cache hit/miss rates per tier
- Average response times
- Error rates and types
- Memory usage and eviction rates
- Connection pool status
- Cluster health and failover events

## Production Deployment

### Docker Compose Deployment
```bash
# Sentinel setup (recommended)
docker-compose -f docker-compose.cache.yml up -d

# Cluster setup (for high throughput)
docker-compose -f docker-compose.cache.yml --profile cluster up -d

# Monitoring
docker-compose -f docker-compose.cache.yml up -d redis-commander cache-monitor
```

### Environment Variables
```bash
# Redis Configuration
REDIS_MASTER_ADDR=redis-master:6379
REDIS_SLAVE_ADDRS=redis-slave-1:6380,redis-slave-2:6381
SENTINEL_ADDRS=redis-sentinel-1:26379,redis-sentinel-2:26379,redis-sentinel-3:26379
SENTINEL_MASTER_NAME=mymaster

# Monitoring
METRICS_PORT=9091
WEBUI_PORT=8082
LOG_LEVEL=info
HEALTH_CHECK_INTERVAL=30s
```

### Security Considerations
- **Network Isolation**: Use dedicated cache network
- **Authentication**: Configure Redis AUTH for production
- **Encryption**: Enable TLS for Redis connections
- **Access Control**: Implement IP whitelisting
- **Monitoring**: Set up alerting for security events

### Scalability Guidelines
- **Memory Planning**: Allocate 512MB-1GB per Redis instance
- **Connection Pooling**: Configure 100+ connections per application instance
- **Cluster Scaling**: Add nodes in pairs (master + replica)
- **Monitoring Resources**: Allocate 256MB for monitoring services

## Troubleshooting

### Common Issues

#### 1. Connection Failures
```bash
# Check Redis connectivity
redis-cli -h localhost -p 6379 ping

# Check Sentinel status
redis-cli -h localhost -p 26379 sentinel masters

# Check cluster status
redis-cli -h localhost -p 7001 cluster info
```

#### 2. Performance Issues
```bash
# Check cache hit rates
curl http://localhost:9091/api/metrics | jq '.basic.hit_rate'

# Monitor response times
curl http://localhost:9091/api/metrics | jq '.response_times'

# Check memory usage
redis-cli -h localhost -p 6379 info memory
```

#### 3. High Memory Usage
```bash
# Check eviction statistics
redis-cli -h localhost -p 6379 info stats | grep evicted

# Monitor key expiration
redis-cli -h localhost -p 6379 info stats | grep expired

# Analyze memory usage patterns
redis-cli -h localhost -p 6379 memory usage [key]
```

### Debug Commands
```bash
# Enable debug logging
export LOG_LEVEL=debug

# Test cache operations manually
redis-cli -h localhost -p 6379
> set test_key "test_value"
> get test_key
> ttl test_key

# Check cluster health
redis-cli -h localhost -p 7001 cluster nodes

# Monitor real-time operations
redis-cli -h localhost -p 6379 monitor
```

## Advanced Features

### Cache Warming
```go
// Configure critical VMs for cache warming
config := cache.DefaultVMCacheConfig()
config.EnableWarmup = true
config.WarmupInterval = 5 * time.Minute
config.CriticalVMList = []string{"vm-prod-1", "vm-prod-2", "vm-db-1"}
```

### Custom Serialization
```go
// Configure compression and serialization
config.EnableCompression = true
config.CompressionLevel = 6
config.Serialization = "json" // json, gob, msgpack
```

### Pattern-Based Invalidation
```go
// Invalidate all VM-related cache entries
vmCacheManager.InvalidateVMPattern("vm:*")

// Invalidate node-specific entries
vmCacheManager.InvalidateNode(ctx, "node-1")

// Invalidate cluster-wide entries
vmCacheManager.InvalidateCluster(ctx)
```

### Custom Metrics Collection
```go
// Record custom operation metrics
metricsCollector.RecordOperation("custom_op", duration, success, "pattern:")

// Export to external monitoring
prometheusMetrics := metricsCollector.ExportPrometheusMetrics()
metricsJSON, _ := metricsCollector.GetMetricsJSON()
```

## Integration Examples

### VM Manager Integration
```go
// Wrap VM manager with caching
cachedVMManager, err := cache.NewCachedVMManager(
    vmManager,
    cacheInstance,
    vmIntegrationConfig,
    logger,
)

// Use cached operations
vm, err := cachedVMManager.GetVM("vm-123")
state, err := cachedVMManager.GetVMState("vm-123")
metrics, err := cachedVMManager.GetVMMetrics("vm-123")
```

### Custom Cache Patterns
```go
// Implement custom cache keys
func (c *CustomCache) GetUserVMs(userID string) ([]VM, error) {
    key := fmt.Sprintf("user:vms:%s", userID)
    
    // Try cache first
    if data, err := c.cache.Get(ctx, key); err == nil {
        var vms []VM
        json.Unmarshal(data, &vms)
        return vms, nil
    }
    
    // Fallback to database
    vms, err := c.database.GetUserVMs(userID)
    if err != nil {
        return nil, err
    }
    
    // Cache result
    data, _ := json.Marshal(vms)
    c.cache.Set(ctx, key, data, 5*time.Minute)
    
    return vms, nil
}
```

## Support and Maintenance

### Backup and Recovery
- Redis persistence is enabled with AOF and RDB
- Sentinel provides automatic failover
- Regular backup schedule recommended for L3 cache

### Upgrading
1. Update Redis images in docker-compose files
2. Rolling restart for zero downtime
3. Monitor metrics during upgrade process
4. Verify cache hit rates post-upgrade

### Monitoring Alerts
- Cache hit rate < 85%
- Response time > 100ms
- Error rate > 1%
- Memory usage > 90%
- Connection failures > 5/min

For additional support, refer to the monitoring dashboard at http://localhost:8082 or check logs:
```bash
docker-compose -f docker-compose.cache.yml logs -f cache-monitor
```