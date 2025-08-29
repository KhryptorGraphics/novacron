# Redis Caching Layer Architecture for NovaCron

## Executive Summary

This document presents a comprehensive Redis-based caching architecture for NovaCron, designed to dramatically improve performance for VM metadata, resource allocation, migration state, and real-time operations. The architecture includes intelligent cache invalidation strategies, Redis clustering for high availability, and multi-tier caching for optimal performance across all NovaCron components.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   VM Manager    │  │   Scheduler     │  │   Monitor       │  │
│  │     Cache       │  │     Cache       │  │    Cache        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Cache Abstraction Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   Intelligent   │  │   Invalidation  │  │   Performance   │  │
│  │   Cache Router  │  │    Manager      │  │   Optimizer     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Multi-Tier Cache                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   L1: Local     │  │  L2: Redis      │  │  L3: Distributed│  │
│  │   Memory        │  │  Cluster        │  │   Storage       │  │
│  │   (1ms)         │  │  (5ms)          │  │   (50ms)        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Redis Infrastructure                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Master-Replica  │  │ Sentinel HA     │  │ Cluster Mode    │  │
│  │   Replication   │  │  Management     │  │  Sharding       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 1. Core Caching Architecture

### 1.1 Cache Manager Interface

```go
package cache

import (
    "context"
    "fmt"
    "sync"
    "time"
)

// CacheManager provides a unified interface for all caching operations
type CacheManager struct {
    localCache      *LocalCache
    redisCache      *RedisCache
    distributedCache *DistributedCache
    router          *CacheRouter
    invalidator     *InvalidationManager
    metrics         *CacheMetrics
    config          *CacheConfig
}

type CacheConfig struct {
    LocalCacheSize      int           `json:"local_cache_size"`
    LocalCacheTTL       time.Duration `json:"local_cache_ttl"`
    RedisCacheTTL       time.Duration `json:"redis_cache_ttl"`
    RedisCluster        []string      `json:"redis_cluster_nodes"`
    EnableCompression   bool          `json:"enable_compression"`
    EnableEncryption    bool          `json:"enable_encryption"`
    MaxRetries          int           `json:"max_retries"`
    RetryDelay          time.Duration `json:"retry_delay"`
    CircuitBreakerConfig CircuitBreakerConfig `json:"circuit_breaker"`
}

// NewCacheManager creates a new cache manager with multi-tier caching
func NewCacheManager(config *CacheConfig) (*CacheManager, error) {
    localCache, err := NewLocalCache(config.LocalCacheSize, config.LocalCacheTTL)
    if err != nil {
        return nil, fmt.Errorf("failed to create local cache: %w", err)
    }
    
    redisCache, err := NewRedisCache(config)
    if err != nil {
        return nil, fmt.Errorf("failed to create Redis cache: %w", err)
    }
    
    distributedCache, err := NewDistributedCache(config)
    if err != nil {
        return nil, fmt.Errorf("failed to create distributed cache: %w", err)
    }
    
    router := NewCacheRouter(localCache, redisCache, distributedCache)
    invalidator := NewInvalidationManager(localCache, redisCache, distributedCache)
    
    return &CacheManager{
        localCache:       localCache,
        redisCache:       redisCache,
        distributedCache: distributedCache,
        router:          router,
        invalidator:     invalidator,
        metrics:         NewCacheMetrics(),
        config:          config,
    }, nil
}

// Get retrieves a value from the cache with intelligent tier selection
func (cm *CacheManager) Get(ctx context.Context, key string) (interface{}, bool, error) {
    startTime := time.Now()
    defer func() {
        cm.metrics.RecordGetLatency(time.Since(startTime))
    }()
    
    // Try L1 cache first (fastest)
    if value, found := cm.localCache.Get(key); found {
        cm.metrics.RecordHit("local")
        return value, true, nil
    }
    cm.metrics.RecordMiss("local")
    
    // Try L2 cache (Redis)
    value, found, err := cm.redisCache.Get(ctx, key)
    if err != nil {
        cm.metrics.RecordError("redis_get")
        // Fallback to L3 if Redis fails
    } else if found {
        cm.metrics.RecordHit("redis")
        // Populate L1 cache asynchronously
        go cm.localCache.Set(key, value, cm.config.LocalCacheTTL)
        return value, true, nil
    }
    cm.metrics.RecordMiss("redis")
    
    // Try L3 cache (distributed storage)
    value, found, err = cm.distributedCache.Get(ctx, key)
    if err != nil {
        cm.metrics.RecordError("distributed_get")
        return nil, false, fmt.Errorf("all cache tiers failed: %w", err)
    }
    
    if found {
        cm.metrics.RecordHit("distributed")
        // Populate higher tiers asynchronously
        go func() {
            cm.redisCache.Set(ctx, key, value, cm.config.RedisCacheTTL)
            cm.localCache.Set(key, value, cm.config.LocalCacheTTL)
        }()
        return value, true, nil
    }
    
    cm.metrics.RecordMiss("distributed")
    return nil, false, nil
}

// Set stores a value in appropriate cache tiers based on the data type
func (cm *CacheManager) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    startTime := time.Now()
    defer func() {
        cm.metrics.RecordSetLatency(time.Since(startTime))
    }()
    
    // Determine optimal cache tier based on data characteristics
    tier := cm.router.DetermineCacheTier(key, value)
    
    switch tier {
    case CacheTierAll:
        // Store in all tiers for frequently accessed data
        return cm.setInAllTiers(ctx, key, value, ttl)
    case CacheTierRedisOnly:
        // Store in Redis for shared data
        return cm.redisCache.Set(ctx, key, value, ttl)
    case CacheTierLocalOnly:
        // Store in local cache for node-specific data
        return cm.localCache.Set(key, value, ttl)
    default:
        return fmt.Errorf("invalid cache tier: %v", tier)
    }
}

func (cm *CacheManager) setInAllTiers(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    var wg sync.WaitGroup
    var errors []error
    var mu sync.Mutex
    
    // Set in local cache
    wg.Add(1)
    go func() {
        defer wg.Done()
        if err := cm.localCache.Set(key, value, ttl); err != nil {
            mu.Lock()
            errors = append(errors, fmt.Errorf("local cache set failed: %w", err))
            mu.Unlock()
        }
    }()
    
    // Set in Redis cache
    wg.Add(1)
    go func() {
        defer wg.Done()
        if err := cm.redisCache.Set(ctx, key, value, ttl); err != nil {
            mu.Lock()
            errors = append(errors, fmt.Errorf("redis cache set failed: %w", err))
            mu.Unlock()
        }
    }()
    
    // Set in distributed cache
    wg.Add(1)
    go func() {
        defer wg.Done()
        if err := cm.distributedCache.Set(ctx, key, value, ttl); err != nil {
            mu.Lock()
            errors = append(errors, fmt.Errorf("distributed cache set failed: %w", err))
            mu.Unlock()
        }
    }()
    
    wg.Wait()
    
    if len(errors) > 0 {
        return fmt.Errorf("cache set errors: %v", errors)
    }
    
    return nil
}
```

### 1.2 Local Memory Cache (L1)

```go
type LocalCache struct {
    cache    *sync.Map
    ttl      time.Duration
    maxSize  int
    eviction *LRUEviction
    stats    *LocalCacheStats
}

type CacheEntry struct {
    Value     interface{}
    ExpiresAt time.Time
    AccessCount int64
    LastAccess  time.Time
    Size        int64
}

func NewLocalCache(maxSize int, defaultTTL time.Duration) (*LocalCache, error) {
    lc := &LocalCache{
        cache:   &sync.Map{},
        ttl:     defaultTTL,
        maxSize: maxSize,
        eviction: NewLRUEviction(maxSize),
        stats:   NewLocalCacheStats(),
    }
    
    // Start cleanup goroutine
    go lc.startCleanup()
    
    return lc, nil
}

func (lc *LocalCache) Get(key string) (interface{}, bool) {
    value, ok := lc.cache.Load(key)
    if !ok {
        lc.stats.RecordMiss()
        return nil, false
    }
    
    entry := value.(*CacheEntry)
    
    // Check expiration
    if time.Now().After(entry.ExpiresAt) {
        lc.cache.Delete(key)
        lc.eviction.Remove(key)
        lc.stats.RecordExpiration()
        return nil, false
    }
    
    // Update access statistics
    entry.AccessCount++
    entry.LastAccess = time.Now()
    lc.eviction.Access(key)
    
    lc.stats.RecordHit()
    return entry.Value, true
}

func (lc *LocalCache) Set(key string, value interface{}, ttl time.Duration) error {
    if ttl == 0 {
        ttl = lc.ttl
    }
    
    entry := &CacheEntry{
        Value:     value,
        ExpiresAt: time.Now().Add(ttl),
        AccessCount: 1,
        LastAccess:  time.Now(),
        Size:       lc.calculateSize(value),
    }
    
    // Check if we need to evict
    if lc.needsEviction(entry.Size) {
        lc.evictEntries(entry.Size)
    }
    
    lc.cache.Store(key, entry)
    lc.eviction.Add(key, entry.Size)
    lc.stats.RecordSet()
    
    return nil
}

func (lc *LocalCache) needsEviction(newEntrySize int64) bool {
    currentSize := lc.eviction.GetCurrentSize()
    return currentSize+newEntrySize > int64(lc.maxSize)
}

func (lc *LocalCache) evictEntries(sizeNeeded int64) {
    freedSize := int64(0)
    
    for freedSize < sizeNeeded {
        key := lc.eviction.GetLRUKey()
        if key == "" {
            break // No more entries to evict
        }
        
        if value, ok := lc.cache.Load(key); ok {
            entry := value.(*CacheEntry)
            freedSize += entry.Size
            lc.cache.Delete(key)
            lc.eviction.Remove(key)
            lc.stats.RecordEviction()
        }
    }
}

func (lc *LocalCache) startCleanup() {
    ticker := time.NewTicker(1 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        lc.cleanupExpired()
    }
}

func (lc *LocalCache) cleanupExpired() {
    now := time.Now()
    expiredKeys := make([]string, 0)
    
    lc.cache.Range(func(key, value interface{}) bool {
        entry := value.(*CacheEntry)
        if now.After(entry.ExpiresAt) {
            expiredKeys = append(expiredKeys, key.(string))
        }
        return true
    })
    
    for _, key := range expiredKeys {
        lc.cache.Delete(key)
        lc.eviction.Remove(key)
        lc.stats.RecordExpiration()
    }
}
```

### 1.3 Redis Cache (L2)

```go
package cache

import (
    "context"
    "encoding/json"
    "fmt"
    "time"
    
    "github.com/go-redis/redis/v8"
)

type RedisCache struct {
    client        redis.UniversalClient
    serializer    Serializer
    compressor    Compressor
    encryptor     Encryptor
    circuitBreaker *CircuitBreaker
    config        *RedisCacheConfig
    metrics       *RedisCacheMetrics
}

type RedisCacheConfig struct {
    Addresses        []string      `json:"addresses"`
    MasterName       string        `json:"master_name"`
    Password         string        `json:"password"`
    Database         int           `json:"database"`
    PoolSize         int           `json:"pool_size"`
    MinIdleConns     int           `json:"min_idle_conns"`
    MaxRetries       int           `json:"max_retries"`
    RetryDelay       time.Duration `json:"retry_delay"`
    DialTimeout      time.Duration `json:"dial_timeout"`
    ReadTimeout      time.Duration `json:"read_timeout"`
    WriteTimeout     time.Duration `json:"write_timeout"`
    EnableCluster    bool          `json:"enable_cluster"`
    EnableSentinel   bool          `json:"enable_sentinel"`
    EnableCompression bool         `json:"enable_compression"`
    EnableEncryption bool          `json:"enable_encryption"`
}

func NewRedisCache(config *RedisCacheConfig) (*RedisCache, error) {
    var client redis.UniversalClient
    
    if config.EnableCluster {
        client = redis.NewClusterClient(&redis.ClusterOptions{
            Addrs:        config.Addresses,
            Password:     config.Password,
            PoolSize:     config.PoolSize,
            MinIdleConns: config.MinIdleConns,
            MaxRetries:   config.MaxRetries,
            DialTimeout:  config.DialTimeout,
            ReadTimeout:  config.ReadTimeout,
            WriteTimeout: config.WriteTimeout,
        })
    } else if config.EnableSentinel {
        client = redis.NewFailoverClient(&redis.FailoverOptions{
            MasterName:       config.MasterName,
            SentinelAddrs:    config.Addresses,
            Password:         config.Password,
            DB:              config.Database,
            PoolSize:        config.PoolSize,
            MinIdleConns:    config.MinIdleConns,
            MaxRetries:      config.MaxRetries,
            DialTimeout:     config.DialTimeout,
            ReadTimeout:     config.ReadTimeout,
            WriteTimeout:    config.WriteTimeout,
        })
    } else {
        client = redis.NewClient(&redis.Options{
            Addr:         config.Addresses[0],
            Password:     config.Password,
            DB:          config.Database,
            PoolSize:    config.PoolSize,
            MinIdleConns: config.MinIdleConns,
            MaxRetries:   config.MaxRetries,
            DialTimeout:  config.DialTimeout,
            ReadTimeout:  config.ReadTimeout,
            WriteTimeout: config.WriteTimeout,
        })
    }
    
    // Test connection
    if err := client.Ping(context.Background()).Err(); err != nil {
        return nil, fmt.Errorf("failed to connect to Redis: %w", err)
    }
    
    rc := &RedisCache{
        client:       client,
        serializer:   NewJSONSerializer(),
        config:       config,
        metrics:      NewRedisCacheMetrics(),
        circuitBreaker: NewCircuitBreaker(CircuitBreakerConfig{
            MaxFailures: 5,
            Timeout:     30 * time.Second,
            Interval:    10 * time.Second,
        }),
    }
    
    if config.EnableCompression {
        rc.compressor = NewGzipCompressor()
    }
    
    if config.EnableEncryption {
        encryptor, err := NewAESEncryptor()
        if err != nil {
            return nil, fmt.Errorf("failed to create encryptor: %w", err)
        }
        rc.encryptor = encryptor
    }
    
    return rc, nil
}

func (rc *RedisCache) Get(ctx context.Context, key string) (interface{}, bool, error) {
    if !rc.circuitBreaker.Allow() {
        rc.metrics.RecordCircuitBreakerOpen()
        return nil, false, fmt.Errorf("circuit breaker is open")
    }
    
    startTime := time.Now()
    defer func() {
        rc.metrics.RecordLatency("get", time.Since(startTime))
    }()
    
    data, err := rc.client.Get(ctx, key).Bytes()
    if err != nil {
        if err == redis.Nil {
            rc.circuitBreaker.RecordSuccess()
            rc.metrics.RecordMiss()
            return nil, false, nil
        }
        rc.circuitBreaker.RecordFailure()
        rc.metrics.RecordError("get")
        return nil, false, fmt.Errorf("redis get failed: %w", err)
    }
    
    // Decrypt if encryption is enabled
    if rc.encryptor != nil {
        decrypted, err := rc.encryptor.Decrypt(data)
        if err != nil {
            return nil, false, fmt.Errorf("decryption failed: %w", err)
        }
        data = decrypted
    }
    
    // Decompress if compression is enabled
    if rc.compressor != nil {
        decompressed, err := rc.compressor.Decompress(data)
        if err != nil {
            return nil, false, fmt.Errorf("decompression failed: %w", err)
        }
        data = decompressed
    }
    
    // Deserialize
    var value interface{}
    if err := rc.serializer.Deserialize(data, &value); err != nil {
        return nil, false, fmt.Errorf("deserialization failed: %w", err)
    }
    
    rc.circuitBreaker.RecordSuccess()
    rc.metrics.RecordHit()
    return value, true, nil
}

func (rc *RedisCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    if !rc.circuitBreaker.Allow() {
        rc.metrics.RecordCircuitBreakerOpen()
        return fmt.Errorf("circuit breaker is open")
    }
    
    startTime := time.Now()
    defer func() {
        rc.metrics.RecordLatency("set", time.Since(startTime))
    }()
    
    // Serialize
    data, err := rc.serializer.Serialize(value)
    if err != nil {
        return fmt.Errorf("serialization failed: %w", err)
    }
    
    // Compress if compression is enabled
    if rc.compressor != nil {
        compressed, err := rc.compressor.Compress(data)
        if err != nil {
            return fmt.Errorf("compression failed: %w", err)
        }
        data = compressed
    }
    
    // Encrypt if encryption is enabled
    if rc.encryptor != nil {
        encrypted, err := rc.encryptor.Encrypt(data)
        if err != nil {
            return fmt.Errorf("encryption failed: %w", err)
        }
        data = encrypted
    }
    
    // Store in Redis
    err = rc.client.Set(ctx, key, data, ttl).Err()
    if err != nil {
        rc.circuitBreaker.RecordFailure()
        rc.metrics.RecordError("set")
        return fmt.Errorf("redis set failed: %w", err)
    }
    
    rc.circuitBreaker.RecordSuccess()
    rc.metrics.RecordSet()
    return nil
}

// Bulk operations for better performance
func (rc *RedisCache) MGet(ctx context.Context, keys ...string) (map[string]interface{}, error) {
    if len(keys) == 0 {
        return make(map[string]interface{}), nil
    }
    
    results, err := rc.client.MGet(ctx, keys...).Result()
    if err != nil {
        return nil, fmt.Errorf("redis mget failed: %w", err)
    }
    
    values := make(map[string]interface{})
    for i, result := range results {
        if result != nil {
            key := keys[i]
            data := []byte(result.(string))
            
            // Process data (decrypt, decompress, deserialize)
            processedValue, err := rc.processGetData(data)
            if err != nil {
                continue // Skip invalid entries
            }
            
            values[key] = processedValue
        }
    }
    
    return values, nil
}

func (rc *RedisCache) MSet(ctx context.Context, pairs map[string]interface{}, ttl time.Duration) error {
    pipe := rc.client.Pipeline()
    
    for key, value := range pairs {
        // Process data (serialize, compress, encrypt)
        data, err := rc.processSetData(value)
        if err != nil {
            return fmt.Errorf("failed to process data for key %s: %w", key, err)
        }
        
        pipe.Set(ctx, key, data, ttl)
    }
    
    _, err := pipe.Exec(ctx)
    if err != nil {
        return fmt.Errorf("redis pipeline exec failed: %w", err)
    }
    
    return nil
}
```

## 2. VM Metadata Caching

### 2.1 VM State Cache

```go
type VMMetadataCache struct {
    cache          *CacheManager
    eventListener  *VMEventListener
    keyGenerator   *VMCacheKeyGenerator
    config         *VMCacheConfig
}

type VMCacheConfig struct {
    StateCacheTTL      time.Duration `json:"state_cache_ttl"`
    MetricsCacheTTL    time.Duration `json:"metrics_cache_ttl"`
    ConfigCacheTTL     time.Duration `json:"config_cache_ttl"`
    EnableRealTimeSync bool          `json:"enable_real_time_sync"`
    BatchSize          int           `json:"batch_size"`
    SyncInterval       time.Duration `json:"sync_interval"`
}

func NewVMMetadataCache(cache *CacheManager, config *VMCacheConfig) *VMMetadataCache {
    vmc := &VMMetadataCache{
        cache:        cache,
        keyGenerator: NewVMCacheKeyGenerator(),
        config:       config,
    }
    
    if config.EnableRealTimeSync {
        vmc.eventListener = NewVMEventListener(vmc)
        vmc.startEventProcessing()
    }
    
    return vmc
}

func (vmc *VMMetadataCache) GetVMState(ctx context.Context, vmID string) (*VMState, error) {
    key := vmc.keyGenerator.VMStateKey(vmID)
    
    if value, found, err := vmc.cache.Get(ctx, key); err == nil && found {
        if state, ok := value.(*VMState); ok {
            return state, nil
        }
    }
    
    // Cache miss - fetch from source
    state, err := vmc.fetchVMStateFromSource(ctx, vmID)
    if err != nil {
        return nil, fmt.Errorf("failed to fetch VM state: %w", err)
    }
    
    // Cache the result
    go vmc.cache.Set(ctx, key, state, vmc.config.StateCacheTTL)
    
    return state, nil
}

func (vmc *VMMetadataCache) GetVMMetrics(ctx context.Context, vmID string, timeRange TimeRange) (*VMMetrics, error) {
    key := vmc.keyGenerator.VMMetricsKey(vmID, timeRange)
    
    if value, found, err := vmc.cache.Get(ctx, key); err == nil && found {
        if metrics, ok := value.(*VMMetrics); ok {
            return metrics, nil
        }
    }
    
    // Cache miss - fetch from monitoring system
    metrics, err := vmc.fetchVMMetricsFromSource(ctx, vmID, timeRange)
    if err != nil {
        return nil, fmt.Errorf("failed to fetch VM metrics: %w", err)
    }
    
    // Cache with shorter TTL for metrics
    go vmc.cache.Set(ctx, key, metrics, vmc.config.MetricsCacheTTL)
    
    return metrics, nil
}

func (vmc *VMMetadataCache) GetVMConfiguration(ctx context.Context, vmID string) (*VMConfiguration, error) {
    key := vmc.keyGenerator.VMConfigKey(vmID)
    
    if value, found, err := vmc.cache.Get(ctx, key); err == nil && found {
        if config, ok := value.(*VMConfiguration); ok {
            return config, nil
        }
    }
    
    // Cache miss - fetch from database
    config, err := vmc.fetchVMConfigFromSource(ctx, vmID)
    if err != nil {
        return nil, fmt.Errorf("failed to fetch VM configuration: %w", err)
    }
    
    // Cache with longer TTL for configuration
    go vmc.cache.Set(ctx, key, config, vmc.config.ConfigCacheTTL)
    
    return config, nil
}

// Batch operations for better performance
func (vmc *VMMetadataCache) GetMultipleVMStates(ctx context.Context, vmIDs []string) (map[string]*VMState, error) {
    keys := make([]string, len(vmIDs))
    for i, vmID := range vmIDs {
        keys[i] = vmc.keyGenerator.VMStateKey(vmID)
    }
    
    // Try to get from cache first
    cached := make(map[string]*VMState)
    missing := make([]string, 0)
    
    cachedResults, err := vmc.cache.MGet(ctx, keys...)
    if err == nil {
        for i, vmID := range vmIDs {
            key := keys[i]
            if value, exists := cachedResults[key]; exists {
                if state, ok := value.(*VMState); ok {
                    cached[vmID] = state
                    continue
                }
            }
            missing = append(missing, vmID)
        }
    } else {
        missing = vmIDs
    }
    
    // Fetch missing VMs from source
    if len(missing) > 0 {
        fetchedStates, err := vmc.fetchMultipleVMStatesFromSource(ctx, missing)
        if err != nil {
            return nil, fmt.Errorf("failed to fetch missing VM states: %w", err)
        }
        
        // Merge results and cache the missing ones
        cacheUpdates := make(map[string]interface{})
        for vmID, state := range fetchedStates {
            cached[vmID] = state
            key := vmc.keyGenerator.VMStateKey(vmID)
            cacheUpdates[key] = state
        }
        
        // Bulk cache update
        go vmc.cache.MSet(ctx, cacheUpdates, vmc.config.StateCacheTTL)
    }
    
    return cached, nil
}

// Event-driven cache invalidation
func (vmc *VMMetadataCache) OnVMStateChange(event *VMStateChangeEvent) {
    ctx := context.Background()
    
    // Invalidate affected cache entries
    keys := vmc.getAffectedCacheKeys(event)
    for _, key := range keys {
        vmc.cache.Delete(ctx, key)
    }
    
    // Pre-populate cache with new state if available
    if event.NewState != nil {
        stateKey := vmc.keyGenerator.VMStateKey(event.VMID)
        go vmc.cache.Set(ctx, stateKey, event.NewState, vmc.config.StateCacheTTL)
    }
}
```

### 2.2 Resource Allocation Cache

```go
type ResourceAllocationCache struct {
    cache          *CacheManager
    scheduler      *Scheduler
    keyGenerator   *ResourceCacheKeyGenerator
    config         *ResourceCacheConfig
}

type ResourceCacheConfig struct {
    AllocationCacheTTL   time.Duration `json:"allocation_cache_ttl"`
    CapacityCacheTTL     time.Duration `json:"capacity_cache_ttl"`
    ScoreCacheTTL        time.Duration `json:"score_cache_ttl"`
    EnablePredictiveCache bool         `json:"enable_predictive_cache"`
    PreCacheThreshold    float64       `json:"pre_cache_threshold"`
}

func NewResourceAllocationCache(cache *CacheManager, scheduler *Scheduler, config *ResourceCacheConfig) *ResourceAllocationCache {
    rac := &ResourceAllocationCache{
        cache:        cache,
        scheduler:    scheduler,
        keyGenerator: NewResourceCacheKeyGenerator(),
        config:       config,
    }
    
    if config.EnablePredictiveCache {
        go rac.startPredictiveCaching()
    }
    
    return rac
}

func (rac *ResourceAllocationCache) GetNodeCapacity(ctx context.Context, nodeID string) (*NodeCapacity, error) {
    key := rac.keyGenerator.NodeCapacityKey(nodeID)
    
    if value, found, err := rac.cache.Get(ctx, key); err == nil && found {
        if capacity, ok := value.(*NodeCapacity); ok {
            // Check if capacity data is still fresh
            if time.Since(capacity.LastUpdated) < time.Minute {
                return capacity, nil
            }
        }
    }
    
    // Fetch fresh capacity data
    capacity, err := rac.fetchNodeCapacityFromSource(ctx, nodeID)
    if err != nil {
        return nil, fmt.Errorf("failed to fetch node capacity: %w", err)
    }
    
    // Cache with appropriate TTL
    go rac.cache.Set(ctx, key, capacity, rac.config.CapacityCacheTTL)
    
    return capacity, nil
}

func (rac *ResourceAllocationCache) GetAllocationScore(ctx context.Context, vmSpec *VMSpecification, nodeID string) (*AllocationScore, error) {
    key := rac.keyGenerator.AllocationScoreKey(vmSpec, nodeID)
    
    if value, found, err := rac.cache.Get(ctx, key); err == nil && found {
        if score, ok := value.(*AllocationScore); ok {
            return score, nil
        }
    }
    
    // Calculate allocation score
    score, err := rac.calculateAllocationScore(ctx, vmSpec, nodeID)
    if err != nil {
        return nil, fmt.Errorf("failed to calculate allocation score: %w", err)
    }
    
    // Cache the score
    go rac.cache.Set(ctx, key, score, rac.config.ScoreCacheTTL)
    
    return score, nil
}

func (rac *ResourceAllocationCache) GetOptimalNodes(ctx context.Context, vmSpec *VMSpecification, count int) ([]*NodeRecommendation, error) {
    key := rac.keyGenerator.OptimalNodesKey(vmSpec, count)
    
    if value, found, err := rac.cache.Get(ctx, key); err == nil && found {
        if recommendations, ok := value.([]*NodeRecommendation); ok {
            // Validate recommendations are still valid
            if rac.areRecommendationsValid(recommendations) {
                return recommendations, nil
            }
        }
    }
    
    // Calculate optimal nodes
    recommendations, err := rac.calculateOptimalNodes(ctx, vmSpec, count)
    if err != nil {
        return nil, fmt.Errorf("failed to calculate optimal nodes: %w", err)
    }
    
    // Cache recommendations
    go rac.cache.Set(ctx, key, recommendations, rac.config.AllocationCacheTTL)
    
    return recommendations, nil
}

// Predictive caching for common resource queries
func (rac *ResourceAllocationCache) startPredictiveCaching() {
    ticker := time.NewTicker(5 * time.Minute)
    defer ticker.Stop()
    
    for range ticker.C {
        rac.performPredictiveCaching(context.Background())
    }
}

func (rac *ResourceAllocationCache) performPredictiveCaching(ctx context.Context) {
    // Analyze access patterns
    patterns := rac.analyzeAccessPatterns()
    
    for _, pattern := range patterns {
        if pattern.Probability > rac.config.PreCacheThreshold {
            go rac.preCacheResource(ctx, pattern)
        }
    }
}

func (rac *ResourceAllocationCache) preCacheResource(ctx context.Context, pattern *AccessPattern) {
    switch pattern.Type {
    case "node_capacity":
        capacity, _ := rac.fetchNodeCapacityFromSource(ctx, pattern.NodeID)
        if capacity != nil {
            key := rac.keyGenerator.NodeCapacityKey(pattern.NodeID)
            rac.cache.Set(ctx, key, capacity, rac.config.CapacityCacheTTL)
        }
    case "allocation_score":
        score, _ := rac.calculateAllocationScore(ctx, pattern.VMSpec, pattern.NodeID)
        if score != nil {
            key := rac.keyGenerator.AllocationScoreKey(pattern.VMSpec, pattern.NodeID)
            rac.cache.Set(ctx, key, score, rac.config.ScoreCacheTTL)
        }
    }
}
```

## 3. Migration State Caching

### 3.1 Migration Progress Cache

```go
type MigrationStateCache struct {
    cache          *CacheManager
    migrationEngine *MigrationEngine
    keyGenerator   *MigrationCacheKeyGenerator
    config         *MigrationCacheConfig
    eventStream    chan *MigrationEvent
}

type MigrationCacheConfig struct {
    StateCacheTTL       time.Duration `json:"state_cache_ttl"`
    ProgressCacheTTL    time.Duration `json:"progress_cache_ttl"`
    CheckpointCacheTTL  time.Duration `json:"checkpoint_cache_ttl"`
    EnableStreaming     bool          `json:"enable_streaming"`
    StreamBufferSize    int           `json:"stream_buffer_size"`
}

func NewMigrationStateCache(cache *CacheManager, migrationEngine *MigrationEngine, config *MigrationCacheConfig) *MigrationStateCache {
    msc := &MigrationStateCache{
        cache:           cache,
        migrationEngine: migrationEngine,
        keyGenerator:   NewMigrationCacheKeyGenerator(),
        config:         config,
        eventStream:    make(chan *MigrationEvent, config.StreamBufferSize),
    }
    
    if config.EnableStreaming {
        go msc.processEventStream()
    }
    
    return msc
}

func (msc *MigrationStateCache) GetMigrationState(ctx context.Context, migrationID string) (*MigrationState, error) {
    key := msc.keyGenerator.MigrationStateKey(migrationID)
    
    if value, found, err := msc.cache.Get(ctx, key); err == nil && found {
        if state, ok := value.(*MigrationState); ok {
            return state, nil
        }
    }
    
    // Fetch from migration engine
    state, err := msc.migrationEngine.GetMigrationState(migrationID)
    if err != nil {
        return nil, fmt.Errorf("failed to get migration state: %w", err)
    }
    
    // Cache the state
    go msc.cache.Set(ctx, key, state, msc.config.StateCacheTTL)
    
    return state, nil
}

func (msc *MigrationStateCache) GetMigrationProgress(ctx context.Context, migrationID string) (*MigrationProgress, error) {
    key := msc.keyGenerator.MigrationProgressKey(migrationID)
    
    if value, found, err := msc.cache.Get(ctx, key); err == nil && found {
        if progress, ok := value.(*MigrationProgress); ok {
            return progress, nil
        }
    }
    
    // Calculate current progress
    progress, err := msc.calculateMigrationProgress(ctx, migrationID)
    if err != nil {
        return nil, fmt.Errorf("failed to calculate migration progress: %w", err)
    }
    
    // Cache with short TTL for real-time updates
    go msc.cache.Set(ctx, key, progress, msc.config.ProgressCacheTTL)
    
    return progress, nil
}

func (msc *MigrationStateCache) StoreMigrationCheckpoint(ctx context.Context, migrationID string, checkpoint *MigrationCheckpoint) error {
    key := msc.keyGenerator.MigrationCheckpointKey(migrationID, checkpoint.SequenceNumber)
    
    // Store checkpoint in cache
    err := msc.cache.Set(ctx, key, checkpoint, msc.config.CheckpointCacheTTL)
    if err != nil {
        return fmt.Errorf("failed to cache migration checkpoint: %w", err)
    }
    
    // Also store in persistent storage asynchronously
    go msc.persistCheckpoint(checkpoint)
    
    return nil
}

func (msc *MigrationStateCache) GetMigrationCheckpoints(ctx context.Context, migrationID string) ([]*MigrationCheckpoint, error) {
    // Use pattern matching to get all checkpoints for this migration
    pattern := msc.keyGenerator.MigrationCheckpointPattern(migrationID)
    
    checkpoints, err := msc.cache.GetByPattern(ctx, pattern)
    if err != nil {
        return nil, fmt.Errorf("failed to get migration checkpoints: %w", err)
    }
    
    // Convert to checkpoint objects and sort by sequence number
    result := make([]*MigrationCheckpoint, 0, len(checkpoints))
    for _, value := range checkpoints {
        if checkpoint, ok := value.(*MigrationCheckpoint); ok {
            result = append(result, checkpoint)
        }
    }
    
    // Sort by sequence number
    sort.Slice(result, func(i, j int) bool {
        return result[i].SequenceNumber < result[j].SequenceNumber
    })
    
    return result, nil
}

// Real-time migration event processing
func (msc *MigrationStateCache) processEventStream() {
    for event := range msc.eventStream {
        ctx := context.Background()
        msc.handleMigrationEvent(ctx, event)
    }
}

func (msc *MigrationStateCache) handleMigrationEvent(ctx context.Context, event *MigrationEvent) {
    switch event.Type {
    case MigrationStarted:
        msc.cacheMigrationStart(ctx, event)
    case MigrationProgressUpdated:
        msc.cacheMigrationProgress(ctx, event)
    case MigrationCheckpointed:
        msc.cacheMigrationCheckpoint(ctx, event)
    case MigrationCompleted:
        msc.cacheMigrationCompletion(ctx, event)
    case MigrationFailed:
        msc.cacheMigrationFailure(ctx, event)
    }
}

func (msc *MigrationStateCache) cacheMigrationProgress(ctx context.Context, event *MigrationEvent) {
    key := msc.keyGenerator.MigrationProgressKey(event.MigrationID)
    
    progress := &MigrationProgress{
        MigrationID:      event.MigrationID,
        PercentComplete:  event.Progress.PercentComplete,
        BytesTransferred: event.Progress.BytesTransferred,
        TotalBytes:       event.Progress.TotalBytes,
        TransferRate:     event.Progress.TransferRate,
        EstimatedTimeRemaining: event.Progress.EstimatedTimeRemaining,
        LastUpdated:      time.Now(),
    }
    
    // Cache with very short TTL for real-time updates
    msc.cache.Set(ctx, key, progress, 5*time.Second)
    
    // Also invalidate related caches
    stateKey := msc.keyGenerator.MigrationStateKey(event.MigrationID)
    msc.cache.Delete(ctx, stateKey)
}

// Streaming API for real-time migration updates
func (msc *MigrationStateCache) StreamMigrationUpdates(ctx context.Context, migrationID string) (<-chan *MigrationUpdate, error) {
    updateChan := make(chan *MigrationUpdate, 100)
    
    go func() {
        defer close(updateChan)
        
        ticker := time.NewTicker(1 * time.Second)
        defer ticker.Stop()
        
        lastUpdate := time.Time{}
        
        for {
            select {
            case <-ctx.Done():
                return
            case <-ticker.C:
                // Check for progress updates
                progress, err := msc.GetMigrationProgress(ctx, migrationID)
                if err != nil {
                    continue
                }
                
                if progress.LastUpdated.After(lastUpdate) {
                    update := &MigrationUpdate{
                        Type:         ProgressUpdate,
                        MigrationID:  migrationID,
                        Progress:     progress,
                        Timestamp:    time.Now(),
                    }
                    
                    select {
                    case updateChan <- update:
                        lastUpdate = progress.LastUpdated
                    case <-ctx.Done():
                        return
                    }
                }
            }
        }
    }()
    
    return updateChan, nil
}
```

## 4. Cache Invalidation Strategies

### 4.1 Intelligent Invalidation Manager

```go
type InvalidationManager struct {
    strategies    map[string]InvalidationStrategy
    eventBus      *EventBus
    dependencies  *DependencyGraph
    batching      *BatchInvalidator
    metrics       *InvalidationMetrics
}

type InvalidationStrategy interface {
    ShouldInvalidate(event *CacheEvent) bool
    GetAffectedKeys(event *CacheEvent) []string
    GetPriority() int
}

// Time-based invalidation
type TTLInvalidationStrategy struct {
    defaultTTL time.Duration
    ttlMap     map[string]time.Duration
}

func (ttl *TTLInvalidationStrategy) ShouldInvalidate(event *CacheEvent) bool {
    if event.Type != CacheEntryExpired {
        return false
    }
    
    now := time.Now()
    expiryTime := event.Timestamp.Add(ttl.getTTL(event.Key))
    return now.After(expiryTime)
}

// Event-driven invalidation
type EventDrivenInvalidationStrategy struct {
    eventMappings map[string][]string // event type -> cache key patterns
}

func (eds *EventDrivenInvalidationStrategy) ShouldInvalidate(event *CacheEvent) bool {
    patterns, exists := eds.eventMappings[event.Type]
    return exists && len(patterns) > 0
}

func (eds *EventDrivenInvalidationStrategy) GetAffectedKeys(event *CacheEvent) []string {
    patterns := eds.eventMappings[event.Type]
    keys := make([]string, 0)
    
    for _, pattern := range patterns {
        // Replace placeholders with actual values from event
        actualPattern := eds.substituteVariables(pattern, event.Data)
        matchingKeys := eds.findMatchingKeys(actualPattern)
        keys = append(keys, matchingKeys...)
    }
    
    return keys
}

// Dependency-based invalidation
type DependencyInvalidationStrategy struct {
    dependencyGraph *CacheDependencyGraph
}

func (dis *DependencyInvalidationStrategy) ShouldInvalidate(event *CacheEvent) bool {
    return event.Type == CacheEntryUpdated || event.Type == CacheEntryDeleted
}

func (dis *DependencyInvalidationStrategy) GetAffectedKeys(event *CacheEvent) []string {
    // Find all keys that depend on the changed key
    dependentKeys := dis.dependencyGraph.GetDependentKeys(event.Key)
    
    // Include transitive dependencies
    allAffectedKeys := make([]string, 0)
    visited := make(map[string]bool)
    
    var collectDependencies func(string)
    collectDependencies = func(key string) {
        if visited[key] {
            return
        }
        visited[key] = true
        allAffectedKeys = append(allAffectedKeys, key)
        
        for _, dependentKey := range dis.dependencyGraph.GetDependentKeys(key) {
            collectDependencies(dependentKey)
        }
    }
    
    for _, key := range dependentKeys {
        collectDependencies(key)
    }
    
    return allAffectedKeys
}

// Smart invalidation based on access patterns and data freshness
type SmartInvalidationStrategy struct {
    accessTracker    *AccessPatternTracker
    freshnessAnalyzer *DataFreshnessAnalyzer
    costCalculator   *InvalidationCostCalculator
}

func (sis *SmartInvalidationStrategy) ShouldInvalidate(event *CacheEvent) bool {
    // Analyze access patterns
    accessPattern := sis.accessTracker.GetPattern(event.Key)
    
    // Check data freshness
    freshness := sis.freshnessAnalyzer.GetFreshness(event.Key)
    
    // Calculate invalidation cost vs benefit
    cost := sis.costCalculator.CalculateInvalidationCost(event.Key)
    benefit := sis.calculateInvalidationBenefit(accessPattern, freshness)
    
    return benefit > cost
}

func (sis *SmartInvalidationStrategy) calculateInvalidationBenefit(pattern *AccessPattern, freshness *DataFreshness) float64 {
    // High access frequency + stale data = high benefit
    accessWeight := pattern.AccessFrequency / pattern.MaxAccessFrequency
    stalenessWeight := freshness.StalenessScore
    
    return accessWeight * stalenessWeight
}

// Batch invalidation for performance
type BatchInvalidator struct {
    batchSize      int
    batchTimeout   time.Duration
    pendingKeys    []string
    pendingMutex   sync.Mutex
    flushTicker    *time.Ticker
    cacheManager   *CacheManager
}

func NewBatchInvalidator(batchSize int, batchTimeout time.Duration, cacheManager *CacheManager) *BatchInvalidator {
    bi := &BatchInvalidator{
        batchSize:    batchSize,
        batchTimeout: batchTimeout,
        cacheManager: cacheManager,
        pendingKeys:  make([]string, 0, batchSize),
        flushTicker:  time.NewTicker(batchTimeout),
    }
    
    go bi.startBatchProcessor()
    return bi
}

func (bi *BatchInvalidator) InvalidateKey(key string) {
    bi.pendingMutex.Lock()
    defer bi.pendingMutex.Unlock()
    
    bi.pendingKeys = append(bi.pendingKeys, key)
    
    if len(bi.pendingKeys) >= bi.batchSize {
        go bi.flushBatch()
    }
}

func (bi *BatchInvalidator) flushBatch() {
    bi.pendingMutex.Lock()
    if len(bi.pendingKeys) == 0 {
        bi.pendingMutex.Unlock()
        return
    }
    
    keys := make([]string, len(bi.pendingKeys))
    copy(keys, bi.pendingKeys)
    bi.pendingKeys = bi.pendingKeys[:0]
    bi.pendingMutex.Unlock()
    
    // Perform batch invalidation
    ctx := context.Background()
    bi.cacheManager.DeleteMultiple(ctx, keys)
}

func (bi *BatchInvalidator) startBatchProcessor() {
    for range bi.flushTicker.C {
        bi.flushBatch()
    }
}
```

### 4.2 Cache Consistency Manager

```go
type CacheConsistencyManager struct {
    caches           []CacheLayer
    consistencyModel ConsistencyModel
    conflictResolver *ConflictResolver
    versionManager   *VersionManager
}

type ConsistencyModel int

const (
    EventualConsistency ConsistencyModel = iota
    StrongConsistency
    SessionConsistency
    MonotonicReadConsistency
)

type VersionedCacheEntry struct {
    Value     interface{}   `json:"value"`
    Version   int64         `json:"version"`
    Timestamp time.Time     `json:"timestamp"`
    Checksum  string        `json:"checksum"`
    TTL       time.Duration `json:"ttl"`
}

func (ccm *CacheConsistencyManager) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    entry := &VersionedCacheEntry{
        Value:     value,
        Version:   ccm.versionManager.GetNextVersion(key),
        Timestamp: time.Now(),
        Checksum:  ccm.calculateChecksum(value),
        TTL:       ttl,
    }
    
    switch ccm.consistencyModel {
    case StrongConsistency:
        return ccm.setWithStrongConsistency(ctx, key, entry)
    case EventualConsistency:
        return ccm.setWithEventualConsistency(ctx, key, entry)
    case SessionConsistency:
        return ccm.setWithSessionConsistency(ctx, key, entry)
    default:
        return ccm.setWithEventualConsistency(ctx, key, entry)
    }
}

func (ccm *CacheConsistencyManager) setWithStrongConsistency(ctx context.Context, key string, entry *VersionedCacheEntry) error {
    // Use distributed locking to ensure atomicity
    lock, err := ccm.acquireDistributedLock(ctx, key)
    if err != nil {
        return fmt.Errorf("failed to acquire lock: %w", err)
    }
    defer lock.Release()
    
    // Set in all caches synchronously
    var wg sync.WaitGroup
    errors := make([]error, len(ccm.caches))
    
    for i, cache := range ccm.caches {
        wg.Add(1)
        go func(i int, cache CacheLayer) {
            defer wg.Done()
            errors[i] = cache.Set(ctx, key, entry, entry.TTL)
        }(i, cache)
    }
    
    wg.Wait()
    
    // Check for errors
    for _, err := range errors {
        if err != nil {
            // Rollback on any failure
            ccm.rollbackSet(ctx, key, entry)
            return fmt.Errorf("strong consistency set failed: %w", err)
        }
    }
    
    return nil
}

func (ccm *CacheConsistencyManager) setWithEventualConsistency(ctx context.Context, key string, entry *VersionedCacheEntry) error {
    // Set in primary cache first
    primaryCache := ccm.caches[0]
    if err := primaryCache.Set(ctx, key, entry, entry.TTL); err != nil {
        return fmt.Errorf("primary cache set failed: %w", err)
    }
    
    // Asynchronously propagate to other caches
    for _, cache := range ccm.caches[1:] {
        go func(cache CacheLayer) {
            cache.Set(ctx, key, entry, entry.TTL)
        }(cache)
    }
    
    return nil
}

func (ccm *CacheConsistencyManager) Get(ctx context.Context, key string) (*VersionedCacheEntry, error) {
    switch ccm.consistencyModel {
    case StrongConsistency:
        return ccm.getWithStrongConsistency(ctx, key)
    case MonotonicReadConsistency:
        return ccm.getWithMonotonicReadConsistency(ctx, key)
    default:
        return ccm.getFromBestCache(ctx, key)
    }
}

func (ccm *CacheConsistencyManager) getWithStrongConsistency(ctx context.Context, key string) (*VersionedCacheEntry, error) {
    // Read from all caches and resolve conflicts
    entries := make([]*VersionedCacheEntry, len(ccm.caches))
    var wg sync.WaitGroup
    
    for i, cache := range ccm.caches {
        wg.Add(1)
        go func(i int, cache CacheLayer) {
            defer wg.Done()
            if value, found, err := cache.Get(ctx, key); err == nil && found {
                if entry, ok := value.(*VersionedCacheEntry); ok {
                    entries[i] = entry
                }
            }
        }(i, cache)
    }
    
    wg.Wait()
    
    // Resolve conflicts and return the most recent version
    return ccm.conflictResolver.ResolveConflicts(entries), nil
}

func (ccm *CacheConsistencyManager) getWithMonotonicReadConsistency(ctx context.Context, key string) (*VersionedCacheEntry, error) {
    sessionID := ccm.getSessionID(ctx)
    lastVersion := ccm.versionManager.GetLastReadVersion(sessionID, key)
    
    for _, cache := range ccm.caches {
        if value, found, err := cache.Get(ctx, key); err == nil && found {
            if entry, ok := value.(*VersionedCacheEntry); ok {
                if entry.Version >= lastVersion {
                    ccm.versionManager.UpdateLastReadVersion(sessionID, key, entry.Version)
                    return entry, nil
                }
            }
        }
    }
    
    return nil, fmt.Errorf("no version found meeting monotonic read consistency")
}

type ConflictResolver struct {
    resolutionStrategy ConflictResolutionStrategy
}

type ConflictResolutionStrategy int

const (
    LastWriteWins ConflictResolutionStrategy = iota
    HighestVersion
    MostRecentTimestamp
    CustomMerge
)

func (cr *ConflictResolver) ResolveConflicts(entries []*VersionedCacheEntry) *VersionedCacheEntry {
    validEntries := make([]*VersionedCacheEntry, 0)
    for _, entry := range entries {
        if entry != nil {
            validEntries = append(validEntries, entry)
        }
    }
    
    if len(validEntries) == 0 {
        return nil
    }
    
    if len(validEntries) == 1 {
        return validEntries[0]
    }
    
    switch cr.resolutionStrategy {
    case HighestVersion:
        return cr.resolveByHighestVersion(validEntries)
    case MostRecentTimestamp:
        return cr.resolveByMostRecentTimestamp(validEntries)
    default:
        return cr.resolveByHighestVersion(validEntries)
    }
}

func (cr *ConflictResolver) resolveByHighestVersion(entries []*VersionedCacheEntry) *VersionedCacheEntry {
    maxVersion := int64(-1)
    var winner *VersionedCacheEntry
    
    for _, entry := range entries {
        if entry.Version > maxVersion {
            maxVersion = entry.Version
            winner = entry
        }
    }
    
    return winner
}
```

## 5. High Availability and Clustering

### 5.1 Redis Cluster Configuration

```yaml
# Redis Cluster Configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-cluster-config
  namespace: novacron
data:
  redis.conf: |
    # Cluster configuration
    cluster-enabled yes
    cluster-config-file nodes.conf
    cluster-node-timeout 15000
    cluster-announce-ip ${POD_IP}
    cluster-announce-port 6379
    cluster-announce-bus-port 16379
    
    # Memory configuration
    maxmemory 2gb
    maxmemory-policy allkeys-lru
    
    # Persistence
    save 900 1
    save 300 10
    save 60 10000
    
    # Network
    bind 0.0.0.0
    port 6379
    tcp-keepalive 60
    timeout 0
    
    # Performance
    tcp-backlog 511
    databases 16
    stop-writes-on-bgsave-error no
    
    # Security
    requirepass ${REDIS_PASSWORD}
    rename-command FLUSHDB ""
    rename-command FLUSHALL ""
    rename-command DEBUG ""
    rename-command CONFIG "CONFIG_8f3a9d2b1e"
    
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
  namespace: novacron
spec:
  serviceName: redis-cluster-service
  replicas: 6
  selector:
    matchLabels:
      app: redis-cluster
  template:
    metadata:
      labels:
        app: redis-cluster
    spec:
      containers:
      - name: redis
        image: redis:7.0-alpine
        ports:
        - containerPort: 6379
          name: client
        - containerPort: 16379
          name: gossip
        command:
        - redis-server
        - /etc/redis/redis.conf
        env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-auth
              key: password
        volumeMounts:
        - name: config
          mountPath: /etc/redis
        - name: data
          mountPath: /data
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 2Gi
        livenessProbe:
          exec:
            command:
            - redis-cli
            - -a
            - $(REDIS_PASSWORD)
            - ping
          initialDelaySeconds: 30
          periodSeconds: 5
        readinessProbe:
          exec:
            command:
            - redis-cli
            - -a
            - $(REDIS_PASSWORD)
            - ping
          initialDelaySeconds: 5
          periodSeconds: 1
      volumes:
      - name: config
        configMap:
          name: redis-cluster-config
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
---
apiVersion: v1
kind: Service
metadata:
  name: redis-cluster-service
  namespace: novacron
spec:
  type: ClusterIP
  clusterIP: None
  ports:
  - port: 6379
    targetPort: 6379
    name: client
  - port: 16379
    targetPort: 16379
    name: gossip
  selector:
    app: redis-cluster
```

### 5.2 Sentinel Configuration for High Availability

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-sentinel-config
  namespace: novacron
data:
  sentinel.conf: |
    port 26379
    dir /data
    
    # Master configuration
    sentinel monitor mymaster redis-master 6379 2
    sentinel auth-pass mymaster ${REDIS_PASSWORD}
    sentinel down-after-milliseconds mymaster 10000
    sentinel parallel-syncs mymaster 1
    sentinel failover-timeout mymaster 60000
    
    # Notification scripts
    sentinel notification-script mymaster /scripts/notify.sh
    sentinel client-reconfig-script mymaster /scripts/reconfig.sh
    
    # Logging
    logfile /var/log/sentinel.log
    loglevel notice
    
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: redis-sentinel
  namespace: novacron
spec:
  replicas: 3
  selector:
    matchLabels:
      app: redis-sentinel
  template:
    metadata:
      labels:
        app: redis-sentinel
    spec:
      containers:
      - name: sentinel
        image: redis:7.0-alpine
        ports:
        - containerPort: 26379
          name: sentinel
        command:
        - redis-sentinel
        - /etc/redis/sentinel.conf
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-auth
              key: password
        volumeMounts:
        - name: config
          mountPath: /etc/redis
        - name: scripts
          mountPath: /scripts
        resources:
          requests:
            cpu: 50m
            memory: 64Mi
          limits:
            cpu: 200m
            memory: 256Mi
      volumes:
      - name: config
        configMap:
          name: redis-sentinel-config
      - name: scripts
        configMap:
          name: redis-sentinel-scripts
          defaultMode: 0755
```

### 5.3 Cache Health Monitoring

```go
type CacheHealthMonitor struct {
    cacheManager    *CacheManager
    healthCheckers  []HealthChecker
    metrics         *HealthMetrics
    alertManager    *AlertManager
    config          *HealthConfig
}

type HealthConfig struct {
    CheckInterval     time.Duration `json:"check_interval"`
    FailureThreshold  int           `json:"failure_threshold"`
    RecoveryThreshold int           `json:"recovery_threshold"`
    AlertEnabled      bool          `json:"alert_enabled"`
    MetricsEnabled    bool          `json:"metrics_enabled"`
}

type HealthChecker interface {
    CheckHealth(ctx context.Context) HealthStatus
    GetName() string
    GetWeight() float64
}

type RedisHealthChecker struct {
    client  redis.UniversalClient
    timeout time.Duration
}

func (rhc *RedisHealthChecker) CheckHealth(ctx context.Context) HealthStatus {
    ctx, cancel := context.WithTimeout(ctx, rhc.timeout)
    defer cancel()
    
    start := time.Now()
    
    // Test basic connectivity
    pong, err := rhc.client.Ping(ctx).Result()
    if err != nil {
        return HealthStatus{
            Status:      StatusUnhealthy,
            Message:     fmt.Sprintf("ping failed: %v", err),
            Latency:     time.Since(start),
            LastChecked: time.Now(),
        }
    }
    
    if pong != "PONG" {
        return HealthStatus{
            Status:      StatusUnhealthy,
            Message:     "unexpected ping response",
            Latency:     time.Since(start),
            LastChecked: time.Now(),
        }
    }
    
    // Test read/write operations
    testKey := fmt.Sprintf("health_check_%d", time.Now().UnixNano())
    testValue := "health_check_value"
    
    // Write test
    if err := rhc.client.Set(ctx, testKey, testValue, time.Minute).Err(); err != nil {
        return HealthStatus{
            Status:      StatusDegraded,
            Message:     fmt.Sprintf("write test failed: %v", err),
            Latency:     time.Since(start),
            LastChecked: time.Now(),
        }
    }
    
    // Read test
    result, err := rhc.client.Get(ctx, testKey).Result()
    if err != nil {
        return HealthStatus{
            Status:      StatusDegraded,
            Message:     fmt.Sprintf("read test failed: %v", err),
            Latency:     time.Since(start),
            LastChecked: time.Now(),
        }
    }
    
    if result != testValue {
        return HealthStatus{
            Status:      StatusDegraded,
            Message:     "read test returned incorrect value",
            Latency:     time.Since(start),
            LastChecked: time.Now(),
        }
    }
    
    // Cleanup
    rhc.client.Del(ctx, testKey)
    
    return HealthStatus{
        Status:      StatusHealthy,
        Message:     "all tests passed",
        Latency:     time.Since(start),
        LastChecked: time.Now(),
    }
}

type LocalCacheHealthChecker struct {
    cache *LocalCache
}

func (lchc *LocalCacheHealthChecker) CheckHealth(ctx context.Context) HealthStatus {
    start := time.Now()
    
    // Test basic operations
    testKey := fmt.Sprintf("health_check_%d", time.Now().UnixNano())
    testValue := "health_check_value"
    
    // Write test
    if err := lchc.cache.Set(testKey, testValue, time.Minute); err != nil {
        return HealthStatus{
            Status:      StatusUnhealthy,
            Message:     fmt.Sprintf("local cache write failed: %v", err),
            Latency:     time.Since(start),
            LastChecked: time.Now(),
        }
    }
    
    // Read test
    value, found := lchc.cache.Get(testKey)
    if !found {
        return HealthStatus{
            Status:      StatusDegraded,
            Message:     "local cache read missed",
            Latency:     time.Since(start),
            LastChecked: time.Now(),
        }
    }
    
    if value != testValue {
        return HealthStatus{
            Status:      StatusDegraded,
            Message:     "local cache read returned incorrect value",
            Latency:     time.Since(start),
            LastChecked: time.Now(),
        }
    }
    
    return HealthStatus{
        Status:      StatusHealthy,
        Message:     "local cache healthy",
        Latency:     time.Since(start),
        LastChecked: time.Now(),
    }
}

func (chm *CacheHealthMonitor) StartMonitoring(ctx context.Context) error {
    ticker := time.NewTicker(chm.config.CheckInterval)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            chm.performHealthChecks(ctx)
        }
    }
}

func (chm *CacheHealthMonitor) performHealthChecks(ctx context.Context) {
    var wg sync.WaitGroup
    
    for _, checker := range chm.healthCheckers {
        wg.Add(1)
        go func(checker HealthChecker) {
            defer wg.Done()
            
            status := checker.CheckHealth(ctx)
            chm.processHealthStatus(checker.GetName(), status)
        }(checker)
    }
    
    wg.Wait()
}

func (chm *CacheHealthMonitor) processHealthStatus(checkerName string, status HealthStatus) {
    // Record metrics
    if chm.config.MetricsEnabled {
        chm.metrics.RecordHealthCheck(checkerName, status)
    }
    
    // Check for alerts
    if chm.config.AlertEnabled {
        chm.checkForAlerts(checkerName, status)
    }
    
    // Update overall health status
    chm.updateOverallHealth(checkerName, status)
}

func (chm *CacheHealthMonitor) checkForAlerts(checkerName string, status HealthStatus) {
    if status.Status == StatusUnhealthy {
        alert := &Alert{
            Type:        "cache_health",
            Severity:    "critical",
            Source:      checkerName,
            Message:     status.Message,
            Timestamp:   time.Now(),
            Details: map[string]interface{}{
                "latency":      status.Latency.String(),
                "last_checked": status.LastChecked,
            },
        }
        
        chm.alertManager.TriggerAlert(alert)
    } else if status.Status == StatusDegraded {
        alert := &Alert{
            Type:        "cache_performance",
            Severity:    "warning",
            Source:      checkerName,
            Message:     status.Message,
            Timestamp:   time.Now(),
            Details: map[string]interface{}{
                "latency":      status.Latency.String(),
                "last_checked": status.LastChecked,
            },
        }
        
        chm.alertManager.TriggerAlert(alert)
    }
}
```

## 6. Performance Optimization and Monitoring

### 6.1 Cache Performance Metrics

```go
type CacheMetrics struct {
    hitCount         prometheus.CounterVec
    missCount        prometheus.CounterVec
    errorCount       prometheus.CounterVec
    latencyHistogram prometheus.HistogramVec
    sizeGauge        prometheus.GaugeVec
    evictionCount    prometheus.CounterVec
    throughput       prometheus.CounterVec
}

func NewCacheMetrics() *CacheMetrics {
    return &CacheMetrics{
        hitCount: *prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "cache_hits_total",
                Help: "Total number of cache hits",
            },
            []string{"cache_layer", "cache_type"},
        ),
        
        missCount: *prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "cache_misses_total",
                Help: "Total number of cache misses",
            },
            []string{"cache_layer", "cache_type"},
        ),
        
        latencyHistogram: *prometheus.NewHistogramVec(
            prometheus.HistogramOpts{
                Name:    "cache_operation_duration_seconds",
                Help:    "Duration of cache operations",
                Buckets: prometheus.ExponentialBuckets(0.001, 2, 10),
            },
            []string{"operation", "cache_layer"},
        ),
        
        sizeGauge: *prometheus.NewGaugeVec(
            prometheus.GaugeOpts{
                Name: "cache_size_bytes",
                Help: "Current size of cache in bytes",
            },
            []string{"cache_layer", "cache_type"},
        ),
        
        throughput: *prometheus.NewCounterVec(
            prometheus.CounterOpts{
                Name: "cache_throughput_ops_total",
                Help: "Total number of cache operations",
            },
            []string{"operation", "cache_layer"},
        ),
    }
}

func (cm *CacheMetrics) RecordHit(cacheLayer string) {
    cm.hitCount.WithLabelValues(cacheLayer, "general").Inc()
}

func (cm *CacheMetrics) RecordMiss(cacheLayer string) {
    cm.missCount.WithLabelValues(cacheLayer, "general").Inc()
}

func (cm *CacheMetrics) RecordLatency(operation string, cacheLayer string, duration time.Duration) {
    cm.latencyHistogram.WithLabelValues(operation, cacheLayer).Observe(duration.Seconds())
}

func (cm *CacheMetrics) UpdateCacheSize(cacheLayer string, size int64) {
    cm.sizeGauge.WithLabelValues(cacheLayer, "general").Set(float64(size))
}
```

### 6.2 Cache Performance Dashboard

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cache-dashboard
  namespace: monitoring
data:
  cache-dashboard.json: |
    {
      "dashboard": {
        "id": null,
        "title": "NovaCron Cache Performance",
        "tags": ["novacron", "cache"],
        "timezone": "browser",
        "panels": [
          {
            "id": 1,
            "title": "Cache Hit Rate",
            "type": "stat",
            "targets": [
              {
                "expr": "rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))",
                "legendFormat": "{{cache_layer}}"
              }
            ],
            "fieldConfig": {
              "defaults": {
                "unit": "percentunit",
                "min": 0,
                "max": 1,
                "thresholds": {
                  "steps": [
                    {"color": "red", "value": 0},
                    {"color": "yellow", "value": 0.7},
                    {"color": "green", "value": 0.9}
                  ]
                }
              }
            },
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
          },
          {
            "id": 2,
            "title": "Cache Latency",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(cache_operation_duration_seconds_bucket[5m]))",
                "legendFormat": "p95 {{cache_layer}} {{operation}}"
              },
              {
                "expr": "histogram_quantile(0.50, rate(cache_operation_duration_seconds_bucket[5m]))",
                "legendFormat": "p50 {{cache_layer}} {{operation}}"
              }
            ],
            "yAxes": [
              {"unit": "s", "min": 0}
            ],
            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
          },
          {
            "id": 3,
            "title": "Cache Throughput",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(cache_throughput_ops_total[5m])",
                "legendFormat": "{{operation}} {{cache_layer}}"
              }
            ],
            "yAxes": [
              {"unit": "ops", "min": 0}
            ],
            "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8}
          }
        ],
        "time": {"from": "now-1h", "to": "now"},
        "refresh": "5s"
      }
    }
```

## 7. API Integration

### 7.1 Cache Management APIs

```go
// GET /api/cache/stats
type CacheStatsResponse struct {
    LocalCache  CacheLayerStats `json:"local_cache"`
    RedisCache  CacheLayerStats `json:"redis_cache"`
    DistCache   CacheLayerStats `json:"distributed_cache"`
    Overall     OverallStats    `json:"overall"`
}

type CacheLayerStats struct {
    HitCount      int64         `json:"hit_count"`
    MissCount     int64         `json:"miss_count"`
    HitRate       float64       `json:"hit_rate"`
    AvgLatency    time.Duration `json:"avg_latency"`
    Size          int64         `json:"size_bytes"`
    EntryCount    int64         `json:"entry_count"`
    EvictionCount int64         `json:"eviction_count"`
    ErrorCount    int64         `json:"error_count"`
}

// POST /api/cache/invalidate
type CacheInvalidationRequest struct {
    Keys     []string `json:"keys,omitempty"`
    Patterns []string `json:"patterns,omitempty"`
    Tags     []string `json:"tags,omitempty"`
    Force    bool     `json:"force"`
}

// GET /api/cache/health
type CacheHealthResponse struct {
    Status      string            `json:"status"`
    Layers      []LayerHealth     `json:"layers"`
    LastCheck   time.Time         `json:"last_check"`
    Uptime      time.Duration     `json:"uptime"`
    Metrics     HealthMetrics     `json:"metrics"`
}

// POST /api/cache/warmup
type CacheWarmupRequest struct {
    Type        string                 `json:"type"` // vm_metadata, resource_allocation, etc.
    Scope       string                 `json:"scope"` // node, cluster, all
    Targets     []string               `json:"targets,omitempty"`
    Priority    int                    `json:"priority"`
    Parameters  map[string]interface{} `json:"parameters,omitempty"`
}
```

## Conclusion

This comprehensive Redis caching architecture transforms NovaCron's performance through:

1. **Multi-Tier Caching**: L1 local memory, L2 Redis cluster, L3 distributed storage
2. **Intelligent Cache Management**: Smart routing, invalidation, and consistency
3. **High Availability**: Redis clustering with Sentinel failover
4. **Performance Optimization**: Compression, encryption, batch operations
5. **Comprehensive Monitoring**: Real-time metrics and health checks
6. **Event-Driven Updates**: Real-time cache invalidation and updates

The architecture scales horizontally, provides sub-millisecond access for frequently used data, and maintains strong consistency guarantees where needed while offering eventual consistency for better performance where appropriate.