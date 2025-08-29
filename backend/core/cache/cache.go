package cache

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// Cache interface defines the caching contract
type Cache interface {
	Get(ctx context.Context, key string) ([]byte, error)
	Set(ctx context.Context, key string, value []byte, ttl time.Duration) error
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)
	Clear(ctx context.Context) error
	GetMulti(ctx context.Context, keys []string) (map[string][]byte, error)
	SetMulti(ctx context.Context, items map[string]CacheItem) error
	DeleteMulti(ctx context.Context, keys []string) error
	GetStats() CacheStats
	Close() error
}

// CacheItem represents an item to be cached
type CacheItem struct {
	Value []byte        `json:"value"`
	TTL   time.Duration `json:"ttl"`
}

// CacheStats provides cache performance metrics
type CacheStats struct {
	Hits              uint64    `json:"hits"`
	Misses            uint64    `json:"misses"`
	Sets              uint64    `json:"sets"`
	Deletes           uint64    `json:"deletes"`
	Errors            uint64    `json:"errors"`
	HitRate           float64   `json:"hit_rate"`
	L1Hits            uint64    `json:"l1_hits"`
	L2Hits            uint64    `json:"l2_hits"`
	L3Hits            uint64    `json:"l3_hits"`
	AvgResponseTimeNs int64     `json:"avg_response_time_ns"`
	LastUpdated       time.Time `json:"last_updated"`
}

// CacheConfig holds cache configuration
type CacheConfig struct {
	// L1 Cache (Memory) configuration
	L1Enabled    bool          `json:"l1_enabled"`
	L1MaxSize    int           `json:"l1_max_size"`    // Max items in L1
	L1TTL        time.Duration `json:"l1_ttl"`         // Default L1 TTL
	L1CleanupInt time.Duration `json:"l1_cleanup_int"` // Cleanup interval

	// L2 Cache (Redis) configuration
	L2Enabled       bool     `json:"l2_enabled"`
	RedisAddrs      []string `json:"redis_addrs"`
	RedisPassword   string   `json:"redis_password"`
	RedisDB         int      `json:"redis_db"`
	RedisCluster    bool     `json:"redis_cluster"`
	SentinelEnabled bool     `json:"sentinel_enabled"`
	SentinelAddrs   []string `json:"sentinel_addrs"`
	SentinelMaster  string   `json:"sentinel_master"`
	PoolSize        int      `json:"pool_size"`
	MinIdleConns    int      `json:"min_idle_conns"`
	MaxRetries      int      `json:"max_retries"`
	RetryDelay      time.Duration `json:"retry_delay"`
	ReadTimeout     time.Duration `json:"read_timeout"`
	WriteTimeout    time.Duration `json:"write_timeout"`
	DialTimeout     time.Duration `json:"dial_timeout"`

	// L3 Cache (Persistent Storage) configuration  
	L3Enabled bool   `json:"l3_enabled"`
	L3Path    string `json:"l3_path"` // File system path for L3

	// General configuration
	DefaultTTL        time.Duration `json:"default_ttl"`
	EnableCompression bool          `json:"enable_compression"`
	CompressionLevel  int           `json:"compression_level"`
	Serialization     string        `json:"serialization"` // json, gob, msgpack
	EnableMetrics     bool          `json:"enable_metrics"`
	MetricsInterval   time.Duration `json:"metrics_interval"`
}

// DefaultCacheConfig returns a default cache configuration
func DefaultCacheConfig() *CacheConfig {
	return &CacheConfig{
		L1Enabled:    true,
		L1MaxSize:    10000,
		L1TTL:        5 * time.Minute,
		L1CleanupInt: 30 * time.Second,

		L2Enabled:       true,
		RedisAddrs:      []string{"localhost:6379"},
		RedisPassword:   "",
		RedisDB:         0,
		RedisCluster:    false,
		SentinelEnabled: false,
		PoolSize:        100,
		MinIdleConns:    10,
		MaxRetries:      3,
		RetryDelay:      100 * time.Millisecond,
		ReadTimeout:     3 * time.Second,
		WriteTimeout:    3 * time.Second,
		DialTimeout:     5 * time.Second,

		L3Enabled: false,
		L3Path:    "/var/cache/novacron",

		DefaultTTL:        15 * time.Minute,
		EnableCompression: true,
		CompressionLevel:  6,
		Serialization:     "json",
		EnableMetrics:     true,
		MetricsInterval:   10 * time.Second,
	}
}

// MultiTierCache implements a multi-tier caching system
type MultiTierCache struct {
	config  *CacheConfig
	l1Cache Cache // Memory cache
	l2Cache Cache // Redis cache
	l3Cache Cache // Persistent cache
	stats   CacheStats
	logger  *logrus.Logger
	mutex   sync.RWMutex
	
	// Internal state
	closed bool
	ctx    context.Context
	cancel context.CancelFunc
}

// NewMultiTierCache creates a new multi-tier cache
func NewMultiTierCache(config *CacheConfig, logger *logrus.Logger) (*MultiTierCache, error) {
	if config == nil {
		config = DefaultCacheConfig()
	}
	
	if logger == nil {
		logger = logrus.New()
	}

	ctx, cancel := context.WithCancel(context.Background())
	
	mtc := &MultiTierCache{
		config: config,
		stats:  CacheStats{LastUpdated: time.Now()},
		logger: logger,
		ctx:    ctx,
		cancel: cancel,
	}

	// Initialize L1 cache (memory)
	if config.L1Enabled {
		l1Cache, err := NewMemoryCache(&MemoryCacheConfig{
			MaxSize:       config.L1MaxSize,
			DefaultTTL:    config.L1TTL,
			CleanupInt:    config.L1CleanupInt,
			EnableMetrics: config.EnableMetrics,
		}, logger)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to initialize L1 cache: %w", err)
		}
		mtc.l1Cache = l1Cache
		logger.Info("L1 (Memory) cache initialized")
	}

	// Initialize L2 cache (Redis)
	if config.L2Enabled {
		l2Cache, err := NewRedisCache(config, logger)
		if err != nil {
			cancel()
			mtc.Close() // Clean up L1 if it was created
			return nil, fmt.Errorf("failed to initialize L2 cache: %w", err)
		}
		mtc.l2Cache = l2Cache
		logger.Info("L2 (Redis) cache initialized")
	}

	// Initialize L3 cache (Persistent)
	if config.L3Enabled {
		l3Cache, err := NewPersistentCache(&PersistentCacheConfig{
			StoragePath:   config.L3Path,
			EnableMetrics: config.EnableMetrics,
		}, logger)
		if err != nil {
			cancel()
			mtc.Close() // Clean up L1 and L2 if they were created
			return nil, fmt.Errorf("failed to initialize L3 cache: %w", err)
		}
		mtc.l3Cache = l3Cache
		logger.Info("L3 (Persistent) cache initialized")
	}

	// Start metrics collection if enabled
	if config.EnableMetrics {
		go mtc.collectMetrics()
	}

	logger.WithFields(logrus.Fields{
		"l1_enabled": config.L1Enabled,
		"l2_enabled": config.L2Enabled,
		"l3_enabled": config.L3Enabled,
	}).Info("Multi-tier cache initialized successfully")

	return mtc, nil
}

// Get retrieves a value from cache, checking L1 -> L2 -> L3 in order
func (mtc *MultiTierCache) Get(ctx context.Context, key string) ([]byte, error) {
	start := time.Now()
	defer func() {
		mtc.updateResponseTime(time.Since(start))
	}()

	// Try L1 cache first
	if mtc.l1Cache != nil {
		if value, err := mtc.l1Cache.Get(ctx, key); err == nil {
			mtc.mutex.Lock()
			mtc.stats.Hits++
			mtc.stats.L1Hits++
			mtc.mutex.Unlock()
			return value, nil
		}
	}

	// Try L2 cache
	if mtc.l2Cache != nil {
		if value, err := mtc.l2Cache.Get(ctx, key); err == nil {
			// Populate L1 cache if available
			if mtc.l1Cache != nil {
				go func() {
					_ = mtc.l1Cache.Set(context.Background(), key, value, mtc.config.L1TTL)
				}()
			}
			
			mtc.mutex.Lock()
			mtc.stats.Hits++
			mtc.stats.L2Hits++
			mtc.mutex.Unlock()
			return value, nil
		}
	}

	// Try L3 cache
	if mtc.l3Cache != nil {
		if value, err := mtc.l3Cache.Get(ctx, key); err == nil {
			// Populate L1 and L2 caches if available
			if mtc.l1Cache != nil {
				go func() {
					_ = mtc.l1Cache.Set(context.Background(), key, value, mtc.config.L1TTL)
				}()
			}
			if mtc.l2Cache != nil {
				go func() {
					_ = mtc.l2Cache.Set(context.Background(), key, value, mtc.config.DefaultTTL)
				}()
			}
			
			mtc.mutex.Lock()
			mtc.stats.Hits++
			mtc.stats.L3Hits++
			mtc.mutex.Unlock()
			return value, nil
		}
	}

	mtc.mutex.Lock()
	mtc.stats.Misses++
	mtc.mutex.Unlock()
	
	return nil, ErrCacheMiss
}

// Set stores a value in all available cache tiers
func (mtc *MultiTierCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	start := time.Now()
	defer func() {
		mtc.updateResponseTime(time.Since(start))
	}()

	if ttl == 0 {
		ttl = mtc.config.DefaultTTL
	}

	var errs []error

	// Set in L1 cache
	if mtc.l1Cache != nil {
		if err := mtc.l1Cache.Set(ctx, key, value, ttl); err != nil {
			errs = append(errs, fmt.Errorf("L1 set failed: %w", err))
		}
	}

	// Set in L2 cache
	if mtc.l2Cache != nil {
		if err := mtc.l2Cache.Set(ctx, key, value, ttl); err != nil {
			errs = append(errs, fmt.Errorf("L2 set failed: %w", err))
		}
	}

	// Set in L3 cache
	if mtc.l3Cache != nil {
		if err := mtc.l3Cache.Set(ctx, key, value, ttl); err != nil {
			errs = append(errs, fmt.Errorf("L3 set failed: %w", err))
		}
	}

	mtc.mutex.Lock()
	mtc.stats.Sets++
	if len(errs) > 0 {
		mtc.stats.Errors++
	}
	mtc.mutex.Unlock()

	if len(errs) > 0 {
		return fmt.Errorf("cache set errors: %v", errs)
	}

	return nil
}

// Delete removes a key from all cache tiers
func (mtc *MultiTierCache) Delete(ctx context.Context, key string) error {
	var errs []error

	// Delete from L1 cache
	if mtc.l1Cache != nil {
		if err := mtc.l1Cache.Delete(ctx, key); err != nil {
			errs = append(errs, fmt.Errorf("L1 delete failed: %w", err))
		}
	}

	// Delete from L2 cache
	if mtc.l2Cache != nil {
		if err := mtc.l2Cache.Delete(ctx, key); err != nil {
			errs = append(errs, fmt.Errorf("L2 delete failed: %w", err))
		}
	}

	// Delete from L3 cache
	if mtc.l3Cache != nil {
		if err := mtc.l3Cache.Delete(ctx, key); err != nil {
			errs = append(errs, fmt.Errorf("L3 delete failed: %w", err))
		}
	}

	mtc.mutex.Lock()
	mtc.stats.Deletes++
	if len(errs) > 0 {
		mtc.stats.Errors++
	}
	mtc.mutex.Unlock()

	if len(errs) > 0 {
		return fmt.Errorf("cache delete errors: %v", errs)
	}

	return nil
}

// Exists checks if a key exists in any cache tier
func (mtc *MultiTierCache) Exists(ctx context.Context, key string) (bool, error) {
	// Check L1 cache first
	if mtc.l1Cache != nil {
		if exists, err := mtc.l1Cache.Exists(ctx, key); err == nil && exists {
			return true, nil
		}
	}

	// Check L2 cache
	if mtc.l2Cache != nil {
		if exists, err := mtc.l2Cache.Exists(ctx, key); err == nil && exists {
			return true, nil
		}
	}

	// Check L3 cache
	if mtc.l3Cache != nil {
		if exists, err := mtc.l3Cache.Exists(ctx, key); err == nil && exists {
			return true, nil
		}
	}

	return false, nil
}

// Clear removes all keys from all cache tiers
func (mtc *MultiTierCache) Clear(ctx context.Context) error {
	var errs []error

	if mtc.l1Cache != nil {
		if err := mtc.l1Cache.Clear(ctx); err != nil {
			errs = append(errs, fmt.Errorf("L1 clear failed: %w", err))
		}
	}

	if mtc.l2Cache != nil {
		if err := mtc.l2Cache.Clear(ctx); err != nil {
			errs = append(errs, fmt.Errorf("L2 clear failed: %w", err))
		}
	}

	if mtc.l3Cache != nil {
		if err := mtc.l3Cache.Clear(ctx); err != nil {
			errs = append(errs, fmt.Errorf("L3 clear failed: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("cache clear errors: %v", errs)
	}

	return nil
}

// GetMulti retrieves multiple keys
func (mtc *MultiTierCache) GetMulti(ctx context.Context, keys []string) (map[string][]byte, error) {
	result := make(map[string][]byte)
	remaining := make([]string, 0, len(keys))

	// Try L1 cache first
	if mtc.l1Cache != nil {
		l1Results, _ := mtc.l1Cache.GetMulti(ctx, keys)
		for key, value := range l1Results {
			result[key] = value
		}
	}

	// Collect remaining keys
	for _, key := range keys {
		if _, found := result[key]; !found {
			remaining = append(remaining, key)
		}
	}

	// Try L2 cache for remaining keys
	if len(remaining) > 0 && mtc.l2Cache != nil {
		l2Results, _ := mtc.l2Cache.GetMulti(ctx, remaining)
		for key, value := range l2Results {
			result[key] = value
			// Populate L1 cache asynchronously
			if mtc.l1Cache != nil {
				go func(k string, v []byte) {
					_ = mtc.l1Cache.Set(context.Background(), k, v, mtc.config.L1TTL)
				}(key, value)
			}
		}
	}

	// Update remaining keys again
	remaining = remaining[:0]
	for _, key := range keys {
		if _, found := result[key]; !found {
			remaining = append(remaining, key)
		}
	}

	// Try L3 cache for remaining keys
	if len(remaining) > 0 && mtc.l3Cache != nil {
		l3Results, _ := mtc.l3Cache.GetMulti(ctx, remaining)
		for key, value := range l3Results {
			result[key] = value
			// Populate L1 and L2 caches asynchronously
			if mtc.l1Cache != nil {
				go func(k string, v []byte) {
					_ = mtc.l1Cache.Set(context.Background(), k, v, mtc.config.L1TTL)
				}(key, value)
			}
			if mtc.l2Cache != nil {
				go func(k string, v []byte) {
					_ = mtc.l2Cache.Set(context.Background(), k, v, mtc.config.DefaultTTL)
				}(key, value)
			}
		}
	}

	return result, nil
}

// SetMulti sets multiple key-value pairs
func (mtc *MultiTierCache) SetMulti(ctx context.Context, items map[string]CacheItem) error {
	var errs []error

	if mtc.l1Cache != nil {
		if err := mtc.l1Cache.SetMulti(ctx, items); err != nil {
			errs = append(errs, fmt.Errorf("L1 setMulti failed: %w", err))
		}
	}

	if mtc.l2Cache != nil {
		if err := mtc.l2Cache.SetMulti(ctx, items); err != nil {
			errs = append(errs, fmt.Errorf("L2 setMulti failed: %w", err))
		}
	}

	if mtc.l3Cache != nil {
		if err := mtc.l3Cache.SetMulti(ctx, items); err != nil {
			errs = append(errs, fmt.Errorf("L3 setMulti failed: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("cache setMulti errors: %v", errs)
	}

	return nil
}

// DeleteMulti removes multiple keys
func (mtc *MultiTierCache) DeleteMulti(ctx context.Context, keys []string) error {
	var errs []error

	if mtc.l1Cache != nil {
		if err := mtc.l1Cache.DeleteMulti(ctx, keys); err != nil {
			errs = append(errs, fmt.Errorf("L1 deleteMulti failed: %w", err))
		}
	}

	if mtc.l2Cache != nil {
		if err := mtc.l2Cache.DeleteMulti(ctx, keys); err != nil {
			errs = append(errs, fmt.Errorf("L2 deleteMulti failed: %w", err))
		}
	}

	if mtc.l3Cache != nil {
		if err := mtc.l3Cache.DeleteMulti(ctx, keys); err != nil {
			errs = append(errs, fmt.Errorf("L3 deleteMulti failed: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("cache deleteMulti errors: %v", errs)
	}

	return nil
}

// GetStats returns cache statistics
func (mtc *MultiTierCache) GetStats() CacheStats {
	mtc.mutex.RLock()
	defer mtc.mutex.RUnlock()
	
	stats := mtc.stats
	total := stats.Hits + stats.Misses
	if total > 0 {
		stats.HitRate = float64(stats.Hits) / float64(total)
	}
	stats.LastUpdated = time.Now()
	
	return stats
}

// Close closes all cache tiers and cleanup resources
func (mtc *MultiTierCache) Close() error {
	mtc.mutex.Lock()
	defer mtc.mutex.Unlock()

	if mtc.closed {
		return nil
	}

	mtc.cancel()
	mtc.closed = true

	var errs []error

	if mtc.l1Cache != nil {
		if err := mtc.l1Cache.Close(); err != nil {
			errs = append(errs, fmt.Errorf("L1 close failed: %w", err))
		}
	}

	if mtc.l2Cache != nil {
		if err := mtc.l2Cache.Close(); err != nil {
			errs = append(errs, fmt.Errorf("L2 close failed: %w", err))
		}
	}

	if mtc.l3Cache != nil {
		if err := mtc.l3Cache.Close(); err != nil {
			errs = append(errs, fmt.Errorf("L3 close failed: %w", err))
		}
	}

	if len(errs) > 0 {
		return fmt.Errorf("cache close errors: %v", errs)
	}

	mtc.logger.Info("Multi-tier cache closed successfully")
	return nil
}

// collectMetrics runs periodic metrics collection
func (mtc *MultiTierCache) collectMetrics() {
	ticker := time.NewTicker(mtc.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-mtc.ctx.Done():
			return
		case <-ticker.C:
			mtc.updateStats()
		}
	}
}

// updateStats aggregates stats from all cache tiers
func (mtc *MultiTierCache) updateStats() {
	mtc.mutex.Lock()
	defer mtc.mutex.Unlock()

	// Aggregate stats from individual caches if they support it
	if mtc.l1Cache != nil {
		l1Stats := mtc.l1Cache.GetStats()
		// Update relevant stats
		mtc.stats.L1Hits = l1Stats.Hits
	}

	if mtc.l2Cache != nil {
		l2Stats := mtc.l2Cache.GetStats()
		mtc.stats.L2Hits = l2Stats.Hits
	}

	if mtc.l3Cache != nil {
		l3Stats := mtc.l3Cache.GetStats()
		mtc.stats.L3Hits = l3Stats.Hits
	}

	mtc.stats.LastUpdated = time.Now()
}

// updateResponseTime updates average response time
func (mtc *MultiTierCache) updateResponseTime(duration time.Duration) {
	mtc.mutex.Lock()
	defer mtc.mutex.Unlock()

	// Simple moving average calculation
	if mtc.stats.AvgResponseTimeNs == 0 {
		mtc.stats.AvgResponseTimeNs = duration.Nanoseconds()
	} else {
		mtc.stats.AvgResponseTimeNs = (mtc.stats.AvgResponseTimeNs + duration.Nanoseconds()) / 2
	}
}