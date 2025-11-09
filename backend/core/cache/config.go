package cache

import (
	"time"
)

// CacheConfig defines the configuration for the intelligent caching system
type CacheConfig struct {
	// Tier sizes
	L1Size int64 // Edge cache (hot data) - Default: 10GB
	L2Size int64 // Regional cache (warm data) - Default: 100GB
	L3Size int64 // Global cache (cold data) - Default: 1TB

	// Eviction policies
	EvictionPolicy string // "ml", "lru", "lfu", "arc"

	// Prefetching
	EnablePrefetch      bool
	PrefetchWindow      int     // Number of items to prefetch
	PrefetchAggression  float64 // 0.0-1.0, how aggressive prefetching is
	MinPrefetchAccuracy float64 // Minimum accuracy before enabling

	// Deduplication
	EnableDedup bool
	ChunkSize   int    // 4KB, 64KB, 1MB
	HashAlgo    string // "sha256", "blake3"

	// Compression
	EnableCompression bool
	CompressionAlgo   string  // "zstd", "lz4", "snappy"
	CompressionLevel  int     // Compression level (1-9)
	MinCompressionRatio float64 // Only cache if compression ratio exceeds this

	// Consistency
	ConsistencyMode string        // "strong", "eventual"
	WriteMode       string        // "write-through", "write-back"
	FlushInterval   time.Duration // For write-back mode
	DefaultTTL      time.Duration

	// ML Configuration
	MLModelPath       string
	MLUpdateInterval  time.Duration
	MLFeatureCount    int
	MLLearningRate    float64
	EnableOnline      bool // Enable online learning

	// Warming
	EnableWarming     bool
	WarmingSchedule   string // Cron expression
	WarmingPatterns   []string
	WarmingConcurrency int

	// Monitoring
	MetricsInterval   time.Duration
	EnableDetailedMetrics bool

	// Performance
	MaxConcurrentOps  int
	ReadTimeout       time.Duration
	WriteTimeout      time.Duration
}

// DefaultConfig returns a production-ready default configuration
func DefaultConfig() *CacheConfig {
	return &CacheConfig{
		// Tier sizes
		L1Size: 10 * 1024 * 1024 * 1024,   // 10GB
		L2Size: 100 * 1024 * 1024 * 1024,  // 100GB
		L3Size: 1024 * 1024 * 1024 * 1024, // 1TB

		// Eviction
		EvictionPolicy: "ml",

		// Prefetching
		EnablePrefetch:      true,
		PrefetchWindow:      10,
		PrefetchAggression:  0.7,
		MinPrefetchAccuracy: 0.85,

		// Deduplication
		EnableDedup: true,
		ChunkSize:   64 * 1024, // 64KB
		HashAlgo:    "sha256",

		// Compression
		EnableCompression:   true,
		CompressionAlgo:     "zstd",
		CompressionLevel:    3,
		MinCompressionRatio: 1.2,

		// Consistency
		ConsistencyMode: "eventual",
		WriteMode:       "write-back",
		FlushInterval:   30 * time.Second,
		DefaultTTL:      24 * time.Hour,

		// ML Configuration
		MLModelPath:      "/var/lib/novacron/cache/ml_model.bin",
		MLUpdateInterval: 1 * time.Hour,
		MLFeatureCount:   8,
		MLLearningRate:   0.01,
		EnableOnline:     true,

		// Warming
		EnableWarming:      true,
		WarmingSchedule:    "0 6 * * *", // Daily at 6 AM
		WarmingPatterns:    []string{"vm-templates/*", "popular-images/*"},
		WarmingConcurrency: 4,

		// Monitoring
		MetricsInterval:       10 * time.Second,
		EnableDetailedMetrics: true,

		// Performance
		MaxConcurrentOps: 1000,
		ReadTimeout:      5 * time.Second,
		WriteTimeout:     10 * time.Second,
	}
}

// EdgeConfig returns configuration optimized for edge caching
func EdgeConfig() *CacheConfig {
	cfg := DefaultConfig()
	cfg.L1Size = 5 * 1024 * 1024 * 1024 // 5GB
	cfg.L2Size = 0                       // No L2 at edge
	cfg.L3Size = 0                       // No L3 at edge
	cfg.EnableCompression = true
	cfg.CompressionAlgo = "lz4" // Faster for edge
	cfg.PrefetchAggression = 0.9 // More aggressive at edge
	return cfg
}

// RegionalConfig returns configuration optimized for regional caching
func RegionalConfig() *CacheConfig {
	cfg := DefaultConfig()
	cfg.L1Size = 20 * 1024 * 1024 * 1024  // 20GB
	cfg.L2Size = 200 * 1024 * 1024 * 1024 // 200GB
	cfg.L3Size = 0                         // No L3 at regional
	cfg.PrefetchAggression = 0.6
	return cfg
}

// GlobalConfig returns configuration optimized for global caching
func GlobalConfig() *CacheConfig {
	cfg := DefaultConfig()
	cfg.L1Size = 50 * 1024 * 1024 * 1024   // 50GB
	cfg.L2Size = 500 * 1024 * 1024 * 1024  // 500GB
	cfg.L3Size = 5 * 1024 * 1024 * 1024 * 1024 // 5TB
	cfg.EnableDedup = true
	cfg.PrefetchAggression = 0.5 // Less aggressive at global
	return cfg
}

// Validate ensures configuration is valid
func (c *CacheConfig) Validate() error {
	if c.L1Size <= 0 {
		return ErrInvalidConfig("L1Size must be positive")
	}
	if c.EvictionPolicy == "" {
		c.EvictionPolicy = "ml"
	}
	if c.ChunkSize <= 0 {
		c.ChunkSize = 64 * 1024
	}
	if c.PrefetchWindow <= 0 {
		c.PrefetchWindow = 10
	}
	if c.PrefetchAggression < 0 || c.PrefetchAggression > 1 {
		c.PrefetchAggression = 0.7
	}
	if c.MinPrefetchAccuracy < 0 || c.MinPrefetchAccuracy > 1 {
		c.MinPrefetchAccuracy = 0.85
	}
	if c.MLFeatureCount <= 0 {
		c.MLFeatureCount = 8
	}
	if c.MLLearningRate <= 0 {
		c.MLLearningRate = 0.01
	}
	return nil
}
