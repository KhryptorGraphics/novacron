package performance

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// CacheConfig represents Redis cache configuration
type CacheConfig struct {
	Host            string        `yaml:"host" json:"host"`
	Port            int           `yaml:"port" json:"port"`
	Password        string        `yaml:"password" json:"password"`
	Database        int           `yaml:"database" json:"database"`
	PoolSize        int           `yaml:"poolSize" json:"poolSize"`
	MinIdleConns    int           `yaml:"minIdleConns" json:"minIdleConns"`
	MaxRetries      int           `yaml:"maxRetries" json:"maxRetries"`
	DialTimeout     time.Duration `yaml:"dialTimeout" json:"dialTimeout"`
	ReadTimeout     time.Duration `yaml:"readTimeout" json:"readTimeout"`
	WriteTimeout    time.Duration `yaml:"writeTimeout" json:"writeTimeout"`
	PoolTimeout     time.Duration `yaml:"poolTimeout" json:"poolTimeout"`
	IdleTimeout     time.Duration `yaml:"idleTimeout" json:"idleTimeout"`
	DefaultTTL      time.Duration `yaml:"defaultTTL" json:"defaultTTL"`
	EnableMetrics   bool          `yaml:"enableMetrics" json:"enableMetrics"`
}

// DefaultCacheConfig returns optimized default cache configuration
func DefaultCacheConfig() CacheConfig {
	return CacheConfig{
		Host:            "localhost",
		Port:            6379,
		Database:        0,
		PoolSize:        20,         // Optimized pool size
		MinIdleConns:    5,          // Minimum idle connections
		MaxRetries:      3,          // Retry failed operations
		DialTimeout:     5 * time.Second,
		ReadTimeout:     3 * time.Second,
		WriteTimeout:    3 * time.Second,
		PoolTimeout:     4 * time.Second,
		IdleTimeout:     5 * time.Minute,
		DefaultTTL:      1 * time.Hour, // 1 hour default TTL
		EnableMetrics:   true,
	}
}

// CacheClient represents an optimized Redis cache client
type CacheClient struct {
	client  *redis.Client
	config  CacheConfig
	metrics *CacheMetrics
	logger  logger.Logger
}

// CacheMetrics tracks cache performance metrics
type CacheMetrics struct {
	Hits           int64         `json:"hits"`
	Misses         int64         `json:"misses"`
	Sets           int64         `json:"sets"`
	Deletes        int64         `json:"deletes"`
	Errors         int64         `json:"errors"`
	HitRate        float64       `json:"hitRate"`
	AverageLatency time.Duration `json:"averageLatency"`
	LastUpdate     time.Time     `json:"lastUpdate"`
}

// NewCacheClient creates a new optimized cache client
func NewCacheClient(config CacheConfig, logger logger.Logger) (*CacheClient, error) {
	rdb := redis.NewClient(&redis.Options{
		Addr:            fmt.Sprintf("%s:%d", config.Host, config.Port),
		Password:        config.Password,
		DB:              config.Database,
		PoolSize:        config.PoolSize,
		MinIdleConns:    config.MinIdleConns,
		MaxRetries:      config.MaxRetries,
		DialTimeout:     config.DialTimeout,
		ReadTimeout:     config.ReadTimeout,
		WriteTimeout:    config.WriteTimeout,
		PoolTimeout:     config.PoolTimeout,
		IdleTimeout:     config.IdleTimeout,
	})

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	client := &CacheClient{
		client:  rdb,
		config:  config,
		metrics: &CacheMetrics{},
		logger:  logger,
	}

	logger.Info("Cache client initialized",
		"host", config.Host,
		"port", config.Port,
		"poolSize", config.PoolSize,
		"defaultTTL", config.DefaultTTL,
	)

	return client, nil
}

// Get retrieves a value from cache with performance tracking
func (c *CacheClient) Get(ctx context.Context, key string, dest interface{}) error {
	start := time.Now()
	defer c.trackOperation(time.Since(start))

	val, err := c.client.Get(ctx, key).Result()
	if err != nil {
		if err == redis.Nil {
			c.metrics.Misses++
			return ErrCacheMiss
		}
		c.metrics.Errors++
		return fmt.Errorf("cache get error: %w", err)
	}

	c.metrics.Hits++
	
	if err := json.Unmarshal([]byte(val), dest); err != nil {
		c.metrics.Errors++
		return fmt.Errorf("cache unmarshal error: %w", err)
	}

	return nil
}

// Set stores a value in cache with performance tracking
func (c *CacheClient) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	start := time.Now()
	defer c.trackOperation(time.Since(start))

	if ttl == 0 {
		ttl = c.config.DefaultTTL
	}

	data, err := json.Marshal(value)
	if err != nil {
		c.metrics.Errors++
		return fmt.Errorf("cache marshal error: %w", err)
	}

	if err := c.client.Set(ctx, key, data, ttl).Err(); err != nil {
		c.metrics.Errors++
		return fmt.Errorf("cache set error: %w", err)
	}

	c.metrics.Sets++
	return nil
}

// Delete removes a value from cache
func (c *CacheClient) Delete(ctx context.Context, key string) error {
	start := time.Now()
	defer c.trackOperation(time.Since(start))

	if err := c.client.Del(ctx, key).Err(); err != nil {
		c.metrics.Errors++
		return fmt.Errorf("cache delete error: %w", err)
	}

	c.metrics.Deletes++
	return nil
}

// Exists checks if a key exists in cache
func (c *CacheClient) Exists(ctx context.Context, key string) (bool, error) {
	start := time.Now()
	defer c.trackOperation(time.Since(start))

	count, err := c.client.Exists(ctx, key).Result()
	if err != nil {
		c.metrics.Errors++
		return false, fmt.Errorf("cache exists error: %w", err)
	}

	return count > 0, nil
}

// GetOrSet gets a value from cache or sets it if not found
func (c *CacheClient) GetOrSet(ctx context.Context, key string, dest interface{}, setter func() (interface{}, error), ttl time.Duration) error {
	// Try to get from cache first
	err := c.Get(ctx, key, dest)
	if err == nil {
		return nil // Found in cache
	}
	
	if err != ErrCacheMiss {
		return err // Actual error
	}

	// Not found in cache, call setter function
	value, err := setter()
	if err != nil {
		return fmt.Errorf("setter function error: %w", err)
	}

	// Set in cache for future requests
	if err := c.Set(ctx, key, value, ttl); err != nil {
		c.logger.Warn("Failed to set cache", "key", key, "error", err)
	}

	// Copy the value to destination
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("marshal error: %w", err)
	}

	return json.Unmarshal(data, dest)
}

// Clear clears all cache entries (use with caution)
func (c *CacheClient) Clear(ctx context.Context) error {
	start := time.Now()
	defer c.trackOperation(time.Since(start))

	if err := c.client.FlushDB(ctx).Err(); err != nil {
		c.metrics.Errors++
		return fmt.Errorf("cache clear error: %w", err)
	}

	return nil
}

// trackOperation tracks cache operation metrics
func (c *CacheClient) trackOperation(duration time.Duration) {
	// Update average latency (simple moving average)
	totalOps := c.metrics.Hits + c.metrics.Misses + c.metrics.Sets + c.metrics.Deletes
	if totalOps == 1 {
		c.metrics.AverageLatency = duration
	} else {
		c.metrics.AverageLatency = time.Duration(
			(int64(c.metrics.AverageLatency) + int64(duration)) / 2,
		)
	}

	// Calculate hit rate
	if c.metrics.Hits+c.metrics.Misses > 0 {
		c.metrics.HitRate = float64(c.metrics.Hits) / float64(c.metrics.Hits+c.metrics.Misses)
	}

	c.metrics.LastUpdate = time.Now()
}

// GetMetrics returns current cache performance metrics
func (c *CacheClient) GetMetrics() CacheMetrics {
	return *c.metrics
}

// HealthCheck performs a cache health check
func (c *CacheClient) HealthCheck(ctx context.Context) error {
	return c.client.Ping(ctx).Err()
}

// GetStats returns detailed Redis statistics
func (c *CacheClient) GetStats(ctx context.Context) (map[string]string, error) {
	return c.client.Info(ctx).Result()
}

// Close closes the cache client gracefully
func (c *CacheClient) Close() error {
	c.logger.Info("Closing cache client")
	return c.client.Close()
}

// Errors
var (
	ErrCacheMiss = fmt.Errorf("cache miss")
)

// CacheKey generates a consistent cache key
func CacheKey(prefix string, parts ...string) string {
	key := prefix
	for _, part := range parts {
		key += ":" + part
	}
	return key
}