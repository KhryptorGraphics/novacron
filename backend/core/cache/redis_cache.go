package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/sirupsen/logrus"
)

// RedisCache implements Redis-based caching with cluster and sentinel support
type RedisCache struct {
	config      *CacheConfig
	client      redis.Cmdable
	logger      *logrus.Logger
	stats       CacheStats
	clusterMode bool
}

// NewRedisCache creates a new Redis cache instance
func NewRedisCache(config *CacheConfig, logger *logrus.Logger) (*RedisCache, error) {
	if logger == nil {
		logger = logrus.New()
	}

	rc := &RedisCache{
		config: config,
		logger: logger,
		stats:  CacheStats{LastUpdated: time.Now()},
	}

	var client redis.Cmdable
	var err error

	// Initialize Redis client based on configuration
	if config.SentinelEnabled {
		client, err = rc.initSentinelClient()
	} else if config.RedisCluster {
		client, err = rc.initClusterClient()
		rc.clusterMode = true
	} else {
		client, err = rc.initStandaloneClient()
	}

	if err != nil {
		return nil, fmt.Errorf("failed to initialize Redis client: %w", err)
	}

	rc.client = client

	// Test connection
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := rc.client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to ping Redis: %w", err)
	}

	logger.WithFields(logrus.Fields{
		"cluster":  config.RedisCluster,
		"sentinel": config.SentinelEnabled,
		"addrs":    config.RedisAddrs,
	}).Info("Redis cache initialized successfully")

	return rc, nil
}

// initStandaloneClient creates a standalone Redis client
func (rc *RedisCache) initStandaloneClient() (*redis.Client, error) {
	if len(rc.config.RedisAddrs) == 0 {
		return nil, fmt.Errorf("no Redis addresses configured")
	}

	opts := &redis.Options{
		Addr:         rc.config.RedisAddrs[0],
		Password:     rc.config.RedisPassword,
		DB:           rc.config.RedisDB,
		PoolSize:     rc.config.PoolSize,
		MinIdleConns: rc.config.MinIdleConns,
		MaxRetries:   rc.config.MaxRetries,
		// RetryDelay removed in newer Redis client versions
		ReadTimeout:  rc.config.ReadTimeout,
		WriteTimeout: rc.config.WriteTimeout,
		DialTimeout:  rc.config.DialTimeout,
	}

	return redis.NewClient(opts), nil
}

// initClusterClient creates a Redis cluster client
func (rc *RedisCache) initClusterClient() (*redis.ClusterClient, error) {
	if len(rc.config.RedisAddrs) == 0 {
		return nil, fmt.Errorf("no Redis cluster addresses configured")
	}

	opts := &redis.ClusterOptions{
		Addrs:        rc.config.RedisAddrs,
		Password:     rc.config.RedisPassword,
		PoolSize:     rc.config.PoolSize,
		MinIdleConns: rc.config.MinIdleConns,
		MaxRetries:   rc.config.MaxRetries,
		// RetryDelay removed in newer Redis client versions
		ReadTimeout:  rc.config.ReadTimeout,
		WriteTimeout: rc.config.WriteTimeout,
		DialTimeout:  rc.config.DialTimeout,
	}

	return redis.NewClusterClient(opts), nil
}

// initSentinelClient creates a Redis sentinel client
func (rc *RedisCache) initSentinelClient() (*redis.Client, error) {
	if len(rc.config.SentinelAddrs) == 0 {
		return nil, fmt.Errorf("no Redis sentinel addresses configured")
	}

	if rc.config.SentinelMaster == "" {
		return nil, fmt.Errorf("no Redis sentinel master name configured")
	}

	opts := &redis.FailoverOptions{
		MasterName:    rc.config.SentinelMaster,
		SentinelAddrs: rc.config.SentinelAddrs,
		Password:      rc.config.RedisPassword,
		DB:            rc.config.RedisDB,
		PoolSize:      rc.config.PoolSize,
		MinIdleConns:  rc.config.MinIdleConns,
		MaxRetries:    rc.config.MaxRetries,
		// RetryDelay removed in newer Redis client versions
		ReadTimeout:   rc.config.ReadTimeout,
		WriteTimeout:  rc.config.WriteTimeout,
		DialTimeout:   rc.config.DialTimeout,
	}

	return redis.NewFailoverClient(opts), nil
}

// Get retrieves a value from Redis cache
func (rc *RedisCache) Get(ctx context.Context, key string) ([]byte, error) {
	start := time.Now()
	defer func() {
		rc.updateResponseTime(time.Since(start))
	}()

	result, err := rc.client.Get(ctx, key).Bytes()
	if err != nil {
		if err == redis.Nil {
			rc.stats.Misses++
			return nil, ErrCacheMiss
		}
		rc.stats.Errors++
		return nil, fmt.Errorf("Redis GET failed: %w", err)
	}

	rc.stats.Hits++
	return result, nil
}

// Set stores a value in Redis cache
func (rc *RedisCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	start := time.Now()
	defer func() {
		rc.updateResponseTime(time.Since(start))
	}()

	if ttl == 0 {
		ttl = rc.config.DefaultTTL
	}

	err := rc.client.Set(ctx, key, value, ttl).Err()
	if err != nil {
		rc.stats.Errors++
		return fmt.Errorf("Redis SET failed: %w", err)
	}

	rc.stats.Sets++
	return nil
}

// Delete removes a key from Redis cache
func (rc *RedisCache) Delete(ctx context.Context, key string) error {
	err := rc.client.Del(ctx, key).Err()
	if err != nil {
		rc.stats.Errors++
		return fmt.Errorf("Redis DEL failed: %w", err)
	}

	rc.stats.Deletes++
	return nil
}

// Exists checks if a key exists in Redis cache
func (rc *RedisCache) Exists(ctx context.Context, key string) (bool, error) {
	result, err := rc.client.Exists(ctx, key).Result()
	if err != nil {
		rc.stats.Errors++
		return false, fmt.Errorf("Redis EXISTS failed: %w", err)
	}

	return result > 0, nil
}

// Clear removes all keys from Redis cache
func (rc *RedisCache) Clear(ctx context.Context) error {
	var err error

	if rc.clusterMode {
		// For cluster mode, we need to clear each node
		if clusterClient, ok := rc.client.(*redis.ClusterClient); ok {
			err = clusterClient.ForEachMaster(ctx, func(ctx context.Context, client *redis.Client) error {
				return client.FlushDB(ctx).Err()
			})
		}
	} else {
		err = rc.client.FlushDB(ctx).Err()
	}

	if err != nil {
		rc.stats.Errors++
		return fmt.Errorf("Redis FLUSHDB failed: %w", err)
	}

	return nil
}

// GetMulti retrieves multiple keys from Redis cache
func (rc *RedisCache) GetMulti(ctx context.Context, keys []string) (map[string][]byte, error) {
	if len(keys) == 0 {
		return make(map[string][]byte), nil
	}

	start := time.Now()
	defer func() {
		rc.updateResponseTime(time.Since(start))
	}()

	// Use pipeline for efficient multi-get
	pipe := rc.client.Pipeline()
	cmds := make([]*redis.StringCmd, len(keys))

	for i, key := range keys {
		cmds[i] = pipe.Get(ctx, key)
	}

	_, err := pipe.Exec(ctx)
	if err != nil && err != redis.Nil {
		rc.stats.Errors++
		return nil, fmt.Errorf("Redis pipeline execution failed: %w", err)
	}

	result := make(map[string][]byte)
	for i, cmd := range cmds {
		if cmd.Err() == nil {
			result[keys[i]], _ = cmd.Bytes()
			rc.stats.Hits++
		} else if cmd.Err() == redis.Nil {
			rc.stats.Misses++
		} else {
			rc.stats.Errors++
		}
	}

	return result, nil
}

// SetMulti sets multiple key-value pairs in Redis cache
func (rc *RedisCache) SetMulti(ctx context.Context, items map[string]CacheItem) error {
	if len(items) == 0 {
		return nil
	}

	start := time.Now()
	defer func() {
		rc.updateResponseTime(time.Since(start))
	}()

	// Use pipeline for efficient multi-set
	pipe := rc.client.Pipeline()

	for key, item := range items {
		ttl := item.TTL
		if ttl == 0 {
			ttl = rc.config.DefaultTTL
		}
		pipe.Set(ctx, key, item.Value, ttl)
	}

	_, err := pipe.Exec(ctx)
	if err != nil {
		rc.stats.Errors++
		return fmt.Errorf("Redis pipeline execution failed: %w", err)
	}

	rc.stats.Sets += uint64(len(items))
	return nil
}

// DeleteMulti removes multiple keys from Redis cache
func (rc *RedisCache) DeleteMulti(ctx context.Context, keys []string) error {
	if len(keys) == 0 {
		return nil
	}

	err := rc.client.Del(ctx, keys...).Err()
	if err != nil {
		rc.stats.Errors++
		return fmt.Errorf("Redis DEL failed: %w", err)
	}

	rc.stats.Deletes += uint64(len(keys))
	return nil
}

// GetStats returns Redis cache statistics
func (rc *RedisCache) GetStats() CacheStats {
	stats := rc.stats
	total := stats.Hits + stats.Misses
	if total > 0 {
		stats.HitRate = float64(stats.Hits) / float64(total)
	}
	stats.LastUpdated = time.Now()

	return stats
}

// Close closes the Redis client
func (rc *RedisCache) Close() error {
	var err error

	switch client := rc.client.(type) {
	case *redis.Client:
		err = client.Close()
	case *redis.ClusterClient:
		err = client.Close()
	default:
		rc.logger.Warn("Unknown Redis client type, cannot close properly")
	}

	if err != nil {
		return fmt.Errorf("failed to close Redis client: %w", err)
	}

	rc.logger.Info("Redis cache closed")
	return nil
}

// GetInfo returns Redis server information
func (rc *RedisCache) GetInfo(ctx context.Context) (map[string]string, error) {
	info, err := rc.client.Info(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("Redis INFO failed: %w", err)
	}

	// Parse info string into map
	result := make(map[string]string)
	lines := splitLines(info)
	for _, line := range lines {
		if parts := splitKeyValue(line, ":"); len(parts) == 2 {
			result[parts[0]] = parts[1]
		}
	}

	return result, nil
}

// GetClusterInfo returns Redis cluster information (cluster mode only)
func (rc *RedisCache) GetClusterInfo(ctx context.Context) (map[string]interface{}, error) {
	if !rc.clusterMode {
		return nil, fmt.Errorf("not in cluster mode")
	}

	clusterClient, ok := rc.client.(*redis.ClusterClient)
	if !ok {
		return nil, fmt.Errorf("invalid cluster client")
	}

	result := make(map[string]interface{})

	// Get cluster nodes
	nodes, err := clusterClient.ClusterNodes(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get cluster nodes: %w", err)
	}
	result["nodes"] = nodes

	// Get cluster info
	info, err := clusterClient.ClusterInfo(ctx).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get cluster info: %w", err)
	}
	result["info"] = info

	return result, nil
}

// Ping tests Redis connection
func (rc *RedisCache) Ping(ctx context.Context) error {
	return rc.client.Ping(ctx).Err()
}

// updateResponseTime updates average response time
func (rc *RedisCache) updateResponseTime(duration time.Duration) {
	// Simple moving average calculation
	if rc.stats.AvgResponseTimeNs == 0 {
		rc.stats.AvgResponseTimeNs = duration.Nanoseconds()
	} else {
		rc.stats.AvgResponseTimeNs = (rc.stats.AvgResponseTimeNs + duration.Nanoseconds()) / 2
	}
}

// Helper functions

func splitLines(s string) []string {
	var lines []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			if i > start {
				line := s[start:i]
				if len(line) > 0 && line[len(line)-1] == '\r' {
					line = line[:len(line)-1]
				}
				if line != "" && line[0] != '#' {
					lines = append(lines, line)
				}
			}
			start = i + 1
		}
	}
	if start < len(s) {
		line := s[start:]
		if line != "" && line[0] != '#' {
			lines = append(lines, line)
		}
	}
	return lines
}

func splitKeyValue(s, sep string) []string {
	idx := -1
	for i := 0; i <= len(s)-len(sep); i++ {
		if s[i:i+len(sep)] == sep {
			idx = i
			break
		}
	}
	if idx == -1 {
		return []string{s}
	}
	return []string{s[:idx], s[idx+len(sep):]}
}