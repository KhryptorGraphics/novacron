package cache

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// MockManager implements Manager for testing
type MockManager struct {
	clusters map[string]Cluster
	mutex    sync.RWMutex
}

// NewMockManager creates a new mock cache manager
func NewMockManager() Manager {
	return &MockManager{
		clusters: make(map[string]Cluster),
	}
}

// GetOrCreateCluster gets or creates a mock cache cluster
func (m *MockManager) GetOrCreateCluster(clusterID string, config *ClusterConfig) (Cluster, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if cluster, exists := m.clusters[clusterID]; exists {
		return cluster, nil
	}

	cluster := &MockCluster{
		id:       clusterID,
		config:   config,
		data:     make(map[string]*CacheEntry),
		stats:    &CacheStats{},
		strategy: &Strategy{Type: "cache-aside"},
	}

	m.clusters[clusterID] = cluster
	return cluster, nil
}

// GetCluster gets an existing cluster
func (m *MockManager) GetCluster(clusterID string) (Cluster, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	cluster, exists := m.clusters[clusterID]
	if !exists {
		return nil, fmt.Errorf("cluster %s not found", clusterID)
	}

	return cluster, nil
}

// DisconnectCluster disconnects from a cluster
func (m *MockManager) DisconnectCluster(clusterID string) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	if cluster, exists := m.clusters[clusterID]; exists {
		if err := cluster.Close(); err != nil {
			return err
		}
		delete(m.clusters, clusterID)
	}

	return nil
}

// ListClusters lists all clusters
func (m *MockManager) ListClusters() []string {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var clusters []string
	for id := range m.clusters {
		clusters = append(clusters, id)
	}
	return clusters
}

// MockCluster implements Cluster for testing
type MockCluster struct {
	id        string
	config    *ClusterConfig
	data      map[string]*CacheEntry
	stats     *CacheStats
	strategy  *Strategy
	ttlPolicies []TTLPolicy
	mutex     sync.RWMutex
}

// Get retrieves a value from the mock cache
func (c *MockCluster) Get(key string) (string, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, exists := c.data[key]
	if !exists {
		c.stats.Misses++
		return "", fmt.Errorf("key not found: %s", key)
	}

	// Check expiration
	if entry.Expiration > 0 && time.Now().Unix() > entry.Expiration {
		c.mutex.RUnlock()
		c.mutex.Lock()
		delete(c.data, key)
		c.stats.ExpiredKeys++
		c.mutex.Unlock()
		c.mutex.RLock()
		c.stats.Misses++
		return "", fmt.Errorf("key expired: %s", key)
	}

	c.stats.Hits++
	return entry.Value, nil
}

// Set stores a value in the mock cache
func (c *MockCluster) Set(key string, value string, ttl string) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	entry := &CacheEntry{
		Key:   key,
		Value: value,
	}

	// Parse TTL
	if ttl != "" {
		ttlSeconds, err := c.parseTTL(ttl)
		if err != nil {
			return fmt.Errorf("invalid TTL: %w", err)
		}
		entry.TTL = ttlSeconds
		if ttlSeconds > 0 {
			entry.Expiration = time.Now().Unix() + ttlSeconds
		}
	}

	c.data[key] = entry
	c.stats.TotalKeys = int64(len(c.data))
	return nil
}

// Delete removes a key from the mock cache
func (c *MockCluster) Delete(key string) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	if _, exists := c.data[key]; exists {
		delete(c.data, key)
		c.stats.TotalKeys = int64(len(c.data))
		return nil
	}

	return fmt.Errorf("key not found: %s", key)
}

// Exists checks if a key exists
func (c *MockCluster) Exists(key string) (bool, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	entry, exists := c.data[key]
	if !exists {
		return false, nil
	}

	// Check expiration
	if entry.Expiration > 0 && time.Now().Unix() > entry.Expiration {
		return false, nil
	}

	return true, nil
}

// MGet retrieves multiple values
func (c *MockCluster) MGet(keys []string) (map[string]string, error) {
	result := make(map[string]string)
	
	for _, key := range keys {
		value, err := c.Get(key)
		if err == nil {
			result[key] = value
		}
	}

	return result, nil
}

// MSet stores multiple values
func (c *MockCluster) MSet(values map[string]string, ttl string) error {
	for key, value := range values {
		if err := c.Set(key, value, ttl); err != nil {
			return err
		}
	}
	return nil
}

// MDelete deletes multiple keys
func (c *MockCluster) MDelete(keys []string) error {
	for _, key := range keys {
		_ = c.Delete(key) // Ignore errors for non-existent keys
	}
	return nil
}

// ConfigureStrategy configures caching strategy
func (c *MockCluster) ConfigureStrategy(strategy *Strategy) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	c.strategy = strategy
	return nil
}

// SetTTLPolicy sets TTL policy
func (c *MockCluster) SetTTLPolicy(policy *TTLPolicy) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	// Add or update policy
	for i, existing := range c.ttlPolicies {
		if existing.Pattern == policy.Pattern {
			c.ttlPolicies[i] = *policy
			return nil
		}
	}
	
	c.ttlPolicies = append(c.ttlPolicies, *policy)
	return nil
}

// ConfigureEviction configures eviction policy
func (c *MockCluster) ConfigureEviction(policy *EvictionPolicy) error {
	// Mock implementation - just store the policy
	// In real implementation, this would configure Redis eviction
	return nil
}

// WarmCache performs cache warming
func (c *MockCluster) WarmCache(strategy *WarmingStrategy) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	keysWarmed := int64(0)
	
	switch strategy.Type {
	case "preload":
		// Preload specified data
		for key, value := range strategy.Data {
			c.data[key] = &CacheEntry{
				Key:   key,
				Value: value,
			}
			keysWarmed++
		}
	case "pattern":
		// Generate data based on patterns
		for _, pattern := range strategy.Patterns {
			// Simple pattern implementation - generate sample data
			for i := 0; i < 10; i++ {
				key := fmt.Sprintf("%s-%d", pattern, i)
				value := fmt.Sprintf("warmed-value-%d", i)
				c.data[key] = &CacheEntry{
					Key:   key,
					Value: value,
				}
				keysWarmed++
			}
		}
	}

	c.stats.TotalKeys = int64(len(c.data))
	return nil
}

// GetHealth returns cluster health
func (c *MockCluster) GetHealth() (*Health, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	// Generate mock health status
	status := "healthy"
	if rand.Float32() < 0.1 { // 10% chance of degraded status
		status = "degraded"
	}

	nodes := []NodeStatus{}
	for i := range c.config.Endpoints {
		nodeStatus := "online"
		if rand.Float32() < 0.05 { // 5% chance of offline node
			nodeStatus = "offline"
		}

		nodes = append(nodes, NodeStatus{
			NodeID:   fmt.Sprintf("node-%d", i),
			Status:   nodeStatus,
			Role:     "master",
			Memory:   fmt.Sprintf("%.1fMB", 100+rand.Float64()*900),
			LastSeen: &metav1.Time{Time: time.Now()},
		})
	}

	return &Health{
		Status:      status,
		Nodes:       nodes,
		Replication: "ok",
	}, nil
}

// GetPerformanceMetrics returns performance metrics
func (c *MockCluster) GetPerformanceMetrics() (*PerformanceMetrics, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	totalOps := c.stats.Hits + c.stats.Misses
	hitRate := 0.0
	if totalOps > 0 {
		hitRate = float64(c.stats.Hits) / float64(totalOps)
	}

	// Add some realistic variance
	hitRate = hitRate*0.9 + rand.Float64()*0.1 // 90% actual + 10% random

	return &PerformanceMetrics{
		HitRate:      hitRate,
		MissRate:     1.0 - hitRate,
		ResponseTime: fmt.Sprintf("%.2fms", 0.5+rand.Float64()*2.0), // 0.5-2.5ms
		Throughput:   1000 + rand.Float64()*9000, // 1000-10000 ops/sec
		Operations:   totalOps,
		Errors:       int64(rand.Intn(10)), // Random small number of errors
	}, nil
}

// GetMemoryUsage returns memory usage
func (c *MockCluster) GetMemoryUsage() (*MemoryUsage, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	// Mock memory calculations
	totalMemoryMB := 1000.0 // 1GB
	usedMemoryMB := float64(len(c.data)) * 0.01 // Assume 10KB per key

	// Add some base usage and variance
	usedMemoryMB += 100 + rand.Float64()*200 // 100-300MB base + data

	return &MemoryUsage{
		TotalMemory:        fmt.Sprintf("%.1fMB", totalMemoryMB),
		UsedMemory:         fmt.Sprintf("%.1fMB", usedMemoryMB),
		UsageByLevel:       map[string]string{
			"L1": fmt.Sprintf("%.1fMB", usedMemoryMB*0.3),
			"L2": fmt.Sprintf("%.1fMB", usedMemoryMB*0.7),
		},
		FragmentationRatio: 1.0 + rand.Float64()*0.5, // 1.0-1.5 ratio
	}, nil
}

// FlushDB flushes the database
func (c *MockCluster) FlushDB() error {
	c.mutex.Lock()
	defer c.mutex.Unlock()

	c.data = make(map[string]*CacheEntry)
	c.stats.TotalKeys = 0
	return nil
}

// FlushAll flushes all databases
func (c *MockCluster) FlushAll() error {
	return c.FlushDB() // Same as FlushDB for mock
}

// Info returns server information
func (c *MockCluster) Info() (map[string]string, error) {
	c.mutex.RLock()
	defer c.mutex.RUnlock()

	return map[string]string{
		"redis_version":    "6.2.0",
		"redis_mode":      "cluster",
		"cluster_enabled":  "1",
		"connected_clients": strconv.Itoa(rand.Intn(100) + 10),
		"used_memory":      "1048576",
		"total_system_memory": "4294967296",
		"uptime_in_seconds": strconv.Itoa(rand.Intn(86400) + 3600),
	}, nil
}

// Ping tests connectivity
func (c *MockCluster) Ping() error {
	// Mock ping - occasionally fail to simulate network issues
	if rand.Float32() < 0.02 { // 2% failure rate
		return fmt.Errorf("connection timeout")
	}
	return nil
}

// Close closes the connection
func (c *MockCluster) Close() error {
	// Mock close - nothing to actually close
	return nil
}

// Helper methods

func (c *MockCluster) parseTTL(ttl string) (int64, error) {
	if ttl == "" {
		return 0, nil
	}

	// Parse simple TTL formats like "5m", "30s", "1h"
	ttl = strings.ToLower(strings.TrimSpace(ttl))
	
	var multiplier int64 = 1
	if strings.HasSuffix(ttl, "s") {
		multiplier = 1
		ttl = strings.TrimSuffix(ttl, "s")
	} else if strings.HasSuffix(ttl, "m") {
		multiplier = 60
		ttl = strings.TrimSuffix(ttl, "m")
	} else if strings.HasSuffix(ttl, "h") {
		multiplier = 3600
		ttl = strings.TrimSuffix(ttl, "h")
	}

	value, err := strconv.ParseInt(ttl, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid TTL format: %s", ttl)
	}

	return value * multiplier, nil
}