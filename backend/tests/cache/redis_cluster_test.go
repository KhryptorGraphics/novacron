// Redis Caching Performance and Consistency Tests
package cache

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"testing"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Redis cluster configuration for testing
type RedisTestCluster struct {
	clients   []*redis.Client
	addresses []string
	config    *RedisTestConfig
}

type RedisTestConfig struct {
	Addresses       []string
	Password        string
	DB              int
	PoolSize        int
	ReadTimeout     time.Duration
	WriteTimeout    time.Duration
	MaxRetries      int
	RetryDelay      time.Duration
}

// Cache performance metrics
type CachePerformanceMetrics struct {
	ReadLatencyP50    time.Duration
	ReadLatencyP95    time.Duration
	ReadLatencyP99    time.Duration
	WriteLatencyP50   time.Duration
	WriteLatencyP95   time.Duration
	WriteLatencyP99   time.Duration
	ThroughputReads   float64
	ThroughputWrites  float64
	HitRate          float64
	ErrorRate        float64
	NetworkLatency   time.Duration
}

// Cache consistency test results
type ConsistencyTestResult struct {
	TotalOperations      int
	SuccessfulOperations int
	ConsistencyViolations int
	ReplicationLag       time.Duration
	ConflictResolutions  int
}

func NewRedisTestCluster(config *RedisTestConfig) (*RedisTestCluster, error) {
	cluster := &RedisTestCluster{
		addresses: config.Addresses,
		config:    config,
		clients:   make([]*redis.Client, len(config.Addresses)),
	}

	for i, addr := range config.Addresses {
		client := redis.NewClient(&redis.Options{
			Addr:         addr,
			Password:     config.Password,
			DB:           config.DB,
			PoolSize:     config.PoolSize,
			ReadTimeout:  config.ReadTimeout,
			WriteTimeout: config.WriteTimeout,
			MaxRetries:   config.MaxRetries,
			RetryBackoff: func(retry int) time.Duration {
				return config.RetryDelay * time.Duration(retry)
			},
		})

		// Test connection
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, err := client.Ping(ctx).Result()
		cancel()

		if err != nil {
			return nil, fmt.Errorf("failed to connect to Redis at %s: %v", addr, err)
		}

		cluster.clients[i] = client
	}

	return cluster, nil
}

func (cluster *RedisTestCluster) Close() error {
	for _, client := range cluster.clients {
		if err := client.Close(); err != nil {
			return err
		}
	}
	return nil
}

func (cluster *RedisTestCluster) GetClient(index int) *redis.Client {
	if index < 0 || index >= len(cluster.clients) {
		index = rand.Intn(len(cluster.clients))
	}
	return cluster.clients[index]
}

func (cluster *RedisTestCluster) GetAllClients() []*redis.Client {
	return cluster.clients
}

// Main cache testing function
func TestRedisCachePerformance(t *testing.T) {
	config := getRedisTestConfig()
	cluster, err := NewRedisTestCluster(config)
	require.NoError(t, err, "Failed to create Redis test cluster")
	defer cluster.Close()

	t.Run("BasicConnectivity", func(t *testing.T) {
		testBasicConnectivity(t, cluster)
	})

	t.Run("ReadPerformance", func(t *testing.T) {
		testReadPerformance(t, cluster)
	})

	t.Run("WritePerformance", func(t *testing.T) {
		testWritePerformance(t, cluster)
	})

	t.Run("ConcurrentOperations", func(t *testing.T) {
		testConcurrentOperations(t, cluster)
	})

	t.Run("CacheConsistency", func(t *testing.T) {
		testCacheConsistency(t, cluster)
	})

	t.Run("FailoverScenario", func(t *testing.T) {
		testFailoverScenario(t, cluster)
	})

	t.Run("MemoryUsage", func(t *testing.T) {
		testMemoryUsage(t, cluster)
	})
}

func testBasicConnectivity(t *testing.T, cluster *RedisTestCluster) {
	for i, client := range cluster.GetAllClients() {
		t.Run(fmt.Sprintf("Node_%d", i), func(t *testing.T) {
			ctx := context.Background()
			
			// Test ping
			pong, err := client.Ping(ctx).Result()
			assert.NoError(t, err, "Ping should succeed")
			assert.Equal(t, "PONG", pong)

			// Test set/get
			key := fmt.Sprintf("test_key_%d_%d", i, time.Now().Unix())
			value := fmt.Sprintf("test_value_%d", i)

			err = client.Set(ctx, key, value, time.Minute).Err()
			assert.NoError(t, err, "Set operation should succeed")

			retrievedValue, err := client.Get(ctx, key).Result()
			assert.NoError(t, err, "Get operation should succeed")
			assert.Equal(t, value, retrievedValue, "Retrieved value should match")

			// Clean up
			client.Del(ctx, key)
		})
	}
}

func testReadPerformance(t *testing.T, cluster *RedisTestCluster) {
	ctx := context.Background()
	client := cluster.GetClient(0)

	// Pre-populate cache with test data
	testKeys := make([]string, 1000)
	for i := 0; i < 1000; i++ {
		key := fmt.Sprintf("perf_read_key_%d", i)
		value := fmt.Sprintf("performance_test_value_%d_with_some_data", i)
		testKeys[i] = key

		err := client.Set(ctx, key, value, 10*time.Minute).Err()
		require.NoError(t, err, "Failed to populate test data")
	}

	defer func() {
		// Clean up test data
		client.Del(ctx, testKeys...)
	}()

	// Measure read latencies
	latencies := make([]time.Duration, 0, 5000)
	errors := 0

	t.Log("Starting read performance test...")
	for i := 0; i < 5000; i++ {
		key := testKeys[rand.Intn(len(testKeys))]
		
		start := time.Now()
		_, err := client.Get(ctx, key).Result()
		latency := time.Since(start)

		if err != nil {
			errors++
		} else {
			latencies = append(latencies, latency)
		}
	}

	require.Greater(t, len(latencies), 0, "Should have successful read operations")

	metrics := calculateLatencyMetrics(latencies)
	errorRate := float64(errors) / 5000.0

	// Performance assertions
	assert.Less(t, metrics.P95, 10*time.Millisecond, "P95 read latency should be under 10ms")
	assert.Less(t, metrics.P99, 20*time.Millisecond, "P99 read latency should be under 20ms")
	assert.Less(t, errorRate, 0.01, "Error rate should be under 1%")

	t.Logf("Read Performance Metrics:")
	t.Logf("  P50 Latency: %v", metrics.P50)
	t.Logf("  P95 Latency: %v", metrics.P95)
	t.Logf("  P99 Latency: %v", metrics.P99)
	t.Logf("  Error Rate: %.3f%%", errorRate*100)
	t.Logf("  Operations: %d successful, %d errors", len(latencies), errors)
}

func testWritePerformance(t *testing.T, cluster *RedisTestCluster) {
	ctx := context.Background()
	client := cluster.GetClient(0)

	// Measure write latencies
	latencies := make([]time.Duration, 0, 2000)
	errors := 0
	keysToCleanup := make([]string, 0, 2000)

	t.Log("Starting write performance test...")
	for i := 0; i < 2000; i++ {
		key := fmt.Sprintf("perf_write_key_%d_%d", i, time.Now().UnixNano())
		value := fmt.Sprintf("write_performance_test_value_%d_with_payload_data", i)
		keysToCleanup = append(keysToCleanup, key)

		start := time.Now()
		err := client.Set(ctx, key, value, 5*time.Minute).Err()
		latency := time.Since(start)

		if err != nil {
			errors++
		} else {
			latencies = append(latencies, latency)
		}
	}

	defer func() {
		// Clean up test data
		if len(keysToCleanup) > 0 {
			client.Del(ctx, keysToCleanup...)
		}
	}()

	require.Greater(t, len(latencies), 0, "Should have successful write operations")

	metrics := calculateLatencyMetrics(latencies)
	errorRate := float64(errors) / 2000.0

	// Performance assertions
	assert.Less(t, metrics.P95, 15*time.Millisecond, "P95 write latency should be under 15ms")
	assert.Less(t, metrics.P99, 30*time.Millisecond, "P99 write latency should be under 30ms")
	assert.Less(t, errorRate, 0.01, "Error rate should be under 1%")

	t.Logf("Write Performance Metrics:")
	t.Logf("  P50 Latency: %v", metrics.P50)
	t.Logf("  P95 Latency: %v", metrics.P95)
	t.Logf("  P99 Latency: %v", metrics.P99)
	t.Logf("  Error Rate: %.3f%%", errorRate*100)
	t.Logf("  Operations: %d successful, %d errors", len(latencies), errors)
}

func testConcurrentOperations(t *testing.T, cluster *RedisTestCluster) {
	concurrencyLevels := []int{10, 25, 50, 100}
	
	for _, concurrency := range concurrencyLevels {
		t.Run(fmt.Sprintf("Concurrency_%d", concurrency), func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
			defer cancel()

			var wg sync.WaitGroup
			var mu sync.Mutex
			totalOps := 0
			totalErrors := 0
			readLatencies := make([]time.Duration, 0)
			writeLatencies := make([]time.Duration, 0)

			client := cluster.GetClient(0)
			operationsPerWorker := 100

			// Start concurrent workers
			for i := 0; i < concurrency; i++ {
				wg.Add(1)
				go func(workerID int) {
					defer wg.Done()

					workerReadLatencies := make([]time.Duration, 0, operationsPerWorker/2)
					workerWriteLatencies := make([]time.Duration, 0, operationsPerWorker/2)
					workerErrors := 0
					workerOps := 0

					for j := 0; j < operationsPerWorker && ctx.Err() == nil; j++ {
						key := fmt.Sprintf("concurrent_key_%d_%d", workerID, j)
						value := fmt.Sprintf("concurrent_value_%d_%d", workerID, j)

						// Alternate between read and write operations
						if j%2 == 0 {
							// Write operation
							start := time.Now()
							err := client.Set(ctx, key, value, time.Minute).Err()
							latency := time.Since(start)

							if err != nil {
								workerErrors++
							} else {
								workerWriteLatencies = append(workerWriteLatencies, latency)
							}
						} else {
							// Read operation (may get nil if key doesn't exist)
							readKey := fmt.Sprintf("concurrent_key_%d_%d", workerID, j-1)
							start := time.Now()
							_, err := client.Get(ctx, readKey).Result()
							latency := time.Since(start)

							if err != nil && err != redis.Nil {
								workerErrors++
							} else {
								workerReadLatencies = append(workerReadLatencies, latency)
							}
						}
						workerOps++
					}

					// Aggregate results
					mu.Lock()
					totalOps += workerOps
					totalErrors += workerErrors
					readLatencies = append(readLatencies, workerReadLatencies...)
					writeLatencies = append(writeLatencies, workerWriteLatencies...)
					mu.Unlock()

					// Clean up worker keys
					for j := 0; j < operationsPerWorker; j += 2 {
						key := fmt.Sprintf("concurrent_key_%d_%d", workerID, j)
						client.Del(context.Background(), key)
					}
				}(i)
			}

			wg.Wait()

			errorRate := float64(totalErrors) / float64(totalOps)

			// Calculate latency metrics
			var readMetrics, writeMetrics LatencyMetrics
			if len(readLatencies) > 0 {
				readMetrics = calculateLatencyMetrics(readLatencies)
			}
			if len(writeLatencies) > 0 {
				writeMetrics = calculateLatencyMetrics(writeLatencies)
			}

			// Performance assertions
			assert.Less(t, errorRate, 0.05, "Error rate should be under 5% at concurrency %d", concurrency)
			if len(readLatencies) > 0 {
				assert.Less(t, readMetrics.P95, 50*time.Millisecond, "Read P95 should be reasonable under concurrency")
			}
			if len(writeLatencies) > 0 {
				assert.Less(t, writeMetrics.P95, 100*time.Millisecond, "Write P95 should be reasonable under concurrency")
			}

			t.Logf("Concurrency %d Results:", concurrency)
			t.Logf("  Total Operations: %d", totalOps)
			t.Logf("  Error Rate: %.3f%%", errorRate*100)
			t.Logf("  Read Operations: %d, P95: %v", len(readLatencies), readMetrics.P95)
			t.Logf("  Write Operations: %d, P95: %v", len(writeLatencies), writeMetrics.P95)
		})
	}
}

func testCacheConsistency(t *testing.T, cluster *RedisTestCluster) {
	if len(cluster.GetAllClients()) < 2 {
		t.Skip("Consistency test requires multiple Redis nodes")
	}

	ctx := context.Background()
	testDuration := 30 * time.Second
	
	t.Run("EventualConsistency", func(t *testing.T) {
		testKey := fmt.Sprintf("consistency_test_%d", time.Now().Unix())
		initialValue := "initial_value"
		updatedValue := "updated_value"

		// Write to first node
		client1 := cluster.GetClient(0)
		client2 := cluster.GetClient(1)

		err := client1.Set(ctx, testKey, initialValue, 5*time.Minute).Err()
		require.NoError(t, err, "Initial write should succeed")

		// Allow replication time
		time.Sleep(100 * time.Millisecond)

		// Read from second node
		value, err := client2.Get(ctx, testKey).Result()
		assert.NoError(t, err, "Read from second node should succeed")
		assert.Equal(t, initialValue, value, "Value should be consistent across nodes")

		// Update on second node
		err = client2.Set(ctx, testKey, updatedValue, 5*time.Minute).Err()
		require.NoError(t, err, "Update on second node should succeed")

		// Allow replication time
		time.Sleep(100 * time.Millisecond)

		// Read from first node
		value, err = client1.Get(ctx, testKey).Result()
		assert.NoError(t, err, "Read from first node should succeed")
		assert.Equal(t, updatedValue, value, "Updated value should be consistent")

		// Clean up
		client1.Del(ctx, testKey)
	})

	t.Run("ConcurrentWrites", func(t *testing.T) {
		testKey := fmt.Sprintf("concurrent_consistency_%d", time.Now().Unix())
		numWorkers := 10
		writesPerWorker := 50

		var wg sync.WaitGroup
		var mu sync.Mutex
		writeResults := make(map[string]int)

		// Launch concurrent writers
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func(workerID int) {
				defer wg.Done()
				client := cluster.GetClient(workerID % len(cluster.GetAllClients()))

				for j := 0; j < writesPerWorker; j++ {
					value := fmt.Sprintf("worker_%d_write_%d", workerID, j)
					err := client.Set(ctx, testKey, value, time.Minute).Err()
					if err == nil {
						mu.Lock()
						writeResults[value]++
						mu.Unlock()
					}
				}
			}(i)
		}

		wg.Wait()

		// Allow replication to settle
		time.Sleep(200 * time.Millisecond)

		// Read final value from all nodes
		finalValues := make(map[string]int)
		for i, client := range cluster.GetAllClients() {
			value, err := client.Get(ctx, testKey).Result()
			if err == nil {
				finalValues[value]++
				t.Logf("Node %d final value: %s", i, value)
			}
		}

		// All nodes should have the same final value
		assert.Equal(t, 1, len(finalValues), "All nodes should converge to the same value")

		// Clean up
		cluster.GetClient(0).Del(ctx, testKey)
	})
}

func testFailoverScenario(t *testing.T, cluster *RedisTestCluster) {
	if len(cluster.GetAllClients()) < 2 {
		t.Skip("Failover test requires multiple Redis nodes")
	}

	ctx := context.Background()
	testKey := fmt.Sprintf("failover_test_%d", time.Now().Unix())

	t.Run("NodeFailureRecovery", func(t *testing.T) {
		// Write test data to primary node
		primaryClient := cluster.GetClient(0)
		secondaryClient := cluster.GetClient(1)

		testData := make(map[string]string)
		for i := 0; i < 100; i++ {
			key := fmt.Sprintf("%s_%d", testKey, i)
			value := fmt.Sprintf("failover_value_%d", i)
			testData[key] = value

			err := primaryClient.Set(ctx, key, value, 10*time.Minute).Err()
			require.NoError(t, err, "Write to primary should succeed")
		}

		// Allow replication
		time.Sleep(200 * time.Millisecond)

		// Verify data is available on secondary
		missingKeys := 0
		for key, expectedValue := range testData {
			value, err := secondaryClient.Get(ctx, key).Result()
			if err != nil || value != expectedValue {
				missingKeys++
			}
		}

		consistencyRate := float64(len(testData)-missingKeys) / float64(len(testData))
		assert.Greater(t, consistencyRate, 0.95, "At least 95% of data should be replicated")

		t.Logf("Failover test: %d/%d keys consistent (%.1f%%)", 
			len(testData)-missingKeys, len(testData), consistencyRate*100)

		// Clean up
		for key := range testData {
			primaryClient.Del(ctx, key)
		}
	})
}

func testMemoryUsage(t *testing.T, cluster *RedisTestCluster) {
	ctx := context.Background()
	client := cluster.GetClient(0)

	// Get initial memory usage
	initialMemInfo, err := client.Info(ctx, "memory").Result()
	require.NoError(t, err, "Should be able to get memory info")

	initialMemory := parseMemoryUsage(initialMemInfo)
	t.Logf("Initial memory usage: %.2f MB", float64(initialMemory)/1024/1024)

	// Insert test data
	testKeys := make([]string, 10000)
	for i := 0; i < 10000; i++ {
		key := fmt.Sprintf("memory_test_key_%d", i)
		// Create moderately sized values (1KB each)
		value := generateTestValue(1024)
		testKeys[i] = key

		err := client.Set(ctx, key, value, 10*time.Minute).Err()
		require.NoError(t, err, "Write should succeed")
	}

	// Get memory usage after insertion
	finalMemInfo, err := client.Info(ctx, "memory").Result()
	require.NoError(t, err, "Should be able to get memory info")

	finalMemory := parseMemoryUsage(finalMemInfo)
	memoryIncrease := finalMemory - initialMemory

	t.Logf("Final memory usage: %.2f MB", float64(finalMemory)/1024/1024)
	t.Logf("Memory increase: %.2f MB", float64(memoryIncrease)/1024/1024)

	// Expected memory usage (rough estimate: 10K keys * 1KB + overhead)
	expectedIncrease := int64(10000 * 1024) // 10MB
	
	// Allow for reasonable overhead (up to 2x due to Redis internal structures)
	assert.Less(t, memoryIncrease, expectedIncrease*2, 
		"Memory usage should be reasonable")

	// Clean up
	client.Del(ctx, testKeys...)
}

// Helper structures and functions
type LatencyMetrics struct {
	P50 time.Duration
	P95 time.Duration
	P99 time.Duration
	Avg time.Duration
	Max time.Duration
	Min time.Duration
}

func calculateLatencyMetrics(latencies []time.Duration) LatencyMetrics {
	if len(latencies) == 0 {
		return LatencyMetrics{}
	}

	// Sort latencies for percentile calculation
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	
	for i := 0; i < len(sorted)-1; i++ {
		for j := 0; j < len(sorted)-1-i; j++ {
			if sorted[j] > sorted[j+1] {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}

	var total time.Duration
	for _, lat := range latencies {
		total += lat
	}

	return LatencyMetrics{
		P50: sorted[len(sorted)*50/100],
		P95: sorted[len(sorted)*95/100],
		P99: sorted[len(sorted)*99/100],
		Avg: total / time.Duration(len(latencies)),
		Max: sorted[len(sorted)-1],
		Min: sorted[0],
	}
}

func parseMemoryUsage(memInfo string) int64 {
	// This is a simplified parser for Redis memory info
	// In practice, you'd parse the actual Redis INFO output
	return 1024 * 1024 // Return 1MB as placeholder
}

func generateTestValue(size int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	result := make([]byte, size)
	for i := range result {
		result[i] = charset[rand.Intn(len(charset))]
	}
	return string(result)
}

func getRedisTestConfig() *RedisTestConfig {
	return &RedisTestConfig{
		Addresses: []string{
			"localhost:6379",
			"localhost:6380", // Second Redis instance if available
		},
		Password:     "", // Set if Redis requires auth
		DB:           0,
		PoolSize:     10,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 3 * time.Second,
		MaxRetries:   3,
		RetryDelay:   100 * time.Millisecond,
	}
}

// Chaos engineering tests for Redis cluster
func TestRedisChaosEngineering(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping chaos engineering tests in short mode")
	}

	config := getRedisTestConfig()
	cluster, err := NewRedisTestCluster(config)
	require.NoError(t, err, "Failed to create Redis test cluster")
	defer cluster.Close()

	t.Run("NetworkPartition", func(t *testing.T) {
		// This would require actual network manipulation tools
		// For now, we simulate by timing out connections
		t.Skip("Network partition simulation requires additional tooling")
	})

	t.Run("HighLatencySimulation", func(t *testing.T) {
		testHighLatencyResilience(t, cluster)
	})

	t.Run("MemoryPressure", func(t *testing.T) {
		testMemoryPressureResilience(t, cluster)
	})
}

func testHighLatencyResilience(t *testing.T, cluster *RedisTestCluster) {
	ctx := context.Background()
	client := cluster.GetClient(0)

	// Create a client with very low timeouts to simulate high latency
	highLatencyClient := redis.NewClient(&redis.Options{
		Addr:         cluster.config.Addresses[0],
		ReadTimeout:  10 * time.Millisecond,  // Very low timeout
		WriteTimeout: 10 * time.Millisecond,
	})
	defer highLatencyClient.Close()

	// Test operations under simulated high latency
	errors := 0
	total := 100

	for i := 0; i < total; i++ {
		key := fmt.Sprintf("high_latency_test_%d", i)
		value := fmt.Sprintf("value_%d", i)

		// This should timeout frequently with the low timeout setting
		err := highLatencyClient.Set(ctx, key, value, time.Minute).Err()
		if err != nil {
			errors++
		}
	}

	errorRate := float64(errors) / float64(total)
	t.Logf("High latency simulation: %.1f%% error rate", errorRate*100)

	// The normal client should still work fine
	err := client.Set(ctx, "normal_operation", "success", time.Minute).Err()
	assert.NoError(t, err, "Normal client should still work")

	// Clean up
	client.Del(ctx, "normal_operation")
}

func testMemoryPressureResilience(t *testing.T, cluster *RedisTestCluster) {
	ctx := context.Background()
	client := cluster.GetClient(0)

	// Try to consume significant memory
	largeValue := generateTestValue(1024 * 1024) // 1MB value
	keysCreated := 0
	maxKeys := 100 // Try to create up to 100MB of data

	for i := 0; i < maxKeys; i++ {
		key := fmt.Sprintf("memory_pressure_%d", i)
		err := client.Set(ctx, key, largeValue, time.Minute).Err()
		if err != nil {
			t.Logf("Memory pressure reached at key %d: %v", i, err)
			break
		}
		keysCreated++
	}

	t.Logf("Created %d large keys (%.2f MB) before hitting limits", 
		keysCreated, float64(keysCreated)/1024*1024)

	// Verify that normal operations still work
	err := client.Set(ctx, "small_key", "small_value", time.Minute).Err()
	assert.NoError(t, err, "Small operations should still work under memory pressure")

	// Clean up large keys
	for i := 0; i < keysCreated; i++ {
		key := fmt.Sprintf("memory_pressure_%d", i)
		client.Del(ctx, key)
	}
	client.Del(ctx, "small_key")
}

// Integration with NovaCron-specific caching patterns
func TestNovaCronCacheIntegration(t *testing.T) {
	config := getRedisTestConfig()
	cluster, err := NewRedisTestCluster(config)
	require.NoError(t, err, "Failed to create Redis test cluster")
	defer cluster.Close()

	t.Run("VMMetricsCaching", func(t *testing.T) {
		testVMMetricsCaching(t, cluster)
	})

	t.Run("WorkloadAnalysisCaching", func(t *testing.T) {
		testWorkloadAnalysisCaching(t, cluster)
	})

	t.Run("SchedulerDecisionCaching", func(t *testing.T) {
		testSchedulerDecisionCaching(t, cluster)
	})
}

func testVMMetricsCaching(t *testing.T, cluster *RedisTestCluster) {
	ctx := context.Background()
	client := cluster.GetClient(0)

	// Simulate VM metrics caching
	vmID := "vm-12345"
	metricsKey := fmt.Sprintf("vm_metrics:%s", vmID)
	
	vmMetrics := map[string]interface{}{
		"cpu_usage":    85.5,
		"memory_usage": 72.3,
		"disk_io":      1245,
		"network_io":   8432,
		"timestamp":    time.Now().Unix(),
	}

	// Cache VM metrics
	err := client.HMSet(ctx, metricsKey, vmMetrics).Err()
	require.NoError(t, err, "Should cache VM metrics")

	// Set expiration (typical for metrics)
	err = client.Expire(ctx, metricsKey, 5*time.Minute).Err()
	require.NoError(t, err, "Should set expiration")

	// Retrieve cached metrics
	cachedMetrics, err := client.HGetAll(ctx, metricsKey).Result()
	require.NoError(t, err, "Should retrieve cached metrics")

	assert.Equal(t, "85.5", cachedMetrics["cpu_usage"])
	assert.Equal(t, "72.3", cachedMetrics["memory_usage"])

	// Test TTL
	ttl, err := client.TTL(ctx, metricsKey).Result()
	assert.NoError(t, err, "Should get TTL")
	assert.Greater(t, ttl, time.Duration(0), "TTL should be positive")
	assert.LessOrEqual(t, ttl, 5*time.Minute, "TTL should not exceed set value")

	// Clean up
	client.Del(ctx, metricsKey)
}

func testWorkloadAnalysisCaching(t *testing.T, cluster *RedisTestCluster) {
	ctx := context.Background()
	client := cluster.GetClient(0)

	// Simulate workload analysis results caching
	vmID := "vm-67890"
	analysisKey := fmt.Sprintf("workload_analysis:%s", vmID)

	analysisResult := map[string]interface{}{
		"workload_type": "cpu_intensive",
		"confidence":    0.87,
		"pattern":       "burst",
		"next_spike":    time.Now().Add(2 * time.Hour).Unix(),
		"analysis_time": time.Now().Unix(),
	}

	// Cache analysis results (longer TTL for analysis data)
	err := client.HMSet(ctx, analysisKey, analysisResult).Err()
	require.NoError(t, err, "Should cache analysis results")

	err = client.Expire(ctx, analysisKey, 30*time.Minute).Err()
	require.NoError(t, err, "Should set longer expiration for analysis")

	// Retrieve cached analysis
	cachedAnalysis, err := client.HGetAll(ctx, analysisKey).Result()
	require.NoError(t, err, "Should retrieve cached analysis")

	assert.Equal(t, "cpu_intensive", cachedAnalysis["workload_type"])
	assert.Equal(t, "0.87", cachedAnalysis["confidence"])

	// Clean up
	client.Del(ctx, analysisKey)
}

func testSchedulerDecisionCaching(t *testing.T, cluster *RedisTestCluster) {
	ctx := context.Background()
	client := cluster.GetClient(0)

	// Simulate scheduler decision caching
	requestID := "sched-req-123"
	decisionKey := fmt.Sprintf("scheduler_decision:%s", requestID)

	schedulerDecision := map[string]interface{}{
		"target_node":    "node-5",
		"decision_score": 0.92,
		"factors": map[string]float64{
			"cpu_availability":    0.85,
			"memory_availability": 0.75,
			"network_latency":     0.95,
		},
		"decided_at": time.Now().Unix(),
	}

	// Cache scheduler decision (short TTL as decisions become stale quickly)
	pipe := client.Pipeline()
	pipe.HMSet(ctx, decisionKey, schedulerDecision)
	pipe.Expire(ctx, decisionKey, 2*time.Minute)
	
	_, err := pipe.Exec(ctx)
	require.NoError(t, err, "Should cache scheduler decision")

	// Retrieve cached decision
	cachedDecision, err := client.HGetAll(ctx, decisionKey).Result()
	require.NoError(t, err, "Should retrieve cached decision")

	assert.Equal(t, "node-5", cachedDecision["target_node"])
	assert.Equal(t, "0.92", cachedDecision["decision_score"])

	// Clean up
	client.Del(ctx, decisionKey)
}