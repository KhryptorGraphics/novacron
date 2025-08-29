package cache

import (
	"context"
	"fmt"
	"time"

	"github.com/sirupsen/logrus"
)

// ExampleUsage demonstrates how to use the NovaCron cache infrastructure
func ExampleUsage() {
	logger := logrus.New()
	logger.SetLevel(logrus.InfoLevel)

	// Example 1: Basic cache setup with default configuration
	fmt.Println("=== Example 1: Basic Multi-Tier Cache Setup ===")
	
	config := DefaultCacheConfig()
	cache, err := NewMultiTierCache(config, logger)
	if err != nil {
		logger.WithError(err).Fatal("Failed to create cache")
	}
	defer cache.Close()

	// Basic cache operations
	ctx := context.Background()
	
	// Set a value
	err = cache.Set(ctx, "test_key", []byte("test_value"), 5*time.Minute)
	if err != nil {
		logger.WithError(err).Error("Failed to set cache value")
	} else {
		logger.Info("Successfully set cache value")
	}

	// Get a value
	value, err := cache.Get(ctx, "test_key")
	if err != nil {
		logger.WithError(err).Error("Failed to get cache value")
	} else {
		logger.WithField("value", string(value)).Info("Successfully retrieved cache value")
	}

	// Get cache stats
	stats := cache.GetStats()
	logger.WithFields(logrus.Fields{
		"hit_rate":    stats.HitRate,
		"total_hits":  stats.Hits,
		"total_misses": stats.Misses,
	}).Info("Cache statistics")

	fmt.Println()

	// Example 2: Redis cluster cache setup
	fmt.Println("=== Example 2: Redis Cluster Cache Setup ===")
	
	redisConfig := DefaultCacheConfig()
	redisConfig.L1Enabled = false // Disable L1 for this example
	redisConfig.L3Enabled = false // Disable L3 for this example
	redisConfig.RedisCluster = true
	redisConfig.RedisAddrs = []string{
		"localhost:7001",
		"localhost:7002", 
		"localhost:7003",
		"localhost:7004",
		"localhost:7005",
		"localhost:7006",
	}

	redisCache, err := NewMultiTierCache(redisConfig, logger)
	if err != nil {
		logger.WithError(err).Warn("Failed to create Redis cluster cache (expected if cluster not running)")
	} else {
		defer redisCache.Close()
		logger.Info("Redis cluster cache created successfully")
	}

	fmt.Println()

	// Example 3: Redis Sentinel setup
	fmt.Println("=== Example 3: Redis Sentinel Cache Setup ===")
	
	sentinelConfig := DefaultCacheConfig()
	sentinelConfig.L1Enabled = false
	sentinelConfig.L3Enabled = false
	sentinelConfig.SentinelEnabled = true
	sentinelConfig.SentinelAddrs = []string{
		"localhost:26379",
		"localhost:26380",
		"localhost:26381",
	}
	sentinelConfig.SentinelMaster = "mymaster"

	sentinelCache, err := NewMultiTierCache(sentinelConfig, logger)
	if err != nil {
		logger.WithError(err).Warn("Failed to create Redis sentinel cache (expected if sentinel not running)")
	} else {
		defer sentinelCache.Close()
		logger.Info("Redis sentinel cache created successfully")
	}

	fmt.Println()

	// Example 4: VM cache integration
	fmt.Println("=== Example 4: VM Cache Integration ===")
	
	vmCacheManager := NewVMCacheManager(cache, DefaultVMCacheConfig(), logger)
	
	// Simulate VM state caching
	vmID := "vm-123"
	vmState := map[string]interface{}{
		"id":         vmID,
		"name":       "test-vm",
		"state":      "running",
		"type":       "kvm",
		"node_id":    "node-1",
		"created_at": time.Now().Unix(),
		"updated_at": time.Now().Unix(),
	}

	// Cache VM state
	err = vmCacheManager.SetVMState(ctx, vmID, vmState)
	if err != nil {
		logger.WithError(err).Error("Failed to cache VM state")
	} else {
		logger.WithField("vm_id", vmID).Info("Successfully cached VM state")
	}

	// Retrieve VM state from cache
	cachedState, err := vmCacheManager.GetVMState(ctx, vmID)
	if err != nil {
		logger.WithError(err).Error("Failed to retrieve VM state from cache")
	} else {
		logger.WithField("cached_state", cachedState).Info("Successfully retrieved VM state from cache")
	}

	// Test VM cache invalidation
	vmCacheManager.OnVMStateChange(ctx, vmID, "running", "stopped")
	logger.WithField("vm_id", vmID).Info("Triggered VM state change cache invalidation")

	fmt.Println()

	// Example 5: Metrics collection
	fmt.Println("=== Example 5: Cache Metrics Collection ===")
	
	metricsConfig := &MetricsConfig{
		CollectionInterval: 5 * time.Second,
		EnableHistograms:   true,
		EnableHeatmap:      true,
		RetentionDays:      1,
	}

	metricsCollector := NewCacheMetricsCollector(cache, metricsConfig, logger)
	defer metricsCollector.Stop()

	// Generate some cache activity for metrics
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("metrics_test_%d", i)
		cache.Set(ctx, key, []byte(fmt.Sprintf("value_%d", i)), time.Minute)
		cache.Get(ctx, key)
	}

	// Wait a bit for metrics collection
	time.Sleep(2 * time.Second)

	// Get metrics summary
	summary := metricsCollector.GetSummary()
	logger.WithField("metrics_summary", summary).Info("Cache metrics summary")

	// Get detailed metrics
	detailedMetrics := metricsCollector.GetMetrics()
	logger.WithFields(logrus.Fields{
		"hit_rate":         detailedMetrics.Basic.HitRate,
		"avg_response_time": fmt.Sprintf("%.2fμs", float64(detailedMetrics.Basic.AvgResponseTimeNs)/1000),
		"total_operations": detailedMetrics.Basic.Hits + detailedMetrics.Basic.Misses + detailedMetrics.Basic.Sets,
	}).Info("Detailed cache metrics")

	fmt.Println()

	// Example 6: Cache monitoring
	fmt.Println("=== Example 6: Cache Monitoring ===")
	
	monitorConfig := &MonitorConfig{
		HealthCheckInterval: 10 * time.Second,
		MetricsPort:         9091,
		EnableWebUI:         false, // Disable for this example
		AlertThresholds: &AlertThresholds{
			HitRateMin:      0.8,
			ResponseTimeMax: 100 * time.Millisecond,
			ErrorRateMax:    0.05,
		},
	}

	monitor := NewCacheMonitor(cache, metricsCollector, monitorConfig, logger)

	// Get health status
	health := monitor.GetHealthStatus()
	logger.WithFields(logrus.Fields{
		"overall_health":   health.Overall,
		"components_count": len(health.Components),
		"alerts_count":     len(health.Alerts),
	}).Info("Cache health status")

	// Print component health details
	for component, componentHealth := range health.Components {
		logger.WithFields(logrus.Fields{
			"component": component,
			"status":    componentHealth.Status,
			"message":   componentHealth.Message,
		}).Info("Component health")
	}

	// Print any alerts
	for _, alert := range health.Alerts {
		logger.WithFields(logrus.Fields{
			"alert_level":   alert.Level,
			"alert_component": alert.Component,
			"alert_message": alert.Message,
		}).Warn("Cache alert")
	}

	fmt.Println()

	// Example 7: Batch operations
	fmt.Println("=== Example 7: Batch Cache Operations ===")
	
	// Prepare batch data
	batchItems := make(map[string]CacheItem)
	for i := 0; i < 5; i++ {
		key := fmt.Sprintf("batch_key_%d", i)
		value := fmt.Sprintf("batch_value_%d", i)
		batchItems[key] = CacheItem{
			Value: []byte(value),
			TTL:   5 * time.Minute,
		}
	}

	// Batch set
	err = cache.SetMulti(ctx, batchItems)
	if err != nil {
		logger.WithError(err).Error("Failed to set batch items")
	} else {
		logger.WithField("items_count", len(batchItems)).Info("Successfully set batch items")
	}

	// Batch get
	keys := make([]string, 0, len(batchItems))
	for key := range batchItems {
		keys = append(keys, key)
	}

	results, err := cache.GetMulti(ctx, keys)
	if err != nil {
		logger.WithError(err).Error("Failed to get batch items")
	} else {
		logger.WithFields(logrus.Fields{
			"requested_items": len(keys),
			"retrieved_items": len(results),
		}).Info("Successfully retrieved batch items")
	}

	// Batch delete
	err = cache.DeleteMulti(ctx, keys)
	if err != nil {
		logger.WithError(err).Error("Failed to delete batch items")
	} else {
		logger.WithField("deleted_items", len(keys)).Info("Successfully deleted batch items")
	}

	fmt.Println()

	// Example 8: Performance testing
	fmt.Println("=== Example 8: Performance Testing ===")
	
	performanceTest(cache, logger)
}

// performanceTest runs a basic performance test
func performanceTest(cache Cache, logger *logrus.Logger) {
	ctx := context.Background()
	numOperations := 1000
	
	logger.WithField("operations", numOperations).Info("Starting cache performance test")
	
	start := time.Now()

	// Write performance
	writeStart := time.Now()
	for i := 0; i < numOperations; i++ {
		key := fmt.Sprintf("perf_test_%d", i)
		value := []byte(fmt.Sprintf("performance_test_value_%d", i))
		cache.Set(ctx, key, value, 10*time.Minute)
	}
	writeDuration := time.Since(writeStart)

	// Read performance
	readStart := time.Now()
	hits := 0
	for i := 0; i < numOperations; i++ {
		key := fmt.Sprintf("perf_test_%d", i)
		if _, err := cache.Get(ctx, key); err == nil {
			hits++
		}
	}
	readDuration := time.Since(readStart)

	totalDuration := time.Since(start)

	logger.WithFields(logrus.Fields{
		"total_duration":     totalDuration,
		"write_duration":     writeDuration,
		"read_duration":      readDuration,
		"write_ops_per_sec":  float64(numOperations) / writeDuration.Seconds(),
		"read_ops_per_sec":   float64(numOperations) / readDuration.Seconds(),
		"total_ops_per_sec":  float64(numOperations*2) / totalDuration.Seconds(),
		"cache_hit_rate":     float64(hits) / float64(numOperations),
		"avg_write_latency":  writeDuration / time.Duration(numOperations),
		"avg_read_latency":   readDuration / time.Duration(numOperations),
	}).Info("Performance test completed")

	// Clean up performance test data
	for i := 0; i < numOperations; i++ {
		key := fmt.Sprintf("perf_test_%d", i)
		cache.Delete(ctx, key)
	}

	logger.Info("Performance test cleanup completed")
}

// ExampleVMIntegration demonstrates VM-specific cache usage
func ExampleVMIntegration() {
	logger := logrus.New()
	logger.SetLevel(logrus.InfoLevel)

	fmt.Println("=== VM Cache Integration Example ===")

	// Create cache
	config := DefaultCacheConfig()
	cache, err := NewMultiTierCache(config, logger)
	if err != nil {
		logger.WithError(err).Fatal("Failed to create cache")
	}
	defer cache.Close()

	// Create VM cache manager
	vmConfig := DefaultVMCacheConfig()
	vmConfig.CriticalVMList = []string{"vm-prod-1", "vm-prod-2", "vm-db-1"}
	vmCacheManager := NewVMCacheManager(cache, vmConfig, logger)

	ctx := context.Background()

	// Simulate caching multiple VM states
	vms := []struct {
		ID     string
		Name   string
		State  string
		NodeID string
	}{
		{"vm-prod-1", "production-web", "running", "node-1"},
		{"vm-prod-2", "production-api", "running", "node-2"},
		{"vm-db-1", "database-primary", "running", "node-3"},
		{"vm-test-1", "test-environment", "stopped", "node-1"},
	}

	// Cache VM states
	for _, vm := range vms {
		vmState := map[string]interface{}{
			"id":         vm.ID,
			"name":       vm.Name,
			"state":      vm.State,
			"type":       "kvm",
			"node_id":    vm.NodeID,
			"created_at": time.Now().Add(-24 * time.Hour).Unix(),
			"updated_at": time.Now().Unix(),
		}

		err := vmCacheManager.SetVMState(ctx, vm.ID, vmState)
		if err != nil {
			logger.WithError(err).WithField("vm_id", vm.ID).Error("Failed to cache VM state")
		}

		// Also cache some metrics
		metrics := map[string]interface{}{
			"cpu_usage":    float64(20 + (len(vm.ID) % 60)), // Mock CPU usage
			"memory_usage": float64(30 + (len(vm.ID) % 50)), // Mock memory usage
			"disk_usage":   float64(10 + (len(vm.ID) % 30)), // Mock disk usage
			"network_io": map[string]interface{}{
				"rx_bytes": 1024 * 1024 * (len(vm.ID) % 100),
				"tx_bytes": 1024 * 1024 * (len(vm.ID) % 80),
			},
			"last_updated": time.Now().Unix(),
		}

		err = vmCacheManager.SetVMMetrics(ctx, vm.ID, metrics)
		if err != nil {
			logger.WithError(err).WithField("vm_id", vm.ID).Error("Failed to cache VM metrics")
		}
	}

	logger.WithField("vms_cached", len(vms)).Info("Cached VM data")

	// Test batch retrieval
	vmIDs := make([]string, len(vms))
	for i, vm := range vms {
		vmIDs[i] = vm.ID
	}

	batchStates, err := vmCacheManager.GetMultipleVMStates(ctx, vmIDs)
	if err != nil {
		logger.WithError(err).Error("Failed to get batch VM states")
	} else {
		logger.WithField("retrieved_states", len(batchStates)).Info("Successfully retrieved batch VM states")
		for vmID, state := range batchStates {
			logger.WithFields(logrus.Fields{
				"vm_id": vmID,
				"state": state,
			}).Debug("Retrieved VM state")
		}
	}

	batchMetrics, err := vmCacheManager.GetMultipleVMMetrics(ctx, vmIDs)
	if err != nil {
		logger.WithError(err).Error("Failed to get batch VM metrics")
	} else {
		logger.WithField("retrieved_metrics", len(batchMetrics)).Info("Successfully retrieved batch VM metrics")
	}

	// Simulate VM state changes
	logger.Info("Simulating VM state changes...")
	
	vmCacheManager.OnVMStateChange(ctx, "vm-test-1", "stopped", "running")
	vmCacheManager.OnVMResourceChange(ctx, "vm-prod-1")
	vmCacheManager.OnVMCreated(ctx, "vm-new-1", "node-2")
	vmCacheManager.OnVMDeleted(ctx, "vm-old-1", "node-3")

	// Get cache statistics
	stats := vmCacheManager.GetCacheStats()
	logger.WithFields(logrus.Fields{
		"hit_rate":     stats.HitRate,
		"l1_hits":      stats.L1Hits,
		"l2_hits":      stats.L2Hits,
		"l3_hits":      stats.L3Hits,
		"total_operations": stats.Hits + stats.Misses + stats.Sets + stats.Deletes,
	}).Info("VM cache statistics")

	logger.Info("VM cache integration example completed")
}

// ExamplePrometheusIntegration shows how to integrate with Prometheus
func ExamplePrometheusIntegration() {
	logger := logrus.New()
	logger.SetLevel(logrus.InfoLevel)

	fmt.Println("=== Prometheus Integration Example ===")

	// Create cache and metrics collector
	config := DefaultCacheConfig()
	cache, err := NewMultiTierCache(config, logger)
	if err != nil {
		logger.WithError(err).Fatal("Failed to create cache")
	}
	defer cache.Close()

	metricsConfig := &MetricsConfig{
		CollectionInterval: 5 * time.Second,
		EnableHistograms:   true,
		EnableHeatmap:      false, // Disable for this example
		RetentionDays:      1,
	}

	metricsCollector := NewCacheMetricsCollector(cache, metricsConfig, logger)
	defer metricsCollector.Stop()

	// Generate some activity
	ctx := context.Background()
	for i := 0; i < 50; i++ {
		key := fmt.Sprintf("prometheus_test_%d", i)
		value := []byte(fmt.Sprintf("test_value_%d", i))
		
		cache.Set(ctx, key, value, 5*time.Minute)
		cache.Get(ctx, key)
		
		// Record custom metrics
		metricsCollector.RecordOperation("get", time.Duration(i*1000), true, "prometheus_test:")
	}

	// Wait for metrics collection
	time.Sleep(2 * time.Second)

	// Export Prometheus metrics
	prometheusMetrics := metricsCollector.ExportPrometheusMetrics()
	logger.Info("Prometheus metrics exported:")
	fmt.Println(prometheusMetrics)

	// Get JSON metrics
	if metricsJSON, err := metricsCollector.GetMetricsJSON(); err == nil {
		logger.Info("JSON metrics available for external monitoring systems")
		logger.Debug("JSON metrics:", string(metricsJSON)[:200], "...") // First 200 chars
	}

	logger.Info("Prometheus integration example completed")
}