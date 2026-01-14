package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/cache"
	"github.com/sirupsen/logrus"
)

func main() {
	// Command line flags
	var (
		redisMasterAddr    = flag.String("redis-master", getEnvOrDefault("REDIS_MASTER_ADDR", "localhost:6379"), "Redis master address")
		redisSlaveAddrs    = flag.String("redis-slaves", getEnvOrDefault("REDIS_SLAVE_ADDRS", ""), "Redis slave addresses (comma-separated)")
		sentinelAddrs      = flag.String("sentinel-addrs", getEnvOrDefault("SENTINEL_ADDRS", ""), "Redis sentinel addresses (comma-separated)")
		sentinelMaster     = flag.String("sentinel-master", getEnvOrDefault("SENTINEL_MASTER_NAME", "mymaster"), "Redis sentinel master name")
		metricsPort        = flag.String("metrics-port", getEnvOrDefault("METRICS_PORT", "9091"), "Metrics server port")
		webUIPort          = flag.String("webui-port", getEnvOrDefault("WEBUI_PORT", "8082"), "Web UI port")
		logLevel           = flag.String("log-level", getEnvOrDefault("LOG_LEVEL", "info"), "Log level")
		enableWebUI        = flag.Bool("enable-webui", getEnvOrDefault("ENABLE_WEBUI", "true") == "true", "Enable web UI")
		healthCheckInterval = flag.String("health-interval", getEnvOrDefault("HEALTH_CHECK_INTERVAL", "30s"), "Health check interval")
	)
	flag.Parse()

	// Setup logger
	logger := logrus.New()
	level, err := logrus.ParseLevel(*logLevel)
	if err != nil {
		logger.WithError(err).Warn("Invalid log level, using info")
		level = logrus.InfoLevel
	}
	logger.SetLevel(level)

	logger.WithFields(logrus.Fields{
		"redis_master":    *redisMasterAddr,
		"redis_slaves":    *redisSlaveAddrs,
		"sentinel_addrs":  *sentinelAddrs,
		"sentinel_master": *sentinelMaster,
		"metrics_port":    *metricsPort,
		"webui_port":      *webUIPort,
		"enable_webui":    *enableWebUI,
	}).Info("Starting NovaCron Cache Monitor")

	// Parse health check interval
	healthInterval, err := time.ParseDuration(*healthCheckInterval)
	if err != nil {
		logger.WithError(err).Warn("Invalid health check interval, using 30s")
		healthInterval = 30 * time.Second
	}

	// Create cache configuration
	cacheConfig := cache.DefaultCacheConfig()
	
	// Configure Redis addresses
	cacheConfig.RedisAddrs = []string{*redisMasterAddr}
	if *redisSlaveAddrs != "" {
		slaves := strings.Split(*redisSlaveAddrs, ",")
		for _, slave := range slaves {
			cacheConfig.RedisAddrs = append(cacheConfig.RedisAddrs, strings.TrimSpace(slave))
		}
	}

	// Configure Sentinel if provided
	if *sentinelAddrs != "" {
		cacheConfig.SentinelEnabled = true
		cacheConfig.SentinelAddrs = strings.Split(*sentinelAddrs, ",")
		for i, addr := range cacheConfig.SentinelAddrs {
			cacheConfig.SentinelAddrs[i] = strings.TrimSpace(addr)
		}
		cacheConfig.SentinelMaster = *sentinelMaster
	}

	// Disable L1 and L3 for monitor (focus on Redis)
	cacheConfig.L1Enabled = false
	cacheConfig.L3Enabled = false

	// Create cache instance
	cacheInstance, err := cache.NewMultiTierCache(cacheConfig, logger)
	if err != nil {
		logger.WithError(err).Fatal("Failed to create cache instance")
	}
	defer cacheInstance.Close()

	// Create metrics collector
	metricsConfig := &cache.MetricsConfig{
		CollectionInterval: 10 * time.Second,
		RetentionDays:      7,
		EnableHistograms:   true,
		HistogramBuckets:   []float64{0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000},
		EnableHeatmap:      true,
		HeatmapResolution:  1 * time.Minute,
	}

	metricsCollector := cache.NewCacheMetricsCollector(cacheInstance, metricsConfig, logger)
	defer metricsCollector.Stop()

	// Parse ports
	var metricsPortInt, webUIPortInt int
	if _, err := fmt.Sscanf(*metricsPort, "%d", &metricsPortInt); err != nil {
		logger.WithError(err).Fatal("Invalid metrics port")
	}
	if _, err := fmt.Sscanf(*webUIPort, "%d", &webUIPortInt); err != nil {
		logger.WithError(err).Fatal("Invalid web UI port")
	}

	// Create monitor configuration
	monitorConfig := &cache.MonitorConfig{
		HealthCheckInterval: healthInterval,
		MetricsPort:         metricsPortInt,
		EnableWebUI:         *enableWebUI,
		WebUIPort:          webUIPortInt,
		AlertThresholds: &cache.AlertThresholds{
			HitRateMin:         0.85,
			ResponseTimeMax:    100 * time.Millisecond,
			ErrorRateMax:       0.01,
			MemoryUsageMax:     1024 * 1024 * 1024, // 1GB
			ConnectionFailures: 5,
		},
	}

	// Create cache monitor
	_ = cache.NewCacheMonitor(cacheInstance, metricsCollector, monitorConfig, logger)

	logger.Info("Cache monitor started successfully")

	// Test cache connectivity
	ctx := context.Background()
	testConnectivity(ctx, cacheInstance, logger)

	// Setup graceful shutdown
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt, syscall.SIGTERM)

	// Wait for signal
	<-c
	logger.Info("Shutting down cache monitor...")

	// Graceful shutdown
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Stop monitoring components
	metricsCollector.Stop()
	
	// Close cache
	if err := cacheInstance.Close(); err != nil {
		logger.WithError(err).Error("Error closing cache")
	}

	select {
	case <-shutdownCtx.Done():
		logger.Error("Shutdown timeout exceeded")
	default:
		logger.Info("Cache monitor shutdown complete")
	}
}

// testConnectivity tests cache connectivity and logs results
func testConnectivity(ctx context.Context, cacheInstance cache.Cache, logger *logrus.Logger) {
	logger.Info("Testing cache connectivity...")

	// Test basic operations
	testKey := "monitor_connectivity_test"
	testValue := []byte("connectivity_test_value")

	// Test SET
	if err := cacheInstance.Set(ctx, testKey, testValue, 1*time.Minute); err != nil {
		logger.WithError(err).Error("Cache SET test failed")
		return
	}

	// Test GET
	if value, err := cacheInstance.Get(ctx, testKey); err != nil {
		logger.WithError(err).Error("Cache GET test failed")
		return
	} else if string(value) != string(testValue) {
		logger.Error("Cache GET test returned incorrect value")
		return
	}

	// Test EXISTS
	if exists, err := cacheInstance.Exists(ctx, testKey); err != nil {
		logger.WithError(err).Error("Cache EXISTS test failed")
		return
	} else if !exists {
		logger.Error("Cache EXISTS test returned false for existing key")
		return
	}

	// Test DELETE
	if err := cacheInstance.Delete(ctx, testKey); err != nil {
		logger.WithError(err).Error("Cache DELETE test failed")
		return
	}

	// Verify DELETE
	if exists, err := cacheInstance.Exists(ctx, testKey); err != nil {
		logger.WithError(err).Warn("Cache EXISTS test failed after delete")
	} else if exists {
		logger.Warn("Cache EXISTS test returned true after delete")
	}

	// Get cache stats
	stats := cacheInstance.GetStats()
	logger.WithFields(logrus.Fields{
		"hit_rate":          stats.HitRate,
		"total_operations":  stats.Hits + stats.Misses + stats.Sets + stats.Deletes,
		"avg_response_time": fmt.Sprintf("%.2fμs", float64(stats.AvgResponseTimeNs)/1000),
		"errors":           stats.Errors,
	}).Info("Cache connectivity test passed")
}

// getEnvOrDefault returns environment variable value or default
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

