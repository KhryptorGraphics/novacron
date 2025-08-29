package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/sirupsen/logrus"
)

// CacheMonitor provides monitoring and health checking for cache infrastructure
type CacheMonitor struct {
	cache           Cache
	metricsCollector *CacheMetricsCollector
	logger          *logrus.Logger
	config          *MonitorConfig
}

// MonitorConfig holds cache monitoring configuration
type MonitorConfig struct {
	HealthCheckInterval time.Duration `json:"health_check_interval"`
	MetricsPort         int           `json:"metrics_port"`
	AlertThresholds     *AlertThresholds `json:"alert_thresholds"`
	EnableWebUI         bool          `json:"enable_web_ui"`
	WebUIPort          int           `json:"web_ui_port"`
}

// AlertThresholds defines thresholds for cache alerts
type AlertThresholds struct {
	HitRateMin         float64       `json:"hit_rate_min"`           // Minimum acceptable hit rate
	ResponseTimeMax    time.Duration `json:"response_time_max"`      // Maximum acceptable response time
	ErrorRateMax       float64       `json:"error_rate_max"`         // Maximum acceptable error rate
	MemoryUsageMax     int64         `json:"memory_usage_max"`       // Maximum memory usage in bytes
	ConnectionFailures int           `json:"connection_failures"`    // Max connection failures before alert
}

// HealthStatus represents cache health information
type HealthStatus struct {
	Overall     string                 `json:"overall"`      // healthy, degraded, unhealthy
	Timestamp   time.Time              `json:"timestamp"`
	Components  map[string]ComponentHealth `json:"components"`
	Metrics     *DetailedCacheMetrics  `json:"metrics"`
	Alerts      []Alert                `json:"alerts"`
}

// ComponentHealth represents health of individual cache components
type ComponentHealth struct {
	Status      string            `json:"status"`      // healthy, degraded, unhealthy
	Message     string            `json:"message"`
	LastChecked time.Time         `json:"last_checked"`
	Metrics     map[string]interface{} `json:"metrics"`
}

// Alert represents a cache alert
type Alert struct {
	ID          string    `json:"id"`
	Level       string    `json:"level"`        // info, warning, critical
	Component   string    `json:"component"`
	Message     string    `json:"message"`
	Timestamp   time.Time `json:"timestamp"`
	Acknowledged bool     `json:"acknowledged"`
}

// NewCacheMonitor creates a new cache monitor
func NewCacheMonitor(cache Cache, metricsCollector *CacheMetricsCollector, config *MonitorConfig, logger *logrus.Logger) *CacheMonitor {
	if config == nil {
		config = &MonitorConfig{
			HealthCheckInterval: 30 * time.Second,
			MetricsPort:         9091,
			AlertThresholds: &AlertThresholds{
				HitRateMin:         0.85,
				ResponseTimeMax:    100 * time.Millisecond,
				ErrorRateMax:       0.01,
				MemoryUsageMax:     1024 * 1024 * 1024, // 1GB
				ConnectionFailures: 5,
			},
			EnableWebUI: true,
			WebUIPort:   8082,
		}
	}

	if logger == nil {
		logger = logrus.New()
	}

	monitor := &CacheMonitor{
		cache:            cache,
		metricsCollector: metricsCollector,
		logger:           logger,
		config:           config,
	}

	// Start monitoring services
	go monitor.startHealthChecker()
	go monitor.startMetricsServer()
	
	if config.EnableWebUI {
		go monitor.startWebUI()
	}

	logger.WithFields(logrus.Fields{
		"metrics_port": config.MetricsPort,
		"web_ui_port":  config.WebUIPort,
		"web_ui_enabled": config.EnableWebUI,
	}).Info("Cache monitor initialized")

	return monitor
}

// GetHealthStatus returns current health status
func (cm *CacheMonitor) GetHealthStatus() *HealthStatus {
	ctx := context.Background()
	
	status := &HealthStatus{
		Timestamp:  time.Now(),
		Components: make(map[string]ComponentHealth),
		Alerts:     []Alert{},
	}

	// Get detailed metrics
	if cm.metricsCollector != nil {
		status.Metrics = cm.metricsCollector.GetMetrics()
	}

	// Check cache health
	cacheHealth := cm.checkCacheHealth(ctx)
	status.Components["cache"] = cacheHealth

	// Check Redis health (if Redis cache)
	if redisCache, ok := cm.cache.(*RedisCache); ok {
		redisHealth := cm.checkRedisHealth(ctx, redisCache)
		status.Components["redis"] = redisHealth
	}

	// Check multi-tier cache health
	if multiTierCache, ok := cm.cache.(*MultiTierCache); ok {
		l1Health := cm.checkL1Health(ctx, multiTierCache)
		l2Health := cm.checkL2Health(ctx, multiTierCache)
		l3Health := cm.checkL3Health(ctx, multiTierCache)
		
		status.Components["l1_cache"] = l1Health
		status.Components["l2_cache"] = l2Health
		status.Components["l3_cache"] = l3Health
	}

	// Generate alerts
	status.Alerts = cm.generateAlerts(status)

	// Determine overall health
	status.Overall = cm.calculateOverallHealth(status.Components, status.Alerts)

	return status
}

// checkCacheHealth performs basic cache health checks
func (cm *CacheMonitor) checkCacheHealth(ctx context.Context) ComponentHealth {
	health := ComponentHealth{
		LastChecked: time.Now(),
		Metrics:     make(map[string]interface{}),
	}

	// Test basic cache operations
	testKey := fmt.Sprintf("health_check_%d", time.Now().Unix())
	testValue := []byte("health_check_value")
	
	// Test SET operation
	if err := cm.cache.Set(ctx, testKey, testValue, 10*time.Second); err != nil {
		health.Status = "unhealthy"
		health.Message = fmt.Sprintf("Cache SET failed: %v", err)
		return health
	}

	// Test GET operation
	if _, err := cm.cache.Get(ctx, testKey); err != nil {
		health.Status = "unhealthy"
		health.Message = fmt.Sprintf("Cache GET failed: %v", err)
		return health
	}

	// Test DELETE operation
	if err := cm.cache.Delete(ctx, testKey); err != nil {
		cm.logger.WithError(err).Warn("Cache DELETE failed during health check")
	}

	// Get cache stats
	stats := cm.cache.GetStats()
	health.Metrics = map[string]interface{}{
		"hit_rate":           stats.HitRate,
		"total_operations":   stats.Hits + stats.Misses + stats.Sets + stats.Deletes,
		"error_rate":         float64(stats.Errors) / float64(stats.Hits + stats.Misses + stats.Sets + stats.Deletes + stats.Errors),
		"avg_response_time":  stats.AvgResponseTimeNs,
	}

	health.Status = "healthy"
	health.Message = "Cache operations successful"
	return health
}

// checkRedisHealth performs Redis-specific health checks
func (cm *CacheMonitor) checkRedisHealth(ctx context.Context, redisCache *RedisCache) ComponentHealth {
	health := ComponentHealth{
		LastChecked: time.Now(),
		Metrics:     make(map[string]interface{}),
	}

	// Test Redis ping
	if err := redisCache.Ping(ctx); err != nil {
		health.Status = "unhealthy"
		health.Message = fmt.Sprintf("Redis ping failed: %v", err)
		return health
	}

	// Get Redis info
	info, err := redisCache.GetInfo(ctx)
	if err != nil {
		health.Status = "degraded"
		health.Message = fmt.Sprintf("Failed to get Redis info: %v", err)
	} else {
		health.Metrics["redis_info"] = info
		
		// Extract key metrics
		if connectedClients, ok := info["connected_clients"]; ok {
			health.Metrics["connected_clients"] = connectedClients
		}
		if usedMemory, ok := info["used_memory"]; ok {
			health.Metrics["used_memory"] = usedMemory
		}
		if hitRate, ok := info["keyspace_hit_rate"]; ok {
			health.Metrics["keyspace_hit_rate"] = hitRate
		}
	}

	// Check cluster info if in cluster mode
	if redisCache.clusterMode {
		if clusterInfo, err := redisCache.GetClusterInfo(ctx); err == nil {
			health.Metrics["cluster_info"] = clusterInfo
		}
	}

	if health.Status == "" {
		health.Status = "healthy"
		health.Message = "Redis health check passed"
	}

	return health
}

// checkL1Health checks L1 cache health
func (cm *CacheMonitor) checkL1Health(ctx context.Context, multiTierCache *MultiTierCache) ComponentHealth {
	health := ComponentHealth{
		LastChecked: time.Now(),
		Metrics:     make(map[string]interface{}),
	}

	if multiTierCache.l1Cache != nil {
		stats := multiTierCache.l1Cache.GetStats()
		health.Metrics = map[string]interface{}{
			"hits":               stats.Hits,
			"misses":             stats.Misses,
			"hit_rate":           stats.HitRate,
			"avg_response_time":  stats.AvgResponseTimeNs,
		}
		health.Status = "healthy"
		health.Message = "L1 cache operational"
	} else {
		health.Status = "unavailable"
		health.Message = "L1 cache not enabled"
	}

	return health
}

// checkL2Health checks L2 cache health
func (cm *CacheMonitor) checkL2Health(ctx context.Context, multiTierCache *MultiTierCache) ComponentHealth {
	health := ComponentHealth{
		LastChecked: time.Now(),
		Metrics:     make(map[string]interface{}),
	}

	if multiTierCache.l2Cache != nil {
		stats := multiTierCache.l2Cache.GetStats()
		health.Metrics = map[string]interface{}{
			"hits":               stats.Hits,
			"misses":             stats.Misses,
			"hit_rate":           stats.HitRate,
			"avg_response_time":  stats.AvgResponseTimeNs,
		}
		health.Status = "healthy"
		health.Message = "L2 cache operational"
	} else {
		health.Status = "unavailable"
		health.Message = "L2 cache not enabled"
	}

	return health
}

// checkL3Health checks L3 cache health
func (cm *CacheMonitor) checkL3Health(ctx context.Context, multiTierCache *MultiTierCache) ComponentHealth {
	health := ComponentHealth{
		LastChecked: time.Now(),
		Metrics:     make(map[string]interface{}),
	}

	if multiTierCache.l3Cache != nil {
		stats := multiTierCache.l3Cache.GetStats()
		health.Metrics = map[string]interface{}{
			"hits":               stats.Hits,
			"misses":             stats.Misses,
			"hit_rate":           stats.HitRate,
			"avg_response_time":  stats.AvgResponseTimeNs,
		}
		health.Status = "healthy"
		health.Message = "L3 cache operational"
	} else {
		health.Status = "unavailable"
		health.Message = "L3 cache not enabled"
	}

	return health
}

// generateAlerts creates alerts based on thresholds
func (cm *CacheMonitor) generateAlerts(status *HealthStatus) []Alert {
	var alerts []Alert
	thresholds := cm.config.AlertThresholds

	if status.Metrics != nil {
		// Check hit rate
		if status.Metrics.Basic.HitRate < thresholds.HitRateMin {
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("hit_rate_%d", time.Now().Unix()),
				Level:     "warning",
				Component: "cache",
				Message:   fmt.Sprintf("Cache hit rate %.2f%% below threshold %.2f%%", status.Metrics.Basic.HitRate*100, thresholds.HitRateMin*100),
				Timestamp: time.Now(),
			})
		}

		// Check response time
		avgResponseTime := time.Duration(status.Metrics.Basic.AvgResponseTimeNs)
		if avgResponseTime > thresholds.ResponseTimeMax {
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("response_time_%d", time.Now().Unix()),
				Level:     "warning",
				Component: "cache",
				Message:   fmt.Sprintf("Cache response time %v above threshold %v", avgResponseTime, thresholds.ResponseTimeMax),
				Timestamp: time.Now(),
			})
		}

		// Check error rate
		if status.Metrics.ErrorMetrics != nil && status.Metrics.ErrorMetrics.ErrorRate > thresholds.ErrorRateMax {
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("error_rate_%d", time.Now().Unix()),
				Level:     "critical",
				Component: "cache",
				Message:   fmt.Sprintf("Cache error rate %.2f%% above threshold %.2f%%", status.Metrics.ErrorMetrics.ErrorRate*100, thresholds.ErrorRateMax*100),
				Timestamp: time.Now(),
			})
		}
	}

	// Check component health
	for component, health := range status.Components {
		if health.Status == "unhealthy" {
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("%s_unhealthy_%d", component, time.Now().Unix()),
				Level:     "critical",
				Component: component,
				Message:   fmt.Sprintf("Component %s is unhealthy: %s", component, health.Message),
				Timestamp: time.Now(),
			})
		} else if health.Status == "degraded" {
			alerts = append(alerts, Alert{
				ID:        fmt.Sprintf("%s_degraded_%d", component, time.Now().Unix()),
				Level:     "warning",
				Component: component,
				Message:   fmt.Sprintf("Component %s is degraded: %s", component, health.Message),
				Timestamp: time.Now(),
			})
		}
	}

	return alerts
}

// calculateOverallHealth determines overall system health
func (cm *CacheMonitor) calculateOverallHealth(components map[string]ComponentHealth, alerts []Alert) string {
	criticalAlerts := 0
	warningAlerts := 0
	unhealthyComponents := 0
	degradedComponents := 0

	for _, alert := range alerts {
		if alert.Level == "critical" {
			criticalAlerts++
		} else if alert.Level == "warning" {
			warningAlerts++
		}
	}

	for _, health := range components {
		if health.Status == "unhealthy" {
			unhealthyComponents++
		} else if health.Status == "degraded" {
			degradedComponents++
		}
	}

	if criticalAlerts > 0 || unhealthyComponents > 0 {
		return "unhealthy"
	}

	if warningAlerts > 0 || degradedComponents > 0 {
		return "degraded"
	}

	return "healthy"
}

// startHealthChecker runs periodic health checks
func (cm *CacheMonitor) startHealthChecker() {
	ticker := time.NewTicker(cm.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			health := cm.GetHealthStatus()
			
			// Log health status
			cm.logger.WithFields(logrus.Fields{
				"overall_health":   health.Overall,
				"components_count": len(health.Components),
				"alerts_count":     len(health.Alerts),
			}).Info("Cache health check completed")

			// Log critical alerts
			for _, alert := range health.Alerts {
				if alert.Level == "critical" {
					cm.logger.WithFields(logrus.Fields{
						"alert_id":   alert.ID,
						"component":  alert.Component,
						"message":    alert.Message,
					}).Error("Critical cache alert")
				}
			}
		}
	}
}

// startMetricsServer starts HTTP server for metrics endpoint
func (cm *CacheMonitor) startMetricsServer() {
	mux := http.NewServeMux()
	
	// Prometheus metrics endpoint
	mux.HandleFunc("/metrics", cm.handlePrometheusMetrics)
	
	// Health check endpoint
	mux.HandleFunc("/health", cm.handleHealthCheck)
	
	// Detailed metrics endpoint
	mux.HandleFunc("/api/metrics", cm.handleDetailedMetrics)
	
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", cm.config.MetricsPort),
		Handler: mux,
	}

	cm.logger.WithField("port", cm.config.MetricsPort).Info("Starting cache metrics server")
	
	if err := server.ListenAndServe(); err != nil {
		cm.logger.WithError(err).Error("Metrics server failed")
	}
}

// startWebUI starts web UI server
func (cm *CacheMonitor) startWebUI() {
	mux := http.NewServeMux()
	
	// Serve static files (would need to be implemented)
	mux.HandleFunc("/", cm.handleWebUI)
	mux.HandleFunc("/api/status", cm.handleStatusAPI)
	
	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", cm.config.WebUIPort),
		Handler: mux,
	}

	cm.logger.WithField("port", cm.config.WebUIPort).Info("Starting cache monitoring web UI")
	
	if err := server.ListenAndServe(); err != nil {
		cm.logger.WithError(err).Error("Web UI server failed")
	}
}

// HTTP handlers

func (cm *CacheMonitor) handlePrometheusMetrics(w http.ResponseWriter, r *http.Request) {
	if cm.metricsCollector != nil {
		metrics := cm.metricsCollector.ExportPrometheusMetrics()
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte(metrics))
	} else {
		http.Error(w, "Metrics collector not available", http.StatusServiceUnavailable)
	}
}

func (cm *CacheMonitor) handleHealthCheck(w http.ResponseWriter, r *http.Request) {
	health := cm.GetHealthStatus()
	
	w.Header().Set("Content-Type", "application/json")
	
	if health.Overall == "unhealthy" {
		w.WriteHeader(http.StatusServiceUnavailable)
	} else if health.Overall == "degraded" {
		w.WriteHeader(http.StatusPartialContent)
	} else {
		w.WriteHeader(http.StatusOK)
	}
	
	json.NewEncoder(w).Encode(health)
}

func (cm *CacheMonitor) handleDetailedMetrics(w http.ResponseWriter, r *http.Request) {
	if cm.metricsCollector != nil {
		metrics := cm.metricsCollector.GetMetrics()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(metrics)
	} else {
		http.Error(w, "Metrics collector not available", http.StatusServiceUnavailable)
	}
}

func (cm *CacheMonitor) handleWebUI(w http.ResponseWriter, r *http.Request) {
	// Simple HTML dashboard (in production, this would serve static files)
	html := `
<!DOCTYPE html>
<html>
<head>
    <title>NovaCron Cache Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .healthy { background-color: #d4edda; border: 1px solid #c3e6cb; }
        .degraded { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .unhealthy { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .metrics { margin: 20px 0; }
        .metric { display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ccc; }
    </style>
    <script>
        function refreshStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').innerHTML = 
                        '<div class="status ' + data.overall + '">Overall Status: ' + data.overall + '</div>';
                    document.getElementById('timestamp').innerHTML = 'Last Updated: ' + data.timestamp;
                });
        }
        setInterval(refreshStatus, 30000);
        window.onload = refreshStatus;
    </script>
</head>
<body>
    <h1>NovaCron Cache Monitor</h1>
    <div id="status"></div>
    <div id="timestamp"></div>
    <div class="metrics">
        <h2>Quick Links</h2>
        <a href="/health">Health Check JSON</a> |
        <a href="/metrics">Prometheus Metrics</a> |
        <a href="/api/metrics">Detailed Metrics JSON</a>
    </div>
</body>
</html>`
	
	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(html))
}

func (cm *CacheMonitor) handleStatusAPI(w http.ResponseWriter, r *http.Request) {
	health := cm.GetHealthStatus()
	
	// Return simplified status for web UI
	status := map[string]interface{}{
		"overall":   health.Overall,
		"timestamp": health.Timestamp.Format(time.RFC3339),
		"alerts":    len(health.Alerts),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(status)
}