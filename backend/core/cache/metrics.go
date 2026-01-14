package cache

import (
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// CacheMetricsCollector collects and aggregates cache performance metrics
type CacheMetricsCollector struct {
	cache         Cache
	logger        *logrus.Logger
	metrics       *DetailedCacheMetrics
	mutex         sync.RWMutex
	config        *MetricsConfig
	stopChan      chan struct{}
	histogramData map[string]*HistogramData
}

// MetricsConfig holds metrics collection configuration
type MetricsConfig struct {
	CollectionInterval time.Duration `json:"collection_interval"`
	RetentionDays      int           `json:"retention_days"`
	EnableHistograms   bool          `json:"enable_histograms"`
	HistogramBuckets   []float64     `json:"histogram_buckets"` // Response time buckets in milliseconds
	EnableHeatmap      bool          `json:"enable_heatmap"`
	HeatmapResolution  time.Duration `json:"heatmap_resolution"`
}

// DetailedCacheMetrics provides comprehensive cache performance data
type DetailedCacheMetrics struct {
	Basic              CacheStats                    `json:"basic"`
	ResponseTimes      *ResponseTimeMetrics          `json:"response_times"`
	ThroughputMetrics  *ThroughputMetrics            `json:"throughput_metrics"`
	ErrorMetrics       *ErrorMetrics                 `json:"error_metrics"`
	PatternMetrics     map[string]*PatternMetrics    `json:"pattern_metrics"`
	TierMetrics        map[string]*TierMetrics       `json:"tier_metrics"`
	HistogramData      map[string]*HistogramData     `json:"histogram_data"`
	HeatmapData        *HeatmapData                  `json:"heatmap_data"`
	LastCollection     time.Time                     `json:"last_collection"`
}

// ResponseTimeMetrics tracks response time statistics
type ResponseTimeMetrics struct {
	P50         time.Duration `json:"p50"`
	P95         time.Duration `json:"p95"`
	P99         time.Duration `json:"p99"`
	Min         time.Duration `json:"min"`
	Max         time.Duration `json:"max"`
	Mean        time.Duration `json:"mean"`
	StdDev      time.Duration `json:"std_dev"`
	SampleCount uint64        `json:"sample_count"`
}

// ThroughputMetrics tracks throughput statistics
type ThroughputMetrics struct {
	RequestsPerSecond    float64   `json:"requests_per_second"`
	BytesPerSecond       float64   `json:"bytes_per_second"`
	PeakRequestsPerSec   float64   `json:"peak_requests_per_sec"`
	PeakBytesPerSec      float64   `json:"peak_bytes_per_sec"`
	LastMinuteRequests   uint64    `json:"last_minute_requests"`
	LastMinuteBytes      uint64    `json:"last_minute_bytes"`
	MeasurementPeriod    time.Time `json:"measurement_period"`
}

// ErrorMetrics tracks error statistics
type ErrorMetrics struct {
	TotalErrors        uint64            `json:"total_errors"`
	ErrorRate          float64           `json:"error_rate"`
	ErrorsByType       map[string]uint64 `json:"errors_by_type"`
	TimeoutErrors      uint64            `json:"timeout_errors"`
	ConnectionErrors   uint64            `json:"connection_errors"`
	SerializationErrors uint64           `json:"serialization_errors"`
	LastError          string            `json:"last_error"`
	LastErrorTime      time.Time         `json:"last_error_time"`
}

// PatternMetrics tracks metrics for specific key patterns
type PatternMetrics struct {
	Pattern     string    `json:"pattern"`
	Hits        uint64    `json:"hits"`
	Misses      uint64    `json:"misses"`
	Sets        uint64    `json:"sets"`
	Deletes     uint64    `json:"deletes"`
	HitRate     float64   `json:"hit_rate"`
	LastAccess  time.Time `json:"last_access"`
}

// TierMetrics tracks metrics for each cache tier
type TierMetrics struct {
	TierName         string        `json:"tier_name"`
	Hits             uint64        `json:"hits"`
	Misses           uint64        `json:"misses"`
	Sets             uint64        `json:"sets"`
	Deletes          uint64        `json:"deletes"`
	Errors           uint64        `json:"errors"`
	HitRate          float64       `json:"hit_rate"`
	AvgResponseTime  time.Duration `json:"avg_response_time"`
	TotalMemoryUsage int64         `json:"total_memory_usage"`
	ItemCount        int64         `json:"item_count"`
}

// HistogramData tracks response time distribution
type HistogramData struct {
	Buckets   []float64 `json:"buckets"`   // Bucket boundaries in milliseconds
	Counts    []uint64  `json:"counts"`    // Count for each bucket
	TotalSamples uint64 `json:"total_samples"`
	LastUpdate   time.Time `json:"last_update"`
}

// HeatmapData provides time-series heatmap data
type HeatmapData struct {
	TimeSlots    []time.Time `json:"time_slots"`
	ResponseTimes [][]float64 `json:"response_times"` // 2D array: [time][percentile]
	Resolution   time.Duration `json:"resolution"`
	MaxRetention time.Duration `json:"max_retention"`
}

// NewCacheMetricsCollector creates a new metrics collector
func NewCacheMetricsCollector(cache Cache, config *MetricsConfig, logger *logrus.Logger) *CacheMetricsCollector {
	if config == nil {
		config = &MetricsConfig{
			CollectionInterval: 10 * time.Second,
			RetentionDays:      7,
			EnableHistograms:   true,
			HistogramBuckets:   []float64{0.1, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 500, 1000}, // milliseconds
			EnableHeatmap:      true,
			HeatmapResolution:  1 * time.Minute,
		}
	}

	if logger == nil {
		logger = logrus.New()
	}

	cmc := &CacheMetricsCollector{
		cache:    cache,
		logger:   logger,
		config:   config,
		stopChan: make(chan struct{}),
		metrics: &DetailedCacheMetrics{
			PatternMetrics: make(map[string]*PatternMetrics),
			TierMetrics:    make(map[string]*TierMetrics),
			HistogramData:  make(map[string]*HistogramData),
			HeatmapData: &HeatmapData{
				TimeSlots:    make([]time.Time, 0),
				ResponseTimes: make([][]float64, 0),
				Resolution:   config.HeatmapResolution,
				MaxRetention: time.Duration(config.RetentionDays) * 24 * time.Hour,
			},
		},
		histogramData: make(map[string]*HistogramData),
	}

	// Initialize histogram data for different operation types
	if config.EnableHistograms {
		for _, operation := range []string{"get", "set", "delete", "exists"} {
			cmc.histogramData[operation] = &HistogramData{
				Buckets: config.HistogramBuckets,
				Counts:  make([]uint64, len(config.HistogramBuckets)),
			}
		}
	}

	// Start metrics collection
	go cmc.startCollection()

	logger.Info("Cache metrics collector initialized")
	return cmc
}

// startCollection begins the metrics collection loop
func (cmc *CacheMetricsCollector) startCollection() {
	ticker := time.NewTicker(cmc.config.CollectionInterval)
	defer ticker.Stop()

	for {
		select {
		case <-cmc.stopChan:
			return
		case <-ticker.C:
			cmc.collectMetrics()
		}
	}
}

// collectMetrics gathers current metrics from the cache
func (cmc *CacheMetricsCollector) collectMetrics() {
	cmc.mutex.Lock()
	defer cmc.mutex.Unlock()

	// Get basic cache stats
	basicStats := cmc.cache.GetStats()
	cmc.metrics.Basic = basicStats

	// Update tier metrics if available
	cmc.updateTierMetrics()

	// Update pattern metrics
	cmc.updatePatternMetrics()

	// Update throughput metrics
	cmc.updateThroughputMetrics()

	// Update response time metrics
	cmc.updateResponseTimeMetrics()

	// Update error metrics
	cmc.updateErrorMetrics()

	// Update heatmap data
	if cmc.config.EnableHeatmap {
		cmc.updateHeatmapData()
	}

	cmc.metrics.LastCollection = time.Now()

	// Log metrics periodically
	if time.Now().Unix()%60 == 0 { // Every minute
		cmc.logMetricsSummary()
	}
}

// updateTierMetrics updates metrics for each cache tier
func (cmc *CacheMetricsCollector) updateTierMetrics() {
	// This would be implemented based on the specific cache implementation
	// For multi-tier cache, we can get metrics from each tier
	
	if multiTierCache, ok := cmc.cache.(*MultiTierCache); ok {
		// L1 Cache metrics
		if multiTierCache.l1Cache != nil {
			l1Stats := multiTierCache.l1Cache.GetStats()
			cmc.metrics.TierMetrics["l1"] = &TierMetrics{
				TierName:        "L1 (Memory)",
				Hits:            l1Stats.Hits,
				Misses:          l1Stats.Misses,
				Sets:            l1Stats.Sets,
				Deletes:         l1Stats.Deletes,
				Errors:          l1Stats.Errors,
				HitRate:         l1Stats.HitRate,
				AvgResponseTime: time.Duration(l1Stats.AvgResponseTimeNs),
			}
		}

		// L2 Cache metrics
		if multiTierCache.l2Cache != nil {
			l2Stats := multiTierCache.l2Cache.GetStats()
			cmc.metrics.TierMetrics["l2"] = &TierMetrics{
				TierName:        "L2 (Redis)",
				Hits:            l2Stats.Hits,
				Misses:          l2Stats.Misses,
				Sets:            l2Stats.Sets,
				Deletes:         l2Stats.Deletes,
				Errors:          l2Stats.Errors,
				HitRate:         l2Stats.HitRate,
				AvgResponseTime: time.Duration(l2Stats.AvgResponseTimeNs),
			}
		}

		// L3 Cache metrics
		if multiTierCache.l3Cache != nil {
			l3Stats := multiTierCache.l3Cache.GetStats()
			cmc.metrics.TierMetrics["l3"] = &TierMetrics{
				TierName:        "L3 (Persistent)",
				Hits:            l3Stats.Hits,
				Misses:          l3Stats.Misses,
				Sets:            l3Stats.Sets,
				Deletes:         l3Stats.Deletes,
				Errors:          l3Stats.Errors,
				HitRate:         l3Stats.HitRate,
				AvgResponseTime: time.Duration(l3Stats.AvgResponseTimeNs),
			}
		}
	}
}

// updatePatternMetrics updates metrics for specific key patterns
func (cmc *CacheMetricsCollector) updatePatternMetrics() {
	// This would analyze cache keys to identify patterns
	patterns := []string{
		"vm:state:",
		"vm:resources:",
		"vm:migration:",
		"vm:metrics:",
		"vm:config:",
		"node:resources:",
		"cluster:",
	}

	for _, pattern := range patterns {
		if _, exists := cmc.metrics.PatternMetrics[pattern]; !exists {
			cmc.metrics.PatternMetrics[pattern] = &PatternMetrics{
				Pattern: pattern,
			}
		}
		// In a real implementation, you would query the cache for pattern-specific stats
	}
}

// updateThroughputMetrics calculates throughput statistics
func (cmc *CacheMetricsCollector) updateThroughputMetrics() {
	now := time.Now()
	basic := cmc.metrics.Basic

	if cmc.metrics.ThroughputMetrics == nil {
		cmc.metrics.ThroughputMetrics = &ThroughputMetrics{
			MeasurementPeriod: now,
		}
		return
	}

	timeDelta := now.Sub(cmc.metrics.ThroughputMetrics.MeasurementPeriod).Seconds()
	if timeDelta > 0 {
		totalRequests := basic.Hits + basic.Misses
		requestsPerSecond := float64(totalRequests) / timeDelta

		cmc.metrics.ThroughputMetrics.RequestsPerSecond = requestsPerSecond
		if requestsPerSecond > cmc.metrics.ThroughputMetrics.PeakRequestsPerSec {
			cmc.metrics.ThroughputMetrics.PeakRequestsPerSec = requestsPerSecond
		}

		cmc.metrics.ThroughputMetrics.MeasurementPeriod = now
	}
}

// updateResponseTimeMetrics updates response time statistics
func (cmc *CacheMetricsCollector) updateResponseTimeMetrics() {
	avgResponseTime := time.Duration(cmc.metrics.Basic.AvgResponseTimeNs)
	
	if cmc.metrics.ResponseTimes == nil {
		cmc.metrics.ResponseTimes = &ResponseTimeMetrics{
			Min:  avgResponseTime,
			Max:  avgResponseTime,
			Mean: avgResponseTime,
		}
		return
	}

	// Update min/max
	if avgResponseTime < cmc.metrics.ResponseTimes.Min {
		cmc.metrics.ResponseTimes.Min = avgResponseTime
	}
	if avgResponseTime > cmc.metrics.ResponseTimes.Max {
		cmc.metrics.ResponseTimes.Max = avgResponseTime
	}

	// Simple running average update
	cmc.metrics.ResponseTimes.Mean = avgResponseTime
	cmc.metrics.ResponseTimes.SampleCount++
}

// updateErrorMetrics updates error statistics
func (cmc *CacheMetricsCollector) updateErrorMetrics() {
	if cmc.metrics.ErrorMetrics == nil {
		cmc.metrics.ErrorMetrics = &ErrorMetrics{
			ErrorsByType: make(map[string]uint64),
		}
	}

	totalOps := cmc.metrics.Basic.Hits + cmc.metrics.Basic.Misses + cmc.metrics.Basic.Sets + cmc.metrics.Basic.Deletes
	if totalOps > 0 {
		cmc.metrics.ErrorMetrics.ErrorRate = float64(cmc.metrics.Basic.Errors) / float64(totalOps)
	}
	cmc.metrics.ErrorMetrics.TotalErrors = cmc.metrics.Basic.Errors
}

// updateHeatmapData updates the time-series heatmap data
func (cmc *CacheMetricsCollector) updateHeatmapData() {
	now := time.Now()
	heatmap := cmc.metrics.HeatmapData

	// Add new time slot
	heatmap.TimeSlots = append(heatmap.TimeSlots, now)
	
	// Add response time percentiles (mock data for now)
	responseTimePercentiles := []float64{
		float64(cmc.metrics.ResponseTimes.P50.Nanoseconds()) / 1000000, // Convert to ms
		float64(cmc.metrics.ResponseTimes.P95.Nanoseconds()) / 1000000,
		float64(cmc.metrics.ResponseTimes.P99.Nanoseconds()) / 1000000,
	}
	
	heatmap.ResponseTimes = append(heatmap.ResponseTimes, responseTimePercentiles)

	// Trim old data based on retention policy
	maxSlots := int(heatmap.MaxRetention / heatmap.Resolution)
	if len(heatmap.TimeSlots) > maxSlots {
		excess := len(heatmap.TimeSlots) - maxSlots
		heatmap.TimeSlots = heatmap.TimeSlots[excess:]
		heatmap.ResponseTimes = heatmap.ResponseTimes[excess:]
	}
}

// RecordOperation records metrics for a specific cache operation
func (cmc *CacheMetricsCollector) RecordOperation(operation string, duration time.Duration, success bool, keyPattern string) {
	cmc.mutex.Lock()
	defer cmc.mutex.Unlock()

	// Update histogram data
	if cmc.config.EnableHistograms {
		if histogramData, exists := cmc.histogramData[operation]; exists {
			durationMs := float64(duration.Nanoseconds()) / 1000000
			
			// Find appropriate bucket
			for i, bucket := range histogramData.Buckets {
				if durationMs <= bucket {
					histogramData.Counts[i]++
					break
				}
			}
			histogramData.TotalSamples++
			histogramData.LastUpdate = time.Now()
		}
	}

	// Update pattern metrics
	if keyPattern != "" {
		if patternMetric, exists := cmc.metrics.PatternMetrics[keyPattern]; exists {
			patternMetric.LastAccess = time.Now()
			switch operation {
			case "get":
				if success {
					patternMetric.Hits++
				} else {
					patternMetric.Misses++
				}
			case "set":
				patternMetric.Sets++
			case "delete":
				patternMetric.Deletes++
			}
			
			total := patternMetric.Hits + patternMetric.Misses
			if total > 0 {
				patternMetric.HitRate = float64(patternMetric.Hits) / float64(total)
			}
		}
	}
}

// GetMetrics returns current detailed metrics
func (cmc *CacheMetricsCollector) GetMetrics() *DetailedCacheMetrics {
	cmc.mutex.RLock()
	defer cmc.mutex.RUnlock()

	// Create a deep copy to avoid race conditions
	metricsCopy := *cmc.metrics
	return &metricsCopy
}

// GetMetricsJSON returns metrics as JSON
func (cmc *CacheMetricsCollector) GetMetricsJSON() ([]byte, error) {
	metrics := cmc.GetMetrics()
	return json.MarshalIndent(metrics, "", "  ")
}

// GetSummary returns a summary of key metrics
func (cmc *CacheMetricsCollector) GetSummary() map[string]interface{} {
	metrics := cmc.GetMetrics()
	
	return map[string]interface{}{
		"hit_rate":           metrics.Basic.HitRate,
		"avg_response_time":  fmt.Sprintf("%.2fms", float64(metrics.Basic.AvgResponseTimeNs)/1000000),
		"requests_per_sec":   metrics.ThroughputMetrics.RequestsPerSecond,
		"error_rate":         metrics.ErrorMetrics.ErrorRate,
		"total_operations":   metrics.Basic.Hits + metrics.Basic.Misses + metrics.Basic.Sets + metrics.Basic.Deletes,
		"l1_hit_rate":        func() float64 {
			if l1, exists := metrics.TierMetrics["l1"]; exists {
				return l1.HitRate
			}
			return 0
		}(),
		"l2_hit_rate": func() float64 {
			if l2, exists := metrics.TierMetrics["l2"]; exists {
				return l2.HitRate
			}
			return 0
		}(),
		"last_updated": metrics.LastCollection.Format(time.RFC3339),
	}
}

// logMetricsSummary logs a summary of metrics
func (cmc *CacheMetricsCollector) logMetricsSummary() {
	summary := cmc.GetSummary()
	cmc.logger.WithFields(logrus.Fields{
		"hit_rate":          summary["hit_rate"],
		"avg_response_time": summary["avg_response_time"],
		"requests_per_sec":  summary["requests_per_sec"],
		"error_rate":        summary["error_rate"],
	}).Info("Cache metrics summary")
}

// Stop stops the metrics collector
func (cmc *CacheMetricsCollector) Stop() {
	close(cmc.stopChan)
	cmc.logger.Info("Cache metrics collector stopped")
}

// ExportPrometheusMetrics returns metrics in Prometheus format
func (cmc *CacheMetricsCollector) ExportPrometheusMetrics() string {
	metrics := cmc.GetMetrics()
	
	return fmt.Sprintf(`# HELP novacron_cache_hit_rate Cache hit rate
# TYPE novacron_cache_hit_rate gauge
novacron_cache_hit_rate %.4f

# HELP novacron_cache_operations_total Total cache operations
# TYPE novacron_cache_operations_total counter
novacron_cache_operations_total{operation="hit"} %d
novacron_cache_operations_total{operation="miss"} %d
novacron_cache_operations_total{operation="set"} %d
novacron_cache_operations_total{operation="delete"} %d

# HELP novacron_cache_response_time_nanoseconds Cache response time
# TYPE novacron_cache_response_time_nanoseconds gauge
novacron_cache_response_time_nanoseconds %d

# HELP novacron_cache_errors_total Total cache errors
# TYPE novacron_cache_errors_total counter
novacron_cache_errors_total %d

# HELP novacron_cache_requests_per_second Cache requests per second
# TYPE novacron_cache_requests_per_second gauge
novacron_cache_requests_per_second %.2f
`,
		metrics.Basic.HitRate,
		metrics.Basic.Hits,
		metrics.Basic.Misses,
		metrics.Basic.Sets,
		metrics.Basic.Deletes,
		metrics.Basic.AvgResponseTimeNs,
		metrics.Basic.Errors,
		metrics.ThroughputMetrics.RequestsPerSecond,
	)
}