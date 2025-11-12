package monitoring

import (
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// PerformanceTracker compares v1 and v3 performance side-by-side
type PerformanceTracker struct {
	mu sync.RWMutex

	// Version tracking
	v1Metrics *VersionMetrics
	v3Metrics *VersionMetrics

	// Feature flag rollout tracking
	rolloutPercentage float64 // 0-100%
	rolloutHistory    []*RolloutSnapshot

	// Regression detection
	regressionThreshold float64 // percentage degradation to flag
	regressions         []*RegressionAlert

	logger *zap.Logger
}

// VersionMetrics tracks performance for a specific version
type VersionMetrics struct {
	mu sync.RWMutex

	Version string

	// Transfer metrics
	TotalTransfers       int64
	SuccessfulTransfers  int64
	FailedTransfers      int64
	TotalBytesTransferred uint64
	AverageThroughputMbps float64
	P50LatencyMs          float64
	P95LatencyMs          float64
	P99LatencyMs          float64

	// Resource utilization
	AvgCPUUsage     float64 // percentage
	AvgMemoryUsageMB float64
	AvgNetworkIOps  float64

	// Compression and bandwidth savings
	CompressionRatio    float64
	BandwidthSavedGB    float64
	BandwidthSavedPercent float64

	// Latency buckets for percentile calculation
	latencyHistogram []float64
	maxHistogramSize int
}

// RolloutSnapshot captures metrics at a specific rollout percentage
type RolloutSnapshot struct {
	Timestamp         time.Time
	RolloutPercentage float64
	V1Metrics         *VersionMetrics
	V3Metrics         *VersionMetrics
	Comparison        *ComparisonResult
}

// ComparisonResult compares v1 vs v3 performance
type ComparisonResult struct {
	ThroughputImprovement   float64 // percentage
	LatencyImprovement      float64 // percentage
	CPUImprovement          float64 // percentage
	MemoryImprovement       float64 // percentage
	BandwidthSavings        float64 // percentage
	V3FasterThanV1          bool
	SignificantImprovement  bool
}

// RegressionAlert indicates potential performance degradation
type RegressionAlert struct {
	Timestamp   time.Time
	Component   string
	Metric      string
	V1Value     float64
	V3Value     float64
	Degradation float64 // percentage
	Severity    string  // "low", "medium", "high", "critical"
}

// NewPerformanceTracker creates a new performance tracker
func NewPerformanceTracker(logger *zap.Logger) *PerformanceTracker {
	return &PerformanceTracker{
		v1Metrics: &VersionMetrics{
			Version:          "v1",
			latencyHistogram: make([]float64, 0, 10000),
			maxHistogramSize: 10000,
		},
		v3Metrics: &VersionMetrics{
			Version:          "v3",
			latencyHistogram: make([]float64, 0, 10000),
			maxHistogramSize: 10000,
		},
		rolloutPercentage:   0.0,
		rolloutHistory:      make([]*RolloutSnapshot, 0),
		regressions:         make([]*RegressionAlert, 0),
		regressionThreshold: 10.0, // 10% degradation triggers alert
		logger:              logger,
	}
}

// RecordV1Transfer records a v1 transfer event
func (pt *PerformanceTracker) RecordV1Transfer(success bool, latencyMs float64, throughputMbps float64, bytes uint64) {
	pt.v1Metrics.mu.Lock()
	defer pt.v1Metrics.mu.Unlock()

	pt.v1Metrics.TotalTransfers++
	if success {
		pt.v1Metrics.SuccessfulTransfers++
	} else {
		pt.v1Metrics.FailedTransfers++
	}

	pt.v1Metrics.TotalBytesTransferred += bytes

	// Update latency histogram
	if len(pt.v1Metrics.latencyHistogram) < pt.v1Metrics.maxHistogramSize {
		pt.v1Metrics.latencyHistogram = append(pt.v1Metrics.latencyHistogram, latencyMs)
	}

	// Update average throughput (exponential moving average)
	if pt.v1Metrics.AverageThroughputMbps == 0 {
		pt.v1Metrics.AverageThroughputMbps = throughputMbps
	} else {
		pt.v1Metrics.AverageThroughputMbps = pt.v1Metrics.AverageThroughputMbps*0.9 + throughputMbps*0.1
	}

	// Recalculate percentiles
	pt.calculatePercentilesV1()
}

// RecordV3Transfer records a v3 transfer event
func (pt *PerformanceTracker) RecordV3Transfer(success bool, latencyMs float64, throughputMbps float64, bytes uint64) {
	pt.v3Metrics.mu.Lock()
	defer pt.v3Metrics.mu.Unlock()

	pt.v3Metrics.TotalTransfers++
	if success {
		pt.v3Metrics.SuccessfulTransfers++
	} else {
		pt.v3Metrics.FailedTransfers++
	}

	pt.v3Metrics.TotalBytesTransferred += bytes

	// Update latency histogram
	if len(pt.v3Metrics.latencyHistogram) < pt.v3Metrics.maxHistogramSize {
		pt.v3Metrics.latencyHistogram = append(pt.v3Metrics.latencyHistogram, latencyMs)
	}

	// Update average throughput (exponential moving average)
	if pt.v3Metrics.AverageThroughputMbps == 0 {
		pt.v3Metrics.AverageThroughputMbps = throughputMbps
	} else {
		pt.v3Metrics.AverageThroughputMbps = pt.v3Metrics.AverageThroughputMbps*0.9 + throughputMbps*0.1
	}

	// Recalculate percentiles
	pt.calculatePercentilesV3()
}

// SetRolloutPercentage updates the current rollout percentage
func (pt *PerformanceTracker) SetRolloutPercentage(percentage float64) {
	pt.mu.Lock()
	defer pt.mu.Unlock()

	oldPercentage := pt.rolloutPercentage
	pt.rolloutPercentage = percentage

	// Create snapshot at key rollout milestones
	if pt.shouldSnapshot(oldPercentage, percentage) {
		snapshot := pt.createSnapshot()
		pt.rolloutHistory = append(pt.rolloutHistory, snapshot)

		pt.logger.Info("Rollout snapshot created",
			zap.Float64("percentage", percentage),
			zap.Bool("v3_faster", snapshot.Comparison.V3FasterThanV1))
	}

	// Check for regressions
	pt.detectRegressions()
}

// shouldSnapshot determines if we should create a snapshot
func (pt *PerformanceTracker) shouldSnapshot(oldPct, newPct float64) bool {
	milestones := []float64{0, 10, 25, 50, 75, 100}
	for _, milestone := range milestones {
		if oldPct < milestone && newPct >= milestone {
			return true
		}
	}
	return false
}

// createSnapshot creates a performance snapshot
func (pt *PerformanceTracker) createSnapshot() *RolloutSnapshot {
	comparison := pt.compareVersions()

	return &RolloutSnapshot{
		Timestamp:         time.Now(),
		RolloutPercentage: pt.rolloutPercentage,
		V1Metrics:         pt.copyV1Metrics(),
		V3Metrics:         pt.copyV3Metrics(),
		Comparison:        comparison,
	}
}

// compareVersions compares v1 and v3 performance
func (pt *PerformanceTracker) compareVersions() *ComparisonResult {
	pt.v1Metrics.mu.RLock()
	v1Throughput := pt.v1Metrics.AverageThroughputMbps
	v1Latency := pt.v1Metrics.P95LatencyMs
	v1CPU := pt.v1Metrics.AvgCPUUsage
	v1Memory := pt.v1Metrics.AvgMemoryUsageMB
	pt.v1Metrics.mu.RUnlock()

	pt.v3Metrics.mu.RLock()
	v3Throughput := pt.v3Metrics.AverageThroughputMbps
	v3Latency := pt.v3Metrics.P95LatencyMs
	v3CPU := pt.v3Metrics.AvgCPUUsage
	v3Memory := pt.v3Metrics.AvgMemoryUsageMB
	v3BandwidthSaved := pt.v3Metrics.BandwidthSavedPercent
	pt.v3Metrics.mu.RUnlock()

	// Calculate improvements (positive = v3 better)
	throughputImprovement := 0.0
	if v1Throughput > 0 {
		throughputImprovement = ((v3Throughput - v1Throughput) / v1Throughput) * 100.0
	}

	latencyImprovement := 0.0
	if v1Latency > 0 {
		latencyImprovement = ((v1Latency - v3Latency) / v1Latency) * 100.0 // Lower is better
	}

	cpuImprovement := 0.0
	if v1CPU > 0 {
		cpuImprovement = ((v1CPU - v3CPU) / v1CPU) * 100.0 // Lower is better
	}

	memoryImprovement := 0.0
	if v1Memory > 0 {
		memoryImprovement = ((v1Memory - v3Memory) / v1Memory) * 100.0 // Lower is better
	}

	// v3 is faster if throughput improved OR latency improved
	v3Faster := throughputImprovement > 0 || latencyImprovement > 0

	// Significant if any metric improved by >15%
	significant := throughputImprovement > 15 || latencyImprovement > 15 ||
	              cpuImprovement > 15 || memoryImprovement > 15

	return &ComparisonResult{
		ThroughputImprovement:  throughputImprovement,
		LatencyImprovement:     latencyImprovement,
		CPUImprovement:         cpuImprovement,
		MemoryImprovement:      memoryImprovement,
		BandwidthSavings:       v3BandwidthSaved,
		V3FasterThanV1:         v3Faster,
		SignificantImprovement: significant,
	}
}

// detectRegressions checks for performance degradations
func (pt *PerformanceTracker) detectRegressions() {
	comparison := pt.compareVersions()

	// Check throughput regression
	if comparison.ThroughputImprovement < -pt.regressionThreshold {
		pt.addRegression("throughput", comparison.ThroughputImprovement)
	}

	// Check latency regression
	if comparison.LatencyImprovement < -pt.regressionThreshold {
		pt.addRegression("latency", comparison.LatencyImprovement)
	}

	// Check CPU regression
	if comparison.CPUImprovement < -pt.regressionThreshold {
		pt.addRegression("cpu", comparison.CPUImprovement)
	}

	// Check memory regression
	if comparison.MemoryImprovement < -pt.regressionThreshold {
		pt.addRegression("memory", comparison.MemoryImprovement)
	}
}

// addRegression adds a regression alert
func (pt *PerformanceTracker) addRegression(metric string, degradation float64) {
	severity := "low"
	if degradation < -30 {
		severity = "critical"
	} else if degradation < -20 {
		severity = "high"
	} else if degradation < -15 {
		severity = "medium"
	}

	alert := &RegressionAlert{
		Timestamp:   time.Now(),
		Component:   "dwcp_v3",
		Metric:      metric,
		Degradation: -degradation, // Make positive for display
		Severity:    severity,
	}

	pt.regressions = append(pt.regressions, alert)

	pt.logger.Warn("Performance regression detected",
		zap.String("metric", metric),
		zap.Float64("degradation_percent", -degradation),
		zap.String("severity", severity))
}

// GetComparisonReport returns a comprehensive comparison report
func (pt *PerformanceTracker) GetComparisonReport() map[string]interface{} {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	comparison := pt.compareVersions()

	report := map[string]interface{}{
		"rollout_percentage": pt.rolloutPercentage,
		"comparison": map[string]interface{}{
			"throughput_improvement_percent": comparison.ThroughputImprovement,
			"latency_improvement_percent":    comparison.LatencyImprovement,
			"cpu_improvement_percent":        comparison.CPUImprovement,
			"memory_improvement_percent":     comparison.MemoryImprovement,
			"bandwidth_savings_percent":      comparison.BandwidthSavings,
			"v3_faster_than_v1":              comparison.V3FasterThanV1,
			"significant_improvement":        comparison.SignificantImprovement,
		},
		"v1_metrics": pt.getV1MetricsSummary(),
		"v3_metrics": pt.getV3MetricsSummary(),
		"regressions": pt.getRegressionsSummary(),
		"rollout_history": len(pt.rolloutHistory),
	}

	return report
}

// GetRegressions returns active regressions
func (pt *PerformanceTracker) GetRegressions() []*RegressionAlert {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	// Return copy
	regressions := make([]*RegressionAlert, len(pt.regressions))
	copy(regressions, pt.regressions)
	return regressions
}

// ClearRegressions clears regression alerts
func (pt *PerformanceTracker) ClearRegressions() {
	pt.mu.Lock()
	defer pt.mu.Unlock()
	pt.regressions = make([]*RegressionAlert, 0)
	pt.logger.Info("Regression alerts cleared")
}

// GetRolloutHistory returns historical snapshots
func (pt *PerformanceTracker) GetRolloutHistory() []*RolloutSnapshot {
	pt.mu.RLock()
	defer pt.mu.RUnlock()

	history := make([]*RolloutSnapshot, len(pt.rolloutHistory))
	copy(history, pt.rolloutHistory)
	return history
}

// Helper methods

func (pt *PerformanceTracker) calculatePercentilesV1() {
	if len(pt.v1Metrics.latencyHistogram) == 0 {
		return
	}

	sorted := make([]float64, len(pt.v1Metrics.latencyHistogram))
	copy(sorted, pt.v1Metrics.latencyHistogram)
	sortFloat64s(sorted)

	pt.v1Metrics.P50LatencyMs = calculatePercentile(sorted, 50)
	pt.v1Metrics.P95LatencyMs = calculatePercentile(sorted, 95)
	pt.v1Metrics.P99LatencyMs = calculatePercentile(sorted, 99)
}

func (pt *PerformanceTracker) calculatePercentilesV3() {
	if len(pt.v3Metrics.latencyHistogram) == 0 {
		return
	}

	sorted := make([]float64, len(pt.v3Metrics.latencyHistogram))
	copy(sorted, pt.v3Metrics.latencyHistogram)
	sortFloat64s(sorted)

	pt.v3Metrics.P50LatencyMs = calculatePercentile(sorted, 50)
	pt.v3Metrics.P95LatencyMs = calculatePercentile(sorted, 95)
	pt.v3Metrics.P99LatencyMs = calculatePercentile(sorted, 99)
}

func calculatePercentile(sorted []float64, percentile float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	index := int((percentile / 100.0) * float64(len(sorted)))
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func sortFloat64s(data []float64) {
	// Simple bubble sort (replace with sort.Float64s in production)
	for i := 0; i < len(data); i++ {
		for j := i + 1; j < len(data); j++ {
			if data[i] > data[j] {
				data[i], data[j] = data[j], data[i]
			}
		}
	}
}

func (pt *PerformanceTracker) copyV1Metrics() *VersionMetrics {
	pt.v1Metrics.mu.RLock()
	defer pt.v1Metrics.mu.RUnlock()

	return &VersionMetrics{
		Version:               pt.v1Metrics.Version,
		TotalTransfers:        pt.v1Metrics.TotalTransfers,
		SuccessfulTransfers:   pt.v1Metrics.SuccessfulTransfers,
		FailedTransfers:       pt.v1Metrics.FailedTransfers,
		TotalBytesTransferred: pt.v1Metrics.TotalBytesTransferred,
		AverageThroughputMbps: pt.v1Metrics.AverageThroughputMbps,
		P50LatencyMs:          pt.v1Metrics.P50LatencyMs,
		P95LatencyMs:          pt.v1Metrics.P95LatencyMs,
		P99LatencyMs:          pt.v1Metrics.P99LatencyMs,
		AvgCPUUsage:           pt.v1Metrics.AvgCPUUsage,
		AvgMemoryUsageMB:      pt.v1Metrics.AvgMemoryUsageMB,
	}
}

func (pt *PerformanceTracker) copyV3Metrics() *VersionMetrics {
	pt.v3Metrics.mu.RLock()
	defer pt.v3Metrics.mu.RUnlock()

	return &VersionMetrics{
		Version:               pt.v3Metrics.Version,
		TotalTransfers:        pt.v3Metrics.TotalTransfers,
		SuccessfulTransfers:   pt.v3Metrics.SuccessfulTransfers,
		FailedTransfers:       pt.v3Metrics.FailedTransfers,
		TotalBytesTransferred: pt.v3Metrics.TotalBytesTransferred,
		AverageThroughputMbps: pt.v3Metrics.AverageThroughputMbps,
		P50LatencyMs:          pt.v3Metrics.P50LatencyMs,
		P95LatencyMs:          pt.v3Metrics.P95LatencyMs,
		P99LatencyMs:          pt.v3Metrics.P99LatencyMs,
		AvgCPUUsage:           pt.v3Metrics.AvgCPUUsage,
		AvgMemoryUsageMB:      pt.v3Metrics.AvgMemoryUsageMB,
		CompressionRatio:      pt.v3Metrics.CompressionRatio,
		BandwidthSavedPercent: pt.v3Metrics.BandwidthSavedPercent,
	}
}

func (pt *PerformanceTracker) getV1MetricsSummary() map[string]interface{} {
	pt.v1Metrics.mu.RLock()
	defer pt.v1Metrics.mu.RUnlock()

	return map[string]interface{}{
		"total_transfers":       pt.v1Metrics.TotalTransfers,
		"successful_transfers":  pt.v1Metrics.SuccessfulTransfers,
		"avg_throughput_mbps":   pt.v1Metrics.AverageThroughputMbps,
		"p95_latency_ms":        pt.v1Metrics.P95LatencyMs,
		"avg_cpu_usage":         pt.v1Metrics.AvgCPUUsage,
	}
}

func (pt *PerformanceTracker) getV3MetricsSummary() map[string]interface{} {
	pt.v3Metrics.mu.RLock()
	defer pt.v3Metrics.mu.RUnlock()

	return map[string]interface{}{
		"total_transfers":       pt.v3Metrics.TotalTransfers,
		"successful_transfers":  pt.v3Metrics.SuccessfulTransfers,
		"avg_throughput_mbps":   pt.v3Metrics.AverageThroughputMbps,
		"p95_latency_ms":        pt.v3Metrics.P95LatencyMs,
		"avg_cpu_usage":         pt.v3Metrics.AvgCPUUsage,
		"compression_ratio":     pt.v3Metrics.CompressionRatio,
		"bandwidth_saved_pct":   pt.v3Metrics.BandwidthSavedPercent,
	}
}

func (pt *PerformanceTracker) getRegressionsSummary() []map[string]interface{} {
	summary := make([]map[string]interface{}, len(pt.regressions))
	for i, reg := range pt.regressions {
		summary[i] = map[string]interface{}{
			"timestamp":   reg.Timestamp.Format(time.RFC3339),
			"metric":      reg.Metric,
			"degradation": fmt.Sprintf("%.2f%%", reg.Degradation),
			"severity":    reg.Severity,
		}
	}
	return summary
}

// RecordResourceUsage records CPU and memory usage
func (pt *PerformanceTracker) RecordResourceUsage(version string, cpuPercent, memoryMB float64) {
	if version == "v1" {
		pt.v1Metrics.mu.Lock()
		defer pt.v1Metrics.mu.Unlock()
		// Exponential moving average
		if pt.v1Metrics.AvgCPUUsage == 0 {
			pt.v1Metrics.AvgCPUUsage = cpuPercent
		} else {
			pt.v1Metrics.AvgCPUUsage = pt.v1Metrics.AvgCPUUsage*0.9 + cpuPercent*0.1
		}
		if pt.v1Metrics.AvgMemoryUsageMB == 0 {
			pt.v1Metrics.AvgMemoryUsageMB = memoryMB
		} else {
			pt.v1Metrics.AvgMemoryUsageMB = pt.v1Metrics.AvgMemoryUsageMB*0.9 + memoryMB*0.1
		}
	} else if version == "v3" {
		pt.v3Metrics.mu.Lock()
		defer pt.v3Metrics.mu.Unlock()
		if pt.v3Metrics.AvgCPUUsage == 0 {
			pt.v3Metrics.AvgCPUUsage = cpuPercent
		} else {
			pt.v3Metrics.AvgCPUUsage = pt.v3Metrics.AvgCPUUsage*0.9 + cpuPercent*0.1
		}
		if pt.v3Metrics.AvgMemoryUsageMB == 0 {
			pt.v3Metrics.AvgMemoryUsageMB = memoryMB
		} else {
			pt.v3Metrics.AvgMemoryUsageMB = pt.v3Metrics.AvgMemoryUsageMB*0.9 + memoryMB*0.1
		}
	}
}

// RecordBandwidthSavings records bandwidth savings from compression
func (pt *PerformanceTracker) RecordBandwidthSavings(originalBytes, compressedBytes uint64) {
	pt.v3Metrics.mu.Lock()
	defer pt.v3Metrics.mu.Unlock()

	if originalBytes > 0 {
		savedBytes := originalBytes - compressedBytes
		savedGB := float64(savedBytes) / (1024 * 1024 * 1024)
		pt.v3Metrics.BandwidthSavedGB += savedGB

		// Calculate percentage
		savedPercent := (float64(savedBytes) / float64(originalBytes)) * 100.0
		// Exponential moving average
		if pt.v3Metrics.BandwidthSavedPercent == 0 {
			pt.v3Metrics.BandwidthSavedPercent = savedPercent
		} else {
			pt.v3Metrics.BandwidthSavedPercent = pt.v3Metrics.BandwidthSavedPercent*0.9 + savedPercent*0.1
		}
	}
}
