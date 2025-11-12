package monitoring

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ZScoreDetector implements statistical anomaly detection using Z-score
type ZScoreDetector struct {
	window    int
	threshold float64
	logger    *zap.Logger

	metrics   map[string]*MetricStats
	mu        sync.RWMutex
}

// MetricStats holds statistical information for a metric
type MetricStats struct {
	mean   float64
	stddev float64
	values []float64
	mu     sync.RWMutex
}

// NewZScoreDetector creates a new Z-score detector
func NewZScoreDetector(window int, threshold float64, logger *zap.Logger) *ZScoreDetector {
	if logger == nil {
		logger = zap.NewNop()
	}

	if window <= 0 {
		window = 100
	}

	if threshold <= 0 {
		threshold = 3.0 // Standard 3-sigma rule
	}

	return &ZScoreDetector{
		window:    window,
		threshold: threshold,
		logger:    logger,
		metrics:   make(map[string]*MetricStats),
	}
}

// Detect detects anomalies using Z-score
func (zsd *ZScoreDetector) Detect(ctx context.Context, metrics *MetricVector) (*Anomaly, error) {
	metricNames := []string{
		"bandwidth", "latency", "packet_loss", "jitter",
		"cpu_usage", "memory_usage", "error_rate",
	}

	values := metrics.ToSlice()

	maxZScore := 0.0
	maxMetric := ""
	maxValue := 0.0
	maxExpected := 0.0

	// Check each metric
	for i, name := range metricNames {
		value := values[i]

		zscore, expected, err := zsd.calculateZScore(name, value)
		if err != nil {
			continue
		}

		if math.Abs(zscore) > math.Abs(maxZScore) {
			maxZScore = zscore
			maxMetric = name
			maxValue = value
			maxExpected = expected
		}
	}

	// Update statistics for all metrics
	for i, name := range metricNames {
		zsd.updateStats(name, values[i])
	}

	// Check if anomaly
	if math.Abs(maxZScore) <= zsd.threshold {
		return nil, nil
	}

	// Calculate confidence
	confidence := math.Min(math.Abs(maxZScore)/zsd.threshold, 1.0)

	deviation := math.Abs(maxValue - maxExpected)
	severity := calculateSeverity(confidence, deviation/maxExpected*100)

	return &Anomaly{
		Timestamp:   time.Now(),
		MetricName:  maxMetric,
		Value:       maxValue,
		Expected:    maxExpected,
		Deviation:   deviation,
		Severity:    severity,
		Confidence:  confidence,
		ModelType:   "zscore",
		Description: fmt.Sprintf("Z-score detector found statistical anomaly in %s (z-score: %.2f)", maxMetric, maxZScore),
		Context: map[string]interface{}{
			"zscore":    maxZScore,
			"threshold": zsd.threshold,
			"window":    zsd.window,
		},
	}, nil
}

// Train trains the Z-score detector
func (zsd *ZScoreDetector) Train(ctx context.Context, normalData []*MetricVector) error {
	if len(normalData) == 0 {
		return fmt.Errorf("no training data provided")
	}

	zsd.logger.Info("Training Z-score detector",
		zap.Int("samples", len(normalData)),
		zap.Int("window", zsd.window))

	metricNames := []string{
		"bandwidth", "latency", "packet_loss", "jitter",
		"cpu_usage", "memory_usage", "error_rate",
	}

	// Initialize stats for each metric
	for _, name := range metricNames {
		values := make([]float64, 0, len(normalData))
		for _, mv := range normalData {
			idx := zsd.getMetricIndex(name)
			values = append(values, mv.ToSlice()[idx])
		}

		// Keep only last 'window' values
		if len(values) > zsd.window {
			values = values[len(values)-zsd.window:]
		}

		stats := &MetricStats{
			values: values,
		}

		stats.mean = mean(values)
		stats.stddev = stddev(values)

		zsd.mu.Lock()
		zsd.metrics[name] = stats
		zsd.mu.Unlock()
	}

	zsd.logger.Info("Z-score detector training completed",
		zap.Int("metrics", len(zsd.metrics)))

	return nil
}

// Name returns the detector name
func (zsd *ZScoreDetector) Name() string {
	return "zscore"
}

// calculateZScore calculates the Z-score for a value
func (zsd *ZScoreDetector) calculateZScore(metricName string, value float64) (float64, float64, error) {
	zsd.mu.RLock()
	stats, exists := zsd.metrics[metricName]
	zsd.mu.RUnlock()

	if !exists {
		// Initialize if doesn't exist
		stats = &MetricStats{
			values: make([]float64, 0, zsd.window),
		}
		zsd.mu.Lock()
		zsd.metrics[metricName] = stats
		zsd.mu.Unlock()

		return 0, value, nil
	}

	stats.mu.RLock()
	defer stats.mu.RUnlock()

	if len(stats.values) < 2 {
		return 0, value, fmt.Errorf("insufficient data")
	}

	if stats.stddev == 0 {
		return 0, stats.mean, nil
	}

	zscore := (value - stats.mean) / stats.stddev

	return zscore, stats.mean, nil
}

// updateStats updates the rolling statistics for a metric
func (zsd *ZScoreDetector) updateStats(metricName string, value float64) {
	zsd.mu.RLock()
	stats, exists := zsd.metrics[metricName]
	zsd.mu.RUnlock()

	if !exists {
		stats = &MetricStats{
			values: make([]float64, 0, zsd.window),
		}
		zsd.mu.Lock()
		zsd.metrics[metricName] = stats
		zsd.mu.Unlock()
	}

	stats.mu.Lock()
	defer stats.mu.Unlock()

	// Add new value
	stats.values = append(stats.values, value)

	// Keep only last 'window' values
	if len(stats.values) > zsd.window {
		stats.values = stats.values[1:]
	}

	// Recalculate statistics
	stats.mean = mean(stats.values)
	stats.stddev = stddev(stats.values)
}

// getMetricIndex returns the index of a metric name
func (zsd *ZScoreDetector) getMetricIndex(metricName string) int {
	metrics := map[string]int{
		"bandwidth":    0,
		"latency":      1,
		"packet_loss":  2,
		"jitter":       3,
		"cpu_usage":    4,
		"memory_usage": 5,
		"error_rate":   6,
	}
	return metrics[metricName]
}

// GetStats returns current statistics for a metric
func (zsd *ZScoreDetector) GetStats(metricName string) (mean, stddev float64, count int) {
	zsd.mu.RLock()
	stats, exists := zsd.metrics[metricName]
	zsd.mu.RUnlock()

	if !exists {
		return 0, 0, 0
	}

	stats.mu.RLock()
	defer stats.mu.RUnlock()

	return stats.mean, stats.stddev, len(stats.values)
}
