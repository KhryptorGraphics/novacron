package monitoring

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"go.uber.org/zap"
)

// SeasonalESDDetector implements Seasonal Hybrid ESD (Extreme Studentized Deviate) test
// for detecting anomalies in seasonal time series
type SeasonalESDDetector struct {
	period        int     // Seasonal period (e.g., 24 for hourly data with daily seasonality)
	maxAnomalies  int     // Maximum number of anomalies to detect
	alpha         float64 // Significance level
	logger        *zap.Logger

	timeSeries    map[string]*TimeSeriesData
	mu            sync.RWMutex
}

// TimeSeriesData holds time series data for a metric
type TimeSeriesData struct {
	timestamps []time.Time
	values     []float64
	mu         sync.RWMutex
}

// NewSeasonalESDDetector creates a new Seasonal ESD detector
func NewSeasonalESDDetector(period, maxAnomalies int, logger *zap.Logger) *SeasonalESDDetector {
	if logger == nil {
		logger = zap.NewNop()
	}

	if period <= 0 {
		period = 24 // Default to 24 hours
	}

	if maxAnomalies <= 0 {
		maxAnomalies = 10
	}

	return &SeasonalESDDetector{
		period:       period,
		maxAnomalies: maxAnomalies,
		alpha:        0.05, // 95% confidence
		logger:       logger,
		timeSeries:   make(map[string]*TimeSeriesData),
	}
}

// Detect detects seasonal anomalies
func (sed *SeasonalESDDetector) Detect(ctx context.Context, metrics *MetricVector) (*Anomaly, error) {
	metricNames := []string{
		"bandwidth", "latency", "packet_loss", "jitter",
		"cpu_usage", "memory_usage", "error_rate",
	}

	values := metrics.ToSlice()

	// Update time series for all metrics
	for i, name := range metricNames {
		sed.updateTimeSeries(name, metrics.Timestamp, values[i])
	}

	// Check each metric for anomalies
	var mostSignificantAnomaly *Anomaly
	maxSignificance := 0.0

	for i, name := range metricNames {
		anomaly := sed.detectMetricAnomaly(name, values[i], metrics.Timestamp)
		if anomaly != nil && anomaly.Confidence > maxSignificance {
			mostSignificantAnomaly = anomaly
			maxSignificance = anomaly.Confidence
		}
	}

	return mostSignificantAnomaly, nil
}

// Train trains the Seasonal ESD detector
func (sed *SeasonalESDDetector) Train(ctx context.Context, normalData []*MetricVector) error {
	if len(normalData) < sed.period*2 {
		return fmt.Errorf("insufficient training data: need at least %d samples", sed.period*2)
	}

	sed.logger.Info("Training Seasonal ESD detector",
		zap.Int("samples", len(normalData)),
		zap.Int("period", sed.period))

	metricNames := []string{
		"bandwidth", "latency", "packet_loss", "jitter",
		"cpu_usage", "memory_usage", "error_rate",
	}

	// Initialize time series for each metric
	for _, name := range metricNames {
		ts := &TimeSeriesData{
			timestamps: make([]time.Time, 0, len(normalData)),
			values:     make([]float64, 0, len(normalData)),
		}

		idx := sed.getMetricIndex(name)
		for _, mv := range normalData {
			ts.timestamps = append(ts.timestamps, mv.Timestamp)
			ts.values = append(ts.values, mv.ToSlice()[idx])
		}

		sed.mu.Lock()
		sed.timeSeries[name] = ts
		sed.mu.Unlock()
	}

	sed.logger.Info("Seasonal ESD detector training completed",
		zap.Int("metrics", len(sed.timeSeries)))

	return nil
}

// Name returns the detector name
func (sed *SeasonalESDDetector) Name() string {
	return "seasonal_esd"
}

// detectMetricAnomaly detects anomalies for a specific metric
func (sed *SeasonalESDDetector) detectMetricAnomaly(metricName string, currentValue float64, timestamp time.Time) *Anomaly {
	sed.mu.RLock()
	ts, exists := sed.timeSeries[metricName]
	sed.mu.RUnlock()

	if !exists || len(ts.values) < sed.period*2 {
		return nil
	}

	ts.mu.RLock()
	defer ts.mu.RUnlock()

	// Decompose time series into trend, seasonal, and residual components
	trend, seasonal, residual := sed.stlDecompose(ts.values)

	// Get expected value based on trend and seasonality
	expectedValue := sed.getExpectedValue(trend, seasonal, len(ts.values)-1)

	// Calculate residual for current value
	currentResidual := currentValue - expectedValue

	// Perform ESD test on residuals
	isAnomaly, significance := sed.esdTest(residual, currentResidual)

	if !isAnomaly {
		return nil
	}

	deviation := math.Abs(currentResidual)
	severity := calculateSeverity(significance, deviation/expectedValue*100)

	return &Anomaly{
		Timestamp:   timestamp,
		MetricName:  metricName,
		Value:       currentValue,
		Expected:    expectedValue,
		Deviation:   deviation,
		Severity:    severity,
		Confidence:  significance,
		ModelType:   "seasonal_esd",
		Description: fmt.Sprintf("Seasonal ESD detected anomaly in %s (residual: %.4f)", metricName, currentResidual),
		Context: map[string]interface{}{
			"residual": currentResidual,
			"period":   sed.period,
		},
	}
}

// stlDecompose performs STL (Seasonal and Trend decomposition using Loess) decomposition
// Simplified version - in production, use a proper STL library
func (sed *SeasonalESDDetector) stlDecompose(values []float64) (trend, seasonal, residual []float64) {
	n := len(values)
	trend = make([]float64, n)
	seasonal = make([]float64, n)
	residual = make([]float64, n)

	// Calculate trend using moving average
	windowSize := sed.period
	for i := 0; i < n; i++ {
		start := i - windowSize/2
		end := i + windowSize/2 + 1

		if start < 0 {
			start = 0
		}
		if end > n {
			end = n
		}

		sum := 0.0
		count := 0
		for j := start; j < end; j++ {
			sum += values[j]
			count++
		}

		trend[i] = sum / float64(count)
	}

	// Calculate seasonal component
	detrended := make([]float64, n)
	for i := 0; i < n; i++ {
		detrended[i] = values[i] - trend[i]
	}

	// Average values for each seasonal position
	seasonalAvg := make([]float64, sed.period)
	seasonalCount := make([]int, sed.period)

	for i := 0; i < n; i++ {
		pos := i % sed.period
		seasonalAvg[pos] += detrended[i]
		seasonalCount[pos]++
	}

	for i := 0; i < sed.period; i++ {
		if seasonalCount[i] > 0 {
			seasonalAvg[i] /= float64(seasonalCount[i])
		}
	}

	// Assign seasonal values
	for i := 0; i < n; i++ {
		seasonal[i] = seasonalAvg[i%sed.period]
	}

	// Calculate residual
	for i := 0; i < n; i++ {
		residual[i] = values[i] - trend[i] - seasonal[i]
	}

	return trend, seasonal, residual
}

// getExpectedValue calculates expected value from trend and seasonal components
func (sed *SeasonalESDDetector) getExpectedValue(trend, seasonal []float64, index int) float64 {
	if index < 0 || index >= len(trend) {
		return 0
	}
	return trend[index] + seasonal[index]
}

// esdTest performs the Extreme Studentized Deviate test
func (sed *SeasonalESDDetector) esdTest(residuals []float64, currentResidual float64) (bool, float64) {
	// Calculate mean and standard deviation of residuals
	m := mean(residuals)
	s := stddev(residuals)

	if s == 0 {
		return false, 0
	}

	// Calculate test statistic
	testStat := math.Abs(currentResidual - m) / s

	// Calculate critical value
	n := len(residuals)
	p := 1.0 - sed.alpha/(2.0*float64(n))

	// Approximate critical value using t-distribution
	// For simplicity, using 3-sigma rule approximation
	criticalValue := 3.0 + 0.5*math.Log(float64(n))

	isAnomaly := testStat > criticalValue

	// Calculate significance as normalized test statistic
	significance := math.Min(testStat/criticalValue, 1.0)

	return isAnomaly, significance
}

// updateTimeSeries updates the time series data for a metric
func (sed *SeasonalESDDetector) updateTimeSeries(metricName string, timestamp time.Time, value float64) {
	sed.mu.RLock()
	ts, exists := sed.timeSeries[metricName]
	sed.mu.RUnlock()

	if !exists {
		ts = &TimeSeriesData{
			timestamps: make([]time.Time, 0),
			values:     make([]float64, 0),
		}
		sed.mu.Lock()
		sed.timeSeries[metricName] = ts
		sed.mu.Unlock()
	}

	ts.mu.Lock()
	defer ts.mu.Unlock()

	ts.timestamps = append(ts.timestamps, timestamp)
	ts.values = append(ts.values, value)

	// Keep only data for last 2 periods
	maxSize := sed.period * 2
	if len(ts.values) > maxSize {
		ts.timestamps = ts.timestamps[len(ts.timestamps)-maxSize:]
		ts.values = ts.values[len(ts.values)-maxSize:]
	}
}

// getMetricIndex returns the index of a metric name
func (sed *SeasonalESDDetector) getMetricIndex(metricName string) int {
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

// GetTimeSeriesLength returns the length of time series for a metric
func (sed *SeasonalESDDetector) GetTimeSeriesLength(metricName string) int {
	sed.mu.RLock()
	ts, exists := sed.timeSeries[metricName]
	sed.mu.RUnlock()

	if !exists {
		return 0
	}

	ts.mu.RLock()
	defer ts.mu.RUnlock()

	return len(ts.values)
}

// detectMultipleAnomalies detects multiple anomalies in a time series
func (sed *SeasonalESDDetector) detectMultipleAnomalies(values []float64) []int {
	var anomalies []int
	workingData := make([]float64, len(values))
	copy(workingData, values)

	for k := 0; k < sed.maxAnomalies; k++ {
		// Find maximum deviation
		m := mean(workingData)
		s := stddev(workingData)

		if s == 0 {
			break
		}

		maxDeviation := 0.0
		maxIndex := -1

		for i, v := range workingData {
			deviation := math.Abs(v - m) / s
			if deviation > maxDeviation {
				maxDeviation = deviation
				maxIndex = i
			}
		}

		// Calculate critical value
		n := len(workingData)
		p := 1.0 - sed.alpha/(2.0*float64(n-k))
		criticalValue := 3.0 + 0.5*math.Log(float64(n-k))

		if maxDeviation <= criticalValue {
			break
		}

		anomalies = append(anomalies, maxIndex)

		// Remove the anomaly and continue
		workingData = append(workingData[:maxIndex], workingData[maxIndex+1:]...)
	}

	sort.Ints(anomalies)
	return anomalies
}
