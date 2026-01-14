package analytics

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// TrendAnalyzer analyzes trends in metrics
type TrendAnalyzer struct {
	// ID is the unique identifier for this analyzer
	ID string

	// Name is the human-readable name of the analyzer
	Name string

	// Description is a description of the analyzer
	Description string

	// DataSource is the key in the context data to analyze
	DataSource string

	// ResultPrefix is the prefix for the result data keys
	ResultPrefix string
}

// NewTrendAnalyzer creates a new trend analyzer
func NewTrendAnalyzer(id, name, description string) *TrendAnalyzer {
	return &TrendAnalyzer{
		ID:           id,
		Name:         name,
		Description:  description,
		DataSource:   "metrics.",
		ResultPrefix: "trend.",
	}
}

// Analyze analyzes data and updates the context
func (a *TrendAnalyzer) Analyze(ctx *PipelineContext) error {
	// Find all metrics that match the data source prefix
	metrics := make([]string, 0)
	for key := range ctx.Data {
		if _, ok := ctx.Data[key].([]interface{}); ok && startsWith(key, a.DataSource) {
			metrics = append(metrics, key)
		}
	}

	// Analyze each metric
	for _, metricKey := range metrics {
		values, ok := ctx.Data[metricKey].([]interface{})
		if !ok || len(values) < 2 {
			continue
		}

		// Convert to points with timestamp and value
		points := make([]Point, 0, len(values))
		for _, v := range values {
			if point, ok := convertToPoint(v); ok {
				points = append(points, point)
			}
		}

		if len(points) < 2 {
			continue
		}

		// Sort points by timestamp
		sort.Slice(points, func(i, j int) bool {
			return points[i].Timestamp.Before(points[j].Timestamp)
		})

		// Calculate linear regression
		slope, intercept, r2 := linearRegression(points)

		// Determine trend direction
		var trend string
		if math.Abs(slope) < 0.001 {
			trend = "stable"
		} else if slope > 0 {
			trend = "increasing"
		} else {
			trend = "decreasing"
		}

		// Store results
		resultKey := a.ResultPrefix + metricKey[len(a.DataSource):]
		ctx.Data[resultKey+".slope"] = slope
		ctx.Data[resultKey+".intercept"] = intercept
		ctx.Data[resultKey+".r2"] = r2
		ctx.Data[resultKey+".direction"] = trend

		// Predict future values
		now := time.Now()
		ctx.Data[resultKey+".prediction.hour"] = predictValue(points, now.Add(time.Hour), slope, intercept)
		ctx.Data[resultKey+".prediction.day"] = predictValue(points, now.Add(24*time.Hour), slope, intercept)
		ctx.Data[resultKey+".prediction.week"] = predictValue(points, now.Add(7*24*time.Hour), slope, intercept)
	}

	return nil
}

// GetMetadata returns metadata about the analyzer
func (a *TrendAnalyzer) GetMetadata() AnalyzerMetadata {
	return AnalyzerMetadata{
		ID:           a.ID,
		Name:         a.Name,
		Description:  a.Description,
		RequiredData: []string{a.DataSource + "*"},
		ProducedData: []string{
			a.ResultPrefix + "*.slope",
			a.ResultPrefix + "*.intercept",
			a.ResultPrefix + "*.r2",
			a.ResultPrefix + "*.direction",
			a.ResultPrefix + "*.prediction.hour",
			a.ResultPrefix + "*.prediction.day",
			a.ResultPrefix + "*.prediction.week",
		},
	}
}

// AnomalyDetector detects anomalies in metrics
type AnomalyDetector struct {
	// ID is the unique identifier for this analyzer
	ID string

	// Name is the human-readable name of the analyzer
	Name string

	// Description is a description of the analyzer
	Description string

	// DataSource is the key in the context data to analyze
	DataSource string

	// ResultPrefix is the prefix for the result data keys
	ResultPrefix string

	// ZScoreThreshold is the threshold for z-score anomaly detection
	ZScoreThreshold float64

	// MADThreshold is the threshold for MAD anomaly detection
	MADThreshold float64
}

// NewAnomalyDetector creates a new anomaly detector
func NewAnomalyDetector(id, name, description string) *AnomalyDetector {
	return &AnomalyDetector{
		ID:              id,
		Name:            name,
		Description:     description,
		DataSource:      "metrics.",
		ResultPrefix:    "anomaly.",
		ZScoreThreshold: 3.0,
		MADThreshold:    3.5,
	}
}

// Analyze analyzes data and updates the context
func (a *AnomalyDetector) Analyze(ctx *PipelineContext) error {
	// Find all metrics that match the data source prefix
	metrics := make([]string, 0)
	for key := range ctx.Data {
		if _, ok := ctx.Data[key].([]interface{}); ok && startsWith(key, a.DataSource) {
			metrics = append(metrics, key)
		}
	}

	// Analyze each metric
	for _, metricKey := range metrics {
		values, ok := ctx.Data[metricKey].([]interface{})
		if !ok || len(values) < 5 { // Need at least 5 points for anomaly detection
			continue
		}

		// Convert to points with timestamp and value
		points := make([]Point, 0, len(values))
		for _, v := range values {
			if point, ok := convertToPoint(v); ok {
				points = append(points, point)
			}
		}

		if len(points) < 5 {
			continue
		}

		// Sort points by timestamp
		sort.Slice(points, func(i, j int) bool {
			return points[i].Timestamp.Before(points[j].Timestamp)
		})

		// Extract just the values
		vals := make([]float64, len(points))
		for i, p := range points {
			vals[i] = p.Value
		}

		// Detect anomalies using z-score method
		anomalies := detectZScoreAnomalies(vals, a.ZScoreThreshold)

		// Detect anomalies using MAD method
		madAnomalies := detectMADAnomalies(vals, a.MADThreshold)

		// Combine results from both methods
		allAnomalies := make([]int, 0)
		for idx := range anomalies {
			if _, ok := madAnomalies[idx]; ok {
				allAnomalies = append(allAnomalies, idx)
			}
		}

		// Create anomaly points
		anomalyPoints := make([]Point, 0, len(allAnomalies))
		for _, idx := range allAnomalies {
			if idx >= 0 && idx < len(points) {
				anomalyPoints = append(anomalyPoints, points[idx])
			}
		}

		// Store results
		resultKey := a.ResultPrefix + metricKey[len(a.DataSource):]
		ctx.Data[resultKey+".count"] = len(anomalyPoints)
		ctx.Data[resultKey+".points"] = anomalyPoints
		ctx.Data[resultKey+".percentage"] = float64(len(anomalyPoints)) / float64(len(points)) * 100.0

		// Calculate severity
		var severity string
		anomalyPercent := float64(len(anomalyPoints)) / float64(len(points)) * 100.0
		if anomalyPercent < 1.0 {
			severity = "low"
		} else if anomalyPercent < 5.0 {
			severity = "medium"
		} else {
			severity = "high"
		}
		ctx.Data[resultKey+".severity"] = severity
	}

	return nil
}

// GetMetadata returns metadata about the analyzer
func (a *AnomalyDetector) GetMetadata() AnalyzerMetadata {
	return AnalyzerMetadata{
		ID:           a.ID,
		Name:         a.Name,
		Description:  a.Description,
		RequiredData: []string{a.DataSource + "*"},
		ProducedData: []string{
			a.ResultPrefix + "*.count",
			a.ResultPrefix + "*.points",
			a.ResultPrefix + "*.percentage",
			a.ResultPrefix + "*.severity",
		},
	}
}

// CapacityAnalyzer analyzes resource capacity and utilization
type CapacityAnalyzer struct {
	// ID is the unique identifier for this analyzer
	ID string

	// Name is the human-readable name of the analyzer
	Name string

	// Description is a description of the analyzer
	Description string

	// VMResourcesKey is the key for VM resources data
	VMResourcesKey string

	// ResultPrefix is the prefix for the result data keys
	ResultPrefix string
}

// NewCapacityAnalyzer creates a new capacity analyzer
func NewCapacityAnalyzer(id, name, description string) *CapacityAnalyzer {
	return &CapacityAnalyzer{
		ID:             id,
		Name:           name,
		Description:    description,
		VMResourcesKey: "vm.resources",
		ResultPrefix:   "capacity.",
	}
}

// Analyze analyzes data and updates the context
func (a *CapacityAnalyzer) Analyze(ctx *PipelineContext) error {
	// Get VM resources data
	vmResourcesData, ok := ctx.Data[a.VMResourcesKey]
	if !ok {
		return fmt.Errorf("VM resources data not found: %s", a.VMResourcesKey)
	}

	vmResources, ok := vmResourcesData.(map[string]map[string][]interface{})
	if !ok {
		return fmt.Errorf("invalid VM resources data format")
	}

	// Analyze capacity for each VM
	vmCapacity := make(map[string]map[string]interface{})

	for vmID, resources := range vmResources {
		vmCap := make(map[string]interface{})

		// Analyze CPU capacity
		if cpuValues, ok := resources["cpu"]; ok && len(cpuValues) > 0 {
			// Extract values
			values := make([]float64, 0, len(cpuValues))
			for _, v := range cpuValues {
				if point, ok := convertToPoint(v); ok {
					values = append(values, point.Value)
				}
			}

			// Calculate statistics
			if len(values) > 0 {
				avg, max, p95 := calculateStatistics(values)
				vmCap["cpu.average"] = avg
				vmCap["cpu.max"] = max
				vmCap["cpu.p95"] = p95

				// Determine capacity status
				var status string
				if p95 > 85.0 {
					status = "critical"
				} else if p95 > 70.0 {
					status = "warning"
				} else {
					status = "ok"
				}
				vmCap["cpu.status"] = status

				// Estimate headroom
				headroom := 100.0 - p95
				vmCap["cpu.headroom"] = headroom
			}
		}

		// Analyze memory capacity
		if memValues, ok := resources["memory"]; ok && len(memValues) > 0 {
			// Extract values
			values := make([]float64, 0, len(memValues))
			for _, v := range memValues {
				if point, ok := convertToPoint(v); ok {
					values = append(values, point.Value)
				}
			}

			// Calculate statistics
			if len(values) > 0 {
				avg, max, p95 := calculateStatistics(values)
				vmCap["memory.average"] = avg
				vmCap["memory.max"] = max
				vmCap["memory.p95"] = p95

				// Determine capacity status
				var status string
				if p95 > 90.0 {
					status = "critical"
				} else if p95 > 80.0 {
					status = "warning"
				} else {
					status = "ok"
				}
				vmCap["memory.status"] = status

				// Estimate headroom
				headroom := 100.0 - p95
				vmCap["memory.headroom"] = headroom
			}
		}

		// Analyze disk capacity
		if diskValues, ok := resources["disk"]; ok && len(diskValues) > 0 {
			// Extract values
			values := make([]float64, 0, len(diskValues))
			for _, v := range diskValues {
				if point, ok := convertToPoint(v); ok {
					values = append(values, point.Value)
				}
			}

			// Calculate statistics
			if len(values) > 0 {
				avg, max, p95 := calculateStatistics(values)
				vmCap["disk.average"] = avg
				vmCap["disk.max"] = max
				vmCap["disk.p95"] = p95

				// Determine capacity status
				var status string
				if p95 > 90.0 {
					status = "critical"
				} else if p95 > 80.0 {
					status = "warning"
				} else {
					status = "ok"
				}
				vmCap["disk.status"] = status

				// Estimate headroom
				headroom := 100.0 - p95
				vmCap["disk.headroom"] = headroom
			}
		}

		// Determine overall capacity status
		var overallStatus string
		if vmCap["cpu.status"] == "critical" || vmCap["memory.status"] == "critical" || vmCap["disk.status"] == "critical" {
			overallStatus = "critical"
		} else if vmCap["cpu.status"] == "warning" || vmCap["memory.status"] == "warning" || vmCap["disk.status"] == "warning" {
			overallStatus = "warning"
		} else {
			overallStatus = "ok"
		}
		vmCap["overall.status"] = overallStatus

		vmCapacity[vmID] = vmCap
	}

	// Store capacity analysis results
	ctx.Data[a.ResultPrefix+"vm"] = vmCapacity

	return nil
}

// GetMetadata returns metadata about the analyzer
func (a *CapacityAnalyzer) GetMetadata() AnalyzerMetadata {
	return AnalyzerMetadata{
		ID:           a.ID,
		Name:         a.Name,
		Description:  a.Description,
		RequiredData: []string{a.VMResourcesKey},
		ProducedData: []string{a.ResultPrefix + "vm"},
	}
}

// Point represents a point with timestamp and value
type Point struct {
	Timestamp time.Time
	Value     float64
}

// Helper function to calculate linear regression
func linearRegression(points []Point) (slope, intercept, r2 float64) {
	n := float64(len(points))
	if n < 2 {
		return 0, 0, 0
	}

	// Convert timestamps to seconds since first point
	baseTime := points[0].Timestamp.Unix()
	xs := make([]float64, len(points))
	ys := make([]float64, len(points))
	for i, p := range points {
		xs[i] = float64(p.Timestamp.Unix() - baseTime)
		ys[i] = p.Value
	}

	// Calculate means
	sumX, sumY := 0.0, 0.0
	for i := 0; i < len(points); i++ {
		sumX += xs[i]
		sumY += ys[i]
	}
	meanX := sumX / n
	meanY := sumY / n

	// Calculate slope and intercept
	numerator, denominator := 0.0, 0.0
	for i := 0; i < len(points); i++ {
		numerator += (xs[i] - meanX) * (ys[i] - meanY)
		denominator += (xs[i] - meanX) * (xs[i] - meanX)
	}

	if denominator == 0 {
		return 0, 0, 0
	}

	slope = numerator / denominator
	intercept = meanY - slope*meanX

	// Calculate R-squared
	ssRes, ssTot := 0.0, 0.0
	for i := 0; i < len(points); i++ {
		predicted := slope*xs[i] + intercept
		ssRes += (ys[i] - predicted) * (ys[i] - predicted)
		ssTot += (ys[i] - meanY) * (ys[i] - meanY)
	}

	if ssTot == 0 {
		r2 = 1.0 // Perfect fit
	} else {
		r2 = 1.0 - (ssRes / ssTot)
	}

	return slope, intercept, r2
}

// Helper function to predict a value at a future time
func predictValue(points []Point, when time.Time, slope, intercept float64) float64 {
	if len(points) == 0 {
		return 0
	}

	baseTime := points[0].Timestamp.Unix()
	x := float64(when.Unix() - baseTime)
	return slope*x + intercept
}

// Helper function to detect anomalies using z-score
func detectZScoreAnomalies(values []float64, threshold float64) map[int]bool {
	n := len(values)
	if n == 0 {
		return map[int]bool{}
	}

	// Calculate mean
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean := sum / float64(n)

	// Calculate standard deviation
	sumSquaredDiff := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquaredDiff += diff * diff
	}
	stdDev := math.Sqrt(sumSquaredDiff / float64(n))

	// If stdDev is too small, can't detect anomalies
	if stdDev < 0.0001 {
		return map[int]bool{}
	}

	// Detect anomalies
	anomalies := make(map[int]bool)
	for i, v := range values {
		zScore := math.Abs(v-mean) / stdDev
		if zScore > threshold {
			anomalies[i] = true
		}
	}

	return anomalies
}

// Helper function to detect anomalies using Median Absolute Deviation (MAD)
func detectMADAnomalies(values []float64, threshold float64) map[int]bool {
	n := len(values)
	if n == 0 {
		return map[int]bool{}
	}

	// Calculate median
	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)
	var median float64
	if n%2 == 0 {
		median = (sorted[n/2-1] + sorted[n/2]) / 2.0
	} else {
		median = sorted[n/2]
	}

	// Calculate absolute deviations
	deviations := make([]float64, n)
	for i, v := range values {
		deviations[i] = math.Abs(v - median)
	}

	// Calculate median of absolute deviations
	sort.Float64s(deviations)
	var mad float64
	if n%2 == 0 {
		mad = (deviations[n/2-1] + deviations[n/2]) / 2.0
	} else {
		mad = deviations[n/2]
	}

	// If MAD is too small, can't detect anomalies
	if mad < 0.0001 {
		return map[int]bool{}
	}

	// Detect anomalies
	anomalies := make(map[int]bool)
	for i, v := range values {
		madScore := math.Abs(v-median) / mad
		if madScore > threshold {
			anomalies[i] = true
		}
	}

	return anomalies
}

// Helper function to calculate statistics (average, max, 95th percentile)
func calculateStatistics(values []float64) (avg, max, p95 float64) {
	n := len(values)
	if n == 0 {
		return 0, 0, 0
	}

	// Calculate average and max
	sum := 0.0
	max = values[0]
	for _, v := range values {
		sum += v
		if v > max {
			max = v
		}
	}
	avg = sum / float64(n)

	// Calculate 95th percentile
	sorted := make([]float64, n)
	copy(sorted, values)
	sort.Float64s(sorted)
	idx95 := int(math.Ceil(float64(n)*0.95)) - 1
	if idx95 < 0 {
		idx95 = 0
	}
	if idx95 >= n {
		idx95 = n - 1
	}
	p95 = sorted[idx95]

	return avg, max, p95
}

// Helper function to convert interface{} to Point
func convertToPoint(v interface{}) (Point, bool) {
	// Check if it's already a Point
	if p, ok := v.(Point); ok {
		return p, true
	}

	// Check if it's a map with timestamp and value
	if m, ok := v.(map[string]interface{}); ok {
		// Extract timestamp
		var timestamp time.Time
		if ts, ok := m["timestamp"].(time.Time); ok {
			timestamp = ts
		} else if tsStr, ok := m["timestamp"].(string); ok {
			var err error
			timestamp, err = time.Parse(time.RFC3339, tsStr)
			if err != nil {
				return Point{}, false
			}
		} else {
			return Point{}, false
		}

		// Extract value
		var value float64
		if val, ok := m["value"].(float64); ok {
			value = val
		} else if val, ok := m["value"].(int); ok {
			value = float64(val)
		} else if val, ok := m["value"].(int64); ok {
			value = float64(val)
		} else {
			return Point{}, false
		}

		return Point{Timestamp: timestamp, Value: value}, true
	}

	return Point{}, false
}

// Helper function to check if a string starts with a prefix
func startsWith(s, prefix string) bool {
	return len(s) >= len(prefix) && s[0:len(prefix)] == prefix
}
