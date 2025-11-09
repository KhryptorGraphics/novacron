package monitoring

import (
	"math"
	"sync"
	"time"
)

// CapacityPlanner performs capacity planning
type CapacityPlanner struct {
	mu sync.RWMutex

	// Historical data
	history map[string]*ResourceHistory

	// Forecasts
	forecasts map[string]*CapacityForecast

	// Thresholds
	warningThreshold  float64 // 80%
	criticalThreshold float64 // 90%
}

// ResourceHistory stores historical resource utilization
type ResourceHistory struct {
	ResourceName string
	DataPoints   []DataPoint
	MaxSize      int
}

// DataPoint represents a resource utilization data point
type DataPoint struct {
	Timestamp   time.Time
	Value       float64
	Capacity    float64
	Utilization float64
}

// CapacityForecast represents capacity forecast
type CapacityForecast struct {
	ResourceName   string
	CurrentValue   float64
	CurrentCapacity float64
	Forecast30d    float64
	Forecast60d    float64
	Forecast90d    float64
	GrowthRate     float64 // per day
	DaysToCapacity int
	Recommendation string
	GeneratedAt    time.Time
}

// BottleneckAnalysis identifies performance bottlenecks
type BottleneckAnalysis struct {
	Resource       string
	BottleneckType string
	Severity       float64 // 0-1
	Impact         string
	Recommendation string
}

// ScaleOutRecommendation provides scale-out recommendations
type ScaleOutRecommendation struct {
	Resource      string
	CurrentNodes  int
	RecommendedNodes int
	TimeFrame     time.Duration
	Justification string
}

// NewCapacityPlanner creates a new capacity planner
func NewCapacityPlanner() *CapacityPlanner {
	return &CapacityPlanner{
		history:           make(map[string]*ResourceHistory),
		forecasts:         make(map[string]*CapacityForecast),
		warningThreshold:  80.0,
		criticalThreshold: 90.0,
	}
}

// RecordUtilization records resource utilization
func (cp *CapacityPlanner) RecordUtilization(resource string, value, capacity float64) {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	history, ok := cp.history[resource]
	if !ok {
		history = &ResourceHistory{
			ResourceName: resource,
			DataPoints:   make([]DataPoint, 0),
			MaxSize:      10080, // 1 week at 1-minute intervals
		}
		cp.history[resource] = history
	}

	utilization := (value / capacity) * 100.0

	dataPoint := DataPoint{
		Timestamp:   time.Now(),
		Value:       value,
		Capacity:    capacity,
		Utilization: utilization,
	}

	history.DataPoints = append(history.DataPoints, dataPoint)

	// Maintain max size
	if len(history.DataPoints) > history.MaxSize {
		history.DataPoints = history.DataPoints[1:]
	}
}

// GenerateForecast generates capacity forecast
func (cp *CapacityPlanner) GenerateForecast(resource string) (*CapacityForecast, error) {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	history, ok := cp.history[resource]
	if !ok || len(history.DataPoints) < 100 {
		return nil, nil // Not enough data
	}

	forecast := &CapacityForecast{
		ResourceName: resource,
		GeneratedAt:  time.Now(),
	}

	points := history.DataPoints

	// Calculate current values
	current := points[len(points)-1]
	forecast.CurrentValue = current.Value
	forecast.CurrentCapacity = current.Capacity

	// Calculate growth rate using linear regression
	growthRate := cp.calculateGrowthRate(points)
	forecast.GrowthRate = growthRate

	// Forecast future values
	forecast.Forecast30d = current.Value + (growthRate * 30)
	forecast.Forecast60d = current.Value + (growthRate * 60)
	forecast.Forecast90d = current.Value + (growthRate * 90)

	// Calculate days to capacity
	if growthRate > 0 {
		remainingCapacity := current.Capacity - current.Value
		forecast.DaysToCapacity = int(remainingCapacity / growthRate)
	} else {
		forecast.DaysToCapacity = -1 // No capacity issues
	}

	// Generate recommendation
	forecast.Recommendation = cp.generateRecommendation(forecast)

	cp.forecasts[resource] = forecast
	return forecast, nil
}

// calculateGrowthRate calculates growth rate using linear regression
func (cp *CapacityPlanner) calculateGrowthRate(points []DataPoint) float64 {
	if len(points) < 2 {
		return 0
	}

	// Simple linear regression
	n := float64(len(points))
	var sumX, sumY, sumXY, sumX2 float64

	for i, point := range points {
		x := float64(i)
		y := point.Value

		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	// Slope = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)

	// Convert to per-day rate
	minutesPerDay := 1440.0
	return slope * minutesPerDay
}

// generateRecommendation generates capacity recommendation
func (cp *CapacityPlanner) generateRecommendation(forecast *CapacityForecast) string {
	if forecast.DaysToCapacity < 0 {
		return "No action needed - capacity is sufficient"
	}

	if forecast.DaysToCapacity < 30 {
		return "URGENT: Scale out immediately - capacity will be exhausted in less than 30 days"
	} else if forecast.DaysToCapacity < 60 {
		return "WARNING: Plan scale-out within next month"
	} else if forecast.DaysToCapacity < 90 {
		return "NOTICE: Consider planning scale-out in next quarter"
	}

	return "Monitor - capacity is adequate for next 90 days"
}

// IdentifyBottlenecks identifies performance bottlenecks
func (cp *CapacityPlanner) IdentifyBottlenecks() []*BottleneckAnalysis {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	var bottlenecks []*BottleneckAnalysis

	for resource, history := range cp.history {
		if len(history.DataPoints) == 0 {
			continue
		}

		// Check recent utilization
		recent := history.DataPoints[len(history.DataPoints)-1]

		if recent.Utilization >= cp.criticalThreshold {
			bottlenecks = append(bottlenecks, &BottleneckAnalysis{
				Resource:       resource,
				BottleneckType: "Critical Utilization",
				Severity:       0.9,
				Impact:         "Performance degradation likely",
				Recommendation: "Immediate scale-out required",
			})
		} else if recent.Utilization >= cp.warningThreshold {
			bottlenecks = append(bottlenecks, &BottleneckAnalysis{
				Resource:       resource,
				BottleneckType: "High Utilization",
				Severity:       0.6,
				Impact:         "Approaching capacity limits",
				Recommendation: "Plan scale-out soon",
			})
		}
	}

	return bottlenecks
}

// GetTrendAnalysis provides trend analysis
func (cp *CapacityPlanner) GetTrendAnalysis(resource string, window time.Duration) map[string]interface{} {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	history, ok := cp.history[resource]
	if !ok || len(history.DataPoints) == 0 {
		return nil
	}

	cutoff := time.Now().Add(-window)
	var values []float64

	for _, point := range history.DataPoints {
		if point.Timestamp.After(cutoff) {
			values = append(values, point.Value)
		}
	}

	if len(values) == 0 {
		return nil
	}

	// Calculate statistics
	var sum, sumSq float64
	min, max := values[0], values[0]

	for _, v := range values {
		sum += v
		sumSq += v * v
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	mean := sum / float64(len(values))
	variance := (sumSq / float64(len(values))) - (mean * mean)
	stdDev := math.Sqrt(variance)

	return map[string]interface{}{
		"mean":     mean,
		"min":      min,
		"max":      max,
		"std_dev":  stdDev,
		"variance": variance,
		"samples":  len(values),
	}
}

// RecommendScaleOut provides scale-out recommendations
func (cp *CapacityPlanner) RecommendScaleOut(resource string, currentNodes int) *ScaleOutRecommendation {
	forecast, err := cp.GenerateForecast(resource)
	if err != nil || forecast == nil {
		return nil
	}

	if forecast.DaysToCapacity < 0 {
		return nil // No scale-out needed
	}

	// Calculate required nodes
	utilizationIncrease := (forecast.Forecast90d - forecast.CurrentValue) / forecast.CurrentCapacity
	additionalNodes := int(math.Ceil(utilizationIncrease * float64(currentNodes)))

	recommendedNodes := currentNodes + additionalNodes

	timeFrame := time.Duration(forecast.DaysToCapacity) * 24 * time.Hour

	return &ScaleOutRecommendation{
		Resource:         resource,
		CurrentNodes:     currentNodes,
		RecommendedNodes: recommendedNodes,
		TimeFrame:        timeFrame,
		Justification:    forecast.Recommendation,
	}
}

// GetForecast retrieves forecast for a resource
func (cp *CapacityPlanner) GetForecast(resource string) (*CapacityForecast, bool) {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	forecast, ok := cp.forecasts[resource]
	return forecast, ok
}

// GetAllForecasts retrieves all forecasts
func (cp *CapacityPlanner) GetAllForecasts() map[string]*CapacityForecast {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	result := make(map[string]*CapacityForecast)
	for k, v := range cp.forecasts {
		result[k] = v
	}
	return result
}
