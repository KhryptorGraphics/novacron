package security

import (
	"reflect"
	"sync"
	"time"
)

// MetricsCollector collects and aggregates metrics for the backpressure manager
type MetricsCollector struct {
	interval        time.Duration
	mu              sync.RWMutex
	eventCounts     map[EventPriority]int64
	processingTimes map[EventPriority][]time.Duration
	errorCounts     map[string]int64
	lastReset       time.Time
	totalEvents     int64
	droppedEvents   int64
	spilledEvents   int64
	throttledEvents int64
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(interval time.Duration) *MetricsCollector {
	return &MetricsCollector{
		interval:        interval,
		eventCounts:     make(map[EventPriority]int64),
		processingTimes: make(map[EventPriority][]time.Duration),
		errorCounts:     make(map[string]int64),
		lastReset:       time.Now(),
	}
}

// RecordEvent records an event processing
func (mc *MetricsCollector) RecordEvent(priority EventPriority, duration time.Duration) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.eventCounts[priority]++
	mc.totalEvents++

	// Keep a sliding window of processing times
	times := mc.processingTimes[priority]
	times = append(times, duration)

	// Keep only the last 1000 measurements
	if len(times) > 1000 {
		times = times[len(times)-1000:]
	}
	mc.processingTimes[priority] = times
}

// RecordError records an error
func (mc *MetricsCollector) RecordError(errorType string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.errorCounts[errorType]++
}

// RecordDrop records a dropped event
func (mc *MetricsCollector) RecordDrop() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.droppedEvents++
}

// RecordSpill records a spilled event
func (mc *MetricsCollector) RecordSpill() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.spilledEvents++
}

// RecordThrottle records a throttled event
func (mc *MetricsCollector) RecordThrottle() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.throttledEvents++
}

// GetStats returns current statistics
func (mc *MetricsCollector) GetStats() MetricsStats {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	elapsed := time.Since(mc.lastReset)

	stats := MetricsStats{
		Interval:        elapsed,
		TotalEvents:     mc.totalEvents,
		DroppedEvents:   mc.droppedEvents,
		SpilledEvents:   mc.spilledEvents,
		ThrottledEvents: mc.throttledEvents,
		EventCounts:     make(map[EventPriority]int64),
		ProcessingStats: make(map[EventPriority]ProcessingStats),
		ErrorCounts:     make(map[string]int64),
	}

	// Copy event counts
	for priority, count := range mc.eventCounts {
		stats.EventCounts[priority] = count
	}

	// Calculate processing statistics
	for priority, times := range mc.processingTimes {
		if len(times) > 0 {
			stats.ProcessingStats[priority] = mc.calculateProcessingStats(times)
		}
	}

	// Copy error counts
	for errorType, count := range mc.errorCounts {
		stats.ErrorCounts[errorType] = count
	}

	// Calculate rates
	if elapsed.Seconds() > 0 {
		stats.EventsPerSecond = float64(mc.totalEvents) / elapsed.Seconds()
		stats.DropsPerSecond = float64(mc.droppedEvents) / elapsed.Seconds()
	}

	return stats
}

// calculateProcessingStats calculates statistics for processing times
func (mc *MetricsCollector) calculateProcessingStats(times []time.Duration) ProcessingStats {
	if len(times) == 0 {
		return ProcessingStats{}
	}

	// Calculate min, max, and sum
	min := times[0]
	max := times[0]
	var sum time.Duration

	for _, t := range times {
		if t < min {
			min = t
		}
		if t > max {
			max = t
		}
		sum += t
	}

	avg := sum / time.Duration(len(times))

	// Calculate percentiles (simple implementation)
	sortedTimes := make([]time.Duration, len(times))
	copy(sortedTimes, times)

	// Simple bubble sort for small arrays
	n := len(sortedTimes)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if sortedTimes[j] > sortedTimes[j+1] {
				sortedTimes[j], sortedTimes[j+1] = sortedTimes[j+1], sortedTimes[j]
			}
		}
	}

	return ProcessingStats{
		Count:  int64(len(times)),
		Min:    min,
		Max:    max,
		Avg:    avg,
		P50:    sortedTimes[len(sortedTimes)/2],
		P95:    sortedTimes[int(float64(len(sortedTimes))*0.95)],
		P99:    sortedTimes[int(float64(len(sortedTimes))*0.99)],
	}
}

// Reset resets all metrics
func (mc *MetricsCollector) Reset() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.eventCounts = make(map[EventPriority]int64)
	mc.processingTimes = make(map[EventPriority][]time.Duration)
	mc.errorCounts = make(map[string]int64)
	mc.totalEvents = 0
	mc.droppedEvents = 0
	mc.spilledEvents = 0
	mc.throttledEvents = 0
	mc.lastReset = time.Now()
}

// MetricsStats represents collected metrics
type MetricsStats struct {
	Interval         time.Duration                    `json:"interval"`
	TotalEvents      int64                           `json:"total_events"`
	DroppedEvents    int64                           `json:"dropped_events"`
	SpilledEvents    int64                           `json:"spilled_events"`
	ThrottledEvents  int64                           `json:"throttled_events"`
	EventsPerSecond  float64                         `json:"events_per_second"`
	DropsPerSecond   float64                         `json:"drops_per_second"`
	EventCounts      map[EventPriority]int64         `json:"event_counts"`
	ProcessingStats  map[EventPriority]ProcessingStats `json:"processing_stats"`
	ErrorCounts      map[string]int64                `json:"error_counts"`
}

// ProcessingStats represents processing time statistics
type ProcessingStats struct {
	Count int64         `json:"count"`
	Min   time.Duration `json:"min"`
	Max   time.Duration `json:"max"`
	Avg   time.Duration `json:"avg"`
	P50   time.Duration `json:"p50"`
	P95   time.Duration `json:"p95"`
	P99   time.Duration `json:"p99"`
}

// GetHealthStatus returns health status based on metrics
func (stats MetricsStats) GetHealthStatus() HealthStatus {
	status := HealthStatus{
		Healthy:  true,
		Issues:   []string{},
		Warnings: []string{},
	}

	// Check drop rate
	if stats.DropsPerSecond > 10 {
		status.Healthy = false
		status.Issues = append(status.Issues, "High drop rate detected")
	} else if stats.DropsPerSecond > 1 {
		status.Warnings = append(status.Warnings, "Elevated drop rate")
	}

	// Check processing latency
	for priority, procStats := range stats.ProcessingStats {
		if procStats.P99 > time.Second {
			status.Healthy = false
			status.Issues = append(status.Issues,
				fmt.Sprintf("High P99 latency for %s priority: %v", priority, procStats.P99))
		} else if procStats.P95 > 500*time.Millisecond {
			status.Warnings = append(status.Warnings,
				fmt.Sprintf("Elevated P95 latency for %s priority: %v", priority, procStats.P95))
		}
	}

	// Check spill rate
	spillRate := float64(stats.SpilledEvents)
	if stats.Interval.Seconds() > 0 {
		spillRate = float64(stats.SpilledEvents) / stats.Interval.Seconds()
	}

	if spillRate > 50 {
		status.Warnings = append(status.Warnings, "High spill rate - consider increasing queue sizes")
	}

	return status
}

// HealthStatus represents the health status of the system
type HealthStatus struct {
	Healthy  bool     `json:"healthy"`
	Issues   []string `json:"issues"`
	Warnings []string `json:"warnings"`
}