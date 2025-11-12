package loadbalancing

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

// MetricsCollector collects and tracks load balancer metrics
type MetricsCollector struct {
	config *LoadBalancerConfig

	// Request metrics
	totalRequests   uint64
	totalFailures   uint64
	totalFailovers  uint64
	requestsPerSec  uint64

	// Latency tracking
	routingLatencies  []time.Duration
	responseLatencies []time.Duration
	failoverTimes     []time.Duration

	// Connection tracking
	activeConnections int32
	totalConnections  uint64

	// Geographic distribution
	requestsByRegion map[string]uint64

	mu       sync.RWMutex
	ctx      context.Context
	cancel   context.CancelFunc
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(config *LoadBalancerConfig) *MetricsCollector {
	ctx, cancel := context.WithCancel(context.Background())

	return &MetricsCollector{
		config:            config,
		routingLatencies:  make([]time.Duration, 0, 10000),
		responseLatencies: make([]time.Duration, 0, 10000),
		failoverTimes:     make([]time.Duration, 0, 1000),
		requestsByRegion:  make(map[string]uint64),
		ctx:               ctx,
		cancel:            cancel,
	}
}

// Start begins metrics collection
func (mc *MetricsCollector) Start() {
	go mc.runMetricsAggregation()
}

// Stop stops metrics collection
func (mc *MetricsCollector) Stop() {
	mc.cancel()
}

// RecordRoutingDecision records a routing decision
func (mc *MetricsCollector) RecordRoutingDecision(decision *RoutingDecision) {
	atomic.AddUint64(&mc.totalRequests, 1)

	mc.mu.Lock()
	defer mc.mu.Unlock()

	// Record routing latency
	if len(mc.routingLatencies) < 10000 {
		mc.routingLatencies = append(mc.routingLatencies, decision.Latency)
	} else {
		// Rotate buffer
		copy(mc.routingLatencies, mc.routingLatencies[1:])
		mc.routingLatencies[len(mc.routingLatencies)-1] = decision.Latency
	}

	// Track by region
	if decision.Server != nil {
		mc.requestsByRegion[decision.Server.Region]++
	}

	// Track failovers
	if decision.IsFailover {
		atomic.AddUint64(&mc.totalFailovers, 1)
	}
}

// RecordResponse records a response from a backend server
func (mc *MetricsCollector) RecordResponse(responseTime time.Duration, success bool) {
	if !success {
		atomic.AddUint64(&mc.totalFailures, 1)
	}

	mc.mu.Lock()
	defer mc.mu.Unlock()

	// Record response latency
	if len(mc.responseLatencies) < 10000 {
		mc.responseLatencies = append(mc.responseLatencies, responseTime)
	} else {
		// Rotate buffer
		copy(mc.responseLatencies, mc.responseLatencies[1:])
		mc.responseLatencies[len(mc.responseLatencies)-1] = responseTime
	}
}

// RecordFailover records a failover event
func (mc *MetricsCollector) RecordFailover(failoverTime time.Duration) {
	atomic.AddUint64(&mc.totalFailovers, 1)

	mc.mu.Lock()
	defer mc.mu.Unlock()

	if len(mc.failoverTimes) < 1000 {
		mc.failoverTimes = append(mc.failoverTimes, failoverTime)
	} else {
		// Rotate buffer
		copy(mc.failoverTimes, mc.failoverTimes[1:])
		mc.failoverTimes[len(mc.failoverTimes)-1] = failoverTime
	}
}

// RecordFailure records a request failure
func (mc *MetricsCollector) RecordFailure() {
	atomic.AddUint64(&mc.totalFailures, 1)
}

// IncrementConnections increments active connection count
func (mc *MetricsCollector) IncrementConnections() {
	atomic.AddInt32(&mc.activeConnections, 1)
	atomic.AddUint64(&mc.totalConnections, 1)
}

// DecrementConnections decrements active connection count
func (mc *MetricsCollector) DecrementConnections() {
	atomic.AddInt32(&mc.activeConnections, -1)
}

// runMetricsAggregation aggregates metrics periodically
func (mc *MetricsCollector) runMetricsAggregation() {
	ticker := time.NewTicker(mc.config.MetricsInterval)
	defer ticker.Stop()

	lastRequestCount := uint64(0)
	lastTime := time.Now()

	for {
		select {
		case <-mc.ctx.Done():
			return
		case <-ticker.C:
			currentRequests := atomic.LoadUint64(&mc.totalRequests)
			currentTime := time.Now()

			// Calculate requests per second
			elapsed := currentTime.Sub(lastTime).Seconds()
			if elapsed > 0 {
				rps := float64(currentRequests-lastRequestCount) / elapsed
				atomic.StoreUint64(&mc.requestsPerSec, uint64(rps))
			}

			lastRequestCount = currentRequests
			lastTime = currentTime
		}
	}
}

// GetMetrics returns current metrics snapshot
func (mc *MetricsCollector) GetMetrics() *LoadBalancerStats {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	stats := &LoadBalancerStats{
		TotalRequests:    atomic.LoadUint64(&mc.totalRequests),
		TotalFailures:    atomic.LoadUint64(&mc.totalFailures),
		TotalFailovers:   atomic.LoadUint64(&mc.totalFailovers),
		TotalConnections: atomic.LoadInt32(&mc.activeConnections),
		RequestsPerSecond: float64(atomic.LoadUint64(&mc.requestsPerSec)),
	}

	// Calculate percentiles for routing latency
	if len(mc.routingLatencies) > 0 {
		stats.AvgRoutingLatency = mc.calculateAverage(mc.routingLatencies)
	}

	// Calculate percentiles for response time
	if len(mc.responseLatencies) > 0 {
		stats.P50ResponseTime = mc.calculatePercentile(mc.responseLatencies, 0.50)
		stats.P95ResponseTime = mc.calculatePercentile(mc.responseLatencies, 0.95)
		stats.P99ResponseTime = mc.calculatePercentile(mc.responseLatencies, 0.99)
	}

	// Calculate average failover time
	if len(mc.failoverTimes) > 0 {
		stats.AvgFailoverTime = mc.calculateAverage(mc.failoverTimes)
	}

	return stats
}

// calculateAverage calculates the average of a duration slice
func (mc *MetricsCollector) calculateAverage(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}

	total := time.Duration(0)
	for _, d := range durations {
		total += d
	}

	return total / time.Duration(len(durations))
}

// calculatePercentile calculates the percentile of a duration slice
func (mc *MetricsCollector) calculatePercentile(durations []time.Duration, percentile float64) time.Duration {
	if len(durations) == 0 {
		return 0
	}

	// Create a copy and sort
	sorted := make([]time.Duration, len(durations))
	copy(sorted, durations)

	// Simple insertion sort (efficient for small/mostly sorted data)
	for i := 1; i < len(sorted); i++ {
		key := sorted[i]
		j := i - 1
		for j >= 0 && sorted[j] > key {
			sorted[j+1] = sorted[j]
			j--
		}
		sorted[j+1] = key
	}

	// Calculate percentile index
	index := int(float64(len(sorted)-1) * percentile)
	if index >= len(sorted) {
		index = len(sorted) - 1
	}

	return sorted[index]
}

// GetRegionDistribution returns request distribution by region
func (mc *MetricsCollector) GetRegionDistribution() map[string]uint64 {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	dist := make(map[string]uint64, len(mc.requestsByRegion))
	for region, count := range mc.requestsByRegion {
		dist[region] = count
	}

	return dist
}

// Reset resets all metrics
func (mc *MetricsCollector) Reset() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	atomic.StoreUint64(&mc.totalRequests, 0)
	atomic.StoreUint64(&mc.totalFailures, 0)
	atomic.StoreUint64(&mc.totalFailovers, 0)
	atomic.StoreUint64(&mc.requestsPerSec, 0)
	atomic.StoreInt32(&mc.activeConnections, 0)
	atomic.StoreUint64(&mc.totalConnections, 0)

	mc.routingLatencies = make([]time.Duration, 0, 10000)
	mc.responseLatencies = make([]time.Duration, 0, 10000)
	mc.failoverTimes = make([]time.Duration, 0, 1000)
	mc.requestsByRegion = make(map[string]uint64)
}
