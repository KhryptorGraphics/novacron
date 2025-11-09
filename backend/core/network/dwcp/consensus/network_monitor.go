package consensus

import (
	"sync"
	"time"
)

// NetworkMonitor monitors network conditions for adaptive consensus
type NetworkMonitor struct {
	mu sync.RWMutex

	// Current metrics
	currentMetrics NetworkMetrics

	// Historical metrics
	metricsHistory []NetworkMetrics
	historySize    int

	// Region metrics
	regionMetrics map[string]*RegionMetrics

	// Monitoring configuration
	updateInterval time.Duration
	lastUpdate     time.Time
}

// NewNetworkMonitor creates a new network monitor
func NewNetworkMonitor() *NetworkMonitor {
	return &NetworkMonitor{
		currentMetrics: NetworkMetrics{
			RegionCount:  1,
			AvgLatency:   50 * time.Millisecond,
			MaxLatency:   100 * time.Millisecond,
			PacketLoss:   0.0,
			Bandwidth:    1000000000, // 1 Gbps
			ConflictRate: 0.0,
			Stability:    1.0,
			LastUpdate:   time.Now(),
		},
		metricsHistory: make([]NetworkMetrics, 0, 100),
		historySize:    100,
		regionMetrics:  make(map[string]*RegionMetrics),
		updateInterval: 5 * time.Second,
		lastUpdate:     time.Now(),
	}
}

// GetMetrics returns the current network metrics
func (nm *NetworkMonitor) GetMetrics() NetworkMetrics {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	return nm.currentMetrics
}

// UpdateMetrics updates the network metrics
func (nm *NetworkMonitor) UpdateMetrics(metrics NetworkMetrics) {
	nm.mu.Lock()
	defer nm.mu.Unlock()

	metrics.LastUpdate = time.Now()
	nm.currentMetrics = metrics

	// Add to history
	nm.metricsHistory = append(nm.metricsHistory, metrics)
	if len(nm.metricsHistory) > nm.historySize {
		nm.metricsHistory = nm.metricsHistory[1:]
	}

	nm.lastUpdate = time.Now()
}

// UpdateRegionMetrics updates metrics for a specific region
func (nm *NetworkMonitor) UpdateRegionMetrics(regionID string, metrics *RegionMetrics) {
	nm.mu.Lock()
	defer nm.mu.Unlock()

	metrics.LastUpdate = time.Now()
	nm.regionMetrics[regionID] = metrics

	// Recalculate aggregate metrics
	nm.recalculateAggregateMetrics()
}

// recalculateAggregateMetrics recalculates aggregate metrics from regions
func (nm *NetworkMonitor) recalculateAggregateMetrics() {
	if len(nm.regionMetrics) == 0 {
		return
	}

	var totalLatency time.Duration
	var maxLatency time.Duration
	var totalPacketLoss float64
	var totalBandwidth int64
	regionCount := len(nm.regionMetrics)

	for _, region := range nm.regionMetrics {
		totalLatency += region.Latency
		if region.Latency > maxLatency {
			maxLatency = region.Latency
		}
		totalPacketLoss += region.PacketLoss
		totalBandwidth += region.Bandwidth
	}

	nm.currentMetrics.RegionCount = regionCount
	nm.currentMetrics.AvgLatency = totalLatency / time.Duration(regionCount)
	nm.currentMetrics.MaxLatency = maxLatency
	nm.currentMetrics.PacketLoss = totalPacketLoss / float64(regionCount)
	nm.currentMetrics.Bandwidth = totalBandwidth / int64(regionCount)
	nm.currentMetrics.LastUpdate = time.Now()
}

// GetRegionMetrics returns metrics for a specific region
func (nm *NetworkMonitor) GetRegionMetrics(regionID string) (*RegionMetrics, bool) {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	metrics, exists := nm.regionMetrics[regionID]
	return metrics, exists
}

// GetAllRegionMetrics returns metrics for all regions
func (nm *NetworkMonitor) GetAllRegionMetrics() map[string]*RegionMetrics {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	result := make(map[string]*RegionMetrics)
	for id, metrics := range nm.regionMetrics {
		result[id] = metrics
	}

	return result
}

// GetMetricsHistory returns historical metrics
func (nm *NetworkMonitor) GetMetricsHistory() []NetworkMetrics {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	history := make([]NetworkMetrics, len(nm.metricsHistory))
	copy(history, nm.metricsHistory)

	return history
}

// CalculateStability calculates network stability based on recent metrics
func (nm *NetworkMonitor) CalculateStability() float64 {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	if len(nm.metricsHistory) < 10 {
		return 1.0 // Assume stable if not enough data
	}

	// Calculate variance in latency
	recentHistory := nm.metricsHistory[len(nm.metricsHistory)-10:]
	var avgLatency time.Duration
	for _, m := range recentHistory {
		avgLatency += m.AvgLatency
	}
	avgLatency /= time.Duration(len(recentHistory))

	var variance float64
	for _, m := range recentHistory {
		diff := float64(m.AvgLatency - avgLatency)
		variance += diff * diff
	}
	variance /= float64(len(recentHistory))

	// Stability inversely proportional to variance
	// Lower variance = higher stability
	stdDev := variance
	if avgLatency == 0 {
		return 1.0
	}

	coefficientOfVariation := stdDev / float64(avgLatency)

	// Map to 0-1 range (lower CV = higher stability)
	stability := 1.0 / (1.0 + coefficientOfVariation)

	return stability
}

// UpdateConflictRate updates the conflict rate metric
func (nm *NetworkMonitor) UpdateConflictRate(conflictRate float64) {
	nm.mu.Lock()
	defer nm.mu.Unlock()

	nm.currentMetrics.ConflictRate = conflictRate
}

// MonitorContinuously starts continuous monitoring
func (nm *NetworkMonitor) MonitorContinuously() {
	ticker := time.NewTicker(nm.updateInterval)
	defer ticker.Stop()

	for range ticker.C {
		nm.performMonitoring()
	}
}

// performMonitoring performs one monitoring cycle
func (nm *NetworkMonitor) performMonitoring() {
	// Calculate stability from history
	stability := nm.CalculateStability()

	nm.mu.Lock()
	nm.currentMetrics.Stability = stability
	nm.currentMetrics.LastUpdate = time.Now()
	nm.mu.Unlock()

	// In real implementation, would:
	// 1. Ping all regions to measure latency
	// 2. Measure packet loss rates
	// 3. Test bandwidth
	// 4. Analyze conflict patterns
}

// PredictLatency predicts future latency based on trends
func (nm *NetworkMonitor) PredictLatency(horizon time.Duration) time.Duration {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	if len(nm.metricsHistory) < 5 {
		return nm.currentMetrics.AvgLatency
	}

	// Simple linear trend prediction
	recent := nm.metricsHistory[len(nm.metricsHistory)-5:]
	var sumX, sumY, sumXY, sumX2 float64

	for i, m := range recent {
		x := float64(i)
		y := float64(m.AvgLatency)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	n := float64(len(recent))
	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)

	// Project forward
	futureSteps := float64(horizon / nm.updateInterval)
	predictedLatency := sumY/n + slope*futureSteps

	if predictedLatency < 0 {
		predictedLatency = float64(nm.currentMetrics.AvgLatency)
	}

	return time.Duration(predictedLatency)
}

// IsHealthy checks if the network is healthy
func (nm *NetworkMonitor) IsHealthy() bool {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	// Network is healthy if:
	// 1. Packet loss < 5%
	// 2. Stability > 0.7
	// 3. Latency within reasonable bounds

	return nm.currentMetrics.PacketLoss < 0.05 &&
		nm.currentMetrics.Stability > 0.7 &&
		nm.currentMetrics.AvgLatency < 500*time.Millisecond
}

// GetBottleneckRegion identifies the region with worst metrics
func (nm *NetworkMonitor) GetBottleneckRegion() (string, *RegionMetrics) {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	var worstRegionID string
	var worstMetrics *RegionMetrics
	var worstScore float64 = -1

	for regionID, metrics := range nm.regionMetrics {
		// Higher score = worse performance
		score := float64(metrics.Latency)/float64(time.Millisecond) +
			metrics.PacketLoss*1000 +
			(1.0 / float64(metrics.Bandwidth))

		if score > worstScore {
			worstScore = score
			worstRegionID = regionID
			worstMetrics = metrics
		}
	}

	return worstRegionID, worstMetrics
}
