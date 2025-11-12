package upgrade

import (
	"context"
	"sync"
	"time"
)

// NetworkMode represents DWCP operation mode
type NetworkMode int

const (
	ModeDatacenter NetworkMode = iota // v1: RDMA, 10-100 Gbps, <10ms latency
	ModeInternet                       // v3: TCP, 100-900 Mbps, 50-500ms latency
	ModeHybrid                         // Adaptive switching between modes
)

// String returns the string representation of NetworkMode
func (m NetworkMode) String() string {
	switch m {
	case ModeDatacenter:
		return "datacenter"
	case ModeInternet:
		return "internet"
	case ModeHybrid:
		return "hybrid"
	default:
		return "unknown"
	}
}

// ModeDetector automatically detects optimal network mode based on conditions
type ModeDetector struct {
	mu sync.RWMutex

	// Current detected mode
	currentMode NetworkMode

	// Thresholds for mode detection
	datacenterLatencyThreshold   time.Duration // <10ms for datacenter
	internetLatencyThreshold     time.Duration // >50ms for internet
	datacenterBandwidthThreshold int64         // >1 Gbps for datacenter
	internetBandwidthThreshold   int64         // <1 Gbps for internet

	// Historical metrics for better detection
	latencyHistory   []time.Duration
	bandwidthHistory []int64
	historySize      int

	// Metrics collector (interface to avoid circular dependency)
	metricsCollector interface{
		GetAverageLatency() time.Duration
		GetAverageBandwidth() int64
	}
}

// NewModeDetector creates a new mode detector with default thresholds
func NewModeDetector() *ModeDetector {
	return &ModeDetector{
		currentMode:                  ModeHybrid, // Start with hybrid mode
		datacenterLatencyThreshold:   10 * time.Millisecond,
		internetLatencyThreshold:     50 * time.Millisecond,
		datacenterBandwidthThreshold: 1e9, // 1 Gbps
		internetBandwidthThreshold:   1e9, // 1 Gbps
		latencyHistory:               make([]time.Duration, 0, 10),
		bandwidthHistory:             make([]int64, 0, 10),
		historySize:                  10,
	}
}

// DetectMode analyzes network conditions and returns optimal mode
func (md *ModeDetector) DetectMode(ctx context.Context) NetworkMode {
	md.mu.Lock()
	defer md.mu.Unlock()

	// Measure current conditions
	latency := md.measureLatency(ctx)
	bandwidth := md.measureBandwidth(ctx)

	// Add to history
	md.addToHistory(latency, bandwidth)

	// Calculate average from history for stability
	avgLatency := md.averageLatency()
	avgBandwidth := md.averageBandwidth()

	// Datacenter mode: low latency AND high bandwidth
	if avgLatency < md.datacenterLatencyThreshold && avgBandwidth >= md.datacenterBandwidthThreshold {
		md.currentMode = ModeDatacenter
		return ModeDatacenter
	}

	// Internet mode: high latency OR low bandwidth
	if avgLatency > md.internetLatencyThreshold || avgBandwidth < md.internetBandwidthThreshold {
		md.currentMode = ModeInternet
		return ModeInternet
	}

	// Hybrid mode: borderline conditions
	md.currentMode = ModeHybrid
	return ModeHybrid
}

// GetCurrentMode returns the currently detected mode
func (md *ModeDetector) GetCurrentMode() NetworkMode {
	md.mu.RLock()
	defer md.mu.RUnlock()
	return md.currentMode
}

// measureLatency measures RTT to peer nodes
func (md *ModeDetector) measureLatency(ctx context.Context) time.Duration {
	// TODO: Implement actual latency measurement
	// Options:
	// 1. ICMP ping to peer nodes
	// 2. TCP handshake timing
	// 3. Application-level ping (DWCP heartbeat)

	// For now, return placeholder based on existing metrics
	if md.metricsCollector != nil {
		// Get latency from metrics collector
		return md.metricsCollector.GetAverageLatency()
	}

	// Default placeholder: 5ms (datacenter)
	return 5 * time.Millisecond
}

// measureBandwidth measures available bandwidth
func (md *ModeDetector) measureBandwidth(ctx context.Context) int64 {
	// TODO: Implement actual bandwidth measurement
	// Options:
	// 1. iperf-style bandwidth test
	// 2. Monitor actual transfer rates
	// 3. Use historical throughput data

	// For now, return placeholder based on existing metrics
	if md.metricsCollector != nil {
		// Get bandwidth from metrics collector
		return md.metricsCollector.GetAverageBandwidth()
	}

	// Default placeholder: 10 Gbps (datacenter)
	return 10e9
}

// addToHistory adds measurements to history with circular buffer
func (md *ModeDetector) addToHistory(latency time.Duration, bandwidth int64) {
	// Add latency
	if len(md.latencyHistory) >= md.historySize {
		// Remove oldest
		md.latencyHistory = md.latencyHistory[1:]
	}
	md.latencyHistory = append(md.latencyHistory, latency)

	// Add bandwidth
	if len(md.bandwidthHistory) >= md.historySize {
		// Remove oldest
		md.bandwidthHistory = md.bandwidthHistory[1:]
	}
	md.bandwidthHistory = append(md.bandwidthHistory, bandwidth)
}

// averageLatency calculates average latency from history
func (md *ModeDetector) averageLatency() time.Duration {
	if len(md.latencyHistory) == 0 {
		return md.datacenterLatencyThreshold
	}

	var sum time.Duration
	for _, lat := range md.latencyHistory {
		sum += lat
	}
	return sum / time.Duration(len(md.latencyHistory))
}

// averageBandwidth calculates average bandwidth from history
func (md *ModeDetector) averageBandwidth() int64 {
	if len(md.bandwidthHistory) == 0 {
		return md.datacenterBandwidthThreshold
	}

	var sum int64
	for _, bw := range md.bandwidthHistory {
		sum += bw
	}
	return sum / int64(len(md.bandwidthHistory))
}

// SetMetricsCollector sets the metrics collector for accurate measurements
func (md *ModeDetector) SetMetricsCollector(mc interface{
	GetAverageLatency() time.Duration
	GetAverageBandwidth() int64
}) {
	md.mu.Lock()
	defer md.mu.Unlock()
	md.metricsCollector = mc
}

// AutoDetectLoop continuously detects mode in the background
func (md *ModeDetector) AutoDetectLoop(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			mode := md.DetectMode(ctx)
			// Log mode changes
			if mode != md.GetCurrentMode() {
				// TODO: Add structured logging
				// log.Infof("Network mode changed: %s -> %s", md.currentMode, mode)
			}
		}
	}
}

// ForceMode manually sets the network mode (for testing/debugging)
func (md *ModeDetector) ForceMode(mode NetworkMode) {
	md.mu.Lock()
	defer md.mu.Unlock()
	md.currentMode = mode
}
