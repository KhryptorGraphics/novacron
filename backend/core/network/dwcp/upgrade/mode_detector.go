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
	ModeInternet                      // v3: TCP, 100-900 Mbps, 50-500ms latency
	ModeHybrid                        // Adaptive switching between modes
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
	datacenterPacketLossMax      float64       // <=0.1% for datacenter
	internetPacketLossThreshold  float64       // >1% for internet

	// Historical metrics for better detection
	latencyHistory    []time.Duration
	bandwidthHistory  []int64
	packetLossHistory []float64
	historySize       int

	// Metrics collector (interface to avoid circular dependency)
	metricsCollector interface {
		GetAverageLatency() time.Duration
		GetAverageBandwidth() int64
	}

	modeChangeHandler func(previous, current NetworkMode)
}

// NewModeDetector creates a new mode detector with default thresholds
func NewModeDetector() *ModeDetector {
	return &ModeDetector{
		currentMode:                  ModeHybrid, // Start with hybrid mode
		datacenterLatencyThreshold:   10 * time.Millisecond,
		internetLatencyThreshold:     50 * time.Millisecond,
		datacenterBandwidthThreshold: 1e9, // 1 Gbps
		internetBandwidthThreshold:   1e9, // 1 Gbps
		datacenterPacketLossMax:      0.001,
		internetPacketLossThreshold:  0.01,
		latencyHistory:               make([]time.Duration, 0, 10),
		bandwidthHistory:             make([]int64, 0, 10),
		packetLossHistory:            make([]float64, 0, 10),
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
	packetLoss := md.measurePacketLoss(ctx)

	// Add to history
	md.addToHistory(latency, bandwidth, packetLoss)

	// Calculate average from history for stability
	avgLatency := md.averageLatency()
	avgBandwidth := md.averageBandwidth()
	avgPacketLoss := md.averagePacketLoss()

	// Datacenter mode: low latency AND high bandwidth
	if avgLatency < md.datacenterLatencyThreshold &&
		avgBandwidth >= md.datacenterBandwidthThreshold &&
		avgPacketLoss <= md.datacenterPacketLossMax {
		md.currentMode = ModeDatacenter
		return ModeDatacenter
	}

	// Internet mode: high latency OR low bandwidth
	if avgLatency > md.internetLatencyThreshold ||
		avgBandwidth < md.internetBandwidthThreshold ||
		avgPacketLoss > md.internetPacketLossThreshold {
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
	if md.metricsCollector != nil {
		if latency := md.metricsCollector.GetAverageLatency(); latency > 0 {
			return latency
		}
	}

	if len(md.latencyHistory) > 0 {
		return md.averageLatency()
	}

	return md.datacenterLatencyThreshold
}

// measureBandwidth measures available bandwidth
func (md *ModeDetector) measureBandwidth(ctx context.Context) int64 {
	if md.metricsCollector != nil {
		if bandwidth := md.metricsCollector.GetAverageBandwidth(); bandwidth > 0 {
			return bandwidth
		}
	}

	if len(md.bandwidthHistory) > 0 {
		return md.averageBandwidth()
	}

	return md.datacenterBandwidthThreshold
}

// measurePacketLoss measures packet loss ratio from optional collector telemetry.
func (md *ModeDetector) measurePacketLoss(ctx context.Context) float64 {
	if md.metricsCollector != nil {
		if collector, ok := md.metricsCollector.(interface {
			GetPacketLossRatio() float64
		}); ok {
			if packetLoss := collector.GetPacketLossRatio(); packetLoss > 0 {
				return packetLoss
			}
		}
	}

	if len(md.packetLossHistory) > 0 {
		return md.averagePacketLoss()
	}

	return 0
}

// addToHistory adds measurements to history with circular buffer
func (md *ModeDetector) addToHistory(latency time.Duration, bandwidth int64, packetLoss float64) {
	if len(md.latencyHistory) >= md.historySize {
		md.latencyHistory = md.latencyHistory[1:]
	}
	md.latencyHistory = append(md.latencyHistory, latency)

	if len(md.bandwidthHistory) >= md.historySize {
		md.bandwidthHistory = md.bandwidthHistory[1:]
	}
	md.bandwidthHistory = append(md.bandwidthHistory, bandwidth)

	if len(md.packetLossHistory) >= md.historySize {
		md.packetLossHistory = md.packetLossHistory[1:]
	}
	md.packetLossHistory = append(md.packetLossHistory, packetLoss)
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

// averagePacketLoss calculates average packet loss ratio from history.
func (md *ModeDetector) averagePacketLoss() float64 {
	if len(md.packetLossHistory) == 0 {
		return 0
	}

	var sum float64
	for _, loss := range md.packetLossHistory {
		sum += loss
	}
	return sum / float64(len(md.packetLossHistory))
}

// SetMetricsCollector sets the metrics collector for accurate measurements
func (md *ModeDetector) SetMetricsCollector(mc interface {
	GetAverageLatency() time.Duration
	GetAverageBandwidth() int64
}) {
	md.mu.Lock()
	defer md.mu.Unlock()
	md.metricsCollector = mc
}

// SetModeChangeHandler sets a callback invoked when the auto-detected mode changes.
func (md *ModeDetector) SetModeChangeHandler(handler func(previous, current NetworkMode)) {
	md.mu.Lock()
	defer md.mu.Unlock()
	md.modeChangeHandler = handler
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
			previousMode := md.GetCurrentMode()
			mode := md.DetectMode(ctx)
			if mode != previousMode {
				md.notifyModeChange(previousMode, mode)
			}
		}
	}
}

func (md *ModeDetector) notifyModeChange(previous, current NetworkMode) {
	md.mu.RLock()
	handler := md.modeChangeHandler
	md.mu.RUnlock()

	if handler != nil {
		handler(previous, current)
	}
}

// ForceMode manually sets the network mode (for testing/debugging)
func (md *ModeDetector) ForceMode(mode NetworkMode) {
	md.mu.Lock()
	defer md.mu.Unlock()
	md.currentMode = mode
}
