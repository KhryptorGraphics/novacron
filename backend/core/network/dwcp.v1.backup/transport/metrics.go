package transport

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// Prometheus metrics for AMST
	metricsOnce sync.Once

	// Stream metrics
	activeStreamsGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "active_streams",
			Help:      "Number of active TCP/RDMA streams",
		},
		[]string{"transport_type", "remote_addr"},
	)

	totalStreamsGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "total_streams",
			Help:      "Total number of configured streams",
		},
		[]string{"transport_type", "remote_addr"},
	)

	// Transfer metrics
	bytesTransferredCounter = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "bytes_transferred_total",
			Help:      "Total bytes transferred (sent + received)",
		},
		[]string{"transport_type", "remote_addr", "direction"},
	)

	throughputGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "throughput_mbps",
			Help:      "Current throughput in Mbps",
		},
		[]string{"transport_type", "remote_addr"},
	)

	// Error metrics
	errorsCounter = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "errors_total",
			Help:      "Total number of errors",
		},
		[]string{"transport_type", "remote_addr", "error_type"},
	)

	reconnectsCounter = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "reconnects_total",
			Help:      "Total number of stream reconnections",
		},
		[]string{"transport_type", "remote_addr"},
	)

	// Performance metrics
	latencyHistogram = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "latency_seconds",
			Help:      "Latency histogram in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.0001, 2, 15), // 0.1ms to ~3.2s
		},
		[]string{"transport_type", "remote_addr", "operation"},
	)

	packetLossGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "packet_loss_rate",
			Help:      "Packet loss rate (0-1)",
		},
		[]string{"transport_type", "remote_addr"},
	)

	bandwidthUtilizationGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "bandwidth_utilization",
			Help:      "Bandwidth utilization percentage (0-100)",
		},
		[]string{"transport_type", "remote_addr"},
	)

	// Health metrics
	healthStatusGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "health_status",
			Help:      "Health status (1=healthy, 0=unhealthy)",
		},
		[]string{"transport_type", "remote_addr"},
	)

	healthCheckDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "health_check_duration_seconds",
			Help:      "Health check duration in seconds",
			Buckets:   prometheus.DefBuckets,
		},
		[]string{"transport_type", "remote_addr"},
	)

	// Congestion control metrics
	congestionWindowGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "novacron",
			Subsystem: "amst",
			Name:      "congestion_window",
			Help:      "TCP congestion window size",
		},
		[]string{"transport_type", "remote_addr", "algorithm"},
	)
)

// MetricsCollector collects and reports transport metrics
type MetricsCollector struct {
	transportType string
	remoteAddr    string
	mu            sync.RWMutex

	// Accumulated metrics
	totalBytesSent    uint64
	totalBytesRecv    uint64
	totalErrors       uint64
	totalReconnects   uint64

	// Current state
	currentThroughput float64
	currentLatency    float64
	currentPacketLoss float64
	currentBandwidth  float64
	isHealthy         bool

	// Timing
	lastUpdate        time.Time
	startTime         time.Time
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(transportType, remoteAddr string) *MetricsCollector {
	metricsOnce.Do(func() {
		// Register all metrics (already done via promauto)
	})

	return &MetricsCollector{
		transportType: transportType,
		remoteAddr:    remoteAddr,
		startTime:     time.Now(),
		lastUpdate:    time.Now(),
		isHealthy:     true,
	}
}

// RecordActiveStreams records the current number of active streams
func (mc *MetricsCollector) RecordActiveStreams(count int32) {
	activeStreamsGauge.WithLabelValues(mc.transportType, mc.remoteAddr).Set(float64(count))
}

// RecordTotalStreams records the total number of streams
func (mc *MetricsCollector) RecordTotalStreams(count int) {
	totalStreamsGauge.WithLabelValues(mc.transportType, mc.remoteAddr).Set(float64(count))
}

// RecordBytesSent records bytes sent
func (mc *MetricsCollector) RecordBytesSent(bytes uint64) {
	mc.mu.Lock()
	mc.totalBytesSent += bytes
	mc.mu.Unlock()

	bytesTransferredCounter.WithLabelValues(mc.transportType, mc.remoteAddr, "sent").Add(float64(bytes))
}

// RecordBytesReceived records bytes received
func (mc *MetricsCollector) RecordBytesReceived(bytes uint64) {
	mc.mu.Lock()
	mc.totalBytesRecv += bytes
	mc.mu.Unlock()

	bytesTransferredCounter.WithLabelValues(mc.transportType, mc.remoteAddr, "received").Add(float64(bytes))
}

// RecordThroughput records current throughput in Mbps
func (mc *MetricsCollector) RecordThroughput(mbps float64) {
	mc.mu.Lock()
	mc.currentThroughput = mbps
	mc.mu.Unlock()

	throughputGauge.WithLabelValues(mc.transportType, mc.remoteAddr).Set(mbps)
}

// RecordError records an error occurrence
func (mc *MetricsCollector) RecordError(errorType string) {
	mc.mu.Lock()
	mc.totalErrors++
	mc.mu.Unlock()

	errorsCounter.WithLabelValues(mc.transportType, mc.remoteAddr, errorType).Inc()
}

// RecordReconnect records a stream reconnection
func (mc *MetricsCollector) RecordReconnect() {
	mc.mu.Lock()
	mc.totalReconnects++
	mc.mu.Unlock()

	reconnectsCounter.WithLabelValues(mc.transportType, mc.remoteAddr).Inc()
}

// RecordLatency records operation latency
func (mc *MetricsCollector) RecordLatency(operation string, duration time.Duration) {
	mc.mu.Lock()
	mc.currentLatency = duration.Seconds() * 1000 // Convert to ms
	mc.mu.Unlock()

	latencyHistogram.WithLabelValues(mc.transportType, mc.remoteAddr, operation).Observe(duration.Seconds())
}

// RecordPacketLoss records packet loss rate
func (mc *MetricsCollector) RecordPacketLoss(rate float64) {
	mc.mu.Lock()
	mc.currentPacketLoss = rate
	mc.mu.Unlock()

	packetLossGauge.WithLabelValues(mc.transportType, mc.remoteAddr).Set(rate)
}

// RecordBandwidthUtilization records bandwidth utilization percentage
func (mc *MetricsCollector) RecordBandwidthUtilization(percentage float64) {
	mc.mu.Lock()
	mc.currentBandwidth = percentage
	mc.mu.Unlock()

	bandwidthUtilizationGauge.WithLabelValues(mc.transportType, mc.remoteAddr).Set(percentage)
}

// RecordHealthStatus records health status
func (mc *MetricsCollector) RecordHealthStatus(healthy bool) {
	mc.mu.Lock()
	mc.isHealthy = healthy
	mc.mu.Unlock()

	var status float64
	if healthy {
		status = 1.0
	}
	healthStatusGauge.WithLabelValues(mc.transportType, mc.remoteAddr).Set(status)
}

// RecordHealthCheckDuration records health check duration
func (mc *MetricsCollector) RecordHealthCheckDuration(duration time.Duration) {
	healthCheckDuration.WithLabelValues(mc.transportType, mc.remoteAddr).Observe(duration.Seconds())
}

// RecordCongestionWindow records TCP congestion window size
func (mc *MetricsCollector) RecordCongestionWindow(algorithm string, size int) {
	congestionWindowGauge.WithLabelValues(mc.transportType, mc.remoteAddr, algorithm).Set(float64(size))
}

// GetMetrics returns current metrics snapshot
func (mc *MetricsCollector) GetMetrics() TransportMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	return TransportMetrics{
		TotalBytesSent:    mc.totalBytesSent,
		TotalBytesRecv:    mc.totalBytesRecv,
		ThroughputMbps:    mc.currentThroughput,
		AverageLatencyMs:  mc.currentLatency,
		PacketLossRate:    mc.currentPacketLoss,
		BandwidthUtilized: mc.currentBandwidth,
		TransportType:     mc.transportType,
		Healthy:           mc.isHealthy,
		LastHealthCheck:   mc.lastUpdate,
	}
}

// Reset resets all metrics
func (mc *MetricsCollector) Reset() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.totalBytesSent = 0
	mc.totalBytesRecv = 0
	mc.totalErrors = 0
	mc.totalReconnects = 0
	mc.currentThroughput = 0
	mc.currentLatency = 0
	mc.currentPacketLoss = 0
	mc.currentBandwidth = 0
	mc.startTime = time.Now()
	mc.lastUpdate = time.Now()
}
