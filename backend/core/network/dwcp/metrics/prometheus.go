// Package metrics provides Prometheus metrics for DWCP components
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// AMST (Adaptive Multi-Stream Transport) Metrics

	// AMSTStreamsActive tracks the current number of active AMST streams
	AMSTStreamsActive = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "amst",
			Name:      "streams_active",
			Help:      "Current number of active AMST streams",
		},
		[]string{"cluster", "node"},
	)

	// AMSTStreamsTotal counts total AMST streams created
	AMSTStreamsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "dwcp",
			Subsystem: "amst",
			Name:      "streams_total",
			Help:      "Total number of AMST streams created",
		},
		[]string{"cluster", "node", "result"},
	)

	// AMSTBytesSent tracks bytes sent per stream
	AMSTBytesSent = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "dwcp",
			Subsystem: "amst",
			Name:      "bytes_sent_total",
			Help:      "Total bytes sent through AMST streams",
		},
		[]string{"cluster", "node", "stream_id"},
	)

	// AMSTBytesReceived tracks bytes received per stream
	AMSTBytesReceived = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "dwcp",
			Subsystem: "amst",
			Name:      "bytes_received_total",
			Help:      "Total bytes received through AMST streams",
		},
		[]string{"cluster", "node", "stream_id"},
	)

	// AMSTErrorsTotal counts stream errors by type
	AMSTErrorsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "dwcp",
			Subsystem: "amst",
			Name:      "errors_total",
			Help:      "Total AMST stream errors by type",
		},
		[]string{"cluster", "node", "error_type"},
	)

	// AMSTLatency tracks stream latency distribution
	AMSTLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "dwcp",
			Subsystem: "amst",
			Name:      "latency_seconds",
			Help:      "AMST stream latency distribution in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.001, 2, 12), // 1ms to ~4s
		},
		[]string{"cluster", "node", "operation"},
	)

	// AMSTBandwidthUtilization tracks percentage of available bandwidth used
	AMSTBandwidthUtilization = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "amst",
			Name:      "bandwidth_utilization_percent",
			Help:      "Percentage of available bandwidth utilized by AMST",
		},
		[]string{"cluster", "node"},
	)

	// HDE (Hierarchical Delta Encoding) Metrics

	// HDECompressionRatio tracks compression ratio distribution
	HDECompressionRatio = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "dwcp",
			Subsystem: "hde",
			Name:      "compression_ratio",
			Help:      "HDE compression ratio distribution (compressed/original)",
			Buckets:   []float64{1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50},
		},
		[]string{"cluster", "node", "data_type"},
	)

	// HDEOperationsTotal counts encode/decode operations
	HDEOperationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "dwcp",
			Subsystem: "hde",
			Name:      "operations_total",
			Help:      "Total HDE encode/decode operations",
		},
		[]string{"cluster", "node", "operation", "result"},
	)

	// HDEDeltaHitRate tracks percentage of delta encoding hits
	HDEDeltaHitRate = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "hde",
			Name:      "delta_hit_rate_percent",
			Help:      "Percentage of successful delta encoding hits",
		},
		[]string{"cluster", "node"},
	)

	// HDEBaselineCount tracks active baselines
	HDEBaselineCount = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "hde",
			Name:      "baseline_count",
			Help:      "Number of active HDE baselines",
		},
		[]string{"cluster", "node", "baseline_type"},
	)

	// HDEDictionaryEfficiency tracks dictionary compression improvement
	HDEDictionaryEfficiency = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "hde",
			Name:      "dictionary_efficiency_percent",
			Help:      "Dictionary compression improvement percentage",
		},
		[]string{"cluster", "node"},
	)

	// HDEBytesOriginal tracks original data size before compression
	HDEBytesOriginal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "dwcp",
			Subsystem: "hde",
			Name:      "bytes_original_total",
			Help:      "Total bytes before HDE compression",
		},
		[]string{"cluster", "node", "data_type"},
	)

	// HDEBytesCompressed tracks compressed data size
	HDEBytesCompressed = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "dwcp",
			Subsystem: "hde",
			Name:      "bytes_compressed_total",
			Help:      "Total bytes after HDE compression",
		},
		[]string{"cluster", "node", "data_type"},
	)

	// Integration Metrics

	// MigrationDuration tracks VM migration time distribution
	MigrationDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "dwcp",
			Subsystem: "migration",
			Name:      "duration_seconds",
			Help:      "VM migration duration distribution in seconds",
			Buckets:   prometheus.ExponentialBuckets(1, 2, 10), // 1s to ~17min
		},
		[]string{"cluster", "source_node", "dest_node", "dwcp_enabled"},
	)

	// MigrationSpeedupFactor tracks DWCP vs standard migration speedup
	MigrationSpeedupFactor = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "migration",
			Name:      "speedup_factor",
			Help:      "Migration speedup factor with DWCP enabled (vs standard)",
		},
		[]string{"cluster", "vm_type"},
	)

	// FederationSyncDuration tracks state synchronization duration
	FederationSyncDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "dwcp",
			Subsystem: "federation",
			Name:      "sync_duration_seconds",
			Help:      "Federation state sync duration in seconds",
			Buckets:   prometheus.ExponentialBuckets(0.01, 2, 10), // 10ms to ~10s
		},
		[]string{"cluster", "remote_cluster", "sync_type"},
	)

	// FederationBandwidthSaved tracks bandwidth savings from DWCP
	FederationBandwidthSaved = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "dwcp",
			Subsystem: "federation",
			Name:      "bandwidth_saved_bytes_total",
			Help:      "Total bandwidth saved through DWCP compression",
		},
		[]string{"cluster", "remote_cluster"},
	)

	// System Metrics

	// ComponentHealth tracks component health status
	ComponentHealth = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "system",
			Name:      "component_health",
			Help:      "Component health status (0=down, 1=degraded, 2=healthy)",
		},
		[]string{"cluster", "node", "component"},
	)

	// ConfigEnabled tracks DWCP enabled status
	ConfigEnabled = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "system",
			Name:      "config_enabled",
			Help:      "DWCP configuration enabled status (0=disabled, 1=enabled)",
		},
		[]string{"cluster", "node", "feature"},
	)

	// VersionInfo provides version information as labels
	VersionInfo = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Namespace: "dwcp",
			Subsystem: "system",
			Name:      "version_info",
			Help:      "DWCP version information",
		},
		[]string{"cluster", "version", "git_commit", "build_date"},
	)
)

// HealthStatus represents component health levels
type HealthStatus int

const (
	HealthDown HealthStatus = iota
	HealthDegraded
	HealthHealthy
)

// RecordAMSTStreamStart records a new AMST stream creation
func RecordAMSTStreamStart(cluster, node string, success bool) {
	result := "success"
	if !success {
		result = "failure"
	}
	AMSTStreamsTotal.WithLabelValues(cluster, node, result).Inc()
	if success {
		AMSTStreamsActive.WithLabelValues(cluster, node).Inc()
	}
}

// RecordAMSTStreamEnd records AMST stream completion
func RecordAMSTStreamEnd(cluster, node string) {
	AMSTStreamsActive.WithLabelValues(cluster, node).Dec()
}

// RecordAMSTData records data sent/received on AMST stream
func RecordAMSTData(cluster, node, streamID string, bytesSent, bytesReceived int64) {
	AMSTBytesSent.WithLabelValues(cluster, node, streamID).Add(float64(bytesSent))
	AMSTBytesReceived.WithLabelValues(cluster, node, streamID).Add(float64(bytesReceived))
}

// RecordAMSTError records an AMST error
func RecordAMSTError(cluster, node, errorType string) {
	AMSTErrorsTotal.WithLabelValues(cluster, node, errorType).Inc()
}

// RecordHDECompression records HDE compression operation
func RecordHDECompression(cluster, node, dataType string, originalBytes, compressedBytes int64, success bool) {
	result := "success"
	if !success {
		result = "failure"
	}

	HDEOperationsTotal.WithLabelValues(cluster, node, "encode", result).Inc()

	if success && originalBytes > 0 {
		ratio := float64(originalBytes) / float64(compressedBytes)
		HDECompressionRatio.WithLabelValues(cluster, node, dataType).Observe(ratio)
		HDEBytesOriginal.WithLabelValues(cluster, node, dataType).Add(float64(originalBytes))
		HDEBytesCompressed.WithLabelValues(cluster, node, dataType).Add(float64(compressedBytes))
	}
}

// RecordHDEDecompression records HDE decompression operation
func RecordHDEDecompression(cluster, node string, success bool) {
	result := "success"
	if !success {
		result = "failure"
	}
	HDEOperationsTotal.WithLabelValues(cluster, node, "decode", result).Inc()
}

// RecordMigration records a VM migration
func RecordMigration(cluster, sourceNode, destNode string, durationSeconds float64, dwcpEnabled bool) {
	dwcpStatus := "false"
	if dwcpEnabled {
		dwcpStatus = "true"
	}
	MigrationDuration.WithLabelValues(cluster, sourceNode, destNode, dwcpStatus).Observe(durationSeconds)
}

// SetComponentHealth sets component health status
func SetComponentHealth(cluster, node, component string, status HealthStatus) {
	ComponentHealth.WithLabelValues(cluster, node, component).Set(float64(status))
}

// SetFeatureEnabled sets feature enabled status
func SetFeatureEnabled(cluster, node, feature string, enabled bool) {
	value := 0.0
	if enabled {
		value = 1.0
	}
	ConfigEnabled.WithLabelValues(cluster, node, feature).Set(value)
}

// SetVersionInfo sets version information (call once at startup)
func SetVersionInfo(cluster, version, gitCommit, buildDate string) {
	VersionInfo.WithLabelValues(cluster, version, gitCommit, buildDate).Set(1)
}
