package dwcp

import (
	"time"
)

// DWCPVersion represents the protocol version
const DWCPVersion = "1.0.0"

// StreamState represents the state of a transport stream
type StreamState int

const (
	StreamStateIdle StreamState = iota
	StreamStateActive
	StreamStateSaturated
	StreamStateClosed
)

// TransportMode defines the underlying transport mechanism
type TransportMode int

const (
	TransportModeTCP TransportMode = iota
	TransportModeRDMA
	TransportModeHybrid
)

// CompressionLevel defines the compression strategy
type CompressionLevel int

const (
	CompressionLevelNone CompressionLevel = iota
	CompressionLevelFast                  // Zstandard level 0
	CompressionLevelBalanced              // Zstandard level 3
	CompressionLevelMax                   // Zstandard level 9
)

// NetworkTier represents the network tier classification
type NetworkTier int

const (
	NetworkTierTier1 NetworkTier = iota // <10ms, >10Gbps
	NetworkTierTier2                    // <50ms, >1Gbps
	NetworkTierTier3                    // >50ms, <1Gbps
)

// TransportMetrics contains transport layer statistics
type TransportMetrics struct {
	StreamCount       int           `json:"stream_count"`
	ActiveStreams     int           `json:"active_streams"`
	TotalBytesSent    uint64        `json:"total_bytes_sent"`
	TotalBytesRecv    uint64        `json:"total_bytes_recv"`
	BandwidthMbps     float64       `json:"bandwidth_mbps"`
	Utilization       float64       `json:"utilization"` // 0.0-1.0
	AverageLatency    time.Duration `json:"average_latency"`
	PacketLossRate    float64       `json:"packet_loss_rate"`
	RetransmissionRate float64       `json:"retransmission_rate"`
	Timestamp         time.Time     `json:"timestamp"`
}

// CompressionMetrics contains compression statistics
type CompressionMetrics struct {
	BytesIn         uint64        `json:"bytes_in"`
	BytesOut        uint64        `json:"bytes_out"`
	CompressionRatio float64       `json:"compression_ratio"`
	CompressionTime time.Duration `json:"compression_time"`
	DecompressionTime time.Duration `json:"decompression_time"`
	Level           CompressionLevel `json:"level"`
	DeltaHitRate    float64       `json:"delta_hit_rate"` // % of data that was delta-encoded
	Timestamp       time.Time     `json:"timestamp"`
}

// DWCPMetrics aggregates all DWCP statistics
type DWCPMetrics struct {
	Transport        TransportMetrics   `json:"transport"`
	Compression      CompressionMetrics `json:"compression"`
	Tier             NetworkTier        `json:"tier"`
	Mode             TransportMode      `json:"mode"`
	Enabled          bool               `json:"enabled"`
	Version          string             `json:"version"`
	DegradationLevel string             `json:"degradation_level"` // Resilience: normal, degraded, severely_degraded, emergency
	IsHealthy        bool               `json:"is_healthy"`        // Overall health status
}

// Error types
type DWCPError struct {
	Code    string
	Message string
	Cause   error
}

func (e *DWCPError) Error() string {
	if e.Cause != nil {
		return e.Message + ": " + e.Cause.Error()
	}
	return e.Message
}

// Common error codes
const (
	ErrCodeStreamCreation    = "STREAM_CREATION_FAILED"
	ErrCodeStreamClosed      = "STREAM_CLOSED"
	ErrCodeCompressionFailed = "COMPRESSION_FAILED"
	ErrCodeDecompressionFailed = "DECOMPRESSION_FAILED"
	ErrCodeInvalidConfig     = "INVALID_CONFIG"
	ErrCodeNetworkTimeout    = "NETWORK_TIMEOUT"
)
