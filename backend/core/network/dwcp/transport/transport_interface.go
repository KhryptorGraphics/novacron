package transport

import (
	"context"
	"time"
)

// Transport defines the common interface for all transport implementations
type Transport interface {
	// Lifecycle management
	Start() error
	Close() error
	IsStarted() bool

	// Data transfer
	Send(data []byte) error
	Receive(expectedSize int) ([]byte, error)

	// Dynamic adaptation
	AdjustStreams(bandwidthMbps, latencyMs float64) error

	// Metrics and monitoring
	GetMetrics() TransportMetrics
	HealthCheck() error
}

// TransportMetrics provides transport-level metrics
type TransportMetrics struct {
	// Connection metrics
	ActiveStreams     int32  `json:"active_streams"`
	TotalStreams      int    `json:"total_streams"`
	FailedConnections uint64 `json:"failed_connections"`

	// Transfer metrics
	TotalBytesSent uint64  `json:"total_bytes_sent"`
	TotalBytesRecv uint64  `json:"total_bytes_recv"`
	ThroughputMbps float64 `json:"throughput_mbps"`

	// Performance metrics
	AverageLatencyMs  float64 `json:"average_latency_ms"`
	PacketLossRate    float64 `json:"packet_loss_rate"`
	BandwidthUtilized float64 `json:"bandwidth_utilized"` // percentage

	// Transport type and mode
	TransportType     string `json:"transport_type"`     // "tcp", "rdma", "hybrid"
	TransportMode     string `json:"transport_mode"`     // "datacenter", "internet", "hybrid"
	CongestionControl string `json:"congestion_control"` // "bbr", "cubic", etc.

	// Health status
	Healthy         bool      `json:"healthy"`
	LastError       string    `json:"last_error,omitempty"`
	LastHealthCheck time.Time `json:"last_health_check"`
}

// TransportConfig is the base configuration for all transports
type TransportConfig struct {
	// Common settings
	RemoteAddr     string        `json:"remote_addr" yaml:"remote_addr"`
	ConnectTimeout time.Duration `json:"connect_timeout" yaml:"connect_timeout"`

	// Stream configuration
	MinStreams int `json:"min_streams" yaml:"min_streams"`
	MaxStreams int `json:"max_streams" yaml:"max_streams"`

	// Performance tuning
	ChunkSizeKB   int   `json:"chunk_size_kb" yaml:"chunk_size_kb"`
	AutoTune      bool  `json:"auto_tune" yaml:"auto_tune"`
	PacingEnabled bool  `json:"pacing_enabled" yaml:"pacing_enabled"`
	PacingRate    int64 `json:"pacing_rate" yaml:"pacing_rate"` // bytes per second

	// Transport-specific
	EnableRDMA          bool   `json:"enable_rdma" yaml:"enable_rdma"`
	RDMADevice          string `json:"rdma_device" yaml:"rdma_device"`
	RDMAPort            int    `json:"rdma_port" yaml:"rdma_port"`
	CongestionAlgorithm string `json:"congestion_algorithm" yaml:"congestion_algorithm"` // "bbr", "cubic"

	// Reliability
	EnableRetries       bool          `json:"enable_retries" yaml:"enable_retries"`
	MaxRetries          int           `json:"max_retries" yaml:"max_retries"`
	RetryBackoffMs      int           `json:"retry_backoff_ms" yaml:"retry_backoff_ms"`
	HealthCheckInterval time.Duration `json:"health_check_interval" yaml:"health_check_interval"`
}

// DefaultTransportConfig returns sensible defaults
func DefaultTransportConfig() *TransportConfig {
	return &TransportConfig{
		ConnectTimeout:      30 * time.Second,
		MinStreams:          16,
		MaxStreams:          256,
		ChunkSizeKB:         256,
		AutoTune:            true,
		PacingEnabled:       true,
		PacingRate:          1000 * 1024 * 1024, // 1 Gbps
		EnableRDMA:          false,
		RDMADevice:          "mlx5_0",
		RDMAPort:            1,
		CongestionAlgorithm: "bbr",
		EnableRetries:       true,
		MaxRetries:          3,
		RetryBackoffMs:      100,
		HealthCheckInterval: 10 * time.Second,
	}
}

// StreamHealth represents the health status of a single stream
type StreamHealth struct {
	StreamID         int       `json:"stream_id"`
	Healthy          bool      `json:"healthy"`
	BytesSent        uint64    `json:"bytes_sent"`
	BytesRecv        uint64    `json:"bytes_recv"`
	LastActive       time.Time `json:"last_active"`
	LastError        string    `json:"last_error,omitempty"`
	Reconnects       int       `json:"reconnects"`
	ConsecutiveFails int       `json:"consecutive_fails"`
}

// TransportFactory creates transport instances based on configuration
type TransportFactory interface {
	CreateTransport(ctx context.Context, config *TransportConfig) (Transport, error)
	SupportsRDMA() bool
}
