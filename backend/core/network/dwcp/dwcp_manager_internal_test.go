package dwcp

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/transport"
	"go.uber.org/zap"
)

type managerTestTransport struct {
	metrics transport.TransportMetrics
	started bool
	health  error
}

func (t *managerTestTransport) Start() error      { t.started = true; return nil }
func (t *managerTestTransport) Close() error      { t.started = false; return nil }
func (t *managerTestTransport) IsStarted() bool   { return t.started }
func (t *managerTestTransport) Send([]byte) error { return nil }
func (t *managerTestTransport) Receive(expectedSize int) ([]byte, error) {
	return make([]byte, expectedSize), nil
}
func (t *managerTestTransport) AdjustStreams(bandwidthMbps, latencyMs float64) error { return nil }
func (t *managerTestTransport) GetMetrics() transport.TransportMetrics               { return t.metrics }
func (t *managerTestTransport) HealthCheck() error                                   { return t.health }

type managerTestCompression struct {
	metrics *CompressionMetrics
	healthy bool
}

func (c *managerTestCompression) Start(context.Context) error { return nil }
func (c *managerTestCompression) Stop() error                 { return nil }
func (c *managerTestCompression) IsRunning() bool             { return true }
func (c *managerTestCompression) HealthCheck() error {
	if c.healthy {
		return nil
	}
	return errors.New("compression unhealthy")
}
func (c *managerTestCompression) IsHealthy() bool { return c.healthy }
func (c *managerTestCompression) Encode(key string, data []byte, tier int) (*EncodedData, error) {
	return &EncodedData{Data: data, OriginalSize: len(data), CompressedSize: len(data), Tier: tier}, nil
}
func (c *managerTestCompression) Decode(key string, data *EncodedData) ([]byte, error) {
	return data.Data, nil
}
func (c *managerTestCompression) GetMetrics() *CompressionMetrics { return c.metrics }

type managerTestSync struct {
	started bool
	err     error
}

func (s *managerTestSync) Start(context.Context) error {
	if s.err != nil {
		return s.err
	}
	s.started = true
	return nil
}
func (s *managerTestSync) Stop() error               { s.started = false; return nil }
func (s *managerTestSync) IsRunning() bool           { return s.started }
func (s *managerTestSync) HealthCheck() error        { return nil }
func (s *managerTestSync) IsHealthy() bool           { return true }
func (s *managerTestSync) Sync(string, []byte) error { return nil }
func (s *managerTestSync) GetMetrics() *SyncMetrics  { return &SyncMetrics{} }

type managerTestConsensus struct {
	started bool
	err     error
}

func (c *managerTestConsensus) Start(context.Context) error {
	if c.err != nil {
		return c.err
	}
	c.started = true
	return nil
}
func (c *managerTestConsensus) Stop() error                   { c.started = false; return nil }
func (c *managerTestConsensus) IsRunning() bool               { return c.started }
func (c *managerTestConsensus) HealthCheck() error            { return nil }
func (c *managerTestConsensus) IsHealthy() bool               { return true }
func (c *managerTestConsensus) Propose([]byte) error          { return nil }
func (c *managerTestConsensus) GetMetrics() *ConsensusMetrics { return &ConsensusMetrics{} }

func TestCollectMetricsCopiesComponentMetrics(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	manager, err := NewManager(config, zap.NewNop())
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	now := time.Now().UTC()
	manager.transport = &managerTestTransport{
		metrics: transport.TransportMetrics{
			ActiveStreams:     12,
			TotalStreams:      24,
			TotalBytesSent:    4096,
			TotalBytesRecv:    8192,
			ThroughputMbps:    750,
			AverageLatencyMs:  12.5,
			PacketLossRate:    0.002,
			BandwidthUtilized: 72,
			TransportType:     "hybrid",
			Healthy:           true,
			LastHealthCheck:   now,
		},
	}
	manager.compression = &managerTestCompression{
		healthy: true,
		metrics: &CompressionMetrics{
			BytesIn:          1000,
			BytesOut:         250,
			CompressionRatio: 4,
			Level:            CompressionLevelBalanced,
			Timestamp:        now,
		},
	}

	manager.collectMetrics()

	metrics := manager.GetMetrics()
	if !metrics.Enabled || metrics.Version != DWCPVersion {
		t.Fatalf("basic metrics not updated: enabled=%v version=%q", metrics.Enabled, metrics.Version)
	}
	if metrics.Transport.StreamCount != 24 || metrics.Transport.ActiveStreams != 12 {
		t.Fatalf("transport stream metrics not copied: %+v", metrics.Transport)
	}
	if metrics.Transport.TotalBytesSent != 4096 || metrics.Transport.TotalBytesRecv != 8192 {
		t.Fatalf("transport byte metrics not copied: %+v", metrics.Transport)
	}
	if metrics.Transport.BandwidthMbps != 750 {
		t.Fatalf("transport bandwidth not copied: %+v", metrics.Transport)
	}
	if metrics.Transport.Utilization != 0.72 {
		t.Fatalf("transport utilization not normalized: got %v", metrics.Transport.Utilization)
	}
	if metrics.Transport.AverageLatency != 12500*time.Microsecond {
		t.Fatalf("transport latency not converted: got %v", metrics.Transport.AverageLatency)
	}
	if metrics.Tier != NetworkTierTier2 {
		t.Fatalf("network tier not derived from latency: got %v", metrics.Tier)
	}
	if metrics.Mode != TransportModeHybrid {
		t.Fatalf("transport mode not derived from metrics: got %v", metrics.Mode)
	}
	if metrics.Compression.BytesIn != 1000 || metrics.Compression.BytesOut != 250 {
		t.Fatalf("compression metrics not copied: %+v", metrics.Compression)
	}
	if !metrics.IsHealthy {
		t.Fatal("manager should report healthy when components are healthy")
	}
}

func TestCollectMetricsMarksUnhealthyComponents(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	manager, err := NewManager(config, zap.NewNop())
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	manager.transport = &managerTestTransport{
		metrics: transport.TransportMetrics{
			AverageLatencyMs: 200,
			TransportType:    "tcp",
			Healthy:          false,
			LastHealthCheck:  time.Now(),
		},
	}
	manager.compression = &managerTestCompression{healthy: false, metrics: &CompressionMetrics{}}

	manager.collectMetrics()

	metrics := manager.GetMetrics()
	if metrics.IsHealthy {
		t.Fatal("manager should report unhealthy when components are unhealthy")
	}
	if metrics.Tier != NetworkTierTier4 {
		t.Fatalf("high-latency network tier not classified as global WAN: got %v", metrics.Tier)
	}
}

func TestHealthCheckReportsUnhealthyCompression(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	manager, err := NewManager(config, zap.NewNop())
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}
	manager.started = true
	manager.compression = &managerTestCompression{healthy: false, metrics: &CompressionMetrics{}}

	err = manager.HealthCheck()
	if err == nil || !strings.Contains(err.Error(), "compression layer unhealthy") {
		t.Fatalf("expected compression health error, got %v", err)
	}
}

func TestStartPhase2ComponentsStartsConfiguredLayers(t *testing.T) {
	config := DefaultConfig()
	config.Sync.Enabled = true
	config.Consensus.Enabled = true
	manager, err := NewManager(config, zap.NewNop())
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}
	syncLayer := &managerTestSync{}
	consensusLayer := &managerTestConsensus{}
	manager.sync = syncLayer
	manager.consensus = consensusLayer

	if err := manager.startPhase2Components(context.Background()); err != nil {
		t.Fatalf("startPhase2Components failed: %v", err)
	}
	if !syncLayer.started {
		t.Fatal("sync layer was not started")
	}
	if !consensusLayer.started {
		t.Fatal("consensus layer was not started")
	}
}

func TestStartPhase2ComponentsReturnsStartErrors(t *testing.T) {
	config := DefaultConfig()
	config.Sync.Enabled = true
	manager, err := NewManager(config, zap.NewNop())
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}
	manager.sync = &managerTestSync{err: errors.New("sync start failed")}

	err = manager.startPhase2Components(context.Background())
	if err == nil || !strings.Contains(err.Error(), "failed to start sync layer") {
		t.Fatalf("expected sync start error, got %v", err)
	}
}

func TestStopStopsPreconfiguredPartitionerBeforeManagerStart(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	manager, err := NewManager(config, zap.NewNop())
	if err != nil {
		t.Fatalf("NewManager failed: %v", err)
	}

	if err := manager.AddTaskPartitioner(""); err != nil {
		t.Fatalf("AddTaskPartitioner failed: %v", err)
	}
	partitioner := manager.partitioner
	t.Cleanup(partitioner.Destroy)

	if !partitioner.IsRunning() {
		t.Fatal("partitioner should start enabled when added")
	}

	if err := manager.Stop(); err != nil {
		t.Fatalf("Stop failed: %v", err)
	}

	if partitioner.IsRunning() {
		t.Fatal("Stop should stop a preconfigured partitioner even before manager start")
	}
}
