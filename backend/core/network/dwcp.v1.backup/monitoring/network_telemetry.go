package monitoring

import (
	"context"
	"sync"
	"time"
)

// NetworkTelemetry monitors network metrics
type NetworkTelemetry struct {
	mu sync.RWMutex

	// Inter-region metrics
	bandwidth    map[string]*BandwidthMetrics
	latency      map[string]*LatencyMetrics
	packetLoss   map[string]*PacketLossMetrics
	routes       map[string]*RouteMetrics

	// VPN tunnel health
	tunnels      map[string]*TunnelHealth

	// Topology
	topology     *NetworkTopology
}

// BandwidthMetrics tracks bandwidth utilization
type BandwidthMetrics struct {
	Source      string
	Destination string
	BytesSent   int64
	BytesRecv   int64
	Timestamp   time.Time
	Utilization float64 // 0-100%
	Capacity    int64   // bytes/sec
}

// LatencyMetrics tracks latency measurements
type LatencyMetrics struct {
	Source      string
	Destination string
	LatencyMs   float64
	JitterMs    float64
	Timestamp   time.Time
	Protocol    string // ICMP, TCP
	Samples     []float64
}

// PacketLossMetrics tracks packet loss
type PacketLossMetrics struct {
	Source      string
	Destination string
	Sent        int64
	Received    int64
	LossRate    float64
	Timestamp   time.Time
}

// RouteMetrics tracks route path
type RouteMetrics struct {
	Source      string
	Destination string
	Hops        []string
	HopCount    int
	TotalDelay  float64
	Timestamp   time.Time
}

// TunnelHealth tracks VPN tunnel health
type TunnelHealth struct {
	TunnelID    string
	Source      string
	Destination string
	Status      TunnelStatus
	BytesSent   int64
	BytesRecv   int64
	Latency     float64
	PacketLoss  float64
	Uptime      time.Duration
	LastSeen    time.Time
}

// TunnelStatus represents tunnel status
type TunnelStatus int

const (
	TunnelUp TunnelStatus = iota
	TunnelDown
	TunnelDegraded
)

// NetworkTopology represents network topology
type NetworkTopology struct {
	Nodes []NetworkNode
	Links []NetworkLink
}

// NetworkNode represents a network node
type NetworkNode struct {
	ID       string
	Region   string
	Type     string // router, switch, host
	Status   string
	Metadata map[string]interface{}
}

// NetworkLink represents a network link
type NetworkLink struct {
	Source      string
	Destination string
	Bandwidth   int64
	Latency     float64
	PacketLoss  float64
	Status      string
}

// NewNetworkTelemetry creates a new network telemetry collector
func NewNetworkTelemetry() *NetworkTelemetry {
	return &NetworkTelemetry{
		bandwidth:  make(map[string]*BandwidthMetrics),
		latency:    make(map[string]*LatencyMetrics),
		packetLoss: make(map[string]*PacketLossMetrics),
		routes:     make(map[string]*RouteMetrics),
		tunnels:    make(map[string]*TunnelHealth),
		topology:   &NetworkTopology{},
	}
}

// RecordBandwidth records bandwidth metrics
func (nt *NetworkTelemetry) RecordBandwidth(source, dest string, bytesSent, bytesRecv, capacity int64) {
	nt.mu.Lock()
	defer nt.mu.Unlock()

	key := source + "-" + dest

	utilization := float64(bytesSent+bytesRecv) / float64(capacity) * 100.0

	nt.bandwidth[key] = &BandwidthMetrics{
		Source:      source,
		Destination: dest,
		BytesSent:   bytesSent,
		BytesRecv:   bytesRecv,
		Timestamp:   time.Now(),
		Utilization: utilization,
		Capacity:    capacity,
	}
}

// RecordLatency records latency measurement
func (nt *NetworkTelemetry) RecordLatency(source, dest string, latencyMs float64, protocol string) {
	nt.mu.Lock()
	defer nt.mu.Unlock()

	key := source + "-" + dest

	metrics, ok := nt.latency[key]
	if !ok {
		metrics = &LatencyMetrics{
			Source:      source,
			Destination: dest,
			Protocol:    protocol,
			Samples:     make([]float64, 0),
		}
		nt.latency[key] = metrics
	}

	// Add sample
	metrics.Samples = append(metrics.Samples, latencyMs)
	if len(metrics.Samples) > 100 {
		metrics.Samples = metrics.Samples[1:]
	}

	// Calculate jitter (variance)
	if len(metrics.Samples) > 1 {
		var sum, sumSq float64
		for _, v := range metrics.Samples {
			sum += v
			sumSq += v * v
		}
		mean := sum / float64(len(metrics.Samples))
		variance := (sumSq / float64(len(metrics.Samples))) - (mean * mean)
		metrics.JitterMs = variance
	}

	metrics.LatencyMs = latencyMs
	metrics.Timestamp = time.Now()
}

// RecordPacketLoss records packet loss
func (nt *NetworkTelemetry) RecordPacketLoss(source, dest string, sent, received int64) {
	nt.mu.Lock()
	defer nt.mu.Unlock()

	key := source + "-" + dest

	lossRate := 0.0
	if sent > 0 {
		lossRate = float64(sent-received) / float64(sent) * 100.0
	}

	nt.packetLoss[key] = &PacketLossMetrics{
		Source:      source,
		Destination: dest,
		Sent:        sent,
		Received:    received,
		LossRate:    lossRate,
		Timestamp:   time.Now(),
	}
}

// UpdateTunnelHealth updates VPN tunnel health
func (nt *NetworkTelemetry) UpdateTunnelHealth(tunnelID, source, dest string, status TunnelStatus, metrics map[string]interface{}) {
	nt.mu.Lock()
	defer nt.mu.Unlock()

	tunnel, ok := nt.tunnels[tunnelID]
	if !ok {
		tunnel = &TunnelHealth{
			TunnelID:    tunnelID,
			Source:      source,
			Destination: dest,
		}
		nt.tunnels[tunnelID] = tunnel
	}

	tunnel.Status = status
	tunnel.LastSeen = time.Now()

	if bytesSent, ok := metrics["bytes_sent"].(int64); ok {
		tunnel.BytesSent = bytesSent
	}
	if bytesRecv, ok := metrics["bytes_recv"].(int64); ok {
		tunnel.BytesRecv = bytesRecv
	}
	if latency, ok := metrics["latency"].(float64); ok {
		tunnel.Latency = latency
	}
	if packetLoss, ok := metrics["packet_loss"].(float64); ok {
		tunnel.PacketLoss = packetLoss
	}
}

// GetBandwidthMetrics retrieves bandwidth metrics
func (nt *NetworkTelemetry) GetBandwidthMetrics(source, dest string) (*BandwidthMetrics, bool) {
	nt.mu.RLock()
	defer nt.mu.RUnlock()

	key := source + "-" + dest
	metrics, ok := nt.bandwidth[key]
	return metrics, ok
}

// GetLatencyMetrics retrieves latency metrics
func (nt *NetworkTelemetry) GetLatencyMetrics(source, dest string) (*LatencyMetrics, bool) {
	nt.mu.RLock()
	defer nt.mu.RUnlock()

	key := source + "-" + dest
	metrics, ok := nt.latency[key]
	return metrics, ok
}

// GetTunnelHealth retrieves tunnel health
func (nt *NetworkTelemetry) GetTunnelHealth(tunnelID string) (*TunnelHealth, bool) {
	nt.mu.RLock()
	defer nt.mu.RUnlock()

	health, ok := nt.tunnels[tunnelID]
	return health, ok
}

// UpdateTopology updates network topology
func (nt *NetworkTelemetry) UpdateTopology(topology *NetworkTopology) {
	nt.mu.Lock()
	defer nt.mu.Unlock()

	nt.topology = topology
}

// GetTopology retrieves network topology
func (nt *NetworkTelemetry) GetTopology() *NetworkTopology {
	nt.mu.RLock()
	defer nt.mu.RUnlock()

	return nt.topology
}

// MonitorLatency continuously monitors latency
func (nt *NetworkTelemetry) MonitorLatency(ctx context.Context, pairs []RoutePair, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			for _, pair := range pairs {
				latency := nt.measureLatency(pair.Source, pair.Destination)
				nt.RecordLatency(pair.Source, pair.Destination, latency, "ICMP")
			}
		}
	}
}

// RoutePair represents a source-destination pair
type RoutePair struct {
	Source      string
	Destination string
}

// measureLatency measures latency (simplified)
func (nt *NetworkTelemetry) measureLatency(source, dest string) float64 {
	// Production would use actual ping/TCP measurements
	return 10.0 // placeholder
}

// GetAllMetrics retrieves all network metrics
func (nt *NetworkTelemetry) GetAllMetrics() map[string]interface{} {
	nt.mu.RLock()
	defer nt.mu.RUnlock()

	return map[string]interface{}{
		"bandwidth_pairs":   len(nt.bandwidth),
		"latency_pairs":     len(nt.latency),
		"packet_loss_pairs": len(nt.packetLoss),
		"tunnels":           len(nt.tunnels),
	}
}
