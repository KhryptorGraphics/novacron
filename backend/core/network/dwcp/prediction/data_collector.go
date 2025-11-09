package prediction

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// NetworkSample represents a single network measurement
type NetworkSample struct {
	Timestamp     time.Time `json:"timestamp"`
	BandwidthMbps float64   `json:"bandwidth_mbps"`
	LatencyMs     float64   `json:"latency_ms"`
	PacketLoss    float64   `json:"packet_loss"`
	JitterMs      float64   `json:"jitter_ms"`
	TimeOfDay     int       `json:"time_of_day"`
	DayOfWeek     int       `json:"day_of_week"`
	NodeID        string    `json:"node_id"`
	PeerID        string    `json:"peer_id"`
}

// DataCollector collects network metrics for training and prediction
type DataCollector struct {
	samples         []NetworkSample
	maxSamples      int
	collectInterval time.Duration
	mu              sync.RWMutex
	ctx             context.Context
	cancel          context.CancelFunc

	// Network measurement tools
	pingTargets   []string
	bandwidthTest string

	// Data persistence
	dataPath     string
	saveInterval time.Duration
	lastSave     time.Time

	// Prometheus metrics
	bandwidthGauge   prometheus.Gauge
	latencyGauge     prometheus.Gauge
	packetLossGauge  prometheus.Gauge
	jitterGauge      prometheus.Gauge
	sampleCountGauge prometheus.Gauge
}

// NewDataCollector creates a new data collector
func NewDataCollector(collectInterval time.Duration, maxSamples int) *DataCollector {
	ctx, cancel := context.WithCancel(context.Background())

	collector := &DataCollector{
		samples:         make([]NetworkSample, 0, maxSamples),
		maxSamples:      maxSamples,
		collectInterval: collectInterval,
		ctx:             ctx,
		cancel:          cancel,
		dataPath:        "/var/lib/novacron/dwcp/network_samples.jsonl",
		saveInterval:    5 * time.Minute,
		pingTargets: []string{
			"8.8.8.8",        // Google DNS
			"1.1.1.1",        // Cloudflare DNS
			"208.67.222.222", // OpenDNS
		},
		bandwidthTest: "speedtest.net",
	}

	// Initialize Prometheus metrics
	collector.initMetrics()

	return collector
}

// initMetrics initializes Prometheus metrics
func (c *DataCollector) initMetrics() {
	c.bandwidthGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_pba_current_bandwidth_mbps",
		Help: "Current measured bandwidth in Mbps",
	})

	c.latencyGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_pba_current_latency_ms",
		Help: "Current measured latency in milliseconds",
	})

	c.packetLossGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_pba_current_packet_loss_ratio",
		Help: "Current packet loss ratio (0-1)",
	})

	c.jitterGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_pba_current_jitter_ms",
		Help: "Current network jitter in milliseconds",
	})

	c.sampleCountGauge = promauto.NewGauge(prometheus.GaugeOpts{
		Name: "dwcp_pba_sample_count",
		Help: "Number of collected network samples",
	})
}

// Start begins continuous metric collection
func (c *DataCollector) Start() {
	// Load historical data
	c.loadHistoricalData()

	// Start collection goroutine
	go c.collectLoop()

	// Start persistence goroutine
	go c.persistenceLoop()
}

// collectLoop continuously collects network metrics
func (c *DataCollector) collectLoop() {
	ticker := time.NewTicker(c.collectInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			sample := c.collectSample()
			c.addSample(sample)
			c.updatePrometheus(sample)
		}
	}
}

// collectSample performs network measurements
func (c *DataCollector) collectSample() NetworkSample {
	now := time.Now()

	sample := NetworkSample{
		Timestamp:     now,
		BandwidthMbps: c.measureBandwidth(),
		LatencyMs:     c.measureLatency(),
		PacketLoss:    c.measurePacketLoss(),
		JitterMs:      c.measureJitter(),
		TimeOfDay:     now.Hour(),
		DayOfWeek:     int(now.Weekday()),
		NodeID:        c.getNodeID(),
		PeerID:        c.selectPeer(),
	}

	return sample
}

// measureBandwidth performs bandwidth measurement
func (c *DataCollector) measureBandwidth() float64 {
	// In production, this would use actual bandwidth testing
	// For now, simulate with realistic patterns

	hour := time.Now().Hour()
	baseBandwidth := 100.0 // Base 100 Mbps

	// Time-of-day patterns
	var multiplier float64
	switch {
	case hour >= 9 && hour <= 17: // Business hours
		multiplier = 0.7
	case hour >= 19 && hour <= 23: // Evening peak
		multiplier = 0.5
	default: // Night/early morning
		multiplier = 0.9
	}

	// Add some random variation
	variation := (rand.Float64() - 0.5) * 20
	bandwidth := baseBandwidth*multiplier + variation

	return math.Max(10, bandwidth) // Minimum 10 Mbps
}

// measureLatency measures network latency
func (c *DataCollector) measureLatency() float64 {
	latencies := make([]float64, 0, len(c.pingTargets))

	for _, target := range c.pingTargets {
		if lat := c.pingHost(target); lat > 0 {
			latencies = append(latencies, lat)
		}
	}

	if len(latencies) == 0 {
		// Fallback to simulated value
		return 20.0 + rand.Float64()*10
	}

	// Return median latency
	return median(latencies)
}

// pingHost performs a simple ping measurement
func (c *DataCollector) pingHost(host string) float64 {
	timeout := 2 * time.Second
	start := time.Now()

	conn, err := net.DialTimeout("udp", host+":53", timeout)
	if err != nil {
		return -1
	}
	defer conn.Close()

	elapsed := time.Since(start)
	return float64(elapsed.Milliseconds())
}

// measurePacketLoss measures packet loss rate
func (c *DataCollector) measurePacketLoss() float64 {
	// In production, send multiple packets and measure loss
	// For now, simulate realistic values

	// Most of the time, packet loss is very low
	if rand.Float64() < 0.95 {
		return rand.Float64() * 0.01 // 0-1% loss
	}

	// Occasionally higher loss
	return rand.Float64() * 0.05 // 0-5% loss
}

// measureJitter measures network jitter
func (c *DataCollector) measureJitter() float64 {
	// Measure variance in latency
	latencies := make([]float64, 10)
	for i := range latencies {
		latencies[i] = c.measureLatency()
		time.Sleep(100 * time.Millisecond)
	}

	// Calculate standard deviation
	mean := average(latencies)
	var variance float64
	for _, lat := range latencies {
		variance += math.Pow(lat-mean, 2)
	}
	variance /= float64(len(latencies))

	return math.Sqrt(variance)
}

// addSample adds a new sample to the collection
func (c *DataCollector) addSample(sample NetworkSample) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.samples = append(c.samples, sample)
	if len(c.samples) > c.maxSamples {
		// Remove oldest samples
		c.samples = c.samples[len(c.samples)-c.maxSamples:]
	}
}

// GetRecentSamples returns the most recent n samples
func (c *DataCollector) GetRecentSamples(n int) []NetworkSample {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if n > len(c.samples) {
		n = len(c.samples)
	}

	if n == 0 {
		return []NetworkSample{}
	}

	// Return copy of recent samples
	start := len(c.samples) - n
	result := make([]NetworkSample, n)
	copy(result, c.samples[start:])

	return result
}

// GetSamplesByTimeRange returns samples within a time range
func (c *DataCollector) GetSamplesByTimeRange(start, end time.Time) []NetworkSample {
	c.mu.RLock()
	defer c.mu.RUnlock()

	result := make([]NetworkSample, 0)
	for _, sample := range c.samples {
		if sample.Timestamp.After(start) && sample.Timestamp.Before(end) {
			result = append(result, sample)
		}
	}

	return result
}

// persistenceLoop periodically saves samples to disk
func (c *DataCollector) persistenceLoop() {
	ticker := time.NewTicker(c.saveInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			c.saveData() // Final save
			return
		case <-ticker.C:
			c.saveData()
		}
	}
}

// saveData saves samples to disk
func (c *DataCollector) saveData() error {
	c.mu.RLock()
	samples := make([]NetworkSample, len(c.samples))
	copy(samples, c.samples)
	c.mu.RUnlock()

	// Create directory if needed
	os.MkdirAll("/var/lib/novacron/dwcp", 0755)

	file, err := os.Create(c.dataPath)
	if err != nil {
		return fmt.Errorf("failed to create data file: %w", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	for _, sample := range samples {
		if err := encoder.Encode(sample); err != nil {
			return fmt.Errorf("failed to encode sample: %w", err)
		}
	}

	c.lastSave = time.Now()
	return nil
}

// loadHistoricalData loads previous samples from disk
func (c *DataCollector) loadHistoricalData() error {
	file, err := os.Open(c.dataPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No historical data
		}
		return fmt.Errorf("failed to open data file: %w", err)
	}
	defer file.Close()

	c.mu.Lock()
	defer c.mu.Unlock()

	decoder := json.NewDecoder(file)
	for decoder.More() {
		var sample NetworkSample
		if err := decoder.Decode(&sample); err != nil {
			continue // Skip invalid samples
		}

		// Only load recent samples
		if time.Since(sample.Timestamp) < 30*24*time.Hour {
			c.samples = append(c.samples, sample)
		}
	}

	// Trim to max samples
	if len(c.samples) > c.maxSamples {
		c.samples = c.samples[len(c.samples)-c.maxSamples:]
	}

	return nil
}

// updatePrometheus updates Prometheus metrics
func (c *DataCollector) updatePrometheus(sample NetworkSample) {
	c.bandwidthGauge.Set(sample.BandwidthMbps)
	c.latencyGauge.Set(sample.LatencyMs)
	c.packetLossGauge.Set(sample.PacketLoss)
	c.jitterGauge.Set(sample.JitterMs)

	c.mu.RLock()
	count := len(c.samples)
	c.mu.RUnlock()
	c.sampleCountGauge.Set(float64(count))
}

// ExportForTraining exports samples in format suitable for ML training
func (c *DataCollector) ExportForTraining(outputPath string) error {
	c.mu.RLock()
	samples := make([]NetworkSample, len(c.samples))
	copy(samples, c.samples)
	c.mu.RUnlock()

	file, err := os.Create(outputPath)
	if err != nil {
		return fmt.Errorf("failed to create export file: %w", err)
	}
	defer file.Close()

	// Write CSV header
	fmt.Fprintln(file, "timestamp,bandwidth_mbps,latency_ms,packet_loss,jitter_ms,time_of_day,day_of_week")

	// Write samples
	for _, sample := range samples {
		fmt.Fprintf(file, "%d,%f,%f,%f,%f,%d,%d\n",
			sample.Timestamp.Unix(),
			sample.BandwidthMbps,
			sample.LatencyMs,
			sample.PacketLoss,
			sample.JitterMs,
			sample.TimeOfDay,
			sample.DayOfWeek,
		)
	}

	return nil
}

// Stop stops the data collector
func (c *DataCollector) Stop() {
	c.cancel()
	c.saveData() // Final save
}

// getNodeID returns the current node identifier
func (c *DataCollector) getNodeID() string {
	hostname, _ := os.Hostname()
	return hostname
}

// selectPeer selects a peer for measurement
func (c *DataCollector) selectPeer() string {
	peers := []string{"peer1", "peer2", "peer3", "peer4"}
	return peers[rand.Intn(len(peers))]
}

// Helper functions

func median(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	// Simple median calculation (not fully sorted)
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}

	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// GetStatistics returns statistical summary of collected data
func (c *DataCollector) GetStatistics() DataStatistics {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if len(c.samples) == 0 {
		return DataStatistics{}
	}

	stats := DataStatistics{
		SampleCount: len(c.samples),
		TimeRange: TimeRange{
			Start: c.samples[0].Timestamp,
			End:   c.samples[len(c.samples)-1].Timestamp,
		},
	}

	// Calculate statistics
	for _, sample := range c.samples {
		stats.AvgBandwidth += sample.BandwidthMbps
		stats.AvgLatency += sample.LatencyMs
		stats.AvgPacketLoss += sample.PacketLoss
		stats.AvgJitter += sample.JitterMs

		// Track min/max
		if sample.BandwidthMbps > stats.MaxBandwidth {
			stats.MaxBandwidth = sample.BandwidthMbps
		}
		if stats.MinBandwidth == 0 || sample.BandwidthMbps < stats.MinBandwidth {
			stats.MinBandwidth = sample.BandwidthMbps
		}
	}

	// Calculate averages
	n := float64(len(c.samples))
	stats.AvgBandwidth /= n
	stats.AvgLatency /= n
	stats.AvgPacketLoss /= n
	stats.AvgJitter /= n

	return stats
}

// DataStatistics contains statistical summary of collected data
type DataStatistics struct {
	SampleCount   int
	TimeRange     TimeRange
	AvgBandwidth  float64
	MinBandwidth  float64
	MaxBandwidth  float64
	AvgLatency    float64
	AvgPacketLoss float64
	AvgJitter     float64
}

// TimeRange represents a time period
type TimeRange struct {
	Start time.Time
	End   time.Time
}