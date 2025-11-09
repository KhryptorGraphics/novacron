package multiregion

import (
	"fmt"
	"sync"
	"time"
)

// NetworkTelemetry collects and exports network metrics
type NetworkTelemetry struct {
	collector *MetricsCollector
	exporter  *PrometheusExporter
	topology  *GlobalTopology
	stopChan  chan struct{}
	wg        sync.WaitGroup
}

// MetricsCollector collects network metrics
type MetricsCollector struct {
	metrics map[string]*Metric
	mu      sync.RWMutex
}

// Metric represents a telemetry metric
type Metric struct {
	Name      string
	Type      MetricType
	Value     float64
	Labels    map[string]string
	Timestamp time.Time
}

// MetricType defines the type of metric
type MetricType int

const (
	MetricGauge MetricType = iota
	MetricCounter
	MetricHistogram
	MetricSummary
)

// PrometheusExporter exports metrics in Prometheus format
type PrometheusExporter struct {
	metrics []string
	mu      sync.RWMutex
}

// LinkMetricSnapshot captures link metrics at a point in time
type LinkMetricSnapshot struct {
	LinkID     LinkID
	Latency    time.Duration
	Throughput int64
	PacketLoss float64
	Utilization float64
	Health     LinkHealth
	Timestamp  time.Time
}

// RegionMetricSnapshot captures region metrics
type RegionMetricSnapshot struct {
	RegionID        string
	ActiveInstances int
	CPUUtilization  float64
	MemoryUsage     int64
	NetworkIn       int64
	NetworkOut      int64
	Timestamp       time.Time
}

// NewNetworkTelemetry creates a new network telemetry system
func NewNetworkTelemetry(topology *GlobalTopology) *NetworkTelemetry {
	return &NetworkTelemetry{
		collector: &MetricsCollector{
			metrics: make(map[string]*Metric),
		},
		exporter: &PrometheusExporter{
			metrics: make([]string, 0),
		},
		topology: topology,
		stopChan: make(chan struct{}),
	}
}

// Start begins metric collection
func (nt *NetworkTelemetry) Start() {
	nt.wg.Add(1)
	go nt.collectMetrics()
}

// Stop stops metric collection
func (nt *NetworkTelemetry) Stop() {
	close(nt.stopChan)
	nt.wg.Wait()
}

// collectMetrics runs the metric collection loop
func (nt *NetworkTelemetry) collectMetrics() {
	defer nt.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			nt.collectLinkMetrics()
			nt.collectRegionMetrics()
		case <-nt.stopChan:
			return
		}
	}
}

// collectLinkMetrics collects metrics for all links
func (nt *NetworkTelemetry) collectLinkMetrics() {
	links := nt.topology.ListLinks()

	for _, link := range links {
		// Measure latency
		latency := nt.measureLatency(link)

		// Measure throughput
		throughput := nt.measureThroughput(link)

		// Measure packet loss
		packetLoss := nt.measurePacketLoss(link)

		// Create snapshot
		snapshot := &LinkMetricSnapshot{
			LinkID:      link.ID,
			Latency:     latency,
			Throughput:  throughput,
			PacketLoss:  packetLoss,
			Utilization: link.Utilization,
			Health:      link.Health,
			Timestamp:   time.Now(),
		}

		// Export to Prometheus
		nt.exporter.RecordLinkMetrics(snapshot)

		// Store in collector
		nt.collector.RecordLinkSnapshot(snapshot)
	}
}

// collectRegionMetrics collects metrics for all regions
func (nt *NetworkTelemetry) collectRegionMetrics() {
	regions := nt.topology.ListRegions()

	for _, region := range regions {
		snapshot := &RegionMetricSnapshot{
			RegionID:        region.ID,
			ActiveInstances: 0, // Would be populated from actual data
			CPUUtilization:  0.0,
			MemoryUsage:     0,
			NetworkIn:       0,
			NetworkOut:      0,
			Timestamp:       time.Now(),
		}

		nt.exporter.RecordRegionMetrics(snapshot)
		nt.collector.RecordRegionSnapshot(snapshot)
	}
}

// measureLatency measures link latency
func (nt *NetworkTelemetry) measureLatency(link *InterRegionLink) time.Duration {
	// In production, send ICMP echo or timestamp packets
	// For now, return the stored latency with some jitter

	link.mu.RLock()
	baseLatency := link.Latency
	link.mu.RUnlock()

	// Add up to ±10% jitter
	jitter := time.Duration(float64(baseLatency) * 0.1)
	return baseLatency + jitter/2 - jitter/4
}

// measureThroughput measures link throughput
func (nt *NetworkTelemetry) measureThroughput(link *InterRegionLink) int64 {
	// In production, measure actual data transfer rate
	// For now, calculate from utilization

	link.mu.RLock()
	throughput := link.Bandwidth * int64(link.Utilization) / 100
	link.mu.RUnlock()

	return throughput
}

// measurePacketLoss measures packet loss percentage
func (nt *NetworkTelemetry) measurePacketLoss(link *InterRegionLink) float64 {
	// In production, track sent vs received packets
	// For now, derive from link health

	link.mu.RLock()
	health := link.Health
	utilization := link.Utilization
	link.mu.RUnlock()

	switch health {
	case HealthUp:
		// Low utilization = low loss
		if utilization < 50 {
			return 0.1
		} else if utilization < 80 {
			return 0.5
		} else {
			return 1.0
		}
	case HealthDegraded:
		return 5.0
	case HealthDown:
		return 100.0
	default:
		return 0.0
	}
}

// MetricsCollector methods

// RecordLinkSnapshot records a link metric snapshot
func (mc *MetricsCollector) RecordLinkSnapshot(snapshot *LinkMetricSnapshot) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	prefix := fmt.Sprintf("link_%s", snapshot.LinkID)

	mc.metrics[prefix+"_latency"] = &Metric{
		Name:      "link_latency_ms",
		Type:      MetricGauge,
		Value:     float64(snapshot.Latency.Milliseconds()),
		Labels:    map[string]string{"link_id": string(snapshot.LinkID)},
		Timestamp: snapshot.Timestamp,
	}

	mc.metrics[prefix+"_throughput"] = &Metric{
		Name:      "link_throughput_mbps",
		Type:      MetricGauge,
		Value:     float64(snapshot.Throughput),
		Labels:    map[string]string{"link_id": string(snapshot.LinkID)},
		Timestamp: snapshot.Timestamp,
	}

	mc.metrics[prefix+"_packet_loss"] = &Metric{
		Name:      "link_packet_loss_percent",
		Type:      MetricGauge,
		Value:     snapshot.PacketLoss,
		Labels:    map[string]string{"link_id": string(snapshot.LinkID)},
		Timestamp: snapshot.Timestamp,
	}

	mc.metrics[prefix+"_utilization"] = &Metric{
		Name:      "link_utilization_percent",
		Type:      MetricGauge,
		Value:     snapshot.Utilization,
		Labels:    map[string]string{"link_id": string(snapshot.LinkID)},
		Timestamp: snapshot.Timestamp,
	}
}

// RecordRegionSnapshot records a region metric snapshot
func (mc *MetricsCollector) RecordRegionSnapshot(snapshot *RegionMetricSnapshot) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	prefix := fmt.Sprintf("region_%s", snapshot.RegionID)

	mc.metrics[prefix+"_instances"] = &Metric{
		Name:      "region_active_instances",
		Type:      MetricGauge,
		Value:     float64(snapshot.ActiveInstances),
		Labels:    map[string]string{"region_id": snapshot.RegionID},
		Timestamp: snapshot.Timestamp,
	}

	mc.metrics[prefix+"_cpu"] = &Metric{
		Name:      "region_cpu_utilization",
		Type:      MetricGauge,
		Value:     snapshot.CPUUtilization,
		Labels:    map[string]string{"region_id": snapshot.RegionID},
		Timestamp: snapshot.Timestamp,
	}

	mc.metrics[prefix+"_network_in"] = &Metric{
		Name:      "region_network_in_bytes",
		Type:      MetricCounter,
		Value:     float64(snapshot.NetworkIn),
		Labels:    map[string]string{"region_id": snapshot.RegionID},
		Timestamp: snapshot.Timestamp,
	}

	mc.metrics[prefix+"_network_out"] = &Metric{
		Name:      "region_network_out_bytes",
		Type:      MetricCounter,
		Value:     float64(snapshot.NetworkOut),
		Labels:    map[string]string{"region_id": snapshot.RegionID},
		Timestamp: snapshot.Timestamp,
	}
}

// GetMetric retrieves a specific metric
func (mc *MetricsCollector) GetMetric(name string) (*Metric, bool) {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	metric, exists := mc.metrics[name]
	return metric, exists
}

// GetAllMetrics returns all collected metrics
func (mc *MetricsCollector) GetAllMetrics() []*Metric {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	metrics := make([]*Metric, 0, len(mc.metrics))
	for _, metric := range mc.metrics {
		metrics = append(metrics, metric)
	}

	return metrics
}

// PrometheusExporter methods

// RecordLinkMetrics records link metrics in Prometheus format
func (pe *PrometheusExporter) RecordLinkMetrics(snapshot *LinkMetricSnapshot) {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	linkIDStr := string(snapshot.LinkID)

	// Latency metric
	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_link_latency_milliseconds{link_id="%s"} %d`,
		linkIDStr, snapshot.Latency.Milliseconds(),
	))

	// Throughput metric
	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_link_throughput_mbps{link_id="%s"} %d`,
		linkIDStr, snapshot.Throughput,
	))

	// Packet loss metric
	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_link_packet_loss_percent{link_id="%s"} %.2f`,
		linkIDStr, snapshot.PacketLoss,
	))

	// Utilization metric
	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_link_utilization_percent{link_id="%s"} %.2f`,
		linkIDStr, snapshot.Utilization,
	))

	// Health metric (0=down, 1=degraded, 2=up)
	healthValue := 0
	switch snapshot.Health {
	case HealthUp:
		healthValue = 2
	case HealthDegraded:
		healthValue = 1
	case HealthDown:
		healthValue = 0
	}

	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_link_health{link_id="%s"} %d`,
		linkIDStr, healthValue,
	))
}

// RecordRegionMetrics records region metrics in Prometheus format
func (pe *PrometheusExporter) RecordRegionMetrics(snapshot *RegionMetricSnapshot) {
	pe.mu.Lock()
	defer pe.mu.Unlock()

	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_region_active_instances{region_id="%s"} %d`,
		snapshot.RegionID, snapshot.ActiveInstances,
	))

	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_region_cpu_utilization{region_id="%s"} %.2f`,
		snapshot.RegionID, snapshot.CPUUtilization,
	))

	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_region_network_in_bytes{region_id="%s"} %d`,
		snapshot.RegionID, snapshot.NetworkIn,
	))

	pe.metrics = append(pe.metrics, fmt.Sprintf(
		`dwcp_region_network_out_bytes{region_id="%s"} %d`,
		snapshot.RegionID, snapshot.NetworkOut,
	))
}

// GetMetrics returns all metrics in Prometheus format
func (pe *PrometheusExporter) GetMetrics() []string {
	pe.mu.RLock()
	defer pe.mu.RUnlock()

	metrics := make([]string, len(pe.metrics))
	copy(metrics, pe.metrics)
	return metrics
}

// ClearMetrics clears all metrics
func (pe *PrometheusExporter) ClearMetrics() {
	pe.mu.Lock()
	defer pe.mu.Unlock()
	pe.metrics = make([]string, 0)
}

// String methods

func (mt MetricType) String() string {
	switch mt {
	case MetricGauge:
		return "gauge"
	case MetricCounter:
		return "counter"
	case MetricHistogram:
		return "histogram"
	case MetricSummary:
		return "summary"
	default:
		return "unknown"
	}
}
