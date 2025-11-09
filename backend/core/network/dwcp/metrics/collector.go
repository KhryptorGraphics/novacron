// Package metrics provides metric collection logic for DWCP components
package metrics

import (
	"sync"
	"time"
)

// Collector aggregates metrics from DWCP components
type Collector struct {
	mu            sync.RWMutex
	cluster       string
	node          string

	// AMST tracking
	activeStreams map[string]*StreamMetrics

	// HDE tracking
	deltaHits     int64
	deltaMisses   int64
	baselineCount int

	// Bandwidth tracking
	totalBandwidth      int64
	availableBandwidth  int64

	// Migration tracking
	migrationStartTimes map[string]time.Time
}

// StreamMetrics tracks individual stream metrics
type StreamMetrics struct {
	ID           string
	StartTime    time.Time
	BytesSent    int64
	BytesReceived int64
	LastActivity time.Time
}

// NewCollector creates a new metrics collector
func NewCollector(cluster, node string) *Collector {
	return &Collector{
		cluster:             cluster,
		node:                node,
		activeStreams:       make(map[string]*StreamMetrics),
		migrationStartTimes: make(map[string]time.Time),
	}
}

// StartStream begins tracking a new AMST stream
func (c *Collector) StartStream(streamID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.activeStreams[streamID] = &StreamMetrics{
		ID:        streamID,
		StartTime: time.Now(),
	}

	RecordAMSTStreamStart(c.cluster, c.node, true)
}

// EndStream stops tracking an AMST stream
func (c *Collector) EndStream(streamID string) {
	c.mu.Lock()
	stream, exists := c.activeStreams[streamID]
	delete(c.activeStreams, streamID)
	c.mu.Unlock()

	if exists {
		// Record final metrics
		RecordAMSTData(c.cluster, c.node, streamID, stream.BytesSent, stream.BytesReceived)

		// Record latency
		duration := time.Since(stream.StartTime).Seconds()
		AMSTLatency.WithLabelValues(c.cluster, c.node, "stream_duration").Observe(duration)

		RecordAMSTStreamEnd(c.cluster, c.node)
	}
}

// RecordStreamData updates stream data counters
func (c *Collector) RecordStreamData(streamID string, bytesSent, bytesReceived int64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if stream, exists := c.activeStreams[streamID]; exists {
		stream.BytesSent += bytesSent
		stream.BytesReceived += bytesReceived
		stream.LastActivity = time.Now()

		// Update metrics
		RecordAMSTData(c.cluster, c.node, streamID, bytesSent, bytesReceived)
	}
}

// RecordStreamError records a stream error
func (c *Collector) RecordStreamError(streamID, errorType string) {
	RecordAMSTError(c.cluster, c.node, errorType)
}

// RecordCompressionOperation records an HDE compression
func (c *Collector) RecordCompressionOperation(dataType string, originalBytes, compressedBytes int64, deltaHit bool) {
	c.mu.Lock()
	if deltaHit {
		c.deltaHits++
	} else {
		c.deltaMisses++
	}

	// Calculate hit rate
	totalAttempts := c.deltaHits + c.deltaMisses
	hitRate := 0.0
	if totalAttempts > 0 {
		hitRate = float64(c.deltaHits) / float64(totalAttempts) * 100
	}
	c.mu.Unlock()

	// Record compression metrics
	RecordHDECompression(c.cluster, c.node, dataType, originalBytes, compressedBytes, true)

	// Update hit rate
	HDEDeltaHitRate.WithLabelValues(c.cluster, c.node).Set(hitRate)
}

// RecordDecompressionOperation records an HDE decompression
func (c *Collector) RecordDecompressionOperation(success bool) {
	RecordHDEDecompression(c.cluster, c.node, success)
}

// UpdateBaselineCount updates the number of active baselines
func (c *Collector) UpdateBaselineCount(baselineType string, count int) {
	c.mu.Lock()
	c.baselineCount = count
	c.mu.Unlock()

	HDEBaselineCount.WithLabelValues(c.cluster, c.node, baselineType).Set(float64(count))
}

// UpdateBandwidthUtilization updates bandwidth metrics
func (c *Collector) UpdateBandwidthUtilization(used, available int64) {
	c.mu.Lock()
	c.totalBandwidth = used
	c.availableBandwidth = available
	c.mu.Unlock()

	utilization := 0.0
	if available > 0 {
		utilization = float64(used) / float64(available) * 100
	}

	AMSTBandwidthUtilization.WithLabelValues(c.cluster, c.node).Set(utilization)
}

// StartMigration begins tracking a VM migration
func (c *Collector) StartMigration(migrationID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.migrationStartTimes[migrationID] = time.Now()
}

// EndMigration completes migration tracking
func (c *Collector) EndMigration(migrationID, destNode string, dwcpEnabled bool) {
	c.mu.Lock()
	startTime, exists := c.migrationStartTimes[migrationID]
	delete(c.migrationStartTimes, migrationID)
	c.mu.Unlock()

	if exists {
		duration := time.Since(startTime).Seconds()
		RecordMigration(c.cluster, c.node, destNode, duration, dwcpEnabled)
	}
}

// RecordFederationSync records a federation sync operation
func (c *Collector) RecordFederationSync(remoteCluster, syncType string, duration time.Duration, bandwidthSaved int64) {
	FederationSyncDuration.WithLabelValues(c.cluster, remoteCluster, syncType).Observe(duration.Seconds())

	if bandwidthSaved > 0 {
		FederationBandwidthSaved.WithLabelValues(c.cluster, remoteCluster).Add(float64(bandwidthSaved))
	}
}

// UpdateComponentHealth updates component health status
func (c *Collector) UpdateComponentHealth(component string, status HealthStatus) {
	SetComponentHealth(c.cluster, c.node, component, status)
}

// UpdateFeatureStatus updates feature enabled status
func (c *Collector) UpdateFeatureStatus(feature string, enabled bool) {
	SetFeatureEnabled(c.cluster, c.node, feature, enabled)
}

// RecordOperationLatency records operation latency
func (c *Collector) RecordOperationLatency(operation string, duration time.Duration) {
	AMSTLatency.WithLabelValues(c.cluster, c.node, operation).Observe(duration.Seconds())
}

// GetActiveStreamCount returns the number of active streams
func (c *Collector) GetActiveStreamCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.activeStreams)
}

// GetDeltaHitRate returns the current delta hit rate
func (c *Collector) GetDeltaHitRate() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	totalAttempts := c.deltaHits + c.deltaMisses
	if totalAttempts == 0 {
		return 0.0
	}
	return float64(c.deltaHits) / float64(totalAttempts) * 100
}

// UpdateDictionaryEfficiency updates dictionary compression efficiency
func (c *Collector) UpdateDictionaryEfficiency(efficiency float64) {
	HDEDictionaryEfficiency.WithLabelValues(c.cluster, c.node).Set(efficiency)
}

// UpdateSpeedupFactor updates migration speedup factor
func (c *Collector) UpdateSpeedupFactor(vmType string, factor float64) {
	MigrationSpeedupFactor.WithLabelValues(c.cluster, vmType).Set(factor)
}
