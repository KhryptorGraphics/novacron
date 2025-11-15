package observability

import (
	"context"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ConsensusMetricsCollector collects consensus-related metrics
type ConsensusMetricsCollector struct {
	mu     sync.RWMutex
	logger *zap.Logger

	// Consensus metrics
	latencyMeasurements []float64
	roundCount          int64
	timeoutCount        int64
	byzantineEvents     int64

	// Recent measurements (circular buffer)
	recentLatencies []float64
	bufferSize      int
	bufferIndex     int
}

// NewConsensusMetricsCollector creates a new consensus metrics collector
func NewConsensusMetricsCollector(logger *zap.Logger) *ConsensusMetricsCollector {
	return &ConsensusMetricsCollector{
		logger:          logger,
		recentLatencies: make([]float64, 1000),
		bufferSize:      1000,
	}
}

// RecordConsensusRound records a consensus round completion
func (cmc *ConsensusMetricsCollector) RecordConsensusRound(latencyMs float64) {
	cmc.mu.Lock()
	defer cmc.mu.Unlock()

	cmc.latencyMeasurements = append(cmc.latencyMeasurements, latencyMs)
	cmc.roundCount++

	// Update circular buffer
	cmc.recentLatencies[cmc.bufferIndex] = latencyMs
	cmc.bufferIndex = (cmc.bufferIndex + 1) % cmc.bufferSize
}

// RecordTimeout records a consensus timeout
func (cmc *ConsensusMetricsCollector) RecordTimeout() {
	cmc.mu.Lock()
	defer cmc.mu.Unlock()

	cmc.timeoutCount++
}

// RecordByzantineEvent records a Byzantine fault detection
func (cmc *ConsensusMetricsCollector) RecordByzantineEvent() {
	cmc.mu.Lock()
	defer cmc.mu.Unlock()

	cmc.byzantineEvents++
}

// CollectMetrics implements MetricCollector interface
func (cmc *ConsensusMetricsCollector) CollectMetrics(ctx context.Context) ([]Metric, error) {
	cmc.mu.RLock()
	defer cmc.mu.RUnlock()

	now := time.Now()
	metrics := make([]Metric, 0)

	// Consensus latency metrics
	if len(cmc.latencyMeasurements) > 0 {
		metrics = append(metrics, Metric{
			Name:      "consensus_latency",
			Type:      MetricTypeHistogram,
			Value:     cmc.calculateP95(),
			Labels:    map[string]string{"percentile": "p95"},
			Timestamp: now,
		})

		metrics = append(metrics, Metric{
			Name:      "consensus_latency",
			Type:      MetricTypeHistogram,
			Value:     cmc.calculateP99(),
			Labels:    map[string]string{"percentile": "p99"},
			Timestamp: now,
		})
	}

	// Consensus round count
	metrics = append(metrics, Metric{
		Name:      "consensus_rounds_total",
		Type:      MetricTypeCounter,
		Value:     float64(cmc.roundCount),
		Labels:    map[string]string{},
		Timestamp: now,
	})

	// Timeout count
	metrics = append(metrics, Metric{
		Name:      "consensus_timeouts_total",
		Type:      MetricTypeCounter,
		Value:     float64(cmc.timeoutCount),
		Labels:    map[string]string{},
		Timestamp: now,
	})

	// Byzantine events
	metrics = append(metrics, Metric{
		Name:      "consensus_byzantine_events_total",
		Type:      MetricTypeCounter,
		Value:     float64(cmc.byzantineEvents),
		Labels:    map[string]string{},
		Timestamp: now,
	})

	return metrics, nil
}

// GetName implements MetricCollector interface
func (cmc *ConsensusMetricsCollector) GetName() string {
	return "consensus"
}

// calculateP95 calculates 95th percentile from recent measurements
func (cmc *ConsensusMetricsCollector) calculateP95() float64 {
	return calculatePercentile(cmc.recentLatencies, 0.95)
}

// calculateP99 calculates 99th percentile from recent measurements
func (cmc *ConsensusMetricsCollector) calculateP99() float64 {
	return calculatePercentile(cmc.recentLatencies, 0.99)
}

// Reset clears all measurements
func (cmc *ConsensusMetricsCollector) Reset() {
	cmc.mu.Lock()
	defer cmc.mu.Unlock()

	cmc.latencyMeasurements = nil
	cmc.roundCount = 0
	cmc.timeoutCount = 0
	cmc.byzantineEvents = 0
	cmc.recentLatencies = make([]float64, cmc.bufferSize)
	cmc.bufferIndex = 0
}
