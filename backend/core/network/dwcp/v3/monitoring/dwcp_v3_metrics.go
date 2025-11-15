package monitoring

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"
)

// DWCPv3MetricsCollector collects comprehensive metrics for DWCP v3 components
type DWCPv3MetricsCollector struct {
	mu sync.RWMutex

	// Configuration
	nodeID           string
	clusterID        string
	enablePrometheus bool
	logger           *zap.Logger

	// Mode-specific metrics
	datacenterMetrics *ModeMetrics
	internetMetrics   *ModeMetrics
	hybridMetrics     *ModeMetrics
	currentMode       atomic.Value // upgrade.NetworkMode

	// Component-specific metrics
	amstMetrics *AMSTMetrics
	hdeMetrics  *HDEMetrics
	pbaMetrics  *PBAMetrics
	assMetrics  *ASSMetrics
	acpMetrics  *ACPMetrics
	itpMetrics  *ITPMetrics

	// Global performance counters
	totalMigrations       atomic.Uint64
	successfulMigrations  atomic.Uint64
	failedMigrations      atomic.Uint64
	totalBytesTransferred atomic.Uint64
	totalModeSwitches     atomic.Uint64

	// Prometheus metrics
	promMetrics *PrometheusMetrics

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
}

// ModeMetrics tracks mode-specific performance
type ModeMetrics struct {
	mu sync.RWMutex

	Mode string

	// RDMA metrics (datacenter)
	RDMAThroughput   float64 // Gbps
	RDMALatency      float64 // microseconds
	RDMABandwidthUse float64 // percentage

	// TCP metrics (internet)
	TCPStreams          int32
	TCPCompressionRatio float64
	TCPPacketLoss       float64 // percentage
	ByzantineEvents     int64

	// Common metrics
	AverageThroughput float64 // Mbps
	AverageLatency    float64 // milliseconds
	P95Latency        float64 // milliseconds
	P99Latency        float64 // milliseconds
	ErrorRate         float64 // percentage

	// Historical data for percentile calculations
	latencyHistogram []float64
	maxHistogramSize int
}

// AMSTMetrics tracks Adaptive Multi-Stream Transport metrics
type AMSTMetrics struct {
	mu sync.RWMutex

	ActiveStreams    int32
	TotalStreams     int32
	StreamEfficiency float64 // percentage
	BytesSent        uint64
	BytesReceived    uint64
	FailedStreams    int64
	ModeSwitches     int64
	CongestionEvents int64

	// Per-mode stream counts
	DatacenterStreamCount int32
	InternetStreamCount   int32
}

// HDEMetrics tracks Hierarchical Delta Encoding metrics
type HDEMetrics struct {
	mu sync.RWMutex

	CompressionsTotal      int64
	DecompressionsTotal    int64
	BytesOriginal          int64
	BytesCompressed        int64
	CompressionRatio       float64
	DeltaHitRate           float64 // percentage
	MLSelectionAccuracy    float64 // percentage
	CRDTMerges             int64
	CRDTConflicts          int64
	AverageCompressionTime time.Duration

	// Algorithm distribution
	AlgorithmUsage map[string]int64
}

// PBAMetrics tracks Predictive Bandwidth Allocation metrics
type PBAMetrics struct {
	mu sync.RWMutex

	TotalPredictions      int64
	DatacenterPredictions int64
	InternetPredictions   int64
	PredictionAccuracy    float64 // percentage
	AvgPredictionLatency  time.Duration
	MaxPredictionLatency  time.Duration
	PredictionErrors      int64
	ModelSwitches         int64
}

// ASSMetrics tracks Adaptive State Synchronization metrics
type ASSMetrics struct {
	mu sync.RWMutex

	SyncOperations      int64
	SuccessfulSyncs     int64
	FailedSyncs         int64
	AvgSyncLatency      time.Duration
	ConflictResolutions int64
	CRDTOperations      int64
	FullSyncs           int64
	IncrementalSyncs    int64
}

// ACPMetrics tracks Adaptive Consensus Protocol metrics
type ACPMetrics struct {
	mu sync.RWMutex

	ConsensusOperations int64
	RaftConsensus       int64
	PBFTConsensus       int64
	AvgConsensusTime    time.Duration
	ConsensusTimeouts   int64
	ByzantineDetections int64
	Failovers           int64
	LeaderChanges       int64
}

// ITPMetrics tracks Intelligent Task Placement metrics
type ITPMetrics struct {
	mu sync.RWMutex

	PlacementDecisions      int64
	OptimalPlacements       int64
	SuboptimalPlacements    int64
	PlacementScore          float64 // 0-100
	MLPredictionAccuracy    float64 // percentage
	GeographicOptimizations int64
	CostOptimizations       int64
	AvgPlacementTime        time.Duration
}

// PrometheusMetrics wraps Prometheus metrics
type PrometheusMetrics struct {
	// Mode metrics
	ModeSwitches   *prometheus.CounterVec
	ModeLatency    *prometheus.HistogramVec
	ModeThroughput *prometheus.GaugeVec

	// Component metrics
	ComponentOperations *prometheus.CounterVec
	ComponentLatency    *prometheus.HistogramVec
	ComponentErrors     *prometheus.CounterVec

	// Transfer metrics
	BytesTransferred  *prometheus.CounterVec
	MigrationDuration *prometheus.HistogramVec
	MigrationSuccess  *prometheus.CounterVec

	// Resource metrics
	ActiveStreams      *prometheus.GaugeVec
	CompressionRatio   *prometheus.GaugeVec
	PredictionAccuracy *prometheus.GaugeVec
}

// NewDWCPv3MetricsCollector creates a new metrics collector
func NewDWCPv3MetricsCollector(nodeID, clusterID string, enablePrometheus bool, logger *zap.Logger) *DWCPv3MetricsCollector {
	ctx, cancel := context.WithCancel(context.Background())

	collector := &DWCPv3MetricsCollector{
		nodeID:            nodeID,
		clusterID:         clusterID,
		enablePrometheus:  enablePrometheus,
		logger:            logger,
		datacenterMetrics: newModeMetrics("datacenter"),
		internetMetrics:   newModeMetrics("internet"),
		hybridMetrics:     newModeMetrics("hybrid"),
		amstMetrics:       &AMSTMetrics{},
		hdeMetrics:        &HDEMetrics{AlgorithmUsage: make(map[string]int64)},
		pbaMetrics:        &PBAMetrics{},
		assMetrics:        &ASSMetrics{},
		acpMetrics:        &ACPMetrics{},
		itpMetrics:        &ITPMetrics{},
		ctx:               ctx,
		cancel:            cancel,
	}

	collector.currentMode.Store(upgrade.ModeHybrid)

	// Initialize Prometheus metrics if enabled
	if enablePrometheus {
		collector.promMetrics = collector.initPrometheusMetrics()
	}

	// Start background metric aggregation
	go collector.aggregateMetricsLoop()

	return collector
}

// initPrometheusMetrics initializes Prometheus metrics
func (c *DWCPv3MetricsCollector) initPrometheusMetrics() *PrometheusMetrics {
	return &PrometheusMetrics{
		ModeSwitches: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "dwcp_v3_mode_switches_total",
				Help: "Total number of network mode switches",
			},
			[]string{"node_id", "from_mode", "to_mode"},
		),
		ModeLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "dwcp_v3_mode_latency_seconds",
				Help:    "Operation latency by network mode",
				Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to 16s
			},
			[]string{"node_id", "mode", "operation"},
		),
		ModeThroughput: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "dwcp_v3_mode_throughput_mbps",
				Help: "Current throughput in Mbps by mode",
			},
			[]string{"node_id", "mode"},
		),
		ComponentOperations: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "dwcp_v3_component_operations_total",
				Help: "Total operations by component",
			},
			[]string{"node_id", "component", "operation_type"},
		),
		ComponentLatency: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "dwcp_v3_component_latency_seconds",
				Help:    "Component operation latency",
				Buckets: prometheus.ExponentialBuckets(0.0001, 2, 15), // 0.1ms to 1.6s
			},
			[]string{"node_id", "component"},
		),
		ComponentErrors: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "dwcp_v3_component_errors_total",
				Help: "Total errors by component",
			},
			[]string{"node_id", "component", "error_type"},
		),
		BytesTransferred: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "dwcp_v3_bytes_transferred_total",
				Help: "Total bytes transferred",
			},
			[]string{"node_id", "mode", "direction"},
		),
		MigrationDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "dwcp_v3_migration_duration_seconds",
				Help:    "VM migration duration",
				Buckets: prometheus.ExponentialBuckets(1, 2, 12), // 1s to 68 minutes
			},
			[]string{"node_id", "mode"},
		),
		MigrationSuccess: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "dwcp_v3_migration_success_total",
				Help: "Successful migrations",
			},
			[]string{"node_id", "mode"},
		),
		ActiveStreams: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "dwcp_v3_active_streams",
				Help: "Current active streams",
			},
			[]string{"node_id", "mode"},
		),
		CompressionRatio: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "dwcp_v3_compression_ratio",
				Help: "Current compression ratio",
			},
			[]string{"node_id", "algorithm"},
		),
		PredictionAccuracy: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "dwcp_v3_prediction_accuracy",
				Help: "Bandwidth prediction accuracy percentage",
			},
			[]string{"node_id", "mode"},
		),
	}
}

// RecordModeSwitch records a network mode switch
func (c *DWCPv3MetricsCollector) RecordModeSwitch(fromMode, toMode upgrade.NetworkMode) {
	c.totalModeSwitches.Add(1)
	c.currentMode.Store(toMode)

	if c.enablePrometheus {
		c.promMetrics.ModeSwitches.WithLabelValues(
			c.nodeID,
			fromMode.String(),
			toMode.String(),
		).Inc()
	}

	c.logger.Info("Network mode switch recorded",
		zap.String("from", fromMode.String()),
		zap.String("to", toMode.String()),
		zap.Uint64("total_switches", c.totalModeSwitches.Load()))
}

// RecordAMSTMetric records AMST transport metrics
func (c *DWCPv3MetricsCollector) RecordAMSTMetric(metric string, value interface{}) {
	c.amstMetrics.mu.Lock()
	defer c.amstMetrics.mu.Unlock()

	switch metric {
	case "stream_created":
		c.amstMetrics.TotalStreams++
		c.amstMetrics.ActiveStreams++
	case "stream_closed":
		if c.amstMetrics.ActiveStreams > 0 {
			c.amstMetrics.ActiveStreams--
		}
	case "bytes_sent":
		if bytes, ok := value.(uint64); ok {
			c.amstMetrics.BytesSent += bytes
			c.totalBytesTransferred.Add(bytes)
		}
	case "mode_switch":
		c.amstMetrics.ModeSwitches++
	case "congestion_event":
		c.amstMetrics.CongestionEvents++
	}

	// Update Prometheus
	if c.enablePrometheus {
		mode := c.currentMode.Load().(upgrade.NetworkMode)
		c.promMetrics.ActiveStreams.WithLabelValues(c.nodeID, mode.String()).Set(float64(c.amstMetrics.ActiveStreams))
		c.promMetrics.ComponentOperations.WithLabelValues(c.nodeID, "amst", metric).Inc()
	}
}

// RecordHDEMetric records HDE compression metrics
func (c *DWCPv3MetricsCollector) RecordHDEMetric(metric string, value interface{}) {
	c.hdeMetrics.mu.Lock()
	defer c.hdeMetrics.mu.Unlock()

	switch metric {
	case "compression":
		c.hdeMetrics.CompressionsTotal++
	case "decompression":
		c.hdeMetrics.DecompressionsTotal++
	case "compression_ratio":
		if ratio, ok := value.(float64); ok {
			c.hdeMetrics.CompressionRatio = ratio
		}
	case "delta_hit":
		// Update hit rate (exponential moving average)
		currentHitRate := c.hdeMetrics.DeltaHitRate
		c.hdeMetrics.DeltaHitRate = currentHitRate*0.9 + 10.0 // +10% for hit
	case "algorithm_used":
		if algo, ok := value.(string); ok {
			c.hdeMetrics.AlgorithmUsage[algo]++
		}
	case "crdt_merge":
		c.hdeMetrics.CRDTMerges++
	}

	// Update Prometheus
	if c.enablePrometheus {
		c.promMetrics.ComponentOperations.WithLabelValues(c.nodeID, "hde", metric).Inc()
		if c.hdeMetrics.CompressionRatio > 0 {
			c.promMetrics.CompressionRatio.WithLabelValues(c.nodeID, "zstd").Set(c.hdeMetrics.CompressionRatio)
		}
	}
}

// RecordPBAMetric records PBA prediction metrics
func (c *DWCPv3MetricsCollector) RecordPBAMetric(metric string, value interface{}) {
	c.pbaMetrics.mu.Lock()
	defer c.pbaMetrics.mu.Unlock()

	switch metric {
	case "prediction":
		c.pbaMetrics.TotalPredictions++
		mode := c.currentMode.Load().(upgrade.NetworkMode)
		if mode == upgrade.ModeDatacenter {
			c.pbaMetrics.DatacenterPredictions++
		} else if mode == upgrade.ModeInternet {
			c.pbaMetrics.InternetPredictions++
		}
	case "prediction_accuracy":
		if accuracy, ok := value.(float64); ok {
			// Exponential moving average
			c.pbaMetrics.PredictionAccuracy = c.pbaMetrics.PredictionAccuracy*0.9 + accuracy*0.1
		}
	case "prediction_latency":
		if latency, ok := value.(time.Duration); ok {
			if latency > c.pbaMetrics.MaxPredictionLatency {
				c.pbaMetrics.MaxPredictionLatency = latency
			}
			// Update average
			total := c.pbaMetrics.TotalPredictions
			if total > 0 {
				c.pbaMetrics.AvgPredictionLatency = time.Duration(
					(c.pbaMetrics.AvgPredictionLatency.Nanoseconds()*int64(total-1) + latency.Nanoseconds()) / int64(total),
				)
			}
		}
	}

	// Update Prometheus
	if c.enablePrometheus {
		mode := c.currentMode.Load().(upgrade.NetworkMode)
		c.promMetrics.ComponentOperations.WithLabelValues(c.nodeID, "pba", metric).Inc()
		c.promMetrics.PredictionAccuracy.WithLabelValues(c.nodeID, mode.String()).Set(c.pbaMetrics.PredictionAccuracy)
	}
}

// RecordASSMetric records ASS synchronization metrics
func (c *DWCPv3MetricsCollector) RecordASSMetric(metric string, value interface{}) {
	c.assMetrics.mu.Lock()
	defer c.assMetrics.mu.Unlock()

	switch metric {
	case "sync_start":
		c.assMetrics.SyncOperations++
	case "sync_success":
		c.assMetrics.SuccessfulSyncs++
	case "sync_failed":
		c.assMetrics.FailedSyncs++
	case "conflict_resolved":
		c.assMetrics.ConflictResolutions++
	case "full_sync":
		c.assMetrics.FullSyncs++
	case "incremental_sync":
		c.assMetrics.IncrementalSyncs++
	}

	// Update Prometheus
	if c.enablePrometheus {
		c.promMetrics.ComponentOperations.WithLabelValues(c.nodeID, "ass", metric).Inc()
	}
}

// RecordACPMetric records ACP consensus metrics
func (c *DWCPv3MetricsCollector) RecordACPMetric(metric string, value interface{}) {
	c.acpMetrics.mu.Lock()
	defer c.acpMetrics.mu.Unlock()

	switch metric {
	case "consensus_start":
		c.acpMetrics.ConsensusOperations++
	case "raft_consensus":
		c.acpMetrics.RaftConsensus++
	case "pbft_consensus":
		c.acpMetrics.PBFTConsensus++
	case "byzantine_detected":
		c.acpMetrics.ByzantineDetections++
	case "failover":
		c.acpMetrics.Failovers++
	case "consensus_latency":
		if latency, ok := value.(time.Duration); ok {
			total := c.acpMetrics.ConsensusOperations
			if total > 0 {
				c.acpMetrics.AvgConsensusTime = time.Duration(
					(c.acpMetrics.AvgConsensusTime.Nanoseconds()*int64(total-1) + latency.Nanoseconds()) / int64(total),
				)
			}
		}
	}

	// Update Prometheus
	if c.enablePrometheus {
		c.promMetrics.ComponentOperations.WithLabelValues(c.nodeID, "acp", metric).Inc()
	}
}

// RecordITPMetric records ITP placement metrics
func (c *DWCPv3MetricsCollector) RecordITPMetric(metric string, value interface{}) {
	c.itpMetrics.mu.Lock()
	defer c.itpMetrics.mu.Unlock()

	switch metric {
	case "placement_decision":
		c.itpMetrics.PlacementDecisions++
	case "optimal_placement":
		c.itpMetrics.OptimalPlacements++
	case "suboptimal_placement":
		c.itpMetrics.SuboptimalPlacements++
	case "placement_score":
		if score, ok := value.(float64); ok {
			c.itpMetrics.PlacementScore = score
		}
	case "ml_accuracy":
		if accuracy, ok := value.(float64); ok {
			c.itpMetrics.MLPredictionAccuracy = accuracy
		}
	}

	// Update Prometheus
	if c.enablePrometheus {
		c.promMetrics.ComponentOperations.WithLabelValues(c.nodeID, "itp", metric).Inc()
	}
}

// RecordMigration records a migration event
func (c *DWCPv3MetricsCollector) RecordMigration(success bool, duration time.Duration, bytesTransferred uint64) {
	c.totalMigrations.Add(1)
	if success {
		c.successfulMigrations.Add(1)
	} else {
		c.failedMigrations.Add(1)
	}

	mode := c.currentMode.Load().(upgrade.NetworkMode)

	if c.enablePrometheus {
		if success {
			c.promMetrics.MigrationSuccess.WithLabelValues(c.nodeID, mode.String()).Inc()
		}
		c.promMetrics.MigrationDuration.WithLabelValues(c.nodeID, mode.String()).Observe(duration.Seconds())
		c.promMetrics.BytesTransferred.WithLabelValues(c.nodeID, mode.String(), "sent").Add(float64(bytesTransferred))
	}

	c.logger.Debug("Migration recorded",
		zap.Bool("success", success),
		zap.Duration("duration", duration),
		zap.Uint64("bytes", bytesTransferred),
		zap.String("mode", mode.String()))
}

// GetComprehensiveMetrics returns all metrics as a structured map
func (c *DWCPv3MetricsCollector) GetComprehensiveMetrics() map[string]interface{} {
	mode := c.currentMode.Load().(upgrade.NetworkMode)

	metrics := map[string]interface{}{
		"node_id":    c.nodeID,
		"cluster_id": c.clusterID,
		"mode":       mode.String(),
		"global": map[string]interface{}{
			"total_migrations":        c.totalMigrations.Load(),
			"successful_migrations":   c.successfulMigrations.Load(),
			"failed_migrations":       c.failedMigrations.Load(),
			"total_bytes_transferred": c.totalBytesTransferred.Load(),
			"total_mode_switches":     c.totalModeSwitches.Load(),
			"success_rate":            c.calculateSuccessRate(),
		},
	}

	// Add component metrics
	metrics["amst"] = c.getAMSTMetrics()
	metrics["hde"] = c.getHDEMetrics()
	metrics["pba"] = c.getPBAMetrics()
	metrics["ass"] = c.getASSMetrics()
	metrics["acp"] = c.getACPMetrics()
	metrics["itp"] = c.getITPMetrics()

	// Add mode-specific metrics
	metrics["datacenter_mode"] = c.getModeMetrics(c.datacenterMetrics)
	metrics["internet_mode"] = c.getModeMetrics(c.internetMetrics)
	metrics["hybrid_mode"] = c.getModeMetrics(c.hybridMetrics)

	return metrics
}

// Helper methods to get component metrics
func (c *DWCPv3MetricsCollector) getAMSTMetrics() map[string]interface{} {
	c.amstMetrics.mu.RLock()
	defer c.amstMetrics.mu.RUnlock()
	return map[string]interface{}{
		"active_streams":    c.amstMetrics.ActiveStreams,
		"total_streams":     c.amstMetrics.TotalStreams,
		"bytes_sent":        c.amstMetrics.BytesSent,
		"bytes_received":    c.amstMetrics.BytesReceived,
		"mode_switches":     c.amstMetrics.ModeSwitches,
		"congestion_events": c.amstMetrics.CongestionEvents,
	}
}

func (c *DWCPv3MetricsCollector) getHDEMetrics() map[string]interface{} {
	c.hdeMetrics.mu.RLock()
	defer c.hdeMetrics.mu.RUnlock()
	return map[string]interface{}{
		"compressions_total":  c.hdeMetrics.CompressionsTotal,
		"decompression_total": c.hdeMetrics.DecompressionsTotal,
		"compression_ratio":   c.hdeMetrics.CompressionRatio,
		"delta_hit_rate":      c.hdeMetrics.DeltaHitRate,
		"crdt_merges":         c.hdeMetrics.CRDTMerges,
		"algorithm_usage":     c.hdeMetrics.AlgorithmUsage,
	}
}

func (c *DWCPv3MetricsCollector) getPBAMetrics() map[string]interface{} {
	c.pbaMetrics.mu.RLock()
	defer c.pbaMetrics.mu.RUnlock()
	return map[string]interface{}{
		"total_predictions":      c.pbaMetrics.TotalPredictions,
		"datacenter_predictions": c.pbaMetrics.DatacenterPredictions,
		"internet_predictions":   c.pbaMetrics.InternetPredictions,
		"prediction_accuracy":    c.pbaMetrics.PredictionAccuracy,
		"avg_prediction_latency": c.pbaMetrics.AvgPredictionLatency.String(),
		"max_prediction_latency": c.pbaMetrics.MaxPredictionLatency.String(),
	}
}

func (c *DWCPv3MetricsCollector) getASSMetrics() map[string]interface{} {
	c.assMetrics.mu.RLock()
	defer c.assMetrics.mu.RUnlock()
	return map[string]interface{}{
		"sync_operations":      c.assMetrics.SyncOperations,
		"successful_syncs":     c.assMetrics.SuccessfulSyncs,
		"failed_syncs":         c.assMetrics.FailedSyncs,
		"conflict_resolutions": c.assMetrics.ConflictResolutions,
		"full_syncs":           c.assMetrics.FullSyncs,
		"incremental_syncs":    c.assMetrics.IncrementalSyncs,
	}
}

func (c *DWCPv3MetricsCollector) getACPMetrics() map[string]interface{} {
	c.acpMetrics.mu.RLock()
	defer c.acpMetrics.mu.RUnlock()
	return map[string]interface{}{
		"consensus_operations": c.acpMetrics.ConsensusOperations,
		"raft_consensus":       c.acpMetrics.RaftConsensus,
		"pbft_consensus":       c.acpMetrics.PBFTConsensus,
		"avg_consensus_time":   c.acpMetrics.AvgConsensusTime.String(),
		"byzantine_detections": c.acpMetrics.ByzantineDetections,
		"failovers":            c.acpMetrics.Failovers,
	}
}

func (c *DWCPv3MetricsCollector) getITPMetrics() map[string]interface{} {
	c.itpMetrics.mu.RLock()
	defer c.itpMetrics.mu.RUnlock()
	return map[string]interface{}{
		"placement_decisions":    c.itpMetrics.PlacementDecisions,
		"optimal_placements":     c.itpMetrics.OptimalPlacements,
		"placement_score":        c.itpMetrics.PlacementScore,
		"ml_prediction_accuracy": c.itpMetrics.MLPredictionAccuracy,
	}
}

func (c *DWCPv3MetricsCollector) getModeMetrics(m *ModeMetrics) map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return map[string]interface{}{
		"average_throughput": m.AverageThroughput,
		"average_latency":    m.AverageLatency,
		"p95_latency":        m.P95Latency,
		"p99_latency":        m.P99Latency,
		"error_rate":         m.ErrorRate,
		"rdma_throughput":    m.RDMAThroughput,
		"tcp_streams":        m.TCPStreams,
		"compression_ratio":  m.TCPCompressionRatio,
	}
}

// calculateSuccessRate calculates migration success rate
func (c *DWCPv3MetricsCollector) calculateSuccessRate() float64 {
	total := c.totalMigrations.Load()
	if total == 0 {
		return 100.0
	}
	successful := c.successfulMigrations.Load()
	return (float64(successful) / float64(total)) * 100.0
}

// aggregateMetricsLoop periodically aggregates and processes metrics
func (c *DWCPv3MetricsCollector) aggregateMetricsLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.computeDerivedMetrics()
		}
	}
}

// computeDerivedMetrics calculates derived metrics
func (c *DWCPv3MetricsCollector) computeDerivedMetrics() {
	// Calculate stream efficiency
	c.amstMetrics.mu.Lock()
	if c.amstMetrics.TotalStreams > 0 {
		c.amstMetrics.StreamEfficiency = (float64(c.amstMetrics.ActiveStreams) / float64(c.amstMetrics.TotalStreams)) * 100.0
	}
	c.amstMetrics.mu.Unlock()

	// Calculate HDE metrics
	c.hdeMetrics.mu.Lock()
	if c.hdeMetrics.BytesCompressed > 0 && c.hdeMetrics.BytesOriginal > 0 {
		c.hdeMetrics.CompressionRatio = float64(c.hdeMetrics.BytesOriginal) / float64(c.hdeMetrics.BytesCompressed)
	}
	c.hdeMetrics.mu.Unlock()
}

// Close stops the metrics collector
func (c *DWCPv3MetricsCollector) Close() error {
	c.cancel()
	c.logger.Info("DWCP v3 metrics collector closed")
	return nil
}

// Helper function to create mode metrics
func newModeMetrics(mode string) *ModeMetrics {
	return &ModeMetrics{
		Mode:             mode,
		latencyHistogram: make([]float64, 0, 10000),
		maxHistogramSize: 10000,
	}
}
