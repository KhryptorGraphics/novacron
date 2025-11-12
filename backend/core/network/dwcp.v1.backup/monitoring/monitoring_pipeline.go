package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"go.uber.org/zap"
)

var (
	anomalyDetectedCounter = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "dwcp_anomalies_detected_total",
			Help: "Total number of anomalies detected",
		},
		[]string{"metric_name", "severity", "model_type"},
	)

	anomalyConfidenceGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_anomaly_confidence",
			Help: "Confidence score of detected anomalies",
		},
		[]string{"metric_name", "model_type"},
	)

	detectionLatencyHistogram = promauto.NewHistogram(
		prometheus.HistogramOpts{
			Name:    "dwcp_anomaly_detection_latency_seconds",
			Help:    "Latency of anomaly detection",
			Buckets: prometheus.DefBuckets,
		},
	)

	metricsProcessedCounter = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "dwcp_metrics_processed_total",
			Help: "Total number of metrics processed",
		},
	)
)

// MonitoringPipeline coordinates real-time anomaly detection and alerting
type MonitoringPipeline struct {
	detector       *AnomalyDetector
	metricsBuffer  *MetricsBuffer
	alertManager   *AlertManager
	checkInterval  time.Duration

	logger         *zap.Logger
	ctx            context.Context
	cancel         context.CancelFunc
	wg             sync.WaitGroup

	// Statistics
	stats          *PipelineStats
	statsMu        sync.RWMutex
}

// PipelineStats tracks pipeline statistics
type PipelineStats struct {
	MetricsProcessed   int64
	AnomaliesDetected  int64
	AlertsSent         int64
	LastCheckTime      time.Time
	AverageLatency     time.Duration
	ErrorCount         int64
}

// MetricsBuffer buffers incoming metrics for anomaly detection
type MetricsBuffer struct {
	buffer    []*MetricVector
	maxSize   int
	mu        sync.RWMutex
}

// NewMetricsBuffer creates a new metrics buffer
func NewMetricsBuffer(maxSize int) *MetricsBuffer {
	if maxSize <= 0 {
		maxSize = 1000
	}

	return &MetricsBuffer{
		buffer:  make([]*MetricVector, 0, maxSize),
		maxSize: maxSize,
	}
}

// Add adds a metric vector to the buffer
func (mb *MetricsBuffer) Add(metrics *MetricVector) {
	mb.mu.Lock()
	defer mb.mu.Unlock()

	mb.buffer = append(mb.buffer, metrics)

	// Keep only last maxSize entries
	if len(mb.buffer) > mb.maxSize {
		mb.buffer = mb.buffer[1:]
	}
}

// GetLatest returns the most recent metric vector
func (mb *MetricsBuffer) GetLatest() *MetricVector {
	mb.mu.RLock()
	defer mb.mu.RUnlock()

	if len(mb.buffer) == 0 {
		return nil
	}

	return mb.buffer[len(mb.buffer)-1]
}

// GetRecent returns the N most recent metric vectors
func (mb *MetricsBuffer) GetRecent(n int) []*MetricVector {
	mb.mu.RLock()
	defer mb.mu.RUnlock()

	if len(mb.buffer) == 0 {
		return nil
	}

	start := len(mb.buffer) - n
	if start < 0 {
		start = 0
	}

	result := make([]*MetricVector, len(mb.buffer)-start)
	copy(result, mb.buffer[start:])

	return result
}

// GetAll returns all buffered metric vectors
func (mb *MetricsBuffer) GetAll() []*MetricVector {
	mb.mu.RLock()
	defer mb.mu.RUnlock()

	result := make([]*MetricVector, len(mb.buffer))
	copy(result, mb.buffer)

	return result
}

// Clear clears the buffer
func (mb *MetricsBuffer) Clear() {
	mb.mu.Lock()
	defer mb.mu.Unlock()

	mb.buffer = make([]*MetricVector, 0, mb.maxSize)
}

// NewMonitoringPipeline creates a new monitoring pipeline
func NewMonitoringPipeline(
	detector *AnomalyDetector,
	alertManager *AlertManager,
	checkInterval time.Duration,
	logger *zap.Logger,
) *MonitoringPipeline {
	if logger == nil {
		logger = zap.NewNop()
	}

	if checkInterval <= 0 {
		checkInterval = 10 * time.Second
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &MonitoringPipeline{
		detector:      detector,
		metricsBuffer: NewMetricsBuffer(1000),
		alertManager:  alertManager,
		checkInterval: checkInterval,
		logger:        logger,
		ctx:           ctx,
		cancel:        cancel,
		stats: &PipelineStats{
			LastCheckTime: time.Now(),
		},
	}
}

// Start starts the monitoring pipeline
func (mp *MonitoringPipeline) Start() error {
	mp.logger.Info("Starting monitoring pipeline",
		zap.Duration("check_interval", mp.checkInterval))

	mp.wg.Add(1)
	go mp.monitoringLoop()

	return nil
}

// Stop stops the monitoring pipeline
func (mp *MonitoringPipeline) Stop() error {
	mp.logger.Info("Stopping monitoring pipeline")

	mp.cancel()
	mp.wg.Wait()

	mp.logger.Info("Monitoring pipeline stopped")

	return nil
}

// ProcessMetrics processes incoming metrics
func (mp *MonitoringPipeline) ProcessMetrics(metrics *MetricVector) error {
	// Add to buffer
	mp.metricsBuffer.Add(metrics)

	metricsProcessedCounter.Inc()

	mp.statsMu.Lock()
	mp.stats.MetricsProcessed++
	mp.statsMu.Unlock()

	return nil
}

// monitoringLoop runs the continuous monitoring loop
func (mp *MonitoringPipeline) monitoringLoop() {
	defer mp.wg.Done()

	ticker := time.NewTicker(mp.checkInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mp.checkForAnomalies()

		case <-mp.ctx.Done():
			return
		}
	}
}

// checkForAnomalies checks the latest metrics for anomalies
func (mp *MonitoringPipeline) checkForAnomalies() {
	startTime := time.Now()

	// Get latest metrics
	metrics := mp.metricsBuffer.GetLatest()
	if metrics == nil {
		return
	}

	// Run anomaly detection
	ctx, cancel := context.WithTimeout(mp.ctx, 30*time.Second)
	defer cancel()

	anomalies, err := mp.detector.Detect(ctx, metrics)
	if err != nil {
		mp.logger.Error("Anomaly detection failed", zap.Error(err))
		mp.statsMu.Lock()
		mp.stats.ErrorCount++
		mp.statsMu.Unlock()
		return
	}

	// Record detection latency
	latency := time.Since(startTime)
	detectionLatencyHistogram.Observe(latency.Seconds())

	mp.statsMu.Lock()
	mp.stats.LastCheckTime = time.Now()
	mp.stats.AverageLatency = latency
	mp.statsMu.Unlock()

	// Process anomalies
	for _, anomaly := range anomalies {
		mp.handleAnomaly(anomaly)
	}
}

// handleAnomaly handles a detected anomaly
func (mp *MonitoringPipeline) handleAnomaly(anomaly *Anomaly) {
	// Log anomaly
	mp.logger.Warn("Anomaly detected",
		zap.String("metric", anomaly.MetricName),
		zap.Float64("value", anomaly.Value),
		zap.Float64("expected", anomaly.Expected),
		zap.Float64("deviation", anomaly.Deviation),
		zap.String("severity", anomaly.Severity.String()),
		zap.Float64("confidence", anomaly.Confidence),
		zap.String("model", anomaly.ModelType),
	)

	// Update Prometheus metrics
	anomalyDetectedCounter.WithLabelValues(
		anomaly.MetricName,
		anomaly.Severity.String(),
		anomaly.ModelType,
	).Inc()

	anomalyConfidenceGauge.WithLabelValues(
		anomaly.MetricName,
		anomaly.ModelType,
	).Set(anomaly.Confidence)

	// Update statistics
	mp.statsMu.Lock()
	mp.stats.AnomaliesDetected++
	mp.statsMu.Unlock()

	// Send alerts based on severity
	if mp.alertManager != nil {
		switch anomaly.Severity {
		case SeverityCritical:
			if err := mp.alertManager.SendCriticalAlert(anomaly); err != nil {
				mp.logger.Error("Failed to send critical alert", zap.Error(err))
			} else {
				mp.statsMu.Lock()
				mp.stats.AlertsSent++
				mp.statsMu.Unlock()
			}

		case SeverityWarning:
			if err := mp.alertManager.SendWarningAlert(anomaly); err != nil {
				mp.logger.Error("Failed to send warning alert", zap.Error(err))
			} else {
				mp.statsMu.Lock()
				mp.stats.AlertsSent++
				mp.statsMu.Unlock()
			}

		case SeverityInfo:
			// Log only, no alert
			mp.logger.Info("Informational anomaly",
				zap.String("metric", anomaly.MetricName),
				zap.String("description", anomaly.Description),
			)
		}
	}
}

// GetStats returns current pipeline statistics
func (mp *MonitoringPipeline) GetStats() PipelineStats {
	mp.statsMu.RLock()
	defer mp.statsMu.RUnlock()

	return *mp.stats
}

// TrainDetector trains the anomaly detector with historical data
func (mp *MonitoringPipeline) TrainDetector(ctx context.Context, normalData []*MetricVector) error {
	mp.logger.Info("Training anomaly detector",
		zap.Int("samples", len(normalData)))

	if err := mp.detector.Train(ctx, normalData); err != nil {
		return fmt.Errorf("detector training failed: %w", err)
	}

	mp.logger.Info("Anomaly detector training completed")

	return nil
}

// EnableDetection enables anomaly detection
func (mp *MonitoringPipeline) EnableDetection() {
	mp.detector.Enable()
	mp.logger.Info("Anomaly detection enabled")
}

// DisableDetection disables anomaly detection
func (mp *MonitoringPipeline) DisableDetection() {
	mp.detector.Disable()
	mp.logger.Info("Anomaly detection disabled")
}

// GetRecentAnomalies returns recently detected anomalies (from logs)
// In production, this should query from a persistent store
func (mp *MonitoringPipeline) GetRecentAnomalies(duration time.Duration) []*Anomaly {
	// This is a placeholder - implement with actual anomaly storage
	return nil
}
