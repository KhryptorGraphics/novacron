// Package metrics provides integration with DWCP components
package metrics

import (
	"sync"

	"github.com/rs/zerolog/log"
)

var (
	// globalCollector is the singleton metrics collector
	globalCollector *Collector
	collectorOnce   sync.Once
	collectorMu     sync.RWMutex

	// globalExporter is the singleton metrics exporter
	globalExporter *Exporter
	exporterOnce   sync.Once
)

// InitializeMetrics sets up the global metrics collector and exporter
func InitializeMetrics(cluster, node string, port int) error {
	var initErr error

	// Initialize collector
	collectorOnce.Do(func() {
		collectorMu.Lock()
		globalCollector = NewCollector(cluster, node)
		collectorMu.Unlock()

		log.Info().
			Str("cluster", cluster).
			Str("node", node).
			Msg("DWCP metrics collector initialized")
	})

	// Initialize exporter
	exporterOnce.Do(func() {
		globalExporter = NewExporter(port)
		if err := globalExporter.Start(); err != nil {
			initErr = err
			log.Error().
				Err(err).
				Int("port", port).
				Msg("Failed to start metrics exporter")
			return
		}

		log.Info().
			Int("port", port).
			Msg("DWCP metrics exporter started")
	})

	return initErr
}

// GetCollector returns the global metrics collector
func GetCollector() *Collector {
	collectorMu.RLock()
	defer collectorMu.RUnlock()
	return globalCollector
}

// GetExporter returns the global metrics exporter
func GetExporter() *Exporter {
	return globalExporter
}

// ShutdownMetrics gracefully shuts down the metrics system
func ShutdownMetrics() error {
	if globalExporter != nil {
		if err := globalExporter.Stop(nil); err != nil {
			log.Error().Err(err).Msg("Failed to stop metrics exporter")
			return err
		}
	}
	log.Info().Msg("DWCP metrics system shut down")
	return nil
}

// AMSTMetricsWrapper wraps AMST operations with metrics collection
type AMSTMetricsWrapper struct {
	collector *Collector
}

// NewAMSTMetricsWrapper creates a new AMST metrics wrapper
func NewAMSTMetricsWrapper() *AMSTMetricsWrapper {
	return &AMSTMetricsWrapper{
		collector: GetCollector(),
	}
}

// OnStreamStart should be called when a new AMST stream starts
func (w *AMSTMetricsWrapper) OnStreamStart(streamID string) {
	if w.collector != nil {
		w.collector.StartStream(streamID)
	}
}

// OnStreamEnd should be called when an AMST stream ends
func (w *AMSTMetricsWrapper) OnStreamEnd(streamID string) {
	if w.collector != nil {
		w.collector.EndStream(streamID)
	}
}

// OnStreamData should be called when data is sent/received on a stream
func (w *AMSTMetricsWrapper) OnStreamData(streamID string, bytesSent, bytesReceived int64) {
	if w.collector != nil {
		w.collector.RecordStreamData(streamID, bytesSent, bytesReceived)
	}
}

// OnStreamError should be called when a stream error occurs
func (w *AMSTMetricsWrapper) OnStreamError(streamID, errorType string) {
	if w.collector != nil {
		w.collector.RecordStreamError(streamID, errorType)
	}
}

// OnBandwidthUpdate should be called when bandwidth utilization changes
func (w *AMSTMetricsWrapper) OnBandwidthUpdate(used, available int64) {
	if w.collector != nil {
		w.collector.UpdateBandwidthUtilization(used, available)
	}
}

// HDEMetricsWrapper wraps HDE operations with metrics collection
type HDEMetricsWrapper struct {
	collector *Collector
}

// NewHDEMetricsWrapper creates a new HDE metrics wrapper
func NewHDEMetricsWrapper() *HDEMetricsWrapper {
	return &HDEMetricsWrapper{
		collector: GetCollector(),
	}
}

// OnCompressionComplete should be called after compression completes
func (w *HDEMetricsWrapper) OnCompressionComplete(dataType string, originalBytes, compressedBytes int64, deltaHit bool) {
	if w.collector != nil {
		w.collector.RecordCompressionOperation(dataType, originalBytes, compressedBytes, deltaHit)
	}
}

// OnDecompressionComplete should be called after decompression completes
func (w *HDEMetricsWrapper) OnDecompressionComplete(success bool) {
	if w.collector != nil {
		w.collector.RecordDecompressionOperation(success)
	}
}

// OnBaselineUpdate should be called when baselines are updated
func (w *HDEMetricsWrapper) OnBaselineUpdate(baselineType string, count int) {
	if w.collector != nil {
		w.collector.UpdateBaselineCount(baselineType, count)
	}
}

// OnDictionaryUpdate should be called when dictionary efficiency changes
func (w *HDEMetricsWrapper) OnDictionaryUpdate(efficiency float64) {
	if w.collector != nil {
		w.collector.UpdateDictionaryEfficiency(efficiency)
	}
}

// MigrationMetricsWrapper wraps migration operations with metrics collection
type MigrationMetricsWrapper struct {
	collector *Collector
}

// NewMigrationMetricsWrapper creates a new migration metrics wrapper
func NewMigrationMetricsWrapper() *MigrationMetricsWrapper {
	return &MigrationMetricsWrapper{
		collector: GetCollector(),
	}
}

// OnMigrationStart should be called when a migration starts
func (w *MigrationMetricsWrapper) OnMigrationStart(migrationID string) {
	if w.collector != nil {
		w.collector.StartMigration(migrationID)
	}
}

// OnMigrationComplete should be called when a migration completes
func (w *MigrationMetricsWrapper) OnMigrationComplete(migrationID, destNode string, dwcpEnabled bool) {
	if w.collector != nil {
		w.collector.EndMigration(migrationID, destNode, dwcpEnabled)
	}
}

// OnSpeedupCalculated should be called when speedup factor is calculated
func (w *MigrationMetricsWrapper) OnSpeedupCalculated(vmType string, speedupFactor float64) {
	if w.collector != nil {
		w.collector.UpdateSpeedupFactor(vmType, speedupFactor)
	}
}

// SystemMetricsWrapper wraps system operations with metrics collection
type SystemMetricsWrapper struct {
	collector *Collector
}

// NewSystemMetricsWrapper creates a new system metrics wrapper
func NewSystemMetricsWrapper() *SystemMetricsWrapper {
	return &SystemMetricsWrapper{
		collector: GetCollector(),
	}
}

// OnComponentHealthChange should be called when component health changes
func (w *SystemMetricsWrapper) OnComponentHealthChange(component string, status HealthStatus) {
	if w.collector != nil {
		w.collector.UpdateComponentHealth(component, status)
	}
}

// OnFeatureToggle should be called when a feature is enabled/disabled
func (w *SystemMetricsWrapper) OnFeatureToggle(feature string, enabled bool) {
	if w.collector != nil {
		w.collector.UpdateFeatureStatus(feature, enabled)
	}
}

// Helper functions for easy integration

// RecordStreamMetrics is a convenience function for recording stream metrics
func RecordStreamMetrics(streamID string, bytesSent, bytesReceived int64, errorType string) {
	wrapper := NewAMSTMetricsWrapper()
	if errorType != "" {
		wrapper.OnStreamError(streamID, errorType)
	} else {
		wrapper.OnStreamData(streamID, bytesSent, bytesReceived)
	}
}

// RecordCompressionMetrics is a convenience function for recording compression metrics
func RecordCompressionMetrics(dataType string, originalSize, compressedSize int64, deltaHit bool) {
	wrapper := NewHDEMetricsWrapper()
	wrapper.OnCompressionComplete(dataType, originalSize, compressedSize, deltaHit)
}

// RecordComponentHealth is a convenience function for recording component health
func RecordComponentHealth(component string, healthy bool) {
	wrapper := NewSystemMetricsWrapper()
	status := HealthHealthy
	if !healthy {
		status = HealthDown
	}
	wrapper.OnComponentHealthChange(component, status)
}
