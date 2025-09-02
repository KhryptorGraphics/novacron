package migration

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// MigrationMonitor provides real-time monitoring of migration operations
type MigrationMonitor struct {
	activeMigrations map[string]*MonitoredMigration
	progressTracker  *ProgressTracker
	metricsExporter  *PrometheusExporter
	alertManager     *AlertManager
	dashboardData    *DashboardData
	eventBus         *EventBus
	mu               sync.RWMutex
	ctx              context.Context
	cancel           context.CancelFunc
}

// MonitoredMigration represents a migration being monitored
type MonitoredMigration struct {
	ID               string                 `json:"id"`
	VMID             string                 `json:"vm_id"`
	VMName           string                 `json:"vm_name"`
	SourceNode       string                 `json:"source_node"`
	DestinationNode  string                 `json:"destination_node"`
	Type             string                 `json:"type"`
	Status           string                 `json:"status"`
	Progress         atomic.Value           `json:"-"` // MigrationProgress
	StartTime        time.Time              `json:"start_time"`
	EstimatedEndTime atomic.Value           `json:"-"` // time.Time
	Metrics          *MigrationMetrics      `json:"metrics"`
	Alerts           []Alert                `json:"alerts"`
	Events           []MigrationMonitorEvent `json:"events"`
	mu               sync.RWMutex
}

// MigrationProgress tracks detailed migration progress
type MigrationProgress struct {
	Phase            MigrationPhase `json:"phase"`
	OverallProgress  float64        `json:"overall_progress"`
	PhaseProgress    float64        `json:"phase_progress"`
	BytesTransferred int64          `json:"bytes_transferred"`
	TotalBytes       int64          `json:"total_bytes"`
	PagesTransferred int64          `json:"pages_transferred"`
	TotalPages       int64          `json:"total_pages"`
	DirtyPages       int64          `json:"dirty_pages"`
	Iterations       int            `json:"iterations"`
	CurrentIteration int            `json:"current_iteration"`
	TransferRate     int64          `json:"transfer_rate"` // bytes per second
	ETA              time.Duration  `json:"eta"`
}

// MigrationPhase represents the current phase of migration
type MigrationPhase string

const (
	PhaseInitializing   MigrationPhase = "initializing"
	PhasePreCopy        MigrationPhase = "pre_copy"
	PhaseMemoryCopy     MigrationPhase = "memory_copy"
	PhaseDiskCopy       MigrationPhase = "disk_copy"
	PhaseDowntime       MigrationPhase = "downtime"
	PhaseActivation     MigrationPhase = "activation"
	PhaseVerification   MigrationPhase = "verification"
	PhaseCompleted      MigrationPhase = "completed"
	PhaseFailed         MigrationPhase = "failed"
)

// MigrationMetrics contains performance metrics for a migration
type MigrationMetrics struct {
	// Performance metrics
	TransferRate        atomic.Int64  // bytes per second
	CompressionRatio    atomic.Value  // float64
	NetworkLatency      atomic.Int64  // milliseconds
	CPUUsage            atomic.Value  // float64 percentage
	MemoryUsage         atomic.Int64  // bytes
	DiskIOPS            atomic.Int64  // operations per second
	NetworkBandwidth    atomic.Int64  // bytes per second
	
	// Migration-specific metrics
	DirtyPageRate       atomic.Int64  // pages per second
	ConvergenceRate     atomic.Value  // float64
	DowntimeEstimate    atomic.Int64  // milliseconds
	ActualDowntime      atomic.Int64  // milliseconds
	
	// Error metrics
	RetryCount          atomic.Int32
	ErrorCount          atomic.Int32
	PacketLoss          atomic.Value  // float64 percentage
	
	// Resource metrics
	SourceCPUUsage      atomic.Value  // float64 percentage
	SourceMemoryUsage   atomic.Int64  // bytes
	DestCPUUsage        atomic.Value  // float64 percentage
	DestMemoryUsage     atomic.Int64  // bytes
}

// ProgressTracker tracks migration progress with ETA calculation
type ProgressTracker struct {
	history      []ProgressSnapshot
	maxHistory   int
	mu           sync.RWMutex
}

// ProgressSnapshot represents a point-in-time progress snapshot
type ProgressSnapshot struct {
	Timestamp        time.Time
	BytesTransferred int64
	Progress         float64
	TransferRate     int64
}

// PrometheusExporter exports metrics to Prometheus
type PrometheusExporter struct {
	// Migration metrics
	migrationsTotal      prometheus.Counter
	migrationsActive     prometheus.Gauge
	migrationDuration    prometheus.Histogram
	migrationProgress    *prometheus.GaugeVec
	
	// Performance metrics
	transferRate         *prometheus.GaugeVec
	compressionRatio     *prometheus.GaugeVec
	networkLatency       *prometheus.GaugeVec
	downtimeMillis       *prometheus.HistogramVec
	
	// Error metrics
	migrationErrors      *prometheus.CounterVec
	migrationRetries     *prometheus.CounterVec
	
	// Resource metrics
	cpuUsage            *prometheus.GaugeVec
	memoryUsage         *prometheus.GaugeVec
	diskIOPS            *prometheus.GaugeVec
	networkBandwidth    *prometheus.GaugeVec
}

// AlertManager manages migration alerts
type AlertManager struct {
	alerts       []Alert
	thresholds   AlertThresholds
	subscribers  []AlertSubscriber
	mu           sync.RWMutex
}

// Alert represents a migration alert
type Alert struct {
	ID          string      `json:"id"`
	MigrationID string      `json:"migration_id"`
	Severity    AlertSeverity `json:"severity"`
	Type        AlertType   `json:"type"`
	Message     string      `json:"message"`
	Timestamp   time.Time   `json:"timestamp"`
	Resolved    bool        `json:"resolved"`
	ResolvedAt  *time.Time  `json:"resolved_at,omitempty"`
}

// AlertSeverity represents alert severity levels
type AlertSeverity string

const (
	AlertSeverityInfo     AlertSeverity = "info"
	AlertSeverityWarning  AlertSeverity = "warning"
	AlertSeverityError    AlertSeverity = "error"
	AlertSeverityCritical AlertSeverity = "critical"
)

// AlertType represents types of alerts
type AlertType string

const (
	AlertTypeSlowTransfer    AlertType = "slow_transfer"
	AlertTypeHighLatency     AlertType = "high_latency"
	AlertTypePacketLoss      AlertType = "packet_loss"
	AlertTypeConvergenceIssue AlertType = "convergence_issue"
	AlertTypeResourceLimit   AlertType = "resource_limit"
	AlertTypeMigrationStalled AlertType = "migration_stalled"
	AlertTypeMigrationFailed AlertType = "migration_failed"
)

// AlertThresholds defines thresholds for triggering alerts
type AlertThresholds struct {
	MinTransferRate      int64   // bytes per second
	MaxLatency           int64   // milliseconds
	MaxPacketLoss        float64 // percentage
	MaxDowntime          int64   // milliseconds
	StallTimeout         time.Duration
	ConvergenceTimeout   time.Duration
	MaxCPUUsage          float64 // percentage
	MaxMemoryUsage       int64   // bytes
}

// AlertSubscriber receives alert notifications
type AlertSubscriber interface {
	OnAlert(alert Alert)
}

// DashboardData provides data for migration dashboard
type DashboardData struct {
	Summary          DashboardSummary         `json:"summary"`
	ActiveMigrations []MigrationDashboardItem `json:"active_migrations"`
	RecentAlerts     []Alert                  `json:"recent_alerts"`
	PerformanceData  PerformanceData          `json:"performance"`
	mu               sync.RWMutex
}

// DashboardSummary provides summary statistics
type DashboardSummary struct {
	TotalMigrations      int     `json:"total_migrations"`
	ActiveMigrations     int     `json:"active_migrations"`
	CompletedMigrations  int     `json:"completed_migrations"`
	FailedMigrations     int     `json:"failed_migrations"`
	SuccessRate          float64 `json:"success_rate"`
	AverageTransferRate  int64   `json:"average_transfer_rate"`
	AverageDowntime      int64   `json:"average_downtime"`
}

// MigrationDashboardItem represents a migration in the dashboard
type MigrationDashboardItem struct {
	ID              string    `json:"id"`
	VMName          string    `json:"vm_name"`
	Progress        float64   `json:"progress"`
	Phase           string    `json:"phase"`
	TransferRate    int64     `json:"transfer_rate"`
	ETA             string    `json:"eta"`
	Status          string    `json:"status"`
	StartTime       time.Time `json:"start_time"`
}

// PerformanceData contains performance metrics for the dashboard
type PerformanceData struct {
	TransferRateHistory []DataPoint `json:"transfer_rate_history"`
	ProgressHistory     []DataPoint `json:"progress_history"`
	LatencyHistory      []DataPoint `json:"latency_history"`
	ResourceUsage       ResourceUsageData `json:"resource_usage"`
}

// DataPoint represents a time-series data point
type DataPoint struct {
	Timestamp time.Time   `json:"timestamp"`
	Value     interface{} `json:"value"`
}

// ResourceUsageData contains resource usage information
type ResourceUsageData struct {
	SourceCPU    float64 `json:"source_cpu"`
	SourceMemory int64   `json:"source_memory"`
	DestCPU      float64 `json:"dest_cpu"`
	DestMemory   int64   `json:"dest_memory"`
}

// EventBus manages migration events
type EventBus struct {
	subscribers map[string][]EventSubscriber
	mu          sync.RWMutex
}

// EventSubscriber receives migration events
type EventSubscriber interface {
	OnEvent(event MigrationMonitorEvent)
}

// MigrationMonitorEvent represents a migration event
type MigrationMonitorEvent struct {
	ID          string                 `json:"id"`
	MigrationID string                 `json:"migration_id"`
	Type        string                 `json:"type"`
	Timestamp   time.Time              `json:"timestamp"`
	Data        map[string]interface{} `json:"data"`
}

// NewMigrationMonitor creates a new migration monitor
func NewMigrationMonitor() *MigrationMonitor {
	ctx, cancel := context.WithCancel(context.Background())
	
	monitor := &MigrationMonitor{
		activeMigrations: make(map[string]*MonitoredMigration),
		progressTracker:  NewProgressTracker(100),
		metricsExporter:  NewPrometheusExporter(),
		alertManager:     NewAlertManager(),
		dashboardData:    NewDashboardData(),
		eventBus:         NewEventBus(),
		ctx:              ctx,
		cancel:           cancel,
	}
	
	// Start monitoring goroutines
	go monitor.updateLoop()
	go monitor.alertLoop()
	
	return monitor
}

// StartMonitoring starts monitoring a migration
func (m *MigrationMonitor) StartMonitoring(migrationID, vmID, vmName, sourceNode, destNode, migrationType string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if _, exists := m.activeMigrations[migrationID]; exists {
		return errors.New("migration already being monitored")
	}
	
	monitored := &MonitoredMigration{
		ID:              migrationID,
		VMID:            vmID,
		VMName:          vmName,
		SourceNode:      sourceNode,
		DestinationNode: destNode,
		Type:            migrationType,
		Status:          "initializing",
		StartTime:       time.Now(),
		Metrics:         &MigrationMetrics{},
		Alerts:          []Alert{},
		Events:          []MigrationMonitorEvent{},
	}
	
	// Initialize progress
	progress := MigrationProgress{
		Phase:           PhaseInitializing,
		OverallProgress: 0,
	}
	monitored.Progress.Store(progress)
	
	m.activeMigrations[migrationID] = monitored
	
	// Emit start event
	m.eventBus.Publish(MigrationMonitorEvent{
		ID:          uuid.New().String(),
		MigrationID: migrationID,
		Type:        "migration_started",
		Timestamp:   time.Now(),
		Data: map[string]interface{}{
			"vm_id":   vmID,
			"vm_name": vmName,
			"type":    migrationType,
		},
	})
	
	// Update Prometheus metrics
	m.metricsExporter.migrationsTotal.Inc()
	m.metricsExporter.migrationsActive.Inc()
	
	return nil
}

// UpdateProgress updates migration progress
func (m *MigrationMonitor) UpdateProgress(migrationID string, progress MigrationProgress) error {
	m.mu.RLock()
	migration, exists := m.activeMigrations[migrationID]
	m.mu.RUnlock()
	
	if !exists {
		return errors.New("migration not found")
	}
	
	// Update progress
	migration.Progress.Store(progress)
	
	// Calculate ETA
	eta := m.calculateETA(migrationID, progress)
	migration.EstimatedEndTime.Store(time.Now().Add(eta))
	
	// Update progress tracker
	m.progressTracker.AddSnapshot(ProgressSnapshot{
		Timestamp:        time.Now(),
		BytesTransferred: progress.BytesTransferred,
		Progress:         progress.OverallProgress,
		TransferRate:     progress.TransferRate,
	})
	
	// Update Prometheus metrics
	m.metricsExporter.migrationProgress.WithLabelValues(migrationID).Set(progress.OverallProgress)
	m.metricsExporter.transferRate.WithLabelValues(migrationID).Set(float64(progress.TransferRate))
	
	// Check for alerts
	m.checkProgressAlerts(migrationID, progress)
	
	return nil
}

// calculateETA calculates estimated time to completion
func (m *MigrationMonitor) calculateETA(migrationID string, progress MigrationProgress) time.Duration {
	if progress.TransferRate <= 0 || progress.TotalBytes <= 0 {
		return 0
	}
	
	remainingBytes := progress.TotalBytes - progress.BytesTransferred
	if remainingBytes <= 0 {
		return 0
	}
	
	// Simple ETA calculation
	secondsRemaining := float64(remainingBytes) / float64(progress.TransferRate)
	
	// Adjust for dirty pages in live migration
	if progress.Phase == PhaseMemoryCopy && progress.DirtyPages > 0 {
		// Account for pages that need to be retransferred
		dirtyBytes := progress.DirtyPages * 4096 // Assuming 4KB pages
		secondsRemaining += float64(dirtyBytes) / float64(progress.TransferRate)
	}
	
	return time.Duration(secondsRemaining) * time.Second
}

// checkProgressAlerts checks for progress-related alerts
func (m *MigrationMonitor) checkProgressAlerts(migrationID string, progress MigrationProgress) {
	thresholds := m.alertManager.thresholds
	
	// Check transfer rate
	if progress.TransferRate < thresholds.MinTransferRate {
		m.alertManager.CreateAlert(Alert{
			ID:          uuid.New().String(),
			MigrationID: migrationID,
			Severity:    AlertSeverityWarning,
			Type:        AlertTypeSlowTransfer,
			Message:     fmt.Sprintf("Transfer rate (%d B/s) below threshold (%d B/s)", 
				progress.TransferRate, thresholds.MinTransferRate),
			Timestamp:   time.Now(),
		})
	}
	
	// Check convergence for live migration
	if progress.Phase == PhaseMemoryCopy && progress.CurrentIteration > 10 {
		if progress.DirtyPages > progress.PagesTransferred/2 {
			m.alertManager.CreateAlert(Alert{
				ID:          uuid.New().String(),
				MigrationID: migrationID,
				Severity:    AlertSeverityWarning,
				Type:        AlertTypeConvergenceIssue,
				Message:     "Migration may not converge: high dirty page rate",
				Timestamp:   time.Now(),
			})
		}
	}
}

// UpdateMetrics updates migration metrics
func (m *MigrationMonitor) UpdateMetrics(migrationID string, metrics map[string]interface{}) error {
	m.mu.RLock()
	migration, exists := m.activeMigrations[migrationID]
	m.mu.RUnlock()
	
	if !exists {
		return errors.New("migration not found")
	}
	
	// Update metrics
	if val, ok := metrics["transfer_rate"].(int64); ok {
		migration.Metrics.TransferRate.Store(val)
	}
	if val, ok := metrics["compression_ratio"].(float64); ok {
		migration.Metrics.CompressionRatio.Store(val)
	}
	if val, ok := metrics["network_latency"].(int64); ok {
		migration.Metrics.NetworkLatency.Store(val)
	}
	if val, ok := metrics["cpu_usage"].(float64); ok {
		migration.Metrics.CPUUsage.Store(val)
	}
	if val, ok := metrics["memory_usage"].(int64); ok {
		migration.Metrics.MemoryUsage.Store(val)
	}
	if val, ok := metrics["dirty_page_rate"].(int64); ok {
		migration.Metrics.DirtyPageRate.Store(val)
	}
	
	// Check for metric-based alerts
	m.checkMetricAlerts(migrationID, migration.Metrics)
	
	return nil
}

// checkMetricAlerts checks for metric-based alerts
func (m *MigrationMonitor) checkMetricAlerts(migrationID string, metrics *MigrationMetrics) {
	thresholds := m.alertManager.thresholds
	
	// Check network latency
	if latency := metrics.NetworkLatency.Load(); latency > thresholds.MaxLatency {
		m.alertManager.CreateAlert(Alert{
			ID:          uuid.New().String(),
			MigrationID: migrationID,
			Severity:    AlertSeverityWarning,
			Type:        AlertTypeHighLatency,
			Message:     fmt.Sprintf("Network latency (%d ms) exceeds threshold (%d ms)", 
				latency, thresholds.MaxLatency),
			Timestamp:   time.Now(),
		})
	}
	
	// Check packet loss
	if loss := metrics.PacketLoss.Load(); loss != nil {
		if lossPercent := loss.(float64); lossPercent > thresholds.MaxPacketLoss {
			m.alertManager.CreateAlert(Alert{
				ID:          uuid.New().String(),
				MigrationID: migrationID,
				Severity:    AlertSeverityError,
				Type:        AlertTypePacketLoss,
				Message:     fmt.Sprintf("Packet loss (%.2f%%) exceeds threshold (%.2f%%)", 
					lossPercent, thresholds.MaxPacketLoss),
				Timestamp:   time.Now(),
			})
		}
	}
	
	// Check CPU usage
	if cpu := metrics.CPUUsage.Load(); cpu != nil {
		if cpuPercent := cpu.(float64); cpuPercent > thresholds.MaxCPUUsage {
			m.alertManager.CreateAlert(Alert{
				ID:          uuid.New().String(),
				MigrationID: migrationID,
				Severity:    AlertSeverityWarning,
				Type:        AlertTypeResourceLimit,
				Message:     fmt.Sprintf("CPU usage (%.2f%%) exceeds threshold (%.2f%%)", 
					cpuPercent, thresholds.MaxCPUUsage),
				Timestamp:   time.Now(),
			})
		}
	}
}

// CompleteMigration marks a migration as completed
func (m *MigrationMonitor) CompleteMigration(migrationID string, success bool, downtime time.Duration) error {
	m.mu.Lock()
	migration, exists := m.activeMigrations[migrationID]
	if !exists {
		m.mu.Unlock()
		return errors.New("migration not found")
	}
	
	// Update status
	if success {
		migration.Status = "completed"
		progress := migration.Progress.Load().(MigrationProgress)
		progress.Phase = PhaseCompleted
		progress.OverallProgress = 100
		migration.Progress.Store(progress)
	} else {
		migration.Status = "failed"
		progress := migration.Progress.Load().(MigrationProgress)
		progress.Phase = PhaseFailed
		migration.Progress.Store(progress)
	}
	
	// Record downtime
	migration.Metrics.ActualDowntime.Store(int64(downtime.Milliseconds()))
	
	// Remove from active migrations
	delete(m.activeMigrations, migrationID)
	m.mu.Unlock()
	
	// Update Prometheus metrics
	m.metricsExporter.migrationsActive.Dec()
	duration := time.Since(migration.StartTime)
	m.metricsExporter.migrationDuration.Observe(duration.Seconds())
	m.metricsExporter.downtimeMillis.WithLabelValues(migration.Type).Observe(float64(downtime.Milliseconds()))
	
	// Emit completion event
	m.eventBus.Publish(MigrationMonitorEvent{
		ID:          uuid.New().String(),
		MigrationID: migrationID,
		Type:        "migration_completed",
		Timestamp:   time.Now(),
		Data: map[string]interface{}{
			"success":  success,
			"duration": duration.String(),
			"downtime": downtime.String(),
		},
	})
	
	return nil
}

// GetMigrationStatus returns the status of a migration
func (m *MigrationMonitor) GetMigrationStatus(migrationID string) (map[string]interface{}, error) {
	m.mu.RLock()
	migration, exists := m.activeMigrations[migrationID]
	m.mu.RUnlock()
	
	if !exists {
		return nil, errors.New("migration not found")
	}
	
	progress := migration.Progress.Load().(MigrationProgress)
	
	status := map[string]interface{}{
		"id":               migration.ID,
		"vm_id":            migration.VMID,
		"vm_name":          migration.VMName,
		"source_node":      migration.SourceNode,
		"destination_node": migration.DestinationNode,
		"type":             migration.Type,
		"status":           migration.Status,
		"progress":         progress,
		"start_time":       migration.StartTime,
		"metrics":          m.getMetricsMap(migration.Metrics),
		"alerts":           migration.Alerts,
	}
	
	if eta := migration.EstimatedEndTime.Load(); eta != nil {
		status["estimated_end_time"] = eta.(time.Time)
	}
	
	return status, nil
}

// getMetricsMap converts metrics to a map
func (m *MigrationMonitor) getMetricsMap(metrics *MigrationMetrics) map[string]interface{} {
	result := make(map[string]interface{})
	
	result["transfer_rate"] = metrics.TransferRate.Load()
	if ratio := metrics.CompressionRatio.Load(); ratio != nil {
		result["compression_ratio"] = ratio.(float64)
	}
	result["network_latency"] = metrics.NetworkLatency.Load()
	if cpu := metrics.CPUUsage.Load(); cpu != nil {
		result["cpu_usage"] = cpu.(float64)
	}
	result["memory_usage"] = metrics.MemoryUsage.Load()
	result["dirty_page_rate"] = metrics.DirtyPageRate.Load()
	result["retry_count"] = metrics.RetryCount.Load()
	result["error_count"] = metrics.ErrorCount.Load()
	
	return result
}

// GetDashboardData returns data for the migration dashboard
func (m *MigrationMonitor) GetDashboardData() *DashboardData {
	m.dashboardData.mu.RLock()
	defer m.dashboardData.mu.RUnlock()
	
	return m.dashboardData
}

// updateLoop continuously updates monitoring data
func (m *MigrationMonitor) updateLoop() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.updateDashboardData()
		}
	}
}

// updateDashboardData updates the dashboard data
func (m *MigrationMonitor) updateDashboardData() {
	m.mu.RLock()
	activeMigrations := make([]MigrationDashboardItem, 0, len(m.activeMigrations))
	
	for _, migration := range m.activeMigrations {
		progress := migration.Progress.Load().(MigrationProgress)
		
		item := MigrationDashboardItem{
			ID:           migration.ID,
			VMName:       migration.VMName,
			Progress:     progress.OverallProgress,
			Phase:        string(progress.Phase),
			TransferRate: progress.TransferRate,
			Status:       migration.Status,
			StartTime:    migration.StartTime,
		}
		
		if progress.ETA > 0 {
			item.ETA = progress.ETA.String()
		}
		
		activeMigrations = append(activeMigrations, item)
	}
	m.mu.RUnlock()
	
	m.dashboardData.mu.Lock()
	m.dashboardData.ActiveMigrations = activeMigrations
	m.dashboardData.Summary.ActiveMigrations = len(activeMigrations)
	m.dashboardData.mu.Unlock()
}

// alertLoop continuously checks for alert conditions
func (m *MigrationMonitor) alertLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.checkStalledMigrations()
		}
	}
}

// checkStalledMigrations checks for stalled migrations
func (m *MigrationMonitor) checkStalledMigrations() {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	stallTimeout := m.alertManager.thresholds.StallTimeout
	
	for migrationID, migration := range m.activeMigrations {
		progress := migration.Progress.Load().(MigrationProgress)
		
		// Check if progress hasn't changed for too long
		lastSnapshot := m.progressTracker.GetLastSnapshot()
		if lastSnapshot != nil && time.Since(lastSnapshot.Timestamp) > stallTimeout {
			if math.Abs(lastSnapshot.Progress - progress.OverallProgress) < 0.01 {
				m.alertManager.CreateAlert(Alert{
					ID:          uuid.New().String(),
					MigrationID: migrationID,
					Severity:    AlertSeverityError,
					Type:        AlertTypeMigrationStalled,
					Message:     fmt.Sprintf("Migration stalled: no progress for %s", stallTimeout),
					Timestamp:   time.Now(),
				})
			}
		}
	}
}

// Close shuts down the migration monitor
func (m *MigrationMonitor) Close() {
	m.cancel()
}

// NewProgressTracker creates a new progress tracker
func NewProgressTracker(maxHistory int) *ProgressTracker {
	return &ProgressTracker{
		history:    make([]ProgressSnapshot, 0, maxHistory),
		maxHistory: maxHistory,
	}
}

// AddSnapshot adds a progress snapshot
func (pt *ProgressTracker) AddSnapshot(snapshot ProgressSnapshot) {
	pt.mu.Lock()
	defer pt.mu.Unlock()
	
	pt.history = append(pt.history, snapshot)
	
	// Trim history if needed
	if len(pt.history) > pt.maxHistory {
		pt.history = pt.history[len(pt.history)-pt.maxHistory:]
	}
}

// GetLastSnapshot returns the last snapshot
func (pt *ProgressTracker) GetLastSnapshot() *ProgressSnapshot {
	pt.mu.RLock()
	defer pt.mu.RUnlock()
	
	if len(pt.history) == 0 {
		return nil
	}
	
	return &pt.history[len(pt.history)-1]
}

// NewPrometheusExporter creates a new Prometheus exporter
func NewPrometheusExporter() *PrometheusExporter {
	return &PrometheusExporter{
		migrationsTotal: promauto.NewCounter(prometheus.CounterOpts{
			Name: "novacron_migrations_total",
			Help: "Total number of migrations started",
		}),
		migrationsActive: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "novacron_migrations_active",
			Help: "Number of currently active migrations",
		}),
		migrationDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "novacron_migration_duration_seconds",
			Help:    "Migration duration in seconds",
			Buckets: prometheus.ExponentialBuckets(10, 2, 10),
		}),
		migrationProgress: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "novacron_migration_progress",
			Help: "Migration progress percentage",
		}, []string{"migration_id"}),
		transferRate: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "novacron_migration_transfer_rate_bytes",
			Help: "Migration transfer rate in bytes per second",
		}, []string{"migration_id"}),
		compressionRatio: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "novacron_migration_compression_ratio",
			Help: "Migration data compression ratio",
		}, []string{"migration_id"}),
		networkLatency: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "novacron_migration_network_latency_ms",
			Help: "Network latency in milliseconds",
		}, []string{"migration_id"}),
		downtimeMillis: promauto.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "novacron_migration_downtime_milliseconds",
			Help:    "VM downtime during migration in milliseconds",
			Buckets: prometheus.ExponentialBuckets(10, 2, 10),
		}, []string{"migration_type"}),
		migrationErrors: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "novacron_migration_errors_total",
			Help: "Total number of migration errors",
		}, []string{"migration_id", "error_type"}),
		migrationRetries: promauto.NewCounterVec(prometheus.CounterOpts{
			Name: "novacron_migration_retries_total",
			Help: "Total number of migration retry attempts",
		}, []string{"migration_id"}),
		cpuUsage: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "novacron_migration_cpu_usage_percent",
			Help: "CPU usage percentage during migration",
		}, []string{"migration_id", "node"}),
		memoryUsage: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "novacron_migration_memory_usage_bytes",
			Help: "Memory usage in bytes during migration",
		}, []string{"migration_id", "node"}),
		diskIOPS: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "novacron_migration_disk_iops",
			Help: "Disk IOPS during migration",
		}, []string{"migration_id", "node"}),
		networkBandwidth: promauto.NewGaugeVec(prometheus.GaugeOpts{
			Name: "novacron_migration_network_bandwidth_bytes",
			Help: "Network bandwidth usage in bytes per second",
		}, []string{"migration_id", "direction"}),
	}
}

// NewAlertManager creates a new alert manager
func NewAlertManager() *AlertManager {
	return &AlertManager{
		alerts:      []Alert{},
		subscribers: []AlertSubscriber{},
		thresholds: AlertThresholds{
			MinTransferRate:    1024 * 1024,      // 1 MB/s
			MaxLatency:         100,              // 100ms
			MaxPacketLoss:      1.0,              // 1%
			MaxDowntime:        30000,            // 30 seconds
			StallTimeout:       30 * time.Second,
			ConvergenceTimeout: 5 * time.Minute,
			MaxCPUUsage:        80.0,             // 80%
			MaxMemoryUsage:     8 * 1024 * 1024 * 1024, // 8GB
		},
	}
}

// CreateAlert creates a new alert
func (am *AlertManager) CreateAlert(alert Alert) {
	am.mu.Lock()
	am.alerts = append(am.alerts, alert)
	am.mu.Unlock()
	
	// Notify subscribers
	for _, subscriber := range am.subscribers {
		go subscriber.OnAlert(alert)
	}
}

// Subscribe adds an alert subscriber
func (am *AlertManager) Subscribe(subscriber AlertSubscriber) {
	am.mu.Lock()
	defer am.mu.Unlock()
	
	am.subscribers = append(am.subscribers, subscriber)
}

// NewDashboardData creates new dashboard data
func NewDashboardData() *DashboardData {
	return &DashboardData{
		Summary:          DashboardSummary{},
		ActiveMigrations: []MigrationDashboardItem{},
		RecentAlerts:     []Alert{},
		PerformanceData:  PerformanceData{},
	}
}

// NewEventBus creates a new event bus
func NewEventBus() *EventBus {
	return &EventBus{
		subscribers: make(map[string][]EventSubscriber),
	}
}

// Subscribe adds an event subscriber
func (eb *EventBus) Subscribe(eventType string, subscriber EventSubscriber) {
	eb.mu.Lock()
	defer eb.mu.Unlock()
	
	eb.subscribers[eventType] = append(eb.subscribers[eventType], subscriber)
}

// Publish publishes an event
func (eb *EventBus) Publish(event MigrationMonitorEvent) {
	eb.mu.RLock()
	subscribers := eb.subscribers[event.Type]
	eb.mu.RUnlock()
	
	for _, subscriber := range subscribers {
		go subscriber.OnEvent(event)
	}
}