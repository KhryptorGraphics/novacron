package tiering

import (
	"context"
	"encoding/json"
	"sync"
	"time"
)

// MetricsCollector collects and aggregates tiering metrics
type MetricsCollector struct {
	volumeMetrics  map[string]*VolumeMetricsHistory
	tierMetrics    map[TierLevel]*TierMetricsHistory
	systemMetrics  *SystemMetricsHistory
	mu             sync.RWMutex
	collectInterval time.Duration
	ctx            context.Context
	cancel         context.CancelFunc
}

// VolumeMetricsHistory tracks metrics history for a volume
type VolumeMetricsHistory struct {
	VolumeID      string                  `json:"volume_id"`
	AccessHistory []AccessEvent           `json:"access_history"`
	TierHistory   []TierMigrationEvent    `json:"tier_history"`
	SizeHistory   []SizeEvent             `json:"size_history"`
	CostHistory   []CostEvent             `json:"cost_history"`
	mu            sync.RWMutex
}

// TierMetricsHistory tracks metrics history for a tier
type TierMetricsHistory struct {
	TierLevel      TierLevel              `json:"tier_level"`
	CapacityHistory []CapacityEvent       `json:"capacity_history"`
	PerformanceHistory []PerformanceEvent `json:"performance_history"`
	VolumeCount    []VolumeCountEvent     `json:"volume_count"`
	CostHistory    []TierCostEvent        `json:"cost_history"`
	mu             sync.RWMutex
}

// SystemMetricsHistory tracks overall system metrics
type SystemMetricsHistory struct {
	TierMigrations      []MigrationEvent        `json:"tier_migrations"`
	PolicyExecutions    []PolicyExecutionEvent  `json:"policy_executions"`
	ResourceUtilization []ResourceEvent         `json:"resource_utilization"`
	CostEfficiency      []CostEfficiencyEvent   `json:"cost_efficiency"`
	mu                  sync.RWMutex
}

// Event types for metrics tracking
type AccessEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	AccessType  string    `json:"access_type"` // read, write, delete
	Size        int64     `json:"size"`
	Duration    time.Duration `json:"duration"`
	UserID      string    `json:"user_id"`
}

type TierMigrationEvent struct {
	Timestamp    time.Time `json:"timestamp"`
	FromTier     TierLevel `json:"from_tier"`
	ToTier       TierLevel `json:"to_tier"`
	Reason       string    `json:"reason"`
	PolicyName   string    `json:"policy_name"`
	Duration     time.Duration `json:"duration"`
	Success      bool      `json:"success"`
	ErrorMessage string    `json:"error_message,omitempty"`
}

type SizeEvent struct {
	Timestamp time.Time `json:"timestamp"`
	SizeGB    float64   `json:"size_gb"`
	Change    float64   `json:"change_gb"`
}

type CostEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	DailyCost   float64   `json:"daily_cost"`
	MonthlyCost float64   `json:"monthly_cost"`
	TierLevel   TierLevel `json:"tier_level"`
}

type CapacityEvent struct {
	Timestamp     time.Time `json:"timestamp"`
	TotalGB       int64     `json:"total_gb"`
	UsedGB        int64     `json:"used_gb"`
	AvailableGB   int64     `json:"available_gb"`
	UsagePercent  float64   `json:"usage_percent"`
	VolumeCount   int       `json:"volume_count"`
}

type PerformanceEvent struct {
	Timestamp    time.Time `json:"timestamp"`
	IOPS         float64   `json:"iops"`
	Throughput   float64   `json:"throughput_mbps"`
	Latency      time.Duration `json:"latency"`
	ErrorRate    float64   `json:"error_rate"`
	Availability float64   `json:"availability"`
}

type VolumeCountEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	Count       int       `json:"count"`
	ActiveCount int       `json:"active_count"`
}

type TierCostEvent struct {
	Timestamp      time.Time `json:"timestamp"`
	TotalCost      float64   `json:"total_cost"`
	CostPerGB      float64   `json:"cost_per_gb"`
	VolumeCount    int       `json:"volume_count"`
	EfficiencyRate float64   `json:"efficiency_rate"`
}

type MigrationEvent struct {
	Timestamp    time.Time `json:"timestamp"`
	VolumeID     string    `json:"volume_id"`
	FromTier     TierLevel `json:"from_tier"`
	ToTier       TierLevel `json:"to_tier"`
	SizeGB       float64   `json:"size_gb"`
	Duration     time.Duration `json:"duration"`
	CostSaving   float64   `json:"cost_saving"`
	PolicyName   string    `json:"policy_name"`
}

type PolicyExecutionEvent struct {
	Timestamp      time.Time `json:"timestamp"`
	PolicyName     string    `json:"policy_name"`
	VolumesEvaluated int     `json:"volumes_evaluated"`
	VolumesMoved   int       `json:"volumes_moved"`
	Duration       time.Duration `json:"duration"`
	ErrorCount     int       `json:"error_count"`
}

type ResourceEvent struct {
	Timestamp       time.Time `json:"timestamp"`
	CPUUsage        float64   `json:"cpu_usage"`
	MemoryUsage     float64   `json:"memory_usage"`
	StorageIOPS     float64   `json:"storage_iops"`
	NetworkBandwidth float64   `json:"network_bandwidth"`
	TotalVolumes    int       `json:"total_volumes"`
}

type CostEfficiencyEvent struct {
	Timestamp        time.Time `json:"timestamp"`
	TotalCost        float64   `json:"total_cost"`
	CostSavings      float64   `json:"cost_savings"`
	EfficiencyRatio  float64   `json:"efficiency_ratio"`
	OptimalTierRatio float64   `json:"optimal_tier_ratio"`
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(collectInterval time.Duration) *MetricsCollector {
	ctx, cancel := context.WithCancel(context.Background())
	return &MetricsCollector{
		volumeMetrics:   make(map[string]*VolumeMetricsHistory),
		tierMetrics:     make(map[TierLevel]*TierMetricsHistory),
		systemMetrics:   &SystemMetricsHistory{},
		collectInterval: collectInterval,
		ctx:             ctx,
		cancel:          cancel,
	}
}

// Start starts the metrics collection
func (mc *MetricsCollector) Start() {
	go mc.collectMetrics()
}

// Stop stops the metrics collection
func (mc *MetricsCollector) Stop() {
	mc.cancel()
}

// RecordVolumeAccess records a volume access event
func (mc *MetricsCollector) RecordVolumeAccess(volumeID string, accessType string, size int64, duration time.Duration, userID string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	if _, exists := mc.volumeMetrics[volumeID]; !exists {
		mc.volumeMetrics[volumeID] = &VolumeMetricsHistory{
			VolumeID: volumeID,
		}
	}

	metrics := mc.volumeMetrics[volumeID]
	metrics.mu.Lock()
	defer metrics.mu.Unlock()

	event := AccessEvent{
		Timestamp:  time.Now(),
		AccessType: accessType,
		Size:       size,
		Duration:   duration,
		UserID:     userID,
	}

	metrics.AccessHistory = append(metrics.AccessHistory, event)

	// Keep only the last 1000 events to prevent memory bloat
	if len(metrics.AccessHistory) > 1000 {
		metrics.AccessHistory = metrics.AccessHistory[len(metrics.AccessHistory)-1000:]
	}
}

// RecordTierMigration records a tier migration event
func (mc *MetricsCollector) RecordTierMigration(volumeID string, fromTier, toTier TierLevel, reason, policyName string, duration time.Duration, success bool, errorMessage string) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	// Record in volume metrics
	if _, exists := mc.volumeMetrics[volumeID]; !exists {
		mc.volumeMetrics[volumeID] = &VolumeMetricsHistory{
			VolumeID: volumeID,
		}
	}

	volumeMetrics := mc.volumeMetrics[volumeID]
	volumeMetrics.mu.Lock()
	volumeEvent := TierMigrationEvent{
		Timestamp:    time.Now(),
		FromTier:     fromTier,
		ToTier:       toTier,
		Reason:       reason,
		PolicyName:   policyName,
		Duration:     duration,
		Success:      success,
		ErrorMessage: errorMessage,
	}
	volumeMetrics.TierHistory = append(volumeMetrics.TierHistory, volumeEvent)
	volumeMetrics.mu.Unlock()

	// Record in system metrics
	mc.systemMetrics.mu.Lock()
	systemEvent := MigrationEvent{
		Timestamp:  time.Now(),
		VolumeID:   volumeID,
		FromTier:   fromTier,
		ToTier:     toTier,
		Duration:   duration,
		PolicyName: policyName,
	}
	mc.systemMetrics.TierMigrations = append(mc.systemMetrics.TierMigrations, systemEvent)
	mc.systemMetrics.mu.Unlock()
}

// RecordPolicyExecution records a policy execution event
func (mc *MetricsCollector) RecordPolicyExecution(policyName string, volumesEvaluated, volumesMoved, errorCount int, duration time.Duration) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.systemMetrics.mu.Lock()
	defer mc.systemMetrics.mu.Unlock()

	event := PolicyExecutionEvent{
		Timestamp:        time.Now(),
		PolicyName:       policyName,
		VolumesEvaluated: volumesEvaluated,
		VolumesMoved:     volumesMoved,
		Duration:         duration,
		ErrorCount:       errorCount,
	}

	mc.systemMetrics.PolicyExecutions = append(mc.systemMetrics.PolicyExecutions, event)

	// Keep only the last 500 events
	if len(mc.systemMetrics.PolicyExecutions) > 500 {
		mc.systemMetrics.PolicyExecutions = mc.systemMetrics.PolicyExecutions[len(mc.systemMetrics.PolicyExecutions)-500:]
	}
}

// GetVolumeMetrics returns metrics for a specific volume
func (mc *MetricsCollector) GetVolumeMetrics(volumeID string) (*VolumeMetricsHistory, bool) {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	metrics, exists := mc.volumeMetrics[volumeID]
	if !exists {
		return nil, false
	}

	// Return a copy to avoid data races
	metrics.mu.RLock()
	defer metrics.mu.RUnlock()

	copy := &VolumeMetricsHistory{
		VolumeID:      metrics.VolumeID,
		AccessHistory: make([]AccessEvent, len(metrics.AccessHistory)),
		TierHistory:   make([]TierMigrationEvent, len(metrics.TierHistory)),
		SizeHistory:   make([]SizeEvent, len(metrics.SizeHistory)),
		CostHistory:   make([]CostEvent, len(metrics.CostHistory)),
	}

	copySlice(metrics.AccessHistory, copy.AccessHistory)
	copySlice(metrics.TierHistory, copy.TierHistory)
	copySlice(metrics.SizeHistory, copy.SizeHistory)
	copySlice(metrics.CostHistory, copy.CostHistory)

	return copy, true
}

// GetTierMetrics returns metrics for a specific tier
func (mc *MetricsCollector) GetTierMetrics(tier TierLevel) (*TierMetricsHistory, bool) {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	metrics, exists := mc.tierMetrics[tier]
	return metrics, exists
}

// GetSystemMetrics returns overall system metrics
func (mc *MetricsCollector) GetSystemMetrics() *SystemMetricsHistory {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	mc.systemMetrics.mu.RLock()
	defer mc.systemMetrics.mu.RUnlock()

	// Return a copy
	copy := &SystemMetricsHistory{
		TierMigrations:      make([]MigrationEvent, len(mc.systemMetrics.TierMigrations)),
		PolicyExecutions:    make([]PolicyExecutionEvent, len(mc.systemMetrics.PolicyExecutions)),
		ResourceUtilization: make([]ResourceEvent, len(mc.systemMetrics.ResourceUtilization)),
		CostEfficiency:      make([]CostEfficiencyEvent, len(mc.systemMetrics.CostEfficiency)),
	}

	copySlice(mc.systemMetrics.TierMigrations, copy.TierMigrations)
	copySlice(mc.systemMetrics.PolicyExecutions, copy.PolicyExecutions)
	copySlice(mc.systemMetrics.ResourceUtilization, copy.ResourceUtilization)
	copySlice(mc.systemMetrics.CostEfficiency, copy.CostEfficiency)

	return copy
}

// CalculateAccessFrequency calculates access frequency for a volume over a time period
func (mc *MetricsCollector) CalculateAccessFrequency(volumeID string, period time.Duration) (float64, error) {
	metrics, exists := mc.GetVolumeMetrics(volumeID)
	if !exists {
		return 0, nil
	}

	since := time.Now().Add(-period)
	accessCount := 0

	for _, access := range metrics.AccessHistory {
		if access.Timestamp.After(since) {
			accessCount++
		}
	}

	days := period.Hours() / 24.0
	if days == 0 {
		days = 1
	}

	return float64(accessCount) / days, nil
}

// CalculateCostSavings calculates cost savings from tier migrations
func (mc *MetricsCollector) CalculateCostSavings(period time.Duration) float64 {
	systemMetrics := mc.GetSystemMetrics()
	since := time.Now().Add(-period)
	totalSavings := 0.0

	for _, migration := range systemMetrics.TierMigrations {
		if migration.Timestamp.After(since) {
			totalSavings += migration.CostSaving
		}
	}

	return totalSavings
}

// ExportMetrics exports metrics to JSON format
func (mc *MetricsCollector) ExportMetrics() ([]byte, error) {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	export := struct {
		VolumeMetrics map[string]*VolumeMetricsHistory `json:"volume_metrics"`
		TierMetrics   map[TierLevel]*TierMetricsHistory `json:"tier_metrics"`
		SystemMetrics *SystemMetricsHistory             `json:"system_metrics"`
		Timestamp     time.Time                         `json:"timestamp"`
	}{
		VolumeMetrics: mc.volumeMetrics,
		TierMetrics:   mc.tierMetrics,
		SystemMetrics: mc.GetSystemMetrics(),
		Timestamp:     time.Now(),
	}

	return json.MarshalIndent(export, "", "  ")
}

// collectMetrics runs periodic metrics collection
func (mc *MetricsCollector) collectMetrics() {
	ticker := time.NewTicker(mc.collectInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			mc.collectSystemResourceMetrics()
		case <-mc.ctx.Done():
			return
		}
	}
}

// collectSystemResourceMetrics collects system-level resource metrics
func (mc *MetricsCollector) collectSystemResourceMetrics() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.systemMetrics.mu.Lock()
	defer mc.systemMetrics.mu.Unlock()

	// In a real implementation, these would be gathered from system APIs
	event := ResourceEvent{
		Timestamp:        time.Now(),
		CPUUsage:         0.0,  // Would be gathered from system
		MemoryUsage:      0.0,  // Would be gathered from system
		StorageIOPS:      0.0,  // Would be gathered from storage drivers
		NetworkBandwidth: 0.0,  // Would be gathered from network interfaces
		TotalVolumes:     len(mc.volumeMetrics),
	}

	mc.systemMetrics.ResourceUtilization = append(mc.systemMetrics.ResourceUtilization, event)

	// Keep only the last 1000 resource events
	if len(mc.systemMetrics.ResourceUtilization) > 1000 {
		mc.systemMetrics.ResourceUtilization = mc.systemMetrics.ResourceUtilization[len(mc.systemMetrics.ResourceUtilization)-1000:]
	}
}

// Helper function for copying slices
func copySlice[T any](src, dst []T) {
	for i, item := range src {
		if i < len(dst) {
			dst[i] = item
		}
	}
}