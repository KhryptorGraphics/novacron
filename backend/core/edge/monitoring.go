package edge

import (
	"context"
	"sync"
	"time"
)

// EdgeMonitoring handles edge resource telemetry and monitoring
type EdgeMonitoring struct {
	config       *EdgeConfig
	discovery    *EdgeDiscovery
	metrics      map[string]*EdgeMetrics
	metricsMu    sync.RWMutex
	alerts       []MonitoringAlert
	alertsMu     sync.RWMutex
	stopCh       chan struct{}
	wg           sync.WaitGroup
}

// MonitoringAlert represents a monitoring alert
type MonitoringAlert struct {
	AlertID     string        `json:"alert_id"`
	EdgeNodeID  string        `json:"edge_node_id"`
	Severity    AlertSeverity `json:"severity"`
	Type        AlertType     `json:"type"`
	Message     string        `json:"message"`
	Value       float64       `json:"value"`
	Threshold   float64       `json:"threshold"`
	CreatedAt   time.Time     `json:"created_at"`
	ResolvedAt  *time.Time    `json:"resolved_at,omitempty"`
}

// AlertSeverity represents alert severity
type AlertSeverity string

const (
	AlertSeverityCritical AlertSeverity = "critical"
	AlertSeverityWarning  AlertSeverity = "warning"
	AlertSeverityInfo     AlertSeverity = "info"
)

// AlertType represents alert type
type AlertType string

const (
	AlertTypeHighCPU         AlertType = "high_cpu"
	AlertTypeHighMemory      AlertType = "high_memory"
	AlertTypeHighLatency     AlertType = "high_latency"
	AlertTypeNodeDown        AlertType = "node_down"
	AlertTypeMigrationFailed AlertType = "migration_failed"
	AlertTypeBandwidthLimit  AlertType = "bandwidth_limit"
)

// NewEdgeMonitoring creates a new edge monitoring instance
func NewEdgeMonitoring(config *EdgeConfig, discovery *EdgeDiscovery) *EdgeMonitoring {
	return &EdgeMonitoring{
		config:    config,
		discovery: discovery,
		metrics:   make(map[string]*EdgeMetrics),
		alerts:    make([]MonitoringAlert, 0),
		stopCh:    make(chan struct{}),
	}
}

// Start starts the monitoring service
func (em *EdgeMonitoring) Start(ctx context.Context) error {
	em.wg.Add(2)
	go em.metricsCollectionLoop(ctx)
	go em.healthCheckLoop(ctx)

	return nil
}

// Stop stops the monitoring service
func (em *EdgeMonitoring) Stop() error {
	close(em.stopCh)
	em.wg.Wait()
	return nil
}

// metricsCollectionLoop collects metrics periodically
func (em *EdgeMonitoring) metricsCollectionLoop(ctx context.Context) {
	defer em.wg.Done()

	ticker := time.NewTicker(em.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-em.stopCh:
			return
		case <-ticker.C:
			em.collectMetrics(ctx)
		}
	}
}

// healthCheckLoop performs health checks periodically
func (em *EdgeMonitoring) healthCheckLoop(ctx context.Context) {
	defer em.wg.Done()

	ticker := time.NewTicker(em.config.HealthCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-em.stopCh:
			return
		case <-ticker.C:
			em.performHealthChecks(ctx)
		}
	}
}

// collectMetrics collects metrics from all edge nodes
func (em *EdgeMonitoring) collectMetrics(ctx context.Context) {
	nodes := em.discovery.GetHealthyNodes()

	for _, node := range nodes {
		metrics := em.calculateNodeMetrics(node)

		em.metricsMu.Lock()
		em.metrics[node.ID] = metrics
		em.metricsMu.Unlock()

		// Check for alert conditions
		em.checkAlertConditions(node, metrics)
	}
}

// calculateNodeMetrics calculates metrics for a node
func (em *EdgeMonitoring) calculateNodeMetrics(node *EdgeNode) *EdgeMetrics {
	return &EdgeMetrics{
		NodeID:             node.ID,
		CPUUtilization:     float64(node.Resources.UsedCPUCores) / float64(node.Resources.TotalCPUCores) * 100.0,
		MemoryUtilization:  float64(node.Resources.UsedMemoryMB) / float64(node.Resources.TotalMemoryMB) * 100.0,
		StorageUtilization: float64(node.Resources.UsedStorageGB) / float64(node.Resources.TotalStorageGB) * 100.0,
		NetworkUtilization: float64(node.Resources.UsedBandwidthMbps) / float64(node.Resources.TotalBandwidthMbps) * 100.0,
		ActiveVMs:          node.Status.ActiveVMs,
		TotalRequests:      0, // Would be populated from actual metrics
		AvgLatencyMs:       float64(node.Latency.RTTAvg) / float64(time.Millisecond),
		P95LatencyMs:       float64(node.Latency.RTTMax) / float64(time.Millisecond),
		ErrorRate:          0.0, // Would be calculated from actual errors
		Timestamp:          time.Now(),
	}
}

// checkAlertConditions checks for alert conditions
func (em *EdgeMonitoring) checkAlertConditions(node *EdgeNode, metrics *EdgeMetrics) {
	// High CPU utilization
	if metrics.CPUUtilization > 90.0 {
		em.createAlert(node.ID, AlertSeverityCritical, AlertTypeHighCPU,
			"CPU utilization above 90%", metrics.CPUUtilization, 90.0)
	} else if metrics.CPUUtilization > 80.0 {
		em.createAlert(node.ID, AlertSeverityWarning, AlertTypeHighCPU,
			"CPU utilization above 80%", metrics.CPUUtilization, 80.0)
	}

	// High memory utilization
	if metrics.MemoryUtilization > 90.0 {
		em.createAlert(node.ID, AlertSeverityCritical, AlertTypeHighMemory,
			"Memory utilization above 90%", metrics.MemoryUtilization, 90.0)
	} else if metrics.MemoryUtilization > 80.0 {
		em.createAlert(node.ID, AlertSeverityWarning, AlertTypeHighMemory,
			"Memory utilization above 80%", metrics.MemoryUtilization, 80.0)
	}

	// High latency
	if metrics.AvgLatencyMs > 100.0 {
		em.createAlert(node.ID, AlertSeverityWarning, AlertTypeHighLatency,
			"Average latency above 100ms", metrics.AvgLatencyMs, 100.0)
	}

	// Bandwidth limit
	if metrics.NetworkUtilization > 90.0 {
		em.createAlert(node.ID, AlertSeverityWarning, AlertTypeBandwidthLimit,
			"Bandwidth utilization above 90%", metrics.NetworkUtilization, 90.0)
	}
}

// createAlert creates a new alert
func (em *EdgeMonitoring) createAlert(nodeID string, severity AlertSeverity, alertType AlertType, message string, value, threshold float64) {
	em.alertsMu.Lock()
	defer em.alertsMu.Unlock()

	alert := MonitoringAlert{
		AlertID:    generateAlertID(),
		EdgeNodeID: nodeID,
		Severity:   severity,
		Type:       alertType,
		Message:    message,
		Value:      value,
		Threshold:  threshold,
		CreatedAt:  time.Now(),
	}

	em.alerts = append(em.alerts, alert)

	// In production:
	// 1. Send to alerting system (PagerDuty, Slack, etc.)
	// 2. Create incident if critical
	// 3. Trigger auto-remediation if configured
}

// performHealthChecks performs health checks on all nodes
func (em *EdgeMonitoring) performHealthChecks(ctx context.Context) {
	nodes := em.discovery.GetAllNodes()

	for _, node := range nodes {
		healthy := em.checkNodeHealth(node)

		if !healthy && node.Status.State == EdgeNodeStateOnline {
			// Node became unhealthy
			node.Status.State = EdgeNodeStateDegraded
			node.Status.Health = HealthStatusUnhealthy

			em.createAlert(node.ID, AlertSeverityCritical, AlertTypeNodeDown,
				"Edge node health check failed", 0, 0)
		} else if healthy && node.Status.State == EdgeNodeStateDegraded {
			// Node recovered
			node.Status.State = EdgeNodeStateOnline
			node.Status.Health = HealthStatusHealthy

			// Resolve previous alerts
			em.resolveAlerts(node.ID, AlertTypeNodeDown)
		}
	}
}

// checkNodeHealth checks if a node is healthy
func (em *EdgeMonitoring) checkNodeHealth(node *EdgeNode) bool {
	// Check last seen time
	if time.Since(node.LastSeenAt) > 5*time.Minute {
		return false
	}

	// Check resource utilization
	if node.Resources.UtilizationPercent > 95.0 {
		return false
	}

	// Check error count
	if node.Status.ErrorCount > 10 {
		return false
	}

	return true
}

// resolveAlerts resolves alerts for a node
func (em *EdgeMonitoring) resolveAlerts(nodeID string, alertType AlertType) {
	em.alertsMu.Lock()
	defer em.alertsMu.Unlock()

	now := time.Now()
	for i := range em.alerts {
		if em.alerts[i].EdgeNodeID == nodeID &&
			em.alerts[i].Type == alertType &&
			em.alerts[i].ResolvedAt == nil {
			em.alerts[i].ResolvedAt = &now
		}
	}
}

// GetNodeMetrics retrieves metrics for a node
func (em *EdgeMonitoring) GetNodeMetrics(nodeID string) (*EdgeMetrics, error) {
	em.metricsMu.RLock()
	defer em.metricsMu.RUnlock()

	metrics, exists := em.metrics[nodeID]
	if !exists {
		return nil, ErrEdgeNodeNotFound
	}

	return metrics, nil
}

// GetAllMetrics retrieves metrics for all nodes
func (em *EdgeMonitoring) GetAllMetrics() []*EdgeMetrics {
	em.metricsMu.RLock()
	defer em.metricsMu.RUnlock()

	metrics := make([]*EdgeMetrics, 0, len(em.metrics))
	for _, m := range em.metrics {
		metrics = append(metrics, m)
	}

	return metrics
}

// GetActiveAlerts retrieves active alerts
func (em *EdgeMonitoring) GetActiveAlerts() []MonitoringAlert {
	em.alertsMu.RLock()
	defer em.alertsMu.RUnlock()

	active := make([]MonitoringAlert, 0)
	for _, alert := range em.alerts {
		if alert.ResolvedAt == nil {
			active = append(active, alert)
		}
	}

	return active
}

// GetUserProximityMetrics tracks user proximity to edge nodes
func (em *EdgeMonitoring) GetUserProximityMetrics(ctx context.Context) (*ProximityMetrics, error) {
	// In production:
	// 1. Track user locations (opt-in, privacy-compliant)
	// 2. Calculate distances to edge nodes
	// 3. Monitor latency experienced by users
	// 4. Identify optimal edge placements

	return &ProximityMetrics{
		TotalUsers:          1250,
		AvgDistanceKM:       45.2,
		AvgLatencyMs:        12.5,
		UsersUnder50ms:      987,
		UsersUnder100ms:     1180,
		OptimalPlacementPct: 85.4,
		Timestamp:           time.Now(),
	}, nil
}

// ProximityMetrics represents user proximity metrics
type ProximityMetrics struct {
	TotalUsers          int       `json:"total_users"`
	AvgDistanceKM       float64   `json:"avg_distance_km"`
	AvgLatencyMs        float64   `json:"avg_latency_ms"`
	UsersUnder50ms      int       `json:"users_under_50ms"`
	UsersUnder100ms     int       `json:"users_under_100ms"`
	OptimalPlacementPct float64   `json:"optimal_placement_pct"`
	Timestamp           time.Time `json:"timestamp"`
}

// GetDashboardMetrics retrieves comprehensive dashboard metrics
func (em *EdgeMonitoring) GetDashboardMetrics(ctx context.Context) (*DashboardMetrics, error) {
	nodes := em.discovery.GetAllNodes()
	healthyNodes := em.discovery.GetHealthyNodes()

	totalCPU := 0
	usedCPU := 0
	totalMemory := int64(0)
	usedMemory := int64(0)
	totalVMs := 0

	for _, node := range nodes {
		totalCPU += node.Resources.TotalCPUCores
		usedCPU += node.Resources.UsedCPUCores
		totalMemory += node.Resources.TotalMemoryMB
		usedMemory += node.Resources.UsedMemoryMB
		totalVMs += node.Status.ActiveVMs
	}

	activeAlerts := em.GetActiveAlerts()
	criticalAlerts := 0
	for _, alert := range activeAlerts {
		if alert.Severity == AlertSeverityCritical {
			criticalAlerts++
		}
	}

	return &DashboardMetrics{
		TotalEdgeNodes:    len(nodes),
		HealthyNodes:      len(healthyNodes),
		TotalVMs:          totalVMs,
		TotalCPUCores:     totalCPU,
		UsedCPUCores:      usedCPU,
		CPUUtilizationPct: float64(usedCPU) / float64(totalCPU) * 100.0,
		TotalMemoryMB:     totalMemory,
		UsedMemoryMB:      usedMemory,
		MemUtilizationPct: float64(usedMemory) / float64(totalMemory) * 100.0,
		ActiveAlerts:      len(activeAlerts),
		CriticalAlerts:    criticalAlerts,
		AvgLatencyMs:      calculateAvgLatency(nodes),
		Timestamp:         time.Now(),
	}, nil
}

// DashboardMetrics represents dashboard metrics
type DashboardMetrics struct {
	TotalEdgeNodes    int       `json:"total_edge_nodes"`
	HealthyNodes      int       `json:"healthy_nodes"`
	TotalVMs          int       `json:"total_vms"`
	TotalCPUCores     int       `json:"total_cpu_cores"`
	UsedCPUCores      int       `json:"used_cpu_cores"`
	CPUUtilizationPct float64   `json:"cpu_utilization_pct"`
	TotalMemoryMB     int64     `json:"total_memory_mb"`
	UsedMemoryMB      int64     `json:"used_memory_mb"`
	MemUtilizationPct float64   `json:"mem_utilization_pct"`
	ActiveAlerts      int       `json:"active_alerts"`
	CriticalAlerts    int       `json:"critical_alerts"`
	AvgLatencyMs      float64   `json:"avg_latency_ms"`
	Timestamp         time.Time `json:"timestamp"`
}

// calculateAvgLatency calculates average latency across nodes
func calculateAvgLatency(nodes []*EdgeNode) float64 {
	if len(nodes) == 0 {
		return 0
	}

	total := time.Duration(0)
	for _, node := range nodes {
		total += node.Latency.RTTAvg
	}

	avg := total / time.Duration(len(nodes))
	return float64(avg) / float64(time.Millisecond)
}

// generateAlertID generates a unique alert ID
func generateAlertID() string {
	return fmt.Sprintf("alert-%d", time.Now().UnixNano())
}

// ExportMetrics exports metrics for external systems (Prometheus, etc.)
func (em *EdgeMonitoring) ExportMetrics(ctx context.Context) ([]byte, error) {
	// In production, export in Prometheus format:
	// # HELP edge_cpu_utilization Edge node CPU utilization
	// # TYPE edge_cpu_utilization gauge
	// edge_cpu_utilization{node="edge-1"} 45.2

	return nil, nil
}
