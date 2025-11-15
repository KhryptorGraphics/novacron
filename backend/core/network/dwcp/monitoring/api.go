package monitoring

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// MonitoringAPI provides unified monitoring API
type MonitoringAPI struct {
	mu sync.RWMutex

	// Components
	metricsCollector *MetricsCollector
	tracingSystem    *TracingSystem
	dashboard        *DashboardManager
	alerting         *AlertingSystem
	anomalyDetector  *AnomalyDetector
	healthMonitor    *HealthMonitor
	slaMonitor       *SLAMonitor
	logAggregator    *LogAggregator
	networkTelemetry *NetworkTelemetry
	profiler         *Profiler
	capacityPlanner  *CapacityPlanner

	// Configuration
	region string
}

// TimeRange represents a time range for queries
type TimeRange struct {
	Start time.Time
	End   time.Time
}

// APIConfig configures the monitoring API connections
type APIConfig struct {
	Region           string
	PrometheusURL    string
	GrafanaURL       string
	GrafanaAPIKey    string
	JaegerEndpoint   string
	ElasticsearchURL string
}

// NewMonitoringAPI creates a new monitoring API
func NewMonitoringAPI(apiConfig *APIConfig, monConfig *MonitoringConfig) (*MonitoringAPI, error) {
	// Initialize metrics collector
	metricsCollector, err := NewMetricsCollector(apiConfig.Region)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics collector: %w", err)
	}

	// Initialize tracing system
	traceConfig := &TraceConfig{
		ServiceName:    "dwcp",
		JaegerEndpoint: apiConfig.JaegerEndpoint,
		SamplingRate:   0.1,
		Strategy:       SamplingAdaptive,
	}
	tracingSystem, err := NewTracingSystem(traceConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create tracing system: %w", err)
	}

	// Initialize dashboard manager
	dashboardConfig := &DashboardConfig{
		GrafanaURL: apiConfig.GrafanaURL,
		APIKey:     apiConfig.GrafanaAPIKey,
	}
	dashboard := NewDashboardManager(dashboardConfig)

	// Initialize other components
	alerting := NewAlertingSystem()
	anomalyDetector, err := NewAnomalyDetector(monConfig.Detector, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create anomaly detector: %w", err)
	}
	healthMonitor := NewHealthMonitor()
	slaMonitor := NewSLAMonitor()
	logAggregator := NewLogAggregator(apiConfig.ElasticsearchURL, "dwcp-logs")
	networkTelemetry := NewNetworkTelemetry()
	profiler := NewProfiler()
	capacityPlanner := NewCapacityPlanner()

	api := &MonitoringAPI{
		metricsCollector: metricsCollector,
		tracingSystem:    tracingSystem,
		dashboard:        dashboard,
		alerting:         alerting,
		anomalyDetector:  anomalyDetector,
		healthMonitor:    healthMonitor,
		slaMonitor:       slaMonitor,
		logAggregator:    logAggregator,
		networkTelemetry: networkTelemetry,
		profiler:         profiler,
		capacityPlanner:  capacityPlanner,
		region:           apiConfig.Region,
	}

	return api, nil
}

// GetMetrics retrieves metrics for a region and time range
func (api *MonitoringAPI) GetMetrics(region string, timeRange TimeRange) (*MetricData, error) {
	api.mu.RLock()
	defer api.mu.RUnlock()

	metrics := api.metricsCollector.GetMetrics(timeRange.Start)

	if len(metrics) == 0 {
		return nil, fmt.Errorf("no metrics found for region %s", region)
	}

	// Return first metric as example (production would aggregate)
	return metrics[0], nil
}

// GetTraces retrieves traces matching filter
func (api *MonitoringAPI) GetTraces(filter TraceFilter) ([]*StoredTrace, error) {
	api.mu.RLock()
	defer api.mu.RUnlock()

	traces := api.tracingSystem.GetTraces(filter)
	return traces, nil
}

// GetAlerts retrieves alerts by severity
func (api *MonitoringAPI) GetAlerts(severity AlertSeverity) ([]*Alert, error) {
	api.mu.RLock()
	defer api.mu.RUnlock()

	alerts := api.alerting.GetActiveAlerts(severity)
	return alerts, nil
}

// GetRegionHealth retrieves health score for a region
func (api *MonitoringAPI) GetRegionHealth(region string) (*HealthScore, error) {
	api.mu.RLock()
	defer api.mu.RUnlock()

	health, ok := api.healthMonitor.GetRegionHealth(region)
	if !ok {
		return nil, fmt.Errorf("health data not found for region: %s", region)
	}

	return health, nil
}

// GetSLACompliance retrieves SLA compliance report
func (api *MonitoringAPI) GetSLACompliance(slaID string) (*SLAReport, error) {
	api.mu.RLock()
	defer api.mu.RUnlock()

	report, ok := api.slaMonitor.GetSLACompliance(slaID)
	if !ok {
		return nil, fmt.Errorf("SLA report not found: %s", slaID)
	}

	return report, nil
}

// RecordMetric records a metric
func (api *MonitoringAPI) RecordMetric(ctx context.Context, name string, value float64, labels map[string]string) error {
	api.metricsCollector.RecordGauge(name, value, labels)
	return nil
}

// RecordTrace records a trace
func (api *MonitoringAPI) RecordTrace(ctx context.Context, name string) (context.Context, func()) {
	ctx, span := api.tracingSystem.StartSpan(ctx, name)

	return ctx, func() {
		span.End()
	}
}

// RecordLog records a log entry
func (api *MonitoringAPI) RecordLog(entry *LogEntry) error {
	return api.logAggregator.Ingest(entry)
}

// TriggerAlert manually triggers an alert
func (api *MonitoringAPI) TriggerAlert(alert *Alert) error {
	api.mu.Lock()
	defer api.mu.Unlock()

	// Evaluate and route alert
	receivers := api.alerting.RouteAlert(alert)

	// Send to receivers (simplified)
	fmt.Printf("Alert routed to receivers: %v\n", receivers)

	return nil
}

// DetectAnomaly detects anomaly in a metric
func (api *MonitoringAPI) DetectAnomaly(metricName string, value float64) (*AnomalyResult, error) {
	result := api.anomalyDetector.DetectAnomaly(metricName, value)
	return result, nil
}

// GenerateDashboard generates a dashboard
func (api *MonitoringAPI) GenerateDashboard(dashboardType string) error {
	api.mu.Lock()
	defer api.mu.Unlock()

	switch dashboardType {
	case "global":
		return api.dashboard.CreateGlobalOverviewDashboard()
	case "regional":
		return api.dashboard.CreateRegionalDashboard(api.region)
	case "protocol":
		return api.dashboard.CreateDWCPProtocolDashboard()
	case "loadbalancer":
		return api.dashboard.CreateLoadBalancerDashboard()
	default:
		return fmt.Errorf("unknown dashboard type: %s", dashboardType)
	}
}

// GetCapacityForecast retrieves capacity forecast
func (api *MonitoringAPI) GetCapacityForecast(resource string) (*CapacityForecast, error) {
	forecast, err := api.capacityPlanner.GenerateForecast(resource)
	if err != nil {
		return nil, err
	}
	return forecast, nil
}

// GetNetworkMetrics retrieves network metrics
func (api *MonitoringAPI) GetNetworkMetrics(source, dest string) (map[string]interface{}, error) {
	bandwidth, _ := api.networkTelemetry.GetBandwidthMetrics(source, dest)
	latency, _ := api.networkTelemetry.GetLatencyMetrics(source, dest)

	metrics := make(map[string]interface{})
	if bandwidth != nil {
		metrics["bandwidth"] = bandwidth
	}
	if latency != nil {
		metrics["latency"] = latency
	}

	return metrics, nil
}

// StartProfiling starts profiling
func (api *MonitoringAPI) StartProfiling(profileType ProfileType, duration time.Duration) error {
	api.mu.Lock()
	defer api.mu.Unlock()

	switch profileType {
	case ProfileCPU:
		return api.profiler.StartCPUProfile(duration)
	case ProfileMemory:
		_, err := api.profiler.CaptureMemoryProfile()
		return err
	case ProfileGoroutine:
		_, err := api.profiler.CaptureGoroutineProfile()
		return err
	default:
		return fmt.Errorf("unsupported profile type: %v", profileType)
	}
}

// GetAllMetrics retrieves all monitoring metrics
func (api *MonitoringAPI) GetAllMetrics() map[string]interface{} {
	api.mu.RLock()
	defer api.mu.RUnlock()

	return map[string]interface{}{
		"tracing":  api.tracingSystem.GetStatistics(),
		"logs":     api.logAggregator.GetStatistics(),
		"network":  api.networkTelemetry.GetAllMetrics(),
		"profiler": api.profiler.GetStatistics(),
	}
}

// Shutdown gracefully shuts down monitoring system
func (api *MonitoringAPI) Shutdown(ctx context.Context) error {
	api.mu.Lock()
	defer api.mu.Unlock()

	if err := api.metricsCollector.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown metrics collector: %w", err)
	}

	if err := api.tracingSystem.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown tracing system: %w", err)
	}

	return nil
}

// Initialize initializes all monitoring components
func (api *MonitoringAPI) Initialize(ctx context.Context) error {
	api.mu.Lock()
	defer api.mu.Unlock()

	// Create default dashboards
	if err := api.dashboard.CreateGlobalOverviewDashboard(); err != nil {
		return fmt.Errorf("failed to create global dashboard: %w", err)
	}

	// Create default SLAs
	availabilitySLA := api.slaMonitor.CreateAvailabilitySLA("dwcp", 99.95)
	api.slaMonitor.DefineSLA(availabilitySLA)

	latencySLA := api.slaMonitor.CreateLatencySLA("dwcp", 100)
	api.slaMonitor.DefineSLA(latencySLA)

	// Start auto-cleanup tasks
	go api.metricsCollector.StartAutoCleanup(ctx, 1*time.Hour)
	go api.logAggregator.StartAutoCleanup(ctx, 1*time.Hour)

	return nil
}
