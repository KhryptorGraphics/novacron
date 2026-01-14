package security

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

// SecurityDashboard provides real-time security metrics and monitoring
type SecurityDashboard struct {
	auditEngine       *SecurityAuditEngine
	continuousMonitor *ContinuousMonitor
	metricsCollector  *DashboardMetricsCollector
	alertManager      *DashboardAlertManager
	webSocketManager  *WebSocketManager
	config           *DashboardConfig
	router           *mux.Router
	server           *http.Server
	isRunning        bool
	mutex            sync.RWMutex
}

// DashboardConfig holds dashboard configuration
type DashboardConfig struct {
	Port                int           `json:"port"`
	RefreshInterval     time.Duration `json:"refresh_interval"`
	DataRetentionPeriod time.Duration `json:"data_retention_period"`
	EnableRealTime      bool          `json:"enable_real_time"`
	EnableAuth          bool          `json:"enable_auth"`
	AuthToken          string        `json:"auth_token,omitempty"`
	TLSCertPath        string        `json:"tls_cert_path,omitempty"`
	TLSKeyPath         string        `json:"tls_key_path,omitempty"`
	EnableMetrics       bool          `json:"enable_metrics"`
	MetricsEndpoint    string        `json:"metrics_endpoint"`
}

// DashboardMetricsCollector aggregates metrics for the dashboard
type DashboardMetricsCollector struct {
	metrics          map[string]*DashboardMetric
	historicalData   map[string][]HistoricalDataPoint
	alertMetrics     *AlertMetrics
	complianceMetrics *ComplianceMetrics
	performanceMetrics *PerformanceMetrics
	mutex            sync.RWMutex
	lastUpdate       time.Time
}

// DashboardMetric represents a dashboard metric
type DashboardMetric struct {
	ID             string                 `json:"id"`
	Name           string                 `json:"name"`
	Value          interface{}            `json:"value"`
	Unit           string                 `json:"unit"`
	Type           MetricType             `json:"type"`
	Category       MetricCategory         `json:"category"`
	Severity       MetricSeverity         `json:"severity"`
	Threshold      *DashboardThreshold    `json:"threshold,omitempty"`
	Trend          TrendIndicator         `json:"trend"`
	LastUpdate     time.Time             `json:"last_update"`
	Description    string                 `json:"description"`
	Tags           []string               `json:"tags"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// MetricCategory categorizes dashboard metrics
type MetricCategory string

const (
	CategorySecurity    MetricCategory = "security"
	CategoryCompliance  MetricCategory = "compliance"
	CategoryIncidents   MetricCategory = "incidents"
	CategoryVulnerabilities MetricCategory = "vulnerabilities"
	CategoryPerformance MetricCategory = "performance"
	CategorySystem      MetricCategory = "system"
	CategoryUsers       MetricCategory = "users"
)

// MetricSeverity defines metric severity levels
type MetricSeverity string

const (
	SeverityNormal   MetricSeverity = "normal"
	SeverityWarning  MetricSeverity = "warning"
	SeverityError    MetricSeverity = "error"
	SeverityCritical MetricSeverity = "critical"
)

// TrendIndicator shows metric trend direction
type TrendIndicator string

const (
	TrendUp     TrendIndicator = "up"
	TrendDown   TrendIndicator = "down"
	TrendStable TrendIndicator = "stable"
	TrendUnknown TrendIndicator = "unknown"
)

// DashboardThreshold defines threshold values for metrics
type DashboardThreshold struct {
	Warning  float64 `json:"warning"`
	Error    float64 `json:"error"`
	Critical float64 `json:"critical"`
	Operator string  `json:"operator"`
}

// HistoricalDataPoint represents a historical data point
type HistoricalDataPoint struct {
	Timestamp time.Time   `json:"timestamp"`
	Value     interface{} `json:"value"`
	Label     string      `json:"label,omitempty"`
}

// AlertMetrics tracks alert-related metrics
type AlertMetrics struct {
	TotalAlerts        int                    `json:"total_alerts"`
	ActiveAlerts       int                    `json:"active_alerts"`
	CriticalAlerts     int                    `json:"critical_alerts"`
	ResolvedToday      int                    `json:"resolved_today"`
	AverageResponseTime time.Duration         `json:"average_response_time"`
	AlertsByCategory   map[AlertType]int      `json:"alerts_by_category"`
	AlertTrend         []HistoricalDataPoint  `json:"alert_trend"`
	TopAlerts          []SecurityAlert        `json:"top_alerts"`
}

// ComplianceMetrics tracks compliance-related metrics
type ComplianceMetrics struct {
	OverallScore       float64                        `json:"overall_score"`
	FrameworkScores    map[string]float64             `json:"framework_scores"`
	TotalControls      int                           `json:"total_controls"`
	CompliantControls  int                           `json:"compliant_controls"`
	FailedControls     int                           `json:"failed_controls"`
	PolicyViolations   int                           `json:"policy_violations"`
	ComplianceTrend    []HistoricalDataPoint         `json:"compliance_trend"`
	FrameworkDetails   map[string]*FrameworkMetric   `json:"framework_details"`
}

// FrameworkMetric provides detailed framework metrics
type FrameworkMetric struct {
	Score          float64               `json:"score"`
	Status         AssessmentStatus      `json:"status"`
	LastAssessment time.Time            `json:"last_assessment"`
	NextAssessment time.Time            `json:"next_assessment"`
	ControlStats   map[string]int       `json:"control_stats"`
	RecentFindings []ComplianceFinding  `json:"recent_findings"`
}

// PerformanceMetrics tracks system performance metrics
type PerformanceMetrics struct {
	ScanDuration       time.Duration         `json:"scan_duration"`
	AuditPerformance   float64              `json:"audit_performance"`
	SystemLoad         float64              `json:"system_load"`
	MemoryUsage        float64              `json:"memory_usage"`
	DiskUsage          float64              `json:"disk_usage"`
	NetworkLatency     time.Duration        `json:"network_latency"`
	ThroughputMetrics  map[string]float64   `json:"throughput_metrics"`
}

// DashboardAlertManager manages dashboard-specific alerts
type DashboardAlertManager struct {
	rules           []DashboardAlertRule
	activeAlerts    map[string]*DashboardAlert
	webhooks        []WebhookEndpoint
	notifications   []NotificationConfig
	mutex           sync.RWMutex
}

// DashboardAlertRule defines alerting rules for dashboard metrics
type DashboardAlertRule struct {
	ID               string                 `json:"id"`
	Name             string                 `json:"name"`
	MetricID         string                 `json:"metric_id"`
	Condition        string                 `json:"condition"`
	Threshold        float64                `json:"threshold"`
	Duration         time.Duration          `json:"duration"`
	Severity         AlertSeverity          `json:"severity"`
	Enabled          bool                   `json:"enabled"`
	Notifications    []string               `json:"notifications"`
	Metadata         map[string]interface{} `json:"metadata"`
}

// DashboardAlert represents a dashboard alert
type DashboardAlert struct {
	ID             string                 `json:"id"`
	RuleID         string                 `json:"rule_id"`
	MetricID       string                 `json:"metric_id"`
	Title          string                 `json:"title"`
	Description    string                 `json:"description"`
	Severity       AlertSeverity          `json:"severity"`
	Status         AlertStatus            `json:"status"`
	TriggeredAt    time.Time             `json:"triggered_at"`
	ResolvedAt     *time.Time            `json:"resolved_at,omitempty"`
	Value          interface{}           `json:"value"`
	Threshold      float64               `json:"threshold"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// WebhookEndpoint defines webhook configuration
type WebhookEndpoint struct {
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
	Enabled bool              `json:"enabled"`
}

// NotificationConfig defines notification configuration
type NotificationConfig struct {
	Type    string                 `json:"type"`
	Target  string                 `json:"target"`
	Config  map[string]interface{} `json:"config"`
	Enabled bool                   `json:"enabled"`
}

// WebSocketManager manages real-time WebSocket connections
type WebSocketManager struct {
	upgrader    websocket.Upgrader
	connections map[string]*websocket.Conn
	mutex       sync.RWMutex
}

// DashboardData aggregates all dashboard data
type DashboardData struct {
	Summary            *SecuritySummary       `json:"summary"`
	Metrics            map[string]*DashboardMetric `json:"metrics"`
	AlertMetrics       *AlertMetrics          `json:"alert_metrics"`
	ComplianceMetrics  *ComplianceMetrics     `json:"compliance_metrics"`
	PerformanceMetrics *PerformanceMetrics    `json:"performance_metrics"`
	RecentIncidents    []SecurityIncident     `json:"recent_incidents"`
	SystemHealth       *SystemHealthStatus    `json:"system_health"`
	Timestamp          time.Time             `json:"timestamp"`
}

// SecuritySummary provides high-level security overview
type SecuritySummary struct {
	OverallScore       float64           `json:"overall_score"`
	RiskLevel          RiskLevel         `json:"risk_level"`
	TotalVulnerabilities int             `json:"total_vulnerabilities"`
	CriticalIssues     int               `json:"critical_issues"`
	ActiveIncidents    int               `json:"active_incidents"`
	ComplianceScore    float64           `json:"compliance_score"`
	LastScanTime       time.Time         `json:"last_scan_time"`
	TrendIndicator     TrendIndicator    `json:"trend_indicator"`
}

// SystemHealthStatus tracks system health
type SystemHealthStatus struct {
	Overall       float64                   `json:"overall"`
	Components    map[string]ComponentHealth `json:"components"`
	Uptime        time.Duration             `json:"uptime"`
	LastCheck     time.Time                 `json:"last_check"`
	Issues        []HealthIssue             `json:"issues"`
}

// ComponentHealth tracks individual component health
type ComponentHealth struct {
	Name      string    `json:"name"`
	Status    string    `json:"status"`
	Health    float64   `json:"health"`
	LastCheck time.Time `json:"last_check"`
	Message   string    `json:"message"`
}

// NewSecurityDashboard creates a new security dashboard
func NewSecurityDashboard(auditEngine *SecurityAuditEngine, continuousMonitor *ContinuousMonitor, config *DashboardConfig) *SecurityDashboard {
	dashboard := &SecurityDashboard{
		auditEngine:       auditEngine,
		continuousMonitor: continuousMonitor,
		metricsCollector:  NewDashboardMetricsCollector(),
		alertManager:      NewDashboardAlertManager(),
		webSocketManager:  NewWebSocketManager(),
		config:           config,
		router:           mux.NewRouter(),
		isRunning:        false,
	}
	
	dashboard.setupRoutes()
	dashboard.initializeMetrics()
	
	return dashboard
}

// Start starts the security dashboard server
func (sd *SecurityDashboard) Start(ctx context.Context) error {
	sd.mutex.Lock()
	defer sd.mutex.Unlock()
	
	if sd.isRunning {
		return fmt.Errorf("dashboard is already running")
	}
	
	// Setup HTTP server
	sd.server = &http.Server{
		Addr:         fmt.Sprintf(":%d", sd.config.Port),
		Handler:      sd.router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	
	// Start metrics collection
	go sd.startMetricsCollection(ctx)
	
	// Start real-time updates if enabled
	if sd.config.EnableRealTime {
		go sd.startRealTimeUpdates(ctx)
	}
	
	// Start alert processing
	go sd.startAlertProcessing(ctx)
	
	sd.isRunning = true
	
	// Start server
	go func() {
		var err error
		if sd.config.TLSCertPath != "" && sd.config.TLSKeyPath != "" {
			fmt.Printf("🔒 Security dashboard starting with HTTPS on port %d\n", sd.config.Port)
			err = sd.server.ListenAndServeTLS(sd.config.TLSCertPath, sd.config.TLSKeyPath)
		} else {
			fmt.Printf("📊 Security dashboard starting on port %d\n", sd.config.Port)
			err = sd.server.ListenAndServe()
		}
		
		if err != nil && err != http.ErrServerClosed {
			fmt.Printf("Dashboard server error: %v\n", err)
		}
	}()
	
	return nil
}

// Stop stops the security dashboard server
func (sd *SecurityDashboard) Stop(ctx context.Context) error {
	sd.mutex.Lock()
	defer sd.mutex.Unlock()
	
	if !sd.isRunning {
		return fmt.Errorf("dashboard is not running")
	}
	
	// Close WebSocket connections
	sd.webSocketManager.CloseAllConnections()
	
	// Shutdown server
	if sd.server != nil {
		return sd.server.Shutdown(ctx)
	}
	
	sd.isRunning = false
	return nil
}

// setupRoutes configures HTTP routes
func (sd *SecurityDashboard) setupRoutes() {
	// API routes
	api := sd.router.PathPrefix("/api/v1").Subrouter()
	
	if sd.config.EnableAuth {
		api.Use(sd.authMiddleware)
	}
	
	// Dashboard data endpoints
	api.HandleFunc("/dashboard", sd.getDashboardData).Methods("GET")
	api.HandleFunc("/summary", sd.getSummary).Methods("GET")
	api.HandleFunc("/metrics", sd.getMetrics).Methods("GET")
	api.HandleFunc("/metrics/{category}", sd.getMetricsByCategory).Methods("GET")
	api.HandleFunc("/alerts", sd.getAlerts).Methods("GET")
	api.HandleFunc("/incidents", sd.getIncidents).Methods("GET")
	api.HandleFunc("/compliance", sd.getComplianceStatus).Methods("GET")
	api.HandleFunc("/health", sd.getSystemHealth).Methods("GET")
	
	// Real-time WebSocket endpoint
	api.HandleFunc("/ws", sd.handleWebSocket)
	
	// Management endpoints
	api.HandleFunc("/scan", sd.triggerScan).Methods("POST")
	api.HandleFunc("/alerts/{id}/acknowledge", sd.acknowledgeAlert).Methods("POST")
	api.HandleFunc("/incidents/{id}/resolve", sd.resolveIncident).Methods("POST")
	
	// Historical data endpoints
	api.HandleFunc("/history/{metric}", sd.getHistoricalData).Methods("GET")
	api.HandleFunc("/trends", sd.getTrends).Methods("GET")
	
	// Configuration endpoints
	api.HandleFunc("/config", sd.getConfig).Methods("GET")
	api.HandleFunc("/config", sd.updateConfig).Methods("PUT")
	
	// Static files for web dashboard
	sd.router.PathPrefix("/").Handler(http.FileServer(http.Dir("./web/dashboard/")))
}

// Authentication middleware
func (sd *SecurityDashboard) authMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := r.Header.Get("Authorization")
		if token != "Bearer "+sd.config.AuthToken {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// getDashboardData returns complete dashboard data
func (sd *SecurityDashboard) getDashboardData(w http.ResponseWriter, r *http.Request) {
	data, err := sd.collectDashboardData()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

// getSummary returns security summary
func (sd *SecurityDashboard) getSummary(w http.ResponseWriter, r *http.Request) {
	summary := sd.generateSummary()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(summary)
}

// getMetrics returns all metrics
func (sd *SecurityDashboard) getMetrics(w http.ResponseWriter, r *http.Request) {
	sd.metricsCollector.mutex.RLock()
	metrics := sd.metricsCollector.metrics
	sd.metricsCollector.mutex.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// getMetricsByCategory returns metrics by category
func (sd *SecurityDashboard) getMetricsByCategory(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	category := MetricCategory(vars["category"])
	
	sd.metricsCollector.mutex.RLock()
	filteredMetrics := make(map[string]*DashboardMetric)
	for id, metric := range sd.metricsCollector.metrics {
		if metric.Category == category {
			filteredMetrics[id] = metric
		}
	}
	sd.metricsCollector.mutex.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(filteredMetrics)
}

// getAlerts returns current alerts
func (sd *SecurityDashboard) getAlerts(w http.ResponseWriter, r *http.Request) {
	sd.alertManager.mutex.RLock()
	alerts := sd.alertManager.activeAlerts
	sd.alertManager.mutex.RUnlock()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(alerts)
}

// getIncidents returns recent incidents
func (sd *SecurityDashboard) getIncidents(w http.ResponseWriter, r *http.Request) {
	incidents := sd.auditEngine.incidentResponder.ListAuditSessions()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(incidents)
}

// getComplianceStatus returns compliance status
func (sd *SecurityDashboard) getComplianceStatus(w http.ResponseWriter, r *http.Request) {
	compliance := sd.metricsCollector.complianceMetrics
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(compliance)
}

// getSystemHealth returns system health status
func (sd *SecurityDashboard) getSystemHealth(w http.ResponseWriter, r *http.Request) {
	health := sd.generateSystemHealth()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

// handleWebSocket handles WebSocket connections for real-time updates
func (sd *SecurityDashboard) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	conn, err := sd.webSocketManager.upgrader.Upgrade(w, r, nil)
	if err != nil {
		fmt.Printf("WebSocket upgrade failed: %v\n", err)
		return
	}
	
	clientID := r.Header.Get("X-Client-ID")
	if clientID == "" {
		clientID = fmt.Sprintf("client-%d", time.Now().UnixNano())
	}
	
	sd.webSocketManager.addConnection(clientID, conn)
	
	// Handle connection
	go sd.handleWebSocketConnection(clientID, conn)
}

// startMetricsCollection starts periodic metrics collection
func (sd *SecurityDashboard) startMetricsCollection(ctx context.Context) {
	ticker := time.NewTicker(sd.config.RefreshInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			sd.collectAndUpdateMetrics()
		case <-ctx.Done():
			return
		}
	}
}

// startRealTimeUpdates starts real-time updates to WebSocket clients
func (sd *SecurityDashboard) startRealTimeUpdates(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second) // Real-time updates every 5 seconds
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			data, err := sd.collectDashboardData()
			if err != nil {
				continue
			}
			
			sd.webSocketManager.broadcastUpdate("dashboard_update", data)
		case <-ctx.Done():
			return
		}
	}
}

// startAlertProcessing starts alert processing
func (sd *SecurityDashboard) startAlertProcessing(ctx context.Context) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ticker.C:
			sd.processAlerts()
		case <-ctx.Done():
			return
		}
	}
}

// collectDashboardData aggregates all dashboard data
func (sd *SecurityDashboard) collectDashboardData() (*DashboardData, error) {
	sd.metricsCollector.mutex.RLock()
	defer sd.metricsCollector.mutex.RUnlock()
	
	data := &DashboardData{
		Summary:            sd.generateSummary(),
		Metrics:            sd.metricsCollector.metrics,
		AlertMetrics:       sd.metricsCollector.alertMetrics,
		ComplianceMetrics:  sd.metricsCollector.complianceMetrics,
		PerformanceMetrics: sd.metricsCollector.performanceMetrics,
		SystemHealth:       sd.generateSystemHealth(),
		Timestamp:          time.Now(),
	}
	
	// Get recent incidents
	sessions := sd.auditEngine.ListAuditSessions()
	if len(sessions) > 10 {
		sessions = sessions[:10] // Limit to 10 most recent
	}
	
	for _, session := range sessions {
		if session.Results != nil && session.Results.IncidentResults != nil {
			// Convert audit sessions to incidents for display
			// This is a simplified conversion - real implementation would be more detailed
		}
	}
	
	return data, nil
}

// generateSummary creates security summary
func (sd *SecurityDashboard) generateSummary() *SecuritySummary {
	// Get latest audit results
	sessions := sd.auditEngine.ListAuditSessions()
	if len(sessions) == 0 {
		return &SecuritySummary{
			RiskLevel:          RiskLevelMedium,
			TrendIndicator:     TrendUnknown,
			LastScanTime:       time.Time{},
		}
	}
	
	// Get most recent completed session
	var latestSession *AuditSession
	for _, session := range sessions {
		if session.Status == AuditStatusCompleted && session.Results != nil {
			if latestSession == nil || session.StartTime.After(latestSession.StartTime) {
				latestSession = session
			}
		}
	}
	
	if latestSession == nil {
		return &SecuritySummary{
			RiskLevel:          RiskLevelMedium,
			TrendIndicator:     TrendUnknown,
			LastScanTime:       time.Time{},
		}
	}
	
	results := latestSession.Results
	
	return &SecuritySummary{
		OverallScore:         results.OverallScore,
		RiskLevel:           results.RiskLevel,
		TotalVulnerabilities: results.VulnerabilityResults.Total,
		CriticalIssues:       results.VulnerabilityResults.Critical,
		ActiveIncidents:      results.IncidentResults.OpenIncidents,
		ComplianceScore:      results.ComplianceResults.OverallScore,
		LastScanTime:         latestSession.StartTime,
		TrendIndicator:       sd.calculateTrend(results.OverallScore),
	}
}

// generateSystemHealth creates system health status
func (sd *SecurityDashboard) generateSystemHealth() *SystemHealthStatus {
	results, _ := sd.continuousMonitor.GetStatus(context.Background())
	
	components := make(map[string]ComponentHealth)
	for name, status := range results.MonitorStatus {
		health := 100.0
		if !status.Healthy {
			health = 50.0 - float64(status.ErrorCount*10)
		}
		if health < 0 {
			health = 0
		}
		
		components[name] = ComponentHealth{
			Name:      name,
			Status:    map[bool]string{true: "healthy", false: "unhealthy"}[status.Healthy],
			Health:    health,
			LastCheck: status.LastCheck,
			Message:   status.Details,
		}
	}
	
	return &SystemHealthStatus{
		Overall:    results.HealthScore,
		Components: components,
		Uptime:     results.Uptime,
		LastCheck:  time.Now(),
		Issues:     []HealthIssue{}, // Would be populated from actual health checker
	}
}

// calculateTrend determines trend indicator based on score history
func (sd *SecurityDashboard) calculateTrend(currentScore float64) TrendIndicator {
	// Simple trend calculation - in real implementation would use historical data
	if currentScore >= 80 {
		return TrendUp
	} else if currentScore >= 60 {
		return TrendStable
	} else {
		return TrendDown
	}
}

// collectAndUpdateMetrics collects and updates all metrics
func (sd *SecurityDashboard) collectAndUpdateMetrics() {
	// Collect vulnerability metrics
	sd.updateVulnerabilityMetrics()
	
	// Collect compliance metrics
	sd.updateComplianceMetrics()
	
	// Collect incident metrics
	sd.updateIncidentMetrics()
	
	// Collect performance metrics
	sd.updatePerformanceMetrics()
	
	// Update historical data
	sd.updateHistoricalData()
}

// updateVulnerabilityMetrics updates vulnerability-related metrics
func (sd *SecurityDashboard) updateVulnerabilityMetrics() {
	sessions := sd.auditEngine.ListAuditSessions()
	if len(sessions) == 0 {
		return
	}
	
	// Find latest completed session with results
	var latest *AuditSession
	for _, session := range sessions {
		if session.Status == AuditStatusCompleted && session.Results != nil {
			if latest == nil || session.StartTime.After(latest.StartTime) {
				latest = session
			}
		}
	}
	
	if latest == nil || latest.Results.VulnerabilityResults == nil {
		return
	}
	
	vulnResults := latest.Results.VulnerabilityResults
	
	// Update vulnerability metrics
	sd.metricsCollector.updateMetric("total_vulnerabilities", &DashboardMetric{
		ID:          "total_vulnerabilities",
		Name:        "Total Vulnerabilities",
		Value:       vulnResults.Total,
		Unit:        "count",
		Type:        MetricTypeCounter,
		Category:    CategoryVulnerabilities,
		Severity:    sd.getSeverityFromCount(vulnResults.Critical),
		LastUpdate:  time.Now(),
		Description: "Total number of identified vulnerabilities",
		Tags:        []string{"security", "vulnerabilities"},
	})
	
	sd.metricsCollector.updateMetric("critical_vulnerabilities", &DashboardMetric{
		ID:          "critical_vulnerabilities",
		Name:        "Critical Vulnerabilities",
		Value:       vulnResults.Critical,
		Unit:        "count",
		Type:        MetricTypeCounter,
		Category:    CategoryVulnerabilities,
		Severity:    vulnResults.Critical > 0 && SeverityCritical || SeverityNormal,
		Threshold:   &DashboardThreshold{Warning: 1, Error: 3, Critical: 5, Operator: "gte"},
		LastUpdate:  time.Now(),
		Description: "Number of critical severity vulnerabilities",
		Tags:        []string{"security", "critical"},
	})
	
	sd.metricsCollector.updateMetric("high_vulnerabilities", &DashboardMetric{
		ID:          "high_vulnerabilities",
		Name:        "High Vulnerabilities",
		Value:       vulnResults.High,
		Unit:        "count",
		Type:        MetricTypeCounter,
		Category:    CategoryVulnerabilities,
		Severity:    sd.getSeverityFromCount(vulnResults.High),
		LastUpdate:  time.Now(),
		Description: "Number of high severity vulnerabilities",
		Tags:        []string{"security", "high"},
	})
}

// updateComplianceMetrics updates compliance-related metrics
func (sd *SecurityDashboard) updateComplianceMetrics() {
	sessions := sd.auditEngine.ListAuditSessions()
	if len(sessions) == 0 {
		return
	}
	
	// Find latest session with compliance results
	var latest *AuditSession
	for _, session := range sessions {
		if session.Status == AuditStatusCompleted && session.Results != nil && session.Results.ComplianceResults != nil {
			if latest == nil || session.StartTime.After(latest.StartTime) {
				latest = session
			}
		}
	}
	
	if latest == nil {
		return
	}
	
	compResults := latest.Results.ComplianceResults
	
	sd.metricsCollector.updateMetric("compliance_score", &DashboardMetric{
		ID:          "compliance_score",
		Name:        "Compliance Score",
		Value:       compResults.OverallScore,
		Unit:        "percentage",
		Type:        MetricTypeGauge,
		Category:    CategoryCompliance,
		Severity:    sd.getSeverityFromScore(compResults.OverallScore),
		Threshold:   &DashboardThreshold{Warning: 70, Error: 50, Critical: 30, Operator: "lte"},
		LastUpdate:  time.Now(),
		Description: "Overall compliance score across all frameworks",
		Tags:        []string{"compliance", "overall"},
	})
	
	// Update framework-specific scores
	for framework, assessment := range compResults.FrameworkResults {
		metricID := fmt.Sprintf("compliance_%s", framework)
		sd.metricsCollector.updateMetric(metricID, &DashboardMetric{
			ID:          metricID,
			Name:        fmt.Sprintf("%s Compliance", framework),
			Value:       assessment.OverallScore,
			Unit:        "percentage",
			Type:        MetricTypeGauge,
			Category:    CategoryCompliance,
			Severity:    sd.getSeverityFromScore(assessment.OverallScore),
			LastUpdate:  time.Now(),
			Description: fmt.Sprintf("Compliance score for %s framework", framework),
			Tags:        []string{"compliance", framework},
		})
	}
}

// updateIncidentMetrics updates incident-related metrics
func (sd *SecurityDashboard) updateIncidentMetrics() {
	sessions := sd.auditEngine.ListAuditSessions()
	if len(sessions) == 0 {
		return
	}
	
	// Count active incidents across all sessions
	activeIncidents := 0
	totalIncidents := 0
	
	for _, session := range sessions {
		if session.Results != nil && session.Results.IncidentResults != nil {
			activeIncidents += session.Results.IncidentResults.OpenIncidents
			totalIncidents += session.Results.IncidentResults.OpenIncidents + session.Results.IncidentResults.UnresolvedCritical
		}
	}
	
	sd.metricsCollector.updateMetric("active_incidents", &DashboardMetric{
		ID:          "active_incidents",
		Name:        "Active Incidents",
		Value:       activeIncidents,
		Unit:        "count",
		Type:        MetricTypeGauge,
		Category:    CategoryIncidents,
		Severity:    sd.getSeverityFromCount(activeIncidents),
		Threshold:   &DashboardThreshold{Warning: 5, Error: 10, Critical: 20, Operator: "gte"},
		LastUpdate:  time.Now(),
		Description: "Number of currently active security incidents",
		Tags:        []string{"incidents", "active"},
	})
}

// updatePerformanceMetrics updates system performance metrics
func (sd *SecurityDashboard) updatePerformanceMetrics() {
	results, err := sd.continuousMonitor.GetStatus(context.Background())
	if err != nil {
		return
	}
	
	sd.metricsCollector.updateMetric("system_health", &DashboardMetric{
		ID:          "system_health",
		Name:        "System Health Score",
		Value:       results.HealthScore,
		Unit:        "percentage",
		Type:        MetricTypeGauge,
		Category:    CategorySystem,
		Severity:    sd.getSeverityFromScore(results.HealthScore),
		Threshold:   &DashboardThreshold{Warning: 80, Error: 60, Critical: 40, Operator: "lte"},
		LastUpdate:  time.Now(),
		Description: "Overall system health score",
		Tags:        []string{"system", "health"},
	})
}

// updateHistoricalData updates historical data for trending
func (sd *SecurityDashboard) updateHistoricalData() {
	now := time.Now()
	
	sd.metricsCollector.mutex.Lock()
	defer sd.metricsCollector.mutex.Unlock()
	
	for metricID, metric := range sd.metricsCollector.metrics {
		if sd.metricsCollector.historicalData[metricID] == nil {
			sd.metricsCollector.historicalData[metricID] = []HistoricalDataPoint{}
		}
		
		// Add current value to history
		sd.metricsCollector.historicalData[metricID] = append(
			sd.metricsCollector.historicalData[metricID],
			HistoricalDataPoint{
				Timestamp: now,
				Value:     metric.Value,
			},
		)
		
		// Keep only last 24 hours of data (assuming 5-minute intervals)
		maxPoints := 288 // 24 hours * 60 minutes / 5 minutes
		if len(sd.metricsCollector.historicalData[metricID]) > maxPoints {
			sd.metricsCollector.historicalData[metricID] = sd.metricsCollector.historicalData[metricID][len(sd.metricsCollector.historicalData[metricID])-maxPoints:]
		}
	}
}

// getSeverityFromCount determines severity based on count
func (sd *SecurityDashboard) getSeverityFromCount(count int) MetricSeverity {
	if count == 0 {
		return SeverityNormal
	} else if count <= 5 {
		return SeverityWarning
	} else if count <= 15 {
		return SeverityError
	} else {
		return SeverityCritical
	}
}

// getSeverityFromScore determines severity based on score
func (sd *SecurityDashboard) getSeverityFromScore(score float64) MetricSeverity {
	if score >= 90 {
		return SeverityNormal
	} else if score >= 70 {
		return SeverityWarning
	} else if score >= 50 {
		return SeverityError
	} else {
		return SeverityCritical
	}
}

// processAlerts evaluates metrics and triggers alerts
func (sd *SecurityDashboard) processAlerts() {
	sd.metricsCollector.mutex.RLock()
	metrics := sd.metricsCollector.metrics
	sd.metricsCollector.mutex.RUnlock()
	
	sd.alertManager.mutex.Lock()
	defer sd.alertManager.mutex.Unlock()
	
	for _, rule := range sd.alertManager.rules {
		if !rule.Enabled {
			continue
		}
		
		metric, exists := metrics[rule.MetricID]
		if !exists {
			continue
		}
		
		// Evaluate alert condition
		triggered := sd.evaluateAlertCondition(rule, metric)
		
		existingAlert, hasActiveAlert := sd.alertManager.activeAlerts[rule.ID]
		
		if triggered && !hasActiveAlert {
			// Create new alert
			alert := &DashboardAlert{
				ID:          fmt.Sprintf("%s-%d", rule.ID, time.Now().Unix()),
				RuleID:      rule.ID,
				MetricID:    rule.MetricID,
				Title:       fmt.Sprintf("Alert: %s", rule.Name),
				Description: fmt.Sprintf("Metric %s has exceeded threshold", metric.Name),
				Severity:    rule.Severity,
				Status:      AlertStatusOpen,
				TriggeredAt: time.Now(),
				Value:       metric.Value,
				Threshold:   rule.Threshold,
				Metadata:    make(map[string]interface{}),
			}
			
			sd.alertManager.activeAlerts[rule.ID] = alert
			
			// Send notifications
			sd.sendAlertNotifications(alert, rule)
			
		} else if !triggered && hasActiveAlert && existingAlert.Status == AlertStatusOpen {
			// Resolve alert
			now := time.Now()
			existingAlert.Status = AlertStatusResolved
			existingAlert.ResolvedAt = &now
		}
	}
}

// evaluateAlertCondition evaluates if an alert condition is met
func (sd *SecurityDashboard) evaluateAlertCondition(rule DashboardAlertRule, metric *DashboardMetric) bool {
	value, ok := metric.Value.(float64)
	if !ok {
		// Try to convert to float64
		switch v := metric.Value.(type) {
		case int:
			value = float64(v)
		case int64:
			value = float64(v)
		default:
			return false
		}
	}
	
	switch rule.Condition {
	case "gt":
		return value > rule.Threshold
	case "gte":
		return value >= rule.Threshold
	case "lt":
		return value < rule.Threshold
	case "lte":
		return value <= rule.Threshold
	case "eq":
		return value == rule.Threshold
	case "ne":
		return value != rule.Threshold
	default:
		return false
	}
}

// sendAlertNotifications sends alert notifications
func (sd *SecurityDashboard) sendAlertNotifications(alert *DashboardAlert, rule DashboardAlertRule) {
	// Send to WebSocket clients
	sd.webSocketManager.broadcastUpdate("alert", alert)
	
	// Send to configured notification channels
	for _, notificationID := range rule.Notifications {
		for _, notification := range sd.alertManager.notifications {
			if notification.Type == notificationID && notification.Enabled {
				sd.sendNotification(notification, alert)
			}
		}
	}
	
	// Send to webhooks
	for _, webhook := range sd.alertManager.webhooks {
		if webhook.Enabled {
			sd.sendWebhookNotification(webhook, alert)
		}
	}
}

// sendNotification sends a notification
func (sd *SecurityDashboard) sendNotification(notification NotificationConfig, alert *DashboardAlert) {
	// Implementation would depend on notification type (email, slack, etc.)
	fmt.Printf("Sending %s notification for alert: %s\n", notification.Type, alert.Title)
}

// sendWebhookNotification sends a webhook notification
func (sd *SecurityDashboard) sendWebhookNotification(webhook WebhookEndpoint, alert *DashboardAlert) {
	// Implementation would send HTTP POST to webhook URL
	fmt.Printf("Sending webhook notification to %s for alert: %s\n", webhook.URL, alert.Title)
}

// Component initialization and helper functions

// NewDashboardMetricsCollector creates a new dashboard metrics collector
func NewDashboardMetricsCollector() *DashboardMetricsCollector {
	return &DashboardMetricsCollector{
		metrics:          make(map[string]*DashboardMetric),
		historicalData:   make(map[string][]HistoricalDataPoint),
		alertMetrics:     &AlertMetrics{},
		complianceMetrics: &ComplianceMetrics{},
		performanceMetrics: &PerformanceMetrics{},
	}
}

// updateMetric updates or creates a metric
func (dmc *DashboardMetricsCollector) updateMetric(id string, metric *DashboardMetric) {
	dmc.mutex.Lock()
	defer dmc.mutex.Unlock()
	
	dmc.metrics[id] = metric
	dmc.lastUpdate = time.Now()
}

// NewDashboardAlertManager creates a new dashboard alert manager
func NewDashboardAlertManager() *DashboardAlertManager {
	return &DashboardAlertManager{
		rules:        []DashboardAlertRule{},
		activeAlerts: make(map[string]*DashboardAlert),
		webhooks:     []WebhookEndpoint{},
		notifications: []NotificationConfig{},
	}
}

// NewWebSocketManager creates a new WebSocket manager
func NewWebSocketManager() *WebSocketManager {
	return &WebSocketManager{
		upgrader: websocket.Upgrader{
			CheckOrigin: func(r *http.Request) bool {
				// In production, implement proper origin checking
				return true
			},
		},
		connections: make(map[string]*websocket.Conn),
	}
}

// addConnection adds a WebSocket connection
func (wsm *WebSocketManager) addConnection(clientID string, conn *websocket.Conn) {
	wsm.mutex.Lock()
	defer wsm.mutex.Unlock()
	
	wsm.connections[clientID] = conn
}

// removeConnection removes a WebSocket connection
func (wsm *WebSocketManager) removeConnection(clientID string) {
	wsm.mutex.Lock()
	defer wsm.mutex.Unlock()
	
	if conn, exists := wsm.connections[clientID]; exists {
		conn.Close()
		delete(wsm.connections, clientID)
	}
}

// broadcastUpdate broadcasts an update to all connected clients
func (wsm *WebSocketManager) broadcastUpdate(messageType string, data interface{}) {
	wsm.mutex.RLock()
	defer wsm.mutex.RUnlock()
	
	message := map[string]interface{}{
		"type":      messageType,
		"data":      data,
		"timestamp": time.Now(),
	}
	
	for clientID, conn := range wsm.connections {
		err := conn.WriteJSON(message)
		if err != nil {
			fmt.Printf("Failed to send update to client %s: %v\n", clientID, err)
			// Remove failed connection
			go wsm.removeConnection(clientID)
		}
	}
}

// CloseAllConnections closes all WebSocket connections
func (wsm *WebSocketManager) CloseAllConnections() {
	wsm.mutex.Lock()
	defer wsm.mutex.Unlock()
	
	for clientID, conn := range wsm.connections {
		conn.Close()
		delete(wsm.connections, clientID)
	}
}

// handleWebSocketConnection handles a WebSocket connection
func (sd *SecurityDashboard) handleWebSocketConnection(clientID string, conn *websocket.Conn) {
	defer sd.webSocketManager.removeConnection(clientID)
	
	// Send initial data
	data, err := sd.collectDashboardData()
	if err == nil {
		conn.WriteJSON(map[string]interface{}{
			"type": "initial_data",
			"data": data,
			"timestamp": time.Now(),
		})
	}
	
	// Keep connection alive and handle client messages
	for {
		var msg map[string]interface{}
		err := conn.ReadJSON(&msg)
		if err != nil {
			fmt.Printf("WebSocket connection %s closed: %v\n", clientID, err)
			break
		}
		
		// Handle client messages (ping, requests, etc.)
		sd.handleWebSocketMessage(clientID, conn, msg)
	}
}

// handleWebSocketMessage handles incoming WebSocket messages
func (sd *SecurityDashboard) handleWebSocketMessage(clientID string, conn *websocket.Conn, msg map[string]interface{}) {
	msgType, ok := msg["type"].(string)
	if !ok {
		return
	}
	
	switch msgType {
	case "ping":
		conn.WriteJSON(map[string]interface{}{
			"type": "pong",
			"timestamp": time.Now(),
		})
	case "request_update":
		data, err := sd.collectDashboardData()
		if err == nil {
			conn.WriteJSON(map[string]interface{}{
				"type": "dashboard_update",
				"data": data,
				"timestamp": time.Now(),
			})
		}
	}
}

// initializeMetrics sets up initial metrics and alert rules
func (sd *SecurityDashboard) initializeMetrics() {
	// Initialize default alert rules
	sd.alertManager.rules = []DashboardAlertRule{
		{
			ID:           "critical_vulns_alert",
			Name:         "Critical Vulnerabilities Alert",
			MetricID:     "critical_vulnerabilities",
			Condition:    "gt",
			Threshold:    0,
			Duration:     time.Minute,
			Severity:     AlertSeverityCritical,
			Enabled:      true,
			Notifications: []string{"email", "webhook"},
		},
		{
			ID:           "compliance_score_alert",
			Name:         "Low Compliance Score Alert",
			MetricID:     "compliance_score",
			Condition:    "lt",
			Threshold:    70,
			Duration:     5 * time.Minute,
			Severity:     AlertSeverityHigh,
			Enabled:      true,
			Notifications: []string{"email"},
		},
		{
			ID:           "system_health_alert",
			Name:         "System Health Alert",
			MetricID:     "system_health",
			Condition:    "lt",
			Threshold:    60,
			Duration:     time.Minute,
			Severity:     AlertSeverityMedium,
			Enabled:      true,
			Notifications: []string{"webhook"},
		},
	}
}