package security

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// SecurityMonitor provides real-time security monitoring and alerting
type SecurityMonitor struct {
	config              SecurityMonitorConfig
	alertManager        *AlertManager
	threatDetector      *ThreatDetectionEngine
	incidentManager     *SecurityIncidentManager
	metricsCollector    *SecurityMetricsCollector
	auditLogger         audit.AuditLogger
	subscribers         map[string]chan SecurityEvent
	eventQueue          chan SecurityEvent
	spilloverQueue      *SpilloverQueue // Disk-backed overflow queue
	backpressureManager *EventQueueBackpressureManager
	mu                  sync.RWMutex
}

// SecurityMonitorConfig defines monitoring configuration
type SecurityMonitorConfig struct {
	EnableRealTimeMonitoring   bool                    `json:"enable_realtime"`
	ThreatDetectionConfig      ThreatDetectionConfig   `json:"threat_detection"`
	AlertingConfig             AlertingConfig          `json:"alerting"`
	IncidentManagementConfig   IncidentConfig          `json:"incident_management"`
	ComplianceMonitoringConfig ComplianceMonitorConfig `json:"compliance_monitoring"`
	MetricsConfig              MetricsConfig           `json:"metrics"`
	NotificationConfig         NotificationConfig      `json:"notifications"`
}

// MonitoringMonitoringThreatDetectionConfig defines threat detection parameters
type ThreatDetectionConfig struct {
	EnableBehaviorAnalysis bool              `json:"enable_behavior_analysis"`
	EnableAnomalyDetection bool              `json:"enable_anomaly_detection"`
	EnableMLDetection      bool              `json:"enable_ml_detection"`
	BruteForceThreshold    int               `json:"brute_force_threshold"`
	RateLimitThreshold     int               `json:"rate_limit_threshold"`
	GeoLocationEnabled     bool              `json:"geolocation_enabled"`
	AllowedCountries       []string          `json:"allowed_countries"`
	ThreatIntelFeeds       []ThreatIntelFeed `json:"threat_intel_feeds"`
	CustomRules            []DetectionRule   `json:"custom_rules"`
}

// ThreatIntelFeed represents external threat intelligence feed
type ThreatIntelFeed struct {
	Name       string        `json:"name"`
	URL        string        `json:"url"`
	APIKey     string        `json:"api_key"`
	UpdateFreq time.Duration `json:"update_frequency"`
	LastUpdate time.Time     `json:"last_update"`
	Enabled    bool          `json:"enabled"`
	Categories []string      `json:"categories"`
}

// DetectionRule represents custom security detection rule
type DetectionRule struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Severity    ThreatLevel       `json:"severity"`
	Pattern     string            `json:"pattern"`
	Conditions  map[string]string `json:"conditions"`
	Actions     []string          `json:"actions"`
	Enabled     bool              `json:"enabled"`
}

// AlertingConfig defines alerting behavior
type AlertingConfig struct {
	EnableEmailAlerts bool                    `json:"enable_email_alerts"`
	EnableSlackAlerts bool                    `json:"enable_slack_alerts"`
	EnableSMSAlerts   bool                    `json:"enable_sms_alerts"`
	EnableWebhooks    bool                    `json:"enable_webhooks"`
	EscalationPolicy  EscalationPolicy        `json:"escalation_policy"`
	AlertThresholds   map[string]int          `json:"alert_thresholds"`
	SuppressionsRules []SuppressionRule       `json:"suppression_rules"`
	AlertChannels     map[string]AlertChannel `json:"alert_channels"`
}

// EscalationPolicy defines alert escalation behavior
type EscalationPolicy struct {
	Levels         []EscalationLevel `json:"levels"`
	DefaultTimeout time.Duration     `json:"default_timeout"`
	MaxEscalations int               `json:"max_escalations"`
	AutoResolve    bool              `json:"auto_resolve"`
	ResolveTimeout time.Duration     `json:"resolve_timeout"`
}

// EscalationLevel represents escalation tier
type EscalationLevel struct {
	Level      int           `json:"level"`
	Recipients []string      `json:"recipients"`
	Channels   []string      `json:"channels"`
	Timeout    time.Duration `json:"timeout"`
	Actions    []string      `json:"actions"`
}

// SuppressionRule defines when to suppress alerts
type SuppressionRule struct {
	ID         string            `json:"id"`
	Name       string            `json:"name"`
	Pattern    string            `json:"pattern"`
	Conditions map[string]string `json:"conditions"`
	Duration   time.Duration     `json:"duration"`
	Enabled    bool              `json:"enabled"`
}

// AlertChannel defines alert delivery channel
type AlertChannel struct {
	Type     string            `json:"type"`
	Endpoint string            `json:"endpoint"`
	APIKey   string            `json:"api_key"`
	Template string            `json:"template"`
	Enabled  bool              `json:"enabled"`
	Metadata map[string]string `json:"metadata"`
}

// IncidentConfig defines incident management settings
type IncidentConfig struct {
	EnableAutoIncidents bool                     `json:"enable_auto_incidents"`
	IncidentThresholds  map[string]int           `json:"incident_thresholds"`
	AutoAssignmentRules []AssignmentRule         `json:"auto_assignment_rules"`
	SLATargets          map[string]time.Duration `json:"sla_targets"`
	IntegrationConfig   IntegrationConfig        `json:"integrations"`
}

// AssignmentRule defines automatic incident assignment
type AssignmentRule struct {
	ID        string            `json:"id"`
	Condition map[string]string `json:"condition"`
	Assignee  string            `json:"assignee"`
	Team      string            `json:"team"`
	Priority  IncidentPriority  `json:"priority"`
	Enabled   bool              `json:"enabled"`
}

// MonitoringMonitoringIntegrationConfig defines external integrations
type IntegrationConfig struct {
	JiraEnabled      bool   `json:"jira_enabled"`
	JiraURL          string `json:"jira_url"`
	JiraToken        string `json:"jira_token"`
	SlackEnabled     bool   `json:"slack_enabled"`
	SlackWebhook     string `json:"slack_webhook"`
	PagerDutyEnabled bool   `json:"pagerduty_enabled"`
	PagerDutyToken   string `json:"pagerduty_token"`
}

// ComplianceMonitorConfig defines compliance monitoring
type ComplianceMonitorConfig struct {
	EnableSOC2Monitoring   bool              `json:"enable_soc2"`
	EnableISO27001Monitor  bool              `json:"enable_iso27001"`
	EnableGDPRMonitoring   bool              `json:"enable_gdpr"`
	EnableHIPAAMonitoring  bool              `json:"enable_hipaa"`
	EnablePCIDSSMonitoring bool              `json:"enable_pci_dss"`
	ReportingFrequency     time.Duration     `json:"reporting_frequency"`
	AutoRemediation        bool              `json:"auto_remediation"`
	ComplianceChecks       []ComplianceCheck `json:"compliance_checks"`
}

// ComplianceCheck represents compliance validation
type ComplianceCheck struct {
	ID          string                 `json:"id"`
	Framework   string                 `json:"framework"`
	Control     string                 `json:"control"`
	Description string                 `json:"description"`
	Severity    ComplianceSeverity     `json:"severity"`
	Frequency   time.Duration          `json:"frequency"`
	Query       string                 `json:"query"`
	Threshold   map[string]interface{} `json:"threshold"`
	Remediation string                 `json:"remediation"`
	Enabled     bool                   `json:"enabled"`
}

// MetricsConfig defines security metrics collection
type MetricsConfig struct {
	EnablePrometheus    bool            `json:"enable_prometheus"`
	EnableCustomMetrics bool            `json:"enable_custom_metrics"`
	MetricRetention     time.Duration   `json:"metric_retention"`
	CollectionInterval  time.Duration   `json:"collection_interval"`
	AggregationPeriods  []time.Duration `json:"aggregation_periods"`
}

// MonitoringMonitoringSecurityEvent represents a security event
type SecurityEvent struct {
	ID          string                 `json:"id"`
	Type        SecurityEventType      `json:"type"`
	Timestamp   time.Time              `json:"timestamp"`
	Source      string                 `json:"source"`
	Severity    ThreatLevel            `json:"severity"`
	UserID      string                 `json:"user_id"`
	IP          string                 `json:"ip"`
	UserAgent   string                 `json:"user_agent"`
	Endpoint    string                 `json:"endpoint"`
	Method      string                 `json:"method"`
	StatusCode  int                    `json:"status_code"`
	Message     string                 `json:"message"`
	Details     map[string]interface{} `json:"details"`
	Location    *GeoLocation           `json:"location,omitempty"`
	ThreatIntel *ThreatIntelData       `json:"threat_intel,omitempty"`
	RiskScore   float64                `json:"risk_score"`
	Remediated  bool                   `json:"remediated"`
}

// MonitoringMonitoringMonitoringSecurityEventType represents type of security event
type SecurityEventType string

const (
	EventAuthFailure         SecurityEventType = "auth_failure"
	EventBruteForceAttempt   SecurityEventType = "brute_force_attempt"
	EventRateLimitExceeded   SecurityEventType = "rate_limit_exceeded"
	EventSuspiciousActivity  SecurityEventType = "suspicious_activity"
	EventUnauthorizedAccess  SecurityEventType = "unauthorized_access"
	EventDataBreach          SecurityEventType = "data_breach"
	EventMalwareDetected     SecurityEventType = "malware_detected"
	EventComplianceViolation SecurityEventType = "compliance_violation"
	EventVulnerabilityFound  SecurityEventType = "vulnerability_found"
	EventIncidentCreated     SecurityEventType = "incident_created"
)

// ThreatLevel represents severity levels
type ThreatLevel string

const (
	ThreatLevelCritical ThreatLevel = "critical"
	ThreatLevelHigh     ThreatLevel = "high"
	ThreatLevelMedium   ThreatLevel = "medium"
	ThreatLevelLow      ThreatLevel = "low"
	ThreatLevelInfo     ThreatLevel = "info"
)

// ComplianceSeverity represents compliance violation severity
type ComplianceSeverity string

const (
	ComplianceCritical ComplianceSeverity = "critical"
	ComplianceHigh     ComplianceSeverity = "high"
	ComplianceMedium   ComplianceSeverity = "medium"
	ComplianceLow      ComplianceSeverity = "low"
)

// IncidentPriority represents incident priority
type IncidentPriority string

const (
	IncidentP1 IncidentPriority = "p1" // Critical
	IncidentP2 IncidentPriority = "p2" // High
	IncidentP3 IncidentPriority = "p3" // Medium
	IncidentP4 IncidentPriority = "p4" // Low
)

// GeoLocation represents geographical location
type GeoLocation struct {
	Country     string  `json:"country"`
	CountryCode string  `json:"country_code"`
	City        string  `json:"city"`
	Region      string  `json:"region"`
	Latitude    float64 `json:"latitude"`
	Longitude   float64 `json:"longitude"`
	ISP         string  `json:"isp"`
	ASN         string  `json:"asn"`
}

// ThreatIntelData represents threat intelligence information
type ThreatIntelData struct {
	IsMalicious bool              `json:"is_malicious"`
	ThreatType  string            `json:"threat_type"`
	Confidence  float64           `json:"confidence"`
	Source      string            `json:"source"`
	LastSeen    time.Time         `json:"last_seen"`
	Categories  []string          `json:"categories"`
	Indicators  map[string]string `json:"indicators"`
}

// Prometheus metrics
var (
	monitoringSecurityEventsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_security_events_total",
			Help: "Total number of security events",
		},
		[]string{"type", "severity", "source"},
	)

	alertsTriggeredTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_security_alerts_total",
			Help: "Total number of security alerts triggered",
		},
		[]string{"type", "severity", "channel"},
	)

	incidentsCreatedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_security_incidents_total",
			Help: "Total number of security incidents created",
		},
		[]string{"severity", "type", "status"},
	)

	complianceViolationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "novacron_compliance_violations_total",
			Help: "Total number of compliance violations",
		},
		[]string{"framework", "control", "severity"},
	)

	threatDetectionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "novacron_threat_detection_duration_seconds",
			Help: "Time taken to detect and process threats",
		},
		[]string{"type"},
	)

	// Backpressure metrics
	securityEventsDroppedTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "novacron_security_events_dropped_total",
			Help: "Total number of security events dropped due to backpressure",
		},
	)

	eventQueueSize = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_security_event_queue_size",
			Help: "Current size of the security event queue",
		},
	)

	eventQueueCapacity = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_security_event_queue_capacity",
			Help: "Maximum capacity of the security event queue",
		},
	)

	spilloverQueueSize = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_security_spillover_queue_size",
			Help: "Current size of the spillover queue",
		},
	)

	backpressureActive = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "novacron_security_backpressure_active",
			Help: "Whether backpressure is currently active (1 = active, 0 = inactive)",
		},
	)
)

// NewSecurityMonitor creates a new security monitor instance
func NewSecurityMonitor(config SecurityMonitorConfig, auditLogger audit.AuditLogger) (*SecurityMonitor, error) {
	sm := &SecurityMonitor{
		config:      config,
		auditLogger: auditLogger,
		subscribers: make(map[string]chan SecurityEvent),
		eventQueue:  make(chan SecurityEvent, 10000),
	}

	// Initialize alert manager
	alertManager, err := NewAlertManager(config.AlertingConfig, auditLogger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize alert manager: %w", err)
	}
	sm.alertManager = alertManager

	// Initialize threat detector
	threatDetector, err := NewThreatDetectionEngine(config.ThreatDetectionConfig, auditLogger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize threat detector: %w", err)
	}
	sm.threatDetector = threatDetector

	// Initialize incident manager
	incidentManager, err := NewSecurityIncidentManager(config.IncidentManagementConfig, auditLogger)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize incident manager: %w", err)
	}
	sm.incidentManager = incidentManager

	// Initialize metrics collector
	metricsCollector, err := NewSecurityMetricsCollector(config.MetricsConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize metrics collector: %w", err)
	}
	sm.metricsCollector = metricsCollector

	// Start background processes
	go sm.eventProcessor()
	go sm.complianceMonitor()
	go sm.threatIntelUpdater()

	return sm, nil
}

// ProcessMonitoringSecurityEvent processes a security event
func (sm *SecurityMonitor) ProcessSecurityEvent(event SecurityEvent) error {
	// Add timestamp if not set
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}

	// Generate ID if not set
	if event.ID == "" {
		event.ID = generateEventID()
	}

	// Enrich event with threat intelligence
	if sm.config.ThreatDetectionConfig.EnableMLDetection {
		sm.enrichEventWithThreatIntel(&event)
	}

	// Calculate risk score
	event.RiskScore = sm.calculateRiskScore(event)

	// Queue event for processing with backpressure handling
	select {
	case sm.eventQueue <- event:
		// Event queued successfully
		eventQueueSize.Set(float64(len(sm.eventQueue)))
	default:
		// Queue is full, try spillover queue
		if sm.spilloverQueue != nil {
			if err := sm.spilloverQueue.Enqueue(event); err != nil {
				// Spillover also full, increment dropped metric
				securityEventsDroppedTotal.Inc()

				// Log event drop with context
				sm.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
					Action:   audit.ActionUpdate,
					Resource: "event_queue",
					Result:   audit.ResultFailure,
					Details: map[string]interface{}{
						"event_id":       event.ID,
						"reason":         "queue_full",
						"queue_size":     len(sm.eventQueue),
						"spillover_size": sm.spilloverQueue.Size(),
					},
				})

				// Activate backpressure
				backpressureActive.Set(1)

				// Apply sampling if configured
				if sm.backpressureManager != nil {
					if sm.backpressureManager.ShouldDrop() {
						return fmt.Errorf("event dropped due to backpressure")
					}
				}

				return fmt.Errorf("security event queue and spillover are full")
			}
			spilloverQueueSize.Set(float64(sm.spilloverQueue.Size()))
		} else {
			// No spillover configured, drop event
			securityEventsDroppedTotal.Inc()
			return fmt.Errorf("security event queue is full, no spillover configured")
		}
	}

	// Update metrics
	monitoringSecurityEventsTotal.WithLabelValues(
		string(event.Type),
		string(event.Severity),
		event.Source,
	).Inc()

	return nil
}

// eventProcessor processes security events from the queue
func (sm *SecurityMonitor) eventProcessor() {
	for event := range sm.eventQueue {
		start := time.Now()

		// Process with threat detector
		threatResult, err := sm.threatDetector.AnalyzeEvent(event)
		if err != nil {
			sm.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
				Action:   audit.ActionRead,
				Resource: "threat_analysis",
				Result:   audit.ResultFailure,
				Details:  map[string]interface{}{"description": fmt.Sprintf("Failed to analyze event %s: %v", event.ID, err)},
			})
		}

		// Check if alert should be triggered
		if sm.shouldTriggerAlert(event, threatResult) {
			alert := sm.createAlert(event, threatResult)
			if err := sm.alertManager.TriggerAlert(alert); err != nil {
				sm.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
					Action:   audit.ActionCreate,
					Resource: "security_alert",
					Result:   audit.ResultFailure,
					Details:  map[string]interface{}{"description": fmt.Sprintf("Failed to trigger alert for event %s: %v", event.ID, err)},
				})
			}
		}

		// Check if incident should be created
		if sm.shouldCreateIncident(event, threatResult) {
			incident := sm.createIncident(event, threatResult)
			if err := sm.incidentManager.CreateIncident(incident); err != nil {
				sm.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
					Action:   audit.ActionCreate,
					Resource: "security_incident",
					Result:   audit.ResultFailure,
					Details:  map[string]interface{}{"description": fmt.Sprintf("Failed to create incident for event %s: %v", event.ID, err)},
				})
			}
		}

		// Notify subscribers
		sm.notifySubscribers(event)

		// Update metrics
		threatDetectionLatency.WithLabelValues(string(event.Type)).Observe(time.Since(start).Seconds())

		// Audit log
		sm.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
			Action:   audit.ActionRead,
			Resource: "security_event",
			Result:   audit.ResultSuccess,
			Details:  map[string]interface{}{"description": fmt.Sprintf("Processed security event %s of type %s", event.ID, event.Type)},
		})
	}
}

// Subscribe subscribes to security events
func (sm *SecurityMonitor) Subscribe(subscriberID string) <-chan SecurityEvent {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	ch := make(chan SecurityEvent, 100)
	sm.subscribers[subscriberID] = ch
	return ch
}

// Unsubscribe unsubscribes from security events
func (sm *SecurityMonitor) Unsubscribe(subscriberID string) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	if ch, exists := sm.subscribers[subscriberID]; exists {
		close(ch)
		delete(sm.subscribers, subscriberID)
	}
}

// notifySubscribers notifies all subscribers of a security event
func (sm *SecurityMonitor) notifySubscribers(event SecurityEvent) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	for _, ch := range sm.subscribers {
		select {
		case ch <- event:
			// Event sent successfully
		default:
			// Subscriber channel is full, skip
		}
	}
}

// enrichEventWithThreatIntel enriches event with threat intelligence data
func (sm *SecurityMonitor) enrichEventWithThreatIntel(event *SecurityEvent) {
	if event.IP == "" {
		return
	}

	// Check threat intelligence for IP
	if threatIntel := sm.threatDetector.GetThreatIntel(event.IP); threatIntel != nil {
		event.ThreatIntel = threatIntel

		// Increase risk score if IP is malicious
		if threatIntel.IsMalicious {
			event.RiskScore += 0.3
		}
	}

	// Get geolocation data
	if location := sm.threatDetector.GetGeoLocation(event.IP); location != nil {
		event.Location = location

		// Check if country is allowed
		if !sm.isCountryAllowed(location.CountryCode) {
			event.RiskScore += 0.2
		}
	}
}

// calculateRiskScore calculates risk score for an event
func (sm *SecurityMonitor) calculateRiskScore(event SecurityEvent) float64 {
	score := 0.0

	// Base score by event type
	switch event.Type {
	case EventAuthFailure:
		score = 0.3
	case EventBruteForceAttempt:
		score = 0.7
	case EventRateLimitExceeded:
		score = 0.4
	case EventSuspiciousActivity:
		score = 0.6
	case EventUnauthorizedAccess:
		score = 0.8
	case EventDataBreach:
		score = 1.0
	case EventMalwareDetected:
		score = 0.9
	case EventComplianceViolation:
		score = 0.5
	}

	// Adjust by severity
	switch event.Severity {
	case ThreatLevelCritical:
		score *= 1.5
	case ThreatLevelHigh:
		score *= 1.2
	case ThreatLevelMedium:
		score *= 1.0
	case ThreatLevelLow:
		score *= 0.8
	case ThreatLevelInfo:
		score *= 0.5
	}

	// Cap at 1.0
	if score > 1.0 {
		score = 1.0
	}

	return score
}

// shouldTriggerAlert determines if alert should be triggered
func (sm *SecurityMonitor) shouldTriggerAlert(event SecurityEvent, threatResult *ThreatAnalysisResult) bool {
	// Always alert on critical events
	if event.Severity == ThreatLevelCritical {
		return true
	}

	// Alert on high-risk events
	if event.RiskScore >= 0.7 {
		return true
	}

	// Alert on specific event types
	switch event.Type {
	case EventBruteForceAttempt, EventUnauthorizedAccess, EventDataBreach, EventMalwareDetected:
		return true
	}

	// Check threat analysis result
	if threatResult != nil && threatResult.IsThreat && threatResult.Confidence >= 0.8 {
		return true
	}

	return false
}

// shouldCreateIncident determines if incident should be created
func (sm *SecurityMonitor) shouldCreateIncident(event SecurityEvent, threatResult *ThreatAnalysisResult) bool {
	// Always create incident for critical events
	if event.Severity == ThreatLevelCritical {
		return true
	}

	// Create incident for specific high-risk events
	switch event.Type {
	case EventDataBreach, EventMalwareDetected, EventUnauthorizedAccess:
		return true
	}

	// Check if threshold is exceeded for incident creation
	if threshold, exists := sm.config.IncidentManagementConfig.IncidentThresholds[string(event.Type)]; exists {
		// This would require tracking event counts - simplified for example
		if event.RiskScore >= 0.8 {
			return true
		}
	}

	return false
}

// createAlert creates alert from security event
func (sm *SecurityMonitor) createAlert(event SecurityEvent, threatResult *ThreatAnalysisResult) *SecurityAlert {
	alert := &SecurityAlert{
		ID:          generateAlertID(),
		EventID:     event.ID,
		Type:        AlertTypeFromEvent(event.Type),
		Severity:    AlertSeverityFromThreat(event.Severity),
		Title:       generateAlertTitle(event),
		Description: generateAlertDescription(event, threatResult),
		Source:      event.Source,
		Timestamp:   time.Now(),
		Status:      AlertStatusOpen,
		RiskScore:   event.RiskScore,
		Metadata:    event.Details,
	}

	if event.Location != nil {
		alert.Location = *event.Location
	}

	return alert
}

// createIncident creates incident from security event
func (sm *SecurityMonitor) createIncident(event SecurityEvent, threatResult *ThreatAnalysisResult) *SecurityIncident {
	incident := &SecurityIncident{
		ID:          generateIncidentID(),
		EventID:     event.ID,
		Type:        IncidentTypeFromEvent(event.Type),
		Priority:    IncidentPriorityFromSeverity(event.Severity),
		Title:       generateIncidentTitle(event),
		Description: generateIncidentDescription(event, threatResult),
		Status:      IncidentStatusOpen,
		CreatedAt:   time.Now(),
		RiskScore:   event.RiskScore,
		Metadata:    event.Details,
	}

	// Auto-assign based on rules
	if assignee := sm.getAutoAssignee(event); assignee != "" {
		incident.AssignedTo = assignee
	}

	return incident
}

// isCountryAllowed checks if country is in allowed list
func (sm *SecurityMonitor) isCountryAllowed(countryCode string) bool {
	if len(sm.config.MonitoringThreatDetectionConfig.AllowedCountries) == 0 {
		return true // No restrictions
	}

	for _, allowed := range sm.config.ThreatDetectionConfig.AllowedCountries {
		if allowed == countryCode {
			return true
		}
	}
	return false
}

// getAutoAssignee gets assignee based on auto-assignment rules
func (sm *SecurityMonitor) getAutoAssignee(event SecurityEvent) string {
	for _, rule := range sm.config.IncidentManagementConfig.AutoAssignmentRules {
		if !rule.Enabled {
			continue
		}

		// Simple condition matching - would be more sophisticated in practice
		if eventType, ok := rule.Condition["event_type"]; ok {
			if eventType == string(event.Type) {
				return rule.Assignee
			}
		}
	}
	return ""
}

// complianceMonitor monitors compliance violations
func (sm *SecurityMonitor) complianceMonitor() {
	ticker := time.NewTicker(sm.config.ComplianceMonitoringConfig.ReportingFrequency)
	defer ticker.Stop()

	for range ticker.C {
		sm.runComplianceChecks()
	}
}

// runComplianceChecks runs compliance validation checks
func (sm *SecurityMonitor) runComplianceChecks() {
	for _, check := range sm.config.ComplianceMonitoringConfig.ComplianceChecks {
		if !check.Enabled {
			continue
		}

		go func(check ComplianceCheck) {
			if violation := sm.executeComplianceCheck(check); violation != nil {
				event := SecurityEvent {
					Type:      EventComplianceViolation,
					Timestamp: time.Now(),
					Source:    "compliance_monitor",
					Severity:  ThreatLevelFromComplianceSeverity(check.Severity),
					Message:   fmt.Sprintf("Compliance violation: %s", violation.Description),
					Details: map[string]interface{}{
						"framework":   check.Framework,
						"control":     check.Control,
						"violation":   violation,
						"remediation": check.Remediation,
					},
				}

				sm.ProcessSecurityEvent(event)

				// Update compliance metrics
				complianceViolationsTotal.WithLabelValues(
					check.Framework,
					check.Control,
					string(check.Severity),
				).Inc()
			}
		}(check)
	}
}

// executeComplianceCheck executes a compliance check
func (sm *SecurityMonitor) executeComplianceCheck(check ComplianceCheck) *ComplianceViolation {
	// This would contain actual compliance check logic
	// For now, return nil (no violation)
	return nil
}

// threatIntelUpdater updates threat intelligence feeds
func (sm *SecurityMonitor) threatIntelUpdater() {
	ticker := time.NewTicker(1 * time.Hour) // Update hourly
	defer ticker.Stop()

	for range ticker.C {
		sm.updateThreatIntelFeeds()
	}
}

// updateThreatIntelFeeds updates threat intelligence data
func (sm *SecurityMonitor) updateThreatIntelFeeds() {
	for _, feed := range sm.config.ThreatDetectionConfig.ThreatIntelFeeds {
		if !feed.Enabled {
			continue
		}

		go func(feed ThreatIntelFeed) {
			if time.Since(feed.LastUpdate) < feed.UpdateFreq {
				return // Too soon to update
			}

			// Update threat intel feed - implementation would fetch from external API
			sm.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
				Action:   audit.ActionUpdate,
				Resource: "threat_intelligence",
				Result:   audit.ResultSuccess,
				Details:  map[string]interface{}{"description": fmt.Sprintf("Updated threat intelligence feed: %s", feed.Name)},
			})
		}(feed)
	}
}

// GetSecurityMetrics returns current security metrics
func (sm *SecurityMonitor) GetSecurityMetrics() map[string]interface{} {
	return sm.metricsCollector.GetCurrentMetrics()
}

// GetHealthStatus returns monitoring system health
func (sm *SecurityMonitor) GetHealthStatus() map[string]interface{} {
	return map[string]interface{}{
		"event_queue_length":      len(sm.eventQueue),
		"active_subscribers":      len(sm.subscribers),
		"threat_detector_health":  sm.threatDetector.GetHealthStatus(),
		"alert_manager_health":    sm.alertManager.GetHealthStatus(),
		"incident_manager_health": sm.incidentManager.GetHealthStatus(),
	}
}

// Helper functions for mapping between types
func AlertTypeFromEvent(eventType SecurityEventType) AlertType {
	switch eventType {
	case EventAuthFailure:
		return AlertTypeAuthFailure
	case EventBruteForceAttempt:
		return AlertTypeBruteForce
	case EventRateLimitExceeded:
		return AlertTypeRateLimit
	case EventSuspiciousActivity:
		return AlertTypeSuspiciousActivity
	case EventUnauthorizedAccess:
		return AlertTypeUnauthorizedAccess
	case EventDataBreach:
		return AlertTypeDataBreach
	case EventMalwareDetected:
		return AlertTypeMalware
	case EventComplianceViolation:
		return AlertTypeCompliance
	default:
		return AlertTypeGeneral
	}
}

func AlertSeverityFromThreat(threat ThreatLevel) AlertSeverity {
	switch threat {
	case ThreatLevelCritical:
		return AlertSeverityCritical
	case ThreatLevelHigh:
		return AlertSeverityHigh
	case ThreatLevelMedium:
		return AlertSeverityMedium
	case ThreatLevelLow:
		return AlertSeverityLow
	default:
		return AlertSeverityInfo
	}
}

func IncidentTypeFromEvent(eventType SecurityEventType) IncidentType {
	switch eventType {
	case EventDataBreach:
		return IncidentTypeDataBreach
	case EventMalwareDetected:
		return IncidentTypeMalware
	case EventUnauthorizedAccess:
		return IncidentTypeUnauthorizedAccess
	case EventComplianceViolation:
		return IncidentTypeCompliance
	default:
		return IncidentTypeSecurity
	}
}

func IncidentPriorityFromSeverity(severity ThreatLevel) IncidentPriority {
	switch severity {
	case ThreatLevelCritical:
		return IncidentP1
	case ThreatLevelHigh:
		return IncidentP2
	case ThreatLevelMedium:
		return IncidentP3
	case ThreatLevelLow:
		return IncidentP4
	default:
		return IncidentP4
	}
}

func ThreatLevelFromComplianceSeverity(severity ComplianceSeverity) ThreatLevel {
	switch severity {
	case ComplianceCritical:
		return ThreatLevelCritical
	case ComplianceHigh:
		return ThreatLevelHigh
	case ComplianceMedium:
		return ThreatLevelMedium
	case ComplianceLow:
		return ThreatLevelLow
	default:
		return ThreatLevelInfo
	}
}

// Utility functions
func generateEventID() string {
	return fmt.Sprintf("evt_%d", time.Now().UnixNano())
}

func generateAlertID() string {
	return fmt.Sprintf("alt_%d", time.Now().UnixNano())
}

func generateIncidentID() string {
	return fmt.Sprintf("inc_%d", time.Now().UnixNano())
}

func generateAlertTitle(event SecurityEvent) string {
	return fmt.Sprintf("Security Alert: %s", event.Type)
}

func generateAlertDescription(event SecurityEvent, result *ThreatAnalysisResult) string {
	description := fmt.Sprintf("Security event detected: %s from %s", event.Type, event.IP)
	if result != nil && result.IsThreat {
		description += fmt.Sprintf(" (Threat confidence: %.2f)", result.Confidence)
	}
	return description
}

func generateIncidentTitle(event SecurityEvent) string {
	return fmt.Sprintf("Security Incident: %s", event.Type)
}

func generateIncidentDescription(event SecurityEvent, result *ThreatAnalysisResult) string {
	description := fmt.Sprintf("Security incident created from event: %s", event.Message)
	if result != nil && result.IsThreat {
		description += fmt.Sprintf(" Threat analysis indicates %s with %.2f confidence", result.ThreatType, result.Confidence)
	}
	return description
}

// Placeholder types that would be defined elsewhere
type SecurityAlert struct {
	ID          string
	EventID     string
	Type        AlertType
	Severity    AlertSeverity
	Title       string
	Description string
	Source      string
	Timestamp   time.Time
	Status      AlertStatus
	RiskScore   float64
	Location    GeoLocation
	Metadata    map[string]interface{}
}

type SecurityIncident struct {
	ID          string
	EventID     string
	Type        IncidentType
	Priority    IncidentPriority
	Title       string
	Description string
	Status      IncidentStatus
	CreatedAt   time.Time
	AssignedTo  string
	RiskScore   float64
	Metadata    map[string]interface{}
}

type ComplianceViolation struct {
	ID          string
	Description string
	Framework   string
	Control     string
	Severity    ComplianceSeverity
}

type ThreatAnalysisResult struct {
	IsThreat   bool
	ThreatType string
	Confidence float64
	Indicators []string
}

// Enum types (would typically be defined elsewhere)
type AlertType string
type AlertSeverity string
type AlertStatus string
type IncidentType string
type IncidentStatus string

const (
	AlertTypeAuthFailure        AlertType = "auth_failure"
	AlertTypeBruteForce         AlertType = "brute_force"
	AlertTypeRateLimit          AlertType = "rate_limit"
	AlertTypeSuspiciousActivity AlertType = "suspicious_activity"
	AlertTypeUnauthorizedAccess AlertType = "unauthorized_access"
	AlertTypeDataBreach         AlertType = "data_breach"
	AlertTypeMalware            AlertType = "malware"
	AlertTypeCompliance         AlertType = "compliance"
	AlertTypeGeneral            AlertType = "general"

	AlertSeverityCritical AlertSeverity = "critical"
	AlertSeverityHigh     AlertSeverity = "high"
	AlertSeverityMedium   AlertSeverity = "medium"
	AlertSeverityLow      AlertSeverity = "low"
	AlertSeverityInfo     AlertSeverity = "info"

	AlertStatusOpen     AlertStatus = "open"
	AlertStatusResolved AlertStatus = "resolved"

	IncidentTypeSecurity           IncidentType = "security"
	IncidentTypeDataBreach         IncidentType = "data_breach"
	IncidentTypeMalware            IncidentType = "malware"
	IncidentTypeUnauthorizedAccess IncidentType = "unauthorized_access"
	IncidentTypeCompliance         IncidentType = "compliance"

	IncidentStatusOpen       IncidentStatus = "open"
	IncidentStatusInProgress IncidentStatus = "in_progress"
	IncidentStatusResolved   IncidentStatus = "resolved"
	IncidentStatusClosed     IncidentStatus = "closed"
)
