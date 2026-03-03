// Package security provides unified event handling for all security domains
package security

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/audit"
)

// UnifiedSecurityEventHandler handles security events from all domains
type UnifiedSecurityEventHandler struct {
	distributedHandler SecurityEventHandler
	monitoringHandler  SecurityEventHandler
	auditLogger        audit.AuditLogger
	eventConverter     *SecurityEventConverter
	registry          *SecurityEventTypeRegistry
}

// NewUnifiedSecurityEventHandler creates a unified handler for all security domains
func NewUnifiedSecurityEventHandler(auditLogger audit.AuditLogger) *UnifiedSecurityEventHandler {
	registry := NewSecurityEventTypeRegistry()

	return &UnifiedSecurityEventHandler{
		distributedHandler: NewDistributedSecurityEventHandler(),
		monitoringHandler:  NewMonitoringSecurityEventHandler(),
		auditLogger:        auditLogger,
		eventConverter:     NewSecurityEventConverter(registry),
		registry:          registry,
	}
}

// HandleEvent processes security events from any domain
func (h *UnifiedSecurityEventHandler) HandleEvent(event interface{}) error {
	ctx := context.Background()

	switch e := event.(type) {
	case DistributedSecurityEvent:
		return h.handleDistributedEvent(ctx, e)
	case MonitoringSecurityEvent:
		return h.handleMonitoringEvent(ctx, e)
	case *DistributedSecurityEvent:
		return h.handleDistributedEvent(ctx, *e)
	case *MonitoringSecurityEvent:
		return h.handleMonitoringEvent(ctx, *e)
	default:
		return fmt.Errorf("unsupported event type: %T", event)
	}
}

// GetEventType returns the unified event type identifier
func (h *UnifiedSecurityEventHandler) GetEventType() string {
	return "unified_security_handler"
}

// GetSeverity extracts severity from any supported event type
func (h *UnifiedSecurityEventHandler) GetSeverity(event interface{}) SecuritySeverity {
	switch e := event.(type) {
	case DistributedSecurityEvent:
		return e.Severity
	case MonitoringSecurityEvent:
		return e.Severity
	case *DistributedSecurityEvent:
		return e.Severity
	case *MonitoringSecurityEvent:
		return e.Severity
	default:
		return SeverityMedium // Default fallback
	}
}

// CanHandle determines if this handler can process the given event
func (h *UnifiedSecurityEventHandler) CanHandle(event interface{}) bool {
	switch event.(type) {
	case DistributedSecurityEvent, *DistributedSecurityEvent:
		return true
	case MonitoringSecurityEvent, *MonitoringSecurityEvent:
		return true
	default:
		return false
	}
}

// handleDistributedEvent processes distributed security events
func (h *UnifiedSecurityEventHandler) handleDistributedEvent(ctx context.Context, event DistributedSecurityEvent) error {
	// Log the distributed event
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   event.NodeID,
		Action:   audit.ActionRead,
		Resource: "distributed_security_event",
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"event_id":    event.ID,
			"event_type":  string(event.Type),
			"cluster_id":  event.ClusterID,
			"severity":    string(event.Severity),
			"description": event.Description,
		},
	})

	// Handle the event with the distributed handler
	if err := h.distributedHandler.HandleEvent(event); err != nil {
		log.Printf("Failed to handle distributed event %s: %v", event.ID, err)
		return err
	}

	// Convert to monitoring event for cross-domain analysis
	monitoringEvent := h.eventConverter.ConvertToMonitoring(event)

	// Process with monitoring handler for additional analysis
	if err := h.monitoringHandler.HandleEvent(monitoringEvent); err != nil {
		log.Printf("Failed to handle converted monitoring event %s: %v", monitoringEvent.ID, err)
		// Don't return error - this is supplementary processing
	}

	return nil
}

// handleMonitoringEvent processes monitoring security events
func (h *UnifiedSecurityEventHandler) handleMonitoringEvent(ctx context.Context, event MonitoringSecurityEvent) error {
	// Log the monitoring event
	h.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   event.UserID,
		Action:   audit.ActionRead,
		Resource: "monitoring_security_event",
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"event_id":    event.ID,
			"event_type":  string(event.Type),
			"user_id":     event.UserID,
			"ip":          event.IP,
			"severity":    string(event.Severity),
			"risk_score":  event.RiskScore,
			"description": event.Description,
		},
	})

	// Handle the event with the monitoring handler
	if err := h.monitoringHandler.HandleEvent(event); err != nil {
		log.Printf("Failed to handle monitoring event %s: %v", event.ID, err)
		return err
	}

	// If this is a critical security event, convert to distributed event for cluster-wide awareness
	if event.Severity == SeverityCritical {
		distributedEvent := h.eventConverter.ConvertToDistributed(event)

		// Process with distributed handler for cluster-wide response
		if err := h.distributedHandler.HandleEvent(distributedEvent); err != nil {
			log.Printf("Failed to handle converted distributed event %s: %v", distributedEvent.ID, err)
			// Don't return error - this is supplementary processing
		}
	}

	return nil
}

// DistributedSecurityEventHandler handles distributed security events
type DistributedSecurityEventHandler struct {
	threatAnalyzer    ThreatDetector
	responseOrchestrator ResponseOrchestrator
}

// NewDistributedSecurityEventHandler creates a handler for distributed security events
func NewDistributedSecurityEventHandler() *DistributedSecurityEventHandler {
	return &DistributedSecurityEventHandler{
		threatAnalyzer:       NewDistributedThreatDetector(),
		responseOrchestrator: NewDistributedResponseOrchestrator(),
	}
}

func (h *DistributedSecurityEventHandler) HandleEvent(event interface{}) error {
	distEvent, ok := event.(DistributedSecurityEvent)
	if !ok {
		if ptr, ok := event.(*DistributedSecurityEvent); ok {
			distEvent = *ptr
		} else {
			return fmt.Errorf("expected DistributedSecurityEvent, got %T", event)
		}
	}

	// Analyze for threats
	threats, err := h.threatAnalyzer.DetectThreats(distEvent)
	if err != nil {
		return fmt.Errorf("threat analysis failed: %w", err)
	}

	// Execute response actions based on threats
	for _, threat := range threats {
		if err := h.responseOrchestrator.RespondToThreat(threat, distEvent); err != nil {
			log.Printf("Failed to respond to threat %s: %v", threat.Type, err)
		}
	}

	return nil
}

func (h *DistributedSecurityEventHandler) GetEventType() string {
	return "distributed_security_handler"
}

func (h *DistributedSecurityEventHandler) GetSeverity(event interface{}) SecuritySeverity {
	if distEvent, ok := event.(DistributedSecurityEvent); ok {
		return distEvent.Severity
	}
	return SeverityMedium
}

func (h *DistributedSecurityEventHandler) CanHandle(event interface{}) bool {
	switch event.(type) {
	case DistributedSecurityEvent, *DistributedSecurityEvent:
		return true
	default:
		return false
	}
}

// MonitoringSecurityEventHandler handles monitoring security events
type MonitoringSecurityEventHandler struct {
	threatAnalyzer    ThreatDetector
	alertManager     AlertManager
	incidentManager  IncidentManager
}

// NewMonitoringSecurityEventHandler creates a handler for monitoring security events
func NewMonitoringSecurityEventHandler() *MonitoringSecurityEventHandler {
	return &MonitoringSecurityEventHandler{
		threatAnalyzer:  NewMonitoringThreatDetector(),
		alertManager:    NewAlertManager(),
		incidentManager: NewIncidentManager(),
	}
}

func (h *MonitoringSecurityEventHandler) HandleEvent(event interface{}) error {
	monEvent, ok := event.(MonitoringSecurityEvent)
	if !ok {
		if ptr, ok := event.(*MonitoringSecurityEvent); ok {
			monEvent = *ptr
		} else {
			return fmt.Errorf("expected MonitoringSecurityEvent, got %T", event)
		}
	}

	// Analyze for threats
	threats, err := h.threatAnalyzer.DetectThreats(monEvent)
	if err != nil {
		return fmt.Errorf("threat analysis failed: %w", err)
	}

	// Create alerts for significant threats
	for _, threat := range threats {
		if threat.Severity >= SeverityMedium {
			alert := h.createAlertFromThreat(threat, monEvent)
			if err := h.alertManager.TriggerAlert(alert); err != nil {
				log.Printf("Failed to trigger alert for threat %s: %v", threat.Type, err)
			}
		}

		// Create incidents for critical threats
		if threat.Severity == SeverityCritical {
			incident := h.createIncidentFromThreat(threat, monEvent)
			if err := h.incidentManager.CreateIncident(incident); err != nil {
				log.Printf("Failed to create incident for threat %s: %v", threat.Type, err)
			}
		}
	}

	return nil
}

func (h *MonitoringSecurityEventHandler) GetEventType() string {
	return "monitoring_security_handler"
}

func (h *MonitoringSecurityEventHandler) GetSeverity(event interface{}) SecuritySeverity {
	if monEvent, ok := event.(MonitoringSecurityEvent); ok {
		return monEvent.Severity
	}
	return SeverityMedium
}

func (h *MonitoringSecurityEventHandler) CanHandle(event interface{}) bool {
	switch event.(type) {
	case MonitoringSecurityEvent, *MonitoringSecurityEvent:
		return true
	default:
		return false
	}
}

func (h *MonitoringSecurityEventHandler) createAlertFromThreat(threat ThreatIndicator, event MonitoringSecurityEvent) *SecurityAlert {
	return &SecurityAlert{
		ID:          fmt.Sprintf("alert_%s_%d", threat.Type, time.Now().Unix()),
		EventID:     event.ID,
		Type:        mapThreatToAlertType(threat.Type),
		Severity:    mapSeverityToAlertSeverity(threat.Severity),
		Title:       fmt.Sprintf("Security Threat Detected: %s", threat.Type),
		Description: fmt.Sprintf("Threat: %s (Confidence: %.2f) - %s", threat.Type, threat.Confidence, threat.Description),
		Source:      event.Source,
		Timestamp:   time.Now(),
		Status:      "open",
		RiskScore:   event.RiskScore,
		Metadata:    threat.Indicators,
	}
}

func (h *MonitoringSecurityEventHandler) createIncidentFromThreat(threat ThreatIndicator, event MonitoringSecurityEvent) *SecurityIncident {
	return &SecurityIncident{
		ID:          fmt.Sprintf("incident_%s_%d", threat.Type, time.Now().Unix()),
		EventID:     event.ID,
		Type:        mapThreatToIncidentType(threat.Type),
		Priority:    mapSeverityToIncidentPriority(threat.Severity),
		Title:       fmt.Sprintf("Critical Security Incident: %s", threat.Type),
		Description: fmt.Sprintf("Critical threat detected: %s (Confidence: %.2f)", threat.Type, threat.Confidence),
		Status:      "open",
		CreatedAt:   time.Now(),
		RiskScore:   event.RiskScore,
		Metadata:    threat.Indicators,
	}
}

// SecurityEventConverter provides conversion between domain-specific event types
type SecurityEventConverter struct {
	registry *SecurityEventTypeRegistry
}

func NewSecurityEventConverter(registry *SecurityEventTypeRegistry) *SecurityEventConverter {
	return &SecurityEventConverter{
		registry: registry,
	}
}

func (c *SecurityEventConverter) ConvertToMonitoring(event DistributedSecurityEvent) MonitoringSecurityEvent {
	monitoringType, exists := c.registry.MapDistributedToMonitoring(event.Type)
	if !exists {
		monitoringType = MonitoringEventSuspiciousActivity // Default fallback
	}

	return MonitoringSecurityEvent{
		SecurityEventBase: event.SecurityEventBase,
		Type:             monitoringType,
		UserID:           extractUserID(event.Data),
		IP:               extractIP(event.Data),
		UserAgent:        extractUserAgent(event.Data),
		Endpoint:         extractEndpoint(event.Data),
		Method:           extractMethod(event.Data),
		StatusCode:       extractStatusCode(event.Data),
		Message:          event.Description,
		Details:          event.Data,
		RiskScore:        calculateRiskScore(event),
		Remediated:       false,
	}
}

func (c *SecurityEventConverter) ConvertToDistributed(event MonitoringSecurityEvent) DistributedSecurityEvent {
	distributedType, exists := c.registry.MapMonitoringToDistributed(event.Type)
	if !exists {
		distributedType = DistributedEventSuspiciousActivity // Default fallback
	}

	return DistributedSecurityEvent{
		SecurityEventBase: event.SecurityEventBase,
		Type:             distributedType,
		Data:             combineEventData(event.Details, event.UserID, event.IP, event.UserAgent),
		ClusterID:        extractClusterID(event.Metadata),
		NodeID:           extractNodeID(event.Metadata),
		Signature:        generateEventSignature(event),
		Propagated:       false,
	}
}

// Helper functions for data extraction and conversion
func extractUserID(data map[string]interface{}) string {
	if userID, ok := data["user_id"].(string); ok {
		return userID
	}
	return ""
}

func extractIP(data map[string]interface{}) string {
	if ip, ok := data["ip"].(string); ok {
		return ip
	}
	if clientIP, ok := data["client_ip"].(string); ok {
		return clientIP
	}
	return ""
}

func extractUserAgent(data map[string]interface{}) string {
	if ua, ok := data["user_agent"].(string); ok {
		return ua
	}
	return ""
}

func extractEndpoint(data map[string]interface{}) string {
	if endpoint, ok := data["endpoint"].(string); ok {
		return endpoint
	}
	if path, ok := data["path"].(string); ok {
		return path
	}
	return ""
}

func extractMethod(data map[string]interface{}) string {
	if method, ok := data["method"].(string); ok {
		return method
	}
	return ""
}

func extractStatusCode(data map[string]interface{}) int {
	if code, ok := data["status_code"].(int); ok {
		return code
	}
	if code, ok := data["status_code"].(float64); ok {
		return int(code)
	}
	return 0
}

func extractClusterID(metadata map[string]interface{}) string {
	if clusterID, ok := metadata["cluster_id"].(string); ok {
		return clusterID
	}
	return "default-cluster"
}

func extractNodeID(metadata map[string]interface{}) string {
	if nodeID, ok := metadata["node_id"].(string); ok {
		return nodeID
	}
	return "unknown-node"
}

func calculateRiskScore(event DistributedSecurityEvent) float64 {
	// Basic risk score calculation based on event type and severity
	baseScore := 0.0

	switch event.Type {
	case DistributedEventSecurityBreach, DistributedEventDataExfiltration:
		baseScore = 0.9
	case DistributedEventUnauthorizedAccess, DistributedEventIntrusionAttempt:
		baseScore = 0.7
	case DistributedEventSuspiciousActivity, DistributedEventComplianceViolation:
		baseScore = 0.5
	case DistributedEventAuthFailure:
		baseScore = 0.3
	default:
		baseScore = 0.4
	}

	// Adjust based on severity
	switch event.Severity {
	case SeverityCritical:
		baseScore *= 1.5
	case SeverityHigh:
		baseScore *= 1.2
	case SeverityMedium:
		baseScore *= 1.0
	case SeverityLow:
		baseScore *= 0.8
	case SeverityInfo:
		baseScore *= 0.5
	}

	// Cap at 1.0
	if baseScore > 1.0 {
		baseScore = 1.0
	}

	return baseScore
}

func combineEventData(details map[string]interface{}, userID, ip, userAgent string) map[string]interface{} {
	combined := make(map[string]interface{})

	// Copy existing details
	for k, v := range details {
		combined[k] = v
	}

	// Add additional context
	if userID != "" {
		combined["user_id"] = userID
	}
	if ip != "" {
		combined["ip"] = ip
	}
	if userAgent != "" {
		combined["user_agent"] = userAgent
	}

	return combined
}

func generateEventSignature(event MonitoringSecurityEvent) string {
	// Generate a simple signature based on event data
	return fmt.Sprintf("mon_%s_%s_%d", event.ID, string(event.Type), event.Timestamp.Unix())
}

// Type mapping functions
func mapThreatToAlertType(threatType string) string {
	switch threatType {
	case "auth_failure":
		return "auth_failure"
	case "brute_force":
		return "brute_force"
	case "malware":
		return "malware"
	case "data_breach":
		return "data_breach"
	default:
		return "general"
	}
}

func mapSeverityToAlertSeverity(severity SecuritySeverity) string {
	return string(severity)
}

func mapThreatToIncidentType(threatType string) string {
	switch threatType {
	case "data_breach":
		return "data_breach"
	case "malware":
		return "malware"
	case "unauthorized_access":
		return "unauthorized_access"
	default:
		return "security"
	}
}

func mapSeverityToIncidentPriority(severity SecuritySeverity) string {
	switch severity {
	case SeverityCritical:
		return "p1"
	case SeverityHigh:
		return "p2"
	case SeverityMedium:
		return "p3"
	case SeverityLow:
		return "p4"
	default:
		return "p4"
	}
}

// Interface implementations for threat detection and response
type ResponseOrchestrator interface {
	RespondToThreat(threat ThreatIndicator, event interface{}) error
}

type AlertManager interface {
	TriggerAlert(alert *SecurityAlert) error
}

type IncidentManager interface {
	CreateIncident(incident *SecurityIncident) error
}

// Placeholder implementations
func NewDistributedThreatDetector() ThreatDetector {
	return &MockThreatDetector{detectorType: "distributed"}
}

func NewMonitoringThreatDetector() ThreatDetector {
	return &MockThreatDetector{detectorType: "monitoring"}
}

func NewDistributedResponseOrchestrator() ResponseOrchestrator {
	return &MockResponseOrchestrator{}
}

func NewAlertManager() AlertManager {
	return &MockAlertManager{}
}

func NewIncidentManager() IncidentManager {
	return &MockIncidentManager{}
}

// Mock implementations for interfaces
type MockThreatDetector struct {
	detectorType string
}

func (m *MockThreatDetector) DetectThreats(event interface{}) ([]ThreatIndicator, error) {
	// Mock threat detection - in real implementation this would analyze the event
	var severity SecuritySeverity
	var threatType string

	switch e := event.(type) {
	case DistributedSecurityEvent:
		severity = e.Severity
		threatType = string(e.Type)
	case MonitoringSecurityEvent:
		severity = e.Severity
		threatType = string(e.Type)
	default:
		return nil, fmt.Errorf("unsupported event type for threat detection")
	}

	// Return mock threat if severity is medium or higher
	if severity >= SeverityMedium {
		return []ThreatIndicator{
			{
				Type:        threatType,
				Severity:    severity,
				Confidence:  0.8,
				Description: fmt.Sprintf("Detected %s threat", threatType),
				Indicators:  map[string]interface{}{"source": m.detectorType},
				Timestamp:   time.Now(),
			},
		}, nil
	}

	return []ThreatIndicator{}, nil
}

func (m *MockThreatDetector) GetDetectorType() string {
	return m.detectorType
}

func (m *MockThreatDetector) Configure(config interface{}) error {
	return nil
}

type MockResponseOrchestrator struct{}

func (m *MockResponseOrchestrator) RespondToThreat(threat ThreatIndicator, event interface{}) error {
	log.Printf("Responding to %s threat with severity %s", threat.Type, threat.Severity)
	return nil
}

type MockAlertManager struct{}

func (m *MockAlertManager) TriggerAlert(alert *SecurityAlert) error {
	log.Printf("Alert triggered: %s - %s", alert.ID, alert.Title)
	return nil
}

type MockIncidentManager struct{}

func (m *MockIncidentManager) CreateIncident(incident *SecurityIncident) error {
	log.Printf("Incident created: %s - %s", incident.ID, incident.Title)
	return nil
}

// Additional types for mock implementations
type SecurityAlert struct {
	ID          string                 `json:"id"`
	EventID     string                 `json:"event_id"`
	Type        string                 `json:"type"`
	Severity    string                 `json:"severity"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Source      string                 `json:"source"`
	Timestamp   time.Time              `json:"timestamp"`
	Status      string                 `json:"status"`
	RiskScore   float64                `json:"risk_score"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type SecurityIncident struct {
	ID          string                 `json:"id"`
	EventID     string                 `json:"event_id"`
	Type        string                 `json:"type"`
	Priority    string                 `json:"priority"`
	Title       string                 `json:"title"`
	Description string                 `json:"description"`
	Status      string                 `json:"status"`
	CreatedAt   time.Time              `json:"created_at"`
	RiskScore   float64                `json:"risk_score"`
	Metadata    map[string]interface{} `json:"metadata"`
}