// Package security provides unified type definitions for NovaCron security modules
// This file resolves type redeclaration conflicts by establishing domain-specific
// type hierarchies with clear semantic boundaries
package security

import (
	"time"
)

// =============================================================================
// DOMAIN-SPECIFIC TYPE PREFIXES STRATEGY
// =============================================================================
// Each security domain uses specific prefixes to avoid naming conflicts:
// - Distributed: Distributed*, Cluster*, Cross*
// - Monitoring: Monitor*, Alert*, Incident*
// - Enterprise: Enterprise*, Policy*, Compliance*
// - DatingApp: Dating*, User*, Content*
// - General: Security*, Threat*, Audit*
// =============================================================================

// =============================================================================
// 1. CORE SECURITY TYPES (Base Domain)
// =============================================================================

// SecuritySeverity provides base severity enumeration for all security domains
type SecuritySeverity string

const (
	SeverityInfo     SecuritySeverity = "info"
	SeverityLow      SecuritySeverity = "low"
	SeverityMedium   SecuritySeverity = "medium"
	SeverityHigh     SecuritySeverity = "high"
	SeverityCritical SecuritySeverity = "critical"
)

// SecurityEventBase provides common fields for all security events
type SecurityEventBase struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	Source      string                 `json:"source"`
	Severity    SecuritySeverity       `json:"severity"`
	Description string                 `json:"description"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ThreatSeverity provides threat classification (alias for SecuritySeverity for compatibility)
type ThreatSeverity = SecuritySeverity

// =============================================================================
// 2. DISTRIBUTED SECURITY DOMAIN
// =============================================================================

// DistributedSecurityEventType defines distributed security event types
type DistributedSecurityEventType string

const (
	DistributedEventAuthFailure        DistributedSecurityEventType = "auth_failure"
	DistributedEventUnauthorizedAccess DistributedSecurityEventType = "unauthorized_access"
	DistributedEventSuspiciousActivity DistributedSecurityEventType = "suspicious_activity"
	DistributedEventVulnerabilityFound DistributedSecurityEventType = "vulnerability_found"
	DistributedEventComplianceViolation DistributedSecurityEventType = "compliance_violation"
	DistributedEventSecurityBreach     DistributedSecurityEventType = "security_breach"
	DistributedEventIntrusionAttempt   DistributedSecurityEventType = "intrusion_attempt"
	DistributedEventDataExfiltration   DistributedSecurityEventType = "data_exfiltration"
	DistributedEventCrossClusterThreat DistributedSecurityEventType = "cross_cluster_threat"
)

// DistributedSecurityEvent represents security events in distributed systems
type DistributedSecurityEvent struct {
	SecurityEventBase
	Type        DistributedSecurityEventType `json:"type"`
	Target      string                       `json:"target,omitempty"`
	Data        map[string]interface{}       `json:"data"`
	ClusterID   string                       `json:"cluster_id"`
	NodeID      string                       `json:"node_id"`
	Signature   string                       `json:"signature"`
	Propagated  bool                         `json:"propagated"`
}

// DistributedMessageType defines types of distributed messages
type DistributedMessageType string

const (
	DistributedMessageClusterSync     DistributedMessageType = "cluster_sync"
	DistributedMessageFederationEvent DistributedMessageType = "federation_event"
	DistributedMessageStateUpdate     DistributedMessageType = "state_update"
	DistributedMessageMigrationEvent  DistributedMessageType = "migration_event"
	DistributedMessageHeartbeat       DistributedMessageType = "heartbeat"
	DistributedMessageSecurityEvent   DistributedMessageType = "security_event"
	DistributedMessageControlPlane    DistributedMessageType = "control_plane"
	DistributedMessageDataPlane       DistributedMessageType = "data_plane"
)

// =============================================================================
// 3. MONITORING DOMAIN
// =============================================================================

// MonitoringSecurityEventType defines monitoring-specific security event types
type MonitoringSecurityEventType string

const (
	MonitoringEventAuthFailure         MonitoringSecurityEventType = "auth_failure"
	MonitoringEventBruteForceAttempt   MonitoringSecurityEventType = "brute_force_attempt"
	MonitoringEventRateLimitExceeded   MonitoringSecurityEventType = "rate_limit_exceeded"
	MonitoringEventSuspiciousActivity  MonitoringSecurityEventType = "suspicious_activity"
	MonitoringEventUnauthorizedAccess  MonitoringSecurityEventType = "unauthorized_access"
	MonitoringEventDataBreach          MonitoringSecurityEventType = "data_breach"
	MonitoringEventMalwareDetected     MonitoringSecurityEventType = "malware_detected"
	MonitoringEventComplianceViolation MonitoringSecurityEventType = "compliance_violation"
	MonitoringEventVulnerabilityFound  MonitoringSecurityEventType = "vulnerability_found"
	MonitoringEventIncidentCreated     MonitoringSecurityEventType = "incident_created"
)

// MonitoringSecurityEvent represents security events in monitoring systems
type MonitoringSecurityEvent struct {
	SecurityEventBase
	Type        MonitoringSecurityEventType `json:"type"`
	UserID      string                      `json:"user_id"`
	IP          string                      `json:"ip"`
	UserAgent   string                      `json:"user_agent"`
	Endpoint    string                      `json:"endpoint"`
	Method      string                      `json:"method"`
	StatusCode  int                         `json:"status_code"`
	Message     string                      `json:"message"`
	Details     map[string]interface{}      `json:"details"`
	Location    *GeoLocation                `json:"location,omitempty"`
	ThreatIntel *ThreatIntelData           `json:"threat_intel,omitempty"`
	RiskScore   float64                     `json:"risk_score"`
	Remediated  bool                        `json:"remediated"`
}

// MonitoringThreatDetectionConfig defines monitoring threat detection parameters
type MonitoringThreatDetectionConfig struct {
	EnableBehaviorAnalysis bool                    `json:"enable_behavior_analysis"`
	EnableAnomalyDetection bool                    `json:"enable_anomaly_detection"`
	EnableMLDetection      bool                    `json:"enable_ml_detection"`
	BruteForceThreshold    int                     `json:"brute_force_threshold"`
	RateLimitThreshold     int                     `json:"rate_limit_threshold"`
	GeoLocationEnabled     bool                    `json:"geolocation_enabled"`
	AllowedCountries       []string                `json:"allowed_countries"`
	ThreatIntelFeeds       []ThreatIntelFeed       `json:"threat_intel_feeds"`
	CustomRules            []DetectionRule         `json:"custom_rules"`
}

// MonitoringIntegrationConfig defines monitoring system integrations
type MonitoringIntegrationConfig struct {
	JiraEnabled      bool   `json:"jira_enabled"`
	JiraURL          string `json:"jira_url"`
	JiraToken        string `json:"jira_token"`
	SlackEnabled     bool   `json:"slack_enabled"`
	SlackWebhook     string `json:"slack_webhook"`
	PagerDutyEnabled bool   `json:"pagerduty_enabled"`
	PagerDutyToken   string `json:"pagerduty_token"`
}

// =============================================================================
// 4. ENTERPRISE DOMAIN
// =============================================================================

// EnterpriseThreatDetectionConfig defines enterprise-level threat detection
type EnterpriseThreatDetectionConfig struct {
	Enabled             bool                    `json:"enabled"`
	MachineLearning     bool                    `json:"machine_learning"`
	BehavioralAnalysis  bool                    `json:"behavioral_analysis"`
	ThreatIntelligence  bool                    `json:"threat_intelligence"`
	AnomalyDetection    bool                    `json:"anomaly_detection"`
	ResponseActions     []ThreatResponseAction  `json:"response_actions"`
	Integrations        map[string]interface{}  `json:"integrations"`
}

// =============================================================================
// 5. DATING APP DOMAIN
// =============================================================================

// DatingAppMessageType defines message types in dating applications
type DatingAppMessageType string

const (
	DatingAppMessageText     DatingAppMessageType = "text"
	DatingAppMessageImage    DatingAppMessageType = "image"
	DatingAppMessageVideo    DatingAppMessageType = "video"
	DatingAppMessageAudio    DatingAppMessageType = "audio"
	DatingAppMessageFile     DatingAppMessageType = "file"
	DatingAppMessageGif      DatingAppMessageType = "gif"
	DatingAppMessageLocation DatingAppMessageType = "location"
)

// =============================================================================
// 6. GENERAL INTEGRATION CONFIG DOMAIN
// =============================================================================

// GeneralIntegrationConfig defines general system integrations
type GeneralIntegrationConfig struct {
	EnableAWSIntegration   bool       `json:"enable_aws_integration"`
	EnableGCPIntegration   bool       `json:"enable_gcp_integration"`
	EnableAzureIntegration bool       `json:"enable_azure_integration"`
	EnableSIEMIntegration  bool       `json:"enable_siem_integration"`
	SIEMConfig             SIEMConfig `json:"siem_config"`
}

// =============================================================================
// 7. SHARED SUPPORTING TYPES
// =============================================================================

// GeoLocation represents geographical location data
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
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Severity    SecuritySeverity       `json:"severity"`
	Pattern     string                 `json:"pattern"`
	Conditions  map[string]string      `json:"conditions"`
	Actions     []string               `json:"actions"`
	Enabled     bool                   `json:"enabled"`
}

// ThreatResponseAction defines actions for threat responses
type ThreatResponseAction struct {
	ThreatType string `json:"threat_type"`
	Action     string `json:"action"`     // block, alert, quarantine, investigate
	Severity   string `json:"severity"`   // low, medium, high, critical
	Automatic  bool   `json:"automatic"`
}

// SIEMConfig defines SIEM system configuration
type SIEMConfig struct {
	Provider string `json:"provider"`
	Endpoint string `json:"endpoint"`
	Index    string `json:"index"`
	Username string `json:"username"`
	Password string `json:"password"`
}

// =============================================================================
// 8. INTERFACE CONTRACTS FOR CROSS-DOMAIN INTERACTION
// =============================================================================

// SecurityEventHandler provides a unified interface for handling security events
// across different domains
type SecurityEventHandler interface {
	// HandleEvent processes a security event regardless of its domain type
	HandleEvent(event interface{}) error
	// GetEventType returns the domain-specific event type
	GetEventType() string
	// GetSeverity extracts severity from any event type
	GetSeverity(event interface{}) SecuritySeverity
	// CanHandle determines if this handler can process the given event
	CanHandle(event interface{}) bool
}

// ThreatDetector provides unified threat detection interface
type ThreatDetector interface {
	// DetectThreats analyzes events for threats regardless of domain
	DetectThreats(event interface{}) ([]ThreatIndicator, error)
	// GetDetectorType returns the detector's domain specialization
	GetDetectorType() string
	// Configure sets up detector with domain-specific config
	Configure(config interface{}) error
}

// SecurityIntegrator provides cross-domain security coordination
type SecurityIntegrator interface {
	// IntegrateEvent forwards events between security domains
	IntegrateEvent(fromDomain, toDomain string, event interface{}) error
	// GetSupportedDomains returns domains this integrator can coordinate
	GetSupportedDomains() []string
	// TransformEvent converts events between domain-specific types
	TransformEvent(event interface{}, targetDomain string) (interface{}, error)
}

// ThreatIndicator represents detected threat information
type ThreatIndicator struct {
	Type        string                 `json:"type"`
	Severity    SecuritySeverity       `json:"severity"`
	Confidence  float64               `json:"confidence"`
	Description string                 `json:"description"`
	Indicators  map[string]interface{} `json:"indicators"`
	Timestamp   time.Time             `json:"timestamp"`
}

// =============================================================================
// 9. TYPE HIERARCHY AND RELATIONSHIPS
// =============================================================================

// SecurityEventTypeRegistry maintains mappings between domain-specific event types
type SecurityEventTypeRegistry struct {
	// distributedToMonitoring maps distributed events to monitoring events
	distributedToMonitoring map[DistributedSecurityEventType]MonitoringSecurityEventType
	// monitoringToDistributed maps monitoring events to distributed events
	monitoringToDistributed map[MonitoringSecurityEventType]DistributedSecurityEventType
	// severity maps event types to their default severity levels
	severity map[string]SecuritySeverity
}

// NewSecurityEventTypeRegistry creates a registry with default mappings
func NewSecurityEventTypeRegistry() *SecurityEventTypeRegistry {
	registry := &SecurityEventTypeRegistry{
		distributedToMonitoring: make(map[DistributedSecurityEventType]MonitoringSecurityEventType),
		monitoringToDistributed: make(map[MonitoringSecurityEventType]DistributedSecurityEventType),
		severity: make(map[string]SecuritySeverity),
	}

	// Initialize default mappings
	registry.initializeDefaultMappings()

	return registry
}

// initializeDefaultMappings sets up default cross-domain type mappings
func (r *SecurityEventTypeRegistry) initializeDefaultMappings() {
	// Map common event types between domains
	r.distributedToMonitoring[DistributedEventAuthFailure] = MonitoringEventAuthFailure
	r.distributedToMonitoring[DistributedEventUnauthorizedAccess] = MonitoringEventUnauthorizedAccess
	r.distributedToMonitoring[DistributedEventSuspiciousActivity] = MonitoringEventSuspiciousActivity
	r.distributedToMonitoring[DistributedEventComplianceViolation] = MonitoringEventComplianceViolation
	r.distributedToMonitoring[DistributedEventVulnerabilityFound] = MonitoringEventVulnerabilityFound
	r.distributedToMonitoring[DistributedEventSecurityBreach] = MonitoringEventDataBreach

	// Reverse mappings
	for dist, mon := range r.distributedToMonitoring {
		r.monitoringToDistributed[mon] = dist
	}

	// Default severity levels
	r.severity[string(DistributedEventAuthFailure)] = SeverityMedium
	r.severity[string(DistributedEventUnauthorizedAccess)] = SeverityHigh
	r.severity[string(DistributedEventSuspiciousActivity)] = SeverityMedium
	r.severity[string(DistributedEventSecurityBreach)] = SeverityCritical
	r.severity[string(DistributedEventDataExfiltration)] = SeverityCritical
	r.severity[string(DistributedEventComplianceViolation)] = SeverityHigh
	r.severity[string(DistributedEventVulnerabilityFound)] = SeverityMedium
}

// MapDistributedToMonitoring converts distributed event type to monitoring event type
func (r *SecurityEventTypeRegistry) MapDistributedToMonitoring(eventType DistributedSecurityEventType) (MonitoringSecurityEventType, bool) {
	mapped, exists := r.distributedToMonitoring[eventType]
	return mapped, exists
}

// MapMonitoringToDistributed converts monitoring event type to distributed event type
func (r *SecurityEventTypeRegistry) MapMonitoringToDistributed(eventType MonitoringSecurityEventType) (DistributedSecurityEventType, bool) {
	mapped, exists := r.monitoringToDistributed[eventType]
	return mapped, exists
}

// GetDefaultSeverity returns the default severity for an event type
func (r *SecurityEventTypeRegistry) GetDefaultSeverity(eventType string) SecuritySeverity {
	if severity, exists := r.severity[eventType]; exists {
		return severity
	}
	return SeverityMedium // Default fallback
}

// =============================================================================
// 10. ARCHITECTURAL STANDARDS FOR FUTURE TYPE ADDITIONS
// =============================================================================

// DomainTypeNamespace defines naming conventions for new security domains
type DomainTypeNamespace struct {
	// Domain is the security domain identifier (e.g., "blockchain", "iot", "cloud")
	Domain string
	// Prefix is the type name prefix for this domain (e.g., "Blockchain", "IoT", "Cloud")
	Prefix string
	// EventTypesSuffix defines the suffix for event types (default: "SecurityEventType")
	EventTypesSuffix string
	// EventSuffix defines the suffix for event structs (default: "SecurityEvent")
	EventSuffix string
}

// SecurityDomainRegistry maintains registered security domains and their namespaces
var SecurityDomainRegistry = map[string]DomainTypeNamespace{
	"distributed": {
		Domain:           "distributed",
		Prefix:          "Distributed",
		EventTypesSuffix: "SecurityEventType",
		EventSuffix:     "SecurityEvent",
	},
	"monitoring": {
		Domain:           "monitoring",
		Prefix:          "Monitoring",
		EventTypesSuffix: "SecurityEventType",
		EventSuffix:     "SecurityEvent",
	},
	"enterprise": {
		Domain:           "enterprise",
		Prefix:          "Enterprise",
		EventTypesSuffix: "ThreatDetectionConfig",
		EventSuffix:     "SecurityConfig",
	},
	"dating_app": {
		Domain:           "dating_app",
		Prefix:          "DatingApp",
		EventTypesSuffix: "MessageType",
		EventSuffix:     "SecurityManager",
	},
	"general": {
		Domain:           "general",
		Prefix:          "General",
		EventTypesSuffix: "Config",
		EventSuffix:     "Config",
	},
}

// RegisterSecurityDomain adds a new security domain to the registry
func RegisterSecurityDomain(domain string, namespace DomainTypeNamespace) {
	SecurityDomainRegistry[domain] = namespace
}

// GetSecurityDomain retrieves a security domain's namespace
func GetSecurityDomain(domain string) (DomainTypeNamespace, bool) {
	namespace, exists := SecurityDomainRegistry[domain]
	return namespace, exists
}

// GenerateTypeName creates a properly namespaced type name for a domain
func GenerateTypeName(domain, typeName, suffix string) string {
	if namespace, exists := GetSecurityDomain(domain); exists {
		return namespace.Prefix + typeName + suffix
	}
	return typeName + suffix // Fallback
}