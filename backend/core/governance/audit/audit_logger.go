package audit

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// AuditEventType represents the type of audit event
type AuditEventType string

const (
	EventUserLogin        AuditEventType = "user.login"
	EventUserLogout       AuditEventType = "user.logout"
	EventUserCreate       AuditEventType = "user.create"
	EventUserDelete       AuditEventType = "user.delete"
	EventUserModify       AuditEventType = "user.modify"
	EventAccessGranted    AuditEventType = "access.granted"
	EventAccessDenied     AuditEventType = "access.denied"
	EventConfigChange     AuditEventType = "config.change"
	EventVMCreate         AuditEventType = "vm.create"
	EventVMDelete         AuditEventType = "vm.delete"
	EventVMStart          AuditEventType = "vm.start"
	EventVMStop           AuditEventType = "vm.stop"
	EventNetworkChange    AuditEventType = "network.change"
	EventPolicyChange     AuditEventType = "policy.change"
	EventSecretAccess     AuditEventType = "secret.access"
	EventComplianceViolation AuditEventType = "compliance.violation"
	EventAPICall          AuditEventType = "api.call"
)

// AuditSeverity represents the severity of an audit event
type AuditSeverity string

const (
	SeverityInfo     AuditSeverity = "info"
	SeverityWarning  AuditSeverity = "warning"
	SeverityError    AuditSeverity = "error"
	SeverityCritical AuditSeverity = "critical"
)

// AuditEvent represents a single audit log entry
type AuditEvent struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	EventType   AuditEventType         `json:"event_type"`
	Severity    AuditSeverity          `json:"severity"`
	Actor       Actor                  `json:"actor"`
	Target      Target                 `json:"target"`
	Action      string                 `json:"action"`
	Result      string                 `json:"result"` // success, failure
	Details     map[string]interface{} `json:"details"`
	IPAddress   string                 `json:"ip_address"`
	UserAgent   string                 `json:"user_agent"`
	SessionID   string                 `json:"session_id"`
	TenantID    string                 `json:"tenant_id"`
	Hash        string                 `json:"hash"`         // For tamper detection
	PreviousHash string                `json:"previous_hash"` // Chain to previous event
	Metadata    map[string]string      `json:"metadata"`
}

// Actor represents the entity performing an action
type Actor struct {
	Type       string `json:"type"` // user, service, system
	ID         string `json:"id"`
	Name       string `json:"name"`
	Roles      []string `json:"roles"`
	Permissions []string `json:"permissions"`
}

// Target represents the entity being acted upon
type Target struct {
	Type       string `json:"type"` // vm, user, config, policy
	ID         string `json:"id"`
	Name       string `json:"name"`
	Attributes map[string]string `json:"attributes"`
}

// AuditLogger implements immutable audit logging with tamper protection
type AuditLogger struct {
	mu                sync.RWMutex
	events            []*AuditEvent
	lastHash          string
	retentionPeriod   time.Duration
	immutableStorage  bool
	tamperProtection  bool
	searchIndex       *SearchIndex
	forensicsEngine   *ForensicsEngine
	backupEnabled     bool
	backupDestination string
	metrics           *AuditMetrics
}

// SearchIndex provides searchable audit log index
type SearchIndex struct {
	mu            sync.RWMutex
	byEventType   map[AuditEventType][]*AuditEvent
	byActor       map[string][]*AuditEvent
	byTarget      map[string][]*AuditEvent
	byTenant      map[string][]*AuditEvent
	byTimeRange   map[string][]*AuditEvent // indexed by day
}

// ForensicsEngine provides forensic log analysis capabilities
type ForensicsEngine struct {
	mu             sync.RWMutex
	analysisResults map[string]*ForensicAnalysis
}

// ForensicAnalysis represents a forensic analysis result
type ForensicAnalysis struct {
	ID              string    `json:"id"`
	StartTime       time.Time `json:"start_time"`
	EndTime         time.Time `json:"end_time"`
	Query           string    `json:"query"`
	EventsAnalyzed  int       `json:"events_analyzed"`
	Anomalies       []Anomaly `json:"anomalies"`
	Timeline        []TimelineEvent `json:"timeline"`
	Recommendations []string  `json:"recommendations"`
}

// Anomaly represents a detected anomaly
type Anomaly struct {
	Type        string    `json:"type"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	Timestamp   time.Time `json:"timestamp"`
	Events      []string  `json:"events"` // Event IDs
}

// TimelineEvent represents an event in the forensic timeline
type TimelineEvent struct {
	Timestamp   time.Time `json:"timestamp"`
	EventType   string    `json:"event_type"`
	Description string    `json:"description"`
	Actor       string    `json:"actor"`
	Target      string    `json:"target"`
}

// AuditMetrics tracks audit logging metrics
type AuditMetrics struct {
	mu                    sync.RWMutex
	TotalEvents           int64
	EventsByType          map[AuditEventType]int64
	EventsBySeverity      map[AuditSeverity]int64
	AccessDeniedCount     int64
	ComplianceViolations  int64
	TamperAttempts        int64
	SearchQueries         int64
	ForensicAnalysisCount int64
}

// NewAuditLogger creates a new audit logger
func NewAuditLogger(retentionPeriod time.Duration, immutableStorage, tamperProtection bool) *AuditLogger {
	return &AuditLogger{
		events:           make([]*AuditEvent, 0),
		retentionPeriod:  retentionPeriod,
		immutableStorage: immutableStorage,
		tamperProtection: tamperProtection,
		searchIndex:      newSearchIndex(),
		forensicsEngine:  newForensicsEngine(),
		metrics: &AuditMetrics{
			EventsByType:     make(map[AuditEventType]int64),
			EventsBySeverity: make(map[AuditSeverity]int64),
		},
	}
}

func newSearchIndex() *SearchIndex {
	return &SearchIndex{
		byEventType: make(map[AuditEventType][]*AuditEvent),
		byActor:     make(map[string][]*AuditEvent),
		byTarget:    make(map[string][]*AuditEvent),
		byTenant:    make(map[string][]*AuditEvent),
		byTimeRange: make(map[string][]*AuditEvent),
	}
}

func newForensicsEngine() *ForensicsEngine {
	return &ForensicsEngine{
		analysisResults: make(map[string]*ForensicAnalysis),
	}
}

// LogEvent logs an audit event with tamper protection
func (al *AuditLogger) LogEvent(ctx context.Context, event *AuditEvent) error {
	al.mu.Lock()
	defer al.mu.Unlock()

	// Generate unique ID
	event.ID = fmt.Sprintf("audit-%d-%s", time.Now().UnixNano(), generateRandomID())
	event.Timestamp = time.Now()

	// Add tamper protection (blockchain-style chaining)
	if al.tamperProtection {
		event.PreviousHash = al.lastHash
		event.Hash = al.calculateEventHash(event)
		al.lastHash = event.Hash
	}

	// Append to immutable log (in production, this would be write-only storage)
	al.events = append(al.events, event)

	// Update search index
	al.updateSearchIndex(event)

	// Update metrics
	al.updateMetrics(event)

	// Check for anomalies
	go al.detectAnomalies(event)

	return nil
}

// calculateEventHash calculates SHA256 hash of event for tamper detection
func (al *AuditLogger) calculateEventHash(event *AuditEvent) string {
	data, _ := json.Marshal(struct {
		Timestamp    time.Time              `json:"timestamp"`
		EventType    AuditEventType         `json:"event_type"`
		Actor        Actor                  `json:"actor"`
		Target       Target                 `json:"target"`
		Action       string                 `json:"action"`
		Result       string                 `json:"result"`
		Details      map[string]interface{} `json:"details"`
		PreviousHash string                 `json:"previous_hash"`
	}{
		Timestamp:    event.Timestamp,
		EventType:    event.EventType,
		Actor:        event.Actor,
		Target:       event.Target,
		Action:       event.Action,
		Result:       event.Result,
		Details:      event.Details,
		PreviousHash: event.PreviousHash,
	})

	hash := sha256.Sum256(data)
	return hex.EncodeToString(hash[:])
}

// updateSearchIndex updates the searchable index
func (al *AuditLogger) updateSearchIndex(event *AuditEvent) {
	al.searchIndex.mu.Lock()
	defer al.searchIndex.mu.Unlock()

	// Index by event type
	al.searchIndex.byEventType[event.EventType] = append(al.searchIndex.byEventType[event.EventType], event)

	// Index by actor
	actorKey := fmt.Sprintf("%s:%s", event.Actor.Type, event.Actor.ID)
	al.searchIndex.byActor[actorKey] = append(al.searchIndex.byActor[actorKey], event)

	// Index by target
	targetKey := fmt.Sprintf("%s:%s", event.Target.Type, event.Target.ID)
	al.searchIndex.byTarget[targetKey] = append(al.searchIndex.byTarget[targetKey], event)

	// Index by tenant
	if event.TenantID != "" {
		al.searchIndex.byTenant[event.TenantID] = append(al.searchIndex.byTenant[event.TenantID], event)
	}

	// Index by time range (day)
	dayKey := event.Timestamp.Format("2006-01-02")
	al.searchIndex.byTimeRange[dayKey] = append(al.searchIndex.byTimeRange[dayKey], event)
}

// updateMetrics updates audit metrics
func (al *AuditLogger) updateMetrics(event *AuditEvent) {
	al.metrics.mu.Lock()
	defer al.metrics.mu.Unlock()

	al.metrics.TotalEvents++
	al.metrics.EventsByType[event.EventType]++
	al.metrics.EventsBySeverity[event.Severity]++

	if event.EventType == EventAccessDenied {
		al.metrics.AccessDeniedCount++
	}

	if event.EventType == EventComplianceViolation {
		al.metrics.ComplianceViolations++
	}
}

// detectAnomalies detects anomalies in audit events
func (al *AuditLogger) detectAnomalies(event *AuditEvent) {
	// Anomaly detection patterns
	anomalies := make([]Anomaly, 0)

	// Pattern 1: Multiple failed login attempts
	if event.EventType == EventUserLogin && event.Result == "failure" {
		recentFailures := al.countRecentFailedLogins(event.Actor.ID, 5*time.Minute)
		if recentFailures >= 5 {
			anomalies = append(anomalies, Anomaly{
				Type:        "brute-force",
				Severity:    "critical",
				Description: fmt.Sprintf("Multiple failed login attempts for user %s", event.Actor.ID),
				Timestamp:   event.Timestamp,
			})
		}
	}

	// Pattern 2: Access denied to sensitive resources
	if event.EventType == EventAccessDenied && event.Severity == SeverityCritical {
		anomalies = append(anomalies, Anomaly{
			Type:        "unauthorized-access-attempt",
			Severity:    "high",
			Description: fmt.Sprintf("Unauthorized access attempt to %s by %s", event.Target.ID, event.Actor.ID),
			Timestamp:   event.Timestamp,
		})
	}

	// Pattern 3: Unusual activity hours
	hour := event.Timestamp.Hour()
	if hour < 6 || hour > 22 {
		anomalies = append(anomalies, Anomaly{
			Type:        "unusual-activity-hours",
			Severity:    "medium",
			Description: fmt.Sprintf("Activity detected during unusual hours: %02d:00", hour),
			Timestamp:   event.Timestamp,
		})
	}

	// Log anomalies
	for _, anomaly := range anomalies {
		al.logAnomaly(anomaly)
	}
}

// countRecentFailedLogins counts recent failed login attempts
func (al *AuditLogger) countRecentFailedLogins(actorID string, window time.Duration) int {
	al.searchIndex.mu.RLock()
	defer al.searchIndex.mu.RUnlock()

	actorKey := fmt.Sprintf("user:%s", actorID)
	events := al.searchIndex.byActor[actorKey]

	count := 0
	cutoff := time.Now().Add(-window)

	for _, event := range events {
		if event.EventType == EventUserLogin && event.Result == "failure" && event.Timestamp.After(cutoff) {
			count++
		}
	}

	return count
}

// logAnomaly logs a detected anomaly
func (al *AuditLogger) logAnomaly(anomaly Anomaly) {
	// In production, this would trigger alerts
	fmt.Printf("ANOMALY DETECTED: [%s] %s\n", anomaly.Severity, anomaly.Description)
}

// SearchEvents searches audit events based on criteria
func (al *AuditLogger) SearchEvents(ctx context.Context, criteria SearchCriteria) ([]*AuditEvent, error) {
	al.searchIndex.mu.RLock()
	defer al.searchIndex.mu.RUnlock()

	al.metrics.mu.Lock()
	al.metrics.SearchQueries++
	al.metrics.mu.Unlock()

	var results []*AuditEvent

	// Search by event type
	if criteria.EventType != "" {
		results = al.searchIndex.byEventType[AuditEventType(criteria.EventType)]
	}

	// Search by actor
	if criteria.ActorID != "" {
		actorKey := fmt.Sprintf("%s:%s", criteria.ActorType, criteria.ActorID)
		results = al.searchIndex.byActor[actorKey]
	}

	// Search by target
	if criteria.TargetID != "" {
		targetKey := fmt.Sprintf("%s:%s", criteria.TargetType, criteria.TargetID)
		results = al.searchIndex.byTarget[targetKey]
	}

	// Search by tenant
	if criteria.TenantID != "" {
		results = al.searchIndex.byTenant[criteria.TenantID]
	}

	// Search by time range
	if !criteria.StartTime.IsZero() && !criteria.EndTime.IsZero() {
		results = al.filterByTimeRange(results, criteria.StartTime, criteria.EndTime)
	}

	return results, nil
}

// SearchCriteria defines search criteria for audit events
type SearchCriteria struct {
	EventType  string    `json:"event_type"`
	ActorType  string    `json:"actor_type"`
	ActorID    string    `json:"actor_id"`
	TargetType string    `json:"target_type"`
	TargetID   string    `json:"target_id"`
	TenantID   string    `json:"tenant_id"`
	StartTime  time.Time `json:"start_time"`
	EndTime    time.Time `json:"end_time"`
	Severity   string    `json:"severity"`
}

// filterByTimeRange filters events by time range
func (al *AuditLogger) filterByTimeRange(events []*AuditEvent, start, end time.Time) []*AuditEvent {
	filtered := make([]*AuditEvent, 0)

	for _, event := range events {
		if event.Timestamp.After(start) && event.Timestamp.Before(end) {
			filtered = append(filtered, event)
		}
	}

	return filtered
}

// PerformForensicAnalysis performs forensic analysis on audit logs
func (al *AuditLogger) PerformForensicAnalysis(ctx context.Context, query string, start, end time.Time) (*ForensicAnalysis, error) {
	al.forensicsEngine.mu.Lock()
	defer al.forensicsEngine.mu.Unlock()

	al.metrics.mu.Lock()
	al.metrics.ForensicAnalysisCount++
	al.metrics.mu.Unlock()

	analysis := &ForensicAnalysis{
		ID:        fmt.Sprintf("forensic-%d", time.Now().UnixNano()),
		StartTime: start,
		EndTime:   end,
		Query:     query,
		Anomalies: make([]Anomaly, 0),
		Timeline:  make([]TimelineEvent, 0),
	}

	// Get events in time range
	criteria := SearchCriteria{
		StartTime: start,
		EndTime:   end,
	}

	events, err := al.SearchEvents(ctx, criteria)
	if err != nil {
		return nil, err
	}

	analysis.EventsAnalyzed = len(events)

	// Build timeline
	for _, event := range events {
		timelineEvent := TimelineEvent{
			Timestamp:   event.Timestamp,
			EventType:   string(event.EventType),
			Description: event.Action,
			Actor:       event.Actor.Name,
			Target:      event.Target.Name,
		}
		analysis.Timeline = append(analysis.Timeline, timelineEvent)
	}

	// Detect patterns and anomalies
	analysis.Anomalies = al.analyzePatterns(events)

	// Generate recommendations
	analysis.Recommendations = al.generateForensicRecommendations(analysis)

	al.forensicsEngine.analysisResults[analysis.ID] = analysis

	return analysis, nil
}

// analyzePatterns analyzes events for patterns
func (al *AuditLogger) analyzePatterns(events []*AuditEvent) []Anomaly {
	anomalies := make([]Anomaly, 0)

	// Pattern analysis logic
	failedAccessCount := 0
	complianceViolationCount := 0

	for _, event := range events {
		if event.EventType == EventAccessDenied {
			failedAccessCount++
		}
		if event.EventType == EventComplianceViolation {
			complianceViolationCount++
		}
	}

	if failedAccessCount > 10 {
		anomalies = append(anomalies, Anomaly{
			Type:        "excessive-access-denials",
			Severity:    "high",
			Description: fmt.Sprintf("Detected %d access denial events", failedAccessCount),
		})
	}

	if complianceViolationCount > 0 {
		anomalies = append(anomalies, Anomaly{
			Type:        "compliance-violations",
			Severity:    "critical",
			Description: fmt.Sprintf("Detected %d compliance violation events", complianceViolationCount),
		})
	}

	return anomalies
}

// generateForensicRecommendations generates forensic recommendations
func (al *AuditLogger) generateForensicRecommendations(analysis *ForensicAnalysis) []string {
	recommendations := make([]string, 0)

	if len(analysis.Anomalies) > 0 {
		recommendations = append(recommendations, "Review detected anomalies and investigate potential security incidents")
	}

	if analysis.EventsAnalyzed > 10000 {
		recommendations = append(recommendations, "Consider implementing additional logging filters to reduce noise")
	}

	recommendations = append(recommendations, "Ensure all critical events are being monitored and alerted on")
	recommendations = append(recommendations, "Review access control policies based on access denial patterns")

	return recommendations
}

// VerifyIntegrity verifies the integrity of the audit log chain
func (al *AuditLogger) VerifyIntegrity() (bool, []string) {
	al.mu.RLock()
	defer al.mu.RUnlock()

	if !al.tamperProtection {
		return true, []string{"Tamper protection not enabled"}
	}

	tamperedEvents := make([]string, 0)

	for i := 1; i < len(al.events); i++ {
		event := al.events[i]
		previousEvent := al.events[i-1]

		// Verify hash chain
		if event.PreviousHash != previousEvent.Hash {
			tamperedEvents = append(tamperedEvents, event.ID)
			al.metrics.mu.Lock()
			al.metrics.TamperAttempts++
			al.metrics.mu.Unlock()
		}

		// Verify event hash
		calculatedHash := al.calculateEventHash(event)
		if calculatedHash != event.Hash {
			tamperedEvents = append(tamperedEvents, event.ID)
			al.metrics.mu.Lock()
			al.metrics.TamperAttempts++
			al.metrics.mu.Unlock()
		}
	}

	return len(tamperedEvents) == 0, tamperedEvents
}

// GetMetrics returns audit metrics
func (al *AuditLogger) GetMetrics() *AuditMetrics {
	al.metrics.mu.RLock()
	defer al.metrics.mu.RUnlock()

	metrics := &AuditMetrics{
		TotalEvents:           al.metrics.TotalEvents,
		EventsByType:          make(map[AuditEventType]int64),
		EventsBySeverity:      make(map[AuditSeverity]int64),
		AccessDeniedCount:     al.metrics.AccessDeniedCount,
		ComplianceViolations:  al.metrics.ComplianceViolations,
		TamperAttempts:        al.metrics.TamperAttempts,
		SearchQueries:         al.metrics.SearchQueries,
		ForensicAnalysisCount: al.metrics.ForensicAnalysisCount,
	}

	for k, v := range al.metrics.EventsByType {
		metrics.EventsByType[k] = v
	}

	for k, v := range al.metrics.EventsBySeverity {
		metrics.EventsBySeverity[k] = v
	}

	return metrics
}

// generateRandomID generates a random ID
func generateRandomID() string {
	return fmt.Sprintf("%x", time.Now().UnixNano()%1000000)
}
