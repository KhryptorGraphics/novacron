package conflict

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	conflictEventsLogged = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_conflict_events_logged_total",
		Help: "Total number of conflict events logged",
	}, []string{"event_type"})

	conflictPatterns = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "dwcp_conflict_patterns_total",
		Help: "Recurring conflict patterns detected",
	}, []string{"pattern"})
)

// ConflictEventType describes types of conflict events
type ConflictEventType int

const (
	EventDetected ConflictEventType = iota
	EventResolved
	EventEscalated
	EventRolledBack
	EventFailed
)

func (et ConflictEventType) String() string {
	return [...]string{"Detected", "Resolved", "Escalated", "RolledBack", "Failed"}[et]
}

// ConflictEvent represents a conflict event for audit
type ConflictEvent struct {
	ID           string
	ConflictID   string
	EventType    ConflictEventType
	Timestamp    time.Time
	ResourceID   string
	Strategy     StrategyType
	Success      bool
	Message      string
	Metadata     map[string]interface{}
	UserID       string
	NodeID       string
}

// AuditLog maintains conflict history
type AuditLog struct {
	mu                 sync.RWMutex
	events             []ConflictEvent
	eventsByConflict   map[string][]ConflictEvent
	eventsByResource   map[string][]ConflictEvent
	rollbackHistory    map[string][]RollbackPoint
	patternDetector    *PatternDetector
	maxEvents          int
	retentionPeriod    time.Duration
	enableCompression  bool
}

// RollbackPoint represents a state snapshot for rollback
type RollbackPoint struct {
	ID          string
	ConflictID  string
	Timestamp   time.Time
	ResourceID  string
	State       interface{}
	Checksum    string
	Description string
}

// NewAuditLog creates a new audit log
func NewAuditLog(maxEvents int, retention time.Duration) *AuditLog {
	return &AuditLog{
		events:             make([]ConflictEvent, 0, maxEvents),
		eventsByConflict:   make(map[string][]ConflictEvent),
		eventsByResource:   make(map[string][]ConflictEvent),
		rollbackHistory:    make(map[string][]RollbackPoint),
		patternDetector:    NewPatternDetector(),
		maxEvents:          maxEvents,
		retentionPeriod:    retention,
		enableCompression:  true,
	}
}

// LogEvent logs a conflict event
func (al *AuditLog) LogEvent(event ConflictEvent) {
	al.mu.Lock()
	defer al.mu.Unlock()

	event.Timestamp = time.Now()

	// Add to main log
	al.events = append(al.events, event)

	// Index by conflict ID
	al.eventsByConflict[event.ConflictID] = append(al.eventsByConflict[event.ConflictID], event)

	// Index by resource ID
	al.eventsByResource[event.ResourceID] = append(al.eventsByResource[event.ResourceID], event)

	// Update metrics
	conflictEventsLogged.WithLabelValues(event.EventType.String()).Inc()

	// Detect patterns
	al.patternDetector.AnalyzeEvent(event)

	// Trim if necessary
	if len(al.events) > al.maxEvents {
		al.events = al.events[len(al.events)-al.maxEvents:]
	}
}

// GetConflictHistory returns all events for a conflict
func (al *AuditLog) GetConflictHistory(conflictID string) []ConflictEvent {
	al.mu.RLock()
	defer al.mu.RUnlock()
	return al.eventsByConflict[conflictID]
}

// GetResourceHistory returns all conflict events for a resource
func (al *AuditLog) GetResourceHistory(resourceID string) []ConflictEvent {
	al.mu.RLock()
	defer al.mu.RUnlock()
	return al.eventsByResource[resourceID]
}

// CreateRollbackPoint creates a rollback point
func (al *AuditLog) CreateRollbackPoint(conflictID, resourceID string, state interface{}, description string) (string, error) {
	al.mu.Lock()
	defer al.mu.Unlock()

	// Serialize state
	stateJSON, err := json.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("failed to serialize state: %w", err)
	}

	// Calculate checksum (simplified)
	checksum := fmt.Sprintf("%x", len(stateJSON))

	point := RollbackPoint{
		ID:          generateRollbackID(),
		ConflictID:  conflictID,
		Timestamp:   time.Now(),
		ResourceID:  resourceID,
		State:       state,
		Checksum:    checksum,
		Description: description,
	}

	al.rollbackHistory[conflictID] = append(al.rollbackHistory[conflictID], point)

	return point.ID, nil
}

// GetRollbackPoints returns all rollback points for a conflict
func (al *AuditLog) GetRollbackPoints(conflictID string) []RollbackPoint {
	al.mu.RLock()
	defer al.mu.RUnlock()
	return al.rollbackHistory[conflictID]
}

// Rollback performs rollback to a specific point
func (al *AuditLog) Rollback(rollbackID string) (interface{}, error) {
	al.mu.RLock()
	defer al.mu.RUnlock()

	for _, points := range al.rollbackHistory {
		for _, point := range points {
			if point.ID == rollbackID {
				return point.State, nil
			}
		}
	}

	return nil, fmt.Errorf("rollback point %s not found", rollbackID)
}

// CleanupOldEvents removes events older than retention period
func (al *AuditLog) CleanupOldEvents() {
	al.mu.Lock()
	defer al.mu.Unlock()

	cutoff := time.Now().Add(-al.retentionPeriod)

	// Clean main events
	validEvents := make([]ConflictEvent, 0)
	for _, event := range al.events {
		if event.Timestamp.After(cutoff) {
			validEvents = append(validEvents, event)
		}
	}
	al.events = validEvents

	// Clean indexed events
	for conflictID, events := range al.eventsByConflict {
		validEvents := make([]ConflictEvent, 0)
		for _, event := range events {
			if event.Timestamp.After(cutoff) {
				validEvents = append(validEvents, event)
			}
		}
		if len(validEvents) > 0 {
			al.eventsByConflict[conflictID] = validEvents
		} else {
			delete(al.eventsByConflict, conflictID)
		}
	}

	for resourceID, events := range al.eventsByResource {
		validEvents := make([]ConflictEvent, 0)
		for _, event := range events {
			if event.Timestamp.After(cutoff) {
				validEvents = append(validEvents, event)
			}
		}
		if len(validEvents) > 0 {
			al.eventsByResource[resourceID] = validEvents
		} else {
			delete(al.eventsByResource, resourceID)
		}
	}

	// Clean rollback points
	for conflictID, points := range al.rollbackHistory {
		validPoints := make([]RollbackPoint, 0)
		for _, point := range points {
			if point.Timestamp.After(cutoff) {
				validPoints = append(validPoints, point)
			}
		}
		if len(validPoints) > 0 {
			al.rollbackHistory[conflictID] = validPoints
		} else {
			delete(al.rollbackHistory, conflictID)
		}
	}
}

// GetStatistics returns audit statistics
func (al *AuditLog) GetStatistics() AuditStatistics {
	al.mu.RLock()
	defer al.mu.RUnlock()

	stats := AuditStatistics{
		TotalEvents:        len(al.events),
		EventsByType:       make(map[string]int),
		EventsByStrategy:   make(map[string]int),
		TotalConflicts:     len(al.eventsByConflict),
		TotalResources:     len(al.eventsByResource),
		TotalRollbacks:     0,
	}

	for _, event := range al.events {
		stats.EventsByType[event.EventType.String()]++
		stats.EventsByStrategy[event.Strategy.String()]++
	}

	for _, points := range al.rollbackHistory {
		stats.TotalRollbacks += len(points)
	}

	return stats
}

// AuditStatistics provides audit statistics
type AuditStatistics struct {
	TotalEvents      int
	EventsByType     map[string]int
	EventsByStrategy map[string]int
	TotalConflicts   int
	TotalResources   int
	TotalRollbacks   int
}

// PatternDetector detects recurring conflict patterns
type PatternDetector struct {
	mu               sync.RWMutex
	resourcePatterns map[string]*ResourcePattern
	alertThreshold   int
}

// ResourcePattern tracks conflict patterns for a resource
type ResourcePattern struct {
	ResourceID      string
	ConflictCount   int
	LastConflict    time.Time
	ConflictTypes   map[ConflictType]int
	Strategies      map[StrategyType]int
	FailureRate     float64
}

// NewPatternDetector creates a new pattern detector
func NewPatternDetector() *PatternDetector {
	return &PatternDetector{
		resourcePatterns: make(map[string]*ResourcePattern),
		alertThreshold:   5,
	}
}

// AnalyzeEvent analyzes an event for patterns
func (pd *PatternDetector) AnalyzeEvent(event ConflictEvent) {
	pd.mu.Lock()
	defer pd.mu.Unlock()

	pattern, exists := pd.resourcePatterns[event.ResourceID]
	if !exists {
		pattern = &ResourcePattern{
			ResourceID:    event.ResourceID,
			ConflictTypes: make(map[ConflictType]int),
			Strategies:    make(map[StrategyType]int),
		}
		pd.resourcePatterns[event.ResourceID] = pattern
	}

	if event.EventType == EventDetected {
		pattern.ConflictCount++
		pattern.LastConflict = time.Now()
	}

	if event.EventType == EventFailed {
		pattern.FailureRate = float64(pattern.FailureRate*float64(pattern.ConflictCount-1)+1) / float64(pattern.ConflictCount)
	}

	// Alert on recurring patterns
	if pattern.ConflictCount >= pd.alertThreshold {
		conflictPatterns.WithLabelValues(event.ResourceID).Inc()
	}
}

// GetHotspots returns resources with frequent conflicts
func (pd *PatternDetector) GetHotspots(limit int) []*ResourcePattern {
	pd.mu.RLock()
	defer pd.mu.RUnlock()

	patterns := make([]*ResourcePattern, 0, len(pd.resourcePatterns))
	for _, pattern := range pd.resourcePatterns {
		patterns = append(patterns, pattern)
	}

	// Sort by conflict count (simplified)
	// In production, use proper sorting
	if len(patterns) > limit {
		patterns = patterns[:limit]
	}

	return patterns
}

func generateRollbackID() string {
	return fmt.Sprintf("rb-%d", time.Now().UnixNano())
}

// ExportEvents exports events to JSON
func (al *AuditLog) ExportEvents(ctx context.Context, startTime, endTime time.Time) ([]byte, error) {
	al.mu.RLock()
	defer al.mu.RUnlock()

	filteredEvents := make([]ConflictEvent, 0)
	for _, event := range al.events {
		if event.Timestamp.After(startTime) && event.Timestamp.Before(endTime) {
			filteredEvents = append(filteredEvents, event)
		}
	}

	return json.MarshalIndent(filteredEvents, "", "  ")
}

// ImportEvents imports events from JSON
func (al *AuditLog) ImportEvents(ctx context.Context, data []byte) error {
	var events []ConflictEvent
	if err := json.Unmarshal(data, &events); err != nil {
		return fmt.Errorf("failed to unmarshal events: %w", err)
	}

	al.mu.Lock()
	defer al.mu.Unlock()

	for _, event := range events {
		al.events = append(al.events, event)
		al.eventsByConflict[event.ConflictID] = append(al.eventsByConflict[event.ConflictID], event)
		al.eventsByResource[event.ResourceID] = append(al.eventsByResource[event.ResourceID], event)
	}

	return nil
}
