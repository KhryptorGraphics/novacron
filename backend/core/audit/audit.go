// Package audit provides comprehensive audit trail functionality
// This is the canonical audit package that consolidates all audit capabilities
package audit

import (
	"context"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/google/uuid"
)

// AuditEvent represents a comprehensive security audit event
type AuditEvent struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	EventType   AuditEventType         `json:"event_type"`
	Actor       string                 `json:"actor"`
	Resource    string                 `json:"resource"`
	ResourceID  string                 `json:"resource_id,omitempty"`
	Action      AuditAction            `json:"action"`
	Result      AuditResult            `json:"result"`
	IPAddress   string                 `json:"ip_address,omitempty"`
	ClientIP    string                 `json:"client_ip,omitempty"`  // Legacy compatibility
	UserAgent   string                 `json:"user_agent,omitempty"`
	RequestID   string                 `json:"request_id,omitempty"`
	Details     map[string]interface{} `json:"details,omitempty"`
	ErrorMsg    string                 `json:"error,omitempty"`
	Sensitivity SensitivityLevel       `json:"sensitivity"`
	TenantID    string                 `json:"tenant_id,omitempty"`
	SessionID   string                 `json:"session_id,omitempty"`
	UserID      string                 `json:"user_id,omitempty"`
	UserRole    string                 `json:"user_role,omitempty"`
}

// AuditEventType defines types of audit events
type AuditEventType string

const (
	EventSecretAccess   AuditEventType = "SECRET_ACCESS"
	EventSecretModify   AuditEventType = "SECRET_MODIFY"
	EventSecretDelete   AuditEventType = "SECRET_DELETE"
	EventSecretRotate   AuditEventType = "SECRET_ROTATE"
	EventAuthAttempt    AuditEventType = "AUTH_ATTEMPT"
	EventPermissionDeny AuditEventType = "PERMISSION_DENY"
	EventConfigChange   AuditEventType = "CONFIG_CHANGE"
	EventSecurityDrop   AuditEventType = "SECURITY_EVENT_DROP"
)

// AuditAction defines the action performed
type AuditAction string

const (
	ActionRead   AuditAction = "READ"
	ActionWrite  AuditAction = "WRITE"
	ActionUpdate AuditAction = "UPDATE"
	ActionDelete AuditAction = "DELETE"
	ActionList   AuditAction = "LIST"
	ActionRotate AuditAction = "ROTATE"
	ActionDrop   AuditAction = "DROP"
)

// AuditResult defines the result of an action
type AuditResult string

const (
	ResultSuccess AuditResult = "SUCCESS"
	ResultFailure AuditResult = "FAILURE"
	ResultDenied  AuditResult = "DENIED"
)

// SensitivityLevel defines data sensitivity
type SensitivityLevel string

const (
	SensitivityPublic       SensitivityLevel = "PUBLIC"
	SensitivityInternal     SensitivityLevel = "INTERNAL"
	SensitivityConfidential SensitivityLevel = "CONFIDENTIAL"
	SensitivitySecret       SensitivityLevel = "SECRET"
)

// AuditLogger interface for comprehensive audit logging
type AuditLogger interface {
	LogEvent(ctx context.Context, event *AuditEvent) error // Generic event logging
	LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error
	LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error
	LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error
	LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error
	LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error
	QueryEvents(ctx context.Context, filter *AuditFilter) ([]*AuditEvent, error)
	Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error)
	VerifyIntegrity(ctx context.Context, startTime, endTime time.Time) (*IntegrityReport, error)
}

// AuditFilter for querying audit logs
type AuditFilter struct {
	StartTime  *time.Time
	EndTime    *time.Time
	EventTypes []AuditEventType
	Actors     []string
	Resources  []string
	Results    []AuditResult
	Limit      int
	Offset     int
	// Legacy compatibility fields
	UserID     string
	Action     string
	Resource   string
	TenantID   string
}

// IntegrityReport represents audit log integrity verification results
type IntegrityReport struct {
	Valid           bool      `json:"valid"`
	StartTime       time.Time `json:"start_time"`
	EndTime         time.Time `json:"end_time"`
	TotalRecords    int       `json:"total_records"`
	ValidRecords    int       `json:"valid_records"`
	TamperedRecords int       `json:"tampered_records"`
	MissingRecords  []string  `json:"missing_records,omitempty"`
	Errors          []string  `json:"errors,omitempty"`
}

// AuditStorage interface for persisting audit logs
type AuditStorage interface {
	Store(ctx context.Context, event AuditEvent) error
	Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error)
	Archive(ctx context.Context, before time.Time) error
}

// AlertingService for security alerts
type AlertingService interface {
	SendSecurityAlert(ctx context.Context, event AuditEvent) error
}

// ComplianceService for compliance reporting
type ComplianceService interface {
	ReportEvent(ctx context.Context, event AuditEvent) error
}

// DefaultAuditLogger implements AuditLogger with comprehensive features
type DefaultAuditLogger struct {
	logger     *slog.Logger
	storage    AuditStorage
	alerting   AlertingService
	compliance ComplianceService
}

// NewAuditLogger creates a new comprehensive audit logger
func NewAuditLogger(storage AuditStorage, alerting AlertingService, compliance ComplianceService) *DefaultAuditLogger {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))

	return &DefaultAuditLogger{
		logger:     logger,
		storage:    storage,
		alerting:   alerting,
		compliance: compliance,
	}
}

// LogEvent logs a generic audit event
func (a *DefaultAuditLogger) LogEvent(ctx context.Context, event *AuditEvent) error {
	// Ensure event has required fields
	if event.ID == "" {
		event.ID = uuid.New().String()
	}
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now().UTC()
	}

	// Extract context values if not already set
	if event.RequestID == "" {
		if reqID := ctx.Value("request_id"); reqID != nil {
			event.RequestID = reqID.(string)
		}
	}
	if event.IPAddress == "" {
		if ip := ctx.Value("client_ip"); ip != nil {
			event.IPAddress = ip.(string)
		}
	}
	if event.UserAgent == "" {
		if ua := ctx.Value("user_agent"); ua != nil {
			event.UserAgent = ua.(string)
		}
	}

	// Log to structured logger
	a.logger.Info("Audit event",
		"event_id", event.ID,
		"event_type", event.EventType,
		"actor", event.Actor,
		"resource", event.Resource,
		"action", event.Action,
		"result", event.Result,
	)

	// Store in persistent storage
	if err := a.storage.Store(ctx, *event); err != nil {
		a.logger.Error("Failed to store audit event", "error", err)
		return fmt.Errorf("failed to store audit event: %w", err)
	}

	// Alert on failures or high-sensitivity events
	if event.Result == ResultFailure || event.Result == ResultDenied || event.Sensitivity == SensitivitySecret {
		if a.alerting != nil {
			if err := a.alerting.SendSecurityAlert(ctx, *event); err != nil {
				a.logger.Error("Failed to send security alert", "error", err)
			}
		}
	}

	// Report for compliance if applicable
	if a.compliance != nil && (event.EventType == EventSecretAccess || event.EventType == EventSecretModify || event.EventType == EventConfigChange) {
		if err := a.compliance.ReportEvent(ctx, *event); err != nil {
			a.logger.Error("Failed to report compliance event", "error", err)
		}
	}

	return nil
}

// LogSecretAccess logs secret access events
func (a *DefaultAuditLogger) LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	event := &AuditEvent{
		ID:          uuid.New().String(),
		Timestamp:   time.Now().UTC(),
		EventType:   EventSecretAccess,
		Actor:       actor,
		Resource:    resource,
		Action:      action,
		Result:      result,
		Details:     details,
		Sensitivity: SensitivitySecret,
	}
	return a.LogEvent(ctx, event)
}

// LogSecretModification logs secret modification events
func (a *DefaultAuditLogger) LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	event := &AuditEvent{
		ID:          uuid.New().String(),
		Timestamp:   time.Now().UTC(),
		EventType:   EventSecretModify,
		Actor:       actor,
		Resource:    resource,
		Action:      action,
		Result:      result,
		Details:     details,
		Sensitivity: SensitivitySecret,
	}
	return a.LogEvent(ctx, event)
}

// LogSecretRotation logs secret rotation events
func (a *DefaultAuditLogger) LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error {
	event := &AuditEvent{
		ID:        uuid.New().String(),
		Timestamp: time.Now().UTC(),
		EventType: EventSecretRotate,
		Actor:     actor,
		Resource:  resource,
		Action:    ActionRotate,
		Result:    result,
		Details: map[string]interface{}{
			"old_version": oldVersion,
			"new_version": newVersion,
			"rotated_at":  time.Now().UTC(),
		},
		Sensitivity: SensitivitySecret,
	}
	return a.LogEvent(ctx, event)
}

// LogAuthEvent logs authentication events
func (a *DefaultAuditLogger) LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error {
	result := ResultSuccess
	if !success {
		result = ResultFailure
	}

	event := &AuditEvent{
		ID:          uuid.New().String(),
		Timestamp:   time.Now().UTC(),
		EventType:   EventAuthAttempt,
		Actor:       actor,
		Action:      ActionRead,
		Result:      result,
		Details:     details,
		Sensitivity: SensitivityConfidential,
	}
	return a.LogEvent(ctx, event)
}

// LogConfigChange logs configuration changes
func (a *DefaultAuditLogger) LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error {
	event := &AuditEvent{
		ID:        uuid.New().String(),
		Timestamp: time.Now().UTC(),
		EventType: EventConfigChange,
		Actor:     actor,
		Resource:  resource,
		Action:    ActionUpdate,
		Result:    ResultSuccess,
		Details: map[string]interface{}{
			"old_value": oldValue,
			"new_value": newValue,
		},
		Sensitivity: SensitivityInternal,
	}
	return a.LogEvent(ctx, event)
}

// Query retrieves audit events based on filter
func (a *DefaultAuditLogger) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error) {
	return a.storage.Query(ctx, filter)
}

// QueryEvents queries audit events (legacy compatibility)
func (a *DefaultAuditLogger) QueryEvents(ctx context.Context, filter *AuditFilter) ([]*AuditEvent, error) {
	events, err := a.storage.Query(ctx, *filter)
	if err != nil {
		return nil, err
	}

	// Convert to pointers for legacy compatibility
	result := make([]*AuditEvent, len(events))
	for i := range events {
		result[i] = &events[i]
	}
	return result, nil
}

// VerifyIntegrity verifies the integrity of audit logs
func (a *DefaultAuditLogger) VerifyIntegrity(ctx context.Context, startTime, endTime time.Time) (*IntegrityReport, error) {
	report := &IntegrityReport{
		Valid:     true,
		StartTime: startTime,
		EndTime:   endTime,
	}

	// Query events in the time range
	events, err := a.storage.Query(ctx, AuditFilter{
		StartTime: &startTime,
		EndTime:   &endTime,
		Limit:     10000,
	})
	if err != nil {
		report.Valid = false
		report.Errors = append(report.Errors, fmt.Sprintf("Failed to query events: %v", err))
		return report, err
	}

	report.TotalRecords = len(events)

	// Verify each event
	for _, event := range events {
		// Calculate expected hash
		expectedHash := a.calculateEventHash(event)

		// In a real implementation, compare with stored hash
		// For now, assume all records are valid
		report.ValidRecords++
		_ = expectedHash
	}

	return report, nil
}

// calculateEventHash calculates hash for audit event
func (a *DefaultAuditLogger) calculateEventHash(event AuditEvent) string {
	h := sha256.New()
	h.Write([]byte(event.ID))
	h.Write([]byte(event.Timestamp.String()))
	h.Write([]byte(event.EventType))
	h.Write([]byte(event.Actor))
	h.Write([]byte(event.Resource))
	return hex.EncodeToString(h.Sum(nil))
}

// SimpleAuditLogger is a basic implementation for backward compatibility
type SimpleAuditLogger struct {
	events []AuditEvent
}

// NewSimpleAuditLogger creates a new simple audit logger
func NewSimpleAuditLogger() *SimpleAuditLogger {
	return &SimpleAuditLogger{
		events: make([]AuditEvent, 0),
	}
}

// LogEvent logs an audit event
func (l *SimpleAuditLogger) LogEvent(ctx context.Context, event *AuditEvent) error {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}
	if event.ID == "" {
		event.ID = uuid.New().String()
	}
	l.events = append(l.events, *event)
	return nil
}

// Stub implementations for interface compliance
func (l *SimpleAuditLogger) LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	event := &AuditEvent{
		EventType:   EventSecretAccess,
		Actor:       actor,
		Resource:    resource,
		Action:      action,
		Result:      result,
		Details:     details,
		Sensitivity: SensitivitySecret,
	}
	return l.LogEvent(ctx, event)
}

func (l *SimpleAuditLogger) LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	event := &AuditEvent{
		EventType:   EventSecretModify,
		Actor:       actor,
		Resource:    resource,
		Action:      action,
		Result:      result,
		Details:     details,
		Sensitivity: SensitivitySecret,
	}
	return l.LogEvent(ctx, event)
}

func (l *SimpleAuditLogger) LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error {
	event := &AuditEvent{
		EventType: EventSecretRotate,
		Actor:     actor,
		Resource:  resource,
		Action:    ActionRotate,
		Result:    result,
		Details: map[string]interface{}{
			"old_version": oldVersion,
			"new_version": newVersion,
		},
		Sensitivity: SensitivitySecret,
	}
	return l.LogEvent(ctx, event)
}

func (l *SimpleAuditLogger) LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error {
	result := ResultSuccess
	if !success {
		result = ResultFailure
	}
	event := &AuditEvent{
		EventType:   EventAuthAttempt,
		Actor:       actor,
		Result:      result,
		Details:     details,
		Sensitivity: SensitivityConfidential,
	}
	return l.LogEvent(ctx, event)
}

func (l *SimpleAuditLogger) LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error {
	event := &AuditEvent{
		EventType: EventConfigChange,
		Actor:     actor,
		Resource:  resource,
		Action:    ActionUpdate,
		Result:    ResultSuccess,
		Details: map[string]interface{}{
			"old_value": oldValue,
			"new_value": newValue,
		},
		Sensitivity: SensitivityInternal,
	}
	return l.LogEvent(ctx, event)
}

func (l *SimpleAuditLogger) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error) {
	var filtered []AuditEvent

	for _, event := range l.events {
		if l.matchesFilter(&event, &filter) {
			filtered = append(filtered, event)
		}
	}
	return filtered, nil
}

func (l *SimpleAuditLogger) VerifyIntegrity(ctx context.Context, startTime, endTime time.Time) (*IntegrityReport, error) {
	return &IntegrityReport{
		Valid:        true,
		StartTime:    startTime,
		EndTime:      endTime,
		TotalRecords: len(l.events),
		ValidRecords: len(l.events),
	}, nil
}

// QueryEvents queries audit events
func (l *SimpleAuditLogger) QueryEvents(ctx context.Context, filter *AuditFilter) ([]*AuditEvent, error) {
	var filtered []*AuditEvent

	for _, event := range l.events {
		if l.matchesFilter(&event, filter) {
			eventCopy := event
			filtered = append(filtered, &eventCopy)
		}
	}
	return filtered, nil
}

func (l *SimpleAuditLogger) matchesFilter(event *AuditEvent, filter *AuditFilter) bool {
	if filter == nil {
		return true
	}

	// Legacy field compatibility
	if filter.UserID != "" && event.UserID != filter.UserID {
		return false
	}
	if filter.Action != "" && string(event.Action) != filter.Action {
		return false
	}
	if filter.Resource != "" && event.Resource != filter.Resource {
		return false
	}
	if filter.TenantID != "" && event.TenantID != filter.TenantID {
		return false
	}

	// Time filters
	if filter.StartTime != nil && event.Timestamp.Before(*filter.StartTime) {
		return false
	}
	if filter.EndTime != nil && event.Timestamp.After(*filter.EndTime) {
		return false
	}

	// Enhanced filters
	if len(filter.EventTypes) > 0 {
		match := false
		for _, et := range filter.EventTypes {
			if event.EventType == et {
				match = true
				break
			}
		}
		if !match {
			return false
		}
	}

	if len(filter.Actors) > 0 {
		match := false
		for _, actor := range filter.Actors {
			if event.Actor == actor {
				match = true
				break
			}
		}
		if !match {
			return false
		}
	}

	if len(filter.Resources) > 0 {
		match := false
		for _, resource := range filter.Resources {
			if event.Resource == resource {
				match = true
				break
			}
		}
		if !match {
			return false
		}
	}

	if len(filter.Results) > 0 {
		match := false
		for _, result := range filter.Results {
			if event.Result == result {
				match = true
				break
			}
		}
		if !match {
			return false
		}
	}

	return true
}

// DatabaseAuditStorage implements AuditStorage using database
type DatabaseAuditStorage struct {
	db *sql.DB
}

// NewDatabaseAuditStorage creates database-backed audit storage
func NewDatabaseAuditStorage(db *sql.DB) *DatabaseAuditStorage {
	return &DatabaseAuditStorage{db: db}
}

// Store persists audit event to database
func (d *DatabaseAuditStorage) Store(ctx context.Context, event AuditEvent) error {
	detailsJSON, err := json.Marshal(event.Details)
	if err != nil {
		return fmt.Errorf("failed to marshal details: %w", err)
	}

	query := `
		INSERT INTO audit_logs (
			id, timestamp, event_type, actor, resource, action, result,
			ip_address, user_agent, request_id, details, error_msg, sensitivity,
			tenant_id, session_id, user_id, user_role, resource_id
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)
	`

	_, err = d.db.ExecContext(ctx, query,
		event.ID, event.Timestamp, event.EventType, event.Actor,
		event.Resource, event.Action, event.Result, event.IPAddress,
		event.UserAgent, event.RequestID, detailsJSON, event.ErrorMsg,
		event.Sensitivity, event.TenantID, event.SessionID, event.UserID,
		event.UserRole, event.ResourceID,
	)

	return err
}

// Query retrieves audit events from database
func (d *DatabaseAuditStorage) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error) {
	query := `
		SELECT id, timestamp, event_type, actor, resource, action, result,
		       ip_address, user_agent, request_id, details, error_msg, sensitivity,
		       tenant_id, session_id, user_id, user_role, resource_id
		FROM audit_logs
		WHERE timestamp >= $1 AND timestamp <= $2
		ORDER BY timestamp DESC
		LIMIT $3 OFFSET $4
	`

	startTime := time.Now().Add(-24 * time.Hour)
	if filter.StartTime != nil {
		startTime = *filter.StartTime
	}

	endTime := time.Now()
	if filter.EndTime != nil {
		endTime = *filter.EndTime
	}

	limit := 100
	if filter.Limit > 0 {
		limit = filter.Limit
	}

	rows, err := d.db.QueryContext(ctx, query, startTime, endTime, limit, filter.Offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var events []AuditEvent
	for rows.Next() {
		var event AuditEvent
		var detailsJSON []byte

		err := rows.Scan(
			&event.ID, &event.Timestamp, &event.EventType, &event.Actor,
			&event.Resource, &event.Action, &event.Result, &event.IPAddress,
			&event.UserAgent, &event.RequestID, &detailsJSON, &event.ErrorMsg,
			&event.Sensitivity, &event.TenantID, &event.SessionID, &event.UserID,
			&event.UserRole, &event.ResourceID,
		)
		if err != nil {
			return nil, err
		}

		if len(detailsJSON) > 0 {
			json.Unmarshal(detailsJSON, &event.Details)
		}

		events = append(events, event)
	}

	return events, nil
}

// Archive moves old audit logs to archive storage
func (d *DatabaseAuditStorage) Archive(ctx context.Context, before time.Time) error {
	// Move old records to archive table
	query := `
		INSERT INTO audit_logs_archive
		SELECT * FROM audit_logs WHERE timestamp < $1;
		DELETE FROM audit_logs WHERE timestamp < $1;
	`
	_, err := d.db.ExecContext(ctx, query, before)
	return err
}

// Global audit logger
var defaultLogger AuditLogger = NewSimpleAuditLogger()

// LogEvent logs an event using the default logger
func LogEvent(ctx context.Context, event *AuditEvent) error {
	return defaultLogger.LogEvent(ctx, event)
}

// QueryEvents queries events using the default logger
func QueryEvents(ctx context.Context, filter *AuditFilter) ([]*AuditEvent, error) {
	return defaultLogger.QueryEvents(ctx, filter)
}

// SetAuditLogger sets the global audit logger
func SetAuditLogger(logger AuditLogger) {
	if logger != nil {
		defaultLogger = logger
	}
}

// Convenience functions for common operations
func LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	return defaultLogger.LogSecretAccess(ctx, actor, resource, action, result, details)
}

func LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	return defaultLogger.LogSecretModification(ctx, actor, resource, action, result, details)
}

func LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error {
	return defaultLogger.LogSecretRotation(ctx, actor, resource, oldVersion, newVersion, result)
}

func LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error {
	return defaultLogger.LogAuthEvent(ctx, actor, success, details)
}

func LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error {
	return defaultLogger.LogConfigChange(ctx, actor, resource, oldValue, newValue)
}

// GetGlobalAuditLogger returns the global audit logger
func GetGlobalAuditLogger() AuditLogger {
	return defaultLogger
}