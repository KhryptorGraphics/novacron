package security

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/google/uuid"
)

// AuditEvent represents a security audit event
type AuditEvent struct {
	ID          string                 `json:"id"`
	Timestamp   time.Time              `json:"timestamp"`
	EventType   AuditEventType         `json:"event_type"`
	Actor       string                 `json:"actor"`
	Resource    string                 `json:"resource"`
	Action      AuditAction            `json:"action"`
	Result      AuditResult            `json:"result"`
	IPAddress   string                 `json:"ip_address,omitempty"`
	UserAgent   string                 `json:"user_agent,omitempty"`
	RequestID   string                 `json:"request_id,omitempty"`
	Details     map[string]interface{} `json:"details,omitempty"`
	ErrorMsg    string                 `json:"error,omitempty"`
	Sensitivity SensitivityLevel       `json:"sensitivity"`
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

// AuditLogger interface for audit logging
type AuditLogger interface {
	LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error
	LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error
	LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error
	LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error
	LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error
	Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error)
}

// AuditFilter for querying audit logs
type AuditFilter struct {
	StartTime   *time.Time
	EndTime     *time.Time
	EventTypes  []AuditEventType
	Actors      []string
	Resources   []string
	Results     []AuditResult
	Limit       int
	Offset      int
}

// DefaultAuditLogger implements AuditLogger
type DefaultAuditLogger struct {
	logger      *slog.Logger
	storage     AuditStorage
	alerting    AlertingService
	compliance  ComplianceService
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

// NewAuditLogger creates a new audit logger
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

// LogSecretAccess logs secret access events
func (a *DefaultAuditLogger) LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	event := AuditEvent{
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

	// Extract context values
	if reqID := ctx.Value("request_id"); reqID != nil {
		event.RequestID = reqID.(string)
	}
	if ip := ctx.Value("client_ip"); ip != nil {
		event.IPAddress = ip.(string)
	}
	if ua := ctx.Value("user_agent"); ua != nil {
		event.UserAgent = ua.(string)
	}

	// Log to structured logger
	a.logger.Info("Secret access audit",
		"event_id", event.ID,
		"actor", actor,
		"resource", resource,
		"action", action,
		"result", result,
	)

	// Store in persistent storage
	if err := a.storage.Store(ctx, event); err != nil {
		a.logger.Error("Failed to store audit event", "error", err)
		return fmt.Errorf("failed to store audit event: %w", err)
	}

	// Alert on failures or sensitive operations
	if result == ResultFailure || result == ResultDenied {
		if err := a.alerting.SendSecurityAlert(ctx, event); err != nil {
			a.logger.Error("Failed to send security alert", "error", err)
		}
	}

	// Report for compliance
	if err := a.compliance.ReportEvent(ctx, event); err != nil {
		a.logger.Error("Failed to report compliance event", "error", err)
	}

	return nil
}

// LogSecretModification logs secret modification events
func (a *DefaultAuditLogger) LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	event := AuditEvent{
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

	// Extract context values
	if reqID := ctx.Value("request_id"); reqID != nil {
		event.RequestID = reqID.(string)
	}

	// Log and store
	a.logger.Warn("Secret modification audit",
		"event_id", event.ID,
		"actor", actor,
		"resource", resource,
		"action", action,
	)

	if err := a.storage.Store(ctx, event); err != nil {
		return fmt.Errorf("failed to store modification audit: %w", err)
	}

	// Always alert on modifications
	if err := a.alerting.SendSecurityAlert(ctx, event); err != nil {
		a.logger.Error("Failed to send modification alert", "error", err)
	}

	return a.compliance.ReportEvent(ctx, event)
}

// LogSecretRotation logs secret rotation events
func (a *DefaultAuditLogger) LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error {
	event := AuditEvent{
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

	a.logger.Info("Secret rotation audit",
		"event_id", event.ID,
		"actor", actor,
		"resource", resource,
		"result", result,
	)

	return a.storage.Store(ctx, event)
}

// LogAuthEvent logs authentication events
func (a *DefaultAuditLogger) LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error {
	result := ResultSuccess
	if !success {
		result = ResultFailure
	}

	event := AuditEvent{
		ID:          uuid.New().String(),
		Timestamp:   time.Now().UTC(),
		EventType:   EventAuthAttempt,
		Actor:       actor,
		Action:      ActionRead,
		Result:      result,
		Details:     details,
		Sensitivity: SensitivityConfidential,
	}

	// Log failed auth attempts at higher level
	if !success {
		a.logger.Warn("Failed authentication attempt",
			"actor", actor,
			"details", details,
		)
		// Alert on failed auth
		if err := a.alerting.SendSecurityAlert(ctx, event); err != nil {
			a.logger.Error("Failed to send auth alert", "error", err)
		}
	}

	return a.storage.Store(ctx, event)
}

// LogConfigChange logs configuration changes
func (a *DefaultAuditLogger) LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error {
	event := AuditEvent{
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

	a.logger.Info("Configuration change audit",
		"actor", actor,
		"resource", resource,
	)

	return a.storage.Store(ctx, event)
}

// Query retrieves audit events based on filter
func (a *DefaultAuditLogger) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error) {
	return a.storage.Query(ctx, filter)
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
			ip_address, user_agent, request_id, details, error_msg, sensitivity
		) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
	`

	_, err = d.db.ExecContext(ctx, query,
		event.ID, event.Timestamp, event.EventType, event.Actor,
		event.Resource, event.Action, event.Result, event.IPAddress,
		event.UserAgent, event.RequestID, detailsJSON, event.ErrorMsg,
		event.Sensitivity,
	)

	return err
}

// Query retrieves audit events from database
func (d *DatabaseAuditStorage) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error) {
	// Implementation would build dynamic query based on filter
	// Simplified for brevity
	query := `
		SELECT id, timestamp, event_type, actor, resource, action, result,
		       ip_address, user_agent, request_id, details, error_msg, sensitivity
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
			&event.Sensitivity,
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