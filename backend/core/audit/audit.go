// Package audit provides audit trail functionality
package audit

import (
	"context"
	"time"
)

// AuditEvent represents an audit event
type AuditEvent struct {
	ID            string                 `json:"id"`
	Timestamp     time.Time              `json:"timestamp"`
	UserID        string                 `json:"user_id"`
	UserRole      string                 `json:"user_role"`
	Action        string                 `json:"action"`
	Resource      string                 `json:"resource"`
	ResourceID    string                 `json:"resource_id"`
	Result        string                 `json:"result"`
	ClientIP      string                 `json:"client_ip"`
	UserAgent     string                 `json:"user_agent"`
	Details       map[string]interface{} `json:"details"`
	TenantID      string                 `json:"tenant_id"`
	SessionID     string                 `json:"session_id"`
}

// AuditLogger interface for audit logging
type AuditLogger interface {
	LogEvent(ctx context.Context, event *AuditEvent) error
	QueryEvents(ctx context.Context, filter *AuditFilter) ([]*AuditEvent, error)
}

// AuditFilter represents filters for audit queries
type AuditFilter struct {
	UserID     string
	Action     string
	Resource   string
	StartTime  *time.Time
	EndTime    *time.Time
	TenantID   string
}

// SimpleAuditLogger is a basic implementation
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
	l.events = append(l.events, *event)
	return nil
}

// QueryEvents queries audit events
func (l *SimpleAuditLogger) QueryEvents(ctx context.Context, filter *AuditFilter) ([]*AuditEvent, error) {
	var filtered []*AuditEvent
	
	for _, event := range l.events {
		if filter != nil {
			if filter.UserID != "" && event.UserID != filter.UserID {
				continue
			}
			if filter.Action != "" && event.Action != filter.Action {
				continue
			}
			if filter.Resource != "" && event.Resource != filter.Resource {
				continue
			}
			if filter.TenantID != "" && event.TenantID != filter.TenantID {
				continue
			}
			if filter.StartTime != nil && event.Timestamp.Before(*filter.StartTime) {
				continue
			}
			if filter.EndTime != nil && event.Timestamp.After(*filter.EndTime) {
				continue
			}
		}
		
		filtered = append(filtered, &event)
	}
	
	return filtered, nil
}

// Global audit logger
var defaultLogger = NewSimpleAuditLogger()

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
	// In a real implementation, would set the global logger
}