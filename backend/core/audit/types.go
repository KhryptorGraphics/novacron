// Package audit provides comprehensive audit type definitions
package audit

import (
	"fmt"
	"time"
)

// Legacy compatibility types - kept for backward compatibility with existing code

// LegacyAuditEvent represents the original audit event structure
// Deprecated: Use AuditEvent instead
type LegacyAuditEvent struct {
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

// Auth service compatibility types
// AuditEntry represents an audit entry for auth service compatibility
type AuditEntry struct {
	ID             string                 `json:"id"`
	UserID         string                 `json:"user_id"`
	TenantID       string                 `json:"tenant_id"`
	ResourceType   string                 `json:"resource_type"`
	ResourceID     string                 `json:"resource_id"`
	Action         string                 `json:"action"`
	Success        bool                   `json:"success"`
	Timestamp      time.Time              `json:"timestamp"`
	IPAddress      string                 `json:"ip_address"`
	UserAgent      string                 `json:"user_agent"`
	Reason         string                 `json:"reason,omitempty"`
	AdditionalData map[string]interface{} `json:"additional_data,omitempty"`
}

// Auth service interface compatibility
type AuthAuditService interface {
	LogAccess(entry *AuditEntry) error
	GetAuditTrail(userID, tenantID, resourceType, resourceID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error)
	GetUserActions(userID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error)
	GetResourceActions(resourceType, resourceID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error)
	Log(entry AuditEntry) error
}

// Constants for standardized audit actions and results
const (
	// Common action types
	ACTION_LOGIN           = "LOGIN"
	ACTION_LOGOUT          = "LOGOUT"
	ACTION_CREATE          = "CREATE"
	ACTION_READ            = "READ"
	ACTION_UPDATE          = "UPDATE"
	ACTION_DELETE          = "DELETE"
	ACTION_EXPORT          = "EXPORT"
	ACTION_IMPORT          = "IMPORT"
	ACTION_PERMISSION_DENY = "PERMISSION_DENY"

	// Common result types
	RESULT_SUCCESS = "SUCCESS"
	RESULT_FAILURE = "FAILURE"
	RESULT_DENIED  = "DENIED"
	RESULT_ERROR   = "ERROR"

	// Sensitivity levels
	SENSITIVITY_PUBLIC       = "PUBLIC"
	SENSITIVITY_INTERNAL     = "INTERNAL"
	SENSITIVITY_CONFIDENTIAL = "CONFIDENTIAL"
	SENSITIVITY_SECRET       = "SECRET"
)

// Helper functions for type conversion
func ConvertLegacyToAuditEvent(legacy LegacyAuditEvent) AuditEvent {
	return AuditEvent{
		ID:          legacy.ID,
		Timestamp:   legacy.Timestamp,
		Actor:       legacy.UserID,
		UserID:      legacy.UserID,
		UserRole:    legacy.UserRole,
		Action:      AuditAction(legacy.Action),
		Resource:    legacy.Resource,
		ResourceID:  legacy.ResourceID,
		Result:      AuditResult(legacy.Result),
		ClientIP:    legacy.ClientIP,
		IPAddress:   legacy.ClientIP,
		UserAgent:   legacy.UserAgent,
		Details:     legacy.Details,
		TenantID:    legacy.TenantID,
		SessionID:   legacy.SessionID,
		EventType:   "LEGACY_EVENT",
		Sensitivity: SensitivityInternal,
	}
}

func ConvertAuditEventToLegacy(event AuditEvent) LegacyAuditEvent {
	return LegacyAuditEvent{
		ID:         event.ID,
		Timestamp:  event.Timestamp,
		UserID:     event.UserID,
		UserRole:   event.UserRole,
		Action:     string(event.Action),
		Resource:   event.Resource,
		ResourceID: event.ResourceID,
		Result:     string(event.Result),
		ClientIP:   event.ClientIP,
		UserAgent:  event.UserAgent,
		Details:    event.Details,
		TenantID:   event.TenantID,
		SessionID:  event.SessionID,
	}
}

func ConvertAuditEntryToAuditEvent(entry AuditEntry) AuditEvent {
	result := ResultSuccess
	if !entry.Success {
		result = ResultFailure
	}

	return AuditEvent{
		ID:          entry.ID,
		Timestamp:   entry.Timestamp,
		Actor:       entry.UserID,
		UserID:      entry.UserID,
		Resource:    entry.ResourceType + "/" + entry.ResourceID,
		ResourceID:  entry.ResourceID,
		Action:      AuditAction(entry.Action),
		Result:      result,
		IPAddress:   entry.IPAddress,
		UserAgent:   entry.UserAgent,
		TenantID:    entry.TenantID,
		Details: map[string]interface{}{
			"resource_type":   entry.ResourceType,
			"reason":          entry.Reason,
			"additional_data": entry.AdditionalData,
		},
		EventType:   AuditEventType(entry.Action),
		Sensitivity: SensitivityInternal,
	}
}

func ConvertAuditEventToAuditEntry(event AuditEvent) AuditEntry {
	return AuditEntry{
		ID:           event.ID,
		UserID:       event.UserID,
		TenantID:     event.TenantID,
		ResourceType: event.Resource,
		ResourceID:   event.ResourceID,
		Action:       string(event.Action),
		Success:      event.Result == ResultSuccess,
		Timestamp:    event.Timestamp,
		IPAddress:    event.IPAddress,
		UserAgent:    event.UserAgent,
		AdditionalData: event.Details,
	}

	// Extract reason if available
	if event.Details != nil {
		if reason, ok := event.Details["reason"].(string); ok {
			return AuditEntry{
				ID:             event.ID,
				UserID:         event.UserID,
				TenantID:       event.TenantID,
				ResourceType:   event.Resource,
				ResourceID:     event.ResourceID,
				Action:         string(event.Action),
				Success:        event.Result == ResultSuccess,
				Timestamp:      event.Timestamp,
				IPAddress:      event.IPAddress,
				UserAgent:      event.UserAgent,
				Reason:         reason,
				AdditionalData: event.Details,
			}
		}
	}

	return AuditEntry{
		ID:             event.ID,
		UserID:         event.UserID,
		TenantID:       event.TenantID,
		ResourceType:   event.Resource,
		ResourceID:     event.ResourceID,
		Action:         string(event.Action),
		Success:        event.Result == ResultSuccess,
		Timestamp:      event.Timestamp,
		IPAddress:      event.IPAddress,
		UserAgent:      event.UserAgent,
		AdditionalData: event.Details,
	}
}

// Validation helpers
func (e AuditEvent) IsValid() bool {
	if e.ID == "" || e.Timestamp.IsZero() || e.Actor == "" || e.Resource == "" || e.Action == "" || e.Result == "" {
		return false
	}
	return true
}

func (e AuditEntry) IsValid() bool {
	if e.UserID == "" || e.ResourceType == "" || e.Action == "" || e.Timestamp.IsZero() {
		return false
	}
	return true
}

// Standardized error types for audit operations
type AuditError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details,omitempty"`
}

func (e AuditError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("audit error %s: %s (%s)", e.Code, e.Message, e.Details)
	}
	return fmt.Sprintf("audit error %s: %s", e.Code, e.Message)
}

// Standard audit error codes
const (
	ERR_INVALID_EVENT    = "INVALID_EVENT"
	ERR_STORAGE_FAILED   = "STORAGE_FAILED"
	ERR_QUERY_FAILED     = "QUERY_FAILED"
	ERR_INTEGRITY_FAILED = "INTEGRITY_FAILED"
	ERR_PERMISSION_DENIED = "PERMISSION_DENIED"
)

// Helper for creating standardized errors
func NewAuditError(code, message, details string) *AuditError {
	return &AuditError{
		Code:    code,
		Message: message,
		Details: details,
	}
}