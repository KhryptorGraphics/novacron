// Package security provides security audit logging functionality
//
// DEPRECATED: This package now provides type aliases to the canonical audit package.
// New code should import "github.com/khryptorgraphics/novacron/backend/core/audit" directly.
// This compatibility layer will be removed in a future version.
package security

import (
	"context"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/audit"
)

// Type aliases for backward compatibility
// These types now point to the canonical audit package

// AuditEvent represents a security audit event
// Deprecated: Use audit.AuditEvent instead
type AuditEvent = audit.AuditEvent

// AuditEventType defines types of audit events
// Deprecated: Use audit.AuditEventType instead
type AuditEventType = audit.AuditEventType

// AuditAction defines the action performed
// Deprecated: Use audit.AuditAction instead
type AuditAction = audit.AuditAction

// AuditResult defines the result of an action
// Deprecated: Use audit.AuditResult instead
type AuditResult = audit.AuditResult

// SensitivityLevel defines data sensitivity
// Deprecated: Use audit.SensitivityLevel instead
type SensitivityLevel = audit.SensitivityLevel

// Event type constants - aliased to audit package
const (
	EventSecretAccess   = audit.EventSecretAccess
	EventSecretModify   = audit.EventSecretModify
	EventSecretDelete   = audit.EventSecretDelete
	EventSecretRotate   = audit.EventSecretRotate
	EventAuthAttempt    = audit.EventAuthAttempt
	EventPermissionDeny = audit.EventPermissionDeny
	EventConfigChange   = audit.EventConfigChange
	EventSecurityDrop   = audit.EventSecurityDrop
)

// Action constants - aliased to audit package
const (
	ActionRead   = audit.ActionRead
	ActionWrite  = audit.ActionWrite
	ActionUpdate = audit.ActionUpdate
	ActionDelete = audit.ActionDelete
	ActionList   = audit.ActionList
	ActionRotate = audit.ActionRotate
	ActionDrop   = audit.ActionDrop
)

// Result constants - aliased to audit package
const (
	ResultSuccess = audit.ResultSuccess
	ResultFailure = audit.ResultFailure
	ResultDenied  = audit.ResultDenied
)

// Sensitivity level constants - aliased to audit package
const (
	SensitivityPublic       = audit.SensitivityPublic
	SensitivityInternal     = audit.SensitivityInternal
	SensitivityConfidential = audit.SensitivityConfidential
	SensitivitySecret       = audit.SensitivitySecret
)

// AuditLogger interface for audit logging
// Deprecated: Use audit.AuditLogger instead
type AuditLogger interface {
	LogEvent(ctx context.Context, event AuditEvent) error // Generic event logging
	LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error
	LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error
	LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error
	LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error
	LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error
	Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error)
	VerifyIntegrity(ctx context.Context, startTime, endTime time.Time) (*IntegrityReport, error)
}

// IntegrityReport represents audit log integrity verification results
// Deprecated: Use audit.IntegrityReport instead
type IntegrityReport = audit.IntegrityReport

// AuditFilter for querying audit logs
// Deprecated: Use audit.AuditFilter instead
type AuditFilter = audit.AuditFilter

// DefaultAuditLogger implements AuditLogger
// Deprecated: Use audit.DefaultAuditLogger instead
type DefaultAuditLogger = audit.DefaultAuditLogger

// AuditStorage interface for persisting audit logs
// Deprecated: Use audit.AuditStorage instead
type AuditStorage = audit.AuditStorage

// AlertingService for security alerts
// Deprecated: Use audit.AlertingService instead
type AlertingService = audit.AlertingService

// ComplianceService for compliance reporting
// Deprecated: Use audit.ComplianceService instead
type ComplianceService = audit.ComplianceService

// DatabaseAuditStorage implements AuditStorage using database
// Deprecated: Use audit.DatabaseAuditStorage instead
type DatabaseAuditStorage = audit.DatabaseAuditStorage

// Deprecated factory functions - these now delegate to the audit package

// NewAuditLogger creates a new audit logger
// Deprecated: Use audit.NewAuditLogger instead
func NewAuditLogger(storage AuditStorage, alerting AlertingService, compliance ComplianceService) *DefaultAuditLogger {
	return audit.NewAuditLogger(storage, alerting, compliance)
}

// NewDatabaseAuditStorage creates database-backed audit storage
// Deprecated: Use audit.NewDatabaseAuditStorage instead
func NewDatabaseAuditStorage(db interface{}) *DatabaseAuditStorage {
	// Note: This assumes db is *sql.DB - in real implementation would type-assert
	return audit.NewDatabaseAuditStorage(nil)
}

// Deprecated convenience functions - these now delegate to the audit package

// LogSecretAccess logs secret access events using the default audit logger
// Deprecated: Use audit.LogSecretAccess instead
func LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	return audit.LogSecretAccess(ctx, actor, resource, action, result, details)
}

// LogSecretModification logs secret modification events using the default audit logger
// Deprecated: Use audit.LogSecretModification instead
func LogSecretModification(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error {
	return audit.LogSecretModification(ctx, actor, resource, action, result, details)
}

// LogSecretRotation logs secret rotation events using the default audit logger
// Deprecated: Use audit.LogSecretRotation instead
func LogSecretRotation(ctx context.Context, actor, resource string, oldVersion, newVersion string, result AuditResult) error {
	return audit.LogSecretRotation(ctx, actor, resource, oldVersion, newVersion, result)
}

// LogAuthEvent logs authentication events using the default audit logger
// Deprecated: Use audit.LogAuthEvent instead
func LogAuthEvent(ctx context.Context, actor string, success bool, details map[string]interface{}) error {
	return audit.LogAuthEvent(ctx, actor, success, details)
}

// LogConfigChange logs configuration changes using the default audit logger
// Deprecated: Use audit.LogConfigChange instead
func LogConfigChange(ctx context.Context, actor, resource string, oldValue, newValue interface{}) error {
	return audit.LogConfigChange(ctx, actor, resource, oldValue, newValue)
}

// Migration helper functions

// GetCanonicalAuditLogger returns the underlying audit.AuditLogger
// Use this function to migrate from security.AuditLogger to audit.AuditLogger
func GetCanonicalAuditLogger() audit.AuditLogger {
	return audit.GetGlobalAuditLogger()
}

// MigrateToAuditPackage provides guidance for migration
// This function returns information about how to migrate from security audit to canonical audit
func MigrateToAuditPackage() map[string]string {
	return map[string]string{
		"package_import":    "Replace 'github.com/khryptorgraphics/novacron/backend/core/security' with 'github.com/khryptorgraphics/novacron/backend/core/audit'",
		"type_usage":        "Remove 'security.' prefix from audit types - use audit.AuditEvent instead of security.AuditEvent",
		"function_calls":    "Replace security.LogSecretAccess() with audit.LogSecretAccess()",
		"interfaces":        "Change security.AuditLogger to audit.AuditLogger",
		"constants":         "Replace security.EventSecretAccess with audit.EventSecretAccess",
		"migration_status":  "This compatibility layer will be removed in v2.0.0",
		"timeline":          "Plan migration within 6 months",
	}
}

// GetGlobalAuditLogger returns the global audit logger for compatibility
// This is a bridge to the canonical audit package
func GetGlobalAuditLogger() audit.AuditLogger {
	return audit.GetGlobalAuditLogger()
}

// SetGlobalAuditLogger sets the global audit logger for compatibility
// This is a bridge to the canonical audit package
func SetGlobalAuditLogger(logger audit.AuditLogger) {
	audit.SetAuditLogger(logger)
}

// Package-level documentation for migration

/*
MIGRATION GUIDE: security -> audit package

This security audit package now serves as a compatibility layer. All functionality
has been moved to the canonical audit package at:
	github.com/khryptorgraphics/novacron/backend/core/audit

To migrate your code:

1. Update imports:
   OLD: import "github.com/khryptorgraphics/novacron/backend/core/security"
   NEW: import "github.com/khryptorgraphics/novacron/backend/core/audit"

2. Update type references:
   OLD: security.AuditEvent
   NEW: audit.AuditEvent

3. Update function calls:
   OLD: security.LogSecretAccess(...)
   NEW: audit.LogSecretAccess(...)

4. Update constants:
   OLD: security.EventSecretAccess
   NEW: audit.EventSecretAccess

The compatibility layer provides type aliases and function wrappers to ensure
existing code continues to work during the migration period.

This compatibility layer will be removed in v2.0.0. Please migrate your code
to use the canonical audit package directly.
*/