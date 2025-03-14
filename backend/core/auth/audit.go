package auth

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// AuditAction represents an auditable action
type AuditAction string

const (
	// Common audit actions
	AuditActionCreate    AuditAction = "create"
	AuditActionRead      AuditAction = "read"
	AuditActionUpdate    AuditAction = "update"
	AuditActionDelete    AuditAction = "delete"
	AuditActionLogin     AuditAction = "login"
	AuditActionLogout    AuditAction = "logout"
	AuditActionAccess    AuditAction = "access"
	AuditActionAttempt   AuditAction = "attempt"
	AuditActionApprove   AuditAction = "approve"
	AuditActionReject    AuditAction = "reject"
	AuditActionExecute   AuditAction = "execute"
	AuditActionDisable   AuditAction = "disable"
	AuditActionEnable    AuditAction = "enable"
	AuditActionAssociate AuditAction = "associate"
)

// AuditOutcome represents the outcome of an auditable action
type AuditOutcome string

const (
	// AuditOutcomeSuccess indicates the action was successful
	AuditOutcomeSuccess AuditOutcome = "success"

	// AuditOutcomeFailure indicates the action failed
	AuditOutcomeFailure AuditOutcome = "failure"

	// AuditOutcomeWarning indicates the action completed with warnings
	AuditOutcomeWarning AuditOutcome = "warning"

	// AuditOutcomeInfo indicates the action is informational
	AuditOutcomeInfo AuditOutcome = "info"
)

// AuditSeverity represents the severity of an audit entry
type AuditSeverity string

const (
	// AuditSeverityCritical indicates a critical severity
	AuditSeverityCritical AuditSeverity = "critical"

	// AuditSeverityHigh indicates a high severity
	AuditSeverityHigh AuditSeverity = "high"

	// AuditSeverityMedium indicates a medium severity
	AuditSeverityMedium AuditSeverity = "medium"

	// AuditSeverityLow indicates a low severity
	AuditSeverityLow AuditSeverity = "low"

	// AuditSeverityInfo indicates an informational severity
	AuditSeverityInfo AuditSeverity = "info"
)

// AuditEntry represents an entry in the audit log
type AuditEntry struct {
	// ID is the unique identifier for this audit entry
	ID string `json:"id"`

	// Timestamp is when the entry was created
	Timestamp time.Time `json:"timestamp"`

	// Action is the action being audited
	Action string `json:"action"`

	// Outcome is the outcome of the action
	Outcome AuditOutcome `json:"outcome,omitempty"`

	// Severity is the severity of the action
	Severity AuditSeverity `json:"severity,omitempty"`

	// Resource is the type of resource being audited
	Resource string `json:"resource"`

	// ResourceID is the ID of the resource being audited
	ResourceID string `json:"resourceId"`

	// Description is a human-readable description of the action
	Description string `json:"description,omitempty"`

	// UserID is the ID of the user who performed the action
	UserID string `json:"userId,omitempty"`

	// UserName is the name of the user who performed the action
	UserName string `json:"userName,omitempty"`

	// TenantID is the ID of the tenant where the action occurred
	TenantID string `json:"tenantId,omitempty"`

	// ClientIP is the IP address of the client
	ClientIP string `json:"clientIp,omitempty"`

	// UserAgent is the user agent of the client
	UserAgent string `json:"userAgent,omitempty"`

	// RequestID is the ID of the request
	RequestID string `json:"requestId,omitempty"`

	// Metadata contains additional metadata
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// AuditLogService defines the interface for audit logging services
type AuditLogService interface {
	// Log logs an audit entry
	Log(entry AuditEntry) error

	// Get gets an audit entry by ID
	Get(id string) (*AuditEntry, error)

	// Search searches for audit entries
	Search(filter map[string]interface{}, limit, offset int) ([]*AuditEntry, error)

	// Count counts audit entries based on filter
	Count(filter map[string]interface{}) (int, error)

	// Export exports audit entries to a specified format
	Export(filter map[string]interface{}, format string) ([]byte, error)
}

// InMemoryAuditLogService is an in-memory implementation of AuditLogService
type InMemoryAuditLogService struct {
	entries      []*AuditEntry
	entriesMutex sync.RWMutex
	idCounter    int
}

// NewInMemoryAuditLogService creates a new in-memory audit log service
func NewInMemoryAuditLogService() *InMemoryAuditLogService {
	return &InMemoryAuditLogService{
		entries:   make([]*AuditEntry, 0),
		idCounter: 0,
	}
}

// Log logs an audit entry
func (s *InMemoryAuditLogService) Log(entry AuditEntry) error {
	s.entriesMutex.Lock()
	defer s.entriesMutex.Unlock()

	// Set defaults if not provided
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}
	if entry.ID == "" {
		s.idCounter++
		entry.ID = fmt.Sprintf("audit-%d", s.idCounter)
	}
	if entry.Outcome == "" {
		entry.Outcome = AuditOutcomeSuccess
	}
	if entry.Severity == "" {
		entry.Severity = AuditSeverityInfo
	}

	// Store a copy of the entry
	entryCopy := entry
	s.entries = append(s.entries, &entryCopy)

	return nil
}

// Get gets an audit entry by ID
func (s *InMemoryAuditLogService) Get(id string) (*AuditEntry, error) {
	s.entriesMutex.RLock()
	defer s.entriesMutex.RUnlock()

	for _, entry := range s.entries {
		if entry.ID == id {
			// Return a copy of the entry
			entryCopy := *entry
			return &entryCopy, nil
		}
	}

	return nil, fmt.Errorf("audit entry not found: %s", id)
}

// Search searches for audit entries
func (s *InMemoryAuditLogService) Search(filter map[string]interface{}, limit, offset int) ([]*AuditEntry, error) {
	s.entriesMutex.RLock()
	defer s.entriesMutex.RUnlock()

	// Apply filter
	filtered := make([]*AuditEntry, 0)
	for _, entry := range s.entries {
		if matchesFilter(entry, filter) {
			filtered = append(filtered, entry)
		}
	}

	// Sort by timestamp (descending)
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].Timestamp.After(filtered[j].Timestamp)
	})

	// Apply pagination
	if offset >= len(filtered) {
		return []*AuditEntry{}, nil
	}
	end := offset + limit
	if end > len(filtered) {
		end = len(filtered)
	}
	result := filtered[offset:end]

	// Return copies of the entries
	copies := make([]*AuditEntry, len(result))
	for i, entry := range result {
		entryCopy := *entry
		copies[i] = &entryCopy
	}

	return copies, nil
}

// Count counts audit entries based on filter
func (s *InMemoryAuditLogService) Count(filter map[string]interface{}) (int, error) {
	s.entriesMutex.RLock()
	defer s.entriesMutex.RUnlock()

	count := 0
	for _, entry := range s.entries {
		if matchesFilter(entry, filter) {
			count++
		}
	}

	return count, nil
}

// Export exports audit entries to a specified format
func (s *InMemoryAuditLogService) Export(filter map[string]interface{}, format string) ([]byte, error) {
	// Get all matching entries
	entries, err := s.Search(filter, len(s.entries), 0)
	if err != nil {
		return nil, err
	}

	// In a real implementation, this would format the entries
	// For now, just return a simple message
	return []byte(fmt.Sprintf("Exported %d audit entries in %s format", len(entries), format)), nil
}

// matchesFilter checks if an entry matches a filter
func matchesFilter(entry *AuditEntry, filter map[string]interface{}) bool {
	for key, value := range filter {
		switch key {
		case "action":
			if entry.Action != value.(string) {
				return false
			}
		case "resource":
			if entry.Resource != value.(string) {
				return false
			}
		case "resourceId":
			if entry.ResourceID != value.(string) {
				return false
			}
		case "userId":
			if entry.UserID != value.(string) {
				return false
			}
		case "tenantId":
			if entry.TenantID != value.(string) {
				return false
			}
		case "outcome":
			if entry.Outcome != value.(AuditOutcome) {
				return false
			}
		case "severity":
			if entry.Severity != value.(AuditSeverity) {
				return false
			}
		case "fromTimestamp":
			if entry.Timestamp.Before(value.(time.Time)) {
				return false
			}
		case "toTimestamp":
			if entry.Timestamp.After(value.(time.Time)) {
				return false
			}
		}
	}
	return true
}
