package auth

import (
	"time"
)

// AuditEntry represents an audit log entry
type AuditEntry struct {
	// ID is the unique identifier for this entry
	ID string `json:"id"`

	// UserID is the ID of the user who performed the action
	UserID string `json:"userId"`

	// TenantID is the ID of the tenant context
	TenantID string `json:"tenantId"`

	// ResourceType is the type of resource accessed
	ResourceType string `json:"resourceType"`

	// ResourceID is the ID of the resource accessed
	ResourceID string `json:"resourceId"`

	// Action is the action performed
	Action string `json:"action"`

	// Success indicates if the action was successful
	Success bool `json:"success"`

	// Reason provides the reason for the decision (especially for denials)
	Reason string `json:"reason"`

	// Timestamp is when the action was performed
	Timestamp time.Time `json:"timestamp"`

	// IPAddress is the IP address of the user
	IPAddress string `json:"ipAddress"`

	// UserAgent is the user agent of the client
	UserAgent string `json:"userAgent"`

	// AdditionalData contains additional information about the action
	AdditionalData map[string]interface{} `json:"additionalData,omitempty"`

	// Compatibility fields
	Resource    string `json:"resource,omitempty"`
	Description string `json:"description,omitempty"`
}

// AuditService provides operations for audit logging
type AuditService interface {
	// LogAccess logs an access attempt
	LogAccess(entry *AuditEntry) error

	// GetAuditTrail gets the audit trail for a user or resource
	GetAuditTrail(userID, tenantID, resourceType, resourceID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error)

	// GetUserActions gets all actions performed by a user
	GetUserActions(userID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error)

	// GetResourceActions gets all actions performed on a resource
	GetResourceActions(resourceType, resourceID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error)

	// Log method for compatibility
	Log(entry AuditEntry) error
}

// AuditLogService is an alias for AuditService for compatibility
type AuditLogService = AuditService

// InMemoryAuditService is an in-memory implementation of AuditService
type InMemoryAuditService struct {
	entries []*AuditEntry
}

// NewInMemoryAuditService creates a new in-memory audit service
func NewInMemoryAuditService() *InMemoryAuditService {
	return &InMemoryAuditService{
		entries: make([]*AuditEntry, 0),
	}
}

// LogAccess logs an access attempt
func (s *InMemoryAuditService) LogAccess(entry *AuditEntry) error {
	if entry.ID == "" {
		entry.ID = time.Now().Format(time.RFC3339Nano)
	}
	s.entries = append(s.entries, entry)
	return nil
}

// GetAuditTrail gets the audit trail for a user or resource
func (s *InMemoryAuditService) GetAuditTrail(userID, tenantID, resourceType, resourceID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error) {
	var results []*AuditEntry

	for _, entry := range s.entries {
		if (userID == "" || entry.UserID == userID) &&
			(tenantID == "" || entry.TenantID == tenantID) &&
			(resourceType == "" || entry.ResourceType == resourceType) &&
			(resourceID == "" || entry.ResourceID == resourceID) &&
			(startTime.IsZero() || !entry.Timestamp.Before(startTime)) &&
			(endTime.IsZero() || !entry.Timestamp.After(endTime)) {
			results = append(results, entry)
		}
	}

	// Apply pagination
	if offset >= len(results) {
		return []*AuditEntry{}, nil
	}

	end := offset + limit
	if end > len(results) || limit <= 0 {
		end = len(results)
	}

	return results[offset:end], nil
}

// GetUserActions gets all actions performed by a user
func (s *InMemoryAuditService) GetUserActions(userID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error) {
	return s.GetAuditTrail(userID, "", "", "", startTime, endTime, limit, offset)
}

// GetResourceActions gets all actions performed on a resource
func (s *InMemoryAuditService) GetResourceActions(resourceType, resourceID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error) {
	return s.GetAuditTrail("", "", resourceType, resourceID, startTime, endTime, limit, offset)
}

// Log logs an audit entry (compatibility method)
func (s *InMemoryAuditService) Log(entry AuditEntry) error {
	if entry.ID == "" {
		entry.ID = time.Now().Format(time.RFC3339Nano)
	}
	if entry.Timestamp.IsZero() {
		entry.Timestamp = time.Now()
	}
	// Map compatibility fields to proper fields
	if entry.Resource != "" && entry.ResourceType == "" {
		entry.ResourceType = entry.Resource
	}
	if entry.Description != "" && entry.Reason == "" {
		entry.Reason = entry.Description
	}
	return s.LogAccess(&entry)
}
