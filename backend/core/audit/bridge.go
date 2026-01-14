// Package audit provides simple bridge functionality for compatibility
package audit

import (
	"context"
	"time"
)

// Simple bridge functions that use the global audit logger

// InitializeBridge initializes the audit system with a logger
func InitializeBridge(logger AuditLogger) {
	SetAuditLogger(logger)
}

// GetBridgeAuditLogger returns the global audit logger for compatibility
func GetBridgeAuditLogger() AuditLogger {
	return GetGlobalAuditLogger()
}

// GetBridgeAuthService returns a simple auth service implementation
func GetBridgeAuthService() AuthAuditService {
	return &SimpleAuthService{}
}

// SimpleAuthService implements AuthAuditService using the global audit logger
type SimpleAuthService struct{}

func (s *SimpleAuthService) LogAccess(entry *AuditEntry) error {
	event := ConvertAuditEntryToAuditEvent(*entry)
	return LogEvent(context.Background(), &event)
}

func (s *SimpleAuthService) GetAuditTrail(userID, tenantID, resourceType, resourceID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error) {
	filter := &AuditFilter{
		UserID:    userID,
		TenantID:  tenantID,
		Resource:  resourceType,
		StartTime: &startTime,
		EndTime:   &endTime,
		Limit:     limit,
		Offset:    offset,
	}

	events, err := QueryEvents(context.Background(), filter)
	if err != nil {
		return nil, err
	}

	entries := make([]*AuditEntry, len(events))
	for i, e := range events {
		entry := ConvertAuditEventToAuditEntry(*e)
		entries[i] = &entry
	}

	return entries, nil
}

func (s *SimpleAuthService) GetUserActions(userID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error) {
	return s.GetAuditTrail(userID, "", "", "", startTime, endTime, limit, offset)
}

func (s *SimpleAuthService) GetResourceActions(resourceType, resourceID string, startTime, endTime time.Time, limit, offset int) ([]*AuditEntry, error) {
	return s.GetAuditTrail("", "", resourceType, resourceID, startTime, endTime, limit, offset)
}

func (s *SimpleAuthService) Log(entry AuditEntry) error {
	return s.LogAccess(&entry)
}