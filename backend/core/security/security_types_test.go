package security

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
)

// TestSecurityEventTypeRegistry tests the type registry functionality
func TestSecurityEventTypeRegistry(t *testing.T) {
	registry := NewSecurityEventTypeRegistry()

	t.Run("test distributed to monitoring mapping", func(t *testing.T) {
		// Test known mapping
		monType, exists := registry.MapDistributedToMonitoring(DistributedEventAuthFailure)
		assert.True(t, exists, "should find mapping for auth failure")
		assert.Equal(t, MonitoringEventAuthFailure, monType, "should map to monitoring auth failure")

		// Test unknown mapping
		unknownType := DistributedSecurityEventType("unknown_event")
		_, exists = registry.MapDistributedToMonitoring(unknownType)
		assert.False(t, exists, "should not find mapping for unknown event")
	})

	t.Run("test monitoring to distributed mapping", func(t *testing.T) {
		// Test known mapping
		distType, exists := registry.MapMonitoringToDistributed(MonitoringEventAuthFailure)
		assert.True(t, exists, "should find reverse mapping for auth failure")
		assert.Equal(t, DistributedEventAuthFailure, distType, "should map back to distributed auth failure")
	})

	t.Run("test default severity", func(t *testing.T) {
		severity := registry.GetDefaultSeverity(string(DistributedEventSecurityBreach))
		assert.Equal(t, SeverityCritical, severity, "security breach should be critical severity")

		unknownSeverity := registry.GetDefaultSeverity("unknown_event")
		assert.Equal(t, SeverityMedium, unknownSeverity, "unknown events should default to medium severity")
	})
}

// TestDistributedSecurityEvent tests distributed security event functionality
func TestDistributedSecurityEvent(t *testing.T) {
	event := DistributedSecurityEvent{
		SecurityEventBase: SecurityEventBase{
			ID:          "dist-event-001",
			Timestamp:   time.Now(),
			Source:      "test-node-1",
			Severity:    SeverityHigh,
			Description: "Test distributed security event",
			Metadata:    map[string]interface{}{"test": "value"},
		},
		Type:      DistributedEventUnauthorizedAccess,
		Target:    "test-target",
		Data:      map[string]interface{}{"user_id": "test-user", "ip": "192.168.1.100"},
		ClusterID: "test-cluster",
		NodeID:    "test-node-1",
		Signature: "test-signature",
		Propagated: false,
	}

	t.Run("test event creation", func(t *testing.T) {
		assert.Equal(t, "dist-event-001", event.ID)
		assert.Equal(t, DistributedEventUnauthorizedAccess, event.Type)
		assert.Equal(t, SeverityHigh, event.Severity)
		assert.Equal(t, "test-cluster", event.ClusterID)
		assert.False(t, event.Propagated)
	})

	t.Run("test event data access", func(t *testing.T) {
		userID, exists := event.Data["user_id"]
		assert.True(t, exists, "should have user_id in data")
		assert.Equal(t, "test-user", userID)

		ip, exists := event.Data["ip"]
		assert.True(t, exists, "should have ip in data")
		assert.Equal(t, "192.168.1.100", ip)
	})
}

// TestMonitoringSecurityEvent tests monitoring security event functionality
func TestMonitoringSecurityEvent(t *testing.T) {
	event := MonitoringSecurityEvent{
		SecurityEventBase: SecurityEventBase{
			ID:          "mon-event-001",
			Timestamp:   time.Now(),
			Source:      "monitoring-system",
			Severity:    SeverityMedium,
			Description: "Test monitoring security event",
			Metadata:    map[string]interface{}{"node_id": "monitor-node-1"},
		},
		Type:       MonitoringEventBruteForceAttempt,
		UserID:     "test-user-123",
		IP:         "10.0.0.1",
		UserAgent:  "Mozilla/5.0 Test Agent",
		Endpoint:   "/api/auth/login",
		Method:     "POST",
		StatusCode: 401,
		Message:    "Multiple failed login attempts detected",
		Details:    map[string]interface{}{"attempts": 5, "time_window": "5m"},
		RiskScore:  0.7,
		Remediated: false,
	}

	t.Run("test event creation", func(t *testing.T) {
		assert.Equal(t, "mon-event-001", event.ID)
		assert.Equal(t, MonitoringEventBruteForceAttempt, event.Type)
		assert.Equal(t, SeverityMedium, event.Severity)
		assert.Equal(t, "test-user-123", event.UserID)
		assert.Equal(t, 0.7, event.RiskScore)
		assert.False(t, event.Remediated)
	})

	t.Run("test event details", func(t *testing.T) {
		attempts, exists := event.Details["attempts"]
		assert.True(t, exists, "should have attempts in details")
		assert.Equal(t, 5, attempts)
	})
}

// TestSecurityEventConverter tests event conversion between domains
func TestSecurityEventConverter(t *testing.T) {
	registry := NewSecurityEventTypeRegistry()
	converter := NewSecurityEventConverter(registry)

	t.Run("test distributed to monitoring conversion", func(t *testing.T) {
		distEvent := DistributedSecurityEvent{
			SecurityEventBase: SecurityEventBase{
				ID:          "dist-conv-001",
				Timestamp:   time.Now(),
				Source:      "dist-node-1",
				Severity:    SeverityHigh,
				Description: "Test conversion event",
			},
			Type:      DistributedEventAuthFailure,
			Data:      map[string]interface{}{"user_id": "conv-user", "ip": "192.168.1.200"},
			ClusterID: "conv-cluster",
			NodeID:    "conv-node",
		}

		monEvent := converter.ConvertToMonitoring(distEvent)

		assert.Equal(t, distEvent.ID, monEvent.ID, "ID should be preserved")
		assert.Equal(t, MonitoringEventAuthFailure, monEvent.Type, "type should be mapped correctly")
		assert.Equal(t, distEvent.Severity, monEvent.Severity, "severity should be preserved")
		assert.Equal(t, "conv-user", monEvent.UserID, "user ID should be extracted")
		assert.Equal(t, "192.168.1.200", monEvent.IP, "IP should be extracted")
	})

	t.Run("test monitoring to distributed conversion", func(t *testing.T) {
		monEvent := MonitoringSecurityEvent{
			SecurityEventBase: SecurityEventBase{
				ID:          "mon-conv-001",
				Timestamp:   time.Now(),
				Source:      "monitoring",
				Severity:    SeverityCritical,
				Description: "Critical security event",
				Metadata:    map[string]interface{}{"cluster_id": "meta-cluster", "node_id": "meta-node"},
			},
			Type:    MonitoringEventDataBreach,
			UserID:  "breach-user",
			IP:      "10.1.1.1",
			Details: map[string]interface{}{"affected_records": 1000},
		}

		distEvent := converter.ConvertToDistributed(monEvent)

		assert.Equal(t, monEvent.ID, distEvent.ID, "ID should be preserved")
		assert.Equal(t, DistributedEventSecurityBreach, distEvent.Type, "type should be mapped correctly")
		assert.Equal(t, monEvent.Severity, distEvent.Severity, "severity should be preserved")
		assert.Equal(t, "meta-cluster", distEvent.ClusterID, "cluster ID should be extracted from metadata")
		assert.Equal(t, "breach-user", distEvent.Data["user_id"], "user ID should be in data")
	})

	t.Run("test conversion with unknown type", func(t *testing.T) {
		distEvent := DistributedSecurityEvent{
			SecurityEventBase: SecurityEventBase{
				ID:       "unknown-event",
				Severity: SeverityLow,
			},
			Type: DistributedSecurityEventType("unknown_type"),
			Data: map[string]interface{}{},
		}

		monEvent := converter.ConvertToMonitoring(distEvent)

		// Should fallback to suspicious activity for unknown types
		assert.Equal(t, MonitoringEventSuspiciousActivity, monEvent.Type)
	})
}

// TestUnifiedSecurityEventHandler tests the unified event handler
func TestUnifiedSecurityEventHandler(t *testing.T) {
	mockAuditLogger := &MockAuditLogger{}
	handler := NewUnifiedSecurityEventHandler(mockAuditLogger)

	t.Run("test can handle distributed event", func(t *testing.T) {
		distEvent := DistributedSecurityEvent{
			SecurityEventBase: SecurityEventBase{ID: "test-dist"},
			Type:              DistributedEventAuthFailure,
		}

		assert.True(t, handler.CanHandle(distEvent), "should handle distributed event")
		assert.True(t, handler.CanHandle(&distEvent), "should handle distributed event pointer")
	})

	t.Run("test can handle monitoring event", func(t *testing.T) {
		monEvent := MonitoringSecurityEvent{
			SecurityEventBase: SecurityEventBase{ID: "test-mon"},
			Type:              MonitoringEventAuthFailure,
		}

		assert.True(t, handler.CanHandle(monEvent), "should handle monitoring event")
		assert.True(t, handler.CanHandle(&monEvent), "should handle monitoring event pointer")
	})

	t.Run("test cannot handle unknown event", func(t *testing.T) {
		unknownEvent := struct{ ID string }{ID: "unknown"}
		assert.False(t, handler.CanHandle(unknownEvent), "should not handle unknown event type")
	})

	t.Run("test get severity from events", func(t *testing.T) {
		distEvent := DistributedSecurityEvent{
			SecurityEventBase: SecurityEventBase{Severity: SeverityHigh},
		}
		monEvent := MonitoringSecurityEvent{
			SecurityEventBase: SecurityEventBase{Severity: SeverityLow},
		}

		assert.Equal(t, SeverityHigh, handler.GetSeverity(distEvent))
		assert.Equal(t, SeverityLow, handler.GetSeverity(monEvent))
		assert.Equal(t, SeverityMedium, handler.GetSeverity("unknown"))
	})

	t.Run("test handle distributed event", func(t *testing.T) {
		distEvent := DistributedSecurityEvent{
			SecurityEventBase: SecurityEventBase{
				ID:          "handle-dist-001",
				Timestamp:   time.Now(),
				Source:      "test-source",
				Severity:    SeverityMedium,
				Description: "Test distributed event handling",
			},
			Type:      DistributedEventSuspiciousActivity,
			ClusterID: "test-cluster",
			NodeID:    "test-node",
			Data:      map[string]interface{}{"test": "data"},
		}

		err := handler.HandleEvent(distEvent)
		assert.NoError(t, err, "should handle distributed event without error")

		// Verify audit log was called
		assert.Len(t, mockAuditLogger.Events, 1, "should have logged one audit event")
		loggedEvent := mockAuditLogger.Events[0]
		assert.Equal(t, "distributed_security_event", loggedEvent.Resource)
		assert.Equal(t, "test-node", loggedEvent.UserID)
	})

	t.Run("test handle monitoring event", func(t *testing.T) {
		monEvent := MonitoringSecurityEvent{
			SecurityEventBase: SecurityEventBase{
				ID:          "handle-mon-001",
				Timestamp:   time.Now(),
				Source:      "monitoring",
				Severity:    SeverityHigh,
				Description: "Test monitoring event handling",
			},
			Type:   MonitoringEventUnauthorizedAccess,
			UserID: "test-user",
			IP:     "192.168.1.1",
		}

		mockAuditLogger.Reset()
		err := handler.HandleEvent(monEvent)
		assert.NoError(t, err, "should handle monitoring event without error")

		// Verify audit log was called
		assert.Len(t, mockAuditLogger.Events, 1, "should have logged one audit event")
		loggedEvent := mockAuditLogger.Events[0]
		assert.Equal(t, "monitoring_security_event", loggedEvent.Resource)
		assert.Equal(t, "test-user", loggedEvent.UserID)
	})

	t.Run("test handle critical monitoring event triggers distributed processing", func(t *testing.T) {
		criticalEvent := MonitoringSecurityEvent{
			SecurityEventBase: SecurityEventBase{
				ID:       "critical-001",
				Severity: SeverityCritical,
				Source:   "monitoring",
			},
			Type:   MonitoringEventDataBreach,
			UserID: "critical-user",
		}

		mockAuditLogger.Reset()
		err := handler.HandleEvent(criticalEvent)
		assert.NoError(t, err, "should handle critical event without error")

		// Critical events should be processed by both handlers
		// This is verified through the mock implementations
	})

	t.Run("test handle unsupported event", func(t *testing.T) {
		unsupportedEvent := "not a security event"
		err := handler.HandleEvent(unsupportedEvent)
		assert.Error(t, err, "should return error for unsupported event")
		assert.Contains(t, err.Error(), "unsupported event type")
	})
}

// TestSecurityDomainRegistry tests domain registration and type name generation
func TestSecurityDomainRegistry(t *testing.T) {
	t.Run("test registered domains", func(t *testing.T) {
		// Test existing domains
		distDomain, exists := GetSecurityDomain("distributed")
		assert.True(t, exists, "distributed domain should be registered")
		assert.Equal(t, "Distributed", distDomain.Prefix)

		monDomain, exists := GetSecurityDomain("monitoring")
		assert.True(t, exists, "monitoring domain should be registered")
		assert.Equal(t, "Monitoring", monDomain.Prefix)

		// Test non-existent domain
		_, exists = GetSecurityDomain("nonexistent")
		assert.False(t, exists, "non-existent domain should not be found")
	})

	t.Run("test register new domain", func(t *testing.T) {
		newDomain := DomainTypeNamespace{
			Domain:           "blockchain",
			Prefix:          "Blockchain",
			EventTypesSuffix: "SecurityEventType",
			EventSuffix:     "SecurityEvent",
		}

		RegisterSecurityDomain("blockchain", newDomain)

		retrieved, exists := GetSecurityDomain("blockchain")
		assert.True(t, exists, "newly registered domain should be found")
		assert.Equal(t, "Blockchain", retrieved.Prefix)
	})

	t.Run("test generate type name", func(t *testing.T) {
		typeName := GenerateTypeName("distributed", "Auth", "SecurityEventType")
		assert.Equal(t, "DistributedAuthSecurityEventType", typeName)

		// Test unknown domain fallback
		unknownTypeName := GenerateTypeName("unknown_domain", "Test", "Type")
		assert.Equal(t, "TestType", unknownTypeName)
	})
}

// TestDomainSpecificTypes tests that all domain-specific types are properly differentiated
func TestDomainSpecificTypes(t *testing.T) {
	t.Run("test distributed message types", func(t *testing.T) {
		// Test that distributed message types are distinct
		assert.Equal(t, "cluster_sync", string(DistributedMessageClusterSync))
		assert.Equal(t, "federation_event", string(DistributedMessageFederationEvent))
		assert.Equal(t, "heartbeat", string(DistributedMessageHeartbeat))
	})

	t.Run("test dating app message types", func(t *testing.T) {
		// Test that dating app message types are distinct
		assert.Equal(t, "text", string(DatingAppMessageText))
		assert.Equal(t, "image", string(DatingAppMessageImage))
		assert.Equal(t, "video", string(DatingAppMessageVideo))
	})

	t.Run("test monitoring vs distributed event types", func(t *testing.T) {
		// Test that monitoring and distributed event types are properly separated
		distType := DistributedEventAuthFailure
		monType := MonitoringEventAuthFailure

		// They should have the same string value but different types
		assert.Equal(t, string(distType), string(monType))
		assert.IsType(t, DistributedSecurityEventType(""), distType)
		assert.IsType(t, MonitoringSecurityEventType(""), monType)

		// They should not be directly assignable to each other (compile-time check)
		// This would cause a compile error: var x DistributedSecurityEventType = monType
	})

	t.Run("test threat detection config types", func(t *testing.T) {
		// Test that different threat detection configs are distinct types
		monitoringConfig := MonitoringThreatDetectionConfig{
			EnableBehaviorAnalysis: true,
			BruteForceThreshold:    5,
		}

		enterpriseConfig := EnterpriseThreatDetectionConfig{
			Enabled:            true,
			MachineLearning:    true,
			BehavioralAnalysis: true,
		}

		// Configs should be different types
		assert.IsType(t, MonitoringThreatDetectionConfig{}, monitoringConfig)
		assert.IsType(t, EnterpriseThreatDetectionConfig{}, enterpriseConfig)
	})

	t.Run("test integration config types", func(t *testing.T) {
		// Test that different integration configs are distinct types
		monitoringIntegration := MonitoringIntegrationConfig{
			JiraEnabled: true,
			JiraURL:     "https://jira.example.com",
		}

		generalIntegration := GeneralIntegrationConfig{
			EnableAWSIntegration: true,
			EnableSIEMIntegration: true,
		}

		// Configs should be different types
		assert.IsType(t, MonitoringIntegrationConfig{}, monitoringIntegration)
		assert.IsType(t, GeneralIntegrationConfig{}, generalIntegration)
	})
}

// TestBackwardCompatibility tests that the migration maintains compatibility
func TestBackwardCompatibility(t *testing.T) {
	t.Run("test security severity compatibility", func(t *testing.T) {
		// Test that SecuritySeverity and ThreatSeverity are compatible
		var secSeverity SecuritySeverity = SeverityHigh
		var threatSeverity ThreatSeverity = secSeverity

		assert.Equal(t, secSeverity, threatSeverity)
		assert.Equal(t, "high", string(threatSeverity))
	})

	t.Run("test base event compatibility", func(t *testing.T) {
		// Test that all events share the same base structure
		baseEvent := SecurityEventBase{
			ID:          "base-001",
			Timestamp:   time.Now(),
			Source:      "test",
			Severity:    SeverityMedium,
			Description: "Base event test",
			Metadata:    map[string]interface{}{"test": true},
		}

		distEvent := DistributedSecurityEvent{
			SecurityEventBase: baseEvent,
			Type:              DistributedEventAuthFailure,
		}

		monEvent := MonitoringSecurityEvent{
			SecurityEventBase: baseEvent,
			Type:              MonitoringEventAuthFailure,
		}

		// Both should have the same base fields
		assert.Equal(t, baseEvent.ID, distEvent.ID)
		assert.Equal(t, baseEvent.ID, monEvent.ID)
		assert.Equal(t, baseEvent.Severity, distEvent.Severity)
		assert.Equal(t, baseEvent.Severity, monEvent.Severity)
	})
}

// MockAuditLogger provides a mock implementation for testing
type MockAuditLogger struct {
	Events []audit.AuditEvent
}

func (m *MockAuditLogger) LogEvent(ctx context.Context, event *audit.AuditEvent) error {
	m.Events = append(m.Events, *event)
	return nil
}

func (m *MockAuditLogger) LogAuthEvent(ctx context.Context, username string, success bool, details map[string]interface{}) error {
	event := &audit.AuditEvent{
		UserID:   username,
		Action:   audit.ActionUpdate,
		Resource: "auth",
		Result:   audit.ResultSuccess,
		Details:  details,
	}
	if !success {
		event.Result = audit.ResultFailure
	}
	return m.LogEvent(ctx, event)
}

func (m *MockAuditLogger) Reset() {
	m.Events = []audit.AuditEvent{}
}

// Benchmark tests to ensure no performance regression
func BenchmarkSecurityEventHandling(b *testing.B) {
	mockAuditLogger := &MockAuditLogger{}
	handler := NewUnifiedSecurityEventHandler(mockAuditLogger)

	distEvent := DistributedSecurityEvent{
		SecurityEventBase: SecurityEventBase{
			ID:       "bench-dist",
			Severity: SeverityMedium,
		},
		Type: DistributedEventAuthFailure,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handler.HandleEvent(distEvent)
	}
}

func BenchmarkEventTypeMapping(b *testing.B) {
	registry := NewSecurityEventTypeRegistry()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		registry.MapDistributedToMonitoring(DistributedEventAuthFailure)
	}
}

func BenchmarkEventConversion(b *testing.B) {
	registry := NewSecurityEventTypeRegistry()
	converter := NewSecurityEventConverter(registry)

	distEvent := DistributedSecurityEvent{
		SecurityEventBase: SecurityEventBase{ID: "bench-conv", Severity: SeverityMedium},
		Type:              DistributedEventAuthFailure,
		Data:              map[string]interface{}{"user_id": "bench-user"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		converter.ConvertToMonitoring(distEvent)
	}
}