package audit

import (
	"context"
	"testing"
	"time"
)

func TestAuditLogger(t *testing.T) {
	ctx := context.Background()

	t.Run("Log Event", func(t *testing.T) {
		al := NewAuditLogger(7*365*24*time.Hour, true, true)

		event := &AuditEvent{
			EventType: EventUserLogin,
			Severity:  SeverityInfo,
			Actor: Actor{
				Type: "user",
				ID:   "user-123",
				Name: "test-user",
			},
			Target: Target{
				Type: "system",
				ID:   "system-1",
				Name: "NovaCron",
			},
			Action:    "login",
			Result:    "success",
			IPAddress: "192.168.1.1",
			TenantID:  "tenant-1",
		}

		err := al.LogEvent(ctx, event)
		if err != nil {
			t.Fatalf("Failed to log event: %v", err)
		}

		if event.ID == "" {
			t.Error("Expected event ID to be generated")
		}

		if event.Hash == "" {
			t.Error("Expected event hash to be generated")
		}
	})

	t.Run("Search Events", func(t *testing.T) {
		al := NewAuditLogger(7*365*24*time.Hour, true, true)

		// Log multiple events
		for i := 0; i < 5; i++ {
			event := &AuditEvent{
				EventType: EventVMCreate,
				Severity:  SeverityInfo,
				Actor: Actor{
					Type: "user",
					ID:   "user-123",
					Name: "test-user",
				},
				Target: Target{
					Type: "vm",
					ID:   "vm-1",
					Name: "test-vm",
				},
				Action:   "create",
				Result:   "success",
				TenantID: "tenant-1",
			}
			al.LogEvent(ctx, event)
		}

		// Search by event type
		criteria := SearchCriteria{
			EventType: string(EventVMCreate),
		}

		results, err := al.SearchEvents(ctx, criteria)
		if err != nil {
			t.Fatalf("Failed to search events: %v", err)
		}

		if len(results) != 5 {
			t.Errorf("Expected 5 results, got %d", len(results))
		}
	})

	t.Run("Verify Integrity", func(t *testing.T) {
		al := NewAuditLogger(7*365*24*time.Hour, true, true)

		// Log events
		for i := 0; i < 3; i++ {
			event := &AuditEvent{
				EventType: EventConfigChange,
				Severity:  SeverityWarning,
				Actor: Actor{
					Type: "user",
					ID:   "user-123",
					Name: "admin",
				},
				Target: Target{
					Type: "config",
					ID:   "config-1",
					Name: "system-config",
				},
				Action: "update",
				Result: "success",
			}
			al.LogEvent(ctx, event)
		}

		// Verify integrity
		intact, tamperedEvents := al.VerifyIntegrity()
		if !intact {
			t.Errorf("Expected integrity to be intact, found tampered events: %v", tamperedEvents)
		}
	})

	t.Run("Forensic Analysis", func(t *testing.T) {
		al := NewAuditLogger(7*365*24*time.Hour, true, true)

		// Log various events
		events := []AuditEventType{
			EventUserLogin,
			EventAccessGranted,
			EventVMCreate,
			EventAccessDenied,
			EventComplianceViolation,
		}

		for _, eventType := range events {
			event := &AuditEvent{
				EventType: eventType,
				Severity:  SeverityInfo,
				Actor: Actor{
					Type: "user",
					ID:   "user-123",
					Name: "test-user",
				},
				Target: Target{
					Type: "resource",
					ID:   "res-1",
				},
				Action: "test",
				Result: "success",
			}
			al.LogEvent(ctx, event)
		}

		// Perform forensic analysis
		startTime := time.Now().Add(-1 * time.Hour)
		endTime := time.Now()

		analysis, err := al.PerformForensicAnalysis(ctx, "security-investigation", startTime, endTime)
		if err != nil {
			t.Fatalf("Failed to perform forensic analysis: %v", err)
		}

		if analysis.EventsAnalyzed == 0 {
			t.Error("Expected events to be analyzed")
		}

		if len(analysis.Timeline) == 0 {
			t.Error("Expected timeline to be generated")
		}
	})

	t.Run("Anomaly Detection", func(t *testing.T) {
		al := NewAuditLogger(7*365*24*time.Hour, true, true)

		// Simulate multiple failed login attempts
		for i := 0; i < 6; i++ {
			event := &AuditEvent{
				EventType: EventUserLogin,
				Severity:  SeverityWarning,
				Actor: Actor{
					Type: "user",
					ID:   "user-123",
					Name: "attacker",
				},
				Target: Target{
					Type: "system",
					ID:   "system-1",
				},
				Action: "login",
				Result: "failure",
			}
			al.LogEvent(ctx, event)
			time.Sleep(10 * time.Millisecond)
		}

		// Anomalies should be detected
		// In production, this would trigger alerts
	})

	t.Run("Audit Metrics", func(t *testing.T) {
		al := NewAuditLogger(7*365*24*time.Hour, true, true)

		// Log events
		al.LogEvent(ctx, &AuditEvent{
			EventType: EventUserLogin,
			Severity:  SeverityInfo,
			Actor:     Actor{Type: "user", ID: "user-1"},
			Target:    Target{Type: "system", ID: "sys-1"},
			Action:    "login",
			Result:    "success",
		})

		metrics := al.GetMetrics()

		if metrics.TotalEvents == 0 {
			t.Error("Expected total events to be tracked")
		}

		if metrics.EventsByType[EventUserLogin] == 0 {
			t.Error("Expected event type to be tracked")
		}
	})
}

func BenchmarkAuditLogging(b *testing.B) {
	ctx := context.Background()
	al := NewAuditLogger(7*365*24*time.Hour, true, true)

	event := &AuditEvent{
		EventType: EventAPICall,
		Severity:  SeverityInfo,
		Actor: Actor{
			Type: "service",
			ID:   "api-server",
		},
		Target: Target{
			Type: "endpoint",
			ID:   "/api/v1/vms",
		},
		Action: "GET",
		Result: "success",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		al.LogEvent(ctx, event)
	}
}

func BenchmarkAuditSearch(b *testing.B) {
	ctx := context.Background()
	al := NewAuditLogger(7*365*24*time.Hour, true, true)

	// Populate with events
	for i := 0; i < 1000; i++ {
		al.LogEvent(ctx, &AuditEvent{
			EventType: EventVMCreate,
			Severity:  SeverityInfo,
			Actor:     Actor{Type: "user", ID: "user-1"},
			Target:    Target{Type: "vm", ID: "vm-1"},
			Action:    "create",
			Result:    "success",
		})
	}

	criteria := SearchCriteria{
		EventType: string(EventVMCreate),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		al.SearchEvents(ctx, criteria)
	}
}
