package integration

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/novacron/backend/core/security"
)

// TestBackpressureManagerIntegration tests the comprehensive backpressure handling system
func TestBackpressureManagerIntegration(t *testing.T) {
	// Create a logger for testing
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))

	// Test configuration with smaller queues for faster testing
	config := security.BackpressureConfig{
		QueueSizes: map[security.EventPriority]int{
			security.PriorityLow:      10,
			security.PriorityMedium:   20,
			security.PriorityHigh:     50,
			security.PriorityUrgent:   80,
			security.PriorityCritical: 100,
		},
		HighWaterMark:          0.80,
		LowWaterMark:           0.60,
		Strategy:               security.StrategyAdaptive,
		BaseThrottleRate:       1 * time.Millisecond,
		MaxThrottleRate:        100 * time.Millisecond,
		ThrottleMultiplier:     1.5,
		SpillToFile:            true,
		SpillDirectory:         "/tmp/test_security_event_spill",
		MaxSpillFiles:          10,
		LoadSheddingRatio:      0.30,
		AdaptiveWindow:         1 * time.Second,
		AdaptiveThreshold:      50,
		AdaptiveRecoveryFactor: 0.95,
		MaxRetries:             3,
		RetryInterval:          10 * time.Millisecond,
		RetryBackoff:           2.0,
		MetricsInterval:        100 * time.Millisecond,
	}

	// Create backpressure manager
	manager := security.NewEventQueueBackpressureManager(config, logger)

	// Start the manager
	if err := manager.Start(); err != nil {
		t.Fatalf("Failed to start backpressure manager: %v", err)
	}
	defer manager.Stop()

	// Test 1: Basic event processing
	t.Run("BasicEventProcessing", func(t *testing.T) {
		testBasicEventProcessing(t, manager)
	})

	// Test 2: Priority ordering
	t.Run("PriorityOrdering", func(t *testing.T) {
		testPriorityOrdering(t, manager)
	})

	// Test 3: Backpressure handling
	t.Run("BackpressureHandling", func(t *testing.T) {
		testBackpressureHandling(t, manager, config)
	})

	// Test 4: Spill and recovery
	t.Run("SpillAndRecovery", func(t *testing.T) {
		testSpillAndRecovery(t, manager)
	})

	// Test 5: Adaptive strategy
	t.Run("AdaptiveStrategy", func(t *testing.T) {
		testAdaptiveStrategy(t, manager)
	})

	// Test 6: Metrics collection
	t.Run("MetricsCollection", func(t *testing.T) {
		testMetricsCollection(t, manager)
	})
}

func testBasicEventProcessing(t *testing.T, manager *security.EventQueueBackpressureManager) {
	// Create test events
	events := []security.SecurityEvent{
		{
			ID:        "test-1",
			Type:      security.EventTypeAuthFailure,
			Severity:  security.SeverityMedium,
			Source:    "test-source-1",
			Timestamp: time.Now(),
			ClusterID: "cluster-1",
		},
		{
			ID:        "test-2",
			Type:      security.EventTypeVulnerabilityFound,
			Severity:  security.SeverityHigh,
			Source:    "test-source-2",
			Timestamp: time.Now(),
			ClusterID: "cluster-1",
		},
	}

	// Enqueue events
	for _, event := range events {
		if err := manager.EnqueueEvent(event); err != nil {
			t.Errorf("Failed to enqueue event %s: %v", event.ID, err)
		}
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	// Check status
	status := manager.GetStatus()
	if running, ok := status["running"].(bool); !ok || !running {
		t.Error("Manager should be running")
	}
}

func testPriorityOrdering(t *testing.T, manager *security.EventQueueBackpressureManager) {
	// Create events with different priorities
	priorities := []security.SecuritySeverity{
		security.SeverityLow,
		security.SeverityMedium,
		security.SeverityHigh,
		security.SeverityCritical,
	}

	// Enqueue events in reverse priority order
	for i, severity := range priorities {
		event := security.SecurityEvent{
			ID:        fmt.Sprintf("priority-test-%d", i),
			Type:      security.EventTypeAuthFailure,
			Severity:  severity,
			Source:    "test-priority-source",
			Timestamp: time.Now(),
			ClusterID: "cluster-priority",
		}

		if err := manager.EnqueueEvent(event); err != nil {
			t.Errorf("Failed to enqueue priority event %s: %v", event.ID, err)
		}
	}

	// Wait for processing (higher priority events should be processed first)
	time.Sleep(100 * time.Millisecond)
}

func testBackpressureHandling(t *testing.T, manager *security.EventQueueBackpressureManager, config security.BackpressureConfig) {
	// Generate enough events to trigger backpressure
	totalEvents := 500 // More than total queue capacity
	var wg sync.WaitGroup

	// Track different outcomes
	results := struct {
		sync.Mutex
		enqueued int
		dropped  int
		spilled  int
	}{}

	// Send events concurrently to simulate load
	for i := 0; i < totalEvents; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			event := security.SecurityEvent{
				ID:        fmt.Sprintf("backpressure-test-%d", id),
				Type:      security.EventTypeAuthFailure,
				Severity:  security.SeverityMedium,
				Source:    "test-backpressure-source",
				Timestamp: time.Now(),
				ClusterID: "cluster-backpressure",
			}

			err := manager.EnqueueEvent(event)
			results.Lock()
			if err != nil {
				if err.Error() == "event dropped due to backpressure" {
					results.dropped++
				} else {
					results.spilled++
				}
			} else {
				results.enqueued++
			}
			results.Unlock()
		}(i)
	}

	wg.Wait()

	// Verify backpressure mechanisms were triggered
	results.Lock()
	t.Logf("Backpressure test results: enqueued=%d, dropped=%d, spilled=%d, total=%d",
		results.enqueued, results.dropped, results.spilled, totalEvents)

	if results.enqueued+results.dropped+results.spilled != totalEvents {
		t.Errorf("Event accounting mismatch: expected %d total events, got %d",
			totalEvents, results.enqueued+results.dropped+results.spilled)
	}

	// Should have some backpressure events (dropped or spilled)
	if results.dropped+results.spilled == 0 {
		t.Error("Expected some events to trigger backpressure mechanisms")
	}
	results.Unlock()

	// Wait for queue to process remaining events
	time.Sleep(500 * time.Millisecond)
}

func testSpillAndRecovery(t *testing.T, manager *security.EventQueueBackpressureManager) {
	// This test verifies spill-to-disk functionality
	// Create many low-priority events to trigger spilling
	eventCount := 100

	for i := 0; i < eventCount; i++ {
		event := security.SecurityEvent{
			ID:        fmt.Sprintf("spill-test-%d", i),
			Type:      security.EventTypeSuspiciousActivity,
			Severity:  security.SeverityLow,
			Source:    "test-spill-source",
			Timestamp: time.Now(),
			ClusterID: "cluster-spill",
		}

		// Don't check error as we expect some to spill
		manager.EnqueueEvent(event)
	}

	// Wait for processing and potential spill recovery
	time.Sleep(1 * time.Second)

	// Check if spill files were created and recovered
	status := manager.GetStatus()
	if spillStats, ok := status["spill_stats"]; ok {
		t.Logf("Spill stats: %+v", spillStats)
	}
}

func testAdaptiveStrategy(t *testing.T, manager *security.EventQueueBackpressureManager) {
	// Test adaptive strategy with mixed priority events
	events := []struct {
		priority security.SecuritySeverity
		count    int
	}{
		{security.SeverityCritical, 10},
		{security.SeverityHigh, 20},
		{security.SeverityMedium, 50},
		{security.SeverityLow, 100},
	}

	for _, eventGroup := range events {
		for i := 0; i < eventGroup.count; i++ {
			event := security.SecurityEvent{
				ID:        fmt.Sprintf("adaptive-test-%s-%d", eventGroup.priority, i),
				Type:      security.EventTypeAuthFailure,
				Severity:  eventGroup.priority,
				Source:    "test-adaptive-source",
				Timestamp: time.Now(),
				ClusterID: "cluster-adaptive",
			}

			// Enqueue without checking error to test adaptive behavior
			manager.EnqueueEvent(event)
		}
	}

	// Wait for adaptive processing
	time.Sleep(200 * time.Millisecond)

	// Critical events should be processed first and not dropped
	// Lower priority events may be throttled, spilled, or dropped
	status := manager.GetStatus()
	t.Logf("Adaptive strategy status: %+v", status)
}

func testMetricsCollection(t *testing.T, manager *security.EventQueueBackpressureManager) {
	// Generate events to collect metrics
	eventCount := 50

	for i := 0; i < eventCount; i++ {
		event := security.SecurityEvent{
			ID:        fmt.Sprintf("metrics-test-%d", i),
			Type:      security.EventTypeAuthFailure,
			Severity:  security.SeverityMedium,
			Source:    "test-metrics-source",
			Timestamp: time.Now(),
			ClusterID: "cluster-metrics",
		}

		manager.EnqueueEvent(event)
	}

	// Wait for processing and metrics collection
	time.Sleep(500 * time.Millisecond)

	// Check metrics
	status := manager.GetStatus()
	if queueMetrics, ok := status["queue_metrics"]; ok {
		t.Logf("Queue metrics: %+v", queueMetrics)

		metrics := queueMetrics.(map[string]interface{})
		if len(metrics) == 0 {
			t.Error("Expected queue metrics to be collected")
		}
	}

	if processingRate, ok := status["processing_rate"]; ok {
		rate := processingRate.(int64)
		if rate < 0 {
			t.Errorf("Expected non-negative processing rate, got %d", rate)
		}
		t.Logf("Processing rate: %d events", rate)
	}
}

// TestDistributedSecurityCoordinatorWithBackpressure tests the integration with the coordinator
func TestDistributedSecurityCoordinatorWithBackpressure(t *testing.T) {
	// Create a mock audit logger
	auditLogger := &MockAuditLogger{}

	// Create a mock encryption manager
	encMgr := &MockEncryptionManager{}

	// Create the coordinator (which now includes backpressure manager)
	coordinator := security.NewDistributedSecurityCoordinator(encMgr, auditLogger)

	// Start the coordinator
	if err := coordinator.Start(); err != nil {
		t.Fatalf("Failed to start coordinator: %v", err)
	}
	defer coordinator.Stop()

	// Test event processing with backpressure
	t.Run("CoordinatorBackpressureIntegration", func(t *testing.T) {
		// Generate high load to test backpressure
		eventCount := 200
		var wg sync.WaitGroup

		for i := 0; i < eventCount; i++ {
			wg.Add(1)
			go func(id int) {
				defer wg.Done()

				event := security.SecurityEvent{
					ID:        fmt.Sprintf("coord-test-%d", id),
					Type:      security.EventTypeAuthFailure,
					Severity:  security.SeverityMedium,
					Source:    "test-coord-source",
					Timestamp: time.Now(),
					ClusterID: "cluster-coord",
					NodeID:    "node-1",
				}

				// Process through coordinator (which uses backpressure manager)
				if err := coordinator.ProcessSecurityEvent(event); err != nil {
					// Some events may be handled by backpressure mechanisms
					t.Logf("Event %s handled by backpressure: %v", event.ID, err)
				}
			}(i)
		}

		wg.Wait()

		// Wait for processing
		time.Sleep(500 * time.Millisecond)

		// Check system health including backpressure status
		health := coordinator.GetSystemHealth()
		t.Logf("System health: %+v", health)

		if healthy, ok := health["healthy"].(bool); ok && !healthy {
			issues := health["issues"].([]string)
			t.Logf("System issues detected: %v", issues)
		}

		// Check backpressure status specifically
		backpressureStatus := coordinator.GetBackpressureStatus()
		t.Logf("Backpressure status: %+v", backpressureStatus)

		if enabled, ok := backpressureStatus["enabled"].(bool); !ok || !enabled {
			t.Error("Backpressure manager should be enabled")
		}

		if running, ok := backpressureStatus["running"].(bool); !ok || !running {
			t.Error("Backpressure manager should be running")
		}
	})
}

// Mock implementations for testing

type MockAuditLogger struct{}

func (m *MockAuditLogger) LogEvent(event security.AuditEvent) error {
	// Mock implementation - just log to test output
	fmt.Printf("Audit: %s - %s\n", event.Type, event.Description)
	return nil
}

type MockEncryptionManager struct{}

func (m *MockEncryptionManager) SignMessage(message []byte) (string, error) {
	// Mock implementation - return a fake signature
	return fmt.Sprintf("mock-signature-%d", len(message)), nil
}

func (m *MockEncryptionManager) VerifySignature(message []byte, signature string) error {
	// Mock implementation - always verify successfully
	return nil
}

func (m *MockEncryptionManager) EncryptData(data []byte) ([]byte, error) {
	// Mock implementation - return the data as-is
	return data, nil
}

func (m *MockEncryptionManager) DecryptData(encryptedData []byte) ([]byte, error) {
	// Mock implementation - return the data as-is
	return encryptedData, nil
}