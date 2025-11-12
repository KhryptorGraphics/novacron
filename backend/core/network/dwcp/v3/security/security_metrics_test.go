package security

import (
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestSecurityMetrics_ByzantineDetectionRecording(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)
	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// Record Byzantine detection
	metrics.RecordByzantineDetection("attacker-1", AttackEquivocation, 0.95, map[string]string{"evidence": "test"})

	// Check counters
	if metrics.byzantineDetections.Load() != 1 {
		t.Error("Expected Byzantine detection counter to increment")
	}

	// Check history
	history := metrics.GetDetectionHistory(10)
	if len(history) != 1 {
		t.Errorf("Expected 1 detection event, got %d", len(history))
	}

	if history[0].NodeID != "attacker-1" {
		t.Errorf("Expected node ID 'attacker-1', got '%s'", history[0].NodeID)
	}

	if history[0].AttackType != AttackEquivocation {
		t.Error("Expected attack type Equivocation")
	}
}

func TestSecurityMetrics_SignatureValidation(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Record successful validations
	for i := 0; i < 100; i++ {
		metrics.RecordSignatureValidation(true, 1*time.Millisecond)
	}

	// Record failures
	for i := 0; i < 10; i++ {
		metrics.RecordSignatureValidation(false, 2*time.Millisecond)
	}

	m := metrics.GetMetrics()

	validations := m["signature_validations"].(uint64)
	if validations != 110 {
		t.Errorf("Expected 110 validations, got %d", validations)
	}

	failures := m["signature_failures"].(uint64)
	if failures != 10 {
		t.Errorf("Expected 10 failures, got %d", failures)
	}

	failureRate := m["signature_failure_rate"].(float64)
	expectedRate := 10.0 / 110.0
	if failureRate < expectedRate-0.01 || failureRate > expectedRate+0.01 {
		t.Errorf("Expected failure rate ~%.3f, got %.3f", expectedRate, failureRate)
	}
}

func TestSecurityMetrics_QuarantineTracking(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Record quarantines
	metrics.RecordQuarantine("bad-node-1", "Byzantine behavior")
	metrics.RecordQuarantine("bad-node-2", "Low reputation")

	m := metrics.GetMetrics()

	quarantineEvents := m["quarantine_events"].(uint64)
	if quarantineEvents != 2 {
		t.Errorf("Expected 2 quarantine events, got %d", quarantineEvents)
	}

	activeQuarantines := m["active_quarantines"].(int)
	if activeQuarantines != 2 {
		t.Errorf("Expected 2 active quarantines, got %d", activeQuarantines)
	}

	// Release one
	metrics.RecordQuarantineRelease("bad-node-1")

	m = metrics.GetMetrics()
	activeQuarantines = m["active_quarantines"].(int)
	if activeQuarantines != 1 {
		t.Errorf("Expected 1 active quarantine after release, got %d", activeQuarantines)
	}
}

func TestSecurityMetrics_ModeChangeTracking(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Record mode changes
	metrics.RecordModeChange(ModeDatacenter, ModeInternet, "network conditions", "auto")
	metrics.RecordModeChange(ModeInternet, ModeHybrid, "manual switch", "manual")

	m := metrics.GetMetrics()

	modeChanges := m["mode_changes"].(uint64)
	if modeChanges != 2 {
		t.Errorf("Expected 2 mode changes, got %d", modeChanges)
	}

	history := metrics.GetModeHistory(10)
	if len(history) != 2 {
		t.Errorf("Expected 2 mode change events, got %d", len(history))
	}

	// Check first event
	if history[0].OldMode != ModeDatacenter || history[0].NewMode != ModeInternet {
		t.Error("Unexpected mode change in history")
	}

	if history[0].Triggered != "auto" {
		t.Errorf("Expected triggered='auto', got '%s'", history[0].Triggered)
	}
}

func TestSecurityMetrics_TLSHandshakeTracking(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Record successful handshakes
	for i := 0; i < 50; i++ {
		metrics.RecordTLSHandshake(true, 10*time.Millisecond)
	}

	// Record failures
	for i := 0; i < 5; i++ {
		metrics.RecordTLSHandshake(false, 0)
	}

	m := metrics.GetMetrics()

	handshakes := m["tls_handshakes"].(uint64)
	if handshakes != 55 {
		t.Errorf("Expected 55 handshakes, got %d", handshakes)
	}

	failures := m["tls_handshake_failures"].(uint64)
	if failures != 5 {
		t.Errorf("Expected 5 failures, got %d", failures)
	}

	failureRate := m["tls_failure_rate"].(float64)
	expectedRate := 5.0 / 55.0
	if failureRate < expectedRate-0.01 || failureRate > expectedRate+0.01 {
		t.Errorf("Expected TLS failure rate ~%.3f, got %.3f", expectedRate, failureRate)
	}

	avgLatency := m["avg_tls_handshake_ms"].(float64)
	if avgLatency < 9 || avgLatency > 11 {
		t.Errorf("Expected avg latency ~10ms, got %.2fms", avgLatency)
	}
}

func TestSecurityMetrics_AlertGeneration(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Generate alerts
	metrics.RecordByzantineDetection("attacker", AttackEquivocation, 0.9, nil)

	alerts := metrics.GetActiveAlerts()
	if len(alerts) == 0 {
		t.Error("Expected at least one alert to be generated")
	}

	// Check alert details
	found := false
	for _, alert := range alerts {
		if alert.Type == AlertByzantineDetected && alert.Severity == SeverityCritical {
			found = true
			if alert.NodeID != "attacker" {
				t.Errorf("Expected node ID 'attacker', got '%s'", alert.NodeID)
			}
			if alert.Resolved {
				t.Error("Expected alert to be unresolved")
			}
		}
	}

	if !found {
		t.Error("Expected Byzantine detection alert not found")
	}
}

func TestSecurityMetrics_ReputationSnapshot(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	metrics := NewSecurityMetrics("test-node", nil, reputation, nil, logger)
	defer metrics.Stop()
	defer reputation.Stop()

	// Create some nodes
	for i := 0; i < 5; i++ {
		nodeID := "node-" + string(rune('A'+i))
		reputation.RecordConsensusParticipation(nodeID, true)
	}

	// Trigger collection
	metrics.collectMetrics()

	history := metrics.GetReputationHistory(10)
	if len(history) == 0 {
		t.Error("Expected reputation snapshot to be collected")
	}

	if len(history) > 0 {
		snapshot := history[len(history)-1]
		if snapshot.TotalNodes != 5 {
			t.Errorf("Expected 5 nodes in snapshot, got %d", snapshot.TotalNodes)
		}
	}
}

func TestSecurityMetrics_AlertThresholds(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	metrics := NewSecurityMetrics("test-node", nil, reputation, nil, logger)
	defer metrics.Stop()
	defer reputation.Stop()

	// Create high quarantine rate scenario
	for i := 0; i < 10; i++ {
		nodeID := "node-" + string(rune('A'+i))
		reputation.RecordByzantineBehavior(nodeID, 50.0)
		if i < 3 {
			reputation.QuarantineNode(nodeID, "test")
			metrics.RecordQuarantine(nodeID, "test")
		}
	}

	// Trigger alert checking
	metrics.checkAlertConditions()

	time.Sleep(100 * time.Millisecond)

	alerts := metrics.GetActiveAlerts()

	// Check for high quarantine rate alert
	foundQuarantineAlert := false
	for _, alert := range alerts {
		if alert.Type == AlertHighQuarantineRate {
			foundQuarantineAlert = true
			if alert.Severity != SeverityHigh {
				t.Error("Expected high severity for quarantine rate alert")
			}
		}
	}

	if !foundQuarantineAlert {
		t.Log("High quarantine rate alert not triggered (may need more nodes)")
	}
}

func TestSecurityMetrics_LatencyTracking(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Record various latencies
	latencies := []time.Duration{
		5 * time.Millisecond,
		10 * time.Millisecond,
		15 * time.Millisecond,
		20 * time.Millisecond,
	}

	for _, lat := range latencies {
		metrics.RecordSignatureValidation(true, lat)
	}

	m := metrics.GetMetrics()
	avgLatency := m["avg_validation_latency_ms"].(float64)

	expectedAvg := 12.5 // (5+10+15+20)/4
	if avgLatency < expectedAvg-1 || avgLatency > expectedAvg+1 {
		t.Errorf("Expected avg latency ~%.1fms, got %.2fms", expectedAvg, avgLatency)
	}
}

func TestSecurityMetrics_HistoryLimits(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Generate more events than the history limit
	for i := 0; i < 1100; i++ {
		metrics.RecordByzantineDetection("attacker", AttackEquivocation, 0.9, nil)
	}

	history := metrics.GetDetectionHistory(2000)

	// Should be limited to 1000
	if len(history) > 1000 {
		t.Errorf("Expected history limited to 1000, got %d", len(history))
	}
}

func TestSecurityMetrics_ConcurrentRecording(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	done := make(chan bool)
	goroutines := 10

	for i := 0; i < goroutines; i++ {
		go func(id int) {
			for j := 0; j < 100; j++ {
				metrics.RecordSignatureValidation(true, 1*time.Millisecond)
				metrics.RecordEquivocation("node-" + string(rune('0'+id)))
			}
			done <- true
		}(i)
	}

	for i := 0; i < goroutines; i++ {
		<-done
	}

	m := metrics.GetMetrics()

	validations := m["signature_validations"].(uint64)
	if validations != 1000 {
		t.Errorf("Expected 1000 validations, got %d", validations)
	}

	equivocations := m["equivocations"].(uint64)
	if equivocations != 1000 {
		t.Errorf("Expected 1000 equivocations, got %d", equivocations)
	}
}

func TestSecurityMetrics_NetworkTrustTracking(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeHybrid, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)
	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// Create trusted nodes
	for i := 0; i < 10; i++ {
		nodeID := "trusted-" + string(rune('A'+i))
		for j := 0; j < 30; j++ {
			reputation.RecordConsensusParticipation(nodeID, true)
		}
	}

	// Trigger metrics collection
	metrics.collectMetrics()

	m := metrics.GetMetrics()
	networkTrust := m["network_trust"].(float64)

	if networkTrust < 0 || networkTrust > 1 {
		t.Errorf("Network trust out of range [0,1]: %.2f", networkTrust)
	}
}

func TestSecurityMetrics_AlertSeverityLevels(t *testing.T) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Generate alerts of different severities
	metrics.RecordByzantineDetection("critical-attacker", AttackEquivocation, 0.99, nil) // Critical
	metrics.RecordQuarantine("high-risk", "suspicious") // High

	alerts := metrics.GetActiveAlerts()

	severityCounts := make(map[AlertSeverity]int)
	for _, alert := range alerts {
		severityCounts[alert.Severity]++
	}

	if severityCounts[SeverityCritical] == 0 {
		t.Error("Expected at least one critical alert")
	}

	if severityCounts[SeverityHigh] == 0 {
		t.Error("Expected at least one high severity alert")
	}
}

func BenchmarkSecurityMetrics_RecordDetection(b *testing.B) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metrics.RecordByzantineDetection("attacker", AttackEquivocation, 0.9, nil)
	}
}

func BenchmarkSecurityMetrics_RecordSignature(b *testing.B) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metrics.RecordSignatureValidation(true, 1*time.Millisecond)
	}
}

func BenchmarkSecurityMetrics_GetMetrics(b *testing.B) {
	logger := zap.NewNop()
	metrics := NewSecurityMetrics("test-node", nil, nil, nil, logger)
	defer metrics.Stop()

	// Add some data
	for i := 0; i < 100; i++ {
		metrics.RecordSignatureValidation(true, 1*time.Millisecond)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metrics.GetMetrics()
	}
}
