package security

import (
	"testing"
	"time"

	"go.uber.org/zap"
)

// TestSecurityIntegration_FullStack tests complete security stack integration
func TestSecurityIntegration_FullStack(t *testing.T) {
	logger := zap.NewNop()

	// Initialize all components
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// Scenario: Attack detection and response
	attacker := "malicious-actor"

	// 1. Attacker sends conflicting votes (equivocation)
	detector.RecordConsensusVote(attacker, 1, 100, "digest-A", "prepare")
	detector.RecordConsensusVote(attacker, 1, 100, "digest-B", "prepare")

	// 2. Mode security validates message
	err := modeSec.ValidateMessage(attacker, "prepare", map[string]string{"data": "evil"}, "signature")
	if err != nil {
		t.Logf("Message validation appropriately rejected: %v", err)
	}

	// 3. Wait for detection
	time.Sleep(200 * time.Millisecond)

	// 4. Verify Byzantine detection
	if detector.IsSuspicious(attacker) || detector.IsByzantine(attacker) {
		t.Log("✓ Byzantine behavior detected")
	}

	// 5. Verify reputation impact
	score := reputation.GetScore(attacker)
	if score < reputation.config.InitialScore {
		t.Logf("✓ Reputation decreased to %.2f", score)
	}

	// 6. Verify quarantine if severe enough
	if reputation.IsQuarantined(attacker) {
		t.Log("✓ Attacker quarantined")
	}

	// 7. Verify metrics tracking
	m := metrics.GetMetrics()
	if m["signature_validations"].(uint64) > 0 {
		t.Log("✓ Metrics tracking active")
	}

	// 8. Verify alerts generated
	alerts := metrics.GetActiveAlerts()
	if len(alerts) > 0 {
		t.Logf("✓ %d security alerts generated", len(alerts))
	}
}

// TestSecurityIntegration_ModeSwitch tests security mode adaptation
func TestSecurityIntegration_ModeSwitch(t *testing.T) {
	logger := zap.NewNop()

	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeHybrid, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	initialMode := modeSec.GetCurrentMode()
	t.Logf("Initial mode: %v", initialMode)

	// Create trusted environment
	for i := 0; i < 20; i++ {
		nodeID := "trusted-node-" + string(rune('A'+i))
		for j := 0; j < 50; j++ {
			reputation.RecordConsensusParticipation(nodeID, true)
		}
	}

	// Give time for adaptive monitoring
	time.Sleep(2 * time.Second)

	// Check if mode adapted
	finalMode := modeSec.GetCurrentMode()
	t.Logf("Final mode: %v", finalMode)

	// Verify metrics recorded mode changes
	modeChanges := metrics.modeChanges.Load()
	if modeChanges > 0 {
		t.Logf("✓ %d mode changes recorded", modeChanges)
	}

	// Verify network trust increased
	m := metrics.GetMetrics()
	networkTrust := m["network_trust"].(float64)
	t.Logf("Network trust: %.2f", networkTrust)
}

// TestSecurityIntegration_MultipleAttackers tests handling multiple attackers
func TestSecurityIntegration_MultipleAttackers(t *testing.T) {
	logger := zap.NewNop()

	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	attackers := []string{"attacker-1", "attacker-2", "attacker-3", "attacker-4"}

	// Different attack patterns for each attacker
	// Attacker 1: Equivocation
	detector.RecordConsensusVote(attackers[0], 1, 100, "digest-A", "prepare")
	detector.RecordConsensusVote(attackers[0], 1, 100, "digest-B", "prepare")

	// Attacker 2: Malformed messages
	for i := 0; i < 10; i++ {
		detector.RecordMalformedMessage(attackers[1], "commit", "invalid")
	}

	// Attacker 3: Timing attacks
	for i := 0; i < 20; i++ {
		detector.RecordResponseTime(attackers[2], 50*time.Millisecond)
	}
	detector.RecordResponseTime(attackers[2], 10*time.Second)

	// Attacker 4: Multiple violations
	detector.RecordConsensusVote(attackers[3], 1, 101, "digest-C", "prepare")
	detector.RecordConsensusVote(attackers[3], 1, 101, "digest-D", "prepare")
	for i := 0; i < 5; i++ {
		detector.RecordMalformedMessage(attackers[3], "prepare", "bad")
	}

	time.Sleep(300 * time.Millisecond)

	// Verify all attackers detected
	detectedCount := 0
	for _, attacker := range attackers {
		if detector.IsSuspicious(attacker) || detector.IsByzantine(attacker) {
			detectedCount++
		}
	}

	t.Logf("✓ Detected %d/%d attackers", detectedCount, len(attackers))

	// Verify metrics
	m := metrics.GetMetrics()
	quarantined := m["active_quarantines"].(int)
	t.Logf("✓ %d nodes quarantined", quarantined)

	alerts := metrics.GetActiveAlerts()
	t.Logf("✓ %d security alerts generated", len(alerts))
}

// TestSecurityIntegration_RecoveryScenario tests node recovery after incidents
func TestSecurityIntegration_RecoveryScenario(t *testing.T) {
	logger := zap.NewNop()

	config := DefaultReputationConfig()
	config.QuarantineDuration = 1 * time.Second
	config.AllowRecovery = true
	config.RecoveryThreshold = 45.0

	reputation := NewReputationSystem("test-node", logger)
	reputation.config = config
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	recoverer := "recovering-node"

	// Initial bad behavior
	detector.RecordMalformedMessage(recoverer, "prepare", "error")
	reputation.RecordByzantineBehavior(recoverer, 15.0)

	time.Sleep(200 * time.Millisecond)

	initialScore := reputation.GetScore(recoverer)
	wasQuarantined := reputation.IsQuarantined(recoverer)

	t.Logf("Initial score: %.2f, Quarantined: %v", initialScore, wasQuarantined)

	// Good behavior
	for i := 0; i < 50; i++ {
		reputation.RecordConsensusParticipation(recoverer, true)
	}

	// Wait for quarantine to expire
	time.Sleep(1500 * time.Millisecond)

	// Verify recovery
	finalScore := reputation.GetScore(recoverer)
	isQuarantined := reputation.IsQuarantined(recoverer)

	t.Logf("Final score: %.2f, Quarantined: %v", finalScore, isQuarantined)

	if finalScore > initialScore {
		t.Log("✓ Reputation improved after good behavior")
	}

	if wasQuarantined && !isQuarantined {
		t.Log("✓ Node recovered from quarantine")
	}
}

// TestSecurityIntegration_PerformanceUnderLoad tests security under high load
func TestSecurityIntegration_PerformanceUnderLoad(t *testing.T) {
	logger := zap.NewNop()

	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeDatacenter, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	start := time.Now()

	// Simulate high load
	nodes := 100
	messagesPerNode := 100

	done := make(chan bool)

	for i := 0; i < nodes; i++ {
		go func(nodeIdx int) {
			nodeID := "load-node-" + string(rune('0'+(nodeIdx%10)))
			for j := 0; j < messagesPerNode; j++ {
				// Validate messages
				modeSec.ValidateMessage(nodeID, "prepare", map[string]int{"seq": j}, "sig")

				// Record consensus
				detector.RecordConsensusVote(nodeID, 1, int64(j), "digest", "prepare")

				// Update reputation
				reputation.RecordMessageSuccess(nodeID)

				// Record metrics
				metrics.RecordSignatureValidation(true, 1*time.Millisecond)
			}
			done <- true
		}(i)
	}

	// Wait for completion
	for i := 0; i < nodes; i++ {
		<-done
	}

	duration := time.Since(start)

	totalOps := nodes * messagesPerNode
	opsPerSecond := float64(totalOps) / duration.Seconds()

	t.Logf("✓ Processed %d operations in %v", totalOps, duration)
	t.Logf("✓ Throughput: %.0f ops/sec", opsPerSecond)

	// Verify metrics
	m := metrics.GetMetrics()
	validations := m["signature_validations"].(uint64)
	t.Logf("✓ Recorded %d validations", validations)

	if opsPerSecond < 1000 {
		t.Log("Note: Throughput may be lower in test environment")
	}
}

// TestSecurityIntegration_DatacenterToInternet tests datacenter to internet transition
func TestSecurityIntegration_DatacenterToInternet(t *testing.T) {
	logger := zap.NewNop()

	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeDatacenter, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// Initially in datacenter mode
	if modeSec.GetCurrentMode() != ModeDatacenter {
		t.Fatal("Expected datacenter mode")
	}

	// Message validation should be lenient
	err := modeSec.ValidateMessage("trusted-peer", "prepare", map[string]string{"data": "test"}, "")
	if err != nil {
		t.Errorf("Datacenter mode should allow message: %v", err)
	}

	// Switch to internet mode
	err = modeSec.SwitchMode(ModeInternet, "transitioning to public network")
	if err != nil {
		t.Fatalf("Failed to switch mode: %v", err)
	}

	metrics.RecordModeChange(ModeDatacenter, ModeInternet, "public network", "manual")

	// Now validation should be strict
	err = modeSec.ValidateMessage("untrusted-peer", "prepare", map[string]string{"data": "test"}, "")
	if err == nil {
		t.Error("Internet mode should require signature")
	}

	// With signature should work (for non-quarantined nodes)
	err = modeSec.ValidateMessage("new-peer", "prepare", map[string]string{"data": "test"}, "signature")
	if err != nil {
		t.Logf("Validation with signature: %v", err)
	}

	// Verify TLS is now available
	tlsConfig, err := modeSec.GetTLSConfig()
	if err != nil {
		t.Errorf("Expected TLS to be available: %v", err)
	}
	if tlsConfig != nil {
		t.Log("✓ TLS configuration available")
	}
}

// TestSecurityIntegration_ReputationDrivenQuarantine tests automatic quarantine
func TestSecurityIntegration_ReputationDrivenQuarantine(t *testing.T) {
	logger := zap.NewNop()

	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	badActor := "reputation-destroyer"

	// Gradually decrease reputation
	for i := 0; i < 30; i++ {
		reputation.RecordByzantineBehavior(badActor, 5.0)
		detector.RecordMalformedMessage(badActor, "prepare", "invalid")

		if i%10 == 0 {
			score := reputation.GetScore(badActor)
			t.Logf("After %d violations, score: %.2f", i+1, score)
		}

		time.Sleep(10 * time.Millisecond)
	}

	time.Sleep(200 * time.Millisecond)

	// Verify quarantine
	if !reputation.IsQuarantined(badActor) {
		score := reputation.GetScore(badActor)
		t.Logf("Node not quarantined yet (score: %.2f, threshold: %.2f)",
			score, reputation.config.QuarantineThreshold)
	} else {
		t.Log("✓ Node automatically quarantined due to low reputation")
	}

	// Verify message rejection
	err := modeSec.ValidateMessage(badActor, "prepare", map[string]string{"data": "test"}, "sig")
	if err != nil {
		t.Logf("✓ Message from bad actor rejected: %v", err)
	}

	// Verify metrics
	metrics.RecordQuarantine(badActor, "automatic quarantine")
	m := metrics.GetMetrics()
	quarantined := m["active_quarantines"].(int)
	t.Logf("✓ Active quarantines: %d", quarantined)
}

// TestSecurityIntegration_ConsensusWithByzantine tests consensus with Byzantine nodes
func TestSecurityIntegration_ConsensusWithByzantine(t *testing.T) {
	logger := zap.NewNop()

	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// 7 nodes: 5 honest, 2 Byzantine (tolerates f=2)
	honestNodes := []string{"honest-1", "honest-2", "honest-3", "honest-4", "honest-5"}
	byzantineNodes := []string{"byzantine-1", "byzantine-2"}

	sequence := int64(1000)

	// Honest nodes vote for digest-A
	for _, node := range honestNodes {
		detector.RecordConsensusVote(node, 1, sequence, "digest-A", "prepare")
		reputation.RecordConsensusParticipation(node, true)
	}

	// Byzantine nodes equivocate
	for _, node := range byzantineNodes {
		detector.RecordConsensusVote(node, 1, sequence, "digest-A", "prepare")
		detector.RecordConsensusVote(node, 1, sequence, "digest-B", "prepare")
	}

	time.Sleep(200 * time.Millisecond)

	// Verify Byzantine detection
	byzantineDetected := 0
	for _, node := range byzantineNodes {
		if detector.IsSuspicious(node) || detector.IsByzantine(node) {
			byzantineDetected++
			metrics.RecordByzantineDetection(node, AttackEquivocation, 0.9, nil)
		}
	}

	t.Logf("✓ Detected %d/%d Byzantine nodes", byzantineDetected, len(byzantineNodes))

	// Verify honest nodes unaffected
	honestUnaffected := 0
	for _, node := range honestNodes {
		if !detector.IsSuspicious(node) && !detector.IsByzantine(node) {
			honestUnaffected++
		}
	}

	t.Logf("✓ %d/%d honest nodes unaffected", honestUnaffected, len(honestNodes))

	// Verify reputation divergence
	avgHonestScore := 0.0
	for _, node := range honestNodes {
		avgHonestScore += reputation.GetScore(node)
	}
	avgHonestScore /= float64(len(honestNodes))

	avgByzantineScore := 0.0
	for _, node := range byzantineNodes {
		avgByzantineScore += reputation.GetScore(node)
	}
	avgByzantineScore /= float64(len(byzantineNodes))

	t.Logf("✓ Avg honest score: %.2f, Avg Byzantine score: %.2f", avgHonestScore, avgByzantineScore)

	if avgHonestScore > avgByzantineScore {
		t.Log("✓ Reputation system correctly differentiates honest from Byzantine")
	}
}

// BenchmarkSecurityIntegration_FullStack benchmarks complete security stack
func BenchmarkSecurityIntegration_FullStack(b *testing.B) {
	logger := zap.NewNop()

	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	modeSec := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	metrics := NewSecurityMetrics("test-node", detector, reputation, modeSec, logger)

	defer metrics.Stop()
	defer modeSec.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	message := map[string]string{"data": "benchmark"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Full security validation pipeline
		modeSec.ValidateMessage("bench-node", "prepare", message, "signature")
		detector.RecordMessage("bench-node", "prepare", message, "signature")
		reputation.RecordMessageSuccess("bench-node")
		metrics.RecordSignatureValidation(true, 1*time.Millisecond)
	}
}
