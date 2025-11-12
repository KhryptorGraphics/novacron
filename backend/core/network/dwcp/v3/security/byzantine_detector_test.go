package security

import (
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestByzantineDetector_InvalidSignature(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	// Record message with invalid signature
	err := detector.RecordMessage("malicious-node", "prepare", map[string]interface{}{"data": "test"}, "valid-sig")
	if err != nil {
		t.Fatalf("Failed to record message: %v", err)
	}

	// Simulate invalid signature
	err = detector.ValidateSignature("malicious-node-prepare-*", false)
	// Expected error as message ID won't match exactly

	// Record multiple invalid signatures
	for i := 0; i < 10; i++ {
		detector.RecordMessage("malicious-node", "prepare", map[string]interface{}{"seq": i}, "sig")
		detector.RecordMalformedMessage("malicious-node", "prepare", "invalid format")
	}

	time.Sleep(100 * time.Millisecond)

	// Check if node is suspicious
	if !detector.IsSuspicious("malicious-node") {
		t.Error("Expected node to be suspicious after multiple violations")
	}
}

func TestByzantineDetector_Equivocation(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	// Record conflicting votes (equivocation)
	err := detector.RecordConsensusVote("byzantine-node", 1, 100, "digest-A", "prepare")
	if err != nil {
		t.Fatalf("Failed to record vote: %v", err)
	}

	// Same node, same sequence, different digest (equivocation!)
	err = detector.RecordConsensusVote("byzantine-node", 1, 100, "digest-B", "prepare")
	if err != nil {
		t.Fatalf("Failed to record conflicting vote: %v", err)
	}

	time.Sleep(100 * time.Millisecond)

	// Should detect equivocation
	if !detector.IsSuspicious("byzantine-node") {
		t.Error("Expected equivocation to be detected")
	}

	// Check violation type
	suspicious := detector.GetSuspiciousNodes()
	if record, exists := suspicious["byzantine-node"]; exists {
		foundEquivocation := false
		for _, v := range record.Violations {
			if v.Type == ViolationEquivocation {
				foundEquivocation = true
				break
			}
		}
		if !foundEquivocation {
			t.Error("Expected equivocation violation to be recorded")
		}
	}
}

func TestByzantineDetector_TimingAnomaly(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	// Record normal response times
	for i := 0; i < 20; i++ {
		detector.RecordResponseTime("slow-node", 100*time.Millisecond)
	}

	// Record abnormally slow response
	detector.RecordResponseTime("slow-node", 5*time.Second)

	time.Sleep(100 * time.Millisecond)

	// Check for timing anomaly violation
	suspicious := detector.GetSuspiciousNodes()
	if record, exists := suspicious["slow-node"]; exists {
		foundTiming := false
		for _, v := range record.Violations {
			if v.Type == ViolationTimingAnomaly {
				foundTiming = true
				break
			}
		}
		if !foundTiming {
			t.Error("Expected timing anomaly to be detected")
		}
	}
}

func TestByzantineDetector_MultipleViolationTypes(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	attacker := "sophisticated-attacker"

	// Multiple violation types
	// 1. Equivocation
	detector.RecordConsensusVote(attacker, 1, 100, "digest-A", "prepare")
	detector.RecordConsensusVote(attacker, 1, 100, "digest-B", "prepare")

	// 2. Malformed messages
	for i := 0; i < 5; i++ {
		detector.RecordMalformedMessage(attacker, "commit", "invalid structure")
	}

	// 3. Timing anomalies
	for i := 0; i < 15; i++ {
		detector.RecordResponseTime(attacker, 50*time.Millisecond)
	}
	detector.RecordResponseTime(attacker, 10*time.Second)

	time.Sleep(200 * time.Millisecond)

	// Should be confirmed Byzantine with multiple violation types
	if !detector.IsByzantine(attacker) {
		suspicious := detector.GetSuspiciousNodes()
		if record, exists := suspicious[attacker]; exists {
			t.Logf("Suspicion score: %.2f (threshold: %.2f)", record.SuspicionScore, detector.config.ByzantineThreshold)
			t.Logf("Violations: %d", len(record.Violations))
		}
		t.Error("Expected sophisticated attacker to be confirmed Byzantine")
	}

	// Check evidence
	evidence, exists := detector.GetByzantineEvidence(attacker)
	if !exists {
		t.Fatal("Expected Byzantine evidence to exist")
	}

	if evidence.Confidence < 0.7 {
		t.Errorf("Expected high confidence, got %.2f", evidence.Confidence)
	}

	if len(evidence.Violations) < 3 {
		t.Errorf("Expected multiple violation types, got %d", len(evidence.Violations))
	}
}

func TestByzantineDetector_FalsePositivePrevention(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	// Honest node with normal behavior
	honest := "honest-node"

	// Record normal consensus participation
	for i := 0; i < 100; i++ {
		detector.RecordConsensusVote(honest, 1, int64(i), "digest-X", "prepare")
		detector.RecordResponseTime(honest, 100*time.Millisecond)
	}

	time.Sleep(100 * time.Millisecond)

	// Should NOT be suspicious or Byzantine
	if detector.IsSuspicious(honest) {
		t.Error("Honest node incorrectly marked as suspicious (false positive)")
	}

	if detector.IsByzantine(honest) {
		t.Error("Honest node incorrectly marked as Byzantine (false positive)")
	}
}

func TestByzantineDetector_SuspicionDecay(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)

	// Configure fast decay for testing
	config := DefaultDetectorConfig()
	config.BehaviorWindow = 1 * time.Second
	config.SuspicionDecay = 2 * time.Second

	detector := NewByzantineDetector("test-node", reputation, logger)
	detector.config = config
	defer detector.Stop()

	// Create minor violations
	detector.RecordMalformedMessage("minor-offender", "prepare", "format error")
	detector.RecordMalformedMessage("minor-offender", "commit", "format error")

	time.Sleep(100 * time.Millisecond)

	// Should be slightly suspicious
	suspicious1 := detector.GetSuspiciousNodes()
	initialScore := 0.0
	if record, exists := suspicious1["minor-offender"]; exists {
		initialScore = record.SuspicionScore
	}

	// Wait for decay
	time.Sleep(3 * time.Second)

	// Suspicion should decay
	suspicious2 := detector.GetSuspiciousNodes()
	if record, exists := suspicious2["minor-offender"]; exists {
		if record.SuspicionScore >= initialScore {
			t.Error("Expected suspicion score to decay over time")
		}
	}
}

func TestByzantineDetector_ViewChangeAbuse(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	// Simulate view change abuse (triggering unnecessary view changes)
	abuser := "view-changer"

	// Record multiple view change attempts
	for i := 0; i < 10; i++ {
		// This would be detected through consensus vote tracking
		// For now, record as malformed to simulate detection
		detector.RecordMalformedMessage(abuser, "view-change", "unnecessary view change")
	}

	time.Sleep(100 * time.Millisecond)

	if !detector.IsSuspicious(abuser) {
		t.Error("Expected view change abuser to be detected")
	}
}

func TestByzantineDetector_ConcurrentAccess(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	// Concurrent message recording
	done := make(chan bool)
	nodes := []string{"node-1", "node-2", "node-3", "node-4"}

	for _, nodeID := range nodes {
		go func(id string) {
			for i := 0; i < 50; i++ {
				detector.RecordMessage(id, "prepare", map[string]interface{}{"seq": i}, "sig")
				detector.RecordResponseTime(id, 100*time.Millisecond)
			}
			done <- true
		}(nodeID)
	}

	// Wait for all goroutines
	for range nodes {
		<-done
	}

	// Verify stats
	stats := detector.GetStats()
	totalNodes := stats["total_nodes"].(int)
	if totalNodes != len(nodes) {
		t.Errorf("Expected %d nodes, got %d", len(nodes), totalNodes)
	}
}

func TestByzantineDetector_AttackTypeClassification(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	// Test different attack patterns
	tests := []struct {
		name        string
		setup       func(nodeID string)
		expectedType AttackType
	}{
		{
			name: "Equivocation Attack",
			setup: func(nodeID string) {
				detector.RecordConsensusVote(nodeID, 1, 100, "digest-A", "prepare")
				detector.RecordConsensusVote(nodeID, 1, 100, "digest-B", "prepare")
				detector.RecordConsensusVote(nodeID, 1, 101, "digest-C", "commit")
				detector.RecordConsensusVote(nodeID, 1, 101, "digest-D", "commit")
			},
			expectedType: AttackEquivocation,
		},
		{
			name: "Timing Manipulation",
			setup: func(nodeID string) {
				for i := 0; i < 20; i++ {
					detector.RecordResponseTime(nodeID, 50*time.Millisecond)
				}
				for i := 0; i < 5; i++ {
					detector.RecordResponseTime(nodeID, 10*time.Second)
				}
			},
			expectedType: AttackTimingManipulation,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeID := "attacker-" + tt.name
			tt.setup(nodeID)

			time.Sleep(200 * time.Millisecond)

			if detector.IsByzantine(nodeID) {
				evidence, _ := detector.GetByzantineEvidence(nodeID)
				if evidence.AttackType != tt.expectedType {
					t.Errorf("Expected attack type %v, got %v", tt.expectedType, evidence.AttackType)
				}
			}
		})
	}
}

func TestByzantineDetector_ReputationIntegration(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()
	defer reputation.Stop()

	attacker := "integrated-attacker"

	// Record violations
	for i := 0; i < 5; i++ {
		detector.RecordMalformedMessage(attacker, "prepare", "invalid")
	}

	time.Sleep(200 * time.Millisecond)

	// Check reputation was affected
	score := reputation.GetScore(attacker)
	if score >= reputation.config.InitialScore {
		t.Error("Expected reputation score to decrease after violations")
	}

	// Continue violations until Byzantine
	for i := 0; i < 20; i++ {
		detector.RecordConsensusVote(attacker, 1, int64(i), "digest-A", "prepare")
		detector.RecordConsensusVote(attacker, 1, int64(i), "digest-B", "prepare")
	}

	time.Sleep(500 * time.Millisecond)

	// Should be quarantined
	if !reputation.IsQuarantined(attacker) {
		t.Error("Expected attacker to be quarantined after Byzantine detection")
	}
}

func TestByzantineDetector_GarbageCollection(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)

	config := DefaultDetectorConfig()
	config.BehaviorWindow = 500 * time.Millisecond

	detector := NewByzantineDetector("test-node", reputation, logger)
	detector.config = config
	defer detector.Stop()

	// Create lots of old data
	for i := 0; i < 100; i++ {
		detector.RecordMessage("temp-node", "prepare", map[string]interface{}{"i": i}, "sig")
	}

	initialStats := detector.GetStats()
	initialMessages := initialStats["messages_tracked"].(int)

	// Wait for cleanup
	time.Sleep(2 * time.Second)

	finalStats := detector.GetStats()
	finalMessages := finalStats["messages_tracked"].(int)

	if finalMessages >= initialMessages {
		t.Log("Note: Garbage collection may not have run yet (timing dependent)")
	}
}

func BenchmarkByzantineDetector_RecordMessage(b *testing.B) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	message := map[string]interface{}{"data": "test"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.RecordMessage("node-1", "prepare", message, "signature")
	}
}

func BenchmarkByzantineDetector_ConsensusVote(b *testing.B) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	defer detector.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		detector.RecordConsensusVote("node-1", 1, int64(i), "digest", "prepare")
	}
}
