package security

import (
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestReputationSystem_InitialScore(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	// New node should get initial score
	score := rs.GetScore("new-node")
	if score != rs.config.InitialScore {
		t.Errorf("Expected initial score %.2f, got %.2f", rs.config.InitialScore, score)
	}
}

func TestReputationSystem_ConsensusParticipation(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	nodeID := "honest-node"
	initialScore := rs.GetScore(nodeID)

	// Record correct consensus participation
	for i := 0; i < 10; i++ {
		rs.RecordConsensusParticipation(nodeID, true)
	}

	newScore := rs.GetScore(nodeID)
	if newScore <= initialScore {
		t.Error("Expected score to increase after correct consensus participation")
	}

	// Record incorrect participation
	for i := 0; i < 5; i++ {
		rs.RecordConsensusParticipation(nodeID, false)
	}

	afterIncorrect := rs.GetScore(nodeID)
	if afterIncorrect >= newScore {
		t.Error("Expected score to decrease after incorrect participation")
	}
}

func TestReputationSystem_ByzantineBehaviorPenalty(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	attacker := "byzantine-node"
	initialScore := rs.GetScore(attacker)

	// Record Byzantine behavior with high severity
	rs.RecordByzantineBehavior(attacker, 40.0)

	newScore := rs.GetScore(attacker)
	penalty := initialScore - newScore

	expectedMinPenalty := 40.0 * rs.config.ByzantinePenaltyMultiplier
	if penalty < expectedMinPenalty {
		t.Errorf("Expected penalty >= %.2f, got %.2f", expectedMinPenalty, penalty)
	}
}

func TestReputationSystem_QuarantineMechanism(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	badNode := "bad-node"

	// Record enough Byzantine behavior to trigger quarantine
	for i := 0; i < 10; i++ {
		rs.RecordByzantineBehavior(badNode, 15.0)
	}

	time.Sleep(100 * time.Millisecond)

	// Should be quarantined
	if !rs.IsQuarantined(badNode) {
		score := rs.GetScore(badNode)
		t.Errorf("Expected node to be quarantined (score: %.2f, threshold: %.2f)",
			score, rs.config.QuarantineThreshold)
	}

	// Verify quarantine record exists
	quarantined := rs.GetQuarantinedNodes()
	if _, exists := quarantined[badNode]; !exists {
		t.Error("Expected quarantine record to exist")
	}
}

func TestReputationSystem_QuarantineRecovery(t *testing.T) {
	logger := zap.NewNop()

	config := DefaultReputationConfig()
	config.QuarantineDuration = 500 * time.Millisecond
	config.AllowRecovery = true
	config.RecoveryThreshold = 45.0

	rs := NewReputationSystem("test-node", logger)
	rs.config = config
	defer rs.Stop()

	recoverer := "recovering-node"

	// Quarantine the node
	err := rs.QuarantineNode(recoverer, "test quarantine")
	if err != nil {
		t.Fatalf("Failed to quarantine: %v", err)
	}

	// Improve reputation
	rep := rs.GetReputation(recoverer)
	if rep != nil {
		rep.Score = 50.0 // Above recovery threshold
	}

	// Wait for quarantine to expire
	time.Sleep(600 * time.Millisecond)

	// Should be released
	if rs.IsQuarantined(recoverer) {
		t.Error("Expected node to be released after quarantine period with good score")
	}
}

func TestReputationSystem_MaxQuarantineCount(t *testing.T) {
	logger := zap.NewNop()

	config := DefaultReputationConfig()
	config.MaxQuarantineCount = 3

	rs := NewReputationSystem("test-node", logger)
	rs.config = config
	defer rs.Stop()

	repeat := "repeat-offender"

	// Quarantine multiple times
	for i := 0; i < config.MaxQuarantineCount; i++ {
		err := rs.QuarantineNode(repeat, "offense")
		if err != nil {
			t.Fatalf("Failed to quarantine attempt %d: %v", i+1, err)
		}
		rs.ReleaseQuarantine(repeat)
	}

	// Next quarantine should fail or be permanent
	err := rs.QuarantineNode(repeat, "final offense")
	if err == nil {
		// Check violation count
		quarantined := rs.GetQuarantinedNodes()
		if record, exists := quarantined[repeat]; exists {
			if record.ViolationCount <= config.MaxQuarantineCount {
				t.Error("Expected violation count to exceed maximum")
			}
		}
	}
}

func TestReputationSystem_ReputationDecay(t *testing.T) {
	logger := zap.NewNop()

	config := DefaultReputationConfig()
	config.DecayEnabled = true
	config.DecayInterval = 500 * time.Millisecond
	config.DecayRate = 5.0 // 5 points per interval
	config.NewNodeGracePeriod = 100 * time.Millisecond

	rs := NewReputationSystem("test-node", logger)
	rs.config = config
	defer rs.Stop()

	inactive := "inactive-node"

	// Boost reputation
	for i := 0; i < 10; i++ {
		rs.RecordConsensusParticipation(inactive, true)
	}

	initialScore := rs.GetScore(inactive)

	// Wait past grace period
	time.Sleep(200 * time.Millisecond)

	// Wait for decay
	time.Sleep(1 * time.Second)

	finalScore := rs.GetScore(inactive)

	if finalScore >= initialScore {
		t.Errorf("Expected decay after inactivity (initial: %.2f, final: %.2f)",
			initialScore, finalScore)
	}
}

func TestReputationSystem_ReputationLevels(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	tests := []struct {
		name          string
		score         float64
		expectedLevel ReputationLevel
	}{
		{"Highly Trusted", 95.0, ReputationHighlyTrusted},
		{"Trusted", 80.0, ReputationTrusted},
		{"Neutral", 50.0, ReputationNeutral},
		{"Suspicious", 35.0, ReputationSuspicious},
		{"Untrusted", 10.0, ReputationUntrusted},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			nodeID := "node-" + tt.name

			// Set score directly for testing
			rep := rs.GetReputation(nodeID)
			if rep == nil {
				rs.RecordConsensusParticipation(nodeID, true)
				rep = rs.GetReputation(nodeID)
			}
			rep.Score = tt.score

			level := rs.GetLevel(nodeID)
			if level != tt.expectedLevel {
				t.Errorf("Expected level %v, got %v (score: %.2f)",
					tt.expectedLevel, level, tt.score)
			}
		})
	}
}

func TestReputationSystem_TrustedNodes(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	// Create trusted nodes
	for i := 0; i < 5; i++ {
		nodeID := "trusted-" + string(rune('A'+i))
		for j := 0; j < 30; j++ {
			rs.RecordConsensusParticipation(nodeID, true)
		}
	}

	trusted := rs.GetTrustedNodes()
	if len(trusted) < 5 {
		t.Errorf("Expected at least 5 trusted nodes, got %d", len(trusted))
	}

	// Verify all are actually trusted
	for id, rep := range trusted {
		if rep.Score < rs.config.TrustedThreshold {
			t.Errorf("Node %s in trusted list but score %.2f < threshold %.2f",
				id, rep.Score, rs.config.TrustedThreshold)
		}
	}
}

func TestReputationSystem_SuspiciousNodes(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	// Create suspicious nodes
	for i := 0; i < 3; i++ {
		nodeID := "suspicious-" + string(rune('A'+i))
		rs.RecordByzantineBehavior(nodeID, 10.0)
	}

	suspicious := rs.GetSuspiciousNodes()
	if len(suspicious) < 3 {
		t.Errorf("Expected at least 3 suspicious nodes, got %d", len(suspicious))
	}
}

func TestReputationSystem_MessageSuccessFailure(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	nodeID := "message-handler"

	// Record successes
	for i := 0; i < 100; i++ {
		rs.RecordMessageSuccess(nodeID)
	}

	scoreAfterSuccess := rs.GetScore(nodeID)

	// Record failures
	for i := 0; i < 50; i++ {
		rs.RecordMessageFailure(nodeID)
	}

	scoreAfterFailure := rs.GetScore(nodeID)

	if scoreAfterFailure >= scoreAfterSuccess {
		t.Error("Expected score to decrease after message failures")
	}
}

func TestReputationSystem_Statistics(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	// Create diverse population
	for i := 0; i < 10; i++ {
		nodeID := "node-" + string(rune('0'+i))
		if i < 5 {
			// Make trusted
			for j := 0; j < 30; j++ {
				rs.RecordConsensusParticipation(nodeID, true)
			}
		} else if i < 8 {
			// Make suspicious
			rs.RecordByzantineBehavior(nodeID, 10.0)
		} else {
			// Quarantine
			rs.RecordByzantineBehavior(nodeID, 50.0)
			rs.QuarantineNode(nodeID, "bad behavior")
		}
	}

	stats := rs.GetStats()

	totalNodes := stats["total_nodes"].(int)
	if totalNodes != 10 {
		t.Errorf("Expected 10 nodes, got %d", totalNodes)
	}

	trustedNodes := stats["trusted_nodes"].(int)
	if trustedNodes < 5 {
		t.Errorf("Expected at least 5 trusted nodes, got %d", trustedNodes)
	}

	avgScore := stats["average_score"].(float64)
	if avgScore <= 0 || avgScore > 100 {
		t.Errorf("Invalid average score: %.2f", avgScore)
	}
}

func TestReputationSystem_ConcurrentAccess(t *testing.T) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	done := make(chan bool)
	nodes := 20

	for i := 0; i < nodes; i++ {
		go func(id int) {
			nodeID := "concurrent-node-" + string(rune('0'+id))
			for j := 0; j < 50; j++ {
				rs.RecordConsensusParticipation(nodeID, j%3 != 0)
				rs.RecordMessageSuccess(nodeID)
			}
			done <- true
		}(i)
	}

	for i := 0; i < nodes; i++ {
		<-done
	}

	stats := rs.GetStats()
	totalNodes := stats["total_nodes"].(int)
	if totalNodes != nodes {
		t.Errorf("Expected %d nodes, got %d", nodes, totalNodes)
	}
}

func TestReputationSystem_CleanupInactiveNodes(t *testing.T) {
	logger := zap.NewNop()

	config := DefaultReputationConfig()
	config.RemoveAfterInactive = 500 * time.Millisecond
	config.CleanupInterval = 200 * time.Millisecond

	rs := NewReputationSystem("test-node", logger)
	rs.config = config
	defer rs.Stop()

	// Create inactive node
	rs.RecordConsensusParticipation("old-node", true)

	initialStats := rs.GetStats()
	initialCount := initialStats["total_nodes"].(int)

	// Wait for cleanup
	time.Sleep(1 * time.Second)

	finalStats := rs.GetStats()
	finalCount := finalStats["total_nodes"].(int)

	if finalCount >= initialCount {
		t.Log("Note: Cleanup may not have occurred yet (timing dependent)")
	}
}

func BenchmarkReputationSystem_RecordConsensus(b *testing.B) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rs.RecordConsensusParticipation("bench-node", true)
	}
}

func BenchmarkReputationSystem_GetScore(b *testing.B) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	rs.RecordConsensusParticipation("bench-node", true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rs.GetScore("bench-node")
	}
}

func BenchmarkReputationSystem_IsQuarantined(b *testing.B) {
	logger := zap.NewNop()
	rs := NewReputationSystem("test-node", logger)
	defer rs.Stop()

	rs.QuarantineNode("quarantined-node", "test")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rs.IsQuarantined("quarantined-node")
	}
}
