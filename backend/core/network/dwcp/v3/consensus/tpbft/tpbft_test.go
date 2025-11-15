// Package tpbft implements tests for T-PBFT with EigenTrust
package tpbft

import (
	"fmt"
	"testing"
	"time"
)

// TestEigenTrustBasic tests basic EigenTrust functionality
func TestEigenTrustBasic(t *testing.T) {
	et := NewEigenTrust()

	// Record some interactions
	et.UpdateLocalTrust("node1", "node2", 0.9)
	et.UpdateLocalTrust("node1", "node3", 0.8)
	et.UpdateLocalTrust("node2", "node3", 1.0)
	et.UpdateLocalTrust("node3", "node1", 0.7)

	// Compute global trust
	et.ComputeGlobalTrust()

	// Verify trust scores were computed
	score2 := et.GetTrustScore("node2")
	score3 := et.GetTrustScore("node3")

	if score2 == 0 || score3 == 0 {
		t.Errorf("Trust scores not computed: node2=%f, node3=%f", score2, score3)
	}

	// Node3 should have higher trust (receives more positive ratings)
	if score3 <= score2 {
		t.Logf("Warning: Expected node3 trust (%f) > node2 trust (%f)", score3, score2)
	}
}

// TestEigenTrustConvergence tests convergence of trust computation
func TestEigenTrustConvergence(t *testing.T) {
	et := NewEigenTrust()

	// Create a network with clear trust hierarchy
	et.UpdateLocalTrust("node1", "node2", 1.0)
	et.UpdateLocalTrust("node2", "node3", 1.0)
	et.UpdateLocalTrust("node3", "node4", 1.0)
	et.UpdateLocalTrust("node4", "node1", 1.0) // Circular trust

	// Compute multiple times
	et.ComputeGlobalTrust()
	scores1 := make(map[string]float64)
	for i := 1; i <= 4; i++ {
		nodeID := fmt.Sprintf("node%d", i)
		scores1[nodeID] = et.GetTrustScore(nodeID)
	}

	// Compute again
	et.ComputeGlobalTrust()
	scores2 := make(map[string]float64)
	for i := 1; i <= 4; i++ {
		nodeID := fmt.Sprintf("node%d", i)
		scores2[nodeID] = et.GetTrustScore(nodeID)
	}

	// Scores should be stable (converged)
	for node, score1 := range scores1 {
		score2 := scores2[node]
		diff := abs(score1 - score2)
		if diff > 0.01 {
			t.Errorf("Trust not converged for %s: %f vs %f (diff=%f)", node, score1, score2, diff)
		}
	}
}

// TestTopNodes tests selection of top trusted nodes
func TestTopNodes(t *testing.T) {
	et := NewEigenTrust()

	// Create network with varying trust
	et.UpdateLocalTrust("node1", "nodeA", 1.0)
	et.UpdateLocalTrust("node2", "nodeA", 1.0)
	et.UpdateLocalTrust("node3", "nodeA", 1.0)

	et.UpdateLocalTrust("node1", "nodeB", 0.8)
	et.UpdateLocalTrust("node2", "nodeB", 0.8)

	et.UpdateLocalTrust("node1", "nodeC", 0.5)

	et.ComputeGlobalTrust()

	// Get top 2 nodes
	topNodes := et.GetTopNodes(2)

	if len(topNodes) != 2 {
		t.Errorf("Expected 2 top nodes, got %d", len(topNodes))
	}

	// nodeA should be first (highest trust)
	if len(topNodes) > 0 && topNodes[0] != "nodeA" {
		t.Logf("Warning: Expected nodeA as top node, got %s", topNodes[0])
	}
}

// TestTPBFTCommitteeSelection tests trust-based committee selection
func TestTPBFTCommitteeSelection(t *testing.T) {
	et := NewEigenTrust()
	tpbft := NewTPBFT("node1", et)

	// Create nodes with trust scores
	nodes := []string{"node1", "node2", "node3", "node4", "node5"}
	for _, node := range nodes {
		et.SetPreTrust(node, 0.5)
	}

	// Give some nodes higher trust
	et.UpdateLocalTrust("system", "node2", 1.0)
	et.UpdateLocalTrust("system", "node3", 1.0)
	et.ComputeGlobalTrust()

	// Select committee
	committee := tpbft.SelectCommittee()

	if len(committee) == 0 {
		t.Error("Committee selection returned empty set")
	}

	// Verify committee size
	if len(committee) > tpbft.committeeSize {
		t.Errorf("Committee size %d exceeds max %d", len(committee), tpbft.committeeSize)
	}

	t.Logf("Selected committee: %v", committee)
}

// TestTPBFTConsensusFlow tests basic consensus flow
func TestTPBFTConsensusFlow(t *testing.T) {
	et := NewEigenTrust()
	tpbft := NewTPBFT("node1", et)

	// Setup committee
	committee := []string{"node1", "node2", "node3", "node4"}
	for _, node := range committee {
		et.SetPreTrust(node, 0.8)
	}
	et.ComputeGlobalTrust()
	tpbft.committee = committee
	tpbft.f = 1 // Allow 1 Byzantine node

	// Create request
	request := Request{
		ID:        "req-001",
		Timestamp: time.Now(),
		Data:      []byte("test transaction"),
		ClientID:  "client1",
	}

	// Note: Full consensus requires network simulation
	// Test individual phases
	digest, err := tpbft.calculateDigest(request)
	if err != nil {
		t.Fatalf("Failed to calculate digest: %v", err)
	}

	if digest == "" {
		t.Error("Empty digest calculated")
	}

	t.Logf("Request digest: %s", digest)
}

// TestTrustManagerInteractions tests trust manager interaction recording
func TestTrustManagerInteractions(t *testing.T) {
	tm := NewTrustManager()

	// Record various interactions
	tm.RecordInteraction("node1", "node2", CorrectVote, "consensus vote")
	tm.RecordInteraction("node1", "node3", TimelyResponse, "quick response")
	tm.RecordInteraction("node2", "node3", ValidMessage, "valid message")

	// Recompute trust
	tm.RecomputeTrust()

	// Check trust scores
	score2 := tm.GetTrustScore("node2")
	score3 := tm.GetTrustScore("node3")

	if score2 == 0 || score3 == 0 {
		t.Errorf("Trust scores not updated: node2=%f, node3=%f", score2, score3)
	}

	// Check statistics
	stats := tm.GetStats()
	if stats["total_interactions"].(int) != 3 {
		t.Errorf("Expected 3 interactions, got %d", stats["total_interactions"].(int))
	}
}

// TestByzantineDetection tests Byzantine behavior detection
func TestByzantineDetection(t *testing.T) {
	tm := NewTrustManager()

	// Record normal interactions
	tm.RecordInteraction("node1", "node2", CorrectVote, "vote")

	// Record Byzantine behavior
	tm.RecordInteraction("node1", "node3", ByzantineBehavior, "double voting")

	tm.RecomputeTrust()

	// Byzantine node should have very low trust
	byzantineTrust := tm.GetTrustScore("node3")
	normalTrust := tm.GetTrustScore("node2")

	if byzantineTrust >= normalTrust {
		t.Errorf("Byzantine node trust (%f) should be lower than normal node (%f)",
			byzantineTrust, normalTrust)
	}

	// Check Byzantine detection count
	stats := tm.GetStats()
	if stats["byzantine_detected"].(int) != 1 {
		t.Errorf("Expected 1 Byzantine detection, got %d", stats["byzantine_detected"].(int))
	}
}

// TestCommitteeValidation tests trust-based committee validation
func TestCommitteeValidation(t *testing.T) {
	tm := NewTrustManager()

	// Setup nodes with different trust levels via interactions
	// (Pre-trust alone doesn't populate trustScores until interactions occur)
	tm.RecordInteraction("system", "node1", CorrectVote, "high trust")
	tm.RecordInteraction("system", "node1", CorrectVote, "high trust")
	tm.RecordInteraction("system", "node1", TimelyResponse, "high trust")

	tm.RecordInteraction("system", "node2", CorrectVote, "medium trust")
	tm.RecordInteraction("system", "node2", TimelyResponse, "medium trust")

	tm.RecordInteraction("system", "node3", IncorrectVote, "low trust")
	tm.RecordInteraction("system", "node3", LateResponse, "low trust")
	tm.RecordInteraction("system", "node3", InvalidMessage, "low trust")

	// Also set pre-trust to influence scores
	tm.SetPreTrustedNode("node1", 0.9)
	tm.SetPreTrustedNode("node2", 0.8)
	tm.SetPreTrustedNode("node3", 0.3) // Lower pre-trust

	// Validate committee with threshold 0.5
	committee := []string{"node1", "node2", "node3"}
	valid, invalid := tm.ValidateCommittee(committee, 0.5)

	if valid {
		t.Error("Committee should be invalid due to node3")
	}

	if len(invalid) != 1 || invalid[0] != "node3" {
		t.Errorf("Expected [node3] as invalid, got %v (scores: node1=%f, node2=%f, node3=%f)",
			invalid,
			tm.GetTrustScore("node1"),
			tm.GetTrustScore("node2"),
			tm.GetTrustScore("node3"))
	}
}

// TestPerformanceMetrics tests T-PBFT performance tracking
func TestPerformanceMetrics(t *testing.T) {
	et := NewEigenTrust()
	tpbft := NewTPBFT("node1", et)

	// Setup committee
	tpbft.committee = []string{"node1", "node2", "node3"}
	tpbft.totalRequests = 100
	tpbft.consensusLatency = 50 * time.Millisecond
	tpbft.throughput = 2000.0

	// Get metrics
	metrics := tpbft.GetMetrics()

	if metrics["total_requests"].(int) != 100 {
		t.Errorf("Expected 100 requests, got %d", metrics["total_requests"].(int))
	}

	if metrics["throughput"].(float64) != 2000.0 {
		t.Errorf("Expected throughput 2000, got %f", metrics["throughput"].(float64))
	}

	t.Logf("T-PBFT Metrics: %+v", metrics)
}

// Benchmark T-PBFT trust computation
func BenchmarkTrustComputation(b *testing.B) {
	et := NewEigenTrust()

	// Setup network
	nodes := make([]string, 100)
	for i := 0; i < 100; i++ {
		nodes[i] = fmt.Sprintf("node%d", i)
	}

	// Create interactions
	for i := 0; i < 100; i++ {
		for j := 0; j < 10; j++ {
			from := nodes[i]
			to := nodes[(i+j+1)%100]
			et.UpdateLocalTrust(from, to, 0.8)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		et.ComputeGlobalTrust()
	}
}

// Benchmark committee selection
func BenchmarkCommitteeSelection(b *testing.B) {
	et := NewEigenTrust()
	tpbft := NewTPBFT("node1", et)

	// Setup nodes
	for i := 0; i < 1000; i++ {
		nodeID := fmt.Sprintf("node%d", i)
		et.SetPreTrust(nodeID, float64(i%100)/100.0)
	}
	et.ComputeGlobalTrust()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tpbft.SelectCommittee()
	}
}

// Helper function
func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
