package consensus

import (
	"testing"
	"time"

	"go.uber.org/zap"
)

// TestByzantineDetectorInitialization tests detector creation
func TestByzantineDetectorInitialization(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	detector := NewByzantineDetector(logger)

	if detector == nil {
		t.Fatal("Failed to create Byzantine detector")
	}

	metrics := detector.GetMetrics()
	if metrics["total_nodes"] != int64(0) {
		t.Errorf("Expected 0 nodes, got %v", metrics["total_nodes"])
	}
}

// TestByzantineDetectorMessageFailure tests failure tracking
func TestByzantineDetectorMessageFailure(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	detector := NewByzantineDetector(logger)

	nodeID := "node-1"

	// Record multiple failures
	for i := 0; i < 5; i++ {
		detector.RecordMessageFailure(nodeID)
	}

	// Check suspicion level
	suspicion := detector.GetSuspicionLevel(nodeID)
	if suspicion < 50 {
		t.Errorf("Expected suspicion >= 50, got %d", suspicion)
	}

	// Check if quarantined
	if !detector.IsQuarantined(nodeID) {
		t.Error("Expected node to be quarantined")
	}
}

// TestByzantineDetectorInconsistentVotes tests vote tracking
func TestByzantineDetectorInconsistentVotes(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	detector := NewByzantineDetector(logger)

	nodeID := "node-2"

	// Record messages first
	for i := 0; i < 10; i++ {
		detector.RecordMessageFailure(nodeID)
	}

	// Record inconsistent votes
	for i := 0; i < 5; i++ {
		detector.RecordInconsistentVote(nodeID)
	}

	stats := detector.GetNodeStats(nodeID)
	if stats == nil {
		t.Fatal("Expected node stats")
	}

	if stats.InconsistentVotes != 5 {
		t.Errorf("Expected 5 inconsistent votes, got %d", stats.InconsistentVotes)
	}
}

// TestByzantineDetectorResponseTime tests timeout detection
func TestByzantineDetectorResponseTime(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	detector := NewByzantineDetector(logger)

	nodeID := "node-3"

	// Record slow responses
	for i := 0; i < 5; i++ {
		detector.RecordResponseTime(nodeID, 10*time.Second)
	}

	suspicion := detector.GetSuspicionLevel(nodeID)
	if suspicion < 50 {
		t.Errorf("Expected suspicion >= 50, got %d", suspicion)
	}
}

// TestThreatAnalyzerInitialization tests analyzer creation
func TestThreatAnalyzerInitialization(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	analyzer := NewThreatAnalyzer(logger)

	if analyzer == nil {
		t.Fatal("Failed to create threat analyzer")
	}

	if analyzer.GetCurrentThreatLevel() != ThreatLevelLow {
		t.Error("Expected initial threat level to be LOW")
	}

	if analyzer.GetRecommendedProtocol() != "raft" {
		t.Error("Expected initial protocol to be raft")
	}
}

// TestThreatAnalyzerLowThreat tests low threat assessment
func TestThreatAnalyzerLowThreat(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	analyzer := NewThreatAnalyzer(logger)

	// 5% Byzantine nodes (low threat)
	analyzer.UpdateThreatAssessment(1, 20)

	if analyzer.GetCurrentThreatLevel() != ThreatLevelLow {
		t.Error("Expected LOW threat level")
	}

	if analyzer.GetRecommendedProtocol() != "raft" {
		t.Error("Expected raft protocol for low threat")
	}
}

// TestThreatAnalyzerMediumThreat tests medium threat assessment
func TestThreatAnalyzerMediumThreat(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	analyzer := NewThreatAnalyzer(logger)

	// 15% Byzantine nodes (medium threat)
	analyzer.UpdateThreatAssessment(3, 20)

	if analyzer.GetCurrentThreatLevel() != ThreatLevelMedium {
		t.Error("Expected MEDIUM threat level")
	}

	if analyzer.GetRecommendedProtocol() != "pbft" {
		t.Error("Expected pbft protocol for medium threat")
	}
}

// TestThreatAnalyzerHighThreat tests high threat assessment
func TestThreatAnalyzerHighThreat(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	analyzer := NewThreatAnalyzer(logger)

	// 25% Byzantine nodes (high threat)
	analyzer.UpdateThreatAssessment(5, 20)

	if analyzer.GetCurrentThreatLevel() != ThreatLevelHigh {
		t.Error("Expected HIGH threat level")
	}

	if analyzer.GetRecommendedProtocol() != "probft" {
		t.Error("Expected probft protocol for high threat")
	}
}

// TestThreatAnalyzerCriticalThreat tests critical threat assessment
func TestThreatAnalyzerCriticalThreat(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	analyzer := NewThreatAnalyzer(logger)

	// 40% Byzantine nodes (critical threat)
	analyzer.UpdateThreatAssessment(8, 20)

	if analyzer.GetCurrentThreatLevel() != ThreatLevelCritical {
		t.Error("Expected CRITICAL threat level")
	}

	if analyzer.GetRecommendedProtocol() != "tpbft" {
		t.Error("Expected tpbft protocol for critical threat")
	}
}

// TestThreatAnalyzerProtocolSwitching tests protocol switching
func TestThreatAnalyzerProtocolSwitching(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	analyzer := NewThreatAnalyzer(logger)

	// Start with low threat
	analyzer.UpdateThreatAssessment(1, 20)
	protocol1 := analyzer.GetRecommendedProtocol()

	// Increase to high threat
	analyzer.UpdateThreatAssessment(5, 20)
	protocol2 := analyzer.GetRecommendedProtocol()

	if protocol1 == protocol2 {
		t.Error("Expected protocol to switch")
	}

	metrics := analyzer.GetMetrics()
	if metrics["protocol_switches"] != int64(1) {
		t.Errorf("Expected 1 protocol switch, got %v", metrics["protocol_switches"])
	}
}

// TestByzantineDetectorMetrics tests metrics collection
func TestByzantineDetectorMetrics(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	detector := NewByzantineDetector(logger)

	// Record some failures
	detector.RecordMessageFailure("node-1")
	detector.RecordMessageFailure("node-1")
	detector.RecordMessageFailure("node-2")

	metrics := detector.GetMetrics()

	if metrics["total_nodes"] != int64(2) {
		t.Errorf("Expected 2 nodes, got %v", metrics["total_nodes"])
	}
}

// TestThreatAnalyzerMetrics tests threat analyzer metrics
func TestThreatAnalyzerMetrics(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	analyzer := NewThreatAnalyzer(logger)

	analyzer.UpdateThreatAssessment(2, 10)
	analyzer.UpdateThreatAssessment(3, 10)

	metrics := analyzer.GetMetrics()

	if metrics["threat_assessments"] != int64(2) {
		t.Errorf("Expected 2 assessments, got %v", metrics["threat_assessments"])
	}

	if metrics["byzantine_ratio"] != 0.3 {
		t.Errorf("Expected 0.3 ratio, got %v", metrics["byzantine_ratio"])
	}
}

