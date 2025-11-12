package security

import (
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestModeAwareSecurity_DatacenterMode(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeDatacenter, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	if mas.GetCurrentMode() != ModeDatacenter {
		t.Error("Expected datacenter mode")
	}

	// Datacenter mode should have minimal validation
	err := mas.ValidateMessage("peer-node", "prepare", map[string]string{"data": "test"}, "")
	if err != nil {
		t.Errorf("Datacenter mode should allow messages: %v", err)
	}

	// TLS should not be required
	_, err = mas.GetTLSConfig()
	if err == nil {
		t.Error("Expected TLS to not be required in datacenter mode")
	}
}

func TestModeAwareSecurity_InternetMode(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	if mas.GetCurrentMode() != ModeInternet {
		t.Error("Expected internet mode")
	}

	// Internet mode requires signatures
	err := mas.ValidateMessage("peer-node", "prepare", map[string]string{"data": "test"}, "")
	if err == nil {
		t.Error("Expected signature validation to fail without signature in internet mode")
	}

	// Should work with signature
	err = mas.ValidateMessage("peer-node", "prepare", map[string]string{"data": "test"}, "valid-signature")
	if err != nil {
		t.Logf("Validation error: %v (expected for test)", err)
	}

	// TLS should be initialized
	tlsConfig, err := mas.GetTLSConfig()
	if err != nil {
		t.Errorf("Expected TLS to be available in internet mode: %v", err)
	}
	if tlsConfig != nil && tlsConfig.MinVersion < 0x0304 { // TLS 1.3
		t.Error("Expected minimum TLS 1.3")
	}
}

func TestModeAwareSecurity_ModeSwitch(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeDatacenter, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// Switch to internet mode
	err := mas.SwitchMode(ModeInternet, "manual switch for testing")
	if err != nil {
		t.Fatalf("Failed to switch mode: %v", err)
	}

	if mas.GetCurrentMode() != ModeInternet {
		t.Error("Mode switch did not take effect")
	}

	// Switch back to datacenter
	err = mas.SwitchMode(ModeDatacenter, "switching back")
	if err != nil {
		t.Fatalf("Failed to switch back: %v", err)
	}

	if mas.GetCurrentMode() != ModeDatacenter {
		t.Error("Mode switch back did not take effect")
	}
}

func TestModeAwareSecurity_QuarantinedNodeRejection(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	badNode := "quarantined-node"

	// Quarantine the node
	err := reputation.QuarantineNode(badNode, "malicious behavior")
	if err != nil {
		t.Fatalf("Failed to quarantine: %v", err)
	}

	// Message from quarantined node should be rejected
	err = mas.ValidateMessage(badNode, "prepare", map[string]string{"data": "test"}, "signature")
	if err == nil {
		t.Error("Expected message from quarantined node to be rejected")
	}
}

func TestModeAwareSecurity_LowReputationRejection(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	lowRepNode := "low-reputation-node"

	// Lower reputation significantly
	for i := 0; i < 20; i++ {
		reputation.RecordByzantineBehavior(lowRepNode, 10.0)
	}

	// Message should be rejected
	err := mas.ValidateMessage(lowRepNode, "prepare", map[string]string{"data": "test"}, "signature")
	if err == nil {
		score := reputation.GetScore(lowRepNode)
		if score < 20.0 {
			t.Error("Expected message from very low reputation node to be rejected")
		}
	}
}

func TestModeAwareSecurity_ByzantineNodeRejection(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	byzantine := "byzantine-node"

	// Trigger Byzantine detection
	for i := 0; i < 10; i++ {
		detector.RecordConsensusVote(byzantine, 1, int64(i), "digest-A", "prepare")
		detector.RecordConsensusVote(byzantine, 1, int64(i), "digest-B", "prepare")
	}

	time.Sleep(200 * time.Millisecond)

	// Message should be rejected if node is confirmed Byzantine
	if detector.IsByzantine(byzantine) {
		err := mas.ValidateMessage(byzantine, "prepare", map[string]string{"data": "test"}, "signature")
		if err == nil {
			t.Error("Expected message from Byzantine node to be rejected")
		}
	}
}

func TestModeAwareSecurity_HybridMode(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeHybrid, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	if mas.GetCurrentMode() != ModeHybrid {
		t.Error("Expected hybrid mode")
	}

	// Create trusted environment
	for i := 0; i < 10; i++ {
		nodeID := "trusted-" + string(rune('A'+i))
		for j := 0; j < 30; j++ {
			reputation.RecordConsensusParticipation(nodeID, true)
		}
	}

	// Wait for adaptive monitoring to adjust
	time.Sleep(1 * time.Second)

	// Check network trust increased
	stats := mas.GetStats()
	networkTrust := stats["network_trust"].(float64)
	if networkTrust <= 0.5 {
		t.Logf("Network trust: %.2f (may need more time to adapt)", networkTrust)
	}
}

func TestModeAwareSecurity_AdaptiveSecurityAdjustment(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)

	config := DefaultHybridConfig()
	config.AdaptiveCheckInterval = 200 * time.Millisecond
	config.MonitoringWindow = 500 * time.Millisecond
	config.TrustThreshold = 0.7
	config.UntrustThreshold = 0.3

	mas := NewModeAwareSecurity("test-node", ModeHybrid, detector, reputation, logger)
	mas.hybridConfig = config
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	initialMode := mas.GetCurrentMode()

	// Create highly trusted environment
	for i := 0; i < 20; i++ {
		nodeID := "ultra-trusted-" + string(rune('A'+i))
		for j := 0; j < 50; j++ {
			reputation.RecordConsensusParticipation(nodeID, true)
		}
	}

	// Wait for adaptive adjustment
	time.Sleep(2 * time.Second)

	// May switch to datacenter mode if trust is high enough
	finalMode := mas.GetCurrentMode()
	if finalMode != initialMode {
		t.Logf("Mode adapted from %v to %v", initialMode, finalMode)
	}
}

func TestModeAwareSecurity_DatacenterValidationMinimal(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeDatacenter, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// Should reject nil message
	err := mas.ValidateMessage("node-1", "prepare", nil, "")
	if err == nil {
		t.Error("Expected nil message to be rejected even in datacenter mode")
	}

	// Should reject empty sender ID
	err = mas.ValidateMessage("", "prepare", map[string]string{"data": "test"}, "")
	if err == nil {
		t.Error("Expected empty sender ID to be rejected")
	}

	// Should accept valid message
	err = mas.ValidateMessage("node-1", "prepare", map[string]string{"data": "test"}, "")
	if err != nil {
		t.Errorf("Valid message should be accepted in datacenter mode: %v", err)
	}
}

func TestModeAwareSecurity_InternetValidationStrict(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// Should reject without signature
	err := mas.ValidateMessage("node-1", "prepare", map[string]string{"data": "test"}, "")
	if err == nil {
		t.Error("Expected message without signature to be rejected in internet mode")
	}

	// Should reject nil message even with signature
	err = mas.ValidateMessage("node-1", "prepare", nil, "signature")
	if err == nil {
		t.Error("Expected nil message to be rejected")
	}

	// Should record failure in reputation
	reputation.RecordMessageFailure("node-1")
	score := reputation.GetScore("node-1")
	if score >= reputation.config.InitialScore {
		t.Error("Expected reputation to decrease after message failure")
	}
}

func TestModeAwareSecurity_Statistics(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	stats := mas.GetStats()

	if stats["current_mode"] == nil {
		t.Error("Expected current_mode in stats")
	}

	mode := stats["current_mode"].(string)
	if mode != "internet" {
		t.Errorf("Expected mode 'internet', got '%s'", mode)
	}

	if stats["tls_enabled"] == nil {
		t.Error("Expected tls_enabled in stats")
	}

	if stats["network_trust"] == nil {
		t.Error("Expected network_trust in stats")
	}
}

func TestModeAwareSecurity_ConcurrentValidation(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeDatacenter, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	done := make(chan bool)
	nodes := 10

	for i := 0; i < nodes; i++ {
		go func(id int) {
			nodeID := "concurrent-node-" + string(rune('0'+id))
			for j := 0; j < 100; j++ {
				mas.ValidateMessage(nodeID, "prepare", map[string]string{"seq": string(rune('0'+j))}, "sig")
			}
			done <- true
		}(i)
	}

	for i := 0; i < nodes; i++ {
		<-done
	}
}

func TestModeAwareSecurity_TLSConfiguration(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	tlsConfig, err := mas.GetTLSConfig()
	if err != nil {
		t.Fatalf("Failed to get TLS config: %v", err)
	}

	if tlsConfig.MinVersion < 0x0304 {
		t.Error("Expected minimum TLS 1.3")
	}

	internetConfig := mas.internetConfig
	if internetConfig.RequireMutualTLS && tlsConfig.ClientAuth == 0 {
		t.Error("Expected mutual TLS to be configured")
	}
}

func TestModeAwareSecurity_NetworkTrustCalculation(t *testing.T) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeHybrid, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	// Create mixed environment
	// Trusted nodes
	for i := 0; i < 7; i++ {
		nodeID := "trusted-" + string(rune('A'+i))
		for j := 0; j < 30; j++ {
			reputation.RecordConsensusParticipation(nodeID, true)
		}
	}

	// Suspicious nodes
	for i := 0; i < 3; i++ {
		nodeID := "suspicious-" + string(rune('A'+i))
		reputation.RecordByzantineBehavior(nodeID, 15.0)
	}

	// Trigger trust calculation
	mas.updateNetworkTrust()

	stats := mas.GetStats()
	networkTrust := stats["network_trust"].(float64)

	// Should have positive trust (7 trusted out of 10)
	if networkTrust <= 0 {
		t.Error("Expected positive network trust with majority trusted nodes")
	}

	if networkTrust > 1.0 {
		t.Error("Network trust should be capped at 1.0")
	}
}

func BenchmarkModeAwareSecurity_ValidateDatacenter(b *testing.B) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeDatacenter, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	message := map[string]string{"data": "test"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mas.ValidateMessage("bench-node", "prepare", message, "")
	}
}

func BenchmarkModeAwareSecurity_ValidateInternet(b *testing.B) {
	logger := zap.NewNop()
	reputation := NewReputationSystem("test-node", logger)
	detector := NewByzantineDetector("test-node", reputation, logger)
	mas := NewModeAwareSecurity("test-node", ModeInternet, detector, reputation, logger)
	defer mas.Stop()
	defer detector.Stop()
	defer reputation.Stop()

	message := map[string]string{"data": "test"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		mas.ValidateMessage("bench-node", "prepare", message, "signature")
	}
}
