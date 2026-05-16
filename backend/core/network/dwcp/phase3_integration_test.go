package dwcp

import (
	"bytes"
	"context"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/conflict"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/consensus"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/loadbalancing"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/monitoring"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/multiregion"
	dwcpsync "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/sync/crdt"
	"go.uber.org/zap"
)

type phase3TestTransport struct{}

func (phase3TestTransport) Send(*dwcpsync.RegionPeer, *dwcpsync.Message) error {
	return nil
}

func (phase3TestTransport) Receive() (*dwcpsync.Message, error) {
	return nil, context.Canceled
}

func (phase3TestTransport) Close() error {
	return nil
}

// TestPhase3EndToEnd validates complete Phase 3 integration
func TestPhase3EndToEnd(t *testing.T) {
	t.Run("MultiRegionDeployment", testMultiRegionDeployment)
	t.Run("GlobalStateSync", testGlobalStateSync)
	t.Run("AdaptiveConsensus", testAdaptiveConsensus)
	t.Run("LoadBalancingFailover", testLoadBalancingFailover)
	t.Run("ConflictResolution", testConflictResolution)
	t.Run("MonitoringIntegration", testMonitoringIntegration)
	t.Run("DisasterRecovery", testDisasterRecovery)
}

func testMultiRegionDeployment(t *testing.T) {
	// Test 3-region deployment with networking
	regions := []string{"us-east-1", "eu-west-1", "ap-southeast-1"}

	topology := multiregion.NewGlobalTopology()

	// Add regions
	for _, region := range regions {
		r := &multiregion.Region{
			ID:   region,
			Name: region,
			Location: multiregion.GeoLocation{
				Latitude:  0.0,
				Longitude: 0.0,
			},
		}
		if err := topology.AddRegion(r); err != nil {
			t.Fatalf("Failed to add region %s: %v", region, err)
		}
	}

	// Verify all regions registered
	if len(topology.ListRegions()) != 3 {
		t.Errorf("Expected 3 regions, got %d", len(topology.ListRegions()))
	}

	runGoTest(t, "multiregion", "TestGlobalTopology|TestRoutingEngine|TestTunnelManager|TestTrafficEngineer|TestPathRedundancy|TestBandwidthManager|TestNetworkTelemetry|TestRouteUpdater|TestPerformanceMetrics")

	t.Log("✅ Multi-region deployment successful")
}

func testGlobalStateSync(t *testing.T) {
	// Test ASS/CRDT synchronization across regions
	// Create 3 ASS engines
	engines := make([]*dwcpsync.ASSEngine, 3)
	for i := 0; i < 3; i++ {
		engines[i] = dwcpsync.NewASSEngine(string('A'+rune(i)), phase3TestTransport{}, zap.NewNop())
	}

	// Start all engines
	for _, engine := range engines {
		if err := engine.Start(); err != nil {
			t.Fatalf("Failed to start ASS engine: %v", err)
		}
		defer engine.Stop()
	}

	// Perform update on first engine
	vmState := crdt.NewORSet("A")
	vmState.Add("running")

	if err := engines[0].Set("test-vm-1", vmState); err != nil {
		t.Fatalf("Failed to store CRDT update: %v", err)
	}

	storedState, ok := engines[0].Get("test-vm-1")
	if !ok {
		t.Fatal("Expected stored CRDT state")
	}

	storedSet, ok := storedState.(*crdt.ORSet)
	if !ok || !storedSet.Contains("running") {
		t.Fatal("Expected stored OR-Set to contain running state")
	}

	runGoTest(t, "sync", "TestCRDTConvergence|TestAntiEntropyConvergence|TestStateConvergenceTime|TestNetworkPartition")

	t.Log("✅ Global state synchronization working")
}

func testAdaptiveConsensus(t *testing.T) {
	// Test ACP algorithm selection and switching
	acp := consensus.NewACPEngine("node-1", "us-east-1", consensus.NewSimpleStateMachine())

	// Simulate low-latency environment (should choose Raft)
	acp.UpdateNetworkMetrics(consensus.NetworkMetrics{
		RegionCount: 2,
		AvgLatency:  30 * time.Millisecond,
		Bandwidth:   1000,
		PacketLoss:  0.001,
	})

	algo := acp.DecideAlgorithm()
	if algo != consensus.AlgorithmRaft {
		t.Errorf("Expected Raft for low latency, got %v", algo)
	}

	// Simulate high-latency environment (should choose Eventual)
	acp.UpdateNetworkMetrics(consensus.NetworkMetrics{
		RegionCount: 5,
		AvgLatency:  250 * time.Millisecond,
		Bandwidth:   100,
		PacketLoss:  0.05,
	})

	algo = acp.DecideAlgorithm()
	if algo != consensus.AlgorithmEventual {
		t.Errorf("Expected Eventual for high latency, got %v", algo)
	}

	runGoTest(t, "consensus", "TestACPAlgorithmDecision|TestACPSwitching|TestRaftConsensus|TestPaxosConsensus|TestEPaxosConsensus|TestEventualConsistency|TestHybridConsensus")

	t.Log("✅ Adaptive consensus algorithm selection working")
}

func testLoadBalancingFailover(t *testing.T) {
	// Test global load balancing with failover
	config := loadbalancing.DefaultConfig()
	config.Algorithm = loadbalancing.AlgorithmRoundRobin

	lb, err := loadbalancing.NewGeoLoadBalancer(config)
	if err != nil {
		t.Fatalf("Failed to create load balancer: %v", err)
	}

	// Add servers from multiple regions
	servers := []*loadbalancing.Server{
		{ID: "us-1", Region: "us-east-1", Address: "10.0.1.1", Port: 8080, Weight: 100},
		{ID: "eu-1", Region: "eu-west-1", Address: "10.0.2.1", Port: 8080, Weight: 100},
		{ID: "ap-1", Region: "ap-southeast-1", Address: "10.0.3.1", Port: 8080, Weight: 100},
	}

	for _, server := range servers {
		if err := lb.AddServer(server); err != nil {
			t.Fatalf("Failed to add server %s: %v", server.ID, err)
		}
	}

	// Simulate server failure
	servers[0].Health = loadbalancing.ServerUnhealthy

	// Get server for US client (should get EU due to US failure)
	clientIP := "1.2.3.4" // Mock US IP
	decision, err := lb.SelectServer(clientIP, "")
	if err != nil {
		t.Fatalf("Failed to select server: %v", err)
	}

	if decision.Server.ID == "us-1" {
		t.Error("Should not select unhealthy server")
	}

	runGoTest(t, "loadbalancing", "TestSelectServerRoundRobin|TestSelectServerLeastConnections|TestSelectServerGeoProximity|TestSelectServerWithSessionAffinity|TestFailoverOnServerFailure|TestNoHealthyServersError|TestRoutingLatency|TestConcurrentRequests|TestHealthCheckerStartStop|TestCircuitBreakerRecovery|TestMetricsAggregation")

	t.Logf("✅ Load balancing failover working (selected: %s)", decision.Server.ID)
}

func testConflictResolution(t *testing.T) {
	// Test conflict detection and resolution
	engine := conflict.NewMergeEngine(conflict.DefaultMergeConfig())

	// Create concurrent updates
	base := map[string]interface{}{
		"vm_id":       "vm-001",
		"power_state": "stopped",
		"cpu":         2,
	}

	version1 := map[string]interface{}{
		"vm_id":       "vm-001",
		"power_state": "running",
		"cpu":         4,
	}

	version2 := map[string]interface{}{
		"vm_id":       "vm-001",
		"power_state": "stopped",
		"cpu":         2,
		"memory":      8,
	}

	result, err := engine.ThreeWayMerge(context.Background(), base, version1, version2)
	if err != nil {
		t.Fatalf("Merge failed: %v", err)
	}

	resultMap, ok := result.(map[string]interface{})
	if !ok {
		t.Fatalf("Expected merged map, got %T", result)
	}

	// The current generic map merger preserves the local side on ambiguous map conflicts.
	if resultMap["power_state"] != "running" || resultMap["cpu"] != 4 {
		t.Errorf("Expected local map to win ambiguous conflict, got %#v", resultMap)
	}

	runGoTest(t, "conflict", "TestVectorClockCompare|TestVectorClockConcurrent|TestConflictDetection|TestNoConflictDetection|TestConflictComplexityCalculation|TestConflictCleanup|TestLastWriteWinsStrategy|TestMultiValueRegisterStrategy|TestAutomaticRollbackStrategy|TestConsensusVoteStrategy|TestStrategyRegistry|TestThreeWayMergeIdentical|TestThreeWayMergeLocalChange|TestThreeWayMergeRemoteChange|TestThreeWayMergeConflict|TestMapMerge|TestStructuralDiff")

	t.Log("✅ Conflict resolution working")
}

func testMonitoringIntegration(t *testing.T) {
	// Test monitoring metrics collection
	collector, err := monitoring.NewMetricsCollector("us-east-1")
	if err != nil {
		t.Fatalf("Failed to create collector: %v", err)
	}

	// Record some metrics
	ctx := context.Background()
	collector.RecordRequest(ctx, "replicate", map[string]string{"region": "us-east-1", "status": "200"})
	collector.RecordLatency(ctx, "replicate", 50, map[string]string{"region": "us-east-1"})
	collector.RecordError(ctx, "replicate", "server_error", map[string]string{"region": "ap-southeast-1"})

	// Get metrics
	metrics := collector.GetMetrics(time.Now().Add(-1 * time.Minute))

	if len(metrics) == 0 {
		t.Error("Expected metrics data")
	}

	runGoTest(t, "monitoring", "TestMetricsCollector|TestAggregatedMetric|TestTracingSystem|TestNetworkTelemetryMeasureLatencyUsesRecordedSample|TestNetworkTelemetryMeasureLatencyUsesTopology|TestNetworkTelemetryMeasureLatencyUsesTunnelHealth|TestNetworkTelemetryMeasureLatencyUnknownPair|TestDetectAnomalyReportsExpectedAndDeviation|TestDetectAnomalyUsesValueAsExpectedInsideRange|TestMonitoringPipelineStoresRecentAnomalies|TestMonitoringPipelineAnomalyHistoryBound|TestMetricVector_ToSlice|TestIsolationForest_Detection|TestLSTMAutoencoder_Detection|TestZScoreDetector_Detection|TestSeasonalESD_Detection|TestEnsembleDetector_Aggregation|TestMonitoringPipeline_ProcessMetrics")

	t.Log("✅ Monitoring integration working")
}

func testDisasterRecovery(t *testing.T) {
	runGoTest(t, filepath.Join("..", "..", "dr"), "TestDROrchestrator|TestFailoverExecution|TestSplitBrainPrevention|TestRegionalFailover")

	t.Log("✅ Disaster recovery framework validated")
}

func runGoTest(t *testing.T, dir, testPattern string) {
	t.Helper()

	cmd := exec.Command("go", "test", ".", "-run", testPattern, "-count=1", "-timeout", "30s")
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go test failed in %s with pattern %q: %v\n%s", dir, testPattern, err, string(output))
	}

	if len(output) == 0 {
		t.Fatalf("go test in %s with pattern %q produced no output", dir, testPattern)
	}

	if testing.Verbose() {
		t.Logf("go test passed in %s with pattern %q:\n%s", dir, testPattern, string(output))
	} else if !containsPassOutput(output) {
		t.Fatalf("go test in %s with pattern %q did not report a package pass:\n%s", dir, testPattern, string(output))
	}
}

func containsPassOutput(output []byte) bool {
	trimmed := bytes.TrimSpace(output)
	return len(trimmed) >= 2 && (bytes.HasPrefix(trimmed, []byte("ok")) || bytes.Contains(trimmed, []byte("\nok")) || bytes.Contains(trimmed, []byte("\nPASS\n")) || bytes.Contains(trimmed, []byte("PASS\nok")))
}

// Benchmark Phase 3 Performance
func BenchmarkPhase3Performance(b *testing.B) {
	b.Run("CRDTMerge", benchmarkCRDTMerge)
	b.Run("ConsensusDecision", benchmarkConsensusDecision)
	b.Run("LoadBalancerSelect", benchmarkLoadBalancerSelect)
	b.Run("ConflictDetection", benchmarkConflictDetection)
}

func benchmarkCRDTMerge(b *testing.B) {
	set1 := crdt.NewORSet("node-1")
	set2 := crdt.NewORSet("node-2")

	for i := 0; i < 100; i++ {
		set1.Add(string(rune('A' + i)))
		set2.Add(string(rune('A' + i + 50)))
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		set1.Merge(set2)
	}
}

func benchmarkConsensusDecision(b *testing.B) {
	acp := consensus.NewACPEngine("node-1", "us-east-1", consensus.NewSimpleStateMachine())

	metrics := consensus.NetworkMetrics{
		RegionCount: 3,
		AvgLatency:  50 * time.Millisecond,
		Bandwidth:   500,
		PacketLoss:  0.01,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		acp.UpdateNetworkMetrics(metrics)
		acp.DecideAlgorithm()
	}
}

func benchmarkLoadBalancerSelect(b *testing.B) {
	config := loadbalancing.DefaultConfig()
	config.Algorithm = loadbalancing.AlgorithmRoundRobin
	config.MaxConnections = 10000

	lb, err := loadbalancing.NewGeoLoadBalancer(config)
	if err != nil {
		b.Fatalf("Failed to create load balancer: %v", err)
	}

	for i := 0; i < 10; i++ {
		if err := lb.AddServer(&loadbalancing.Server{
			ID:      string(rune('A' + i)),
			Region:  "us-east-1",
			Address: "10.0.0." + string(rune('1'+i)),
			Port:    8080,
			Weight:  100,
		}); err != nil {
			b.Fatalf("Failed to add server: %v", err)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := lb.SelectServer("192.168.1.1", ""); err != nil {
			b.Fatalf("Failed to select server: %v", err)
		}
	}
}

func benchmarkConflictDetection(b *testing.B) {
	detector := conflict.NewConflictDetector(conflict.DefaultDetectorConfig())

	vc1 := conflict.NewVectorClock()
	vc1.Increment("A")
	vc1.Increment("A")

	vc2 := conflict.NewVectorClock()
	vc2.Increment("B")
	vc2.Increment("B")

	local := &conflict.Version{
		VectorClock: vc1,
		Timestamp:   time.Now(),
		NodeID:      "A",
		Data:        map[string]interface{}{"key": "value1"},
	}
	remote := &conflict.Version{
		VectorClock: vc2,
		Timestamp:   time.Now(),
		NodeID:      "B",
		Data:        map[string]interface{}{"key": "value2"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := detector.DetectConflict(context.Background(), "resource-1", local, remote); err != nil {
			b.Fatalf("Failed to detect conflict: %v", err)
		}
	}
}
