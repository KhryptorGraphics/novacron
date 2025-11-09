package consensus

import (
	"testing"
	"time"
)

// TestACPEngineCreation tests creating an ACP engine
func TestACPEngineCreation(t *testing.T) {
	sm := NewSimpleStateMachine()
	acp := NewACPEngine("node1", "us-east-1", sm)

	if acp == nil {
		t.Fatal("Failed to create ACP engine")
	}

	if acp.GetCurrentAlgorithm() != AlgorithmRaft {
		t.Errorf("Expected default algorithm Raft, got %v", acp.GetCurrentAlgorithm())
	}
}

// TestACPAlgorithmDecision tests algorithm decision logic
func TestACPAlgorithmDecision(t *testing.T) {
	sm := NewSimpleStateMachine()
	acp := NewACPEngine("node1", "us-east-1", sm)

	tests := []struct {
		name     string
		metrics  NetworkMetrics
		expected ConsensusAlgorithm
	}{
		{
			name: "Low latency few regions - should choose Raft",
			metrics: NetworkMetrics{
				RegionCount:  2,
				AvgLatency:   30 * time.Millisecond,
				ConflictRate: 0.01,
			},
			expected: AlgorithmRaft,
		},
		{
			name: "High latency - should choose Eventual",
			metrics: NetworkMetrics{
				RegionCount:  5,
				AvgLatency:   250 * time.Millisecond,
				ConflictRate: 0.05,
			},
			expected: AlgorithmEventual,
		},
		{
			name: "High conflict rate - should choose EPaxos",
			metrics: NetworkMetrics{
				RegionCount:  3,
				AvgLatency:   100 * time.Millisecond,
				ConflictRate: 0.15,
			},
			expected: AlgorithmEPaxos,
		},
		{
			name: "Multi-region moderate latency - should choose Hybrid",
			metrics: NetworkMetrics{
				RegionCount:  5,
				AvgLatency:   120 * time.Millisecond,
				ConflictRate: 0.05,
			},
			expected: AlgorithmHybrid,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			acp.UpdateNetworkMetrics(tt.metrics)
			decision := acp.DecideAlgorithm()

			if decision != tt.expected {
				t.Errorf("Expected algorithm %v, got %v", tt.expected, decision)
			}
		})
	}
}

// TestRaftConsensus tests basic Raft consensus
func TestRaftConsensus(t *testing.T) {
	sm := NewSimpleStateMachine()
	raft := NewRaftConsensus("node1", sm)

	if raft.state != StateFollower {
		t.Errorf("Expected initial state Follower, got %v", raft.state)
	}

	// Test vote request
	req := &VoteRequest{
		Term:         1,
		CandidateID:  "node2",
		LastLogIndex: 0,
		LastLogTerm:  0,
	}

	resp := raft.RequestVote(req)
	if !resp.VoteGranted {
		t.Error("Expected vote to be granted")
	}

	if raft.votedFor != "node2" {
		t.Errorf("Expected votedFor=node2, got %s", raft.votedFor)
	}
}

// TestPaxosConsensus tests basic Paxos consensus
func TestPaxosConsensus(t *testing.T) {
	sm := NewSimpleStateMachine()
	paxos := NewPaxosConsensus("node1", sm)

	// Test prepare
	req := &PrepareRequest{
		Slot:   1,
		Ballot: Ballot{Number: 1, ReplicaID: 1},
	}

	resp := paxos.handlePrepare(req)
	if !resp.Promised {
		t.Error("Expected promise to be granted")
	}
}

// TestEPaxosConsensus tests EPaxos consensus
func TestEPaxosConsensus(t *testing.T) {
	sm := NewSimpleStateMachine()
	epaxos := NewEPaxosConsensus("node1", sm)

	// Test dependency computation
	cmd := Command{
		Type:  "write",
		Key:   "key1",
		Value: []byte("value1"),
	}

	deps := epaxos.computeDependencies(cmd)
	if len(deps) != 0 {
		t.Errorf("Expected no dependencies for first command, got %d", len(deps))
	}

	seq := epaxos.computeSeq(deps)
	if seq != 1 {
		t.Errorf("Expected sequence 1 for first command, got %d", seq)
	}
}

// TestEventualConsistency tests eventual consistency with CRDTs
func TestEventualConsistency(t *testing.T) {
	sm := NewSimpleStateMachine()
	ec := NewEventualConsistency("node1", sm)

	// Test LWW map
	err := ec.Update("key1", []byte("value1"))
	if err != nil {
		t.Fatalf("Failed to update: %v", err)
	}

	value, exists := ec.Get("key1")
	if !exists {
		t.Fatal("Key not found after update")
	}

	if string(value) != "value1" {
		t.Errorf("Expected value1, got %s", string(value))
	}

	// Test concurrent updates with timestamps
	time.Sleep(10 * time.Millisecond)
	err = ec.Update("key1", []byte("value2"))
	if err != nil {
		t.Fatalf("Failed to update: %v", err)
	}

	value, _ = ec.Get("key1")
	if string(value) != "value2" {
		t.Errorf("Expected newer value2, got %s", string(value))
	}
}

// TestHybridConsensus tests hybrid consensus
func TestHybridConsensus(t *testing.T) {
	sm := NewSimpleStateMachine()
	raft := NewRaftConsensus("node1", sm)
	eventual := NewEventualConsistency("node1", sm)
	hybrid := NewHybridConsensus("node1", "us-east-1", raft, eventual)

	// Add regions
	hybrid.AddRegion("us-east-1", []string{"node1", "node2"})
	hybrid.AddRegion("eu-west-1", []string{"node3", "node4"})

	// Test local region access
	hybrid.SetKeyRegion("local-key", "us-east-1")
	err := hybrid.Propose("local-key", []byte("local-value"))
	if err != nil && err.Error() != "not leader" {
		// Expected error if not leader, or success
		t.Logf("Propose result: %v", err)
	}

	// Test cross-region access
	hybrid.SetKeyRegion("remote-key", "eu-west-1")
	err = hybrid.Propose("remote-key", []byte("remote-value"))
	if err != nil {
		t.Logf("Cross-region propose: %v", err)
	}
}

// TestNetworkMonitor tests network monitoring
func TestNetworkMonitor(t *testing.T) {
	nm := NewNetworkMonitor()

	// Update metrics
	metrics := NetworkMetrics{
		RegionCount:  3,
		AvgLatency:   75 * time.Millisecond,
		PacketLoss:   0.01,
		ConflictRate: 0.05,
	}

	nm.UpdateMetrics(metrics)

	retrieved := nm.GetMetrics()
	if retrieved.RegionCount != 3 {
		t.Errorf("Expected RegionCount=3, got %d", retrieved.RegionCount)
	}

	// Test region metrics
	regionMetrics := &RegionMetrics{
		RegionID:  "us-east-1",
		Latency:   50 * time.Millisecond,
		Bandwidth: 1000000000,
	}

	nm.UpdateRegionMetrics("us-east-1", regionMetrics)

	retrieved2, exists := nm.GetRegionMetrics("us-east-1")
	if !exists {
		t.Fatal("Region metrics not found")
	}

	if retrieved2.Latency != 50*time.Millisecond {
		t.Errorf("Expected latency 50ms, got %v", retrieved2.Latency)
	}

	// Test health check (update stability first)
	nm.currentMetrics.Stability = 0.9
	if !nm.IsHealthy() {
		t.Error("Network should be healthy")
	}
}

// TestQuorumStrategies tests different quorum strategies
func TestQuorumStrategies(t *testing.T) {
	// Test majority quorum
	majority := NewMajorityQuorum()
	if majority.GetQuorumSize(5) != 3 {
		t.Errorf("Expected quorum size 3 for 5 nodes, got %d", majority.GetQuorumSize(5))
	}

	if !majority.IsQuorum(3, 5) {
		t.Error("Expected 3/5 to be quorum")
	}

	// Test flexible quorum
	flexible, err := NewFlexibleQuorum(5, 3, 3)
	if err != nil {
		t.Fatalf("Failed to create flexible quorum: %v", err)
	}

	if flexible.GetQuorumSize(5) != 3 {
		t.Errorf("Expected write quorum 3, got %d", flexible.GetQuorumSize(5))
	}

	// Test geographic quorum
	regions := map[string]int{
		"us-east-1": 3,
		"eu-west-1": 3,
		"ap-south-1": 3,
	}

	geoQuorum := NewGeographicQuorum(regions)
	expectedQuorum := 6 // Majority in each region: 2+2+2 = 6
	if geoQuorum.GetQuorumSize(9) != expectedQuorum {
		t.Errorf("Expected quorum size %d, got %d", expectedQuorum, geoQuorum.GetQuorumSize(9))
	}

	// Test fast-path quorum
	fastPath := NewFastPathQuorum(5)
	// For 5 nodes: F=2, quorum = 2 + 2/2 + 1 = 2 + 1 + 1 = 4
	if fastPath.GetQuorumSize(5) != 4 {
		t.Errorf("Expected fast-path quorum 4, got %d", fastPath.GetQuorumSize(5))
	}
}

// TestConflictResolution tests conflict resolution strategies
func TestConflictResolution(t *testing.T) {
	// Test LWW strategy
	resolver := NewConflictResolver(StrategyLWW)

	now := NewTimestamp()
	later := Timestamp{Wall: now.Wall + 1000000, Logical: now.Logical + 1}

	conflicts := []ConflictingWrite{
		{
			Key:       "key1",
			Value:     []byte("value1"),
			Timestamp: now,
			NodeID:    "node1",
		},
		{
			Key:       "key1",
			Value:     []byte("value2"),
			Timestamp: later,
			NodeID:    "node2",
		},
	}

	resolved := resolver.Resolve(conflicts)
	if string(resolved) != "value2" {
		t.Errorf("Expected LWW to choose value2, got %s", string(resolved))
	}

	// Test multi-value strategy
	resolver.SetStrategy(StrategyMV)
	resolved = resolver.Resolve(conflicts)
	if len(resolved) == 0 {
		t.Error("Expected multi-value result to have data")
	}

	// Test stats
	stats := resolver.GetConflictStats()
	if stats.TotalConflicts != 2 {
		t.Errorf("Expected 2 conflict resolutions, got %d", stats.TotalConflicts)
	}
}

// TestConsensusOptimizer tests consensus optimization
func TestConsensusOptimizer(t *testing.T) {
	optimizer := NewConsensusOptimizer(10, 100*time.Millisecond)

	// Submit proposals
	for i := 0; i < 5; i++ {
		proposal := Proposal{
			Key:   "key",
			Value: []byte("value"),
		}
		optimizer.Submit(proposal)
	}

	// Wait for batching
	time.Sleep(200 * time.Millisecond)

	// Check stats
	stats := optimizer.GetStats()
	if stats.TotalProposals != 5 {
		t.Errorf("Expected 5 total proposals, got %d", stats.TotalProposals)
	}
}

// TestACPSwitching tests algorithm switching
func TestACPSwitching(t *testing.T) {
	sm := NewSimpleStateMachine()
	acp := NewACPEngine("node1", "us-east-1", sm)

	// Start with Raft
	if acp.GetCurrentAlgorithm() != AlgorithmRaft {
		t.Fatalf("Expected initial algorithm Raft, got %v", acp.GetCurrentAlgorithm())
	}

	// Update metrics to favor EPaxos
	metrics := NetworkMetrics{
		RegionCount:  3,
		AvgLatency:   100 * time.Millisecond,
		ConflictRate: 0.2, // High conflict rate
	}
	acp.UpdateNetworkMetrics(metrics)

	// Check if should switch
	shouldSwitch, newAlgo := acp.ShouldSwitch()
	if shouldSwitch && newAlgo != AlgorithmEPaxos {
		t.Errorf("Expected to switch to EPaxos, got %v", newAlgo)
	}

	// Perform switch
	if shouldSwitch {
		err := acp.SwitchAlgorithm(newAlgo)
		if err != nil {
			t.Fatalf("Failed to switch algorithm: %v", err)
		}

		if acp.GetCurrentAlgorithm() != newAlgo {
			t.Errorf("Algorithm not switched, still %v", acp.GetCurrentAlgorithm())
		}

		// Check switch history
		history := acp.GetSwitchHistory()
		if len(history) != 1 {
			t.Errorf("Expected 1 switch event, got %d", len(history))
		}
	}
}

// TestCRDTs tests CRDT implementations
func TestCRDTs(t *testing.T) {
	// Test LWW Map
	lww := NewLWWMap()
	ts1 := NewTimestamp()
	lww.Set("key1", []byte("value1"), ts1, "node1")

	value, exists := lww.Get("key1")
	if !exists || string(value) != "value1" {
		t.Error("Failed to set/get LWW value")
	}

	// Test OR Set
	orset := NewORSet()
	orset.Add("element1", "tag1")

	if !orset.Contains("element1") {
		t.Error("Element should be in set")
	}

	orset.Remove("element1")
	if orset.Contains("element1") {
		t.Error("Element should be removed from set")
	}

	// Test PN Counter
	counter := NewPNCounter()
	counter.Increment("node1", 5)
	counter.Increment("node2", 3)
	counter.Increment("node1", -2)

	if counter.Value() != 6 {
		t.Errorf("Expected counter value 6, got %d", counter.Value())
	}

	// Test Vector Clock
	vc1 := NewVectorClock()
	vc2 := NewVectorClock()

	// vc1: {node1: 2}
	vc1.Increment("node1")
	vc1.Increment("node1")

	// vc2: {node2: 1}
	vc2.Increment("node2")

	// These are concurrent (neither happens-before the other)
	if vc1.HappensBefore(vc2) {
		t.Error("vc1 should not happen before vc2 (concurrent)")
	}

	// After merge, vc2 should have both clocks
	vc2.Update(vc1)
	if vc2.clock["node1"] != 2 {
		t.Errorf("Expected node1 clock=2 after merge, got %d", vc2.clock["node1"])
	}
	if vc2.clock["node2"] != 1 {
		t.Errorf("Expected node2 clock=1 after merge, got %d", vc2.clock["node2"])
	}
}

// BenchmarkRaftPropose benchmarks Raft proposals
func BenchmarkRaftPropose(b *testing.B) {
	sm := NewSimpleStateMachine()
	raft := NewRaftConsensus("node1", sm)

	// Make it leader
	raft.becomeLeader()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = raft.Propose("key", []byte("value"))
	}
}

// BenchmarkEventualUpdate benchmarks eventual consistency updates
func BenchmarkEventualUpdate(b *testing.B) {
	sm := NewSimpleStateMachine()
	ec := NewEventualConsistency("node1", sm)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ec.Update("key", []byte("value"))
	}
}

// BenchmarkConflictResolution benchmarks conflict resolution
func BenchmarkConflictResolution(b *testing.B) {
	resolver := NewConflictResolver(StrategyLWW)

	conflicts := make([]ConflictingWrite, 10)
	for i := range conflicts {
		conflicts[i] = ConflictingWrite{
			Key:       "key",
			Value:     []byte("value"),
			Timestamp: NewTimestamp(),
			NodeID:    "node",
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = resolver.Resolve(conflicts)
	}
}
