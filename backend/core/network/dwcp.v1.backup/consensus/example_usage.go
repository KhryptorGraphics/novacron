package consensus

import (
	"fmt"
	"log"
	"time"
)

// ExampleBasicUsage demonstrates basic ACP usage
func ExampleBasicUsage() {
	// Create state machine
	sm := NewSimpleStateMachine()

	// Create ACP engine
	acp := NewACPEngine("node1", "us-east-1", sm)

	// Start adaptive monitoring
	go acp.MonitorAndAdapt()

	// Submit proposals
	for i := 0; i < 10; i++ {
		key := fmt.Sprintf("user:%d", i)
		value := []byte(fmt.Sprintf(`{"id":%d,"name":"User%d"}`, i, i))

		err := acp.Propose(key, value)
		if err != nil {
			log.Printf("Proposal failed: %v", err)
		}
	}

	// Get current algorithm
	algo := acp.GetCurrentAlgorithm()
	fmt.Printf("Current algorithm: %v\n", algo)
}

// ExampleNetworkAdaptation demonstrates network-based adaptation
func ExampleNetworkAdaptation() {
	sm := NewSimpleStateMachine()
	acp := NewACPEngine("node1", "us-east-1", sm)

	// Simulate changing network conditions
	scenarios := []struct {
		name    string
		metrics NetworkMetrics
	}{
		{
			name: "Low latency single region",
			metrics: NetworkMetrics{
				RegionCount:  1,
				AvgLatency:   25 * time.Millisecond,
				ConflictRate: 0.01,
			},
		},
		{
			name: "Multi-region moderate latency",
			metrics: NetworkMetrics{
				RegionCount:  3,
				AvgLatency:   100 * time.Millisecond,
				ConflictRate: 0.05,
			},
		},
		{
			name: "High conflict workload",
			metrics: NetworkMetrics{
				RegionCount:  2,
				AvgLatency:   80 * time.Millisecond,
				ConflictRate: 0.15,
			},
		},
		{
			name: "Geo-distributed high latency",
			metrics: NetworkMetrics{
				RegionCount:  5,
				AvgLatency:   250 * time.Millisecond,
				ConflictRate: 0.03,
			},
		},
	}

	for _, scenario := range scenarios {
		fmt.Printf("\n=== Scenario: %s ===\n", scenario.name)

		// Update network metrics
		acp.UpdateNetworkMetrics(scenario.metrics)

		// Check recommended algorithm
		recommended := acp.DecideAlgorithm()
		fmt.Printf("Recommended algorithm: %v\n", recommended)

		// Check if should switch
		shouldSwitch, newAlgo := acp.ShouldSwitch()
		if shouldSwitch {
			fmt.Printf("Switching from %v to %v\n",
				acp.GetCurrentAlgorithm(), newAlgo)

			err := acp.SwitchAlgorithm(newAlgo)
			if err != nil {
				log.Printf("Switch failed: %v", err)
			}
		}

		time.Sleep(100 * time.Millisecond)
	}

	// Print switch history
	history := acp.GetSwitchHistory()
	fmt.Printf("\nSwitch history (%d events):\n", len(history))
	for _, event := range history {
		fmt.Printf("  %v -> %v (benefit: %.2f, cost: %.2f)\n",
			event.From, event.To, event.Benefit, event.Cost)
	}
}

// ExampleHybridDeployment demonstrates hybrid consensus for multi-region
func ExampleHybridDeployment() {
	// Setup state machine
	sm := NewSimpleStateMachine()

	// Create consensus components
	raft := NewRaftConsensus("node1", sm)
	eventual := NewEventualConsistency("node1", sm)
	hybrid := NewHybridConsensus("node1", "us-east-1", raft, eventual)

	// Configure regions
	hybrid.AddRegion("us-east-1", []string{"node1", "node2", "node3"})
	hybrid.AddRegion("eu-west-1", []string{"node4", "node5", "node6"})
	hybrid.AddRegion("ap-south-1", []string{"node7", "node8", "node9"})

	// Update region health
	hybrid.UpdateRegionHealth("us-east-1", true, 20)
	hybrid.UpdateRegionHealth("eu-west-1", true, 85)
	hybrid.UpdateRegionHealth("ap-south-1", true, 120)

	// Partition keys by region
	keys := map[string]string{
		"us-users":   "us-east-1",
		"eu-users":   "eu-west-1",
		"asia-users": "ap-south-1",
		"global-config": "us-east-1",
	}

	for key, region := range keys {
		hybrid.SetKeyRegion(key, region)
		fmt.Printf("Key %s assigned to region %s\n", key, region)
	}

	// Perform operations
	fmt.Println("\n=== Performing operations ===")

	// Local write (strong consistency via Raft)
	err := hybrid.Propose("us-users", []byte("user data"))
	if err != nil {
		log.Printf("Local write: %v", err)
	} else {
		fmt.Println("✓ Local write completed (Raft)")
	}

	// Cross-region write (eventual consistency)
	hybrid.SetKeyRegion("cross-region-data", "eu-west-1")
	err = hybrid.Propose("cross-region-data", []byte("shared data"))
	if err != nil {
		log.Printf("Cross-region write: %v", err)
	} else {
		fmt.Println("✓ Cross-region write completed (Eventual)")
	}

	// Sync across regions
	fmt.Println("\n=== Syncing regions ===")
	err = hybrid.SyncCrossRegion()
	if err != nil {
		log.Printf("Sync failed: %v", err)
	} else {
		fmt.Println("✓ Cross-region sync completed")
	}

	// Check region load
	load := hybrid.GetRegionLoad()
	fmt.Println("\n=== Region load ===")
	for region, count := range load {
		fmt.Printf("  %s: %d keys\n", region, count)
	}

	// Rebalance if needed
	fmt.Println("\n=== Rebalancing ===")
	err = hybrid.RebalanceKeys("eu-west-1", 2)
	if err != nil {
		log.Printf("Rebalance failed: %v", err)
	} else {
		fmt.Println("✓ Rebalanced 2 keys to eu-west-1")
	}
}

// ExampleConflictResolution demonstrates conflict resolution
func ExampleConflictResolution() {
	// Create resolver
	resolver := NewConflictResolver(StrategyLWW)

	// Simulate concurrent writes
	now := NewTimestamp()
	now.NodeID = "node1"

	later := NewTimestamp()
	later.Wall = now.Wall + 1000000000 // 1 second later
	later.Logical = now.Logical + 1
	later.NodeID = "node2"

	conflicts := []ConflictingWrite{
		{
			Key:       "document:123",
			Value:     []byte(`{"title":"Original","version":1}`),
			Timestamp: now,
			NodeID:    "node1",
			Version:   1,
		},
		{
			Key:       "document:123",
			Value:     []byte(`{"title":"Updated","version":2}`),
			Timestamp: later,
			NodeID:    "node2",
			Version:   2,
		},
	}

	fmt.Println("=== Last-Write-Wins Resolution ===")
	resolved := resolver.Resolve(conflicts)
	fmt.Printf("Resolved value: %s\n", string(resolved))

	// Try multi-value strategy
	fmt.Println("\n=== Multi-Value Resolution ===")
	resolver.SetStrategy(StrategyMV)
	resolved = resolver.Resolve(conflicts)
	fmt.Printf("Multi-value result: %s\n", string(resolved))

	// Custom resolution
	fmt.Println("\n=== Custom Resolution ===")
	resolver.SetStrategy(StrategyCustom)
	resolver.SetCustomResolver(func(conflicts []ConflictingWrite) []byte {
		// Custom logic: prefer higher version
		var highest ConflictingWrite
		for _, c := range conflicts {
			if c.Version > highest.Version {
				highest = c
			}
		}
		return highest.Value
	})
	resolved = resolver.Resolve(conflicts)
	fmt.Printf("Custom resolved: %s\n", string(resolved))

	// Get statistics
	stats := resolver.GetConflictStats()
	fmt.Printf("\n=== Resolution Stats ===\n")
	fmt.Printf("Total conflicts: %d\n", stats.TotalConflicts)
	fmt.Printf("Avg resolution time: %d ns\n", stats.AvgResolutionTime)
}

// ExamplePerformanceOptimization demonstrates optimization features
func ExamplePerformanceOptimization() {
	// Create optimizer
	optimizer := NewConsensusOptimizer(100, 10*time.Millisecond)
	optimizer.Start()
	defer optimizer.Stop()

	fmt.Println("=== Submitting proposals ===")

	// Submit batch of proposals
	start := time.Now()
	for i := 0; i < 1000; i++ {
		proposal := Proposal{
			ID:    fmt.Sprintf("prop-%d", i),
			Key:   fmt.Sprintf("key-%d", i),
			Value: []byte(fmt.Sprintf("value-%d", i)),
		}
		optimizer.Submit(proposal)
	}

	// Wait for processing
	time.Sleep(200 * time.Millisecond)

	duration := time.Since(start)
	fmt.Printf("Submitted 1000 proposals in %v\n", duration)

	// Get statistics
	stats := optimizer.GetStats()
	fmt.Printf("\n=== Optimizer Statistics ===\n")
	fmt.Printf("Total proposals: %d\n", stats.TotalProposals)
	fmt.Printf("Batched proposals: %d\n", stats.BatchedProposals)
	fmt.Printf("Batch count: %d\n", stats.BatchCount)
	fmt.Printf("Avg batch size: %.2f\n", stats.AvgBatchSize)
	fmt.Printf("Throughput: %.2f ops/sec\n", stats.ThroughputOps)
	fmt.Printf("Avg latency: %v\n", stats.AvgLatency)

	// Optimize batch size
	fmt.Println("\n=== Auto-optimizing batch size ===")
	optimizer.OptimizeBatchSize()
	fmt.Println("✓ Batch size optimized based on throughput")
}

// ExampleCRDTOperations demonstrates CRDT usage
func ExampleCRDTOperations() {
	fmt.Println("=== CRDT Examples ===")

	// LWW Map
	fmt.Println("\n--- LWW Map ---")
	lww := NewLWWMap()

	ts1 := NewTimestamp()
	lww.Set("user:name", []byte("Alice"), ts1, "node1")

	time.Sleep(10 * time.Millisecond)
	ts2 := NewTimestamp()
	lww.Set("user:name", []byte("Bob"), ts2, "node2")

	value, _ := lww.Get("user:name")
	fmt.Printf("LWW result: %s (newer timestamp wins)\n", string(value))

	// OR Set
	fmt.Println("\n--- OR Set ---")
	orset := NewORSet()

	orset.Add("item1", "tag1")
	orset.Add("item2", "tag2")
	orset.Add("item3", "tag3")

	fmt.Printf("Contains item1: %v\n", orset.Contains("item1"))

	orset.Remove("item2")
	fmt.Printf("Contains item2 (after remove): %v\n", orset.Contains("item2"))

	// PN Counter
	fmt.Println("\n--- PN Counter ---")
	counter := NewPNCounter()

	counter.Increment("node1", 10)
	counter.Increment("node2", 5)
	counter.Increment("node3", -3)

	fmt.Printf("Counter value: %d\n", counter.Value())

	// Vector Clock
	fmt.Println("\n--- Vector Clock ---")
	vc1 := NewVectorClock()
	vc2 := NewVectorClock()

	vc1.Increment("node1")
	vc1.Increment("node1")
	fmt.Printf("VC1: %v\n", vc1.clock)

	vc2.Increment("node2")
	fmt.Printf("VC2: %v\n", vc2.clock)

	fmt.Printf("VC1 happens before VC2: %v\n", vc1.HappensBefore(vc2))

	vc2.Update(vc1)
	fmt.Printf("VC2 after merge: %v\n", vc2.clock)
}

// ExampleQuorumStrategies demonstrates different quorum configurations
func ExampleQuorumStrategies() {
	fmt.Println("=== Quorum Strategies ===")

	totalNodes := 5

	// Majority quorum
	fmt.Println("\n--- Majority Quorum ---")
	majority := NewMajorityQuorum()
	quorum := majority.GetQuorumSize(totalNodes)
	fmt.Printf("Quorum for %d nodes: %d\n", totalNodes, quorum)
	fmt.Printf("Is 3/%d quorum: %v\n", totalNodes, majority.IsQuorum(3, totalNodes))

	// Flexible quorum
	fmt.Println("\n--- Flexible Quorum ---")
	flexible, _ := NewFlexibleQuorum(totalNodes, 3, 3)
	fmt.Printf("Write quorum: %d, Read quorum: %d\n",
		flexible.WriteQuorum, flexible.ReadQuorum)

	// Geographic quorum
	fmt.Println("\n--- Geographic Quorum ---")
	regions := map[string]int{
		"us-east-1":  3,
		"eu-west-1":  3,
		"ap-south-1": 3,
	}
	geoQuorum := NewGeographicQuorum(regions)
	fmt.Printf("Total quorum for %d regions: %d\n",
		len(regions), geoQuorum.GetQuorumSize(9))

	// Fast-path quorum (EPaxos)
	fmt.Println("\n--- Fast-Path Quorum ---")
	fastPath := NewFastPathQuorum(totalNodes)
	fmt.Printf("Fast-path quorum: %d\n", fastPath.GetQuorumSize(totalNodes))
	fmt.Printf("Slow-path quorum: %d\n", fastPath.GetSlowPathQuorumSize(totalNodes))
}
