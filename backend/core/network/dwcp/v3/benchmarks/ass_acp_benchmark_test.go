package benchmarks

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkASSRaftConsensusLatency tests Raft consensus performance
func BenchmarkASSRaftConsensusLatency(b *testing.B) {
	scenarios := []struct {
		name      string
		nodes     int
		valueSize int
	}{
		{"3Nodes_1KB", 3, 1024},
		{"5Nodes_1KB", 5, 1024},
		{"7Nodes_1KB", 7, 1024},
		{"3Nodes_64KB", 3, 65536},
		{"5Nodes_64KB", 5, 65536},
		{"7Nodes_64KB", 7, 65536},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkRaftConsensus(b, sc.nodes, sc.valueSize)
		})
	}
}

func benchmarkRaftConsensus(b *testing.B, nodeCount int, valueSize int) {
	b.ReportAllocs()

	cluster := newMockRaftCluster(nodeCount)
	value := make([]byte, valueSize)
	rand.Read(value)

	var totalLatency time.Duration
	var commitCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		// Simulate Raft consensus process
		cluster.propose(value)

		totalLatency += time.Since(start)
		atomic.AddInt64(&commitCount, 1)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Microseconds()) / float64(b.N) / 1000.0
	commitsPerSecond := float64(commitCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyMs, "ms/commit")
	b.ReportMetric(commitsPerSecond, "commits/sec")
}

// BenchmarkACPPBFTConsensusThroughput tests PBFT consensus throughput
func BenchmarkACPPBFTConsensusThroughput(b *testing.B) {
	scenarios := []struct {
		name       string
		nodes      int
		batchSize  int
		concurrent int
	}{
		{"4Nodes_Batch1_Seq", 4, 1, 1},
		{"4Nodes_Batch10_Seq", 4, 10, 1},
		{"4Nodes_Batch1_Parallel10", 4, 1, 10},
		{"7Nodes_Batch1_Seq", 7, 1, 1},
		{"7Nodes_Batch10_Seq", 7, 10, 1},
		{"7Nodes_Batch1_Parallel10", 7, 1, 10},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkPBFTConsensus(b, sc.nodes, sc.batchSize, sc.concurrent)
		})
	}
}

func benchmarkPBFTConsensus(b *testing.B, nodeCount, batchSize, concurrent int) {
	b.ReportAllocs()

	cluster := newMockPBFTCluster(nodeCount)

	var totalOperations int64
	startTime := time.Now()

	b.ResetTimer()

	var wg sync.WaitGroup
	for i := 0; i < concurrent; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < b.N/concurrent; j++ {
				cluster.processBatch(batchSize)
				atomic.AddInt64(&totalOperations, int64(batchSize))
			}
		}()
	}

	wg.Wait()

	b.StopTimer()

	duration := time.Since(startTime)
	throughput := float64(totalOperations) / duration.Seconds()

	b.ReportMetric(throughput, "ops/sec")
	b.ReportMetric(float64(batchSize), "batch_size")
}

// BenchmarkASSStateSyncPerformance tests state synchronization performance
func BenchmarkASSStateSyncPerformance(b *testing.B) {
	scenarios := []struct {
		name      string
		stateSize int64
		nodes     int
	}{
		{"1MB_3Nodes", 1048576, 3},
		{"10MB_3Nodes", 10485760, 3},
		{"100MB_3Nodes", 104857600, 3},
		{"1MB_7Nodes", 1048576, 7},
		{"10MB_7Nodes", 10485760, 7},
		{"100MB_7Nodes", 104857600, 7},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkStateSync(b, sc.stateSize, sc.nodes)
		})
	}
}

func benchmarkStateSync(b *testing.B, stateSize int64, nodeCount int) {
	b.ReportAllocs()

	state := make([]byte, stateSize)
	rand.Read(state)

	cluster := newMockRaftCluster(nodeCount)

	var totalBytes int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		cluster.syncState(state)

		totalLatency += time.Since(start)
		atomic.AddInt64(&totalBytes, stateSize*int64(nodeCount-1))
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Milliseconds()) / float64(b.N)
	throughputMBps := float64(totalBytes) / b.Elapsed().Seconds() / 1e6

	b.ReportMetric(avgLatencyMs, "ms/sync")
	b.ReportMetric(throughputMBps, "MB/s")
}

// BenchmarkACPConflictResolution tests conflict resolution performance
func BenchmarkACPConflictResolution(b *testing.B) {
	scenarios := []struct {
		name           string
		conflictRate   float64
		conflictSize   int
	}{
		{"LowConflict_Small", 0.05, 100},
		{"MediumConflict_Small", 0.20, 100},
		{"HighConflict_Small", 0.50, 100},
		{"LowConflict_Large", 0.05, 10000},
		{"MediumConflict_Large", 0.20, 10000},
		{"HighConflict_Large", 0.50, 10000},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkConflictResolution(b, sc.conflictRate, sc.conflictSize)
		})
	}
}

func benchmarkConflictResolution(b *testing.B, conflictRate float64, stateSize int) {
	b.ReportAllocs()

	// Generate conflicting states
	state1 := generateRandomState(stateSize)
	state2 := generateRandomState(stateSize)

	// Introduce conflicts
	conflictCount := int(float64(stateSize) * conflictRate)
	for i := 0; i < conflictCount; i++ {
		idx := fmt.Sprintf("key_%d", rand.Intn(stateSize))
		if val, ok := state1[idx].(int); ok {
			state2[idx] = val ^ 0xFF
		}
	}

	var resolutionCount int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		_ = resolveConflicts(state1, state2)

		totalLatency += time.Since(start)
		atomic.AddInt64(&resolutionCount, 1)
	}

	b.StopTimer()

	avgLatencyUs := float64(totalLatency.Microseconds()) / float64(b.N)
	resolutionsPerSecond := float64(resolutionCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyUs, "us/resolution")
	b.ReportMetric(resolutionsPerSecond, "resolutions/sec")
}

// BenchmarkASSLeaderElection tests leader election performance
func BenchmarkASSLeaderElection(b *testing.B) {
	nodeCounts := []int{3, 5, 7, 9, 11}

	for _, nodeCount := range nodeCounts {
		b.Run(fmt.Sprintf("%dNodes", nodeCount), func(b *testing.B) {
			benchmarkLeaderElection(b, nodeCount)
		})
	}
}

func benchmarkLeaderElection(b *testing.B, nodeCount int) {
	b.ReportAllocs()

	cluster := newMockRaftCluster(nodeCount)

	var totalLatency time.Duration
	var electionCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		cluster.electLeader()

		totalLatency += time.Since(start)
		atomic.AddInt64(&electionCount, 1)
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Milliseconds()) / float64(b.N)
	electionsPerSecond := float64(electionCount) / b.Elapsed().Seconds()

	b.ReportMetric(avgLatencyMs, "ms/election")
	b.ReportMetric(electionsPerSecond, "elections/sec")
}

// BenchmarkACPQuorumVerification tests quorum verification performance
func BenchmarkACPQuorumVerification(b *testing.B) {
	scenarios := []struct {
		name         string
		totalNodes   int
		responseRate float64
	}{
		{"3Nodes_Full", 3, 1.0},
		{"3Nodes_Quorum", 3, 0.67},
		{"7Nodes_Full", 7, 1.0},
		{"7Nodes_Quorum", 7, 0.57},
		{"11Nodes_Full", 11, 1.0},
		{"11Nodes_Quorum", 11, 0.55},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkQuorumVerification(b, sc.totalNodes, sc.responseRate)
		})
	}
}

func benchmarkQuorumVerification(b *testing.B, totalNodes int, responseRate float64) {
	b.ReportAllocs()

	responses := int(float64(totalNodes) * responseRate)
	votes := make([]bool, responses)
	for i := range votes {
		votes[i] = true
	}

	var verificationCount int64

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		hasQuorum := verifyQuorum(votes, totalNodes)
		_ = hasQuorum
		atomic.AddInt64(&verificationCount, 1)
	}

	b.StopTimer()

	verificationsPerSecond := float64(verificationCount) / b.Elapsed().Seconds()

	b.ReportMetric(verificationsPerSecond, "verifications/sec")
	b.ReportMetric(float64(totalNodes), "total_nodes")
}

// BenchmarkASSMessageBroadcast tests message broadcast performance
func BenchmarkASSMessageBroadcast(b *testing.B) {
	scenarios := []struct {
		name        string
		nodes       int
		messageSize int
		parallel    bool
	}{
		{"3Nodes_1KB_Sequential", 3, 1024, false},
		{"7Nodes_1KB_Sequential", 7, 1024, false},
		{"11Nodes_1KB_Sequential", 11, 1024, false},
		{"3Nodes_1KB_Parallel", 3, 1024, true},
		{"7Nodes_1KB_Parallel", 7, 1024, true},
		{"11Nodes_1KB_Parallel", 11, 1024, true},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkMessageBroadcast(b, sc.nodes, sc.messageSize, sc.parallel)
		})
	}
}

func benchmarkMessageBroadcast(b *testing.B, nodeCount, messageSize int, parallel bool) {
	b.ReportAllocs()

	message := make([]byte, messageSize)
	rand.Read(message)

	var totalBytes int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		if parallel {
			var wg sync.WaitGroup
			for j := 0; j < nodeCount; j++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					// Simulate send
					time.Sleep(100 * time.Microsecond)
				}()
			}
			wg.Wait()
		} else {
			for j := 0; j < nodeCount; j++ {
				// Simulate send
				time.Sleep(100 * time.Microsecond)
			}
		}

		totalLatency += time.Since(start)
		atomic.AddInt64(&totalBytes, int64(messageSize*nodeCount))
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Microseconds()) / float64(b.N) / 1000.0
	throughputMBps := float64(totalBytes) / b.Elapsed().Seconds() / 1e6

	b.ReportMetric(avgLatencyMs, "ms/broadcast")
	b.ReportMetric(throughputMBps, "MB/s")
}

// BenchmarkACPByzantineFaultTolerance tests Byzantine fault handling
func BenchmarkACPByzantineFaultTolerance(b *testing.B) {
	scenarios := []struct {
		name         string
		nodes        int
		faultyNodes  int
	}{
		{"4Nodes_0Faulty", 4, 0},
		{"4Nodes_1Faulty", 4, 1},
		{"7Nodes_0Faulty", 7, 0},
		{"7Nodes_2Faulty", 7, 2},
		{"10Nodes_0Faulty", 10, 0},
		{"10Nodes_3Faulty", 10, 3},
	}

	for _, sc := range scenarios {
		b.Run(sc.name, func(b *testing.B) {
			benchmarkByzantineFaultTolerance(b, sc.nodes, sc.faultyNodes)
		})
	}
}

func benchmarkByzantineFaultTolerance(b *testing.B, totalNodes, faultyNodes int) {
	b.ReportAllocs()

	cluster := newMockPBFTCluster(totalNodes)
	cluster.setFaultyNodes(faultyNodes)

	var successfulOperations int64
	var totalLatency time.Duration

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		success := cluster.processBatch(1)

		totalLatency += time.Since(start)
		if success {
			atomic.AddInt64(&successfulOperations, 1)
		}
	}

	b.StopTimer()

	avgLatencyMs := float64(totalLatency.Milliseconds()) / float64(b.N)
	successRate := float64(successfulOperations) / float64(b.N) * 100

	b.ReportMetric(avgLatencyMs, "ms/op")
	b.ReportMetric(successRate, "success_%")
}

// Helper types and functions

type mockRaftCluster struct {
	nodeCount int
	leader    int
	mu        sync.Mutex
}

func newMockRaftCluster(nodeCount int) *mockRaftCluster {
	return &mockRaftCluster{
		nodeCount: nodeCount,
		leader:    0,
	}
}

func (c *mockRaftCluster) propose(value []byte) {
	// Simulate Raft propose: leader replication + quorum wait
	quorum := c.nodeCount/2 + 1

	// Simulate network latency to replicas
	for i := 0; i < quorum-1; i++ {
		time.Sleep(200 * time.Microsecond)
	}
}

func (c *mockRaftCluster) syncState(state []byte) {
	// Simulate state sync to all followers
	for i := 0; i < c.nodeCount-1; i++ {
		// Simulate transfer latency
		time.Sleep(time.Duration(len(state)/1000) * time.Microsecond)
	}
}

func (c *mockRaftCluster) electLeader() {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Simulate election timeout + voting round
	time.Sleep(time.Duration(c.nodeCount) * 500 * time.Microsecond)

	c.leader = rand.Intn(c.nodeCount)
}

type mockPBFTCluster struct {
	nodeCount   int
	faultyNodes int
	mu          sync.Mutex
}

func newMockPBFTCluster(nodeCount int) *mockPBFTCluster {
	return &mockPBFTCluster{
		nodeCount:   nodeCount,
		faultyNodes: 0,
	}
}

func (c *mockPBFTCluster) setFaultyNodes(count int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.faultyNodes = count
}

func (c *mockPBFTCluster) processBatch(batchSize int) bool {
	// Simulate PBFT phases: pre-prepare, prepare, commit
	// Each phase needs 2f+1 responses

	requiredResponses := 2*(c.nodeCount/3) + 1
	availableHonestNodes := c.nodeCount - c.faultyNodes

	if availableHonestNodes < requiredResponses {
		return false
	}

	// Simulate three-phase protocol
	for phase := 0; phase < 3; phase++ {
		time.Sleep(time.Duration(c.nodeCount*100) * time.Microsecond)
	}

	return true
}

func generateRandomState(size int) map[string]interface{} {
	state := make(map[string]interface{}, size)
	for i := 0; i < size; i++ {
		key := fmt.Sprintf("key_%d", i)
		state[key] = rand.Int()
	}
	return state
}

func resolveConflicts(state1, state2 map[string]interface{}) map[string]interface{} {
	resolved := make(map[string]interface{})

	// Last-write-wins strategy
	for k, v := range state1 {
		resolved[k] = v
	}

	for k, v := range state2 {
		resolved[k] = v
	}

	return resolved
}

func verifyQuorum(votes []bool, totalNodes int) bool {
	quorum := totalNodes/2 + 1
	positiveVotes := 0

	for _, vote := range votes {
		if vote {
			positiveVotes++
		}
	}

	return positiveVotes >= quorum
}
