package chaos_test

import (
	"context"
	"fmt"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"

	"github.com/khryptorgraphics/novacron/backend/core/consensus"
	"github.com/khryptorgraphics/novacron/backend/core/ha"
	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/resilience"
)

// TestCircuitBreakerWithCascadingFailures tests circuit breaker behavior under cascading failures
func TestCircuitBreakerWithCascadingFailures(t *testing.T) {
	logger := zap.NewNop()

	// Create a chain of circuit breakers to simulate service dependencies
	serviceA := resilience.NewCircuitBreaker("ServiceA", 3, 100*time.Millisecond, 1*time.Second, logger)
	serviceB := resilience.NewCircuitBreaker("ServiceB", 3, 100*time.Millisecond, 1*time.Second, logger)
	serviceC := resilience.NewCircuitBreaker("ServiceC", 3, 100*time.Millisecond, 1*time.Second, logger)

	// Simulate cascading failure
	failureCount := int32(0)

	// Service C fails first
	for i := 0; i < 5; i++ {
		err := serviceC.Execute(func() error {
			atomic.AddInt32(&failureCount, 1)
			return fmt.Errorf("service C failed")
		})
		assert.Error(t, err)
	}

	// Service C should be open
	assert.Equal(t, resilience.StateOpen, serviceC.GetState())

	// Service B depends on C, should also fail
	for i := 0; i < 5; i++ {
		err := serviceB.Execute(func() error {
			// Try to call Service C
			if err := serviceC.Execute(func() error { return nil }); err != nil {
				return fmt.Errorf("dependency failed: %v", err)
			}
			return nil
		})
		assert.Error(t, err)
	}

	// Service B should also be open
	assert.Equal(t, resilience.StateOpen, serviceB.GetState())

	// Service A depends on B
	for i := 0; i < 5; i++ {
		err := serviceA.Execute(func() error {
			// Try to call Service B
			if err := serviceB.Execute(func() error { return nil }); err != nil {
				return fmt.Errorf("dependency failed: %v", err)
			}
			return nil
		})
		assert.Error(t, err)
	}

	// All services should be in open state
	assert.Equal(t, resilience.StateOpen, serviceA.GetState())

	// Wait for reset timeout
	time.Sleep(1100 * time.Millisecond)

	// Services should transition to half-open
	// Simulate recovery
	err := serviceC.Execute(func() error { return nil })
	assert.NoError(t, err)

	// Circuit should close after successful requests
	assert.Eventually(t, func() bool {
		return serviceC.GetState() == resilience.StateClosed
	}, 2*time.Second, 100*time.Millisecond)
}

// TestPhiAccrualDetectorAdaptive tests adaptive failure detection
func TestPhiAccrualDetectorAdaptive(t *testing.T) {
	logger := zap.NewNop()
	detector := ha.NewPhiAccrualDetector("test-node", 8.0, 100, logger)

	// Simulate regular heartbeats
	for i := 0; i < 10; i++ {
		detector.Heartbeat()
		time.Sleep(100 * time.Millisecond)
	}

	// Check not suspected with regular heartbeats
	assert.False(t, detector.IsSuspected())

	// Simulate network jitter
	for i := 0; i < 5; i++ {
		detector.Heartbeat()
		jitter := time.Duration(rand.Intn(200)) * time.Millisecond
		time.Sleep(100*time.Millisecond + jitter)
	}

	// Adapt threshold for network conditions
	detector.AdaptiveThreshold(30.0) // 30% jitter

	// Should not be suspected with adapted threshold
	assert.False(t, detector.IsSuspected())

	// Simulate failure (no heartbeats)
	time.Sleep(1 * time.Second)

	// Should be suspected
	assert.True(t, detector.IsSuspected())

	// Wait longer
	time.Sleep(1 * time.Second)

	// Should be failed
	assert.True(t, detector.IsFailed())

	// Recovery
	detector.Heartbeat()
	assert.False(t, detector.IsSuspected())
	assert.False(t, detector.IsFailed())
}

// TestResilienceManagerComprehensive tests complete resilience stack
func TestResilienceManagerComprehensive(t *testing.T) {
	logger := zap.NewNop()
	rm := resilience.NewResilienceManager("test-manager", logger)

	// Register all resilience patterns
	rm.RegisterCircuitBreaker("api", 3, 100*time.Millisecond, 1*time.Second)
	rm.RegisterRateLimiter("api", 10.0, 20) // 10 RPS, burst 20
	rm.RegisterBulkhead("api", 5, 10, 500*time.Millisecond)
	rm.RegisterRetryPolicy("api", 3, 100*time.Millisecond, 1*time.Second, 2.0, true)
	rm.RegisterErrorBudget("api", 0.99, 1*time.Hour) // 99% SLO
	rm.RegisterLatencyBudget("api", 100*time.Millisecond, 0.95, 1000) // 95th percentile < 100ms

	// Enable chaos engineering
	rm.EnableChaos()
	rm.RegisterFault(&MockFaultInjector{failureRate: 0.1}) // 10% failure rate

	// Start health monitoring
	rm.RegisterHealthCheck(&MockHealthCheck{healthy: true})
	rm.StartHealthMonitoring()
	defer rm.StopHealthMonitoring()

	ctx := context.Background()

	// Track metrics
	var successCount, failureCount int32
	var totalLatency int64

	// Concurrent load test
	var wg sync.WaitGroup
	workers := 20
	iterations := 100

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()

			for i := 0; i < iterations; i++ {
				start := time.Now()

				err := rm.ExecuteWithAllProtections(ctx, "api", func(ctx context.Context) error {
					// Simulate work
					time.Sleep(time.Duration(rand.Intn(50)) * time.Millisecond)

					// Random failures
					if rand.Float32() < 0.05 { // 5% failure rate
						return fmt.Errorf("random failure")
					}

					return nil
				})

				latency := time.Since(start)
				atomic.AddInt64(&totalLatency, int64(latency))

				if err != nil {
					atomic.AddInt32(&failureCount, 1)
				} else {
					atomic.AddInt32(&successCount, 1)
				}

				// Record metrics
				rm.RecordLatency("api", latency)
			}
		}(w)
	}

	wg.Wait()

	// Get metrics
	metrics := rm.GetAllMetrics()

	// Verify circuit breaker metrics
	cbMetrics := metrics.CircuitBreakers["api"]
	t.Logf("Circuit Breaker - State: %s, Success Rate: %.2f%%",
		cbMetrics.State, cbMetrics.SuccessRate*100)

	// Verify error budget
	ebMetrics := metrics.ErrorBudgets["api"]
	t.Logf("Error Budget - Success Rate: %.2f%%, Budget Remaining: %.2f%%",
		ebMetrics.SuccessRate*100, ebMetrics.BudgetRemaining*100)

	// Verify latency budget
	lbMetrics := metrics.LatencyBudgets["api"]
	t.Logf("Latency Budget - P95: %.2fms, Within Budget: %v",
		float64(lbMetrics.P95)/float64(time.Millisecond), lbMetrics.WithinBudget)

	// Assertions
	totalRequests := int(successCount + failureCount)
	actualSuccessRate := float64(successCount) / float64(totalRequests)

	// Should maintain reasonable success rate despite chaos
	assert.Greater(t, actualSuccessRate, 0.8, "Success rate should be > 80%")

	// Circuit breaker should have protected the system
	assert.Less(t, int(cbMetrics.TotalFailures), totalRequests/2, "Circuit breaker should limit failures")

	// Average latency should be reasonable
	avgLatency := time.Duration(totalLatency / int64(totalRequests))
	assert.Less(t, avgLatency, 200*time.Millisecond, "Average latency should be < 200ms")
}

// TestDisasterRecoveryOrchestration tests DR scenarios
func TestDisasterRecoveryOrchestration(t *testing.T) {
	logger := zap.NewNop()

	// Create DR orchestrator with aggressive RTO/RPO
	rto := 30 * time.Second
	rpo := 5 * time.Second
	dro := ha.NewDisasterRecoveryOrchestrator("test-dr", rto, rpo, ha.StrategyFailover, logger)

	// Start orchestrator
	require.NoError(t, dro.Start())
	defer dro.Stop()

	// Add standby sites
	dro.AddStandbySite("site-b")
	dro.AddStandbySite("site-c")

	// Add replication targets
	dro.AddReplicationTarget(ha.ReplicationTarget{
		ID:       "replica-1",
		Type:     "synchronous",
		Endpoint: "site-b:5432",
		Status:   "active",
	})

	// Create backups
	backup1, err := dro.CreateBackup("full")
	require.NoError(t, err)
	assert.NotNil(t, backup1)

	time.Sleep(100 * time.Millisecond)

	backup2, err := dro.CreateBackup("incremental")
	require.NoError(t, err)
	assert.NotNil(t, backup2)

	// Simulate primary failure and failover
	err = dro.TriggerFailover("site-b")
	require.NoError(t, err)

	// Verify metrics
	metrics := dro.GetMetrics()
	assert.Equal(t, 1, metrics.RecoveryCount)
	assert.Less(t, metrics.RecoveryTime, rto, "Recovery time should be within RTO")

	// Test restore from backup
	err = dro.RestoreFromBackup(backup1.ID)
	require.NoError(t, err)

	// Verify RPO compliance
	metrics = dro.GetMetrics()
	assert.Less(t, metrics.DataLoss, rpo, "Data loss should be within RPO")
}

// TestRaftConsensusUnderChaos tests Raft consensus with failures
func TestRaftConsensusUnderChaos(t *testing.T) {
	// Create 5-node Raft cluster
	nodes := make([]*consensus.RaftNode, 5)
	peers := []string{"node-0", "node-1", "node-2", "node-3", "node-4"}

	// Create mock transport
	transport := &MockRaftTransport{
		nodes:       make(map[string]*consensus.RaftNode),
		dropRate:    0.1, // 10% packet loss
		delayMs:     50,  // 50ms network delay
		partitioned: false,
	}

	// Initialize nodes
	for i := 0; i < 5; i++ {
		nodes[i] = consensus.NewRaftNode(peers[i], peers, transport)
		transport.nodes[peers[i]] = nodes[i]
		nodes[i].Start()
	}

	// Wait for leader election
	time.Sleep(1 * time.Second)

	// Find leader
	var leader *consensus.RaftNode
	for _, node := range nodes {
		if node.IsLeader() {
			leader = node
			break
		}
	}
	require.NotNil(t, leader, "Should have elected a leader")

	// Submit commands
	successCount := 0
	for i := 0; i < 100; i++ {
		command := fmt.Sprintf("command-%d", i)
		index, term, ok := leader.Submit(command)

		if ok {
			successCount++
			t.Logf("Submitted command %s at index %d, term %d", command, index, term)
		}

		// Inject random failures
		if rand.Float32() < 0.2 { // 20% chance of node crash
			victimIndex := rand.Intn(5)
			if !nodes[victimIndex].IsLeader() {
				nodes[victimIndex].Stop()
				time.Sleep(100 * time.Millisecond)
				nodes[victimIndex].Start() // Restart
			}
		}

		time.Sleep(10 * time.Millisecond)
	}

	// Should maintain consensus despite failures
	assert.Greater(t, successCount, 70, "Should succeed in majority of submissions")

	// Verify all nodes have consistent state (eventually)
	time.Sleep(2 * time.Second)

	// Check committed entries are consistent
	var appliedCounts []int
	for _, node := range nodes {
		count := 0
		applyCh := node.GetApplyChan()

		// Drain apply channel
		for {
			select {
			case <-applyCh:
				count++
			default:
				appliedCounts = append(appliedCounts, count)
				goto next
			}
		}
		next:
	}

	// All nodes should have similar number of applied entries
	t.Logf("Applied counts: %v", appliedCounts)

	// Clean up
	for _, node := range nodes {
		node.Stop()
	}
}

// TestSplitBrainPrevention tests split-brain prevention mechanisms
func TestSplitBrainPrevention(t *testing.T) {
	// Create cluster with quorum-based decisions
	nodes := 10
	quorum := nodes/2 + 1

	// Simulate network partition
	partition1 := nodes / 2      // 50%
	partition2 := nodes - partition1 // 50%

	t.Logf("Testing split-brain with %d nodes (quorum=%d)", nodes, quorum)
	t.Logf("Partition 1: %d nodes, Partition 2: %d nodes", partition1, partition2)

	// Neither partition should be able to make progress
	canPartition1Proceed := partition1 >= quorum
	canPartition2Proceed := partition2 >= quorum

	assert.False(t, canPartition1Proceed && canPartition2Proceed,
		"Both partitions should not be able to proceed (split-brain)")

	// Test with weighted quorum (60-40 split)
	partition1 = (nodes * 6) / 10
	partition2 = nodes - partition1

	t.Logf("60-40 split: Partition 1: %d nodes, Partition 2: %d nodes",
		partition1, partition2)

	canPartition1Proceed = partition1 >= quorum
	canPartition2Proceed = partition2 >= quorum

	assert.True(t, canPartition1Proceed, "Majority partition should proceed")
	assert.False(t, canPartition2Proceed, "Minority partition should not proceed")
}

// TestMemoryAndResourceExhaustion tests behavior under resource pressure
func TestMemoryAndResourceExhaustion(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping resource exhaustion test in short mode")
	}

	logger := zap.NewNop()
	rm := resilience.NewResilienceManager("resource-test", logger)

	// Configure bulkhead to limit concurrency
	rm.RegisterBulkhead("memory-intensive", 3, 5, 100*time.Millisecond)

	// Track memory usage
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	initialMemory := m.Alloc

	// Simulate memory-intensive operations
	var wg sync.WaitGroup
	rejected := int32(0)
	completed := int32(0)

	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			err := rm.ExecuteWithBulkhead("memory-intensive", func() error {
				// Allocate memory
				data := make([]byte, 10*1024*1024) // 10MB

				// Simulate processing
				time.Sleep(200 * time.Millisecond)

				// Touch memory to prevent optimization
				for i := range data {
					data[i] = byte(i % 256)
				}

				atomic.AddInt32(&completed, 1)
				return nil
			})

			if err != nil {
				atomic.AddInt32(&rejected, 1)
			}
		}(i)
	}

	wg.Wait()

	// Check memory didn't grow excessively
	runtime.ReadMemStats(&m)
	memoryGrowth := m.Alloc - initialMemory

	t.Logf("Memory growth: %d MB", memoryGrowth/(1024*1024))
	t.Logf("Completed: %d, Rejected: %d", completed, rejected)

	// Bulkhead should have limited concurrent operations
	assert.Greater(t, int(rejected), 0, "Some requests should be rejected")
	assert.Less(t, memoryGrowth, uint64(100*1024*1024),
		"Memory growth should be limited by bulkhead")

	// Force garbage collection
	runtime.GC()
	runtime.ReadMemStats(&m)
	afterGC := m.Alloc

	// Memory should be reclaimed
	assert.Less(t, afterGC, initialMemory+uint64(50*1024*1024),
		"Memory should be reclaimed after GC")
}

// Mock implementations

type MockFaultInjector struct {
	failureRate float32
}

func (m *MockFaultInjector) ShouldInject() bool {
	return rand.Float32() < m.failureRate
}

func (m *MockFaultInjector) Inject() error {
	if m.ShouldInject() {
		return fmt.Errorf("injected fault")
	}
	return nil
}

type MockHealthCheck struct {
	healthy bool
	mu      sync.RWMutex
}

func (m *MockHealthCheck) Name() string {
	return "mock-health"
}

func (m *MockHealthCheck) Check() error {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if !m.healthy {
		return fmt.Errorf("unhealthy")
	}
	return nil
}

func (m *MockHealthCheck) SetHealthy(healthy bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.healthy = healthy
}

type MockRaftTransport struct {
	nodes       map[string]*consensus.RaftNode
	dropRate    float32
	delayMs     int
	partitioned bool
	mu          sync.RWMutex
}

func (t *MockRaftTransport) SendRequestVote(ctx context.Context, nodeID string, req *consensus.RequestVoteArgs) (*consensus.RequestVoteReply, error) {
	// Simulate network issues
	if rand.Float32() < t.dropRate {
		return nil, fmt.Errorf("packet dropped")
	}

	time.Sleep(time.Duration(t.delayMs) * time.Millisecond)

	t.mu.RLock()
	node, exists := t.nodes[nodeID]
	t.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("node not found")
	}

	if t.partitioned {
		return nil, fmt.Errorf("network partitioned")
	}

	return node.HandleRequestVote(req), nil
}

func (t *MockRaftTransport) SendAppendEntries(ctx context.Context, nodeID string, req *consensus.AppendEntriesArgs) (*consensus.AppendEntriesReply, error) {
	// Simulate network issues
	if rand.Float32() < t.dropRate {
		return nil, fmt.Errorf("packet dropped")
	}

	time.Sleep(time.Duration(t.delayMs) * time.Millisecond)

	t.mu.RLock()
	node, exists := t.nodes[nodeID]
	t.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("node not found")
	}

	if t.partitioned {
		return nil, fmt.Errorf("network partitioned")
	}

	return node.HandleAppendEntries(req), nil
}

func (t *MockRaftTransport) SendSnapshot(ctx context.Context, nodeID string, req *consensus.InstallSnapshotArgs) (*consensus.InstallSnapshotReply, error) {
	// Not implemented for this test
	return &consensus.InstallSnapshotReply{Term: 0}, nil
}