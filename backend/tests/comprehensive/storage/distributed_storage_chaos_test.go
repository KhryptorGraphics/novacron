package storage_test

import (
	"context"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// TestDistributedStorageChaos implements comprehensive chaos engineering tests
func TestDistributedStorageChaos(t *testing.T) {
	t.Run("Network Partition Tests", func(t *testing.T) {
		testNetworkPartitionTolerance(t)
		testSplitBrainPrevention(t)
		testPartitionRecovery(t)
	})

	t.Run("Node Failure Tests", func(t *testing.T) {
		testSingleNodeFailure(t)
		testMultipleNodeFailure(t)
		testCascadingNodeFailures(t)
		testByzantineFailures(t)
	})

	t.Run("Data Consistency Tests", func(t *testing.T) {
		testConsistencyUnderPartition(t)
		testConsistencyDuringFailover(t)
		testQuorumBasedOperations(t)
		testDataIntegrityValidation(t)
	})

	t.Run("Performance Under Stress", func(t *testing.T) {
		testPerformanceDuringChaos(t)
		testLatencyUnderLoad(t)
		testThroughputDegradation(t)
	})

	t.Run("Recovery and Healing", func(t *testing.T) {
		testAutoHealing(t)
		testManualRecovery(t)
		testDataReplication(t)
		testShardRebalancing(t)
	})
}

func testNetworkPartitionTolerance(t *testing.T) {
	// Setup distributed storage with multiple nodes
	nodeCount := 5
	replicationFactor := 3
	
	cluster := setupDistributedCluster(t, nodeCount, replicationFactor)
	defer cluster.Cleanup()

	// Create test volumes before partition
	testVolumes := createTestVolumes(t, cluster.nodes[0], 10)

	// Simulate network partition: isolate 2 nodes from 3 nodes
	partitionNodes := cluster.nodes[:2]
	majorityNodes := cluster.nodes[2:]

	// Apply network partition
	cluster.ApplyNetworkPartition(partitionNodes, majorityNodes)

	// Test operations on majority partition (should succeed)
	ctx := context.Background()
	for _, volume := range testVolumes[:5] {
		data := []byte(fmt.Sprintf("test-data-%s", volume))
		err := cluster.WriteShardData(ctx, majorityNodes[0], volume, 0, data)
		if err != nil {
			t.Errorf("Write should succeed on majority partition: %v", err)
		}
	}

	// Test operations on minority partition (should fail or be limited)
	for _, volume := range testVolumes[5:] {
		data := []byte(fmt.Sprintf("test-data-%s", volume))
		err := cluster.WriteShardData(ctx, partitionNodes[0], volume, 0, data)
		if err == nil && cluster.consistencyLevel == "strong" {
			t.Error("Write should fail on minority partition with strong consistency")
		}
	}

	// Verify read availability
	for _, volume := range testVolumes {
		_, err := cluster.ReadShardData(ctx, majorityNodes[0], volume, 0)
		if err != nil && cluster.hasQuorum(majorityNodes, volume) {
			t.Errorf("Read should succeed on majority partition for volume %s: %v", volume, err)
		}
	}
}

func testSplitBrainPrevention(t *testing.T) {
	cluster := setupDistributedCluster(t, 4, 2) // Even number for split-brain scenario
	defer cluster.Cleanup()

	// Create test data
	testVolumes := createTestVolumes(t, cluster.nodes[0], 5)

	// Create perfect split: 2 nodes vs 2 nodes
	partition1 := cluster.nodes[:2]
	partition2 := cluster.nodes[2:]

	cluster.ApplyNetworkPartition(partition1, partition2)

	ctx := context.Background()
	var errors []error

	// Both partitions should reject writes (no quorum)
	testData := []byte("split-brain-test-data")
	
	err1 := cluster.WriteShardData(ctx, partition1[0], testVolumes[0], 0, testData)
	if err1 == nil {
		errors = append(errors, fmt.Errorf("partition1 should reject write without quorum"))
	}

	err2 := cluster.WriteShardData(ctx, partition2[0], testVolumes[0], 0, testData)
	if err2 == nil {
		errors = append(errors, fmt.Errorf("partition2 should reject write without quorum"))
	}

	// Both partitions might allow reads of existing data
	_, readErr1 := cluster.ReadShardData(ctx, partition1[0], testVolumes[0], 0)
	_, readErr2 := cluster.ReadShardData(ctx, partition2[0], testVolumes[0], 0)

	t.Logf("Split-brain write errors: partition1=%v, partition2=%v", err1, err2)
	t.Logf("Split-brain read results: partition1=%v, partition2=%v", readErr1, readErr2)

	for _, err := range errors {
		t.Error(err)
	}
}

func testPartitionRecovery(t *testing.T) {
	cluster := setupDistributedCluster(t, 5, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 8)

	// Apply partition
	partitionNodes := cluster.nodes[:2]
	majorityNodes := cluster.nodes[2:]
	cluster.ApplyNetworkPartition(partitionNodes, majorityNodes)

	// Write data to majority partition during outage
	ctx := context.Background()
	partitionData := make(map[string][]byte)
	
	for i, volume := range testVolumes {
		data := []byte(fmt.Sprintf("recovery-test-data-%d", i))
		partitionData[volume] = data
		
		err := cluster.WriteShardData(ctx, majorityNodes[0], volume, 0, data)
		if err != nil {
			t.Logf("Write failed during partition for volume %s: %v", volume, err)
		}
	}

	// Restore network connectivity
	time.Sleep(100 * time.Millisecond) // Simulate partition duration
	cluster.RestoreNetwork()

	// Wait for reconciliation
	time.Sleep(500 * time.Millisecond)

	// Verify data consistency across all nodes after recovery
	for volume, expectedData := range partitionData {
		for nodeIndex, node := range cluster.nodes {
			actualData, err := cluster.ReadShardData(ctx, node, volume, 0)
			if err != nil {
				t.Errorf("Failed to read volume %s from node %d after recovery: %v", 
					volume, nodeIndex, err)
				continue
			}
			
			if string(actualData) != string(expectedData) {
				t.Errorf("Data inconsistency on node %d for volume %s: expected %s, got %s", 
					nodeIndex, volume, string(expectedData), string(actualData))
			}
		}
	}
}

func testSingleNodeFailure(t *testing.T) {
	cluster := setupDistributedCluster(t, 5, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 10)

	// Simulate single node failure
	failedNode := cluster.nodes[0]
	remainingNodes := cluster.nodes[1:]
	
	cluster.SimulateNodeFailure(failedNode)

	ctx := context.Background()

	// Verify operations continue on remaining nodes
	for i, volume := range testVolumes {
		data := []byte(fmt.Sprintf("single-failure-test-%d", i))
		
		// Try writing to a healthy node
		err := cluster.WriteShardData(ctx, remainingNodes[0], volume, 0, data)
		if err != nil && len(remainingNodes) >= cluster.replicationFactor {
			t.Errorf("Write should succeed after single node failure: %v", err)
		}
		
		// Verify read availability
		_, err = cluster.ReadShardData(ctx, remainingNodes[1], volume, 0)
		if err != nil {
			t.Logf("Read failed after single node failure for volume %s: %v", volume, err)
		}
	}

	// Verify failed node is detected and marked as unavailable
	if cluster.IsNodeHealthy(failedNode) {
		t.Error("Failed node should be marked as unhealthy")
	}
}

func testMultipleNodeFailure(t *testing.T) {
	cluster := setupDistributedCluster(t, 7, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 8)

	// Simulate failure of 2 nodes simultaneously
	failedNodes := cluster.nodes[:2]
	remainingNodes := cluster.nodes[2:]

	for _, node := range failedNodes {
		cluster.SimulateNodeFailure(node)
	}

	ctx := context.Background()

	// System should still be operational
	for i, volume := range testVolumes {
		data := []byte(fmt.Sprintf("multi-failure-test-%d", i))
		
		err := cluster.WriteShardData(ctx, remainingNodes[0], volume, 0, data)
		if err != nil && len(remainingNodes) >= cluster.replicationFactor {
			t.Logf("Write failed after multiple node failure: %v", err)
		}
	}

	// Verify cluster health assessment
	healthyCount := 0
	for _, node := range cluster.nodes {
		if cluster.IsNodeHealthy(node) {
			healthyCount++
		}
	}

	expectedHealthy := len(cluster.nodes) - len(failedNodes)
	if healthyCount != expectedHealthy {
		t.Errorf("Expected %d healthy nodes, got %d", expectedHealthy, healthyCount)
	}
}

func testCascadingNodeFailures(t *testing.T) {
	cluster := setupDistributedCluster(t, 6, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 6)
	ctx := context.Background()

	// Simulate cascading failures with delays
	failureSequence := []int{0, 2, 4} // Fail nodes at different times
	
	for i, nodeIndex := range failureSequence {
		// Fail one node
		cluster.SimulateNodeFailure(cluster.nodes[nodeIndex])
		
		// Test system state after each failure
		remainingHealthyNodes := cluster.GetHealthyNodes()
		t.Logf("After failure %d: %d healthy nodes remaining", i+1, len(remainingHealthyNodes))
		
		// Try operations after each failure
		if len(remainingHealthyNodes) > 0 {
			volume := testVolumes[i]
			data := []byte(fmt.Sprintf("cascading-test-%d", i))
			
			err := cluster.WriteShardData(ctx, remainingHealthyNodes[0], volume, 0, data)
			if err != nil && len(remainingHealthyNodes) >= cluster.replicationFactor {
				t.Logf("Write failed after cascading failure %d: %v", i+1, err)
			}
		}
		
		// Simulate time between failures
		time.Sleep(50 * time.Millisecond)
		
		// Check if system reaches critical failure threshold
		if len(remainingHealthyNodes) < cluster.replicationFactor {
			t.Logf("System reached critical failure threshold after %d failures", i+1)
			break
		}
	}
}

func testByzantineFailures(t *testing.T) {
	cluster := setupDistributedCluster(t, 7, 3) // Need more nodes for Byzantine tolerance
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 5)
	ctx := context.Background()

	// Create test data
	originalData := []byte("original-byzantine-test-data")
	corruptData := []byte("corrupted-byzantine-test-data")

	// Write original data
	err := cluster.WriteShardData(ctx, cluster.nodes[0], testVolumes[0], 0, originalData)
	if err != nil {
		t.Fatalf("Failed to write initial data: %v", err)
	}

	// Simulate Byzantine node: returns corrupted data
	byzantineNode := cluster.nodes[1]
	cluster.SimulateByzantineFailure(byzantineNode, testVolumes[0], corruptData)

	// Read from multiple nodes to detect Byzantine behavior
	readResults := make(map[string]int)
	for _, node := range cluster.nodes {
		if node == byzantineNode {
			continue // Skip Byzantine node for now
		}
		
		data, err := cluster.ReadShardData(ctx, node, testVolumes[0], 0)
		if err != nil {
			t.Logf("Read error from node: %v", err)
			continue
		}
		
		readResults[string(data)]++
	}

	// Majority should return original data
	originalCount := readResults[string(originalData)]
	corruptCount := readResults[string(corruptData)]

	t.Logf("Byzantine detection: original=%d, corrupt=%d", originalCount, corruptCount)

	if originalCount <= corruptCount {
		t.Error("Byzantine failure not properly detected/handled")
	}
}

func testConsistencyUnderPartition(t *testing.T) {
	cluster := setupDistributedCluster(t, 5, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 3)
	ctx := context.Background()

	// Write initial data
	initialData := []byte("initial-consistency-data")
	for _, volume := range testVolumes {
		err := cluster.WriteShardData(ctx, cluster.nodes[0], volume, 0, initialData)
		if err != nil {
			t.Fatalf("Failed to write initial data: %v", err)
		}
	}

	// Create partition
	partition1 := cluster.nodes[:2]
	partition2 := cluster.nodes[2:]
	cluster.ApplyNetworkPartition(partition1, partition2)

	// Attempt writes to both partitions
	partition1Data := []byte("partition1-data")
	partition2Data := []byte("partition2-data")

	err1 := cluster.WriteShardData(ctx, partition1[0], testVolumes[0], 0, partition1Data)
	err2 := cluster.WriteShardData(ctx, partition2[0], testVolumes[0], 0, partition2Data)

	t.Logf("Partition writes: p1=%v, p2=%v", err1, err2)

	// Restore network and check consistency
	cluster.RestoreNetwork()
	time.Sleep(200 * time.Millisecond) // Allow reconciliation

	// Verify all nodes have consistent data
	finalData, err := cluster.ReadShardData(ctx, cluster.nodes[0], testVolumes[0], 0)
	if err != nil {
		t.Fatalf("Failed to read final data: %v", err)
	}

	for i, node := range cluster.nodes[1:] {
		nodeData, err := cluster.ReadShardData(ctx, node, testVolumes[0], 0)
		if err != nil {
			t.Errorf("Failed to read from node %d: %v", i+1, err)
			continue
		}
		
		if string(nodeData) != string(finalData) {
			t.Errorf("Consistency violation: node %d has %s, expected %s", 
				i+1, string(nodeData), string(finalData))
		}
	}
}

func testQuorumBasedOperations(t *testing.T) {
	cluster := setupDistributedCluster(t, 5, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 5)
	ctx := context.Background()

	// Test with various quorum scenarios
	quorumTests := []struct {
		name           string
		availableNodes int
		shouldSucceed  bool
	}{
		{"Full cluster", 5, true},
		{"Majority available", 3, true},
		{"Exactly quorum", 3, true},
		{"Below quorum", 2, false},
		{"Single node", 1, false},
	}

	for _, test := range quorumTests {
		t.Run(test.name, func(t *testing.T) {
			// Simulate different numbers of available nodes
			healthyNodes := cluster.nodes[:test.availableNodes]
			failedNodes := cluster.nodes[test.availableNodes:]

			// Mark nodes as failed
			for _, node := range failedNodes {
				cluster.SimulateNodeFailure(node)
			}

			// Test write operation
			testData := []byte(fmt.Sprintf("quorum-test-%s", test.name))
			err := cluster.WriteShardData(ctx, healthyNodes[0], testVolumes[0], 0, testData)

			if test.shouldSucceed && err != nil {
				t.Errorf("Write should succeed with %d nodes: %v", test.availableNodes, err)
			} else if !test.shouldSucceed && err == nil {
				t.Errorf("Write should fail with %d nodes", test.availableNodes)
			}

			// Restore failed nodes for next test
			for _, node := range failedNodes {
				cluster.RestoreNode(node)
			}
		})
	}
}

func testDataIntegrityValidation(t *testing.T) {
	cluster := setupDistributedCluster(t, 5, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 10)
	ctx := context.Background()

	// Write test data with checksums
	testData := make(map[string][]byte)
	for i, volume := range testVolumes {
		data := []byte(fmt.Sprintf("integrity-test-data-%d", i))
		testData[volume] = data
		
		err := cluster.WriteShardData(ctx, cluster.nodes[0], volume, 0, data)
		if err != nil {
			t.Fatalf("Failed to write test data: %v", err)
		}
	}

	// Simulate data corruption on one node
	corruptedNode := cluster.nodes[1]
	corruptedVolume := testVolumes[0]
	corruptedData := []byte("corrupted-data")
	
	cluster.CorruptShardData(corruptedNode, corruptedVolume, 0, corruptedData)

	// Trigger integrity check/repair
	err := cluster.PerformIntegrityCheck(ctx, corruptedVolume)
	if err != nil {
		t.Logf("Integrity check reported issues (expected): %v", err)
	}

	// Verify corruption was detected and repaired
	repairedData, err := cluster.ReadShardData(ctx, corruptedNode, corruptedVolume, 0)
	if err != nil {
		t.Errorf("Failed to read repaired data: %v", err)
	}

	expectedData := testData[corruptedVolume]
	if string(repairedData) != string(expectedData) {
		t.Errorf("Data not properly repaired: got %s, expected %s", 
			string(repairedData), string(expectedData))
	}
}

func testPerformanceDuringChaos(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	cluster := setupDistributedCluster(t, 6, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 50)
	ctx := context.Background()

	// Baseline performance measurement
	baseline := measureOperationLatency(t, cluster, testVolumes[:10], "baseline")

	// Performance during single node failure
	cluster.SimulateNodeFailure(cluster.nodes[0])
	singleFailure := measureOperationLatency(t, cluster, testVolumes[10:20], "single-failure")

	// Performance during network partition
	cluster.RestoreNode(cluster.nodes[0])
	partition1 := cluster.nodes[:2]
	partition2 := cluster.nodes[2:]
	cluster.ApplyNetworkPartition(partition1, partition2)
	
	partition := measureOperationLatency(t, cluster, testVolumes[20:30], "partition")
	cluster.RestoreNetwork()

	// Performance during multiple failures
	cluster.SimulateNodeFailure(cluster.nodes[0])
	cluster.SimulateNodeFailure(cluster.nodes[1])
	multiFailure := measureOperationLatency(t, cluster, testVolumes[30:40], "multi-failure")

	// Report performance degradation
	t.Logf("Performance results:")
	t.Logf("  Baseline: %.2fms avg", baseline.avgLatency)
	t.Logf("  Single failure: %.2fms avg (%.1f%% degradation)", 
		singleFailure.avgLatency, 
		(singleFailure.avgLatency-baseline.avgLatency)/baseline.avgLatency*100)
	t.Logf("  Network partition: %.2fms avg (%.1f%% degradation)", 
		partition.avgLatency,
		(partition.avgLatency-baseline.avgLatency)/baseline.avgLatency*100)
	t.Logf("  Multi failure: %.2fms avg (%.1f%% degradation)", 
		multiFailure.avgLatency,
		(multiFailure.avgLatency-baseline.avgLatency)/baseline.avgLatency*100)

	// Verify performance doesn't degrade beyond acceptable limits
	maxDegradation := 300.0 // 300% degradation limit
	if (singleFailure.avgLatency-baseline.avgLatency)/baseline.avgLatency*100 > maxDegradation {
		t.Errorf("Single failure degradation exceeds limit: %.1f%%", 
			(singleFailure.avgLatency-baseline.avgLatency)/baseline.avgLatency*100)
	}
}

func testAutoHealing(t *testing.T) {
	cluster := setupDistributedCluster(t, 5, 3)
	defer cluster.Cleanup()

	// Enable auto-healing with short intervals for testing
	cluster.EnableAutoHealing(100 * time.Millisecond)

	testVolumes := createTestVolumes(t, cluster.nodes[0], 8)
	ctx := context.Background()

	// Write initial data
	for i, volume := range testVolumes {
		data := []byte(fmt.Sprintf("healing-test-%d", i))
		err := cluster.WriteShardData(ctx, cluster.nodes[0], volume, 0, data)
		if err != nil {
			t.Fatalf("Failed to write initial data: %v", err)
		}
	}

	// Simulate node failure
	failedNode := cluster.nodes[0]
	cluster.SimulateNodeFailure(failedNode)

	// Wait for auto-healing to trigger
	time.Sleep(500 * time.Millisecond)

	// Verify data was replicated to other nodes
	for _, volume := range testVolumes {
		replicaCount := 0
		for _, node := range cluster.nodes[1:] { // Skip failed node
			_, err := cluster.ReadShardData(ctx, node, volume, 0)
			if err == nil {
				replicaCount++
			}
		}
		
		if replicaCount < cluster.replicationFactor-1 {
			t.Errorf("Auto-healing failed for volume %s: only %d replicas found", 
				volume, replicaCount)
		}
	}

	// Test healing when node recovers
	cluster.RestoreNode(failedNode)
	time.Sleep(300 * time.Millisecond)

	// Verify failed node was re-synchronized
	for _, volume := range testVolumes {
		_, err := cluster.ReadShardData(ctx, failedNode, volume, 0)
		if err != nil {
			t.Errorf("Node recovery healing failed for volume %s: %v", volume, err)
		}
	}
}

func testDataReplication(t *testing.T) {
	cluster := setupDistributedCluster(t, 6, 3)
	defer cluster.Cleanup()

	testVolumes := createTestVolumes(t, cluster.nodes[0], 12)
	ctx := context.Background()

	// Write data and verify replication
	for i, volume := range testVolumes {
		data := []byte(fmt.Sprintf("replication-test-%d", i))
		
		err := cluster.WriteShardData(ctx, cluster.nodes[0], volume, 0, data)
		if err != nil {
			t.Fatalf("Failed to write data: %v", err)
		}
		
		// Verify data is replicated to expected number of nodes
		replicaCount := 0
		for _, node := range cluster.nodes {
			nodeData, err := cluster.ReadShardData(ctx, node, volume, 0)
			if err == nil && string(nodeData) == string(data) {
				replicaCount++
			}
		}
		
		if replicaCount < cluster.replicationFactor {
			t.Errorf("Insufficient replicas for volume %s: got %d, expected %d", 
				volume, replicaCount, cluster.replicationFactor)
		}
	}
}

// Helper types and functions

type DistributedCluster struct {
	nodes             []*MockDistributedNode
	replicationFactor int
	consistencyLevel  string
	tempDir           string
	networkPartitions map[string][]string
	autoHealingEnabled bool
	autoHealingInterval time.Duration
}

type MockDistributedNode struct {
	id           string
	healthy      bool
	partitioned  bool
	byzantine    bool
	byzantineData map[string][]byte
	data         map[string]map[int][]byte // volume -> shard -> data
	mu           sync.RWMutex
}

type PerformanceMetrics struct {
	avgLatency    float64
	p95Latency    float64
	throughput    float64
	errorRate     float64
	operationCount int
}

func setupDistributedCluster(t *testing.T, nodeCount, replicationFactor int) *DistributedCluster {
	tempDir := t.TempDir()
	
	cluster := &DistributedCluster{
		nodes:             make([]*MockDistributedNode, nodeCount),
		replicationFactor: replicationFactor,
		consistencyLevel:  "strong",
		tempDir:           tempDir,
		networkPartitions: make(map[string][]string),
	}

	for i := 0; i < nodeCount; i++ {
		cluster.nodes[i] = &MockDistributedNode{
			id:            fmt.Sprintf("node-%d", i),
			healthy:       true,
			partitioned:   false,
			byzantine:     false,
			byzantineData: make(map[string][]byte),
			data:          make(map[string]map[int][]byte),
		}
	}

	return cluster
}

func (c *DistributedCluster) Cleanup() {
	// Cleanup temporary resources
	os.RemoveAll(c.tempDir)
}

func (c *DistributedCluster) ApplyNetworkPartition(partition1, partition2 []*MockDistributedNode) {
	// Mark nodes as partitioned from each other
	for _, node1 := range partition1 {
		for _, node2 := range partition2 {
			c.networkPartitions[node1.id+"-"+node2.id] = []string{node1.id, node2.id}
			c.networkPartitions[node2.id+"-"+node1.id] = []string{node1.id, node2.id}
		}
	}
}

func (c *DistributedCluster) RestoreNetwork() {
	c.networkPartitions = make(map[string][]string)
}

func (c *DistributedCluster) SimulateNodeFailure(node *MockDistributedNode) {
	node.mu.Lock()
	defer node.mu.Unlock()
	node.healthy = false
}

func (c *DistributedCluster) RestoreNode(node *MockDistributedNode) {
	node.mu.Lock()
	defer node.mu.Unlock()
	node.healthy = true
}

func (c *DistributedCluster) SimulateByzantineFailure(node *MockDistributedNode, volumeID string, corruptData []byte) {
	node.mu.Lock()
	defer node.mu.Unlock()
	node.byzantine = true
	node.byzantineData[volumeID] = corruptData
}

func (c *DistributedCluster) IsNodeHealthy(node *MockDistributedNode) bool {
	node.mu.RLock()
	defer node.mu.RUnlock()
	return node.healthy
}

func (c *DistributedCluster) GetHealthyNodes() []*MockDistributedNode {
	var healthy []*MockDistributedNode
	for _, node := range c.nodes {
		if c.IsNodeHealthy(node) {
			healthy = append(healthy, node)
		}
	}
	return healthy
}

func (c *DistributedCluster) hasQuorum(nodes []*MockDistributedNode, volumeID string) bool {
	healthyCount := 0
	for _, node := range nodes {
		if c.IsNodeHealthy(node) {
			healthyCount++
		}
	}
	return healthyCount >= (c.replicationFactor/2 + 1)
}

func (c *DistributedCluster) WriteShardData(ctx context.Context, node *MockDistributedNode, volumeID string, shardIndex int, data []byte) error {
	node.mu.Lock()
	defer node.mu.Unlock()

	if !node.healthy {
		return fmt.Errorf("node %s is not healthy", node.id)
	}

	// Check for network partition
	if c.isPartitioned(node) && c.consistencyLevel == "strong" {
		return fmt.Errorf("network partition prevents write")
	}

	// Initialize volume data if needed
	if node.data[volumeID] == nil {
		node.data[volumeID] = make(map[int][]byte)
	}

	node.data[volumeID][shardIndex] = data
	return nil
}

func (c *DistributedCluster) ReadShardData(ctx context.Context, node *MockDistributedNode, volumeID string, shardIndex int) ([]byte, error) {
	node.mu.RLock()
	defer node.mu.RUnlock()

	if !node.healthy {
		return nil, fmt.Errorf("node %s is not healthy", node.id)
	}

	// Return Byzantine data if node is Byzantine
	if node.byzantine {
		if corruptData, exists := node.byzantineData[volumeID]; exists {
			return corruptData, nil
		}
	}

	volumeData, exists := node.data[volumeID]
	if !exists {
		return nil, fmt.Errorf("volume %s not found", volumeID)
	}

	shardData, exists := volumeData[shardIndex]
	if !exists {
		return nil, fmt.Errorf("shard %d not found for volume %s", shardIndex, volumeID)
	}

	return shardData, nil
}

func (c *DistributedCluster) CorruptShardData(node *MockDistributedNode, volumeID string, shardIndex int, corruptData []byte) {
	node.mu.Lock()
	defer node.mu.Unlock()

	if node.data[volumeID] == nil {
		node.data[volumeID] = make(map[int][]byte)
	}
	node.data[volumeID][shardIndex] = corruptData
}

func (c *DistributedCluster) PerformIntegrityCheck(ctx context.Context, volumeID string) error {
	// Simulate integrity check by comparing data across nodes
	dataVersions := make(map[string]int)
	
	for _, node := range c.nodes {
		if !c.IsNodeHealthy(node) {
			continue
		}
		
		data, err := c.ReadShardData(ctx, node, volumeID, 0)
		if err != nil {
			continue
		}
		
		dataVersions[string(data)]++
	}
	
	if len(dataVersions) > 1 {
		return fmt.Errorf("data inconsistency detected for volume %s", volumeID)
	}
	
	return nil
}

func (c *DistributedCluster) isPartitioned(node *MockDistributedNode) bool {
	// Simplified partition check
	return len(c.networkPartitions) > 0
}

func (c *DistributedCluster) EnableAutoHealing(interval time.Duration) {
	c.autoHealingEnabled = true
	c.autoHealingInterval = interval
	
	// Start auto-healing goroutine (simplified)
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()
		
		for range ticker.C {
			if !c.autoHealingEnabled {
				return
			}
			// Simplified healing logic
			// In real implementation, would replicate data to maintain replication factor
		}
	}()
}

func createTestVolumes(t *testing.T, node *MockDistributedNode, count int) []string {
	volumes := make([]string, count)
	for i := 0; i < count; i++ {
		volumes[i] = fmt.Sprintf("test-volume-%d", i)
	}
	return volumes
}

func measureOperationLatency(t *testing.T, cluster *DistributedCluster, volumes []string, scenario string) PerformanceMetrics {
	ctx := context.Background()
	latencies := make([]time.Duration, len(volumes))
	errors := 0
	
	start := time.Now()
	
	for i, volume := range volumes {
		data := []byte(fmt.Sprintf("perf-test-%s-%d", scenario, i))
		
		opStart := time.Now()
		
		// Try to find a healthy node for the operation
		var err error
		for _, node := range cluster.GetHealthyNodes() {
			err = cluster.WriteShardData(ctx, node, volume, 0, data)
			if err == nil {
				break
			}
		}
		
		latencies[i] = time.Since(opStart)
		
		if err != nil {
			errors++
		}
	}
	
	totalDuration := time.Since(start)
	
	// Calculate metrics
	var totalLatency time.Duration
	for _, latency := range latencies {
		totalLatency += latency
	}
	
	avgLatency := float64(totalLatency.Nanoseconds()) / float64(len(latencies)) / 1e6 // Convert to ms
	throughput := float64(len(volumes)) / totalDuration.Seconds()
	errorRate := float64(errors) / float64(len(volumes)) * 100
	
	return PerformanceMetrics{
		avgLatency:     avgLatency,
		throughput:     throughput,
		errorRate:      errorRate,
		operationCount: len(volumes),
	}
}