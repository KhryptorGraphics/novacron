package federation

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// MockLogger implements the Logger interface for testing
type MockLogger struct {
	logs []string
}

func (m *MockLogger) Debug(msg string, args ...interface{}) {
	m.logs = append(m.logs, "DEBUG: "+msg)
}

func (m *MockLogger) Info(msg string, args ...interface{}) {
	m.logs = append(m.logs, "INFO: "+msg)
}

func (m *MockLogger) Warn(msg string, args ...interface{}) {
	m.logs = append(m.logs, "WARN: "+msg)
}

func (m *MockLogger) Error(msg string, args ...interface{}) {
	m.logs = append(m.logs, "ERROR: "+msg)
}

func TestNewManager(t *testing.T) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 5 * time.Second,
		ElectionTimeout:   10 * time.Second,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, err := NewManager(config, logger)
	require.NoError(t, err)
	assert.NotNil(t, manager)
	assert.Equal(t, "test-node-1", manager.localNode.ID)
	assert.Equal(t, "test-cluster", manager.localNode.ClusterID)
	assert.Equal(t, NodeStateDiscovering, manager.localNode.State)
}

func TestManagerStartStop(t *testing.T) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 100 * time.Millisecond,
		ElectionTimeout:   200 * time.Millisecond,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, err := NewManager(config, logger)
	require.NoError(t, err)

	ctx := context.Background()

	// Start manager
	err = manager.Start(ctx)
	require.NoError(t, err)
	assert.True(t, manager.isRunning.Load())

	// Should not be able to start again
	err = manager.Start(ctx)
	assert.Error(t, err)

	// Wait a bit for background goroutines to start
	time.Sleep(100 * time.Millisecond)

	// Stop manager
	err = manager.Stop(ctx)
	require.NoError(t, err)
	assert.False(t, manager.isRunning.Load())

	// Should not be able to stop again
	err = manager.Stop(ctx)
	assert.Error(t, err)
}

func TestGetNodes(t *testing.T) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 5 * time.Second,
		ElectionTimeout:   10 * time.Second,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, err := NewManager(config, logger)
	require.NoError(t, err)

	ctx := context.Background()

	// Add some test nodes
	testNodes := []*Node{
		{
			ID:        "node-1",
			ClusterID: "test-cluster",
			Address:   "127.0.0.1:8081",
			State:     NodeStateActive,
		},
		{
			ID:        "node-2",
			ClusterID: "test-cluster",
			Address:   "127.0.0.1:8082",
			State:     NodeStateActive,
		},
	}

	for _, node := range testNodes {
		manager.addNode(node)
	}

	// Get nodes
	nodes, err := manager.GetNodes(ctx)
	require.NoError(t, err)
	assert.Len(t, nodes, 2)

	// Get specific node
	node, err := manager.GetNode(ctx, "node-1")
	require.NoError(t, err)
	assert.Equal(t, "node-1", node.ID)

	// Try to get non-existent node
	_, err = manager.GetNode(ctx, "non-existent")
	assert.Error(t, err)
}

func TestResourceAllocation(t *testing.T) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 5 * time.Second,
		ElectionTimeout:   10 * time.Second,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
		ResourceSharingPolicy: ResourcePolicy{
			EnableSharing:      true,
			MaxSharePercentage: 80,
			Priority:           5,
		},
	}

	manager, err := NewManager(config, logger)
	require.NoError(t, err)

	ctx := context.Background()

	// Add a node with resources
	node := &Node{
		ID:        "resource-node",
		ClusterID: "test-cluster",
		Address:   "127.0.0.1:8081",
		State:     NodeStateActive,
		Capabilities: NodeCapabilities{
			CPUCores:  8,
			MemoryGB:  32,
			StorageGB: 500,
			Resources: ResourceInventory{
				TotalCPU:     8,
				UsedCPU:      2,
				TotalMemory:  32 * 1024 * 1024 * 1024,
				UsedMemory:   8 * 1024 * 1024 * 1024,
				TotalStorage: 500 * 1024 * 1024 * 1024,
				UsedStorage:  100 * 1024 * 1024 * 1024,
			},
		},
	}
	manager.addNode(node)

	// Update resource inventory
	err = manager.resourcePool.UpdateResourceInventory(ctx, node.ID, &node.Capabilities.Resources)
	require.NoError(t, err)

	// Request resources
	request := &ResourceRequest{
		ID:           "test-request",
		ResourceType: "vm",
		CPUCores:     2,
		MemoryGB:     8,
		StorageGB:    50,
		Duration:     1 * time.Hour,
		Priority:     5,
	}

	allocation, err := manager.RequestResources(ctx, request)
	require.NoError(t, err)
	assert.NotNil(t, allocation)
	assert.Equal(t, "test-request", allocation.RequestID)
	assert.Equal(t, float64(2), allocation.AllocatedCPU)
	assert.Equal(t, float64(8), allocation.AllocatedMem)
	assert.Equal(t, AllocationActive, allocation.Status)

	// Release resources
	err = manager.ReleaseResources(ctx, allocation.ID)
	require.NoError(t, err)
}

func TestHealthChecking(t *testing.T) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 100 * time.Millisecond,
		ElectionTimeout:   200 * time.Millisecond,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, err := NewManager(config, logger)
	require.NoError(t, err)

	ctx := context.Background()

	// Start manager
	err = manager.Start(ctx)
	require.NoError(t, err)
	defer manager.Stop(ctx)

	// Get health
	health, err := manager.GetHealth(ctx)
	require.NoError(t, err)
	assert.NotNil(t, health)
	assert.Equal(t, manager.localNode.ID, health.NodeID)
	assert.True(t, health.Healthy)
}

func TestIsLeader(t *testing.T) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 5 * time.Second,
		ElectionTimeout:   10 * time.Second,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, err := NewManager(config, logger)
	require.NoError(t, err)

	// Initially should not be leader
	assert.False(t, manager.IsLeader())

	// After starting as single node, should become leader
	ctx := context.Background()
	err = manager.Start(ctx)
	require.NoError(t, err)
	defer manager.Stop(ctx)

	// Wait for election
	time.Sleep(200 * time.Millisecond)

	// Should be leader now (single node cluster)
	assert.True(t, manager.IsLeader())
}

func TestNodeMaintenance(t *testing.T) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 50 * time.Millisecond,
		ElectionTimeout:   100 * time.Millisecond,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, err := NewManager(config, logger)
	require.NoError(t, err)

	// Add a stale node
	staleNode := &Node{
		ID:        "stale-node",
		ClusterID: "test-cluster",
		Address:   "127.0.0.1:8082",
		State:     NodeStateActive,
		LastSeen:  time.Now().Add(-1 * time.Hour), // Very old
	}
	manager.addNode(staleNode)

	// Perform maintenance
	manager.performNodeMaintenance()

	// Check that stale node is marked unhealthy
	ctx := context.Background()
	node, err := manager.GetNode(ctx, "stale-node")
	require.NoError(t, err)
	assert.Equal(t, NodeStateUnhealthy, node.State)
}

func TestMetricsCollection(t *testing.T) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 5 * time.Second,
		ElectionTimeout:   10 * time.Second,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, err := NewManager(config, logger)
	require.NoError(t, err)

	// Add some nodes
	for i := 0; i < 5; i++ {
		node := &Node{
			ID:        fmt.Sprintf("node-%d", i),
			ClusterID: "test-cluster",
			Address:   fmt.Sprintf("127.0.0.1:808%d", i),
			State:     NodeStateActive,
		}
		manager.addNode(node)
	}

	// Collect metrics
	manager.collectMetrics()

	// Check metrics
	assert.Equal(t, int64(5), manager.metrics.NodesTotal)
	assert.Equal(t, int64(5), manager.metrics.NodesHealthy)
}

// Benchmark tests

func BenchmarkResourceAllocation(b *testing.B) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 5 * time.Second,
		ElectionTimeout:   10 * time.Second,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, _ := NewManager(config, logger)
	ctx := context.Background()

	// Add nodes with resources
	for i := 0; i < 10; i++ {
		node := &Node{
			ID:        fmt.Sprintf("node-%d", i),
			ClusterID: "test-cluster",
			Address:   fmt.Sprintf("127.0.0.1:808%d", i),
			State:     NodeStateActive,
			Capabilities: NodeCapabilities{
				CPUCores:  16,
				MemoryGB:  64,
				StorageGB: 1000,
				Resources: ResourceInventory{
					TotalCPU:     16,
					UsedCPU:      4,
					TotalMemory:  64 * 1024 * 1024 * 1024,
					UsedMemory:   16 * 1024 * 1024 * 1024,
					TotalStorage: 1000 * 1024 * 1024 * 1024,
					UsedStorage:  200 * 1024 * 1024 * 1024,
				},
			},
		}
		manager.addNode(node)
		manager.resourcePool.UpdateResourceInventory(ctx, node.ID, &node.Capabilities.Resources)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		request := &ResourceRequest{
			ID:           fmt.Sprintf("request-%d", i),
			ResourceType: "vm",
			CPUCores:     2,
			MemoryGB:     8,
			StorageGB:    50,
			Duration:     1 * time.Hour,
			Priority:     5,
		}

		allocation, _ := manager.RequestResources(ctx, request)
		if allocation != nil {
			manager.ReleaseResources(ctx, allocation.ID)
		}
	}
}

func BenchmarkHealthCheck(b *testing.B) {
	logger := &MockLogger{}
	config := &FederationConfig{
		ClusterID:         "test-cluster",
		NodeID:            "test-node-1",
		BindAddress:       "127.0.0.1:8080",
		AdvertiseAddress:  "127.0.0.1:8080",
		HeartbeatInterval: 5 * time.Second,
		ElectionTimeout:   10 * time.Second,
		FailureThreshold:  3,
		EnableMDNS:        false,
		EnableGossip:      false,
	}

	manager, _ := NewManager(config, logger)
	ctx := context.Background()

	node := &Node{
		ID:        "test-node",
		ClusterID: "test-cluster",
		Address:   "127.0.0.1:8081",
		State:     NodeStateActive,
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		manager.healthChecker.CheckHealth(ctx, node)
	}
}