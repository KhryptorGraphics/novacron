package sync

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/yourusername/novacron/backend/core/network/dwcp/sync/crdt"
	"go.uber.org/zap"
)

// MockTransport is a mock transport for testing
type MockTransport struct {
	messages chan *Message
	peers    map[string]*RegionPeer
}

func NewMockTransport() *MockTransport {
	return &MockTransport{
		messages: make(chan *Message, 100),
		peers:    make(map[string]*RegionPeer),
	}
}

func (mt *MockTransport) Send(peer *RegionPeer, message *Message) error {
	mt.messages <- message
	return nil
}

func (mt *MockTransport) Receive() (*Message, error) {
	select {
	case msg := <-mt.messages:
		return msg, nil
	case <-time.After(1 * time.Second):
		return nil, &SyncError{Message: "receive timeout"}
	}
}

func (mt *MockTransport) Close() error {
	close(mt.messages)
	return nil
}

// TestCRDTConvergence tests that CRDTs converge to the same state
func TestCRDTConvergence(t *testing.T) {
	tests := []struct {
		name     string
		crdtType string
		setup    func(node1, node2 crdt.CvRDT)
		verify   func(t *testing.T, node1, node2 crdt.CvRDT)
	}{
		{
			name:     "G-Counter convergence",
			crdtType: "g_counter",
			setup: func(node1, node2 crdt.CvRDT) {
				node1.(*crdt.GCounter).Increment(5)
				node2.(*crdt.GCounter).Increment(3)
			},
			verify: func(t *testing.T, node1, node2 crdt.CvRDT) {
				assert.Equal(t, uint64(8), node1.Value())
				assert.Equal(t, uint64(8), node2.Value())
			},
		},
		{
			name:     "OR-Set convergence",
			crdtType: "or_set",
			setup: func(node1, node2 crdt.CvRDT) {
				set1 := node1.(*crdt.ORSet)
				set2 := node2.(*crdt.ORSet)

				set1.Add("a")
				set1.Add("b")
				set2.Add("c")
				set2.Add("d")
				set1.Add("e")
			},
			verify: func(t *testing.T, node1, node2 crdt.CvRDT) {
				set1 := node1.(*crdt.ORSet)
				set2 := node2.(*crdt.ORSet)

				assert.True(t, set1.Contains("a"))
				assert.True(t, set1.Contains("b"))
				assert.True(t, set1.Contains("c"))
				assert.True(t, set1.Contains("d"))
				assert.True(t, set1.Contains("e"))

				assert.True(t, set2.Contains("a"))
				assert.True(t, set2.Contains("b"))
				assert.True(t, set2.Contains("c"))
				assert.True(t, set2.Contains("d"))
				assert.True(t, set2.Contains("e"))
			},
		},
		{
			name:     "LWW-Register convergence",
			crdtType: "lww_register",
			setup: func(node1, node2 crdt.CvRDT) {
				reg1 := node1.(*crdt.LWWRegister)
				reg2 := node2.(*crdt.LWWRegister)

				reg1.Set("value1")
				time.Sleep(10 * time.Millisecond)
				reg2.Set("value2")
			},
			verify: func(t *testing.T, node1, node2 crdt.CvRDT) {
				// Both should have the later value
				assert.Equal(t, "value2", node1.Value())
				assert.Equal(t, "value2", node2.Value())
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var node1, node2 crdt.CvRDT

			switch tt.crdtType {
			case "g_counter":
				node1 = crdt.NewGCounter("node1")
				node2 = crdt.NewGCounter("node2")
			case "or_set":
				node1 = crdt.NewORSet("node1")
				node2 = crdt.NewORSet("node2")
			case "lww_register":
				node1 = crdt.NewLWWRegister("node1")
				node2 = crdt.NewLWWRegister("node2")
			}

			// Perform concurrent updates
			tt.setup(node1, node2)

			// Merge states
			err := node1.Merge(node2)
			require.NoError(t, err)

			err = node2.Merge(node1)
			require.NoError(t, err)

			// Verify convergence
			tt.verify(t, node1, node2)
		})
	}
}

// TestConcurrentUpdates tests handling of concurrent updates
func TestConcurrentUpdates(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	transport1 := NewMockTransport()
	transport2 := NewMockTransport()

	engine1 := NewASSEngine("node1", transport1, logger)
	engine2 := NewASSEngine("node2", transport2, logger)

	// Register each other as peers
	peer1 := &RegionPeer{ID: "node1", Region: "us-east", Endpoint: "node1:8080"}
	peer2 := &RegionPeer{ID: "node2", Region: "eu-west", Endpoint: "node2:8080"}

	engine1.RegisterPeer(peer2)
	engine2.RegisterPeer(peer1)

	// Perform concurrent updates
	counter1 := crdt.NewGCounter("node1")
	counter1.Increment(10)
	err := engine1.Set("counter", counter1)
	require.NoError(t, err)

	counter2 := crdt.NewGCounter("node2")
	counter2.Increment(15)
	err = engine2.Set("counter", counter2)
	require.NoError(t, err)

	// Simulate synchronization by merging states
	value1, _ := engine1.Get("counter")
	value2, _ := engine2.Get("counter")

	err = value1.Merge(value2)
	require.NoError(t, err)

	// Verify convergence
	assert.Equal(t, uint64(25), value1.Value())
}

// TestNetworkPartition tests handling of network partitions
func TestNetworkPartition(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Create three nodes
	transport1 := NewMockTransport()
	transport2 := NewMockTransport()
	transport3 := NewMockTransport()

	engine1 := NewASSEngine("node1", transport1, logger)
	engine2 := NewASSEngine("node2", transport2, logger)
	engine3 := NewASSEngine("node3", transport3, logger)

	// Initially all nodes can communicate
	peer1 := &RegionPeer{ID: "node1", Region: "us-east", Endpoint: "node1:8080"}
	peer2 := &RegionPeer{ID: "node2", Region: "eu-west", Endpoint: "node2:8080"}
	peer3 := &RegionPeer{ID: "node3", Region: "ap-south", Endpoint: "node3:8080"}

	engine1.RegisterPeer(peer2)
	engine1.RegisterPeer(peer3)
	engine2.RegisterPeer(peer1)
	engine2.RegisterPeer(peer3)
	engine3.RegisterPeer(peer1)
	engine3.RegisterPeer(peer2)

	// Update on node1
	set1 := crdt.NewORSet("node1")
	set1.Add("item1")
	engine1.Set("items", set1)

	// Simulate partition: node2 and node3 can't reach node1
	// But they make updates independently

	set2 := crdt.NewORSet("node2")
	set2.Add("item2")
	engine2.Set("items", set2)

	set3 := crdt.NewORSet("node3")
	set3.Add("item3")
	engine3.Set("items", set3)

	// Partition heals - merge all states
	value1, _ := engine1.Get("items")
	value2, _ := engine2.Get("items")
	value3, _ := engine3.Get("items")

	value1.Merge(value2)
	value1.Merge(value3)

	// Verify all updates are preserved
	finalSet := value1.(*crdt.ORSet)
	assert.True(t, finalSet.Contains("item1"))
	assert.True(t, finalSet.Contains("item2"))
	assert.True(t, finalSet.Contains("item3"))
}

// TestAntiEntropyConvergence tests anti-entropy convergence
func TestAntiEntropyConvergence(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	transport1 := NewMockTransport()
	transport2 := NewMockTransport()

	engine1 := NewASSEngine("node1", transport1, logger)
	engine2 := NewASSEngine("node2", transport2, logger)

	peer1 := &RegionPeer{ID: "node1", Region: "us-east", Endpoint: "node1:8080"}
	peer2 := &RegionPeer{ID: "node2", Region: "eu-west", Endpoint: "node2:8080"}

	engine1.RegisterPeer(peer2)
	engine2.RegisterPeer(peer1)

	// Node 1 has some data
	for i := 0; i < 10; i++ {
		counter := crdt.NewGCounter("node1")
		counter.Increment(uint64(i))
		engine1.Set(string(rune('a'+i)), counter)
	}

	// Node 2 has different data
	for i := 0; i < 10; i++ {
		counter := crdt.NewGCounter("node2")
		counter.Increment(uint64(i * 2))
		engine2.Set(string(rune('k'+i)), counter)
	}

	// Perform digest-based synchronization
	digest1 := engine1.crdtStore.Digest("node1", engine1.vectorClock.Get())
	digest2 := engine2.crdtStore.Digest("node2", engine2.vectorClock.Get())

	delta := engine1.computeDelta(digest1, digest2)

	// Verify delta contains correct missing keys
	assert.Greater(t, len(delta.Missing), 0)
	assert.Greater(t, len(delta.Theirs), 0)
}

// TestStateConvergenceTime tests convergence time
func TestStateConvergenceTime(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	// Create 5 nodes simulating different regions
	nodes := make([]*ASSEngine, 5)
	transports := make([]*MockTransport, 5)

	for i := 0; i < 5; i++ {
		transports[i] = NewMockTransport()
		nodes[i] = NewASSEngine(string(rune('A'+i)), transports[i], logger)
	}

	// Fully connect all nodes
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			if i != j {
				peer := &RegionPeer{
					ID:       string(rune('A' + j)),
					Region:   "region-" + string(rune('A'+j)),
					Endpoint: string(rune('A'+j)) + ":8080",
				}
				nodes[i].RegisterPeer(peer)
			}
		}
	}

	// Perform 1000 random updates across all nodes
	startTime := time.Now()

	for i := 0; i < 1000; i++ {
		nodeIdx := i % 5
		counter := crdt.NewGCounter(string(rune('A' + nodeIdx)))
		counter.Increment(1)
		nodes[nodeIdx].Set("vm-counter", counter)
	}

	// Simulate convergence by merging all states
	for i := 1; i < 5; i++ {
		value0, _ := nodes[0].Get("vm-counter")
		valuei, _ := nodes[i].Get("vm-counter")
		if value0 != nil && valuei != nil {
			value0.Merge(valuei)
		}
	}

	convergenceTime := time.Since(startTime)

	// Verify convergence time is acceptable (< 5 minutes requirement)
	assert.Less(t, convergenceTime, 5*time.Minute)

	t.Logf("Convergence time for 1000 updates across 5 nodes: %v", convergenceTime)
}

// TestClusterMetadata tests cluster metadata operations
func TestClusterMetadata(t *testing.T) {
	cm := NewClusterMetadata("node1")

	// Test VM state updates
	vmState := VMState{
		ID:       "vm1",
		Status:   "running",
		NodeID:   "node1",
		CPUCores: 4,
		MemoryMB: 8192,
	}

	err := cm.UpdateVMState("vm1", vmState)
	require.NoError(t, err)

	retrieved, err := cm.GetVMState("vm1")
	require.NoError(t, err)
	assert.Equal(t, "vm1", retrieved.ID)
	assert.Equal(t, "running", retrieved.Status)
	assert.Equal(t, 4, retrieved.CPUCores)

	// Test node status updates
	nodeStatus := NodeStatus{
		ID:        "node1",
		Region:    "us-east",
		Status:    "active",
		CPUUsage:  45.5,
		VMCount:   10,
	}

	err = cm.UpdateNodeStatus("node1", nodeStatus)
	require.NoError(t, err)

	retrievedNode, err := cm.GetNodeStatus("node1")
	require.NoError(t, err)
	assert.Equal(t, "node1", retrievedNode.ID)
	assert.Equal(t, "active", retrievedNode.Status)

	// Test VM assignments
	err = cm.AssignVM("vm1", "node1")
	require.NoError(t, err)

	assignment, err := cm.GetVMAssignment("vm1")
	require.NoError(t, err)
	assert.Equal(t, "node1", assignment)
}

// TestClusterMetadataMerge tests merging of cluster metadata
func TestClusterMetadataMerge(t *testing.T) {
	cm1 := NewClusterMetadata("node1")
	cm2 := NewClusterMetadata("node2")

	// Node 1 updates
	cm1.UpdateVMState("vm1", VMState{ID: "vm1", Status: "running", NodeID: "node1"})
	cm1.UpdateNodeStatus("node1", NodeStatus{ID: "node1", Status: "active"})

	// Node 2 updates
	cm2.UpdateVMState("vm2", VMState{ID: "vm2", Status: "running", NodeID: "node2"})
	cm2.UpdateNodeStatus("node2", NodeStatus{ID: "node2", Status: "active"})

	// Merge
	err := cm1.Merge(cm2)
	require.NoError(t, err)

	// Verify both nodes have all data
	vms, err := cm1.ListVMs()
	require.NoError(t, err)
	assert.Equal(t, 2, len(vms))

	nodes, err := cm1.ListNodes()
	require.NoError(t, err)
	assert.Equal(t, 2, len(nodes))
}

// TestNovaCronIntegration tests the NovaCron integration
func TestNovaCronIntegration(t *testing.T) {
	logger, _ := zap.NewDevelopment()
	transport := NewMockTransport()

	config := IntegrationConfig{
		NodeID:              "node1",
		Region:              "us-east",
		AntiEntropyInterval: 30 * time.Second,
		GossipFanout:        3,
		GossipInterval:      5 * time.Second,
		MaxGossipHops:       10,
		Transport:           transport,
		Logger:              logger,
	}

	integration := NewNovaCronIntegration(config)

	err := integration.Start()
	require.NoError(t, err)

	// Test VM operations
	vmState := VMState{
		ID:       "vm-test-1",
		Status:   "running",
		NodeID:   "node1",
		CPUCores: 2,
		MemoryMB: 4096,
	}

	err = integration.UpdateVMState("vm-test-1", vmState)
	require.NoError(t, err)

	retrieved, err := integration.GetVMState("vm-test-1")
	require.NoError(t, err)
	assert.Equal(t, "vm-test-1", retrieved.ID)

	// Test node operations
	nodeStatus := NodeStatus{
		ID:     "node1",
		Region: "us-east",
		Status: "active",
	}

	err = integration.UpdateNodeStatus("node1", nodeStatus)
	require.NoError(t, err)

	// Get statistics
	stats := integration.GetStats()
	assert.Greater(t, stats.CRDTCount, 0)

	err = integration.Stop()
	require.NoError(t, err)
}

// BenchmarkCRDTOperations benchmarks CRDT operations
func BenchmarkCRDTOperations(b *testing.B) {
	b.Run("GCounter-Increment", func(b *testing.B) {
		counter := crdt.NewGCounter("node1")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			counter.Increment(1)
		}
	})

	b.Run("ORSet-Add", func(b *testing.B) {
		set := crdt.NewORSet("node1")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			set.Add(string(rune(i)))
		}
	})

	b.Run("LWWRegister-Set", func(b *testing.B) {
		register := crdt.NewLWWRegister("node1")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			register.Set(i)
		}
	})

	b.Run("ORMap-Set", func(b *testing.B) {
		ormap := crdt.NewORMap("node1")
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ormap.SetLWW(string(rune(i)), i)
		}
	})
}

// BenchmarkSerialization benchmarks CRDT serialization
func BenchmarkSerialization(b *testing.B) {
	counter := crdt.NewGCounter("node1")
	for i := 0; i < 1000; i++ {
		counter.Increment(1)
	}

	b.Run("Marshal", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = counter.Marshal()
		}
	})

	data, _ := counter.Marshal()

	b.Run("Unmarshal", func(b *testing.B) {
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			newCounter := crdt.NewGCounter("node1")
			_ = newCounter.Unmarshal(data)
		}
	})
}
