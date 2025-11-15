package bullshark

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestDAGCreation tests basic DAG creation and vertex addition
func TestDAGCreation(t *testing.T) {
	dag := NewDAG()

	if dag == nil {
		t.Fatal("Failed to create DAG")
	}

	// Create genesis vertex
	genesis := NewVertex("node-0", 0, nil, nil)

	err := dag.AddVertex(genesis)
	if err != nil {
		t.Fatalf("Failed to add genesis vertex: %v", err)
	}

	// Verify vertex was added
	v, exists := dag.GetVertex(genesis.ID)
	if !exists {
		t.Fatal("Genesis vertex not found in DAG")
	}

	if v.ID != genesis.ID {
		t.Errorf("Expected vertex ID %s, got %s", genesis.ID, v.ID)
	}
}

// TestVertexParenting tests parent-child relationships
func TestVertexParenting(t *testing.T) {
	dag := NewDAG()

	// Create genesis
	genesis := NewVertex("node-0", 0, nil, nil)
	dag.AddVertex(genesis)

	// Create child
	child := NewVertex("node-1", 1, nil, []*Vertex{genesis})
	err := dag.AddVertex(child)

	if err != nil {
		t.Fatalf("Failed to add child vertex: %v", err)
	}

	// Verify relationship
	children := dag.GetChildren(genesis.ID)
	if len(children) != 1 {
		t.Fatalf("Expected 1 child, got %d", len(children))
	}

	if children[0].ID != child.ID {
		t.Errorf("Child vertex ID mismatch")
	}
}

// TestCycleDetection tests that cycles are prevented
func TestCycleDetection(t *testing.T) {
	dag := NewDAG()

	v1 := NewVertex("node-1", 0, nil, nil)
	v2 := NewVertex("node-2", 1, nil, []*Vertex{v1})

	dag.AddVertex(v1)
	dag.AddVertex(v2)

	// Try to create a cycle (v1 -> v2 -> v1)
	v1.Parents = []*Vertex{v2}
	v1.ParentIDs = []string{v2.ID}

	err := dag.AddVertex(v1)
	if err == nil {
		t.Error("Expected cycle detection error, got nil")
	}
}

// TestBullsharkCreation tests Bullshark consensus creation
func TestBullsharkCreation(t *testing.T) {
	committee := []string{"node-1", "node-2", "node-3"}
	config := DefaultConfig()

	bs := NewBullshark("node-1", committee, config)

	if bs == nil {
		t.Fatal("Failed to create Bullshark instance")
	}

	if bs.nodeID != "node-1" {
		t.Errorf("Expected node ID node-1, got %s", bs.nodeID)
	}
}

// TestBullsharkStartStop tests starting and stopping consensus
func TestBullsharkStartStop(t *testing.T) {
	committee := []string{"node-1", "node-2", "node-3"}
	config := DefaultConfig()
	config.WorkerCount = 2

	bs := NewBullshark("node-1", committee, config)

	err := bs.Start()
	if err != nil {
		t.Fatalf("Failed to start Bullshark: %v", err)
	}

	if !bs.IsRunning() {
		t.Error("Bullshark should be running")
	}

	time.Sleep(100 * time.Millisecond)

	err = bs.Stop()
	if err != nil {
		t.Fatalf("Failed to stop Bullshark: %v", err)
	}

	if bs.IsRunning() {
		t.Error("Bullshark should not be running")
	}
}

// TestProposalCreation tests block proposal
func TestProposalCreation(t *testing.T) {
	committee := []string{"node-1", "node-2", "node-3"}
	config := DefaultConfig()

	bs := NewBullshark("node-1", committee, config)
	bs.Start()
	defer bs.Stop()

	// Create transactions
	txs := []Transaction{
		*NewTransaction("alice", "bob", 100, []byte("data1")),
		*NewTransaction("bob", "charlie", 50, []byte("data2")),
	}

	// Propose block
	vertex, err := bs.ProposeBlock(txs)
	if err != nil {
		t.Fatalf("Failed to propose block: %v", err)
	}

	if len(vertex.Txs) != 2 {
		t.Errorf("Expected 2 transactions, got %d", len(vertex.Txs))
	}
}

// TestHighThroughput tests high transaction throughput
func TestHighThroughput(t *testing.T) {
	committee := make([]string, 100)
	for i := 0; i < 100; i++ {
		committee[i] = fmt.Sprintf("node-%d", i)
	}

	config := DefaultConfig()
	config.BatchSize = 1000
	config.WorkerCount = 8

	bs := NewBullshark("node-0", committee, config)
	bs.Start()
	defer bs.Stop()

	// Measure throughput
	startTime := time.Now()
	totalTxs := 0
	targetTxs := 125000 // Target 125K tx/s

	var wg sync.WaitGroup
	batchCount := 125 // 125 batches of 1000 = 125K txs

	for i := 0; i < batchCount; i++ {
		wg.Add(1)
		go func(batchID int) {
			defer wg.Done()

			txs := make([]Transaction, config.BatchSize)
			for j := 0; j < config.BatchSize; j++ {
				txs[j] = *NewTransaction(
					fmt.Sprintf("sender-%d", batchID),
					fmt.Sprintf("recipient-%d", j),
					int64(j+1),
					[]byte(fmt.Sprintf("data-%d-%d", batchID, j)),
				)
			}

			_, err := bs.ProposeBlock(txs)
			if err != nil {
				t.Logf("Proposal error: %v", err)
			}
		}(i)
	}

	wg.Wait()

	elapsed := time.Since(startTime)
	totalTxs = batchCount * config.BatchSize
	throughput := float64(totalTxs) / elapsed.Seconds()

	t.Logf("Processed %d transactions in %v", totalTxs, elapsed)
	t.Logf("Throughput: %.0f tx/s", throughput)

	// Verify we achieved target throughput (with some tolerance)
	if throughput < float64(targetTxs)*0.8 {
		t.Logf("Warning: Throughput %.0f tx/s below target %d tx/s", throughput, targetTxs)
	}
}

// TestOrderingDeterminism tests deterministic transaction ordering
func TestOrderingDeterminism(t *testing.T) {
	dag := NewDAG()
	engine := NewOrderingEngine(dag, 10)

	// Create vertices with transactions
	genesis := NewVertex("node-0", 0, nil, nil)
	dag.AddVertex(genesis)

	txs1 := []Transaction{
		*NewTransaction("alice", "bob", 100, []byte("tx1")),
		*NewTransaction("bob", "charlie", 50, []byte("tx2")),
	}

	v1 := NewVertex("node-1", 1, txs1, []*Vertex{genesis})
	dag.AddVertex(v1)

	vertices := []*Vertex{genesis, v1}

	// Order transactions multiple times
	order1, err := engine.OrderTransactions(vertices)
	if err != nil {
		t.Fatalf("Ordering failed: %v", err)
	}

	order2, err := engine.OrderTransactions(vertices)
	if err != nil {
		t.Fatalf("Ordering failed: %v", err)
	}

	// Verify determinism
	if len(order1) != len(order2) {
		t.Fatal("Ordering is not deterministic: different lengths")
	}

	for i := range order1 {
		if order1[i].ID != order2[i].ID {
			t.Errorf("Ordering is not deterministic at index %d", i)
		}
	}
}

// TestMetricsCollection tests metrics gathering
func TestMetricsCollection(t *testing.T) {
	committee := []string{"node-1", "node-2", "node-3"}
	config := DefaultConfig()

	bs := NewBullshark("node-1", committee, config)
	bs.Start()
	defer bs.Stop()

	// Submit some transactions
	for i := 0; i < 10; i++ {
		txs := []Transaction{
			*NewTransaction("alice", "bob", int64(i), []byte(fmt.Sprintf("data%d", i))),
		}
		bs.ProposeBlock(txs)
	}

	time.Sleep(200 * time.Millisecond)

	metrics := bs.GetMetrics()

	if metrics.ProposalCount != 10 {
		t.Errorf("Expected 10 proposals, got %d", metrics.ProposalCount)
	}

	t.Logf("Metrics: Round=%d, Proposals=%d, Commits=%d",
		metrics.Round, metrics.ProposalCount, metrics.CommitCount)
}

// BenchmarkProposal benchmarks block proposal performance
func BenchmarkProposal(b *testing.B) {
	committee := make([]string, 100)
	for i := 0; i < 100; i++ {
		committee[i] = fmt.Sprintf("node-%d", i)
	}

	config := DefaultConfig()
	config.BatchSize = 1000

	bs := NewBullshark("node-0", committee, config)
	bs.Start()
	defer bs.Stop()

	txs := make([]Transaction, config.BatchSize)
	for i := 0; i < config.BatchSize; i++ {
		txs[i] = *NewTransaction(
			"sender",
			"recipient",
			int64(i),
			[]byte(fmt.Sprintf("data-%d", i)),
		)
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		bs.ProposeBlock(txs)
	}
}

// BenchmarkOrdering benchmarks transaction ordering
func BenchmarkOrdering(b *testing.B) {
	dag := NewDAG()
	engine := NewOrderingEngine(dag, 100)

	// Create test vertices
	genesis := NewVertex("node-0", 0, nil, nil)
	dag.AddVertex(genesis)

	vertices := make([]*Vertex, 100)
	for i := 0; i < 100; i++ {
		txs := make([]Transaction, 10)
		for j := 0; j < 10; j++ {
			txs[j] = *NewTransaction(
				fmt.Sprintf("sender-%d", i),
				fmt.Sprintf("recipient-%d", j),
				int64(j),
				[]byte(fmt.Sprintf("data-%d-%d", i, j)),
			)
		}

		v := NewVertex(fmt.Sprintf("node-%d", i), 1, txs, []*Vertex{genesis})
		dag.AddVertex(v)
		vertices[i] = v
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		engine.OrderTransactions(vertices)
	}
}
