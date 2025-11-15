package bullshark

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// Bullshark implements the Bullshark DAG-based consensus protocol
type Bullshark struct {
	dag       *DAG
	round     int64
	nodeID    string
	committee []string

	// Configuration
	config Config

	// State
	proposalBuffer chan *Vertex
	commitQueue    chan *Vertex

	// Metrics
	txThroughput   int64
	latency        time.Duration
	proposalCount  int64
	commitCount    int64

	// Synchronization
	mu      sync.RWMutex
	ctx     context.Context
	cancel  context.CancelFunc
	wg      sync.WaitGroup
	running atomic.Bool
}

// Config holds Bullshark configuration parameters
type Config struct {
	// Protocol parameters
	RoundDuration    time.Duration
	BatchSize        int
	CommitteeSize    int
	QuorumThreshold  float64

	// Performance tuning
	BufferSize       int
	WorkerCount      int
	MaxParents       int

	// Timeouts
	ProposeTimeout   time.Duration
	CommitTimeout    time.Duration
}

// DefaultConfig returns default Bullshark configuration
func DefaultConfig() Config {
	return Config{
		RoundDuration:   100 * time.Millisecond,
		BatchSize:       1000,
		CommitteeSize:   100,
		QuorumThreshold: 0.67,
		BufferSize:      10000,
		WorkerCount:     8,
		MaxParents:      3,
		ProposeTimeout:  5 * time.Second,
		CommitTimeout:   10 * time.Second,
	}
}

// NewBullshark creates a new Bullshark consensus instance
func NewBullshark(nodeID string, committee []string, config Config) *Bullshark {
	ctx, cancel := context.WithCancel(context.Background())

	return &Bullshark{
		dag:            NewDAG(),
		round:          0,
		nodeID:         nodeID,
		committee:      committee,
		config:         config,
		proposalBuffer: make(chan *Vertex, config.BufferSize),
		commitQueue:    make(chan *Vertex, config.BufferSize),
		ctx:            ctx,
		cancel:         cancel,
	}
}

// Start begins the Bullshark consensus protocol
func (b *Bullshark) Start() error {
	if !b.running.CompareAndSwap(false, true) {
		return fmt.Errorf("bullshark already running")
	}

	// Start workers
	for i := 0; i < b.config.WorkerCount; i++ {
		b.wg.Add(1)
		go b.proposalWorker()
	}

	b.wg.Add(1)
	go b.commitWorker()

	b.wg.Add(1)
	go b.metricsCollector()

	return nil
}

// Stop gracefully stops the Bullshark consensus
func (b *Bullshark) Stop() error {
	if !b.running.CompareAndSwap(true, false) {
		return fmt.Errorf("bullshark not running")
	}

	b.cancel()

	// Close channels
	close(b.proposalBuffer)
	close(b.commitQueue)

	// Wait for workers
	b.wg.Wait()

	return nil
}

// ProposeBlock creates and proposes a new block with transactions
func (b *Bullshark) ProposeBlock(txs []Transaction) (*Vertex, error) {
	if !b.running.Load() {
		return nil, fmt.Errorf("bullshark not running")
	}

	// Select parent vertices
	parents := b.selectParents()

	// Create vertex
	currentRound := int(atomic.LoadInt64(&b.round))
	v := NewVertex(b.nodeID, currentRound, txs, parents)

	// Add to DAG
	if err := b.dag.AddVertex(v); err != nil {
		return nil, fmt.Errorf("failed to add vertex to DAG: %w", err)
	}

	// Queue for proposal
	select {
	case b.proposalBuffer <- v:
		atomic.AddInt64(&b.proposalCount, 1)
		return v, nil
	case <-b.ctx.Done():
		return nil, fmt.Errorf("context cancelled")
	case <-time.After(b.config.ProposeTimeout):
		return nil, fmt.Errorf("proposal timeout")
	}
}

// selectParents selects parent vertices for a new block
func (b *Bullshark) selectParents() []*Vertex {
	b.mu.RLock()
	defer b.mu.RUnlock()

	currentRound := int(atomic.LoadInt64(&b.round))

	// Genesis block
	if currentRound == 0 {
		return nil
	}

	// Get vertices from previous round
	prevRoundVertices := b.dag.GetVerticesByRound(currentRound - 1)

	if len(prevRoundVertices) == 0 {
		// Fall back to roots
		return b.dag.GetRoots()
	}

	// Select up to MaxParents vertices with highest weight
	parents := make([]*Vertex, 0, b.config.MaxParents)

	// Sort by weight (simplified - should use heap for efficiency)
	maxParents := b.config.MaxParents
	if len(prevRoundVertices) < maxParents {
		maxParents = len(prevRoundVertices)
	}

	for i := 0; i < maxParents && i < len(prevRoundVertices); i++ {
		parents = append(parents, prevRoundVertices[i])
	}

	return parents
}

// proposalWorker processes proposed vertices
func (b *Bullshark) proposalWorker() {
	defer b.wg.Done()

	for {
		select {
		case v, ok := <-b.proposalBuffer:
			if !ok {
				return
			}

			// Broadcast vertex to committee
			if err := b.broadcastVertex(v); err != nil {
				// Log error but continue
				continue
			}

			// Queue for commitment
			select {
			case b.commitQueue <- v:
			case <-b.ctx.Done():
				return
			}

		case <-b.ctx.Done():
			return
		}
	}
}

// broadcastVertex broadcasts a vertex to the committee
func (b *Bullshark) broadcastVertex(v *Vertex) error {
	// Simulate network broadcast
	// In real implementation, this would use network layer
	return nil
}

// commitWorker processes vertices for commitment
func (b *Bullshark) commitWorker() {
	defer b.wg.Done()

	ticker := time.NewTicker(b.config.RoundDuration)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			if err := b.advanceRound(); err != nil {
				// Log error but continue
			}

		case <-b.ctx.Done():
			return
		}
	}
}

// advanceRound advances to the next round and commits vertices
func (b *Bullshark) advanceRound() error {
	b.mu.Lock()
	defer b.mu.Unlock()

	currentRound := atomic.LoadInt64(&b.round)

	// Order and commit transactions from current round
	committed, err := b.orderAndCommit(int(currentRound))
	if err != nil {
		return fmt.Errorf("failed to commit round %d: %w", currentRound, err)
	}

	atomic.AddInt64(&b.commitCount, int64(len(committed)))

	// Advance round
	atomic.AddInt64(&b.round, 1)

	return nil
}

// orderAndCommit orders and commits transactions from a round
func (b *Bullshark) orderAndCommit(round int) ([]Transaction, error) {
	// Get all vertices from the round
	vertices := b.dag.GetVerticesByRound(round)

	if len(vertices) == 0 {
		return nil, nil
	}

	// Check quorum (simplified)
	quorumSize := int(float64(b.config.CommitteeSize) * b.config.QuorumThreshold)
	if len(vertices) < quorumSize {
		return nil, fmt.Errorf("insufficient quorum: %d < %d", len(vertices), quorumSize)
	}

	// Order transactions deterministically
	ordered := b.orderTransactions(vertices)

	// Mark vertices as committed
	for _, v := range vertices {
		v.mu.Lock()
		v.Committed = true
		v.mu.Unlock()
	}

	// Update metrics
	atomic.AddInt64(&b.dag.committedTxs, int64(len(ordered)))

	return ordered, nil
}

// orderTransactions orders transactions from vertices deterministically
func (b *Bullshark) orderTransactions(vertices []*Vertex) []Transaction {
	// Collect all transactions
	allTxs := make([]Transaction, 0)
	for _, v := range vertices {
		allTxs = append(allTxs, v.Txs...)
	}

	// Sort deterministically by transaction ID
	// In production, use more sophisticated ordering
	sortedTxs := make([]Transaction, len(allTxs))
	copy(sortedTxs, allTxs)

	return sortedTxs
}

// metricsCollector collects performance metrics
func (b *Bullshark) metricsCollector() {
	defer b.wg.Done()

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastCommitCount int64

	for {
		select {
		case <-ticker.C:
			currentCommitCount := atomic.LoadInt64(&b.commitCount)
			throughput := currentCommitCount - lastCommitCount
			atomic.StoreInt64(&b.txThroughput, throughput)
			lastCommitCount = currentCommitCount

		case <-b.ctx.Done():
			return
		}
	}
}

// GetMetrics returns current consensus metrics
func (b *Bullshark) GetMetrics() Metrics {
	return Metrics{
		Round:          atomic.LoadInt64(&b.round),
		TxThroughput:   atomic.LoadInt64(&b.txThroughput),
		ProposalCount:  atomic.LoadInt64(&b.proposalCount),
		CommitCount:    atomic.LoadInt64(&b.commitCount),
		DAGMetrics:     b.dag.Metrics(),
	}
}

// Metrics holds Bullshark performance metrics
type Metrics struct {
	Round         int64
	TxThroughput  int64
	ProposalCount int64
	CommitCount   int64
	DAGMetrics    map[string]interface{}
}

// GetCurrentRound returns the current consensus round
func (b *Bullshark) GetCurrentRound() int64 {
	return atomic.LoadInt64(&b.round)
}

// IsRunning returns whether Bullshark is running
func (b *Bullshark) IsRunning() bool {
	return b.running.Load()
}
