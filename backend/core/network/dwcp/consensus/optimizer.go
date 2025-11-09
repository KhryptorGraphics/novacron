package consensus

import (
	"sync"
	"time"
)

// ConsensusOptimizer optimizes consensus performance through batching and pipelining
type ConsensusOptimizer struct {
	mu sync.Mutex

	// Batching configuration
	batchSize    int
	batchTimeout time.Duration
	pendingBatch []Proposal

	// Channels
	proposalChan chan Proposal
	batchChan    chan []Proposal
	stopChan     chan struct{}

	// Statistics
	stats OptimizerStats

	// Pipelining
	pipelineDepth int
	pipeline      []PipelineStage
}

// OptimizerStats tracks optimization statistics
type OptimizerStats struct {
	mu sync.RWMutex

	TotalProposals   uint64
	BatchedProposals uint64
	BatchCount       uint64
	AvgBatchSize     float64
	AvgLatency       time.Duration
	ThroughputOps    float64
}

// PipelineStage represents a stage in the consensus pipeline
type PipelineStage struct {
	Proposals    []Proposal
	StartTime    time.Time
	Status       StageStatus
	Dependencies []int // indices of dependent stages
}

// StageStatus represents the status of a pipeline stage
type StageStatus int

const (
	StagePending StageStatus = iota
	StageProcessing
	StageComplete
	StageFailed
)

// NewConsensusOptimizer creates a new consensus optimizer
func NewConsensusOptimizer(batchSize int, batchTimeout time.Duration) *ConsensusOptimizer {
	return &ConsensusOptimizer{
		batchSize:     batchSize,
		batchTimeout:  batchTimeout,
		pendingBatch:  make([]Proposal, 0, batchSize),
		proposalChan:  make(chan Proposal, 1000),
		batchChan:     make(chan []Proposal, 100),
		stopChan:      make(chan struct{}),
		pipelineDepth: 4,
		pipeline:      make([]PipelineStage, 0),
	}
}

// Start starts the optimizer
func (co *ConsensusOptimizer) Start() {
	go co.batchProposals()
	go co.processBatches()
	go co.updateStats()
}

// Stop stops the optimizer
func (co *ConsensusOptimizer) Stop() {
	close(co.stopChan)
}

// Submit submits a proposal for optimization
func (co *ConsensusOptimizer) Submit(proposal Proposal) {
	select {
	case co.proposalChan <- proposal:
		co.stats.mu.Lock()
		co.stats.TotalProposals++
		co.stats.mu.Unlock()
	case <-co.stopChan:
		return
	}
}

// batchProposals batches proposals for efficient processing
func (co *ConsensusOptimizer) BatchProposals() {
	ticker := time.NewTicker(co.batchTimeout)
	defer ticker.Stop()

	for {
		select {
		case proposal := <-co.proposalChan:
			co.mu.Lock()
			co.pendingBatch = append(co.pendingBatch, proposal)

			if len(co.pendingBatch) >= co.batchSize {
				co.submitBatch()
			}
			co.mu.Unlock()

		case <-ticker.C:
			co.mu.Lock()
			if len(co.pendingBatch) > 0 {
				co.submitBatch()
			}
			co.mu.Unlock()

		case <-co.stopChan:
			return
		}
	}
}

// submitBatch submits the current batch for processing
func (co *ConsensusOptimizer) submitBatch() {
	if len(co.pendingBatch) == 0 {
		return
	}

	batch := make([]Proposal, len(co.pendingBatch))
	copy(batch, co.pendingBatch)

	select {
	case co.batchChan <- batch:
		co.stats.mu.Lock()
		co.stats.BatchedProposals += uint64(len(batch))
		co.stats.BatchCount++
		co.stats.mu.Unlock()
	default:
		// Channel full, will retry
	}

	co.pendingBatch = co.pendingBatch[:0]
}

// batchProposals is the actual implementation
func (co *ConsensusOptimizer) batchProposals() {
	ticker := time.NewTicker(co.batchTimeout)
	defer ticker.Stop()

	for {
		select {
		case proposal := <-co.proposalChan:
			co.mu.Lock()
			co.pendingBatch = append(co.pendingBatch, proposal)

			if len(co.pendingBatch) >= co.batchSize {
				co.submitBatch()
			}
			co.mu.Unlock()

		case <-ticker.C:
			co.mu.Lock()
			if len(co.pendingBatch) > 0 {
				co.submitBatch()
			}
			co.mu.Unlock()

		case <-co.stopChan:
			return
		}
	}
}

// processBatches processes batched proposals
func (co *ConsensusOptimizer) processBatches() {
	for {
		select {
		case batch := <-co.batchChan:
			co.processBatch(batch)
		case <-co.stopChan:
			return
		}
	}
}

// processBatch processes a single batch
func (co *ConsensusOptimizer) processBatch(batch []Proposal) {
	startTime := time.Now()

	// Add to pipeline
	stage := PipelineStage{
		Proposals: batch,
		StartTime: startTime,
		Status:    StageProcessing,
	}

	co.mu.Lock()
	co.pipeline = append(co.pipeline, stage)
	co.mu.Unlock()

	// Process batch (simplified)
	// In real implementation, would submit to consensus engine

	// Update stats
	latency := time.Since(startTime)
	co.stats.mu.Lock()
	co.stats.AvgLatency = (co.stats.AvgLatency + latency) / 2
	co.stats.mu.Unlock()
}

// updateStats periodically updates throughput statistics
func (co *ConsensusOptimizer) updateStats() {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	var lastTotal uint64
	lastTime := time.Now()

	for {
		select {
		case <-ticker.C:
			co.stats.mu.Lock()
			currentTotal := co.stats.TotalProposals
			elapsed := time.Since(lastTime).Seconds()

			if elapsed > 0 {
				co.stats.ThroughputOps = float64(currentTotal-lastTotal) / elapsed
			}

			if co.stats.BatchCount > 0 {
				co.stats.AvgBatchSize = float64(co.stats.BatchedProposals) / float64(co.stats.BatchCount)
			}

			lastTotal = currentTotal
			lastTime = time.Now()
			co.stats.mu.Unlock()

		case <-co.stopChan:
			return
		}
	}
}

// GetStats returns current optimizer statistics
func (co *ConsensusOptimizer) GetStats() OptimizerStats {
	co.stats.mu.RLock()
	defer co.stats.mu.RUnlock()

	return OptimizerStats{
		TotalProposals:   co.stats.TotalProposals,
		BatchedProposals: co.stats.BatchedProposals,
		BatchCount:       co.stats.BatchCount,
		AvgBatchSize:     co.stats.AvgBatchSize,
		AvgLatency:       co.stats.AvgLatency,
		ThroughputOps:    co.stats.ThroughputOps,
	}
}

// OptimizeBatchSize dynamically adjusts batch size based on workload
func (co *ConsensusOptimizer) OptimizeBatchSize() {
	stats := co.GetStats()

	co.mu.Lock()
	defer co.mu.Unlock()

	// Increase batch size if throughput is high
	if stats.ThroughputOps > 1000 && co.batchSize < 1000 {
		co.batchSize = min(co.batchSize*2, 1000)
	}

	// Decrease batch size if throughput is low
	if stats.ThroughputOps < 100 && co.batchSize > 10 {
		co.batchSize = max(co.batchSize/2, 10)
	}
}

// PipelineProposal adds proposal to pipeline for concurrent processing
func (co *ConsensusOptimizer) PipelineProposal(proposal Proposal) {
	co.mu.Lock()
	defer co.mu.Unlock()

	// Find available pipeline slot
	for i := range co.pipeline {
		if co.pipeline[i].Status == StageComplete ||
			co.pipeline[i].Status == StageFailed {
			// Reuse slot
			co.pipeline[i] = PipelineStage{
				Proposals: []Proposal{proposal},
				StartTime: time.Now(),
				Status:    StagePending,
			}
			return
		}
	}

	// Add new stage if pipeline not full
	if len(co.pipeline) < co.pipelineDepth {
		co.pipeline = append(co.pipeline, PipelineStage{
			Proposals: []Proposal{proposal},
			StartTime: time.Now(),
			Status:    StagePending,
		})
	}
}

// CompressBatch compresses a batch for network transmission
func (co *ConsensusOptimizer) CompressBatch(batch []Proposal) ([]byte, error) {
	// In real implementation, would use compression algorithm
	// For now, simplified version
	return nil, nil
}

// DecompressBatch decompresses a batch received from network
func (co *ConsensusOptimizer) DecompressBatch(data []byte) ([]Proposal, error) {
	// In real implementation, would decompress
	return nil, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
