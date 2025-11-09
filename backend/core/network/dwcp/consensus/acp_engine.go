package consensus

import (
	"fmt"
	"sync"
	"time"
)

// ACPEngine is the Adaptive Consensus Protocol engine that dynamically
// switches between consensus algorithms based on network conditions
type ACPEngine struct {
	mu sync.RWMutex

	// Current active algorithm
	currentAlgorithm ConsensusAlgorithm

	// Algorithm implementations
	raft     *RaftConsensus
	paxos    *PaxosConsensus
	epaxos   *EPaxosConsensus
	eventual *EventualConsistency
	hybrid   *HybridConsensus

	// Monitoring and decision making
	networkMonitor  *NetworkMonitor
	switchThreshold SwitchingCriteria
	optimizer       *ConsensusOptimizer

	// State management
	stateMachine    StateMachine
	lastSwitch      time.Time
	switchHistory   []SwitchEvent
	inflightCount   int
	inflightCond    *sync.Cond

	// Configuration
	nodeID     string
	localRegion string
	enabled     bool
}

// StateMachine interface for applying committed commands
type StateMachine interface {
	Apply(cmd Command) ([]byte, error)
	Snapshot() (*Snapshot, error)
	Restore(snapshot *Snapshot) error
}

// NewACPEngine creates a new Adaptive Consensus Protocol engine
func NewACPEngine(nodeID, localRegion string, sm StateMachine) *ACPEngine {
	acp := &ACPEngine{
		nodeID:          nodeID,
		localRegion:     localRegion,
		currentAlgorithm: AlgorithmRaft, // Default to Raft
		switchThreshold: DefaultSwitchingCriteria(),
		stateMachine:    sm,
		switchHistory:   make([]SwitchEvent, 0),
		enabled:         true,
	}

	acp.inflightCond = sync.NewCond(&acp.mu)

	// Initialize network monitor
	acp.networkMonitor = NewNetworkMonitor()

	// Initialize algorithm implementations
	acp.raft = NewRaftConsensus(nodeID, sm)
	acp.paxos = NewPaxosConsensus(nodeID, sm)
	acp.epaxos = NewEPaxosConsensus(nodeID, sm)
	acp.eventual = NewEventualConsistency(nodeID, sm)
	acp.hybrid = NewHybridConsensus(nodeID, localRegion, acp.raft, acp.eventual)

	// Initialize optimizer
	acp.optimizer = NewConsensusOptimizer(100, 10*time.Millisecond)

	return acp
}

// DecideAlgorithm determines the optimal consensus algorithm based on current metrics
func (acp *ACPEngine) DecideAlgorithm() ConsensusAlgorithm {
	metrics := acp.networkMonitor.GetMetrics()

	// Decision tree based on network conditions

	// Low latency, few regions -> Raft (simple and fast)
	if metrics.RegionCount <= acp.switchThreshold.MaxRegionsForRaft &&
		metrics.AvgLatency < acp.switchThreshold.LowLatencyThreshold {
		return AlgorithmRaft
	}

	// Very high latency -> Eventual consistency with CRDTs
	if metrics.AvgLatency > acp.switchThreshold.HighLatencyThreshold {
		return AlgorithmEventual
	}

	// High conflict rate -> EPaxos (handles conflicts well)
	if metrics.ConflictRate > acp.switchThreshold.ConflictRateThreshold {
		return AlgorithmEPaxos
	}

	// Cross-region with moderate latency -> Hybrid
	if metrics.RegionCount > acp.switchThreshold.MaxRegionsForRaft &&
		metrics.AvgLatency < acp.switchThreshold.HighLatencyThreshold {
		return AlgorithmHybrid
	}

	// Default for moderate conditions -> Paxos
	return AlgorithmPaxos
}

// Propose submits a proposal to the consensus system
func (acp *ACPEngine) Propose(key string, value []byte) error {
	acp.mu.RLock()
	algorithm := acp.currentAlgorithm
	acp.inflightCount++
	acp.mu.RUnlock()

	defer func() {
		acp.mu.Lock()
		acp.inflightCount--
		acp.inflightCond.Broadcast()
		acp.mu.Unlock()
	}()

	// Route to appropriate algorithm
	switch algorithm {
	case AlgorithmRaft:
		return acp.raft.Propose(key, value)
	case AlgorithmPaxos:
		return acp.paxos.Propose(key, value)
	case AlgorithmEPaxos:
		return acp.epaxos.Propose(key, value)
	case AlgorithmEventual:
		return acp.eventual.Update(key, value)
	case AlgorithmHybrid:
		return acp.hybrid.Propose(key, value)
	default:
		return fmt.Errorf("unknown consensus algorithm: %v", algorithm)
	}
}

// ShouldSwitch determines if algorithm switching is beneficial
func (acp *ACPEngine) ShouldSwitch() (bool, ConsensusAlgorithm) {
	acp.mu.RLock()
	current := acp.currentAlgorithm
	lastSwitch := acp.lastSwitch
	acp.mu.RUnlock()

	// Rate limit switching
	if time.Since(lastSwitch) < acp.switchThreshold.MinTimeBetweenSwitches {
		return false, current
	}

	optimal := acp.DecideAlgorithm()

	if current == optimal {
		return false, current
	}

	// Calculate benefit and cost with hysteresis
	benefit := acp.estimateBenefit(current, optimal)
	cost := acp.estimateSwitchCost(current, optimal)

	// Only switch if benefit exceeds cost by margin
	if benefit > cost*acp.switchThreshold.SwitchBenefitMargin {
		return true, optimal
	}

	return false, current
}

// SwitchAlgorithm performs a safe algorithm switch
func (acp *ACPEngine) SwitchAlgorithm(to ConsensusAlgorithm) error {
	acp.mu.Lock()
	from := acp.currentAlgorithm
	acp.mu.Unlock()

	if from == to {
		return nil
	}

	// Step 1: Drain in-flight proposals
	if err := acp.drainInflight(); err != nil {
		return fmt.Errorf("failed to drain inflight proposals: %w", err)
	}

	// Step 2: Take snapshot of current state
	snapshot, err := acp.takeSnapshot()
	if err != nil {
		return fmt.Errorf("failed to take snapshot: %w", err)
	}

	// Step 3: Initialize new algorithm with snapshot
	if err := acp.loadSnapshotToAlgorithm(to, snapshot); err != nil {
		return fmt.Errorf("failed to load snapshot to new algorithm: %w", err)
	}

	// Step 4: Atomic switch
	acp.mu.Lock()
	acp.currentAlgorithm = to
	acp.lastSwitch = time.Now()
	acp.switchHistory = append(acp.switchHistory, SwitchEvent{
		From:      from,
		To:        to,
		Timestamp: time.Now(),
		Benefit:   acp.estimateBenefit(from, to),
		Cost:      acp.estimateSwitchCost(from, to),
	})
	acp.mu.Unlock()

	return nil
}

// drainInflight waits for all in-flight proposals to complete
func (acp *ACPEngine) drainInflight() error {
	acp.mu.Lock()
	defer acp.mu.Unlock()

	timeout := time.NewTimer(30 * time.Second)
	defer timeout.Stop()

	done := make(chan struct{})
	go func() {
		for acp.inflightCount > 0 {
			acp.inflightCond.Wait()
		}
		close(done)
	}()

	select {
	case <-done:
		return nil
	case <-timeout.C:
		return fmt.Errorf("timeout waiting for inflight proposals to drain")
	}
}

// takeSnapshot creates a snapshot from the current algorithm
func (acp *ACPEngine) takeSnapshot() (*Snapshot, error) {
	return acp.stateMachine.Snapshot()
}

// loadSnapshotToAlgorithm loads a snapshot into the target algorithm
func (acp *ACPEngine) loadSnapshotToAlgorithm(algo ConsensusAlgorithm, snapshot *Snapshot) error {
	switch algo {
	case AlgorithmRaft:
		return acp.raft.LoadSnapshot(snapshot)
	case AlgorithmPaxos:
		return acp.paxos.LoadSnapshot(snapshot)
	case AlgorithmEPaxos:
		return acp.epaxos.LoadSnapshot(snapshot)
	case AlgorithmEventual:
		return acp.eventual.LoadSnapshot(snapshot)
	case AlgorithmHybrid:
		return acp.hybrid.LoadSnapshot(snapshot)
	default:
		return fmt.Errorf("unknown algorithm: %v", algo)
	}
}

// estimateBenefit estimates the performance benefit of switching algorithms
func (acp *ACPEngine) estimateBenefit(from, to ConsensusAlgorithm) float64 {
	metrics := acp.networkMonitor.GetMetrics()

	// Benefit scoring based on network conditions
	var benefit float64

	switch to {
	case AlgorithmRaft:
		// Raft benefits from low latency and few regions
		if metrics.AvgLatency < acp.switchThreshold.LowLatencyThreshold {
			benefit += 0.5
		}
		if metrics.RegionCount <= acp.switchThreshold.MaxRegionsForRaft {
			benefit += 0.3
		}

	case AlgorithmEPaxos:
		// EPaxos benefits from high conflict rates
		if metrics.ConflictRate > acp.switchThreshold.ConflictRateThreshold {
			benefit += 0.6
		}

	case AlgorithmEventual:
		// Eventual consistency benefits from high latency
		if metrics.AvgLatency > acp.switchThreshold.HighLatencyThreshold {
			benefit += 0.7
		}

	case AlgorithmHybrid:
		// Hybrid benefits from multi-region with moderate latency
		if metrics.RegionCount > acp.switchThreshold.MaxRegionsForRaft {
			benefit += 0.4
		}
	}

	return benefit
}

// estimateSwitchCost estimates the cost of switching algorithms
func (acp *ACPEngine) estimateSwitchCost(from, to ConsensusAlgorithm) float64 {
	// Base cost for any switch
	baseCost := 0.1

	// Additional cost based on complexity of transition
	var transitionCost float64

	// Switching to/from eventual consistency is more complex
	if from == AlgorithmEventual || to == AlgorithmEventual {
		transitionCost += 0.3
	}

	// EPaxos has complex state
	if from == AlgorithmEPaxos || to == AlgorithmEPaxos {
		transitionCost += 0.2
	}

	return baseCost + transitionCost
}

// MonitorAndAdapt continuously monitors network conditions and adapts
func (acp *ACPEngine) MonitorAndAdapt() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		if !acp.enabled {
			continue
		}

		shouldSwitch, newAlgo := acp.ShouldSwitch()
		if shouldSwitch {
			if err := acp.SwitchAlgorithm(newAlgo); err != nil {
				// Log error but continue monitoring
				fmt.Printf("Failed to switch algorithm: %v\n", err)
			}
		}
	}
}

// GetCurrentAlgorithm returns the currently active algorithm
func (acp *ACPEngine) GetCurrentAlgorithm() ConsensusAlgorithm {
	acp.mu.RLock()
	defer acp.mu.RUnlock()
	return acp.currentAlgorithm
}

// GetSwitchHistory returns the algorithm switching history
func (acp *ACPEngine) GetSwitchHistory() []SwitchEvent {
	acp.mu.RLock()
	defer acp.mu.RUnlock()
	return append([]SwitchEvent{}, acp.switchHistory...)
}

// UpdateNetworkMetrics updates the network monitoring metrics
func (acp *ACPEngine) UpdateNetworkMetrics(metrics NetworkMetrics) {
	acp.networkMonitor.UpdateMetrics(metrics)
}

// Enable enables adaptive switching
func (acp *ACPEngine) Enable() {
	acp.mu.Lock()
	defer acp.mu.Unlock()
	acp.enabled = true
}

// Disable disables adaptive switching
func (acp *ACPEngine) Disable() {
	acp.mu.Lock()
	defer acp.mu.Unlock()
	acp.enabled = false
}
