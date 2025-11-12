package consensus

import (
	"time"
)

// ConsensusAlgorithm represents the type of consensus algorithm
type ConsensusAlgorithm int

const (
	AlgorithmRaft ConsensusAlgorithm = iota
	AlgorithmPaxos
	AlgorithmEPaxos
	AlgorithmEventual
	AlgorithmHybrid
)

func (a ConsensusAlgorithm) String() string {
	switch a {
	case AlgorithmRaft:
		return "Raft"
	case AlgorithmPaxos:
		return "Paxos"
	case AlgorithmEPaxos:
		return "EPaxos"
	case AlgorithmEventual:
		return "Eventual"
	case AlgorithmHybrid:
		return "Hybrid"
	default:
		return "Unknown"
	}
}

// NetworkMetrics contains network condition measurements
type NetworkMetrics struct {
	RegionCount  int
	AvgLatency   time.Duration
	MaxLatency   time.Duration
	PacketLoss   float64
	Bandwidth    int64
	ConflictRate float64
	Stability    float64
	LastUpdate   time.Time
}

// SwitchingCriteria defines thresholds for algorithm switching
type SwitchingCriteria struct {
	LowLatencyThreshold    time.Duration
	HighLatencyThreshold   time.Duration
	MaxRegionsForRaft      int
	ConflictRateThreshold  float64
	StabilityThreshold     float64
	SwitchBenefitMargin    float64
	MinTimeBetweenSwitches time.Duration
}

// DefaultSwitchingCriteria returns sensible defaults
func DefaultSwitchingCriteria() SwitchingCriteria {
	return SwitchingCriteria{
		LowLatencyThreshold:    50 * time.Millisecond,
		HighLatencyThreshold:   200 * time.Millisecond,
		MaxRegionsForRaft:      3,
		ConflictRateThreshold:  0.1,
		StabilityThreshold:     0.8,
		SwitchBenefitMargin:    1.5,
		MinTimeBetweenSwitches: 5 * time.Minute,
	}
}

// Proposal represents a consensus proposal
type Proposal struct {
	ID        string
	Key       string
	Value     []byte
	Timestamp time.Time
	Region    string
}

// Command represents a state machine command
type Command struct {
	Type      string
	Key       string
	Value     []byte
	Timestamp Timestamp
}

// Snapshot represents a consensus state snapshot
type Snapshot struct {
	Data      map[string][]byte
	Index     uint64
	Term      uint64
	Timestamp time.Time
}

// SwitchEvent records algorithm switching events
type SwitchEvent struct {
	From      ConsensusAlgorithm
	To        ConsensusAlgorithm
	Reason    string
	Timestamp time.Time
	Benefit   float64
	Cost      float64
}

// RegionMetrics contains metrics for a specific region
type RegionMetrics struct {
	RegionID   string
	Latency    time.Duration
	PacketLoss float64
	Bandwidth  int64
	NodeCount  int
	LastUpdate time.Time
}

// ConflictingWrite represents concurrent conflicting writes
type ConflictingWrite struct {
	Key       string
	Value     []byte
	Timestamp Timestamp
	NodeID    string
	Version   uint64
}

// ResolutionStrategy defines conflict resolution strategy
type ResolutionStrategy int

const (
	StrategyLWW ResolutionStrategy = iota // Last-Write-Wins
	StrategyMV                             // Multi-Value
	StrategyCustom                         // Custom resolver
)

func (s ResolutionStrategy) String() string {
	switch s {
	case StrategyLWW:
		return "Last-Write-Wins"
	case StrategyMV:
		return "Multi-Value"
	case StrategyCustom:
		return "Custom"
	default:
		return "Unknown"
	}
}

// InstanceID uniquely identifies a consensus instance
type InstanceID struct {
	ReplicaID int32
	Sequence  uint64
}

// InstanceStatus represents the status of a consensus instance
type InstanceStatus int

const (
	StatusNone InstanceStatus = iota
	StatusPreAccepted
	StatusAccepted
	StatusCommitted
	StatusExecuted
)

// Ballot represents a ballot number for Paxos/EPaxos
type Ballot struct {
	Number    uint64
	ReplicaID int32
}

func (b Ballot) GreaterThan(other Ballot) bool {
	if b.Number != other.Number {
		return b.Number > other.Number
	}
	return b.ReplicaID > other.ReplicaID
}

// QuorumType defines different quorum strategies
type QuorumType int

const (
	QuorumMajority QuorumType = iota
	QuorumFlexible
	QuorumGeographic
	QuorumFastPath
)
