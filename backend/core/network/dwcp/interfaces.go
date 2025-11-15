package dwcp

import (
	"context"
	"time"
)

// Lifecycle defines the standard lifecycle interface for all DWCP components
// All components must implement this interface for proper initialization and shutdown
type Lifecycle interface {
	// Start initializes the component with the given context
	// Returns error if initialization fails
	Start(ctx context.Context) error

	// Stop gracefully shuts down the component
	// Should clean up all resources (goroutines, connections, file handles)
	Stop() error

	// IsRunning returns true if the component is currently running
	IsRunning() bool
}

// HealthChecker defines the health monitoring interface
// Components that support health monitoring should implement this interface
type HealthChecker interface {
	// HealthCheck performs a health check and returns error if unhealthy
	HealthCheck() error

	// IsHealthy returns true if the component is healthy
	IsHealthy() bool
}

// CompressionLayer handles hierarchical delta encoding (HDE)
// This interface will be implemented by the HDE component in Phase 0-1
type CompressionLayer interface {
	Lifecycle
	HealthChecker

	// Encode compresses data using hierarchical delta encoding
	// tier specifies the compression level (local/regional/global)
	Encode(key string, data []byte, tier int) (*EncodedData, error)

	// Decode decompresses data
	Decode(key string, data *EncodedData) ([]byte, error)

	// GetMetrics returns current compression metrics
	GetMetrics() *CompressionMetrics
}

// PredictionEngine handles ML-based predictions for bandwidth, latency, etc.
// This interface will be implemented by the prediction component in Phase 2
type PredictionEngine interface {
	Lifecycle
	HealthChecker

	// PredictBandwidth predicts available bandwidth to a node
	PredictBandwidth(nodeID string) (float64, error)

	// PredictLatency predicts network latency to a node
	PredictLatency(nodeID string) (time.Duration, error)

	// GetMetrics returns current prediction metrics
	GetMetrics() *PredictionMetrics
}

// SyncLayer handles state synchronization across nodes
// This interface will be implemented by the sync component in Phase 3
type SyncLayer interface {
	Lifecycle
	HealthChecker

	// Sync synchronizes a key-value pair across nodes
	Sync(key string, value []byte) error

	// GetMetrics returns current sync metrics
	GetMetrics() *SyncMetrics
}

// ConsensusLayer handles distributed consensus
// This interface will be implemented by the consensus component in Phase 3
type ConsensusLayer interface {
	Lifecycle
	HealthChecker

	// Propose proposes a value for consensus
	Propose(value []byte) error

	// GetMetrics returns current consensus metrics
	GetMetrics() *ConsensusMetrics
}

// TaskPartitionerInterface defines the interface for intelligent task partitioning (ITP)
type TaskPartitionerInterface interface {
	Lifecycle

	// PartitionTask partitions a task across available resources
	PartitionTask(task GenericTask) ([]GenericTaskPartition, error)

	// GetPartitionStrategy returns the current partitioning strategy
	GetPartitionStrategy() PartitionStrategy
}

// CircuitBreakerInterface defines the interface for circuit breaker pattern
type CircuitBreakerInterface interface {
	Lifecycle

	// Execute executes the given function with circuit breaker protection
	Execute(fn func() error) error

	// GetState returns the current circuit breaker state
	GetState() CircuitBreakerState
}

// EncodedData represents compressed data with metadata
type EncodedData struct {
	Data           []byte
	OriginalSize   int
	CompressedSize int
	Tier           int
	Timestamp      time.Time
}

// CompressionMetrics is defined in types.go and reused here for interfaces

// PredictionMetrics tracks prediction accuracy
type PredictionMetrics struct {
	TotalPredictions    int64
	AccuratePredictions int64
	Accuracy            float64
	AvgPredictionTime   time.Duration
	Errors              int64
}

// SyncMetrics tracks synchronization performance
type SyncMetrics struct {
	TotalSyncs      int64
	SuccessfulSyncs int64
	FailedSyncs     int64
	AvgSyncTime     time.Duration
	BytesSynced     int64
}

// ConsensusMetrics tracks consensus performance
type ConsensusMetrics struct {
	TotalProposals    int64
	AcceptedProposals int64
	RejectedProposals int64
	AvgConsensusTime  time.Duration
	QuorumSize        int
}

// Supporting types for new interfaces
type GenericTask struct {
	ID       string
	Data     []byte
	Priority int
}

type GenericTaskPartition struct {
	ID       string
	Data     []byte
	NodeID   string
	Weight   float64
}

type PartitionStrategy string

const (
	PartitionStrategyRoundRobin PartitionStrategy = "round_robin"
	PartitionStrategyWeighted   PartitionStrategy = "weighted"
	PartitionStrategyAdaptive   PartitionStrategy = "adaptive"
)

type CircuitBreakerState string

const (
	CircuitBreakerClosed   CircuitBreakerState = "closed"
	CircuitBreakerOpen     CircuitBreakerState = "open"
	CircuitBreakerHalfOpen CircuitBreakerState = "half_open"
)

// Component initialization order (dependencies first)
// Phase 0: Core Infrastructure
//   1. Transport Layer (AMST) - Foundation for all communication
//   2. Compression Layer (HDE) - Data compression for transport
//
// Phase 1: Intelligence Layer
//   3. Prediction Engine (PBA) - Bandwidth prediction for optimization
//   4. Task Partitioner (ITP) - Intelligent task distribution
//
// Phase 2: Coordination Layer
//   5. Sync Layer (ASS) - State synchronization
//   6. Consensus Layer (ACP) - Distributed consensus
//
// Phase 3: Resilience Layer
//   7. Circuit Breaker - Failure protection
//   8. Resilience Manager - Overall resilience coordination
//
// Shutdown order: Reverse of initialization (Phase 3 → Phase 0)
