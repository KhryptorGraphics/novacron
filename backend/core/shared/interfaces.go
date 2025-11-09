// Package shared contains common interfaces and types used across core packages
// This package breaks import cycles by providing shared abstractions
package shared

import (
	"context"
	"time"
)

// FederationManager defines the interface for managing cluster federation
// This interface is used by both vm and federation packages
type FederationManager interface {
	// RegisterCluster registers a new cluster in the federation
	RegisterCluster(ctx context.Context, clusterID string, endpoint string) error

	// UnregisterCluster removes a cluster from the federation
	UnregisterCluster(ctx context.Context, clusterID string) error

	// GetCluster retrieves cluster information
	GetCluster(ctx context.Context, clusterID string) (ClusterInfo, error)

	// ListClusters returns all registered clusters
	ListClusters(ctx context.Context) ([]ClusterInfo, error)

	// SendMessage sends a message to another cluster
	SendMessage(ctx context.Context, message interface{}) error

	// GetHealth returns the health status of the federation
	GetHealth(ctx context.Context) error
}

// ClusterInfo contains information about a federated cluster
type ClusterInfo struct {
	ID           string
	Endpoint     string
	LastSeen     time.Time
	IsHealthy    bool
	Capabilities []string
	Metadata     map[string]string
}

// DistributedStateCoordinator defines the interface for coordinating distributed VM state
type DistributedStateCoordinator interface {
	// SyncState synchronizes VM state across clusters
	SyncState(ctx context.Context, vmID string) error

	// GetState retrieves the current state of a VM
	GetState(ctx context.Context, vmID string) (interface{}, error)

	// UpdateState updates the state of a VM
	UpdateState(ctx context.Context, vmID string, state interface{}) error
}

// VMState represents the basic VM state information
type VMState struct {
	ID          string
	ClusterID   string
	Status      string
	LastUpdated time.Time
	Metadata    map[string]interface{}
}

// ConsensusLog represents a consensus log entry for replication
type ConsensusLog struct {
	Term  uint64
	Index uint64
	Type  LogType
	Data  []byte
}

// LogType represents the type of consensus log entry
type LogType string

const (
	LogTypeCommand   LogType = "command"
	LogTypeNoOp      LogType = "noop"
	LogTypeBarrier   LogType = "barrier"
	LogTypeConfig    LogType = "config"
)
