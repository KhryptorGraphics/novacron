package storage

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// Replicator abstracts the transport used to place shard replicas on
// distinct storage backends (nodes). Implementations MUST persist each
// replica to storage exclusively owned by the given nodeID, so that replicas
// placed on different nodes are genuinely independent: losing the backend
// for one node MUST NOT affect the data held for any other node.
//
// This interface is the seam between shard placement (which decides WHICH
// distinct node IDs a shard's replicas live on) and shard transport (which
// actually moves the bytes to/from those nodes). A production implementation
// would dial a remote storage-agent (e.g. over gRPC/HTTP) per nodeID; see
// LoopbackReplicator below for the in-process stepping-stone implementation
// used until that lands (tracked separately — see novacron-bvw follow-up).
type Replicator interface {
	// WriteReplica durably stores shard data on the backend owned by nodeID.
	WriteReplica(ctx context.Context, nodeID, volumeID string, shardIndex int, data []byte) error

	// ReadReplica retrieves shard data from the backend owned by nodeID. It
	// MUST return an error (never fabricated data) if nodeID's backend is
	// unreachable or has no data for the shard.
	ReadReplica(ctx context.Context, nodeID, volumeID string, shardIndex int) ([]byte, error)

	// DeleteReplica removes the shard replica from the backend owned by
	// nodeID. Deleting a replica that does not exist is not an error.
	DeleteReplica(ctx context.Context, nodeID, volumeID string, shardIndex int) error

	// NodeAvailable reports whether the backend for nodeID is currently
	// reachable. Callers MAY use this to skip a known-down node before
	// attempting a read/write, but WriteReplica/ReadReplica MUST also fail
	// on their own if the backend turns out to be unreachable.
	NodeAvailable(nodeID string) bool
}

// LoopbackReplicator is a real, in-process multi-backend transport: each
// nodeID owns its own isolated storage root on disk
// (rootDir/nodes/<nodeID>/...), so replicas placed on different nodes
// physically land in different backends. This replaces the prior "simulated"
// behavior that wrote every replica of a shard to one shared path regardless
// of which node ID it was supposedly stored on.
//
// It is "loopback" because every backend currently lives in the same
// process/host rather than behind a network call, but each backend is a
// fully distinct, independently-failable storage root with its own
// availability switch (SetNodeDown) — sufficient to prove real per-replica
// placement across distinct backends and read failover when one is lost.
// A production network transport (dialing a remote storage-agent per node)
// is the natural next implementation of the same Replicator interface.
type LoopbackReplicator struct {
	rootDir string

	mu   sync.RWMutex
	down map[string]bool // nodeID -> forced-unavailable (simulated outage)
}

// NewLoopbackReplicator creates a loopback replicator rooted at rootDir.
func NewLoopbackReplicator(rootDir string) *LoopbackReplicator {
	return &LoopbackReplicator{
		rootDir: rootDir,
		down:    make(map[string]bool),
	}
}

// NodePath returns the on-disk path backing nodeID's copy of the given
// shard. Exported so tests/operators can verify replicas physically land on
// distinct backends and inspect/remove a specific backend directly.
func (r *LoopbackReplicator) NodePath(nodeID, volumeID string, shardIndex int) string {
	return filepath.Join(r.rootDir, "nodes", nodeID, volumeID, fmt.Sprintf("shard_%d", shardIndex))
}

// SetNodeDown marks a backend as unavailable (down=true) or available
// (down=false) without touching its stored data, simulating a node/network
// outage independent of data loss.
func (r *LoopbackReplicator) SetNodeDown(nodeID string, down bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.down[nodeID] = down
}

// NodeAvailable implements Replicator.
func (r *LoopbackReplicator) NodeAvailable(nodeID string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return !r.down[nodeID]
}

// WriteReplica implements Replicator.
func (r *LoopbackReplicator) WriteReplica(ctx context.Context, nodeID, volumeID string, shardIndex int, data []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !r.NodeAvailable(nodeID) {
		return fmt.Errorf("replicator: node %s is unavailable", nodeID)
	}

	path := r.NodePath(nodeID, volumeID, shardIndex)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("replicator: node %s: %w", nodeID, err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("replicator: node %s: %w", nodeID, err)
	}
	return nil
}

// ReadReplica implements Replicator.
func (r *LoopbackReplicator) ReadReplica(ctx context.Context, nodeID, volumeID string, shardIndex int) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !r.NodeAvailable(nodeID) {
		return nil, fmt.Errorf("replicator: node %s is unavailable", nodeID)
	}

	path := r.NodePath(nodeID, volumeID, shardIndex)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("replicator: node %s: %w", nodeID, err)
	}
	return data, nil
}

// DeleteReplica implements Replicator. Used to simulate total loss of a
// backend (as opposed to SetNodeDown, which simulates a reachability
// outage) and by repair/rebalance paths that relocate replicas.
func (r *LoopbackReplicator) DeleteReplica(ctx context.Context, nodeID, volumeID string, shardIndex int) error {
	path := r.NodePath(nodeID, volumeID, shardIndex)
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("replicator: node %s: %w", nodeID, err)
	}
	return nil
}
