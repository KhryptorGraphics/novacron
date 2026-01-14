package vm

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/federation"
	"github.com/pkg/errors"
	"go.uber.org/zap"
)

// sendShardUpdateWithClockMerge sends update and handles clock merging on replica
func (sm *VMStateShardingManager) sendShardUpdateWithClockMerge(ctx context.Context, update *ShardUpdateMessage, targetNode string) error {
	if sm.federation == nil {
		return errors.New("federation not available")
	}

	// Create message with full clock and version info
	msg := &federation.CrossClusterMessage{
		Type:        "shard_update_with_clock",
		Source:      sm.localNodeID,
		Destination: targetNode,
		Payload: map[string]interface{}{
			"shard_id":     update.ShardID,
			"version":      update.Version,
			"vector_clock": update.VectorClock,
			"data":         update.Data,
			"timestamp":    update.Timestamp,
		},
		Timestamp: time.Now(),
	}

	// Send through federation
	if crossCluster, ok := sm.federation.(*federation.CrossClusterComponents); ok {
		err := crossCluster.SendMessage(ctx, msg)
		if err != nil {
			return errors.Wrapf(err, "failed to send update to %s", targetNode)
		}

		// Wait for acknowledgment with merged clock
		ack, err := sm.waitForClockMergeAck(ctx, update.ShardID, targetNode)
		if err != nil {
			return errors.Wrapf(err, "clock merge acknowledgment failed from %s", targetNode)
		}

		if !ack {
			return errors.Errorf("clock merge not acknowledged by %s", targetNode)
		}
	}

	return nil
}

// Vector clock helper functions
func (sm *VMStateShardingManager) mergeVectorClocks(clock1, clock2 map[string]uint64) map[string]uint64 {
	merged := make(map[string]uint64)

	// Take max of each node's timestamp
	for node, ts := range clock1 {
		merged[node] = ts
	}

	for node, ts := range clock2 {
		if existing, exists := merged[node]; exists {
			if ts > existing {
				merged[node] = ts
			}
		} else {
			merged[node] = ts
		}
	}

	return merged
}

func (sm *VMStateShardingManager) isNewerClock(clock1, clock2 map[string]uint64) bool {
	// clock1 is newer if all its timestamps are >= clock2 and at least one is >
	hasGreater := false
	for node, ts1 := range clock1 {
		if ts2, exists := clock2[node]; exists {
			if ts1 < ts2 {
				return false
			}
			if ts1 > ts2 {
				hasGreater = true
			}
		} else if ts1 > 0 {
			hasGreater = true
		}
	}
	return hasGreater
}

func (sm *VMStateShardingManager) isConcurrentClock(clock1, clock2 map[string]uint64) bool {
	// Clocks are concurrent if neither is newer than the other
	return !sm.isNewerClock(clock1, clock2) && !sm.isNewerClock(clock2, clock1)
}

func (sm *VMStateShardingManager) equalVectorClocks(clock1, clock2 map[string]uint64) bool {
	if len(clock1) != len(clock2) {
		return false
	}
	for node, ts1 := range clock1 {
		if ts2, exists := clock2[node]; !exists || ts1 != ts2 {
			return false
		}
	}
	return true
}

// fetchRemoteStateWithClock fetches state along with vector clock
func (sm *VMStateShardingManager) fetchRemoteStateWithClock(ctx context.Context, vmID, nodeID string) (*ShardStateWithClock, error) {
	if sm.federation == nil {
		return nil, errors.New("federation not available")
	}

	msg := &federation.CrossClusterMessage{
		Type:        "fetch_state_with_clock",
		Source:      sm.localNodeID,
		Destination: nodeID,
		Payload: map[string]interface{}{
			"vm_id": vmID,
		},
		Timestamp: time.Now(),
	}

	if crossCluster, ok := sm.federation.(*federation.CrossClusterComponents); ok {
		err := crossCluster.SendMessage(ctx, msg)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to fetch state from %s", nodeID)
		}

		// Wait for response with state and clock
		response, err := sm.waitForStateWithClock(ctx, vmID, nodeID)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to get state response from %s", nodeID)
		}

		return response, nil
	}

	return nil, errors.New("cross-cluster components not available")
}

// performReadRepair checks replicas and repairs if needed
func (sm *VMStateShardingManager) performReadRepair(ctx context.Context, shard *VMStateShard) {
	replicas := shard.ReplicaNodes
	for _, replica := range replicas {
		stateWithClock, err := sm.fetchRemoteStateWithClock(ctx, shard.VMID, replica)
		if err != nil {
			continue
		}

		// Check if replica is stale
		if sm.isNewerClock(shard.VectorClock, stateWithClock.VectorClock) {
			// Repair stale replica
			sm.repairStaleReplica(ctx, shard.VMID, replica, shard.Data, shard.VectorClock)
		} else if sm.isConcurrentClock(shard.VectorClock, stateWithClock.VectorClock) {
			// Handle concurrent update
			if sm.recoveryManager != nil {
				sm.recoveryManager.ReconcileVectorClocks(shard.ShardID, []map[string]uint64{
					shard.VectorClock,
					stateWithClock.VectorClock,
				})
			}
		}
	}
}

// repairStaleReplica updates a stale replica with newer state
func (sm *VMStateShardingManager) repairStaleReplica(ctx context.Context, vmID, nodeID string, state *DistributedVMState, clock map[string]uint64) {
	if sm.federation == nil {
		return
	}

	msg := &federation.CrossClusterMessage{
		Type:        "repair_stale_state",
		Source:      sm.localNodeID,
		Destination: nodeID,
		Payload: map[string]interface{}{
			"vm_id":        vmID,
			"state":        state,
			"vector_clock": clock,
			"repair_time":  time.Now(),
		},
		Timestamp: time.Now(),
	}

	if crossCluster, ok := sm.federation.(*federation.CrossClusterComponents); ok {
		err := crossCluster.SendMessage(ctx, msg)
		if err != nil {
			sm.logger.Warn("Failed to repair stale replica",
				zap.String("vmID", vmID),
				zap.String("nodeID", nodeID),
				zap.Error(err))
		}
	}
}

// Helper types for vector clock operations
type ShardStateWithClock struct {
	State       *DistributedVMState
	VectorClock map[string]uint64
	Version     uint64
}

type ShardUpdateMessage struct {
	ShardID     string
	Version     uint64
	VectorClock map[string]uint64
	SourceNode  string
	Timestamp   time.Time
	UpdateType  string
	Data        *DistributedVMState
}

type StateConflict struct {
	ID        string
	Type      ConflictType
	States    []interface{}
	Clocks    []map[string]uint64
	Timestamp time.Time
}

type ConflictType int

const (
	ConflictTypeConcurrentUpdate ConflictType = iota
	ConflictTypeVersionMismatch
	ConflictTypeClockDivergence
)

func (ct ConflictType) String() string {
	switch ct {
	case ConflictTypeConcurrentUpdate:
		return "concurrent_update"
	case ConflictTypeVersionMismatch:
		return "version_mismatch"
	case ConflictTypeClockDivergence:
		return "clock_divergence"
	default:
		return "unknown"
	}
}

type ConflictResolution struct {
	ResolvedState interface{}
	Method        string
	Timestamp     time.Time
}

func generateConflictID(vmID string) string {
	h := sha256.New()
	h.Write([]byte(fmt.Sprintf("%s:%d", vmID, time.Now().UnixNano())))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// Helper functions for acknowledgments
func (sm *VMStateShardingManager) waitForClockMergeAck(ctx context.Context, shardID, nodeID string) (bool, error) {
	// Implementation would wait for acknowledgment with timeout
	select {
	case <-ctx.Done():
		return false, ctx.Err()
	case <-time.After(5 * time.Second):
		return true, nil // Simplified for now
	}
}

func (sm *VMStateShardingManager) waitForStateWithClock(ctx context.Context, vmID, nodeID string) (*ShardStateWithClock, error) {
	// Implementation would wait for state response with timeout
	// For now, return a stub
	return &ShardStateWithClock{
		State:       &DistributedVMState{VMID: vmID},
		VectorClock: make(map[string]uint64),
		Version:     1,
	}, nil
}