package storage

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"
)

// TestDistributedStorageService_ReplicationAcrossDistinctBackends is a
// discrimination test for the WriteShard/ReadShard replica-placement bug
// (novacron-bvw): previously every "replica" of a shard was written to the
// SAME local path
//
//	filepath.Join(RootDir, volumeID, "shard_N")
//
// regardless of which node ID it was supposedly stored on. That meant (a)
// replicas were never actually distinct copies and (b) losing that single
// path lost every replica simultaneously — there was no real redundancy.
//
// This test proves:
//  1. A shard written with replication factor R lands on R physically
//     DISTINCT backends (distinct on-disk paths, one per node, each
//     non-empty).
//  2. After completely losing one backend (its on-disk data removed), the
//     shard is still readable via failover to a surviving replica.
//
// Under the old same-path implementation this test fails at step 1 (all
// replica paths collapse to one, "distinct backend paths" assertion fires)
// and would also fail at step 2 (deleting the one shared path destroys the
// only copy, so the post-loss read errors out).
func TestDistributedStorageService_ReplicationAcrossDistinctBackends(t *testing.T) {
	baseConfig := StorageManagerConfig{
		BasePath:    t.TempDir(),
		Compression: false,
		Encryption:  false,
		Dedup:       false,
	}
	baseManager, err := NewStorageManager(baseConfig)
	if err != nil {
		t.Fatalf("failed to create base manager: %v", err)
	}

	distConfig := DefaultDistributedStorageConfig()
	distConfig.RootDir = t.TempDir()
	// Require every replica to be written before WriteShard returns, so the
	// full replication factor is honored synchronously (the default async
	// mode only guarantees the primary write, by design).
	distConfig.SynchronousReplication = true

	distService, err := NewDistributedStorageService(baseManager, distConfig)
	if err != nil {
		t.Fatalf("failed to create distributed service: %v", err)
	}
	if err := distService.Start(); err != nil {
		t.Fatalf("failed to start distributed service: %v", err)
	}
	defer distService.Stop()

	const replicationFactor = 3
	const nodeCount = replicationFactor + 1 // one spare node beyond RF
	for i := range nodeCount {
		id := fmt.Sprintf("node-%d", i)
		distService.AddNode(NodeInfo{
			ID:        id,
			Name:      id,
			Address:   fmt.Sprintf("10.0.0.%d", i+1),
			Port:      9000,
			Available: true,
			JoinedAt:  time.Now(),
			LastSeen:  time.Now(),
		})
	}

	opts := VolumeCreateOptions{
		Name:   "repl-discrimination-volume",
		Type:   VolumeTypeDistributed,
		Format: VolumeFormatRAW,
		Size:   1, // small: a single shard is enough to exercise replication
	}

	ctx := context.Background()
	volume, err := distService.CreateDistributedVolume(ctx, opts, replicationFactor)
	if err != nil {
		t.Fatalf("failed to create distributed volume: %v", err)
	}

	const shardIndex = 0
	payload := []byte("distinct-backend-replication-payload")

	if err := distService.WriteShard(ctx, volume.ID, shardIndex, payload); err != nil {
		t.Fatalf("WriteShard failed: %v", err)
	}

	distVolume, err := distService.GetDistributedVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("failed to get distributed volume: %v", err)
	}
	distVolume.mu.RLock()
	replicaNodes := append([]string(nil), distVolume.DistInfo.Shards[shardIndex].NodeIDs...)
	distVolume.mu.RUnlock()

	if len(replicaNodes) != replicationFactor {
		t.Fatalf("expected %d replica nodes, got %d (%v)", replicationFactor, len(replicaNodes), replicaNodes)
	}

	loopback, ok := distService.replicator.(*LoopbackReplicator)
	if !ok {
		t.Fatalf("expected default replicator to be *LoopbackReplicator, got %T", distService.replicator)
	}

	// --- Discriminator #1: R distinct backend paths, each actually holding data. ---
	seenPaths := make(map[string]string) // path -> owning nodeID
	for _, nodeID := range replicaNodes {
		path := loopback.NodePath(nodeID, volume.ID, shardIndex)
		info, statErr := os.Stat(path)
		if statErr != nil {
			t.Fatalf("replica for node %s missing on disk at %s: %v", nodeID, path, statErr)
		}
		if info.Size() == 0 {
			t.Fatalf("replica for node %s at %s is empty", nodeID, path)
		}
		if owner, exists := seenPaths[path]; exists {
			t.Fatalf("replica for node %s shares the SAME backend path as node %s (%s) — "+
				"this is the old same-local-path bug: replicas are not physically distinct", nodeID, owner, path)
		}
		seenPaths[path] = nodeID
	}
	if len(seenPaths) != replicationFactor {
		t.Fatalf("expected %d distinct backend paths, got %d", replicationFactor, len(seenPaths))
	}

	// Sanity: the pipeline (checksum/compression/etc) round-trips correctly.
	readBack, err := distService.ReadShard(ctx, volume.ID, shardIndex)
	if err != nil {
		t.Fatalf("ReadShard failed: %v", err)
	}
	if string(readBack) != string(payload) {
		t.Fatalf("read data mismatch: expected %q, got %q", payload, readBack)
	}

	// --- Discriminator #2: lose one backend entirely; shard must still be
	// readable via failover to a surviving distinct replica. ---
	lostNode := replicaNodes[0]
	if err := loopback.DeleteReplica(ctx, lostNode, volume.ID, shardIndex); err != nil {
		t.Fatalf("failed to delete replica for %s: %v", lostNode, err)
	}
	if _, statErr := os.Stat(loopback.NodePath(lostNode, volume.ID, shardIndex)); !os.IsNotExist(statErr) {
		t.Fatalf("expected backend for %s to be gone after DeleteReplica, stat err=%v", lostNode, statErr)
	}

	afterLoss, err := distService.ReadShard(ctx, volume.ID, shardIndex)
	if err != nil {
		t.Fatalf("ReadShard failed after losing one backend (failover should have used a surviving replica): %v", err)
	}
	if string(afterLoss) != string(payload) {
		t.Fatalf("read data mismatch after backend loss: expected %q, got %q", payload, afterLoss)
	}

	// The surviving replicas must remain intact and distinct from each other.
	survivorPaths := make(map[string]bool)
	for _, nodeID := range replicaNodes[1:] {
		path := loopback.NodePath(nodeID, volume.ID, shardIndex)
		if _, statErr := os.Stat(path); statErr != nil {
			t.Fatalf("surviving replica for node %s missing at %s: %v", nodeID, path, statErr)
		}
		survivorPaths[path] = true
	}
	if len(survivorPaths) != replicationFactor-1 {
		t.Fatalf("expected %d distinct surviving backend paths, got %d", replicationFactor-1, len(survivorPaths))
	}
}

// TestLoopbackReplicator_NodeIsolationAndFailover unit-tests the Replicator
// implementation directly: writes to distinct node backends must not leak
// into each other, and a node marked down must fail reads/writes without
// touching its stored data (distinguishing an outage from data loss).
func TestLoopbackReplicator_NodeIsolationAndFailover(t *testing.T) {
	ctx := context.Background()
	r := NewLoopbackReplicator(t.TempDir())

	volumeID := "vol-1"
	shardIndex := 0

	if err := r.WriteReplica(ctx, "node-a", volumeID, shardIndex, []byte("from-a")); err != nil {
		t.Fatalf("write to node-a failed: %v", err)
	}
	if err := r.WriteReplica(ctx, "node-b", volumeID, shardIndex, []byte("from-b")); err != nil {
		t.Fatalf("write to node-b failed: %v", err)
	}

	if r.NodePath("node-a", volumeID, shardIndex) == r.NodePath("node-b", volumeID, shardIndex) {
		t.Fatalf("node-a and node-b resolved to the same backend path")
	}

	dataA, err := r.ReadReplica(ctx, "node-a", volumeID, shardIndex)
	if err != nil {
		t.Fatalf("read from node-a failed: %v", err)
	}
	if string(dataA) != "from-a" {
		t.Fatalf("node-a returned %q, want %q (possible cross-node leakage)", dataA, "from-a")
	}

	dataB, err := r.ReadReplica(ctx, "node-b", volumeID, shardIndex)
	if err != nil {
		t.Fatalf("read from node-b failed: %v", err)
	}
	if string(dataB) != "from-b" {
		t.Fatalf("node-b returned %q, want %q (possible cross-node leakage)", dataB, "from-b")
	}

	// Marking a node down must fail reads/writes without deleting its data.
	r.SetNodeDown("node-a", true)
	if _, err := r.ReadReplica(ctx, "node-a", volumeID, shardIndex); err == nil {
		t.Fatalf("expected read from down node-a to fail")
	}
	if err := r.WriteReplica(ctx, "node-a", volumeID, shardIndex, []byte("should-not-write")); err == nil {
		t.Fatalf("expected write to down node-a to fail")
	}
	if !r.NodeAvailable("node-b") {
		t.Fatalf("node-b should remain available when only node-a is marked down")
	}

	r.SetNodeDown("node-a", false)
	dataA, err = r.ReadReplica(ctx, "node-a", volumeID, shardIndex)
	if err != nil {
		t.Fatalf("read from node-a failed after bringing it back up: %v", err)
	}
	if string(dataA) != "from-a" {
		t.Fatalf("node-a data corrupted by the down/up cycle: got %q, want %q", dataA, "from-a")
	}
}
