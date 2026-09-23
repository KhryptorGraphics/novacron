package main

// Node drain (lifecycle): POST /api/nodes/{id}/drain moves every drainable VM
// off a node so it can be taken out of service; GET /api/nodes/{id} reports the
// node's drain state and how many VMs still live on it.
//
// Lifecycle: cluster_nodes.drain_state (migration 000013) is
// active -> draining -> drained. Duplicate POSTs while draining get 409.
//
// Drain is cooperative, not destructive: VMs are queued onto the SAME
// admission-controlled transfer machinery the public /api/transfers endpoint
// uses (per-link serialization, measured budget, compression decision). A VM
// that already has an active/queued transfer is skipped harmlessly — that
// transfer IS its move off the node. A VM no reachable peer has room for is
// marked 'migrating' (a pending-move marker an operator or retry can pick up)
// instead of failing the whole drain; only when NO VM could be placed
// anywhere does the drain fail (state rolls back to 'active', 503).
//
// Honest scope: the drain->drained watcher is in-process. If this api-server
// restarts mid-drain, drain_state stays 'draining' (the row survives); the
// next POST returns 409 and an operator can reset the row by hand. Migrations
// themselves do survive: they are the transfer machinery's durable records.

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"net/http"
	"sort"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// drain states (mirrors migration 000013's CHECK constraint).
const (
	nodeDrainActive   = "active"
	nodeDrainDraining = "draining"
	nodeDrainDrained  = "drained"
)

// drainPollInterval is how often the drain coordinator re-checks whether the
// node is empty. Exported (via struct field on newDrainCoordinatorForTest) so
// tests can shrink it.
const drainPollInterval = 5 * time.Second

// drainVM is one VM row relevant to a drain.
type drainVM struct {
	ID       string
	MemoryMB int64
	State    string
}

// drainCandidate is a peer that may accept VMs, with the measured facts the
// placement rule consumes.
type drainCandidate struct {
	NodeID    string
	LinkBps   float64 // measured link throughput; <= 0 means unmeasured
	MemFreeMB int64   // guest memory the node can still reserve
}

// drainAssignment records that one VM should move to one peer.
type drainAssignment struct {
	VM         drainVM
	TargetNode string
}

// medianBWOrder returns candidates ordered median-link-bandwidth first, then
// alternating higher/lower ranks, with unmeasured links last. Rationale:
// queueing drain traffic median-first spreads a node's load across the middle
// of the link-quality band instead of piling every VM onto the single fastest
// link (failing-over outward covers both directions).
func medianBWOrder(candidates []drainCandidate) []drainCandidate {
	cs := make([]drainCandidate, len(candidates))
	copy(cs, candidates)
	sort.SliceStable(cs, func(i, j int) bool {
		bi, bj := cs[i].LinkBps, cs[j].LinkBps
		if bi <= 0 {
			return false // unmeasured links sort last
		}
		if bj <= 0 {
			return true
		}
		return bi < bj
	})
	n := len(cs)
	if n == 0 {
		return cs
	}
	out := make([]drainCandidate, 0, n)
	// Lower median first, then alternate higher/lower ranks until every
	// candidate (the unmeasured tail included) has been offered once.
	m := (n - 1) / 2
	up, down := m+1, m-1
	out = append(out, cs[m])
	for len(out) < n {
		if up < n {
			out = append(out, cs[up])
			up++
		}
		if len(out) < n && down >= 0 {
			out = append(out, cs[down])
			down--
		}
	}
	return out
}

// planDrain assigns each VM to the first median-ordered peer that can still
// hold it (reservations are summed so several VMs can share one peer). VMs
// with no fitting peer come back in unplaced.
func planDrain(vms []drainVM, candidates []drainCandidate) (assignments []drainAssignment, unplaced []drainVM) {
	ordered := medianBWOrder(candidates)
	for _, vm := range vms {
		placed := false
		for i := range ordered {
			if ordered[i].MemFreeMB >= vm.MemoryMB {
				assignments = append(assignments, drainAssignment{VM: vm, TargetNode: ordered[i].NodeID})
				ordered[i].MemFreeMB -= vm.MemoryMB
				placed = true
				break
			}
		}
		if !placed {
			unplaced = append(unplaced, vm)
		}
	}
	return assignments, unplaced
}

// drainCoordinator owns one drain's queueing and its final-state watcher.
type drainCoordinator struct {
	db          *sql.DB
	vmManager   *core_vm.VMManager
	storagePath string
	store       *transferStore
	// candidates lists placeable peers; overridable in tests (live path uses
	// nodeProfiles so only REACHABLE peers with real capacity are offered).
	candidates func(excludeNode string) []drainCandidate
	// pollInterval paces the completion watcher.
	pollInterval time.Duration
}

func newDrainCoordinator(db *sql.DB, vmManager *core_vm.VMManager, storagePath string, store *transferStore) *drainCoordinator {
	c := &drainCoordinator{
		db:           db,
		vmManager:    vmManager,
		storagePath:  storagePath,
		store:        store,
		pollInterval: drainPollInterval,
	}
	c.candidates = func(excludeNode string) []drainCandidate {
		var out []drainCandidate
		for _, n := range nodeProfiles(vmManager, storagePath, db) {
			if !n.Reachable || n.NodeID == excludeNode {
				continue
			}
			bps := 0.0
			if n.Link != nil && !n.Link.Stale && n.Link.ThroughputBps != nil {
				bps = *n.Link.ThroughputBps
			}
			out = append(out, drainCandidate{
				NodeID:    n.NodeID,
				LinkBps:   bps,
				MemFreeMB: n.memAvailMB(),
			})
		}
		return out
	}
	return c
}

// --- DB helpers ------------------------------------------------------------

// nodeKnown reports whether id names a cluster node: this node, a persisted
// peer, or a node with a lifecycle row already.
func nodeKnown(db *sql.DB, nodeID string) (bool, error) {
	if nodeID == selfNodeID() {
		return true, nil
	}
	var known bool
	err := db.QueryRow(`SELECT EXISTS (SELECT 1 FROM cluster_peers WHERE node_id = $1)
		OR EXISTS (SELECT 1 FROM cluster_nodes WHERE node_id = $1)`, nodeID).Scan(&known)
	return known, err
}

// nodeDrainState reads the current drain state; absent row = active.
func nodeDrainState(db *sql.DB, nodeID string) (string, error) {
	var state string
	err := db.QueryRow(`SELECT drain_state FROM cluster_nodes WHERE node_id = $1`, nodeID).Scan(&state)
	if err == sql.ErrNoRows {
		return nodeDrainActive, nil
	}
	return state, err
}

// setNodeDrainState upserts the lifecycle row.
func setNodeDrainState(db *sql.DB, nodeID, state string) error {
	_, err := db.Exec(`INSERT INTO cluster_nodes (node_id, drain_state) VALUES ($1, $2)
		ON CONFLICT (node_id) DO UPDATE SET drain_state = EXCLUDED.drain_state, updated_at = NOW()`,
		nodeID, state)
	return err
}

// drainableVMs lists the VMs a drain must move: running or stopped guests on
// the node.
func drainableVMs(db *sql.DB, nodeID string) ([]drainVM, error) {
	rows, err := db.Query(`SELECT id, memory_mb, state FROM vms
		WHERE node_id = $1 AND state IN ('running','stopped')`, nodeID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := make([]drainVM, 0)
	for rows.Next() {
		var vm drainVM
		if err := rows.Scan(&vm.ID, &vm.MemoryMB, &vm.State); err != nil {
			return nil, err
		}
		out = append(out, vm)
	}
	return out, rows.Err()
}

// remainingDrainVMs counts VMs the watcher still waits on. 'migrating' counts
// too: an unplaced-but-marked VM has not left the node.
func remainingDrainVMs(db *sql.DB, nodeID string) (int, error) {
	var n int
	err := db.QueryRow(`SELECT COUNT(*) FROM vms
		WHERE node_id = $1 AND state IN ('running','stopped','migrating')`, nodeID).Scan(&n)
	return n, err
}

// markVMMigrating flags an unplaced VM so operators (and a retry) can see it
// needs a home.
func markVMMigrating(db *sql.DB, vmID string) error {
	_, err := db.Exec(`UPDATE vms SET state = 'migrating', updated_at = NOW() WHERE id = $1`, vmID)
	return err
}

// --- queueing + completion -------------------------------------------------

// transferStoreFor returns the store the coordinator admits into.
func (c *drainCoordinator) transferStoreFor() *transferStore {
	if c.store != nil {
		return c.store
	}
	return transfers
}

// transferInFlightFor returns true when a queued/running transfer already
// moves this VM — re-admitting a duplicate is a no-op, not an error.
func transferInFlightFor(store *transferStore, vmID string) bool {
	for _, t := range store.list() {
		if t.VMID == vmID && (t.Status == transferQueued || t.Status == transferRunning) {
			return true
		}
	}
	return false
}

// queueDrainTransfer admits one migration transfer for a drained VM. Mirrors
// the /transfers route's guards that don't need an HTTP request (peer holds a
// real addr; VM exists on this node).
func (c *drainCoordinator) queueDrainTransfer(vm drainVM, targetNode string) (string, error) {
	store := c.transferStoreFor()
	if transferInFlightFor(store, vm.ID) {
		return "", errDrainDuplicate
	}
	if _, err := c.vmManager.GetVM(vm.ID); err != nil {
		return "", fmt.Errorf("vm %s not managed by this node: %w", vm.ID, err)
	}
	if c.vmManager.MigrationPeers()[targetNode] == "" {
		return "", fmt.Errorf("target node %q is not a registered peer", targetNode)
	}
	bytesEst := vm.MemoryMB << 20
	if bytesEst <= 0 {
		bytesEst = 1 << 20
	}
	t := &fabricTransfer{
		ID:             uuid.NewString(),
		Kind:           "migration",
		VMID:           vm.ID,
		TargetNode:     targetNode,
		BytesEstimated: bytesEst,
		CreatedAt:      time.Now().UTC(),
		Decision: transferDecisionInputs{
			Reason: fmt.Sprintf("node drain of %s — target auto-picked by median-bandwidth placement", selfNodeID()),
		},
	}
	store.admit(t)
	snapshot, _ := store.get(t.ID)
	return snapshot.ID, nil
}

var errDrainDuplicate = fmt.Errorf("transfer already in flight for this vm")

// settle performs ONE completion check: when no drainable (or
// unplaced-but-marked) VM remains on the node the drain is done and the row
// flips to drained. The watcher loops on this; tests call it directly so the
// final transition is observable synchronously.
func (c *drainCoordinator) settle(nodeID string) (bool, error) {
	remaining, err := remainingDrainVMs(c.db, nodeID)
	if err != nil {
		return false, err
	}
	if remaining > 0 {
		return false, nil
	}
	if err := setNodeDrainState(c.db, nodeID, nodeDrainDrained); err != nil {
		return false, err
	}
	return true, nil
}

// watch polls until the node drains or the server shuts down. It logs and
// stops on repeated DB errors instead of spinning.
func (c *drainCoordinator) watch(ctx context.Context, nodeID string) {
	for {
		// Wait BEFORE checking: a just-queued drain always has VMs on the
		// node; settle-first would just burn a query per poll cycle.
		select {
		case <-ctx.Done():
			return
		case <-time.After(c.pollInterval):
		}
		done, err := c.settle(nodeID)
		if err != nil {
			log.Printf("drain watcher for %s: settle failed: %v (stopping watcher; state stays draining)", nodeID, err)
			return
		}
		if done {
			log.Printf("drain of node %s finished: drain_state=%s", nodeID, nodeDrainDrained)
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-time.After(c.pollInterval):
		}
	}
}

// --- HTTP surface ----------------------------------------------------------

// registerNodeAdminRoutes wires the authed node-lifecycle endpoints:
//
//	GET  /nodes/{id}        node detail: drain_state + vm_count
//	POST /nodes/{id}/drain  begin draining the node
func registerNodeAdminRoutes(apiRouter *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, storagePath string) {
	registerNodeAdminRoutesWithCoordinator(apiRouter, newDrainCoordinator(db, vmManager, storagePath, nil))
}

// registerNodeAdminRoutesWithCoordinator wires the handlers against a specific
// coordinator — the seam tests use to inject placeable candidates and a
// stubbed transfer store.
func registerNodeAdminRoutesWithCoordinator(apiRouter *mux.Router, coord *drainCoordinator) {
	db := coord.db
	apiRouter.HandleFunc("/nodes/{id}", func(w http.ResponseWriter, r *http.Request) {
		nodeID := mux.Vars(r)["id"]
		known, err := nodeKnown(db, nodeID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query node")
			return
		}
		if !known {
			writeJSONError(w, http.StatusNotFound, "node not found")
			return
		}
		state, err := nodeDrainState(db, nodeID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query node drain state")
			return
		}
		var vmCount int
		if err := db.QueryRow(`SELECT COUNT(*) FROM vms WHERE node_id = $1`, nodeID).Scan(&vmCount); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to count node VMs")
			return
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"node_id":     nodeID,
			"drain_state": state,
			"vm_count":    vmCount,
		})
	}).Methods(http.MethodGet)

	apiRouter.HandleFunc("/nodes/{id}/drain", func(w http.ResponseWriter, r *http.Request) {
		nodeID := mux.Vars(r)["id"]

		known, err := nodeKnown(db, nodeID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query node")
			return
		}
		if !known {
			writeJSONError(w, http.StatusNotFound, "node not found")
			return
		}
		state, err := nodeDrainState(db, nodeID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query node drain state")
			return
		}
		if state == nodeDrainDraining {
			writeJSONError(w, http.StatusConflict, "node is already draining")
			return
		}
		if err := setNodeDrainState(db, nodeID, nodeDrainDraining); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to mark node draining")
			return
		}

		vms, err := drainableVMs(db, nodeID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to list node VMs")
			return
		}
		if len(vms) == 0 {
			// Nothing to move: the drain is already finished.
			if err := setNodeDrainState(db, nodeID, nodeDrainDrained); err != nil {
				writeJSONError(w, http.StatusInternalServerError, "failed to mark node drained")
				return
			}
			writeJSON(w, http.StatusAccepted, map[string]interface{}{
				"node_id":         nodeID,
				"drain_state":     nodeDrainDrained,
				"migration_ids":   []string{},
				"vms_total":       0,
				"vms_queued":      0,
				"vms_unplaced":    0,
				"vms_duplicating": 0,
			})
			return
		}

		assignments, unplaced := planDrain(vms, coord.candidates(nodeID))
		if len(assignments) == 0 {
			// No peer could take ANY VM: the drain fails; roll the state back
			// so the node is not stuck 'draining' forever and a retry is legal.
			_ = setNodeDrainState(db, nodeID, nodeDrainActive)
			writeJSONError(w, http.StatusServiceUnavailable,
				fmt.Sprintf("drain failed: no peer had capacity for any of the %d VMs on this node", len(vms)))
			return
		}

		migrationIDs := make([]string, 0, len(assignments))
		duplicates := 0
		for _, a := range assignments {
			id, qerr := coord.queueDrainTransfer(a.VM, a.TargetNode)
			if qerr == errDrainDuplicate {
				duplicates++
				continue
			}
			if qerr != nil {
				log.Printf("drain %s: failed to queue vm %s -> %s: %v", nodeID, a.VM.ID, a.TargetNode, qerr)
				unplaced = append(unplaced, a.VM)
				continue
			}
			migrationIDs = append(migrationIDs, id)
		}

		// Unplaced VMs are parked in 'migrating' (not an error for the drain):
		// the watcher keeps them visible until they leave.
		for _, vm := range unplaced {
			if err := markVMMigrating(db, vm.ID); err != nil {
				// Honest failure: the marker could not be persisted, so the
				// watcher (which counts 'migrating' rows too) would finish the
				// drain while this VM still sits on the node. Fail instead.
				_ = setNodeDrainState(db, nodeID, nodeDrainActive)
				writeJSONError(w, http.StatusInternalServerError,
					fmt.Sprintf("failed to mark unplaced vm %s migrating; drain aborted", vm.ID))
				return
			}
		}

		// Background ctx, not r.Context(): the request-scoped context is
		// canceled when this handler returns, which would kill the watcher.
		go coord.watch(context.Background(), nodeID)

		writeJSON(w, http.StatusAccepted, map[string]interface{}{
			"node_id":         nodeID,
			"drain_state":     nodeDrainDraining,
			"migration_ids":   migrationIDs,
			"vms_total":       len(vms),
			"vms_queued":      len(migrationIDs),
			"vms_unplaced":    len(unplaced),
			"vms_duplicating": duplicates,
		})
	}).Methods(http.MethodPost)
}
