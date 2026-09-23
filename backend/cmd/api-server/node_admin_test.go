package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)


// TestMedianBWOrderPicksMedianFirst pins the drain queue order: the
// median-bandwidth peer goes first, then alternating higher/lower ranks,
// unmeasured links last.
func TestMedianBWOrderPicksMedianFirst(t *testing.T) {
	cands := []drainCandidate{
		{NodeID: "slow", LinkBps: 100e6, MemFreeMB: 1 << 20},
		{NodeID: "mid", LinkBps: 500e6, MemFreeMB: 1 << 20},
		{NodeID: "fast", LinkBps: 900e6, MemFreeMB: 1 << 20},
		{NodeID: "unmeasured", LinkBps: 0, MemFreeMB: 1 << 20},
	}
	got := medianBWOrder(cands)
	var ids []string
	for _, c := range got {
		ids = append(ids, c.NodeID)
	}
	if strings.Join(ids, ",") != "mid,fast,slow,unmeasured" {
		t.Fatalf("median-first order: got %v, want mid,fast,slow,unmeasured", ids)
	}
}

// TestPlanDrainFallsBackAndParksUnplaced proves the two placement rules the
// drain relies on: a peer without room is skipped in favor of the next
// median-ordered peer (failover fallback), and a VM no peer can hold is
// reported unplaced (drain does NOT fail because of it).
func TestPlanDrainFallsBackAndParksUnplaced(t *testing.T) {
	vms := []drainVM{
		{ID: "vm-1", MemoryMB: 512},
		{ID: "vm-2", MemoryMB: 4096}, // only "huge" fits
		{ID: "vm-3", MemoryMB: 99999}, // fits nowhere
	}
	cands := []drainCandidate{
		{NodeID: "small", LinkBps: 100e6, MemFreeMB: 512},
		{NodeID: "mid", LinkBps: 500e6, MemFreeMB: 1024},
		{NodeID: "huge", LinkBps: 900e6, MemFreeMB: 8192},
	}
	assignments, unplaced := planDrain(vms, cands)
	if len(assignments) != 2 {
		t.Fatalf("expected 2 assignments, got %d (%+v)", len(assignments), assignments)
	}
	if assignments[0].VM.ID != "vm-1" || assignments[0].TargetNode != "mid" {
		t.Fatalf("vm-1: first offer must be the median peer with room, got %+v", assignments[0])
	}
	if assignments[1].VM.ID != "vm-2" || assignments[1].TargetNode != "huge" {
		t.Fatalf("vm-2: must fail over to the next fitting peer, got %+v", assignments[1])
	}
	if len(unplaced) != 1 || unplaced[0].ID != "vm-3" {
		t.Fatalf("vm-3 must be the only unplaced VM, got %+v", unplaced)
	}
}

// TestNodeDrainHandlerQueuesMigrationsAndFinishesDrained runs the real POST
// /nodes/{id}/drain handler against a sqlmock DB and the stub VM manager: the
// handler must 202 with the queued migration ids, and once the node has no
// drainable VMs left the coordinator's completion check must flip the node to
// the final 'drained' state.
func TestNodeDrainHandlerQueuesMigrationsAndFinishesDrained(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()

	const nodeID = "node-a"
	const peerID = "node-b"
	const vmID = "vm-drain-1"
	seedManagerVM(t, manager, vmID)
	manager.RegisterMigrationPeer(peerID, "http://127.0.0.1:1")

	// A private transfer store: migrations "complete" instantly, no qemu.
	store := newTransferStore(16, func(ctx context.Context, _ *fabricTransfer, _ *core_vm.VMManager) error {
		return nil
	})
	coord := &drainCoordinator{
		db:          db,
		vmManager:   manager,
		store:       store,
		pollInterval: time.Hour, // the async watcher must not race the sync asserts
		candidates: func(exclude string) []drainCandidate {
			if exclude != nodeID {
				t.Errorf("candidates called with exclude=%q, want %q", exclude, nodeID)
			}
			return []drainCandidate{{NodeID: peerID, LinkBps: 500e6, MemFreeMB: 4096}}
		},
	}

	router := mux.NewRouter()
	registerNodeAdminRoutesWithCoordinator(router, coord)

	// Known node -> not yet draining.
	mock.ExpectQuery("SELECT EXISTS").
		WithArgs(nodeID).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	mock.ExpectQuery("SELECT drain_state FROM cluster_nodes").
		WithArgs(nodeID).
		WillReturnRows(sqlmock.NewRows([]string{"drain_state"})) // no row: active
	mock.ExpectExec("INSERT INTO cluster_nodes").
		WithArgs(nodeID, nodeDrainDraining).
		WillReturnResult(sqlmock.NewResult(1, 1))
	mock.ExpectQuery("SELECT id, memory_mb, state FROM vms").
		WithArgs(nodeID).
		WillReturnRows(sqlmock.NewRows([]string{"id", "memory_mb", "state"}).AddRow(vmID, 256, "running"))

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/nodes/"+nodeID+"/drain", nil))
	if rec.Code != http.StatusAccepted {
		t.Fatalf("drain: expected 202, got %d (%s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	if !strings.Contains(body, `"migration_ids":[`) || !strings.Contains(body, nodeDrainDraining) {
		t.Fatalf("drain response must carry the migration id list and drain_state=%s, got %s", nodeDrainDraining, body)
	}

	// The queued transfer must complete against the stub store (the manager's
	// VM moves off as far as the fabric is concerned).
	deadline := time.Now().Add(2 * time.Second)
	var snapshot fabricTransfer
	for time.Now().Before(deadline) {
		ts := store.list()
		if len(ts) == 1 {
			snapshot = ts[0]
			if snapshot.Status == transferCompleted {
				break
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	if snapshot.VMID != vmID || snapshot.TargetNode != peerID {
		t.Fatalf("queued transfer must move %s to %s, got %+v", vmID, peerID, snapshot)
	}
	if snapshot.Status != transferCompleted {
		t.Fatalf("stub migration must complete, got status %q", snapshot.Status)
	}

	// Completion check: no drainable (or parked 'migrating') VMs remain ->
	// the transition finishes on the 'drained' string.
	mock.ExpectQuery("SELECT COUNT\\(\\*\\) FROM vms").
		WithArgs(nodeID).
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(0))
	mock.ExpectExec("INSERT INTO cluster_nodes").
		WithArgs(nodeID, nodeDrainDrained).
		WillReturnResult(sqlmock.NewResult(1, 1))
	done, err := coord.settle(nodeID)
	if err != nil {
		t.Fatalf("settle: %v", err)
	}
	if !done {
		t.Fatalf("settle: node with zero remaining VMs must report the drain finished")
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unfulfilled sql expectations: %v", err)
	}
}

// TestNodeDrainHandlerConflictAndNotFound pins the two error paths: draining
// again while a drain is in flight is 409, an unknown node id is 404.
func TestNodeDrainHandlerConflictAndNotFound(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()

	coord := newDrainCoordinator(db, manager, "", nil)
	coord.pollInterval = time.Hour
	router := mux.NewRouter()
	registerNodeAdminRoutesWithCoordinator(router, coord)

	// Duplicate drain: known node already draining -> 409.
	mock.ExpectQuery("SELECT EXISTS").
		WithArgs("node-a").
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	mock.ExpectQuery("SELECT drain_state FROM cluster_nodes").
		WithArgs("node-a").
		WillReturnRows(sqlmock.NewRows([]string{"drain_state"}).AddRow(nodeDrainDraining))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/nodes/node-a/drain", nil))
	if rec.Code != http.StatusConflict {
		t.Fatalf("duplicate drain: expected 409, got %d (%s)", rec.Code, rec.Body.String())
	}

	// Unknown node: 404 before anything else.
	mock.ExpectQuery("SELECT EXISTS").
		WithArgs("ghost-node").
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/nodes/ghost-node/drain", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("unknown node: expected 404, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unfulfilled sql expectations: %v", err)
	}
}

// TestNodeGetReportsDrainStateAndVMCount covers GET /nodes/{id}: drain_state
// plus the live vm_count, 404 for unknown ids.
func TestNodeGetReportsDrainStateAndVMCount(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	manager := newStubVMManager(t)
	defer manager.Stop()

	coord := newDrainCoordinator(db, manager, "", nil)
	router := mux.NewRouter()
	registerNodeAdminRoutesWithCoordinator(router, coord)

	mock.ExpectQuery("SELECT EXISTS").
		WithArgs("node-a").
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	mock.ExpectQuery("SELECT drain_state FROM cluster_nodes").
		WithArgs("node-a").
		WillReturnRows(sqlmock.NewRows([]string{"drain_state"}).AddRow(nodeDrainDraining))
	mock.ExpectQuery("SELECT COUNT\\(\\*\\) FROM vms").
		WithArgs("node-a").
		WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(3))

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/nodes/node-a", nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("get node: expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	body := rec.Body.String()
	for _, want := range []string{`"drain_state":"draining"`, `"vm_count":3`, `"node_id":"node-a"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("get node: response missing %s: %s", want, body)
		}
	}

	// Unknown node id -> 404.
	mock.ExpectQuery("SELECT EXISTS").
		WithArgs("ghost-node").
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/nodes/ghost-node", nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("get unknown node: expected 404, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unfulfilled sql expectations: %v", err)
	}
}
