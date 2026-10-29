package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"runtime"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// These tests never stub the host architecture: peers report an arch computed
// from runtime.GOARCH (same-arch alias / guaranteed-different arch), so they hold
// on x86_64 and aarch64 CI alike.

// hostArchAlias is the host's architecture spelled the way uname reports it
// (not Go's GOARCH), so a "compatible" peer also exercises normalization.
func hostArchAlias() string {
	switch normalizeArch(runtime.GOARCH) {
	case "x86_64":
		return "x86_64"
	case "aarch64":
		return "aarch64"
	}
	return runtime.GOARCH
}

// foreignArch is an architecture guaranteed to differ from the host's.
func foreignArch() string {
	if normalizeArch(runtime.GOARCH) == "aarch64" {
		return "amd64"
	}
	return "arm64"
}

// archPeer serves /internal/cluster/capacity like a real peer. arch == "" models
// an older build whose capacity response has no arch field at all.
func archPeer(t *testing.T, nodeID, arch string, memTotalMB int64) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/internal/cluster/capacity" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		body := map[string]interface{}{"node_id": nodeID, "reachable": true, "cores": 4, "mem_total_mb": memTotalMB}
		if arch != "" {
			body["arch"] = arch
		}
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(srv.Close)
	return joinTestRequestAddr(t, srv.URL)
}

func adminCtx(req *http.Request) *http.Request {
	return req.WithContext(context.WithValue(req.Context(), "role", "admin"))
}

func TestNormalizeArch(t *testing.T) {
	for in, want := range map[string]string{
		"amd64": "x86_64", "x86_64": "x86_64", "X86-64": "x86_64", " AMD64 ": "x86_64", "x64": "x86_64",
		"arm64": "aarch64", "aarch64": "aarch64", "ARM64": "aarch64", "armv8": "aarch64",
		"386": "i386", "i686": "i386", "armv7l": "arm", "riscv64": "riscv64", "": "", "  ": "",
	} {
		if got := normalizeArch(in); got != want {
			t.Errorf("normalizeArch(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestArchsIncompatible(t *testing.T) {
	for _, tc := range []struct {
		a, b string
		want bool
	}{
		{"arm64", "amd64", true},
		{"aarch64", "x86_64", true},
		{"x86_64", "arm64", true},
		{"amd64", "x86_64", false},
		{"arm64", "aarch64", false},
		{"arm64", "", false}, // unknown is never a mismatch
		{"", "amd64", false},
		{"", "", false},
	} {
		if got := archsIncompatible(tc.a, tc.b); got != tc.want {
			t.Errorf("archsIncompatible(%q, %q) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
	}
}

func TestCheckMigrationArch(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "self-node")
	mgr := newStubVMManager(t)
	defer mgr.Stop()
	mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 4096))
	mgr.RegisterMigrationPeer("peer-same", archPeer(t, "peer-same", hostArchAlias(), 4096))
	mgr.RegisterMigrationPeer("peer-legacy", archPeer(t, "peer-legacy", "", 4096))
	mgr.RegisterMigrationPeer("peer-down", "127.0.0.1:1") // nothing listens: connection refused

	t.Run("foreign arch is refused with both sides named", func(t *testing.T) {
		warning, refusal := checkMigrationArch(mgr, "peer-foreign", "")
		if refusal == nil || warning != "" {
			t.Fatalf("want refusal, got warning=%q refusal=%v", warning, refusal)
		}
		if refusal.SourceNode != "self-node" || refusal.TargetNode != "peer-foreign" ||
			refusal.SourceArch != normalizeArch(runtime.GOARCH) || refusal.TargetArch != normalizeArch(foreignArch()) {
			t.Fatalf("refusal names the wrong nodes/arches: %+v", refusal)
		}
	})
	t.Run("same arch under an alias spelling passes silently", func(t *testing.T) {
		if warning, refusal := checkMigrationArch(mgr, "peer-same", ""); refusal != nil || warning != "" {
			t.Fatalf("same arch must pass cleanly, got warning=%q refusal=%v", warning, refusal)
		}
	})
	for _, name := range []string{"peer-legacy", "peer-down", "not-a-peer"} {
		t.Run("unknown arch is allowed with a warning: "+name, func(t *testing.T) {
			warning, refusal := checkMigrationArch(mgr, name, "")
			if refusal != nil {
				t.Fatalf("unknown arch must not block, got %v", refusal)
			}
			if !strings.Contains(warning, name) || !strings.Contains(warning, "unknown") {
				t.Fatalf("unknown arch must warn about %q, got %q", name, warning)
			}
		})
	}
	t.Run("explicit probe address wins over the registered one", func(t *testing.T) {
		// peer-same is registered same-arch, but the guest is being sent to a
		// foreign-arch host at an explicit address.
		_, refusal := checkMigrationArch(mgr, "peer-same", archPeer(t, "peer-same", foreignArch(), 4096))
		if refusal == nil {
			t.Fatal("explicit foreign-arch address must be refused")
		}
	})
	t.Run("migrating to self is never refused", func(t *testing.T) {
		if warning, refusal := checkMigrationArch(mgr, "self-node", ""); refusal != nil || warning != "" {
			t.Fatalf("self target: warning=%q refusal=%v", warning, refusal)
		}
	})
}

// TestMigrateRouteRefusesCrossArch goes through the PRODUCTION constructor
// (registerVMMigrateRoute with a real manager), so it fails if production wiring
// passes no gate -- an injected-gate test could not catch that.
func TestMigrateRouteRefusesCrossArch(t *testing.T) {
	mgr := newStubVMManager(t)
	defer mgr.Stop()
	mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 4096))
	mgr.RegisterMigrationPeer("peer-same", archPeer(t, "peer-same", hostArchAlias(), 4096))
	router := mux.NewRouter()
	registerVMMigrateRoute(router, nil, mgr)
	post := func(target string) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, adminCtx(mustJSONRequest(t, http.MethodPost, "/vms/vm-x/migrate", map[string]string{"target_node": target})))
		return rec
	}

	rec := post("peer-foreign")
	if rec.Code != http.StatusUnprocessableEntity {
		t.Fatalf("cross-arch migrate: status %d, want 422; body=%s", rec.Code, rec.Body.String())
	}
	var body map[string]string
	decodeJSONBody(t, rec, &body)
	if body["code"] != archMismatchCode || body["target_node"] != "peer-foreign" ||
		body["source_arch"] != normalizeArch(runtime.GOARCH) || body["target_arch"] != normalizeArch(foreignArch()) ||
		!strings.Contains(body["error"], "same-architecture") {
		t.Fatalf("refusal body does not carry the contract: %v", body)
	}

	// Same-arch peer clears the gate and reaches the REAL manager, which then
	// 404s the (unseeded) VM -- proof the 422 above came from the gate, not from
	// a generic failure.
	if rec := post("peer-same"); rec.Code != http.StatusNotFound || strings.Contains(rec.Body.String(), archMismatchCode) {
		t.Fatalf("same-arch migrate must pass the gate and reach the manager (404 for the unknown VM), got %d body=%s", rec.Code, rec.Body.String())
	}
}

// TestMigrateRouteReportsUnknownArchWarning: an unknown-arch destination is not
// blocked and the success body carries a "warnings" entry.
func TestMigrateRouteReportsUnknownArchWarning(t *testing.T) {
	mgr := newStubVMManager(t)
	defer mgr.Stop()
	mgr.RegisterMigrationPeer("peer-legacy", archPeer(t, "peer-legacy", "", 4096))
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	mock.ExpectExec(`DELETE FROM vms WHERE id = \$1`).WithArgs("vm-x").WillReturnResult(sqlmock.NewResult(0, 1))
	router := mux.NewRouter()
	moved := false
	registerVMMigrateRouteWithPreflight(router, db, func(context.Context, string, string, map[string]string) error {
		moved = true
		return nil
	}, newMigrationPreflight(mgr))

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, adminCtx(mustJSONRequest(t, http.MethodPost, "/vms/vm-x/migrate", map[string]string{"target_node": "peer-legacy"})))
	if rec.Code != http.StatusOK || !moved {
		t.Fatalf("unknown arch must not block: status %d moved=%v body=%s", rec.Code, moved, rec.Body.String())
	}
	var body struct {
		Warnings []string `json:"warnings"`
	}
	decodeJSONBody(t, rec, &body)
	if len(body.Warnings) != 1 || !strings.Contains(body.Warnings[0], "peer-legacy") {
		t.Fatalf("success body must warn about the unverified architecture, got %v", body.Warnings)
	}
}

// TestMigrateRouteGateRunsBeforeExecutor: with a gate wired, a cross-arch
// request never reaches the executor and no source row is touched.
func TestMigrateRouteGateRunsBeforeExecutor(t *testing.T) {
	mgr := newStubVMManager(t)
	defer mgr.Stop()
	mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 4096))
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	router := mux.NewRouter()
	called := false
	registerVMMigrateRouteWithPreflight(router, db, func(context.Context, string, string, map[string]string) error {
		called = true
		return nil
	}, newMigrationPreflight(mgr))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, adminCtx(mustJSONRequest(t, http.MethodPost, "/vms/vm-x/migrate", map[string]string{"target_node": "peer-foreign"})))
	if rec.Code != http.StatusUnprocessableEntity || called {
		t.Fatalf("status %d executor called=%v body=%s", rec.Code, called, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil { // no DELETE was issued
		t.Fatalf("source row must be untouched: %v", err)
	}
}

// TestMigrateAsyncRouteRefusesCrossArch: production constructor; the refusal is
// synchronous (422, not a 202 + failed job) and leaves no job behind.
func TestMigrateAsyncRouteRefusesCrossArch(t *testing.T) {
	mgr := newStubVMManager(t)
	defer mgr.Stop()
	mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 4096))
	mgr.RegisterMigrationPeer("peer-same", archPeer(t, "peer-same", hostArchAlias(), 4096))
	router := mux.NewRouter()
	registerVMMigrateAsyncRoutes(router, nil, mgr)
	jobCount := func() int {
		migrateJobs.mu.Lock()
		defer migrateJobs.mu.Unlock()
		return len(migrateJobs.jobs)
	}
	before := jobCount()

	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, adminCtx(mustJSONRequest(t, http.MethodPost, "/vms/vm-x/migrate/async", map[string]string{"target_node": "peer-foreign"})))
	if rec.Code != http.StatusUnprocessableEntity || !strings.Contains(rec.Body.String(), archMismatchCode) {
		t.Fatalf("cross-arch async migrate: status %d, want 422 with %s; body=%s", rec.Code, archMismatchCode, rec.Body.String())
	}
	if after := jobCount(); after != before {
		t.Fatalf("a refused migration must not create a job (jobs %d -> %d)", before, after)
	}

	// Same-arch: accepted (202); the real manager then fails the unknown VM, so
	// the job ends "failed" -- the gate did not stand in the way.
	rec = httptest.NewRecorder()
	router.ServeHTTP(rec, adminCtx(mustJSONRequest(t, http.MethodPost, "/vms/vm-x/migrate/async", map[string]string{"target_node": "peer-same"})))
	if rec.Code != http.StatusAccepted {
		t.Fatalf("same-arch async migrate: status %d, want 202; body=%s", rec.Code, rec.Body.String())
	}
	var accepted struct {
		JobID string `json:"job_id"`
	}
	decodeJSONBody(t, rec, &accepted)
	if job := waitForTerminal(t, router, accepted.JobID); job.Status != migrateStatusFailed {
		t.Fatalf("job should reach the manager and fail on the unknown VM, got %+v", job)
	}
}

// seedTransferTestVM registers a local VM and returns a sqlmock-backed transfer
// router for it (org-scoped viewer token so requireOrgScope's EXISTS probe runs).
func postTransferRequest(t *testing.T, vmID, targetNode string, mgr *core_vm.VMManager, db *sql.DB) *httptest.ResponseRecorder {
	t.Helper()
	const orgA = "11111111-1111-1111-1111-111111111111"
	router, authMgr := newTransferRouterWithVMManager(t, db, mgr)
	req := httptest.NewRequest(http.MethodPost, "/api/transfers", bytes.NewReader(marshalBody(t, map[string]interface{}{
		"kind": "migration", "vm_id": vmID, "target_node": targetNode,
	})))
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgA, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	return rec
}

// TestTransferPostRefusesCrossArch: production constructor
// (registerFabricTransferRoutes); a cross-arch target gets 422 and nothing is
// queued, a same-arch one is admitted.
func TestTransferPostRefusesCrossArch(t *testing.T) {
	const orgA = "11111111-1111-1111-1111-111111111111"
	const vmID = "dddddddd-dddd-dddd-dddd-dddddddddddd"
	defer setupTransferStore(t)()
	mgr := newStubVMManager(t)
	defer mgr.Stop()
	seedManagerVM(t, mgr, vmID)
	mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 4096))
	mgr.RegisterMigrationPeer("peer-same", archPeer(t, "peer-same", hostArchAlias(), 4096))

	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	for i := 0; i < 2; i++ { // one scope probe + one org load per request
		mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
			WithArgs(vmID, orgA).WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
		mock.ExpectQuery(`SELECT organization_id FROM vms WHERE id = \$1`).
			WithArgs(vmID).WillReturnRows(sqlmock.NewRows([]string{"organization_id"}).AddRow(orgA))
	}

	rec := postTransferRequest(t, vmID, "peer-foreign", mgr, db)
	if rec.Code != http.StatusUnprocessableEntity || !strings.Contains(rec.Body.String(), archMismatchCode) {
		t.Fatalf("cross-arch transfer: status %d, want 422 with %s; body=%s", rec.Code, archMismatchCode, rec.Body.String())
	}
	if got := transfers.list(); len(got) != 0 {
		t.Fatalf("a refused transfer must not be queued, got %d", len(got))
	}

	// Keep the admitted transfer from running a real migration.
	transfers.runFn = func(context.Context, *fabricTransfer, *core_vm.VMManager) error { return sql.ErrNoRows }
	if rec := postTransferRequest(t, vmID, "peer-same", mgr, db); rec.Code != http.StatusAccepted {
		t.Fatalf("same-arch transfer: status %d, want 202; body=%s", rec.Code, rec.Body.String())
	}
}

// TestQueueMigrationTransferRefusesCrossArch: the drain/healing admission path.
func TestQueueMigrationTransferRefusesCrossArch(t *testing.T) {
	mgr := newStubVMManager(t)
	defer mgr.Stop()
	seedManagerVM(t, mgr, "vm-q")
	mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 4096))
	mgr.RegisterMigrationPeer("peer-same", archPeer(t, "peer-same", hostArchAlias(), 4096))
	store := newTransferStore(16, func(context.Context, *fabricTransfer, *core_vm.VMManager) error { return nil })

	_, err := queueMigrationTransfer(store, mgr, "vm-q", 256, "peer-foreign", "test")
	var mismatch *archMismatchError
	if !errors.As(err, &mismatch) || mismatch.TargetNode != "peer-foreign" {
		t.Fatalf("cross-arch queue must fail with *archMismatchError, got %v", err)
	}
	if got := store.list(); len(got) != 0 {
		t.Fatalf("refused migration must not be queued, got %d", len(got))
	}
	if id, err := queueMigrationTransfer(store, mgr, "vm-q", 256, "peer-same", "test"); err != nil || id == "" {
		t.Fatalf("same-arch queue must succeed, got id=%q err=%v", id, err)
	}
}

// TestMigrationCandidatesSkipCrossArch: drain/healing target selection drops a
// known-foreign peer (even one with more free memory) and keeps unknown ones.
func TestMigrationCandidatesSkipCrossArch(t *testing.T) {
	t.Setenv("NOVACRON_NODE_ID", "self-node")
	mgr := newStubVMManager(t)
	defer mgr.Stop()
	mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 1<<20))
	mgr.RegisterMigrationPeer("peer-same", archPeer(t, "peer-same", hostArchAlias(), 4096))
	mgr.RegisterMigrationPeer("peer-legacy", archPeer(t, "peer-legacy", "", 2048))

	candidates := migrationCandidatesFunc(mgr, t.TempDir(), nil)(selfNodeID())
	var ids []string
	for _, c := range candidates {
		ids = append(ids, c.NodeID)
	}
	sort.Strings(ids)
	if strings.Join(ids, ",") != "peer-legacy,peer-same" {
		t.Fatalf("candidates = %v, want [peer-legacy peer-same] (foreign arch skipped, unknown kept)", ids)
	}
}

// TestDrainSkipsCrossArchPeers drives the real POST /nodes/{id}/drain handler:
// with a foreign-arch peer that has far more room, the VM still lands on the
// compatible peer; with ONLY a foreign peer the drain fails (503) and rolls the
// node back to active instead of queueing an impossible migration.
func TestDrainSkipsCrossArchPeers(t *testing.T) {
	const nodeID = "node-a"
	t.Setenv("NOVACRON_NODE_ID", nodeID)
	expectDrainStart := func(mock sqlmock.Sqlmock) {
		// nodeKnown short-circuits for this node's own id: no EXISTS probe.
		mock.ExpectQuery("SELECT drain_state FROM cluster_nodes").WithArgs(nodeID).WillReturnRows(sqlmock.NewRows([]string{"drain_state"}))
		mock.ExpectExec("INSERT INTO cluster_nodes").WithArgs(nodeID, nodeDrainDraining).WillReturnResult(sqlmock.NewResult(1, 1))
		mock.ExpectQuery("SELECT id, memory_mb, state FROM vms").WithArgs(nodeID).
			WillReturnRows(sqlmock.NewRows([]string{"id", "memory_mb", "state"}).AddRow("vm-drain", 256, "running"))
	}
	newRouter := func(t *testing.T, mgr *core_vm.VMManager, db *sql.DB) (*mux.Router, *transferStore) {
		store := newTransferStore(16, func(context.Context, *fabricTransfer, *core_vm.VMManager) error { return nil })
		coord := &drainCoordinator{
			db: db, vmManager: mgr, store: store, pollInterval: time.Hour,
			candidates: migrationCandidatesFunc(mgr, t.TempDir(), nil),
		}
		router := mux.NewRouter()
		registerNodeAdminRoutesWithCoordinator(router, coord)
		return router, store
	}
	drain := func(router *mux.Router) *httptest.ResponseRecorder {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/nodes/"+nodeID+"/drain", nil))
		return rec
	}

	t.Run("lands on the compatible peer", func(t *testing.T) {
		mgr := newStubVMManager(t)
		defer mgr.Stop()
		seedManagerVM(t, mgr, "vm-drain")
		mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 1<<20))
		mgr.RegisterMigrationPeer("peer-same", archPeer(t, "peer-same", hostArchAlias(), 4096))
		db, mock, err := sqlmock.New()
		if err != nil {
			t.Fatalf("sqlmock: %v", err)
		}
		defer db.Close()
		expectDrainStart(mock)
		router, store := newRouter(t, mgr, db)
		rec := drain(router)
		if rec.Code != http.StatusAccepted {
			t.Fatalf("drain: status %d, want 202; body=%s", rec.Code, rec.Body.String())
		}
		got := store.list()
		if len(got) != 1 || got[0].TargetNode != "peer-same" {
			t.Fatalf("VM must be queued to the compatible peer, got %+v", got)
		}
	})

	t.Run("only a foreign peer: drain fails, node rolled back to active", func(t *testing.T) {
		mgr := newStubVMManager(t)
		defer mgr.Stop()
		seedManagerVM(t, mgr, "vm-drain")
		mgr.RegisterMigrationPeer("peer-foreign", archPeer(t, "peer-foreign", foreignArch(), 1<<20))
		db, mock, err := sqlmock.New()
		if err != nil {
			t.Fatalf("sqlmock: %v", err)
		}
		defer db.Close()
		expectDrainStart(mock)
		mock.ExpectExec("INSERT INTO cluster_nodes").WithArgs(nodeID, nodeDrainActive).WillReturnResult(sqlmock.NewResult(1, 1))
		router, store := newRouter(t, mgr, db)
		rec := drain(router)
		if rec.Code != http.StatusServiceUnavailable || !strings.Contains(rec.Body.String(), "same-architecture") {
			t.Fatalf("drain with only a foreign peer: status %d body=%s, want 503 naming the architecture rule", rec.Code, rec.Body.String())
		}
		if got := store.list(); len(got) != 0 {
			t.Fatalf("nothing may be queued toward a foreign-arch peer, got %+v", got)
		}
		if err := mock.ExpectationsWereMet(); err != nil {
			t.Fatalf("drain state must roll back to active: %v", err)
		}
	})
}
