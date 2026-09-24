package main

import (
	"bytes"
	"context"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// newTransferRouter wires only the fabric transfer routes under test so
// requireOrgScope runs in isolation. Read-route tests and denied POST requests
// do not need a live VM manager.
func newTransferRouter(t *testing.T, db *sql.DB) (*mux.Router, *auth.SimpleAuthManager) {
	return newTransferRouterWithVMManager(t, db, nil)
}

func newTransferRouterWithVMManager(t *testing.T, db *sql.DB, vmManager *core_vm.VMManager) (*mux.Router, *auth.SimpleAuthManager) {
	t.Helper()
	authMgr := auth.NewSimpleAuthManager("test-secret", db)
	router := mux.NewRouter()
	api := router.PathPrefix("/api").Subrouter()
	api.Use(requireAuth(authMgr))
	registerFabricTransferRoutes(api, db, vmManager, t.TempDir())
	return router, authMgr
}

// setupTransferStore isolates the global transfers registry for the test.
// Returns a cleanup function that restores the original registry.
func setupTransferStore(t *testing.T) func() {
	t.Helper()
	original := transfers
	transfers = newTransferStore(512, nil)
	return func() { transfers = original }
}

// seedTransfer inserts a transfer directly into the global registry without
// starting the background migration runner.
func seedTransfer(t *testing.T, vmID, orgID, targetNode string) *fabricTransfer {
	t.Helper()
	now := time.Now().UTC()
	tr := &fabricTransfer{
		ID:             "transfer-" + vmID,
		Kind:           "migration",
		VMID:           vmID,
		TargetNode:     targetNode,
		Status:         transferQueued,
		Compression:    "none",
		CreatedAt:      now,
		OrganizationID: orgID,
		Decision: transferDecisionInputs{
			LinkBps:       100e6,
			LinkMeasured:  true,
			SampleRatio:   1.0,
			ThresholdBps:  compressionLinkThresholdBps,
			ThresholdRate: compressionRatioThreshold,
			Reason:        "test",
		},
	}
	transfers.mu.Lock()
	transfers.byID[tr.ID] = tr
	transfers.order = append(transfers.order, tr.ID)
	transfers.mu.Unlock()
	return tr
}

// TestTransferPostDeniesCrossOrg: a non-admin from org B gets 404 (not 403)
// when POSTing a transfer for a VM whose organization_id is org A.
//
// The actual handler first runs requireOrgScope(ctx, db, vmID) which issues an
// EXISTS probe against (vmID, orgB) — false → visible=false → 404 without
// ever touching the row, the manager, or creating a transfer.
func TestTransferPostDeniesCrossOrg(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cleanup := setupTransferStore(t)
	defer cleanup()

	orgB := "22222222-2222-2222-2222-222222222222"
	vmID := "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

	// Cross-org transfer: requireOrgScope(ctx, db, vmID) issues an EXISTS probe
	// against (vmID, orgB) — false → visible=false → 404 without ever
	// touching the row or the manager.
	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgB).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))

	router, authMgr := newTransferRouter(t, db)

	req := httptest.NewRequest(http.MethodPost, "/api/transfers",
		bytes.NewReader(marshalBody(t, map[string]interface{}{
			"kind":        "migration",
			"vm_id":       vmID,
			"target_node": "node-B",
		})))
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgB, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	const notFoundBody = "{\"error\":\"vm not found on this node\"}\n"
	if rec.Code != http.StatusNotFound || rec.Body.String() != notFoundBody {
		t.Fatalf("cross-org POST must match hidden-VM 404, got %d body=%s", rec.Code, rec.Body.String())
	}
	// Verify no transfer was created.
	if got := transfers.list(); len(got) != 0 {
		t.Fatalf("cross-org POST must not create a transfer, got %d transfers", len(got))
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}

// TestTransferPostSameOrg: a VM whose org matches the claim and exists in the
// vms table passes scope and gets a transfer admitted.
func TestTransferPostSameOrg(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cleanup := setupTransferStore(t)
	defer cleanup()

	orgA := "11111111-1111-1111-1111-111111111111"
	vmID := "cccccccc-cccc-cccc-cccc-cccccccccccc"

	// requireOrgScope: EXISTS probe for same-org VM — true.
	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgA).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))

	// Load VM org for transfer ownership.
	mock.ExpectQuery(`SELECT organization_id FROM vms WHERE id = \$1`).
		WithArgs(vmID).
		WillReturnRows(sqlmock.NewRows([]string{"organization_id"}).AddRow(orgA))

	manager, err := core_vm.NewVMManager(core_vm.DefaultVMManagerConfig())
	if err != nil {
		t.Fatalf("new VM manager: %v", err)
	}
	defer manager.Shutdown()
	vm, err := core_vm.NewVM(core_vm.VMConfig{ID: vmID, Name: "test-vm", Type: core_vm.VMTypeKVM})
	if err != nil {
		t.Fatalf("new VM: %v", err)
	}
	manager.AddVM(vm)
	t.Setenv("NOVACRON_PEERS", "node-B=127.0.0.1:9000")
	registerConfiguredPeers(manager)

	router, authMgr := newTransferRouterWithVMManager(t, db, manager)
	// Keep the request focused on admission/ownership, not a real migration.
	transfers.runFn = func(context.Context, *fabricTransfer, *core_vm.VMManager) error { return sql.ErrNoRows }

	req := httptest.NewRequest(http.MethodPost, "/api/transfers",
		bytes.NewReader(marshalBody(t, map[string]interface{}{
			"kind":        "migration",
			"vm_id":       vmID,
			"target_node": "node-B",
		})))
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgA, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusAccepted {
		t.Fatalf("same-org POST must 202, got %d body=%s", rec.Code, rec.Body.String())
	}
	// Verify transfer was created with correct org and ownership remains internal.
	got := transfers.list()
	if len(got) != 1 {
		t.Fatalf("expected 1 transfer, got %d", len(got))
	}
	if got[0].OrganizationID != orgA {
		t.Fatalf("transfer org must be %s, got %s", orgA, got[0].OrganizationID)
	}
	if strings.Contains(rec.Body.String(), orgA) || strings.Contains(rec.Body.String(), "organization_id") {
		t.Fatalf("POST response must not expose internal organization data: %s", rec.Body.String())
	}
	// Admins may submit on behalf of another tenant; ownership must still be
	// bound to the VM's actual database organization.
	mock.ExpectQuery(`SELECT organization_id FROM vms WHERE id = \$1`).
		WithArgs(vmID).
		WillReturnRows(sqlmock.NewRows([]string{"organization_id"}).AddRow(orgA))
	adminReq := httptest.NewRequest(http.MethodPost, "/api/transfers",
		bytes.NewReader(marshalBody(t, map[string]interface{}{
			"kind":        "migration",
			"vm_id":       vmID,
			"target_node": "node-B",
		})))
	adminReq.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-admin", "", "admin"))
	adminRec := httptest.NewRecorder()
	router.ServeHTTP(adminRec, adminReq)
	if adminRec.Code != http.StatusAccepted {
		t.Fatalf("admin POST must 202, got %d body=%s", adminRec.Code, adminRec.Body.String())
	}
	got = transfers.list()
	if len(got) != 2 {
		t.Fatalf("expected both same-org and admin transfer admissions, got %#v", got)
	}
	for _, transfer := range got {
		if transfer.OrganizationID != orgA {
			t.Fatalf("every transfer for VM must retain org %s, got %s", orgA, transfer.OrganizationID)
		}
	}
	if strings.Contains(adminRec.Body.String(), orgA) || strings.Contains(adminRec.Body.String(), "organization_id") {
		t.Fatalf("admin POST response must not expose internal organization data: %s", adminRec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}

// TestTransferPostMissingVMFailsClosed: a VM that fails the EXISTS probe or
// the subsequent org lookup fails closed — no transfer is enqueued.
func TestTransferPostMissingVMFailsClosed(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cleanup := setupTransferStore(t)
	defer cleanup()

	orgA := "11111111-1111-1111-1111-111111111111"
	vmID := "dddddddd-dddd-dddd-dddd-dddddddddddd"

	// EXISTS probe passes (same-org, VM visible)...
	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgA).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))

	// ...but the org lookup fails (simulating a race where the VM row vanished
	// between checks, or a db error).
	mock.ExpectQuery(`SELECT organization_id FROM vms WHERE id = \$1`).
		WithArgs(vmID).
		WillReturnError(sql.ErrNoRows)

	manager, err := core_vm.NewVMManager(core_vm.DefaultVMManagerConfig())
	if err != nil {
		t.Fatalf("new VM manager: %v", err)
	}
	defer manager.Shutdown()
	vm, err := core_vm.NewVM(core_vm.VMConfig{ID: vmID, Name: "test-vm", Type: core_vm.VMTypeKVM})
	if err != nil {
		t.Fatalf("new VM: %v", err)
	}
	manager.AddVM(vm)
	router, authMgr := newTransferRouterWithVMManager(t, db, manager)
	req := httptest.NewRequest(http.MethodPost, "/api/transfers",
		bytes.NewReader(marshalBody(t, map[string]interface{}{
			"kind":        "migration",
			"vm_id":       vmID,
			"target_node": "node-B",
		})))
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgA, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	const notFoundBody = "{\"error\":\"vm not found on this node\"}\n"
	if rec.Code != http.StatusNotFound || rec.Body.String() != notFoundBody {
		t.Fatalf("missing VM must match hidden-VM 404, got %d body=%s", rec.Code, rec.Body.String())
	}
	// Verify no transfer was created.
	if got := transfers.list(); len(got) != 0 {
		t.Fatalf("failed org lookup must not create a transfer, got %d transfers", len(got))
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}

// TestTransferListHidesCrossOrg: a non-admin from org B does not see transfers
// belonging to org A in the list.
func TestTransferListHidesCrossOrg(t *testing.T) {
	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cleanup := setupTransferStore(t)
	defer cleanup()

	orgA := "11111111-1111-1111-1111-111111111111"
	orgB := "22222222-2222-2222-2222-222222222222"

	// Seed transfers for both orgs.
	seedTransfer(t, "vm-orgA-1", orgA, "node-B")
	seedTransfer(t, "vm-orgA-2", orgA, "node-C")
	seedTransfer(t, "vm-orgB-1", orgB, "node-B")

	router, authMgr := newTransferRouter(t, db)

	req := httptest.NewRequest(http.MethodGet, "/api/transfers", nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgB, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("list must 200, got %d", rec.Code)
	}
	var payload struct {
		Transfers []fabricTransfer `json:"transfers"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Only the orgB transfer should be visible.
	if len(payload.Transfers) != 1 {
		t.Fatalf("orgB must see exactly 1 transfer, got %d", len(payload.Transfers))
	}
	if payload.Transfers[0].VMID != "vm-orgB-1" {
		t.Fatalf("expected vm-orgB-1, got %s", payload.Transfers[0].VMID)
	}
}

// TestTransferDetailHidesCrossOrg: a non-admin from org B gets 404 when
// fetching a transfer owned by org A.
func TestTransferDetailHidesCrossOrg(t *testing.T) {
	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cleanup := setupTransferStore(t)
	defer cleanup()

	orgA := "11111111-1111-1111-1111-111111111111"
	orgB := "22222222-2222-2222-2222-222222222222"

	// Seed a transfer owned by org A.
	tr := seedTransfer(t, "vm-orgA-1", orgA, "node-B")

	router, authMgr := newTransferRouter(t, db)

	req := httptest.NewRequest(http.MethodGet, "/api/transfers/"+tr.ID, nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgB, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("cross-org detail must 404, got %d", rec.Code)
	}
}

// TestTransferDetailSameOrg: a non-admin from the same org can read the
// transfer detail.
func TestTransferDetailSameOrg(t *testing.T) {
	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cleanup := setupTransferStore(t)
	defer cleanup()

	orgA := "11111111-1111-1111-1111-111111111111"

	// Seed a transfer owned by org A.
	tr := seedTransfer(t, "vm-orgA-1", orgA, "node-B")

	router, authMgr := newTransferRouter(t, db)

	req := httptest.NewRequest(http.MethodGet, "/api/transfers/"+tr.ID, nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgA, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("same-org detail must 200, got %d", rec.Code)
	}
	body := rec.Body.String()
	if strings.Contains(body, orgA) || strings.Contains(body, "organization_id") || strings.Contains(body, "OrganizationID") {
		t.Fatalf("detail response must not expose internal organization data: %s", body)
	}
	var payload fabricTransfer
	if err := json.NewDecoder(strings.NewReader(body)).Decode(&payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if payload.VMID != "vm-orgA-1" {
		t.Fatalf("expected vm-orgA-1, got %s", payload.VMID)
	}
}

// TestTransferDetailNotFound: a non-existent transfer ID returns 404.
func TestTransferDetailNotFound(t *testing.T) {
	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cleanup := setupTransferStore(t)
	defer cleanup()

	orgA := "11111111-1111-1111-1111-111111111111"

	router, authMgr := newTransferRouter(t, db)

	req := httptest.NewRequest(http.MethodGet, "/api/transfers/nonexistent-id", nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgA, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("nonexistent transfer must 404, got %d", rec.Code)
	}
}

// TestTransferAdminSeesAll: an admin sees all transfers regardless of org.
func TestTransferAdminSeesAll(t *testing.T) {
	db, _, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	cleanup := setupTransferStore(t)
	defer cleanup()

	orgA := "11111111-1111-1111-1111-111111111111"
	orgB := "22222222-2222-2222-2222-222222222222"

	// Seed transfers for both orgs.
	seedTransfer(t, "vm-orgA-1", orgA, "node-B")
	seedTransfer(t, "vm-orgB-1", orgB, "node-C")
	seedTransfer(t, "vm-orgB-2", orgB, "node-D")

	router, authMgr := newTransferRouter(t, db)

	// Admin list: sees all.
	req := httptest.NewRequest(http.MethodGet, "/api/transfers", nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-admin", "", "admin"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("admin list must 200, got %d", rec.Code)
	}
	var payload struct {
		Transfers []fabricTransfer `json:"transfers"`
	}
	body := rec.Body.String()
	if err := json.NewDecoder(strings.NewReader(body)).Decode(&payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if len(payload.Transfers) != 3 {
		t.Fatalf("admin must see all 3 transfers, got %d", len(payload.Transfers))
	}
	if strings.Contains(body, orgA) || strings.Contains(body, orgB) || strings.Contains(body, "organization_id") || strings.Contains(body, "OrganizationID") {
		t.Fatalf("admin list must not expose internal organization data: %s", body)
	}

	// Admin detail: can read a transfer from another organization as well.
	target := "transfer-vm-orgA-1"
	detailReq := httptest.NewRequest(http.MethodGet, "/api/transfers/"+target, nil)
	detailReq.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-admin", "", "admin"))
	detailRec := httptest.NewRecorder()
	router.ServeHTTP(detailRec, detailReq)
	if detailRec.Code != http.StatusOK {
		t.Fatalf("admin detail must 200, got %d body=%s", detailRec.Code, detailRec.Body.String())
	}
	if strings.Contains(detailRec.Body.String(), orgA) || strings.Contains(detailRec.Body.String(), "organization_id") || strings.Contains(detailRec.Body.String(), "OrganizationID") {
		t.Fatalf("admin detail must not expose internal organization data: %s", detailRec.Body.String())
	}
}
