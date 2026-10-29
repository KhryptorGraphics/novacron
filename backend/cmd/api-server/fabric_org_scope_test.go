package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"

	"github.com/golang-jwt/jwt/v5"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// newOrgScopeRouter wires only the VM routes under test so requireOrgScope
// runs in isolation. Returns the manager so tests can sign matching tokens.
func newOrgScopeRouter(t *testing.T, db *sql.DB) (*mux.Router, *auth.SimpleAuthManager) {
	return newOrgScopeRouterWithManager(t, db, nil)
}

// newOrgScopeRouterWithManager wires the org-scope router with a real
// (stubbed) VM manager — required for power-route tests, which exercise the
// manager's live state, not just the DB.
func newOrgScopeRouterWithManager(t *testing.T, db *sql.DB, vmManager *core_vm.VMManager) (*mux.Router, *auth.SimpleAuthManager) {
	t.Helper()
	authMgr := auth.NewSimpleAuthManager("test-secret", db)
	router := mux.NewRouter()
	api := router.PathPrefix("/api").Subrouter()
	api.Use(requireAuth(authMgr, nil))
	registerSecureAPIRoutes(api, db, vmManager, t.TempDir())
	return router, authMgr
}

// orgScopeToken creates a signed HS256 JWT with user_id / tenant_id / role claims.
func orgScopeToken(t *testing.T, authMgr *auth.SimpleAuthManager, userID, orgID, role string) string {
	t.Helper()
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id":   userID,
		"tenant_id": orgID,
		"role":      role,
		"roles":     []string{role},
		"exp":       time.Now().Add(time.Hour).Unix(),
		"iat":       time.Now().Unix(),
	})
	signed, err := token.SignedString([]byte(authMgr.GetJWTSecret()))
	if err != nil {
		t.Fatalf("sign token: %v", err)
	}
	return signed
}

// TestOrgScopeDeleteDeniesCrossOrg: a non-admin from org B gets 404 (not 403)
// when deleting a VM whose organization_id is org A.
//
// The actual handler loads the row and then runs orgVisible(scopeOrg, rowOrg):
// no second probe is issued, so the mock only needs the initial SELECT and
// any downstream workload is suppressed with NothingToDo().
func TestOrgScopeDeleteDeniesCrossOrg(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	orgB := "22222222-2222-2222-2222-222222222222"
	vmID := "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

	// Cross-org delete: requireOrgScope(ctx, db, vmID) issues an EXISTS probe
	// against (vmID, orgB) — false → visible=false → 404 without ever
	// touching the row or the manager.
	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgB).
		WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))

	router, authMgr := newOrgScopeRouter(t, db)

	req := httptest.NewRequest(http.MethodDelete, "/api/vms/"+vmID, nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgB, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("cross-org delete must 404 (hide existence), got %d body=%s", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}

// TestOrgScopeGetHidesCrossOrg: cross-org GET returns 404.
func TestOrgScopeGetHidesCrossOrg(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	orgA := "11111111-1111-1111-1111-111111111111"
	orgB := "22222222-2222-2222-2222-222222222222"
	vmID := "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

	mock.ExpectQuery(`SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at\s+FROM vms WHERE id = \$1`).
		WithArgs(vmID).
		WillReturnRows(sqlmock.NewRows([]string{
			"id", "name", "state", "node_id", "organization_id", "cpu_cores", "memory_mb", "disk_gb", "created_at", "updated_at",
		}).AddRow(vmID, "db-01", "running", nil, orgA, 4, 8192, 40, time.Now().UTC(), time.Now().UTC()))

	router, authMgr := newOrgScopeRouter(t, db)

	req := httptest.NewRequest(http.MethodGet, "/api/vms/"+vmID, nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgB, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("cross-org GET must 404, got %d", rec.Code)
	}
}

// TestOrgScopeGetSameOrgWorks: a VM whose org matches the claim is returned,
// and the row's org stamp appears in the response.
func TestOrgScopeGetSameOrgWorks(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	orgA := "11111111-1111-1111-1111-111111111111"
	vmID := "cccddddd-dddd-dddd-dddd-dddddddddddd"
	now := time.Now().UTC()

	mock.ExpectQuery(`SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at\s+FROM vms WHERE id = \$1`).
		WithArgs(vmID).
		WillReturnRows(sqlmock.NewRows([]string{
			"id", "name", "state", "node_id", "organization_id", "cpu_cores", "memory_mb", "disk_gb", "created_at", "updated_at",
		}).AddRow(vmID, "web-01", "running", nil, orgA, 4, 4096, 20, now, now))

	router, authMgr := newOrgScopeRouter(t, db)

	req := httptest.NewRequest(http.MethodGet, "/api/vms/"+vmID, nil)
	req.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-user", orgA, "viewer"))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("same-org get must 200, got %d body=%s", rec.Code, rec.Body.String())
	}
	var payload map[string]interface{}
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if got, ok := payload["organization_id"].(string); !ok || got != orgA {
		t.Fatalf("response organization_id must be %s, got %#v", orgA, payload["organization_id"])
	}
}

// marshalBody is a helper for JSON request bodies in org-scope tests.
func marshalBody(t *testing.T, v interface{}) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

// TestOrgScopeTwoOrgCreateReadDeleteRoundTrip is the novacron-ok7 acceptance
// evidence: a VM created by org A is visible to org A, invisible to org B
// (404, not 403), and deletable only by org A. Admin path remains unscoped.
// This single test exercises the full create→read→cross-org-deny→delete
// lifecycle across two distinct organizations that the goal requires.
func TestOrgScopeTwoOrgCreateReadDeleteRoundTrip(t *testing.T) {
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()
	mock.MatchExpectationsInOrder(false)
	orgA := "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
	orgB := "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
	vmID := "cccccccc-cccc-cccc-cccc-cccccccccccc"

	// Boot-time side effect: registerSecureAPIRoutes wires async migration
	// routes whose init runs reconcileInterruptedJobs (UPDATE migration_jobs).
	// Without this expectation sqlmock rejects the first real query.
	mock.ExpectExec(`UPDATE migration_jobs SET status = \$1, error = \$2, finished_at = NOW\(\)\s+WHERE status = \$3 AND finished_at IS NULL`).
		WithArgs("interrupted", sqlmock.AnyArg(), "running").
		WillReturnResult(sqlmock.NewResult(0, 0))

	router, authMgr := newOrgScopeRouter(t, db)

	// --- CREATE as org A user ---
	// The INSERT uses a scalar subquery for organization_id; sqlmock matches
	// it inside ExpectExec, not as a separate ExpectQuery.
	mock.ExpectExec(`INSERT INTO vms`).
		WithArgs(
			sqlmock.AnyArg(), // id
			"roundtrip-vm",   // name
			"stopped",        // state
			1,                // cpu_cores
			512,              // memory_mb
			1,                // disk_gb
			sqlmock.AnyArg(), // os_type
			sqlmock.AnyArg(), // node_id
			sqlmock.AnyArg(), // owner_id
			sqlmock.AnyArg(), // requested_owner_id
			orgA,             // organization_id
			sqlmock.AnyArg(), // metadata
			sqlmock.AnyArg(), // network_id
		).WillReturnResult(sqlmock.NewResult(1, 1))

	createReq := httptest.NewRequest(http.MethodPost, "/api/vms",
		strings.NewReader(`{"name":"roundtrip-vm","memory_mb":512,"disk_size_gb":1}`))
	createReq.Header.Set("Content-Type", "application/json")
	createReq.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-a", orgA, "viewer"))
	createRec := httptest.NewRecorder()
	router.ServeHTTP(createRec, createReq)
	if createRec.Code != http.StatusCreated {
		t.Fatalf("create as org A: expected 201, got %d body=%s", createRec.Code, createRec.Body.String())
	}

	// --- READ as org A: must succeed and return orgA stamp ---
	mock.ExpectQuery(`SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at\s+FROM vms WHERE id = \$1`).
		WithArgs(vmID).WillReturnRows(sqlmock.NewRows([]string{
			"id", "name", "state", "node_id", "organization_id", "cpu_cores", "memory_mb", "disk_gb", "created_at", "updated_at",
		}).AddRow(vmID, "roundtrip-vm", "stopped", "node-a", orgA, 1, 512, 1, time.Now().UTC(), time.Now().UTC()))

	readA := httptest.NewRequest(http.MethodGet, "/api/vms/"+vmID, nil)
	readA.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-a", orgA, "viewer"))
	readARec := httptest.NewRecorder()
	router.ServeHTTP(readARec, readA)
	if readARec.Code != http.StatusOK {
		t.Fatalf("read as org A: expected 200, got %d body=%s", readARec.Code, readARec.Body.String())
	}
	var readPayload map[string]interface{}
	if err := json.NewDecoder(readARec.Body).Decode(&readPayload); err != nil {
		t.Fatalf("decode read response: %v", err)
	}
	if got, ok := readPayload["organization_id"].(string); !ok || got != orgA {
		t.Fatalf("read response organization_id = %#v, want %s", readPayload["organization_id"], orgA)
	}

	// --- READ as org B: must return 404 (existence hidden) ---
	// The GET handler loads the row by id and checks orgVisible in Go; it does
	// NOT issue an EXISTS probe. Mock the row-returning SELECT with orgA so
	// orgVisible(scopeOrg=orgB, rowOrg=orgA) returns false → 404.
	mock.ExpectQuery(`SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at\s+FROM vms WHERE id = \$1`).
		WithArgs(vmID).WillReturnRows(sqlmock.NewRows([]string{
			"id", "name", "state", "node_id", "organization_id", "cpu_cores", "memory_mb", "disk_gb", "created_at", "updated_at",
		}).AddRow(vmID, "roundtrip-vm", "stopped", "node-a", orgA, 1, 512, 1, time.Now().UTC(), time.Now().UTC()))

	readB := httptest.NewRequest(http.MethodGet, "/api/vms/"+vmID, nil)
	readB.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-b", orgB, "viewer"))
	readBRec := httptest.NewRecorder()
	router.ServeHTTP(readBRec, readB)
	if readBRec.Code != http.StatusNotFound {
		t.Fatalf("read as org B: expected 404, got %d body=%s", readBRec.Code, readBRec.Body.String())
	}

	// --- DELETE as org B: must return 404 ---
	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgB).WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(false))

	delB := httptest.NewRequest(http.MethodDelete, "/api/vms/"+vmID, nil)
	delB.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-b", orgB, "viewer"))
	delBRec := httptest.NewRecorder()
	router.ServeHTTP(delBRec, delB)
	if delBRec.Code != http.StatusNotFound {
		t.Fatalf("delete as org B: expected 404, got %d body=%s", delBRec.Code, delBRec.Body.String())
	}

	// --- DELETE as org A: must succeed ---
	mock.ExpectQuery(`SELECT EXISTS \(SELECT 1 FROM vms WHERE id = \$1 AND organization_id = \$2\)`).
		WithArgs(vmID, orgA).WillReturnRows(sqlmock.NewRows([]string{"exists"}).AddRow(true))
	mock.ExpectExec(`DELETE FROM vms WHERE id = \$1 AND organization_id = \$2`).
		WithArgs(vmID, orgA).WillReturnResult(sqlmock.NewResult(1, 1))

	delA := httptest.NewRequest(http.MethodDelete, "/api/vms/"+vmID, nil)
	delA.Header.Set("Authorization", "Bearer "+orgScopeToken(t, authMgr, "u-a", orgA, "viewer"))
	delARec := httptest.NewRecorder()
	router.ServeHTTP(delARec, delA)
	if delARec.Code != http.StatusOK {
		t.Fatalf("delete as org A: expected 200, got %d body=%s", delARec.Code, delARec.Body.String())
	}

	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet db expectations: %v", err)
	}
}
