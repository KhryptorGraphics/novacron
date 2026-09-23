package main

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"

	"github.com/golang-jwt/jwt/v5"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// newOrgScopeRouter wires only the VM routes under test so requireOrgScope
// runs in isolation. Returns the manager so tests can sign matching tokens.
func newOrgScopeRouter(t *testing.T, db *sql.DB) (*mux.Router, *auth.SimpleAuthManager) {
	t.Helper()
	authMgr := auth.NewSimpleAuthManager("test-secret", db)
	router := mux.NewRouter()
	api := router.PathPrefix("/api").Subrouter()
	api.Use(requireAuth(authMgr))
	registerSecureAPIRoutes(api, db, nil, t.TempDir())
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
