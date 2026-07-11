package handlers

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/gorilla/mux"
)

func newRBACTestRouter(t *testing.T) (*mux.Router, sqlmock.Sqlmock) {
	t.Helper()
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	t.Cleanup(func() { db.Close() })

	h := NewSecurityHandlers(nil).WithRBACStore(NewPostgresRBACStore(db))
	router := mux.NewRouter()
	h.RegisterRoutes(router)
	return router, mock
}

// TestCreateRoleHandlerRoundTrip proves the CreateRole HTTP handler is wired
// (POST /api/security/rbac/roles), issues the expected INSERT, and responds
// 201 with a JSON body shaped {id,name,description,permissions} — the exact
// SecurityRoleDefinition shape frontend/src/lib/api/security.ts expects.
func TestCreateRoleHandlerRoundTrip(t *testing.T) {
	router, mock := newRBACTestRouter(t)

	mock.ExpectQuery("INSERT INTO roles").
		WithArgs("auditor", "Auditor", "Audits things", []byte(`["audit.read"]`)).
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "description", "permissions"}).
			AddRow("auditor", "Auditor", "Audits things", []byte(`["audit.read"]`)))

	body, _ := json.Marshal(RoleDefinition{ID: "auditor", Name: "Auditor", Description: "Audits things", Permissions: []string{"audit.read"}})
	req := httptest.NewRequest(http.MethodPost, "/api/security/rbac/roles", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d (%s)", rec.Code, rec.Body.String())
	}

	var got map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	for _, key := range []string{"id", "name", "description", "permissions"} {
		if _, ok := got[key]; !ok {
			t.Fatalf("response missing frontend-required key %q: %#v", key, got)
		}
	}
	if got["id"] != "auditor" {
		t.Fatalf("unexpected id: %#v", got["id"])
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestCreateRoleHandlerRejectsMissingName(t *testing.T) {
	router, mock := newRBACTestRouter(t)

	body, _ := json.Marshal(RoleDefinition{ID: "no-name"})
	req := httptest.NewRequest(http.MethodPost, "/api/security/rbac/roles", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations (no query should run): %v", err)
	}
}

func TestUpdateRoleHandlerRoundTrip(t *testing.T) {
	router, mock := newRBACTestRouter(t)

	mock.ExpectQuery("UPDATE roles SET").
		WithArgs("Updated Name", "admin").
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "description", "permissions"}).
			AddRow("admin", "Updated Name", "desc", []byte(`["read"]`)))

	body, _ := json.Marshal(RoleUpdate{Name: "Updated Name"})
	req := httptest.NewRequest(http.MethodPut, "/api/security/rbac/roles/admin", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}
	var got RoleDefinition
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if got.Name != "Updated Name" {
		t.Fatalf("unexpected name: %#v", got)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestUpdateRoleHandlerNotFound(t *testing.T) {
	router, mock := newRBACTestRouter(t)

	mock.ExpectQuery("UPDATE roles SET").WillReturnError(sql.ErrNoRows)

	body, _ := json.Marshal(RoleUpdate{Name: "X"})
	req := httptest.NewRequest(http.MethodPut, "/api/security/rbac/roles/ghost", bytes.NewReader(body))
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestDeleteRoleHandlerRoundTrip(t *testing.T) {
	router, mock := newRBACTestRouter(t)

	mock.ExpectExec("DELETE FROM roles").
		WithArgs("auditor").
		WillReturnResult(sqlmock.NewResult(0, 1))

	req := httptest.NewRequest(http.MethodDelete, "/api/security/rbac/roles/auditor", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Fatalf("expected 204, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestDeleteRoleHandlerNotFound(t *testing.T) {
	router, mock := newRBACTestRouter(t)

	mock.ExpectExec("DELETE FROM roles").
		WithArgs("ghost").
		WillReturnResult(sqlmock.NewResult(0, 0))

	req := httptest.NewRequest(http.MethodDelete, "/api/security/rbac/roles/ghost", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusNotFound {
		t.Fatalf("expected 404, got %d (%s)", rec.Code, rec.Body.String())
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

// TestGetRolesHandlerJSONShape locks the exact wire shape
// frontend/src/lib/api/security.ts's securityAPI.getRoles() depends on:
// {"roles": [{"id","name","description","permissions"}, ...]}.
func TestGetRolesHandlerJSONShape(t *testing.T) {
	router, mock := newRBACTestRouter(t)

	mock.ExpectQuery("SELECT id, name, description, permissions FROM roles").
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "description", "permissions"}).
			AddRow("admin", "Administrator", "Admin access", []byte(`["read","write"]`)))

	req := httptest.NewRequest(http.MethodGet, "/api/security/rbac/roles", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (%s)", rec.Code, rec.Body.String())
	}

	var got struct {
		Roles []RoleDefinition `json:"roles"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &got); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if len(got.Roles) != 1 || got.Roles[0].ID != "admin" || len(got.Roles[0].Permissions) != 2 {
		t.Fatalf("unexpected roles payload: %#v", got.Roles)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

// TestRoleHandlersUnsupportedWithoutStore proves CreateRole/UpdateRole/
// DeleteRole fail closed (501) rather than panicking when no RBAC store is
// configured, matching the existing GetRoles/AssignUserRoles convention.
func TestRoleHandlersUnsupportedWithoutStore(t *testing.T) {
	h := NewSecurityHandlers(nil)
	router := mux.NewRouter()
	h.RegisterRoutes(router)

	cases := []struct {
		method, path string
	}{
		{http.MethodPost, "/api/security/rbac/roles"},
		{http.MethodPut, "/api/security/rbac/roles/admin"},
		{http.MethodDelete, "/api/security/rbac/roles/admin"},
	}
	for _, c := range cases {
		req := httptest.NewRequest(c.method, c.path, bytes.NewReader([]byte(`{}`)))
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		if rec.Code != http.StatusNotImplemented {
			t.Fatalf("%s %s: expected 501 without a store, got %d (%s)", c.method, c.path, rec.Code, rec.Body.String())
		}
	}
}
