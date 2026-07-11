package handlers

import (
	"context"
	"database/sql"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
)

func newRBACTestStore(t *testing.T) (*PostgresRBACStore, sqlmock.Sqlmock) {
	t.Helper()
	db, mock, err := sqlmock.New()
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return NewPostgresRBACStore(db), mock
}

// TestListRolesReadsFromDB proves ListRoles is DB-backed (not the old
// hardcoded roleCatalog map): it returns exactly what the roles table rows
// say, JSONB permissions decoded into []string, in query order.
func TestListRolesReadsFromDB(t *testing.T) {
	store, mock := newRBACTestStore(t)

	rows := sqlmock.NewRows([]string{"id", "name", "description", "permissions"}).
		AddRow("admin", "Administrator", "Admin access", []byte(`["read","write"]`)).
		AddRow("super-admin", "Super Admin", "Full access", []byte(`["*"]`))
	mock.ExpectQuery("SELECT id, name, description, permissions FROM roles").WillReturnRows(rows)

	roles, err := store.ListRoles(context.Background())
	if err != nil {
		t.Fatalf("ListRoles: %v", err)
	}
	if len(roles) != 2 {
		t.Fatalf("expected 2 roles, got %d", len(roles))
	}
	if roles[0].ID != "admin" || len(roles[0].Permissions) != 2 || roles[0].Permissions[1] != "write" {
		t.Fatalf("unexpected role[0]: %#v", roles[0])
	}
	if roles[1].ID != "super-admin" || roles[1].Permissions[0] != "*" {
		t.Fatalf("unexpected role[1]: %#v", roles[1])
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestListPermissionsReadsFromDB(t *testing.T) {
	store, mock := newRBACTestStore(t)

	rows := sqlmock.NewRows([]string{"id", "name", "description"}).
		AddRow("read", "Read", "Read access to resources").
		AddRow("write", "Write", "Write access to resources")
	mock.ExpectQuery("SELECT id, name, description FROM permissions").WillReturnRows(rows)

	permissions, err := store.ListPermissions(context.Background())
	if err != nil {
		t.Fatalf("ListPermissions: %v", err)
	}
	if len(permissions) != 2 || permissions[0].ID != "read" || permissions[1].ID != "write" {
		t.Fatalf("unexpected permissions: %#v", permissions)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestCreateRoleInsertsAndReturnsRole(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectQuery("INSERT INTO roles").
		WithArgs("auditor", "Auditor", "Audits things", []byte(`["audit.read"]`)).
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "description", "permissions"}).
			AddRow("auditor", "Auditor", "Audits things", []byte(`["audit.read"]`)))

	created, err := store.CreateRole(context.Background(), RoleDefinition{
		ID: "Auditor", Name: "Auditor", Description: "Audits things", Permissions: []string{"audit.read"},
	})
	if err != nil {
		t.Fatalf("CreateRole: %v", err)
	}
	// ID must be normalized (lower-cased) even though the caller sent "Auditor".
	if created.ID != "auditor" {
		t.Fatalf("expected normalized id 'auditor', got %q", created.ID)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestCreateRoleRequiresIDAndName(t *testing.T) {
	store, mock := newRBACTestStore(t)

	if _, err := store.CreateRole(context.Background(), RoleDefinition{Name: "No ID"}); err == nil {
		t.Fatal("expected error for missing id")
	}
	if _, err := store.CreateRole(context.Background(), RoleDefinition{ID: "no-name"}); err == nil {
		t.Fatal("expected error for missing name")
	}
	// No queries should have been issued for invalid input.
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestCreateRoleDuplicateIDIsRejected(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectQuery("INSERT INTO roles").
		WillReturnError(&mockPQError{msg: `pq: duplicate key value violates unique constraint "roles_pkey"`})

	_, err := store.CreateRole(context.Background(), RoleDefinition{ID: "admin", Name: "Administrator"})
	if err == nil {
		t.Fatal("expected duplicate-key error")
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestUpdateRolePartialUpdate(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectQuery("UPDATE roles SET name = \\$1, updated_at = NOW\\(\\) WHERE id = \\$2").
		WithArgs("New Name", "admin").
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "description", "permissions"}).
			AddRow("admin", "New Name", "old desc", []byte(`["read"]`)))

	updated, err := store.UpdateRole(context.Background(), "admin", RoleUpdate{Name: "New Name"})
	if err != nil {
		t.Fatalf("UpdateRole: %v", err)
	}
	if updated.Name != "New Name" {
		t.Fatalf("expected updated name, got %#v", updated)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestUpdateRoleNotFound(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectQuery("UPDATE roles SET").WillReturnError(sql.ErrNoRows)

	_, err := store.UpdateRole(context.Background(), "ghost", RoleUpdate{Name: "X"})
	if err != sql.ErrNoRows {
		t.Fatalf("expected sql.ErrNoRows, got %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestUpdateRoleNoFieldsIsRejected(t *testing.T) {
	store, mock := newRBACTestStore(t)

	if _, err := store.UpdateRole(context.Background(), "admin", RoleUpdate{}); err == nil {
		t.Fatal("expected error when no fields are provided")
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations (no query should run): %v", err)
	}
}

func TestDeleteRoleSuccess(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectExec("DELETE FROM roles WHERE id = \\$1").
		WithArgs("auditor").
		WillReturnResult(sqlmock.NewResult(0, 1))

	if err := store.DeleteRole(context.Background(), "Auditor"); err != nil {
		t.Fatalf("DeleteRole: %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestDeleteRoleNotFound(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectExec("DELETE FROM roles WHERE id = \\$1").
		WithArgs("ghost").
		WillReturnResult(sqlmock.NewResult(0, 0))

	if err := store.DeleteRole(context.Background(), "ghost"); err != sql.ErrNoRows {
		t.Fatalf("expected sql.ErrNoRows, got %v", err)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

// TestAssignUserRolesValidatesAgainstDBCatalog proves role validation now
// queries the roles table instead of the old hardcoded roleCatalog map.
func TestAssignUserRolesValidatesAgainstDBCatalog(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectQuery("SELECT id, name, description, permissions FROM roles WHERE id = \\$1").
		WithArgs("operator").
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "description", "permissions"}).
			AddRow("operator", "Operator", "", []byte(`["read"]`)))
	mock.ExpectExec("UPDATE users SET role = \\$1").
		WithArgs("operator", "user-1").
		WillReturnResult(sqlmock.NewResult(0, 1))

	roles, err := store.AssignUserRoles(context.Background(), "user-1", []string{"operator"})
	if err != nil {
		t.Fatalf("AssignUserRoles: %v", err)
	}
	if len(roles) != 1 || roles[0] != "operator" {
		t.Fatalf("unexpected roles: %#v", roles)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations: %v", err)
	}
}

func TestAssignUserRolesRejectsUnknownRole(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectQuery("SELECT id, name, description, permissions FROM roles WHERE id = \\$1").
		WithArgs("ghost-role").
		WillReturnError(sql.ErrNoRows)

	_, err := store.AssignUserRoles(context.Background(), "user-1", []string{"ghost-role"})
	if err == nil {
		t.Fatal("expected error for unsupported role")
	}
	// Must fail closed: no UPDATE should be attempted for an unknown role.
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations (UPDATE must not run): %v", err)
	}
}

// TestGetUserPermissionsResolvesSingleRoleOnce proves GetUserPermissions
// fetches the role definition exactly once (users.role lookup + one roles
// lookup), not once per call site, and returns that role's permissions
// sorted.
func TestGetUserPermissionsResolvesSingleRoleOnce(t *testing.T) {
	store, mock := newRBACTestStore(t)

	mock.ExpectQuery("SELECT role FROM users WHERE id = \\$1").
		WithArgs("user-1").
		WillReturnRows(sqlmock.NewRows([]string{"role"}).AddRow("operator"))
	mock.ExpectQuery("SELECT id, name, description, permissions FROM roles WHERE id = \\$1").
		WithArgs("operator").
		WillReturnRows(sqlmock.NewRows([]string{"id", "name", "description", "permissions"}).
			AddRow("operator", "Operator", "", []byte(`["vm.manage","read"]`)))

	permissions, err := store.GetUserPermissions(context.Background(), "user-1")
	if err != nil {
		t.Fatalf("GetUserPermissions: %v", err)
	}
	if len(permissions) != 2 || permissions[0] != "read" || permissions[1] != "vm.manage" {
		t.Fatalf("expected sorted [read vm.manage], got %#v", permissions)
	}
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("unmet expectations (exactly 2 queries expected): %v", err)
	}
}

// mockPQError simulates the error text Postgres/lib-pq produces for a unique
// constraint violation, without importing lib/pq's error type.
type mockPQError struct{ msg string }

func (e *mockPQError) Error() string { return e.msg }
