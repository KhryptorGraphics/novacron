package handlers

import (
	"context"
	"database/sql"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	"github.com/golang-migrate/migrate/v4"
	_ "github.com/golang-migrate/migrate/v4/database/postgres"
	_ "github.com/golang-migrate/migrate/v4/source/file"
	_ "github.com/lib/pq"
)

// migrationsSourceURL locates database/migrations relative to this test
// file's own location (not the `go test` working directory), so the test
// works regardless of which directory `go test` is invoked from.
func migrationsSourceURL(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("failed to resolve test file path")
	}
	// backend/api/security/rbac_migration_test.go -> repo root
	root := filepath.Join(filepath.Dir(thisFile), "..", "..", "..")
	migrationsDir, err := filepath.Abs(filepath.Join(root, "database", "migrations"))
	if err != nil {
		t.Fatalf("resolve absolute migrations path: %v", err)
	}
	if _, err := os.Stat(migrationsDir); err != nil {
		t.Fatalf("database/migrations not found at %s: %v", migrationsDir, err)
	}
	return "file://" + migrationsDir
}

// testDBBaseURL returns the Postgres server to run this test against. Set
// DB_TEST_URL to point at a reachable server (matches the Makefile's
// db-test-setup convention); otherwise falls back to the same local default
// the Makefile uses. The test skips outright if nothing is reachable there.
func testDBBaseURL() string {
	if raw := os.Getenv("DB_TEST_URL"); raw != "" {
		return raw
	}
	return "postgres://postgres:postgres@localhost:5432/novacron_test?sslmode=disable"
}

// TestRBACMigrationSeedsDefaultRoles proves the 000004_add_rbac_roles
// migration (a) applies cleanly on top of the full migration chain, (b)
// seeds exactly the 6 roles / 11 permissions the old hardcoded
// roleCatalog/permissionCatalog vars shipped (so behavior is unchanged after
// migrating), and (c) its down migration cleanly removes both tables. Runs
// against a real, disposable Postgres database created just for this test —
// skips if no server is reachable.
func TestRBACMigrationSeedsDefaultRoles(t *testing.T) {
	base, err := url.Parse(testDBBaseURL())
	if err != nil {
		t.Skipf("skipping: invalid test DB URL: %v", err)
	}

	adminURL := *base
	adminURL.Path = "/postgres"
	adminDB, err := sql.Open("postgres", adminURL.String())
	if err != nil {
		t.Skipf("skipping: cannot open admin connection: %v", err)
	}
	defer adminDB.Close()
	if err := adminDB.Ping(); err != nil {
		t.Skipf("skipping: no reachable Postgres at %s: %v", adminURL.Host, err)
	}

	scratchName := fmt.Sprintf("novacron_rbac_migrate_test_%d", time.Now().UnixNano())
	if _, err := adminDB.Exec("CREATE DATABASE " + scratchName); err != nil {
		t.Fatalf("create scratch database %s: %v", scratchName, err)
	}
	defer func() {
		// Best-effort teardown of the uniquely-named throwaway DB: kick any
		// lingering connections first so DROP DATABASE doesn't fail.
		_, _ = adminDB.Exec(fmt.Sprintf(
			"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '%s' AND pid <> pg_backend_pid()",
			scratchName))
		_, _ = adminDB.Exec("DROP DATABASE IF EXISTS " + scratchName)
	}()

	scratchURL := *base
	scratchURL.Path = "/" + scratchName

	m, err := migrate.New(migrationsSourceURL(t), scratchURL.String())
	if err != nil {
		t.Fatalf("migrate.New: %v", err)
	}
	defer m.Close()

	if err := m.Up(); err != nil {
		t.Fatalf("migrate up: %v", err)
	}

	scratchDB, err := sql.Open("postgres", scratchURL.String())
	if err != nil {
		t.Fatalf("open scratch db: %v", err)
	}
	defer scratchDB.Close()

	store := NewPostgresRBACStore(scratchDB)
	ctx := context.Background()

	roles, err := store.ListRoles(ctx)
	if err != nil {
		t.Fatalf("ListRoles after fresh migrate: %v", err)
	}
	wantRoles := []string{"admin", "operator", "readonly", "super-admin", "user", "viewer"}
	if len(roles) != len(wantRoles) {
		t.Fatalf("expected %d default roles after fresh migrate, got %d: %#v", len(wantRoles), len(roles), roles)
	}
	for i, want := range wantRoles {
		if roles[i].ID != want {
			t.Fatalf("role[%d]: expected id %q, got %q (full: %#v)", i, want, roles[i].ID, roles)
		}
	}

	// super-admin must retain its wildcard permission and admin its full set,
	// matching the old hardcoded roleCatalog exactly.
	byID := make(map[string]RoleDefinition, len(roles))
	for _, r := range roles {
		byID[r.ID] = r
	}
	if perms := byID["super-admin"].Permissions; len(perms) != 1 || perms[0] != "*" {
		t.Fatalf("super-admin: expected [\"*\"], got %#v", perms)
	}
	if perms := byID["admin"].Permissions; len(perms) != 11 {
		t.Fatalf("admin: expected 11 permissions, got %d: %#v", len(perms), perms)
	}

	permissions, err := store.ListPermissions(ctx)
	if err != nil {
		t.Fatalf("ListPermissions after fresh migrate: %v", err)
	}
	if len(permissions) != 11 {
		t.Fatalf("expected 11 default permissions after fresh migrate, got %d: %#v", len(permissions), permissions)
	}

	// Down migration must cleanly remove the RBAC tables.
	if err := m.Steps(-1); err != nil {
		t.Fatalf("migrate down: %v", err)
	}
	if _, err := store.ListRoles(ctx); err == nil {
		t.Fatal("expected ListRoles to fail after the rbac migration was rolled back (roles table should be gone)")
	}
}
