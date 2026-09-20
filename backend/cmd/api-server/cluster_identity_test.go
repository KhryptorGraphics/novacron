package main

import (
	"context"
	"database/sql"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"testing"

	"github.com/google/uuid"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	_ "github.com/lib/pq"
)

// migrationsDir locates database/migrations relative to this test file, so
// the test works regardless of the invoker's working directory.
func migrationsDir(t *testing.T) string {
	t.Helper()
	_, thisFile, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	// backend/cmd/api-server/<this file> -> repo root is three levels up.
	dir := filepath.Join(filepath.Dir(thisFile), "..", "..", "..", "database", "migrations")
	if _, err := os.Stat(dir); err != nil {
		t.Skipf("skip: migrations dir not found at %s: %v", dir, err)
	}
	return dir
}

// applyMigrations runs every *.up.sql file in migrationsDir, in filename
// order, against db. Uses psql (not the go pq driver) so multi-statement
// files execute exactly as they would under the real migrate tool.
func applyMigrations(t *testing.T, dsn, dbName string) {
	t.Helper()
	dir := migrationsDir(t)
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read migrations dir: %v", err)
	}
	var ups []string
	for _, e := range entries {
		name := e.Name()
		if len(name) > 7 && name[len(name)-7:] == ".up.sql" {
			ups = append(ups, name)
		}
	}
	sort.Strings(ups)
	for _, name := range ups {
		cmd := exec.Command("psql", dsn, "-v", "ON_ERROR_STOP=1", "-f", filepath.Join(dir, name))
		cmd.Env = append(os.Environ(), "PGDATABASE="+dbName)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("apply %s: %v: %s", name, err, out)
		}
	}
}

// pgHost/pgAdminDSN mirror the harness's default local postgres connection.
func pgAdminDSN() string {
	host := envOr("FABRIC_PGHOST", "127.0.0.1")
	port := envOr("FABRIC_PGPORT", "5432")
	user := envOr("FABRIC_PGUSER", "postgres")
	pass := envOr("FABRIC_PGPASSWORD", "postgres")
	return "postgres://" + user + ":" + pass + "@" + host + ":" + port + "/postgres?sslmode=disable"
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func dbDSN(name string) string {
	host := envOr("FABRIC_PGHOST", "127.0.0.1")
	port := envOr("FABRIC_PGPORT", "5432")
	user := envOr("FABRIC_PGUSER", "postgres")
	pass := envOr("FABRIC_PGPASSWORD", "postgres")
	return "postgres://" + user + ":" + pass + "@" + host + ":" + port + "/" + name + "?sslmode=disable"
}

// TestClusterIdentityModelCrossNodeForeignOwner proves novacron-ok7's fix end
// to end against two REAL, independently-migrated postgres databases (not
// sqlmock, since the defect and its fix are exactly about real schema/FK
// behaviour): a VM created locally on "node A" records a real, queryable
// node_id and a real local owner_id; the SAME VM registered as migrated-in on
// "node B" (whose users table does not have that owner) records node B's own
// node_id, a NULL owner_id (the local FK correctly refuses a foreign UUID),
// and the original owner preserved in requested_owner_id -- all as typed
// columns, no metadata JSON parsing required.
func TestClusterIdentityModelCrossNodeForeignOwner(t *testing.T) {
	admin, err := sql.Open("postgres", pgAdminDSN())
	if err != nil {
		t.Skipf("skip: cannot open postgres admin connection: %v", err)
	}
	if err := admin.Ping(); err != nil {
		admin.Close()
		t.Skipf("skip: postgres not reachable: %v", err)
	}

	const dbA, dbB = "novacron_identity_test_a", "novacron_identity_test_b"
	for _, name := range []string{dbA, dbB} {
		_, _ = admin.Exec(`DROP DATABASE IF EXISTS "` + name + `"`)
		if _, err := admin.Exec(`CREATE DATABASE "` + name + `"`); err != nil {
			t.Fatalf("create database %s: %v", name, err)
		}
	}

	applyMigrations(t, dbDSN(dbA), dbA)
	applyMigrations(t, dbDSN(dbB), dbB)

	sqlA, err := sql.Open("postgres", dbDSN(dbA))
	if err != nil {
		t.Fatalf("open db A: %v", err)
	}
	sqlB, err := sql.Open("postgres", dbDSN(dbB))
	if err != nil {
		t.Fatalf("open db B: %v", err)
	}
	// Cleanup order matters: postgres refuses DROP DATABASE while a
	// connection is open against it, so sqlA/sqlB must close before the
	// DROP, and admin (which issues the DROP) must close last.
	t.Cleanup(func() {
		sqlA.Close()
		sqlB.Close()
		for _, name := range []string{dbA, dbB} {
			_, _ = admin.Exec(`DROP DATABASE IF EXISTS "` + name + `"`)
		}
		admin.Close()
	})

	// A real user that exists ONLY on node A -- the whole point of the test.
	ownerID := uuid.NewString()
	if _, err := sqlA.Exec(`INSERT INTO users (id, email, username, password_hash) VALUES ($1, $2, $3, 'x')`,
		ownerID, "owner@node-a.invalid", "owner-node-a"); err != nil {
		t.Fatalf("seed owner on node A: %v", err)
	}

	ctx := context.Background()
	vmID := uuid.NewString()

	// --- create on node A ---------------------------------------------------
	t.Setenv("NOVACRON_NODE_ID", "node-a-test")
	managerA := newStubVMManager(t)
	defer managerA.Stop()
	gotVMID, _, err := createVMLocal(ctx, sqlA, managerA, clusterCreateSpec{
		Name: "identity-test-vm", MemoryMB: 128, VCPUs: 1, OwnerID: ownerID,
	})
	if err != nil {
		t.Fatalf("createVMLocal on node A: %v", err)
	}
	vmID = gotVMID

	var nodeIDA, ownerIDA sql.NullString
	var requestedOwnerA sql.NullString
	if err := sqlA.QueryRow(`SELECT node_id, owner_id::text, requested_owner_id::text FROM vms WHERE id = $1`, vmID).
		Scan(&nodeIDA, &ownerIDA, &requestedOwnerA); err != nil {
		t.Fatalf("query node A row: %v", err)
	}
	if nodeIDA.String != "node-a-test" {
		t.Fatalf("node A: expected node_id=node-a-test, got %q", nodeIDA.String)
	}
	if ownerIDA.String != ownerID {
		t.Fatalf("node A: expected owner_id=%s (local owner resolves), got %q", ownerID, ownerIDA.String)
	}
	if requestedOwnerA.Valid {
		t.Fatalf("node A: expected requested_owner_id NULL (owner resolved locally, no divergence), got %q", requestedOwnerA.String)
	}
	t.Logf("node A row: node_id=%s owner_id=%s requested_owner_id=NULL (owner resolved locally)", nodeIDA.String, ownerIDA.String)

	// --- the VM "migrates" to node B, whose users table has no such owner ---
	t.Setenv("NOVACRON_NODE_ID", "node-b-test")
	managerB := newStubVMManager(t)
	defer managerB.Stop()
	registerMigratedDest(sqlB, managerB, vmID, core_vm.VMConfig{
		ID: vmID, Name: "identity-test-vm", Type: core_vm.VMTypeKVM, MemoryMB: 128, VCPUs: 1, OwnerID: ownerID,
	}, "node-b-test")

	var nodeIDB, ownerIDB sql.NullString
	var requestedOwnerB sql.NullString
	if err := sqlB.QueryRow(`SELECT node_id, owner_id::text, requested_owner_id::text FROM vms WHERE id = $1`, vmID).
		Scan(&nodeIDB, &ownerIDB, &requestedOwnerB); err != nil {
		t.Fatalf("query node B row: %v", err)
	}
	if nodeIDB.String != "node-b-test" {
		t.Fatalf("node B: expected node_id=node-b-test, got %q", nodeIDB.String)
	}
	if ownerIDB.Valid {
		t.Fatalf("node B: expected owner_id NULL (foreign owner, local FK correctly refuses it), got %q", ownerIDB.String)
	}
	if requestedOwnerB.String != ownerID {
		t.Fatalf("node B: expected requested_owner_id=%s (the foreign owner preserved for audit), got %q", ownerID, requestedOwnerB.String)
	}
	t.Logf("node B row: node_id=%s owner_id=NULL requested_owner_id=%s (foreign owner preserved, not silently invented)", nodeIDB.String, requestedOwnerB.String)
}
