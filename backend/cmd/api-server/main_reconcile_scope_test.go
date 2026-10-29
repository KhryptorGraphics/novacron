package main

import (
	"database/sql"
	"os"
	"path/filepath"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/golang-jwt/jwt/v5"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// TestReconcileVMStateOnlyTouchesOwnRows proves the boot reconcile never
// rewrites a row that belongs to ANOTHER node.
//
// The liveness check is inherently local: it looks for vmBase/<id>/qemu.pid,
// which only exists on the node that actually runs the guest. Before the fix
// the query was "SELECT id, state FROM vms" with no owner filter, so under a
// shared database (the topology deploy/README.md §1 describes as "at least
// one per fabric") a restart on node A read every peer row as "not alive
// here" and flipped it from 'running' to 'stopped' -- corrupting a live
// guest's state on a node that was never touched.
//
// This test drives the reconcile against a real migrated postgres database
// with two rows: one owned by this node and one owned by a peer. Only the
// owned row may be reconciled.
func TestReconcileVMStateOnlyTouchesOwnRows(t *testing.T) {
	admin, err := sql.Open("postgres", pgAdminDSN())
	if err != nil {
		t.Skipf("skip: cannot open postgres admin connection: %v", err)
	}
	if err := admin.Ping(); err != nil {
		admin.Close()
		t.Skipf("skip: postgres not reachable: %v", err)
	}

	const dbName = "novacron_reconcile_scope_test"
	_, _ = admin.Exec(`DROP DATABASE IF EXISTS "` + dbName + `"`)
	if _, err := admin.Exec(`CREATE DATABASE "` + dbName + `"`); err != nil {
		t.Fatalf("create database %s: %v", dbName, err)
	}
	applyMigrations(t, dbDSN(dbName), dbName)

	db, err := sql.Open("postgres", dbDSN(dbName))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() {
		db.Close()
		_, _ = admin.Exec(`DROP DATABASE IF EXISTS "` + dbName + `"`)
		admin.Close()
	})

	const (
		ownVM = "11111111-1111-1111-1111-111111111111"
		peerVM = "22222222-2222-2222-2222-222222222222"
	)
	self := selfNodeID()
	if self == "" {
		t.Fatal("selfNodeID() is empty")
	}

	if _, err := db.Exec(
		`INSERT INTO vms (id, name, state, cpu_cores, memory_mb, disk_gb, node_id) VALUES
		 ($1, 'own-vm', 'running', 1, 128, 1, $3),
		 ($2, 'peer-vm', 'running', 1, 128, 1, 'some-other-node')`,
		ownVM, peerVM, self); err != nil {
		t.Fatalf("seed vms rows: %v", err)
	}

	// No pidfile exists for either VM, so BOTH rows read "not alive" to the
	// liveness check. Only the owned row may be updated.
	vmBase := t.TempDir()
	reconcileVMState(db, vmBase, nil)

	var ownState, peerState string
	if err := db.QueryRow(`SELECT state FROM vms WHERE id = $1`, ownVM).Scan(&ownState); err != nil {
		t.Fatalf("read own row: %v", err)
	}
	if err := db.QueryRow(`SELECT state FROM vms WHERE id = $1`, peerVM).Scan(&peerState); err != nil {
		t.Fatalf("read peer row: %v", err)
	}

	if ownState != "stopped" {
		t.Fatalf("owned row with no live qemu should reconcile to 'stopped', got %q", ownState)
	}
	if peerState != "running" {
		t.Fatalf("peer-owned row was rewritten to %q — the reconcile must only touch rows this node owns", peerState)
	}
}

// TestReconcileVMStateQueryIsScopedToSelf pins the query itself, so a future
// refactor cannot drop the owner filter without a test failing. The owned-row
// behaviour is covered by the real-postgres test above; this one is the cheap
// guard that the filter is still there.
func TestReconcileVMStateQueryIsScopedToSelf(t *testing.T) {
	db, mock, err := sqlmock.New(sqlmock.QueryMatcherOption(sqlmock.QueryMatcherRegexp))
	if err != nil {
		t.Fatalf("sqlmock: %v", err)
	}
	defer db.Close()

	mock.ExpectQuery(`SELECT id, state FROM vms WHERE node_id = \$1 OR node_id IS NULL OR node_id = ''`).
		WithArgs(selfNodeID()).
		WillReturnRows(sqlmock.NewRows([]string{"id", "state"}))
	mock.ExpectationsWereMet()

	reconcileVMState(db, t.TempDir(), nil)
	if err := mock.ExpectationsWereMet(); err != nil {
		t.Fatalf("the reconcile query must be scoped to this node: %v", err)
	}
}

// Keep the imports used by the sibling test file in this package honest.
var _ = os.Getenv
var _ = filepath.Join
var _ = jwt.NewWithClaims
var _ = auth.NewSimpleAuthManager
