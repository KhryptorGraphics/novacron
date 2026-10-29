package main

import (
	"database/sql"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// TestRegisterMigratedDestStampsReceivingNode proves the destination stamps
// vms.node_id with ITS OWN id, not the target the source claimed.
//
// node_id is load-bearing: the boot reconcile only repairs rows this node
// owns (reconcileVMState scopes its SELECT to node_id = selfNodeID()), and it
// flips a 'running' row with no live qemu to 'stopped'. So a row stamped with
// some other node's id is skipped by its own node's reconcile - the guest
// stays 'running' forever after a power loss, which is the mirror image of
// the peer-row corruption the scoping fixed.
//
// The mismatch is reachable: a migration request carries both a target_node
// and a target_addr, and the addr can point at a node other than target_node.
// The SOURCE's claim is therefore not authority over where the guest actually
// ran; the receiving node is.
func TestRegisterMigratedDestStampsReceivingNode(t *testing.T) {
	admin, err := sql.Open("postgres", pgAdminDSN())
	if err != nil {
		t.Skipf("skip: cannot open postgres admin connection: %v", err)
	}
	if err := admin.Ping(); err != nil {
		admin.Close()
		t.Skipf("skip: postgres not reachable: %v", err)
	}

	const dbName = "novacron_migrated_node_test"
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

	// This node is node-c. The SOURCE claims the guest landed on "node-b".
	t.Setenv("NOVACRON_NODE_ID", "node-c")

	const vmID = "33333333-3333-3333-3333-333333333333"
	manager := newStubVMManager(t)
	defer manager.Stop()

	registerMigratedDest(db, manager, vmID, vm.VMConfig{
		ID: vmID, Name: "migrated-in-vm", Type: vm.VMTypeKVM, MemoryMB: 128, VCPUs: 1,
	}, "node-b", "")

	var nodeID sql.NullString
	if err := db.QueryRow(`SELECT node_id FROM vms WHERE id = $1`, vmID).Scan(&nodeID); err != nil {
		t.Fatalf("query migrated row: %v", err)
	}
	if nodeID.String != "node-c" {
		t.Fatalf("node_id = %q, want %q — the row must name the node the guest ACTUALLY runs on, or this node's boot reconcile will skip it",
			nodeID.String, "node-c")
	}
}
