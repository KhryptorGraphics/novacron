package auth

import (
	"fmt"
	"sync"
	"testing"
)

// TestRoleMemoryStoreConcurrentAccess hammers every RoleMemoryStore method
// concurrently on overlapping role IDs, plus concurrent reads of the shared
// "admin" system role. Run with -race: it must be clean.
func TestRoleMemoryStoreConcurrentAccess(t *testing.T) {
	store := NewRoleMemoryStore()

	const goroutines = 32
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			id := fmt.Sprintf("role-%d", g%8)
			_ = store.Create(NewRole(id, id, "race test role", ""))
			_, _ = store.Get(id)
			_, _ = store.Get("admin")
			_, _ = store.List(nil)
			_ = store.Update(&Role{ID: id, Name: id, Permissions: []Permission{{Resource: "vm", Action: "read", Effect: "allow"}}})
			_ = store.AddPermission(id, Permission{Resource: "vm", Action: "write", Effect: "allow"})
			_ = store.RemovePermission(id, "vm", "write")
			_, _ = store.HasPermission(id, "vm", "read")
			_, _ = store.HasPermission("admin", "*", "*")
			_ = store.Delete(id)
		}(g)
	}
	wg.Wait()
}

// TestRoleMemoryStoreInstancesDoNotShareSystemRoles proves each store
// instance owns an independent copy of the package-level SystemRoles roles,
// closing the shared-*Role-pointer bug where every NewRoleMemoryStore()'s
// "admin" role aliased the same *Role.
func TestRoleMemoryStoreInstancesDoNotShareSystemRoles(t *testing.T) {
	a := NewRoleMemoryStore()
	b := NewRoleMemoryStore()

	got, err := a.Get("admin")
	if err != nil {
		t.Fatalf("Get admin from a: %v", err)
	}
	got.Permissions[0].Effect = "deny"

	bAdmin, err := b.Get("admin")
	if err != nil {
		t.Fatalf("Get admin from b: %v", err)
	}
	if bAdmin.Permissions[0].Effect == "deny" {
		t.Fatalf("store b's admin role was mutated through store a's Get() result")
	}
}

// TestRoleMemoryStoreAddPermissionUpdatePersists proves that calling
// AddPermission a second time with the same resource/action but a different
// Effect actually persists onto the stored role, closing the pre-existing
// no-op bug where the "update existing permission" branch ranged over a
// copy of role.Permissions (mutating p.Effect had no effect on the stored
// slice element) instead of indexing into it.
func TestRoleMemoryStoreAddPermissionUpdatePersists(t *testing.T) {
	store := NewRoleMemoryStore()
	if err := store.Create(NewRole("deny-role", "Deny Role", "", "")); err != nil {
		t.Fatalf("Create: %v", err)
	}

	if err := store.AddPermission("deny-role", Permission{Resource: "vm", Action: "write", Effect: "allow"}); err != nil {
		t.Fatalf("AddPermission (allow): %v", err)
	}
	allowed, err := store.HasPermission("deny-role", "vm", "write")
	if err != nil {
		t.Fatalf("HasPermission (allow): %v", err)
	}
	if !allowed {
		t.Fatalf("HasPermission = false, want true after granting an allow permission")
	}

	// Update the same resource/action to deny.
	if err := store.AddPermission("deny-role", Permission{Resource: "vm", Action: "write", Effect: "deny"}); err != nil {
		t.Fatalf("AddPermission (deny): %v", err)
	}
	allowed, err = store.HasPermission("deny-role", "vm", "write")
	if err != nil {
		t.Fatalf("HasPermission (deny): %v", err)
	}
	if allowed {
		t.Fatalf("HasPermission = true, want false after updating the permission to deny")
	}

	role, err := store.Get("deny-role")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(role.Permissions) != 1 {
		t.Fatalf("Permissions = %v, want exactly one entry (update, not append)", role.Permissions)
	}
}
