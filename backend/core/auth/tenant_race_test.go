package auth

import (
	"fmt"
	"sync"
	"testing"
)

// TestTenantMemoryStoreConcurrentAccess hammers every TenantMemoryStore method
// concurrently on overlapping tenant IDs. Run with -race: it must be clean.
func TestTenantMemoryStoreConcurrentAccess(t *testing.T) {
	store := NewTenantMemoryStore()

	const goroutines = 32
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			id := fmt.Sprintf("tenant-%d", g%8)
			_ = store.Create(NewTenant(id, id, "race test tenant"))
			_, _ = store.Get(id)
			_, _ = store.List(nil)
			_ = store.Update(&Tenant{ID: id, Name: id, Status: TenantStatusActive})
			_ = store.UpdateStatus(id, TenantStatusSuspended)
			_ = store.SetResourceQuota(id, "vm.count", int64(g))
			_, _ = store.GetResourceQuota(id, "vm.count")
			_, _ = store.GetResourceQuotas(id)
			_ = store.Delete(id)
		}(g)
	}
	wg.Wait()
}

// TestTenantMemoryStoreInstancesDoNotShareDefaultQuotas proves each store
// instance owns an independent copy of DefaultResourceQuotas, closing the
// global-mutable-state bug where every NewTenantMemoryStore()'s "default"
// tenant aliased the same package-level map.
func TestTenantMemoryStoreInstancesDoNotShareDefaultQuotas(t *testing.T) {
	a := NewTenantMemoryStore()
	b := NewTenantMemoryStore()

	if err := a.SetResourceQuota("default", "vm.count", 999); err != nil {
		t.Fatalf("SetResourceQuota on a: %v", err)
	}

	bQuota, err := b.GetResourceQuota("default", "vm.count")
	if err != nil {
		t.Fatalf("GetResourceQuota on b: %v", err)
	}
	if bQuota == 999 {
		t.Fatalf("store b's default quota was mutated by a write to store a: got %d", bQuota)
	}
	if want := DefaultResourceQuotas["vm.count"]; bQuota != want {
		t.Fatalf("store b's default quota = %d, want unmodified default %d", bQuota, want)
	}
}

// TestTenantMemoryStoreGetReturnsIsolatedCopy proves mutating a Get() result
// cannot affect the store's internal state.
func TestTenantMemoryStoreGetReturnsIsolatedCopy(t *testing.T) {
	store := NewTenantMemoryStore()

	got, err := store.Get("default")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	got.Name = "mutated"
	got.ResourceQuotas["vm.count"] = 12345

	again, err := store.Get("default")
	if err != nil {
		t.Fatalf("Get (again): %v", err)
	}
	if again.Name == "mutated" {
		t.Fatalf("store's Name was mutated through a Get() result: %q", again.Name)
	}
	if again.ResourceQuotas["vm.count"] == 12345 {
		t.Fatalf("store's ResourceQuotas was mutated through a Get() result: %d", again.ResourceQuotas["vm.count"])
	}
}
