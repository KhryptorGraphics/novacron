package auth

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestUserMemoryStoreConcurrentAccess reproduces auth.go Login's exact
// Get-then-mutate-then-Update pattern (and RecordLogin's direct-stamp
// pattern) at the store level, concurrently with every other mutator. Run
// with -race: it must be clean.
func TestUserMemoryStoreConcurrentAccess(t *testing.T) {
	store := NewUserMemoryStore()

	const goroutines = 32
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			id := fmt.Sprintf("user-%d", g%8)
			user := NewUser(id, fmt.Sprintf("%s@example.com", id), "default")
			user.ID = id
			_ = store.Create(user, "Password@123")

			if got, err := store.Get(id); err == nil {
				got.LastLogin = time.Now()
				_ = store.Update(got)
			}
			_ = store.RecordLogin(id, time.Now())
			_, _ = store.GetByUsername(id)
			_, _ = store.GetByEmail(fmt.Sprintf("%s@example.com", id))
			_, _ = store.List(nil)
			_ = store.AddRole(id, "user")
			_ = store.RemoveRole(id, "user")
		}(g)
	}
	wg.Wait()
}

// TestUserMemoryStoreGetReturnsIsolatedCopy proves mutating a Get() result
// cannot affect the store's internal state.
func TestUserMemoryStoreGetReturnsIsolatedCopy(t *testing.T) {
	store := NewUserMemoryStore()
	user := NewUser("isolated-user", "isolated@example.com", "default")
	if err := store.Create(user, "Password@123"); err != nil {
		t.Fatalf("Create: %v", err)
	}

	got, err := store.Get("isolated-user")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	got.Status = UserStatusLocked
	got.Metadata["mutated"] = true

	again, err := store.Get("isolated-user")
	if err != nil {
		t.Fatalf("Get (again): %v", err)
	}
	if again.Status == UserStatusLocked {
		t.Fatalf("store's Status was mutated through a Get() result")
	}
	if _, ok := again.Metadata["mutated"]; ok {
		t.Fatalf("store's Metadata was mutated through a Get() result")
	}
}
