package auth

import (
	"fmt"
	"sync"
	"testing"
	"time"
)

// TestInMemoryTokenRevocationConcurrentAccess hammers every
// InMemoryTokenRevocation method concurrently on overlapping JTIs. Run with
// -race: it must be clean (closes the unsynchronized delete-on-read in
// IsRevoked racing with RevokeTokenWithReason's map write).
func TestInMemoryTokenRevocationConcurrentAccess(t *testing.T) {
	m := NewInMemoryTokenRevocation()

	const goroutines = 32
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(g int) {
			defer wg.Done()
			jti := fmt.Sprintf("jti-%d", g%8)
			_ = m.RevokeTokenWithReason(jti, time.Now().Add(time.Hour), "race_test")
			_, _ = m.IsRevoked(jti)
			_, _ = m.GetRevocationReason(jti)
			_ = m.CleanupExpired()
		}(g)
	}
	wg.Wait()
}

// TestInMemoryTokenRevocationSweepsExpiredEntries proves an already-expired
// entry is evicted (lazily, on a subsequent write), bounding the map's
// growth without a background goroutine.
func TestInMemoryTokenRevocationSweepsExpiredEntries(t *testing.T) {
	m := NewInMemoryTokenRevocation()

	if err := m.RevokeTokenWithReason("expired-jti", time.Now().Add(-time.Hour), "already_past"); err != nil {
		t.Fatalf("RevokeTokenWithReason (expired): %v", err)
	}

	// Trigger the lazy sweep by revoking a second, fresh JTI.
	if err := m.RevokeTokenWithReason("fresh-jti", time.Now().Add(time.Hour), "fresh"); err != nil {
		t.Fatalf("RevokeTokenWithReason (fresh): %v", err)
	}

	if _, err := m.GetRevocationReason("expired-jti"); err == nil {
		t.Fatalf("expired-jti still present after a sweep-triggering write")
	}
}
