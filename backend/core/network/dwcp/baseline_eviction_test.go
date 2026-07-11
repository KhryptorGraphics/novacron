package dwcp

import (
	"crypto/rand"
	"fmt"
	"testing"
)

// TestBaselineEviction_BoundedByMaxMemoryUsage proves that adapter.vmBaselines
// can no longer grow without bound as distinct vmIDs churn through migration
// (novacron-y45).
//
// Before the fix, storeMemoryBaseline/storeDiskBaselines stored a full byte
// copy of every migrated VM's memory (and disk blocks) keyed by unique vmID,
// with no delete() anywhere and no code path that ever reads a stored
// baseline's content back — a pure write-only memory leak on any deployment
// with VM churn. The fix bounds total retained baseline bytes to
// MigrationAdapterConfig.MaxMemoryUsage via LRU eviction.
//
// This test measures retained state through the pre-existing VMBaseline fields
// (MemoryBaseline/DiskBaselines) rather than the fix's internal byte counter,
// so it compiles and runs against the pre-fix code too and discriminates
// purely on behavior: pre-fix it retains all numVMs entries (~12.8 MiB, far
// over the 1 MiB cap); post-fix it retains at most capBytes/entrySize.
func TestBaselineEviction_BoundedByMaxMemoryUsage(t *testing.T) {
	const (
		entrySize = 64 * 1024          // 64 KiB per VM memory baseline
		capBytes  = 16 * entrySize     // 1 MiB cap => at most 16 entries retained
		numVMs    = 200                // 200 * 64 KiB = 12.8 MiB, far over cap
	)

	adapter, err := NewMigrationAdapter(MigrationAdapterConfig{
		EnableDWCP:     false,
		MaxMemoryUsage: capBytes,
	})
	if err != nil {
		t.Fatalf("NewMigrationAdapter failed: %v", err)
	}
	defer adapter.Close()

	seed := make([]byte, entrySize)
	if _, err := rand.Read(seed); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}

	for i := range numVMs {
		// Distinct backing slice per vmID so retained bytes are genuinely
		// additive (not aliased).
		buf := make([]byte, entrySize)
		copy(buf, seed)
		adapter.storeMemoryBaseline(fmt.Sprintf("vm-%d", i), buf)
	}

	adapter.mu.RLock()
	count := len(adapter.vmBaselines)
	var retained int64
	for _, b := range adapter.vmBaselines {
		b.mu.RLock()
		retained += int64(len(b.MemoryBaseline))
		for _, blk := range b.DiskBaselines {
			retained += int64(len(blk))
		}
		b.mu.RUnlock()
	}
	adapter.mu.RUnlock()

	if retained > capBytes {
		t.Errorf("retained baseline bytes = %d, exceeds MaxMemoryUsage cap = %d (unbounded growth: baselines not evicted)", retained, capBytes)
	}
	if count > capBytes/entrySize {
		t.Errorf("retained baseline count = %d, exceeds capBytes/entrySize = %d", count, capBytes/entrySize)
	}
	// Pre-fix keeps every entry; post-fix must have evicted the overflow.
	if count >= numVMs {
		t.Errorf("retained baseline count = %d; expected far fewer than %d after eviction (no eviction => memory leak)", count, numVMs)
	}
	// Sanity: the cap must still retain the most recent entries, not drop all.
	if count == 0 {
		t.Error("all baselines evicted; expected the cap to retain the most recent entries")
	}
}
