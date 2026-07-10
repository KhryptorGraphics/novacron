package dwcp_test

import (
	"context"
	"crypto/rand"
	"fmt"
	"testing"
	"time"
)

// generateMixedVMData builds a realistic VM memory/disk data mix: 1/3
// zero pages (empty), 1/3 repeating byte pattern (compressible), 1/3
// random (incompressible) — in 4KB blocks, matching real VM memory/disk
// composition. Used for every subtest here so standard vs dwcp
// comparisons share identical payloads and dwcp compression ratios are
// representative rather than artifacts of pure-random test data.
func generateMixedVMData(size int) []byte {
	data := make([]byte, size)
	for i := 0; i < size; i += 4096 {
		end := i + 4096
		if end > size {
			end = size
		}
		switch (i / 4096) % 3 {
		case 0:
			// Zero page — leave as-is.
		case 1:
			for j := i; j < end; j++ {
				data[j] = byte(j % 256)
			}
		case 2:
			rand.Read(data[i:end])
		}
	}
	return data
}

// BenchmarkMigrationAdapterEndToEnd measures real, production-path
// MigrationAdapter.MigrateVMMemory/MigrateVMDisk time — send, real
// network transfer, receive, and (for the DWCP subtests) decompress —
// via OnMemoryReceived/OnDiskReceived synchronization, not just how long
// MigrateVMMemory takes to return (which only covers the sender's own
// Transfer call, not receive-side completion).
//
// This is the FIRST benchmark of novacron-38p's Phase 0 effort that
// exercises the actual, shipped MigrationAdapter (novacron-lce) rather
// than composing HDE+AMST primitives directly
// (BenchmarkMigrationPipelinePrimitives, v3/tests) or a plain drain sink
// (the other v3/tests benchmarks). Two consequences that make its
// numbers non-comparable to those primitives benchmarks, both load-
// bearing for how novacron-38p's Go/No-Go should read these numbers:
//
//  1. Config: MigrationAdapter forces EnableDelta/EnableDictionary/
//     EnableQuantization off and single-stream AMST, unconditionally
//     (NewMigrationAdapter) — the production-safe path, not the
//     multi-stream/delta-enabled ceiling BenchmarkMigrationPipelinePrimitives
//     measures. This benchmark's compressed-path numbers are what
//     MigrationAdapter can ACTUALLY deliver today, not a best case.
//  2. Tier: selectCompressionTier (migration_adapter.go) picks a
//     compression tier from AMST's latency_ms metric, which defaults to
//     0 unless something calls AMST.UpdateMetrics — nothing in
//     createConnection does. On loopback this means every subtest here
//     compresses at CompressionLocal (<10ms bucket), never Regional or
//     Global, regardless of payload — so even this "real adapter"
//     number is not WAN-representative on compression tier, only on
//     wire-protocol/framing correctness. A real WAN or latency-emulated
//     (tc/netem) path is still required before novacron-38p's Go/No-Go
//     can treat any of these throughput numbers as representative — see
//     TestSelectCompressionTier_LatencyThresholds for direct coverage of
//     the tier-selection logic itself, independent of this gap.
//
// Loopback TCP is also memcpy/syscall-bound, not WAN-bound, same caveat
// as every other benchmark in this session's work.
func BenchmarkMigrationAdapterEndToEnd(b *testing.B) {
	vmSizes := []int{16 * 1024 * 1024, 64 * 1024 * 1024}

	for _, size := range vmSizes {
		for _, enableDWCP := range []bool{false, true} {
			name := "standard"
			if enableDWCP {
				name = "dwcp"
			}
			b.Run(fmt.Sprintf("vm_%dMB_%s", size/1024/1024, name), func(b *testing.B) {
				adapter, memCh, _ := newLoopbackMigrationAdapter(b, enableDWCP)
				data := generateMixedVMData(size)

				b.SetBytes(int64(size))
				b.ResetTimer()
				for i := range b.N {
					vmID := fmt.Sprintf("bench-e2e-mem-%s-%d", name, i)
					if err := adapter.MigrateVMMemory(context.Background(), vmID, data, "127.0.0.1", nil); err != nil {
						b.Fatal(err)
					}
					waitForMemory(b, memCh, 30*time.Second)
				}
			})
		}
	}
}

// BenchmarkMigrationAdapterEndToEndDisk is the disk-path analogue of
// BenchmarkMigrationAdapterEndToEnd — same scope, same caveats.
func BenchmarkMigrationAdapterEndToEndDisk(b *testing.B) {
	blockCount := 4
	blockSize := 4 * 1024 * 1024

	for _, enableDWCP := range []bool{false, true} {
		name := "standard"
		if enableDWCP {
			name = "dwcp"
		}
		b.Run(fmt.Sprintf("vm_%dMB_%s", blockCount*blockSize/1024/1024, name), func(b *testing.B) {
			adapter, _, diskCh := newLoopbackMigrationAdapter(b, enableDWCP)

			blocks := make(map[int][]byte, blockCount)
			for i := 0; i < blockCount; i++ {
				blocks[i] = generateMixedVMData(blockSize)
			}

			totalSize := int64(blockCount * blockSize)
			b.SetBytes(totalSize)
			b.ResetTimer()
			for i := range b.N {
				vmID := fmt.Sprintf("bench-e2e-disk-%s-%d", name, i)
				if err := adapter.MigrateVMDisk(context.Background(), vmID, blocks, "127.0.0.1", nil); err != nil {
					b.Fatal(err)
				}
				waitForDiskBlocks(b, diskCh, blockCount, 30*time.Second)
			}
		})
	}
}
