package tests

import (
	"context"
	"crypto/rand"
	"fmt"
	"net"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// generateVMMemoryData builds a realistic VM memory mix: 1/3 zero pages
// (empty), 1/3 repeating code/text patterns, 1/3 random heap data —
// approximating real guest memory entropy instead of uniform random or
// all-zero data.
func generateVMMemoryData(size int) []byte {
	data := make([]byte, size)
	pageSize := 4096
	for i := 0; i < size; i += pageSize {
		end := i + pageSize
		if end > size {
			end = size
		}
		switch (i / pageSize) % 3 {
		case 0:
			// zero page: leave as-is
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

// transferOverLoopback sends payload through a real AMST.Connect+Transfer
// into a real draining loopback listener (see drainListener in
// benchmark_amst_bandwidth_test.go). Reused here so migration time and AMST
// bandwidth measure the exact same real transfer code path.
func transferOverLoopback(b *testing.B, payload []byte, streams int) {
	b.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		b.Fatal(err)
	}
	defer ln.Close()
	go drainListener(ln)

	addr := ln.Addr().(*net.TCPAddr)

	amst, err := dwcp.NewAMST(dwcp.AMSTConfig{InitialStreams: streams, ChunkSize: 64 * 1024})
	if err != nil {
		b.Fatal(err)
	}
	defer amst.Close()

	if err := amst.Connect(context.Background(), "127.0.0.1", addr.Port); err != nil {
		b.Fatal(err)
	}
	if err := amst.Transfer(context.Background(), payload, nil); err != nil {
		b.Fatal(err)
	}
}

// BenchmarkMigrationPipelinePrimitives measures the real HDE.CompressMemory +
// real AMST.Transfer PRIMITIVES a VM migration would use, compressed vs
// uncompressed, over the identical real transfer path. This deliberately
// does NOT exercise production MigrationAdapter.MigrateVMMemory end-to-end:
// MigrationAdapter.receiveMemory/receiveDisk (migration_adapter.go:743-752)
// are empty stubs, so there is no functional receive path to benchmark
// against — see the Step 5 follow-up issue. This number is real for the
// compress+transfer primitives; it is NOT a measurement of production
// migration time, and must not be reported as one.
func BenchmarkMigrationPipelinePrimitives(b *testing.B) {
	vmSizes := []int{16 * 1024 * 1024, 64 * 1024 * 1024, 256 * 1024 * 1024}

	for _, size := range vmSizes {
		b.Run(fmt.Sprintf("vm_%dMB_standard_uncompressed", size/1024/1024), func(b *testing.B) {
			data := generateVMMemoryData(size)
			b.SetBytes(int64(size))
			b.ResetTimer()
			for range b.N {
				transferOverLoopback(b, data, 8)
			}
		})

		b.Run(fmt.Sprintf("vm_%dMB_dwcp_hde_compressed", size/1024/1024), func(b *testing.B) {
			hde, err := dwcp.NewHDE(dwcp.HDEConfig{GlobalLevel: 9, EnableDelta: true})
			if err != nil {
				b.Fatal(err)
			}
			defer hde.Close()

			data := generateVMMemoryData(size)
			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := range b.N {
				compressed, cerr := hde.CompressMemory(fmt.Sprintf("bench-vm-%d", i), data, dwcp.CompressionGlobal)
				if cerr != nil {
					b.Fatal(cerr)
				}
				transferOverLoopback(b, compressed, 8)
			}
		})
	}
}

// BenchmarkMigrationPipelinePrimitivesDeltaReuse measures the real speedup
// from delta encoding on a second migration of the same VM after a ~5%
// memory change — the realistic live-migration iterative-copy scenario.
// Same primitives-only scope as BenchmarkMigrationPipelinePrimitives above.
//
// Each iteration re-primes its own uniquely-vmID'd baseline (via a
// b.StopTimer/b.StartTimer-bracketed priming call, excluded from the timed
// measurement) before compressing+transferring the ~5%-changed buffer:
// CompressMemory unconditionally overwrites the stored baseline with
// whatever it was just called with (hde.go:229-256), so reusing one fixed
// vmID across b.N iterations would make every iteration after the first
// diff the modified buffer against itself (a zero-size delta) instead of
// against the unmodified base this benchmark exists to model — silently
// collapsing both the measured time and the transferred byte count after
// iteration 1.
func BenchmarkMigrationPipelinePrimitivesDeltaReuse(b *testing.B) {
	size := 64 * 1024 * 1024

	hde, err := dwcp.NewHDE(dwcp.HDEConfig{GlobalLevel: 9, EnableDelta: true, BlockSize: 4096, DeltaThreshold: 0.7})
	if err != nil {
		b.Fatal(err)
	}
	defer hde.Close()

	base := generateVMMemoryData(size)

	modified := make([]byte, size)
	copy(modified, base)
	for i := 0; i < len(modified); i += 20 * 4096 { // touch ~5% of 4KB pages
		for j := 0; j < 4096 && i+j < len(modified); j++ {
			modified[i+j] = byte(i + j)
		}
	}

	b.SetBytes(int64(size))
	b.ResetTimer()
	for i := range b.N {
		vmID := fmt.Sprintf("bench-vm-delta-reuse-%d", i)

		b.StopTimer()
		if _, err := hde.CompressMemory(vmID, base, dwcp.CompressionGlobal); err != nil {
			b.Fatal(err)
		}
		b.StartTimer()

		compressed, cerr := hde.CompressMemory(vmID, modified, dwcp.CompressionGlobal)
		if cerr != nil {
			b.Fatal(cerr)
		}
		transferOverLoopback(b, compressed, 8)
	}
}
