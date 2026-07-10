package tests

import (
	"context"
	"crypto/rand"
	"fmt"
	"net"
	"syscall"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// cpuTimeNanos returns the process's cumulative user+system CPU time in
// nanoseconds via getrusage(RUSAGE_SELF) — real CPU time, not a wall-clock
// proxy. Caveat, stated once: this is process-wide, not goroutine-scoped, so
// background goroutines (e.g. AMST's optimizer loop) contribute to the
// delta. Acceptable for these benchmarks since b.N iterations dominate the
// sample and each subtest isolates one operation.
func cpuTimeNanos() int64 {
	var ru syscall.Rusage
	syscall.Getrusage(syscall.RUSAGE_SELF, &ru)
	user := ru.Utime.Sec*1e9 + ru.Utime.Usec*1e3
	sys := ru.Stime.Sec*1e9 + ru.Stime.Usec*1e3
	return user + sys
}

// BenchmarkCPUOverhead measures real CPU time and allocations for DWCP hot paths.
func BenchmarkCPUOverhead(b *testing.B) {
	b.Run("hde_compress_1MB_global", func(b *testing.B) {
		hde, err := dwcp.NewHDE(dwcp.HDEConfig{GlobalLevel: 9})
		if err != nil {
			b.Fatal(err)
		}
		defer hde.Close()

		data := make([]byte, 1024*1024)
		rand.Read(data)

		b.ReportAllocs()
		startCPU := cpuTimeNanos()
		b.SetBytes(1024 * 1024)
		b.ResetTimer()

		for range b.N {
			if _, err := hde.CompressMemory("vm-bench", data, dwcp.CompressionGlobal); err != nil {
				b.Fatal(err)
			}
		}

		b.StopTimer()
		endCPU := cpuTimeNanos()
		b.ReportMetric(float64(endCPU-startCPU)/float64(b.N), "cpu-ns/op")
	})

	b.Run("hde_delta_encode_5pct_change", func(b *testing.B) {
		hde, err := dwcp.NewHDE(dwcp.HDEConfig{GlobalLevel: 3, EnableDelta: true, BlockSize: 4096, DeltaThreshold: 0.7})
		if err != nil {
			b.Fatal(err)
		}
		defer hde.Close()

		base := make([]byte, 1024*1024)
		rand.Read(base)

		modified := make([]byte, 1024*1024)
		copy(modified, base)
		for i := 0; i < len(modified); i += 20 * 1024 {
			modified[i] = byte(i % 256)
		}

		b.ReportAllocs()
		b.SetBytes(1024 * 1024)
		b.ResetTimer()

		// Each iteration re-primes its own uniquely-vmID'd baseline (outside
		// both the wall-clock timer and the CPU-time window) before
		// delta-encoding the ~5%-changed buffer. CompressMemory
		// unconditionally overwrites the stored baseline with whatever it
		// was just called with (hde.go:229-256), so a fixed vmID reused
		// across b.N iterations would make every iteration after the first
		// diff `modified` against itself (a zero-size delta) instead of
		// against `base` — understating the real 5%-change delta-encode
		// cost this subtest exists to measure. Priming is excluded from
		// both b's native timer (so ns/op and MB/s reflect one compress,
		// matching SetBytes(1024*1024)) and from cpuDelta (so cpu-ns/op
		// isolates the timed CompressMemory call only).
		var cpuDelta int64
		for i := range b.N {
			vmID := fmt.Sprintf("vm-bench-delta-%d", i)

			b.StopTimer()
			if _, err := hde.CompressMemory(vmID, base, dwcp.CompressionGlobal); err != nil {
				b.Fatal(err)
			}
			b.StartTimer()

			start := cpuTimeNanos()
			if _, err := hde.CompressMemory(vmID, modified, dwcp.CompressionGlobal); err != nil {
				b.Fatal(err)
			}
			cpuDelta += cpuTimeNanos() - start
		}

		b.StopTimer()
		b.ReportMetric(float64(cpuDelta)/float64(b.N), "cpu-ns/op")
	})

	b.Run("amst_metrics_bookkeeping", func(b *testing.B) {
		amst, err := dwcp.NewAMST(dwcp.AMSTConfig{InitialStreams: 8})
		if err != nil {
			b.Fatal(err)
		}
		defer amst.Close()

		b.ReportAllocs()
		startCPU := cpuTimeNanos()
		b.ResetTimer()

		for range b.N {
			amst.UpdateMetrics(5, 0.001, 10e9)
			_ = amst.GetMetrics()
		}

		b.StopTimer()
		endCPU := cpuTimeNanos()
		b.ReportMetric(float64(endCPU-startCPU)/float64(b.N), "cpu-ns/op")
	})

	b.Run("hde_dictionary_training_100_samples", func(b *testing.B) {
		hde, err := dwcp.NewHDE(dwcp.HDEConfig{GlobalLevel: 3, EnableDictionary: true, DictSize: 1024, TrainingSamples: 100})
		if err != nil {
			b.Fatal(err)
		}
		defer hde.Close()

		samples := make([][]byte, 100)
		for i := range samples {
			samples[i] = make([]byte, 4096)
			rand.Read(samples[i])
		}

		// HDE.TrainDictionary is unconditionally broken, not something a
		// benchmark-side data change can work around: it calls
		// zstd.BuildDict with only Contents set, never History (hde.go
		// builds the concatenated `trainingData` and then never uses it —
		// evidently meant to be passed as History and the line was
		// dropped). zstd.BuildDict hard-requires len(History) >= 8 before
		// touching Contents at all, so this fails for every input with
		// "dictionary of size 0 < 8". Confirmed by direct reproduction
		// below. Root-caused and filed as novacron-o0e (also root-causes
		// the pre-existing quarantined TestHDEv3DictionaryTraining,
		// hde_v3_test.go:271, t.Skip("...see novacron-v4y")). No exported
		// API lets a caller supply History, so there is no way to make
		// this call succeed from this package — matching how Step 5 of
		// this plan handles MigrationAdapter's receiveMemory/receiveDisk
		// stubs (novacron-lce): document and skip, don't patch production
		// code from a benchmark-implementation change.
		if err := hde.TrainDictionary("vm-type-bench", samples); err == nil {
			b.Fatal("TrainDictionary unexpectedly succeeded — novacron-o0e may be fixed; remove this b.Skip and restore the real benchmark loop below")
		}
		b.Skip("HDE.TrainDictionary is unconditionally broken (zstd.BuildDict missing required History field) — see novacron-o0e")
	})

	b.Run("amst_transfer_1MB_8streams", func(b *testing.B) {
		ln, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			b.Fatal(err)
		}
		defer ln.Close()
		go drainListener(ln)
		addr := ln.Addr().(*net.TCPAddr)

		amst, err := dwcp.NewAMST(dwcp.AMSTConfig{InitialStreams: 8, ChunkSize: 64 * 1024})
		if err != nil {
			b.Fatal(err)
		}
		defer amst.Close()
		if err := amst.Connect(context.Background(), "127.0.0.1", addr.Port); err != nil {
			b.Fatal(err)
		}

		data := make([]byte, 1024*1024)
		rand.Read(data)

		b.ReportAllocs()
		startCPU := cpuTimeNanos()
		b.SetBytes(1024 * 1024)
		b.ResetTimer()

		for range b.N {
			if err := amst.Transfer(context.Background(), data, nil); err != nil {
				b.Fatal(err)
			}
		}

		b.StopTimer()
		endCPU := cpuTimeNanos()
		b.ReportMetric(float64(endCPU-startCPU)/float64(b.N), "cpu-ns/op")
	})
}
