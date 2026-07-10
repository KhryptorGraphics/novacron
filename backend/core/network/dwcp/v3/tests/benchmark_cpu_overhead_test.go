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

		// Mixed zero/pattern/random shape, matching generateVMMemoryData's
		// realistic VM-memory composition elsewhere in this package.
		// TrainDictionary was fixed this session (novacron-o0e: hde.go was
		// missing BuildDictOptions.History and a non-zero ID; the
		// underlying zstd.BuildDict also has a real, separate panic bug
		// for certain sample shapes — see dictionary_fix_test.go in the
		// dwcp package — which TrainDictionary now recovers from as a
		// returned error rather than crashing). This shape is confirmed
		// via standalone reproduction not to trigger that panic.
		samples := make([][]byte, 100)
		for i := range samples {
			s := make([]byte, 4096)
			switch i % 3 {
			case 0:
				// zero page: leave as-is
			case 1:
				for j := range s {
					s[j] = byte(j % 256)
				}
			case 2:
				rand.Read(s)
			}
			samples[i] = s
		}

		b.ReportAllocs()
		startCPU := cpuTimeNanos()
		b.ResetTimer()

		for range b.N {
			if err := hde.TrainDictionary("vm-type-bench", samples); err != nil {
				b.Fatal(err)
			}
		}

		b.StopTimer()
		endCPU := cpuTimeNanos()
		b.ReportMetric(float64(endCPU-startCPU)/float64(b.N), "cpu-ns/op")
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
