package tests

import (
	"context"
	"crypto/rand"
	"fmt"
	"net"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// generateIncompressibleData builds a near-random payload representing
// the worst realistic case for compression: encrypted VM memory or
// already-compressed application data. Used alongside
// generateVMMemoryData (typical guest memory, ~3x compressible) so
// BenchmarkMigrationWANBandwidthConstrained reports the break-even case,
// not just the favorable one.
func generateIncompressibleData(size int) []byte {
	data := make([]byte, size)
	rand.Read(data)
	return data
}

// transferOverLoopbackBandwidthLimited sends payload through a real
// AMST.Connect+Transfer, throttled to bandwidthBps via AMSTConfig's real
// token-bucket rate.Limiter (amst.go: rateLimiter.WaitN per chunk during
// Transfer) - not a synthetic delay, a genuine per-chunk blocking wait
// until enough tokens accumulate, so wall-clock transfer time is real
// and proportional to len(payload)/bandwidthBps, same as it would be on
// an actual bandwidth-constrained link. Single stream, matching
// MigrationAdapter's forced production config (NewMigrationAdapter pins
// MinStreams=MaxStreams=InitialStreams=1).
func transferOverLoopbackBandwidthLimited(b *testing.B, payload []byte, bandwidthBps int64) time.Duration {
	b.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		b.Fatal(err)
	}
	defer ln.Close()
	go drainListener(ln)

	addr := ln.Addr().(*net.TCPAddr)

	amst, err := dwcp.NewAMST(dwcp.AMSTConfig{
		MinStreams: 1, MaxStreams: 1, InitialStreams: 1,
		ChunkSize:      64 * 1024,
		BandwidthLimit: bandwidthBps,
		// Default burst is BandwidthLimit/10 (100ms worth) - for a 64MB
		// payload at 200Mbps that's ~2.5MB transferring "free" before
		// throttling kicks in, a small but real systematic underestimate
		// of total time. Pin burst to 4x chunk size instead (256KB) so
		// the measured time reflects sustained throttled throughput from
		// the start, not the initial burst allowance. Must stay >=
		// ChunkSize or Transfer's per-chunk WaitN(n) errors when a
		// chunk exceeds the bucket's max size. EnableAdaptive is
		// intentionally left at its zero value (false, no optimizer
		// created) so chunkSize stays pinned at 64KB for the AMST
		// instance's lifetime - if adaptive optimization grew chunks
		// toward MaxChunkSize (1MB default), a chunk could exceed this
		// burst and Transfer's WaitN would hard-fail.
		BurstSize: 4 * 64 * 1024,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer amst.Close()

	if err := amst.Connect(context.Background(), "127.0.0.1", addr.Port); err != nil {
		b.Fatal(err)
	}
	start := time.Now()
	if err := amst.Transfer(context.Background(), payload, nil); err != nil {
		b.Fatal(err)
	}
	return time.Since(start)
}

// BenchmarkMigrationWANBandwidthConstrained answers the question
// novacron-38p's Go/No-Go actually needs answered and that no benchmark
// in this repo answered before it: does DWCP compression reduce total
// migration time (compress + transfer + decompress) on a
// bandwidth-constrained WAN link, as opposed to loopback where
// near-infinite bandwidth means compression's CPU cost always loses
// (BenchmarkMigrationAdapterEndToEnd, migration_adapter_benchmark_test.go).
//
// Two load-bearing assumptions, both stated explicitly rather than
// buried in the numbers:
//
//  1. Bandwidth profiles are ASSUMED representative values, not measured
//     from a real link (this environment has no real WAN path to measure
//     against). Regional: 200 Mbps (representative same-continent
//     cross-region cloud link; real values vary 100Mbps-1Gbps+ by
//     provider/tier). Global: 50 Mbps (representative cross-continent
//     public internet path; real values vary widely and can be much
//     lower under congestion).
//  2. On a bandwidth-bound link with compress time small relative to
//     transfer time, total speedup approaches the compression RATIO
//     itself - so the result is conditional on how compressible the
//     payload actually is, not an independent property of "compression"
//     in the abstract. This benchmark therefore runs BOTH a realistic
//     mixed VM-memory payload (generateVMMemoryData: 1/3 zero pages,
//     1/3 repeating pattern, 1/3 random - representative of typical
//     guest memory's zero-page/redundancy fraction) AND a
//     near-incompressible payload (pure random - representative of
//     encrypted memory or already-compressed application data, the
//     worst realistic case) so the Go/No-Go sees the break-even point,
//     not just the favorable case.
//
// Every number this benchmark reports is a real wall-clock measurement
// of real HDE.CompressMemory + real AMST.Transfer (throttled via AMST's
// real token-bucket rate limiter, AMSTConfig.BandwidthLimit) + real
// HDE.Decompress on the received bytes - not the synthetic/fabricated
// numbers this session found in BENCHMARK_RESULTS.md, and not a
// closed-form formula. Decompression happens locally on the transferred
// bytes rather than through a second real network hop (drainListener on
// the receive end just discards - see transferOverLoopbackBandwidthLimited)
// since decompression itself doesn't depend on the network path; this
// keeps the reported total honest about the full compress+transfer+
// decompress cost without needing a second bandwidth-limited connection.
//
// This still does not model real packet loss, jitter, or RTT-bound TCP
// window effects - it models the two variables (bandwidth, payload
// compressibility) that determine whether compression's CPU cost is
// worth paying. A network namespace + tc/netem or real multi-region
// deployment would be needed for full packet-level fidelity; not
// attempted here to avoid mutating network state on a machine shared
// with other concurrent sessions (see novacron-38p session notes).
func BenchmarkMigrationWANBandwidthConstrained(b *testing.B) {
	const mbps = 1024 * 1024 / 8 // bytes/sec per 1 Mbps

	bandwidthProfiles := []struct {
		name         string
		bandwidthBps int64
		tier         dwcp.CompressionLevel
	}{
		{"regional_200mbps", 200 * mbps, dwcp.CompressionRegional},
		{"global_50mbps", 50 * mbps, dwcp.CompressionGlobal},
	}

	size := 64 * 1024 * 1024 // 64MB, matching this session's other VM-memory benchmarks

	dataProfiles := []struct {
		name string
		data []byte
	}{
		{"typical_vm_memory", generateVMMemoryData(size)},
		{"incompressible", generateIncompressibleData(size)},
	}

	hde, err := dwcp.NewHDE(dwcp.HDEConfig{RegionalLevel: 3, GlobalLevel: 9})
	if err != nil {
		b.Fatal(err)
	}
	defer hde.Close()

	for _, bp := range bandwidthProfiles {
		b.Run(fmt.Sprintf("%s_uncompressed", bp.name), func(b *testing.B) {
			// Uncompressed transfer time does not depend on payload
			// compressibility - one subtest per bandwidth profile
			// covers both data profiles' baseline.
			data := dataProfiles[0].data
			b.SetBytes(int64(size))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				elapsed := transferOverLoopbackBandwidthLimited(b, data, bp.bandwidthBps)
				b.ReportMetric(elapsed.Seconds(), "total_sec/op")
			}
		})

		for _, dp := range dataProfiles {
			b.Run(fmt.Sprintf("%s_compressed_%s", bp.name, dp.name), func(b *testing.B) {
				b.SetBytes(int64(size))
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					compressStart := time.Now()
					compressed, cerr := hde.CompressMemory(fmt.Sprintf("bench-wan-%s-%s-%d", bp.name, dp.name, i), dp.data, bp.tier)
					if cerr != nil {
						b.Fatal(cerr)
					}
					compressDuration := time.Since(compressStart)

					transferDuration := transferOverLoopbackBandwidthLimited(b, compressed, bp.bandwidthBps)

					decompressStart := time.Now()
					decompressed, derr := hde.Decompress(compressed)
					if derr != nil {
						b.Fatal(derr)
					}
					decompressDuration := time.Since(decompressStart)
					if len(decompressed) != len(dp.data) {
						b.Fatalf("decompressed size mismatch: got %d, want %d", len(decompressed), len(dp.data))
					}

					total := compressDuration + transferDuration + decompressDuration
					b.ReportMetric(total.Seconds(), "total_sec/op")
					b.ReportMetric(compressDuration.Seconds(), "compress_sec/op")
					b.ReportMetric(transferDuration.Seconds(), "transfer_sec/op")
					b.ReportMetric(decompressDuration.Seconds(), "decompress_sec/op")
					b.ReportMetric(float64(len(dp.data))/float64(len(compressed)), "ratio")
				}
			})
		}
	}
}
