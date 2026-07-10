package tests

import (
	"context"
	"crypto/rand"
	"fmt"
	"io"
	"net"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// drainListener accepts connections on ln until it is closed, discarding
// every byte read. AMST.Transfer's per-stream goroutines only ever write to
// their connection (verified: amst.go Transfer never calls conn.Read) so a
// raw discard sink is sufficient for Transfer to complete — no receiver-side
// AMST/Receive coordination or handshake is required.
func drainListener(ln net.Listener) {
	for {
		c, err := ln.Accept()
		if err != nil {
			return // listener closed
		}
		go io.Copy(io.Discard, c)
	}
}

// BenchmarkAMSTThroughputReal measures real AMST send-path throughput: real
// chunking (amst.go Transfer), real N-stream parallelism, real socket writes
// over loopback TCP. Loopback means this is memcpy/syscall-bound, not
// WAN-bound — report as "loopback send throughput", not "WAN bandwidth".
//
// MinStreams is pinned to 1: NewAMST defaults MinStreams to 4 when unset
// (amst.go:126-140) and then floors InitialStreams up to MinStreams, which
// would silently turn the streams_1 subtest into a 4-stream run and void
// the 1-vs-4 scaling comparison this benchmark exists to produce.
func BenchmarkAMSTThroughputReal(b *testing.B) {
	payloadSizes := []int{64 * 1024, 1024 * 1024, 16 * 1024 * 1024}
	streamCounts := []int{1, 4, 8, 16}

	for _, size := range payloadSizes {
		for _, streams := range streamCounts {
			b.Run(fmt.Sprintf("size_%dKB_streams_%d", size/1024, streams), func(b *testing.B) {
				ln, err := net.Listen("tcp", "127.0.0.1:0")
				if err != nil {
					b.Fatal(err)
				}
				defer ln.Close()
				go drainListener(ln)

				addr := ln.Addr().(*net.TCPAddr)

				amst, err := dwcp.NewAMST(dwcp.AMSTConfig{
					MinStreams:     1,
					InitialStreams: streams,
					ChunkSize:      64 * 1024,
				})
				if err != nil {
					b.Fatal(err)
				}
				defer amst.Close()

				if err := amst.Connect(context.Background(), "127.0.0.1", addr.Port); err != nil {
					b.Fatal(err)
				}

				payload := make([]byte, size)
				rand.Read(payload)

				b.SetBytes(int64(size))
				b.ResetTimer()

				for range b.N {
					if err := amst.Transfer(context.Background(), payload, nil); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// BenchmarkAMSTThroughputModeComparison compares v1 (static config, no
// adaptive) vs v3 (EnableAdaptive) real send-path throughput over the same
// loopback harness, using a fixed 1MB payload and 8 streams. The v3 case
// calls UpdateMetrics with representative network conditions and sets a
// short OptimizeInterval so the adaptive optimizer actually executes its
// logic (verified: amst.go optimize() early-returns when transferRate==0,
// which is only ever set by UpdateMetrics — without this the two subtests
// would be identical regardless of EnableAdaptive). optimize() prints
// "AMST: Optimized ..." to stdout when it adjusts parameters; expect noisy
// `go test -bench` output for the v3_adaptive subtest.
func BenchmarkAMSTThroughputModeComparison(b *testing.B) {
	size := 1024 * 1024
	streams := 8

	configs := []struct {
		name     string
		adaptive bool
	}{
		{"v1_static", false},
		{"v3_adaptive", true},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				b.Fatal(err)
			}
			defer ln.Close()
			go drainListener(ln)

			addr := ln.Addr().(*net.TCPAddr)

			amstCfg := dwcp.AMSTConfig{
				InitialStreams: streams,
				ChunkSize:      64 * 1024,
				EnableAdaptive: cfg.adaptive,
			}
			if cfg.adaptive {
				amstCfg.OptimizeInterval = 10 * time.Millisecond
			}

			amst, err := dwcp.NewAMST(amstCfg)
			if err != nil {
				b.Fatal(err)
			}
			defer amst.Close()

			if err := amst.Connect(context.Background(), "127.0.0.1", addr.Port); err != nil {
				b.Fatal(err)
			}

			if cfg.adaptive {
				// Representative network conditions: 20ms latency, 1% loss,
				// 50MB/s rate. Stored via atomics, so this single call keeps
				// transferRate non-zero for the whole benchmark, letting every
				// optimizationLoop tick run optimize()'s real logic.
				amst.UpdateMetrics(20, 0.01, 50*1024*1024)
			}

			data := make([]byte, size)
			rand.Read(data)

			b.SetBytes(int64(size))
			b.ResetTimer()

			for range b.N {
				if err := amst.Transfer(context.Background(), data, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
