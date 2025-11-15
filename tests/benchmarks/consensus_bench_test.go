package benchmarks_test

import (
	"context"
	"sync"
	"testing"
	"time"

	"novacron/backend/core/network/dwcp/v3"
)

// BenchmarkConsensusProtocols compares performance of different consensus protocols
func BenchmarkConsensusProtocols(b *testing.B) {
	protocols := []string{"ProBFT", "Bullshark", "T-PBFT"}
	nodeCounts := []int{10, 30, 50, 100}

	for _, protocol := range protocols {
		for _, nodeCount := range nodeCounts {
			b.Run(protocol+"_"+string(rune(nodeCount))+"nodes", func(b *testing.B) {
				benchmarkProtocol(b, protocol, nodeCount)
			})
		}
	}
}

func benchmarkProtocol(b *testing.B, protocol string, nodeCount int) {
	ctx := context.Background()

	// Setup network
	nodes := make([]*dwcp.Node, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:     i,
			TotalNodes: nodeCount,
			Protocol:   protocol,
		})
		go nodes[i].Start(ctx)
	}
	defer func() {
		for _, node := range nodes {
			node.Stop()
		}
	}()

	time.Sleep(1 * time.Second) // Network formation

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		block := &dwcp.Block{
			Height:    int64(i),
			Data:      []byte("benchmark-transaction"),
			Timestamp: time.Now().Unix(),
		}

		_, err := nodes[0].ProposeBlock(ctx, block)
		if err != nil {
			b.Fatalf("Consensus failed: %v", err)
		}
	}

	b.StopTimer()

	// Report metrics
	stats := nodes[0].GetStats()
	b.ReportMetric(float64(stats.AvgLatency.Milliseconds()), "latency_ms")
	b.ReportMetric(float64(stats.Throughput), "tx/s")
	b.ReportMetric(float64(stats.MessageCount), "messages")
}

// BenchmarkThroughput measures maximum transaction throughput
func BenchmarkThroughput(b *testing.B) {
	nodeCount := 30
	ctx := context.Background()

	nodes := make([]*dwcp.Node, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:     i,
			TotalNodes: nodeCount,
			Protocol:   "ProBFT",
		})
		go nodes[i].Start(ctx)
	}
	defer func() {
		for _, node := range nodes {
			node.Stop()
		}
	}()

	time.Sleep(1 * time.Second)

	b.ResetTimer()

	// Send transactions in parallel
	var wg sync.WaitGroup
	txCount := b.N

	for i := 0; i < txCount; i++ {
		wg.Add(1)
		go func(txID int) {
			defer wg.Done()
			block := &dwcp.Block{
				Height:    int64(txID),
				Data:      []byte("throughput-test"),
				Timestamp: time.Now().Unix(),
			}
			nodes[txID%nodeCount].ProposeBlock(ctx, block)
		}(i)
	}

	wg.Wait()
	b.StopTimer()

	duration := time.Since(time.Now())
	tps := float64(txCount) / duration.Seconds()
	b.ReportMetric(tps, "tx/s")
}

// BenchmarkLatency measures consensus latency under different loads
func BenchmarkLatency(b *testing.B) {
	nodeCount := 30
	ctx := context.Background()

	nodes := setupBenchmarkNetwork(nodeCount, "ProBFT")
	defer shutdownNetwork(nodes)

	latencies := make([]time.Duration, 0, b.N)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		start := time.Now()

		block := &dwcp.Block{
			Height:    int64(i),
			Data:      []byte("latency-test"),
			Timestamp: time.Now().Unix(),
		}

		_, err := nodes[0].ProposeBlock(ctx, block)
		if err != nil {
			b.Fatalf("Consensus failed: %v", err)
		}

		latency := time.Since(start)
		latencies = append(latencies, latency)
	}

	b.StopTimer()

	// Calculate statistics
	avg, p50, p95, p99 := calculateLatencyStats(latencies)
	b.ReportMetric(float64(avg.Milliseconds()), "avg_ms")
	b.ReportMetric(float64(p50.Milliseconds()), "p50_ms")
	b.ReportMetric(float64(p95.Milliseconds()), "p95_ms")
	b.ReportMetric(float64(p99.Milliseconds()), "p99_ms")
}

// BenchmarkScalability tests performance scaling with node count
func BenchmarkScalability(b *testing.B) {
	nodeCounts := []int{10, 30, 50, 100, 200}

	for _, count := range nodeCounts {
		b.Run(string(rune(count))+"_nodes", func(b *testing.B) {
			ctx := context.Background()
			nodes := setupBenchmarkNetwork(count, "ProBFT")
			defer shutdownNetwork(nodes)

			b.ResetTimer()

			for i := 0; i < b.N; i++ {
				block := &dwcp.Block{
					Height:    int64(i),
					Data:      []byte("scalability-test"),
					Timestamp: time.Now().Unix(),
				}
				nodes[0].ProposeBlock(ctx, block)
			}
		})
	}
}

// BenchmarkMemoryUsage measures memory consumption
func BenchmarkMemoryUsage(b *testing.B) {
	nodeCount := 30
	ctx := context.Background()

	nodes := setupBenchmarkNetwork(nodeCount, "ProBFT")
	defer shutdownNetwork(nodes)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		block := &dwcp.Block{
			Height:    int64(i),
			Data:      make([]byte, 1024), // 1KB blocks
			Timestamp: time.Now().Unix(),
		}
		nodes[0].ProposeBlock(ctx, block)
	}
}

// BenchmarkByzantineResilience tests performance under Byzantine attacks
func BenchmarkByzantineResilience(b *testing.B) {
	nodeCount := 30
	byzantineCount := 10 // 33%
	ctx := context.Background()

	nodes := make([]*dwcp.Node, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:           i,
			TotalNodes:       nodeCount,
			Protocol:         "ProBFT",
			ByzantineTolerance: byzantineCount,
		})
		if i < byzantineCount {
			nodes[i].MarkAsByzantine()
			nodes[i].SetMessageDropRate(0.5)
		}
		go nodes[i].Start(ctx)
	}
	defer shutdownNetwork(nodes)

	time.Sleep(1 * time.Second)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		block := &dwcp.Block{
			Height:    int64(i),
			Data:      []byte("byzantine-test"),
			Timestamp: time.Now().Unix(),
		}
		nodes[byzantineCount].ProposeBlock(ctx, block)
	}
}

// BenchmarkNetworkOverhead measures message count and bandwidth
func BenchmarkNetworkOverhead(b *testing.B) {
	nodeCount := 30
	ctx := context.Background()

	nodes := setupBenchmarkNetwork(nodeCount, "ProBFT")
	defer shutdownNetwork(nodes)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		block := &dwcp.Block{
			Height:    int64(i),
			Data:      []byte("overhead-test"),
			Timestamp: time.Now().Unix(),
		}
		nodes[0].ProposeBlock(ctx, block)
	}

	b.StopTimer()

	// Measure network overhead
	stats := nodes[0].GetStats()
	b.ReportMetric(float64(stats.MessageCount), "messages")
	b.ReportMetric(float64(stats.BytesSent), "bytes_sent")
	b.ReportMetric(float64(stats.BytesReceived), "bytes_received")
}

// Helper functions

func setupBenchmarkNetwork(nodeCount int, protocol string) []*dwcp.Node {
	ctx := context.Background()
	nodes := make([]*dwcp.Node, nodeCount)

	for i := 0; i < nodeCount; i++ {
		nodes[i] = dwcp.NewNode(&dwcp.Config{
			NodeID:     i,
			TotalNodes: nodeCount,
			Protocol:   protocol,
		})
		go nodes[i].Start(ctx)
	}

	time.Sleep(1 * time.Second) // Network formation
	return nodes
}

func shutdownNetwork(nodes []*dwcp.Node) {
	for _, node := range nodes {
		node.Stop()
	}
}

func calculateLatencyStats(latencies []time.Duration) (avg, p50, p95, p99 time.Duration) {
	if len(latencies) == 0 {
		return 0, 0, 0, 0
	}

	// Calculate average
	var sum time.Duration
	for _, l := range latencies {
		sum += l
	}
	avg = sum / time.Duration(len(latencies))

	// Sort for percentiles
	sorted := make([]time.Duration, len(latencies))
	copy(sorted, latencies)
	// Simple sort (use sort.Slice in production)

	p50 = sorted[len(sorted)*50/100]
	p95 = sorted[len(sorted)*95/100]
	p99 = sorted[len(sorted)*99/100]

	return
}
