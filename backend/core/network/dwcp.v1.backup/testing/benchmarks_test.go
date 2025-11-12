package testing

import (
	"testing"
	"time"
)

// BenchmarkCrossRegionMigration benchmarks cross-region VM migration
func BenchmarkCrossRegionMigration(b *testing.B) {
	harness := setupBenchmarkHarness()
	scenario := NewCrossRegionScenario()
	scenario.Duration = 1 * time.Minute // Shorter for benchmarking

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := harness.RunScenario(scenario)
		if err != nil {
			b.Fatalf("Scenario failed: %v", err)
		}
		if !result.Passed {
			b.Fatalf("Scenario did not pass")
		}
	}
}

// BenchmarkVariousLatencies benchmarks migration with different latencies
func BenchmarkVariousLatencies(b *testing.B) {
	harness := setupBenchmarkHarness()

	latencies := []time.Duration{
		10 * time.Millisecond,
		50 * time.Millisecond,
		100 * time.Millisecond,
		200 * time.Millisecond,
		500 * time.Millisecond,
	}

	for _, latency := range latencies {
		b.Run(latency.String(), func(b *testing.B) {
			scenario := createScenarioWithLatency(latency)
			scenario.Duration = 30 * time.Second

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				harness.RunScenario(scenario)
			}
		})
	}
}

// BenchmarkVariousPacketLoss benchmarks migration with different packet loss rates
func BenchmarkVariousPacketLoss(b *testing.B) {
	harness := setupBenchmarkHarness()

	lossRates := []float64{0.1, 0.5, 1.0, 2.0, 5.0}

	for _, lossRate := range lossRates {
		b.Run(formatLossRate(lossRate), func(b *testing.B) {
			scenario := createScenarioWithPacketLoss(lossRate)
			scenario.Duration = 30 * time.Second

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				harness.RunScenario(scenario)
			}
		})
	}
}

// BenchmarkVariousBandwidths benchmarks migration with different bandwidths
func BenchmarkVariousBandwidths(b *testing.B) {
	harness := setupBenchmarkHarness()

	bandwidths := []int{10, 100, 1000, 10000} // Mbps

	for _, bandwidth := range bandwidths {
		b.Run(formatBandwidth(bandwidth), func(b *testing.B) {
			scenario := createScenarioWithBandwidth(bandwidth)
			scenario.Duration = 30 * time.Second

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				harness.RunScenario(scenario)
			}
		})
	}
}

// BenchmarkConcurrentMigrations benchmarks concurrent VM migrations
func BenchmarkConcurrentMigrations(b *testing.B) {
	harness := setupBenchmarkHarness()

	concurrencies := []int{1, 2, 4, 8, 16}

	for _, concurrency := range concurrencies {
		b.Run(formatConcurrency(concurrency), func(b *testing.B) {
			scenario := createScenarioWithConcurrency(concurrency)
			scenario.Duration = 1 * time.Minute

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				harness.RunScenario(scenario)
			}
		})
	}
}

// BenchmarkWorkloadPatterns benchmarks different workload patterns
func BenchmarkWorkloadPatterns(b *testing.B) {
	harness := setupBenchmarkHarness()

	patterns := []WorkloadPattern{
		PatternConstant,
		PatternBursty,
		PatternSinusoidal,
		PatternRealWorld,
	}

	for _, pattern := range patterns {
		b.Run(string(pattern), func(b *testing.B) {
			scenario := createScenarioWithPattern(pattern)
			scenario.Duration = 30 * time.Second

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				harness.RunScenario(scenario)
			}
		})
	}
}

// BenchmarkVMSizes benchmarks migration of different VM sizes
func BenchmarkVMSizes(b *testing.B) {
	harness := setupBenchmarkHarness()

	sizes := []int64{
		1 * 1024 * 1024 * 1024,  // 1 GB
		2 * 1024 * 1024 * 1024,  // 2 GB
		4 * 1024 * 1024 * 1024,  // 4 GB
		8 * 1024 * 1024 * 1024,  // 8 GB
		16 * 1024 * 1024 * 1024, // 16 GB
	}

	for _, size := range sizes {
		b.Run(formatSize(size), func(b *testing.B) {
			scenario := createScenarioWithVMSize(size)
			scenario.Duration = 1 * time.Minute

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				harness.RunScenario(scenario)
			}
		})
	}
}

// BenchmarkCompressionEfficiency benchmarks compression performance
func BenchmarkCompressionEfficiency(b *testing.B) {
	generator := NewWorkloadGenerator(PatternRealWorld, 4*1024*1024*1024)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		data := generator.GenerateVMMemory(4 * 1024 * 1024 * 1024)
		_ = data // Use data to prevent optimization
	}
}

// BenchmarkNetworkSimulator benchmarks the network simulator
func BenchmarkNetworkSimulator(b *testing.B) {
	topology := NewCrossRegionScenario().Topology
	simulator := NewNetworkSimulator(topology)
	simulator.ApplyTopology(topology)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = simulator.SimulateLatency("us-east-1", "eu-west-1")
		_ = simulator.SimulatePacketLoss("us-east-1", "eu-west-1")
		_ = simulator.GetAvailableBandwidth("us-east-1", "eu-west-1")
	}
}

// Helper functions for benchmark setup

func setupBenchmarkHarness() *TestHarness {
	return NewTestHarness()
}

func createScenarioWithLatency(latency time.Duration) *TestScenario {
	scenario := NewCrossRegionScenario()
	for _, link := range scenario.Topology.Links {
		link.Latency.BaseLatency = latency
		link.Latency.Jitter = latency / 10
	}
	return scenario
}

func createScenarioWithPacketLoss(lossRate float64) *TestScenario {
	scenario := NewCrossRegionScenario()
	for _, link := range scenario.Topology.Links {
		link.PacketLoss.Rate = lossRate / 100.0
	}
	return scenario
}

func createScenarioWithBandwidth(bandwidth int) *TestScenario {
	scenario := NewCrossRegionScenario()
	for _, link := range scenario.Topology.Links {
		link.Bandwidth.Capacity = bandwidth
	}
	return scenario
}

func createScenarioWithConcurrency(concurrency int) *TestScenario {
	scenario := NewCrossRegionScenario()
	scenario.Workload.Concurrency = concurrency
	scenario.Workload.Operations = concurrency * 5
	return scenario
}

func createScenarioWithPattern(pattern WorkloadPattern) *TestScenario {
	scenario := NewCrossRegionScenario()
	scenario.Workload.Pattern = pattern
	return scenario
}

func createScenarioWithVMSize(size int64) *TestScenario {
	scenario := NewCrossRegionScenario()
	scenario.Workload.VMSize = size
	return scenario
}

func formatLossRate(rate float64) string {
	return fmt.Sprintf("%.1f%%", rate)
}

func formatBandwidth(mbps int) string {
	if mbps >= 1000 {
		return fmt.Sprintf("%dGbps", mbps/1000)
	}
	return fmt.Sprintf("%dMbps", mbps)
}

func formatConcurrency(n int) string {
	return fmt.Sprintf("%d_concurrent", n)
}

func formatSize(bytes int64) string {
	gb := bytes / (1024 * 1024 * 1024)
	return fmt.Sprintf("%dGB", gb)
}
