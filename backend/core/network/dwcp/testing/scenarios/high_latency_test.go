package scenarios

import (
	"testing"
	"time"

	testing_pkg "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/testing"
)

// TestHighLatencyMigration tests migration under high latency
func TestHighLatencyMigration(t *testing.T) {
	harness := testing_pkg.NewTestHarness()
	scenario := testing_pkg.NewHighLatencyScenario()
	scenario.Duration = 2 * time.Minute

	result, err := harness.RunScenario(scenario)
	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	if !result.Passed {
		t.Errorf("High latency scenario failed")
		for _, reason := range result.FailureReasons {
			t.Errorf("  Failure: %s", reason)
		}
	}

	// Verify compression is effective under high latency
	hasCompressionAssertion := false
	for _, assertion := range result.Assertions {
		if assertion.Type == testing_pkg.AssertionCompressionRatio {
			hasCompressionAssertion = true
			if assertion.Actual < 10.0 {
				t.Logf("Warning: Compression ratio %.2fx is below optimal (expected >10x)", assertion.Actual)
			}
		}
	}

	if !hasCompressionAssertion {
		t.Error("No compression ratio assertion found")
	}
}

// TestExtremeLatency tests migration with extreme latency (satellite-like)
func TestExtremeLatency(t *testing.T) {
	scenario := testing_pkg.NewHighLatencyScenario()
	scenario.Name = "Extreme Latency (Satellite)"
	scenario.Duration = 5 * time.Minute

	// Simulate satellite latency (600ms)
	for _, link := range scenario.Topology.Links {
		link.Latency.BaseLatency = 600 * time.Millisecond
		link.Latency.Jitter = 50 * time.Millisecond
	}

	// Reduce workload size due to extreme latency
	scenario.Workload.VMs = 2
	scenario.Workload.VMSize = 1 * 1024 * 1024 * 1024 // 1 GB

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Verify system handles extreme latency gracefully
	if result.Metrics.LatencySamples > 0 {
		avgLatency := float64(result.Metrics.TotalLatency.Milliseconds()) / float64(result.Metrics.LatencySamples)
		t.Logf("Average latency: %.2fms", avgLatency)

		if avgLatency < 500 {
			t.Errorf("Latency simulation not working correctly: %.2fms (expected >500ms)", avgLatency)
		}
	}
}

// TestLatencyJitter tests migration with high jitter
func TestLatencyJitter(t *testing.T) {
	scenario := testing_pkg.NewHighLatencyScenario()
	scenario.Name = "High Jitter"
	scenario.Duration = 2 * time.Minute

	// Configure high jitter
	for _, link := range scenario.Topology.Links {
		link.Latency.BaseLatency = 100 * time.Millisecond
		link.Latency.Jitter = 80 * time.Millisecond // 80% jitter
		link.Latency.Distribution = testing_pkg.DistributionPareto
	}

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	t.Logf("High jitter test completed in %v", result.Duration)

	// System should handle jitter gracefully
	if result.Metrics.PacketsLost > result.Metrics.PacketsSent/2 {
		t.Errorf("Too many packets lost under jitter: %d/%d",
			result.Metrics.PacketsLost, result.Metrics.PacketsSent)
	}
}

// TestLatencySpikes tests migration with occasional latency spikes
func TestLatencySpikes(t *testing.T) {
	scenario := testing_pkg.NewHighLatencyScenario()
	scenario.Name = "Latency Spikes"
	scenario.Duration = 3 * time.Minute

	// Configure for occasional spikes
	for _, link := range scenario.Topology.Links {
		link.Latency.BaseLatency = 50 * time.Millisecond
		link.Latency.Jitter = 200 * time.Millisecond // Large jitter for spikes
		link.Latency.Distribution = testing_pkg.DistributionPareto
	}

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Verify operations can complete despite spikes
	if len(result.Metrics.OperationResults) == 0 {
		t.Error("No operations completed")
	}

	successCount := 0
	for _, op := range result.Metrics.OperationResults {
		if op.Success {
			successCount++
		}
	}

	successRate := float64(successCount) / float64(len(result.Metrics.OperationResults))
	if successRate < 0.90 {
		t.Errorf("Success rate too low under spikes: %.2f%% (expected >= 90%%)", successRate*100)
	}

	t.Logf("Success rate under latency spikes: %.2f%%", successRate*100)
}
