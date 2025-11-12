package scenarios

import (
	"testing"
	"time"

	testing_pkg "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

// TestCrossRegionMigration tests cross-region VM migration
func TestCrossRegionMigration(t *testing.T) {
	harness := testing_pkg.NewTestHarness()
	scenario := testing_pkg.NewCrossRegionScenario()

	// Reduce duration for testing
	scenario.Duration = 2 * time.Minute

	result, err := harness.RunScenario(scenario)
	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	if !result.Passed {
		t.Errorf("Scenario failed: %s", scenario.Name)
		for _, reason := range result.FailureReasons {
			t.Errorf("  Failure: %s", reason)
		}
	}

	// Verify specific assertions
	for _, assertion := range result.Assertions {
		t.Logf("Assertion %s: Expected=%.2f, Actual=%.2f, Passed=%v",
			assertion.Type, assertion.Expected, assertion.Actual, assertion.Passed)
	}

	// Verify metrics
	if result.Metrics.TotalBytes == 0 {
		t.Error("No data transferred")
	}

	if result.Metrics.LatencySamples == 0 {
		t.Error("No latency samples collected")
	}

	t.Logf("Migration completed in %v", result.Duration)
	t.Logf("Total bytes transferred: %d", result.Metrics.TotalBytes)
	t.Logf("Average latency: %.2fms",
		float64(result.Metrics.TotalLatency.Milliseconds())/float64(result.Metrics.LatencySamples))
}

// TestUSToEuropeMigration tests specific US-Europe migration
func TestUSToEuropeMigration(t *testing.T) {
	scenario := testing_pkg.NewCrossRegionScenario()
	scenario.Name = "US-Europe Migration"
	scenario.Duration = 1 * time.Minute

	// Focus on US-Europe link
	scenario.Workload.VMs = 5

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	if !result.Passed {
		t.Errorf("US-Europe migration failed")
	}

	// Check latency is within expected range (80ms ± 30ms)
	if result.Metrics.LatencySamples > 0 {
		avgLatency := float64(result.Metrics.TotalLatency.Milliseconds()) / float64(result.Metrics.LatencySamples)
		if avgLatency < 50 || avgLatency > 150 {
			t.Errorf("Unexpected latency: %.2fms (expected 50-150ms)", avgLatency)
		}
	}
}

// TestUSToAsiaMigration tests specific US-Asia migration
func TestUSToAsiaMigration(t *testing.T) {
	scenario := testing_pkg.NewCrossRegionScenario()
	scenario.Name = "US-Asia Migration"
	scenario.Duration = 1 * time.Minute

	// Higher latency expected
	scenario.Workload.VMs = 3

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Check latency is within expected range (200ms ± 50ms)
	if result.Metrics.LatencySamples > 0 {
		avgLatency := float64(result.Metrics.TotalLatency.Milliseconds()) / float64(result.Metrics.LatencySamples)
		if avgLatency < 150 || avgLatency > 250 {
			t.Logf("Warning: Unexpected latency: %.2fms (expected 150-250ms)", avgLatency)
		}
	}
}

// TestMultiRegionFailover tests failover across multiple regions
func TestMultiRegionFailover(t *testing.T) {
	scenario := testing_pkg.NewCrossRegionScenario()
	scenario.Name = "Multi-Region Failover"
	scenario.Duration = 3 * time.Minute

	// Simulate multiple VMs failing over
	scenario.Workload.VMs = 20
	scenario.Workload.Concurrency = 4

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Verify success rate is high
	successCount := 0
	for _, op := range result.Metrics.OperationResults {
		if op.Success {
			successCount++
		}
	}

	successRate := float64(successCount) / float64(len(result.Metrics.OperationResults))
	if successRate < 0.95 {
		t.Errorf("Success rate too low: %.2f%% (expected >= 95%%)", successRate*100)
	}

	t.Logf("Failover success rate: %.2f%%", successRate*100)
}
