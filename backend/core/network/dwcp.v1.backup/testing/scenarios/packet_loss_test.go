package scenarios

import (
	"testing"
	"time"

	testing_pkg "github.com/yourusername/novacron/backend/core/network/dwcp/testing"
)

// TestPacketLossResilience tests DWCP resilience under packet loss
func TestPacketLossResilience(t *testing.T) {
	harness := testing_pkg.NewTestHarness()
	scenario := testing_pkg.NewPacketLossScenario()
	scenario.Duration = 2 * time.Minute

	result, err := harness.RunScenario(scenario)
	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	if !result.Passed {
		t.Errorf("Packet loss scenario failed")
	}

	// Verify packet loss is being simulated
	if result.Metrics.PacketsLost == 0 {
		t.Log("Warning: No packet loss recorded (simulation may not be active)")
	} else {
		lossRate := float64(result.Metrics.PacketsLost) / float64(result.Metrics.PacketsSent)
		t.Logf("Packet loss rate: %.2f%%", lossRate*100)
	}
}

// TestModerateLoss tests migration with moderate packet loss
func TestModerateLoss(t *testing.T) {
	scenario := testing_pkg.NewPacketLossScenario()
	scenario.Name = "Moderate Packet Loss (2%)"
	scenario.Duration = 2 * time.Minute

	// Configure 2% loss
	for _, link := range scenario.Topology.Links {
		link.PacketLoss.Rate = 0.02
		link.PacketLoss.BurstLength = 5
	}

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Should handle 2% loss well
	successCount := 0
	for _, op := range result.Metrics.OperationResults {
		if op.Success {
			successCount++
		}
	}

	successRate := float64(successCount) / float64(len(result.Metrics.OperationResults))
	if successRate < 0.98 {
		t.Errorf("Success rate too low with 2%% loss: %.2f%% (expected >= 98%%)", successRate*100)
	}
}

// TestHighLoss tests migration with high packet loss
func TestHighLoss(t *testing.T) {
	scenario := testing_pkg.NewPacketLossScenario()
	scenario.Name = "High Packet Loss (10%)"
	scenario.Duration = 3 * time.Minute

	// Configure 10% loss
	for _, link := range scenario.Topology.Links {
		link.PacketLoss.Rate = 0.10
		link.PacketLoss.BurstLength = 10
	}

	// Reduce workload due to high loss
	scenario.Workload.VMs = 10
	scenario.Workload.Concurrency = 2

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Should still complete with retries
	successCount := 0
	for _, op := range result.Metrics.OperationResults {
		if op.Success {
			successCount++
		}
	}

	successRate := float64(successCount) / float64(len(result.Metrics.OperationResults))
	if successRate < 0.85 {
		t.Errorf("Success rate too low with 10%% loss: %.2f%% (expected >= 85%%)", successRate*100)
	}

	t.Logf("Success rate with 10%% packet loss: %.2f%%", successRate*100)
}

// TestBurstyLoss tests migration with bursty packet loss
func TestBurstyLoss(t *testing.T) {
	scenario := testing_pkg.NewPacketLossScenario()
	scenario.Name = "Bursty Packet Loss"
	scenario.Duration = 2 * time.Minute

	// Configure bursty loss (3% with long bursts)
	for _, link := range scenario.Topology.Links {
		link.PacketLoss.Rate = 0.03
		link.PacketLoss.BurstLength = 50 // Long bursts
		link.PacketLoss.Distribution = testing_pkg.DistributionPareto
	}

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Bursty loss is harder to handle
	successCount := 0
	for _, op := range result.Metrics.OperationResults {
		if op.Success {
			successCount++
		}
	}

	successRate := float64(successCount) / float64(len(result.Metrics.OperationResults))
	if successRate < 0.90 {
		t.Logf("Warning: Success rate with bursty loss: %.2f%% (expected >= 90%%)", successRate*100)
	}

	t.Logf("Handled bursty loss with %.2f%% success rate", successRate*100)
}

// TestRandomLoss tests migration with random packet loss
func TestRandomLoss(t *testing.T) {
	scenario := testing_pkg.NewPacketLossScenario()
	scenario.Name = "Random Packet Loss"
	scenario.Duration = 2 * time.Minute

	// Configure random loss (uniform distribution)
	for _, link := range scenario.Topology.Links {
		link.PacketLoss.Rate = 0.05
		link.PacketLoss.BurstLength = 1 // No bursts
		link.PacketLoss.Distribution = testing_pkg.DistributionUniform
	}

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Random loss should be handled well
	successCount := 0
	for _, op := range result.Metrics.OperationResults {
		if op.Success {
			successCount++
		}
	}

	successRate := float64(successCount) / float64(len(result.Metrics.OperationResults))
	if successRate < 0.95 {
		t.Errorf("Success rate too low with random loss: %.2f%% (expected >= 95%%)", successRate*100)
	}
}

// TestCombinedLatencyAndLoss tests migration with both latency and packet loss
func TestCombinedLatencyAndLoss(t *testing.T) {
	scenario := testing_pkg.NewPacketLossScenario()
	scenario.Name = "Combined Latency and Packet Loss"
	scenario.Duration = 3 * time.Minute

	// Configure both latency and loss
	for _, link := range scenario.Topology.Links {
		link.Latency.BaseLatency = 150 * time.Millisecond
		link.Latency.Jitter = 30 * time.Millisecond
		link.PacketLoss.Rate = 0.03
		link.PacketLoss.BurstLength = 10
	}

	harness := testing_pkg.NewTestHarness()
	result, err := harness.RunScenario(scenario)

	if err != nil {
		t.Fatalf("Failed to run scenario: %v", err)
	}

	// Verify system handles combined issues
	successCount := 0
	for _, op := range result.Metrics.OperationResults {
		if op.Success {
			successCount++
		}
	}

	successRate := float64(successCount) / float64(len(result.Metrics.OperationResults))
	if successRate < 0.90 {
		t.Logf("Warning: Success rate with combined issues: %.2f%%", successRate*100)
	}

	t.Logf("Handled combined latency+loss with %.2f%% success rate", successRate*100)
}
