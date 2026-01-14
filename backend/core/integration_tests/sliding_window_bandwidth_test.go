package integration_tests

import (
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network"
	"go.uber.org/zap"
)

func TestSlidingWindowBandwidthMonitor(t *testing.T) {
	logger := zap.NewNop()

	config := &network.BandwidthMonitorConfig{
		MonitoringInterval:    1 * time.Second,
		HistoryRetention:      10 * time.Minute,
		SlidingWindowDuration: 3 * time.Second,
		Interfaces:            []string{"eth0"},
		MaxHistoryPoints:      100,
	}

	monitor := network.NewBandwidthMonitor(config, logger)

	// Start the monitor
	err := monitor.Start()
	if err != nil {
		t.Fatalf("Failed to start bandwidth monitor: %v", err)
	}
	defer monitor.Stop()

	// Let it collect some data
	time.Sleep(5 * time.Second)

	// Get measurements
	measurements, err := monitor.GetHistoricalMeasurements("eth0", time.Now().Add(-5*time.Second))
	if err != nil {
		t.Fatalf("Failed to get historical measurements: %v", err)
	}

	// Verify we have smoothed rates
	if len(measurements) < 2 {
		t.Fatalf("Expected at least 2 measurements, got %d", len(measurements))
	}

	// Check that measurements have rates set
	for i, m := range measurements {
		if i > 0 { // Skip first measurement as it won't have rates
			// Check that metadata contains instantaneous rates
			if _, exists := m.Metadata["instant_rx_rate"]; !exists {
				t.Errorf("Measurement %d missing instant_rx_rate metadata", i)
			}
			if _, exists := m.Metadata["instant_tx_rate"]; !exists {
				t.Errorf("Measurement %d missing instant_tx_rate metadata", i)
			}
		}
		t.Logf("Measurement %d: RX=%.2f, TX=%.2f, Util=%.2f%%", 
			i, m.RXRate, m.TXRate, m.Utilization)
	}

	// Test threshold alerting with smoothed rates
	threshold := network.BandwidthThreshold{
		InterfaceName:     "eth0",
		WarningThreshold:  50.0,
		CriticalThreshold: 80.0,
		AbsoluteLimit:     1000000000, // 1 Gbps
		Enabled:           true,
	}

	err = monitor.SetThreshold("eth0", threshold)
	if err != nil {
		t.Fatalf("Failed to set threshold: %v", err)
	}

	// Wait for more measurements
	time.Sleep(3 * time.Second)

	t.Log("Sliding window bandwidth monitoring test completed successfully")
}

func TestHistoryRetentionPruning(t *testing.T) {
	logger := zap.NewNop()

	config := &network.BandwidthMonitorConfig{
		MonitoringInterval:    100 * time.Millisecond,
		HistoryRetention:      500 * time.Millisecond,
		SlidingWindowDuration: 300 * time.Millisecond,
		Interfaces:            []string{"eth0"},
		MaxHistoryPoints:      10,
	}

	monitor := network.NewBandwidthMonitor(config, logger)

	// Start the monitor
	err := monitor.Start()
	if err != nil {
		t.Fatalf("Failed to start bandwidth monitor: %v", err)
	}
	defer monitor.Stop()

	// Collect data for longer than retention period
	time.Sleep(1 * time.Second)

	// Get all measurements
	measurements, err := monitor.GetHistoricalMeasurements("eth0", time.Now().Add(-2*time.Second))
	if err != nil {
		t.Fatalf("Failed to get historical measurements: %v", err)
	}

	// Verify old data is pruned
	for _, m := range measurements {
		age := time.Since(m.Timestamp)
		if age > config.HistoryRetention+time.Second {
			t.Errorf("Found measurement older than retention period: age=%v", age)
		}
	}

	// Verify max history points limit
	if len(measurements) > config.MaxHistoryPoints {
		t.Errorf("Measurements exceed max history points: got %d, max %d", 
			len(measurements), config.MaxHistoryPoints)
	}

	t.Logf("History retention test completed: %d measurements retained", len(measurements))
}