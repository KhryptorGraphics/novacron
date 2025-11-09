package neuromorphic

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/snn"
)

func TestNeuromorphicManager(t *testing.T) {
	config := DefaultNeuromorphicConfig()
	config.HardwareType = HardwareLoihi2

	nm, err := NewNeuromorphicManager(config)
	if err != nil {
		t.Fatalf("Failed to create neuromorphic manager: %v", err)
	}
	defer nm.Close()

	// Test network creation
	network, err := nm.CreateNetwork("test-network", snn.LIF)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	if network == nil {
		t.Fatal("Network is nil")
	}

	// Test network retrieval
	retrieved, err := nm.GetNetwork("test-network")
	if err != nil {
		t.Fatalf("Failed to get network: %v", err)
	}

	if retrieved != network {
		t.Fatal("Retrieved network doesn't match created network")
	}
}

func TestNeuromorphicInference(t *testing.T) {
	config := DefaultNeuromorphicConfig()
	nm, err := NewNeuromorphicManager(config)
	if err != nil {
		t.Fatalf("Failed to create neuromorphic manager: %v", err)
	}
	defer nm.Close()

	// Create network
	network, err := nm.CreateNetwork("inference-test", snn.LIF)
	if err != nil {
		t.Fatalf("Failed to create network: %v", err)
	}

	// Add neurons
	for i := 0; i < 10; i++ {
		network.AddNeuron(snn.LIF)
	}

	// Run inference
	ctx := context.Background()
	input := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
	spikes, err := nm.RunInference(ctx, "inference-test", input)
	if err != nil {
		t.Fatalf("Failed to run inference: %v", err)
	}

	if len(spikes) == 0 {
		t.Log("Warning: No output spikes generated (this may be normal for small networks)")
	}

	t.Logf("Inference completed: %d output spikes", len(spikes))
}

func TestEnergyMonitoring(t *testing.T) {
	config := DefaultNeuromorphicConfig()
	nm, err := NewNeuromorphicManager(config)
	if err != nil {
		t.Fatalf("Failed to create neuromorphic manager: %v", err)
	}
	defer nm.Close()

	// Create network and run inferences
	network, _ := nm.CreateNetwork("energy-test", snn.LIF)
	for i := 0; i < 10; i++ {
		network.AddNeuron(snn.LIF)
	}

	ctx := context.Background()
	input := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

	for i := 0; i < 100; i++ {
		_, err := nm.RunInference(ctx, "energy-test", input)
		if err != nil {
			t.Fatalf("Failed to run inference: %v", err)
		}
	}

	// Get energy metrics
	metrics := nm.GetMetrics()
	energyMetrics, ok := metrics["energy"]
	if !ok {
		t.Fatal("Energy metrics not found")
	}

	t.Logf("Energy metrics: %+v", energyMetrics)
}

func TestBenchmarking(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping benchmark test in short mode")
	}

	config := DefaultNeuromorphicConfig()
	nm, err := NewNeuromorphicManager(config)
	if err != nil {
		t.Fatalf("Failed to create neuromorphic manager: %v", err)
	}
	defer nm.Close()

	// Create network
	network, _ := nm.CreateNetwork("benchmark-test", snn.LIF)
	for i := 0; i < 100; i++ {
		network.AddNeuron(snn.LIF)
	}

	// Run benchmarks
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err = nm.RunBenchmarks(ctx, "benchmark-test")
	if err != nil {
		t.Fatalf("Failed to run benchmarks: %v", err)
	}

	// Check results
	metrics := nm.GetMetrics()
	results, ok := metrics["benchmark_results"]
	if !ok {
		t.Fatal("Benchmark results not found")
	}

	t.Logf("Benchmark results: %+v", results)
}

func TestHardwareIntegration(t *testing.T) {
	config := DefaultNeuromorphicConfig()
	config.HardwareType = HardwareLoihi2

	nm, err := NewNeuromorphicManager(config)
	if err != nil {
		t.Fatalf("Failed to create neuromorphic manager: %v", err)
	}
	defer nm.Close()

	// Check hardware status
	status := nm.GetStatus()
	if status["hardware_type"] != HardwareLoihi2 {
		t.Fatalf("Expected hardware type %s, got %s", HardwareLoihi2, status["hardware_type"])
	}

	metrics := nm.GetMetrics()
	hwMetrics, ok := metrics["hardware"]
	if !ok {
		t.Fatal("Hardware metrics not found")
	}

	t.Logf("Hardware metrics: %+v", hwMetrics)
}

func TestSpikeEncoding(t *testing.T) {
	config := DefaultNeuromorphicConfig()

	tests := []struct {
		name     string
		encoding string
	}{
		{"Rate Encoding", EncodingRate},
		{"Temporal Encoding", EncodingTemporal},
		{"Phase Encoding", EncodingPhase},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config.SNNConfig.SpikeEncoding = tt.encoding
			nm, err := NewNeuromorphicManager(config)
			if err != nil {
				t.Fatalf("Failed to create neuromorphic manager: %v", err)
			}
			defer nm.Close()

			input := []float64{0.5, 0.7, 0.3, 0.9}
			spikes := nm.encodeInput(input)

			if len(spikes) == 0 {
				t.Fatal("No spikes generated")
			}

			t.Logf("%s: Generated %d spikes", tt.name, len(spikes))
		})
	}
}

func TestMetricsCollection(t *testing.T) {
	config := DefaultNeuromorphicConfig()
	nm, err := NewNeuromorphicManager(config)
	if err != nil {
		t.Fatalf("Failed to create neuromorphic manager: %v", err)
	}
	defer nm.Close()

	// Run some inferences
	network, _ := nm.CreateNetwork("metrics-test", snn.LIF)
	for i := 0; i < 10; i++ {
		network.AddNeuron(snn.LIF)
	}

	ctx := context.Background()
	input := []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

	for i := 0; i < 50; i++ {
		_, err := nm.RunInference(ctx, "metrics-test", input)
		if err != nil {
			t.Fatalf("Failed to run inference: %v", err)
		}
	}

	// Get metrics
	metrics := nm.GetMetrics()

	requiredKeys := []string{"energy", "total_networks"}
	for _, key := range requiredKeys {
		if _, ok := metrics[key]; !ok {
			t.Errorf("Missing required metric: %s", key)
		}
	}

	t.Logf("Collected metrics: %+v", metrics)
}

func TestNeuromorphicStatus(t *testing.T) {
	config := DefaultNeuromorphicConfig()
	nm, err := NewNeuromorphicManager(config)
	if err != nil {
		t.Fatalf("Failed to create neuromorphic manager: %v", err)
	}
	defer nm.Close()

	status := nm.GetStatus()

	if !status["enabled"].(bool) {
		t.Error("Neuromorphic computing should be enabled")
	}

	if status["networks"].(int) != 0 {
		t.Errorf("Expected 0 networks, got %d", status["networks"])
	}

	// Create a network and check again
	nm.CreateNetwork("status-test", snn.LIF)

	status = nm.GetStatus()
	if status["networks"].(int) != 1 {
		t.Errorf("Expected 1 network, got %d", status["networks"])
	}
}

func BenchmarkInference(b *testing.B) {
	config := DefaultNeuromorphicConfig()
	nm, _ := NewNeuromorphicManager(config)
	defer nm.Close()

	network, _ := nm.CreateNetwork("benchmark", snn.LIF)
	for i := 0; i < 100; i++ {
		network.AddNeuron(snn.LIF)
	}

	ctx := context.Background()
	input := make([]float64, 100)
	for i := range input {
		input[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nm.RunInference(ctx, "benchmark", input)
	}
}

func BenchmarkSpikeEncoding(b *testing.B) {
	config := DefaultNeuromorphicConfig()
	nm, _ := NewNeuromorphicManager(config)
	defer nm.Close()

	input := make([]float64, 1000)
	for i := range input {
		input[i] = float64(i) / 1000.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nm.encodeInput(input)
	}
}
