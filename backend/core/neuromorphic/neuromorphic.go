package neuromorphic

import (
	"context"
	"fmt"
	"sync"

	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/benchmark"
	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/edge"
	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/energy"
	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/hardware"
	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/metrics"
	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/snn"
)

// NeuromorphicManager manages neuromorphic computing components
type NeuromorphicManager struct {
	mu               sync.RWMutex
	config           *NeuromorphicConfig
	hardwareManager  *hardware.HardwareManager
	energyMonitor    *energy.EnergyMonitor
	edgeDeployer     *edge.EdgeDeployer
	metricsCollector *metrics.MetricsCollector
	benchmarkSuite   *benchmark.BenchmarkSuite
	networks         map[string]*snn.SNNNetwork
	enabled          bool
}

// NewNeuromorphicManager creates a new neuromorphic manager
func NewNeuromorphicManager(config *NeuromorphicConfig) (*NeuromorphicManager, error) {
	if config == nil {
		config = DefaultNeuromorphicConfig()
	}

	nm := &NeuromorphicManager{
		config:           config,
		hardwareManager:  hardware.NewHardwareManager(),
		energyMonitor:    energy.NewEnergyMonitor(config.EnergyConfig.SamplingRate),
		edgeDeployer:     edge.NewEdgeDeployer(),
		metricsCollector: metrics.NewMetricsCollector(),
		benchmarkSuite:   benchmark.NewBenchmarkSuite(),
		networks:         make(map[string]*snn.SNNNetwork),
		enabled:          config.EnableSNN,
	}

	// Register neuromorphic hardware if specified
	if config.HardwareType != "" {
		err := nm.registerHardware(config.HardwareType)
		if err != nil {
			return nil, fmt.Errorf("failed to register hardware: %w", err)
		}
	}

	return nm, nil
}

// registerHardware registers neuromorphic hardware
func (nm *NeuromorphicManager) registerHardware(hwType string) error {
	ctx := context.Background()

	var hardwareType hardware.HardwareType
	switch hwType {
	case HardwareLoihi2:
		hardwareType = hardware.Loihi2
	case HardwareTrueNorth:
		hardwareType = hardware.TrueNorth
	case HardwareAkida:
		hardwareType = hardware.Akida
	case HardwareSpinnaker:
		hardwareType = hardware.Spinnaker
	case HardwareNeurogrid:
		hardwareType = hardware.Neurogrid
	default:
		return fmt.Errorf("unsupported hardware type: %s", hwType)
	}

	deviceID := fmt.Sprintf("%s-device-0", hwType)
	return nm.hardwareManager.RegisterDevice(ctx, hardwareType, deviceID)
}

// CreateNetwork creates a new SNN network
func (nm *NeuromorphicManager) CreateNetwork(networkID string, neuronModel snn.NeuronModel) (*snn.SNNNetwork, error) {
	nm.mu.Lock()
	defer nm.mu.Unlock()

	if !nm.enabled {
		return nil, fmt.Errorf("neuromorphic computing is disabled")
	}

	stdpConfig := &snn.STDPConfig{
		Enable:   nm.config.SNNConfig.EnablePlasticity,
		TauPlus:  20.0,
		TauMinus: 20.0,
		APlus:    0.01,
		AMinus:   0.012,
	}

	network := snn.NewSNNNetwork(nm.config.SNNConfig.TimeStep, stdpConfig)
	nm.networks[networkID] = network

	return network, nil
}

// GetNetwork returns an existing network
func (nm *NeuromorphicManager) GetNetwork(networkID string) (*snn.SNNNetwork, error) {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	network, exists := nm.networks[networkID]
	if !exists {
		return nil, fmt.Errorf("network not found: %s", networkID)
	}

	return network, nil
}

// RunInference runs inference on neuromorphic hardware
func (nm *NeuromorphicManager) RunInference(ctx context.Context, networkID string, input []float64) ([]snn.Spike, error) {
	network, err := nm.GetNetwork(networkID)
	if err != nil {
		return nil, err
	}

	// Encode input to spikes
	inputSpikes := nm.encodeInput(input)

	// Send spikes to hardware
	if nm.hardwareManager != nil {
		err = nm.hardwareManager.SendSpikes(ctx, convertSpikes(inputSpikes))
		if err != nil {
			return nil, err
		}
	}

	// Run simulation
	outputSpikes, err := network.Run(ctx, nm.config.SNNConfig.SimulationTime, func(t float64) []snn.Spike {
		if t < 1.0 {
			return inputSpikes
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	// Record metrics
	nm.energyMonitor.RecordInference()

	return outputSpikes, nil
}

// encodeInput encodes input to spikes based on encoding strategy
func (nm *NeuromorphicManager) encodeInput(input []float64) []snn.Spike {
	encoding := nm.config.SNNConfig.SpikeEncoding
	spikes := make([]snn.Spike, 0)

	switch encoding {
	case EncodingRate:
		// Rate coding
		for i, val := range input {
			numSpikes := int(val * 10)
			for j := 0; j < numSpikes; j++ {
				spikes = append(spikes, snn.Spike{
					NeuronID:  int64(i),
					Timestamp: float64(j) * 2.0,
					Weight:    1.0,
				})
			}
		}

	case EncodingTemporal:
		// Temporal coding
		for i, val := range input {
			timing := 10.0 * (1.0 - val)
			spikes = append(spikes, snn.Spike{
				NeuronID:  int64(i),
				Timestamp: timing,
				Weight:    1.0,
			})
		}

	case EncodingPhase:
		// Phase coding
		period := 10.0
		for i, val := range input {
			phase := val * period
			spikes = append(spikes, snn.Spike{
				NeuronID:  int64(i),
				Timestamp: phase,
				Weight:    1.0,
			})
		}
	}

	return spikes
}

// DeployToEdge deploys a network to an edge device
func (nm *NeuromorphicManager) DeployToEdge(ctx context.Context, networkID, deviceID string) (*edge.Deployment, error) {
	_, err := nm.GetNetwork(networkID)
	if err != nil {
		return nil, err
	}

	compressionLevel := nm.config.EdgeConfig.CompressionLevel
	deployment, err := nm.edgeDeployer.DeployModel(ctx, deviceID, networkID, compressionLevel)
	if err != nil {
		return nil, err
	}

	return deployment, nil
}

// RunBenchmarks runs all neuromorphic benchmarks
func (nm *NeuromorphicManager) RunBenchmarks(ctx context.Context, networkID string) error {
	network, err := nm.GetNetwork(networkID)
	if err != nil {
		return err
	}

	return nm.benchmarkSuite.RunAllBenchmarks(ctx, network)
}

// GetMetrics returns comprehensive neuromorphic metrics
func (nm *NeuromorphicManager) GetMetrics() map[string]interface{} {
	nm.mu.RLock()
	defer nm.mu.RUnlock()

	metrics := make(map[string]interface{})

	// Energy metrics
	energyMetrics := nm.energyMonitor.GetMetrics()
	metrics["energy"] = energyMetrics

	// Network metrics
	metrics["total_networks"] = len(nm.networks)

	// Benchmark results
	benchmarkResults := nm.benchmarkSuite.GetResults()
	metrics["benchmark_results"] = benchmarkResults

	// Hardware metrics (if available)
	if nm.hardwareManager != nil {
		device, err := nm.hardwareManager.GetActiveDevice()
		if err == nil {
			metrics["hardware"] = map[string]interface{}{
				"type":        device.Type,
				"status":      device.Status,
				"temperature": device.Temperature,
				"power_usage": device.PowerUsage,
				"utilization": device.Utilization,
			}
		}
	}

	return metrics
}

// GetStatus returns the status of the neuromorphic system
func (nm *NeuromorphicManager) GetStatus() map[string]interface{} {
	return map[string]interface{}{
		"enabled":          nm.enabled,
		"hardware_type":    nm.config.HardwareType,
		"networks":         len(nm.networks),
		"edge_devices":     len(nm.edgeDeployer.ListDevices()),
		"power_budget":     nm.config.PowerBudget,
		"target_latency":   nm.config.TargetLatency.Microseconds(),
		"accuracy_threshold": nm.config.AccuracyThreshold,
	}
}

// Close closes the neuromorphic manager
func (nm *NeuromorphicManager) Close() error {
	if nm.hardwareManager != nil {
		nm.hardwareManager.Close()
	}
	if nm.energyMonitor != nil {
		nm.energyMonitor.Close()
	}
	if nm.edgeDeployer != nil {
		nm.edgeDeployer.Close()
	}
	return nil
}

// Helper function to convert spikes
func convertSpikes(spikes []snn.Spike) []hardware.Spike {
	hwSpikes := make([]hardware.Spike, len(spikes))
	for i, spike := range spikes {
		hwSpikes[i] = hardware.Spike{
			NeuronID:  spike.NeuronID,
			Timestamp: spike.Timestamp,
			Weight:    spike.Weight,
		}
	}
	return hwSpikes
}
