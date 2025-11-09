package neuromorphic

import (
	"time"
)

// NeuromorphicConfig defines the neuromorphic computing configuration
type NeuromorphicConfig struct {
	// Hardware configuration
	HardwareType        string  `json:"hardware_type"`         // "loihi2", "truenorth", "akida", "spinnaker", "neurogrid"
	EnableSNN           bool    `json:"enable_snn"`
	PowerBudget         float64 `json:"power_budget"`          // watts
	TargetLatency       time.Duration `json:"target_latency"`  // target inference latency
	AccuracyThreshold   float64 `json:"accuracy_threshold"`    // minimum accuracy (0.0-1.0)
	EdgeDeployment      bool    `json:"edge_deployment"`

	// SNN configuration
	SNNConfig SNNConfig `json:"snn_config"`

	// Bio-inspired algorithms
	BioInspiredConfig BioInspiredConfig `json:"bioinspired_config"`

	// Edge deployment
	EdgeConfig EdgeConfig `json:"edge_config"`

	// Energy management
	EnergyConfig EnergyConfig `json:"energy_config"`
}

// SNNConfig defines SNN-specific configuration
type SNNConfig struct {
	NeuronModel         string  `json:"neuron_model"`          // "lif", "izhikevich", "hodgkin-huxley"
	LearningAlgorithm   string  `json:"learning_algorithm"`    // "stdp", "r-stdp", "supervised-stdp"
	SpikeEncoding       string  `json:"spike_encoding"`        // "rate", "temporal", "phase"
	TimeStep            float64 `json:"time_step"`             // ms
	SimulationTime      float64 `json:"simulation_time"`       // ms
	EnablePlasticity    bool    `json:"enable_plasticity"`
	EnableHomeostasis   bool    `json:"enable_homeostasis"`
}

// BioInspiredConfig defines bio-inspired algorithm configuration
type BioInspiredConfig struct {
	EnableAntColony     bool    `json:"enable_ant_colony"`
	EnableParticleSwarm bool    `json:"enable_particle_swarm"`
	EnableGenetic       bool    `json:"enable_genetic"`
	EnableImmuneSystem  bool    `json:"enable_immune_system"`
	EnableSwarmIntel    bool    `json:"enable_swarm_intel"`
	PopulationSize      int     `json:"population_size"`
	MaxIterations       int     `json:"max_iterations"`
}

// EdgeConfig defines edge deployment configuration
type EdgeConfig struct {
	AutoDeploy          bool    `json:"auto_deploy"`
	CompressionLevel    string  `json:"compression_level"`     // "none", "low", "medium", "high", "ultra"
	PowerMode           string  `json:"power_mode"`            // "normal", "low-power", "ultra-low"
	ThermalLimit        float64 `json:"thermal_limit"`         // celsius
	EnableOTA           bool    `json:"enable_ota"`
	UpdateInterval      time.Duration `json:"update_interval"`
}

// EnergyConfig defines energy monitoring configuration
type EnergyConfig struct {
	EnableMonitoring    bool    `json:"enable_monitoring"`
	SamplingRate        time.Duration `json:"sampling_rate"`
	EnergyTarget        float64 `json:"energy_target"`         // joules per inference
	EnablePowerGating   bool    `json:"enable_power_gating"`
	EnableDVFS          bool    `json:"enable_dvfs"`           // Dynamic Voltage and Frequency Scaling
}

// DefaultNeuromorphicConfig returns default configuration
func DefaultNeuromorphicConfig() *NeuromorphicConfig {
	return &NeuromorphicConfig{
		HardwareType:      "loihi2",
		EnableSNN:         true,
		PowerBudget:       1.0, // 1 watt
		TargetLatency:     1 * time.Millisecond,
		AccuracyThreshold: 0.95,
		EdgeDeployment:    true,

		SNNConfig: SNNConfig{
			NeuronModel:       "lif",
			LearningAlgorithm: "stdp",
			SpikeEncoding:     "temporal",
			TimeStep:          1.0,
			SimulationTime:    100.0,
			EnablePlasticity:  true,
			EnableHomeostasis: true,
		},

		BioInspiredConfig: BioInspiredConfig{
			EnableAntColony:     true,
			EnableParticleSwarm: true,
			EnableGenetic:       true,
			EnableImmuneSystem:  false,
			EnableSwarmIntel:    true,
			PopulationSize:      100,
			MaxIterations:       1000,
		},

		EdgeConfig: EdgeConfig{
			AutoDeploy:       true,
			CompressionLevel: "medium",
			PowerMode:        "low-power",
			ThermalLimit:     85.0,
			EnableOTA:        true,
			UpdateInterval:   24 * time.Hour,
		},

		EnergyConfig: EnergyConfig{
			EnableMonitoring: true,
			SamplingRate:     100 * time.Millisecond,
			EnergyTarget:     0.001, // 1 millijoule
			EnablePowerGating: true,
			EnableDVFS:       true,
		},
	}
}

// HardwareType constants
const (
	HardwareLoihi2    = "loihi2"
	HardwareTrueNorth = "truenorth"
	HardwareAkida     = "akida"
	HardwareSpinnaker = "spinnaker"
	HardwareNeurogrid = "neurogrid"
)

// NeuronModel constants
const (
	NeuronLIF          = "lif"
	NeuronIzhikevich   = "izhikevich"
	NeuronHodgkinHuxley = "hodgkin-huxley"
)

// SpikeEncoding constants
const (
	EncodingRate     = "rate"
	EncodingTemporal = "temporal"
	EncodingPhase    = "phase"
)

// PowerMode constants
const (
	PowerNormal    = "normal"
	PowerLow       = "low-power"
	PowerUltraLow  = "ultra-low"
)
