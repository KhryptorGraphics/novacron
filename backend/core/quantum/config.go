package quantum

import (
	"time"
)

// QuantumConfig represents the configuration for quantum computing integration
type QuantumConfig struct {
	// Core settings
	EnableQuantum       bool   `json:"enable_quantum" yaml:"enable_quantum"`
	PreferredProvider   string `json:"preferred_provider" yaml:"preferred_provider"` // "ibm", "google", "aws", "rigetti", "ionq", "simulator"
	SimulatorFallback   bool   `json:"simulator_fallback" yaml:"simulator_fallback"`
	MaxQubits           int    `json:"max_qubits" yaml:"max_qubits"`
	CircuitTimeout      time.Duration `json:"circuit_timeout" yaml:"circuit_timeout"`

	// Algorithm settings
	EnableShorAlgorithm    bool `json:"enable_shor_algorithm" yaml:"enable_shor_algorithm"`
	EnableGroverSearch     bool `json:"enable_grover_search" yaml:"enable_grover_search"`
	EnableVQE              bool `json:"enable_vqe" yaml:"enable_vqe"`
	EnableQAOA             bool `json:"enable_qaoa" yaml:"enable_qaoa"`
	EnableQuantumML        bool `json:"enable_quantum_ml" yaml:"enable_quantum_ml"`

	// Security settings
	UseQKD                 bool   `json:"use_qkd" yaml:"use_qkd"`
	QKDProtocol            string `json:"qkd_protocol" yaml:"qkd_protocol"` // "bb84", "e91", "decoy"
	ErrorCorrection        bool   `json:"error_correction" yaml:"error_correction"`
	ErrorCorrectionCode    string `json:"error_correction_code" yaml:"error_correction_code"` // "surface", "shor9", "steane"

	// Compiler settings
	OptimizationLevel      int  `json:"optimization_level" yaml:"optimization_level"` // 0-3
	GateReduction          bool `json:"gate_reduction" yaml:"gate_reduction"`
	CircuitDecomposition   bool `json:"circuit_decomposition" yaml:"circuit_decomposition"`

	// Simulator settings
	SimulatorType          string `json:"simulator_type" yaml:"simulator_type"` // "state_vector", "density_matrix", "mps", "clifford_t"
	MaxSimulatedQubits     int    `json:"max_simulated_qubits" yaml:"max_simulated_qubits"`
	UseGPUAcceleration     bool   `json:"use_gpu_acceleration" yaml:"use_gpu_acceleration"`
	NoiseModel             string `json:"noise_model" yaml:"noise_model"` // "ideal", "depolarizing", "amplitude_damping"

	// Provider settings
	IBMToken               string `json:"ibm_token,omitempty" yaml:"ibm_token,omitempty"`
	IBMBackend             string `json:"ibm_backend" yaml:"ibm_backend"`
	GoogleProjectID        string `json:"google_project_id,omitempty" yaml:"google_project_id,omitempty"`
	AWSAccessKey           string `json:"aws_access_key,omitempty" yaml:"aws_access_key,omitempty"`
	AWSSecretKey           string `json:"aws_secret_key,omitempty" yaml:"aws_secret_key,omitempty"`
	RigettiAPIKey          string `json:"rigetti_api_key,omitempty" yaml:"rigetti_api_key,omitempty"`
	IonQAPIKey             string `json:"ionq_api_key,omitempty" yaml:"ionq_api_key,omitempty"`

	// Performance settings
	DefaultShots           int           `json:"default_shots" yaml:"default_shots"`
	MaxConcurrentCircuits  int           `json:"max_concurrent_circuits" yaml:"max_concurrent_circuits"`
	CacheResults           bool          `json:"cache_results" yaml:"cache_results"`
	CacheTTL               time.Duration `json:"cache_ttl" yaml:"cache_ttl"`

	// Cost management
	MaxCostPerCircuit      float64 `json:"max_cost_per_circuit" yaml:"max_cost_per_circuit"` // USD
	MonitorCosts           bool    `json:"monitor_costs" yaml:"monitor_costs"`

	// Benchmarking
	EnableBenchmarking     bool `json:"enable_benchmarking" yaml:"enable_benchmarking"`
	RandomizedBenchmarking bool `json:"randomized_benchmarking" yaml:"randomized_benchmarking"`
}

// DefaultQuantumConfig returns default quantum configuration
func DefaultQuantumConfig() *QuantumConfig {
	return &QuantumConfig{
		EnableQuantum:          true,
		PreferredProvider:      "simulator",
		SimulatorFallback:      true,
		MaxQubits:              100,
		CircuitTimeout:         30 * time.Second,

		EnableShorAlgorithm:    true,
		EnableGroverSearch:     true,
		EnableVQE:              true,
		EnableQAOA:             true,
		EnableQuantumML:        true,

		UseQKD:                 false, // Requires special hardware
		QKDProtocol:            "bb84",
		ErrorCorrection:        true,
		ErrorCorrectionCode:    "surface",

		OptimizationLevel:      2,
		GateReduction:          true,
		CircuitDecomposition:   true,

		SimulatorType:          "state_vector",
		MaxSimulatedQubits:     30,
		UseGPUAcceleration:     true,
		NoiseModel:             "depolarizing",

		IBMBackend:             "ibmq_qasm_simulator",

		DefaultShots:           1024,
		MaxConcurrentCircuits:  10,
		CacheResults:           true,
		CacheTTL:               15 * time.Minute,

		MaxCostPerCircuit:      10.0, // $10 per circuit max
		MonitorCosts:           true,

		EnableBenchmarking:     true,
		RandomizedBenchmarking: true,
	}
}

// ProductionQuantumConfig returns production-optimized configuration
func ProductionQuantumConfig() *QuantumConfig {
	config := DefaultQuantumConfig()
	config.PreferredProvider = "ibm" // Use real quantum hardware
	config.MaxQubits = 127 // IBM Quantum Eagle
	config.MaxSimulatedQubits = 30
	config.OptimizationLevel = 3
	config.ErrorCorrection = true
	config.MaxCostPerCircuit = 50.0
	config.DefaultShots = 8192 // More shots for production
	return config
}

// DevelopmentQuantumConfig returns development-optimized configuration
func DevelopmentQuantumConfig() *QuantumConfig {
	config := DefaultQuantumConfig()
	config.PreferredProvider = "simulator"
	config.MaxQubits = 30
	config.NoiseModel = "ideal" // No noise for development
	config.DefaultShots = 1024
	config.OptimizationLevel = 1
	config.MaxCostPerCircuit = 0.0 // Free simulator
	return config
}

// ValidateConfig validates quantum configuration
func (qc *QuantumConfig) ValidateConfig() error {
	if qc.MaxQubits < 1 {
		return ErrInvalidQubitCount
	}

	if qc.MaxQubits > 1000 {
		return ErrTooManyQubits
	}

	if qc.OptimizationLevel < 0 || qc.OptimizationLevel > 3 {
		return ErrInvalidOptimizationLevel
	}

	if qc.DefaultShots < 1 {
		return ErrInvalidShotCount
	}

	if qc.PreferredProvider == "" {
		qc.PreferredProvider = "simulator"
	}

	// Validate provider-specific settings
	switch qc.PreferredProvider {
	case "ibm":
		if qc.IBMToken == "" && qc.IBMBackend != "ibmq_qasm_simulator" {
			return ErrMissingProviderCredentials
		}
	case "google":
		if qc.GoogleProjectID == "" {
			return ErrMissingProviderCredentials
		}
	case "aws":
		if qc.AWSAccessKey == "" || qc.AWSSecretKey == "" {
			return ErrMissingProviderCredentials
		}
	case "rigetti":
		if qc.RigettiAPIKey == "" {
			return ErrMissingProviderCredentials
		}
	case "ionq":
		if qc.IonQAPIKey == "" {
			return ErrMissingProviderCredentials
		}
	case "simulator":
		// No validation needed for simulator
	default:
		return ErrUnsupportedProvider
	}

	return nil
}
