package autonomous

import (
	"time"
)

// AutonomousConfig defines configuration for the autonomous self-healing and evolution system
type AutonomousConfig struct {
	// Healing configuration
	EnableHealing          bool          `yaml:"enable_healing"`
	HealingInterval        time.Duration `yaml:"healing_interval"`
	MaxHealingAttempts     int           `yaml:"max_healing_attempts"`
	HealingTimeout         time.Duration `yaml:"healing_timeout"`
	SubSecondDetection     bool          `yaml:"sub_second_detection"`

	// Prediction configuration
	EnablePrediction       bool          `yaml:"enable_prediction"`
	PredictionHorizon      time.Duration `yaml:"prediction_horizon"`      // 72h
	PredictionInterval     time.Duration `yaml:"prediction_interval"`
	AnomalyThreshold       float64       `yaml:"anomaly_threshold"`
	MinPredictionAccuracy  float64       `yaml:"min_prediction_accuracy"` // 0.95

	// Evolution configuration
	EnableEvolution        bool          `yaml:"enable_evolution"`
	EvolutionGenerations   int           `yaml:"evolution_generations"`   // 500
	PopulationSize         int           `yaml:"population_size"`
	MutationRate           float64       `yaml:"mutation_rate"`
	CrossoverRate          float64       `yaml:"crossover_rate"`
	EliteRatio             float64       `yaml:"elite_ratio"`

	// Code generation configuration
	EnableCodeGen          bool          `yaml:"enable_code_gen"`
	CodeQualityThreshold   float64       `yaml:"code_quality_threshold"`  // 0.9
	GPT4APIKey             string        `yaml:"gpt4_api_key"`
	MaxCodeGenAttempts     int           `yaml:"max_codegen_attempts"`
	AutoDeploy             bool          `yaml:"auto_deploy"`
	CanaryDeployRatio      float64       `yaml:"canary_deploy_ratio"`

	// Digital twin configuration
	EnableDigitalTwin      bool          `yaml:"enable_digital_twin"`
	SimulationSpeed        int           `yaml:"simulation_speed"`        // 100x
	FutureStatePrediction  time.Duration `yaml:"future_state_prediction"` // 24h
	WhatIfScenarios        int           `yaml:"what_if_scenarios"`

	// Learning configuration
	EnableLearning         bool          `yaml:"enable_learning"`
	LearningRate           float64       `yaml:"learning_rate"`
	ReplayBufferSize       int           `yaml:"replay_buffer_size"`
	ExplorationRate        float64       `yaml:"exploration_rate"`
	RLHFEnabled            bool          `yaml:"rlhf_enabled"`
	TransferLearning       bool          `yaml:"transfer_learning"`

	// A/B testing configuration
	EnableABTesting        bool          `yaml:"enable_ab_testing"`
	SignificanceLevel      float64       `yaml:"significance_level"`
	MinSampleSize          int           `yaml:"min_sample_size"`
	MaxTestDuration        time.Duration `yaml:"max_test_duration"`
	AutoRollout            bool          `yaml:"auto_rollout"`

	// Self-optimization configuration
	EnableOptimization     bool          `yaml:"enable_optimization"`
	OptimizationFrequency  time.Duration `yaml:"optimization_frequency"`   // Daily
	BayesianIterations     int           `yaml:"bayesian_iterations"`
	ParetoObjectives       []string      `yaml:"pareto_objectives"`

	// Incident response configuration
	EnableIncidentResponse bool          `yaml:"enable_incident_response"`
	MTTD                   time.Duration `yaml:"mttd"`                    // <10s
	MTTR                   time.Duration `yaml:"mttr"`                    // <1min
	AutoPostMortem         bool          `yaml:"auto_post_mortem"`
	EscalationThreshold    string        `yaml:"escalation_threshold"`    // P0

	// Safety and control
	HumanApprovalRequired  bool          `yaml:"human_approval_required"` // false for full autonomy
	SafetyMode             bool          `yaml:"safety_mode"`
	RollbackEnabled        bool          `yaml:"rollback_enabled"`
	DryRunMode             bool          `yaml:"dry_run_mode"`
	MaxConcurrentActions   int           `yaml:"max_concurrent_actions"`
}

// DefaultConfig returns the default autonomous configuration
func DefaultConfig() *AutonomousConfig {
	return &AutonomousConfig{
		// Healing
		EnableHealing:          true,
		HealingInterval:        100 * time.Millisecond, // Sub-second
		MaxHealingAttempts:     3,
		HealingTimeout:         30 * time.Second,
		SubSecondDetection:     true,

		// Prediction
		EnablePrediction:       true,
		PredictionHorizon:      72 * time.Hour,
		PredictionInterval:     5 * time.Minute,
		AnomalyThreshold:       0.85,
		MinPredictionAccuracy:  0.95,

		// Evolution
		EnableEvolution:        true,
		EvolutionGenerations:   500,
		PopulationSize:         100,
		MutationRate:           0.1,
		CrossoverRate:          0.7,
		EliteRatio:             0.1,

		// Code generation
		EnableCodeGen:          true,
		CodeQualityThreshold:   0.9,
		MaxCodeGenAttempts:     5,
		AutoDeploy:             true,
		CanaryDeployRatio:      0.1,

		// Digital twin
		EnableDigitalTwin:      true,
		SimulationSpeed:        100,
		FutureStatePrediction:  24 * time.Hour,
		WhatIfScenarios:        10,

		// Learning
		EnableLearning:         true,
		LearningRate:           0.001,
		ReplayBufferSize:       10000,
		ExplorationRate:        0.1,
		RLHFEnabled:            true,
		TransferLearning:       true,

		// A/B testing
		EnableABTesting:        true,
		SignificanceLevel:      0.05,
		MinSampleSize:          1000,
		MaxTestDuration:        7 * 24 * time.Hour,
		AutoRollout:            true,

		// Self-optimization
		EnableOptimization:     true,
		OptimizationFrequency:  24 * time.Hour,
		BayesianIterations:     100,
		ParetoObjectives:       []string{"performance", "cost", "reliability"},

		// Incident response
		EnableIncidentResponse: true,
		MTTD:                   10 * time.Second,
		MTTR:                   1 * time.Minute,
		AutoPostMortem:         true,
		EscalationThreshold:    "P0",

		// Safety
		HumanApprovalRequired:  false, // Full autonomy
		SafetyMode:             false,
		RollbackEnabled:        true,
		DryRunMode:             false,
		MaxConcurrentActions:   10,
	}
}

// IncidentPriority defines incident priority levels
type IncidentPriority int

const (
	P0 IncidentPriority = iota // Catastrophic
	P1                          // Critical
	P2                          // Major
	P3                          // Minor
	P4                          // Informational
)

// HealingAction defines types of healing actions
type HealingAction string

const (
	ServiceRestart     HealingAction = "service_restart"
	VMMigration        HealingAction = "vm_migration"
	ScaleOut           HealingAction = "scale_out"
	ScaleIn            HealingAction = "scale_in"
	ConfigRollback     HealingAction = "config_rollback"
	NetworkReroute     HealingAction = "network_reroute"
	ResourceRebalance  HealingAction = "resource_rebalance"
	SecurityPatch      HealingAction = "security_patch"
	PerformanceTune    HealingAction = "performance_tune"
	DataRecovery       HealingAction = "data_recovery"
)