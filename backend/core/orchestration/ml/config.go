// Package ml provides configuration for ML model training and optimization
package ml

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// MLConfig contains global ML system configuration
type MLConfig struct {
	// Model Training
	DefaultTrainingConfig map[ModelType]TrainingConfig `json:"default_training_config"`
	
	// Model Storage
	ModelStoragePath      string        `json:"model_storage_path"`
	ModelRetentionDays    int           `json:"model_retention_days"`
	MaxModelVersions      int           `json:"max_model_versions"`
	
	// Performance Monitoring
	BenchmarkSchedule     string        `json:"benchmark_schedule"`
	PerformanceThresholds PerformanceThresholds `json:"performance_thresholds"`
	RetrainingTriggers    RetrainingTriggers    `json:"retraining_triggers"`
	
	// A/B Testing
	ABTestingConfig       ABTestingConfig       `json:"ab_testing_config"`
	
	// Resource Limits
	TrainingResourceLimits ResourceLimits       `json:"training_resource_limits"`
	InferenceResourceLimits ResourceLimits      `json:"inference_resource_limits"`
}

// PerformanceThresholds defines minimum acceptable performance metrics
type PerformanceThresholds struct {
	MinAccuracy           float64       `json:"min_accuracy"`
	MaxLatencyMs          int           `json:"max_latency_ms"`
	MinThroughputRPS      float64       `json:"min_throughput_rps"`
	MaxMemoryUsageMB      float64       `json:"max_memory_usage_mb"`
	MinConfidenceScore    float64       `json:"min_confidence_score"`
	MaxErrorRate          float64       `json:"max_error_rate"`
}

// RetrainingTriggers defines conditions that trigger model retraining
type RetrainingTriggers struct {
	AccuracyDrop          float64       `json:"accuracy_drop"`
	DataDriftThreshold    float64       `json:"data_drift_threshold"`
	TimeSinceLastTraining time.Duration `json:"time_since_last_training"`
	NewDataSamples        int           `json:"new_data_samples"`
	PerformanceDegradation float64      `json:"performance_degradation"`
}

// ABTestingConfig configures A/B testing parameters
type ABTestingConfig struct {
	Enabled               bool          `json:"enabled"`
	TrafficSplitPercent   int           `json:"traffic_split_percent"`
	MinTestDuration       time.Duration `json:"min_test_duration"`
	MaxTestDuration       time.Duration `json:"max_test_duration"`
	SignificanceLevel     float64       `json:"significance_level"`
	MinSampleSize         int           `json:"min_sample_size"`
}

// ResourceLimits defines resource constraints for ML operations
type ResourceLimits struct {
	MaxCPUCores           int           `json:"max_cpu_cores"`
	MaxMemoryGB           float64       `json:"max_memory_gb"`
	MaxGPUs               int           `json:"max_gpus"`
	MaxTrainingTimeHours  int           `json:"max_training_time_hours"`
	MaxDiskSpaceGB        float64       `json:"max_disk_space_gb"`
}

// ModelVersioning contains version management settings
type ModelVersioning struct {
	VersioningStrategy    string        `json:"versioning_strategy"` // semantic, timestamp, incremental
	AutoPromoteThreshold  float64       `json:"auto_promote_threshold"`
	RollbackThreshold     float64       `json:"rollback_threshold"`
	CanaryDeploymentPct   int           `json:"canary_deployment_pct"`
}

// GetDefaultMLConfig returns default ML configuration
func GetDefaultMLConfig() *MLConfig {
	return &MLConfig{
		DefaultTrainingConfig: map[ModelType]TrainingConfig{
			ModelTypePlacementPredictor: {
				ModelType:           ModelTypePlacementPredictor,
				ValidationSplit:     0.2,
				TestSplit:          0.1,
				BatchSize:          64,
				LearningRate:       0.01,
				Epochs:             100,
				EarlyStoppingRounds: 10,
				HyperparameterTuning: true,
				CrossValidationFolds: 5,
			},
			ModelTypeScalingPredictor: {
				ModelType:           ModelTypeScalingPredictor,
				ValidationSplit:     0.2,
				TestSplit:          0.1,
				BatchSize:          32,
				LearningRate:       0.005,
				Epochs:             150,
				EarlyStoppingRounds: 15,
				HyperparameterTuning: true,
				CrossValidationFolds: 5,
			},
			ModelTypeResourcePredictor: {
				ModelType:           ModelTypeResourcePredictor,
				ValidationSplit:     0.15,
				TestSplit:          0.1,
				BatchSize:          128,
				LearningRate:       0.01,
				Epochs:             80,
				EarlyStoppingRounds: 8,
				HyperparameterTuning: false,
				CrossValidationFolds: 3,
			},
			ModelTypeFailurePredictor: {
				ModelType:           ModelTypeFailurePredictor,
				ValidationSplit:     0.25,
				TestSplit:          0.15,
				BatchSize:          64,
				LearningRate:       0.001,
				Epochs:             200,
				EarlyStoppingRounds: 20,
				HyperparameterTuning: true,
				CrossValidationFolds: 10,
			},
		},
		
		ModelStoragePath:   "/var/lib/novacron/ml-models",
		ModelRetentionDays: 90,
		MaxModelVersions:   10,
		
		BenchmarkSchedule: "0 2 * * *", // Daily at 2 AM
		
		PerformanceThresholds: PerformanceThresholds{
			MinAccuracy:        0.85,
			MaxLatencyMs:       100,
			MinThroughputRPS:   100,
			MaxMemoryUsageMB:   512,
			MinConfidenceScore: 0.7,
			MaxErrorRate:       0.05,
		},
		
		RetrainingTriggers: RetrainingTriggers{
			AccuracyDrop:          0.05, // 5% accuracy drop triggers retraining
			DataDriftThreshold:    0.1,
			TimeSinceLastTraining: 7 * 24 * time.Hour, // Weekly retraining
			NewDataSamples:        1000,
			PerformanceDegradation: 0.1,
		},
		
		ABTestingConfig: ABTestingConfig{
			Enabled:             true,
			TrafficSplitPercent: 10, // 10% traffic for new model
			MinTestDuration:     24 * time.Hour,
			MaxTestDuration:     7 * 24 * time.Hour,
			SignificanceLevel:   0.05,
			MinSampleSize:       1000,
		},
		
		TrainingResourceLimits: ResourceLimits{
			MaxCPUCores:          8,
			MaxMemoryGB:          16,
			MaxGPUs:              2,
			MaxTrainingTimeHours: 12,
			MaxDiskSpaceGB:       100,
		},
		
		InferenceResourceLimits: ResourceLimits{
			MaxCPUCores:          4,
			MaxMemoryGB:          8,
			MaxGPUs:              1,
			MaxTrainingTimeHours: 0, // Not applicable for inference
			MaxDiskSpaceGB:       10,
		},
	}
}

// LoadMLConfig loads ML configuration from file
func LoadMLConfig(configPath string) (*MLConfig, error) {
	if configPath == "" {
		return GetDefaultMLConfig(), nil
	}
	
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		// Create default config file
		config := GetDefaultMLConfig()
		if err := SaveMLConfig(config, configPath); err != nil {
			return nil, fmt.Errorf("failed to create default config: %w", err)
		}
		return config, nil
	}
	
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	
	var config MLConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}
	
	// Validate configuration
	if err := validateMLConfig(&config); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}
	
	return &config, nil
}

// SaveMLConfig saves ML configuration to file
func SaveMLConfig(config *MLConfig, configPath string) error {
	// Ensure directory exists
	dir := filepath.Dir(configPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}
	
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}
	
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}
	
	return nil
}

// validateMLConfig validates ML configuration
func validateMLConfig(config *MLConfig) error {
	// Validate training configs
	for modelType, trainingConfig := range config.DefaultTrainingConfig {
		if err := validateTrainingConfig(modelType, trainingConfig); err != nil {
			return fmt.Errorf("invalid training config for %s: %w", modelType, err)
		}
	}
	
	// Validate performance thresholds
	if config.PerformanceThresholds.MinAccuracy < 0 || config.PerformanceThresholds.MinAccuracy > 1 {
		return fmt.Errorf("min_accuracy must be between 0 and 1")
	}
	
	if config.PerformanceThresholds.MaxLatencyMs <= 0 {
		return fmt.Errorf("max_latency_ms must be positive")
	}
	
	if config.PerformanceThresholds.MinThroughputRPS <= 0 {
		return fmt.Errorf("min_throughput_rps must be positive")
	}
	
	// Validate A/B testing config
	if config.ABTestingConfig.TrafficSplitPercent < 1 || config.ABTestingConfig.TrafficSplitPercent > 50 {
		return fmt.Errorf("traffic_split_percent must be between 1 and 50")
	}
	
	// Validate resource limits
	if config.TrainingResourceLimits.MaxCPUCores <= 0 {
		return fmt.Errorf("max_cpu_cores must be positive")
	}
	
	if config.TrainingResourceLimits.MaxMemoryGB <= 0 {
		return fmt.Errorf("max_memory_gb must be positive")
	}
	
	return nil
}

// validateTrainingConfig validates individual training configuration
func validateTrainingConfig(modelType ModelType, config TrainingConfig) error {
	if config.ValidationSplit < 0 || config.ValidationSplit > 0.5 {
		return fmt.Errorf("validation_split must be between 0 and 0.5")
	}
	
	if config.TestSplit < 0 || config.TestSplit > 0.5 {
		return fmt.Errorf("test_split must be between 0 and 0.5")
	}
	
	if config.ValidationSplit + config.TestSplit >= 1.0 {
		return fmt.Errorf("validation_split + test_split must be less than 1.0")
	}
	
	if config.BatchSize <= 0 {
		return fmt.Errorf("batch_size must be positive")
	}
	
	if config.LearningRate <= 0 || config.LearningRate > 1 {
		return fmt.Errorf("learning_rate must be between 0 and 1")
	}
	
	if config.Epochs <= 0 {
		return fmt.Errorf("epochs must be positive")
	}
	
	if config.CrossValidationFolds < 2 {
		return fmt.Errorf("cross_validation_folds must be at least 2")
	}
	
	return nil
}

// GetModelConfig returns configuration for a specific model type
func (config *MLConfig) GetModelConfig(modelType ModelType) (TrainingConfig, error) {
	trainingConfig, exists := config.DefaultTrainingConfig[modelType]
	if !exists {
		return TrainingConfig{}, fmt.Errorf("no configuration found for model type %s", modelType)
	}
	
	// Set paths if not already set
	if trainingConfig.ModelOutputPath == "" {
		trainingConfig.ModelOutputPath = filepath.Join(config.ModelStoragePath, string(modelType))
	}
	
	return trainingConfig, nil
}

// ShouldTriggerRetraining checks if model retraining should be triggered
func (config *MLConfig) ShouldTriggerRetraining(currentPerformance ModelPerformance, lastTraining time.Time, newDataSamples int) bool {
	triggers := config.RetrainingTriggers
	
	// Check accuracy drop
	if currentPerformance.Accuracy < config.PerformanceThresholds.MinAccuracy - triggers.AccuracyDrop {
		return true
	}
	
	// Check time since last training
	if time.Since(lastTraining) > triggers.TimeSinceLastTraining {
		return true
	}
	
	// Check new data samples
	if newDataSamples >= triggers.NewDataSamples {
		return true
	}
	
	return false
}

// IsPerformanceAcceptable checks if model performance meets thresholds
func (config *MLConfig) IsPerformanceAcceptable(performance ModelPerformance) bool {
	thresholds := config.PerformanceThresholds
	
	if performance.Accuracy < thresholds.MinAccuracy {
		return false
	}
	
	if performance.InferenceTime.Milliseconds() > int64(thresholds.MaxLatencyMs) {
		return false
	}
	
	if performance.ModelSizeMB > thresholds.MaxMemoryUsageMB {
		return false
	}
	
	return true
}

// GetBenchmarkConfig returns benchmark configuration based on ML config
func (config *MLConfig) GetBenchmarkConfig(modelType ModelType) BenchmarkConfig {
	return BenchmarkConfig{
		ModelType: modelType,
		BenchmarkTypes: []BenchmarkType{
			BenchmarkTypeAccuracy,
			BenchmarkTypePerformance,
			BenchmarkTypeRobustness,
		},
		TestDataSize:       1000,
		ConcurrentRequests: 10,
		Duration:          2 * time.Minute,
		WarmupRequests:    100,
		NoiseLevel:        0.1,
		StressTestEnabled: true,
		CompareBaseline:   true,
	}
}

// Environment-specific configurations

// GetDevelopmentConfig returns configuration optimized for development
func GetDevelopmentConfig() *MLConfig {
	config := GetDefaultMLConfig()
	
	// Reduce resource usage for development
	for modelType, trainingConfig := range config.DefaultTrainingConfig {
		trainingConfig.Epochs = 20
		trainingConfig.EarlyStoppingRounds = 5
		trainingConfig.HyperparameterTuning = false
		trainingConfig.CrossValidationFolds = 3
		config.DefaultTrainingConfig[modelType] = trainingConfig
	}
	
	config.TrainingResourceLimits.MaxCPUCores = 2
	config.TrainingResourceLimits.MaxMemoryGB = 4
	config.TrainingResourceLimits.MaxGPUs = 0
	config.TrainingResourceLimits.MaxTrainingTimeHours = 2
	
	config.BenchmarkSchedule = "0 */6 * * *" // Every 6 hours
	config.RetrainingTriggers.TimeSinceLastTraining = 24 * time.Hour
	
	return config
}

// GetProductionConfig returns configuration optimized for production
func GetProductionConfig() *MLConfig {
	config := GetDefaultMLConfig()
	
	// Increase training quality for production
	for modelType, trainingConfig := range config.DefaultTrainingConfig {
		trainingConfig.Epochs = 200
		trainingConfig.EarlyStoppingRounds = 20
		trainingConfig.HyperparameterTuning = true
		trainingConfig.CrossValidationFolds = 10
		config.DefaultTrainingConfig[modelType] = trainingConfig
	}
	
	// Higher performance thresholds
	config.PerformanceThresholds.MinAccuracy = 0.90
	config.PerformanceThresholds.MaxLatencyMs = 50
	config.PerformanceThresholds.MinThroughputRPS = 200
	
	// More conservative retraining triggers
	config.RetrainingTriggers.AccuracyDrop = 0.03
	config.RetrainingTriggers.TimeSinceLastTraining = 3 * 24 * time.Hour
	
	// Enable comprehensive A/B testing
	config.ABTestingConfig.MinTestDuration = 48 * time.Hour
	config.ABTestingConfig.MinSampleSize = 5000
	
	return config
}

// GetTestingConfig returns configuration optimized for testing
func GetTestingConfig() *MLConfig {
	config := GetDefaultMLConfig()
	
	// Minimal configuration for fast testing
	for modelType, trainingConfig := range config.DefaultTrainingConfig {
		trainingConfig.Epochs = 5
		trainingConfig.EarlyStoppingRounds = 2
		trainingConfig.HyperparameterTuning = false
		trainingConfig.CrossValidationFolds = 2
		config.DefaultTrainingConfig[modelType] = trainingConfig
	}
	
	config.TrainingResourceLimits.MaxCPUCores = 1
	config.TrainingResourceLimits.MaxMemoryGB = 1
	config.TrainingResourceLimits.MaxGPUs = 0
	config.TrainingResourceLimits.MaxTrainingTimeHours = 1
	
	// Relaxed performance thresholds for testing
	config.PerformanceThresholds.MinAccuracy = 0.5
	config.PerformanceThresholds.MaxLatencyMs = 1000
	config.PerformanceThresholds.MinThroughputRPS = 10
	
	config.ABTestingConfig.Enabled = false
	
	return config
}