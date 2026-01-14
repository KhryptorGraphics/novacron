package policies

import (
	"context"
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// ScalingPolicy defines auto-scaling behavior for different workload types
type ScalingPolicy struct {
	Name        string
	Type        WorkloadType
	Priority    int
	Enabled     bool
	
	// Scaling thresholds
	ScaleUpThreshold   float64
	ScaleDownThreshold float64
	
	// Rate limits
	MaxScaleUpRate   int32
	MaxScaleDownRate int32
	
	// Timing
	ScaleUpDelay     time.Duration
	ScaleDownDelay   time.Duration
	CooldownPeriod   time.Duration
	
	// Resource targets
	TargetCPU        int32
	TargetMemory     int32
	TargetLatency    time.Duration
	TargetThroughput float64
	
	// Custom metrics
	CustomMetrics    map[string]*MetricPolicy
	
	// Cost constraints
	CostConstraints  *CostPolicy
	
	// ML configuration
	MLConfig        *MLPolicy
}

// WorkloadType represents different workload patterns
type WorkloadType string

const (
	BatchWorkloadType       WorkloadType = "batch"
	StreamingWorkloadType   WorkloadType = "streaming"
	InteractiveWorkloadType WorkloadType = "interactive"
	MLTrainingWorkloadType  WorkloadType = "ml-training"
	MicroserviceWorkloadType WorkloadType = "microservice"
	DatabaseWorkloadType    WorkloadType = "database"
	CacheWorkloadType       WorkloadType = "cache"
)

// MetricPolicy defines behavior for custom metrics
type MetricPolicy struct {
	Name          string
	Source        string
	Aggregation   AggregationType
	Window        time.Duration
	Target        float64
	Weight        float64
	ScalingFactor float64
}

// CostPolicy defines cost optimization constraints
type CostPolicy struct {
	MaxCostPerHour     float64
	PreferSpot         bool
	SpotMaxPrice       float64
	ReservedInstances  int32
	BurstableInstances bool
	RegionPreferences  []string
}

// MLPolicy defines ML-based scaling configuration
type MLPolicy struct {
	EnablePrediction   bool
	PredictionHorizon  time.Duration
	ModelType          string
	UpdateFrequency    time.Duration
	ConfidenceThreshold float64
	FeatureEngineering bool
}

// PolicyManager manages and applies scaling policies
type PolicyManager struct {
	mu       sync.RWMutex
	policies map[string]*ScalingPolicy
	active   map[string]bool
	metrics  *PolicyMetrics
}

// PolicyMetrics tracks policy effectiveness
type PolicyMetrics struct {
	ApplicationCount  prometheus.Counter
	SuccessRate      prometheus.Gauge
	ScalingAccuracy  prometheus.Histogram
	CostSavings      prometheus.Counter
	ViolationRate    prometheus.Gauge
}

// NewPolicyManager creates a policy manager
func NewPolicyManager() *PolicyManager {
	return &PolicyManager{
		policies: make(map[string]*ScalingPolicy),
		active:   make(map[string]bool),
		metrics:  NewPolicyMetrics(),
	}
}

// CreateBatchWorkloadPolicy creates policy for batch processing
func CreateBatchWorkloadPolicy() *ScalingPolicy {
	return &ScalingPolicy{
		Name:     "batch-processing",
		Type:     BatchWorkloadType,
		Priority: 5,
		Enabled:  true,
		
		// Aggressive scaling for batch jobs
		ScaleUpThreshold:   60,
		ScaleDownThreshold: 20,
		
		// Fast scaling
		MaxScaleUpRate:   10,
		MaxScaleDownRate: 5,
		
		// Quick response
		ScaleUpDelay:     30 * time.Second,
		ScaleDownDelay:   5 * time.Minute,
		CooldownPeriod:   2 * time.Minute,
		
		// Resource targets
		TargetCPU:        70,
		TargetMemory:     80,
		TargetThroughput: 1000, // jobs/minute
		
		CustomMetrics: map[string]*MetricPolicy{
			"queue_depth": {
				Name:          "queue_depth",
				Source:        "sqs",
				Aggregation:   Average,
				Window:        1 * time.Minute,
				Target:        100,
				Weight:        0.4,
				ScalingFactor: 0.01,
			},
			"job_completion_rate": {
				Name:          "job_completion_rate",
				Source:        "application",
				Aggregation:   Rate,
				Window:        5 * time.Minute,
				Target:        50,
				Weight:        0.3,
				ScalingFactor: 0.02,
			},
		},
		
		CostConstraints: &CostPolicy{
			MaxCostPerHour:     100,
			PreferSpot:         true,
			SpotMaxPrice:       0.5,
			BurstableInstances: true,
		},
		
		MLConfig: &MLPolicy{
			EnablePrediction:    true,
			PredictionHorizon:   30 * time.Minute,
			ModelType:          "arima",
			UpdateFrequency:     10 * time.Minute,
			ConfidenceThreshold: 0.8,
		},
	}
}

// CreateStreamingWorkloadPolicy creates policy for streaming data
func CreateStreamingWorkloadPolicy() *ScalingPolicy {
	return &ScalingPolicy{
		Name:     "streaming-data",
		Type:     StreamingWorkloadType,
		Priority: 8,
		Enabled:  true,
		
		// Stable scaling for continuous streams
		ScaleUpThreshold:   75,
		ScaleDownThreshold: 40,
		
		// Moderate scaling
		MaxScaleUpRate:   5,
		MaxScaleDownRate: 2,
		
		// Balanced timing
		ScaleUpDelay:     1 * time.Minute,
		ScaleDownDelay:   10 * time.Minute,
		CooldownPeriod:   5 * time.Minute,
		
		// Resource targets
		TargetCPU:        60,
		TargetMemory:     70,
		TargetLatency:    100 * time.Millisecond,
		TargetThroughput: 10000, // events/second
		
		CustomMetrics: map[string]*MetricPolicy{
			"kafka_lag": {
				Name:          "kafka_lag",
				Source:        "kafka",
				Aggregation:   Max,
				Window:        30 * time.Second,
				Target:        1000,
				Weight:        0.5,
				ScalingFactor: 0.001,
			},
			"processing_latency": {
				Name:          "processing_latency",
				Source:        "application",
				Aggregation:   Percentile95,
				Window:        1 * time.Minute,
				Target:        50,
				Weight:        0.3,
				ScalingFactor: 0.01,
			},
		},
		
		MLConfig: &MLPolicy{
			EnablePrediction:    true,
			PredictionHorizon:   15 * time.Minute,
			ModelType:          "lstm",
			UpdateFrequency:     5 * time.Minute,
			ConfidenceThreshold: 0.85,
			FeatureEngineering: true,
		},
	}
}

// CreateInteractiveWorkloadPolicy creates policy for user-facing services
func CreateInteractiveWorkloadPolicy() *ScalingPolicy {
	return &ScalingPolicy{
		Name:     "interactive-service",
		Type:     InteractiveWorkloadType,
		Priority: 10, // Highest priority
		Enabled:  true,
		
		// Responsive scaling for user experience
		ScaleUpThreshold:   50,
		ScaleDownThreshold: 30,
		
		// Fast scale-up, slow scale-down
		MaxScaleUpRate:   8,
		MaxScaleDownRate: 1,
		
		// Very quick response
		ScaleUpDelay:     15 * time.Second,
		ScaleDownDelay:   15 * time.Minute,
		CooldownPeriod:   3 * time.Minute,
		
		// Resource targets
		TargetCPU:        40,
		TargetMemory:     60,
		TargetLatency:    50 * time.Millisecond,
		TargetThroughput: 1000, // requests/second
		
		CustomMetrics: map[string]*MetricPolicy{
			"response_time_p99": {
				Name:          "response_time_p99",
				Source:        "istio",
				Aggregation:   Percentile99,
				Window:        30 * time.Second,
				Target:        100,
				Weight:        0.6,
				ScalingFactor: 0.01,
			},
			"error_rate": {
				Name:          "error_rate",
				Source:        "prometheus",
				Aggregation:   Rate,
				Window:        1 * time.Minute,
				Target:        0.01,
				Weight:        0.2,
				ScalingFactor: 100,
			},
		},
		
		CostConstraints: &CostPolicy{
			MaxCostPerHour:     200,
			PreferSpot:         false, // Stability over cost
			ReservedInstances:  5,
		},
		
		MLConfig: &MLPolicy{
			EnablePrediction:    true,
			PredictionHorizon:   10 * time.Minute,
			ModelType:          "prophet",
			UpdateFrequency:     1 * time.Minute,
			ConfidenceThreshold: 0.9,
		},
	}
}

// CreateMLTrainingWorkloadPolicy creates policy for ML training jobs
func CreateMLTrainingWorkloadPolicy() *ScalingPolicy {
	return &ScalingPolicy{
		Name:     "ml-training",
		Type:     MLTrainingWorkloadType,
		Priority: 7,
		Enabled:  true,
		
		// GPU-aware scaling
		ScaleUpThreshold:   80,
		ScaleDownThreshold: 10,
		
		// Batch scaling for GPU efficiency
		MaxScaleUpRate:   4,
		MaxScaleDownRate: 4,
		
		// Longer delays for GPU warmup
		ScaleUpDelay:     2 * time.Minute,
		ScaleDownDelay:   5 * time.Minute,
		CooldownPeriod:   10 * time.Minute,
		
		// Resource targets
		TargetCPU:        90,
		TargetMemory:     85,
		TargetThroughput: 100, // batches/minute
		
		CustomMetrics: map[string]*MetricPolicy{
			"gpu_utilization": {
				Name:          "gpu_utilization",
				Source:        "nvidia_dcgm",
				Aggregation:   Average,
				Window:        1 * time.Minute,
				Target:        85,
				Weight:        0.7,
				ScalingFactor: 0.01,
			},
			"training_loss": {
				Name:          "training_loss",
				Source:        "mlflow",
				Aggregation:   MovingAverage,
				Window:        10 * time.Minute,
				Target:        0.01,
				Weight:        0.2,
				ScalingFactor: 10,
			},
			"checkpoint_frequency": {
				Name:          "checkpoint_frequency",
				Source:        "storage",
				Aggregation:   Rate,
				Window:        5 * time.Minute,
				Target:        1,
				Weight:        0.1,
				ScalingFactor: 1,
			},
		},
		
		CostConstraints: &CostPolicy{
			MaxCostPerHour:     500,
			PreferSpot:         true,
			SpotMaxPrice:       2.0,
			BurstableInstances: false, // Need consistent GPU performance
		},
		
		MLConfig: &MLPolicy{
			EnablePrediction:    true,
			PredictionHorizon:   1 * time.Hour,
			ModelType:          "ensemble",
			UpdateFrequency:     15 * time.Minute,
			ConfidenceThreshold: 0.75,
			FeatureEngineering: true,
		},
	}
}

// ApplyPolicy evaluates and applies a scaling policy
func (pm *PolicyManager) ApplyPolicy(ctx context.Context, policy *ScalingPolicy, metrics map[string]float64) (*ScalingDecision, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	if !policy.Enabled {
		return nil, nil
	}

	decision := &ScalingDecision{
		PolicyName: policy.Name,
		Timestamp:  time.Now(),
	}

	// Evaluate resource metrics
	cpuScore := pm.evaluateMetric(metrics["cpu"], float64(policy.TargetCPU))
	memScore := pm.evaluateMetric(metrics["memory"], float64(policy.TargetMemory))
	
	// Evaluate custom metrics
	customScore := 0.0
	customWeight := 0.0
	
	for name, metricPolicy := range policy.CustomMetrics {
		if value, exists := metrics[name]; exists {
			score := pm.evaluateMetric(value, metricPolicy.Target)
			customScore += score * metricPolicy.Weight
			customWeight += metricPolicy.Weight
		}
	}
	
	if customWeight > 0 {
		customScore /= customWeight
	}
	
	// Combine scores
	totalScore := cpuScore*0.3 + memScore*0.3 + customScore*0.4
	
	// Determine scaling action
	if totalScore > policy.ScaleUpThreshold/100.0 {
		decision.Action = ScaleUp
		decision.Amount = pm.calculateScaleAmount(totalScore, policy.MaxScaleUpRate)
	} else if totalScore < policy.ScaleDownThreshold/100.0 {
		decision.Action = ScaleDown
		decision.Amount = pm.calculateScaleAmount(1-totalScore, policy.MaxScaleDownRate)
	} else {
		decision.Action = NoChange
	}
	
	// Apply ML predictions if enabled
	if policy.MLConfig.EnablePrediction {
		decision = pm.applyMLPrediction(decision, policy, metrics)
	}
	
	// Apply cost constraints
	if policy.CostConstraints != nil {
		decision = pm.applyCostConstraints(decision, policy.CostConstraints)
	}
	
	// Record metrics
	pm.metrics.ApplicationCount.Inc()
	
	return decision, nil
}

// evaluateMetric calculates normalized score for a metric
func (pm *PolicyManager) evaluateMetric(current, target float64) float64 {
	if target == 0 {
		return 0
	}
	
	ratio := current / target
	
	// Apply sigmoid function for smooth scaling
	return 1.0 / (1.0 + math.Exp(-5*(ratio-1)))
}

// calculateScaleAmount determines how many instances to scale
func (pm *PolicyManager) calculateScaleAmount(score float64, maxRate int32) int32 {
	// Linear mapping with score
	amount := int32(score * float64(maxRate))
	
	if amount < 1 {
		amount = 1
	}
	
	return amount
}

// applyMLPrediction adjusts decision based on ML predictions
func (pm *PolicyManager) applyMLPrediction(decision *ScalingDecision, policy *ScalingPolicy, metrics map[string]float64) *ScalingDecision {
	// This would integrate with the ML predictor
	// For now, return the original decision
	return decision
}

// applyCostConstraints adjusts decision for cost optimization
func (pm *PolicyManager) applyCostConstraints(decision *ScalingDecision, constraints *CostPolicy) *ScalingDecision {
	// Calculate current cost
	currentCost := pm.calculateCurrentCost()
	
	// Check if scaling up would exceed budget
	if decision.Action == ScaleUp {
		projectedCost := currentCost * float64(1+decision.Amount/10)
		if projectedCost > constraints.MaxCostPerHour {
			// Reduce scaling or switch to spot instances
			decision.Amount = decision.Amount / 2
			decision.UseSpot = constraints.PreferSpot
		}
	}
	
	return decision
}

// calculateCurrentCost estimates current hourly cost
func (pm *PolicyManager) calculateCurrentCost() float64 {
	// Placeholder implementation
	return 50.0
}

// ScalingDecision represents a scaling action
type ScalingDecision struct {
	PolicyName string
	Timestamp  time.Time
	Action     ScalingAction
	Amount     int32
	UseSpot    bool
	Reason     string
	Confidence float64
}

// ScalingAction represents the type of scaling
type ScalingAction int

const (
	NoChange ScalingAction = iota
	ScaleUp
	ScaleDown
	Rebalance
)

// AggregationType defines metric aggregation methods
type AggregationType int

const (
	Average AggregationType = iota
	Max
	Min
	Sum
	Rate
	Percentile95
	Percentile99
	MovingAverage
)