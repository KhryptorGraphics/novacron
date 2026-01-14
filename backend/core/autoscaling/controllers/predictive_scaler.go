package controllers

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	v1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// PredictiveScaler implements ML-based predictive auto-scaling
type PredictiveScaler struct {
	mu sync.RWMutex

	// Kubernetes client
	k8sClient kubernetes.Interface

	// ML models
	arimaModel   *ARIMAModel
	lstmModel    *LSTMModel
	prophetModel *ProphetModel
	ensemble     *EnsemblePredictor

	// Control systems
	pidController *PIDController
	mpcController *ModelPredictiveController

	// Metrics collection
	metricsCollector *MetricsCollector
	customMetrics    map[string]MetricProvider

	// Scaling configuration
	config       *ScalingConfig
	policies     map[string]*ScalingPolicy
	cooldowns    map[string]*CooldownManager
	stabilizers  map[string]*StabilizationWindow

	// Cost optimization
	costOptimizer *CostOptimizer
	spotManager   *SpotInstanceManager

	// State tracking
	scalingHistory *ScalingHistory
	predictions    *PredictionCache
	confidence     *ConfidenceTracker
}

// ScalingConfig defines auto-scaling configuration
type ScalingConfig struct {
	// Basic settings
	MinReplicas int32
	MaxReplicas int32
	TargetCPU   int32
	TargetMem   int32

	// Advanced settings
	PredictionHorizon   time.Duration
	ScaleUpThreshold    float64
	ScaleDownThreshold  float64
	ConfidenceThreshold float64

	// Control parameters
	PIDGains         PIDGains
	CooldownPeriod   time.Duration
	StabilizationWindow time.Duration

	// Cost settings
	CostAware        bool
	MaxCostPerHour   float64
	SpotPercentage   float64
	FallbackStrategy string
}

// MetricProvider defines custom metric interface
type MetricProvider interface {
	GetMetric(ctx context.Context) (float64, error)
	GetAggregated(ctx context.Context, window time.Duration) (float64, error)
	GetPercentile(ctx context.Context, percentile float64) (float64, error)
}

// ARIMAModel implements ARIMA time series forecasting
type ARIMAModel struct {
	p, d, q      int
	coefficients []float64
	residuals    []float64
	data         []float64
	trained      bool
}

// LSTMModel implements LSTM neural network for prediction
type LSTMModel struct {
	model       *NeuralNetwork
	sequenceLen int
	features    int
	hidden      int
	trained     bool
}

// ProphetModel implements Facebook Prophet forecasting
type ProphetModel struct {
	trend       *TrendComponent
	seasonality *SeasonalityComponent
	holidays    *HolidayComponent
	trained     bool
}

// EnsemblePredictor combines multiple models
type EnsemblePredictor struct {
	models  []Predictor
	weights []float64
	voting  VotingStrategy
}

// PIDController implements proportional-integral-derivative control
type PIDController struct {
	kp, ki, kd float64
	integral   float64
	lastError  float64
	lastTime   time.Time
	antiWindup float64
	setpoint   float64
}

// ModelPredictiveController implements MPC for scaling
type ModelPredictiveController struct {
	model         SystemModel
	horizon       int
	controlHorizon int
	constraints   *Constraints
	optimizer     *QPSolver
}

// CostOptimizer manages cost-aware scaling
type CostOptimizer struct {
	pricing      *PricingModel
	spotPricing  *SpotPricingPredictor
	reservations *ReservationManager
	optimizer    *LinearProgramSolver
}

// SpotInstanceManager handles spot instance allocation
type SpotInstanceManager struct {
	bidStrategy  BidStrategy
	fallback     FallbackStrategy
	availability *AvailabilityPredictor
	termination  *TerminationHandler
}

// NewPredictiveScaler creates an advanced auto-scaler
func NewPredictiveScaler(client kubernetes.Interface, config *ScalingConfig) *PredictiveScaler {
	ps := &PredictiveScaler{
		k8sClient:    client,
		config:       config,
		policies:     make(map[string]*ScalingPolicy),
		cooldowns:    make(map[string]*CooldownManager),
		stabilizers:  make(map[string]*StabilizationWindow),
		customMetrics: make(map[string]MetricProvider),
		predictions:  NewPredictionCache(),
		confidence:   NewConfidenceTracker(),
	}

	// Initialize ML models
	ps.arimaModel = NewARIMAModel(2, 1, 2) // ARIMA(2,1,2)
	ps.lstmModel = NewLSTMModel(24, 5, 128) // 24 timesteps, 5 features, 128 hidden
	ps.prophetModel = NewProphetModel()
	
	// Create ensemble predictor
	ps.ensemble = &EnsemblePredictor{
		models: []Predictor{ps.arimaModel, ps.lstmModel, ps.prophetModel},
		weights: []float64{0.3, 0.4, 0.3},
		voting: WeightedAverage,
	}

	// Initialize controllers
	ps.pidController = &PIDController{
		kp: config.PIDGains.Kp,
		ki: config.PIDGains.Ki,
		kd: config.PIDGains.Kd,
		antiWindup: 100.0,
	}

	ps.mpcController = NewModelPredictiveController(10, 3)

	// Initialize metrics collector
	ps.metricsCollector = NewMetricsCollector(client)

	// Initialize cost optimization if enabled
	if config.CostAware {
		ps.costOptimizer = NewCostOptimizer()
		ps.spotManager = NewSpotInstanceManager(config.SpotPercentage)
	}

	// Initialize scaling history
	ps.scalingHistory = NewScalingHistory(1000)

	return ps
}

// ScaleDeployment performs intelligent scaling decision
func (ps *PredictiveScaler) ScaleDeployment(ctx context.Context, namespace, name string) error {
	ps.mu.Lock()
	defer ps.mu.Unlock()

	// Check cooldown
	if ps.isInCooldown(namespace, name) {
		return nil
	}

	// Collect current metrics
	metrics, err := ps.collectMetrics(ctx, namespace, name)
	if err != nil {
		return fmt.Errorf("failed to collect metrics: %v", err)
	}

	// Get current deployment
	deployment, err := ps.k8sClient.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get deployment: %v", err)
	}

	currentReplicas := *deployment.Spec.Replicas

	// Generate predictions
	predictions := ps.generatePredictions(metrics)

	// Calculate desired replicas using multiple strategies
	pidReplicas := ps.calculatePIDReplicas(metrics, currentReplicas)
	mpcReplicas := ps.calculateMPCReplicas(metrics, predictions)
	mlReplicas := ps.calculateMLReplicas(predictions)

	// Combine decisions with weighted voting
	desiredReplicas := ps.combineScalingDecisions(
		currentReplicas,
		pidReplicas,
		mpcReplicas,
		mlReplicas,
		predictions.Confidence,
	)

	// Apply cost optimization if enabled
	if ps.config.CostAware {
		desiredReplicas = ps.optimizeForCost(desiredReplicas, metrics)
	}

	// Apply stabilization
	desiredReplicas = ps.stabilize(namespace, name, desiredReplicas)

	// Apply bounds
	desiredReplicas = ps.applyBounds(desiredReplicas)

	// Check if scaling is needed
	if desiredReplicas == currentReplicas {
		return nil
	}

	// Perform scaling
	if err := ps.performScaling(ctx, deployment, desiredReplicas); err != nil {
		return err
	}

	// Record scaling event
	ps.recordScalingEvent(namespace, name, currentReplicas, desiredReplicas, metrics, predictions)

	// Update cooldown
	ps.updateCooldown(namespace, name)

	// Train models with feedback
	go ps.trainModels(metrics, desiredReplicas)

	return nil
}

// generatePredictions uses ML models to forecast metrics
func (ps *PredictiveScaler) generatePredictions(metrics *MetricsSnapshot) *PredictionResult {
	// Prepare time series data
	timeSeries := ps.prepareTimeSeries(metrics)

	// Get predictions from each model
	arimaPred := ps.arimaModel.Predict(timeSeries, ps.config.PredictionHorizon)
	lstmPred := ps.lstmModel.Predict(timeSeries, ps.config.PredictionHorizon)
	prophetPred := ps.prophetModel.Predict(timeSeries, ps.config.PredictionHorizon)

	// Ensemble predictions
	ensemblePred := ps.ensemble.Predict(timeSeries, ps.config.PredictionHorizon)

	// Calculate confidence intervals
	confidence := ps.calculateConfidence(arimaPred, lstmPred, prophetPred)

	return &PredictionResult{
		Predictions: ensemblePred,
		Confidence:  confidence,
		Upper95CI:   ps.calculateUpperCI(ensemblePred, confidence),
		Lower95CI:   ps.calculateLowerCI(ensemblePred, confidence),
		Horizon:     ps.config.PredictionHorizon,
	}
}

// calculatePIDReplicas uses PID control for scaling
func (ps *PredictiveScaler) calculatePIDReplicas(metrics *MetricsSnapshot, current int32) int32 {
	// Set target based on configuration
	ps.pidController.setpoint = float64(ps.config.TargetCPU)

	// Calculate error
	error := ps.pidController.setpoint - metrics.CPU

	// Apply PID control
	output := ps.pidController.Calculate(error, time.Now())

	// Convert to replica count
	scaleFactor := 1.0 + output/100.0
	desired := int32(float64(current) * scaleFactor)

	return desired
}

// calculateMPCReplicas uses Model Predictive Control
func (ps *PredictiveScaler) calculateMPCReplicas(metrics *MetricsSnapshot, predictions *PredictionResult) int32 {
	// Define state vector
	state := []float64{
		metrics.CPU,
		metrics.Memory,
		metrics.RequestRate,
		metrics.ResponseTime,
	}

	// Define constraints
	constraints := &Constraints{
		MinReplicas: ps.config.MinReplicas,
		MaxReplicas: ps.config.MaxReplicas,
		MaxScaleUp:  5,
		MaxScaleDown: 2,
	}

	// Solve MPC optimization problem
	control := ps.mpcController.Solve(state, predictions.Predictions, constraints)

	return int32(control[0])
}

// calculateMLReplicas uses pure ML prediction
func (ps *PredictiveScaler) calculateMLReplicas(predictions *PredictionResult) int32 {
	// Find peak predicted load
	peakLoad := 0.0
	for _, pred := range predictions.Predictions {
		if pred > peakLoad {
			peakLoad = pred
		}
	}

	// Calculate required replicas for peak load
	// Using Little's Law and queueing theory
	arrivalRate := peakLoad
	serviceRate := 100.0 // requests per second per replica
	utilization := 0.7   // target utilization

	replicas := math.Ceil(arrivalRate / (serviceRate * utilization))

	// Apply confidence adjustment
	if predictions.Confidence < ps.config.ConfidenceThreshold {
		// Add safety margin for low confidence
		replicas *= 1.2
	}

	return int32(replicas)
}

// combineScalingDecisions merges multiple scaling strategies
func (ps *PredictiveScaler) combineScalingDecisions(current, pid, mpc, ml int32, confidence float64) int32 {
	// Weight based on confidence
	pidWeight := 0.3
	mpcWeight := 0.4
	mlWeight := 0.3

	if confidence < 0.7 {
		// Reduce ML weight when confidence is low
		pidWeight = 0.5
		mpcWeight = 0.4
		mlWeight = 0.1
	}

	// Weighted average
	weighted := float64(pid)*pidWeight + float64(mpc)*mpcWeight + float64(ml)*mlWeight

	// Round to nearest integer
	desired := int32(math.Round(weighted))

	// Apply hysteresis to prevent flapping
	if math.Abs(float64(desired-current)) < 1 {
		return current
	}

	return desired
}

// optimizeForCost adjusts scaling based on cost constraints
func (ps *PredictiveScaler) optimizeForCost(desired int32, metrics *MetricsSnapshot) int32 {
	// Get current pricing
	onDemandPrice := ps.costOptimizer.GetOnDemandPrice()
	spotPrice := ps.costOptimizer.GetSpotPrice()
	
	// Calculate cost for desired replicas
	costPerHour := ps.calculateCost(desired, spotPrice, onDemandPrice)
	
	// Check if within budget
	if costPerHour <= ps.config.MaxCostPerHour {
		return desired
	}
	
	// Optimize replica mix for cost
	spotReplicas, onDemandReplicas := ps.optimizeReplicaMix(desired, spotPrice, onDemandPrice)
	
	// Check spot availability
	if ps.spotManager.CheckAvailability(spotReplicas) {
		return spotReplicas + onDemandReplicas
	}
	
	// Fallback to reduced scaling
	return ps.reducedScaling(desired, ps.config.MaxCostPerHour, onDemandPrice)
}

// stabilize applies stabilization window to prevent oscillation
func (ps *PredictiveScaler) stabilize(namespace, name string, desired int32) int32 {
	key := fmt.Sprintf("%s/%s", namespace, name)
	
	stabilizer, exists := ps.stabilizers[key]
	if !exists {
		stabilizer = NewStabilizationWindow(ps.config.StabilizationWindow)
		ps.stabilizers[key] = stabilizer
	}
	
	return stabilizer.Stabilize(desired)
}

// trainModels performs online learning for ML models
func (ps *PredictiveScaler) trainModels(metrics *MetricsSnapshot, actualReplicas int32) {
	// Prepare training data
	features := ps.extractFeatures(metrics)
	target := float64(actualReplicas)
	
	// Online training for ARIMA
	ps.arimaModel.UpdateCoefficients(features, target)
	
	// Batch training for LSTM (accumulate then train)
	ps.lstmModel.AddSample(features, target)
	if ps.lstmModel.ShouldTrain() {
		ps.lstmModel.Train()
	}
	
	// Update Prophet model
	ps.prophetModel.AddObservation(time.Now(), target)
	
	// Update ensemble weights based on performance
	ps.updateEnsembleWeights()
}

// PIDController.Calculate computes PID output
func (pid *PIDController) Calculate(error float64, now time.Time) float64 {
	// Time delta
	dt := now.Sub(pid.lastTime).Seconds()
	if dt == 0 {
		dt = 1.0
	}
	
	// Proportional term
	p := pid.kp * error
	
	// Integral term with anti-windup
	pid.integral += error * dt
	if math.Abs(pid.integral) > pid.antiWindup {
		pid.integral = pid.antiWindup * sign(pid.integral)
	}
	i := pid.ki * pid.integral
	
	// Derivative term
	derivative := (error - pid.lastError) / dt
	d := pid.kd * derivative
	
	// Update state
	pid.lastError = error
	pid.lastTime = now
	
	// Calculate output
	return p + i + d
}

// Workload pattern detection
type WorkloadPattern struct {
	Type        WorkloadType
	Periodicity time.Duration
	Amplitude   float64
	Trend       TrendType
	Burstiness  float64
}

type WorkloadType int

const (
	BatchWorkload WorkloadType = iota
	StreamingWorkload
	InteractiveWorkload
	MLTrainingWorkload
	MixedWorkload
)

// detectWorkloadPattern identifies workload characteristics
func (ps *PredictiveScaler) detectWorkloadPattern(metrics *MetricsSnapshot) *WorkloadPattern {
	pattern := &WorkloadPattern{}
	
	// Analyze time series for periodicity
	pattern.Periodicity = ps.detectPeriodicity(metrics.History)
	
	// Detect trend
	pattern.Trend = ps.detectTrend(metrics.History)
	
	// Calculate burstiness (variance/mean ratio)
	pattern.Burstiness = ps.calculateBurstiness(metrics.History)
	
	// Classify workload type
	pattern.Type = ps.classifyWorkload(metrics)
	
	return pattern
}

// applyWorkloadSpecificScaling applies pattern-specific scaling
func (ps *PredictiveScaler) applyWorkloadSpecificScaling(pattern *WorkloadPattern, baseReplicas int32) int32 {
	switch pattern.Type {
	case BatchWorkload:
		// Scale aggressively for batch jobs
		return ps.scaleBatchWorkload(pattern, baseReplicas)
		
	case StreamingWorkload:
		// Maintain steady state with buffer
		return ps.scaleStreamingWorkload(pattern, baseReplicas)
		
	case InteractiveWorkload:
		// Fast scale-up, gradual scale-down
		return ps.scaleInteractiveWorkload(pattern, baseReplicas)
		
	case MLTrainingWorkload:
		// GPU-aware scaling
		return ps.scaleMLWorkload(pattern, baseReplicas)
		
	default:
		return baseReplicas
	}
}

// Helper functions
func sign(x float64) float64 {
	if x < 0 {
		return -1
	}
	return 1
}