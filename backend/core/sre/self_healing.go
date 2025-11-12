// Self-Healing Infrastructure with Predictive Remediation
// Implements autonomous healing with machine learning-based prediction

package sre

import (
	"context"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

// HealingStrategy defines the approach for self-healing
type HealingStrategy int

const (
	StrategyReactive HealingStrategy = iota  // React to detected issues
	StrategyProactive                        // Prevent issues before they occur
	StrategyPredictive                       // Predict and prevent future issues
	StrategyAdaptive                         // Learn and adapt healing strategies
)

// HealthState represents component health status
type HealthState int

const (
	HealthHealthy HealthState = iota
	HealthDegraded
	HealthCritical
	HealthFailed
	HealthRecovering
)

// HealingAction represents a self-healing action
type HealingAction struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Component   string                 `json:"component"`
	Strategy    HealingStrategy        `json:"strategy"`
	Priority    int                    `json:"priority"`
	Confidence  float64                `json:"confidence"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     *time.Time             `json:"end_time,omitempty"`
	Success     bool                   `json:"success"`
	Metrics     map[string]interface{} `json:"metrics"`
	RollbackID  string                 `json:"rollback_id,omitempty"`
}

// Component represents a system component that can be healed
type Component struct {
	ID           string                 `json:"id"`
	Type         string                 `json:"type"`
	Name         string                 `json:"name"`
	State        HealthState            `json:"state"`
	Dependencies []string               `json:"dependencies"`
	Metrics      ComponentMetrics       `json:"metrics"`
	LastCheck    time.Time              `json:"last_check"`
	LastHealed   *time.Time             `json:"last_healed,omitempty"`
	HealCount    int                    `json:"heal_count"`
	Config       map[string]interface{} `json:"config"`
	mu           sync.RWMutex
}

// ComponentMetrics tracks component health metrics
type ComponentMetrics struct {
	CPU            float64            `json:"cpu"`
	Memory         float64            `json:"memory"`
	Disk           float64            `json:"disk"`
	NetworkLatency float64            `json:"network_latency"`
	ErrorRate      float64            `json:"error_rate"`
	RequestRate    float64            `json:"request_rate"`
	Custom         map[string]float64 `json:"custom"`
}

// PredictiveModel implements ML-based failure prediction
type PredictiveModel struct {
	timeSeriesPredictor *TimeSeriesPredictor
	anomalyDetector     *AnomalyDetector
	riskCalculator      *RiskCalculator
	threshold           float64
	windowSize          time.Duration
	mu                  sync.RWMutex
}

// TimeSeriesPredictor predicts future metric values
type TimeSeriesPredictor struct {
	model       *ARIMAModel
	features    []string
	horizon     int // Prediction horizon in minutes
	confidence  float64
	mu          sync.RWMutex
}

// ARIMAModel implements ARIMA time series forecasting
type ARIMAModel struct {
	p          int     // Autoregressive order
	d          int     // Differencing order
	q          int     // Moving average order
	parameters []float64
	residuals  []float64
	fitted     bool
}

// AnomalyDetector detects anomalies in component behavior
type AnomalyDetector struct {
	algorithm    AnomalyAlgorithm
	sensitivity  float64
	windowSize   int
	baseline     map[string]*BaselineModel
	anomalyScore map[string]float64
	mu           sync.RWMutex
}

// AnomalyAlgorithm defines the anomaly detection algorithm
type AnomalyAlgorithm int

const (
	AlgorithmIsolationForest AnomalyAlgorithm = iota
	AlgorithmLOF // Local Outlier Factor
	AlgorithmDBSCAN
	AlgorithmAutoencoder
	AlgorithmProphet
)

// BaselineModel represents normal behavior baseline
type BaselineModel struct {
	Mean              float64
	StdDev            float64
	Percentiles       map[int]float64
	SeasonalPattern   []float64
	TrendComponent    float64
	LastUpdated       time.Time
}

// RiskCalculator calculates failure risk scores
type RiskCalculator struct {
	weights           map[string]float64
	riskFactors       []RiskFactor
	historicalFailures map[string][]FailureEvent
	mtbf              map[string]time.Duration // Mean Time Between Failures
	mu                sync.RWMutex
}

// RiskFactor represents a factor contributing to failure risk
type RiskFactor struct {
	Name        string
	Weight      float64
	Current     float64
	Threshold   float64
	Criticality float64
}

// FailureEvent represents a historical failure
type FailureEvent struct {
	Timestamp   time.Time
	Component   string
	Type        string
	Duration    time.Duration
	Impact      float64
	RootCause   string
	Remediation string
}

// HealingOrchestrator manages self-healing operations
type HealingOrchestrator struct {
	components      sync.Map // map[string]*Component
	healers         map[string]Healer
	predictor       *PredictiveModel
	scheduler       *HealingScheduler
	history         *HealingHistory
	config          *HealingConfig
	metrics         *HealingMetrics
	logger          *zap.Logger
	shutdownCh      chan struct{}
	wg              sync.WaitGroup
}

// Healer interface for component-specific healing
type Healer interface {
	Diagnose(ctx context.Context, component *Component) (*Diagnosis, error)
	Heal(ctx context.Context, component *Component, action *HealingAction) error
	Rollback(ctx context.Context, component *Component, action *HealingAction) error
	Validate(ctx context.Context, component *Component) error
}

// Diagnosis represents the result of component diagnosis
type Diagnosis struct {
	Component       string
	Issue           string
	Severity        float64
	CanAutoHeal     bool
	RecommendedAction *HealingAction
	AlternativeActions []*HealingAction
	EstimatedTime   time.Duration
	RiskScore       float64
}

// HealingScheduler schedules healing actions
type HealingScheduler struct {
	queue           PriorityQueue
	executing       sync.Map
	rateLimiter     *RateLimiter
	maxConcurrent   int
	currentLoad     int32
	mu              sync.RWMutex
}

// PriorityQueue implements a priority queue for healing actions
type PriorityQueue struct {
	items []*QueueItem
	mu    sync.RWMutex
}

// QueueItem represents an item in the priority queue
type QueueItem struct {
	Action   *HealingAction
	Priority float64
	AddedAt  time.Time
}

// HealingHistory tracks healing action history
type HealingHistory struct {
	actions      []HealingAction
	successRate  map[string]float64
	avgHealTime  map[string]time.Duration
	lastFailures map[string]time.Time
	mu           sync.RWMutex
}

// HealingConfig configures self-healing behavior
type HealingConfig struct {
	Enabled              bool
	Strategy             HealingStrategy
	MaxConcurrentHeals   int
	HealingCooldown      time.Duration
	PredictionHorizon    time.Duration
	ConfidenceThreshold  float64
	MaxRetries           int
	RollbackOnFailure    bool
	DryRun              bool
}

// HealingMetrics tracks self-healing metrics
type HealingMetrics struct {
	healingAttempts    prometheus.Counter
	healingSuccesses   prometheus.Counter
	healingFailures    prometheus.Counter
	healingDuration    prometheus.Histogram
	predictedFailures  prometheus.Counter
	preventedIncidents prometheus.Counter
	componentHealth    prometheus.GaugeVec
	riskScore         prometheus.GaugeVec
}

// NewHealingOrchestrator creates a new healing orchestrator
func NewHealingOrchestrator(config *HealingConfig, logger *zap.Logger) *HealingOrchestrator {
	return &HealingOrchestrator{
		healers: make(map[string]Healer),
		predictor: &PredictiveModel{
			timeSeriesPredictor: NewTimeSeriesPredictor(30), // 30-minute horizon
			anomalyDetector:     NewAnomalyDetector(AlgorithmIsolationForest, 0.95),
			riskCalculator:      NewRiskCalculator(),
			threshold:           config.ConfidenceThreshold,
			windowSize:          config.PredictionHorizon,
		},
		scheduler: &HealingScheduler{
			queue:         PriorityQueue{items: make([]*QueueItem, 0)},
			rateLimiter:   NewRateLimiter(10, time.Minute),
			maxConcurrent: config.MaxConcurrentHeals,
		},
		history:    NewHealingHistory(),
		config:     config,
		metrics:    NewHealingMetrics(),
		logger:     logger,
		shutdownCh: make(chan struct{}),
	}
}

// Start begins the self-healing orchestration
func (o *HealingOrchestrator) Start(ctx context.Context) error {
	o.logger.Info("Starting self-healing orchestrator",
		zap.String("strategy", o.strategyName()),
		zap.Int("max_concurrent", o.config.MaxConcurrentHeals))

	// Start health monitoring
	o.wg.Add(1)
	go o.monitorHealth(ctx)

	// Start predictive analysis if enabled
	if o.config.Strategy == StrategyPredictive || o.config.Strategy == StrategyAdaptive {
		o.wg.Add(1)
		go o.runPredictiveAnalysis(ctx)
	}

	// Start healing executor
	o.wg.Add(1)
	go o.executeHealingActions(ctx)

	// Start metrics collection
	o.wg.Add(1)
	go o.collectMetrics(ctx)

	return nil
}

// RegisterComponent registers a component for self-healing
func (o *HealingOrchestrator) RegisterComponent(component *Component, healer Healer) {
	o.components.Store(component.ID, component)
	o.healers[component.Type] = healer

	o.logger.Info("Component registered for self-healing",
		zap.String("id", component.ID),
		zap.String("type", component.Type))
}

// monitorHealth continuously monitors component health
func (o *HealingOrchestrator) monitorHealth(ctx context.Context) {
	defer o.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.shutdownCh:
			return
		case <-ticker.C:
			o.components.Range(func(key, value interface{}) bool {
				component := value.(*Component)
				o.checkComponentHealth(ctx, component)
				return true
			})
		}
	}
}

// checkComponentHealth checks and potentially heals a component
func (o *HealingOrchestrator) checkComponentHealth(ctx context.Context, component *Component) {
	healer, exists := o.healers[component.Type]
	if !exists {
		o.logger.Warn("No healer found for component type",
			zap.String("type", component.Type))
		return
	}

	// Diagnose component
	diagnosis, err := healer.Diagnose(ctx, component)
	if err != nil {
		o.logger.Error("Failed to diagnose component",
			zap.String("id", component.ID),
			zap.Error(err))
		return
	}

	// Determine if healing is needed
	if diagnosis.CanAutoHeal && diagnosis.Severity > 0.5 {
		// Check cooldown period
		if o.isInCooldown(component) {
			o.logger.Debug("Component in healing cooldown",
				zap.String("id", component.ID))
			return
		}

		// Create healing action
		action := o.createHealingAction(component, diagnosis)

		// Schedule healing
		o.scheduleHealing(action)
	}

	// Update component state based on diagnosis
	o.updateComponentState(component, diagnosis)
}

// runPredictiveAnalysis performs predictive failure analysis
func (o *HealingOrchestrator) runPredictiveAnalysis(ctx context.Context) {
	defer o.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.shutdownCh:
			return
		case <-ticker.C:
			o.components.Range(func(key, value interface{}) bool {
				component := value.(*Component)
				o.predictFailures(ctx, component)
				return true
			})
		}
	}
}

// predictFailures predicts potential failures for a component
func (o *HealingOrchestrator) predictFailures(ctx context.Context, component *Component) {
	// Collect historical metrics
	metrics := o.collectComponentMetrics(component)

	// Time series prediction
	predictions := o.predictor.timeSeriesPredictor.Predict(metrics)

	// Anomaly detection
	anomalyScore := o.predictor.anomalyDetector.Detect(metrics)

	// Risk calculation
	riskScore := o.predictor.riskCalculator.Calculate(component, predictions, anomalyScore)

	// Take preemptive action if risk is high
	if riskScore > o.config.ConfidenceThreshold {
		o.logger.Warn("High failure risk detected",
			zap.String("component", component.ID),
			zap.Float64("risk_score", riskScore))

		// Create predictive healing action
		action := &HealingAction{
			ID:         fmt.Sprintf("pred-%s-%d", component.ID, time.Now().Unix()),
			Type:       "predictive",
			Component:  component.ID,
			Strategy:   StrategyPredictive,
			Priority:   int(riskScore * 10),
			Confidence: riskScore,
			StartTime:  time.Now(),
		}

		// Schedule preemptive healing
		o.scheduleHealing(action)

		// Update metrics
		o.metrics.predictedFailures.Inc()
	}
}

// executeHealingActions executes scheduled healing actions
func (o *HealingOrchestrator) executeHealingActions(ctx context.Context) {
	defer o.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.shutdownCh:
			return
		default:
			// Get next action from queue
			action := o.scheduler.getNext()
			if action == nil {
				time.Sleep(1 * time.Second)
				continue
			}

			// Check rate limit
			if !o.scheduler.rateLimiter.Allow() {
				// Re-queue the action
				o.scheduler.queue.Push(action, action.Priority)
				time.Sleep(100 * time.Millisecond)
				continue
			}

			// Check concurrent healing limit
			if atomic.LoadInt32(&o.scheduler.currentLoad) >= int32(o.config.MaxConcurrentHeals) {
				// Re-queue the action
				o.scheduler.queue.Push(action, action.Priority)
				time.Sleep(100 * time.Millisecond)
				continue
			}

			// Execute healing asynchronously
			atomic.AddInt32(&o.scheduler.currentLoad, 1)
			go o.executeHealing(ctx, action)
		}
	}
}

// executeHealing executes a single healing action
func (o *HealingOrchestrator) executeHealing(ctx context.Context, action *HealingAction) {
	defer atomic.AddInt32(&o.scheduler.currentLoad, -1)

	startTime := time.Now()
	o.metrics.healingAttempts.Inc()

	// Get component
	value, exists := o.components.Load(action.Component)
	if !exists {
		o.logger.Error("Component not found for healing",
			zap.String("component", action.Component))
		return
	}
	component := value.(*Component)

	// Get healer
	healer, exists := o.healers[component.Type]
	if !exists {
		o.logger.Error("No healer found for component type",
			zap.String("type", component.Type))
		return
	}

	// Execute healing
	var err error
	if o.config.DryRun {
		o.logger.Info("DRY RUN: Would execute healing",
			zap.String("action", action.ID),
			zap.String("component", component.ID))
		err = nil
	} else {
		err = healer.Heal(ctx, component, action)
	}

	if err != nil {
		o.handleHealingFailure(ctx, component, action, err)
		o.metrics.healingFailures.Inc()
	} else {
		o.handleHealingSuccess(ctx, component, action)
		o.metrics.healingSuccesses.Inc()
	}

	// Record healing duration
	duration := time.Since(startTime)
	o.metrics.healingDuration.Observe(duration.Seconds())

	// Update history
	endTime := time.Now()
	action.EndTime = &endTime
	action.Success = err == nil
	o.history.Add(action)
}

// handleHealingSuccess handles successful healing
func (o *HealingOrchestrator) handleHealingSuccess(ctx context.Context, component *Component, action *HealingAction) {
	o.logger.Info("Healing successful",
		zap.String("action", action.ID),
		zap.String("component", component.ID),
		zap.Duration("duration", time.Since(action.StartTime)))

	// Update component state
	component.mu.Lock()
	component.State = HealthHealthy
	now := time.Now()
	component.LastHealed = &now
	component.HealCount++
	component.mu.Unlock()

	// Validate healing
	healer := o.healers[component.Type]
	if err := healer.Validate(ctx, component); err != nil {
		o.logger.Warn("Healing validation failed",
			zap.String("component", component.ID),
			zap.Error(err))
	}

	// Update metrics
	if action.Strategy == StrategyPredictive {
		o.metrics.preventedIncidents.Inc()
	}
}

// handleHealingFailure handles failed healing
func (o *HealingOrchestrator) handleHealingFailure(ctx context.Context, component *Component, action *HealingAction, err error) {
	o.logger.Error("Healing failed",
		zap.String("action", action.ID),
		zap.String("component", component.ID),
		zap.Error(err))

	// Attempt rollback if configured
	if o.config.RollbackOnFailure && action.RollbackID != "" {
		healer := o.healers[component.Type]
		rollbackErr := healer.Rollback(ctx, component, action)
		if rollbackErr != nil {
			o.logger.Error("Rollback failed",
				zap.String("component", component.ID),
				zap.Error(rollbackErr))
		}
	}

	// Update component state
	component.mu.Lock()
	component.State = HealthFailed
	component.mu.Unlock()

	// Escalate to incident response if critical
	if component.State == HealthFailed || component.State == HealthCritical {
		o.escalateToIncidentResponse(component, action, err)
	}
}

// Helper functions

func (o *HealingOrchestrator) strategyName() string {
	switch o.config.Strategy {
	case StrategyReactive:
		return "reactive"
	case StrategyProactive:
		return "proactive"
	case StrategyPredictive:
		return "predictive"
	case StrategyAdaptive:
		return "adaptive"
	default:
		return "unknown"
	}
}

func (o *HealingOrchestrator) isInCooldown(component *Component) bool {
	if component.LastHealed == nil {
		return false
	}
	return time.Since(*component.LastHealed) < o.config.HealingCooldown
}

func (o *HealingOrchestrator) createHealingAction(component *Component, diagnosis *Diagnosis) *HealingAction {
	action := diagnosis.RecommendedAction
	if action == nil {
		action = &HealingAction{
			ID:        fmt.Sprintf("heal-%s-%d", component.ID, time.Now().Unix()),
			Type:      "automatic",
			Component: component.ID,
			Strategy:  o.config.Strategy,
		}
	}
	action.Priority = int(diagnosis.Severity * 10)
	action.Confidence = 1.0 - diagnosis.RiskScore
	action.StartTime = time.Now()
	return action
}

func (o *HealingOrchestrator) scheduleHealing(action *HealingAction) {
	o.scheduler.queue.Push(action, float64(action.Priority))
	o.logger.Info("Healing action scheduled",
		zap.String("action", action.ID),
		zap.String("component", action.Component),
		zap.Int("priority", action.Priority))
}

func (o *HealingOrchestrator) updateComponentState(component *Component, diagnosis *Diagnosis) {
	component.mu.Lock()
	defer component.mu.Unlock()

	// Map severity to health state
	if diagnosis.Severity < 0.3 {
		component.State = HealthHealthy
	} else if diagnosis.Severity < 0.5 {
		component.State = HealthDegraded
	} else if diagnosis.Severity < 0.8 {
		component.State = HealthCritical
	} else {
		component.State = HealthFailed
	}

	component.LastCheck = time.Now()
}

func (o *HealingOrchestrator) collectComponentMetrics(component *Component) []float64 {
	component.mu.RLock()
	defer component.mu.RUnlock()

	return []float64{
		component.Metrics.CPU,
		component.Metrics.Memory,
		component.Metrics.Disk,
		component.Metrics.NetworkLatency,
		component.Metrics.ErrorRate,
		component.Metrics.RequestRate,
	}
}

func (o *HealingOrchestrator) escalateToIncidentResponse(component *Component, action *HealingAction, err error) {
	o.logger.Error("Escalating failed healing to incident response",
		zap.String("component", component.ID),
		zap.String("action", action.ID),
		zap.Error(err))
	// Would trigger incident response system here
}

func (o *HealingOrchestrator) collectMetrics(ctx context.Context) {
	defer o.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.shutdownCh:
			return
		case <-ticker.C:
			// Update component health metrics
			o.components.Range(func(key, value interface{}) bool {
				component := value.(*Component)
				o.metrics.componentHealth.WithLabelValues(component.ID).Set(float64(component.State))
				return true
			})

			// Update history metrics
			o.history.mu.RLock()
			for componentID, rate := range o.history.successRate {
				o.metrics.componentHealth.WithLabelValues(componentID + "_success_rate").Set(rate)
			}
			o.history.mu.RUnlock()
		}
	}
}

// Supporting functions

func NewTimeSeriesPredictor(horizon int) *TimeSeriesPredictor {
	return &TimeSeriesPredictor{
		model: &ARIMAModel{
			p: 2, // AR order
			d: 1, // Differencing
			q: 2, // MA order
		},
		horizon:    horizon,
		confidence: 0.95,
	}
}

func (t *TimeSeriesPredictor) Predict(data []float64) []float64 {
	// Simplified prediction - would implement ARIMA
	predictions := make([]float64, t.horizon)
	if len(data) > 0 {
		lastValue := data[len(data)-1]
		for i := range predictions {
			predictions[i] = lastValue * (1.0 + 0.01*float64(i))
		}
	}
	return predictions
}

func NewAnomalyDetector(algorithm AnomalyAlgorithm, sensitivity float64) *AnomalyDetector {
	return &AnomalyDetector{
		algorithm:    algorithm,
		sensitivity:  sensitivity,
		windowSize:   100,
		baseline:     make(map[string]*BaselineModel),
		anomalyScore: make(map[string]float64),
	}
}

func (a *AnomalyDetector) Detect(data []float64) float64 {
	// Simplified anomaly detection
	if len(data) < 2 {
		return 0.0
	}

	// Calculate mean and stddev
	var sum, sumSq float64
	for _, v := range data {
		sum += v
		sumSq += v * v
	}
	mean := sum / float64(len(data))
	variance := (sumSq / float64(len(data))) - (mean * mean)
	stddev := math.Sqrt(variance)

	// Calculate z-score for last value
	if stddev == 0 {
		return 0.0
	}
	zScore := math.Abs((data[len(data)-1] - mean) / stddev)

	// Convert to anomaly score (0-1)
	anomalyScore := 1.0 - math.Exp(-zScore/2.0)
	return anomalyScore * a.sensitivity
}

func NewRiskCalculator() *RiskCalculator {
	return &RiskCalculator{
		weights: map[string]float64{
			"prediction":  0.3,
			"anomaly":     0.3,
			"historical":  0.2,
			"dependency":  0.2,
		},
		historicalFailures: make(map[string][]FailureEvent),
		mtbf:              make(map[string]time.Duration),
	}
}

func (r *RiskCalculator) Calculate(component *Component, predictions []float64, anomalyScore float64) float64 {
	riskScore := 0.0

	// Prediction risk
	if len(predictions) > 0 {
		maxPredicted := predictions[0]
		for _, p := range predictions {
			if p > maxPredicted {
				maxPredicted = p
			}
		}
		predictionRisk := math.Min(maxPredicted/100.0, 1.0)
		riskScore += predictionRisk * r.weights["prediction"]
	}

	// Anomaly risk
	riskScore += anomalyScore * r.weights["anomaly"]

	// Historical risk
	if failures, exists := r.historicalFailures[component.ID]; exists {
		recentFailures := 0
		cutoff := time.Now().Add(-24 * time.Hour)
		for _, f := range failures {
			if f.Timestamp.After(cutoff) {
				recentFailures++
			}
		}
		historicalRisk := math.Min(float64(recentFailures)/10.0, 1.0)
		riskScore += historicalRisk * r.weights["historical"]
	}

	// Dependency risk
	dependencyRisk := float64(len(component.Dependencies)) / 20.0
	riskScore += math.Min(dependencyRisk, 1.0) * r.weights["dependency"]

	return math.Min(riskScore, 1.0)
}

func NewHealingHistory() *HealingHistory {
	return &HealingHistory{
		actions:      make([]HealingAction, 0),
		successRate:  make(map[string]float64),
		avgHealTime:  make(map[string]time.Duration),
		lastFailures: make(map[string]time.Time),
	}
}

func (h *HealingHistory) Add(action *HealingAction) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.actions = append(h.actions, *action)

	// Update success rate
	componentActions := 0
	componentSuccesses := 0
	totalHealTime := time.Duration(0)

	for _, a := range h.actions {
		if a.Component == action.Component {
			componentActions++
			if a.Success {
				componentSuccesses++
				if a.EndTime != nil {
					totalHealTime += a.EndTime.Sub(a.StartTime)
				}
			} else if a.EndTime != nil {
				h.lastFailures[a.Component] = *a.EndTime
			}
		}
	}

	if componentActions > 0 {
		h.successRate[action.Component] = float64(componentSuccesses) / float64(componentActions)
		if componentSuccesses > 0 {
			h.avgHealTime[action.Component] = totalHealTime / time.Duration(componentSuccesses)
		}
	}
}

func (s *HealingScheduler) getNext() *HealingAction {
	return s.queue.Pop()
}

func (q *PriorityQueue) Push(action *HealingAction, priority float64) {
	q.mu.Lock()
	defer q.mu.Unlock()

	item := &QueueItem{
		Action:   action,
		Priority: priority,
		AddedAt:  time.Now(),
	}

	// Insert in priority order
	inserted := false
	for i, existing := range q.items {
		if priority > existing.Priority {
			q.items = append(q.items[:i], append([]*QueueItem{item}, q.items[i:]...)...)
			inserted = true
			break
		}
	}

	if !inserted {
		q.items = append(q.items, item)
	}
}

func (q *PriorityQueue) Pop() *HealingAction {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.items) == 0 {
		return nil
	}

	item := q.items[0]
	q.items = q.items[1:]
	return item.Action
}

func NewHealingMetrics() *HealingMetrics {
	return &HealingMetrics{
		healingAttempts: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "self_healing_attempts_total",
			Help: "Total number of self-healing attempts",
		}),
		healingSuccesses: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "self_healing_successes_total",
			Help: "Total number of successful self-healing actions",
		}),
		healingFailures: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "self_healing_failures_total",
			Help: "Total number of failed self-healing actions",
		}),
		healingDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "self_healing_duration_seconds",
			Help:    "Duration of self-healing actions",
			Buckets: prometheus.ExponentialBuckets(1, 2, 10),
		}),
		predictedFailures: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "predicted_failures_total",
			Help: "Total number of predicted failures",
		}),
		preventedIncidents: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "prevented_incidents_total",
			Help: "Total number of incidents prevented by predictive healing",
		}),
		componentHealth: *prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "component_health_state",
			Help: "Current health state of components",
		}, []string{"component"}),
		riskScore: *prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "component_risk_score",
			Help: "Current risk score for components",
		}, []string{"component"}),
	}
}