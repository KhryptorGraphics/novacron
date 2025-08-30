package autoscaling

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
)

// DefaultAutoScaler implements the AutoScaler interface
type DefaultAutoScaler struct {
	mu               sync.RWMutex
	logger           *logrus.Logger
	metricsCollector MetricsCollector
	predictor        Predictor
	decisionEngine   ScalingDecisionEngine
	eventBus         events.EventBus
	
	// Configuration
	monitoringInterval time.Duration
	predictionHorizon  int // minutes
	
	// State
	running    bool
	ctx        context.Context
	cancel     context.CancelFunc
	targets    map[string]*AutoScalerTarget // targetID -> target config
	
	// Metrics
	decisionsCount  uint64
	predictionsCount uint64
	lastDecisionTime time.Time
}

// AutoScalerTarget represents a target for auto-scaling
type AutoScalerTarget struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Enabled     bool                   `json:"enabled"`
	Thresholds  *ScalingThresholds     `json:"thresholds"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// NewDefaultAutoScaler creates a new auto-scaler
func NewDefaultAutoScaler(logger *logrus.Logger, eventBus events.EventBus) *DefaultAutoScaler {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &DefaultAutoScaler{
		logger:             logger,
		eventBus:           eventBus,
		metricsCollector:   NewDefaultMetricsCollector(logger),
		predictor:          NewARIMAPredictor(ARIMAOrder{P: 3, D: 1, Q: 2}),
		decisionEngine:     NewDefaultScalingDecisionEngine(logger),
		monitoringInterval: 30 * time.Second,
		predictionHorizon:  30, // 30 minutes
		ctx:                ctx,
		cancel:             cancel,
		targets:            make(map[string]*AutoScalerTarget),
	}
}

// StartMonitoring begins monitoring and prediction
func (as *DefaultAutoScaler) StartMonitoring() error {
	as.mu.Lock()
	defer as.mu.Unlock()

	if as.running {
		return fmt.Errorf("auto-scaler is already running")
	}

	as.logger.Info("Starting auto-scaler monitoring")

	// Start metrics collection
	if err := as.metricsCollector.StartCollection(); err != nil {
		return fmt.Errorf("failed to start metrics collection: %w", err)
	}

	// Subscribe to metrics updates
	if err := as.metricsCollector.Subscribe(as); err != nil {
		return fmt.Errorf("failed to subscribe to metrics: %w", err)
	}

	// Train initial model with historical data if available
	go as.trainInitialModel()

	// Start monitoring loop
	as.running = true
	go as.monitoringLoop()

	as.logger.WithFields(logrus.Fields{
		"monitoring_interval": as.monitoringInterval,
		"prediction_horizon":  as.predictionHorizon,
		"targets_count":       len(as.targets),
	}).Info("Auto-scaler monitoring started")

	return nil
}

// StopMonitoring stops all monitoring activities
func (as *DefaultAutoScaler) StopMonitoring() error {
	as.mu.Lock()
	defer as.mu.Unlock()

	if !as.running {
		return fmt.Errorf("auto-scaler is not running")
	}

	as.logger.Info("Stopping auto-scaler monitoring")

	// Cancel context to stop all goroutines
	as.cancel()

	// Stop metrics collection
	if err := as.metricsCollector.StopCollection(); err != nil {
		as.logger.WithError(err).Error("Error stopping metrics collection")
	}

	as.running = false
	as.logger.Info("Auto-scaler monitoring stopped")

	return nil
}

// GetScalingDecision gets a scaling decision based on current metrics
func (as *DefaultAutoScaler) GetScalingDecision(targetID string) (*ScalingDecision, error) {
	as.mu.RLock()
	target, exists := as.targets[targetID]
	as.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("target %s not found", targetID)
	}

	if !target.Enabled {
		return &ScalingDecision{
			TargetID:     targetID,
			DecisionTime: time.Now(),
			Action:       ScalingActionNoAction,
			Reason:       "Auto-scaling disabled for this target",
			Confidence:   1.0,
		}, nil
	}

	// Get current metrics
	currentMetrics, err := as.metricsCollector.CollectMetrics()
	if err != nil {
		return nil, fmt.Errorf("failed to collect current metrics: %w", err)
	}

	// Get prediction
	prediction, err := as.GetPrediction(targetID, as.predictionHorizon)
	if err != nil {
		as.logger.WithError(err).Warn("Failed to get prediction, using current metrics only")
		// Create a simple prediction based on current metrics
		prediction = &ResourcePrediction{
			TargetID:         targetID,
			PredictionTime:   time.Now(),
			HorizonMinutes:   as.predictionHorizon,
			PredictedCPU:     currentMetrics.CPUUsage,
			PredictedMemory:  currentMetrics.MemoryUsage,
			PredictedLoad:    currentMetrics.CPUUsage,
			Confidence:       0.5,
			TrendDirection:   TrendStable,
			SeasonalFactor:   1.0,
			AnomalyScore:     0.0,
		}
	}

	// Make scaling decision
	decision, err := as.decisionEngine.MakeDecision(prediction, currentMetrics)
	if err != nil {
		return nil, fmt.Errorf("failed to make scaling decision: %w", err)
	}

	// Update metrics
	as.mu.Lock()
	as.decisionsCount++
	as.lastDecisionTime = time.Now()
	as.mu.Unlock()

	// Publish scaling decision event
	if err := as.publishScalingEvent(EventTypeScalingDecisionMade, targetID, decision, prediction); err != nil {
		as.logger.WithError(err).Error("Failed to publish scaling decision event")
	}

	return decision, nil
}

// GetPrediction gets a prediction for future resource needs
func (as *DefaultAutoScaler) GetPrediction(targetID string, horizonMinutes int) (*ResourcePrediction, error) {
	// Get current metrics
	currentMetrics, err := as.metricsCollector.CollectMetrics()
	if err != nil {
		return nil, fmt.Errorf("failed to collect current metrics: %w", err)
	}

	// Generate prediction
	prediction, err := as.predictor.Predict(currentMetrics, horizonMinutes)
	if err != nil {
		return nil, fmt.Errorf("prediction failed: %w", err)
	}

	// Update metrics
	as.mu.Lock()
	as.predictionsCount++
	as.mu.Unlock()

	// Publish prediction event
	if err := as.publishScalingEvent(EventTypePredictionGenerated, targetID, nil, prediction); err != nil {
		as.logger.WithError(err).Error("Failed to publish prediction event")
	}

	return prediction, nil
}

// UpdateMetrics updates the metrics store with new data
func (as *DefaultAutoScaler) UpdateMetrics(metrics *MetricsData) error {
	// This would typically be called by external systems
	// For now, we rely on the internal metrics collector
	as.logger.WithFields(logrus.Fields{
		"target_id":    metrics.TargetID,
		"cpu_usage":    metrics.CPUUsage,
		"memory_usage": metrics.MemoryUsage,
		"timestamp":    metrics.Timestamp,
	}).Debug("External metrics update received")

	return nil
}

// HandleMetrics implements MetricsHandler interface
func (as *DefaultAutoScaler) HandleMetrics(metrics *MetricsData) error {
	// This is called when new metrics are collected
	as.logger.WithFields(logrus.Fields{
		"target_id":    metrics.TargetID,
		"cpu_usage":    metrics.CPUUsage,
		"memory_usage": metrics.MemoryUsage,
	}).Debug("Processing new metrics")

	// Check if any targets need immediate attention
	for targetID, target := range as.targets {
		if !target.Enabled {
			continue
		}

		// Check for threshold breaches that need immediate action
		thresholds := as.decisionEngine.GetThresholds()
		if metrics.CPUUsage > thresholds.CPUScaleUpThreshold*1.2 || 
		   metrics.MemoryUsage > thresholds.MemoryScaleUpThreshold*1.2 {
			
			// Publish threshold breach event
			if err := as.publishScalingEvent(EventTypeThresholdsBreach, targetID, nil, nil); err != nil {
				as.logger.WithError(err).Error("Failed to publish threshold breach event")
			}
		}
	}

	return nil
}

// AddTarget adds a new auto-scaling target
func (as *DefaultAutoScaler) AddTarget(target *AutoScalerTarget) error {
	if target == nil {
		return fmt.Errorf("target cannot be nil")
	}

	if target.ID == "" {
		return fmt.Errorf("target ID cannot be empty")
	}

	as.mu.Lock()
	defer as.mu.Unlock()

	target.CreatedAt = time.Now()
	target.UpdatedAt = time.Now()

	as.targets[target.ID] = target

	// Set target-specific thresholds if provided
	if target.Thresholds != nil {
		if err := as.decisionEngine.SetThresholds(target.Thresholds); err != nil {
			return fmt.Errorf("failed to set target thresholds: %w", err)
		}
	}

	as.logger.WithFields(logrus.Fields{
		"target_id":   target.ID,
		"target_type": target.Type,
		"enabled":     target.Enabled,
	}).Info("Auto-scaling target added")

	return nil
}

// RemoveTarget removes an auto-scaling target
func (as *DefaultAutoScaler) RemoveTarget(targetID string) error {
	as.mu.Lock()
	defer as.mu.Unlock()

	if _, exists := as.targets[targetID]; !exists {
		return fmt.Errorf("target %s not found", targetID)
	}

	delete(as.targets, targetID)

	as.logger.WithField("target_id", targetID).Info("Auto-scaling target removed")

	return nil
}

// GetTargets returns all auto-scaling targets
func (as *DefaultAutoScaler) GetTargets() map[string]*AutoScalerTarget {
	as.mu.RLock()
	defer as.mu.RUnlock()

	// Return a copy to avoid race conditions
	targets := make(map[string]*AutoScalerTarget)
	for id, target := range as.targets {
		targetCopy := *target
		targets[id] = &targetCopy
	}

	return targets
}

// GetStatus returns the current status of the auto-scaler
func (as *DefaultAutoScaler) GetStatus() *AutoScalerStatus {
	as.mu.RLock()
	defer as.mu.RUnlock()

	return &AutoScalerStatus{
		Running:          as.running,
		TargetsCount:     len(as.targets),
		DecisionsCount:   as.decisionsCount,
		PredictionsCount: as.predictionsCount,
		LastDecisionTime: as.lastDecisionTime,
		ModelInfo:        as.predictor.GetModelInfo(),
		Uptime:           0, // Simplified for now
	}
}

// AutoScalerStatus represents the status of the auto-scaler
type AutoScalerStatus struct {
	Running          bool          `json:"running"`
	TargetsCount     int           `json:"targets_count"`
	DecisionsCount   uint64        `json:"decisions_count"`
	PredictionsCount uint64        `json:"predictions_count"`
	LastDecisionTime time.Time     `json:"last_decision_time"`
	ModelInfo        ModelInfo     `json:"model_info"`
	Uptime           time.Duration `json:"uptime"`
}

// Private methods

func (as *DefaultAutoScaler) monitoringLoop() {
	ticker := time.NewTicker(as.monitoringInterval)
	defer ticker.Stop()

	for {
		select {
		case <-as.ctx.Done():
			return
		case <-ticker.C:
			as.performMonitoringCycle()
		}
	}
}

func (as *DefaultAutoScaler) performMonitoringCycle() {
	as.logger.Debug("Performing auto-scaling monitoring cycle")

	// Check each enabled target
	as.mu.RLock()
	targets := make([]*AutoScalerTarget, 0, len(as.targets))
	for _, target := range as.targets {
		if target.Enabled {
			targetCopy := *target
			targets = append(targets, &targetCopy)
		}
	}
	as.mu.RUnlock()

	for _, target := range targets {
		if _, err := as.GetScalingDecision(target.ID); err != nil {
			as.logger.WithError(err).WithField("target_id", target.ID).
				Error("Failed to get scaling decision during monitoring cycle")
		}
	}
}

func (as *DefaultAutoScaler) trainInitialModel() {
	as.logger.Info("Training initial prediction model")

	// Get historical data for training
	end := time.Now()
	start := end.Add(-24 * time.Hour) // Last 24 hours

	historicalData, err := as.metricsCollector.GetHistoricalMetrics(start, end)
	if err != nil {
		as.logger.WithError(err).Warn("Failed to get historical data for initial training")
		return
	}

	if len(historicalData) < 10 {
		as.logger.Warn("Insufficient historical data for initial training")
		return
	}

	if err := as.predictor.Train(historicalData); err != nil {
		as.logger.WithError(err).Error("Failed to train initial prediction model")
		return
	}

	as.logger.WithFields(logrus.Fields{
		"data_points": len(historicalData),
		"accuracy":    as.predictor.GetAccuracy(),
		"model_info":  as.predictor.GetModelInfo().ModelType,
	}).Info("Initial prediction model trained successfully")

	// Publish model retrained event
	if err := as.publishScalingEvent(EventTypeModelRetrained, "global", nil, nil); err != nil {
		as.logger.WithError(err).Error("Failed to publish model retrained event")
	}
}

func (as *DefaultAutoScaler) publishScalingEvent(eventType EventType, targetID string, decision *ScalingDecision, prediction *ResourcePrediction) error {
	if as.eventBus == nil {
		return nil // Event bus not configured
	}

	event := &events.OrchestrationEvent{
		Type:      events.EventType(eventType),
		Source:    "autoscaler",
		Target:    targetID,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"event_type": eventType,
			"target_id":  targetID,
		},
		Priority: events.PriorityNormal,
	}

	if decision != nil {
		event.Data["decision"] = decision
	}

	if prediction != nil {
		event.Data["prediction"] = prediction
	}

	return as.eventBus.Publish(as.ctx, event)
}