package autoscaling

import (
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// DefaultScalingDecisionEngine implements the ScalingDecisionEngine interface
type DefaultScalingDecisionEngine struct {
	mu          sync.RWMutex
	logger      *logrus.Logger
	thresholds  *ScalingThresholds
	cooldowns   map[string]time.Time // targetID -> last scaling time
	scaleStates map[string]*ScaleState // targetID -> current scale state
}

// ScaleState represents the current scaling state for a target
type ScaleState struct {
	CurrentReplicas int
	DesiredReplicas int
	LastScaleTime   time.Time
	ScaleHistory    []ScalingEvent
}

// ScalingEvent represents a historical scaling event
type ScalingEvent struct {
	Timestamp time.Time
	Action    ScalingAction
	From      int
	To        int
	Reason    string
}

// NewDefaultScalingDecisionEngine creates a new scaling decision engine
func NewDefaultScalingDecisionEngine(logger *logrus.Logger) *DefaultScalingDecisionEngine {
	return &DefaultScalingDecisionEngine{
		logger:   logger,
		cooldowns: make(map[string]time.Time),
		scaleStates: make(map[string]*ScaleState),
		thresholds: &ScalingThresholds{
			CPUScaleUpThreshold:      0.7,
			CPUScaleDownThreshold:    0.3,
			MemoryScaleUpThreshold:   0.8,
			MemoryScaleDownThreshold: 0.4,
			MinReplicas:              1,
			MaxReplicas:              10,
			CooldownPeriod:           5 * time.Minute,
			ScaleUpStabilization:     30 * time.Second,
			ScaleDownStabilization:   5 * time.Minute,
			PredictionWeight:         0.3, // 30% weight to prediction, 70% to current
		},
	}
}

// MakeDecision makes a scaling decision based on prediction and current metrics
func (e *DefaultScalingDecisionEngine) MakeDecision(prediction *ResourcePrediction, current *MetricsData) (*ScalingDecision, error) {
	if prediction == nil || current == nil {
		return nil, fmt.Errorf("prediction and current metrics cannot be nil")
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	targetID := current.TargetID
	now := time.Now()

	// Initialize scale state if not exists
	if _, exists := e.scaleStates[targetID]; !exists {
		e.scaleStates[targetID] = &ScaleState{
			CurrentReplicas: current.ActiveVMs,
			DesiredReplicas: current.ActiveVMs,
			LastScaleTime:   now,
			ScaleHistory:    make([]ScalingEvent, 0),
		}
	}

	scaleState := e.scaleStates[targetID]

	// Check cooldown period
	if lastCooldown, exists := e.cooldowns[targetID]; exists {
		if now.Before(lastCooldown.Add(e.thresholds.CooldownPeriod)) {
			return &ScalingDecision{
				TargetID:      targetID,
				DecisionTime:  now,
				Action:        ScalingActionNoAction,
				CurrentScale:  scaleState.CurrentReplicas,
				TargetScale:   scaleState.CurrentReplicas,
				Reason:        fmt.Sprintf("In cooldown period until %v", lastCooldown.Add(e.thresholds.CooldownPeriod)),
				Confidence:    1.0,
				CooldownUntil: lastCooldown.Add(e.thresholds.CooldownPeriod),
			}, nil
		}
	}

	// Calculate combined metrics using prediction weight
	combinedCPU := e.combineMetrics(current.CPUUsage, prediction.PredictedCPU)
	combinedMemory := e.combineMetrics(current.MemoryUsage, prediction.PredictedMemory)

	// Make scaling decision
	action, targetScale, reason, confidence := e.evaluateScalingNeed(
		combinedCPU,
		combinedMemory,
		scaleState,
		prediction,
	)

	decision := &ScalingDecision{
		TargetID:     targetID,
		DecisionTime: now,
		Action:       action,
		CurrentScale: scaleState.CurrentReplicas,
		TargetScale:  targetScale,
		Reason:       reason,
		Confidence:   confidence,
		Metadata: map[string]interface{}{
			"current_cpu":      current.CPUUsage,
			"predicted_cpu":    prediction.PredictedCPU,
			"combined_cpu":     combinedCPU,
			"current_memory":   current.MemoryUsage,
			"predicted_memory": prediction.PredictedMemory,
			"combined_memory":  combinedMemory,
			"trend_direction":  prediction.TrendDirection,
			"anomaly_score":    prediction.AnomalyScore,
		},
	}

	// Update state if scaling action is needed
	if action != ScalingActionNoAction {
		e.updateScaleState(targetID, action, targetScale, reason, now)
		e.cooldowns[targetID] = now
	}

	e.logger.WithFields(logrus.Fields{
		"target_id":       targetID,
		"action":          action,
		"current_scale":   decision.CurrentScale,
		"target_scale":    decision.TargetScale,
		"combined_cpu":    combinedCPU,
		"combined_memory": combinedMemory,
		"confidence":      confidence,
		"reason":          reason,
	}).Info("Scaling decision made")

	return decision, nil
}

// SetThresholds sets the scaling thresholds
func (e *DefaultScalingDecisionEngine) SetThresholds(thresholds *ScalingThresholds) error {
	if thresholds == nil {
		return fmt.Errorf("thresholds cannot be nil")
	}

	if err := e.validateThresholds(thresholds); err != nil {
		return fmt.Errorf("invalid thresholds: %w", err)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	e.thresholds = thresholds

	e.logger.WithFields(logrus.Fields{
		"cpu_scale_up":     thresholds.CPUScaleUpThreshold,
		"cpu_scale_down":   thresholds.CPUScaleDownThreshold,
		"memory_scale_up":  thresholds.MemoryScaleUpThreshold,
		"memory_scale_down": thresholds.MemoryScaleDownThreshold,
		"min_replicas":     thresholds.MinReplicas,
		"max_replicas":     thresholds.MaxReplicas,
	}).Info("Scaling thresholds updated")

	return nil
}

// GetThresholds returns current scaling thresholds
func (e *DefaultScalingDecisionEngine) GetThresholds() *ScalingThresholds {
	e.mu.RLock()
	defer e.mu.RUnlock()

	// Return a copy to avoid race conditions
	thresholds := *e.thresholds
	return &thresholds
}

// GetScaleState returns the current scale state for a target
func (e *DefaultScalingDecisionEngine) GetScaleState(targetID string) (*ScaleState, bool) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	state, exists := e.scaleStates[targetID]
	if !exists {
		return nil, false
	}

	// Return a copy
	stateCopy := *state
	stateCopy.ScaleHistory = make([]ScalingEvent, len(state.ScaleHistory))
	copy(stateCopy.ScaleHistory, state.ScaleHistory)

	return &stateCopy, true
}

// GetScalingHistory returns the scaling history for a target
func (e *DefaultScalingDecisionEngine) GetScalingHistory(targetID string, limit int) ([]ScalingEvent, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	state, exists := e.scaleStates[targetID]
	if !exists {
		return nil, fmt.Errorf("no scaling state found for target %s", targetID)
	}

	history := state.ScaleHistory
	if limit > 0 && len(history) > limit {
		// Return the most recent events
		start := len(history) - limit
		history = history[start:]
	}

	// Return a copy
	result := make([]ScalingEvent, len(history))
	copy(result, history)

	return result, nil
}

// Private methods

func (e *DefaultScalingDecisionEngine) combineMetrics(current, predicted float64) float64 {
	weight := e.thresholds.PredictionWeight
	return current*(1.0-weight) + predicted*weight
}

func (e *DefaultScalingDecisionEngine) evaluateScalingNeed(
	cpu, memory float64,
	state *ScaleState,
	prediction *ResourcePrediction,
) (ScalingAction, int, string, float64) {

	currentReplicas := state.CurrentReplicas
	targetReplicas := currentReplicas

	// Check if we need to scale up
	scaleUpNeeded := false
	scaleUpReason := ""

	if cpu > e.thresholds.CPUScaleUpThreshold {
		scaleUpNeeded = true
		scaleUpReason = fmt.Sprintf("CPU usage %.2f%% exceeds threshold %.2f%%", 
			cpu*100, e.thresholds.CPUScaleUpThreshold*100)
	} else if memory > e.thresholds.MemoryScaleUpThreshold {
		scaleUpNeeded = true
		scaleUpReason = fmt.Sprintf("Memory usage %.2f%% exceeds threshold %.2f%%",
			memory*100, e.thresholds.MemoryScaleUpThreshold*100)
	}

	// Consider trend direction for proactive scaling
	if prediction.TrendDirection == TrendIncreasing && prediction.Confidence > 0.7 {
		if cpu > e.thresholds.CPUScaleUpThreshold*0.8 {
			scaleUpNeeded = true
			scaleUpReason = fmt.Sprintf("Proactive scaling due to increasing trend (confidence: %.2f)", 
				prediction.Confidence)
		}
	}

	// Check if we need to scale down
	scaleDownNeeded := false
	scaleDownReason := ""

	if cpu < e.thresholds.CPUScaleDownThreshold && memory < e.thresholds.MemoryScaleDownThreshold {
		scaleDownNeeded = true
		scaleDownReason = fmt.Sprintf("CPU usage %.2f%% and memory usage %.2f%% below thresholds",
			cpu*100, memory*100)
	}

	// Determine action and target scale
	var action ScalingAction
	var reason string
	confidence := 0.8 // Base confidence

	if scaleUpNeeded && !scaleDownNeeded {
		targetReplicas = e.calculateScaleUpTarget(currentReplicas, cpu, memory)
		if targetReplicas > currentReplicas && targetReplicas <= e.thresholds.MaxReplicas {
			action = ScalingActionScaleUp
			reason = scaleUpReason
			confidence = e.calculateConfidence(prediction, true)
		} else {
			action = ScalingActionNoAction
			reason = "Scale up desired but would exceed max replicas or no change needed"
		}
	} else if scaleDownNeeded && !scaleUpNeeded {
		targetReplicas = e.calculateScaleDownTarget(currentReplicas, cpu, memory)
		if targetReplicas < currentReplicas && targetReplicas >= e.thresholds.MinReplicas {
			action = ScalingActionScaleDown
			reason = scaleDownReason
			confidence = e.calculateConfidence(prediction, false)
		} else {
			action = ScalingActionNoAction
			reason = "Scale down desired but would go below min replicas or no change needed"
		}
	} else {
		action = ScalingActionNoAction
		reason = "Metrics within acceptable thresholds"
		confidence = 1.0
	}

	// Handle conflicting signals (both scale up and down needed)
	if scaleUpNeeded && scaleDownNeeded {
		action = ScalingActionNoAction
		reason = "Conflicting scaling signals - no action taken"
		confidence = 0.5
	}

	return action, targetReplicas, reason, confidence
}

func (e *DefaultScalingDecisionEngine) calculateScaleUpTarget(current int, cpu, memory float64) int {
	// Calculate how much more capacity we need
	utilizationFactor := math.Max(cpu, memory)
	
	// Simple calculation: if we're at 80% utilization, we need 25% more capacity
	// targetCapacity = currentCapacity / (targetUtilization / currentUtilization)
	targetUtilization := 0.6 // Target 60% utilization after scaling
	scaleFactor := utilizationFactor / targetUtilization
	
	targetReplicas := int(math.Ceil(float64(current) * scaleFactor))
	
	// Don't scale up too aggressively - limit to doubling
	if targetReplicas > current*2 {
		targetReplicas = current * 2
	}
	
	// Ensure minimum increase
	if targetReplicas == current {
		targetReplicas = current + 1
	}
	
	return targetReplicas
}

func (e *DefaultScalingDecisionEngine) calculateScaleDownTarget(current int, cpu, memory float64) int {
	// Calculate how much capacity we can remove
	utilizationFactor := math.Max(cpu, memory)
	
	// Conservative scale down - ensure we maintain at least 70% target utilization
	targetUtilization := 0.7
	scaleFactor := utilizationFactor / targetUtilization
	
	targetReplicas := int(math.Floor(float64(current) * scaleFactor))
	
	// Don't scale down too aggressively - limit to halving
	if targetReplicas < current/2 {
		targetReplicas = current / 2
	}
	
	// Ensure minimum decrease
	if targetReplicas == current {
		targetReplicas = current - 1
	}
	
	// Ensure we don't go below minimum
	if targetReplicas < e.thresholds.MinReplicas {
		targetReplicas = e.thresholds.MinReplicas
	}
	
	return targetReplicas
}

func (e *DefaultScalingDecisionEngine) calculateConfidence(prediction *ResourcePrediction, isScaleUp bool) float64 {
	baseConfidence := prediction.Confidence
	
	// Adjust confidence based on trend direction consistency
	if isScaleUp && prediction.TrendDirection == TrendIncreasing {
		baseConfidence += 0.1
	} else if !isScaleUp && prediction.TrendDirection == TrendDecreasing {
		baseConfidence += 0.1
	} else if prediction.TrendDirection == TrendVolatile {
		baseConfidence -= 0.2
	}
	
	// Adjust for anomaly score
	if prediction.AnomalyScore > 0.5 {
		baseConfidence -= prediction.AnomalyScore * 0.3
	}
	
	// Ensure confidence is within valid range
	if baseConfidence < 0.1 {
		baseConfidence = 0.1
	} else if baseConfidence > 1.0 {
		baseConfidence = 1.0
	}
	
	return baseConfidence
}

func (e *DefaultScalingDecisionEngine) updateScaleState(targetID string, action ScalingAction, targetScale int, reason string, timestamp time.Time) {
	state := e.scaleStates[targetID]
	
	// Create scaling event
	event := ScalingEvent{
		Timestamp: timestamp,
		Action:    action,
		From:      state.CurrentReplicas,
		To:        targetScale,
		Reason:    reason,
	}
	
	// Update state
	state.DesiredReplicas = targetScale
	state.LastScaleTime = timestamp
	state.ScaleHistory = append(state.ScaleHistory, event)
	
	// Keep only last 100 events
	if len(state.ScaleHistory) > 100 {
		state.ScaleHistory = state.ScaleHistory[1:]
	}
}

func (e *DefaultScalingDecisionEngine) validateThresholds(thresholds *ScalingThresholds) error {
	if thresholds.CPUScaleUpThreshold <= thresholds.CPUScaleDownThreshold {
		return fmt.Errorf("CPU scale up threshold must be greater than scale down threshold")
	}
	
	if thresholds.MemoryScaleUpThreshold <= thresholds.MemoryScaleDownThreshold {
		return fmt.Errorf("Memory scale up threshold must be greater than scale down threshold")
	}
	
	if thresholds.MinReplicas < 1 {
		return fmt.Errorf("minimum replicas must be at least 1")
	}
	
	if thresholds.MaxReplicas < thresholds.MinReplicas {
		return fmt.Errorf("maximum replicas must be greater than or equal to minimum replicas")
	}
	
	if thresholds.PredictionWeight < 0 || thresholds.PredictionWeight > 1 {
		return fmt.Errorf("prediction weight must be between 0 and 1")
	}
	
	return nil
}