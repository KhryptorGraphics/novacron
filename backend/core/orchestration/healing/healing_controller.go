package healing

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
)

// DefaultHealingController implements the HealingController interface
type DefaultHealingController struct {
	mu                sync.RWMutex
	logger            *logrus.Logger
	eventBus          events.EventBus
	failureDetector   FailureDetector
	recoveryStrategies map[string]RecoveryStrategy
	
	// Configuration
	monitoringInterval time.Duration
	maxConcurrentHealing int
	
	// State
	running           bool
	targets           map[string]*HealingTarget
	healthStatuses    map[string]*HealthStatus
	activeHealings    map[string]*HealingDecision
	healingHistory    map[string][]*HealingDecision
	
	// Context for lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
	
	// Metrics
	healingAttempts   uint64
	successfulHealing uint64
	failedHealing     uint64
}

// NewDefaultHealingController creates a new healing controller
func NewDefaultHealingController(logger *logrus.Logger, eventBus events.EventBus) *DefaultHealingController {
	ctx, cancel := context.WithCancel(context.Background())
	
	// Initialize failure detector
	failureDetector := NewPhiAccrualFailureDetector(logger)
	
	// Initialize recovery strategies
	strategies := make(map[string]RecoveryStrategy)
	strategies["restart"] = NewRestartRecoveryStrategy(logger)
	strategies["migrate"] = NewMigrateRecoveryStrategy(logger)
	strategies["scale"] = NewScaleRecoveryStrategy(logger)
	strategies["failover"] = NewFailoverRecoveryStrategy(logger)
	
	return &DefaultHealingController{
		logger:             logger,
		eventBus:           eventBus,
		failureDetector:    failureDetector,
		recoveryStrategies: strategies,
		monitoringInterval: 30 * time.Second,
		maxConcurrentHealing: 5,
		ctx:                ctx,
		cancel:             cancel,
		targets:            make(map[string]*HealingTarget),
		healthStatuses:     make(map[string]*HealthStatus),
		activeHealings:     make(map[string]*HealingDecision),
		healingHistory:     make(map[string][]*HealingDecision),
	}
}

// StartMonitoring begins health monitoring
func (hc *DefaultHealingController) StartMonitoring() error {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	if hc.running {
		return fmt.Errorf("healing controller is already running")
	}

	hc.logger.Info("Starting healing controller monitoring")
	
	hc.running = true
	go hc.monitoringLoop()
	go hc.healthCheckLoop()

	hc.logger.WithFields(logrus.Fields{
		"monitoring_interval":    hc.monitoringInterval,
		"max_concurrent_healing": hc.maxConcurrentHealing,
		"registered_targets":     len(hc.targets),
		"recovery_strategies":    len(hc.recoveryStrategies),
	}).Info("Healing controller monitoring started")

	return nil
}

// StopMonitoring stops all monitoring activities
func (hc *DefaultHealingController) StopMonitoring() error {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	if !hc.running {
		return fmt.Errorf("healing controller is not running")
	}

	hc.logger.Info("Stopping healing controller monitoring")

	// Cancel all active healing operations
	for targetID := range hc.activeHealings {
		hc.logger.WithField("target_id", targetID).Info("Cancelling active healing operation")
	}

	hc.cancel()
	hc.running = false

	hc.logger.Info("Healing controller monitoring stopped")

	return nil
}

// RegisterTarget registers a target for health monitoring
func (hc *DefaultHealingController) RegisterTarget(target *HealingTarget) error {
	if target == nil {
		return fmt.Errorf("target cannot be nil")
	}

	if target.ID == "" {
		return fmt.Errorf("target ID cannot be empty")
	}

	hc.mu.Lock()
	defer hc.mu.Unlock()

	// Set default values if not provided
	if target.HealthCheckConfig == nil {
		target.HealthCheckConfig = &HealthCheckConfig{
			Interval:              30 * time.Second,
			Timeout:               10 * time.Second,
			HealthyThreshold:      3,
			UnhealthyThreshold:    2,
			FailureThreshold:      5,
			CheckType:             HealthCheckTypeMetrics,
			EnableProactiveChecks: true,
		}
	}

	if target.RecoveryConfig == nil {
		target.RecoveryConfig = &RecoveryConfig{
			EnableAutoRecovery:  true,
			MaxRecoveryAttempts: 3,
			RecoveryTimeout:     10 * time.Minute,
			BackoffStrategy:     BackoffExponential,
		}
	}

	target.CreatedAt = time.Now()
	target.UpdatedAt = time.Now()

	hc.targets[target.ID] = target

	// Initialize health status
	hc.healthStatuses[target.ID] = &HealthStatus{
		TargetID:            target.ID,
		Healthy:             true,
		HealthScore:         1.0,
		LastCheckTime:       time.Now(),
		ConsecutiveFailures: 0,
		ConsecutiveSuccess:  0,
		UptimePercentage:    100.0,
	}

	// Initialize healing history
	hc.healingHistory[target.ID] = make([]*HealingDecision, 0)

	hc.logger.WithFields(logrus.Fields{
		"target_id":   target.ID,
		"target_type": target.Type,
		"target_name": target.Name,
		"enabled":     target.Enabled,
	}).Info("Healing target registered")

	// Publish registration event
	if err := hc.publishEvent(EventTypeTargetRegistered, target.ID, nil, nil, nil, nil); err != nil {
		hc.logger.WithError(err).Error("Failed to publish target registration event")
	}

	return nil
}

// UnregisterTarget removes a target from monitoring
func (hc *DefaultHealingController) UnregisterTarget(targetID string) error {
	hc.mu.Lock()
	defer hc.mu.Unlock()

	if _, exists := hc.targets[targetID]; !exists {
		return fmt.Errorf("target %s not found", targetID)
	}

	// Cancel any active healing for this target
	if _, healing := hc.activeHealings[targetID]; healing {
		hc.logger.WithField("target_id", targetID).Info("Cancelling active healing operation for unregistered target")
		delete(hc.activeHealings, targetID)
	}

	delete(hc.targets, targetID)
	delete(hc.healthStatuses, targetID)
	delete(hc.healingHistory, targetID)

	hc.logger.WithField("target_id", targetID).Info("Healing target unregistered")

	return nil
}

// GetHealthStatus gets the current health status of a target
func (hc *DefaultHealingController) GetHealthStatus(targetID string) (*HealthStatus, error) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	status, exists := hc.healthStatuses[targetID]
	if !exists {
		return nil, fmt.Errorf("target %s not found", targetID)
	}

	// Return a copy to avoid race conditions
	statusCopy := *status
	if status.RecoveryStatus != nil {
		recoveryCopy := *status.RecoveryStatus
		statusCopy.RecoveryStatus = &recoveryCopy
	}

	return &statusCopy, nil
}

// TriggerHealing manually triggers healing for a target
func (hc *DefaultHealingController) TriggerHealing(targetID string, reason string) (*HealingDecision, error) {
	hc.mu.Lock()
	target, targetExists := hc.targets[targetID]
	status, statusExists := hc.healthStatuses[targetID]
	
	// Check if healing is already active
	if _, healing := hc.activeHealings[targetID]; healing {
		hc.mu.Unlock()
		return nil, fmt.Errorf("healing already in progress for target %s", targetID)
	}
	hc.mu.Unlock()

	if !targetExists || !statusExists {
		return nil, fmt.Errorf("target %s not found", targetID)
	}

	// Create failure info for manual trigger
	failureInfo := &FailureInfo{
		TargetID:        targetID,
		FailureType:     FailureTypeCustom,
		Severity:        SeverityMedium,
		Description:     fmt.Sprintf("Manual healing trigger: %s", reason),
		DetectedAt:      time.Now(),
		LastOccurrence:  time.Now(),
		OccurrenceCount: 1,
		Symptoms:        []string{"Manual intervention requested"},
	}

	return hc.executeHealing(target, status, failureInfo)
}

// GetTargets returns all registered targets
func (hc *DefaultHealingController) GetTargets() map[string]*HealingTarget {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	// Return copies to avoid race conditions
	targets := make(map[string]*HealingTarget)
	for id, target := range hc.targets {
		targetCopy := *target
		targets[id] = &targetCopy
	}

	return targets
}

// GetHealingHistory returns the healing history for a target
func (hc *DefaultHealingController) GetHealingHistory(targetID string, limit int) ([]*HealingDecision, error) {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	history, exists := hc.healingHistory[targetID]
	if !exists {
		return nil, fmt.Errorf("target %s not found", targetID)
	}

	// Return the most recent entries up to limit
	if limit > 0 && len(history) > limit {
		start := len(history) - limit
		history = history[start:]
	}

	// Return copies
	result := make([]*HealingDecision, len(history))
	for i, decision := range history {
		decisionCopy := *decision
		result[i] = &decisionCopy
	}

	return result, nil
}

// GetStatus returns the current status of the healing controller
func (hc *DefaultHealingController) GetStatus() *HealingControllerStatus {
	hc.mu.RLock()
	defer hc.mu.RUnlock()

	return &HealingControllerStatus{
		Running:            hc.running,
		TargetsCount:       len(hc.targets),
		ActiveHealings:     len(hc.activeHealings),
		HealingAttempts:    hc.healingAttempts,
		SuccessfulHealing:  hc.successfulHealing,
		FailedHealing:      hc.failedHealing,
		SuccessRate:        hc.calculateSuccessRate(),
		MonitoringInterval: hc.monitoringInterval,
	}
}

// HealingControllerStatus represents the status of the healing controller
type HealingControllerStatus struct {
	Running            bool          `json:"running"`
	TargetsCount       int           `json:"targets_count"`
	ActiveHealings     int           `json:"active_healings"`
	HealingAttempts    uint64        `json:"healing_attempts"`
	SuccessfulHealing  uint64        `json:"successful_healing"`
	FailedHealing      uint64        `json:"failed_healing"`
	SuccessRate        float64       `json:"success_rate"`
	MonitoringInterval time.Duration `json:"monitoring_interval"`
}

// Private methods

func (hc *DefaultHealingController) monitoringLoop() {
	ticker := time.NewTicker(hc.monitoringInterval)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			hc.performMonitoringCycle()
		}
	}
}

func (hc *DefaultHealingController) healthCheckLoop() {
	ticker := time.NewTicker(10 * time.Second) // More frequent health checks
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			hc.performHealthChecks()
		}
	}
}

func (hc *DefaultHealingController) performMonitoringCycle() {
	hc.logger.Debug("Performing healing monitoring cycle")

	hc.mu.RLock()
	targets := make([]*HealingTarget, 0, len(hc.targets))
	for _, target := range hc.targets {
		if target.Enabled {
			targetCopy := *target
			targets = append(targets, &targetCopy)
		}
	}
	hc.mu.RUnlock()

	// Check each target for failures and trigger healing if needed
	for _, target := range targets {
		if err := hc.evaluateTargetHealth(target); err != nil {
			hc.logger.WithError(err).WithField("target_id", target.ID).
				Error("Failed to evaluate target health")
		}
	}
}

func (hc *DefaultHealingController) performHealthChecks() {
	hc.mu.RLock()
	targets := make([]*HealingTarget, 0, len(hc.targets))
	for _, target := range hc.targets {
		if target.Enabled {
			targetCopy := *target
			targets = append(targets, &targetCopy)
		}
	}
	hc.mu.RUnlock()

	// Perform health checks for each target
	for _, target := range targets {
		healthSample := hc.collectHealthSample(target)
		if err := hc.failureDetector.AddSample(target.ID, healthSample); err != nil {
			hc.logger.WithError(err).WithField("target_id", target.ID).
				Error("Failed to add health sample")
		}
	}
}

func (hc *DefaultHealingController) collectHealthSample(target *HealingTarget) *HealthSample {
	// Simulate health check based on target type
	// In real implementation, this would perform actual health checks
	
	healthy := hc.simulateHealthCheck(target)
	responseTime := hc.simulateResponseTime(target, healthy)
	
	sample := &HealthSample{
		TargetID:     target.ID,
		Timestamp:    time.Now(),
		Healthy:      healthy,
		ResponseTime: responseTime,
		Metrics: map[string]float64{
			"cpu_usage":    hc.simulateMetric("cpu", 0.0, 1.0),
			"memory_usage": hc.simulateMetric("memory", 0.0, 1.0),
			"response_time": float64(responseTime.Milliseconds()),
		},
	}

	if !healthy {
		sample.ErrorMessage = "Simulated health check failure"
	}

	return sample
}

func (hc *DefaultHealingController) evaluateTargetHealth(target *HealingTarget) error {
	// Get health assessment from failure detector
	assessment, err := hc.failureDetector.IsHealthy(target.ID)
	if err != nil {
		return fmt.Errorf("failed to get health assessment: %w", err)
	}

	// Update health status
	hc.mu.Lock()
	status := hc.healthStatuses[target.ID]
	
	previouslyHealthy := status.Healthy
	status.Healthy = assessment.Healthy
	status.HealthScore = assessment.HealthScore
	status.LastCheckTime = assessment.Timestamp

	if assessment.Healthy {
		status.ConsecutiveFailures = 0
		status.ConsecutiveSuccess++
	} else {
		status.ConsecutiveFailures++
		status.ConsecutiveSuccess = 0
		status.FailureReason = fmt.Sprintf("Health assessment failed: %v", assessment.Reasons)
	}

	// Calculate uptime percentage (simplified)
	status.UptimePercentage = assessment.HealthScore * 100
	
	hc.mu.Unlock()

	// Check if healing is needed
	if !assessment.Healthy && previouslyHealthy {
		// Health degraded - publish event
		if err := hc.publishEvent(EventTypeHealthDegraded, target.ID, nil, nil, nil, status); err != nil {
			hc.logger.WithError(err).Error("Failed to publish health degraded event")
		}
	} else if assessment.Healthy && !previouslyHealthy {
		// Health restored - publish event
		if err := hc.publishEvent(EventTypeHealthRestored, target.ID, nil, nil, nil, status); err != nil {
			hc.logger.WithError(err).Error("Failed to publish health restored event")
		}
	}

	// Trigger healing if needed
	if !assessment.Healthy && target.RecoveryConfig.EnableAutoRecovery {
		if err := hc.considerHealing(target, status, assessment); err != nil {
			hc.logger.WithError(err).WithField("target_id", target.ID).
				Error("Failed to consider healing")
		}
	}

	return nil
}

func (hc *DefaultHealingController) considerHealing(target *HealingTarget, status *HealthStatus, assessment *HealthAssessment) error {
	hc.mu.RLock()
	_, healingActive := hc.activeHealings[target.ID]
	hc.mu.RUnlock()

	// Don't start new healing if already active
	if healingActive {
		return nil
	}

	// Check if failure threshold is reached
	if status.ConsecutiveFailures < target.HealthCheckConfig.FailureThreshold {
		return nil
	}

	// Create failure info
	failureInfo := &FailureInfo{
		TargetID:        target.ID,
		FailureType:     hc.determineFailureType(assessment),
		Severity:        hc.determineSeverity(status, assessment),
		Description:     fmt.Sprintf("Health assessment failure: %v", assessment.Reasons),
		DetectedAt:      time.Now(),
		LastOccurrence:  assessment.Timestamp,
		OccurrenceCount: status.ConsecutiveFailures,
		Symptoms:        assessment.Reasons,
	}

	// Execute healing
	if _, err := hc.executeHealing(target, status, failureInfo); err != nil {
		return fmt.Errorf("failed to execute healing: %w", err)
	}

	return nil
}

func (hc *DefaultHealingController) executeHealing(target *HealingTarget, status *HealthStatus, failure *FailureInfo) (*HealingDecision, error) {
	// Check concurrent healing limit
	hc.mu.RLock()
	if len(hc.activeHealings) >= hc.maxConcurrentHealing {
		hc.mu.RUnlock()
		return nil, fmt.Errorf("maximum concurrent healing operations reached")
	}
	hc.mu.RUnlock()

	// Select recovery strategy
	strategy := hc.selectRecoveryStrategy(failure, target)
	if strategy == nil {
		return nil, fmt.Errorf("no suitable recovery strategy found for failure type %s", failure.FailureType)
	}

	// Create healing decision
	decision := &HealingDecision{
		ID:           uuid.New().String(),
		TargetID:     target.ID,
		DecisionTime: time.Now(),
		FailureInfo:  failure,
		Strategy:     strategy.GetName(),
		EstimatedTime: strategy.EstimateTime(failure),
		Confidence:   hc.calculateHealingConfidence(failure, strategy),
		Reason:       fmt.Sprintf("Auto-healing triggered for %s failure", failure.FailureType),
		Status:       HealingStatusPending,
		Actions: []HealingAction{
			{
				Type:   ActionRestart, // Simplified
				Target: target.ID,
			},
		},
	}

	// Update state
	hc.mu.Lock()
	hc.activeHealings[target.ID] = decision
	hc.healingAttempts++
	
	// Update recovery status
	if status.RecoveryStatus == nil {
		status.RecoveryStatus = &RecoveryStatus{}
	}
	status.RecoveryStatus.InProgress = true
	status.RecoveryStatus.CurrentStrategy = strategy.GetName()
	status.RecoveryStatus.Attempts++
	status.RecoveryStatus.LastAttemptTime = time.Now()
	hc.mu.Unlock()

	// Publish healing started event
	if err := hc.publishEvent(EventTypeRecoveryStarted, target.ID, failure, decision, nil, nil); err != nil {
		hc.logger.WithError(err).Error("Failed to publish recovery started event")
	}

	// Execute healing asynchronously
	go hc.performHealing(target, decision, strategy)

	hc.logger.WithFields(logrus.Fields{
		"target_id":      target.ID,
		"decision_id":    decision.ID,
		"strategy":       decision.Strategy,
		"failure_type":   failure.FailureType,
		"estimated_time": decision.EstimatedTime,
	}).Info("Healing decision made and execution started")

	return decision, nil
}

func (hc *DefaultHealingController) performHealing(target *HealingTarget, decision *HealingDecision, strategy RecoveryStrategy) {
	hc.logger.WithFields(logrus.Fields{
		"target_id":   target.ID,
		"decision_id": decision.ID,
		"strategy":    strategy.GetName(),
	}).Info("Starting healing execution")

	// Update decision status
	decision.Status = HealingStatusExecuting

	// Execute recovery
	result, _ := strategy.Recover(decision.FailureInfo, target)
	
	// Update decision with result
	decision.Result = &HealingResult{
		Success:             result.Success,
		TotalTime:          result.Duration,
		StrategiesAttempted: []string{strategy.GetName()},
		FinalStrategy:      strategy.GetName(),
		RecoveryResults:    []*RecoveryResult{result},
	}

	hc.mu.Lock()
	defer hc.mu.Unlock()

	// Update status
	if result.Success {
		decision.Status = HealingStatusSuccessful
		hc.successfulHealing++
		
		// Update recovery status
		if status, exists := hc.healthStatuses[target.ID]; exists && status.RecoveryStatus != nil {
			status.RecoveryStatus.InProgress = false
			status.RecoveryStatus.LastAttemptResult = "success"
		}
	} else {
		decision.Status = HealingStatusFailed
		hc.failedHealing++
		
		// Update recovery status
		if status, exists := hc.healthStatuses[target.ID]; exists && status.RecoveryStatus != nil {
			status.RecoveryStatus.InProgress = false
			status.RecoveryStatus.LastAttemptResult = fmt.Sprintf("failed: %s", result.Message)
		}
	}

	// Remove from active healings
	delete(hc.activeHealings, target.ID)

	// Add to history
	hc.healingHistory[target.ID] = append(hc.healingHistory[target.ID], decision)
	
	// Keep only last 50 healing decisions
	if len(hc.healingHistory[target.ID]) > 50 {
		hc.healingHistory[target.ID] = hc.healingHistory[target.ID][1:]
	}

	hc.logger.WithFields(logrus.Fields{
		"target_id":   target.ID,
		"decision_id": decision.ID,
		"success":     result.Success,
		"duration":    result.Duration,
		"message":     result.Message,
	}).Info("Healing execution completed")

	// Publish healing completed event
	eventType := EventTypeRecoveryCompleted
	if err := hc.publishEvent(eventType, target.ID, decision.FailureInfo, decision, decision.Result, nil); err != nil {
		hc.logger.WithError(err).Error("Failed to publish recovery completed event")
	}
}

func (hc *DefaultHealingController) selectRecoveryStrategy(failure *FailureInfo, target *HealingTarget) RecoveryStrategy {
	// Get all strategies that can handle this failure
	var candidates []RecoveryStrategy
	for _, strategy := range hc.recoveryStrategies {
		if strategy.CanRecover(failure) {
			candidates = append(candidates, strategy)
		}
	}

	if len(candidates) == 0 {
		return nil
	}

	// Sort by priority (highest first)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].GetPriority() > candidates[j].GetPriority()
	})

	// Check preferred strategies from target config
	if target.RecoveryConfig.PreferredStrategies != nil {
		for _, preferred := range target.RecoveryConfig.PreferredStrategies {
			for _, candidate := range candidates {
				if candidate.GetName() == preferred {
					return candidate
				}
			}
		}
	}

	// Return highest priority strategy
	return candidates[0]
}

// Helper methods for simulation

func (hc *DefaultHealingController) simulateHealthCheck(target *HealingTarget) bool {
	// Simulate varying health based on target type and time
	now := time.Now()
	
	// Base health probability
	healthProb := 0.95
	
	// Vary based on target type
	switch target.Type {
	case TargetTypeNode:
		healthProb = 0.98 // Nodes are generally more stable
	case TargetTypeService:
		healthProb = 0.93 // Services may have more issues
	case TargetTypeVM:
		healthProb = 0.95 // VMs are moderately stable
	}
	
	// Add some time-based variation
	if now.Hour() >= 9 && now.Hour() <= 17 {
		healthProb -= 0.02 // Slightly less stable during business hours
	}
	
	// Use a simple pseudo-random check
	random := float64(now.UnixNano()%1000) / 1000.0
	return random < healthProb
}

func (hc *DefaultHealingController) simulateResponseTime(target *HealingTarget, healthy bool) time.Duration {
	base := 50 * time.Millisecond
	
	if !healthy {
		base *= 5 // Unhealthy targets respond slower
	}
	
	// Add some variation
	variation := time.Duration(time.Now().UnixNano()%int64(base/2))
	return base + variation
}

func (hc *DefaultHealingController) simulateMetric(metricType string, min, max float64) float64 {
	random := float64(time.Now().UnixNano()%1000) / 1000.0
	return min + (max-min)*random
}

func (hc *DefaultHealingController) determineFailureType(assessment *HealthAssessment) FailureType {
	// Determine failure type based on assessment reasons
	for _, reason := range assessment.Reasons {
		if len(reason) > 0 {
			// Simplified mapping
			return FailureTypeUnresponsive
		}
	}
	return FailureTypeUnresponsive
}

func (hc *DefaultHealingController) determineSeverity(status *HealthStatus, assessment *HealthAssessment) FailureSeverity {
	if status.ConsecutiveFailures >= 10 {
		return SeverityCritical
	} else if status.ConsecutiveFailures >= 5 {
		return SeverityHigh
	} else if status.ConsecutiveFailures >= 3 {
		return SeverityMedium
	}
	return SeverityLow
}

func (hc *DefaultHealingController) calculateHealingConfidence(failure *FailureInfo, strategy RecoveryStrategy) float64 {
	baseConfidence := 0.8
	
	// Adjust based on failure severity
	switch failure.Severity {
	case SeverityLow:
		baseConfidence += 0.1
	case SeverityMedium:
		// No adjustment
	case SeverityHigh:
		baseConfidence -= 0.1
	case SeverityCritical:
		baseConfidence -= 0.2
	}
	
	// Adjust based on strategy priority
	if strategy.GetPriority() > 7 {
		baseConfidence += 0.1
	} else if strategy.GetPriority() < 4 {
		baseConfidence -= 0.1
	}
	
	return baseConfidence
}

func (hc *DefaultHealingController) calculateSuccessRate() float64 {
	if hc.healingAttempts == 0 {
		return 100.0
	}
	return (float64(hc.successfulHealing) / float64(hc.healingAttempts)) * 100.0
}

func (hc *DefaultHealingController) publishEvent(eventType EventType, targetID string, failure *FailureInfo, decision *HealingDecision, result *HealingResult, status *HealthStatus) error {
	if hc.eventBus == nil {
		return nil
	}

	event := &events.OrchestrationEvent{
		Type:      events.EventType(eventType),
		Source:    "healing-controller",
		Target:    targetID,
		Timestamp: time.Now(),
		Data: map[string]interface{}{
			"event_type": eventType,
			"target_id":  targetID,
		},
		Priority: events.PriorityNormal,
	}

	if failure != nil {
		event.Data["failure"] = failure
	}
	if decision != nil {
		event.Data["decision"] = decision
	}
	if result != nil {
		event.Data["result"] = result
	}
	if status != nil {
		event.Data["status"] = status
	}

	return hc.eventBus.Publish(hc.ctx, event)
}