package metrics

import (
	"sync"
	"time"
)

// AINetworkMetrics tracks AI network optimization metrics
type AINetworkMetrics struct {
	mu sync.RWMutex

	// RL Routing metrics
	RoutingDecisions      int64
	AvgRoutingLatency     time.Duration
	RoutingSuccessRate    float64

	// Congestion prediction
	CongestionPredictions int64
	PredictionAccuracy    float64
	ProactiveReroutes     int64

	// QoS classification
	FlowsClassified       int64
	ClassificationAccuracy float64
	QoSPolicyUpdates      int64

	// Self-healing
	FailuresDetected      int64
	HealingAttempts       int64
	HealingSuccessRate    float64
	AvgHealingTime        time.Duration

	// Anomaly detection
	AnomaliesDetected     int64
	FalsePositiveRate     float64
	DetectionLatency      time.Duration

	// Intent translation
	IntentsProcessed      int64
	TranslationSuccessRate float64
	AvgTranslationTime    time.Duration

	// Traffic engineering
	LinkUtilization       float64
	OptimizationRuns      int64

	// Network slicing
	ActiveSlices          int
	SliceCreationTime     time.Duration
	ResourceUtilization   float64

	// Digital twin
	SimulationsRun        int64
	PredictionAccuracy    float64

	// Optimization
	OptimizationIterations int64
	BestFitness           float64
	ParetoFrontSize       int
}

// NewMetrics creates new metrics collector
func NewMetrics() *AINetworkMetrics {
	return &AINetworkMetrics{}
}

// UpdateRoutingMetrics updates routing metrics
func (m *AINetworkMetrics) UpdateRoutingMetrics(decisions int64, latency time.Duration, successRate float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.RoutingDecisions = decisions
	m.AvgRoutingLatency = latency
	m.RoutingSuccessRate = successRate
}

// UpdateCongestionMetrics updates congestion prediction metrics
func (m *AINetworkMetrics) UpdateCongestionMetrics(predictions int64, accuracy float64, reroutes int64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.CongestionPredictions = predictions
	m.PredictionAccuracy = accuracy
	m.ProactiveReroutes = reroutes
}

// UpdateQoSMetrics updates QoS metrics
func (m *AINetworkMetrics) UpdateQoSMetrics(classified int64, accuracy float64, updates int64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.FlowsClassified = classified
	m.ClassificationAccuracy = accuracy
	m.QoSPolicyUpdates = updates
}

// UpdateHealingMetrics updates self-healing metrics
func (m *AINetworkMetrics) UpdateHealingMetrics(detected int64, attempts int64, successRate float64, healTime time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.FailuresDetected = detected
	m.HealingAttempts = attempts
	m.HealingSuccessRate = successRate
	m.AvgHealingTime = healTime
}

// UpdateAnomalyMetrics updates anomaly detection metrics
func (m *AINetworkMetrics) UpdateAnomalyMetrics(detected int64, fpRate float64, latency time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.AnomaliesDetected = detected
	m.FalsePositiveRate = fpRate
	m.DetectionLatency = latency
}

// UpdateIntentMetrics updates intent-based networking metrics
func (m *AINetworkMetrics) UpdateIntentMetrics(processed int64, successRate float64, transTime time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.IntentsProcessed = processed
	m.TranslationSuccessRate = successRate
	m.AvgTranslationTime = transTime
}

// UpdateTrafficEngineering updates traffic engineering metrics
func (m *AINetworkMetrics) UpdateTrafficEngineering(utilization float64, runs int64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.LinkUtilization = utilization
	m.OptimizationRuns = runs
}

// UpdateSlicingMetrics updates network slicing metrics
func (m *AINetworkMetrics) UpdateSlicingMetrics(activeSlices int, creationTime time.Duration, resourceUtil float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.ActiveSlices = activeSlices
	m.SliceCreationTime = creationTime
	m.ResourceUtilization = resourceUtil
}

// UpdateDigitalTwinMetrics updates digital twin metrics
func (m *AINetworkMetrics) UpdateDigitalTwinMetrics(simulations int64, accuracy float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.SimulationsRun = simulations
	m.PredictionAccuracy = accuracy
}

// UpdateOptimizationMetrics updates optimization metrics
func (m *AINetworkMetrics) UpdateOptimizationMetrics(iterations int64, fitness float64, paretoSize int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.OptimizationIterations = iterations
	m.BestFitness = fitness
	m.ParetoFrontSize = paretoSize
}

// GetAllMetrics returns all metrics
func (m *AINetworkMetrics) GetAllMetrics() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]interface{}{
		// RL Routing
		"routing_decisions":       m.RoutingDecisions,
		"avg_routing_latency_us":  m.AvgRoutingLatency.Microseconds(),
		"routing_success_rate":    m.RoutingSuccessRate,

		// Congestion
		"congestion_predictions":  m.CongestionPredictions,
		"prediction_accuracy":     m.PredictionAccuracy,
		"proactive_reroutes":      m.ProactiveReroutes,

		// QoS
		"flows_classified":         m.FlowsClassified,
		"classification_accuracy":  m.ClassificationAccuracy,
		"qos_policy_updates":       m.QoSPolicyUpdates,

		// Self-healing
		"failures_detected":       m.FailuresDetected,
		"healing_attempts":        m.HealingAttempts,
		"healing_success_rate":    m.HealingSuccessRate,
		"avg_healing_time_ms":     m.AvgHealingTime.Milliseconds(),

		// Anomaly detection
		"anomalies_detected":      m.AnomaliesDetected,
		"false_positive_rate":     m.FalsePositiveRate,
		"detection_latency_ms":    m.DetectionLatency.Milliseconds(),

		// Intent-based
		"intents_processed":        m.IntentsProcessed,
		"translation_success_rate": m.TranslationSuccessRate,
		"avg_translation_time_ms":  m.AvgTranslationTime.Milliseconds(),

		// Traffic engineering
		"link_utilization":        m.LinkUtilization,
		"te_optimization_runs":    m.OptimizationRuns,

		// Network slicing
		"active_slices":           m.ActiveSlices,
		"slice_creation_time_ms":  m.SliceCreationTime.Milliseconds(),
		"resource_utilization":    m.ResourceUtilization,

		// Digital twin
		"simulations_run":         m.SimulationsRun,
		"twin_prediction_accuracy": m.PredictionAccuracy,

		// Optimization
		"optimization_iterations": m.OptimizationIterations,
		"best_fitness":            m.BestFitness,
		"pareto_front_size":       m.ParetoFrontSize,
	}
}

// GetPerformanceSummary returns performance summary
func (m *AINetworkMetrics) GetPerformanceSummary() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return map[string]interface{}{
		"routing_performance": map[string]interface{}{
			"latency_target_met": m.AvgRoutingLatency < 1*time.Millisecond,
			"success_rate":       m.RoutingSuccessRate,
		},
		"prediction_performance": map[string]interface{}{
			"accuracy_target_met": m.PredictionAccuracy > 0.9,
			"accuracy":           m.PredictionAccuracy,
		},
		"qos_performance": map[string]interface{}{
			"accuracy_target_met": m.ClassificationAccuracy > 0.95,
			"accuracy":           m.ClassificationAccuracy,
		},
		"healing_performance": map[string]interface{}{
			"time_target_met": m.AvgHealingTime < 100*time.Millisecond,
			"success_rate":    m.HealingSuccessRate,
		},
		"anomaly_performance": map[string]interface{}{
			"latency_target_met": m.DetectionLatency < 1*time.Second,
			"false_positive_rate": m.FalsePositiveRate,
		},
		"utilization_performance": map[string]interface{}{
			"target_met": m.LinkUtilization > 0.95,
			"utilization": m.LinkUtilization,
		},
	}
}