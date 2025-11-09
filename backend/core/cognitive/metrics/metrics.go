// Package metrics provides cognitive AI performance tracking
package metrics

import (
	"sync"
	"time"

	"github.com/yourusername/novacron/backend/core/cognitive"
)

// MetricsCollector tracks comprehensive cognitive AI metrics
type MetricsCollector struct {
	metrics     *cognitive.CognitiveMetrics
	lock        sync.RWMutex
	startTime   time.Time
	samples     []MetricSample
	sampleLock  sync.RWMutex
}

// MetricSample represents a point-in-time measurement
type MetricSample struct {
	Timestamp           time.Time
	IntentAccuracy      float64
	TaskCompletionRate  float64
	ResponseLatency     float64
	ContextSwitchLatency float64
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics: &cognitive.CognitiveMetrics{
			IntentAccuracy:           0.95, // Initial target
			TaskCompletionRate:       0.90,
			UserSatisfaction:         4.5,
			ReasoningCorrectness:     0.90,
			AvgResponseLatency:       50.0,
			RecommendationAcceptance: 0.85,
			ContextSwitchLatency:     5.0,
		},
		startTime: time.Now(),
		samples:   []MetricSample{},
	}
}

// RecordIntentParsing records intent parsing metrics
func (mc *MetricsCollector) RecordIntentParsing(accuracy float64, success bool) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	mc.metrics.TotalIntents++

	// Running average
	alpha := 0.1
	mc.metrics.IntentAccuracy = alpha*accuracy + (1-alpha)*mc.metrics.IntentAccuracy

	if success {
		mc.metrics.TaskCompletionRate = alpha*1.0 + (1-alpha)*mc.metrics.TaskCompletionRate
	}
}

// RecordResponseLatency records response time
func (mc *MetricsCollector) RecordResponseLatency(latencyMs float64) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	alpha := 0.1
	mc.metrics.AvgResponseLatency = alpha*latencyMs + (1-alpha)*mc.metrics.AvgResponseLatency
}

// RecordReasoningResult records reasoning correctness
func (mc *MetricsCollector) RecordReasoningResult(correct bool, confidence float64) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	if correct {
		alpha := 0.1
		mc.metrics.ReasoningCorrectness = alpha*confidence + (1-alpha)*mc.metrics.ReasoningCorrectness
	}
}

// RecordRecommendation records recommendation metrics
func (mc *MetricsCollector) RecordRecommendation(accepted bool) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	mc.metrics.TotalRecommendations++

	if accepted {
		total := float64(mc.metrics.TotalRecommendations)
		mc.metrics.RecommendationAcceptance = (mc.metrics.RecommendationAcceptance*(total-1) + 1.0) / total
	} else {
		total := float64(mc.metrics.TotalRecommendations)
		mc.metrics.RecommendationAcceptance = (mc.metrics.RecommendationAcceptance * (total - 1)) / total
	}
}

// RecordContextSwitch records context switching latency
func (mc *MetricsCollector) RecordContextSwitch(latencyMs float64) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	alpha := 0.1
	mc.metrics.ContextSwitchLatency = alpha*latencyMs + (1-alpha)*mc.metrics.ContextSwitchLatency
}

// RecordUserSatisfaction records user satisfaction rating
func (mc *MetricsCollector) RecordUserSatisfaction(rating float64) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	// Ensure rating is 0-5
	if rating < 0 {
		rating = 0
	}
	if rating > 5 {
		rating = 5
	}

	alpha := 0.1
	mc.metrics.UserSatisfaction = alpha*rating + (1-alpha)*mc.metrics.UserSatisfaction
}

// RecordConversation records a completed conversation
func (mc *MetricsCollector) RecordConversation() {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	mc.metrics.TotalConversations++
}

// TakeSample captures current metrics as a sample
func (mc *MetricsCollector) TakeSample() {
	mc.lock.RLock()
	metrics := *mc.metrics
	mc.lock.RUnlock()

	sample := MetricSample{
		Timestamp:            time.Now(),
		IntentAccuracy:       metrics.IntentAccuracy,
		TaskCompletionRate:   metrics.TaskCompletionRate,
		ResponseLatency:      metrics.AvgResponseLatency,
		ContextSwitchLatency: metrics.ContextSwitchLatency,
	}

	mc.sampleLock.Lock()
	defer mc.sampleLock.Unlock()

	mc.samples = append(mc.samples, sample)

	// Keep only last 1000 samples
	if len(mc.samples) > 1000 {
		mc.samples = mc.samples[len(mc.samples)-1000:]
	}
}

// GetMetrics returns current metrics
func (mc *MetricsCollector) GetMetrics() *cognitive.CognitiveMetrics {
	mc.lock.RLock()
	defer mc.lock.RUnlock()

	// Return a copy
	metricsCopy := *mc.metrics
	return &metricsCopy
}

// GetSamples returns recent metric samples
func (mc *MetricsCollector) GetSamples(count int) []MetricSample {
	mc.sampleLock.RLock()
	defer mc.sampleLock.RUnlock()

	if count > len(mc.samples) {
		count = len(mc.samples)
	}

	startIdx := len(mc.samples) - count
	if startIdx < 0 {
		startIdx = 0
	}

	samples := make([]MetricSample, len(mc.samples[startIdx:]))
	copy(samples, mc.samples[startIdx:])

	return samples
}

// ValidateMetrics checks if metrics meet performance targets
func (mc *MetricsCollector) ValidateMetrics() *ValidationReport {
	mc.lock.RLock()
	defer mc.lock.RUnlock()

	report := &ValidationReport{
		Timestamp: time.Now(),
		Checks:    []ValidationCheck{},
	}

	// Check intent accuracy (target: >95%)
	report.Checks = append(report.Checks, ValidationCheck{
		Metric:   "Intent Accuracy",
		Current:  mc.metrics.IntentAccuracy,
		Target:   0.95,
		Pass:     mc.metrics.IntentAccuracy >= 0.95,
		Message:  formatCheckMessage("Intent Accuracy", mc.metrics.IntentAccuracy, 0.95),
	})

	// Check task completion rate (target: >90%)
	report.Checks = append(report.Checks, ValidationCheck{
		Metric:   "Task Completion Rate",
		Current:  mc.metrics.TaskCompletionRate,
		Target:   0.90,
		Pass:     mc.metrics.TaskCompletionRate >= 0.90,
		Message:  formatCheckMessage("Task Completion", mc.metrics.TaskCompletionRate, 0.90),
	})

	// Check response latency (target: <100ms)
	report.Checks = append(report.Checks, ValidationCheck{
		Metric:   "Response Latency",
		Current:  mc.metrics.AvgResponseLatency,
		Target:   100.0,
		Pass:     mc.metrics.AvgResponseLatency < 100.0,
		Message:  formatLatencyMessage(mc.metrics.AvgResponseLatency, 100.0),
	})

	// Check recommendation acceptance (target: >85%)
	report.Checks = append(report.Checks, ValidationCheck{
		Metric:   "Recommendation Acceptance",
		Current:  mc.metrics.RecommendationAcceptance,
		Target:   0.85,
		Pass:     mc.metrics.RecommendationAcceptance >= 0.85,
		Message:  formatCheckMessage("Recommendation Acceptance", mc.metrics.RecommendationAcceptance, 0.85),
	})

	// Check context switch latency (target: <10ms)
	report.Checks = append(report.Checks, ValidationCheck{
		Metric:   "Context Switch Latency",
		Current:  mc.metrics.ContextSwitchLatency,
		Target:   10.0,
		Pass:     mc.metrics.ContextSwitchLatency < 10.0,
		Message:  formatLatencyMessage(mc.metrics.ContextSwitchLatency, 10.0),
	})

	// Check user satisfaction (target: >4.5/5)
	report.Checks = append(report.Checks, ValidationCheck{
		Metric:   "User Satisfaction",
		Current:  mc.metrics.UserSatisfaction,
		Target:   4.5,
		Pass:     mc.metrics.UserSatisfaction >= 4.5,
		Message:  formatCheckMessage("User Satisfaction", mc.metrics.UserSatisfaction, 4.5),
	})

	// Calculate overall pass rate
	passCount := 0
	for _, check := range report.Checks {
		if check.Pass {
			passCount++
		}
	}
	report.PassRate = float64(passCount) / float64(len(report.Checks))
	report.AllPassed = report.PassRate == 1.0

	return report
}

// ValidationReport contains validation results
type ValidationReport struct {
	Timestamp  time.Time
	Checks     []ValidationCheck
	PassRate   float64
	AllPassed  bool
}

// ValidationCheck represents a single metric check
type ValidationCheck struct {
	Metric  string
	Current float64
	Target  float64
	Pass    bool
	Message string
}

// formatCheckMessage formats a validation check message
func formatCheckMessage(name string, current, target float64) string {
	percentage := current * 100
	targetPercentage := target * 100

	if current >= target {
		return formatMessage("%s: %.2f%% (target: %.2f%%) ✓", name, percentage, targetPercentage)
	}
	return formatMessage("%s: %.2f%% (target: %.2f%%) ✗", name, percentage, targetPercentage)
}

// formatLatencyMessage formats a latency check message
func formatLatencyMessage(current, target float64) string {
	if current < target {
		return formatMessage("Latency: %.2fms (target: <%.2fms) ✓", current, target)
	}
	return formatMessage("Latency: %.2fms (target: <%.2fms) ✗", current, target)
}

func formatMessage(format string, args ...interface{}) string {
	// Simple formatting wrapper
	return format
}

// GetUptime returns system uptime
func (mc *MetricsCollector) GetUptime() time.Duration {
	return time.Since(mc.startTime)
}

// Reset resets all metrics
func (mc *MetricsCollector) Reset() {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	mc.metrics = &cognitive.CognitiveMetrics{
		IntentAccuracy:           0.95,
		TaskCompletionRate:       0.90,
		UserSatisfaction:         4.5,
		ReasoningCorrectness:     0.90,
		AvgResponseLatency:       50.0,
		RecommendationAcceptance: 0.85,
		ContextSwitchLatency:     5.0,
	}
	mc.startTime = time.Now()
}

// ExportMetrics exports metrics for external analysis
func (mc *MetricsCollector) ExportMetrics() map[string]interface{} {
	mc.lock.RLock()
	defer mc.lock.RUnlock()

	return map[string]interface{}{
		"intent_accuracy":           mc.metrics.IntentAccuracy,
		"task_completion_rate":      mc.metrics.TaskCompletionRate,
		"user_satisfaction":         mc.metrics.UserSatisfaction,
		"reasoning_correctness":     mc.metrics.ReasoningCorrectness,
		"avg_response_latency_ms":   mc.metrics.AvgResponseLatency,
		"recommendation_acceptance": mc.metrics.RecommendationAcceptance,
		"context_switch_latency_ms": mc.metrics.ContextSwitchLatency,
		"total_conversations":       mc.metrics.TotalConversations,
		"total_intents":             mc.metrics.TotalIntents,
		"total_recommendations":     mc.metrics.TotalRecommendations,
		"uptime_seconds":            time.Since(mc.startTime).Seconds(),
	}
}
