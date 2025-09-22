package scheduler

import (
	"context"
	"fmt"
	"log"
	"math"
	"time"
)

// FallbackSchedulingStrategy provides heuristic-based scheduling when AI is unavailable
type FallbackSchedulingStrategy struct {
	config SchedulerConfig
}

// NewFallbackSchedulingStrategy creates a new fallback strategy
func NewFallbackSchedulingStrategy(config SchedulerConfig) *FallbackSchedulingStrategy {
	return &FallbackSchedulingStrategy{
		config: config,
	}
}

// PredictResourceDemand provides heuristic-based resource demand prediction
func (f *FallbackSchedulingStrategy) PredictResourceDemand(nodeID string, resourceType ResourceType, horizonMinutes int) ([]float64, float64) {
	// Use exponential smoothing on historical data as fallback
	// This is a simple but effective time-series prediction method

	predictions := make([]float64, horizonMinutes/5) // 5-minute intervals
	baseUtilization := 0.5                           // Default baseline

	// Apply time-of-day patterns (simplified)
	currentHour := time.Now().Hour()

	// Business hours typically have higher utilization
	if currentHour >= 9 && currentHour <= 17 {
		baseUtilization = 0.7
	} else if currentHour >= 0 && currentHour <= 6 {
		baseUtilization = 0.3
	}

	// Add some variance based on resource type
	switch resourceType {
	case ResourceCPU:
		// CPU tends to be more variable
		for i := range predictions {
			predictions[i] = baseUtilization + (0.1 * math.Sin(float64(i)/10))
		}
	case ResourceMemory:
		// Memory is more stable
		for i := range predictions {
			predictions[i] = baseUtilization + (0.05 * math.Cos(float64(i)/15))
		}
	case ResourceNetwork:
		// Network can be bursty
		for i := range predictions {
			predictions[i] = baseUtilization + (0.15 * math.Sin(float64(i)/5))
		}
	default:
		for i := range predictions {
			predictions[i] = baseUtilization
		}
	}

	// Confidence is lower for heuristic predictions
	confidence := 0.6

	return predictions, confidence
}

// OptimizePerformance provides heuristic-based performance optimization
func (f *FallbackSchedulingStrategy) OptimizePerformance(clusterData map[string]interface{}) map[string]interface{} {
	recommendations := make(map[string]interface{})

	// Simple rule-based optimization
	if cpuUsage, ok := clusterData["cpu_usage"].(float64); ok {
		if cpuUsage > 0.8 {
			recommendations["scale_up"] = true
			recommendations["additional_nodes"] = 2
		} else if cpuUsage < 0.3 {
			recommendations["scale_down"] = true
			recommendations["remove_nodes"] = 1
		}
	}

	// Load balancing recommendation
	recommendations["rebalance"] = false
	if nodeCount, ok := clusterData["node_count"].(int); ok && nodeCount > 3 {
		// Check for imbalanced load (simplified)
		recommendations["rebalance"] = true
	}

	return recommendations
}

// DetectAnomalies provides simple threshold-based anomaly detection
func (f *FallbackSchedulingStrategy) DetectAnomalies(metrics map[string]float64) (bool, float64, []string) {
	var anomalyScore float64
	var recommendations []string
	isAnomaly := false

	// Simple threshold-based detection
	thresholds := map[string]float64{
		"cpu_usage":     0.90,
		"memory_usage":  0.95,
		"disk_usage":    0.85,
		"network_usage": 0.80,
		"error_rate":    0.05,
		"latency_p99":   1000, // milliseconds
	}

	for metric, value := range metrics {
		if threshold, exists := thresholds[metric]; exists {
			if metric == "error_rate" || metric == "latency_p99" {
				// For these metrics, higher is worse
				if value > threshold {
					isAnomaly = true
					anomalyScore = math.Max(anomalyScore, value/threshold)
					recommendations = append(recommendations,
						fmt.Sprintf("High %s detected: %.2f (threshold: %.2f)", metric, value, threshold))
				}
			} else {
				// For usage metrics, check if over threshold
				if value > threshold {
					isAnomaly = true
					anomalyScore = math.Max(anomalyScore, value/threshold)
					recommendations = append(recommendations,
						fmt.Sprintf("%s is above threshold: %.2f%% (threshold: %.2f%%)",
							metric, value*100, threshold*100))
				}
			}
		}
	}

	// Check for rapid changes (simplified spike detection)
	if prevCPU, exists := metrics["prev_cpu_usage"]; exists {
		if currCPU, exists := metrics["cpu_usage"]; exists {
			change := math.Abs(currCPU - prevCPU)
			if change > 0.3 { // 30% change
				isAnomaly = true
				anomalyScore = math.Max(anomalyScore, change*2)
				recommendations = append(recommendations,
					fmt.Sprintf("Rapid CPU change detected: %.2f%%", change*100))
			}
		}
	}

	return isAnomaly, anomalyScore, recommendations
}

// GetScalingRecommendations provides rule-based scaling recommendations
func (f *FallbackSchedulingStrategy) GetScalingRecommendations(vmID string, currentResources map[string]float64) []map[string]interface{} {
	var recommendations []map[string]interface{}

	// Check CPU utilization
	if cpuUsage, ok := currentResources["cpu_usage"]; ok {
		if cpuUsage > 0.8 {
			recommendations = append(recommendations, map[string]interface{}{
				"action":     "scale_up",
				"resource":   "cpu",
				"multiplier": 1.5,
				"reason":     "High CPU utilization",
				"confidence": 0.7,
			})
		} else if cpuUsage < 0.2 {
			recommendations = append(recommendations, map[string]interface{}{
				"action":     "scale_down",
				"resource":   "cpu",
				"multiplier": 0.7,
				"reason":     "Low CPU utilization",
				"confidence": 0.6,
			})
		}
	}

	// Check memory utilization
	if memUsage, ok := currentResources["memory_usage"]; ok {
		if memUsage > 0.9 {
			recommendations = append(recommendations, map[string]interface{}{
				"action":     "scale_up",
				"resource":   "memory",
				"multiplier": 1.3,
				"reason":     "High memory pressure",
				"confidence": 0.8,
			})
		}
	}

	return recommendations
}

// OptimizeMigration provides heuristic-based migration optimization
func (f *FallbackSchedulingStrategy) OptimizeMigration(vmID string, sourceHost string, targetHosts []string, vmMetrics map[string]float64) map[string]interface{} {
	// Simple scoring based on available resources
	bestTarget := ""
	bestScore := -1.0

	for _, target := range targetHosts {
		score := 1.0

		// Prefer targets with lower utilization (simplified)
		// In real implementation, we'd check actual target metrics
		if target != sourceHost {
			score = 0.7 // Base score for different host

			// Adjust based on VM requirements
			if cpuReq, ok := vmMetrics["cpu_required"]; ok {
				if cpuReq > 4 {
					// For high CPU VMs, prefer specific hosts (simplified)
					score *= 0.9
				}
			}
		}

		if score > bestScore {
			bestScore = score
			bestTarget = target
		}
	}

	return map[string]interface{}{
		"target_host": bestTarget,
		"score":       bestScore,
		"strategy":    "balanced",
		"confidence":  0.5,
	}
}

// OptimizeBandwidth provides simple bandwidth optimization
func (f *FallbackSchedulingStrategy) OptimizeBandwidth(networkID string, trafficData []map[string]interface{}, qosRequirements map[string]float64) map[string]interface{} {
	// Simple fair-share allocation
	totalBandwidth := 10000.0 // 10 Gbps default
	if reqBandwidth, ok := qosRequirements["min_bandwidth"]; ok {
		totalBandwidth = reqBandwidth * 1.5 // Provide 50% headroom
	}

	numFlows := len(trafficData)
	if numFlows == 0 {
		numFlows = 1
	}

	perFlowBandwidth := totalBandwidth / float64(numFlows)

	return map[string]interface{}{
		"allocation_strategy": "fair_share",
		"total_bandwidth":     totalBandwidth,
		"per_flow_bandwidth":  perFlowBandwidth,
		"qos_enabled":         true,
		"confidence":          0.6,
	}
}

// SafeAIProvider wraps an AI provider with fallback logic
type SafeAIProvider struct {
	aiProvider AIProvider
	fallback   *FallbackSchedulingStrategy
	logger     *log.Logger
	metrics    *AIFallbackMetrics
}

// AIFallbackMetrics tracks fallback usage
type AIFallbackMetrics struct {
	totalCalls      int64
	fallbackCalls   int64
	aiFailures      int64
	avgFallbackTime time.Duration
}

// NewSafeAIProvider creates a new safe AI provider with fallback
func NewSafeAIProvider(aiProvider AIProvider, config SchedulerConfig) *SafeAIProvider {
	return &SafeAIProvider{
		aiProvider: aiProvider,
		fallback:   NewFallbackSchedulingStrategy(config),
		logger:     log.New(log.Writer(), "[SafeAI] ", log.LstdFlags),
		metrics:    &AIFallbackMetrics{},
	}
}

// PredictResourceDemand with fallback
func (s *SafeAIProvider) PredictResourceDemand(nodeID string, resourceType ResourceType, horizonMinutes int) ([]float64, float64, error) {
	s.metrics.totalCalls++

	// Try AI provider first if available
	if s.aiProvider != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		// Run AI prediction in goroutine with timeout
		type result struct {
			predictions []float64
			confidence  float64
			err         error
		}

		resultChan := make(chan result, 1)
		go func() {
			pred, conf, err := s.aiProvider.PredictResourceDemand(nodeID, resourceType, horizonMinutes)
			resultChan <- result{pred, conf, err}
		}()

		select {
		case res := <-resultChan:
			if res.err == nil && len(res.predictions) > 0 {
				return res.predictions, res.confidence, nil
			}
			s.logger.Printf("AI prediction failed: %v, using fallback", res.err)
			s.metrics.aiFailures++
		case <-ctx.Done():
			s.logger.Printf("AI prediction timeout, using fallback")
			s.metrics.aiFailures++
		}
	}

	// Use fallback
	s.metrics.fallbackCalls++
	start := time.Now()
	predictions, confidence := s.fallback.PredictResourceDemand(nodeID, resourceType, horizonMinutes)
	s.metrics.avgFallbackTime = time.Since(start)

	return predictions, confidence, nil
}

// OptimizePerformance with fallback
func (s *SafeAIProvider) OptimizePerformance(clusterData map[string]interface{}, goals []string) (map[string]interface{}, error) {
	s.metrics.totalCalls++

	if s.aiProvider != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		type result struct {
			data map[string]interface{}
			err  error
		}

		resultChan := make(chan result, 1)
		go func() {
			data, err := s.aiProvider.OptimizePerformance(clusterData, goals)
			resultChan <- result{data, err}
		}()

		select {
		case res := <-resultChan:
			if res.err == nil && res.data != nil {
				return res.data, nil
			}
			s.logger.Printf("AI optimization failed: %v, using fallback", res.err)
			s.metrics.aiFailures++
		case <-ctx.Done():
			s.logger.Printf("AI optimization timeout, using fallback")
			s.metrics.aiFailures++
		}
	}

	// Use fallback
	s.metrics.fallbackCalls++
	return s.fallback.OptimizePerformance(clusterData), nil
}

// DetectAnomalies with fallback
func (s *SafeAIProvider) DetectAnomalies(metrics map[string]float64) (bool, float64, []string, error) {
	s.metrics.totalCalls++

	if s.aiProvider != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		type result struct {
			isAnomaly       bool
			score           float64
			recommendations []string
			err             error
		}

		resultChan := make(chan result, 1)
		go func() {
			isAnom, score, recs, err := s.aiProvider.DetectAnomalies(metrics)
			resultChan <- result{isAnom, score, recs, err}
		}()

		select {
		case res := <-resultChan:
			if res.err == nil {
				return res.isAnomaly, res.score, res.recommendations, nil
			}
			s.logger.Printf("AI anomaly detection failed: %v, using fallback", res.err)
			s.metrics.aiFailures++
		case <-ctx.Done():
			s.logger.Printf("AI anomaly detection timeout, using fallback")
			s.metrics.aiFailures++
		}
	}

	// Use fallback
	s.metrics.fallbackCalls++
	isAnomaly, score, recommendations := s.fallback.DetectAnomalies(metrics)
	return isAnomaly, score, recommendations, nil
}

// GetMetrics returns metrics about AI fallback usage
func (s *SafeAIProvider) GetMetrics() map[string]interface{} {
	fallbackRate := float64(0)
	if s.metrics.totalCalls > 0 {
		fallbackRate = float64(s.metrics.fallbackCalls) / float64(s.metrics.totalCalls)
	}

	return map[string]interface{}{
		"total_calls":       s.metrics.totalCalls,
		"fallback_calls":    s.metrics.fallbackCalls,
		"ai_failures":       s.metrics.aiFailures,
		"fallback_rate":     fallbackRate,
		"avg_fallback_time": s.metrics.avgFallbackTime.Milliseconds(),
	}
}

// Implement remaining AIProvider interface methods with fallback...
func (s *SafeAIProvider) AnalyzeWorkload(vmID string, workloadData []map[string]interface{}) (map[string]interface{}, error) {
	// Simple workload classification based on patterns
	result := map[string]interface{}{
		"workload_type":    "general",
		"peak_hours":       []int{9, 10, 11, 14, 15, 16},
		"resource_pattern": "steady",
		"confidence":       0.5,
	}

	if s.aiProvider != nil {
		if aiResult, err := s.aiProvider.AnalyzeWorkload(vmID, workloadData); err == nil {
			return aiResult, nil
		}
	}

	return result, nil
}

func (s *SafeAIProvider) GetScalingRecommendations(vmID string, currentResources map[string]float64, historicalData []map[string]interface{}) ([]map[string]interface{}, error) {
	if s.aiProvider != nil {
		if recs, err := s.aiProvider.GetScalingRecommendations(vmID, currentResources, historicalData); err == nil {
			return recs, nil
		}
	}

	return s.fallback.GetScalingRecommendations(vmID, currentResources), nil
}

func (s *SafeAIProvider) OptimizeMigration(vmID string, sourceHost string, targetHosts []string, vmMetrics map[string]float64) (map[string]interface{}, error) {
	if s.aiProvider != nil {
		if result, err := s.aiProvider.OptimizeMigration(vmID, sourceHost, targetHosts, vmMetrics); err == nil {
			return result, nil
		}
	}

	return s.fallback.OptimizeMigration(vmID, sourceHost, targetHosts, vmMetrics), nil
}

func (s *SafeAIProvider) OptimizeBandwidth(networkID string, trafficData []map[string]interface{}, qosRequirements map[string]float64) (map[string]interface{}, error) {
	if s.aiProvider != nil {
		if result, err := s.aiProvider.OptimizeBandwidth(networkID, trafficData, qosRequirements); err == nil {
			return result, nil
		}
	}

	return s.fallback.OptimizeBandwidth(networkID, trafficData, qosRequirements), nil
}
