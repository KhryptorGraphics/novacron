package rightsizing

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// Engine performs automatic VM right-sizing
type Engine struct {
	config      RightSizingConfig
	mu          sync.RWMutex
	observations map[string]*VMObservation
	recommendations map[string]*Recommendation
}

// RightSizingConfig defines right-sizing parameters
type RightSizingConfig struct {
	CPUTargetMin        float64 // 0.60
	CPUTargetMax        float64 // 0.80
	MemoryTargetMin     float64 // 0.70
	MemoryTargetMax     float64 // 0.85
	ObservationPeriod   time.Duration
	ConfidenceThreshold float64 // 0.90
	CostSavingsMin      float64 // 0.10
}

// VMObservation stores resource usage observations
type VMObservation struct {
	VMID              string
	CPUUtilization    []float64
	MemoryUtilization []float64
	IOPSUtilization   []float64
	NetworkUtilization []float64
	Timestamps        []time.Time
	CurrentSize       VMSize
}

// VMSize represents VM configuration
type VMSize struct {
	Name         string
	VCPUs        int
	MemoryGB     float64
	StorageGB    int
	NetworkMbps  int
	HourlyCost   float64
}

// Recommendation represents a right-sizing recommendation
type Recommendation struct {
	VMID              string
	CurrentSize       VMSize
	RecommendedSize   VMSize
	Action            string // "upsize", "downsize", "maintain"
	Confidence        float64
	EstimatedSavings  float64
	PerformanceImpact string
	Rationale         string
	Metrics           RightSizingMetrics
}

// RightSizingMetrics contains analysis metrics
type RightSizingMetrics struct {
	AvgCPU    float64
	P95CPU    float64
	P99CPU    float64
	AvgMemory float64
	P95Memory float64
	P99Memory float64
	CPUTrend  string // "increasing", "decreasing", "stable"
	MemoryTrend string
}

// NewEngine creates right-sizing engine
func NewEngine(config RightSizingConfig) *Engine {
	return &Engine{
		config:          config,
		observations:    make(map[string]*VMObservation),
		recommendations: make(map[string]*Recommendation),
	}
}

// ObserveVM records VM resource usage
func (e *Engine) ObserveVM(vmID string, cpu, memory, iops, network float64, size VMSize) {
	e.mu.Lock()
	defer e.mu.Unlock()

	obs, exists := e.observations[vmID]
	if !exists {
		obs = &VMObservation{
			VMID:        vmID,
			CurrentSize: size,
		}
		e.observations[vmID] = obs
	}

	obs.CPUUtilization = append(obs.CPUUtilization, cpu)
	obs.MemoryUtilization = append(obs.MemoryUtilization, memory)
	obs.IOPSUtilization = append(obs.IOPSUtilization, iops)
	obs.NetworkUtilization = append(obs.NetworkUtilization, network)
	obs.Timestamps = append(obs.Timestamps, time.Now())

	// Keep only recent observations
	cutoff := time.Now().Add(-e.config.ObservationPeriod)
	e.pruneObservations(obs, cutoff)
}

// pruneObservations removes old observations
func (e *Engine) pruneObservations(obs *VMObservation, cutoff time.Time) {
	validIdx := 0
	for i, ts := range obs.Timestamps {
		if ts.After(cutoff) {
			validIdx = i
			break
		}
	}

	if validIdx > 0 {
		obs.CPUUtilization = obs.CPUUtilization[validIdx:]
		obs.MemoryUtilization = obs.MemoryUtilization[validIdx:]
		obs.IOPSUtilization = obs.IOPSUtilization[validIdx:]
		obs.NetworkUtilization = obs.NetworkUtilization[validIdx:]
		obs.Timestamps = obs.Timestamps[validIdx:]
	}
}

// AnalyzeAndRecommend analyzes VMs and generates recommendations
func (e *Engine) AnalyzeAndRecommend(ctx context.Context) ([]*Recommendation, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	var recommendations []*Recommendation

	for vmID, obs := range e.observations {
		if len(obs.CPUUtilization) < 100 { // Need minimum samples
			continue
		}

		rec := e.analyzeVM(obs)
		if rec != nil && rec.Confidence >= e.config.ConfidenceThreshold {
			recommendations = append(recommendations, rec)
			e.recommendations[vmID] = rec
		}
	}

	return recommendations, nil
}

// analyzeVM analyzes single VM
func (e *Engine) analyzeVM(obs *VMObservation) *Recommendation {
	metrics := e.calculateMetrics(obs)

	// Determine if resize needed
	action := "maintain"
	var recommendedSize VMSize
	var rationale string

	// CPU analysis
	cpuUnderutilized := metrics.P95CPU < e.config.CPUTargetMin
	cpuOverutilized := metrics.P95CPU > e.config.CPUTargetMax

	// Memory analysis
	memUnderutilized := metrics.P95Memory < e.config.MemoryTargetMin
	memOverutilized := metrics.P95Memory > e.config.MemoryTargetMax

	if cpuOverutilized || memOverutilized {
		action = "upsize"
		recommendedSize = e.findLargerSize(obs.CurrentSize)
		rationale = fmt.Sprintf("CPU P95: %.1f%%, Memory P95: %.1f%% - exceeds target range",
			metrics.P95CPU*100, metrics.P95Memory*100)
	} else if cpuUnderutilized && memUnderutilized {
		action = "downsize"
		recommendedSize = e.findSmallerSize(obs.CurrentSize)
		rationale = fmt.Sprintf("CPU P95: %.1f%%, Memory P95: %.1f%% - below target range",
			metrics.P95CPU*100, metrics.P95Memory*100)
	} else {
		recommendedSize = obs.CurrentSize
		rationale = "Resource utilization within target range"
	}

	// Calculate savings and confidence
	savings := (obs.CurrentSize.HourlyCost - recommendedSize.HourlyCost) * 730 // Monthly
	confidence := e.calculateConfidence(obs, metrics)

	// Don't recommend if savings too small
	if action == "downsize" && savings < obs.CurrentSize.HourlyCost*730*e.config.CostSavingsMin {
		action = "maintain"
		recommendedSize = obs.CurrentSize
		rationale = "Potential savings below threshold"
	}

	performanceImpact := "none"
	if action == "downsize" {
		performanceImpact = "minimal"
	} else if action == "upsize" {
		performanceImpact = "improvement"
	}

	return &Recommendation{
		VMID:              obs.VMID,
		CurrentSize:       obs.CurrentSize,
		RecommendedSize:   recommendedSize,
		Action:            action,
		Confidence:        confidence,
		EstimatedSavings:  savings,
		PerformanceImpact: performanceImpact,
		Rationale:         rationale,
		Metrics:           metrics,
	}
}

// calculateMetrics calculates resource metrics
func (e *Engine) calculateMetrics(obs *VMObservation) RightSizingMetrics {
	return RightSizingMetrics{
		AvgCPU:      average(obs.CPUUtilization),
		P95CPU:      percentile(obs.CPUUtilization, 0.95),
		P99CPU:      percentile(obs.CPUUtilization, 0.99),
		AvgMemory:   average(obs.MemoryUtilization),
		P95Memory:   percentile(obs.MemoryUtilization, 0.95),
		P99Memory:   percentile(obs.MemoryUtilization, 0.99),
		CPUTrend:    e.detectTrend(obs.CPUUtilization),
		MemoryTrend: e.detectTrend(obs.MemoryUtilization),
	}
}

// calculateConfidence calculates recommendation confidence
func (e *Engine) calculateConfidence(obs *VMObservation, metrics RightSizingMetrics) float64 {
	confidence := 1.0

	// Reduce confidence if high variance
	cpuVariance := variance(obs.CPUUtilization)
	memVariance := variance(obs.MemoryUtilization)

	if cpuVariance > 0.1 {
		confidence *= 0.9
	}
	if memVariance > 0.1 {
		confidence *= 0.9
	}

	// Reduce confidence if trending
	if metrics.CPUTrend != "stable" {
		confidence *= 0.85
	}
	if metrics.MemoryTrend != "stable" {
		confidence *= 0.85
	}

	// Reduce confidence if insufficient data
	if len(obs.CPUUtilization) < 500 {
		confidence *= 0.8
	}

	return confidence
}

// detectTrend detects resource usage trend
func (e *Engine) detectTrend(values []float64) string {
	if len(values) < 10 {
		return "unknown"
	}

	// Simple linear regression
	n := float64(len(values))
	var sumX, sumY, sumXY, sumX2 float64

	for i, y := range values {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)

	if slope > 0.001 {
		return "increasing"
	} else if slope < -0.001 {
		return "decreasing"
	}
	return "stable"
}

// findSmallerSize finds next smaller VM size
func (e *Engine) findSmallerSize(current VMSize) VMSize {
	// Simplified - in production, use actual VM catalog
	return VMSize{
		Name:        "smaller-" + current.Name,
		VCPUs:       max(1, current.VCPUs-2),
		MemoryGB:    max(1, current.MemoryGB-4),
		StorageGB:   current.StorageGB,
		NetworkMbps: current.NetworkMbps,
		HourlyCost:  current.HourlyCost * 0.7,
	}
}

// findLargerSize finds next larger VM size
func (e *Engine) findLargerSize(current VMSize) VMSize {
	return VMSize{
		Name:        "larger-" + current.Name,
		VCPUs:       current.VCPUs + 2,
		MemoryGB:    current.MemoryGB + 4,
		StorageGB:   current.StorageGB,
		NetworkMbps: current.NetworkMbps,
		HourlyCost:  current.HourlyCost * 1.4,
	}
}

// Helper functions
func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	// Simple sort
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[i] > sorted[j] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}
	idx := int(float64(len(sorted)-1) * p)
	return sorted[idx]
}

func variance(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	avg := average(values)
	sumSquares := 0.0
	for _, v := range values {
		diff := v - avg
		sumSquares += diff * diff
	}
	return sumSquares / float64(len(values))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
