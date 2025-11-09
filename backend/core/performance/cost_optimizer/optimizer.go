package cost_optimizer

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// Optimizer performs cost-performance optimization
type Optimizer struct {
	config CostOptimizerConfig
	mu     sync.RWMutex
	vms    map[string]*VMCostData
}

// CostOptimizerConfig defines cost optimization settings
type CostOptimizerConfig struct {
	MultiObjectiveOptimize bool
	ParetoFrontierAnalysis bool
	SpotInstanceRecommend  bool
	ReservedInstancePlan   bool
	SavingsPlansAnalyze    bool
	CostPredictionEnabled  bool
	SLAConstraints         map[string]float64
	OptimizationInterval   time.Duration
}

// VMCostData stores VM cost and performance data
type VMCostData struct {
	VMID            string
	CurrentCost     float64 // Hourly
	CurrentPerf     PerformanceMetrics
	InstanceType    string
	Region          string
	PricingModel    string // "on-demand", "spot", "reserved"
	Utilization     ResourceUtilization
	SLARequirements SLARequirements
	History         []CostPerfSnapshot
}

// PerformanceMetrics stores performance data
type PerformanceMetrics struct {
	Throughput      float64 // ops/sec
	Latency         float64 // ms
	Availability    float64 // percentage
	ErrorRate       float64
	ResponseTimeP95 float64
}

// ResourceUtilization stores resource usage
type ResourceUtilization struct {
	CPUAvg     float64
	CPUP95     float64
	MemoryAvg  float64
	MemoryP95  float64
	NetworkAvg float64
	IOPSAvg    float64
}

// SLARequirements defines SLA constraints
type SLARequirements struct {
	MaxLatency      float64 // ms
	MinAvailability float64 // percentage
	MaxErrorRate    float64
	MinThroughput   float64
}

// CostPerfSnapshot stores point-in-time data
type CostPerfSnapshot struct {
	Timestamp   time.Time
	Cost        float64
	Performance PerformanceMetrics
	Utilization ResourceUtilization
}

// Recommendation represents cost optimization recommendation
type Recommendation struct {
	VMID                string
	RecommendationType  string // "spot", "reserved", "rightsize", "terminate"
	CurrentCost         float64
	ProjectedCost       float64
	EstimatedSavings    float64
	SavingsPercent      float64
	PerformanceImpact   string
	RiskLevel           string // "low", "medium", "high"
	ImplementationSteps []string
	Confidence          float64
	ParetoOptimal       bool
}

// ParetoPoint represents point on Pareto frontier
type ParetoPoint struct {
	Cost        float64
	Performance float64
	Config      string
	IsDominated bool
}

// NewOptimizer creates cost optimizer
func NewOptimizer(config CostOptimizerConfig) *Optimizer {
	if config.OptimizationInterval == 0 {
		config.OptimizationInterval = 1 * time.Hour
	}

	return &Optimizer{
		config: config,
		vms:    make(map[string]*VMCostData),
	}
}

// RecordVMData records VM cost and performance data
func (o *Optimizer) RecordVMData(vmID string, cost float64, perf PerformanceMetrics, util ResourceUtilization) {
	o.mu.Lock()
	defer o.mu.Unlock()

	vm, exists := o.vms[vmID]
	if !exists {
		vm = &VMCostData{
			VMID:        vmID,
			History:     make([]CostPerfSnapshot, 0),
		}
		o.vms[vmID] = vm
	}

	vm.CurrentCost = cost
	vm.CurrentPerf = perf
	vm.Utilization = util

	// Add to history
	vm.History = append(vm.History, CostPerfSnapshot{
		Timestamp:   time.Now(),
		Cost:        cost,
		Performance: perf,
		Utilization: util,
	})

	// Keep last 7 days
	cutoff := time.Now().Add(-7 * 24 * time.Hour)
	for i, snap := range vm.History {
		if snap.Timestamp.After(cutoff) {
			vm.History = vm.History[i:]
			break
		}
	}
}

// Optimize generates cost optimization recommendations
func (o *Optimizer) Optimize(ctx context.Context) ([]*Recommendation, error) {
	o.mu.RLock()
	defer o.mu.RUnlock()

	var recommendations []*Recommendation

	for vmID, vm := range o.vms {
		// Generate recommendations for each VM
		recs := o.analyzeVM(vmID, vm)
		recommendations = append(recommendations, recs...)
	}

	// Pareto frontier analysis
	if o.config.ParetoFrontierAnalysis {
		o.markParetoOptimal(recommendations)
	}

	// Sort by savings
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].EstimatedSavings > recommendations[j].EstimatedSavings
	})

	return recommendations, nil
}

// analyzeVM analyzes single VM
func (o *Optimizer) analyzeVM(vmID string, vm *VMCostData) []*Recommendation {
	var recommendations []*Recommendation

	// Spot instance recommendation
	if o.config.SpotInstanceRecommend && vm.PricingModel == "on-demand" {
		spotRec := o.analyzeSpotInstance(vmID, vm)
		if spotRec != nil {
			recommendations = append(recommendations, spotRec)
		}
	}

	// Reserved instance recommendation
	if o.config.ReservedInstancePlan && vm.PricingModel == "on-demand" {
		reservedRec := o.analyzeReservedInstance(vmID, vm)
		if reservedRec != nil {
			recommendations = append(recommendations, reservedRec)
		}
	}

	// Right-sizing recommendation
	rightsizeRec := o.analyzeRightSizing(vmID, vm)
	if rightsizeRec != nil {
		recommendations = append(recommendations, rightsizeRec)
	}

	// Termination recommendation (idle VMs)
	terminateRec := o.analyzeTermination(vmID, vm)
	if terminateRec != nil {
		recommendations = append(recommendations, terminateRec)
	}

	return recommendations
}

// analyzeSpotInstance analyzes spot instance savings
func (o *Optimizer) analyzeSpotInstance(vmID string, vm *VMCostData) *Recommendation {
	// Spot instances typically 70% cheaper
	spotDiscount := 0.70
	projectedCost := vm.CurrentCost * (1 - spotDiscount)
	savings := vm.CurrentCost - projectedCost

	// Check if workload can tolerate interruptions
	if vm.SLARequirements.MinAvailability > 0.95 {
		// Too high availability requirement for spot
		return nil
	}

	return &Recommendation{
		VMID:               vmID,
		RecommendationType: "spot",
		CurrentCost:        vm.CurrentCost,
		ProjectedCost:      projectedCost,
		EstimatedSavings:   savings,
		SavingsPercent:     spotDiscount * 100,
		PerformanceImpact:  "minimal",
		RiskLevel:          "medium",
		ImplementationSteps: []string{
			"Implement spot instance request",
			"Configure instance fallback strategy",
			"Test workload on spot instances",
		},
		Confidence: 0.8,
	}
}

// analyzeReservedInstance analyzes reserved instance savings
func (o *Optimizer) analyzeReservedInstance(vmID string, vm *VMCostData) *Recommendation {
	// Reserved instances typically 40% cheaper for 1-year commitment
	reservedDiscount := 0.40
	projectedCost := vm.CurrentCost * (1 - reservedDiscount)
	monthlySavings := (vm.CurrentCost - projectedCost) * 730 // Monthly hours

	// Check if VM has stable usage pattern
	if len(vm.History) < 168 { // 1 week of hourly data
		return nil
	}

	// Calculate usage stability
	variance := o.calculateCostVariance(vm.History)
	if variance > 0.2 {
		// Too variable for reserved instance
		return nil
	}

	return &Recommendation{
		VMID:               vmID,
		RecommendationType: "reserved",
		CurrentCost:        vm.CurrentCost,
		ProjectedCost:      projectedCost,
		EstimatedSavings:   monthlySavings,
		SavingsPercent:     reservedDiscount * 100,
		PerformanceImpact:  "none",
		RiskLevel:          "low",
		ImplementationSteps: []string{
			"Purchase 1-year reserved instance",
			"Monitor usage to ensure full utilization",
		},
		Confidence: 0.9,
	}
}

// analyzeRightSizing analyzes right-sizing opportunity
func (o *Optimizer) analyzeRightSizing(vmID string, vm *VMCostData) *Recommendation {
	// Check if underutilized
	if vm.Utilization.CPUP95 < 0.30 && vm.Utilization.MemoryP95 < 0.40 {
		// Severely underutilized - recommend downsizing
		downsizeSavings := vm.CurrentCost * 0.5 // 50% smaller instance
		projectedCost := vm.CurrentCost * 0.5

		return &Recommendation{
			VMID:               vmID,
			RecommendationType: "rightsize",
			CurrentCost:        vm.CurrentCost,
			ProjectedCost:      projectedCost,
			EstimatedSavings:   downsizeSavings,
			SavingsPercent:     50,
			PerformanceImpact:  "minimal",
			RiskLevel:          "low",
			ImplementationSteps: []string{
				"Resize to smaller instance type",
				"Monitor performance after resize",
				"Rollback if degradation detected",
			},
			Confidence: 0.85,
		}
	}

	return nil
}

// analyzeTermination analyzes termination opportunity
func (o *Optimizer) analyzeTermination(vmID string, vm *VMCostData) *Recommendation {
	// Check if idle (very low utilization)
	if vm.Utilization.CPUP95 < 0.05 && vm.Utilization.MemoryAvg < 0.10 {
		monthlySavings := vm.CurrentCost * 730

		return &Recommendation{
			VMID:               vmID,
			RecommendationType: "terminate",
			CurrentCost:        vm.CurrentCost,
			ProjectedCost:      0,
			EstimatedSavings:   monthlySavings,
			SavingsPercent:     100,
			PerformanceImpact:  "complete",
			RiskLevel:          "high",
			ImplementationSteps: []string{
				"Verify VM is not in use",
				"Backup any necessary data",
				"Terminate instance",
			},
			Confidence: 0.7,
		}
	}

	return nil
}

// markParetoOptimal marks Pareto optimal recommendations
func (o *Optimizer) markParetoOptimal(recommendations []*Recommendation) {
	// Build Pareto frontier
	points := make([]ParetoPoint, len(recommendations))
	for i, rec := range recommendations {
		// Performance score (higher is better)
		perfScore := 100.0
		if rec.PerformanceImpact == "minimal" {
			perfScore = 95.0
		} else if rec.PerformanceImpact == "moderate" {
			perfScore = 80.0
		} else if rec.PerformanceImpact == "significant" {
			perfScore = 60.0
		}

		points[i] = ParetoPoint{
			Cost:        rec.ProjectedCost,
			Performance: perfScore,
			Config:      rec.RecommendationType,
		}
	}

	// Identify dominated points
	for i := range points {
		dominated := false
		for j := range points {
			if i == j {
				continue
			}
			// Point j dominates point i if it has lower cost and higher performance
			if points[j].Cost <= points[i].Cost && points[j].Performance >= points[i].Performance {
				if points[j].Cost < points[i].Cost || points[j].Performance > points[i].Performance {
					dominated = true
					break
				}
			}
		}
		points[i].IsDominated = dominated
		recommendations[i].ParetoOptimal = !dominated
	}
}

// PredictCost predicts future cost
func (o *Optimizer) PredictCost(vmID string, horizon time.Duration) (float64, error) {
	if !o.config.CostPredictionEnabled {
		return 0, fmt.Errorf("cost prediction not enabled")
	}

	o.mu.RLock()
	vm, exists := o.vms[vmID]
	o.mu.RUnlock()

	if !exists {
		return 0, fmt.Errorf("VM %s not found", vmID)
	}

	if len(vm.History) < 24 {
		// Insufficient data
		return vm.CurrentCost * horizon.Hours(), nil
	}

	// Simple linear regression on cost trend
	trend := o.calculateCostTrend(vm.History)

	// Predict future cost
	currentCost := vm.CurrentCost
	futureHours := horizon.Hours()
	predictedHourlyCost := currentCost + (trend * futureHours)

	totalCost := predictedHourlyCost * futureHours

	return totalCost, nil
}

// calculateCostVariance calculates variance in cost
func (o *Optimizer) calculateCostVariance(history []CostPerfSnapshot) float64 {
	if len(history) == 0 {
		return 0
	}

	// Calculate mean
	mean := 0.0
	for _, snap := range history {
		mean += snap.Cost
	}
	mean /= float64(len(history))

	// Calculate variance
	variance := 0.0
	for _, snap := range history {
		diff := snap.Cost - mean
		variance += diff * diff
	}
	variance /= float64(len(history))

	return variance / (mean * mean) // Coefficient of variation
}

// calculateCostTrend calculates cost trend (per hour)
func (o *Optimizer) calculateCostTrend(history []CostPerfSnapshot) float64 {
	if len(history) < 2 {
		return 0
	}

	// Simple linear regression
	n := float64(len(history))
	var sumX, sumY, sumXY, sumX2 float64

	for i, snap := range history {
		x := float64(i)
		y := snap.Cost
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	return slope
}

// CalculateROI calculates ROI for optimization
func (o *Optimizer) CalculateROI(rec *Recommendation, implementationCost float64) float64 {
	monthlySavings := rec.EstimatedSavings
	if monthlySavings <= 0 {
		return 0
	}

	// ROI = (Gain - Cost) / Cost
	// Assume 1-year time horizon
	yearSavings := monthlySavings * 12
	roi := (yearSavings - implementationCost) / implementationCost * 100

	return roi
}
