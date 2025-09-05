package analytics

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/montanaflynn/stats"
	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

// PerformanceMonitor tracks and analyzes system performance
type PerformanceMonitor struct {
	mu sync.RWMutex
	
	// Metrics tracking
	cacheMetrics    *CacheMetricsTracker
	scalingMetrics  *ScalingMetricsTracker
	resourceMetrics *ResourceMetricsTracker
	costMetrics     *CostMetricsTracker
	
	// Pattern recognition
	anomalyDetector *AnomalyDetector
	patternAnalyzer *PatternAnalyzer
	trendDetector   *TrendDetector
	
	// Optimization recommendations
	optimizer       *PerformanceOptimizer
	recommendations chan *Recommendation
	
	// Regression detection
	regressionDetector *RegressionDetector
	baselines          map[string]*PerformanceBaseline
	
	// Analytics engine
	analyticsEngine *AnalyticsEngine
	
	// Logging
	logger *zap.Logger
}

// CacheMetricsTracker monitors cache performance
type CacheMetricsTracker struct {
	mu sync.RWMutex
	
	// Hit rates by tier
	l1HitRate      *RollingWindow
	l2HitRate      *RollingWindow
	l3HitRate      *RollingWindow
	overallHitRate *RollingWindow
	
	// Latency tracking
	readLatency  *LatencyTracker
	writeLatency *LatencyTracker
	
	// Cache efficiency
	evictionRate    *RateCounter
	missRate        *RateCounter
	coherenceErrors *ErrorTracker
	
	// ML model performance
	predictionAccuracy *AccuracyTracker
	optimizationGains  *GainTracker
}

// ScalingMetricsTracker monitors auto-scaling decisions
type ScalingMetricsTracker struct {
	mu sync.RWMutex
	
	// Decision tracking
	scalingDecisions  []*ScalingEvent
	decisionAccuracy  *AccuracyTracker
	
	// Performance impact
	responseTimeImpact *ImpactAnalyzer
	throughputImpact   *ImpactAnalyzer
	
	// Stability metrics
	flappingDetector *FlappingDetector
	stabilityScore   float64
	
	// ML model metrics
	predictionError   *ErrorMetrics
	confidenceTracker *ConfidenceMetrics
}

// ResourceMetricsTracker monitors resource utilization
type ResourceMetricsTracker struct {
	mu sync.RWMutex
	
	// Utilization patterns
	cpuUtilization    *UtilizationTracker
	memoryUtilization *UtilizationTracker
	diskUtilization   *UtilizationTracker
	networkUtilization *UtilizationTracker
	
	// Efficiency metrics
	resourceEfficiency float64
	wasteAnalyzer     *WasteAnalyzer
	
	// Capacity planning
	capacityPredictor *CapacityPredictor
}

// AnomalyDetector identifies abnormal patterns
type AnomalyDetector struct {
	model           *IsolationForest
	zscore          *ZScoreDetector
	mahalanobis     *MahalanobisDetector
	changepoint     *ChangepointDetector
	sensitivityLevel float64
}

// PatternAnalyzer identifies recurring patterns
type PatternAnalyzer struct {
	seasonalDetector *SeasonalDetector
	correlationMatrix map[string]map[string]float64
	clusterAnalyzer  *ClusterAnalyzer
	sequenceDetector *SequenceDetector
}

// PerformanceOptimizer generates optimization recommendations
type PerformanceOptimizer struct {
	reinforcementLearner *RLOptimizer
	geneticOptimizer     *GeneticAlgorithm
	bayesianOptimizer    *BayesianOptimization
}

// RegressionDetector identifies performance degradation
type RegressionDetector struct {
	baselineWindow   time.Duration
	detectionWindow  time.Duration
	sensitivityLevel float64
	regressions      []*RegressionEvent
}

// NewPerformanceMonitor creates a comprehensive performance monitor
func NewPerformanceMonitor(logger *zap.Logger) *PerformanceMonitor {
	pm := &PerformanceMonitor{
		logger:          logger,
		recommendations: make(chan *Recommendation, 100),
		baselines:       make(map[string]*PerformanceBaseline),
	}
	
	// Initialize metrics trackers
	pm.cacheMetrics = &CacheMetricsTracker{
		l1HitRate:          NewRollingWindow(100),
		l2HitRate:          NewRollingWindow(100),
		l3HitRate:          NewRollingWindow(100),
		overallHitRate:     NewRollingWindow(100),
		readLatency:        NewLatencyTracker(),
		writeLatency:       NewLatencyTracker(),
		evictionRate:       NewRateCounter(),
		missRate:           NewRateCounter(),
		coherenceErrors:    NewErrorTracker(),
		predictionAccuracy: NewAccuracyTracker(),
		optimizationGains:  NewGainTracker(),
	}
	
	pm.scalingMetrics = &ScalingMetricsTracker{
		scalingDecisions:   make([]*ScalingEvent, 0, 1000),
		decisionAccuracy:   NewAccuracyTracker(),
		responseTimeImpact: NewImpactAnalyzer(),
		throughputImpact:   NewImpactAnalyzer(),
		flappingDetector:   NewFlappingDetector(),
		predictionError:    NewErrorMetrics(),
		confidenceTracker:  NewConfidenceMetrics(),
	}
	
	pm.resourceMetrics = &ResourceMetricsTracker{
		cpuUtilization:     NewUtilizationTracker(),
		memoryUtilization:  NewUtilizationTracker(),
		diskUtilization:    NewUtilizationTracker(),
		networkUtilization: NewUtilizationTracker(),
		wasteAnalyzer:      NewWasteAnalyzer(),
		capacityPredictor:  NewCapacityPredictor(),
	}
	
	pm.costMetrics = NewCostMetricsTracker()
	
	// Initialize pattern recognition
	pm.anomalyDetector = &AnomalyDetector{
		model:            NewIsolationForest(),
		zscore:           NewZScoreDetector(3.0),
		mahalanobis:      NewMahalanobisDetector(),
		changepoint:      NewChangepointDetector(),
		sensitivityLevel: 0.95,
	}
	
	pm.patternAnalyzer = &PatternAnalyzer{
		seasonalDetector:  NewSeasonalDetector(),
		correlationMatrix: make(map[string]map[string]float64),
		clusterAnalyzer:   NewClusterAnalyzer(),
		sequenceDetector:  NewSequenceDetector(),
	}
	
	pm.trendDetector = NewTrendDetector()
	
	// Initialize optimizer
	pm.optimizer = &PerformanceOptimizer{
		reinforcementLearner: NewRLOptimizer(),
		geneticOptimizer:     NewGeneticAlgorithm(),
		bayesianOptimizer:    NewBayesianOptimization(),
	}
	
	// Initialize regression detector
	pm.regressionDetector = &RegressionDetector{
		baselineWindow:   24 * time.Hour,
		detectionWindow:  1 * time.Hour,
		sensitivityLevel: 0.95,
		regressions:      make([]*RegressionEvent, 0),
	}
	
	// Initialize analytics engine
	pm.analyticsEngine = NewAnalyticsEngine()
	
	// Start monitoring loops
	go pm.runMonitoringLoop()
	go pm.runAnalyticsLoop()
	go pm.runOptimizationLoop()
	
	return pm
}

// RecordCacheMetrics updates cache performance metrics
func (pm *PerformanceMonitor) RecordCacheMetrics(metrics *CacheMetrics) {
	pm.cacheMetrics.mu.Lock()
	defer pm.cacheMetrics.mu.Unlock()
	
	// Update hit rates
	pm.cacheMetrics.l1HitRate.Add(metrics.L1HitRate)
	pm.cacheMetrics.l2HitRate.Add(metrics.L2HitRate)
	pm.cacheMetrics.l3HitRate.Add(metrics.L3HitRate)
	pm.cacheMetrics.overallHitRate.Add(metrics.OverallHitRate)
	
	// Update latencies
	pm.cacheMetrics.readLatency.Record(metrics.ReadLatency)
	pm.cacheMetrics.writeLatency.Record(metrics.WriteLatency)
	
	// Update rates
	pm.cacheMetrics.evictionRate.Record(metrics.Evictions)
	pm.cacheMetrics.missRate.Record(metrics.Misses)
	
	// Check for anomalies
	if pm.detectCacheAnomaly(metrics) {
		pm.generateCacheRecommendation(metrics)
	}
}

// RecordScalingEvent tracks auto-scaling decisions
func (pm *PerformanceMonitor) RecordScalingEvent(event *ScalingEvent) {
	pm.scalingMetrics.mu.Lock()
	defer pm.scalingMetrics.mu.Unlock()
	
	// Store event
	pm.scalingMetrics.scalingDecisions = append(pm.scalingMetrics.scalingDecisions, event)
	
	// Update accuracy if we have ground truth
	if event.ActualRequired != 0 {
		accuracy := 1.0 - math.Abs(float64(event.Predicted-event.ActualRequired))/float64(event.ActualRequired)
		pm.scalingMetrics.decisionAccuracy.Record(accuracy)
	}
	
	// Analyze impact
	pm.scalingMetrics.responseTimeImpact.Analyze(event.ResponseTimeChange)
	pm.scalingMetrics.throughputImpact.Analyze(event.ThroughputChange)
	
	// Check for flapping
	if pm.scalingMetrics.flappingDetector.IsFlapping(event) {
		pm.generateStabilizationRecommendation()
	}
	
	// Update stability score
	pm.updateStabilityScore()
}

// AnalyzeCostOptimization performs cost analysis
func (pm *PerformanceMonitor) AnalyzeCostOptimization() *CostAnalysis {
	analysis := &CostAnalysis{
		Timestamp: time.Now(),
	}
	
	// Calculate current costs
	analysis.CurrentHourlyCost = pm.costMetrics.CalculateHourlyCost()
	analysis.ProjectedMonthlyCost = analysis.CurrentHourlyCost * 24 * 30
	
	// Identify optimization opportunities
	analysis.Opportunities = pm.identifyCostOptimizations()
	
	// Calculate potential savings
	for _, opp := range analysis.Opportunities {
		analysis.PotentialSavings += opp.EstimatedSavings
	}
	
	// Generate recommendations
	analysis.Recommendations = pm.generateCostRecommendations(analysis)
	
	return analysis
}

// DetectPerformanceRegression identifies performance degradation
func (pm *PerformanceMonitor) DetectPerformanceRegression(metric string, current float64) bool {
	pm.mu.RLock()
	baseline, exists := pm.baselines[metric]
	pm.mu.RUnlock()
	
	if !exists {
		return false
	}
	
	// Calculate z-score
	zscore := (current - baseline.Mean) / baseline.StdDev
	
	// Check if regression
	if math.Abs(zscore) > 3.0 {
		regression := &RegressionEvent{
			Metric:    metric,
			Timestamp: time.Now(),
			Baseline:  baseline.Mean,
			Current:   current,
			Severity:  pm.calculateSeverity(zscore),
		}
		
		pm.regressionDetector.regressions = append(pm.regressionDetector.regressions, regression)
		
		// Generate alert
		pm.generateRegressionAlert(regression)
		
		return true
	}
	
	return false
}

// GeneratePerformanceReport creates comprehensive performance report
func (pm *PerformanceMonitor) GeneratePerformanceReport() *PerformanceReport {
	report := &PerformanceReport{
		Timestamp: time.Now(),
		Period:    24 * time.Hour,
	}
	
	// Cache performance
	report.CachePerformance = &CachePerformanceReport{
		OverallHitRate:     pm.cacheMetrics.overallHitRate.Average(),
		L1HitRate:         pm.cacheMetrics.l1HitRate.Average(),
		L2HitRate:         pm.cacheMetrics.l2HitRate.Average(),
		L3HitRate:         pm.cacheMetrics.l3HitRate.Average(),
		AvgReadLatency:    pm.cacheMetrics.readLatency.Average(),
		AvgWriteLatency:   pm.cacheMetrics.writeLatency.Average(),
		EvictionRate:      pm.cacheMetrics.evictionRate.GetRate(),
		OptimizationGains: pm.cacheMetrics.optimizationGains.GetTotal(),
	}
	
	// Scaling performance
	report.ScalingPerformance = &ScalingPerformanceReport{
		TotalDecisions:    len(pm.scalingMetrics.scalingDecisions),
		DecisionAccuracy:  pm.scalingMetrics.decisionAccuracy.GetAverage(),
		StabilityScore:    pm.scalingMetrics.stabilityScore,
		AvgResponseImpact: pm.scalingMetrics.responseTimeImpact.GetAverage(),
		AvgThroughputGain: pm.scalingMetrics.throughputImpact.GetAverage(),
		PredictionError:   pm.scalingMetrics.predictionError.GetMAPE(),
	}
	
	// Resource utilization
	report.ResourceUtilization = &ResourceReport{
		AvgCPU:             pm.resourceMetrics.cpuUtilization.GetAverage(),
		AvgMemory:          pm.resourceMetrics.memoryUtilization.GetAverage(),
		AvgDisk:            pm.resourceMetrics.diskUtilization.GetAverage(),
		AvgNetwork:         pm.resourceMetrics.networkUtilization.GetAverage(),
		ResourceEfficiency: pm.resourceMetrics.resourceEfficiency,
		WastedResources:    pm.resourceMetrics.wasteAnalyzer.GetWastePercentage(),
	}
	
	// Cost analysis
	report.CostAnalysis = pm.AnalyzeCostOptimization()
	
	// Detected patterns
	report.Patterns = pm.patternAnalyzer.GetDetectedPatterns()
	
	// Anomalies
	report.Anomalies = pm.getRecentAnomalies()
	
	// Recommendations
	report.Recommendations = pm.getTopRecommendations(10)
	
	return report
}

// runMonitoringLoop continuously monitors performance
func (pm *PerformanceMonitor) runMonitoringLoop() {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for range ticker.C {
		// Collect current metrics
		metrics := pm.collectCurrentMetrics()
		
		// Detect anomalies
		anomalies := pm.anomalyDetector.Detect(metrics)
		for _, anomaly := range anomalies {
			pm.handleAnomaly(anomaly)
		}
		
		// Update baselines
		pm.updateBaselines(metrics)
		
		// Check for regressions
		pm.checkRegressions(metrics)
	}
}

// runAnalyticsLoop performs continuous analysis
func (pm *PerformanceMonitor) runAnalyticsLoop() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		// Pattern analysis
		patterns := pm.patternAnalyzer.AnalyzePatterns()
		
		// Trend detection
		trends := pm.trendDetector.DetectTrends()
		
		// Correlation analysis
		pm.updateCorrelations()
		
		// Update predictions
		pm.updatePredictions(patterns, trends)
	}
}

// runOptimizationLoop generates optimization recommendations
func (pm *PerformanceMonitor) runOptimizationLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	
	for range ticker.C {
		// Run optimization algorithms
		recommendations := pm.optimizer.Optimize()
		
		// Validate recommendations
		validRecs := pm.validateRecommendations(recommendations)
		
		// Queue recommendations
		for _, rec := range validRecs {
			select {
			case pm.recommendations <- rec:
			default:
				// Channel full, drop oldest
				<-pm.recommendations
				pm.recommendations <- rec
			}
		}
	}
}

// Helper types and methods

type CacheMetrics struct {
	L1HitRate      float64
	L2HitRate      float64
	L3HitRate      float64
	OverallHitRate float64
	ReadLatency    time.Duration
	WriteLatency   time.Duration
	Evictions      int64
	Misses         int64
}

type ScalingEvent struct {
	Timestamp          time.Time
	Predicted          int32
	ActualRequired     int32
	ResponseTimeChange float64
	ThroughputChange   float64
}

type RegressionEvent struct {
	Metric    string
	Timestamp time.Time
	Baseline  float64
	Current   float64
	Severity  string
}

type Recommendation struct {
	ID          string
	Type        string
	Priority    int
	Description string
	Impact      string
	Effort      string
	Confidence  float64
}

type PerformanceBaseline struct {
	Metric  string
	Mean    float64
	StdDev  float64
	P50     float64
	P95     float64
	P99     float64
	Updated time.Time
}

// RollingWindow maintains a sliding window of values
type RollingWindow struct {
	values []float64
	size   int
	index  int
	full   bool
}

func NewRollingWindow(size int) *RollingWindow {
	return &RollingWindow{
		values: make([]float64, size),
		size:   size,
	}
}

func (rw *RollingWindow) Add(value float64) {
	rw.values[rw.index] = value
	rw.index = (rw.index + 1) % rw.size
	if rw.index == 0 {
		rw.full = true
	}
}

func (rw *RollingWindow) Average() float64 {
	sum := 0.0
	count := rw.size
	if !rw.full {
		count = rw.index
	}
	
	for i := 0; i < count; i++ {
		sum += rw.values[i]
	}
	
	if count == 0 {
		return 0
	}
	
	return sum / float64(count)
}