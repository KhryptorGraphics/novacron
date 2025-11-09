package testing

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ABTester implements automatic A/B testing
type ABTester struct {
	logger             *zap.Logger
	experiments        map[string]*Experiment
	statisticalEngine  *StatisticalEngine
	allocationStrategy *AllocationStrategy
	autoDeployer       *AutoDeployer
	rollbackManager    *RollbackManager
	significanceLevel  float64
	minSampleSize      int
	mu                 sync.RWMutex
}

// Experiment represents an A/B test experiment
type Experiment struct {
	ID            string
	Name          string
	Description   string
	Control       *Variant
	Treatments    []*Variant
	StartTime     time.Time
	EndTime       *time.Time
	Status        ExperimentStatus
	Metrics       []*Metric
	Results       *ExperimentResults
	AutoDeploy    bool
	RollbackOnFail bool
}

// Variant represents a test variant
type Variant struct {
	ID          string
	Name        string
	Config      map[string]interface{}
	Traffic     float64
	Samples     int
	Conversions int
	Metrics     map[string]*MetricValue
}

// ExperimentStatus defines experiment status
type ExperimentStatus string

const (
	ExperimentPending   ExperimentStatus = "pending"
	ExperimentRunning   ExperimentStatus = "running"
	ExperimentCompleted ExperimentStatus = "completed"
	ExperimentStopped   ExperimentStatus = "stopped"
)

// Metric represents a tracked metric
type Metric struct {
	Name        string
	Type        MetricType
	Primary     bool
	Direction   MetricDirection
	MinDelta    float64
}

// MetricType defines metric types
type MetricType string

const (
	BinaryMetric     MetricType = "binary"
	ContinuousMetric MetricType = "continuous"
	CountMetric      MetricType = "count"
)

// MetricDirection defines improvement direction
type MetricDirection string

const (
	MetricIncrease MetricDirection = "increase"
	MetricDecrease MetricDirection = "decrease"
)

// MetricValue contains metric measurements
type MetricValue struct {
	Count       int
	Sum         float64
	SumSquares  float64
	Mean        float64
	Variance    float64
	StdDev      float64
}

// ExperimentResults contains experiment results
type ExperimentResults struct {
	Winner           *Variant
	SignificantDiff  bool
	PValue           float64
	ConfidenceLevel  float64
	EffectSize       float64
	SampleSizeReached bool
	Duration         time.Duration
	Recommendations  []string
}

// StatisticalEngine performs statistical analysis
type StatisticalEngine struct {
	logger *zap.Logger
	tests  map[string]StatisticalTest
}

// StatisticalTest interface for different tests
type StatisticalTest interface {
	Calculate(control, treatment *Variant, metric *Metric) *TestResult
}

// TestResult contains statistical test results
type TestResult struct {
	PValue          float64
	Significant     bool
	EffectSize      float64
	ConfidenceInterval [2]float64
	Power           float64
}

// AllocationStrategy manages traffic allocation
type AllocationStrategy struct {
	logger    *zap.Logger
	algorithm AllocationAlgorithm
	bandit    *MultiArmedBandit
}

// AllocationAlgorithm defines allocation algorithms
type AllocationAlgorithm string

const (
	FixedAllocation    AllocationAlgorithm = "fixed"
	AdaptiveAllocation AllocationAlgorithm = "adaptive"
	BanditAllocation   AllocationAlgorithm = "bandit"
)

// MultiArmedBandit implements multi-armed bandit allocation
type MultiArmedBandit struct {
	arms        map[string]*BanditArm
	exploration float64
	mu          sync.RWMutex
}

// BanditArm represents a bandit arm
type BanditArm struct {
	ID          string
	Pulls       int
	Rewards     float64
	AvgReward   float64
	UCB         float64
}

// AutoDeployer handles automatic winner deployment
type AutoDeployer struct {
	logger         *zap.Logger
	deploymentMgr  *DeploymentManager
	canaryRatio    float64
	rolloutSpeed   time.Duration
}

// RollbackManager handles experiment rollbacks
type RollbackManager struct {
	logger        *zap.Logger
	checkInterval time.Duration
	threshold     float64
}

// NewABTester creates a new A/B tester
func NewABTester(logger *zap.Logger) *ABTester {
	return &ABTester{
		logger:             logger,
		experiments:        make(map[string]*Experiment),
		statisticalEngine:  NewStatisticalEngine(logger),
		allocationStrategy: NewAllocationStrategy(logger),
		autoDeployer:       NewAutoDeployer(logger),
		rollbackManager:    NewRollbackManager(logger),
		significanceLevel:  0.05,
		minSampleSize:      1000,
	}
}

// CreateExperiment creates a new A/B test experiment
func (abt *ABTester) CreateExperiment(ctx context.Context, config *ExperimentConfig) (*Experiment, error) {
	abt.logger.Info("Creating new experiment",
		zap.String("name", config.Name))

	experiment := &Experiment{
		ID:          generateExperimentID(),
		Name:        config.Name,
		Description: config.Description,
		Control:     config.Control,
		Treatments:  config.Treatments,
		StartTime:   time.Now(),
		Status:      ExperimentPending,
		Metrics:     config.Metrics,
		AutoDeploy:  config.AutoDeploy,
		RollbackOnFail: config.RollbackOnFail,
	}

	// Validate experiment configuration
	if err := abt.validateExperiment(experiment); err != nil {
		return nil, err
	}

	// Store experiment
	abt.mu.Lock()
	abt.experiments[experiment.ID] = experiment
	abt.mu.Unlock()

	// Start experiment
	go abt.runExperiment(ctx, experiment)

	abt.logger.Info("Experiment created",
		zap.String("id", experiment.ID),
		zap.Int("variants", len(experiment.Treatments)+1))

	return experiment, nil
}

// runExperiment runs an A/B test experiment
func (abt *ABTester) runExperiment(ctx context.Context, exp *Experiment) {
	exp.Status = ExperimentRunning

	// Monitor experiment
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			abt.stopExperiment(exp)
			return
		case <-ticker.C:
			// Update metrics
			abt.updateMetrics(exp)

			// Check for statistical significance
			if abt.checkSignificance(exp) {
				abt.concludeExperiment(exp)
				return
			}

			// Check for degradation and rollback if needed
			if exp.RollbackOnFail && abt.rollbackManager.ShouldRollback(exp) {
				abt.rollbackExperiment(exp)
				return
			}

			// Update allocation if using adaptive strategy
			if abt.allocationStrategy.algorithm == AdaptiveAllocation {
				abt.updateAllocation(exp)
			}
		}
	}
}

// updateMetrics updates experiment metrics
func (abt *ABTester) updateMetrics(exp *Experiment) {
	// Simulate metric collection (in production, would fetch from monitoring)
	for _, variant := range append([]*Variant{exp.Control}, exp.Treatments...) {
		variant.Samples += rand.Intn(100)
		variant.Conversions += rand.Intn(variant.Samples / 10)

		for _, metric := range exp.Metrics {
			abt.updateVariantMetric(variant, metric)
		}
	}
}

// updateVariantMetric updates a specific metric for a variant
func (abt *ABTester) updateVariantMetric(variant *Variant, metric *Metric) {
	if variant.Metrics == nil {
		variant.Metrics = make(map[string]*MetricValue)
	}

	if _, exists := variant.Metrics[metric.Name]; !exists {
		variant.Metrics[metric.Name] = &MetricValue{}
	}

	mv := variant.Metrics[metric.Name]

	// Simulate metric values
	value := rand.Float64() * 100
	mv.Count++
	mv.Sum += value
	mv.SumSquares += value * value
	mv.Mean = mv.Sum / float64(mv.Count)

	if mv.Count > 1 {
		mv.Variance = (mv.SumSquares - mv.Sum*mv.Sum/float64(mv.Count)) / float64(mv.Count-1)
		mv.StdDev = math.Sqrt(mv.Variance)
	}
}

// checkSignificance checks for statistical significance
func (abt *ABTester) checkSignificance(exp *Experiment) bool {
	// Check sample size
	totalSamples := exp.Control.Samples
	for _, treatment := range exp.Treatments {
		totalSamples += treatment.Samples
	}

	if totalSamples < abt.minSampleSize {
		return false
	}

	// Perform statistical tests
	results := abt.analyzeExperiment(exp)

	if results.SignificantDiff && results.SampleSizeReached {
		exp.Results = results
		return true
	}

	// Check for auto-termination based on duration
	if time.Since(exp.StartTime) > 7*24*time.Hour {
		exp.Results = results
		return true
	}

	return false
}

// analyzeExperiment performs statistical analysis
func (abt *ABTester) analyzeExperiment(exp *Experiment) *ExperimentResults {
	results := &ExperimentResults{
		Duration:         time.Since(exp.StartTime),
		SampleSizeReached: exp.Control.Samples >= abt.minSampleSize,
		Recommendations:  make([]string, 0),
	}

	// Find primary metric
	var primaryMetric *Metric
	for _, metric := range exp.Metrics {
		if metric.Primary {
			primaryMetric = metric
			break
		}
	}

	if primaryMetric == nil && len(exp.Metrics) > 0 {
		primaryMetric = exp.Metrics[0]
	}

	// Compare each treatment to control
	bestVariant := exp.Control
	bestPerformance := 0.0

	for _, treatment := range exp.Treatments {
		testResult := abt.statisticalEngine.Test(exp.Control, treatment, primaryMetric)

		if testResult.Significant && testResult.PValue < abt.significanceLevel {
			results.SignificantDiff = true
			results.PValue = testResult.PValue
			results.EffectSize = testResult.EffectSize

			// Determine winner based on metric direction
			performance := abt.calculatePerformance(treatment, primaryMetric)
			if performance > bestPerformance {
				bestPerformance = performance
				bestVariant = treatment
			}
		}
	}

	results.Winner = bestVariant
	results.ConfidenceLevel = 1 - abt.significanceLevel

	// Add recommendations
	if results.SignificantDiff {
		results.Recommendations = append(results.Recommendations,
			fmt.Sprintf("Deploy %s variant for %.2f%% improvement", bestVariant.Name, results.EffectSize*100))
	} else {
		results.Recommendations = append(results.Recommendations,
			"Continue experiment to reach statistical significance")
	}

	return results
}

// Test performs statistical test
func (se *StatisticalEngine) Test(control, treatment *Variant, metric *Metric) *TestResult {
	// Perform appropriate test based on metric type
	var test StatisticalTest

	switch metric.Type {
	case BinaryMetric:
		test = &ChiSquareTest{}
	case ContinuousMetric:
		test = &TTest{}
	default:
		test = &MannWhitneyTest{}
	}

	return test.Calculate(control, treatment, metric)
}

// concludeExperiment concludes the experiment
func (abt *ABTester) concludeExperiment(exp *Experiment) {
	exp.Status = ExperimentCompleted
	endTime := time.Now()
	exp.EndTime = &endTime

	abt.logger.Info("Experiment concluded",
		zap.String("id", exp.ID),
		zap.String("winner", exp.Results.Winner.Name),
		zap.Float64("p_value", exp.Results.PValue),
		zap.Duration("duration", exp.Results.Duration))

	// Auto-deploy winner if configured
	if exp.AutoDeploy && exp.Results.SignificantDiff {
		abt.deployWinner(exp)
	}
}

// deployWinner deploys the winning variant
func (abt *ABTester) deployWinner(exp *Experiment) {
	abt.logger.Info("Auto-deploying winner",
		zap.String("experiment", exp.ID),
		zap.String("winner", exp.Results.Winner.Name))

	// Gradual rollout
	deployment := &Deployment{
		ExperimentID: exp.ID,
		VariantID:    exp.Results.Winner.ID,
		Config:       exp.Results.Winner.Config,
		Strategy:     "canary",
		StartRatio:   abt.autoDeployer.canaryRatio,
		EndRatio:     1.0,
		Duration:     abt.autoDeployer.rolloutSpeed,
	}

	abt.autoDeployer.Deploy(deployment)
}

// updateAllocation updates traffic allocation using multi-armed bandit
func (abt *ABTester) updateAllocation(exp *Experiment) {
	if abt.allocationStrategy.algorithm != BanditAllocation {
		return
	}

	// Update bandit arms with rewards
	for _, variant := range append([]*Variant{exp.Control}, exp.Treatments...) {
		reward := float64(variant.Conversions) / float64(max(variant.Samples, 1))
		abt.allocationStrategy.bandit.UpdateArm(variant.ID, reward)
	}

	// Get new allocation
	allocation := abt.allocationStrategy.bandit.GetAllocation()

	// Apply allocation
	for _, variant := range append([]*Variant{exp.Control}, exp.Treatments...) {
		if traffic, exists := allocation[variant.ID]; exists {
			variant.Traffic = traffic
		}
	}
}

// UpdateArm updates a bandit arm with reward
func (mab *MultiArmedBandit) UpdateArm(armID string, reward float64) {
	mab.mu.Lock()
	defer mab.mu.Unlock()

	if _, exists := mab.arms[armID]; !exists {
		mab.arms[armID] = &BanditArm{ID: armID}
	}

	arm := mab.arms[armID]
	arm.Pulls++
	arm.Rewards += reward
	arm.AvgReward = arm.Rewards / float64(arm.Pulls)

	// Update UCB (Upper Confidence Bound)
	totalPulls := 0
	for _, a := range mab.arms {
		totalPulls += a.Pulls
	}

	if totalPulls > 0 && arm.Pulls > 0 {
		arm.UCB = arm.AvgReward + math.Sqrt(2*math.Log(float64(totalPulls))/float64(arm.Pulls))
	}
}

// GetAllocation returns traffic allocation based on UCB
func (mab *MultiArmedBandit) GetAllocation() map[string]float64 {
	mab.mu.RLock()
	defer mab.mu.RUnlock()

	allocation := make(map[string]float64)
	totalUCB := 0.0

	for _, arm := range mab.arms {
		totalUCB += arm.UCB
	}

	if totalUCB > 0 {
		for id, arm := range mab.arms {
			allocation[id] = arm.UCB / totalUCB
		}
	}

	return allocation
}

// ShouldRollback checks if experiment should be rolled back
func (rm *RollbackManager) ShouldRollback(exp *Experiment) bool {
	// Check for significant degradation in control metrics
	for _, metric := range exp.Metrics {
		if metric.Name == "error_rate" || metric.Name == "latency" {
			for _, treatment := range exp.Treatments {
				controlValue := exp.Control.Metrics[metric.Name].Mean
				treatmentValue := treatment.Metrics[metric.Name].Mean

				// Rollback if degradation exceeds threshold
				degradation := (treatmentValue - controlValue) / controlValue
				if degradation > rm.threshold {
					return true
				}
			}
		}
	}

	return false
}

// Helper functions

func (abt *ABTester) validateExperiment(exp *Experiment) error {
	if exp.Control == nil {
		return fmt.Errorf("control variant required")
	}

	if len(exp.Treatments) == 0 {
		return fmt.Errorf("at least one treatment variant required")
	}

	if len(exp.Metrics) == 0 {
		return fmt.Errorf("at least one metric required")
	}

	// Validate traffic allocation sums to 1
	totalTraffic := exp.Control.Traffic
	for _, treatment := range exp.Treatments {
		totalTraffic += treatment.Traffic
	}

	if math.Abs(totalTraffic-1.0) > 0.01 {
		return fmt.Errorf("traffic allocation must sum to 1.0")
	}

	return nil
}

func (abt *ABTester) calculatePerformance(variant *Variant, metric *Metric) float64 {
	if mv, exists := variant.Metrics[metric.Name]; exists {
		if metric.Direction == MetricIncrease {
			return mv.Mean
		}
		return -mv.Mean
	}
	return 0
}

func (abt *ABTester) stopExperiment(exp *Experiment) {
	exp.Status = ExperimentStopped
	endTime := time.Now()
	exp.EndTime = &endTime
}

func (abt *ABTester) rollbackExperiment(exp *Experiment) {
	abt.logger.Warn("Rolling back experiment due to degradation",
		zap.String("id", exp.ID))

	exp.Status = ExperimentStopped
	// Restore control configuration
}

func generateExperimentID() string {
	return "exp-" + generateID()
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Supporting types

type ExperimentConfig struct {
	Name           string
	Description    string
	Control        *Variant
	Treatments     []*Variant
	Metrics        []*Metric
	AutoDeploy     bool
	RollbackOnFail bool
}

type Deployment struct {
	ExperimentID string
	VariantID    string
	Config       map[string]interface{}
	Strategy     string
	StartRatio   float64
	EndRatio     float64
	Duration     time.Duration
}

type DeploymentManager struct{}

func (ad *AutoDeployer) Deploy(deployment *Deployment) {}

// Statistical test implementations

type ChiSquareTest struct{}

func (cst *ChiSquareTest) Calculate(control, treatment *Variant, metric *Metric) *TestResult {
	// Chi-square test for binary metrics
	n1, x1 := float64(control.Samples), float64(control.Conversions)
	n2, x2 := float64(treatment.Samples), float64(treatment.Conversions)

	p1 := x1 / n1
	p2 := x2 / n2
	p := (x1 + x2) / (n1 + n2)

	z := (p1 - p2) / math.Sqrt(p*(1-p)*(1/n1+1/n2))
	pValue := 2 * (1 - normalCDF(math.Abs(z)))

	return &TestResult{
		PValue:      pValue,
		Significant: pValue < 0.05,
		EffectSize:  (p2 - p1) / p1,
	}
}

type TTest struct{}

func (tt *TTest) Calculate(control, treatment *Variant, metric *Metric) *TestResult {
	// T-test for continuous metrics
	mv1 := control.Metrics[metric.Name]
	mv2 := treatment.Metrics[metric.Name]

	t := (mv2.Mean - mv1.Mean) / math.Sqrt(mv1.Variance/float64(control.Samples)+mv2.Variance/float64(treatment.Samples))
	df := float64(control.Samples + treatment.Samples - 2)
	pValue := 2 * (1 - tCDF(math.Abs(t), df))

	return &TestResult{
		PValue:      pValue,
		Significant: pValue < 0.05,
		EffectSize:  (mv2.Mean - mv1.Mean) / mv1.Mean,
	}
}

type MannWhitneyTest struct{}

func (mwt *MannWhitneyTest) Calculate(control, treatment *Variant, metric *Metric) *TestResult {
	// Mann-Whitney U test (simplified)
	return &TestResult{
		PValue:      0.05,
		Significant: false,
		EffectSize:  0,
	}
}

func normalCDF(z float64) float64 {
	// Simplified normal CDF
	return 0.5 * (1 + math.Erf(z/math.Sqrt(2)))
}

func tCDF(t, df float64) float64 {
	// Simplified t-distribution CDF
	return normalCDF(t)
}

// Constructor functions

func NewStatisticalEngine(logger *zap.Logger) *StatisticalEngine {
	return &StatisticalEngine{
		logger: logger,
		tests:  make(map[string]StatisticalTest),
	}
}

func NewAllocationStrategy(logger *zap.Logger) *AllocationStrategy {
	return &AllocationStrategy{
		logger:    logger,
		algorithm: AdaptiveAllocation,
		bandit: &MultiArmedBandit{
			arms:        make(map[string]*BanditArm),
			exploration: 0.1,
		},
	}
}

func NewAutoDeployer(logger *zap.Logger) *AutoDeployer {
	return &AutoDeployer{
		logger:        logger,
		deploymentMgr: &DeploymentManager{},
		canaryRatio:   0.1,
		rolloutSpeed:  24 * time.Hour,
	}
}

func NewRollbackManager(logger *zap.Logger) *RollbackManager {
	return &RollbackManager{
		logger:        logger,
		checkInterval: time.Minute,
		threshold:     0.1,
	}
}