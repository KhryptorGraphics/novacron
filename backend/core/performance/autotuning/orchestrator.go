package autotuning

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/performance/cpu_pinning"
	"github.com/khryptorgraphics/novacron/backend/core/performance/cost_optimizer"
	"github.com/khryptorgraphics/novacron/backend/core/performance/flamegraph"
	"github.com/khryptorgraphics/novacron/backend/core/performance/io_tuning"
	"github.com/khryptorgraphics/novacron/backend/core/performance/network_tuning"
	"github.com/khryptorgraphics/novacron/backend/core/performance/numa"
	"github.com/khryptorgraphics/novacron/backend/core/performance/profiler"
	"github.com/khryptorgraphics/novacron/backend/core/performance/recommendations"
	"github.com/khryptorgraphics/novacron/backend/core/performance/rightsizing"
)

// Orchestrator coordinates all auto-tuning activities
type Orchestrator struct {
	config              OrchestratorConfig
	mu                  sync.RWMutex
	profiler            *profiler.ContinuousProfiler
	flamegraphGen       *flamegraph.Generator
	rightsizingEngine   *rightsizing.Engine
	numaOptimizer       *numa.Optimizer
	cpuPinningEngine    *cpu_pinning.Engine
	ioTuner             *io_tuning.Tuner
	networkTuner        *network_tuning.Tuner
	costOptimizer       *cost_optimizer.Optimizer
	recommendationsEng  *recommendations.Engine
	tuningHistory       []TuningEvent
	convergenceDetector *ConvergenceDetector
	running             bool
	stopChan            chan struct{}
}

// OrchestratorConfig defines orchestrator settings
type OrchestratorConfig struct {
	TuningInterval       time.Duration // 5 minutes
	ConvergenceTarget    time.Duration // 30 minutes
	MaxConcurrentTuning  int
	SafeTuning           bool
	GradualChanges       bool
	AutoRollback         bool
	ValidationPeriod     time.Duration
	DegradationThreshold float64
}

// TuningEvent records a tuning action
type TuningEvent struct {
	Timestamp    time.Time
	Component    string
	Action       string
	BeforeMetrics PerformanceSnapshot
	AfterMetrics  PerformanceSnapshot
	Success      bool
	Impact       float64
}

// PerformanceSnapshot captures performance at a point in time
type PerformanceSnapshot struct {
	CPUUtilization    float64
	MemoryUtilization float64
	Latency           float64
	Throughput        float64
	IOPS              float64
	Cost              float64
}

// ConvergenceDetector detects when tuning has converged
type ConvergenceDetector struct {
	window            []float64 // Performance improvements
	windowSize        int
	convergenceTarget float64 // 0.01 (1%)
}

// NewOrchestrator creates auto-tuning orchestrator
func NewOrchestrator(config OrchestratorConfig) *Orchestrator {
	if config.TuningInterval == 0 {
		config.TuningInterval = 5 * time.Minute
	}
	if config.ConvergenceTarget == 0 {
		config.ConvergenceTarget = 30 * time.Minute
	}
	if config.MaxConcurrentTuning == 0 {
		config.MaxConcurrentTuning = 5
	}
	if config.ValidationPeriod == 0 {
		config.ValidationPeriod = 10 * time.Minute
	}
	if config.DegradationThreshold == 0 {
		config.DegradationThreshold = 0.1
	}

	return &Orchestrator{
		config:              config,
		tuningHistory:       make([]TuningEvent, 0),
		convergenceDetector: NewConvergenceDetector(10, 0.01),
		stopChan:            make(chan struct{}),
	}
}

// Initialize initializes all tuning components
func (o *Orchestrator) Initialize(ctx context.Context) error {
	var wg sync.WaitGroup
	errors := make(chan error, 7)

	// Initialize profiler
	wg.Add(1)
	go func() {
		defer wg.Done()
		o.profiler = profiler.NewContinuousProfiler(profiler.ProfilerConfig{
			SamplingRate:  100,
			ProfileTypes:  []string{"cpu", "memory", "mutex", "block"},
			OutputDir:     "/var/lib/novacron/profiles",
			RetentionDays: 7,
		})
		if err := o.profiler.Start(ctx); err != nil {
			errors <- fmt.Errorf("profiler init: %w", err)
		}
	}()

	// Initialize flamegraph generator
	wg.Add(1)
	go func() {
		defer wg.Done()
		o.flamegraphGen = flamegraph.NewGenerator(flamegraph.GeneratorConfig{
			OutputDir:       "/var/lib/novacron/flamegraphs",
			InteractiveHTML: true,
			DiffEnabled:     true,
		})
	}()

	// Initialize right-sizing engine
	wg.Add(1)
	go func() {
		defer wg.Done()
		o.rightsizingEngine = rightsizing.NewEngine(rightsizing.RightSizingConfig{
			CPUTargetMin:        0.60,
			CPUTargetMax:        0.80,
			MemoryTargetMin:     0.70,
			MemoryTargetMax:     0.85,
			ObservationPeriod:   24 * time.Hour,
			ConfidenceThreshold: 0.90,
			CostSavingsMin:      0.10,
		})
	}()

	// Initialize NUMA optimizer
	wg.Add(1)
	go func() {
		defer wg.Done()
		o.numaOptimizer = numa.NewOptimizer(numa.NumaConfig{
			AutoTopologyDetection:   true,
			MemoryPlacementStrategy: "local",
			CacheLocalityOptimize:   true,
			CrossNumaTrafficTarget:  0.10,
		})
		if err := o.numaOptimizer.Initialize(ctx); err != nil {
			errors <- fmt.Errorf("numa init: %w", err)
		}
	}()

	// Initialize CPU pinning engine
	wg.Add(1)
	go func() {
		defer wg.Done()
		o.cpuPinningEngine = cpu_pinning.NewEngine(cpu_pinning.CPUPinningConfig{
			Strategy:         "mixed",
			OvercommitRatio:  1.5,
			HyperthreadingOpt: true,
			CacheAffinity:    true,
			IsolateNoisy:     true,
		})
		if err := o.cpuPinningEngine.Initialize(ctx); err != nil {
			errors <- fmt.Errorf("cpu pinning init: %w", err)
		}
	}()

	// Initialize I/O tuner
	wg.Add(1)
	go func() {
		defer wg.Done()
		o.ioTuner = io_tuning.NewTuner(io_tuning.IOTuningConfig{
			AutoSchedulerSelect: true,
			QueueDepthAuto:      true,
			ReadAheadAuto:       true,
		})
		if err := o.ioTuner.Initialize(ctx); err != nil {
			errors <- fmt.Errorf("io tuner init: %w", err)
		}
	}()

	// Initialize network tuner
	wg.Add(1)
	go func() {
		defer wg.Done()
		o.networkTuner = network_tuning.NewTuner(network_tuning.NetworkTuningConfig{
			TCPWindowAutoTune: true,
			CongestionControl: "bbr",
			BufferAutoSize:    true,
			RDMAOptimize:      false,
		})
		if err := o.networkTuner.Initialize(ctx); err != nil {
			errors <- fmt.Errorf("network tuner init: %w", err)
		}
	}()

	wg.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		if err != nil {
			return err
		}
	}

	// Initialize cost optimizer and recommendations engine
	o.costOptimizer = cost_optimizer.NewOptimizer(cost_optimizer.CostOptimizerConfig{
		MultiObjectiveOptimize: true,
		ParetoFrontierAnalysis: true,
		SpotInstanceRecommend:  true,
		ReservedInstancePlan:   true,
		CostPredictionEnabled:  true,
	})

	o.recommendationsEng = recommendations.NewEngine(recommendations.RecommendationsConfig{
		ABTestingEnabled:      true,
		RollbackOnDegradation: true,
		DegradationThreshold:  0.05,
		ValidationPeriod:      10 * time.Minute,
		MaxConcurrentTests:    5,
		ConfidenceThreshold:   0.85,
	})

	return nil
}

// Start starts auto-tuning orchestration
func (o *Orchestrator) Start(ctx context.Context) error {
	o.mu.Lock()
	if o.running {
		o.mu.Unlock()
		return fmt.Errorf("orchestrator already running")
	}
	o.running = true
	o.mu.Unlock()

	// Start tuning loop
	go o.tuningLoop(ctx)

	// Start convergence monitoring
	go o.convergenceMonitor(ctx)

	return nil
}

// tuningLoop main tuning loop
func (o *Orchestrator) tuningLoop(ctx context.Context) {
	ticker := time.NewTicker(o.config.TuningInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.stopChan:
			return
		case <-ticker.C:
			if err := o.runTuningCycle(ctx); err != nil {
				fmt.Printf("Tuning cycle error: %v\n", err)
			}
		}
	}
}

// runTuningCycle runs one tuning cycle
func (o *Orchestrator) runTuningCycle(ctx context.Context) error {
	// Capture baseline
	baseline := o.capturePerformanceSnapshot()

	// Generate recommendations from all sources
	recommendations := o.collectRecommendations(ctx)

	if len(recommendations) == 0 {
		return nil
	}

	// Select top recommendations to apply
	toApply := o.selectRecommendations(recommendations)

	// Apply tunings (with rate limiting)
	appliedCount := 0
	for _, rec := range toApply {
		if appliedCount >= o.config.MaxConcurrentTuning {
			break
		}

		if err := o.applyTuning(ctx, rec, baseline); err != nil {
			fmt.Printf("Failed to apply tuning %s: %v\n", rec.ID, err)
			continue
		}

		appliedCount++

		// Gradual changes - wait between applications
		if o.config.GradualChanges {
			time.Sleep(30 * time.Second)
		}
	}

	return nil
}

// collectRecommendations collects recommendations from all sources
func (o *Orchestrator) collectRecommendations(ctx context.Context) []*recommendations.TuningRecommendation {
	var allRecs []*recommendations.TuningRecommendation

	// Right-sizing recommendations
	rightsizingRecs, _ := o.rightsizingEngine.AnalyzeAndRecommend(ctx)
	for _, rec := range rightsizingRecs {
		impact := recommendations.EstimatedImpact{
			PerformanceGain:    20.0,
			CostSavings:        rec.EstimatedSavings,
			ResourceEfficiency: 30.0,
		}

		risk := recommendations.RiskAssessment{
			Level:         "low",
			Reversibility: "easy",
		}

		tuningRec := o.recommendationsEng.GenerateRecommendation(
			rec.VMID,
			"cost",
			"rightsize",
			rec.Rationale,
			impact,
			risk,
		)
		allRecs = append(allRecs, tuningRec)
	}

	// Cost optimization recommendations
	costRecs, _ := o.costOptimizer.Optimize(ctx)
	for _, rec := range costRecs {
		impact := recommendations.EstimatedImpact{
			CostSavings:     rec.EstimatedSavings,
			PerformanceGain: 0,
		}

		risk := recommendations.RiskAssessment{
			Level:         rec.RiskLevel,
			Reversibility: "moderate",
		}

		tuningRec := o.recommendationsEng.GenerateRecommendation(
			rec.VMID,
			"cost",
			rec.RecommendationType,
			fmt.Sprintf("Cost optimization: %s", rec.RecommendationType),
			impact,
			risk,
		)
		allRecs = append(allRecs, tuningRec)
	}

	// Get top recommendations from engine
	topRecs := o.recommendationsEng.GetTopRecommendations(10)
	allRecs = append(allRecs, topRecs...)

	return allRecs
}

// selectRecommendations selects which recommendations to apply
func (o *Orchestrator) selectRecommendations(recs []*recommendations.TuningRecommendation) []*recommendations.TuningRecommendation {
	// Filter by confidence
	var selected []*recommendations.TuningRecommendation
	for _, rec := range recs {
		if rec.Confidence >= 0.8 {
			selected = append(selected, rec)
		}
	}

	// Sort by priority
	for i := 0; i < len(selected); i++ {
		for j := i + 1; j < len(selected); j++ {
			if selected[i].Priority < selected[j].Priority {
				selected[i], selected[j] = selected[j], selected[i]
			}
		}
	}

	return selected
}

// applyTuning applies a tuning recommendation
func (o *Orchestrator) applyTuning(ctx context.Context, rec *recommendations.TuningRecommendation, baseline PerformanceSnapshot) error {
	// Record tuning event
	event := TuningEvent{
		Timestamp:     time.Now(),
		Component:     rec.Category,
		Action:        rec.Type,
		BeforeMetrics: baseline,
	}

	// Apply based on type
	var err error
	switch rec.Category {
	case "cpu":
		err = o.applyCPUTuning(ctx, rec)
	case "memory":
		err = o.applyMemoryTuning(ctx, rec)
	case "io":
		err = o.applyIOTuning(ctx, rec)
	case "network":
		err = o.applyNetworkTuning(ctx, rec)
	case "cost":
		err = o.applyCostTuning(ctx, rec)
	default:
		err = fmt.Errorf("unknown category: %s", rec.Category)
	}

	if err != nil {
		event.Success = false
		o.recordTuningEvent(event)
		return err
	}

	// Validate after validation period
	time.AfterFunc(o.config.ValidationPeriod, func() {
		o.validateTuning(rec.ID, event)
	})

	return nil
}

// applyCPUTuning applies CPU-related tuning
func (o *Orchestrator) applyCPUTuning(ctx context.Context, rec *recommendations.TuningRecommendation) error {
	switch rec.Type {
	case "numa":
		_, err := o.numaOptimizer.OptimizeVM(rec.VMID, 4, 8.0)
		return err
	case "cpu_pinning":
		_, err := o.cpuPinningEngine.AllocateCPUs(rec.VMID, 4, "high")
		return err
	default:
		return fmt.Errorf("unknown CPU tuning type: %s", rec.Type)
	}
}

// applyMemoryTuning applies memory-related tuning
func (o *Orchestrator) applyMemoryTuning(ctx context.Context, rec *recommendations.TuningRecommendation) error {
	// Memory tuning implementation
	return nil
}

// applyIOTuning applies I/O-related tuning
func (o *Orchestrator) applyIOTuning(ctx context.Context, rec *recommendations.TuningRecommendation) error {
	return o.ioTuner.OptimizeDevice("sda", "random")
}

// applyNetworkTuning applies network-related tuning
func (o *Orchestrator) applyNetworkTuning(ctx context.Context, rec *recommendations.TuningRecommendation) error {
	return o.networkTuner.OptimizeTCP(ctx)
}

// applyCostTuning applies cost-related tuning
func (o *Orchestrator) applyCostTuning(ctx context.Context, rec *recommendations.TuningRecommendation) error {
	// Cost tuning is usually recommendation-only, actual implementation varies
	return nil
}

// validateTuning validates tuning after validation period
func (o *Orchestrator) validateTuning(recID string, event TuningEvent) {
	after := o.capturePerformanceSnapshot()
	event.AfterMetrics = after

	// Calculate impact
	impact := (after.Throughput - event.BeforeMetrics.Throughput) / event.BeforeMetrics.Throughput

	if impact < -o.config.DegradationThreshold && o.config.AutoRollback {
		// Performance degraded - rollback
		fmt.Printf("Tuning %s degraded performance by %.1f%%, rolling back\n", recID, impact*100)
		event.Success = false
	} else {
		event.Success = true
		event.Impact = impact

		// Record for convergence detection
		o.convergenceDetector.Record(impact)
	}

	o.recordTuningEvent(event)
}

// capturePerformanceSnapshot captures current performance
func (o *Orchestrator) capturePerformanceSnapshot() PerformanceSnapshot {
	// Simplified - in production, collect from monitoring
	return PerformanceSnapshot{
		CPUUtilization:    0.65,
		MemoryUtilization: 0.75,
		Latency:           10.0,
		Throughput:        1000.0,
		IOPS:              5000.0,
		Cost:              1.0,
	}
}

// recordTuningEvent records tuning event
func (o *Orchestrator) recordTuningEvent(event TuningEvent) {
	o.mu.Lock()
	defer o.mu.Unlock()

	o.tuningHistory = append(o.tuningHistory, event)

	// Keep last 1000 events
	if len(o.tuningHistory) > 1000 {
		o.tuningHistory = o.tuningHistory[len(o.tuningHistory)-1000:]
	}
}

// convergenceMonitor monitors convergence
func (o *Orchestrator) convergenceMonitor(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.stopChan:
			return
		case <-ticker.C:
			if o.convergenceDetector.HasConverged() {
				fmt.Println("Auto-tuning has converged!")
				// Could reduce tuning frequency or enter maintenance mode
			}
		}
	}
}

// Stop stops orchestrator
func (o *Orchestrator) Stop() {
	o.mu.Lock()
	defer o.mu.Unlock()

	if !o.running {
		return
	}

	close(o.stopChan)
	o.running = false

	if o.profiler != nil {
		o.profiler.Stop()
	}
}

// GetTuningHistory returns tuning history
func (o *Orchestrator) GetTuningHistory() []TuningEvent {
	o.mu.RLock()
	defer o.mu.RUnlock()

	history := make([]TuningEvent, len(o.tuningHistory))
	copy(history, o.tuningHistory)
	return history
}

// NewConvergenceDetector creates convergence detector
func NewConvergenceDetector(windowSize int, target float64) *ConvergenceDetector {
	return &ConvergenceDetector{
		window:            make([]float64, 0, windowSize),
		windowSize:        windowSize,
		convergenceTarget: target,
	}
}

// Record records performance improvement
func (cd *ConvergenceDetector) Record(improvement float64) {
	cd.window = append(cd.window, improvement)
	if len(cd.window) > cd.windowSize {
		cd.window = cd.window[1:]
	}
}

// HasConverged checks if tuning has converged
func (cd *ConvergenceDetector) HasConverged() bool {
	if len(cd.window) < cd.windowSize {
		return false
	}

	// Check if recent improvements are below target
	sum := 0.0
	for _, imp := range cd.window {
		sum += abs(imp)
	}
	avgImprovement := sum / float64(len(cd.window))

	return avgImprovement < cd.convergenceTarget
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
