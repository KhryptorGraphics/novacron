package metrics

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// EvolutionMetrics tracks autonomous evolution metrics
type EvolutionMetrics struct {
	// Healing metrics
	HealingSuccessRate     prometheus.Gauge
	HealingAttempts        prometheus.Counter
	HealingSuccesses       prometheus.Counter
	HealingFailures        prometheus.Counter
	HealingDuration        prometheus.Histogram
	FaultDetectionTime     prometheus.Histogram

	// Prediction metrics
	PredictionAccuracy     prometheus.Gauge
	PredictionsGenerated   prometheus.Counter
	PredictionsCorrect     prometheus.Counter
	PredictionHorizon      prometheus.Gauge
	AnomalyScores          prometheus.Histogram

	// Code generation metrics
	CodeQualityScore       prometheus.Gauge
	CodeGenAttempts        prometheus.Counter
	CodeGenSuccesses       prometheus.Counter
	DeploymentSuccess      prometheus.Counter
	LinesOfCodeGenerated   prometheus.Counter

	// Architecture evolution metrics
	ArchitectureFitness    prometheus.Gauge
	EvolutionGenerations   prometheus.Counter
	MutationRate           prometheus.Gauge
	ConvergenceSpeed       prometheus.Histogram
	ImprovementRate        prometheus.Gauge

	// Learning metrics
	LearningProgress       prometheus.Gauge
	RewardAverage          prometheus.Gauge
	ExplorationRate        prometheus.Gauge
	KnowledgeTransfers     prometheus.Counter
	ModelAccuracy          prometheus.Gauge

	// Human intervention metrics
	HumanInterventions     prometheus.Counter
	AutoResolutionRate     prometheus.Gauge
	EscalationRate         prometheus.Gauge
	MTTD                   prometheus.Gauge
	MTTR                   prometheus.Gauge

	// System metrics
	SystemAvailability     prometheus.Gauge
	PerformanceScore       prometheus.Gauge
	CostEfficiency         prometheus.Gauge
	ResourceUtilization    prometheus.Gauge

	// Digital twin metrics
	SimulationAccuracy     prometheus.Gauge
	PredictionConfidence   prometheus.Gauge
	ScenarioAnalyses       prometheus.Counter
	OptimalPathsFound      prometheus.Counter

	// A/B testing metrics
	ExperimentsRunning     prometheus.Gauge
	ExperimentsCompleted   prometheus.Counter
	SignificantResults     prometheus.Counter
	AutoDeployments        prometheus.Counter

	// Self-optimization metrics
	OptimizationCycles     prometheus.Counter
	ConfigurationChanges   prometheus.Counter
	PerformanceImprovement prometheus.Gauge
	ParetoEfficiency       prometheus.Gauge

	mu                     sync.RWMutex
	startTime              time.Time
	totalHealing           int64
	successfulHealing      int64
}

// NewEvolutionMetrics creates new evolution metrics
func NewEvolutionMetrics() *EvolutionMetrics {
	return &EvolutionMetrics{
		startTime: time.Now(),

		// Healing metrics
		HealingSuccessRate: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_healing_success_rate",
			Help: "Success rate of autonomous healing actions",
		}),
		HealingAttempts: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_healing_attempts_total",
			Help: "Total number of healing attempts",
		}),
		HealingSuccesses: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_healing_successes_total",
			Help: "Total number of successful healings",
		}),
		HealingFailures: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_healing_failures_total",
			Help: "Total number of failed healings",
		}),
		HealingDuration: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "autonomous_healing_duration_seconds",
			Help:    "Duration of healing actions",
			Buckets: []float64{0.1, 0.5, 1, 5, 10, 30, 60},
		}),
		FaultDetectionTime: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "autonomous_fault_detection_seconds",
			Help:    "Time to detect faults",
			Buckets: []float64{0.001, 0.01, 0.1, 0.5, 1.0},
		}),

		// Prediction metrics
		PredictionAccuracy: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_prediction_accuracy",
			Help: "Accuracy of failure predictions",
		}),
		PredictionsGenerated: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_predictions_generated_total",
			Help: "Total number of predictions generated",
		}),
		PredictionsCorrect: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_predictions_correct_total",
			Help: "Total number of correct predictions",
		}),
		PredictionHorizon: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_prediction_horizon_hours",
			Help: "Prediction horizon in hours",
		}),
		AnomalyScores: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "autonomous_anomaly_scores",
			Help:    "Distribution of anomaly scores",
			Buckets: []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		}),

		// Code generation metrics
		CodeQualityScore: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_code_quality_score",
			Help: "Quality score of generated code",
		}),
		CodeGenAttempts: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_codegen_attempts_total",
			Help: "Total code generation attempts",
		}),
		CodeGenSuccesses: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_codegen_successes_total",
			Help: "Successful code generations",
		}),
		DeploymentSuccess: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_deployment_successes_total",
			Help: "Successful autonomous deployments",
		}),
		LinesOfCodeGenerated: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_lines_generated_total",
			Help: "Total lines of code generated",
		}),

		// Architecture evolution metrics
		ArchitectureFitness: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_architecture_fitness",
			Help: "Fitness score of evolved architecture",
		}),
		EvolutionGenerations: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_evolution_generations_total",
			Help: "Total evolution generations",
		}),
		MutationRate: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_mutation_rate",
			Help: "Current mutation rate",
		}),
		ConvergenceSpeed: promauto.NewHistogram(prometheus.HistogramOpts{
			Name:    "autonomous_convergence_generations",
			Help:    "Generations to convergence",
			Buckets: []float64{10, 50, 100, 200, 500, 1000},
		}),
		ImprovementRate: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_improvement_rate_percent",
			Help: "Architecture improvement rate percentage",
		}),

		// Learning metrics
		LearningProgress: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_learning_progress",
			Help: "Learning progress indicator",
		}),
		RewardAverage: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_reward_average",
			Help: "Average reward from reinforcement learning",
		}),
		ExplorationRate: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_exploration_rate",
			Help: "Current exploration rate",
		}),
		KnowledgeTransfers: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_knowledge_transfers_total",
			Help: "Total knowledge transfers",
		}),
		ModelAccuracy: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_model_accuracy",
			Help: "ML model accuracy",
		}),

		// Human intervention metrics
		HumanInterventions: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_human_interventions_total",
			Help: "Total human interventions required",
		}),
		AutoResolutionRate: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_auto_resolution_rate",
			Help: "Rate of automatic incident resolution",
		}),
		EscalationRate: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_escalation_rate",
			Help: "Rate of incident escalation",
		}),
		MTTD: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_mttd_seconds",
			Help: "Mean time to detection in seconds",
		}),
		MTTR: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_mttr_seconds",
			Help: "Mean time to resolution in seconds",
		}),

		// System metrics
		SystemAvailability: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_system_availability",
			Help: "System availability percentage",
		}),
		PerformanceScore: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_performance_score",
			Help: "Overall performance score",
		}),
		CostEfficiency: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_cost_efficiency",
			Help: "Cost efficiency ratio",
		}),
		ResourceUtilization: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_resource_utilization",
			Help: "Resource utilization percentage",
		}),

		// Digital twin metrics
		SimulationAccuracy: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_simulation_accuracy",
			Help: "Digital twin simulation accuracy",
		}),
		PredictionConfidence: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_prediction_confidence",
			Help: "Prediction confidence level",
		}),
		ScenarioAnalyses: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_scenario_analyses_total",
			Help: "Total scenario analyses performed",
		}),
		OptimalPathsFound: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_optimal_paths_found_total",
			Help: "Total optimal paths found",
		}),

		// A/B testing metrics
		ExperimentsRunning: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_experiments_running",
			Help: "Number of experiments currently running",
		}),
		ExperimentsCompleted: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_experiments_completed_total",
			Help: "Total experiments completed",
		}),
		SignificantResults: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_significant_results_total",
			Help: "Experiments with significant results",
		}),
		AutoDeployments: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_auto_deployments_total",
			Help: "Total automatic deployments from experiments",
		}),

		// Self-optimization metrics
		OptimizationCycles: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_optimization_cycles_total",
			Help: "Total optimization cycles completed",
		}),
		ConfigurationChanges: promauto.NewCounter(prometheus.CounterOpts{
			Name: "autonomous_configuration_changes_total",
			Help: "Total configuration changes applied",
		}),
		PerformanceImprovement: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_performance_improvement_percent",
			Help: "Performance improvement percentage",
		}),
		ParetoEfficiency: promauto.NewGauge(prometheus.GaugeOpts{
			Name: "autonomous_pareto_efficiency",
			Help: "Pareto efficiency score",
		}),
	}
}

// RecordHealing records a healing event
func (em *EvolutionMetrics) RecordHealing(event *HealingEvent) {
	em.mu.Lock()
	defer em.mu.Unlock()

	em.totalHealing++
	em.HealingAttempts.Inc()

	if event.Success {
		em.successfulHealing++
		em.HealingSuccesses.Inc()
	} else {
		em.HealingFailures.Inc()
	}

	// Update success rate
	if em.totalHealing > 0 {
		successRate := float64(em.successfulHealing) / float64(em.totalHealing)
		em.HealingSuccessRate.Set(successRate)
	}

	// Record duration
	em.HealingDuration.Observe(event.Duration.Seconds())
}

// RecordDetectionTime records fault detection time
func (em *EvolutionMetrics) RecordDetectionTime(duration time.Duration) {
	em.FaultDetectionTime.Observe(duration.Seconds())
}

// UpdatePredictionAccuracy updates prediction accuracy
func (em *EvolutionMetrics) UpdatePredictionAccuracy(accuracy float64) {
	em.PredictionAccuracy.Set(accuracy)
}

// RecordPrediction records a prediction event
func (em *EvolutionMetrics) RecordPrediction(correct bool) {
	em.PredictionsGenerated.Inc()
	if correct {
		em.PredictionsCorrect.Inc()
	}
}

// UpdateCodeQuality updates code quality score
func (em *EvolutionMetrics) UpdateCodeQuality(score float64) {
	em.CodeQualityScore.Set(score)
}

// RecordCodeGeneration records code generation event
func (em *EvolutionMetrics) RecordCodeGeneration(success bool, lines int) {
	em.CodeGenAttempts.Inc()
	if success {
		em.CodeGenSuccesses.Inc()
		em.LinesOfCodeGenerated.Add(float64(lines))
	}
}

// UpdateArchitectureFitness updates architecture fitness
func (em *EvolutionMetrics) UpdateArchitectureFitness(fitness float64) {
	em.ArchitectureFitness.Set(fitness)
}

// RecordEvolutionGeneration records an evolution generation
func (em *EvolutionMetrics) RecordEvolutionGeneration(improvement float64) {
	em.EvolutionGenerations.Inc()
	em.ImprovementRate.Set(improvement)
}

// UpdateLearningMetrics updates learning metrics
func (em *EvolutionMetrics) UpdateLearningMetrics(progress, reward, exploration float64) {
	em.LearningProgress.Set(progress)
	em.RewardAverage.Set(reward)
	em.ExplorationRate.Set(exploration)
}

// RecordIncident records incident metrics
func (em *EvolutionMetrics) RecordIncident(autoResolved, escalated bool, mttd, mttr time.Duration) {
	if !autoResolved {
		em.HumanInterventions.Inc()
	}

	em.MTTD.Set(mttd.Seconds())
	em.MTTR.Set(mttr.Seconds())
}

// UpdateSystemMetrics updates system-wide metrics
func (em *EvolutionMetrics) UpdateSystemMetrics(availability, performance, cost, utilization float64) {
	em.SystemAvailability.Set(availability)
	em.PerformanceScore.Set(performance)
	em.CostEfficiency.Set(cost)
	em.ResourceUtilization.Set(utilization)
}

// UpdateDigitalTwinMetrics updates digital twin metrics
func (em *EvolutionMetrics) UpdateDigitalTwinMetrics(accuracy, confidence float64) {
	em.SimulationAccuracy.Set(accuracy)
	em.PredictionConfidence.Set(confidence)
}

// RecordScenarioAnalysis records scenario analysis
func (em *EvolutionMetrics) RecordScenarioAnalysis() {
	em.ScenarioAnalyses.Inc()
}

// RecordOptimalPath records finding optimal path
func (em *EvolutionMetrics) RecordOptimalPath() {
	em.OptimalPathsFound.Inc()
}

// UpdateExperimentMetrics updates A/B testing metrics
func (em *EvolutionMetrics) UpdateExperimentMetrics(running int, significant bool) {
	em.ExperimentsRunning.Set(float64(running))
	em.ExperimentsCompleted.Inc()
	if significant {
		em.SignificantResults.Inc()
	}
}

// RecordAutoDeployment records automatic deployment
func (em *EvolutionMetrics) RecordAutoDeployment() {
	em.AutoDeployments.Inc()
}

// RecordOptimizationCycle records optimization cycle
func (em *EvolutionMetrics) RecordOptimizationCycle(improvement float64) {
	em.OptimizationCycles.Inc()
	em.PerformanceImprovement.Set(improvement)
}

// RecordConfigurationChange records configuration change
func (em *EvolutionMetrics) RecordConfigurationChange() {
	em.ConfigurationChanges.Inc()
}

// UpdateParetoEfficiency updates Pareto efficiency
func (em *EvolutionMetrics) UpdateParetoEfficiency(efficiency float64) {
	em.ParetoEfficiency.Set(efficiency)
}

// GetSummary returns metrics summary
func (em *EvolutionMetrics) GetSummary() *MetricsSummary {
	em.mu.RLock()
	defer em.mu.RUnlock()

	uptime := time.Since(em.startTime)
	healingRate := float64(0)
	if em.totalHealing > 0 {
		healingRate = float64(em.successfulHealing) / float64(em.totalHealing) * 100
	}

	return &MetricsSummary{
		Uptime:             uptime,
		HealingSuccessRate: healingRate,
		TotalHealings:      em.totalHealing,
		SuccessfulHealings: em.successfulHealing,
	}
}

// GetCPUUsage returns current CPU usage (mock)
func (em *EvolutionMetrics) GetCPUUsage() float64 {
	return 0.65 // Mock value
}

// GetMemoryUsage returns current memory usage (mock)
func (em *EvolutionMetrics) GetMemoryUsage() float64 {
	return 0.72 // Mock value
}

// GetDiskUsage returns current disk usage (mock)
func (em *EvolutionMetrics) GetDiskUsage() float64 {
	return 0.45 // Mock value
}

// GetNetworkLoad returns current network load (mock)
func (em *EvolutionMetrics) GetNetworkLoad() float64 {
	return 0.38 // Mock value
}

// Supporting types

type HealingEvent struct {
	ID       string
	Success  bool
	Duration time.Duration
}

type MetricsSummary struct {
	Uptime             time.Duration
	HealingSuccessRate float64
	TotalHealings      int64
	SuccessfulHealings int64
}

// HealingMetrics for the healing engine
type HealingMetrics struct {
	*EvolutionMetrics
}

// NewHealingMetrics creates new healing metrics
func NewHealingMetrics() *HealingMetrics {
	return &HealingMetrics{
		EvolutionMetrics: NewEvolutionMetrics(),
	}
}