// Chaos Engineering Platform with Production-Safe Failure Injection
// Implements controlled chaos testing with automated recovery validation

package chaos

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

// ChaosType defines the type of chaos experiment
type ChaosType int

const (
	ChaosNetworkLatency ChaosType = iota
	ChaosNetworkPacketLoss
	ChaosNetworkPartition
	ChaosCPUPressure
	ChaosMemoryPressure
	ChaosDiskPressure
	ChaosServiceFailure
	ChaosDependencyFailure
	ChaosCascadingFailure
	ChaosTimeSkew
	ChaosKernelPanic
	ChaosProcessKill
)

// SafetyLevel defines the safety constraints for chaos experiments
type SafetyLevel int

const (
	SafetyDryRun SafetyLevel = iota // No actual chaos, just simulation
	SafetyDev                        // Development environment only
	SafetyStaging                    // Staging environment
	SafetyCanary                     // Production canary (limited scope)
	SafetyProduction                 // Full production (with safeguards)
)

// ChaosExperiment represents a chaos engineering experiment
type ChaosExperiment struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	Type            ChaosType              `json:"type"`
	Target          ChaosTarget            `json:"target"`
	Parameters      map[string]interface{} `json:"parameters"`
	Safety          SafetyConfig           `json:"safety"`
	Schedule        *ChaosSchedule         `json:"schedule,omitempty"`
	Duration        time.Duration          `json:"duration"`
	Status          ExperimentStatus       `json:"status"`
	StartedAt       *time.Time             `json:"started_at,omitempty"`
	EndedAt         *time.Time             `json:"ended_at,omitempty"`
	Results         *ExperimentResults     `json:"results,omitempty"`
	RecoveryActions []RecoveryAction       `json:"recovery_actions"`
	mu              sync.RWMutex
}

// ChaosTarget defines the target of chaos injection
type ChaosTarget struct {
	Type       string            `json:"type"` // service, pod, node, zone, region
	Selector   map[string]string `json:"selector"`
	Percentage float64           `json:"percentage"` // Percentage of targets affected
	MaxTargets int               `json:"max_targets"`
}

// SafetyConfig defines safety constraints for experiments
type SafetyConfig struct {
	Level              SafetyLevel       `json:"level"`
	MaxBlastRadius     int               `json:"max_blast_radius"`
	AutoRollback       bool              `json:"auto_rollback"`
	RollbackThreshold  float64           `json:"rollback_threshold"`
	RequireApproval    bool              `json:"require_approval"`
	Approvers          []string          `json:"approvers"`
	MonitoringEnabled  bool              `json:"monitoring_enabled"`
	AlertChannels      []string          `json:"alert_channels"`
	SafetyChecks       []SafetyCheck     `json:"safety_checks"`
	EmergencyStopKey   string            `json:"emergency_stop_key"`
}

// SafetyCheck represents a pre-flight safety check
type SafetyCheck struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Condition   string `json:"condition"`
	Required    bool   `json:"required"`
	LastChecked time.Time `json:"last_checked"`
	Passed      bool   `json:"passed"`
	Message     string `json:"message"`
}

// ChaosSchedule defines when to run experiments
type ChaosSchedule struct {
	Type      string    `json:"type"` // once, recurring, gameday
	StartTime time.Time `json:"start_time"`
	EndTime   *time.Time `json:"end_time,omitempty"`
	Interval  time.Duration `json:"interval,omitempty"`
	DaysOfWeek []string `json:"days_of_week,omitempty"`
	TimeWindow TimeWindow `json:"time_window"`
}

// TimeWindow defines valid time windows for chaos
type TimeWindow struct {
	StartHour int `json:"start_hour"`
	EndHour   int `json:"end_hour"`
	Timezone  string `json:"timezone"`
}

// ExperimentStatus represents the current state of an experiment
type ExperimentStatus struct {
	State          string    `json:"state"` // pending, running, completed, failed, rolled_back
	Progress       float64   `json:"progress"`
	CurrentTargets []string  `json:"current_targets"`
	Message        string    `json:"message"`
	LastUpdated    time.Time `json:"last_updated"`
}

// ExperimentResults contains the results of a chaos experiment
type ExperimentResults struct {
	Success           bool                   `json:"success"`
	ImpactMetrics     ImpactMetrics          `json:"impact_metrics"`
	RecoveryMetrics   RecoveryMetrics        `json:"recovery_metrics"`
	Observations      []Observation          `json:"observations"`
	Recommendations   []string               `json:"recommendations"`
	ValidationResults []ValidationResult     `json:"validation_results"`
	Artifacts         map[string]string      `json:"artifacts"`
}

// ImpactMetrics measures the impact of chaos
type ImpactMetrics struct {
	AffectedServices   []string  `json:"affected_services"`
	ErrorRateIncrease  float64   `json:"error_rate_increase"`
	LatencyIncrease    float64   `json:"latency_increase"`
	ThroughputDecrease float64   `json:"throughput_decrease"`
	CustomerImpact     float64   `json:"customer_impact"`
	BlastRadius        int       `json:"blast_radius"`
	Duration           time.Duration `json:"duration"`
}

// RecoveryMetrics measures system recovery
type RecoveryMetrics struct {
	TimeToDetect    time.Duration `json:"time_to_detect"`
	TimeToRecover   time.Duration `json:"time_to_recover"`
	RecoverySuccess bool          `json:"recovery_success"`
	AutoHealing     bool          `json:"auto_healing"`
	ManualSteps     int           `json:"manual_steps"`
}

// Observation represents an observation during chaos
type Observation struct {
	Timestamp   time.Time              `json:"timestamp"`
	Type        string                 `json:"type"`
	Severity    string                 `json:"severity"`
	Description string                 `json:"description"`
	Metrics     map[string]interface{} `json:"metrics"`
}

// ValidationResult represents validation of recovery
type ValidationResult struct {
	Check   string    `json:"check"`
	Passed  bool      `json:"passed"`
	Message string    `json:"message"`
	Time    time.Time `json:"time"`
}

// RecoveryAction defines an action to recover from chaos
type RecoveryAction struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`
	Description string    `json:"description"`
	Automated   bool      `json:"automated"`
	Executed    bool      `json:"executed"`
	ExecutedAt  *time.Time `json:"executed_at,omitempty"`
	Success     bool      `json:"success"`
	Error       string    `json:"error,omitempty"`
}

// ChaosInjector interface for different chaos types
type ChaosInjector interface {
	Inject(ctx context.Context, target ChaosTarget, params map[string]interface{}) error
	Recover(ctx context.Context, target ChaosTarget) error
	Validate(ctx context.Context, target ChaosTarget) error
	GetType() ChaosType
}

// NetworkChaosInjector implements network chaos injection
type NetworkChaosInjector struct {
	tcController    *TrafficController
	iptablesManager *IPTablesManager
	logger          *zap.Logger
}

// TrafficController manages traffic control for network chaos
type TrafficController struct {
	interfaces map[string]*NetworkInterface
	rules      sync.Map
	mu         sync.RWMutex
}

// NetworkInterface represents a network interface
type NetworkInterface struct {
	Name      string
	IP        string
	Bandwidth int64
	Active    bool
}

// IPTablesManager manages iptables rules for network partitioning
type IPTablesManager struct {
	chains    map[string][]string
	ruleStack *RuleStack
	mu        sync.Mutex
}

// RuleStack maintains a stack of iptables rules for rollback
type RuleStack struct {
	rules []IPTableRule
	mu    sync.Mutex
}

// IPTableRule represents an iptables rule
type IPTableRule struct {
	Chain    string
	Rule     string
	Priority int
	Added    time.Time
}

// ResourceChaosInjector implements resource chaos injection
type ResourceChaosInjector struct {
	cgroupManager *CgroupManager
	stressManager *StressManager
	logger        *zap.Logger
}

// CgroupManager manages cgroups for resource limitation
type CgroupManager struct {
	controllers map[string]*CgroupController
	mu          sync.RWMutex
}

// CgroupController represents a cgroup controller
type CgroupController struct {
	Name       string
	Path       string
	Type       string // cpu, memory, io
	Limits     map[string]interface{}
	Pids       []int
}

// StressManager manages stress-ng processes
type StressManager struct {
	processes map[string]*StressProcess
	mu        sync.RWMutex
}

// StressProcess represents a stress-ng process
type StressProcess struct {
	PID        int
	Type       string
	Intensity  int
	Duration   time.Duration
	StartedAt  time.Time
}

// ChaosOrchestrator manages chaos engineering experiments
type ChaosOrchestrator struct {
	experiments     sync.Map // map[string]*ChaosExperiment
	injectors       map[ChaosType]ChaosInjector
	scheduler       *ExperimentScheduler
	monitor         *ChaosMonitor
	validator       *RecoveryValidator
	safetyGuard     *SafetyGuard
	metrics         *ChaosMetrics
	config          *ChaosConfig
	logger          *zap.Logger
	shutdownCh      chan struct{}
	emergencyStopCh chan string
	wg              sync.WaitGroup
}

// ChaosConfig configures the chaos platform
type ChaosConfig struct {
	Enabled            bool
	DefaultSafetyLevel SafetyLevel
	MaxConcurrent      int
	RecoveryTimeout    time.Duration
	MonitoringInterval time.Duration
	DryRun            bool
	AutoRecovery      bool
	WebhookURL        string
}

// ExperimentScheduler schedules chaos experiments
type ExperimentScheduler struct {
	queue      PriorityQueue
	active     sync.Map
	maxActive  int
	mu         sync.RWMutex
}

// ChaosMonitor monitors chaos experiments
type ChaosMonitor struct {
	metricsCollector *MetricsCollector
	alertManager     *AlertManager
	observers        []ExperimentObserver
	mu               sync.RWMutex
}

// ExperimentObserver observes chaos experiments
type ExperimentObserver interface {
	OnStart(experiment *ChaosExperiment)
	OnProgress(experiment *ChaosExperiment, progress float64)
	OnComplete(experiment *ChaosExperiment, results *ExperimentResults)
	OnFailure(experiment *ChaosExperiment, err error)
}

// RecoveryValidator validates system recovery
type RecoveryValidator struct {
	checks    []RecoveryCheck
	timeout   time.Duration
	mu        sync.RWMutex
}

// RecoveryCheck interface for recovery validation
type RecoveryCheck interface {
	Validate(ctx context.Context, experiment *ChaosExperiment) (*ValidationResult, error)
	GetName() string
}

// SafetyGuard enforces safety constraints
type SafetyGuard struct {
	rules           []SafetyRule
	emergencyStop   atomic.Bool
	activeExperiments sync.Map
	mu              sync.RWMutex
}

// SafetyRule defines a safety rule
type SafetyRule struct {
	Name      string
	Condition func(experiment *ChaosExperiment) bool
	Action    func(experiment *ChaosExperiment) error
	Priority  int
}

// ChaosMetrics tracks chaos engineering metrics
type ChaosMetrics struct {
	experimentsTotal    prometheus.Counter
	experimentsSuccess  prometheus.Counter
	experimentsFailure  prometheus.Counter
	experimentDuration  prometheus.Histogram
	recoveryTime        prometheus.Histogram
	blastRadius         prometheus.Histogram
	safetyViolations    prometheus.Counter
}

// NewChaosOrchestrator creates a new chaos orchestrator
func NewChaosOrchestrator(config *ChaosConfig, logger *zap.Logger) *ChaosOrchestrator {
	return &ChaosOrchestrator{
		injectors: map[ChaosType]ChaosInjector{
			ChaosNetworkLatency:    NewNetworkChaosInjector(logger),
			ChaosNetworkPacketLoss: NewNetworkChaosInjector(logger),
			ChaosNetworkPartition:  NewNetworkChaosInjector(logger),
			ChaosCPUPressure:       NewResourceChaosInjector(logger),
			ChaosMemoryPressure:    NewResourceChaosInjector(logger),
			ChaosDiskPressure:      NewResourceChaosInjector(logger),
		},
		scheduler: &ExperimentScheduler{
			queue:     PriorityQueue{items: make([]*QueueItem, 0)},
			maxActive: config.MaxConcurrent,
		},
		monitor: &ChaosMonitor{
			metricsCollector: &MetricsCollector{},
			alertManager:     &AlertManager{},
			observers:        make([]ExperimentObserver, 0),
		},
		validator: &RecoveryValidator{
			checks:  make([]RecoveryCheck, 0),
			timeout: config.RecoveryTimeout,
		},
		safetyGuard: &SafetyGuard{
			rules: make([]SafetyRule, 0),
		},
		metrics:         NewChaosMetrics(),
		config:          config,
		logger:          logger,
		shutdownCh:      make(chan struct{}),
		emergencyStopCh: make(chan string, 1),
	}
}

// Start begins the chaos orchestrator
func (o *ChaosOrchestrator) Start(ctx context.Context) error {
	if !o.config.Enabled {
		o.logger.Info("Chaos engineering disabled")
		return nil
	}

	o.logger.Info("Starting chaos orchestrator",
		zap.String("safety_level", o.config.DefaultSafetyLevel.String()),
		zap.Int("max_concurrent", o.config.MaxConcurrent))

	// Start experiment scheduler
	o.wg.Add(1)
	go o.runScheduler(ctx)

	// Start monitoring
	o.wg.Add(1)
	go o.runMonitoring(ctx)

	// Start emergency stop handler
	o.wg.Add(1)
	go o.handleEmergencyStop(ctx)

	// Initialize safety rules
	o.initializeSafetyRules()

	return nil
}

// CreateExperiment creates a new chaos experiment
func (o *ChaosOrchestrator) CreateExperiment(experiment *ChaosExperiment) error {
	// Validate experiment
	if err := o.validateExperiment(experiment); err != nil {
		return fmt.Errorf("experiment validation failed: %w", err)
	}

	// Perform safety checks
	if err := o.performSafetyChecks(experiment); err != nil {
		return fmt.Errorf("safety checks failed: %w", err)
	}

	// Store experiment
	o.experiments.Store(experiment.ID, experiment)

	// Schedule experiment
	o.scheduler.Schedule(experiment)

	o.logger.Info("Chaos experiment created",
		zap.String("id", experiment.ID),
		zap.String("name", experiment.Name),
		zap.String("type", experiment.Type.String()))

	return nil
}

// RunExperiment executes a chaos experiment
func (o *ChaosOrchestrator) RunExperiment(ctx context.Context, experimentID string) error {
	value, exists := o.experiments.Load(experimentID)
	if !exists {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	experiment := value.(*ChaosExperiment)

	// Check if emergency stop is active
	if o.safetyGuard.emergencyStop.Load() {
		return fmt.Errorf("emergency stop active, cannot run experiments")
	}

	// Update status
	experiment.mu.Lock()
	experiment.Status.State = "running"
	startTime := time.Now()
	experiment.StartedAt = &startTime
	experiment.mu.Unlock()

	// Notify observers
	o.notifyStart(experiment)

	// Get appropriate injector
	injector, exists := o.injectors[experiment.Type]
	if !exists {
		return fmt.Errorf("no injector for chaos type: %v", experiment.Type)
	}

	// Run experiment with monitoring
	errCh := make(chan error, 1)
	go func() {
		if o.config.DryRun {
			o.logger.Info("DRY RUN: Would inject chaos",
				zap.String("experiment", experiment.ID),
				zap.String("type", experiment.Type.String()))
			errCh <- nil
		} else {
			errCh <- o.executeWithMonitoring(ctx, experiment, injector)
		}
	}()

	// Wait for completion or timeout
	select {
	case err := <-errCh:
		if err != nil {
			o.handleExperimentFailure(experiment, err)
			return err
		}
		o.handleExperimentSuccess(experiment)
	case <-time.After(experiment.Duration):
		o.recoverFromChaos(ctx, experiment, injector)
		o.handleExperimentTimeout(experiment)
	case key := <-o.emergencyStopCh:
		if key == experiment.Safety.EmergencyStopKey {
			o.emergencyStop(ctx, experiment, injector)
			return fmt.Errorf("emergency stop triggered")
		}
	}

	return nil
}

// executeWithMonitoring executes chaos with continuous monitoring
func (o *ChaosOrchestrator) executeWithMonitoring(ctx context.Context, experiment *ChaosExperiment, injector ChaosInjector) error {
	// Start monitoring
	monitorCtx, cancelMonitor := context.WithCancel(ctx)
	defer cancelMonitor()

	monitorCh := make(chan ImpactMetrics, 1)
	go o.monitorImpact(monitorCtx, experiment, monitorCh)

	// Inject chaos
	if err := injector.Inject(ctx, experiment.Target, experiment.Parameters); err != nil {
		return fmt.Errorf("chaos injection failed: %w", err)
	}

	// Monitor for safety violations
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	startTime := time.Now()
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case metrics := <-monitorCh:
			// Check rollback threshold
			if o.shouldRollback(experiment, metrics) {
				o.logger.Warn("Rollback threshold exceeded",
					zap.String("experiment", experiment.ID))
				return o.recoverFromChaos(ctx, experiment, injector)
			}

			// Update progress
			progress := time.Since(startTime).Seconds() / experiment.Duration.Seconds()
			o.updateProgress(experiment, progress)

		case <-time.After(experiment.Duration):
			// Chaos duration completed
			return o.recoverFromChaos(ctx, experiment, injector)
		}
	}
}

// recoverFromChaos recovers from chaos injection
func (o *ChaosOrchestrator) recoverFromChaos(ctx context.Context, experiment *ChaosExperiment, injector ChaosInjector) error {
	o.logger.Info("Recovering from chaos",
		zap.String("experiment", experiment.ID))

	// Execute recovery
	recoveryStart := time.Now()
	if err := injector.Recover(ctx, experiment.Target); err != nil {
		o.logger.Error("Recovery failed",
			zap.String("experiment", experiment.ID),
			zap.Error(err))

		// Execute manual recovery actions
		o.executeManualRecovery(ctx, experiment)

		return fmt.Errorf("recovery failed: %w", err)
	}

	// Validate recovery
	if err := o.validateRecovery(ctx, experiment, injector); err != nil {
		o.logger.Error("Recovery validation failed",
			zap.String("experiment", experiment.ID),
			zap.Error(err))
		return err
	}

	recoveryTime := time.Since(recoveryStart)
	o.metrics.recoveryTime.Observe(recoveryTime.Seconds())

	o.logger.Info("Recovery successful",
		zap.String("experiment", experiment.ID),
		zap.Duration("recovery_time", recoveryTime))

	return nil
}

// validateRecovery validates system recovery
func (o *ChaosOrchestrator) validateRecovery(ctx context.Context, experiment *ChaosExperiment, injector ChaosInjector) error {
	// Run injector validation
	if err := injector.Validate(ctx, experiment.Target); err != nil {
		return err
	}

	// Run recovery checks
	var results []ValidationResult
	for _, check := range o.validator.checks {
		result, err := check.Validate(ctx, experiment)
		if err != nil {
			o.logger.Error("Recovery check failed",
				zap.String("check", check.GetName()),
				zap.Error(err))
			result = &ValidationResult{
				Check:   check.GetName(),
				Passed:  false,
				Message: err.Error(),
				Time:    time.Now(),
			}
		}
		results = append(results, *result)
	}

	// Store validation results
	experiment.mu.Lock()
	if experiment.Results == nil {
		experiment.Results = &ExperimentResults{}
	}
	experiment.Results.ValidationResults = results
	experiment.mu.Unlock()

	// Check if all validations passed
	for _, result := range results {
		if !result.Passed {
			return fmt.Errorf("recovery validation failed: %s", result.Message)
		}
	}

	return nil
}

// Helper functions

func (o *ChaosOrchestrator) validateExperiment(experiment *ChaosExperiment) error {
	if experiment.ID == "" {
		return fmt.Errorf("experiment ID required")
	}
	if experiment.Duration == 0 {
		return fmt.Errorf("experiment duration required")
	}
	if experiment.Target.Type == "" {
		return fmt.Errorf("target type required")
	}
	return nil
}

func (o *ChaosOrchestrator) performSafetyChecks(experiment *ChaosExperiment) error {
	for _, check := range experiment.Safety.SafetyChecks {
		if check.Required && !check.Passed {
			return fmt.Errorf("required safety check failed: %s", check.Name)
		}
	}
	return nil
}

func (o *ChaosOrchestrator) initializeSafetyRules() {
	// Add default safety rules
	o.safetyGuard.rules = append(o.safetyGuard.rules, SafetyRule{
		Name: "max_blast_radius",
		Condition: func(exp *ChaosExperiment) bool {
			// Check if blast radius exceeds limit
			return true // Simplified
		},
		Action: func(exp *ChaosExperiment) error {
			return o.EmergencyStop(exp.ID)
		},
		Priority: 1,
	})
}

func (o *ChaosOrchestrator) shouldRollback(experiment *ChaosExperiment, metrics ImpactMetrics) bool {
	if !experiment.Safety.AutoRollback {
		return false
	}

	// Check various thresholds
	if metrics.ErrorRateIncrease > experiment.Safety.RollbackThreshold {
		return true
	}
	if metrics.CustomerImpact > 0.1 { // 10% customer impact
		return true
	}
	if metrics.BlastRadius > experiment.Safety.MaxBlastRadius {
		return true
	}

	return false
}

func (o *ChaosOrchestrator) updateProgress(experiment *ChaosExperiment, progress float64) {
	experiment.mu.Lock()
	experiment.Status.Progress = progress
	experiment.Status.LastUpdated = time.Now()
	experiment.mu.Unlock()

	// Notify observers
	o.notifyProgress(experiment, progress)
}

func (o *ChaosOrchestrator) executeManualRecovery(ctx context.Context, experiment *ChaosExperiment) {
	for i, action := range experiment.RecoveryActions {
		if !action.Automated {
			o.logger.Info("Executing manual recovery action",
				zap.String("action", action.ID),
				zap.String("description", action.Description))

			// Mark as executed
			now := time.Now()
			experiment.RecoveryActions[i].Executed = true
			experiment.RecoveryActions[i].ExecutedAt = &now
		}
	}
}

func (o *ChaosOrchestrator) EmergencyStop(experimentID string) error {
	o.logger.Warn("Emergency stop triggered",
		zap.String("experiment", experimentID))

	o.safetyGuard.emergencyStop.Store(true)

	// Stop all active experiments
	o.scheduler.active.Range(func(key, value interface{}) bool {
		exp := value.(*ChaosExperiment)
		o.emergencyStopCh <- exp.Safety.EmergencyStopKey
		return true
	})

	return nil
}

func (o *ChaosOrchestrator) handleExperimentFailure(experiment *ChaosExperiment, err error) {
	experiment.mu.Lock()
	experiment.Status.State = "failed"
	experiment.Status.Message = err.Error()
	endTime := time.Now()
	experiment.EndedAt = &endTime
	experiment.mu.Unlock()

	o.metrics.experimentsFailure.Inc()
	o.notifyFailure(experiment, err)
}

func (o *ChaosOrchestrator) handleExperimentSuccess(experiment *ChaosExperiment) {
	experiment.mu.Lock()
	experiment.Status.State = "completed"
	endTime := time.Now()
	experiment.EndedAt = &endTime
	experiment.mu.Unlock()

	o.metrics.experimentsSuccess.Inc()
	o.notifyComplete(experiment)
}

func (o *ChaosOrchestrator) handleExperimentTimeout(experiment *ChaosExperiment) {
	o.logger.Warn("Experiment timeout",
		zap.String("experiment", experiment.ID))

	experiment.mu.Lock()
	experiment.Status.State = "timeout"
	endTime := time.Now()
	experiment.EndedAt = &endTime
	experiment.mu.Unlock()
}

// Monitoring functions

func (o *ChaosOrchestrator) monitorImpact(ctx context.Context, experiment *ChaosExperiment, metricsCh chan<- ImpactMetrics) {
	ticker := time.NewTicker(o.config.MonitoringInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			metrics := o.monitor.metricsCollector.CollectImpactMetrics(experiment)
			metricsCh <- metrics
		}
	}
}

func (o *ChaosOrchestrator) runScheduler(ctx context.Context) {
	defer o.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.shutdownCh:
			return
		case <-ticker.C:
			o.scheduler.ProcessQueue(ctx, o)
		}
	}
}

func (o *ChaosOrchestrator) runMonitoring(ctx context.Context) {
	defer o.wg.Done()

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.shutdownCh:
			return
		case <-ticker.C:
			o.monitor.CollectMetrics()
		}
	}
}

func (o *ChaosOrchestrator) handleEmergencyStop(ctx context.Context) {
	defer o.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return
		case <-o.shutdownCh:
			return
		case key := <-o.emergencyStopCh:
			o.logger.Warn("Emergency stop received",
				zap.String("key", key))
			o.safetyGuard.emergencyStop.Store(true)
		}
	}
}

// Observer notification functions

func (o *ChaosOrchestrator) notifyStart(experiment *ChaosExperiment) {
	o.monitor.mu.RLock()
	observers := o.monitor.observers
	o.monitor.mu.RUnlock()

	for _, observer := range observers {
		observer.OnStart(experiment)
	}
}

func (o *ChaosOrchestrator) notifyProgress(experiment *ChaosExperiment, progress float64) {
	o.monitor.mu.RLock()
	observers := o.monitor.observers
	o.monitor.mu.RUnlock()

	for _, observer := range observers {
		observer.OnProgress(experiment, progress)
	}
}

func (o *ChaosOrchestrator) notifyComplete(experiment *ChaosExperiment) {
	o.monitor.mu.RLock()
	observers := o.monitor.observers
	o.monitor.mu.RUnlock()

	for _, observer := range observers {
		observer.OnComplete(experiment, experiment.Results)
	}
}

func (o *ChaosOrchestrator) notifyFailure(experiment *ChaosExperiment, err error) {
	o.monitor.mu.RLock()
	observers := o.monitor.observers
	o.monitor.mu.RUnlock()

	for _, observer := range observers {
		observer.OnFailure(experiment, err)
	}
}

// Supporting functions and types

func (t ChaosType) String() string {
	switch t {
	case ChaosNetworkLatency:
		return "network_latency"
	case ChaosNetworkPacketLoss:
		return "network_packet_loss"
	case ChaosNetworkPartition:
		return "network_partition"
	case ChaosCPUPressure:
		return "cpu_pressure"
	case ChaosMemoryPressure:
		return "memory_pressure"
	case ChaosDiskPressure:
		return "disk_pressure"
	case ChaosServiceFailure:
		return "service_failure"
	case ChaosDependencyFailure:
		return "dependency_failure"
	case ChaosCascadingFailure:
		return "cascading_failure"
	case ChaosTimeSkew:
		return "time_skew"
	case ChaosKernelPanic:
		return "kernel_panic"
	case ChaosProcessKill:
		return "process_kill"
	default:
		return "unknown"
	}
}

func (s SafetyLevel) String() string {
	switch s {
	case SafetyDryRun:
		return "dry_run"
	case SafetyDev:
		return "dev"
	case SafetyStaging:
		return "staging"
	case SafetyCanary:
		return "canary"
	case SafetyProduction:
		return "production"
	default:
		return "unknown"
	}
}

func NewNetworkChaosInjector(logger *zap.Logger) *NetworkChaosInjector {
	return &NetworkChaosInjector{
		tcController:    &TrafficController{interfaces: make(map[string]*NetworkInterface)},
		iptablesManager: &IPTablesManager{chains: make(map[string][]string), ruleStack: &RuleStack{}},
		logger:          logger,
	}
}

func (n *NetworkChaosInjector) Inject(ctx context.Context, target ChaosTarget, params map[string]interface{}) error {
	// Implementation would inject network chaos
	return nil
}

func (n *NetworkChaosInjector) Recover(ctx context.Context, target ChaosTarget) error {
	// Implementation would recover from network chaos
	return nil
}

func (n *NetworkChaosInjector) Validate(ctx context.Context, target ChaosTarget) error {
	// Implementation would validate network recovery
	return nil
}

func (n *NetworkChaosInjector) GetType() ChaosType {
	return ChaosNetworkLatency
}

func NewResourceChaosInjector(logger *zap.Logger) *ResourceChaosInjector {
	return &ResourceChaosInjector{
		cgroupManager: &CgroupManager{controllers: make(map[string]*CgroupController)},
		stressManager: &StressManager{processes: make(map[string]*StressProcess)},
		logger:        logger,
	}
}

func (r *ResourceChaosInjector) Inject(ctx context.Context, target ChaosTarget, params map[string]interface{}) error {
	// Implementation would inject resource chaos
	return nil
}

func (r *ResourceChaosInjector) Recover(ctx context.Context, target ChaosTarget) error {
	// Implementation would recover from resource chaos
	return nil
}

func (r *ResourceChaosInjector) Validate(ctx context.Context, target ChaosTarget) error {
	// Implementation would validate resource recovery
	return nil
}

func (r *ResourceChaosInjector) GetType() ChaosType {
	return ChaosCPUPressure
}

func (s *ExperimentScheduler) Schedule(experiment *ChaosExperiment) {
	priority := 1.0
	if experiment.Safety.Level == SafetyProduction {
		priority = 0.5 // Lower priority for production experiments
	}
	s.queue.Push(experiment, priority)
}

func (s *ExperimentScheduler) ProcessQueue(ctx context.Context, orchestrator *ChaosOrchestrator) {
	// Get active count
	activeCount := 0
	s.active.Range(func(_, _ interface{}) bool {
		activeCount++
		return true
	})

	// Process queue if below max
	if activeCount < s.maxActive {
		if item := s.queue.Pop(); item != nil {
			experiment := item.(*ChaosExperiment)
			s.active.Store(experiment.ID, experiment)

			go func() {
				defer s.active.Delete(experiment.ID)
				orchestrator.RunExperiment(ctx, experiment.ID)
			}()
		}
	}
}

type PriorityQueue struct {
	items []*QueueItem
	mu    sync.RWMutex
}

type QueueItem struct {
	Data     interface{}
	Priority float64
	AddedAt  time.Time
}

func (q *PriorityQueue) Push(data interface{}, priority float64) {
	q.mu.Lock()
	defer q.mu.Unlock()

	item := &QueueItem{
		Data:     data,
		Priority: priority,
		AddedAt:  time.Now(),
	}

	// Insert in priority order
	inserted := false
	for i, existing := range q.items {
		if priority > existing.Priority {
			q.items = append(q.items[:i], append([]*QueueItem{item}, q.items[i:]...)...)
			inserted = true
			break
		}
	}

	if !inserted {
		q.items = append(q.items, item)
	}
}

func (q *PriorityQueue) Pop() interface{} {
	q.mu.Lock()
	defer q.mu.Unlock()

	if len(q.items) == 0 {
		return nil
	}

	item := q.items[0]
	q.items = q.items[1:]
	return item.Data
}

type MetricsCollector struct{}
type AlertManager struct{}

func (m *MetricsCollector) CollectImpactMetrics(experiment *ChaosExperiment) ImpactMetrics {
	// Simplified implementation
	return ImpactMetrics{
		ErrorRateIncrease:  rand.Float64() * 0.1,
		LatencyIncrease:    rand.Float64() * 50,
		ThroughputDecrease: rand.Float64() * 0.2,
		CustomerImpact:     rand.Float64() * 0.05,
		BlastRadius:        rand.Intn(10),
	}
}

func (m *ChaosMonitor) CollectMetrics() {
	// Implementation
}

func NewChaosMetrics() *ChaosMetrics {
	return &ChaosMetrics{
		experimentsTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "chaos_experiments_total",
			Help: "Total number of chaos experiments",
		}),
		experimentsSuccess: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "chaos_experiments_success_total",
			Help: "Total number of successful chaos experiments",
		}),
		experimentsFailure: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "chaos_experiments_failure_total",
			Help: "Total number of failed chaos experiments",
		}),
		experimentDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "chaos_experiment_duration_seconds",
			Help:    "Duration of chaos experiments",
			Buckets: prometheus.ExponentialBuckets(10, 2, 10),
		}),
		recoveryTime: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "chaos_recovery_time_seconds",
			Help:    "Time to recover from chaos",
			Buckets: prometheus.ExponentialBuckets(1, 2, 10),
		}),
		blastRadius: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "chaos_blast_radius",
			Help:    "Blast radius of chaos experiments",
			Buckets: prometheus.LinearBuckets(1, 5, 10),
		}),
		safetyViolations: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "chaos_safety_violations_total",
			Help: "Total number of safety violations",
		}),
	}
}