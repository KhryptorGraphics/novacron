// Package chaos - Safety mechanisms and compliance for chaos engineering
package chaos

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"go.uber.org/zap"
)

// SafetyController ensures chaos experiments don't violate safety constraints
type SafetyController struct {
	config       SafetyConfig
	monitors     []SafetyMonitor
	validators   []SafetyValidator
	slaChecker   *SLAChecker
	compliance   *ComplianceManager
	emergencyStop chan struct{}
	logger       *zap.Logger
	mu           sync.RWMutex
	violations   []Violation
	metrics      *SafetyMetrics
}

// SafetyMonitor monitors safety conditions
type SafetyMonitor interface {
	Monitor(ctx context.Context, experiment *ChaosExperiment) error
	GetViolations() []Violation
}

// SafetyValidator validates safety constraints
type SafetyValidator interface {
	Validate(experiment *ChaosExperiment) error
	CheckConstraints(experiment *ChaosExperiment) []string
}

// Violation represents a safety violation
type Violation struct {
	ID           string    `json:"id"`
	ExperimentID string    `json:"experiment_id"`
	Type         string    `json:"type"`
	Severity     string    `json:"severity"` // low, medium, high, critical
	Description  string    `json:"description"`
	Metric       string    `json:"metric"`
	Threshold    float64   `json:"threshold"`
	ActualValue  float64   `json:"actual_value"`
	Timestamp    time.Time `json:"timestamp"`
	AutoResolved bool      `json:"auto_resolved"`
}

// NewSafetyController creates a new safety controller
func NewSafetyController(config SafetyConfig, logger *zap.Logger) *SafetyController {
	sc := &SafetyController{
		config:        config,
		monitors:      []SafetyMonitor{},
		validators:    []SafetyValidator{},
		emergencyStop: make(chan struct{}),
		logger:        logger,
		violations:    []Violation{},
		metrics:       NewSafetyMetrics(),
	}
	
	// Initialize components
	sc.slaChecker = NewSLAChecker(config.SLAThresholds, logger)
	sc.compliance = NewComplianceManager(config.ComplianceMode, logger)
	
	// Register default monitors
	sc.registerDefaultMonitors()
	
	// Register default validators
	sc.registerDefaultValidators()
	
	return sc
}

// Start begins safety monitoring
func (sc *SafetyController) Start(ctx context.Context) error {
	sc.logger.Info("Starting safety controller",
		zap.Bool("auto_rollback", sc.config.AutoRollback),
		zap.Bool("compliance_mode", sc.config.ComplianceMode))
	
	// Start SLA checker
	go sc.slaChecker.Start(ctx)
	
	// Start compliance manager if enabled
	if sc.config.ComplianceMode {
		go sc.compliance.Start(ctx)
	}
	
	// Start violation monitor
	go sc.violationMonitor(ctx)
	
	return nil
}

// Stop stops safety monitoring
func (sc *SafetyController) Stop() {
	close(sc.emergencyStop)
	sc.slaChecker.Stop()
	if sc.config.ComplianceMode {
		sc.compliance.Stop()
	}
}

// CheckViolation checks if experiment violates safety constraints
func (sc *SafetyController) CheckViolation(experiment *ChaosExperiment) bool {
	// Check SLA violations
	if violations := sc.slaChecker.Check(experiment); len(violations) > 0 {
		sc.recordViolations(violations)
		return true
	}
	
	// Check safety validators
	for _, validator := range sc.validators {
		if err := validator.Validate(experiment); err != nil {
			sc.recordViolation(Violation{
				ExperimentID: experiment.ID,
				Type:         "validation_failure",
				Severity:     "high",
				Description:  err.Error(),
				Timestamp:    time.Now(),
			})
			return true
		}
	}
	
	// Check emergency stop
	select {
	case <-sc.emergencyStop:
		return true
	default:
	}
	
	return false
}

// TriggerEmergencyStop triggers emergency stop for all experiments
func (sc *SafetyController) TriggerEmergencyStop(reason string) {
	sc.logger.Error("Emergency stop triggered", zap.String("reason", reason))
	
	select {
	case <-sc.emergencyStop:
		// Already triggered
	default:
		close(sc.emergencyStop)
	}
	
	// Record critical violation
	sc.recordViolation(Violation{
		Type:        "emergency_stop",
		Severity:    "critical",
		Description: reason,
		Timestamp:   time.Now(),
	})
	
	// Notify all channels
	sc.notifyEmergencyStop(reason)
}

// violationMonitor continuously monitors for violations
func (sc *SafetyController) violationMonitor(ctx context.Context) {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			sc.checkAllMonitors()
		}
	}
}

// checkAllMonitors checks all safety monitors
func (sc *SafetyController) checkAllMonitors() {
	for _, monitor := range sc.monitors {
		if violations := monitor.GetViolations(); len(violations) > 0 {
			sc.recordViolations(violations)
			
			// Check if any are critical
			for _, v := range violations {
				if v.Severity == "critical" {
					sc.TriggerEmergencyStop(v.Description)
					return
				}
			}
		}
	}
}

// recordViolation records a safety violation
func (sc *SafetyController) recordViolation(v Violation) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	
	v.ID = generateViolationID()
	sc.violations = append(sc.violations, v)
	
	// Update metrics
	sc.metrics.RecordViolation(v)
	
	// Log violation
	sc.logger.Warn("Safety violation detected",
		zap.String("id", v.ID),
		zap.String("type", v.Type),
		zap.String("severity", v.Severity),
		zap.String("description", v.Description))
	
	// Send notification if configured
	if sc.shouldNotify(v) {
		sc.sendNotification(v)
	}
}

// recordViolations records multiple violations
func (sc *SafetyController) recordViolations(violations []Violation) {
	for _, v := range violations {
		sc.recordViolation(v)
	}
}

// registerDefaultMonitors registers default safety monitors
func (sc *SafetyController) registerDefaultMonitors() {
	sc.monitors = append(sc.monitors,
		NewErrorRateMonitor(sc.logger),
		NewLatencyMonitor(sc.logger),
		NewAvailabilityMonitor(sc.logger),
		NewResourceMonitor(sc.logger),
		NewDataIntegrityMonitor(sc.logger),
	)
}

// registerDefaultValidators registers default safety validators
func (sc *SafetyController) registerDefaultValidators() {
	sc.validators = append(sc.validators,
		NewBlastRadiusValidator(sc.logger),
		NewDurationValidator(sc.logger),
		NewComplianceValidator(sc.logger),
		NewDependencyValidator(sc.logger),
	)
}

// shouldNotify determines if violation should trigger notification
func (sc *SafetyController) shouldNotify(v Violation) bool {
	return v.Severity == "high" || v.Severity == "critical"
}

// sendNotification sends violation notification
func (sc *SafetyController) sendNotification(v Violation) {
	// Send to configured channels
	for _, channel := range sc.config.NotificationChannels {
		// Implementation would send actual notifications
		sc.logger.Info("Sending notification",
			zap.String("channel", channel),
			zap.String("violation_id", v.ID))
	}
}

// notifyEmergencyStop notifies about emergency stop
func (sc *SafetyController) notifyEmergencyStop(reason string) {
	// Send critical alerts to all channels
	for _, channel := range sc.config.NotificationChannels {
		// Implementation would send actual critical alerts
		sc.logger.Info("Sending emergency stop notification",
			zap.String("channel", channel),
			zap.String("reason", reason))
	}
}

// SLAChecker checks SLA compliance
type SLAChecker struct {
	thresholds map[string]float64
	monitors   map[string]MetricMonitor
	logger     *zap.Logger
	mu         sync.RWMutex
}

// MetricMonitor monitors a specific metric
type MetricMonitor interface {
	GetCurrentValue() float64
	GetMetricName() string
}

// NewSLAChecker creates SLA checker
func NewSLAChecker(thresholds map[string]float64, logger *zap.Logger) *SLAChecker {
	return &SLAChecker{
		thresholds: thresholds,
		monitors:   make(map[string]MetricMonitor),
		logger:     logger,
	}
}

// Start begins SLA monitoring
func (s *SLAChecker) Start(ctx context.Context) {
	// Initialize metric monitors
	s.initializeMonitors()
	
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.checkThresholds()
		}
	}
}

// Stop stops SLA monitoring
func (s *SLAChecker) Stop() {
	// Cleanup monitors
}

// Check checks SLA compliance for experiment
func (s *SLAChecker) Check(experiment *ChaosExperiment) []Violation {
	violations := []Violation{}
	
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	for metric, threshold := range experiment.Safety.SLAThresholds {
		monitor, exists := s.monitors[metric]
		if !exists {
			continue
		}
		
		currentValue := monitor.GetCurrentValue()
		
		// Check if threshold is violated
		violated := false
		switch metric {
		case "error_rate", "latency_p99":
			violated = currentValue > threshold
		case "availability":
			violated = currentValue < threshold
		}
		
		if violated {
			violations = append(violations, Violation{
				ExperimentID: experiment.ID,
				Type:         "sla_violation",
				Severity:     "high",
				Description:  fmt.Sprintf("SLA violation: %s", metric),
				Metric:       metric,
				Threshold:    threshold,
				ActualValue:  currentValue,
				Timestamp:    time.Now(),
			})
		}
	}
	
	return violations
}

// checkThresholds checks all SLA thresholds
func (s *SLAChecker) checkThresholds() {
	s.mu.RLock()
	defer s.mu.RUnlock()
	
	for metric, threshold := range s.thresholds {
		monitor, exists := s.monitors[metric]
		if !exists {
			continue
		}
		
		currentValue := monitor.GetCurrentValue()
		
		// Log if close to threshold
		ratio := currentValue / threshold
		if ratio > 0.9 {
			s.logger.Warn("Approaching SLA threshold",
				zap.String("metric", metric),
				zap.Float64("current", currentValue),
				zap.Float64("threshold", threshold),
				zap.Float64("ratio", ratio))
		}
	}
}

// initializeMonitors initializes metric monitors
func (s *SLAChecker) initializeMonitors() {
	// Initialize various metric monitors
	// This would connect to actual monitoring systems
}

// ComplianceManager ensures regulatory compliance
type ComplianceManager struct {
	enabled     bool
	regulations []Regulation
	auditor     *Auditor
	logger      *zap.Logger
	mu          sync.RWMutex
}

// Regulation represents a compliance regulation
type Regulation struct {
	Name         string   `json:"name"`
	Standard     string   `json:"standard"` // GDPR, HIPAA, SOC2, etc.
	Requirements []string `json:"requirements"`
	Restrictions []string `json:"restrictions"`
}

// NewComplianceManager creates compliance manager
func NewComplianceManager(enabled bool, logger *zap.Logger) *ComplianceManager {
	return &ComplianceManager{
		enabled:     enabled,
		regulations: []Regulation{},
		auditor:     NewAuditor(logger),
		logger:      logger,
	}
}

// Start begins compliance monitoring
func (c *ComplianceManager) Start(ctx context.Context) {
	if !c.enabled {
		return
	}
	
	// Load regulations
	c.loadRegulations()
	
	// Start auditor
	c.auditor.Start(ctx)
	
	// Periodic compliance checks
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			c.performComplianceCheck()
		}
	}
}

// Stop stops compliance monitoring
func (c *ComplianceManager) Stop() {
	if c.auditor != nil {
		c.auditor.Stop()
	}
}

// ValidateExperiment validates experiment for compliance
func (c *ComplianceManager) ValidateExperiment(experiment *ChaosExperiment) error {
	if !c.enabled {
		return nil
	}
	
	c.mu.RLock()
	defer c.mu.RUnlock()
	
	for _, reg := range c.regulations {
		// Check restrictions
		for _, restriction := range reg.Restrictions {
			if c.violatesRestriction(experiment, restriction) {
				return fmt.Errorf("compliance violation: %s regulation restricts %s", reg.Name, restriction)
			}
		}
		
		// Check requirements
		for _, requirement := range reg.Requirements {
			if !c.meetsRequirement(experiment, requirement) {
				return fmt.Errorf("compliance requirement not met: %s requires %s", reg.Name, requirement)
			}
		}
	}
	
	// Audit the validation
	c.auditor.LogValidation(experiment)
	
	return nil
}

// loadRegulations loads compliance regulations
func (c *ComplianceManager) loadRegulations() {
	// Load regulations based on configuration
	// This would load from configuration or database
	c.regulations = []Regulation{
		{
			Name:     "GDPR",
			Standard: "GDPR",
			Requirements: []string{
				"data_encryption",
				"audit_logging",
				"data_residency_eu",
			},
			Restrictions: []string{
				"no_production_data_corruption",
				"no_customer_data_exposure",
			},
		},
		{
			Name:     "HIPAA",
			Standard: "HIPAA",
			Requirements: []string{
				"phi_protection",
				"access_controls",
				"audit_trails",
			},
			Restrictions: []string{
				"no_healthcare_data_chaos",
			},
		},
	}
}

// performComplianceCheck performs periodic compliance check
func (c *ComplianceManager) performComplianceCheck() {
	c.logger.Info("Performing compliance check")
	
	// Check all active experiments
	// Generate compliance report
	// Alert on violations
}

// violatesRestriction checks if experiment violates restriction
func (c *ComplianceManager) violatesRestriction(experiment *ChaosExperiment, restriction string) bool {
	// Check specific restrictions
	switch restriction {
	case "no_production_data_corruption":
		return experiment.Type == ChaosDataCorruption && experiment.Tags["environment"] == "production"
	case "no_customer_data_exposure":
		return experiment.Type == ChaosDataCorruption || experiment.Type == ChaosDataLoss
	case "no_healthcare_data_chaos":
		return experiment.Tags["data_type"] == "healthcare"
	default:
		return false
	}
}

// meetsRequirement checks if experiment meets requirement
func (c *ComplianceManager) meetsRequirement(experiment *ChaosExperiment, requirement string) bool {
	// Check specific requirements
	switch requirement {
	case "data_encryption":
		return experiment.Tags["encryption"] == "enabled"
	case "audit_logging":
		return experiment.Safety.AuditLogging
	case "data_residency_eu":
		return experiment.Tags["region"] == "eu"
	case "phi_protection":
		return experiment.Tags["phi_protected"] == "true"
	case "access_controls":
		return experiment.Safety.RequireApproval
	case "audit_trails":
		return experiment.Safety.AuditLogging
	default:
		return true
	}
}

// Auditor handles audit logging for compliance
type Auditor struct {
	logger  *zap.Logger
	entries []AuditEntry
	mu      sync.Mutex
}

// AuditEntry represents an audit log entry
type AuditEntry struct {
	ID           string                 `json:"id"`
	Timestamp    time.Time              `json:"timestamp"`
	Type         string                 `json:"type"`
	User         string                 `json:"user"`
	Action       string                 `json:"action"`
	Resource     string                 `json:"resource"`
	Result       string                 `json:"result"`
	Details      map[string]interface{} `json:"details"`
}

// NewAuditor creates new auditor
func NewAuditor(logger *zap.Logger) *Auditor {
	return &Auditor{
		logger:  logger,
		entries: []AuditEntry{},
	}
}

// Start begins audit logging
func (a *Auditor) Start(ctx context.Context) {
	// Start audit log persistence
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			a.flush()
		}
	}
}

// Stop stops audit logging
func (a *Auditor) Stop() {
	a.flush()
}

// LogValidation logs experiment validation
func (a *Auditor) LogValidation(experiment *ChaosExperiment) {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	entry := AuditEntry{
		ID:        generateAuditID(),
		Timestamp: time.Now(),
		Type:      "experiment_validation",
		Action:    "validate",
		Resource:  experiment.ID,
		Result:    "success",
		Details: map[string]interface{}{
			"experiment_name": experiment.Name,
			"experiment_type": experiment.Type,
			"blast_radius":    experiment.BlastRadius,
		},
	}
	
	a.entries = append(a.entries, entry)
}

// flush persists audit entries
func (a *Auditor) flush() {
	a.mu.Lock()
	defer a.mu.Unlock()
	
	if len(a.entries) == 0 {
		return
	}
	
	// Persist entries to storage
	// This would write to database or file
	
	a.logger.Info("Flushed audit entries", zap.Int("count", len(a.entries)))
	
	// Clear entries
	a.entries = []AuditEntry{}
}

// SafetyMetrics tracks safety-related metrics
type SafetyMetrics struct {
	violationsTotal   *prometheus.CounterVec
	violationDuration *prometheus.HistogramVec
	emergencyStops    prometheus.Counter
	slaViolations     *prometheus.CounterVec
	complianceScore   prometheus.Gauge
}

// NewSafetyMetrics creates safety metrics
func NewSafetyMetrics() *SafetyMetrics {
	return &SafetyMetrics{
		violationsTotal: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "chaos_safety_violations_total",
				Help: "Total number of safety violations",
			},
			[]string{"type", "severity"},
		),
		violationDuration: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Name: "chaos_violation_duration_seconds",
				Help: "Duration of safety violations",
			},
			[]string{"type"},
		),
		emergencyStops: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "chaos_emergency_stops_total",
				Help: "Total number of emergency stops",
			},
		),
		slaViolations: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Name: "chaos_sla_violations_total",
				Help: "Total number of SLA violations",
			},
			[]string{"metric"},
		),
		complianceScore: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "chaos_compliance_score",
				Help: "Current compliance score",
			},
		),
	}
}

// RecordViolation records a safety violation
func (m *SafetyMetrics) RecordViolation(v Violation) {
	m.violationsTotal.WithLabelValues(v.Type, v.Severity).Inc()
	
	if v.Type == "sla_violation" {
		m.slaViolations.WithLabelValues(v.Metric).Inc()
	}
	
	if v.Type == "emergency_stop" {
		m.emergencyStops.Inc()
	}
}

// Additional monitor implementations would follow...