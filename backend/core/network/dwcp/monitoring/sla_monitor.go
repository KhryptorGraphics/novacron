package monitoring

import (
	"sync"
	"time"
)

// SLAMonitor monitors SLA compliance
type SLAMonitor struct {
	mu   sync.RWMutex
	slas map[string]*SLA

	// Compliance tracking
	violations map[string][]*SLAViolation
	reports    map[string]*SLAReport
}

// SLA defines a Service Level Agreement
type SLA struct {
	ID          string
	Name        string
	Description string
	Targets     []*SLATarget
	Window      time.Duration
	Enabled     bool
}

// SLATarget defines a specific SLA target
type SLATarget struct {
	Metric    string
	Operator  string // >=, <=, ==
	Threshold float64
	Type      SLATargetType
}

// SLATargetType defines type of SLA target
type SLATargetType int

const (
	TargetAvailability SLATargetType = iota
	TargetLatency
	TargetThroughput
	TargetErrorRate
)

// SLAViolation represents an SLA violation
type SLAViolation struct {
	SLAID      string
	Target     *SLATarget
	ActualValue float64
	Timestamp   time.Time
	Duration    time.Duration
	Severity    ViolationSeverity
}

// ViolationSeverity defines violation severity
type ViolationSeverity int

const (
	ViolationMinor ViolationSeverity = iota
	ViolationMajor
	ViolationCritical
)

// SLAReport represents SLA compliance report
type SLAReport struct {
	SLAID           string
	Period          Period
	Compliant       bool
	ComplianceRate  float64
	Violations      []*SLAViolation
	ErrorBudget     float64
	ErrorBudgetUsed float64
	Targets         map[string]*TargetCompliance
	GeneratedAt     time.Time
}

// Period represents a time period
type Period struct {
	Start time.Time
	End   time.Time
}

// TargetCompliance represents compliance for a target
type TargetCompliance struct {
	Target         *SLATarget
	ActualValue    float64
	ExpectedValue  float64
	Compliant      bool
	ComplianceRate float64
}

// NewSLAMonitor creates a new SLA monitor
func NewSLAMonitor() *SLAMonitor {
	return &SLAMonitor{
		slas:       make(map[string]*SLA),
		violations: make(map[string][]*SLAViolation),
		reports:    make(map[string]*SLAReport),
	}
}

// DefineSLA defines a new SLA
func (sm *SLAMonitor) DefineSLA(sla *SLA) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	sm.slas[sla.ID] = sla
}

// CheckCompliance checks SLA compliance
func (sm *SLAMonitor) CheckCompliance(slaID string, metrics map[string]float64) bool {
	sm.mu.RLock()
	sla, ok := sm.slas[slaID]
	sm.mu.RUnlock()

	if !ok || !sla.Enabled {
		return true
	}

	compliant := true

	for _, target := range sla.Targets {
		value, ok := metrics[target.Metric]
		if !ok {
			continue
		}

		targetMet := sm.evaluateTarget(target, value)

		if !targetMet {
			compliant = false

			// Record violation
			violation := &SLAViolation{
				SLAID:       slaID,
				Target:      target,
				ActualValue: value,
				Timestamp:   time.Now(),
				Severity:    sm.calculateSeverity(target, value),
			}

			sm.mu.Lock()
			sm.violations[slaID] = append(sm.violations[slaID], violation)
			sm.mu.Unlock()
		}
	}

	return compliant
}

// evaluateTarget evaluates if target is met
func (sm *SLAMonitor) evaluateTarget(target *SLATarget, actualValue float64) bool {
	switch target.Operator {
	case ">=":
		return actualValue >= target.Threshold
	case "<=":
		return actualValue <= target.Threshold
	case "==":
		return actualValue == target.Threshold
	case ">":
		return actualValue > target.Threshold
	case "<":
		return actualValue < target.Threshold
	default:
		return false
	}
}

// calculateSeverity calculates violation severity
func (sm *SLAMonitor) calculateSeverity(target *SLATarget, actualValue float64) ViolationSeverity {
	deviation := (actualValue - target.Threshold) / target.Threshold

	if deviation > 0.2 {
		return ViolationCritical
	} else if deviation > 0.1 {
		return ViolationMajor
	}
	return ViolationMinor
}

// GetSLACompliance retrieves SLA compliance report
func (sm *SLAMonitor) GetSLACompliance(slaID string) (*SLAReport, bool) {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	report, ok := sm.reports[slaID]
	return report, ok
}

// GenerateReport generates SLA compliance report
func (sm *SLAMonitor) GenerateReport(slaID string, period Period) *SLAReport {
	sm.mu.RLock()
	sla, ok := sm.slas[slaID]
	if !ok {
		sm.mu.RUnlock()
		return nil
	}

	violations := sm.getViolationsInPeriod(slaID, period)
	sm.mu.RUnlock()

	report := &SLAReport{
		SLAID:       slaID,
		Period:      period,
		Violations:  violations,
		Targets:     make(map[string]*TargetCompliance),
		GeneratedAt: time.Now(),
	}

	// Calculate compliance rate
	totalTime := period.End.Sub(period.Start)
	var violationTime time.Duration

	for _, v := range violations {
		violationTime += v.Duration
	}

	report.ComplianceRate = 100 * (1 - float64(violationTime)/float64(totalTime))
	report.Compliant = report.ComplianceRate >= 99.9 // 99.9% SLA

	// Calculate error budget
	report.ErrorBudget = 100 - 99.9 // 0.1% error budget
	report.ErrorBudgetUsed = 100 - report.ComplianceRate

	// Per-target compliance
	for _, target := range sla.Targets {
		targetViolations := sm.countTargetViolations(violations, target)
		compliant := targetViolations == 0

		report.Targets[target.Metric] = &TargetCompliance{
			Target:         target,
			Compliant:      compliant,
			ComplianceRate: 100 * (1 - float64(targetViolations)/float64(len(violations)+1)),
		}
	}

	sm.mu.Lock()
	sm.reports[slaID] = report
	sm.mu.Unlock()

	return report
}

// getViolationsInPeriod retrieves violations within a time period
func (sm *SLAMonitor) getViolationsInPeriod(slaID string, period Period) []*SLAViolation {
	var result []*SLAViolation

	violations, ok := sm.violations[slaID]
	if !ok {
		return result
	}

	for _, v := range violations {
		if v.Timestamp.After(period.Start) && v.Timestamp.Before(period.End) {
			result = append(result, v)
		}
	}

	return result
}

// countTargetViolations counts violations for a specific target
func (sm *SLAMonitor) countTargetViolations(violations []*SLAViolation, target *SLATarget) int {
	count := 0
	for _, v := range violations {
		if v.Target.Metric == target.Metric {
			count++
		}
	}
	return count
}

// CalculateErrorBudget calculates remaining error budget
func (sm *SLAMonitor) CalculateErrorBudget(slaID string, period Period) float64 {
	report := sm.GenerateReport(slaID, period)
	if report == nil {
		return 0
	}

	return report.ErrorBudget - report.ErrorBudgetUsed
}

// GetViolations retrieves all violations for an SLA
func (sm *SLAMonitor) GetViolations(slaID string) []*SLAViolation {
	sm.mu.RLock()
	defer sm.mu.RUnlock()

	violations, ok := sm.violations[slaID]
	if !ok {
		return nil
	}

	result := make([]*SLAViolation, len(violations))
	copy(result, violations)
	return result
}

// CreateAvailabilitySLA creates standard availability SLA
func (sm *SLAMonitor) CreateAvailabilitySLA(name string, targetPercent float64) *SLA {
	return &SLA{
		ID:          "sla-availability-" + name,
		Name:        name + " Availability",
		Description: "Uptime availability target",
		Window:      30 * 24 * time.Hour, // 30 days
		Enabled:     true,
		Targets: []*SLATarget{
			{
				Metric:    "availability",
				Operator:  ">=",
				Threshold: targetPercent,
				Type:      TargetAvailability,
			},
		},
	}
}

// CreateLatencySLA creates standard latency SLA
func (sm *SLAMonitor) CreateLatencySLA(name string, maxLatencyMs float64) *SLA {
	return &SLA{
		ID:          "sla-latency-" + name,
		Name:        name + " Latency",
		Description: "Response time target",
		Window:      24 * time.Hour,
		Enabled:     true,
		Targets: []*SLATarget{
			{
				Metric:    "p95_latency",
				Operator:  "<=",
				Threshold: maxLatencyMs,
				Type:      TargetLatency,
			},
		},
	}
}

// CreateThroughputSLA creates standard throughput SLA
func (sm *SLAMonitor) CreateThroughputSLA(name string, minThroughput float64) *SLA {
	return &SLA{
		ID:          "sla-throughput-" + name,
		Name:        name + " Throughput",
		Description: "Minimum throughput target",
		Window:      1 * time.Hour,
		Enabled:     true,
		Targets: []*SLATarget{
			{
				Metric:    "throughput",
				Operator:  ">=",
				Threshold: minThroughput,
				Type:      TargetThroughput,
			},
		},
	}
}
