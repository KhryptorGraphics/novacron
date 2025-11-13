package metrics

import (
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/zeroops"
)

// ZeroOpsMetrics aggregates all zero-ops metrics
type ZeroOpsMetrics struct {
	mu                     sync.RWMutex
	humanInterventions     int64
	totalDecisions         int64
	automatedDecisions     int64
	automationSuccesses    int64
	automationFailures     int64
	mttd                   []time.Duration
	mttr                   []time.Duration
	costSavings            float64
	availability           float64
	changeAttempts         int64
	changeSuccesses        int64
	falseAlerts            int64
	totalAlerts            int64
}

// NewZeroOpsMetrics creates new zero-ops metrics
func NewZeroOpsMetrics() *ZeroOpsMetrics {
	return &ZeroOpsMetrics{
		availability: 0.99999, // Start at 99.999%
	}
}

// RecordDecision records an automated decision
func (zom *ZeroOpsMetrics) RecordDecision(automated bool, success bool) {
	zom.mu.Lock()
	defer zom.mu.Unlock()

	zom.totalDecisions++

	if automated {
		zom.automatedDecisions++
		if success {
			zom.automationSuccesses++
		} else {
			zom.automationFailures++
		}
	} else {
		zom.humanInterventions++
	}
}

// RecordMTTD records mean time to detect
func (zom *ZeroOpsMetrics) RecordMTTD(d time.Duration) {
	zom.mu.Lock()
	defer zom.mu.Unlock()
	zom.mttd = append(zom.mttd, d)
}

// RecordMTTR records mean time to resolve
func (zom *ZeroOpsMetrics) RecordMTTR(d time.Duration) {
	zom.mu.Lock()
	defer zom.mu.Unlock()
	zom.mttr = append(zom.mttr, d)
}

// RecordCostSavings records cost savings
func (zom *ZeroOpsMetrics) RecordCostSavings(amount float64) {
	zom.mu.Lock()
	defer zom.mu.Unlock()
	zom.costSavings += amount
}

// UpdateAvailability updates system availability
func (zom *ZeroOpsMetrics) UpdateAvailability(availability float64) {
	zom.mu.Lock()
	defer zom.mu.Unlock()
	zom.availability = availability
}

// RecordChange records a change attempt
func (zom *ZeroOpsMetrics) RecordChange(success bool) {
	zom.mu.Lock()
	defer zom.mu.Unlock()
	zom.changeAttempts++
	if success {
		zom.changeSuccesses++
	}
}

// RecordAlert records an alert
func (zom *ZeroOpsMetrics) RecordAlert(falsePositive bool) {
	zom.mu.Lock()
	defer zom.mu.Unlock()
	zom.totalAlerts++
	if falsePositive {
		zom.falseAlerts++
	}
}

// GetMetrics returns current automation metrics
func (zom *ZeroOpsMetrics) GetMetrics() *zeroops.AutomationMetrics {
	zom.mu.RLock()
	defer zom.mu.RUnlock()

	humanRate := float64(zom.humanInterventions) / float64(zom.totalDecisions)
	automationRate := float64(zom.automatedDecisions) / float64(zom.totalDecisions)
	successRate := float64(zom.automationSuccesses) / float64(zom.automatedDecisions)
	avgMTTD := calculateAverage(zom.mttd)
	avgMTTR := calculateAverage(zom.mttr)
	changeSuccessRate := float64(zom.changeSuccesses) / float64(zom.changeAttempts)
	falseAlertRate := float64(zom.falseAlerts) / float64(zom.totalAlerts)

	return &zeroops.AutomationMetrics{
		Timestamp:             time.Now(),
		HumanInterventionRate: humanRate,
		AutomationSuccessRate: successRate,
		AverageMTTD:           avgMTTD,
		AverageMTTR:           avgMTTR,
		CostOptimizationSavings: zom.costSavings,
		Availability:          zom.availability,
		ChangeSuccessRate:     changeSuccessRate,
		FalseAlertRate:        falseAlertRate,
		TotalDecisions:        zom.totalDecisions,
		AutomatedDecisions:    zom.automatedDecisions,
		ManualDecisions:       zom.humanInterventions,
	}
}

// GetDashboardData returns dashboard-friendly data
func (zom *ZeroOpsMetrics) GetDashboardData() *DashboardData {
	metrics := zom.GetMetrics()

	return &DashboardData{
		AutomationPercentage: (1.0 - metrics.HumanInterventionRate) * 100,
		SuccessRate:         metrics.AutomationSuccessRate * 100,
		MTTD:                metrics.AverageMTTD,
		MTTR:                metrics.AverageMTTR,
		CostSavings:         metrics.CostOptimizationSavings,
		Availability:        metrics.Availability * 100,
		ChangeSuccessRate:   metrics.ChangeSuccessRate * 100,
		FalseAlertRate:      metrics.FalseAlertRate * 100,
		Status:              zom.calculateStatus(metrics),
		Targets:             zom.getTargets(),
	}
}

// calculateStatus calculates overall system status
func (zom *ZeroOpsMetrics) calculateStatus(metrics *zeroops.AutomationMetrics) string {
	// Check if metrics meet targets
	if metrics.HumanInterventionRate > 0.001 {
		return "warning" // >0.1% human intervention
	}
	if metrics.AutomationSuccessRate < 0.999 {
		return "warning" // <99.9% success
	}
	if metrics.AverageMTTD > 10 {
		return "warning" // >10s MTTD
	}
	if metrics.AverageMTTR > 60 {
		return "warning" // >60s MTTR
	}
	if metrics.Availability < 0.99999 {
		return "warning" // <99.999% availability
	}

	return "healthy"
}

// getTargets returns target metrics
func (zom *ZeroOpsMetrics) getTargets() *TargetMetrics {
	return &TargetMetrics{
		HumanInterventionRate: 0.001,  // <0.1%
		AutomationSuccessRate: 0.999,  // >99.9%
		MTTD:                  10.0,   // <10s
		MTTR:                  60.0,   // <60s
		Availability:          0.99999, // 99.999%
		FalseAlertRate:        0.0001, // <0.01%
	}
}

// DashboardData contains dashboard metrics
type DashboardData struct {
	AutomationPercentage float64         `json:"automation_percentage"`
	SuccessRate          float64         `json:"success_rate"`
	MTTD                 float64         `json:"mttd_seconds"`
	MTTR                 float64         `json:"mttr_seconds"`
	CostSavings          float64         `json:"cost_savings"`
	Availability         float64         `json:"availability_percentage"`
	ChangeSuccessRate    float64         `json:"change_success_rate"`
	FalseAlertRate       float64         `json:"false_alert_rate"`
	Status               string          `json:"status"`
	Targets              *TargetMetrics  `json:"targets"`
}

// TargetMetrics contains target values
type TargetMetrics struct {
	HumanInterventionRate float64 `json:"human_intervention_rate"`
	AutomationSuccessRate float64 `json:"automation_success_rate"`
	MTTD                  float64 `json:"mttd_seconds"`
	MTTR                  float64 `json:"mttr_seconds"`
	Availability          float64 `json:"availability"`
	FalseAlertRate        float64 `json:"false_alert_rate"`
}

func calculateAverage(durations []time.Duration) float64 {
	if len(durations) == 0 {
		return 0
	}
	var sum time.Duration
	for _, d := range durations {
		sum += d
	}
	return float64(sum) / float64(len(durations)) / float64(time.Second)
}
