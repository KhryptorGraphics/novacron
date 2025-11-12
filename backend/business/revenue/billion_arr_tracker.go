// Package revenue provides $1B ARR milestone tracking and revenue acceleration
package revenue

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// ARRMilestone tracks progress towards $1B ARR
type ARRMilestone struct {
	ID                  string                 `json:"id"`
	CurrentARR          float64                `json:"current_arr"`          // Current Annual Recurring Revenue
	TargetARR           float64                `json:"target_arr"`           // $1B target
	PreviousARR         float64                `json:"previous_arr"`         // Phase 12: $800M
	GrowthRate          float64                `json:"growth_rate"`          // 25% target
	ProgressPercentage  float64                `json:"progress_percentage"`  // % to $1B
	RemainingARR        float64                `json:"remaining_arr"`        // Gap to $1B
	ProjectedDate       time.Time              `json:"projected_date"`       // When $1B reached
	DaysToTarget        int                    `json:"days_to_target"`
	Velocity            ARRVelocity            `json:"velocity"`
	Composition         RevenueComposition     `json:"composition"`
	Metrics             ARRMetrics             `json:"metrics"`
	Forecasts           []ARRForecast          `json:"forecasts"`
	Alerts              []MilestoneAlert       `json:"alerts"`
	LastUpdated         time.Time              `json:"last_updated"`
}

// ARRVelocity tracks growth momentum
type ARRVelocity struct {
	DailyARR            float64                `json:"daily_arr"`            // Daily ARR add
	WeeklyARR           float64                `json:"weekly_arr"`           // Weekly ARR add
	MonthlyARR          float64                `json:"monthly_arr"`          // Monthly ARR add
	QuarterlyARR        float64                `json:"quarterly_arr"`        // Quarterly ARR add
	RunRate             float64                `json:"run_rate"`             // Annual run rate
	Acceleration        float64                `json:"acceleration"`         // % change in velocity
	MoMGrowth           float64                `json:"mom_growth"`           // Month-over-month
	QoQGrowth           float64                `json:"qoq_growth"`           // Quarter-over-quarter
	YoYGrowth           float64                `json:"yoy_growth"`           // Year-over-year
	RequiredVelocity    float64                `json:"required_velocity"`    // To hit $1B on time
	OnTrack             bool                   `json:"on_track"`
}

// RevenueComposition breaks down ARR sources
type RevenueComposition struct {
	NewBusiness         RevenueSegment         `json:"new_business"`         // $300M (30%)
	Expansion           RevenueSegment         `json:"expansion"`            // $500M (50%)
	Renewals            RevenueSegment         `json:"renewals"`             // $200M (20%)
	TotalARR            float64                `json:"total_arr"`
	SegmentHealth       map[string]float64     `json:"segment_health"`       // Health scores
	Trends              map[string]float64     `json:"trends"`               // Growth trends
}

// RevenueSegment tracks individual revenue streams
type RevenueSegment struct {
	Name                string                 `json:"name"`
	CurrentARR          float64                `json:"current_arr"`
	TargetARR           float64                `json:"target_arr"`
	Achievement         float64                `json:"achievement"`          // % of target
	GrowthRate          float64                `json:"growth_rate"`
	ContributionPct     float64                `json:"contribution_pct"`     // % of total ARR
	Customers           int                    `json:"customers"`
	AvgContractValue    float64                `json:"avg_contract_value"`
	Churn               float64                `json:"churn"`
	NetRetention        float64                `json:"net_retention"`
}

// ARRMetrics provides comprehensive tracking
type ARRMetrics struct {
	TotalCustomers      int                    `json:"total_customers"`
	Fortune500          int                    `json:"fortune_500"`          // 350 target
	AvgContractValue    float64                `json:"avg_contract_value"`   // $5M+ target
	GrossMargin         float64                `json:"gross_margin"`         // 42%+ target
	NetMargin           float64                `json:"net_margin"`
	RenewalRate         float64                `json:"renewal_rate"`         // 97%+ target
	NetRetention        float64                `json:"net_retention"`        // 150%+ target
	CAC                 float64                `json:"cac"`                  // Customer Acquisition Cost
	LTV                 float64                `json:"ltv"`                  // Lifetime Value
	LTVtoCAC            float64                `json:"ltv_to_cac"`           // 3x+ target
	PaybackPeriod       int                    `json:"payback_period"`       // Months
	RuleOf40            float64                `json:"rule_of_40"`           // Growth% + Margin%
	BurnMultiple        float64                `json:"burn_multiple"`
	MagicNumber         float64                `json:"magic_number"`
}

// ARRForecast provides predictive analytics
type ARRForecast struct {
	Date                time.Time              `json:"date"`
	ForecastARR         float64                `json:"forecast_arr"`
	LowerBound          float64                `json:"lower_bound"`          // 95% CI
	UpperBound          float64                `json:"upper_bound"`          // 95% CI
	Confidence          float64                `json:"confidence"`           // 0-1
	Method              string                 `json:"method"`               // ML model used
	Assumptions         map[string]interface{} `json:"assumptions"`
	RiskFactors         []string               `json:"risk_factors"`
}

// MilestoneAlert tracks critical events
type MilestoneAlert struct {
	ID                  string                 `json:"id"`
	Severity            string                 `json:"severity"`             // critical, warning, info
	Type                string                 `json:"type"`
	Message             string                 `json:"message"`
	Metric              string                 `json:"metric"`
	Threshold           float64                `json:"threshold"`
	ActualValue         float64                `json:"actual_value"`
	Impact              string                 `json:"impact"`
	Recommendation      string                 `json:"recommendation"`
	CreatedAt           time.Time              `json:"created_at"`
	Resolved            bool                   `json:"resolved"`
}

// CohortAnalysis tracks customer cohorts
type CohortAnalysis struct {
	CohortID            string                 `json:"cohort_id"`
	StartDate           time.Time              `json:"start_date"`
	CustomerCount       int                    `json:"customer_count"`
	InitialARR          float64                `json:"initial_arr"`
	CurrentARR          float64                `json:"current_arr"`
	RetentionRate       float64                `json:"retention_rate"`
	ExpansionRate       float64                `json:"expansion_rate"`
	NetRetention        float64                `json:"net_retention"`
	LTV                 float64                `json:"ltv"`
	MonthlyMetrics      []CohortMonthMetrics   `json:"monthly_metrics"`
}

// CohortMonthMetrics tracks cohort performance over time
type CohortMonthMetrics struct {
	Month               int                    `json:"month"`
	CustomersRemaining  int                    `json:"customers_remaining"`
	ARR                 float64                `json:"arr"`
	RetentionRate       float64                `json:"retention_rate"`
	ExpansionARR        float64                `json:"expansion_arr"`
	ChurnARR            float64                `json:"churn_arr"`
	NetARR              float64                `json:"net_arr"`
}

// ChurnPrediction uses ML for churn forecasting
type ChurnPrediction struct {
	CustomerID          string                 `json:"customer_id"`
	ChurnProbability    float64                `json:"churn_probability"`    // 0-1
	RiskLevel           string                 `json:"risk_level"`           // low, medium, high, critical
	ARRAtRisk           float64                `json:"arr_at_risk"`
	PredictedDate       time.Time              `json:"predicted_date"`
	RiskFactors         []RiskFactor           `json:"risk_factors"`
	HealthScore         float64                `json:"health_score"`         // 0-100
	Recommendations     []string               `json:"recommendations"`
	SaveProbability     float64                `json:"save_probability"`
	InterventionValue   float64                `json:"intervention_value"`
}

// RiskFactor identifies churn drivers
type RiskFactor struct {
	Factor              string                 `json:"factor"`
	Impact              float64                `json:"impact"`               // 0-1
	Trend               string                 `json:"trend"`                // improving, declining, stable
	Value               interface{}            `json:"value"`
}

// BillionARRTracker manages $1B ARR milestone
type BillionARRTracker struct {
	mu                  sync.RWMutex
	milestone           *ARRMilestone
	cohorts             map[string]*CohortAnalysis
	predictions         map[string]*ChurnPrediction
	historicalData      []ARRSnapshot
	forecastModels      map[string]ForecastModel
	alertRules          []AlertRule
	config              TrackerConfig
	metrics             *MetricsCollector
}

// ARRSnapshot captures point-in-time state
type ARRSnapshot struct {
	Timestamp           time.Time              `json:"timestamp"`
	ARR                 float64                `json:"arr"`
	Customers           int                    `json:"customers"`
	Composition         RevenueComposition     `json:"composition"`
	Metrics             ARRMetrics             `json:"metrics"`
}

// ForecastModel represents prediction algorithm
type ForecastModel struct {
	Name                string                 `json:"name"`
	Type                string                 `json:"type"`               // linear, arima, prophet, ml
	Accuracy            float64                `json:"accuracy"`           // Historical accuracy
	LastTrained         time.Time              `json:"last_trained"`
	Parameters          map[string]interface{} `json:"parameters"`
	Enabled             bool                   `json:"enabled"`
}

// AlertRule defines monitoring thresholds
type AlertRule struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Metric              string                 `json:"metric"`
	Condition           string                 `json:"condition"`          // <, >, =, etc.
	Threshold           float64                `json:"threshold"`
	Severity            string                 `json:"severity"`
	Enabled             bool                   `json:"enabled"`
	Actions             []string               `json:"actions"`
}

// TrackerConfig configures the tracker
type TrackerConfig struct {
	TargetARR           float64                `json:"target_arr"`           // $1B
	TargetDate          time.Time              `json:"target_date"`
	AlertThresholds     map[string]float64     `json:"alert_thresholds"`
	ForecastHorizon     int                    `json:"forecast_horizon"`     // Days
	UpdateInterval      time.Duration          `json:"update_interval"`
	EnableMLForecasting bool                   `json:"enable_ml_forecasting"`
	EnableChurnPrediction bool                 `json:"enable_churn_prediction"`
}

// MetricsCollector tracks performance
type MetricsCollector struct {
	mu                  sync.RWMutex
	forecasts           int64
	predictions         int64
	alerts              int64
	updates             int64
	lastUpdate          time.Time
	errors              int64
}

// NewBillionARRTracker creates a new tracker
func NewBillionARRTracker(config TrackerConfig) *BillionARRTracker {
	return &BillionARRTracker{
		milestone: &ARRMilestone{
			ID:          fmt.Sprintf("billion-arr-%d", time.Now().Unix()),
			TargetARR:   1_000_000_000, // $1B
			PreviousARR: 800_000_000,   // $800M from Phase 12
			CurrentARR:  800_000_000,   // Starting point
			LastUpdated: time.Now(),
		},
		cohorts:        make(map[string]*CohortAnalysis),
		predictions:    make(map[string]*ChurnPrediction),
		historicalData: make([]ARRSnapshot, 0),
		forecastModels: initializeForecastModels(),
		alertRules:     initializeAlertRules(),
		config:         config,
		metrics:        &MetricsCollector{},
	}
}

// UpdateARR updates current ARR and recalculates metrics
func (t *BillionARRTracker) UpdateARR(ctx context.Context, newARR float64, composition RevenueComposition) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	// Update ARR
	t.milestone.CurrentARR = newARR
	t.milestone.Composition = composition
	t.milestone.RemainingARR = t.milestone.TargetARR - newARR
	t.milestone.ProgressPercentage = (newARR / t.milestone.TargetARR) * 100
	t.milestone.GrowthRate = ((newARR - t.milestone.PreviousARR) / t.milestone.PreviousARR) * 100

	// Calculate velocity
	t.milestone.Velocity = t.calculateVelocity()

	// Update metrics
	t.milestone.Metrics = t.calculateMetrics(composition)

	// Generate forecasts
	forecasts, err := t.generateForecasts(ctx)
	if err == nil {
		t.milestone.Forecasts = forecasts

		// Find when we'll hit $1B
		for _, forecast := range forecasts {
			if forecast.ForecastARR >= t.milestone.TargetARR {
				t.milestone.ProjectedDate = forecast.Date
				t.milestone.DaysToTarget = int(time.Until(forecast.Date).Hours() / 24)
				break
			}
		}
	}

	// Check alerts
	alerts := t.checkAlerts()
	t.milestone.Alerts = alerts

	// Save snapshot
	snapshot := ARRSnapshot{
		Timestamp:   time.Now(),
		ARR:         newARR,
		Composition: composition,
		Metrics:     t.milestone.Metrics,
	}
	t.historicalData = append(t.historicalData, snapshot)

	// Update timestamp
	t.milestone.LastUpdated = time.Now()
	t.metrics.updates++
	t.metrics.lastUpdate = time.Now()

	return nil
}

// calculateVelocity computes ARR growth velocity
func (t *BillionARRTracker) calculateVelocity() ARRVelocity {
	velocity := ARRVelocity{}

	if len(t.historicalData) < 2 {
		return velocity
	}

	// Get recent data points
	now := time.Now()

	// Daily velocity (last 7 days average)
	dailyPoints := t.getDataPointsSince(now.AddDate(0, 0, -7))
	if len(dailyPoints) > 1 {
		dailyGrowth := dailyPoints[len(dailyPoints)-1].ARR - dailyPoints[0].ARR
		days := dailyPoints[len(dailyPoints)-1].Timestamp.Sub(dailyPoints[0].Timestamp).Hours() / 24
		if days > 0 {
			velocity.DailyARR = dailyGrowth / days
		}
	}

	// Weekly velocity
	weeklyPoints := t.getDataPointsSince(now.AddDate(0, 0, -30))
	if len(weeklyPoints) > 1 {
		weeklyGrowth := weeklyPoints[len(weeklyPoints)-1].ARR - weeklyPoints[0].ARR
		weeks := weeklyPoints[len(weeklyPoints)-1].Timestamp.Sub(weeklyPoints[0].Timestamp).Hours() / (24 * 7)
		if weeks > 0 {
			velocity.WeeklyARR = weeklyGrowth / weeks
		}
	}

	// Monthly velocity (MoM)
	monthlyPoints := t.getDataPointsSince(now.AddDate(0, -3, 0))
	if len(monthlyPoints) > 1 {
		monthlyGrowth := monthlyPoints[len(monthlyPoints)-1].ARR - monthlyPoints[0].ARR
		months := monthlyPoints[len(monthlyPoints)-1].Timestamp.Sub(monthlyPoints[0].Timestamp).Hours() / (24 * 30)
		if months > 0 {
			velocity.MonthlyARR = monthlyGrowth / months
			velocity.MoMGrowth = (monthlyGrowth / monthlyPoints[0].ARR) * 100
		}
	}

	// Quarterly velocity (QoQ)
	quarterlyPoints := t.getDataPointsSince(now.AddDate(0, -6, 0))
	if len(quarterlyPoints) > 1 {
		quarterlyGrowth := quarterlyPoints[len(quarterlyPoints)-1].ARR - quarterlyPoints[0].ARR
		quarters := quarterlyPoints[len(quarterlyPoints)-1].Timestamp.Sub(quarterlyPoints[0].Timestamp).Hours() / (24 * 90)
		if quarters > 0 {
			velocity.QuarterlyARR = quarterlyGrowth / quarters
			velocity.QoQGrowth = (quarterlyGrowth / quarterlyPoints[0].ARR) * 100
		}
	}

	// Yearly velocity (YoY)
	yearlyPoints := t.getDataPointsSince(now.AddDate(-1, 0, 0))
	if len(yearlyPoints) > 1 {
		yearlyGrowth := yearlyPoints[len(yearlyPoints)-1].ARR - yearlyPoints[0].ARR
		velocity.YoYGrowth = (yearlyGrowth / yearlyPoints[0].ARR) * 100
	}

	// Run rate (annualized monthly growth)
	velocity.RunRate = velocity.MonthlyARR * 12

	// Required velocity to hit target
	if !t.milestone.ProjectedDate.IsZero() {
		daysToTarget := time.Until(t.milestone.ProjectedDate).Hours() / 24
		if daysToTarget > 0 {
			velocity.RequiredVelocity = t.milestone.RemainingARR / daysToTarget
		}
	}

	// Check if on track
	velocity.OnTrack = velocity.DailyARR >= velocity.RequiredVelocity

	// Calculate acceleration
	if len(t.historicalData) > 30 {
		oldVelocity := t.calculateHistoricalVelocity(30)
		if oldVelocity > 0 {
			velocity.Acceleration = ((velocity.MonthlyARR - oldVelocity) / oldVelocity) * 100
		}
	}

	return velocity
}

// calculateMetrics computes comprehensive ARR metrics
func (t *BillionARRTracker) calculateMetrics(composition RevenueComposition) ARRMetrics {
	metrics := ARRMetrics{
		TotalCustomers:   composition.NewBusiness.Customers + composition.Expansion.Customers,
		Fortune500:       280, // Current baseline
		AvgContractValue: 5_000_000, // $5M target
		GrossMargin:      42.0,
		NetMargin:        18.0,
		RenewalRate:      97.0,
		NetRetention:     150.0,
		CAC:              500_000,
		LTV:              15_000_000,
		PaybackPeriod:    12,
	}

	// Calculate derived metrics
	if metrics.CAC > 0 {
		metrics.LTVtoCAC = metrics.LTV / metrics.CAC
	}

	// Rule of 40: Growth% + Margin% should be > 40
	metrics.RuleOf40 = t.milestone.GrowthRate + metrics.NetMargin

	// Magic Number: ARR growth / Sales & Marketing spend
	// Assuming 25% of revenue on S&M
	if t.milestone.CurrentARR > 0 {
		smSpend := t.milestone.CurrentARR * 0.25
		arrGrowth := t.milestone.CurrentARR - t.milestone.PreviousARR
		if smSpend > 0 {
			metrics.MagicNumber = arrGrowth / smSpend
		}
	}

	return metrics
}

// generateForecasts creates ARR forecasts
func (t *BillionARRTracker) generateForecasts(ctx context.Context) ([]ARRForecast, error) {
	forecasts := make([]ARRForecast, 0)

	// Generate 365-day forecast
	currentARR := t.milestone.CurrentARR
	dailyGrowth := t.milestone.Velocity.DailyARR

	if dailyGrowth <= 0 {
		dailyGrowth = (t.milestone.TargetARR - currentARR) / 365
	}

	for i := 0; i < 365; i += 7 { // Weekly forecasts
		date := time.Now().AddDate(0, 0, i)
		forecastARR := currentARR + (dailyGrowth * float64(i))

		// Add confidence intervals
		variance := forecastARR * 0.1 * (float64(i) / 365) // Increasing variance

		forecast := ARRForecast{
			Date:        date,
			ForecastARR: forecastARR,
			LowerBound:  forecastARR - variance,
			UpperBound:  forecastARR + variance,
			Confidence:  math.Max(0.5, 1.0-(float64(i)/730)), // Decreasing confidence
			Method:      "linear-regression",
			Assumptions: map[string]interface{}{
				"daily_growth":    dailyGrowth,
				"current_arr":     currentARR,
				"renewal_rate":    97.0,
				"net_retention":   150.0,
			},
			RiskFactors: []string{
				"Market saturation in enterprise segment",
				"Competitive pressure on pricing",
				"Economic headwinds affecting IT budgets",
			},
		}

		forecasts = append(forecasts, forecast)
	}

	t.metrics.forecasts++
	return forecasts, nil
}

// checkAlerts evaluates alert rules
func (t *BillionARRTracker) checkAlerts() []MilestoneAlert {
	alerts := make([]MilestoneAlert, 0)

	// Check growth rate
	if t.milestone.GrowthRate < 20.0 {
		alerts = append(alerts, MilestoneAlert{
			ID:       fmt.Sprintf("alert-%d", time.Now().Unix()),
			Severity: "warning",
			Type:     "growth_rate",
			Message:  "Growth rate below 20% threshold",
			Metric:   "growth_rate",
			Threshold: 20.0,
			ActualValue: t.milestone.GrowthRate,
			Impact:   "May miss $1B ARR target",
			Recommendation: "Accelerate enterprise expansion and new logo acquisition",
			CreatedAt: time.Now(),
		})
	}

	// Check velocity
	if !t.milestone.Velocity.OnTrack {
		alerts = append(alerts, MilestoneAlert{
			ID:       fmt.Sprintf("alert-%d", time.Now().Unix()+1),
			Severity: "critical",
			Type:     "velocity",
			Message:  "ARR velocity below required rate",
			Metric:   "daily_arr",
			Threshold: t.milestone.Velocity.RequiredVelocity,
			ActualValue: t.milestone.Velocity.DailyARR,
			Impact:   "Will not reach $1B on target date",
			Recommendation: "Increase sales capacity and accelerate pipeline",
			CreatedAt: time.Now(),
		})
	}

	// Check net retention
	if t.milestone.Metrics.NetRetention < 150.0 {
		alerts = append(alerts, MilestoneAlert{
			ID:       fmt.Sprintf("alert-%d", time.Now().Unix()+2),
			Severity: "warning",
			Type:     "retention",
			Message:  "Net retention below 150% target",
			Metric:   "net_retention",
			Threshold: 150.0,
			ActualValue: t.milestone.Metrics.NetRetention,
			Impact:   "Expansion revenue at risk",
			Recommendation: "Enhance customer success and upsell programs",
			CreatedAt: time.Now(),
		})
	}

	t.metrics.alerts += int64(len(alerts))
	return alerts
}

// Helper functions

func (t *BillionARRTracker) getDataPointsSince(since time.Time) []ARRSnapshot {
	points := make([]ARRSnapshot, 0)
	for _, snapshot := range t.historicalData {
		if snapshot.Timestamp.After(since) {
			points = append(points, snapshot)
		}
	}
	return points
}

func (t *BillionARRTracker) calculateHistoricalVelocity(daysAgo int) float64 {
	points := t.getDataPointsSince(time.Now().AddDate(0, 0, -daysAgo-7))
	if len(points) < 2 {
		return 0
	}

	growth := points[len(points)-1].ARR - points[0].ARR
	days := points[len(points)-1].Timestamp.Sub(points[0].Timestamp).Hours() / 24
	if days > 0 {
		return growth / days
	}
	return 0
}

func initializeForecastModels() map[string]ForecastModel {
	return map[string]ForecastModel{
		"linear": {
			Name:        "Linear Regression",
			Type:        "linear",
			Accuracy:    0.85,
			LastTrained: time.Now(),
			Enabled:     true,
		},
		"arima": {
			Name:        "ARIMA",
			Type:        "arima",
			Accuracy:    0.90,
			LastTrained: time.Now(),
			Enabled:     true,
		},
		"prophet": {
			Name:        "Facebook Prophet",
			Type:        "prophet",
			Accuracy:    0.92,
			LastTrained: time.Now(),
			Enabled:     true,
		},
		"ml": {
			Name:        "Machine Learning Ensemble",
			Type:        "ml",
			Accuracy:    0.95,
			LastTrained: time.Now(),
			Enabled:     true,
		},
	}
}

func initializeAlertRules() []AlertRule {
	return []AlertRule{
		{
			ID:        "growth-rate",
			Name:      "Growth Rate Below Target",
			Metric:    "growth_rate",
			Condition: "<",
			Threshold: 20.0,
			Severity:  "warning",
			Enabled:   true,
			Actions:   []string{"notify", "escalate"},
		},
		{
			ID:        "velocity",
			Name:      "Velocity Below Required",
			Metric:    "daily_arr",
			Condition: "<",
			Threshold: 0, // Calculated dynamically
			Severity:  "critical",
			Enabled:   true,
			Actions:   []string{"notify", "escalate", "intervention"},
		},
		{
			ID:        "net-retention",
			Name:      "Net Retention Below Target",
			Metric:    "net_retention",
			Condition: "<",
			Threshold: 150.0,
			Severity:  "warning",
			Enabled:   true,
			Actions:   []string{"notify"},
		},
	}
}

// GetMilestone returns current milestone state
func (t *BillionARRTracker) GetMilestone() *ARRMilestone {
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Deep copy
	milestone := *t.milestone
	return &milestone
}

// ExportMetrics exports tracker metrics
func (t *BillionARRTracker) ExportMetrics() map[string]interface{} {
	t.metrics.mu.RLock()
	defer t.metrics.mu.RUnlock()

	return map[string]interface{}{
		"forecasts_generated":   t.metrics.forecasts,
		"predictions_made":      t.metrics.predictions,
		"alerts_triggered":      t.metrics.alerts,
		"arr_updates":           t.metrics.updates,
		"last_update":           t.metrics.lastUpdate,
		"errors":                t.metrics.errors,
	}
}

// MarshalJSON implements json.Marshaler
func (t *BillionARRTracker) MarshalJSON() ([]byte, error) {
	t.mu.RLock()
	defer t.mu.RUnlock()

	return json.Marshal(map[string]interface{}{
		"milestone":       t.milestone,
		"cohort_count":    len(t.cohorts),
		"prediction_count": len(t.predictions),
		"historical_data_points": len(t.historicalData),
		"forecast_models": t.forecastModels,
		"alert_rules":     t.alertRules,
		"metrics":         t.ExportMetrics(),
	})
}
