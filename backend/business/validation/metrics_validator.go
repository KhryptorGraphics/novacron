// Package validation implements business metrics validation for Phase 12 Market Domination
// Validates $1B ARR milestone tracking, 50%+ market share achievement, and business targets.
package validation

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// BusinessTargets represents Phase 12 market domination targets
type BusinessTargets struct {
	// Revenue targets
	ARRTarget              float64 `json:"arr_target"`                // $1B ARR
	ARRCurrent             float64 `json:"arr_current"`               // Current ARR
	ARRGrowthRequired      float64 `json:"arr_growth_required"`       // 10x growth

	// Market share targets
	MarketShareTarget      float64 `json:"market_share_target"`       // 50%+ market share
	MarketShareCurrent     float64 `json:"market_share_current"`      // Current share

	// Customer targets
	Fortune500Target       int     `json:"fortune500_target"`         // 300/500 (60%)
	Fortune500Current      int     `json:"fortune500_current"`        // Current count
	NewCustomersTarget     int     `json:"new_customers_target"`      // 50,000+ new customers

	// Competitive targets
	CompetitiveWinRate     float64 `json:"competitive_win_rate"`      // 90%+ win rate
	VMwareDisplacementRate float64 `json:"vmware_displacement_rate"`  // 70%+ vs VMware
	AWSDisplacementRate    float64 `json:"aws_displacement_rate"`     // 60%+ vs AWS
	K8sDisplacementRate    float64 `json:"k8s_displacement_rate"`     // 80%+ vs K8s

	// Vertical penetration targets
	FinancialTarget        float64 `json:"financial_target"`          // 80% of top 100 banks
	HealthcareTarget       float64 `json:"healthcare_target"`         // 70% of top 100 hospitals
	TelecomTarget          float64 `json:"telecom_target"`            // 75% of carriers
	RetailTarget           float64 `json:"retail_target"`             // 60% of Fortune 500 retailers

	// Partner ecosystem targets
	PartnerCountTarget     int     `json:"partner_count_target"`      // 5,000+ partners
	PartnerRevenueTarget   float64 `json:"partner_revenue_target"`    // $200M+ partner revenue
}

// ValidationResult represents validation outcome
type ValidationResult struct {
	Category        string    `json:"category"`
	Metric          string    `json:"metric"`
	Target          float64   `json:"target"`
	Actual          float64   `json:"actual"`
	Achievement     float64   `json:"achievement"`      // % of target achieved
	Status          string    `json:"status"`           // achieved, on_track, at_risk, critical
	GapAnalysis     string    `json:"gap_analysis"`
	Recommendation  string    `json:"recommendation"`
	ValidatedAt     time.Time `json:"validated_at"`
}

// MetricsValidator validates business metrics against targets
type MetricsValidator struct {
	mu              sync.RWMutex
	targets         BusinessTargets
	validations     []*ValidationResult
	overallStatus   string
	ctx             context.Context
	cancel          context.CancelFunc
}

// NewMetricsValidator creates business metrics validator
func NewMetricsValidator() *MetricsValidator {
	ctx, cancel := context.WithCancel(context.Background())

	return &MetricsValidator{
		targets: BusinessTargets{
			// Phase 12 targets
			ARRTarget:              1_000_000_000, // $1B
			ARRCurrent:             120_000_000,   // $120M (from Phase 11)
			ARRGrowthRequired:      8.33,          // 10x growth

			MarketShareTarget:      0.50, // 50%+
			MarketShareCurrent:     0.35, // 35% (from Phase 11)

			Fortune500Target:       300, // 300/500 (60%)
			Fortune500Current:      150, // Current from Phase 11
			NewCustomersTarget:     50000,

			CompetitiveWinRate:     0.90, // 90%+
			VMwareDisplacementRate: 0.70, // 70%+
			AWSDisplacementRate:    0.60, // 60%+
			K8sDisplacementRate:    0.80, // 80%+

			FinancialTarget:        0.80, // 80% of top 100 banks
			HealthcareTarget:       0.70, // 70% of top 100 hospitals
			TelecomTarget:          0.75, // 75% of carriers
			RetailTarget:           0.60, // 60% of retailers

			PartnerCountTarget:     5000,        // 5,000+ partners
			PartnerRevenueTarget:   200_000_000, // $200M+
		},
		validations:   make([]*ValidationResult, 0),
		overallStatus: "initializing",
		ctx:           ctx,
		cancel:        cancel,
	}
}

// ValidateRevenueMetrics validates $1B ARR milestone
func (v *MetricsValidator) ValidateRevenueMetrics(currentARR, quarterlyGrowth, nrr float64) *ValidationResult {
	v.mu.Lock()
	defer v.mu.Unlock()

	achievement := (currentARR / v.targets.ARRTarget) * 100

	status := "critical"
	if achievement >= 95 {
		status = "achieved"
	} else if achievement >= 85 {
		status = "on_track"
	} else if achievement >= 70 {
		status = "at_risk"
	}

	gap := v.targets.ARRTarget - currentARR
	recommendation := ""

	if status != "achieved" {
		recommendation = fmt.Sprintf(
			"Gap: $%.0fM to $1B target. Required actions: " +
			"(1) Accelerate Fortune 500 land & expand (+$300M), " +
			"(2) Scale mid-market (+$250M), " +
			"(3) Maximize expansion revenue (+$200M), " +
			"(4) Execute competitive displacement (+$150M)",
			gap/1_000_000,
		)
	}

	result := &ValidationResult{
		Category:       "Revenue",
		Metric:         "Annual Recurring Revenue (ARR)",
		Target:         v.targets.ARRTarget,
		Actual:         currentARR,
		Achievement:    achievement,
		Status:         status,
		GapAnalysis:    fmt.Sprintf("$%.0fM gap to $1B target", gap/1_000_000),
		Recommendation: recommendation,
		ValidatedAt:    time.Now(),
	}

	v.validations = append(v.validations, result)
	return result
}

// ValidateMarketShareMetrics validates 50%+ market share achievement
func (v *MetricsValidator) ValidateMarketShareMetrics(currentShare, competitiveWinRate float64) *ValidationResult {
	v.mu.Lock()
	defer v.mu.Unlock()

	achievement := (currentShare / v.targets.MarketShareTarget) * 100

	status := "critical"
	if currentShare >= 0.50 {
		status = "achieved"
	} else if currentShare >= 0.45 {
		status = "on_track"
	} else if currentShare >= 0.38 {
		status = "at_risk"
	}

	gap := v.targets.MarketShareTarget - currentShare
	recommendation := ""

	if status != "achieved" {
		recommendation = fmt.Sprintf(
			"Gap: %.1f%% to 50%% target. Required actions: " +
			"(1) Increase competitive win rate to 90%% (current: %.1f%%), " +
			"(2) Execute M&A acquisitions (5+ targets), " +
			"(3) Accelerate vertical domination strategy, " +
			"(4) Expand partner ecosystem to 5,000+",
			gap*100, competitiveWinRate*100,
		)
	}

	result := &ValidationResult{
		Category:       "Market Share",
		Metric:         "Market Share %",
		Target:         v.targets.MarketShareTarget,
		Actual:         currentShare,
		Achievement:    achievement,
		Status:         status,
		GapAnalysis:    fmt.Sprintf("%.1f%% gap to 50%% target", gap*100),
		Recommendation: recommendation,
		ValidatedAt:    time.Now(),
	}

	v.validations = append(v.validations, result)
	return result
}

// ValidateFortune500Penetration validates 300/500 Fortune 500 penetration
func (v *MetricsValidator) ValidateFortune500Penetration(currentCount int) *ValidationResult {
	v.mu.Lock()
	defer v.mu.Unlock()

	achievement := (float64(currentCount) / float64(v.targets.Fortune500Target)) * 100

	status := "critical"
	if currentCount >= 300 {
		status = "achieved"
	} else if currentCount >= 250 {
		status = "on_track"
	} else if currentCount >= 200 {
		status = "at_risk"
	}

	gap := v.targets.Fortune500Target - currentCount
	penetrationRate := (float64(currentCount) / 500) * 100

	recommendation := ""
	if status != "achieved" {
		recommendation = fmt.Sprintf(
			"Gap: %d Fortune 500 companies to 300 target (60%% penetration). Required actions: " +
			"(1) Launch dedicated Fortune 500 ABM campaigns, " +
			"(2) Assign platinum account teams, " +
			"(3) Leverage competitive displacement (VMware uncertainty), " +
			"(4) Execute C-level engagement program",
			gap,
		)
	}

	result := &ValidationResult{
		Category:       "Customer Acquisition",
		Metric:         "Fortune 500 Customers",
		Target:         float64(v.targets.Fortune500Target),
		Actual:         float64(currentCount),
		Achievement:    achievement,
		Status:         status,
		GapAnalysis:    fmt.Sprintf("%d customers gap (current: %d/500 = %.1f%%)", gap, currentCount, penetrationRate),
		Recommendation: recommendation,
		ValidatedAt:    time.Now(),
	}

	v.validations = append(v.validations, result)
	return result
}

// ValidateCompetitiveMetrics validates 90%+ competitive win rate
func (v *MetricsValidator) ValidateCompetitiveMetrics(overallWinRate, vmwareWinRate, awsWinRate, k8sWinRate float64) []*ValidationResult {
	v.mu.Lock()
	defer v.mu.Unlock()

	results := make([]*ValidationResult, 0)

	// Overall competitive win rate
	overallAchievement := (overallWinRate / v.targets.CompetitiveWinRate) * 100
	overallStatus := v.determineStatus(overallAchievement, 95, 90, 85)

	overallResult := &ValidationResult{
		Category:       "Competitive",
		Metric:         "Overall Competitive Win Rate",
		Target:         v.targets.CompetitiveWinRate,
		Actual:         overallWinRate,
		Achievement:    overallAchievement,
		Status:         overallStatus,
		GapAnalysis:    fmt.Sprintf("%.1f%% win rate (target: 90%%+)", overallWinRate*100),
		Recommendation: v.generateCompetitiveRecommendation(overallWinRate, "overall"),
		ValidatedAt:    time.Now(),
	}
	results = append(results, overallResult)
	v.validations = append(v.validations, overallResult)

	// VMware displacement
	vmwareAchievement := (vmwareWinRate / v.targets.VMwareDisplacementRate) * 100
	vmwareStatus := v.determineStatus(vmwareAchievement, 100, 95, 90)

	vmwareResult := &ValidationResult{
		Category:       "Competitive",
		Metric:         "VMware Displacement Win Rate",
		Target:         v.targets.VMwareDisplacementRate,
		Actual:         vmwareWinRate,
		Achievement:    vmwareAchievement,
		Status:         vmwareStatus,
		GapAnalysis:    fmt.Sprintf("%.1f%% win rate vs VMware (target: 70%%+)", vmwareWinRate*100),
		Recommendation: v.generateCompetitiveRecommendation(vmwareWinRate, "vmware"),
		ValidatedAt:    time.Now(),
	}
	results = append(results, vmwareResult)
	v.validations = append(v.validations, vmwareResult)

	// AWS displacement
	awsAchievement := (awsWinRate / v.targets.AWSDisplacementRate) * 100
	awsStatus := v.determineStatus(awsAchievement, 100, 95, 90)

	awsResult := &ValidationResult{
		Category:       "Competitive",
		Metric:         "AWS Displacement Win Rate",
		Target:         v.targets.AWSDisplacementRate,
		Actual:         awsWinRate,
		Achievement:    awsAchievement,
		Status:         awsStatus,
		GapAnalysis:    fmt.Sprintf("%.1f%% win rate vs AWS (target: 60%%+)", awsWinRate*100),
		Recommendation: v.generateCompetitiveRecommendation(awsWinRate, "aws"),
		ValidatedAt:    time.Now(),
	}
	results = append(results, awsResult)
	v.validations = append(v.validations, awsResult)

	// Kubernetes displacement
	k8sAchievement := (k8sWinRate / v.targets.K8sDisplacementRate) * 100
	k8sStatus := v.determineStatus(k8sAchievement, 100, 95, 90)

	k8sResult := &ValidationResult{
		Category:       "Competitive",
		Metric:         "Kubernetes Displacement Win Rate",
		Target:         v.targets.K8sDisplacementRate,
		Actual:         k8sWinRate,
		Achievement:    k8sAchievement,
		Status:         k8sStatus,
		GapAnalysis:    fmt.Sprintf("%.1f%% win rate vs K8s (target: 80%%+)", k8sWinRate*100),
		Recommendation: v.generateCompetitiveRecommendation(k8sWinRate, "kubernetes"),
		ValidatedAt:    time.Now(),
	}
	results = append(results, k8sResult)
	v.validations = append(v.validations, k8sResult)

	return results
}

// ValidateVerticalPenetration validates industry vertical targets
func (v *MetricsValidator) ValidateVerticalPenetration(financial, healthcare, telecom, retail float64) []*ValidationResult {
	v.mu.Lock()
	defer v.mu.Unlock()

	results := make([]*ValidationResult, 0)

	// Financial services
	financialResult := v.createVerticalResult("Financial Services (Top 100 Banks)", v.targets.FinancialTarget, financial)
	results = append(results, financialResult)
	v.validations = append(v.validations, financialResult)

	// Healthcare
	healthcareResult := v.createVerticalResult("Healthcare (Top 100 Hospitals)", v.targets.HealthcareTarget, healthcare)
	results = append(results, healthcareResult)
	v.validations = append(v.validations, healthcareResult)

	// Telecommunications
	telecomResult := v.createVerticalResult("Telecommunications (Global Carriers)", v.targets.TelecomTarget, telecom)
	results = append(results, telecomResult)
	v.validations = append(v.validations, telecomResult)

	// Retail
	retailResult := v.createVerticalResult("Retail (Fortune 500 Retailers)", v.targets.RetailTarget, retail)
	results = append(results, retailResult)
	v.validations = append(v.validations, retailResult)

	return results
}

// ValidatePartnerEcosystem validates 5,000+ partner target
func (v *MetricsValidator) ValidatePartnerEcosystem(partnerCount int, partnerRevenue float64) []*ValidationResult {
	v.mu.Lock()
	defer v.mu.Unlock()

	results := make([]*ValidationResult, 0)

	// Partner count
	countAchievement := (float64(partnerCount) / float64(v.targets.PartnerCountTarget)) * 100
	countStatus := v.determineStatus(countAchievement, 95, 85, 75)

	countResult := &ValidationResult{
		Category:       "Partner Ecosystem",
		Metric:         "Total Partner Count",
		Target:         float64(v.targets.PartnerCountTarget),
		Actual:         float64(partnerCount),
		Achievement:    countAchievement,
		Status:         countStatus,
		GapAnalysis:    fmt.Sprintf("%d partners gap to 5,000 target", v.targets.PartnerCountTarget-partnerCount),
		Recommendation: fmt.Sprintf("Accelerate partner recruitment: %d additional partners needed", v.targets.PartnerCountTarget-partnerCount),
		ValidatedAt:    time.Now(),
	}
	results = append(results, countResult)
	v.validations = append(v.validations, countResult)

	// Partner revenue
	revenueAchievement := (partnerRevenue / v.targets.PartnerRevenueTarget) * 100
	revenueStatus := v.determineStatus(revenueAchievement, 95, 85, 75)

	revenueResult := &ValidationResult{
		Category:       "Partner Ecosystem",
		Metric:         "Partner-Sourced Revenue",
		Target:         v.targets.PartnerRevenueTarget,
		Actual:         partnerRevenue,
		Achievement:    revenueAchievement,
		Status:         revenueStatus,
		GapAnalysis:    fmt.Sprintf("$%.0fM gap to $200M target", (v.targets.PartnerRevenueTarget-partnerRevenue)/1_000_000),
		Recommendation: fmt.Sprintf("Increase partner deal registration and co-selling: $%.0fM additional revenue needed", (v.targets.PartnerRevenueTarget-partnerRevenue)/1_000_000),
		ValidatedAt:    time.Now(),
	}
	results = append(results, revenueResult)
	v.validations = append(v.validations, revenueResult)

	return results
}

// determineStatus determines validation status based on achievement
func (v *MetricsValidator) determineStatus(achievement, achievedThreshold, onTrackThreshold, atRiskThreshold float64) string {
	if achievement >= achievedThreshold {
		return "achieved"
	} else if achievement >= onTrackThreshold {
		return "on_track"
	} else if achievement >= atRiskThreshold {
		return "at_risk"
	}
	return "critical"
}

// createVerticalResult creates validation result for vertical market
func (v *MetricsValidator) createVerticalResult(verticalName string, target, actual float64) *ValidationResult {
	achievement := (actual / target) * 100
	status := v.determineStatus(achievement, 95, 85, 75)

	gap := target - actual
	recommendation := ""
	if status != "achieved" {
		recommendation = fmt.Sprintf(
			"Gap: %.1f%% to target. Launch vertical-specific campaigns and compliance certifications.",
			gap*100,
		)
	}

	return &ValidationResult{
		Category:       "Vertical Penetration",
		Metric:         verticalName,
		Target:         target,
		Actual:         actual,
		Achievement:    achievement,
		Status:         status,
		GapAnalysis:    fmt.Sprintf("%.1f%% penetration (target: %.1f%%)", actual*100, target*100),
		Recommendation: recommendation,
		ValidatedAt:    time.Now(),
	}
}

// generateCompetitiveRecommendation creates competitive strategy recommendation
func (v *MetricsValidator) generateCompetitiveRecommendation(winRate float64, competitor string) string {
	if winRate >= 0.90 {
		return fmt.Sprintf("Excellent %s competitive performance - maintain battlecards and playbooks", competitor)
	}

	recommendations := map[string]string{
		"overall":    "Enhance competitive battlecards, accelerate sales training, deploy win/loss analysis",
		"vmware":     "Leverage Broadcom uncertainty, emphasize TCO advantage, accelerate migration programs",
		"aws":        "Focus on workload repatriation ROI, highlight cost predictability, target FinOps teams",
		"kubernetes": "Emphasize operational simplicity, VM-native advantage, unified management",
	}

	return fmt.Sprintf("Win rate below target - %s", recommendations[competitor])
}

// GenerateValidationReport creates comprehensive validation report
func (v *MetricsValidator) GenerateValidationReport() *ValidationReport {
	v.mu.RLock()
	defer v.mu.RUnlock()

	// Calculate overall status
	achieved := 0
	onTrack := 0
	atRisk := 0
	critical := 0

	for _, validation := range v.validations {
		switch validation.Status {
		case "achieved":
			achieved++
		case "on_track":
			onTrack++
		case "at_risk":
			atRisk++
		case "critical":
			critical++
		}
	}

	overallStatus := "critical"
	if critical == 0 && atRisk == 0 {
		overallStatus = "achieved"
	} else if critical == 0 {
		overallStatus = "on_track"
	} else if critical <= 2 {
		overallStatus = "at_risk"
	}

	return &ValidationReport{
		Targets:       v.targets,
		Validations:   v.validations,
		OverallStatus: overallStatus,
		Summary: ValidationSummary{
			Achieved: achieved,
			OnTrack:  onTrack,
			AtRisk:   atRisk,
			Critical: critical,
		},
		GeneratedAt: time.Now(),
	}
}

// ValidationReport represents comprehensive validation output
type ValidationReport struct {
	Targets       BusinessTargets      `json:"targets"`
	Validations   []*ValidationResult  `json:"validations"`
	OverallStatus string               `json:"overall_status"`
	Summary       ValidationSummary    `json:"summary"`
	GeneratedAt   time.Time            `json:"generated_at"`
}

// ValidationSummary summarizes validation results
type ValidationSummary struct {
	Achieved int `json:"achieved"`
	OnTrack  int `json:"on_track"`
	AtRisk   int `json:"at_risk"`
	Critical int `json:"critical"`
}

// ExportValidation exports validation report as JSON
func (v *MetricsValidator) ExportValidation() ([]byte, error) {
	report := v.GenerateValidationReport()
	return json.MarshalIndent(report, "", "  ")
}

// Close shuts down the metrics validator
func (v *MetricsValidator) Close() error {
	v.cancel()
	return nil
}
