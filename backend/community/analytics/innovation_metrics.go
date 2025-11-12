// Package analytics implements Innovation Metrics & Analytics
// Feature velocity, contribution metrics, ecosystem health, innovation ROI
// Target: 100+ features/year from community
package analytics

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// MetricsPeriod represents time period for metrics
type MetricsPeriod string

const (
	PeriodDaily     MetricsPeriod = "daily"
	PeriodWeekly    MetricsPeriod = "weekly"
	PeriodMonthly   MetricsPeriod = "monthly"
	PeriodQuarterly MetricsPeriod = "quarterly"
	PeriodYearly    MetricsPeriod = "yearly"
)

// InnovationMetrics tracks ecosystem innovation
type InnovationMetrics struct {
	Period              MetricsPeriod
	StartDate           time.Time
	EndDate             time.Time
	FeatureVelocity     FeatureVelocityMetrics
	ContributionMetrics ContributionMetricsData
	EcosystemHealth     EcosystemHealthScore
	DeveloperMetrics    DeveloperEngagementMetrics
	MarketplaceMetrics  MarketplaceAnalytics
	InnovationROI       InnovationROIMetrics
	PatentPipeline      PatentMetrics
	CommunityGrowth     CommunityGrowthMetrics
	QualityMetrics      QualityScoreMetrics
	CollectedAt         time.Time
}

// FeatureVelocityMetrics tracks feature development speed
type FeatureVelocityMetrics struct {
	TotalFeatures           int
	CommunityFeatures       int
	InternalFeatures        int
	FeaturesPerMonth        float64
	AverageTimeToMarket     float64 // days
	FeaturesByCategory      map[string]int
	FeaturesByPriority      map[string]int
	CompletionRate          float64
	BacklogSize             int
	VelocityTrend           string // increasing, stable, decreasing
	TopContributors         []TopContributor
}

// TopContributor represents top feature contributor
type TopContributor struct {
	ID              string
	Name            string
	FeaturesCreated int
	ImpactScore     float64
	Rank            int
}

// ContributionMetricsData tracks community contributions
type ContributionMetricsData struct {
	TotalContributions      int
	ContributionsThisPeriod int
	ContributionsByType     map[string]int
	TotalContributors       int
	ActiveContributors      int
	NewContributors         int
	ReturnContributors      int
	ContributionRate        float64
	AcceptanceRate          float64
	AverageReviewTime       float64 // hours
	AverageMergeTime        float64 // hours
	CodeQualityScore        float64
	TestCoverageAverage     float64
	DocumentationScore      float64
	ContributionTrend       TrendData
}

// TrendData represents metric trend
type TrendData struct {
	Current     float64
	Previous    float64
	Change      float64
	ChangePercent float64
	Direction   string // up, down, stable
}

// EcosystemHealthScore represents overall ecosystem health
type EcosystemHealthScore struct {
	OverallScore            float64 // 0-100
	DeveloperSatisfaction   float64
	ApiStability            float64
	DocumentationQuality    float64
	CommunityEngagement     float64
	PlatformReliability     float64
	InnovationIndex         float64
	GrowthMomentum          float64
	SecurityPosture         float64
	PerformanceScore        float64
	EcosystemDiversity      float64
	HealthTrend             string // improving, stable, declining
	RiskFactors             []RiskFactor
	StrengthAreas           []string
	ImprovementAreas        []string
}

// RiskFactor represents ecosystem risk
type RiskFactor struct {
	Category    string
	Severity    string // critical, high, medium, low
	Description string
	Impact      float64
	Mitigation  string
}

// DeveloperEngagementMetrics tracks developer engagement
type DeveloperEngagementMetrics struct {
	DAU                     int // Daily Active Users
	MAU                     int // Monthly Active Users
	WAU                     int // Weekly Active Users
	NewDevelopers           int
	ReturnDevelopers        int
	ChurnedDevelopers       int
	RetentionRate           float64
	ChurnRate               float64
	AverageSessionDuration  float64 // minutes
	APICallsPerDeveloper    float64
	FeaturesUsedPerDeveloper float64
	EngagementScore         float64
	ActivationRate          float64
	TimeToFirstValue        float64 // hours
	PowerUserPercentage     float64
	CasualUserPercentage    float64
	EngagementTrend         TrendData
}

// MarketplaceAnalytics tracks marketplace performance
type MarketplaceAnalytics struct {
	TotalApps               int
	NewApps                 int
	ActiveApps              int
	TotalInstalls           int64
	NewInstalls             int64
	TotalRevenue            float64
	RevenueThisPeriod       float64
	AverageRevenuePerApp    float64
	TopGrossingApps         []AppPerformance
	FastestGrowingApps      []AppPerformance
	AppsByCategory          map[string]int
	AverageRating           float64
	ConversionRate          float64
	ChurnRate               float64
	ARPU                    float64 // Average Revenue Per User
	LTV                     float64 // Lifetime Value
	CAC                     float64 // Customer Acquisition Cost
	MarketplaceTrend        TrendData
}

// AppPerformance represents app performance data
type AppPerformance struct {
	AppID       string
	AppName     string
	Revenue     float64
	Installs    int64
	Rating      float64
	GrowthRate  float64
	Rank        int
}

// InnovationROIMetrics tracks innovation return on investment
type InnovationROIMetrics struct {
	TotalInvestment         float64
	CommunityInvestment     float64
	InternalInvestment      float64
	DirectRevenue           float64
	IndirectRevenue         float64
	CostSavings             float64
	TotalReturn             float64
	ROI                     float64 // Return / Investment
	ROIByCategory           map[string]float64
	PaybackPeriod           float64 // months
	TimeToValue             float64 // months
	ValueCreated            float64
	ProductivityGains       float64
	DeveloperEfficiency     float64
	InnovationEffectiveness float64
}

// PatentMetrics tracks patent pipeline
type PatentMetrics struct {
	TotalPatents            int
	PatentsFromCommunity    int
	PatentsFiled            int
	PatentsGranted          int
	PatentsPending          int
	PatentsByCategory       map[string]int
	AverageTimeToGrant      float64 // months
	PatentValue             float64
	CommercializationRate   float64
	LicensingRevenue        float64
}

// CommunityGrowthMetrics tracks community expansion
type CommunityGrowthMetrics struct {
	TotalMembers            int
	NewMembers              int
	ActiveMembers           int
	MemberGrowthRate        float64
	ActivationRate          float64
	EngagementRate          float64
	RetentionRate           float64
	ChurnRate               float64
	CertifiedDevelopers     int
	CertificationRate       float64
	GeographicDistribution  map[string]int
	IndustryDistribution    map[string]int
	CompanySizeDistribution map[string]int
	GrowthTrend             TrendData
	ViraCoefficient         float64
	NetPromoterScore        float64
}

// QualityScoreMetrics tracks quality metrics
type QualityScoreMetrics struct {
	OverallQuality          float64
	CodeQuality             float64
	TestCoverage            float64
	DocumentationQuality    float64
	SecurityScore           float64
	PerformanceScore        float64
	ReliabilityScore        float64
	UsabilityScore          float64
	MaintainabilityScore    float64
	BugDensity              float64
	TechnicalDebtRatio      float64
	QualityTrend            TrendData
}

// InnovationAnalyticsManager manages innovation analytics
type InnovationAnalyticsManager struct {
	mu              sync.RWMutex
	metrics         map[string]*InnovationMetrics
	historicalData  []InnovationMetrics
	realTimeMetrics *RealTimeMetrics
}

// RealTimeMetrics tracks real-time metrics
type RealTimeMetrics struct {
	CurrentDAU          int
	CurrentAPICallsPerSec int
	CurrentEventRate    int
	ActiveSessions      int
	ErrorRate           float64
	AvgResponseTime     float64
	UpdatedAt           time.Time
}

// NewInnovationAnalyticsManager creates analytics manager
func NewInnovationAnalyticsManager() *InnovationAnalyticsManager {
	iam := &InnovationAnalyticsManager{
		metrics:        make(map[string]*InnovationMetrics),
		historicalData: []InnovationMetrics{},
		realTimeMetrics: &RealTimeMetrics{
			UpdatedAt: time.Now(),
		},
	}

	iam.initializeHistoricalData()

	return iam
}

// initializeHistoricalData creates historical metrics
func (iam *InnovationAnalyticsManager) initializeHistoricalData() {
	// Create 12 months of historical data
	for i := 0; i < 12; i++ {
		startDate := time.Now().AddDate(0, -12+i, 0)
		endDate := startDate.AddDate(0, 1, 0)

		metrics := &InnovationMetrics{
			Period:    PeriodMonthly,
			StartDate: startDate,
			EndDate:   endDate,
			FeatureVelocity: FeatureVelocityMetrics{
				TotalFeatures:       8 + i,
				CommunityFeatures:   3 + i/2,
				InternalFeatures:    5 + i/2,
				FeaturesPerMonth:    8.0 + float64(i)*0.5,
				AverageTimeToMarket: 30.0 - float64(i)*0.5,
				CompletionRate:      0.85 + float64(i)*0.01,
				BacklogSize:         50 - i,
				VelocityTrend:       "increasing",
			},
			ContributionMetrics: ContributionMetricsData{
				TotalContributions:      80 + i*10,
				ContributionsThisPeriod: 80 + i*10,
				TotalContributors:       100 + i*20,
				ActiveContributors:      50 + i*10,
				NewContributors:         20 + i*2,
				ContributionRate:        0.75 + float64(i)*0.01,
				AcceptanceRate:          0.85 + float64(i)*0.01,
				AverageReviewTime:       24.0 - float64(i)*0.5,
				AverageMergeTime:        48.0 - float64(i),
				CodeQualityScore:        85.0 + float64(i)*0.5,
				TestCoverageAverage:     80.0 + float64(i)*0.5,
			},
			EcosystemHealth: EcosystemHealthScore{
				OverallScore:         80.0 + float64(i)*1.5,
				DeveloperSatisfaction: 85.0 + float64(i)*0.5,
				ApiStability:         90.0 + float64(i)*0.3,
				DocumentationQuality: 82.0 + float64(i)*0.8,
				CommunityEngagement:  78.0 + float64(i)*1.0,
				PlatformReliability:  95.0 + float64(i)*0.2,
				InnovationIndex:      75.0 + float64(i)*1.2,
				GrowthMomentum:       80.0 + float64(i)*1.0,
				SecurityPosture:      92.0 + float64(i)*0.3,
				PerformanceScore:     88.0 + float64(i)*0.5,
				HealthTrend:          "improving",
			},
			DeveloperMetrics: DeveloperEngagementMetrics{
				DAU:                    5000 + i*500,
				MAU:                    20000 + i*2000,
				WAU:                    12000 + i*1200,
				NewDevelopers:          500 + i*50,
				RetentionRate:          0.85 + float64(i)*0.01,
				ChurnRate:              0.15 - float64(i)*0.01,
				EngagementScore:        75.0 + float64(i)*1.0,
				ActivationRate:         0.65 + float64(i)*0.02,
				PowerUserPercentage:    0.20 + float64(i)*0.01,
			},
			MarketplaceMetrics: MarketplaceAnalytics{
				TotalApps:         100 + i*10,
				NewApps:           10 + i,
				ActiveApps:        90 + i*9,
				TotalInstalls:     int64(10000 + i*1000),
				NewInstalls:       int64(1000 + i*100),
				TotalRevenue:      float64(100000 + i*10000),
				RevenueThisPeriod: float64(10000 + i*1000),
				AverageRating:     4.5 + float64(i)*0.01,
				ConversionRate:    0.15 + float64(i)*0.01,
				ARPU:              50.0 + float64(i)*2,
			},
			InnovationROI: InnovationROIMetrics{
				TotalInvestment:     float64(1000000 + i*100000),
				DirectRevenue:       float64(2000000 + i*200000),
				IndirectRevenue:     float64(500000 + i*50000),
				CostSavings:         float64(300000 + i*30000),
				ROI:                 2.8 + float64(i)*0.1,
				TimeToValue:         6.0 - float64(i)*0.2,
				ProductivityGains:   1.5 + float64(i)*0.05,
			},
			PatentPipeline: PatentMetrics{
				TotalPatents:         10 + i/2,
				PatentsFromCommunity: 3 + i/4,
				PatentsFiled:         2 + i/6,
				PatentsGranted:       1 + i/12,
				PatentsPending:       9 + i/3,
			},
			CommunityGrowth: CommunityGrowthMetrics{
				TotalMembers:        10000 + i*1000,
				NewMembers:          1000 + i*100,
				ActiveMembers:       8000 + i*800,
				MemberGrowthRate:    0.10 + float64(i)*0.01,
				ActivationRate:      0.75 + float64(i)*0.01,
				EngagementRate:      0.65 + float64(i)*0.02,
				RetentionRate:       0.85 + float64(i)*0.01,
				CertifiedDevelopers: 1000 + i*100,
				NetPromoterScore:    60.0 + float64(i)*2,
			},
			QualityMetrics: QualityScoreMetrics{
				OverallQuality:       85.0 + float64(i)*0.5,
				CodeQuality:          87.0 + float64(i)*0.4,
				TestCoverage:         82.0 + float64(i)*0.6,
				DocumentationQuality: 80.0 + float64(i)*0.8,
				SecurityScore:        90.0 + float64(i)*0.3,
				PerformanceScore:     88.0 + float64(i)*0.4,
				ReliabilityScore:     92.0 + float64(i)*0.2,
			},
			CollectedAt: startDate,
		}

		iam.historicalData = append(iam.historicalData, *metrics)
		iam.metrics[fmt.Sprintf("%s-%s", metrics.Period, startDate.Format("2006-01"))] = metrics
	}
}

// CollectMetrics collects current period metrics
func (iam *InnovationAnalyticsManager) CollectMetrics(ctx context.Context, period MetricsPeriod) (*InnovationMetrics, error) {
	iam.mu.Lock()
	defer iam.mu.Unlock()

	now := time.Now()
	startDate, endDate := iam.getPeriodDates(period, now)

	metrics := &InnovationMetrics{
		Period:      period,
		StartDate:   startDate,
		EndDate:     endDate,
		CollectedAt: now,
	}

	// In production, collect from actual data sources
	// This is simplified simulation
	metrics.FeatureVelocity = FeatureVelocityMetrics{
		TotalFeatures:       120,
		CommunityFeatures:   45,
		FeaturesPerMonth:    10.0,
		CompletionRate:      0.92,
		VelocityTrend:       "increasing",
	}

	metrics.EcosystemHealth = EcosystemHealthScore{
		OverallScore:         92.5,
		DeveloperSatisfaction: 90.0,
		HealthTrend:          "improving",
	}

	key := fmt.Sprintf("%s-%s", period, startDate.Format("2006-01-02"))
	iam.metrics[key] = metrics
	iam.historicalData = append(iam.historicalData, *metrics)

	return metrics, nil
}

// GetMetrics retrieves metrics for period
func (iam *InnovationAnalyticsManager) GetMetrics(ctx context.Context, period MetricsPeriod, date time.Time) (*InnovationMetrics, error) {
	iam.mu.RLock()
	defer iam.mu.RUnlock()

	key := fmt.Sprintf("%s-%s", period, date.Format("2006-01-02"))
	metrics, exists := iam.metrics[key]
	if !exists {
		return nil, fmt.Errorf("metrics not found for period: %s", key)
	}

	return metrics, nil
}

// GetHistoricalData returns historical metrics
func (iam *InnovationAnalyticsManager) GetHistoricalData(ctx context.Context, period MetricsPeriod, months int) ([]InnovationMetrics, error) {
	iam.mu.RLock()
	defer iam.mu.RUnlock()

	var result []InnovationMetrics
	for i := len(iam.historicalData) - 1; i >= 0 && len(result) < months; i-- {
		if iam.historicalData[i].Period == period {
			result = append([]InnovationMetrics{iam.historicalData[i]}, result...)
		}
	}

	return result, nil
}

// GetRealTimeMetrics returns real-time metrics
func (iam *InnovationAnalyticsManager) GetRealTimeMetrics(ctx context.Context) *RealTimeMetrics {
	iam.mu.RLock()
	defer iam.mu.RUnlock()

	return iam.realTimeMetrics
}

// GenerateReport generates comprehensive analytics report
func (iam *InnovationAnalyticsManager) GenerateReport(ctx context.Context, period MetricsPeriod, date time.Time) ([]byte, error) {
	iam.mu.RLock()
	defer iam.mu.RUnlock()

	metrics, err := iam.GetMetrics(ctx, period, date)
	if err != nil {
		return nil, err
	}

	report := map[string]interface{}{
		"period":      period,
		"date":        date,
		"metrics":     metrics,
		"generated_at": time.Now(),
	}

	return json.MarshalIndent(report, "", "  ")
}

// getPeriodDates calculates period start and end dates
func (iam *InnovationAnalyticsManager) getPeriodDates(period MetricsPeriod, date time.Time) (time.Time, time.Time) {
	switch period {
	case PeriodDaily:
		start := time.Date(date.Year(), date.Month(), date.Day(), 0, 0, 0, 0, date.Location())
		return start, start.AddDate(0, 0, 1)
	case PeriodWeekly:
		start := date.AddDate(0, 0, -int(date.Weekday()))
		return start, start.AddDate(0, 0, 7)
	case PeriodMonthly:
		start := time.Date(date.Year(), date.Month(), 1, 0, 0, 0, 0, date.Location())
		return start, start.AddDate(0, 1, 0)
	case PeriodQuarterly:
		quarter := (int(date.Month())-1)/3 + 1
		start := time.Date(date.Year(), time.Month((quarter-1)*3+1), 1, 0, 0, 0, 0, date.Location())
		return start, start.AddDate(0, 3, 0)
	case PeriodYearly:
		start := time.Date(date.Year(), 1, 1, 0, 0, 0, 0, date.Location())
		return start, start.AddDate(1, 0, 0)
	}
	return date, date
}

// CalculateTrend calculates metric trend
func (iam *InnovationAnalyticsManager) CalculateTrend(current, previous float64) TrendData {
	change := current - previous
	changePercent := 0.0
	if previous > 0 {
		changePercent = (change / previous) * 100
	}

	direction := "stable"
	if change > 0 {
		direction = "up"
	} else if change < 0 {
		direction = "down"
	}

	return TrendData{
		Current:       current,
		Previous:      previous,
		Change:        change,
		ChangePercent: changePercent,
		Direction:     direction,
	}
}
