// Package competitive implements Market Share Tracking & Competitive Intelligence
// for achieving 50%+ market domination, competitive displacement, and M&A strategy.
package competitive

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// MarketShareTarget represents the 50%+ market share goal
type MarketShareTarget struct {
	CurrentShare      float64   `json:"current_share"`       // 35% current
	TargetShare       float64   `json:"target_share"`        // 50%+ target
	TAM               float64   `json:"tam"`                 // Total addressable market
	SAM               float64   `json:"sam"`                 // Serviceable addressable market
	SOM               float64   `json:"som"`                 // Serviceable obtainable market
	MarketGrowthRate  float64   `json:"market_growth_rate"`  // Annual market growth
	CompetitorCount   int       `json:"competitor_count"`
	TargetDate        time.Time `json:"target_date"`         // 2027 target
	MarketPosition    int       `json:"market_position"`     // #1 = market leader
	LastUpdated       time.Time `json:"last_updated"`
}

// CompetitorProfile represents competitive intelligence
type CompetitorProfile struct {
	CompetitorID        string                 `json:"competitor_id"`
	CompetitorName      string                 `json:"competitor_name"`
	MarketShare         float64                `json:"market_share"`
	EstimatedRevenue    float64                `json:"estimated_revenue"`
	CustomerCount       int                    `json:"customer_count"`
	GrowthRate          float64                `json:"growth_rate"`
	Strengths           []string               `json:"strengths"`
	Weaknesses          []string               `json:"weaknesses"`
	CompetitiveWinRate  float64                `json:"competitive_win_rate"`  // Our win rate against them
	DisplacementTarget  bool                   `json:"displacement_target"`
	AcquisitionTarget   bool                   `json:"acquisition_target"`
	EstimatedValuation  float64                `json:"estimated_valuation"`
	TechnologyStack     []string               `json:"technology_stack"`
	KeyCustomers        []string               `json:"key_customers"`
	RecentNews          []string               `json:"recent_news"`
	ThreatLevel         string                 `json:"threat_level"` // critical, high, medium, low
	CustomAttributes    map[string]interface{} `json:"custom_attributes"`
	LastUpdated         time.Time              `json:"last_updated"`
}

// CompetitiveWin represents a competitive displacement win
type CompetitiveWin struct {
	WinID               string    `json:"win_id"`
	AccountName         string    `json:"account_name"`
	CompetitorDisplaced string    `json:"competitor_displaced"`
	DealValue           float64   `json:"deal_value"`
	WinReason           []string  `json:"win_reasons"`
	DisplacementType    string    `json:"displacement_type"` // full_replacement, partial, greenfield
	TimeToWin           int       `json:"time_to_win"`       // Days
	DiscountOffered     float64   `json:"discount_offered"`
	MigrationComplexity string    `json:"migration_complexity"` // low, medium, high
	CustomerSatisfaction float64  `json:"customer_satisfaction"`
	WinDate             time.Time `json:"win_date"`
	LastUpdated         time.Time `json:"last_updated"`
}

// CompetitiveLoss represents a competitive loss for analysis
type CompetitiveLoss struct {
	LossID              string    `json:"loss_id"`
	AccountName         string    `json:"account_name"`
	CompetitorWon       string    `json:"competitor_won"`
	DealValue           float64   `json:"deal_value"`
	LossReason          []string  `json:"loss_reasons"`
	PriceGap            float64   `json:"price_gap"`         // % difference
	FeatureGaps         []string  `json:"feature_gaps"`
	LessonsLearned      []string  `json:"lessons_learned"`
	PreventionStrategy  string    `json:"prevention_strategy"`
	LossDate            time.Time `json:"loss_date"`
	LastUpdated         time.Time `json:"last_updated"`
}

// AcquisitionTarget represents M&A pipeline opportunity
type AcquisitionTarget struct {
	TargetID            string                 `json:"target_id"`
	CompanyName         string                 `json:"company_name"`
	EstimatedValuation  float64                `json:"estimated_valuation"`
	EstimatedRevenue    float64                `json:"estimated_revenue"`
	CustomerCount       int                    `json:"customer_count"`
	MarketShare         float64                `json:"market_share"`
	TechnologyValue     float64                `json:"technology_value"`     // IP, patents
	StrategicFit        float64                `json:"strategic_fit"`        // 0-1 score
	IntegrationRisk     float64                `json:"integration_risk"`     // 0-1 score
	SynergyValue        float64                `json:"synergy_value"`        // Expected cost savings
	AcquisitionStage    string                 `json:"acquisition_stage"`    // research, approach, negotiation, diligence
	ExpectedCloseDate   time.Time              `json:"expected_close_date"`
	KeyAssets           []string               `json:"key_assets"`
	RegulatoryRisk      float64                `json:"regulatory_risk"`
	CustomAttributes    map[string]interface{} `json:"custom_attributes"`
	LastUpdated         time.Time              `json:"last_updated"`
}

// MarketShareTracker manages market domination strategy
type MarketShareTracker struct {
	mu                  sync.RWMutex
	target              MarketShareTarget
	competitors         map[string]*CompetitorProfile
	competitiveWins     map[string]*CompetitiveWin
	competitiveLosses   map[string]*CompetitiveLoss
	acquisitionPipeline map[string]*AcquisitionTarget
	marketMetrics       *MarketMetrics
	ctx                 context.Context
	cancel              context.CancelFunc
}

// MarketMetrics tracks real-time market performance
type MarketMetrics struct {
	CurrentMarketShare      float64            `json:"current_market_share"`
	MarketShareGrowth       float64            `json:"market_share_growth"`
	CompetitiveWinRate      float64            `json:"competitive_win_rate"`
	WinRateByCompetitor     map[string]float64 `json:"win_rate_by_competitor"`
	TotalCompetitiveWins    int                `json:"total_competitive_wins"`
	TotalCompetitiveLosses  int                `json:"total_competitive_losses"`
	AvgDealSize             float64            `json:"avg_deal_size"`
	AvgTimeToWin            float64            `json:"avg_time_to_win"`
	AvgDiscount             float64            `json:"avg_discount"`
	MarketLeadershipScore   float64            `json:"market_leadership_score"`
	BrandRecognitionScore   float64            `json:"brand_recognition_score"`
	CustomerPreferenceScore float64            `json:"customer_preference_score"`
	LastUpdated             time.Time          `json:"last_updated"`
}

// NewMarketShareTracker creates the market share tracking system
func NewMarketShareTracker() *MarketShareTracker {
	ctx, cancel := context.WithCancel(context.Background())

	tracker := &MarketShareTracker{
		target: MarketShareTarget{
			CurrentShare:     0.35, // 35% current market share
			TargetShare:      0.50, // 50%+ target
			TAM:              100_000_000_000, // $100B TAM
			SAM:              50_000_000_000,  // $50B SAM
			SOM:              25_000_000_000,  // $25B SOM
			MarketGrowthRate: 0.25,            // 25% annual growth
			CompetitorCount:  25,
			TargetDate:       time.Date(2027, 12, 31, 0, 0, 0, 0, time.UTC),
			MarketPosition:   1, // Already #1
			LastUpdated:      time.Now(),
		},
		competitors:         make(map[string]*CompetitorProfile),
		competitiveWins:     make(map[string]*CompetitiveWin),
		competitiveLosses:   make(map[string]*CompetitiveLoss),
		acquisitionPipeline: make(map[string]*AcquisitionTarget),
		marketMetrics:       &MarketMetrics{},
		ctx:                 ctx,
		cancel:              cancel,
	}

	tracker.initializeCompetitors()
	tracker.initializeAcquisitionTargets()

	return tracker
}

// initializeCompetitors sets up competitive landscape
func (t *MarketShareTracker) initializeCompetitors() {
	t.competitors["vmware"] = &CompetitorProfile{
		CompetitorID:       "vmware",
		CompetitorName:     "VMware vSphere",
		MarketShare:        0.28, // 28% market share
		EstimatedRevenue:   12_000_000_000,
		CustomerCount:      500000,
		GrowthRate:         -0.05, // Declining
		Strengths:          []string{"Legacy install base", "Enterprise relationships", "Ecosystem"},
		Weaknesses:         []string{"High licensing costs", "Complex management", "Legacy architecture", "Broadcom uncertainty"},
		CompetitiveWinRate: 0.70,  // 70% win rate vs VMware
		DisplacementTarget: true,
		AcquisitionTarget:  false, // Too large
		EstimatedValuation: 60_000_000_000,
		TechnologyStack:    []string{"ESXi", "vCenter", "vSAN", "NSX"},
		KeyCustomers:       []string{"Fortune 500 enterprises", "Service providers"},
		RecentNews:         []string{"Broadcom acquisition completed", "Price increases announced", "Customer backlash"},
		ThreatLevel:        "medium", // Declining
		LastUpdated:        time.Now(),
	}

	t.competitors["aws-ec2"] = &CompetitorProfile{
		CompetitorID:       "aws-ec2",
		CompetitorName:     "AWS EC2",
		MarketShare:        0.18,
		EstimatedRevenue:   25_000_000_000,
		CustomerCount:      1000000,
		GrowthRate:         0.30,
		Strengths:          []string{"Market leader", "Global infrastructure", "Service breadth", "Innovation pace"},
		Weaknesses:         []string{"High costs at scale", "Vendor lock-in", "Compliance complexity", "Opaque pricing"},
		CompetitiveWinRate: 0.60, // 60% win rate vs AWS
		DisplacementTarget: true,
		AcquisitionTarget:  false,
		EstimatedValuation: 500_000_000_000,
		TechnologyStack:    []string{"EC2", "S3", "RDS", "Lambda"},
		KeyCustomers:       []string{"Startups", "Digital natives", "Enterprises"},
		RecentNews:         []string{"Continued growth", "New regions", "Price optimization tools"},
		ThreatLevel:        "high",
		LastUpdated:        time.Now(),
	}

	t.competitors["azure"] = &CompetitorProfile{
		CompetitorID:       "azure",
		CompetitorName:     "Microsoft Azure",
		MarketShare:        0.12,
		EstimatedRevenue:   20_000_000_000,
		CustomerCount:      800000,
		GrowthRate:         0.40,
		Strengths:          []string{"Microsoft ecosystem", "Enterprise relationships", "Hybrid capabilities"},
		Weaknesses:         []string{"Complexity", "Service reliability", "Fragmented experience"},
		CompetitiveWinRate: 0.60,
		DisplacementTarget: true,
		AcquisitionTarget:  false,
		EstimatedValuation: 400_000_000_000,
		TechnologyStack:    []string{"Azure VMs", "Azure Storage", "Azure SQL"},
		KeyCustomers:       []string{"Microsoft shops", "Enterprises"},
		RecentNews:         []string{"AI investments", "OpenAI partnership"},
		ThreatLevel:        "high",
		LastUpdated:        time.Now(),
	}

	t.competitors["kubernetes"] = &CompetitorProfile{
		CompetitorID:       "kubernetes",
		CompetitorName:     "Kubernetes (DIY)",
		MarketShare:        0.05,
		EstimatedRevenue:   5_000_000_000, // Indirect
		CustomerCount:      200000,
		GrowthRate:         0.50,
		Strengths:          []string{"Open source", "Container-native", "Large ecosystem"},
		Weaknesses:         []string{"Operational complexity", "Security challenges", "VM support lacking", "Steep learning curve"},
		CompetitiveWinRate: 0.80, // 80% win rate vs K8s (VM-native advantage)
		DisplacementTarget: true,
		AcquisitionTarget:  false, // Open source
		EstimatedValuation: 0,
		TechnologyStack:    []string{"Kubernetes", "Docker", "Helm"},
		KeyCustomers:       []string{"Cloud-native startups", "DevOps teams"},
		RecentNews:         []string{"CNCF growth", "Security concerns", "Complexity complaints"},
		ThreatLevel:        "medium",
		LastUpdated:        time.Now(),
	}

	t.competitors["nutanix"] = &CompetitorProfile{
		CompetitorID:       "nutanix",
		CompetitorName:     "Nutanix",
		MarketShare:        0.02,
		EstimatedRevenue:   1_500_000_000,
		CustomerCount:      25000,
		GrowthRate:         0.15,
		Strengths:          []string{"HCI focus", "Simple management"},
		Weaknesses:         []string{"Limited scale", "High hardware costs", "Slower innovation"},
		CompetitiveWinRate: 0.85, // 85% win rate vs Nutanix
		DisplacementTarget: true,
		AcquisitionTarget:  true, // Good acquisition target
		EstimatedValuation: 5_000_000_000,
		TechnologyStack:    []string{"AHV", "Prism", "Acropolis"},
		KeyCustomers:       []string{"Mid-market enterprises"},
		RecentNews:         []string{"Cloud transition struggles", "Competition from hyperscalers"},
		ThreatLevel:        "low",
		LastUpdated:        time.Now(),
	}
}

// initializeAcquisitionTargets sets up M&A pipeline
func (t *MarketShareTracker) initializeAcquisitionTargets() {
	t.acquisitionPipeline["nutanix"] = &AcquisitionTarget{
		TargetID:           "nutanix",
		CompanyName:        "Nutanix",
		EstimatedValuation: 5_000_000_000,
		EstimatedRevenue:   1_500_000_000,
		CustomerCount:      25000,
		MarketShare:        0.02,
		TechnologyValue:    1_000_000_000,
		StrategicFit:       0.85,
		IntegrationRisk:    0.40,
		SynergyValue:       500_000_000, // $500M synergies
		AcquisitionStage:   "research",
		ExpectedCloseDate:  time.Date(2026, 6, 30, 0, 0, 0, 0, time.UTC),
		KeyAssets:          []string{"HCI patents", "Customer base", "Management software"},
		RegulatoryRisk:     0.30,
		LastUpdated:        time.Now(),
	}

	t.acquisitionPipeline["rancher"] = &AcquisitionTarget{
		TargetID:           "rancher",
		CompanyName:        "Rancher Labs",
		EstimatedValuation: 1_000_000_000,
		EstimatedRevenue:   150_000_000,
		CustomerCount:      15000,
		MarketShare:        0.005,
		TechnologyValue:    500_000_000,
		StrategicFit:       0.90,
		IntegrationRisk:    0.30,
		SynergyValue:       200_000_000,
		AcquisitionStage:   "approach",
		ExpectedCloseDate:  time.Date(2026, 3, 31, 0, 0, 0, 0, time.UTC),
		KeyAssets:          []string{"K8s management", "Multi-cluster orchestration", "Developer tools"},
		RegulatoryRisk:     0.15,
		LastUpdated:        time.Now(),
	}

	t.acquisitionPipeline["scale-computing"] = &AcquisitionTarget{
		TargetID:           "scale-computing",
		CompanyName:        "Scale Computing",
		EstimatedValuation: 500_000_000,
		EstimatedRevenue:   100_000_000,
		CustomerCount:      8000,
		MarketShare:        0.003,
		TechnologyValue:    200_000_000,
		StrategicFit:       0.75,
		IntegrationRisk:    0.25,
		SynergyValue:       100_000_000,
		AcquisitionStage:   "research",
		ExpectedCloseDate:  time.Date(2026, 9, 30, 0, 0, 0, 0, time.UTC),
		KeyAssets:          []string{"Edge computing", "SMB customer base"},
		RegulatoryRisk:     0.10,
		LastUpdated:        time.Now(),
	}

	t.acquisitionPipeline["platform9"] = &AcquisitionTarget{
		TargetID:           "platform9",
		CompanyName:        "Platform9",
		EstimatedValuation: 300_000_000,
		EstimatedRevenue:   50_000_000,
		CustomerCount:      5000,
		MarketShare:        0.002,
		TechnologyValue:    150_000_000,
		StrategicFit:       0.80,
		IntegrationRisk:    0.35,
		SynergyValue:       75_000_000,
		AcquisitionStage:   "approach",
		ExpectedCloseDate:  time.Date(2026, 12, 31, 0, 0, 0, 0, time.UTC),
		KeyAssets:          []string{"Managed K8s", "OpenStack expertise", "SaaS management"},
		RegulatoryRisk:     0.10,
		LastUpdated:        time.Now(),
	}

	t.acquisitionPipeline["morpheus-data"] = &AcquisitionTarget{
		TargetID:           "morpheus-data",
		CompanyName:        "Morpheus Data",
		EstimatedValuation: 400_000_000,
		EstimatedRevenue:   75_000_000,
		CustomerCount:      6000,
		MarketShare:        0.002,
		TechnologyValue:    180_000_000,
		StrategicFit:       0.85,
		IntegrationRisk:    0.30,
		SynergyValue:       120_000_000,
		AcquisitionStage:   "research",
		ExpectedCloseDate:  time.Date(2027, 3, 31, 0, 0, 0, 0, time.UTC),
		KeyAssets:          []string{"Multi-cloud orchestration", "Application catalog", "Self-service portal"},
		RegulatoryRisk:     0.15,
		LastUpdated:        time.Now(),
	}
}

// RecordCompetitiveWin tracks competitive displacement success
func (t *MarketShareTracker) RecordCompetitiveWin(win *CompetitiveWin) {
	t.mu.Lock()
	defer t.mu.Unlock()

	win.LastUpdated = time.Now()
	t.competitiveWins[win.WinID] = win

	// Update competitor win rate
	if competitor, exists := t.competitors[win.CompetitorDisplaced]; exists {
		t.updateCompetitorWinRate(competitor.CompetitorID)
	}
}

// RecordCompetitiveLoss tracks competitive losses for learning
func (t *MarketShareTracker) RecordCompetitiveLoss(loss *CompetitiveLoss) {
	t.mu.Lock()
	defer t.mu.Unlock()

	loss.LastUpdated = time.Now()
	t.competitiveLosses[loss.LossID] = loss

	// Update competitor data
	if competitor, exists := t.competitors[loss.CompetitorWon]; exists {
		t.updateCompetitorWinRate(competitor.CompetitorID)
	}
}

// updateCompetitorWinRate recalculates win rate against specific competitor
func (t *MarketShareTracker) updateCompetitorWinRate(competitorID string) {
	wins := 0
	losses := 0

	for _, win := range t.competitiveWins {
		if win.CompetitorDisplaced == competitorID {
			wins++
		}
	}

	for _, loss := range t.competitiveLosses {
		if loss.CompetitorWon == competitorID {
			losses++
		}
	}

	total := wins + losses
	if total > 0 {
		t.competitors[competitorID].CompetitiveWinRate = float64(wins) / float64(total)
	}
}

// CalculateMarketMetrics computes real-time market performance
func (t *MarketShareTracker) CalculateMarketMetrics() *MarketMetrics {
	t.mu.RLock()
	defer t.mu.RUnlock()

	totalWins := len(t.competitiveWins)
	totalLosses := len(t.competitiveLosses)

	competitiveWinRate := 0.0
	if totalWins+totalLosses > 0 {
		competitiveWinRate = float64(totalWins) / float64(totalWins+totalLosses)
	}

	// Calculate win rate by competitor
	winRateByCompetitor := make(map[string]float64)
	for compID := range t.competitors {
		wins := 0
		losses := 0

		for _, win := range t.competitiveWins {
			if win.CompetitorDisplaced == compID {
				wins++
			}
		}

		for _, loss := range t.competitiveLosses {
			if loss.CompetitorWon == compID {
				losses++
			}
		}

		total := wins + losses
		if total > 0 {
			winRateByCompetitor[compID] = float64(wins) / float64(total)
		}
	}

	// Calculate average deal metrics
	var totalDealValue float64
	var totalTimeToWin float64
	var totalDiscount float64

	for _, win := range t.competitiveWins {
		totalDealValue += win.DealValue
		totalTimeToWin += float64(win.TimeToWin)
		totalDiscount += win.DiscountOffered
	}

	avgDealSize := 0.0
	avgTimeToWin := 0.0
	avgDiscount := 0.0

	if totalWins > 0 {
		avgDealSize = totalDealValue / float64(totalWins)
		avgTimeToWin = totalTimeToWin / float64(totalWins)
		avgDiscount = totalDiscount / float64(totalWins)
	}

	metrics := &MarketMetrics{
		CurrentMarketShare:      t.target.CurrentShare,
		MarketShareGrowth:       t.calculateMarketShareGrowth(),
		CompetitiveWinRate:      competitiveWinRate,
		WinRateByCompetitor:     winRateByCompetitor,
		TotalCompetitiveWins:    totalWins,
		TotalCompetitiveLosses:  totalLosses,
		AvgDealSize:             avgDealSize,
		AvgTimeToWin:            avgTimeToWin,
		AvgDiscount:             avgDiscount,
		MarketLeadershipScore:   t.calculateLeadershipScore(),
		BrandRecognitionScore:   t.calculateBrandScore(),
		CustomerPreferenceScore: t.calculatePreferenceScore(),
		LastUpdated:             time.Now(),
	}

	t.marketMetrics = metrics
	return metrics
}

// calculateMarketShareGrowth computes market share growth rate
func (t *MarketShareTracker) calculateMarketShareGrowth() float64 {
	// Simulate market share growth based on competitive wins
	winsThisQuarter := len(t.competitiveWins) // Simplified
	return float64(winsThisQuarter) * 0.001 // Each win = 0.1% market share
}

// calculateLeadershipScore computes market leadership position
func (t *MarketShareTracker) calculateLeadershipScore() float64 {
	// Factors: market share, win rate, customer count, brand recognition
	shareScore := t.target.CurrentShare
	winRate := t.marketMetrics.CompetitiveWinRate

	// Weighted average
	return (shareScore*0.4 + winRate*0.3 + 0.85*0.3)
}

// calculateBrandScore computes brand recognition strength
func (t *MarketShareTracker) calculateBrandScore() float64 {
	// Based on market share and competitive wins
	return math.Min(t.target.CurrentShare*2 + 0.15, 1.0)
}

// calculatePreferenceScore computes customer preference rating
func (t *MarketShareTracker) calculatePreferenceScore() float64 {
	// Based on win rate and customer satisfaction
	return t.marketMetrics.CompetitiveWinRate * 0.95
}

// GetTopCompetitors returns ranked competitors by threat level
func (t *MarketShareTracker) GetTopCompetitors(limit int) []*CompetitorProfile {
	t.mu.RLock()
	defer t.mu.RUnlock()

	competitors := make([]*CompetitorProfile, 0, len(t.competitors))
	for _, comp := range t.competitors {
		competitors = append(competitors, comp)
	}

	// Sort by market share
	sort.Slice(competitors, func(i, j int) bool {
		return competitors[i].MarketShare > competitors[j].MarketShare
	})

	if limit > len(competitors) {
		limit = len(competitors)
	}

	return competitors[:limit]
}

// GetAcquisitionPipeline returns M&A targets ranked by strategic value
func (t *MarketShareTracker) GetAcquisitionPipeline() []*AcquisitionTarget {
	t.mu.RLock()
	defer t.mu.RUnlock()

	targets := make([]*AcquisitionTarget, 0, len(t.acquisitionPipeline))
	for _, target := range t.acquisitionPipeline {
		targets = append(targets, target)
	}

	// Sort by strategic fit and synergy value
	sort.Slice(targets, func(i, j int) bool {
		scoreI := targets[i].StrategicFit * targets[i].SynergyValue / targets[i].EstimatedValuation
		scoreJ := targets[j].StrategicFit * targets[j].SynergyValue / targets[j].EstimatedValuation
		return scoreI > scoreJ
	})

	return targets
}

// GetCompetitiveWins returns recent competitive wins
func (t *MarketShareTracker) GetCompetitiveWins(limit int) []*CompetitiveWin {
	t.mu.RLock()
	defer t.mu.RUnlock()

	wins := make([]*CompetitiveWin, 0, len(t.competitiveWins))
	for _, win := range t.competitiveWins {
		wins = append(wins, win)
	}

	sort.Slice(wins, func(i, j int) bool {
		return wins[i].WinDate.After(wins[j].WinDate)
	})

	if limit > len(wins) {
		limit = len(wins)
	}

	return wins[:limit]
}

// ProjectMarketShare forecasts market share trajectory
func (t *MarketShareTracker) ProjectMarketShare(quarters int) []MarketShareProjection {
	t.mu.RLock()
	defer t.mu.RUnlock()

	projections := make([]MarketShareProjection, quarters)
	currentShare := t.target.CurrentShare

	for i := 0; i < quarters; i++ {
		// Calculate quarterly growth needed to reach target
		remainingQuarters := float64(quarters - i)
		quarterlyGrowth := math.Pow(t.target.TargetShare/currentShare, 1/remainingQuarters)

		currentShare *= quarterlyGrowth

		projections[i] = MarketShareProjection{
			Quarter:           i + 1,
			ProjectedShare:    currentShare,
			OrganicGrowth:     currentShare * 0.6,
			CompetitiveWins:   currentShare * 0.3,
			Acquisitions:      currentShare * 0.1,
			Confidence:        t.calculateProjectionConfidence(i),
			LastUpdated:       time.Now(),
		}
	}

	return projections
}

// calculateProjectionConfidence determines forecast accuracy
func (t *MarketShareTracker) calculateProjectionConfidence(quartersOut int) float64 {
	baseConfidence := 0.92
	decayRate := 0.03
	return baseConfidence * math.Exp(-decayRate*float64(quartersOut))
}

// MarketShareProjection represents forecasted market share
type MarketShareProjection struct {
	Quarter         int       `json:"quarter"`
	ProjectedShare  float64   `json:"projected_share"`
	OrganicGrowth   float64   `json:"organic_growth"`
	CompetitiveWins float64   `json:"competitive_wins"`
	Acquisitions    float64   `json:"acquisitions"`
	Confidence      float64   `json:"confidence"`
	LastUpdated     time.Time `json:"last_updated"`
}

// GenerateMarketReport creates comprehensive competitive intelligence report
func (t *MarketShareTracker) GenerateMarketReport() *MarketReport {
	t.mu.RLock()
	defer t.mu.RUnlock()

	metrics := t.CalculateMarketMetrics()
	competitors := t.GetTopCompetitors(10)
	acquisitionTargets := t.GetAcquisitionPipeline()
	recentWins := t.GetCompetitiveWins(20)
	projections := t.ProjectMarketShare(12) // 3-year forecast

	report := &MarketReport{
		Target:             t.target,
		CurrentMetrics:     metrics,
		TopCompetitors:     competitors,
		AcquisitionTargets: acquisitionTargets,
		RecentWins:         recentWins,
		MarketProjections:  projections,
		HealthStatus:       t.assessMarketHealth(),
		Recommendations:    t.generateMarketRecommendations(),
		GeneratedAt:        time.Now(),
	}

	return report
}

// assessMarketHealth evaluates market domination progress
func (t *MarketShareTracker) assessMarketHealth() string {
	metrics := t.marketMetrics

	if metrics.CurrentMarketShare >= 0.45 && metrics.CompetitiveWinRate >= 0.88 {
		return "excellent"
	} else if metrics.CurrentMarketShare >= 0.38 && metrics.CompetitiveWinRate >= 0.80 {
		return "good"
	} else if metrics.CurrentMarketShare >= 0.30 {
		return "fair"
	}

	return "needs_attention"
}

// generateMarketRecommendations creates strategic recommendations
func (t *MarketShareTracker) generateMarketRecommendations() []string {
	recommendations := []string{}

	metrics := t.marketMetrics

	if metrics.CurrentMarketShare < 0.45 {
		recommendations = append(recommendations,
			fmt.Sprintf("🎯 Current market share %.1f%% - accelerate to 50%+ through competitive displacement",
				metrics.CurrentMarketShare*100))
	}

	if metrics.CompetitiveWinRate < 0.90 {
		recommendations = append(recommendations,
			fmt.Sprintf("⚔️ Competitive win rate %.1f%% - target 90%+ through enhanced battlecards",
				metrics.CompetitiveWinRate*100))
	}

	// VMware displacement opportunity
	if vmware, exists := t.competitors["vmware"]; exists && vmware.CompetitiveWinRate < 0.75 {
		recommendations = append(recommendations,
			"💼 VMware displacement: increase win rate to 70%+ with Broadcom uncertainty messaging")
	}

	// M&A recommendations
	if len(t.acquisitionPipeline) > 0 {
		topTarget := t.GetAcquisitionPipeline()[0]
		recommendations = append(recommendations,
			fmt.Sprintf("🤝 Priority M&A: %s ($%.0fM valuation, $%.0fM synergies)",
				topTarget.CompanyName, topTarget.EstimatedValuation/1_000_000,
				topTarget.SynergyValue/1_000_000))
	}

	return recommendations
}

// MarketReport represents comprehensive market intelligence
type MarketReport struct {
	Target             MarketShareTarget         `json:"target"`
	CurrentMetrics     *MarketMetrics            `json:"current_metrics"`
	TopCompetitors     []*CompetitorProfile      `json:"top_competitors"`
	AcquisitionTargets []*AcquisitionTarget      `json:"acquisition_targets"`
	RecentWins         []*CompetitiveWin         `json:"recent_wins"`
	MarketProjections  []MarketShareProjection   `json:"market_projections"`
	HealthStatus       string                    `json:"health_status"`
	Recommendations    []string                  `json:"recommendations"`
	GeneratedAt        time.Time                 `json:"generated_at"`
}

// ExportMetrics exports market metrics in JSON format
func (t *MarketShareTracker) ExportMetrics() ([]byte, error) {
	report := t.GenerateMarketReport()
	return json.MarshalIndent(report, "", "  ")
}

// Close shuts down the market share tracker
func (t *MarketShareTracker) Close() error {
	t.cancel()
	return nil
}
