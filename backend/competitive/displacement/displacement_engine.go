// Competitive Displacement Engine
// Automated playbooks for displacing VMware, AWS, Azure, GCP, and Kubernetes
// Real-time competitive intelligence and win/loss analysis

package displacement

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// DisplacementEngine orchestrates competitive displacement strategies
type DisplacementEngine struct {
	id                string
	playbooks         map[string]*DisplacementPlaybook
	battleCards       map[string]*CompetitiveBattleCard
	opportunities     map[string]*DisplacementOpportunity
	winLossTracker    *WinLossTracker
	intelligenceHub   *CompetitiveIntelligence
	migrationEngine   *CustomerMigrationEngine
	roiCalculator     *DisplacementROICalculator
	mu                sync.RWMutex
	autoUpdateEnabled bool
	updateInterval    time.Duration
}

// DisplacementPlaybook defines competitive displacement strategy
type DisplacementPlaybook struct {
	PlaybookID       string                 `json:"playbook_id"`
	CompetitorName   string                 `json:"competitor_name"`
	CompetitorType   string                 `json:"competitor_type"`
	TargetWinRate    float64                `json:"target_win_rate"`
	CurrentWinRate   float64                `json:"current_win_rate"`
	TotalOpportunities int                  `json:"total_opportunities"`
	WonDeals         int                    `json:"won_deals"`
	LostDeals        int                    `json:"lost_deals"`
	ActiveDeals      int                    `json:"active_deals"`
	Tactics          []DisplacementTactic   `json:"tactics"`
	ValueProps       []ValueProposition     `json:"value_props"`
	PainPoints       []CompetitorPainPoint  `json:"pain_points"`
	CaseStudies      []string               `json:"case_studies"`
	ROIMetrics       map[string]float64     `json:"roi_metrics"`
	LastUpdated      time.Time              `json:"last_updated"`
	Effectiveness    float64                `json:"effectiveness"`
}

// DisplacementTactic defines specific competitive strategy
type DisplacementTactic struct {
	TacticID        string                 `json:"tactic_id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	TargetScenario  string                 `json:"target_scenario"`
	Steps           []string               `json:"steps"`
	Resources       []string               `json:"resources"`
	SuccessRate     float64                `json:"success_rate"`
	AvgTimeToWin    int                    `json:"avg_time_to_win"`
	KeyMessages     []string               `json:"key_messages"`
	Objections      map[string]string      `json:"objections"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// ValueProposition defines competitive advantage
type ValueProposition struct {
	PropID          string   `json:"prop_id"`
	Title           string   `json:"title"`
	Description     string   `json:"description"`
	QuantifiedValue float64  `json:"quantified_value"`
	Metric          string   `json:"metric"`
	ProofPoints     []string `json:"proof_points"`
	Differentiator  string   `json:"differentiator"`
	Priority        string   `json:"priority"`
}

// CompetitorPainPoint identifies competitor weaknesses
type CompetitorPainPoint struct {
	PainPointID   string   `json:"pain_point_id"`
	Category      string   `json:"category"`
	Description   string   `json:"description"`
	Impact        string   `json:"impact"`
	Frequency     int      `json:"frequency"`
	CustomerQuotes []string `json:"customer_quotes"`
	Exploitable   bool     `json:"exploitable"`
}

// CompetitiveBattleCard provides real-time competitive intelligence
type CompetitiveBattleCard struct {
	CardID           string                 `json:"card_id"`
	CompetitorName   string                 `json:"competitor_name"`
	Overview         string                 `json:"overview"`
	MarketPosition   string                 `json:"market_position"`
	MarketShare      float64                `json:"market_share"`
	Strengths        []string               `json:"strengths"`
	Weaknesses       []string               `json:"weaknesses"`
	Pricing          PricingIntelligence    `json:"pricing"`
	Products         []ProductComparison    `json:"products"`
	WhyWeWin         []string               `json:"why_we_win"`
	WhyWeLose        []string               `json:"why_we_lose"`
	KeyAccounts      []string               `json:"key_accounts"`
	RecentWins       []string               `json:"recent_wins"`
	RecentLosses     []string               `json:"recent_losses"`
	TalkTrack        []string               `json:"talk_track"`
	LandMines        []string               `json:"land_mines"`
	IntelligenceSources []string            `json:"intelligence_sources"`
	LastUpdated      time.Time              `json:"last_updated"`
	AutoUpdated      bool                   `json:"auto_updated"`
}

// PricingIntelligence tracks competitor pricing
type PricingIntelligence struct {
	Model           string                 `json:"model"`
	StartingPrice   float64                `json:"starting_price"`
	AveragePrice    float64                `json:"average_price"`
	DiscountRange   string                 `json:"discount_range"`
	HiddenCosts     []string               `json:"hidden_costs"`
	OurAdvantage    string                 `json:"our_advantage"`
	PriceComparison map[string]interface{} `json:"price_comparison"`
}

// ProductComparison compares competitive offerings
type ProductComparison struct {
	ProductName     string            `json:"product_name"`
	Category        string            `json:"category"`
	TheirFeatures   []string          `json:"their_features"`
	OurFeatures     []string          `json:"our_features"`
	GapAnalysis     map[string]string `json:"gap_analysis"`
	OurAdvantages   []string          `json:"our_advantages"`
	TheirAdvantages []string          `json:"their_advantages"`
}

// DisplacementOpportunity tracks individual displacement deals
type DisplacementOpportunity struct {
	OpportunityID      string                 `json:"opportunity_id"`
	AccountName        string                 `json:"account_name"`
	AccountValue       float64                `json:"account_value"`
	CurrentVendor      string                 `json:"current_vendor"`
	DisplacementReason string                 `json:"displacement_reason"`
	Stage              string                 `json:"stage"`
	Probability        float64                `json:"probability"`
	CloseDate          time.Time              `json:"close_date"`
	PlaybookApplied    string                 `json:"playbook_applied"`
	Tactics            []string               `json:"tactics"`
	CompetitiveThreats []string               `json:"competitive_threats"`
	KeyStakeholders    []Stakeholder          `json:"key_stakeholders"`
	RiskFactors        []string               `json:"risk_factors"`
	NextSteps          []string               `json:"next_steps"`
	MigrationPlan      string                 `json:"migration_plan"`
	ROIProjection      float64                `json:"roi_projection"`
	Status             string                 `json:"status"`
	Metadata           map[string]interface{} `json:"metadata"`
}

// Stakeholder represents key decision maker
type Stakeholder struct {
	Name          string   `json:"name"`
	Role          string   `json:"role"`
	Influence     string   `json:"influence"`
	Champion      bool     `json:"champion"`
	Concerns      []string `json:"concerns"`
	EngagementPlan string  `json:"engagement_plan"`
}

// WinLossTracker analyzes competitive outcomes
type WinLossTracker struct {
	totalDeals        int
	wins              int
	losses            int
	winRate           float64
	avgDealSize       float64
	avgSalesCycle     int
	winReasons        map[string]int
	lossReasons       map[string]int
	competitorWinRate map[string]float64
	dealHistory       []DealOutcome
	mu                sync.RWMutex
}

// DealOutcome records deal result
type DealOutcome struct {
	DealID         string    `json:"deal_id"`
	AccountName    string    `json:"account_name"`
	DealValue      float64   `json:"deal_value"`
	Competitor     string    `json:"competitor"`
	Outcome        string    `json:"outcome"`
	Reason         string    `json:"reason"`
	PlaybookUsed   string    `json:"playbook_used"`
	SalesCycleDays int       `json:"sales_cycle_days"`
	CloseDate      time.Time `json:"close_date"`
}

// CompetitiveIntelligence aggregates market intelligence
type CompetitiveIntelligence struct {
	sources          map[string]IntelligenceSource
	intelligenceData map[string]*IntelligenceReport
	autoScraping     bool
	scrapingInterval time.Duration
	mu               sync.RWMutex
}

// IntelligenceSource defines intelligence gathering method
type IntelligenceSource interface {
	GetSourceName() string
	GatherIntelligence(ctx context.Context, competitor string) (*IntelligenceReport, error)
	GetUpdateFrequency() time.Duration
}

// IntelligenceReport contains competitive intelligence
type IntelligenceReport struct {
	ReportID       string                 `json:"report_id"`
	CompetitorName string                 `json:"competitor_name"`
	SourceName     string                 `json:"source_name"`
	Timestamp      time.Time              `json:"timestamp"`
	Updates        []string               `json:"updates"`
	KeyFindings    []string               `json:"key_findings"`
	Vulnerabilities []string              `json:"vulnerabilities"`
	Opportunities  []string               `json:"opportunities"`
	Threats        []string               `json:"threats"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// CustomerMigrationEngine automates migration from competitors
type CustomerMigrationEngine struct {
	migrationPlans map[string]*MigrationPlan
	automationTools map[string]MigrationTool
	successRate    float64
	avgMigrationTime int
	mu             sync.RWMutex
}

// MigrationPlan defines customer migration strategy
type MigrationPlan struct {
	PlanID            string                 `json:"plan_id"`
	SourceVendor      string                 `json:"source_vendor"`
	TargetProduct     string                 `json:"target_product"`
	Phases            []MigrationPhase       `json:"phases"`
	EstimatedDuration int                    `json:"estimated_duration"`
	RiskMitigation    []string               `json:"risk_mitigation"`
	Automation        []string               `json:"automation"`
	SuccessRate       float64                `json:"success_rate"`
	Resources         []string               `json:"resources"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// MigrationPhase defines migration step
type MigrationPhase struct {
	PhaseNumber  int      `json:"phase_number"`
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	Duration     int      `json:"duration"`
	Tasks        []string `json:"tasks"`
	Dependencies []string `json:"dependencies"`
	Automated    bool     `json:"automated"`
	RollbackPlan string   `json:"rollback_plan"`
}

// MigrationTool defines automation capability
type MigrationTool interface {
	GetToolName() string
	CanMigrate(sourceVendor string) bool
	ExecuteMigration(ctx context.Context, plan *MigrationPlan) error
}

// DisplacementROICalculator computes displacement value
type DisplacementROICalculator struct {
	costFactors    map[string]float64
	benefitFactors map[string]float64
	mu             sync.RWMutex
}

// NewDisplacementEngine creates a new displacement engine
func NewDisplacementEngine() *DisplacementEngine {
	return &DisplacementEngine{
		id:                uuid.New().String(),
		playbooks:         make(map[string]*DisplacementPlaybook),
		battleCards:       make(map[string]*CompetitiveBattleCard),
		opportunities:     make(map[string]*DisplacementOpportunity),
		winLossTracker:    NewWinLossTracker(),
		intelligenceHub:   NewCompetitiveIntelligence(),
		migrationEngine:   NewCustomerMigrationEngine(),
		roiCalculator:     NewDisplacementROICalculator(),
		autoUpdateEnabled: true,
		updateInterval:    1 * time.Hour,
	}
}

// NewWinLossTracker creates a new win/loss tracker
func NewWinLossTracker() *WinLossTracker {
	return &WinLossTracker{
		winReasons:        make(map[string]int),
		lossReasons:       make(map[string]int),
		competitorWinRate: make(map[string]float64),
		dealHistory:       make([]DealOutcome, 0),
	}
}

// NewCompetitiveIntelligence creates intelligence hub
func NewCompetitiveIntelligence() *CompetitiveIntelligence {
	return &CompetitiveIntelligence{
		sources:          make(map[string]IntelligenceSource),
		intelligenceData: make(map[string]*IntelligenceReport),
		autoScraping:     true,
		scrapingInterval: 6 * time.Hour,
	}
}

// NewCustomerMigrationEngine creates migration engine
func NewCustomerMigrationEngine() *CustomerMigrationEngine {
	return &CustomerMigrationEngine{
		migrationPlans:  make(map[string]*MigrationPlan),
		automationTools: make(map[string]MigrationTool),
		successRate:     0.94,
		avgMigrationTime: 45,
	}
}

// NewDisplacementROICalculator creates ROI calculator
func NewDisplacementROICalculator() *DisplacementROICalculator {
	return &DisplacementROICalculator{
		costFactors: map[string]float64{
			"sales_effort":      1.0,
			"technical_support": 0.8,
			"migration_cost":    1.2,
			"discount":          0.5,
		},
		benefitFactors: map[string]float64{
			"recurring_revenue": 3.0,
			"expansion_potential": 1.5,
			"reference_value":   0.8,
			"competitive_impact": 1.0,
		},
	}
}

// InitializePlaybooks creates competitive displacement playbooks
func (de *DisplacementEngine) InitializePlaybooks() error {
	de.mu.Lock()
	defer de.mu.Unlock()

	// VMware Displacement Playbook (75% win rate)
	vmwarePlaybook := &DisplacementPlaybook{
		PlaybookID:     "vmware-displacement",
		CompetitorName: "VMware",
		CompetitorType: "legacy_virtualization",
		TargetWinRate:  0.75,
		CurrentWinRate: 0.75,
		Tactics: []DisplacementTactic{
			{
				TacticID:       "vmware-cost-attack",
				Name:           "Total Cost of Ownership Attack",
				Description:    "Highlight VMware licensing complexity and hidden costs",
				TargetScenario: "License renewal or expansion",
				Steps: []string{
					"Conduct TCO analysis showing 60% cost savings",
					"Highlight CPU-based licensing trap",
					"Show per-socket vs per-VM savings",
					"Demonstrate elimination of vSphere, vCenter, NSX costs",
					"Calculate 3-year total savings including support",
				},
				SuccessRate:  0.82,
				AvgTimeToWin: 90,
				KeyMessages: []string{
					"No per-CPU licensing - unlimited scaling",
					"60% lower TCO over 3 years",
					"No vendor lock-in with open standards",
					"Built-in security without NSX costs",
				},
				Objections: map[string]string{
					"vmware_ecosystem": "Our marketplace has 500+ integrations, plus native Kubernetes",
					"maturity":         "10+ years in production, 280+ Fortune 500 customers",
					"support":          "24/7 enterprise support with 99.99% SLA",
				},
			},
			{
				TacticID:       "vmware-cloud-native",
				Name:           "Cloud-Native Migration Path",
				Description:    "Position as modern alternative to legacy VMware",
				TargetScenario: "Digital transformation or cloud migration",
				Steps: []string{
					"Show Kubernetes-native architecture",
					"Demonstrate container and VM support",
					"Highlight multi-cloud portability",
					"Automate migration from vSphere",
					"Provide side-by-side pilot",
				},
				SuccessRate:  0.78,
				AvgTimeToWin: 120,
				KeyMessages: []string{
					"Native Kubernetes with VM support",
					"Multi-cloud without vendor lock-in",
					"Automated vSphere migration",
					"Future-proof cloud-native platform",
				},
			},
		},
		ValueProps: []ValueProposition{
			{
				PropID:          "vmware-cost-savings",
				Title:           "60% Lower TCO",
				Description:     "Eliminate complex per-CPU licensing and hidden costs",
				QuantifiedValue: 60.0,
				Metric:          "percentage_savings",
				ProofPoints: []string{
					"No per-CPU or per-socket fees",
					"Included networking and security",
					"Single management plane",
					"Unlimited scaling within license",
				},
				Differentiator: "Simple consumption-based pricing",
				Priority:       "critical",
			},
			{
				PropID:          "vmware-modernization",
				Title:           "Cloud-Native Platform",
				Description:     "Modern Kubernetes-native architecture vs legacy hypervisor",
				QuantifiedValue: 85.0,
				Metric:          "modernization_score",
				ProofPoints: []string{
					"Native Kubernetes orchestration",
					"Container and VM support",
					"Multi-cloud portability",
					"GitOps-driven operations",
				},
				Differentiator: "Built for cloud-native era",
				Priority:       "high",
			},
		},
		PainPoints: []CompetitorPainPoint{
			{
				PainPointID:  "vmware-licensing",
				Category:     "pricing",
				Description:  "Complex per-CPU licensing with hidden costs",
				Impact:       "60% higher TCO",
				Frequency:    95,
				CustomerQuotes: []string{
					"VMware licensing became our #1 budget concern",
					"Every CPU upgrade means more licenses",
					"We spent 40% of budget just on VMware licenses",
				},
				Exploitable: true,
			},
			{
				PainPointID:  "vmware-cloud-lag",
				Category:     "technology",
				Description:  "Legacy architecture not cloud-native",
				Impact:       "Slows digital transformation",
				Frequency:    78,
				CustomerQuotes: []string{
					"VMware doesn't understand Kubernetes",
					"We needed separate platforms for VMs and containers",
					"Multi-cloud with VMware is painful",
				},
				Exploitable: true,
			},
		},
		CaseStudies: []string{
			"fortune500-bank-vmware-to-novacron",
			"global-telco-vmware-displacement",
			"healthcare-vmware-migration",
		},
		ROIMetrics: map[string]float64{
			"avg_cost_savings": 60.0,
			"migration_time":   45.0,
			"roi_months":       8.0,
		},
	}

	// AWS/Azure/GCP Displacement Playbook (65% win rate)
	cloudPlaybook := &DisplacementPlaybook{
		PlaybookID:     "cloud-repatriation",
		CompetitorName: "AWS/Azure/GCP",
		CompetitorType: "public_cloud",
		TargetWinRate:  0.65,
		CurrentWinRate: 0.65,
		Tactics: []DisplacementTactic{
			{
				TacticID:       "cloud-cost-optimization",
				Name:           "Cloud Cost Repatriation",
				Description:    "Show hybrid cloud savings vs public cloud only",
				TargetScenario: "Rising cloud costs or FinOps initiatives",
				Steps: []string{
					"Analyze current cloud spend",
					"Identify repatriation candidates",
					"Calculate hybrid savings (70% typical)",
					"Design optimal hybrid architecture",
					"Provide migration automation",
				},
				SuccessRate:  0.72,
				AvgTimeToWin: 150,
				KeyMessages: []string{
					"70% cost reduction for predictable workloads",
					"Keep cloud for bursting and DR",
					"Unified management across hybrid",
					"Avoid egress fees with edge processing",
				},
			},
			{
				TacticID:       "cloud-data-sovereignty",
				Name:           "Data Sovereignty and Compliance",
				Description:    "Address regulatory and data residency concerns",
				TargetScenario: "Regulated industries or geo-specific requirements",
				Steps: []string{
					"Identify compliance requirements",
					"Show on-premises data control",
					"Demonstrate audit capabilities",
					"Provide regulatory certifications",
					"Design compliant architecture",
				},
				SuccessRate:  0.68,
				AvgTimeToWin: 120,
				KeyMessages: []string{
					"Complete data sovereignty",
					"Air-gapped deployment option",
					"SOC2, ISO 27001, HIPAA, PCI-DSS",
					"Zero trust security architecture",
				},
			},
		},
		ValueProps: []ValueProposition{
			{
				PropID:          "cloud-cost-savings",
				Title:           "70% Hybrid Cloud Savings",
				Description:     "Optimize workload placement for cost efficiency",
				QuantifiedValue: 70.0,
				Metric:          "percentage_savings",
				ProofPoints: []string{
					"On-premises for predictable workloads",
					"Cloud for bursting and DR",
					"Eliminate egress fees",
					"Unified cost management",
				},
				Differentiator: "Best of both worlds",
				Priority:       "critical",
			},
		},
		PainPoints: []CompetitorPainPoint{
			{
				PainPointID:  "cloud-cost-overrun",
				Category:     "cost",
				Description:  "Unpredictable and escalating cloud costs",
				Impact:       "Budget overruns",
				Frequency:    88,
				CustomerQuotes: []string{
					"Our AWS bill tripled in 2 years",
					"Egress fees alone cost us $50K/month",
					"Cloud was supposed to save money",
				},
				Exploitable: true,
			},
		},
	}

	// Kubernetes Displacement Playbook (85% win rate)
	k8sPlaybook := &DisplacementPlaybook{
		PlaybookID:     "kubernetes-simplification",
		CompetitorName: "DIY Kubernetes",
		CompetitorType: "open_source_complexity",
		TargetWinRate:  0.85,
		CurrentWinRate: 0.85,
		Tactics: []DisplacementTactic{
			{
				TacticID:       "k8s-operational-burden",
				Name:           "Eliminate K8s Operational Complexity",
				Description:    "Show managed platform vs DIY maintenance burden",
				TargetScenario: "Organizations struggling with K8s operations",
				Steps: []string{
					"Audit current K8s operational costs",
					"Calculate FTE burden (3-5 engineers typical)",
					"Show automated lifecycle management",
					"Demonstrate integrated observability",
					"Provide migration path",
				},
				SuccessRate:  0.88,
				AvgTimeToWin: 60,
				KeyMessages: []string{
					"Eliminate 3-5 FTE maintenance burden",
					"Automated upgrades and patching",
					"Integrated security and compliance",
					"Multi-cluster management",
				},
			},
		},
		ValueProps: []ValueProposition{
			{
				PropID:          "k8s-simplification",
				Title:           "Eliminate K8s Complexity",
				Description:     "Fully managed platform vs DIY maintenance",
				QuantifiedValue: 80.0,
				Metric:          "operational_reduction",
				ProofPoints: []string{
					"Automated lifecycle management",
					"Integrated security and observability",
					"Multi-cluster orchestration",
					"Self-healing infrastructure",
				},
				Differentiator: "Zero operational burden",
				Priority:       "critical",
			},
		},
	}

	de.playbooks["vmware-displacement"] = vmwarePlaybook
	de.playbooks["cloud-repatriation"] = cloudPlaybook
	de.playbooks["kubernetes-simplification"] = k8sPlaybook

	return nil
}

// CreateBattleCard generates competitive battle card
func (de *DisplacementEngine) CreateBattleCard(ctx context.Context, competitor string) (*CompetitiveBattleCard, error) {
	de.mu.Lock()
	defer de.mu.Unlock()

	// Generate battle card based on competitor
	var battleCard *CompetitiveBattleCard

	switch competitor {
	case "VMware":
		battleCard = &CompetitiveBattleCard{
			CardID:         "vmware-battle-card",
			CompetitorName: "VMware",
			Overview:       "Legacy virtualization vendor transitioning to cloud",
			MarketPosition: "Declining leader in traditional virtualization",
			MarketShare:    22.0,
			Strengths: []string{
				"Large installed base",
				"Mature ecosystem",
				"Enterprise relationships",
				"Broad product portfolio",
			},
			Weaknesses: []string{
				"Complex and expensive licensing",
				"Legacy architecture not cloud-native",
				"Broadcom acquisition uncertainty",
				"Limited Kubernetes expertise",
				"High TCO vs modern alternatives",
			},
			Pricing: PricingIntelligence{
				Model:         "Per-CPU perpetual + support",
				StartingPrice: 50000,
				AveragePrice:  250000,
				DiscountRange: "20-40% for large deals",
				HiddenCosts: []string{
					"Per-CPU licensing traps",
					"vCenter licensing",
					"NSX for networking",
					"vSAN for storage",
					"Annual support (20-25%)",
				},
				OurAdvantage: "60% lower TCO with simple consumption pricing",
			},
			WhyWeWin: []string{
				"60% lower TCO",
				"Cloud-native Kubernetes architecture",
				"No vendor lock-in",
				"Faster time to market",
				"Modern DevOps workflows",
				"Multi-cloud portability",
			},
			WhyWeLose: []string{
				"Existing VMware investment",
				"Change management resistance",
				"VMware-specific integrations",
				"Perceived migration risk",
			},
			KeyAccounts: []string{
				"Fortune 500 banks",
				"Global telcos",
				"Large healthcare systems",
				"Enterprise retailers",
			},
			TalkTrack: []string{
				"We help enterprises modernize beyond legacy VMware",
				"Kubernetes-native with VM support for migration",
				"60% TCO savings without sacrificing features",
				"Proven migration automation from vSphere",
			},
			LandMines: []string{
				"Don't criticize existing VMware investment",
				"Acknowledge ecosystem maturity",
				"Avoid 'rip and replace' language",
				"Focus on modernization path",
			},
			LastUpdated: time.Now(),
			AutoUpdated: true,
		}

	case "AWS":
		battleCard = &CompetitiveBattleCard{
			CardID:         "aws-battle-card",
			CompetitorName: "AWS",
			Overview:       "Dominant public cloud provider",
			MarketPosition: "Public cloud leader",
			MarketShare:    33.0,
			Strengths: []string{
				"Largest cloud ecosystem",
				"Broadest service portfolio",
				"Global infrastructure",
				"Market leader position",
			},
			Weaknesses: []string{
				"Complex pricing with hidden costs",
				"Vendor lock-in",
				"High egress fees",
				"Limited hybrid cloud",
				"Cost overruns common",
			},
			Pricing: PricingIntelligence{
				Model:         "Pay-as-you-go with commitments",
				StartingPrice: 0,
				AveragePrice:  150000,
				DiscountRange: "30-50% with EDP commitments",
				HiddenCosts: []string{
					"Data egress fees",
					"Cross-AZ transfer",
					"Premium support",
					"Proprietary service costs",
					"Unexpected usage spikes",
				},
				OurAdvantage: "70% savings for hybrid workloads, no egress fees",
			},
			WhyWeWin: []string{
				"70% cost savings for hybrid",
				"Data sovereignty and control",
				"No vendor lock-in",
				"Predictable costs",
				"On-premises performance",
			},
			WhyWeLose: []string{
				"AWS ecosystem breadth",
				"Existing AWS commitment",
				"Cloud-native applications",
				"Developer familiarity",
			},
		}
	}

	if battleCard != nil {
		de.battleCards[battleCard.CardID] = battleCard
	}

	return battleCard, nil
}

// TrackOpportunity adds displacement opportunity
func (de *DisplacementEngine) TrackOpportunity(ctx context.Context, opp *DisplacementOpportunity) error {
	de.mu.Lock()
	defer de.mu.Unlock()

	de.opportunities[opp.OpportunityID] = opp

	// Calculate ROI projection
	roi, err := de.roiCalculator.CalculateROI(opp)
	if err == nil {
		opp.ROIProjection = roi
	}

	return nil
}

// RecordDealOutcome logs win/loss for analysis
func (de *DisplacementEngine) RecordDealOutcome(ctx context.Context, outcome *DealOutcome) error {
	return de.winLossTracker.RecordOutcome(outcome)
}

// RecordOutcome adds deal outcome to tracker
func (wlt *WinLossTracker) RecordOutcome(outcome *DealOutcome) error {
	wlt.mu.Lock()
	defer wlt.mu.Unlock()

	wlt.totalDeals++
	wlt.dealHistory = append(wlt.dealHistory, *outcome)

	if outcome.Outcome == "won" {
		wlt.wins++
		wlt.winReasons[outcome.Reason]++
	} else {
		wlt.losses++
		wlt.lossReasons[outcome.Reason]++
	}

	// Update win rate
	wlt.winRate = (float64(wlt.wins) / float64(wlt.totalDeals)) * 100

	// Update competitor-specific win rate
	competitorKey := outcome.Competitor
	competitorWins := 0
	competitorTotal := 0

	for _, deal := range wlt.dealHistory {
		if deal.Competitor == competitorKey {
			competitorTotal++
			if deal.Outcome == "won" {
				competitorWins++
			}
		}
	}

	if competitorTotal > 0 {
		wlt.competitorWinRate[competitorKey] = (float64(competitorWins) / float64(competitorTotal)) * 100
	}

	return nil
}

// GetWinLossAnalysis returns comprehensive win/loss data
func (de *DisplacementEngine) GetWinLossAnalysis() map[string]interface{} {
	de.winLossTracker.mu.RLock()
	defer de.winLossTracker.mu.RUnlock()

	return map[string]interface{}{
		"total_deals":         de.winLossTracker.totalDeals,
		"wins":                de.winLossTracker.wins,
		"losses":              de.winLossTracker.losses,
		"win_rate":            de.winLossTracker.winRate,
		"win_reasons":         de.winLossTracker.winReasons,
		"loss_reasons":        de.winLossTracker.lossReasons,
		"competitor_win_rate": de.winLossTracker.competitorWinRate,
		"recent_outcomes":     de.winLossTracker.dealHistory[max(0, len(de.winLossTracker.dealHistory)-10):],
	}
}

// CalculateROI computes displacement opportunity ROI
func (calc *DisplacementROICalculator) CalculateROI(opp *DisplacementOpportunity) (float64, error) {
	calc.mu.RLock()
	defer calc.mu.RUnlock()

	// Calculate costs
	totalCost := opp.AccountValue * (calc.costFactors["sales_effort"] +
		calc.costFactors["technical_support"] +
		calc.costFactors["migration_cost"] +
		calc.costFactors["discount"])

	// Calculate benefits
	totalBenefit := opp.AccountValue * (calc.benefitFactors["recurring_revenue"] +
		calc.benefitFactors["expansion_potential"] +
		calc.benefitFactors["reference_value"] +
		calc.benefitFactors["competitive_impact"])

	// ROI = (Benefit - Cost) / Cost * 100
	if totalCost == 0 {
		return 0, fmt.Errorf("zero cost")
	}

	roi := ((totalBenefit - totalCost) / totalCost) * 100

	return roi, nil
}

// GetPlaybookEffectiveness returns playbook performance metrics
func (de *DisplacementEngine) GetPlaybookEffectiveness() map[string]interface{} {
	de.mu.RLock()
	defer de.mu.RUnlock()

	effectiveness := make(map[string]interface{})

	for id, playbook := range de.playbooks {
		effectiveness[id] = map[string]interface{}{
			"competitor":        playbook.CompetitorName,
			"target_win_rate":   playbook.TargetWinRate,
			"current_win_rate":  playbook.CurrentWinRate,
			"total_opportunities": playbook.TotalOpportunities,
			"won_deals":         playbook.WonDeals,
			"active_deals":      playbook.ActiveDeals,
			"effectiveness":     playbook.Effectiveness,
		}
	}

	return effectiveness
}

// ExportDisplacementMetrics exports comprehensive displacement data
func (de *DisplacementEngine) ExportDisplacementMetrics() ([]byte, error) {
	de.mu.RLock()
	defer de.mu.RUnlock()

	metrics := map[string]interface{}{
		"engine_id":       de.id,
		"playbooks":       de.playbooks,
		"battle_cards":    de.battleCards,
		"opportunities":   de.opportunities,
		"win_loss":        de.GetWinLossAnalysis(),
		"effectiveness":   de.GetPlaybookEffectiveness(),
		"timestamp":       time.Now(),
	}

	return json.MarshalIndent(metrics, "", "  ")
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
