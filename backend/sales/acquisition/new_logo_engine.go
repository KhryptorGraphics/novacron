// Package acquisition provides new logo acquisition and Fortune 500 targeting
package acquisition

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// NewLogoEngine drives Fortune 500 acquisition
type NewLogoEngine struct {
	mu                  sync.RWMutex
	prospects           map[string]*Prospect
	accounts            map[string]*StrategicAccount
	campaigns           map[string]*AcquisitionCampaign
	pipeline            *SalesPipeline
	playbooks           map[string]*SalesPlaybook
	competitiveIntel    *CompetitiveIntelligence
	automationEngine    *SalesAutomation
	velocityOptimizer   *VelocityOptimizer
	metrics             *AcquisitionMetrics
	config              AcquisitionConfig
}

// Prospect represents potential customer
type Prospect struct {
	ID                  string                 `json:"id"`
	CompanyName         string                 `json:"company_name"`
	Domain              string                 `json:"domain"`
	Industry            string                 `json:"industry"`
	Size                string                 `json:"size"`                // Fortune 500 rank
	Revenue             float64                `json:"revenue"`
	Employees           int                    `json:"employees"`
	Headquarters        string                 `json:"headquarters"`
	IsFortune500        bool                   `json:"is_fortune_500"`
	Fortune500Rank      int                    `json:"fortune_500_rank"`
	Status              string                 `json:"status"`              // prospecting, qualified, engaged
	Score               float64                `json:"score"`               // 0-100
	FitScore            float64                `json:"fit_score"`           // 0-100
	IntentScore         float64                `json:"intent_score"`        // 0-100
	EstimatedARR        float64                `json:"estimated_arr"`
	Probability         float64                `json:"probability"`         // 0-1
	ExpectedValue       float64                `json:"expected_value"`
	Contacts            []Contact              `json:"contacts"`
	Technologies        []Technology           `json:"technologies"`
	Competitors         []string               `json:"competitors"`        // Current vendors
	BuyingSignals       []BuyingSignal         `json:"buying_signals"`
	PainPoints          []PainPoint            `json:"pain_points"`
	Timeline            ProspectTimeline       `json:"timeline"`
	CreatedAt           time.Time              `json:"created_at"`
	UpdatedAt           time.Time              `json:"updated_at"`
}

// Contact represents decision maker
type Contact struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Email               string                 `json:"email"`
	Title               string                 `json:"title"`
	Department          string                 `json:"department"`
	Role                string                 `json:"role"`                // champion, influencer, decision-maker, blocker
	Seniority           string                 `json:"seniority"`
	InfluenceLevel      int                    `json:"influence_level"`     // 1-10
	Engagement          EngagementMetrics      `json:"engagement"`
	Sentiment           string                 `json:"sentiment"`
	LinkedInURL         string                 `json:"linkedin_url"`
	Phone               string                 `json:"phone"`
	LastContact         time.Time              `json:"last_contact"`
}

// EngagementMetrics tracks contact engagement
type EngagementMetrics struct {
	EmailOpens          int                    `json:"email_opens"`
	EmailClicks         int                    `json:"email_clicks"`
	Meetings            int                    `json:"meetings"`
	WebsiteVisits       int                    `json:"website_visits"`
	ContentDownloads    int                    `json:"content_downloads"`
	ResponseRate        float64                `json:"response_rate"`
	EngagementScore     float64                `json:"engagement_score"`    // 0-100
	LastActivity        time.Time              `json:"last_activity"`
}

// Technology tracks prospect tech stack
type Technology struct {
	Name                string                 `json:"name"`
	Category            string                 `json:"category"`
	Vendor              string                 `json:"vendor"`
	IsCompetitor        bool                   `json:"is_competitor"`
	ReplacementOpportunity bool                `json:"replacement_opportunity"`
	ContractExpiry      *time.Time             `json:"contract_expiry,omitempty"`
}

// BuyingSignal indicates purchase intent
type BuyingSignal struct {
	Type                string                 `json:"type"`
	Source              string                 `json:"source"`
	Strength            float64                `json:"strength"`            // 0-1
	Description         string                 `json:"description"`
	DetectedAt          time.Time              `json:"detected_at"`
	DataPoints          map[string]interface{} `json:"data_points"`
}

// PainPoint identifies customer challenges
type PainPoint struct {
	Category            string                 `json:"category"`
	Description         string                 `json:"description"`
	Severity            string                 `json:"severity"`            // critical, high, medium, low
	Impact              string                 `json:"impact"`
	OurSolution         string                 `json:"our_solution"`
	CompetitorSolution  string                 `json:"competitor_solution"`
	Priority            int                    `json:"priority"`
}

// ProspectTimeline tracks sales cycle
type ProspectTimeline struct {
	IdentifiedDate      time.Time              `json:"identified_date"`
	QualifiedDate       *time.Time             `json:"qualified_date,omitempty"`
	FirstContact        *time.Time             `json:"first_contact,omitempty"`
	DemoDate            *time.Time             `json:"demo_date,omitempty"`
	ProposalDate        *time.Time             `json:"proposal_date,omitempty"`
	NegotiationDate     *time.Time             `json:"negotiation_date,omitempty"`
	ClosedDate          *time.Time             `json:"closed_date,omitempty"`
	ExpectedCloseDate   time.Time              `json:"expected_close_date"`
	SalesCycleLength    int                    `json:"sales_cycle_length"`  // Current days
	TargetCycleLength   int                    `json:"target_cycle_length"` // 120 days target
	VelocityScore       float64                `json:"velocity_score"`
}

// StrategicAccount manages Fortune 500 accounts
type StrategicAccount struct {
	ID                  string                 `json:"id"`
	ProspectID          string                 `json:"prospect_id"`
	CompanyName         string                 `json:"company_name"`
	Fortune500Rank      int                    `json:"fortune_500_rank"`
	AccountTier         string                 `json:"account_tier"`        // strategic, enterprise, mid-market
	TargetARR           float64                `json:"target_arr"`          // $5M+ for Fortune 500
	Opportunities       []Opportunity          `json:"opportunities"`
	KeyStakeholders     []Contact              `json:"key_stakeholders"`
	AccountPlan         AccountPlan            `json:"account_plan"`
	CompetitiveLandscape CompetitiveAnalysis   `json:"competitive_landscape"`
	ExecutiveSponsor    string                 `json:"executive_sponsor"`
	AccountManager      string                 `json:"account_manager"`
	SalesEngineer       string                 `json:"sales_engineer"`
	Status              string                 `json:"status"`
	HealthScore         float64                `json:"health_score"`
	CreatedAt           time.Time              `json:"created_at"`
	LastActivity        time.Time              `json:"last_activity"`
}

// Opportunity represents sales opportunity
type Opportunity struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	AccountID           string                 `json:"account_id"`
	Stage               string                 `json:"stage"`
	Amount              float64                `json:"amount"`
	Probability         float64                `json:"probability"`
	ExpectedCloseDate   time.Time              `json:"expected_close_date"`
	Products            []ProductLine          `json:"products"`
	CompetitorInfo      []CompetitorInfo       `json:"competitor_info"`
	NextSteps           []NextStep             `json:"next_steps"`
	RiskFactors         []RiskFactor           `json:"risk_factors"`
	SuccessFactors      []SuccessFactor        `json:"success_factors"`
	CreatedAt           time.Time              `json:"created_at"`
	UpdatedAt           time.Time              `json:"updated_at"`
}

// ProductLine represents products in deal
type ProductLine struct {
	ProductID           string                 `json:"product_id"`
	ProductName         string                 `json:"product_name"`
	Quantity            int                    `json:"quantity"`
	UnitPrice           float64                `json:"unit_price"`
	Discount            float64                `json:"discount"`
	TotalAmount         float64                `json:"total_amount"`
	ARR                 float64                `json:"arr"`
}

// CompetitorInfo tracks competitive situation
type CompetitorInfo struct {
	Competitor          string                 `json:"competitor"`
	Status              string                 `json:"status"`              // incumbent, evaluating, eliminated
	Strengths           []string               `json:"strengths"`
	Weaknesses          []string               `json:"weaknesses"`
	Pricing             float64                `json:"pricing"`
	LikelyWinner        bool                   `json:"likely_winner"`
}

// NextStep defines sales actions
type NextStep struct {
	Description         string                 `json:"description"`
	Owner               string                 `json:"owner"`
	DueDate             time.Time              `json:"due_date"`
	Status              string                 `json:"status"`
	Outcome             string                 `json:"outcome"`
}

// RiskFactor identifies deal risks
type RiskFactor struct {
	Risk                string                 `json:"risk"`
	Impact              string                 `json:"impact"`              // high, medium, low
	Probability         string                 `json:"probability"`
	Mitigation          string                 `json:"mitigation"`
	Status              string                 `json:"status"`
}

// SuccessFactor identifies deal strengths
type SuccessFactor struct {
	Factor              string                 `json:"factor"`
	Impact              string                 `json:"impact"`
	Confidence          float64                `json:"confidence"`
}

// AccountPlan defines account strategy
type AccountPlan struct {
	Objectives          []string               `json:"objectives"`
	Strategy            string                 `json:"strategy"`
	ValueProposition    string                 `json:"value_proposition"`
	KeyMessages         []string               `json:"key_messages"`
	Milestones          []Milestone            `json:"milestones"`
	Budget              float64                `json:"budget"`
	Resources           []string               `json:"resources"`
	Timeline            int                    `json:"timeline"`            // Days
	SuccessMetrics      []string               `json:"success_metrics"`
}

// Milestone tracks account progress
type Milestone struct {
	Name                string                 `json:"name"`
	TargetDate          time.Time              `json:"target_date"`
	Status              string                 `json:"status"`
	Activities          []string               `json:"activities"`
	Completed           bool                   `json:"completed"`
	CompletedDate       *time.Time             `json:"completed_date,omitempty"`
}

// CompetitiveAnalysis tracks competitive landscape
type CompetitiveAnalysis struct {
	PrimaryCompetitor   string                 `json:"primary_competitor"`
	Competitors         []CompetitorInfo       `json:"competitors"`
	OurPosition         string                 `json:"our_position"`
	Differentiators     []string               `json:"differentiators"`
	CompetitiveRisks    []string               `json:"competitive_risks"`
	WinStrategy         string                 `json:"win_strategy"`
}

// AcquisitionCampaign manages sales campaigns
type AcquisitionCampaign struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Type                string                 `json:"type"`
	TargetSegment       TargetSegment          `json:"target_segment"`
	TargetAccounts      int                    `json:"target_accounts"`
	TargetARR           float64                `json:"target_arr"`
	ActualARR           float64                `json:"actual_arr"`
	Pipeline            float64                `json:"pipeline"`
	ProspectCount       int                    `json:"prospect_count"`
	QualifiedCount      int                    `json:"qualified_count"`
	OpportunityCount    int                    `json:"opportunity_count"`
	ClosedWon           int                    `json:"closed_won"`
	ConversionRate      float64                `json:"conversion_rate"`
	AvgDealSize         float64                `json:"avg_deal_size"`
	AvgSalesCycle       int                    `json:"avg_sales_cycle"`
	StartDate           time.Time              `json:"start_date"`
	EndDate             time.Time              `json:"end_date"`
	Budget              float64                `json:"budget"`
	Spend               float64                `json:"spend"`
	ROI                 float64                `json:"roi"`
	Status              string                 `json:"status"`
}

// TargetSegment defines campaign target
type TargetSegment struct {
	Industries          []string               `json:"industries"`
	CompanySize         []string               `json:"company_size"`
	RevenueRange        [2]float64             `json:"revenue_range"`
	Geographies         []string               `json:"geographies"`
	Technologies        []string               `json:"technologies"`
	Fortune500Only      bool                   `json:"fortune_500_only"`
}

// SalesPipeline tracks sales funnel
type SalesPipeline struct {
	mu                  sync.RWMutex
	stages              map[string]*PipelineStage
	totalValue          float64
	weightedValue       float64
	forecast            float64
	velocity            PipelineVelocity
}

// PipelineStage represents funnel stage
type PipelineStage struct {
	Name                string                 `json:"name"`
	Order               int                    `json:"order"`
	OpportunityCount    int                    `json:"opportunity_count"`
	TotalValue          float64                `json:"total_value"`
	Probability         float64                `json:"probability"`
	WeightedValue       float64                `json:"weighted_value"`
	AvgDaysInStage      int                    `json:"avg_days_in_stage"`
	ConversionRate      float64                `json:"conversion_rate"`
}

// PipelineVelocity tracks sales momentum
type PipelineVelocity struct {
	DealsCreated        int                    `json:"deals_created"`
	DealsAdvanced       int                    `json:"deals_advanced"`
	DealsStalled        int                    `json:"deals_stalled"`
	DealsClosed         int                    `json:"deals_closed"`
	AvgTimeToClose      int                    `json:"avg_time_to_close"`    // Days
	VelocityScore       float64                `json:"velocity_score"`
	Trend               string                 `json:"trend"`
}

// SalesPlaybook defines sales strategies
type SalesPlaybook struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	TargetSegment       string                 `json:"target_segment"`
	Objectives          []string               `json:"objectives"`
	Methodology         string                 `json:"methodology"`
	Phases              []PlaybookPhase        `json:"phases"`
	Tactics             []SalesTactic          `json:"tactics"`
	Objections          []ObjectionHandling    `json:"objections"`
	SuccessRate         float64                `json:"success_rate"`
	AvgDealSize         float64                `json:"avg_deal_size"`
	AvgSalesCycle       int                    `json:"avg_sales_cycle"`
	Enabled             bool                   `json:"enabled"`
}

// PlaybookPhase represents sales stage
type PlaybookPhase struct {
	Name                string                 `json:"name"`
	Duration            int                    `json:"duration"`             // Days
	Objectives          []string               `json:"objectives"`
	Activities          []Activity             `json:"activities"`
	Deliverables        []string               `json:"deliverables"`
	ExitCriteria        []string               `json:"exit_criteria"`
}

// Activity defines sales task
type Activity struct {
	Name                string                 `json:"name"`
	Type                string                 `json:"type"`
	Description         string                 `json:"description"`
	Owner               string                 `json:"owner"`
	Duration            int                    `json:"duration"`             // Minutes
	Automated           bool                   `json:"automated"`
}

// SalesTactic represents sales approach
type SalesTactic struct {
	Name                string                 `json:"name"`
	Description         string                 `json:"description"`
	WhenToUse           string                 `json:"when_to_use"`
	Steps               []string               `json:"steps"`
	SuccessRate         float64                `json:"success_rate"`
	Examples            []string               `json:"examples"`
}

// ObjectionHandling provides responses
type ObjectionHandling struct {
	Objection           string                 `json:"objection"`
	Category            string                 `json:"category"`
	Frequency           string                 `json:"frequency"`
	Response            string                 `json:"response"`
	ProofPoints         []string               `json:"proof_points"`
	Alternatives        []string               `json:"alternatives"`
}

// CompetitiveIntelligence tracks competitors
type CompetitiveIntelligence struct {
	mu                  sync.RWMutex
	competitors         map[string]*Competitor
	battleCards         map[string]*BattleCard
	winLossAnalysis     []WinLossRecord
}

// Competitor represents competitive vendor
type Competitor struct {
	Name                string                 `json:"name"`
	MarketShare         float64                `json:"market_share"`
	Strengths           []string               `json:"strengths"`
	Weaknesses          []string               `json:"weaknesses"`
	Pricing             PricingInfo            `json:"pricing"`
	RecentNews          []NewsItem             `json:"recent_news"`
	CustomerSentiment   float64                `json:"customer_sentiment"`
	DisplacementRate    float64                `json:"displacement_rate"`
}

// BattleCard provides competitive guidance
type BattleCard struct {
	Competitor          string                 `json:"competitor"`
	Overview            string                 `json:"overview"`
	KeyDifferentiators  []string               `json:"key_differentiators"`
	TheirStrengths      []string               `json:"their_strengths"`
	TheirWeaknesses     []string               `json:"their_weaknesses"`
	TalkTrack           []string               `json:"talk_track"`
	TrapQuestions       []string               `json:"trap_questions"`
	ProofPoints         []string               `json:"proof_points"`
	CaseStudies         []string               `json:"case_studies"`
}

// PricingInfo tracks competitor pricing
type PricingInfo struct {
	StartingPrice       float64                `json:"starting_price"`
	AveragePrice        float64                `json:"average_price"`
	PricingModel        string                 `json:"pricing_model"`
	DiscountStrategy    string                 `json:"discount_strategy"`
	LastUpdated         time.Time              `json:"last_updated"`
}

// NewsItem tracks competitive intelligence
type NewsItem struct {
	Title               string                 `json:"title"`
	Source              string                 `json:"source"`
	Date                time.Time              `json:"date"`
	Impact              string                 `json:"impact"`
	Summary             string                 `json:"summary"`
}

// WinLossRecord analyzes deals
type WinLossRecord struct {
	OpportunityID       string                 `json:"opportunity_id"`
	Outcome             string                 `json:"outcome"`            // won, lost
	CompanyName         string                 `json:"company_name"`
	DealSize            float64                `json:"deal_size"`
	SalesCycle          int                    `json:"sales_cycle"`
	PrimaryReason       string                 `json:"primary_reason"`
	CompetitorChosen    string                 `json:"competitor_chosen"`
	Feedback            string                 `json:"feedback"`
	Insights            []string               `json:"insights"`
	Date                time.Time              `json:"date"`
}

// SalesAutomation handles automation
type SalesAutomation struct {
	mu                  sync.RWMutex
	workflows           map[string]*Workflow
	sequences           map[string]*EmailSequence
	scoringModels       map[string]*ScoringModel
}

// Workflow defines automation flow
type Workflow struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Trigger             WorkflowTrigger        `json:"trigger"`
	Steps               []WorkflowStep         `json:"steps"`
	Enabled             bool                   `json:"enabled"`
	ExecutionCount      int64                  `json:"execution_count"`
	SuccessRate         float64                `json:"success_rate"`
}

// WorkflowTrigger defines when workflow runs
type WorkflowTrigger struct {
	Type                string                 `json:"type"`
	Event               string                 `json:"event"`
	Conditions          []TriggerCondition     `json:"conditions"`
}

// TriggerCondition evaluates trigger
type TriggerCondition struct {
	Field               string                 `json:"field"`
	Operator            string                 `json:"operator"`
	Value               interface{}            `json:"value"`
}

// WorkflowStep defines automation action
type WorkflowStep struct {
	Order               int                    `json:"order"`
	Type                string                 `json:"type"`
	Action              string                 `json:"action"`
	Parameters          map[string]interface{} `json:"parameters"`
	DelayMinutes        int                    `json:"delay_minutes"`
}

// EmailSequence manages outreach
type EmailSequence struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	TargetSegment       string                 `json:"target_segment"`
	Emails              []SequenceEmail        `json:"emails"`
	Enabled             bool                   `json:"enabled"`
	OpenRate            float64                `json:"open_rate"`
	ClickRate           float64                `json:"click_rate"`
	ResponseRate        float64                `json:"response_rate"`
	MeetingBookedRate   float64                `json:"meeting_booked_rate"`
}

// SequenceEmail represents email in sequence
type SequenceEmail struct {
	Day                 int                    `json:"day"`
	Subject             string                 `json:"subject"`
	Body                string                 `json:"body"`
	Type                string                 `json:"type"`                // initial, follow-up, breakup
}

// ScoringModel defines lead scoring
type ScoringModel struct {
	Name                string                 `json:"name"`
	Factors             []ScoringFactor        `json:"factors"`
	Threshold           float64                `json:"threshold"`
	Enabled             bool                   `json:"enabled"`
}

// ScoringFactor contributes to score
type ScoringFactor struct {
	Name                string                 `json:"name"`
	Weight              float64                `json:"weight"`
	DataSource          string                 `json:"data_source"`
}

// VelocityOptimizer reduces sales cycle
type VelocityOptimizer struct {
	mu                  sync.RWMutex
	optimizations       []VelocityOptimization
	benchmarks          SalesCycleBenchmarks
	bottlenecks         []SalesBottleneck
}

// VelocityOptimization defines improvement
type VelocityOptimization struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	TargetStage         string                 `json:"target_stage"`
	CurrentDuration     int                    `json:"current_duration"`
	TargetDuration      int                    `json:"target_duration"`
	Improvement         float64                `json:"improvement"`        // Percentage
	Tactics             []string               `json:"tactics"`
	Status              string                 `json:"status"`
}

// SalesCycleBenchmarks provides targets
type SalesCycleBenchmarks struct {
	Overall             int                    `json:"overall"`            // 120 days target
	ByStage             map[string]int         `json:"by_stage"`
	BySegment           map[string]int         `json:"by_segment"`
	ByDealSize          map[string]int         `json:"by_deal_size"`
	Industry            float64                `json:"industry"`
	TopPerformers       int                    `json:"top_performers"`
}

// SalesBottleneck identifies delays
type SalesBottleneck struct {
	Stage               string                 `json:"stage"`
	AvgDelay            int                    `json:"avg_delay"`
	Frequency           int                    `json:"frequency"`
	RootCauses          []string               `json:"root_causes"`
	Impact              float64                `json:"impact"`
	Solutions           []string               `json:"solutions"`
}

// AcquisitionMetrics tracks performance
type AcquisitionMetrics struct {
	mu                  sync.RWMutex
	TotalProspects      int64                  `json:"total_prospects"`
	Fortune500Prospects int64                  `json:"fortune_500_prospects"`
	QualifiedProspects  int64                  `json:"qualified_prospects"`
	ActiveOpportunities int64                  `json:"active_opportunities"`
	TotalPipeline       float64                `json:"total_pipeline"`
	ClosedWon           int64                  `json:"closed_won"`
	TotalARR            float64                `json:"total_arr"`
	AvgDealSize         float64                `json:"avg_deal_size"`
	AvgSalesCycle       int                    `json:"avg_sales_cycle"`
	WinRate             float64                `json:"win_rate"`
	VelocityImprovement float64                `json:"velocity_improvement"`
}

// AcquisitionConfig configures engine
type AcquisitionConfig struct {
	TargetFortune500    int                    `json:"target_fortune_500"`   // 350
	TargetNewARR        float64                `json:"target_new_arr"`       // $300M
	MinDealSize         float64                `json:"min_deal_size"`        // $5M
	TargetSalesCycle    int                    `json:"target_sales_cycle"`   // 120 days
	EnableAutomation    bool                   `json:"enable_automation"`
	EnableAIScoring     bool                   `json:"enable_ai_scoring"`
}

// NewNewLogoEngine creates acquisition engine
func NewNewLogoEngine(config AcquisitionConfig) *NewLogoEngine {
	return &NewLogoEngine{
		prospects:        make(map[string]*Prospect),
		accounts:         make(map[string]*StrategicAccount),
		campaigns:        make(map[string]*AcquisitionCampaign),
		pipeline:         initializePipeline(),
		playbooks:        initializeSalesPlaybooks(),
		competitiveIntel: initializeCompetitiveIntel(),
		automationEngine: initializeSalesAutomation(),
		velocityOptimizer: initializeVelocityOptimizer(),
		metrics:          &AcquisitionMetrics{},
		config:           config,
	}
}

// AddProspect adds new prospect
func (e *NewLogoEngine) AddProspect(ctx context.Context, prospect *Prospect) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Score prospect
	prospect.FitScore = e.calculateFitScore(prospect)
	prospect.IntentScore = e.calculateIntentScore(prospect)
	prospect.Score = (prospect.FitScore + prospect.IntentScore) / 2

	// Estimate ARR
	prospect.EstimatedARR = e.estimateARR(prospect)
	prospect.Probability = e.calculateProbability(prospect)
	prospect.ExpectedValue = prospect.EstimatedARR * prospect.Probability

	// Store prospect
	e.prospects[prospect.ID] = prospect

	// Update metrics
	e.metrics.mu.Lock()
	e.metrics.TotalProspects++
	if prospect.IsFortune500 {
		e.metrics.Fortune500Prospects++
	}
	e.metrics.mu.Unlock()

	return nil
}

// Helper functions
func (e *NewLogoEngine) calculateFitScore(p *Prospect) float64 {
	score := 50.0

	// Fortune 500 boost
	if p.IsFortune500 {
		score += 30
	}

	// Revenue fit
	if p.Revenue > 10_000_000_000 {
		score += 20
	}

	return math.Min(100, score)
}

func (e *NewLogoEngine) calculateIntentScore(p *Prospect) float64 {
	score := 30.0

	// Buying signals
	score += float64(len(p.BuyingSignals)) * 10

	return math.Min(100, score)
}

func (e *NewLogoEngine) estimateARR(p *Prospect) float64 {
	if p.IsFortune500 {
		return 5_000_000 // $5M base for Fortune 500
	}
	return 1_000_000
}

func (e *NewLogoEngine) calculateProbability(p *Prospect) float64 {
	prob := 0.15 // Base 15%

	if p.Score > 80 {
		prob += 0.20
	}

	return math.Min(0.95, prob)
}

func initializePipeline() *SalesPipeline {
	return &SalesPipeline{
		stages: map[string]*PipelineStage{
			"prospecting": {Name: "Prospecting", Order: 1, Probability: 0.10},
			"qualified":   {Name: "Qualified", Order: 2, Probability: 0.25},
			"demo":        {Name: "Demo", Order: 3, Probability: 0.40},
			"proposal":    {Name: "Proposal", Order: 4, Probability: 0.60},
			"negotiation": {Name: "Negotiation", Order: 5, Probability: 0.80},
			"closed-won":  {Name: "Closed Won", Order: 6, Probability: 1.00},
		},
	}
}

func initializeSalesPlaybooks() map[string]*SalesPlaybook {
	return map[string]*SalesPlaybook{
		"fortune-500": {
			ID:            "fortune-500",
			Name:          "Fortune 500 Enterprise",
			TargetSegment: "Fortune 500",
			SuccessRate:   0.45,
			AvgDealSize:   5_000_000,
			AvgSalesCycle: 180,
			Enabled:       true,
		},
	}
}

func initializeCompetitiveIntel() *CompetitiveIntelligence {
	return &CompetitiveIntelligence{
		competitors:     make(map[string]*Competitor),
		battleCards:     make(map[string]*BattleCard),
		winLossAnalysis: make([]WinLossRecord, 0),
	}
}

func initializeSalesAutomation() *SalesAutomation {
	return &SalesAutomation{
		workflows:     make(map[string]*Workflow),
		sequences:     make(map[string]*EmailSequence),
		scoringModels: make(map[string]*ScoringModel),
	}
}

func initializeVelocityOptimizer() *VelocityOptimizer {
	return &VelocityOptimizer{
		optimizations: make([]VelocityOptimization, 0),
		benchmarks: SalesCycleBenchmarks{
			Overall: 120, // Target 4 months
		},
		bottlenecks: make([]SalesBottleneck, 0),
	}
}

// ExportMetrics exports acquisition metrics
func (e *NewLogoEngine) ExportMetrics() map[string]interface{} {
	e.metrics.mu.RLock()
	defer e.metrics.mu.RUnlock()

	return map[string]interface{}{
		"total_prospects":       e.metrics.TotalProspects,
		"fortune_500_prospects": e.metrics.Fortune500Prospects,
		"total_pipeline":        e.metrics.TotalPipeline,
		"total_arr":             e.metrics.TotalARR,
		"avg_deal_size":         e.metrics.AvgDealSize,
		"avg_sales_cycle":       e.metrics.AvgSalesCycle,
		"win_rate":              e.metrics.WinRate,
	}
}

// MarshalJSON implements json.Marshaler
func (e *NewLogoEngine) MarshalJSON() ([]byte, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return json.Marshal(map[string]interface{}{
		"prospects":     len(e.prospects),
		"accounts":      len(e.accounts),
		"campaigns":     len(e.campaigns),
		"playbooks":     len(e.playbooks),
		"metrics":       e.ExportMetrics(),
	})
}
