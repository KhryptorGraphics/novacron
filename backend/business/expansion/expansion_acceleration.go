// Package expansion provides enterprise expansion and 150% NRR optimization
package expansion

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"
	"time"
)

// ExpansionEngine drives 150% net revenue retention
type ExpansionEngine struct {
	mu                  sync.RWMutex
	opportunities       map[string]*ExpansionOpportunity
	playbooks           map[string]*ExpansionPlaybook
	customers           map[string]*CustomerExpansion
	campaigns           map[string]*ExpansionCampaign
	productCatalog      *ProductCatalog
	pricingEngine       *PricingEngine
	automationRules     []AutomationRule
	metrics             *ExpansionMetrics
	config              ExpansionConfig
}

// ExpansionOpportunity represents upsell/cross-sell potential
type ExpansionOpportunity struct {
	ID                  string                 `json:"id"`
	CustomerID          string                 `json:"customer_id"`
	Type                string                 `json:"type"`                // upsell, cross-sell, upgrade
	Status              string                 `json:"status"`              // identified, qualified, engaged, closed
	Priority            string                 `json:"priority"`            // critical, high, medium, low
	Score               float64                `json:"score"`               // 0-100
	EstimatedARR        float64                `json:"estimated_arr"`
	Probability         float64                `json:"probability"`         // 0-1
	ExpectedValue       float64                `json:"expected_value"`      // ARR * Probability
	Products            []ProductRecommendation `json:"products"`
	Timeline            ExpansionTimeline      `json:"timeline"`
	Triggers            []ExpansionTrigger     `json:"triggers"`
	Barriers            []ExpansionBarrier     `json:"barriers"`
	Strategy            ExpansionStrategy      `json:"strategy"`
	Activities          []ExpansionActivity    `json:"activities"`
	ROI                 float64                `json:"roi"`
	CreatedAt           time.Time              `json:"created_at"`
	UpdatedAt           time.Time              `json:"updated_at"`
	ClosedAt            *time.Time             `json:"closed_at,omitempty"`
}

// ProductRecommendation suggests expansion products
type ProductRecommendation struct {
	ProductID           string                 `json:"product_id"`
	ProductName         string                 `json:"product_name"`
	Category            string                 `json:"category"`
	ARRIncrease         float64                `json:"arr_increase"`
	Relevance           float64                `json:"relevance"`          // 0-1
	Reasoning           string                 `json:"reasoning"`
	UseCases            []string               `json:"use_cases"`
	ROI                 float64                `json:"roi"`
	TimeToValue         int                    `json:"time_to_value"`      // Days
	Dependencies        []string               `json:"dependencies"`
	Pricing             PricingProposal        `json:"pricing"`
}

// PricingProposal details expansion pricing
type PricingProposal struct {
	ListPrice           float64                `json:"list_price"`
	DiscountedPrice     float64                `json:"discounted_price"`
	Discount            float64                `json:"discount"`           // Percentage
	BillingFrequency    string                 `json:"billing_frequency"`  // monthly, annual
	Term                int                    `json:"term"`               // Months
	PaymentTerms        string                 `json:"payment_terms"`
	Incentives          []string               `json:"incentives"`
	TotalContractValue  float64                `json:"total_contract_value"`
}

// ExpansionTimeline tracks expansion progress
type ExpansionTimeline struct {
	IdentifiedDate      time.Time              `json:"identified_date"`
	QualifiedDate       *time.Time             `json:"qualified_date,omitempty"`
	EngagedDate         *time.Time             `json:"engaged_date,omitempty"`
	ProposalDate        *time.Time             `json:"proposal_date,omitempty"`
	NegotiationDate     *time.Time             `json:"negotiation_date,omitempty"`
	ClosedDate          *time.Time             `json:"closed_date,omitempty"`
	ExpectedCloseDate   time.Time              `json:"expected_close_date"`
	DaysInStage         int                    `json:"days_in_stage"`
	TotalDaysOpen       int                    `json:"total_days_open"`
	VelocityScore       float64                `json:"velocity_score"`     // Compared to avg
}

// ExpansionTrigger identifies expansion signals
type ExpansionTrigger struct {
	Type                string                 `json:"type"`
	Description         string                 `json:"description"`
	Strength            float64                `json:"strength"`           // 0-1
	DetectedAt          time.Time              `json:"detected_at"`
	DataPoints          map[string]interface{} `json:"data_points"`
}

// ExpansionBarrier tracks obstacles
type ExpansionBarrier struct {
	Type                string                 `json:"type"`
	Description         string                 `json:"description"`
	Impact              string                 `json:"impact"`             // high, medium, low
	Mitigation          string                 `json:"mitigation"`
	Status              string                 `json:"status"`             // active, resolved, monitoring
}

// ExpansionStrategy defines approach
type ExpansionStrategy struct {
	Name                string                 `json:"name"`
	Playbook            string                 `json:"playbook"`
	Tactics             []string               `json:"tactics"`
	KeyMessages         []string               `json:"key_messages"`
	Stakeholders        []Stakeholder          `json:"stakeholders"`
	Timeline            int                    `json:"timeline"`           // Days
	Budget              float64                `json:"budget"`
	Resources           []string               `json:"resources"`
}

// Stakeholder represents decision maker
type Stakeholder struct {
	Name                string                 `json:"name"`
	Role                string                 `json:"role"`
	Department          string                 `json:"department"`
	InfluenceLevel      string                 `json:"influence_level"`    // champion, influencer, decision-maker
	Sentiment           string                 `json:"sentiment"`          // positive, neutral, negative
	Engagement          float64                `json:"engagement"`         // 0-1
	LastContact         time.Time              `json:"last_contact"`
}

// ExpansionActivity tracks execution
type ExpansionActivity struct {
	ID                  string                 `json:"id"`
	Type                string                 `json:"type"`
	Description         string                 `json:"description"`
	Status              string                 `json:"status"`
	Owner               string                 `json:"owner"`
	DueDate             time.Time              `json:"due_date"`
	CompletedDate       *time.Time             `json:"completed_date,omitempty"`
	Outcome             string                 `json:"outcome"`
	NextSteps           []string               `json:"next_steps"`
}

// ExpansionPlaybook defines expansion strategies
type ExpansionPlaybook struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Type                string                 `json:"type"`
	Description         string                 `json:"description"`
	Criteria            PlaybookCriteria       `json:"criteria"`
	Strategy            PlaybookStrategy       `json:"strategy"`
	Milestones          []PlaybookMilestone    `json:"milestones"`
	SuccessRate         float64                `json:"success_rate"`
	AvgTimeToClose      int                    `json:"avg_time_to_close"`  // Days
	AvgARRIncrease      float64                `json:"avg_arr_increase"`
	Enabled             bool                   `json:"enabled"`
}

// PlaybookCriteria defines when to use playbook
type PlaybookCriteria struct {
	CustomerSegment     []string               `json:"customer_segment"`
	ARRRange            [2]float64             `json:"arr_range"`
	UsagePattern        []string               `json:"usage_pattern"`
	HealthScore         [2]float64             `json:"health_score"`
	Triggers            []string               `json:"triggers"`
}

// PlaybookStrategy defines execution approach
type PlaybookStrategy struct {
	Phases              []StrategyPhase        `json:"phases"`
	Touchpoints         []string               `json:"touchpoints"`
	Resources           []string               `json:"resources"`
	Automation          []string               `json:"automation"`
	SuccessMetrics      []string               `json:"success_metrics"`
}

// StrategyPhase represents execution stage
type StrategyPhase struct {
	Name                string                 `json:"name"`
	Duration            int                    `json:"duration"`           // Days
	Objectives          []string               `json:"objectives"`
	Activities          []string               `json:"activities"`
	Deliverables        []string               `json:"deliverables"`
	SuccessCriteria     []string               `json:"success_criteria"`
}

// PlaybookMilestone tracks progress
type PlaybookMilestone struct {
	Name                string                 `json:"name"`
	Day                 int                    `json:"day"`
	Required            bool                   `json:"required"`
	Activities          []string               `json:"activities"`
	ExpectedOutcome     string                 `json:"expected_outcome"`
}

// CustomerExpansion tracks customer-level expansion
type CustomerExpansion struct {
	CustomerID          string                 `json:"customer_id"`
	CurrentARR          float64                `json:"current_arr"`
	InitialARR          float64                `json:"initial_arr"`
	ExpansionARR        float64                `json:"expansion_arr"`
	NetRetention        float64                `json:"net_retention"`
	GrossRetention      float64                `json:"gross_retention"`
	ExpansionRate       float64                `json:"expansion_rate"`
	Products            []CustomerProduct      `json:"products"`
	Opportunities       []string               `json:"opportunities"`      // IDs
	History             []ExpansionEvent       `json:"history"`
	Potential           ExpansionPotential     `json:"potential"`
	HealthScore         float64                `json:"health_score"`
	RiskLevel           string                 `json:"risk_level"`
	LastExpansion       *time.Time             `json:"last_expansion,omitempty"`
	NextReview          time.Time              `json:"next_review"`
}

// CustomerProduct tracks product adoption
type CustomerProduct struct {
	ProductID           string                 `json:"product_id"`
	ProductName         string                 `json:"product_name"`
	ARR                 float64                `json:"arr"`
	Seats               int                    `json:"seats"`
	Adoption            float64                `json:"adoption"`           // 0-1
	Usage               UsageMetrics           `json:"usage"`
	Satisfaction        float64                `json:"satisfaction"`       // 0-5
	StartDate           time.Time              `json:"start_date"`
	RenewalDate         time.Time              `json:"renewal_date"`
	AtRisk              bool                   `json:"at_risk"`
}

// UsageMetrics tracks product usage
type UsageMetrics struct {
	ActiveUsers         int                    `json:"active_users"`
	TotalLogins         int64                  `json:"total_logins"`
	AvgSessionDuration  int                    `json:"avg_session_duration"` // Minutes
	FeaturesUsed        int                    `json:"features_used"`
	FeaturesAvailable   int                    `json:"features_available"`
	APICallsPerDay      int64                  `json:"api_calls_per_day"`
	DataVolume          int64                  `json:"data_volume"`         // GB
	Trend               string                 `json:"trend"`               // increasing, stable, decreasing
}

// ExpansionEvent records expansion history
type ExpansionEvent struct {
	Type                string                 `json:"type"`
	Date                time.Time              `json:"date"`
	ARRChange           float64                `json:"arr_change"`
	Products            []string               `json:"products"`
	Reason              string                 `json:"reason"`
	Notes               string                 `json:"notes"`
}

// ExpansionPotential estimates future expansion
type ExpansionPotential struct {
	TotalPotential      float64                `json:"total_potential"`
	ShortTerm           float64                `json:"short_term"`         // 90 days
	MediumTerm          float64                `json:"medium_term"`        // 180 days
	LongTerm            float64                `json:"long_term"`          // 365 days
	Confidence          float64                `json:"confidence"`         // 0-1
	Drivers             []string               `json:"drivers"`
	Requirements        []string               `json:"requirements"`
}

// ExpansionCampaign manages expansion initiatives
type ExpansionCampaign struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Type                string                 `json:"type"`
	Status              string                 `json:"status"`
	TargetSegment       SegmentCriteria        `json:"target_segment"`
	Products            []string               `json:"products"`
	TargetARR           float64                `json:"target_arr"`
	ActualARR           float64                `json:"actual_arr"`
	Achievement         float64                `json:"achievement"`
	CustomerCount       int                    `json:"customer_count"`
	ConversionRate      float64                `json:"conversion_rate"`
	AvgDealSize         float64                `json:"avg_deal_size"`
	StartDate           time.Time              `json:"start_date"`
	EndDate             time.Time              `json:"end_date"`
	Budget              float64                `json:"budget"`
	Spend               float64                `json:"spend"`
	ROI                 float64                `json:"roi"`
	Metrics             CampaignMetrics        `json:"metrics"`
}

// SegmentCriteria defines target customers
type SegmentCriteria struct {
	ARRRange            [2]float64             `json:"arr_range"`
	Industry            []string               `json:"industry"`
	CompanySize         []string               `json:"company_size"`
	CurrentProducts     []string               `json:"current_products"`
	HealthScore         [2]float64             `json:"health_score"`
	UsageThreshold      float64                `json:"usage_threshold"`
}

// CampaignMetrics tracks campaign performance
type CampaignMetrics struct {
	OpportunitiesCreated int                   `json:"opportunities_created"`
	OpportunitiesQualified int                 `json:"opportunities_qualified"`
	OpportunitiesClosed  int                   `json:"opportunities_closed"`
	TotalARR            float64                `json:"total_arr"`
	AvgTimeToClose      int                    `json:"avg_time_to_close"`
	WinRate             float64                `json:"win_rate"`
	CustomerSatisfaction float64               `json:"customer_satisfaction"`
}

// ProductCatalog manages expansion products
type ProductCatalog struct {
	mu                  sync.RWMutex
	products            map[string]*Product
	bundles             map[string]*ProductBundle
	relationships       map[string][]string   // Product affinities
}

// Product represents an expansion product
type Product struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Category            string                 `json:"category"`
	Description         string                 `json:"description"`
	BasePrice           float64                `json:"base_price"`
	PricingModel        string                 `json:"pricing_model"`      // per-seat, usage-based, flat
	TargetSegment       []string               `json:"target_segment"`
	Prerequisites       []string               `json:"prerequisites"`      // Required products
	Complements         []string               `json:"complements"`        // Recommended products
	AvgARR              float64                `json:"avg_arr"`
	AdoptionRate        float64                `json:"adoption_rate"`
	TimeToValue         int                    `json:"time_to_value"`      // Days
	ChurnRate           float64                `json:"churn_rate"`
	CustomerSatisfaction float64               `json:"customer_satisfaction"`
}

// ProductBundle groups products
type ProductBundle struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Products            []string               `json:"products"`
	BundlePrice         float64                `json:"bundle_price"`
	Discount            float64                `json:"discount"`
	PopularityScore     float64                `json:"popularity_score"`
}

// PricingEngine optimizes expansion pricing
type PricingEngine struct {
	mu                  sync.RWMutex
	pricingRules        []PricingRule
	discountMatrix      map[string]map[string]float64
	competitivePricing  map[string]float64
}

// PricingRule defines pricing logic
type PricingRule struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Conditions          []PricingCondition     `json:"conditions"`
	Action              PricingAction          `json:"action"`
	Priority            int                    `json:"priority"`
	Enabled             bool                   `json:"enabled"`
}

// PricingCondition evaluates pricing criteria
type PricingCondition struct {
	Field               string                 `json:"field"`
	Operator            string                 `json:"operator"`
	Value               interface{}            `json:"value"`
}

// PricingAction defines pricing adjustment
type PricingAction struct {
	Type                string                 `json:"type"`               // discount, markup, fixed
	Value               float64                `json:"value"`
	Reason              string                 `json:"reason"`
}

// AutomationRule defines expansion automation
type AutomationRule struct {
	ID                  string                 `json:"id"`
	Name                string                 `json:"name"`
	Trigger             AutomationTrigger      `json:"trigger"`
	Conditions          []AutomationCondition  `json:"conditions"`
	Actions             []AutomationAction     `json:"actions"`
	Enabled             bool                   `json:"enabled"`
	ExecutionCount      int64                  `json:"execution_count"`
	SuccessRate         float64                `json:"success_rate"`
}

// AutomationTrigger defines when automation runs
type AutomationTrigger struct {
	Type                string                 `json:"type"`
	Event               string                 `json:"event"`
	Schedule            string                 `json:"schedule,omitempty"`
}

// AutomationCondition evaluates automation criteria
type AutomationCondition struct {
	Field               string                 `json:"field"`
	Operator            string                 `json:"operator"`
	Value               interface{}            `json:"value"`
}

// AutomationAction defines automation behavior
type AutomationAction struct {
	Type                string                 `json:"type"`
	Target              string                 `json:"target"`
	Parameters          map[string]interface{} `json:"parameters"`
}

// ExpansionMetrics tracks performance
type ExpansionMetrics struct {
	mu                  sync.RWMutex
	TotalOpportunities  int64                  `json:"total_opportunities"`
	OpenOpportunities   int64                  `json:"open_opportunities"`
	ClosedWon           int64                  `json:"closed_won"`
	ClosedLost          int64                  `json:"closed_lost"`
	TotalExpansionARR   float64                `json:"total_expansion_arr"`
	AvgDealSize         float64                `json:"avg_deal_size"`
	WinRate             float64                `json:"win_rate"`
	AvgTimeToClose      int                    `json:"avg_time_to_close"`
	NetRetention        float64                `json:"net_retention"`
	GrossRetention      float64                `json:"gross_retention"`
	ProductionRate      float64                `json:"production_rate"`     // Opp created/day
}

// ExpansionConfig configures the engine
type ExpansionConfig struct {
	TargetNRR           float64                `json:"target_nrr"`          // 150%
	TargetExpansionARR  float64                `json:"target_expansion_arr"` // $500M
	MinOpportunitySize  float64                `json:"min_opportunity_size"`
	AutoScoring         bool                   `json:"auto_scoring"`
	AutoPlaybooks       bool                   `json:"auto_playbooks"`
	AutoPricing         bool                   `json:"auto_pricing"`
	EnableMLRecommendations bool               `json:"enable_ml_recommendations"`
}

// NewExpansionEngine creates expansion engine
func NewExpansionEngine(config ExpansionConfig) *ExpansionEngine {
	return &ExpansionEngine{
		opportunities:   make(map[string]*ExpansionOpportunity),
		playbooks:       initializePlaybooks(),
		customers:       make(map[string]*CustomerExpansion),
		campaigns:       make(map[string]*ExpansionCampaign),
		productCatalog:  initializeProductCatalog(),
		pricingEngine:   initializePricingEngine(),
		automationRules: initializeAutomationRules(),
		metrics:         &ExpansionMetrics{},
		config:          config,
	}
}

// IdentifyOpportunity creates new expansion opportunity
func (e *ExpansionEngine) IdentifyOpportunity(ctx context.Context, customerID string, triggers []ExpansionTrigger) (*ExpansionOpportunity, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Get customer data
	customer, exists := e.customers[customerID]
	if !exists {
		return nil, fmt.Errorf("customer not found: %s", customerID)
	}

	// Analyze opportunity
	opp := &ExpansionOpportunity{
		ID:         fmt.Sprintf("exp-%s-%d", customerID, time.Now().Unix()),
		CustomerID: customerID,
		Type:       determineOpportunityType(triggers),
		Status:     "identified",
		Priority:   "medium",
		Triggers:   triggers,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
	}

	// Score opportunity
	opp.Score = e.scoreOpportunity(opp, customer)

	// Get product recommendations
	opp.Products = e.recommendProducts(customer, triggers)

	// Calculate estimated ARR
	totalARR := 0.0
	for _, product := range opp.Products {
		totalARR += product.ARRIncrease
	}
	opp.EstimatedARR = totalARR

	// Calculate probability
	opp.Probability = e.calculateProbability(opp, customer)
	opp.ExpectedValue = opp.EstimatedARR * opp.Probability

	// Select playbook
	playbook := e.selectPlaybook(opp, customer)
	if playbook != nil {
		opp.Strategy = ExpansionStrategy{
			Name:     playbook.Name,
			Playbook: playbook.ID,
			Timeline: playbook.AvgTimeToClose,
		}

		// Set expected close date
		opp.Timeline.ExpectedCloseDate = time.Now().AddDate(0, 0, playbook.AvgTimeToClose)
	}

	// Store opportunity
	e.opportunities[opp.ID] = opp

	// Update customer
	customer.Opportunities = append(customer.Opportunities, opp.ID)

	// Update metrics
	e.metrics.mu.Lock()
	e.metrics.TotalOpportunities++
	e.metrics.OpenOpportunities++
	e.metrics.mu.Unlock()

	return opp, nil
}

// Helper functions (simplified for brevity)

func determineOpportunityType(triggers []ExpansionTrigger) string {
	if len(triggers) == 0 {
		return "cross-sell"
	}

	for _, trigger := range triggers {
		if trigger.Type == "usage_increase" {
			return "upsell"
		}
	}

	return "cross-sell"
}

func (e *ExpansionEngine) scoreOpportunity(opp *ExpansionOpportunity, customer *CustomerExpansion) float64 {
	score := 50.0 // Base score

	// Health score impact
	score += customer.HealthScore * 0.3

	// ARR impact
	if opp.EstimatedARR > 1_000_000 {
		score += 20
	}

	// Trigger strength
	for _, trigger := range opp.Triggers {
		score += trigger.Strength * 10
	}

	return math.Min(100, score)
}

func (e *ExpansionEngine) recommendProducts(customer *CustomerExpansion, triggers []ExpansionTrigger) []ProductRecommendation {
	recommendations := make([]ProductRecommendation, 0)

	// Simple recommendation logic
	for productID, product := range e.productCatalog.products {
		// Check if customer already has it
		hasProduct := false
		for _, cp := range customer.Products {
			if cp.ProductID == productID {
				hasProduct = true
				break
			}
		}

		if !hasProduct {
			recommendations = append(recommendations, ProductRecommendation{
				ProductID:   productID,
				ProductName: product.Name,
				Category:    product.Category,
				ARRIncrease: product.AvgARR,
				Relevance:   0.8,
				Reasoning:   "Complementary product based on current usage",
				ROI:         3.5,
				TimeToValue: product.TimeToValue,
				Pricing: PricingProposal{
					ListPrice:       product.BasePrice,
					DiscountedPrice: product.BasePrice * 0.9,
					Discount:        10,
					BillingFrequency: "annual",
					Term:            12,
				},
			})
		}
	}

	return recommendations
}

func (e *ExpansionEngine) calculateProbability(opp *ExpansionOpportunity, customer *CustomerExpansion) float64 {
	prob := 0.5 // Base probability

	// Health score impact
	prob += (customer.HealthScore / 100) * 0.3

	// Historical expansion
	if customer.LastExpansion != nil {
		daysSince := time.Since(*customer.LastExpansion).Hours() / 24
		if daysSince < 180 {
			prob += 0.1
		}
	}

	return math.Min(1.0, prob)
}

func (e *ExpansionEngine) selectPlaybook(opp *ExpansionOpportunity, customer *CustomerExpansion) *ExpansionPlaybook {
	for _, playbook := range e.playbooks {
		if !playbook.Enabled {
			continue
		}

		// Check criteria
		if customer.CurrentARR >= playbook.Criteria.ARRRange[0] &&
		   customer.CurrentARR <= playbook.Criteria.ARRRange[1] {
			return playbook
		}
	}

	return nil
}

func initializePlaybooks() map[string]*ExpansionPlaybook {
	return map[string]*ExpansionPlaybook{
		"enterprise-upsell": {
			ID:          "enterprise-upsell",
			Name:        "Enterprise Upsell",
			Type:        "upsell",
			Description: "Upsell strategy for enterprise customers",
			SuccessRate: 0.65,
			AvgTimeToClose: 60,
			AvgARRIncrease: 500_000,
			Enabled:     true,
			Criteria: PlaybookCriteria{
				ARRRange:    [2]float64{1_000_000, 10_000_000},
				HealthScore: [2]float64{70, 100},
			},
		},
	}
}

func initializeProductCatalog() *ProductCatalog {
	return &ProductCatalog{
		products: map[string]*Product{
			"advanced-analytics": {
				ID:           "advanced-analytics",
				Name:         "Advanced Analytics",
				Category:     "analytics",
				BasePrice:    250_000,
				PricingModel: "per-seat",
				AvgARR:       500_000,
				AdoptionRate: 0.45,
				TimeToValue:  30,
			},
		},
		bundles:       make(map[string]*ProductBundle),
		relationships: make(map[string][]string),
	}
}

func initializePricingEngine() *PricingEngine {
	return &PricingEngine{
		pricingRules:       make([]PricingRule, 0),
		discountMatrix:     make(map[string]map[string]float64),
		competitivePricing: make(map[string]float64),
	}
}

func initializeAutomationRules() []AutomationRule {
	return []AutomationRule{
		{
			ID:      "auto-identify-high-usage",
			Name:    "Auto-Identify High Usage Opportunities",
			Enabled: true,
			Trigger: AutomationTrigger{
				Type:  "scheduled",
				Schedule: "daily",
			},
		},
	}
}

// ExportMetrics exports expansion metrics
func (e *ExpansionEngine) ExportMetrics() map[string]interface{} {
	e.metrics.mu.RLock()
	defer e.metrics.mu.RUnlock()

	return map[string]interface{}{
		"total_opportunities":   e.metrics.TotalOpportunities,
		"open_opportunities":    e.metrics.OpenOpportunities,
		"closed_won":            e.metrics.ClosedWon,
		"total_expansion_arr":   e.metrics.TotalExpansionARR,
		"avg_deal_size":         e.metrics.AvgDealSize,
		"win_rate":              e.metrics.WinRate,
		"net_retention":         e.metrics.NetRetention,
	}
}

// MarshalJSON implements json.Marshaler
func (e *ExpansionEngine) MarshalJSON() ([]byte, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return json.Marshal(map[string]interface{}{
		"opportunities":  len(e.opportunities),
		"customers":      len(e.customers),
		"campaigns":      len(e.campaigns),
		"playbooks":      len(e.playbooks),
		"metrics":        e.ExportMetrics(),
	})
}
