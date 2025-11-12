// Package partners implements Partner Ecosystem Scaling Platform
// for managing 5,000+ channel partners, co-selling automation, and partner revenue tracking
// to achieve $200M+ partner-sourced revenue.
package partners

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"
	"time"
)

// PartnerTier represents partner program tiers
type PartnerTier string

const (
	TierPlatinum PartnerTier = "platinum"
	TierGold     PartnerTier = "gold"
	TierSilver   PartnerTier = "silver"
	TierBronze   PartnerTier = "bronze"
	TierRegistered PartnerTier = "registered"
)

// PartnerType represents partner categories
type PartnerType string

const (
	TypeReseller     PartnerType = "reseller"
	TypeSystemIntegrator PartnerType = "system_integrator"
	TypeISV          PartnerType = "isv"          // Independent Software Vendor
	TypeMSP          PartnerType = "msp"          // Managed Service Provider
	TypeDistributor  PartnerType = "distributor"
	TypeOEM          PartnerType = "oem"
	TypeTechnology   PartnerType = "technology"   // Technology alliance
)

// PartnerProfile represents a channel partner
type PartnerProfile struct {
	PartnerID          string                 `json:"partner_id"`
	CompanyName        string                 `json:"company_name"`
	PartnerType        PartnerType            `json:"partner_type"`
	Tier               PartnerTier            `json:"tier"`
	Status             string                 `json:"status"` // active, inactive, suspended

	// Company attributes
	EmployeeCount      int                    `json:"employee_count"`
	AnnualRevenue      float64                `json:"annual_revenue"`
	Geography          []string               `json:"geography"`       // Countries/regions
	Verticals          []string               `json:"verticals"`       // Industry focus
	TechnicalCerts     int                    `json:"technical_certs"` // Certified engineers

	// Performance metrics
	TotalRevenue       float64                `json:"total_revenue"`       // Lifetime revenue
	QuarterlyRevenue   float64                `json:"quarterly_revenue"`
	DealCount          int                    `json:"deal_count"`
	AverageDealSize    float64                `json:"average_deal_size"`
	WinRate            float64                `json:"win_rate"`

	// Engagement metrics
	LeadsProvided      int                    `json:"leads_provided"`
	LeadsAccepted      int                    `json:"leads_accepted"`
	LeadConversionRate float64                `json:"lead_conversion_rate"`
	CoSellingDeals     int                    `json:"co_selling_deals"`

	// Program compliance
	TrainingCompleted  int                    `json:"training_completed"`
	CertificationsHeld int                    `json:"certifications_held"`
	ProgramCompliance  float64                `json:"program_compliance"` // 0-1 score

	// Strategic value
	StrategicPartner   bool                   `json:"strategic_partner"`
	ReferenceValue     float64                `json:"reference_value"`
	MarketingROI       float64                `json:"marketing_roi"`

	// Contact info
	PartnerManager     string                 `json:"partner_manager"` // Assigned PAM
	PrimaryContact     string                 `json:"primary_contact"`

	CustomAttributes   map[string]interface{} `json:"custom_attributes"`
	JoinedDate         time.Time              `json:"joined_date"`
	LastActivity       time.Time              `json:"last_activity"`
	LastUpdated        time.Time              `json:"last_updated"`
}

// PartnerDeal represents a partner-sourced deal
type PartnerDeal struct {
	DealID             string    `json:"deal_id"`
	PartnerID          string    `json:"partner_id"`
	AccountName        string    `json:"account_name"`
	DealValue          float64   `json:"deal_value"`
	PartnerCommission  float64   `json:"partner_commission"`
	DealType           string    `json:"deal_type"` // direct, co-sell, referral
	Stage              string    `json:"stage"`
	Probability        float64   `json:"probability"`
	ExpectedCloseDate  time.Time `json:"expected_close_date"`
	ActualCloseDate    time.Time `json:"actual_close_date"`
	Status             string    `json:"status"` // open, won, lost
	LastUpdated        time.Time `json:"last_updated"`
}

// PartnerProgram represents partner program structure
type PartnerProgram struct {
	ProgramID          string                 `json:"program_id"`
	ProgramName        string                 `json:"program_name"`
	Tier               PartnerTier            `json:"tier"`
	Requirements       map[string]interface{} `json:"requirements"`
	Benefits           []string               `json:"benefits"`
	CommissionRates    map[string]float64     `json:"commission_rates"` // Deal type -> rate
	MDFBudget          float64                `json:"mdf_budget"`       // Market Development Funds
	TrainingRequired   []string               `json:"training_required"`
	CertificationReqs  []string               `json:"certification_reqs"`
	LastUpdated        time.Time              `json:"last_updated"`
}

// CoSellingOpportunity represents co-selling engagement
type CoSellingOpportunity struct {
	OpportunityID      string    `json:"opportunity_id"`
	PartnerID          string    `json:"partner_id"`
	AccountName        string    `json:"account_name"`
	OpportunityValue   float64   `json:"opportunity_value"`
	PartnerRole        string    `json:"partner_role"`        // lead, assist, fulfill
	CoSellingStage     string    `json:"co_selling_stage"`    // discovery, qualification, proposal, negotiation
	ResourcesNeeded    []string  `json:"resources_needed"`
	SalesTeamAssigned  string    `json:"sales_team_assigned"`
	Status             string    `json:"status"`
	CreatedDate        time.Time `json:"created_date"`
	LastUpdated        time.Time `json:"last_updated"`
}

// PartnerEcosystemPlatform manages partner network scaling
type PartnerEcosystemPlatform struct {
	mu                 sync.RWMutex
	partners           map[string]*PartnerProfile
	deals              map[string]*PartnerDeal
	programs           map[PartnerTier]*PartnerProgram
	coSellingOpps      map[string]*CoSellingOpportunity
	ecosystemMetrics   *EcosystemMetrics
	ctx                context.Context
	cancel             context.CancelFunc
}

// EcosystemMetrics tracks partner ecosystem performance
type EcosystemMetrics struct {
	TotalPartners           int                `json:"total_partners"`
	ActivePartners          int                `json:"active_partners"`
	PartnersByTier          map[PartnerTier]int `json:"partners_by_tier"`
	PartnersByType          map[PartnerType]int `json:"partners_by_type"`
	TotalPartnerRevenue     float64            `json:"total_partner_revenue"`
	QuarterlyPartnerRevenue float64            `json:"quarterly_partner_revenue"`
	PartnerSourcingRate     float64            `json:"partner_sourcing_rate"` // % of revenue from partners
	AvgPartnerDealSize      float64            `json:"avg_partner_deal_size"`
	AvgPartnerWinRate       float64            `json:"avg_partner_win_rate"`
	CoSellingDeals          int                `json:"co_selling_deals"`
	CertifiedEngineers      int                `json:"certified_engineers"`
	LastUpdated             time.Time          `json:"last_updated"`
}

// NewPartnerEcosystemPlatform creates the partner ecosystem management system
func NewPartnerEcosystemPlatform() *PartnerEcosystemPlatform {
	ctx, cancel := context.WithCancel(context.Background())

	platform := &PartnerEcosystemPlatform{
		partners:         make(map[string]*PartnerProfile),
		deals:            make(map[string]*PartnerDeal),
		programs:         make(map[PartnerTier]*PartnerProgram),
		coSellingOpps:    make(map[string]*CoSellingOpportunity),
		ecosystemMetrics: &EcosystemMetrics{},
		ctx:              ctx,
		cancel:           cancel,
	}

	platform.initializePrograms()

	return platform
}

// initializePrograms sets up partner program tiers
func (p *PartnerEcosystemPlatform) initializePrograms() {
	p.programs[TierPlatinum] = &PartnerProgram{
		ProgramID:   "platinum-partner",
		ProgramName: "Platinum Partner Program",
		Tier:        TierPlatinum,
		Requirements: map[string]interface{}{
			"annual_revenue_min":     10_000_000, // $10M min
			"certified_engineers":    50,
			"customer_references":    20,
			"verticals_covered":      3,
			"training_completion":    "100%",
		},
		Benefits: []string{
			"Highest commission rates (30%)",
			"Dedicated Partner Account Manager",
			"Priority co-selling support",
			"$500K MDF budget annually",
			"Executive engagement program",
			"Joint marketing campaigns",
			"Early access to new products",
			"NFR licenses (50 nodes)",
		},
		CommissionRates: map[string]float64{
			"direct":   0.30, // 30% commission
			"co_sell":  0.25, // 25% commission
			"referral": 0.15, // 15% commission
		},
		MDFBudget: 500_000, // $500K
		TrainingRequired: []string{
			"Sales fundamentals",
			"Technical architecture",
			"Competitive positioning",
			"Solution design",
		},
		CertificationReqs: []string{
			"NovaCron Certified Architect (50+ engineers)",
			"NovaCron Sales Specialist",
		},
		LastUpdated: time.Now(),
	}

	p.programs[TierGold] = &PartnerProgram{
		ProgramID:   "gold-partner",
		ProgramName: "Gold Partner Program",
		Tier:        TierGold,
		Requirements: map[string]interface{}{
			"annual_revenue_min":     3_000_000,
			"certified_engineers":    20,
			"customer_references":    10,
			"verticals_covered":      2,
			"training_completion":    "100%",
		},
		Benefits: []string{
			"25% commission rates",
			"Shared Partner Account Manager",
			"Co-selling support",
			"$200K MDF budget annually",
			"Joint marketing opportunities",
			"NFR licenses (25 nodes)",
		},
		CommissionRates: map[string]float64{
			"direct":   0.25,
			"co_sell":  0.20,
			"referral": 0.12,
		},
		MDFBudget: 200_000,
		TrainingRequired: []string{
			"Sales fundamentals",
			"Technical architecture",
		},
		CertificationReqs: []string{
			"NovaCron Certified Architect (20+ engineers)",
		},
		LastUpdated: time.Now(),
	}

	p.programs[TierSilver] = &PartnerProgram{
		ProgramID:   "silver-partner",
		ProgramName: "Silver Partner Program",
		Tier:        TierSilver,
		Requirements: map[string]interface{}{
			"annual_revenue_min":     1_000_000,
			"certified_engineers":    10,
			"customer_references":    5,
			"training_completion":    "100%",
		},
		Benefits: []string{
			"20% commission rates",
			"Standard support",
			"$50K MDF budget annually",
			"NFR licenses (10 nodes)",
		},
		CommissionRates: map[string]float64{
			"direct":   0.20,
			"co_sell":  0.15,
			"referral": 0.10,
		},
		MDFBudget: 50_000,
		TrainingRequired: []string{
			"Sales fundamentals",
		},
		CertificationReqs: []string{
			"NovaCron Certified Architect (10+ engineers)",
		},
		LastUpdated: time.Now(),
	}

	p.programs[TierBronze] = &PartnerProgram{
		ProgramID:   "bronze-partner",
		ProgramName: "Bronze Partner Program",
		Tier:        TierBronze,
		Requirements: map[string]interface{}{
			"annual_revenue_min":     250_000,
			"certified_engineers":    5,
			"training_completion":    "100%",
		},
		Benefits: []string{
			"15% commission rates",
			"Community support",
			"$10K MDF budget annually",
			"NFR licenses (5 nodes)",
		},
		CommissionRates: map[string]float64{
			"direct":   0.15,
			"co_sell":  0.12,
			"referral": 0.08,
		},
		MDFBudget: 10_000,
		TrainingRequired: []string{
			"Sales fundamentals",
		},
		CertificationReqs: []string{
			"NovaCron Certified Architect (5+ engineers)",
		},
		LastUpdated: time.Now(),
	}

	p.programs[TierRegistered] = &PartnerProgram{
		ProgramID:   "registered-partner",
		ProgramName: "Registered Partner Program",
		Tier:        TierRegistered,
		Requirements: map[string]interface{}{
			"registration_complete": true,
		},
		Benefits: []string{
			"10% commission rates",
			"Self-service portal",
			"NFR licenses (2 nodes)",
		},
		CommissionRates: map[string]float64{
			"direct":   0.10,
			"referral": 0.05,
		},
		MDFBudget: 0,
		TrainingRequired: []string{},
		CertificationReqs: []string{},
		LastUpdated: time.Now(),
	}
}

// AddPartner adds or updates a partner profile
func (p *PartnerEcosystemPlatform) AddPartner(partner *PartnerProfile) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	partner.LastUpdated = time.Now()
	p.partners[partner.PartnerID] = partner

	return nil
}

// RegisterDeal registers a partner-sourced deal
func (p *PartnerEcosystemPlatform) RegisterDeal(deal *PartnerDeal) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Validate partner exists
	if _, exists := p.partners[deal.PartnerID]; !exists {
		return fmt.Errorf("partner %s not found", deal.PartnerID)
	}

	// Calculate commission based on partner tier
	partner := p.partners[deal.PartnerID]
	program := p.programs[partner.Tier]

	commissionRate := program.CommissionRates[deal.DealType]
	deal.PartnerCommission = deal.DealValue * commissionRate

	deal.LastUpdated = time.Now()
	p.deals[deal.DealID] = deal

	// Update partner metrics
	p.updatePartnerMetrics(deal.PartnerID)

	return nil
}

// updatePartnerMetrics recalculates partner performance metrics
func (p *PartnerEcosystemPlatform) updatePartnerMetrics(partnerID string) {
	partner := p.partners[partnerID]

	var totalRevenue float64
	var quarterlyRevenue float64
	dealCount := 0
	wins := 0
	losses := 0

	quarterStart := time.Now().AddDate(0, -3, 0)

	for _, deal := range p.deals {
		if deal.PartnerID == partnerID {
			dealCount++

			if deal.Status == "won" {
				totalRevenue += deal.DealValue
				wins++

				if deal.ActualCloseDate.After(quarterStart) {
					quarterlyRevenue += deal.DealValue
				}
			} else if deal.Status == "lost" {
				losses++
			}
		}
	}

	partner.TotalRevenue = totalRevenue
	partner.QuarterlyRevenue = quarterlyRevenue
	partner.DealCount = dealCount

	if dealCount > 0 {
		partner.AverageDealSize = totalRevenue / float64(wins)
	}

	totalDeals := wins + losses
	if totalDeals > 0 {
		partner.WinRate = float64(wins) / float64(totalDeals)
	}

	partner.LastUpdated = time.Now()
}

// CreateCoSellingOpportunity creates a co-selling engagement
func (p *PartnerEcosystemPlatform) CreateCoSellingOpportunity(opp *CoSellingOpportunity) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.partners[opp.PartnerID]; !exists {
		return fmt.Errorf("partner %s not found", opp.PartnerID)
	}

	opp.CreatedDate = time.Now()
	opp.LastUpdated = time.Now()
	p.coSellingOpps[opp.OpportunityID] = opp

	return nil
}

// CalculateEcosystemMetrics computes partner ecosystem performance
func (p *PartnerEcosystemPlatform) CalculateEcosystemMetrics() *EcosystemMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()

	partnersByTier := make(map[PartnerTier]int)
	partnersByType := make(map[PartnerType]int)

	activePartners := 0
	certifiedEngineers := 0

	for _, partner := range p.partners {
		partnersByTier[partner.Tier]++
		partnersByType[partner.PartnerType]++
		certifiedEngineers += partner.TechnicalCerts

		if partner.Status == "active" {
			activePartners++
		}
	}

	// Calculate revenue metrics
	var totalRevenue float64
	var quarterlyRevenue float64
	var totalDealValue float64
	dealCount := 0
	wins := 0
	losses := 0

	quarterStart := time.Now().AddDate(0, -3, 0)

	for _, deal := range p.deals {
		if deal.Status == "won" {
			totalRevenue += deal.DealValue
			wins++
			totalDealValue += deal.DealValue
			dealCount++

			if deal.ActualCloseDate.After(quarterStart) {
				quarterlyRevenue += deal.DealValue
			}
		} else if deal.Status == "lost" {
			losses++
		}
	}

	avgDealSize := 0.0
	if wins > 0 {
		avgDealSize = totalRevenue / float64(wins)
	}

	avgWinRate := 0.0
	totalDeals := wins + losses
	if totalDeals > 0 {
		avgWinRate = float64(wins) / float64(totalDeals)
	}

	// Count co-selling deals
	coSellingCount := len(p.coSellingOpps)

	// Calculate partner sourcing rate (assuming $1B total revenue target)
	totalCompanyRevenue := 1_000_000_000.0 // $1B target
	partnerSourcingRate := totalRevenue / totalCompanyRevenue

	metrics := &EcosystemMetrics{
		TotalPartners:           len(p.partners),
		ActivePartners:          activePartners,
		PartnersByTier:          partnersByTier,
		PartnersByType:          partnersByType,
		TotalPartnerRevenue:     totalRevenue,
		QuarterlyPartnerRevenue: quarterlyRevenue,
		PartnerSourcingRate:     partnerSourcingRate,
		AvgPartnerDealSize:      avgDealSize,
		AvgPartnerWinRate:       avgWinRate,
		CoSellingDeals:          coSellingCount,
		CertifiedEngineers:      certifiedEngineers,
		LastUpdated:             time.Now(),
	}

	p.ecosystemMetrics = metrics
	return metrics
}

// GetTopPartners returns highest performing partners
func (p *PartnerEcosystemPlatform) GetTopPartners(limit int) []*PartnerProfile {
	p.mu.RLock()
	defer p.mu.RUnlock()

	partners := make([]*PartnerProfile, 0, len(p.partners))
	for _, partner := range p.partners {
		partners = append(partners, partner)
	}

	// Sort by total revenue
	sort.Slice(partners, func(i, j int) bool {
		return partners[i].TotalRevenue > partners[j].TotalRevenue
	})

	if limit > len(partners) {
		limit = len(partners)
	}

	return partners[:limit]
}

// GetPartnersByTier returns partners in a specific tier
func (p *PartnerEcosystemPlatform) GetPartnersByTier(tier PartnerTier) []*PartnerProfile {
	p.mu.RLock()
	defer p.mu.RUnlock()

	partners := make([]*PartnerProfile, 0)
	for _, partner := range p.partners {
		if partner.Tier == tier {
			partners = append(partners, partner)
		}
	}

	return partners
}

// EvaluateTierUpgrade evaluates if partner qualifies for tier upgrade
func (p *PartnerEcosystemPlatform) EvaluateTierUpgrade(partnerID string) (bool, PartnerTier, string) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	partner, exists := p.partners[partnerID]
	if !exists {
		return false, "", "Partner not found"
	}

	currentTier := partner.Tier

	// Check if qualifies for higher tier
	tiers := []PartnerTier{TierPlatinum, TierGold, TierSilver, TierBronze, TierRegistered}

	for _, tier := range tiers {
		if tier == currentTier {
			break
		}

		program := p.programs[tier]
		if p.meetsRequirements(partner, program) {
			return true, tier, fmt.Sprintf("Partner qualifies for %s tier upgrade", tier)
		}
	}

	return false, currentTier, "Partner does not qualify for upgrade"
}

// meetsRequirements checks if partner meets program requirements
func (p *PartnerEcosystemPlatform) meetsRequirements(partner *PartnerProfile, program *PartnerProgram) bool {
	reqs := program.Requirements

	// Check annual revenue
	if minRevenue, ok := reqs["annual_revenue_min"].(float64); ok {
		if partner.TotalRevenue < minRevenue {
			return false
		}
	}

	// Check certified engineers
	if minEngineers, ok := reqs["certified_engineers"].(int); ok {
		if partner.TechnicalCerts < minEngineers {
			return false
		}
	}

	// All requirements met
	return true
}

// ProjectPartnerGrowth forecasts partner ecosystem growth
func (p *PartnerEcosystemPlatform) ProjectPartnerGrowth(quarters int) []PartnerProjection {
	p.mu.RLock()
	defer p.mu.RUnlock()

	projections := make([]PartnerProjection, quarters)

	currentPartners := len(p.partners)
	targetPartners := 5000 // 5,000 partner target
	currentRevenue := p.ecosystemMetrics.TotalPartnerRevenue

	for i := 0; i < quarters; i++ {
		remainingQuarters := float64(quarters - i)

		// Calculate quarterly growth
		partnerGrowth := math.Pow(float64(targetPartners)/float64(currentPartners), 1/remainingQuarters)
		currentPartners = int(float64(currentPartners) * partnerGrowth)

		// Project revenue growth
		revenueGrowth := 1.25 // 25% quarterly growth
		currentRevenue *= revenueGrowth

		projections[i] = PartnerProjection{
			Quarter:          i + 1,
			ProjectedPartners: currentPartners,
			ProjectedRevenue: currentRevenue,
			Confidence:       p.calculateProjectionConfidence(i),
			LastUpdated:      time.Now(),
		}
	}

	return projections
}

// calculateProjectionConfidence determines forecast accuracy
func (p *PartnerEcosystemPlatform) calculateProjectionConfidence(quartersOut int) float64 {
	baseConfidence := 0.88
	decayRate := 0.04
	return baseConfidence * math.Exp(-decayRate*float64(quartersOut))
}

// PartnerProjection represents forecasted partner ecosystem
type PartnerProjection struct {
	Quarter           int       `json:"quarter"`
	ProjectedPartners int       `json:"projected_partners"`
	ProjectedRevenue  float64   `json:"projected_revenue"`
	Confidence        float64   `json:"confidence"`
	LastUpdated       time.Time `json:"last_updated"`
}

// GenerateEcosystemReport creates comprehensive partner ecosystem report
func (p *PartnerEcosystemPlatform) GenerateEcosystemReport() *EcosystemReport {
	p.mu.RLock()
	defer p.mu.RUnlock()

	metrics := p.CalculateEcosystemMetrics()
	topPartners := p.GetTopPartners(50)
	projections := p.ProjectPartnerGrowth(12)

	report := &EcosystemReport{
		Metrics:         metrics,
		TopPartners:     topPartners,
		Programs:        p.getProgramsList(),
		Projections:     projections,
		HealthStatus:    p.assessEcosystemHealth(),
		Recommendations: p.generateEcosystemRecommendations(),
		GeneratedAt:     time.Now(),
	}

	return report
}

// getProgramsList returns all partner programs
func (p *PartnerEcosystemPlatform) getProgramsList() []*PartnerProgram {
	programs := make([]*PartnerProgram, 0, len(p.programs))
	for _, program := range p.programs {
		programs = append(programs, program)
	}
	return programs
}

// assessEcosystemHealth evaluates partner ecosystem health
func (p *PartnerEcosystemPlatform) assessEcosystemHealth() string {
	metrics := p.ecosystemMetrics

	if metrics.TotalPartners >= 4000 && metrics.PartnerSourcingRate >= 0.35 {
		return "excellent"
	} else if metrics.TotalPartners >= 2500 && metrics.PartnerSourcingRate >= 0.25 {
		return "good"
	} else if metrics.TotalPartners >= 1000 {
		return "fair"
	}

	return "needs_attention"
}

// generateEcosystemRecommendations creates strategic recommendations
func (p *PartnerEcosystemPlatform) generateEcosystemRecommendations() []string {
	recommendations := []string{}

	metrics := p.ecosystemMetrics

	if metrics.TotalPartners < 5000 {
		recommendations = append(recommendations,
			fmt.Sprintf("🤝 Partner count: %d of 5,000 target - accelerate recruitment",
				metrics.TotalPartners))
	}

	if metrics.PartnerSourcingRate < 0.40 {
		recommendations = append(recommendations,
			fmt.Sprintf("📊 Partner sourcing rate: %.1f%% - target 40%+ of revenue",
				metrics.PartnerSourcingRate*100))
	}

	// Tier recommendations
	platinumCount := metrics.PartnersByTier[TierPlatinum]
	if platinumCount < 100 {
		recommendations = append(recommendations,
			fmt.Sprintf("⭐ Only %d Platinum partners - develop top tier to 100+", platinumCount))
	}

	// Certification recommendations
	avgCertsPerPartner := 0
	if metrics.TotalPartners > 0 {
		avgCertsPerPartner = metrics.CertifiedEngineers / metrics.TotalPartners
	}
	if avgCertsPerPartner < 10 {
		recommendations = append(recommendations,
			"📚 Increase partner certifications - target 10+ certified engineers per partner")
	}

	return recommendations
}

// EcosystemReport represents comprehensive partner ecosystem intelligence
type EcosystemReport struct {
	Metrics         *EcosystemMetrics    `json:"metrics"`
	TopPartners     []*PartnerProfile    `json:"top_partners"`
	Programs        []*PartnerProgram    `json:"programs"`
	Projections     []PartnerProjection  `json:"projections"`
	HealthStatus    string               `json:"health_status"`
	Recommendations []string             `json:"recommendations"`
	GeneratedAt     time.Time            `json:"generated_at"`
}

// ExportMetrics exports ecosystem metrics in JSON format
func (p *PartnerEcosystemPlatform) ExportMetrics() ([]byte, error) {
	report := p.GenerateEcosystemReport()
	return json.MarshalIndent(report, "", "  ")
}

// Close shuts down the partner ecosystem platform
func (p *PartnerEcosystemPlatform) Close() error {
	p.cancel()
	return nil
}
