// Package ecosystem provides ecosystem value creation and management
// Manages joint ventures, licensing revenue, co-innovation labs, and partnerships
package ecosystem

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// EcosystemType defines the type of ecosystem initiative
type EcosystemType string

const (
	TypeJointVenture    EcosystemType = "joint_venture"     // Joint venture
	TypeLicensing       EcosystemType = "licensing"         // Technology licensing
	TypeCoInnovation    EcosystemType = "co_innovation"     // Co-innovation lab
	TypeAcademic        EcosystemType = "academic"          // Academic partnership
	TypeGovernment      EcosystemType = "government"        // Government partnership
	TypeDeveloper       EcosystemType = "developer"         // Developer ecosystem
	TypeStartup         EcosystemType = "startup"           // Startup accelerator
	TypeResearch        EcosystemType = "research"          // Research collaboration
)

// EcosystemStatus defines the status of ecosystem initiative
type EcosystemStatus string

const (
	StatusProspect   EcosystemStatus = "prospect"    // Prospective initiative
	StatusNegotiation EcosystemStatus = "negotiation" // In negotiation
	StatusActive     EcosystemStatus = "active"      // Active initiative
	StatusPaused     EcosystemStatus = "paused"      // Paused
	StatusCompleted  EcosystemStatus = "completed"   // Completed
	StatusTerminated EcosystemStatus = "terminated"  // Terminated
)

// EcosystemInitiative represents an ecosystem initiative
type EcosystemInitiative struct {
	ID              string          `json:"id"`
	Name            string          `json:"name"`
	Type            EcosystemType   `json:"type"`
	Status          EcosystemStatus `json:"status"`
	Description     string          `json:"description"`
	Partners        []Partner       `json:"partners"`         // Partner organizations
	Objectives      []string        `json:"objectives"`       // Initiative objectives
	Value           ValueMetrics    `json:"value"`            // Value metrics
	Investment      Investment      `json:"investment"`       // Investment details
	Governance      GovernanceModel `json:"governance"`       // Governance model
	IPRights        IPRights        `json:"ip_rights"`        // IP rights
	Performance     Performance     `json:"performance"`      // Performance metrics
	Timeline        Timeline        `json:"timeline"`         // Timeline
	Deliverables    []Deliverable   `json:"deliverables"`     // Key deliverables
	Stakeholders    []Stakeholder   `json:"stakeholders"`     // Stakeholders
	RiskAssessment  []Risk          `json:"risk_assessment"`  // Risk assessment
	Documents       []string        `json:"documents"`        // Legal documents
	CreatedAt       time.Time       `json:"created_at"`
	UpdatedAt       time.Time       `json:"updated_at"`
}

// Partner represents a partner organization
type Partner struct {
	Name            string  `json:"name"`
	Type            string  `json:"type"`             // corporation, university, government
	Ownership       float64 `json:"ownership"`        // Ownership %
	Investment      float64 `json:"investment"`       // Investment amount ($M)
	Contribution    string  `json:"contribution"`     // Key contributions
	Contacts        []Contact `json:"contacts"`       // Partner contacts
	Role            string  `json:"role"`             // Partner role
	Responsibilities []string `json:"responsibilities"` // Responsibilities
}

// Contact represents a contact person
type Contact struct {
	Name     string `json:"name"`
	Title    string `json:"title"`
	Email    string `json:"email"`
	Phone    string `json:"phone"`
	Role     string `json:"role"`      // Primary, technical, legal
}

// ValueMetrics represents value creation metrics
type ValueMetrics struct {
	TotalValue        float64 `json:"total_value"`         // Total value created ($M)
	RevenueGenerated  float64 `json:"revenue_generated"`   // Revenue generated ($M)
	CostSavings       float64 `json:"cost_savings"`        // Cost savings ($M)
	MarketAccess      float64 `json:"market_access"`       // Market access value ($M)
	IPValue           float64 `json:"ip_value"`            // IP value ($M)
	TechnologyValue   float64 `json:"technology_value"`    // Technology value ($M)
	BrandValue        float64 `json:"brand_value"`         // Brand value ($M)
	ROI               float64 `json:"roi"`                 // Return on investment %
	Payback           float64 `json:"payback"`             // Payback period (years)
	NPV               float64 `json:"npv"`                 // Net present value ($M)
	IRR               float64 `json:"irr"`                 // Internal rate of return %
	StrategicValue    float64 `json:"strategic_value"`     // Strategic value score (0-100)
}

// Investment represents investment details
type Investment struct {
	TotalInvestment   float64           `json:"total_investment"`    // Total investment ($M)
	NovaCronShare     float64           `json:"novacron_share"`      // NovaCron investment ($M)
	PartnerShare      float64           `json:"partner_share"`       // Partner investment ($M)
	InvestmentType    string            `json:"investment_type"`     // cash, ip, resources
	FundingRounds     []FundingRound    `json:"funding_rounds"`      // Funding rounds
	CapitalCalls      []CapitalCall     `json:"capital_calls"`       // Capital calls
	DistributionPolicy string           `json:"distribution_policy"` // Distribution policy
	ExitStrategy      string            `json:"exit_strategy"`       // Exit strategy
	Valuation         float64           `json:"valuation"`           // Current valuation ($M)
}

// FundingRound represents a funding round
type FundingRound struct {
	Round       string    `json:"round"`        // Seed, Series A, B, C
	Amount      float64   `json:"amount"`       // Amount raised ($M)
	Valuation   float64   `json:"valuation"`    // Post-money valuation ($M)
	Date        time.Time `json:"date"`
	Investors   []string  `json:"investors"`    // Investor names
	Terms       string    `json:"terms"`        // Key terms
}

// CapitalCall represents a capital call
type CapitalCall struct {
	CallNumber  int       `json:"call_number"`
	Amount      float64   `json:"amount"`       // Amount ($M)
	DueDate     time.Time `json:"due_date"`
	Purpose     string    `json:"purpose"`
	Status      string    `json:"status"`       // pending, paid
	PaidDate    time.Time `json:"paid_date"`
}

// GovernanceModel represents governance structure
type GovernanceModel struct {
	Structure       string          `json:"structure"`        // LLC, Corporation, Partnership
	Board           []BoardMember   `json:"board"`            // Board members
	VotingRights    VotingRights    `json:"voting_rights"`    // Voting rights
	DecisionRights  []DecisionRight `json:"decision_rights"`  // Decision rights
	MeetingCadence  string          `json:"meeting_cadence"`  // Meeting frequency
	ReportingReqs   []string        `json:"reporting_reqs"`   // Reporting requirements
	AuditRights     string          `json:"audit_rights"`     // Audit rights
	DissolutionTerms string         `json:"dissolution_terms"` // Dissolution terms
}

// BoardMember represents a board member
type BoardMember struct {
	Name        string `json:"name"`
	Organization string `json:"organization"`
	Title       string `json:"title"`
	Role        string `json:"role"`        // Chair, member, observer
	Appointed   time.Time `json:"appointed"`
	Term        int    `json:"term"`        // Term length (years)
}

// VotingRights represents voting rights structure
type VotingRights struct {
	NovaCronVotes  float64 `json:"novacron_votes"`   // NovaCron voting %
	PartnerVotes   float64 `json:"partner_votes"`    // Partner voting %
	SuperMajority  float64 `json:"super_majority"`   // Super majority %
	VetoRights     []string `json:"veto_rights"`     // Veto rights
	QuorumReq      float64 `json:"quorum_req"`       // Quorum requirement %
}

// DecisionRight represents decision rights
type DecisionRight struct {
	Category      string   `json:"category"`       // strategic, operational, financial
	Description   string   `json:"description"`
	Authority     string   `json:"authority"`      // board, ceo, majority
	Threshold     float64  `json:"threshold"`      // Voting threshold %
}

// IPRights represents intellectual property rights
type IPRights struct {
	BackgroundIP    []IPAsset  `json:"background_ip"`     // Pre-existing IP
	ForegroundIP    []IPAsset  `json:"foreground_ip"`     // Newly created IP
	LicenseGrants   []License  `json:"license_grants"`    // License grants
	Ownership       string     `json:"ownership"`         // joint, separate, assigned
	PatentStrategy  string     `json:"patent_strategy"`   // Patent filing strategy
	TradeSecrets    []string   `json:"trade_secrets"`     // Protected trade secrets
	OpenSource      []string   `json:"open_source"`       // Open source contributions
	IPDisputeRes    string     `json:"ip_dispute_res"`    // IP dispute resolution
}

// IPAsset represents an IP asset
type IPAsset struct {
	Type         string    `json:"type"`          // patent, trademark, copyright, trade_secret
	Name         string    `json:"name"`
	Description  string    `json:"description"`
	Owner        string    `json:"owner"`
	Value        float64   `json:"value"`         // Estimated value ($M)
	FilingDate   time.Time `json:"filing_date"`
	GrantDate    time.Time `json:"grant_date"`
	ExpiryDate   time.Time `json:"expiry_date"`
	Jurisdiction string    `json:"jurisdiction"`
	Status       string    `json:"status"`        // pending, granted, expired
}

// License represents a license grant
type License struct {
	Type           string    `json:"type"`            // exclusive, non-exclusive
	Licensor       string    `json:"licensor"`
	Licensee       string    `json:"licensee"`
	Technology     string    `json:"technology"`
	Territory      string    `json:"territory"`
	Field          string    `json:"field"`           // Field of use
	Term           int       `json:"term"`            // Term (years)
	Royalty        float64   `json:"royalty"`         // Royalty rate %
	MinimumRoyalty float64   `json:"minimum_royalty"` // Minimum annual royalty ($K)
	StartDate      time.Time `json:"start_date"`
	EndDate        time.Time `json:"end_date"`
	Sublicense     bool      `json:"sublicense"`      // Sublicense rights
}

// Performance represents performance metrics
type Performance struct {
	RevenueMilestones   []Milestone  `json:"revenue_milestones"`    // Revenue milestones
	TechMilestones      []Milestone  `json:"tech_milestones"`       // Technology milestones
	ProductLaunches     []Product    `json:"product_launches"`      // Product launches
	Patents             int          `json:"patents"`               // Patents filed
	Publications        int          `json:"publications"`          // Research publications
	Customers           int          `json:"customers"`             // Customer count
	Partnerships        int          `json:"partnerships"`          // Sub-partnerships
	TeamSize            int          `json:"team_size"`             // Team size
	BurnRate            float64      `json:"burn_rate"`             // Monthly burn ($M)
	Runway              int          `json:"runway"`                // Runway (months)
	KeyMetrics          map[string]float64 `json:"key_metrics"`     // KPIs
}

// Milestone represents a milestone
type Milestone struct {
	Name         string    `json:"name"`
	Description  string    `json:"description"`
	Type         string    `json:"type"`          // revenue, technology, customer
	Target       float64   `json:"target"`        // Target value
	Actual       float64   `json:"actual"`        // Actual value
	TargetDate   time.Time `json:"target_date"`
	AchievedDate time.Time `json:"achieved_date"`
	Status       string    `json:"status"`        // pending, achieved, missed
	Impact       string    `json:"impact"`        // Impact description
}

// Product represents a product launch
type Product struct {
	Name         string    `json:"name"`
	Description  string    `json:"description"`
	LaunchDate   time.Time `json:"launch_date"`
	Market       string    `json:"market"`
	Revenue      float64   `json:"revenue"`       // Revenue ($M)
	Customers    int       `json:"customers"`     // Customer count
	GrowthRate   float64   `json:"growth_rate"`   // YoY growth %
	MarketShare  float64   `json:"market_share"`  // Market share %
}

// Timeline represents initiative timeline
type Timeline struct {
	StartDate       time.Time `json:"start_date"`
	PlannedEnd      time.Time `json:"planned_end"`
	ForecastEnd     time.Time `json:"forecast_end"`
	ActualEnd       time.Time `json:"actual_end"`
	Duration        int       `json:"duration"`        // Duration (months)
	Phases          []Phase   `json:"phases"`          // Timeline phases
	ExtensionOption bool      `json:"extension_option"` // Extension option
	ExtensionTerms  string    `json:"extension_terms"`  // Extension terms
}

// Phase represents a timeline phase
type Phase struct {
	Name        string    `json:"name"`
	Description string    `json:"description"`
	StartDate   time.Time `json:"start_date"`
	EndDate     time.Time `json:"end_date"`
	Deliverables []string `json:"deliverables"`
	Status      string    `json:"status"`        // planned, in_progress, complete
	Progress    float64   `json:"progress"`      // Progress %
}

// Deliverable represents a deliverable
type Deliverable struct {
	Name         string    `json:"name"`
	Description  string    `json:"description"`
	Type         string    `json:"type"`          // report, product, technology
	Owner        string    `json:"owner"`
	DueDate      time.Time `json:"due_date"`
	DeliveryDate time.Time `json:"delivery_date"`
	Status       string    `json:"status"`        // pending, in_progress, delivered
	Quality      float64   `json:"quality"`       // Quality score (0-100)
}

// Stakeholder represents a stakeholder
type Stakeholder struct {
	Name         string   `json:"name"`
	Organization string   `json:"organization"`
	Role         string   `json:"role"`
	Influence    string   `json:"influence"`     // high, medium, low
	Interest     string   `json:"interest"`      // high, medium, low
	Engagement   string   `json:"engagement"`    // supporter, neutral, blocker
	Concerns     []string `json:"concerns"`
	Actions      []string `json:"actions"`       // Engagement actions
}

// Risk represents a risk
type Risk struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Category    string    `json:"category"`      // strategic, financial, operational, technical
	Impact      string    `json:"impact"`        // low, medium, high, critical
	Probability string    `json:"probability"`   // low, medium, high
	RiskScore   float64   `json:"risk_score"`    // Risk score
	Mitigation  string    `json:"mitigation"`    // Mitigation plan
	Owner       string    `json:"owner"`
	Status      string    `json:"status"`        // open, mitigated, closed
	Date        time.Time `json:"date"`
}

// EcosystemManager manages ecosystem value creation
type EcosystemManager struct {
	initiatives  map[string]*EcosystemInitiative
	mu           sync.RWMutex
	licensing    *LicensingEngine
	jvManagement *JVManagementEngine
	innovation   *InnovationEngine
	metrics      *EcosystemMetrics
}

// EcosystemMetrics tracks ecosystem metrics
type EcosystemMetrics struct {
	TotalInitiatives   int     `json:"total_initiatives"`
	ActiveInitiatives  int     `json:"active_initiatives"`
	ByType             map[EcosystemType]int `json:"by_type"`
	TotalValue         float64 `json:"total_value"`          // Total value created ($M)
	TotalRevenue       float64 `json:"total_revenue"`        // Total revenue ($M)
	TotalInvestment    float64 `json:"total_investment"`     // Total investment ($M)
	AverageROI         float64 `json:"average_roi"`          // Average ROI %
	PatentsFiled       int     `json:"patents_filed"`        // Total patents filed
	ProductsLaunched   int     `json:"products_launched"`    // Total products launched
	PartnerCount       int     `json:"partner_count"`        // Total partners
	AcademicPartners   int     `json:"academic_partners"`    // Academic partners
	GovernmentPartners int     `json:"government_partners"`  // Government partners
}

// NewEcosystemManager creates a new ecosystem manager
func NewEcosystemManager() *EcosystemManager {
	return &EcosystemManager{
		initiatives:  make(map[string]*EcosystemInitiative),
		licensing:    NewLicensingEngine(),
		jvManagement: NewJVManagementEngine(),
		innovation:   NewInnovationEngine(),
		metrics: &EcosystemMetrics{
			ByType: make(map[EcosystemType]int),
		},
	}
}

// CreateInitiative creates a new ecosystem initiative
func (em *EcosystemManager) CreateInitiative(ctx context.Context, initiative *EcosystemInitiative) error {
	em.mu.Lock()
	defer em.mu.Unlock()

	initiative.CreatedAt = time.Now()
	initiative.UpdatedAt = time.Now()

	em.initiatives[initiative.ID] = initiative
	em.updateMetrics()

	return nil
}

// UpdateInitiative updates an existing initiative
func (em *EcosystemManager) UpdateInitiative(ctx context.Context, initiativeID string, updates *EcosystemInitiative) error {
	em.mu.Lock()
	defer em.mu.Unlock()

	initiative, exists := em.initiatives[initiativeID]
	if !exists {
		return fmt.Errorf("initiative not found: %s", initiativeID)
	}

	if updates.Status != "" {
		initiative.Status = updates.Status
	}
	if len(updates.Objectives) > 0 {
		initiative.Objectives = updates.Objectives
	}

	initiative.UpdatedAt = time.Now()
	em.updateMetrics()

	return nil
}

// TrackValue tracks value creation
func (em *EcosystemManager) TrackValue(ctx context.Context, initiativeID string, value ValueMetrics) error {
	em.mu.Lock()
	defer em.mu.Unlock()

	initiative, exists := em.initiatives[initiativeID]
	if !exists {
		return fmt.Errorf("initiative not found: %s", initiativeID)
	}

	initiative.Value = value
	initiative.UpdatedAt = time.Now()
	em.updateMetrics()

	return nil
}

// RecordMilestone records a milestone achievement
func (em *EcosystemManager) RecordMilestone(ctx context.Context, initiativeID string, milestone Milestone) error {
	em.mu.Lock()
	defer em.mu.Unlock()

	initiative, exists := em.initiatives[initiativeID]
	if !exists {
		return fmt.Errorf("initiative not found: %s", initiativeID)
	}

	milestone.AchievedDate = time.Now()
	milestone.Status = "achieved"

	// Add to appropriate milestone category
	switch milestone.Type {
	case "revenue":
		initiative.Performance.RevenueMilestones = append(initiative.Performance.RevenueMilestones, milestone)
	case "technology":
		initiative.Performance.TechMilestones = append(initiative.Performance.TechMilestones, milestone)
	}

	initiative.UpdatedAt = time.Now()

	return nil
}

// GetInitiative retrieves an initiative by ID
func (em *EcosystemManager) GetInitiative(initiativeID string) (*EcosystemInitiative, error) {
	em.mu.RLock()
	defer em.mu.RUnlock()

	initiative, exists := em.initiatives[initiativeID]
	if !exists {
		return nil, fmt.Errorf("initiative not found: %s", initiativeID)
	}

	return initiative, nil
}

// ListInitiatives returns all initiatives with optional filtering
func (em *EcosystemManager) ListInitiatives(initiativeType EcosystemType, status EcosystemStatus) []*EcosystemInitiative {
	em.mu.RLock()
	defer em.mu.RUnlock()

	var initiatives []*EcosystemInitiative
	for _, i := range em.initiatives {
		if initiativeType != "" && i.Type != initiativeType {
			continue
		}
		if status != "" && i.Status != status {
			continue
		}
		initiatives = append(initiatives, i)
	}

	return initiatives
}

// GetMetrics returns ecosystem metrics
func (em *EcosystemManager) GetMetrics() *EcosystemMetrics {
	em.mu.RLock()
	defer em.mu.RUnlock()
	return em.metrics
}

// updateMetrics updates ecosystem metrics (must be called with lock held)
func (em *EcosystemManager) updateMetrics() {
	em.metrics.TotalInitiatives = len(em.initiatives)
	em.metrics.ActiveInitiatives = 0
	em.metrics.ByType = make(map[EcosystemType]int)
	em.metrics.TotalValue = 0
	em.metrics.TotalRevenue = 0
	em.metrics.TotalInvestment = 0
	totalROI := 0.0
	em.metrics.PatentsFiled = 0
	em.metrics.ProductsLaunched = 0
	partnerMap := make(map[string]bool)
	em.metrics.AcademicPartners = 0
	em.metrics.GovernmentPartners = 0

	for _, i := range em.initiatives {
		if i.Status == StatusActive {
			em.metrics.ActiveInitiatives++
		}
		em.metrics.ByType[i.Type]++
		em.metrics.TotalValue += i.Value.TotalValue
		em.metrics.TotalRevenue += i.Value.RevenueGenerated
		em.metrics.TotalInvestment += i.Investment.TotalInvestment
		totalROI += i.Value.ROI
		em.metrics.PatentsFiled += i.Performance.Patents
		em.metrics.ProductsLaunched += len(i.Performance.ProductLaunches)

		// Count unique partners
		for _, p := range i.Partners {
			partnerMap[p.Name] = true
			if p.Type == "university" {
				em.metrics.AcademicPartners++
			}
			if p.Type == "government" {
				em.metrics.GovernmentPartners++
			}
		}
	}

	em.metrics.PartnerCount = len(partnerMap)

	if len(em.initiatives) > 0 {
		em.metrics.AverageROI = totalROI / float64(len(em.initiatives))
	}
}

// LicensingEngine manages technology licensing
type LicensingEngine struct{}

func NewLicensingEngine() *LicensingEngine {
	return &LicensingEngine{}
}

// JVManagementEngine manages joint ventures
type JVManagementEngine struct{}

func NewJVManagementEngine() *JVManagementEngine {
	return &JVManagementEngine{}
}

// InnovationEngine manages co-innovation programs
type InnovationEngine struct{}

func NewInnovationEngine() *InnovationEngine {
	return &InnovationEngine{}
}

// ExportToJSON exports ecosystem data to JSON
func (em *EcosystemManager) ExportToJSON() ([]byte, error) {
	em.mu.RLock()
	defer em.mu.RUnlock()

	data := struct {
		Initiatives []*EcosystemInitiative `json:"initiatives"`
		Metrics     *EcosystemMetrics      `json:"metrics"`
	}{
		Initiatives: make([]*EcosystemInitiative, 0, len(em.initiatives)),
		Metrics:     em.metrics,
	}

	for _, i := range em.initiatives {
		data.Initiatives = append(data.Initiatives, i)
	}

	return json.MarshalIndent(data, "", "  ")
}
