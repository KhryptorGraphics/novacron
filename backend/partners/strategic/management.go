// Package strategic provides strategic partnership management capabilities
// Supports 20+ partnerships: Cloud providers, hardware vendors, telcos, system integrators
package strategic

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// PartnershipType defines the type of strategic partnership
type PartnershipType string

const (
	TypeCloudProvider   PartnershipType = "cloud_provider"    // AWS, Azure, GCP, Oracle
	TypeHardwareVendor  PartnershipType = "hardware_vendor"   // Intel, AMD, NVIDIA, IBM
	TypeTelco           PartnershipType = "telecom"           // Verizon, AT&T, T-Mobile
	TypeSystemIntegrator PartnershipType = "system_integrator" // Accenture, Deloitte, IBM Services
	TypeTechnology      PartnershipType = "technology"        // Technology partnerships
	TypeChannel         PartnershipType = "channel"           // Channel partners
	TypeOEM             PartnershipType = "oem"               // OEM partnerships
	TypeStrategic       PartnershipType = "strategic"         // Other strategic partnerships
)

// PartnershipTier defines the partnership tier level
type PartnershipTier string

const (
	TierPlatinum PartnershipTier = "platinum" // Top-tier strategic partnership
	TierGold     PartnershipTier = "gold"     // Gold-level partnership
	TierSilver   PartnershipTier = "silver"   // Silver-level partnership
	TierBronze   PartnershipTier = "bronze"   // Bronze-level partnership
)

// PartnershipStatus defines the status of a partnership
type PartnershipStatus string

const (
	StatusProspect   PartnershipStatus = "prospect"    // Prospective partner
	StatusNegotiation PartnershipStatus = "negotiation" // In negotiation
	StatusActive     PartnershipStatus = "active"      // Active partnership
	StatusRenewal    PartnershipStatus = "renewal"     // Up for renewal
	StatusInactive   PartnershipStatus = "inactive"    // Inactive partnership
)

// Partnership represents a strategic partnership
type Partnership struct {
	ID                string            `json:"id"`
	Name              string            `json:"name"`
	Type              PartnershipType   `json:"type"`
	Tier              PartnershipTier   `json:"tier"`
	Status            PartnershipStatus `json:"status"`
	Description       string            `json:"description"`
	Objectives        []string          `json:"objectives"`        // Partnership objectives
	Value             PartnershipValue  `json:"value"`             // Value metrics
	Agreement         AgreementDetails  `json:"agreement"`         // Agreement details
	CoSelling         CoSellingProgram  `json:"co_selling"`        // Co-selling program
	Technical         TechnicalInteg    `json:"technical"`         // Technical integration
	Marketing         MarketingProgram  `json:"marketing"`         // Marketing programs
	Support           SupportProgram    `json:"support"`           // Support programs
	Performance       PartnerPerformance `json:"performance"`      // Performance metrics
	Contacts          []PartnerContact  `json:"contacts"`          // Partner contacts
	Activities        []PartnerActivity `json:"activities"`        // Recent activities
	Documents         []string          `json:"documents"`         // Contract documents
	CreatedAt         time.Time         `json:"created_at"`
	UpdatedAt         time.Time         `json:"updated_at"`
	RenewalDate       time.Time         `json:"renewal_date"`
}

// PartnershipValue represents the value metrics of a partnership
type PartnershipValue struct {
	AnnualRevenue      float64 `json:"annual_revenue"`       // Annual revenue ($M)
	PipelineValue      float64 `json:"pipeline_value"`       // Pipeline value ($M)
	CustomerCount      int     `json:"customer_count"`       // Joint customers
	OpportunityCount   int     `json:"opportunity_count"`    // Active opportunities
	DealSize           float64 `json:"deal_size"`            // Average deal size ($K)
	WinRate            float64 `json:"win_rate"`             // Win rate %
	CoSellMultiplier   float64 `json:"co_sell_multiplier"`   // Revenue multiplier
	StrategicValue     float64 `json:"strategic_value"`      // Strategic value score (0-100)
	MarketAccess       float64 `json:"market_access"`        // Market access value ($M)
	TechnologyAccess   float64 `json:"technology_access"`    // Technology access value ($M)
}

// AgreementDetails represents partnership agreement details
type AgreementDetails struct {
	Type              string    `json:"type"`               // Agreement type
	SignedDate        time.Time `json:"signed_date"`
	EffectiveDate     time.Time `json:"effective_date"`
	ExpirationDate    time.Time `json:"expiration_date"`
	AutoRenewal       bool      `json:"auto_renewal"`
	TermLength        int       `json:"term_length"`        // Term length (months)
	NoticeRequired    int       `json:"notice_required"`    // Notice period (days)
	RevenueShare      float64   `json:"revenue_share"`      // Revenue share %
	CommitmentValue   float64   `json:"commitment_value"`   // Commitment value ($M)
	MinimumCommitment float64   `json:"minimum_commitment"` // Minimum commitment ($M)
	KeyTerms          []string  `json:"key_terms"`          // Key agreement terms
	Exclusivity       string    `json:"exclusivity"`        // Exclusivity terms
}

// CoSellingProgram represents co-selling program details
type CoSellingProgram struct {
	Enabled           bool               `json:"enabled"`
	Status            string             `json:"status"`             // Status
	JointSolutions    []JointSolution    `json:"joint_solutions"`    // Joint solutions
	MarketplaceListings []MarketplaceList `json:"marketplace_listings"` // Marketplace listings
	CoSellMotions     []CoSellMotion     `json:"co_sell_motions"`    // Co-sell motions
	SalesCertification bool              `json:"sales_certification"` // Sales team certified
	LeadSharing       LeadSharingConfig  `json:"lead_sharing"`       // Lead sharing config
	DealRegistration  DealRegConfig      `json:"deal_registration"`  // Deal registration
	Incentives        IncentiveProgram   `json:"incentives"`         // Incentive programs
}

// JointSolution represents a joint solution offering
type JointSolution struct {
	Name             string   `json:"name"`
	Description      string   `json:"description"`
	UseCase          string   `json:"use_case"`
	TargetMarket     string   `json:"target_market"`
	ValueProp        string   `json:"value_prop"`
	TechnicalDoc     string   `json:"technical_doc"`
	SalesEnablement  string   `json:"sales_enablement"`
	LaunchDate       time.Time `json:"launch_date"`
	Status           string   `json:"status"`
	Pipeline         float64  `json:"pipeline"`          // Pipeline value ($M)
	ClosedDeals      int      `json:"closed_deals"`      // Closed deals count
	Revenue          float64  `json:"revenue"`           // Revenue generated ($M)
}

// MarketplaceList represents a marketplace listing
type MarketplaceList struct {
	Marketplace      string    `json:"marketplace"`       // AWS, Azure, GCP
	ListingURL       string    `json:"listing_url"`
	Status           string    `json:"status"`            // live, pending, draft
	PublishDate      time.Time `json:"publish_date"`
	Category         string    `json:"category"`
	Pricing          string    `json:"pricing"`
	MonthlyLeads     int       `json:"monthly_leads"`     // Monthly leads
	ConversionRate   float64   `json:"conversion_rate"`   // Conversion rate %
	MonthlyRevenue   float64   `json:"monthly_revenue"`   // Monthly revenue ($K)
}

// CoSellMotion represents a co-selling motion/playbook
type CoSellMotion struct {
	Name            string   `json:"name"`
	Description     string   `json:"description"`
	TargetAccount   string   `json:"target_account"`    // Enterprise, mid-market, SMB
	MotionType      string   `json:"motion_type"`       // Outbound, inbound, referral
	Playbook        string   `json:"playbook"`          // Playbook document
	SalesProcess    []string `json:"sales_process"`     // Sales process steps
	PartnerRole     string   `json:"partner_role"`      // Partner role
	NovaCronRole    string   `json:"novacron_role"`     // NovaCron role
	SuccessCriteria []string `json:"success_criteria"`  // Success criteria
	ActiveDeals     int      `json:"active_deals"`      // Active deals using motion
	WinRate         float64  `json:"win_rate"`          // Win rate %
}

// LeadSharingConfig represents lead sharing configuration
type LeadSharingConfig struct {
	Enabled          bool     `json:"enabled"`
	LeadTypes        []string `json:"lead_types"`        // MQL, SQL, opportunity
	SharingCriteria  []string `json:"sharing_criteria"`  // Criteria for sharing
	ResponseSLA      int      `json:"response_sla"`      // Response time (hours)
	QualificationSLA int      `json:"qualification_sla"` // Qualification time (hours)
	LeadsShared      int      `json:"leads_shared"`      // Total leads shared
	LeadsAccepted    int      `json:"leads_accepted"`    // Leads accepted
	LeadsConverted   int      `json:"leads_converted"`   // Leads converted
	ConversionRate   float64  `json:"conversion_rate"`   // Conversion rate %
}

// DealRegConfig represents deal registration configuration
type DealRegConfig struct {
	Enabled           bool    `json:"enabled"`
	ProtectionPeriod  int     `json:"protection_period"`  // Days
	ApprovalSLA       int     `json:"approval_sla"`       // Approval time (hours)
	Incentive         float64 `json:"incentive"`          // Incentive %
	ActiveRegistrations int   `json:"active_registrations"` // Active registrations
	ApprovedDeals     int     `json:"approved_deals"`     // Approved deals
	WonDeals          int     `json:"won_deals"`          // Won deals
}

// IncentiveProgram represents partner incentive programs
type IncentiveProgram struct {
	SPIFFs           []SPIFF           `json:"spiffs"`            // Sales incentives
	Rebates          []Rebate          `json:"rebates"`           // Volume rebates
	MDF              MDFProgram        `json:"mdf"`               // Market development funds
	CoopMarketing    float64           `json:"coop_marketing"`    // Co-op marketing funds ($K)
	TotalIncentives  float64           `json:"total_incentives"`  // Total incentives paid ($M)
	ROI              float64           `json:"roi"`               // Incentive ROI
}

// SPIFF represents a sales incentive
type SPIFF struct {
	Name         string    `json:"name"`
	Description  string    `json:"description"`
	Amount       float64   `json:"amount"`        // Incentive amount ($)
	Criteria     []string  `json:"criteria"`      // Qualification criteria
	StartDate    time.Time `json:"start_date"`
	EndDate      time.Time `json:"end_date"`
	PaidOut      float64   `json:"paid_out"`      // Total paid out ($K)
	DealsClosed  int       `json:"deals_closed"`  // Deals closed
}

// Rebate represents a volume rebate program
type Rebate struct {
	Name         string    `json:"name"`
	Tier         string    `json:"tier"`          // Bronze, silver, gold, platinum
	ThresholdMin float64   `json:"threshold_min"` // Min revenue ($M)
	ThresholdMax float64   `json:"threshold_max"` // Max revenue ($M)
	RebateRate   float64   `json:"rebate_rate"`   // Rebate %
	EarnedRebate float64   `json:"earned_rebate"` // Earned rebate ($K)
}

// MDFProgram represents market development fund program
type MDFProgram struct {
	AnnualBudget     float64       `json:"annual_budget"`     // Annual MDF budget ($K)
	Allocated        float64       `json:"allocated"`         // Allocated funds ($K)
	Utilized         float64       `json:"utilized"`          // Utilized funds ($K)
	Approved         float64       `json:"approved"`          // Approved requests ($K)
	Pending          float64       `json:"pending"`           // Pending requests ($K)
	Campaigns        []MDFCampaign `json:"campaigns"`         // MDF campaigns
}

// MDFCampaign represents an MDF campaign
type MDFCampaign struct {
	Name         string    `json:"name"`
	Type         string    `json:"type"`          // Event, webinar, content, advertising
	Budget       float64   `json:"budget"`        // Budget ($K)
	Status       string    `json:"status"`        // approved, in_progress, complete
	StartDate    time.Time `json:"start_date"`
	EndDate      time.Time `json:"end_date"`
	Leads        int       `json:"leads"`         // Leads generated
	Pipeline     float64   `json:"pipeline"`      // Pipeline generated ($M)
	ROI          float64   `json:"roi"`           // Campaign ROI
}

// TechnicalInteg represents technical integration details
type TechnicalInteg struct {
	IntegrationType  []string         `json:"integration_type"`   // API, SDK, platform
	CertifiedProducts []CertifiedProd  `json:"certified_products"` // Certified products
	TechnicalDocs    []TechnicalDoc   `json:"technical_docs"`     // Technical documentation
	DevSupport       DevSupportProgram `json:"dev_support"`       // Developer support
	JointRoadmap     []RoadmapItem    `json:"joint_roadmap"`      // Joint technology roadmap
	IntegrationStatus string          `json:"integration_status"` // Status
}

// CertifiedProd represents a certified product
type CertifiedProd struct {
	Name             string    `json:"name"`
	Version          string    `json:"version"`
	CertificationLevel string  `json:"certification_level"` // Basic, advanced, premier
	CertifiedDate    time.Time `json:"certified_date"`
	ExpirationDate   time.Time `json:"expiration_date"`
	TestResults      string    `json:"test_results"`
	BadgeURL         string    `json:"badge_url"`
}

// TechnicalDoc represents technical documentation
type TechnicalDoc struct {
	Title        string    `json:"title"`
	Type         string    `json:"type"`          // Integration guide, API docs, whitepaper
	URL          string    `json:"url"`
	Version      string    `json:"version"`
	PublishDate  time.Time `json:"publish_date"`
	Downloads    int       `json:"downloads"`
}

// DevSupportProgram represents developer support program
type DevSupportProgram struct {
	TechnicalContact string   `json:"technical_contact"` // Technical contact
	SupportLevel     string   `json:"support_level"`     // Basic, enhanced, premier
	ResponseSLA      int      `json:"response_sla"`      // Response time (hours)
	TicketsOpen      int      `json:"tickets_open"`      // Open support tickets
	TicketsClosed    int      `json:"tickets_closed"`    // Closed tickets
	SatisfactionScore float64 `json:"satisfaction_score"` // CSAT score
	SlackChannel     string   `json:"slack_channel"`     // Shared Slack channel
	MeetingCadence   string   `json:"meeting_cadence"`   // Meeting frequency
}

// RoadmapItem represents a joint roadmap item
type RoadmapItem struct {
	Feature       string    `json:"feature"`
	Description   string    `json:"description"`
	Target        string    `json:"target"`        // Quarter/year
	Status        string    `json:"status"`        // planned, in_progress, complete
	Owner         string    `json:"owner"`         // NovaCron or partner
	Dependencies  []string  `json:"dependencies"`
	LaunchDate    time.Time `json:"launch_date"`
}

// MarketingProgram represents joint marketing programs
type MarketingProgram struct {
	CoBranding       bool              `json:"co_branding"`        // Co-branding enabled
	JointEvents      []JointEvent      `json:"joint_events"`       // Joint events
	ContentMarketing []ContentAsset    `json:"content_marketing"`  // Content marketing
	PRCampaigns      []PRCampaign      `json:"pr_campaigns"`       // PR campaigns
	SocialMedia      SocialMediaProg   `json:"social_media"`       // Social media
	Webinars         []Webinar         `json:"webinars"`           // Webinars
	CaseStudies      []CaseStudy       `json:"case_studies"`       // Case studies
	Budget           float64           `json:"budget"`             // Marketing budget ($K)
	LeadsGenerated   int               `json:"leads_generated"`    // Total leads
	PipelineGenerated float64          `json:"pipeline_generated"` // Pipeline ($M)
}

// JointEvent represents a joint marketing event
type JointEvent struct {
	Name          string    `json:"name"`
	Type          string    `json:"type"`           // Conference, roadshow, workshop
	Location      string    `json:"location"`
	Date          time.Time `json:"date"`
	Attendees     int       `json:"attendees"`      // Expected attendees
	ActualAttendees int     `json:"actual_attendees"` // Actual attendees
	Leads         int       `json:"leads"`          // Leads generated
	Pipeline      float64   `json:"pipeline"`       // Pipeline generated ($M)
	Cost          float64   `json:"cost"`           // Event cost ($K)
	ROI           float64   `json:"roi"`            // Event ROI
}

// ContentAsset represents a marketing content asset
type ContentAsset struct {
	Title        string    `json:"title"`
	Type         string    `json:"type"`         // Whitepaper, ebook, blog, video
	PublishDate  time.Time `json:"publish_date"`
	URL          string    `json:"url"`
	Views        int       `json:"views"`        // Total views
	Downloads    int       `json:"downloads"`    // Total downloads
	Leads        int       `json:"leads"`        // Leads generated
	Engagement   float64   `json:"engagement"`   // Engagement score
}

// PRCampaign represents a PR campaign
type PRCampaign struct {
	Name         string    `json:"name"`
	Type         string    `json:"type"`          // Press release, media briefing
	LaunchDate   time.Time `json:"launch_date"`
	MediaOutlets []string  `json:"media_outlets"` // Media outlets
	Articles     int       `json:"articles"`      // Articles published
	Reach        int       `json:"reach"`         // Total reach
	Impressions  int       `json:"impressions"`   // Total impressions
	Sentiment    float64   `json:"sentiment"`     // Sentiment score (0-100)
}

// SocialMediaProg represents social media programs
type SocialMediaProg struct {
	Campaigns    []SocialCampaign `json:"campaigns"`     // Social campaigns
	Followers    int              `json:"followers"`     // Combined followers
	Engagement   float64          `json:"engagement"`    // Engagement rate %
	Impressions  int              `json:"impressions"`   // Total impressions
	Clicks       int              `json:"clicks"`        // Total clicks
	Leads        int              `json:"leads"`         // Leads generated
}

// SocialCampaign represents a social media campaign
type SocialCampaign struct {
	Name        string    `json:"name"`
	Platform    string    `json:"platform"`     // LinkedIn, Twitter, Facebook
	StartDate   time.Time `json:"start_date"`
	EndDate     time.Time `json:"end_date"`
	Posts       int       `json:"posts"`        // Number of posts
	Impressions int       `json:"impressions"`  // Total impressions
	Engagement  float64   `json:"engagement"`   // Engagement rate %
	Clicks      int       `json:"clicks"`       // Total clicks
	Leads       int       `json:"leads"`        // Leads generated
}

// Webinar represents a joint webinar
type Webinar struct {
	Title        string    `json:"title"`
	Date         time.Time `json:"date"`
	Registrants  int       `json:"registrants"`  // Registrants
	Attendees    int       `json:"attendees"`    // Attendees
	Recording    string    `json:"recording"`    // Recording URL
	Views        int       `json:"views"`        // Recording views
	Leads        int       `json:"leads"`        // Leads generated
	Pipeline     float64   `json:"pipeline"`     // Pipeline generated ($M)
}

// CaseStudy represents a customer case study
type CaseStudy struct {
	Title        string    `json:"title"`
	Customer     string    `json:"customer"`
	Industry     string    `json:"industry"`
	UseCase      string    `json:"use_case"`
	Results      []string  `json:"results"`      // Key results
	PublishDate  time.Time `json:"publish_date"`
	URL          string    `json:"url"`
	Downloads    int       `json:"downloads"`    // Total downloads
	Attribution  int       `json:"attribution"`  // Deal attributions
}

// SupportProgram represents partner support programs
type SupportProgram struct {
	EnablementProgram EnablementProg   `json:"enablement_program"` // Sales enablement
	Training         TrainingProgram   `json:"training"`           // Training programs
	Certification    CertificationProg `json:"certification"`      // Certification programs
	PartnerPortal    PortalAccess      `json:"partner_portal"`     // Partner portal access
	Resources        []Resource        `json:"resources"`          // Partner resources
}

// EnablementProg represents sales enablement program
type EnablementProg struct {
	SalesPlaybooks   []string  `json:"sales_playbooks"`    // Sales playbooks
	BattleCards      []string  `json:"battle_cards"`       // Competitive battle cards
	DemoEnvironments []string  `json:"demo_environments"`  // Demo environments
	ROICalculators   []string  `json:"roi_calculators"`    // ROI calculators
	PitchDecks       []string  `json:"pitch_decks"`        // Pitch decks
	TrainedReps      int       `json:"trained_reps"`       // Trained sales reps
	CertifiedReps    int       `json:"certified_reps"`     // Certified reps
}

// TrainingProgram represents partner training programs
type TrainingProgram struct {
	Courses          []TrainingCourse `json:"courses"`           // Training courses
	TotalEnrollments int              `json:"total_enrollments"` // Total enrollments
	Completions      int              `json:"completions"`       // Course completions
	Satisfaction     float64          `json:"satisfaction"`      // Satisfaction score
}

// TrainingCourse represents a training course
type TrainingCourse struct {
	Name         string    `json:"name"`
	Type         string    `json:"type"`          // Technical, sales, business
	Duration     int       `json:"duration"`      // Duration (hours)
	Format       string    `json:"format"`        // Online, in-person, hybrid
	LaunchDate   time.Time `json:"launch_date"`
	Enrollments  int       `json:"enrollments"`   // Total enrollments
	Completions  int       `json:"completions"`   // Total completions
	Rating       float64   `json:"rating"`        // Course rating (0-5)
}

// CertificationProg represents certification programs
type CertificationProg struct {
	Certifications []Certification `json:"certifications"`  // Available certifications
	TotalCertified int             `json:"total_certified"` // Total certified individuals
}

// Certification represents a certification
type Certification struct {
	Name         string    `json:"name"`
	Level        string    `json:"level"`         // Associate, professional, expert
	Requirements []string  `json:"requirements"`  // Requirements
	ExamDuration int       `json:"exam_duration"` // Exam duration (minutes)
	PassRate     float64   `json:"pass_rate"`     // Pass rate %
	Certified    int       `json:"certified"`     // Total certified
	ExpirationMonths int   `json:"expiration_months"` // Certification validity (months)
}

// PortalAccess represents partner portal access
type PortalAccess struct {
	URL              string    `json:"url"`
	Enabled          bool      `json:"enabled"`
	Features         []string  `json:"features"`          // Portal features
	Users            int       `json:"users"`             // Portal users
	ActiveUsers      int       `json:"active_users"`      // Active users (30 days)
	MonthlyLogins    int       `json:"monthly_logins"`    // Monthly logins
	ResourceDownloads int      `json:"resource_downloads"` // Resource downloads
}

// Resource represents a partner resource
type Resource struct {
	Title        string    `json:"title"`
	Type         string    `json:"type"`         // Document, video, tool
	Category     string    `json:"category"`     // Sales, technical, marketing
	URL          string    `json:"url"`
	PublishDate  time.Time `json:"publish_date"`
	Downloads    int       `json:"downloads"`    // Total downloads
	Views        int       `json:"views"`        // Total views
	Rating       float64   `json:"rating"`       // Resource rating (0-5)
}

// PartnerPerformance represents partner performance metrics
type PartnerPerformance struct {
	Overall          float64              `json:"overall"`           // Overall score (0-100)
	RevenueTarget    float64              `json:"revenue_target"`    // Revenue target ($M)
	RevenueActual    float64              `json:"revenue_actual"`    // Actual revenue ($M)
	RevenueAttainment float64             `json:"revenue_attainment"` // Attainment %
	Quarterly        []QuarterlyPerf      `json:"quarterly"`         // Quarterly performance
	YearlyTrend      []YearlyPerf         `json:"yearly_trend"`      // Yearly trend
	KPIs             map[string]float64   `json:"kpis"`              // Key performance indicators
	HealthScore      float64              `json:"health_score"`      // Partnership health (0-100)
	Risk             string               `json:"risk"`              // low, medium, high
}

// QuarterlyPerf represents quarterly performance
type QuarterlyPerf struct {
	Quarter      string  `json:"quarter"`       // Q1, Q2, Q3, Q4
	Year         int     `json:"year"`
	Revenue      float64 `json:"revenue"`       // Revenue ($M)
	Target       float64 `json:"target"`        // Target ($M)
	Attainment   float64 `json:"attainment"`    // Attainment %
	NewCustomers int     `json:"new_customers"` // New customers
	Pipeline     float64 `json:"pipeline"`      // Pipeline ($M)
	WinRate      float64 `json:"win_rate"`      // Win rate %
}

// YearlyPerf represents yearly performance
type YearlyPerf struct {
	Year         int     `json:"year"`
	Revenue      float64 `json:"revenue"`       // Revenue ($M)
	Growth       float64 `json:"growth"`        // YoY growth %
	Customers    int     `json:"customers"`     // Total customers
	NewCustomers int     `json:"new_customers"` // New customers
	Deals        int     `json:"deals"`         // Total deals
	AvgDealSize  float64 `json:"avg_deal_size"` // Average deal size ($K)
}

// PartnerContact represents a partner contact
type PartnerContact struct {
	Name         string   `json:"name"`
	Title        string   `json:"title"`
	Role         string   `json:"role"`          // Executive, sales, technical, marketing
	Email        string   `json:"email"`
	Phone        string   `json:"phone"`
	LinkedIn     string   `json:"linkedin"`
	IsPrimary    bool     `json:"is_primary"`    // Primary contact
	Responsibilities []string `json:"responsibilities"` // Responsibilities
}

// PartnerActivity represents a partnership activity
type PartnerActivity struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`        // Meeting, deal, event, campaign
	Description string    `json:"description"`
	Date        time.Time `json:"date"`
	Participants []string `json:"participants"` // Participants
	Outcome     string    `json:"outcome"`
	NextSteps   []string  `json:"next_steps"`
	Documents   []string  `json:"documents"`
}

// PartnershipManager manages strategic partnerships
type PartnershipManager struct {
	partnerships map[string]*Partnership
	mu           sync.RWMutex
	coSelling    *CoSellingEngine
	marketing    *MarketingEngine
	performance  *PerformanceTracker
	metrics      *PartnershipMetrics
}

// PartnershipMetrics tracks partnership metrics
type PartnershipMetrics struct {
	TotalPartnerships  int     `json:"total_partnerships"`
	ActivePartnerships int     `json:"active_partnerships"`
	ByType             map[PartnershipType]int `json:"by_type"`
	ByTier             map[PartnershipTier]int `json:"by_tier"`
	TotalRevenue       float64 `json:"total_revenue"`       // Total partnership revenue ($M)
	TotalPipeline      float64 `json:"total_pipeline"`      // Total pipeline ($M)
	TotalCustomers     int     `json:"total_customers"`     // Total joint customers
	AvgHealthScore     float64 `json:"avg_health_score"`    // Average health score
	RenewalRate        float64 `json:"renewal_rate"`        // Partnership renewal rate %
	ExpansionRate      float64 `json:"expansion_rate"`      // Revenue expansion rate %
}

// NewPartnershipManager creates a new partnership manager
func NewPartnershipManager() *PartnershipManager {
	return &PartnershipManager{
		partnerships: make(map[string]*Partnership),
		coSelling:    NewCoSellingEngine(),
		marketing:    NewMarketingEngine(),
		performance:  NewPerformanceTracker(),
		metrics: &PartnershipMetrics{
			ByType: make(map[PartnershipType]int),
			ByTier: make(map[PartnershipTier]int),
		},
	}
}

// CreatePartnership creates a new strategic partnership
func (pm *PartnershipManager) CreatePartnership(ctx context.Context, partnership *Partnership) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	partnership.CreatedAt = time.Now()
	partnership.UpdatedAt = time.Now()

	pm.partnerships[partnership.ID] = partnership
	pm.updateMetrics()

	return nil
}

// UpdatePartnership updates an existing partnership
func (pm *PartnershipManager) UpdatePartnership(ctx context.Context, partnerID string, updates *Partnership) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	partner, exists := pm.partnerships[partnerID]
	if !exists {
		return fmt.Errorf("partnership not found: %s", partnerID)
	}

	// Update fields
	if updates.Status != "" {
		partner.Status = updates.Status
	}
	if updates.Tier != "" {
		partner.Tier = updates.Tier
	}
	if len(updates.Objectives) > 0 {
		partner.Objectives = updates.Objectives
	}

	partner.UpdatedAt = time.Now()
	pm.updateMetrics()

	return nil
}

// TrackActivity logs a partnership activity
func (pm *PartnershipManager) TrackActivity(ctx context.Context, partnerID string, activity PartnerActivity) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	partner, exists := pm.partnerships[partnerID]
	if !exists {
		return fmt.Errorf("partnership not found: %s", partnerID)
	}

	activity.ID = fmt.Sprintf("activity-%d", time.Now().Unix())
	activity.Date = time.Now()
	partner.Activities = append(partner.Activities, activity)
	partner.UpdatedAt = time.Now()

	return nil
}

// UpdatePerformance updates partnership performance metrics
func (pm *PartnershipManager) UpdatePerformance(ctx context.Context, partnerID string, revenue float64, customers int) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	partner, exists := pm.partnerships[partnerID]
	if !exists {
		return fmt.Errorf("partnership not found: %s", partnerID)
	}

	partner.Value.AnnualRevenue = revenue
	partner.Value.CustomerCount = customers
	partner.UpdatedAt = time.Now()

	// Update health score
	healthScore := pm.performance.CalculateHealthScore(partner)
	partner.Performance.HealthScore = healthScore

	pm.updateMetrics()

	return nil
}

// GetPartnership retrieves a partnership by ID
func (pm *PartnershipManager) GetPartnership(partnerID string) (*Partnership, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	partner, exists := pm.partnerships[partnerID]
	if !exists {
		return nil, fmt.Errorf("partnership not found: %s", partnerID)
	}

	return partner, nil
}

// ListPartnerships returns all partnerships with optional filtering
func (pm *PartnershipManager) ListPartnerships(partnerType PartnershipType, tier PartnershipTier, status PartnershipStatus) []*Partnership {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	var partnerships []*Partnership
	for _, p := range pm.partnerships {
		if partnerType != "" && p.Type != partnerType {
			continue
		}
		if tier != "" && p.Tier != tier {
			continue
		}
		if status != "" && p.Status != status {
			continue
		}
		partnerships = append(partnerships, p)
	}

	return partnerships
}

// GetMetrics returns partnership metrics
func (pm *PartnershipManager) GetMetrics() *PartnershipMetrics {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	return pm.metrics
}

// updateMetrics updates partnership metrics (must be called with lock held)
func (pm *PartnershipManager) updateMetrics() {
	pm.metrics.TotalPartnerships = len(pm.partnerships)
	pm.metrics.ActivePartnerships = 0
	pm.metrics.ByType = make(map[PartnershipType]int)
	pm.metrics.ByTier = make(map[PartnershipTier]int)
	pm.metrics.TotalRevenue = 0
	pm.metrics.TotalPipeline = 0
	pm.metrics.TotalCustomers = 0
	totalHealthScore := 0.0

	for _, p := range pm.partnerships {
		if p.Status == StatusActive {
			pm.metrics.ActivePartnerships++
		}
		pm.metrics.ByType[p.Type]++
		pm.metrics.ByTier[p.Tier]++
		pm.metrics.TotalRevenue += p.Value.AnnualRevenue
		pm.metrics.TotalPipeline += p.Value.PipelineValue
		pm.metrics.TotalCustomers += p.Value.CustomerCount
		totalHealthScore += p.Performance.HealthScore
	}

	if len(pm.partnerships) > 0 {
		pm.metrics.AvgHealthScore = totalHealthScore / float64(len(pm.partnerships))
	}
}

// CoSellingEngine manages co-selling activities
type CoSellingEngine struct{}

func NewCoSellingEngine() *CoSellingEngine {
	return &CoSellingEngine{}
}

// MarketingEngine manages joint marketing programs
type MarketingEngine struct{}

func NewMarketingEngine() *MarketingEngine {
	return &MarketingEngine{}
}

// PerformanceTracker tracks partnership performance
type PerformanceTracker struct{}

func NewPerformanceTracker() *PerformanceTracker {
	return &PerformanceTracker{}
}

// CalculateHealthScore calculates partnership health score
func (pt *PerformanceTracker) CalculateHealthScore(partner *Partnership) float64 {
	score := 50.0 // Base score

	// Revenue attainment
	if partner.Performance.RevenueAttainment >= 100 {
		score += 20.0
	} else if partner.Performance.RevenueAttainment >= 80 {
		score += 15.0
	} else if partner.Performance.RevenueAttainment >= 60 {
		score += 10.0
	}

	// Win rate
	if partner.Value.WinRate >= 50 {
		score += 10.0
	} else if partner.Value.WinRate >= 30 {
		score += 5.0
	}

	// Customer count
	if partner.Value.CustomerCount >= 50 {
		score += 10.0
	} else if partner.Value.CustomerCount >= 20 {
		score += 5.0
	}

	// Co-sell activity
	if partner.CoSelling.Enabled {
		score += 10.0
	}

	// Cap at 100
	if score > 100 {
		score = 100
	}

	return score
}

// ExportToJSON exports partnership data to JSON
func (pm *PartnershipManager) ExportToJSON() ([]byte, error) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	data := struct {
		Partnerships []*Partnership      `json:"partnerships"`
		Metrics      *PartnershipMetrics `json:"metrics"`
	}{
		Partnerships: make([]*Partnership, 0, len(pm.partnerships)),
		Metrics:      pm.metrics,
	}

	for _, p := range pm.partnerships {
		data.Partnerships = append(data.Partnerships, p)
	}

	return json.MarshalIndent(data, "", "  ")
}
