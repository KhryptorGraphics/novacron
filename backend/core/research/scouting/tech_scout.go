package scouting

import (
	"fmt"
	"sync"
	"time"
)

// Technology represents an emerging technology
type Technology struct {
	ID          string
	Name        string
	Description string
	Category    string
	Maturity    MaturityLevel
	TrendScore  float64

	// Tracking
	FirstSeen   time.Time
	LastUpdate  time.Time
	Sources     []Source
	Keywords    []string

	// Analysis
	Opportunities []Opportunity
	Threats       []Threat
	Competitors   []Competitor
	Adoption      AdoptionMetrics

	// Strategic fit
	Relevance   float64
	Impact      float64
	Urgency     float64
	Priority    int
}

// MaturityLevel represents technology maturity
type MaturityLevel string

const (
	MaturityResearch    MaturityLevel = "research"
	MaturityProofOfConcept MaturityLevel = "proof_of_concept"
	MaturityEarlyAdoption MaturityLevel = "early_adoption"
	MaturityGrowth      MaturityLevel = "growth"
	MaturityMature      MaturityLevel = "mature"
	MaturityDeclining   MaturityLevel = "declining"
)

// Source represents information source
type Source struct {
	Type      string // research_paper, news, patent, startup
	Reference string
	URL       string
	Date      time.Time
	Credibility float64
}

// Opportunity represents a business opportunity
type Opportunity struct {
	Description string
	Impact      string
	Timeframe   time.Duration
	Investment  int64
}

// Threat represents a competitive threat
type Threat struct {
	Description string
	Severity    string
	Timeframe   time.Duration
	Mitigation  string
}

// Competitor represents a competitor
type Competitor struct {
	Name        string
	Type        string // startup, bigtech, research_lab
	Focus       string
	Funding     int64
	Stage       string
	Advantage   string
}

// AdoptionMetrics tracks adoption metrics
type AdoptionMetrics struct {
	Companies   int
	Users       int64
	GrowthRate  float64
	Investment  int64
	Startups    int
}

// Startup represents a startup
type Startup struct {
	ID          string
	Name        string
	Description string
	URL         string
	Founded     time.Time

	// Team
	Founders    []Founder
	TeamSize    int

	// Business
	Industry    string
	Technology  []string
	Stage       StartupStage
	BusinessModel string

	// Funding
	TotalFunding int64
	LastRound    FundingRound
	Investors    []string
	Valuation    int64

	// Traction
	Revenue     int64
	Customers   int
	GrowthRate  float64

	// Strategic fit
	MAOpportunity bool
	Partnership   bool
	Competition   bool
}

// StartupStage represents startup stage
type StartupStage string

const (
	StageSeed      StartupStage = "seed"
	StageSeriesA   StartupStage = "series_a"
	StageSeriesB   StartupStage = "series_b"
	StageSeriesC   StartupStage = "series_c"
	StageGrowth    StartupStage = "growth"
)

// Founder represents a founder
type Founder struct {
	Name       string
	Background string
	LinkedIn   string
}

// FundingRound represents a funding round
type FundingRound struct {
	Round     string
	Amount    int64
	Date      time.Time
	LeadInvestor string
	Investors []string
}

// TechnologyRadar represents technology radar
type TechnologyRadar struct {
	GeneratedAt time.Time
	Rings       map[MaturityLevel][]Technology
	Quadrants   map[string][]Technology
	Moving      []TechnologyMovement
}

// TechnologyMovement tracks technology movement
type TechnologyMovement struct {
	Technology string
	From       MaturityLevel
	To         MaturityLevel
	Date       time.Time
	Reason     string
}

// TechnologyScout scouts emerging technologies
type TechnologyScout struct {
	technologies map[string]*Technology
	startups     map[string]*Startup
	radar        *TechnologyRadar
	mu           sync.RWMutex
}

// NewTechnologyScout creates a new technology scout
func NewTechnologyScout() *TechnologyScout {
	return &TechnologyScout{
		technologies: make(map[string]*Technology),
		startups:     make(map[string]*Startup),
		radar: &TechnologyRadar{
			Rings:     make(map[MaturityLevel][]Technology),
			Quadrants: make(map[string][]Technology),
			Moving:    make([]TechnologyMovement, 0),
		},
	}
}

// TrackTechnology tracks an emerging technology
func (ts *TechnologyScout) TrackTechnology(tech *Technology) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if _, exists := ts.technologies[tech.ID]; exists {
		return fmt.Errorf("technology already tracked: %s", tech.ID)
	}

	tech.FirstSeen = time.Now()
	tech.LastUpdate = time.Now()
	ts.technologies[tech.ID] = tech

	// Add to radar
	ts.radar.Rings[tech.Maturity] = append(ts.radar.Rings[tech.Maturity], *tech)
	ts.radar.Quadrants[tech.Category] = append(ts.radar.Quadrants[tech.Category], *tech)

	return nil
}

// UpdateTechnology updates technology information
func (ts *TechnologyScout) UpdateTechnology(techID string, updates map[string]interface{}) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	tech, exists := ts.technologies[techID]
	if !exists {
		return fmt.Errorf("technology not found: %s", techID)
	}

	oldMaturity := tech.Maturity

	// Apply updates
	if maturity, ok := updates["maturity"].(MaturityLevel); ok {
		tech.Maturity = maturity

		// Record movement
		if maturity != oldMaturity {
			ts.radar.Moving = append(ts.radar.Moving, TechnologyMovement{
				Technology: tech.Name,
				From:       oldMaturity,
				To:         maturity,
				Date:       time.Now(),
				Reason:     "Maturity progression",
			})
		}
	}

	if score, ok := updates["trend_score"].(float64); ok {
		tech.TrendScore = score
	}

	tech.LastUpdate = time.Now()

	return nil
}

// TrackStartup tracks a startup
func (ts *TechnologyScout) TrackStartup(startup *Startup) error {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	if _, exists := ts.startups[startup.ID]; exists {
		return fmt.Errorf("startup already tracked: %s", startup.ID)
	}

	ts.startups[startup.ID] = startup

	// Analyze strategic fit
	ts.analyzeStartupFit(startup)

	return nil
}

// analyzeStartupFit analyzes startup strategic fit
func (ts *TechnologyScout) analyzeStartupFit(startup *Startup) {
	// Check M&A opportunity
	if startup.Stage == StageSeed || startup.Stage == StageSeriesA {
		if startup.TotalFunding < 10000000 { // < $10M
			startup.MAOpportunity = true
		}
	}

	// Check partnership opportunity
	for _, tech := range startup.Technology {
		if ts.isRelevantTechnology(tech) {
			startup.Partnership = true
			break
		}
	}

	// Check competition
	if startup.Industry == "cloud computing" || startup.Industry == "distributed systems" {
		startup.Competition = true
	}
}

// isRelevantTechnology checks if technology is relevant
func (ts *TechnologyScout) isRelevantTechnology(tech string) bool {
	relevant := []string{
		"distributed systems", "cloud computing", "edge computing",
		"quantum computing", "machine learning", "blockchain",
	}

	for _, r := range relevant {
		if tech == r {
			return true
		}
	}

	return false
}

// ScanYCombinator scans Y Combinator for startups
func (ts *TechnologyScout) ScanYCombinator() ([]Startup, error) {
	// In production, scrape YC website or use API
	startups := make([]Startup, 0)

	// Example startups
	startups = append(startups, Startup{
		ID:          "yc-1",
		Name:        "CloudScale",
		Description: "Distributed computing platform",
		Technology:  []string{"distributed systems", "cloud computing"},
		Stage:       StageSeed,
		TotalFunding: 2000000,
	})

	return startups, nil
}

// ScanTechCrunch scans TechCrunch for news
func (ts *TechnologyScout) ScanTechCrunch() ([]Technology, error) {
	// In production, use TechCrunch API
	technologies := make([]Technology, 0)

	// Example technologies
	technologies = append(technologies, Technology{
		ID:          "tc-1",
		Name:        "Quantum Networking",
		Description: "Next-gen quantum communication",
		Category:    "networking",
		Maturity:    MaturityResearch,
		TrendScore:  0.8,
	})

	return technologies, nil
}

// IdentifyMATargets identifies M&A targets
func (ts *TechnologyScout) IdentifyMATargets() []*Startup {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	targets := make([]*Startup, 0)

	for _, startup := range ts.startups {
		if startup.MAOpportunity {
			targets = append(targets, startup)
		}
	}

	return targets
}

// IdentifyPartners identifies partnership opportunities
func (ts *TechnologyScout) IdentifyPartners() []*Startup {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	partners := make([]*Startup, 0)

	for _, startup := range ts.startups {
		if startup.Partnership {
			partners = append(partners, startup)
		}
	}

	return partners
}

// GetCompetitors identifies competitive threats
func (ts *TechnologyScout) GetCompetitors() []*Startup {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	competitors := make([]*Startup, 0)

	for _, startup := range ts.startups {
		if startup.Competition {
			competitors = append(competitors, startup)
		}
	}

	return competitors
}

// GenerateRadar generates technology radar
func (ts *TechnologyScout) GenerateRadar() *TechnologyRadar {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	ts.radar.GeneratedAt = time.Now()

	// Clear existing
	ts.radar.Rings = make(map[MaturityLevel][]Technology)
	ts.radar.Quadrants = make(map[string][]Technology)

	// Populate radar
	for _, tech := range ts.technologies {
		ts.radar.Rings[tech.Maturity] = append(ts.radar.Rings[tech.Maturity], *tech)
		ts.radar.Quadrants[tech.Category] = append(ts.radar.Quadrants[tech.Category], *tech)
	}

	return ts.radar
}

// GetTrendingTechnologies returns trending technologies
func (ts *TechnologyScout) GetTrendingTechnologies(limit int) []*Technology {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	technologies := make([]*Technology, 0)
	for _, tech := range ts.technologies {
		technologies = append(technologies, tech)
	}

	// Sort by trend score (simplified)
	if limit > len(technologies) {
		limit = len(technologies)
	}

	return technologies[:limit]
}

// GetEmergingTechnologies returns emerging technologies
func (ts *TechnologyScout) GetEmergingTechnologies() []*Technology {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	emerging := make([]*Technology, 0)

	for _, tech := range ts.technologies {
		if tech.Maturity == MaturityResearch || tech.Maturity == MaturityProofOfConcept {
			if tech.TrendScore > 0.7 {
				emerging = append(emerging, tech)
			}
		}
	}

	return emerging
}

// GenerateScoutingReport generates scouting report
func (ts *TechnologyScout) GenerateScoutingReport() *ScoutingReport {
	ts.mu.RLock()
	defer ts.mu.RUnlock()

	report := &ScoutingReport{
		GeneratedAt:        time.Now(),
		TechnologiesTracked: len(ts.technologies),
		StartupsTracked:    len(ts.startups),
		MATargets:          len(ts.IdentifyMATargets()),
		PartnershipOps:     len(ts.IdentifyPartners()),
		CompetitiveThreats: len(ts.GetCompetitors()),
		TrendingTech:       ts.GetTrendingTechnologies(5),
		EmergingTech:       ts.GetEmergingTechnologies(),
		Radar:              ts.GenerateRadar(),
	}

	return report
}

// ScoutingReport contains scouting findings
type ScoutingReport struct {
	GeneratedAt        time.Time
	TechnologiesTracked int
	StartupsTracked    int
	MATargets          int
	PartnershipOps     int
	CompetitiveThreats int
	TrendingTech       []*Technology
	EmergingTech       []*Technology
	Radar              *TechnologyRadar
}
