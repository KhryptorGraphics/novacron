package patents

import (
	"fmt"
	"sync"
	"time"
)

// Patent represents a patent application or grant
type Patent struct {
	ID              string
	ApplicationID   string
	Title           string
	Abstract        string
	Description     string
	Claims          []Claim
	Inventors       []Inventor
	Assignee        string

	// Status
	Status          PatentStatus
	FilingDate      time.Time
	PublicationDate time.Time
	GrantDate       time.Time
	ExpiryDate      time.Time

	// Classification
	IPCClasses      []string // International Patent Classification
	CPCClasses      []string // Cooperative Patent Classification
	Keywords        []string
	TechnologyArea  string

	// Prior art
	PriorArt        []PriorArt
	Citations       []string
	FamilyMembers   []string

	// Legal
	Jurisdiction    string
	Attorney        string
	Fees            []Fee
	Maintenance     MaintenanceInfo
}

// PatentStatus represents patent status
type PatentStatus string

const (
	StatusDraft         PatentStatus = "draft"
	StatusNoveltyCheck  PatentStatus = "novelty_check"
	StatusReview        PatentStatus = "review"
	StatusFiled         PatentStatus = "filed"
	StatusPublished     PatentStatus = "published"
	StatusGranted       PatentStatus = "granted"
	StatusRejected      PatentStatus = "rejected"
	StatusAbandoned     PatentStatus = "abandoned"
	StatusExpired       PatentStatus = "expired"
)

// Claim represents a patent claim
type Claim struct {
	Number      int
	Type        string // independent, dependent
	Text        string
	Dependencies []int
}

// Inventor represents a patent inventor
type Inventor struct {
	Name    string
	Email   string
	Address string
	Contribution string
}

// PriorArt represents prior art
type PriorArt struct {
	Type        string // patent, publication, product
	Reference   string
	Date        time.Time
	Relevance   float64
	Differences string
}

// Fee represents a patent fee
type Fee struct {
	Type      string
	Amount    int64
	DueDate   time.Time
	PaidDate  time.Time
	Paid      bool
}

// MaintenanceInfo tracks patent maintenance
type MaintenanceInfo struct {
	NextFeeDate    time.Time
	NextFeeAmount  int64
	MaintenancePaid bool
	Alerts         []Alert
}

// Alert represents a patent alert
type Alert struct {
	Type      string
	Message   string
	Timestamp time.Time
	Priority  string
}

// PatentIdea represents a potential patent idea
type PatentIdea struct {
	ID          string
	Title       string
	Description string
	Inventor    string
	SubmittedAt time.Time

	// Evaluation
	NoveltyScore    float64
	CommercialValue float64
	TechnicalMerit  float64
	Patentability   float64

	Status      string
	PatentID    string
}

// PatentManager manages patent portfolio
type PatentManager struct {
	patents     map[string]*Patent
	ideas       map[string]*PatentIdea
	portfolio   Portfolio
	mu          sync.RWMutex
}

// Portfolio tracks patent portfolio metrics
type Portfolio struct {
	TotalPatents      int
	ActivePatents     int
	GrantedPatents    int
	PendingPatents    int
	FamilySize        int
	TechnologyAreas   map[string]int
	AnnualFilings     map[int]int
	TotalValue        int64
}

// NewPatentManager creates a new patent manager
func NewPatentManager() *PatentManager {
	return &PatentManager{
		patents: make(map[string]*Patent),
		ideas:   make(map[string]*PatentIdea),
		portfolio: Portfolio{
			TechnologyAreas: make(map[string]int),
			AnnualFilings:   make(map[int]int),
		},
	}
}

// SubmitIdea submits a patent idea
func (pm *PatentManager) SubmitIdea(idea *PatentIdea) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	if _, exists := pm.ideas[idea.ID]; exists {
		return fmt.Errorf("idea already exists: %s", idea.ID)
	}

	idea.SubmittedAt = time.Now()
	idea.Status = "submitted"
	pm.ideas[idea.ID] = idea

	return nil
}

// EvaluateIdea evaluates a patent idea
func (pm *PatentManager) EvaluateIdea(ideaID string) (*IdeaEvaluation, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	idea, exists := pm.ideas[ideaID]
	if !exists {
		return nil, fmt.Errorf("idea not found: %s", ideaID)
	}

	evaluation := &IdeaEvaluation{
		IdeaID:      ideaID,
		EvaluatedAt: time.Now(),
	}

	// Check novelty
	evaluation.NoveltyScore = pm.checkNovelty(idea)
	idea.NoveltyScore = evaluation.NoveltyScore

	// Assess commercial value
	evaluation.CommercialValue = pm.assessCommercialValue(idea)
	idea.CommercialValue = evaluation.CommercialValue

	// Evaluate technical merit
	evaluation.TechnicalMerit = pm.evaluateTechnicalMerit(idea)
	idea.TechnicalMerit = evaluation.TechnicalMerit

	// Calculate patentability
	evaluation.Patentability = (evaluation.NoveltyScore*0.4 +
		evaluation.CommercialValue*0.3 +
		evaluation.TechnicalMerit*0.3)
	idea.Patentability = evaluation.Patentability

	// Make recommendation
	if evaluation.Patentability >= 0.7 {
		evaluation.Recommendation = "PROCEED_TO_FILING"
		idea.Status = "approved"
	} else if evaluation.Patentability >= 0.5 {
		evaluation.Recommendation = "NEEDS_REFINEMENT"
		idea.Status = "refinement"
	} else {
		evaluation.Recommendation = "DO_NOT_FILE"
		idea.Status = "rejected"
	}

	return evaluation, nil
}

// IdeaEvaluation contains idea evaluation results
type IdeaEvaluation struct {
	IdeaID          string
	EvaluatedAt     time.Time
	NoveltyScore    float64
	CommercialValue float64
	TechnicalMerit  float64
	Patentability   float64
	Recommendation  string
	PriorArtFound   []PriorArt
}

// checkNovelty checks novelty against prior art
func (pm *PatentManager) checkNovelty(idea *PatentIdea) float64 {
	// Simplified novelty check
	// In production, search patent databases (USPTO, EPO, etc.)

	score := 1.0

	// Check against existing patents
	for _, patent := range pm.patents {
		if pm.isSimilar(idea, patent) {
			score -= 0.2
		}
	}

	if score < 0 {
		score = 0
	}

	return score
}

// isSimilar checks if idea is similar to patent
func (pm *PatentManager) isSimilar(idea *PatentIdea, patent *Patent) bool {
	// Simplified similarity check
	// In production, use NLP and semantic analysis
	return false
}

// assessCommercialValue assesses commercial value
func (pm *PatentManager) assessCommercialValue(idea *PatentIdea) float64 {
	// Assess market size, competitive advantage, licensing potential
	return 0.7 // Simplified
}

// evaluateTechnicalMerit evaluates technical merit
func (pm *PatentManager) evaluateTechnicalMerit(idea *PatentIdea) float64 {
	// Evaluate innovation level, implementation feasibility
	return 0.8 // Simplified
}

// CreatePatent creates a patent from an idea
func (pm *PatentManager) CreatePatent(ideaID string, inventors []Inventor) (*Patent, error) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	idea, exists := pm.ideas[ideaID]
	if !exists {
		return nil, fmt.Errorf("idea not found: %s", ideaID)
	}

	if idea.Status != "approved" {
		return nil, fmt.Errorf("idea not approved for filing")
	}

	patent := &Patent{
		ID:          fmt.Sprintf("pat-%d", time.Now().Unix()),
		Title:       idea.Title,
		Abstract:    idea.Description,
		Description: idea.Description,
		Inventors:   inventors,
		Status:      StatusDraft,
		FilingDate:  time.Now(),
		Jurisdiction: "US",
		Claims:      make([]Claim, 0),
		Keywords:    make([]string, 0),
		PriorArt:    make([]PriorArt, 0),
	}

	pm.patents[patent.ID] = patent
	pm.portfolio.TotalPatents++

	idea.Status = "filed"
	idea.PatentID = patent.ID

	return patent, nil
}

// GeneratePatentDraft generates patent draft using AI
func (pm *PatentManager) GeneratePatentDraft(patentID string) (*PatentDraft, error) {
	pm.mu.RLock()
	patent, exists := pm.patents[patentID]
	pm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("patent not found: %s", patentID)
	}

	draft := &PatentDraft{
		PatentID:    patentID,
		GeneratedAt: time.Now(),
	}

	// Generate title
	draft.Title = patent.Title

	// Generate abstract
	draft.Abstract = pm.generateAbstract(patent)

	// Generate background
	draft.Background = pm.generateBackground(patent)

	// Generate summary
	draft.Summary = pm.generateSummary(patent)

	// Generate detailed description
	draft.DetailedDescription = pm.generateDetailedDescription(patent)

	// Generate claims
	draft.Claims = pm.generateClaims(patent)

	// Generate drawings description
	draft.Drawings = pm.generateDrawings(patent)

	return draft, nil
}

// PatentDraft contains generated patent draft
type PatentDraft struct {
	PatentID            string
	GeneratedAt         time.Time
	Title               string
	Abstract            string
	Background          string
	Summary             string
	DetailedDescription string
	Claims              []string
	Drawings            []string
}

// generateAbstract generates patent abstract
func (pm *PatentManager) generateAbstract(patent *Patent) string {
	return fmt.Sprintf("A method and system for %s is disclosed. The invention provides %s with improved %s.",
		patent.Title, patent.Description, "performance and efficiency")
}

// generateBackground generates background section
func (pm *PatentManager) generateBackground(patent *Patent) string {
	return "Background of the invention..."
}

// generateSummary generates summary section
func (pm *PatentManager) generateSummary(patent *Patent) string {
	return "Summary of the invention..."
}

// generateDetailedDescription generates detailed description
func (pm *PatentManager) generateDetailedDescription(patent *Patent) string {
	return "Detailed description of the invention..."
}

// generateClaims generates patent claims
func (pm *PatentManager) generateClaims(patent *Patent) []string {
	claims := []string{
		"1. A system for " + patent.Title + " comprising...",
		"2. The system of claim 1, wherein...",
		"3. A method for " + patent.Title + " comprising the steps of...",
	}
	return claims
}

// generateDrawings generates drawings description
func (pm *PatentManager) generateDrawings(patent *Patent) []string {
	return []string{
		"Figure 1: System architecture",
		"Figure 2: Process flow diagram",
		"Figure 3: Component interaction",
	}
}

// FilePatent files a patent application
func (pm *PatentManager) FilePatent(patentID string) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	patent, exists := pm.patents[patentID]
	if !exists {
		return fmt.Errorf("patent not found: %s", patentID)
	}

	patent.Status = StatusFiled
	patent.FilingDate = time.Now()
	patent.ExpiryDate = patent.FilingDate.AddDate(20, 0, 0) // 20 years

	// Record filing
	year := patent.FilingDate.Year()
	pm.portfolio.AnnualFilings[year]++
	pm.portfolio.PendingPatents++

	// Schedule fees
	pm.scheduleFees(patent)

	return nil
}

// scheduleFees schedules patent fees
func (pm *PatentManager) scheduleFees(patent *Patent) {
	// Filing fee
	patent.Fees = append(patent.Fees, Fee{
		Type:    "filing",
		Amount:  1000,
		DueDate: patent.FilingDate,
		Paid:    true,
		PaidDate: patent.FilingDate,
	})

	// Maintenance fees (3.5, 7.5, 11.5 years)
	maintenanceYears := []int{3, 7, 11}
	amounts := []int64{1600, 3600, 7400}

	for i, years := range maintenanceYears {
		dueDate := patent.FilingDate.AddDate(years, 6, 0)
		patent.Fees = append(patent.Fees, Fee{
			Type:    fmt.Sprintf("maintenance_%d", years),
			Amount:  amounts[i],
			DueDate: dueDate,
			Paid:    false,
		})
	}
}

// UpdatePatentStatus updates patent status
func (pm *PatentManager) UpdatePatentStatus(patentID string, status PatentStatus) error {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	patent, exists := pm.patents[patentID]
	if !exists {
		return fmt.Errorf("patent not found: %s", patentID)
	}

	oldStatus := patent.Status
	patent.Status = status

	// Update portfolio metrics
	if status == StatusGranted {
		patent.GrantDate = time.Now()
		pm.portfolio.GrantedPatents++
		if oldStatus == StatusFiled {
			pm.portfolio.PendingPatents--
		}
		pm.portfolio.ActivePatents++
	} else if status == StatusRejected || status == StatusAbandoned {
		if oldStatus == StatusFiled {
			pm.portfolio.PendingPatents--
		}
	}

	return nil
}

// GetPortfolioReport generates portfolio report
func (pm *PatentManager) GetPortfolioReport() *PortfolioReport {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	report := &PortfolioReport{
		GeneratedAt:      time.Now(),
		TotalPatents:     pm.portfolio.TotalPatents,
		ActivePatents:    pm.portfolio.ActivePatents,
		GrantedPatents:   pm.portfolio.GrantedPatents,
		PendingPatents:   pm.portfolio.PendingPatents,
		TechnologyAreas:  pm.portfolio.TechnologyAreas,
		AnnualFilings:    pm.portfolio.AnnualFilings,
		UpcomingFees:     pm.getUpcomingFees(),
		ExpiringPatents:  pm.getExpiringPatents(365),
	}

	// Calculate portfolio value
	report.EstimatedValue = int64(pm.portfolio.GrantedPatents) * 500000 // $500K per patent

	return report
}

// PortfolioReport contains portfolio information
type PortfolioReport struct {
	GeneratedAt     time.Time
	TotalPatents    int
	ActivePatents   int
	GrantedPatents  int
	PendingPatents  int
	TechnologyAreas map[string]int
	AnnualFilings   map[int]int
	EstimatedValue  int64
	UpcomingFees    []Fee
	ExpiringPatents []*Patent
}

// getUpcomingFees returns upcoming fees
func (pm *PatentManager) getUpcomingFees() []Fee {
	fees := make([]Fee, 0)
	cutoff := time.Now().AddDate(0, 6, 0) // Next 6 months

	for _, patent := range pm.patents {
		for _, fee := range patent.Fees {
			if !fee.Paid && fee.DueDate.Before(cutoff) {
				fees = append(fees, fee)
			}
		}
	}

	return fees
}

// getExpiringPatents returns patents expiring within days
func (pm *PatentManager) getExpiringPatents(days int) []*Patent {
	expiring := make([]*Patent, 0)
	cutoff := time.Now().AddDate(0, 0, days)

	for _, patent := range pm.patents {
		if patent.Status == StatusGranted &&
			patent.ExpiryDate.Before(cutoff) {
			expiring = append(expiring, patent)
		}
	}

	return expiring
}

// SearchPriorArt searches for prior art
func (pm *PatentManager) SearchPriorArt(query string) ([]PriorArt, error) {
	// In production:
	// - Search USPTO database
	// - Search EPO database
	// - Search Google Patents
	// - Search academic publications
	// - Search product databases

	priorArt := make([]PriorArt, 0)

	// Simplified example
	priorArt = append(priorArt, PriorArt{
		Type:      "patent",
		Reference: "US1234567",
		Date:      time.Now().AddDate(-5, 0, 0),
		Relevance: 0.7,
		Differences: "Different implementation approach",
	})

	return priorArt, nil
}
