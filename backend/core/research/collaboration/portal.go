package collaboration

import (
	"fmt"
	"sync"
	"time"
)

// Partner represents an academic partner
type Partner struct {
	ID              string
	Name            string
	Type            string // university, research_institute, company
	Department      string
	Country         string
	ContactEmail    string
	Website         string

	// Collaboration details
	Focus           []string
	JointProjects   []*Project
	Publications    []*Publication
	Students        []*Student

	// Metrics
	PapersPublished int
	PatentsFiled    int
	StartDate       time.Time
	LastActivity    time.Time
}

// Project represents a collaborative research project
type Project struct {
	ID          string
	Title       string
	Description string
	Partners    []string
	Researchers []string
	Students    []string

	StartDate   time.Time
	EndDate     time.Time
	Status      ProjectStatus

	Budget      int64
	Funding     []FundingSource

	Deliverables []Deliverable
	Publications []*Publication
	Patents      []string

	Repository  string
	Website     string
}

// ProjectStatus represents project status
type ProjectStatus string

const (
	StatusProposal  ProjectStatus = "proposal"
	StatusActive    ProjectStatus = "active"
	StatusCompleted ProjectStatus = "completed"
	StatusOnHold    ProjectStatus = "on_hold"
)

// FundingSource represents a funding source
type FundingSource struct {
	Source string
	Amount int64
	Type   string // grant, internal, industry
}

// Deliverable represents a project deliverable
type Deliverable struct {
	Name        string
	Description string
	DueDate     time.Time
	Status      string
	Artifacts   []string
}

// Publication represents a research publication
type Publication struct {
	ID          string
	Title       string
	Authors     []string
	Venue       string
	Year        int
	Citations   int
	DOI         string
	URL         string
	Type        string // conference, journal, workshop, arxiv
}

// Student represents an intern or visiting student
type Student struct {
	ID          string
	Name        string
	Email       string
	University  string
	Program     string // internship, visiting, phd
	Advisor     string
	StartDate   time.Time
	EndDate     time.Time
	Project     string
	Mentor      string
}

// AcademicCollaborationPortal manages academic collaborations
type AcademicCollaborationPortal struct {
	partners     map[string]*Partner
	projects     map[string]*Project
	publications map[string]*Publication
	students     map[string]*Student
	mu           sync.RWMutex
}

// NewAcademicCollaborationPortal creates a new collaboration portal
func NewAcademicCollaborationPortal() *AcademicCollaborationPortal {
	return &AcademicCollaborationPortal{
		partners:     make(map[string]*Partner),
		projects:     make(map[string]*Project),
		publications: make(map[string]*Publication),
		students:     make(map[string]*Student),
	}
}

// AddPartner adds a new academic partner
func (acp *AcademicCollaborationPortal) AddPartner(partner *Partner) error {
	acp.mu.Lock()
	defer acp.mu.Unlock()

	if _, exists := acp.partners[partner.ID]; exists {
		return fmt.Errorf("partner already exists: %s", partner.ID)
	}

	partner.StartDate = time.Now()
	partner.LastActivity = time.Now()
	acp.partners[partner.ID] = partner

	return nil
}

// CreateProject creates a new collaborative project
func (acp *AcademicCollaborationPortal) CreateProject(project *Project) error {
	acp.mu.Lock()
	defer acp.mu.Unlock()

	if _, exists := acp.projects[project.ID]; exists {
		return fmt.Errorf("project already exists: %s", project.ID)
	}

	project.StartDate = time.Now()
	project.Status = StatusActive
	acp.projects[project.ID] = project

	// Update partner projects
	for _, partnerID := range project.Partners {
		if partner, exists := acp.partners[partnerID]; exists {
			partner.JointProjects = append(partner.JointProjects, project)
			partner.LastActivity = time.Now()
		}
	}

	return nil
}

// AddPublication adds a new publication
func (acp *AcademicCollaborationPortal) AddPublication(publication *Publication) error {
	acp.mu.Lock()
	defer acp.mu.Unlock()

	if _, exists := acp.publications[publication.ID]; exists {
		return fmt.Errorf("publication already exists: %s", publication.ID)
	}

	acp.publications[publication.ID] = publication

	// Update partner metrics
	for _, author := range publication.Authors {
		for _, partner := range acp.partners {
			// Check if author is from this partner
			if acp.isPartnerAuthor(author, partner) {
				partner.PapersPublished++
				partner.Publications = append(partner.Publications, publication)
				partner.LastActivity = time.Now()
			}
		}
	}

	return nil
}

// isPartnerAuthor checks if author is from partner institution
func (acp *AcademicCollaborationPortal) isPartnerAuthor(author string, partner *Partner) bool {
	// Simplified check - in production, maintain author-institution mapping
	return true
}

// EnrollStudent enrolls a student
func (acp *AcademicCollaborationPortal) EnrollStudent(student *Student) error {
	acp.mu.Lock()
	defer acp.mu.Unlock()

	if _, exists := acp.students[student.ID]; exists {
		return fmt.Errorf("student already enrolled: %s", student.ID)
	}

	student.StartDate = time.Now()
	acp.students[student.ID] = student

	// Update partner students
	for _, partner := range acp.partners {
		if partner.Name == student.University {
			partner.Students = append(partner.Students, student)
			partner.LastActivity = time.Now()
			break
		}
	}

	return nil
}

// GetPartnerStats returns partner statistics
func (acp *AcademicCollaborationPortal) GetPartnerStats(partnerID string) (*PartnerStats, error) {
	acp.mu.RLock()
	defer acp.mu.RUnlock()

	partner, exists := acp.partners[partnerID]
	if !exists {
		return nil, fmt.Errorf("partner not found: %s", partnerID)
	}

	stats := &PartnerStats{
		PartnerID:         partnerID,
		PartnerName:       partner.Name,
		ActiveProjects:    0,
		CompletedProjects: 0,
		TotalPublications: partner.PapersPublished,
		TotalPatents:      partner.PatentsFiled,
		ActiveStudents:    0,
		TotalStudents:     len(partner.Students),
		CollaborationYears: int(time.Since(partner.StartDate).Hours() / (24 * 365)),
	}

	// Count active projects
	for _, project := range partner.JointProjects {
		if project.Status == StatusActive {
			stats.ActiveProjects++
		} else if project.Status == StatusCompleted {
			stats.CompletedProjects++
		}
	}

	// Count active students
	now := time.Now()
	for _, student := range partner.Students {
		if student.EndDate.After(now) {
			stats.ActiveStudents++
		}
	}

	return stats, nil
}

// PartnerStats contains partner statistics
type PartnerStats struct {
	PartnerID          string
	PartnerName        string
	ActiveProjects     int
	CompletedProjects  int
	TotalPublications  int
	TotalPatents       int
	ActiveStudents     int
	TotalStudents      int
	CollaborationYears int
}

// GetProjectStatus returns project status
func (acp *AcademicCollaborationPortal) GetProjectStatus(projectID string) (*ProjectStatusReport, error) {
	acp.mu.RLock()
	defer acp.mu.RUnlock()

	project, exists := acp.projects[projectID]
	if !exists {
		return nil, fmt.Errorf("project not found: %s", projectID)
	}

	report := &ProjectStatusReport{
		ProjectID:   projectID,
		Title:       project.Title,
		Status:      project.Status,
		Progress:    acp.calculateProjectProgress(project),
		Budget:      project.Budget,
		SpentBudget: acp.calculateSpentBudget(project),
		Deliverables: len(project.Deliverables),
		CompletedDeliverables: acp.countCompletedDeliverables(project),
		Publications: len(project.Publications),
		DaysRemaining: int(time.Until(project.EndDate).Hours() / 24),
	}

	return report, nil
}

// ProjectStatusReport contains project status information
type ProjectStatusReport struct {
	ProjectID             string
	Title                 string
	Status                ProjectStatus
	Progress              float64
	Budget                int64
	SpentBudget           int64
	Deliverables          int
	CompletedDeliverables int
	Publications          int
	DaysRemaining         int
}

// calculateProjectProgress calculates project progress
func (acp *AcademicCollaborationPortal) calculateProjectProgress(project *Project) float64 {
	if len(project.Deliverables) == 0 {
		return 0.0
	}

	completed := acp.countCompletedDeliverables(project)
	return float64(completed) / float64(len(project.Deliverables))
}

// countCompletedDeliverables counts completed deliverables
func (acp *AcademicCollaborationPortal) countCompletedDeliverables(project *Project) int {
	count := 0
	for _, d := range project.Deliverables {
		if d.Status == "completed" {
			count++
		}
	}
	return count
}

// calculateSpentBudget calculates spent budget
func (acp *AcademicCollaborationPortal) calculateSpentBudget(project *Project) int64 {
	// Simplified - in production track actual expenses
	progress := acp.calculateProjectProgress(project)
	return int64(float64(project.Budget) * progress)
}

// GetTopPublications returns top publications by citations
func (acp *AcademicCollaborationPortal) GetTopPublications(limit int) []*Publication {
	acp.mu.RLock()
	defer acp.mu.RUnlock()

	// Convert to slice
	pubs := make([]*Publication, 0, len(acp.publications))
	for _, pub := range acp.publications {
		pubs = append(pubs, pub)
	}

	// Sort by citations (simplified - use proper sort in production)
	// Return top N
	if limit > len(pubs) {
		limit = len(pubs)
	}

	return pubs[:limit]
}

// GetActiveStudents returns currently active students
func (acp *AcademicCollaborationPortal) GetActiveStudents() []*Student {
	acp.mu.RLock()
	defer acp.mu.RUnlock()

	active := make([]*Student, 0)
	now := time.Now()

	for _, student := range acp.students {
		if student.StartDate.Before(now) && student.EndDate.After(now) {
			active = append(active, student)
		}
	}

	return active
}

// GenerateCollaborationReport generates a collaboration report
func (acp *AcademicCollaborationPortal) GenerateCollaborationReport() *CollaborationReport {
	acp.mu.RLock()
	defer acp.mu.RUnlock()

	report := &CollaborationReport{
		GeneratedAt:    time.Now(),
		TotalPartners:  len(acp.partners),
		TotalProjects:  len(acp.projects),
		TotalPublications: len(acp.publications),
		TotalStudents:  len(acp.students),
		PartnersByType: make(map[string]int),
		ProjectsByStatus: make(map[ProjectStatus]int),
	}

	// Count by type
	for _, partner := range acp.partners {
		report.PartnersByType[partner.Type]++
	}

	// Count by status
	for _, project := range acp.projects {
		report.ProjectsByStatus[project.Status]++
	}

	// Calculate total citations
	for _, pub := range acp.publications {
		report.TotalCitations += pub.Citations
	}

	// Count active students
	now := time.Now()
	for _, student := range acp.students {
		if student.StartDate.Before(now) && student.EndDate.After(now) {
			report.ActiveStudents++
		}
	}

	return report
}

// CollaborationReport contains collaboration metrics
type CollaborationReport struct {
	GeneratedAt       time.Time
	TotalPartners     int
	TotalProjects     int
	TotalPublications int
	TotalCitations    int
	TotalStudents     int
	ActiveStudents    int
	PartnersByType    map[string]int
	ProjectsByStatus  map[ProjectStatus]int
}

// OrganizeConference organizes a conference/workshop
func (acp *AcademicCollaborationPortal) OrganizeConference(conf *Conference) error {
	acp.mu.Lock()
	defer acp.mu.Unlock()

	// Conference organization logic
	// - Send invitations to partners
	// - Schedule talks/sessions
	// - Coordinate logistics

	return nil
}

// Conference represents an organized conference
type Conference struct {
	ID          string
	Name        string
	Date        time.Time
	Location    string
	Type        string // conference, workshop, seminar
	Speakers    []string
	Attendees   []string
	Topics      []string
	Proceedings string
}
