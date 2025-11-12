// Package certification implements Advanced Developer Certification System
// 5-tier certification with hands-on labs, real-world assessments, and peer review
// Target: 10,000+ certified developers with 1,000+ labs
package certification

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// CertificationTier represents developer certification levels
type CertificationTier int

const (
	TierDeveloper CertificationTier = iota + 1 // Entry level
	TierArchitect                               // Intermediate
	TierExpert                                  // Advanced
	TierMaster                                  // Expert
	TierGrandMaster                             // Elite
)

func (t CertificationTier) String() string {
	return [...]string{"", "Developer", "Architect", "Expert", "Master", "Grand Master"}[t]
}

// CertificationLevel defines requirements for each tier
type CertificationLevel struct {
	Tier              CertificationTier
	RequiredLabs      int
	RequiredProjects  int
	RequiredCEUs      int // Continuing Education Units per year
	RequiredScore     float64
	RequiredYears     int // Years of experience
	RequiredCerts     []CertificationTier
	CostUSD           float64
	ValidityYears     int
	RecertInterval    int // Months
}

// DeveloperProfile represents certified developer
type DeveloperProfile struct {
	ID                 string
	Email              string
	Name               string
	CurrentTier        CertificationTier
	CertificationID    string
	CertifiedAt        time.Time
	ExpiresAt          time.Time
	RecertDueAt        time.Time
	CompletedLabs      []string
	CompletedProjects  []string
	CEUsEarned         int
	TotalScore         float64
	YearsExperience    int
	Specializations    []string
	Endorsements       []Endorsement
	Achievements       []Achievement
	JobMatches         []JobMatch
	MentorRating       float64
	MenteeCount        int
	PeerReviews        []PeerReview
	ContinuingEd       []ContinuingEducation
	CreatedAt          time.Time
	UpdatedAt          time.Time
}

// Lab represents hands-on learning environment
type Lab struct {
	ID                string
	Title             string
	Description       string
	Category          string
	Tier              CertificationTier
	DifficultyLevel   int // 1-10
	EstimatedHours    int
	Prerequisites     []string
	LearningObjectives []string
	Technologies      []string
	Environment       LabEnvironment
	Tasks             []LabTask
	Validation        LabValidation
	CompletionRate    float64
	AverageScore      float64
	StudentCount      int
	CreatedAt         time.Time
	UpdatedAt         time.Time
}

// LabEnvironment defines isolated lab infrastructure
type LabEnvironment struct {
	Type           string // kubernetes, vm, container, serverless
	Resources      ResourceSpec
	PreConfigured  bool
	AccessMethod   string // web, ssh, api
	TimeLimit      int    // minutes
	AutoDestroy    bool
	SnapshotPoints []string
}

// ResourceSpec defines lab resource requirements
type ResourceSpec struct {
	CPUCores      int
	MemoryGB      int
	StorageGB     int
	GPURequired   bool
	NetworkPolicy string
	Bandwidth     string
}

// LabTask represents individual lab assignment
type LabTask struct {
	ID           string
	Title        string
	Description  string
	Instructions []string
	Hints        []string
	TimeLimit    int // minutes
	Points       int
	Validation   TaskValidation
	Order        int
}

// TaskValidation defines automated validation
type TaskValidation struct {
	Type       string // code, output, api, deployment
	Criteria   []ValidationCriterion
	TestCases  []TestCase
	AutoGrade  bool
	ManualReview bool
}

// ValidationCriterion represents validation rule
type ValidationCriterion struct {
	Name        string
	Type        string // functional, performance, security, style
	Rule        string
	Weight      float64
	Required    bool
	AutoChecked bool
}

// TestCase represents automated test
type TestCase struct {
	ID          string
	Description string
	Input       interface{}
	Expected    interface{}
	Timeout     int
	Points      int
}

// LabValidation defines overall lab completion validation
type LabValidation struct {
	MinScore         float64
	RequiredTasks    []string
	PeerReviewRequired bool
	MentorApproval   bool
	TimeConstraint   int // minutes
	AttemptsAllowed  int
}

// Project represents real-world assessment project
type Project struct {
	ID              string
	Title           string
	Description     string
	Category        string
	Tier            CertificationTier
	Complexity      int // 1-10
	EstimatedWeeks  int
	Requirements    []ProjectRequirement
	Deliverables    []Deliverable
	EvaluationCriteria []EvaluationCriterion
	PeerReviewRequired bool
	MentorshipIncluded bool
	RealWorldContext   string
	CompanySponsored   bool
	SponsorCompany     string
	PrizePool          float64
	SuccessRate        float64
	AverageScore       float64
	CreatedAt          time.Time
	UpdatedAt          time.Time
}

// ProjectRequirement defines project requirements
type ProjectRequirement struct {
	ID          string
	Category    string // functional, technical, quality, documentation
	Description string
	Priority    string // must, should, could
	Validation  string
}

// Deliverable represents project output
type Deliverable struct {
	ID          string
	Type        string // code, documentation, deployment, presentation
	Description string
	Format      string
	DueDate     time.Time
	Weight      float64
	Validation  DeliverableValidation
}

// DeliverableValidation defines deliverable validation
type DeliverableValidation struct {
	AutoChecks    []string
	PeerReview    bool
	MentorReview  bool
	PublicDemo    bool
	MinQuality    float64
}

// EvaluationCriterion defines project evaluation
type EvaluationCriterion struct {
	Name        string
	Category    string // technical, design, quality, innovation
	Description string
	Weight      float64
	Rubric      []RubricLevel
}

// RubricLevel defines scoring rubric
type RubricLevel struct {
	Score       int
	Label       string // excellent, good, satisfactory, needs_improvement
	Description string
	Examples    []string
}

// Endorsement represents peer endorsement
type Endorsement struct {
	ID           string
	FromDeveloper string
	SkillArea    string
	Rating       int // 1-5
	Comment      string
	Verified     bool
	CreatedAt    time.Time
}

// Achievement represents earned achievement
type Achievement struct {
	ID          string
	Type        string // lab, project, contribution, mentorship
	Title       string
	Description string
	BadgeURL    string
	EarnedAt    time.Time
	Points      int
}

// JobMatch represents job marketplace matching
type JobMatch struct {
	ID           string
	JobID        string
	Company      string
	Position     string
	MatchScore   float64
	Requirements []string
	MatchedSkills []string
	SalaryRange  string
	Location     string
	Remote       bool
	AppliedAt    *time.Time
	Status       string
}

// PeerReview represents peer code review
type PeerReview struct {
	ID          string
	ReviewerID  string
	ItemType    string // lab, project, contribution
	ItemID      string
	Rating      float64
	Comments    []ReviewComment
	Helpful     int
	CreatedAt   time.Time
}

// ReviewComment represents review feedback
type ReviewComment struct {
	Category    string // code_quality, design, security, performance
	Severity    string // critical, major, minor, suggestion
	Line        int
	Comment     string
	CodeSnippet string
	Suggestion  string
}

// ContinuingEducation represents ongoing learning
type ContinuingEducation struct {
	ID          string
	Type        string // course, workshop, conference, publication
	Title       string
	Provider    string
	CEUs        int
	CompletedAt time.Time
	Certificate string
}

// CertificationManager manages developer certification
type CertificationManager struct {
	mu                sync.RWMutex
	profiles          map[string]*DeveloperProfile
	labs              map[string]*Lab
	projects          map[string]*Project
	levels            map[CertificationTier]*CertificationLevel
	labCompletions    map[string][]LabCompletion
	projectSubmissions map[string][]ProjectSubmission
	peerReviews       map[string][]PeerReview
	jobMatches        map[string][]JobMatch
	stats             CertificationStats
}

// LabCompletion represents lab completion record
type LabCompletion struct {
	ID          string
	DeveloperID string
	LabID       string
	StartedAt   time.Time
	CompletedAt time.Time
	Score       float64
	TimeSpent   int // minutes
	Attempts    int
	TaskScores  map[string]float64
	PeerReviews []string
	Certificate string
}

// ProjectSubmission represents project submission
type ProjectSubmission struct {
	ID           string
	DeveloperID  string
	ProjectID    string
	SubmittedAt  time.Time
	Deliverables map[string]string // deliverable_id -> url
	Score        float64
	Feedback     []ProjectFeedback
	PeerReviews  []string
	MentorReview *MentorReview
	Status       string // submitted, under_review, approved, rejected
	Certificate  string
}

// ProjectFeedback represents project evaluation feedback
type ProjectFeedback struct {
	CriterionID string
	Score       float64
	Comment     string
	Suggestions []string
	Examples    []string
}

// MentorReview represents mentor evaluation
type MentorReview struct {
	MentorID    string
	Rating      float64
	Strengths   []string
	Improvements []string
	Recommendation string
	Approved    bool
	ReviewedAt  time.Time
}

// CertificationStats tracks certification metrics
type CertificationStats struct {
	TotalCertified      int
	ByTier              map[CertificationTier]int
	TotalLabs           int
	TotalProjects       int
	LabCompletionRate   float64
	ProjectSuccessRate  float64
	AverageScore        float64
	CertifiedThisMonth  int
	RecertThisMonth     int
	ActiveMentors       int
	ActiveMentees       int
	JobPlacements       int
	UpdatedAt           time.Time
}

// NewCertificationManager creates certification manager
func NewCertificationManager() *CertificationManager {
	cm := &CertificationManager{
		profiles:          make(map[string]*DeveloperProfile),
		labs:              make(map[string]*Lab),
		projects:          make(map[string]*Project),
		levels:            make(map[CertificationTier]*CertificationLevel),
		labCompletions:    make(map[string][]LabCompletion),
		projectSubmissions: make(map[string][]ProjectSubmission),
		peerReviews:       make(map[string][]PeerReview),
		jobMatches:        make(map[string][]JobMatch),
	}

	cm.initializeLevels()
	cm.initializeDefaultLabs()
	cm.initializeDefaultProjects()

	return cm
}

// initializeLevels sets up certification tier requirements
func (cm *CertificationManager) initializeLevels() {
	cm.levels[TierDeveloper] = &CertificationLevel{
		Tier:             TierDeveloper,
		RequiredLabs:     10,
		RequiredProjects: 1,
		RequiredCEUs:     20,
		RequiredScore:    70.0,
		RequiredYears:    0,
		RequiredCerts:    []CertificationTier{},
		CostUSD:          299,
		ValidityYears:    2,
		RecertInterval:   24,
	}

	cm.levels[TierArchitect] = &CertificationLevel{
		Tier:             TierArchitect,
		RequiredLabs:     25,
		RequiredProjects: 3,
		RequiredCEUs:     30,
		RequiredScore:    75.0,
		RequiredYears:    2,
		RequiredCerts:    []CertificationTier{TierDeveloper},
		CostUSD:          599,
		ValidityYears:    2,
		RecertInterval:   24,
	}

	cm.levels[TierExpert] = &CertificationLevel{
		Tier:             TierExpert,
		RequiredLabs:     50,
		RequiredProjects: 5,
		RequiredCEUs:     40,
		RequiredScore:    80.0,
		RequiredYears:    4,
		RequiredCerts:    []CertificationTier{TierDeveloper, TierArchitect},
		CostUSD:          999,
		ValidityYears:    3,
		RecertInterval:   36,
	}

	cm.levels[TierMaster] = &CertificationLevel{
		Tier:             TierMaster,
		RequiredLabs:     100,
		RequiredProjects: 10,
		RequiredCEUs:     50,
		RequiredScore:    85.0,
		RequiredYears:    6,
		RequiredCerts:    []CertificationTier{TierDeveloper, TierArchitect, TierExpert},
		CostUSD:          1999,
		ValidityYears:    3,
		RecertInterval:   36,
	}

	cm.levels[TierGrandMaster] = &CertificationLevel{
		Tier:             TierGrandMaster,
		RequiredLabs:     200,
		RequiredProjects: 20,
		RequiredCEUs:     60,
		RequiredScore:    90.0,
		RequiredYears:    10,
		RequiredCerts:    []CertificationTier{TierDeveloper, TierArchitect, TierExpert, TierMaster},
		CostUSD:          4999,
		ValidityYears:    5,
		RecertInterval:   60,
	}
}

// initializeDefaultLabs creates 1,000+ lab templates
func (cm *CertificationManager) initializeDefaultLabs() {
	// Sample labs across different categories and tiers
	categories := []string{
		"distributed_systems", "cloud_native", "security", "performance",
		"data_engineering", "ml_ops", "devops", "microservices",
		"blockchain", "iot", "edge_computing", "serverless",
	}

	for i := 0; i < 1000; i++ {
		tier := CertificationTier((i % 5) + 1)
		category := categories[i%len(categories)]

		lab := &Lab{
			ID:             fmt.Sprintf("LAB-%04d", i+1),
			Title:          fmt.Sprintf("%s Lab %d - %s", tier.String(), i+1, category),
			Description:    fmt.Sprintf("Hands-on lab for %s certification in %s", tier.String(), category),
			Category:       category,
			Tier:           tier,
			DifficultyLevel: (i % 10) + 1,
			EstimatedHours: ((i % 8) + 1) * 2,
			Prerequisites:  []string{},
			LearningObjectives: []string{
				"Understand core concepts",
				"Implement practical solutions",
				"Apply best practices",
				"Troubleshoot common issues",
			},
			Technologies: []string{"Go", "Kubernetes", "Docker", "PostgreSQL"},
			Environment: LabEnvironment{
				Type:         "kubernetes",
				Resources:    ResourceSpec{CPUCores: 4, MemoryGB: 16, StorageGB: 100, GPURequired: false},
				PreConfigured: true,
				AccessMethod: "web",
				TimeLimit:    240,
				AutoDestroy:  true,
			},
			Tasks: cm.generateLabTasks(5),
			Validation: LabValidation{
				MinScore:       70.0,
				RequiredTasks:  []string{"task-1", "task-2"},
				PeerReviewRequired: tier >= TierArchitect,
				MentorApproval: tier >= TierExpert,
				TimeConstraint: 240,
				AttemptsAllowed: 3,
			},
			CompletionRate: 0.75,
			AverageScore:   82.5,
			StudentCount:   0,
			CreatedAt:      time.Now(),
			UpdatedAt:      time.Now(),
		}

		cm.labs[lab.ID] = lab
	}
}

// generateLabTasks creates lab tasks
func (cm *CertificationManager) generateLabTasks(count int) []LabTask {
	tasks := make([]LabTask, count)

	for i := 0; i < count; i++ {
		tasks[i] = LabTask{
			ID:          fmt.Sprintf("task-%d", i+1),
			Title:       fmt.Sprintf("Task %d", i+1),
			Description: "Complete the assigned task",
			Instructions: []string{
				"Read the requirements",
				"Implement the solution",
				"Test your implementation",
				"Submit for validation",
			},
			Hints:     []string{"Check the documentation", "Review examples"},
			TimeLimit: 30,
			Points:    20,
			Validation: TaskValidation{
				Type:       "code",
				AutoGrade:  true,
				ManualReview: false,
				Criteria: []ValidationCriterion{
					{Name: "Correctness", Type: "functional", Rule: "passes_tests", Weight: 0.6, Required: true, AutoChecked: true},
					{Name: "Code Quality", Type: "style", Rule: "follows_standards", Weight: 0.2, Required: false, AutoChecked: true},
					{Name: "Performance", Type: "performance", Rule: "meets_sla", Weight: 0.2, Required: false, AutoChecked: true},
				},
				TestCases: []TestCase{
					{ID: "test-1", Description: "Basic functionality", Points: 10, Timeout: 5},
					{ID: "test-2", Description: "Edge cases", Points: 10, Timeout: 5},
				},
			},
			Order: i + 1,
		}
	}

	return tasks
}

// initializeDefaultProjects creates project templates
func (cm *CertificationManager) initializeDefaultProjects() {
	categories := []string{
		"distributed_system", "microservices", "data_pipeline",
		"security_platform", "ml_platform", "devops_automation",
	}

	for i := 0; i < 100; i++ {
		tier := CertificationTier((i % 5) + 1)
		category := categories[i%len(categories)]

		project := &Project{
			ID:             fmt.Sprintf("PROJ-%04d", i+1),
			Title:          fmt.Sprintf("%s Project %d - %s", tier.String(), i+1, category),
			Description:    fmt.Sprintf("Real-world project for %s certification", tier.String()),
			Category:       category,
			Tier:           tier,
			Complexity:     (i % 10) + 1,
			EstimatedWeeks: ((i % 12) + 1) * 2,
			Requirements: []ProjectRequirement{
				{ID: "req-1", Category: "functional", Description: "Core functionality", Priority: "must"},
				{ID: "req-2", Category: "technical", Description: "Technical implementation", Priority: "must"},
				{ID: "req-3", Category: "quality", Description: "Quality standards", Priority: "should"},
			},
			Deliverables: []Deliverable{
				{ID: "del-1", Type: "code", Description: "Source code", Weight: 0.5},
				{ID: "del-2", Type: "documentation", Description: "Documentation", Weight: 0.2},
				{ID: "del-3", Type: "deployment", Description: "Deployed system", Weight: 0.3},
			},
			EvaluationCriteria: []EvaluationCriterion{
				{Name: "Technical Quality", Category: "technical", Weight: 0.4},
				{Name: "Design", Category: "design", Weight: 0.3},
				{Name: "Innovation", Category: "innovation", Weight: 0.3},
			},
			PeerReviewRequired: true,
			MentorshipIncluded: tier >= TierArchitect,
			RealWorldContext:   "Based on production use case",
			CompanySponsored:   (i % 3) == 0,
			SuccessRate:        0.65,
			AverageScore:       78.5,
			CreatedAt:          time.Now(),
			UpdatedAt:          time.Now(),
		}

		if project.CompanySponsored {
			project.SponsorCompany = "Tech Corp"
			project.PrizePool = 10000.0
		}

		cm.projects[project.ID] = project
	}
}

// RegisterDeveloper creates new developer profile
func (cm *CertificationManager) RegisterDeveloper(ctx context.Context, email, name string) (*DeveloperProfile, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	profile := &DeveloperProfile{
		ID:                 cm.generateID("DEV"),
		Email:              email,
		Name:               name,
		CurrentTier:        0, // Not certified yet
		CompletedLabs:      []string{},
		CompletedProjects:  []string{},
		CEUsEarned:         0,
		TotalScore:         0,
		YearsExperience:    0,
		Specializations:    []string{},
		Endorsements:       []Endorsement{},
		Achievements:       []Achievement{},
		JobMatches:         []JobMatch{},
		MentorRating:       0,
		MenteeCount:        0,
		PeerReviews:        []PeerReview{},
		ContinuingEd:       []ContinuingEducation{},
		CreatedAt:          time.Now(),
		UpdatedAt:          time.Now(),
	}

	cm.profiles[profile.ID] = profile

	return profile, nil
}

// StartLab initiates lab session for developer
func (cm *CertificationManager) StartLab(ctx context.Context, developerID, labID string) (*LabCompletion, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	profile, exists := cm.profiles[developerID]
	if !exists {
		return nil, fmt.Errorf("developer not found: %s", developerID)
	}

	lab, exists := cm.labs[labID]
	if !exists {
		return nil, fmt.Errorf("lab not found: %s", labID)
	}

	// Check prerequisites
	if err := cm.checkLabPrerequisites(profile, lab); err != nil {
		return nil, err
	}

	completion := &LabCompletion{
		ID:          cm.generateID("LC"),
		DeveloperID: developerID,
		LabID:       labID,
		StartedAt:   time.Now(),
		Score:       0,
		TimeSpent:   0,
		Attempts:    1,
		TaskScores:  make(map[string]float64),
	}

	cm.labCompletions[developerID] = append(cm.labCompletions[developerID], *completion)

	return completion, nil
}

// checkLabPrerequisites validates lab prerequisites
func (cm *CertificationManager) checkLabPrerequisites(profile *DeveloperProfile, lab *Lab) error {
	for _, prereqID := range lab.Prerequisites {
		found := false
		for _, completedID := range profile.CompletedLabs {
			if completedID == prereqID {
				found = true
				break
			}
		}
		if !found {
			return fmt.Errorf("prerequisite lab not completed: %s", prereqID)
		}
	}
	return nil
}

// CompleteLab marks lab as completed with score
func (cm *CertificationManager) CompleteLab(ctx context.Context, completionID string, score float64) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	// Find and update completion
	for devID, completions := range cm.labCompletions {
		for i := range completions {
			if completions[i].ID == completionID {
				completions[i].CompletedAt = time.Now()
				completions[i].Score = score
				completions[i].TimeSpent = int(time.Since(completions[i].StartedAt).Minutes())

				// Update developer profile
				if profile, exists := cm.profiles[devID]; exists {
					profile.CompletedLabs = append(profile.CompletedLabs, completions[i].LabID)
					profile.TotalScore = (profile.TotalScore*float64(len(profile.CompletedLabs)-1) + score) / float64(len(profile.CompletedLabs))
					profile.UpdatedAt = time.Now()

					// Award achievement
					achievement := Achievement{
						ID:          cm.generateID("ACH"),
						Type:        "lab",
						Title:       "Lab Completed",
						Description: fmt.Sprintf("Completed lab with score %.2f", score),
						EarnedAt:    time.Now(),
						Points:      int(score),
					}
					profile.Achievements = append(profile.Achievements, achievement)
				}

				// Update lab stats
				if lab, exists := cm.labs[completions[i].LabID]; exists {
					lab.StudentCount++
					lab.AverageScore = (lab.AverageScore*float64(lab.StudentCount-1) + score) / float64(lab.StudentCount)
					lab.UpdatedAt = time.Now()
				}

				return nil
			}
		}
	}

	return fmt.Errorf("completion not found: %s", completionID)
}

// SubmitProject submits project for evaluation
func (cm *CertificationManager) SubmitProject(ctx context.Context, developerID, projectID string, deliverables map[string]string) (*ProjectSubmission, error) {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	profile, exists := cm.profiles[developerID]
	if !exists {
		return nil, fmt.Errorf("developer not found: %s", developerID)
	}

	project, exists := cm.projects[projectID]
	if !exists {
		return nil, fmt.Errorf("project not found: %s", projectID)
	}

	submission := &ProjectSubmission{
		ID:           cm.generateID("PS"),
		DeveloperID:  developerID,
		ProjectID:    projectID,
		SubmittedAt:  time.Now(),
		Deliverables: deliverables,
		Score:        0,
		Feedback:     []ProjectFeedback{},
		PeerReviews:  []string{},
		Status:       "submitted",
	}

	cm.projectSubmissions[developerID] = append(cm.projectSubmissions[developerID], *submission)

	// Award submission achievement
	achievement := Achievement{
		ID:          cm.generateID("ACH"),
		Type:        "project",
		Title:       "Project Submitted",
		Description: fmt.Sprintf("Submitted project: %s", project.Title),
		EarnedAt:    time.Now(),
		Points:      50,
	}
	profile.Achievements = append(profile.Achievements, achievement)
	profile.UpdatedAt = time.Now()

	return submission, nil
}

// ApplyCertification applies for certification tier
func (cm *CertificationManager) ApplyCertification(ctx context.Context, developerID string, tier CertificationTier) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	profile, exists := cm.profiles[developerID]
	if !exists {
		return fmt.Errorf("developer not found: %s", developerID)
	}

	level, exists := cm.levels[tier]
	if !exists {
		return fmt.Errorf("invalid tier: %d", tier)
	}

	// Validate requirements
	if err := cm.validateCertificationRequirements(profile, level); err != nil {
		return fmt.Errorf("certification requirements not met: %w", err)
	}

	// Grant certification
	profile.CurrentTier = tier
	profile.CertificationID = cm.generateCertificationID(developerID, tier)
	profile.CertifiedAt = time.Now()
	profile.ExpiresAt = time.Now().AddDate(level.ValidityYears, 0, 0)
	profile.RecertDueAt = time.Now().AddDate(0, level.RecertInterval, 0)
	profile.UpdatedAt = time.Now()

	// Award certification achievement
	achievement := Achievement{
		ID:          cm.generateID("ACH"),
		Type:        "certification",
		Title:       fmt.Sprintf("%s Certified", tier.String()),
		Description: fmt.Sprintf("Achieved %s certification", tier.String()),
		EarnedAt:    time.Now(),
		Points:      100 * int(tier),
	}
	profile.Achievements = append(profile.Achievements, achievement)

	// Update stats
	cm.stats.TotalCertified++
	if cm.stats.ByTier == nil {
		cm.stats.ByTier = make(map[CertificationTier]int)
	}
	cm.stats.ByTier[tier]++
	cm.stats.CertifiedThisMonth++
	cm.stats.UpdatedAt = time.Now()

	return nil
}

// validateCertificationRequirements checks if developer meets requirements
func (cm *CertificationManager) validateCertificationRequirements(profile *DeveloperProfile, level *CertificationLevel) error {
	// Check completed labs
	if len(profile.CompletedLabs) < level.RequiredLabs {
		return fmt.Errorf("insufficient labs: %d/%d", len(profile.CompletedLabs), level.RequiredLabs)
	}

	// Check completed projects
	if len(profile.CompletedProjects) < level.RequiredProjects {
		return fmt.Errorf("insufficient projects: %d/%d", len(profile.CompletedProjects), level.RequiredProjects)
	}

	// Check CEUs
	if profile.CEUsEarned < level.RequiredCEUs {
		return fmt.Errorf("insufficient CEUs: %d/%d", profile.CEUsEarned, level.RequiredCEUs)
	}

	// Check score
	if profile.TotalScore < level.RequiredScore {
		return fmt.Errorf("insufficient score: %.2f/%.2f", profile.TotalScore, level.RequiredScore)
	}

	// Check experience
	if profile.YearsExperience < level.RequiredYears {
		return fmt.Errorf("insufficient experience: %d/%d years", profile.YearsExperience, level.RequiredYears)
	}

	// Check prerequisite certifications
	for _, reqTier := range level.RequiredCerts {
		if profile.CurrentTier < reqTier {
			return fmt.Errorf("prerequisite certification not met: %s", reqTier.String())
		}
	}

	return nil
}

// AddPeerReview adds peer review
func (cm *CertificationManager) AddPeerReview(ctx context.Context, reviewerID, itemType, itemID string, rating float64, comments []ReviewComment) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	review := PeerReview{
		ID:         cm.generateID("PR"),
		ReviewerID: reviewerID,
		ItemType:   itemType,
		ItemID:     itemID,
		Rating:     rating,
		Comments:   comments,
		Helpful:    0,
		CreatedAt:  time.Now(),
	}

	cm.peerReviews[itemID] = append(cm.peerReviews[itemID], review)

	// Update reviewer's profile
	if profile, exists := cm.profiles[reviewerID]; exists {
		profile.PeerReviews = append(profile.PeerReviews, review)
		profile.UpdatedAt = time.Now()
	}

	return nil
}

// MatchJobs matches developers with job opportunities
func (cm *CertificationManager) MatchJobs(ctx context.Context, developerID string) ([]JobMatch, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	profile, exists := cm.profiles[developerID]
	if !exists {
		return nil, fmt.Errorf("developer not found: %s", developerID)
	}

	// Simulate job matching based on certification tier and skills
	matches := []JobMatch{
		{
			ID:          cm.generateID("JM"),
			JobID:       "JOB-001",
			Company:     "Tech Corp",
			Position:    "Senior Developer",
			MatchScore:  0.85,
			Requirements: []string{"Go", "Kubernetes", "Microservices"},
			MatchedSkills: profile.Specializations,
			SalaryRange: "$120K - $180K",
			Location:    "San Francisco",
			Remote:      true,
			Status:      "active",
		},
	}

	cm.jobMatches[developerID] = matches
	profile.JobMatches = matches

	return matches, nil
}

// RecordContinuingEducation records continuing education
func (cm *CertificationManager) RecordContinuingEducation(ctx context.Context, developerID string, edu ContinuingEducation) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	profile, exists := cm.profiles[developerID]
	if !exists {
		return fmt.Errorf("developer not found: %s", developerID)
	}

	edu.ID = cm.generateID("CE")
	edu.CompletedAt = time.Now()

	profile.ContinuingEd = append(profile.ContinuingEd, edu)
	profile.CEUsEarned += edu.CEUs
	profile.UpdatedAt = time.Now()

	return nil
}

// GetCertificationStats returns certification statistics
func (cm *CertificationManager) GetCertificationStats(ctx context.Context) CertificationStats {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	stats := cm.stats
	stats.TotalLabs = len(cm.labs)
	stats.TotalProjects = len(cm.projects)

	// Calculate completion rates
	totalAttempts := 0
	totalCompleted := 0
	for _, completions := range cm.labCompletions {
		for _, c := range completions {
			totalAttempts++
			if !c.CompletedAt.IsZero() {
				totalCompleted++
			}
		}
	}
	if totalAttempts > 0 {
		stats.LabCompletionRate = float64(totalCompleted) / float64(totalAttempts)
	}

	// Calculate project success rate
	totalSubmissions := 0
	totalApproved := 0
	for _, submissions := range cm.projectSubmissions {
		for _, s := range submissions {
			totalSubmissions++
			if s.Status == "approved" {
				totalApproved++
			}
		}
	}
	if totalSubmissions > 0 {
		stats.ProjectSuccessRate = float64(totalApproved) / float64(totalSubmissions)
	}

	// Calculate average score
	totalScore := 0.0
	count := 0
	for _, profile := range cm.profiles {
		if profile.TotalScore > 0 {
			totalScore += profile.TotalScore
			count++
		}
	}
	if count > 0 {
		stats.AverageScore = totalScore / float64(count)
	}

	stats.UpdatedAt = time.Now()

	return stats
}

// generateID generates unique ID
func (cm *CertificationManager) generateID(prefix string) string {
	timestamp := time.Now().UnixNano()
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s-%d", prefix, timestamp)))
	return fmt.Sprintf("%s-%s", prefix, hex.EncodeToString(hash[:8]))
}

// generateCertificationID generates certification ID
func (cm *CertificationManager) generateCertificationID(developerID string, tier CertificationTier) string {
	data := fmt.Sprintf("%s-%d-%d", developerID, tier, time.Now().Unix())
	hash := sha256.Sum256([]byte(data))
	return fmt.Sprintf("CERT-%s-%s", tier.String(), hex.EncodeToString(hash[:12]))
}

// ExportProfile exports developer profile as JSON
func (cm *CertificationManager) ExportProfile(ctx context.Context, developerID string) ([]byte, error) {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	profile, exists := cm.profiles[developerID]
	if !exists {
		return nil, fmt.Errorf("developer not found: %s", developerID)
	}

	return json.MarshalIndent(profile, "", "  ")
}

// GetLeaderboard returns top certified developers
func (cm *CertificationManager) GetLeaderboard(ctx context.Context, tier CertificationTier, limit int) []*DeveloperProfile {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	var leaders []*DeveloperProfile
	for _, profile := range cm.profiles {
		if tier == 0 || profile.CurrentTier == tier {
			leaders = append(leaders, profile)
		}
	}

	// Sort by total score
	for i := 0; i < len(leaders); i++ {
		for j := i + 1; j < len(leaders); j++ {
			if leaders[j].TotalScore > leaders[i].TotalScore {
				leaders[i], leaders[j] = leaders[j], leaders[i]
			}
		}
	}

	if len(leaders) > limit {
		leaders = leaders[:limit]
	}

	return leaders
}
