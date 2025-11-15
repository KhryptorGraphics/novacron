// Package university implements University Partnership Program
// Curriculum integration, student developer program, research grants, internships
// Target: 100+ university partnerships, 500+ interns/year
package university

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// UniversityTier represents partnership level
type UniversityTier string

const (
	TierPlatinum UniversityTier = "platinum"
	TierGold     UniversityTier = "gold"
	TierSilver   UniversityTier = "silver"
	TierBronze   UniversityTier = "bronze"
)

// University represents academic institution
type University struct {
	ID              string
	Name            string
	Country         string
	Region          string
	Type            string // research, teaching, community_college
	Ranking         int
	Tier            UniversityTier
	PartnerSince    time.Time
	Departments     []Department
	Faculty         []Faculty
	Students        []Student
	Courses         []Course
	ResearchProjects []ResearchProject
	Internships     []Internship
	Benefits        PartnershipBenefits
	Metrics         UniversityMetrics
	Contacts        []Contact
	AgreementURL    string
	Status          string // active, pending, expired
	RenewalDate     time.Time
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// Department represents academic department
type Department struct {
	ID              string
	Name            string
	Code            string
	Faculty         []string
	Courses         []string
	Students        int
	ResearchAreas   []string
	Labs            []Lab
	Collaborations  []string
}

// Lab represents research lab
type Lab struct {
	ID              string
	Name            string
	Director        string
	FocusArea       string
	Equipment       []string
	Projects        []string
	Funding         float64
	Publications    int
}

// Faculty represents university professor
type Faculty struct {
	ID              string
	Name            string
	Title           string
	Department      string
	Email           string
	Research        []string
	Courses         []string
	Publications    []Publication
	Students        []string
	CertifiedInstructor bool
	TrainingDate    *time.Time
	Photo           string
	Bio             string
	LinkedIn        string
	GoogleScholar   string
}

// Publication represents research publication
type Publication struct {
	Title       string
	Authors     []string
	Venue       string
	Year        int
	Citations   int
	DOI         string
	URL         string
}

// Student represents university student
type Student struct {
	ID              string
	Name            string
	Email           string
	Major           string
	Year            int // 1-4
	GPA             float64
	Skills          []string
	Interests       []string
	CoursesCompleted []string
	ProjectsCompleted []string
	CertificationsEarned []string
	InternshipStatus string
	Credits         float64 // free platform credits
	Status          string // active, graduated, withdrawn
	JoinedAt        time.Time
}

// Course represents university course
type Course struct {
	ID              string
	Code            string
	Name            string
	Department      string
	Instructor      string
	Semester        string
	Credits         int
	Enrollment      int
	Description     string
	Syllabus        CourseSyllabus
	Materials       []CourseMaterial
	Assignments     []Assignment
	Projects        []CourseProject
	Labs            []CourseLab
	PlatformIntegration bool
	CompletionRate  float64
	AverageGrade    float64
	StudentFeedback float64
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// CourseSyllabus defines course structure
type CourseSyllabus struct {
	Overview        string
	LearningObjectives []string
	Prerequisites   []string
	Topics          []WeeklyTopic
	GradingPolicy   GradingPolicy
	TextBooks       []TextBook
	SoftwareRequired []string
}

// WeeklyTopic represents course topic
type WeeklyTopic struct {
	Week        int
	Topic       string
	Readings    []string
	Labs        []string
	Assignments []string
	Notes       string
}

// GradingPolicy defines grading structure
type GradingPolicy struct {
	Assignments     float64 // percentage
	Projects        float64
	Midterm         float64
	Final           float64
	Participation   float64
	LetterGrades    map[string]float64
}

// TextBook represents course textbook
type TextBook struct {
	Title       string
	Authors     []string
	ISBN        string
	Edition     int
	Publisher   string
	Required    bool
}

// CourseMaterial represents learning material
type CourseMaterial struct {
	ID          string
	Type        string // lecture, video, slides, reading
	Title       string
	Description string
	URL         string
	Duration    int // minutes
	Week        int
	Required    bool
}

// Assignment represents course assignment
type Assignment struct {
	ID              string
	Title           string
	Description     string
	Type            string // homework, quiz, exam
	DueDate         time.Time
	Points          int
	Submissions     int
	AverageScore    float64
	AutoGraded      bool
	PlatformBased   bool
}

// CourseProject represents course project
type CourseProject struct {
	ID              string
	Title           string
	Description     string
	Requirements    []string
	Deliverables    []string
	TeamSize        int
	DueDate         time.Time
	Points          int
	RubricURL       string
	ExampleProjects []string
}

// CourseLab represents hands-on lab
type CourseLab struct {
	ID              string
	Title           string
	Description     string
	Duration        int // minutes
	Difficulty      string
	Prerequisites   []string
	PlatformLabID   string
	CompletionRate  float64
	AverageScore    float64
}

// ResearchProject represents research initiative
type ResearchProject struct {
	ID              string
	Title           string
	Description     string
	PrincipalInvestigator string
	CoInvestigators []string
	Students        []string
	FocusArea       string
	Methodology     string
	StartDate       time.Time
	EndDate         time.Time
	Status          string // proposal, active, completed, published
	Funding         ResearchFunding
	Publications    []Publication
	Datasets        []Dataset
	CodeRepository  string
	Results         ResearchResults
	Impact          ResearchImpact
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// ResearchFunding tracks research funding
type ResearchFunding struct {
	TotalAmount     float64
	Source          string // internal, grant, industry
	GrantNumber     string
	Duration        int // months
	Disbursed       float64
	Remaining       float64
}

// Dataset represents research dataset
type Dataset struct {
	Name            string
	Description     string
	Size            string
	Format          string
	URL             string
	License         string
	Citations       int
}

// ResearchResults contains research outcomes
type ResearchResults struct {
	Summary         string
	KeyFindings     []string
	Metrics         map[string]float64
	Visualizations  []string
	SourceCode      string
	Reproducible    bool
}

// ResearchImpact measures research impact
type ResearchImpact struct {
	Citations       int
	Downloads       int
	MediaCoverage   []MediaMention
	IndustryAdoption []string
	Patents         []Patent
	Awards          []ResearchAward
}

// MediaMention represents media coverage
type MediaMention struct {
	Outlet      string
	Title       string
	URL         string
	Date        time.Time
	Reach       int
}

// Patent represents filed patent
type Patent struct {
	Number      string
	Title       string
	Inventors   []string
	FiledDate   time.Time
	Status      string
	Country     string
}

// ResearchAward represents research recognition
type ResearchAward struct {
	Name        string
	Organization string
	Year        int
	Value       float64
}

// Internship represents internship opportunity
type Internship struct {
	ID              string
	Title           string
	Description     string
	Department      string
	Type            string // full-time, part-time, remote
	Duration        int // weeks
	Location        string
	Remote          bool
	Requirements    []string
	Responsibilities []string
	Compensation    InternshipCompensation
	Mentor          string
	MaxInterns      int
	Applicants      []InternApplication
	Selected        []string
	Status          string // open, filled, completed
	StartDate       time.Time
	EndDate         time.Time
	CreatedAt       time.Time
}

// InternshipCompensation defines intern compensation
type InternshipCompensation struct {
	Stipend         float64
	Benefits        []string
	PlatformCredits float64
	Certificate     bool
	FutureJobOffer  bool
}

// InternApplication represents internship application
type InternApplication struct {
	ID              string
	StudentID       string
	InternshipID    string
	Resume          string
	CoverLetter     string
	Transcript      string
	Portfolio       string
	References      []Reference
	Status          string // submitted, reviewing, accepted, rejected
	AppliedAt       time.Time
	ReviewedAt      *time.Time
	Decision        string
	Feedback        string
}

// Reference represents job reference
type Reference struct {
	Name        string
	Title       string
	Organization string
	Email       string
	Phone       string
	Relationship string
}

// PartnershipBenefits defines university benefits
type PartnershipBenefits struct {
	FreeStudentCredits  float64 // per student
	FacultyTraining     bool
	CurriculumSupport   bool
	GuestLectures       int // per year
	ResearchGrants      float64 // per year
	InternshipSlots     int
	HackathonSponsorship float64
	LabEquipment        []string
	CloudResources      bool
	TechnicalSupport    bool
	CertificationDiscount float64
	JobPlacementSupport bool
}

// UniversityMetrics tracks partnership metrics
type UniversityMetrics struct {
	TotalStudents       int
	ActiveStudents      int
	CompletedCourses    int
	CertifiedStudents   int
	InternsPlaced       int
	ResearchProjects    int
	Publications        int
	CreditsUsed         float64
	CreditsRemaining    float64
	CourseIntegrations  int
	FacultyCertified    int
	StudentSatisfaction float64
	FacultySatisfaction float64
	PlacementRate       float64
	AverageSalary       float64
	UpdatedAt           time.Time
}

// Contact represents university contact
type Contact struct {
	Name        string
	Title       string
	Department  string
	Email       string
	Phone       string
	Role        string // primary, technical, billing
}

// AcademicProgramManager manages university partnerships
type AcademicProgramManager struct {
	mu              sync.RWMutex
	universities    map[string]*University
	students        map[string]*Student
	faculty         map[string]*Faculty
	courses         map[string]*Course
	research        map[string]*ResearchProject
	internships     map[string]*Internship
	stats           AcademicStats
}

// AcademicStats tracks academic program metrics
type AcademicStats struct {
	TotalUniversities   int
	ActivePartnerships  int
	TotalStudents       int
	ActiveStudents      int
	TotalFaculty        int
	CertifiedFaculty    int
	TotalCourses        int
	ActiveCourses       int
	TotalInterns        int
	InternsPlaсed       int
	ResearchProjects    int
	TotalPublications   int
	ResearchFunding     float64
	TotalCreditsIssued  float64
	TotalCreditsUsed    float64
	JobPlacementRate    float64
	AverageStartingSalary float64
	StudentSatisfaction float64
	FacultySatisfaction float64
	UpdatedAt           time.Time
	InternsPlaced int
}

// NewAcademicProgramManager creates academic program manager
func NewAcademicProgramManager() *AcademicProgramManager {
	apm := &AcademicProgramManager{
		universities: make(map[string]*University),
		students:     make(map[string]*Student),
		faculty:      make(map[string]*Faculty),
		courses:      make(map[string]*Course),
		research:     make(map[string]*ResearchProject),
		internships:  make(map[string]*Internship),
	}

	apm.initializeSampleUniversities()

	return apm
}

// initializeSampleUniversities creates sample universities
func (apm *AcademicProgramManager) initializeSampleUniversities() {
	countries := []string{"USA", "UK", "Canada", "Germany", "India", "China", "Australia"}
	tiers := []UniversityTier{TierPlatinum, TierGold, TierSilver, TierBronze}

	for i := 0; i < 100; i++ {
		tier := tiers[i%len(tiers)]
		country := countries[i%len(countries)]

		university := &University{
			ID:           apm.generateID("UNI"),
			Name:         fmt.Sprintf("University %d", i+1),
			Country:      country,
			Region:       "Region " + country,
			Type:         "research",
			Ranking:      i + 1,
			Tier:         tier,
			PartnerSince: time.Now().AddDate(-2, 0, 0),
			Benefits:     apm.getBenefitsByTier(tier),
			Status:       "active",
			RenewalDate:  time.Now().AddDate(1, 0, 0),
			CreatedAt:    time.Now().AddDate(-2, 0, 0),
			UpdatedAt:    time.Now(),
		}

		// Add departments
		university.Departments = []Department{
			{
				ID:            apm.generateID("DEPT"),
				Name:          "Computer Science",
				Code:          "CS",
				Students:      500,
				ResearchAreas: []string{"AI/ML", "Systems", "Security"},
			},
			{
				ID:            apm.generateID("DEPT"),
				Name:          "Engineering",
				Code:          "ENG",
				Students:      800,
				ResearchAreas: []string{"Cloud", "IoT", "Robotics"},
			},
		}

		// Add metrics
		university.Metrics = UniversityMetrics{
			TotalStudents:       1000 + i*10,
			ActiveStudents:      800 + i*8,
			CompletedCourses:    50 + i,
			CertifiedStudents:   100 + i*2,
			InternsPlaced:       20 + i,
			ResearchProjects:    10 + i/10,
			Publications:        5 + i/20,
			CreditsUsed:         float64(100000 + i*1000),
			CreditsRemaining:    float64(50000 + i*500),
			CourseIntegrations:  5 + i/10,
			FacultyCertified:    10 + i/10,
			StudentSatisfaction: 4.5,
			FacultySatisfaction: 4.6,
			PlacementRate:       0.85,
			AverageSalary:       75000.0 + float64(i*1000),
			UpdatedAt:           time.Now(),
		}

		apm.universities[university.ID] = university
	}
}

// getBenefitsByTier returns benefits for tier
func (apm *AcademicProgramManager) getBenefitsByTier(tier UniversityTier) PartnershipBenefits {
	benefits := PartnershipBenefits{
		CloudResources:    true,
		TechnicalSupport:  true,
		JobPlacementSupport: true,
	}

	switch tier {
	case TierPlatinum:
		benefits.FreeStudentCredits = 1000
		benefits.FacultyTraining = true
		benefits.CurriculumSupport = true
		benefits.GuestLectures = 12
		benefits.ResearchGrants = 100000
		benefits.InternshipSlots = 50
		benefits.HackathonSponsorship = 50000
		benefits.LabEquipment = []string{"servers", "workstations", "licenses"}
		benefits.CertificationDiscount = 0.50
	case TierGold:
		benefits.FreeStudentCredits = 500
		benefits.FacultyTraining = true
		benefits.CurriculumSupport = true
		benefits.GuestLectures = 6
		benefits.ResearchGrants = 50000
		benefits.InternshipSlots = 30
		benefits.HackathonSponsorship = 25000
		benefits.CertificationDiscount = 0.30
	case TierSilver:
		benefits.FreeStudentCredits = 250
		benefits.FacultyTraining = true
		benefits.GuestLectures = 4
		benefits.ResearchGrants = 25000
		benefits.InternshipSlots = 15
		benefits.CertificationDiscount = 0.20
	case TierBronze:
		benefits.FreeStudentCredits = 100
		benefits.GuestLectures = 2
		benefits.ResearchGrants = 10000
		benefits.InternshipSlots = 5
		benefits.CertificationDiscount = 0.10
	}

	return benefits
}

// AddUniversity adds university partnership
func (apm *AcademicProgramManager) AddUniversity(ctx context.Context, university *University) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	if university.ID == "" {
		university.ID = apm.generateID("UNI")
	}

	university.Status = "active"
	university.PartnerSince = time.Now()
	university.RenewalDate = time.Now().AddDate(1, 0, 0)
	university.CreatedAt = time.Now()
	university.UpdatedAt = time.Now()

	apm.universities[university.ID] = university

	apm.stats.TotalUniversities++
	apm.stats.ActivePartnerships++
	apm.stats.UpdatedAt = time.Now()

	return nil
}

// RegisterStudent registers student
func (apm *AcademicProgramManager) RegisterStudent(ctx context.Context, student *Student) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	if student.ID == "" {
		student.ID = apm.generateID("STU")
	}

	student.Status = "active"
	student.JoinedAt = time.Now()

	// Assign free credits based on university tier
	// (In production, lookup university tier)
	student.Credits = 500.0

	apm.students[student.ID] = student

	apm.stats.TotalStudents++
	apm.stats.ActiveStudents++
	apm.stats.TotalCreditsIssued += student.Credits
	apm.stats.UpdatedAt = time.Now()

	return nil
}

// CertifyFaculty certifies faculty member
func (apm *AcademicProgramManager) CertifyFaculty(ctx context.Context, facultyID string) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	faculty, exists := apm.faculty[facultyID]
	if !exists {
		return fmt.Errorf("faculty not found: %s", facultyID)
	}

	now := time.Now()
	faculty.CertifiedInstructor = true
	faculty.TrainingDate = &now

	apm.stats.CertifiedFaculty++
	apm.stats.UpdatedAt = time.Now()

	return nil
}

// CreateCourse creates university course
func (apm *AcademicProgramManager) CreateCourse(ctx context.Context, course *Course) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	if course.ID == "" {
		course.ID = apm.generateID("CRS")
	}

	course.CreatedAt = time.Now()
	course.UpdatedAt = time.Now()

	apm.courses[course.ID] = course

	apm.stats.TotalCourses++
	apm.stats.ActiveCourses++
	apm.stats.UpdatedAt = time.Now()

	return nil
}

// SubmitResearch submits research project
func (apm *AcademicProgramManager) SubmitResearch(ctx context.Context, project *ResearchProject) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	if project.ID == "" {
		project.ID = apm.generateID("RES")
	}

	project.Status = "proposal"
	project.CreatedAt = time.Now()
	project.UpdatedAt = time.Now()

	apm.research[project.ID] = project

	apm.stats.ResearchProjects++
	apm.stats.UpdatedAt = time.Now()

	return nil
}

// AwardResearchGrant awards research grant
func (apm *AcademicProgramManager) AwardResearchGrant(ctx context.Context, projectID string, amount float64) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	project, exists := apm.research[projectID]
	if !exists {
		return fmt.Errorf("research project not found: %s", projectID)
	}

	project.Funding = ResearchFunding{
		TotalAmount: amount,
		Source:      "industry",
		Duration:    12,
		Disbursed:   0,
		Remaining:   amount,
	}
	project.Status = "active"
	project.UpdatedAt = time.Now()

	apm.stats.ResearchFunding += amount
	apm.stats.UpdatedAt = time.Now()

	return nil
}

// PostInternship posts internship opportunity
func (apm *AcademicProgramManager) PostInternship(ctx context.Context, internship *Internship) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	if internship.ID == "" {
		internship.ID = apm.generateID("INT")
	}

	internship.Status = "open"
	internship.CreatedAt = time.Now()

	apm.internships[internship.ID] = internship

	apm.stats.TotalInterns += internship.MaxInterns
	apm.stats.UpdatedAt = time.Now()

	return nil
}

// ApplyInternship applies for internship
func (apm *AcademicProgramManager) ApplyInternship(ctx context.Context, application *InternApplication) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	internship, exists := apm.internships[application.InternshipID]
	if !exists {
		return fmt.Errorf("internship not found: %s", application.InternshipID)
	}

	if application.ID == "" {
		application.ID = apm.generateID("APP")
	}

	application.Status = "submitted"
	application.AppliedAt = time.Now()

	internship.Applicants = append(internship.Applicants, *application)

	return nil
}

// SelectIntern selects intern
func (apm *AcademicProgramManager) SelectIntern(ctx context.Context, internshipID, studentID string) error {
	apm.mu.Lock()
	defer apm.mu.Unlock()

	internship, exists := apm.internships[internshipID]
	if !exists {
		return fmt.Errorf("internship not found: %s", internshipID)
	}

	internship.Selected = append(internship.Selected, studentID)

	if len(internship.Selected) >= internship.MaxInterns {
		internship.Status = "filled"
	}

	apm.stats.InternsPlaced++
	apm.stats.UpdatedAt = time.Now()

	return nil
}

// GetAcademicStats returns academic program statistics
func (apm *AcademicProgramManager) GetAcademicStats(ctx context.Context) AcademicStats {
	apm.mu.RLock()
	defer apm.mu.RUnlock()

	stats := apm.stats

	// Calculate derived metrics
	if stats.TotalStudents > 0 {
		stats.JobPlacementRate = float64(stats.InternsPlaced) / float64(stats.TotalStudents)
	}

	// Count publications
	totalPubs := 0
	for _, project := range apm.research {
		totalPubs += len(project.Publications)
	}
	stats.TotalPublications = totalPubs

	stats.UpdatedAt = time.Now()

	return stats
}

// generateID generates unique ID
func (apm *AcademicProgramManager) generateID(prefix string) string {
	timestamp := time.Now().UnixNano()
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s-%d", prefix, timestamp)))
	return fmt.Sprintf("%s-%s", prefix, hex.EncodeToString(hash[:8]))
}

// ExportUniversityData exports university data as JSON
func (apm *AcademicProgramManager) ExportUniversityData(ctx context.Context, universityID string) ([]byte, error) {
	apm.mu.RLock()
	defer apm.mu.RUnlock()

	university, exists := apm.universities[universityID]
	if !exists {
		return nil, fmt.Errorf("university not found: %s", universityID)
	}

	return json.MarshalIndent(university, "", "  ")
}
