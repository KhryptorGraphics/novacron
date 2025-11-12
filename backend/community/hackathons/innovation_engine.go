// Package hackathons implements Hackathon & Innovation Programs
// Monthly hackathons ($100K+ prize pools), startup accelerator, venture funding
// Target: 100+ hackathons/year, 50+ startups funded
package hackathons

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// HackathonType represents hackathon format
type HackathonType string

const (
	TypeCompetition  HackathonType = "competition"
	TypeChallenge    HackathonType = "challenge"
	TypeIdeathon     HackathonType = "ideathon"
	TypeBuildathon   HackathonType = "buildathon"
	TypeStartupWeek  HackathonType = "startup_week"
)

// HackathonStatus represents hackathon lifecycle
type HackathonStatus string

const (
	StatusPlanning      HackathonStatus = "planning"
	StatusRegistration  HackathonStatus = "registration"
	StatusActive        HackathonStatus = "active"
	StatusJudging       HackathonStatus = "judging"
	StatusCompleted     HackathonStatus = "completed"
	StatusCancelled     HackathonStatus = "cancelled"
)

// Hackathon represents innovation hackathon event
type Hackathon struct {
	ID                  string
	Name                string
	Description         string
	Type                HackathonType
	Status              HackathonStatus
	Theme               string
	Challenges          []Challenge
	PrizePool           float64
	Prizes              []Prize
	Sponsors            []Sponsor
	Judges              []Judge
	Mentors             []Mentor
	Schedule            HackathonSchedule
	Rules               HackathonRules
	Requirements        ParticipationRequirements
	Teams               []Team
	Submissions         []Submission
	Winners             []Winner
	Tracks              []Track
	WorkshopSchedule    []Workshop
	Resources           []Resource
	Metrics             HackathonMetrics
	StartDate           time.Time
	EndDate             time.Time
	RegistrationDeadline time.Time
	CreatedAt           time.Time
	UpdatedAt           time.Time
}

// Challenge represents hackathon challenge
type Challenge struct {
	ID              string
	Title           string
	Description     string
	Category        string
	Difficulty      string // beginner, intermediate, advanced
	Problem         string
	Context         string
	Dataset         string
	APIs            []string
	Constraints     []string
	EvaluationCriteria []EvaluationCriterion
	Prize           float64
	SponsorID       string
	MaxTeamSize     int
	SubmissionCount int
}

// Prize represents hackathon prize
type Prize struct {
	ID          string
	Place       int
	Name        string
	Amount      float64
	Currency    string
	Description string
	Benefits    []string // AWS credits, mentorship, etc.
	TrackID     string
}

// Sponsor represents hackathon sponsor
type Sponsor struct {
	ID              string
	Name            string
	Logo            string
	Tier            string // platinum, gold, silver, bronze
	Contribution    float64
	Benefits        []string
	ChallengeID     string
	Resources       []Resource
	Recruiters      []Recruiter
}

// Recruiter represents company recruiter
type Recruiter struct {
	ID          string
	Name        string
	Email       string
	Company     string
	Position    string
	LookingFor  []string
	InterestedTeams []string
}

// Judge represents hackathon judge
type Judge struct {
	ID              string
	Name            string
	Title           string
	Company         string
	Bio             string
	Expertise       []string
	Photo           string
	LinkedinURL     string
	AssignedTracks  []string
}

// Mentor represents hackathon mentor
type Mentor struct {
	ID              string
	Name            string
	Expertise       []string
	Company         string
	Bio             string
	Photo           string
	AvailableHours  []MentorSession
	AssignedTeams   []string
	Rating          float64
	SessionCount    int
}

// MentorSession represents mentor availability
type MentorSession struct {
	StartTime   time.Time
	EndTime     time.Time
	Capacity    int
	Booked      int
	Format      string // video, chat, in-person
}

// HackathonSchedule defines event timeline
type HackathonSchedule struct {
	OpeningCeremony     time.Time
	TeamFormation       time.Time
	HackingBegins       time.Time
	CheckpointMilestones []Milestone
	SubmissionDeadline  time.Time
	DemoDay             time.Time
	JudgingPeriod       TimeRange
	WinnerAnnouncement  time.Time
	ClosingCeremony     time.Time
}

// Milestone represents progress checkpoint
type Milestone struct {
	ID          string
	Name        string
	Description string
	DueDate     time.Time
	Deliverable string
	Required    bool
}

// TimeRange represents time period
type TimeRange struct {
	Start time.Time
	End   time.Time
}

// HackathonRules defines participation rules
type HackathonRules struct {
	TeamMinSize         int
	TeamMaxSize         int
	SubmissionsPerTeam  int
	TracksPerTeam       int
	CodeOfConduct       string
	IPRights            string
	DisqualificationRules []string
	AllowedTechnologies []string
	BannedTechnologies  []string
	OpenSource          bool
	OriginalWork        bool
}

// ParticipationRequirements defines eligibility
type ParticipationRequirements struct {
	MinAge              int
	Geography           []string // allowed countries
	EducationLevel      string
	ProfessionalLevel   string
	RequiredSkills      []string
	PriorExperience     bool
	BackgroundCheck     bool
}

// Team represents hackathon team
type Team struct {
	ID              string
	Name            string
	Tagline         string
	Members         []TeamMember
	CaptainID       string
	SelectedTrack   string
	SelectedChallenge string
	Mentors         []string
	Status          string // registered, active, submitted, disqualified
	Repository      string
	DemoURL         string
	PitchDeck       string
	Progress        TeamProgress
	Communications  []TeamMessage
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// TeamMember represents team participant
type TeamMember struct {
	UserID      string
	Name        string
	Email       string
	Role        string // developer, designer, pm
	Skills      []string
	Photo       string
	LinkedinURL string
	GithubURL   string
	JoinedAt    time.Time
}

// TeamProgress tracks team progress
type TeamProgress struct {
	MilestonesCompleted []string
	CommitsCount        int
	LastCommitAt        time.Time
	DemoStatus          string
	PitchStatus         string
	CurrentPhase        string
	Blockers            []string
	MentorSessions      int
}

// TeamMessage represents team communication
type TeamMessage struct {
	ID          string
	FromUserID  string
	FromName    string
	Message     string
	Timestamp   time.Time
	Type        string // chat, announcement, mentor
}

// Submission represents hackathon submission
type Submission struct {
	ID              string
	TeamID          string
	ChallengeID     string
	Title           string
	Description     string
	Problem         string
	Solution        string
	TechStack       []string
	Repository      string
	DemoURL         string
	VideoURL        string
	PitchDeck       string
	Documentation   string
	Screenshots     []string
	LiveDemo        bool
	SubmittedAt     time.Time
	Scores          []Score
	TotalScore      float64
	Rank            int
	FeedbackPublic  []Feedback
	FeedbackPrivate []Feedback
	Status          string // submitted, under_review, accepted, rejected
}

// Score represents judge score
type Score struct {
	JudgeID     string
	JudgeName   string
	Criterion   string
	Score       float64
	MaxScore    float64
	Comment     string
	ScoredAt    time.Time
}

// Feedback represents submission feedback
type Feedback struct {
	FromID      string
	FromName    string
	Category    string
	Rating      int
	Comment     string
	Suggestions []string
	CreatedAt   time.Time
}

// EvaluationCriterion defines judging criterion
type EvaluationCriterion struct {
	Name        string
	Description string
	Weight      float64
	MaxScore    float64
	Rubric      []RubricLevel
}

// RubricLevel defines scoring rubric
type RubricLevel struct {
	Score       float64
	Label       string
	Description string
}

// Winner represents hackathon winner
type Winner struct {
	TeamID          string
	TeamName        string
	SubmissionID    string
	Prize           Prize
	TrackID         string
	ChallengeID     string
	FinalScore      float64
	AnnouncedAt     time.Time
	Awards          []Award
}

// Award represents additional recognition
type Award struct {
	ID          string
	Name        string
	Description string
	Value       float64
	Certificate string
	Badge       string
}

// Track represents competition track
type Track struct {
	ID              string
	Name            string
	Description     string
	Category        string
	PrizePool       float64
	Judges          []string
	Mentors         []string
	Teams           []string
	Requirements    []string
	Technologies    []string
	PartnerCompany  string
}

// Workshop represents learning workshop
type Workshop struct {
	ID              string
	Title           string
	Description     string
	Speaker         string
	Company         string
	StartTime       time.Time
	EndTime         time.Time
	Location        string
	Virtual         bool
	MeetingURL      string
	Capacity        int
	Registered      int
	Recording       string
	Materials       []string
}

// Resource represents hackathon resource
type Resource struct {
	ID          string
	Type        string // api, dataset, tool, library, tutorial
	Name        string
	Description string
	URL         string
	Provider    string
	Free        bool
	Credits     float64
	Documentation string
}

// HackathonMetrics tracks hackathon performance
type HackathonMetrics struct {
	TotalParticipants   int
	TotalTeams          int
	TotalSubmissions    int
	CompletionRate      float64
	AverageTeamSize     float64
	TotalCommits        int
	TotalMentorSessions int
	ParticipantSatisfaction float64
	SponsorSatisfaction float64
	MediaReach          int
	SocialEngagement    int
	JobOffers           int
	StartupsFunded      int
	FollowOnFunding     float64
}

// StartupProgram represents startup accelerator
type StartupProgram struct {
	ID                  string
	Name                string
	Cohort              int
	Description         string
	Duration            int // weeks
	Status              string // accepting, in_progress, completed
	Startups            []Startup
	Mentors             []Mentor
	Investors           []Investor
	Curriculum          []CurriculumModule
	Milestones          []ProgramMilestone
	DemoDay             time.Time
	TotalFunding        float64
	SuccessRate         float64
	GraduationRate      float64
	StartDate           time.Time
	EndDate             time.Time
	ApplicationDeadline time.Time
	CreatedAt           time.Time
}

// Startup represents accelerator startup
type Startup struct {
	ID              string
	Name            string
	Description     string
	Problem         string
	Solution        string
	Market          string
	Team            []Founder
	Industry        string
	Stage           string // idea, mvp, beta, launched
	Revenue         float64
	Customers       int
	Growth          GrowthMetrics
	Funding         FundingInfo
	Pitch           PitchInfo
	Traction        TractionMetrics
	Product         ProductInfo
	BusinessModel   string
	Status          string // active, graduated, acquired, failed
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// Founder represents startup founder
type Founder struct {
	UserID      string
	Name        string
	Role        string
	Background  string
	Skills      []string
	LinkedIn    string
	Twitter     string
	Photo       string
}

// GrowthMetrics tracks startup growth
type GrowthMetrics struct {
	MRR             float64 // Monthly Recurring Revenue
	ARR             float64 // Annual Recurring Revenue
	GrowthRate      float64
	ChurnRate       float64
	CAC             float64 // Customer Acquisition Cost
	LTV             float64 // Lifetime Value
	Runway          int     // months
	BurnRate        float64
}

// FundingInfo tracks funding status
type FundingInfo struct {
	TotalRaised     float64
	CurrentRound    string // seed, series_a, etc.
	Valuation       float64
	Investors       []string
	LastFundingDate time.Time
	NextRound       string
	FundingGoal     float64
}

// PitchInfo contains pitch materials
type PitchInfo struct {
	DeckURL         string
	VideoURL        string
	OneLinePitch    string
	ElevatorPitch   string
	ProblemStatement string
	Solution        string
	USP             string // Unique Selling Proposition
	GoToMarket      string
}

// TractionMetrics tracks startup traction
type TractionMetrics struct {
	Users           int
	ActiveUsers     int
	PayingCustomers int
	Trials          int
	Waitlist        int
	Partnerships    int
	MediaMentions   int
	Awards          []string
}

// ProductInfo describes product
type ProductInfo struct {
	Name            string
	Description     string
	Status          string // development, beta, launched
	URL             string
	Features        []string
	Technologies    []string
	Screenshots     []string
	VideoDemo       string
}

// Investor represents venture investor
type Investor struct {
	ID              string
	Name            string
	Firm            string
	Role            string
	Focus           []string // industries
	Stage           []string // seed, series_a
	TicketSize      TicketRange
	Portfolio       []string
	Notable         []string
	LinkedinURL     string
	Email           string
	InterestedIn    []string
}

// TicketRange represents investment range
type TicketRange struct {
	Min float64
	Max float64
}

// CurriculumModule represents program module
type CurriculumModule struct {
	Week        int
	Title       string
	Topics      []string
	Deliverable string
	Mentor      string
	Resources   []string
}

// ProgramMilestone represents program checkpoint
type ProgramMilestone struct {
	Week        int
	Name        string
	Description string
	Required    bool
	Deliverable string
	Review      bool
}

// InnovationEngine manages hackathons and programs
type InnovationEngine struct {
	mu              sync.RWMutex
	hackathons      map[string]*Hackathon
	teams           map[string]*Team
	submissions     map[string]*Submission
	programs        map[string]*StartupProgram
	startups        map[string]*Startup
	investors       map[string]*Investor
	stats           InnovationStats
}

// InnovationStats tracks innovation metrics
type InnovationStats struct {
	TotalHackathons         int
	ActiveHackathons        int
	TotalParticipants       int
	TotalTeams              int
	TotalSubmissions        int
	TotalPrizePool          float64
	TotalPrizesAwarded      float64
	AverageParticipation    float64
	TotalStartups           int
	ActiveStartups          int
	FundedStartups          int
	TotalFunding            float64
	AverageValuation        float64
	SuccessRate             float64
	JobsCreated             int
	TotalInvestors          int
	ActivePrograms          int
	UpdatedAt               time.Time
}

// NewInnovationEngine creates innovation engine
func NewInnovationEngine() *InnovationEngine {
	ie := &InnovationEngine{
		hackathons:  make(map[string]*Hackathon),
		teams:       make(map[string]*Team),
		submissions: make(map[string]*Submission),
		programs:    make(map[string]*StartupProgram),
		startups:    make(map[string]*Startup),
		investors:   make(map[string]*Investor),
	}

	ie.initializeSampleData()

	return ie
}

// initializeSampleData creates sample hackathons and programs
func (ie *InnovationEngine) initializeSampleData() {
	// Create 12 monthly hackathons
	for i := 0; i < 12; i++ {
		startDate := time.Now().AddDate(0, -12+i, 0)
		hackathon := &Hackathon{
			ID:          ie.generateID("HACK"),
			Name:        fmt.Sprintf("Innovation Hackathon %d", i+1),
			Description: "Build innovative solutions to real-world problems",
			Type:        TypeCompetition,
			Status:      StatusCompleted,
			Theme:       []string{"AI/ML", "Cloud Native", "Security", "Sustainability"}[i%4],
			PrizePool:   100000.0 + float64(i*10000),
			Prizes: []Prize{
				{ID: "1st", Place: 1, Name: "First Place", Amount: 50000, Currency: "USD"},
				{ID: "2nd", Place: 2, Name: "Second Place", Amount: 30000, Currency: "USD"},
				{ID: "3rd", Place: 3, Name: "Third Place", Amount: 20000, Currency: "USD"},
			},
			StartDate:            startDate,
			EndDate:              startDate.AddDate(0, 0, 3),
			RegistrationDeadline: startDate.AddDate(0, 0, -7),
			CreatedAt:            startDate.AddDate(0, 0, -30),
			UpdatedAt:            time.Now(),
		}

		// Add challenges
		hackathon.Challenges = []Challenge{
			{
				ID:          ie.generateID("CHAL"),
				Title:       "Challenge 1",
				Category:    "Technical",
				Difficulty:  "Advanced",
				Prize:       25000,
				MaxTeamSize: 5,
			},
			{
				ID:          ie.generateID("CHAL"),
				Title:       "Challenge 2",
				Category:    "Business",
				Difficulty:  "Intermediate",
				Prize:       15000,
				MaxTeamSize: 4,
			},
		}

		// Add metrics
		hackathon.Metrics = HackathonMetrics{
			TotalParticipants:       (i + 1) * 200,
			TotalTeams:              (i + 1) * 50,
			TotalSubmissions:        (i + 1) * 45,
			CompletionRate:          0.85,
			AverageTeamSize:         4.2,
			TotalCommits:            (i + 1) * 5000,
			TotalMentorSessions:     (i + 1) * 150,
			ParticipantSatisfaction: 4.5,
			SponsorSatisfaction:     4.8,
			MediaReach:              (i + 1) * 50000,
			JobOffers:               (i + 1) * 25,
			StartupsFunded:          (i + 1) * 5,
			FollowOnFunding:         float64((i + 1) * 500000),
		}

		ie.hackathons[hackathon.ID] = hackathon
	}

	// Create 4 startup programs
	for i := 0; i < 4; i++ {
		program := &StartupProgram{
			ID:             ie.generateID("PROG"),
			Name:           fmt.Sprintf("Accelerator Cohort %d", i+1),
			Cohort:         i + 1,
			Description:    "12-week intensive startup accelerator",
			Duration:       12,
			Status:         "completed",
			TotalFunding:   float64((i + 1) * 2000000),
			SuccessRate:    0.65,
			GraduationRate: 0.85,
			StartDate:      time.Now().AddDate(0, -12+(i*3), 0),
			EndDate:        time.Now().AddDate(0, -9+(i*3), 0),
			CreatedAt:      time.Now().AddDate(0, -13+(i*3), 0),
		}

		ie.programs[program.ID] = program
	}

	// Create sample investors
	investorNames := []string{"Tech Ventures", "Innovation Capital", "Startup Fund", "Growth Partners"}
	for i, name := range investorNames {
		investor := &Investor{
			ID:   ie.generateID("INV"),
			Name: name,
			Firm: name,
			Role: "Partner",
			Focus: []string{"AI/ML", "Cloud", "SaaS"},
			Stage: []string{"seed", "series_a"},
			TicketRange: TicketRange{
				Min: 100000,
				Max: 5000000,
			},
			Portfolio: []string{},
			Notable:   []string{"Startup A", "Startup B"},
		}

		ie.investors[investor.ID] = investor
	}
}

// CreateHackathon creates new hackathon
func (ie *InnovationEngine) CreateHackathon(ctx context.Context, hackathon *Hackathon) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	if hackathon.ID == "" {
		hackathon.ID = ie.generateID("HACK")
	}

	hackathon.Status = StatusPlanning
	hackathon.CreatedAt = time.Now()
	hackathon.UpdatedAt = time.Now()

	ie.hackathons[hackathon.ID] = hackathon

	ie.stats.TotalHackathons++
	ie.stats.TotalPrizePool += hackathon.PrizePool
	ie.stats.UpdatedAt = time.Now()

	return nil
}

// RegisterTeam registers team for hackathon
func (ie *InnovationEngine) RegisterTeam(ctx context.Context, hackathonID string, team *Team) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	hackathon, exists := ie.hackathons[hackathonID]
	if !exists {
		return fmt.Errorf("hackathon not found: %s", hackathonID)
	}

	if time.Now().After(hackathon.RegistrationDeadline) {
		return fmt.Errorf("registration deadline passed")
	}

	if team.ID == "" {
		team.ID = ie.generateID("TEAM")
	}

	team.Status = "registered"
	team.CreatedAt = time.Now()
	team.UpdatedAt = time.Now()

	ie.teams[team.ID] = team
	hackathon.Teams = append(hackathon.Teams, *team)
	hackathon.Metrics.TotalTeams++
	hackathon.Metrics.TotalParticipants += len(team.Members)

	ie.stats.TotalTeams++
	ie.stats.TotalParticipants += len(team.Members)
	ie.stats.UpdatedAt = time.Now()

	return nil
}

// SubmitProject submits hackathon project
func (ie *InnovationEngine) SubmitProject(ctx context.Context, submission *Submission) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	team, exists := ie.teams[submission.TeamID]
	if !exists {
		return fmt.Errorf("team not found: %s", submission.TeamID)
	}

	if submission.ID == "" {
		submission.ID = ie.generateID("SUB")
	}

	submission.Status = "submitted"
	submission.SubmittedAt = time.Now()

	ie.submissions[submission.ID] = submission
	team.Status = "submitted"
	team.UpdatedAt = time.Now()

	ie.stats.TotalSubmissions++
	ie.stats.UpdatedAt = time.Now()

	return nil
}

// ScoreSubmission scores submission
func (ie *InnovationEngine) ScoreSubmission(ctx context.Context, submissionID, judgeID string, scores []Score) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	submission, exists := ie.submissions[submissionID]
	if !exists {
		return fmt.Errorf("submission not found: %s", submissionID)
	}

	submission.Scores = append(submission.Scores, scores...)

	// Calculate total score
	totalScore := 0.0
	for _, score := range submission.Scores {
		totalScore += score.Score
	}
	submission.TotalScore = totalScore / float64(len(submission.Scores))

	return nil
}

// AnnounceWinners announces hackathon winners
func (ie *InnovationEngine) AnnounceWinners(ctx context.Context, hackathonID string, winners []Winner) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	hackathon, exists := ie.hackathons[hackathonID]
	if !exists {
		return fmt.Errorf("hackathon not found: %s", hackathonID)
	}

	hackathon.Winners = winners
	hackathon.Status = StatusCompleted
	hackathon.UpdatedAt = time.Now()

	// Update stats
	totalPrizes := 0.0
	for _, winner := range winners {
		totalPrizes += winner.Prize.Amount
	}
	ie.stats.TotalPrizesAwarded += totalPrizes
	ie.stats.UpdatedAt = time.Now()

	return nil
}

// CreateStartupProgram creates accelerator program
func (ie *InnovationEngine) CreateStartupProgram(ctx context.Context, program *StartupProgram) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	if program.ID == "" {
		program.ID = ie.generateID("PROG")
	}

	program.Status = "accepting"
	program.CreatedAt = time.Now()

	ie.programs[program.ID] = program

	ie.stats.ActivePrograms++
	ie.stats.UpdatedAt = time.Now()

	return nil
}

// AcceptStartup accepts startup to program
func (ie *InnovationEngine) AcceptStartup(ctx context.Context, programID string, startup *Startup) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	program, exists := ie.programs[programID]
	if !exists {
		return fmt.Errorf("program not found: %s", programID)
	}

	if startup.ID == "" {
		startup.ID = ie.generateID("STARTUP")
	}

	startup.Status = "active"
	startup.CreatedAt = time.Now()
	startup.UpdatedAt = time.Now()

	ie.startups[startup.ID] = startup
	program.Startups = append(program.Startups, *startup)

	ie.stats.TotalStartups++
	ie.stats.ActiveStartups++
	ie.stats.UpdatedAt = time.Now()

	return nil
}

// RecordFunding records startup funding
func (ie *InnovationEngine) RecordFunding(ctx context.Context, startupID string, funding FundingInfo) error {
	ie.mu.Lock()
	defer ie.mu.Unlock()

	startup, exists := ie.startups[startupID]
	if !exists {
		return fmt.Errorf("startup not found: %s", startupID)
	}

	startup.Funding = funding
	startup.UpdatedAt = time.Now()

	ie.stats.FundedStartups++
	ie.stats.TotalFunding += funding.TotalRaised
	ie.stats.UpdatedAt = time.Now()

	return nil
}

// GetInnovationStats returns innovation statistics
func (ie *InnovationEngine) GetInnovationStats(ctx context.Context) InnovationStats {
	ie.mu.RLock()
	defer ie.mu.RUnlock()

	stats := ie.stats

	// Calculate averages
	if stats.TotalHackathons > 0 {
		stats.AverageParticipation = float64(stats.TotalParticipants) / float64(stats.TotalHackathons)
	}

	if stats.TotalStartups > 0 {
		stats.SuccessRate = float64(stats.FundedStartups) / float64(stats.TotalStartups)
	}

	// Count active hackathons
	activeCount := 0
	for _, h := range ie.hackathons {
		if h.Status == StatusActive || h.Status == StatusRegistration {
			activeCount++
		}
	}
	stats.ActiveHackathons = activeCount

	stats.TotalInvestors = len(ie.investors)
	stats.UpdatedAt = time.Now()

	return stats
}

// generateID generates unique ID
func (ie *InnovationEngine) generateID(prefix string) string {
	timestamp := time.Now().UnixNano()
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s-%d", prefix, timestamp)))
	return fmt.Sprintf("%s-%s", prefix, hex.EncodeToString(hash[:8]))
}

// ExportHackathonData exports hackathon data as JSON
func (ie *InnovationEngine) ExportHackathonData(ctx context.Context, hackathonID string) ([]byte, error) {
	ie.mu.RLock()
	defer ie.mu.RUnlock()

	hackathon, exists := ie.hackathons[hackathonID]
	if !exists {
		return nil, fmt.Errorf("hackathon not found: %s", hackathonID)
	}

	return json.MarshalIndent(hackathon, "", "  ")
}
