// Package certification provides comprehensive developer certification platform
// Implements 3-tier certification system with blockchain verification
package certification

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// CertificationLevel represents the certification tier
type CertificationLevel string

const (
	LevelDeveloper CertificationLevel = "DEVELOPER"
	LevelArchitect CertificationLevel = "ARCHITECT"
	LevelExpert    CertificationLevel = "EXPERT"
)

// CertificationStatus represents current certification state
type CertificationStatus string

const (
	StatusPending   CertificationStatus = "PENDING"
	StatusActive    CertificationStatus = "ACTIVE"
	StatusExpired   CertificationStatus = "EXPIRED"
	StatusRevoked   CertificationStatus = "REVOKED"
	StatusSuspended CertificationStatus = "SUSPENDED"
)

// ExamStatus represents exam attempt status
type ExamStatus string

const (
	ExamStatusScheduled  ExamStatus = "SCHEDULED"
	ExamStatusInProgress ExamStatus = "IN_PROGRESS"
	ExamStatusCompleted  ExamStatus = "COMPLETED"
	ExamStatusPassed     ExamStatus = "PASSED"
	ExamStatusFailed     ExamStatus = "FAILED"
	ExamStatusCancelled  ExamStatus = "CANCELLED"
)

// CertificationRequirements defines requirements for each level
type CertificationRequirements struct {
	Level              CertificationLevel
	MinimumStudyHours  int
	MinimumExamScore   float64
	YearsExperience    int
	CommunityPoints    int
	PrerequisiteCerts  []CertificationLevel
	PracticalProjects  int
	Description        string
	ValidityYears      int
	RenewalCEURequired int // Continuing Education Units
}

// Certificate represents an issued certification
type Certificate struct {
	ID                string              `json:"id"`
	UserID            string              `json:"user_id"`
	Level             CertificationLevel  `json:"level"`
	Status            CertificationStatus `json:"status"`
	IssueDate         time.Time           `json:"issue_date"`
	ExpiryDate        time.Time           `json:"expiry_date"`
	CertificateNumber string              `json:"certificate_number"`
	BlockchainHash    string              `json:"blockchain_hash"`
	VerificationURL   string              `json:"verification_url"`
	Metadata          CertificateMetadata `json:"metadata"`
	RenewalHistory    []RenewalRecord     `json:"renewal_history"`
	CreatedAt         time.Time           `json:"created_at"`
	UpdatedAt         time.Time           `json:"updated_at"`
}

// CertificateMetadata contains additional certificate information
type CertificateMetadata struct {
	ExamScore          float64           `json:"exam_score"`
	StudyHours         int               `json:"study_hours"`
	PracticalProjects  []ProjectRecord   `json:"practical_projects"`
	CommunityPoints    int               `json:"community_points"`
	YearsExperience    int               `json:"years_experience"`
	SpecializationTags []string          `json:"specialization_tags"`
	Endorsements       []Endorsement     `json:"endorsements"`
	BadgesEarned       []string          `json:"badges_earned"`
}

// RenewalRecord tracks certification renewals
type RenewalRecord struct {
	ID               string    `json:"id"`
	RenewalDate      time.Time `json:"renewal_date"`
	CEUCompleted     int       `json:"ceu_completed"`
	CEURequired      int       `json:"ceu_required"`
	PreviousExpiry   time.Time `json:"previous_expiry"`
	NewExpiry        time.Time `json:"new_expiry"`
	RenewalExamScore float64   `json:"renewal_exam_score,omitempty"`
}

// ProjectRecord represents a practical project for certification
type ProjectRecord struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	GitHubURL   string    `json:"github_url"`
	DemoURL     string    `json:"demo_url"`
	Score       float64   `json:"score"`
	Feedback    string    `json:"feedback"`
	SubmittedAt time.Time `json:"submitted_at"`
	ReviewedAt  time.Time `json:"reviewed_at"`
}

// Endorsement represents professional endorsement
type Endorsement struct {
	ID          string    `json:"id"`
	EndorserID  string    `json:"endorser_id"`
	EndorserName string   `json:"endorser_name"`
	Relationship string   `json:"relationship"`
	Comments    string    `json:"comments"`
	CreatedAt   time.Time `json:"created_at"`
}

// Exam represents a certification exam
type Exam struct {
	ID                 string              `json:"id"`
	Level              CertificationLevel  `json:"level"`
	Title              string              `json:"title"`
	Description        string              `json:"description"`
	DurationMinutes    int                 `json:"duration_minutes"`
	TotalQuestions     int                 `json:"total_questions"`
	PassingScore       float64             `json:"passing_score"`
	QuestionBank       []Question          `json:"question_bank"`
	PracticalLabs      []PracticalLab      `json:"practical_labs"`
	ProctoringRequired bool                `json:"proctoring_required"`
	AllowedAttempts    int                 `json:"allowed_attempts"`
	RetakeWaitingDays  int                 `json:"retake_waiting_days"`
	CreatedAt          time.Time           `json:"created_at"`
	UpdatedAt          time.Time           `json:"updated_at"`
}

// Question represents an exam question
type Question struct {
	ID              string         `json:"id"`
	Type            QuestionType   `json:"type"`
	Category        string         `json:"category"`
	Difficulty      string         `json:"difficulty"`
	Question        string         `json:"question"`
	Options         []string       `json:"options,omitempty"`
	CorrectAnswer   interface{}    `json:"correct_answer"`
	Explanation     string         `json:"explanation"`
	Points          int            `json:"points"`
	CodeSnippet     string         `json:"code_snippet,omitempty"`
	Tags            []string       `json:"tags"`
}

// QuestionType defines question format
type QuestionType string

const (
	QuestionTypeMultipleChoice QuestionType = "MULTIPLE_CHOICE"
	QuestionTypeMultipleAnswer QuestionType = "MULTIPLE_ANSWER"
	QuestionTypeTrueFalse      QuestionType = "TRUE_FALSE"
	QuestionTypeShortAnswer    QuestionType = "SHORT_ANSWER"
	QuestionTypeCoding         QuestionType = "CODING"
	QuestionTypeCaseStudy      QuestionType = "CASE_STUDY"
)

// PracticalLab represents hands-on lab exercise
type PracticalLab struct {
	ID              string        `json:"id"`
	Title           string        `json:"title"`
	Description     string        `json:"description"`
	Instructions    string        `json:"instructions"`
	TimeLimit       time.Duration `json:"time_limit"`
	Environment     LabEnvironment `json:"environment"`
	ValidationTests []ValidationTest `json:"validation_tests"`
	Points          int           `json:"points"`
}

// LabEnvironment defines lab sandbox environment
type LabEnvironment struct {
	ContainerImage  string            `json:"container_image"`
	Resources       ResourceLimits    `json:"resources"`
	PreloadedFiles  map[string]string `json:"preloaded_files"`
	AllowedPorts    []int             `json:"allowed_ports"`
	NetworkAccess   bool              `json:"network_access"`
}

// ResourceLimits defines resource constraints for lab environment
type ResourceLimits struct {
	CPUCores       float64 `json:"cpu_cores"`
	MemoryMB       int     `json:"memory_mb"`
	DiskMB         int     `json:"disk_mb"`
	ExecutionTime  int     `json:"execution_time_seconds"`
}

// ValidationTest represents automated test for lab
type ValidationTest struct {
	ID          string   `json:"id"`
	Description string   `json:"description"`
	TestCommand string   `json:"test_command"`
	Expected    string   `json:"expected"`
	Points      int      `json:"points"`
}

// ExamAttempt tracks individual exam attempts
type ExamAttempt struct {
	ID                 string      `json:"id"`
	ExamID             string      `json:"exam_id"`
	UserID             string      `json:"user_id"`
	AttemptNumber      int         `json:"attempt_number"`
	Status             ExamStatus  `json:"status"`
	ScheduledTime      time.Time   `json:"scheduled_time"`
	StartTime          time.Time   `json:"start_time"`
	EndTime            time.Time   `json:"end_time"`
	Answers            []Answer    `json:"answers"`
	LabResults         []LabResult `json:"lab_results"`
	Score              float64     `json:"score"`
	Passed             bool        `json:"passed"`
	ProctorRecordingURL string     `json:"proctor_recording_url"`
	ProctoringFlags    []ProctoringFlag `json:"proctoring_flags"`
	ReviewNotes        string      `json:"review_notes"`
	CreatedAt          time.Time   `json:"created_at"`
	UpdatedAt          time.Time   `json:"updated_at"`
}

// Answer represents answer to a question
type Answer struct {
	QuestionID      string      `json:"question_id"`
	ProvidedAnswer  interface{} `json:"provided_answer"`
	IsCorrect       bool        `json:"is_correct"`
	PointsEarned    int         `json:"points_earned"`
	TimeSpentSeconds int        `json:"time_spent_seconds"`
}

// LabResult represents practical lab submission result
type LabResult struct {
	LabID              string    `json:"lab_id"`
	SubmissionTime     time.Time `json:"submission_time"`
	ValidationResults  []ValidationResult `json:"validation_results"`
	TotalPointsEarned  int       `json:"total_points_earned"`
	TotalPointsPossible int      `json:"total_points_possible"`
	ManualReviewNeeded bool      `json:"manual_review_needed"`
	ReviewerFeedback   string    `json:"reviewer_feedback"`
}

// ValidationResult represents result of a validation test
type ValidationResult struct {
	TestID       string `json:"test_id"`
	Passed       bool   `json:"passed"`
	ActualOutput string `json:"actual_output"`
	PointsEarned int    `json:"points_earned"`
}

// ProctoringFlag represents suspicious activity during exam
type ProctoringFlag struct {
	Timestamp   time.Time `json:"timestamp"`
	FlagType    string    `json:"flag_type"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	ScreenshotURL string  `json:"screenshot_url"`
}

// StudyProgress tracks learner progress towards certification
type StudyProgress struct {
	UserID              string              `json:"user_id"`
	TargetLevel         CertificationLevel  `json:"target_level"`
	CompletedModules    []string            `json:"completed_modules"`
	TotalModules        int                 `json:"total_modules"`
	StudyHoursLogged    int                 `json:"study_hours_logged"`
	PracticeTestScores  []float64           `json:"practice_test_scores"`
	ProjectsCompleted   []string            `json:"projects_completed"`
	LabsCompleted       []string            `json:"labs_completed"`
	ReadinessScore      float64             `json:"readiness_score"`
	RecommendedExamDate time.Time           `json:"recommended_exam_date"`
	LastUpdated         time.Time           `json:"last_updated"`
}

// ContinuingEducation tracks CEU credits for renewal
type ContinuingEducation struct {
	ID            string    `json:"id"`
	UserID        string    `json:"user_id"`
	CertificateID string    `json:"certificate_id"`
	ActivityType  string    `json:"activity_type"`
	Title         string    `json:"title"`
	Description   string    `json:"description"`
	CEUCredits    int       `json:"ceu_credits"`
	CompletionDate time.Time `json:"completion_date"`
	VerificationURL string  `json:"verification_url"`
	ApprovalStatus string   `json:"approval_status"`
	CreatedAt     time.Time `json:"created_at"`
}

// CertificationPlatform manages the entire certification system
type CertificationPlatform struct {
	mu                    sync.RWMutex
	certificates          map[string]*Certificate
	exams                 map[string]*Exam
	examAttempts          map[string]*ExamAttempt
	studyProgress         map[string]*StudyProgress
	continuingEducation   map[string][]*ContinuingEducation
	requirements          map[CertificationLevel]*CertificationRequirements
	blockchainVerifier    BlockchainVerifier
	proctoringService     ProctoringService
	labEnvironmentManager LabEnvironmentManager
	notificationService   NotificationService
	metricsCollector      MetricsCollector
}

// BlockchainVerifier handles certificate blockchain verification
type BlockchainVerifier interface {
	RecordCertificate(ctx context.Context, cert *Certificate) (string, error)
	VerifyCertificate(ctx context.Context, hash string) (*Certificate, error)
	RevokeCertificate(ctx context.Context, certID string) error
}

// ProctoringService handles exam proctoring
type ProctoringService interface {
	StartProctoring(ctx context.Context, attemptID string) error
	MonitorExam(ctx context.Context, attemptID string) ([]ProctoringFlag, error)
	EndProctoring(ctx context.Context, attemptID string) (string, error)
	ReviewRecording(ctx context.Context, recordingURL string) ([]ProctoringFlag, error)
}

// LabEnvironmentManager manages hands-on lab environments
type LabEnvironmentManager interface {
	CreateEnvironment(ctx context.Context, lab *PracticalLab, userID string) (string, error)
	ExecuteValidation(ctx context.Context, envID string, test ValidationTest) (*ValidationResult, error)
	DestroyEnvironment(ctx context.Context, envID string) error
	GetEnvironmentStatus(ctx context.Context, envID string) (string, error)
}

// NotificationService handles user notifications
type NotificationService interface {
	NotifyExamScheduled(ctx context.Context, userID string, exam *Exam) error
	NotifyExamResult(ctx context.Context, userID string, attempt *ExamAttempt) error
	NotifyCertificateIssued(ctx context.Context, userID string, cert *Certificate) error
	NotifyCertificateExpiring(ctx context.Context, userID string, cert *Certificate) error
	NotifyRenewalRequired(ctx context.Context, userID string, cert *Certificate) error
}

// MetricsCollector collects platform metrics
type MetricsCollector interface {
	RecordExamAttempt(ctx context.Context, attempt *ExamAttempt)
	RecordCertificateIssued(ctx context.Context, cert *Certificate)
	RecordStudyProgress(ctx context.Context, progress *StudyProgress)
	RecordCEUActivity(ctx context.Context, activity *ContinuingEducation)
}

// NewCertificationPlatform creates a new certification platform instance
func NewCertificationPlatform(
	blockchainVerifier BlockchainVerifier,
	proctoringService ProctoringService,
	labManager LabEnvironmentManager,
	notificationService NotificationService,
	metricsCollector MetricsCollector,
) *CertificationPlatform {
	platform := &CertificationPlatform{
		certificates:          make(map[string]*Certificate),
		exams:                 make(map[string]*Exam),
		examAttempts:          make(map[string]*ExamAttempt),
		studyProgress:         make(map[string]*StudyProgress),
		continuingEducation:   make(map[string][]*ContinuingEducation),
		requirements:          make(map[CertificationLevel]*CertificationRequirements),
		blockchainVerifier:    blockchainVerifier,
		proctoringService:     proctoringService,
		labEnvironmentManager: labManager,
		notificationService:   notificationService,
		metricsCollector:      metricsCollector,
	}

	// Initialize certification requirements
	platform.initializeRequirements()

	return platform
}

// initializeRequirements sets up certification level requirements
func (cp *CertificationPlatform) initializeRequirements() {
	cp.requirements[LevelDeveloper] = &CertificationRequirements{
		Level:              LevelDeveloper,
		MinimumStudyHours:  100,
		MinimumExamScore:   90.0,
		YearsExperience:    0,
		CommunityPoints:    0,
		PrerequisiteCerts:  []CertificationLevel{},
		PracticalProjects:  1,
		Description:        "NovaCron Certified Developer - Foundation level certification for DWCP development",
		ValidityYears:      2,
		RenewalCEURequired: 20,
	}

	cp.requirements[LevelArchitect] = &CertificationRequirements{
		Level:              LevelArchitect,
		MinimumStudyHours:  200,
		MinimumExamScore:   95.0,
		YearsExperience:    2,
		CommunityPoints:    500,
		PrerequisiteCerts:  []CertificationLevel{LevelDeveloper},
		PracticalProjects:  3,
		Description:        "NovaCron Certified Architect - Advanced certification for system architecture and design",
		ValidityYears:      2,
		RenewalCEURequired: 30,
	}

	cp.requirements[LevelExpert] = &CertificationRequirements{
		Level:              LevelExpert,
		MinimumStudyHours:  500,
		MinimumExamScore:   95.0,
		YearsExperience:    5,
		CommunityPoints:    2000,
		PrerequisiteCerts:  []CertificationLevel{LevelDeveloper, LevelArchitect},
		PracticalProjects:  5,
		Description:        "NovaCron Certified Expert - Master level certification with significant community contribution",
		ValidityYears:      2,
		RenewalCEURequired: 50,
	}
}

// GetRequirements returns requirements for a certification level
func (cp *CertificationPlatform) GetRequirements(level CertificationLevel) (*CertificationRequirements, error) {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	req, exists := cp.requirements[level]
	if !exists {
		return nil, errors.New("certification level not found")
	}

	return req, nil
}

// CreateExam creates a new certification exam
func (cp *CertificationPlatform) CreateExam(ctx context.Context, exam *Exam) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if exam.ID == "" {
		exam.ID = uuid.New().String()
	}

	exam.CreatedAt = time.Now()
	exam.UpdatedAt = time.Now()

	cp.exams[exam.ID] = exam

	return nil
}

// ScheduleExam schedules an exam attempt for a user
func (cp *CertificationPlatform) ScheduleExam(ctx context.Context, userID, examID string, scheduledTime time.Time) (*ExamAttempt, error) {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	exam, exists := cp.exams[examID]
	if !exists {
		return nil, errors.New("exam not found")
	}

	// Check if user meets requirements
	req, _ := cp.requirements[exam.Level]
	progress, hasProgress := cp.studyProgress[userID]

	if hasProgress && progress.StudyHoursLogged < req.MinimumStudyHours {
		return nil, fmt.Errorf("insufficient study hours: %d required, %d logged",
			req.MinimumStudyHours, progress.StudyHoursLogged)
	}

	// Count previous attempts
	attemptNumber := cp.countUserAttempts(userID, examID) + 1

	if exam.AllowedAttempts > 0 && attemptNumber > exam.AllowedAttempts {
		return nil, fmt.Errorf("maximum attempts (%d) exceeded", exam.AllowedAttempts)
	}

	// Check retake waiting period
	lastAttempt := cp.getLastAttempt(userID, examID)
	if lastAttempt != nil && exam.RetakeWaitingDays > 0 {
		waitUntil := lastAttempt.EndTime.Add(time.Duration(exam.RetakeWaitingDays) * 24 * time.Hour)
		if time.Now().Before(waitUntil) {
			return nil, fmt.Errorf("must wait until %s before retaking exam", waitUntil.Format(time.RFC3339))
		}
	}

	attempt := &ExamAttempt{
		ID:            uuid.New().String(),
		ExamID:        examID,
		UserID:        userID,
		AttemptNumber: attemptNumber,
		Status:        ExamStatusScheduled,
		ScheduledTime: scheduledTime,
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
	}

	cp.examAttempts[attempt.ID] = attempt

	// Send notification
	if cp.notificationService != nil {
		go cp.notificationService.NotifyExamScheduled(ctx, userID, exam)
	}

	return attempt, nil
}

// StartExam begins an exam attempt
func (cp *CertificationPlatform) StartExam(ctx context.Context, attemptID string) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	attempt, exists := cp.examAttempts[attemptID]
	if !exists {
		return errors.New("exam attempt not found")
	}

	if attempt.Status != ExamStatusScheduled {
		return fmt.Errorf("exam cannot be started from status: %s", attempt.Status)
	}

	exam, exists := cp.exams[attempt.ExamID]
	if !exists {
		return errors.New("exam not found")
	}

	attempt.Status = ExamStatusInProgress
	attempt.StartTime = time.Now()
	attempt.UpdatedAt = time.Now()

	// Start proctoring if required
	if exam.ProctoringRequired && cp.proctoringService != nil {
		if err := cp.proctoringService.StartProctoring(ctx, attemptID); err != nil {
			return fmt.Errorf("failed to start proctoring: %w", err)
		}
	}

	return nil
}

// SubmitAnswer submits an answer to a question
func (cp *CertificationPlatform) SubmitAnswer(ctx context.Context, attemptID, questionID string, answer interface{}, timeSpent int) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	attempt, exists := cp.examAttempts[attemptID]
	if !exists {
		return errors.New("exam attempt not found")
	}

	if attempt.Status != ExamStatusInProgress {
		return errors.New("exam is not in progress")
	}

	exam, exists := cp.exams[attempt.ExamID]
	if !exists {
		return errors.New("exam not found")
	}

	// Find the question
	var question *Question
	for i := range exam.QuestionBank {
		if exam.QuestionBank[i].ID == questionID {
			question = &exam.QuestionBank[i]
			break
		}
	}

	if question == nil {
		return errors.New("question not found")
	}

	// Grade the answer
	isCorrect := cp.gradeAnswer(question, answer)
	pointsEarned := 0
	if isCorrect {
		pointsEarned = question.Points
	}

	answerRecord := Answer{
		QuestionID:       questionID,
		ProvidedAnswer:   answer,
		IsCorrect:        isCorrect,
		PointsEarned:     pointsEarned,
		TimeSpentSeconds: timeSpent,
	}

	attempt.Answers = append(attempt.Answers, answerRecord)
	attempt.UpdatedAt = time.Now()

	return nil
}

// gradeAnswer evaluates if an answer is correct
func (cp *CertificationPlatform) gradeAnswer(question *Question, providedAnswer interface{}) bool {
	switch question.Type {
	case QuestionTypeMultipleChoice, QuestionTypeTrueFalse:
		return providedAnswer == question.CorrectAnswer
	case QuestionTypeMultipleAnswer:
		// Compare slices
		provided, ok1 := providedAnswer.([]string)
		correct, ok2 := question.CorrectAnswer.([]string)
		if !ok1 || !ok2 {
			return false
		}
		if len(provided) != len(correct) {
			return false
		}
		correctMap := make(map[string]bool)
		for _, c := range correct {
			correctMap[c] = true
		}
		for _, p := range provided {
			if !correctMap[p] {
				return false
			}
		}
		return true
	case QuestionTypeShortAnswer:
		// Case-insensitive string comparison
		provided, ok1 := providedAnswer.(string)
		correct, ok2 := question.CorrectAnswer.(string)
		if !ok1 || !ok2 {
			return false
		}
		return provided == correct
	case QuestionTypeCoding, QuestionTypeCaseStudy:
		// These require manual review
		return false
	default:
		return false
	}
}

// SubmitLabSolution submits a practical lab solution
func (cp *CertificationPlatform) SubmitLabSolution(ctx context.Context, attemptID, labID string) (*LabResult, error) {
	cp.mu.Lock()
	attempt, exists := cp.examAttempts[attemptID]
	if !exists {
		cp.mu.Unlock()
		return nil, errors.New("exam attempt not found")
	}

	if attempt.Status != ExamStatusInProgress {
		cp.mu.Unlock()
		return nil, errors.New("exam is not in progress")
	}

	exam, exists := cp.exams[attempt.ExamID]
	if !exists {
		cp.mu.Unlock()
		return nil, errors.New("exam not found")
	}

	// Find the lab
	var lab *PracticalLab
	for i := range exam.PracticalLabs {
		if exam.PracticalLabs[i].ID == labID {
			lab = &exam.PracticalLabs[i]
			break
		}
	}

	if lab == nil {
		cp.mu.Unlock()
		return nil, errors.New("lab not found")
	}
	cp.mu.Unlock()

	// Create lab environment
	envID, err := cp.labEnvironmentManager.CreateEnvironment(ctx, lab, attempt.UserID)
	if err != nil {
		return nil, fmt.Errorf("failed to create lab environment: %w", err)
	}
	defer cp.labEnvironmentManager.DestroyEnvironment(ctx, envID)

	// Run validation tests
	validationResults := make([]ValidationResult, 0, len(lab.ValidationTests))
	totalPoints := 0
	earnedPoints := 0

	for _, test := range lab.ValidationTests {
		result, err := cp.labEnvironmentManager.ExecuteValidation(ctx, envID, test)
		if err != nil {
			result = &ValidationResult{
				TestID:       test.ID,
				Passed:       false,
				ActualOutput: fmt.Sprintf("Error: %v", err),
				PointsEarned: 0,
			}
		}
		validationResults = append(validationResults, *result)
		totalPoints += test.Points
		earnedPoints += result.PointsEarned
	}

	labResult := &LabResult{
		LabID:               labID,
		SubmissionTime:      time.Now(),
		ValidationResults:   validationResults,
		TotalPointsEarned:   earnedPoints,
		TotalPointsPossible: totalPoints,
		ManualReviewNeeded:  false,
	}

	cp.mu.Lock()
	defer cp.mu.Unlock()

	attempt.LabResults = append(attempt.LabResults, *labResult)
	attempt.UpdatedAt = time.Now()

	return labResult, nil
}

// CompleteExam finalizes an exam attempt
func (cp *CertificationPlatform) CompleteExam(ctx context.Context, attemptID string) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	attempt, exists := cp.examAttempts[attemptID]
	if !exists {
		return errors.New("exam attempt not found")
	}

	if attempt.Status != ExamStatusInProgress {
		return fmt.Errorf("exam cannot be completed from status: %s", attempt.Status)
	}

	exam, exists := cp.exams[attempt.ExamID]
	if !exists {
		return errors.New("exam not found")
	}

	attempt.Status = ExamStatusCompleted
	attempt.EndTime = time.Now()
	attempt.UpdatedAt = time.Now()

	// Calculate score
	totalPoints := 0
	earnedPoints := 0

	for _, answer := range attempt.Answers {
		earnedPoints += answer.PointsEarned
	}

	for _, question := range exam.QuestionBank {
		totalPoints += question.Points
	}

	for _, lab := range exam.PracticalLabs {
		totalPoints += lab.Points
	}

	for _, labResult := range attempt.LabResults {
		earnedPoints += labResult.TotalPointsEarned
	}

	if totalPoints > 0 {
		attempt.Score = (float64(earnedPoints) / float64(totalPoints)) * 100.0
	}

	// Determine pass/fail
	attempt.Passed = attempt.Score >= exam.PassingScore

	if attempt.Passed {
		attempt.Status = ExamStatusPassed
	} else {
		attempt.Status = ExamStatusFailed
	}

	// End proctoring
	if exam.ProctoringRequired && cp.proctoringService != nil {
		recordingURL, err := cp.proctoringService.EndProctoring(ctx, attemptID)
		if err == nil {
			attempt.ProctorRecordingURL = recordingURL

			// Monitor for flags
			flags, err := cp.proctoringService.MonitorExam(ctx, attemptID)
			if err == nil {
				attempt.ProctoringFlags = flags
			}
		}
	}

	// Send notification
	if cp.notificationService != nil {
		go cp.notificationService.NotifyExamResult(ctx, attempt.UserID, attempt)
	}

	// Record metrics
	if cp.metricsCollector != nil {
		go cp.metricsCollector.RecordExamAttempt(ctx, attempt)
	}

	return nil
}

// IssueCertificate issues a certificate after successful exam
func (cp *CertificationPlatform) IssueCertificate(ctx context.Context, userID string, level CertificationLevel, metadata CertificateMetadata) (*Certificate, error) {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	req, exists := cp.requirements[level]
	if !exists {
		return nil, errors.New("certification level not found")
	}

	// Validate requirements
	if metadata.StudyHours < req.MinimumStudyHours {
		return nil, fmt.Errorf("insufficient study hours: %d required, %d completed",
			req.MinimumStudyHours, metadata.StudyHours)
	}

	if metadata.ExamScore < req.MinimumExamScore {
		return nil, fmt.Errorf("insufficient exam score: %.2f required, %.2f achieved",
			req.MinimumExamScore, metadata.ExamScore)
	}

	if len(metadata.PracticalProjects) < req.PracticalProjects {
		return nil, fmt.Errorf("insufficient practical projects: %d required, %d completed",
			req.PracticalProjects, len(metadata.PracticalProjects))
	}

	// Check prerequisites
	for _, prereq := range req.PrerequisiteCerts {
		if !cp.hasValidCertification(userID, prereq) {
			return nil, fmt.Errorf("missing prerequisite certification: %s", prereq)
		}
	}

	now := time.Now()
	expiryDate := now.AddDate(req.ValidityYears, 0, 0)

	cert := &Certificate{
		ID:                uuid.New().String(),
		UserID:            userID,
		Level:             level,
		Status:            StatusActive,
		IssueDate:         now,
		ExpiryDate:        expiryDate,
		CertificateNumber: cp.generateCertificateNumber(level),
		Metadata:          metadata,
		RenewalHistory:    []RenewalRecord{},
		CreatedAt:         now,
		UpdatedAt:         now,
	}

	// Record on blockchain
	if cp.blockchainVerifier != nil {
		hash, err := cp.blockchainVerifier.RecordCertificate(ctx, cert)
		if err != nil {
			return nil, fmt.Errorf("failed to record certificate on blockchain: %w", err)
		}
		cert.BlockchainHash = hash
		cert.VerificationURL = fmt.Sprintf("https://verify.novacron.io/cert/%s", hash)
	}

	cp.certificates[cert.ID] = cert

	// Send notification
	if cp.notificationService != nil {
		go cp.notificationService.NotifyCertificateIssued(ctx, userID, cert)
	}

	// Record metrics
	if cp.metricsCollector != nil {
		go cp.metricsCollector.RecordCertificateIssued(ctx, cert)
	}

	return cert, nil
}

// generateCertificateNumber generates unique certificate number
func (cp *CertificationPlatform) generateCertificateNumber(level CertificationLevel) string {
	prefix := ""
	switch level {
	case LevelDeveloper:
		prefix = "NCD"
	case LevelArchitect:
		prefix = "NCA"
	case LevelExpert:
		prefix = "NCE"
	}

	timestamp := time.Now().Unix()
	random := uuid.New().String()[:8]

	return fmt.Sprintf("%s-%d-%s", prefix, timestamp, random)
}

// hasValidCertification checks if user has valid certification at level
func (cp *CertificationPlatform) hasValidCertification(userID string, level CertificationLevel) bool {
	for _, cert := range cp.certificates {
		if cert.UserID == userID && cert.Level == level && cert.Status == StatusActive {
			if time.Now().Before(cert.ExpiryDate) {
				return true
			}
		}
	}
	return false
}

// countUserAttempts counts exam attempts for user
func (cp *CertificationPlatform) countUserAttempts(userID, examID string) int {
	count := 0
	for _, attempt := range cp.examAttempts {
		if attempt.UserID == userID && attempt.ExamID == examID {
			count++
		}
	}
	return count
}

// getLastAttempt gets user's most recent exam attempt
func (cp *CertificationPlatform) getLastAttempt(userID, examID string) *ExamAttempt {
	var lastAttempt *ExamAttempt
	var lastTime time.Time

	for _, attempt := range cp.examAttempts {
		if attempt.UserID == userID && attempt.ExamID == examID {
			if attempt.EndTime.After(lastTime) {
				lastTime = attempt.EndTime
				lastAttempt = attempt
			}
		}
	}

	return lastAttempt
}

// RenewCertificate renews an existing certificate
func (cp *CertificationPlatform) RenewCertificate(ctx context.Context, certID string, ceuCompleted int, renewalExamScore float64) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	cert, exists := cp.certificates[certID]
	if !exists {
		return errors.New("certificate not found")
	}

	req, exists := cp.requirements[cert.Level]
	if !exists {
		return errors.New("certification requirements not found")
	}

	// Validate CEU requirements
	if ceuCompleted < req.RenewalCEURequired {
		return fmt.Errorf("insufficient CEU credits: %d required, %d completed",
			req.RenewalCEURequired, ceuCompleted)
	}

	// Validate renewal exam if required
	if renewalExamScore > 0 && renewalExamScore < req.MinimumExamScore {
		return fmt.Errorf("insufficient renewal exam score: %.2f required, %.2f achieved",
			req.MinimumExamScore, renewalExamScore)
	}

	now := time.Now()
	newExpiry := now.AddDate(req.ValidityYears, 0, 0)

	renewal := RenewalRecord{
		ID:               uuid.New().String(),
		RenewalDate:      now,
		CEUCompleted:     ceuCompleted,
		CEURequired:      req.RenewalCEURequired,
		PreviousExpiry:   cert.ExpiryDate,
		NewExpiry:        newExpiry,
		RenewalExamScore: renewalExamScore,
	}

	cert.RenewalHistory = append(cert.RenewalHistory, renewal)
	cert.ExpiryDate = newExpiry
	cert.Status = StatusActive
	cert.UpdatedAt = now

	return nil
}

// RevokeCertificate revokes a certificate
func (cp *CertificationPlatform) RevokeCertificate(ctx context.Context, certID, reason string) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	cert, exists := cp.certificates[certID]
	if !exists {
		return errors.New("certificate not found")
	}

	cert.Status = StatusRevoked
	cert.UpdatedAt = time.Now()

	// Record revocation on blockchain
	if cp.blockchainVerifier != nil {
		if err := cp.blockchainVerifier.RevokeCertificate(ctx, certID); err != nil {
			return fmt.Errorf("failed to revoke on blockchain: %w", err)
		}
	}

	return nil
}

// VerifyCertificate verifies certificate authenticity
func (cp *CertificationPlatform) VerifyCertificate(ctx context.Context, certID string) (*Certificate, error) {
	cp.mu.RLock()
	cert, exists := cp.certificates[certID]
	cp.mu.RUnlock()

	if !exists {
		return nil, errors.New("certificate not found")
	}

	// Verify on blockchain
	if cp.blockchainVerifier != nil && cert.BlockchainHash != "" {
		blockchainCert, err := cp.blockchainVerifier.VerifyCertificate(ctx, cert.BlockchainHash)
		if err != nil {
			return nil, fmt.Errorf("blockchain verification failed: %w", err)
		}

		// Verify cert data matches
		if blockchainCert.ID != cert.ID {
			return nil, errors.New("certificate data mismatch with blockchain record")
		}
	}

	return cert, nil
}

// UpdateStudyProgress updates learner's study progress
func (cp *CertificationPlatform) UpdateStudyProgress(ctx context.Context, progress *StudyProgress) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	progress.LastUpdated = time.Now()

	// Calculate readiness score
	req, _ := cp.requirements[progress.TargetLevel]
	readiness := 0.0

	// Study hours component (40%)
	hoursScore := float64(progress.StudyHoursLogged) / float64(req.MinimumStudyHours)
	if hoursScore > 1.0 {
		hoursScore = 1.0
	}
	readiness += hoursScore * 0.4

	// Module completion (30%)
	if progress.TotalModules > 0 {
		moduleScore := float64(len(progress.CompletedModules)) / float64(progress.TotalModules)
		readiness += moduleScore * 0.3
	}

	// Practice test scores (30%)
	if len(progress.PracticeTestScores) > 0 {
		avgScore := 0.0
		for _, score := range progress.PracticeTestScores {
			avgScore += score
		}
		avgScore /= float64(len(progress.PracticeTestScores))
		testScore := avgScore / 100.0
		readiness += testScore * 0.3
	}

	progress.ReadinessScore = readiness * 100.0

	// Recommend exam date when 80%+ ready
	if progress.ReadinessScore >= 80.0 && progress.RecommendedExamDate.IsZero() {
		progress.RecommendedExamDate = time.Now().Add(7 * 24 * time.Hour)
	}

	cp.studyProgress[progress.UserID] = progress

	// Record metrics
	if cp.metricsCollector != nil {
		go cp.metricsCollector.RecordStudyProgress(ctx, progress)
	}

	return nil
}

// RecordCEUActivity records continuing education activity
func (cp *CertificationPlatform) RecordCEUActivity(ctx context.Context, activity *ContinuingEducation) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if activity.ID == "" {
		activity.ID = uuid.New().String()
	}

	activity.CreatedAt = time.Now()
	activity.ApprovalStatus = "PENDING"

	cp.continuingEducation[activity.UserID] = append(
		cp.continuingEducation[activity.UserID],
		activity,
	)

	// Record metrics
	if cp.metricsCollector != nil {
		go cp.metricsCollector.RecordCEUActivity(ctx, activity)
	}

	return nil
}

// GetUserCertificates retrieves all certificates for a user
func (cp *CertificationPlatform) GetUserCertificates(userID string) []*Certificate {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	certs := make([]*Certificate, 0)
	for _, cert := range cp.certificates {
		if cert.UserID == userID {
			certs = append(certs, cert)
		}
	}

	return certs
}

// GetPlatformStatistics returns platform-wide statistics
func (cp *CertificationPlatform) GetPlatformStatistics() map[string]interface{} {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	stats := make(map[string]interface{})

	// Certificate stats
	certCounts := make(map[CertificationLevel]int)
	statusCounts := make(map[CertificationStatus]int)

	for _, cert := range cp.certificates {
		certCounts[cert.Level]++
		statusCounts[cert.Status]++
	}

	stats["total_certificates"] = len(cp.certificates)
	stats["certificates_by_level"] = certCounts
	stats["certificates_by_status"] = statusCounts

	// Exam stats
	examStats := make(map[ExamStatus]int)
	totalAttempts := 0
	passedAttempts := 0

	for _, attempt := range cp.examAttempts {
		examStats[attempt.Status]++
		totalAttempts++
		if attempt.Passed {
			passedAttempts++
		}
	}

	stats["total_exam_attempts"] = totalAttempts
	stats["exam_attempts_by_status"] = examStats

	if totalAttempts > 0 {
		stats["exam_pass_rate"] = float64(passedAttempts) / float64(totalAttempts) * 100.0
	}

	// Study progress stats
	stats["active_learners"] = len(cp.studyProgress)

	// CEU stats
	totalCEU := 0
	for _, activities := range cp.continuingEducation {
		for _, activity := range activities {
			if activity.ApprovalStatus == "APPROVED" {
				totalCEU += activity.CEUCredits
			}
		}
	}
	stats["total_ceu_credits"] = totalCEU

	return stats
}

// ComputeBlockchainHash computes hash for certificate
func ComputeBlockchainHash(cert *Certificate) string {
	data := fmt.Sprintf("%s:%s:%s:%s:%d",
		cert.ID,
		cert.UserID,
		cert.Level,
		cert.CertificateNumber,
		cert.IssueDate.Unix(),
	)

	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

// ExportCertificate exports certificate to JSON
func (cp *CertificationPlatform) ExportCertificate(certID string) (string, error) {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	cert, exists := cp.certificates[certID]
	if !exists {
		return "", errors.New("certificate not found")
	}

	data, err := json.MarshalIndent(cert, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal certificate: %w", err)
	}

	return string(data), nil
}

// CheckExpiringCertificates identifies certificates expiring soon
func (cp *CertificationPlatform) CheckExpiringCertificates(ctx context.Context, daysThreshold int) []*Certificate {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	expiringSoon := make([]*Certificate, 0)
	threshold := time.Now().Add(time.Duration(daysThreshold) * 24 * time.Hour)

	for _, cert := range cp.certificates {
		if cert.Status == StatusActive && cert.ExpiryDate.Before(threshold) {
			expiringSoon = append(expiringSoon, cert)

			// Send notification
			if cp.notificationService != nil {
				go cp.notificationService.NotifyCertificateExpiring(ctx, cert.UserID, cert)
			}
		}
	}

	return expiringSoon
}
