// Package learning provides comprehensive developer learning management system
// Implements interactive tutorials, video courses, and hands-on labs
package learning

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/google/uuid"
)

// CourseLevel represents course difficulty level
type CourseLevel string

const (
	LevelBeginner     CourseLevel = "BEGINNER"
	LevelIntermediate CourseLevel = "INTERMEDIATE"
	LevelAdvanced     CourseLevel = "ADVANCED"
	LevelExpert       CourseLevel = "EXPERT"
)

// ContentType represents type of learning content
type ContentType string

const (
	ContentTypeVideo       ContentType = "VIDEO"
	ContentTypeArticle     ContentType = "ARTICLE"
	ContentTypeInteractive ContentType = "INTERACTIVE"
	ContentTypeQuiz        ContentType = "QUIZ"
	ContentTypeLab         ContentType = "LAB"
	ContentTypeAssignment  ContentType = "ASSIGNMENT"
)

// EnrollmentStatus represents learner enrollment status
type EnrollmentStatus string

const (
	EnrollmentActive    EnrollmentStatus = "ACTIVE"
	EnrollmentCompleted EnrollmentStatus = "COMPLETED"
	EnrollmentDropped   EnrollmentStatus = "DROPPED"
	EnrollmentSuspended EnrollmentStatus = "SUSPENDED"
)

// Course represents a complete learning course
type Course struct {
	ID               string           `json:"id"`
	Title            string           `json:"title"`
	Description      string           `json:"description"`
	Level            CourseLevel      `json:"level"`
	DurationHours    int              `json:"duration_hours"`
	Modules          []Module         `json:"modules"`
	Prerequisites    []string         `json:"prerequisites"`
	LearningPath     string           `json:"learning_path"`
	Instructors      []Instructor     `json:"instructors"`
	Tags             []string         `json:"tags"`
	CertificationID  string           `json:"certification_id,omitempty"`
	ThumbnailURL     string           `json:"thumbnail_url"`
	IntroVideoURL    string           `json:"intro_video_url"`
	EstimatedCEU     int              `json:"estimated_ceu"`
	IsPublished      bool             `json:"is_published"`
	EnrollmentCount  int              `json:"enrollment_count"`
	AverageRating    float64          `json:"average_rating"`
	CompletionRate   float64          `json:"completion_rate"`
	CreatedAt        time.Time        `json:"created_at"`
	UpdatedAt        time.Time        `json:"updated_at"`
}

// Module represents a course module
type Module struct {
	ID               string          `json:"id"`
	CourseID         string          `json:"course_id"`
	Title            string          `json:"title"`
	Description      string          `json:"description"`
	OrderIndex       int             `json:"order_index"`
	EstimatedMinutes int             `json:"estimated_minutes"`
	Lessons          []Lesson        `json:"lessons"`
	Quiz             *Quiz           `json:"quiz,omitempty"`
	Assignments      []Assignment    `json:"assignments"`
	Resources        []Resource      `json:"resources"`
	UnlocksAt        time.Time       `json:"unlocks_at,omitempty"`
	CreatedAt        time.Time       `json:"created_at"`
	UpdatedAt        time.Time       `json:"updated_at"`
}

// Lesson represents individual lesson content
type Lesson struct {
	ID               string      `json:"id"`
	ModuleID         string      `json:"module_id"`
	Title            string      `json:"title"`
	ContentType      ContentType `json:"content_type"`
	OrderIndex       int         `json:"order_index"`
	EstimatedMinutes int         `json:"estimated_minutes"`
	Content          interface{} `json:"content"`
	CompletionCriteria CompletionCriteria `json:"completion_criteria"`
	CreatedAt        time.Time   `json:"created_at"`
	UpdatedAt        time.Time   `json:"updated_at"`
}

// VideoContent represents video lesson content
type VideoContent struct {
	VideoURL      string              `json:"video_url"`
	Duration      int                 `json:"duration_seconds"`
	Transcript    string              `json:"transcript"`
	Subtitles     map[string]string   `json:"subtitles"` // language -> subtitle URL
	Chapters      []VideoChapter      `json:"chapters"`
	StreamQuality []string            `json:"stream_quality"`
}

// VideoChapter represents chapter in video
type VideoChapter struct {
	Title     string `json:"title"`
	StartTime int    `json:"start_time_seconds"`
	EndTime   int    `json:"end_time_seconds"`
}

// ArticleContent represents article lesson content
type ArticleContent struct {
	Body             string            `json:"body"`
	Format           string            `json:"format"` // markdown, html, etc
	EstimatedReadMin int               `json:"estimated_read_minutes"`
	CodeSnippets     []CodeSnippet     `json:"code_snippets"`
	Images           []ImageReference  `json:"images"`
	RelatedLinks     []string          `json:"related_links"`
}

// CodeSnippet represents code example
type CodeSnippet struct {
	ID          string `json:"id"`
	Language    string `json:"language"`
	Code        string `json:"code"`
	Description string `json:"description"`
	Runnable    bool   `json:"runnable"`
	TestCases   []TestCase `json:"test_cases,omitempty"`
}

// TestCase represents test for runnable code
type TestCase struct {
	Input    string `json:"input"`
	Expected string `json:"expected"`
}

// ImageReference represents image in content
type ImageReference struct {
	URL         string `json:"url"`
	Alt         string `json:"alt"`
	Caption     string `json:"caption"`
	Attribution string `json:"attribution"`
}

// InteractiveContent represents hands-on interactive lesson
type InteractiveContent struct {
	ExerciseType    string              `json:"exercise_type"`
	Instructions    string              `json:"instructions"`
	InitialCode     string              `json:"initial_code"`
	Solution        string              `json:"solution"`
	Hints           []string            `json:"hints"`
	ValidationTests []ValidationTest    `json:"validation_tests"`
	SandboxConfig   SandboxConfiguration `json:"sandbox_config"`
}

// ValidationTest represents automated validation
type ValidationTest struct {
	ID          string `json:"id"`
	Description string `json:"description"`
	TestCode    string `json:"test_code"`
	Points      int    `json:"points"`
}

// SandboxConfiguration defines sandbox environment
type SandboxConfiguration struct {
	Language       string            `json:"language"`
	Version        string            `json:"version"`
	AllowedPackages []string         `json:"allowed_packages"`
	MemoryLimitMB  int               `json:"memory_limit_mb"`
	TimeoutSeconds int               `json:"timeout_seconds"`
	Environment    map[string]string `json:"environment"`
}

// Quiz represents module quiz
type Quiz struct {
	ID               string     `json:"id"`
	ModuleID         string     `json:"module_id"`
	Title            string     `json:"title"`
	Description      string     `json:"description"`
	Questions        []Question `json:"questions"`
	PassingScore     float64    `json:"passing_score"`
	TimeLimit        int        `json:"time_limit_minutes"`
	MaxAttempts      int        `json:"max_attempts"`
	RandomizeOrder   bool       `json:"randomize_order"`
	ShowCorrectAnswers bool     `json:"show_correct_answers"`
	CreatedAt        time.Time  `json:"created_at"`
	UpdatedAt        time.Time  `json:"updated_at"`
}

// Question represents quiz question
type Question struct {
	ID             string      `json:"id"`
	Type           string      `json:"type"`
	Question       string      `json:"question"`
	Options        []string    `json:"options,omitempty"`
	CorrectAnswer  interface{} `json:"correct_answer"`
	Explanation    string      `json:"explanation"`
	Points         int         `json:"points"`
	Difficulty     string      `json:"difficulty"`
	Tags           []string    `json:"tags"`
}

// Assignment represents practical assignment
type Assignment struct {
	ID               string              `json:"id"`
	ModuleID         string              `json:"module_id"`
	Title            string              `json:"title"`
	Description      string              `json:"description"`
	Instructions     string              `json:"instructions"`
	DueDate          time.Time           `json:"due_date,omitempty"`
	Points           int                 `json:"points"`
	SubmissionType   string              `json:"submission_type"`
	RubricCriteria   []RubricCriterion   `json:"rubric_criteria"`
	PeerReviewCount  int                 `json:"peer_review_count"`
	RequiresPeerReview bool              `json:"requires_peer_review"`
	CreatedAt        time.Time           `json:"created_at"`
	UpdatedAt        time.Time           `json:"updated_at"`
}

// RubricCriterion represents grading criteria
type RubricCriterion struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	MaxPoints   int    `json:"max_points"`
	Weight      float64 `json:"weight"`
}

// Resource represents learning resource
type Resource struct {
	ID          string      `json:"id"`
	Title       string      `json:"title"`
	Type        string      `json:"type"`
	URL         string      `json:"url"`
	Description string      `json:"description"`
	FileSize    int64       `json:"file_size,omitempty"`
	CreatedAt   time.Time   `json:"created_at"`
}

// CompletionCriteria defines lesson completion requirements
type CompletionCriteria struct {
	RequiresVideo       bool    `json:"requires_video"`
	MinimumVideoPercent float64 `json:"minimum_video_percent"`
	RequiresQuiz        bool    `json:"requires_quiz"`
	MinimumQuizScore    float64 `json:"minimum_quiz_score"`
	RequiresInteractive bool    `json:"requires_interactive"`
	ManualApproval      bool    `json:"manual_approval"`
}

// Instructor represents course instructor
type Instructor struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Title       string   `json:"title"`
	Bio         string   `json:"bio"`
	AvatarURL   string   `json:"avatar_url"`
	LinkedInURL string   `json:"linkedin_url"`
	TwitterURL  string   `json:"twitter_url"`
	Expertise   []string `json:"expertise"`
}

// Enrollment represents user course enrollment
type Enrollment struct {
	ID                string           `json:"id"`
	UserID            string           `json:"user_id"`
	CourseID          string           `json:"course_id"`
	Status            EnrollmentStatus `json:"status"`
	EnrolledAt        time.Time        `json:"enrolled_at"`
	CompletedAt       time.Time        `json:"completed_at,omitempty"`
	LastAccessedAt    time.Time        `json:"last_accessed_at"`
	Progress          EnrollmentProgress `json:"progress"`
	CertificateIssued bool             `json:"certificate_issued"`
	Rating            float64          `json:"rating,omitempty"`
	Review            string           `json:"review,omitempty"`
	CreatedAt         time.Time        `json:"created_at"`
	UpdatedAt         time.Time        `json:"updated_at"`
}

// EnrollmentProgress tracks learner progress
type EnrollmentProgress struct {
	CompletedModules  []string          `json:"completed_modules"`
	CompletedLessons  []string          `json:"completed_lessons"`
	CompletedQuizzes  []string          `json:"completed_quizzes"`
	QuizScores        map[string]float64 `json:"quiz_scores"`
	AssignmentScores  map[string]float64 `json:"assignment_scores"`
	TotalTimeSpent    int               `json:"total_time_spent_minutes"`
	CurrentModule     string            `json:"current_module"`
	CurrentLesson     string            `json:"current_lesson"`
	PercentComplete   float64           `json:"percent_complete"`
	AchievementBadges []string          `json:"achievement_badges"`
	StudyStreak       int               `json:"study_streak_days"`
	LastStudyDate     time.Time         `json:"last_study_date"`
}

// LessonProgress tracks individual lesson progress
type LessonProgress struct {
	ID                string    `json:"id"`
	EnrollmentID      string    `json:"enrollment_id"`
	LessonID          string    `json:"lesson_id"`
	StartedAt         time.Time `json:"started_at"`
	CompletedAt       time.Time `json:"completed_at,omitempty"`
	TimeSpentMinutes  int       `json:"time_spent_minutes"`
	PercentComplete   float64   `json:"percent_complete"`
	VideoPosition     int       `json:"video_position_seconds,omitempty"`
	InteractiveState  string    `json:"interactive_state,omitempty"`
	LastAccessedAt    time.Time `json:"last_accessed_at"`
}

// QuizAttempt tracks quiz attempts
type QuizAttempt struct {
	ID           string              `json:"id"`
	EnrollmentID string              `json:"enrollment_id"`
	QuizID       string              `json:"quiz_id"`
	AttemptNumber int                `json:"attempt_number"`
	StartedAt    time.Time           `json:"started_at"`
	CompletedAt  time.Time           `json:"completed_at,omitempty"`
	Answers      map[string]interface{} `json:"answers"`
	Score        float64             `json:"score"`
	Passed       bool                `json:"passed"`
	TimeSpent    int                 `json:"time_spent_minutes"`
	CreatedAt    time.Time           `json:"created_at"`
}

// AssignmentSubmission represents assignment submission
type AssignmentSubmission struct {
	ID             string              `json:"id"`
	EnrollmentID   string              `json:"enrollment_id"`
	AssignmentID   string              `json:"assignment_id"`
	SubmittedAt    time.Time           `json:"submitted_at"`
	Content        string              `json:"content"`
	Attachments    []string            `json:"attachments"`
	Score          float64             `json:"score,omitempty"`
	Feedback       string              `json:"feedback,omitempty"`
	GradedAt       time.Time           `json:"graded_at,omitempty"`
	GradedBy       string              `json:"graded_by,omitempty"`
	RubricScores   map[string]int      `json:"rubric_scores"`
	PeerReviews    []PeerReview        `json:"peer_reviews"`
	Status         string              `json:"status"`
	CreatedAt      time.Time           `json:"created_at"`
	UpdatedAt      time.Time           `json:"updated_at"`
}

// PeerReview represents peer review of submission
type PeerReview struct {
	ID            string         `json:"id"`
	ReviewerID    string         `json:"reviewer_id"`
	RubricScores  map[string]int `json:"rubric_scores"`
	Comments      string         `json:"comments"`
	SubmittedAt   time.Time      `json:"submitted_at"`
}

// Discussion represents forum discussion
type Discussion struct {
	ID           string         `json:"id"`
	CourseID     string         `json:"course_id"`
	ModuleID     string         `json:"module_id,omitempty"`
	LessonID     string         `json:"lesson_id,omitempty"`
	AuthorID     string         `json:"author_id"`
	Title        string         `json:"title"`
	Content      string         `json:"content"`
	Tags         []string       `json:"tags"`
	Replies      []Reply        `json:"replies"`
	Upvotes      int            `json:"upvotes"`
	Views        int            `json:"views"`
	IsPinned     bool           `json:"is_pinned"`
	IsResolved   bool           `json:"is_resolved"`
	CreatedAt    time.Time      `json:"created_at"`
	UpdatedAt    time.Time      `json:"updated_at"`
}

// Reply represents discussion reply
type Reply struct {
	ID            string    `json:"id"`
	DiscussionID  string    `json:"discussion_id"`
	AuthorID      string    `json:"author_id"`
	Content       string    `json:"content"`
	ParentReplyID string    `json:"parent_reply_id,omitempty"`
	Upvotes       int       `json:"upvotes"`
	IsAnswer      bool      `json:"is_answer"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// LiveSession represents live coding workshop
type LiveSession struct {
	ID              string       `json:"id"`
	CourseID        string       `json:"course_id"`
	Title           string       `json:"title"`
	Description     string       `json:"description"`
	InstructorID    string       `json:"instructor_id"`
	ScheduledAt     time.Time    `json:"scheduled_at"`
	DurationMinutes int          `json:"duration_minutes"`
	MeetingURL      string       `json:"meeting_url"`
	RecordingURL    string       `json:"recording_url,omitempty"`
	Agenda          []string     `json:"agenda"`
	MaxParticipants int          `json:"max_participants"`
	Participants    []string     `json:"participants"`
	Status          string       `json:"status"`
	CreatedAt       time.Time    `json:"created_at"`
	UpdatedAt       time.Time    `json:"updated_at"`
}

// LearningPlatform manages the learning management system
type LearningPlatform struct {
	mu                  sync.RWMutex
	courses             map[string]*Course
	enrollments         map[string]*Enrollment
	lessonProgress      map[string][]*LessonProgress
	quizAttempts        map[string][]*QuizAttempt
	assignmentSubmissions map[string][]*AssignmentSubmission
	discussions         map[string]*Discussion
	liveSessions        map[string]*LiveSession
	videoStreamService  VideoStreamService
	sandboxManager      SandboxManager
	notificationService NotificationService
	analyticsCollector  AnalyticsCollector
	gamificationEngine  GamificationEngine
}

// VideoStreamService handles video streaming
type VideoStreamService interface {
	StreamVideo(ctx context.Context, videoURL string, quality string) (string, error)
	TrackProgress(ctx context.Context, userID, videoURL string, position int) error
	GenerateSubtitles(ctx context.Context, videoURL, language string) (string, error)
}

// SandboxManager manages interactive code sandboxes
type SandboxManager interface {
	CreateSandbox(ctx context.Context, config SandboxConfiguration) (string, error)
	ExecuteCode(ctx context.Context, sandboxID, code string) (string, error)
	ValidateSubmission(ctx context.Context, sandboxID string, tests []ValidationTest) ([]ValidationResult, error)
	DestroySandbox(ctx context.Context, sandboxID string) error
}

// ValidationResult represents validation test result
type ValidationResult struct {
	TestID       string `json:"test_id"`
	Passed       bool   `json:"passed"`
	Output       string `json:"output"`
	ErrorMessage string `json:"error_message,omitempty"`
	PointsEarned int    `json:"points_earned"`
}

// NotificationService handles learner notifications
type NotificationService interface {
	NotifyEnrollment(ctx context.Context, userID, courseID string) error
	NotifyModuleUnlock(ctx context.Context, userID, moduleID string) error
	NotifyAssignmentDue(ctx context.Context, userID, assignmentID string) error
	NotifyLiveSessionStarting(ctx context.Context, userID, sessionID string) error
	NotifyCourseCompletion(ctx context.Context, userID, courseID string) error
}

// AnalyticsCollector collects learning analytics
type AnalyticsCollector interface {
	RecordEnrollment(ctx context.Context, enrollment *Enrollment)
	RecordLessonProgress(ctx context.Context, progress *LessonProgress)
	RecordQuizAttempt(ctx context.Context, attempt *QuizAttempt)
	RecordTimeSpent(ctx context.Context, userID, courseID string, minutes int)
}

// GamificationEngine handles achievements and gamification
type GamificationEngine interface {
	AwardBadge(ctx context.Context, userID, badgeID string) error
	UpdateStreak(ctx context.Context, userID string) (int, error)
	CalculatePoints(ctx context.Context, userID string) (int, error)
	CheckAchievements(ctx context.Context, userID string) ([]string, error)
}

// NewLearningPlatform creates new learning platform instance
func NewLearningPlatform(
	videoStreamService VideoStreamService,
	sandboxManager SandboxManager,
	notificationService NotificationService,
	analyticsCollector AnalyticsCollector,
	gamificationEngine GamificationEngine,
) *LearningPlatform {
	return &LearningPlatform{
		courses:               make(map[string]*Course),
		enrollments:           make(map[string]*Enrollment),
		lessonProgress:        make(map[string][]*LessonProgress),
		quizAttempts:          make(map[string][]*QuizAttempt),
		assignmentSubmissions: make(map[string][]*AssignmentSubmission),
		discussions:           make(map[string]*Discussion),
		liveSessions:          make(map[string]*LiveSession),
		videoStreamService:    videoStreamService,
		sandboxManager:        sandboxManager,
		notificationService:   notificationService,
		analyticsCollector:    analyticsCollector,
		gamificationEngine:    gamificationEngine,
	}
}

// CreateCourse creates a new course
func (lp *LearningPlatform) CreateCourse(ctx context.Context, course *Course) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	if course.ID == "" {
		course.ID = uuid.New().String()
	}

	course.CreatedAt = time.Now()
	course.UpdatedAt = time.Now()

	lp.courses[course.ID] = course

	return nil
}

// EnrollUser enrolls a user in a course
func (lp *LearningPlatform) EnrollUser(ctx context.Context, userID, courseID string) (*Enrollment, error) {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	course, exists := lp.courses[courseID]
	if !exists {
		return nil, errors.New("course not found")
	}

	// Check prerequisites
	if len(course.Prerequisites) > 0 {
		for _, prereqID := range course.Prerequisites {
			if !lp.hasCompletedCourse(userID, prereqID) {
				return nil, fmt.Errorf("prerequisite course %s not completed", prereqID)
			}
		}
	}

	enrollment := &Enrollment{
		ID:             uuid.New().String(),
		UserID:         userID,
		CourseID:       courseID,
		Status:         EnrollmentActive,
		EnrolledAt:     time.Now(),
		LastAccessedAt: time.Now(),
		Progress: EnrollmentProgress{
			CompletedModules:  []string{},
			CompletedLessons:  []string{},
			CompletedQuizzes:  []string{},
			QuizScores:        make(map[string]float64),
			AssignmentScores:  make(map[string]float64),
			AchievementBadges: []string{},
			PercentComplete:   0,
		},
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	lp.enrollments[enrollment.ID] = enrollment

	course.EnrollmentCount++

	// Send notification
	if lp.notificationService != nil {
		go lp.notificationService.NotifyEnrollment(ctx, userID, courseID)
	}

	// Record analytics
	if lp.analyticsCollector != nil {
		go lp.analyticsCollector.RecordEnrollment(ctx, enrollment)
	}

	return enrollment, nil
}

// hasCompletedCourse checks if user has completed a course
func (lp *LearningPlatform) hasCompletedCourse(userID, courseID string) bool {
	for _, enrollment := range lp.enrollments {
		if enrollment.UserID == userID && enrollment.CourseID == courseID {
			return enrollment.Status == EnrollmentCompleted
		}
	}
	return false
}

// StartLesson starts a lesson for a user
func (lp *LearningPlatform) StartLesson(ctx context.Context, enrollmentID, lessonID string) (*LessonProgress, error) {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	enrollment, exists := lp.enrollments[enrollmentID]
	if !exists {
		return nil, errors.New("enrollment not found")
	}

	if enrollment.Status != EnrollmentActive {
		return nil, errors.New("enrollment is not active")
	}

	progress := &LessonProgress{
		ID:              uuid.New().String(),
		EnrollmentID:    enrollmentID,
		LessonID:        lessonID,
		StartedAt:       time.Now(),
		LastAccessedAt:  time.Now(),
		PercentComplete: 0,
	}

	lp.lessonProgress[enrollmentID] = append(lp.lessonProgress[enrollmentID], progress)

	enrollment.LastAccessedAt = time.Now()
	enrollment.Progress.CurrentLesson = lessonID

	// Update study streak
	if lp.gamificationEngine != nil {
		go lp.gamificationEngine.UpdateStreak(ctx, enrollment.UserID)
	}

	// Record analytics
	if lp.analyticsCollector != nil {
		go lp.analyticsCollector.RecordLessonProgress(ctx, progress)
	}

	return progress, nil
}

// CompleteLesson marks a lesson as complete
func (lp *LearningPlatform) CompleteLesson(ctx context.Context, enrollmentID, lessonID string) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	enrollment, exists := lp.enrollments[enrollmentID]
	if !exists {
		return errors.New("enrollment not found")
	}

	// Find lesson progress
	var progress *LessonProgress
	for _, lp := range lp.lessonProgress[enrollmentID] {
		if lp.LessonID == lessonID {
			progress = lp
			break
		}
	}

	if progress == nil {
		return errors.New("lesson not started")
	}

	progress.CompletedAt = time.Now()
	progress.PercentComplete = 100

	// Update enrollment progress
	if !contains(enrollment.Progress.CompletedLessons, lessonID) {
		enrollment.Progress.CompletedLessons = append(enrollment.Progress.CompletedLessons, lessonID)
	}

	// Calculate overall progress
	course := lp.courses[enrollment.CourseID]
	totalLessons := lp.countTotalLessons(course)
	if totalLessons > 0 {
		enrollment.Progress.PercentComplete = float64(len(enrollment.Progress.CompletedLessons)) / float64(totalLessons) * 100
	}

	enrollment.UpdatedAt = time.Now()

	// Check achievements
	if lp.gamificationEngine != nil {
		go lp.gamificationEngine.CheckAchievements(ctx, enrollment.UserID)
	}

	return nil
}

// countTotalLessons counts total lessons in course
func (lp *LearningPlatform) countTotalLessons(course *Course) int {
	count := 0
	for _, module := range course.Modules {
		count += len(module.Lessons)
	}
	return count
}

// StartQuizAttempt starts a quiz attempt
func (lp *LearningPlatform) StartQuizAttempt(ctx context.Context, enrollmentID, quizID string) (*QuizAttempt, error) {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	enrollment, exists := lp.enrollments[enrollmentID]
	if !exists {
		return nil, errors.New("enrollment not found")
	}

	// Count previous attempts
	attempts := lp.quizAttempts[enrollmentID]
	attemptCount := 0
	for _, attempt := range attempts {
		if attempt.QuizID == quizID {
			attemptCount++
		}
	}

	// Check max attempts (if quiz has limit)
	// For now, allow unlimited attempts

	attempt := &QuizAttempt{
		ID:            uuid.New().String(),
		EnrollmentID:  enrollmentID,
		QuizID:        quizID,
		AttemptNumber: attemptCount + 1,
		StartedAt:     time.Now(),
		Answers:       make(map[string]interface{}),
		CreatedAt:     time.Now(),
	}

	lp.quizAttempts[enrollmentID] = append(lp.quizAttempts[enrollmentID], attempt)

	enrollment.LastAccessedAt = time.Now()

	return attempt, nil
}

// SubmitQuiz submits quiz answers
func (lp *LearningPlatform) SubmitQuiz(ctx context.Context, attemptID string, answers map[string]interface{}) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	var attempt *QuizAttempt
	var enrollment *Enrollment

	// Find attempt
	for enrollmentID, attempts := range lp.quizAttempts {
		for _, a := range attempts {
			if a.ID == attemptID {
				attempt = a
				enrollment = lp.enrollments[enrollmentID]
				break
			}
		}
		if attempt != nil {
			break
		}
	}

	if attempt == nil {
		return errors.New("quiz attempt not found")
	}

	// Find quiz
	var quiz *Quiz
	course := lp.courses[enrollment.CourseID]
	for _, module := range course.Modules {
		if module.Quiz != nil && module.Quiz.ID == attempt.QuizID {
			quiz = module.Quiz
			break
		}
	}

	if quiz == nil {
		return errors.New("quiz not found")
	}

	attempt.Answers = answers
	attempt.CompletedAt = time.Now()
	attempt.TimeSpent = int(time.Since(attempt.StartedAt).Minutes())

	// Grade quiz
	totalPoints := 0
	earnedPoints := 0

	for _, question := range quiz.Questions {
		totalPoints += question.Points
		if answer, exists := answers[question.ID]; exists {
			if lp.isAnswerCorrect(question, answer) {
				earnedPoints += question.Points
			}
		}
	}

	if totalPoints > 0 {
		attempt.Score = float64(earnedPoints) / float64(totalPoints) * 100
	}

	attempt.Passed = attempt.Score >= quiz.PassingScore

	// Update enrollment progress
	enrollment.Progress.QuizScores[quiz.ID] = attempt.Score

	if attempt.Passed && !contains(enrollment.Progress.CompletedQuizzes, quiz.ID) {
		enrollment.Progress.CompletedQuizzes = append(enrollment.Progress.CompletedQuizzes, quiz.ID)
	}

	enrollment.UpdatedAt = time.Now()

	// Record analytics
	if lp.analyticsCollector != nil {
		go lp.analyticsCollector.RecordQuizAttempt(ctx, attempt)
	}

	return nil
}

// isAnswerCorrect checks if answer is correct
func (lp *LearningPlatform) isAnswerCorrect(question Question, providedAnswer interface{}) bool {
	// Similar logic to certification platform
	switch question.Type {
	case "MULTIPLE_CHOICE", "TRUE_FALSE":
		return providedAnswer == question.CorrectAnswer
	case "MULTIPLE_ANSWER":
		provided, ok1 := providedAnswer.([]string)
		correct, ok2 := question.CorrectAnswer.([]string)
		if !ok1 || !ok2 || len(provided) != len(correct) {
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
	default:
		return false
	}
}

// SubmitAssignment submits an assignment
func (lp *LearningPlatform) SubmitAssignment(ctx context.Context, enrollmentID, assignmentID, content string, attachments []string) (*AssignmentSubmission, error) {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	enrollment, exists := lp.enrollments[enrollmentID]
	if !exists {
		return nil, errors.New("enrollment not found")
	}

	submission := &AssignmentSubmission{
		ID:           uuid.New().String(),
		EnrollmentID: enrollmentID,
		AssignmentID: assignmentID,
		SubmittedAt:  time.Now(),
		Content:      content,
		Attachments:  attachments,
		RubricScores: make(map[string]int),
		PeerReviews:  []PeerReview{},
		Status:       "SUBMITTED",
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}

	lp.assignmentSubmissions[enrollmentID] = append(lp.assignmentSubmissions[enrollmentID], submission)

	enrollment.LastAccessedAt = time.Now()
	enrollment.UpdatedAt = time.Now()

	return submission, nil
}

// CreateDiscussion creates a new discussion
func (lp *LearningPlatform) CreateDiscussion(ctx context.Context, discussion *Discussion) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	if discussion.ID == "" {
		discussion.ID = uuid.New().String()
	}

	discussion.CreatedAt = time.Now()
	discussion.UpdatedAt = time.Now()
	discussion.Replies = []Reply{}

	lp.discussions[discussion.ID] = discussion

	return nil
}

// AddReply adds a reply to discussion
func (lp *LearningPlatform) AddReply(ctx context.Context, discussionID, authorID, content string) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	discussion, exists := lp.discussions[discussionID]
	if !exists {
		return errors.New("discussion not found")
	}

	reply := Reply{
		ID:           uuid.New().String(),
		DiscussionID: discussionID,
		AuthorID:     authorID,
		Content:      content,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
	}

	discussion.Replies = append(discussion.Replies, reply)
	discussion.UpdatedAt = time.Now()

	return nil
}

// ScheduleLiveSession schedules a live coding workshop
func (lp *LearningPlatform) ScheduleLiveSession(ctx context.Context, session *LiveSession) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	if session.ID == "" {
		session.ID = uuid.New().String()
	}

	session.Status = "SCHEDULED"
	session.Participants = []string{}
	session.CreatedAt = time.Now()
	session.UpdatedAt = time.Now()

	lp.liveSessions[session.ID] = session

	return nil
}

// RegisterForLiveSession registers user for live session
func (lp *LearningPlatform) RegisterForLiveSession(ctx context.Context, sessionID, userID string) error {
	lp.mu.Lock()
	defer lp.mu.Unlock()

	session, exists := lp.liveSessions[sessionID]
	if !exists {
		return errors.New("live session not found")
	}

	if session.MaxParticipants > 0 && len(session.Participants) >= session.MaxParticipants {
		return errors.New("session is full")
	}

	if !contains(session.Participants, userID) {
		session.Participants = append(session.Participants, userID)
	}

	session.UpdatedAt = time.Now()

	// Send notification
	if lp.notificationService != nil {
		go lp.notificationService.NotifyLiveSessionStarting(ctx, userID, sessionID)
	}

	return nil
}

// GetCourseByID retrieves course by ID
func (lp *LearningPlatform) GetCourseByID(courseID string) (*Course, error) {
	lp.mu.RLock()
	defer lp.mu.RUnlock()

	course, exists := lp.courses[courseID]
	if !exists {
		return nil, errors.New("course not found")
	}

	return course, nil
}

// GetUserEnrollments retrieves all enrollments for user
func (lp *LearningPlatform) GetUserEnrollments(userID string) []*Enrollment {
	lp.mu.RLock()
	defer lp.mu.RUnlock()

	enrollments := make([]*Enrollment, 0)
	for _, enrollment := range lp.enrollments {
		if enrollment.UserID == userID {
			enrollments = append(enrollments, enrollment)
		}
	}

	return enrollments
}

// GetPlatformStatistics returns platform statistics
func (lp *LearningPlatform) GetPlatformStatistics() map[string]interface{} {
	lp.mu.RLock()
	defer lp.mu.RUnlock()

	stats := make(map[string]interface{})

	stats["total_courses"] = len(lp.courses)
	stats["total_enrollments"] = len(lp.enrollments)
	stats["total_discussions"] = len(lp.discussions)
	stats["total_live_sessions"] = len(lp.liveSessions)

	activeEnrollments := 0
	completedEnrollments := 0

	for _, enrollment := range lp.enrollments {
		if enrollment.Status == EnrollmentActive {
			activeEnrollments++
		} else if enrollment.Status == EnrollmentCompleted {
			completedEnrollments++
		}
	}

	stats["active_enrollments"] = activeEnrollments
	stats["completed_enrollments"] = completedEnrollments

	if len(lp.enrollments) > 0 {
		stats["completion_rate"] = float64(completedEnrollments) / float64(len(lp.enrollments)) * 100
	}

	return stats
}

// contains checks if slice contains string
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}
