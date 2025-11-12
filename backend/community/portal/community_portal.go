// Package portal provides comprehensive community engagement platform
// Implements Q&A forums, project showcases, and community features
package portal

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/google/uuid"
)

// UserRole represents user role in community
type UserRole string

const (
	RoleMember      UserRole = "MEMBER"
	RoleModerator   UserRole = "MODERATOR"
	RoleExpert      UserRole = "EXPERT"
	RoleAmbassador  UserRole = "AMBASSADOR"
	RoleAdministrator UserRole = "ADMINISTRATOR"
)

// ReputationTier represents reputation level
type ReputationTier string

const (
	TierNovice      ReputationTier = "NOVICE"
	TierIntermediate ReputationTier = "INTERMEDIATE"
	TierAdvanced    ReputationTier = "ADVANCED"
	TierExpert      ReputationTier = "EXPERT"
	TierMaster      ReputationTier = "MASTER"
)

// ContentType represents type of community content
type ContentType string

const (
	ContentTypeQuestion  ContentType = "QUESTION"
	ContentTypeArticle   ContentType = "ARTICLE"
	ContentTypeShowcase  ContentType = "SHOWCASE"
	ContentTypeEvent     ContentType = "EVENT"
)

// ContentStatus represents content moderation status
type ContentStatus string

const (
	StatusPublished ContentStatus = "PUBLISHED"
	StatusDraft     ContentStatus = "DRAFT"
	StatusPending   ContentStatus = "PENDING"
	StatusFlagged   ContentStatus = "FLAGGED"
	StatusRemoved   ContentStatus = "REMOVED"
)

// UserProfile represents community member profile
type UserProfile struct {
	ID                string          `json:"id"`
	Username          string          `json:"username"`
	DisplayName       string          `json:"display_name"`
	Email             string          `json:"email"`
	AvatarURL         string          `json:"avatar_url"`
	Bio               string          `json:"bio"`
	Location          string          `json:"location"`
	Company           string          `json:"company"`
	Website           string          `json:"website"`
	GitHubUsername    string          `json:"github_username"`
	TwitterUsername   string          `json:"twitter_username"`
	LinkedInURL       string          `json:"linkedin_url"`
	Role              UserRole        `json:"role"`
	ReputationPoints  int             `json:"reputation_points"`
	ReputationTier    ReputationTier  `json:"reputation_tier"`
	Badges            []Badge         `json:"badges"`
	Certifications    []string        `json:"certifications"`
	Specializations   []string        `json:"specializations"`
	ContributionStats ContributionStats `json:"contribution_stats"`
	JoinedAt          time.Time       `json:"joined_at"`
	LastSeenAt        time.Time       `json:"last_seen_at"`
	CreatedAt         time.Time       `json:"created_at"`
	UpdatedAt         time.Time       `json:"updated_at"`
}

// Badge represents achievement badge
type Badge struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	IconURL     string    `json:"icon_url"`
	EarnedAt    time.Time `json:"earned_at"`
	Rarity      string    `json:"rarity"`
}

// ContributionStats tracks user contributions
type ContributionStats struct {
	QuestionsAsked       int            `json:"questions_asked"`
	AnswersProvided      int            `json:"answers_provided"`
	AcceptedAnswers      int            `json:"accepted_answers"`
	ArticlesPublished    int            `json:"articles_published"`
	ProjectsShowcased    int            `json:"projects_showcased"`
	HelpfulVotes         int            `json:"helpful_votes"`
	CommentsPosted       int            `json:"comments_posted"`
	EventsAttended       int            `json:"events_attended"`
	EventsOrganized      int            `json:"events_organized"`
	MentoringSessions    int            `json:"mentoring_sessions"`
	CodeReviewsProvided  int            `json:"code_reviews_provided"`
	ContributionStreak   int            `json:"contribution_streak_days"`
	LastContributionDate time.Time      `json:"last_contribution_date"`
}

// Question represents Q&A forum question
type Question struct {
	ID              string        `json:"id"`
	AuthorID        string        `json:"author_id"`
	Title           string        `json:"title"`
	Body            string        `json:"body"`
	Tags            []string      `json:"tags"`
	Category        string        `json:"category"`
	Status          ContentStatus `json:"status"`
	Answers         []Answer      `json:"answers"`
	AcceptedAnswerID string       `json:"accepted_answer_id,omitempty"`
	Views           int           `json:"views"`
	Upvotes         int           `json:"upvotes"`
	Downvotes       int           `json:"downvotes"`
	IsFeatured      bool          `json:"is_featured"`
	IsPinned        bool          `json:"is_pinned"`
	IsClosed        bool          `json:"is_closed"`
	ClosedReason    string        `json:"closed_reason,omitempty"`
	Bounty          int           `json:"bounty,omitempty"`
	CreatedAt       time.Time     `json:"created_at"`
	UpdatedAt       time.Time     `json:"updated_at"`
}

// Answer represents answer to question
type Answer struct {
	ID               string    `json:"id"`
	QuestionID       string    `json:"question_id"`
	AuthorID         string    `json:"author_id"`
	Body             string    `json:"body"`
	CodeSnippets     []CodeSnippet `json:"code_snippets"`
	IsAccepted       bool      `json:"is_accepted"`
	Upvotes          int       `json:"upvotes"`
	Downvotes        int       `json:"downvotes"`
	Comments         []Comment `json:"comments"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

// CodeSnippet represents code in answer
type CodeSnippet struct {
	ID          string `json:"id"`
	Language    string `json:"language"`
	Code        string `json:"code"`
	Description string `json:"description"`
	Filename    string `json:"filename,omitempty"`
}

// Comment represents comment on answer
type Comment struct {
	ID        string    `json:"id"`
	AuthorID  string    `json:"author_id"`
	Content   string    `json:"content"`
	Upvotes   int       `json:"upvotes"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// Article represents community blog article
type Article struct {
	ID             string        `json:"id"`
	AuthorID       string        `json:"author_id"`
	Title          string        `json:"title"`
	Subtitle       string        `json:"subtitle"`
	Body           string        `json:"body"`
	CoverImageURL  string        `json:"cover_image_url"`
	Tags           []string      `json:"tags"`
	Category       string        `json:"category"`
	Status         ContentStatus `json:"status"`
	Views          int           `json:"views"`
	ReadTimeMinutes int          `json:"read_time_minutes"`
	Likes          int           `json:"likes"`
	Comments       []Comment     `json:"comments"`
	IsFeatured     bool          `json:"is_featured"`
	PublishedAt    time.Time     `json:"published_at,omitempty"`
	CreatedAt      time.Time     `json:"created_at"`
	UpdatedAt      time.Time     `json:"updated_at"`
}

// ProjectShowcase represents showcased project
type ProjectShowcase struct {
	ID             string        `json:"id"`
	AuthorID       string        `json:"author_id"`
	Title          string        `json:"title"`
	Description    string        `json:"description"`
	LongDescription string       `json:"long_description"`
	ThumbnailURL   string        `json:"thumbnail_url"`
	Screenshots    []string      `json:"screenshots"`
	VideoURL       string        `json:"video_url,omitempty"`
	GitHubURL      string        `json:"github_url"`
	LiveDemoURL    string        `json:"live_demo_url,omitempty"`
	Technologies   []string      `json:"technologies"`
	Tags           []string      `json:"tags"`
	Category       string        `json:"category"`
	Status         ContentStatus `json:"status"`
	Views          int           `json:"views"`
	Stars          int           `json:"stars"`
	Comments       []Comment     `json:"comments"`
	IsFeatured     bool          `json:"is_featured"`
	IsWinner       bool          `json:"is_winner"`
	WinnerOf       string        `json:"winner_of,omitempty"`
	CreatedAt      time.Time     `json:"created_at"`
	UpdatedAt      time.Time     `json:"updated_at"`
}

// Event represents community event
type Event struct {
	ID              string        `json:"id"`
	OrganizerID     string        `json:"organizer_id"`
	Title           string        `json:"title"`
	Description     string        `json:"description"`
	EventType       string        `json:"event_type"`
	Format          string        `json:"format"` // online, in-person, hybrid
	Location        string        `json:"location,omitempty"`
	VirtualURL      string        `json:"virtual_url,omitempty"`
	StartTime       time.Time     `json:"start_time"`
	EndTime         time.Time     `json:"end_time"`
	Timezone        string        `json:"timezone"`
	MaxAttendees    int           `json:"max_attendees"`
	RegisteredUsers []string      `json:"registered_users"`
	WaitlistUsers   []string      `json:"waitlist_users"`
	Tags            []string      `json:"tags"`
	Status          ContentStatus `json:"status"`
	IsFeatured      bool          `json:"is_featured"`
	RecordingURL    string        `json:"recording_url,omitempty"`
	Materials       []Resource    `json:"materials"`
	CreatedAt       time.Time     `json:"created_at"`
	UpdatedAt       time.Time     `json:"updated_at"`
}

// Resource represents event resource
type Resource struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Type        string `json:"type"`
	URL         string `json:"url"`
	Description string `json:"description"`
}

// JobPosting represents job opportunity
type JobPosting struct {
	ID              string    `json:"id"`
	CompanyID       string    `json:"company_id"`
	CompanyName     string    `json:"company_name"`
	CompanyLogoURL  string    `json:"company_logo_url"`
	Title           string    `json:"title"`
	Description     string    `json:"description"`
	Requirements    []string  `json:"requirements"`
	Responsibilities []string `json:"responsibilities"`
	Location        string    `json:"location"`
	RemotePolicy    string    `json:"remote_policy"`
	SalaryRange     string    `json:"salary_range"`
	EmploymentType  string    `json:"employment_type"`
	ExperienceLevel string    `json:"experience_level"`
	Skills          []string  `json:"skills"`
	ApplyURL        string    `json:"apply_url"`
	Views           int       `json:"views"`
	Applications    int       `json:"applications"`
	IsActive        bool      `json:"is_active"`
	ExpiresAt       time.Time `json:"expires_at"`
	CreatedAt       time.Time `json:"created_at"`
	UpdatedAt       time.Time `json:"updated_at"`
}

// Vote represents user vote on content
type Vote struct {
	ID        string    `json:"id"`
	UserID    string    `json:"user_id"`
	ContentID string    `json:"content_id"`
	ContentType ContentType `json:"content_type"`
	IsUpvote  bool      `json:"is_upvote"`
	CreatedAt time.Time `json:"created_at"`
}

// ModerationAction represents moderation activity
type ModerationAction struct {
	ID          string    `json:"id"`
	ModeratorID string    `json:"moderator_id"`
	ContentID   string    `json:"content_id"`
	ContentType ContentType `json:"content_type"`
	ActionType  string    `json:"action_type"`
	Reason      string    `json:"reason"`
	Notes       string    `json:"notes"`
	CreatedAt   time.Time `json:"created_at"`
}

// CommunityPortal manages the community platform
type CommunityPortal struct {
	mu                  sync.RWMutex
	users               map[string]*UserProfile
	questions           map[string]*Question
	answers             map[string]*Answer
	articles            map[string]*Article
	projects            map[string]*ProjectShowcase
	events              map[string]*Event
	jobs                map[string]*JobPosting
	votes               map[string]*Vote
	moderationActions   []*ModerationAction
	reputationEngine    ReputationEngine
	notificationService NotificationService
	searchService       SearchService
	analyticsCollector  AnalyticsCollector
}

// ReputationEngine calculates and manages reputation
type ReputationEngine interface {
	CalculateReputation(ctx context.Context, userID string) (int, error)
	AwardPoints(ctx context.Context, userID string, points int, reason string) error
	DeductPoints(ctx context.Context, userID string, points int, reason string) error
	GetTier(points int) ReputationTier
	AwardBadge(ctx context.Context, userID, badgeID string) error
}

// NotificationService handles user notifications
type NotificationService interface {
	NotifyNewAnswer(ctx context.Context, questionID, answererID string) error
	NotifyAcceptedAnswer(ctx context.Context, answerID string) error
	NotifyMention(ctx context.Context, mentionedUserID, contentID string) error
	NotifyEventReminder(ctx context.Context, eventID string, attendees []string) error
}

// SearchService provides search functionality
type SearchService interface {
	IndexContent(ctx context.Context, contentType ContentType, contentID string, content interface{}) error
	Search(ctx context.Context, query string, filters map[string]interface{}) ([]SearchResult, error)
	SearchUsers(ctx context.Context, query string) ([]*UserProfile, error)
}

// SearchResult represents search result
type SearchResult struct {
	ContentType ContentType `json:"content_type"`
	ContentID   string      `json:"content_id"`
	Title       string      `json:"title"`
	Snippet     string      `json:"snippet"`
	Relevance   float64     `json:"relevance"`
	URL         string      `json:"url"`
}

// AnalyticsCollector collects community analytics
type AnalyticsCollector interface {
	RecordView(ctx context.Context, contentType ContentType, contentID, userID string)
	RecordInteraction(ctx context.Context, interactionType, contentID, userID string)
	RecordSearch(ctx context.Context, query string, resultsCount int)
}

// NewCommunityPortal creates new community portal instance
func NewCommunityPortal(
	reputationEngine ReputationEngine,
	notificationService NotificationService,
	searchService SearchService,
	analyticsCollector AnalyticsCollector,
) *CommunityPortal {
	return &CommunityPortal{
		users:               make(map[string]*UserProfile),
		questions:           make(map[string]*Question),
		answers:             make(map[string]*Answer),
		articles:            make(map[string]*Article),
		projects:            make(map[string]*ProjectShowcase),
		events:              make(map[string]*Event),
		jobs:                make(map[string]*JobPosting),
		votes:               make(map[string]*Vote),
		moderationActions:   []*ModerationAction{},
		reputationEngine:    reputationEngine,
		notificationService: notificationService,
		searchService:       searchService,
		analyticsCollector:  analyticsCollector,
	}
}

// CreateUser creates new community user profile
func (cp *CommunityPortal) CreateUser(ctx context.Context, profile *UserProfile) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if profile.ID == "" {
		profile.ID = uuid.New().String()
	}

	profile.Role = RoleMember
	profile.ReputationPoints = 0
	profile.ReputationTier = TierNovice
	profile.Badges = []Badge{}
	profile.JoinedAt = time.Now()
	profile.LastSeenAt = time.Now()
	profile.CreatedAt = time.Now()
	profile.UpdatedAt = time.Now()

	cp.users[profile.ID] = profile

	return nil
}

// AskQuestion creates new question
func (cp *CommunityPortal) AskQuestion(ctx context.Context, question *Question) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if question.ID == "" {
		question.ID = uuid.New().String()
	}

	question.Status = StatusPublished
	question.Answers = []Answer{}
	question.Views = 0
	question.Upvotes = 0
	question.Downvotes = 0
	question.CreatedAt = time.Now()
	question.UpdatedAt = time.Now()

	cp.questions[question.ID] = question

	// Index for search
	if cp.searchService != nil {
		go cp.searchService.IndexContent(ctx, ContentTypeQuestion, question.ID, question)
	}

	// Award reputation points
	if cp.reputationEngine != nil {
		go cp.reputationEngine.AwardPoints(ctx, question.AuthorID, 5, "question_posted")
	}

	return nil
}

// AnswerQuestion creates answer to question
func (cp *CommunityPortal) AnswerQuestion(ctx context.Context, answer *Answer) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	question, exists := cp.questions[answer.QuestionID]
	if !exists {
		return errors.New("question not found")
	}

	if question.IsClosed {
		return errors.New("question is closed for answers")
	}

	if answer.ID == "" {
		answer.ID = uuid.New().String()
	}

	answer.CreatedAt = time.Now()
	answer.UpdatedAt = time.Now()
	answer.Comments = []Comment{}

	question.Answers = append(question.Answers, *answer)
	question.UpdatedAt = time.Now()

	cp.answers[answer.ID] = answer

	// Send notification
	if cp.notificationService != nil {
		go cp.notificationService.NotifyNewAnswer(ctx, answer.QuestionID, answer.AuthorID)
	}

	// Award reputation points
	if cp.reputationEngine != nil {
		go cp.reputationEngine.AwardPoints(ctx, answer.AuthorID, 10, "answer_posted")
	}

	return nil
}

// AcceptAnswer marks answer as accepted
func (cp *CommunityPortal) AcceptAnswer(ctx context.Context, questionID, answerID, userID string) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	question, exists := cp.questions[questionID]
	if !exists {
		return errors.New("question not found")
	}

	if question.AuthorID != userID {
		return errors.New("only question author can accept answers")
	}

	answer, exists := cp.answers[answerID]
	if !exists {
		return errors.New("answer not found")
	}

	// Update question
	question.AcceptedAnswerID = answerID
	question.UpdatedAt = time.Now()

	// Update answer
	answer.IsAccepted = true
	answer.UpdatedAt = time.Now()

	// Update answer in question's answer list
	for i := range question.Answers {
		if question.Answers[i].ID == answerID {
			question.Answers[i].IsAccepted = true
			break
		}
	}

	// Send notification
	if cp.notificationService != nil {
		go cp.notificationService.NotifyAcceptedAnswer(ctx, answerID)
	}

	// Award reputation points
	if cp.reputationEngine != nil {
		go cp.reputationEngine.AwardPoints(ctx, answer.AuthorID, 15, "answer_accepted")
	}

	return nil
}

// VoteContent votes on content (question, answer, etc)
func (cp *CommunityPortal) VoteContent(ctx context.Context, userID, contentID string, contentType ContentType, isUpvote bool) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	// Check if user already voted
	voteKey := fmt.Sprintf("%s:%s", userID, contentID)
	if existingVote, exists := cp.votes[voteKey]; exists {
		// Update existing vote
		if existingVote.IsUpvote != isUpvote {
			cp.updateVoteCounts(contentID, contentType, existingVote.IsUpvote, false)
			cp.updateVoteCounts(contentID, contentType, isUpvote, true)
			existingVote.IsUpvote = isUpvote
		}
		return nil
	}

	// Create new vote
	vote := &Vote{
		ID:          uuid.New().String(),
		UserID:      userID,
		ContentID:   contentID,
		ContentType: contentType,
		IsUpvote:    isUpvote,
		CreatedAt:   time.Now(),
	}

	cp.votes[voteKey] = vote

	// Update vote counts
	cp.updateVoteCounts(contentID, contentType, isUpvote, true)

	// Award/deduct reputation
	if cp.reputationEngine != nil && contentType == ContentTypeQuestion {
		question := cp.questions[contentID]
		if question != nil {
			points := 2
			if !isUpvote {
				points = -1
			}
			go cp.reputationEngine.AwardPoints(ctx, question.AuthorID, points, "vote_received")
		}
	}

	return nil
}

// updateVoteCounts updates vote counts for content
func (cp *CommunityPortal) updateVoteCounts(contentID string, contentType ContentType, isUpvote, increment bool) {
	delta := 1
	if !increment {
		delta = -1
	}

	switch contentType {
	case ContentTypeQuestion:
		if question, exists := cp.questions[contentID]; exists {
			if isUpvote {
				question.Upvotes += delta
			} else {
				question.Downvotes += delta
			}
		}
	}
}

// PublishArticle publishes community article
func (cp *CommunityPortal) PublishArticle(ctx context.Context, article *Article) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if article.ID == "" {
		article.ID = uuid.New().String()
	}

	article.Status = StatusPublished
	article.PublishedAt = time.Now()
	article.Comments = []Comment{}
	article.CreatedAt = time.Now()
	article.UpdatedAt = time.Now()

	cp.articles[article.ID] = article

	// Index for search
	if cp.searchService != nil {
		go cp.searchService.IndexContent(ctx, ContentTypeArticle, article.ID, article)
	}

	// Award reputation points
	if cp.reputationEngine != nil {
		go cp.reputationEngine.AwardPoints(ctx, article.AuthorID, 20, "article_published")
	}

	return nil
}

// ShowcaseProject showcases community project
func (cp *CommunityPortal) ShowcaseProject(ctx context.Context, project *ProjectShowcase) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if project.ID == "" {
		project.ID = uuid.New().String()
	}

	project.Status = StatusPublished
	project.Comments = []Comment{}
	project.CreatedAt = time.Now()
	project.UpdatedAt = time.Now()

	cp.projects[project.ID] = project

	// Index for search
	if cp.searchService != nil {
		go cp.searchService.IndexContent(ctx, ContentTypeShowcase, project.ID, project)
	}

	// Award reputation points
	if cp.reputationEngine != nil {
		go cp.reputationEngine.AwardPoints(ctx, project.AuthorID, 25, "project_showcased")
	}

	return nil
}

// CreateEvent creates community event
func (cp *CommunityPortal) CreateEvent(ctx context.Context, event *Event) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if event.ID == "" {
		event.ID = uuid.New().String()
	}

	event.Status = StatusPublished
	event.RegisteredUsers = []string{}
	event.WaitlistUsers = []string{}
	event.CreatedAt = time.Now()
	event.UpdatedAt = time.Now()

	cp.events[event.ID] = event

	// Index for search
	if cp.searchService != nil {
		go cp.searchService.IndexContent(ctx, ContentTypeEvent, event.ID, event)
	}

	return nil
}

// RegisterForEvent registers user for event
func (cp *CommunityPortal) RegisterForEvent(ctx context.Context, eventID, userID string) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	event, exists := cp.events[eventID]
	if !exists {
		return errors.New("event not found")
	}

	// Check if already registered
	for _, uid := range event.RegisteredUsers {
		if uid == userID {
			return errors.New("already registered")
		}
	}

	// Check capacity
	if event.MaxAttendees > 0 && len(event.RegisteredUsers) >= event.MaxAttendees {
		// Add to waitlist
		event.WaitlistUsers = append(event.WaitlistUsers, userID)
		return nil
	}

	event.RegisteredUsers = append(event.RegisteredUsers, userID)
	event.UpdatedAt = time.Now()

	return nil
}

// PostJobOpportunity posts job opening
func (cp *CommunityPortal) PostJobOpportunity(ctx context.Context, job *JobPosting) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if job.ID == "" {
		job.ID = uuid.New().String()
	}

	job.IsActive = true
	job.Views = 0
	job.Applications = 0
	job.CreatedAt = time.Now()
	job.UpdatedAt = time.Now()

	cp.jobs[job.ID] = job

	return nil
}

// GetTrendingQuestions returns trending questions
func (cp *CommunityPortal) GetTrendingQuestions(limit int) []*Question {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	questions := make([]*Question, 0, len(cp.questions))
	for _, q := range cp.questions {
		if q.Status == StatusPublished {
			questions = append(questions, q)
		}
	}

	// Sort by score (upvotes - downvotes) and recency
	sort.Slice(questions, func(i, j int) bool {
		scoreI := questions[i].Upvotes - questions[i].Downvotes
		scoreJ := questions[j].Upvotes - questions[j].Downvotes
		if scoreI == scoreJ {
			return questions[i].CreatedAt.After(questions[j].CreatedAt)
		}
		return scoreI > scoreJ
	})

	if len(questions) > limit {
		questions = questions[:limit]
	}

	return questions
}

// GetFeaturedContent returns featured content
func (cp *CommunityPortal) GetFeaturedContent() map[string]interface{} {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	featured := make(map[string]interface{})

	// Featured articles
	featuredArticles := make([]*Article, 0)
	for _, article := range cp.articles {
		if article.IsFeatured && article.Status == StatusPublished {
			featuredArticles = append(featuredArticles, article)
		}
	}
	featured["articles"] = featuredArticles

	// Featured projects
	featuredProjects := make([]*ProjectShowcase, 0)
	for _, project := range cp.projects {
		if project.IsFeatured && project.Status == StatusPublished {
			featuredProjects = append(featuredProjects, project)
		}
	}
	featured["projects"] = featuredProjects

	// Featured events
	upcomingEvents := make([]*Event, 0)
	now := time.Now()
	for _, event := range cp.events {
		if event.IsFeatured && event.StartTime.After(now) {
			upcomingEvents = append(upcomingEvents, event)
		}
	}
	featured["events"] = upcomingEvents

	return featured
}

// GetUserProfile retrieves user profile
func (cp *CommunityPortal) GetUserProfile(userID string) (*UserProfile, error) {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	profile, exists := cp.users[userID]
	if !exists {
		return nil, errors.New("user not found")
	}

	return profile, nil
}

// GetPlatformStatistics returns platform statistics
func (cp *CommunityPortal) GetPlatformStatistics() map[string]interface{} {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	stats := make(map[string]interface{})

	stats["total_users"] = len(cp.users)
	stats["total_questions"] = len(cp.questions)
	stats["total_answers"] = len(cp.answers)
	stats["total_articles"] = len(cp.articles)
	stats["total_projects"] = len(cp.projects)
	stats["total_events"] = len(cp.events)
	stats["total_jobs"] = len(cp.jobs)

	// Count answered questions
	answeredCount := 0
	for _, q := range cp.questions {
		if len(q.Answers) > 0 {
			answeredCount++
		}
	}
	stats["answered_questions"] = answeredCount

	// Count accepted answers
	acceptedCount := 0
	for _, q := range cp.questions {
		if q.AcceptedAnswerID != "" {
			acceptedCount++
		}
	}
	stats["accepted_answers"] = acceptedCount

	return stats
}
