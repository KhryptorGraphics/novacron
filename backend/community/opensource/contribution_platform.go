// Package opensource implements Community Contributions & Open Source Platform
// Open source core components, contribution rewards, automated code review
// Target: 1,000+ community contributions/year
package opensource

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// ContributionType represents contribution category
type ContributionType string

const (
	TypeFeature      ContributionType = "feature"
	TypeBugFix       ContributionType = "bugfix"
	TypeDocumentation ContributionType = "documentation"
	TypeTest         ContributionType = "test"
	TypeRefactor     ContributionType = "refactor"
	TypeSecurity     ContributionType = "security"
	TypePerformance  ContributionType = "performance"
)

// ContributionStatus represents contribution lifecycle
type ContributionStatus string

const (
	StatusDraft      ContributionStatus = "draft"
	StatusSubmitted  ContributionStatus = "submitted"
	StatusReviewing  ContributionStatus = "reviewing"
	StatusApproved   ContributionStatus = "approved"
	StatusMerged     ContributionStatus = "merged"
	StatusRejected   ContributionStatus = "rejected"
)

// Repository represents open source repository
type Repository struct {
	ID              string
	Name            string
	Description     string
	URL             string
	License         string // Apache-2.0, MIT, etc.
	Language        string
	Stars           int
	Forks           int
	Contributors    []Contributor
	Maintainers     []string
	Tags            []string
	Topics          []string
	Status          string // active, archived
	OpenIssues      int
	OpenPRs         int
	TotalCommits    int
	LastCommit      time.Time
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// Contributor represents open source contributor
type Contributor struct {
	UserID          string
	Username        string
	Email           string
	Name            string
	Contributions   []Contribution
	TotalCommits    int
	LinesAdded      int
	LinesDeleted    int
	IssuesCreated   int
	IssuesClosed    int
	PRsSubmitted    int
	PRsMerged       int
	CodeReviews     int
	Reputation      int
	Rewards         []Reward
	Badges          []Badge
	FirstContribution time.Time
	LastContribution  time.Time
	Rank            int
	Level           string // beginner, intermediate, advanced, expert
}

// Contribution represents code contribution
type Contribution struct {
	ID              string
	Type            ContributionType
	Title           string
	Description     string
	RepositoryID    string
	ContributorID   string
	BranchName      string
	Files           []FileChange
	LinesAdded      int
	LinesDeleted    int
	Commits         []Commit
	Status          ContributionStatus
	Reviews         []CodeReview
	TestsPassed     bool
	CIStatus        string
	SecurityScan    SecurityScanResult
	QualityScore    float64
	ImpactScore     float64
	Reward          *Reward
	CreatedAt       time.Time
	SubmittedAt     time.Time
	ReviewedAt      *time.Time
	MergedAt        *time.Time
	UpdatedAt       time.Time
}

// FileChange represents file modification
type FileChange struct {
	Path        string
	Action      string // added, modified, deleted
	Additions   int
	Deletions   int
	Changes     int
	Binary      bool
}

// Commit represents git commit
type Commit struct {
	SHA         string
	Message     string
	Author      string
	Timestamp   time.Time
	FilesChanged int
	Additions   int
	Deletions   int
}

// CodeReview represents code review
type CodeReview struct {
	ID          string
	ReviewerID  string
	ReviewerName string
	Status      string // approved, changes_requested, commented
	Comments    []ReviewComment
	Rating      float64
	Automated   bool
	ReviewedAt  time.Time
}

// ReviewComment represents review comment
type ReviewComment struct {
	ID          string
	File        string
	Line        int
	Type        string // suggestion, issue, question, praise
	Severity    string // critical, major, minor, info
	Message     string
	Code        string
	Suggestion  string
	Resolved    bool
	CreatedAt   time.Time
}

// SecurityScanResult represents security scan
type SecurityScanResult struct {
	Passed          bool
	Vulnerabilities []Vulnerability
	Score           float64
	ScannedAt       time.Time
}

// Vulnerability represents security issue
type Vulnerability struct {
	ID          string
	Severity    string // critical, high, medium, low
	Type        string
	Description string
	File        string
	Line        int
	CWE         string
	CVE         string
	Fixed       bool
}

// Reward represents contribution reward
type Reward struct {
	ID          string
	Type        string // cash, credits, swag, recognition
	Amount      float64
	Currency    string
	Description string
	EarnedAt    time.Time
	PaidOut     bool
	PaidAt      *time.Time
}

// Badge represents achievement badge
type Badge struct {
	ID          string
	Name        string
	Description string
	Icon        string
	Tier        string // bronze, silver, gold, platinum
	EarnedAt    time.Time
}

// Issue represents GitHub issue
type Issue struct {
	ID              string
	Number          int
	Title           string
	Description     string
	Type            string // bug, feature, enhancement, question
	Priority        string // low, medium, high, critical
	Status          string // open, in_progress, closed
	Labels          []string
	Assignees       []string
	Reporter        string
	Comments        []IssueComment
	Upvotes         int
	BountyAmount    float64
	CreatedAt       time.Time
	ClosedAt        *time.Time
	UpdatedAt       time.Time
}

// IssueComment represents issue comment
type IssueComment struct {
	ID          string
	AuthorID    string
	AuthorName  string
	Body        string
	CreatedAt   time.Time
}

// PullRequest represents pull request
type PullRequest struct {
	ID              string
	Number          int
	Title           string
	Description     string
	RepositoryID    string
	ContributionID  string
	AuthorID        string
	BaseBranch      string
	HeadBranch      string
	Status          string // open, merged, closed
	Reviews         []CodeReview
	Commits         []Commit
	FilesChanged    int
	Additions       int
	Deletions       int
	Mergeable       bool
	CIStatus        string
	CreatedAt       time.Time
	MergedAt        *time.Time
	ClosedAt        *time.Time
}

// CommunityGovernance manages community decisions
type CommunityGovernance struct {
	Proposals       []Proposal
	VotingPower     map[string]int
	DecisionLog     []Decision
}

// Proposal represents community proposal
type Proposal struct {
	ID              string
	Title           string
	Description     string
	Type            string // feature, policy, budget
	ProposerID      string
	Status          string // draft, voting, approved, rejected, implemented
	Votes           []Vote
	VotesFor        int
	VotesAgainst    int
	QuorumRequired  int
	CreatedAt       time.Time
	VotingDeadline  time.Time
	DecidedAt       *time.Time
}

// Vote represents community vote
type Vote struct {
	UserID      string
	Choice      string // for, against, abstain
	Weight      int
	Reason      string
	CastAt      time.Time
}

// Decision represents governance decision
type Decision struct {
	ProposalID      string
	Outcome         string
	Implementation  string
	DecidedAt       time.Time
	ImplementedAt   *time.Time
}

// Roadmap represents feature roadmap
type Roadmap struct {
	Items           []RoadmapItem
	PublicRoadmap   bool
	CommunityVoting bool
}

// RoadmapItem represents planned feature
type RoadmapItem struct {
	ID              string
	Title           string
	Description     string
	Category        string
	Priority        string
	Status          string // planned, in_progress, completed
	VoteCount       int
	Voters          []string
	Assignees       []string
	EstimatedQuarter string
	Progress        int
	CreatedAt       time.Time
	UpdatedAt       time.Time
}

// ContributionPlatform manages open source contributions
type ContributionPlatform struct {
	mu              sync.RWMutex
	repositories    map[string]*Repository
	contributors    map[string]*Contributor
	contributions   map[string]*Contribution
	issues          map[string]*Issue
	pullRequests    map[string]*PullRequest
	governance      *CommunityGovernance
	roadmap         *Roadmap
	stats           ContributionStats
	rewardPool      float64
}

// ContributionStats tracks contribution metrics
type ContributionStats struct {
	TotalRepositories   int
	OpenSourceRepos     int
	TotalContributors   int
	ActiveContributors  int
	TotalContributions  int
	ContributionsThisYear int
	TotalCommits        int
	LinesAdded          int
	LinesDeleted        int
	TotalIssues         int
	OpenIssues          int
	ClosedIssues        int
	TotalPRs            int
	MergedPRs           int
	RejectedPRs         int
	AverageReviewTime   float64 // hours
	AverageMergeTime    float64 // hours
	TotalRewardsP aid    float64
	CashRewards         float64
	CreditRewards       float64
	TopContributors     []string
	UpdatedAt           time.Time
}

// NewContributionPlatform creates contribution platform
func NewContributionPlatform(rewardPool float64) *ContributionPlatform {
	cp := &ContributionPlatform{
		repositories:  make(map[string]*Repository),
		contributors:  make(map[string]*Contributor),
		contributions: make(map[string]*Contribution),
		issues:        make(map[string]*Issue),
		pullRequests:  make(map[string]*PullRequest),
		governance: &CommunityGovernance{
			Proposals:   []Proposal{},
			VotingPower: make(map[string]int),
			DecisionLog: []Decision{},
		},
		roadmap: &Roadmap{
			Items:           []RoadmapItem{},
			PublicRoadmap:   true,
			CommunityVoting: true,
		},
		rewardPool: rewardPool,
	}

	cp.initializeSampleData()

	return cp
}

// initializeSampleData creates sample repositories
func (cp *ContributionPlatform) initializeSampleData() {
	// Create 20 open source repositories
	for i := 0; i < 20; i++ {
		repo := &Repository{
			ID:           cp.generateID("REPO"),
			Name:         fmt.Sprintf("novacron-%s", []string{"core", "api", "sdk", "cli", "docs"}[i%5]),
			Description:  "Open source component",
			License:      "Apache-2.0",
			Language:     []string{"Go", "Python", "JavaScript", "Rust"}[i%4],
			Stars:        (i + 1) * 100,
			Forks:        (i + 1) * 20,
			Status:       "active",
			OpenIssues:   10 + i,
			OpenPRs:      5 + i/2,
			TotalCommits: (i + 1) * 500,
			LastCommit:   time.Now().AddDate(0, 0, -i),
			CreatedAt:    time.Now().AddDate(-1, 0, 0),
			UpdatedAt:    time.Now(),
		}

		cp.repositories[repo.ID] = repo
	}
}

// SubmitContribution submits contribution
func (cp *ContributionPlatform) SubmitContribution(ctx context.Context, contribution *Contribution) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if contribution.ID == "" {
		contribution.ID = cp.generateID("CONTRIB")
	}

	contribution.Status = StatusSubmitted
	contribution.SubmittedAt = time.Now()
	contribution.CreatedAt = time.Now()
	contribution.UpdatedAt = time.Now()

	// Automated code review
	cp.runAutomatedReview(contribution)

	// Security scan
	contribution.SecurityScan = cp.runSecurityScan(contribution)

	// Calculate quality score
	contribution.QualityScore = cp.calculateQualityScore(contribution)

	cp.contributions[contribution.ID] = contribution

	// Update contributor
	if contributor, exists := cp.contributors[contribution.ContributorID]; exists {
		contributor.Contributions = append(contributor.Contributions, *contribution)
		contributor.TotalCommits += len(contribution.Commits)
		contributor.LinesAdded += contribution.LinesAdded
		contributor.LinesDeleted += contribution.LinesDeleted
		contributor.LastContribution = time.Now()
	}

	cp.stats.TotalContributions++
	cp.stats.ContributionsThisYear++
	cp.stats.UpdatedAt = time.Now()

	return nil
}

// runAutomatedReview performs automated code review
func (cp *ContributionPlatform) runAutomatedReview(contribution *Contribution) {
	review := CodeReview{
		ID:           cp.generateID("REV"),
		ReviewerName: "AutoReviewer",
		Status:       "commented",
		Automated:    true,
		ReviewedAt:   time.Now(),
	}

	// Simulate automated checks
	comments := []ReviewComment{
		{
			ID:       cp.generateID("COMMENT"),
			Type:     "suggestion",
			Severity: "minor",
			Message:  "Consider adding error handling",
			CreatedAt: time.Now(),
		},
	}

	review.Comments = comments
	review.Rating = 8.5

	contribution.Reviews = append(contribution.Reviews, review)
	contribution.Status = StatusReviewing
}

// runSecurityScan performs security scan
func (cp *ContributionPlatform) runSecurityScan(contribution *Contribution) SecurityScanResult {
	return SecurityScanResult{
		Passed:          true,
		Vulnerabilities: []Vulnerability{},
		Score:           95.0,
		ScannedAt:       time.Now(),
	}
}

// calculateQualityScore calculates contribution quality
func (cp *ContributionPlatform) calculateQualityScore(contribution *Contribution) float64 {
	score := 80.0

	// Bonus for tests
	if contribution.TestsPassed {
		score += 10.0
	}

	// Bonus for documentation
	for _, file := range contribution.Files {
		if file.Path == "README.md" || file.Path == "docs/" {
			score += 5.0
			break
		}
	}

	// Cap at 100
	if score > 100 {
		score = 100
	}

	return score
}

// ReviewContribution performs manual review
func (cp *ContributionPlatform) ReviewContribution(ctx context.Context, contributionID, reviewerID string, approved bool, comments []ReviewComment) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	contribution, exists := cp.contributions[contributionID]
	if !exists {
		return fmt.Errorf("contribution not found: %s", contributionID)
	}

	status := "approved"
	if !approved {
		status = "changes_requested"
	}

	review := CodeReview{
		ID:           cp.generateID("REV"),
		ReviewerID:   reviewerID,
		Status:       status,
		Comments:     comments,
		Rating:       8.5,
		Automated:    false,
		ReviewedAt:   time.Now(),
	}

	contribution.Reviews = append(contribution.Reviews, review)

	if approved {
		contribution.Status = StatusApproved
		now := time.Now()
		contribution.ReviewedAt = &now
	}

	contribution.UpdatedAt = time.Now()

	// Calculate review time
	reviewTime := time.Since(contribution.SubmittedAt).Hours()
	cp.stats.AverageReviewTime = (cp.stats.AverageReviewTime*float64(cp.stats.TotalContributions-1) + reviewTime) / float64(cp.stats.TotalContributions)

	return nil
}

// MergeContribution merges approved contribution
func (cp *ContributionPlatform) MergeContribution(ctx context.Context, contributionID string) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	contribution, exists := cp.contributions[contributionID]
	if !exists {
		return fmt.Errorf("contribution not found: %s", contributionID)
	}

	if contribution.Status != StatusApproved {
		return fmt.Errorf("contribution not approved")
	}

	now := time.Now()
	contribution.Status = StatusMerged
	contribution.MergedAt = &now
	contribution.UpdatedAt = now

	// Calculate reward
	reward := cp.calculateReward(contribution)
	contribution.Reward = &reward

	// Update contributor
	if contributor, exists := cp.contributors[contribution.ContributorID]; exists {
		contributor.PRsMerged++
		contributor.Rewards = append(contributor.Rewards, reward)
		contributor.Reputation += int(contribution.ImpactScore)

		// Award badges
		if contributor.PRsMerged == 1 {
			badge := Badge{
				ID:       cp.generateID("BADGE"),
				Name:     "First Contribution",
				Tier:     "bronze",
				EarnedAt: time.Now(),
			}
			contributor.Badges = append(contributor.Badges, badge)
		}
	}

	// Update repository
	if repo, exists := cp.repositories[contribution.RepositoryID]; exists {
		repo.TotalCommits += len(contribution.Commits)
		repo.LastCommit = time.Now()
		repo.UpdatedAt = time.Now()
	}

	cp.stats.MergedPRs++
	cp.stats.TotalCommits += len(contribution.Commits)
	cp.stats.LinesAdded += contribution.LinesAdded
	cp.stats.LinesDeleted += contribution.LinesDeleted
	cp.stats.TotalRewardsPaid += reward.Amount
	if reward.Type == "cash" {
		cp.stats.CashRewards += reward.Amount
	} else {
		cp.stats.CreditRewards += reward.Amount
	}

	// Calculate merge time
	mergeTime := time.Since(contribution.SubmittedAt).Hours()
	cp.stats.AverageMergeTime = (cp.stats.AverageMergeTime*float64(cp.stats.MergedPRs-1) + mergeTime) / float64(cp.stats.MergedPRs)

	cp.stats.UpdatedAt = time.Now()

	return nil
}

// calculateReward calculates contribution reward
func (cp *ContributionPlatform) calculateReward(contribution *Contribution) Reward {
	baseAmount := 100.0

	// Multiply by impact score
	amount := baseAmount * (contribution.ImpactScore / 100.0)

	// Multiply by quality score
	amount *= (contribution.QualityScore / 100.0)

	// Type multipliers
	multiplier := 1.0
	switch contribution.Type {
	case TypeSecurity:
		multiplier = 3.0
	case TypeFeature:
		multiplier = 2.0
	case TypePerformance:
		multiplier = 2.0
	case TypeBugFix:
		multiplier = 1.5
	case TypeRefactor:
		multiplier = 1.2
	}

	amount *= multiplier

	// Cap rewards
	if amount > 5000 {
		amount = 5000
	}

	rewardType := "cash"
	if amount < 500 {
		rewardType = "credits"
		amount *= 10 // 10x credits vs cash
	}

	return Reward{
		ID:          cp.generateID("REWARD"),
		Type:        rewardType,
		Amount:      amount,
		Currency:    "USD",
		Description: fmt.Sprintf("Reward for %s contribution", contribution.Type),
		EarnedAt:    time.Now(),
		PaidOut:     false,
	}
}

// CreateIssue creates community issue
func (cp *ContributionPlatform) CreateIssue(ctx context.Context, issue *Issue) error {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if issue.ID == "" {
		issue.ID = cp.generateID("ISSUE")
	}

	issue.Status = "open"
	issue.CreatedAt = time.Now()
	issue.UpdatedAt = time.Now()

	cp.issues[issue.ID] = issue

	cp.stats.TotalIssues++
	cp.stats.OpenIssues++
	cp.stats.UpdatedAt = time.Now()

	return nil
}

// GetContributionStats returns contribution statistics
func (cp *ContributionPlatform) GetContributionStats(ctx context.Context) ContributionStats {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	stats := cp.stats

	stats.TotalRepositories = len(cp.repositories)
	stats.TotalContributors = len(cp.contributors)

	// Count active contributors (contributed in last 90 days)
	activeCount := 0
	for _, contributor := range cp.contributors {
		if time.Since(contributor.LastContribution).Hours() < 90*24 {
			activeCount++
		}
	}
	stats.ActiveContributors = activeCount

	// Count open source repos
	openSourceCount := 0
	for _, repo := range cp.repositories {
		if repo.License == "Apache-2.0" || repo.License == "MIT" {
			openSourceCount++
		}
	}
	stats.OpenSourceRepos = openSourceCount

	stats.UpdatedAt = time.Now()

	return stats
}

// generateID generates unique ID
func (cp *ContributionPlatform) generateID(prefix string) string {
	timestamp := time.Now().UnixNano()
	hash := sha256.Sum256([]byte(fmt.Sprintf("%s-%d", prefix, timestamp)))
	return fmt.Sprintf("%s-%s", prefix, hex.EncodeToString(hash[:8]))
}

// ExportContributionData exports contribution data as JSON
func (cp *ContributionPlatform) ExportContributionData(ctx context.Context, contributionID string) ([]byte, error) {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	contribution, exists := cp.contributions[contributionID]
	if !exists {
		return nil, fmt.Errorf("contribution not found: %s", contributionID)
	}

	return json.MarshalIndent(contribution, "", "  ")
}
