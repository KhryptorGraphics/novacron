package lab

import (
	"fmt"
	"sync"
	"time"
)

// InnovationLab manages research and innovation projects
type InnovationLab struct {
	config     LabConfig
	ideas      map[string]*Idea
	experiments map[string]*Experiment
	features   map[string]*Feature
	metrics    LabMetrics
	mu         sync.RWMutex
}

// LabConfig configures the innovation lab
type LabConfig struct {
	ResearchTimePercent float64 // 20% time
	MaxActiveExperiments int
	BudgetPerYear       int64
	IdeaEvaluationCriteria []string
	ExperimentDuration  time.Duration
}

// Idea represents an innovation idea
type Idea struct {
	ID          string
	Title       string
	Description string
	Submitter   string
	SubmittedAt time.Time

	// Evaluation
	Category    string // research, feature, optimization, tool
	Priority    int
	Feasibility float64
	Impact      float64
	Cost        int64

	Status      IdeaStatus
	Votes       int
	Comments    []Comment
	Tags        []string

	// Progression
	ExperimentID string
	FeatureID    string
}

// IdeaStatus represents idea status
type IdeaStatus string

const (
	StatusSubmitted IdeaStatus = "submitted"
	StatusEvaluating IdeaStatus = "evaluating"
	StatusApproved  IdeaStatus = "approved"
	StatusExperiment IdeaStatus = "experiment"
	StatusImplemented IdeaStatus = "implemented"
	StatusRejected  IdeaStatus = "rejected"
)

// Comment represents a comment on an idea
type Comment struct {
	Author    string
	Content   string
	Timestamp time.Time
}

// Experiment represents an innovation experiment
type Experiment struct {
	ID          string
	IdeaID      string
	Title       string
	Description string
	Hypothesis  string

	Researcher  string
	Team        []string
	StartDate   time.Time
	EndDate     time.Time
	Status      ExperimentStatus

	// Methodology
	Methodology string
	Variables   []Variable
	Metrics     []Metric

	// Results
	Results     ExperimentResults
	Conclusion  string
	NextSteps   []string

	// Resources
	Budget      int64
	Resources   []string
	Repository  string
}

// ExperimentStatus represents experiment status
type ExperimentStatus string

const (
	ExpStatusPlanning   ExperimentStatus = "planning"
	ExpStatusRunning    ExperimentStatus = "running"
	ExpStatusAnalyzing  ExperimentStatus = "analyzing"
	ExpStatusCompleted  ExperimentStatus = "completed"
	ExpStatusAbandoned  ExperimentStatus = "abandoned"
)

// Variable represents an experimental variable
type Variable struct {
	Name        string
	Type        string
	Values      []interface{}
	Control     interface{}
}

// Metric represents an experimental metric
type Metric struct {
	Name        string
	Unit        string
	Target      float64
	Baseline    float64
	Current     float64
}

// ExperimentResults contains experiment results
type ExperimentResults struct {
	Success     bool
	DataPoints  int
	Improvement float64
	Findings    []string
	Visualizations []string
	RawData     string
}

// Feature represents a feature derived from research
type Feature struct {
	ID          string
	ExperimentID string
	Title       string
	Description string

	Status      FeatureStatus
	Priority    int
	Owner       string
	Team        []string

	StartDate   time.Time
	LaunchDate  time.Time

	// Implementation
	Specification string
	Repository    string
	PRs           []string
	Tests         []string
	Documentation string

	// Impact
	AdoptionRate  float64
	UserFeedback  []Feedback
	Metrics       FeatureMetrics
}

// FeatureStatus represents feature status
type FeatureStatus string

const (
	FeatureStatusProposed   FeatureStatus = "proposed"
	FeatureStatusDevelopment FeatureStatus = "development"
	FeatureStatusTesting    FeatureStatus = "testing"
	FeatureStatusRollout    FeatureStatus = "rollout"
	FeatureStatusLaunched   FeatureStatus = "launched"
	FeatureStatusDeprecated FeatureStatus = "deprecated"
)

// Feedback represents user feedback
type Feedback struct {
	User      string
	Rating    int
	Comment   string
	Timestamp time.Time
}

// FeatureMetrics tracks feature metrics
type FeatureMetrics struct {
	Usage           int64
	ActiveUsers     int
	Satisfaction    float64
	PerformanceGain float64
	BugCount        int
}

// LabMetrics tracks innovation lab metrics
type LabMetrics struct {
	TotalIdeas         int
	ApprovedIdeas      int
	RejectedIdeas      int
	TotalExperiments   int
	SuccessfulExperiments int
	TotalFeatures      int
	LaunchedFeatures   int
	PapersPublished    int
	PatentsFiled       int
	InnovationROI      float64
}

// NewInnovationLab creates a new innovation lab
func NewInnovationLab(config LabConfig) *InnovationLab {
	return &InnovationLab{
		config:      config,
		ideas:       make(map[string]*Idea),
		experiments: make(map[string]*Experiment),
		features:    make(map[string]*Feature),
	}
}

// SubmitIdea submits a new idea
func (il *InnovationLab) SubmitIdea(idea *Idea) error {
	il.mu.Lock()
	defer il.mu.Unlock()

	if _, exists := il.ideas[idea.ID]; exists {
		return fmt.Errorf("idea already exists: %s", idea.ID)
	}

	idea.SubmittedAt = time.Now()
	idea.Status = StatusSubmitted
	idea.Comments = make([]Comment, 0)
	idea.Votes = 0

	il.ideas[idea.ID] = idea
	il.metrics.TotalIdeas++

	return nil
}

// EvaluateIdea evaluates an idea
func (il *InnovationLab) EvaluateIdea(ideaID string) (*IdeaEvaluation, error) {
	il.mu.Lock()
	defer il.mu.Unlock()

	idea, exists := il.ideas[ideaID]
	if !exists {
		return nil, fmt.Errorf("idea not found: %s", ideaID)
	}

	evaluation := &IdeaEvaluation{
		IdeaID:      ideaID,
		EvaluatedAt: time.Now(),
	}

	// Evaluate against criteria
	for _, criterion := range il.config.IdeaEvaluationCriteria {
		score := il.evaluateCriterion(idea, criterion)
		evaluation.Scores = append(evaluation.Scores, CriterionScore{
			Criterion: criterion,
			Score:     score,
		})
		evaluation.TotalScore += score
	}

	// Calculate overall score
	if len(evaluation.Scores) > 0 {
		evaluation.TotalScore /= float64(len(evaluation.Scores))
	}

	// Make recommendation
	if evaluation.TotalScore >= 0.7 {
		evaluation.Recommendation = "APPROVE"
		idea.Status = StatusApproved
		il.metrics.ApprovedIdeas++
	} else if evaluation.TotalScore >= 0.5 {
		evaluation.Recommendation = "CONDITIONAL"
		idea.Status = StatusEvaluating
	} else {
		evaluation.Recommendation = "REJECT"
		idea.Status = StatusRejected
		il.metrics.RejectedIdeas++
	}

	return evaluation, nil
}

// IdeaEvaluation contains idea evaluation results
type IdeaEvaluation struct {
	IdeaID         string
	EvaluatedAt    time.Time
	Scores         []CriterionScore
	TotalScore     float64
	Recommendation string
}

// CriterionScore represents a criterion score
type CriterionScore struct {
	Criterion string
	Score     float64
}

// evaluateCriterion evaluates a single criterion
func (il *InnovationLab) evaluateCriterion(idea *Idea, criterion string) float64 {
	switch criterion {
	case "novelty":
		return il.evaluateNovelty(idea)
	case "feasibility":
		return idea.Feasibility
	case "impact":
		return idea.Impact
	case "alignment":
		return il.evaluateAlignment(idea)
	case "cost":
		return il.evaluateCost(idea)
	default:
		return 0.5
	}
}

// evaluateNovelty evaluates idea novelty
func (il *InnovationLab) evaluateNovelty(idea *Idea) float64 {
	// Check for similar ideas
	similar := 0
	for _, existingIdea := range il.ideas {
		if existingIdea.ID != idea.ID && existingIdea.Category == idea.Category {
			similar++
		}
	}

	// Novelty decreases with similar ideas
	novelty := 1.0 - (float64(similar) / 10.0)
	if novelty < 0 {
		novelty = 0
	}

	return novelty
}

// evaluateAlignment evaluates strategic alignment
func (il *InnovationLab) evaluateAlignment(idea *Idea) float64 {
	// Strategic alignment with company goals
	return 0.8 // Simplified
}

// evaluateCost evaluates cost-effectiveness
func (il *InnovationLab) evaluateCost(idea *Idea) float64 {
	// Lower cost = higher score
	if idea.Cost <= 10000 {
		return 1.0
	} else if idea.Cost <= 50000 {
		return 0.8
	} else if idea.Cost <= 100000 {
		return 0.6
	} else if idea.Cost <= 500000 {
		return 0.4
	}
	return 0.2
}

// CreateExperiment creates an experiment from an idea
func (il *InnovationLab) CreateExperiment(ideaID string, researcher string) (*Experiment, error) {
	il.mu.Lock()
	defer il.mu.Unlock()

	idea, exists := il.ideas[ideaID]
	if !exists {
		return nil, fmt.Errorf("idea not found: %s", ideaID)
	}

	if idea.Status != StatusApproved {
		return nil, fmt.Errorf("idea not approved: %s", ideaID)
	}

	// Check active experiments limit
	activeExperiments := 0
	for _, exp := range il.experiments {
		if exp.Status == ExpStatusRunning {
			activeExperiments++
		}
	}

	if activeExperiments >= il.config.MaxActiveExperiments {
		return nil, fmt.Errorf("max active experiments reached")
	}

	experiment := &Experiment{
		ID:          fmt.Sprintf("exp-%d", time.Now().Unix()),
		IdeaID:      ideaID,
		Title:       idea.Title,
		Description: idea.Description,
		Researcher:  researcher,
		StartDate:   time.Now(),
		EndDate:     time.Now().Add(il.config.ExperimentDuration),
		Status:      ExpStatusPlanning,
		Budget:      idea.Cost,
		Variables:   make([]Variable, 0),
		Metrics:     make([]Metric, 0),
	}

	il.experiments[experiment.ID] = experiment
	il.metrics.TotalExperiments++

	idea.Status = StatusExperiment
	idea.ExperimentID = experiment.ID

	return experiment, nil
}

// CompleteExperiment completes an experiment
func (il *InnovationLab) CompleteExperiment(experimentID string, results ExperimentResults) error {
	il.mu.Lock()
	defer il.mu.Unlock()

	experiment, exists := il.experiments[experimentID]
	if !exists {
		return fmt.Errorf("experiment not found: %s", experimentID)
	}

	experiment.Results = results
	experiment.Status = ExpStatusCompleted

	if results.Success {
		il.metrics.SuccessfulExperiments++
	}

	return nil
}

// CreateFeature creates a feature from an experiment
func (il *InnovationLab) CreateFeature(experimentID string, owner string) (*Feature, error) {
	il.mu.Lock()
	defer il.mu.Unlock()

	experiment, exists := il.experiments[experimentID]
	if !exists {
		return nil, fmt.Errorf("experiment not found: %s", experimentID)
	}

	if experiment.Status != ExpStatusCompleted || !experiment.Results.Success {
		return nil, fmt.Errorf("experiment not successful")
	}

	feature := &Feature{
		ID:           fmt.Sprintf("feat-%d", time.Now().Unix()),
		ExperimentID: experimentID,
		Title:        experiment.Title,
		Description:  experiment.Description,
		Status:       FeatureStatusProposed,
		Owner:        owner,
		StartDate:    time.Now(),
	}

	il.features[feature.ID] = feature
	il.metrics.TotalFeatures++

	// Update idea status
	if idea, exists := il.ideas[experiment.IdeaID]; exists {
		idea.Status = StatusImplemented
		idea.FeatureID = feature.ID
	}

	return feature, nil
}

// LaunchFeature launches a feature
func (il *InnovationLab) LaunchFeature(featureID string) error {
	il.mu.Lock()
	defer il.mu.Unlock()

	feature, exists := il.features[featureID]
	if !exists {
		return fmt.Errorf("feature not found: %s", featureID)
	}

	feature.Status = FeatureStatusLaunched
	feature.LaunchDate = time.Now()
	il.metrics.LaunchedFeatures++

	return nil
}

// GetMetrics returns innovation lab metrics
func (il *InnovationLab) GetMetrics() LabMetrics {
	il.mu.RLock()
	defer il.mu.RUnlock()

	// Calculate ROI
	totalInvestment := int64(0)
	totalReturn := int64(0)

	for _, feature := range il.features {
		if feature.Status == FeatureStatusLaunched {
			// Estimate return based on usage and impact
			totalReturn += int64(feature.Metrics.Usage) * 100
		}
	}

	for _, experiment := range il.experiments {
		totalInvestment += experiment.Budget
	}

	if totalInvestment > 0 {
		il.metrics.InnovationROI = float64(totalReturn) / float64(totalInvestment)
	}

	return il.metrics
}

// GenerateInnovationReport generates an innovation report
func (il *InnovationLab) GenerateInnovationReport() *InnovationReport {
	il.mu.RLock()
	defer il.mu.RUnlock()

	report := &InnovationReport{
		GeneratedAt: time.Now(),
		Metrics:     il.metrics,
		TopIdeas:    il.getTopIdeas(5),
		ActiveExperiments: il.getActiveExperiments(),
		RecentFeatures: il.getRecentFeatures(5),
	}

	return report
}

// InnovationReport contains innovation metrics and highlights
type InnovationReport struct {
	GeneratedAt       time.Time
	Metrics           LabMetrics
	TopIdeas          []*Idea
	ActiveExperiments []*Experiment
	RecentFeatures    []*Feature
}

// getTopIdeas returns top ideas by votes
func (il *InnovationLab) getTopIdeas(limit int) []*Idea {
	ideas := make([]*Idea, 0)
	for _, idea := range il.ideas {
		ideas = append(ideas, idea)
	}

	// Sort by votes (simplified)
	if limit > len(ideas) {
		limit = len(ideas)
	}

	return ideas[:limit]
}

// getActiveExperiments returns active experiments
func (il *InnovationLab) getActiveExperiments() []*Experiment {
	experiments := make([]*Experiment, 0)
	for _, exp := range il.experiments {
		if exp.Status == ExpStatusRunning {
			experiments = append(experiments, exp)
		}
	}
	return experiments
}

// getRecentFeatures returns recently launched features
func (il *InnovationLab) getRecentFeatures(limit int) []*Feature {
	features := make([]*Feature, 0)
	for _, feat := range il.features {
		if feat.Status == FeatureStatusLaunched {
			features = append(features, feat)
		}
	}

	// Sort by launch date (simplified)
	if limit > len(features) {
		limit = len(features)
	}

	return features[:limit]
}

// VoteIdea adds a vote to an idea
func (il *InnovationLab) VoteIdea(ideaID string) error {
	il.mu.Lock()
	defer il.mu.Unlock()

	idea, exists := il.ideas[ideaID]
	if !exists {
		return fmt.Errorf("idea not found: %s", ideaID)
	}

	idea.Votes++
	return nil
}

// AddComment adds a comment to an idea
func (il *InnovationLab) AddComment(ideaID string, comment Comment) error {
	il.mu.Lock()
	defer il.mu.Unlock()

	idea, exists := il.ideas[ideaID]
	if !exists {
		return fmt.Errorf("idea not found: %s", ideaID)
	}

	comment.Timestamp = time.Now()
	idea.Comments = append(idea.Comments, comment)

	return nil
}
