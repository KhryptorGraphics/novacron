package analysis

import (
	"testing"
	"time"

	"github.com/novacron/backend/core/research/monitoring"
)

func TestNewFeasibilityAnalyzer(t *testing.T) {
	config := AnalyzerConfig{
		MinTechnicalScore: 0.6,
		MinResourceScore:  0.5,
		MinTimelineScore:  0.5,
		MinROIScore:       0.5,
		MinOverallScore:   0.6,
	}

	analyzer := NewFeasibilityAnalyzer(config)

	if analyzer == nil {
		t.Fatal("NewFeasibilityAnalyzer returned nil")
	}
}

func TestAnalyzeTechnicalFeasibility(t *testing.T) {
	config := AnalyzerConfig{}
	analyzer := NewFeasibilityAnalyzer(config)

	tests := []struct {
		name     string
		paper    *monitoring.ResearchPaper
		minScore float64
	}{
		{
			name: "Practical implementation",
			paper: &monitoring.ResearchPaper{
				Title:    "Practical Implementation of Distributed Systems",
				Abstract: "We present a practical implementation with experiments and benchmarks",
			},
			minScore: 0.5,
		},
		{
			name: "Theoretical paper",
			paper: &monitoring.ResearchPaper{
				Title:    "Theoretical Analysis of Algorithms",
				Abstract: "A theoretical framework remains open for future work",
			},
			minScore: 0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			score := analyzer.analyzeTechnicalFeasibility(tt.paper)

			if score < 0 || score > 1 {
				t.Errorf("Score out of range [0,1]: %f", score)
			}

			if score < tt.minScore {
				t.Errorf("Expected score >= %f, got %f", tt.minScore, score)
			}
		})
	}
}

func TestEstimateResources(t *testing.T) {
	config := AnalyzerConfig{}
	analyzer := NewFeasibilityAnalyzer(config)

	paper := &monitoring.ResearchPaper{
		Abstract: "A distributed machine learning system with quantum cryptography",
	}

	estimate := analyzer.estimateResources(paper)

	if estimate.Engineers < 1 {
		t.Error("Expected at least 1 engineer")
	}

	if estimate.Researchers < 1 {
		t.Error("Expected at least 1 researcher")
	}

	if estimate.EstimatedCost <= 0 {
		t.Error("Expected positive estimated cost")
	}

	if estimate.Budget <= 0 {
		t.Error("Expected positive budget")
	}
}

func TestEstimateTimeline(t *testing.T) {
	config := AnalyzerConfig{}
	analyzer := NewFeasibilityAnalyzer(config)

	paper := &monitoring.ResearchPaper{
		Abstract: "Implementation of distributed consensus",
	}

	timeline := analyzer.estimateTimeline(paper, 0.8)

	if timeline.Research <= 0 {
		t.Error("Expected positive research time")
	}

	if timeline.Total <= 0 {
		t.Error("Expected positive total time")
	}

	// Check that total equals sum of phases
	sum := timeline.Research + timeline.Prototyping +
		timeline.Development + timeline.Testing + timeline.Production

	if sum != timeline.Total {
		t.Errorf("Total time mismatch: %v != %v", timeline.Total, sum)
	}
}

func TestAssessRisks(t *testing.T) {
	config := AnalyzerConfig{}
	analyzer := NewFeasibilityAnalyzer(config)

	paper := &monitoring.ResearchPaper{
		Abstract: "Novel quantum distributed system",
	}

	risks := analyzer.assessRisks(paper, 0.5)

	if len(risks) == 0 {
		t.Error("Expected some risks to be identified")
	}

	for _, risk := range risks {
		if risk.Probability < 0 || risk.Probability > 1 {
			t.Errorf("Risk probability out of range: %f", risk.Probability)
		}

		if risk.Impact < 0 || risk.Impact > 1 {
			t.Errorf("Risk impact out of range: %f", risk.Impact)
		}

		if risk.Type == "" {
			t.Error("Risk type should not be empty")
		}

		if risk.Mitigation == "" {
			t.Error("Risk mitigation should not be empty")
		}
	}
}

func TestCalculateROI(t *testing.T) {
	config := AnalyzerConfig{}
	analyzer := NewFeasibilityAnalyzer(config)

	paper := &monitoring.ResearchPaper{
		Abstract: "Performance optimization with security enhancements",
	}

	resources := ResourceEstimate{
		Budget: 1000000,
	}

	timeline := TimelineEstimate{
		Total: 180 * 24 * time.Hour,
	}

	roi := analyzer.calculateROI(paper, resources, timeline)

	if roi.Investment != resources.Budget {
		t.Errorf("Expected investment %d, got %d", resources.Budget, roi.Investment)
	}

	if roi.ExpectedReturn <= 0 {
		t.Error("Expected positive return")
	}

	if roi.NetPresentValue <= 0 {
		t.Error("Expected positive NPV")
	}

	if len(roi.QualitativeBenefits) == 0 {
		t.Error("Expected qualitative benefits")
	}
}

func TestAnalyze(t *testing.T) {
	config := AnalyzerConfig{
		MinTechnicalScore: 0.5,
		MinResourceScore:  0.5,
		MinTimelineScore:  0.5,
		MinROIScore:       0.5,
		MinOverallScore:   0.5,
	}

	analyzer := NewFeasibilityAnalyzer(config)

	paper := &monitoring.ResearchPaper{
		ID:       "test-paper-1",
		Title:    "Practical Distributed Consensus",
		Abstract: "A practical implementation with experiments and benchmarks",
		Authors:  []string{"Author 1"},
	}

	analysis, err := analyzer.Analyze(paper)

	if err != nil {
		t.Fatalf("Analysis failed: %v", err)
	}

	if analysis.PaperID != paper.ID {
		t.Errorf("Expected paper ID %s, got %s", paper.ID, analysis.PaperID)
	}

	if analysis.OverallScore < 0 || analysis.OverallScore > 1 {
		t.Errorf("Overall score out of range: %f", analysis.OverallScore)
	}

	if analysis.TechnicalScore < 0 || analysis.TechnicalScore > 1 {
		t.Errorf("Technical score out of range: %f", analysis.TechnicalScore)
	}

	if analysis.Recommendation == "" {
		t.Error("Recommendation should not be empty")
	}

	if analysis.Priority < 1 || analysis.Priority > 5 {
		t.Errorf("Priority out of range [1,5]: %d", analysis.Priority)
	}

	if len(analysis.NextSteps) == 0 {
		t.Error("Expected next steps")
	}

	if len(analysis.Risks) == 0 {
		t.Error("Expected some risks")
	}
}

func TestGenerateRecommendation(t *testing.T) {
	config := AnalyzerConfig{}
	analyzer := NewFeasibilityAnalyzer(config)

	tests := []struct {
		name          string
		overallScore  float64
		expectedMatch string
	}{
		{"High score", 0.85, "HIGHLY RECOMMENDED"},
		{"Medium score", 0.65, "RECOMMENDED"},
		{"Conditional", 0.45, "CONDITIONAL"},
		{"Low score", 0.25, "NOT RECOMMENDED"},
		{"Very low", 0.05, "REJECT"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analysis := &FeasibilityAnalysis{
				OverallScore: tt.overallScore,
			}

			rec := analyzer.generateRecommendation(analysis)

			if rec == "" {
				t.Error("Recommendation should not be empty")
			}
		})
	}
}

func TestCalculatePriority(t *testing.T) {
	config := AnalyzerConfig{}
	analyzer := NewFeasibilityAnalyzer(config)

	tests := []struct {
		name     string
		overall  float64
		roi      float64
		expected int
	}{
		{"Highest priority", 0.85, 0.85, 1},
		{"High priority", 0.75, 0.6, 2},
		{"Medium priority", 0.55, 0.5, 3},
		{"Low priority", 0.35, 0.3, 4},
		{"Lowest priority", 0.15, 0.1, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analysis := &FeasibilityAnalysis{
				OverallScore: tt.overall,
				ROIScore:     tt.roi,
			}

			priority := analyzer.calculatePriority(analysis)

			if priority < 1 || priority > 5 {
				t.Errorf("Priority out of range [1,5]: %d", priority)
			}
		})
	}
}
