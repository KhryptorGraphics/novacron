package analysis

import (
	"fmt"
	"time"

	"github.com/novacron/backend/core/research/monitoring"
)

// FeasibilityAnalysis represents research-to-production feasibility
type FeasibilityAnalysis struct {
	PaperID     string
	Title       string

	// Scores (0-1)
	TechnicalScore    float64
	ResourceScore     float64
	TimelineScore     float64
	ROIScore          float64
	OverallScore      float64

	// Assessments
	TechnicalFeasibility string
	RequiredResources    ResourceEstimate
	Timeline             TimelineEstimate
	Risks                []Risk
	ROI                  ROIEstimate

	// Decision
	Recommendation string
	Priority       int
	NextSteps      []string

	AnalyzedAt time.Time
	AnalyzedBy string
}

// ResourceEstimate estimates required resources
type ResourceEstimate struct {
	Engineers      int
	Researchers    int
	Budget         int64
	Hardware       []string
	Software       []string
	Partnerships   []string
	EstimatedCost  int64
}

// TimelineEstimate estimates implementation timeline
type TimelineEstimate struct {
	Research      time.Duration
	Prototyping   time.Duration
	Development   time.Duration
	Testing       time.Duration
	Production    time.Duration
	Total         time.Duration
}

// Risk represents implementation risk
type Risk struct {
	Type        string
	Description string
	Probability float64
	Impact      float64
	Mitigation  string
}

// ROIEstimate estimates return on investment
type ROIEstimate struct {
	Investment       int64
	ExpectedReturn   int64
	PaybackPeriod    time.Duration
	NetPresentValue  int64
	IRR              float64
	QualitativeBenefits []string
}

// FeasibilityAnalyzer analyzes research feasibility
type FeasibilityAnalyzer struct {
	config AnalyzerConfig
}

// AnalyzerConfig configures feasibility analysis
type AnalyzerConfig struct {
	MinTechnicalScore float64
	MinResourceScore  float64
	MinTimelineScore  float64
	MinROIScore       float64
	MinOverallScore   float64
}

// NewFeasibilityAnalyzer creates a new feasibility analyzer
func NewFeasibilityAnalyzer(config AnalyzerConfig) *FeasibilityAnalyzer {
	return &FeasibilityAnalyzer{
		config: config,
	}
}

// Analyze performs feasibility analysis on a research paper
func (fa *FeasibilityAnalyzer) Analyze(paper *monitoring.ResearchPaper) (*FeasibilityAnalysis, error) {
	analysis := &FeasibilityAnalysis{
		PaperID:    paper.ID,
		Title:      paper.Title,
		AnalyzedAt: time.Now(),
		AnalyzedBy: "AI-Analyzer-v1.0",
	}

	// Analyze technical feasibility
	analysis.TechnicalScore = fa.analyzeTechnicalFeasibility(paper)
	analysis.TechnicalFeasibility = fa.describeTechnicalFeasibility(analysis.TechnicalScore)

	// Estimate resources
	analysis.RequiredResources = fa.estimateResources(paper)
	analysis.ResourceScore = fa.scoreResources(analysis.RequiredResources)

	// Estimate timeline
	analysis.Timeline = fa.estimateTimeline(paper, analysis.TechnicalScore)
	analysis.TimelineScore = fa.scoreTimeline(analysis.Timeline)

	// Assess risks
	analysis.Risks = fa.assessRisks(paper, analysis.TechnicalScore)

	// Calculate ROI
	analysis.ROI = fa.calculateROI(paper, analysis.RequiredResources, analysis.Timeline)
	analysis.ROIScore = fa.scoreROI(analysis.ROI)

	// Overall score
	analysis.OverallScore = fa.calculateOverallScore(analysis)

	// Generate recommendation
	analysis.Recommendation = fa.generateRecommendation(analysis)
	analysis.Priority = fa.calculatePriority(analysis)
	analysis.NextSteps = fa.generateNextSteps(analysis)

	return analysis, nil
}

// analyzeTechnicalFeasibility analyzes technical feasibility
func (fa *FeasibilityAnalyzer) analyzeTechnicalFeasibility(paper *monitoring.ResearchPaper) float64 {
	score := 0.5 // baseline

	// Check for practical implementations
	practicalKeywords := []string{
		"implementation", "prototype", "experiment", "benchmark",
		"open source", "github", "practical", "deployment",
	}

	text := paper.Title + " " + paper.Abstract
	for _, keyword := range practicalKeywords {
		if contains(text, keyword) {
			score += 0.05
		}
	}

	// Check for theoretical barriers
	theoreticalKeywords := []string{
		"theoretical", "conjecture", "unproven", "speculative",
		"future work", "remains open",
	}

	for _, keyword := range theoreticalKeywords {
		if contains(text, keyword) {
			score -= 0.1
		}
	}

	// Check for hardware requirements
	if contains(text, "quantum") && !contains(text, "NISQ") {
		score -= 0.2 // Requires quantum hardware
	}

	// Normalize
	if score < 0 {
		score = 0
	}
	if score > 1 {
		score = 1
	}

	return score
}

// describeTechnicalFeasibility provides text description
func (fa *FeasibilityAnalyzer) describeTechnicalFeasibility(score float64) string {
	if score >= 0.8 {
		return "Highly feasible - Can be implemented with existing technology"
	} else if score >= 0.6 {
		return "Feasible - Some technical challenges but achievable"
	} else if score >= 0.4 {
		return "Moderately feasible - Significant technical hurdles"
	} else if score >= 0.2 {
		return "Challenging - Requires major breakthroughs"
	}
	return "Not feasible - Theoretical or requires unavailable technology"
}

// estimateResources estimates required resources
func (fa *FeasibilityAnalyzer) estimateResources(paper *monitoring.ResearchPaper) ResourceEstimate {
	estimate := ResourceEstimate{
		Hardware: make([]string, 0),
		Software: make([]string, 0),
		Partnerships: make([]string, 0),
	}

	// Base team size
	estimate.Engineers = 3
	estimate.Researchers = 1

	// Adjust based on complexity
	if contains(paper.Abstract, "distributed") {
		estimate.Engineers += 2
		estimate.Hardware = append(estimate.Hardware, "Distributed cluster (20+ nodes)")
	}

	if contains(paper.Abstract, "quantum") {
		estimate.Researchers += 2
		estimate.Hardware = append(estimate.Hardware, "Quantum computer access")
		estimate.Partnerships = append(estimate.Partnerships, "IBM Quantum, Google Quantum AI")
	}

	if contains(paper.Abstract, "machine learning") {
		estimate.Engineers += 2
		estimate.Hardware = append(estimate.Hardware, "GPU cluster (8+ GPUs)")
		estimate.Software = append(estimate.Software, "PyTorch, TensorFlow")
	}

	if contains(paper.Abstract, "cryptography") {
		estimate.Researchers += 1
		estimate.Software = append(estimate.Software, "Cryptographic libraries")
	}

	// Estimate costs
	estimate.EstimatedCost = int64(estimate.Engineers*200000 + estimate.Researchers*250000)
	estimate.Budget = estimate.EstimatedCost * 2 // 2x for overhead

	return estimate
}

// scoreResources scores resource feasibility
func (fa *FeasibilityAnalyzer) scoreResources(estimate ResourceEstimate) float64 {
	// Score based on team size and budget
	score := 1.0

	if estimate.Engineers > 5 {
		score -= 0.1
	}
	if estimate.Researchers > 3 {
		score -= 0.1
	}
	if estimate.Budget > 2000000 {
		score -= 0.2
	}
	if len(estimate.Hardware) > 3 {
		score -= 0.1
	}
	if len(estimate.Partnerships) > 2 {
		score -= 0.15
	}

	if score < 0 {
		score = 0
	}

	return score
}

// estimateTimeline estimates implementation timeline
func (fa *FeasibilityAnalyzer) estimateTimeline(paper *monitoring.ResearchPaper, technicalScore float64) TimelineEstimate {
	estimate := TimelineEstimate{}

	// Base timeline
	estimate.Research = 30 * 24 * time.Hour      // 1 month
	estimate.Prototyping = 60 * 24 * time.Hour   // 2 months
	estimate.Development = 90 * 24 * time.Hour   // 3 months
	estimate.Testing = 30 * 24 * time.Hour       // 1 month
	estimate.Production = 30 * 24 * time.Hour    // 1 month

	// Adjust based on technical feasibility
	if technicalScore < 0.6 {
		estimate.Research *= 2
		estimate.Prototyping *= 2
	}

	// Adjust based on complexity
	if contains(paper.Abstract, "distributed") {
		estimate.Development += 60 * 24 * time.Hour
		estimate.Testing += 30 * 24 * time.Hour
	}

	if contains(paper.Abstract, "quantum") {
		estimate.Research += 60 * 24 * time.Hour
		estimate.Prototyping += 90 * 24 * time.Hour
	}

	estimate.Total = estimate.Research + estimate.Prototyping +
		estimate.Development + estimate.Testing + estimate.Production

	return estimate
}

// scoreTimeline scores timeline feasibility
func (fa *FeasibilityAnalyzer) scoreTimeline(estimate TimelineEstimate) float64 {
	months := estimate.Total.Hours() / (24 * 30)

	if months <= 6 {
		return 1.0
	} else if months <= 12 {
		return 0.8
	} else if months <= 18 {
		return 0.6
	} else if months <= 24 {
		return 0.4
	}
	return 0.2
}

// assessRisks assesses implementation risks
func (fa *FeasibilityAnalyzer) assessRisks(paper *monitoring.ResearchPaper, technicalScore float64) []Risk {
	risks := make([]Risk, 0)

	// Technical risk
	if technicalScore < 0.7 {
		risks = append(risks, Risk{
			Type:        "Technical",
			Description: "Technology may not be mature enough for production",
			Probability: 1.0 - technicalScore,
			Impact:      0.9,
			Mitigation:  "Extensive prototyping and proof-of-concept phase",
		})
	}

	// Resource risk
	if contains(paper.Abstract, "quantum") {
		risks = append(risks, Risk{
			Type:        "Resource",
			Description: "Requires access to quantum computing hardware",
			Probability: 0.6,
			Impact:      0.8,
			Mitigation:  "Partner with IBM Quantum or Google Quantum AI",
		})
	}

	// Timeline risk
	if contains(paper.Abstract, "novel") || contains(paper.Abstract, "breakthrough") {
		risks = append(risks, Risk{
			Type:        "Timeline",
			Description: "Novel approaches may have unexpected challenges",
			Probability: 0.5,
			Impact:      0.7,
			Mitigation:  "Build contingency time into schedule",
		})
	}

	// Market risk
	risks = append(risks, Risk{
		Type:        "Market",
		Description: "Market needs may change during development",
		Probability: 0.4,
		Impact:      0.6,
		Mitigation:  "Continuous customer feedback and agile development",
	})

	return risks
}

// calculateROI calculates return on investment
func (fa *FeasibilityAnalyzer) calculateROI(paper *monitoring.ResearchPaper, resources ResourceEstimate, timeline TimelineEstimate) ROIEstimate {
	estimate := ROIEstimate{
		Investment:          resources.Budget,
		PaybackPeriod:       timeline.Total * 2,
		QualitativeBenefits: make([]string, 0),
	}

	// Estimate return (simplified)
	multiplier := 5.0 // Base 5x return

	if contains(paper.Abstract, "performance") {
		multiplier += 2.0
		estimate.QualitativeBenefits = append(estimate.QualitativeBenefits,
			"Significant performance improvements")
	}

	if contains(paper.Abstract, "security") {
		multiplier += 3.0
		estimate.QualitativeBenefits = append(estimate.QualitativeBenefits,
			"Enhanced security posture")
	}

	if contains(paper.Abstract, "cost") || contains(paper.Abstract, "efficiency") {
		multiplier += 2.0
		estimate.QualitativeBenefits = append(estimate.QualitativeBenefits,
			"Reduced operational costs")
	}

	estimate.ExpectedReturn = int64(float64(estimate.Investment) * multiplier)
	estimate.NetPresentValue = estimate.ExpectedReturn - estimate.Investment

	// Simple IRR approximation
	years := estimate.PaybackPeriod.Hours() / (24 * 365)
	if years > 0 {
		estimate.IRR = (multiplier - 1.0) / years
	}

	return estimate
}

// scoreROI scores ROI feasibility
func (fa *FeasibilityAnalyzer) scoreROI(estimate ROIEstimate) float64 {
	if estimate.IRR >= 1.0 {
		return 1.0
	} else if estimate.IRR >= 0.5 {
		return 0.8
	} else if estimate.IRR >= 0.3 {
		return 0.6
	} else if estimate.IRR >= 0.2 {
		return 0.4
	}
	return 0.2
}

// calculateOverallScore calculates overall feasibility score
func (fa *FeasibilityAnalyzer) calculateOverallScore(analysis *FeasibilityAnalysis) float64 {
	weights := map[string]float64{
		"technical": 0.35,
		"resource":  0.25,
		"timeline":  0.20,
		"roi":       0.20,
	}

	score := analysis.TechnicalScore*weights["technical"] +
		analysis.ResourceScore*weights["resource"] +
		analysis.TimelineScore*weights["timeline"] +
		analysis.ROIScore*weights["roi"]

	// Adjust for high-impact risks
	for _, risk := range analysis.Risks {
		if risk.Probability*risk.Impact > 0.7 {
			score *= 0.9
		}
	}

	return score
}

// generateRecommendation generates implementation recommendation
func (fa *FeasibilityAnalyzer) generateRecommendation(analysis *FeasibilityAnalysis) string {
	if analysis.OverallScore >= 0.8 {
		return "HIGHLY RECOMMENDED - Proceed with implementation immediately"
	} else if analysis.OverallScore >= 0.6 {
		return "RECOMMENDED - Proceed after risk mitigation planning"
	} else if analysis.OverallScore >= 0.4 {
		return "CONDITIONAL - Prototype first to validate feasibility"
	} else if analysis.OverallScore >= 0.2 {
		return "NOT RECOMMENDED - Monitor for future opportunities"
	}
	return "REJECT - Not feasible with current technology/resources"
}

// calculatePriority calculates implementation priority
func (fa *FeasibilityAnalyzer) calculatePriority(analysis *FeasibilityAnalysis) int {
	// Priority 1-5 (1 = highest)
	if analysis.OverallScore >= 0.8 && analysis.ROIScore >= 0.8 {
		return 1
	} else if analysis.OverallScore >= 0.7 {
		return 2
	} else if analysis.OverallScore >= 0.5 {
		return 3
	} else if analysis.OverallScore >= 0.3 {
		return 4
	}
	return 5
}

// generateNextSteps generates recommended next steps
func (fa *FeasibilityAnalyzer) generateNextSteps(analysis *FeasibilityAnalysis) []string {
	steps := make([]string, 0)

	if analysis.OverallScore >= 0.6 {
		steps = append(steps, "Form research team")
		steps = append(steps, "Secure budget approval")
		steps = append(steps, "Begin detailed design")

		if len(analysis.RequiredResources.Partnerships) > 0 {
			steps = append(steps, "Establish partnerships: "+
				fmt.Sprintf("%v", analysis.RequiredResources.Partnerships))
		}

		steps = append(steps, "Develop prototype")
		steps = append(steps, "Conduct pilot testing")
		steps = append(steps, "Plan production rollout")
	} else if analysis.OverallScore >= 0.4 {
		steps = append(steps, "Conduct feasibility study")
		steps = append(steps, "Build proof of concept")
		steps = append(steps, "Reassess after PoC results")
	} else {
		steps = append(steps, "Monitor research developments")
		steps = append(steps, "Revisit in 6 months")
	}

	return steps
}

// Helper function
func contains(text, substr string) bool {
	return len(text) > 0 && len(substr) > 0 &&
		(text == substr || len(text) > len(substr) &&
		(text[:len(substr)] == substr ||
		text[len(text)-len(substr):] == substr ||
		containsMiddle(text, substr)))
}

func containsMiddle(text, substr string) bool {
	for i := 0; i <= len(text)-len(substr); i++ {
		if text[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
