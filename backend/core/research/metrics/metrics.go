package metrics

import (
	"sync"
	"time"
)

// ResearchMetrics tracks research innovation metrics
type ResearchMetrics struct {
	// Papers
	PapersMonitored    int
	PapersIntegrated   int
	PapersPerYear      int
	AvgTimeToIntegration time.Duration
	TopPapers          []PaperMetric

	// Prototypes
	PrototypesCreated  int
	PrototypesSuccessful int
	AvgPrototypeTime   time.Duration
	PrototypeSuccessRate float64

	// Open Source
	RepositoriesCreated int
	TotalStars          int
	TotalForks          int
	TotalContributors   int
	TotalDownloads      int64
	CommunityEngagement float64

	// Academic Collaboration
	UniversityPartners  int
	ActiveProjects      int
	PublicationsCoauthored int
	ActiveStudents      int
	TotalCitations      int

	// Patents
	PatentIdeas         int
	PatentsFiled        int
	PatentsGranted      int
	PatentsPending      int
	AvgFilingTime       time.Duration
	PatentValue         int64

	// Innovation Lab
	IdeasSubmitted      int
	ExperimentsRun      int
	FeaturesShipped     int
	InnovationROI       float64

	// Technology Scouting
	TechnologiesTracked int
	StartupsTracked     int
	MAOpportunities     int
	Partnerships        int

	// Overall Impact
	ResearchInvestment  int64
	EstimatedReturn     int64
	ROI                 float64
	InnovationIndex     float64

	// Timestamps
	LastUpdated         time.Time
	ReportingPeriod     time.Duration
}

// PaperMetric tracks individual paper metrics
type PaperMetric struct {
	PaperID     string
	Title       string
	Citations   int
	Integrated  bool
	TimeToIntegration time.Duration
	Impact      float64
}

// ResearchMetricsCollector collects research metrics
type ResearchMetricsCollector struct {
	metrics     ResearchMetrics
	history     []HistoricalMetrics
	mu          sync.RWMutex
}

// HistoricalMetrics stores historical metrics
type HistoricalMetrics struct {
	Timestamp time.Time
	Metrics   ResearchMetrics
}

// NewResearchMetricsCollector creates a new metrics collector
func NewResearchMetricsCollector() *ResearchMetricsCollector {
	return &ResearchMetricsCollector{
		metrics: ResearchMetrics{
			LastUpdated:     time.Now(),
			ReportingPeriod: 30 * 24 * time.Hour, // 30 days
		},
		history: make([]HistoricalMetrics, 0),
	}
}

// UpdatePaperMetrics updates paper-related metrics
func (rmc *ResearchMetricsCollector) UpdatePaperMetrics(monitored, integrated, perYear int, avgTime time.Duration) {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	rmc.metrics.PapersMonitored = monitored
	rmc.metrics.PapersIntegrated = integrated
	rmc.metrics.PapersPerYear = perYear
	rmc.metrics.AvgTimeToIntegration = avgTime
	rmc.metrics.LastUpdated = time.Now()
}

// UpdatePrototypeMetrics updates prototype metrics
func (rmc *ResearchMetricsCollector) UpdatePrototypeMetrics(created, successful int, avgTime time.Duration) {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	rmc.metrics.PrototypesCreated = created
	rmc.metrics.PrototypesSuccessful = successful
	rmc.metrics.AvgPrototypeTime = avgTime

	if created > 0 {
		rmc.metrics.PrototypeSuccessRate = float64(successful) / float64(created)
	}

	rmc.metrics.LastUpdated = time.Now()
}

// UpdateOpenSourceMetrics updates open source metrics
func (rmc *ResearchMetricsCollector) UpdateOpenSourceMetrics(repos, stars, forks, contributors int, downloads int64) {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	rmc.metrics.RepositoriesCreated = repos
	rmc.metrics.TotalStars = stars
	rmc.metrics.TotalForks = forks
	rmc.metrics.TotalContributors = contributors
	rmc.metrics.TotalDownloads = downloads

	// Calculate community engagement
	if repos > 0 {
		rmc.metrics.CommunityEngagement = float64(stars+forks+contributors) / float64(repos)
	}

	rmc.metrics.LastUpdated = time.Now()
}

// UpdateAcademicMetrics updates academic collaboration metrics
func (rmc *ResearchMetricsCollector) UpdateAcademicMetrics(partners, projects, pubs, students, citations int) {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	rmc.metrics.UniversityPartners = partners
	rmc.metrics.ActiveProjects = projects
	rmc.metrics.PublicationsCoauthored = pubs
	rmc.metrics.ActiveStudents = students
	rmc.metrics.TotalCitations = citations
	rmc.metrics.LastUpdated = time.Now()
}

// UpdatePatentMetrics updates patent metrics
func (rmc *ResearchMetricsCollector) UpdatePatentMetrics(ideas, filed, granted, pending int, avgTime time.Duration, value int64) {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	rmc.metrics.PatentIdeas = ideas
	rmc.metrics.PatentsFiled = filed
	rmc.metrics.PatentsGranted = granted
	rmc.metrics.PatentsPending = pending
	rmc.metrics.AvgFilingTime = avgTime
	rmc.metrics.PatentValue = value
	rmc.metrics.LastUpdated = time.Now()
}

// UpdateInnovationLabMetrics updates innovation lab metrics
func (rmc *ResearchMetricsCollector) UpdateInnovationLabMetrics(ideas, experiments, features int, roi float64) {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	rmc.metrics.IdeasSubmitted = ideas
	rmc.metrics.ExperimentsRun = experiments
	rmc.metrics.FeaturesShipped = features
	rmc.metrics.InnovationROI = roi
	rmc.metrics.LastUpdated = time.Now()
}

// UpdateScoutingMetrics updates technology scouting metrics
func (rmc *ResearchMetricsCollector) UpdateScoutingMetrics(techs, startups, ma, partnerships int) {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	rmc.metrics.TechnologiesTracked = techs
	rmc.metrics.StartupsTracked = startups
	rmc.metrics.MAOpportunities = ma
	rmc.metrics.Partnerships = partnerships
	rmc.metrics.LastUpdated = time.Now()
}

// CalculateROI calculates overall research ROI
func (rmc *ResearchMetricsCollector) CalculateROI(investment, returns int64) {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	rmc.metrics.ResearchInvestment = investment
	rmc.metrics.EstimatedReturn = returns

	if investment > 0 {
		rmc.metrics.ROI = float64(returns) / float64(investment)
	}

	rmc.metrics.LastUpdated = time.Now()
}

// CalculateInnovationIndex calculates innovation index
func (rmc *ResearchMetricsCollector) CalculateInnovationIndex() float64 {
	rmc.mu.RLock()
	defer rmc.mu.RUnlock()

	// Weighted components
	index := 0.0

	// Papers (20%)
	if rmc.metrics.PapersPerYear >= 10 {
		index += 0.2
	} else {
		index += 0.2 * float64(rmc.metrics.PapersPerYear) / 10.0
	}

	// Patents (20%)
	if rmc.metrics.PatentsFiled >= 20 {
		index += 0.2
	} else {
		index += 0.2 * float64(rmc.metrics.PatentsFiled) / 20.0
	}

	// Open source (15%)
	if rmc.metrics.TotalStars >= 10000 {
		index += 0.15
	} else {
		index += 0.15 * float64(rmc.metrics.TotalStars) / 10000.0
	}

	// Academic collaboration (15%)
	if rmc.metrics.TotalCitations >= 100 {
		index += 0.15
	} else {
		index += 0.15 * float64(rmc.metrics.TotalCitations) / 100.0
	}

	// Innovation ROI (15%)
	if rmc.metrics.InnovationROI >= 10.0 {
		index += 0.15
	} else {
		index += 0.15 * rmc.metrics.InnovationROI / 10.0
	}

	// Features shipped (15%)
	if rmc.metrics.FeaturesShipped >= 20 {
		index += 0.15
	} else {
		index += 0.15 * float64(rmc.metrics.FeaturesShipped) / 20.0
	}

	rmc.metrics.InnovationIndex = index
	return index
}

// GetMetrics returns current metrics
func (rmc *ResearchMetricsCollector) GetMetrics() ResearchMetrics {
	rmc.mu.RLock()
	defer rmc.mu.RUnlock()

	return rmc.metrics
}

// TakeSnapshot takes a snapshot of current metrics
func (rmc *ResearchMetricsCollector) TakeSnapshot() {
	rmc.mu.Lock()
	defer rmc.mu.Unlock()

	snapshot := HistoricalMetrics{
		Timestamp: time.Now(),
		Metrics:   rmc.metrics,
	}

	rmc.history = append(rmc.history, snapshot)

	// Keep last 12 months
	if len(rmc.history) > 12 {
		rmc.history = rmc.history[len(rmc.history)-12:]
	}
}

// GetTrends returns metric trends
func (rmc *ResearchMetricsCollector) GetTrends() *MetricTrends {
	rmc.mu.RLock()
	defer rmc.mu.RUnlock()

	if len(rmc.history) < 2 {
		return &MetricTrends{}
	}

	current := rmc.history[len(rmc.history)-1].Metrics
	previous := rmc.history[len(rmc.history)-2].Metrics

	trends := &MetricTrends{
		PapersGrowth:     calculateGrowth(float64(previous.PapersIntegrated), float64(current.PapersIntegrated)),
		PatentsGrowth:    calculateGrowth(float64(previous.PatentsFiled), float64(current.PatentsFiled)),
		OpenSourceGrowth: calculateGrowth(float64(previous.TotalStars), float64(current.TotalStars)),
		CitationsGrowth:  calculateGrowth(float64(previous.TotalCitations), float64(current.TotalCitations)),
		ROIGrowth:        calculateGrowth(previous.ROI, current.ROI),
		IndexGrowth:      calculateGrowth(previous.InnovationIndex, current.InnovationIndex),
	}

	return trends
}

// MetricTrends tracks metric trends
type MetricTrends struct {
	PapersGrowth     float64
	PatentsGrowth    float64
	OpenSourceGrowth float64
	CitationsGrowth  float64
	ROIGrowth        float64
	IndexGrowth      float64
}

// calculateGrowth calculates growth percentage
func calculateGrowth(previous, current float64) float64 {
	if previous == 0 {
		return 0
	}
	return ((current - previous) / previous) * 100
}

// GenerateReport generates comprehensive metrics report
func (rmc *ResearchMetricsCollector) GenerateReport() *MetricsReport {
	rmc.mu.RLock()
	defer rmc.mu.RUnlock()

	report := &MetricsReport{
		GeneratedAt: time.Now(),
		Period:      rmc.metrics.ReportingPeriod,
		Metrics:     rmc.metrics,
		Trends:      rmc.GetTrends(),
		Summary:     rmc.generateSummary(),
		Achievements: rmc.identifyAchievements(),
		Gaps:        rmc.identifyGaps(),
		Recommendations: rmc.generateRecommendations(),
	}

	return report
}

// MetricsReport contains comprehensive metrics report
type MetricsReport struct {
	GeneratedAt     time.Time
	Period          time.Duration
	Metrics         ResearchMetrics
	Trends          *MetricTrends
	Summary         string
	Achievements    []Achievement
	Gaps            []Gap
	Recommendations []string
}

// Achievement represents an achievement
type Achievement struct {
	Category    string
	Description string
	Date        time.Time
}

// Gap represents a gap or area for improvement
type Gap struct {
	Category    string
	Description string
	Priority    string
	Action      string
}

// generateSummary generates executive summary
func (rmc *ResearchMetricsCollector) generateSummary() string {
	m := rmc.metrics

	return "Research Innovation Performance Summary:\n" +
		fmt.Sprintf("- Integrated %d research papers (target: 10/year)\n", m.PapersPerYear) +
		fmt.Sprintf("- Filed %d patents (target: 20/year)\n", m.PatentsFiled) +
		fmt.Sprintf("- Achieved %d GitHub stars (target: 10,000)\n", m.TotalStars) +
		fmt.Sprintf("- Generated %d citations (target: 100/year)\n", m.TotalCitations) +
		fmt.Sprintf("- Innovation ROI: %.1fx (target: 10x)\n", m.InnovationROI) +
		fmt.Sprintf("- Innovation Index: %.2f/1.0\n", m.InnovationIndex)
}

// identifyAchievements identifies key achievements
func (rmc *ResearchMetricsCollector) identifyAchievements() []Achievement {
	achievements := make([]Achievement, 0)
	m := rmc.metrics

	if m.PapersPerYear >= 10 {
		achievements = append(achievements, Achievement{
			Category:    "Research Integration",
			Description: "Met annual paper integration target",
			Date:        time.Now(),
		})
	}

	if m.PatentsFiled >= 20 {
		achievements = append(achievements, Achievement{
			Category:    "Intellectual Property",
			Description: "Met annual patent filing target",
			Date:        time.Now(),
		})
	}

	if m.TotalStars >= 10000 {
		achievements = append(achievements, Achievement{
			Category:    "Open Source",
			Description: "Achieved 10,000+ GitHub stars",
			Date:        time.Now(),
		})
	}

	if m.TotalCitations >= 100 {
		achievements = append(achievements, Achievement{
			Category:    "Academic Impact",
			Description: "Generated 100+ citations",
			Date:        time.Now(),
		})
	}

	if m.InnovationROI >= 10.0 {
		achievements = append(achievements, Achievement{
			Category:    "Financial Impact",
			Description: "Achieved 10x innovation ROI",
			Date:        time.Now(),
		})
	}

	return achievements
}

// identifyGaps identifies gaps and areas for improvement
func (rmc *ResearchMetricsCollector) identifyGaps() []Gap {
	gaps := make([]Gap, 0)
	m := rmc.metrics

	if m.PapersPerYear < 10 {
		gaps = append(gaps, Gap{
			Category:    "Research Integration",
			Description: "Below paper integration target",
			Priority:    "high",
			Action:      "Increase research monitoring and feasibility analysis",
		})
	}

	if m.PatentsFiled < 20 {
		gaps = append(gaps, Gap{
			Category:    "Intellectual Property",
			Description: "Below patent filing target",
			Priority:    "high",
			Action:      "Enhance patent idea generation and novelty detection",
		})
	}

	if m.TotalStars < 10000 {
		gaps = append(gaps, Gap{
			Category:    "Open Source",
			Description: "Below GitHub star target",
			Priority:    "medium",
			Action:      "Improve community engagement and marketing",
		})
	}

	if m.PrototypeSuccessRate < 0.7 {
		gaps = append(gaps, Gap{
			Category:    "Prototyping",
			Description: "Low prototype success rate",
			Priority:    "medium",
			Action:      "Improve feasibility analysis and prototype methodology",
		})
	}

	return gaps
}

// generateRecommendations generates recommendations
func (rmc *ResearchMetricsCollector) generateRecommendations() []string {
	recommendations := make([]string, 0)
	m := rmc.metrics

	if m.PapersPerYear < 10 {
		recommendations = append(recommendations,
			"Expand research monitoring to additional conferences and journals")
	}

	if m.PatentsFiled < 20 {
		recommendations = append(recommendations,
			"Implement quarterly patent brainstorming sessions")
	}

	if m.TotalStars < 10000 {
		recommendations = append(recommendations,
			"Launch marketing campaign for open source projects")
	}

	if m.InnovationROI < 10.0 {
		recommendations = append(recommendations,
			"Focus on high-impact, commercially viable research")
	}

	if m.ActiveStudents < 10 {
		recommendations = append(recommendations,
			"Expand university partnership and internship programs")
	}

	return recommendations
}

import "fmt"
