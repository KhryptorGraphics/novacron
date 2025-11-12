// Package lab provides comprehensive research infrastructure for DWCP innovation
// This framework manages experiments, prototypes, publications, patents, and
// university partnerships to maintain DWCP's competitive edge through continuous innovation.
//
// Target: 10+ research papers/year, 50+ patents in 10 years
package lab

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// ResearchFramework coordinates all research activities
type ResearchFramework struct {
	db                *sql.DB
	experimentManager *ExperimentManager
	prototypeTracker  *PrototypeTracker
	publicationEngine *PublicationEngine
	patentManager     *PatentManager
	partnershipCoord  *PartnershipCoordinator
	grantTracker      *GrantTracker
	metricsDashboard  *ResearchMetrics
}

// NewResearchFramework initializes research infrastructure
func NewResearchFramework(db *sql.DB) (*ResearchFramework, error) {
	return &ResearchFramework{
		db:                db,
		experimentManager: NewExperimentManager(db),
		prototypeTracker:  NewPrototypeTracker(db),
		publicationEngine: NewPublicationEngine(db),
		patentManager:     NewPatentManager(db),
		partnershipCoord:  NewPartnershipCoordinator(db),
		grantTracker:      NewGrantTracker(db),
		metricsDashboard:  NewResearchMetrics(db),
	}, nil
}

// ===========================
// Experiment Management System
// ===========================

// Experiment represents a research experiment
type Experiment struct {
	ID              string                 `json:"id"`
	Title           string                 `json:"title"`
	Hypothesis      string                 `json:"hypothesis"`
	Methodology     string                 `json:"methodology"`
	Status          ExperimentStatus       `json:"status"`
	LeadResearcher  string                 `json:"lead_researcher"`
	Team            []string               `json:"team"`
	StartDate       time.Time              `json:"start_date"`
	EndDate         *time.Time             `json:"end_date,omitempty"`
	Budget          float64                `json:"budget"` // USD
	Results         *ExperimentResults     `json:"results,omitempty"`
	Publications    []string               `json:"publications"` // Paper IDs
	Patents         []string               `json:"patents"`      // Patent IDs
	Prototypes      []string               `json:"prototypes"`   // Prototype IDs
	Metadata        map[string]interface{} `json:"metadata"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// ExperimentStatus represents experiment lifecycle stage
type ExperimentStatus string

const (
	ExperimentProposal   ExperimentStatus = "proposal"
	ExperimentApproved   ExperimentStatus = "approved"
	ExperimentInProgress ExperimentStatus = "in_progress"
	ExperimentCompleted  ExperimentStatus = "completed"
	ExperimentPublished  ExperimentStatus = "published"
	ExperimentAbandoned  ExperimentStatus = "abandoned"
)

// ExperimentResults captures experimental outcomes
type ExperimentResults struct {
	Success         bool                   `json:"success"`
	KeyFindings     []string               `json:"key_findings"`
	Data            []DataPoint            `json:"data"`
	StatSignificance float64               `json:"statistical_significance"` // p-value
	Reproducibility  float64               `json:"reproducibility"`           // 0-1 score
	Conclusions     string                 `json:"conclusions"`
	FutureWork      []string               `json:"future_work"`
	Metrics         map[string]float64     `json:"metrics"`
	Artifacts       []string               `json:"artifacts"` // URLs to data, code, models
	CreatedAt       time.Time              `json:"created_at"`
}

// DataPoint represents a single measurement
type DataPoint struct {
	Timestamp   time.Time              `json:"timestamp"`
	Measurement string                 `json:"measurement"`
	Value       float64                `json:"value"`
	Unit        string                 `json:"unit"`
	Confidence  float64                `json:"confidence"` // 0-1
	Metadata    map[string]interface{} `json:"metadata"`
}

// ExperimentManager handles experiment lifecycle
type ExperimentManager struct {
	db *sql.DB
}

// NewExperimentManager creates experiment manager
func NewExperimentManager(db *sql.DB) *ExperimentManager {
	return &ExperimentManager{db: db}
}

// CreateExperiment proposes new research experiment
func (em *ExperimentManager) CreateExperiment(ctx context.Context, exp *Experiment) error {
	exp.ID = uuid.New().String()
	exp.Status = ExperimentProposal
	exp.CreatedAt = time.Now()
	exp.UpdatedAt = time.Now()

	query := `
		INSERT INTO experiments
		(id, title, hypothesis, methodology, status, lead_researcher, team,
		 start_date, budget, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
	`

	teamJSON, _ := json.Marshal(exp.Team)
	metadataJSON, _ := json.Marshal(exp.Metadata)

	_, err := em.db.ExecContext(ctx, query,
		exp.ID, exp.Title, exp.Hypothesis, exp.Methodology, exp.Status,
		exp.LeadResearcher, teamJSON, exp.StartDate, exp.Budget,
		metadataJSON, exp.CreatedAt, exp.UpdatedAt,
	)

	return err
}

// UpdateExperimentStatus transitions experiment through lifecycle
func (em *ExperimentManager) UpdateExperimentStatus(ctx context.Context, expID string, status ExperimentStatus) error {
	query := `UPDATE experiments SET status = $1, updated_at = $2 WHERE id = $3`
	_, err := em.db.ExecContext(ctx, query, status, time.Now(), expID)
	return err
}

// RecordResults captures experimental outcomes
func (em *ExperimentManager) RecordResults(ctx context.Context, expID string, results *ExperimentResults) error {
	results.CreatedAt = time.Now()
	resultsJSON, _ := json.Marshal(results)

	query := `UPDATE experiments SET results = $1, status = $2, end_date = $3, updated_at = $4 WHERE id = $5`
	_, err := em.db.ExecContext(ctx, query, resultsJSON, ExperimentCompleted, time.Now(), time.Now(), expID)

	return err
}

// ListExperiments retrieves experiments with filters
func (em *ExperimentManager) ListExperiments(ctx context.Context, filters ExperimentFilters) ([]*Experiment, error) {
	query := `
		SELECT id, title, hypothesis, methodology, status, lead_researcher,
		       team, start_date, end_date, budget, results, publications,
		       patents, prototypes, metadata, created_at, updated_at
		FROM experiments
		WHERE 1=1
	`

	args := []interface{}{}
	argIdx := 1

	if filters.Status != "" {
		query += fmt.Sprintf(" AND status = $%d", argIdx)
		args = append(args, filters.Status)
		argIdx++
	}

	if filters.LeadResearcher != "" {
		query += fmt.Sprintf(" AND lead_researcher = $%d", argIdx)
		args = append(args, filters.LeadResearcher)
		argIdx++
	}

	if !filters.StartDate.IsZero() {
		query += fmt.Sprintf(" AND start_date >= $%d", argIdx)
		args = append(args, filters.StartDate)
		argIdx++
	}

	query += " ORDER BY created_at DESC LIMIT 100"

	rows, err := em.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	experiments := []*Experiment{}
	for rows.Next() {
		exp := &Experiment{}
		var teamJSON, resultsJSON, publicationsJSON, patentsJSON, prototypesJSON, metadataJSON []byte
		var endDate *time.Time

		err := rows.Scan(
			&exp.ID, &exp.Title, &exp.Hypothesis, &exp.Methodology, &exp.Status,
			&exp.LeadResearcher, &teamJSON, &exp.StartDate, &endDate, &exp.Budget,
			&resultsJSON, &publicationsJSON, &patentsJSON, &prototypesJSON,
			&metadataJSON, &exp.CreatedAt, &exp.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}

		exp.EndDate = endDate
		json.Unmarshal(teamJSON, &exp.Team)
		json.Unmarshal(resultsJSON, &exp.Results)
		json.Unmarshal(publicationsJSON, &exp.Publications)
		json.Unmarshal(patentsJSON, &exp.Patents)
		json.Unmarshal(prototypesJSON, &exp.Prototypes)
		json.Unmarshal(metadataJSON, &exp.Metadata)

		experiments = append(experiments, exp)
	}

	return experiments, nil
}

// ExperimentFilters for querying experiments
type ExperimentFilters struct {
	Status         ExperimentStatus
	LeadResearcher string
	StartDate      time.Time
	BudgetMin      float64
	BudgetMax      float64
}

// ===========================
// Prototype Tracking System
// ===========================

// Prototype represents a research prototype
type Prototype struct {
	ID             string           `json:"id"`
	Name           string           `json:"name"`
	Description    string           `json:"description"`
	Version        string           `json:"version"`
	Status         PrototypeStatus  `json:"status"`
	ExperimentID   string           `json:"experiment_id"`
	LeadEngineer   string           `json:"lead_engineer"`
	Team           []string         `json:"team"`
	CodeRepository string           `json:"code_repository"` // GitHub URL
	Documentation  string           `json:"documentation"`   // URL
	Performance    PerformanceData  `json:"performance"`
	Deployments    []Deployment     `json:"deployments"`
	TransitionPath *ProductionPath  `json:"transition_path,omitempty"`
	CreatedAt      time.Time        `json:"created_at"`
	UpdatedAt      time.Time        `json:"updated_at"`
}

// PrototypeStatus represents prototype maturity
type PrototypeStatus string

const (
	PrototypeConceptual     PrototypeStatus = "conceptual"
	PrototypeAlpha          PrototypeStatus = "alpha"
	PrototypeBeta           PrototypeStatus = "beta"
	PrototypeProductionReady PrototypeStatus = "production_ready"
	PrototypeIntegrated     PrototypeStatus = "integrated"
	PrototypeDeprecated     PrototypeStatus = "deprecated"
)

// PerformanceData captures prototype benchmarks
type PerformanceData struct {
	Latency       time.Duration   `json:"latency"`         // P99 latency
	Throughput    float64         `json:"throughput"`      // ops/sec
	Accuracy      float64         `json:"accuracy"`        // 0-1
	Scalability   ScalabilityTest `json:"scalability"`
	EnergyEff     float64         `json:"energy_efficiency"` // ops/joule
	BenchmarkDate time.Time       `json:"benchmark_date"`
}

// ScalabilityTest captures horizontal scaling behavior
type ScalabilityTest struct {
	MinNodes     int     `json:"min_nodes"`
	MaxNodes     int     `json:"max_nodes"`
	Efficiency   float64 `json:"efficiency"`    // 0-1 (perfect = 1.0)
	BottleneckID string  `json:"bottleneck_id"` // Resource bottleneck
}

// Deployment represents prototype deployment instance
type Deployment struct {
	Environment string    `json:"environment"` // "staging", "production", "pilot"
	Hosts       []string  `json:"hosts"`
	Traffic     float64   `json:"traffic"`     // Percentage (0-100)
	StartDate   time.Time `json:"start_date"`
	EndDate     *time.Time `json:"end_date,omitempty"`
	Metrics     DeploymentMetrics `json:"metrics"`
}

// DeploymentMetrics tracks deployment health
type DeploymentMetrics struct {
	Uptime       float64   `json:"uptime"`        // Percentage
	ErrorRate    float64   `json:"error_rate"`    // Errors per request
	P99Latency   time.Duration `json:"p99_latency"`
	Requests     int64     `json:"requests"`
	LastUpdated  time.Time `json:"last_updated"`
}

// ProductionPath defines transition to production
type ProductionPath struct {
	Milestones   []Milestone `json:"milestones"`
	EstimatedETA time.Time   `json:"estimated_eta"`
	Confidence   float64     `json:"confidence"` // 0-1
	Blockers     []string    `json:"blockers"`
	Owner        string      `json:"owner"`
}

// Milestone represents production transition step
type Milestone struct {
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Completed   bool      `json:"completed"`
	DueDate     time.Time `json:"due_date"`
	Owner       string    `json:"owner"`
}

// PrototypeTracker manages prototype lifecycle
type PrototypeTracker struct {
	db *sql.DB
}

// NewPrototypeTracker creates prototype tracker
func NewPrototypeTracker(db *sql.DB) *PrototypeTracker {
	return &PrototypeTracker{db: db}
}

// CreatePrototype registers new prototype
func (pt *PrototypeTracker) CreatePrototype(ctx context.Context, proto *Prototype) error {
	proto.ID = uuid.New().String()
	proto.Status = PrototypeConceptual
	proto.CreatedAt = time.Now()
	proto.UpdatedAt = time.Now()

	query := `
		INSERT INTO prototypes
		(id, name, description, version, status, experiment_id, lead_engineer,
		 team, code_repository, documentation, performance, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
	`

	teamJSON, _ := json.Marshal(proto.Team)
	performanceJSON, _ := json.Marshal(proto.Performance)

	_, err := pt.db.ExecContext(ctx, query,
		proto.ID, proto.Name, proto.Description, proto.Version, proto.Status,
		proto.ExperimentID, proto.LeadEngineer, teamJSON, proto.CodeRepository,
		proto.Documentation, performanceJSON, proto.CreatedAt, proto.UpdatedAt,
	)

	return err
}

// UpdatePerformance records benchmark results
func (pt *PrototypeTracker) UpdatePerformance(ctx context.Context, protoID string, perf PerformanceData) error {
	perf.BenchmarkDate = time.Now()
	perfJSON, _ := json.Marshal(perf)

	query := `UPDATE prototypes SET performance = $1, updated_at = $2 WHERE id = $3`
	_, err := pt.db.ExecContext(ctx, query, perfJSON, time.Now(), protoID)

	return err
}

// AddDeployment records prototype deployment
func (pt *PrototypeTracker) AddDeployment(ctx context.Context, protoID string, deployment Deployment) error {
	// Fetch current deployments
	proto, err := pt.GetPrototype(ctx, protoID)
	if err != nil {
		return err
	}

	proto.Deployments = append(proto.Deployments, deployment)
	deploymentsJSON, _ := json.Marshal(proto.Deployments)

	query := `UPDATE prototypes SET deployments = $1, updated_at = $2 WHERE id = $3`
	_, err = pt.db.ExecContext(ctx, query, deploymentsJSON, time.Now(), protoID)

	return err
}

// DefineProductionPath creates transition plan
func (pt *PrototypeTracker) DefineProductionPath(ctx context.Context, protoID string, path *ProductionPath) error {
	pathJSON, _ := json.Marshal(path)

	query := `UPDATE prototypes SET transition_path = $1, updated_at = $2 WHERE id = $3`
	_, err := pt.db.ExecContext(ctx, query, pathJSON, time.Now(), protoID)

	return err
}

// GetPrototype retrieves prototype by ID
func (pt *PrototypeTracker) GetPrototype(ctx context.Context, protoID string) (*Prototype, error) {
	query := `
		SELECT id, name, description, version, status, experiment_id, lead_engineer,
		       team, code_repository, documentation, performance, deployments,
		       transition_path, created_at, updated_at
		FROM prototypes WHERE id = $1
	`

	proto := &Prototype{}
	var teamJSON, performanceJSON, deploymentsJSON, transitionPathJSON []byte

	err := pt.db.QueryRowContext(ctx, query, protoID).Scan(
		&proto.ID, &proto.Name, &proto.Description, &proto.Version, &proto.Status,
		&proto.ExperimentID, &proto.LeadEngineer, &teamJSON, &proto.CodeRepository,
		&proto.Documentation, &performanceJSON, &deploymentsJSON,
		&transitionPathJSON, &proto.CreatedAt, &proto.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}

	json.Unmarshal(teamJSON, &proto.Team)
	json.Unmarshal(performanceJSON, &proto.Performance)
	json.Unmarshal(deploymentsJSON, &proto.Deployments)
	json.Unmarshal(transitionPathJSON, &proto.TransitionPath)

	return proto, nil
}

// ListPrototypes retrieves prototypes with filters
func (pt *PrototypeTracker) ListPrototypes(ctx context.Context, status PrototypeStatus) ([]*Prototype, error) {
	query := `
		SELECT id, name, description, version, status, experiment_id, lead_engineer,
		       team, code_repository, documentation, performance, deployments,
		       transition_path, created_at, updated_at
		FROM prototypes
		WHERE status = $1
		ORDER BY created_at DESC
		LIMIT 100
	`

	rows, err := pt.db.QueryContext(ctx, query, status)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	prototypes := []*Prototype{}
	for rows.Next() {
		proto := &Prototype{}
		var teamJSON, performanceJSON, deploymentsJSON, transitionPathJSON []byte

		err := rows.Scan(
			&proto.ID, &proto.Name, &proto.Description, &proto.Version, &proto.Status,
			&proto.ExperimentID, &proto.LeadEngineer, &teamJSON, &proto.CodeRepository,
			&proto.Documentation, &performanceJSON, &deploymentsJSON,
			&transitionPathJSON, &proto.CreatedAt, &proto.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}

		json.Unmarshal(teamJSON, &proto.Team)
		json.Unmarshal(performanceJSON, &proto.Performance)
		json.Unmarshal(deploymentsJSON, &proto.Deployments)
		json.Unmarshal(transitionPathJSON, &proto.TransitionPath)

		prototypes = append(prototypes, proto)
	}

	return prototypes, nil
}

// ===========================
// Publication Engine
// ===========================

// Publication represents research paper
type Publication struct {
	ID            string            `json:"id"`
	Title         string            `json:"title"`
	Abstract      string            `json:"abstract"`
	Authors       []Author          `json:"authors"`
	Venue         string            `json:"venue"`        // Conference or journal
	Type          PublicationType   `json:"type"`
	Status        PublicationStatus `json:"status"`
	SubmissionDate time.Time        `json:"submission_date"`
	PublishDate   *time.Time        `json:"publish_date,omitempty"`
	DOI           string            `json:"doi,omitempty"`
	ArxivID       string            `json:"arxiv_id,omitempty"`
	PDF           string            `json:"pdf"` // URL
	Code          string            `json:"code"` // GitHub repo
	Data          string            `json:"data"` // Dataset URL
	Citations     int               `json:"citations"`
	ImpactScore   float64           `json:"impact_score"` // Citation-based
	ExperimentIDs []string          `json:"experiment_ids"`
	CreatedAt     time.Time         `json:"created_at"`
	UpdatedAt     time.Time         `json:"updated_at"`
}

// Author represents paper author
type Author struct {
	Name         string   `json:"name"`
	Affiliation  string   `json:"affiliation"`
	Email        string   `json:"email"`
	ORCID        string   `json:"orcid,omitempty"`
	IsCorresponding bool  `json:"is_corresponding"`
	Contribution string   `json:"contribution"` // Author's role
}

// PublicationType categorizes papers
type PublicationType string

const (
	PubConference  PublicationType = "conference"
	PubJournal     PublicationType = "journal"
	PubWorkshop    PublicationType = "workshop"
	PubArxiv       PublicationType = "arxiv"
	PubTechReport  PublicationType = "technical_report"
)

// PublicationStatus tracks publication lifecycle
type PublicationStatus string

const (
	PubDraft      PublicationStatus = "draft"
	PubSubmitted  PublicationStatus = "submitted"
	PubUnderReview PublicationStatus = "under_review"
	PubRevision    PublicationStatus = "revision"
	PubAccepted    PublicationStatus = "accepted"
	PubPublished   PublicationStatus = "published"
	PubRejected    PublicationStatus = "rejected"
)

// PublicationEngine manages research publications
type PublicationEngine struct {
	db *sql.DB
}

// NewPublicationEngine creates publication engine
func NewPublicationEngine(db *sql.DB) *PublicationEngine {
	return &PublicationEngine{db: db}
}

// CreatePublication registers new paper
func (pe *PublicationEngine) CreatePublication(ctx context.Context, pub *Publication) error {
	pub.ID = uuid.New().String()
	pub.Status = PubDraft
	pub.CreatedAt = time.Now()
	pub.UpdatedAt = time.Now()

	query := `
		INSERT INTO publications
		(id, title, abstract, authors, venue, type, status, submission_date,
		 pdf, code, data, citations, impact_score, experiment_ids, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
	`

	authorsJSON, _ := json.Marshal(pub.Authors)
	experimentIDsJSON, _ := json.Marshal(pub.ExperimentIDs)

	_, err := pe.db.ExecContext(ctx, query,
		pub.ID, pub.Title, pub.Abstract, authorsJSON, pub.Venue, pub.Type,
		pub.Status, pub.SubmissionDate, pub.PDF, pub.Code, pub.Data,
		pub.Citations, pub.ImpactScore, experimentIDsJSON, pub.CreatedAt, pub.UpdatedAt,
	)

	return err
}

// UpdatePublicationStatus transitions publication lifecycle
func (pe *PublicationEngine) UpdatePublicationStatus(ctx context.Context, pubID string, status PublicationStatus) error {
	query := `UPDATE publications SET status = $1, updated_at = $2 WHERE id = $3`
	_, err := pe.db.ExecContext(ctx, query, status, time.Now(), pubID)
	return err
}

// RecordPublication records accepted/published paper
func (pe *PublicationEngine) RecordPublication(ctx context.Context, pubID string, publishDate time.Time, doi string) error {
	query := `UPDATE publications SET status = $1, publish_date = $2, doi = $3, updated_at = $4 WHERE id = $5`
	_, err := pe.db.ExecContext(ctx, query, PubPublished, publishDate, doi, time.Now(), pubID)
	return err
}

// UpdateCitations updates citation count
func (pe *PublicationEngine) UpdateCitations(ctx context.Context, pubID string, citations int) error {
	// Calculate impact score (simplified: citations / years since publication)
	pub, err := pe.GetPublication(ctx, pubID)
	if err != nil {
		return err
	}

	if pub.PublishDate != nil {
		yearsSincePublication := time.Since(*pub.PublishDate).Hours() / (24 * 365)
		if yearsSincePublication > 0 {
			pub.ImpactScore = float64(citations) / yearsSincePublication
		}
	}

	query := `UPDATE publications SET citations = $1, impact_score = $2, updated_at = $3 WHERE id = $4`
	_, err = pe.db.ExecContext(ctx, query, citations, pub.ImpactScore, time.Now(), pubID)

	return err
}

// GetPublication retrieves publication by ID
func (pe *PublicationEngine) GetPublication(ctx context.Context, pubID string) (*Publication, error) {
	query := `
		SELECT id, title, abstract, authors, venue, type, status, submission_date,
		       publish_date, doi, arxiv_id, pdf, code, data, citations, impact_score,
		       experiment_ids, created_at, updated_at
		FROM publications WHERE id = $1
	`

	pub := &Publication{}
	var authorsJSON, experimentIDsJSON []byte
	var publishDate *time.Time

	err := pe.db.QueryRowContext(ctx, query, pubID).Scan(
		&pub.ID, &pub.Title, &pub.Abstract, &authorsJSON, &pub.Venue, &pub.Type,
		&pub.Status, &pub.SubmissionDate, &publishDate, &pub.DOI, &pub.ArxivID,
		&pub.PDF, &pub.Code, &pub.Data, &pub.Citations, &pub.ImpactScore,
		&experimentIDsJSON, &pub.CreatedAt, &pub.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}

	pub.PublishDate = publishDate
	json.Unmarshal(authorsJSON, &pub.Authors)
	json.Unmarshal(experimentIDsJSON, &pub.ExperimentIDs)

	return pub, nil
}

// ListPublications retrieves publications with filters
func (pe *PublicationEngine) ListPublications(ctx context.Context, status PublicationStatus, limit int) ([]*Publication, error) {
	query := `
		SELECT id, title, abstract, authors, venue, type, status, submission_date,
		       publish_date, doi, arxiv_id, pdf, code, data, citations, impact_score,
		       experiment_ids, created_at, updated_at
		FROM publications
		WHERE status = $1
		ORDER BY impact_score DESC
		LIMIT $2
	`

	rows, err := pe.db.QueryContext(ctx, query, status, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	publications := []*Publication{}
	for rows.Next() {
		pub := &Publication{}
		var authorsJSON, experimentIDsJSON []byte
		var publishDate *time.Time

		err := rows.Scan(
			&pub.ID, &pub.Title, &pub.Abstract, &authorsJSON, &pub.Venue, &pub.Type,
			&pub.Status, &pub.SubmissionDate, &publishDate, &pub.DOI, &pub.ArxivID,
			&pub.PDF, &pub.Code, &pub.Data, &pub.Citations, &pub.ImpactScore,
			&experimentIDsJSON, &pub.CreatedAt, &pub.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}

		pub.PublishDate = publishDate
		json.Unmarshal(authorsJSON, &pub.Authors)
		json.Unmarshal(experimentIDsJSON, &pub.ExperimentIDs)

		publications = append(publications, pub)
	}

	return publications, nil
}

// ===========================
// Patent Management System
// ===========================

// Patent represents intellectual property
type Patent struct {
	ID              string        `json:"id"`
	Title           string        `json:"title"`
	Abstract        string        `json:"abstract"`
	Inventors       []Inventor    `json:"inventors"`
	Assignee        string        `json:"assignee"` // "NovaCron Inc."
	Status          PatentStatus  `json:"status"`
	ApplicationDate time.Time     `json:"application_date"`
	GrantDate       *time.Time    `json:"grant_date,omitempty"`
	PatentNumber    string        `json:"patent_number,omitempty"`
	Country         string        `json:"country"` // "US", "EP", "CN", etc.
	Claims          []string      `json:"claims"`
	PriorArt        []string      `json:"prior_art"` // Referenced patents
	Licensing       *LicenseInfo  `json:"licensing,omitempty"`
	ExperimentIDs   []string      `json:"experiment_ids"`
	PrototypeIDs    []string      `json:"prototype_ids"`
	Revenue         float64       `json:"revenue"` // Licensing revenue
	CreatedAt       time.Time     `json:"created_at"`
	UpdatedAt       time.Time     `json:"updated_at"`
}

// Inventor represents patent inventor
type Inventor struct {
	Name        string `json:"name"`
	Affiliation string `json:"affiliation"`
	Email       string `json:"email"`
	Country     string `json:"country"`
}

// PatentStatus tracks patent lifecycle
type PatentStatus string

const (
	PatentDraft     PatentStatus = "draft"
	PatentFiled     PatentStatus = "filed"
	PatentPending   PatentStatus = "pending"
	PatentGranted   PatentStatus = "granted"
	PatentRejected  PatentStatus = "rejected"
	PatentAbandoned PatentStatus = "abandoned"
	PatentExpired   PatentStatus = "expired"
)

// LicenseInfo captures licensing terms
type LicenseInfo struct {
	Type       string    `json:"type"` // "exclusive", "non-exclusive", "open"
	Licensees  []string  `json:"licensees"`
	StartDate  time.Time `json:"start_date"`
	EndDate    time.Time `json:"end_date"`
	Royalty    float64   `json:"royalty"` // Percentage
	TotalRevenue float64 `json:"total_revenue"`
}

// PatentManager handles patent portfolio
type PatentManager struct {
	db *sql.DB
}

// NewPatentManager creates patent manager
func NewPatentManager(db *sql.DB) *PatentManager {
	return &PatentManager{db: db}
}

// CreatePatent registers new patent application
func (pm *PatentManager) CreatePatent(ctx context.Context, patent *Patent) error {
	patent.ID = uuid.New().String()
	patent.Status = PatentDraft
	patent.CreatedAt = time.Now()
	patent.UpdatedAt = time.Now()

	query := `
		INSERT INTO patents
		(id, title, abstract, inventors, assignee, status, application_date,
		 country, claims, prior_art, experiment_ids, prototype_ids, revenue,
		 created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
	`

	inventorsJSON, _ := json.Marshal(patent.Inventors)
	claimsJSON, _ := json.Marshal(patent.Claims)
	priorArtJSON, _ := json.Marshal(patent.PriorArt)
	experimentIDsJSON, _ := json.Marshal(patent.ExperimentIDs)
	prototypeIDsJSON, _ := json.Marshal(patent.PrototypeIDs)

	_, err := pm.db.ExecContext(ctx, query,
		patent.ID, patent.Title, patent.Abstract, inventorsJSON, patent.Assignee,
		patent.Status, patent.ApplicationDate, patent.Country, claimsJSON,
		priorArtJSON, experimentIDsJSON, prototypeIDsJSON, patent.Revenue,
		patent.CreatedAt, patent.UpdatedAt,
	)

	return err
}

// UpdatePatentStatus transitions patent lifecycle
func (pm *PatentManager) UpdatePatentStatus(ctx context.Context, patentID string, status PatentStatus) error {
	query := `UPDATE patents SET status = $1, updated_at = $2 WHERE id = $3`
	_, err := pm.db.ExecContext(ctx, query, status, time.Now(), patentID)
	return err
}

// RecordGrant records patent grant
func (pm *PatentManager) RecordGrant(ctx context.Context, patentID string, grantDate time.Time, patentNumber string) error {
	query := `UPDATE patents SET status = $1, grant_date = $2, patent_number = $3, updated_at = $4 WHERE id = $5`
	_, err := pm.db.ExecContext(ctx, query, PatentGranted, grantDate, patentNumber, time.Now(), patentID)
	return err
}

// UpdateRevenue records licensing revenue
func (pm *PatentManager) UpdateRevenue(ctx context.Context, patentID string, revenue float64) error {
	query := `UPDATE patents SET revenue = $1, updated_at = $2 WHERE id = $3`
	_, err := pm.db.ExecContext(ctx, query, revenue, time.Now(), patentID)
	return err
}

// GetPatent retrieves patent by ID
func (pm *PatentManager) GetPatent(ctx context.Context, patentID string) (*Patent, error) {
	query := `
		SELECT id, title, abstract, inventors, assignee, status, application_date,
		       grant_date, patent_number, country, claims, prior_art, licensing,
		       experiment_ids, prototype_ids, revenue, created_at, updated_at
		FROM patents WHERE id = $1
	`

	patent := &Patent{}
	var inventorsJSON, claimsJSON, priorArtJSON, licensingJSON, experimentIDsJSON, prototypeIDsJSON []byte
	var grantDate *time.Time

	err := pm.db.QueryRowContext(ctx, query, patentID).Scan(
		&patent.ID, &patent.Title, &patent.Abstract, &inventorsJSON, &patent.Assignee,
		&patent.Status, &patent.ApplicationDate, &grantDate, &patent.PatentNumber,
		&patent.Country, &claimsJSON, &priorArtJSON, &licensingJSON,
		&experimentIDsJSON, &prototypeIDsJSON, &patent.Revenue, &patent.CreatedAt, &patent.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}

	patent.GrantDate = grantDate
	json.Unmarshal(inventorsJSON, &patent.Inventors)
	json.Unmarshal(claimsJSON, &patent.Claims)
	json.Unmarshal(priorArtJSON, &patent.PriorArt)
	json.Unmarshal(licensingJSON, &patent.Licensing)
	json.Unmarshal(experimentIDsJSON, &patent.ExperimentIDs)
	json.Unmarshal(prototypeIDsJSON, &patent.PrototypeIDs)

	return patent, nil
}

// ListPatents retrieves patents with filters
func (pm *PatentManager) ListPatents(ctx context.Context, status PatentStatus) ([]*Patent, error) {
	query := `
		SELECT id, title, abstract, inventors, assignee, status, application_date,
		       grant_date, patent_number, country, claims, prior_art, licensing,
		       experiment_ids, prototype_ids, revenue, created_at, updated_at
		FROM patents
		WHERE status = $1
		ORDER BY application_date DESC
		LIMIT 100
	`

	rows, err := pm.db.QueryContext(ctx, query, status)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	patents := []*Patent{}
	for rows.Next() {
		patent := &Patent{}
		var inventorsJSON, claimsJSON, priorArtJSON, licensingJSON, experimentIDsJSON, prototypeIDsJSON []byte
		var grantDate *time.Time

		err := rows.Scan(
			&patent.ID, &patent.Title, &patent.Abstract, &inventorsJSON, &patent.Assignee,
			&patent.Status, &patent.ApplicationDate, &grantDate, &patent.PatentNumber,
			&patent.Country, &claimsJSON, &priorArtJSON, &licensingJSON,
			&experimentIDsJSON, &prototypeIDsJSON, &patent.Revenue, &patent.CreatedAt, &patent.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}

		patent.GrantDate = grantDate
		json.Unmarshal(inventorsJSON, &patent.Inventors)
		json.Unmarshal(claimsJSON, &patent.Claims)
		json.Unmarshal(priorArtJSON, &patent.PriorArt)
		json.Unmarshal(licensingJSON, &patent.Licensing)
		json.Unmarshal(experimentIDsJSON, &patent.ExperimentIDs)
		json.Unmarshal(prototypeIDsJSON, &patent.PrototypeIDs)

		patents = append(patents, patent)
	}

	return patents, nil
}

// ===========================
// Partnership Coordination
// ===========================

// AcademicPartnership represents university collaboration
type AcademicPartnership struct {
	ID            string              `json:"id"`
	University    string              `json:"university"`
	Department    string              `json:"department"`
	Type          PartnershipType     `json:"type"`
	Status        PartnershipStatus   `json:"status"`
	Contacts      []Contact           `json:"contacts"`
	ResearchAreas []string            `json:"research_areas"`
	Funding       float64             `json:"funding"` // USD per year
	PhDStudents   int                 `json:"phd_students"` // Sponsored
	JointProjects []string            `json:"joint_projects"` // Experiment IDs
	Publications  []string            `json:"publications"` // Publication IDs
	StartDate     time.Time           `json:"start_date"`
	EndDate       *time.Time          `json:"end_date,omitempty"`
	ROI           PartnershipROI      `json:"roi"`
	CreatedAt     time.Time           `json:"created_at"`
	UpdatedAt     time.Time           `json:"updated_at"`
}

// Contact represents partnership contact
type Contact struct {
	Name        string `json:"name"`
	Title       string `json:"title"`
	Email       string `json:"email"`
	Phone       string `json:"phone,omitempty"`
	IsPrimary   bool   `json:"is_primary"`
}

// PartnershipType categorizes partnerships
type PartnershipType string

const (
	PartnershipAcademic  PartnershipType = "academic"
	PartnershipIndustry  PartnershipType = "industry"
	PartnershipGovernment PartnershipType = "government"
	PartnershipStandards  PartnershipType = "standards"
)

// PartnershipStatus tracks partnership health
type PartnershipStatus string

const (
	PartnershipProposed  PartnershipStatus = "proposed"
	PartnershipActive    PartnershipStatus = "active"
	PartnershipRenewal   PartnershipStatus = "renewal"
	PartnershipExpired   PartnershipStatus = "expired"
	PartnershipTerminated PartnershipStatus = "terminated"
)

// PartnershipROI measures partnership value
type PartnershipROI struct {
	PublicationsCount int     `json:"publications_count"`
	PatentsCount      int     `json:"patents_count"`
	PrototypesCount   int     `json:"prototypes_count"`
	TalentAcquired    int     `json:"talent_acquired"` // Hired PhD students
	TotalValue        float64 `json:"total_value"` // Estimated value (USD)
	ROIRatio          float64 `json:"roi_ratio"` // (Value - Funding) / Funding
	LastUpdated       time.Time `json:"last_updated"`
}

// PartnershipCoordinator manages university partnerships
type PartnershipCoordinator struct {
	db *sql.DB
}

// NewPartnershipCoordinator creates partnership coordinator
func NewPartnershipCoordinator(db *sql.DB) *PartnershipCoordinator {
	return &PartnershipCoordinator{db: db}
}

// CreatePartnership establishes new partnership
func (pc *PartnershipCoordinator) CreatePartnership(ctx context.Context, partnership *AcademicPartnership) error {
	partnership.ID = uuid.New().String()
	partnership.Status = PartnershipProposed
	partnership.CreatedAt = time.Now()
	partnership.UpdatedAt = time.Now()

	query := `
		INSERT INTO partnerships
		(id, university, department, type, status, contacts, research_areas,
		 funding, phd_students, joint_projects, publications, start_date,
		 roi, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
	`

	contactsJSON, _ := json.Marshal(partnership.Contacts)
	researchAreasJSON, _ := json.Marshal(partnership.ResearchAreas)
	jointProjectsJSON, _ := json.Marshal(partnership.JointProjects)
	publicationsJSON, _ := json.Marshal(partnership.Publications)
	roiJSON, _ := json.Marshal(partnership.ROI)

	_, err := pc.db.ExecContext(ctx, query,
		partnership.ID, partnership.University, partnership.Department, partnership.Type,
		partnership.Status, contactsJSON, researchAreasJSON, partnership.Funding,
		partnership.PhDStudents, jointProjectsJSON, publicationsJSON,
		partnership.StartDate, roiJSON, partnership.CreatedAt, partnership.UpdatedAt,
	)

	return err
}

// UpdatePartnershipROI calculates partnership value
func (pc *PartnershipCoordinator) UpdatePartnershipROI(ctx context.Context, partnershipID string) error {
	partnership, err := pc.GetPartnership(ctx, partnershipID)
	if err != nil {
		return err
	}

	// Calculate ROI based on outputs
	roi := PartnershipROI{
		PublicationsCount: len(partnership.Publications),
		PatentsCount:      0, // Count patents from joint projects
		PrototypesCount:   0, // Count prototypes from joint projects
		TalentAcquired:    0, // Count hired PhDs
		LastUpdated:       time.Now(),
	}

	// Estimate value ($1M per paper, $10M per patent, $5M per prototype, $500K per hire)
	roi.TotalValue = float64(roi.PublicationsCount)*1e6 +
		float64(roi.PatentsCount)*10e6 +
		float64(roi.PrototypesCount)*5e6 +
		float64(roi.TalentAcquired)*500e3

	// Calculate ROI ratio
	totalFunding := partnership.Funding * float64(time.Since(partnership.StartDate).Hours()/8760) // Years
	if totalFunding > 0 {
		roi.ROIRatio = (roi.TotalValue - totalFunding) / totalFunding
	}

	roiJSON, _ := json.Marshal(roi)
	query := `UPDATE partnerships SET roi = $1, updated_at = $2 WHERE id = $3`
	_, err = pc.db.ExecContext(ctx, query, roiJSON, time.Now(), partnershipID)

	return err
}

// GetPartnership retrieves partnership by ID
func (pc *PartnershipCoordinator) GetPartnership(ctx context.Context, partnershipID string) (*AcademicPartnership, error) {
	query := `
		SELECT id, university, department, type, status, contacts, research_areas,
		       funding, phd_students, joint_projects, publications, start_date,
		       end_date, roi, created_at, updated_at
		FROM partnerships WHERE id = $1
	`

	partnership := &AcademicPartnership{}
	var contactsJSON, researchAreasJSON, jointProjectsJSON, publicationsJSON, roiJSON []byte
	var endDate *time.Time

	err := pc.db.QueryRowContext(ctx, query, partnershipID).Scan(
		&partnership.ID, &partnership.University, &partnership.Department, &partnership.Type,
		&partnership.Status, &contactsJSON, &researchAreasJSON, &partnership.Funding,
		&partnership.PhDStudents, &jointProjectsJSON, &publicationsJSON,
		&partnership.StartDate, &endDate, &roiJSON, &partnership.CreatedAt, &partnership.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}

	partnership.EndDate = endDate
	json.Unmarshal(contactsJSON, &partnership.Contacts)
	json.Unmarshal(researchAreasJSON, &partnership.ResearchAreas)
	json.Unmarshal(jointProjectsJSON, &partnership.JointProjects)
	json.Unmarshal(publicationsJSON, &partnership.Publications)
	json.Unmarshal(roiJSON, &partnership.ROI)

	return partnership, nil
}

// ListPartnerships retrieves partnerships with filters
func (pc *PartnershipCoordinator) ListPartnerships(ctx context.Context, status PartnershipStatus) ([]*AcademicPartnership, error) {
	query := `
		SELECT id, university, department, type, status, contacts, research_areas,
		       funding, phd_students, joint_projects, publications, start_date,
		       end_date, roi, created_at, updated_at
		FROM partnerships
		WHERE status = $1
		ORDER BY roi->>'roi_ratio' DESC
		LIMIT 50
	`

	rows, err := pc.db.QueryContext(ctx, query, status)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	partnerships := []*AcademicPartnership{}
	for rows.Next() {
		partnership := &AcademicPartnership{}
		var contactsJSON, researchAreasJSON, jointProjectsJSON, publicationsJSON, roiJSON []byte
		var endDate *time.Time

		err := rows.Scan(
			&partnership.ID, &partnership.University, &partnership.Department, &partnership.Type,
			&partnership.Status, &contactsJSON, &researchAreasJSON, &partnership.Funding,
			&partnership.PhDStudents, &jointProjectsJSON, &publicationsJSON,
			&partnership.StartDate, &endDate, &roiJSON, &partnership.CreatedAt, &partnership.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}

		partnership.EndDate = endDate
		json.Unmarshal(contactsJSON, &partnership.Contacts)
		json.Unmarshal(researchAreasJSON, &partnership.ResearchAreas)
		json.Unmarshal(jointProjectsJSON, &partnership.JointProjects)
		json.Unmarshal(publicationsJSON, &partnership.Publications)
		json.Unmarshal(roiJSON, &partnership.ROI)

		partnerships = append(partnerships, partnership)
	}

	return partnerships, nil
}

// ===========================
// Grant Tracking System
// ===========================

// Grant represents research funding
type Grant struct {
	ID            string      `json:"id"`
	Title         string      `json:"title"`
	Agency        string      `json:"agency"` // NSF, DARPA, DOE, etc.
	Program       string      `json:"program"`
	PI            string      `json:"pi"` // Principal Investigator
	CoIs          []string    `json:"co_investigators"`
	Amount        float64     `json:"amount"` // USD
	Status        GrantStatus `json:"status"`
	SubmissionDate time.Time  `json:"submission_date"`
	AwardDate     *time.Time  `json:"award_date,omitempty"`
	StartDate     *time.Time  `json:"start_date,omitempty"`
	EndDate       *time.Time  `json:"end_date,omitempty"`
	ExperimentIDs []string    `json:"experiment_ids"`
	Publications  []string    `json:"publications"` // Resulting papers
	CreatedAt     time.Time   `json:"created_at"`
	UpdatedAt     time.Time   `json:"updated_at"`
}

// GrantStatus tracks grant lifecycle
type GrantStatus string

const (
	GrantDraft     GrantStatus = "draft"
	GrantSubmitted GrantStatus = "submitted"
	GrantUnderReview GrantStatus = "under_review"
	GrantAwarded   GrantStatus = "awarded"
	GrantActive    GrantStatus = "active"
	GrantCompleted GrantStatus = "completed"
	GrantRejected  GrantStatus = "rejected"
)

// GrantTracker manages research grants
type GrantTracker struct {
	db *sql.DB
}

// NewGrantTracker creates grant tracker
func NewGrantTracker(db *sql.DB) *GrantTracker {
	return &GrantTracker{db: db}
}

// CreateGrant registers new grant application
func (gt *GrantTracker) CreateGrant(ctx context.Context, grant *Grant) error {
	grant.ID = uuid.New().String()
	grant.Status = GrantDraft
	grant.CreatedAt = time.Now()
	grant.UpdatedAt = time.Now()

	query := `
		INSERT INTO grants
		(id, title, agency, program, pi, co_investigators, amount, status,
		 submission_date, experiment_ids, publications, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
	`

	coIsJSON, _ := json.Marshal(grant.CoIs)
	experimentIDsJSON, _ := json.Marshal(grant.ExperimentIDs)
	publicationsJSON, _ := json.Marshal(grant.Publications)

	_, err := gt.db.ExecContext(ctx, query,
		grant.ID, grant.Title, grant.Agency, grant.Program, grant.PI,
		coIsJSON, grant.Amount, grant.Status, grant.SubmissionDate,
		experimentIDsJSON, publicationsJSON, grant.CreatedAt, grant.UpdatedAt,
	)

	return err
}

// UpdateGrantStatus transitions grant lifecycle
func (gt *GrantTracker) UpdateGrantStatus(ctx context.Context, grantID string, status GrantStatus) error {
	query := `UPDATE grants SET status = $1, updated_at = $2 WHERE id = $3`
	_, err := gt.db.ExecContext(ctx, query, status, time.Now(), grantID)
	return err
}

// RecordAward records grant award
func (gt *GrantTracker) RecordAward(ctx context.Context, grantID string, awardDate, startDate, endDate time.Time) error {
	query := `UPDATE grants SET status = $1, award_date = $2, start_date = $3, end_date = $4, updated_at = $5 WHERE id = $6`
	_, err := gt.db.ExecContext(ctx, query, GrantAwarded, awardDate, startDate, endDate, time.Now(), grantID)
	return err
}

// ===========================
// Research Metrics Dashboard
// ===========================

// ResearchMetrics aggregates research output
type ResearchMetrics struct {
	db *sql.DB
}

// NewResearchMetrics creates metrics dashboard
func NewResearchMetrics(db *sql.DB) *ResearchMetrics {
	return &ResearchMetrics{db: db}
}

// AggregateMetrics computes research KPIs
func (rm *ResearchMetrics) AggregateMetrics(ctx context.Context, year int) (*ResearchKPIs, error) {
	kpis := &ResearchKPIs{Year: year}

	// Count experiments
	rm.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM experiments WHERE EXTRACT(YEAR FROM created_at) = $1", year).Scan(&kpis.ExperimentsCount)

	// Count publications
	rm.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM publications WHERE EXTRACT(YEAR FROM publish_date) = $1", year).Scan(&kpis.PublicationsCount)

	// Count patents
	rm.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM patents WHERE EXTRACT(YEAR FROM application_date) = $1", year).Scan(&kpis.PatentsCount)

	// Count prototypes
	rm.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM prototypes WHERE EXTRACT(YEAR FROM created_at) = $1", year).Scan(&kpis.PrototypesCount)

	// Count grants
	rm.db.QueryRowContext(ctx,
		"SELECT COUNT(*), COALESCE(SUM(amount), 0) FROM grants WHERE EXTRACT(YEAR FROM award_date) = $1",
		year).Scan(&kpis.GrantsCount, &kpis.GrantsFunding)

	// Count partnerships
	rm.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM partnerships WHERE status = 'active' AND EXTRACT(YEAR FROM start_date) = $1",
		year).Scan(&kpis.PartnershipsCount)

	// Calculate total citations
	rm.db.QueryRowContext(ctx,
		"SELECT COALESCE(SUM(citations), 0) FROM publications WHERE EXTRACT(YEAR FROM publish_date) = $1",
		year).Scan(&kpis.TotalCitations)

	// Calculate average impact score
	rm.db.QueryRowContext(ctx,
		"SELECT COALESCE(AVG(impact_score), 0) FROM publications WHERE EXTRACT(YEAR FROM publish_date) = $1",
		year).Scan(&kpis.AvgImpactScore)

	return kpis, nil
}

// ResearchKPIs captures annual research metrics
type ResearchKPIs struct {
	Year               int     `json:"year"`
	ExperimentsCount   int     `json:"experiments_count"`
	PublicationsCount  int     `json:"publications_count"`
	PatentsCount       int     `json:"patents_count"`
	PrototypesCount    int     `json:"prototypes_count"`
	GrantsCount        int     `json:"grants_count"`
	GrantsFunding      float64 `json:"grants_funding"`
	PartnershipsCount  int     `json:"partnerships_count"`
	TotalCitations     int     `json:"total_citations"`
	AvgImpactScore     float64 `json:"avg_impact_score"`
}

// GetTopPublications retrieves most cited papers
func (rm *ResearchMetrics) GetTopPublications(ctx context.Context, limit int) ([]*Publication, error) {
	query := `
		SELECT id, title, abstract, authors, venue, type, status, submission_date,
		       publish_date, doi, arxiv_id, pdf, code, data, citations, impact_score,
		       experiment_ids, created_at, updated_at
		FROM publications
		WHERE status = 'published'
		ORDER BY citations DESC
		LIMIT $1
	`

	rows, err := rm.db.QueryContext(ctx, query, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	publications := []*Publication{}
	for rows.Next() {
		pub := &Publication{}
		var authorsJSON, experimentIDsJSON []byte
		var publishDate *time.Time

		err := rows.Scan(
			&pub.ID, &pub.Title, &pub.Abstract, &authorsJSON, &pub.Venue, &pub.Type,
			&pub.Status, &pub.SubmissionDate, &publishDate, &pub.DOI, &pub.ArxivID,
			&pub.PDF, &pub.Code, &pub.Data, &pub.Citations, &pub.ImpactScore,
			&experimentIDsJSON, &pub.CreatedAt, &pub.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}

		pub.PublishDate = publishDate
		json.Unmarshal(authorsJSON, &pub.Authors)
		json.Unmarshal(experimentIDsJSON, &pub.ExperimentIDs)

		publications = append(publications, pub)
	}

	return publications, nil
}

// GetPatentPortfolio retrieves granted patents
func (rm *ResearchMetrics) GetPatentPortfolio(ctx context.Context) ([]*Patent, error) {
	query := `
		SELECT id, title, abstract, inventors, assignee, status, application_date,
		       grant_date, patent_number, country, claims, prior_art, licensing,
		       experiment_ids, prototype_ids, revenue, created_at, updated_at
		FROM patents
		WHERE status = 'granted'
		ORDER BY grant_date DESC
	`

	rows, err := rm.db.QueryContext(ctx, query)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	patents := []*Patent{}
	for rows.Next() {
		patent := &Patent{}
		var inventorsJSON, claimsJSON, priorArtJSON, licensingJSON, experimentIDsJSON, prototypeIDsJSON []byte
		var grantDate *time.Time

		err := rows.Scan(
			&patent.ID, &patent.Title, &patent.Abstract, &inventorsJSON, &patent.Assignee,
			&patent.Status, &patent.ApplicationDate, &grantDate, &patent.PatentNumber,
			&patent.Country, &claimsJSON, &priorArtJSON, &licensingJSON,
			&experimentIDsJSON, &prototypeIDsJSON, &patent.Revenue, &patent.CreatedAt, &patent.UpdatedAt,
		)
		if err != nil {
			return nil, err
		}

		patent.GrantDate = grantDate
		json.Unmarshal(inventorsJSON, &patent.Inventors)
		json.Unmarshal(claimsJSON, &patent.Claims)
		json.Unmarshal(priorArtJSON, &patent.PriorArt)
		json.Unmarshal(licensingJSON, &patent.Licensing)
		json.Unmarshal(experimentIDsJSON, &patent.ExperimentIDs)
		json.Unmarshal(prototypeIDsJSON, &patent.PrototypeIDs)

		patents = append(patents, patent)
	}

	return patents, nil
}

// GenerateAnnualReport creates comprehensive research report
func (rm *ResearchMetrics) GenerateAnnualReport(ctx context.Context, year int) (*AnnualResearchReport, error) {
	report := &AnnualResearchReport{
		Year:      year,
		Generated: time.Now(),
	}

	// Aggregate KPIs
	kpis, err := rm.AggregateMetrics(ctx, year)
	if err != nil {
		return nil, err
	}
	report.KPIs = kpis

	// Top publications
	topPubs, err := rm.GetTopPublications(ctx, 10)
	if err != nil {
		return nil, err
	}
	report.TopPublications = topPubs

	// Patent portfolio
	patents, err := rm.GetPatentPortfolio(ctx)
	if err != nil {
		return nil, err
	}
	report.Patents = patents

	return report, nil
}

// AnnualResearchReport captures yearly research output
type AnnualResearchReport struct {
	Year            int              `json:"year"`
	KPIs            *ResearchKPIs    `json:"kpis"`
	TopPublications []*Publication   `json:"top_publications"`
	Patents         []*Patent        `json:"patents"`
	Generated       time.Time        `json:"generated"`
}
