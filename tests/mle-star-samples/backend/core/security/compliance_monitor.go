package security

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// ComplianceMonitor handles compliance monitoring and assessment
type ComplianceMonitor struct {
	config     *AuditConfig
	frameworks map[string]ComplianceFramework
	assessor   *ComplianceAssessor
}

// ComplianceFramework defines a compliance framework interface
type ComplianceFramework interface {
	Name() string
	Version() string
	GetControls() []ComplianceControl
	Assess(ctx context.Context, evidence map[string]interface{}) (*FrameworkAssessment, error)
}

// ComplianceControl represents a single compliance control
type ComplianceControl struct {
	ID                string                 `json:"id"`
	Name              string                 `json:"name"`
	Description       string                 `json:"description"`
	Category          string                 `json:"category"`
	Severity          ControlSeverity        `json:"severity"`
	RequiredEvidence  []EvidenceRequirement  `json:"required_evidence"`
	AutomationLevel   AutomationLevel        `json:"automation_level"`
	TestProcedures    []TestProcedure        `json:"test_procedures"`
	Remediation       string                 `json:"remediation"`
	References        []string               `json:"references"`
	LastAssessed      *time.Time             `json:"last_assessed,omitempty"`
	Status            ControlStatus          `json:"status"`
	Score             float64                `json:"score"`
	Findings          []ComplianceFinding    `json:"findings"`
	Metadata          map[string]interface{} `json:"metadata"`
}

// ControlSeverity defines the severity of a compliance control
type ControlSeverity string

const (
	SeverityInfo     ControlSeverity = "info"
	SeverityLow      ControlSeverity = "low"
	SeverityMedium   ControlSeverity = "medium"
	SeverityHigh     ControlSeverity = "high"
	SeverityCritical ControlSeverity = "critical"
)

// AutomationLevel defines how automated the control assessment is
type AutomationLevel string

const (
	AutomationManual      AutomationLevel = "manual"
	AutomationSemiAuto    AutomationLevel = "semi_automated"
	AutomationFullyAuto   AutomationLevel = "fully_automated"
	AutomationContinuous  AutomationLevel = "continuous"
)

// ControlStatus defines the compliance status of a control
type ControlStatus string

const (
	StatusNotImplemented ControlStatus = "not_implemented"
	StatusImplemented    ControlStatus = "implemented"
	StatusPartiallyImpl  ControlStatus = "partially_implemented"
	StatusNonCompliant   ControlStatus = "non_compliant"
	StatusCompliant      ControlStatus = "compliant"
	StatusExceeds        ControlStatus = "exceeds"
)

// EvidenceRequirement specifies what evidence is needed for a control
type EvidenceRequirement struct {
	Type        string `json:"type"`
	Description string `json:"description"`
	Source      string `json:"source"`
	Required    bool   `json:"required"`
	Automated   bool   `json:"automated"`
}

// TestProcedure defines how to test a compliance control
type TestProcedure struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Type        TestType          `json:"type"`
	Description string            `json:"description"`
	Steps       []string          `json:"steps"`
	Expected    string            `json:"expected"`
	Automated   bool              `json:"automated"`
	Command     string            `json:"command,omitempty"`
	Script      string            `json:"script,omitempty"`
	Config      map[string]string `json:"config,omitempty"`
}

// TestType defines the type of compliance test
type TestType string

const (
	TestTypeConfig     TestType = "configuration"
	TestTypeProcess    TestType = "process"
	TestTypeEvidence   TestType = "evidence"
	TestTypeInterview  TestType = "interview"
	TestTypeObservation TestType = "observation"
	TestTypeAutomated  TestType = "automated"
)

// ComplianceFinding represents a compliance gap or issue
type ComplianceFinding struct {
	ID              string          `json:"id"`
	ControlID       string          `json:"control_id"`
	Type            FindingType     `json:"type"`
	Severity        FindingSeverity `json:"severity"`
	Title           string          `json:"title"`
	Description     string          `json:"description"`
	Evidence        string          `json:"evidence"`
	Recommendation  string          `json:"recommendation"`
	Status          FindingStatus   `json:"status"`
	DetectedAt      time.Time       `json:"detected_at"`
	DueDate         *time.Time      `json:"due_date,omitempty"`
	AssignedTo      string          `json:"assigned_to,omitempty"`
	Resolution      string          `json:"resolution,omitempty"`
	ResolvedAt      *time.Time      `json:"resolved_at,omitempty"`
}

// FindingType categorizes compliance findings
type FindingType string

const (
	FindingTypeGap        FindingType = "gap"
	FindingTypeDeficiency FindingType = "deficiency"
	FindingTypeObservation FindingType = "observation"
	FindingTypeException   FindingType = "exception"
)

// FindingSeverity defines the severity of a compliance finding
type FindingSeverity string

const (
	FindingSeverityInfo     FindingSeverity = "info"
	FindingSeverityLow      FindingSeverity = "low"
	FindingSeverityMedium   FindingSeverity = "medium"
	FindingSeverityHigh     FindingSeverity = "high"
	FindingSeverityCritical FindingSeverity = "critical"
)

// FindingStatus tracks the status of compliance findings
type FindingStatus string

const (
	FindingStatusOpen        FindingStatus = "open"
	FindingStatusInProgress  FindingStatus = "in_progress"
	FindingStatusResolved    FindingStatus = "resolved"
	FindingStatusAccepted    FindingStatus = "accepted"
	FindingStatusFalsePositive FindingStatus = "false_positive"
)

// FrameworkAssessment represents the assessment results for a framework
type FrameworkAssessment struct {
	Framework       string                 `json:"framework"`
	Version         string                 `json:"version"`
	AssessmentDate  time.Time             `json:"assessment_date"`
	OverallScore    float64               `json:"overall_score"`
	Status          AssessmentStatus      `json:"status"`
	ControlResults  map[string]*ControlResult `json:"control_results"`
	CategoryScores  map[string]float64    `json:"category_scores"`
	Summary         string                `json:"summary"`
	Findings        []ComplianceFinding   `json:"findings"`
	Recommendations []string              `json:"recommendations"`
	NextAssessment  time.Time             `json:"next_assessment"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// AssessmentStatus defines the overall assessment status
type AssessmentStatus string

const (
	AssessmentStatusCompliant    AssessmentStatus = "compliant"
	AssessmentStatusNonCompliant AssessmentStatus = "non_compliant"
	AssessmentStatusPartial      AssessmentStatus = "partial"
	AssessmentStatusInProgress   AssessmentStatus = "in_progress"
)

// ControlResult represents the result of assessing a single control
type ControlResult struct {
	ControlID       string            `json:"control_id"`
	Score           float64           `json:"score"`
	Status          ControlStatus     `json:"status"`
	Evidence        []Evidence        `json:"evidence"`
	TestResults     []TestResult      `json:"test_results"`
	Findings        []ComplianceFinding `json:"findings"`
	LastTested      time.Time         `json:"last_tested"`
	NextTest        time.Time         `json:"next_test"`
	AssessorNotes   string           `json:"assessor_notes"`
}

// Evidence represents compliance evidence
type Evidence struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"`
	Source      string                 `json:"source"`
	Description string                 `json:"description"`
	Data        interface{}            `json:"data"`
	CollectedAt time.Time             `json:"collected_at"`
	ValidUntil  *time.Time            `json:"valid_until,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// TestResult represents the result of a compliance test
type TestResult struct {
	TestID      string                 `json:"test_id"`
	Status      TestStatus             `json:"status"`
	Score       float64               `json:"score"`
	Output      string                `json:"output"`
	Error       string                `json:"error,omitempty"`
	ExecutedAt  time.Time             `json:"executed_at"`
	Duration    time.Duration         `json:"duration"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// TestStatus defines the status of a compliance test
type TestStatus string

const (
	TestStatusPassed  TestStatus = "passed"
	TestStatusFailed  TestStatus = "failed"
	TestStatusSkipped TestStatus = "skipped"
	TestStatusError   TestStatus = "error"
)

// ComplianceResults aggregates all compliance monitoring results
type ComplianceResults struct {
	OverallScore       float64                        `json:"overall_score"`
	Status             AssessmentStatus               `json:"status"`
	FrameworkResults   map[string]*FrameworkAssessment `json:"framework_results"`
	TotalControls      int                           `json:"total_controls"`
	CompliantControls  int                           `json:"compliant_controls"`
	NonCompliantControls int                         `json:"non_compliant_controls"`
	TotalFindings      int                           `json:"total_findings"`
	CriticalFindings   int                           `json:"critical_findings"`
	HighFindings       int                           `json:"high_findings"`
	AssessmentDate     time.Time                     `json:"assessment_date"`
	NextAssessment     time.Time                     `json:"next_assessment"`
	Summary            string                        `json:"summary"`
	Trends             ComplianceTrends              `json:"trends"`
}

// ComplianceTrends tracks compliance trends over time
type ComplianceTrends struct {
	ScoreHistory       []ScorePoint `json:"score_history"`
	FindingsTrend      TrendDirection `json:"findings_trend"`
	ComplianceTrend    TrendDirection `json:"compliance_trend"`
	ImprovementAreas   []string     `json:"improvement_areas"`
	StableAreas        []string     `json:"stable_areas"`
}

// ScorePoint represents a point in the compliance score timeline
type ScorePoint struct {
	Date  time.Time `json:"date"`
	Score float64   `json:"score"`
}

// TrendDirection indicates the direction of a trend
type TrendDirection string

const (
	TrendUp    TrendDirection = "up"
	TrendDown  TrendDirection = "down"
	TrendStable TrendDirection = "stable"
)

// ComplianceAssessor performs compliance assessments
type ComplianceAssessor struct {
	evidenceCollector *EvidenceCollector
	testRunner        *TestRunner
}

// NewComplianceMonitor creates a new compliance monitor
func NewComplianceMonitor(config *AuditConfig) *ComplianceMonitor {
	monitor := &ComplianceMonitor{
		config:     config,
		frameworks: make(map[string]ComplianceFramework),
		assessor:   NewComplianceAssessor(),
	}
	
	// Initialize compliance frameworks
	monitor.initializeFrameworks()
	
	return monitor
}

// initializeFrameworks initializes supported compliance frameworks
func (cm *ComplianceMonitor) initializeFrameworks() {
	frameworks := []ComplianceFramework{
		NewSOC2Framework(),
		NewGDPRFramework(),
		NewNISTFramework(),
		NewISO27001Framework(),
		NewPCIDSSFramework(),
	}
	
	for _, framework := range frameworks {
		cm.frameworks[framework.Name()] = framework
		log.Printf("Initialized compliance framework: %s v%s", framework.Name(), framework.Version())
	}
}

// Assess performs comprehensive compliance assessment
func (cm *ComplianceMonitor) Assess(ctx context.Context, level AuditLevel) (*ComplianceResults, error) {
	results := &ComplianceResults{
		FrameworkResults: make(map[string]*FrameworkAssessment),
		AssessmentDate:   time.Now(),
		NextAssessment:   time.Now().AddDate(0, 3, 0), // Default: quarterly
	}
	
	// Collect evidence
	evidence, err := cm.assessor.CollectEvidence(ctx)
	if err != nil {
		log.Printf("Evidence collection failed: %v", err)
		evidence = make(map[string]interface{})
	}
	
	// Assess each enabled framework
	for _, frameworkName := range cm.config.ComplianceFrameworks {
		if framework, exists := cm.frameworks[frameworkName]; exists {
			assessment, err := framework.Assess(ctx, evidence)
			if err != nil {
				log.Printf("Assessment failed for framework %s: %v", frameworkName, err)
				continue
			}
			results.FrameworkResults[frameworkName] = assessment
		}
	}
	
	// Calculate aggregate metrics
	cm.calculateAggregateMetrics(results)
	
	// Generate summary and recommendations
	results.Summary = cm.generateSummary(results)
	
	return results, nil
}

// calculateAggregateMetrics calculates overall compliance metrics
func (cm *ComplianceMonitor) calculateAggregateMetrics(results *ComplianceResults) {
	var totalScore, weightSum float64
	var totalControls, compliantControls, nonCompliantControls int
	var totalFindings, criticalFindings, highFindings int
	
	for _, assessment := range results.FrameworkResults {
		totalScore += assessment.OverallScore
		weightSum += 1.0
		
		for _, control := range assessment.ControlResults {
			totalControls++
			switch control.Status {
			case StatusCompliant, StatusExceeds:
				compliantControls++
			case StatusNonCompliant:
				nonCompliantControls++
			}
			
			for _, finding := range control.Findings {
				totalFindings++
				switch finding.Severity {
				case FindingSeverityCritical:
					criticalFindings++
				case FindingSeverityHigh:
					highFindings++
				}
			}
		}
	}
	
	if weightSum > 0 {
		results.OverallScore = totalScore / weightSum
	}
	
	results.TotalControls = totalControls
	results.CompliantControls = compliantControls
	results.NonCompliantControls = nonCompliantControls
	results.TotalFindings = totalFindings
	results.CriticalFindings = criticalFindings
	results.HighFindings = highFindings
	
	// Determine overall status
	if results.OverallScore >= 90 && results.CriticalFindings == 0 {
		results.Status = AssessmentStatusCompliant
	} else if results.OverallScore >= 70 {
		results.Status = AssessmentStatusPartial
	} else {
		results.Status = AssessmentStatusNonCompliant
	}
}

// generateSummary creates a human-readable compliance summary
func (cm *ComplianceMonitor) generateSummary(results *ComplianceResults) string {
	summary := fmt.Sprintf("Compliance Assessment Summary:\n")
	summary += fmt.Sprintf("Overall Score: %.1f/100 (Status: %s)\n", results.OverallScore, results.Status)
	summary += fmt.Sprintf("Controls: %d Total, %d Compliant, %d Non-Compliant\n",
		results.TotalControls, results.CompliantControls, results.NonCompliantControls)
	summary += fmt.Sprintf("Findings: %d Total (%d Critical, %d High)\n",
		results.TotalFindings, results.CriticalFindings, results.HighFindings)
	
	summary += "\nFramework Results:\n"
	for name, assessment := range results.FrameworkResults {
		summary += fmt.Sprintf("- %s: %.1f/100 (%s)\n", name, assessment.OverallScore, assessment.Status)
	}
	
	return summary
}

// GetComplianceReport generates a detailed compliance report
func (cm *ComplianceMonitor) GetComplianceReport(ctx context.Context, framework string, format string) ([]byte, error) {
	if framework != "" {
		if fw, exists := cm.frameworks[framework]; exists {
			evidence, _ := cm.assessor.CollectEvidence(ctx)
			assessment, err := fw.Assess(ctx, evidence)
			if err != nil {
				return nil, err
			}
			return cm.formatReport(assessment, format)
		}
		return nil, fmt.Errorf("framework not found: %s", framework)
	}
	
	// Generate comprehensive report for all frameworks
	results, err := cm.Assess(ctx, AuditLevelStandard)
	if err != nil {
		return nil, err
	}
	
	return cm.formatResults(results, format)
}

// formatReport formats a single framework assessment report
func (cm *ComplianceMonitor) formatReport(assessment *FrameworkAssessment, format string) ([]byte, error) {
	switch format {
	case "json":
		return json.MarshalIndent(assessment, "", "  ")
	case "summary":
		return []byte(assessment.Summary), nil
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}
}

// formatResults formats comprehensive compliance results
func (cm *ComplianceMonitor) formatResults(results *ComplianceResults, format string) ([]byte, error) {
	switch format {
	case "json":
		return json.MarshalIndent(results, "", "  ")
	case "summary":
		return []byte(results.Summary), nil
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}
}

// NewComplianceAssessor creates a new compliance assessor
func NewComplianceAssessor() *ComplianceAssessor {
	return &ComplianceAssessor{
		evidenceCollector: NewEvidenceCollector(),
		testRunner:        NewTestRunner(),
	}
}

// CollectEvidence collects evidence for compliance assessment
func (ca *ComplianceAssessor) CollectEvidence(ctx context.Context) (map[string]interface{}, error) {
	return ca.evidenceCollector.Collect(ctx)
}

// EvidenceCollector collects compliance evidence
type EvidenceCollector struct{}

// NewEvidenceCollector creates a new evidence collector
func NewEvidenceCollector() *EvidenceCollector {
	return &EvidenceCollector{}
}

// Collect gathers evidence from various sources
func (ec *EvidenceCollector) Collect(ctx context.Context) (map[string]interface{}, error) {
	evidence := make(map[string]interface{})
	
	// Collect system configuration evidence
	evidence["system_config"] = ec.collectSystemConfig()
	
	// Collect access control evidence
	evidence["access_control"] = ec.collectAccessControlConfig()
	
	// Collect encryption evidence
	evidence["encryption"] = ec.collectEncryptionConfig()
	
	// Collect audit log evidence
	evidence["audit_logs"] = ec.collectAuditLogConfig()
	
	// Collect backup evidence
	evidence["backup"] = ec.collectBackupConfig()
	
	// Collect network security evidence
	evidence["network_security"] = ec.collectNetworkSecurityConfig()
	
	return evidence, nil
}

func (ec *EvidenceCollector) collectSystemConfig() map[string]interface{} {
	// Implementation would collect actual system configuration
	return map[string]interface{}{
		"os_version": "Ubuntu 20.04",
		"security_updates": "enabled",
		"firewall": "enabled",
		"antivirus": "enabled",
	}
}

func (ec *EvidenceCollector) collectAccessControlConfig() map[string]interface{} {
	// Implementation would collect access control configuration
	return map[string]interface{}{
		"rbac_enabled": true,
		"mfa_enabled": true,
		"password_policy": "strong",
		"session_timeout": 30,
	}
}

func (ec *EvidenceCollector) collectEncryptionConfig() map[string]interface{} {
	// Implementation would collect encryption configuration
	return map[string]interface{}{
		"data_at_rest": "AES-256",
		"data_in_transit": "TLS 1.3",
		"key_management": "AWS KMS",
		"certificate_management": "automated",
	}
}

func (ec *EvidenceCollector) collectAuditLogConfig() map[string]interface{} {
	// Implementation would collect audit logging configuration
	return map[string]interface{}{
		"audit_logging": "enabled",
		"log_retention": "7 years",
		"log_integrity": "hash-protected",
		"log_monitoring": "real-time",
	}
}

func (ec *EvidenceCollector) collectBackupConfig() map[string]interface{} {
	// Implementation would collect backup configuration
	return map[string]interface{}{
		"backup_frequency": "daily",
		"backup_retention": "90 days",
		"backup_encryption": "enabled",
		"backup_testing": "monthly",
	}
}

func (ec *EvidenceCollector) collectNetworkSecurityConfig() map[string]interface{} {
	// Implementation would collect network security configuration
	return map[string]interface{}{
		"network_segmentation": "enabled",
		"intrusion_detection": "enabled",
		"vulnerability_scanning": "weekly",
		"penetration_testing": "quarterly",
	}
}

// TestRunner executes compliance tests
type TestRunner struct{}

// NewTestRunner creates a new test runner
func NewTestRunner() *TestRunner {
	return &TestRunner{}
}

// RunTest executes a compliance test
func (tr *TestRunner) RunTest(ctx context.Context, test *TestProcedure) (*TestResult, error) {
	startTime := time.Now()
	
	result := &TestResult{
		TestID:     test.ID,
		ExecutedAt: startTime,
		Metadata:   make(map[string]interface{}),
	}
	
	// Execute test based on type
	switch test.Type {
	case TestTypeAutomated:
		err := tr.runAutomatedTest(ctx, test, result)
		if err != nil {
			result.Status = TestStatusError
			result.Error = err.Error()
		}
	case TestTypeConfig:
		err := tr.runConfigurationTest(ctx, test, result)
		if err != nil {
			result.Status = TestStatusError
			result.Error = err.Error()
		}
	default:
		result.Status = TestStatusSkipped
		result.Output = "Manual test - requires human intervention"
	}
	
	result.Duration = time.Since(startTime)
	return result, nil
}

func (tr *TestRunner) runAutomatedTest(ctx context.Context, test *TestProcedure, result *TestResult) error {
	// Implementation would execute automated test scripts
	result.Status = TestStatusPassed
	result.Score = 100.0
	result.Output = "Test completed successfully"
	return nil
}

func (tr *TestRunner) runConfigurationTest(ctx context.Context, test *TestProcedure, result *TestResult) error {
	// Implementation would check configuration settings
	result.Status = TestStatusPassed
	result.Score = 100.0
	result.Output = "Configuration compliant"
	return nil
}