// Package validation provides comprehensive production readiness validation
// including pre-deployment checks, smoke tests, performance regression detection,
// security scanning, and compliance verification.
package validation

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// ReadinessValidator validates production readiness across multiple dimensions
type ReadinessValidator struct {
	mu                    sync.RWMutex
	validations           map[string]*ValidationResult
	smokeTestRunner       SmokeTestRunner
	perfRegressionChecker PerformanceRegressionChecker
	securityScanner       SecurityScanner
	complianceChecker     ComplianceChecker
	capacityValidator     CapacityValidator
	dependencyChecker     DependencyChecker
	ctx                   context.Context
	cancel                context.CancelFunc
}

// ValidationResult represents the result of a readiness validation
type ValidationResult struct {
	ID                    string                     `json:"id"`
	DeploymentID          string                     `json:"deployment_id"`
	Version               string                     `json:"version"`
	Status                ValidationStatus           `json:"status"`
	OverallScore          float64                    `json:"overall_score"`
	Passed                bool                       `json:"passed"`
	StartTime             time.Time                  `json:"start_time"`
	EndTime               *time.Time                 `json:"end_time,omitempty"`
	Duration              time.Duration              `json:"duration"`
	SmokeTests            *SmokeTestResults          `json:"smoke_tests"`
	Performance           *PerformanceResults        `json:"performance"`
	Security              *SecurityResults           `json:"security"`
	Compliance            *ComplianceResults         `json:"compliance"`
	Capacity              *CapacityResults           `json:"capacity"`
	Dependencies          *DependencyResults         `json:"dependencies"`
	FailureReasons        []string                   `json:"failure_reasons,omitempty"`
	Recommendations       []string                   `json:"recommendations,omitempty"`
	BlockingIssues        []BlockingIssue            `json:"blocking_issues,omitempty"`
}

// ValidationStatus represents validation status
type ValidationStatus string

const (
	ValidationStatusPending    ValidationStatus = "pending"
	ValidationStatusRunning    ValidationStatus = "running"
	ValidationStatusPassed     ValidationStatus = "passed"
	ValidationStatusFailed     ValidationStatus = "failed"
	ValidationStatusWarning    ValidationStatus = "warning"
)

// SmokeTestResults contains smoke test results
type SmokeTestResults struct {
	TotalTests     int               `json:"total_tests"`
	PassedTests    int               `json:"passed_tests"`
	FailedTests    int               `json:"failed_tests"`
	SkippedTests   int               `json:"skipped_tests"`
	Duration       time.Duration     `json:"duration"`
	Coverage       float64           `json:"coverage"`
	Tests          []SmokeTest       `json:"tests"`
	CriticalPassed bool              `json:"critical_passed"`
}

// SmokeTest represents a single smoke test
type SmokeTest struct {
	Name        string        `json:"name"`
	Category    string        `json:"category"`
	Critical    bool          `json:"critical"`
	Passed      bool          `json:"passed"`
	Duration    time.Duration `json:"duration"`
	Error       string        `json:"error,omitempty"`
	Details     interface{}   `json:"details,omitempty"`
	ExecutedAt  time.Time     `json:"executed_at"`
}

// PerformanceResults contains performance regression results
type PerformanceResults struct {
	BaselineVersion    string                `json:"baseline_version"`
	CurrentVersion     string                `json:"current_version"`
	RegressionDetected bool                  `json:"regression_detected"`
	Metrics            *PerformanceMetrics   `json:"metrics"`
	Comparison         *PerformanceComparison `json:"comparison"`
	Benchmarks         []BenchmarkResult     `json:"benchmarks"`
	Passed             bool                  `json:"passed"`
}

// PerformanceMetrics contains performance metrics
type PerformanceMetrics struct {
	Latency           *LatencyMetrics    `json:"latency"`
	Throughput        *ThroughputMetrics `json:"throughput"`
	ResourceUsage     *ResourceMetrics   `json:"resource_usage"`
	ErrorRate         float64            `json:"error_rate"`
	Timestamp         time.Time          `json:"timestamp"`
}

// LatencyMetrics contains latency measurements
type LatencyMetrics struct {
	P50    time.Duration `json:"p50"`
	P95    time.Duration `json:"p95"`
	P99    time.Duration `json:"p99"`
	Mean   time.Duration `json:"mean"`
	Max    time.Duration `json:"max"`
	StdDev time.Duration `json:"std_dev"`
}

// ThroughputMetrics contains throughput measurements
type ThroughputMetrics struct {
	RequestsPerSecond float64 `json:"requests_per_second"`
	BytesPerSecond    float64 `json:"bytes_per_second"`
	TransactionsPerSec float64 `json:"transactions_per_second"`
}

// ResourceMetrics contains resource usage metrics
type ResourceMetrics struct {
	CPUPercent    float64 `json:"cpu_percent"`
	MemoryMB      float64 `json:"memory_mb"`
	DiskIOPS      int64   `json:"disk_iops"`
	NetworkMbps   float64 `json:"network_mbps"`
}

// PerformanceComparison compares current vs baseline performance
type PerformanceComparison struct {
	LatencyChange       float64 `json:"latency_change_percent"`
	ThroughputChange    float64 `json:"throughput_change_percent"`
	ErrorRateChange     float64 `json:"error_rate_change_percent"`
	ResourceUsageChange float64 `json:"resource_usage_change_percent"`
	AcceptableThreshold float64 `json:"acceptable_threshold"`
	RegressionDetected  bool    `json:"regression_detected"`
}

// BenchmarkResult represents a benchmark test result
type BenchmarkResult struct {
	Name           string        `json:"name"`
	OpsPerSecond   float64       `json:"ops_per_second"`
	AllocsPerOp    int64         `json:"allocs_per_op"`
	BytesPerOp     int64         `json:"bytes_per_op"`
	Duration       time.Duration `json:"duration"`
	Iterations     int           `json:"iterations"`
}

// SecurityResults contains security scan results
type SecurityResults struct {
	ScanType            string              `json:"scan_type"`
	VulnerabilitiesFound int                `json:"vulnerabilities_found"`
	CriticalIssues      int                 `json:"critical_issues"`
	HighIssues          int                 `json:"high_issues"`
	MediumIssues        int                 `json:"medium_issues"`
	LowIssues           int                 `json:"low_issues"`
	Vulnerabilities     []Vulnerability     `json:"vulnerabilities"`
	SecurityScore       float64             `json:"security_score"`
	Passed              bool                `json:"passed"`
	ScanDuration        time.Duration       `json:"scan_duration"`
}

// Vulnerability represents a security vulnerability
type Vulnerability struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	Severity    string    `json:"severity"`
	CVSS        float64   `json:"cvss"`
	Description string    `json:"description"`
	Component   string    `json:"component"`
	Version     string    `json:"version"`
	FixVersion  string    `json:"fix_version,omitempty"`
	CVE         string    `json:"cve,omitempty"`
	CWE         string    `json:"cwe,omitempty"`
	DetectedAt  time.Time `json:"detected_at"`
}

// ComplianceResults contains compliance check results
type ComplianceResults struct {
	Frameworks      []ComplianceFramework `json:"frameworks"`
	TotalControls   int                   `json:"total_controls"`
	PassedControls  int                   `json:"passed_controls"`
	FailedControls  int                   `json:"failed_controls"`
	ComplianceScore float64               `json:"compliance_score"`
	Passed          bool                  `json:"passed"`
	Issues          []ComplianceIssue     `json:"issues"`
}

// ComplianceFramework represents a compliance framework
type ComplianceFramework struct {
	Name           string             `json:"name"`
	Version        string             `json:"version"`
	Compliant      bool               `json:"compliant"`
	Controls       []ComplianceControl `json:"controls"`
	Score          float64            `json:"score"`
}

// ComplianceControl represents a compliance control
type ComplianceControl struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Category    string    `json:"category"`
	Passed      bool      `json:"passed"`
	Required    bool      `json:"required"`
	Evidence    string    `json:"evidence,omitempty"`
	CheckedAt   time.Time `json:"checked_at"`
}

// ComplianceIssue represents a compliance issue
type ComplianceIssue struct {
	Framework   string    `json:"framework"`
	Control     string    `json:"control"`
	Severity    string    `json:"severity"`
	Description string    `json:"description"`
	Remediation string    `json:"remediation"`
	DetectedAt  time.Time `json:"detected_at"`
}

// CapacityResults contains capacity validation results
type CapacityResults struct {
	Resources         []ResourceCapacity `json:"resources"`
	SufficientCapacity bool              `json:"sufficient_capacity"`
	UtilizationScore  float64           `json:"utilization_score"`
	Warnings          []string          `json:"warnings,omitempty"`
	Passed            bool              `json:"passed"`
}

// ResourceCapacity represents capacity for a resource
type ResourceCapacity struct {
	ResourceType string  `json:"resource_type"`
	Required     float64 `json:"required"`
	Available    float64 `json:"available"`
	Used         float64 `json:"used"`
	Utilization  float64 `json:"utilization"`
	Sufficient   bool    `json:"sufficient"`
}

// DependencyResults contains dependency check results
type DependencyResults struct {
	TotalDependencies   int                  `json:"total_dependencies"`
	HealthyDependencies int                  `json:"healthy_dependencies"`
	UnhealthyDependencies int                `json:"unhealthy_dependencies"`
	Dependencies        []DependencyStatus   `json:"dependencies"`
	AllHealthy          bool                 `json:"all_healthy"`
	Passed              bool                 `json:"passed"`
}

// DependencyStatus represents status of a dependency
type DependencyStatus struct {
	Name        string        `json:"name"`
	Type        string        `json:"type"`
	Version     string        `json:"version"`
	Healthy     bool          `json:"healthy"`
	Latency     time.Duration `json:"latency"`
	Error       string        `json:"error,omitempty"`
	CheckedAt   time.Time     `json:"checked_at"`
}

// BlockingIssue represents an issue that blocks deployment
type BlockingIssue struct {
	Category    string    `json:"category"`
	Severity    string    `json:"severity"`
	Title       string    `json:"title"`
	Description string    `json:"description"`
	Impact      string    `json:"impact"`
	Remediation string    `json:"remediation"`
	DetectedAt  time.Time `json:"detected_at"`
}

// SmokeTestRunner executes smoke tests
type SmokeTestRunner interface {
	RunSmokeTests(ctx context.Context, deploymentID string, version string) (*SmokeTestResults, error)
	GetTestSuite(category string) []SmokeTest
}

// PerformanceRegressionChecker checks for performance regressions
type PerformanceRegressionChecker interface {
	CheckRegression(ctx context.Context, baselineVersion, currentVersion string) (*PerformanceResults, error)
	CollectMetrics(ctx context.Context, version string) (*PerformanceMetrics, error)
	RunBenchmarks(ctx context.Context, version string) ([]BenchmarkResult, error)
}

// SecurityScanner scans for security vulnerabilities
type SecurityScanner interface {
	ScanVulnerabilities(ctx context.Context, deploymentID string, version string) (*SecurityResults, error)
	ScanDependencies(ctx context.Context, manifestPath string) ([]Vulnerability, error)
	ScanContainer(ctx context.Context, imageName string) (*SecurityResults, error)
}

// ComplianceChecker checks compliance requirements
type ComplianceChecker interface {
	CheckCompliance(ctx context.Context, deploymentID string, frameworks []string) (*ComplianceResults, error)
	ValidateFramework(ctx context.Context, framework string) (*ComplianceFramework, error)
	GetRequiredControls(framework string) []ComplianceControl
}

// CapacityValidator validates infrastructure capacity
type CapacityValidator interface {
	ValidateCapacity(ctx context.Context, requirements *CapacityRequirements) (*CapacityResults, error)
	GetCurrentCapacity(ctx context.Context) ([]ResourceCapacity, error)
	ProjectCapacity(ctx context.Context, requirements *CapacityRequirements) (*CapacityProjection, error)
}

// CapacityRequirements specifies capacity requirements
type CapacityRequirements struct {
	CPUCores    float64 `json:"cpu_cores"`
	MemoryGB    float64 `json:"memory_gb"`
	DiskGB      float64 `json:"disk_gb"`
	NetworkMbps float64 `json:"network_mbps"`
	Instances   int     `json:"instances"`
}

// CapacityProjection projects future capacity needs
type CapacityProjection struct {
	Current    []ResourceCapacity `json:"current"`
	Projected  []ResourceCapacity `json:"projected"`
	Sufficient bool               `json:"sufficient"`
	Timeline   time.Duration      `json:"timeline"`
}

// DependencyChecker checks external dependencies
type DependencyChecker interface {
	CheckDependencies(ctx context.Context, deploymentID string) (*DependencyResults, error)
	CheckDependency(ctx context.Context, dep *DependencyConfig) (*DependencyStatus, error)
	GetDependencies(deploymentID string) ([]DependencyConfig, error)
}

// DependencyConfig configures a dependency
type DependencyConfig struct {
	Name        string        `json:"name"`
	Type        string        `json:"type"`
	Endpoint    string        `json:"endpoint"`
	Version     string        `json:"version"`
	Timeout     time.Duration `json:"timeout"`
	Required    bool          `json:"required"`
}

// NewReadinessValidator creates a new readiness validator
func NewReadinessValidator(
	smokeTestRunner SmokeTestRunner,
	perfRegressionChecker PerformanceRegressionChecker,
	securityScanner SecurityScanner,
	complianceChecker ComplianceChecker,
	capacityValidator CapacityValidator,
	dependencyChecker DependencyChecker,
) *ReadinessValidator {
	ctx, cancel := context.WithCancel(context.Background())

	return &ReadinessValidator{
		validations:           make(map[string]*ValidationResult),
		smokeTestRunner:       smokeTestRunner,
		perfRegressionChecker: perfRegressionChecker,
		securityScanner:       securityScanner,
		complianceChecker:     complianceChecker,
		capacityValidator:     capacityValidator,
		dependencyChecker:     dependencyChecker,
		ctx:                   ctx,
		cancel:                cancel,
	}
}

// ValidateReadiness performs comprehensive production readiness validation
func (rv *ReadinessValidator) ValidateReadiness(
	ctx context.Context,
	deploymentID string,
	version string,
	config *ValidationConfig,
) (*ValidationResult, error) {
	validationID := generateValidationID(deploymentID, version)

	result := &ValidationResult{
		ID:           validationID,
		DeploymentID: deploymentID,
		Version:      version,
		Status:       ValidationStatusRunning,
		StartTime:    time.Now(),
	}

	rv.mu.Lock()
	rv.validations[validationID] = result
	rv.mu.Unlock()

	var wg sync.WaitGroup
	var mu sync.Mutex
	failures := []string{}
	blockingIssues := []BlockingIssue{}

	// Run smoke tests
	wg.Add(1)
	go func() {
		defer wg.Done()
		smokeTests, err := rv.smokeTestRunner.RunSmokeTests(ctx, deploymentID, version)
		if err != nil {
			mu.Lock()
			failures = append(failures, fmt.Sprintf("Smoke tests failed: %v", err))
			mu.Unlock()
			return
		}

		mu.Lock()
		result.SmokeTests = smokeTests
		if !smokeTests.CriticalPassed {
			failures = append(failures, "Critical smoke tests failed")
			blockingIssues = append(blockingIssues, BlockingIssue{
				Category:    "smoke_tests",
				Severity:    "critical",
				Title:       "Critical smoke tests failed",
				Description: fmt.Sprintf("%d critical tests failed", smokeTests.FailedTests),
				Impact:      "Deployment cannot proceed",
				Remediation: "Fix failing smoke tests before deployment",
				DetectedAt:  time.Now(),
			})
		}
		mu.Unlock()
	}()

	// Check performance regression
	if config.CheckPerformance {
		wg.Add(1)
		go func() {
			defer wg.Done()
			perfResults, err := rv.perfRegressionChecker.CheckRegression(ctx, config.BaselineVersion, version)
			if err != nil {
				mu.Lock()
				failures = append(failures, fmt.Sprintf("Performance check failed: %v", err))
				mu.Unlock()
				return
			}

			mu.Lock()
			result.Performance = perfResults
			if perfResults.RegressionDetected {
				failures = append(failures, "Performance regression detected")
				blockingIssues = append(blockingIssues, BlockingIssue{
					Category:    "performance",
					Severity:    "high",
					Title:       "Performance regression detected",
					Description: fmt.Sprintf("Latency increased by %.2f%%", perfResults.Comparison.LatencyChange),
					Impact:      "User experience may degrade",
					Remediation: "Optimize performance before deployment",
					DetectedAt:  time.Now(),
				})
			}
			mu.Unlock()
		}()
	}

	// Scan for security vulnerabilities
	if config.CheckSecurity {
		wg.Add(1)
		go func() {
			defer wg.Done()
			securityResults, err := rv.securityScanner.ScanVulnerabilities(ctx, deploymentID, version)
			if err != nil {
				mu.Lock()
				failures = append(failures, fmt.Sprintf("Security scan failed: %v", err))
				mu.Unlock()
				return
			}

			mu.Lock()
			result.Security = securityResults
			if securityResults.CriticalIssues > 0 {
				failures = append(failures, fmt.Sprintf("Found %d critical security issues", securityResults.CriticalIssues))
				blockingIssues = append(blockingIssues, BlockingIssue{
					Category:    "security",
					Severity:    "critical",
					Title:       "Critical security vulnerabilities found",
					Description: fmt.Sprintf("%d critical vulnerabilities detected", securityResults.CriticalIssues),
					Impact:      "System security compromised",
					Remediation: "Patch all critical vulnerabilities before deployment",
					DetectedAt:  time.Now(),
				})
			}
			mu.Unlock()
		}()
	}

	// Check compliance
	if config.CheckCompliance {
		wg.Add(1)
		go func() {
			defer wg.Done()
			complianceResults, err := rv.complianceChecker.CheckCompliance(ctx, deploymentID, config.ComplianceFrameworks)
			if err != nil {
				mu.Lock()
				failures = append(failures, fmt.Sprintf("Compliance check failed: %v", err))
				mu.Unlock()
				return
			}

			mu.Lock()
			result.Compliance = complianceResults
			if !complianceResults.Passed {
				failures = append(failures, "Compliance checks failed")
				for _, issue := range complianceResults.Issues {
					if issue.Severity == "critical" {
						blockingIssues = append(blockingIssues, BlockingIssue{
							Category:    "compliance",
							Severity:    "critical",
							Title:       fmt.Sprintf("%s compliance violation", issue.Framework),
							Description: issue.Description,
							Impact:      "Legal/regulatory requirements not met",
							Remediation: issue.Remediation,
							DetectedAt:  time.Now(),
						})
					}
				}
			}
			mu.Unlock()
		}()
	}

	// Validate capacity
	if config.CheckCapacity {
		wg.Add(1)
		go func() {
			defer wg.Done()
			capacityResults, err := rv.capacityValidator.ValidateCapacity(ctx, config.CapacityRequirements)
			if err != nil {
				mu.Lock()
				failures = append(failures, fmt.Sprintf("Capacity validation failed: %v", err))
				mu.Unlock()
				return
			}

			mu.Lock()
			result.Capacity = capacityResults
			if !capacityResults.SufficientCapacity {
				failures = append(failures, "Insufficient infrastructure capacity")
				blockingIssues = append(blockingIssues, BlockingIssue{
					Category:    "capacity",
					Severity:    "critical",
					Title:       "Insufficient infrastructure capacity",
					Description: "Required resources exceed available capacity",
					Impact:      "Deployment may fail or cause service degradation",
					Remediation: "Scale infrastructure before deployment",
					DetectedAt:  time.Now(),
				})
			}
			mu.Unlock()
		}()
	}

	// Check dependencies
	if config.CheckDependencies {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dependencyResults, err := rv.dependencyChecker.CheckDependencies(ctx, deploymentID)
			if err != nil {
				mu.Lock()
				failures = append(failures, fmt.Sprintf("Dependency check failed: %v", err))
				mu.Unlock()
				return
			}

			mu.Lock()
			result.Dependencies = dependencyResults
			if !dependencyResults.AllHealthy {
				failures = append(failures, "Unhealthy dependencies detected")
				blockingIssues = append(blockingIssues, BlockingIssue{
					Category:    "dependencies",
					Severity:    "high",
					Title:       "Unhealthy dependencies",
					Description: fmt.Sprintf("%d dependencies are unhealthy", dependencyResults.UnhealthyDependencies),
					Impact:      "Service may fail to operate correctly",
					Remediation: "Ensure all dependencies are healthy",
					DetectedAt:  time.Now(),
				})
			}
			mu.Unlock()
		}()
	}

	// Wait for all validations to complete
	wg.Wait()

	// Calculate overall score
	score := rv.calculateOverallScore(result)

	// Determine pass/fail
	passed := len(failures) == 0 && score >= config.MinimumScore

	// Update result
	now := time.Now()
	result.EndTime = &now
	result.Duration = now.Sub(result.StartTime)
	result.OverallScore = score
	result.Passed = passed
	result.FailureReasons = failures
	result.BlockingIssues = blockingIssues
	result.Recommendations = rv.generateRecommendations(result)

	if passed {
		result.Status = ValidationStatusPassed
	} else {
		result.Status = ValidationStatusFailed
	}

	rv.mu.Lock()
	rv.validations[validationID] = result
	rv.mu.Unlock()

	return result, nil
}

// ValidationConfig configures validation behavior
type ValidationConfig struct {
	BaselineVersion       string                  `json:"baseline_version"`
	CheckPerformance      bool                    `json:"check_performance"`
	CheckSecurity         bool                    `json:"check_security"`
	CheckCompliance       bool                    `json:"check_compliance"`
	CheckCapacity         bool                    `json:"check_capacity"`
	CheckDependencies     bool                    `json:"check_dependencies"`
	ComplianceFrameworks  []string                `json:"compliance_frameworks"`
	CapacityRequirements  *CapacityRequirements   `json:"capacity_requirements"`
	MinimumScore          float64                 `json:"minimum_score"`
}

// calculateOverallScore calculates the overall validation score
func (rv *ReadinessValidator) calculateOverallScore(result *ValidationResult) float64 {
	var totalScore float64
	var components int

	if result.SmokeTests != nil {
		smokeScore := float64(result.SmokeTests.PassedTests) / float64(result.SmokeTests.TotalTests)
		totalScore += smokeScore
		components++
	}

	if result.Performance != nil && result.Performance.Passed {
		totalScore += 1.0
		components++
	}

	if result.Security != nil {
		securityScore := result.Security.SecurityScore / 100.0
		totalScore += securityScore
		components++
	}

	if result.Compliance != nil {
		complianceScore := result.Compliance.ComplianceScore / 100.0
		totalScore += complianceScore
		components++
	}

	if result.Capacity != nil {
		capacityScore := result.Capacity.UtilizationScore
		totalScore += capacityScore
		components++
	}

	if result.Dependencies != nil && result.Dependencies.AllHealthy {
		totalScore += 1.0
		components++
	}

	if components == 0 {
		return 0.0
	}

	return totalScore / float64(components)
}

// generateRecommendations generates recommendations based on validation results
func (rv *ReadinessValidator) generateRecommendations(result *ValidationResult) []string {
	recommendations := []string{}

	if result.SmokeTests != nil && result.SmokeTests.FailedTests > 0 {
		recommendations = append(recommendations,
			fmt.Sprintf("Fix %d failed smoke tests before deployment", result.SmokeTests.FailedTests))
	}

	if result.Performance != nil && result.Performance.RegressionDetected {
		recommendations = append(recommendations,
			"Investigate and resolve performance regression")
	}

	if result.Security != nil && result.Security.VulnerabilitiesFound > 0 {
		recommendations = append(recommendations,
			fmt.Sprintf("Address %d security vulnerabilities (Critical: %d, High: %d)",
				result.Security.VulnerabilitiesFound,
				result.Security.CriticalIssues,
				result.Security.HighIssues))
	}

	if result.Compliance != nil && !result.Compliance.Passed {
		recommendations = append(recommendations,
			fmt.Sprintf("Resolve %d compliance issues", result.Compliance.FailedControls))
	}

	if result.Capacity != nil && !result.Capacity.SufficientCapacity {
		recommendations = append(recommendations,
			"Scale infrastructure to meet capacity requirements")
	}

	if result.Dependencies != nil && !result.Dependencies.AllHealthy {
		recommendations = append(recommendations,
			fmt.Sprintf("Fix %d unhealthy dependencies", result.Dependencies.UnhealthyDependencies))
	}

	if result.OverallScore < 0.9 {
		recommendations = append(recommendations,
			"Overall readiness score is below optimal threshold (90%)")
	}

	return recommendations
}

// GetValidation retrieves a validation result
func (rv *ReadinessValidator) GetValidation(validationID string) (*ValidationResult, error) {
	rv.mu.RLock()
	defer rv.mu.RUnlock()

	result, exists := rv.validations[validationID]
	if !exists {
		return nil, fmt.Errorf("validation %s not found", validationID)
	}

	return result, nil
}

// ListValidations lists all validation results
func (rv *ReadinessValidator) ListValidations() []*ValidationResult {
	rv.mu.RLock()
	defer rv.mu.RUnlock()

	results := make([]*ValidationResult, 0, len(rv.validations))
	for _, result := range rv.validations {
		results = append(results, result)
	}

	return results
}

// Shutdown gracefully shuts down the validator
func (rv *ReadinessValidator) Shutdown(ctx context.Context) error {
	rv.cancel()
	return nil
}

// generateValidationID generates a unique validation ID
func generateValidationID(deploymentID, version string) string {
	data := fmt.Sprintf("%s:%s:%d", deploymentID, version, time.Now().UnixNano())
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])[:16]
}

// MarshalJSON implements custom JSON marshaling
func (vr *ValidationResult) MarshalJSON() ([]byte, error) {
	type Alias ValidationResult

	return json.Marshal(&struct {
		*Alias
		PassPercentage float64 `json:"pass_percentage"`
	}{
		Alias:          (*Alias)(vr),
		PassPercentage: vr.OverallScore * 100,
	})
}

// HTTPReadinessProbe provides HTTP endpoint for readiness checks
type HTTPReadinessProbe struct {
	validator *ReadinessValidator
}

// NewHTTPReadinessProbe creates a new HTTP readiness probe
func NewHTTPReadinessProbe(validator *ReadinessValidator) *HTTPReadinessProbe {
	return &HTTPReadinessProbe{
		validator: validator,
	}
}

// ServeHTTP implements http.Handler
func (hrp *HTTPReadinessProbe) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	validationID := r.URL.Query().Get("validation_id")
	if validationID == "" {
		http.Error(w, "validation_id required", http.StatusBadRequest)
		return
	}

	result, err := hrp.validator.GetValidation(validationID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")

	if result.Passed {
		w.WriteHeader(http.StatusOK)
	} else {
		w.WriteHeader(http.StatusServiceUnavailable)
	}

	json.NewEncoder(w).Encode(result)
}
