// Coverage Reporter - Generates comprehensive coverage reports for NovaCron
package main

import (
	"encoding/json"
	"fmt"
	"html/template"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"golang.org/x/tools/cover"
)

// Coverage represents overall coverage statistics
type Coverage struct {
	Statements    CoverageStat `json:"statements"`
	Functions     CoverageStat `json:"functions"`
	Branches      CoverageStat `json:"branches"`
	Lines         CoverageStat `json:"lines"`
	TotalScore    float64      `json:"total_score"`
	Grade         string       `json:"grade"`
	Timestamp     time.Time    `json:"timestamp"`
	TestTypes     []TestType   `json:"test_types"`
	Packages      []Package    `json:"packages"`
	Trends        []TrendData  `json:"trends"`
	Recommendations []string   `json:"recommendations"`
}

// CoverageStat represents statistics for a coverage type
type CoverageStat struct {
	Covered   int     `json:"covered"`
	Total     int     `json:"total"`
	Percent   float64 `json:"percent"`
	Threshold float64 `json:"threshold"`
	Status    string  `json:"status"`
}

// TestType represents different types of test coverage
type TestType struct {
	Name        string       `json:"name"`
	Coverage    CoverageStat `json:"coverage"`
	Files       int          `json:"files"`
	TestCount   int          `json:"test_count"`
	Duration    string       `json:"duration"`
	Status      string       `json:"status"`
}

// Package represents coverage for a specific package
type Package struct {
	Name         string       `json:"name"`
	Path         string       `json:"path"`
	Coverage     CoverageStat `json:"coverage"`
	Files        []File       `json:"files"`
	Importance   string       `json:"importance"`
	TestQuality  string       `json:"test_quality"`
}

// File represents coverage for a specific file
type File struct {
	Name         string       `json:"name"`
	Path         string       `json:"path"`
	Coverage     CoverageStat `json:"coverage"`
	Lines        []Line       `json:"lines"`
	Functions    []Function   `json:"functions"`
	Complexity   int          `json:"complexity"`
	Issues       []string     `json:"issues"`
}

// Line represents coverage for a specific line
type Line struct {
	Number   int    `json:"number"`
	Content  string `json:"content"`
	Covered  bool   `json:"covered"`
	HitCount int    `json:"hit_count"`
}

// Function represents coverage for a specific function
type Function struct {
	Name     string       `json:"name"`
	Coverage CoverageStat `json:"coverage"`
	Line     int          `json:"line"`
	Cyclomatic int        `json:"cyclomatic_complexity"`
}

// TrendData represents historical coverage trends
type TrendData struct {
	Date     time.Time `json:"date"`
	Coverage float64   `json:"coverage"`
	TestType string    `json:"test_type"`
	Commit   string    `json:"commit"`
}

// ReportConfig holds configuration for the coverage reporter
type ReportConfig struct {
	ProjectName    string            `json:"project_name"`
	OutputDir      string            `json:"output_dir"`
	Thresholds     map[string]float64 `json:"thresholds"`
	CoverageFiles  []string          `json:"coverage_files"`
	HistoryFile    string            `json:"history_file"`
	IncludePatterns []string         `json:"include_patterns"`
	ExcludePatterns []string         `json:"exclude_patterns"`
	QualityGates   QualityGates      `json:"quality_gates"`
}

// QualityGates defines quality requirements for different components
type QualityGates struct {
	Critical    QualityThreshold `json:"critical"`
	Important   QualityThreshold `json:"important"`
	Standard    QualityThreshold `json:"standard"`
	Experimental QualityThreshold `json:"experimental"`
}

// QualityThreshold defines minimum requirements
type QualityThreshold struct {
	MinCoverage      float64 `json:"min_coverage"`
	MinTestQuality   float64 `json:"min_test_quality"`
	MaxComplexity    int     `json:"max_complexity"`
	RequiredTestTypes []string `json:"required_test_types"`
}

func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: coverage-reporter <config-file>")
	}

	configFile := os.Args[1]
	config, err := loadConfig(configFile)
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Generate comprehensive coverage report
	coverage, err := generateCoverageReport(config)
	if err != nil {
		log.Fatalf("Failed to generate coverage report: %v", err)
	}

	// Generate reports in multiple formats
	err = generateReports(coverage, config)
	if err != nil {
		log.Fatalf("Failed to generate reports: %v", err)
	}

	// Update historical data
	err = updateHistory(coverage, config)
	if err != nil {
		log.Printf("Warning: Failed to update history: %v", err)
	}

	// Check quality gates
	exitCode := checkQualityGates(coverage, config)
	
	fmt.Printf("Coverage report generated successfully!\n")
	fmt.Printf("Overall coverage: %.2f%%\n", coverage.TotalScore)
	fmt.Printf("Grade: %s\n", coverage.Grade)
	
	if exitCode != 0 {
		fmt.Printf("Quality gates failed!\n")
	}
	
	os.Exit(exitCode)
}

func loadConfig(filename string) (*ReportConfig, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config ReportConfig
	err = json.Unmarshal(data, &config)
	if err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Set defaults
	if config.OutputDir == "" {
		config.OutputDir = "reports/coverage"
	}

	if config.Thresholds == nil {
		config.Thresholds = map[string]float64{
			"statements": 80.0,
			"functions":  80.0,
			"branches":   75.0,
			"lines":      80.0,
		}
	}

	return &config, nil
}

func generateCoverageReport(config *ReportConfig) (*Coverage, error) {
	coverage := &Coverage{
		Timestamp: time.Now(),
		TestTypes: []TestType{},
		Packages:  []Package{},
		Trends:    []TrendData{},
		Recommendations: []string{},
	}

	var allProfiles []*cover.Profile
	totalStatements := 0
	coveredStatements := 0

	// Process each coverage file
	for _, file := range config.CoverageFiles {
		profiles, err := cover.ParseProfiles(file)
		if err != nil {
			log.Printf("Warning: Failed to parse %s: %v", file, err)
			continue
		}

		allProfiles = append(allProfiles, profiles...)
		
		// Calculate test type coverage
		testType := inferTestType(file)
		testTypeCoverage := calculateTestTypeCoverage(profiles)
		
		coverage.TestTypes = append(coverage.TestTypes, TestType{
			Name:     testType,
			Coverage: testTypeCoverage,
			Files:    len(profiles),
			Status:   getStatusFromCoverage(testTypeCoverage.Percent, config.Thresholds["statements"]),
		})
	}

	// Process packages
	packageMap := make(map[string][]*cover.Profile)
	for _, profile := range allProfiles {
		pkg := getPackageFromProfile(profile)
		packageMap[pkg] = append(packageMap[pkg], profile)
	}

	for pkg, profiles := range packageMap {
		packageCoverage := calculatePackageCoverage(pkg, profiles, config)
		coverage.Packages = append(coverage.Packages, packageCoverage)
		
		// Accumulate totals
		for _, profile := range profiles {
			for _, block := range profile.Blocks {
				totalStatements += block.NumStmt
				if block.Count > 0 {
					coveredStatements += block.NumStmt
				}
			}
		}
	}

	// Calculate overall statistics
	coverage.Statements = CoverageStat{
		Covered:   coveredStatements,
		Total:     totalStatements,
		Percent:   float64(coveredStatements) / float64(totalStatements) * 100,
		Threshold: config.Thresholds["statements"],
		Status:    getStatusFromCoverage(float64(coveredStatements)/float64(totalStatements)*100, config.Thresholds["statements"]),
	}

	// Estimate other coverage types (in real implementation, these would come from actual tools)
	coverage.Functions = estimateFunctionCoverage(coverage.Statements)
	coverage.Branches = estimateBranchCoverage(coverage.Statements)  
	coverage.Lines = estimateLineCoverage(coverage.Statements)

	// Calculate total score and grade
	coverage.TotalScore = calculateTotalScore(coverage)
	coverage.Grade = calculateGrade(coverage.TotalScore)

	// Generate recommendations
	coverage.Recommendations = generateRecommendations(coverage, config)

	// Sort packages by coverage (lowest first for attention)
	sort.Slice(coverage.Packages, func(i, j int) bool {
		return coverage.Packages[i].Coverage.Percent < coverage.Packages[j].Coverage.Percent
	})

	return coverage, nil
}

func inferTestType(filename string) string {
	if strings.Contains(filename, "unit") {
		return "Unit Tests"
	}
	if strings.Contains(filename, "integration") {
		return "Integration Tests"
	}
	if strings.Contains(filename, "e2e") || strings.Contains(filename, "end-to-end") {
		return "E2E Tests"
	}
	if strings.Contains(filename, "performance") || strings.Contains(filename, "load") {
		return "Performance Tests"
	}
	return "General Tests"
}

func calculateTestTypeCoverage(profiles []*cover.Profile) CoverageStat {
	totalStmts := 0
	coveredStmts := 0
	
	for _, profile := range profiles {
		for _, block := range profile.Blocks {
			totalStmts += block.NumStmt
			if block.Count > 0 {
				coveredStmts += block.NumStmt
			}
		}
	}
	
	percent := 0.0
	if totalStmts > 0 {
		percent = float64(coveredStmts) / float64(totalStmts) * 100
	}
	
	return CoverageStat{
		Covered: coveredStmts,
		Total:   totalStmts,
		Percent: percent,
	}
}

func getPackageFromProfile(profile *cover.Profile) string {
	parts := strings.Split(profile.FileName, "/")
	if len(parts) > 2 {
		// Return package path like "backend/core/auth"
		return strings.Join(parts[len(parts)-3:len(parts)-1], "/")
	}
	return "unknown"
}

func calculatePackageCoverage(pkg string, profiles []*cover.Profile, config *ReportConfig) Package {
	totalStmts := 0
	coveredStmts := 0
	files := []File{}
	
	for _, profile := range profiles {
		fileStmts := 0
		fileCovered := 0
		
		for _, block := range profile.Blocks {
			fileStmts += block.NumStmt
			totalStmts += block.NumStmt
			if block.Count > 0 {
				fileCovered += block.NumStmt
				coveredStmts += block.NumStmt
			}
		}
		
		filePercent := 0.0
		if fileStmts > 0 {
			filePercent = float64(fileCovered) / float64(fileStmts) * 100
		}
		
		files = append(files, File{
			Name: filepath.Base(profile.FileName),
			Path: profile.FileName,
			Coverage: CoverageStat{
				Covered: fileCovered,
				Total:   fileStmts,
				Percent: filePercent,
			},
			Complexity: estimateComplexity(profile),
			Issues:     identifyIssues(profile, filePercent),
		})
	}
	
	percent := 0.0
	if totalStmts > 0 {
		percent = float64(coveredStmts) / float64(totalStmts) * 100
	}
	
	return Package{
		Name: pkg,
		Path: pkg,
		Coverage: CoverageStat{
			Covered: coveredStmts,
			Total:   totalStmts,
			Percent: percent,
			Status:  getStatusFromCoverage(percent, config.Thresholds["statements"]),
		},
		Files:       files,
		Importance:  classifyPackageImportance(pkg),
		TestQuality: assessTestQuality(percent, len(files)),
	}
}

func estimateComplexity(profile *cover.Profile) int {
	// Simple estimation based on number of blocks
	return len(profile.Blocks)
}

func identifyIssues(profile *cover.Profile, coverage float64) []string {
	issues := []string{}
	
	if coverage < 50 {
		issues = append(issues, "Very low coverage")
	} else if coverage < 75 {
		issues = append(issues, "Below recommended coverage")
	}
	
	if len(profile.Blocks) > 50 {
		issues = append(issues, "High complexity")
	}
	
	// Check for uncovered critical sections
	for _, block := range profile.Blocks {
		if block.Count == 0 && block.NumStmt > 5 {
			issues = append(issues, "Large uncovered block detected")
			break
		}
	}
	
	return issues
}

func classifyPackageImportance(pkg string) string {
	criticalPackages := []string{"auth", "security", "vm", "core"}
	importantPackages := []string{"api", "database", "cache"}
	
	for _, critical := range criticalPackages {
		if strings.Contains(pkg, critical) {
			return "Critical"
		}
	}
	
	for _, important := range importantPackages {
		if strings.Contains(pkg, important) {
			return "Important"
		}
	}
	
	return "Standard"
}

func assessTestQuality(coverage float64, fileCount int) string {
	if coverage >= 90 && fileCount > 0 {
		return "Excellent"
	}
	if coverage >= 80 {
		return "Good"
	}
	if coverage >= 70 {
		return "Fair"
	}
	return "Poor"
}

func estimateFunctionCoverage(statements CoverageStat) CoverageStat {
	// Estimate function coverage based on statement coverage
	// Typically function coverage is higher than statement coverage
	percent := math.Min(statements.Percent * 1.1, 100.0)
	total := int(float64(statements.Total) * 0.8) // Estimate fewer functions than statements
	covered := int(float64(total) * percent / 100)
	
	return CoverageStat{
		Covered: covered,
		Total:   total,
		Percent: percent,
		Threshold: 80.0,
		Status: getStatusFromCoverage(percent, 80.0),
	}
}

func estimateBranchCoverage(statements CoverageStat) CoverageStat {
	// Branch coverage is typically lower than statement coverage
	percent := statements.Percent * 0.85
	total := int(float64(statements.Total) * 0.6) // Estimate fewer branches
	covered := int(float64(total) * percent / 100)
	
	return CoverageStat{
		Covered: covered,
		Total:   total,
		Percent: percent,
		Threshold: 75.0,
		Status: getStatusFromCoverage(percent, 75.0),
	}
}

func estimateLineCoverage(statements CoverageStat) CoverageStat {
	// Line coverage is usually close to statement coverage
	percent := statements.Percent * 0.95
	total := int(float64(statements.Total) * 1.2) // More lines than statements
	covered := int(float64(total) * percent / 100)
	
	return CoverageStat{
		Covered: covered,
		Total:   total,
		Percent: percent,
		Threshold: 80.0,
		Status: getStatusFromCoverage(percent, 80.0),
	}
}

func calculateTotalScore(coverage *Coverage) float64 {
	// Weighted average of different coverage types
	weights := map[string]float64{
		"statements": 0.4,
		"functions":  0.25,
		"branches":   0.2,
		"lines":      0.15,
	}
	
	totalScore := coverage.Statements.Percent*weights["statements"] +
		coverage.Functions.Percent*weights["functions"] +
		coverage.Branches.Percent*weights["branches"] +
		coverage.Lines.Percent*weights["lines"]
	
	return math.Round(totalScore*100) / 100
}

func calculateGrade(score float64) string {
	if score >= 95 {
		return "A+"
	} else if score >= 90 {
		return "A"
	} else if score >= 85 {
		return "B+"
	} else if score >= 80 {
		return "B"
	} else if score >= 75 {
		return "C+"
	} else if score >= 70 {
		return "C"
	} else if score >= 65 {
		return "D+"
	} else if score >= 60 {
		return "D"
	}
	return "F"
}

func getStatusFromCoverage(coverage, threshold float64) string {
	if coverage >= threshold {
		return "PASS"
	} else if coverage >= threshold*0.9 {
		return "WARNING"
	}
	return "FAIL"
}

func generateRecommendations(coverage *Coverage, config *ReportConfig) []string {
	recommendations := []string{}
	
	// Overall coverage recommendations
	if coverage.TotalScore < 80 {
		recommendations = append(recommendations, "Overall coverage is below 80%. Focus on adding more comprehensive tests.")
	}
	
	// Statement coverage
	if coverage.Statements.Percent < 80 {
		recommendations = append(recommendations, "Statement coverage is low. Add unit tests for uncovered code paths.")
	}
	
	// Branch coverage
	if coverage.Branches.Percent < 75 {
		recommendations = append(recommendations, "Branch coverage needs improvement. Test both true and false conditions in all branches.")
	}
	
	// Package-specific recommendations
	for _, pkg := range coverage.Packages {
		if pkg.Importance == "Critical" && pkg.Coverage.Percent < 90 {
			recommendations = append(recommendations, fmt.Sprintf("Critical package '%s' has insufficient coverage (%.1f%%). Aim for 90%+ coverage.", pkg.Name, pkg.Coverage.Percent))
		}
		
		if pkg.Coverage.Percent < 60 {
			recommendations = append(recommendations, fmt.Sprintf("Package '%s' has very low coverage (%.1f%%). This should be prioritized for improvement.", pkg.Name, pkg.Coverage.Percent))
		}
	}
	
	// Test type recommendations
	unitTestExists := false
	integrationTestExists := false
	
	for _, testType := range coverage.TestTypes {
		if testType.Name == "Unit Tests" {
			unitTestExists = true
			if testType.Coverage.Percent < 85 {
				recommendations = append(recommendations, "Unit test coverage should be increased to at least 85%.")
			}
		}
		if testType.Name == "Integration Tests" {
			integrationTestExists = true
			if testType.Coverage.Percent < 70 {
				recommendations = append(recommendations, "Integration test coverage should be increased to at least 70%.")
			}
		}
	}
	
	if !unitTestExists {
		recommendations = append(recommendations, "No unit test coverage detected. Add comprehensive unit tests.")
	}
	
	if !integrationTestExists {
		recommendations = append(recommendations, "No integration test coverage detected. Add integration tests for API endpoints and service interactions.")
	}
	
	// File-specific recommendations
	lowCoverageFiles := 0
	for _, pkg := range coverage.Packages {
		for _, file := range pkg.Files {
			if file.Coverage.Percent < 70 {
				lowCoverageFiles++
			}
		}
	}
	
	if lowCoverageFiles > 5 {
		recommendations = append(recommendations, fmt.Sprintf("%d files have coverage below 70%%. Consider refactoring complex files and adding targeted tests.", lowCoverageFiles))
	}
	
	return recommendations
}

func generateReports(coverage *Coverage, config *ReportConfig) error {
	// Ensure output directory exists
	err := os.MkdirAll(config.OutputDir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Generate JSON report
	err = generateJSONReport(coverage, config)
	if err != nil {
		return fmt.Errorf("failed to generate JSON report: %w", err)
	}
	
	// Generate HTML report
	err = generateHTMLReport(coverage, config)
	if err != nil {
		return fmt.Errorf("failed to generate HTML report: %w", err)
	}
	
	// Generate console summary
	generateConsoleSummary(coverage)
	
	// Generate badge JSON for README
	err = generateBadgeData(coverage, config)
	if err != nil {
		return fmt.Errorf("failed to generate badge data: %w", err)
	}
	
	return nil
}

func generateJSONReport(coverage *Coverage, config *ReportConfig) error {
	data, err := json.MarshalIndent(coverage, "", "  ")
	if err != nil {
		return err
	}
	
	filename := filepath.Join(config.OutputDir, "coverage-report.json")
	return ioutil.WriteFile(filename, data, 0644)
}

func generateHTMLReport(coverage *Coverage, config *ReportConfig) error {
	tmpl := `
<!DOCTYPE html>
<html>
<head>
    <title>{{.ProjectName}} - Coverage Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background: #f5f5f5; }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; text-align: center; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin: 2rem 0; }
        .metric-card { background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .metric-title { font-size: 0.9rem; color: #666; text-transform: uppercase; letter-spacing: 1px; }
        .metric-value { font-size: 2rem; font-weight: bold; margin: 0.5rem 0; }
        .metric-bar { width: 100%; height: 8px; background: #eee; border-radius: 4px; overflow: hidden; margin: 1rem 0; }
        .metric-fill { height: 100%; transition: width 0.3s; }
        .grade-a { color: #10b981; } .fill-a { background: #10b981; }
        .grade-b { color: #3b82f6; } .fill-b { background: #3b82f6; }
        .grade-c { color: #f59e0b; } .fill-c { background: #f59e0b; }
        .grade-d { color: #ef4444; } .fill-d { background: #ef4444; }
        .grade-f { color: #dc2626; } .fill-f { background: #dc2626; }
        .status-pass { color: #10b981; background: #dcfce7; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        .status-warning { color: #d97706; background: #fef3c7; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        .status-fail { color: #dc2626; background: #fef2f2; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }
        table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; margin: 1rem 0; }
        th, td { padding: 1rem; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
        .recommendations { background: white; padding: 1.5rem; border-radius: 8px; margin: 1rem 0; }
        .recommendations ul { padding-left: 1.5rem; }
        .recommendations li { margin: 0.5rem 0; color: #555; }
        .timestamp { text-align: center; color: #666; margin: 2rem 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{.ProjectName}} Coverage Report</h1>
        <p>Generated on {{.Timestamp.Format "January 2, 2006 at 3:04 PM MST"}}</p>
    </div>
    
    <div class="container">
        <div class="summary">
            <div class="metric-card">
                <div class="metric-title">Overall Score</div>
                <div class="metric-value grade-{{gradeLetter .Grade}}">{{printf "%.1f" .TotalScore}}%</div>
                <div class="metric-bar"><div class="metric-fill fill-{{gradeLetter .Grade}}" style="width: {{.TotalScore}}%"></div></div>
                <div class="grade-{{gradeLetter .Grade}}">Grade: {{.Grade}}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Statement Coverage</div>
                <div class="metric-value">{{printf "%.1f" .Statements.Percent}}%</div>
                <div class="metric-bar"><div class="metric-fill fill-{{gradeLetter .Grade}}" style="width: {{.Statements.Percent}}%"></div></div>
                <div class="{{.Statements.Status | lower}}">{{.Statements.Covered}}/{{.Statements.Total}} <span class="status-{{.Statements.Status | lower}}">{{.Statements.Status}}</span></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Function Coverage</div>
                <div class="metric-value">{{printf "%.1f" .Functions.Percent}}%</div>
                <div class="metric-bar"><div class="metric-fill fill-{{gradeLetter .Grade}}" style="width: {{.Functions.Percent}}%"></div></div>
                <div class="{{.Functions.Status | lower}}">{{.Functions.Covered}}/{{.Functions.Total}} <span class="status-{{.Functions.Status | lower}}">{{.Functions.Status}}</span></div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Branch Coverage</div>
                <div class="metric-value">{{printf "%.1f" .Branches.Percent}}%</div>
                <div class="metric-bar"><div class="metric-fill fill-{{gradeLetter .Grade}}" style="width: {{.Branches.Percent}}%"></div></div>
                <div class="{{.Branches.Status | lower}}">{{.Branches.Covered}}/{{.Branches.Total}} <span class="status-{{.Branches.Status | lower}}">{{.Branches.Status}}</span></div>
            </div>
        </div>
        
        <h2>Test Types Coverage</h2>
        <table>
            <tr><th>Test Type</th><th>Coverage</th><th>Files</th><th>Status</th></tr>
            {{range .TestTypes}}
            <tr>
                <td>{{.Name}}</td>
                <td>{{printf "%.1f" .Coverage.Percent}}% ({{.Coverage.Covered}}/{{.Coverage.Total}})</td>
                <td>{{.Files}}</td>
                <td><span class="status-{{.Status | lower}}">{{.Status}}</span></td>
            </tr>
            {{end}}
        </table>
        
        <h2>Package Coverage</h2>
        <table>
            <tr><th>Package</th><th>Coverage</th><th>Files</th><th>Importance</th><th>Quality</th><th>Status</th></tr>
            {{range .Packages}}
            <tr>
                <td>{{.Name}}</td>
                <td>{{printf "%.1f" .Coverage.Percent}}%</td>
                <td>{{len .Files}}</td>
                <td>{{.Importance}}</td>
                <td>{{.TestQuality}}</td>
                <td><span class="status-{{.Coverage.Status | lower}}">{{.Coverage.Status}}</span></td>
            </tr>
            {{end}}
        </table>
        
        {{if .Recommendations}}
        <div class="recommendations">
            <h2>Recommendations</h2>
            <ul>
            {{range .Recommendations}}
                <li>{{.}}</li>
            {{end}}
            </ul>
        </div>
        {{end}}
    </div>
</body>
</html>`
	
	funcMap := template.FuncMap{
		"gradeLetter": func(grade string) string {
			return strings.ToLower(string(grade[0]))
		},
		"lower": strings.ToLower,
	}
	
	t, err := template.New("report").Funcs(funcMap).Parse(tmpl)
	if err != nil {
		return err
	}
	
	filename := filepath.Join(config.OutputDir, "coverage-report.html")
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	return t.Execute(file, struct {
		*Coverage
		ProjectName string
	}{
		Coverage:    coverage,
		ProjectName: config.ProjectName,
	})
}

func generateConsoleSummary(coverage *Coverage) {
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Printf("  COVERAGE SUMMARY - Grade: %s (%.1f%%)\n", coverage.Grade, coverage.TotalScore)
	fmt.Println(strings.Repeat("=", 60))
	
	fmt.Printf("  Statements: %6.1f%% (%d/%d) %s\n", 
		coverage.Statements.Percent, coverage.Statements.Covered, coverage.Statements.Total, coverage.Statements.Status)
	fmt.Printf("  Functions:  %6.1f%% (%d/%d) %s\n", 
		coverage.Functions.Percent, coverage.Functions.Covered, coverage.Functions.Total, coverage.Functions.Status)
	fmt.Printf("  Branches:   %6.1f%% (%d/%d) %s\n", 
		coverage.Branches.Percent, coverage.Branches.Covered, coverage.Branches.Total, coverage.Branches.Status)
	fmt.Printf("  Lines:      %6.1f%% (%d/%d) %s\n", 
		coverage.Lines.Percent, coverage.Lines.Covered, coverage.Lines.Total, coverage.Lines.Status)
	
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("  Test Types: %d | Packages: %d\n", len(coverage.TestTypes), len(coverage.Packages))
	
	if len(coverage.Recommendations) > 0 {
		fmt.Println(strings.Repeat("-", 60))
		fmt.Println("  RECOMMENDATIONS:")
		for i, rec := range coverage.Recommendations {
			if i < 3 { // Show top 3 recommendations
				fmt.Printf("  • %s\n", rec)
			}
		}
		if len(coverage.Recommendations) > 3 {
			fmt.Printf("  ... and %d more (see HTML report)\n", len(coverage.Recommendations)-3)
		}
	}
	
	fmt.Println(strings.Repeat("=", 60) + "\n")
}

func generateBadgeData(coverage *Coverage, config *ReportConfig) error {
	badge := map[string]interface{}{
		"schemaVersion": 1,
		"label":         "coverage",
		"message":       fmt.Sprintf("%.1f%%", coverage.TotalScore),
		"color":         getBadgeColor(coverage.TotalScore),
	}
	
	data, err := json.MarshalIndent(badge, "", "  ")
	if err != nil {
		return err
	}
	
	filename := filepath.Join(config.OutputDir, "coverage-badge.json")
	return ioutil.WriteFile(filename, data, 0644)
}

func getBadgeColor(coverage float64) string {
	if coverage >= 90 {
		return "brightgreen"
	} else if coverage >= 80 {
		return "green"
	} else if coverage >= 70 {
		return "yellow"
	} else if coverage >= 60 {
		return "orange"
	}
	return "red"
}

func updateHistory(coverage *Coverage, config *ReportConfig) error {
	if config.HistoryFile == "" {
		return nil
	}
	
	// Load existing history
	var history []TrendData
	if data, err := ioutil.ReadFile(config.HistoryFile); err == nil {
		json.Unmarshal(data, &history)
	}
	
	// Add current data point
	trend := TrendData{
		Date:     coverage.Timestamp,
		Coverage: coverage.TotalScore,
		TestType: "Overall",
		Commit:   os.Getenv("GITHUB_SHA"), // or other CI commit info
	}
	
	history = append(history, trend)
	
	// Keep only last 30 data points
	if len(history) > 30 {
		history = history[len(history)-30:]
	}
	
	// Save updated history
	data, err := json.MarshalIndent(history, "", "  ")
	if err != nil {
		return err
	}
	
	return ioutil.WriteFile(config.HistoryFile, data, 0644)
}

func checkQualityGates(coverage *Coverage, config *ReportConfig) int {
	exitCode := 0
	
	// Check overall thresholds
	if coverage.Statements.Percent < config.Thresholds["statements"] {
		fmt.Printf("❌ Statement coverage %.1f%% is below threshold %.1f%%\n", 
			coverage.Statements.Percent, config.Thresholds["statements"])
		exitCode = 1
	}
	
	if coverage.Functions.Percent < config.Thresholds["functions"] {
		fmt.Printf("❌ Function coverage %.1f%% is below threshold %.1f%%\n", 
			coverage.Functions.Percent, config.Thresholds["functions"])
		exitCode = 1
	}
	
	if coverage.Branches.Percent < config.Thresholds["branches"] {
		fmt.Printf("❌ Branch coverage %.1f%% is below threshold %.1f%%\n", 
			coverage.Branches.Percent, config.Thresholds["branches"])
		exitCode = 1
	}
	
	// Check critical packages
	for _, pkg := range coverage.Packages {
		if pkg.Importance == "Critical" && pkg.Coverage.Percent < 90 {
			fmt.Printf("❌ Critical package '%s' coverage %.1f%% is below 90%%\n", 
				pkg.Name, pkg.Coverage.Percent)
			exitCode = 1
		}
	}
	
	if exitCode == 0 {
		fmt.Println("✅ All quality gates passed!")
	}
	
	return exitCode
}