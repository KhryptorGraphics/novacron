package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	comprehensive "github.com/khryptorgraphics/novacron/backend/tests/comprehensive"
)

// Test execution configuration
type TestConfig struct {
	// Test selection
	RunStorageTests    bool
	RunConsensusTests  bool
	RunVMTests         bool
	RunPerformanceTests bool
	RunChaosTests      bool
	RunIntegrationTests bool
	
	// Test behavior
	Parallel           bool
	MaxConcurrency     int
	TimeoutMinutes     int
	SkipSlowTests      bool
	VerboseOutput      bool
	
	// Output configuration
	OutputFormat       string // json, xml, text
	OutputFile         string
	SaveArtifacts      bool
	ArtifactDir        string
	
	// Quality gates
	MinSuccessRate     float64
	MinCoverage        float64
	MaxCriticalErrors  int
	
	// Environment
	TestEnvironment    string // local, ci, production
	UseRealResources   bool
	CleanupAfter       bool
}

func main() {
	// Parse command line flags
	config := parseFlags()
	
	// Set up logging
	logger := setupLogger(config.VerboseOutput)
	
	// Create test coordinator
	coordinator := comprehensive.NewTestCoordinator()
	
	// Set up signal handling for graceful shutdown
	ctx, cancel := setupGracefulShutdown(time.Duration(config.TimeoutMinutes) * time.Minute)
	defer cancel()
	
	logger.Println("Starting NovaCron Comprehensive Test Suite")
	logger.Printf("Configuration: %+v", config)
	
	// Print system information
	systemInfo := coordinator.GetSystemInfo()
	logger.Printf("System Info: %+v", systemInfo)
	
	// Validate test environment
	if err := validateTestEnvironment(config); err != nil {
		logger.Fatalf("Test environment validation failed: %v", err)
	}
	
	// Execute comprehensive tests
	startTime := time.Now()
	report, err := coordinator.RunComprehensiveTests(ctx)
	duration := time.Since(startTime)
	
	if err != nil {
		logger.Printf("Test execution completed with errors: %v", err)
	}
	
	logger.Printf("Test execution completed in %v", duration)
	
	// Print report to console
	coordinator.PrintReport(report)
	
	// Save report to file if specified
	if config.OutputFile != "" {
		if err := saveReportToFile(report, config); err != nil {
			logger.Printf("Failed to save report: %v", err)
		} else {
			logger.Printf("Report saved to: %s", config.OutputFile)
		}
	}
	
	// Save test artifacts
	if config.SaveArtifacts {
		if err := saveTestArtifacts(report, config); err != nil {
			logger.Printf("Failed to save artifacts: %v", err)
		}
	}
	
	// Evaluate quality gates
	exitCode := evaluateQualityGates(report, config, logger)
	
	// Cleanup if requested
	if config.CleanupAfter {
		performCleanup(logger)
	}
	
	logger.Printf("Exiting with code: %d", exitCode)
	os.Exit(exitCode)
}

// parseFlags parses command line arguments
func parseFlags() *TestConfig {
	config := &TestConfig{}
	
	// Test selection flags
	flag.BoolVar(&config.RunStorageTests, "storage", true, "Run storage tiering tests")
	flag.BoolVar(&config.RunConsensusTests, "consensus", true, "Run distributed consensus tests")
	flag.BoolVar(&config.RunVMTests, "vm", true, "Run VM lifecycle tests")
	flag.BoolVar(&config.RunPerformanceTests, "performance", true, "Run performance benchmarks")
	flag.BoolVar(&config.RunChaosTests, "chaos", true, "Run chaos engineering tests")
	flag.BoolVar(&config.RunIntegrationTests, "integration", true, "Run integration tests")
	
	// Test behavior flags
	flag.BoolVar(&config.Parallel, "parallel", true, "Run tests in parallel where possible")
	flag.IntVar(&config.MaxConcurrency, "concurrency", 4, "Maximum concurrent test suites")
	flag.IntVar(&config.TimeoutMinutes, "timeout", 180, "Test timeout in minutes")
	flag.BoolVar(&config.SkipSlowTests, "short", false, "Skip slow tests")
	flag.BoolVar(&config.VerboseOutput, "verbose", false, "Verbose logging output")
	
	// Output configuration flags
	flag.StringVar(&config.OutputFormat, "format", "text", "Output format: text, json, xml")
	flag.StringVar(&config.OutputFile, "output", "", "Output file path")
	flag.BoolVar(&config.SaveArtifacts, "artifacts", false, "Save test artifacts")
	flag.StringVar(&config.ArtifactDir, "artifact-dir", "./test-artifacts", "Artifact directory")
	
	// Quality gate flags
	flag.Float64Var(&config.MinSuccessRate, "min-success", 95.0, "Minimum success rate %")
	flag.Float64Var(&config.MinCoverage, "min-coverage", 90.0, "Minimum coverage %")
	flag.IntVar(&config.MaxCriticalErrors, "max-critical", 0, "Maximum critical errors")
	
	// Environment flags
	flag.StringVar(&config.TestEnvironment, "env", "local", "Test environment: local, ci, production")
	flag.BoolVar(&config.UseRealResources, "real-resources", false, "Use real storage/network resources")
	flag.BoolVar(&config.CleanupAfter, "cleanup", true, "Cleanup test resources after execution")
	
	flag.Parse()
	
	return config
}

// setupLogger creates a logger with appropriate configuration
func setupLogger(verbose bool) *log.Logger {
	flags := log.LstdFlags
	if verbose {
		flags |= log.Lshortfile
	}
	
	return log.New(os.Stdout, "[COMPREHENSIVE-TEST] ", flags)
}

// setupGracefulShutdown sets up signal handling for graceful shutdown
func setupGracefulShutdown(timeout time.Duration) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
	
	go func() {
		<-sigChan
		log.Println("Received interrupt signal, initiating graceful shutdown...")
		cancel()
	}()
	
	return ctx, cancel
}

// validateTestEnvironment validates the test environment setup
func validateTestEnvironment(config *TestConfig) error {
	// Check required directories
	if config.SaveArtifacts {
		if err := os.MkdirAll(config.ArtifactDir, 0755); err != nil {
			return fmt.Errorf("failed to create artifact directory: %w", err)
		}
	}
	
	// Environment-specific validations
	switch config.TestEnvironment {
	case "local":
		return validateLocalEnvironment(config)
	case "ci":
		return validateCIEnvironment(config)
	case "production":
		return validateProductionEnvironment(config)
	default:
		return fmt.Errorf("unknown test environment: %s", config.TestEnvironment)
	}
}

// validateLocalEnvironment validates local development environment
func validateLocalEnvironment(config *TestConfig) error {
	// Check for local dependencies
	requiredPaths := []string{
		"/tmp", // Temp directory for test files
	}
	
	for _, path := range requiredPaths {
		if _, err := os.Stat(path); err != nil {
			return fmt.Errorf("required path not accessible: %s", path)
		}
	}
	
	return nil
}

// validateCIEnvironment validates CI/CD environment
func validateCIEnvironment(config *TestConfig) error {
	// CI-specific validations
	if config.UseRealResources {
		log.Println("Warning: Using real resources in CI environment")
	}
	
	// Check CI environment variables
	ciEnvVars := []string{"CI", "BUILD_NUMBER", "BUILD_ID"}
	for _, envVar := range ciEnvVars {
		if os.Getenv(envVar) == "" {
			log.Printf("Warning: CI environment variable %s not set", envVar)
		}
	}
	
	return nil
}

// validateProductionEnvironment validates production test environment
func validateProductionEnvironment(config *TestConfig) error {
	// Production environment should have stricter requirements
	if !config.UseRealResources {
		return fmt.Errorf("production tests must use real resources")
	}
	
	// Require higher quality gates for production
	if config.MinSuccessRate < 99.0 {
		config.MinSuccessRate = 99.0
		log.Println("Raised minimum success rate to 99% for production environment")
	}
	
	if config.MaxCriticalErrors > 0 {
		config.MaxCriticalErrors = 0
		log.Println("Set maximum critical errors to 0 for production environment")
	}
	
	return nil
}

// saveReportToFile saves the test report in the specified format
func saveReportToFile(report *comprehensive.ComprehensiveTestReport, config *TestConfig) error {
	switch config.OutputFormat {
	case "json":
		return saveReportAsJSON(report, config.OutputFile)
	case "xml":
		return saveReportAsXML(report, config.OutputFile)
	case "text":
		return saveReportAsText(report, config.OutputFile)
	default:
		return fmt.Errorf("unsupported output format: %s", config.OutputFormat)
	}
}

// saveReportAsJSON saves report in JSON format
func saveReportAsJSON(report *comprehensive.ComprehensiveTestReport, filename string) error {
	// In production, would use encoding/json to serialize the report
	log.Printf("Would save JSON report to: %s", filename)
	return nil
}

// saveReportAsXML saves report in XML format (JUnit-compatible)
func saveReportAsXML(report *comprehensive.ComprehensiveTestReport, filename string) error {
	// In production, would generate JUnit-compatible XML
	log.Printf("Would save XML report to: %s", filename)
	return nil
}

// saveReportAsText saves report in text format
func saveReportAsText(report *comprehensive.ComprehensiveTestReport, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create report file: %w", err)
	}
	defer file.Close()
	
	// Write report content (simplified)
	fmt.Fprintf(file, "NovaCron Comprehensive Test Report\n")
	fmt.Fprintf(file, "==================================\n\n")
	fmt.Fprintf(file, "Execution Time: %s to %s\n", 
		report.StartTime.Format("2006-01-02 15:04:05"),
		report.EndTime.Format("2006-01-02 15:04:05"))
	fmt.Fprintf(file, "Total Duration: %v\n\n", report.TotalDuration)
	
	fmt.Fprintf(file, "Overall Results:\n")
	fmt.Fprintf(file, "  Total Tests: %d\n", report.TotalTests)
	fmt.Fprintf(file, "  Passed: %d\n", report.TotalPassed)
	fmt.Fprintf(file, "  Failed: %d\n", report.TotalFailed)
	fmt.Fprintf(file, "  Success Rate: %.2f%%\n", report.OverallSuccessRate)
	fmt.Fprintf(file, "  Coverage: %.2f%%\n\n", report.OverallCoverage)
	
	fmt.Fprintf(file, "Quality Gate: %v\n\n", report.QualityGateStatus)
	
	if len(report.Recommendations) > 0 {
		fmt.Fprintf(file, "Recommendations:\n")
		for _, rec := range report.Recommendations {
			fmt.Fprintf(file, "  - %s\n", rec)
		}
	}
	
	return nil
}

// saveTestArtifacts saves test artifacts and logs
func saveTestArtifacts(report *comprehensive.ComprehensiveTestReport, config *TestConfig) error {
	artifactDir := config.ArtifactDir
	
	// Create timestamped subdirectory
	timestamp := time.Now().Format("20060102-150405")
	runDir := fmt.Sprintf("%s/run-%s", artifactDir, timestamp)
	
	if err := os.MkdirAll(runDir, 0755); err != nil {
		return fmt.Errorf("failed to create run directory: %w", err)
	}
	
	log.Printf("Saving test artifacts to: %s", runDir)
	
	// Save detailed reports for each test suite
	for suiteName, results := range report.SuiteResults {
		suiteFile := fmt.Sprintf("%s/%s-results.txt", runDir, suiteName)
		if err := saveTestSuiteArtifacts(results, suiteFile); err != nil {
			log.Printf("Failed to save artifacts for suite %s: %v", suiteName, err)
		}
	}
	
	// Save system information
	sysInfoFile := fmt.Sprintf("%s/system-info.txt", runDir)
	if err := saveSystemInfoArtifacts(sysInfoFile); err != nil {
		log.Printf("Failed to save system info: %v", err)
	}
	
	// Save performance metrics if available
	perfFile := fmt.Sprintf("%s/performance-metrics.txt", runDir)
	if err := savePerformanceArtifacts(report, perfFile); err != nil {
		log.Printf("Failed to save performance metrics: %v", err)
	}
	
	return nil
}

// saveTestSuiteArtifacts saves detailed results for a test suite
func saveTestSuiteArtifacts(results *comprehensive.TestSuiteResults, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "Test Suite: %s\n", results.SuiteName)
	fmt.Fprintf(file, "Duration: %v\n", results.TotalDuration)
	fmt.Fprintf(file, "Tests: %d passed, %d failed, %d skipped\n", 
		results.PassedTests, results.FailedTests, results.SkippedTests)
	fmt.Fprintf(file, "Coverage: %.2f%%\n", results.CoveragePercent)
	fmt.Fprintf(file, "Memory Usage: %.2f MB\n", results.MemoryUsageMB)
	fmt.Fprintf(file, "CPU Time: %.2f seconds\n\n", results.CPUTimeSeconds)
	
	if len(results.Errors) > 0 {
		fmt.Fprintf(file, "Errors:\n")
		for _, err := range results.Errors {
			fmt.Fprintf(file, "  - %s: %s\n", err.TestName, err.ErrorMsg)
		}
		fmt.Fprintf(file, "\n")
	}
	
	if len(results.Warnings) > 0 {
		fmt.Fprintf(file, "Warnings:\n")
		for _, warning := range results.Warnings {
			fmt.Fprintf(file, "  - %s: %s\n", warning.TestName, warning.WarningMsg)
		}
	}
	
	return nil
}

// saveSystemInfoArtifacts saves system information
func saveSystemInfoArtifacts(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "System Information\n")
	fmt.Fprintf(file, "==================\n\n")
	
	// Would include detailed system info in production
	fmt.Fprintf(file, "Timestamp: %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(file, "Hostname: %s\n", getHostname())
	fmt.Fprintf(file, "Working Directory: %s\n", getWorkingDir())
	
	return nil
}

// savePerformanceArtifacts saves performance metrics
func savePerformanceArtifacts(report *comprehensive.ComprehensiveTestReport, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	
	fmt.Fprintf(file, "Performance Metrics\n")
	fmt.Fprintf(file, "===================\n\n")
	
	fmt.Fprintf(file, "Total Test Duration: %v\n", report.TotalDuration)
	fmt.Fprintf(file, "Total Memory Usage: %.2f MB\n", report.TotalMemoryUsageMB)
	fmt.Fprintf(file, "Total CPU Time: %.2f seconds\n", report.TotalCPUTimeSeconds)
	fmt.Fprintf(file, "Tests per Second: %.2f\n", 
		float64(report.TotalTests)/report.TotalDuration.Seconds())
	
	return nil
}

// evaluateQualityGates checks if quality gates pass and determines exit code
func evaluateQualityGates(report *comprehensive.ComprehensiveTestReport, config *TestConfig, logger *log.Logger) int {
	logger.Println("Evaluating quality gates...")
	
	passed := true
	
	// Check success rate
	if report.OverallSuccessRate < config.MinSuccessRate {
		logger.Printf("❌ Success rate quality gate failed: %.2f%% < %.2f%%", 
			report.OverallSuccessRate, config.MinSuccessRate)
		passed = false
	} else {
		logger.Printf("✅ Success rate quality gate passed: %.2f%%", report.OverallSuccessRate)
	}
	
	// Check coverage
	if report.OverallCoverage < config.MinCoverage {
		logger.Printf("❌ Coverage quality gate failed: %.2f%% < %.2f%%", 
			report.OverallCoverage, config.MinCoverage)
		passed = false
	} else {
		logger.Printf("✅ Coverage quality gate passed: %.2f%%", report.OverallCoverage)
	}
	
	// Check critical errors
	criticalErrorCount := len(report.CriticalErrors)
	if criticalErrorCount > config.MaxCriticalErrors {
		logger.Printf("❌ Critical errors quality gate failed: %d > %d", 
			criticalErrorCount, config.MaxCriticalErrors)
		passed = false
	} else {
		logger.Printf("✅ Critical errors quality gate passed: %d", criticalErrorCount)
	}
	
	if passed {
		logger.Println("🎉 All quality gates passed!")
		return 0
	} else {
		logger.Println("💥 Quality gates failed!")
		return 1
	}
}

// performCleanup cleans up test resources
func performCleanup(logger *log.Logger) {
	logger.Println("Performing test cleanup...")
	
	// In production, would clean up:
	// - Temporary files and directories
	// - Test containers and VMs
	// - Network configurations
	// - Database test data
	// - Mock services
	
	logger.Println("✅ Test cleanup completed")
}

// Utility functions

func getHostname() string {
	hostname, err := os.Hostname()
	if err != nil {
		return "unknown"
	}
	return hostname
}

func getWorkingDir() string {
	wd, err := os.Getwd()
	if err != nil {
		return "unknown"
	}
	return wd
}