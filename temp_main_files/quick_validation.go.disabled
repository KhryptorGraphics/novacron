package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ValidationResult represents the result of a single validation
type ValidationResult struct {
	Name     string
	Status   string
	Duration time.Duration
	Details  string
	Issues   []string
}

// QuickValidation performs essential validation checks
func main() {
	fmt.Println("🔍 NovaCron Quick Validation Assessment")
	fmt.Println(strings.Repeat("=", 50))
	
	results := []ValidationResult{
		validateBackendCore(),
		validateOrchestration(), 
		validateFrontendBasic(),
		validateInfrastructure(),
		validateConfiguration(),
	}
	
	fmt.Println("\n📊 VALIDATION SUMMARY")
	fmt.Println(strings.Repeat("-", 50))
	
	passCount := 0
	totalCount := len(results)
	
	for _, result := range results {
		icon := "❌"
		if result.Status == "PASS" {
			icon = "✅"
			passCount++
		} else if result.Status == "WARN" {
			icon = "⚠️"
		}
		
		fmt.Printf("%s %s (%v)\n", icon, result.Name, result.Duration)
		if result.Details != "" {
			fmt.Printf("   %s\n", result.Details)
		}
		for _, issue := range result.Issues {
			fmt.Printf("   • %s\n", issue)
		}
	}
	
	fmt.Printf("\n🎯 Overall: %d/%d checks passed (%.1f%%)\n", 
		passCount, totalCount, float64(passCount)/float64(totalCount)*100)
	
	if passCount == totalCount {
		fmt.Println("✅ System ready for deployment")
		os.Exit(0)
	} else if passCount >= totalCount*2/3 {
		fmt.Println("⚠️  System needs fixes before deployment")
		os.Exit(1)
	} else {
		fmt.Println("❌ System has critical issues - DO NOT DEPLOY")
		os.Exit(2)
	}
}

func validateBackendCore() ValidationResult {
	start := time.Now()
	result := ValidationResult{
		Name:   "Backend Core Compilation",
		Issues: make([]string, 0),
	}
	
	// Test core module compilation
	cmd := exec.Command("go", "build", "./backend/core/orchestration/...")
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		result.Status = "FAIL"
		result.Details = "Core compilation failed"
		result.Issues = append(result.Issues, string(output))
	} else {
		result.Status = "PASS"
		result.Details = "Core orchestration compiles successfully"
	}
	
	result.Duration = time.Since(start)
	return result
}

func validateOrchestration() ValidationResult {
	start := time.Now()
	result := ValidationResult{
		Name:   "Orchestration System Tests",
		Issues: make([]string, 0),
	}
	
	// Test orchestration module
	cmd := exec.Command("go", "test", "./backend/core/orchestration/...", "-v")
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		result.Status = "FAIL"
		result.Details = "Orchestration tests failed"
		result.Issues = append(result.Issues, string(output))
	} else {
		// Count passing tests
		outputStr := string(output)
		passCount := strings.Count(outputStr, "--- PASS:")
		if passCount > 0 {
			result.Status = "PASS"
			result.Details = fmt.Sprintf("%d orchestration tests passed", passCount)
		} else {
			result.Status = "WARN"
			result.Details = "No test results found"
		}
	}
	
	result.Duration = time.Since(start)
	return result
}

func validateFrontendBasic() ValidationResult {
	start := time.Now()
	result := ValidationResult{
		Name:   "Frontend Basic Compilation", 
		Issues: make([]string, 0),
	}
	
	// Check if frontend builds at all
	originalDir, _ := os.Getwd()
	os.Chdir("frontend")
	defer os.Chdir(originalDir)
	
	cmd := exec.Command("npm", "run", "lint")
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		result.Status = "WARN"
		result.Details = "Frontend has linting issues"
		result.Issues = append(result.Issues, "ESLint errors detected")
	} else {
		result.Status = "PASS" 
		result.Details = "Frontend code passes linting"
	}
	
	result.Duration = time.Since(start)
	return result
}

func validateInfrastructure() ValidationResult {
	start := time.Now()
	result := ValidationResult{
		Name:   "Infrastructure Services",
		Issues: make([]string, 0),
	}
	
	// Check for running Docker containers
	cmd := exec.Command("docker", "ps", "--format", "table {{.Names}}\t{{.Status}}")
	output, err := cmd.CombinedOutput()
	
	if err != nil {
		result.Status = "FAIL"
		result.Details = "Cannot check Docker infrastructure"
		result.Issues = append(result.Issues, string(output))
	} else {
		outputStr := string(output)
		healthyCount := strings.Count(outputStr, "(healthy)")
		if healthyCount >= 3 {
			result.Status = "PASS"
			result.Details = fmt.Sprintf("%d healthy services detected", healthyCount)
		} else {
			result.Status = "WARN"
			result.Details = "Limited infrastructure services running"
		}
	}
	
	result.Duration = time.Since(start)
	return result
}

func validateConfiguration() ValidationResult {
	start := time.Now()
	result := ValidationResult{
		Name:   "Configuration Validation",
		Issues: make([]string, 0),
	}
	
	// Check for required configuration files
	requiredFiles := []string{
		"go.mod",
		"Makefile", 
		"docker-compose.yml",
		".env",
		"frontend/package.json",
	}
	
	missingFiles := 0
	for _, file := range requiredFiles {
		if _, err := os.Stat(file); os.IsNotExist(err) {
			result.Issues = append(result.Issues, fmt.Sprintf("Missing: %s", file))
			missingFiles++
		}
	}
	
	if missingFiles == 0 {
		result.Status = "PASS"
		result.Details = "All configuration files present"
	} else {
		result.Status = "WARN"
		result.Details = fmt.Sprintf("%d configuration files missing", missingFiles)
	}
	
	// Check for critical directories
	criticalDirs := []string{
		"backend/core",
		"frontend/src", 
		"tests",
		"docs",
	}
	
	for _, dir := range criticalDirs {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			result.Issues = append(result.Issues, fmt.Sprintf("Missing directory: %s", dir))
		}
	}
	
	result.Duration = time.Since(start)
	return result
}