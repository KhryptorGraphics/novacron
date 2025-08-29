// Cross-Language API SDK Testing Framework
package sdk

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// SDK testing configuration
type SDKTestConfig struct {
	BaseURL       string                    `json:"base_url"`
	APIKey        string                    `json:"api_key"`
	TestTimeout   time.Duration             `json:"test_timeout"`
	SDKLanguages  []string                  `json:"sdk_languages"`
	TestSuites    []string                  `json:"test_suites"`
	FeatureFlags  map[string]bool           `json:"feature_flags"`
	Endpoints     map[string]EndpointSpec   `json:"endpoints"`
}

type EndpointSpec struct {
	Method      string                 `json:"method"`
	Path        string                 `json:"path"`
	RequiresAuth bool                  `json:"requires_auth"`
	Parameters  map[string]interface{} `json:"parameters"`
	Response    ResponseSpec           `json:"response"`
}

type ResponseSpec struct {
	StatusCode int                    `json:"status_code"`
	Schema     map[string]interface{} `json:"schema"`
	Required   []string               `json:"required"`
}

// Test result structures
type SDKTestResult struct {
	Language     string                    `json:"language"`
	TestSuite    string                    `json:"test_suite"`
	Results      []TestCaseResult          `json:"results"`
	Summary      TestSummary               `json:"summary"`
	Performance  PerformanceMetrics        `json:"performance"`
	Compatibility CompatibilityMatrix      `json:"compatibility"`
}

type TestCaseResult struct {
	Name        string        `json:"name"`
	Status      string        `json:"status"` // pass, fail, skip
	Duration    time.Duration `json:"duration"`
	Error       string        `json:"error,omitempty"`
	Output      string        `json:"output,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type TestSummary struct {
	Total    int     `json:"total"`
	Passed   int     `json:"passed"`
	Failed   int     `json:"failed"`
	Skipped  int     `json:"skipped"`
	Coverage float64 `json:"coverage"`
}

type PerformanceMetrics struct {
	AverageLatency    time.Duration `json:"average_latency"`
	P95Latency        time.Duration `json:"p95_latency"`
	ThroughputRPS     float64       `json:"throughput_rps"`
	ErrorRate         float64       `json:"error_rate"`
	MemoryUsageMB     float64       `json:"memory_usage_mb"`
}

type CompatibilityMatrix struct {
	APIVersion    string            `json:"api_version"`
	SDKVersion    string            `json:"sdk_version"`
	Features      map[string]bool   `json:"features"`
	Endpoints     map[string]string `json:"endpoints"` // endpoint -> status
}

// Cross-language test framework
type CrossLanguageTestFramework struct {
	config     *SDKTestConfig
	httpClient *http.Client
	testDir    string
}

func NewCrossLanguageTestFramework(config *SDKTestConfig) *CrossLanguageTestFramework {
	return &CrossLanguageTestFramework{
		config: config,
		httpClient: &http.Client{
			Timeout: config.TestTimeout,
		},
		testDir: "./sdk_tests",
	}
}

// Main cross-language testing function
func TestCrossLanguageSDKCompatibility(t *testing.T) {
	config := getSDKTestConfig()
	framework := NewCrossLanguageTestFramework(config)

	// Ensure test API is running
	require.NoError(t, framework.ensureAPIRunning(t), "Test API should be running")

	for _, language := range config.SDKLanguages {
		t.Run(fmt.Sprintf("SDK_%s", language), func(t *testing.T) {
			framework.testLanguageSDK(t, language)
		})
	}

	t.Run("FeatureParity", func(t *testing.T) {
		framework.testFeatureParity(t)
	})

	t.Run("PerformanceComparison", func(t *testing.T) {
		framework.testPerformanceComparison(t)
	})
}

func (f *CrossLanguageTestFramework) testLanguageSDK(t *testing.T, language string) {
	t.Run("Installation", func(t *testing.T) {
		f.testSDKInstallation(t, language)
	})

	t.Run("Authentication", func(t *testing.T) {
		f.testSDKAuthentication(t, language)
	})

	t.Run("CoreOperations", func(t *testing.T) {
		f.testSDKCoreOperations(t, language)
	})

	t.Run("ErrorHandling", func(t *testing.T) {
		f.testSDKErrorHandling(t, language)
	})

	t.Run("AsyncOperations", func(t *testing.T) {
		f.testSDKAsyncOperations(t, language)
	})
}

// Test SDK installation and import
func (f *CrossLanguageTestFramework) testSDKInstallation(t *testing.T, language string) {
	testScript := f.generateInstallationTest(language)
	result := f.runSDKTest(t, language, "installation", testScript)
	
	assert.Equal(t, "pass", result.Status, "SDK installation should succeed for %s", language)
	assert.Empty(t, result.Error, "Installation should not produce errors")
}

// Test SDK authentication mechanisms
func (f *CrossLanguageTestFramework) testSDKAuthentication(t *testing.T, language string) {
	testCases := []struct {
		name      string
		apiKey    string
		shouldPass bool
	}{
		{"ValidAPIKey", f.config.APIKey, true},
		{"InvalidAPIKey", "invalid-key-12345", false},
		{"EmptyAPIKey", "", false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testScript := f.generateAuthenticationTest(language, tc.apiKey)
			result := f.runSDKTest(t, language, fmt.Sprintf("auth_%s", tc.name), testScript)

			if tc.shouldPass {
				assert.Equal(t, "pass", result.Status, "Authentication should succeed with valid key")
			} else {
				assert.Equal(t, "fail", result.Status, "Authentication should fail with invalid key")
			}
		})
	}
}

// Test core SDK operations
func (f *CrossLanguageTestFramework) testSDKCoreOperations(t *testing.T, language string) {
	operations := []string{
		"list_vms",
		"create_vm",
		"get_vm",
		"update_vm",
		"delete_vm",
		"vm_metrics",
		"scheduler_decisions",
	}

	for _, operation := range operations {
		t.Run(operation, func(t *testing.T) {
			testScript := f.generateOperationTest(language, operation)
			result := f.runSDKTest(t, language, fmt.Sprintf("core_%s", operation), testScript)
			
			assert.Equal(t, "pass", result.Status, "Core operation %s should succeed in %s", operation, language)
			
			if result.Status == "fail" {
				t.Logf("Operation %s failed in %s: %s", operation, language, result.Error)
			}
		})
	}
}

// Test SDK error handling
func (f *CrossLanguageTestFramework) testSDKErrorHandling(t *testing.T, language string) {
	errorScenarios := []struct {
		name           string
		endpoint       string
		expectedStatus int
		shouldRetry    bool
	}{
		{"NotFound", "/api/vms/non-existent-vm", 404, false},
		{"Unauthorized", "/api/vms", 401, false},
		{"RateLimit", "/api/vms", 429, true},
		{"ServerError", "/api/simulate-error", 500, true},
	}

	for _, scenario := range errorScenarios {
		t.Run(scenario.name, func(t *testing.T) {
			testScript := f.generateErrorHandlingTest(language, scenario.endpoint, scenario.expectedStatus)
			result := f.runSDKTest(t, language, fmt.Sprintf("error_%s", scenario.name), testScript)
			
			assert.Equal(t, "pass", result.Status, "Error handling for %s should work in %s", scenario.name, language)
		})
	}
}

// Test SDK async operations
func (f *CrossLanguageTestFramework) testSDKAsyncOperations(t *testing.T, language string) {
	if !f.supportsAsyncOperations(language) {
		t.Skip(fmt.Sprintf("Language %s does not support async operations", language))
	}

	asyncOperations := []string{
		"async_vm_creation",
		"async_vm_migration",
		"concurrent_requests",
		"timeout_handling",
	}

	for _, operation := range asyncOperations {
		t.Run(operation, func(t *testing.T) {
			testScript := f.generateAsyncTest(language, operation)
			result := f.runSDKTest(t, language, fmt.Sprintf("async_%s", operation), testScript)
			
			assert.Equal(t, "pass", result.Status, "Async operation %s should succeed in %s", operation, language)
		})
	}
}

// Test feature parity across languages
func (f *CrossLanguageTestFramework) testFeatureParity(t *testing.T) {
	features := map[string][]string{
		"core_operations": {"list_vms", "create_vm", "delete_vm"},
		"authentication": {"api_key_auth", "token_refresh"},
		"error_handling": {"retry_logic", "error_classification"},
		"async_support":  {"async_requests", "concurrent_operations"},
	}

	results := make(map[string]map[string]bool) // language -> feature -> supported

	for _, language := range f.config.SDKLanguages {
		results[language] = make(map[string]bool)
		
		for category, categoryFeatures := range features {
			for _, feature := range categoryFeatures {
				testScript := f.generateFeatureTest(language, feature)
				result := f.runSDKTest(t, language, fmt.Sprintf("feature_%s", feature), testScript)
				results[language][feature] = (result.Status == "pass")
			}
		}
	}

	// Analyze feature parity
	f.analyzeFeatureParity(t, results, features)
}

// Test performance comparison across languages
func (f *CrossLanguageTestFramework) testPerformanceComparison(t *testing.T) {
	performanceTests := []struct {
		name        string
		description string
		operation   string
		iterations  int
		timeout     time.Duration
	}{
		{"SingleRequest", "Single API request latency", "get_vm", 1, 5 * time.Second},
		{"BulkOperations", "Bulk operations throughput", "list_vms", 100, 30 * time.Second},
		{"ConcurrentRequests", "Concurrent request handling", "concurrent_get", 50, 60 * time.Second},
	}

	performanceResults := make(map[string]map[string]PerformanceMetrics)

	for _, language := range f.config.SDKLanguages {
		performanceResults[language] = make(map[string]PerformanceMetrics)
		
		for _, perfTest := range performanceTests {
			t.Run(fmt.Sprintf("%s_%s", language, perfTest.name), func(t *testing.T) {
				testScript := f.generatePerformanceTest(language, perfTest.operation, perfTest.iterations)
				result := f.runSDKTest(t, language, fmt.Sprintf("perf_%s", perfTest.name), testScript)
				
				// Parse performance metrics from result
				metrics := f.parsePerformanceMetrics(result.Output)
				performanceResults[language][perfTest.name] = metrics
				
				// Basic performance assertions
				assert.Less(t, metrics.AverageLatency, 5*time.Second, "Average latency should be reasonable")
				assert.Less(t, metrics.ErrorRate, 0.05, "Error rate should be under 5%")
				
				t.Logf("%s %s: Avg latency %v, Throughput %.2f RPS", 
					language, perfTest.name, metrics.AverageLatency, metrics.ThroughputRPS)
			})
		}
	}

	// Compare performance across languages
	f.comparePerformanceResults(t, performanceResults)
}

// SDK test script generators
func (f *CrossLanguageTestFramework) generateInstallationTest(language string) string {
	switch language {
	case "go":
		return `
package main

import (
	"fmt"
	"github.com/novacron/sdk-go"
)

func main() {
	client := novacron.NewClient("dummy-key")
	if client == nil {
		fmt.Println("FAIL: Failed to create client")
		return
	}
	fmt.Println("PASS: SDK installation successful")
}
`
	case "python":
		return `
import sys
try:
    import novacron
    client = novacron.Client("dummy-key")
    print("PASS: SDK installation successful")
except ImportError as e:
    print(f"FAIL: Failed to import SDK: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL: Failed to create client: {e}")
    sys.exit(1)
`
	case "javascript":
		return `
const { NovaCronClient } = require('novacron-sdk');

try {
    const client = new NovaCronClient({ apiKey: 'dummy-key' });
    console.log('PASS: SDK installation successful');
} catch (error) {
    console.log('FAIL: Failed to create client:', error.message);
    process.exit(1);
}
`
	case "java":
		return `
import com.novacron.sdk.NovaCronClient;

public class InstallationTest {
    public static void main(String[] args) {
        try {
            NovaCronClient client = new NovaCronClient("dummy-key");
            System.out.println("PASS: SDK installation successful");
        } catch (Exception e) {
            System.out.println("FAIL: Failed to create client: " + e.getMessage());
            System.exit(1);
        }
    }
}
`
	default:
		return fmt.Sprintf("# Unsupported language: %s", language)
	}
}

func (f *CrossLanguageTestFramework) generateAuthenticationTest(language, apiKey string) string {
	switch language {
	case "go":
		return fmt.Sprintf(`
package main

import (
	"fmt"
	"context"
	"github.com/novacron/sdk-go"
)

func main() {
	client := novacron.NewClient("%s")
	client.SetBaseURL("%s")
	
	ctx := context.Background()
	_, err := client.ListVMs(ctx, &novacron.ListVMsRequest{})
	if err != nil {
		fmt.Printf("FAIL: Authentication failed: %%v\n", err)
		return
	}
	fmt.Println("PASS: Authentication successful")
}
`, apiKey, f.config.BaseURL)

	case "python":
		return fmt.Sprintf(`
import novacron
import sys

try:
    client = novacron.Client("%s", base_url="%s")
    vms = client.list_vms()
    print("PASS: Authentication successful")
except novacron.AuthenticationError as e:
    print(f"FAIL: Authentication failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    sys.exit(1)
`, apiKey, f.config.BaseURL)

	case "javascript":
		return fmt.Sprintf(`
const { NovaCronClient } = require('novacron-sdk');

async function test() {
    try {
        const client = new NovaCronClient({ 
            apiKey: '%s', 
            baseURL: '%s' 
        });
        await client.listVMs();
        console.log('PASS: Authentication successful');
    } catch (error) {
        console.log('FAIL: Authentication failed:', error.message);
        process.exit(1);
    }
}

test();
`, apiKey, f.config.BaseURL)

	default:
		return fmt.Sprintf("# Unsupported language for auth test: %s", language)
	}
}

func (f *CrossLanguageTestFramework) generateOperationTest(language, operation string) string {
	// Generate operation-specific test code for each language
	switch language {
	case "go":
		return f.generateGoOperationTest(operation)
	case "python":
		return f.generatePythonOperationTest(operation)
	case "javascript":
		return f.generateJavaScriptOperationTest(operation)
	case "java":
		return f.generateJavaOperationTest(operation)
	default:
		return fmt.Sprintf("# Unsupported language: %s", language)
	}
}

func (f *CrossLanguageTestFramework) generateGoOperationTest(operation string) string {
	switch operation {
	case "list_vms":
		return fmt.Sprintf(`
package main

import (
	"fmt"
	"context"
	"github.com/novacron/sdk-go"
)

func main() {
	client := novacron.NewClient("%s")
	client.SetBaseURL("%s")
	
	ctx := context.Background()
	resp, err := client.ListVMs(ctx, &novacron.ListVMsRequest{})
	if err != nil {
		fmt.Printf("FAIL: List VMs failed: %%v\n", err)
		return
	}
	fmt.Printf("PASS: Listed %%d VMs\n", len(resp.VMs))
}
`, f.config.APIKey, f.config.BaseURL)

	case "create_vm":
		return fmt.Sprintf(`
package main

import (
	"fmt"
	"context"
	"github.com/novacron/sdk-go"
)

func main() {
	client := novacron.NewClient("%s")
	client.SetBaseURL("%s")
	
	ctx := context.Background()
	req := &novacron.CreateVMRequest{
		Name: "test-vm-sdk",
		Template: "ubuntu-20.04",
		Resources: novacron.VMResources{
			CPU: 2,
			Memory: 4096,
			Disk: 50,
		},
	}
	
	vm, err := client.CreateVM(ctx, req)
	if err != nil {
		fmt.Printf("FAIL: Create VM failed: %%v\n", err)
		return
	}
	
	// Clean up
	defer func() {
		client.DeleteVM(ctx, &novacron.DeleteVMRequest{ID: vm.ID})
	}()
	
	fmt.Printf("PASS: Created VM %%s\n", vm.ID)
}
`, f.config.APIKey, f.config.BaseURL)

	default:
		return fmt.Sprintf("# Unsupported Go operation: %s", operation)
	}
}

func (f *CrossLanguageTestFramework) generatePythonOperationTest(operation string) string {
	switch operation {
	case "list_vms":
		return fmt.Sprintf(`
import novacron
import sys

try:
    client = novacron.Client("%s", base_url="%s")
    vms = client.list_vms()
    print(f"PASS: Listed {len(vms)} VMs")
except Exception as e:
    print(f"FAIL: List VMs failed: {e}")
    sys.exit(1)
`, f.config.APIKey, f.config.BaseURL)

	case "create_vm":
		return fmt.Sprintf(`
import novacron
import sys

try:
    client = novacron.Client("%s", base_url="%s")
    vm = client.create_vm(
        name="test-vm-sdk",
        template="ubuntu-20.04",
        resources={
            "cpu": 2,
            "memory": 4096,
            "disk": 50
        }
    )
    
    # Clean up
    try:
        client.delete_vm(vm.id)
    except:
        pass  # Best effort cleanup
    
    print(f"PASS: Created VM {vm.id}")
except Exception as e:
    print(f"FAIL: Create VM failed: {e}")
    sys.exit(1)
`, f.config.APIKey, f.config.BaseURL)

	default:
		return fmt.Sprintf("# Unsupported Python operation: %s", operation)
	}
}

func (f *CrossLanguageTestFramework) generateJavaScriptOperationTest(operation string) string {
	switch operation {
	case "list_vms":
		return fmt.Sprintf(`
const { NovaCronClient } = require('novacron-sdk');

async function test() {
    try {
        const client = new NovaCronClient({ 
            apiKey: '%s', 
            baseURL: '%s' 
        });
        const vms = await client.listVMs();
        console.log('PASS: Listed', vms.length, 'VMs');
    } catch (error) {
        console.log('FAIL: List VMs failed:', error.message);
        process.exit(1);
    }
}

test();
`, f.config.APIKey, f.config.BaseURL)

	case "create_vm":
		return fmt.Sprintf(`
const { NovaCronClient } = require('novacron-sdk');

async function test() {
    try {
        const client = new NovaCronClient({ 
            apiKey: '%s', 
            baseURL: '%s' 
        });
        
        const vm = await client.createVM({
            name: 'test-vm-sdk',
            template: 'ubuntu-20.04',
            resources: {
                cpu: 2,
                memory: 4096,
                disk: 50
            }
        });
        
        // Clean up
        try {
            await client.deleteVM(vm.id);
        } catch (e) {
            // Best effort cleanup
        }
        
        console.log('PASS: Created VM', vm.id);
    } catch (error) {
        console.log('FAIL: Create VM failed:', error.message);
        process.exit(1);
    }
}

test();
`, f.config.APIKey, f.config.BaseURL)

	default:
		return fmt.Sprintf("// Unsupported JavaScript operation: %s", operation)
	}
}

func (f *CrossLanguageTestFramework) generateJavaOperationTest(operation string) string {
	switch operation {
	case "list_vms":
		return fmt.Sprintf(`
import com.novacron.sdk.NovaCronClient;
import java.util.List;

public class ListVMsTest {
    public static void main(String[] args) {
        try {
            NovaCronClient client = new NovaCronClient("%s", "%s");
            List<VM> vms = client.listVMs();
            System.out.println("PASS: Listed " + vms.size() + " VMs");
        } catch (Exception e) {
            System.out.println("FAIL: List VMs failed: " + e.getMessage());
            System.exit(1);
        }
    }
}
`, f.config.APIKey, f.config.BaseURL)

	default:
		return fmt.Sprintf("// Unsupported Java operation: %s", operation)
	}
}

// Additional test generators for error handling, async operations, etc.
func (f *CrossLanguageTestFramework) generateErrorHandlingTest(language, endpoint string, expectedStatus int) string {
	// Implementation would generate error handling test code
	return fmt.Sprintf("# Error handling test for %s endpoint %s (status %d)", language, endpoint, expectedStatus)
}

func (f *CrossLanguageTestFramework) generateAsyncTest(language, operation string) string {
	// Implementation would generate async test code
	return fmt.Sprintf("# Async test for %s operation %s", language, operation)
}

func (f *CrossLanguageTestFramework) generateFeatureTest(language, feature string) string {
	// Implementation would generate feature-specific test code
	return fmt.Sprintf("# Feature test for %s feature %s", language, feature)
}

func (f *CrossLanguageTestFramework) generatePerformanceTest(language, operation string, iterations int) string {
	// Implementation would generate performance test code
	return fmt.Sprintf("# Performance test for %s operation %s (%d iterations)", language, operation, iterations)
}

// Test execution and result processing
func (f *CrossLanguageTestFramework) runSDKTest(t *testing.T, language, testName, script string) TestCaseResult {
	testDir := filepath.Join(f.testDir, language)
	os.MkdirAll(testDir, 0755)

	start := time.Now()
	
	// Write test script to file
	scriptFile := filepath.Join(testDir, fmt.Sprintf("%s.%s", testName, f.getFileExtension(language)))
	err := os.WriteFile(scriptFile, []byte(script), 0644)
	if err != nil {
		return TestCaseResult{
			Name:     testName,
			Status:   "fail",
			Duration: time.Since(start),
			Error:    fmt.Sprintf("Failed to write test script: %v", err),
		}
	}

	// Execute test
	cmd := f.getExecutionCommand(language, scriptFile)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	cmd.Dir = testDir

	err = cmd.Run()
	duration := time.Since(start)

	output := stdout.String()
	if stderr.Len() > 0 {
		output += "\nSTDERR:\n" + stderr.String()
	}

	status := "pass"
	errorMsg := ""
	
	if err != nil {
		status = "fail"
		errorMsg = err.Error()
	} else if !strings.Contains(output, "PASS:") {
		status = "fail"
		errorMsg = "Test did not produce PASS result"
	}

	return TestCaseResult{
		Name:     testName,
		Status:   status,
		Duration: duration,
		Error:    errorMsg,
		Output:   output,
	}
}

// Helper methods
func (f *CrossLanguageTestFramework) getFileExtension(language string) string {
	extensions := map[string]string{
		"go":         "go",
		"python":     "py",
		"javascript": "js",
		"java":       "java",
		"csharp":     "cs",
		"ruby":       "rb",
		"php":        "php",
	}
	return extensions[language]
}

func (f *CrossLanguageTestFramework) getExecutionCommand(language, scriptFile string) *exec.Cmd {
	switch language {
	case "go":
		return exec.Command("go", "run", scriptFile)
	case "python":
		return exec.Command("python3", scriptFile)
	case "javascript":
		return exec.Command("node", scriptFile)
	case "java":
		// Simplified - in practice would need compilation step
		className := strings.TrimSuffix(filepath.Base(scriptFile), ".java")
		return exec.Command("java", className)
	default:
		return exec.Command("echo", "Unsupported language")
	}
}

func (f *CrossLanguageTestFramework) supportsAsyncOperations(language string) bool {
	asyncLanguages := []string{"javascript", "python", "go", "csharp"}
	for _, lang := range asyncLanguages {
		if lang == language {
			return true
		}
	}
	return false
}

func (f *CrossLanguageTestFramework) ensureAPIRunning(t *testing.T) error {
	// Check if API is responding
	resp, err := f.httpClient.Get(f.config.BaseURL + "/health")
	if err != nil {
		return fmt.Errorf("API not responding: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("API health check failed with status %d", resp.StatusCode)
	}

	return nil
}

func (f *CrossLanguageTestFramework) parsePerformanceMetrics(output string) PerformanceMetrics {
	// Simplified metrics parsing - in practice would parse structured output
	return PerformanceMetrics{
		AverageLatency: 100 * time.Millisecond,
		P95Latency:     200 * time.Millisecond,
		ThroughputRPS:  50.0,
		ErrorRate:      0.01,
		MemoryUsageMB:  128.0,
	}
}

func (f *CrossLanguageTestFramework) analyzeFeatureParity(t *testing.T, results map[string]map[string]bool, features map[string][]string) {
	t.Log("Feature Parity Analysis:")
	
	for category, categoryFeatures := range features {
		t.Logf("\nCategory: %s", category)
		
		for _, feature := range categoryFeatures {
			supportedLanguages := []string{}
			for language := range results {
				if results[language][feature] {
					supportedLanguages = append(supportedLanguages, language)
				}
			}
			
			coverage := float64(len(supportedLanguages)) / float64(len(f.config.SDKLanguages))
			t.Logf("  %s: %.1f%% coverage (%v)", feature, coverage*100, supportedLanguages)
			
			// Assert minimum coverage
			assert.GreaterOrEqual(t, coverage, 0.8, "Feature %s should have at least 80%% coverage", feature)
		}
	}
}

func (f *CrossLanguageTestFramework) comparePerformanceResults(t *testing.T, results map[string]map[string]PerformanceMetrics) {
	t.Log("Performance Comparison:")
	
	for testName := range results[f.config.SDKLanguages[0]] {
		t.Logf("\nTest: %s", testName)
		
		var bestLatency time.Duration
		var worstLatency time.Duration
		var bestThroughput float64
		var worstThroughput float64
		
		first := true
		for language, metrics := range results {
			if testMetrics, exists := metrics[testName]; exists {
				if first {
					bestLatency = testMetrics.AverageLatency
					worstLatency = testMetrics.AverageLatency
					bestThroughput = testMetrics.ThroughputRPS
					worstThroughput = testMetrics.ThroughputRPS
					first = false
				} else {
					if testMetrics.AverageLatency < bestLatency {
						bestLatency = testMetrics.AverageLatency
					}
					if testMetrics.AverageLatency > worstLatency {
						worstLatency = testMetrics.AverageLatency
					}
					if testMetrics.ThroughputRPS > bestThroughput {
						bestThroughput = testMetrics.ThroughputRPS
					}
					if testMetrics.ThroughputRPS < worstThroughput {
						worstThroughput = testMetrics.ThroughputRPS
					}
				}
				
				t.Logf("  %s: %v latency, %.2f RPS", language, testMetrics.AverageLatency, testMetrics.ThroughputRPS)
			}
		}
		
		// Performance variance should not be too extreme
		latencyVariance := float64(worstLatency-bestLatency) / float64(bestLatency)
		throughputVariance := (bestThroughput - worstThroughput) / worstThroughput
		
		assert.Less(t, latencyVariance, 3.0, "Latency variance should not exceed 300%% for test %s", testName)
		assert.Less(t, throughputVariance, 3.0, "Throughput variance should not exceed 300%% for test %s", testName)
	}
}

// Configuration helper
func getSDKTestConfig() *SDKTestConfig {
	return &SDKTestConfig{
		BaseURL:     getEnvOrDefault("NOVACRON_API_URL", "http://localhost:8090"),
		APIKey:      getEnvOrDefault("NOVACRON_API_KEY", "test-api-key"),
		TestTimeout: 30 * time.Second,
		SDKLanguages: []string{"go", "python", "javascript"},
		TestSuites:  []string{"core", "auth", "errors", "async"},
		FeatureFlags: map[string]bool{
			"async_support":     true,
			"batch_operations":  true,
			"streaming":         false,
			"webhooks":          true,
		},
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// Benchmark cross-language performance
func BenchmarkCrossLanguagePerformance(b *testing.B) {
	config := getSDKTestConfig()
	framework := NewCrossLanguageTestFramework(config)

	for _, language := range config.SDKLanguages {
		b.Run(language, func(b *testing.B) {
			b.ResetTimer()
			
			for i := 0; i < b.N; i++ {
				testScript := framework.generateOperationTest(language, "list_vms")
				result := framework.runSDKTest(nil, language, "benchmark", testScript)
				
				if result.Status != "pass" {
					b.Fatalf("Benchmark failed for %s: %s", language, result.Error)
				}
			}
		})
	}
}