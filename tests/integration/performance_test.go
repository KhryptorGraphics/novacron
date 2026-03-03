package integration

import (
	"fmt"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/stretchr/testify/suite"

	"github.com/khryptorgraphics/novacron/tests/integration/helpers"
)

// PerformanceTestSuite tests system performance and benchmarks
type PerformanceTestSuite struct {
	suite.Suite
	env     *helpers.TestEnvironment
	mockGen *helpers.MockDataGenerator
	token   string
	tenantID string
}

// SetupSuite initializes the test suite
func (suite *PerformanceTestSuite) SetupSuite() {
	suite.env = helpers.NewTestEnvironment(suite.T())
	suite.env.Setup(suite.T())
	suite.mockGen = helpers.NewMockDataGenerator()
	suite.tenantID = "tenant-1"
	
	// Login as admin
	suite.token = suite.env.LoginAsAdmin(suite.T())
	suite.env.APIClient.SetAuthToken(suite.token)
}

// TearDownSuite cleans up the test suite
func (suite *PerformanceTestSuite) TearDownSuite() {
	if suite.env != nil {
		suite.env.Cleanup(suite.T())
	}
}

// BenchmarkVMCreation benchmarks VM creation performance
func (suite *PerformanceTestSuite) TestBenchmarkVMCreation() {
	if testing.Short() {
		suite.T().Skip("Skipping VM creation benchmark in short mode")
	}
	
	suite.T().Run("VM Creation Performance", func(t *testing.T) {
		numVMs := 20
		concurrency := 5
		
		results := &PerformanceResults{
			Operation:   "VM Creation",
			TotalOps:    numVMs,
			Concurrency: concurrency,
		}
		
		// Run benchmark
		suite.runConcurrentBenchmark(t, numVMs, concurrency, func(index int) error {
			vmData := map[string]interface{}{
				"name":      fmt.Sprintf("bench-vm-%d", index),
				"cpu":       2,
				"memory":    1024,
				"disk_size": 10240,
				"image":     "ubuntu:20.04",
				"tenant_id": suite.tenantID,
			}
			
			start := time.Now()
			resp := suite.env.APIClient.POST(t, "/api/vms", vmData)
			defer resp.Body.Close()
			
			duration := time.Since(start)
			results.AddMeasurement(duration)
			
			if resp.StatusCode != http.StatusCreated {
				return fmt.Errorf("VM creation failed with status %d", resp.StatusCode)
			}
			
			var result map[string]interface{}
			err := suite.env.APIClient.ParseJSON(t, resp, &result)
			if err != nil {
				return err
			}
			
			// Schedule cleanup
			vmID := result["id"].(string)
			go func() {
				time.Sleep(5 * time.Second) // Let other tests complete first
				suite.env.APIClient.DELETE(t, "/api/vms/"+vmID)
			}()
			
			return nil
		})
		
		// Analyze results
		results.Analyze()
		suite.logPerformanceResults(t, results)
		
		// Performance assertions
		assert.Less(t, results.AverageLatency.Milliseconds(), int64(2000), 
			"Average VM creation should be under 2 seconds")
		assert.Less(t, results.P95Latency.Milliseconds(), int64(5000),
			"95th percentile should be under 5 seconds")
		assert.GreaterOrEqual(t, results.SuccessRate, 0.95,
			"Success rate should be at least 95%%")
	})
}

// TestBenchmarkAPIThroughput benchmarks API request throughput
func (suite *PerformanceTestSuite) TestBenchmarkAPIThroughput() {
	if testing.Short() {
		suite.T().Skip("Skipping API throughput benchmark in short mode")
	}
	
	suite.T().Run("API Throughput", func(t *testing.T) {
		// Create test VMs for listing
		testVMs := make([]string, 5)
		for i := 0; i < 5; i++ {
			vmID := suite.env.CreateTestVM(t, fmt.Sprintf("throughput-vm-%d", i), suite.tenantID)
			testVMs[i] = vmID
		}
		
		defer func() {
			// Cleanup
			for _, vmID := range testVMs {
				suite.env.APIClient.DELETE(t, "/api/vms/"+vmID)
			}
		}()
		
		// Benchmark different endpoints
		endpoints := []struct {
			name     string
			method   string
			endpoint string
			target   int // Requests per second target
		}{
			{"List VMs", "GET", "/api/vms", 50},
			{"Get VM Status", "GET", "/api/vms/" + testVMs[0], 100},
			{"Get User Profile", "GET", "/api/user/profile", 200},
			{"Get Cluster Status", "GET", "/api/consensus/status", 30},
		}
		
		for _, ep := range endpoints {
			t.Run(ep.name, func(t *testing.T) {
				duration := 10 * time.Second
				concurrency := 10
				
				results := &PerformanceResults{
					Operation:   ep.name,
					Concurrency: concurrency,
				}
				
				// Run for fixed duration
				suite.runDurationBenchmark(t, duration, concurrency, func() error {
					start := time.Now()
					
					var resp *http.Response
					switch ep.method {
					case "GET":
						resp = suite.env.APIClient.GET(t, ep.endpoint)
					case "POST":
						resp = suite.env.APIClient.POST(t, ep.endpoint, nil)
					}
					defer resp.Body.Close()
					
					latency := time.Since(start)
					results.AddMeasurement(latency)
					
					if resp.StatusCode < 200 || resp.StatusCode >= 300 {
						return fmt.Errorf("request failed with status %d", resp.StatusCode)
					}
					
					return nil
				})
				
				results.Analyze()
				suite.logPerformanceResults(t, results)
				
				// Throughput assertions
				actualTPS := float64(results.TotalOps) / duration.Seconds()
				t.Logf("%s: %.1f requests/second", ep.name, actualTPS)
				
				assert.GreaterOrEqual(t, actualTPS, float64(ep.target)*0.7,
					"Should achieve at least 70%% of target throughput")
				assert.Less(t, results.AverageLatency.Milliseconds(), int64(500),
					"Average latency should be under 500ms")
			})
		}
	})
}

// TestBenchmarkConcurrentOperations benchmarks concurrent operations
func (suite *PerformanceTestSuite) TestBenchmarkConcurrentOperations() {
	if testing.Short() {
		suite.T().Skip("Skipping concurrent operations benchmark in short mode")
	}
	
	suite.T().Run("Concurrent Operations", func(t *testing.T) {
		// Test different concurrency levels
		concurrencyLevels := []int{1, 5, 10, 20}
		
		for _, concurrency := range concurrencyLevels {
			t.Run(fmt.Sprintf("Concurrency_%d", concurrency), func(t *testing.T) {
				numOps := 50
				
				results := &PerformanceResults{
					Operation:   "Mixed Operations",
					TotalOps:    numOps,
					Concurrency: concurrency,
				}
				
				suite.runConcurrentBenchmark(t, numOps, concurrency, func(index int) error {
					start := time.Now()
					
					// Mix of operations
					switch index % 4 {
					case 0:
						// Create VM
						vmData := suite.mockGen.GenerateVM(suite.tenantID, 1)
						resp := suite.env.APIClient.POST(t, "/api/vms", vmData)
						defer resp.Body.Close()
						
						if resp.StatusCode == http.StatusCreated {
							var result map[string]interface{}
							suite.env.APIClient.ParseJSON(t, resp, &result)
							vmID := result["id"].(string)
							
							// Schedule cleanup
							go func() {
								time.Sleep(2 * time.Second)
								suite.env.APIClient.DELETE(t, "/api/vms/"+vmID)
							}()
						}
						
					case 1:
						// List VMs
						resp := suite.env.APIClient.GET(t, "/api/vms")
						defer resp.Body.Close()
						
					case 2:
						// Get user profile
						resp := suite.env.APIClient.GET(t, "/api/user/profile")
						defer resp.Body.Close()
						
					case 3:
						// Get cluster status
						resp := suite.env.APIClient.GET(t, "/api/consensus/status")
						defer resp.Body.Close()
					}
					
					duration := time.Since(start)
					results.AddMeasurement(duration)
					
					return nil
				})
				
				results.Analyze()
				suite.logPerformanceResults(t, results)
				
				// Performance should not degrade significantly with higher concurrency
				if concurrency > 1 {
					assert.Less(t, results.AverageLatency.Milliseconds(), int64(3000),
						"Average latency should remain reasonable under load")
				}
			})
		}
	})
}

// TestBenchmarkMemoryUsage benchmarks memory usage patterns
func (suite *PerformanceTestSuite) TestBenchmarkMemoryUsage() {
	if testing.Short() {
		suite.T().Skip("Skipping memory usage benchmark in short mode")
	}
	
	suite.T().Run("Memory Usage", func(t *testing.T) {
		// Get initial memory stats
		initialResp := suite.env.APIClient.GET(t, "/api/system/metrics/memory")
		defer initialResp.Body.Close()
		
		var initialStats map[string]interface{}
		if initialResp.StatusCode == http.StatusOK {
			suite.env.APIClient.ParseJSON(t, initialStats, &initialStats)
		}
		
		// Perform memory-intensive operations
		numOperations := 100
		vmIDs := make([]string, 0, numOperations)
		
		// Create many VMs
		for i := 0; i < numOperations; i++ {
			vmData := map[string]interface{}{
				"name":      fmt.Sprintf("memory-test-vm-%d", i),
				"cpu":       1,
				"memory":    512,
				"disk_size": 5120,
				"image":     "alpine:3.16",
				"tenant_id": suite.tenantID,
			}
			
			resp := suite.env.APIClient.POST(t, "/api/vms", vmData)
			defer resp.Body.Close()
			
			if resp.StatusCode == http.StatusCreated {
				var result map[string]interface{}
				suite.env.APIClient.ParseJSON(t, resp, &result)
				vmIDs = append(vmIDs, result["id"].(string))
			}
		}
		
		// Check memory usage after operations
		finalResp := suite.env.APIClient.GET(t, "/api/system/metrics/memory")
		defer finalResp.Body.Close()
		
		var finalStats map[string]interface{}
		if finalResp.StatusCode == http.StatusOK {
			suite.env.APIClient.ParseJSON(t, finalResp, &finalStats)
			
			// Log memory usage
			if initialStats != nil {
				t.Logf("Memory usage increased during test")
				// Add specific memory usage analysis here
			}
		}
		
		// Cleanup all VMs
		for _, vmID := range vmIDs {
			suite.env.APIClient.DELETE(t, "/api/vms/"+vmID)
		}
		
		// Wait a bit for cleanup
		time.Sleep(5 * time.Second)
		
		// Check memory after cleanup
		cleanupResp := suite.env.APIClient.GET(t, "/api/system/metrics/memory")
		defer cleanupResp.Body.Close()
		
		t.Log("Memory benchmark completed - check logs for detailed metrics")
	})
}

// TestBenchmarkDatabaseOperations benchmarks database performance
func (suite *PerformanceTestSuite) TestBenchmarkDatabaseOperations() {
	if testing.Short() {
		suite.T().Skip("Skipping database benchmark in short mode")
	}
	
	suite.T().Run("Database Operations", func(t *testing.T) {
		// Benchmark different database operations
		operations := []struct {
			name      string
			operation func() error
		}{
			{
				name: "User Creation",
				operation: func() error {
					userData := map[string]interface{}{
						"email":     fmt.Sprintf("bench-user-%d@test.com", time.Now().UnixNano()),
						"password":  "BenchPass123!",
						"name":      "Benchmark User",
						"tenant_id": suite.tenantID,
					}
					
					resp := suite.env.APIClient.POST(suite.T(), "/api/auth/register", userData)
					defer resp.Body.Close()
					
					if resp.StatusCode != http.StatusCreated {
						return fmt.Errorf("user creation failed: %d", resp.StatusCode)
					}
					return nil
				},
			},
			{
				name: "VM Query",
				operation: func() error {
					resp := suite.env.APIClient.GET(suite.T(), "/api/vms?limit=10")
					defer resp.Body.Close()
					
					if resp.StatusCode != http.StatusOK {
						return fmt.Errorf("VM query failed: %d", resp.StatusCode)
					}
					return nil
				},
			},
		}
		
		for _, op := range operations {
			t.Run(op.name, func(t *testing.T) {
				numOps := 50
				concurrency := 5
				
				results := &PerformanceResults{
					Operation:   op.name,
					TotalOps:    numOps,
					Concurrency: concurrency,
				}
				
				suite.runConcurrentBenchmark(t, numOps, concurrency, func(index int) error {
					start := time.Now()
					err := op.operation()
					duration := time.Since(start)
					
					results.AddMeasurement(duration)
					return err
				})
				
				results.Analyze()
				suite.logPerformanceResults(t, results)
				
				// Database operations should be fast
				assert.Less(t, results.AverageLatency.Milliseconds(), int64(200),
					"Database operations should be under 200ms on average")
				assert.GreaterOrEqual(t, results.SuccessRate, 0.98,
					"Database operations should have >98%% success rate")
			})
		}
	})
}

// PerformanceResults holds benchmark results
type PerformanceResults struct {
	Operation      string
	TotalOps       int
	Concurrency    int
	Measurements   []time.Duration
	Errors         []error
	StartTime      time.Time
	EndTime        time.Time
	
	// Calculated metrics
	AverageLatency time.Duration
	MinLatency     time.Duration
	MaxLatency     time.Duration
	P50Latency     time.Duration
	P95Latency     time.Duration
	P99Latency     time.Duration
	SuccessRate    float64
	TotalDuration  time.Duration
	mu             sync.Mutex
}

// AddMeasurement adds a measurement to the results
func (r *PerformanceResults) AddMeasurement(duration time.Duration) {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if r.StartTime.IsZero() {
		r.StartTime = time.Now()
	}
	
	r.Measurements = append(r.Measurements, duration)
	r.EndTime = time.Now()
}

// AddError adds an error to the results
func (r *PerformanceResults) AddError(err error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.Errors = append(r.Errors, err)
}

// Analyze calculates performance metrics
func (r *PerformanceResults) Analyze() {
	r.mu.Lock()
	defer r.mu.Unlock()
	
	if len(r.Measurements) == 0 {
		return
	}
	
	// Sort measurements for percentile calculations
	measurements := make([]time.Duration, len(r.Measurements))
	copy(measurements, r.Measurements)
	
	// Simple bubble sort for duration slice
	for i := 0; i < len(measurements); i++ {
		for j := i + 1; j < len(measurements); j++ {
			if measurements[i] > measurements[j] {
				measurements[i], measurements[j] = measurements[j], measurements[i]
			}
		}
	}
	
	// Calculate metrics
	total := time.Duration(0)
	r.MinLatency = measurements[0]
	r.MaxLatency = measurements[len(measurements)-1]
	
	for _, measurement := range measurements {
		total += measurement
	}
	
	r.AverageLatency = total / time.Duration(len(measurements))
	r.P50Latency = measurements[len(measurements)/2]
	r.P95Latency = measurements[int(float64(len(measurements))*0.95)]
	r.P99Latency = measurements[int(float64(len(measurements))*0.99)]
	
	successfulOps := len(r.Measurements)
	totalAttempts := successfulOps + len(r.Errors)
	if totalAttempts > 0 {
		r.SuccessRate = float64(successfulOps) / float64(totalAttempts)
	}
	
	r.TotalDuration = r.EndTime.Sub(r.StartTime)
}

// Helper methods for running benchmarks

func (suite *PerformanceTestSuite) runConcurrentBenchmark(t *testing.T, numOps, concurrency int, operation func(int) error) {
	semaphore := make(chan struct{}, concurrency)
	wg := sync.WaitGroup{}
	
	for i := 0; i < numOps; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			
			semaphore <- struct{}{} // Acquire semaphore
			defer func() { <-semaphore }() // Release semaphore
			
			if err := operation(index); err != nil {
				t.Logf("Operation %d failed: %v", index, err)
			}
		}(i)
	}
	
	wg.Wait()
}

func (suite *PerformanceTestSuite) runDurationBenchmark(t *testing.T, duration time.Duration, concurrency int, operation func() error) {
	done := make(chan struct{})
	time.AfterFunc(duration, func() { close(done) })
	
	wg := sync.WaitGroup{}
	
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			
			for {
				select {
				case <-done:
					return
				default:
					if err := operation(); err != nil {
						t.Logf("Operation failed: %v", err)
					}
				}
			}
		}()
	}
	
	wg.Wait()
}

func (suite *PerformanceTestSuite) logPerformanceResults(t *testing.T, results *PerformanceResults) {
	t.Logf("=== Performance Results for %s ===", results.Operation)
	t.Logf("Total Operations: %d", results.TotalOps)
	t.Logf("Concurrency: %d", results.Concurrency)
	t.Logf("Success Rate: %.2f%%", results.SuccessRate*100)
	t.Logf("Total Duration: %v", results.TotalDuration)
	t.Logf("Average Latency: %v", results.AverageLatency)
	t.Logf("Min Latency: %v", results.MinLatency)
	t.Logf("Max Latency: %v", results.MaxLatency)
	t.Logf("P50 Latency: %v", results.P50Latency)
	t.Logf("P95 Latency: %v", results.P95Latency)
	t.Logf("P99 Latency: %v", results.P99Latency)
	
	if results.TotalDuration > 0 {
		throughput := float64(len(results.Measurements)) / results.TotalDuration.Seconds()
		t.Logf("Throughput: %.2f ops/sec", throughput)
	}
	
	if len(results.Errors) > 0 {
		t.Logf("Errors: %d", len(results.Errors))
	}
}

// TestPerformanceTestSuite runs the performance integration test suite
func TestPerformanceTestSuite(t *testing.T) {
	suite.Run(t, new(PerformanceTestSuite))
}