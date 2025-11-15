package testing

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// TestHarness executes test scenarios
type TestHarness struct {
	simulator *NetworkSimulator
	metrics   *TestMetrics
	results   []*TestResult
	mu        sync.RWMutex
	ctx       context.Context
	cancel    context.CancelFunc
}

// TestMetrics tracks test metrics
type TestMetrics struct {
	StartTime        time.Time
	EndTime          time.Time
	TotalBytes       int64
	CompressedBytes  int64
	PacketsSent      int64
	PacketsReceived  int64
	PacketsLost      int64
	TotalLatency     time.Duration
	LatencySamples   int
	BandwidthSamples []BandwidthSample
	OperationResults []*OperationResult
	mu               sync.RWMutex
}

// BandwidthSample represents a bandwidth measurement
type BandwidthSample struct {
	Timestamp time.Time
	Bandwidth float64 // Mbps
}

// OperationResult represents the result of a single operation
type OperationResult struct {
	OperationID int
	StartTime   time.Time
	EndTime     time.Time
	Success     bool
	BytesSent   int64
	Error       error
}

// TestResult represents the result of a test scenario
type TestResult struct {
	Scenario       string
	Duration       time.Duration
	Metrics        *TestMetrics
	Passed         bool
	Assertions     []AssertionResult
	FailureReasons []string
}

// AssertionResult represents the result of an assertion
type AssertionResult struct {
	Type     AssertionType
	Expected float64
	Actual   float64
	Passed   bool
	Message  string
}

// NewTestHarness creates a new test harness
func NewTestHarness() *TestHarness {
	ctx, cancel := context.WithCancel(context.Background())

	return &TestHarness{
		simulator: nil,
		metrics:   NewTestMetrics(),
		results:   make([]*TestResult, 0),
		ctx:       ctx,
		cancel:    cancel,
	}
}

// NewTestMetrics creates new test metrics
func NewTestMetrics() *TestMetrics {
	return &TestMetrics{
		BandwidthSamples: make([]BandwidthSample, 0),
		OperationResults: make([]*OperationResult, 0),
	}
}

// RunScenario runs a complete test scenario
func (th *TestHarness) RunScenario(scenario *TestScenario) (*TestResult, error) {
	fmt.Printf("Running scenario: %s\n", scenario.Name)

	// Initialize metrics
	th.metrics = NewTestMetrics()
	th.metrics.StartTime = time.Now()

	// Setup network simulator
	th.simulator = NewNetworkSimulator(scenario.Topology)
	if err := th.simulator.ApplyTopology(scenario.Topology); err != nil {
		return nil, fmt.Errorf("failed to apply topology: %v", err)
	}

	// Start metrics collection
	metricsCtx, metricsCancel := context.WithCancel(th.ctx)
	defer metricsCancel()
	go th.collectMetrics(metricsCtx)

	// Execute workload
	if err := th.executeWorkload(scenario.Workload, scenario.Duration); err != nil {
		return nil, fmt.Errorf("failed to execute workload: %v", err)
	}

	// Stop metrics collection
	metricsCancel()
	th.metrics.EndTime = time.Now()

	// Validate assertions
	assertionResults := th.validateAssertions(scenario.Assertions)

	// Determine if test passed
	passed := true
	failureReasons := make([]string, 0)

	for _, ar := range assertionResults {
		if !ar.Passed {
			passed = false
			if ar.Message != "" {
				failureReasons = append(failureReasons, ar.Message)
			}
		}
	}

	// Cleanup
	th.simulator.Reset()

	result := &TestResult{
		Scenario:       scenario.Name,
		Duration:       th.metrics.EndTime.Sub(th.metrics.StartTime),
		Metrics:        th.metrics,
		Passed:         passed,
		Assertions:     assertionResults,
		FailureReasons: failureReasons,
	}

	th.mu.Lock()
	th.results = append(th.results, result)
	th.mu.Unlock()

	return result, nil
}

// executeWorkload executes the test workload
func (th *TestHarness) executeWorkload(workload *Workload, duration time.Duration) error {
	scheduler := NewWorkloadScheduler(workload)

	// Start scheduling operations
	go scheduler.Schedule()

	// Create worker pool
	workers := make(chan struct{}, workload.Concurrency)
	var wg sync.WaitGroup

	// Timeout context
	ctx, cancel := context.WithTimeout(th.ctx, duration)
	defer cancel()

	// Process operations
	for {
		select {
		case op, ok := <-scheduler.GetOperations():
			if !ok {
				// All operations scheduled
				wg.Wait()
				return nil
			}

			// Acquire worker slot
			workers <- struct{}{}
			wg.Add(1)

			go func(op *WorkloadOperation) {
				defer wg.Done()
				defer func() { <-workers }()

				result := th.executeOperation(op)

				th.metrics.mu.Lock()
				th.metrics.OperationResults = append(th.metrics.OperationResults, result)
				th.metrics.mu.Unlock()
			}(op)

		case <-ctx.Done():
			// Timeout reached
			wg.Wait()
			return fmt.Errorf("workload execution timed out after %v", duration)
		}
	}
}

// executeOperation executes a single operation
func (th *TestHarness) executeOperation(op *WorkloadOperation) *OperationResult {
	result := &OperationResult{
		OperationID: op.ID,
		StartTime:   time.Now(),
		Success:     false,
	}

	// Generate workload data
	generator := NewWorkloadGenerator(PatternRealWorld, op.VMSize)
	data := generator.GenerateVMMemory(op.VMSize)

	// Simulate network transmission
	latency := th.simulator.SimulateLatency(op.Source, op.Target)
	time.Sleep(latency)

	// Simulate packet loss
	if th.simulator.SimulatePacketLoss(op.Source, op.Target) {
		result.Error = fmt.Errorf("packet loss occurred")
		result.EndTime = time.Now()
		return result
	}

	// Update metrics
	th.metrics.mu.Lock()
	th.metrics.TotalBytes += int64(len(data))
	th.metrics.PacketsSent++
	th.metrics.PacketsReceived++
	th.metrics.TotalLatency += latency
	th.metrics.LatencySamples++
	th.metrics.mu.Unlock()

	result.BytesSent = int64(len(data))
	result.Success = true
	result.EndTime = time.Now()

	return result
}

// collectMetrics collects metrics during test execution
func (th *TestHarness) collectMetrics(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			// Calculate current bandwidth
			th.metrics.mu.RLock()
			var recentBytes int64
			cutoff := now.Add(-1 * time.Second)

			for _, result := range th.metrics.OperationResults {
				if result.EndTime.After(cutoff) {
					recentBytes += result.BytesSent
				}
			}
			th.metrics.mu.RUnlock()

			// Convert to Mbps
			bandwidth := float64(recentBytes*8) / 1_000_000

			th.metrics.mu.Lock()
			th.metrics.BandwidthSamples = append(th.metrics.BandwidthSamples, BandwidthSample{
				Timestamp: now,
				Bandwidth: bandwidth,
			})
			th.metrics.mu.Unlock()
		}
	}
}

// validateAssertions validates all test assertions
func (th *TestHarness) validateAssertions(assertions []Assertion) []AssertionResult {
	results := make([]AssertionResult, 0)

	for _, assertion := range assertions {
		result := th.validateAssertion(assertion)
		results = append(results, result)
	}

	return results
}

// validateAssertion validates a single assertion
func (th *TestHarness) validateAssertion(assertion Assertion) AssertionResult {
	result := AssertionResult{
		Type:     assertion.Type,
		Expected: assertion.Threshold,
	}

	th.metrics.mu.RLock()
	defer th.metrics.mu.RUnlock()

	switch assertion.Type {
	case AssertionBandwidthUtilization:
		if len(th.metrics.BandwidthSamples) > 0 {
			var totalBandwidth float64
			for _, sample := range th.metrics.BandwidthSamples {
				totalBandwidth += sample.Bandwidth
			}
			avgBandwidth := totalBandwidth / float64(len(th.metrics.BandwidthSamples))

			// Assume 10 Gbps link
			utilization := avgBandwidth / 10000.0
			result.Actual = utilization
			result.Passed = utilization >= assertion.Threshold
			result.Message = fmt.Sprintf("Bandwidth utilization: %.2f%% (expected >= %.2f%%)",
				utilization*100, assertion.Threshold*100)
		}

	case AssertionMigrationTime:
		duration := th.metrics.EndTime.Sub(th.metrics.StartTime).Seconds()
		result.Actual = duration
		result.Passed = duration <= assertion.Threshold
		result.Message = fmt.Sprintf("Migration time: %.2fs (expected <= %.2fs)",
			duration, assertion.Threshold)

	case AssertionCompressionRatio:
		if th.metrics.TotalBytes > 0 && th.metrics.CompressedBytes > 0 {
			ratio := float64(th.metrics.TotalBytes) / float64(th.metrics.CompressedBytes)
			result.Actual = ratio
			result.Passed = ratio >= assertion.Threshold
			result.Message = fmt.Sprintf("Compression ratio: %.2fx (expected >= %.2fx)",
				ratio, assertion.Threshold)
		}

	case AssertionThroughput:
		if len(th.metrics.BandwidthSamples) > 0 {
			var totalBandwidth float64
			for _, sample := range th.metrics.BandwidthSamples {
				totalBandwidth += sample.Bandwidth
			}
			avgThroughput := totalBandwidth / float64(len(th.metrics.BandwidthSamples))
			result.Actual = avgThroughput
			result.Passed = avgThroughput >= assertion.Threshold
			result.Message = fmt.Sprintf("Throughput: %.2f Mbps (expected >= %.2f Mbps)",
				avgThroughput, assertion.Threshold)
		}

	case AssertionLatency:
		if th.metrics.LatencySamples > 0 {
			avgLatency := float64(th.metrics.TotalLatency.Milliseconds()) / float64(th.metrics.LatencySamples)
			result.Actual = avgLatency
			result.Passed = avgLatency <= assertion.Threshold
			result.Message = fmt.Sprintf("Latency: %.2fms (expected <= %.2fms)",
				avgLatency, assertion.Threshold)
		}

	case AssertionPacketLoss:
		if th.metrics.PacketsSent > 0 {
			lossRate := float64(th.metrics.PacketsLost) / float64(th.metrics.PacketsSent)
			result.Actual = lossRate
			result.Passed = lossRate <= assertion.Threshold
			result.Message = fmt.Sprintf("Packet loss: %.2f%% (expected <= %.2f%%)",
				lossRate*100, assertion.Threshold*100)
		}

	case AssertionSuccessRate:
		total := len(th.metrics.OperationResults)
		if total > 0 {
			successful := 0
			for _, op := range th.metrics.OperationResults {
				if op.Success {
					successful++
				}
			}
			successRate := float64(successful) / float64(total)
			result.Actual = successRate
			result.Passed = successRate >= assertion.Threshold
			result.Message = fmt.Sprintf("Success rate: %.2f%% (expected >= %.2f%%)",
				successRate*100, assertion.Threshold*100)
		}
	}

	return result
}

// GetResults returns all test results
func (th *TestHarness) GetResults() []*TestResult {
	th.mu.RLock()
	defer th.mu.RUnlock()
	return th.results
}

// Stop stops the test harness
func (th *TestHarness) Stop() {
	th.cancel()
	if th.simulator != nil {
		th.simulator.Reset()
	}
}

// PrintResults prints test results in a readable format
func (th *TestHarness) PrintResults() {
	th.mu.RLock()
	defer th.mu.RUnlock()

	fmt.Println("\n=== Test Results ===")
	for _, result := range th.results {
		status := "PASSED"
		if !result.Passed {
			status = "FAILED"
		}

		fmt.Printf("\nScenario: %s [%s]\n", result.Scenario, status)
		fmt.Printf("Duration: %v\n", result.Duration)
		fmt.Printf("Total Operations: %d\n", len(result.Metrics.OperationResults))
		fmt.Printf("Total Bytes: %d\n", result.Metrics.TotalBytes)

		if result.Metrics.LatencySamples > 0 {
			avgLatency := float64(result.Metrics.TotalLatency.Milliseconds()) / float64(result.Metrics.LatencySamples)
			fmt.Printf("Average Latency: %.2fms\n", avgLatency)
		}

		fmt.Println("\nAssertions:")
		for _, ar := range result.Assertions {
			status := "✓"
			if !ar.Passed {
				status = "✗"
			}
			fmt.Printf("  %s %s: %s\n", status, ar.Type, ar.Message)
		}

		if len(result.FailureReasons) > 0 {
			fmt.Println("\nFailure Reasons:")
			for _, reason := range result.FailureReasons {
				fmt.Printf("  - %s\n", reason)
			}
		}
	}
}
