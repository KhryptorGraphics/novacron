package testing

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ContinuousTesting manages continuous testing pipeline
type ContinuousTesting struct {
	scenarios    []*TestScenario
	harness      *TestHarness
	reporter     *TestReporter
	schedule     *TestSchedule
	running      bool
	ctx          context.Context
	cancel       context.CancelFunc
	mu           sync.RWMutex
	results      []*TestRun
}

// TestSchedule defines the testing schedule
type TestSchedule struct {
	Interval      time.Duration
	MaxConcurrent int
	Enabled       bool
}

// TestRun represents a complete test run
type TestRun struct {
	ID        string
	StartTime time.Time
	EndTime   time.Time
	Results   []*TestResult
	Summary   *TestSummary
}

// TestSummary summarizes test results
type TestSummary struct {
	TotalTests    int
	PassedTests   int
	FailedTests   int
	SkippedTests  int
	TotalDuration time.Duration
	SuccessRate   float64
}

// NewContinuousTesting creates a new continuous testing manager
func NewContinuousTesting(scenarios []*TestScenario, schedule *TestSchedule) *ContinuousTesting {
	ctx, cancel := context.WithCancel(context.Background())

	return &ContinuousTesting{
		scenarios: scenarios,
		harness:   NewTestHarness(),
		reporter:  NewTestReporter(),
		schedule:  schedule,
		running:   false,
		ctx:       ctx,
		cancel:    cancel,
		results:   make([]*TestRun, 0),
	}
}

// Start starts the continuous testing pipeline
func (ct *ContinuousTesting) Start() error {
	ct.mu.Lock()
	if ct.running {
		ct.mu.Unlock()
		return fmt.Errorf("continuous testing already running")
	}
	ct.running = true
	ct.mu.Unlock()

	fmt.Println("Starting continuous testing pipeline...")

	// Run initial test suite immediately
	go func() {
		if err := ct.runTestSuite(); err != nil {
			fmt.Printf("Error running initial test suite: %v\n", err)
		}
	}()

	// Start scheduled runs
	go ct.scheduledRuns()

	return nil
}

// Stop stops the continuous testing pipeline
func (ct *ContinuousTesting) Stop() {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if !ct.running {
		return
	}

	fmt.Println("Stopping continuous testing pipeline...")
	ct.cancel()
	ct.running = false
	ct.harness.Stop()
}

// scheduledRuns runs tests on schedule
func (ct *ContinuousTesting) scheduledRuns() {
	ticker := time.NewTicker(ct.schedule.Interval)
	defer ticker.Stop()

	for {
		select {
		case <-ct.ctx.Done():
			return
		case <-ticker.C:
			if ct.schedule.Enabled {
				if err := ct.runTestSuite(); err != nil {
					fmt.Printf("Error running scheduled test suite: %v\n", err)
				}
			}
		}
	}
}

// runTestSuite runs all test scenarios
func (ct *ContinuousTesting) runTestSuite() error {
	runID := fmt.Sprintf("run-%d", time.Now().Unix())
	fmt.Printf("\n=== Starting Test Run: %s ===\n", runID)

	testRun := &TestRun{
		ID:        runID,
		StartTime: time.Now(),
		Results:   make([]*TestResult, 0),
	}

	// Run scenarios with concurrency control
	semaphore := make(chan struct{}, ct.schedule.MaxConcurrent)
	var wg sync.WaitGroup
	var resultsMu sync.Mutex

	for _, scenario := range ct.scenarios {
		wg.Add(1)
		semaphore <- struct{}{} // Acquire

		go func(s *TestScenario) {
			defer wg.Done()
			defer func() { <-semaphore }() // Release

			result, err := ct.harness.RunScenario(s)
			if err != nil {
				fmt.Printf("Error running scenario %s: %v\n", s.Name, err)
				return
			}

			resultsMu.Lock()
			testRun.Results = append(testRun.Results, result)
			resultsMu.Unlock()

			status := "PASSED"
			if !result.Passed {
				status = "FAILED"
			}
			fmt.Printf("Scenario %s: %s (Duration: %v)\n", s.Name, status, result.Duration)
		}(scenario)
	}

	wg.Wait()
	testRun.EndTime = time.Now()

	// Generate summary
	testRun.Summary = ct.generateSummary(testRun.Results)

	// Store results
	ct.mu.Lock()
	ct.results = append(ct.results, testRun)
	// Keep only last 100 runs
	if len(ct.results) > 100 {
		ct.results = ct.results[len(ct.results)-100:]
	}
	ct.mu.Unlock()

	// Generate and publish report
	report := ct.reporter.GenerateReport(testRun.Results)
	ct.reporter.PublishToDashboard(report)

	// Alert on failures
	if testRun.Summary.FailedTests > 0 {
		ct.reporter.SendFailureAlert(testRun)
	}

	fmt.Printf("\n=== Test Run Complete: %s ===\n", runID)
	fmt.Printf("Total: %d, Passed: %d, Failed: %d, Success Rate: %.2f%%\n",
		testRun.Summary.TotalTests,
		testRun.Summary.PassedTests,
		testRun.Summary.FailedTests,
		testRun.Summary.SuccessRate*100)

	return nil
}

// generateSummary generates a test summary
func (ct *ContinuousTesting) generateSummary(results []*TestResult) *TestSummary {
	summary := &TestSummary{
		TotalTests: len(results),
	}

	for _, result := range results {
		if result.Passed {
			summary.PassedTests++
		} else {
			summary.FailedTests++
		}
		summary.TotalDuration += result.Duration
	}

	if summary.TotalTests > 0 {
		summary.SuccessRate = float64(summary.PassedTests) / float64(summary.TotalTests)
	}

	return summary
}

// GetLatestRun returns the most recent test run
func (ct *ContinuousTesting) GetLatestRun() *TestRun {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	if len(ct.results) == 0 {
		return nil
	}

	return ct.results[len(ct.results)-1]
}

// GetRunHistory returns test run history
func (ct *ContinuousTesting) GetRunHistory(limit int) []*TestRun {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	if limit <= 0 || limit > len(ct.results) {
		limit = len(ct.results)
	}

	start := len(ct.results) - limit
	return ct.results[start:]
}

// GetTrendAnalysis analyzes testing trends
func (ct *ContinuousTesting) GetTrendAnalysis() *TrendAnalysis {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	if len(ct.results) < 2 {
		return nil
	}

	analysis := &TrendAnalysis{
		Runs:            len(ct.results),
		AverageSuccess:  0,
		SuccessTrend:    "stable",
		DurationTrend:   "stable",
	}

	var totalSuccess float64
	var totalDuration time.Duration

	for _, run := range ct.results {
		totalSuccess += run.Summary.SuccessRate
		totalDuration += run.Summary.TotalDuration
	}

	analysis.AverageSuccess = totalSuccess / float64(len(ct.results))
	analysis.AverageDuration = totalDuration / time.Duration(len(ct.results))

	// Analyze trends (compare first half to second half)
	mid := len(ct.results) / 2
	firstHalf := ct.results[:mid]
	secondHalf := ct.results[mid:]

	firstSuccess := ct.averageSuccessRate(firstHalf)
	secondSuccess := ct.averageSuccessRate(secondHalf)

	if secondSuccess > firstSuccess+0.05 {
		analysis.SuccessTrend = "improving"
	} else if secondSuccess < firstSuccess-0.05 {
		analysis.SuccessTrend = "declining"
	}

	return analysis
}

// TrendAnalysis represents testing trend analysis
type TrendAnalysis struct {
	Runs            int
	AverageSuccess  float64
	AverageDuration time.Duration
	SuccessTrend    string
	DurationTrend   string
}

// averageSuccessRate calculates average success rate
func (ct *ContinuousTesting) averageSuccessRate(runs []*TestRun) float64 {
	if len(runs) == 0 {
		return 0
	}

	var total float64
	for _, run := range runs {
		total += run.Summary.SuccessRate
	}

	return total / float64(len(runs))
}

// RunOnDemand runs a specific scenario on demand
func (ct *ContinuousTesting) RunOnDemand(scenarioName string) (*TestResult, error) {
	var targetScenario *TestScenario

	for _, scenario := range ct.scenarios {
		if scenario.Name == scenarioName {
			targetScenario = scenario
			break
		}
	}

	if targetScenario == nil {
		return nil, fmt.Errorf("scenario not found: %s", scenarioName)
	}

	fmt.Printf("Running on-demand test: %s\n", scenarioName)
	return ct.harness.RunScenario(targetScenario)
}

// AddScenario adds a new test scenario
func (ct *ContinuousTesting) AddScenario(scenario *TestScenario) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.scenarios = append(ct.scenarios, scenario)
}

// RemoveScenario removes a test scenario
func (ct *ContinuousTesting) RemoveScenario(name string) bool {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	for i, scenario := range ct.scenarios {
		if scenario.Name == name {
			ct.scenarios = append(ct.scenarios[:i], ct.scenarios[i+1:]...)
			return true
		}
	}

	return false
}

// UpdateSchedule updates the test schedule
func (ct *ContinuousTesting) UpdateSchedule(schedule *TestSchedule) {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.schedule = schedule
}

// GetStatus returns the current status
func (ct *ContinuousTesting) GetStatus() map[string]interface{} {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	latestRun := ct.GetLatestRun()
	var latestSummary *TestSummary
	if latestRun != nil {
		latestSummary = latestRun.Summary
	}

	return map[string]interface{}{
		"running":          ct.running,
		"scenarios":        len(ct.scenarios),
		"total_runs":       len(ct.results),
		"schedule_enabled": ct.schedule.Enabled,
		"schedule_interval": ct.schedule.Interval.String(),
		"latest_summary":   latestSummary,
	}
}
