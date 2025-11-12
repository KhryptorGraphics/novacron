package testing

import (
	"fmt"
	"testing"
	"time"
)

// TestFullTestingPipeline tests the complete testing framework
func TestFullTestingPipeline(t *testing.T) {
	// Create test harness
	harness := NewTestHarness()

	// Get all scenarios
	scenarios := GetAllScenarios()

	if len(scenarios) == 0 {
		t.Fatal("No scenarios found")
	}

	t.Logf("Running %d test scenarios", len(scenarios))

	// Run each scenario
	for _, scenario := range scenarios {
		// Reduce duration for testing
		scenario.Duration = 30 * time.Second

		t.Run(scenario.Name, func(t *testing.T) {
			result, err := harness.RunScenario(scenario)
			if err != nil {
				t.Errorf("Failed to run scenario: %v", err)
				return
			}

			if !result.Passed {
				t.Logf("Scenario failed: %s", scenario.Name)
				for _, reason := range result.FailureReasons {
					t.Logf("  Failure: %s", reason)
				}
			} else {
				t.Logf("Scenario passed: %s (Duration: %v)", scenario.Name, result.Duration)
			}
		})
	}

	// Print summary
	results := harness.GetResults()
	passed := 0
	failed := 0

	for _, result := range results {
		if result.Passed {
			passed++
		} else {
			failed++
		}
	}

	t.Logf("\n=== Test Summary ===")
	t.Logf("Total: %d, Passed: %d, Failed: %d", len(results), passed, failed)
}

// TestContinuousTesting tests the continuous testing pipeline
func TestContinuousTesting(t *testing.T) {
	scenarios := []*TestScenario{
		NewCrossRegionScenario(),
		NewPacketLossScenario(),
	}

	// Shorten durations for testing
	for _, scenario := range scenarios {
		scenario.Duration = 15 * time.Second
	}

	schedule := &TestSchedule{
		Interval:      1 * time.Minute,
		MaxConcurrent: 2,
		Enabled:       false, // Don't auto-run during test
	}

	ct := NewContinuousTesting(scenarios, schedule)

	// Run test suite manually
	if err := ct.runTestSuite(); err != nil {
		t.Errorf("Failed to run test suite: %v", err)
	}

	// Check results
	latestRun := ct.GetLatestRun()
	if latestRun == nil {
		t.Fatal("No test run found")
	}

	t.Logf("Test run completed: %d tests, %.2f%% success rate",
		latestRun.Summary.TotalTests,
		latestRun.Summary.SuccessRate*100)

	// Get status
	status := ct.GetStatus()
	t.Logf("Continuous testing status: %+v", status)
}

// TestChaosEngineering tests chaos experiments
func TestChaosEngineering(t *testing.T) {
	// Create network simulator
	topology := NewCrossRegionScenario().Topology
	simulator := NewNetworkSimulator(topology)
	simulator.ApplyTopology(topology)

	// Create traffic controller (requires root/sudo)
	// Skip if not running with privileges
	tcController := NewTrafficController("lo") // Use loopback for testing

	// Create chaos engine
	chaosEngine := NewChaosEngine(simulator, tcController)

	// Add experiments
	experiments := PredefinedExperiments()
	for _, exp := range experiments {
		// Shorten durations for testing
		exp.Duration = 10 * time.Second
		chaosEngine.AddExperiment(exp)
	}

	// Run a simple experiment
	exp := &ChaosExperiment{
		Name:        "Test High Latency",
		Description: "Test latency injection",
		FaultType:   FaultHighLatency,
		Probability: 0.5,
		Duration:    5 * time.Second,
		ImpactLevel: ImpactMedium,
		Recovery:    RecoveryImmediate,
	}

	chaosEngine.AddExperiment(exp)

	// Note: Actual fault injection requires root privileges
	// In CI, this would be skipped or run in privileged container
	t.Log("Chaos engine initialized (fault injection requires root)")

	// Cleanup
	chaosEngine.Stop()
}

// TestReporter tests the test reporter
func TestReporter(t *testing.T) {
	reporter := NewTestReporter()

	// Create mock test results
	results := []*TestResult{
		{
			Scenario: "Test 1",
			Duration: 30 * time.Second,
			Passed:   true,
			Metrics:  NewTestMetrics(),
		},
		{
			Scenario: "Test 2",
			Duration: 45 * time.Second,
			Passed:   false,
			Metrics:  NewTestMetrics(),
			FailureReasons: []string{
				"Latency exceeded threshold",
				"Packet loss too high",
			},
		},
	}

	// Generate report
	report := reporter.GenerateReport(results)

	if report.TotalTests != 2 {
		t.Errorf("Expected 2 tests, got %d", report.TotalTests)
	}

	if report.PassedTests != 1 {
		t.Errorf("Expected 1 passed test, got %d", report.PassedTests)
	}

	if report.FailedTests != 1 {
		t.Errorf("Expected 1 failed test, got %d", report.FailedTests)
	}

	// Generate HTML report
	html := reporter.GenerateHTMLReport(report)
	if len(html) == 0 {
		t.Error("HTML report is empty")
	}

	// Generate CSV export
	csv := reporter.ExportToCSV(report)
	if len(csv) == 0 {
		t.Error("CSV export is empty")
	}

	// Generate JSON export
	jsonData, err := reporter.ExportToJSON(report)
	if err != nil {
		t.Errorf("Failed to export JSON: %v", err)
	}
	if len(jsonData) == 0 {
		t.Error("JSON export is empty")
	}

	// Print report
	reporter.PrintReport(report)
}

// TestWorkloadGenerator tests workload generation
func TestWorkloadGenerator(t *testing.T) {
	patterns := []WorkloadPattern{
		PatternConstant,
		PatternBursty,
		PatternSinusoidal,
		PatternRealWorld,
	}

	size := int64(100 * 1024 * 1024) // 100 MB

	for _, pattern := range patterns {
		t.Run(string(pattern), func(t *testing.T) {
			generator := NewWorkloadGenerator(pattern, size)
			data := generator.GenerateVMMemory(size)

			if int64(len(data)) != size {
				t.Errorf("Expected %d bytes, got %d", size, len(data))
			}

			// Generate snapshot
			snapshot := generator.GenerateMemorySnapshot("test-vm", size)

			if snapshot.Size != size {
				t.Errorf("Expected snapshot size %d, got %d", size, snapshot.Size)
			}

			t.Logf("Pattern %s: Zero pages=%d, Compression ratio=%.2fx, Entropy=%.2f",
				pattern,
				snapshot.Metadata.ZeroPages,
				snapshot.Metadata.CompressRatio,
				snapshot.Metadata.Entropy)
		})
	}
}

// TestNetworkSimulator tests the network simulator
func TestNetworkSimulator(t *testing.T) {
	topology := NewCrossRegionScenario().Topology
	simulator := NewNetworkSimulator(topology)

	if err := simulator.ApplyTopology(topology); err != nil {
		t.Fatalf("Failed to apply topology: %v", err)
	}

	// Test latency simulation
	for i := 0; i < 100; i++ {
		latency := simulator.SimulateLatency("us-east-1", "eu-west-1")
		if latency < 0 {
			t.Error("Latency cannot be negative")
		}
	}

	// Test packet loss simulation
	lossCount := 0
	for i := 0; i < 1000; i++ {
		if simulator.SimulatePacketLoss("us-east-1", "eu-west-1") {
			lossCount++
		}
	}

	lossRate := float64(lossCount) / 1000.0
	t.Logf("Simulated packet loss rate: %.2f%%", lossRate*100)

	// Test bandwidth
	bandwidth := simulator.GetAvailableBandwidth("us-east-1", "eu-west-1")
	if bandwidth <= 0 {
		t.Error("Bandwidth must be positive")
	}

	// Get statistics
	stats := simulator.GetTopologyStats()
	t.Logf("Topology stats: %+v", stats)

	// Cleanup
	simulator.Reset()
}

// TestScenarioSelection tests scenario selection functions
func TestScenarioSelection(t *testing.T) {
	// Test get all scenarios
	scenarios := GetAllScenarios()
	if len(scenarios) == 0 {
		t.Fatal("No scenarios found")
	}
	t.Logf("Found %d scenarios", len(scenarios))

	// Test get by name
	scenario := GetScenarioByName("Cross-Region Migration")
	if scenario == nil {
		t.Error("Failed to find scenario by name")
	}

	// Test get by tag
	migrationScenarios := GetScenariosByTag("migration")
	if len(migrationScenarios) == 0 {
		t.Error("No migration scenarios found")
	}
	t.Logf("Found %d migration scenarios", len(migrationScenarios))

	// Test scenario string representation
	for _, s := range scenarios {
		str := s.String()
		if len(str) == 0 {
			t.Error("Scenario string is empty")
		}
		t.Log(str)
	}
}

// TestTrafficControlProfiles tests predefined network profiles
func TestTrafficControlProfiles(t *testing.T) {
	profiles := PredefinedProfiles()

	if len(profiles) == 0 {
		t.Fatal("No predefined profiles found")
	}

	for name, profile := range profiles {
		t.Logf("Profile: %s", name)
		t.Logf("  Latency: %v", profile.Latency)
		t.Logf("  Jitter: %v", profile.Jitter)
		t.Logf("  Packet Loss: %.2f%%", profile.PacketLoss)
		t.Logf("  Bandwidth: %d Mbps", profile.Bandwidth)
	}
}

// Helper function to create a simple test scenario
func createSimpleTestScenario() *TestScenario {
	return &TestScenario{
		Name:        "Simple Test",
		Description: "Simple test scenario for unit testing",
		Topology: &NetworkTopology{
			Datacenters: map[string]*Datacenter{
				"dc1": {ID: "dc1", Region: "test1"},
				"dc2": {ID: "dc2", Region: "test2"},
			},
			Links: map[string]*Link{
				"dc1-dc2": {
					Source:      "dc1",
					Destination: "dc2",
					Latency: LatencyProfile{
						BaseLatency: 10 * time.Millisecond,
						Jitter:      2 * time.Millisecond,
					},
					Bandwidth: BandwidthProfile{
						Capacity:    1000,
						Utilization: 0.3,
					},
					PacketLoss: LossProfile{
						Rate: 0.001,
					},
				},
			},
		},
		Workload: &Workload{
			Type:        WorkloadMigration,
			VMs:         2,
			VMSize:      1024 * 1024 * 1024, // 1 GB
			Operations:  2,
			Concurrency: 1,
			Pattern:     PatternConstant,
		},
		Duration: 10 * time.Second,
		Assertions: []Assertion{
			{
				Type:      AssertionSuccessRate,
				Threshold: 0.95,
				Critical:  true,
			},
		},
	}
}
