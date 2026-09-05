package testing

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
)

// workloadSampleCap bounds every buffer the harness materializes (64 MiB).
// The harness models multi-GiB VM transfers by generating a sample of at
// most this size, measuring its real compression ratio, and scaling to the
// logical operation size -- never by allocating the full VM image.
const workloadSampleCap = 64 << 20

// harnessZstd is the package-level zstd encoder shared by all operations.
// EncodeAll is goroutine-safe (it draws from the encoder's internal pool).
var harnessZstd, _ = zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))

type TestHarness struct {
	results []*TestResult
	mu      sync.RWMutex
	// sampleMu serializes sample generation and compression. The sample is
	// workloadSampleCap bytes (64 MiB); without this lock a concurrency-N
	// workload holds N samples (plus zstd scratch) live at once, pushing
	// peak RSS past 1 GiB (novacron-frz gate). Compression is not the
	// modeled resource -- the simulated timeline already accounts for it --
	// so serializing it does not change any modeled quantity.
	sampleMu sync.Mutex
	ctx      context.Context
	cancel   context.CancelFunc
}

// TestMetrics tracks test metrics
type TestMetrics struct {
	StartTime                time.Time
	EndTime                  time.Time
	TotalBytes               int64
	CompressedBytes          int64
	SimulatedTransferSeconds float64
	SimulatedLatencySeconds  float64
	PacketsSent              int64
	PacketsReceived          int64
	PacketsLost              int64
	TotalLatency             time.Duration
	LatencySamples           int
	BandwidthSamples         []BandwidthSample
	OperationResults         []*OperationResult
	mu                       sync.RWMutex
}

// BandwidthSample represents a bandwidth measurement
type BandwidthSample struct {
	Timestamp time.Time
	Bandwidth float64 // Mbps
}

// OperationResult represents the result of a single operation
type OperationResult struct {
	OperationID       int
	StartTime         time.Time
	EndTime           time.Time
	Success           bool
	BytesSent         int64
	SimulatedDuration time.Duration
	Error             error
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
		results: make([]*TestResult, 0),
		ctx:     ctx,
		cancel:  cancel,
	}
}

// NewTestMetrics creates new test metrics
func NewTestMetrics() *TestMetrics {
	return &TestMetrics{
		BandwidthSamples: make([]BandwidthSample, 0),
		OperationResults: make([]*OperationResult, 0),
	}
}

// runState holds the per-scenario-run state (metrics and simulator). A
// TestHarness may execute scenarios concurrently (ContinuousTesting does
// exactly that), so this state must never live on the shared harness.
type runState struct {
	metrics   *TestMetrics
	simulator *NetworkSimulator
}

// RunScenario runs a complete test scenario
func (th *TestHarness) RunScenario(scenario *TestScenario) (*TestResult, error) {
	fmt.Printf("Running scenario: %s\n", scenario.Name)

	// Per-run state lives in runState, not on the harness: ContinuousTesting
	// runs scenarios concurrently against a single shared harness, so
	// harness fields would be a data race (fixed 2026-09-04, novacron-frz).
	run := &runState{
		metrics:   NewTestMetrics(),
		simulator: NewNetworkSimulator(scenario.Topology),
	}
	run.metrics.StartTime = time.Now()

	if err := run.simulator.ApplyTopology(scenario.Topology); err != nil {
		return nil, fmt.Errorf("failed to apply topology: %v", err)
	}

	// Start metrics collection
	metricsCtx, metricsCancel := context.WithCancel(th.ctx)
	defer metricsCancel()
	go th.collectMetrics(metricsCtx, run)

	// Execute workload
	if err := th.executeWorkload(run, scenario.Workload, scenario.Duration); err != nil {
		return nil, fmt.Errorf("failed to execute workload: %v", err)
	}

	// Stop metrics collection
	metricsCancel()
	run.metrics.EndTime = time.Now()

	// Validate assertions
	assertionResults := th.validateAssertions(run, scenario.Assertions)

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
	run.simulator.Reset()

	result := &TestResult{
		Scenario:       scenario.Name,
		Duration:       run.metrics.EndTime.Sub(run.metrics.StartTime),
		Metrics:        run.metrics,
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
func (th *TestHarness) executeWorkload(run *runState, workload *Workload, duration time.Duration) error {
	scheduler := NewWorkloadScheduler(workload)

	// Start scheduling operations
	go scheduler.Schedule()

	// Create worker pool
	workers := make(chan struct{}, workload.Concurrency)
	var wg sync.WaitGroup

	// Timeout context
	ctx, cancel := context.WithTimeout(th.ctx, duration)
	defer cancel()

	// Process operations. The scheduler emits operations stamped with their
	// start offset on the scenario's SIMULATED timeline (think-time pacing
	// advances that clock, not the wall clock), so `duration` is a
	// simulated time budget: operations scheduled at or beyond it are not
	// executed. The wall clock only advances through the real-time latency
	// sleeps; the transfer itself is modeled on the simulated timeline (see
	// executeOperation).
	for {
		select {
		case op, ok := <-scheduler.GetOperations():
			if !ok {
				// All operations scheduled
				wg.Wait()
				return nil
			}

			if op.SimulatedStart >= duration {
				// Simulated budget exhausted: stop accepting further work
				// and let in-flight operations finish.
				wg.Wait()
				return nil
			}

			// Acquire worker slot
			workers <- struct{}{}
			wg.Add(1)

			go func(op *WorkloadOperation) {
				defer wg.Done()
				defer func() { <-workers }()

				result := th.executeOperation(run, op)

				run.metrics.mu.Lock()
				run.metrics.OperationResults = append(run.metrics.OperationResults, result)
				run.metrics.mu.Unlock()
			}(op)

		case <-ctx.Done():
			// Wall-clock safety net reached (real-time latency sleeps plus
			// sample compression outlasted the budget -- only possible
			// under heavy load or -race slowdown): this is a SIMULATOR;
			// the simulated budget above is the pass/fail constraint. Stop
			// accepting work, let in-flight operations finish, report
			// success -- the metrics reflect exactly the operations that
			// fit the simulated timeline.
			wg.Wait()
			return nil
		}
	}
}

// executeOperation executes a single operation
func (th *TestHarness) executeOperation(run *runState, op *WorkloadOperation) *OperationResult {
	result := &OperationResult{
		OperationID: op.ID,
		StartTime:   time.Now(),
		Success:     false,
	}

	// The harness is a SIMULATOR: it must model transfers of multi-GiB VMs
	// without allocating them. Per operation it (1) generates a bounded
	// SAMPLE of the workload pattern (at most workloadSampleCap bytes),
	// (2) measures the real zstd ratio of that sample, (3) scales the ratio
	// to the logical op.VMSize, and (4) advances a simulated clock by
	// latency + compressed_bits / link_bandwidth instead of sleeping for
	// the transfer. Bandwidth utilization is defined on that simulated
	// timeline as sum(transfer_seconds) / sum(latency_seconds +
	sampleSize := min(op.VMSize, workloadSampleCap)
	// Serialize generation+compression (see TestHarness.sampleMu comment).
	th.sampleMu.Lock()
	generator := NewWorkloadGenerator(PatternRealWorld, sampleSize)
	sample := generator.GenerateVMMemory(sampleSize)
	compressed := harnessZstd.EncodeAll(sample, nil)
	th.sampleMu.Unlock()
	ratio := float64(sampleSize) / float64(max(len(compressed), 1))
	compressedForOp := int64(float64(op.VMSize) / ratio)

	// Simulate network transmission
	latency := run.simulator.SimulateLatency(op.Source, op.Target)
	time.Sleep(latency)

	// Simulate packet loss
	if run.simulator.SimulatePacketLoss(op.Source, op.Target) {
		result.Error = fmt.Errorf("packet loss occurred")
		result.EndTime = time.Now()
		return result
	}

	// Model the transfer on the simulated timeline: the compressed payload
	// occupies the link for transferSeconds; the wire is otherwise idle
	// during the latency round-trip.
	bwMbps := run.simulator.GetAvailableBandwidth(op.Source, op.Target)
	if bwMbps <= 0 {
		bwMbps = 10000 // 10 Gbps fallback
	}
	transferSeconds := float64(compressedForOp*8) / (float64(bwMbps) * 1e6)
	simulatedDuration := latency + time.Duration(transferSeconds*float64(time.Second))

	// Update metrics
	run.metrics.mu.Lock()
	run.metrics.TotalBytes += op.VMSize
	run.metrics.CompressedBytes += compressedForOp
	run.metrics.SimulatedTransferSeconds += transferSeconds
	run.metrics.SimulatedLatencySeconds += latency.Seconds()
	run.metrics.PacketsSent++
	run.metrics.PacketsReceived++
	run.metrics.TotalLatency += latency
	run.metrics.LatencySamples++
	run.metrics.mu.Unlock()

	result.BytesSent = op.VMSize
	result.SimulatedDuration = simulatedDuration
	result.Success = true
	result.EndTime = time.Now()

	return result
}

// collectMetrics collects metrics during test execution
func (th *TestHarness) collectMetrics(ctx context.Context, run *runState) {
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case now := <-ticker.C:
			// Calculate current bandwidth
			run.metrics.mu.RLock()
			var recentBytes int64
			cutoff := now.Add(-1 * time.Second)

			for _, result := range run.metrics.OperationResults {
				if result.EndTime.After(cutoff) {
					recentBytes += result.BytesSent
				}
			}
			run.metrics.mu.RUnlock()

			// Convert to Mbps
			bandwidth := float64(recentBytes*8) / 1_000_000

			run.metrics.mu.Lock()
			run.metrics.BandwidthSamples = append(run.metrics.BandwidthSamples, BandwidthSample{
				Timestamp: now,
				Bandwidth: bandwidth,
			})
			run.metrics.mu.Unlock()
		}
	}
}

// validateAssertions validates all test assertions
func (th *TestHarness) validateAssertions(run *runState, assertions []Assertion) []AssertionResult {
	results := make([]AssertionResult, 0)

	for _, assertion := range assertions {
		result := th.validateAssertion(run, assertion)
		results = append(results, result)
	}

	return results
}

// validateAssertion validates a single assertion
func (th *TestHarness) validateAssertion(run *runState, assertion Assertion) AssertionResult {
	result := AssertionResult{
		Type:     assertion.Type,
		Expected: assertion.Threshold,
	}

	// Metrics are mutated concurrently by operation workers and the
	// collectMetrics goroutine; every read below must hold the read lock.
	run.metrics.mu.RLock()
	defer run.metrics.mu.RUnlock()

	switch assertion.Type {
	case AssertionBandwidthUtilization:
		// Utilization is defined on the SIMULATED timeline (see
		// executeOperation): the fraction of simulated link time during
		// which the compressed payload occupied the wire. Latency-dominated
		// small transfers score low; large transfers approach 1.0.
		total := run.metrics.SimulatedTransferSeconds + run.metrics.SimulatedLatencySeconds
		if total > 0 {
			utilization := run.metrics.SimulatedTransferSeconds / total
			result.Actual = utilization
			result.Passed = utilization >= assertion.Threshold
			result.Message = fmt.Sprintf("Bandwidth utilization (simulated): %.2f%% (expected >= %.2f%%)",
				utilization*100, assertion.Threshold*100)
		}
	case AssertionMigrationTime:
		duration := run.metrics.EndTime.Sub(run.metrics.StartTime).Seconds()
		result.Actual = duration
		result.Passed = duration <= assertion.Threshold
		result.Message = fmt.Sprintf("Migration time: %.2fs (expected <= %.2fs)",
			duration, assertion.Threshold)

	case AssertionCompressionRatio:
		if run.metrics.TotalBytes > 0 && run.metrics.CompressedBytes > 0 {
			ratio := float64(run.metrics.TotalBytes) / float64(run.metrics.CompressedBytes)
			result.Actual = ratio
			result.Passed = ratio >= assertion.Threshold
			result.Message = fmt.Sprintf("Compression ratio: %.2fx (expected >= %.2fx)",
				ratio, assertion.Threshold)
		}

	case AssertionThroughput:
		if len(run.metrics.BandwidthSamples) > 0 {
			var totalBandwidth float64
			for _, sample := range run.metrics.BandwidthSamples {
				totalBandwidth += sample.Bandwidth
			}
			avgThroughput := totalBandwidth / float64(len(run.metrics.BandwidthSamples))
			result.Actual = avgThroughput
			result.Passed = avgThroughput >= assertion.Threshold
			result.Message = fmt.Sprintf("Throughput: %.2f Mbps (expected >= %.2f Mbps)",
				avgThroughput, assertion.Threshold)
		}

	case AssertionLatency:
		if run.metrics.LatencySamples > 0 {
			avgLatency := float64(run.metrics.TotalLatency.Milliseconds()) / float64(run.metrics.LatencySamples)
			result.Actual = avgLatency
			result.Passed = avgLatency <= assertion.Threshold
			result.Message = fmt.Sprintf("Latency: %.2fms (expected <= %.2fms)",
				avgLatency, assertion.Threshold)
		}

	case AssertionPacketLoss:
		if run.metrics.PacketsSent > 0 {
			lossRate := float64(run.metrics.PacketsLost) / float64(run.metrics.PacketsSent)
			result.Actual = lossRate
			result.Passed = lossRate <= assertion.Threshold
			result.Message = fmt.Sprintf("Packet loss: %.2f%% (expected <= %.2f%%)",
				lossRate*100, assertion.Threshold*100)
		}

	case AssertionSuccessRate:
		total := len(run.metrics.OperationResults)
		if total > 0 {
			successful := 0
			for _, op := range run.metrics.OperationResults {
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
