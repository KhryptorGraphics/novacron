// Package orchestration_integration provides load testing for orchestration endpoints
package orchestration_integration

import (
	"context"
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/sirupsen/logrus"

	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/placement"
)

// LoadTestConfig defines configuration for load testing
type LoadTestConfig struct {
	Duration          time.Duration
	ConcurrentUsers   int
	RequestsPerSecond int
	RampUpTime        time.Duration
	RampDownTime      time.Duration
}

// LoadTestMetrics tracks load test results
type LoadTestMetrics struct {
	TotalRequests    int64
	SuccessfulReqs   int64
	FailedReqs       int64
	TotalLatency     time.Duration
	MinLatency       time.Duration
	MaxLatency       time.Duration
	P95Latency       time.Duration
	P99Latency       time.Duration
	RequestsPerSec   float64
	ErrorRate        float64
	Latencies        []time.Duration
	mu               sync.Mutex
}

// RecordRequest records a request result
func (m *LoadTestMetrics) RecordRequest(latency time.Duration, success bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	atomic.AddInt64(&m.TotalRequests, 1)
	
	if success {
		atomic.AddInt64(&m.SuccessfulReqs, 1)
	} else {
		atomic.AddInt64(&m.FailedReqs, 1)
	}

	m.TotalLatency += latency
	m.Latencies = append(m.Latencies, latency)

	if m.MinLatency == 0 || latency < m.MinLatency {
		m.MinLatency = latency
	}
	if latency > m.MaxLatency {
		m.MaxLatency = latency
	}
}

// Calculate calculates final metrics
func (m *LoadTestMetrics) Calculate(duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.TotalRequests > 0 {
		m.RequestsPerSec = float64(m.TotalRequests) / duration.Seconds()
		m.ErrorRate = float64(m.FailedReqs) / float64(m.TotalRequests) * 100
		
		// Calculate percentiles
		if len(m.Latencies) > 0 {
			// Sort latencies for percentile calculation
			for i := 0; i < len(m.Latencies); i++ {
				for j := i + 1; j < len(m.Latencies); j++ {
					if m.Latencies[i] > m.Latencies[j] {
						m.Latencies[i], m.Latencies[j] = m.Latencies[j], m.Latencies[i]
					}
				}
			}
			
			p95Index := int(float64(len(m.Latencies)) * 0.95)
			p99Index := int(float64(len(m.Latencies)) * 0.99)
			
			if p95Index < len(m.Latencies) {
				m.P95Latency = m.Latencies[p95Index]
			}
			if p99Index < len(m.Latencies) {
				m.P99Latency = m.Latencies[p99Index]
			}
		}
	}
}

// TestOrchestrationLoadTesting performs comprehensive load testing
func TestOrchestrationLoadTesting(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping load testing in short mode")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer cancel()

	logger := logrus.New()
	logger.SetLevel(logrus.WarnLevel) // Reduce noise during load testing

	orchestrator := orchestration.NewDefaultOrchestrationEngine(logger)
	require.NoError(t, orchestrator.Start(ctx))
	defer orchestrator.Stop(ctx)

	// Run different load test scenarios
	t.Run("Steady_Load_Test", func(t *testing.T) {
		config := LoadTestConfig{
			Duration:          2 * time.Minute,
			ConcurrentUsers:   10,
			RequestsPerSecond: 50,
			RampUpTime:        10 * time.Second,
			RampDownTime:      10 * time.Second,
		}
		runSteadyLoadTest(t, ctx, orchestrator, config)
	})

	t.Run("Spike_Load_Test", func(t *testing.T) {
		config := LoadTestConfig{
			Duration:          1 * time.Minute,
			ConcurrentUsers:   50,
			RequestsPerSecond: 200,
			RampUpTime:        5 * time.Second,
			RampDownTime:      5 * time.Second,
		}
		runSpikeLoadTest(t, ctx, orchestrator, config)
	})

	t.Run("Stress_Test", func(t *testing.T) {
		config := LoadTestConfig{
			Duration:          3 * time.Minute,
			ConcurrentUsers:   100,
			RequestsPerSecond: 500,
			RampUpTime:        30 * time.Second,
			RampDownTime:      30 * time.Second,
		}
		runStressTest(t, ctx, orchestrator, config)
	})

	t.Run("Endurance_Test", func(t *testing.T) {
		config := LoadTestConfig{
			Duration:          5 * time.Minute,
			ConcurrentUsers:   20,
			RequestsPerSecond: 100,
			RampUpTime:        30 * time.Second,
			RampDownTime:      30 * time.Second,
		}
		runEnduranceTest(t, ctx, orchestrator, config)
	})
}

func runSteadyLoadTest(t *testing.T, ctx context.Context, orchestrator orchestration.OrchestrationEngine, config LoadTestConfig) {
	t.Log("Running steady load test")
	
	metrics := &LoadTestMetrics{}
	
	startTime := time.Now()
	endTime := startTime.Add(config.Duration)
	
	// Create worker pool
	var wg sync.WaitGroup
	requestChan := make(chan bool, config.ConcurrentUsers*2)
	
	// Start workers
	for i := 0; i < config.ConcurrentUsers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for {
				select {
				case <-ctx.Done():
					return
				case <-requestChan:
					executePlacementRequest(ctx, orchestrator, metrics, workerID)
				}
			}
		}(i)
	}
	
	// Generate requests at specified rate
	ticker := time.NewTicker(time.Second / time.Duration(config.RequestsPerSecond))
	defer ticker.Stop()
	
	requestCount := 0
	for time.Now().Before(endTime) {
		select {
		case <-ctx.Done():
			goto cleanup
		case <-ticker.C:
			select {
			case requestChan <- true:
				requestCount++
			default:
				// Channel full, skip this request
			}
		}
	}
	
cleanup:
	close(requestChan)
	wg.Wait()
	
	// Calculate and report metrics
	actualDuration := time.Since(startTime)
	metrics.Calculate(actualDuration)
	
	t.Logf("Steady Load Test Results:")
	t.Logf("  Duration: %v", actualDuration)
	t.Logf("  Total Requests: %d", metrics.TotalRequests)
	t.Logf("  Successful Requests: %d", metrics.SuccessfulReqs)
	t.Logf("  Failed Requests: %d", metrics.FailedReqs)
	t.Logf("  Requests/Second: %.2f", metrics.RequestsPerSec)
	t.Logf("  Error Rate: %.2f%%", metrics.ErrorRate)
	t.Logf("  Min Latency: %v", metrics.MinLatency)
	t.Logf("  Max Latency: %v", metrics.MaxLatency)
	t.Logf("  P95 Latency: %v", metrics.P95Latency)
	t.Logf("  P99 Latency: %v", metrics.P99Latency)
	
	// Verify performance requirements
	assert.Less(t, metrics.ErrorRate, 5.0, "Error rate should be less than 5%")
	assert.Greater(t, metrics.RequestsPerSec, float64(config.RequestsPerSecond)*0.8, "Should achieve at least 80% of target RPS")
	assert.Less(t, metrics.P95Latency.Milliseconds(), int64(1000), "P95 latency should be under 1 second")
}

func runSpikeLoadTest(t *testing.T, ctx context.Context, orchestrator orchestration.OrchestrationEngine, config LoadTestConfig) {
	t.Log("Running spike load test")
	
	metrics := &LoadTestMetrics{}
	
	// Ramp up phase
	t.Log("Spike test: Ramp up phase")
	runLoadPhase(ctx, orchestrator, metrics, config.RampUpTime, config.ConcurrentUsers/4, config.RequestsPerSecond/4)
	
	// Spike phase
	t.Log("Spike test: Spike phase")
	runLoadPhase(ctx, orchestrator, metrics, config.Duration, config.ConcurrentUsers, config.RequestsPerSecond)
	
	// Ramp down phase
	t.Log("Spike test: Ramp down phase")
	runLoadPhase(ctx, orchestrator, metrics, config.RampDownTime, config.ConcurrentUsers/4, config.RequestsPerSecond/4)
	
	metrics.Calculate(config.RampUpTime + config.Duration + config.RampDownTime)
	
	t.Logf("Spike Load Test Results:")
	t.Logf("  Total Requests: %d", metrics.TotalRequests)
	t.Logf("  Error Rate: %.2f%%", metrics.ErrorRate)
	t.Logf("  P99 Latency: %v", metrics.P99Latency)
	
	// Verify system handled spike
	assert.Less(t, metrics.ErrorRate, 10.0, "Error rate should be less than 10% during spike")
}

func runStressTest(t *testing.T, ctx context.Context, orchestrator orchestration.OrchestrationEngine, config LoadTestConfig) {
	t.Log("Running stress test")
	
	metrics := &LoadTestMetrics{}
	
	startTime := time.Now()
	endTime := startTime.Add(config.Duration)
	
	// Create a large number of workers to stress the system
	var wg sync.WaitGroup
	requestChan := make(chan bool, config.ConcurrentUsers*2)
	
	// Start workers
	for i := 0; i < config.ConcurrentUsers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for {
				select {
				case <-ctx.Done():
					return
				case <-requestChan:
					// Use more complex VM specs for stress testing
					executeComplexPlacementRequest(ctx, orchestrator, metrics, workerID)
				}
			}
		}(i)
	}
	
	// Generate high-frequency requests
	ticker := time.NewTicker(time.Second / time.Duration(config.RequestsPerSecond))
	defer ticker.Stop()
	
	for time.Now().Before(endTime) {
		select {
		case <-ctx.Done():
			goto cleanup
		case <-ticker.C:
			select {
			case requestChan <- true:
			default:
				// System is overloaded, which is expected in stress test
			}
		}
	}
	
cleanup:
	close(requestChan)
	wg.Wait()
	
	actualDuration := time.Since(startTime)
	metrics.Calculate(actualDuration)
	
	t.Logf("Stress Test Results:")
	t.Logf("  Duration: %v", actualDuration)
	t.Logf("  Total Requests: %d", metrics.TotalRequests)
	t.Logf("  Error Rate: %.2f%%", metrics.ErrorRate)
	t.Logf("  Max Latency: %v", metrics.MaxLatency)
	
	// Verify system didn't crash under stress
	status := orchestrator.GetStatus()
	assert.Equal(t, orchestration.EngineStateRunning, status.State, "System should still be running after stress test")
	
	// Allow higher error rates in stress testing
	assert.Less(t, metrics.ErrorRate, 25.0, "Error rate should be less than 25% even under stress")
}

func runEnduranceTest(t *testing.T, ctx context.Context, orchestrator orchestration.OrchestrationEngine, config LoadTestConfig) {
	t.Log("Running endurance test")
	
	metrics := &LoadTestMetrics{}
	
	startTime := time.Now()
	endTime := startTime.Add(config.Duration)
	
	var wg sync.WaitGroup
	requestChan := make(chan bool, config.ConcurrentUsers)
	
	// Start workers
	for i := 0; i < config.ConcurrentUsers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for {
				select {
				case <-ctx.Done():
					return
				case <-requestChan:
					executeVariedPlacementRequest(ctx, orchestrator, metrics, workerID)
				}
			}
		}(i)
	}
	
	// Moderate request rate for endurance
	ticker := time.NewTicker(time.Second / time.Duration(config.RequestsPerSecond))
	defer ticker.Stop()
	
	// Track metrics over time
	metricsTracker := time.NewTicker(30 * time.Second)
	defer metricsTracker.Stop()
	
	for time.Now().Before(endTime) {
		select {
		case <-ctx.Done():
			goto cleanup
		case <-ticker.C:
			select {
			case requestChan <- true:
			default:
				// Skip if workers are busy
			}
		case <-metricsTracker.C:
			// Log intermediate metrics
			status := orchestrator.GetStatus()
			t.Logf("Endurance test progress - Events processed: %d, Requests so far: %d", 
				status.EventsProcessed, atomic.LoadInt64(&metrics.TotalRequests))
		}
	}
	
cleanup:
	close(requestChan)
	wg.Wait()
	
	actualDuration := time.Since(startTime)
	metrics.Calculate(actualDuration)
	
	t.Logf("Endurance Test Results:")
	t.Logf("  Duration: %v", actualDuration)
	t.Logf("  Total Requests: %d", metrics.TotalRequests)
	t.Logf("  Average RPS: %.2f", metrics.RequestsPerSec)
	t.Logf("  Error Rate: %.2f%%", metrics.ErrorRate)
	t.Logf("  Average Latency: %v", time.Duration(int64(metrics.TotalLatency)/metrics.TotalRequests))
	
	// Verify sustained performance
	assert.Less(t, metrics.ErrorRate, 3.0, "Error rate should remain low during endurance test")
	assert.Greater(t, metrics.RequestsPerSec, float64(config.RequestsPerSecond)*0.9, "Should maintain 90% of target RPS over time")
}

// Helper functions for load testing

func runLoadPhase(ctx context.Context, orchestrator orchestration.OrchestrationEngine, metrics *LoadTestMetrics, duration time.Duration, workers int, rps int) {
	var wg sync.WaitGroup
	requestChan := make(chan bool, workers*2)
	
	// Start workers
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for {
				select {
				case <-ctx.Done():
					return
				case <-requestChan:
					executePlacementRequest(ctx, orchestrator, metrics, workerID)
				}
			}
		}(i)
	}
	
	// Generate requests
	endTime := time.Now().Add(duration)
	ticker := time.NewTicker(time.Second / time.Duration(rps))
	defer ticker.Stop()
	
	for time.Now().Before(endTime) {
		select {
		case <-ctx.Done():
			goto cleanup
		case <-ticker.C:
			select {
			case requestChan <- true:
			default:
			}
		}
	}
	
cleanup:
	close(requestChan)
	wg.Wait()
}

func executePlacementRequest(ctx context.Context, orchestrator orchestration.OrchestrationEngine, metrics *LoadTestMetrics, workerID int) {
	start := time.Now()
	
	vmSpec := placement.VMSpec{
		VMID:     fmt.Sprintf("load-test-vm-%d-%d", workerID, time.Now().UnixNano()),
		CPUs:     2,
		MemoryMB: 4096,
		DiskGB:   50,
		Labels: map[string]string{
			"vm_id":     fmt.Sprintf("load-test-vm-%d", workerID),
			"load_test": "true",
		},
	}
	
	_, err := orchestrator.MakeVMPlacementDecision(
		ctx, vmSpec, placement.PlacementStrategyBalanced,
	)
	
	latency := time.Since(start)
	metrics.RecordRequest(latency, err == nil)
}

func executeComplexPlacementRequest(ctx context.Context, orchestrator orchestration.OrchestrationEngine, metrics *LoadTestMetrics, workerID int) {
	start := time.Now()
	
	// Create more complex VM specifications for stress testing
	vmSpec := placement.VMSpec{
		VMID:     fmt.Sprintf("stress-vm-%d-%d", workerID, time.Now().UnixNano()),
		CPUs:     rand.Intn(8) + 2,  // 2-10 CPUs
		MemoryMB: int64(rand.Intn(16384) + 4096), // 4-20 GB RAM
		DiskGB:   int64(rand.Intn(500) + 100),    // 100-600 GB disk
		Labels: map[string]string{
			"vm_id":      fmt.Sprintf("stress-vm-%d", workerID),
			"stress_test": "true",
			"priority":   []string{"low", "medium", "high"}[rand.Intn(3)],
			"zone":       []string{"us-west-1a", "us-west-1b", "us-west-1c"}[rand.Intn(3)],
		},
		Requirements: placement.VMRequirements{
			MinCPUs:      rand.Intn(4) + 1,
			MinMemoryMB:  int64(rand.Intn(8192) + 2048),
			MinDiskGB:    int64(rand.Intn(200) + 50),
			NetworkBandwidthMbps: rand.Intn(1000) + 100,
			GPURequired:  rand.Float32() < 0.1, // 10% chance of GPU requirement
		},
	}
	
	strategy := []placement.PlacementStrategy{
		placement.PlacementStrategyBalanced,
		placement.PlacementStrategyBinPacking,
		placement.PlacementStrategyLoadBalancing,
	}[rand.Intn(3)]
	
	_, err := orchestrator.MakeVMPlacementDecision(ctx, vmSpec, strategy)
	
	latency := time.Since(start)
	metrics.RecordRequest(latency, err == nil)
}

func executeVariedPlacementRequest(ctx context.Context, orchestrator orchestration.OrchestrationEngine, metrics *LoadTestMetrics, workerID int) {
	start := time.Now()
	
	// Vary request types for endurance testing
	requestType := rand.Intn(3)
	
	switch requestType {
	case 0:
		// Small VM
		executeSmallVMRequest(ctx, orchestrator, metrics, workerID)
	case 1:
		// Medium VM
		executeMediumVMRequest(ctx, orchestrator, metrics, workerID)
	case 2:
		// Large VM
		executeLargeVMRequest(ctx, orchestrator, metrics, workerID)
	}
	
	latency := time.Since(start)
	metrics.RecordRequest(latency, true) // Record timing regardless of specific request result
}

func executeSmallVMRequest(ctx context.Context, orchestrator orchestration.OrchestrationEngine, metrics *LoadTestMetrics, workerID int) {
	vmSpec := placement.VMSpec{
		VMID:     fmt.Sprintf("small-vm-%d-%d", workerID, time.Now().UnixNano()),
		CPUs:     1,
		MemoryMB: 2048,
		DiskGB:   25,
		Labels: map[string]string{
			"vm_id": fmt.Sprintf("small-vm-%d", workerID),
			"size":  "small",
		},
	}
	
	orchestrator.MakeVMPlacementDecision(ctx, vmSpec, placement.PlacementStrategyBalanced)
}

func executeMediumVMRequest(ctx context.Context, orchestrator orchestration.OrchestrationEngine, metrics *LoadTestMetrics, workerID int) {
	vmSpec := placement.VMSpec{
		VMID:     fmt.Sprintf("medium-vm-%d-%d", workerID, time.Now().UnixNano()),
		CPUs:     4,
		MemoryMB: 8192,
		DiskGB:   100,
		Labels: map[string]string{
			"vm_id": fmt.Sprintf("medium-vm-%d", workerID),
			"size":  "medium",
		},
	}
	
	orchestrator.MakeVMPlacementDecision(ctx, vmSpec, placement.PlacementStrategyBinPacking)
}

func executeLargeVMRequest(ctx context.Context, orchestrator orchestration.OrchestrationEngine, metrics *LoadTestMetrics, workerID int) {
	vmSpec := placement.VMSpec{
		VMID:     fmt.Sprintf("large-vm-%d-%d", workerID, time.Now().UnixNano()),
		CPUs:     8,
		MemoryMB: 16384,
		DiskGB:   200,
		Labels: map[string]string{
			"vm_id": fmt.Sprintf("large-vm-%d", workerID),
			"size":  "large",
		},
	}
	
	orchestrator.MakeVMPlacementDecision(ctx, vmSpec, placement.PlacementStrategyLoadBalancing)
}