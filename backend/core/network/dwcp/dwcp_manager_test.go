package dwcp_test

import (
	"context"
	"runtime"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"go.uber.org/zap/zaptest"
)

// setupTestManager creates a test manager with default configuration
func setupTestManager(t *testing.T) *dwcp.Manager {
	config := dwcp.DefaultConfig()
	config.Enabled = true
	logger := zaptest.NewLogger(t)

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)

	return manager
}

// setupDisabledManager creates a test manager with DWCP disabled
func setupDisabledManager(t *testing.T) *dwcp.Manager {
	config := dwcp.DefaultConfig()
	config.Enabled = false
	logger := zaptest.NewLogger(t)

	manager, err := dwcp.NewManager(config, logger)
	require.NoError(t, err)
	require.NotNil(t, manager)

	return manager
}

// TestCollectMetricsNoRace tests concurrent metrics collection without race conditions
func TestCollectMetricsNoRace(t *testing.T) {
	// Run with: go test -race
	manager := setupTestManager(t)

	err := manager.Start()
	require.NoError(t, err)
	defer func() {
		err := manager.Stop()
		assert.NoError(t, err)
	}()

	var wg sync.WaitGroup
	iterations := 100

	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			metrics := manager.GetMetrics()
			assert.NotNil(t, metrics)
			assert.Equal(t, dwcp.DWCPVersion, metrics.Version)
		}()
	}

	wg.Wait()
}

// TestGetConfigHeapAllocation tests memory allocation for GetConfig
func TestGetConfigHeapAllocation(t *testing.T) {
	manager := setupTestManager(t)

	// First call to warm up
	config := manager.GetConfig()
	require.NotNil(t, config)

	// Measure allocations
	var m1, m2 runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&m1)

	for i := 0; i < 1000; i++ {
		_ = manager.GetConfig()
	}

	runtime.GC()
	runtime.ReadMemStats(&m2)

	// Allocation should be reasonable (less than 1MB for 1000 calls)
	allocatedBytes := m2.TotalAlloc - m1.TotalAlloc
	assert.Less(t, allocatedBytes, uint64(1024*1024), "Excessive heap allocation detected")
}

// TestValidateDisabledConfig tests configuration validation with disabled components
func TestValidateDisabledConfig(t *testing.T) {
	tests := []struct {
		name      string
		modifyFn  func(*dwcp.Config)
		expectErr bool
	}{
		{
			name: "disabled with valid config",
			modifyFn: func(c *dwcp.Config) {
				c.Enabled = false
				c.Transport.MinStreams = 16
				c.Transport.MaxStreams = 256
			},
			expectErr: false,
		},
		{
			name: "disabled with invalid min_streams",
			modifyFn: func(c *dwcp.Config) {
				c.Enabled = false
				c.Transport.MinStreams = 0
			},
			expectErr: true,
		},
		{
			name: "disabled with invalid max_streams",
			modifyFn: func(c *dwcp.Config) {
				c.Enabled = false
				c.Transport.MinStreams = 100
				c.Transport.MaxStreams = 50
			},
			expectErr: true,
		},
		{
			name: "disabled with invalid initial_streams",
			modifyFn: func(c *dwcp.Config) {
				c.Enabled = false
				c.Transport.MinStreams = 10
				c.Transport.MaxStreams = 100
				c.Transport.InitialStreams = 5 // Less than MinStreams
			},
			expectErr: true,
		},
		{
			name: "disabled with invalid max_delta_chain",
			modifyFn: func(c *dwcp.Config) {
				c.Enabled = false
				c.Compression.MaxDeltaChain = 0
			},
			expectErr: true,
		},
		{
			name: "enabled with valid config",
			modifyFn: func(c *dwcp.Config) {
				c.Enabled = true
			},
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := dwcp.DefaultConfig()
			tt.modifyFn(config)

			err := config.Validate()
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestComponentLifecycle tests the initialization and shutdown lifecycle
func TestComponentLifecycle(t *testing.T) {
	tests := []struct {
		name     string
		enabled  bool
		testFunc func(*testing.T, *dwcp.Manager)
	}{
		{
			name:    "enabled manager lifecycle",
			enabled: true,
			testFunc: func(t *testing.T, m *dwcp.Manager) {
				// Verify initial state
				assert.True(t, m.IsEnabled())
				assert.False(t, m.IsStarted())

				// Start manager
				err := m.Start()
				require.NoError(t, err)
				assert.True(t, m.IsStarted())

				// Cannot start twice
				err = m.Start()
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "already started")

				// Stop manager
				err = m.Stop()
				require.NoError(t, err)
				assert.False(t, m.IsStarted())

				// Can stop multiple times (idempotent)
				err = m.Stop()
				assert.NoError(t, err)
			},
		},
		{
			name:    "disabled manager lifecycle",
			enabled: false,
			testFunc: func(t *testing.T, m *dwcp.Manager) {
				// Verify initial state
				assert.False(t, m.IsEnabled())
				assert.False(t, m.IsStarted())

				// Start should succeed but not actually start components
				err := m.Start()
				require.NoError(t, err)

				// Stop should be idempotent
				err = m.Stop()
				assert.NoError(t, err)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := dwcp.DefaultConfig()
			config.Enabled = tt.enabled
			logger := zaptest.NewLogger(t)

			manager, err := dwcp.NewManager(config, logger)
			require.NoError(t, err)

			tt.testFunc(t, manager)
		})
	}
}

// TestCircuitBreakerStates tests circuit breaker state transitions
func TestCircuitBreakerStates(t *testing.T) {
	maxFailures := 5
	resetTimeout := 100 * time.Millisecond
	cb := dwcp.NewCircuitBreaker(maxFailures, resetTimeout)

	// Initial state should be closed
	assert.Equal(t, dwcp.CircuitClosed, cb.GetState())
	assert.True(t, cb.AllowRequest())

	// Record failures until circuit opens
	for i := 0; i < maxFailures; i++ {
		cb.RecordFailure()
	}

	// Circuit should now be open
	assert.Equal(t, dwcp.CircuitOpen, cb.GetState())
	assert.False(t, cb.AllowRequest())

	// Wait for reset timeout
	time.Sleep(resetTimeout + 50*time.Millisecond)

	// Circuit should transition to half-open
	assert.True(t, cb.AllowRequest())
	assert.Equal(t, dwcp.CircuitHalfOpen, cb.GetState())

	// Record success to close circuit
	cb.RecordSuccess()
	assert.Equal(t, dwcp.CircuitClosed, cb.GetState())
	assert.True(t, cb.AllowRequest())
}

// TestCircuitBreakerCall tests the Call wrapper method
func TestCircuitBreakerCall(t *testing.T) {
	cb := dwcp.NewCircuitBreaker(3, 100*time.Millisecond)

	// Successful calls
	for i := 0; i < 5; i++ {
		err := cb.Call(func() error {
			return nil
		})
		assert.NoError(t, err)
		assert.Equal(t, dwcp.CircuitClosed, cb.GetState())
	}

	// Fail enough times to open circuit
	for i := 0; i < 3; i++ {
		err := cb.Call(func() error {
			return assert.AnError
		})
		assert.Error(t, err)
	}
	assert.Equal(t, dwcp.CircuitOpen, cb.GetState())

	// Circuit is open, calls should be rejected
	err := cb.Call(func() error {
		t.Fatal("Should not be called when circuit is open")
		return nil
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "circuit breaker is open")
}

// TestHealthMonitoring tests health check and recovery
func TestHealthMonitoring(t *testing.T) {
	tests := []struct {
		name      string
		enabled   bool
		started   bool
		expectErr bool
	}{
		{
			name:      "disabled manager is healthy",
			enabled:   false,
			started:   false,
			expectErr: false,
		},
		{
			name:      "enabled but not started is unhealthy",
			enabled:   true,
			started:   false,
			expectErr: true,
		},
		{
			name:      "enabled and started is healthy",
			enabled:   true,
			started:   true,
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := dwcp.DefaultConfig()
			config.Enabled = tt.enabled
			logger := zaptest.NewLogger(t)

			manager, err := dwcp.NewManager(config, logger)
			require.NoError(t, err)

			if tt.started {
				err = manager.Start()
				require.NoError(t, err)
				defer manager.Stop()
			}

			err = manager.HealthCheck()
			if tt.expectErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestConcurrentOperations tests concurrent Start/Stop/GetConfig operations
func TestConcurrentOperations(t *testing.T) {
	manager := setupTestManager(t)

	var wg sync.WaitGroup
	errors := make(chan error, 100)

	// Start goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		if err := manager.Start(); err != nil {
			errors <- err
		}
	}()

	// Multiple GetConfig goroutines
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(time.Millisecond * time.Duration(i))
			config := manager.GetConfig()
			if config == nil {
				errors <- assert.AnError
			}
		}()
	}

	// Multiple GetMetrics goroutines
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(time.Millisecond * time.Duration(i))
			metrics := manager.GetMetrics()
			if metrics == nil {
				errors <- assert.AnError
			}
		}()
	}

	// Multiple IsEnabled/IsStarted goroutines
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			time.Sleep(time.Millisecond * time.Duration(i))
			_ = manager.IsEnabled()
			_ = manager.IsStarted()
		}()
	}

	wg.Wait()
	close(errors)

	// Check for errors
	for err := range errors {
		t.Error(err)
	}

	// Clean stop
	err := manager.Stop()
	assert.NoError(t, err)
}

// TestErrorRecovery tests component failure and recovery scenarios
func TestErrorRecovery(t *testing.T) {
	tests := []struct {
		name     string
		setupFn  func(*dwcp.Config)
		testFunc func(*testing.T, *dwcp.Manager)
	}{
		{
			name: "invalid config rejected",
			setupFn: func(c *dwcp.Config) {
				c.Transport.MinStreams = 0 // Invalid
			},
			testFunc: func(t *testing.T, m *dwcp.Manager) {
				// Manager creation should fail with invalid config
				assert.Nil(t, m)
			},
		},
		{
			name: "update config while running fails",
			setupFn: func(c *dwcp.Config) {
				c.Enabled = true
			},
			testFunc: func(t *testing.T, m *dwcp.Manager) {
				err := m.Start()
				require.NoError(t, err)
				defer m.Stop()

				newConfig := dwcp.DefaultConfig()
				err = m.UpdateConfig(newConfig)
				assert.Error(t, err)
				assert.Contains(t, err.Error(), "while DWCP is running")
			},
		},
		{
			name: "update config when stopped succeeds",
			setupFn: func(c *dwcp.Config) {
				c.Enabled = true
			},
			testFunc: func(t *testing.T, m *dwcp.Manager) {
				newConfig := dwcp.DefaultConfig()
				newConfig.Transport.MinStreams = 8

				err := m.UpdateConfig(newConfig)
				assert.NoError(t, err)

				updatedConfig := m.GetConfig()
				assert.Equal(t, 8, updatedConfig.Transport.MinStreams)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			config := dwcp.DefaultConfig()
			if tt.setupFn != nil {
				tt.setupFn(config)
			}

			logger := zaptest.NewLogger(t)
			manager, err := dwcp.NewManager(config, logger)

			if err == nil {
				tt.testFunc(t, manager)
			} else {
				tt.testFunc(t, nil)
			}
		})
	}
}

// TestManagerCreation tests manager creation with various configurations
func TestManagerCreation(t *testing.T) {
	tests := []struct {
		name      string
		config    *dwcp.Config
		logger    *zap.Logger
		expectErr bool
	}{
		{
			name:      "nil config uses default",
			config:    nil,
			logger:    zaptest.NewLogger(t),
			expectErr: false,
		},
		{
			name:      "nil logger creates production logger",
			config:    dwcp.DefaultConfig(),
			logger:    nil,
			expectErr: false,
		},
		{
			name: "invalid config fails",
			config: func() *dwcp.Config {
				c := dwcp.DefaultConfig()
				c.Transport.MinStreams = -1
				return c
			}(),
			logger:    zaptest.NewLogger(t),
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			manager, err := dwcp.NewManager(tt.config, tt.logger)

			if tt.expectErr {
				assert.Error(t, err)
				assert.Nil(t, manager)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, manager)
			}
		})
	}
}

// TestGetTransport tests transport layer retrieval
func TestGetTransport(t *testing.T) {
	manager := setupTestManager(t)

	// Before start, transport should be nil
	transport := manager.GetTransport()
	assert.Nil(t, transport)

	// After start, transport should be initialized
	err := manager.Start()
	require.NoError(t, err)
	defer manager.Stop()

	transport = manager.GetTransport()
	assert.NotNil(t, transport)
}

// TestMetricsCollection tests metrics collection over time
func TestMetricsCollection(t *testing.T) {
	manager := setupTestManager(t)

	err := manager.Start()
	require.NoError(t, err)
	defer manager.Stop()

	// Wait for at least one metrics collection cycle
	time.Sleep(6 * time.Second)

	metrics := manager.GetMetrics()
	assert.NotNil(t, metrics)
	assert.Equal(t, dwcp.DWCPVersion, metrics.Version)
	assert.True(t, metrics.Enabled)
}

// TestCircuitStateString tests string representation of circuit states
func TestCircuitStateString(t *testing.T) {
	tests := []struct {
		state    dwcp.CircuitState
		expected string
	}{
		{dwcp.CircuitClosed, "closed"},
		{dwcp.CircuitOpen, "open"},
		{dwcp.CircuitHalfOpen, "half-open"},
		{dwcp.CircuitState(99), "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.state.String())
		})
	}
}

// TestConfigValidationEdgeCases tests edge cases in config validation
func TestConfigValidationEdgeCases(t *testing.T) {
	tests := []struct {
		name      string
		config    *dwcp.Config
		expectErr bool
		errMsg    string
	}{
		{
			name: "min_streams exactly 1",
			config: func() *dwcp.Config {
				c := dwcp.DefaultConfig()
				c.Transport.MinStreams = 1
				c.Transport.MaxStreams = 10
				c.Transport.InitialStreams = 5
				return c
			}(),
			expectErr: false,
		},
		{
			name: "max_streams equals min_streams",
			config: func() *dwcp.Config {
				c := dwcp.DefaultConfig()
				c.Transport.MinStreams = 10
				c.Transport.MaxStreams = 10
				c.Transport.InitialStreams = 10
				return c
			}(),
			expectErr: false,
		},
		{
			name: "max_delta_chain exactly 1",
			config: func() *dwcp.Config {
				c := dwcp.DefaultConfig()
				c.Compression.MaxDeltaChain = 1
				return c
			}(),
			expectErr: false,
		},
		{
			name: "initial_streams at min boundary",
			config: func() *dwcp.Config {
				c := dwcp.DefaultConfig()
				c.Transport.MinStreams = 10
				c.Transport.MaxStreams = 100
				c.Transport.InitialStreams = 10
				return c
			}(),
			expectErr: false,
		},
		{
			name: "initial_streams at max boundary",
			config: func() *dwcp.Config {
				c := dwcp.DefaultConfig()
				c.Transport.MinStreams = 10
				c.Transport.MaxStreams = 100
				c.Transport.InitialStreams = 100
				return c
			}(),
			expectErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()

			if tt.expectErr {
				assert.Error(t, err)
				if tt.errMsg != "" {
					assert.Contains(t, err.Error(), tt.errMsg)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestPartitionTask covers both fallback behavior and configured partitioner path.
func TestPartitionTask(t *testing.T) {
	t.Run("not started returns error", func(t *testing.T) {
		manager := setupTestManager(t)
		// Do not start manager

		task := &dwcp.Task{
			ID:       "not-started-task",
			Size:     1024,
			Priority: 0.5,
		}

		decision, err := manager.PartitionTask(context.Background(), task)
		assert.Error(t, err)
		assert.Nil(t, decision)
	})

	t.Run("fallback without partitioner", func(t *testing.T) {
		manager := setupTestManager(t)

		err := manager.Start()
		require.NoError(t, err)
		defer func() {
			_ = manager.Stop()
		}()

		taskSize := 2 * 1024 * 1024 // 2MB
		task := &dwcp.Task{
			ID:       "fallback-task",
			Size:     taskSize,
			Priority: 0.5,
		}

		decision, err := manager.PartitionTask(context.Background(), task)
		require.NoError(t, err)
		require.NotNil(t, decision)

		// Default fallback is a single stream, full task size, fixed confidence/time
		assert.Equal(t, []int{0}, decision.StreamIDs)
		assert.Equal(t, []int{taskSize}, decision.ChunkSizes)
		assert.Equal(t, 0.5, decision.Confidence)
		assert.Equal(t, time.Second, decision.ExpectedTime)
	})

	t.Run("uses partitioner when configured", func(t *testing.T) {
		manager := setupTestManager(t)

		err := manager.Start()
		require.NoError(t, err)
		defer func() {
			_ = manager.Stop()
		}()

		// Add task partitioner; in environments without ONNX/runtime this
		// will operate in heuristic-only mode but must not panic.
		err = manager.AddTaskPartitioner("backend/core/network/dwcp/partition/models/dqn_final.onnx")
		require.NoError(t, err)

		taskSize := 10 * 1024 * 1024 // 10MB
		task := &dwcp.Task{
			ID:       "partitioned-task",
			Size:     taskSize,
			Priority: 0.8,
		}

		decision, err := manager.PartitionTask(context.Background(), task)
		require.NoError(t, err)
		require.NotNil(t, decision)

		require.NotEmpty(t, decision.StreamIDs)
		require.NotEmpty(t, decision.ChunkSizes)
		assert.Equal(t, len(decision.StreamIDs), len(decision.ChunkSizes))

		total := 0
		for _, sz := range decision.ChunkSizes {
			assert.Greater(t, sz, 0)
			total += sz
		}
		assert.Equal(t, taskSize, total)
	})
}

// TestConcurrentMetricsCollection tests concurrent metrics collection for race conditions
func TestConcurrentMetricsCollection(t *testing.T) {
	// This test is designed to be run with: go test -race -count=100
	manager := setupTestManager(t)

	err := manager.Start()
	require.NoError(t, err)
	defer func() {
		err := manager.Stop()
		assert.NoError(t, err)
	}()

	const numGoroutines = 50
	const numIterations = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start multiple goroutines that concurrently access metrics
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				// Concurrent metrics collection
				metrics := manager.GetMetrics()
				assert.NotNil(t, metrics)

				// Brief pause to increase chance of race conditions
				runtime.Gosched()
			}
		}()
	}

	wg.Wait()
}

// TestConcurrentComponentAccess tests concurrent component access for race conditions
func TestConcurrentComponentAccess(t *testing.T) {
	// This test is designed to be run with: go test -race -count=100
	manager := setupTestManager(t)

	err := manager.Start()
	require.NoError(t, err)
	defer func() {
		err := manager.Stop()
		assert.NoError(t, err)
	}()

	const numGoroutines = 30
	const numIterations = 50

	var wg sync.WaitGroup
	wg.Add(numGoroutines * 3) // 3 different access patterns

	// Concurrent metrics access
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				metrics := manager.GetMetrics()
				assert.NotNil(t, metrics)
				runtime.Gosched()
			}
		}()
	}

	// Concurrent config access
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				config := manager.GetConfig()
				assert.NotNil(t, config)
				runtime.Gosched()
			}
		}()
	}

	// Concurrent status checks
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < numIterations; j++ {
				_ = manager.IsEnabled()
				_ = manager.IsRunning()
				_ = manager.IsStarted()
				runtime.Gosched()
			}
		}()
	}

	wg.Wait()
}

// TestConcurrentStartStop tests concurrent Start/Stop operations for race conditions
func TestConcurrentStartStop(t *testing.T) {
	// This test is designed to be run with: go test -race -count=100
	const numCycles = 10

	for cycle := 0; cycle < numCycles; cycle++ {
		manager := setupTestManager(t)

		var wg sync.WaitGroup
		wg.Add(2)

		// Goroutine 1: Start the manager
		go func() {
			defer wg.Done()
			err := manager.Start()
			// Start might fail if Stop is called concurrently, which is expected
			if err != nil {
				t.Logf("Start failed (expected in concurrent test): %v", err)
			}
		}()

		// Goroutine 2: Stop the manager after a brief delay
		go func() {
			defer wg.Done()
			time.Sleep(1 * time.Millisecond) // Brief delay to let Start begin
			err := manager.Stop()
			// Stop should always succeed
			assert.NoError(t, err)
		}()

		wg.Wait()

		// Ensure final stop
		_ = manager.Stop()
	}
}

// TestConcurrentHealthMonitoring tests concurrent health monitoring for race conditions
func TestConcurrentHealthMonitoring(t *testing.T) {
	// This test is designed to be run with: go test -race -count=100
	manager := setupTestManager(t)

	err := manager.Start()
	require.NoError(t, err)
	defer func() {
		err := manager.Stop()
		assert.NoError(t, err)
	}()

	const numGoroutines = 20
	const testDuration = 100 * time.Millisecond

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Start multiple goroutines that access manager state during health monitoring
	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			start := time.Now()
			for time.Since(start) < testDuration {
				// Access various manager methods that might race with health monitoring
				_ = manager.IsRunning()
				_ = manager.GetMetrics()
				_ = manager.GetConfig()
				runtime.Gosched()
			}
		}()
	}

	wg.Wait()
}

// TestRaceDetectorStress runs intensive concurrent operations to detect race conditions
func TestRaceDetectorStress(t *testing.T) {
	// This test is designed to be run with: go test -race -count=100
	if testing.Short() {
		t.Skip("Skipping stress test in short mode")
	}

	manager := setupTestManager(t)

	err := manager.Start()
	require.NoError(t, err)
	defer func() {
		err := manager.Stop()
		assert.NoError(t, err)
	}()

	const numGoroutines = 100
	const testDuration = 200 * time.Millisecond

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	// Intensive concurrent access to all manager methods
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			start := time.Now()
			iteration := 0
			for time.Since(start) < testDuration {
				switch iteration % 4 {
				case 0:
					metrics := manager.GetMetrics()
					assert.NotNil(t, metrics)
				case 1:
					config := manager.GetConfig()
					assert.NotNil(t, config)
				case 2:
					_ = manager.IsEnabled()
					_ = manager.IsRunning()
				case 3:
					_ = manager.IsStarted()
				}
				iteration++
				runtime.Gosched()
			}
		}(i)
	}

	wg.Wait()
}
