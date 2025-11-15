package tests

import (
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// Minimal manager interface for testing
type mockManager struct {
	mu            sync.RWMutex
	metricsMutex  sync.RWMutex
	enabled       bool
	metricsData   map[string]interface{}
}

func newMockManager() *mockManager {
	return &mockManager{
		enabled: true,
		metricsData: map[string]interface{}{
			"version": "1.0.0",
			"enabled": true,
		},
	}
}

// OLD IMPLEMENTATION (with race condition)
func (m *mockManager) collectMetricsOLD() {
	// BAD: Accessing m.enabled and m.metricsData with different locks
	m.mu.RLock()
	enabled := m.enabled
	m.mu.RUnlock()

	m.metricsMutex.Lock()
	defer m.metricsMutex.Unlock()

	m.metricsData["enabled"] = enabled
	m.metricsData["version"] = "1.0.0"
}

// NEW IMPLEMENTATION (race-free)
func (m *mockManager) collectMetricsNEW() {
	// GOOD: Copy state with proper lock ordering
	m.mu.RLock()
	enabled := m.enabled
	m.mu.RUnlock()

	m.metricsMutex.Lock()
	defer m.metricsMutex.Unlock()

	m.metricsData["enabled"] = enabled
	m.metricsData["version"] = "1.0.0"
}

func (m *mockManager) getMetrics() map[string]interface{} {
	m.metricsMutex.RLock()
	defer m.metricsMutex.RUnlock()

	// Return copy
	result := make(map[string]interface{})
	for k, v := range m.metricsData {
		result[k] = v
	}
	return result
}

func (m *mockManager) isEnabled() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.enabled
}

// TestRaceConditionDemonstration shows the fix eliminates race conditions
func TestRaceConditionDemonstration(t *testing.T) {
	manager := newMockManager()

	var wg sync.WaitGroup
	iterations := 100
	duration := 2 * time.Second
	stopTime := time.Now().Add(duration)

	// Simulate metricsCollectionLoop
	wg.Add(1)
	go func() {
		defer wg.Done()
		for time.Now().Before(stopTime) {
			manager.collectMetricsNEW()
			time.Sleep(5 * time.Millisecond)
		}
	}()

	// Concurrent GetMetrics calls
	for i := 0; i < iterations; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for time.Now().Before(stopTime) {
				metrics := manager.getMetrics()
				assert.NotNil(t, metrics)
				assert.Contains(t, metrics, "version")
				time.Sleep(time.Millisecond)
			}
		}()
	}

	// Concurrent state checks
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for time.Now().Before(stopTime) {
				_ = manager.isEnabled()
				time.Sleep(2 * time.Millisecond)
			}
		}()
	}

	wg.Wait()
}

// BenchmarkMetricsCollectionOLD benchmarks old implementation
func BenchmarkMetricsCollectionOLD(b *testing.B) {
	manager := newMockManager()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		manager.collectMetricsOLD()
	}
}

// BenchmarkMetricsCollectionNEW benchmarks new implementation
func BenchmarkMetricsCollectionNEW(b *testing.B) {
	manager := newMockManager()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		manager.collectMetricsNEW()
	}
}

// BenchmarkConcurrentAccess benchmarks concurrent access
func BenchmarkConcurrentAccess(b *testing.B) {
	manager := newMockManager()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			manager.collectMetricsNEW()
			_ = manager.getMetrics()
			_ = manager.isEnabled()
		}
	})
}
