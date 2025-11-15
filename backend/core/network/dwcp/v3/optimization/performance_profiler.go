// Package optimization provides comprehensive performance profiling and analysis for DWCP v3.
package optimization

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sort"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// ProfilerConfig defines configuration for performance profiling.
type ProfilerConfig struct {
	// Profiling intervals
	CPUProfileInterval    time.Duration
	MemoryProfileInterval time.Duration
	GoroutineInterval     time.Duration

	// Storage settings
	ProfileOutputDir string
	RetentionPeriod  time.Duration

	// Sampling rates
	CPUSampleRate    int // Hz
	MemorySampleRate int // per allocation

	// Feature flags
	EnableCPUProfile       bool
	EnableMemoryProfile    bool
	EnableGoroutineProfile bool
	EnableBlockProfile     bool
	EnableMutexProfile     bool
	EnableTracing          bool
}

// DefaultProfilerConfig returns default profiler configuration.
func DefaultProfilerConfig() *ProfilerConfig {
	return &ProfilerConfig{
		CPUProfileInterval:     1 * time.Minute,
		MemoryProfileInterval:  5 * time.Minute,
		GoroutineInterval:      30 * time.Second,
		ProfileOutputDir:       "./profiles",
		RetentionPeriod:        24 * time.Hour,
		CPUSampleRate:          100,
		MemorySampleRate:       1,
		EnableCPUProfile:       true,
		EnableMemoryProfile:    true,
		EnableGoroutineProfile: true,
		EnableBlockProfile:     false,
		EnableMutexProfile:     false,
		EnableTracing:          false,
	}
}

// ComponentMetrics tracks performance metrics for a DWCP component.
type ComponentMetrics struct {
	Name string

	// CPU metrics
	CPUUsage   float64 // percentage
	CPUTime    time.Duration
	Goroutines int

	// Memory metrics
	AllocBytes uint64
	HeapBytes  uint64
	StackBytes uint64
	GCPauses   time.Duration

	// Network metrics
	BytesSent     uint64
	BytesReceived uint64
	PacketsSent   uint64
	PacketsRecvd  uint64

	// Latency metrics
	P50Latency time.Duration
	P95Latency time.Duration
	P99Latency time.Duration
	MaxLatency time.Duration

	// Throughput metrics
	Throughput float64 // ops/sec
	Bandwidth  float64 // bytes/sec

	// Error metrics
	Errors   uint64
	Warnings uint64

	Timestamp time.Time
}

// PerformanceProfiler provides comprehensive performance profiling for DWCP v3.
type PerformanceProfiler struct {
	config *ProfilerConfig
	mu     sync.RWMutex

	// Component tracking
	components map[string]*componentProfiler

	// Metrics collectors
	cpuCollector     prometheus.Collector
	memoryCollector  prometheus.Collector
	networkCollector prometheus.Collector
	latencyCollector prometheus.Collector

	// Profiling state
	cpuProfileActive bool
	cpuProfileFile   *os.File

	// Background tasks
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// componentProfiler tracks profiling for a specific component.
type componentProfiler struct {
	name    string
	metrics *ComponentMetrics
	mu      sync.RWMutex

	// Histogram for latency tracking
	latencies    []time.Duration
	maxLatencies int

	// Counters
	operations uint64
	bytes      uint64
	errors     uint64
}

var (
	// Prometheus metrics
	cpuUsageGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_v3_cpu_usage_percent",
			Help: "CPU usage percentage by component",
		},
		[]string{"component", "mode"},
	)

	memoryUsageGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_v3_memory_usage_bytes",
			Help: "Memory usage in bytes by component",
		},
		[]string{"component", "type"},
	)

	goroutineCountGauge = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "dwcp_v3_goroutines_total",
			Help: "Total number of goroutines",
		},
	)

	latencyHistogram = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "dwcp_v3_operation_latency_seconds",
			Help:    "Operation latency in seconds",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15), // 1ms to 16s
		},
		[]string{"component", "operation"},
	)

	throughputGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_v3_throughput_ops_per_second",
			Help: "Operations per second by component",
		},
		[]string{"component"},
	)

	bandwidthGauge = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "dwcp_v3_bandwidth_bytes_per_second",
			Help: "Bandwidth in bytes per second by component",
		},
		[]string{"component", "direction"},
	)
)

// NewPerformanceProfiler creates a new performance profiler.
func NewPerformanceProfiler(config *ProfilerConfig) (*PerformanceProfiler, error) {
	if config == nil {
		config = DefaultProfilerConfig()
	}

	ctx, cancel := context.WithCancel(context.Background())

	p := &PerformanceProfiler{
		config:     config,
		components: make(map[string]*componentProfiler),
		ctx:        ctx,
		cancel:     cancel,
	}

	// Create profile output directory
	if err := os.MkdirAll(config.ProfileOutputDir, 0755); err != nil {
		cancel()
		return nil, fmt.Errorf("create profile dir: %w", err)
	}

	// Configure runtime profiling
	if config.EnableBlockProfile {
		runtime.SetBlockProfileRate(1)
	}
	if config.EnableMutexProfile {
		runtime.SetMutexProfileFraction(1)
	}

	// Start background profiling
	p.wg.Add(1)
	go p.runContinuousProfiling()

	return p, nil
}

// RegisterComponent registers a component for profiling.
func (p *PerformanceProfiler) RegisterComponent(name string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if _, exists := p.components[name]; !exists {
		p.components[name] = &componentProfiler{
			name: name,
			metrics: &ComponentMetrics{
				Name:      name,
				Timestamp: time.Now(),
			},
			latencies:    make([]time.Duration, 0, 1000),
			maxLatencies: 10000,
		}
	}
}

// RecordOperation records an operation for profiling.
func (p *PerformanceProfiler) RecordOperation(component string, operation string, latency time.Duration, bytes uint64, err error) {
	p.mu.RLock()
	comp, exists := p.components[component]
	p.mu.RUnlock()

	if !exists {
		p.RegisterComponent(component)
		p.mu.RLock()
		comp = p.components[component]
		p.mu.RUnlock()
	}

	comp.mu.Lock()
	defer comp.mu.Unlock()

	// Update counters
	comp.operations++
	comp.bytes += bytes
	if err != nil {
		comp.errors++
	}

	// Track latency
	if len(comp.latencies) < comp.maxLatencies {
		comp.latencies = append(comp.latencies, latency)
	} else {
		// Replace oldest (simple circular buffer)
		comp.latencies[comp.operations%uint64(comp.maxLatencies)] = latency
	}

	// Update Prometheus metrics
	latencyHistogram.WithLabelValues(component, operation).Observe(latency.Seconds())
}

// GetComponentMetrics returns current metrics for a component.
func (p *PerformanceProfiler) GetComponentMetrics(component string) (*ComponentMetrics, error) {
	p.mu.RLock()
	comp, exists := p.components[component]
	p.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("component not registered: %s", component)
	}

	comp.mu.RLock()
	defer comp.mu.RUnlock()

	metrics := &ComponentMetrics{
		Name:      comp.name,
		Timestamp: time.Now(),
	}

	// Calculate latency percentiles
	if len(comp.latencies) > 0 {
		sorted := make([]time.Duration, len(comp.latencies))
		copy(sorted, comp.latencies)
		sort.Slice(sorted, func(i, j int) bool {
			return sorted[i] < sorted[j]
		})

		metrics.P50Latency = sorted[len(sorted)*50/100]
		metrics.P95Latency = sorted[len(sorted)*95/100]
		metrics.P99Latency = sorted[len(sorted)*99/100]
		metrics.MaxLatency = sorted[len(sorted)-1]
	}

	// Calculate throughput
	elapsed := time.Since(comp.metrics.Timestamp)
	if elapsed > 0 {
		metrics.Throughput = float64(comp.operations) / elapsed.Seconds()
		metrics.Bandwidth = float64(comp.bytes) / elapsed.Seconds()
	}

	metrics.Errors = comp.errors

	return metrics, nil
}

// GetAllMetrics returns metrics for all registered components.
func (p *PerformanceProfiler) GetAllMetrics() map[string]*ComponentMetrics {
	p.mu.RLock()
	defer p.mu.RUnlock()

	result := make(map[string]*ComponentMetrics)
	for name := range p.components {
		if metrics, err := p.GetComponentMetrics(name); err == nil {
			result[name] = metrics
		}
	}

	return result
}

// StartCPUProfile starts CPU profiling.
func (p *PerformanceProfiler) StartCPUProfile() error {
	if !p.config.EnableCPUProfile {
		return fmt.Errorf("CPU profiling disabled")
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	if p.cpuProfileActive {
		return fmt.Errorf("CPU profiling already active")
	}

	filename := filepath.Join(p.config.ProfileOutputDir,
		fmt.Sprintf("cpu-%s.prof", time.Now().Format("20060102-150405")))

	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("create CPU profile: %w", err)
	}

	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		return fmt.Errorf("start CPU profile: %w", err)
	}

	p.cpuProfileFile = f
	p.cpuProfileActive = true

	return nil
}

// StopCPUProfile stops CPU profiling.
func (p *PerformanceProfiler) StopCPUProfile() error {
	p.mu.Lock()
	defer p.mu.Unlock()

	if !p.cpuProfileActive {
		return fmt.Errorf("CPU profiling not active")
	}

	pprof.StopCPUProfile()
	p.cpuProfileFile.Close()
	p.cpuProfileActive = false
	p.cpuProfileFile = nil

	return nil
}

// WriteMemoryProfile writes a memory profile.
func (p *PerformanceProfiler) WriteMemoryProfile() error {
	if !p.config.EnableMemoryProfile {
		return fmt.Errorf("memory profiling disabled")
	}

	filename := filepath.Join(p.config.ProfileOutputDir,
		fmt.Sprintf("memory-%s.prof", time.Now().Format("20060102-150405")))

	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("create memory profile: %w", err)
	}
	defer f.Close()

	runtime.GC() // Get up-to-date statistics
	if err := pprof.WriteHeapProfile(f); err != nil {
		return fmt.Errorf("write memory profile: %w", err)
	}

	return nil
}

// WriteGoroutineProfile writes a goroutine profile.
func (p *PerformanceProfiler) WriteGoroutineProfile() error {
	if !p.config.EnableGoroutineProfile {
		return fmt.Errorf("goroutine profiling disabled")
	}

	filename := filepath.Join(p.config.ProfileOutputDir,
		fmt.Sprintf("goroutine-%s.prof", time.Now().Format("20060102-150405")))

	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("create goroutine profile: %w", err)
	}
	defer f.Close()

	profile := pprof.Lookup("goroutine")
	if profile == nil {
		return fmt.Errorf("goroutine profile not available")
	}

	if err := profile.WriteTo(f, 0); err != nil {
		return fmt.Errorf("write goroutine profile: %w", err)
	}

	return nil
}

// runContinuousProfiling runs continuous background profiling.
func (p *PerformanceProfiler) runContinuousProfiling() {
	defer p.wg.Done()

	cpuTicker := time.NewTicker(p.config.CPUProfileInterval)
	defer cpuTicker.Stop()

	memoryTicker := time.NewTicker(p.config.MemoryProfileInterval)
	defer memoryTicker.Stop()

	goroutineTicker := time.NewTicker(p.config.GoroutineInterval)
	defer goroutineTicker.Stop()

	metricsTicker := time.NewTicker(1 * time.Second)
	defer metricsTicker.Stop()

	for {
		select {
		case <-p.ctx.Done():
			return

		case <-cpuTicker.C:
			if err := p.StartCPUProfile(); err == nil {
				time.Sleep(30 * time.Second) // Profile for 30 seconds
				p.StopCPUProfile()
			}

		case <-memoryTicker.C:
			p.WriteMemoryProfile()

		case <-goroutineTicker.C:
			p.WriteGoroutineProfile()

		case <-metricsTicker.C:
			p.updateSystemMetrics()
		}
	}
}

// updateSystemMetrics updates system-level metrics.
func (p *PerformanceProfiler) updateSystemMetrics() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	// Update memory metrics
	memoryUsageGauge.WithLabelValues("system", "alloc").Set(float64(m.Alloc))
	memoryUsageGauge.WithLabelValues("system", "heap").Set(float64(m.HeapAlloc))
	memoryUsageGauge.WithLabelValues("system", "stack").Set(float64(m.StackInuse))

	// Update goroutine count
	goroutineCountGauge.Set(float64(runtime.NumGoroutine()))

	// Update component metrics
	p.mu.RLock()
	defer p.mu.RUnlock()

	for name, _ := range p.components {
		if metrics, err := p.GetComponentMetrics(name); err == nil {
			throughputGauge.WithLabelValues(name).Set(metrics.Throughput)
			bandwidthGauge.WithLabelValues(name, "tx").Set(metrics.Bandwidth)
		}
	}
}

// GetProfileSummary returns a summary of profiling results.
func (p *PerformanceProfiler) GetProfileSummary() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	summary := map[string]interface{}{
		"timestamp":  time.Now(),
		"goroutines": runtime.NumGoroutine(),
		"memory": map[string]interface{}{
			"alloc_bytes":    m.Alloc,
			"heap_bytes":     m.HeapAlloc,
			"stack_bytes":    m.StackInuse,
			"gc_count":       m.NumGC,
			"gc_pause_total": time.Duration(m.PauseTotalNs),
			"gc_pause_last":  time.Duration(m.PauseNs[(m.NumGC+255)%256]),
		},
		"components": p.GetAllMetrics(),
	}

	return summary
}

// Close stops profiling and cleans up resources.
func (p *PerformanceProfiler) Close() error {
	p.cancel()
	p.wg.Wait()

	// Stop any active CPU profiling
	p.mu.Lock()
	if p.cpuProfileActive {
		pprof.StopCPUProfile()
		if p.cpuProfileFile != nil {
			p.cpuProfileFile.Close()
		}
	}
	p.mu.Unlock()

	return nil
}
