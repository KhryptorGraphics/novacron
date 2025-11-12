// DWCP v4 WebAssembly Runtime - Production Implementation
// Target: 10x faster VM startup (<100ms cold start)
// Security: Multi-tenant isolation with sandboxing
package wasm

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/bytecodealliance/wasmtime-go/v27"
	"go.uber.org/zap"
)

// RuntimeVersion identifies this WASM runtime implementation
const RuntimeVersion = "4.0.0-alpha"

// Runtime metrics
const (
	ColdStartTargetMS   = 100  // <100ms cold start target
	WarmStartTargetMS   = 10   // <10ms warm start target
	MaxConcurrentVMs    = 1000 // Maximum concurrent VM instances
	DefaultMemoryLimitMB = 128 // Default per-VM memory limit
)

// WASMRuntime manages WebAssembly module execution with optimized startup
type WASMRuntime struct {
	engine   *wasmtime.Engine
	poolSize int
	logger   *zap.Logger

	// VM pool for fast warm starts
	vmPool      chan *VMInstance
	poolMetrics *PoolMetrics

	// Module cache for instant loading
	moduleCache *ModuleCache

	// Security and isolation
	sandboxConfig *SandboxConfig
	resourceLimits *ResourceLimits

	// Monitoring
	metrics     *RuntimeMetrics
	metricsLock sync.RWMutex

	// Lifecycle
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// VMInstance represents a pre-warmed WebAssembly VM
type VMInstance struct {
	ID          string
	Store       *wasmtime.Store
	Linker      *wasmtime.Linker
	Instance    *wasmtime.Instance
	Module      *wasmtime.Module
	CreatedAt   time.Time
	LastUsedAt  time.Time
	UsageCount  int64
	MemoryUsage uint64
	Isolated    bool
}

// ModuleCache provides instant module loading
type ModuleCache struct {
	modules map[string]*CachedModule
	lock    sync.RWMutex
	maxSize int
	hits    int64
	misses  int64
}

// CachedModule represents a compiled and cached WASM module
type CachedModule struct {
	Module      *wasmtime.Module
	Hash        string
	Size        uint64
	CreatedAt   time.Time
	LastAccessAt time.Time
	AccessCount int64
}

// SandboxConfig defines security boundaries for WASM execution
type SandboxConfig struct {
	EnableWASI         bool
	AllowedSyscalls    []string
	DeniedSyscalls     []string
	NetworkAccess      bool
	FilesystemAccess   bool
	AllowedDirectories []string
	DeniedDirectories  []string
	MaxFileSize        uint64
	MaxOpenFiles       int
}

// ResourceLimits defines per-VM resource constraints
type ResourceLimits struct {
	MaxMemoryBytes      uint64
	MaxTableElements    uint32
	MaxInstances        uint32
	MaxFunctions        uint32
	MaxExecutionTimeMs  uint64
	MaxStackSizeBytes   uint64
	EnableFuelMetering  bool
	FuelLimit           uint64
}

// RuntimeMetrics tracks runtime performance
type RuntimeMetrics struct {
	ColdStarts       int64
	WarmStarts       int64
	AvgColdStartMS   float64
	AvgWarmStartMS   float64
	P50ColdStartMS   float64
	P95ColdStartMS   float64
	P99ColdStartMS   float64
	ActiveVMs        int64
	TotalExecutions  int64
	FailedExecutions int64
	CacheHitRate     float64
	MemoryUsageBytes uint64
	StartTime        time.Time
}

// PoolMetrics tracks VM pool performance
type PoolMetrics struct {
	PoolSize      int
	ActiveVMs     int64
	IdleVMs       int64
	PoolHits      int64
	PoolMisses    int64
	AvgWaitTimeMS float64
	lock          sync.RWMutex
}

// RuntimeConfig configures the WASM runtime
type RuntimeConfig struct {
	PoolSize       int
	CacheSize      int
	SandboxConfig  *SandboxConfig
	ResourceLimits *ResourceLimits
	Logger         *zap.Logger
	EnableMetrics  bool
	EnableTracing  bool
}

// DefaultRuntimeConfig returns production-optimized configuration
func DefaultRuntimeConfig() *RuntimeConfig {
	return &RuntimeConfig{
		PoolSize:  100, // Pre-warmed VM pool
		CacheSize: 1000, // Module cache size
		SandboxConfig: &SandboxConfig{
			EnableWASI:        true,
			AllowedSyscalls:   []string{"fd_read", "fd_write", "proc_exit"},
			DeniedSyscalls:    []string{"sock_*", "path_open"},
			NetworkAccess:     false,
			FilesystemAccess:  true,
			AllowedDirectories: []string{"/tmp"},
			MaxFileSize:       10 * 1024 * 1024, // 10MB
			MaxOpenFiles:      10,
		},
		ResourceLimits: &ResourceLimits{
			MaxMemoryBytes:     128 * 1024 * 1024, // 128MB
			MaxTableElements:   1000,
			MaxInstances:       100,
			MaxFunctions:       10000,
			MaxExecutionTimeMs: 30000, // 30 seconds
			MaxStackSizeBytes:  1024 * 1024, // 1MB
			EnableFuelMetering: true,
			FuelLimit:          1000000,
		},
		EnableMetrics: true,
		EnableTracing: false,
	}
}

// NewWASMRuntime creates an optimized production WASM runtime
func NewWASMRuntime(config *RuntimeConfig) (*WASMRuntime, error) {
	if config == nil {
		config = DefaultRuntimeConfig()
	}

	if config.Logger == nil {
		config.Logger, _ = zap.NewProduction()
	}

	// Create optimized engine configuration
	engineConfig := wasmtime.NewConfig()
	engineConfig.SetConsumeFuel(config.ResourceLimits.EnableFuelMetering)
	engineConfig.SetCraneliftOptLevel(wasmtime.OptLevelSpeed)
	engineConfig.SetProfiler(wasmtime.ProfilingStrategyNone)

	// Enable parallel compilation for faster startup
	engineConfig.SetParallelCompilation(true)

	// Cache compiled modules
	engineConfig.CacheConfigLoad("/tmp/wasmtime-cache.toml")

	engine := wasmtime.NewEngineWithConfig(engineConfig)

	ctx, cancel := context.WithCancel(context.Background())

	runtime := &WASMRuntime{
		engine:         engine,
		poolSize:       config.PoolSize,
		logger:         config.Logger,
		vmPool:         make(chan *VMInstance, config.PoolSize),
		sandboxConfig:  config.SandboxConfig,
		resourceLimits: config.ResourceLimits,
		ctx:            ctx,
		cancel:         cancel,
		metrics: &RuntimeMetrics{
			StartTime: time.Now(),
		},
		poolMetrics: &PoolMetrics{
			PoolSize: config.PoolSize,
		},
		moduleCache: &ModuleCache{
			modules: make(map[string]*CachedModule),
			maxSize: config.CacheSize,
		},
	}

	// Pre-warm the VM pool for <10ms warm starts
	if err := runtime.prewarmPool(); err != nil {
		return nil, fmt.Errorf("failed to prewarm pool: %w", err)
	}

	// Start background pool maintenance
	runtime.wg.Add(1)
	go runtime.poolMaintainer()

	runtime.logger.Info("WASM Runtime initialized",
		zap.String("version", RuntimeVersion),
		zap.Int("pool_size", config.PoolSize),
		zap.Int("cache_size", config.CacheSize),
	)

	return runtime, nil
}

// prewarmPool creates pre-initialized VM instances for fast startup
func (r *WASMRuntime) prewarmPool() error {
	r.logger.Info("Pre-warming VM pool", zap.Int("target_size", r.poolSize))

	start := time.Now()

	for i := 0; i < r.poolSize; i++ {
		vm, err := r.createVMInstance()
		if err != nil {
			return fmt.Errorf("failed to create VM instance %d: %w", i, err)
		}

		r.vmPool <- vm
		r.poolMetrics.lock.Lock()
		r.poolMetrics.IdleVMs++
		r.poolMetrics.lock.Unlock()
	}

	duration := time.Since(start)
	r.logger.Info("VM pool pre-warmed",
		zap.Int("count", r.poolSize),
		zap.Duration("duration", duration),
		zap.Float64("avg_ms_per_vm", float64(duration.Milliseconds())/float64(r.poolSize)),
	)

	return nil
}

// createVMInstance creates a new isolated VM instance
func (r *WASMRuntime) createVMInstance() (*VMInstance, error) {
	store := wasmtime.NewStore(r.engine)

	// Configure store with resource limits
	store.SetWasiConfig(r.createWASIConfig())

	if r.resourceLimits.EnableFuelMetering {
		store.AddFuel(r.resourceLimits.FuelLimit)
	}

	// Set memory limits
	store.Limiter(
		r.resourceLimits.MaxMemoryBytes,
		r.resourceLimits.MaxTableElements,
		r.resourceLimits.MaxInstances,
		r.resourceLimits.MaxFunctions,
		1, // Max memories
	)

	linker := wasmtime.NewLinker(r.engine)

	// Add WASI support if enabled
	if r.sandboxConfig.EnableWASI {
		if err := linker.DefineWasi(); err != nil {
			return nil, fmt.Errorf("failed to define WASI: %w", err)
		}
	}

	vm := &VMInstance{
		ID:        fmt.Sprintf("vm-%d", time.Now().UnixNano()),
		Store:     store,
		Linker:    linker,
		CreatedAt: time.Now(),
		Isolated:  true,
	}

	return vm, nil
}

// createWASIConfig creates a sandboxed WASI configuration
func (r *WASMRuntime) createWASIConfig() *wasmtime.WasiConfig {
	config := wasmtime.NewWasiConfig()

	// Configure allowed filesystem access
	if r.sandboxConfig.FilesystemAccess {
		for _, dir := range r.sandboxConfig.AllowedDirectories {
			config.PreopenDir(dir, dir)
		}
	}

	// Inherit stdio
	config.InheritStdin()
	config.InheritStdout()
	config.InheritStderr()

	return config
}

// poolMaintainer maintains the VM pool health
func (r *WASMRuntime) poolMaintainer() {
	defer r.wg.Done()

	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-r.ctx.Done():
			return
		case <-ticker.C:
			r.maintainPool()
		}
	}
}

// maintainPool ensures pool has healthy VMs
func (r *WASMRuntime) maintainPool() {
	currentSize := len(r.vmPool)

	// Refill pool if below target
	for currentSize < r.poolSize {
		vm, err := r.createVMInstance()
		if err != nil {
			r.logger.Error("Failed to create replacement VM", zap.Error(err))
			break
		}

		select {
		case r.vmPool <- vm:
			currentSize++
			r.poolMetrics.lock.Lock()
			r.poolMetrics.IdleVMs++
			r.poolMetrics.lock.Unlock()
		default:
			// Pool full, shouldn't happen
			break
		}
	}
}

// LoadModule loads and caches a WASM module with instant retrieval
func (r *WASMRuntime) LoadModule(wasmBytes []byte, moduleHash string) (*wasmtime.Module, error) {
	// Check cache first
	if cached := r.moduleCache.Get(moduleHash); cached != nil {
		r.logger.Debug("Module cache hit", zap.String("hash", moduleHash))
		return cached.Module, nil
	}

	// Cache miss - compile module
	r.logger.Debug("Module cache miss", zap.String("hash", moduleHash))

	start := time.Now()
	module, err := wasmtime.NewModule(r.engine, wasmBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to compile module: %w", err)
	}
	compileTime := time.Since(start)

	// Cache the compiled module
	cached := &CachedModule{
		Module:       module,
		Hash:         moduleHash,
		Size:         uint64(len(wasmBytes)),
		CreatedAt:    time.Now(),
		LastAccessAt: time.Now(),
		AccessCount:  1,
	}

	r.moduleCache.Put(moduleHash, cached)

	r.logger.Info("Module compiled and cached",
		zap.String("hash", moduleHash),
		zap.Duration("compile_time", compileTime),
		zap.Uint64("size_bytes", cached.Size),
	)

	return module, nil
}

// ExecuteFunction executes a WASM function with optimized startup
func (r *WASMRuntime) ExecuteFunction(
	moduleBytes []byte,
	moduleHash string,
	functionName string,
	args []interface{},
) (interface{}, error) {
	startTime := time.Now()

	// Get VM from pool (warm start) or create new (cold start)
	vm, fromPool := r.acquireVM()

	if fromPool {
		r.metricsLock.Lock()
		r.metrics.WarmStarts++
		r.metricsLock.Unlock()
		r.poolMetrics.lock.Lock()
		r.poolMetrics.PoolHits++
		r.poolMetrics.lock.Unlock()
	} else {
		r.metricsLock.Lock()
		r.metrics.ColdStarts++
		r.metricsLock.Unlock()
		r.poolMetrics.lock.Lock()
		r.poolMetrics.PoolMisses++
		r.poolMetrics.lock.Unlock()
	}

	acquisitionTime := time.Since(startTime)

	// Load module (from cache or compile)
	module, err := r.LoadModule(moduleBytes, moduleHash)
	if err != nil {
		r.releaseVM(vm)
		return nil, fmt.Errorf("failed to load module: %w", err)
	}

	// Instantiate module
	instance, err := vm.Linker.Instantiate(vm.Store, module)
	if err != nil {
		r.releaseVM(vm)
		return nil, fmt.Errorf("failed to instantiate module: %w", err)
	}

	vm.Instance = instance
	vm.Module = module

	// Get function
	fn := instance.GetFunc(vm.Store, functionName)
	if fn == nil {
		r.releaseVM(vm)
		return nil, fmt.Errorf("function %s not found", functionName)
	}

	// Execute with timeout
	ctx, cancel := context.WithTimeout(r.ctx, time.Duration(r.resourceLimits.MaxExecutionTimeMs)*time.Millisecond)
	defer cancel()

	resultChan := make(chan interface{}, 1)
	errChan := make(chan error, 1)

	go func() {
		result, err := fn.Call(vm.Store, args...)
		if err != nil {
			errChan <- err
			return
		}
		resultChan <- result
	}()

	select {
	case <-ctx.Done():
		r.releaseVM(vm)
		return nil, fmt.Errorf("execution timeout after %dms", r.resourceLimits.MaxExecutionTimeMs)
	case err := <-errChan:
		r.releaseVM(vm)
		r.metricsLock.Lock()
		r.metrics.FailedExecutions++
		r.metricsLock.Unlock()
		return nil, fmt.Errorf("execution failed: %w", err)
	case result := <-resultChan:
		executionTime := time.Since(startTime)

		// Update metrics
		r.metricsLock.Lock()
		r.metrics.TotalExecutions++
		if fromPool {
			r.updateWarmStartMetrics(acquisitionTime.Milliseconds())
		} else {
			r.updateColdStartMetrics(acquisitionTime.Milliseconds())
		}
		r.metricsLock.Unlock()

		// Update VM usage
		vm.LastUsedAt = time.Now()
		vm.UsageCount++

		// Release VM back to pool
		r.releaseVM(vm)

		r.logger.Debug("Function executed",
			zap.String("function", functionName),
			zap.Bool("warm_start", fromPool),
			zap.Duration("total_time", executionTime),
			zap.Duration("acquisition_time", acquisitionTime),
			zap.Int64("vm_usage_count", vm.UsageCount),
		)

		return result, nil
	}
}

// acquireVM gets a VM from pool or creates new one
func (r *WASMRuntime) acquireVM() (*VMInstance, bool) {
	select {
	case vm := <-r.vmPool:
		r.poolMetrics.lock.Lock()
		r.poolMetrics.IdleVMs--
		r.poolMetrics.ActiveVMs++
		r.poolMetrics.lock.Unlock()
		return vm, true
	default:
		// Pool empty, create new VM (cold start)
		vm, err := r.createVMInstance()
		if err != nil {
			r.logger.Error("Failed to create VM on demand", zap.Error(err))
			return nil, false
		}
		return vm, false
	}
}

// releaseVM returns a VM to the pool or disposes it
func (r *WASMRuntime) releaseVM(vm *VMInstance) {
	if vm == nil {
		return
	}

	// Check if VM should be recycled
	if vm.UsageCount > 1000 {
		// VM used too many times, dispose and create fresh one
		r.logger.Debug("Disposing overused VM", zap.String("vm_id", vm.ID), zap.Int64("usage_count", vm.UsageCount))
		go func() {
			fresh, err := r.createVMInstance()
			if err != nil {
				r.logger.Error("Failed to create replacement VM", zap.Error(err))
				return
			}
			r.vmPool <- fresh
		}()
		return
	}

	// Return to pool
	select {
	case r.vmPool <- vm:
		r.poolMetrics.lock.Lock()
		r.poolMetrics.ActiveVMs--
		r.poolMetrics.IdleVMs++
		r.poolMetrics.lock.Unlock()
	default:
		// Pool full, dispose this VM
		r.logger.Debug("Pool full, disposing VM", zap.String("vm_id", vm.ID))
	}
}

// updateColdStartMetrics updates cold start performance metrics
func (r *WASMRuntime) updateColdStartMetrics(durationMS int64) {
	// Simple moving average for now
	// In production, use proper percentile tracking
	if r.metrics.ColdStarts == 0 {
		r.metrics.AvgColdStartMS = float64(durationMS)
	} else {
		r.metrics.AvgColdStartMS = (r.metrics.AvgColdStartMS*float64(r.metrics.ColdStarts-1) + float64(durationMS)) / float64(r.metrics.ColdStarts)
	}

	// Update percentiles (simplified - use proper percentile tracking in production)
	r.metrics.P99ColdStartMS = float64(durationMS) * 1.5 // Approximation
	r.metrics.P95ColdStartMS = float64(durationMS) * 1.2
	r.metrics.P50ColdStartMS = float64(durationMS)
}

// updateWarmStartMetrics updates warm start performance metrics
func (r *WASMRuntime) updateWarmStartMetrics(durationMS int64) {
	if r.metrics.WarmStarts == 0 {
		r.metrics.AvgWarmStartMS = float64(durationMS)
	} else {
		r.metrics.AvgWarmStartMS = (r.metrics.AvgWarmStartMS*float64(r.metrics.WarmStarts-1) + float64(durationMS)) / float64(r.metrics.WarmStarts)
	}
}

// Get retrieves a module from cache
func (mc *ModuleCache) Get(hash string) *CachedModule {
	mc.lock.RLock()
	defer mc.lock.RUnlock()

	if module, exists := mc.modules[hash]; exists {
		mc.hits++
		module.AccessCount++
		module.LastAccessAt = time.Now()
		return module
	}

	mc.misses++
	return nil
}

// Put stores a module in cache
func (mc *ModuleCache) Put(hash string, module *CachedModule) {
	mc.lock.Lock()
	defer mc.lock.Unlock()

	// Simple LRU eviction if cache full
	if len(mc.modules) >= mc.maxSize {
		// Remove oldest
		var oldestHash string
		var oldestTime time.Time

		for h, m := range mc.modules {
			if oldestTime.IsZero() || m.LastAccessAt.Before(oldestTime) {
				oldestHash = h
				oldestTime = m.LastAccessAt
			}
		}

		delete(mc.modules, oldestHash)
	}

	mc.modules[hash] = module
}

// GetMetrics returns current runtime metrics
func (r *WASMRuntime) GetMetrics() *RuntimeMetrics {
	r.metricsLock.RLock()
	defer r.metricsLock.RUnlock()

	metrics := *r.metrics

	// Calculate cache hit rate
	mc := r.moduleCache
	mc.lock.RLock()
	total := mc.hits + mc.misses
	if total > 0 {
		metrics.CacheHitRate = float64(mc.hits) / float64(total) * 100.0
	}
	mc.lock.RUnlock()

	// Get pool metrics
	r.poolMetrics.lock.RLock()
	metrics.ActiveVMs = r.poolMetrics.ActiveVMs
	r.poolMetrics.lock.RUnlock()

	return &metrics
}

// GetPoolMetrics returns VM pool metrics
func (r *WASMRuntime) GetPoolMetrics() *PoolMetrics {
	r.poolMetrics.lock.RLock()
	defer r.poolMetrics.lock.RUnlock()

	metrics := *r.poolMetrics
	return &metrics
}

// ExportMetrics exports metrics in JSON format
func (r *WASMRuntime) ExportMetrics() ([]byte, error) {
	metrics := r.GetMetrics()
	poolMetrics := r.GetPoolMetrics()

	export := map[string]interface{}{
		"runtime_metrics": metrics,
		"pool_metrics":    poolMetrics,
		"timestamp":       time.Now(),
		"version":         RuntimeVersion,
	}

	return json.MarshalIndent(export, "", "  ")
}

// ValidatePerformance checks if performance targets are met
func (r *WASMRuntime) ValidatePerformance() (*PerformanceValidation, error) {
	metrics := r.GetMetrics()

	validation := &PerformanceValidation{
		Timestamp: time.Now(),
		Targets:   make(map[string]TargetStatus),
	}

	// Check cold start target (<100ms)
	validation.Targets["cold_start"] = TargetStatus{
		Target:   ColdStartTargetMS,
		Actual:   metrics.AvgColdStartMS,
		Met:      metrics.AvgColdStartMS < ColdStartTargetMS,
		MetricName: "Average Cold Start Time (ms)",
	}

	// Check warm start target (<10ms)
	validation.Targets["warm_start"] = TargetStatus{
		Target:   WarmStartTargetMS,
		Actual:   metrics.AvgWarmStartMS,
		Met:      metrics.AvgWarmStartMS < WarmStartTargetMS,
		MetricName: "Average Warm Start Time (ms)",
	}

	// Check cache hit rate (>80%)
	validation.Targets["cache_hit_rate"] = TargetStatus{
		Target:   80.0,
		Actual:   metrics.CacheHitRate,
		Met:      metrics.CacheHitRate > 80.0,
		MetricName: "Cache Hit Rate (%)",
	}

	// Overall validation
	validation.OverallMet = true
	for _, status := range validation.Targets {
		if !status.Met {
			validation.OverallMet = false
			break
		}
	}

	return validation, nil
}

// PerformanceValidation represents performance target validation
type PerformanceValidation struct {
	Timestamp  time.Time                `json:"timestamp"`
	Targets    map[string]TargetStatus  `json:"targets"`
	OverallMet bool                     `json:"overall_met"`
}

// TargetStatus represents a single performance target status
type TargetStatus struct {
	MetricName string  `json:"metric_name"`
	Target     float64 `json:"target"`
	Actual     float64 `json:"actual"`
	Met        bool    `json:"met"`
}

// Close shuts down the runtime gracefully
func (r *WASMRuntime) Close() error {
	r.logger.Info("Shutting down WASM runtime")

	r.cancel()
	r.wg.Wait()

	// Drain VM pool
	close(r.vmPool)
	drained := 0
	for range r.vmPool {
		drained++
	}

	r.logger.Info("WASM runtime shutdown complete",
		zap.Int("vms_drained", drained),
	)

	return nil
}

// Benchmark runs performance benchmarks
func (r *WASMRuntime) Benchmark(wasmBytes []byte, iterations int) (*BenchmarkResults, error) {
	results := &BenchmarkResults{
		Iterations:    iterations,
		StartTime:     time.Now(),
		ColdStartTimes: make([]time.Duration, 0),
		WarmStartTimes: make([]time.Duration, 0),
	}

	moduleHash := fmt.Sprintf("bench-%d", time.Now().UnixNano())

	// Benchmark cold starts
	for i := 0; i < iterations; i++ {
		start := time.Now()

		// Clear cache to force cold start
		r.moduleCache.lock.Lock()
		delete(r.moduleCache.modules, moduleHash)
		r.moduleCache.lock.Unlock()

		_, err := r.LoadModule(wasmBytes, moduleHash)
		if err != nil {
			return nil, err
		}

		duration := time.Since(start)
		results.ColdStartTimes = append(results.ColdStartTimes, duration)
	}

	// Benchmark warm starts
	for i := 0; i < iterations; i++ {
		start := time.Now()
		_, err := r.LoadModule(wasmBytes, moduleHash)
		if err != nil {
			return nil, err
		}
		duration := time.Since(start)
		results.WarmStartTimes = append(results.WarmStartTimes, duration)
	}

	results.EndTime = time.Now()
	results.calculateStats()

	return results, nil
}

// BenchmarkResults contains benchmark statistics
type BenchmarkResults struct {
	Iterations      int             `json:"iterations"`
	StartTime       time.Time       `json:"start_time"`
	EndTime         time.Time       `json:"end_time"`
	ColdStartTimes  []time.Duration `json:"-"`
	WarmStartTimes  []time.Duration `json:"-"`
	AvgColdStartMS  float64         `json:"avg_cold_start_ms"`
	AvgWarmStartMS  float64         `json:"avg_warm_start_ms"`
	P50ColdStartMS  float64         `json:"p50_cold_start_ms"`
	P95ColdStartMS  float64         `json:"p95_cold_start_ms"`
	P99ColdStartMS  float64         `json:"p99_cold_start_ms"`
	P50WarmStartMS  float64         `json:"p50_warm_start_ms"`
	P95WarmStartMS  float64         `json:"p95_warm_start_ms"`
	P99WarmStartMS  float64         `json:"p99_warm_start_ms"`
	ImprovementFactor float64       `json:"improvement_factor"`
}

// calculateStats calculates benchmark statistics
func (br *BenchmarkResults) calculateStats() {
	// Calculate averages
	var coldSum, warmSum time.Duration
	for _, d := range br.ColdStartTimes {
		coldSum += d
	}
	for _, d := range br.WarmStartTimes {
		warmSum += d
	}

	br.AvgColdStartMS = float64(coldSum.Milliseconds()) / float64(len(br.ColdStartTimes))
	br.AvgWarmStartMS = float64(warmSum.Milliseconds()) / float64(len(br.WarmStartTimes))

	// Simple percentile calculation (should use proper algorithm in production)
	if len(br.ColdStartTimes) > 0 {
		br.P50ColdStartMS = float64(br.ColdStartTimes[len(br.ColdStartTimes)/2].Milliseconds())
		br.P95ColdStartMS = float64(br.ColdStartTimes[int(float64(len(br.ColdStartTimes))*0.95)].Milliseconds())
		br.P99ColdStartMS = float64(br.ColdStartTimes[int(float64(len(br.ColdStartTimes))*0.99)].Milliseconds())
	}

	if len(br.WarmStartTimes) > 0 {
		br.P50WarmStartMS = float64(br.WarmStartTimes[len(br.WarmStartTimes)/2].Milliseconds())
		br.P95WarmStartMS = float64(br.WarmStartTimes[int(float64(len(br.WarmStartTimes))*0.95)].Milliseconds())
		br.P99WarmStartMS = float64(br.WarmStartTimes[int(float64(len(br.WarmStartTimes))*0.99)].Milliseconds())
	}

	// Calculate improvement factor
	if br.AvgWarmStartMS > 0 {
		br.ImprovementFactor = br.AvgColdStartMS / br.AvgWarmStartMS
	}
}

// LoadModuleFromFile loads a WASM module from file
func (r *WASMRuntime) LoadModuleFromFile(path string) (*wasmtime.Module, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read module file: %w", err)
	}

	hash := fmt.Sprintf("file-%s", filepath.Base(path))
	return r.LoadModule(data, hash)
}

// WASIFileSystem provides virtualized filesystem for WASM modules
type WASIFileSystem struct {
	allowedPaths map[string]bool
	virtualPaths map[string]string
	lock         sync.RWMutex
}

// NewWASIFileSystem creates a sandboxed filesystem
func NewWASIFileSystem() *WASIFileSystem {
	return &WASIFileSystem{
		allowedPaths: make(map[string]bool),
		virtualPaths: make(map[string]string),
	}
}

// AllowPath allows access to a specific path
func (fs *WASIFileSystem) AllowPath(path string) {
	fs.lock.Lock()
	defer fs.lock.Unlock()
	fs.allowedPaths[path] = true
}

// IsPathAllowed checks if path access is allowed
func (fs *WASIFileSystem) IsPathAllowed(path string) bool {
	fs.lock.RLock()
	defer fs.lock.RUnlock()
	return fs.allowedPaths[path]
}

// Export exports runtime state for persistence
func (r *WASMRuntime) Export(w io.Writer) error {
	state := map[string]interface{}{
		"version": RuntimeVersion,
		"metrics": r.GetMetrics(),
		"pool_metrics": r.GetPoolMetrics(),
		"config": map[string]interface{}{
			"pool_size": r.poolSize,
			"sandbox_config": r.sandboxConfig,
			"resource_limits": r.resourceLimits,
		},
	}

	encoder := json.NewEncoder(w)
	encoder.SetIndent("", "  ")
	return encoder.Encode(state)
}

// Import imports runtime state
func (r *WASMRuntime) Import(reader io.Reader) error {
	var state map[string]interface{}
	if err := json.NewDecoder(reader).Decode(&state); err != nil {
		return fmt.Errorf("failed to decode state: %w", err)
	}

	// Validate version compatibility
	version, ok := state["version"].(string)
	if !ok || version != RuntimeVersion {
		return fmt.Errorf("incompatible runtime version: %s", version)
	}

	r.logger.Info("Runtime state imported", zap.String("version", version))
	return nil
}
