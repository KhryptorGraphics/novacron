// Package optimization provides memory optimization for DWCP v3.
package optimization

import (
	"context"
	"fmt"
	"runtime"
	"runtime/debug"
	"sync"
	"time"
)

// MemoryOptimizerConfig defines configuration for memory optimization.
type MemoryOptimizerConfig struct {
	// GC tuning
	GCPercent  int           // GOGC percentage
	GCInterval time.Duration // Force GC interval

	// Buffer pooling
	EnableBufferPool bool
	SmallBufferSize  int // e.g., 4KB
	MediumBufferSize int // e.g., 64KB
	LargeBufferSize  int // e.g., 1MB
	MaxPooledBuffers int

	// Object pooling
	EnableObjectPool bool
	MaxPooledObjects int

	// Memory limits
	MaxHeapSize          uint64  // bytes
	HeapWarningThreshold float64 // 0-1

	// Leak detection
	EnableLeakDetection bool
	LeakCheckInterval   time.Duration

	// Allocation tracking
	TrackAllocations    bool
	AllocationThreshold uint64 // bytes per second
}

// DefaultMemoryOptimizerConfig returns default memory optimizer configuration.
func DefaultMemoryOptimizerConfig() *MemoryOptimizerConfig {
	return &MemoryOptimizerConfig{
		GCPercent:            100,
		GCInterval:           5 * time.Minute,
		EnableBufferPool:     true,
		SmallBufferSize:      4 * 1024,    // 4KB
		MediumBufferSize:     64 * 1024,   // 64KB
		LargeBufferSize:      1024 * 1024, // 1MB
		MaxPooledBuffers:     10000,
		EnableObjectPool:     true,
		MaxPooledObjects:     5000,
		MaxHeapSize:          14 * 1024 * 1024 * 1024, // 14GB
		HeapWarningThreshold: 0.85,
		EnableLeakDetection:  true,
		LeakCheckInterval:    1 * time.Minute,
		TrackAllocations:     true,
		AllocationThreshold:  100 * 1024 * 1024, // 100MB/s
	}
}

// MemoryOptimizer provides memory optimization for DWCP v3.
type MemoryOptimizer struct {
	config *MemoryOptimizerConfig
	mu     sync.RWMutex

	// Buffer pools
	smallBufferPool  *BufferPool
	mediumBufferPool *BufferPool
	largeBufferPool  *BufferPool

	// Object pools
	objectPools map[string]*ObjectPool

	// Allocation tracking
	allocations map[string]*AllocationTracker

	// Leak detection
	leakDetector *LeakDetector

	// Background tasks
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// BufferPool implements a size-stratified buffer pool.
type BufferPool struct {
	size     int
	maxCount int
	pool     *sync.Pool
	count    int
	mu       sync.Mutex
}

// ObjectPool implements a generic object pool.
type ObjectPool struct {
	name     string
	maxCount int
	pool     *sync.Pool
	count    int
	mu       sync.Mutex
	factory  func() interface{}
	reset    func(interface{})
}

// AllocationTracker tracks memory allocations for a component.
type AllocationTracker struct {
	component     string
	allocations   uint64
	deallocations uint64
	currentBytes  uint64
	peakBytes     uint64
	mu            sync.Mutex
}

// LeakDetector detects potential memory leaks.
type LeakDetector struct {
	snapshots    []*MemorySnapshot
	maxSnapshots int
	mu           sync.Mutex
}

// MemorySnapshot captures memory state at a point in time.
type MemorySnapshot struct {
	Timestamp  time.Time
	HeapAlloc  uint64
	HeapSys    uint64
	StackInuse uint64
	Goroutines int
	GCCount    uint32
}

// NewMemoryOptimizer creates a new memory optimizer.
func NewMemoryOptimizer(config *MemoryOptimizerConfig) *MemoryOptimizer {
	if config == nil {
		config = DefaultMemoryOptimizerConfig()
	}

	// Set GOGC
	debug.SetGCPercent(config.GCPercent)

	ctx, cancel := context.WithCancel(context.Background())

	o := &MemoryOptimizer{
		config:      config,
		objectPools: make(map[string]*ObjectPool),
		allocations: make(map[string]*AllocationTracker),
		ctx:         ctx,
		cancel:      cancel,
	}

	// Initialize buffer pools
	if config.EnableBufferPool {
		o.smallBufferPool = NewBufferPool(config.SmallBufferSize, config.MaxPooledBuffers)
		o.mediumBufferPool = NewBufferPool(config.MediumBufferSize, config.MaxPooledBuffers)
		o.largeBufferPool = NewBufferPool(config.LargeBufferSize, config.MaxPooledBuffers)
	}

	// Initialize leak detector
	if config.EnableLeakDetection {
		o.leakDetector = NewLeakDetector(100) // Keep 100 snapshots
	}

	// Start background tasks
	o.wg.Add(2)
	go o.runGCMonitoring()
	go o.runLeakDetection()

	return o
}

// NewBufferPool creates a new buffer pool.
func NewBufferPool(size, maxCount int) *BufferPool {
	bp := &BufferPool{
		size:     size,
		maxCount: maxCount,
	}

	bp.pool = &sync.Pool{
		New: func() interface{} {
			bp.mu.Lock()
			bp.count++
			bp.mu.Unlock()
			return make([]byte, size)
		},
	}

	return bp
}

// Get gets a buffer from the pool.
func (bp *BufferPool) Get() []byte {
	return bp.pool.Get().([]byte)
}

// Put returns a buffer to the pool.
func (bp *BufferPool) Put(buf []byte) {
	if len(buf) != bp.size {
		return // Wrong size, don't pool
	}

	bp.mu.Lock()
	defer bp.mu.Unlock()

	// Limit pool size
	if bp.count > bp.maxCount {
		bp.count--
		return
	}

	// Clear buffer before returning
	for i := range buf {
		buf[i] = 0
	}

	bp.pool.Put(buf)
}

// GetStats returns buffer pool statistics.
func (bp *BufferPool) GetStats() map[string]interface{} {
	bp.mu.Lock()
	defer bp.mu.Unlock()

	return map[string]interface{}{
		"size":      bp.size,
		"count":     bp.count,
		"max_count": bp.maxCount,
	}
}

// GetBuffer gets a buffer of appropriate size.
func (o *MemoryOptimizer) GetBuffer(size int) []byte {
	if !o.config.EnableBufferPool {
		return make([]byte, size)
	}

	switch {
	case size <= o.config.SmallBufferSize:
		return o.smallBufferPool.Get()[:size]
	case size <= o.config.MediumBufferSize:
		return o.mediumBufferPool.Get()[:size]
	case size <= o.config.LargeBufferSize:
		return o.largeBufferPool.Get()[:size]
	default:
		return make([]byte, size) // Too large to pool
	}
}

// PutBuffer returns a buffer to the pool.
func (o *MemoryOptimizer) PutBuffer(buf []byte) {
	if !o.config.EnableBufferPool {
		return
	}

	size := cap(buf)
	switch {
	case size == o.config.SmallBufferSize:
		o.smallBufferPool.Put(buf[:cap(buf)])
	case size == o.config.MediumBufferSize:
		o.mediumBufferPool.Put(buf[:cap(buf)])
	case size == o.config.LargeBufferSize:
		o.largeBufferPool.Put(buf[:cap(buf)])
	}
}

// RegisterObjectPool registers an object pool for a type.
func (o *MemoryOptimizer) RegisterObjectPool(name string, factory func() interface{}, reset func(interface{})) error {
	if !o.config.EnableObjectPool {
		return fmt.Errorf("object pooling disabled")
	}

	o.mu.Lock()
	defer o.mu.Unlock()

	if _, exists := o.objectPools[name]; exists {
		return fmt.Errorf("object pool already registered: %s", name)
	}

	pool := &ObjectPool{
		name:     name,
		maxCount: o.config.MaxPooledObjects,
		factory:  factory,
		reset:    reset,
	}

	pool.pool = &sync.Pool{
		New: func() interface{} {
			pool.mu.Lock()
			pool.count++
			pool.mu.Unlock()
			return factory()
		},
	}

	o.objectPools[name] = pool

	return nil
}

// GetObject gets an object from a pool.
func (o *MemoryOptimizer) GetObject(poolName string) (interface{}, error) {
	o.mu.RLock()
	pool, exists := o.objectPools[poolName]
	o.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("object pool not found: %s", poolName)
	}

	return pool.pool.Get(), nil
}

// PutObject returns an object to a pool.
func (o *MemoryOptimizer) PutObject(poolName string, obj interface{}) error {
	o.mu.RLock()
	pool, exists := o.objectPools[poolName]
	o.mu.RUnlock()

	if !exists {
		return fmt.Errorf("object pool not found: %s", poolName)
	}

	pool.mu.Lock()
	defer pool.mu.Unlock()

	// Limit pool size
	if pool.count > pool.maxCount {
		pool.count--
		return nil
	}

	// Reset object
	if pool.reset != nil {
		pool.reset(obj)
	}

	pool.pool.Put(obj)
	return nil
}

// TrackAllocation tracks a memory allocation.
func (o *MemoryOptimizer) TrackAllocation(component string, bytes uint64) {
	if !o.config.TrackAllocations {
		return
	}

	o.mu.Lock()
	tracker, exists := o.allocations[component]
	if !exists {
		tracker = &AllocationTracker{
			component: component,
		}
		o.allocations[component] = tracker
	}
	o.mu.Unlock()

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	tracker.allocations++
	tracker.currentBytes += bytes
	if tracker.currentBytes > tracker.peakBytes {
		tracker.peakBytes = tracker.currentBytes
	}
}

// TrackDeallocation tracks a memory deallocation.
func (o *MemoryOptimizer) TrackDeallocation(component string, bytes uint64) {
	if !o.config.TrackAllocations {
		return
	}

	o.mu.RLock()
	tracker, exists := o.allocations[component]
	o.mu.RUnlock()

	if !exists {
		return
	}

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	tracker.deallocations++
	if tracker.currentBytes >= bytes {
		tracker.currentBytes -= bytes
	}
}

// GetAllocationStats returns allocation statistics for a component.
func (o *MemoryOptimizer) GetAllocationStats(component string) (*AllocationTracker, error) {
	o.mu.RLock()
	tracker, exists := o.allocations[component]
	o.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("component not tracked: %s", component)
	}

	tracker.mu.Lock()
	defer tracker.mu.Unlock()

	// Return a copy
	return &AllocationTracker{
		component:     tracker.component,
		allocations:   tracker.allocations,
		deallocations: tracker.deallocations,
		currentBytes:  tracker.currentBytes,
		peakBytes:     tracker.peakBytes,
	}, nil
}

// NewLeakDetector creates a new leak detector.
func NewLeakDetector(maxSnapshots int) *LeakDetector {
	return &LeakDetector{
		snapshots:    make([]*MemorySnapshot, 0, maxSnapshots),
		maxSnapshots: maxSnapshots,
	}
}

// TakeSnapshot takes a memory snapshot.
func (ld *LeakDetector) TakeSnapshot() {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	snapshot := &MemorySnapshot{
		Timestamp:  time.Now(),
		HeapAlloc:  m.HeapAlloc,
		HeapSys:    m.HeapSys,
		StackInuse: m.StackInuse,
		Goroutines: runtime.NumGoroutine(),
		GCCount:    m.NumGC,
	}

	ld.mu.Lock()
	defer ld.mu.Unlock()

	ld.snapshots = append(ld.snapshots, snapshot)

	// Limit snapshot count
	if len(ld.snapshots) > ld.maxSnapshots {
		ld.snapshots = ld.snapshots[1:]
	}
}

// DetectLeaks detects potential memory leaks.
func (ld *LeakDetector) DetectLeaks() []string {
	ld.mu.Lock()
	defer ld.mu.Unlock()

	if len(ld.snapshots) < 10 {
		return nil // Need more data
	}

	var leaks []string

	// Check for monotonic heap growth
	recentSnapshots := ld.snapshots[len(ld.snapshots)-10:]
	heapGrowth := true
	for i := 1; i < len(recentSnapshots); i++ {
		if recentSnapshots[i].HeapAlloc <= recentSnapshots[i-1].HeapAlloc {
			heapGrowth = false
			break
		}
	}

	if heapGrowth {
		leaks = append(leaks, "Monotonic heap growth detected")
	}

	// Check for goroutine leaks
	goroutineGrowth := true
	for i := 1; i < len(recentSnapshots); i++ {
		if recentSnapshots[i].Goroutines <= recentSnapshots[i-1].Goroutines {
			goroutineGrowth = false
			break
		}
	}

	if goroutineGrowth {
		leaks = append(leaks, "Monotonic goroutine count growth detected")
	}

	// Check for excessive GC
	first := recentSnapshots[0]
	last := recentSnapshots[len(recentSnapshots)-1]
	gcRate := float64(last.GCCount-first.GCCount) / last.Timestamp.Sub(first.Timestamp).Seconds()

	if gcRate > 10 { // More than 10 GC/second
		leaks = append(leaks, fmt.Sprintf("High GC rate: %.2f/sec", gcRate))
	}

	return leaks
}

// runGCMonitoring runs GC monitoring in the background.
func (o *MemoryOptimizer) runGCMonitoring() {
	defer o.wg.Done()

	ticker := time.NewTicker(o.config.GCInterval)
	defer ticker.Stop()

	for {
		select {
		case <-o.ctx.Done():
			return

		case <-ticker.C:
			var m runtime.MemStats
			runtime.ReadMemStats(&m)

			// Check heap size warning
			if o.config.MaxHeapSize > 0 {
				heapUsage := float64(m.HeapAlloc) / float64(o.config.MaxHeapSize)
				if heapUsage > o.config.HeapWarningThreshold {
					// Trigger GC
					runtime.GC()
				}
			}
		}
	}
}

// runLeakDetection runs leak detection in the background.
func (o *MemoryOptimizer) runLeakDetection() {
	defer o.wg.Done()

	if !o.config.EnableLeakDetection {
		return
	}

	ticker := time.NewTicker(o.config.LeakCheckInterval)
	defer ticker.Stop()

	for {
		select {
		case <-o.ctx.Done():
			return

		case <-ticker.C:
			o.leakDetector.TakeSnapshot()
			leaks := o.leakDetector.DetectLeaks()

			if len(leaks) > 0 {
				// Log leaks (in production, would send to monitoring)
				for _, leak := range leaks {
					fmt.Printf("LEAK DETECTED: %s\n", leak)
				}
			}
		}
	}
}

// GetMemoryStats returns memory optimization statistics.
func (o *MemoryOptimizer) GetMemoryStats() map[string]interface{} {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	stats := map[string]interface{}{
		"heap_alloc":   m.HeapAlloc,
		"heap_sys":     m.HeapSys,
		"stack_inuse":  m.StackInuse,
		"goroutines":   runtime.NumGoroutine(),
		"gc_count":     m.NumGC,
		"gc_pause_ns":  m.PauseTotalNs,
		"buffer_pools": make(map[string]interface{}),
		"object_pools": make(map[string]interface{}),
		"allocations":  make(map[string]interface{}),
	}

	if o.config.EnableBufferPool {
		stats["buffer_pools"].(map[string]interface{})["small"] = o.smallBufferPool.GetStats()
		stats["buffer_pools"].(map[string]interface{})["medium"] = o.mediumBufferPool.GetStats()
		stats["buffer_pools"].(map[string]interface{})["large"] = o.largeBufferPool.GetStats()
	}

	o.mu.RLock()
	for name, pool := range o.objectPools {
		pool.mu.Lock()
		stats["object_pools"].(map[string]interface{})[name] = map[string]interface{}{
			"count":     pool.count,
			"max_count": pool.maxCount,
		}
		pool.mu.Unlock()
	}

	for name, _ := range o.allocations {
		if stats_data, err := o.GetAllocationStats(name); err == nil {
			stats["allocations"].(map[string]interface{})[name] = map[string]interface{}{
				"allocations":   stats_data.allocations,
				"deallocations": stats_data.deallocations,
				"current_bytes": stats_data.currentBytes,
				"peak_bytes":    stats_data.peakBytes,
			}
		}
	}
	o.mu.RUnlock()

	return stats
}

// ForceGC forces garbage collection.
func (o *MemoryOptimizer) ForceGC() {
	runtime.GC()
}

// Close stops the optimizer and cleans up resources.
func (o *MemoryOptimizer) Close() error {
	o.cancel()
	o.wg.Wait()
	return nil
}
