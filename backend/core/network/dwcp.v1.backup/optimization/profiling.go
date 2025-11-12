package optimization

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"runtime/trace"
	"time"
)

// Profiler provides profiling and performance analysis
type Profiler struct {
	cpuProfile   *os.File
	memProfile   *os.File
	blockProfile *os.File
	traceFile    *os.File
	startTime    time.Time
	enabled      bool
}

// NewProfiler creates a new profiler
func NewProfiler() *Profiler {
	return &Profiler{
		enabled: false,
	}
}

// StartCPUProfile starts CPU profiling
func (p *Profiler) StartCPUProfile(filename string) error {
	if p.enabled {
		return fmt.Errorf("profiler already running")
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}

	p.cpuProfile = f
	p.startTime = time.Now()
	p.enabled = true

	return pprof.StartCPUProfile(f)
}

// StopCPUProfile stops CPU profiling
func (p *Profiler) StopCPUProfile() {
	if p.cpuProfile != nil {
		pprof.StopCPUProfile()
		p.cpuProfile.Close()
		p.cpuProfile = nil
		p.enabled = false
	}
}

// WriteMemProfile writes memory profile
func (p *Profiler) WriteMemProfile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	runtime.GC() // Force GC for accurate stats
	return pprof.WriteHeapProfile(f)
}

// WriteBlockProfile writes goroutine blocking profile
func (p *Profiler) WriteBlockProfile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	return pprof.Lookup("block").WriteTo(f, 0)
}

// WriteGoroutineProfile writes goroutine profile
func (p *Profiler) WriteGoroutineProfile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	return pprof.Lookup("goroutine").WriteTo(f, 0)
}

// WriteMutexProfile writes mutex contention profile
func (p *Profiler) WriteMutexProfile(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	return pprof.Lookup("mutex").WriteTo(f, 0)
}

// StartTrace starts execution trace
func (p *Profiler) StartTrace(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}

	p.traceFile = f
	return trace.Start(f)
}

// StopTrace stops execution trace
func (p *Profiler) StopTrace() {
	if p.traceFile != nil {
		trace.Stop()
		p.traceFile.Close()
		p.traceFile = nil
	}
}

// WriteAllProfiles writes all profile types
func (p *Profiler) WriteAllProfiles(prefix string) error {
	timestamp := time.Now().Format("20060102-150405")

	// CPU profile
	if err := p.StartCPUProfile(fmt.Sprintf("%s-cpu-%s.prof", prefix, timestamp)); err != nil {
		return err
	}
	time.Sleep(30 * time.Second)
	p.StopCPUProfile()

	// Memory profile
	if err := p.WriteMemProfile(fmt.Sprintf("%s-mem-%s.prof", prefix, timestamp)); err != nil {
		return err
	}

	// Block profile
	runtime.SetBlockProfileRate(1)
	if err := p.WriteBlockProfile(fmt.Sprintf("%s-block-%s.prof", prefix, timestamp)); err != nil {
		return err
	}

	// Goroutine profile
	if err := p.WriteGoroutineProfile(fmt.Sprintf("%s-goroutine-%s.prof", prefix, timestamp)); err != nil {
		return err
	}

	// Mutex profile
	runtime.SetMutexProfileFraction(1)
	if err := p.WriteMutexProfile(fmt.Sprintf("%s-mutex-%s.prof", prefix, timestamp)); err != nil {
		return err
	}

	return nil
}

// PerformanceMetrics tracks performance metrics
type PerformanceMetrics struct {
	startTime      time.Time
	requestCount   uint64
	errorCount     uint64
	totalLatency   time.Duration
	minLatency     time.Duration
	maxLatency     time.Duration
	bytesProcessed uint64
}

// NewPerformanceMetrics creates performance metrics tracker
func NewPerformanceMetrics() *PerformanceMetrics {
	return &PerformanceMetrics{
		startTime:  time.Now(),
		minLatency: time.Hour, // Initialize to large value
	}
}

// RecordRequest records a request
func (pm *PerformanceMetrics) RecordRequest(latency time.Duration, bytes uint64, err error) {
	atomic.AddUint64(&pm.requestCount, 1)
	atomic.AddUint64(&pm.bytesProcessed, bytes)

	if err != nil {
		atomic.AddUint64(&pm.errorCount, 1)
	}

	// Update latency stats
	pm.totalLatency += latency
	if latency < pm.minLatency {
		pm.minLatency = latency
	}
	if latency > pm.maxLatency {
		pm.maxLatency = latency
	}
}

// Stats returns performance statistics
func (pm *PerformanceMetrics) Stats() Stats {
	requests := atomic.LoadUint64(&pm.requestCount)
	errors := atomic.LoadUint64(&pm.errorCount)
	bytes := atomic.LoadUint64(&pm.bytesProcessed)
	elapsed := time.Since(pm.startTime)

	avgLatency := time.Duration(0)
	if requests > 0 {
		avgLatency = pm.totalLatency / time.Duration(requests)
	}

	throughput := float64(0)
	if elapsed.Seconds() > 0 {
		throughput = float64(bytes) / elapsed.Seconds()
	}

	return Stats{
		Requests:      requests,
		Errors:        errors,
		BytesProcessed: bytes,
		Elapsed:       elapsed,
		AvgLatency:    avgLatency,
		MinLatency:    pm.minLatency,
		MaxLatency:    pm.maxLatency,
		Throughput:    throughput,
	}
}

// Stats contains performance statistics
type Stats struct {
	Requests       uint64
	Errors         uint64
	BytesProcessed uint64
	Elapsed        time.Duration
	AvgLatency     time.Duration
	MinLatency     time.Duration
	MaxLatency     time.Duration
	Throughput     float64 // bytes/sec
}

// String formats stats as string
func (s Stats) String() string {
	return fmt.Sprintf(
		"Requests: %d, Errors: %d, Throughput: %.2f MB/s, Avg Latency: %v, Min: %v, Max: %v",
		s.Requests,
		s.Errors,
		s.Throughput/1024/1024,
		s.AvgLatency,
		s.MinLatency,
		s.MaxLatency,
	)
}

// LatencyHistogram tracks latency distribution
type LatencyHistogram struct {
	buckets []uint64
	limits  []time.Duration
}

// NewLatencyHistogram creates a latency histogram
func NewLatencyHistogram() *LatencyHistogram {
	return &LatencyHistogram{
		buckets: make([]uint64, 10),
		limits: []time.Duration{
			100 * time.Microsecond,
			500 * time.Microsecond,
			1 * time.Millisecond,
			5 * time.Millisecond,
			10 * time.Millisecond,
			50 * time.Millisecond,
			100 * time.Millisecond,
			500 * time.Millisecond,
			1 * time.Second,
			5 * time.Second,
		},
	}
}

// Record records a latency measurement
func (lh *LatencyHistogram) Record(latency time.Duration) {
	for i, limit := range lh.limits {
		if latency <= limit {
			atomic.AddUint64(&lh.buckets[i], 1)
			return
		}
	}
	// Overflow bucket
	atomic.AddUint64(&lh.buckets[len(lh.buckets)-1], 1)
}

// Print prints histogram
func (lh *LatencyHistogram) Print() {
	fmt.Println("Latency Histogram:")
	for i, limit := range lh.limits {
		count := atomic.LoadUint64(&lh.buckets[i])
		fmt.Printf("  <= %v: %d\n", limit, count)
	}
}

// MemoryStats provides memory statistics
type MemoryStats struct {
	Alloc        uint64
	TotalAlloc   uint64
	Sys          uint64
	NumGC        uint32
	PauseTotalNs uint64
	HeapAlloc    uint64
	HeapSys      uint64
	HeapIdle     uint64
	HeapInuse    uint64
	HeapObjects  uint64
}

// GetMemoryStats returns current memory statistics
func GetMemoryStats() MemoryStats {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)

	return MemoryStats{
		Alloc:        m.Alloc,
		TotalAlloc:   m.TotalAlloc,
		Sys:          m.Sys,
		NumGC:        m.NumGC,
		PauseTotalNs: m.PauseTotalNs,
		HeapAlloc:    m.HeapAlloc,
		HeapSys:      m.HeapSys,
		HeapIdle:     m.HeapIdle,
		HeapInuse:    m.HeapInuse,
		HeapObjects:  m.HeapObjects,
	}
}

// PrintMemoryStats prints memory statistics
func PrintMemoryStats() {
	stats := GetMemoryStats()
	fmt.Printf("Memory Stats:\n")
	fmt.Printf("  Alloc: %d MB\n", stats.Alloc/1024/1024)
	fmt.Printf("  TotalAlloc: %d MB\n", stats.TotalAlloc/1024/1024)
	fmt.Printf("  Sys: %d MB\n", stats.Sys/1024/1024)
	fmt.Printf("  NumGC: %d\n", stats.NumGC)
	fmt.Printf("  HeapAlloc: %d MB\n", stats.HeapAlloc/1024/1024)
	fmt.Printf("  HeapSys: %d MB\n", stats.HeapSys/1024/1024)
	fmt.Printf("  HeapIdle: %d MB\n", stats.HeapIdle/1024/1024)
	fmt.Printf("  HeapInuse: %d MB\n", stats.HeapInuse/1024/1024)
	fmt.Printf("  HeapObjects: %d\n", stats.HeapObjects)
}

import "sync/atomic"
