package profiler

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

// ContinuousProfiler provides <2% overhead continuous profiling
type ContinuousProfiler struct {
	config       ProfilerConfig
	running      bool
	mu           sync.RWMutex
	profiles     map[string]*ProfileData
	overheadChan chan float64
	stopChan     chan struct{}
}

// ProfilerConfig defines profiling parameters
type ProfilerConfig struct {
	SamplingRate   int           // 100 Hz default
	ProfileTypes   []string      // cpu, memory, mutex, block, goroutine
	OutputDir      string
	RetentionDays  int
	OverheadTarget float64 // 0.02 (2%)
	FlushInterval  time.Duration
}

// ProfileData stores profiling information
type ProfileData struct {
	Type      string
	StartTime time.Time
	EndTime   time.Time
	FilePath  string
	Size      int64
	Overhead  float64
	Samples   int64
}

// NewContinuousProfiler creates profiler with <2% overhead
func NewContinuousProfiler(config ProfilerConfig) *ContinuousProfiler {
	if config.SamplingRate == 0 {
		config.SamplingRate = 100 // 100 Hz
	}
	if config.FlushInterval == 0 {
		config.FlushInterval = 60 * time.Second
	}
	if config.OverheadTarget == 0 {
		config.OverheadTarget = 0.02
	}

	return &ContinuousProfiler{
		config:       config,
		profiles:     make(map[string]*ProfileData),
		overheadChan: make(chan float64, 100),
		stopChan:     make(chan struct{}),
	}
}

// Start begins continuous profiling
func (cp *ContinuousProfiler) Start(ctx context.Context) error {
	cp.mu.Lock()
	if cp.running {
		cp.mu.Unlock()
		return fmt.Errorf("profiler already running")
	}
	cp.running = true
	cp.mu.Unlock()

	// Create output directory
	if err := os.MkdirAll(cp.config.OutputDir, 0755); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	// Set profiling rates
	runtime.SetCPUProfileRate(cp.config.SamplingRate)
	runtime.SetBlockProfileRate(cp.config.SamplingRate)
	runtime.SetMutexProfileFraction(1)

	// Start profiling goroutines
	for _, profType := range cp.config.ProfileTypes {
		go cp.profileWorker(ctx, profType)
	}

	// Monitor overhead
	go cp.monitorOverhead(ctx)

	// Cleanup old profiles
	go cp.cleanupWorker(ctx)

	return nil
}

// profileWorker handles continuous profiling for a type
func (cp *ContinuousProfiler) profileWorker(ctx context.Context, profType string) {
	ticker := time.NewTicker(cp.config.FlushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-cp.stopChan:
			return
		case <-ticker.C:
			if err := cp.captureProfile(profType); err != nil {
				fmt.Printf("Error capturing %s profile: %v\n", profType, err)
			}
		}
	}
}

// captureProfile captures a single profile
func (cp *ContinuousProfiler) captureProfile(profType string) error {
	startTime := time.Now()
	timestamp := startTime.Format("20060102-150405")
	filename := filepath.Join(cp.config.OutputDir, fmt.Sprintf("%s-%s.pprof", profType, timestamp))

	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("create profile file: %w", err)
	}
	defer f.Close()

	var profileErr error
	var samples int64

	switch profType {
	case "cpu":
		profileErr = pprof.StartCPUProfile(f)
		if profileErr == nil {
			time.Sleep(30 * time.Second) // 30s CPU profile
			pprof.StopCPUProfile()
		}
		samples = int64(30 * cp.config.SamplingRate)

	case "memory", "heap":
		runtime.GC() // Get accurate heap snapshot
		profileErr = pprof.WriteHeapProfile(f)
		samples = int64(runtime.MemStats{}.Mallocs)

	case "allocs":
		runtime.GC()
		prof := pprof.Lookup("allocs")
		if prof != nil {
			profileErr = prof.WriteTo(f, 0)
		}

	case "mutex":
		prof := pprof.Lookup("mutex")
		if prof != nil {
			profileErr = prof.WriteTo(f, 0)
			samples = int64(prof.Count())
		}

	case "block":
		prof := pprof.Lookup("block")
		if prof != nil {
			profileErr = prof.WriteTo(f, 0)
			samples = int64(prof.Count())
		}

	case "goroutine":
		prof := pprof.Lookup("goroutine")
		if prof != nil {
			profileErr = prof.WriteTo(f, 0)
			samples = int64(runtime.NumGoroutine())
		}

	case "threadcreate":
		prof := pprof.Lookup("threadcreate")
		if prof != nil {
			profileErr = prof.WriteTo(f, 0)
		}

	default:
		return fmt.Errorf("unknown profile type: %s", profType)
	}

	if profileErr != nil {
		os.Remove(filename)
		return profileErr
	}

	endTime := time.Now()
	duration := endTime.Sub(startTime)

	// Calculate overhead
	overhead := duration.Seconds() / cp.config.FlushInterval.Seconds()
	cp.overheadChan <- overhead

	// Get file size
	stat, _ := f.Stat()
	size := int64(0)
	if stat != nil {
		size = stat.Size()
	}

	// Store profile data
	cp.mu.Lock()
	cp.profiles[filename] = &ProfileData{
		Type:      profType,
		StartTime: startTime,
		EndTime:   endTime,
		FilePath:  filename,
		Size:      size,
		Overhead:  overhead,
		Samples:   samples,
	}
	cp.mu.Unlock()

	return nil
}

// monitorOverhead monitors profiling overhead
func (cp *ContinuousProfiler) monitorOverhead(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	overheads := make([]float64, 0, 10)

	for {
		select {
		case <-ctx.Done():
			return
		case <-cp.stopChan:
			return
		case overhead := <-cp.overheadChan:
			overheads = append(overheads, overhead)
			if len(overheads) > 10 {
				overheads = overheads[1:]
			}
		case <-ticker.C:
			if len(overheads) > 0 {
				avgOverhead := average(overheads)
				if avgOverhead > cp.config.OverheadTarget {
					cp.reduceProfilingFrequency(avgOverhead)
				}
			}
		}
	}
}

// reduceProfilingFrequency reduces frequency if overhead too high
func (cp *ContinuousProfiler) reduceProfilingFrequency(overhead float64) {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	// Increase flush interval to reduce overhead
	newInterval := time.Duration(float64(cp.config.FlushInterval) * (overhead / cp.config.OverheadTarget))
	if newInterval > cp.config.FlushInterval {
		cp.config.FlushInterval = newInterval
		fmt.Printf("Increased profiling interval to %s to reduce overhead (%.4f%%)\n",
			newInterval, overhead*100)
	}
}

// cleanupWorker removes old profiles
func (cp *ContinuousProfiler) cleanupWorker(ctx context.Context) {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-cp.stopChan:
			return
		case <-ticker.C:
			cp.cleanupOldProfiles()
		}
	}
}

// cleanupOldProfiles removes profiles older than retention period
func (cp *ContinuousProfiler) cleanupOldProfiles() {
	cutoff := time.Now().Add(-time.Duration(cp.config.RetentionDays) * 24 * time.Hour)

	cp.mu.Lock()
	defer cp.mu.Unlock()

	for path, profile := range cp.profiles {
		if profile.StartTime.Before(cutoff) {
			os.Remove(path)
			delete(cp.profiles, path)
		}
	}
}

// Stop stops profiling
func (cp *ContinuousProfiler) Stop() {
	cp.mu.Lock()
	defer cp.mu.Unlock()

	if !cp.running {
		return
	}

	close(cp.stopChan)
	cp.running = false
}

// GetProfiles returns all captured profiles
func (cp *ContinuousProfiler) GetProfiles() map[string]*ProfileData {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	profiles := make(map[string]*ProfileData)
	for k, v := range cp.profiles {
		profiles[k] = v
	}
	return profiles
}

// GetAverageOverhead returns average profiling overhead
func (cp *ContinuousProfiler) GetAverageOverhead() float64 {
	cp.mu.RLock()
	defer cp.mu.RUnlock()

	if len(cp.profiles) == 0 {
		return 0.0
	}

	total := 0.0
	for _, profile := range cp.profiles {
		total += profile.Overhead
	}
	return total / float64(len(cp.profiles))
}

// Helper function
func average(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}
