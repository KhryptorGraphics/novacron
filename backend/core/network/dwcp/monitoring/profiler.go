package monitoring

import (
	"context"
	"fmt"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

// Profiler manages performance profiling
type Profiler struct {
	mu sync.RWMutex

	// Profile types
	cpuProfile    *ProfileData
	memProfile    *ProfileData
	goroutineProfile *ProfileData
	blockProfile  *ProfileData
	mutexProfile  *ProfileData

	// Continuous profiling
	continuous    bool
	interval      time.Duration

	// Profile storage
	profiles      map[string]*ProfileData
}

// ProfileData stores profile data
type ProfileData struct {
	Type        ProfileType
	StartTime   time.Time
	EndTime     time.Time
	Duration    time.Duration
	Data        []byte
	Size        int64
	SampleRate  int
}

// ProfileType defines type of profile
type ProfileType int

const (
	ProfileCPU ProfileType = iota
	ProfileMemory
	ProfileGoroutine
	ProfileBlock
	ProfileMutex
)

func (pt ProfileType) String() string {
	switch pt {
	case ProfileCPU:
		return "cpu"
	case ProfileMemory:
		return "memory"
	case ProfileGoroutine:
		return "goroutine"
	case ProfileBlock:
		return "block"
	case ProfileMutex:
		return "mutex"
	default:
		return "unknown"
	}
}

// ProfileRequest represents a profiling request
type ProfileRequest struct {
	Type     ProfileType
	Duration time.Duration
	OnDemand bool
}

// ProfileComparison compares two profiles
type ProfileComparison struct {
	Before     *ProfileData
	After      *ProfileData
	Difference map[string]interface{}
}

// NewProfiler creates a new profiler
func NewProfiler() *Profiler {
	return &Profiler{
		profiles:   make(map[string]*ProfileData),
		interval:   1 * time.Minute,
		continuous: false,
	}
}

// StartCPUProfile starts CPU profiling
func (p *Profiler) StartCPUProfile(duration time.Duration) error {
	p.mu.Lock()
	defer p.mu.Unlock()

	profile := &ProfileData{
		Type:      ProfileCPU,
		StartTime: time.Now(),
		Duration:  duration,
	}

	// Start CPU profiling (simplified)
	runtime.SetCPUProfileRate(100) // 100 Hz
	pprof.StartCPUProfile(nil)

	// Stop after duration
	go func() {
		time.Sleep(duration)
		p.StopCPUProfile()
	}()

	p.cpuProfile = profile
	return nil
}

// StopCPUProfile stops CPU profiling
func (p *Profiler) StopCPUProfile() {
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.cpuProfile == nil {
		return
	}

	pprof.StopCPUProfile()
	p.cpuProfile.EndTime = time.Now()

	key := fmt.Sprintf("cpu-%d", time.Now().Unix())
	p.profiles[key] = p.cpuProfile
	p.cpuProfile = nil
}

// CaptureMemoryProfile captures memory profile
func (p *Profiler) CaptureMemoryProfile() (*ProfileData, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	profile := &ProfileData{
		Type:      ProfileMemory,
		StartTime: time.Now(),
	}

	// Trigger GC to get accurate stats
	runtime.GC()

	// Get memory profile
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	profile.EndTime = time.Now()
	profile.Size = int64(memStats.Alloc)

	key := fmt.Sprintf("memory-%d", time.Now().Unix())
	p.profiles[key] = profile

	return profile, nil
}

// CaptureGoroutineProfile captures goroutine profile
func (p *Profiler) CaptureGoroutineProfile() (*ProfileData, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	profile := &ProfileData{
		Type:      ProfileGoroutine,
		StartTime: time.Now(),
	}

	goroutineCount := runtime.NumGoroutine()
	profile.Size = int64(goroutineCount)
	profile.EndTime = time.Now()

	key := fmt.Sprintf("goroutine-%d", time.Now().Unix())
	p.profiles[key] = profile

	return profile, nil
}

// CaptureBlockProfile captures block profile
func (p *Profiler) CaptureBlockProfile() (*ProfileData, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	runtime.SetBlockProfileRate(1)

	profile := &ProfileData{
		Type:      ProfileBlock,
		StartTime: time.Now(),
	}

	profile.EndTime = time.Now()

	key := fmt.Sprintf("block-%d", time.Now().Unix())
	p.profiles[key] = profile

	return profile, nil
}

// CaptureMutexProfile captures mutex contention profile
func (p *Profiler) CaptureMutexProfile() (*ProfileData, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	runtime.SetMutexProfileFraction(1)

	profile := &ProfileData{
		Type:      ProfileMutex,
		StartTime: time.Now(),
	}

	profile.EndTime = time.Now()

	key := fmt.Sprintf("mutex-%d", time.Now().Unix())
	p.profiles[key] = profile

	return profile, nil
}

// StartContinuousProfiling starts continuous profiling
func (p *Profiler) StartContinuousProfiling(ctx context.Context) {
	p.mu.Lock()
	p.continuous = true
	p.mu.Unlock()

	ticker := time.NewTicker(p.interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			p.mu.Lock()
			p.continuous = false
			p.mu.Unlock()
			return
		case <-ticker.C:
			// Capture all profile types
			p.CaptureMemoryProfile()
			p.CaptureGoroutineProfile()
			p.CaptureBlockProfile()
			p.CaptureMutexProfile()
		}
	}
}

// CompareProfiles compares two profiles
func (p *Profiler) CompareProfiles(beforeKey, afterKey string) (*ProfileComparison, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	before, ok := p.profiles[beforeKey]
	if !ok {
		return nil, fmt.Errorf("before profile not found: %s", beforeKey)
	}

	after, ok := p.profiles[afterKey]
	if !ok {
		return nil, fmt.Errorf("after profile not found: %s", afterKey)
	}

	comparison := &ProfileComparison{
		Before:     before,
		After:      after,
		Difference: make(map[string]interface{}),
	}

	// Calculate differences
	comparison.Difference["size_delta"] = after.Size - before.Size
	comparison.Difference["time_delta"] = after.EndTime.Sub(before.EndTime)

	return comparison, nil
}

// GetProfile retrieves a profile
func (p *Profiler) GetProfile(key string) (*ProfileData, bool) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	profile, ok := p.profiles[key]
	return profile, ok
}

// ListProfiles lists all profiles
func (p *Profiler) ListProfiles() []string {
	p.mu.RLock()
	defer p.mu.RUnlock()

	keys := make([]string, 0, len(p.profiles))
	for key := range p.profiles {
		keys = append(keys, key)
	}
	return keys
}

// GetStatistics returns profiling statistics
func (p *Profiler) GetStatistics() map[string]interface{} {
	p.mu.RLock()
	defer p.mu.RUnlock()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	return map[string]interface{}{
		"total_profiles":   len(p.profiles),
		"continuous":       p.continuous,
		"goroutines":       runtime.NumGoroutine(),
		"memory_alloc":     memStats.Alloc,
		"memory_sys":       memStats.Sys,
		"gc_runs":          memStats.NumGC,
	}
}

// Cleanup removes old profiles
func (p *Profiler) Cleanup(maxAge time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()

	cutoff := time.Now().Add(-maxAge)

	for key, profile := range p.profiles {
		if profile.EndTime.Before(cutoff) {
			delete(p.profiles, key)
		}
	}
}
