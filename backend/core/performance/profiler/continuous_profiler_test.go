package profiler

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestContinuousProfiler(t *testing.T) {
	tmpDir := t.TempDir()

	config := ProfilerConfig{
		SamplingRate:  100,
		ProfileTypes:  []string{"cpu", "memory"},
		OutputDir:     tmpDir,
		RetentionDays: 1,
		OverheadTarget: 0.02,
		FlushInterval: 5 * time.Second,
	}

	profiler := NewContinuousProfiler(config)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Start profiler
	if err := profiler.Start(ctx); err != nil {
		t.Fatalf("Failed to start profiler: %v", err)
	}

	// Let it run for a bit
	time.Sleep(8 * time.Second)

	// Stop profiler
	profiler.Stop()

	// Check that profiles were created
	profiles := profiler.GetProfiles()
	if len(profiles) == 0 {
		t.Error("No profiles were created")
	}

	// Check overhead
	avgOverhead := profiler.GetAverageOverhead()
	if avgOverhead > config.OverheadTarget {
		t.Errorf("Profiling overhead %.4f%% exceeds target %.4f%%",
			avgOverhead*100, config.OverheadTarget*100)
	}

	t.Logf("Created %d profiles with average overhead %.4f%%",
		len(profiles), avgOverhead*100)
}

func TestProfileRetention(t *testing.T) {
	tmpDir := t.TempDir()

	config := ProfilerConfig{
		SamplingRate:  100,
		ProfileTypes:  []string{"memory"},
		OutputDir:     tmpDir,
		RetentionDays: 1,
		FlushInterval: 1 * time.Second,
	}

	profiler := NewContinuousProfiler(config)

	// Create some old profile files
	oldFile := tmpDir + "/memory-old.pprof"
	os.WriteFile(oldFile, []byte("test"), 0644)
	os.Chtimes(oldFile, time.Now().Add(-48*time.Hour), time.Now().Add(-48*time.Hour))

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	profiler.Start(ctx)
	time.Sleep(2 * time.Second)
	profiler.Stop()

	// Old file should be cleaned up
	if _, err := os.Stat(oldFile); !os.IsNotExist(err) {
		t.Error("Old profile file was not cleaned up")
	}
}

func BenchmarkProfilingOverhead(b *testing.B) {
	tmpDir := b.TempDir()

	config := ProfilerConfig{
		SamplingRate:  100,
		ProfileTypes:  []string{"cpu"},
		OutputDir:     tmpDir,
		FlushInterval: 60 * time.Second,
	}

	profiler := NewContinuousProfiler(config)
	ctx := context.Background()
	profiler.Start(ctx)
	defer profiler.Stop()

	// Run workload
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Simulated workload
		sum := 0
		for j := 0; j < 1000; j++ {
			sum += j * j
		}
	}
}
