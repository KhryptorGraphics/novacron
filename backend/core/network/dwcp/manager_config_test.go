package dwcp

import (
	"sync"
	"testing"

	"go.uber.org/zap"
)

// TestManagerGetConfig verifies GetConfig returns a proper heap-allocated copy
func TestManagerGetConfig(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	config.Transport.MinStreams = 50

	logger, _ := zap.NewDevelopment()
	manager, err := NewManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	// Get config copy
	configCopy := manager.GetConfig()

	// Verify copy is not nil
	if configCopy == nil {
		t.Fatal("GetConfig returned nil")
	}

	// Verify values match
	if configCopy.Enabled != config.Enabled {
		t.Errorf("Config.Enabled mismatch: got %v, want %v", configCopy.Enabled, config.Enabled)
	}
	if configCopy.Transport.MinStreams != config.Transport.MinStreams {
		t.Errorf("Config.Transport.MinStreams mismatch: got %v, want %v", configCopy.Transport.MinStreams, config.Transport.MinStreams)
	}

	// Modify the returned copy
	configCopy.Enabled = false
	configCopy.Transport.MinStreams = 100

	// Get another copy and verify original is unchanged
	configCopy2 := manager.GetConfig()
	if configCopy2.Enabled != true {
		t.Error("Original config was modified through returned copy")
	}
	if configCopy2.Transport.MinStreams != 50 {
		t.Error("Original config.Transport.MinStreams was modified through returned copy")
	}
}

// TestManagerGetConfigConcurrent verifies GetConfig is thread-safe
func TestManagerGetConfigConcurrent(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true

	logger, _ := zap.NewDevelopment()
	manager, err := NewManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	// Run concurrent GetConfig calls
	const numGoroutines = 100
	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func() {
			defer wg.Done()
			cfg := manager.GetConfig()
			if cfg == nil {
				t.Error("GetConfig returned nil in concurrent access")
			}
			if !cfg.Enabled {
				t.Error("Config.Enabled changed during concurrent access")
			}
		}()
	}

	wg.Wait()
}

// TestManagerGetConfigRaceCondition uses race detector to verify thread safety
// Run with: go test -race -run TestManagerGetConfigRaceCondition
func TestManagerGetConfigRaceCondition(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	config.Transport.MinStreams = 32

	logger, _ := zap.NewDevelopment()
	manager, err := NewManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	var wg sync.WaitGroup

	// Concurrent readers
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				cfg := manager.GetConfig()
				// Read all fields to trigger potential races
				_ = cfg.Enabled
				_ = cfg.Version
				_ = cfg.Transport.MinStreams
				_ = cfg.Transport.CongestionAlgorithm
				_ = cfg.Compression.Algorithm
				_ = cfg.Prediction.ModelType
				_ = cfg.Sync.ConflictResolution
				_ = cfg.Consensus.Algorithm
			}
		}()
	}

	wg.Wait()
}

// TestManagerGetConfigMemoryIndependence verifies returned config is independent
func TestManagerGetConfigMemoryIndependence(t *testing.T) {
	config := DefaultConfig()
	config.Enabled = true
	config.Transport.MinStreams = 64
	config.Transport.CongestionAlgorithm = "bbr"
	config.Compression.Algorithm = "zstd"

	logger, _ := zap.NewDevelopment()
	manager, err := NewManager(config, logger)
	if err != nil {
		t.Fatalf("Failed to create manager: %v", err)
	}

	// Get two independent copies
	copy1 := manager.GetConfig()
	copy2 := manager.GetConfig()

	// Verify they have the same values
	if copy1.Transport.MinStreams != copy2.Transport.MinStreams {
		t.Error("Two copies have different values")
	}

	// Modify copy1
	copy1.Enabled = false
	copy1.Transport.MinStreams = 128
	copy1.Transport.CongestionAlgorithm = "cubic"
	copy1.Compression.Algorithm = "lz4"

	// Verify copy2 is unaffected
	if !copy2.Enabled {
		t.Error("Modifying copy1 affected copy2.Enabled")
	}
	if copy2.Transport.MinStreams != 64 {
		t.Error("Modifying copy1 affected copy2.Transport.MinStreams")
	}
	if copy2.Transport.CongestionAlgorithm != "bbr" {
		t.Error("Modifying copy1 affected copy2.Transport.CongestionAlgorithm")
	}
	if copy2.Compression.Algorithm != "zstd" {
		t.Error("Modifying copy1 affected copy2.Compression.Algorithm")
	}
}

// BenchmarkManagerGetConfig measures GetConfig performance
func BenchmarkManagerGetConfig(b *testing.B) {
	config := DefaultConfig()
	logger, _ := zap.NewDevelopment()
	manager, err := NewManager(config, logger)
	if err != nil {
		b.Fatalf("Failed to create manager: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = manager.GetConfig()
	}
}

// BenchmarkManagerGetConfigConcurrent measures concurrent GetConfig performance
func BenchmarkManagerGetConfigConcurrent(b *testing.B) {
	config := DefaultConfig()
	logger, _ := zap.NewDevelopment()
	manager, err := NewManager(config, logger)
	if err != nil {
		b.Fatalf("Failed to create manager: %v", err)
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = manager.GetConfig()
		}
	})
}
