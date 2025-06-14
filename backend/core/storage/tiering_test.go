package storage

import (
	"fmt"
	"testing"
	"time"
)

func TestTieringManager_CreateAndStart(t *testing.T) {
	config := DefaultTieringConfig()
	
	// Create mock storage service
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, err := NewTieringManager(config, storageService)
	if err != nil {
		t.Fatalf("Failed to create tiering manager: %v", err)
	}
	
	err = tieringManager.Start()
	if err != nil {
		t.Fatalf("Failed to start tiering manager: %v", err)
	}
	defer tieringManager.Stop()
	
	// Verify that all tiers are configured
	expectedTiers := []StorageTier{TierHot, TierWarm, TierCold, TierArchive}
	for _, tier := range expectedTiers {
		if _, exists := tieringManager.tiers[tier]; !exists {
			t.Errorf("Expected tier %s to be configured", tier)
		}
	}
}

func TestTieringManager_RecordAccess(t *testing.T) {
	config := DefaultTieringConfig()
	config.EvaluationInterval = 100 * time.Millisecond // Fast evaluation for testing
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, err := NewTieringManager(config, storageService)
	if err != nil {
		t.Fatalf("Failed to create tiering manager: %v", err)
	}
	
	err = tieringManager.Start()
	if err != nil {
		t.Fatalf("Failed to start tiering manager: %v", err)
	}
	defer tieringManager.Stop()
	
	volumeID := "test-volume-123"
	
	// Record some accesses
	tieringManager.RecordAccess(volumeID, "read", 1024)
	tieringManager.RecordAccess(volumeID, "write", 2048)
	tieringManager.RecordAccess(volumeID, "read", 512)
	
	// Get volume stats
	stats, err := tieringManager.GetVolumeStats(volumeID)
	if err != nil {
		t.Fatalf("Failed to get volume stats: %v", err)
	}
	
	if stats.AccessCount != 3 {
		t.Errorf("Expected access count 3, got %d", stats.AccessCount)
	}
	
	if stats.TotalReads != 2 {
		t.Errorf("Expected total reads 2, got %d", stats.TotalReads)
	}
	
	if stats.TotalWrites != 1 {
		t.Errorf("Expected total writes 1, got %d", stats.TotalWrites)
	}
	
	if stats.BytesRead != 1536 { // 1024 + 512
		t.Errorf("Expected bytes read 1536, got %d", stats.BytesRead)
	}
	
	if stats.BytesWritten != 2048 {
		t.Errorf("Expected bytes written 2048, got %d", stats.BytesWritten)
	}
	
	if stats.CurrentTier != TierHot {
		t.Errorf("Expected current tier %s, got %s", TierHot, stats.CurrentTier)
	}
}

func TestTieringManager_CalculateAccessFrequency(t *testing.T) {
	config := DefaultTieringConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, err := NewTieringManager(config, storageService)
	if err != nil {
		t.Fatalf("Failed to create tiering manager: %v", err)
	}
	
	volumeID := "test-volume-frequency"
	
	// Create volume stats with specific hourly access pattern
	stats := &VolumeAccessStats{
		VolumeID:     volumeID,
		AccessCount:  100,
		CurrentTier:  TierHot,
		HourlyAccess: [24]int{5, 3, 2, 1, 0, 0, 0, 0, 0, 0, 8, 10, 12, 15, 20, 18, 10, 8, 5, 3, 2, 1, 0, 0},
	}
	
	frequency := tieringManager.calculateAccessFrequency(stats)
	
	// Sum of HourlyAccess array divided by 24
	expectedSum := 0
	for _, count := range stats.HourlyAccess {
		expectedSum += count
	}
	expectedFrequency := float64(expectedSum) / 24.0
	
	if frequency != expectedFrequency {
		t.Errorf("Expected frequency %.2f, got %.2f", expectedFrequency, frequency)
	}
}

func TestTieringManager_ShouldPromote(t *testing.T) {
	config := DefaultTieringConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, err := NewTieringManager(config, storageService)
	if err != nil {
		t.Fatalf("Failed to create tiering manager: %v", err)
	}
	
	// Test promotion from Cold to Warm
	stats := &VolumeAccessStats{
		VolumeID:     "test-volume",
		AccessCount:  50, // Above threshold
		CurrentTier:  TierCold,
		HourlyAccess: [24]int{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // 2.0 accesses/hour
	}
	
	accessFrequency := tieringManager.calculateAccessFrequency(stats)
	targetTier := tieringManager.shouldPromote(stats, accessFrequency)
	
	if targetTier != TierWarm {
		t.Errorf("Expected promotion to %s, got %s", TierWarm, targetTier)
	}
	
	// Test no promotion due to low access count
	stats.AccessCount = 2 // Below threshold
	targetTier = tieringManager.shouldPromote(stats, accessFrequency)
	
	if targetTier != "" {
		t.Errorf("Expected no promotion due to low access count, got %s", targetTier)
	}
	
	// Test no promotion due to low frequency
	stats.AccessCount = 50
	stats.HourlyAccess = [24]int{} // 0.0 accesses/hour
	lowFrequency := tieringManager.calculateAccessFrequency(stats)
	targetTier = tieringManager.shouldPromote(stats, lowFrequency)
	
	if targetTier != "" {
		t.Errorf("Expected no promotion due to low frequency, got %s", targetTier)
	}
}

func TestTieringManager_ShouldDemote(t *testing.T) {
	config := DefaultTieringConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, err := NewTieringManager(config, storageService)
	if err != nil {
		t.Fatalf("Failed to create tiering manager: %v", err)
	}
	
	// Test demotion from Hot to Warm due to inactivity
	oldTime := time.Now().Add(-2 * time.Hour) // 2 hours ago
	stats := &VolumeAccessStats{
		VolumeID:       "test-volume",
		CurrentTier:    TierHot,
		LastAccessed:   oldTime,
		TierChangeTime: time.Now().Add(-1 * time.Hour), // Moved to hot 1 hour ago
	}
	
	tierConfig := tieringManager.tiers[TierHot]
	targetTier := tieringManager.shouldDemote(stats, tierConfig)
	
	if targetTier != TierWarm {
		t.Errorf("Expected demotion to %s, got %s", TierWarm, targetTier)
	}
	
	// Test no demotion due to recent access
	stats.LastAccessed = time.Now().Add(-30 * time.Minute) // 30 minutes ago
	targetTier = tieringManager.shouldDemote(stats, tierConfig)
	
	if targetTier != "" {
		t.Errorf("Expected no demotion due to recent access, got %s", targetTier)
	}
	
	// Test no demotion due to minimum storage duration
	stats.LastAccessed = oldTime
	stats.TierChangeTime = time.Now().Add(-30 * time.Minute) // Recently moved to hot
	targetTier = tieringManager.shouldDemote(stats, tierConfig)
	
	if targetTier != "" {
		t.Errorf("Expected no demotion due to minimum storage duration, got %s", targetTier)
	}
}

func TestTieringManager_GetTierStats(t *testing.T) {
	config := DefaultTieringConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, err := NewTieringManager(config, storageService)
	if err != nil {
		t.Fatalf("Failed to create tiering manager: %v", err)
	}
	
	// Add some volume stats
	tieringManager.accessStats["volume1"] = &VolumeAccessStats{
		VolumeID:    "volume1",
		CurrentTier: TierHot,
	}
	tieringManager.accessStats["volume2"] = &VolumeAccessStats{
		VolumeID:    "volume2",
		CurrentTier: TierHot,
	}
	tieringManager.accessStats["volume3"] = &VolumeAccessStats{
		VolumeID:    "volume3",
		CurrentTier: TierWarm,
	}
	
	tierStats := tieringManager.GetTierStats()
	
	// Check that all tiers are represented
	expectedTiers := []StorageTier{TierHot, TierWarm, TierCold, TierArchive}
	for _, tier := range expectedTiers {
		if _, exists := tierStats[tier]; !exists {
			t.Errorf("Expected tier %s in stats", tier)
		}
	}
	
	// Check volume counts
	if tierStats[TierHot].VolumeCount != 2 {
		t.Errorf("Expected 2 volumes in hot tier, got %d", tierStats[TierHot].VolumeCount)
	}
	
	if tierStats[TierWarm].VolumeCount != 1 {
		t.Errorf("Expected 1 volume in warm tier, got %d", tierStats[TierWarm].VolumeCount)
	}
	
	if tierStats[TierCold].VolumeCount != 0 {
		t.Errorf("Expected 0 volumes in cold tier, got %d", tierStats[TierCold].VolumeCount)
	}
}

func TestTieringManager_CalculateTierCosts(t *testing.T) {
	config := DefaultTieringConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, err := NewTieringManager(config, storageService)
	if err != nil {
		t.Fatalf("Failed to create tiering manager: %v", err)
	}
	
	tierCosts := tieringManager.CalculateTierCosts()
	
	// Check that all tiers have cost calculations
	expectedTiers := []StorageTier{TierHot, TierWarm, TierCold, TierArchive}
	for _, tier := range expectedTiers {
		if _, exists := tierCosts[tier]; !exists {
			t.Errorf("Expected cost calculation for tier %s", tier)
		}
	}
	
	// With no volumes, all costs should be 0
	for tier, cost := range tierCosts {
		if cost != 0 {
			t.Errorf("Expected 0 cost for empty tier %s, got %.2f", tier, cost)
		}
	}
}

func TestDefaultTieringConfig(t *testing.T) {
	config := DefaultTieringConfig()
	
	// Verify default values
	if config.EvaluationInterval != 1*time.Hour {
		t.Errorf("Expected evaluation interval 1 hour, got %v", config.EvaluationInterval)
	}
	
	if !config.AutoTieringEnabled {
		t.Error("Expected auto tiering to be enabled by default")
	}
	
	if config.DefaultTier != TierHot {
		t.Errorf("Expected default tier %s, got %s", TierHot, config.DefaultTier)
	}
	
	// Verify all tiers are configured
	expectedTiers := []StorageTier{TierHot, TierWarm, TierCold, TierArchive}
	if len(config.Tiers) != len(expectedTiers) {
		t.Errorf("Expected %d tiers, got %d", len(expectedTiers), len(config.Tiers))
	}
	
	// Verify tier order by cost (hot should be most expensive)
	for i := 0; i < len(config.Tiers)-1; i++ {
		if config.Tiers[i].CostPerGBMonth <= config.Tiers[i+1].CostPerGBMonth {
			t.Errorf("Expected tier costs to decrease: %s (%.4f) should be more expensive than %s (%.4f)",
				config.Tiers[i].Name, config.Tiers[i].CostPerGBMonth,
				config.Tiers[i+1].Name, config.Tiers[i+1].CostPerGBMonth)
		}
	}
}

// Benchmark tests for tiering
func BenchmarkTieringManager_RecordAccess(b *testing.B) {
	config := DefaultTieringConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, _ := NewTieringManager(config, storageService)
	tieringManager.Start()
	defer tieringManager.Stop()
	
	volumeID := "bench-volume"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tieringManager.RecordAccess(volumeID, "read", 1024)
	}
}

func BenchmarkTieringManager_CalculateAccessFrequency(b *testing.B) {
	config := DefaultTieringConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, _ := NewTieringManager(config, storageService)
	
	stats := &VolumeAccessStats{
		VolumeID:     "bench-volume",
		HourlyAccess: [24]int{10, 8, 5, 3, 1, 0, 0, 0, 2, 5, 8, 12, 15, 18, 20, 15, 12, 8, 6, 4, 2, 1, 0, 0},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tieringManager.calculateAccessFrequency(stats)
	}
}

func BenchmarkTieringManager_GetTierStats(b *testing.B) {
	config := DefaultTieringConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	tieringManager, _ := NewTieringManager(config, storageService)
	
	// Add many volumes for benchmarking
	for i := 0; i < 1000; i++ {
		volumeID := fmt.Sprintf("volume-%d", i)
		tier := StorageTier([]StorageTier{TierHot, TierWarm, TierCold, TierArchive}[i%4])
		tieringManager.accessStats[volumeID] = &VolumeAccessStats{
			VolumeID:    volumeID,
			CurrentTier: tier,
		}
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tieringManager.GetTierStats()
	}
}

