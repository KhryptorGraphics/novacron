package tiering

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// MockStorageDriver implements a mock storage driver for testing
type MockStorageDriver struct {
	volumes      map[string]*storage.VolumeInfo
	data         map[string][]byte
	initialized  bool
	mu           sync.RWMutex
	failNextOp   bool
	readLatency  time.Duration
	writeLatency time.Duration
}

func NewMockStorageDriver() *MockStorageDriver {
	return &MockStorageDriver{
		volumes: make(map[string]*storage.VolumeInfo),
		data:    make(map[string][]byte),
	}
}

func (m *MockStorageDriver) Initialize() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.failNextOp {
		m.failNextOp = false
		return fmt.Errorf("initialization failed")
	}
	
	m.initialized = true
	return nil
}

func (m *MockStorageDriver) Shutdown() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.initialized = false
	return nil
}

func (m *MockStorageDriver) CreateVolume(ctx context.Context, name string, sizeBytes int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.failNextOp {
		m.failNextOp = false
		return fmt.Errorf("create volume failed")
	}
	
	if _, exists := m.volumes[name]; exists {
		return fmt.Errorf("volume already exists")
	}
	
	m.volumes[name] = &storage.VolumeInfo{
		Name:      name,
		Size:      sizeBytes,
		CreatedAt: time.Now(),
		Status:    "available",
	}
	m.data[name] = make([]byte, sizeBytes)
	
	return nil
}

func (m *MockStorageDriver) DeleteVolume(ctx context.Context, name string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.failNextOp {
		m.failNextOp = false
		return fmt.Errorf("delete volume failed")
	}
	
	delete(m.volumes, name)
	delete(m.data, name)
	return nil
}

func (m *MockStorageDriver) GetVolumeInfo(ctx context.Context, name string) (*storage.VolumeInfo, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.failNextOp {
		m.failNextOp = false
		return nil, fmt.Errorf("get volume info failed")
	}
	
	info, exists := m.volumes[name]
	if !exists {
		return nil, fmt.Errorf("volume not found")
	}
	
	return info, nil
}

func (m *MockStorageDriver) ListVolumes(ctx context.Context) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.failNextOp {
		m.failNextOp = false
		return nil, fmt.Errorf("list volumes failed")
	}
	
	names := make([]string, 0, len(m.volumes))
	for name := range m.volumes {
		names = append(names, name)
	}
	
	return names, nil
}

func (m *MockStorageDriver) ReadVolume(ctx context.Context, name string, offset int64, size int) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	
	if m.readLatency > 0 {
		time.Sleep(m.readLatency)
	}
	
	if m.failNextOp {
		m.failNextOp = false
		return nil, fmt.Errorf("read volume failed")
	}
	
	data, exists := m.data[name]
	if !exists {
		return nil, fmt.Errorf("volume not found")
	}
	
	if offset+int64(size) > int64(len(data)) {
		return nil, fmt.Errorf("read beyond volume size")
	}
	
	result := make([]byte, size)
	copy(result, data[offset:offset+int64(size)])
	
	return result, nil
}

func (m *MockStorageDriver) WriteVolume(ctx context.Context, name string, offset int64, data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.writeLatency > 0 {
		time.Sleep(m.writeLatency)
	}
	
	if m.failNextOp {
		m.failNextOp = false
		return fmt.Errorf("write volume failed")
	}
	
	volumeData, exists := m.data[name]
	if !exists {
		return fmt.Errorf("volume not found")
	}
	
	if offset+int64(len(data)) > int64(len(volumeData)) {
		return fmt.Errorf("write beyond volume size")
	}
	
	copy(volumeData[offset:], data)
	
	return nil
}

func (m *MockStorageDriver) ResizeVolume(ctx context.Context, name string, newSizeBytes int64) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.failNextOp {
		m.failNextOp = false
		return fmt.Errorf("resize volume failed")
	}
	
	info, exists := m.volumes[name]
	if !exists {
		return fmt.Errorf("volume not found")
	}
	
	info.Size = newSizeBytes
	
	// Resize data array
	oldData := m.data[name]
	newData := make([]byte, newSizeBytes)
	copy(newData, oldData)
	m.data[name] = newData
	
	return nil
}

func (m *MockStorageDriver) AttachVolume(ctx context.Context, volumeID, nodeID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.failNextOp {
		m.failNextOp = false
		return fmt.Errorf("attach volume failed")
	}
	
	if _, exists := m.volumes[volumeID]; !exists {
		return fmt.Errorf("volume not found")
	}
	
	return nil
}

func (m *MockStorageDriver) DetachVolume(ctx context.Context, volumeID, nodeID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	
	if m.failNextOp {
		m.failNextOp = false
		return fmt.Errorf("detach volume failed")
	}
	
	if _, exists := m.volumes[volumeID]; !exists {
		return fmt.Errorf("volume not found")
	}
	
	return nil
}

func (m *MockStorageDriver) GetCapabilities() storage.DriverCapabilities {
	return storage.DriverCapabilities{
		SupportsSnapshots:     true,
		SupportsReplication:   false,
		SupportsEncryption:    false,
		SupportsCompression:   false,
		SupportsDeduplication: false,
		MaxVolumeSize:        1024 * 1024 * 1024 * 1024, // 1TB
		MinVolumeSize:        1024 * 1024,                // 1MB
	}
}

func (m *MockStorageDriver) CreateSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return fmt.Errorf("not implemented")
}

func (m *MockStorageDriver) DeleteSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return fmt.Errorf("not implemented")
}

func (m *MockStorageDriver) RestoreSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return fmt.Errorf("not implemented")
}

// Test functions

func TestTierManagerBasicOperations(t *testing.T) {
	tm := NewTierManager()
	
	// Test adding tiers
	hotDriver := NewMockStorageDriver()
	warmDriver := NewMockStorageDriver()
	coldDriver := NewMockStorageDriver()
	
	err := tm.AddTier(TierHot, hotDriver, "Hot-SSD", 0.1, 100)
	if err != nil {
		t.Fatalf("Failed to add hot tier: %v", err)
	}
	
	err = tm.AddTier(TierWarm, warmDriver, "Warm-HDD", 0.05, 500)
	if err != nil {
		t.Fatalf("Failed to add warm tier: %v", err)
	}
	
	err = tm.AddTier(TierCold, coldDriver, "Cold-Object", 0.01, 10000)
	if err != nil {
		t.Fatalf("Failed to add cold tier: %v", err)
	}
	
	// Test duplicate tier
	err = tm.AddTier(TierHot, hotDriver, "Another-Hot", 0.2, 200)
	if err == nil {
		t.Fatal("Expected error when adding duplicate tier")
	}
	
	// Test initialization
	err = tm.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize tier manager: %v", err)
	}
	
	// Test double initialization
	err = tm.Initialize()
	if err == nil {
		t.Fatal("Expected error on double initialization")
	}
}

func TestTierManagerPolicies(t *testing.T) {
	tm := NewTierManager()
	
	// Add tiers
	hotDriver := NewMockStorageDriver()
	warmDriver := NewMockStorageDriver()
	coldDriver := NewMockStorageDriver()
	
	tm.AddTier(TierHot, hotDriver, "Hot", 0.1, 100)
	tm.AddTier(TierWarm, warmDriver, "Warm", 0.05, 500)
	tm.AddTier(TierCold, coldDriver, "Cold", 0.01, 10000)
	
	// Add policies
	policyExecuted := false
	tm.AddPolicy("TestPolicy", func(stats *VolumeStats) (bool, TierLevel) {
		policyExecuted = true
		if stats.AccessFrequency > 1.0 {
			return true, TierHot
		}
		return false, stats.CurrentTier
	}, 100)
	
	// Initialize
	err := tm.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize: %v", err)
	}
	
	// Create test volumes
	ctx := context.Background()
	hotDriver.CreateVolume(ctx, "test-volume", 1024*1024*100) // 100MB
	
	// Record access
	tm.RecordVolumeAccess("test-volume")
	
	// Evaluate policies
	err = tm.EvaluateAndApplyTieringPolicies()
	if err != nil {
		t.Fatalf("Failed to evaluate policies: %v", err)
	}
	
	if !policyExecuted {
		t.Fatal("Policy was not executed")
	}
}

func TestVolumeAccessTracking(t *testing.T) {
	tm := NewTierManager()
	
	// Add a tier
	driver := NewMockStorageDriver()
	tm.AddTier(TierHot, driver, "Hot", 0.1, 100)
	tm.Initialize()
	
	// Record multiple accesses
	volumeName := "test-volume"
	for i := 0; i < 10; i++ {
		tm.RecordVolumeAccess(volumeName)
		time.Sleep(10 * time.Millisecond) // Small delay between accesses
	}
	
	// Check that access was recorded
	tm.mu.RLock()
	stats, exists := tm.volumeUsage[volumeName]
	tm.mu.RUnlock()
	
	if !exists {
		t.Fatal("Volume stats not found")
	}
	
	if stats.Name != volumeName {
		t.Errorf("Expected volume name %s, got %s", volumeName, stats.Name)
	}
	
	if stats.LastAccessed.IsZero() {
		t.Error("Last accessed time not set")
	}
	
	if stats.AccessFrequency <= 0 {
		t.Error("Access frequency not calculated")
	}
}

func TestVolumePinning(t *testing.T) {
	tm := NewTierManager()
	
	// Add tiers
	hotDriver := NewMockStorageDriver()
	warmDriver := NewMockStorageDriver()
	
	tm.AddTier(TierHot, hotDriver, "Hot", 0.1, 100)
	tm.AddTier(TierWarm, warmDriver, "Warm", 0.05, 500)
	tm.Initialize()
	
	// Create a volume in warm tier
	ctx := context.Background()
	warmDriver.CreateVolume(ctx, "test-volume", 1024*1024)
	
	// Track the volume
	tm.RecordVolumeAccess("test-volume")
	
	// Pin to hot tier
	err := tm.PinVolume("test-volume", TierHot)
	if err != nil {
		t.Fatalf("Failed to pin volume: %v", err)
	}
	
	// Check that volume is pinned
	tm.mu.RLock()
	stats := tm.volumeUsage["test-volume"]
	tm.mu.RUnlock()
	
	if !stats.Pinned {
		t.Error("Volume not marked as pinned")
	}
	
	if stats.PinnedTier != TierHot {
		t.Errorf("Expected pinned tier %d, got %d", TierHot, stats.PinnedTier)
	}
	
	// Unpin
	err = tm.UnpinVolume("test-volume")
	if err != nil {
		t.Fatalf("Failed to unpin volume: %v", err)
	}
	
	tm.mu.RLock()
	stats = tm.volumeUsage["test-volume"]
	tm.mu.RUnlock()
	
	if stats.Pinned {
		t.Error("Volume still marked as pinned after unpinning")
	}
}

func TestBackgroundWorker(t *testing.T) {
	tm := NewTierManager()
	
	// Add tiers
	hotDriver := NewMockStorageDriver()
	warmDriver := NewMockStorageDriver()
	
	tm.AddTier(TierHot, hotDriver, "Hot", 0.1, 100)
	tm.AddTier(TierWarm, warmDriver, "Warm", 0.05, 500)
	tm.Initialize()
	
	// Add a policy that will trigger
	policyCalled := 0
	tm.AddPolicy("BackgroundPolicy", func(stats *VolumeStats) (bool, TierLevel) {
		policyCalled++
		return false, stats.CurrentTier
	}, 100)
	
	// Start background worker with short interval
	err := tm.StartBackgroundWorker(100 * time.Millisecond)
	if err != nil {
		t.Fatalf("Failed to start background worker: %v", err)
	}
	
	// Create a volume to evaluate
	ctx := context.Background()
	hotDriver.CreateVolume(ctx, "test-volume", 1024*1024)
	tm.RecordVolumeAccess("test-volume")
	
	// Wait for background worker to run
	time.Sleep(250 * time.Millisecond)
	
	// Stop background worker
	tm.StopBackgroundWorker()
	
	if policyCalled == 0 {
		t.Error("Background worker did not evaluate policies")
	}
}

func TestDefaultPolicies(t *testing.T) {
	tm := NewTierManager()
	
	// Add tiers
	hotDriver := NewMockStorageDriver()
	warmDriver := NewMockStorageDriver()
	coldDriver := NewMockStorageDriver()
	
	tm.AddTier(TierHot, hotDriver, "Hot", 0.1, 100)
	tm.AddTier(TierWarm, warmDriver, "Warm", 0.05, 500)
	tm.AddTier(TierCold, coldDriver, "Cold", 0.01, 10000)
	tm.Initialize()
	
	// Create default policies
	tm.CreateDefaultAgingPolicy()
	tm.CreateCostOptimizationPolicy()
	
	// Test aging policy with hot data
	hotStats := &VolumeStats{
		Name:            "hot-volume",
		CurrentTier:     TierWarm,
		AccessFrequency: 2.0, // More than 1 access per day
	}
	
	shouldMove, targetTier := tm.policies[0].EvaluateFunc(hotStats)
	if !shouldMove || targetTier != TierHot {
		t.Error("Aging policy should move frequently accessed data to hot tier")
	}
	
	// Test aging policy with cold data
	coldStats := &VolumeStats{
		Name:            "cold-volume",
		CurrentTier:     TierHot,
		AccessFrequency: 0.01, // Very infrequent access
	}
	
	shouldMove, targetTier = tm.policies[0].EvaluateFunc(coldStats)
	if !shouldMove || targetTier != TierCold {
		t.Error("Aging policy should move infrequently accessed data to cold tier")
	}
}

func TestGetTierStats(t *testing.T) {
	tm := NewTierManager()
	
	// Add tiers
	hotDriver := NewMockStorageDriver()
	warmDriver := NewMockStorageDriver()
	
	tm.AddTier(TierHot, hotDriver, "Hot-SSD", 0.1, 100)
	tm.AddTier(TierWarm, warmDriver, "Warm-HDD", 0.05, 500)
	tm.Initialize()
	
	// Get stats
	stats := tm.GetTierStats()
	
	if len(stats) != 2 {
		t.Errorf("Expected 2 tiers in stats, got %d", len(stats))
	}
	
	hotStats, exists := stats[TierHot]
	if !exists {
		t.Fatal("Hot tier stats not found")
	}
	
	if hotStats["name"] != "Hot-SSD" {
		t.Errorf("Expected tier name 'Hot-SSD', got %v", hotStats["name"])
	}
	
	if hotStats["max_capacity_gb"] != int64(100) {
		t.Errorf("Expected max capacity 100GB, got %v", hotStats["max_capacity_gb"])
	}
	
	if hotStats["cost_per_gb_month"] != 0.1 {
		t.Errorf("Expected cost 0.1, got %v", hotStats["cost_per_gb_month"])
	}
}

func TestVolumeMigration(t *testing.T) {
	tm := NewTierManager()
	
	// Add tiers
	hotDriver := NewMockStorageDriver()
	warmDriver := NewMockStorageDriver()
	
	tm.AddTier(TierHot, hotDriver, "Hot", 0.1, 100)
	tm.AddTier(TierWarm, warmDriver, "Warm", 0.05, 500)
	tm.Initialize()
	
	// Create a volume in hot tier
	ctx := context.Background()
	testData := []byte("test data for migration")
	volumeName := "migrate-volume"
	
	err := hotDriver.CreateVolume(ctx, volumeName, int64(len(testData)))
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}
	
	err = hotDriver.WriteVolume(ctx, volumeName, 0, testData)
	if err != nil {
		t.Fatalf("Failed to write test data: %v", err)
	}
	
	// Setup metadata
	tm.mu.Lock()
	tm.volumeUsage[volumeName] = &VolumeStats{
		Name:        volumeName,
		CurrentTier: TierHot,
		SizeGB:      float64(len(testData)) / (1024 * 1024 * 1024),
	}
	tm.metadataStore.volumeMetadata[volumeName] = &VolumeMetadata{
		Name:        volumeName,
		CurrentTier: TierHot,
		TierNames:   make(map[TierLevel]string),
	}
	tm.mu.Unlock()
	
	// Migrate to warm tier
	err = tm.moveVolumeBetweenTiers(volumeName, TierHot, TierWarm)
	if err != nil {
		t.Fatalf("Failed to migrate volume: %v", err)
	}
	
	// Verify data was copied
	readData, err := warmDriver.ReadVolume(ctx, volumeName, 0, len(testData))
	if err != nil {
		t.Fatalf("Failed to read migrated data: %v", err)
	}
	
	if string(readData) != string(testData) {
		t.Error("Data not correctly migrated")
	}
	
	// Check metadata was updated
	tm.mu.RLock()
	metadata := tm.metadataStore.volumeMetadata[volumeName]
	tm.mu.RUnlock()
	
	if metadata.CurrentTier != TierWarm {
		t.Errorf("Metadata not updated, expected tier %d, got %d", TierWarm, metadata.CurrentTier)
	}
}

func TestConcurrentAccess(t *testing.T) {
	tm := NewTierManager()
	
	// Add tier
	driver := NewMockStorageDriver()
	tm.AddTier(TierHot, driver, "Hot", 0.1, 100)
	tm.Initialize()
	
	// Concurrent access recording
	var wg sync.WaitGroup
	numGoroutines := 10
	accessesPerGoroutine := 100
	
	wg.Add(numGoroutines)
	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			volumeName := fmt.Sprintf("volume-%d", id)
			
			for j := 0; j < accessesPerGoroutine; j++ {
				tm.RecordVolumeAccess(volumeName)
			}
		}(i)
	}
	
	wg.Wait()
	
	// Verify all accesses were recorded
	tm.mu.RLock()
	defer tm.mu.RUnlock()
	
	if len(tm.volumeUsage) != numGoroutines {
		t.Errorf("Expected %d volumes, got %d", numGoroutines, len(tm.volumeUsage))
	}
	
	for i := 0; i < numGoroutines; i++ {
		volumeName := fmt.Sprintf("volume-%d", i)
		stats, exists := tm.volumeUsage[volumeName]
		if !exists {
			t.Errorf("Volume %s not found", volumeName)
			continue
		}
		
		if stats.Name != volumeName {
			t.Errorf("Volume name mismatch: expected %s, got %s", volumeName, stats.Name)
		}
	}
}

func TestErrorHandling(t *testing.T) {
	tm := NewTierManager()
	
	// Test initialization without tiers
	err := tm.Initialize()
	if err == nil {
		t.Error("Expected error when initializing without tiers")
	}
	
	// Add tier with failing driver
	failDriver := NewMockStorageDriver()
	failDriver.failNextOp = true
	
	tm.AddTier(TierHot, failDriver, "Failing", 0.1, 100)
	
	err = tm.Initialize()
	if err == nil {
		t.Error("Expected error when driver initialization fails")
	}
	
	// Test operations without initialization
	tm2 := NewTierManager()
	driver := NewMockStorageDriver()
	tm2.AddTier(TierHot, driver, "Hot", 0.1, 100)
	
	err = tm2.StartBackgroundWorker(time.Second)
	if err == nil {
		t.Error("Expected error when starting background worker without initialization")
	}
}

// Benchmark tests

func BenchmarkRecordVolumeAccess(b *testing.B) {
	tm := NewTierManager()
	driver := NewMockStorageDriver()
	tm.AddTier(TierHot, driver, "Hot", 0.1, 100)
	tm.Initialize()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tm.RecordVolumeAccess("benchmark-volume")
	}
}

func BenchmarkEvaluatePolicies(b *testing.B) {
	tm := NewTierManager()
	
	// Add tiers
	hotDriver := NewMockStorageDriver()
	warmDriver := NewMockStorageDriver()
	coldDriver := NewMockStorageDriver()
	
	tm.AddTier(TierHot, hotDriver, "Hot", 0.1, 100)
	tm.AddTier(TierWarm, warmDriver, "Warm", 0.05, 500)
	tm.AddTier(TierCold, coldDriver, "Cold", 0.01, 10000)
	tm.Initialize()
	
	// Add policies
	tm.CreateDefaultAgingPolicy()
	tm.CreateCostOptimizationPolicy()
	
	// Create test volumes
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		volumeName := fmt.Sprintf("volume-%d", i)
		hotDriver.CreateVolume(ctx, volumeName, 1024*1024)
		tm.RecordVolumeAccess(volumeName)
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tm.EvaluateAndApplyTieringPolicies()
	}
}

func BenchmarkConcurrentAccess(b *testing.B) {
	tm := NewTierManager()
	driver := NewMockStorageDriver()
	tm.AddTier(TierHot, driver, "Hot", 0.1, 100)
	tm.Initialize()
	
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			volumeName := fmt.Sprintf("volume-%d", i%100)
			tm.RecordVolumeAccess(volumeName)
			i++
		}
	})
}