package storage

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// MockStorageDriver implements StorageDriver for testing
type MockStorageDriver struct {
	shouldFail    bool
	latency       time.Duration
	volumes       map[string]bool
	failureCount  int
	maxFailures   int
}

func NewMockStorageDriver() *MockStorageDriver {
	return &MockStorageDriver{
		volumes: make(map[string]bool),
	}
}

func (m *MockStorageDriver) Initialize() error {
	if m.shouldFail && m.failureCount < m.maxFailures {
		m.failureCount++
		return fmt.Errorf("mock initialization failure %d", m.failureCount)
	}
	return nil
}

func (m *MockStorageDriver) Shutdown() error {
	return nil
}

func (m *MockStorageDriver) CreateVolume(ctx context.Context, volumeID string, sizeBytes int64) error {
	if m.shouldFail && m.failureCount < m.maxFailures {
		m.failureCount++
		return fmt.Errorf("mock create volume failure %d", m.failureCount)
	}
	m.volumes[volumeID] = true
	return nil
}

func (m *MockStorageDriver) DeleteVolume(ctx context.Context, volumeID string) error {
	if m.shouldFail && m.failureCount < m.maxFailures {
		m.failureCount++
		return fmt.Errorf("mock delete volume failure %d", m.failureCount)
	}
	delete(m.volumes, volumeID)
	return nil
}

func (m *MockStorageDriver) AttachVolume(ctx context.Context, volumeID, nodeID string) error {
	return nil
}

func (m *MockStorageDriver) DetachVolume(ctx context.Context, volumeID, nodeID string) error {
	return nil
}

func (m *MockStorageDriver) ReadVolume(ctx context.Context, volumeID string, offset int64, size int) ([]byte, error) {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}
	if m.shouldFail && m.failureCount < m.maxFailures {
		m.failureCount++
		return nil, fmt.Errorf("mock read failure %d", m.failureCount)
	}
	return make([]byte, size), nil
}

func (m *MockStorageDriver) WriteVolume(ctx context.Context, volumeID string, offset int64, data []byte) error {
	if m.latency > 0 {
		time.Sleep(m.latency)
	}
	if m.shouldFail && m.failureCount < m.maxFailures {
		m.failureCount++
		return fmt.Errorf("mock write failure %d", m.failureCount)
	}
	return nil
}

func (m *MockStorageDriver) GetVolumeInfo(ctx context.Context, volumeID string) (*VolumeInfo, error) {
	if _, exists := m.volumes[volumeID]; !exists {
		return nil, ErrVolumeNotFound
	}
	return &VolumeInfo{
		ID:   volumeID,
		Name: volumeID,
		Type: VolumeTypeLocal,
		Size: 1024 * 1024 * 1024, // 1GB
	}, nil
}

func (m *MockStorageDriver) ListVolumes(ctx context.Context) ([]string, error) {
	if m.shouldFail && m.failureCount < m.maxFailures {
		m.failureCount++
		return nil, fmt.Errorf("mock list volumes failure %d", m.failureCount)
	}
	
	volumes := make([]string, 0, len(m.volumes))
	for volumeID := range m.volumes {
		volumes = append(volumes, volumeID)
	}
	return volumes, nil
}

func (m *MockStorageDriver) GetCapabilities() DriverCapabilities {
	return DriverCapabilities{
		SupportsSnapshots:     true,
		SupportsReplication:   false,
		SupportsEncryption:    true,
		SupportsCompression:   false,
		SupportsDeduplication: false,
		MaxVolumeSize:         1024 * 1024 * 1024 * 1024, // 1TB
		MinVolumeSize:         1,
	}
}

func (m *MockStorageDriver) CreateSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return nil
}

func (m *MockStorageDriver) DeleteSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return nil
}

func (m *MockStorageDriver) RestoreSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	return nil
}

func (m *MockStorageDriver) SetShouldFail(shouldFail bool, maxFailures int) {
	m.shouldFail = shouldFail
	m.maxFailures = maxFailures
	m.failureCount = 0
}

func (m *MockStorageDriver) SetLatency(latency time.Duration) {
	m.latency = latency
}

func TestHealthMonitor_CreateAndStart(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	config.CheckInterval = 100 * time.Millisecond // Fast checks for testing
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	err := healthMonitor.Start()
	if err != nil {
		t.Fatalf("Failed to start health monitor: %v", err)
	}
	defer healthMonitor.Stop()
	
	// Verify monitor is running
	if healthMonitor.ctx.Err() != nil {
		t.Error("Expected health monitor context to be active")
	}
}

func TestHealthMonitor_RegisterDriver(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	mockDriver := NewMockStorageDriver()
	driverName := "test-driver"
	
	healthMonitor.RegisterDriver(driverName, mockDriver)
	
	// Verify driver is registered
	if _, exists := healthMonitor.drivers[driverName]; !exists {
		t.Error("Expected driver to be registered")
	}
	
	// Verify health status is initialized
	if _, exists := healthMonitor.componentHealth[driverName]; !exists {
		t.Error("Expected health status to be initialized")
	}
	
	health, err := healthMonitor.GetHealth(driverName)
	if err != nil {
		t.Fatalf("Failed to get health: %v", err)
	}
	
	if health.ComponentID != driverName {
		t.Errorf("Expected component ID %s, got %s", driverName, health.ComponentID)
	}
	
	if health.ComponentType != "driver" {
		t.Errorf("Expected component type 'driver', got %s", health.ComponentType)
	}
	
	if health.Status != HealthStatusUnknown {
		t.Errorf("Expected initial status %s, got %s", HealthStatusUnknown, health.Status)
	}
}

func TestHealthMonitor_UnregisterDriver(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	mockDriver := NewMockStorageDriver()
	driverName := "test-driver"
	
	healthMonitor.RegisterDriver(driverName, mockDriver)
	healthMonitor.UnregisterDriver(driverName)
	
	// Verify driver is unregistered
	if _, exists := healthMonitor.drivers[driverName]; exists {
		t.Error("Expected driver to be unregistered")
	}
	
	// Verify health status is removed
	if _, exists := healthMonitor.componentHealth[driverName]; exists {
		t.Error("Expected health status to be removed")
	}
	
	_, err := healthMonitor.GetHealth(driverName)
	if err == nil {
		t.Error("Expected error when getting health of unregistered driver")
	}
}

func TestHealthMonitor_CheckConnectivity(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	config.CheckTimeout = 1 * time.Second
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	// Test successful connectivity check
	mockDriver := NewMockStorageDriver()
	driverName := "test-driver"
	
	result := healthMonitor.checkConnectivity(driverName, mockDriver)
	
	if result.Type != HealthCheckConnectivity {
		t.Errorf("Expected check type %s, got %s", HealthCheckConnectivity, result.Type)
	}
	
	if result.Status != HealthStatusHealthy {
		t.Errorf("Expected status %s, got %s", HealthStatusHealthy, result.Status)
	}
	
	if result.Error != nil {
		t.Errorf("Expected no error, got %v", result.Error)
	}
	
	// Test failed connectivity check
	mockDriver.SetShouldFail(true, 1)
	result = healthMonitor.checkConnectivity(driverName, mockDriver)
	
	if result.Status != HealthStatusCritical {
		t.Errorf("Expected status %s, got %s", HealthStatusCritical, result.Status)
	}
	
	if result.Error == nil {
		t.Error("Expected error for failed connectivity check")
	}
}

func TestHealthMonitor_CheckPerformance(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	config.CheckTimeout = 5 * time.Second
	config.PerformanceThresholds.MaxLatency = 100 * time.Millisecond
	config.PerformanceThresholds.MinThroughput = 1.0 // 1 MB/s
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	// Test normal performance
	mockDriver := NewMockStorageDriver()
	driverName := "test-driver"
	
	result := healthMonitor.checkPerformance(driverName, mockDriver)
	
	if result.Type != HealthCheckPerformance {
		t.Errorf("Expected check type %s, got %s", HealthCheckPerformance, result.Type)
	}
	
	if result.Status != HealthStatusHealthy {
		t.Errorf("Expected status %s, got %s", HealthStatusHealthy, result.Status)
	}
	
	// Verify metrics are present
	if result.Metrics == nil {
		t.Error("Expected performance metrics")
	}
	
	expectedMetrics := []string{"write_latency_ms", "read_latency_ms", "write_throughput_mb", "read_throughput_mb"}
	for _, metric := range expectedMetrics {
		if _, exists := result.Metrics[metric]; !exists {
			t.Errorf("Expected metric %s in results", metric)
		}
	}
	
	// Test high latency warning
	mockDriver.SetLatency(200 * time.Millisecond) // Above threshold
	result = healthMonitor.checkPerformance(driverName, mockDriver)
	
	if result.Status != HealthStatusWarning {
		t.Errorf("Expected status %s for high latency, got %s", HealthStatusWarning, result.Status)
	}
}

func TestHealthMonitor_UpdateComponentHealth(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	driverName := "test-driver"
	healthMonitor.componentHealth[driverName] = &StorageHealth{
		ComponentID:   driverName,
		ComponentType: "driver",
		Status:        HealthStatusUnknown,
		HealthScore:   100,
	}
	
	// Test update with healthy checks
	checkResults := []HealthCheckResult{
		{
			Type:      HealthCheckConnectivity,
			Status:    HealthStatusHealthy,
			Message:   "Connection OK",
			Duration:  10 * time.Millisecond,
			Timestamp: time.Now(),
		},
		{
			Type:      HealthCheckPerformance,
			Status:    HealthStatusHealthy,
			Message:   "Performance OK",
			Duration:  50 * time.Millisecond,
			Timestamp: time.Now(),
		},
	}
	
	healthMonitor.updateComponentHealth(driverName, checkResults)
	
	health, _ := healthMonitor.GetHealth(driverName)
	
	if health.Status != HealthStatusHealthy {
		t.Errorf("Expected status %s, got %s", HealthStatusHealthy, health.Status)
	}
	
	if health.HealthScore != 100 {
		t.Errorf("Expected health score 100, got %d", health.HealthScore)
	}
	
	if health.ConsecutiveFailures != 0 {
		t.Errorf("Expected 0 consecutive failures, got %d", health.ConsecutiveFailures)
	}
	
	// Test update with critical failure
	checkResults[0].Status = HealthStatusCritical
	checkResults[0].Error = fmt.Errorf("connectivity failed")
	
	healthMonitor.updateComponentHealth(driverName, checkResults)
	
	health, _ = healthMonitor.GetHealth(driverName)
	
	if health.Status != HealthStatusCritical {
		t.Errorf("Expected status %s, got %s", HealthStatusCritical, health.Status)
	}
	
	if health.HealthScore >= 100 {
		t.Errorf("Expected health score < 100, got %d", health.HealthScore)
	}
	
	if health.ConsecutiveFailures != 1 {
		t.Errorf("Expected 1 consecutive failure, got %d", health.ConsecutiveFailures)
	}
	
	if health.UnhealthySince == nil {
		t.Error("Expected UnhealthySince to be set")
	}
}

func TestHealthMonitor_ShouldTriggerHealing(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	// Create health status with critical status
	unhealthyTime := time.Now().Add(-10 * time.Minute)
	health := &StorageHealth{
		Status:              HealthStatusCritical,
		ConsecutiveFailures: 5,
		UnhealthySince:      &unhealthyTime,
		CheckResults: []HealthCheckResult{
			{
				Type:   HealthCheckConnectivity,
				Status: HealthStatusCritical,
			},
		},
	}
	
	// Test trigger with matching conditions
	triggers := []HealingTrigger{
		{
			HealthStatus:        HealthStatusCritical,
			ConsecutiveFailures: 3,
			UnhealthyDuration:   5 * time.Minute,
			FailingChecks:       []HealthCheckType{HealthCheckConnectivity},
		},
	}
	
	shouldTrigger := healthMonitor.shouldTriggerHealing(health, triggers)
	if !shouldTrigger {
		t.Error("Expected healing to be triggered")
	}
	
	// Test no trigger due to insufficient failures
	triggers[0].ConsecutiveFailures = 10
	shouldTrigger = healthMonitor.shouldTriggerHealing(health, triggers)
	if shouldTrigger {
		t.Error("Expected healing not to be triggered due to insufficient failures")
	}
	
	// Test no trigger due to insufficient unhealthy duration
	triggers[0].ConsecutiveFailures = 3
	triggers[0].UnhealthyDuration = 20 * time.Minute
	shouldTrigger = healthMonitor.shouldTriggerHealing(health, triggers)
	if shouldTrigger {
		t.Error("Expected healing not to be triggered due to insufficient unhealthy duration")
	}
	
	// Test no trigger due to wrong failing check type
	triggers[0].UnhealthyDuration = 5 * time.Minute
	triggers[0].FailingChecks = []HealthCheckType{HealthCheckPerformance}
	shouldTrigger = healthMonitor.shouldTriggerHealing(health, triggers)
	if shouldTrigger {
		t.Error("Expected healing not to be triggered due to wrong failing check type")
	}
}

func TestHealthMonitor_GetAllHealth(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	// Register multiple drivers
	for i := 0; i < 3; i++ {
		driverName := fmt.Sprintf("driver-%d", i)
		mockDriver := NewMockStorageDriver()
		healthMonitor.RegisterDriver(driverName, mockDriver)
	}
	
	allHealth := healthMonitor.GetAllHealth()
	
	if len(allHealth) != 3 {
		t.Errorf("Expected 3 health entries, got %d", len(allHealth))
	}
	
	for i := 0; i < 3; i++ {
		driverName := fmt.Sprintf("driver-%d", i)
		if _, exists := allHealth[driverName]; !exists {
			t.Errorf("Expected health entry for %s", driverName)
		}
	}
}

func TestDefaultHealthMonitorConfig(t *testing.T) {
	config := DefaultHealthMonitorConfig()
	
	// Verify default values
	if config.CheckInterval != 30*time.Second {
		t.Errorf("Expected check interval 30s, got %v", config.CheckInterval)
	}
	
	if config.CheckTimeout != 10*time.Second {
		t.Errorf("Expected check timeout 10s, got %v", config.CheckTimeout)
	}
	
	if config.FailureThreshold != 3 {
		t.Errorf("Expected failure threshold 3, got %d", config.FailureThreshold)
	}
	
	if !config.AutoHealingEnabled {
		t.Error("Expected auto healing to be enabled by default")
	}
	
	// Verify performance thresholds
	if config.PerformanceThresholds.MaxLatency != 1*time.Second {
		t.Errorf("Expected max latency 1s, got %v", config.PerformanceThresholds.MaxLatency)
	}
	
	if config.PerformanceThresholds.MinIOPS != 100 {
		t.Errorf("Expected min IOPS 100, got %d", config.PerformanceThresholds.MinIOPS)
	}
	
	// Verify capacity thresholds
	if config.CapacityThresholds.WarningThreshold != 80.0 {
		t.Errorf("Expected warning threshold 80%%, got %.1f", config.CapacityThresholds.WarningThreshold)
	}
	
	if config.CapacityThresholds.CriticalThreshold != 95.0 {
		t.Errorf("Expected critical threshold 95%%, got %.1f", config.CapacityThresholds.CriticalThreshold)
	}
	
	// Verify healing strategies exist
	if len(config.HealingStrategies) == 0 {
		t.Error("Expected healing strategies to be configured")
	}
}

// Benchmark tests for health monitoring
func BenchmarkHealthMonitor_CheckConnectivity(b *testing.B) {
	config := DefaultHealthMonitorConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	mockDriver := NewMockStorageDriver()
	driverName := "bench-driver"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		healthMonitor.checkConnectivity(driverName, mockDriver)
	}
}

func BenchmarkHealthMonitor_CheckPerformance(b *testing.B) {
	config := DefaultHealthMonitorConfig()
	config.CheckTimeout = 30 * time.Second // Longer timeout for benchmark
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	mockDriver := NewMockStorageDriver()
	driverName := "bench-driver"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		healthMonitor.checkPerformance(driverName, mockDriver)
	}
}

func BenchmarkHealthMonitor_UpdateComponentHealth(b *testing.B) {
	config := DefaultHealthMonitorConfig()
	
	storageConfig := DefaultStorageConfig()
	storageService := NewBaseStorageService(storageConfig)
	
	healthMonitor := NewHealthMonitor(config, storageService)
	
	driverName := "bench-driver"
	healthMonitor.componentHealth[driverName] = &StorageHealth{
		ComponentID:   driverName,
		ComponentType: "driver",
		Status:        HealthStatusUnknown,
		HealthScore:   100,
	}
	
	checkResults := []HealthCheckResult{
		{
			Type:      HealthCheckConnectivity,
			Status:    HealthStatusHealthy,
			Duration:  10 * time.Millisecond,
			Timestamp: time.Now(),
		},
		{
			Type:      HealthCheckPerformance,
			Status:    HealthStatusHealthy,
			Duration:  50 * time.Millisecond,
			Timestamp: time.Now(),
		},
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		healthMonitor.updateComponentHealth(driverName, checkResults)
	}
}

