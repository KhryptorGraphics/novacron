package monitoring

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

func TestVMTelemetryCollector_New(t *testing.T) {
	// Create mock VM manager
	vmManager := NewMockVMManager([]string{"test-vm-1", "test-vm-2"})

	// Create a metrics storage
	metricStorage := storage.NewInMemoryStorage()

	// Create distributed collector
	collectorConfig := DefaultDistributedMetricCollectorConfig()
	collector := NewDistributedMetricCollector(collectorConfig, metricStorage)

	// Create telemetry config
	config := &VMTelemetryCollectorConfig{
		CollectionInterval: 1 * time.Second,
		VMManager:          vmManager,
		EnabledMetrics: VMMetricTypes{
			CPU:              true,
			Memory:           true,
			Disk:             true,
			Network:          true,
			IOPs:             true,
			ProcessStats:     false,
			ApplicationStats: false,
			GuestMetrics:     false,
		},
		NodeID:      "test-node",
		DetailLevel: StandardMetrics,
		Tags: map[string]string{
			"test": "true",
		},
	}

	// Create VM telemetry collector
	vmCollector := NewVMTelemetryCollector(config, collector)

	// Verify basic properties
	if vmCollector == nil {
		t.Fatal("Failed to create VM telemetry collector")
	}

	if vmCollector.ID() != "vm-telemetry-collector" {
		t.Errorf("Expected ID 'vm-telemetry-collector', got '%s'", vmCollector.ID())
	}

	if vmCollector.Enabled() {
		t.Error("Expected collector to be disabled before start")
	}

	if vmCollector.config.NodeID != "test-node" {
		t.Errorf("Expected NodeID 'test-node', got '%s'", vmCollector.config.NodeID)
	}

	if vmCollector.vmManager != vmManager {
		t.Error("VM manager not correctly assigned")
	}
}

func TestVMTelemetryCollector_StartStop(t *testing.T) {
	// Create mock VM manager
	vmManager := NewMockVMManager([]string{"test-vm-1"})

	// Create a metrics storage
	metricStorage := storage.NewInMemoryStorage()

	// Create distributed collector
	collectorConfig := DefaultDistributedMetricCollectorConfig()
	collector := NewDistributedMetricCollector(collectorConfig, metricStorage)

	// Create telemetry config
	config := &VMTelemetryCollectorConfig{
		CollectionInterval: 100 * time.Millisecond,
		VMManager:          vmManager,
		EnabledMetrics: VMMetricTypes{
			CPU:    true,
			Memory: true,
		},
		NodeID:      "test-node",
		DetailLevel: BasicMetrics,
	}

	// Create VM telemetry collector
	vmCollector := NewVMTelemetryCollector(config, collector)

	// Start the collector
	err := vmCollector.Start()
	if err != nil {
		t.Fatalf("Failed to start collector: %v", err)
	}

	// Verify that it's now enabled
	if !vmCollector.Enabled() {
		t.Error("Collector should be enabled after start")
	}

	// Let it run briefly to collect metrics
	time.Sleep(200 * time.Millisecond)

	// Stop the collector
	err = vmCollector.Stop()
	if err != nil {
		t.Fatalf("Failed to stop collector: %v", err)
	}

	// Verify that it's now disabled
	if vmCollector.Enabled() {
		t.Error("Collector should be disabled after stop")
	}
}

func TestVMTelemetryCollector_Collect(t *testing.T) {
	// Create mock VM manager with known VMs
	vmIDs := []string{"test-vm-1", "test-vm-2"}
	vmManager := NewMockVMManager(vmIDs)

	// Create a metrics storage
	metricStorage := storage.NewInMemoryStorage()

	// Create distributed collector
	collectorConfig := DefaultDistributedMetricCollectorConfig()
	collector := NewDistributedMetricCollector(collectorConfig, metricStorage)

	// Create telemetry config with all metrics enabled
	config := &VMTelemetryCollectorConfig{
		CollectionInterval: 1 * time.Second,
		VMManager:          vmManager,
		EnabledMetrics: VMMetricTypes{
			CPU:              true,
			Memory:           true,
			Disk:             true,
			Network:          true,
			IOPs:             true,
			ProcessStats:     true,
			ApplicationStats: true,
			GuestMetrics:     true,
		},
		NodeID:      "test-node",
		DetailLevel: DiagnosticMetrics,
		Tags: map[string]string{
			"test": "true",
		},
	}

	// Create VM telemetry collector
	vmCollector := NewVMTelemetryCollector(config, collector)

	// Collect metrics manually
	ctx := context.Background()
	metrics, err := vmCollector.Collect(ctx)
	if err != nil {
		t.Fatalf("Failed to collect metrics: %v", err)
	}

	// Verify that metrics were collected for each VM
	if len(metrics) == 0 {
		t.Fatal("No metrics collected")
	}

	// Count metrics by VM to verify all VMs were collected
	vmMetricCount := make(map[string]int)
	for _, metric := range metrics {
		vmID, found := metric.Tags["vm_id"]
		if found {
			vmMetricCount[vmID]++
		}
	}

	// Verify each VM has metrics
	for _, vmID := range vmIDs {
		count, found := vmMetricCount[vmID]
		if !found {
			t.Errorf("No metrics found for VM %s", vmID)
		} else if count == 0 {
			t.Errorf("Zero metrics found for VM %s", vmID)
		}
	}

	// Verify metrics contain the expected tag
	for _, metric := range metrics {
		testTag, found := metric.Tags["test"]
		if !found || testTag != "true" {
			t.Errorf("Missing or incorrect test tag on metric %s", metric.Name)
		}

		nodeTag, found := metric.Tags["node_id"]
		if !found || nodeTag != "test-node" {
			t.Errorf("Missing or incorrect node_id tag on metric %s", metric.Name)
		}
	}
}

func TestVMTelemetryCollector_NoVMManager(t *testing.T) {
	// Create a metrics storage
	metricStorage := storage.NewInMemoryStorage()

	// Create distributed collector
	collectorConfig := DefaultDistributedMetricCollectorConfig()
	collector := NewDistributedMetricCollector(collectorConfig, metricStorage)

	// Create telemetry config with no VM manager
	config := &VMTelemetryCollectorConfig{
		CollectionInterval: 1 * time.Second,
		VMManager:          nil,
		EnabledMetrics: VMMetricTypes{
			CPU:    true,
			Memory: true,
		},
		NodeID:      "test-node",
		DetailLevel: BasicMetrics,
	}

	// Create VM telemetry collector
	vmCollector := NewVMTelemetryCollector(config, collector)

	// Try to collect metrics
	ctx := context.Background()
	_, err := vmCollector.Collect(ctx)

	// Verify that it fails with appropriate error
	if err == nil {
		t.Error("Expected error when collecting with nil VM manager")
	}
}

func TestVMTelemetryCollector_DefaultConfig(t *testing.T) {
	// Create default config and check values
	config := DefaultVMTelemetryCollectorConfig()

	// Verify default values
	if config.CollectionInterval != 30*time.Second {
		t.Errorf("Expected default collection interval of 30s, got %v", config.CollectionInterval)
	}

	if !config.EnabledMetrics.CPU || !config.EnabledMetrics.Memory || !config.EnabledMetrics.Disk || !config.EnabledMetrics.Network {
		t.Error("Expected basic metrics to be enabled by default")
	}

	if config.EnabledMetrics.ProcessStats {
		t.Error("Process stats should be disabled by default")
	}

	if config.DetailLevel != StandardMetrics {
		t.Errorf("Expected default detail level to be StandardMetrics, got %v", config.DetailLevel)
	}
}

// TestMetricsStorageHook allows tests to verify metrics are stored properly
type TestMetricsStorageHook struct {
	stored []*Metric
}

// NewTestMetricsStorageHook creates a new storage hook for testing
func NewTestMetricsStorageHook() *TestMetricsStorageHook {
	return &TestMetricsStorageHook{
		stored: make([]*Metric, 0),
	}
}

// GetStoredMetrics returns all metrics that have been stored
func (h *TestMetricsStorageHook) GetStoredMetrics() []*Metric {
	return h.stored
}

// GetStoredMetricsByName returns metrics that have been stored with the given name
func (h *TestMetricsStorageHook) GetStoredMetricsByName(name string) []*Metric {
	result := make([]*Metric, 0)
	for _, metric := range h.stored {
		if metric.Name == name {
			result = append(result, metric)
		}
	}
	return result
}

// GetStoredMetricsByVMID returns metrics that have been stored with the given VM ID
func (h *TestMetricsStorageHook) GetStoredMetricsByVMID(vmID string) []*Metric {
	result := make([]*Metric, 0)
	for _, metric := range h.stored {
		if id, exists := metric.Tags["vm_id"]; exists && id == vmID {
			result = append(result, metric)
		}
	}
	return result
}

// Hook handles a metric being stored
func (h *TestMetricsStorageHook) Hook(metric *Metric) {
	h.stored = append(h.stored, metric)
}
