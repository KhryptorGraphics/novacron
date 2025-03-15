package cloud

import (
	"context"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// CloudVMTelemetryIntegration provides integration between cloud providers
// and the VM telemetry system
type CloudVMTelemetryIntegration struct {
	// The provider manager to get cloud provider implementations
	providerManager *ProviderManager

	// Map of provider implementations to VM managers
	vmManagers map[string]monitoring.VMManagerInterface

	// Configuration for the integration
	config *CloudVMTelemetryConfig
}

// CloudVMTelemetryConfig contains configuration for the cloud VM telemetry integration
type CloudVMTelemetryConfig struct {
	// RefreshInterval defines how often cloud provider VM lists are refreshed
	RefreshInterval time.Duration

	// EnabledProviders specifies which cloud providers to integrate with
	EnabledProviders []string

	// DefaultMetricConfig specifies default configuration for each provider
	DefaultMetricConfig map[string]*ProviderMetricConfig
}

// ProviderMetricConfig contains metric collection configuration for a specific provider
type ProviderMetricConfig struct {
	// CollectionInterval specifies how often metrics should be collected from this provider
	CollectionInterval time.Duration

	// DetailLevel specifies the level of detail for metrics
	DetailLevel monitoring.VMMetricDetailLevel

	// EnabledMetrics specifies which metric types to collect
	EnabledMetrics monitoring.VMMetricTypes

	// MetricTags are additional tags to apply to metrics from this provider
	MetricTags map[string]string
}

// NewCloudVMTelemetryIntegration creates a new integration between cloud providers and VM telemetry
func NewCloudVMTelemetryIntegration(providerManager *ProviderManager, config *CloudVMTelemetryConfig) *CloudVMTelemetryIntegration {
	if config == nil {
		config = &CloudVMTelemetryConfig{
			RefreshInterval:  5 * time.Minute,
			EnabledProviders: []string{"aws", "azure", "gcp"},
			DefaultMetricConfig: map[string]*ProviderMetricConfig{
				"aws": {
					CollectionInterval: 60 * time.Second,
					DetailLevel:        monitoring.StandardMetrics,
					EnabledMetrics: monitoring.VMMetricTypes{
						CPU:          true,
						Memory:       true,
						Disk:         true,
						Network:      true,
						IOPs:         true,
						ProcessStats: false,
					},
					MetricTags: map[string]string{
						"provider": "aws",
					},
				},
				"azure": {
					CollectionInterval: 60 * time.Second,
					DetailLevel:        monitoring.StandardMetrics,
					EnabledMetrics: monitoring.VMMetricTypes{
						CPU:          true,
						Memory:       true,
						Disk:         true,
						Network:      true,
						IOPs:         true,
						ProcessStats: false,
					},
					MetricTags: map[string]string{
						"provider": "azure",
					},
				},
				"gcp": {
					CollectionInterval: 60 * time.Second,
					DetailLevel:        monitoring.StandardMetrics,
					EnabledMetrics: monitoring.VMMetricTypes{
						CPU:          true,
						Memory:       true,
						Disk:         true,
						Network:      true,
						IOPs:         true,
						ProcessStats: false,
					},
					MetricTags: map[string]string{
						"provider": "gcp",
					},
				},
			},
		}
	}

	return &CloudVMTelemetryIntegration{
		providerManager: providerManager,
		vmManagers:      make(map[string]monitoring.VMManagerInterface),
		config:          config,
	}
}

// Initialize sets up the cloud VM telemetry integration
func (ci *CloudVMTelemetryIntegration) Initialize(ctx context.Context) error {
	for _, providerName := range ci.config.EnabledProviders {
		provider, err := ci.providerManager.GetProvider(providerName)
		if err != nil {
			return fmt.Errorf("error getting provider %s: %w", providerName, err)
		}

		// Create VM manager for this provider
		vmManager, err := ci.createVMManager(provider)
		if err != nil {
			return fmt.Errorf("error creating VM manager for provider %s: %w", providerName, err)
		}

		ci.vmManagers[providerName] = vmManager
	}

	return nil
}

// CreateVMTelemetryCollector creates a VM telemetry collector configured for cloud VMs
func (ci *CloudVMTelemetryIntegration) CreateVMTelemetryCollector(
	distributedCollector *monitoring.DistributedMetricCollector,
	providerName string,
) (*monitoring.VMTelemetryCollector, error) {
	vmManager, exists := ci.vmManagers[providerName]
	if !exists {
		return nil, fmt.Errorf("VM manager for provider %s not found", providerName)
	}

	providerConfig := ci.config.DefaultMetricConfig[providerName]
	if providerConfig == nil {
		return nil, fmt.Errorf("metric configuration for provider %s not found", providerName)
	}

	// Create VM telemetry collector configuration
	telemetryConfig := &monitoring.VMTelemetryCollectorConfig{
		CollectionInterval: providerConfig.CollectionInterval,
		VMManager:          vmManager,
		EnabledMetrics:     providerConfig.EnabledMetrics,
		Tags:               providerConfig.MetricTags,
		NodeID:             fmt.Sprintf("cloud-%s", providerName),
		DetailLevel:        providerConfig.DetailLevel,
	}

	// Create VM telemetry collector
	collector := monitoring.NewVMTelemetryCollector(telemetryConfig, distributedCollector)

	return collector, nil
}

// CreateMultiProviderCollector creates a collector that aggregates VMs from multiple providers
func (ci *CloudVMTelemetryIntegration) CreateMultiProviderCollector(
	distributedCollector *monitoring.DistributedMetricCollector,
) (*monitoring.VMTelemetryCollector, error) {
	// Create a multi-provider VM manager
	multiManager := NewMultiProviderVMManager()

	// Add all enabled providers
	for _, providerName := range ci.config.EnabledProviders {
		vmManager, exists := ci.vmManagers[providerName]
		if !exists {
			continue
		}

		multiManager.AddVMManager(providerName, vmManager)
	}

	// Create VM telemetry collector configuration
	telemetryConfig := &monitoring.VMTelemetryCollectorConfig{
		CollectionInterval: 60 * time.Second, // Default for multi-provider
		VMManager:          multiManager,
		EnabledMetrics: monitoring.VMMetricTypes{
			CPU:          true,
			Memory:       true,
			Disk:         true,
			Network:      true,
			IOPs:         true,
			ProcessStats: false,
		},
		Tags: map[string]string{
			"collector": "multi-provider",
		},
		NodeID:      "cloud-multi-provider",
		DetailLevel: monitoring.StandardMetrics,
	}

	// Create VM telemetry collector
	collector := monitoring.NewVMTelemetryCollector(telemetryConfig, distributedCollector)

	return collector, nil
}

// createVMManager creates a VM manager implementation for the given cloud provider
func (ci *CloudVMTelemetryIntegration) createVMManager(provider Provider) (monitoring.VMManagerInterface, error) {
	switch p := provider.(type) {
	case *AWSProvider:
		return NewAWSVMManager(p), nil
	case *AzureProvider:
		return NewAzureVMManager(p), nil
	case *GCPProvider:
		return NewGCPVMManager(p), nil
	default:
		return nil, fmt.Errorf("unsupported provider type: %T", provider)
	}
}

// MultiProviderVMManager is a VM manager that combines VMs from multiple providers
type MultiProviderVMManager struct {
	// Map of provider names to VM managers
	managers map[string]monitoring.VMManagerInterface
}

// NewMultiProviderVMManager creates a new multi-provider VM manager
func NewMultiProviderVMManager() *MultiProviderVMManager {
	return &MultiProviderVMManager{
		managers: make(map[string]monitoring.VMManagerInterface),
	}
}

// AddVMManager adds a VM manager for a specific provider
func (m *MultiProviderVMManager) AddVMManager(providerName string, manager monitoring.VMManagerInterface) {
	m.managers[providerName] = manager
}

// GetVMs returns a list of all VM IDs from all providers
func (m *MultiProviderVMManager) GetVMs(ctx context.Context) ([]string, error) {
	var allVMs []string

	for providerName, manager := range m.managers {
		vms, err := manager.GetVMs(ctx)
		if err != nil {
			return nil, fmt.Errorf("error getting VMs from provider %s: %w", providerName, err)
		}

		// Add provider prefix to VM IDs to avoid conflicts
		for _, vmID := range vms {
			allVMs = append(allVMs, fmt.Sprintf("%s-%s", providerName, vmID))
		}
	}

	return allVMs, nil
}

// GetVMStats retrieves stats for a specific VM
func (m *MultiProviderVMManager) GetVMStats(
	ctx context.Context,
	vmID string,
	detailLevel monitoring.VMMetricDetailLevel,
) (*monitoring.VMStats, error) {
	// Parse the VM ID to extract provider and original ID
	var providerName, originalID string

	for provider := range m.managers {
		prefix := fmt.Sprintf("%s-", provider)
		if len(vmID) > len(prefix) && vmID[:len(prefix)] == prefix {
			providerName = provider
			originalID = vmID[len(prefix):]
			break
		}
	}

	if providerName == "" || originalID == "" {
		return nil, fmt.Errorf("invalid VM ID format: %s", vmID)
	}

	// Get the VM manager for this provider
	manager, exists := m.managers[providerName]
	if !exists {
		return nil, fmt.Errorf("VM manager for provider %s not found", providerName)
	}

	// Get stats for the VM
	vmStats, err := manager.GetVMStats(ctx, originalID, detailLevel)
	if err != nil {
		return nil, fmt.Errorf("error getting VM stats from provider %s: %w", providerName, err)
	}

	return vmStats, nil
}

// GetCloudVMStats retrieves cloud-formatted stats for a specific VM
func (m *MultiProviderVMManager) GetCloudVMStats(
	ctx context.Context,
	vmID string,
	detailLevel monitoring.VMMetricDetailLevel,
) (*monitoring.CloudVMStats, error) {
	// Parse the VM ID to extract provider and original ID
	var providerName, originalID string

	for provider := range m.managers {
		prefix := fmt.Sprintf("%s-", provider)
		if len(vmID) > len(prefix) && vmID[:len(prefix)] == prefix {
			providerName = provider
			originalID = vmID[len(prefix):]
			break
		}
	}

	if providerName == "" || originalID == "" {
		return nil, fmt.Errorf("invalid VM ID format: %s", vmID)
	}

	// Get the VM manager for this provider
	manager, exists := m.managers[providerName]
	if !exists {
		return nil, fmt.Errorf("VM manager for provider %s not found", providerName)
	}

	// Get stats for the VM
	vmStats, err := manager.GetVMStats(ctx, originalID, detailLevel)
	if err != nil {
		return nil, fmt.Errorf("error getting VM stats from provider %s: %w", providerName, err)
	}

	// Convert internal VMStats to CloudVMStats using our adapter
	cloudStats := monitoring.ConvertInternalToCloudStats(vmStats)

	// Add provider-specific information
	if cloudStats.Metadata == nil {
		cloudStats.Metadata = make(map[string]string)
	}
	cloudStats.Metadata["provider"] = providerName

	if cloudStats.Tags == nil {
		cloudStats.Tags = make(map[string]string)
	}
	cloudStats.Tags["provider"] = providerName

	return cloudStats, nil
}
