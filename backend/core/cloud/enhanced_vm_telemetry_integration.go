package cloud

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// EnhancedCloudVMTelemetryConfig extends the basic configuration with advanced features
type EnhancedCloudVMTelemetryConfig struct {
	// Base configuration
	CloudVMTelemetryConfig

	// Advanced features
	// MetricCacheTTL configures how long metrics are cached
	MetricCacheTTL time.Duration

	// InventoryCacheTTL configures how long VM inventories are cached
	InventoryCacheTTL time.Duration

	// ParallelCollectionLimit limits concurrent API calls to prevent throttling
	ParallelCollectionLimit int

	// EnableNativeMetrics uses provider-specific metrics when available
	EnableNativeMetrics bool

	// CollectResourceMetadata collects additional resource metadata
	CollectResourceMetadata bool

	// EnableAdaptiveCollection adjusts collection frequency based on VM activity
	EnableAdaptiveCollection bool

	// EnableEventSubscriptions uses provider event subscriptions if available
	EnableEventSubscriptions bool

	// FailoverTimeout specifies how long to wait before failing over to another provider
	FailoverTimeout time.Duration

	// RetryConfig for API request retries
	RetryConfig *RetryConfig

	// BatchSize for API requests that support batching
	BatchSize int

	// EnableCostMetrics includes cost data in metrics
	EnableCostMetrics bool
}

// RetryConfig configures retry behavior for cloud API calls
type RetryConfig struct {
	// MaxRetries is the maximum number of retries
	MaxRetries int

	// InitialBackoff is the initial backoff duration
	InitialBackoff time.Duration

	// MaxBackoff is the maximum backoff duration
	MaxBackoff time.Duration

	// BackoffMultiplier is the multiplier for backoff between retries
	BackoffMultiplier float64

	// RetryableErrors specifies which errors should be retried
	RetryableErrors []string
}

// DefaultEnhancedCloudVMTelemetryConfig returns default enhanced configuration
func DefaultEnhancedCloudVMTelemetryConfig() *EnhancedCloudVMTelemetryConfig {
	baseConfig := &CloudVMTelemetryConfig{
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

	return &EnhancedCloudVMTelemetryConfig{
		CloudVMTelemetryConfig:   *baseConfig,
		MetricCacheTTL:           5 * time.Minute,
		InventoryCacheTTL:        10 * time.Minute,
		ParallelCollectionLimit:  10,
		EnableNativeMetrics:      true,
		CollectResourceMetadata:  true,
		EnableAdaptiveCollection: false,
		EnableEventSubscriptions: false,
		FailoverTimeout:          30 * time.Second,
		RetryConfig: &RetryConfig{
			MaxRetries:        3,
			InitialBackoff:    1 * time.Second,
			MaxBackoff:        30 * time.Second,
			BackoffMultiplier: 2.0,
			RetryableErrors:   []string{"RequestLimitExceeded", "Throttling", "TooManyRequests"},
		},
		BatchSize:         20,
		EnableCostMetrics: false,
	}
}

// EnhancedCloudVMTelemetryIntegration extends the base integration with advanced features
type EnhancedCloudVMTelemetryIntegration struct {
	// Embed the base integration
	*CloudVMTelemetryIntegration

	// Enhanced configuration
	enhancedConfig *EnhancedCloudVMTelemetryConfig

	// VM inventory cache
	vmInventoryCache      map[string]map[string]*VMInventory
	vmInventoryCacheMutex sync.RWMutex
	vmInventoryExpiry     map[string]map[string]time.Time

	// VM metrics cache
	vmMetricsCache      map[string]map[string]*monitoring.VMStats
	vmMetricsCacheMutex sync.RWMutex
	vmMetricsExpiry     map[string]map[string]time.Time

	// Provider status tracking
	providerStatus      map[string]ProviderStatus
	providerStatusMutex sync.RWMutex

	// Rate limiters for each provider
	rateLimiters      map[string]*RateLimiter
	rateLimitersMutex sync.RWMutex

	// Event subscriptions
	eventSubscriptions      map[string]EventSubscription
	eventSubscriptionsMutex sync.RWMutex

	// Cost data cache
	costDataCache      map[string]map[string]*CostData
	costDataCacheMutex sync.RWMutex

	// Metric transformation pipeline
	metricPipeline *MetricTransformationPipeline
}

// VMInventory represents a cached VM inventory item
type VMInventory struct {
	VMID           string
	Name           string
	Type           string
	Region         string
	Zone           string
	State          string
	ProviderID     string
	LaunchTime     time.Time
	Tags           map[string]string
	LastRefreshed  time.Time
	InstanceType   string
	CPUCores       int
	MemoryMB       int
	RootDiskGB     int
	NetworkType    string
	PublicIP       string
	PrivateIP      string
	SecurityGroups []string
}

// ProviderStatus tracks the health of a cloud provider
type ProviderStatus struct {
	Provider          string
	Healthy           bool
	LastChecked       time.Time
	FailureCount      int
	LastError         string
	AvgResponseTimeMS int64
	ThrottleCount     int
	APIQuotaRemaining int
	RegionStatus      map[string]bool
}

// RateLimiter manages API request rate limiting
type RateLimiter struct {
	Provider        string
	RequestsPerSec  int
	BurstSize       int
	Tokens          int
	LastRefill      time.Time
	RefillRate      float64
	RegionLimiters  map[string]*RateLimiter
	LimiterMutex    sync.Mutex
	SharedQuota     bool
	QuotaReset      time.Time
	MaxRequestQueue int
	PendingRequests int
}

// EventSubscription manages provider-specific event streams
type EventSubscription struct {
	Provider      string
	ResourceType  string
	EventTypes    []string
	LastEventTime time.Time
	Active        bool
	EventChannel  chan CloudEvent
	Errors        chan error
	CancelFunc    context.CancelFunc
}

// CloudEvent represents an event from a cloud provider
type CloudEvent struct {
	Provider     string
	ResourceID   string
	ResourceType string
	EventType    string
	EventTime    time.Time
	EventData    map[string]interface{}
	Region       string
	AccountID    string
}

// CostData contains cost information for a cloud resource
type CostData struct {
	ResourceID       string
	Provider         string
	HourlyCost       float64
	DailyCost        float64
	MonthToDateCost  float64
	ForecastedCost   float64
	BillingCurrency  string
	LastUpdated      time.Time
	CostComponents   map[string]float64
	ResourceTags     map[string]string
	BillingTags      map[string]string
	BillingAccountID string
}

// MetricTransformationPipeline handles metric processing
type MetricTransformationPipeline struct {
	Transformers []MetricTransformer
	Pipeline     chan *monitoring.VMStats
	Workers      int
	Active       bool
	WaitGroup    sync.WaitGroup
	Errors       chan error
}

// MetricTransformer defines the interface for metric transformations
type MetricTransformer interface {
	Transform(stats *monitoring.VMStats) (*monitoring.VMStats, error)
	Name() string
}

// NewEnhancedCloudVMTelemetryIntegration creates an enhanced integration
func NewEnhancedCloudVMTelemetryIntegration(
	providerManager *ProviderManager,
	config *EnhancedCloudVMTelemetryConfig,
) *EnhancedCloudVMTelemetryIntegration {
	if config == nil {
		config = DefaultEnhancedCloudVMTelemetryConfig()
	}

	baseIntegration := NewCloudVMTelemetryIntegration(providerManager, &config.CloudVMTelemetryConfig)

	enhanced := &EnhancedCloudVMTelemetryIntegration{
		CloudVMTelemetryIntegration: baseIntegration,
		enhancedConfig:              config,
		vmInventoryCache:            make(map[string]map[string]*VMInventory),
		vmInventoryExpiry:           make(map[string]map[string]time.Time),
		vmMetricsCache:              make(map[string]map[string]*monitoring.VMStats),
		vmMetricsExpiry:             make(map[string]map[string]time.Time),
		providerStatus:              make(map[string]ProviderStatus),
		rateLimiters:                make(map[string]*RateLimiter),
		eventSubscriptions:          make(map[string]EventSubscription),
		costDataCache:               make(map[string]map[string]*CostData),
		metricPipeline:              createDefaultMetricPipeline(),
	}

	// Initialize provider status
	for _, providerName := range config.EnabledProviders {
		enhanced.providerStatus[providerName] = ProviderStatus{
			Provider:     providerName,
			Healthy:      true,
			LastChecked:  time.Now(),
			RegionStatus: make(map[string]bool),
		}

		// Initialize caches for this provider
		enhanced.vmInventoryCache[providerName] = make(map[string]*VMInventory)
		enhanced.vmInventoryExpiry[providerName] = make(map[string]time.Time)
		enhanced.vmMetricsCache[providerName] = make(map[string]*monitoring.VMStats)
		enhanced.vmMetricsExpiry[providerName] = make(map[string]time.Time)
		enhanced.costDataCache[providerName] = make(map[string]*CostData)

		// Initialize rate limiters for this provider
		enhanced.rateLimiters[providerName] = createRateLimiterForProvider(providerName)
	}

	return enhanced
}

// Initialize creates an enhanced VM telemetry integration
func (ei *EnhancedCloudVMTelemetryIntegration) Initialize(ctx context.Context) error {
	// Initialize base integration
	if err := ei.CloudVMTelemetryIntegration.Initialize(ctx); err != nil {
		return err
	}

	// Start metric transformation pipeline
	ei.startMetricPipeline()

	// Set up provider health checks
	ei.startProviderHealthChecks(ctx)

	// Initialize event subscriptions if enabled
	if ei.enhancedConfig.EnableEventSubscriptions {
		ei.initializeEventSubscriptions(ctx)
	}

	// Prefetch VM inventories for faster initial startup
	ei.prefetchVMInventories(ctx)

	// Initialize cost data collection if enabled
	if ei.enhancedConfig.EnableCostMetrics {
		ei.initializeCostCollection(ctx)
	}

	return nil
}

// GetEnhancedVMStats retrieves VM stats with advanced features
func (ei *EnhancedCloudVMTelemetryIntegration) GetEnhancedVMStats(
	ctx context.Context,
	providerName string,
	vmID string,
	detailLevel monitoring.VMMetricDetailLevel,
) (*monitoring.VMStats, error) {
	// Check provider health
	if !ei.isProviderHealthy(providerName) {
		return nil, fmt.Errorf("provider %s is currently unhealthy", providerName)
	}

	// Check cache first
	if stats := ei.getFromMetricsCache(providerName, vmID); stats != nil {
		return stats, nil
	}

	// Get VM manager
	vmManager, exists := ei.vmManagers[providerName]
	if !exists {
		return nil, fmt.Errorf("VM manager for provider %s not found", providerName)
	}

	// Check if we need to rate limit
	limiter := ei.getRateLimiter(providerName)
	if limiter != nil {
		if err := limiter.Wait(ctx); err != nil {
			return nil, fmt.Errorf("rate limit wait error: %w", err)
		}
	}

	// Get basic VM stats with retry logic
	var stats *monitoring.VMStats
	var err error

	operation := func() (interface{}, error) {
		return vmManager.GetVMStats(ctx, vmID, detailLevel)
	}

	result, err := ei.executeWithRetry(ctx, operation, ei.enhancedConfig.RetryConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM stats after retries: %w", err)
	}

	stats = result.(*monitoring.VMStats)

	// Enhance with provider-specific metrics if enabled
	if ei.enhancedConfig.EnableNativeMetrics {
		stats = ei.enhanceWithNativeMetrics(ctx, providerName, vmID, stats)
	}

	// Add resource metadata if enabled
	if ei.enhancedConfig.CollectResourceMetadata {
		inventory := ei.getVMInventory(ctx, providerName, vmID)
		if inventory != nil {
			stats = ei.enrichStatsWithMetadata(stats, inventory)
		}
	}

	// Add cost data if enabled
	if ei.enhancedConfig.EnableCostMetrics {
		costData := ei.getCostData(providerName, vmID)
		if costData != nil {
			stats = ei.enrichStatsWithCostData(stats, costData)
		}
	}

	// Run the stats through the transformation pipeline
	if ei.metricPipeline != nil && ei.metricPipeline.Active {
		transformedStats, err := ei.applyMetricTransformations(stats)
		if err != nil {
			// Log the error but still use the original stats
			fmt.Printf("Error transforming metrics: %v\n", err)
		} else {
			stats = transformedStats
		}
	}

	// Update the cache
	ei.updateMetricsCache(providerName, vmID, stats)

	return stats, nil
}

// CreateEnhancedMultiProviderCollector creates an advanced multi-provider collector
func (ei *EnhancedCloudVMTelemetryIntegration) CreateEnhancedMultiProviderCollector(
	distributedCollector *monitoring.DistributedMetricCollector,
) (*monitoring.VMTelemetryCollector, error) {
	// Create enhanced multi-provider VM manager
	multiManager := NewEnhancedMultiProviderVMManager(ei)

	// Create VM telemetry collector configuration
	telemetryConfig := &monitoring.VMTelemetryCollectorConfig{
		CollectionInterval: 60 * time.Second,
		VMManager:          multiManager,
		EnabledMetrics: monitoring.VMMetricTypes{
			CPU:          true,
			Memory:       true,
			Disk:         true,
			Network:      true,
			IOPs:         true,
			ProcessStats: false,
			GuestMetrics: ei.enhancedConfig.EnableNativeMetrics,
		},
		Tags: map[string]string{
			"collector": "enhanced-multi-provider",
		},
		NodeID:      "cloud-multi-provider",
		DetailLevel: monitoring.StandardMetrics,
	}

	// Create VM telemetry collector
	collector := monitoring.NewVMTelemetryCollector(telemetryConfig, distributedCollector)

	return collector, nil
}

// EnhancedMultiProviderVMManager is an enhanced VM manager for multiple providers
type EnhancedMultiProviderVMManager struct {
	// Embed the base multi-provider VM manager
	*MultiProviderVMManager

	// Integration reference
	integration *EnhancedCloudVMTelemetryIntegration

	// VM inventory cache
	inventoryCache sync.Map

	// VM inventory expiry times
	inventoryExpiry sync.Map

	// Metric cache
	metricCache sync.Map

	// Metric expiry times
	metricExpiry sync.Map

	// Request throttling
	perProviderLimiter map[string]*RateLimiter
	limiterMutex       sync.RWMutex
}

// NewEnhancedMultiProviderVMManager creates a new enhanced multi-provider VM manager
func NewEnhancedMultiProviderVMManager(integration *EnhancedCloudVMTelemetryIntegration) *EnhancedMultiProviderVMManager {
	baseManager := NewMultiProviderVMManager()

	// Add all enabled managers
	for providerName, vmManager := range integration.vmManagers {
		baseManager.AddVMManager(providerName, vmManager)
	}

	enhanced := &EnhancedMultiProviderVMManager{
		MultiProviderVMManager: baseManager,
		integration:            integration,
		perProviderLimiter:     make(map[string]*RateLimiter),
	}

	// Initialize rate limiters
	for providerName := range integration.vmManagers {
		enhanced.perProviderLimiter[providerName] = integration.getRateLimiter(providerName)
	}

	return enhanced
}

// GetVMs returns a list of all VM IDs with caching and adaptive discovery
func (m *EnhancedMultiProviderVMManager) GetVMs(ctx context.Context) ([]string, error) {
	var allVMs []string
	var wg sync.WaitGroup
	var mutex sync.Mutex
	errChan := make(chan error, len(m.managers))

	// For each provider, get VMs with potential caching
	for providerName, manager := range m.managers {
		wg.Add(1)
		go func(provider string, vmManager monitoring.VMManagerInterface) {
			defer wg.Done()

			// Apply rate limiting per provider
			if limiter, exists := m.perProviderLimiter[provider]; exists {
				if err := limiter.Wait(ctx); err != nil {
					errChan <- fmt.Errorf("rate limit error for provider %s: %w", provider, err)
					return
				}
			}

			// Check provider health via integration
			if !m.integration.isProviderHealthy(provider) {
				errChan <- fmt.Errorf("provider %s is unhealthy, skipping VM discovery", provider)
				return
			}

			// Get VMs - use integration cache if possible
			vms, err := m.getVMsWithCache(ctx, provider, vmManager)
			if err != nil {
				errChan <- fmt.Errorf("error getting VMs from provider %s: %w", provider, err)
				return
			}

			// Add provider prefix and store results
			mutex.Lock()
			for _, vmID := range vms {
				allVMs = append(allVMs, fmt.Sprintf("%s-%s", provider, vmID))
			}
			mutex.Unlock()
		}(providerName, manager)
	}

	// Wait for all goroutines
	wg.Wait()
	close(errChan)

	// Check for errors, but don't fail completely if some providers failed
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 && len(errors) == len(m.managers) {
		// All providers failed
		return nil, fmt.Errorf("all cloud providers failed: %v", errors)
	}

	if len(errors) > 0 {
		// Log errors but continue with partial results
		fmt.Printf("Some providers failed during VM discovery: %v\n", errors)
	}

	return allVMs, nil
}

// GetVMStats retrieves stats for a specific VM with enhanced capabilities
func (m *EnhancedMultiProviderVMManager) GetVMStats(
	ctx context.Context,
	vmID string,
	detailLevel monitoring.VMMetricDetailLevel,
) (*monitoring.VMStats, error) {
	// Parse the VM ID to extract provider and original ID
	providerName, originalID, err := m.parseVMID(vmID)
	if err != nil {
		return nil, err
	}

	// Use the enhanced integration to get VM stats
	stats, err := m.integration.GetEnhancedVMStats(ctx, providerName, originalID, detailLevel)
	if err != nil {
		return nil, fmt.Errorf("error getting enhanced VM stats from provider %s: %w", providerName, err)
	}

	return stats, nil
}

// Helper methods and implementations of remaining functionality omitted for brevity
// These would include:
// - Cache management methods
// - Provider health check functions
// - Rate limiting implementation
// - Metric transformation pipeline
// - Event subscription handling
// - Cost data collection and integration

// MetricNormalizer normalizes metrics to standard ranges
type MetricNormalizer struct{}

func (t *MetricNormalizer) Name() string {
	return "metric_normalizer"
}

func (t *MetricNormalizer) Transform(stats *monitoring.VMStats) (*monitoring.VMStats, error) {
	// Create a deep copy to avoid modifying the original
	result := *stats

	// Convert to enhanced stats for easier manipulation
	enhanced := monitoring.ConvertInternalToEnhanced(&result)

	// Normalize CPU metrics to 0-100 range
	if enhanced.CPU != nil && enhanced.CPU.Usage > 0 {
		// If usage is in 0-1 range, convert to 0-100
		if enhanced.CPU.Usage <= 1.0 {
			enhanced.CPU.Usage *= 100.0
		}

		// Ensure usage doesn't exceed 100
		if enhanced.CPU.Usage > 100.0 {
			enhanced.CPU.Usage = 100.0
		}
	}

	// Normalize memory metrics
	if enhanced.Memory != nil {
		// Ensure UsagePercent is in 0-100 range
		if enhanced.Memory.UsagePercent <= 1.0 && enhanced.Memory.UsagePercent > 0 {
			enhanced.Memory.UsagePercent *= 100.0
			enhanced.Memory.Usage = enhanced.Memory.UsagePercent
		}

		// Ensure usage doesn't exceed 100
		if enhanced.Memory.UsagePercent > 100.0 {
			enhanced.Memory.UsagePercent = 100.0
			enhanced.Memory.Usage = 100.0
		}
	}

	// Normalize disk metrics
	if enhanced.Disks != nil {
		for diskName, disk := range enhanced.Disks {
			// Ensure UsagePercent is in 0-100 range
			if disk.UsagePercent <= 1.0 && disk.UsagePercent > 0 {
				disk.UsagePercent *= 100.0
				disk.Usage = disk.UsagePercent
			}

			// Ensure usage doesn't exceed 100
			if disk.UsagePercent > 100.0 {
				disk.UsagePercent = 100.0
				disk.Usage = 100.0
			}

			enhanced.Disks[diskName] = disk
		}
	}

	// Convert back to VMStats
	return monitoring.ConvertEnhancedToInternal(enhanced), nil
}

// AnomalyDetector detects anomalies in metrics
type AnomalyDetector struct {
	// Thresholds for anomaly detection
	CPUThreshold     float64
	MemoryThreshold  float64
	DiskThreshold    float64
	NetworkThreshold float64
}

func NewAnomalyDetector() *AnomalyDetector {
	return &AnomalyDetector{
		CPUThreshold:     90.0, // 90% CPU usage is considered high
		MemoryThreshold:  85.0, // 85% memory usage is considered high
		DiskThreshold:    90.0, // 90% disk usage is considered high
		NetworkThreshold: 80.0, // 80% of max network throughput is considered high
	}
}

func (t *AnomalyDetector) Name() string {
	return "anomaly_detector"
}

func (t *AnomalyDetector) Transform(stats *monitoring.VMStats) (*monitoring.VMStats, error) {
	// Create a deep copy to avoid modifying the original
	result := *stats

	// Convert to enhanced stats for easier manipulation
	enhanced := monitoring.ConvertInternalToEnhanced(&result)

	// Initialize metadata if needed
	if enhanced.Metadata == nil {
		enhanced.Metadata = make(map[string]string)
	}

	// Check CPU usage
	if enhanced.CPU != nil && enhanced.CPU.Usage > t.CPUThreshold {
		enhanced.Metadata["anomaly_cpu"] = "true"
		enhanced.Metadata["anomaly_cpu_value"] = fmt.Sprintf("%.2f", enhanced.CPU.Usage)
	}

	// Check memory usage
	if enhanced.Memory != nil && enhanced.Memory.UsagePercent > t.MemoryThreshold {
		enhanced.Metadata["anomaly_memory"] = "true"
		enhanced.Metadata["anomaly_memory_value"] = fmt.Sprintf("%.2f", enhanced.Memory.UsagePercent)
	}

	// Check disk usage
	if enhanced.Disks != nil {
		for diskName, disk := range enhanced.Disks {
			if disk.UsagePercent > t.DiskThreshold {
				enhanced.Metadata[fmt.Sprintf("anomaly_disk_%s", diskName)] = "true"
				enhanced.Metadata[fmt.Sprintf("anomaly_disk_%s_value", diskName)] = fmt.Sprintf("%.2f", disk.UsagePercent)
			}
		}
	}

	// Convert back to VMStats
	return monitoring.ConvertEnhancedToInternal(enhanced), nil
}

// MetricAggregator aggregates metrics
type MetricAggregator struct{}

func (t *MetricAggregator) Name() string {
	return "metric_aggregator"
}

func (t *MetricAggregator) Transform(stats *monitoring.VMStats) (*monitoring.VMStats, error) {
	// Create a deep copy to avoid modifying the original
	result := *stats

	// Convert to enhanced stats for easier manipulation
	enhanced := monitoring.ConvertInternalToEnhanced(&result)

	// Initialize metadata if needed
	if enhanced.Metadata == nil {
		enhanced.Metadata = make(map[string]string)
	}

	// Calculate total disk usage
	if enhanced.Disks != nil && len(enhanced.Disks) > 0 {
		var totalDiskUsed float64
		var totalDiskSize float64

		for _, disk := range enhanced.Disks {
			totalDiskUsed += disk.Used
			totalDiskSize += disk.Size
		}

		// Add aggregated disk metrics
		enhanced.Metadata["total_disk_used_bytes"] = fmt.Sprintf("%.0f", totalDiskUsed)
		enhanced.Metadata["total_disk_size_bytes"] = fmt.Sprintf("%.0f", totalDiskSize)

		if totalDiskSize > 0 {
			diskUsagePercent := (totalDiskUsed / totalDiskSize) * 100
			enhanced.Metadata["total_disk_usage_percent"] = fmt.Sprintf("%.2f", diskUsagePercent)
		}
	}

	// Calculate total network throughput
	if enhanced.Networks != nil && len(enhanced.Networks) > 0 {
		var totalRxBytes float64
		var totalTxBytes float64

		for _, network := range enhanced.Networks {
			totalRxBytes += network.RxBytes
			totalTxBytes += network.TxBytes
		}

		// Add aggregated network metrics
		enhanced.Metadata["total_network_rx_bytes"] = fmt.Sprintf("%.0f", totalRxBytes)
		enhanced.Metadata["total_network_tx_bytes"] = fmt.Sprintf("%.0f", totalTxBytes)
		enhanced.Metadata["total_network_bytes"] = fmt.Sprintf("%.0f", totalRxBytes+totalTxBytes)
	}

	// Convert back to VMStats
	return monitoring.ConvertEnhancedToInternal(enhanced), nil
}

// createDefaultMetricPipeline creates the default metric transformation pipeline
func createDefaultMetricPipeline() *MetricTransformationPipeline {
	pipeline := &MetricTransformationPipeline{
		Transformers: []MetricTransformer{
			&MetricNormalizer{},
			NewAnomalyDetector(),
			&MetricAggregator{},
		},
		Pipeline: make(chan *monitoring.VMStats, 100),
		Workers:  3,
		Errors:   make(chan error, 10),
	}
	return pipeline
}

// createRateLimiterForProvider creates appropriate rate limiter for a provider
func createRateLimiterForProvider(providerName string) *RateLimiter {
	var rps int
	switch providerName {
	case "aws":
		rps = 5
	case "azure":
		rps = 12
	case "gcp":
		rps = 10
	default:
		rps = 5
	}

	return &RateLimiter{
		Provider:        providerName,
		RequestsPerSec:  rps,
		BurstSize:       rps * 5,
		LastRefill:      time.Now(),
		RefillRate:      float64(rps),
		RegionLimiters:  make(map[string]*RateLimiter),
		MaxRequestQueue: 100,
	}
}

// executeWithRetry implements retry logic for cloud operations
func (ei *EnhancedCloudVMTelemetryIntegration) executeWithRetry(
	ctx context.Context,
	operation func() (interface{}, error),
	config *RetryConfig,
) (interface{}, error) {
	var lastErr error
	var result interface{}

	for attempt := 0; attempt <= config.MaxRetries; attempt++ {
		// Execute the operation
		result, lastErr = operation()
		if lastErr == nil {
			return result, nil
		}

		// Check if we should retry this error
		shouldRetry := false
		errString := lastErr.Error()
		for _, retryableErr := range config.RetryableErrors {
			if containsString(errString, retryableErr) {
				shouldRetry = true
				break
			}
		}

		if !shouldRetry || attempt == config.MaxRetries {
			break
		}

		// Calculate backoff time
		backoff := config.InitialBackoff * time.Duration(int(math.Pow(2, float64(attempt)))*int(config.BackoffMultiplier))
		if backoff > config.MaxBackoff {
			backoff = config.MaxBackoff
		}

		// Wait with context
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("operation cancelled during retry: %w", ctx.Err())
		case <-time.After(backoff):
			// Continue with next attempt
		}
	}

	return nil, lastErr
}

// containsString checks if a string contains a substring
func containsString(s, substr string) bool {
	return s != "" && substr != "" && s != substr && (len(s) >= len(substr)) && s[0:len(substr)] == substr
}

// parseVMID parses a VM ID into provider name and original ID
func (m *EnhancedMultiProviderVMManager) parseVMID(vmID string) (string, string, error) {
	for provider := range m.managers {
		prefix := fmt.Sprintf("%s-", provider)
		if len(vmID) > len(prefix) && vmID[:len(prefix)] == prefix {
			return provider, vmID[len(prefix):], nil
		}
	}
	return "", "", fmt.Errorf("invalid VM ID format: %s", vmID)
}

// Wait implements the rate limiter wait method
func (r *RateLimiter) Wait(ctx context.Context) error {
	r.LimiterMutex.Lock()
	defer r.LimiterMutex.Unlock()

	// Refill tokens based on time elapsed
	now := time.Now()
	elapsed := now.Sub(r.LastRefill).Seconds()
	r.LastRefill = now

	newTokens := int(elapsed * r.RefillRate)
	r.Tokens += newTokens
	if r.Tokens > r.BurstSize {
		r.Tokens = r.BurstSize
	}

	// If we have a token, consume it and return immediately
	if r.Tokens > 0 {
		r.Tokens--
		return nil
	}

	// Calculate wait time to get next token
	waitTime := time.Duration(float64(time.Second) / r.RefillRate)

	// Check pending requests
	if r.PendingRequests >= r.MaxRequestQueue {
		return fmt.Errorf("rate limit queue full for provider %s", r.Provider)
	}

	r.PendingRequests++
	r.LimiterMutex.Unlock()

	// Wait for either context cancellation or timeout
	select {
	case <-ctx.Done():
		r.LimiterMutex.Lock()
		r.PendingRequests--
		return ctx.Err()
	case <-time.After(waitTime):
		r.LimiterMutex.Lock()
		r.PendingRequests--
		r.Tokens--
		return nil
	}
}

// isProviderHealthy checks if a provider is healthy
func (ei *EnhancedCloudVMTelemetryIntegration) isProviderHealthy(providerName string) bool {
	ei.providerStatusMutex.RLock()
	defer ei.providerStatusMutex.RUnlock()

	status, exists := ei.providerStatus[providerName]
	if !exists {
		return false
	}

	return status.Healthy
}

// getRateLimiter gets the rate limiter for a provider
func (ei *EnhancedCloudVMTelemetryIntegration) getRateLimiter(providerName string) *RateLimiter {
	ei.rateLimitersMutex.RLock()
	defer ei.rateLimitersMutex.RUnlock()

	return ei.rateLimiters[providerName]
}

// startMetricPipeline starts the metric transformation pipeline
func (ei *EnhancedCloudVMTelemetryIntegration) startMetricPipeline() {
	if ei.metricPipeline == nil {
		return
	}

	ei.metricPipeline.Active = true

	// Start workers
	for i := 0; i < ei.metricPipeline.Workers; i++ {
		ei.metricPipeline.WaitGroup.Add(1)
		go func() {
			defer ei.metricPipeline.WaitGroup.Done()

			for ei.metricPipeline.Active {
				select {
				case stats, ok := <-ei.metricPipeline.Pipeline:
					if !ok {
						return
					}

					// Apply all transformations
					for _, transformer := range ei.metricPipeline.Transformers {
						var err error
						stats, err = transformer.Transform(stats)
						if err != nil {
							select {
							case ei.metricPipeline.Errors <- err:
								// Error reported
							default:
								// Error channel full, just log
								fmt.Printf("Transformation error: %v\n", err)
							}
							break
						}
					}
				}
			}
		}()
	}
}

// applyMetricTransformations applies metric transformations
func (ei *EnhancedCloudVMTelemetryIntegration) applyMetricTransformations(stats *monitoring.VMStats) (*monitoring.VMStats, error) {
	// Create a deep copy of stats to avoid modifying the original
	statsCopy := *stats

	// Apply each transformer in sequence
	result := &statsCopy
	var err error

	for _, transformer := range ei.metricPipeline.Transformers {
		result, err = transformer.Transform(result)
		if err != nil {
			return stats, fmt.Errorf("transformation %s failed: %w", transformer.Name(), err)
		}
	}

	return result, nil
}

// getVMsWithCache gets VM list with caching
func (m *EnhancedMultiProviderVMManager) getVMsWithCache(
	ctx context.Context,
	provider string,
	vmManager monitoring.VMManagerInterface,
) ([]string, error) {
	// Get cached inventory if available
	cachedVMs, found := m.integration.getCachedVMInventoryList(provider)
	if found {
		return cachedVMs, nil
	}

	// Get VMs from provider
	vms, err := vmManager.GetVMs(ctx)
	if err != nil {
		return nil, fmt.Errorf("error getting VMs from provider %s: %w", provider, err)
	}

	// Cache the results
	if m.integration != nil {
		m.integration.updateVMInventoryList(provider, vms)
	}

	return vms, nil
}

// getCachedVMInventoryList gets cached VM list for a provider
func (ei *EnhancedCloudVMTelemetryIntegration) getCachedVMInventoryList(provider string) ([]string, bool) {
	ei.vmInventoryCacheMutex.RLock()
	defer ei.vmInventoryCacheMutex.RUnlock()

	cache, exists := ei.vmInventoryCache[provider]
	if !exists {
		return nil, false
	}

	// Check if cache is still valid
	now := time.Now()
	anyValid := false
	var vmIDs []string

	for vmID := range cache {
		expiryTime, exists := ei.vmInventoryExpiry[provider][vmID]
		if exists && now.Before(expiryTime) {
			anyValid = true
			vmIDs = append(vmIDs, vmID)
		}
	}

	return vmIDs, anyValid
}

// updateVMInventoryList updates the VM inventory cache for a provider
func (ei *EnhancedCloudVMTelemetryIntegration) updateVMInventoryList(provider string, vmIDs []string) {
	ei.vmInventoryCacheMutex.Lock()
	defer ei.vmInventoryCacheMutex.Unlock()

	cache, exists := ei.vmInventoryCache[provider]
	if !exists {
		cache = make(map[string]*VMInventory)
		ei.vmInventoryCache[provider] = cache
	}

	expiryMap, exists := ei.vmInventoryExpiry[provider]
	if !exists {
		expiryMap = make(map[string]time.Time)
		ei.vmInventoryExpiry[provider] = expiryMap
	}

	// Update expiry for each VM
	expiryTime := time.Now().Add(ei.enhancedConfig.InventoryCacheTTL)
	for _, vmID := range vmIDs {
		// If we don't have full inventory data, at least mark that the VM exists
		if _, exists := cache[vmID]; !exists {
			cache[vmID] = &VMInventory{
				VMID:          vmID,
				ProviderID:    provider,
				LastRefreshed: time.Now(),
			}
		}
		expiryMap[vmID] = expiryTime
	}
}

// updateMetricsCache updates the VM metrics cache
func (ei *EnhancedCloudVMTelemetryIntegration) updateMetricsCache(provider string, vmID string, stats *monitoring.VMStats) {
	ei.vmMetricsCacheMutex.Lock()
	defer ei.vmMetricsCacheMutex.Unlock()

	cache, exists := ei.vmMetricsCache[provider]
	if !exists {
		cache = make(map[string]*monitoring.VMStats)
		ei.vmMetricsCache[provider] = cache
	}

	expiryMap, exists := ei.vmMetricsExpiry[provider]
	if !exists {
		expiryMap = make(map[string]time.Time)
		ei.vmMetricsExpiry[provider] = expiryMap
	}

	// Update cache and expiry
	cache[vmID] = stats
	expiryMap[vmID] = time.Now().Add(ei.enhancedConfig.MetricCacheTTL)
}

// getFromMetricsCache gets VM metrics from cache if available
func (ei *EnhancedCloudVMTelemetryIntegration) getFromMetricsCache(provider string, vmID string) *monitoring.VMStats {
	ei.vmMetricsCacheMutex.RLock()
	defer ei.vmMetricsCacheMutex.RUnlock()

	cache, exists := ei.vmMetricsCache[provider]
	if !exists {
		return nil
	}

	stats, exists := cache[vmID]
	if !exists {
		return nil
	}

	// Check if cache is still valid
	expiryMap, exists := ei.vmMetricsExpiry[provider]
	if !exists {
		return nil
	}

	expiryTime, exists := expiryMap[vmID]
	if !exists || time.Now().After(expiryTime) {
		return nil
	}

	return stats
}

// getVMInventory gets VM inventory data
func (ei *EnhancedCloudVMTelemetryIntegration) getVMInventory(ctx context.Context, provider string, vmID string) *VMInventory {
	ei.vmInventoryCacheMutex.RLock()

	// Check cache first
	cache, exists := ei.vmInventoryCache[provider]
	if exists {
		cachedInventory, exists := cache[vmID]
		if exists {
			expiryMap, exists := ei.vmInventoryExpiry[provider]
			if exists {
				expiryTime, exists := expiryMap[vmID]
				if exists && time.Now().Before(expiryTime) {
					ei.vmInventoryCacheMutex.RUnlock()
					return cachedInventory
				}
			}
		}
	}
	ei.vmInventoryCacheMutex.RUnlock()

	// Not in cache or expired, fetch it
	// This is a placeholder - in a real implementation we would fetch VM metadata
	// from the cloud provider and create a detailed inventory object

	// For now, just create a basic inventory object
	newInventory := &VMInventory{
		VMID:          vmID,
		ProviderID:    provider,
		LastRefreshed: time.Now(),
	}

	// Cache it
	ei.vmInventoryCacheMutex.Lock()
	defer ei.vmInventoryCacheMutex.Unlock()

	if _, exists := ei.vmInventoryCache[provider]; !exists {
		ei.vmInventoryCache[provider] = make(map[string]*VMInventory)
	}

	if _, exists := ei.vmInventoryExpiry[provider]; !exists {
		ei.vmInventoryExpiry[provider] = make(map[string]time.Time)
	}

	ei.vmInventoryCache[provider][vmID] = newInventory
	ei.vmInventoryExpiry[provider][vmID] = time.Now().Add(ei.enhancedConfig.InventoryCacheTTL)

	return newInventory
}

// getCostData gets cost data for a VM
func (ei *EnhancedCloudVMTelemetryIntegration) getCostData(provider string, vmID string) *CostData {
	if !ei.enhancedConfig.EnableCostMetrics {
		return nil
	}

	ei.costDataCacheMutex.RLock()
	defer ei.costDataCacheMutex.RUnlock()

	cache, exists := ei.costDataCache[provider]
	if !exists {
		return nil
	}

	return cache[vmID]
}

// enhanceWithNativeMetrics enhances stats with provider-specific metrics
func (ei *EnhancedCloudVMTelemetryIntegration) enhanceWithNativeMetrics(
	ctx context.Context,
	provider string,
	vmID string,
	stats *monitoring.VMStats,
) *monitoring.VMStats {
	// Create a native metric collector with the same cache TTL as the integration
	collector := NewNativeMetricCollector(ei.providerManager, ei.enhancedConfig.MetricCacheTTL)

	// Use the collector to enhance the stats with native metrics
	enhancedStats, err := collector.EnhanceWithNativeMetrics(ctx, provider, vmID, stats)
	if err != nil {
		// Log the error but return the original stats
		fmt.Printf("Warning: failed to enhance with native metrics: %v\n", err)
		return stats
	}

	return enhancedStats
}

// enrichStatsWithMetadata enriches stats with metadata
func (ei *EnhancedCloudVMTelemetryIntegration) enrichStatsWithMetadata(
	stats *monitoring.VMStats,
	inventory *VMInventory,
) *monitoring.VMStats {
	// Convert to EnhancedCloudVMStats for easier manipulation
	enhancedStats := monitoring.ConvertInternalToEnhanced(stats)

	// Add metadata from inventory
	if inventory != nil {
		// Initialize metadata if needed
		if enhancedStats.Metadata == nil {
			enhancedStats.Metadata = make(map[string]string)
		}

		// Add inventory fields to metadata
		if inventory.InstanceType != "" {
			enhancedStats.Metadata["instance_type"] = inventory.InstanceType
		}

		if inventory.Region != "" {
			enhancedStats.Metadata["region"] = inventory.Region
		}

		if inventory.Zone != "" {
			enhancedStats.Metadata["zone"] = inventory.Zone
		}

		// Copy inventory tags to VM tags
		if len(inventory.Tags) > 0 {
			if enhancedStats.Tags == nil {
				enhancedStats.Tags = make(map[string]string)
			}

			for k, v := range inventory.Tags {
				enhancedStats.Tags[k] = v
			}
		}
	}

	// Convert back to internal VMStats
	return monitoring.ConvertEnhancedToInternal(enhancedStats)
}

// enrichStatsWithCostData enriches stats with cost data
func (ei *EnhancedCloudVMTelemetryIntegration) enrichStatsWithCostData(
	stats *monitoring.VMStats,
	costData *CostData,
) *monitoring.VMStats {
	// Create a CloudVMStats version that we can enrich with cost data
	cloudStats := monitoring.ConvertInternalToCloudStats(stats)

	// Add cost data
	if costData != nil {
		// Initialize tags if needed
		if cloudStats.Tags == nil {
			cloudStats.Tags = make(map[string]string)
		}

		// Add cost information to tags
		cloudStats.Tags["hourly_cost"] = fmt.Sprintf("%.4f", costData.HourlyCost)
		cloudStats.Tags["cost_currency"] = costData.BillingCurrency

		// Add more detailed cost data to metadata
		if cloudStats.Metadata == nil {
			cloudStats.Metadata = make(map[string]string)
		}

		cloudStats.Metadata["daily_cost"] = fmt.Sprintf("%.4f", costData.DailyCost)
		cloudStats.Metadata["month_to_date_cost"] = fmt.Sprintf("%.4f", costData.MonthToDateCost)

		if costData.ForecastedCost > 0 {
			cloudStats.Metadata["forecasted_cost"] = fmt.Sprintf("%.4f", costData.ForecastedCost)
		}
	}

	// Convert back to internal VMStats
	return monitoring.ConvertCloudToInternalStats(cloudStats)
}

// startProviderHealthChecks starts periodic provider health checks
func (ei *EnhancedCloudVMTelemetryIntegration) startProviderHealthChecks(ctx context.Context) {
	// Health check interval (use 1 minute or configurable value)
	healthCheckInterval := 1 * time.Minute

	go func() {
		ticker := time.NewTicker(healthCheckInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				ei.checkAllProvidersHealth(ctx)
			}
		}
	}()
}

// checkAllProvidersHealth checks the health of all enabled providers
func (ei *EnhancedCloudVMTelemetryIntegration) checkAllProvidersHealth(ctx context.Context) {
	for _, providerName := range ei.enhancedConfig.EnabledProviders {
		go func(provider string) {
			healthy, err := ei.checkProviderHealth(ctx, provider)

			ei.providerStatusMutex.Lock()
			defer ei.providerStatusMutex.Unlock()

			status := ei.providerStatus[provider]
			status.LastChecked = time.Now()

			if !healthy {
				status.Healthy = false
				status.FailureCount++
				if err != nil {
					status.LastError = err.Error()
				}
			} else {
				status.Healthy = true
				status.FailureCount = 0
				status.LastError = ""
			}

			ei.providerStatus[provider] = status
		}(providerName)
	}
}

// checkProviderHealth checks if a provider is healthy by making a simple API call
func (ei *EnhancedCloudVMTelemetryIntegration) checkProviderHealth(ctx context.Context, providerName string) (bool, error) {
	provider, err := ei.providerManager.GetProvider(providerName)
	if err != nil {
		return false, fmt.Errorf("failed to get provider: %w", err)
	}

	// Use a simple API call to check health (e.g., list regions)
	startTime := time.Now()
	_, err = provider.GetRegions(ctx)
	responseTime := time.Since(startTime).Milliseconds()

	// Update response time in provider status
	ei.providerStatusMutex.Lock()
	status := ei.providerStatus[providerName]
	status.AvgResponseTimeMS = responseTime
	ei.providerStatus[providerName] = status
	ei.providerStatusMutex.Unlock()

	return err == nil, err
}

// prefetchVMInventories prefetches VM inventories for initial startup
func (ei *EnhancedCloudVMTelemetryIntegration) prefetchVMInventories(ctx context.Context) {
	for _, providerName := range ei.enhancedConfig.EnabledProviders {
		go func(provider string) {
			// Skip if provider is unhealthy
			if !ei.isProviderHealthy(provider) {
				fmt.Printf("Skipping inventory prefetch for unhealthy provider: %s\n", provider)
				return
			}

			// Get VM manager
			vmManager, exists := ei.vmManagers[provider]
			if !exists {
				fmt.Printf("VM manager for provider %s not found\n", provider)
				return
			}

			// Get VMs with rate limiting
			limiter := ei.getRateLimiter(provider)
			if limiter != nil {
				if err := limiter.Wait(ctx); err != nil {
					fmt.Printf("Rate limit wait error during prefetch: %v\n", err)
					return
				}
			}

			// Get VM list
			vms, err := vmManager.GetVMs(ctx)
			if err != nil {
				fmt.Printf("Error prefetching VM list for provider %s: %v\n", provider, err)
				return
			}

			// Update inventory cache
			ei.updateVMInventoryList(provider, vms)

			// Optionally prefetch detailed inventory for each VM
			if ei.enhancedConfig.CollectResourceMetadata {
				for _, vmID := range vms {
					// Use a separate goroutine with rate limiting for each VM
					go func(id string) {
						if limiter != nil {
							if err := limiter.Wait(ctx); err != nil {
								return
							}
						}

						// Fetch and cache detailed inventory
						_ = ei.getVMInventory(ctx, provider, id)
					}(vmID)
				}
			}
		}(providerName)
	}
}

// initializeEventSubscriptions initializes provider event subscriptions
func (ei *EnhancedCloudVMTelemetryIntegration) initializeEventSubscriptions(ctx context.Context) {
	if !ei.enhancedConfig.EnableEventSubscriptions {
		return
	}

	for _, providerName := range ei.enhancedConfig.EnabledProviders {
		// Create event subscription for this provider
		eventChan := make(chan CloudEvent, 100)
		errorChan := make(chan error, 10)

		// Create cancellable context for this subscription
		subCtx, cancelFunc := context.WithCancel(ctx)

		// Create subscription
		subscription := EventSubscription{
			Provider:      providerName,
			ResourceType:  "vm",
			EventTypes:    []string{"state_change", "resize", "migration"},
			LastEventTime: time.Now(),
			Active:        true,
			EventChannel:  eventChan,
			Errors:        errorChan,
			CancelFunc:    cancelFunc,
		}

		// Store subscription
		ei.eventSubscriptionsMutex.Lock()
		ei.eventSubscriptions[providerName] = subscription
		ei.eventSubscriptionsMutex.Unlock()

		// Start event listener
		go ei.listenForEvents(subCtx, providerName, eventChan, errorChan)
	}
}

// listenForEvents listens for events from a provider
func (ei *EnhancedCloudVMTelemetryIntegration) listenForEvents(
	ctx context.Context,
	providerName string,
	eventChan chan CloudEvent,
	errorChan chan error,
) {
	// This would typically use provider-specific APIs to subscribe to events
	// For now, we'll implement a placeholder that simulates events

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case err := <-errorChan:
			fmt.Printf("Event subscription error for provider %s: %v\n", providerName, err)
		case event := <-eventChan:
			// Process the event
			ei.processCloudEvent(event)
		case <-ticker.C:
			// Simulate checking for events periodically
			// In a real implementation, this would poll the provider's event API
			fmt.Printf("Checking for events from provider %s\n", providerName)
		}
	}
}

// processCloudEvent processes a cloud event
func (ei *EnhancedCloudVMTelemetryIntegration) processCloudEvent(event CloudEvent) {
	// Update VM inventory if it's a state change event
	if event.EventType == "state_change" {
		// Invalidate cache for this VM
		ei.vmInventoryCacheMutex.Lock()
		if expiryMap, exists := ei.vmInventoryExpiry[event.Provider]; exists {
			delete(expiryMap, event.ResourceID)
		}
		ei.vmInventoryCacheMutex.Unlock()

		// Invalidate metrics cache for this VM
		ei.vmMetricsCacheMutex.Lock()
		if expiryMap, exists := ei.vmMetricsExpiry[event.Provider]; exists {
			delete(expiryMap, event.ResourceID)
		}
		ei.vmMetricsCacheMutex.Unlock()
	}

	// Store event in enhanced stats for this VM
	enhancedStats := ei.getEnhancedStatsForVM(event.Provider, event.ResourceID)
	if enhancedStats != nil {
		enhancedStats.CloudEvents = append(enhancedStats.CloudEvents,
			fmt.Sprintf("%s: %s", event.EventTime.Format(time.RFC3339), event.EventType))
	}
}

// getEnhancedStatsForVM gets enhanced stats for a VM
func (ei *EnhancedCloudVMTelemetryIntegration) getEnhancedStatsForVM(provider, vmID string) *monitoring.EnhancedCloudVMStats {
	ei.vmMetricsCacheMutex.RLock()
	defer ei.vmMetricsCacheMutex.RUnlock()

	if cacheMap, exists := ei.vmMetricsCache[provider]; exists {
		if stats, exists := cacheMap[vmID]; exists {
			return monitoring.ConvertInternalToEnhanced(stats)
		}
	}

	return nil
}

// initializeCostCollection initializes cost data collection
func (ei *EnhancedCloudVMTelemetryIntegration) initializeCostCollection(ctx context.Context) {
	if !ei.enhancedConfig.EnableCostMetrics {
		return
	}

	// Create cost collector with appropriate refresh interval
	costCollector := NewCostDataCollector(ei.providerManager, 6*time.Hour)

	// Initialize the collector
	if err := costCollector.Initialize(ctx); err != nil {
		fmt.Printf("Error initializing cost collector: %v\n", err)
		return
	}

	// Start background collection
	if err := costCollector.StartBackgroundCollection(ctx); err != nil {
		fmt.Printf("Error starting cost collection: %v\n", err)
		return
	}

	// Perform initial cost data collection
	costCollector.RefreshAllCostData(ctx)

	// Store cost data in our cache
	for _, providerName := range ei.enhancedConfig.EnabledProviders {
		provider, err := ei.providerManager.GetProvider(providerName)
		if err != nil {
			continue
		}

		// Get all instances
		var instances []Instance
		switch p := provider.(type) {
		case *AWSProvider:
			instances, _ = p.GetInstances(ctx, ListOptions{})
		case *AzureProvider:
			instances, _ = p.GetInstances(ctx, ListOptions{})
		case *GCPProvider:
			instances, _ = p.GetInstances(ctx, ListOptions{})
		}

		// For each instance, get cost data and cache it
		for _, instance := range instances {
			costData, err := costCollector.GetCostData(ctx, providerName, instance.ID)
			if err != nil {
				continue
			}

			// Convert to our internal CostData type
			internalCostData := &CostData{
				ResourceID:       costData.ResourceID,
				Provider:         costData.Provider,
				HourlyCost:       costData.HourlyCost,
				DailyCost:        costData.DailyCost,
				MonthToDateCost:  costData.MonthToDateCost,
				ForecastedCost:   costData.ForecastedCost,
				BillingCurrency:  costData.BillingCurrency,
				LastUpdated:      costData.LastUpdated,
				CostComponents:   costData.CostComponents,
				ResourceTags:     costData.ResourceTags,
				BillingTags:      costData.BillingTags,
				BillingAccountID: costData.BillingAccountID,
			}

			// Update our cache
			ei.costDataCacheMutex.Lock()
			if _, exists := ei.costDataCache[providerName]; !exists {
				ei.costDataCache[providerName] = make(map[string]*CostData)
			}
			ei.costDataCache[providerName][instance.ID] = internalCostData
			ei.costDataCacheMutex.Unlock()
		}
	}
}
