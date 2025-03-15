package cloud

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"
)

// HybridCloudOrchestrator manages resources across multiple cloud providers
type HybridCloudOrchestrator struct {
	// Map of provider name to provider interface
	providers map[string]Provider

	// Lock for thread safety
	lock sync.RWMutex

	// Default provider to use if not specified
	defaultProvider string

	// Provider selection strategy
	selectionStrategy ProviderSelectionStrategy

	// Cache for pricing information
	pricingCache map[string]map[string]float64

	// Cache expiration
	cacheExpiration time.Duration

	// Last cache update time
	lastCacheUpdate time.Time
}

// ProviderSelectionStrategy defines strategies for selecting a cloud provider
type ProviderSelectionStrategy string

const (
	// SelectByPrice chooses the provider with the lowest price for a resource
	SelectByPrice ProviderSelectionStrategy = "price"

	// SelectByAvailability chooses the provider with highest availability
	SelectByAvailability ProviderSelectionStrategy = "availability"

	// SelectByPerformance chooses the provider with best expected performance for a resource
	SelectByPerformance ProviderSelectionStrategy = "performance"

	// SelectByLatency chooses the provider with lowest latency to a specified region
	SelectByLatency ProviderSelectionStrategy = "latency"

	// SelectByDistribution distributes resources evenly across providers
	SelectByDistribution ProviderSelectionStrategy = "distribution"
)

// HybridResourceAllocation represents an allocation of resources across providers
type HybridResourceAllocation struct {
	// Resources allocated per provider
	AllocationByProvider map[string][]string

	// Total cost estimate across all providers
	TotalCostEstimate float64

	// Cost estimate per provider
	CostByProvider map[string]float64

	// Creation timestamp
	CreatedAt time.Time
}

// NewHybridCloudOrchestrator creates a new hybrid cloud orchestrator
func NewHybridCloudOrchestrator() *HybridCloudOrchestrator {
	return &HybridCloudOrchestrator{
		providers:         make(map[string]Provider),
		selectionStrategy: SelectByPrice,
		pricingCache:      make(map[string]map[string]float64),
		cacheExpiration:   1 * time.Hour,
		lastCacheUpdate:   time.Time{}, // Zero time
	}
}

// RegisterProvider adds a cloud provider to the orchestrator
func (o *HybridCloudOrchestrator) RegisterProvider(provider Provider) error {
	o.lock.Lock()
	defer o.lock.Unlock()

	name := provider.Name()
	if _, exists := o.providers[name]; exists {
		return fmt.Errorf("provider %q is already registered", name)
	}

	o.providers[name] = provider

	// If this is the first provider, set it as default
	if len(o.providers) == 1 {
		o.defaultProvider = name
	}

	return nil
}

// SetDefaultProvider sets the default provider
func (o *HybridCloudOrchestrator) SetDefaultProvider(name string) error {
	o.lock.Lock()
	defer o.lock.Unlock()

	if _, exists := o.providers[name]; !exists {
		return fmt.Errorf("provider %q is not registered", name)
	}

	o.defaultProvider = name
	return nil
}

// SetSelectionStrategy sets the provider selection strategy
func (o *HybridCloudOrchestrator) SetSelectionStrategy(strategy ProviderSelectionStrategy) {
	o.lock.Lock()
	defer o.lock.Unlock()

	o.selectionStrategy = strategy
}

// GetRegisteredProviders returns a list of registered provider names
func (o *HybridCloudOrchestrator) GetRegisteredProviders() []string {
	o.lock.RLock()
	defer o.lock.RUnlock()

	providers := make([]string, 0, len(o.providers))
	for name := range o.providers {
		providers = append(providers, name)
	}

	sort.Strings(providers)
	return providers
}

// GetProvider returns a provider by name
func (o *HybridCloudOrchestrator) GetProvider(name string) (Provider, error) {
	o.lock.RLock()
	defer o.lock.RUnlock()

	provider, exists := o.providers[name]
	if !exists {
		return nil, fmt.Errorf("provider %q is not registered", name)
	}

	return provider, nil
}

// GetDefaultProvider returns the default provider
func (o *HybridCloudOrchestrator) GetDefaultProvider() (Provider, string) {
	o.lock.RLock()
	defer o.lock.RUnlock()

	provider, exists := o.providers[o.defaultProvider]
	if !exists && len(o.providers) > 0 {
		// If default provider doesn't exist but we have other providers,
		// return the first one alphabetically
		names := make([]string, 0, len(o.providers))
		for name := range o.providers {
			names = append(names, name)
		}
		sort.Strings(names)
		return o.providers[names[0]], names[0]
	}

	return provider, o.defaultProvider
}

// SelectProvider chooses the best provider based on the current strategy
func (o *HybridCloudOrchestrator) SelectProvider(ctx context.Context, resourceType string, specs interface{}) (Provider, string, error) {
	o.lock.RLock()
	defer o.lock.RUnlock()

	if len(o.providers) == 0 {
		return nil, "", fmt.Errorf("no providers registered")
	}

	switch o.selectionStrategy {
	case SelectByPrice:
		return o.selectProviderByPrice(ctx, resourceType, specs)
	case SelectByAvailability:
		return o.selectProviderByAvailability(ctx)
	case SelectByPerformance:
		return o.selectProviderByPerformance(ctx, resourceType, specs)
	case SelectByLatency:
		return o.selectProviderByLatency(ctx, specs)
	case SelectByDistribution:
		return o.selectProviderByDistribution(ctx)
	default:
		// Default to price-based selection
		return o.selectProviderByPrice(ctx, resourceType, specs)
	}
}

// selectProviderByPrice selects the provider with the lowest price for a resource
func (o *HybridCloudOrchestrator) selectProviderByPrice(ctx context.Context, resourceType string, specs interface{}) (Provider, string, error) {
	// Refresh pricing cache if needed
	if time.Since(o.lastCacheUpdate) > o.cacheExpiration {
		if err := o.refreshPricingCache(ctx); err != nil {
			// Log the error but continue with potentially stale data
			fmt.Printf("Error refreshing pricing cache: %v\n", err)
		}
	}

	var bestProvider Provider
	var bestProviderName string
	var bestPrice float64 = -1 // -1 means no price found yet

	// Determine the resource key based on specs
	var resourceKey string
	switch specs := specs.(type) {
	case InstanceSpecs:
		resourceKey = specs.InstanceType
	case StorageVolumeSpecs:
		resourceKey = specs.Type
	default:
		resourceKey = "" // Use empty string as fallback
	}

	// Find the provider with the lowest price
	for name, provider := range o.providers {
		if pricing, ok := o.pricingCache[name]; ok {
			if price, ok := pricing[resourceKey]; ok {
				if bestPrice == -1 || price < bestPrice {
					bestPrice = price
					bestProvider = provider
					bestProviderName = name
				}
			}
		}
	}

	// If no specific pricing found, return default provider
	if bestProvider == nil {
		defaultProvider, defaultName := o.GetDefaultProvider()
		return defaultProvider, defaultName, nil
	}

	return bestProvider, bestProviderName, nil
}

// selectProviderByAvailability selects the provider with highest availability
func (o *HybridCloudOrchestrator) selectProviderByAvailability(ctx context.Context) (Provider, string, error) {
	// For now, just use the default provider
	// In a real implementation, we would check health/status of providers
	defaultProvider, defaultName := o.GetDefaultProvider()
	return defaultProvider, defaultName, nil
}

// selectProviderByPerformance selects the provider with best expected performance
func (o *HybridCloudOrchestrator) selectProviderByPerformance(ctx context.Context, resourceType string, specs interface{}) (Provider, string, error) {
	// For now, just use the default provider
	// In a real implementation, we would analyze performance metrics
	defaultProvider, defaultName := o.GetDefaultProvider()
	return defaultProvider, defaultName, nil
}

// selectProviderByLatency selects the provider with lowest latency
func (o *HybridCloudOrchestrator) selectProviderByLatency(ctx context.Context, specs interface{}) (Provider, string, error) {
	// For now, just use the default provider
	// In a real implementation, we would measure or estimate latency
	defaultProvider, defaultName := o.GetDefaultProvider()
	return defaultProvider, defaultName, nil
}

// selectProviderByDistribution selects providers to distribute load
func (o *HybridCloudOrchestrator) selectProviderByDistribution(ctx context.Context) (Provider, string, error) {
	// For now, just use the default provider
	// In a real implementation, we would track resource allocation and balance
	defaultProvider, defaultName := o.GetDefaultProvider()
	return defaultProvider, defaultName, nil
}

// refreshPricingCache updates the cache of pricing information from all providers
func (o *HybridCloudOrchestrator) refreshPricingCache(ctx context.Context) error {
	for name, provider := range o.providers {
		// Get instance pricing
		instancePricing, err := provider.GetPricing(ctx, "instance")
		if err != nil {
			return fmt.Errorf("failed to get instance pricing from provider %q: %v", name, err)
		}

		// Get storage pricing
		storagePricing, err := provider.GetPricing(ctx, "storage")
		if err != nil {
			return fmt.Errorf("failed to get storage pricing from provider %q: %v", name, err)
		}

		// Initialize provider cache if needed
		if _, ok := o.pricingCache[name]; !ok {
			o.pricingCache[name] = make(map[string]float64)
		}

		// Update instance pricing
		for k, v := range instancePricing {
			o.pricingCache[name][k] = v
		}

		// Update storage pricing
		for k, v := range storagePricing {
			o.pricingCache[name][k] = v
		}
	}

	o.lastCacheUpdate = time.Now()
	return nil
}

// CreateHybridInstance creates an instance using the most appropriate provider
func (o *HybridCloudOrchestrator) CreateHybridInstance(ctx context.Context, specs InstanceSpecs) (*HybridInstance, error) {
	provider, providerName, err := o.SelectProvider(ctx, "instance", specs)
	if err != nil {
		return nil, fmt.Errorf("failed to select provider: %v", err)
	}

	instance, err := provider.CreateInstance(ctx, specs)
	if err != nil {
		return nil, fmt.Errorf("failed to create instance with provider %q: %v", providerName, err)
	}

	return &HybridInstance{
		Instance:      *instance,
		ProviderName:  providerName,
		ProviderID:    instance.ID,
		CreatedAt:     time.Now(),
		LastUpdatedAt: time.Now(),
	}, nil
}

// CreateHybridStorageVolume creates a storage volume using the most appropriate provider
func (o *HybridCloudOrchestrator) CreateHybridStorageVolume(ctx context.Context, specs StorageVolumeSpecs) (*HybridStorageVolume, error) {
	provider, providerName, err := o.SelectProvider(ctx, "storage", specs)
	if err != nil {
		return nil, fmt.Errorf("failed to select provider: %v", err)
	}

	volume, err := provider.CreateStorageVolume(ctx, specs)
	if err != nil {
		return nil, fmt.Errorf("failed to create storage volume with provider %q: %v", providerName, err)
	}

	return &HybridStorageVolume{
		StorageVolume: *volume,
		ProviderName:  providerName,
		ProviderID:    volume.ID,
		CreatedAt:     time.Now(),
		LastUpdatedAt: time.Now(),
	}, nil
}

// AllocateResources allocates resources across multiple cloud providers
func (o *HybridCloudOrchestrator) AllocateResources(ctx context.Context, instanceSpecs []InstanceSpecs, storageSpecs []StorageVolumeSpecs) (*HybridResourceAllocation, error) {
	allocation := &HybridResourceAllocation{
		AllocationByProvider: make(map[string][]string),
		CostByProvider:       make(map[string]float64),
		TotalCostEstimate:    0,
		CreatedAt:            time.Now(),
	}

	// Allocate instances
	for _, specs := range instanceSpecs {
		provider, providerName, err := o.SelectProvider(ctx, "instance", specs)
		if err != nil {
			return nil, fmt.Errorf("failed to select provider for instance: %v", err)
		}

		instance, err := provider.CreateInstance(ctx, specs)
		if err != nil {
			return nil, fmt.Errorf("failed to create instance with provider %q: %v", providerName, err)
		}

		if _, ok := allocation.AllocationByProvider[providerName]; !ok {
			allocation.AllocationByProvider[providerName] = make([]string, 0)
		}
		allocation.AllocationByProvider[providerName] = append(allocation.AllocationByProvider[providerName], instance.ID)

		// Estimate cost (in a real implementation, this would be more sophisticated)
		if pricing, ok := o.pricingCache[providerName]; ok {
			if price, ok := pricing[specs.InstanceType]; ok {
				allocation.CostByProvider[providerName] += price
				allocation.TotalCostEstimate += price
			}
		}
	}

	// Allocate storage volumes
	for _, specs := range storageSpecs {
		provider, providerName, err := o.SelectProvider(ctx, "storage", specs)
		if err != nil {
			return nil, fmt.Errorf("failed to select provider for storage: %v", err)
		}

		volume, err := provider.CreateStorageVolume(ctx, specs)
		if err != nil {
			return nil, fmt.Errorf("failed to create storage volume with provider %q: %v", providerName, err)
		}

		if _, ok := allocation.AllocationByProvider[providerName]; !ok {
			allocation.AllocationByProvider[providerName] = make([]string, 0)
		}
		allocation.AllocationByProvider[providerName] = append(allocation.AllocationByProvider[providerName], volume.ID)

		// Estimate cost (in a real implementation, this would be more sophisticated)
		if pricing, ok := o.pricingCache[providerName]; ok {
			if price, ok := pricing[specs.Type]; ok {
				volumeCost := price * float64(specs.SizeGB)
				allocation.CostByProvider[providerName] += volumeCost
				allocation.TotalCostEstimate += volumeCost
			}
		}
	}

	return allocation, nil
}

// GetCostOptimizationRecommendations analyzes current resource allocation
// and returns recommendations for cost optimization
func (o *HybridCloudOrchestrator) GetCostOptimizationRecommendations(ctx context.Context) ([]CostOptimizationRecommendation, error) {
	// This would be a sophisticated analysis in a real implementation
	// For now, just return a placeholder
	recommendations := []CostOptimizationRecommendation{
		{
			ResourceID:        "instance-example",
			ResourceType:      "instance",
			CurrentCost:       10.0,
			RecommendedAction: "resize",
			ExpectedSavings:   5.0,
			Reason:            "Instance is oversized for its workload",
			Confidence:        0.85,
		},
		{
			ResourceID:        "volume-example",
			ResourceType:      "storage",
			CurrentCost:       20.0,
			RecommendedAction: "migrate",
			TargetProvider:    "gcp",
			ExpectedSavings:   8.0,
			Reason:            "Storage would be cheaper on GCP for this workload",
			Confidence:        0.75,
		},
	}

	return recommendations, nil
}

// PerformFailover performs automatic failover of resources from one provider to another
func (o *HybridCloudOrchestrator) PerformFailover(ctx context.Context, fromProvider, toProvider string, resourceIDs []string) (*FailoverResult, error) {
	// This would be a complex operation in a real implementation
	// For now, just return a placeholder
	result := &FailoverResult{
		SourceProvider:        fromProvider,
		TargetProvider:        toProvider,
		StartTime:             time.Now(),
		CompletionTime:        time.Now().Add(5 * time.Minute),
		ResourcesMigrated:     len(resourceIDs),
		ResourcesFailed:       0,
		DowntimeSeconds:       60,
		Status:                "completed",
		FailedResourceIDs:     []string{},
		SuccessfulResourceIDs: resourceIDs,
	}

	return result, nil
}

// HybridInstance represents an instance created through the hybrid orchestrator
type HybridInstance struct {
	Instance      Instance
	ProviderName  string
	ProviderID    string
	CreatedAt     time.Time
	LastUpdatedAt time.Time
}

// HybridStorageVolume represents a storage volume created through the hybrid orchestrator
type HybridStorageVolume struct {
	StorageVolume StorageVolume
	ProviderName  string
	ProviderID    string
	CreatedAt     time.Time
	LastUpdatedAt time.Time
}

// CostOptimizationRecommendation represents a recommendation for cost optimization
type CostOptimizationRecommendation struct {
	ResourceID        string
	ResourceType      string
	CurrentCost       float64
	RecommendedAction string
	TargetProvider    string
	ExpectedSavings   float64
	Reason            string
	Confidence        float64
}

// FailoverResult represents the result of a failover operation
type FailoverResult struct {
	SourceProvider        string
	TargetProvider        string
	StartTime             time.Time
	CompletionTime        time.Time
	ResourcesMigrated     int
	ResourcesFailed       int
	DowntimeSeconds       int
	Status                string
	FailedResourceIDs     []string
	SuccessfulResourceIDs []string
}
