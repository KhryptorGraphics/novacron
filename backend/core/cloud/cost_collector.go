package cloud

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// ResourceCostData contains billing information for a cloud resource
type ResourceCostData struct {
	// ResourceID is the unique identifier for the resource
	ResourceID string

	// Provider is the cloud provider name
	Provider string

	// HourlyCost is the estimated cost per hour for this resource
	HourlyCost float64

	// DailyCost is the estimated cost per day for this resource
	DailyCost float64

	// MonthToDateCost is the cost accumulated since the beginning of the month
	MonthToDateCost float64

	// ForecastedCost is the projected cost for the entire month
	ForecastedCost float64

	// BillingCurrency is the currency for all cost values
	BillingCurrency string

	// LastUpdated is when this cost data was last retrieved
	LastUpdated time.Time

	// CostComponents breaks down the cost by component (compute, storage, network, etc.)
	CostComponents map[string]float64

	// ResourceTags contains resource tags that might affect billing
	ResourceTags map[string]string

	// BillingTags contains billing-specific tags
	BillingTags map[string]string

	// BillingAccountID is the ID of the billing account
	BillingAccountID string
}

// CostDataCollector is responsible for collecting billing and cost data from cloud providers
type CostDataCollector struct {
	// Provider manager reference
	providerManager *ProviderManager

	// Refresh interval for cost data
	refreshInterval time.Duration

	// Cost data cache
	costCache      map[string]map[string]*ResourceCostData
	costCacheMutex sync.RWMutex
	costExpiry     map[string]map[string]time.Time

	// Initialization flag
	initialized bool
}

// NewCostDataCollector creates a new cost data collector
func NewCostDataCollector(providerManager *ProviderManager, refreshInterval time.Duration) *CostDataCollector {
	if refreshInterval == 0 {
		refreshInterval = 6 * time.Hour // Default refresh interval
	}

	return &CostDataCollector{
		providerManager: providerManager,
		refreshInterval: refreshInterval,
		costCache:       make(map[string]map[string]*ResourceCostData),
		costExpiry:      make(map[string]map[string]time.Time),
		initialized:     false,
	}
}

// Initialize initializes the cost data collector
func (c *CostDataCollector) Initialize(ctx context.Context) error {
	if c.initialized {
		return fmt.Errorf("cost data collector already initialized")
	}

	// Initialize cache for each provider
	// Use ListProviders instead of GetProviderNames since that's what's available
	providers := c.providerManager.ListProviders()
	for _, providerName := range providers {
		c.costCache[providerName] = make(map[string]*ResourceCostData)
		c.costExpiry[providerName] = make(map[string]time.Time)
	}

	c.initialized = true
	return nil
}

// StartBackgroundCollection starts a background goroutine to periodically refresh cost data
func (c *CostDataCollector) StartBackgroundCollection(ctx context.Context) error {
	if !c.initialized {
		return fmt.Errorf("cost data collector not initialized")
	}

	go func() {
		ticker := time.NewTicker(c.refreshInterval)
		defer ticker.Stop()

		// Initial collection
		c.RefreshAllCostData(ctx)

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				c.RefreshAllCostData(ctx)
			}
		}
	}()

	return nil
}

// RefreshAllCostData refreshes cost data for all resources from all providers
func (c *CostDataCollector) RefreshAllCostData(ctx context.Context) {
	// Use ListProviders instead of GetProviderNames since that's what's available
	providers := c.providerManager.ListProviders()

	// For each provider, collect cost data for all resources
	for _, providerName := range providers {
		provider, err := c.providerManager.GetProvider(providerName)
		if err != nil {
			fmt.Printf("Error getting provider %s: %v\n", providerName, err)
			continue
		}

		// Get all instances from this provider
		var instances []Instance
		switch p := provider.(type) {
		case *AWSProvider:
			instances, err = p.GetInstances(ctx, ListOptions{})
		case *AzureProvider:
			instances, err = p.GetInstances(ctx, ListOptions{})
		case *GCPProvider:
			instances, err = p.GetInstances(ctx, ListOptions{})
		default:
			err = fmt.Errorf("unsupported provider type: %T", provider)
		}

		if err != nil {
			fmt.Printf("Error listing instances for provider %s: %v\n", providerName, err)
			continue
		}

		// For each instance, get cost data
		for _, instance := range instances {
			costData, err := c.fetchResourceCostData(ctx, providerName, instance.ID)
			if err != nil {
				fmt.Printf("Error getting cost data for instance %s from provider %s: %v\n",
					instance.ID, providerName, err)
				continue
			}

			// Update cache
			c.updateCostCache(providerName, instance.ID, costData)
		}
	}
}

// fetchResourceCostData gets cost data for a specific resource from a provider
func (c *CostDataCollector) fetchResourceCostData(
	ctx context.Context,
	providerName string,
	resourceID string,
) (*ResourceCostData, error) {
	provider, err := c.providerManager.GetProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("error getting provider %s: %w", providerName, err)
	}

	// Get pricing information for this instance type
	var pricing map[string]float64
	var instanceType string

	switch p := provider.(type) {
	case *AWSProvider:
		instance, err := p.GetInstance(ctx, resourceID)
		if err != nil {
			return nil, fmt.Errorf("error getting AWS instance %s: %w", resourceID, err)
		}
		instanceType = instance.InstanceType
		pricing, err = p.GetPricing(ctx, "instance")
	case *AzureProvider:
		instance, err := p.GetInstance(ctx, resourceID)
		if err != nil {
			return nil, fmt.Errorf("error getting Azure instance %s: %w", resourceID, err)
		}
		instanceType = instance.InstanceType
		pricing, err = p.GetPricing(ctx, "instance")
	case *GCPProvider:
		instance, err := p.GetInstance(ctx, resourceID)
		if err != nil {
			return nil, fmt.Errorf("error getting GCP instance %s: %w", resourceID, err)
		}
		instanceType = instance.InstanceType
		pricing, err = p.GetPricing(ctx, "instance")
	default:
		return nil, fmt.Errorf("unsupported provider type: %T", provider)
	}

	// Create cost data from pricing information
	costData := &ResourceCostData{
		ResourceID:      resourceID,
		Provider:        providerName,
		LastUpdated:     time.Now(),
		CostComponents:  make(map[string]float64),
		ResourceTags:    make(map[string]string),
		BillingTags:     make(map[string]string),
		BillingCurrency: "USD", // Default currency
	}

	// Set hourly cost based on instance type
	if hourlyCost, ok := pricing[instanceType]; ok {
		costData.HourlyCost = hourlyCost
		costData.DailyCost = hourlyCost * 24

		// Estimate month-to-date cost based on days elapsed
		now := time.Now()
		startOfMonth := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, now.Location())
		daysElapsed := now.Sub(startOfMonth).Hours() / 24
		costData.MonthToDateCost = costData.DailyCost * daysElapsed

		// Estimate forecasted cost based on days in month
		daysInMonth := float64(time.Date(now.Year(), now.Month()+1, 0, 0, 0, 0, 0, now.Location()).Day())
		costData.ForecastedCost = costData.DailyCost * daysInMonth

		// Add compute component
		costData.CostComponents["compute"] = costData.HourlyCost
	}

	// For a more comprehensive implementation, additional components would be added:
	// - Storage costs
	// - Network costs
	// - Additional service costs
	// - Tax considerations
	// - Reserved instance discounts
	// - Spot instance pricing
	// - etc.

	return costData, nil
}

// updateCostCache updates the cost data cache for a resource
func (c *CostDataCollector) updateCostCache(providerName, resourceID string, costData *ResourceCostData) {
	c.costCacheMutex.Lock()
	defer c.costCacheMutex.Unlock()

	// Ensure cache maps exist for this provider
	if _, exists := c.costCache[providerName]; !exists {
		c.costCache[providerName] = make(map[string]*ResourceCostData)
	}
	if _, exists := c.costExpiry[providerName]; !exists {
		c.costExpiry[providerName] = make(map[string]time.Time)
	}

	// Update cache
	c.costCache[providerName][resourceID] = costData
	c.costExpiry[providerName][resourceID] = time.Now().Add(c.refreshInterval)
}

// GetCostData retrieves cost data for a resource
func (c *CostDataCollector) GetCostData(ctx context.Context, providerName, resourceID string) (*ResourceCostData, error) {
	if !c.initialized {
		return nil, fmt.Errorf("cost data collector not initialized")
	}

	// Check cache first
	c.costCacheMutex.RLock()
	cacheMap, providerExists := c.costCache[providerName]
	if providerExists {
		costData, resourceExists := cacheMap[resourceID]
		if resourceExists {
			expiryMap := c.costExpiry[providerName]
			expiryTime, expiryExists := expiryMap[resourceID]
			if expiryExists && time.Now().Before(expiryTime) {
				c.costCacheMutex.RUnlock()
				return costData, nil
			}
		}
	}
	c.costCacheMutex.RUnlock()

	// Not in cache or expired, fetch it
	costData, err := c.fetchResourceCostData(ctx, providerName, resourceID)
	if err != nil {
		return nil, err
	}

	// Update cache
	c.updateCostCache(providerName, resourceID, costData)

	return costData, nil
}

// EnrichStatsWithCostData adds cost data to VM stats
func (c *CostDataCollector) EnrichStatsWithCostData(
	ctx context.Context,
	providerName string,
	vmID string,
	stats *monitoring.VMStats,
) (*monitoring.VMStats, error) {
	if stats == nil {
		return nil, fmt.Errorf("cannot enrich nil stats")
	}

	// Get cost data
	costData, err := c.GetCostData(ctx, providerName, vmID)
	if err != nil {
		// Log but don't fail if cost data is unavailable
		fmt.Printf("Warning: failed to get cost data for %s/%s: %v\n", providerName, vmID, err)
		return stats, nil
	}

	// Convert to EnhancedCloudVMStats for easier manipulation
	enhancedStats := monitoring.ConvertInternalToEnhanced(stats)

	// Add cost data to enhanced stats
	if costData.HourlyCost > 0 {
		enhancedStats.CostData["hourly_cost"] = costData.HourlyCost
	}
	if costData.DailyCost > 0 {
		enhancedStats.CostData["daily_cost"] = costData.DailyCost
	}
	if costData.MonthToDateCost > 0 {
		enhancedStats.CostData["month_to_date_cost"] = costData.MonthToDateCost
	}
	if costData.ForecastedCost > 0 {
		enhancedStats.CostData["forecasted_cost"] = costData.ForecastedCost
	}

	// Add component costs
	for component, cost := range costData.CostComponents {
		enhancedStats.CostData[fmt.Sprintf("component_%s", component)] = cost
	}

	// Add cost metadata
	if enhancedStats.Metadata == nil {
		enhancedStats.Metadata = make(map[string]string)
	}
	enhancedStats.Metadata["cost_currency"] = costData.BillingCurrency
	enhancedStats.Metadata["cost_last_updated"] = costData.LastUpdated.Format(time.RFC3339)

	// Add simple cost tag to regular tags
	if enhancedStats.Tags == nil {
		enhancedStats.Tags = make(map[string]string)
	}
	enhancedStats.Tags["hourly_cost"] = fmt.Sprintf("%.4f", costData.HourlyCost)

	// Convert back to VMStats and return
	return monitoring.ConvertEnhancedToInternal(enhancedStats), nil
}
