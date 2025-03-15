package cloud

import (
	"context"
	"fmt"
	"sync"
)

// ProviderManager manages cloud provider instances and operations
type ProviderManager struct {
	// Registry of available provider implementations
	registry *ProviderRegistry

	// Active provider instances
	providers map[string]Provider

	// Lock for providers map
	providersLock sync.RWMutex

	// Default provider name
	defaultProvider string
}

// NewProviderManager creates a new provider manager
func NewProviderManager() *ProviderManager {
	return &ProviderManager{
		registry:  NewProviderRegistry(),
		providers: make(map[string]Provider),
	}
}

// RegisterProviderType registers a provider type with the manager
func (m *ProviderManager) RegisterProviderType(provider Provider) {
	m.registry.RegisterProvider(provider)
}

// InitializeProvider initializes a provider with the given configuration
func (m *ProviderManager) InitializeProvider(providerName string, config ProviderConfig) error {
	// Check if provider type is registered
	providerType, ok := m.registry.GetProvider(providerName)
	if !ok {
		return fmt.Errorf("provider type %q is not registered", providerName)
	}

	// Initialize the provider
	if err := providerType.Initialize(config); err != nil {
		return fmt.Errorf("failed to initialize provider %q: %v", providerName, err)
	}

	// Add to active providers
	m.providersLock.Lock()
	m.providers[providerName] = providerType
	m.providersLock.Unlock()

	// Set as default if it's the first one
	if m.defaultProvider == "" {
		m.defaultProvider = providerName
	}

	return nil
}

// SetDefaultProvider sets the default provider
func (m *ProviderManager) SetDefaultProvider(providerName string) error {
	m.providersLock.RLock()
	defer m.providersLock.RUnlock()

	if _, ok := m.providers[providerName]; !ok {
		return fmt.Errorf("provider %q is not initialized", providerName)
	}

	m.defaultProvider = providerName
	return nil
}

// GetProvider returns the named provider or the default if name is empty
func (m *ProviderManager) GetProvider(name string) (Provider, error) {
	m.providersLock.RLock()
	defer m.providersLock.RUnlock()

	if name == "" {
		name = m.defaultProvider
	}

	provider, ok := m.providers[name]
	if !ok {
		return nil, fmt.Errorf("provider %q is not initialized", name)
	}

	return provider, nil
}

// ListProviders returns a list of available initialized providers
func (m *ProviderManager) ListProviders() []string {
	m.providersLock.RLock()
	defer m.providersLock.RUnlock()

	names := make([]string, 0, len(m.providers))
	for name := range m.providers {
		names = append(names, name)
	}
	return names
}

// ListAvailableProviderTypes returns a list of registered provider types
func (m *ProviderManager) ListAvailableProviderTypes() []string {
	return m.registry.ListProviders()
}

// CloseProvider closes a specific provider
func (m *ProviderManager) CloseProvider(name string) error {
	m.providersLock.Lock()
	defer m.providersLock.Unlock()

	provider, ok := m.providers[name]
	if !ok {
		return fmt.Errorf("provider %q is not initialized", name)
	}

	if err := provider.Close(); err != nil {
		return fmt.Errorf("failed to close provider %q: %v", name, err)
	}

	delete(m.providers, name)

	// Reset default provider if it was the one closed
	if m.defaultProvider == name {
		m.defaultProvider = ""
		// Set a new default if there's at least one provider left
		for name := range m.providers {
			m.defaultProvider = name
			break
		}
	}

	return nil
}

// CloseAllProviders closes all active providers
func (m *ProviderManager) CloseAllProviders() []error {
	m.providersLock.Lock()
	defer m.providersLock.Unlock()

	var errors []error
	for name, provider := range m.providers {
		if err := provider.Close(); err != nil {
			errors = append(errors, fmt.Errorf("failed to close provider %q: %v", name, err))
		}
	}

	m.providers = make(map[string]Provider)
	m.defaultProvider = ""

	return errors
}

// GetInstancesAcrossProviders gets instances from all initialized providers
func (m *ProviderManager) GetInstancesAcrossProviders(ctx context.Context, options ListOptions) (map[string][]Instance, error) {
	m.providersLock.RLock()
	providersCopy := make(map[string]Provider, len(m.providers))
	for name := range m.providers {
		provider := m.providers[name]
		providersCopy[name] = provider
	}
	m.providersLock.RUnlock()

	result := make(map[string][]Instance)
	var errs []error

	// Create wait group to wait for all providers to complete
	var wg sync.WaitGroup
	var resultLock sync.Mutex

	for name, provider := range providersCopy {
		wg.Add(1)
		go func(name string, provider Provider) {
			defer wg.Done()

			instances, err := provider.GetInstances(ctx, options)
			resultLock.Lock()
			defer resultLock.Unlock()

			if err != nil {
				errs = append(errs, fmt.Errorf("error from provider %q: %v", name, err))
				return
			}

			result[name] = instances
		}(name, provider)
	}

	// Wait for all providers to complete
	wg.Wait()

	if len(errs) > 0 {
		return result, fmt.Errorf("errors occurred while getting instances: %v", errs)
	}

	return result, nil
}

// GetStorageVolumesAcrossProviders gets storage volumes from all initialized providers
func (m *ProviderManager) GetStorageVolumesAcrossProviders(ctx context.Context, options ListOptions) (map[string][]StorageVolume, error) {
	m.providersLock.RLock()
	providersCopy := make(map[string]Provider, len(m.providers))
	for name, provider := range m.providers {
		providersCopy[name] = provider
	}
	m.providersLock.RUnlock()

	result := make(map[string][]StorageVolume)
	var errs []error

	// Create wait group to wait for all providers to complete
	var wg sync.WaitGroup
	var resultLock sync.Mutex

	for name, provider := range providersCopy {
		wg.Add(1)
		go func(name string, provider Provider) {
			defer wg.Done()

			volumes, err := provider.GetStorageVolumes(ctx, options)
			resultLock.Lock()
			defer resultLock.Unlock()

			if err != nil {
				errs = append(errs, fmt.Errorf("error from provider %q: %v", name, err))
				return
			}

			result[name] = volumes
		}(name, provider)
	}

	// Wait for all providers to complete
	wg.Wait()

	if len(errs) > 0 {
		return result, fmt.Errorf("errors occurred while getting storage volumes: %v", errs)
	}

	return result, nil
}

// GetImagesAcrossProviders gets images from all initialized providers
func (m *ProviderManager) GetImagesAcrossProviders(ctx context.Context, options ListOptions) (map[string][]Image, error) {
	m.providersLock.RLock()
	providersCopy := make(map[string]Provider, len(m.providers))
	for name, provider := range m.providers {
		providersCopy[name] = provider
	}
	m.providersLock.RUnlock()

	result := make(map[string][]Image)
	var errs []error

	// Create wait group to wait for all providers to complete
	var wg sync.WaitGroup
	var resultLock sync.Mutex

	for name, provider := range providersCopy {
		wg.Add(1)
		go func(name string, provider Provider) {
			defer wg.Done()

			images, err := provider.GetImageList(ctx, options)
			resultLock.Lock()
			defer resultLock.Unlock()

			if err != nil {
				errs = append(errs, fmt.Errorf("error from provider %q: %v", name, err))
				return
			}

			result[name] = images
		}(name, provider)
	}

	// Wait for all providers to complete
	wg.Wait()

	if len(errs) > 0 {
		return result, fmt.Errorf("errors occurred while getting images: %v", errs)
	}

	return result, nil
}

// GetRegionsAcrossProviders gets regions from all initialized providers
func (m *ProviderManager) GetRegionsAcrossProviders(ctx context.Context) (map[string][]Region, error) {
	m.providersLock.RLock()
	providersCopy := make(map[string]Provider, len(m.providers))
	for name, provider := range m.providers {
		providersCopy[name] = provider
	}
	m.providersLock.RUnlock()

	result := make(map[string][]Region)
	var errs []error

	// Create wait group to wait for all providers to complete
	var wg sync.WaitGroup
	var resultLock sync.Mutex

	for name, provider := range providersCopy {
		wg.Add(1)
		go func(name string, provider Provider) {
			defer wg.Done()

			regions, err := provider.GetRegions(ctx)
			resultLock.Lock()
			defer resultLock.Unlock()

			if err != nil {
				errs = append(errs, fmt.Errorf("error from provider %q: %v", name, err))
				return
			}

			result[name] = regions
		}(name, provider)
	}

	// Wait for all providers to complete
	wg.Wait()

	if len(errs) > 0 {
		return result, fmt.Errorf("errors occurred while getting regions: %v", errs)
	}

	return result, nil
}

// GetPricingAcrossProviders gets pricing information from all initialized providers
func (m *ProviderManager) GetPricingAcrossProviders(ctx context.Context, resourceType string) (map[string]map[string]float64, error) {
	m.providersLock.RLock()
	providersCopy := make(map[string]Provider, len(m.providers))
	for name, provider := range m.providers {
		providersCopy[name] = provider
	}
	m.providersLock.RUnlock()

	result := make(map[string]map[string]float64)
	var errs []error

	// Create wait group to wait for all providers to complete
	var wg sync.WaitGroup
	var resultLock sync.Mutex

	for name, provider := range providersCopy {
		wg.Add(1)
		go func(name string, provider Provider) {
			defer wg.Done()

			pricing, err := provider.GetPricing(ctx, resourceType)
			resultLock.Lock()
			defer resultLock.Unlock()

			if err != nil {
				errs = append(errs, fmt.Errorf("error from provider %q: %v", name, err))
				return
			}

			result[name] = pricing
		}(name, provider)
	}

	// Wait for all providers to complete
	wg.Wait()

	if len(errs) > 0 {
		return result, fmt.Errorf("errors occurred while getting pricing: %v", errs)
	}

	return result, nil
}

// FindCheapestProvider finds the cheapest provider for a given instance type
func (m *ProviderManager) FindCheapestProvider(ctx context.Context, instanceType string) (string, float64, error) {
	pricing, err := m.GetPricingAcrossProviders(ctx, "instance")
	if err != nil {
		return "", 0, err
	}

	var cheapestProvider string
	var cheapestPrice float64 = -1

	for providerName, prices := range pricing {
		if price, ok := prices[instanceType]; ok {
			if cheapestPrice == -1 || price < cheapestPrice {
				cheapestPrice = price
				cheapestProvider = providerName
			}
		}
	}

	if cheapestProvider == "" {
		return "", 0, fmt.Errorf("no provider found for instance type %q", instanceType)
	}

	return cheapestProvider, cheapestPrice, nil
}

// FindFastestProvisioningProvider finds the provider with the fastest provisioning time
// based on historical data or provider-reported estimates
func (m *ProviderManager) FindFastestProvisioningProvider(ctx context.Context, instanceType string) (string, error) {
	// This would be implemented with actual logic to determine fastest provider
	// based on historical data or provider specifications

	// For now, we'll return a placeholder implementation
	return m.defaultProvider, nil
}

// FindMostReliableProvider finds the provider with the highest reliability
// based on historical uptime data
func (m *ProviderManager) FindMostReliableProvider(ctx context.Context, region string) (string, error) {
	// This would be implemented with actual logic to determine most reliable provider
	// based on historical data or provider specifications

	// For now, we'll return a placeholder implementation
	return m.defaultProvider, nil
}

// GetProviderFeatures returns a map of provider features
func (m *ProviderManager) GetProviderFeatures() map[string]map[string]bool {
	m.providersLock.RLock()
	defer m.providersLock.RUnlock()

	// List of features to check for each provider
	features := []string{
		"live_migration",
		"gpu_support",
		"spot_instances",
		"auto_scaling",
		"load_balancing",
		"object_storage",
		"managed_databases",
		"kubernetes",
		"serverless",
	}

	result := make(map[string]map[string]bool)

	for name, provider := range m.providers {
		providerFeatures := make(map[string]bool)

		// For a real implementation, we would query the provider for its features
		// For now, we'll use a placeholder that adds the features to all providers
		for _, feature := range features {
			// In a real implementation, this would check if the provider supports this feature
			providerFeatures[feature] = true
		}

		result[name] = providerFeatures
	}

	return result
}
