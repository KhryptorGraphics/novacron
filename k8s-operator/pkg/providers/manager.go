package providers

import (
	"fmt"
	"sync"

	novacronv1 "github.com/khryptorgraphics/novacron/k8s-operator/pkg/apis/novacron/v1"
)

// DefaultCloudProviderManager implements CloudProviderManager
type DefaultCloudProviderManager struct {
	providers map[string]CloudProvider
	mutex     sync.RWMutex
}

// NewCloudProviderManager creates a new cloud provider manager
func NewCloudProviderManager() CloudProviderManager {
	return &DefaultCloudProviderManager{
		providers: make(map[string]CloudProvider),
	}
}

// RegisterProvider registers a new cloud provider
func (m *DefaultCloudProviderManager) RegisterProvider(provider CloudProvider) error {
	m.mutex.Lock()
	defer m.mutex.Unlock()

	name := provider.GetName()
	if _, exists := m.providers[name]; exists {
		return fmt.Errorf("provider %s already registered", name)
	}

	m.providers[name] = provider
	return nil
}

// GetClient returns a cloud provider client
func (m *DefaultCloudProviderManager) GetClient(providerName string) (CloudProvider, error) {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	provider, exists := m.providers[providerName]
	if !exists {
		return nil, fmt.Errorf("provider %s not found", providerName)
	}

	return provider, nil
}

// ListProviders returns all registered providers
func (m *DefaultCloudProviderManager) ListProviders() []string {
	m.mutex.RLock()
	defer m.mutex.RUnlock()

	var providers []string
	for name := range m.providers {
		providers = append(providers, name)
	}
	return providers
}

// EstimateCost estimates cost for a specific provider
func (m *DefaultCloudProviderManager) EstimateCost(providerName, region string, resources ResourceRequirements) (*novacronv1.ResourceCost, error) {
	provider, err := m.GetClient(providerName)
	if err != nil {
		return nil, err
	}

	return provider.EstimateCost(region, resources)
}

// DefaultCloudProviderFactory implements CloudProviderFactory
type DefaultCloudProviderFactory struct{}

// NewCloudProviderFactory creates a new factory
func NewCloudProviderFactory() CloudProviderFactory {
	return &DefaultCloudProviderFactory{}
}

// Create creates a cloud provider instance
func (f *DefaultCloudProviderFactory) Create(config ProviderConfig) (CloudProvider, error) {
	switch config.Type {
	case "aws":
		return NewAWSProvider(config)
	case "azure":
		return NewAzureProvider(config)
	case "gcp":
		return NewGCPProvider(config)
	case "mock":
		return NewMockProvider(config)
	default:
		return nil, fmt.Errorf("unsupported provider type: %s", config.Type)
	}
}

// SupportedTypes returns supported provider types
func (f *DefaultCloudProviderFactory) SupportedTypes() []string {
	return []string{"aws", "azure", "gcp", "mock"}
}