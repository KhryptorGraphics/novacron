// Package factory provides a factory for creating cloud adapters
package factory

import (
	"fmt"

	"github.com/khryptorgraphics/novacron/adapters/pkg/interfaces"
	"github.com/khryptorgraphics/novacron/adapters/pkg/aws"
	"github.com/khryptorgraphics/novacron/adapters/pkg/azure"
	"github.com/khryptorgraphics/novacron/adapters/pkg/gcp"
)

// AdapterFactory implements the AdapterFactory interface
type AdapterFactory struct {
	supportedProviders map[string]func() interfaces.CloudAdapter
}

// NewAdapterFactory creates a new adapter factory
func NewAdapterFactory() *AdapterFactory {
	return &AdapterFactory{
		supportedProviders: map[string]func() interfaces.CloudAdapter{
			"aws":   func() interfaces.CloudAdapter { return aws.NewAdapter() },
			"azure": func() interfaces.CloudAdapter { return azure.NewAdapter() },
			"gcp":   func() interfaces.CloudAdapter { return gcp.NewAdapter() },
		},
	}
}

// CreateAdapter creates a cloud adapter for the specified provider
func (f *AdapterFactory) CreateAdapter(provider string, config interfaces.CloudConfig) (interfaces.CloudAdapter, error) {
	createFn, exists := f.supportedProviders[provider]
	if !exists {
		return nil, fmt.Errorf("unsupported provider: %s", provider)
	}

	adapter := createFn()
	if err := adapter.Configure(config); err != nil {
		return nil, fmt.Errorf("failed to configure adapter: %w", err)
	}

	return adapter, nil
}

// SupportedProviders returns the list of supported cloud providers
func (f *AdapterFactory) SupportedProviders() []string {
	providers := make([]string, 0, len(f.supportedProviders))
	for provider := range f.supportedProviders {
		providers = append(providers, provider)
	}
	return providers
}

// RegisterAdapter registers a custom adapter
func (f *AdapterFactory) RegisterAdapter(provider string, createFn func() interfaces.CloudAdapter) {
	f.supportedProviders[provider] = createFn
}