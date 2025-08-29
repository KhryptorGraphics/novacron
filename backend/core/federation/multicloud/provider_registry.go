package multicloud

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// ProviderRegistry manages all registered cloud providers
type ProviderRegistry struct {
	mu        sync.RWMutex
	providers map[string]CloudProvider
	configs   map[string]*CloudProviderConfig
	health    map[string]*ProviderHealthStatus
	metrics   map[string]*ProviderMetrics
}

// ProviderMetrics tracks provider performance metrics
type ProviderMetrics struct {
	ProviderID      string        `json:"provider_id"`
	RequestCount    int64         `json:"request_count"`
	ErrorCount      int64         `json:"error_count"`
	SuccessRate     float64       `json:"success_rate"`
	AvgResponseTime time.Duration `json:"avg_response_time"`
	LastUpdated     time.Time     `json:"last_updated"`
}

// NewProviderRegistry creates a new provider registry
func NewProviderRegistry() *ProviderRegistry {
	return &ProviderRegistry{
		providers: make(map[string]CloudProvider),
		configs:   make(map[string]*CloudProviderConfig),
		health:    make(map[string]*ProviderHealthStatus),
		metrics:   make(map[string]*ProviderMetrics),
	}
}

// RegisterProvider registers a cloud provider
func (r *ProviderRegistry) RegisterProvider(providerID string, provider CloudProvider, config *CloudProviderConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if providerID == "" {
		return fmt.Errorf("provider ID cannot be empty")
	}

	if provider == nil {
		return fmt.Errorf("provider cannot be nil")
	}

	if config == nil {
		return fmt.Errorf("provider config cannot be nil")
	}

	// Check if already registered
	if _, exists := r.providers[providerID]; exists {
		return fmt.Errorf("provider %s is already registered", providerID)
	}

	// Initialize provider
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := provider.Initialize(ctx, *config); err != nil {
		return fmt.Errorf("failed to initialize provider %s: %v", providerID, err)
	}

	// Validate provider
	if err := provider.Validate(ctx); err != nil {
		return fmt.Errorf("provider %s validation failed: %v", providerID, err)
	}

	// Store provider
	r.providers[providerID] = provider
	r.configs[providerID] = config
	r.metrics[providerID] = &ProviderMetrics{
		ProviderID:  providerID,
		LastUpdated: time.Now(),
	}

	return nil
}

// UnregisterProvider removes a cloud provider
func (r *ProviderRegistry) UnregisterProvider(providerID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.providers[providerID]; !exists {
		return fmt.Errorf("provider %s is not registered", providerID)
	}

	delete(r.providers, providerID)
	delete(r.configs, providerID)
	delete(r.health, providerID)
	delete(r.metrics, providerID)

	return nil
}

// GetProvider returns a registered provider
func (r *ProviderRegistry) GetProvider(providerID string) (CloudProvider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	provider, exists := r.providers[providerID]
	if !exists {
		return nil, fmt.Errorf("provider %s is not registered", providerID)
	}

	return provider, nil
}

// ListProviders returns all registered providers
func (r *ProviderRegistry) ListProviders() map[string]CloudProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]CloudProvider)
	for id, provider := range r.providers {
		result[id] = provider
	}

	return result
}

// GetProvidersByType returns providers of a specific type
func (r *ProviderRegistry) GetProvidersByType(providerType CloudProviderType) map[string]CloudProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]CloudProvider)
	for id, provider := range r.providers {
		if provider.GetProviderType() == providerType {
			result[id] = provider
		}
	}

	return result
}

// GetProviderConfig returns the configuration for a provider
func (r *ProviderRegistry) GetProviderConfig(providerID string) (*CloudProviderConfig, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	config, exists := r.configs[providerID]
	if !exists {
		return nil, fmt.Errorf("provider %s is not registered", providerID)
	}

	return config, nil
}

// UpdateProviderConfig updates the configuration for a provider
func (r *ProviderRegistry) UpdateProviderConfig(providerID string, config *CloudProviderConfig) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	provider, exists := r.providers[providerID]
	if !exists {
		return fmt.Errorf("provider %s is not registered", providerID)
	}

	// Re-initialize provider with new config
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := provider.Initialize(ctx, *config); err != nil {
		return fmt.Errorf("failed to reinitialize provider %s: %v", providerID, err)
	}

	// Validate provider
	if err := provider.Validate(ctx); err != nil {
		return fmt.Errorf("provider %s validation failed with new config: %v", providerID, err)
	}

	r.configs[providerID] = config
	return nil
}

// CheckProviderHealth checks the health of a specific provider
func (r *ProviderRegistry) CheckProviderHealth(providerID string) (*ProviderHealthStatus, error) {
	provider, err := r.GetProvider(providerID)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	health, err := provider.GetProviderHealth(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get health for provider %s: %v", providerID, err)
	}

	r.mu.Lock()
	r.health[providerID] = health
	r.mu.Unlock()

	return health, nil
}

// CheckAllProvidersHealth checks the health of all registered providers
func (r *ProviderRegistry) CheckAllProvidersHealth() map[string]*ProviderHealthStatus {
	providers := r.ListProviders()
	results := make(map[string]*ProviderHealthStatus)

	// Use goroutines for parallel health checks
	var wg sync.WaitGroup
	var mu sync.Mutex

	for providerID := range providers {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			
			health, err := r.CheckProviderHealth(id)
			
			mu.Lock()
			if err != nil {
				results[id] = &ProviderHealthStatus{
					Provider:    providers[id].GetProviderType(),
					Overall:     HealthStatusUnhealthy,
					LastChecked: time.Now(),
					Issues: []HealthIssue{
						{
							ID:          fmt.Sprintf("health-check-%d", time.Now().Unix()),
							Service:     "health-check",
							Severity:    "high",
							Description: err.Error(),
							StartTime:   time.Now(),
						},
					},
				}
			} else {
				results[id] = health
			}
			mu.Unlock()
		}(providerID)
	}

	wg.Wait()
	return results
}

// GetProviderHealth returns the cached health status for a provider
func (r *ProviderRegistry) GetProviderHealth(providerID string) (*ProviderHealthStatus, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	health, exists := r.health[providerID]
	if !exists {
		return nil, fmt.Errorf("no health data available for provider %s", providerID)
	}

	return health, nil
}

// GetAllProviderHealth returns cached health status for all providers
func (r *ProviderRegistry) GetAllProviderHealth() map[string]*ProviderHealthStatus {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]*ProviderHealthStatus)
	for id, health := range r.health {
		result[id] = health
	}

	return result
}

// UpdateProviderMetrics updates performance metrics for a provider
func (r *ProviderRegistry) UpdateProviderMetrics(providerID string, responseTime time.Duration, success bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	metrics, exists := r.metrics[providerID]
	if !exists {
		metrics = &ProviderMetrics{
			ProviderID: providerID,
		}
		r.metrics[providerID] = metrics
	}

	metrics.RequestCount++
	if !success {
		metrics.ErrorCount++
	}

	// Calculate success rate
	if metrics.RequestCount > 0 {
		metrics.SuccessRate = float64(metrics.RequestCount-metrics.ErrorCount) / float64(metrics.RequestCount) * 100
	}

	// Calculate average response time (exponential moving average)
	if metrics.AvgResponseTime == 0 {
		metrics.AvgResponseTime = responseTime
	} else {
		alpha := 0.1 // Smoothing factor
		metrics.AvgResponseTime = time.Duration(float64(metrics.AvgResponseTime)*(1-alpha) + float64(responseTime)*alpha)
	}

	metrics.LastUpdated = time.Now()
}

// GetProviderMetrics returns performance metrics for a provider
func (r *ProviderRegistry) GetProviderMetrics(providerID string) (*ProviderMetrics, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	metrics, exists := r.metrics[providerID]
	if !exists {
		return nil, fmt.Errorf("no metrics available for provider %s", providerID)
	}

	return metrics, nil
}

// GetAllProviderMetrics returns performance metrics for all providers
func (r *ProviderRegistry) GetAllProviderMetrics() map[string]*ProviderMetrics {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]*ProviderMetrics)
	for id, metrics := range r.metrics {
		result[id] = metrics
	}

	return result
}

// GetHealthyProviders returns providers that are currently healthy
func (r *ProviderRegistry) GetHealthyProviders() map[string]CloudProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]CloudProvider)
	for id, provider := range r.providers {
		if health, exists := r.health[id]; exists {
			if health.Overall == HealthStatusHealthy {
				result[id] = provider
			}
		}
	}

	return result
}

// GetProvidersByRegion returns providers that support a specific region
func (r *ProviderRegistry) GetProvidersByRegion(region string) map[string]CloudProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]CloudProvider)
	for id, provider := range r.providers {
		regions := provider.GetRegions()
		for _, r := range regions {
			if r == region {
				result[id] = provider
				break
			}
		}
	}

	return result
}

// GetProvidersByCapability returns providers that support a specific capability
func (r *ProviderRegistry) GetProvidersByCapability(capability CloudCapability) map[string]CloudProvider {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]CloudProvider)
	for id, provider := range r.providers {
		capabilities := provider.GetCapabilities()
		for _, c := range capabilities {
			if c == capability {
				result[id] = provider
				break
			}
		}
	}

	return result
}

// GetBestProvider returns the best provider based on health and performance metrics
func (r *ProviderRegistry) GetBestProvider(criteria *ProviderSelectionCriteria) (string, CloudProvider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var bestProviderID string
	var bestProvider CloudProvider
	var bestScore float64 = -1

	for id, provider := range r.providers {
		// Check basic criteria
		if criteria.ProviderType != "" && provider.GetProviderType() != criteria.ProviderType {
			continue
		}

		if criteria.Region != "" {
			regions := provider.GetRegions()
			found := false
			for _, r := range regions {
				if r == criteria.Region {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		if len(criteria.RequiredCapabilities) > 0 {
			capabilities := provider.GetCapabilities()
			capMap := make(map[CloudCapability]bool)
			for _, cap := range capabilities {
				capMap[cap] = true
			}

			allFound := true
			for _, reqCap := range criteria.RequiredCapabilities {
				if !capMap[reqCap] {
					allFound = false
					break
				}
			}
			if !allFound {
				continue
			}
		}

		// Calculate score based on health and metrics
		score := r.calculateProviderScore(id, criteria)
		if score > bestScore {
			bestScore = score
			bestProviderID = id
			bestProvider = provider
		}
	}

	if bestProvider == nil {
		return "", nil, fmt.Errorf("no provider found matching criteria")
	}

	return bestProviderID, bestProvider, nil
}

// calculateProviderScore calculates a score for provider selection
func (r *ProviderRegistry) calculateProviderScore(providerID string, criteria *ProviderSelectionCriteria) float64 {
	score := 0.0

	// Health score (40% weight)
	if health, exists := r.health[providerID]; exists {
		switch health.Overall {
		case HealthStatusHealthy:
			score += 40.0
		case HealthStatusDegraded:
			score += 20.0
		case HealthStatusUnhealthy:
			score += 0.0
		default:
			score += 10.0 // Unknown status
		}
	}

	// Performance metrics score (30% weight)
	if metrics, exists := r.metrics[providerID]; exists {
		// Success rate component (15% weight)
		score += metrics.SuccessRate * 0.15

		// Response time component (15% weight)
		if metrics.AvgResponseTime > 0 {
			// Convert to score: lower response time = higher score
			responseScore := 15.0
			if metrics.AvgResponseTime > time.Second {
				responseScore = 15.0 / (float64(metrics.AvgResponseTime) / float64(time.Second))
			}
			if responseScore > 15.0 {
				responseScore = 15.0
			}
			score += responseScore
		}
	}

	// Cost preference (20% weight)
	if criteria.CostOptimized {
		// This would need actual cost data from providers
		score += 10.0 // Placeholder
	}

	// Latency preference (10% weight)
	if criteria.LowLatency {
		// This would need actual latency data
		score += 5.0 // Placeholder
	}

	return score
}

// ProviderSelectionCriteria defines criteria for provider selection
type ProviderSelectionCriteria struct {
	ProviderType           CloudProviderType   `json:"provider_type,omitempty"`
	Region                 string              `json:"region,omitempty"`
	RequiredCapabilities   []CloudCapability   `json:"required_capabilities,omitempty"`
	CostOptimized         bool                 `json:"cost_optimized"`
	LowLatency            bool                 `json:"low_latency"`
	HighAvailability      bool                 `json:"high_availability"`
	ExcludeProviders      []string             `json:"exclude_providers,omitempty"`
	PreferredProviders    []string             `json:"preferred_providers,omitempty"`
}

// StartHealthMonitoring starts periodic health monitoring for all providers
func (r *ProviderRegistry) StartHealthMonitoring(interval time.Duration) {
	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for range ticker.C {
			r.CheckAllProvidersHealth()
		}
	}()
}