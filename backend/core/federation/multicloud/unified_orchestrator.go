package multicloud

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/federation"
)

// UnifiedOrchestrator provides a single interface for multi-cloud operations
type UnifiedOrchestrator struct {
	mu               sync.RWMutex
	registry         *ProviderRegistry
	federationMgr    *federation.FederationManager
	migrationEngine  *CrossCloudMigrationEngine
	costOptimizer    *CostOptimizer
	complianceEngine *ComplianceEngine
	policyEngine     *MultiCloudPolicyEngine
	resourceManager  *ResourceManager
	
	// Configuration
	defaultProvider string
	autoFailover    bool
	loadBalancing   bool
}

// NewUnifiedOrchestrator creates a new unified orchestrator
func NewUnifiedOrchestrator(federationMgr *federation.FederationManager) *UnifiedOrchestrator {
	registry := NewProviderRegistry()
	
	return &UnifiedOrchestrator{
		registry:         registry,
		federationMgr:    federationMgr,
		migrationEngine:  NewCrossCloudMigrationEngine(registry),
		costOptimizer:    NewCostOptimizer(registry),
		complianceEngine: NewComplianceEngine(registry),
		policyEngine:     NewMultiCloudPolicyEngine(),
		resourceManager:  NewResourceManager(registry),
		autoFailover:     true,
		loadBalancing:    true,
	}
}

// RegisterCloudProvider registers a new cloud provider
func (o *UnifiedOrchestrator) RegisterCloudProvider(providerID string, provider CloudProvider, config *CloudProviderConfig) error {
	if err := o.registry.RegisterProvider(providerID, provider, config); err != nil {
		return fmt.Errorf("failed to register provider %s: %v", providerID, err)
	}

	// Add provider to federation if it supports it
	capabilities := provider.GetCapabilities()
	supportsFederation := false
	for _, cap := range capabilities {
		if cap == CapabilityVMLiveMigration {
			supportsFederation = true
			break
		}
	}

	if supportsFederation {
		// Create federation cluster representation
		cluster := &federation.Cluster{
			ID:       providerID,
			Name:     config.Name,
			Endpoint: config.Endpoints["api"],
			Role:     federation.PeerCluster,
			State:    federation.ConnectedState,
			LocationInfo: &federation.ClusterLocation{
				Region: config.DefaultRegion,
			},
		}

		if err := o.federationMgr.AddCluster(cluster); err != nil {
			return fmt.Errorf("failed to add provider to federation: %v", err)
		}
	}

	return nil
}

// UnregisterCloudProvider removes a cloud provider
func (o *UnifiedOrchestrator) UnregisterCloudProvider(providerID string) error {
	if err := o.registry.UnregisterProvider(providerID); err != nil {
		return err
	}

	// Remove from federation
	if err := o.federationMgr.RemoveCluster(providerID); err != nil {
		// Log error but don't fail the operation
		fmt.Printf("Warning: failed to remove provider from federation: %v\n", err)
	}

	return nil
}

// CreateVM creates a VM using the best available provider
func (o *UnifiedOrchestrator) CreateVM(ctx context.Context, request *UnifiedVMRequest) (*VMInstance, error) {
	// Determine best provider based on request criteria
	criteria := &ProviderSelectionCriteria{
		Region:               request.Region,
		RequiredCapabilities: request.RequiredCapabilities,
		CostOptimized:       request.CostOptimized,
		LowLatency:          request.LowLatency,
		HighAvailability:    request.HighAvailability,
	}

	if request.PreferredProvider != "" {
		criteria.PreferredProviders = []string{request.PreferredProvider}
	}

	providerID, provider, err := o.registry.GetBestProvider(criteria)
	if err != nil {
		return nil, fmt.Errorf("no suitable provider found: %v", err)
	}

	// Apply compliance policies
	if err := o.complianceEngine.ValidateVMRequest(ctx, providerID, request); err != nil {
		return nil, fmt.Errorf("compliance validation failed: %v", err)
	}

	// Apply multi-cloud policies
	if err := o.policyEngine.EvaluateVMCreation(ctx, providerID, request); err != nil {
		return nil, fmt.Errorf("policy evaluation failed: %v", err)
	}

	// Convert unified request to provider-specific request
	providerRequest := o.convertToProviderRequest(request, provider)

	// Create VM
	startTime := time.Now()
	vm, err := provider.CreateVM(ctx, providerRequest)
	responseTime := time.Since(startTime)
	
	// Update metrics
	o.registry.UpdateProviderMetrics(providerID, responseTime, err == nil)

	if err != nil {
		// Try failover if enabled and error is not a quota/limit error
		if o.autoFailover && !isQuotaError(err) {
			return o.createVMWithFailover(ctx, request, []string{providerID})
		}
		return nil, fmt.Errorf("failed to create VM on provider %s: %v", providerID, err)
	}

	// Track resource allocation
	if err := o.resourceManager.TrackVMCreation(providerID, vm); err != nil {
		fmt.Printf("Warning: failed to track VM creation: %v\n", err)
	}

	return vm, nil
}

// createVMWithFailover attempts to create VM with failover to other providers
func (o *UnifiedOrchestrator) createVMWithFailover(ctx context.Context, request *UnifiedVMRequest, excludeProviders []string) (*VMInstance, error) {
	criteria := &ProviderSelectionCriteria{
		Region:               request.Region,
		RequiredCapabilities: request.RequiredCapabilities,
		CostOptimized:       request.CostOptimized,
		LowLatency:          request.LowLatency,
		HighAvailability:    request.HighAvailability,
		ExcludeProviders:    excludeProviders,
	}

	providerID, provider, err := o.registry.GetBestProvider(criteria)
	if err != nil {
		return nil, fmt.Errorf("no failover provider available: %v", err)
	}

	// Apply compliance and policy checks
	if err := o.complianceEngine.ValidateVMRequest(ctx, providerID, request); err != nil {
		// Try next provider
		return o.createVMWithFailover(ctx, request, append(excludeProviders, providerID))
	}

	if err := o.policyEngine.EvaluateVMCreation(ctx, providerID, request); err != nil {
		// Try next provider
		return o.createVMWithFailover(ctx, request, append(excludeProviders, providerID))
	}

	providerRequest := o.convertToProviderRequest(request, provider)

	startTime := time.Now()
	vm, err := provider.CreateVM(ctx, providerRequest)
	responseTime := time.Since(startTime)
	
	o.registry.UpdateProviderMetrics(providerID, responseTime, err == nil)

	if err != nil && !isQuotaError(err) {
		// Try next provider
		return o.createVMWithFailover(ctx, request, append(excludeProviders, providerID))
	}

	if err != nil {
		return nil, fmt.Errorf("all providers exhausted, last error: %v", err)
	}

	if err := o.resourceManager.TrackVMCreation(providerID, vm); err != nil {
		fmt.Printf("Warning: failed to track VM creation: %v\n", err)
	}

	return vm, nil
}

// GetVM retrieves VM information from the appropriate provider
func (o *UnifiedOrchestrator) GetVM(ctx context.Context, vmID string) (*VMInstance, error) {
	// Find which provider has this VM
	providerID, err := o.resourceManager.FindVMProvider(vmID)
	if err != nil {
		return nil, fmt.Errorf("VM %s not found in any provider: %v", vmID, err)
	}

	provider, err := o.registry.GetProvider(providerID)
	if err != nil {
		return nil, err
	}

	startTime := time.Now()
	vm, err := provider.GetVM(ctx, vmID)
	responseTime := time.Since(startTime)
	
	o.registry.UpdateProviderMetrics(providerID, responseTime, err == nil)

	return vm, err
}

// ListVMs lists VMs across all providers or specific provider
func (o *UnifiedOrchestrator) ListVMs(ctx context.Context, filters *UnifiedVMFilters) ([]*VMInstance, error) {
	var allVMs []*VMInstance
	var mu sync.Mutex
	var wg sync.WaitGroup
	
	providers := o.registry.ListProviders()
	if filters.ProviderID != "" {
		// Filter to specific provider
		if provider, exists := providers[filters.ProviderID]; exists {
			providers = map[string]CloudProvider{filters.ProviderID: provider}
		} else {
			return nil, fmt.Errorf("provider %s not found", filters.ProviderID)
		}
	}

	// Query all providers in parallel
	for providerID, provider := range providers {
		wg.Add(1)
		go func(pID string, p CloudProvider) {
			defer wg.Done()
			
			startTime := time.Now()
			vms, err := p.ListVMs(ctx, filters.ToProviderFilters())
			responseTime := time.Since(startTime)
			
			o.registry.UpdateProviderMetrics(pID, responseTime, err == nil)
			
			if err == nil {
				mu.Lock()
				allVMs = append(allVMs, vms...)
				mu.Unlock()
			}
		}(providerID, provider)
	}

	wg.Wait()

	// Apply unified filters
	return o.applyUnifiedFilters(allVMs, filters), nil
}

// MigrateVM migrates a VM between providers
func (o *UnifiedOrchestrator) MigrateVM(ctx context.Context, request *CrossCloudMigrationRequest) (*CrossCloudMigrationStatus, error) {
	// Validate source and destination providers
	sourceProvider, err := o.registry.GetProvider(request.SourceProviderID)
	if err != nil {
		return nil, fmt.Errorf("source provider not found: %v", err)
	}

	destProvider, err := o.registry.GetProvider(request.DestinationProviderID)
	if err != nil {
		return nil, fmt.Errorf("destination provider not found: %v", err)
	}

	// Apply compliance policies for migration
	if err := o.complianceEngine.ValidateMigration(ctx, request); err != nil {
		return nil, fmt.Errorf("migration compliance check failed: %v", err)
	}

	// Apply multi-cloud migration policies
	if err := o.policyEngine.EvaluateMigration(ctx, request); err != nil {
		return nil, fmt.Errorf("migration policy evaluation failed: %v", err)
	}

	// Execute migration
	return o.migrationEngine.MigrateVM(ctx, request, sourceProvider, destProvider)
}

// GetCostAnalysis provides cost analysis across all providers
func (o *UnifiedOrchestrator) GetCostAnalysis(ctx context.Context, request *CostAnalysisRequest) (*MultiCloudCostAnalysis, error) {
	return o.costOptimizer.AnalyzeCosts(ctx, request)
}

// OptimizeCosts suggests cost optimizations across providers
func (o *UnifiedOrchestrator) OptimizeCosts(ctx context.Context, request *CostOptimizationRequest) (*CostOptimizationPlan, error) {
	return o.costOptimizer.GenerateOptimizationPlan(ctx, request)
}

// GetComplianceReport generates a compliance report across all providers
func (o *UnifiedOrchestrator) GetComplianceReport(ctx context.Context, frameworks []string) (*MultiCloudComplianceReport, error) {
	return o.complianceEngine.GenerateReport(ctx, frameworks)
}

// SetMultiCloudPolicy sets a multi-cloud policy
func (o *UnifiedOrchestrator) SetMultiCloudPolicy(ctx context.Context, policy *MultiCloudPolicy) error {
	return o.policyEngine.SetPolicy(ctx, policy)
}

// GetResourceUtilization gets resource utilization across all providers
func (o *UnifiedOrchestrator) GetResourceUtilization(ctx context.Context) (*MultiCloudResourceUtilization, error) {
	return o.resourceManager.GetUtilization(ctx)
}

// GetProviderHealth gets health status of all providers
func (o *UnifiedOrchestrator) GetProviderHealth(ctx context.Context) map[string]*ProviderHealthStatus {
	return o.registry.GetAllProviderHealth()
}

// GetProviderMetrics gets performance metrics of all providers  
func (o *UnifiedOrchestrator) GetProviderMetrics(ctx context.Context) map[string]*ProviderMetrics {
	return o.registry.GetAllProviderMetrics()
}

// Helper methods

func (o *UnifiedOrchestrator) convertToProviderRequest(request *UnifiedVMRequest, provider CloudProvider) *VMCreateRequest {
	return &VMCreateRequest{
		Name:             request.Name,
		InstanceType:     request.InstanceType,
		ImageID:          request.ImageID,
		Region:           request.Region,
		AvailabilityZone: request.AvailabilityZone,
		KeyPair:          request.KeyPair,
		SecurityGroups:   request.SecurityGroups,
		UserData:         request.UserData,
		Tags:             request.Tags,
		CPU:              request.CPU,
		Memory:           request.Memory,
		Storage:          request.Storage,
		NetworkBandwidth: request.NetworkBandwidth,
		SpotInstance:     request.SpotInstance,
		MaxSpotPrice:     request.MaxSpotPrice,
		CustomOptions:    request.CustomOptions,
	}
}

func (o *UnifiedOrchestrator) applyUnifiedFilters(vms []*VMInstance, filters *UnifiedVMFilters) []*VMInstance {
	if filters == nil {
		return vms
	}

	var filtered []*VMInstance
	for _, vm := range vms {
		if filters.State != "" && vm.State != VMState(filters.State) {
			continue
		}
		if filters.Region != "" && vm.Region != filters.Region {
			continue
		}
		if filters.ProviderType != "" && vm.Provider != CloudProviderType(filters.ProviderType) {
			continue
		}
		if filters.NamePattern != "" {
			// Simple pattern matching - in production, use regex
			if !contains(vm.Name, filters.NamePattern) {
				continue
			}
		}
		filtered = append(filtered, vm)
	}

	return filtered
}

func isQuotaError(err error) bool {
	// Check if error indicates quota/limit exceeded
	errStr := err.Error()
	return contains(errStr, "quota") || contains(errStr, "limit") || contains(errStr, "exceeded")
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && 
		   (s == substr || len(s) > len(substr) && 
		   (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || 
		   (len(s) > len(substr) && contains(s[1:], substr))))
}

// Request and response types

// UnifiedVMRequest represents a unified VM creation request
type UnifiedVMRequest struct {
	Name                  string                 `json:"name"`
	InstanceType          string                 `json:"instance_type"`
	ImageID               string                 `json:"image_id"`
	Region                string                 `json:"region"`
	AvailabilityZone      string                 `json:"availability_zone,omitempty"`
	KeyPair               string                 `json:"key_pair,omitempty"`
	SecurityGroups        []string               `json:"security_groups"`
	UserData              string                 `json:"user_data,omitempty"`
	Tags                  map[string]string      `json:"tags"`
	
	// Resource specifications
	CPU                   int                    `json:"cpu,omitempty"`
	Memory                int64                  `json:"memory,omitempty"`
	Storage               int64                  `json:"storage,omitempty"`
	NetworkBandwidth      int64                  `json:"network_bandwidth,omitempty"`
	
	// Provider selection criteria
	PreferredProvider     string                 `json:"preferred_provider,omitempty"`
	RequiredCapabilities  []CloudCapability      `json:"required_capabilities,omitempty"`
	CostOptimized         bool                   `json:"cost_optimized"`
	LowLatency            bool                   `json:"low_latency"`
	HighAvailability      bool                   `json:"high_availability"`
	
	// Advanced options
	SpotInstance          bool                   `json:"spot_instance"`
	MaxSpotPrice          float64                `json:"max_spot_price,omitempty"`
	ComplianceRequirements []string              `json:"compliance_requirements,omitempty"`
	DataResidencyRegions  []string               `json:"data_residency_regions,omitempty"`
	CustomOptions         map[string]interface{} `json:"custom_options,omitempty"`
}

// UnifiedVMFilters represents filters for listing VMs
type UnifiedVMFilters struct {
	ProviderID   string `json:"provider_id,omitempty"`
	ProviderType string `json:"provider_type,omitempty"`
	Region       string `json:"region,omitempty"`
	State        string `json:"state,omitempty"`
	NamePattern  string `json:"name_pattern,omitempty"`
	TagFilters   map[string]string `json:"tag_filters,omitempty"`
}

// ToProviderFilters converts unified filters to provider-specific filters
func (f *UnifiedVMFilters) ToProviderFilters() map[string]string {
	filters := make(map[string]string)
	if f.Region != "" {
		filters["region"] = f.Region
	}
	if f.State != "" {
		filters["state"] = f.State
	}
	if f.NamePattern != "" {
		filters["name"] = f.NamePattern
	}
	return filters
}

// MultiCloudResourceUtilization represents resource utilization across providers
type MultiCloudResourceUtilization struct {
	TotalVMs         int                                  `json:"total_vms"`
	TotalCPU         int                                  `json:"total_cpu"`
	TotalMemory      int64                                `json:"total_memory"`
	TotalStorage     int64                                `json:"total_storage"`
	TotalCost        float64                              `json:"total_cost"`
	ByProvider       map[string]*ResourceUsage            `json:"by_provider"`
	ByRegion         map[string]*ResourceUsage            `json:"by_region"`
	Trends           *ResourceUtilizationTrends           `json:"trends"`
	Recommendations  []ResourceOptimizationRecommendation `json:"recommendations"`
	LastUpdated      time.Time                            `json:"last_updated"`
}

// ResourceUtilizationTrends represents utilization trends
type ResourceUtilizationTrends struct {
	CPUTrend     string  `json:"cpu_trend"`     // increasing, decreasing, stable
	MemoryTrend  string  `json:"memory_trend"`
	StorageTrend string  `json:"storage_trend"`
	CostTrend    string  `json:"cost_trend"`
	TrendPeriod  string  `json:"trend_period"`  // 24h, 7d, 30d
}

// ResourceOptimizationRecommendation represents optimization recommendations
type ResourceOptimizationRecommendation struct {
	Type        string  `json:"type"`        // rightsizing, migration, termination
	Resource    string  `json:"resource"`    // VM ID or resource identifier
	Provider    string  `json:"provider"`
	Description string  `json:"description"`
	Potential   string  `json:"potential"`   // cost savings, performance improvement
	Confidence  float64 `json:"confidence"`  // 0-1
	Impact      string  `json:"impact"`      // low, medium, high
}