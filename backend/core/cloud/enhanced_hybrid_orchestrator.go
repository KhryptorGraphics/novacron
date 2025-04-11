package cloud

import (
	"context"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/cloud/workload"
)

// EnhancedHybridCloudOrchestrator is a workload-aware extension of the HybridCloudOrchestrator
// that makes intelligent decisions based on workload types and characteristics
type EnhancedHybridCloudOrchestrator struct {
	// Embed the standard orchestrator to inherit its capabilities
	*HybridCloudOrchestrator

	// Workload classifier for workload-aware decision making
	classifier *workload.WorkloadClassifier

	// Cache for workload profiles
	workloadProfiles     map[string]workload.WorkloadProfile
	workloadProfilesLock sync.RWMutex

	// Cost optimization settings
	costOptimizationEnabled  bool
	optimizationInterval     time.Duration
	lastOptimizationRun      time.Time
	optimizationInProgress   bool
	optimizationResultsCache map[string][]CostOptimizationRecommendation
}

// WorkloadAwareSelectionStrategy defines strategies for workload-aware provider selection
type WorkloadAwareSelectionStrategy string

const (
	// WorkloadAwareCost chooses based on workload-specific cost optimization
	WorkloadAwareCost WorkloadAwareSelectionStrategy = "workload-cost"

	// WorkloadAwarePerformance chooses based on workload-specific performance needs
	WorkloadAwarePerformance WorkloadAwareSelectionStrategy = "workload-performance"

	// WorkloadAwareCompliance chooses based on workload-specific compliance requirements
	WorkloadAwareCompliance WorkloadAwareSelectionStrategy = "workload-compliance"

	// WorkloadAwareBalance balances cost, performance, and compliance needs
	WorkloadAwareBalance WorkloadAwareSelectionStrategy = "workload-balance"
)

// NewEnhancedHybridCloudOrchestrator creates a new enhanced hybrid cloud orchestrator
func NewEnhancedHybridCloudOrchestrator() *EnhancedHybridCloudOrchestrator {
	return &EnhancedHybridCloudOrchestrator{
		HybridCloudOrchestrator:  NewHybridCloudOrchestrator(),
		classifier:               workload.NewWorkloadClassifier(),
		workloadProfiles:         make(map[string]workload.WorkloadProfile),
		costOptimizationEnabled:  true,
		optimizationInterval:     6 * time.Hour,
		optimizationResultsCache: make(map[string][]CostOptimizationRecommendation),
	}
}

// RegisterProvider overrides the base method to also update workload mappings
func (o *EnhancedHybridCloudOrchestrator) RegisterProvider(provider Provider) error {
	// Call the parent implementation first
	err := o.HybridCloudOrchestrator.RegisterProvider(provider)
	if err != nil {
		return err
	}

	// Then update the workload classifier with provider-specific mappings
	providerName := provider.Name()
	mappings := o.getProviderWorkloadMappings(providerName)
	o.classifier.UpdateProviderMappings(providerName, mappings)

	// Update pricing data
	o.updateProviderPricingData(context.Background(), providerName, provider)

	return nil
}

// SetWorkloadAwareStrategy sets the workload-aware provider selection strategy
func (o *EnhancedHybridCloudOrchestrator) SetWorkloadAwareStrategy(strategy WorkloadAwareSelectionStrategy) {
	o.lock.Lock()
	defer o.lock.Unlock()

	// Map workload-aware strategies to base strategies as needed
	switch strategy {
	case WorkloadAwareCost:
		o.selectionStrategy = SelectByPrice
	case WorkloadAwarePerformance:
		o.selectionStrategy = SelectByPerformance
	case WorkloadAwareCompliance:
		// No direct mapping - will be handled specially in selection logic
		o.selectionStrategy = ProviderSelectionStrategy(strategy)
	case WorkloadAwareBalance:
		// No direct mapping - will be handled specially in selection logic
		o.selectionStrategy = ProviderSelectionStrategy(strategy)
	default:
		// Default to balanced approach
		o.selectionStrategy = ProviderSelectionStrategy(WorkloadAwareBalance)
	}
}

// EnableCostOptimization enables or disables automatic cost optimization
func (o *EnhancedHybridCloudOrchestrator) EnableCostOptimization(enabled bool, interval time.Duration) {
	o.lock.Lock()
	defer o.lock.Unlock()

	o.costOptimizationEnabled = enabled
	if interval > 0 {
		o.optimizationInterval = interval
	}
}

// AddWorkloadProfile adds or updates a workload profile for future decisions
func (o *EnhancedHybridCloudOrchestrator) AddWorkloadProfile(resourceID string, profile workload.WorkloadProfile) {
	o.workloadProfilesLock.Lock()
	defer o.workloadProfilesLock.Unlock()

	o.workloadProfiles[resourceID] = profile
}

// GetWorkloadProfile retrieves a workload profile by resource ID
func (o *EnhancedHybridCloudOrchestrator) GetWorkloadProfile(resourceID string) (workload.WorkloadProfile, bool) {
	o.workloadProfilesLock.RLock()
	defer o.workloadProfilesLock.RUnlock()

	profile, exists := o.workloadProfiles[resourceID]
	return profile, exists
}

// ClassifyWorkload analyzes metrics and creates a workload profile
func (o *EnhancedHybridCloudOrchestrator) ClassifyWorkload(
	resourceID string,
	resourceType string,
	metrics workload.Metrics) workload.WorkloadProfile {

	// Generate workload characteristics from metrics
	characteristics := o.classifier.ClassifyWorkload(metrics)

	// Create a new workload profile
	profile := workload.WorkloadProfile{
		ID:                resourceID,
		Name:              fmt.Sprintf("%s-%s", resourceType, resourceID),
		Description:       fmt.Sprintf("Automatically classified %s workload", characteristics.Type),
		Tags:              map[string]string{"resource_type": resourceType},
		RequestedCPU:      0, // Would be filled with actual resource requests
		RequestedMemoryGB: 0, // Would be filled with actual resource requests
		RequestedDiskGB:   0, // Would be filled with actual resource requests
		Characteristics:   characteristics,
		Metrics:           metrics,
		ProviderFit:       make(map[string]workload.ProviderFitScore),
	}

	// Store the profile for future decisions
	o.AddWorkloadProfile(resourceID, profile)

	return profile
}

// OptimizeWorkloadPlacement determines the best provider for a workload
func (o *EnhancedHybridCloudOrchestrator) OptimizeWorkloadPlacement(
	ctx context.Context,
	resourceID string) (string, *workload.ProviderFitScore, error) {

	// Get workload profile
	profile, exists := o.GetWorkloadProfile(resourceID)
	if !exists {
		return "", nil, fmt.Errorf("no workload profile found for resource %s", resourceID)
	}

	// Use the classifier to score all providers
	providerScores := o.classifier.OptimizeWorkloadPlacement(profile)

	// Find the best provider
	var bestProvider string
	var bestScore workload.ProviderFitScore
	var bestValue float64 = -1

	o.lock.RLock()
	strategy := o.selectionStrategy
	o.lock.RUnlock()

	for provider, score := range providerScores {
		// Calculate a weighted value based on the current strategy
		var value float64
		switch ProviderSelectionStrategy(strategy) {
		case SelectByPrice, ProviderSelectionStrategy(WorkloadAwareCost):
			value = score.CostScore
		case SelectByPerformance, ProviderSelectionStrategy(WorkloadAwarePerformance):
			value = score.PerformanceScore
		case ProviderSelectionStrategy(WorkloadAwareCompliance):
			value = score.ComplianceScore
		case ProviderSelectionStrategy(WorkloadAwareBalance):
			// Use the overall score for balanced approach
			value = score.OverallScore
		default:
			// Default to overall score
			value = score.OverallScore
		}

		if bestValue == -1 || value > bestValue {
			bestValue = value
			bestProvider = provider
			bestScore = score
		}
	}

	if bestProvider == "" {
		return "", nil, fmt.Errorf("could not determine optimal provider for resource %s", resourceID)
	}

	return bestProvider, &bestScore, nil
}

// SelectProviderForWorkload selects a provider based on workload requirements
func (o *EnhancedHybridCloudOrchestrator) SelectProviderForWorkload(
	ctx context.Context,
	resourceType string,
	metrics workload.Metrics,
	resourceID string) (Provider, string, error) {

	// Classify the workload if not already classified
	_, exists := o.GetWorkloadProfile(resourceID)
	if !exists {
		o.ClassifyWorkload(resourceID, resourceType, metrics)
	}

	// Optimize placement
	bestProviderName, bestScore, err := o.OptimizeWorkloadPlacement(ctx, resourceID)
	if err != nil {
		// Fall back to standard selection if workload-aware selection fails
		return o.HybridCloudOrchestrator.SelectProvider(ctx, resourceType, nil)
	}

	// Get the actual provider
	provider, err := o.GetProvider(bestProviderName)
	if err != nil {
		return nil, "", err
	}

	// Log the decision
	fmt.Printf("Selected %s for workload %s (%s) with score %.2f: %s\n",
		bestProviderName, resourceID, string(bestScore.RecommendedAction),
		bestScore.OverallScore, bestScore.ReasonForScore)

	return provider, bestProviderName, nil
}

// getProviderWorkloadMappings returns workload type to instance type mappings for a provider
func (o *EnhancedHybridCloudOrchestrator) getProviderWorkloadMappings(provider string) map[workload.WorkloadType][]string {
	// This would typically be populated from provider-specific knowledge
	// or a configuration file. For now, we'll use some reasonable defaults.
	mappings := make(map[workload.WorkloadType][]string)

	switch provider {
	case "aws":
		mappings[workload.WebServer] = []string{"t3.medium", "t3.large", "m5.large"}
		mappings[workload.BatchProcessing] = []string{"c5.large", "c5.xlarge", "c5.2xlarge"}
		mappings[workload.DatabaseWorkload] = []string{"r5.large", "r5.xlarge", "db.m5.large"}
		mappings[workload.MLTraining] = []string{"p3.2xlarge", "g4dn.xlarge", "p2.xlarge"}
		mappings[workload.MLInference] = []string{"inf1.xlarge", "g4dn.xlarge", "c5.xlarge"}
		mappings[workload.AnalyticsWorkload] = []string{"r5.2xlarge", "r5.4xlarge", "i3.xlarge"}
		mappings[workload.DevTest] = []string{"t3.micro", "t3.small", "t3a.micro"}
		mappings[workload.GeneralPurpose] = []string{"t3.medium", "m5.large", "m5.xlarge"}

	case "azure":
		mappings[workload.WebServer] = []string{"Standard_B2ms", "Standard_D2s_v3", "Standard_D2_v3"}
		mappings[workload.BatchProcessing] = []string{"Standard_F4s_v2", "Standard_F8s_v2", "Standard_F16s_v2"}
		mappings[workload.DatabaseWorkload] = []string{"Standard_E2s_v3", "Standard_E4s_v3", "Standard_E8s_v3"}
		mappings[workload.MLTraining] = []string{"Standard_NC6s_v3", "Standard_NC12s_v3", "Standard_ND6s"}
		mappings[workload.MLInference] = []string{"Standard_NC4as_T4_v3", "Standard_NC8as_T4_v3", "Standard_F8s_v2"}
		mappings[workload.AnalyticsWorkload] = []string{"Standard_E8s_v3", "Standard_E16s_v3", "Standard_L8s_v2"}
		mappings[workload.DevTest] = []string{"Standard_B1ms", "Standard_B2s", "Standard_B1s"}
		mappings[workload.GeneralPurpose] = []string{"Standard_D2s_v3", "Standard_D4s_v3", "Standard_D8s_v3"}

	case "gcp":
		mappings[workload.WebServer] = []string{"e2-medium", "e2-standard-2", "n1-standard-2"}
		mappings[workload.BatchProcessing] = []string{"c2-standard-4", "c2-standard-8", "c2-standard-16"}
		mappings[workload.DatabaseWorkload] = []string{"n2-highmem-2", "n2-highmem-4", "n2-highmem-8"}
		mappings[workload.MLTraining] = []string{"n1-standard-8-nvidia-tesla-t4", "n1-standard-8-nvidia-tesla-v100", "a2-highgpu-1g"}
		mappings[workload.MLInference] = []string{"n1-standard-4-nvidia-tesla-t4", "n1-standard-2-nvidia-tesla-t4", "e2-standard-4"}
		mappings[workload.AnalyticsWorkload] = []string{"n2-highmem-8", "n2-highmem-16", "n2-standard-16"}
		mappings[workload.DevTest] = []string{"e2-micro", "e2-small", "e2-medium"}
		mappings[workload.GeneralPurpose] = []string{"e2-standard-2", "e2-standard-4", "n2-standard-2"}

	default:
		// Fallback generic mappings
		mappings[workload.GeneralPurpose] = []string{"standard-small", "standard-medium", "standard-large"}
	}

	return mappings
}

// updateProviderPricingData fetches and caches pricing data for a provider
func (o *EnhancedHybridCloudOrchestrator) updateProviderPricingData(ctx context.Context, providerName string, provider Provider) {
	// Get instance pricing
	instancePricing, err := provider.GetPricing(ctx, "instance")
	if err != nil {
		fmt.Printf("Warning: failed to get instance pricing from provider %q: %v\n", providerName, err)
		return
	}

	// Update the workload classifier with pricing data
	o.classifier.UpdatePricingData(providerName, instancePricing)
}

// RefreshPricingData refreshes pricing data for all providers
func (o *EnhancedHybridCloudOrchestrator) RefreshPricingData(ctx context.Context) error {
	o.lock.RLock()
	defer o.lock.RUnlock()

	for name, provider := range o.providers {
		o.updateProviderPricingData(ctx, name, provider)
	}

	return nil
}

// CreateWorkloadAwareInstance creates an instance based on workload requirements
func (o *EnhancedHybridCloudOrchestrator) CreateWorkloadAwareInstance(
	ctx context.Context,
	specs InstanceSpecs,
	metrics workload.Metrics) (*HybridInstance, error) {

	// Generate a resource ID
	resourceID := fmt.Sprintf("instance-%d", time.Now().UnixNano())

	// Select provider based on workload
	provider, providerName, err := o.SelectProviderForWorkload(ctx, "instance", metrics, resourceID)
	if err != nil {
		return nil, fmt.Errorf("failed to select provider: %v", err)
	}

	// Get workload profile
	profile, exists := o.GetWorkloadProfile(resourceID)
	if !exists {
		return nil, fmt.Errorf("workload profile not found for %s", resourceID)
	}

	// Get provider fit information
	providerFit, exists := profile.ProviderFit[providerName]
	if exists && providerFit.OptimalInstanceType != "" {
		// Override instance type with workload-optimized recommendation
		specs.InstanceType = providerFit.OptimalInstanceType
		fmt.Printf("Using workload-optimized instance type %s for %s\n",
			specs.InstanceType, resourceID)
	}

	// Create the instance
	instance, err := provider.CreateInstance(ctx, specs)
	if err != nil {
		return nil, fmt.Errorf("failed to create instance with provider %q: %v", providerName, err)
	}

	// Update the workload profile with the instance ID
	profile.ID = instance.ID
	profile.Name = instance.Name
	o.AddWorkloadProfile(instance.ID, profile)

	// Create and return the hybrid instance
	hybridInstance := &HybridInstance{
		Instance:      *instance,
		ProviderName:  providerName,
		ProviderID:    instance.ID,
		CreatedAt:     time.Now(),
		LastUpdatedAt: time.Now(),
	}

	return hybridInstance, nil
}

// StartCostOptimizationCycle runs a cost optimization cycle for all resources
func (o *EnhancedHybridCloudOrchestrator) StartCostOptimizationCycle(ctx context.Context) error {
	o.lock.Lock()
	if o.optimizationInProgress {
		o.lock.Unlock()
		return fmt.Errorf("optimization cycle already in progress")
	}
	o.optimizationInProgress = true
	o.lock.Unlock()

	defer func() {
		o.lock.Lock()
		o.optimizationInProgress = false
		o.lastOptimizationRun = time.Now()
		o.lock.Unlock()
	}()

	// Get all workload profiles
	o.workloadProfilesLock.RLock()
	resourceIDs := make([]string, 0, len(o.workloadProfiles))
	for id := range o.workloadProfiles {
		resourceIDs = append(resourceIDs, id)
	}
	o.workloadProfilesLock.RUnlock()

	// Process each resource
	results := make(map[string][]CostOptimizationRecommendation)
	for _, id := range resourceIDs {
		recommendations, err := o.optimizeResource(ctx, id)
		if err != nil {
			fmt.Printf("Error optimizing resource %s: %v\n", id, err)
			continue
		}

		if len(recommendations) > 0 {
			results[id] = recommendations
		}
	}

	// Update the cache with new results
	o.lock.Lock()
	o.optimizationResultsCache = results
	o.lock.Unlock()

	return nil
}

// optimizeResource generates optimization recommendations for a single resource
func (o *EnhancedHybridCloudOrchestrator) optimizeResource(
	ctx context.Context,
	resourceID string) ([]CostOptimizationRecommendation, error) {

	// Get workload profile
	profile, exists := o.GetWorkloadProfile(resourceID)
	if !exists {
		return nil, fmt.Errorf("no workload profile found for resource %s", resourceID)
	}

	// Use the classifier to score all providers
	providerScores := o.classifier.OptimizeWorkloadPlacement(profile)

	// Generate recommendations based on scores
	recommendations := make([]CostOptimizationRecommendation, 0)

	// Check current provider (if any)
	o.lock.RLock()
	currentProvider := ""
	// Would typically lookup from a map of resourceID to provider
	o.lock.RUnlock()

	// Find the best provider and store in recommendations
	var bestScore float64
	for provider, score := range providerScores {
		if score.OverallScore > bestScore {
			bestScore = score.OverallScore
		}

		// Add recommendation if it's not the current provider and there's a significant difference
		if provider != currentProvider && score.RecommendedAction != "maintain" {
			// Determine target provider for migration
			var targetProvider string
			if score.RecommendedAction == "migrate" {
				targetProvider = provider
			} else {
				targetProvider = ""
			}

			recommendation := CostOptimizationRecommendation{
				ResourceID:        resourceID,
				ResourceType:      string(profile.Characteristics.Type),
				CurrentCost:       profile.CurrentMonthlyCost,
				RecommendedAction: score.RecommendedAction,
				TargetProvider:    targetProvider,
				ExpectedSavings:   profile.CurrentMonthlyCost - score.EstimatedMonthlyCost,
				Reason:            score.ReasonForScore,
				Confidence:        score.OverallScore,
			}
			recommendations = append(recommendations, recommendation)
		}
	}

	// Sort recommendations by expected savings
	sort.Slice(recommendations, func(i, j int) bool {
		return recommendations[i].ExpectedSavings > recommendations[j].ExpectedSavings
	})

	return recommendations, nil
}

// GetResourceOptimizationRecommendations gets optimization recommendations for a specific resource
func (o *EnhancedHybridCloudOrchestrator) GetResourceOptimizationRecommendations(
	ctx context.Context,
	resourceID string) ([]CostOptimizationRecommendation, error) {

	// Check if we have cached recommendations
	o.lock.RLock()
	cachedRecommendations, exists := o.optimizationResultsCache[resourceID]
	lastRun := o.lastOptimizationRun
	o.lock.RUnlock()

	// If we have recent recommendations, return them
	if exists && time.Since(lastRun) < o.optimizationInterval {
		return cachedRecommendations, nil
	}

	// Otherwise, generate new recommendations
	return o.optimizeResource(ctx, resourceID)
}

// GetCostOptimizationRecommendations overrides the base implementation
// to provide workload-aware recommendations
func (o *EnhancedHybridCloudOrchestrator) GetCostOptimizationRecommendations(
	ctx context.Context) ([]CostOptimizationRecommendation, error) {

	// Use the cached results if available and recent
	o.lock.RLock()
	lastRun := o.lastOptimizationRun
	optimizationEnabled := o.costOptimizationEnabled
	o.lock.RUnlock()

	// If optimization is disabled, use the base implementation
	if !optimizationEnabled {
		return o.HybridCloudOrchestrator.GetCostOptimizationRecommendations(ctx)
	}

	// Check if we need to run a new optimization cycle
	if time.Since(lastRun) > o.optimizationInterval {
		err := o.StartCostOptimizationCycle(ctx)
		if err != nil {
			return nil, fmt.Errorf("failed to start optimization cycle: %v", err)
		}
	}

	// Collect all recommendations from the cache
	o.lock.RLock()
	allRecommendations := make([]CostOptimizationRecommendation, 0)
	for _, recommendations := range o.optimizationResultsCache {
		allRecommendations = append(allRecommendations, recommendations...)
	}
	o.lock.RUnlock()

	// Sort by expected savings
	sort.Slice(allRecommendations, func(i, j int) bool {
		return allRecommendations[i].ExpectedSavings > allRecommendations[j].ExpectedSavings
	})

	return allRecommendations, nil
}
