package cost

import (
	"fmt"
	"strings"
	"time"
)

// GenericCostModel provides a basic cost model for any provider
type GenericCostModel struct {
	providerName    string
	baseHourlyRate  float64              // base cost per CPU core per hour
	memoryRate      float64              // cost per GB memory per hour
	storageRate     float64              // cost per GB storage per hour
	networkRate     float64              // cost per Mbps per hour
	scalingOverhead float64              // percentage overhead for scaling operations
	
	// Pricing tiers
	volumeDiscounts map[int]float64      // instance count -> discount percentage
	
	// Regional multipliers
	regionalMultipliers map[string]float64 // region -> cost multiplier
}

// GenericCostModelConfig configures the generic cost model
type GenericCostModelConfig struct {
	ProviderName        string             `json:"provider_name"`
	BaseHourlyRate      float64            `json:"base_hourly_rate"`      // per CPU core
	MemoryRate          float64            `json:"memory_rate"`           // per GB RAM
	StorageRate         float64            `json:"storage_rate"`          // per GB storage
	NetworkRate         float64            `json:"network_rate"`          // per Mbps
	ScalingOverhead     float64            `json:"scaling_overhead"`      // percentage
	VolumeDiscounts     map[int]float64    `json:"volume_discounts"`      // instance count -> discount
	RegionalMultipliers map[string]float64 `json:"regional_multipliers"`  // region -> multiplier
}

// NewGenericCostModel creates a new generic cost model
func NewGenericCostModel(config GenericCostModelConfig) *GenericCostModel {
	// Set defaults if not provided
	if config.BaseHourlyRate == 0 {
		config.BaseHourlyRate = 0.02 // $0.02 per CPU core per hour
	}
	if config.MemoryRate == 0 {
		config.MemoryRate = 0.005 // $0.005 per GB RAM per hour
	}
	if config.StorageRate == 0 {
		config.StorageRate = 0.0001 // $0.0001 per GB storage per hour (~$0.072/GB/month)
	}
	if config.NetworkRate == 0 {
		config.NetworkRate = 0.0001 // $0.0001 per Mbps per hour
	}
	if config.ScalingOverhead == 0 {
		config.ScalingOverhead = 0.05 // 5% overhead
	}
	
	// Default volume discounts
	if config.VolumeDiscounts == nil {
		config.VolumeDiscounts = map[int]float64{
			10:  0.05, // 5% discount for 10+ instances
			25:  0.10, // 10% discount for 25+ instances
			50:  0.15, // 15% discount for 50+ instances
			100: 0.20, // 20% discount for 100+ instances
		}
	}
	
	// Default regional multipliers
	if config.RegionalMultipliers == nil {
		config.RegionalMultipliers = map[string]float64{
			"default":     1.0,
			"us-central":  0.95,
			"us-east":     1.0,
			"us-west":     1.1,
			"eu-central":  1.15,
			"eu-west":     1.2,
			"asia-east":   1.25,
			"asia-southeast": 1.3,
		}
	}
	
	return &GenericCostModel{
		providerName:        config.ProviderName,
		baseHourlyRate:      config.BaseHourlyRate,
		memoryRate:          config.MemoryRate,
		storageRate:         config.StorageRate,
		networkRate:         config.NetworkRate,
		scalingOverhead:     config.ScalingOverhead,
		volumeDiscounts:     config.VolumeDiscounts,
		regionalMultipliers: config.RegionalMultipliers,
	}
}

// GetHourlyCost implements CostModel interface
func (g *GenericCostModel) GetHourlyCost(resourceType string, config ResourceConfig) (float64, error) {
	if !g.SupportsResourceType(resourceType) {
		return 0, fmt.Errorf("unsupported resource type: %s", resourceType)
	}
	
	var totalCost float64
	
	// Base compute cost (CPU)
	if config.CPUCores > 0 {
		computeCost := float64(config.CPUCores) * g.baseHourlyRate
		totalCost += computeCost
	}
	
	// Memory cost
	if config.MemoryGB > 0 {
		memoryCost := config.MemoryGB * g.memoryRate
		totalCost += memoryCost
	}
	
	// Storage cost
	if config.StorageGB > 0 {
		storageCost := float64(config.StorageGB) * g.storageRate
		totalCost += storageCost
	}
	
	// Network cost
	if config.NetworkMbps > 0 {
		networkCost := float64(config.NetworkMbps) * g.networkRate
		totalCost += networkCost
	}
	
	// Apply regional multiplier
	if multiplier, exists := g.regionalMultipliers[config.Region]; exists {
		totalCost *= multiplier
	} else if multiplier, exists := g.regionalMultipliers["default"]; exists {
		totalCost *= multiplier
	}
	
	// Apply pricing model adjustments
	switch strings.ToLower(config.PricingModel) {
	case "spot":
		totalCost *= 0.3 // 70% discount for spot instances
	case "reserved":
		switch config.ReservedTerm {
		case 12:
			totalCost *= 0.7 // 30% discount for 1-year reserved
		case 36:
			totalCost *= 0.6 // 40% discount for 3-year reserved
		default:
			totalCost *= 0.8 // 20% discount for other reserved terms
		}
	case "on-demand", "":
		// No adjustment for on-demand pricing
	default:
		return 0, fmt.Errorf("unsupported pricing model: %s", config.PricingModel)
	}
	
	return totalCost, nil
}

// GetPredictedCost implements CostModel interface
func (g *GenericCostModel) GetPredictedCost(resourceType string, config ResourceConfig, duration time.Duration) (float64, error) {
	hourlyCost, err := g.GetHourlyCost(resourceType, config)
	if err != nil {
		return 0, err
	}
	
	totalCost := hourlyCost * duration.Hours()
	
	// Apply volume discounts based on duration (longer commitments get better rates)
	hours := duration.Hours()
	if hours > 720 { // > 1 month
		totalCost *= 0.95
	}
	if hours > 8760 { // > 1 year
		totalCost *= 0.90
	}
	if hours > 26280 { // > 3 years
		totalCost *= 0.85
	}
	
	return totalCost, nil
}

// GetScalingCost implements CostModel interface
func (g *GenericCostModel) GetScalingCost(operation ScalingOperation) (float64, error) {
	// Calculate base cost of the from/to configurations
	fromCost, err := g.GetHourlyCost("vm", operation.FromConfig)
	if err != nil {
		return 0, err
	}
	
	toCost, err := g.GetHourlyCost("vm", operation.ToConfig)
	if err != nil {
		return 0, err
	}
	
	// Calculate average cost during operation
	avgCost := (fromCost + toCost) / 2.0
	
	// Base scaling cost is the overhead percentage of average hourly cost
	baseScalingCost := avgCost * g.scalingOverhead
	
	// Add operation-specific costs
	var operationCost float64
	switch operation.Type {
	case "scale_out":
		// Cost of provisioning new resources
		operationCost = baseScalingCost * float64(operation.InstanceCount)
		
	case "scale_in":
		// Cost of graceful shutdown (usually minimal)
		operationCost = baseScalingCost * 0.2 * float64(operation.InstanceCount)
		
	case "scale_up", "scale_down":
		// Vertical scaling involves reconfiguration
		operationCost = baseScalingCost * 2.0 // Higher cost due to complexity
		
		// Add migration cost if data needs to be moved
		if operation.DataTransferGB > 0 {
			migrationCost := operation.DataTransferGB * 0.01 // $0.01 per GB
			operationCost += migrationCost
		}
		
	default:
		return 0, fmt.Errorf("unsupported scaling operation: %s", operation.Type)
	}
	
	// Add time-based cost (opportunity cost of resources during scaling)
	if operation.EstimatedTime > 0 {
		timeCost := avgCost * (operation.EstimatedTime.Hours())
		operationCost += timeCost
	}
	
	return operationCost, nil
}

// GetCostBreakdown implements CostModel interface
func (g *GenericCostModel) GetCostBreakdown(resourceType string, config ResourceConfig) (*CostBreakdown, error) {
	if !g.SupportsResourceType(resourceType) {
		return nil, fmt.Errorf("unsupported resource type: %s", resourceType)
	}
	
	breakdown := &CostBreakdown{
		AdditionalCosts: make(map[string]float64),
		Discounts:       make([]CostDiscount, 0),
	}
	
	// Calculate base costs
	baseCPUCost := float64(config.CPUCores) * g.baseHourlyRate
	baseMemoryCost := config.MemoryGB * g.memoryRate
	baseStorageCost := float64(config.StorageGB) * g.storageRate
	baseNetworkCost := float64(config.NetworkMbps) * g.networkRate
	
	// Apply regional multiplier
	multiplier := 1.0
	if regionMultiplier, exists := g.regionalMultipliers[config.Region]; exists {
		multiplier = regionMultiplier
	} else if defaultMultiplier, exists := g.regionalMultipliers["default"]; exists {
		multiplier = defaultMultiplier
	}
	
	breakdown.ComputeCost = baseCPUCost * multiplier
	breakdown.StorageCost = baseStorageCost * multiplier
	breakdown.NetworkCost = baseNetworkCost * multiplier
	
	// Memory is included in compute cost for simplicity
	breakdown.ComputeCost += baseMemoryCost * multiplier
	
	// Apply pricing model discounts
	originalTotal := breakdown.ComputeCost + breakdown.StorageCost + breakdown.NetworkCost
	
	switch strings.ToLower(config.PricingModel) {
	case "spot":
		discount := CostDiscount{
			Name:        "Spot Instance Discount",
			Type:        "percentage",
			Value:       70.0,
			Description: "70% discount for preemptible instances",
		}
		breakdown.Discounts = append(breakdown.Discounts, discount)
		discountAmount := originalTotal * 0.7
		breakdown.Savings += discountAmount
		
		breakdown.ComputeCost *= 0.3
		breakdown.StorageCost *= 0.3
		breakdown.NetworkCost *= 0.3
		
	case "reserved":
		discountPercent := 20.0
		switch config.ReservedTerm {
		case 12:
			discountPercent = 30.0
		case 36:
			discountPercent = 40.0
		}
		
		discount := CostDiscount{
			Name:        fmt.Sprintf("Reserved Instance (%d months)", config.ReservedTerm),
			Type:        "percentage",
			Value:       discountPercent,
			Description: fmt.Sprintf("%.0f%% discount for reserved capacity", discountPercent),
		}
		breakdown.Discounts = append(breakdown.Discounts, discount)
		
		discountMultiplier := 1.0 - (discountPercent / 100.0)
		discountAmount := originalTotal * (discountPercent / 100.0)
		breakdown.Savings += discountAmount
		
		breakdown.ComputeCost *= discountMultiplier
		breakdown.StorageCost *= discountMultiplier
		breakdown.NetworkCost *= discountMultiplier
	}
	
	// Add regional cost adjustment info
	if multiplier != 1.0 {
		if multiplier > 1.0 {
			breakdown.AdditionalCosts[fmt.Sprintf("Regional Markup (%s)", config.Region)] = originalTotal * (multiplier - 1.0)
		} else {
			discount := (1.0 - multiplier) * 100.0
			regionalDiscount := CostDiscount{
				Name:        fmt.Sprintf("Regional Discount (%s)", config.Region),
				Type:        "percentage",
				Value:       discount,
				Description: fmt.Sprintf("%.0f%% regional pricing adjustment", discount),
			}
			breakdown.Discounts = append(breakdown.Discounts, regionalDiscount)
			breakdown.Savings += originalTotal * (1.0 - multiplier)
		}
	}
	
	// Operating costs (simplified)
	breakdown.OperatingCost = 0.0
	if managementCost, exists := config.Attributes["management_cost"]; exists {
		if mgmtCost, ok := managementCost.(float64); ok {
			breakdown.OperatingCost += mgmtCost
			breakdown.AdditionalCosts["Management & Monitoring"] = mgmtCost
		}
	}
	
	// Calculate total
	breakdown.TotalHourlyCost = breakdown.ComputeCost + breakdown.StorageCost + breakdown.NetworkCost + breakdown.OperatingCost
	
	return breakdown, nil
}

// SupportsResourceType implements CostModel interface
func (g *GenericCostModel) SupportsResourceType(resourceType string) bool {
	supportedTypes := []string{"vm", "instance", "container", "compute", "server"}
	resourceTypeLower := strings.ToLower(resourceType)
	
	for _, supported := range supportedTypes {
		if resourceTypeLower == supported {
			return true
		}
	}
	return false
}

// GetProviderName implements CostModel interface
func (g *GenericCostModel) GetProviderName() string {
	return g.providerName
}

// GetVolumeDiscount calculates volume discount for a given instance count
func (g *GenericCostModel) GetVolumeDiscount(instanceCount int) float64 {
	var maxDiscount float64
	
	// Find the highest applicable discount
	for threshold, discount := range g.volumeDiscounts {
		if instanceCount >= threshold && discount > maxDiscount {
			maxDiscount = discount
		}
	}
	
	return maxDiscount
}

// GetRegionalMultiplier returns the cost multiplier for a specific region
func (g *GenericCostModel) GetRegionalMultiplier(region string) float64 {
	if multiplier, exists := g.regionalMultipliers[region]; exists {
		return multiplier
	}
	
	if defaultMultiplier, exists := g.regionalMultipliers["default"]; exists {
		return defaultMultiplier
	}
	
	return 1.0 // no adjustment if region not found
}

// UpdateRates allows updating the cost rates dynamically
func (g *GenericCostModel) UpdateRates(rates GenericCostModelConfig) {
	if rates.BaseHourlyRate > 0 {
		g.baseHourlyRate = rates.BaseHourlyRate
	}
	if rates.MemoryRate > 0 {
		g.memoryRate = rates.MemoryRate
	}
	if rates.StorageRate > 0 {
		g.storageRate = rates.StorageRate
	}
	if rates.NetworkRate > 0 {
		g.networkRate = rates.NetworkRate
	}
	if rates.ScalingOverhead > 0 {
		g.scalingOverhead = rates.ScalingOverhead
	}
	
	// Update discounts and multipliers if provided
	if rates.VolumeDiscounts != nil {
		for threshold, discount := range rates.VolumeDiscounts {
			g.volumeDiscounts[threshold] = discount
		}
	}
	
	if rates.RegionalMultipliers != nil {
		for region, multiplier := range rates.RegionalMultipliers {
			g.regionalMultipliers[region] = multiplier
		}
	}
}

// GetCostEstimateForWorkload estimates cost for a specific workload pattern
func (g *GenericCostModel) GetCostEstimateForWorkload(workload WorkloadPattern) (*WorkloadCostEstimate, error) {
	estimate := &WorkloadCostEstimate{
		WorkloadName: workload.Name,
		Duration:     workload.Duration,
		Phases:       make([]PhaseCostEstimate, 0),
	}
	
	var totalCost float64
	
	for _, phase := range workload.Phases {
		phaseCost, err := g.GetHourlyCost("vm", phase.ResourceConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to calculate cost for phase %s: %w", phase.Name, err)
		}
		
		// Apply volume discount
		volumeDiscount := g.GetVolumeDiscount(phase.InstanceCount)
		if volumeDiscount > 0 {
			phaseCost *= (1.0 - volumeDiscount)
		}
		
		phaseTotalCost := phaseCost * float64(phase.InstanceCount) * phase.Duration.Hours()
		totalCost += phaseTotalCost
		
		phaseEstimate := PhaseCostEstimate{
			PhaseName:     phase.Name,
			Duration:      phase.Duration,
			InstanceCount: phase.InstanceCount,
			HourlyCost:    phaseCost,
			TotalCost:     phaseTotalCost,
			VolumeDiscount: volumeDiscount,
		}
		
		estimate.Phases = append(estimate.Phases, phaseEstimate)
	}
	
	estimate.TotalCost = totalCost
	estimate.AverageCostPerHour = totalCost / workload.Duration.Hours()
	
	return estimate, nil
}

// WorkloadPattern represents a workload with different resource phases
type WorkloadPattern struct {
	Name     string              `json:"name"`
	Duration time.Duration       `json:"duration"`
	Phases   []WorkloadPhase     `json:"phases"`
}

// WorkloadPhase represents a phase in a workload pattern
type WorkloadPhase struct {
	Name           string         `json:"name"`
	Duration       time.Duration  `json:"duration"`
	ResourceConfig ResourceConfig `json:"resource_config"`
	InstanceCount  int           `json:"instance_count"`
}

// WorkloadCostEstimate represents cost estimate for a workload
type WorkloadCostEstimate struct {
	WorkloadName       string              `json:"workload_name"`
	Duration           time.Duration       `json:"duration"`
	TotalCost          float64             `json:"total_cost"`
	AverageCostPerHour float64             `json:"average_cost_per_hour"`
	Phases             []PhaseCostEstimate `json:"phases"`
}

// PhaseCostEstimate represents cost estimate for a workload phase
type PhaseCostEstimate struct {
	PhaseName      string        `json:"phase_name"`
	Duration       time.Duration `json:"duration"`
	InstanceCount  int           `json:"instance_count"`
	HourlyCost     float64       `json:"hourly_cost"`
	TotalCost      float64       `json:"total_cost"`
	VolumeDiscount float64       `json:"volume_discount"`
}

// GetOptimizedConfiguration suggests the most cost-effective configuration for given requirements
func (g *GenericCostModel) GetOptimizedConfiguration(requirements ResourceRequirements) (*OptimizedConfiguration, error) {
	// Generate different configuration options
	configs := g.generateConfigurationOptions(requirements)
	
	var bestConfig *OptimizedConfiguration
	bestEfficiency := 0.0
	
	for _, config := range configs {
		cost, err := g.GetHourlyCost("vm", config)
		if err != nil {
			continue
		}
		
		// Calculate efficiency (performance per dollar)
		performance := float64(config.CPUCores) + (config.MemoryGB / 4.0) // weighted performance metric
		efficiency := performance / cost
		
		if efficiency > bestEfficiency {
			bestEfficiency = efficiency
			bestConfig = &OptimizedConfiguration{
				ResourceConfig:    config,
				HourlyCost:        cost,
				PerformanceScore:  performance,
				CostEfficiency:    efficiency,
				OptimizationNotes: g.getOptimizationNotes(config, requirements),
			}
		}
	}
	
	if bestConfig == nil {
		return nil, fmt.Errorf("no suitable configuration found for requirements")
	}
	
	return bestConfig, nil
}

// ResourceRequirements represents resource requirements for optimization
type ResourceRequirements struct {
	MinCPUCores    int           `json:"min_cpu_cores"`
	MinMemoryGB    float64       `json:"min_memory_gb"`
	MinStorageGB   int           `json:"min_storage_gb"`
	MaxBudgetHour  float64       `json:"max_budget_hour"`
	Region         string        `json:"region"`
	PreferredModel string        `json:"preferred_model"` // on-demand, spot, reserved
	Duration       time.Duration `json:"duration"`        // expected usage duration
}

// OptimizedConfiguration represents an optimized resource configuration
type OptimizedConfiguration struct {
	ResourceConfig    ResourceConfig `json:"resource_config"`
	HourlyCost        float64        `json:"hourly_cost"`
	PerformanceScore  float64        `json:"performance_score"`
	CostEfficiency    float64        `json:"cost_efficiency"`
	OptimizationNotes []string       `json:"optimization_notes"`
}

// generateConfigurationOptions generates different configuration options
func (g *GenericCostModel) generateConfigurationOptions(req ResourceRequirements) []ResourceConfig {
	configs := make([]ResourceConfig, 0)
	
	// Generate configurations with different CPU/memory ratios
	cpuOptions := []int{req.MinCPUCores}
	if req.MinCPUCores < 8 {
		cpuOptions = append(cpuOptions, req.MinCPUCores*2)
	}
	if req.MinCPUCores < 4 {
		cpuOptions = append(cpuOptions, req.MinCPUCores*4)
	}
	
	memoryOptions := []float64{req.MinMemoryGB}
	if req.MinMemoryGB < 16 {
		memoryOptions = append(memoryOptions, req.MinMemoryGB*2)
	}
	if req.MinMemoryGB < 8 {
		memoryOptions = append(memoryOptions, req.MinMemoryGB*4)
	}
	
	pricingModels := []string{"on-demand"}
	if req.PreferredModel != "" {
		pricingModels = []string{req.PreferredModel}
	} else {
		if req.Duration > 24*time.Hour {
			pricingModels = append(pricingModels, "spot")
		}
		if req.Duration > 30*24*time.Hour {
			pricingModels = append(pricingModels, "reserved")
		}
	}
	
	for _, cpus := range cpuOptions {
		for _, memory := range memoryOptions {
			for _, pricingModel := range pricingModels {
				config := ResourceConfig{
					CPUCores:     cpus,
					MemoryGB:     memory,
					StorageGB:    req.MinStorageGB,
					Region:       req.Region,
					PricingModel: pricingModel,
				}
				
				if pricingModel == "reserved" {
					config.ReservedTerm = 12 // default to 1-year term
				}
				
				configs = append(configs, config)
			}
		}
	}
	
	return configs
}

// getOptimizationNotes generates optimization notes for a configuration
func (g *GenericCostModel) getOptimizationNotes(config ResourceConfig, req ResourceRequirements) []string {
	notes := make([]string, 0)
	
	// CPU optimization notes
	if config.CPUCores > req.MinCPUCores*2 {
		notes = append(notes, "Consider if this much CPU capacity is needed")
	}
	
	// Memory optimization notes
	memoryToCPURatio := config.MemoryGB / float64(config.CPUCores)
	if memoryToCPURatio > 8 {
		notes = append(notes, "High memory-to-CPU ratio - good for memory-intensive workloads")
	} else if memoryToCPURatio < 2 {
		notes = append(notes, "Low memory-to-CPU ratio - may cause memory bottlenecks")
	}
	
	// Pricing model notes
	switch config.PricingModel {
	case "spot":
		notes = append(notes, "Spot pricing offers significant savings but with interruption risk")
	case "reserved":
		notes = append(notes, "Reserved instances provide savings for long-term usage")
	}
	
	// Regional notes
	multiplier := g.GetRegionalMultiplier(config.Region)
	if multiplier > 1.1 {
		notes = append(notes, "Consider alternative regions for potential cost savings")
	} else if multiplier < 0.9 {
		notes = append(notes, "This region offers favorable pricing")
	}
	
	return notes
}