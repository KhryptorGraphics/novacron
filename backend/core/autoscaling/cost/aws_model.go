package cost

import (
	"fmt"
	"strings"
	"time"
)

// AWSCostModel implements the CostModel interface for AWS
type AWSCostModel struct {
	region        string
	pricingData   map[string]AWSInstancePricing
	ebsPricing    map[string]float64  // storage type -> price per GB/month
	networkPricing AWSNetworkPricing
}

// AWSInstancePricing represents AWS instance pricing information
type AWSInstancePricing struct {
	InstanceType    string                     `json:"instance_type"`
	OnDemandPrice   float64                    `json:"on_demand_price"`   // hourly price
	SpotPrice       float64                    `json:"spot_price"`        // current spot price
	ReservedPrices  map[int]float64           `json:"reserved_prices"`   // term -> hourly price
	
	// Instance specifications
	vCPUs          int                        `json:"vcpus"`
	MemoryGB       float64                    `json:"memory_gb"`
	NetworkPerf    string                     `json:"network_perf"`
	EBSOptimized   bool                       `json:"ebs_optimized"`
	
	// Additional costs
	EBSCost        float64                    `json:"ebs_cost"`         // per GB/hour for attached storage
}

// AWSNetworkPricing represents AWS network pricing
type AWSNetworkPricing struct {
	DataTransferOut map[string]float64 `json:"data_transfer_out"` // destination -> price per GB
	DataTransferIn  float64            `json:"data_transfer_in"`  // usually free
	NATGateway      float64            `json:"nat_gateway"`       // hourly cost
	LoadBalancer    float64            `json:"load_balancer"`     // hourly cost
}

// NewAWSCostModel creates a new AWS cost model
func NewAWSCostModel(region string) *AWSCostModel {
	model := &AWSCostModel{
		region:      region,
		pricingData: make(map[string]AWSInstancePricing),
		ebsPricing:  make(map[string]float64),
	}
	
	// Initialize with sample pricing data (in production, this would be loaded from AWS API)
	model.loadDefaultPricing()
	
	return model
}

// loadDefaultPricing loads default AWS pricing (sample data)
func (a *AWSCostModel) loadDefaultPricing() {
	// Sample EC2 instance pricing for us-east-1
	a.pricingData["t3.micro"] = AWSInstancePricing{
		InstanceType:   "t3.micro",
		OnDemandPrice:  0.0104, // $0.0104/hour
		SpotPrice:      0.0035, // typical spot price
		ReservedPrices: map[int]float64{12: 0.0075, 36: 0.0063}, // 1yr, 3yr
		vCPUs:          2,
		MemoryGB:       1.0,
		NetworkPerf:    "Low to Moderate",
		EBSOptimized:   false,
		EBSCost:        0.00001388, // $0.10/GB/month = $0.00001388/GB/hour
	}
	
	a.pricingData["t3.small"] = AWSInstancePricing{
		InstanceType:   "t3.small",
		OnDemandPrice:  0.0208,
		SpotPrice:      0.0070,
		ReservedPrices: map[int]float64{12: 0.0150, 36: 0.0125},
		vCPUs:          2,
		MemoryGB:       2.0,
		NetworkPerf:    "Low to Moderate",
		EBSOptimized:   false,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["t3.medium"] = AWSInstancePricing{
		InstanceType:   "t3.medium",
		OnDemandPrice:  0.0416,
		SpotPrice:      0.0140,
		ReservedPrices: map[int]float64{12: 0.0300, 36: 0.0250},
		vCPUs:          2,
		MemoryGB:       4.0,
		NetworkPerf:    "Low to Moderate",
		EBSOptimized:   false,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["t3.large"] = AWSInstancePricing{
		InstanceType:   "t3.large",
		OnDemandPrice:  0.0832,
		SpotPrice:      0.0280,
		ReservedPrices: map[int]float64{12: 0.0600, 36: 0.0500},
		vCPUs:          2,
		MemoryGB:       8.0,
		NetworkPerf:    "Low to Moderate",
		EBSOptimized:   false,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["m5.large"] = AWSInstancePricing{
		InstanceType:   "m5.large",
		OnDemandPrice:  0.096,
		SpotPrice:      0.032,
		ReservedPrices: map[int]float64{12: 0.069, 36: 0.058},
		vCPUs:          2,
		MemoryGB:       8.0,
		NetworkPerf:    "Up to 10 Gigabit",
		EBSOptimized:   true,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["m5.xlarge"] = AWSInstancePricing{
		InstanceType:   "m5.xlarge",
		OnDemandPrice:  0.192,
		SpotPrice:      0.064,
		ReservedPrices: map[int]float64{12: 0.138, 36: 0.116},
		vCPUs:          4,
		MemoryGB:       16.0,
		NetworkPerf:    "Up to 10 Gigabit",
		EBSOptimized:   true,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["m5.2xlarge"] = AWSInstancePricing{
		InstanceType:   "m5.2xlarge",
		OnDemandPrice:  0.384,
		SpotPrice:      0.128,
		ReservedPrices: map[int]float64{12: 0.277, 36: 0.232},
		vCPUs:          8,
		MemoryGB:       32.0,
		NetworkPerf:    "Up to 10 Gigabit",
		EBSOptimized:   true,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["c5.large"] = AWSInstancePricing{
		InstanceType:   "c5.large",
		OnDemandPrice:  0.085,
		SpotPrice:      0.028,
		ReservedPrices: map[int]float64{12: 0.061, 36: 0.051},
		vCPUs:          2,
		MemoryGB:       4.0,
		NetworkPerf:    "Up to 10 Gigabit",
		EBSOptimized:   true,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["c5.xlarge"] = AWSInstancePricing{
		InstanceType:   "c5.xlarge",
		OnDemandPrice:  0.17,
		SpotPrice:      0.057,
		ReservedPrices: map[int]float64{12: 0.123, 36: 0.103},
		vCPUs:          4,
		MemoryGB:       8.0,
		NetworkPerf:    "Up to 10 Gigabit",
		EBSOptimized:   true,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["r5.large"] = AWSInstancePricing{
		InstanceType:   "r5.large",
		OnDemandPrice:  0.126,
		SpotPrice:      0.042,
		ReservedPrices: map[int]float64{12: 0.091, 36: 0.076},
		vCPUs:          2,
		MemoryGB:       16.0,
		NetworkPerf:    "Up to 10 Gigabit",
		EBSOptimized:   true,
		EBSCost:        0.00001388,
	}
	
	a.pricingData["r5.xlarge"] = AWSInstancePricing{
		InstanceType:   "r5.xlarge",
		OnDemandPrice:  0.252,
		SpotPrice:      0.084,
		ReservedPrices: map[int]float64{12: 0.182, 36: 0.152},
		vCPUs:          4,
		MemoryGB:       32.0,
		NetworkPerf:    "Up to 10 Gigabit",
		EBSOptimized:   true,
		EBSCost:        0.00001388,
	}
	
	// EBS pricing (per GB/month)
	a.ebsPricing["gp2"] = 0.10    // General Purpose SSD
	a.ebsPricing["gp3"] = 0.08    // General Purpose SSD (newer)
	a.ebsPricing["io1"] = 0.125   // Provisioned IOPS SSD
	a.ebsPricing["io2"] = 0.125   // Provisioned IOPS SSD (newer)
	a.ebsPricing["st1"] = 0.045   // Throughput Optimized HDD
	a.ebsPricing["sc1"] = 0.015   // Cold HDD
	
	// Network pricing
	a.networkPricing = AWSNetworkPricing{
		DataTransferOut: map[string]float64{
			"internet":     0.09,  // first 1TB/month
			"cloudfront":   0.085, // to CloudFront
			"same_region":  0.01,  // within same region
			"cross_region": 0.02,  // to other regions
		},
		DataTransferIn:  0.0,   // free
		NATGateway:      0.045, // per hour
		LoadBalancer:    0.0225, // per hour for ALB
	}
}

// GetHourlyCost implements CostModel interface
func (a *AWSCostModel) GetHourlyCost(resourceType string, config ResourceConfig) (float64, error) {
	if !a.SupportsResourceType(resourceType) {
		return 0, fmt.Errorf("unsupported resource type: %s", resourceType)
	}
	
	pricing, exists := a.pricingData[config.InstanceType]
	if !exists {
		return 0, fmt.Errorf("pricing not available for instance type: %s", config.InstanceType)
	}
	
	var baseCost float64
	
	// Select pricing model
	switch strings.ToLower(config.PricingModel) {
	case "spot":
		baseCost = pricing.SpotPrice
	case "reserved":
		if reservedPrice, exists := pricing.ReservedPrices[config.ReservedTerm]; exists {
			baseCost = reservedPrice
		} else {
			baseCost = pricing.OnDemandPrice // fallback to on-demand
		}
	case "on-demand", "":
		baseCost = pricing.OnDemandPrice
	default:
		return 0, fmt.Errorf("unsupported pricing model: %s", config.PricingModel)
	}
	
	// Calculate storage costs
	storageCost := 0.0
	if config.StorageGB > 0 {
		storageType := "gp3" // default
		if storageTypeAttr, exists := config.Attributes["storage_type"]; exists {
			if storageTypeStr, ok := storageTypeAttr.(string); ok {
				storageType = storageTypeStr
			}
		}
		
		if storagePrice, exists := a.ebsPricing[storageType]; exists {
			// Convert monthly price to hourly
			storageCost = (storagePrice * float64(config.StorageGB)) / (30 * 24)
		}
	}
	
	// Calculate network costs (simplified)
	networkCost := 0.0
	if config.NetworkMbps > 0 {
		// Estimate based on network performance requirement
		// This is a simplified calculation
		networkCost = float64(config.NetworkMbps) * 0.0001 // $0.0001 per Mbps per hour
	}
	
	totalCost := baseCost + storageCost + networkCost
	
	return totalCost, nil
}

// GetPredictedCost implements CostModel interface
func (a *AWSCostModel) GetPredictedCost(resourceType string, config ResourceConfig, duration time.Duration) (float64, error) {
	hourlyCost, err := a.GetHourlyCost(resourceType, config)
	if err != nil {
		return 0, err
	}
	
	hours := duration.Hours()
	totalCost := hourlyCost * hours
	
	// Apply volume discounts for long-term usage
	if hours > 744 { // more than 1 month
		totalCost *= 0.95 // 5% volume discount
	}
	
	if hours > 8760 { // more than 1 year
		totalCost *= 0.90 // additional 10% discount (15% total)
	}
	
	return totalCost, nil
}

// GetScalingCost implements CostModel interface
func (a *AWSCostModel) GetScalingCost(operation ScalingOperation) (float64, error) {
	var scalingCost float64
	
	switch operation.Type {
	case "scale_out":
		// Cost to launch new instances
		// AWS doesn't charge for launching instances, but there might be setup time costs
		scalingCost = 0.0
		
		// Add data transfer costs if migrating data
		if operation.DataTransferGB > 0 {
			transferCost := operation.DataTransferGB * a.networkPricing.DataTransferOut["cross_region"]
			scalingCost += transferCost
		}
		
	case "scale_in":
		// No cost to terminate instances in AWS
		scalingCost = 0.0
		
	case "scale_up", "scale_down":
		// Vertical scaling requires stop/start, potential data migration
		scalingCost = 0.0 // AWS doesn't charge for stopping/starting
		
		// Add data transfer costs for migration
		if operation.DataTransferGB > 0 {
			transferCost := operation.DataTransferGB * a.networkPricing.DataTransferOut["same_region"]
			scalingCost += transferCost
		}
		
		// Add opportunity cost for downtime (simplified)
		if operation.EstimatedTime > 0 {
			// Assume 1% of hourly cost per minute of downtime as opportunity cost
			currentHourlyCost, _ := a.GetHourlyCost("vm", operation.FromConfig)
			opportunityCost := currentHourlyCost * (operation.EstimatedTime.Minutes() / 60.0) * 0.01
			scalingCost += opportunityCost
		}
		
	default:
		return 0, fmt.Errorf("unsupported scaling operation: %s", operation.Type)
	}
	
	return scalingCost, nil
}

// GetCostBreakdown implements CostModel interface
func (a *AWSCostModel) GetCostBreakdown(resourceType string, config ResourceConfig) (*CostBreakdown, error) {
	if !a.SupportsResourceType(resourceType) {
		return nil, fmt.Errorf("unsupported resource type: %s", resourceType)
	}
	
	pricing, exists := a.pricingData[config.InstanceType]
	if !exists {
		return nil, fmt.Errorf("pricing not available for instance type: %s", config.InstanceType)
	}
	
	breakdown := &CostBreakdown{
		AdditionalCosts: make(map[string]float64),
		Discounts:       make([]CostDiscount, 0),
	}
	
	// Base compute cost
	switch strings.ToLower(config.PricingModel) {
	case "spot":
		breakdown.ComputeCost = pricing.SpotPrice
		// Add spot savings as discount
		savings := pricing.OnDemandPrice - pricing.SpotPrice
		if savings > 0 {
			discount := CostDiscount{
				Name:        "Spot Instance Discount",
				Type:        "fixed",
				Value:       savings,
				Description: fmt.Sprintf("%.1f%% savings over On-Demand", (savings/pricing.OnDemandPrice)*100),
			}
			breakdown.Discounts = append(breakdown.Discounts, discount)
			breakdown.Savings += savings
		}
		
	case "reserved":
		if reservedPrice, exists := pricing.ReservedPrices[config.ReservedTerm]; exists {
			breakdown.ComputeCost = reservedPrice
			// Add reserved instance savings
			savings := pricing.OnDemandPrice - reservedPrice
			if savings > 0 {
				discount := CostDiscount{
					Name:        fmt.Sprintf("Reserved Instance (%d months)", config.ReservedTerm),
					Type:        "fixed",
					Value:       savings,
					Description: fmt.Sprintf("%.1f%% savings over On-Demand", (savings/pricing.OnDemandPrice)*100),
				}
				breakdown.Discounts = append(breakdown.Discounts, discount)
				breakdown.Savings += savings
			}
		} else {
			breakdown.ComputeCost = pricing.OnDemandPrice
		}
		
	default:
		breakdown.ComputeCost = pricing.OnDemandPrice
	}
	
	// Storage cost
	if config.StorageGB > 0 {
		storageType := "gp3" // default
		if storageTypeAttr, exists := config.Attributes["storage_type"]; exists {
			if storageTypeStr, ok := storageTypeAttr.(string); ok {
				storageType = storageTypeStr
			}
		}
		
		if storagePrice, exists := a.ebsPricing[storageType]; exists {
			// Convert monthly price to hourly
			breakdown.StorageCost = (storagePrice * float64(config.StorageGB)) / (30 * 24)
			breakdown.AdditionalCosts[fmt.Sprintf("EBS %s Storage", strings.ToUpper(storageType))] = breakdown.StorageCost
		}
	}
	
	// Network cost
	if config.NetworkMbps > 0 {
		breakdown.NetworkCost = float64(config.NetworkMbps) * 0.0001
		breakdown.AdditionalCosts["Enhanced Networking"] = breakdown.NetworkCost
	}
	
	// Operating costs (licensing, support, etc.)
	breakdown.OperatingCost = 0.0
	if licensing, exists := config.Attributes["licensing_cost"]; exists {
		if licensingCost, ok := licensing.(float64); ok {
			breakdown.OperatingCost += licensingCost
			breakdown.AdditionalCosts["Software Licensing"] = licensingCost
		}
	}
	
	// Calculate total
	breakdown.TotalHourlyCost = breakdown.ComputeCost + breakdown.StorageCost + breakdown.NetworkCost + breakdown.OperatingCost
	
	return breakdown, nil
}

// SupportsResourceType implements CostModel interface
func (a *AWSCostModel) SupportsResourceType(resourceType string) bool {
	supportedTypes := []string{"vm", "instance", "ec2", "container"}
	for _, supported := range supportedTypes {
		if strings.ToLower(resourceType) == supported {
			return true
		}
	}
	return false
}

// GetProviderName implements CostModel interface
func (a *AWSCostModel) GetProviderName() string {
	return "aws"
}

// GetInstancePricing returns pricing information for a specific instance type
func (a *AWSCostModel) GetInstancePricing(instanceType string) (*AWSInstancePricing, error) {
	if pricing, exists := a.pricingData[instanceType]; exists {
		return &pricing, nil
	}
	return nil, fmt.Errorf("pricing not available for instance type: %s", instanceType)
}

// GetAvailableInstanceTypes returns all available instance types
func (a *AWSCostModel) GetAvailableInstanceTypes() []string {
	types := make([]string, 0, len(a.pricingData))
	for instanceType := range a.pricingData {
		types = append(types, instanceType)
	}
	return types
}

// GetSpotPrice returns the current spot price for an instance type
func (a *AWSCostModel) GetSpotPrice(instanceType string) (float64, error) {
	if pricing, exists := a.pricingData[instanceType]; exists {
		return pricing.SpotPrice, nil
	}
	return 0, fmt.Errorf("pricing not available for instance type: %s", instanceType)
}

// GetReservedInstanceSavings calculates potential savings with reserved instances
func (a *AWSCostModel) GetReservedInstanceSavings(instanceType string, term int, hoursPerMonth float64) (*ReservedInstanceSavings, error) {
	pricing, exists := a.pricingData[instanceType]
	if !exists {
		return nil, fmt.Errorf("pricing not available for instance type: %s", instanceType)
	}
	
	reservedPrice, exists := pricing.ReservedPrices[term]
	if !exists {
		return nil, fmt.Errorf("reserved instance pricing not available for %d month term", term)
	}
	
	// Calculate costs
	onDemandMonthlyCost := pricing.OnDemandPrice * hoursPerMonth
	reservedMonthlyCost := reservedPrice * hoursPerMonth
	monthlySavings := onDemandMonthlyCost - reservedMonthlyCost
	
	totalSavings := monthlySavings * float64(term)
	savingsPercent := (monthlySavings / onDemandMonthlyCost) * 100
	
	return &ReservedInstanceSavings{
		InstanceType:         instanceType,
		Term:                 term,
		OnDemandMonthlyCost:  onDemandMonthlyCost,
		ReservedMonthlyCost:  reservedMonthlyCost,
		MonthlySavings:       monthlySavings,
		TotalSavings:         totalSavings,
		SavingsPercent:       savingsPercent,
		BreakEvenMonths:      0, // No upfront cost in this model
	}, nil
}

// ReservedInstanceSavings represents potential savings with reserved instances
type ReservedInstanceSavings struct {
	InstanceType         string  `json:"instance_type"`
	Term                 int     `json:"term"`
	OnDemandMonthlyCost  float64 `json:"on_demand_monthly_cost"`
	ReservedMonthlyCost  float64 `json:"reserved_monthly_cost"`
	MonthlySavings       float64 `json:"monthly_savings"`
	TotalSavings         float64 `json:"total_savings"`
	SavingsPercent       float64 `json:"savings_percent"`
	BreakEvenMonths      int     `json:"break_even_months"`
}

// UpdateSpotPrices updates spot pricing data (would typically fetch from AWS API)
func (a *AWSCostModel) UpdateSpotPrices() error {
	// In production, this would make API calls to AWS to get current spot prices
	// For now, we'll simulate price updates
	
	for instanceType, pricing := range a.pricingData {
		// Simulate spot price fluctuation (±20% of current price)
		fluctuation := 0.8 + (0.4 * (float64(time.Now().Unix()%100) / 100.0))
		newSpotPrice := pricing.SpotPrice * fluctuation
		
		// Ensure spot price doesn't exceed on-demand price
		if newSpotPrice > pricing.OnDemandPrice {
			newSpotPrice = pricing.OnDemandPrice * 0.9
		}
		
		pricing.SpotPrice = newSpotPrice
		a.pricingData[instanceType] = pricing
	}
	
	return nil
}

// GetCostOptimizedInstanceType returns the most cost-effective instance type for given requirements
func (a *AWSCostModel) GetCostOptimizedInstanceType(minCPUs int, minMemoryGB float64, pricingModel string) (*InstanceRecommendation, error) {
	var bestOption *InstanceRecommendation
	bestCostEfficiency := 0.0
	
	for instanceType, pricing := range a.pricingData {
		// Check if instance meets minimum requirements
		if pricing.vCPUs < minCPUs || pricing.MemoryGB < minMemoryGB {
			continue
		}
		
		// Get price based on pricing model
		var price float64
		switch strings.ToLower(pricingModel) {
		case "spot":
			price = pricing.SpotPrice
		case "reserved":
			if reservedPrice, exists := pricing.ReservedPrices[12]; exists { // 1-year term
				price = reservedPrice
			} else {
				price = pricing.OnDemandPrice
			}
		default:
			price = pricing.OnDemandPrice
		}
		
		// Calculate cost efficiency (performance per dollar)
		// Simple metric: (CPU + Memory) / price
		performance := float64(pricing.vCPUs) + pricing.MemoryGB
		costEfficiency := performance / price
		
		if costEfficiency > bestCostEfficiency {
			bestCostEfficiency = costEfficiency
			bestOption = &InstanceRecommendation{
				InstanceType:     instanceType,
				HourlyCost:       price,
				vCPUs:            pricing.vCPUs,
				MemoryGB:         pricing.MemoryGB,
				CostEfficiency:   costEfficiency,
				PricingModel:     pricingModel,
				SavingsPercent:   0.0,
			}
			
			// Calculate savings compared to on-demand
			if pricingModel != "on-demand" {
				savings := pricing.OnDemandPrice - price
				if savings > 0 {
					bestOption.SavingsPercent = (savings / pricing.OnDemandPrice) * 100
				}
			}
		}
	}
	
	if bestOption == nil {
		return nil, fmt.Errorf("no suitable instance type found for requirements: %d CPUs, %.1f GB memory", minCPUs, minMemoryGB)
	}
	
	return bestOption, nil
}

// InstanceRecommendation represents a cost-optimized instance recommendation
type InstanceRecommendation struct {
	InstanceType     string  `json:"instance_type"`
	HourlyCost       float64 `json:"hourly_cost"`
	vCPUs            int     `json:"vcpus"`
	MemoryGB         float64 `json:"memory_gb"`
	CostEfficiency   float64 `json:"cost_efficiency"`
	PricingModel     string  `json:"pricing_model"`
	SavingsPercent   float64 `json:"savings_percent"`
}

// GetRegionalPricingDiff returns pricing differences across regions
func (a *AWSCostModel) GetRegionalPricingDiff(instanceType string, targetRegions []string) (map[string]float64, error) {
	// In production, this would fetch pricing for different regions
	// For now, simulate regional price differences
	
	pricing, exists := a.pricingData[instanceType]
	if !exists {
		return nil, fmt.Errorf("pricing not available for instance type: %s", instanceType)
	}
	
	baseCost := pricing.OnDemandPrice
	regionalPricing := make(map[string]float64)
	
	// Simulate regional price variations
	regionMultipliers := map[string]float64{
		"us-east-1":      1.0,    // baseline
		"us-west-1":      1.15,   // 15% higher
		"us-west-2":      1.10,   // 10% higher
		"eu-west-1":      1.20,   // 20% higher
		"eu-central-1":   1.25,   // 25% higher
		"ap-southeast-1": 1.30,   // 30% higher
		"ap-northeast-1": 1.35,   // 35% higher
	}
	
	for _, region := range targetRegions {
		multiplier := regionMultipliers[region]
		if multiplier == 0 {
			multiplier = 1.0 // default to baseline if region not found
		}
		regionalPricing[region] = baseCost * multiplier
	}
	
	return regionalPricing, nil
}