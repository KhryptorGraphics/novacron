package workload

import (
	"fmt"
	"math"
	"sort"
	"time"
)

// WorkloadClassifier analyzes workload metrics and characteristics
// to classify and optimize workload placement
type WorkloadClassifier struct {
	// Thresholds for classification
	cpuIntensiveThreshold     float64
	memoryIntensiveThreshold  float64
	ioIntensiveThreshold      float64
	networkIntensiveThreshold float64

	// Pattern detection thresholds
	patternRecognitionThreshold float64
	variabilityThreshold        float64

	// Provider-specific instance type mappings
	// Maps workload types to optimal instance types per provider
	instanceTypeMappings map[string]map[WorkloadType][]string

	// Provider-specific pricing data
	// Maps provider -> instance type -> hourly cost
	pricingData map[string]map[string]float64

	// Last pricing data update
	lastPricingUpdate time.Time
}

// NewWorkloadClassifier creates a new workload classifier with default settings
func NewWorkloadClassifier() *WorkloadClassifier {
	return &WorkloadClassifier{
		cpuIntensiveThreshold:       70.0,   // 70% avg CPU usage
		memoryIntensiveThreshold:    70.0,   // 70% avg memory usage
		ioIntensiveThreshold:        1000.0, // 1000 IOPS average
		networkIntensiveThreshold:   50.0,   // 50 MB/s combined in/out
		patternRecognitionThreshold: 0.75,   // Correlation coefficient threshold
		variabilityThreshold:        0.25,   // Coefficient of variation threshold
		instanceTypeMappings:        make(map[string]map[WorkloadType][]string),
		pricingData:                 make(map[string]map[string]float64),
	}
}

// ClassifyWorkload analyzes a workload's metrics and returns a WorkloadProfile
func (c *WorkloadClassifier) ClassifyWorkload(metrics Metrics) WorkloadCharacteristics {
	// Start with a default classification
	characteristics := WorkloadCharacteristics{
		Type:             GeneralPurpose,
		CPUIntensive:     false,
		CPUStability:     true,
		MemoryIntensive:  false,
		MemoryStability:  true,
		IOIntensive:      false,
		IOPattern:        BalancedIO,
		NetworkIntensive: false,
		NetworkPattern:   BalancedNetwork,
		Interruptible:    true, // Assume interruptible by default
	}

	// CPU characteristics
	characteristics.CPUIntensive = metrics.AvgCPUUtilization >= c.cpuIntensiveThreshold
	characteristics.CPUStability = c.isStable(metrics.AvgCPUUtilization, metrics.CPUUtilizationStdDev)

	// Memory characteristics
	characteristics.MemoryIntensive = metrics.AvgMemoryUtilization >= c.memoryIntensiveThreshold
	characteristics.MemoryStability = c.isStable(metrics.AvgMemoryUtilization, metrics.MemoryUtilizationStdDev)

	// IO characteristics
	characteristics.IOIntensive = metrics.AvgIOPS >= c.ioIntensiveThreshold
	characteristics.IOPattern = c.determineIOPattern(metrics)

	// Network characteristics
	totalNetworkThroughput := metrics.AvgNetworkIn + metrics.AvgNetworkOut
	characteristics.NetworkIntensive = totalNetworkThroughput >= c.networkIntensiveThreshold
	characteristics.NetworkPattern = c.determineNetworkPattern(metrics)

	// Determine workload type based on characteristics
	characteristics.Type = c.determineWorkloadType(metrics, characteristics)

	// Determine scheduling characteristics
	characteristics.Interruptible = c.isInterruptible(metrics, characteristics)
	characteristics.TimeOfDay = c.determineTimeWindows(metrics.TimeOfDayPatterns)
	characteristics.DayOfWeek = c.determineDaysOfWeek(metrics.DayOfWeekPatterns)

	return characteristics
}

// UpdateProviderMappings updates the instance type mappings for a provider
func (c *WorkloadClassifier) UpdateProviderMappings(provider string, mappings map[WorkloadType][]string) {
	if c.instanceTypeMappings == nil {
		c.instanceTypeMappings = make(map[string]map[WorkloadType][]string)
	}
	c.instanceTypeMappings[provider] = mappings
}

// UpdatePricingData updates the pricing data for a provider
func (c *WorkloadClassifier) UpdatePricingData(provider string, pricing map[string]float64) {
	if c.pricingData == nil {
		c.pricingData = make(map[string]map[string]float64)
	}
	c.pricingData[provider] = pricing
	c.lastPricingUpdate = time.Now()
}

// GetLastPricingUpdate returns the time of the last pricing data update
func (c *WorkloadClassifier) GetLastPricingUpdate() time.Time {
	return c.lastPricingUpdate
}

// OptimizeWorkloadPlacement determines the optimal provider and instance type for a workload
func (c *WorkloadClassifier) OptimizeWorkloadPlacement(
	profile WorkloadProfile) map[string]ProviderFitScore {

	result := make(map[string]ProviderFitScore)

	// For each provider, calculate fitness score
	for providerName, instanceTypes := range c.instanceTypeMappings {
		// Get recommended instance types for this workload type
		recommendedTypes, ok := instanceTypes[profile.Characteristics.Type]
		if !ok {
			// If no specific mapping exists, fall back to general purpose
			recommendedTypes = instanceTypes[GeneralPurpose]
		}

		// Find optimal instance type based on metrics and pricing
		optimalType, cost := c.findOptimalInstanceType(
			providerName,
			recommendedTypes,
			profile.Characteristics,
			profile.Metrics,
			profile.RequestedCPU,
			profile.RequestedMemoryGB,
		)

		// Calculate scores
		costScore := c.calculateCostScore(cost, profile.TargetMonthlyCost)
		perfScore := c.calculatePerformanceScore(providerName, optimalType, profile.Characteristics)
		reliabilityScore := c.calculateReliabilityScore(providerName, profile.Characteristics)
		complianceScore := c.calculateComplianceScore(providerName, profile.Characteristics)

		// Calculate overall score (weighted average)
		overallScore := (costScore * 0.4) +
			(perfScore * 0.3) +
			(reliabilityScore * 0.2) +
			(complianceScore * 0.1)

		// Determine recommended action
		recommendedAction, reason := c.determineRecommendedAction(
			providerName,
			optimalType,
			profile.Characteristics,
			costScore,
			overallScore,
		)

		// Create provider fit score
		result[providerName] = ProviderFitScore{
			ProviderName:         providerName,
			OverallScore:         overallScore,
			CostScore:            costScore,
			PerformanceScore:     perfScore,
			ReliabilityScore:     reliabilityScore,
			ComplianceScore:      complianceScore,
			OptimalInstanceType:  optimalType,
			EstimatedMonthlyCost: cost * 730, // 730 hours in a month
			RecommendedAction:    recommendedAction,
			ReasonForScore:       reason,
		}
	}

	return result
}

// Helper functions

// isStable determines if a metric is stable based on its standard deviation
func (c *WorkloadClassifier) isStable(avg, stdDev float64) bool {
	if avg == 0 {
		return true // Avoid division by zero
	}

	// Calculate coefficient of variation
	cv := stdDev / avg
	return cv < c.variabilityThreshold
}

// determineIOPattern analyzes IO metrics to determine the pattern
func (c *WorkloadClassifier) determineIOPattern(metrics Metrics) IOPattern {
	// Determine read/write balance
	if metrics.ReadWriteRatio > 2.0 {
		// More than 2:1 read:write ratio
		return ReadHeavy
	} else if metrics.ReadWriteRatio < 0.5 {
		// More than 2:1 write:read ratio
		return WriteHeavy
	}

	// Determine random vs sequential
	if metrics.RandomIOPercentage > 70 {
		return RandomAccess
	} else if metrics.RandomIOPercentage < 30 {
		return SequentialAccess
	}

	// Default to balanced
	return BalancedIO
}

// determineNetworkPattern analyzes network metrics to determine the pattern
func (c *WorkloadClassifier) determineNetworkPattern(metrics Metrics) NetworkPattern {
	// Calculate total network throughput
	totalNetwork := metrics.AvgNetworkIn + metrics.AvgNetworkOut

	// Skip pattern detection for low traffic
	if totalNetwork < 1.0 {
		return BalancedNetwork
	}

	// Determine inbound/outbound balance
	inboundRatio := metrics.AvgNetworkIn / totalNetwork

	if inboundRatio > 0.7 {
		return InboundHeavy
	} else if inboundRatio < 0.3 {
		return OutboundHeavy
	}

	// Default to balanced
	return BalancedNetwork
}

// determineWorkloadType analyzes characteristics to determine the workload type
func (c *WorkloadClassifier) determineWorkloadType(
	metrics Metrics,
	characteristics WorkloadCharacteristics) WorkloadType {

	// Scoring system for workload types
	scores := make(map[WorkloadType]float64)

	// Web server characteristics
	// High network connections, balanced in/out, stable CPU, low-medium CPU usage
	if characteristics.NetworkPattern == BalancedNetwork ||
		characteristics.NetworkPattern == InboundHeavy {
		if metrics.AvgActiveConnections > 100 &&
			characteristics.CPUStability &&
			metrics.AvgCPUUtilization < 70 {
			scores[WebServer] = 0.8
		}
	}

	// Batch processing characteristics
	// Periodic usage patterns, high CPU or memory during execution
	if hasClearTimePattern(metrics.TimeOfDayPatterns, metrics.WeeklyPatternQuality) {
		if characteristics.CPUIntensive || characteristics.MemoryIntensive {
			scores[BatchProcessing] = 0.7
		}
	}

	// Database workload characteristics
	// High IO, mixture of random/sequential access, stable operation
	if characteristics.IOIntensive &&
		(characteristics.IOPattern == RandomAccess || characteristics.IOPattern == BalancedIO) {
		scores[DatabaseWorkload] = 0.75
	}

	// ML training characteristics
	// Extremely high CPU/GPU utilization, high memory usage
	if characteristics.CPUIntensive && characteristics.MemoryIntensive &&
		metrics.PeakCPUUtilization > 90 && metrics.AvgCPUUtilization > 80 {
		scores[MLTraining] = 0.9
	}

	// ML inference characteristics
	// Burst CPU usage, network traffic, lower average utilization
	if metrics.PeakCPUUtilization > 80 && metrics.AvgCPUUtilization < 50 &&
		characteristics.NetworkIntensive {
		scores[MLInference] = 0.7
	}

	// Analytics workload characteristics
	// High memory usage, periodic execution, IO intensive
	if characteristics.MemoryIntensive && characteristics.IOIntensive &&
		hasClearTimePattern(metrics.TimeOfDayPatterns, metrics.WeeklyPatternQuality) {
		scores[AnalyticsWorkload] = 0.8
	}

	// Development/Testing characteristics
	// Low average utilization, highly variable usage, not consistently used
	if !characteristics.CPUIntensive && !characteristics.MemoryIntensive &&
		!characteristics.IOIntensive && !characteristics.NetworkIntensive &&
		!characteristics.CPUStability {
		scores[DevTest] = 0.6
	}

	// Default score for general purpose
	scores[GeneralPurpose] = 0.3

	// Find workload type with highest score
	var highestType WorkloadType = GeneralPurpose
	var highestScore float64 = 0

	for wType, score := range scores {
		if score > highestScore {
			highestScore = score
			highestType = wType
		}
	}

	return highestType
}

// hasClearTimePattern checks if there's a clear pattern in time-based metrics
func hasClearTimePattern(patterns map[int]float64, quality float64) bool {
	// If the pattern quality is high enough, there's a clear pattern
	if quality >= 0.75 {
		return true
	}

	// Check for significant differences between time periods
	if len(patterns) < 2 {
		return false
	}

	// Convert map to slice for analysis
	values := make([]float64, 0, len(patterns))
	for _, v := range patterns {
		values = append(values, v)
	}

	// Calculate min and max
	min, max := values[0], values[0]
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}

	// If the max is at least 3x the min, there's a clear pattern
	return max >= min*3
}

// isInterruptible determines if a workload can be safely interrupted
func (c *WorkloadClassifier) isInterruptible(
	metrics Metrics,
	characteristics WorkloadCharacteristics) bool {

	// Database workloads typically can't be interrupted
	if characteristics.Type == DatabaseWorkload {
		return false
	}

	// Web servers typically shouldn't be interrupted
	if characteristics.Type == WebServer {
		return false
	}

	// Batch processing can often be interrupted
	if characteristics.Type == BatchProcessing {
		return true
	}

	// Analytics workloads can often be interrupted
	if characteristics.Type == AnalyticsWorkload {
		return true
	}

	// ML Training depends on checkpointing capabilities
	// (assuming most modern frameworks support this)
	if characteristics.Type == MLTraining {
		return true
	}

	// Dev/Test environments are always interruptible
	if characteristics.Type == DevTest {
		return true
	}

	// Default to not interruptible for unknown workloads
	return false
}

// determineTimeWindows analyzes time patterns to find peak usage windows
func (c *WorkloadClassifier) determineTimeWindows(patterns map[int]float64) []TimeWindow {
	if len(patterns) == 0 {
		// Return empty slice if no patterns
		return []TimeWindow{}
	}

	// Calculate average utilization
	var sum float64
	for _, v := range patterns {
		sum += v
	}
	avg := sum / float64(len(patterns))

	// Find hours with above-average utilization
	activeHours := make([]int, 0)
	for hour, utilization := range patterns {
		if utilization > avg {
			activeHours = append(activeHours, hour)
		}
	}

	// Sort hours
	sort.Ints(activeHours)

	// Group consecutive hours into windows
	windows := make([]TimeWindow, 0)
	if len(activeHours) > 0 {
		start := activeHours[0]
		prev := start

		for i := 1; i < len(activeHours); i++ {
			hour := activeHours[i]

			// Check if hour is consecutive
			if hour != prev+1 {
				// End of a window
				windows = append(windows, TimeWindow{
					StartHour: start,
					EndHour:   prev,
				})

				// Start a new window
				start = hour
			}

			prev = hour
		}

		// Add the last window
		windows = append(windows, TimeWindow{
			StartHour: start,
			EndHour:   prev,
		})
	}

	return windows
}

// determineDaysOfWeek analyzes day patterns to find active days
func (c *WorkloadClassifier) determineDaysOfWeek(patterns map[int]float64) []int {
	if len(patterns) == 0 {
		// Return empty slice if no patterns
		return []int{}
	}

	// Calculate average utilization
	var sum float64
	for _, v := range patterns {
		sum += v
	}
	avg := sum / float64(len(patterns))

	// Find days with above-average utilization
	activeDays := make([]int, 0)
	for day, utilization := range patterns {
		if utilization > avg {
			activeDays = append(activeDays, day)
		}
	}

	// Sort days
	sort.Ints(activeDays)

	return activeDays
}

// findOptimalInstanceType finds the best instance type for a workload
func (c *WorkloadClassifier) findOptimalInstanceType(
	provider string,
	candidateTypes []string,
	characteristics WorkloadCharacteristics,
	metrics Metrics,
	requestedCPU int,
	requestedMemoryGB int) (string, float64) {

	// Default to first instance type if no pricing data available
	if len(candidateTypes) == 0 {
		return "", 0
	}

	bestType := candidateTypes[0]
	bestCost := math.MaxFloat64

	// Get pricing data for this provider
	pricing, ok := c.pricingData[provider]
	if !ok {
		return bestType, 0 // No pricing data available
	}

	// For each instance type, calculate fitness and cost
	for _, instanceType := range candidateTypes {
		cost, ok := pricing[instanceType]
		if !ok {
			continue // Skip if no pricing data for this instance type
		}

		// TODO: Implement more sophisticated instance type selection based on
		// workload requirements (CPU, memory, storage, network, etc.)

		// For now, just pick the cheapest one that meets basic requirements
		if cost < bestCost {
			bestType = instanceType
			bestCost = cost
		}
	}

	return bestType, bestCost
}

// calculateCostScore returns a score from 0-1 based on how well the cost
// matches the target cost
func (c *WorkloadClassifier) calculateCostScore(cost, targetCost float64) float64 {
	if targetCost <= 0 {
		// No target cost specified
		return 0.5
	}

	// Calculate ratio of actual cost to target cost
	ratio := cost / targetCost

	if ratio <= 1.0 {
		// Below target cost is good (score approaches 1 as cost decreases)
		return 1.0 - (ratio * 0.5)
	} else {
		// Above target cost is bad (score approaches 0 as cost increases)
		return math.Max(0, 0.5-(ratio-1.0)*0.5)
	}
}

// calculatePerformanceScore returns a score from 0-1 based on expected
// performance for the workload type on the provider
func (c *WorkloadClassifier) calculatePerformanceScore(provider, instanceType string, chars WorkloadCharacteristics) float64 {
	// This would be based on benchmarks, historical performance data, etc.
	// For now, return a placeholder value
	return 0.7
}

// calculateReliabilityScore returns a score from 0-1 based on provider reliability
func (c *WorkloadClassifier) calculateReliabilityScore(provider string, chars WorkloadCharacteristics) float64 {
	// This would be based on historical uptime, SLAs, etc.
	// For now, return placeholder values by provider
	switch provider {
	case "aws":
		return 0.95
	case "azure":
		return 0.93
	case "gcp":
		return 0.94
	default:
		return 0.7
	}
}

// calculateComplianceScore returns 1 if compliant, 0 if not
func (c *WorkloadClassifier) calculateComplianceScore(provider string, chars WorkloadCharacteristics) float64 {
	// Check if provider supports all required compliance standards
	// This would check provider capabilities against workload requirements
	// For now, return a placeholder value
	return 1.0
}

// determineRecommendedAction suggests an action based on workload and scores
func (c *WorkloadClassifier) determineRecommendedAction(
	provider string,
	instanceType string,
	chars WorkloadCharacteristics,
	costScore float64,
	overallScore float64) (string, string) {

	// Decision logic for recommendations
	if costScore < 0.3 {
		return "resize", fmt.Sprintf("Instance type %s is too expensive for this workload", instanceType)
	}

	if overallScore < 0.5 {
		return "migrate", fmt.Sprintf("Workload would perform better on a different provider")
	}

	if chars.Type == BatchProcessing && chars.Interruptible {
		return "use_spot_instance", "Batch workload is suitable for spot/preemptible instances"
	}

	if chars.Type == DevTest {
		return "schedule_shutdown", "Development/test environment can be shut down during non-working hours"
	}

	return "maintain", "Current configuration is optimal"
}
