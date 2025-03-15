package monitoring

// EnhancedCloudVMStats extends the CloudVMStats with additional fields for advanced monitoring
type EnhancedCloudVMStats struct {
	// Embed the base CloudVMStats
	CloudVMStats

	// NativeMetrics contains provider-specific metrics in their raw format
	NativeMetrics map[string]float64

	// CostData contains billing information if available
	CostData map[string]float64

	// ResourceQuotas contains information about resource quotas and limits
	ResourceQuotas map[string]float64

	// PerformanceIndex contains calculated performance values
	PerformanceIndex map[string]float64

	// ResourcePredictions contains predicted resource usage values
	ResourcePredictions map[string]float64

	// CloudEvents contains related cloud events
	CloudEvents []string
}

// ConvertToEnhanced converts a standard CloudVMStats to EnhancedCloudVMStats
func ConvertToEnhanced(stats *CloudVMStats) *EnhancedCloudVMStats {
	if stats == nil {
		return nil
	}

	enhanced := &EnhancedCloudVMStats{
		CloudVMStats:        *stats,
		NativeMetrics:       make(map[string]float64),
		CostData:            make(map[string]float64),
		ResourceQuotas:      make(map[string]float64),
		PerformanceIndex:    make(map[string]float64),
		ResourcePredictions: make(map[string]float64),
		CloudEvents:         []string{},
	}

	return enhanced
}

// ConvertToCloudStats converts an EnhancedCloudVMStats back to CloudVMStats
func (e *EnhancedCloudVMStats) ConvertToCloudStats() *CloudVMStats {
	return &e.CloudVMStats
}

// ConvertEnhancedToInternal converts EnhancedCloudVMStats to internal VMStats
func ConvertEnhancedToInternal(enhanced *EnhancedCloudVMStats) *VMStats {
	if enhanced == nil {
		return nil
	}

	// First convert the base CloudVMStats
	base := ConvertCloudToInternalStats(&enhanced.CloudVMStats)

	// Then add enhanced fields if needed
	// For most cases, these will be added as additional metrics or metadata

	return base
}

// ConvertInternalToEnhanced converts internal VMStats to EnhancedCloudVMStats
func ConvertInternalToEnhanced(stats *VMStats) *EnhancedCloudVMStats {
	if stats == nil {
		return nil
	}

	// First convert to base CloudVMStats
	cloudStats := ConvertInternalToCloudStats(stats)

	// Then wrap with enhanced structure
	enhanced := ConvertToEnhanced(cloudStats)

	return enhanced
}
