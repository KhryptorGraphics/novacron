package migration

import (
	"context"
	"fmt"
	"log"
	"math"
	"time"
)

// FallbackMigrationStrategy provides heuristic-based migration optimization when AI is unavailable
type FallbackMigrationStrategy struct {
	config MigrationConfig
	logger *log.Logger
}

// NewFallbackMigrationStrategy creates a new fallback strategy
func NewFallbackMigrationStrategy(config MigrationConfig) *FallbackMigrationStrategy {
	return &FallbackMigrationStrategy{
		config: config,
		logger: log.New(log.Writer(), "[MigrationFallback] ", log.LstdFlags),
	}
}

// PredictMigrationTime uses historical averages and VM size to estimate migration time
func (f *FallbackMigrationStrategy) PredictMigrationTime(sourceNode, destNode, vmSize string) (time.Duration, float64) {
	// Base estimation based on VM size
	baseDuration := 5 * time.Minute

	switch vmSize {
	case "small":
		baseDuration = 2 * time.Minute
	case "medium":
		baseDuration = 5 * time.Minute
	case "large":
		baseDuration = 10 * time.Minute
	case "xlarge":
		baseDuration = 20 * time.Minute
	default:
		// Parse size if it's in GB format
		baseDuration = 5 * time.Minute
	}

	// Adjust for network distance (simplified)
	if sourceNode != destNode {
		// Different nodes, add network overhead
		baseDuration = time.Duration(float64(baseDuration) * 1.3)
	}

	// Lower confidence for heuristic predictions
	confidence := 0.6

	return baseDuration, confidence
}

// PredictBandwidthRequirements estimates bandwidth needs based on VM size and time constraints
func (f *FallbackMigrationStrategy) PredictBandwidthRequirements(vmSize, networkConditions string) int64 {
	// Base bandwidth calculation
	var vmSizeGB int64 = 10 // Default 10GB

	switch vmSize {
	case "small":
		vmSizeGB = 5
	case "medium":
		vmSizeGB = 20
	case "large":
		vmSizeGB = 50
	case "xlarge":
		vmSizeGB = 100
	}

	// Convert to bytes
	vmSizeBytes := vmSizeGB * 1024 * 1024 * 1024

	// Target transfer time (from config or default)
	targetTime := 10 * time.Minute
	if f.config.MaxDowntime > 0 {
		// Use a portion of max downtime as target
		targetTime = f.config.MaxDowntime * 20 // Rough estimate
	}

	// Calculate required bandwidth (bytes per second)
	requiredBandwidth := vmSizeBytes / int64(targetTime.Seconds())

	// Adjust for network conditions
	switch networkConditions {
	case "congested":
		requiredBandwidth = int64(float64(requiredBandwidth) * 1.5)
	case "optimal":
		requiredBandwidth = int64(float64(requiredBandwidth) * 0.8)
	}

	// Cap at reasonable limits
	maxBandwidth := int64(10 * 1024 * 1024 * 1024 / 8) // 10 Gbps
	if requiredBandwidth > maxBandwidth {
		requiredBandwidth = maxBandwidth
	}

	return requiredBandwidth
}

// PredictOptimalPath provides simple path selection based on heuristics
func (f *FallbackMigrationStrategy) PredictOptimalPath(sourceNode, destNode string, networkTopology map[string]interface{}) []string {
	// Simple direct path as fallback
	path := []string{sourceNode}

	// Check if we need intermediate hops (simplified)
	if sourceNode != destNode {
		// Check if direct connection exists in topology
		if topology, ok := networkTopology["connections"].(map[string][]string); ok {
			if destinations, exists := topology[sourceNode]; exists {
				for _, dest := range destinations {
					if dest == destNode {
						// Direct connection exists
						path = append(path, destNode)
						return path
					}
				}
			}
		}

		// No direct connection, add a relay node (simplified)
		path = append(path, "relay-node")
	}

	path = append(path, destNode)
	return path
}

// OptimizeMigrationStrategy provides rule-based migration strategy selection
func (f *FallbackMigrationStrategy) OptimizeMigrationStrategy(vmData, networkData map[string]interface{}) MigrationStrategy {
	strategy := MigrationStrategy{
		Type:             MigrationTypeLive,
		MemoryIterations: f.config.MemoryIterations,
		CompressionLevel: f.config.CompressionLevel,
		Confidence:       0.5,
	}

	// Adjust based on VM characteristics
	if memSize, ok := vmData["memory_size"].(string); ok {
		// Large memory VMs may need more iterations
		if memSize == "large" || memSize == "xlarge" {
			strategy.MemoryIterations = 5
			strategy.Type = MigrationTypeHybrid // Use hybrid for large VMs
		}
	}

	// Adjust based on network conditions
	if bandwidth, ok := networkData["bandwidth"].(int64); ok {
		if bandwidth < 1024*1024*1024 { // Less than 1 Gbps
			// Low bandwidth, increase compression
			strategy.CompressionLevel = 9
		} else if bandwidth > 10*1024*1024*1024 { // More than 10 Gbps
			// High bandwidth, reduce compression overhead
			strategy.CompressionLevel = 1
		}

		strategy.BandwidthAllocation = bandwidth
	}

	// Workload-specific adjustments
	if workloadType, ok := vmData["workload_type"].(string); ok {
		switch workloadType {
		case "database":
			// Databases need minimal downtime
			strategy.Type = MigrationTypePostCopy
		case "web":
			// Web servers can tolerate brief downtime
			strategy.Type = MigrationTypePreCopy
		case "batch":
			// Batch processing can use cold migration
			strategy.Type = MigrationTypeCold
		}
	}

	return strategy
}

// OptimizeCompressionSettings provides simple compression configuration
func (f *FallbackMigrationStrategy) OptimizeCompressionSettings(dataProfile map[string]interface{}) CompressionConfig {
	config := CompressionConfig{
		Type:       CompressionTypeLZ4, // Default to fast compression
		Level:      5,                  // Medium compression
		ChunkSize:  1024 * 1024,        // 1MB chunks
		Confidence: 0.5,
	}

	// Adjust based on data characteristics
	if dataType, ok := dataProfile["type"].(string); ok {
		switch dataType {
		case "text", "logs":
			// Text compresses well
			config.Type = CompressionTypeGzip
			config.Level = 7
		case "binary", "executable":
			// Binary data needs different approach
			config.Type = CompressionTypeZstd
			config.Level = 3
		case "media", "video":
			// Media is often already compressed
			config.Type = CompressionTypeLZ4
			config.Level = 1
		}
	}

	// Adjust chunk size based on available memory
	if memAvailable, ok := dataProfile["memory_available"].(int64); ok {
		if memAvailable > 8*1024*1024*1024 { // More than 8GB
			config.ChunkSize = 4 * 1024 * 1024 // 4MB chunks
		} else if memAvailable < 2*1024*1024*1024 { // Less than 2GB
			config.ChunkSize = 256 * 1024 // 256KB chunks
		}
	}

	return config
}

// DetectAnomalies provides threshold-based anomaly detection for migrations
func (f *FallbackMigrationStrategy) DetectAnomalies(migrationMetrics map[string]interface{}) []AnomalyAlert {
	var anomalies []AnomalyAlert

	// Check transfer rate
	if transferRate, ok := migrationMetrics["transfer_rate"].(int64); ok {
		expectedRate := f.config.TargetTransferRate
		if expectedRate > 0 && transferRate < expectedRate/2 {
			anomalies = append(anomalies, AnomalyAlert{
				Type:       "performance",
				Severity:   "warning",
				Message:    fmt.Sprintf("Transfer rate below expected: %d bytes/s (expected: %d)", transferRate, expectedRate),
				Confidence: 0.7,
				Timestamp:  time.Now(),
				Recommendations: []string{
					"Check network congestion",
					"Consider increasing compression",
					"Verify source and destination I/O performance",
				},
			})
		}
	}

	// Check dirty pages convergence
	if dirtyPages, ok := migrationMetrics["dirty_pages"].(int64); ok {
		if iterations, ok := migrationMetrics["iterations"].(int32); ok {
			if iterations > 3 && dirtyPages > 1000 {
				anomalies = append(anomalies, AnomalyAlert{
					Type:       "convergence",
					Severity:   "warning",
					Message:    fmt.Sprintf("High dirty page count after %d iterations: %d pages", iterations, dirtyPages),
					Confidence: 0.8,
					Timestamp:  time.Now(),
					Recommendations: []string{
						"Consider switching to post-copy migration",
						"Reduce VM workload if possible",
						"Increase bandwidth allocation",
					},
				})
			}
		}
	}

	// Check migration duration
	if migrationID, ok := migrationMetrics["migration_id"].(string); ok {
		// This is a simple duration check (would need start time in real implementation)
		anomalies = append(anomalies, AnomalyAlert{
			Type:       "duration",
			Severity:   "info",
			Message:    fmt.Sprintf("Migration %s progress check", migrationID),
			Confidence: 0.5,
			Timestamp:  time.Now(),
			Recommendations: []string{
				"Monitor progress closely",
			},
		})
	}

	return anomalies
}

// RecommendDynamicAdjustments provides rule-based migration adjustments
func (f *FallbackMigrationStrategy) RecommendDynamicAdjustments(migrationID string, currentMetrics map[string]interface{}) []AdjustmentRecommendation {
	var recommendations []AdjustmentRecommendation

	// Check transfer rate and recommend bandwidth adjustment
	if transferRate, ok := currentMetrics["transfer_rate"].(int64); ok {
		if transferRate < f.config.TargetTransferRate/2 {
			recommendations = append(recommendations, AdjustmentRecommendation{
				Parameter:        "bandwidth_limit",
				CurrentValue:     transferRate,
				RecommendedValue: transferRate * 2,
				Reason:           "Transfer rate is below target",
				Confidence:       0.6,
				Impact:           "medium",
			})
		}
	}

	// Check dirty pages and recommend iteration adjustment
	if dirtyPages, ok := currentMetrics["dirty_pages"].(int64); ok {
		if iterations, ok := currentMetrics["iterations"].(int32); ok {
			if iterations > 2 && dirtyPages > 5000 {
				recommendations = append(recommendations, AdjustmentRecommendation{
					Parameter:        "memory_iterations",
					CurrentValue:     iterations,
					RecommendedValue: iterations + 2,
					Reason:           "Dirty pages not converging",
					Confidence:       0.7,
					Impact:           "high",
				})
			}
		}
	}

	// Check phase and recommend compression adjustment
	if phase, ok := currentMetrics["phase"].(string); ok {
		if phase == "memory_copy" || phase == "disk_copy" {
			recommendations = append(recommendations, AdjustmentRecommendation{
				Parameter:        "compression_level",
				CurrentValue:     f.config.CompressionLevel,
				RecommendedValue: math.Min(float64(f.config.CompressionLevel+2), 9),
				Reason:           "Optimize transfer during copy phase",
				Confidence:       0.5,
				Impact:           "low",
			})
		}
	}

	return recommendations
}

// SafeMigrationAIProvider wraps an AI provider with fallback logic
type SafeMigrationAIProvider struct {
	aiProvider MigrationAIProvider
	fallback   *FallbackMigrationStrategy
	logger     *log.Logger
	metrics    *MigrationAIFallbackMetrics
}

// MigrationAIFallbackMetrics tracks fallback usage for migrations
type MigrationAIFallbackMetrics struct {
	totalCalls      int64
	fallbackCalls   int64
	aiFailures      int64
	avgFallbackTime time.Duration
}

// NewSafeMigrationAIProvider creates a new safe AI provider with fallback
func NewSafeMigrationAIProvider(aiProvider MigrationAIProvider, config MigrationConfig) *SafeMigrationAIProvider {
	return &SafeMigrationAIProvider{
		aiProvider: aiProvider,
		fallback:   NewFallbackMigrationStrategy(config),
		logger:     log.New(log.Writer(), "[SafeMigrationAI] ", log.LstdFlags),
		metrics:    &MigrationAIFallbackMetrics{},
	}
}

// PredictMigrationTime with fallback
func (s *SafeMigrationAIProvider) PredictMigrationTime(sourceNode, destNode, vmSize string) (time.Duration, float64, error) {
	s.metrics.totalCalls++

	if s.aiProvider != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()

		type result struct {
			duration   time.Duration
			confidence float64
			err        error
		}

		resultChan := make(chan result, 1)
		go func() {
			dur, conf, err := s.aiProvider.PredictMigrationTime(sourceNode, destNode, vmSize)
			resultChan <- result{dur, conf, err}
		}()

		select {
		case res := <-resultChan:
			if res.err == nil {
				return res.duration, res.confidence, nil
			}
			s.logger.Printf("AI prediction failed: %v, using fallback", res.err)
			s.metrics.aiFailures++
		case <-ctx.Done():
			s.logger.Printf("AI prediction timeout, using fallback")
			s.metrics.aiFailures++
		}
	}

	// Use fallback
	s.metrics.fallbackCalls++
	start := time.Now()
	duration, confidence := s.fallback.PredictMigrationTime(sourceNode, destNode, vmSize)
	s.metrics.avgFallbackTime = time.Since(start)

	return duration, confidence, nil
}

// PredictBandwidthRequirements with fallback
func (s *SafeMigrationAIProvider) PredictBandwidthRequirements(vmSize, networkConditions string) (int64, error) {
	s.metrics.totalCalls++

	if s.aiProvider != nil {
		if bandwidth, err := s.aiProvider.PredictBandwidthRequirements(vmSize, networkConditions); err == nil {
			return bandwidth, nil
		}
		s.metrics.aiFailures++
	}

	s.metrics.fallbackCalls++
	return s.fallback.PredictBandwidthRequirements(vmSize, networkConditions), nil
}

// Implement remaining MigrationAIProvider methods with fallback...
func (s *SafeMigrationAIProvider) PredictOptimalPath(sourceNode, destNode string, networkTopology map[string]interface{}) ([]string, error) {
	if s.aiProvider != nil {
		if path, err := s.aiProvider.PredictOptimalPath(sourceNode, destNode, networkTopology); err == nil {
			return path, nil
		}
	}
	return s.fallback.PredictOptimalPath(sourceNode, destNode, networkTopology), nil
}

func (s *SafeMigrationAIProvider) OptimizeMigrationStrategy(vmData, networkData map[string]interface{}) (MigrationStrategy, error) {
	if s.aiProvider != nil {
		if strategy, err := s.aiProvider.OptimizeMigrationStrategy(vmData, networkData); err == nil {
			return strategy, nil
		}
	}
	return s.fallback.OptimizeMigrationStrategy(vmData, networkData), nil
}

func (s *SafeMigrationAIProvider) OptimizeCompressionSettings(dataProfile map[string]interface{}) (CompressionConfig, error) {
	if s.aiProvider != nil {
		if config, err := s.aiProvider.OptimizeCompressionSettings(dataProfile); err == nil {
			return config, nil
		}
	}
	return s.fallback.OptimizeCompressionSettings(dataProfile), nil
}

func (s *SafeMigrationAIProvider) DetectAnomalies(migrationMetrics map[string]interface{}) ([]AnomalyAlert, error) {
	if s.aiProvider != nil {
		if anomalies, err := s.aiProvider.DetectAnomalies(migrationMetrics); err == nil {
			return anomalies, nil
		}
	}
	return s.fallback.DetectAnomalies(migrationMetrics), nil
}

func (s *SafeMigrationAIProvider) RecommendDynamicAdjustments(migrationID string, currentMetrics map[string]interface{}) ([]AdjustmentRecommendation, error) {
	if s.aiProvider != nil {
		if recs, err := s.aiProvider.RecommendDynamicAdjustments(migrationID, currentMetrics); err == nil {
			return recs, nil
		}
	}
	return s.fallback.RecommendDynamicAdjustments(migrationID, currentMetrics), nil
}

// Implement stub methods for remaining interface requirements
func (s *SafeMigrationAIProvider) OptimizeMemoryIterations(vmMemoryPattern map[string]interface{}) (int, error) {
	if s.aiProvider != nil {
		if iterations, err := s.aiProvider.OptimizeMemoryIterations(vmMemoryPattern); err == nil {
			return iterations, nil
		}
	}
	// Fallback to config default
	return s.fallback.config.MemoryIterations, nil
}

func (s *SafeMigrationAIProvider) AnalyzeNetworkConditions(nodeID string) (NetworkConditions, error) {
	if s.aiProvider != nil {
		if conditions, err := s.aiProvider.AnalyzeNetworkConditions(nodeID); err == nil {
			return conditions, nil
		}
	}
	// Return default conditions
	return NetworkConditions{
		Bandwidth:          1024 * 1024 * 1024, // 1 Gbps
		Latency:            10,                 // 10ms
		PacketLoss:         0.01,               // 0.01%
		Jitter:             2,                  // 2ms
		CongestionLevel:    0.3,                // 30%
		PredictedStability: 0.8,                // 80%
	}, nil
}

func (s *SafeMigrationAIProvider) AnalyzeMigrationPatterns(historicalData []MigrationRecord) ([]PatternInsight, error) {
	if s.aiProvider != nil {
		if patterns, err := s.aiProvider.AnalyzeMigrationPatterns(historicalData); err == nil {
			return patterns, nil
		}
	}
	// Return empty patterns
	return []PatternInsight{}, nil
}

func (s *SafeMigrationAIProvider) PredictFailureRisk(migrationParams map[string]interface{}) (float64, error) {
	if s.aiProvider != nil {
		if risk, err := s.aiProvider.PredictFailureRisk(migrationParams); err == nil {
			return risk, nil
		}
	}
	// Default low risk
	return 0.2, nil
}

// GetMetrics returns metrics about AI fallback usage
func (s *SafeMigrationAIProvider) GetMetrics() map[string]interface{} {
	fallbackRate := float64(0)
	if s.metrics.totalCalls > 0 {
		fallbackRate = float64(s.metrics.fallbackCalls) / float64(s.metrics.totalCalls)
	}

	return map[string]interface{}{
		"total_calls":       s.metrics.totalCalls,
		"fallback_calls":    s.metrics.fallbackCalls,
		"ai_failures":       s.metrics.aiFailures,
		"fallback_rate":     fallbackRate,
		"avg_fallback_time": s.metrics.avgFallbackTime.Milliseconds(),
	}
}
