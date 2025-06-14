package storage

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// StorageTier represents a storage tier with specific characteristics
type StorageTier string

const (
	// TierHot represents the hot tier for frequently accessed data
	TierHot StorageTier = "hot"

	// TierWarm represents the warm tier for moderately accessed data
	TierWarm StorageTier = "warm"

	// TierCold represents the cold tier for infrequently accessed data
	TierCold StorageTier = "cold"

	// TierArchive represents the archive tier for rarely accessed data
	TierArchive StorageTier = "archive"
)

// TierConfig contains configuration for a storage tier
type TierConfig struct {
	// Name of the tier
	Name StorageTier `json:"name"`

	// Cost per GB per month
	CostPerGBMonth float64 `json:"cost_per_gb_month"`

	// Access cost per operation
	AccessCostPerOperation float64 `json:"access_cost_per_operation"`

	// Minimum storage duration (data must stay for this long)
	MinStorageDuration time.Duration `json:"min_storage_duration"`

	// Access latency for this tier
	AccessLatency time.Duration `json:"access_latency"`

	// Maximum capacity in bytes (0 = unlimited)
	MaxCapacity int64 `json:"max_capacity"`

	// Driver to use for this tier
	DriverName string `json:"driver_name"`

	// Driver-specific configuration
	DriverConfig map[string]interface{} `json:"driver_config"`

	// Auto-promotion rules
	PromotionRules TierPromotionRules `json:"promotion_rules"`

	// Auto-demotion rules
	DemotionRules TierDemotionRules `json:"demotion_rules"`
}

// TierPromotionRules defines when data should be promoted to a higher tier
type TierPromotionRules struct {
	// Access frequency threshold (accesses per hour)
	AccessFrequencyThreshold float64 `json:"access_frequency_threshold"`

	// Number of accesses in evaluation window
	AccessCountThreshold int `json:"access_count_threshold"`

	// Evaluation window duration
	EvaluationWindow time.Duration `json:"evaluation_window"`

	// Enabled flag
	Enabled bool `json:"enabled"`
}

// TierDemotionRules defines when data should be demoted to a lower tier
type TierDemotionRules struct {
	// Age threshold for demotion
	AgeThreshold time.Duration `json:"age_threshold"`

	// Inactivity threshold
	InactivityThreshold time.Duration `json:"inactivity_threshold"`

	// Size threshold for automatic demotion
	SizeThreshold int64 `json:"size_threshold"`

	// Enabled flag
	Enabled bool `json:"enabled"`
}

// VolumeAccessStats tracks access patterns for a volume
type VolumeAccessStats struct {
	VolumeID     string    `json:"volume_id"`
	AccessCount  int       `json:"access_count"`
	LastAccessed time.Time `json:"last_accessed"`
	CreatedAt    time.Time `json:"created_at"`
	TotalReads   int64     `json:"total_reads"`
	TotalWrites  int64     `json:"total_writes"`
	BytesRead    int64     `json:"bytes_read"`
	BytesWritten int64     `json:"bytes_written"`

	// Access pattern over time (hourly buckets for last 24 hours)
	HourlyAccess [24]int `json:"hourly_access"`

	// Current tier
	CurrentTier StorageTier `json:"current_tier"`

	// Time when moved to current tier
	TierChangeTime time.Time `json:"tier_change_time"`
}

// TieringManager manages automatic data movement between storage tiers
type TieringManager struct {
	// Tier configurations
	tiers map[StorageTier]TierConfig

	// Volume access statistics
	accessStats map[string]*VolumeAccessStats

	// Storage drivers for each tier
	drivers map[StorageTier]StorageDriver

	// Mutex for protecting shared state
	mu sync.RWMutex

	// Context for cancellation
	ctx    context.Context
	cancel context.CancelFunc

	// Evaluation interval
	evaluationInterval time.Duration

	// Base storage service
	storageService StorageService
}

// TieringConfig contains configuration for the tiering manager
type TieringConfig struct {
	// Tier configurations
	Tiers []TierConfig `json:"tiers"`

	// How often to evaluate tiering rules
	EvaluationInterval time.Duration `json:"evaluation_interval"`

	// Enable automatic tiering
	AutoTieringEnabled bool `json:"auto_tiering_enabled"`

	// Default tier for new volumes
	DefaultTier StorageTier `json:"default_tier"`
}

// DefaultTieringConfig returns a default tiering configuration
func DefaultTieringConfig() TieringConfig {
	return TieringConfig{
		Tiers: []TierConfig{
			{
				Name:               TierHot,
				CostPerGBMonth:     0.023, // High-performance SSD
				AccessCostPerOperation: 0.0,
				MinStorageDuration: 0,
				AccessLatency:      1 * time.Millisecond,
				MaxCapacity:        0, // Unlimited
				DriverName:         "local",
				PromotionRules: TierPromotionRules{
					AccessFrequencyThreshold: 10.0, // 10 accesses per hour
					AccessCountThreshold:     50,
					EvaluationWindow:         24 * time.Hour,
					Enabled:                  true,
				},
				DemotionRules: TierDemotionRules{
					InactivityThreshold: 1 * time.Hour,
					Enabled:             true,
				},
			},
			{
				Name:               TierWarm,
				CostPerGBMonth:     0.0125, // Standard SSD
				AccessCostPerOperation: 0.0001,
				MinStorageDuration: 30 * 24 * time.Hour, // 30 days
				AccessLatency:      10 * time.Millisecond,
				MaxCapacity:        0,
				DriverName:         "local",
				PromotionRules: TierPromotionRules{
					AccessFrequencyThreshold: 2.0, // 2 accesses per hour
					AccessCountThreshold:     10,
					EvaluationWindow:         24 * time.Hour,
					Enabled:                  true,
				},
				DemotionRules: TierDemotionRules{
					InactivityThreshold: 7 * 24 * time.Hour, // 7 days
					Enabled:             true,
				},
			},
			{
				Name:               TierCold,
				CostPerGBMonth:     0.004, // Standard HDD
				AccessCostPerOperation: 0.001,
				MinStorageDuration: 90 * 24 * time.Hour, // 90 days
				AccessLatency:      100 * time.Millisecond,
				MaxCapacity:        0,
				DriverName:         "s3",
				PromotionRules: TierPromotionRules{
					AccessFrequencyThreshold: 0.1, // 0.1 accesses per hour
					AccessCountThreshold:     3,
					EvaluationWindow:         7 * 24 * time.Hour,
					Enabled:                  true,
				},
				DemotionRules: TierDemotionRules{
					InactivityThreshold: 30 * 24 * time.Hour, // 30 days
					Enabled:             true,
				},
			},
			{
				Name:               TierArchive,
				CostPerGBMonth:     0.001, // Archive storage
				AccessCostPerOperation: 0.01,
				MinStorageDuration: 180 * 24 * time.Hour, // 180 days
				AccessLatency:      4 * time.Hour,        // Retrieval time
				MaxCapacity:        0,
				DriverName:         "s3",
				DemotionRules: TierDemotionRules{
					InactivityThreshold: 365 * 24 * time.Hour, // 1 year
					Enabled:             true,
				},
			},
		},
		EvaluationInterval: 1 * time.Hour,
		AutoTieringEnabled: true,
		DefaultTier:        TierHot,
	}
}

// NewTieringManager creates a new tiering manager
func NewTieringManager(config TieringConfig, storageService StorageService) (*TieringManager, error) {
	ctx, cancel := context.WithCancel(context.Background())

	manager := &TieringManager{
		tiers:              make(map[StorageTier]TierConfig),
		accessStats:        make(map[string]*VolumeAccessStats),
		drivers:            make(map[StorageTier]StorageDriver),
		ctx:                ctx,
		cancel:             cancel,
		evaluationInterval: config.EvaluationInterval,
		storageService:     storageService,
	}

	// Initialize tier configurations
	for _, tierConfig := range config.Tiers {
		manager.tiers[tierConfig.Name] = tierConfig

		// Create driver for this tier
		driver, err := CreateDriver(tierConfig.DriverName, tierConfig.DriverConfig)
		if err != nil {
			cancel()
			return nil, fmt.Errorf("failed to create driver for tier %s: %w", tierConfig.Name, err)
		}

		if err := driver.Initialize(); err != nil {
			cancel()
			return nil, fmt.Errorf("failed to initialize driver for tier %s: %w", tierConfig.Name, err)
		}

		manager.drivers[tierConfig.Name] = driver
	}

	return manager, nil
}

// Start starts the tiering manager
func (tm *TieringManager) Start() error {
	log.Println("Starting storage tiering manager")

	// Start the evaluation loop
	go tm.evaluationLoop()

	// Start access tracking
	go tm.trackAccessPatterns()

	return nil
}

// Stop stops the tiering manager
func (tm *TieringManager) Stop() error {
	log.Println("Stopping storage tiering manager")

	tm.cancel()

	// Shutdown all drivers
	for tier, driver := range tm.drivers {
		if err := driver.Shutdown(); err != nil {
			log.Printf("Error shutting down driver for tier %s: %v", tier, err)
		}
	}

	return nil
}

// RecordAccess records an access to a volume
func (tm *TieringManager) RecordAccess(volumeID string, accessType string, bytesTransferred int64) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	stats, exists := tm.accessStats[volumeID]
	if !exists {
		stats = &VolumeAccessStats{
			VolumeID:       volumeID,
			CreatedAt:      time.Now(),
			CurrentTier:    TierHot, // Default tier
			TierChangeTime: time.Now(),
		}
		tm.accessStats[volumeID] = stats
	}

	// Update access statistics
	stats.AccessCount++
	stats.LastAccessed = time.Now()

	// Update access pattern (hourly buckets)
	hour := time.Now().Hour()
	stats.HourlyAccess[hour]++

	// Update transfer statistics
	switch accessType {
	case "read":
		stats.TotalReads++
		stats.BytesRead += bytesTransferred
	case "write":
		stats.TotalWrites++
		stats.BytesWritten += bytesTransferred
	}
}

// MoveVolumeToTier moves a volume to a different storage tier
func (tm *TieringManager) MoveVolumeToTier(ctx context.Context, volumeID string, targetTier StorageTier) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	stats, exists := tm.accessStats[volumeID]
	if !exists {
		return fmt.Errorf("volume %s not found in access stats", volumeID)
	}

	if stats.CurrentTier == targetTier {
		return nil // Already in target tier
	}

	// Get tier configurations
	sourceTierConfig, exists := tm.tiers[stats.CurrentTier]
	if !exists {
		return fmt.Errorf("source tier %s not configured", stats.CurrentTier)
	}

	_, exists = tm.tiers[targetTier]
	if !exists {
		return fmt.Errorf("target tier %s not configured", targetTier)
	}

	// Check minimum storage duration
	timeInCurrentTier := time.Since(stats.TierChangeTime)
	if timeInCurrentTier < sourceTierConfig.MinStorageDuration {
		return fmt.Errorf("volume %s must stay in tier %s for %v (current: %v)",
			volumeID, stats.CurrentTier, sourceTierConfig.MinStorageDuration, timeInCurrentTier)
	}

	// Get source and target drivers
	sourceDriver := tm.drivers[stats.CurrentTier]
	targetDriver := tm.drivers[targetTier]

	// Get volume info
	volumeInfo, err := sourceDriver.GetVolumeInfo(ctx, volumeID)
	if err != nil {
		return fmt.Errorf("failed to get volume info: %w", err)
	}

	log.Printf("Moving volume %s from tier %s to tier %s (size: %d bytes)",
		volumeID, stats.CurrentTier, targetTier, volumeInfo.Size)

	// Create volume in target tier
	if err := targetDriver.CreateVolume(ctx, volumeID, volumeInfo.Size); err != nil {
		return fmt.Errorf("failed to create volume in target tier: %w", err)
	}

	// Copy data from source to target
	if err := tm.copyVolumeData(ctx, volumeID, sourceDriver, targetDriver, volumeInfo.Size); err != nil {
		// Clean up target volume on failure
		targetDriver.DeleteVolume(ctx, volumeID)
		return fmt.Errorf("failed to copy volume data: %w", err)
	}

	// Delete volume from source tier
	if err := sourceDriver.DeleteVolume(ctx, volumeID); err != nil {
		log.Printf("Warning: failed to delete volume from source tier: %v", err)
	}

	// Update access statistics
	stats.CurrentTier = targetTier
	stats.TierChangeTime = time.Now()

	log.Printf("Successfully moved volume %s to tier %s", volumeID, targetTier)
	return nil
}

// copyVolumeData copies data between storage drivers
func (tm *TieringManager) copyVolumeData(ctx context.Context, volumeID string, sourceDriver, targetDriver StorageDriver, sizeBytes int64) error {
	const chunkSize = 1024 * 1024 // 1MB chunks

	var offset int64
	for offset < sizeBytes {
		// Calculate chunk size for this iteration
		currentChunkSize := int(chunkSize)
		if offset+int64(currentChunkSize) > sizeBytes {
			currentChunkSize = int(sizeBytes - offset)
		}

		// Read from source
		data, err := sourceDriver.ReadVolume(ctx, volumeID, offset, currentChunkSize)
		if err != nil {
			return fmt.Errorf("failed to read from source at offset %d: %w", offset, err)
		}

		// Write to target
		if err := targetDriver.WriteVolume(ctx, volumeID, offset, data); err != nil {
			return fmt.Errorf("failed to write to target at offset %d: %w", offset, err)
		}

		offset += int64(len(data))

		// Check for cancellation
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	return nil
}

// evaluationLoop runs the tiering evaluation periodically
func (tm *TieringManager) evaluationLoop() {
	ticker := time.NewTicker(tm.evaluationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-tm.ctx.Done():
			return
		case <-ticker.C:
			tm.evaluateAndMoveVolumes()
		}
	}
}

// evaluateAndMoveVolumes evaluates all volumes and moves them if needed
func (tm *TieringManager) evaluateAndMoveVolumes() {
	tm.mu.RLock()
	volumeIDs := make([]string, 0, len(tm.accessStats))
	for volumeID := range tm.accessStats {
		volumeIDs = append(volumeIDs, volumeID)
	}
	tm.mu.RUnlock()

	for _, volumeID := range volumeIDs {
		tm.evaluateVolumeForTiering(volumeID)
	}
}

// evaluateVolumeForTiering evaluates a single volume for tiering
func (tm *TieringManager) evaluateVolumeForTiering(volumeID string) {
	tm.mu.RLock()
	stats, exists := tm.accessStats[volumeID]
	if !exists {
		tm.mu.RUnlock()
		return
	}

	currentTier := stats.CurrentTier
	currentTierConfig := tm.tiers[currentTier]
	tm.mu.RUnlock()

	// Calculate access frequency
	accessFrequency := tm.calculateAccessFrequency(stats)

	// Evaluate for promotion
	if targetTier := tm.shouldPromote(stats, accessFrequency); targetTier != "" {
		log.Printf("Volume %s should be promoted from %s to %s (frequency: %.2f)",
			volumeID, currentTier, targetTier, accessFrequency)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		if err := tm.MoveVolumeToTier(ctx, volumeID, targetTier); err != nil {
			log.Printf("Failed to promote volume %s: %v", volumeID, err)
		}
		cancel()
		return
	}

	// Evaluate for demotion
	if targetTier := tm.shouldDemote(stats, currentTierConfig); targetTier != "" {
		log.Printf("Volume %s should be demoted from %s to %s (inactive for %v)",
			volumeID, currentTier, targetTier, time.Since(stats.LastAccessed))

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		if err := tm.MoveVolumeToTier(ctx, volumeID, targetTier); err != nil {
			log.Printf("Failed to demote volume %s: %v", volumeID, err)
		}
		cancel()
	}
}

// calculateAccessFrequency calculates the access frequency for a volume
func (tm *TieringManager) calculateAccessFrequency(stats *VolumeAccessStats) float64 {
	// Calculate accesses per hour over the last 24 hours
	totalAccesses := 0
	for _, count := range stats.HourlyAccess {
		totalAccesses += count
	}

	return float64(totalAccesses) / 24.0
}

// shouldPromote determines if a volume should be promoted to a higher tier
func (tm *TieringManager) shouldPromote(stats *VolumeAccessStats, accessFrequency float64) StorageTier {
	currentTier := stats.CurrentTier
	currentTierConfig := tm.tiers[currentTier]

	if !currentTierConfig.PromotionRules.Enabled {
		return ""
	}

	// Check access frequency threshold
	if accessFrequency < currentTierConfig.PromotionRules.AccessFrequencyThreshold {
		return ""
	}

	// Check access count threshold
	if stats.AccessCount < currentTierConfig.PromotionRules.AccessCountThreshold {
		return ""
	}

	// Find the next higher tier
	tierOrder := []StorageTier{TierArchive, TierCold, TierWarm, TierHot}
	for i, tier := range tierOrder {
		if tier == currentTier && i > 0 {
			return tierOrder[i-1]
		}
	}

	return ""
}

// shouldDemote determines if a volume should be demoted to a lower tier
func (tm *TieringManager) shouldDemote(stats *VolumeAccessStats, tierConfig TierConfig) StorageTier {
	if !tierConfig.DemotionRules.Enabled {
		return ""
	}

	// Check inactivity threshold
	inactiveDuration := time.Since(stats.LastAccessed)
	if inactiveDuration < tierConfig.DemotionRules.InactivityThreshold {
		return ""
	}

	// Check minimum storage duration
	timeInCurrentTier := time.Since(stats.TierChangeTime)
	if timeInCurrentTier < tierConfig.MinStorageDuration {
		return ""
	}

	// Find the next lower tier
	tierOrder := []StorageTier{TierHot, TierWarm, TierCold, TierArchive}
	for i, tier := range tierOrder {
		if tier == stats.CurrentTier && i < len(tierOrder)-1 {
			return tierOrder[i+1]
		}
	}

	return ""
}

// trackAccessPatterns periodically updates access patterns
func (tm *TieringManager) trackAccessPatterns() {
	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-tm.ctx.Done():
			return
		case <-ticker.C:
			tm.updateAccessPatterns()
		}
	}
}

// updateAccessPatterns updates the hourly access patterns
func (tm *TieringManager) updateAccessPatterns() {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	currentHour := time.Now().Hour()

	// Shift hourly access patterns (rotate the array)
	for _, stats := range tm.accessStats {
		// Move all elements one position to the left
		copy(stats.HourlyAccess[0:], stats.HourlyAccess[1:])
		// Reset the current hour bucket
		stats.HourlyAccess[currentHour] = 0
	}
}

// GetVolumeStats returns statistics for a volume
func (tm *TieringManager) GetVolumeStats(volumeID string) (*VolumeAccessStats, error) {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	stats, exists := tm.accessStats[volumeID]
	if !exists {
		return nil, fmt.Errorf("volume %s not found", volumeID)
	}

	// Return a copy to prevent external modification
	statsCopy := *stats
	return &statsCopy, nil
}

// GetTierStats returns statistics for all tiers
func (tm *TieringManager) GetTierStats() map[StorageTier]TierStats {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	tierStats := make(map[StorageTier]TierStats)

	// Initialize stats for all tiers
	for tier := range tm.tiers {
		tierStats[tier] = TierStats{
			Tier:        tier,
			VolumeCount: 0,
			TotalSize:   0,
		}
	}

	// Count volumes per tier
	for _, stats := range tm.accessStats {
		tierStat := tierStats[stats.CurrentTier]
		tierStat.VolumeCount++
		// Note: We don't have easy access to volume size here
		// In a real implementation, this would query the volume info
		tierStats[stats.CurrentTier] = tierStat
	}

	return tierStats
}

// TierStats contains statistics for a storage tier
type TierStats struct {
	Tier        StorageTier `json:"tier"`
	VolumeCount int         `json:"volume_count"`
	TotalSize   int64       `json:"total_size"`
	TotalCost   float64     `json:"total_cost"`
}

// CalculateTierCosts calculates the monthly cost for each tier
func (tm *TieringManager) CalculateTierCosts() map[StorageTier]float64 {
	tierStats := tm.GetTierStats()
	tierCosts := make(map[StorageTier]float64)

	for tier, stats := range tierStats {
		tierConfig := tm.tiers[tier]
		monthlyCost := float64(stats.TotalSize) / (1024 * 1024 * 1024) * tierConfig.CostPerGBMonth
		tierCosts[tier] = monthlyCost
	}

	return tierCosts
}