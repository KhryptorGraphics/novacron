package tiering

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// TierLevel represents the performance tier of storage
type TierLevel int

const (
	// TierHot is the highest performance tier (SSD, in-memory)
	TierHot TierLevel = iota
	// TierWarm is the medium performance tier (fast HDDs, network storage)
	TierWarm
	// TierCold is the lowest performance tier (archive, object storage)
	TierCold
)

// StorageTier represents a specific storage tier with its driver
type StorageTier struct {
	// Level of the tier (hot, warm, cold)
	Level TierLevel
	// Driver for this tier
	Driver storage.StorageDriver
	// Name of this tier
	Name string
	// Cost per GB per month (for cost optimization)
	CostPerGBMonth float64
	// Maximum capacity in GB (0 for unlimited)
	MaxCapacityGB int64
	// Current usage in GB
	CurrentUsageGB int64
}

// TierManager handles the automatic tiering between storage systems
type TierManager struct {
	// Map of tier levels to storage tiers
	tiers map[TierLevel]*StorageTier
	// Usage statistics by volume
	volumeUsage map[string]*VolumeStats
	// Mutex for protecting concurrent access
	mu sync.RWMutex
	// Tiering policies
	policies []TieringPolicy
	// Is the manager initialized
	initialized bool
	// Background worker for automatic tiering
	backgroundWorker bool
	// Metadata store for tracking volume locations
	metadataStore *TierMetadataStore
	// Context for background worker
	ctx    context.Context
	cancel context.CancelFunc
}

// VolumeStats tracks usage statistics for a volume
type VolumeStats struct {
	// Name of the volume
	Name string
	// Current tier where the volume is stored
	CurrentTier TierLevel
	// Last accessed time
	LastAccessed time.Time
	// Access frequency (accesses per day)
	AccessFrequency float64
	// Size in GB
	SizeGB float64
	// Is this volume pinned to a specific tier
	Pinned bool
	// If pinned, which tier
	PinnedTier TierLevel
}

// TieringPolicy defines when to move volumes between tiers
type TieringPolicy struct {
	// Name of the policy
	Name string
	// Function that determines if a volume should be moved
	EvaluateFunc func(stats *VolumeStats) (shouldMove bool, targetTier TierLevel)
	// Priority of this policy (higher number = higher priority)
	Priority int
}

// TierMetadataStore manages the metadata for tiered volumes
type TierMetadataStore struct {
	// Map of volume name to metadata
	volumeMetadata map[string]*VolumeMetadata
	// Mutex for protecting concurrent access
	mu sync.RWMutex
}

// VolumeMetadata contains metadata about a volume in the tiering system
type VolumeMetadata struct {
	// Name of the volume
	Name string
	// Current tier where the volume is stored
	CurrentTier TierLevel
	// Map of tier to volume name in that tier's driver
	TierNames map[TierLevel]string
	// Last tiering operation time
	LastTiered time.Time
	// Last time this volume was accessed
	LastAccessed time.Time
	// Number of accesses
	AccessCount int64
	// Custom metadata
	Custom map[string]string
}

// NewTierManager creates a new storage tier manager
func NewTierManager() *TierManager {
	ctx, cancel := context.WithCancel(context.Background())
	return &TierManager{
		tiers:            make(map[TierLevel]*StorageTier),
		volumeUsage:      make(map[string]*VolumeStats),
		policies:         make([]TieringPolicy, 0),
		initialized:      false,
		backgroundWorker: false,
		metadataStore: &TierMetadataStore{
			volumeMetadata: make(map[string]*VolumeMetadata),
		},
		ctx:    ctx,
		cancel: cancel,
	}
}

// AddTier adds a storage tier to the tier manager
func (tm *TierManager) AddTier(level TierLevel, driver storage.StorageDriver, name string, costPerGB float64, maxCapacityGB int64) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if _, exists := tm.tiers[level]; exists {
		return fmt.Errorf("tier level %d already exists", level)
	}

	tm.tiers[level] = &StorageTier{
		Level:          level,
		Driver:         driver,
		Name:           name,
		CostPerGBMonth: costPerGB,
		MaxCapacityGB:  maxCapacityGB,
		CurrentUsageGB: 0,
	}

	return nil
}

// AddPolicy adds a tiering policy
func (tm *TierManager) AddPolicy(name string, evaluateFunc func(stats *VolumeStats) (bool, TierLevel), priority int) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	tm.policies = append(tm.policies, TieringPolicy{
		Name:         name,
		EvaluateFunc: evaluateFunc,
		Priority:     priority,
	})

	// Sort policies by priority (higher first)
	for i := 0; i < len(tm.policies)-1; i++ {
		for j := i + 1; j < len(tm.policies); j++ {
			if tm.policies[i].Priority < tm.policies[j].Priority {
				tm.policies[i], tm.policies[j] = tm.policies[j], tm.policies[i]
			}
		}
	}
}

// Initialize initializes the tier manager
func (tm *TierManager) Initialize() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tm.initialized {
		return fmt.Errorf("tier manager already initialized")
	}

	// Verify we have at least one tier
	if len(tm.tiers) == 0 {
		return fmt.Errorf("no storage tiers configured")
	}

	// Initialize each driver if not already initialized
	for _, tier := range tm.tiers {
		// Check if driver implements the StorageDriver interface properly
		if tier.Driver == nil {
			return fmt.Errorf("tier %s has nil driver", tier.Name)
		}

		// Try to initialize the driver, but don't fail if it's already initialized
		if err := tier.Driver.Initialize(); err != nil && !strings.Contains(err.Error(), "already initialized") {
			return fmt.Errorf("failed to initialize tier %s: %v", tier.Name, err)
		}
	}

	tm.initialized = true
	return nil
}

// StartBackgroundWorker starts a background worker for automated tiering
func (tm *TierManager) StartBackgroundWorker(interval time.Duration) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if !tm.initialized {
		return fmt.Errorf("tier manager not initialized")
	}

	if tm.backgroundWorker {
		return fmt.Errorf("background worker already running")
	}

	go func() {
		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ticker.C:
				if err := tm.EvaluateAndApplyTieringPolicies(); err != nil {
					log.Printf("Error in tiering policy evaluation: %v", err)
				}
			case <-tm.ctx.Done():
				return
			}
		}
	}()

	tm.backgroundWorker = true
	return nil
}

// StopBackgroundWorker stops the background worker
func (tm *TierManager) StopBackgroundWorker() {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if tm.backgroundWorker {
		tm.cancel()
		tm.backgroundWorker = false
	}
}

// RecordVolumeAccess records an access to a volume for statistics
func (tm *TierManager) RecordVolumeAccess(volumeName string) {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	// Update volume usage statistics
	stats, exists := tm.volumeUsage[volumeName]
	if !exists {
		// Lookup where the volume is currently stored
		metadata, exists := tm.metadataStore.volumeMetadata[volumeName]
		currentTier := TierHot // Default to hot tier if unknown
		if exists {
			currentTier = metadata.CurrentTier
		}

		stats = &VolumeStats{
			Name:            volumeName,
			CurrentTier:     currentTier,
			LastAccessed:    time.Now(),
			AccessFrequency: 1.0, // Start with 1 access per day
			SizeGB:          0,   // Will be updated later
			Pinned:          false,
		}
		tm.volumeUsage[volumeName] = stats
	} else {
		// Update existing stats
		now := time.Now()
		daysSinceLastAccess := now.Sub(stats.LastAccessed).Hours() / 24.0
		if daysSinceLastAccess > 0 {
			// Weighted calculation of access frequency
			stats.AccessFrequency = (stats.AccessFrequency*0.7 + 1.0/daysSinceLastAccess*0.3)
		}
		stats.LastAccessed = now
	}

	// Update metadata store
	if metadata, exists := tm.metadataStore.volumeMetadata[volumeName]; exists {
		metadata.LastAccessed = time.Now()
		metadata.AccessCount++
	}
}

// PinVolume pins a volume to a specific tier
func (tm *TierManager) PinVolume(volumeName string, tier TierLevel) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	stats, exists := tm.volumeUsage[volumeName]
	if !exists {
		return fmt.Errorf("volume %s not found in usage statistics", volumeName)
	}

	stats.Pinned = true
	stats.PinnedTier = tier

	// If the volume is not already in the target tier, move it
	if stats.CurrentTier != tier {
		if err := tm.moveVolumeBetweenTiers(volumeName, stats.CurrentTier, tier); err != nil {
			return fmt.Errorf("failed to move pinned volume to target tier: %v", err)
		}
		stats.CurrentTier = tier
	}

	return nil
}

// UnpinVolume removes a pin from a volume
func (tm *TierManager) UnpinVolume(volumeName string) error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	stats, exists := tm.volumeUsage[volumeName]
	if !exists {
		return fmt.Errorf("volume %s not found in usage statistics", volumeName)
	}

	stats.Pinned = false
	return nil
}

// EvaluateAndApplyTieringPolicies evaluates tiering policies and moves volumes as needed
func (tm *TierManager) EvaluateAndApplyTieringPolicies() error {
	tm.mu.Lock()
	defer tm.mu.Unlock()

	if !tm.initialized {
		return fmt.Errorf("tier manager not initialized")
	}

	// First, update our knowledge of all volumes
	if err := tm.refreshVolumeInfo(); err != nil {
		return fmt.Errorf("failed to refresh volume info: %v", err)
	}

	// Evaluate each volume against policies
	for volumeName, stats := range tm.volumeUsage {
		// Skip pinned volumes
		if stats.Pinned {
			continue
		}

		// Apply policies in order of priority
		for _, policy := range tm.policies {
			shouldMove, targetTier := policy.EvaluateFunc(stats)
			if shouldMove && targetTier != stats.CurrentTier {
				log.Printf("Policy %s suggests moving volume %s from tier %d to tier %d",
					policy.Name, volumeName, stats.CurrentTier, targetTier)

				if err := tm.moveVolumeBetweenTiers(volumeName, stats.CurrentTier, targetTier); err != nil {
					log.Printf("Failed to move volume %s to tier %d: %v", volumeName, targetTier, err)
					continue
				}

				// Update stats
				stats.CurrentTier = targetTier

				// We applied one policy, don't evaluate the rest
				break
			}
		}
	}

	return nil
}

// refreshVolumeInfo updates our information about all volumes
func (tm *TierManager) refreshVolumeInfo() error {
	// Get volumes from all tiers
	for level, tier := range tm.tiers {
		volumes, err := tier.Driver.ListVolumes(tm.ctx)
		if err != nil {
			return fmt.Errorf("failed to list volumes in tier %s: %v", tier.Name, err)
		}

		// Update our knowledge of these volumes
		for _, volumeName := range volumes {
			// Get volume info
			info, err := tier.Driver.GetVolumeInfo(tm.ctx, volumeName)
			if err != nil {
				log.Printf("Failed to get info for volume %s in tier %s: %v", volumeName, tier.Name, err)
				continue
			}

			// Update or create volume stats
			stats, exists := tm.volumeUsage[volumeName]
			if !exists {
				stats = &VolumeStats{
					Name:            volumeName,
					CurrentTier:     level,
					LastAccessed:    time.Now(), // Assume recent access if we're just discovering it
					AccessFrequency: 0.1,        // Assume relatively low frequency initially
					SizeGB:          0,
					Pinned:          false,
				}
				tm.volumeUsage[volumeName] = stats
			}

			// Update size from VolumeInfo
			stats.SizeGB = float64(info.Size) / (1024 * 1024 * 1024) // Convert bytes to GB

			// Update metadata store
			metadata, exists := tm.metadataStore.volumeMetadata[volumeName]
			if !exists {
				metadata = &VolumeMetadata{
					Name:        volumeName,
					CurrentTier: level,
					TierNames:   make(map[TierLevel]string),
					LastTiered:  time.Now(),
					Custom:      make(map[string]string),
				}
				tm.metadataStore.volumeMetadata[volumeName] = metadata
			}

			// Store the volume name in this tier
			metadata.TierNames[level] = volumeName

			// If this is the current tier we know about, update it
			if level == stats.CurrentTier {
				metadata.CurrentTier = level
			}
		}
	}

	return nil
}

// moveVolumeBetweenTiers moves a volume from one tier to another
func (tm *TierManager) moveVolumeBetweenTiers(volumeName string, sourceTier, targetTier TierLevel) error {
	// Get the source and target tiers
	source, sourceExists := tm.tiers[sourceTier]
	target, targetExists := tm.tiers[targetTier]

	if !sourceExists {
		return fmt.Errorf("source tier %d does not exist", sourceTier)
	}

	if !targetExists {
		return fmt.Errorf("target tier %d does not exist", targetTier)
	}

	// Get metadata for consistent naming
	metadata, exists := tm.metadataStore.volumeMetadata[volumeName]
	if !exists {
		return fmt.Errorf("volume %s not found in metadata store", volumeName)
	}

	sourceVolumeName := volumeName
	if storedName, exists := metadata.TierNames[sourceTier]; exists {
		sourceVolumeName = storedName
	}

	// Check if the volume already exists in the target tier
	targetVolumeName := volumeName
	if storedName, exists := metadata.TierNames[targetTier]; exists {
		// We already have a copy in the target tier
		targetVolumeName = storedName
	} else {
		// Create a new volume in the target tier
		sizeBytes := int64(tm.volumeUsage[volumeName].SizeGB * 1024 * 1024 * 1024) // Convert GB to bytes
		if err := target.Driver.CreateVolume(tm.ctx, targetVolumeName, sizeBytes); err != nil {
			return fmt.Errorf("failed to create target volume: %v", err)
		}

		// Store the target volume name
		metadata.TierNames[targetTier] = targetVolumeName
	}

	// Copy the data from source to target
	err := tm.copyVolumeData(sourceVolumeName, targetVolumeName, source.Driver, target.Driver)
	if err != nil {
		return fmt.Errorf("failed to copy volume data: %v", err)
	}

	// Update metadata
	metadata.CurrentTier = targetTier
	metadata.LastTiered = time.Now()

	// Update usage stats
	stats, exists := tm.volumeUsage[volumeName]
	if exists {
		stats.CurrentTier = targetTier
	}

	return nil
}

// copyVolumeData copies data from one volume to another, potentially across different drivers
func (tm *TierManager) copyVolumeData(sourceVolume, targetVolume string, sourceDriver, targetDriver storage.StorageDriver) error {
	// Get source volume info to determine size
	sourceInfo, err := sourceDriver.GetVolumeInfo(tm.ctx, sourceVolume)
	if err != nil {
		return fmt.Errorf("failed to get source volume info: %v", err)
	}

	// Read data from source volume in chunks to avoid memory issues
	const chunkSize = 64 * 1024 * 1024 // 64MB chunks
	var offset int64 = 0

	for offset < sourceInfo.Size {
		readSize := chunkSize
		if offset+int64(chunkSize) > sourceInfo.Size {
			readSize = int(sourceInfo.Size - offset)
		}

		// Read chunk from source
		data, err := sourceDriver.ReadVolume(tm.ctx, sourceVolume, offset, readSize)
		if err != nil {
			return fmt.Errorf("failed to read from source volume at offset %d: %v", offset, err)
		}

		// Write chunk to target
		err = targetDriver.WriteVolume(tm.ctx, targetVolume, offset, data)
		if err != nil {
			return fmt.Errorf("failed to write to target volume at offset %d: %v", offset, err)
		}

		offset += int64(len(data))
	}

	return nil
}

// CreateDefaultAgingPolicy creates a policy that moves infrequently accessed data to colder tiers
func (tm *TierManager) CreateDefaultAgingPolicy() {
	tm.AddPolicy("DataAging", func(stats *VolumeStats) (bool, TierLevel) {
		// If accessed frequently (more than once per day), keep in hot tier
		if stats.AccessFrequency > 1.0 {
			return stats.CurrentTier != TierHot, TierHot
		}

		// If accessed occasionally (more than once per week), move to warm tier
		if stats.AccessFrequency > 0.15 { // ~1 access per week
			return stats.CurrentTier != TierWarm, TierWarm
		}

		// If accessed rarely, move to cold tier
		return stats.CurrentTier != TierCold, TierCold
	}, 100) // High priority
}

// CreateCostOptimizationPolicy creates a policy that optimizes for cost
func (tm *TierManager) CreateCostOptimizationPolicy() {
	tm.AddPolicy("CostOptimization", func(stats *VolumeStats) (bool, TierLevel) {
		// For large, infrequently accessed volumes, prefer colder tiers to save money
		if stats.SizeGB > 100 && stats.AccessFrequency < 0.5 {
			// For very large volumes (>1TB) with very infrequent access, use cold tier
			if stats.SizeGB > 1000 && stats.AccessFrequency < 0.05 {
				return stats.CurrentTier != TierCold, TierCold
			}
			// Otherwise use warm tier
			return stats.CurrentTier != TierWarm, TierWarm
		}

		// Small, frequently accessed volumes go to hot tier
		if stats.SizeGB < 50 && stats.AccessFrequency > 2.0 {
			return stats.CurrentTier != TierHot, TierHot
		}

		// No change for everything else
		return false, stats.CurrentTier
	}, 50) // Medium priority
}

// GetTierStats gets statistics for all tiers
func (tm *TierManager) GetTierStats() map[TierLevel]map[string]interface{} {
	tm.mu.RLock()
	defer tm.mu.RUnlock()

	stats := make(map[TierLevel]map[string]interface{})

	for level, tier := range tm.tiers {
		tierStats := make(map[string]interface{})
		tierStats["name"] = tier.Name
		tierStats["max_capacity_gb"] = tier.MaxCapacityGB
		tierStats["current_usage_gb"] = tier.CurrentUsageGB
		tierStats["cost_per_gb_month"] = tier.CostPerGBMonth

		// Count volumes in this tier
		volumeCount := 0
		for _, stats := range tm.volumeUsage {
			if stats.CurrentTier == level {
				volumeCount++
			}
		}
		tierStats["volume_count"] = volumeCount

		stats[level] = tierStats
	}

	return stats
}
