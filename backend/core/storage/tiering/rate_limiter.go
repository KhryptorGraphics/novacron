package tiering

import (
	"context"
	"sync"
	"time"
)

// RateLimiter provides rate limiting for storage tier migrations
type RateLimiter struct {
	// Maximum bytes per second for migrations
	maxBytesPerSecond int64
	// Maximum concurrent migrations
	maxConcurrent int
	// Current bytes transferred in the current window
	currentBytes int64
	// Window start time
	windowStart time.Time
	// Window duration
	windowDuration time.Duration
	// Semaphore for concurrent operations
	semaphore chan struct{}
	// Mutex for thread safety
	mu sync.Mutex
	// Metrics
	totalBytesTransferred int64
	totalMigrations      int64
	throttledCount       int64
}

// MigrationRateLimiter is the global rate limiter for migrations
type MigrationRateLimiter struct {
	// Per-tier rate limiters
	tierLimiters map[TierLevel]*RateLimiter
	// Global rate limiter
	globalLimiter *RateLimiter
	// Configuration
	config RateLimiterConfig
	mu     sync.RWMutex
}

// RateLimiterConfig defines rate limiting configuration
type RateLimiterConfig struct {
	// Global limits
	GlobalMaxBytesPerSecond int64
	GlobalMaxConcurrent     int
	// Per-tier limits
	HotTierMaxBytesPerSecond  int64
	WarmTierMaxBytesPerSecond int64
	ColdTierMaxBytesPerSecond int64
	// Adaptive throttling
	EnableAdaptiveThrottling bool
	// Minimum bytes per second (won't go below this)
	MinBytesPerSecond int64
	// Maximum bytes per second (won't go above this)
	MaxBytesPerSecond int64
	// CPU threshold for throttling (0-100)
	CPUThrottleThreshold float64
	// Memory threshold for throttling (0-100)
	MemoryThrottleThreshold float64
	// Network bandwidth threshold (0-100)
	NetworkThrottleThreshold float64
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(maxBytesPerSecond int64, maxConcurrent int) *RateLimiter {
	return &RateLimiter{
		maxBytesPerSecond: maxBytesPerSecond,
		maxConcurrent:    maxConcurrent,
		windowStart:      time.Now(),
		windowDuration:   time.Second,
		semaphore:        make(chan struct{}, maxConcurrent),
	}
}

// NewMigrationRateLimiter creates a new migration rate limiter with default config
func NewMigrationRateLimiter() *MigrationRateLimiter {
	config := RateLimiterConfig{
		GlobalMaxBytesPerSecond:    100 * 1024 * 1024, // 100 MB/s global
		GlobalMaxConcurrent:        5,
		HotTierMaxBytesPerSecond:   50 * 1024 * 1024,  // 50 MB/s for hot tier
		WarmTierMaxBytesPerSecond:  30 * 1024 * 1024,  // 30 MB/s for warm tier
		ColdTierMaxBytesPerSecond:  20 * 1024 * 1024,  // 20 MB/s for cold tier
		EnableAdaptiveThrottling:   true,
		MinBytesPerSecond:         1 * 1024 * 1024,   // 1 MB/s minimum
		MaxBytesPerSecond:         200 * 1024 * 1024, // 200 MB/s maximum
		CPUThrottleThreshold:      80.0,
		MemoryThrottleThreshold:   85.0,
		NetworkThrottleThreshold:  90.0,
	}

	return NewMigrationRateLimiterWithConfig(config)
}

// NewMigrationRateLimiterWithConfig creates a rate limiter with custom config
func NewMigrationRateLimiterWithConfig(config RateLimiterConfig) *MigrationRateLimiter {
	mrl := &MigrationRateLimiter{
		tierLimiters:  make(map[TierLevel]*RateLimiter),
		globalLimiter: NewRateLimiter(config.GlobalMaxBytesPerSecond, config.GlobalMaxConcurrent),
		config:        config,
	}

	// Create per-tier rate limiters
	mrl.tierLimiters[TierHot] = NewRateLimiter(config.HotTierMaxBytesPerSecond, 2)
	mrl.tierLimiters[TierWarm] = NewRateLimiter(config.WarmTierMaxBytesPerSecond, 2)
	mrl.tierLimiters[TierCold] = NewRateLimiter(config.ColdTierMaxBytesPerSecond, 1)

	return mrl
}

// AcquirePermit acquires a permit to perform a migration
func (rl *RateLimiter) AcquirePermit(ctx context.Context) error {
	select {
	case rl.semaphore <- struct{}{}:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ReleasePermit releases a migration permit
func (rl *RateLimiter) ReleasePermit() {
	select {
	case <-rl.semaphore:
	default:
	}
}

// WaitForQuota waits until the specified number of bytes can be transferred
func (rl *RateLimiter) WaitForQuota(ctx context.Context, bytes int64) error {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	for {
		// Check if we need to reset the window
		now := time.Now()
		if now.Sub(rl.windowStart) >= rl.windowDuration {
			rl.currentBytes = 0
			rl.windowStart = now
		}

		// Check if we have quota
		if rl.currentBytes+bytes <= rl.maxBytesPerSecond {
			rl.currentBytes += bytes
			rl.totalBytesTransferred += bytes
			return nil
		}

		// Calculate wait time
		remainingWindow := rl.windowDuration - now.Sub(rl.windowStart)
		if remainingWindow <= 0 {
			continue // Window will reset on next iteration
		}

		rl.throttledCount++
		rl.mu.Unlock()

		// Wait for the window to reset or context cancellation
		select {
		case <-time.After(remainingWindow):
			rl.mu.Lock()
			continue
		case <-ctx.Done():
			rl.mu.Lock()
			return ctx.Err()
		}
	}
}

// StartMigration begins a rate-limited migration
func (mrl *MigrationRateLimiter) StartMigration(ctx context.Context, sourceTier, targetTier TierLevel) (*MigrationToken, error) {
	mrl.mu.RLock()
	defer mrl.mu.RUnlock()

	// Acquire global permit
	if err := mrl.globalLimiter.AcquirePermit(ctx); err != nil {
		return nil, err
	}

	// Acquire source tier permit
	if limiter, exists := mrl.tierLimiters[sourceTier]; exists {
		if err := limiter.AcquirePermit(ctx); err != nil {
			mrl.globalLimiter.ReleasePermit()
			return nil, err
		}
	}

	// Acquire target tier permit
	if limiter, exists := mrl.tierLimiters[targetTier]; exists {
		if err := limiter.AcquirePermit(ctx); err != nil {
			mrl.globalLimiter.ReleasePermit()
			if sourceLimiter, exists := mrl.tierLimiters[sourceTier]; exists {
				sourceLimiter.ReleasePermit()
			}
			return nil, err
		}
	}

	return &MigrationToken{
		sourceTier: sourceTier,
		targetTier: targetTier,
		startTime:  time.Now(),
		limiter:    mrl,
	}, nil
}

// MigrationToken represents an active migration session
type MigrationToken struct {
	sourceTier    TierLevel
	targetTier    TierLevel
	startTime     time.Time
	bytesTransferred int64
	limiter       *MigrationRateLimiter
}

// TransferBytes transfers bytes with rate limiting
func (mt *MigrationToken) TransferBytes(ctx context.Context, bytes int64) error {
	// Apply rate limiting at multiple levels
	
	// Global rate limiting
	if err := mt.limiter.globalLimiter.WaitForQuota(ctx, bytes); err != nil {
		return err
	}

	// Source tier rate limiting
	if limiter, exists := mt.limiter.tierLimiters[mt.sourceTier]; exists {
		if err := limiter.WaitForQuota(ctx, bytes); err != nil {
			return err
		}
	}

	// Target tier rate limiting
	if limiter, exists := mt.limiter.tierLimiters[mt.targetTier]; exists {
		if err := limiter.WaitForQuota(ctx, bytes); err != nil {
			return err
		}
	}

	mt.bytesTransferred += bytes
	return nil
}

// Complete completes the migration and releases resources
func (mt *MigrationToken) Complete() {
	mt.limiter.globalLimiter.ReleasePermit()
	
	if limiter, exists := mt.limiter.tierLimiters[mt.sourceTier]; exists {
		limiter.ReleasePermit()
		limiter.totalMigrations++
	}
	
	if limiter, exists := mt.limiter.tierLimiters[mt.targetTier]; exists {
		limiter.ReleasePermit()
	}
}

// AdaptiveThrottle adjusts rate limits based on system load
func (mrl *MigrationRateLimiter) AdaptiveThrottle(systemLoad SystemLoadInfo) {
	if !mrl.config.EnableAdaptiveThrottling {
		return
	}

	mrl.mu.Lock()
	defer mrl.mu.Unlock()

	// Calculate throttle factor based on system load
	throttleFactor := 1.0

	if systemLoad.CPUUsage > mrl.config.CPUThrottleThreshold {
		throttleFactor *= (100 - systemLoad.CPUUsage) / 20 // Reduce more aggressively as CPU approaches 100%
	}

	if systemLoad.MemoryUsage > mrl.config.MemoryThrottleThreshold {
		throttleFactor *= (100 - systemLoad.MemoryUsage) / 15
	}

	if systemLoad.NetworkBandwidth > mrl.config.NetworkThrottleThreshold {
		throttleFactor *= (100 - systemLoad.NetworkBandwidth) / 10
	}

	// Apply throttle factor to rate limits
	newRate := int64(float64(mrl.config.GlobalMaxBytesPerSecond) * throttleFactor)
	
	// Enforce min/max bounds
	if newRate < mrl.config.MinBytesPerSecond {
		newRate = mrl.config.MinBytesPerSecond
	} else if newRate > mrl.config.MaxBytesPerSecond {
		newRate = mrl.config.MaxBytesPerSecond
	}

	mrl.globalLimiter.maxBytesPerSecond = newRate

	// Adjust tier limits proportionally
	for tier, limiter := range mrl.tierLimiters {
		var baseRate int64
		switch tier {
		case TierHot:
			baseRate = mrl.config.HotTierMaxBytesPerSecond
		case TierWarm:
			baseRate = mrl.config.WarmTierMaxBytesPerSecond
		case TierCold:
			baseRate = mrl.config.ColdTierMaxBytesPerSecond
		}

		tierRate := int64(float64(baseRate) * throttleFactor)
		if tierRate < mrl.config.MinBytesPerSecond/3 {
			tierRate = mrl.config.MinBytesPerSecond / 3
		}
		
		limiter.maxBytesPerSecond = tierRate
	}
}

// GetMetrics returns rate limiter metrics
func (mrl *MigrationRateLimiter) GetMetrics() map[string]interface{} {
	mrl.mu.RLock()
	defer mrl.mu.RUnlock()

	metrics := make(map[string]interface{})
	
	// Global metrics
	metrics["global_bytes_transferred"] = mrl.globalLimiter.totalBytesTransferred
	metrics["global_throttled_count"] = mrl.globalLimiter.throttledCount
	metrics["global_current_rate"] = mrl.globalLimiter.maxBytesPerSecond
	
	// Per-tier metrics
	tierMetrics := make(map[string]map[string]interface{})
	for tier, limiter := range mrl.tierLimiters {
		tierName := ""
		switch tier {
		case TierHot:
			tierName = "hot"
		case TierWarm:
			tierName = "warm"
		case TierCold:
			tierName = "cold"
		}
		
		tierMetrics[tierName] = map[string]interface{}{
			"bytes_transferred": limiter.totalBytesTransferred,
			"migrations":       limiter.totalMigrations,
			"throttled_count":  limiter.throttledCount,
			"current_rate":     limiter.maxBytesPerSecond,
		}
	}
	metrics["tiers"] = tierMetrics
	
	return metrics
}

// PriorityMigration allows high-priority migrations to bypass rate limits temporarily
func (mrl *MigrationRateLimiter) PriorityMigration(ctx context.Context, sourceTier, targetTier TierLevel, priorityMultiplier float64) (*MigrationToken, error) {
	mrl.mu.Lock()
	
	// Temporarily increase rate limits for priority migration
	originalGlobalRate := mrl.globalLimiter.maxBytesPerSecond
	mrl.globalLimiter.maxBytesPerSecond = int64(float64(originalGlobalRate) * priorityMultiplier)
	
	if limiter, exists := mrl.tierLimiters[sourceTier]; exists {
		originalRate := limiter.maxBytesPerSecond
		limiter.maxBytesPerSecond = int64(float64(originalRate) * priorityMultiplier)
		defer func() {
			limiter.maxBytesPerSecond = originalRate
		}()
	}
	
	if limiter, exists := mrl.tierLimiters[targetTier]; exists {
		originalRate := limiter.maxBytesPerSecond
		limiter.maxBytesPerSecond = int64(float64(originalRate) * priorityMultiplier)
		defer func() {
			limiter.maxBytesPerSecond = originalRate
		}()
	}
	
	mrl.mu.Unlock()
	
	// Start the migration with increased limits
	token, err := mrl.StartMigration(ctx, sourceTier, targetTier)
	
	// Restore original rate
	mrl.mu.Lock()
	mrl.globalLimiter.maxBytesPerSecond = originalGlobalRate
	mrl.mu.Unlock()
	
	return token, err
}