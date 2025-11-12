package federation

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// RegionalBaselineCache caches baseline states for cross-datacenter optimization
// Uses HDE v3 baselines to minimize data transfer between regions
type RegionalBaselineCache struct {
	mu        sync.RWMutex
	logger    *zap.Logger
	baselines map[string]*BaselineCacheEntry
	vmBaselines map[string]string // VM ID -> Baseline ID mapping

	// Configuration
	maxEntries       int
	ttl              time.Duration
	refreshInterval  time.Duration

	// Metrics
	hits             uint64
	misses           uint64
	evictions        uint64
	refreshes        uint64

	// Lifecycle
	ctx              context.Context
	cancel           context.CancelFunc
}

// BaselineCacheEntry represents a cached baseline
type BaselineCacheEntry struct {
	BaselineID   string
	VMID         string
	RegionID     string
	Data         []byte
	Checksum     string
	Timestamp    time.Time
	LastAccessed time.Time
	Size         int
	AccessCount  uint64
	Compressed   bool
	Version      int
}

// NewRegionalBaselineCache creates a new regional baseline cache
func NewRegionalBaselineCache(logger *zap.Logger) *RegionalBaselineCache {
	ctx, cancel := context.WithCancel(context.Background())

	cache := &RegionalBaselineCache{
		logger:          logger,
		baselines:       make(map[string]*BaselineCacheEntry),
		vmBaselines:     make(map[string]string),
		maxEntries:      1000,
		ttl:             5 * time.Minute,
		refreshInterval: 1 * time.Minute,
		ctx:             ctx,
		cancel:          cancel,
	}

	// Start background tasks
	go cache.evictionLoop()
	go cache.refreshLoop()

	return cache
}

// Store stores a baseline in the cache
func (c *RegionalBaselineCache) Store(entry *BaselineCacheEntry) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Check capacity
	if len(c.baselines) >= c.maxEntries {
		if err := c.evictLRU(); err != nil {
			return fmt.Errorf("failed to evict LRU entry: %w", err)
		}
	}

	entry.Timestamp = time.Now()
	entry.LastAccessed = time.Now()
	entry.AccessCount = 0

	c.baselines[entry.BaselineID] = entry
	c.vmBaselines[entry.VMID] = entry.BaselineID

	c.logger.Debug("Stored baseline in cache",
		zap.String("baseline_id", entry.BaselineID),
		zap.String("vm_id", entry.VMID),
		zap.String("region", entry.RegionID),
		zap.Int("size", entry.Size))

	return nil
}

// Get retrieves a baseline from the cache
func (c *RegionalBaselineCache) Get(baselineID string) (*BaselineCacheEntry, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, exists := c.baselines[baselineID]
	if !exists {
		c.misses++
		return nil, false
	}

	// Check if expired
	if time.Since(entry.Timestamp) > c.ttl {
		c.mu.RUnlock()
		c.mu.Lock()
		delete(c.baselines, baselineID)
		delete(c.vmBaselines, entry.VMID)
		c.mu.Unlock()
		c.mu.RLock()
		c.misses++
		return nil, false
	}

	// Update access statistics
	c.mu.RUnlock()
	c.mu.Lock()
	entry.LastAccessed = time.Now()
	entry.AccessCount++
	c.mu.Unlock()
	c.mu.RLock()

	c.hits++

	return entry, true
}

// GetByVMID retrieves a baseline by VM ID
func (c *RegionalBaselineCache) GetByVMID(vmID string) (*BaselineCacheEntry, bool) {
	c.mu.RLock()
	baselineID, exists := c.vmBaselines[vmID]
	c.mu.RUnlock()

	if !exists {
		c.misses++
		return nil, false
	}

	return c.Get(baselineID)
}

// Update updates an existing baseline
func (c *RegionalBaselineCache) Update(baselineID string, data []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.baselines[baselineID]
	if !exists {
		return fmt.Errorf("baseline not found: %s", baselineID)
	}

	entry.Data = data
	entry.Size = len(data)
	entry.Timestamp = time.Now()
	entry.Version++

	c.logger.Debug("Updated baseline in cache",
		zap.String("baseline_id", baselineID),
		zap.Int("new_size", entry.Size),
		zap.Int("version", entry.Version))

	return nil
}

// Delete removes a baseline from the cache
func (c *RegionalBaselineCache) Delete(baselineID string) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if entry, exists := c.baselines[baselineID]; exists {
		delete(c.baselines, baselineID)
		delete(c.vmBaselines, entry.VMID)

		c.logger.Debug("Deleted baseline from cache",
			zap.String("baseline_id", baselineID),
			zap.String("vm_id", entry.VMID))
	}
}

// DeleteByVMID removes all baselines for a VM
func (c *RegionalBaselineCache) DeleteByVMID(vmID string) {
	c.mu.RLock()
	baselineID, exists := c.vmBaselines[vmID]
	c.mu.RUnlock()

	if exists {
		c.Delete(baselineID)
	}
}

// ListByRegion lists all baselines for a region
func (c *RegionalBaselineCache) ListByRegion(regionID string) []*BaselineCacheEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var entries []*BaselineCacheEntry
	for _, entry := range c.baselines {
		if entry.RegionID == regionID {
			entries = append(entries, entry)
		}
	}

	return entries
}

// evictLRU evicts the least recently used entry
func (c *RegionalBaselineCache) evictLRU() error {
	var oldestEntry *BaselineCacheEntry
	var oldestID string

	for id, entry := range c.baselines {
		if oldestEntry == nil || entry.LastAccessed.Before(oldestEntry.LastAccessed) {
			oldestEntry = entry
			oldestID = id
		}
	}

	if oldestEntry != nil {
		delete(c.baselines, oldestID)
		delete(c.vmBaselines, oldestEntry.VMID)
		c.evictions++

		c.logger.Debug("Evicted LRU baseline",
			zap.String("baseline_id", oldestID),
			zap.String("vm_id", oldestEntry.VMID),
			zap.Time("last_accessed", oldestEntry.LastAccessed))

		return nil
	}

	return fmt.Errorf("no entries to evict")
}

// evictionLoop periodically evicts expired entries
func (c *RegionalBaselineCache) evictionLoop() {
	ticker := time.NewTicker(c.refreshInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.evictExpired()
		}
	}
}

// evictExpired removes expired entries
func (c *RegionalBaselineCache) evictExpired() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	var toDelete []string

	for id, entry := range c.baselines {
		if now.Sub(entry.Timestamp) > c.ttl {
			toDelete = append(toDelete, id)
		}
	}

	for _, id := range toDelete {
		if entry, exists := c.baselines[id]; exists {
			delete(c.baselines, id)
			delete(c.vmBaselines, entry.VMID)
			c.evictions++
		}
	}

	if len(toDelete) > 0 {
		c.logger.Debug("Evicted expired baselines",
			zap.Int("count", len(toDelete)))
	}
}

// refreshLoop periodically refreshes baseline statistics
func (c *RegionalBaselineCache) refreshLoop() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			c.logStatistics()
			c.refreshes++
		}
	}
}

// logStatistics logs cache statistics
func (c *RegionalBaselineCache) logStatistics() {
	c.mu.RLock()
	defer c.mu.RUnlock()

	totalEntries := len(c.baselines)
	var totalSize int64
	for _, entry := range c.baselines {
		totalSize += int64(entry.Size)
	}

	hitRate := float64(0)
	total := c.hits + c.misses
	if total > 0 {
		hitRate = float64(c.hits) / float64(total) * 100.0
	}

	c.logger.Info("Regional baseline cache statistics",
		zap.Int("entries", totalEntries),
		zap.Int64("total_size_bytes", totalSize),
		zap.Uint64("hits", c.hits),
		zap.Uint64("misses", c.misses),
		zap.Float64("hit_rate_percent", hitRate),
		zap.Uint64("evictions", c.evictions),
		zap.Uint64("refreshes", c.refreshes))
}

// GetStatistics returns cache statistics
func (c *RegionalBaselineCache) GetStatistics() map[string]interface{} {
	c.mu.RLock()
	defer c.mu.RUnlock()

	totalEntries := len(c.baselines)
	var totalSize int64
	for _, entry := range c.baselines {
		totalSize += int64(entry.Size)
	}

	hitRate := float64(0)
	total := c.hits + c.misses
	if total > 0 {
		hitRate = float64(c.hits) / float64(total) * 100.0
	}

	return map[string]interface{}{
		"total_entries":   totalEntries,
		"total_size":      totalSize,
		"hits":            c.hits,
		"misses":          c.misses,
		"hit_rate":        hitRate,
		"evictions":       c.evictions,
		"refreshes":       c.refreshes,
		"max_entries":     c.maxEntries,
		"ttl_seconds":     c.ttl.Seconds(),
	}
}

// Clear removes all entries from the cache
func (c *RegionalBaselineCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.baselines = make(map[string]*BaselineCacheEntry)
	c.vmBaselines = make(map[string]string)

	c.logger.Info("Cleared regional baseline cache")
}

// Close releases cache resources
func (c *RegionalBaselineCache) Close() error {
	c.logger.Info("Closing regional baseline cache")
	c.cancel()
	c.Clear()
	return nil
}

// SetMaxEntries sets the maximum number of cache entries
func (c *RegionalBaselineCache) SetMaxEntries(max int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.maxEntries = max
}

// SetTTL sets the time-to-live for cache entries
func (c *RegionalBaselineCache) SetTTL(ttl time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ttl = ttl
}

// Size returns the number of cached entries
func (c *RegionalBaselineCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.baselines)
}
