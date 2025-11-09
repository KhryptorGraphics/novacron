package cache

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// HierarchicalCache implements a multi-tier cache hierarchy
type HierarchicalCache struct {
	config *CacheConfig

	// Cache tiers
	l1 *CacheTierImpl
	l2 *CacheTierImpl
	l3 *CacheTierImpl

	// Components
	mlReplacer  MLCacheReplacer
	prefetcher  PrefetchEngine
	dedup       ContentAddressedStorage
	compressor  *CompressionEngine

	// Metrics
	stats     *CacheStats
	statsMu   sync.RWMutex

	// State
	ctx       context.Context
	cancel    context.CancelFunc
	wg        sync.WaitGroup
	closed    bool
	closeMu   sync.Mutex
}

// NewHierarchicalCache creates a new hierarchical cache
func NewHierarchicalCache(config *CacheConfig) (*HierarchicalCache, error) {
	if err := config.Validate(); err != nil {
		return nil, err
	}

	ctx, cancel := context.WithCancel(context.Background())

	hc := &HierarchicalCache{
		config: config,
		ctx:    ctx,
		cancel: cancel,
		stats: &CacheStats{
			TierStats: make(map[CacheTier]*TierStats),
			Timestamp: time.Now(),
		},
	}

	// Initialize tiers
	if config.L1Size > 0 {
		hc.l1 = NewCacheTier(L1, config.L1Size)
	}
	if config.L2Size > 0 {
		hc.l2 = NewCacheTier(L2, config.L2Size)
	}
	if config.L3Size > 0 {
		hc.l3 = NewCacheTier(L3, config.L3Size)
	}

	// Initialize ML replacer
	if config.EvictionPolicy == "ml" {
		hc.mlReplacer = NewMLCacheReplacer(config)
		if err := hc.mlReplacer.LoadModel(config.MLModelPath); err != nil {
			// Model doesn't exist yet, will be trained
		}
	}

	// Initialize prefetcher
	if config.EnablePrefetch {
		hc.prefetcher = NewPrefetchEngine(config, hc)
	}

	// Initialize deduplication
	if config.EnableDedup {
		hc.dedup = NewContentAddressedStorage(config)
	}

	// Initialize compression
	if config.EnableCompression {
		hc.compressor = NewCompressionEngine(config)
	}

	// Start background tasks
	hc.startBackgroundTasks()

	return hc, nil
}

// Get retrieves a value from cache
func (hc *HierarchicalCache) Get(key string) ([]byte, error) {
	if hc.closed {
		return nil, ErrCacheClosed
	}

	start := time.Now()
	defer func() {
		hc.recordLatency("read", time.Since(start))
	}()

	// Try L1
	if hc.l1 != nil {
		if entry, ok := hc.l1.Get(key); ok {
			hc.recordHit(L1)
			hc.updateAccessStats(entry)
			return hc.getValue(entry), nil
		}
	}

	// Try L2
	if hc.l2 != nil {
		if entry, ok := hc.l2.Get(key); ok {
			hc.recordHit(L2)
			hc.updateAccessStats(entry)
			// Promote to L1
			hc.promote(entry, L2, L1)
			return hc.getValue(entry), nil
		}
	}

	// Try L3
	if hc.l3 != nil {
		if entry, ok := hc.l3.Get(key); ok {
			hc.recordHit(L3)
			hc.updateAccessStats(entry)
			// Promote to L2
			hc.promote(entry, L3, L2)
			return hc.getValue(entry), nil
		}
	}

	hc.recordMiss()
	return nil, ErrNotFound
}

// Set stores a value in cache
func (hc *HierarchicalCache) Set(key string, value []byte, ttl time.Duration) error {
	if hc.closed {
		return ErrCacheClosed
	}

	start := time.Now()
	defer func() {
		hc.recordLatency("write", time.Since(start))
	}()

	// Create cache entry
	entry := &CacheEntry{
		Key:            key,
		Value:          value,
		Size:           int64(len(value)),
		CreatedAt:      time.Now(),
		LastAccessedAt: time.Now(),
		ExpiresAt:      time.Now().Add(ttl),
		AccessCount:    1,
		Tier:           L1,
	}

	// Apply deduplication
	if hc.dedup != nil {
		hash, err := hc.dedup.StoreChunk(value)
		if err == nil {
			entry.Hash = hash
			hc.dedup.AddRef(hash)
		}
	}

	// Apply compression
	if hc.compressor != nil {
		compressed, ratio, err := hc.compressor.Compress(value)
		if err == nil && ratio >= hc.config.MinCompressionRatio {
			entry.CompressedValue = compressed
			entry.Compressed = true
			entry.CompressionRatio = ratio
		}
	}

	// Extract ML features
	entry.Features = hc.extractFeatures(entry)

	// Insert into L1
	if hc.l1 != nil {
		// Check if eviction needed
		if hc.l1.NeedsEviction(entry.Size) {
			if err := hc.evict(L1, entry.Size); err != nil {
				return err
			}
		}
		hc.l1.Set(entry)
	} else if hc.l2 != nil {
		// No L1, try L2
		entry.Tier = L2
		if hc.l2.NeedsEviction(entry.Size) {
			if err := hc.evict(L2, entry.Size); err != nil {
				return err
			}
		}
		hc.l2.Set(entry)
	}

	// Trigger prefetch prediction
	if hc.prefetcher != nil && hc.config.EnablePrefetch {
		go hc.triggerPrefetch(key)
	}

	return nil
}

// Delete removes a value from cache
func (hc *HierarchicalCache) Delete(key string) error {
	if hc.closed {
		return ErrCacheClosed
	}

	// Delete from all tiers
	if hc.l1 != nil {
		if entry, ok := hc.l1.Get(key); ok {
			hc.releaseEntry(entry)
			hc.l1.Delete(key)
		}
	}
	if hc.l2 != nil {
		if entry, ok := hc.l2.Get(key); ok {
			hc.releaseEntry(entry)
			hc.l2.Delete(key)
		}
	}
	if hc.l3 != nil {
		if entry, ok := hc.l3.Get(key); ok {
			hc.releaseEntry(entry)
			hc.l3.Delete(key)
		}
	}

	return nil
}

// Exists checks if a key exists in cache
func (hc *HierarchicalCache) Exists(key string) bool {
	if hc.l1 != nil && hc.l1.Exists(key) {
		return true
	}
	if hc.l2 != nil && hc.l2.Exists(key) {
		return true
	}
	if hc.l3 != nil && hc.l3.Exists(key) {
		return true
	}
	return false
}

// promote moves an entry to a higher tier
func (hc *HierarchicalCache) promote(entry *CacheEntry, from, to CacheTier) error {
	var targetTier *CacheTierImpl

	switch to {
	case L1:
		targetTier = hc.l1
	case L2:
		targetTier = hc.l2
	default:
		return nil
	}

	if targetTier == nil {
		return nil
	}

	// Check if eviction needed
	if targetTier.NeedsEviction(entry.Size) {
		if err := hc.evict(to, entry.Size); err != nil {
			return err
		}
	}

	// Copy to target tier
	newEntry := *entry
	newEntry.Tier = to
	targetTier.Set(&newEntry)

	hc.recordPromotion(from, to)

	return nil
}

// demote moves an entry to a lower tier
func (hc *HierarchicalCache) demote(entry *CacheEntry, from, to CacheTier) error {
	var targetTier *CacheTierImpl

	switch to {
	case L2:
		targetTier = hc.l2
	case L3:
		targetTier = hc.l3
	default:
		return nil
	}

	if targetTier == nil {
		return nil
	}

	// Check if eviction needed
	if targetTier.NeedsEviction(entry.Size) {
		if err := hc.evict(to, entry.Size); err != nil {
			return err
		}
	}

	// Copy to target tier
	newEntry := *entry
	newEntry.Tier = to
	targetTier.Set(&newEntry)

	hc.recordDemotion(from, to)

	return nil
}

// evict evicts entries to free space
func (hc *HierarchicalCache) evict(tier CacheTier, bytesNeeded int64) error {
	var cacheTier *CacheTierImpl

	switch tier {
	case L1:
		cacheTier = hc.l1
	case L2:
		cacheTier = hc.l2
	case L3:
		cacheTier = hc.l3
	default:
		return fmt.Errorf("invalid tier: %v", tier)
	}

	if cacheTier == nil {
		return fmt.Errorf("tier %v not available", tier)
	}

	// Use ML-based eviction if available
	if hc.mlReplacer != nil && hc.config.EvictionPolicy == "ml" {
		candidates, err := hc.mlReplacer.FindEvictionCandidates(tier, 10)
		if err == nil && len(candidates) > 0 {
			return hc.evictCandidates(cacheTier, tier, candidates, bytesNeeded)
		}
	}

	// Fallback to LRU
	return hc.evictLRU(cacheTier, tier, bytesNeeded)
}

// evictCandidates evicts specific candidates
func (hc *HierarchicalCache) evictCandidates(cacheTier *CacheTierImpl, tier CacheTier, candidates []*EvictionCandidate, bytesNeeded int64) error {
	var freedBytes int64

	for _, candidate := range candidates {
		if freedBytes >= bytesNeeded {
			break
		}

		if entry, ok := cacheTier.Get(candidate.Key); ok {
			// Try to demote before evicting
			lowerTier := tier + 1
			if lowerTier <= L3 {
				if err := hc.demote(entry, tier, lowerTier); err == nil {
					cacheTier.Delete(candidate.Key)
					freedBytes += entry.Size
					hc.recordEviction(tier)
					continue
				}
			}

			// Evict completely
			hc.releaseEntry(entry)
			cacheTier.Delete(candidate.Key)
			freedBytes += entry.Size
			hc.recordEviction(tier)

			// Learn from eviction
			if hc.mlReplacer != nil {
				hc.mlReplacer.Learn(entry, true)
			}
		}
	}

	if freedBytes < bytesNeeded {
		return ErrEvictionFailed
	}

	return nil
}

// evictLRU implements LRU eviction
func (hc *HierarchicalCache) evictLRU(cacheTier *CacheTierImpl, tier CacheTier, bytesNeeded int64) error {
	entries := cacheTier.GetAllEntries()

	// Sort by last access time
	sortedEntries := make([]*CacheEntry, 0, len(entries))
	for _, entry := range entries {
		sortedEntries = append(sortedEntries, entry)
	}

	// Simple bubble sort by LastAccessedAt (oldest first)
	for i := 0; i < len(sortedEntries)-1; i++ {
		for j := 0; j < len(sortedEntries)-i-1; j++ {
			if sortedEntries[j].LastAccessedAt.After(sortedEntries[j+1].LastAccessedAt) {
				sortedEntries[j], sortedEntries[j+1] = sortedEntries[j+1], sortedEntries[j]
			}
		}
	}

	var freedBytes int64
	for _, entry := range sortedEntries {
		if freedBytes >= bytesNeeded {
			break
		}

		// Try to demote
		lowerTier := tier + 1
		if lowerTier <= L3 {
			if err := hc.demote(entry, tier, lowerTier); err == nil {
				cacheTier.Delete(entry.Key)
				freedBytes += entry.Size
				hc.recordEviction(tier)
				continue
			}
		}

		// Evict
		hc.releaseEntry(entry)
		cacheTier.Delete(entry.Key)
		freedBytes += entry.Size
		hc.recordEviction(tier)
	}

	if freedBytes < bytesNeeded {
		return ErrEvictionFailed
	}

	return nil
}

// getValue extracts the actual value from an entry
func (hc *HierarchicalCache) getValue(entry *CacheEntry) []byte {
	if entry.Compressed && hc.compressor != nil {
		if decompressed, err := hc.compressor.Decompress(entry.CompressedValue); err == nil {
			return decompressed
		}
	}
	return entry.Value
}

// releaseEntry releases resources for an entry
func (hc *HierarchicalCache) releaseEntry(entry *CacheEntry) {
	if entry.Hash != nil && hc.dedup != nil {
		hc.dedup.ReleaseRef(entry.Hash)
	}
}

// extractFeatures extracts ML features from an entry
func (hc *HierarchicalCache) extractFeatures(entry *CacheEntry) []float64 {
	now := time.Now()

	return []float64{
		float64(now.Unix() - entry.LastAccessedAt.Unix()), // Recency
		float64(entry.AccessCount),                         // Frequency
		float64(entry.Size),                                // Size
		float64(entry.AccessPattern),                       // Pattern
		float64(now.Hour()),                                // Time of day
		float64(entry.Tier),                                // Current tier
		entry.CompressionRatio,                             // Compression ratio
		float64(entry.RefCount),                            // Reference count
	}
}

// updateAccessStats updates access statistics for an entry
func (hc *HierarchicalCache) updateAccessStats(entry *CacheEntry) {
	entry.LastAccessedAt = time.Now()
	entry.AccessCount++
	entry.Features = hc.extractFeatures(entry)

	// Learn from access
	if hc.mlReplacer != nil && hc.config.EnableOnline {
		hc.mlReplacer.Learn(entry, false)
	}
}

// triggerPrefetch triggers predictive prefetching
func (hc *HierarchicalCache) triggerPrefetch(key string) {
	if hc.prefetcher == nil {
		return
	}

	predicted, err := hc.prefetcher.PredictNext(key, hc.config.PrefetchWindow)
	if err != nil {
		return
	}

	req := &PrefetchRequest{
		Keys:     predicted,
		Priority: 5,
	}

	hc.prefetcher.Prefetch(req)
}

// startBackgroundTasks starts background maintenance tasks
func (hc *HierarchicalCache) startBackgroundTasks() {
	// Metrics collection
	hc.wg.Add(1)
	go hc.metricsCollector()

	// ML model training
	if hc.mlReplacer != nil {
		hc.wg.Add(1)
		go hc.mlModelTrainer()
	}

	// Cache warming
	if hc.config.EnableWarming {
		hc.wg.Add(1)
		go hc.cacheWarmer()
	}

	// Garbage collection
	if hc.dedup != nil {
		hc.wg.Add(1)
		go hc.dedupGC()
	}
}

// metricsCollector collects cache metrics
func (hc *HierarchicalCache) metricsCollector() {
	defer hc.wg.Done()

	ticker := time.NewTicker(hc.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			hc.collectMetrics()
		}
	}
}

// mlModelTrainer trains the ML model periodically
func (hc *HierarchicalCache) mlModelTrainer() {
	defer hc.wg.Done()

	ticker := time.NewTicker(hc.config.MLUpdateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			if err := hc.mlReplacer.SaveModel(hc.config.MLModelPath); err != nil {
				// Log error
			}
		}
	}
}

// cacheWarmer warms the cache periodically
func (hc *HierarchicalCache) cacheWarmer() {
	defer hc.wg.Done()

	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			// Warming logic would go here
		}
	}
}

// dedupGC runs garbage collection for deduplication
func (hc *HierarchicalCache) dedupGC() {
	defer hc.wg.Done()

	ticker := time.NewTicker(1 * time.Hour)
	defer ticker.Stop()

	for {
		select {
		case <-hc.ctx.Done():
			return
		case <-ticker.C:
			hc.dedup.GC()
		}
	}
}

// Stats returns cache statistics
func (hc *HierarchicalCache) Stats() *CacheStats {
	hc.statsMu.RLock()
	defer hc.statsMu.RUnlock()

	statsCopy := *hc.stats
	return &statsCopy
}

// Close closes the cache
func (hc *HierarchicalCache) Close() error {
	hc.closeMu.Lock()
	defer hc.closeMu.Unlock()

	if hc.closed {
		return nil
	}

	hc.closed = true
	hc.cancel()
	hc.wg.Wait()

	// Save ML model
	if hc.mlReplacer != nil {
		hc.mlReplacer.SaveModel(hc.config.MLModelPath)
	}

	return nil
}

// Helper methods for metrics recording
func (hc *HierarchicalCache) recordHit(tier CacheTier) {
	hc.statsMu.Lock()
	defer hc.statsMu.Unlock()
	hc.stats.TotalHits++
	hc.stats.TotalAccesses++
}

func (hc *HierarchicalCache) recordMiss() {
	hc.statsMu.Lock()
	defer hc.statsMu.Unlock()
	hc.stats.TotalMisses++
	hc.stats.TotalAccesses++
}

func (hc *HierarchicalCache) recordEviction(tier CacheTier) {
	hc.statsMu.Lock()
	defer hc.statsMu.Unlock()
	hc.stats.TotalEvictions++
}

func (hc *HierarchicalCache) recordPromotion(from, to CacheTier) {
	// Track tier transitions
}

func (hc *HierarchicalCache) recordDemotion(from, to CacheTier) {
	// Track tier transitions
}

func (hc *HierarchicalCache) recordLatency(op string, duration time.Duration) {
	// Track latency metrics
}

func (hc *HierarchicalCache) collectMetrics() {
	hc.statsMu.Lock()
	defer hc.statsMu.Unlock()

	if hc.stats.TotalAccesses > 0 {
		hc.stats.HitRate = float64(hc.stats.TotalHits) / float64(hc.stats.TotalAccesses)
		hc.stats.MissRate = float64(hc.stats.TotalMisses) / float64(hc.stats.TotalAccesses)
	}

	hc.stats.Timestamp = time.Now()
}
