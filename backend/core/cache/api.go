package cache

import (
	"time"
)

// Ensure HierarchicalCache implements CacheAPI
var _ CacheAPI = (*HierarchicalCache)(nil)

// GetMulti retrieves multiple values from cache
func (hc *HierarchicalCache) GetMulti(keys []string) (map[string][]byte, error) {
	result := make(map[string][]byte)

	for _, key := range keys {
		if value, err := hc.Get(key); err == nil {
			result[key] = value
		}
	}

	return result, nil
}

// SetMulti stores multiple values in cache
func (hc *HierarchicalCache) SetMulti(entries map[string][]byte, ttl time.Duration) error {
	for key, value := range entries {
		if err := hc.Set(key, value, ttl); err != nil {
			return err
		}
	}

	return nil
}

// DeleteMulti removes multiple values from cache
func (hc *HierarchicalCache) DeleteMulti(keys []string) error {
	for _, key := range keys {
		if err := hc.Delete(key); err != nil {
			return err
		}
	}

	return nil
}

// Prefetch executes a prefetch request
func (hc *HierarchicalCache) Prefetch(req *PrefetchRequest) error {
	if hc.prefetcher == nil {
		return ErrPrefetchFailed
	}

	return hc.prefetcher.Prefetch(req)
}

// Warmup warms the cache
func (hc *HierarchicalCache) Warmup(req *WarmupRequest) error {
	// Cache warming implementation
	// This would fetch data matching the pattern and cache it
	return nil
}

// TierStats returns statistics for a specific tier
func (hc *HierarchicalCache) TierStats(tier CacheTier) *TierStats {
	hc.statsMu.RLock()
	defer hc.statsMu.RUnlock()

	if stats, ok := hc.stats.TierStats[tier]; ok {
		return stats
	}

	return &TierStats{Tier: tier}
}

// Flush clears all cache tiers
func (hc *HierarchicalCache) Flush() error {
	if hc.l1 != nil {
		hc.flushTier(hc.l1)
	}
	if hc.l2 != nil {
		hc.flushTier(hc.l2)
	}
	if hc.l3 != nil {
		hc.flushTier(hc.l3)
	}

	return nil
}

// FlushTier clears a specific cache tier
func (hc *HierarchicalCache) FlushTier(tier CacheTier) error {
	switch tier {
	case L1:
		if hc.l1 != nil {
			hc.flushTier(hc.l1)
		}
	case L2:
		if hc.l2 != nil {
			hc.flushTier(hc.l2)
		}
	case L3:
		if hc.l3 != nil {
			hc.flushTier(hc.l3)
		}
	}

	return nil
}

// flushTier internal helper to flush a tier
func (hc *HierarchicalCache) flushTier(tier *CacheTierImpl) {
	tier.mu.Lock()
	defer tier.mu.Unlock()

	// Release all entries
	for _, entry := range tier.entries {
		hc.releaseEntry(entry)
	}

	tier.entries = make(map[string]*CacheEntry)
	tier.usedSize = 0
}
