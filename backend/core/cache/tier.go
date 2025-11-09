package cache

import (
	"sync"
)

// CacheTierImpl implements a single cache tier
type CacheTierImpl struct {
	tier     CacheTier
	maxSize  int64
	usedSize int64
	entries  map[string]*CacheEntry
	mu       sync.RWMutex
}

// NewCacheTier creates a new cache tier
func NewCacheTier(tier CacheTier, maxSize int64) *CacheTierImpl {
	return &CacheTierImpl{
		tier:    tier,
		maxSize: maxSize,
		entries: make(map[string]*CacheEntry),
	}
}

// Get retrieves an entry from the tier
func (ct *CacheTierImpl) Get(key string) (*CacheEntry, bool) {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	entry, ok := ct.entries[key]
	return entry, ok
}

// Set stores an entry in the tier
func (ct *CacheTierImpl) Set(entry *CacheEntry) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	// Remove old entry if exists
	if old, ok := ct.entries[entry.Key]; ok {
		ct.usedSize -= old.Size
	}

	ct.entries[entry.Key] = entry
	ct.usedSize += entry.Size
}

// Delete removes an entry from the tier
func (ct *CacheTierImpl) Delete(key string) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	if entry, ok := ct.entries[key]; ok {
		ct.usedSize -= entry.Size
		delete(ct.entries, key)
	}
}

// Exists checks if a key exists in the tier
func (ct *CacheTierImpl) Exists(key string) bool {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	_, ok := ct.entries[key]
	return ok
}

// NeedsEviction checks if eviction is needed
func (ct *CacheTierImpl) NeedsEviction(newSize int64) bool {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	return ct.usedSize+newSize > ct.maxSize
}

// GetAllEntries returns all entries
func (ct *CacheTierImpl) GetAllEntries() map[string]*CacheEntry {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	entries := make(map[string]*CacheEntry, len(ct.entries))
	for k, v := range ct.entries {
		entries[k] = v
	}
	return entries
}

// Size returns current used size
func (ct *CacheTierImpl) Size() int64 {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	return ct.usedSize
}

// Count returns number of entries
func (ct *CacheTierImpl) Count() int {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	return len(ct.entries)
}
