package cache

import (
	"context"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// MemoryCache implements an in-memory LRU cache with TTL support
type MemoryCache struct {
	config   *MemoryCacheConfig
	items    map[string]*memoryCacheItem
	lru      *lruList
	stats    CacheStats
	logger   *logrus.Logger
	mutex    sync.RWMutex
	stopChan chan struct{}
	closed   bool
}

// MemoryCacheConfig holds memory cache configuration
type MemoryCacheConfig struct {
	MaxSize       int           `json:"max_size"`
	DefaultTTL    time.Duration `json:"default_ttl"`
	CleanupInt    time.Duration `json:"cleanup_interval"`
	EnableMetrics bool          `json:"enable_metrics"`
}

// memoryCacheItem represents a cached item with metadata
type memoryCacheItem struct {
	key       string
	value     []byte
	expiresAt time.Time
	prev      *memoryCacheItem
	next      *memoryCacheItem
}

// lruList maintains the LRU order of cache items
type lruList struct {
	head *memoryCacheItem
	tail *memoryCacheItem
	size int
}

// NewMemoryCache creates a new memory cache
func NewMemoryCache(config *MemoryCacheConfig, logger *logrus.Logger) (*MemoryCache, error) {
	if config == nil {
		config = &MemoryCacheConfig{
			MaxSize:       10000,
			DefaultTTL:    5 * time.Minute,
			CleanupInt:    30 * time.Second,
			EnableMetrics: true,
		}
	}

	if logger == nil {
		logger = logrus.New()
	}

	mc := &MemoryCache{
		config:   config,
		items:    make(map[string]*memoryCacheItem),
		lru:      &lruList{},
		stats:    CacheStats{LastUpdated: time.Now()},
		logger:   logger,
		stopChan: make(chan struct{}),
	}

	// Start cleanup goroutine
	go mc.cleanupLoop()

	logger.WithField("max_size", config.MaxSize).Info("Memory cache initialized")
	return mc, nil
}

// Get retrieves a value from memory cache
func (mc *MemoryCache) Get(ctx context.Context, key string) ([]byte, error) {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	if mc.closed {
		return nil, ErrCacheNotAvailable
	}

	item, exists := mc.items[key]
	if !exists {
		mc.updateStats(false, false)
		return nil, ErrCacheMiss
	}

	// Check if item has expired
	if time.Now().After(item.expiresAt) {
		// Remove expired item (done without lock upgrade for performance)
		go mc.deleteExpired(key)
		mc.updateStats(false, false)
		return nil, ErrCacheMiss
	}

	// Move to front of LRU list (done without lock upgrade for performance)
	go mc.moveToFront(key)

	mc.updateStats(true, false)
	return item.value, nil
}

// Set stores a value in memory cache
func (mc *MemoryCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if mc.closed {
		return ErrCacheNotAvailable
	}

	if ttl == 0 {
		ttl = mc.config.DefaultTTL
	}

	// Check if item already exists
	if existing, exists := mc.items[key]; exists {
		// Update existing item
		existing.value = value
		existing.expiresAt = time.Now().Add(ttl)
		mc.lru.moveToFront(existing)
	} else {
		// Create new item
		item := &memoryCacheItem{
			key:       key,
			value:     value,
			expiresAt: time.Now().Add(ttl),
		}

		// Check if we need to evict items
		if mc.lru.size >= mc.config.MaxSize {
			mc.evictLRU()
		}

		mc.items[key] = item
		mc.lru.addToFront(item)
	}

	mc.updateStats(false, true)
	return nil
}

// Delete removes a key from memory cache
func (mc *MemoryCache) Delete(ctx context.Context, key string) error {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if mc.closed {
		return ErrCacheNotAvailable
	}

	item, exists := mc.items[key]
	if !exists {
		return nil // No error for deleting non-existent key
	}

	delete(mc.items, key)
	mc.lru.remove(item)

	return nil
}

// Exists checks if a key exists in memory cache
func (mc *MemoryCache) Exists(ctx context.Context, key string) (bool, error) {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	if mc.closed {
		return false, ErrCacheNotAvailable
	}

	item, exists := mc.items[key]
	if !exists {
		return false, nil
	}

	// Check if expired
	if time.Now().After(item.expiresAt) {
		go mc.deleteExpired(key)
		return false, nil
	}

	return true, nil
}

// Clear removes all items from memory cache
func (mc *MemoryCache) Clear(ctx context.Context) error {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if mc.closed {
		return ErrCacheNotAvailable
	}

	mc.items = make(map[string]*memoryCacheItem)
	mc.lru = &lruList{}

	return nil
}

// GetMulti retrieves multiple keys from memory cache
func (mc *MemoryCache) GetMulti(ctx context.Context, keys []string) (map[string][]byte, error) {
	result := make(map[string][]byte)

	for _, key := range keys {
		if value, err := mc.Get(ctx, key); err == nil {
			result[key] = value
		}
	}

	return result, nil
}

// SetMulti sets multiple key-value pairs in memory cache
func (mc *MemoryCache) SetMulti(ctx context.Context, items map[string]CacheItem) error {
	for key, item := range items {
		if err := mc.Set(ctx, key, item.Value, item.TTL); err != nil {
			return err
		}
	}
	return nil
}

// DeleteMulti removes multiple keys from memory cache
func (mc *MemoryCache) DeleteMulti(ctx context.Context, keys []string) error {
	for _, key := range keys {
		if err := mc.Delete(ctx, key); err != nil {
			return err
		}
	}
	return nil
}

// GetStats returns memory cache statistics
func (mc *MemoryCache) GetStats() CacheStats {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()

	stats := mc.stats
	total := stats.Hits + stats.Misses
	if total > 0 {
		stats.HitRate = float64(stats.Hits) / float64(total)
	}
	stats.LastUpdated = time.Now()

	return stats
}

// Close closes the memory cache
func (mc *MemoryCache) Close() error {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if mc.closed {
		return nil
	}

	mc.closed = true
	close(mc.stopChan)

	// Clear all items
	mc.items = nil
	mc.lru = nil

	mc.logger.Info("Memory cache closed")
	return nil
}

// cleanupLoop periodically removes expired items
func (mc *MemoryCache) cleanupLoop() {
	ticker := time.NewTicker(mc.config.CleanupInt)
	defer ticker.Stop()

	for {
		select {
		case <-mc.stopChan:
			return
		case <-ticker.C:
			mc.cleanupExpired()
		}
	}
}

// cleanupExpired removes expired items from cache
func (mc *MemoryCache) cleanupExpired() {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if mc.closed {
		return
	}

	now := time.Now()
	expired := make([]string, 0)

	// Find expired items
	for key, item := range mc.items {
		if now.After(item.expiresAt) {
			expired = append(expired, key)
		}
	}

	// Remove expired items
	for _, key := range expired {
		if item, exists := mc.items[key]; exists {
			delete(mc.items, key)
			mc.lru.remove(item)
		}
	}

	if len(expired) > 0 {
		mc.logger.WithField("count", len(expired)).Debug("Cleaned up expired cache items")
	}
}

// deleteExpired deletes an expired item (called from goroutine)
func (mc *MemoryCache) deleteExpired(key string) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if item, exists := mc.items[key]; exists && time.Now().After(item.expiresAt) {
		delete(mc.items, key)
		mc.lru.remove(item)
	}
}

// moveToFront moves an item to the front of LRU list (called from goroutine)
func (mc *MemoryCache) moveToFront(key string) {
	mc.mutex.Lock()
	defer mc.mutex.Unlock()

	if item, exists := mc.items[key]; exists {
		mc.lru.moveToFront(item)
	}
}

// evictLRU removes the least recently used item
func (mc *MemoryCache) evictLRU() {
	if mc.lru.tail != nil {
		delete(mc.items, mc.lru.tail.key)
		mc.lru.remove(mc.lru.tail)
	}
}

// updateStats updates cache statistics
func (mc *MemoryCache) updateStats(hit, set bool) {
	if hit {
		mc.stats.Hits++
	} else if !set {
		mc.stats.Misses++
	}

	if set {
		mc.stats.Sets++
	}

	mc.stats.LastUpdated = time.Now()
}

// LRU list methods

// addToFront adds an item to the front of the LRU list
func (lru *lruList) addToFront(item *memoryCacheItem) {
	if lru.head == nil {
		lru.head = item
		lru.tail = item
	} else {
		item.next = lru.head
		lru.head.prev = item
		lru.head = item
	}
	lru.size++
}

// moveToFront moves an item to the front of the LRU list
func (lru *lruList) moveToFront(item *memoryCacheItem) {
	if item == lru.head {
		return // Already at front
	}

	// Remove from current position
	lru.remove(item)
	
	// Add to front
	lru.addToFront(item)
}

// remove removes an item from the LRU list
func (lru *lruList) remove(item *memoryCacheItem) {
	if item.prev != nil {
		item.prev.next = item.next
	} else {
		lru.head = item.next
	}

	if item.next != nil {
		item.next.prev = item.prev
	} else {
		lru.tail = item.prev
	}

	item.prev = nil
	item.next = nil
	lru.size--
}