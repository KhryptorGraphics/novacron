package cache

import (
	"context"
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// PersistentCache implements file-based persistent caching
type PersistentCache struct {
	config    *PersistentCacheConfig
	indexPath string
	dataPath  string
	index     map[string]*persistentCacheEntry
	stats     CacheStats
	logger    *logrus.Logger
	mutex     sync.RWMutex
	closed    bool
}

// PersistentCacheConfig holds persistent cache configuration
type PersistentCacheConfig struct {
	StoragePath   string `json:"storage_path"`
	EnableMetrics bool   `json:"enable_metrics"`
}

// persistentCacheEntry represents a cached entry with metadata
type persistentCacheEntry struct {
	Key       string    `json:"key"`
	FileName  string    `json:"file_name"`
	Size      int64     `json:"size"`
	ExpiresAt time.Time `json:"expires_at"`
	CreatedAt time.Time `json:"created_at"`
}

// NewPersistentCache creates a new persistent cache instance
func NewPersistentCache(config *PersistentCacheConfig, logger *logrus.Logger) (*PersistentCache, error) {
	if config == nil {
		config = &PersistentCacheConfig{
			StoragePath:   "/var/cache/novacron",
			EnableMetrics: true,
		}
	}

	if logger == nil {
		logger = logrus.New()
	}

	// Create storage directory
	if err := os.MkdirAll(config.StoragePath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %w", err)
	}

	pc := &PersistentCache{
		config:    config,
		indexPath: filepath.Join(config.StoragePath, "cache.index"),
		dataPath:  filepath.Join(config.StoragePath, "data"),
		index:     make(map[string]*persistentCacheEntry),
		stats:     CacheStats{LastUpdated: time.Now()},
		logger:    logger,
	}

	// Create data directory
	if err := os.MkdirAll(pc.dataPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	// Load existing index
	if err := pc.loadIndex(); err != nil {
		logger.WithError(err).Warn("Failed to load cache index, starting with empty cache")
	}

	// Clean up expired entries
	go pc.cleanup()

	logger.WithField("storage_path", config.StoragePath).Info("Persistent cache initialized")
	return pc, nil
}

// Get retrieves a value from persistent cache
func (pc *PersistentCache) Get(ctx context.Context, key string) ([]byte, error) {
	pc.mutex.RLock()
	defer pc.mutex.RUnlock()

	if pc.closed {
		return nil, ErrCacheNotAvailable
	}

	entry, exists := pc.index[key]
	if !exists {
		pc.stats.Misses++
		return nil, ErrCacheMiss
	}

	// Check if expired
	if time.Now().After(entry.ExpiresAt) {
		go pc.deleteExpired(key)
		pc.stats.Misses++
		return nil, ErrCacheMiss
	}

	// Read file
	filePath := filepath.Join(pc.dataPath, entry.FileName)
	data, err := os.ReadFile(filePath)
	if err != nil {
		pc.stats.Errors++
		return nil, fmt.Errorf("failed to read cache file: %w", err)
	}

	pc.stats.Hits++
	return data, nil
}

// Set stores a value in persistent cache
func (pc *PersistentCache) Set(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()

	if pc.closed {
		return ErrCacheNotAvailable
	}

	// Generate filename
	fileName := pc.generateFileName(key)
	filePath := filepath.Join(pc.dataPath, fileName)

	// Write file
	if err := os.WriteFile(filePath, value, 0644); err != nil {
		pc.stats.Errors++
		return fmt.Errorf("failed to write cache file: %w", err)
	}

	// Update index
	entry := &persistentCacheEntry{
		Key:       key,
		FileName:  fileName,
		Size:      int64(len(value)),
		ExpiresAt: time.Now().Add(ttl),
		CreatedAt: time.Now(),
	}

	// Remove old file if exists
	if oldEntry, exists := pc.index[key]; exists {
		oldPath := filepath.Join(pc.dataPath, oldEntry.FileName)
		os.Remove(oldPath) // Ignore error
	}

	pc.index[key] = entry
	pc.stats.Sets++

	// Save index
	go pc.saveIndex()

	return nil
}

// Delete removes a key from persistent cache
func (pc *PersistentCache) Delete(ctx context.Context, key string) error {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()

	if pc.closed {
		return ErrCacheNotAvailable
	}

	entry, exists := pc.index[key]
	if !exists {
		return nil // No error for deleting non-existent key
	}

	// Remove file
	filePath := filepath.Join(pc.dataPath, entry.FileName)
	os.Remove(filePath) // Ignore error

	// Remove from index
	delete(pc.index, key)
	pc.stats.Deletes++

	// Save index
	go pc.saveIndex()

	return nil
}

// Exists checks if a key exists in persistent cache
func (pc *PersistentCache) Exists(ctx context.Context, key string) (bool, error) {
	pc.mutex.RLock()
	defer pc.mutex.RUnlock()

	if pc.closed {
		return false, ErrCacheNotAvailable
	}

	entry, exists := pc.index[key]
	if !exists {
		return false, nil
	}

	// Check if expired
	if time.Now().After(entry.ExpiresAt) {
		go pc.deleteExpired(key)
		return false, nil
	}

	return true, nil
}

// Clear removes all keys from persistent cache
func (pc *PersistentCache) Clear(ctx context.Context) error {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()

	if pc.closed {
		return ErrCacheNotAvailable
	}

	// Remove all data files
	entries, err := os.ReadDir(pc.dataPath)
	if err == nil {
		for _, entry := range entries {
			if !entry.IsDir() {
				os.Remove(filepath.Join(pc.dataPath, entry.Name()))
			}
		}
	}

	// Clear index
	pc.index = make(map[string]*persistentCacheEntry)

	// Save empty index
	go pc.saveIndex()

	return nil
}

// GetMulti retrieves multiple keys from persistent cache
func (pc *PersistentCache) GetMulti(ctx context.Context, keys []string) (map[string][]byte, error) {
	result := make(map[string][]byte)

	for _, key := range keys {
		if value, err := pc.Get(ctx, key); err == nil {
			result[key] = value
		}
	}

	return result, nil
}

// SetMulti sets multiple key-value pairs in persistent cache
func (pc *PersistentCache) SetMulti(ctx context.Context, items map[string]CacheItem) error {
	for key, item := range items {
		if err := pc.Set(ctx, key, item.Value, item.TTL); err != nil {
			return err
		}
	}
	return nil
}

// DeleteMulti removes multiple keys from persistent cache
func (pc *PersistentCache) DeleteMulti(ctx context.Context, keys []string) error {
	for _, key := range keys {
		if err := pc.Delete(ctx, key); err != nil {
			return err
		}
	}
	return nil
}

// GetStats returns persistent cache statistics
func (pc *PersistentCache) GetStats() CacheStats {
	pc.mutex.RLock()
	defer pc.mutex.RUnlock()

	stats := pc.stats
	total := stats.Hits + stats.Misses
	if total > 0 {
		stats.HitRate = float64(stats.Hits) / float64(total)
	}
	stats.LastUpdated = time.Now()

	return stats
}

// Close closes the persistent cache
func (pc *PersistentCache) Close() error {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()

	if pc.closed {
		return nil
	}

	pc.closed = true

	// Save final index
	if err := pc.saveIndexSync(); err != nil {
		pc.logger.WithError(err).Error("Failed to save final cache index")
	}

	pc.logger.Info("Persistent cache closed")
	return nil
}

// generateFileName generates a safe filename for a cache key
func (pc *PersistentCache) generateFileName(key string) string {
	// Simple filename generation - in production, you might want to use a hash
	timestamp := time.Now().UnixNano()
	return fmt.Sprintf("cache_%d.dat", timestamp)
}

// loadIndex loads the cache index from disk
func (pc *PersistentCache) loadIndex() error {
	if _, err := os.Stat(pc.indexPath); os.IsNotExist(err) {
		return nil // No index file exists yet
	}

	file, err := os.Open(pc.indexPath)
	if err != nil {
		return fmt.Errorf("failed to open index file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	if err := decoder.Decode(&pc.index); err != nil {
		return fmt.Errorf("failed to decode index: %w", err)
	}

	pc.logger.WithField("entries", len(pc.index)).Info("Loaded cache index")
	return nil
}

// saveIndex saves the cache index to disk (async)
func (pc *PersistentCache) saveIndex() {
	pc.saveIndexSync()
}

// saveIndexSync saves the cache index to disk (sync)
func (pc *PersistentCache) saveIndexSync() error {
	tempPath := pc.indexPath + ".tmp"

	file, err := os.Create(tempPath)
	if err != nil {
		return fmt.Errorf("failed to create temp index file: %w", err)
	}

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(pc.index)
	file.Close()

	if err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to encode index: %w", err)
	}

	// Atomic move
	if err := os.Rename(tempPath, pc.indexPath); err != nil {
		os.Remove(tempPath)
		return fmt.Errorf("failed to move index file: %w", err)
	}

	return nil
}

// cleanup periodically removes expired entries
func (pc *PersistentCache) cleanup() {
	ticker := time.NewTicker(10 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			pc.cleanupExpired()
		}
	}
}

// cleanupExpired removes expired entries from cache
func (pc *PersistentCache) cleanupExpired() {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()

	if pc.closed {
		return
	}

	now := time.Now()
	expired := make([]string, 0)

	// Find expired entries
	for key, entry := range pc.index {
		if now.After(entry.ExpiresAt) {
			expired = append(expired, key)
		}
	}

	// Remove expired entries
	for _, key := range expired {
		if entry, exists := pc.index[key]; exists {
			filePath := filepath.Join(pc.dataPath, entry.FileName)
			os.Remove(filePath) // Ignore error
			delete(pc.index, key)
		}
	}

	if len(expired) > 0 {
		pc.logger.WithField("count", len(expired)).Debug("Cleaned up expired persistent cache entries")
		go pc.saveIndex()
	}
}

// deleteExpired deletes an expired entry (called from goroutine)
func (pc *PersistentCache) deleteExpired(key string) {
	pc.mutex.Lock()
	defer pc.mutex.Unlock()

	if entry, exists := pc.index[key]; exists && time.Now().After(entry.ExpiresAt) {
		filePath := filepath.Join(pc.dataPath, entry.FileName)
		os.Remove(filePath) // Ignore error
		delete(pc.index, key)
		go pc.saveIndex()
	}
}