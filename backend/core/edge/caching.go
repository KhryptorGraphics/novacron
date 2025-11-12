// Package edge provides intelligent caching for edge computing
package edge

import (
	"container/list"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// CacheEntry represents a cached item
type CacheEntry struct {
	Key          string                 `json:"key"`
	Value        interface{}            `json:"value"`
	Size         uint64                 `json:"size"`
	Type         CacheItemType          `json:"type"`
	AccessCount  uint64                 `json:"access_count"`
	LastAccess   time.Time              `json:"last_access"`
	CreatedAt    time.Time              `json:"created_at"`
	ExpiresAt    *time.Time             `json:"expires_at,omitempty"`
	Cost         float64                `json:"cost"`
	Priority     int                    `json:"priority"`
	Metadata     map[string]interface{} `json:"metadata"`
	Checksum     string                 `json:"checksum"`
	NodeAffinity []string               `json:"node_affinity,omitempty"`
	element      *list.Element
	mu           sync.RWMutex
}

// CacheItemType represents the type of cached item
type CacheItemType string

const (
	CacheItemTypeVMImage     CacheItemType = "vm_image"
	CacheItemTypeContainer   CacheItemType = "container"
	CacheItemTypeData        CacheItemType = "data"
	CacheItemTypeComputed    CacheItemType = "computed"
	CacheItemTypeMLModel     CacheItemType = "ml_model"
	CacheItemTypeStatic      CacheItemType = "static"
)

// EvictionPolicy represents cache eviction policies
type EvictionPolicy string

const (
	EvictionPolicyLRU        EvictionPolicy = "lru"
	EvictionPolicyLFU        EvictionPolicy = "lfu"
	EvictionPolicyFIFO       EvictionPolicy = "fifo"
	EvictionPolicyTTL        EvictionPolicy = "ttl"
	EvictionPolicyARC        EvictionPolicy = "arc"
	EvictionPolicyCost       EvictionPolicy = "cost"
	EvictionPolicyAdaptive   EvictionPolicy = "adaptive"
)

// CacheStats represents cache statistics
type CacheStats struct {
	Hits             uint64    `json:"hits"`
	Misses           uint64    `json:"misses"`
	Evictions        uint64    `json:"evictions"`
	BytesWritten     uint64    `json:"bytes_written"`
	BytesRead        uint64    `json:"bytes_read"`
	TotalSize        uint64    `json:"total_size"`
	ItemCount        int       `json:"item_count"`
	HitRate          float64   `json:"hit_rate"`
	AvgAccessTime    float64   `json:"avg_access_time_ms"`
	LastEviction     time.Time `json:"last_eviction"`
}

// EdgeCache represents the edge caching layer
type EdgeCache struct {
	capacity     uint64
	currentSize  uint64
	entries      map[string]*CacheEntry
	evictionList *list.List
	policy       EvictionPolicy
	predictor    *CachePredictor
	coherency    *CacheCoherency
	preWarmer    *CachePreWarmer
	stats        *CacheStats
	metrics      *CacheMetrics
	config       *CacheConfig
	mu           sync.RWMutex
	ctx          context.Context
	cancel       context.CancelFunc
	wg           sync.WaitGroup
}

// CacheConfig contains cache configuration
type CacheConfig struct {
	MaxSize            uint64
	EvictionPolicy     EvictionPolicy
	TTL                time.Duration
	PreWarmEnabled     bool
	CoherencyEnabled   bool
	CompressionEnabled bool
	PersistentStorage  bool
	StoragePath        string
	SyncInterval       time.Duration
	MinItemSize        uint64
	MaxItemSize        uint64
}

// CachePredictor predicts cache access patterns
type CachePredictor struct {
	model         *PredictionModel
	accessHistory map[string][]time.Time
	patterns      map[string]*AccessPattern
	mu            sync.RWMutex
}

// PredictionModel represents a cache prediction model
type PredictionModel struct {
	weights        map[string]float64
	learningRate   float64
	decayFactor    float64
	windowSize     int
}

// AccessPattern represents an access pattern
type AccessPattern struct {
	Frequency    float64
	Periodicity  time.Duration
	LastAccess   time.Time
	NextPredicted time.Time
	Confidence   float64
}

// CacheCoherency maintains cache coherency across nodes
type CacheCoherency struct {
	nodeManager  *NodeManager
	versionMap   sync.Map // map[string]uint64
	invalidations chan InvalidationMessage
	mu           sync.RWMutex
}

// InvalidationMessage represents a cache invalidation message
type InvalidationMessage struct {
	Key       string    `json:"key"`
	Version   uint64    `json:"version"`
	NodeID    string    `json:"node_id"`
	Timestamp time.Time `json:"timestamp"`
	Scope     string    `json:"scope"` // local, regional, global
}

// CachePreWarmer pre-warms cache based on predictions
type CachePreWarmer struct {
	predictor    *CachePredictor
	scheduler    *PreWarmScheduler
	activeJobs   sync.Map
	maxConcurrent int
	mu           sync.RWMutex
}

// PreWarmScheduler schedules pre-warming jobs
type PreWarmScheduler struct {
	jobs      []*PreWarmJob
	jobQueue  chan *PreWarmJob
	mu        sync.RWMutex
}

// PreWarmJob represents a cache pre-warming job
type PreWarmJob struct {
	ID         string
	Keys       []string
	Priority   int
	Deadline   time.Time
	Status     string
	StartTime  *time.Time
	EndTime    *time.Time
	ItemsWarmed int
}

// CacheMetrics tracks cache metrics
type CacheMetrics struct {
	cacheHits        prometheus.Counter
	cacheMisses      prometheus.Counter
	cacheEvictions   prometheus.Counter
	cacheSize        prometheus.Gauge
	cacheItems       prometheus.Gauge
	hitRate          prometheus.Gauge
	accessLatency    prometheus.Histogram
	evictionLatency  prometheus.Histogram
	coherencyUpdates prometheus.Counter
	preWarmSuccess   prometheus.Counter
	preWarmFailure   prometheus.Counter
}

// ARCCache implements Adaptive Replacement Cache
type ARCCache struct {
	capacity uint64
	p        uint64 // Target size for T1
	t1       *list.List // Recent cache entries
	t2       *list.List // Frequent cache entries
	b1       *list.List // Ghost entries recently evicted from T1
	b2       *list.List // Ghost entries recently evicted from T2
	entries  map[string]*CacheEntry
	mu       sync.RWMutex
}

// LFUCache implements Least Frequently Used cache
type LFUCache struct {
	capacity      uint64
	minFreq       uint64
	entries       map[string]*CacheEntry
	frequencies   map[uint64]*list.List
	mu            sync.RWMutex
}

// NewEdgeCache creates a new edge cache
func NewEdgeCache(config *CacheConfig) *EdgeCache {
	ctx, cancel := context.WithCancel(context.Background())

	cache := &EdgeCache{
		capacity:     config.MaxSize,
		entries:      make(map[string]*CacheEntry),
		evictionList: list.New(),
		policy:       config.EvictionPolicy,
		predictor:    NewCachePredictor(),
		coherency:    NewCacheCoherency(nil),
		preWarmer:    NewCachePreWarmer(nil),
		stats:        &CacheStats{},
		metrics:      NewCacheMetrics(),
		config:       config,
		ctx:          ctx,
		cancel:       cancel,
	}

	// Start background workers
	cache.wg.Add(3)
	go cache.evictionWorker()
	go cache.coherencyWorker()
	go cache.preWarmWorker()

	return cache
}

// NewCachePredictor creates a new cache predictor
func NewCachePredictor() *CachePredictor {
	return &CachePredictor{
		model: &PredictionModel{
			weights:      make(map[string]float64),
			learningRate: 0.01,
			decayFactor:  0.95,
			windowSize:   100,
		},
		accessHistory: make(map[string][]time.Time),
		patterns:      make(map[string]*AccessPattern),
	}
}

// NewCacheCoherency creates a new cache coherency manager
func NewCacheCoherency(nodeManager *NodeManager) *CacheCoherency {
	return &CacheCoherency{
		nodeManager:   nodeManager,
		invalidations: make(chan InvalidationMessage, 1000),
	}
}

// NewCachePreWarmer creates a new cache pre-warmer
func NewCachePreWarmer(predictor *CachePredictor) *CachePreWarmer {
	return &CachePreWarmer{
		predictor:     predictor,
		scheduler:     NewPreWarmScheduler(),
		maxConcurrent: 5,
	}
}

// NewPreWarmScheduler creates a new pre-warm scheduler
func NewPreWarmScheduler() *PreWarmScheduler {
	return &PreWarmScheduler{
		jobs:     make([]*PreWarmJob, 0),
		jobQueue: make(chan *PreWarmJob, 100),
	}
}

// NewCacheMetrics creates new cache metrics
func NewCacheMetrics() *CacheMetrics {
	return &CacheMetrics{
		cacheHits: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_cache_hits_total",
				Help: "Total number of cache hits",
			},
		),
		cacheMisses: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_cache_misses_total",
				Help: "Total number of cache misses",
			},
		),
		cacheEvictions: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_cache_evictions_total",
				Help: "Total number of cache evictions",
			},
		),
		cacheSize: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_cache_size_bytes",
				Help: "Current cache size in bytes",
			},
		),
		cacheItems: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_cache_items_total",
				Help: "Total number of cached items",
			},
		),
		hitRate: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Name: "edge_cache_hit_rate",
				Help: "Cache hit rate percentage",
			},
		),
		accessLatency: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "edge_cache_access_latency_milliseconds",
				Help:    "Cache access latency",
				Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 25, 50, 100},
			},
		),
		evictionLatency: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "edge_cache_eviction_latency_milliseconds",
				Help:    "Cache eviction latency",
				Buckets: []float64{0.1, 0.5, 1, 2, 5, 10, 25, 50, 100},
			},
		),
		coherencyUpdates: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_cache_coherency_updates_total",
				Help: "Total number of cache coherency updates",
			},
		),
		preWarmSuccess: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_cache_prewarm_success_total",
				Help: "Total successful pre-warm operations",
			},
		),
		preWarmFailure: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "edge_cache_prewarm_failure_total",
				Help: "Total failed pre-warm operations",
			},
		),
	}
}

// Get retrieves an item from cache
func (c *EdgeCache) Get(key string) (interface{}, bool) {
	start := time.Now()
	defer func() {
		c.metrics.accessLatency.Observe(float64(time.Since(start).Milliseconds()))
	}()

	c.mu.RLock()
	entry, exists := c.entries[key]
	c.mu.RUnlock()

	if !exists {
		atomic.AddUint64(&c.stats.Misses, 1)
		c.metrics.cacheMisses.Inc()
		c.updateHitRate()
		return nil, false
	}

	// Check expiration
	if entry.ExpiresAt != nil && time.Now().After(*entry.ExpiresAt) {
		c.Delete(key)
		atomic.AddUint64(&c.stats.Misses, 1)
		c.metrics.cacheMisses.Inc()
		return nil, false
	}

	// Update access metadata
	entry.mu.Lock()
	entry.LastAccess = time.Now()
	entry.AccessCount++
	entry.mu.Unlock()

	// Update position based on eviction policy
	c.updatePosition(entry)

	// Record access for prediction
	c.predictor.RecordAccess(key)

	atomic.AddUint64(&c.stats.Hits, 1)
	c.metrics.cacheHits.Inc()
	c.updateHitRate()

	return entry.Value, true
}

// Put stores an item in cache
func (c *EdgeCache) Put(key string, value interface{}, size uint64, itemType CacheItemType) error {
	// Validate size
	if size > c.capacity {
		return fmt.Errorf("item size %d exceeds cache capacity %d", size, c.capacity)
	}

	if c.config.MinItemSize > 0 && size < c.config.MinItemSize {
		return fmt.Errorf("item size %d below minimum %d", size, c.config.MinItemSize)
	}

	if c.config.MaxItemSize > 0 && size > c.config.MaxItemSize {
		return fmt.Errorf("item size %d exceeds maximum %d", size, c.config.MaxItemSize)
	}

	// Create cache entry
	entry := &CacheEntry{
		Key:         key,
		Value:       value,
		Size:        size,
		Type:        itemType,
		AccessCount: 0,
		CreatedAt:   time.Now(),
		LastAccess:  time.Now(),
		Cost:        c.calculateCost(size, itemType),
		Checksum:    c.calculateChecksum(value),
		Metadata:    make(map[string]interface{}),
	}

	// Set expiration if TTL is configured
	if c.config.TTL > 0 {
		expiresAt := time.Now().Add(c.config.TTL)
		entry.ExpiresAt = &expiresAt
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Check if key already exists
	if existingEntry, exists := c.entries[key]; exists {
		c.currentSize -= existingEntry.Size
		c.removeFromEvictionList(existingEntry)
	}

	// Evict items if necessary
	for c.currentSize+size > c.capacity {
		if !c.evictOne() {
			return fmt.Errorf("unable to make space for new item")
		}
	}

	// Add to cache
	c.entries[key] = entry
	c.currentSize += size
	c.addToEvictionList(entry)

	// Update stats
	atomic.AddUint64(&c.stats.BytesWritten, size)
	c.stats.TotalSize = c.currentSize
	c.stats.ItemCount = len(c.entries)

	// Update metrics
	c.metrics.cacheSize.Set(float64(c.currentSize))
	c.metrics.cacheItems.Set(float64(len(c.entries)))

	// Notify coherency manager
	if c.config.CoherencyEnabled {
		c.coherency.NotifyUpdate(key, 1)
	}

	return nil
}

// Delete removes an item from cache
func (c *EdgeCache) Delete(key string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	entry, exists := c.entries[key]
	if !exists {
		return false
	}

	// Remove from cache
	delete(c.entries, key)
	c.currentSize -= entry.Size
	c.removeFromEvictionList(entry)

	// Update stats
	c.stats.TotalSize = c.currentSize
	c.stats.ItemCount = len(c.entries)

	// Update metrics
	c.metrics.cacheSize.Set(float64(c.currentSize))
	c.metrics.cacheItems.Set(float64(len(c.entries)))

	// Notify coherency manager
	if c.config.CoherencyEnabled {
		c.coherency.NotifyInvalidation(key)
	}

	return true
}

// PreWarm pre-warms cache with predicted items
func (c *EdgeCache) PreWarm(keys []string, priority int) error {
	job := &PreWarmJob{
		ID:       fmt.Sprintf("prewarm-%d", time.Now().UnixNano()),
		Keys:     keys,
		Priority: priority,
		Deadline: time.Now().Add(5 * time.Minute),
		Status:   "pending",
	}

	c.preWarmer.scheduler.Schedule(job)

	return nil
}

// Invalidate invalidates cache entries across nodes
func (c *EdgeCache) Invalidate(key string, scope string) {
	msg := InvalidationMessage{
		Key:       key,
		Version:   c.getVersion(key),
		NodeID:    "current", // Would be actual node ID
		Timestamp: time.Now(),
		Scope:     scope,
	}

	c.coherency.invalidations <- msg
}

// evictOne evicts a single item based on policy
func (c *EdgeCache) evictOne() bool {
	start := time.Now()
	defer func() {
		c.metrics.evictionLatency.Observe(float64(time.Since(start).Milliseconds()))
	}()

	if c.evictionList.Len() == 0 {
		return false
	}

	var victim *CacheEntry

	switch c.policy {
	case EvictionPolicyLRU:
		victim = c.evictLRU()
	case EvictionPolicyLFU:
		victim = c.evictLFU()
	case EvictionPolicyFIFO:
		victim = c.evictFIFO()
	case EvictionPolicyTTL:
		victim = c.evictTTL()
	case EvictionPolicyCost:
		victim = c.evictCostBased()
	case EvictionPolicyAdaptive:
		victim = c.evictAdaptive()
	default:
		victim = c.evictLRU()
	}

	if victim == nil {
		return false
	}

	// Remove from cache
	delete(c.entries, victim.Key)
	c.currentSize -= victim.Size
	c.removeFromEvictionList(victim)

	// Update stats
	atomic.AddUint64(&c.stats.Evictions, 1)
	c.stats.LastEviction = time.Now()

	// Update metrics
	c.metrics.cacheEvictions.Inc()

	return true
}

// Eviction strategies

func (c *EdgeCache) evictLRU() *CacheEntry {
	elem := c.evictionList.Back()
	if elem == nil {
		return nil
	}
	return elem.Value.(*CacheEntry)
}

func (c *EdgeCache) evictLFU() *CacheEntry {
	var minAccess uint64 = math.MaxUint64
	var victim *CacheEntry

	for elem := c.evictionList.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(*CacheEntry)
		if entry.AccessCount < minAccess {
			minAccess = entry.AccessCount
			victim = entry
		}
	}

	return victim
}

func (c *EdgeCache) evictFIFO() *CacheEntry {
	elem := c.evictionList.Front()
	if elem == nil {
		return nil
	}
	return elem.Value.(*CacheEntry)
}

func (c *EdgeCache) evictTTL() *CacheEntry {
	var oldestExpiry time.Time
	var victim *CacheEntry

	for elem := c.evictionList.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(*CacheEntry)
		if entry.ExpiresAt != nil {
			if victim == nil || entry.ExpiresAt.Before(oldestExpiry) {
				oldestExpiry = *entry.ExpiresAt
				victim = entry
			}
		}
	}

	if victim == nil {
		// Fall back to LRU if no TTL entries
		return c.evictLRU()
	}

	return victim
}

func (c *EdgeCache) evictCostBased() *CacheEntry {
	var minValue float64 = math.MaxFloat64
	var victim *CacheEntry

	for elem := c.evictionList.Front(); elem != nil; elem = elem.Next() {
		entry := elem.Value.(*CacheEntry)
		value := c.calculateValue(entry)
		if value < minValue {
			minValue = value
			victim = entry
		}
	}

	return victim
}

func (c *EdgeCache) evictAdaptive() *CacheEntry {
	// Adaptive eviction based on access patterns
	hitRate := c.getHitRate()

	if hitRate < 0.5 {
		// Low hit rate, use LFU
		return c.evictLFU()
	} else if hitRate < 0.8 {
		// Medium hit rate, use cost-based
		return c.evictCostBased()
	} else {
		// High hit rate, use LRU
		return c.evictLRU()
	}
}

// Helper methods

func (c *EdgeCache) calculateCost(size uint64, itemType CacheItemType) float64 {
	baseCost := float64(size) / 1024.0 // Cost per KB

	// Adjust cost based on item type
	switch itemType {
	case CacheItemTypeVMImage:
		return baseCost * 2.0 // VM images are expensive
	case CacheItemTypeMLModel:
		return baseCost * 3.0 // ML models are very expensive
	case CacheItemTypeComputed:
		return baseCost * 1.5 // Computed results moderately expensive
	default:
		return baseCost
	}
}

func (c *EdgeCache) calculateValue(entry *CacheEntry) float64 {
	age := time.Since(entry.CreatedAt).Seconds()
	recency := time.Since(entry.LastAccess).Seconds()

	// Value = (access_count * cost) / (age * recency * size)
	value := (float64(entry.AccessCount) * entry.Cost) /
		(age * recency * float64(entry.Size))

	return value
}

func (c *EdgeCache) calculateChecksum(value interface{}) string {
	// Simple checksum calculation
	data := fmt.Sprintf("%v", value)
	hash := sha256.Sum256([]byte(data))
	return hex.EncodeToString(hash[:])
}

func (c *EdgeCache) updatePosition(entry *CacheEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if entry.element != nil {
		c.evictionList.MoveToFront(entry.element)
	}
}

func (c *EdgeCache) addToEvictionList(entry *CacheEntry) {
	entry.element = c.evictionList.PushFront(entry)
}

func (c *EdgeCache) removeFromEvictionList(entry *CacheEntry) {
	if entry.element != nil {
		c.evictionList.Remove(entry.element)
		entry.element = nil
	}
}

func (c *EdgeCache) updateHitRate() {
	hits := atomic.LoadUint64(&c.stats.Hits)
	misses := atomic.LoadUint64(&c.stats.Misses)
	total := hits + misses

	if total > 0 {
		rate := float64(hits) / float64(total) * 100
		c.stats.HitRate = rate
		c.metrics.hitRate.Set(rate)
	}
}

func (c *EdgeCache) getHitRate() float64 {
	return c.stats.HitRate
}

func (c *EdgeCache) getVersion(key string) uint64 {
	if v, exists := c.coherency.versionMap.Load(key); exists {
		return v.(uint64)
	}
	return 0
}

// Predictor methods

func (p *CachePredictor) RecordAccess(key string) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Add to access history
	if _, exists := p.accessHistory[key]; !exists {
		p.accessHistory[key] = []time.Time{}
	}

	p.accessHistory[key] = append(p.accessHistory[key], time.Now())

	// Limit history size
	if len(p.accessHistory[key]) > p.model.windowSize {
		p.accessHistory[key] = p.accessHistory[key][1:]
	}

	// Update pattern
	p.updatePattern(key)
}

func (p *CachePredictor) updatePattern(key string) {
	history := p.accessHistory[key]
	if len(history) < 2 {
		return
	}

	// Calculate frequency
	duration := history[len(history)-1].Sub(history[0])
	frequency := float64(len(history)) / duration.Seconds()

	// Calculate periodicity
	var intervals []time.Duration
	for i := 1; i < len(history); i++ {
		intervals = append(intervals, history[i].Sub(history[i-1]))
	}

	avgInterval := p.averageDuration(intervals)

	// Update or create pattern
	pattern := &AccessPattern{
		Frequency:     frequency,
		Periodicity:   avgInterval,
		LastAccess:    history[len(history)-1],
		NextPredicted: history[len(history)-1].Add(avgInterval),
		Confidence:    p.calculateConfidence(intervals),
	}

	p.patterns[key] = pattern
}

func (p *CachePredictor) averageDuration(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}

	var total int64
	for _, d := range durations {
		total += int64(d)
	}

	return time.Duration(total / int64(len(durations)))
}

func (p *CachePredictor) calculateConfidence(intervals []time.Duration) float64 {
	if len(intervals) < 2 {
		return 0
	}

	avg := p.averageDuration(intervals)
	var variance float64

	for _, interval := range intervals {
		diff := float64(interval - avg)
		variance += diff * diff
	}

	variance /= float64(len(intervals))
	stdDev := math.Sqrt(variance)

	// Confidence inversely proportional to standard deviation
	confidence := 1.0 / (1.0 + stdDev/float64(avg))

	return math.Min(1.0, math.Max(0, confidence))
}

func (p *CachePredictor) PredictNextAccess(key string) (time.Time, float64) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	pattern, exists := p.patterns[key]
	if !exists {
		return time.Time{}, 0
	}

	return pattern.NextPredicted, pattern.Confidence
}

// Coherency methods

func (cc *CacheCoherency) NotifyUpdate(key string, version uint64) {
	cc.versionMap.Store(key, version)
	cc.metrics.coherencyUpdates.Inc()
}

func (cc *CacheCoherency) NotifyInvalidation(key string) {
	msg := InvalidationMessage{
		Key:       key,
		Timestamp: time.Now(),
		Scope:     "local",
	}

	select {
	case cc.invalidations <- msg:
	default:
		// Channel full, drop message
	}
}

// Pre-warmer methods

func (pw *PreWarmScheduler) Schedule(job *PreWarmJob) {
	pw.mu.Lock()
	defer pw.mu.Unlock()

	pw.jobs = append(pw.jobs, job)

	select {
	case pw.jobQueue <- job:
	default:
		// Queue full
	}
}

// Worker loops

func (c *EdgeCache) evictionWorker() {
	defer c.wg.Done()

	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.cleanupExpired()
		case <-c.ctx.Done():
			return
		}
	}
}

func (c *EdgeCache) coherencyWorker() {
	defer c.wg.Done()

	for {
		select {
		case msg := <-c.coherency.invalidations:
			c.handleInvalidation(msg)
		case <-c.ctx.Done():
			return
		}
	}
}

func (c *EdgeCache) preWarmWorker() {
	defer c.wg.Done()

	for {
		select {
		case job := <-c.preWarmer.scheduler.jobQueue:
			c.executePreWarm(job)
		case <-c.ctx.Done():
			return
		}
	}
}

func (c *EdgeCache) cleanupExpired() {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	for key, entry := range c.entries {
		if entry.ExpiresAt != nil && now.After(*entry.ExpiresAt) {
			delete(c.entries, key)
			c.currentSize -= entry.Size
			c.removeFromEvictionList(entry)
		}
	}
}

func (c *EdgeCache) handleInvalidation(msg InvalidationMessage) {
	c.Delete(msg.Key)
}

func (c *EdgeCache) executePreWarm(job *PreWarmJob) {
	start := time.Now()
	job.StartTime = &start
	job.Status = "running"

	for _, key := range job.Keys {
		// Simulate fetching and caching
		// In production, would fetch from origin
		job.ItemsWarmed++
	}

	end := time.Now()
	job.EndTime = &end
	job.Status = "completed"

	c.metrics.preWarmSuccess.Inc()
}

// Stop stops the edge cache
func (c *EdgeCache) Stop() {
	c.cancel()
	c.wg.Wait()
}