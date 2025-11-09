package cache

import (
	"errors"
	"fmt"
	"time"
)

var (
	// Cache errors
	ErrNotFound        = errors.New("key not found in cache")
	ErrTierFull        = errors.New("cache tier is full")
	ErrInvalidKey      = errors.New("invalid cache key")
	ErrInvalidValue    = errors.New("invalid cache value")
	ErrCacheClosed     = errors.New("cache is closed")
	ErrInvalidConfig   = func(msg string) error { return fmt.Errorf("invalid config: %s", msg) }
	ErrEvictionFailed  = errors.New("eviction failed")
	ErrPrefetchFailed  = errors.New("prefetch failed")
	ErrCompressionFailed = errors.New("compression failed")
	ErrDecompressionFailed = errors.New("decompression failed")
)

// CacheTier represents the tier level
type CacheTier int

const (
	L1 CacheTier = iota // Edge cache (hot data)
	L2                  // Regional cache (warm data)
	L3                  // Global cache (cold data)
)

func (t CacheTier) String() string {
	switch t {
	case L1:
		return "L1"
	case L2:
		return "L2"
	case L3:
		return "L3"
	default:
		return "Unknown"
	}
}

// CacheEntry represents a cached item
type CacheEntry struct {
	Key           string
	Value         []byte
	CompressedValue []byte
	Tier          CacheTier
	Size          int64
	Compressed    bool
	CompressionRatio float64
	Hash          []byte // Content hash for deduplication
	RefCount      int    // Reference count for shared chunks

	// Timestamps
	CreatedAt     time.Time
	LastAccessedAt time.Time
	ExpiresAt     time.Time

	// Access tracking
	AccessCount   int64
	AccessPattern AccessPattern

	// ML features
	Features      []float64
	EvictionScore float64
}

// AccessPattern represents the type of access pattern
type AccessPattern int

const (
	PatternRandom AccessPattern = iota
	PatternSequential
	PatternBursty
	PatternPeriodic
)

func (p AccessPattern) String() string {
	switch p {
	case PatternRandom:
		return "Random"
	case PatternSequential:
		return "Sequential"
	case PatternBursty:
		return "Bursty"
	case PatternPeriodic:
		return "Periodic"
	default:
		return "Unknown"
	}
}

// CacheStats represents cache statistics
type CacheStats struct {
	// Hit/Miss rates
	HitRate          float64
	MissRate         float64
	EvictionRate     float64

	// Prefetching
	PrefetchAccuracy float64
	PrefetchCount    int64
	PrefetchHits     int64

	// Deduplication
	DeduplicationRatio float64
	DedupedChunks      int64
	DedupSavedBytes    int64

	// Size metrics
	TotalSize        int64
	UsedSize         int64
	L1Size           int64
	L2Size           int64
	L3Size           int64

	// Access metrics
	TotalAccesses    int64
	TotalHits        int64
	TotalMisses      int64
	TotalEvictions   int64

	// Latency metrics (microseconds)
	AvgReadLatency   float64
	AvgWriteLatency  float64
	P50Latency       float64
	P95Latency       float64
	P99Latency       float64

	// Compression metrics
	CompressionRatio float64
	CompressedBytes  int64

	// ML metrics
	MLModelAccuracy  float64
	MLPredictionTime float64

	// Per-tier stats
	TierStats        map[CacheTier]*TierStats

	// Timestamp
	Timestamp        time.Time
}

// TierStats represents per-tier statistics
type TierStats struct {
	Tier           CacheTier
	Size           int64
	Used           int64
	Entries        int64
	HitRate        float64
	EvictionRate   float64
	PromotionRate  float64
	DemotionRate   float64
}

// PrefetchRequest represents a prefetch request
type PrefetchRequest struct {
	Keys          []string
	Priority      int
	DeadlineBefore time.Time
	Pattern       AccessPattern
}

// WarmupRequest represents a cache warming request
type WarmupRequest struct {
	Pattern       string
	MaxItems      int
	TargetTier    CacheTier
	Priority      int
}

// EvictionCandidate represents a candidate for eviction
type EvictionCandidate struct {
	Key           string
	Tier          CacheTier
	Score         float64
	Size          int64
	LastAccessed  time.Time
	AccessCount   int64
	Features      []float64
}

// CacheAPI defines the main cache interface
type CacheAPI interface {
	// Basic operations
	Get(key string) ([]byte, error)
	Set(key string, value []byte, ttl time.Duration) error
	Delete(key string) error
	Exists(key string) bool

	// Batch operations
	GetMulti(keys []string) (map[string][]byte, error)
	SetMulti(entries map[string][]byte, ttl time.Duration) error
	DeleteMulti(keys []string) error

	// Prefetching
	Prefetch(req *PrefetchRequest) error

	// Warming
	Warmup(req *WarmupRequest) error

	// Statistics
	Stats() *CacheStats
	TierStats(tier CacheTier) *TierStats

	// Management
	Flush() error
	FlushTier(tier CacheTier) error
	Close() error
}

// MLCacheReplacer defines the ML-based cache replacement interface
type MLCacheReplacer interface {
	// Predict eviction score
	PredictEvictionScore(entry *CacheEntry) float64

	// Find eviction candidates
	FindEvictionCandidates(tier CacheTier, count int) ([]*EvictionCandidate, error)

	// Learn from feedback
	Learn(entry *CacheEntry, wasEvicted bool) error

	// Save/load model
	SaveModel(path string) error
	LoadModel(path string) error

	// Model metrics
	Accuracy() float64
}

// PrefetchEngine defines the prefetching interface
type PrefetchEngine interface {
	// Predict next accesses
	PredictNext(currentKey string, count int) ([]string, error)

	// Learn access pattern
	LearnPattern(sequence []string) error

	// Execute prefetch
	Prefetch(req *PrefetchRequest) error

	// Accuracy metrics
	Accuracy() float64
}

// ContentAddressedStorage defines the deduplication interface
type ContentAddressedStorage interface {
	// Store chunk
	StoreChunk(data []byte) (hash []byte, err error)

	// Retrieve chunk
	GetChunk(hash []byte) ([]byte, error)

	// Reference counting
	AddRef(hash []byte) error
	ReleaseRef(hash []byte) error

	// Garbage collection
	GC() (freedBytes int64, err error)

	// Stats
	DeduplicationRatio() float64
	TotalChunks() int64
	UniqueChunks() int64
}
