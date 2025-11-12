package compression

import (
	"sync"
	"time"
)

// CompressionMetrics tracks detailed compression performance metrics
type CompressionMetrics struct {
	// Compression stats
	TotalOperations      uint64
	TotalBytesOriginal   uint64
	TotalBytesCompressed uint64
	TotalBytesDelta      uint64

	// Delta encoding stats
	DeltaHits            uint64
	DeltaMisses          uint64
	BaselineRefreshes    uint64
	BaselineCount        int

	// Dictionary stats
	DictionaryHits       uint64
	DictionaryMisses     uint64
	DictionariesTrained  int
	LastDictionaryUpdate time.Time

	// Algorithm usage
	XORDeltaCount        uint64
	RSyncDeltaCount      uint64
	BSDiffDeltaCount     uint64
	AutoSelectCount      uint64

	// Performance
	TotalCompressionTime time.Duration
	TotalDecompressionTime time.Duration
	FastestCompression   time.Duration
	SlowestCompression   time.Duration

	// Adaptive compression
	LevelAdjustments     uint64
	IncompressibleSkips  uint64

	// Thread safety
	mu sync.RWMutex
}

// NewCompressionMetrics creates a new metrics tracker
func NewCompressionMetrics() *CompressionMetrics {
	return &CompressionMetrics{
		FastestCompression: time.Hour, // Initialize to high value
	}
}

// RecordCompression records a compression operation
func (m *CompressionMetrics) RecordCompression(originalSize, compressedSize int, duration time.Duration, isDelta bool, usedDict bool) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalOperations++
	m.TotalBytesOriginal += uint64(originalSize)
	m.TotalBytesCompressed += uint64(compressedSize)

	if isDelta {
		m.DeltaHits++
		m.TotalBytesDelta += uint64(compressedSize)
	} else {
		m.DeltaMisses++
	}

	if usedDict {
		m.DictionaryHits++
	} else {
		m.DictionaryMisses++
	}

	m.TotalCompressionTime += duration

	if duration < m.FastestCompression {
		m.FastestCompression = duration
	}
	if duration > m.SlowestCompression {
		m.SlowestCompression = duration
	}
}

// RecordDecompression records a decompression operation
func (m *CompressionMetrics) RecordDecompression(duration time.Duration) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.TotalDecompressionTime += duration
}

// RecordBaselineRefresh records a baseline refresh
func (m *CompressionMetrics) RecordBaselineRefresh() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.BaselineRefreshes++
}

// RecordDeltaAlgorithm records which delta algorithm was used
func (m *CompressionMetrics) RecordDeltaAlgorithm(algorithm string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	switch algorithm {
	case "xor":
		m.XORDeltaCount++
	case "rsync":
		m.RSyncDeltaCount++
	case "bsdiff":
		m.BSDiffDeltaCount++
	case "auto":
		m.AutoSelectCount++
	}
}

// RecordDictionaryUpdate records dictionary training
func (m *CompressionMetrics) RecordDictionaryUpdate(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.DictionariesTrained = count
	m.LastDictionaryUpdate = time.Now()
}

// RecordLevelAdjustment records an adaptive compression level change
func (m *CompressionMetrics) RecordLevelAdjustment() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.LevelAdjustments++
}

// RecordIncompressibleSkip records when data was skipped due to poor compressibility
func (m *CompressionMetrics) RecordIncompressibleSkip() {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.IncompressibleSkips++
}

// UpdateBaselineCount updates the current baseline count
func (m *CompressionMetrics) UpdateBaselineCount(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.BaselineCount = count
}

// GetCompressionRatio returns the overall compression ratio
func (m *CompressionMetrics) GetCompressionRatio() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.TotalBytesCompressed == 0 {
		return 0
	}
	return float64(m.TotalBytesOriginal) / float64(m.TotalBytesCompressed)
}

// GetDeltaHitRate returns the delta hit rate as a percentage
func (m *CompressionMetrics) GetDeltaHitRate() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	total := m.DeltaHits + m.DeltaMisses
	if total == 0 {
		return 0
	}
	return float64(m.DeltaHits) / float64(total) * 100.0
}

// GetDictionaryEfficiency returns dictionary hit rate as a percentage
func (m *CompressionMetrics) GetDictionaryEfficiency() float64 {
	m.mu.RLock()
	defer m.mu.RUnlock()

	total := m.DictionaryHits + m.DictionaryMisses
	if total == 0 {
		return 0
	}
	return float64(m.DictionaryHits) / float64(total) * 100.0
}

// GetAverageCompressionTime returns average compression duration
func (m *CompressionMetrics) GetAverageCompressionTime() time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.TotalOperations == 0 {
		return 0
	}
	return m.TotalCompressionTime / time.Duration(m.TotalOperations)
}

// GetAverageDecompressionTime returns average decompression duration
func (m *CompressionMetrics) GetAverageDecompressionTime() time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if m.TotalOperations == 0 {
		return 0
	}
	return m.TotalDecompressionTime / time.Duration(m.TotalOperations)
}

// GetSnapshot returns a snapshot of current metrics
func (m *CompressionMetrics) GetSnapshot() map[string]interface{} {
	m.mu.RLock()
	defer m.mu.RUnlock()

	compressionRatio := float64(0)
	if m.TotalBytesCompressed > 0 {
		compressionRatio = float64(m.TotalBytesOriginal) / float64(m.TotalBytesCompressed)
	}

	deltaHitRate := float64(0)
	if m.DeltaHits+m.DeltaMisses > 0 {
		deltaHitRate = float64(m.DeltaHits) / float64(m.DeltaHits+m.DeltaMisses) * 100.0
	}

	dictEfficiency := float64(0)
	if m.DictionaryHits+m.DictionaryMisses > 0 {
		dictEfficiency = float64(m.DictionaryHits) / float64(m.DictionaryHits+m.DictionaryMisses) * 100.0
	}

	avgCompTime := time.Duration(0)
	if m.TotalOperations > 0 {
		avgCompTime = m.TotalCompressionTime / time.Duration(m.TotalOperations)
	}

	return map[string]interface{}{
		// Overall stats
		"total_operations":        m.TotalOperations,
		"compression_ratio":       compressionRatio,
		"bytes_original":          m.TotalBytesOriginal,
		"bytes_compressed":        m.TotalBytesCompressed,
		"bytes_saved":             m.TotalBytesOriginal - m.TotalBytesCompressed,

		// Delta stats
		"delta_hit_rate":          deltaHitRate,
		"delta_hits":              m.DeltaHits,
		"delta_misses":            m.DeltaMisses,
		"baseline_refreshes":      m.BaselineRefreshes,
		"baseline_count":          m.BaselineCount,

		// Dictionary stats
		"dictionary_efficiency":   dictEfficiency,
		"dictionary_hits":         m.DictionaryHits,
		"dictionary_misses":       m.DictionaryMisses,
		"dictionaries_trained":    m.DictionariesTrained,
		"last_dict_update":        m.LastDictionaryUpdate,

		// Algorithm usage
		"xor_delta_count":         m.XORDeltaCount,
		"rsync_delta_count":       m.RSyncDeltaCount,
		"bsdiff_delta_count":      m.BSDiffDeltaCount,
		"auto_select_count":       m.AutoSelectCount,

		// Performance
		"avg_compression_time_ms": avgCompTime.Milliseconds(),
		"fastest_compression_ms":  m.FastestCompression.Milliseconds(),
		"slowest_compression_ms":  m.SlowestCompression.Milliseconds(),

		// Adaptive
		"level_adjustments":       m.LevelAdjustments,
		"incompressible_skips":    m.IncompressibleSkips,
	}
}

// Reset resets all metrics to zero
func (m *CompressionMetrics) Reset() {
	m.mu.Lock()
	defer m.mu.Unlock()

	*m = CompressionMetrics{
		FastestCompression: time.Hour,
	}
}
