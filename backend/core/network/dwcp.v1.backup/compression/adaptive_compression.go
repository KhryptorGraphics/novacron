package compression

import (
	"runtime"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"
	"go.uber.org/zap"
)

// AdaptiveCompressor manages adaptive compression level selection
type AdaptiveCompressor struct {
	// Current compression level
	currentLevel    zstd.EncoderLevel
	levelMutex      sync.RWMutex

	// Adaptive configuration
	config          *AdaptiveConfig
	logger          *zap.Logger

	// Performance metrics
	recentRatios    []float64
	recentDurations []time.Duration
	metricsMutex    sync.Mutex
	metricsWindow   int

	// CPU monitoring
	lastCPUCheck    time.Time
	cpuAvailable    float64
}

// AdaptiveConfig configuration for adaptive compression
type AdaptiveConfig struct {
	Enabled             bool    `json:"enabled" yaml:"enabled"`
	MinCompressionRatio float64 `json:"min_compression_ratio" yaml:"min_compression_ratio"`
	TargetRatio         float64 `json:"target_ratio" yaml:"target_ratio"`
	MaxCPUUtilization   float64 `json:"max_cpu_utilization" yaml:"max_cpu_utilization"`
	AdjustInterval      time.Duration `json:"adjust_interval" yaml:"adjust_interval"`
	RatioWindow         int     `json:"ratio_window" yaml:"ratio_window"`
}

// DefaultAdaptiveConfig returns sensible defaults
func DefaultAdaptiveConfig() *AdaptiveConfig {
	return &AdaptiveConfig{
		Enabled:             true,
		MinCompressionRatio: 1.1,  // Skip compression if ratio < 1.1
		TargetRatio:         10.0, // Target 10x compression
		MaxCPUUtilization:   0.8,  // Use up to 80% CPU
		AdjustInterval:      30 * time.Second,
		RatioWindow:         100,  // Track last 100 operations
	}
}

// NewAdaptiveCompressor creates a new adaptive compressor
func NewAdaptiveCompressor(config *AdaptiveConfig, logger *zap.Logger) *AdaptiveCompressor {
	if config == nil {
		config = DefaultAdaptiveConfig()
	}

	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	ac := &AdaptiveCompressor{
		currentLevel:    zstd.SpeedDefault,
		config:          config,
		logger:          logger,
		recentRatios:    make([]float64, 0, config.RatioWindow),
		recentDurations: make([]time.Duration, 0, config.RatioWindow),
		metricsWindow:   config.RatioWindow,
		lastCPUCheck:    time.Now(),
		cpuAvailable:    1.0,
	}

	return ac
}

// SelectCompressionLevel selects optimal compression level based on current conditions
func (ac *AdaptiveCompressor) SelectCompressionLevel() zstd.EncoderLevel {
	if !ac.config.Enabled {
		return zstd.SpeedDefault
	}

	ac.levelMutex.RLock()
	level := ac.currentLevel
	ac.levelMutex.RUnlock()

	// Check if we should adjust level
	if time.Since(ac.lastCPUCheck) > ac.config.AdjustInterval {
		ac.adjustCompressionLevel()
	}

	return level
}

// RecordCompressionResult records metrics from a compression operation
func (ac *AdaptiveCompressor) RecordCompressionResult(ratio float64, duration time.Duration) {
	ac.metricsMutex.Lock()
	defer ac.metricsMutex.Unlock()

	// Add to metrics window
	ac.recentRatios = append(ac.recentRatios, ratio)
	ac.recentDurations = append(ac.recentDurations, duration)

	// Trim to window size
	if len(ac.recentRatios) > ac.metricsWindow {
		ac.recentRatios = ac.recentRatios[1:]
	}
	if len(ac.recentDurations) > ac.metricsWindow {
		ac.recentDurations = ac.recentDurations[1:]
	}
}

// ShouldCompress determines if data should be compressed based on characteristics
func (ac *AdaptiveCompressor) ShouldCompress(dataSize int) bool {
	if !ac.config.Enabled {
		return true
	}

	// Very small data - compression overhead not worth it
	if dataSize < 512 {
		return false
	}

	// Check average compression ratio
	avgRatio := ac.getAverageRatio()
	if avgRatio < ac.config.MinCompressionRatio {
		// Recent compressions are ineffective
		return false
	}

	return true
}

// adjustCompressionLevel dynamically adjusts compression level
func (ac *AdaptiveCompressor) adjustCompressionLevel() {
	ac.metricsMutex.Lock()
	avgRatio := ac.calculateAverageRatio()
	avgDuration := ac.calculateAverageDuration()
	ac.metricsMutex.Unlock()

	// Estimate CPU availability
	cpuCount := float64(runtime.NumCPU())
	var numGoroutine float64
	numGoroutine = float64(runtime.NumGoroutine())
	cpuLoad := numGoroutine / cpuCount
	cpuAvailable := 1.0 - cpuLoad

	if cpuAvailable < 0 {
		cpuAvailable = 0
	}
	if cpuAvailable > 1 {
		cpuAvailable = 1
	}

	ac.cpuAvailable = cpuAvailable

	ac.levelMutex.Lock()
	defer ac.levelMutex.Unlock()

	oldLevel := ac.currentLevel
	newLevel := ac.currentLevel

	// Decision logic
	if avgRatio < ac.config.TargetRatio {
		// Not meeting target ratio - increase compression level if CPU available
		if cpuAvailable > ac.config.MaxCPUUtilization {
			newLevel = ac.increaseLevel(ac.currentLevel)
		}
	} else if avgRatio > ac.config.TargetRatio*1.5 {
		// Exceeding target significantly - can reduce level to save CPU
		newLevel = ac.decreaseLevel(ac.currentLevel)
	}

	// Adjust based on duration
	if avgDuration > 100*time.Millisecond {
		// Taking too long - reduce level
		newLevel = ac.decreaseLevel(newLevel)
	}

	// Apply CPU constraint
	if cpuAvailable < 0.2 {
		// Very high CPU usage - use fastest compression
		newLevel = zstd.SpeedFastest
	}

	if newLevel != oldLevel {
		ac.currentLevel = newLevel
		ac.logger.Info("Adjusted compression level",
			zap.String("old_level", ac.levelToString(oldLevel)),
			zap.String("new_level", ac.levelToString(newLevel)),
			zap.Float64("avg_ratio", avgRatio),
			zap.Duration("avg_duration", avgDuration),
			zap.Float64("cpu_available", cpuAvailable))
	}

	ac.lastCPUCheck = time.Now()
}

func (ac *AdaptiveCompressor) increaseLevel(current zstd.EncoderLevel) zstd.EncoderLevel {
	switch current {
	case zstd.SpeedFastest:
		return zstd.SpeedDefault
	case zstd.SpeedDefault:
		return zstd.SpeedBetterCompression
	case zstd.SpeedBetterCompression:
		return zstd.SpeedBestCompression
	default:
		return zstd.SpeedBestCompression
	}
}

func (ac *AdaptiveCompressor) decreaseLevel(current zstd.EncoderLevel) zstd.EncoderLevel {
	switch current {
	case zstd.SpeedBestCompression:
		return zstd.SpeedBetterCompression
	case zstd.SpeedBetterCompression:
		return zstd.SpeedDefault
	case zstd.SpeedDefault:
		return zstd.SpeedFastest
	default:
		return zstd.SpeedFastest
	}
}

func (ac *AdaptiveCompressor) levelToString(level zstd.EncoderLevel) string {
	switch level {
	case zstd.SpeedFastest:
		return "fastest"
	case zstd.SpeedDefault:
		return "default"
	case zstd.SpeedBetterCompression:
		return "better"
	case zstd.SpeedBestCompression:
		return "best"
	default:
		return "unknown"
	}
}

func (ac *AdaptiveCompressor) getAverageRatio() float64 {
	ac.metricsMutex.Lock()
	defer ac.metricsMutex.Unlock()
	return ac.calculateAverageRatio()
}

func (ac *AdaptiveCompressor) calculateAverageRatio() float64 {
	if len(ac.recentRatios) == 0 {
		return ac.config.TargetRatio
	}

	sum := 0.0
	for _, ratio := range ac.recentRatios {
		sum += ratio
	}
	return sum / float64(len(ac.recentRatios))
}

func (ac *AdaptiveCompressor) calculateAverageDuration() time.Duration {
	if len(ac.recentDurations) == 0 {
		return 0
	}

	sum := time.Duration(0)
	for _, dur := range ac.recentDurations {
		sum += dur
	}
	return sum / time.Duration(len(ac.recentDurations))
}

// GetStats returns adaptive compression statistics
func (ac *AdaptiveCompressor) GetStats() map[string]interface{} {
	ac.levelMutex.RLock()
	currentLevel := ac.currentLevel
	ac.levelMutex.RUnlock()

	ac.metricsMutex.Lock()
	avgRatio := ac.calculateAverageRatio()
	avgDuration := ac.calculateAverageDuration()
	sampleCount := len(ac.recentRatios)
	ac.metricsMutex.Unlock()

	return map[string]interface{}{
		"enabled":          ac.config.Enabled,
		"current_level":    ac.levelToString(currentLevel),
		"avg_ratio":        avgRatio,
		"avg_duration_ms":  avgDuration.Milliseconds(),
		"cpu_available":    ac.cpuAvailable,
		"sample_count":     sampleCount,
		"target_ratio":     ac.config.TargetRatio,
	}
}
