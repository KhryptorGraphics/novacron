package prediction

import (
	"context"
	"fmt"
	"sync"
	"time"

	"go.uber.org/zap"
)

// AMSTOptimizer integrates PBA predictions with AMST multi-stream TCP
type AMSTOptimizer struct {
	predictionService *PredictionService
	logger            *zap.Logger
	ctx               context.Context
	cancel            context.CancelFunc

	// Current optimization parameters
	currentStreams    int
	currentBufferSize int
	currentChunkSize  int
	mu                sync.RWMutex

	// Optimization history
	optimizationHistory []OptimizationRecord
	maxHistory          int

	// Configuration
	minStreams int
	maxStreams int
	minBuffer  int
	maxBuffer  int
	updateInterval time.Duration
}

// OptimizationRecord tracks optimization decisions
type OptimizationRecord struct {
	Timestamp      time.Time
	Prediction     BandwidthPrediction
	Streams        int
	BufferSize     int
	ChunkSize      int
	Confidence     float64
	Applied        bool
	ResultingThroughput float64
}

// AMSTParameters contains optimized transport parameters
type AMSTParameters struct {
	NumStreams    int
	BufferSize    int
	ChunkSize     int
	PacingRate    int64
	WindowSize    int
	Confidence    float64
	ValidUntil    time.Time
	Reason        string
}

// NewAMSTOptimizer creates a new AMST optimizer
func NewAMSTOptimizer(predictionService *PredictionService, logger *zap.Logger) *AMSTOptimizer {
	ctx, cancel := context.WithCancel(context.Background())

	optimizer := &AMSTOptimizer{
		predictionService:   predictionService,
		logger:              logger,
		ctx:                 ctx,
		cancel:              cancel,
		minStreams:          2,
		maxStreams:          256,
		minBuffer:           16384,  // 16KB
		maxBuffer:           1048576, // 1MB
		updateInterval:      1 * time.Minute,
		optimizationHistory: make([]OptimizationRecord, 0, 100),
		maxHistory:          100,
		// Start with conservative defaults
		currentStreams:    16,
		currentBufferSize: 65536,
		currentChunkSize:  65536,
	}

	return optimizer
}

// Start begins the optimization loop
func (o *AMSTOptimizer) Start() {
	go o.optimizationLoop()
}

// optimizationLoop continuously optimizes AMST parameters
func (o *AMSTOptimizer) optimizationLoop() {
	ticker := time.NewTicker(o.updateInterval)
	defer ticker.Stop()

	for {
		select {
		case <-o.ctx.Done():
			return
		case <-ticker.C:
			o.optimizeParameters()
		}
	}
}

// optimizeParameters updates AMST parameters based on predictions
func (o *AMSTOptimizer) optimizeParameters() {
	// Get current prediction
	prediction := o.predictionService.GetPrediction()
	if prediction == nil {
		o.logger.Warn("No prediction available for optimization")
		return
	}

	// Only apply optimization if confidence is high enough
	if prediction.Confidence < 0.6 {
		o.logger.Info("Prediction confidence too low, skipping optimization",
			zap.Float64("confidence", prediction.Confidence))
		return
	}

	// Calculate optimal parameters
	params := o.calculateOptimalParameters(prediction)

	// Apply parameters if confidence is high
	applied := prediction.Confidence >= 0.8

	// Record optimization decision
	record := OptimizationRecord{
		Timestamp:  time.Now(),
		Prediction: *prediction,
		Streams:    params.NumStreams,
		BufferSize: params.BufferSize,
		ChunkSize:  params.ChunkSize,
		Confidence: prediction.Confidence,
		Applied:    applied,
	}

	o.mu.Lock()
	if applied {
		o.currentStreams = params.NumStreams
		o.currentBufferSize = params.BufferSize
		o.currentChunkSize = params.ChunkSize
	}

	o.optimizationHistory = append(o.optimizationHistory, record)
	if len(o.optimizationHistory) > o.maxHistory {
		o.optimizationHistory = o.optimizationHistory[1:]
	}
	o.mu.Unlock()

	o.logger.Info("AMST parameters optimized",
		zap.Int("streams", params.NumStreams),
		zap.Int("buffer_size", params.BufferSize),
		zap.Int("chunk_size", params.ChunkSize),
		zap.Float64("confidence", prediction.Confidence),
		zap.Bool("applied", applied),
		zap.String("reason", params.Reason))
}

// calculateOptimalParameters computes optimal AMST parameters from prediction
func (o *AMSTOptimizer) calculateOptimalParameters(pred *BandwidthPrediction) AMSTParameters {
	params := AMSTParameters{
		Confidence: pred.Confidence,
		ValidUntil: pred.ValidUntil,
	}

	// Calculate optimal stream count
	// Formula: balance between parallelism and overhead
	// More streams for higher bandwidth, fewer for high latency
	baseStreams := int(pred.PredictedBandwidthMbps / 10) // 1 stream per 10 Mbps

	// Adjust for latency
	latencyFactor := 1.0
	if pred.PredictedLatencyMs > 100 {
		latencyFactor = 0.7 // Reduce streams for high latency
	} else if pred.PredictedLatencyMs < 20 {
		latencyFactor = 1.3 // Increase streams for low latency
	}

	params.NumStreams = int(float64(baseStreams) * latencyFactor)

	// Adjust for packet loss
	if pred.PredictedPacketLoss > 0.01 {
		// High packet loss: reduce streams, increase redundancy
		params.NumStreams = params.NumStreams * 3 / 4
	}

	// Bounds checking
	if params.NumStreams < o.minStreams {
		params.NumStreams = o.minStreams
	}
	if params.NumStreams > o.maxStreams {
		params.NumStreams = o.maxStreams
	}

	// Calculate optimal buffer size
	// Based on bandwidth-delay product plus jitter buffer
	bandwidthBps := pred.PredictedBandwidthMbps * 1000000 / 8 // Convert to bytes/sec
	delaySeconds := pred.PredictedLatencyMs / 1000

	bdp := int(bandwidthBps * delaySeconds)

	// Add jitter buffer
	jitterBuffer := int(pred.PredictedJitterMs * bandwidthBps / 1000)

	params.BufferSize = bdp + jitterBuffer

	// Bounds checking
	if params.BufferSize < o.minBuffer {
		params.BufferSize = o.minBuffer
	}
	if params.BufferSize > o.maxBuffer {
		params.BufferSize = o.maxBuffer
	}

	// Calculate optimal chunk size
	// Larger chunks for high bandwidth, smaller for high packet loss
	params.ChunkSize = params.BufferSize / params.NumStreams

	if pred.PredictedPacketLoss > 0.02 {
		// High packet loss: use smaller chunks for faster recovery
		params.ChunkSize = params.ChunkSize / 2
	}

	// Ensure chunk size is reasonable
	if params.ChunkSize < 8192 {
		params.ChunkSize = 8192 // Min 8KB
	}
	if params.ChunkSize > 131072 {
		params.ChunkSize = 131072 // Max 128KB
	}

	// Calculate pacing rate
	// Slightly below predicted bandwidth to avoid congestion
	params.PacingRate = int64(pred.PredictedBandwidthMbps * 1000000 * 0.95 / 8)

	// Calculate window size
	params.WindowSize = params.BufferSize * params.NumStreams

	// Generate reason
	params.Reason = fmt.Sprintf(
		"Predicted: %.1f Mbps, %.1f ms latency, %.2f%% loss, %d streams optimal",
		pred.PredictedBandwidthMbps,
		pred.PredictedLatencyMs,
		pred.PredictedPacketLoss*100,
		params.NumStreams,
	)

	return params
}

// GetCurrentParameters returns current AMST parameters
func (o *AMSTOptimizer) GetCurrentParameters() AMSTParameters {
	o.mu.RLock()
	defer o.mu.RUnlock()

	prediction := o.predictionService.GetPrediction()
	if prediction == nil {
		return AMSTParameters{
			NumStreams: o.currentStreams,
			BufferSize: o.currentBufferSize,
			ChunkSize:  o.currentChunkSize,
			Confidence: 0,
		}
	}

	return AMSTParameters{
		NumStreams:    o.currentStreams,
		BufferSize:    o.currentBufferSize,
		ChunkSize:     o.currentChunkSize,
		PacingRate:    int64(prediction.PredictedBandwidthMbps * 1000000 * 0.95 / 8),
		WindowSize:    o.currentBufferSize * o.currentStreams,
		Confidence:    prediction.Confidence,
		ValidUntil:    prediction.ValidUntil,
		Reason:        "Current active parameters",
	}
}

// GetOptimizationHistory returns recent optimization decisions
func (o *AMSTOptimizer) GetOptimizationHistory() []OptimizationRecord {
	o.mu.RLock()
	defer o.mu.RUnlock()

	// Return a copy
	history := make([]OptimizationRecord, len(o.optimizationHistory))
	copy(history, o.optimizationHistory)

	return history
}

// UpdateThroughput updates the resulting throughput for the last optimization
func (o *AMSTOptimizer) UpdateThroughput(throughputMbps float64) {
	o.mu.Lock()
	defer o.mu.Unlock()

	if len(o.optimizationHistory) > 0 {
		last := &o.optimizationHistory[len(o.optimizationHistory)-1]
		if last.Applied {
			last.ResultingThroughput = throughputMbps
		}
	}
}

// GetOptimizationEffectiveness calculates how effective optimizations have been
func (o *AMSTOptimizer) GetOptimizationEffectiveness() float64 {
	o.mu.RLock()
	defer o.mu.RUnlock()

	var totalImprovement float64
	var count int

	for _, record := range o.optimizationHistory {
		if record.Applied && record.ResultingThroughput > 0 {
			predicted := record.Prediction.PredictedBandwidthMbps
			actual := record.ResultingThroughput

			// Calculate improvement factor
			improvement := actual / predicted
			totalImprovement += improvement
			count++
		}
	}

	if count == 0 {
		return 1.0 // No data
	}

	return totalImprovement / float64(count)
}

// ShouldAdjustStreams determines if stream count should be changed
func (o *AMSTOptimizer) ShouldAdjustStreams() (bool, int, string) {
	prediction := o.predictionService.GetPrediction()
	if prediction == nil || prediction.Confidence < 0.7 {
		return false, 0, "Insufficient prediction confidence"
	}

	optimalParams := o.calculateOptimalParameters(prediction)

	o.mu.RLock()
	currentStreams := o.currentStreams
	o.mu.RUnlock()

	// Only adjust if difference is significant (>20%)
	diff := float64(optimalParams.NumStreams-currentStreams) / float64(currentStreams)
	if diff > 0.2 || diff < -0.2 {
		return true, optimalParams.NumStreams, optimalParams.Reason
	}

	return false, currentStreams, "Current streams optimal"
}

// Stop stops the optimizer
func (o *AMSTOptimizer) Stop() {
	o.cancel()
}

// PreemptiveOptimization performs proactive optimization before network changes
func (o *AMSTOptimizer) PreemptiveOptimization() *AMSTParameters {
	prediction := o.predictionService.GetPrediction()
	if prediction == nil {
		return nil
	}

	// Only do preemptive optimization if confidence is very high
	if prediction.Confidence < 0.85 {
		return nil
	}

	params := o.calculateOptimalParameters(prediction)

	// Check if predicted conditions differ significantly from current
	o.mu.RLock()
	currentStreams := o.currentStreams
	o.mu.RUnlock()

	diff := float64(params.NumStreams-currentStreams) / float64(currentStreams)
	if diff > 0.15 || diff < -0.15 {
		o.logger.Info("Preemptive optimization recommended",
			zap.Int("current_streams", currentStreams),
			zap.Int("recommended_streams", params.NumStreams),
			zap.Float64("confidence", prediction.Confidence))
		return &params
	}

	return nil
}
