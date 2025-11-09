package metrics

import (
	"sync"
	"time"
)

// NeuromorphicMetrics tracks neuromorphic computing metrics
type NeuromorphicMetrics struct {
	mu sync.RWMutex

	// Spike metrics
	TotalSpikes      int64   `json:"total_spikes"`
	SpikeRate        float64 `json:"spike_rate_hz"`
	AvgSpikeRate     float64 `json:"avg_spike_rate_hz"`
	PeakSpikeRate    float64 `json:"peak_spike_rate_hz"`

	// Synaptic metrics
	SynapticOps      int64   `json:"synaptic_ops"`
	SynapticOpsPerSec float64 `json:"synaptic_ops_per_sec"`

	// Power metrics
	PowerConsumption float64 `json:"power_consumption_mw"`
	AvgPower         float64 `json:"avg_power_mw"`
	PeakPower        float64 `json:"peak_power_mw"`
	EnergyPerInference float64 `json:"energy_per_inference_mj"`

	// Performance metrics
	InferenceLatency float64 `json:"inference_latency_us"`
	AvgLatency       float64 `json:"avg_latency_us"`
	MinLatency       float64 `json:"min_latency_us"`
	MaxLatency       float64 `json:"max_latency_us"`
	Throughput       float64 `json:"throughput_inferences_per_sec"`

	// Accuracy metrics
	Accuracy         float64 `json:"accuracy"`
	Precision        float64 `json:"precision"`
	Recall           float64 `json:"recall"`
	F1Score          float64 `json:"f1_score"`

	// Resource metrics
	NeuronsUtilized  int64   `json:"neurons_utilized"`
	SynapsesUtilized int64   `json:"synapses_utilized"`
	MemoryUsage      int64   `json:"memory_usage_bytes"`

	// Timestamps
	StartTime        time.Time `json:"start_time"`
	LastUpdate       time.Time `json:"last_update"`
	TotalInferences  int64     `json:"total_inferences"`
}

// MetricsCollector collects neuromorphic metrics
type MetricsCollector struct {
	mu              sync.RWMutex
	metrics         *NeuromorphicMetrics
	latencyBuffer   []float64
	spikeRateBuffer []float64
	powerBuffer     []float64
	bufferSize      int
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		metrics: &NeuromorphicMetrics{
			StartTime:  time.Now(),
			LastUpdate: time.Now(),
			MinLatency: 1e9,
		},
		latencyBuffer:   make([]float64, 0),
		spikeRateBuffer: make([]float64, 0),
		powerBuffer:     make([]float64, 0),
		bufferSize:      1000,
	}
}

// RecordSpikes records spike events
func (mc *MetricsCollector) RecordSpikes(count int64, duration float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.metrics.TotalSpikes += count
	mc.metrics.SpikeRate = float64(count) / duration

	// Update spike rate buffer
	mc.spikeRateBuffer = append(mc.spikeRateBuffer, mc.metrics.SpikeRate)
	if len(mc.spikeRateBuffer) > mc.bufferSize {
		mc.spikeRateBuffer = mc.spikeRateBuffer[1:]
	}

	// Calculate average
	sum := 0.0
	peak := 0.0
	for _, rate := range mc.spikeRateBuffer {
		sum += rate
		if rate > peak {
			peak = rate
		}
	}
	mc.metrics.AvgSpikeRate = sum / float64(len(mc.spikeRateBuffer))
	mc.metrics.PeakSpikeRate = peak

	mc.metrics.LastUpdate = time.Now()
}

// RecordInference records an inference event
func (mc *MetricsCollector) RecordInference(latencyUs, powerMw float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.metrics.TotalInferences++
	mc.metrics.InferenceLatency = latencyUs
	mc.metrics.PowerConsumption = powerMw

	// Update latency buffer
	mc.latencyBuffer = append(mc.latencyBuffer, latencyUs)
	if len(mc.latencyBuffer) > mc.bufferSize {
		mc.latencyBuffer = mc.latencyBuffer[1:]
	}

	// Calculate latency stats
	sum := 0.0
	min := 1e9
	max := 0.0
	for _, lat := range mc.latencyBuffer {
		sum += lat
		if lat < min {
			min = lat
		}
		if lat > max {
			max = lat
		}
	}
	mc.metrics.AvgLatency = sum / float64(len(mc.latencyBuffer))
	mc.metrics.MinLatency = min
	mc.metrics.MaxLatency = max

	// Calculate throughput
	elapsed := time.Since(mc.metrics.StartTime).Seconds()
	if elapsed > 0 {
		mc.metrics.Throughput = float64(mc.metrics.TotalInferences) / elapsed
	}

	// Update power buffer
	mc.powerBuffer = append(mc.powerBuffer, powerMw)
	if len(mc.powerBuffer) > mc.bufferSize {
		mc.powerBuffer = mc.powerBuffer[1:]
	}

	// Calculate power stats
	powerSum := 0.0
	powerPeak := 0.0
	for _, p := range mc.powerBuffer {
		powerSum += p
		if p > powerPeak {
			powerPeak = p
		}
	}
	mc.metrics.AvgPower = powerSum / float64(len(mc.powerBuffer))
	mc.metrics.PeakPower = powerPeak

	// Calculate energy per inference
	if mc.metrics.TotalInferences > 0 {
		totalEnergy := mc.metrics.AvgPower * elapsed * 1000 // mJ
		mc.metrics.EnergyPerInference = totalEnergy / float64(mc.metrics.TotalInferences)
	}

	mc.metrics.LastUpdate = time.Now()
}

// RecordAccuracy records accuracy metrics
func (mc *MetricsCollector) RecordAccuracy(accuracy, precision, recall float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.metrics.Accuracy = accuracy
	mc.metrics.Precision = precision
	mc.metrics.Recall = recall

	// Calculate F1 score
	if precision+recall > 0 {
		mc.metrics.F1Score = 2 * (precision * recall) / (precision + recall)
	}

	mc.metrics.LastUpdate = time.Now()
}

// RecordResourceUsage records resource utilization
func (mc *MetricsCollector) RecordResourceUsage(neurons, synapses, memory int64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.metrics.NeuronsUtilized = neurons
	mc.metrics.SynapsesUtilized = synapses
	mc.metrics.MemoryUsage = memory

	mc.metrics.LastUpdate = time.Now()
}

// RecordSynapticOps records synaptic operations
func (mc *MetricsCollector) RecordSynapticOps(ops int64, duration float64) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.metrics.SynapticOps += ops
	mc.metrics.SynapticOpsPerSec = float64(ops) / duration

	mc.metrics.LastUpdate = time.Now()
}

// GetMetrics returns current metrics
func (mc *MetricsCollector) GetMetrics() *NeuromorphicMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	// Create a copy
	metricsCopy := *mc.metrics
	return &metricsCopy
}

// Reset resets all metrics
func (mc *MetricsCollector) Reset() {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.metrics = &NeuromorphicMetrics{
		StartTime:  time.Now(),
		LastUpdate: time.Now(),
		MinLatency: 1e9,
	}
	mc.latencyBuffer = make([]float64, 0)
	mc.spikeRateBuffer = make([]float64, 0)
	mc.powerBuffer = make([]float64, 0)
}

// GetSummary returns a summary of key metrics
func (mc *MetricsCollector) GetSummary() map[string]interface{} {
	metrics := mc.GetMetrics()

	return map[string]interface{}{
		"total_inferences":        metrics.TotalInferences,
		"avg_latency_us":          metrics.AvgLatency,
		"throughput_inf_per_sec":  metrics.Throughput,
		"avg_power_mw":            metrics.AvgPower,
		"energy_per_inf_mj":       metrics.EnergyPerInference,
		"accuracy":                metrics.Accuracy,
		"f1_score":                metrics.F1Score,
		"neurons_utilized":        metrics.NeuronsUtilized,
		"total_spikes":            metrics.TotalSpikes,
		"avg_spike_rate_hz":       metrics.AvgSpikeRate,
		"uptime_seconds":          time.Since(metrics.StartTime).Seconds(),
	}
}

// GetEfficiencyMetrics returns efficiency comparison metrics
func (mc *MetricsCollector) GetEfficiencyMetrics() map[string]interface{} {
	metrics := mc.GetMetrics()

	// Compare to baselines
	gpuPowerMw := 200000.0 // 200W GPU
	gpuLatencyUs := 10000.0 // 10ms
	cnnPowerMw := 5000.0   // 5W CPU
	cnnLatencyUs := 50000.0 // 50ms

	return map[string]interface{}{
		"power_vs_gpu":    gpuPowerMw / metrics.AvgPower,
		"power_vs_cnn":    cnnPowerMw / metrics.AvgPower,
		"latency_vs_gpu":  gpuLatencyUs / metrics.AvgLatency,
		"latency_vs_cnn":  cnnLatencyUs / metrics.AvgLatency,
		"energy_per_inf":  metrics.EnergyPerInference,
		"inferences_per_joule": 1000.0 / metrics.EnergyPerInference,
	}
}
