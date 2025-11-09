package metrics

import (
	"math"
	"sync"
	"time"
)

// MLMetrics tracks ML model performance
type MLMetrics struct {
	TrainingMetrics   map[string][]float64
	InferenceMetrics  map[string][]float64
	DriftMetrics      map[string]float64
	LastUpdated       time.Time
	mu                sync.RWMutex
}

func NewMLMetrics() *MLMetrics {
	return &MLMetrics{
		TrainingMetrics:  make(map[string][]float64),
		InferenceMetrics: make(map[string][]float64),
		DriftMetrics:     make(map[string]float64),
	}
}

func (m *MLMetrics) RecordTrainingMetric(name string, value float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.TrainingMetrics[name] = append(m.TrainingMetrics[name], value)
	m.LastUpdated = time.Now()
}

func (m *MLMetrics) RecordInferenceMetric(name string, value float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.InferenceMetrics[name] = append(m.InferenceMetrics[name], value)
	m.LastUpdated = time.Now()
}

func (m *MLMetrics) DetectDrift(current, baseline []float64) float64 {
	if len(current) == 0 || len(baseline) == 0 {
		return 0
	}

	// KL divergence for drift detection
	drift := 0.0
	for i := range current {
		if i < len(baseline) && baseline[i] > 0 && current[i] > 0 {
			drift += current[i] * math.Log(current[i]/baseline[i])
		}
	}
	
	m.mu.Lock()
	m.DriftMetrics["kl_divergence"] = drift
	m.mu.Unlock()
	
	return drift
}
