package scheduler

import (
	"time"
)

// LifetimePrediction holds the result of a lifetime prediction
type LifetimePrediction struct {
	VMID               string        `json:"vm_id"`
	PredictedRemaining time.Duration `json:"predicted_remaining"`
	Confidence         float64       `json:"confidence"`
	Timestamp          time.Time     `json:"timestamp"`
	Uptime             time.Duration `json:"uptime"`
}

// LifetimeFeatures are the input features for lifetime prediction matching LAVA paper
type LifetimeFeatures struct {
	Zone              string  `json:"zone"`
	VMShape           string  `json:"vm_shape"`
	VMCategory        string  `json:"vm_category"`
	MetadataID        string  `json:"metadata_id"`
	HasSSD            bool    `json:"has_ssd"`
	ProvisioningModel string  `json:"provisioning_model"`
	Priority          string  `json:"priority"`
	AdmissionPolicy   string  `json:"admission_policy"`
	Uptime            float64 `json:"uptime_hours"` // log scale hours
}

// LifetimePredictor interface for lifetime prediction implementations
type LifetimePredictor interface {
	PredictLifetime(features LifetimeFeatures) (LifetimePrediction, error)
	RepredictLifetime(vmID string, uptime time.Duration, features LifetimeFeatures) (LifetimePrediction, error)
	GetModelMetrics() ModelMetrics
}

// ModelMetrics holds model performance metrics
type ModelMetrics struct {
	Precision     float64       `json:"precision"`
	Recall        float64       `json:"recall"`
	F1Score       float64       `json:"f1_score"`
	MedianLatency time.Duration `json:"median_latency"`
	LastTrainedAt time.Time     `json:"last_trained_at"`
}

// StubLifetimePredictor is a stub implementation for testing before ML model ready
type StubLifetimePredictor struct {
	averages map[string]time.Duration // average lifetime per VM shape
}

// NewStubLifetimePredictor creates a new stub predictor with predefined averages
func NewStubLifetimePredictor() *StubLifetimePredictor {
	return &StubLifetimePredictor{
		averages: map[string]time.Duration{
			"n1-standard-1": 2 * time.Hour,
			"n1-standard-2": 4 * time.Hour,
			"n1-standard-4": 8 * time.Hour,
			"n1-standard-8": 24 * time.Hour,
			"n1-highmem-2":  1 * time.Hour,
			"n1-highcpu-2":  30 * time.Minute,
			"custom-1-2048": 6 * time.Hour,
			// default
			"": 1 * time.Hour,
		},
	}
}

// PredictLifetime implements stub prediction based on VM shape
func (s *StubLifetimePredictor) PredictLifetime(features LifetimeFeatures) (LifetimePrediction, error) {
	avg, ok := s.averages[features.VMShape]
	if !ok {
		avg = 1 * time.Hour
	}
	return LifetimePrediction{
		VMID:               "",
		PredictedRemaining: avg,
		Confidence:         0.8,
		Timestamp:          time.Now(),
		Uptime:             time.Duration(features.Uptime * float64(time.Hour)),
	}, nil
}

// RepredictLifetime implements stub reprediction, reduces remaining based on uptime
func (s *StubLifetimePredictor) RepredictLifetime(vmID string, uptime time.Duration, features LifetimeFeatures) (LifetimePrediction, error) {
	pred, err := s.PredictLifetime(features)
	if err != nil {
		return LifetimePrediction{}, err
	}
	upHours := uptime.Hours()
	// Stub logic: reduce prediction based on uptime fraction
	fractionPassed := upHours / pred.PredictedRemaining.Hours()
	if fractionPassed > 1 {
		fractionPassed = 1
	}
	pred.PredictedRemaining = pred.PredictedRemaining * time.Duration((1-fractionPassed)*float64(time.Second))
	pred.VMID = vmID
	pred.Uptime = uptime
	pred.Timestamp = time.Now()
	pred.Confidence = 0.7 // lower for reprediction
	return pred, nil
}

// GetModelMetrics returns stub metrics
func (s *StubLifetimePredictor) GetModelMetrics() ModelMetrics {
	return ModelMetrics{
		Precision:     0.85,
		Recall:        0.75,
		F1Score:       0.80,
		MedianLatency: 10 * time.Microsecond,
		LastTrainedAt: time.Now().Add(-24 * time.Hour),
	}
}
