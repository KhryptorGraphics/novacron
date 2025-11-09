// Package ai_threat implements AI-powered threat detection
package ai_threat

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// ThreatLevel represents the severity of a threat
type ThreatLevel int

const (
	ThreatNone ThreatLevel = iota
	ThreatLow
	ThreatMedium
	ThreatHigh
	ThreatCritical
)

// ThreatType represents the type of threat
type ThreatType string

const (
	ThreatAnomaly         ThreatType = "anomaly"
	ThreatMalware         ThreatType = "malware"
	ThreatIntrusion       ThreatType = "intrusion"
	ThreatDataExfiltration ThreatType = "data_exfiltration"
	ThreatDDoS            ThreatType = "ddos"
	ThreatPrivilegeEscalation ThreatType = "privilege_escalation"
	ThreatLateralMovement ThreatType = "lateral_movement"
	ThreatZeroDay         ThreatType = "zero_day"
)

// ThreatEvent represents a detected threat
type ThreatEvent struct {
	ID             string
	Type           ThreatType
	Level          ThreatLevel
	Score          float64
	Confidence     float64
	Source         string
	Target         string
	Timestamp      time.Time
	Description    string
	Indicators     []string
	Metadata       map[string]interface{}
	Mitigations    []string
	DetectionModel string
}

// DetectionModel represents an AI detection model
type DetectionModel interface {
	Predict(ctx context.Context, features []float64) (float64, error)
	Train(ctx context.Context, data []TrainingData) error
	Update(ctx context.Context, feedback []Feedback) error
	GetMetrics() ModelMetrics
}

// TrainingData represents training data for ML models
type TrainingData struct {
	Features []float64
	Label    float64 // 0 = benign, 1 = threat
	Weight   float64
}

// Feedback represents feedback for model improvement
type Feedback struct {
	EventID    string
	TrueLabel  float64
	Predicted  float64
	Timestamp  time.Time
}

// ModelMetrics represents model performance metrics
type ModelMetrics struct {
	Accuracy          float64
	Precision         float64
	Recall            float64
	F1Score           float64
	FalsePositiveRate float64
	FalseNegativeRate float64
	AUC               float64
	Latency           time.Duration
}

// Detector implements AI-powered threat detection
type Detector struct {
	models            map[string]DetectionModel
	threshold         float64
	falsePositiveTarget float64
	anomalyDetector   *AnomalyDetector
	behaviorAnalyzer  *BehaviorAnalyzer
	threatIntel       ThreatIntelligence
	events            []*ThreatEvent
	mu                sync.RWMutex
	detectionLatency  time.Duration
	totalDetections   int64
	truePositives     int64
	falsePositives    int64
	trueNegatives     int64
	falseNegatives    int64
}

// AnomalyDetector detects anomalies using unsupervised learning
type AnomalyDetector struct {
	Algorithm string // "isolation_forest", "autoencoder", "one_class_svm"
	Threshold float64
	mu        sync.RWMutex
	baseline  map[string]*Baseline
}

// Baseline represents normal behavior baseline
type Baseline struct {
	Mean   []float64
	StdDev []float64
	Count  int64
}

// BehaviorAnalyzer analyzes entity behavior
type BehaviorAnalyzer struct {
	Profiles map[string]*BehaviorProfile
	mu       sync.RWMutex
}

// BehaviorProfile represents an entity's behavior profile
type BehaviorProfile struct {
	EntityID      string
	EntityType    string // "user", "vm", "network"
	NormalPattern []float64
	RecentActivity []float64
	LastUpdated   time.Time
	AnomalyScore  float64
}

// ThreatIntelligence provides threat intelligence
type ThreatIntelligence interface {
	CheckIndicator(ctx context.Context, indicator string) (bool, float64, error)
	GetThreatScore(ctx context.Context, source string) (float64, error)
	UpdateFeed(ctx context.Context) error
}

// NewDetector creates a new AI threat detector
func NewDetector(threshold, falsePositiveTarget float64) *Detector {
	return &Detector{
		models:              make(map[string]DetectionModel),
		threshold:           threshold,
		falsePositiveTarget: falsePositiveTarget,
		anomalyDetector: &AnomalyDetector{
			Algorithm: "isolation_forest",
			Threshold: 0.7,
			baseline:  make(map[string]*Baseline),
		},
		behaviorAnalyzer: &BehaviorAnalyzer{
			Profiles: make(map[string]*BehaviorProfile),
		},
		events:           make([]*ThreatEvent, 0),
		detectionLatency: 500 * time.Millisecond,
	}
}

// RegisterModel registers a detection model
func (d *Detector) RegisterModel(name string, model DetectionModel) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.models[name] = model
}

// SetThreatIntel sets the threat intelligence source
func (d *Detector) SetThreatIntel(intel ThreatIntelligence) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.threatIntel = intel
}

// Detect analyzes data and detects threats
func (d *Detector) Detect(ctx context.Context, data *DetectionData) (*ThreatEvent, error) {
	startTime := time.Now()
	defer func() {
		d.detectionLatency = time.Since(startTime)
	}()

	// Extract features
	features := d.extractFeatures(data)

	// Ensemble prediction from multiple models
	predictions := make(map[string]float64)
	d.mu.RLock()
	for name, model := range d.models {
		score, err := model.Predict(ctx, features)
		if err != nil {
			d.mu.RUnlock()
			return nil, fmt.Errorf("model %s prediction failed: %w", name, err)
		}
		predictions[name] = score
	}
	d.mu.RUnlock()

	// Anomaly detection
	anomalyScore := d.detectAnomaly(features, data.EntityID)

	// Behavior analysis
	behaviorScore := d.analyzeBehavior(data)

	// Threat intelligence check
	intelScore := 0.0
	if d.threatIntel != nil {
		for _, indicator := range data.Indicators {
			if match, score, err := d.threatIntel.CheckIndicator(ctx, indicator); err == nil && match {
				intelScore = math.Max(intelScore, score)
			}
		}
	}

	// Ensemble scoring
	finalScore := d.ensembleScore(predictions, anomalyScore, behaviorScore, intelScore)

	// Check threshold
	if finalScore < d.threshold {
		d.mu.Lock()
		d.trueNegatives++
		d.mu.Unlock()
		return nil, nil
	}

	// Create threat event
	event := &ThreatEvent{
		ID:             fmt.Sprintf("threat-%d", time.Now().UnixNano()),
		Type:           d.classifyThreat(finalScore, predictions),
		Level:          d.calculateThreatLevel(finalScore),
		Score:          finalScore,
		Confidence:     d.calculateConfidence(predictions),
		Source:         data.Source,
		Target:         data.Target,
		Timestamp:      time.Now(),
		Description:    d.generateDescription(finalScore, predictions),
		Indicators:     data.Indicators,
		Metadata:       data.Metadata,
		Mitigations:    d.recommendMitigations(finalScore, predictions),
		DetectionModel: "ensemble",
	}

	// Store event
	d.mu.Lock()
	d.events = append(d.events, event)
	d.totalDetections++
	d.truePositives++ // Assume true positive, will be corrected by feedback
	d.mu.Unlock()

	return event, nil
}

// DetectionData represents data for threat detection
type DetectionData struct {
	EntityID   string
	EntityType string
	Source     string
	Target     string
	Indicators []string
	Metadata   map[string]interface{}
	Timestamp  time.Time
}

// extractFeatures extracts features from detection data
func (d *Detector) extractFeatures(data *DetectionData) []float64 {
	features := make([]float64, 0, 20)

	// Time-based features
	hour := float64(data.Timestamp.Hour())
	dayOfWeek := float64(data.Timestamp.Weekday())
	features = append(features, hour, dayOfWeek)

	// Indicator count
	features = append(features, float64(len(data.Indicators)))

	// Add more domain-specific features from metadata
	if data.Metadata != nil {
		if networkTraffic, ok := data.Metadata["network_traffic"].(float64); ok {
			features = append(features, networkTraffic)
		}
		if cpuUsage, ok := data.Metadata["cpu_usage"].(float64); ok {
			features = append(features, cpuUsage)
		}
		if memoryUsage, ok := data.Metadata["memory_usage"].(float64); ok {
			features = append(features, memoryUsage)
		}
		if connectionCount, ok := data.Metadata["connection_count"].(float64); ok {
			features = append(features, connectionCount)
		}
	}

	// Pad to fixed size
	for len(features) < 20 {
		features = append(features, 0.0)
	}

	return features
}

// detectAnomaly detects anomalies using statistical methods
func (d *Detector) detectAnomaly(features []float64, entityID string) float64 {
	d.anomalyDetector.mu.Lock()
	defer d.anomalyDetector.mu.Unlock()

	baseline, exists := d.anomalyDetector.baseline[entityID]
	if !exists {
		// Initialize baseline
		baseline = &Baseline{
			Mean:   make([]float64, len(features)),
			StdDev: make([]float64, len(features)),
			Count:  1,
		}
		copy(baseline.Mean, features)
		d.anomalyDetector.baseline[entityID] = baseline
		return 0.0
	}

	// Calculate z-scores
	anomalyScore := 0.0
	for i, feature := range features {
		if i >= len(baseline.Mean) {
			break
		}

		zScore := 0.0
		if baseline.StdDev[i] > 0 {
			zScore = math.Abs((feature - baseline.Mean[i]) / baseline.StdDev[i])
		}

		anomalyScore += zScore
	}

	anomalyScore /= float64(len(features))

	// Update baseline (exponential moving average)
	alpha := 0.1
	for i, feature := range features {
		if i >= len(baseline.Mean) {
			break
		}
		baseline.Mean[i] = alpha*feature + (1-alpha)*baseline.Mean[i]
		variance := math.Pow(feature-baseline.Mean[i], 2)
		baseline.StdDev[i] = math.Sqrt(alpha*variance + (1-alpha)*math.Pow(baseline.StdDev[i], 2))
	}
	baseline.Count++

	return math.Min(anomalyScore/10.0, 1.0) // Normalize to 0-1
}

// analyzeBehavior analyzes entity behavior
func (d *Detector) analyzeBehavior(data *DetectionData) float64 {
	d.behaviorAnalyzer.mu.RLock()
	defer d.behaviorAnalyzer.mu.RUnlock()

	profile, exists := d.behaviorAnalyzer.Profiles[data.EntityID]
	if !exists {
		return 0.0
	}

	return profile.AnomalyScore
}

// ensembleScore combines scores from multiple models
func (d *Detector) ensembleScore(predictions map[string]float64, anomalyScore, behaviorScore, intelScore float64) float64 {
	// Weighted average
	weights := map[string]float64{
		"ml_model":  0.4,
		"anomaly":   0.25,
		"behavior":  0.2,
		"intel":     0.15,
	}

	score := 0.0
	totalWeight := 0.0

	for modelName, prediction := range predictions {
		weight := weights["ml_model"] / float64(len(predictions))
		score += prediction * weight
		totalWeight += weight
	}

	score += anomalyScore * weights["anomaly"]
	score += behaviorScore * weights["behavior"]
	score += intelScore * weights["intel"]
	totalWeight += weights["anomaly"] + weights["behavior"] + weights["intel"]

	if totalWeight > 0 {
		score /= totalWeight
	}

	return score
}

// classifyThreat classifies the type of threat
func (d *Detector) classifyThreat(score float64, predictions map[string]float64) ThreatType {
	// Simple heuristic - can be enhanced with more sophisticated classification
	if score >= 0.95 {
		return ThreatZeroDay
	} else if score >= 0.85 {
		return ThreatIntrusion
	} else if score >= 0.75 {
		return ThreatMalware
	}
	return ThreatAnomaly
}

// calculateThreatLevel calculates threat severity level
func (d *Detector) calculateThreatLevel(score float64) ThreatLevel {
	switch {
	case score >= 0.9:
		return ThreatCritical
	case score >= 0.75:
		return ThreatHigh
	case score >= 0.6:
		return ThreatMedium
	default:
		return ThreatLow
	}
}

// calculateConfidence calculates prediction confidence
func (d *Detector) calculateConfidence(predictions map[string]float64) float64 {
	if len(predictions) == 0 {
		return 0.0
	}

	// Calculate variance of predictions
	mean := 0.0
	for _, score := range predictions {
		mean += score
	}
	mean /= float64(len(predictions))

	variance := 0.0
	for _, score := range predictions {
		variance += math.Pow(score-mean, 2)
	}
	variance /= float64(len(predictions))

	// Lower variance = higher confidence
	confidence := 1.0 - math.Min(variance, 1.0)
	return confidence
}

// generateDescription generates threat description
func (d *Detector) generateDescription(score float64, predictions map[string]float64) string {
	return fmt.Sprintf("AI-detected threat with score %.2f from ensemble of %d models", score, len(predictions))
}

// recommendMitigations recommends mitigation actions
func (d *Detector) recommendMitigations(score float64, predictions map[string]float64) []string {
	mitigations := []string{}

	if score >= 0.9 {
		mitigations = append(mitigations, "Immediate isolation required")
		mitigations = append(mitigations, "Escalate to security team")
		mitigations = append(mitigations, "Collect forensics data")
	} else if score >= 0.75 {
		mitigations = append(mitigations, "Increase monitoring")
		mitigations = append(mitigations, "Block suspicious connections")
	} else {
		mitigations = append(mitigations, "Continue monitoring")
		mitigations = append(mitigations, "Update threat intelligence")
	}

	return mitigations
}

// ProvideFeedback provides feedback for model improvement
func (d *Detector) ProvideFeedback(eventID string, trueLabel float64) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// Find event
	var event *ThreatEvent
	for _, e := range d.events {
		if e.ID == eventID {
			event = e
			break
		}
	}

	if event == nil {
		return fmt.Errorf("event not found: %s", eventID)
	}

	// Update metrics
	if trueLabel == 1.0 && event.Score >= d.threshold {
		// True positive - already counted
	} else if trueLabel == 0.0 && event.Score >= d.threshold {
		// False positive
		d.falsePositives++
		d.truePositives--
	} else if trueLabel == 1.0 && event.Score < d.threshold {
		// False negative
		d.falseNegatives++
		d.trueNegatives--
	}

	// Create feedback for models
	feedback := Feedback{
		EventID:   eventID,
		TrueLabel: trueLabel,
		Predicted: event.Score,
		Timestamp: time.Now(),
	}

	// Update models
	ctx := context.Background()
	for _, model := range d.models {
		model.Update(ctx, []Feedback{feedback})
	}

	return nil
}

// GetMetrics returns threat detection metrics
func (d *Detector) GetMetrics() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	totalPredictions := d.truePositives + d.falsePositives + d.trueNegatives + d.falseNegatives

	accuracy := 0.0
	precision := 0.0
	recall := 0.0
	falsePositiveRate := 0.0

	if totalPredictions > 0 {
		accuracy = float64(d.truePositives+d.trueNegatives) / float64(totalPredictions)
	}

	if d.truePositives+d.falsePositives > 0 {
		precision = float64(d.truePositives) / float64(d.truePositives+d.falsePositives)
	}

	if d.truePositives+d.falseNegatives > 0 {
		recall = float64(d.truePositives) / float64(d.truePositives+d.falseNegatives)
	}

	if d.falsePositives+d.trueNegatives > 0 {
		falsePositiveRate = float64(d.falsePositives) / float64(d.falsePositives+d.trueNegatives)
	}

	return map[string]interface{}{
		"total_detections":     d.totalDetections,
		"true_positives":       d.truePositives,
		"false_positives":      d.falsePositives,
		"true_negatives":       d.trueNegatives,
		"false_negatives":      d.falseNegatives,
		"accuracy":             accuracy,
		"precision":            precision,
		"recall":               recall,
		"false_positive_rate":  falsePositiveRate,
		"detection_latency_ms": d.detectionLatency.Milliseconds(),
		"registered_models":    len(d.models),
	}
}
