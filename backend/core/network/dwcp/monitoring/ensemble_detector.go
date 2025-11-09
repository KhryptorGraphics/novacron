package monitoring

import (
	"fmt"
	"math"
	"strings"

	"go.uber.org/zap"
)

// DetectorResult represents the result from a single detector
type DetectorResult struct {
	Detector  string
	Anomaly   *Anomaly
	Weight    float64
}

// EnsembleDetector combines multiple detectors using weighted voting
type EnsembleDetector struct {
	threshold float64
	logger    *zap.Logger
}

// NewEnsembleDetector creates a new ensemble detector
func NewEnsembleDetector(threshold float64, logger *zap.Logger) *EnsembleDetector {
	if logger == nil {
		logger = zap.NewNop()
	}

	if threshold <= 0 || threshold > 1 {
		threshold = 0.6
	}

	return &EnsembleDetector{
		threshold: threshold,
		logger:    logger,
	}
}

// Aggregate aggregates results from multiple detectors
func (ed *EnsembleDetector) Aggregate(results []DetectorResult, metrics *MetricVector) *Anomaly {
	if len(results) == 0 {
		return nil
	}

	// Count votes and calculate weighted score
	totalWeight := 0.0
	weightedScore := 0.0
	anomalyCount := 0
	detectorNames := []string{}
	metricVotes := make(map[string]float64)

	for _, result := range results {
		totalWeight += result.Weight

		if result.Anomaly != nil {
			anomalyCount++
			detectorNames = append(detectorNames, result.Detector)
			weightedScore += result.Anomaly.Confidence * result.Weight

			// Track which metrics are flagged
			metricVotes[result.Anomaly.MetricName] += result.Weight
		}
	}

	// Need at least 2 detectors to agree for low threshold
	// Or 1 detector with high confidence for high threshold
	if anomalyCount == 0 {
		return nil
	}

	finalScore := weightedScore / totalWeight

	if finalScore < ed.threshold {
		return nil
	}

	// Find the metric with most votes
	maxVotes := 0.0
	consensusMetric := ""
	for metric, votes := range metricVotes {
		if votes > maxVotes {
			maxVotes = votes
			consensusMetric = metric
		}
	}

	// Calculate aggregate expected value and actual value
	expectedValue := 0.0
	actualValue := 0.0
	valueCount := 0

	for _, result := range results {
		if result.Anomaly != nil && result.Anomaly.MetricName == consensusMetric {
			expectedValue += result.Anomaly.Expected
			actualValue += result.Anomaly.Value
			valueCount++
		}
	}

	if valueCount > 0 {
		expectedValue /= float64(valueCount)
		actualValue /= float64(valueCount)
	}

	deviation := math.Abs(actualValue - expectedValue)
	severity := calculateSeverity(finalScore, deviation/expectedValue*100)

	description := fmt.Sprintf(
		"Ensemble of %d/%d detectors (%s) detected anomaly in %s",
		anomalyCount,
		len(results),
		strings.Join(detectorNames, ", "),
		consensusMetric,
	)

	return &Anomaly{
		Timestamp:   metrics.Timestamp,
		MetricName:  consensusMetric,
		Value:       actualValue,
		Expected:    expectedValue,
		Deviation:   deviation,
		Severity:    severity,
		Confidence:  finalScore,
		ModelType:   "ensemble",
		Description: description,
		Context: map[string]interface{}{
			"detectors":      detectorNames,
			"detector_count": anomalyCount,
			"total_detectors": len(results),
			"weighted_score": finalScore,
			"metric_votes":   metricVotes,
		},
	}
}

// CalculateConsensus calculates consensus among detectors
func (ed *EnsembleDetector) CalculateConsensus(results []DetectorResult) float64 {
	if len(results) == 0 {
		return 0
	}

	anomalyCount := 0
	for _, result := range results {
		if result.Anomaly != nil {
			anomalyCount++
		}
	}

	return float64(anomalyCount) / float64(len(results))
}

// GetMajorityVote returns the metric that most detectors flagged
func (ed *EnsembleDetector) GetMajorityVote(results []DetectorResult) string {
	metricVotes := make(map[string]int)

	for _, result := range results {
		if result.Anomaly != nil {
			metricVotes[result.Anomaly.MetricName]++
		}
	}

	maxVotes := 0
	majorityMetric := ""

	for metric, votes := range metricVotes {
		if votes > maxVotes {
			maxVotes = votes
			majorityMetric = metric
		}
	}

	return majorityMetric
}

// GetAverageConfidence calculates average confidence across detectors
func (ed *EnsembleDetector) GetAverageConfidence(results []DetectorResult) float64 {
	if len(results) == 0 {
		return 0
	}

	totalConfidence := 0.0
	count := 0

	for _, result := range results {
		if result.Anomaly != nil {
			totalConfidence += result.Anomaly.Confidence
			count++
		}
	}

	if count == 0 {
		return 0
	}

	return totalConfidence / float64(count)
}

// GetWeightedConfidence calculates weighted average confidence
func (ed *EnsembleDetector) GetWeightedConfidence(results []DetectorResult) float64 {
	totalWeight := 0.0
	weightedConfidence := 0.0

	for _, result := range results {
		totalWeight += result.Weight
		if result.Anomaly != nil {
			weightedConfidence += result.Anomaly.Confidence * result.Weight
		}
	}

	if totalWeight == 0 {
		return 0
	}

	return weightedConfidence / totalWeight
}

// GetDetectorAgreement returns which detectors agree on anomalies
func (ed *EnsembleDetector) GetDetectorAgreement(results []DetectorResult) map[string][]string {
	agreement := make(map[string][]string)

	for _, result := range results {
		if result.Anomaly != nil {
			metric := result.Anomaly.MetricName
			agreement[metric] = append(agreement[metric], result.Detector)
		}
	}

	return agreement
}
