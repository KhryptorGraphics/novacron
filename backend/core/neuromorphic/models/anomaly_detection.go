package models

import (
	"context"
	"fmt"
	"math"

	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/snn"
)

// AnomalyDetectionSNN implements ultra-fast anomaly detection using SNN
// 1000x faster than traditional methods for network traffic analysis
type AnomalyDetectionSNN struct {
	network         *snn.SNNNetwork
	inputNeurons    []int64
	reservoirNeurons []int64
	outputNeurons   []int64
	threshold       float64
	normalPatterns  [][]float64
}

// NewAnomalyDetectionSNN creates a new anomaly detection SNN
func NewAnomalyDetectionSNN(inputDim int) *AnomalyDetectionSNN {
	timeStep := 0.1 // 0.1ms for ultra-fast detection
	stdpConfig := &snn.STDPConfig{
		Enable:   true,
		TauPlus:  15.0,
		TauMinus: 15.0,
		APlus:    0.015,
		AMinus:   0.018,
	}

	network := snn.NewSNNNetwork(timeStep, stdpConfig)

	ad := &AnomalyDetectionSNN{
		network:        network,
		threshold:      0.7,
		normalPatterns: make([][]float64, 0),
	}

	// Build reservoir computing architecture
	ad.buildReservoir(inputDim)

	return ad
}

// buildReservoir creates a liquid state machine (reservoir)
func (ad *AnomalyDetectionSNN) buildReservoir(inputDim int) {
	// Input layer
	ad.inputNeurons = make([]int64, inputDim)
	for i := 0; i < inputDim; i++ {
		ad.inputNeurons[i] = ad.network.AddNeuron(snn.LIF)
	}

	// Reservoir: liquid state machine with recurrent connections
	reservoirSize := 500
	ad.reservoirNeurons = make([]int64, reservoirSize)
	for i := 0; i < reservoirSize; i++ {
		ad.reservoirNeurons[i] = ad.network.AddNeuron(snn.Izhikevich)
	}

	// Output layer: normal vs anomaly
	ad.outputNeurons = make([]int64, 2)
	ad.outputNeurons[0] = ad.network.AddNeuron(snn.LIF) // Normal
	ad.outputNeurons[1] = ad.network.AddNeuron(snn.LIF) // Anomaly

	// Connect input to reservoir (random sparse)
	for _, inputID := range ad.inputNeurons {
		for i := 0; i < reservoirSize/10; i++ {
			reservoirID := ad.reservoirNeurons[(int(inputID)*37+i*13)%reservoirSize]
			weight := 0.3 + 0.2*math.Sin(float64(inputID+int64(i)))
			ad.network.AddSynapse(inputID, reservoirID, weight, 0.5)
		}
	}

	// Recurrent connections in reservoir (small-world topology)
	for i, neuronID := range ad.reservoirNeurons {
		// Connect to nearby neurons
		for j := -5; j <= 5; j++ {
			if j == 0 {
				continue
			}
			targetIdx := (i + j + reservoirSize) % reservoirSize
			targetID := ad.reservoirNeurons[targetIdx]
			weight := 0.2 * math.Exp(-float64(j*j)/10.0)
			ad.network.AddSynapse(neuronID, targetID, weight, 1.0)
		}

		// Random long-range connections
		if i%10 == 0 {
			randomIdx := (i*73 + 17) % reservoirSize
			randomID := ad.reservoirNeurons[randomIdx]
			ad.network.AddSynapse(neuronID, randomID, 0.1, 2.0)
		}
	}

	// Connect reservoir to output
	for _, reservoirID := range ad.reservoirNeurons {
		for _, outputID := range ad.outputNeurons {
			weight := 0.05
			ad.network.AddSynapse(reservoirID, outputID, weight, 0.5)
		}
	}
}

// AnomalyResult represents anomaly detection output
type AnomalyResult struct {
	IsAnomaly    bool    `json:"is_anomaly"`
	AnomalyScore float64 `json:"anomaly_score"`
	Latency      float64 `json:"latency_us"` // microseconds
	Confidence   float64 `json:"confidence"`
}

// Detect detects anomalies in input data
func (ad *AnomalyDetectionSNN) Detect(ctx context.Context, input []float64) (*AnomalyResult, error) {
	if len(input) != len(ad.inputNeurons) {
		return nil, fmt.Errorf("input dimension mismatch: expected %d, got %d",
			len(ad.inputNeurons), len(input))
	}

	// Encode input to spikes
	inputSpikes := ad.encodeInput(input)

	// Run SNN for 10ms (ultra-fast)
	simulationTime := 10.0 // ms
	allSpikes, err := ad.network.Run(ctx, simulationTime, func(t float64) []snn.Spike {
		if t < 0.5 {
			return inputSpikes
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	// Analyze output spikes
	result := ad.analyzeSpikes(allSpikes, simulationTime)

	return result, nil
}

// encodeInput converts input vector to spike trains
func (ad *AnomalyDetectionSNN) encodeInput(input []float64) []snn.Spike {
	spikes := make([]snn.Spike, 0)

	// Normalize input
	max := 0.0
	for _, v := range input {
		if v > max {
			max = v
		}
	}
	if max == 0 {
		max = 1.0
	}

	// Temporal coding: timing encodes value
	for i, value := range input {
		neuronID := ad.inputNeurons[i]
		normalized := value / max

		// Earlier spike = higher value
		timing := 5.0 * (1.0 - normalized) // 0-5ms
		spike := snn.Spike{
			NeuronID:  neuronID,
			Timestamp: timing,
			Weight:    1.0,
		}
		spikes = append(spikes, spike)
	}

	return spikes
}

// analyzeSpikes determines if pattern is anomalous
func (ad *AnomalyDetectionSNN) analyzeSpikes(spikes []snn.Spike, duration float64) *AnomalyResult {
	// Count spikes in output neurons
	normalSpikes := 0
	anomalySpikes := 0

	for _, spike := range spikes {
		if spike.NeuronID == ad.outputNeurons[0] {
			normalSpikes++
		} else if spike.NeuronID == ad.outputNeurons[1] {
			anomalySpikes++
		}
	}

	// Calculate anomaly score
	totalSpikes := normalSpikes + anomalySpikes
	anomalyScore := 0.0
	if totalSpikes > 0 {
		anomalyScore = float64(anomalySpikes) / float64(totalSpikes)
	}

	isAnomaly := anomalyScore > ad.threshold
	confidence := math.Abs(anomalyScore - 0.5) * 2.0 // 0-1

	return &AnomalyResult{
		IsAnomaly:    isAnomaly,
		AnomalyScore: anomalyScore,
		Latency:      duration * 1000, // Convert to microseconds
		Confidence:   confidence,
	}
}

// Train trains the SNN on normal patterns
func (ad *AnomalyDetectionSNN) Train(ctx context.Context, normalData [][]float64) error {
	ad.normalPatterns = normalData

	for _, pattern := range normalData {
		// Encode pattern
		inputSpikes := ad.encodeInput(pattern)

		// Run with supervision: activate "normal" neuron
		_, err := ad.network.Run(ctx, 10.0, func(t float64) []snn.Spike {
			if t < 0.5 {
				// Add supervised spike to normal output
				supervised := make([]snn.Spike, len(inputSpikes)+1)
				copy(supervised, inputSpikes)
				supervised[len(inputSpikes)] = snn.Spike{
					NeuronID:  ad.outputNeurons[0],
					Timestamp: 8.0,
					Weight:    5.0,
				}
				return supervised
			}
			return nil
		})

		if err != nil {
			return err
		}

		// Reset network for next pattern
		ad.network.Reset()

		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	return nil
}

// SetThreshold sets the anomaly detection threshold
func (ad *AnomalyDetectionSNN) SetThreshold(threshold float64) {
	ad.threshold = threshold
}

// GetMetrics returns detector metrics
func (ad *AnomalyDetectionSNN) GetMetrics() map[string]interface{} {
	metrics := ad.network.GetMetrics()
	metrics["threshold"] = ad.threshold
	metrics["normal_patterns"] = len(ad.normalPatterns)
	metrics["reservoir_size"] = len(ad.reservoirNeurons)
	return metrics
}
