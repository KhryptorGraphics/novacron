package models

import (
	"context"
	"fmt"
	"math"

	"github.com/khryptorgraphics/novacron/backend/core/neuromorphic/snn"
)

// ObjectDetectionSNN implements real-time object detection using SNN
type ObjectDetectionSNN struct {
	network        *snn.SNNNetwork
	inputNeurons   []int64
	hiddenNeurons  []int64
	outputNeurons  []int64
	classes        []string
	imageWidth     int
	imageHeight    int
}

// NewObjectDetectionSNN creates a new object detection SNN
func NewObjectDetectionSNN(imageWidth, imageHeight int, classes []string) *ObjectDetectionSNN {
	timeStep := 1.0 // 1ms
	stdpConfig := &snn.STDPConfig{
		Enable:   true,
		TauPlus:  20.0,
		TauMinus: 20.0,
		APlus:    0.01,
		AMinus:   0.012,
	}

	network := snn.NewSNNNetwork(timeStep, stdpConfig)

	od := &ObjectDetectionSNN{
		network:     network,
		classes:     classes,
		imageWidth:  imageWidth,
		imageHeight: imageHeight,
	}

	// Build network architecture
	od.buildNetwork()

	return od
}

// buildNetwork constructs the SNN architecture
func (od *ObjectDetectionSNN) buildNetwork() {
	// Input layer: one neuron per pixel (or downsampled)
	inputSize := od.imageWidth * od.imageHeight
	od.inputNeurons = make([]int64, inputSize)
	for i := 0; i < inputSize; i++ {
		od.inputNeurons[i] = od.network.AddNeuron(snn.LIF)
	}

	// Hidden layers: convolutional-like structure
	hiddenSize := 256
	od.hiddenNeurons = make([]int64, hiddenSize)
	for i := 0; i < hiddenSize; i++ {
		od.hiddenNeurons[i] = od.network.AddNeuron(snn.Izhikevich)
	}

	// Output layer: one neuron per class
	od.outputNeurons = make([]int64, len(od.classes))
	for i := 0; i < len(od.classes); i++ {
		od.outputNeurons[i] = od.network.AddNeuron(snn.LIF)
	}

	// Connect input to hidden (sparse connectivity)
	for _, inputID := range od.inputNeurons {
		// Random sparse connectivity (10% of hidden neurons)
		for i := 0; i < hiddenSize/10; i++ {
			hiddenID := od.hiddenNeurons[i*10+int(inputID)%10]
			weight := 0.5 + 0.5*math.Sin(float64(inputID+int64(i)))
			od.network.AddSynapse(inputID, hiddenID, weight, 1.0)
		}
	}

	// Connect hidden to output (fully connected)
	for _, hiddenID := range od.hiddenNeurons {
		for _, outputID := range od.outputNeurons {
			weight := 0.1 + 0.1*math.Cos(float64(hiddenID+outputID))
			od.network.AddSynapse(hiddenID, outputID, weight, 1.0)
		}
	}
}

// DetectionResult represents object detection output
type DetectionResult struct {
	Class      string  `json:"class"`
	Confidence float64 `json:"confidence"`
	BoundingBox BoundingBox `json:"bounding_box"`
	Latency    float64 `json:"latency_ms"`
}

// BoundingBox represents detection bounding box
type BoundingBox struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

// Detect performs object detection on an image
func (od *ObjectDetectionSNN) Detect(ctx context.Context, image [][]float64) ([]*DetectionResult, error) {
	if len(image) != od.imageHeight || len(image[0]) != od.imageWidth {
		return nil, fmt.Errorf("image size mismatch: expected %dx%d", od.imageWidth, od.imageHeight)
	}

	// Encode image to spikes (rate coding)
	inputSpikes := od.encodeImage(image)

	// Run SNN for 100ms
	simulationTime := 100.0 // ms
	allSpikes, err := od.network.Run(ctx, simulationTime, func(t float64) []snn.Spike {
		// Return input spikes at t=0
		if t < 1.0 {
			return inputSpikes
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	// Decode output spikes to detections
	detections := od.decodeSpikes(allSpikes, simulationTime)

	return detections, nil
}

// encodeImage converts image pixels to spike trains (rate coding)
func (od *ObjectDetectionSNN) encodeImage(image [][]float64) []snn.Spike {
	spikes := make([]snn.Spike, 0)

	for y := 0; y < od.imageHeight; y++ {
		for x := 0; x < od.imageWidth; x++ {
			pixel := image[y][x]
			neuronID := od.inputNeurons[y*od.imageWidth+x]

			// Higher pixel intensity = higher spike rate
			numSpikes := int(pixel * 10) // 0-10 spikes

			for i := 0; i < numSpikes; i++ {
				spike := snn.Spike{
					NeuronID:  neuronID,
					Timestamp: float64(i) * 2.0, // Spread over 20ms
					Weight:    1.0,
				}
				spikes = append(spikes, spike)
			}
		}
	}

	return spikes
}

// decodeSpikes converts output spikes to detection results
func (od *ObjectDetectionSNN) decodeSpikes(spikes []snn.Spike, duration float64) []*DetectionResult {
	// Count spikes per output neuron
	spikeCounts := make(map[int64]int)
	for _, spike := range spikes {
		for i, outputID := range od.outputNeurons {
			if spike.NeuronID == outputID {
				spikeCounts[int64(i)]++
			}
		}
	}

	// Convert to detections
	detections := make([]*DetectionResult, 0)
	for classIdx, count := range spikeCounts {
		confidence := float64(count) / (duration / 1000.0) / 100.0 // Normalize to 0-1

		if confidence > 0.5 { // Threshold
			detection := &DetectionResult{
				Class:      od.classes[classIdx],
				Confidence: confidence,
				BoundingBox: BoundingBox{
					X:      0,
					Y:      0,
					Width:  od.imageWidth,
					Height: od.imageHeight,
				},
				Latency: duration,
			}
			detections = append(detections, detection)
		}
	}

	return detections
}

// Train trains the SNN on labeled data
func (od *ObjectDetectionSNN) Train(ctx context.Context, images [][][]float64, labels []int) error {
	for i, image := range images {
		// Encode image
		inputSpikes := od.encodeImage(image)

		// Create target output (supervised STDP)
		targetClass := labels[i]
		targetNeuron := od.outputNeurons[targetClass]

		// Run simulation
		_, err := od.network.Run(ctx, 100.0, func(t float64) []snn.Spike {
			if t < 1.0 {
				// Add supervised spike to target neuron
				supervised := make([]snn.Spike, len(inputSpikes)+1)
				copy(supervised, inputSpikes)
				supervised[len(inputSpikes)] = snn.Spike{
					NeuronID:  targetNeuron,
					Timestamp: 90.0, // Late spike for supervision
					Weight:    10.0,  // Strong spike
				}
				return supervised
			}
			return nil
		})

		if err != nil {
			return err
		}

		// Check context
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
	}

	return nil
}

// GetMetrics returns model metrics
func (od *ObjectDetectionSNN) GetMetrics() map[string]interface{} {
	metrics := od.network.GetMetrics()
	metrics["classes"] = len(od.classes)
	metrics["input_neurons"] = len(od.inputNeurons)
	metrics["hidden_neurons"] = len(od.hiddenNeurons)
	metrics["output_neurons"] = len(od.outputNeurons)
	return metrics
}
