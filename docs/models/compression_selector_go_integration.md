# DWCP Compression Selector - Go Integration Guide

## Overview

This guide explains how to integrate the trained ML compression selector model into the existing Go codebase for real-time inference with <10ms latency.

---

## 1. Integration Architecture

```
┌──────────────────────────────────────────────────────┐
│         DWCP Go Application                          │
├──────────────────────────────────────────────────────┤
│                                                       │
│  ┌────────────────────────────────────┐             │
│  │  Compression Selector (existing)   │             │
│  │  ml_compression_selector.go        │             │
│  └────────────┬───────────────────────┘             │
│               │                                       │
│               ▼                                       │
│  ┌────────────────────────────────────┐             │
│  │  ML Inference Engine (new)         │             │
│  │  - TFLite runtime (C bindings)     │             │
│  │  - XGBoost Go binding              │             │
│  │  - Feature extraction              │             │
│  │  - Ensemble prediction             │             │
│  └────────────┬───────────────────────┘             │
│               │                                       │
│               ▼                                       │
│  ┌────────────────────────────────────┐             │
│  │  Compression Engines               │             │
│  │  - HDE (delta_encoder.go)          │             │
│  │  - AMST (adaptive_compression.go)  │             │
│  │  - Baseline (baseline_sync.go)     │             │
│  └────────────────────────────────────┘             │
│                                                       │
└──────────────────────────────────────────────────────┘
```

---

## 2. Dependencies

### 2.1 Go Libraries

Add to `go.mod`:

```go
require (
    github.com/mattn/go-tflite v1.0.1          // TFLite bindings for Go
    github.com/dmitryikh/leaves v0.0.0-20210121  // XGBoost inference in Go
    github.com/gonum/gonum v0.14.0             // Numerical operations
)
```

Install dependencies:
```bash
go get github.com/mattn/go-tflite
go get github.com/dmitryikh/leaves
go get github.com/gonum/gonum/mat
```

### 2.2 System Dependencies

**TensorFlow Lite C Library**:
```bash
# Download TFLite C library
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-2.13.0.tar.gz
tar -C /usr/local -xzf libtensorflow-cpu-linux-x86_64-2.13.0.tar.gz
ldconfig
```

---

## 3. Model Loading

### 3.1 Model File Structure

```
backend/core/network/dwcp/compression/models/
├── xgboost_model.json          # XGBoost model (JSON format)
├── neural_network.tflite       # Neural network (TFLite format)
├── feature_scaler.json         # Feature standardization parameters
├── label_encoder.json          # Class labels mapping
└── metadata.json               # Model version and config
```

### 3.2 Go Model Loader

**File**: `backend/core/network/dwcp/compression/ml_model.go`

```go
package compression

import (
    "encoding/json"
    "fmt"
    "os"
    "sync"

    "github.com/dmitryikh/leaves"
    "github.com/mattn/go-tflite"
)

// MLCompressionSelector implements ML-based compression selection
type MLCompressionSelector struct {
    // Models
    xgboostModel *leaves.Ensemble
    tfliteModel  *tflite.Model
    interpreter  *tflite.Interpreter

    // Feature processing
    featureScaler *FeatureScaler
    labelEncoder  *LabelEncoder

    // Configuration
    config        *MLSelectorConfig
    mu            sync.RWMutex

    // Ensemble weights
    xgboostWeight float32
    nnWeight      float32

    // Performance tracking
    predictionCount uint64
    totalLatencyMs  uint64
}

// MLSelectorConfig configuration for ML selector
type MLSelectorConfig struct {
    ModelsPath       string
    XGBoostWeight    float32  // Default: 0.7
    NeuralNetWeight  float32  // Default: 0.3
    FallbackEnabled  bool     // Fallback to rule-based if inference fails
    CacheSize        int      // Feature cache size
}

// FeatureScaler standardizes features
type FeatureScaler struct {
    Mean  []float32 `json:"mean"`
    Scale []float32 `json:"scale"`
}

// LabelEncoder maps class indices to compression algorithms
type LabelEncoder struct {
    Classes []string `json:"classes"`  // e.g., ["hde", "amst", "none"]
}

// NewMLCompressionSelector creates a new ML-based compression selector
func NewMLCompressionSelector(config *MLSelectorConfig) (*MLCompressionSelector, error) {
    if config == nil {
        config = &MLSelectorConfig{
            ModelsPath:      "./models",
            XGBoostWeight:   0.7,
            NeuralNetWeight: 0.3,
            FallbackEnabled: true,
            CacheSize:       1000,
        }
    }

    selector := &MLCompressionSelector{
        config:        config,
        xgboostWeight: config.XGBoostWeight,
        nnWeight:      config.NeuralNetWeight,
    }

    // Load XGBoost model
    xgbPath := fmt.Sprintf("%s/xgboost_model.json", config.ModelsPath)
    xgbModel, err := leaves.XGEnsembleFromJSON(xgbPath)
    if err != nil {
        return nil, fmt.Errorf("failed to load XGBoost model: %w", err)
    }
    selector.xgboostModel = &xgbModel

    // Load TFLite model
    tflitePath := fmt.Sprintf("%s/neural_network.tflite", config.ModelsPath)
    model := tflite.NewModelFromFile(tflitePath)
    if model == nil {
        return nil, fmt.Errorf("failed to load TFLite model")
    }
    selector.tfliteModel = model

    // Create interpreter
    options := tflite.NewInterpreterOptions()
    options.SetNumThread(2)  // 2 threads for inference
    interpreter := tflite.NewInterpreter(model, options)
    if interpreter == nil {
        return nil, fmt.Errorf("failed to create TFLite interpreter")
    }
    selector.interpreter = interpreter

    // Allocate tensors
    status := interpreter.AllocateTensors()
    if status != tflite.OK {
        return nil, fmt.Errorf("failed to allocate tensors")
    }

    // Load feature scaler
    scalerPath := fmt.Sprintf("%s/feature_scaler.json", config.ModelsPath)
    if err := selector.loadFeatureScaler(scalerPath); err != nil {
        return nil, fmt.Errorf("failed to load feature scaler: %w", err)
    }

    // Load label encoder
    encoderPath := fmt.Sprintf("%s/label_encoder.json", config.ModelsPath)
    if err := selector.loadLabelEncoder(encoderPath); err != nil {
        return nil, fmt.Errorf("failed to load label encoder: %w", err)
    }

    return selector, nil
}

// loadFeatureScaler loads feature standardization parameters
func (sel *MLCompressionSelector) loadFeatureScaler(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return err
    }

    sel.featureScaler = &FeatureScaler{}
    return json.Unmarshal(data, sel.featureScaler)
}

// loadLabelEncoder loads class label mappings
func (sel *MLCompressionSelector) loadLabelEncoder(path string) error {
    data, err := os.ReadFile(path)
    if err != nil {
        return err
    }

    sel.labelEncoder = &LabelEncoder{}
    return json.Unmarshal(data, sel.labelEncoder)
}

// Close releases model resources
func (sel *MLCompressionSelector) Close() error {
    if sel.interpreter != nil {
        sel.interpreter.Delete()
    }
    if sel.tfliteModel != nil {
        sel.tfliteModel.Delete()
    }
    return nil
}
```

---

## 4. Feature Extraction

### 4.1 Feature Vector Definition

**File**: `backend/core/network/dwcp/compression/ml_features.go`

```go
package compression

import (
    "math"
)

// FeatureVector represents extracted features for ML inference
type FeatureVector struct {
    // Network characteristics (4 features)
    RTTMs                   float32
    JitterMs                float32
    BandwidthMbps           float32
    NetworkQuality          float32  // bandwidth / (rtt + 1)

    // Data characteristics (3 features)
    DataSizeMB              float32
    Entropy                 float32
    CompressibilityScore    float32

    // System state (2 features)
    CPUUsage                float32
    MemoryPressure          float32  // 1 - (available_mb / 10000)

    // Historical performance (5 features)
    HDECompressionRatio     float32
    HDEDeltaHitRate         float32
    AMSTTransferRateMbps    float32
    BaselineCompressionRatio float32
    HDEEfficiency           float32  // hde_ratio * delta_hit_rate / 100
    AMSTEfficiency          float32  // amst_rate / (bandwidth + 1)

    // Categorical (2 features, encoded)
    LinkTypeEncoded         float32  // 0=dc, 1=metro, 2=wan
    RegionEncoded           float32  // 0-N based on region
}

// ExtractFeatures extracts features from current state
func ExtractFeatures(
    networkState *NetworkState,
    dataChars *DataCharacteristics,
    systemState *SystemState,
    historicalPerf *HistoricalPerformance,
) *FeatureVector {

    fv := &FeatureVector{
        // Network
        RTTMs:         float32(networkState.RTTMs),
        JitterMs:      float32(networkState.JitterMs),
        BandwidthMbps: float32(networkState.BandwidthMbps),
        NetworkQuality: float32(networkState.BandwidthMbps) / (float32(networkState.RTTMs) + 1.0),

        // Data
        DataSizeMB:           float32(dataChars.SizeBytes) / (1024.0 * 1024.0),
        Entropy:              float32(dataChars.Entropy),
        CompressibilityScore: float32(dataChars.CompressibilityScore),

        // System
        CPUUsage:       float32(systemState.CPUUsage),
        MemoryPressure: 1.0 - (float32(systemState.MemoryAvailableMB) / 10000.0),

        // Historical
        HDECompressionRatio:     float32(historicalPerf.HDECompressionRatio),
        HDEDeltaHitRate:         float32(historicalPerf.HDEDeltaHitRate),
        AMSTTransferRateMbps:    float32(historicalPerf.AMSTTransferRateMbps),
        BaselineCompressionRatio: float32(historicalPerf.BaselineCompressionRatio),
    }

    // Engineered features
    fv.HDEEfficiency = fv.HDECompressionRatio * fv.HDEDeltaHitRate / 100.0
    fv.AMSTEfficiency = fv.AMSTTransferRateMbps / (fv.BandwidthMbps + 1.0)

    // Encode link type
    switch networkState.LinkType {
    case "dc":
        fv.LinkTypeEncoded = 0
    case "metro":
        fv.LinkTypeEncoded = 1
    case "wan":
        fv.LinkTypeEncoded = 2
    default:
        fv.LinkTypeEncoded = 1  // Default to metro
    }

    // Encode region (simplified)
    fv.RegionEncoded = float32(encodeRegion(networkState.Region))

    return fv
}

// ToSlice converts feature vector to float32 slice
func (fv *FeatureVector) ToSlice() []float32 {
    return []float32{
        fv.RTTMs,
        fv.JitterMs,
        fv.BandwidthMbps,
        fv.NetworkQuality,
        fv.DataSizeMB,
        fv.Entropy,
        fv.CompressibilityScore,
        fv.CPUUsage,
        fv.MemoryPressure,
        fv.HDECompressionRatio,
        fv.HDEDeltaHitRate,
        fv.AMSTTransferRateMbps,
        fv.BaselineCompressionRatio,
        fv.HDEEfficiency,
        fv.AMSTEfficiency,
        fv.LinkTypeEncoded,
        fv.RegionEncoded,
    }
}

// Standardize applies feature scaling
func (fv *FeatureVector) Standardize(scaler *FeatureScaler) []float32 {
    raw := fv.ToSlice()
    standardized := make([]float32, len(raw))

    for i := 0; i < len(raw); i++ {
        standardized[i] = (raw[i] - scaler.Mean[i]) / scaler.Scale[i]
    }

    return standardized
}

func encodeRegion(region string) int {
    // Simplified region encoding
    regionMap := map[string]int{
        "us-east-1":  0,
        "us-west-2":  1,
        "eu-west-1":  2,
        "ap-south-1": 3,
    }
    if code, ok := regionMap[region]; ok {
        return code
    }
    return 0
}
```

---

## 5. Real-Time Inference

### 5.1 Prediction Function

**File**: `backend/core/network/dwcp/compression/ml_inference.go`

```go
package compression

import (
    "context"
    "fmt"
    "time"

    "github.com/dmitryikh/leaves"
)

// CompressionChoice represents the selected compression algorithm
type CompressionChoice string

const (
    CompressionHDE  CompressionChoice = "hde"
    CompressionAMST CompressionChoice = "amst"
    CompressionNone CompressionChoice = "none"
)

// PredictionResult contains prediction and metadata
type PredictionResult struct {
    Choice      CompressionChoice
    Confidence  float32
    Probabilities [3]float32  // [P(hde), P(amst), P(none)]
    LatencyMs   float64
}

// Predict performs ensemble inference to select compression algorithm
func (sel *MLCompressionSelector) Predict(
    ctx context.Context,
    networkState *NetworkState,
    dataChars *DataCharacteristics,
    systemState *SystemState,
    historicalPerf *HistoricalPerformance,
) (*PredictionResult, error) {

    startTime := time.Now()

    // Extract features
    features := ExtractFeatures(networkState, dataChars, systemState, historicalPerf)
    featureSlice := features.ToSlice()

    // XGBoost prediction (no standardization needed)
    xgbProbs, err := sel.predictXGBoost(featureSlice)
    if err != nil {
        if sel.config.FallbackEnabled {
            return sel.fallbackPrediction(features), nil
        }
        return nil, fmt.Errorf("XGBoost inference failed: %w", err)
    }

    // Neural network prediction (with standardization)
    standardizedFeatures := features.Standardize(sel.featureScaler)
    nnProbs, err := sel.predictNeuralNetwork(standardizedFeatures)
    if err != nil {
        if sel.config.FallbackEnabled {
            return sel.fallbackPrediction(features), nil
        }
        return nil, fmt.Errorf("Neural network inference failed: %w", err)
    }

    // Ensemble (weighted average)
    ensembleProbs := [3]float32{
        sel.xgboostWeight*xgbProbs[0] + sel.nnWeight*nnProbs[0],
        sel.xgboostWeight*xgbProbs[1] + sel.nnWeight*nnProbs[1],
        sel.xgboostWeight*xgbProbs[2] + sel.nnWeight*nnProbs[2],
    }

    // Select class with highest probability
    maxIdx := 0
    maxProb := ensembleProbs[0]
    for i := 1; i < 3; i++ {
        if ensembleProbs[i] > maxProb {
            maxProb = ensembleProbs[i]
            maxIdx = i
        }
    }

    // Map to compression choice
    choice := CompressionChoice(sel.labelEncoder.Classes[maxIdx])

    // Compute latency
    latency := time.Since(startTime)

    // Track metrics
    sel.mu.Lock()
    sel.predictionCount++
    sel.totalLatencyMs += uint64(latency.Milliseconds())
    sel.mu.Unlock()

    return &PredictionResult{
        Choice:        choice,
        Confidence:    maxProb,
        Probabilities: ensembleProbs,
        LatencyMs:     float64(latency.Microseconds()) / 1000.0,
    }, nil
}

// predictXGBoost performs XGBoost inference
func (sel *MLCompressionSelector) predictXGBoost(features []float32) ([3]float32, error) {
    // Convert to float64 for leaves library
    featuresF64 := make([]float64, len(features))
    for i, v := range features {
        featuresF64[i] = float64(v)
    }

    // Predict probabilities
    probs := sel.xgboostModel.PredictSingle(featuresF64, 0)  // 0 = use all trees

    // Convert to [3]float32
    result := [3]float32{
        float32(probs[0]),
        float32(probs[1]),
        float32(probs[2]),
    }

    return result, nil
}

// predictNeuralNetwork performs TFLite inference
func (sel *MLCompressionSelector) predictNeuralNetwork(features []float32) ([3]float32, error) {
    // Get input tensor
    inputTensor := sel.interpreter.GetInputTensor(0)
    if inputTensor == nil {
        return [3]float32{}, fmt.Errorf("failed to get input tensor")
    }

    // Copy features to input tensor
    copy(inputTensor.Float32s(), features)

    // Run inference
    status := sel.interpreter.Invoke()
    if status != tflite.OK {
        return [3]float32{}, fmt.Errorf("inference failed with status: %v", status)
    }

    // Get output tensor
    outputTensor := sel.interpreter.GetOutputTensor(0)
    if outputTensor == nil {
        return [3]float32{}, fmt.Errorf("failed to get output tensor")
    }

    // Extract probabilities
    output := outputTensor.Float32s()
    result := [3]float32{output[0], output[1], output[2]}

    return result, nil
}

// fallbackPrediction provides rule-based fallback
func (sel *MLCompressionSelector) fallbackPrediction(features *FeatureVector) *PredictionResult {
    // Simple rule-based fallback
    var choice CompressionChoice

    if features.RTTMs < 1.0 && features.BandwidthMbps > 500 {
        // Datacenter: prefer speed
        choice = CompressionAMST
    } else if features.HDECompressionRatio > 10 && features.HDEDeltaHitRate > 80 {
        // High compression potential
        choice = CompressionHDE
    } else if features.BandwidthMbps < 100 {
        // Low bandwidth: prefer compression
        choice = CompressionHDE
    } else {
        // Default
        choice = CompressionAMST
    }

    return &PredictionResult{
        Choice:     choice,
        Confidence: 0.5,  // Low confidence for fallback
        LatencyMs:  0.1,  // Fast fallback
    }
}

// GetStats returns inference statistics
func (sel *MLCompressionSelector) GetStats() map[string]interface{} {
    sel.mu.RLock()
    defer sel.mu.RUnlock()

    avgLatency := float64(0)
    if sel.predictionCount > 0 {
        avgLatency = float64(sel.totalLatencyMs) / float64(sel.predictionCount)
    }

    return map[string]interface{}{
        "prediction_count":   sel.predictionCount,
        "avg_latency_ms":     avgLatency,
        "xgboost_weight":     sel.xgboostWeight,
        "neural_net_weight":  sel.nnWeight,
        "fallback_enabled":   sel.config.FallbackEnabled,
    }
}
```

---

## 6. Integration with Existing Selector

### 6.1 Update `ml_compression_selector.go`

```go
package encoding

import (
    "context"

    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/compression"
)

// Update SelectCompression to use ML model
func (cs *CompressionSelector) SelectCompression(
    data []byte,
    mode upgrade.NetworkMode,
) CompressionAlgorithm {

    // If ML selector is enabled and loaded
    if cs.mlSelector != nil {
        result, err := cs.mlSelector.Predict(
            context.Background(),
            cs.getNetworkState(),
            cs.analyzeData(data),
            cs.getSystemState(),
            cs.getHistoricalPerformance(),
        )

        if err == nil && result.Confidence > 0.6 {
            // Use ML prediction if confident
            switch result.Choice {
            case compression.CompressionHDE:
                return CompressionZstdMax  // Map to existing algorithm
            case compression.CompressionAMST:
                return CompressionLZ4
            case compression.CompressionNone:
                return CompressionNone
            }
        }
    }

    // Fallback to existing rule-based logic
    return cs.selectByMode(cs.analyzeData(data), mode)
}
```

---

## 7. Performance Optimization

### 7.1 Feature Caching

```go
// FeatureCache caches recently computed features
type FeatureCache struct {
    cache map[string]*CachedFeature
    mu    sync.RWMutex
    maxSize int
}

type CachedFeature struct {
    Features  *FeatureVector
    Timestamp time.Time
    TTL       time.Duration
}

func (fc *FeatureCache) Get(key string) (*FeatureVector, bool) {
    fc.mu.RLock()
    defer fc.mu.RUnlock()

    cached, ok := fc.cache[key]
    if !ok {
        return nil, false
    }

    // Check if expired
    if time.Since(cached.Timestamp) > cached.TTL {
        return nil, false
    }

    return cached.Features, true
}
```

### 7.2 Batch Prediction

For bulk transfers:

```go
func (sel *MLCompressionSelector) PredictBatch(
    ctx context.Context,
    states []*TransferState,
) ([]*PredictionResult, error) {

    results := make([]*PredictionResult, len(states))

    // Parallel prediction
    sem := make(chan struct{}, 4)  // Limit concurrency
    errChan := make(chan error, len(states))

    for i, state := range states {
        sem <- struct{}{}
        go func(idx int, s *TransferState) {
            defer func() { <-sem }()

            result, err := sel.Predict(ctx, s.Network, s.Data, s.System, s.Historical)
            if err != nil {
                errChan <- err
                return
            }
            results[idx] = result
        }(i, state)
    }

    // Wait for completion
    for i := 0; i < len(states); i++ {
        <-sem
    }

    close(errChan)
    if len(errChan) > 0 {
        return nil, <-errChan
    }

    return results, nil
}
```

---

## 8. Testing

### 8.1 Unit Tests

**File**: `backend/core/network/dwcp/compression/ml_inference_test.go`

```go
package compression

import (
    "context"
    "testing"
    "time"
)

func TestMLCompressionSelector_Predict(t *testing.T) {
    config := &MLSelectorConfig{
        ModelsPath:      "./testdata/models",
        XGBoostWeight:   0.7,
        NeuralNetWeight: 0.3,
        FallbackEnabled: true,
    }

    selector, err := NewMLCompressionSelector(config)
    if err != nil {
        t.Fatalf("Failed to create selector: %v", err)
    }
    defer selector.Close()

    // Test case 1: Datacenter link (should prefer AMST)
    networkState := &NetworkState{
        RTTMs:         0.5,
        BandwidthMbps: 1000,
        LinkType:      "dc",
    }
    dataChars := &DataCharacteristics{
        SizeBytes:            1024 * 1024,
        Entropy:              0.5,
        CompressibilityScore: 0.5,
    }
    systemState := &SystemState{
        CPUUsage:           0.3,
        MemoryAvailableMB:  8192,
    }
    historicalPerf := &HistoricalPerformance{
        HDECompressionRatio:     15.0,
        HDEDeltaHitRate:         85.0,
        AMSTTransferRateMbps:    950.0,
        BaselineCompressionRatio: 3.0,
    }

    result, err := selector.Predict(
        context.Background(),
        networkState, dataChars, systemState, historicalPerf,
    )
    if err != nil {
        t.Errorf("Prediction failed: %v", err)
    }

    if result.Confidence < 0.6 {
        t.Errorf("Low confidence: %f", result.Confidence)
    }

    if result.LatencyMs > 10.0 {
        t.Errorf("Latency too high: %f ms (target: <10ms)", result.LatencyMs)
    }

    t.Logf("Prediction: %s (confidence: %.2f, latency: %.2f ms)",
        result.Choice, result.Confidence, result.LatencyMs)
}

func BenchmarkMLCompressionSelector_Predict(b *testing.B) {
    // ... benchmark code ...
}
```

---

## 9. Deployment

### 9.1 Model Deployment

```bash
# Copy trained models to production
cp backend/core/network/dwcp/compression/training/models/* \
   /opt/dwcp/models/compression/

# Set permissions
chmod 644 /opt/dwcp/models/compression/*
```

### 9.2 Configuration

**File**: `config/compression_ml.yaml`

```yaml
compression_ml:
  enabled: true
  models_path: /opt/dwcp/models/compression
  xgboost_weight: 0.7
  neural_net_weight: 0.3
  fallback_enabled: true
  cache_size: 1000

  # Performance
  inference_timeout_ms: 10
  batch_size: 10

  # Monitoring
  metrics_enabled: true
  log_predictions: false  # Enable for debugging
```

---

## 10. Monitoring

### 10.1 Prometheus Metrics

```go
var (
    mlPredictionLatency = prometheus.NewHistogram(
        prometheus.HistogramOpts{
            Name: "dwcp_ml_prediction_latency_ms",
            Help: "ML compression selector prediction latency",
        },
    )

    mlPredictionTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "dwcp_ml_predictions_total",
            Help: "Total ML predictions",
        },
        []string{"choice", "confidence_bucket"},
    )
)

func init() {
    prometheus.MustRegister(mlPredictionLatency)
    prometheus.MustRegister(mlPredictionTotal)
}
```

---

## 11. Troubleshooting

### 11.1 Common Issues

**Issue**: High inference latency (>10ms)
```
Solution:
- Check CPU throttling
- Reduce TFLite threads to 1
- Enable feature caching
- Use XGBoost-only mode
```

**Issue**: Low prediction confidence
```
Solution:
- Check if features are within training distribution
- Enable fallback mode
- Retrain model with more data
```

**Issue**: Model file not found
```
Solution:
- Verify models_path configuration
- Check file permissions
- Ensure all required files are present
```

---

## 12. Rollback Procedure

If ML selector causes issues:

```go
// Disable ML selector via config
compression_ml:
  enabled: false  # Reverts to rule-based selection

// Or via runtime flag
selector.DisableML()
```

---

## Conclusion

This integration guide enables seamless deployment of the ML compression selector into the DWCP Go codebase with <10ms inference latency, automatic fallback, and comprehensive monitoring.
