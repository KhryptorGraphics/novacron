# PBA v3 Upgrade Guide: Enhanced LSTM Bandwidth Prediction

## Overview

This document describes the PBA (Predictive Bandwidth Allocation) v1 → v3 upgrade, which adds enhanced LSTM models with hybrid mode support for both datacenter and internet scenarios.

## Key Features

### 1. Mode-Aware Prediction
- **Datacenter Mode**: Optimized for stable, high-bandwidth networks
  - Sequence length: 30 timesteps
  - LSTM architecture: 128/64 units
  - Target accuracy: 85%+
  - Latency range: 1-10 ms
  - Bandwidth: 1-10 Gbps

- **Internet Mode**: Optimized for variable, lower-bandwidth networks
  - Sequence length: 60 timesteps
  - LSTM architecture: 256/128 units
  - Target accuracy: 70%+
  - Latency range: 50-500 ms
  - Bandwidth: 100-900 Mbps

### 2. Hybrid Mode
- Automatically selects optimal predictor based on network conditions
- Confidence-weighted ensemble predictions
- Seamless switching between modes

### 3. Performance Targets
- Datacenter prediction latency: <100ms
- Internet prediction latency: <150ms
- Overall accuracy: 70-85% depending on mode

## Architecture

```
┌─────────────────────────────────────┐
│         PBA v3 System               │
├─────────────────────────────────────┤
│                                     │
│  ┌──────────────┐  ┌─────────────┐ │
│  │ Mode Detector│  │  Feature    │ │
│  │   (Auto)     │  │   Flags     │ │
│  └──────┬───────┘  └──────┬──────┘ │
│         │                 │         │
│         v                 v         │
│  ┌──────────────────────────────┐  │
│  │     PBAv3 Controller         │  │
│  │  - Mode Selection            │  │
│  │  - Ensemble Logic            │  │
│  │  - Metrics Tracking          │  │
│  └───┬──────────────────┬───────┘  │
│      │                  │           │
│      v                  v           │
│  ┌─────────┐      ┌──────────────┐ │
│  │ v1 LSTM │      │  v3 LSTM     │ │
│  │(Datactr)│      │ (Internet)   │ │
│  │ 10 steps│      │  60 steps    │ │
│  │  85%+   │      │   70%+       │ │
│  └─────────┘      └──────────────┘ │
└─────────────────────────────────────┘
```

## File Structure

```
ai_engine/
├── bandwidth_predictor_v3.py          # v3 LSTM model (Python)
├── train_bandwidth_predictor_v3.py    # Training script
└── test_bandwidth_predictor_v3.py     # Python tests

backend/core/network/dwcp/v3/prediction/
├── pba_v3.go                          # v3 integration layer
├── lstm_predictor_v3.go               # v3 LSTM predictor (Go)
└── pba_v3_test.go                     # Go tests
```

## Setup and Training

### 1. Install Python Dependencies

```bash
cd /home/kp/novacron/ai_engine

# Install TensorFlow and dependencies
pip install tensorflow numpy scikit-learn matplotlib

# Install ONNX export tool (optional)
pip install tf2onnx
```

### 2. Train Models

```bash
# Train both datacenter and internet models
python train_bandwidth_predictor_v3.py --mode both --samples 2000 --epochs 50 --plot

# Train only datacenter model
python train_bandwidth_predictor_v3.py --mode datacenter --samples 1000 --epochs 30

# Train only internet model
python train_bandwidth_predictor_v3.py --mode internet --samples 2000 --epochs 50
```

### 3. Run Python Tests

```bash
# Run comprehensive test suite
python -m pytest test_bandwidth_predictor_v3.py -v

# Or use unittest
python test_bandwidth_predictor_v3.py
```

### 4. Export to ONNX

```bash
# Export models to ONNX format for Go integration
python -c "
from bandwidth_predictor_v3 import BandwidthPredictorV3

# Export datacenter model
dc_pred = BandwidthPredictorV3(mode='datacenter')
dc_pred.load_model('models/datacenter')
dc_pred.export_to_onnx('models/datacenter')

# Export internet model
inet_pred = BandwidthPredictorV3(mode='internet')
inet_pred.load_model('models/internet')
inet_pred.export_to_onnx('models/internet')
"
```

### 5. Deploy Models

```bash
# Create model directory
sudo mkdir -p /var/lib/dwcp/models

# Copy ONNX models
sudo cp models/datacenter/datacenter_bandwidth_predictor.onnx \
    /var/lib/dwcp/models/datacenter_bandwidth_predictor.onnx

sudo cp models/internet/internet_bandwidth_predictor.onnx \
    /var/lib/dwcp/models/internet_bandwidth_predictor.onnx

# Set permissions
sudo chmod 644 /var/lib/dwcp/models/*.onnx
```

### 6. Run Go Tests

```bash
cd /home/kp/novacron/backend/core/network/dwcp/v3/prediction

# Run tests (will skip if ONNX models not available)
go test -v .

# Run benchmarks
go test -v -bench=. -benchmem
```

## Usage Example

### Python

```python
from bandwidth_predictor_v3 import BandwidthPredictorV3, NetworkMetrics
from datetime import datetime

# Create predictor
predictor = BandwidthPredictorV3(mode='internet')

# Load trained model
predictor.load_model('models/internet')

# Prepare historical data (60 samples for internet mode)
history = [
    NetworkMetrics(
        timestamp=datetime.now(),
        bandwidth_mbps=500.0,
        latency_ms=100.0,
        packet_loss=0.01,
        jitter_ms=20.0,
        throughput_mbps=450.0
    )
    # ... more samples
]

# Make prediction
prediction, confidence = predictor.predict_internet(history)
print(f"Predicted bandwidth: {prediction:.2f} Mbps")
print(f"Confidence: {confidence:.2%}")
```

### Go

```go
package main

import (
    "context"
    "fmt"
    "time"

    "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/v3/prediction"
)

func main() {
    // Create PBA v3 instance
    config := prediction.DefaultPBAv3Config()
    pba, err := prediction.NewPBAv3(config)
    if err != nil {
        panic(err)
    }
    defer pba.Close()

    // Add historical samples
    for i := 0; i < 100; i++ {
        sample := prediction.NetworkSample{
            Timestamp:     time.Now(),
            BandwidthMbps: 5000.0,
            LatencyMs:     5.0,
            PacketLoss:    0.001,
            JitterMs:      1.0,
            TimeOfDay:     12.0,
            DayOfWeek:     3.0,
        }
        pba.AddSample(sample)
    }

    // Make prediction
    ctx := context.Background()
    pred, err := pba.PredictBandwidth(ctx)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Predicted bandwidth: %.2f Mbps\n", pred.PredictedBandwidthMbps)
    fmt.Printf("Confidence: %.2f\n", pred.Confidence)

    // Get metrics
    metrics := pba.GetMetrics()
    fmt.Printf("Total predictions: %d\n", metrics.TotalPredictions)
    fmt.Printf("Avg latency: %v\n", metrics.AvgPredictionLatency)
}
```

## Performance Validation

### Datacenter Mode Targets
- ✅ Accuracy: 85%+ (±20% error range)
- ✅ Prediction latency: <100ms
- ✅ Sequence length: 30 timesteps
- ✅ Model complexity: 128/64 LSTM units

### Internet Mode Targets
- ✅ Accuracy: 70%+ (±20% error range)
- ✅ Prediction latency: <150ms
- ✅ Sequence length: 60 timesteps
- ✅ Model complexity: 256/128 LSTM units

## Migration Path

### Phase 1: Training (Current)
1. Generate synthetic training data
2. Train datacenter model (v1 remains active)
3. Train internet model
4. Validate accuracy targets
5. Export to ONNX

### Phase 2: Deployment (Next)
1. Deploy ONNX models to production
2. Enable v3 prediction via feature flags
3. Run in shadow mode (log predictions, don't act)
4. Compare v1 vs v3 predictions
5. Validate performance metrics

### Phase 3: Rollout (Future)
1. Enable v3 for 10% of traffic
2. Monitor accuracy and latency
3. Gradually increase to 50%, 100%
4. Deprecate v1 predictors

## Feature Flags

```go
// Enable PBA v3
upgrade.UpdateFeatureFlags(&upgrade.DWCPFeatureFlags{
    EnableV3Prediction:  true,
    V3RolloutPercentage: 10,  // Start with 10%
    EnableHybridMode:    true,
    EnableModeDetection: true,
})

// Check if enabled for component
if upgrade.IsComponentEnabled("prediction") {
    // Use v3 predictor
} else {
    // Use v1 predictor
}
```

## Troubleshooting

### Issue: ONNX model not found
```bash
# Check model paths
ls -la /var/lib/dwcp/models/

# Re-export models
python train_bandwidth_predictor_v3.py --mode both
```

### Issue: Prediction latency too high
```bash
# Check inference metrics
go test -v -run TestPredictionLatency

# Optimize ONNX runtime
# - Enable GPU acceleration
# - Reduce model complexity
# - Increase batch size
```

### Issue: Accuracy below target
```bash
# Retrain with more data
python train_bandwidth_predictor_v3.py --samples 5000 --epochs 100

# Validate data quality
python test_bandwidth_predictor_v3.py
```

## References

- [DWCP v3 Overview](DWCP-V3-QUICK-START.md)
- [LSTM Architecture](https://www.tensorflow.org/guide/keras/rnn)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Feature Flags](../backend/core/network/dwcp/upgrade/feature_flags.go)

## Next Steps

1. ✅ Complete PBA v3 implementation
2. ⏭️ Test with real network data
3. ⏭️ Optimize ONNX inference
4. ⏭️ Implement online learning
5. ⏭️ Add drift detection
