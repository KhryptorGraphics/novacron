# DWCP Bandwidth Predictor LSTM Training Guide

## Overview

This guide explains how to train the DWCP Bandwidth Predictor LSTM model to achieve ≥98% accuracy for network bandwidth, latency, packet loss, and jitter prediction.

## Model Architecture

### Enhanced LSTM with Attention Mechanism

The model uses a sophisticated architecture designed for high-accuracy time series prediction:

```
Input Layer (batch, 30, features)
    ↓
LSTM Layer 1: 256 units, dropout=0.3, recurrent_dropout=0.2
    ↓
Batch Normalization
    ↓
LSTM Layer 2: 192 units, dropout=0.3, recurrent_dropout=0.2
    ↓
Batch Normalization
    ↓
LSTM Layer 3: 128 units, dropout=0.2, recurrent_dropout=0.2
    ↓
Batch Normalization
    ↓
Attention Layer: 128 units (custom attention mechanism)
    ↓
Dense Layer 1: 192 units, ReLU, dropout=0.3
    ↓
Dense Layer 2: 128 units, ReLU, dropout=0.2
    ↓
Skip Connection (concatenate attention + dense2)
    ↓
Dense Layer 3: 64 units, ReLU, dropout=0.2
    ↓
Output Layer: 4 units (bandwidth, latency, packet_loss, jitter)
```

**Total Parameters:** ~800K-1M (depending on feature count)

## Feature Engineering

### Primary Features (Required)
1. `throughput_mbps` - Network throughput
2. `rtt_ms` - Round-trip time
3. `packet_loss` - Packet loss ratio
4. `jitter_ms` - Network jitter

### Temporal Features
5. `time_of_day` - Hour of day (0-23)
6. `day_of_week` - Day of week (0-6)

### Network Context Features
7. `congestion_window` - TCP congestion window size
8. `queue_depth` - Queue depth
9. `retransmits` - Number of retransmissions
10. `bytes_total` - Total bytes transferred
11. `bytes_ratio` - Transmit/receive ratio
12. `network_load` - Computed network load indicator
13. `congestion_score` - Composite congestion metric

### Categorical Encodings
14. `region_encoded` - Geographic region
15. `az_encoded` - Availability zone
16. `link_type_encoded` - Link type (dc/metro/wan)
17. `dwcp_mode_encoded` - DWCP mode
18. `network_tier_encoded` - Network tier
19. `transport_type_encoded` - Transport protocol

### Rolling Statistics (Windows: 3, 5, 10)
- `throughput_rolling_mean_N`
- `rtt_rolling_mean_N`
- `packet_loss_rolling_mean_N`
- `jitter_rolling_mean_N`

### Rate of Change Features
- `throughput_change`
- `rtt_change`
- `packet_loss_change`

**Total Features:** 40-50 depending on data availability

## Data Requirements

### Input Data Schema

CSV file with the following columns:

```csv
timestamp,region,az,link_type,node_id,peer_id,rtt_ms,jitter_ms,
throughput_mbps,bytes_tx,bytes_rx,packet_loss,retransmits,
congestion_window,queue_depth,dwcp_mode,network_tier,
transport_type,time_of_day,day_of_week,bandwidth_mbps,latency_ms
```

### Minimum Dataset Size
- **Recommended:** 15,000+ samples (as available)
- **Minimum:** 10,000 samples for reasonable performance
- **Optimal:** 50,000+ samples for best accuracy

### Data Quality Requirements
1. No missing critical fields (throughput, rtt, packet_loss, jitter)
2. Temporal ordering preserved
3. No duplicate timestamps
4. Values within reasonable ranges:
   - Bandwidth: 0-10000 Mbps
   - Latency: 0-1000 ms
   - Packet loss: 0-1 (0-100%)
   - Jitter: 0-500 ms

## Training Configuration

### Hyperparameters (Optimized for ≥98% Accuracy)

```json
{
  "window_size": 30,
  "batch_size": 64,
  "epochs": 200,
  "learning_rate": 0.001,
  "early_stopping_patience": 20,
  "validation_split": 0.15,
  "test_split": 0.15,
  "optimizer": "AdamW",
  "loss": "mse",
  "regularization": "L2(0.001)"
}
```

### Learning Rate Schedule

**Cosine Decay with Restarts:**
- Initial LR: 0.001
- Decay steps: 1000
- T_mul: 2.0 (double restart period each time)
- M_mul: 0.9 (90% of previous peak)
- Alpha: 1e-6 (minimum LR)

This schedule provides:
- Warm-up phase for stable initialization
- Gradual decay for fine-tuning
- Periodic restarts to escape local minima

### Data Split Strategy

**Temporal Split (No Shuffling):**
- Train: 70% (first chronological portion)
- Validation: 15% (middle chronological portion)
- Test: 15% (last chronological portion)

This preserves temporal dependencies and prevents data leakage.

## Training Procedure

### Prerequisites

Install Python dependencies:

```bash
# Ubuntu/Debian
sudo apt install python3-venv python3-pip

# Create virtual environment
python3 -m venv training_venv
source training_venv/bin/activate

# Install dependencies
pip install tensorflow==2.15.0
pip install numpy pandas scikit-learn
pip install tf2onnx onnx
pip install matplotlib seaborn  # Optional: for plots
```

### Training Command

```bash
cd backend/core/network/dwcp/prediction/training

# Activate environment
source training_venv/bin/activate

# Run training
python3 train_lstm_enhanced.py \
  --data-path /path/to/dwcp_training.csv \
  --output-dir ./checkpoints/bandwidth_predictor_enhanced \
  --epochs 200 \
  --batch-size 64 \
  --learning-rate 0.001 \
  --window-size 30 \
  --validation-split 0.15 \
  --test-split 0.15 \
  --seed 42
```

### Expected Training Time

- **CPU Only:** 30-60 minutes (15K samples)
- **GPU (CUDA):** 10-20 minutes (15K samples)
- **GPU (Metal/M1):** 15-25 minutes (15K samples)

### Monitoring Training

The training script provides:
1. **Console output:** Real-time loss and metrics
2. **TensorBoard logs:** `tensorboard --logdir checkpoints/bandwidth_predictor_enhanced/tensorboard`
3. **CSV logs:** `checkpoints/bandwidth_predictor_enhanced/training_log.csv`
4. **Checkpoints:** Best model saved automatically

## Target Metrics

### Success Criteria (ANY of the following)

1. **Correlation Coefficient ≥ 0.98**
   - Measures linear relationship between predicted and actual
   - Primary target metric

2. **Accuracy ≥ 98%**
   - Calculated as: (1 - normalized_MAE) × 100%
   - Alternative success criterion

3. **MAPE ≤ 5%**
   - Mean Absolute Percentage Error
   - Indicates <5% average prediction error

### Per-Target Metrics

The model is evaluated on 4 prediction targets:
- Bandwidth (throughput_mbps)
- Latency (rtt_ms)
- Packet Loss
- Jitter (jitter_ms)

**Report includes:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Correlation Coefficient
- Accuracy Percentage

### Overall Performance

**Average metrics across all 4 targets:**
- Average Correlation: ≥0.98 (target)
- Average MAPE: ≤5% (target)
- Average Accuracy: ≥98% (target)

## Output Artifacts

### Generated Files

```
checkpoints/bandwidth_predictor_enhanced/
├── best_model.keras                          # Keras model (best weights)
├── bandwidth_lstm_vYYYYMMDD_HHMMSS.onnx     # ONNX model for production
├── model_metadata_vYYYYMMDD_HHMMSS.json     # Model metadata
├── training_history_vYYYYMMDD_HHMMSS.json   # Training metrics
├── TRAINING_REPORT.json                      # Comprehensive report
├── training_log.csv                          # Per-epoch metrics
└── tensorboard/                              # TensorBoard logs
```

### ONNX Model Export

**Format:** ONNX Opset 13
**Input Shape:** `(batch, 30, features)`
**Output Shape:** `(batch, 4)`

**Compatible with:**
- Go ONNX Runtime (`github.com/yalue/onnxruntime_go`)
- C++ ONNX Runtime
- Python ONNX Runtime
- TensorRT (for GPU inference)

## Integration with Go Predictor

### Loading the Model

```go
import (
    ort "github.com/yalue/onnxruntime_go"
)

// Initialize ONNX runtime
ort.SetSharedLibraryPath("/usr/local/lib/libonnxruntime.so")
err := ort.InitializeEnvironment()

// Load model
session, err := ort.NewDynamicAdvancedSession(
    "bandwidth_lstm_v20251114_123456.onnx",
    []string{"input"},
    []string{"output"},
    options
)
```

### Running Inference

```go
// Prepare input: [1, 30, features]
inputData := make([]float32, 30 * featureCount)
// ... populate with normalized features ...

// Create tensor
tensor, err := ort.NewTensor(
    []int64{1, 30, int64(featureCount)},
    inputData
)

// Run inference
outputs, err := session.Run([]ort.Value{tensor})

// Parse output: [bandwidth, latency, packet_loss, jitter]
predictions := outputs[0].GetData().([]float32)
```

### Normalization

**IMPORTANT:** Apply the same normalization as training:

1. Load scaler parameters from `model_metadata_*.json`
2. Apply StandardScaler transformation:
   ```
   normalized = (value - mean) / scale
   ```
3. Denormalize predictions:
   ```
   actual_value = (normalized_pred * scale) + mean
   ```

## Troubleshooting

### Low Accuracy (<95%)

**Causes and Solutions:**

1. **Insufficient Training Data**
   - Solution: Collect more samples (target 50K+)
   - Generate synthetic data if needed

2. **Poor Feature Quality**
   - Check for missing values
   - Verify feature distributions
   - Add more context features

3. **Inadequate Training**
   - Increase epochs to 300
   - Reduce early stopping patience to 15
   - Try lower learning rate (0.0005)

4. **Overfitting**
   - Increase dropout rates (0.4-0.5)
   - Add more L2 regularization (0.01)
   - Reduce model capacity

### Training Instability

**Symptoms:** Loss spikes, NaN values

**Solutions:**
- Reduce learning rate to 0.0005
- Increase batch size to 128
- Check for outliers in data
- Add gradient clipping: `clipnorm=1.0`

### Memory Issues

**For large datasets:**
- Reduce batch size to 32
- Use `tf.data` pipeline with prefetching
- Enable mixed precision: `tf.keras.mixed_precision`
- Process in chunks

### ONNX Export Fails

**Common issues:**
- Ensure tf2onnx is installed
- Check TensorFlow compatibility
- Try opset 11 instead of 13
- Verify custom layers are exportable

## Performance Optimization

### For Training
1. **Use GPU:** 3-5x faster than CPU
2. **Mixed Precision:** `policy = tf.keras.mixed_precision.Policy('mixed_float16')`
3. **Data Pipeline:** Use `tf.data` with prefetch
4. **Batch Size:** Increase to GPU memory limit

### For Inference
1. **ONNX Runtime:** Faster than TensorFlow
2. **Quantization:** INT8 for edge deployment
3. **Batch Inference:** Process multiple samples
4. **TensorRT:** For NVIDIA GPU deployment

## Maintenance

### Model Retraining Schedule

**Recommended:**
- Weekly: Incremental retraining with new data
- Monthly: Full retraining from scratch
- Quarterly: Architecture review and optimization

### Performance Monitoring

Track in production:
- Average prediction error
- Inference latency
- Model confidence scores
- Drift detection metrics

### Model Versioning

Use semantic versioning:
- `v1.0.0` - Initial model
- `v1.1.0` - Minor architecture changes
- `v2.0.0` - Major architecture changes

## References

### Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "LSTM Networks for Sequence Prediction" (Hochreiter & Schmidhuber, 1997)
- "Time Series Forecasting with Deep Learning" (Lim & Zohren, 2021)

### Documentation
- TensorFlow: https://tensorflow.org/guide/keras/lstm
- ONNX Runtime: https://onnxruntime.ai/docs/
- Scikit-learn: https://scikit-learn.org/stable/

## Contact

For issues or questions:
- File issue in project repository
- Refer to DWCP documentation
- Contact ML team

---

**Last Updated:** 2025-11-14
**Model Version:** v1.0 (Enhanced LSTM with Attention)
**Target Accuracy:** ≥98%
