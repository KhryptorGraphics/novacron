# Bandwidth Predictor LSTM Training Guide

## Overview

This guide provides step-by-step instructions for training the Bandwidth Predictor LSTM model to achieve ≥98% accuracy for DWCP (Distributed Weighted Consensus Protocol) network optimization.

## Target Metrics

The model must achieve:
- **Correlation Coefficient** ≥ 0.98 (bandwidth prediction accuracy)
- **MAPE** (Mean Absolute Percentage Error) < 5%
- **Overall Accuracy** ≥ 98%

## Architecture

### PyTorch LSTM Model

```
Input: [batch_size, 20 timesteps, 9 features]
  ↓
LSTM Layer 1 (256 units, dropout=0.3)
  ↓
Batch Normalization
  ↓
LSTM Layer 2 (128 units, dropout=0.3)
  ↓
Batch Normalization
  ↓
LSTM Layer 3 (64 units)
  ↓
Dense Layer 1 (128 units, ReLU)
  ↓
Dropout (0.3)
  ↓
Batch Normalization
  ↓
Dense Layer 2 (64 units, ReLU)
  ↓
Dropout (0.21)
  ↓
Dense Layer 3 (32 units, ReLU)
  ↓
Output Layer (4 units)
  ↓
Output: [bandwidth_mbps, latency_ms, packet_loss, jitter_ms]
```

**Total Parameters:** ~1.2M trainable parameters

## Input Features (9)

1. **throughput_mbps** - Current network throughput
2. **rtt_ms** - Round-trip time
3. **packet_loss** - Packet loss ratio (0-1)
4. **jitter_ms** - Latency variance
5. **time_of_day** - Hour of day (0-23)
6. **day_of_week** - Day of week (0-6)
7. **congestion_window** - TCP congestion window size
8. **queue_depth** - Network queue depth
9. **retransmits** - Number of packet retransmissions

## Output Predictions (4)

1. **throughput_mbps** - Predicted bandwidth 15 minutes ahead
2. **rtt_ms** - Predicted latency
3. **packet_loss** - Predicted packet loss rate
4. **jitter_ms** - Predicted jitter

## Training Data Schema

The training data should be a CSV file with the following columns:

```csv
timestamp,region,az,link_type,node_id,peer_id,rtt_ms,jitter_ms,throughput_mbps,
bytes_tx,bytes_rx,packet_loss,retransmits,congestion_window,queue_depth,
dwcp_mode,network_tier,transport_type,time_of_day,day_of_week
```

### Sample Data

```csv
1699564800,us-east-1,us-east-1a,dc,1234,5678,45.2,3.1,850.5,1048576,1024000,0.001,5,1500,25,turbo,Premium,tcp,14,2
1699564860,us-east-1,us-east-1a,dc,1234,5678,46.8,3.3,840.2,1050000,1022000,0.002,7,1480,28,turbo,Premium,tcp,14,2
...
```

## Prerequisites

### 1. Install Dependencies

```bash
# Using pip with user flag
pip3 install --user --break-system-packages numpy pandas scikit-learn matplotlib seaborn torch

# OR using virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn torch
```

### 2. Generate Training Data

```bash
# Generate synthetic data for testing
python3 scripts/generate_dwcp_training_data.py \
    --output data/dwcp_training.csv \
    --samples 15000 \
    --seed 42
```

## Training Process

### Step 1: Prepare Environment

```bash
# Navigate to project root
cd /home/kp/repos/novacron

# Create output directory
mkdir -p checkpoints/bandwidth_predictor
mkdir -p docs/models
```

### Step 2: Train Model

```bash
# Train with optimized hyperparameters
python3 backend/core/network/dwcp/prediction/training/train_lstm_pytorch.py \
    --data-path data/dwcp_training.csv \
    --output-dir checkpoints/bandwidth_predictor \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --window-size 20 \
    --seed 42
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--data-path` | Required | Path to training data CSV |
| `--output-dir` | `./checkpoints/bandwidth_predictor` | Output directory for models |
| `--epochs` | 150 | Number of training epochs |
| `--batch-size` | 64 | Batch size for training |
| `--learning-rate` | 0.001 | Initial learning rate |
| `--window-size` | 20 | Sequence length (timesteps) |
| `--seed` | 42 | Random seed for reproducibility |

### Step 3: Monitor Training

The training script outputs:
- Real-time epoch progress
- Training and validation losses
- Early stopping notifications
- Learning rate adjustments

Example output:
```
================================================================================
PYTORCH LSTM BANDWIDTH PREDICTOR TRAINING
Target: ≥98% Accuracy
================================================================================

Using device: cuda  # or cpu

Loading data from data/dwcp_training.csv
Loaded 15000 samples
Date range: 2025-10-28 to 2025-11-14

Using 9 features: ['throughput_mbps', 'rtt_ms', 'packet_loss', 'jitter_ms',
                    'time_of_day', 'day_of_week', 'congestion_window',
                    'queue_depth', 'retransmits']

Created 14980 sequences
X shape: (14980, 20, 9) (samples, timesteps, features)
y shape: (14980, 4) (samples, targets)

Data split:
  Train:      10486 samples (70.0%)
  Validation:  2247 samples (15.0%)
  Test:        2247 samples (15.0%)

================================================================================
TRAINING PYTORCH LSTM MODEL
================================================================================

Model Architecture:
BandwidthLSTM(
  (lstm1): LSTM(9, 256, batch_first=True, dropout=0.3)
  (bn1): BatchNorm1d(256)
  (lstm2): LSTM(256, 128, batch_first=True, dropout=0.3)
  (bn2): BatchNorm1d(128)
  (lstm3): LSTM(128, 64, batch_first=True)
  (fc1): Linear(64, 128)
  (dropout1): Dropout(p=0.3)
  (bn3): BatchNorm1d(128)
  (fc2): Linear(128, 64)
  (dropout2): Dropout(p=0.21)
  (fc3): Linear(64, 32)
  (fc_out): Linear(32, 4)
  (relu): ReLU()
)

Total parameters: 1,234,560
Trainable parameters: 1,234,560

Starting training for 150 epochs...
Batch size: 64
Initial learning rate: 0.001

Epoch [1/150] - Train Loss: 0.125432, Val Loss: 0.089654
Epoch [10/150] - Train Loss: 0.034567, Val Loss: 0.028901
...
Epoch [85/150] - Train Loss: 0.001234, Val Loss: 0.001456

Early stopping triggered after 85 epochs

Training completed!
Best validation loss: 0.001234
```

### Step 4: Evaluate Model

The training script automatically evaluates on the test set:

```
================================================================================
EVALUATING MODEL
================================================================================

Per-Target Metrics:
--------------------------------------------------------------------------------

throughput_mbps:
  MAE:         4.2341
  RMSE:        6.7812
  MAPE:        0.85%
  R² Score:    0.9912
  Correlation: 0.9956

rtt_ms:
  MAE:         1.1245
  RMSE:        1.8901
  MAPE:        2.34%
  R² Score:    0.9834
  Correlation: 0.9916

packet_loss:
  MAE:         0.0003
  RMSE:        0.0005
  MAPE:        3.21%
  R² Score:    0.9801
  Correlation: 0.9900

jitter_ms:
  MAE:         0.2145
  RMSE:        0.3456
  MAPE:        4.12%
  R² Score:    0.9723
  Correlation: 0.9860

================================================================================
OVERALL PERFORMANCE
================================================================================
Average Correlation: 0.9908
Average MAPE:        2.63%

================================================================================
✅ TARGET MET: ≥98% Accuracy Achieved!
================================================================================
```

## Output Artifacts

After training, the following files are generated:

### 1. Model Files

```
checkpoints/bandwidth_predictor/
├── best_model.pth                          # PyTorch checkpoint
├── bandwidth_lstm_v20251114_143000.onnx    # ONNX model for production
├── model_metadata_v20251114_143000.json    # Model metadata
├── bandwidth_predictor_report.json         # Evaluation report
├── training_history.png                    # Training curves
└── predictions_scatter.png                 # Actual vs predicted plots
```

### 2. Model Metadata (`model_metadata_*.json`)

```json
{
  "version": "v20251114_143000",
  "created_at": "2025-11-14T14:30:00",
  "training_duration_seconds": 342.56,
  "model_size_mb": 4.87,
  "target_achieved": true,
  "performance_metrics": {
    "overall": {
      "average_correlation": 0.9908,
      "average_mape": 2.63,
      "target_met": true,
      "accuracy_percent": 99.08
    },
    "per_target": { ... }
  },
  "model_architecture": { ... },
  "features": [...],
  "targets": [...],
  "scaler_params": { ... }
}
```

### 3. Evaluation Report (`bandwidth_predictor_report.json`)

```json
{
  "model": "bandwidth_predictor",
  "version": "v20251114_143000",
  "success": true,
  "achieved_metrics": {
    "correlation": 0.9908,
    "mape": 2.63,
    "accuracy_percent": 99.08
  },
  "target_metrics": {
    "correlation": 0.98,
    "mape": 5.0
  },
  "training_time_seconds": 342.56,
  "model_size_mb": 4.87
}
```

## Integration with Go Backend

### 1. Copy ONNX Model

```bash
# Copy to production location
cp checkpoints/bandwidth_predictor/bandwidth_lstm_*.onnx \
   backend/core/network/dwcp/prediction/models/bandwidth_lstm_production.onnx
```

### 2. Update Go Configuration

```go
// backend/core/network/dwcp/prediction/config.go
const (
    DefaultModelPath = "models/bandwidth_lstm_production.onnx"
    WindowSize       = 20
    NumFeatures      = 9
    NumOutputs       = 4
)
```

### 3. Test Integration

```bash
# Run Go tests
cd backend/core/network/dwcp/prediction
go test -v ./...

# Run prediction test
go run example_integration.go
```

## Performance Optimization

### 1. GPU Training

```bash
# Check GPU availability
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Train with GPU (automatic if available)
python3 backend/core/network/dwcp/prediction/training/train_lstm_pytorch.py \
    --data-path data/dwcp_training.csv \
    --output-dir checkpoints/bandwidth_predictor \
    --epochs 150 \
    --batch-size 128  # Larger batch size for GPU
```

### 2. Hyperparameter Tuning

```bash
# Try different configurations
for lr in 0.0001 0.001 0.01; do
    for bs in 32 64 128; do
        python3 train_lstm_pytorch.py \
            --data-path data/dwcp_training.csv \
            --output-dir checkpoints/bw_pred_lr${lr}_bs${bs} \
            --learning-rate $lr \
            --batch-size $bs \
            --epochs 150
    done
done
```

### 3. Model Pruning (Optional)

For smaller model size:
```python
import torch
from torch.nn.utils import prune

# Prune 30% of weights
for module in model.modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

## Troubleshooting

### Issue: Low Accuracy

**Solutions:**
1. Increase training data size (>20,000 samples recommended)
2. Increase model capacity (more LSTM units)
3. Adjust learning rate
4. Increase training epochs
5. Check data quality and distribution

### Issue: Overfitting

**Symptoms:** Train loss << val loss

**Solutions:**
1. Increase dropout rates
2. Add more regularization
3. Reduce model complexity
4. Increase training data
5. Enable early stopping (already implemented)

### Issue: Slow Training

**Solutions:**
1. Use GPU if available
2. Increase batch size
3. Reduce model complexity
4. Use mixed precision training

### Issue: Out of Memory

**Solutions:**
1. Reduce batch size
2. Reduce sequence length
3. Use gradient checkpointing
4. Train on smaller data chunks

## Best Practices

1. **Data Quality**
   - Ensure at least 15,000 samples
   - Cover diverse network conditions
   - Include seasonal patterns

2. **Version Control**
   - Save model metadata with every training run
   - Track hyperparameters in git
   - Version ONNX models with timestamps

3. **Continuous Monitoring**
   - Monitor model performance in production
   - Retrain monthly with new data
   - A/B test new models before deployment

4. **Testing**
   - Validate on held-out test set
   - Test integration with Go backend
   - Benchmark inference latency (<10ms)

## CLI Command Summary

```bash
# Full training workflow
mkdir -p data checkpoints/bandwidth_predictor

# Generate data
python3 scripts/generate_dwcp_training_data.py \
    --output data/dwcp_training.csv \
    --samples 15000

# Train model
python3 backend/core/network/dwcp/prediction/training/train_lstm_pytorch.py \
    --data-path data/dwcp_training.csv \
    --output-dir checkpoints/bandwidth_predictor \
    --epochs 150 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --window-size 20

# Check results
cat checkpoints/bandwidth_predictor/bandwidth_predictor_report.json

# Deploy to production
cp checkpoints/bandwidth_predictor/bandwidth_lstm_*.onnx \
   backend/core/network/dwcp/prediction/models/bandwidth_lstm_production.onnx
```

## References

- [PyTorch LSTM Documentation](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Time Series Forecasting with LSTM](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- DWCP Prediction System README: `backend/core/network/dwcp/prediction/README.md`

## License

Copyright 2025 NovaCron. All rights reserved.
