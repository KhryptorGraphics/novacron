# DWCP Neural Training Quick Start Guide

**Last Updated:** 2025-11-14
**Status:** ✅ Ready for Execution

## Overview

This guide provides quick commands to train, validate, and deploy DWCP neural models.

## Prerequisites

### Install Dependencies

```bash
# Option 1: Virtual Environment (Recommended)
cd backend/core/network/dwcp/prediction/training/
./setup_training_env.sh

# Option 2: System Packages
sudo apt install python3-tensorflow python3-sklearn python3-numpy python3-pandas

# Option 3: Docker
docker pull tensorflow/tensorflow:2.15.0
```

### Verify Installation

```bash
python3 -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python3 -c "import sklearn; print('scikit-learn:', sklearn.__version__)"
```

---

## Model 1: Consensus Latency Autoencoder

**Current Performance:** 95.3% detection accuracy
**Target:** ≥98%
**Status:** ⚠️ Needs improvement

### Quick Train

```bash
cd backend/core/network/dwcp/monitoring/training/

# Standard training (10 minutes)
python train_lstm_autoencoder.py \
    --output ../../../../../ml/models/consensus \
    --n-normal 10000 \
    --n-anomalies 500 \
    --epochs 100

# Improved training (recommended)
python train_lstm_autoencoder.py \
    --output ../../../../../ml/models/consensus \
    --n-normal 15000 \
    --n-anomalies 1500 \
    --epochs 150 \
    --encoding-dim 24
```

### Outputs

```
ml/models/consensus/
├── consensus_latency_autoencoder.keras  # Trained model
├── consensus_scaler.pkl                 # Feature scaler
├── consensus_metadata.json              # Model config
├── evaluation_report.png                # Performance plots
└── training_curves.png                  # Training history
```

### Evaluation

Check metrics in `ml/models/consensus/consensus_metadata.json`:

```json
{
  "metrics": {
    "detection_accuracy": 0.9528,  // Target: ≥0.98
    "precision": 1.0,
    "recall": 0.9056,
    "f1_score": 0.9505
  }
}
```

### Convert to ONNX (For Go Integration)

```python
import torch
import torch.onnx

# Load model
model = torch.load('ml/models/consensus/consensus_latency_autoencoder.pth')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 30, 12)

# Export
torch.onnx.export(
    model,
    dummy_input,
    'ml/models/consensus/consensus_latency_autoencoder.onnx',
    input_names=['input'],
    output_names=['output']
)
```

---

## Model 2: Node Reliability (Isolation Forest)

**Current Performance:** Unknown (likely high FP rate)
**Target:** ≥98% recall, <5% FP rate
**Status:** ⚠️ May need supervised learning

### Quick Train

```bash
cd backend/core/network/dwcp/monitoring/training/

# Standard training
python train_isolation_forest.py \
    --synthetic \
    --n-samples 10000 \
    --incident-rate 0.02 \
    --output ../../../../../backend/core/network/dwcp/monitoring/models

# Fast demo
python train_isolation_forest_fast.py --n-samples 5000
```

### Alternative: Supervised Learning (Recommended)

If Isolation Forest doesn't meet targets, use XGBoost:

```python
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data
df = pd.read_csv('node_data.csv')
X = df.drop('label', axis=1)
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y
)

# Train
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    scale_pos_weight=50,  # Handle imbalance
    eval_metric='aucpr'
)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import recall_score, precision_score
y_pred = model.predict(X_test)
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")

# Save
joblib.dump(model, 'models/xgboost_node_reliability.pkl')
```

---

## Model 3: Bandwidth Predictor

**Status:** ⚠️ Not yet trained (infrastructure ready)
**Target:** ≥98% accuracy OR correlation ≥0.98 OR MAPE ≤5%

### Quick Train

```bash
cd backend/core/network/dwcp/prediction/training/

# Automated training (recommended)
./train_bandwidth_predictor.sh

# Manual training
python train_lstm_enhanced.py \
    --data-path ../../../../../../data/dwcp_training.csv \
    --output-dir ./checkpoints/bandwidth_predictor \
    --epochs 200 \
    --batch-size 64
```

### Generate Training Data (If Needed)

```bash
# Create synthetic training data
python ../../../../../scripts/generate_dwcp_training_data.py \
    --output data/dwcp_training.csv \
    --samples 15000
```

### Outputs

```
checkpoints/bandwidth_predictor_enhanced/
├── best_model.keras                    # Keras model
├── bandwidth_lstm_vYYYYMMDD.onnx       # ONNX (for Go)
├── model_metadata_vYYYYMMDD.json       # Config + scalers
├── TRAINING_REPORT.json                # Evaluation
└── training_log.csv                    # Per-epoch metrics
```

### Check Results

```bash
cat checkpoints/bandwidth_predictor_enhanced/TRAINING_REPORT.json | grep -E "success|accuracy|correlation"
```

Expected output:
```json
{
  "success": true,
  "metrics": {
    "overall_accuracy": 0.985,       // ≥0.98
    "correlation": 0.989,            // ≥0.98
    "mape": 3.2                      // ≤5%
  }
}
```

---

## Go Integration

### Deploy ONNX Models

```bash
# Create models directory
mkdir -p backend/core/network/dwcp/models

# Copy trained models
cp ml/models/consensus/consensus_latency_autoencoder.onnx \
   backend/core/network/dwcp/models/

cp checkpoints/bandwidth_predictor_enhanced/bandwidth_lstm_*.onnx \
   backend/core/network/dwcp/models/bandwidth_predictor.onnx
```

### Update Go Code

```go
// Load bandwidth predictor
predictor, err := prediction.NewLSTMPredictor(
    "backend/core/network/dwcp/models/bandwidth_predictor.onnx"
)

// Load metadata for scaler parameters
metadataFile, _ := os.ReadFile("checkpoints/bandwidth_predictor_enhanced/model_metadata.json")
var metadata ModelMetadata
json.Unmarshal(metadataFile, &metadata)

// Use scaler parameters in prepareInput()
```

### Test Inference

```go
// Create test data
history := []NetworkSample{
    {BandwidthMbps: 850, LatencyMs: 15, PacketLoss: 0.001, ...},
    {BandwidthMbps: 860, LatencyMs: 14, PacketLoss: 0.001, ...},
    // ... 10 samples total
}

// Predict
prediction, err := predictor.Predict(history)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Predicted Bandwidth: %.2f Mbps\n", prediction.PredictedBandwidthMbps)
fmt.Printf("Predicted Latency: %.2f ms\n", prediction.PredictedLatencyMs)
fmt.Printf("Confidence: %.2f\n", prediction.Confidence)
```

---

## Performance Validation

### Benchmark Inference Latency

```bash
cd backend/core/network/dwcp/prediction
go test -bench=BenchmarkPredict -benchtime=10s
```

**Target Latency:**
- Consensus Latency: <10ms
- Node Reliability: <5ms
- Bandwidth Predictor: <5ms

### Accuracy Testing

```bash
# Run validation tests
cd backend/core/network/dwcp/prediction
go test -v -run TestPredictionAccuracy
```

---

## Monitoring

### Prometheus Metrics

```bash
# Check prediction metrics
curl localhost:9090/metrics | grep dwcp_pba

# Expected output:
# dwcp_pba_prediction_accuracy 0.985
# dwcp_pba_prediction_latency_ms_bucket{le="5"} 1234
# dwcp_pba_model_version 1
# dwcp_pba_confidence 0.92
```

### Grafana Dashboard

Import dashboard from `docs/deployment/grafana-ml-dashboard.json`

**Key Panels:**
- Prediction Accuracy (real-time)
- Inference Latency (p50, p95, p99)
- Model Version
- Confidence Score
- Prediction Rate

---

## Troubleshooting

### Training Issues

**Problem:** Low accuracy (<95%)

**Solutions:**
1. More training data (target 15K+ samples)
2. Increase epochs: `--epochs 300`
3. Adjust learning rate: `--learning-rate 0.0005`
4. Check data quality (missing values, outliers)

**Problem:** Training too slow

**Solutions:**
1. Reduce batch size: `--batch-size 32`
2. Reduce epochs: `--epochs 100`
3. Use GPU if available
4. Enable mixed precision (automatic in TF 2.15)

**Problem:** Out of memory

**Solutions:**
1. Reduce batch size: `--batch-size 32`
2. Reduce window size: `--window-size 20`
3. Use CPU: `export CUDA_VISIBLE_DEVICES=""`

### Go Integration Issues

**Problem:** ONNX library not found

**Solution:**
```bash
# Install ONNX Runtime
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.0/onnxruntime-linux-x64-1.16.0.tgz
tar -xzf onnxruntime-linux-x64-1.16.0.tgz
sudo cp onnxruntime-linux-x64-1.16.0/lib/* /usr/local/lib/
sudo ldconfig

# Or set environment variable
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

**Problem:** Feature mismatch

**Solution:**
Align Go feature extraction with Python:
```go
// Match train_lstm_enhanced.py feature engineering
features := extractFeatures(samples)  // Should return 40-50 features
```

**Problem:** Inference errors

**Solution:**
```bash
# Validate ONNX model
python -c "
import onnxruntime as ort
session = ort.InferenceSession('model.onnx')
print(session.get_inputs()[0].shape)  # Should match Go input
"
```

---

## Quick Reference

### File Locations

```
Neural Training Infrastructure:
├── backend/core/network/dwcp/
│   ├── monitoring/training/          # Consensus + node reliability
│   │   ├── train_lstm_autoencoder.py
│   │   └── train_isolation_forest.py
│   ├── prediction/training/          # Bandwidth predictor
│   │   ├── train_lstm_enhanced.py
│   │   └── train_bandwidth_predictor.sh
│   └── models/                       # ONNX models for Go
├── ml/models/                        # Trained Python models
│   └── consensus/
└── docs/ml/                          # Documentation
    ├── NEURAL_TRAINING_VALIDATION_REPORT.md
    └── NEURAL_TRAINING_QUICK_START.md (this file)
```

### Command Cheatsheet

```bash
# Setup
./setup_training_env.sh

# Train consensus latency
python train_lstm_autoencoder.py --n-normal 15000 --n-anomalies 1500 --epochs 150

# Train node reliability
python train_isolation_forest.py --synthetic --n-samples 10000

# Train bandwidth predictor
./train_bandwidth_predictor.sh

# Convert to ONNX
python convert_to_onnx.py --model model.pth --output model.onnx

# Test Go integration
go test -v backend/core/network/dwcp/prediction/...

# Check metrics
curl localhost:9090/metrics | grep dwcp_pba
```

---

## Next Steps

1. **Install dependencies** (if not done)
   ```bash
   cd backend/core/network/dwcp/prediction/training/
   ./setup_training_env.sh
   ```

2. **Train all models**
   ```bash
   # Consensus latency
   cd backend/core/network/dwcp/monitoring/training/
   python train_lstm_autoencoder.py --n-normal 15000 --n-anomalies 1500 --epochs 150

   # Bandwidth predictor
   cd ../../prediction/training/
   ./train_bandwidth_predictor.sh
   ```

3. **Convert to ONNX**
   ```bash
   # Create conversion script or use torch.onnx.export
   ```

4. **Deploy to Go**
   ```bash
   cp models/*.onnx backend/core/network/dwcp/models/
   ```

5. **Validate performance**
   ```bash
   go test -v backend/core/network/dwcp/prediction/...
   ```

6. **Monitor in production**
   - Check Grafana dashboard
   - Review Prometheus metrics
   - Track prediction accuracy

---

## Resources

**Documentation:**
- Full Validation Report: `docs/ml/NEURAL_TRAINING_VALIDATION_REPORT.md`
- Bandwidth Predictor Guide: `backend/core/network/dwcp/prediction/training/README.md`
- Node Reliability Guide: `backend/core/network/dwcp/monitoring/training/README.md`

**Support:**
- Check README files in training directories
- Review error messages and logs
- Consult validation report for recommendations

---

**Status:** ✅ Ready to Train
**Estimated Time:** 1-2 hours for all models
**Success Criteria:** All models ≥98% accuracy

**Happy Training!** 🚀
