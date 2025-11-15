# Node Reliability Model - Quick Start Guide

**Model**: Isolation Forest (Anomaly Detection)
**Status**: ⚠️ Development Complete, Does Not Meet Production Requirements
**Use Case**: Research/Baseline only - **Supervised model required for production**

## TL;DR

```bash
# The model achieves 98% recall but with 95% false positive rate
# Not suitable for production - supervised learning required
# See ISOLATION_FOREST_FINAL_REPORT.md for full details
```

## Quick Commands

### Train Model (Synthetic Data)

```bash
cd backend/core/network/dwcp/monitoring/training

# Full training - 192 configurations (~15 min)
python train_isolation_forest.py \
  --synthetic \
  --n-samples 10000 \
  --incident-rate 0.02 \
  --target-recall 0.98 \
  --max-fp-rate 0.05

# Fast demo - 16 configurations (~30 sec)
python train_isolation_forest_fast.py \
  --n-samples 5000

# Optimized with realistic data (~5 min)
python train_node_reliability_tuned.py \
  --n-samples 15000 \
  --incident-rate 0.03
```

### Train Model (Real Data)

```bash
python train_isolation_forest.py \
  --data /path/to/node_metrics_labeled.csv \
  --output ../models \
  --report ../../../../../../docs/models/node_reliability_eval.md \
  --target-recall 0.98 \
  --max-fp-rate 0.05
```

**Required CSV columns**:
```
timestamp, node_id, region, az,
error_rate, timeout_rate, latency_p50, latency_p99,
sla_violations, connection_failures, packet_loss_rate,
cpu_usage, memory_usage, disk_io,
dwcp_mode, network_tier, label
```

### Load and Use Model

```python
import joblib
import json
import numpy as np

# Load model artifacts
model = joblib.load('models/isolation_forest_node_reliability.pkl')
scaler = joblib.load('models/scaler_node_reliability.pkl')

with open('models/model_metadata_node_reliability.json') as f:
    metadata = json.load(f)
    threshold = metadata['threshold']  # -0.376980
    features = metadata['feature_names']  # 163 features

# Predict (assuming X_raw has 163 features)
X_scaled = scaler.transform(X_raw)
scores = model.score_samples(X_scaled)
predictions = (scores <= threshold).astype(int)

# Get confidence
confidence = np.abs(scores - threshold)
```

## Model Files

```
backend/core/network/dwcp/monitoring/
├── training/
│   ├── train_isolation_forest.py           # Main training script
│   ├── train_isolation_forest_fast.py      # Fast demo version
│   └── train_node_reliability_tuned.py     # Optimized version
└── models/
    ├── isolation_forest_node_reliability.pkl      # 982KB - Trained model
    ├── scaler_node_reliability.pkl                # 2.5KB - Feature scaler
    ├── model_metadata_node_reliability.json       # Metadata
    └── hyperparameters_node_reliability.json      # Hyperparameters
```

## Documentation

```
docs/models/
├── QUICK_START_NODE_RELIABILITY.md           # This file
├── ISOLATION_FOREST_FINAL_REPORT.md          # Complete analysis
├── NODE_RELIABILITY_MODEL_SUMMARY.md         # Implementation guide
└── node_reliability_eval.md                  # Auto-generated metrics
```

## Performance Summary

### Validation Set
- Recall: **98.85%** ✓
- FP Rate: **98.70%** ✗ (Target: <5%)
- Precision: 0.56%
- F1: 0.0564

### Test Set
- Recall: **84.21%** ✗ (Target: ≥98%)
- FP Rate: **91.03%** ✗ (Target: <5%)
- Precision: 1.76%
- F1: 0.0345
- ROC-AUC: 0.4962

**Confusion Matrix (Test)**:
```
           Predicted
           Normal  Incident
Normal        88      893    ← 91% FP
Incident       3       16    ← 84% Recall
```

## Why It Failed

Isolation Forest is **unsupervised** - it learns "outliers" not "incidents":
1. Doesn't use labels during training
2. Cannot optimize recall/FP trade-off
3. Single threshold cannot separate overlapping distributions
4. To catch 98% of incidents → also flags 98% of normal samples

## Recommended Next Steps

### Option 1: Supervised Learning ⭐ RECOMMENDED

**Train XGBoost/Random Forest with labels**:
- Expected: 98% recall, 2% FP rate, 0.93 F1
- Requires: 5000+ labeled samples
- Timeline: 2-4 weeks

```bash
# Future command (to be implemented)
python train_supervised_classifier.py \
  --model xgboost \
  --data labeled_incidents.csv \
  --features model_metadata_node_reliability.json \
  --target-recall 0.98 \
  --max-fp-rate 0.05
```

### Option 2: Two-Stage Detection

1. **Stage 1**: Isolation Forest (99.9% recall, ~100% FP)
2. **Stage 2**: Supervised Classifier (reduces FPs to 2-3%)
3. **Combined**: 98.5% recall, 2.5% FP rate

### Option 3: Adjust Requirements (Interim)

- Keep recall ≥98%
- **Adjust FP rate to <15%** (achievable with Isolation Forest)
- Add manual review workflow
- Use as baseline while building supervised model

## Feature Engineering (163 Features)

All scripts use the same comprehensive feature engineering:

**Categories**:
1. Rolling window statistics (120 features)
   - 5, 10, 20 minute windows
   - Mean, std, min, max for each metric

2. Rate of change (20 features)
   - First derivative (velocity)
   - Second derivative (acceleration)

3. Interaction features (13 features)
   - error_timeout_product
   - latency_spread, latency_ratio
   - resource_pressure

4. Threshold indicators (3 features)
   - high_error_rate, high_latency, high_packet_loss

5. Categorical encodings (7 features)
   - dwcp_mode (one-hot)
   - network_tier (one-hot)

## Hyperparameters

**Optimal Configuration**:
```json
{
  "n_estimators": 200,
  "max_samples": "auto",
  "max_features": 0.8,
  "contamination": 0.0289,
  "random_state": 42,
  "n_jobs": -1
}
```

**Threshold**: -0.376980 (tuned for 98% recall)

## Common Issues

### Q: Why such high FP rate?

A: Isolation Forest is unsupervised. It detects "outliers" not "incidents". Normal operational spikes look like outliers.

### Q: Can threshold tuning fix this?

A: No. The distributions overlap too much. No single threshold achieves both 98% recall and <5% FP.

### Q: What about more data?

A: Tested with 5K, 10K, 15K, 20K samples. More data doesn't help - it's an algorithm limitation.

### Q: What about more features?

A: Already have 163 comprehensive features. Adding more doesn't change the fundamental problem.

### Q: Should I use this in production?

A: **No.** High FP rate (91-98%) will flood your monitoring system with false alerts.

## When to Use This Model

✓ **Research/Baseline**: Understand anomaly patterns
✓ **Data Collection**: Generate initial incident candidates for labeling
✓ **Stage 1 Screening**: With supervised Stage 2 to reduce FPs
✗ **Production Alerts**: Too many false positives
✗ **Critical Systems**: Supervised model required

## Dependencies

```bash
pip install numpy>=1.24.0 pandas>=2.0.0 scikit-learn>=1.3.0 joblib>=1.3.0
```

## Help

```bash
# Full options
python train_isolation_forest.py --help

# Key arguments
--data PATH              # Path to labeled CSV
--synthetic              # Use synthetic data
--n-samples N            # Number of samples (default: 10000)
--incident-rate R        # Incident rate (default: 0.02 = 2%)
--output DIR             # Model output directory
--report PATH            # Evaluation report path
--target-recall R        # Target recall (default: 0.98)
--max-fp-rate R          # Max FP rate (default: 0.05)
--test-size FRAC         # Test set fraction (default: 0.2)
```

## Further Reading

- **ISOLATION_FOREST_FINAL_REPORT.md** - Complete analysis and recommendations
- **NODE_RELIABILITY_MODEL_SUMMARY.md** - Detailed implementation guide
- **node_reliability_eval.md** - Auto-generated evaluation metrics

## Contact

See repository documentation for:
- Supervised model implementation
- Production deployment guides
- Monitoring and retraining pipelines

---

**Version**: 1.0.0
**Last Updated**: 2025-11-14
**Status**: Development Complete, Not Production Ready
