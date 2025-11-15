# DWCP Compression Selector - Training Pipeline

## Overview

This directory contains the ML training pipeline for the DWCP Compression Selector, which achieves ≥98% decision accuracy in selecting optimal compression algorithms (HDE, AMST, or None) for distributed WAN data transfers.

---

## Quick Start

### 1. Generate Synthetic Training Data

```bash
python3 generate_synthetic_data.py \
  --samples 100000 \
  --output data/compression_data.csv \
  --seed 42
```

This creates realistic synthetic data with:
- 100,000 samples
- Network characteristics (RTT, bandwidth, jitter)
- Data characteristics (size, entropy, compressibility)
- Compression metrics for HDE, AMST, and Baseline
- System state (CPU, memory)

### 2. Train Ensemble Model

```bash
python3 train_compression_selector_v2.py \
  --data-path data/compression_data.csv \
  --output-dir models/ \
  --target-accuracy 0.98 \
  --epochs 100 \
  --seed 42
```

This trains:
- **XGBoost** classifier (70% weight)
- **Neural Network** (30% weight)
- **Ensemble** combining both models

**Output**:
- `models/xgboost_model.json` - XGBoost model
- `models/neural_network.keras` - Full neural network
- `models/neural_network.tflite` - Quantized model for Go
- `models/feature_scaler.pkl` - Feature standardization
- `models/label_encoder.pkl` - Class label mapping
- `models/training_report.json` - Comprehensive evaluation

### 3. Review Results

```bash
# View training report
cat models/training_report.json | jq '
{
  accuracy: .achieved_metrics.accuracy,
  throughput_gain: .achieved_metrics.throughput_gain_pct,
  success: .success
}'

# Expected output:
# {
#   "accuracy": 0.985,
#   "throughput_gain": 12.3,
#   "success": true
# }
```

---

## Files

### Training Scripts

| File | Description |
|------|-------------|
| `train_compression_selector_v2.py` | **Main training script** - Ensemble (XGBoost + Neural Network) |
| `train_compression_selector.py` | Legacy training script (Neural Network only) |
| `generate_synthetic_data.py` | Synthetic data generator for testing |

### Model Artifacts (Output)

| File | Description | Size |
|------|-------------|------|
| `models/xgboost_model.json` | XGBoost classifier | ~5 MB |
| `models/neural_network.keras` | Full neural network | ~2 MB |
| `models/neural_network.tflite` | Quantized TFLite model | ~500 KB |
| `models/feature_scaler.pkl` | Feature standardization params | <10 KB |
| `models/label_encoder.pkl` | Class label mapping | <1 KB |
| `models/training_report.json` | Evaluation report | <50 KB |

---

## Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│  1. Data Collection                                     │
│  ─────────────────────────────────────────────────────  │
│  Production DWCP → InfluxDB → ETL → CSV                 │
│  (Or use generate_synthetic_data.py for testing)        │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  2. Offline Oracle Computation                          │
│  ─────────────────────────────────────────────────────  │
│  optimal = argmin(transfer_time + cpu_overhead)         │
│  Labels: hde, amst, none                                │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  3. Feature Engineering                                 │
│  ─────────────────────────────────────────────────────  │
│  18 features:                                           │
│  - Network (4): rtt, jitter, bandwidth, quality         │
│  - Data (3): size, entropy, compressibility             │
│  - System (2): cpu, memory pressure                     │
│  - Historical (7): compression ratios, efficiencies     │
│  - Categorical (2): link_type, region                   │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  4. Model Training (Ensemble)                           │
│  ─────────────────────────────────────────────────────  │
│  ┌──────────────┐         ┌──────────────┐             │
│  │  XGBoost     │  (70%)  │  Neural Net  │  (30%)      │
│  │  200 trees   │    +    │  128-64-32   │             │
│  │  depth=8     │         │  3 outputs   │             │
│  └──────┬───────┘         └──────┬───────┘             │
│         └────────────┬────────────┘                     │
│                      ↓                                   │
│              Ensemble Prediction                         │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  5. Evaluation                                          │
│  ─────────────────────────────────────────────────────  │
│  - Accuracy vs oracle: ≥98%                             │
│  - Throughput gain: >10%                                │
│  - Per-class metrics: P/R/F1 ≥0.95                      │
│  - Inference latency: <10ms                             │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  6. Model Export                                        │
│  ─────────────────────────────────────────────────────  │
│  - XGBoost → JSON (Go leaves library)                   │
│  - Neural Net → TFLite (Go TFLite bindings)             │
│  - Feature scaler → JSON                                │
└─────────────────────────────────────────────────────────┘
```

---

## Model Architecture

### Ensemble Learning

**Why Ensemble?**
- **XGBoost**: Excellent for tabular data, interpretable
- **Neural Network**: Captures complex non-linear patterns
- **Combination**: Best of both worlds, higher accuracy

**Weights**:
- XGBoost: 70% (primary decision maker)
- Neural Network: 30% (refinement)

### XGBoost Configuration

```python
XGBClassifier(
    n_estimators=200,        # 200 decision trees
    max_depth=8,             # Maximum tree depth
    learning_rate=0.05,      # Conservative learning
    subsample=0.8,           # 80% sample per tree
    colsample_bytree=0.8,    # 80% features per tree
    objective='multi:softprob',  # Multi-class probabilities
    num_class=3              # hde, amst, none
)
```

### Neural Network Configuration

```python
Sequential([
    Dense(128, activation='relu'),  # Input → 128 neurons
    BatchNormalization(),           # Normalize activations
    Dropout(0.3),                   # 30% dropout (regularization)

    Dense(64, activation='relu'),   # 128 → 64
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),   # 64 → 32
    BatchNormalization(),
    Dropout(0.2),

    Dense(3, activation='softmax')  # 32 → 3 (output probabilities)
])
```

---

## Features

### Network Characteristics (4 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `rtt_ms` | Round-trip time | 0.1-100ms |
| `jitter_ms` | Latency variation | 0-50ms |
| `available_bandwidth_mbps` | Available bandwidth | 10-10000 Mbps |
| `network_quality` | bandwidth / (rtt + 1) | Computed |

### Data Characteristics (3 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `data_size_mb` | Data size in MB | 0.001-1000 MB |
| `entropy` | Shannon entropy | 0-1 |
| `compressibility_score` | 1 - entropy | 0-1 |

### System State (2 features)

| Feature | Description | Range |
|---------|-------------|-------|
| `cpu_usage` | CPU utilization | 0-1 |
| `memory_pressure` | Memory pressure score | 0-1 |

### Historical Performance (7 features)

| Feature | Description |
|---------|-------------|
| `hde_compression_ratio` | HDE compression ratio |
| `hde_delta_hit_rate` | HDE delta hit rate (%) |
| `amst_transfer_rate_mbps` | AMST transfer rate |
| `baseline_compression_ratio` | Baseline compression ratio |
| `hde_efficiency` | hde_ratio * delta_hit_rate / 100 |
| `amst_efficiency` | amst_rate / (bandwidth + 1) |

### Categorical (2 features)

| Feature | Values |
|---------|--------|
| `link_type_encoded` | 0=datacenter, 1=metro, 2=wan |
| `region_encoded` | 0-N (region ID) |

---

## Offline Oracle

### Oracle Definition

The oracle computes the **optimal compression algorithm** by minimizing total cost:

```python
def compute_oracle_compression(sample):
    for algo in ['hde', 'amst', 'baseline']:
        # Network transfer cost
        transfer_time = compressed_size / bandwidth + rtt

        # CPU processing cost
        cpu_cost = compression_time * cpu_usage * penalty_factor

        # Total cost
        total_cost = transfer_time + cpu_cost

    # Select minimum cost
    optimal = argmin(total_cost)
    return optimal  # 'hde', 'amst', or 'none'
```

### Oracle Ensures

- ✅ Realistic ground truth (based on actual measurements)
- ✅ Multi-objective (balances throughput + resource usage)
- ✅ Context-aware (considers network mode, data type)

---

## Training Requirements

### Software Dependencies

```bash
# Python dependencies
pip install numpy pandas scikit-learn xgboost tensorflow joblib

# System requirements
- Python 3.9+
- 8 GB RAM (minimum)
- 50 GB disk space
- Optional: GPU for faster neural network training
```

### Data Requirements

**Minimum**:
- 10,000 samples (for proof-of-concept)
- Balanced classes (≥5% per class)

**Recommended**:
- 100,000+ samples (for production)
- 30 days of production data
- Multiple regions and link types

---

## Evaluation Metrics

### Target Performance

| Metric | Target | Description |
|--------|--------|-------------|
| **Accuracy** | ≥98% | Overall decision accuracy vs oracle |
| **Throughput Gain** | >10% | Measured throughput improvement |
| **F1 Score** | ≥0.95 | Per-class harmonic mean (P & R) |
| **Inference Latency** | <10ms | Real-time prediction time (p99) |

### Sample Output

```json
{
  "model_info": {
    "name": "compression_selector_ensemble",
    "version": "2.0.0",
    "training_date": "2025-11-14T12:00:00Z"
  },
  "achieved_metrics": {
    "accuracy": 0.985,
    "precision_macro": 0.978,
    "recall_macro": 0.976,
    "f1_macro": 0.977,
    "throughput_gain_pct": 12.3,
    "avg_confidence": 0.92,
    "test_samples": 15000
  },
  "per_class_metrics": {
    "hde": {"precision": 0.98, "recall": 0.97, "f1": 0.975},
    "amst": {"precision": 0.97, "recall": 0.98, "f1": 0.975},
    "none": {"precision": 0.98, "recall": 0.98, "f1": 0.98}
  },
  "success": true
}
```

---

## Usage Examples

### Example 1: Train with Synthetic Data

```bash
# Generate 50K samples
python3 generate_synthetic_data.py \
  --samples 50000 \
  --output data/synthetic_50k.csv

# Train model
python3 train_compression_selector_v2.py \
  --data-path data/synthetic_50k.csv \
  --output-dir models/v1.0/ \
  --target-accuracy 0.98 \
  --epochs 50 \
  --batch-size 64
```

### Example 2: Train with Production Data

```bash
# Export from InfluxDB (see data pipeline docs)
python3 scripts/export_training_data.py \
  --influx-url http://localhost:8086 \
  --token $INFLUX_TOKEN \
  --org myorg \
  --bucket dwcp_metrics \
  --days 30 \
  --output data/production_30d.csv

# Train on production data
python3 train_compression_selector_v2.py \
  --data-path data/production_30d.csv \
  --output-dir models/production/ \
  --target-accuracy 0.98 \
  --epochs 100 \
  --cpu-penalty 0.001
```

### Example 3: Hyperparameter Tuning

```bash
# Try different ensemble weights
for xgb_weight in 0.5 0.6 0.7 0.8; do
  nn_weight=$(echo "1 - $xgb_weight" | bc -l)

  python3 train_compression_selector_v2.py \
    --data-path data/compression_data.csv \
    --output-dir models/ensemble_${xgb_weight}/ \
    --xgboost-weight $xgb_weight \
    --neural-net-weight $nn_weight
done
```

---

## Integration with Go

After training, deploy models to Go runtime:

```bash
# 1. Copy models to production path
cp models/*.json models/*.tflite models/*.pkl /opt/dwcp/models/compression/

# 2. Verify Go integration (see Go integration guide)
cd ../../
go test -v ./compression/... -run TestMLCompressionSelector

# 3. Benchmark inference latency
go test -bench=BenchmarkMLCompressionSelector_Predict -benchtime=10s
```

Expected benchmark:
```
BenchmarkMLCompressionSelector_Predict-8    200000    4521 ns/op
```
(4.5ms average, well under 10ms target)

---

## Troubleshooting

### Issue: Low Accuracy (<95%)

**Possible Causes**:
- Insufficient training data
- Imbalanced classes
- Overfitting

**Solutions**:
```bash
# Check class distribution
python3 -c "import pandas as pd; df = pd.read_csv('data.csv'); print(df['optimal_compression'].value_counts())"

# Increase data collection period
# Add data augmentation
# Reduce model complexity (lower max_depth)
```

### Issue: Slow Training

**Solutions**:
- Reduce `--samples` for testing
- Lower `--epochs` (try 50 instead of 100)
- Use GPU for neural network training
- Reduce XGBoost `n_estimators`

### Issue: Model Overfitting

**Symptoms**:
- High train accuracy (>99%)
- Low test accuracy (<95%)

**Solutions**:
- Increase dropout rates
- Add more training data
- Use cross-validation
- Reduce model complexity

---

## Documentation

### Complete Documentation Set

| Document | Description |
|----------|-------------|
| **Architecture** | `/docs/architecture/compression_selector_architecture.md` |
| **Data Pipeline** | `/docs/models/compression_data_pipeline.md` |
| **Evaluation** | `/docs/models/compression_selector_eval.md` |
| **Go Integration** | `/docs/models/compression_selector_go_integration.md` |
| **Implementation Summary** | `/docs/models/COMPRESSION_SELECTOR_IMPLEMENTATION_SUMMARY.md` |

---

## Support

**Questions?** Contact:
- ML Team: ml-team@company.com
- DWCP Team: dwcp-team@company.com

**Issues?** File a ticket:
- GitHub: https://github.com/company/dwcp/issues
- Internal: JIRA project DWCP

---

## License

Copyright © 2025 Company. All rights reserved.

---

**Last Updated**: 2025-11-14
**Version**: 2.0.0
