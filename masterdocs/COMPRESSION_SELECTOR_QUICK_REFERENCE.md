# DWCP Compression Selector - Quick Reference

## One-Page Cheat Sheet for Developers

---

## 🚀 Quick Start (5 Minutes)

```bash
# 1. Generate synthetic data
cd backend/core/network/dwcp/compression/training
./generate_synthetic_data.py --samples 10000 --output data/test.csv

# 2. Train model
./train_compression_selector_v2.py \
  --data-path data/test.csv \
  --output-dir models/test/ \
  --target-accuracy 0.98

# 3. Check results
cat models/test/training_report.json | jq '.success'
# Expected: true
```

---

## 📊 Key Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Accuracy** | ≥98% | Test set vs oracle |
| **Throughput Gain** | >10% | vs baseline compression |
| **Latency (p99)** | <10ms | Real-time inference |
| **F1 Score** | ≥0.95 | Per-class performance |

---

## 🏗️ Architecture at a Glance

```
Input Features (18) → Ensemble → Compression Choice
                      ├─ XGBoost (70%)
                      └─ Neural Net (30%)

Compression Choices:
- HDE:  High compression (10-50x), best for repetitive data
- AMST: Fast transfer (1-5x), best for high bandwidth
- None: No compression, best for small/incompressible data
```

---

## 🔧 Command Reference

### Generate Synthetic Data
```bash
./generate_synthetic_data.py \
  --samples 100000 \
  --output data/synthetic.csv \
  --seed 42
```

### Train Model
```bash
./train_compression_selector_v2.py \
  --data-path data/synthetic.csv \
  --output-dir models/v1.0/ \
  --target-accuracy 0.98 \
  --epochs 100 \
  --batch-size 64 \
  --seed 42
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | Required | Input CSV with compression metrics |
| `--output-dir` | Required | Output directory for models |
| `--target-accuracy` | 0.98 | Minimum accuracy to pass |
| `--epochs` | 100 | Max training epochs (NN) |
| `--batch-size` | 64 | Batch size for NN training |
| `--cpu-penalty` | 0.001 | CPU cost weight for oracle |
| `--seed` | 42 | Random seed |

---

## 📁 Output Files

```
models/
├── xgboost_model.json           # XGBoost model (~5 MB)
├── neural_network.keras         # Full NN model (~2 MB)
├── neural_network.tflite        # Quantized NN (~500 KB) ← Deploy this
├── feature_scaler.pkl           # Feature normalization
├── label_encoder.pkl            # Class labels
├── feature_names.json           # Feature list
└── training_report.json         # Evaluation results
```

---

## 🔍 Data Schema

### Required CSV Columns

**Network Metrics**:
- `rtt_ms`, `jitter_ms`, `available_bandwidth_mbps`, `packet_loss_rate`

**Data Characteristics**:
- `data_size_bytes`, `entropy`, `compressibility_score`

**System State**:
- `cpu_usage`, `memory_available_mb`

**HDE Metrics**:
- `hde_compression_ratio`, `hde_delta_hit_rate`, `hde_compression_time_ms`
- `hde_compressed_size_bytes`

**AMST Metrics**:
- `amst_transfer_rate_mbps`, `amst_compression_time_ms`
- `amst_compressed_size_bytes`, `amst_streams`

**Baseline Metrics**:
- `baseline_compression_ratio`, `baseline_compression_time_ms`
- `baseline_compressed_size_bytes`

**Metadata** (optional):
- `timestamp`, `region`, `az`, `link_type`, `node_id`, `peer_id`

---

## 🧮 Features (18 Total)

**Network** (4):
- rtt_ms, jitter_ms, bandwidth_mbps, network_quality

**Data** (3):
- data_size_mb, entropy, compressibility_score

**System** (2):
- cpu_usage, memory_pressure

**Historical** (7):
- hde_compression_ratio, hde_delta_hit_rate, amst_transfer_rate_mbps
- baseline_compression_ratio, hde_efficiency, amst_efficiency

**Categorical** (2):
- link_type_encoded (0=dc, 1=metro, 2=wan)
- region_encoded (0-N)

---

## 🎯 Offline Oracle

**Optimal Compression** = `argmin(transfer_time + cpu_overhead)`

```python
# For each algorithm (hde, amst, baseline):
transfer_time = compressed_size / bandwidth + rtt
cpu_cost = compression_time * cpu_usage * 0.001
total_cost = transfer_time + cpu_cost

# Select minimum
optimal = argmin(total_cost)  # → 'hde', 'amst', or 'none'
```

---

## 🔬 Model Architecture

### XGBoost (70% weight)
```python
XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    objective='multi:softprob'
)
```

### Neural Network (30% weight)
```
Input (18) → Dense(128) → BN → Dropout(0.3)
           → Dense(64)  → BN → Dropout(0.3)
           → Dense(32)  → BN → Dropout(0.2)
           → Dense(3, softmax)
```

---

## 📈 Evaluation Report Structure

```json
{
  "model_info": {
    "name": "compression_selector_ensemble",
    "version": "2.0.0"
  },
  "achieved_metrics": {
    "accuracy": 0.985,           // ← Must be ≥0.98
    "throughput_gain_pct": 12.3, // ← Must be >10%
    "f1_macro": 0.977,           // ← Must be ≥0.95
    "avg_confidence": 0.92
  },
  "per_class_metrics": {
    "hde":  {"precision": 0.98, "recall": 0.97, "f1": 0.975},
    "amst": {"precision": 0.97, "recall": 0.98, "f1": 0.975},
    "none": {"precision": 0.98, "recall": 0.98, "f1": 0.98}
  },
  "success": true  // ← Overall pass/fail
}
```

---

## 🐛 Troubleshooting

### Low Accuracy (<95%)

```bash
# Check class distribution
python3 -c "
import pandas as pd
df = pd.read_csv('data.csv')
print('Classes:', df['optimal_compression'].value_counts())
print('Balance:', df['optimal_compression'].value_counts(normalize=True))
"

# Solutions:
# 1. Collect more data (target: 100K+ samples)
# 2. Balance classes (each ≥5%)
# 3. Check feature quality
```

### Slow Training

```bash
# Quick test (5 min)
./train_compression_selector_v2.py \
  --data-path data/test.csv \
  --output-dir models/quick/ \
  --epochs 20 \
  --batch-size 128

# Full training (30 min)
# Use default parameters
```

### Model Overfitting

**Symptoms**:
- Train accuracy: >99%
- Test accuracy: <95%

**Solutions**:
- Increase dropout (0.4-0.5)
- Reduce max_depth (6-7)
- Add more training data
- Use cross-validation

---

## 🚢 Deployment Checklist

### 1. Train Model
```bash
./train_compression_selector_v2.py \
  --data-path data/production_30d.csv \
  --output-dir models/production/ \
  --target-accuracy 0.98
```

### 2. Validate Output
```bash
# Check success
jq '.success' models/production/training_report.json

# Check accuracy
jq '.achieved_metrics.accuracy' models/production/training_report.json

# Must be ≥0.98
```

### 3. Deploy to Go
```bash
# Copy models
sudo cp models/production/* /opt/dwcp/models/compression/

# Set permissions
sudo chmod 644 /opt/dwcp/models/compression/*

# Restart service
sudo systemctl restart dwcp
```

### 4. Monitor
```bash
# Check inference latency
curl http://localhost:9090/metrics | grep dwcp_ml_prediction_latency_ms

# Check accuracy (daily)
curl http://localhost:9090/metrics | grep dwcp_ml_accuracy_daily
```

---

## 📚 Documentation Links

| Document | Path |
|----------|------|
| **Architecture** | `/docs/architecture/compression_selector_architecture.md` |
| **Data Pipeline** | `/docs/models/compression_data_pipeline.md` |
| **Evaluation** | `/docs/models/compression_selector_eval.md` |
| **Go Integration** | `/docs/models/compression_selector_go_integration.md` |
| **Implementation Summary** | `/docs/models/COMPRESSION_SELECTOR_IMPLEMENTATION_SUMMARY.md` |
| **Training README** | `/backend/core/network/dwcp/compression/training/README.md` |

---

## 🔑 Key Design Decisions

**Why Ensemble?**
- XGBoost: Best for tabular data, interpretable
- Neural Network: Captures complex patterns
- Combination: 2-3% accuracy improvement

**Why Offline Oracle?**
- Ground truth based on actual measurements
- Multi-objective (throughput + resource efficiency)
- Context-aware (network, data, system state)

**Why <10ms Latency?**
- Real-time compression selection
- No user-perceived delay
- 99th percentile requirement

---

## 💡 Tips & Best Practices

### Data Collection
- ✅ Collect 30+ days of production data
- ✅ Ensure balanced classes (each ≥5%)
- ✅ Include multiple regions and link types
- ✅ Validate data quality before training

### Training
- ✅ Use cross-validation (5-fold)
- ✅ Monitor overfitting (train vs test accuracy)
- ✅ Save best model (early stopping)
- ✅ Document hyperparameters

### Deployment
- ✅ Start with shadow mode (1-2 weeks)
- ✅ Canary deployment (5% → 100%)
- ✅ Monitor accuracy daily
- ✅ Retrain weekly

---

## 📞 Support

**Questions?**
- ML Team: ml-team@company.com
- DWCP Team: dwcp-team@company.com

**Issues?**
- GitHub: github.com/company/dwcp/issues
- Slack: #dwcp-compression

---

**Quick Reference v2.0 | Last Updated: 2025-11-14**
