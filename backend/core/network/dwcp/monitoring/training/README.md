# Node Reliability Model Training

This directory contains training scripts for the DWCP Node Reliability anomaly detection model.

## Quick Start

```bash
# Fast demo (~30 seconds)
python train_isolation_forest_fast.py --n-samples 5000

# Full training (~15 minutes)
python train_isolation_forest.py --synthetic --n-samples 10000

# With real labeled data
python train_isolation_forest.py --data /path/to/labeled_data.csv
```

## Scripts

- `train_isolation_forest.py` - Main training script with full hyperparameter search
- `train_isolation_forest_fast.py` - Fast demo version (reduced search space)
- `train_node_reliability_tuned.py` - Optimized with realistic synthetic data

## Model Status

⚠️ **Isolation Forest does not meet production requirements**

- Achieves: 98% recall, 95% FP rate
- Target: 98% recall, <5% FP rate
- **Recommendation**: Use supervised learning (XGBoost/Random Forest)

See `/docs/models/ISOLATION_FOREST_FINAL_REPORT.md` for details.

## Data Format

CSV with columns:
```
timestamp, node_id, region, az,
error_rate, timeout_rate, latency_p50, latency_p99,
sla_violations, connection_failures, packet_loss_rate,
cpu_usage, memory_usage, disk_io,
dwcp_mode, network_tier, label
```

## Output

Models saved to `../models/`:
- `isolation_forest_node_reliability.pkl` - Trained model
- `scaler_node_reliability.pkl` - Feature scaler
- `model_metadata_node_reliability.json` - Metadata
- `hyperparameters_node_reliability.json` - Hyperparameters

Evaluation report: `/docs/models/node_reliability_eval.md`

## Documentation

- `/docs/models/QUICK_START_NODE_RELIABILITY.md` - Quick start guide
- `/docs/models/ISOLATION_FOREST_FINAL_REPORT.md` - Full analysis
- `/docs/models/NODE_RELIABILITY_MODEL_SUMMARY.md` - Implementation details
