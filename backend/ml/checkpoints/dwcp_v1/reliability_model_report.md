# Node Reliability Isolation Forest - Evaluation Report

**Generated**: 2025-11-14 09:59:56

## Model Configuration

**Hyperparameters**:
- n_estimators: 100
- max_samples: 128
- max_features: 0.5
- contamination: 0.0102

**Decision Threshold**: -0.385641

## Performance Metrics

### Classification Metrics
- **Recall**: 1.0000 (Target: ≥0.98)
- **Precision**: 0.0192
- **F1 Score**: 0.0377
- **ROC-AUC**: 0.5136
- **False Positive Rate**: 0.9893 (Target: <0.05)

### Confusion Matrix

|              | Predicted Normal | Predicted Incident |
|--------------|------------------|-------------------|
| **Actual Normal**   | 21         | 1941          |
| **Actual Incident** | 0         | 38          |

- True Negatives (TN): 21
- False Positives (FP): 1941
- False Negatives (FN): 0
- True Positives (TP): 38

### Test Set Statistics
- Total Samples: 2000
- Incident Samples: 38
- Incident Rate: 1.90%

## Target Achievement

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Recall | ≥98% | 100.00% | ✓ PASS |
| FP Rate | <5% | 98.93% | ✗ FAIL |


## Feature Importance (Top 20)

Features ranked by impact on model performance:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cpu_usage_acceleration | 0.001010 |
| 2 | memory_usage_rolling_max_30 | 0.000973 |
| 3 | cpu_usage_rolling_min_15 | 0.000973 |
| 4 | tier_tier3 | 0.000937 |
| 5 | timeout_rate_rolling_min_5 | 0.000919 |
| 6 | timeout_rate_rolling_mean_15 | 0.000130 |
| 7 | memory_usage_rolling_min_5 | 0.000112 |
| 8 | connection_failures_rolling_max_5 | 0.000112 |
| 9 | connection_failures_rolling_mean_30 | 0.000093 |
| 10 | error_rate_rolling_std_15 | 0.000093 |
| 11 | latency_p99_rolling_max_30 | 0.000075 |
| 12 | packet_loss_rate_rolling_mean_15 | 0.000056 |
| 13 | latency_p99_rolling_std_5 | 0.000056 |
| 14 | latency_p99_rolling_max_5 | 0.000056 |
| 15 | latency_p50_rolling_std_30 | 0.000037 |
| 16 | cpu_usage_rolling_mean_5 | 0.000037 |
| 17 | connection_failures_rolling_std_5 | 0.000037 |
| 18 | disk_io | 0.000037 |
| 19 | cpu_usage_rolling_max_5 | 0.000037 |
| 20 | disk_io_rolling_min_30 | 0.000037 |

## Recommendations

⚠ False positive rate (0.9893) exceeds target (0.05). Consider:
  - Raising decision threshold
  - Reducing contamination parameter
  - Improving feature engineering


## Model Deployment

**Model Files**:
- isolation_forest_node_reliability.pkl - Trained model
- scaler_node_reliability.pkl - Feature scaler
- model_metadata_node_reliability.json - Model configuration
- hyperparameters_node_reliability.json - Optimal hyperparameters

**CLI Reproduction**:
```bash
python train_isolation_forest.py \
  --data /path/to/data.csv \
  --output ../models \
  --target-recall 0.98 \
  --max-fp-rate 0.05
```

## Next Steps

1. Validate model on production data stream
2. Monitor false positive/negative rates in production
3. Retrain monthly with updated incident data
4. A/B test against current alerting system
5. Integrate with DWCP monitoring dashboard
