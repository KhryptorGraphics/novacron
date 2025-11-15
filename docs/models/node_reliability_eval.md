# Node Reliability Isolation Forest - Evaluation Report

**Generated**: 2025-11-14 08:46:46

## Model Configuration

**Hyperparameters**:
- n_estimators: 100
- max_samples: 256
- max_features: 0.75
- contamination: 0.0203

**Decision Threshold**: -0.376980

## Performance Metrics

### Classification Metrics
- **Recall**: 0.8421 (Target: ≥0.98)
- **Precision**: 0.0176
- **F1 Score**: 0.0345
- **ROC-AUC**: 0.4962
- **False Positive Rate**: 0.9103 (Target: <0.05)

### Confusion Matrix

|              | Predicted Normal | Predicted Incident |
|--------------|------------------|-------------------|
| **Actual Normal**   | 88         | 893          |
| **Actual Incident** | 3         | 16          |

- True Negatives (TN): 88
- False Positives (FP): 893
- False Negatives (FN): 3
- True Positives (TP): 16

### Test Set Statistics
- Total Samples: 1000
- Incident Samples: 19
- Incident Rate: 1.90%

## Target Achievement

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Recall | ≥98% | 84.21% | ✗ FAIL |
| FP Rate | <5% | 91.03% | ✗ FAIL |


## Recommendations

⚠ Recall (0.8421) is below target (0.98). Consider:
  - Lowering decision threshold
  - Increasing training data with more incident examples
  - Engineering additional features

⚠ False positive rate (0.9103) exceeds target (0.05). Consider:
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
