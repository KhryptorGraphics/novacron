# DWCP Neural Training Architecture (SPARC-Driven)

**Version:** 1.0.0
**Date:** 2025-11-14
**Status:** Production-Ready Pipeline Design

---

## Executive Summary

This document defines the production-grade neural model training architecture for the Distributed Wide-area Consensus Protocol (DWCP). The architecture supports training of 4 specialized neural models with 98% accuracy targets using SPARC methodology.

### Key Specifications
- **Target Accuracy:** ≥98% for all models
- **Models:** Bandwidth Predictor (LSTM), Compression Selector (Policy Net), Node Reliability (Isolation Forest), Consensus Latency (LSTM Autoencoder)
- **Methodology:** SPARC (Specification → Pseudocode → Architecture → Refinement → Completion)
- **Integration:** Zero Go API changes - models integrate via inference endpoints

---

## Architecture Layers

```
┌─────────────────────────────────────────────────────────┐
│                     Data Layer                          │
│  - dwcp_training_schema.json (Unified Schema)          │
│  - Feature Extractors (Per-Model)                      │
│  - Validation Schemas (Contract Enforcement)           │
│  - Data Versioning (DVC Compatible)                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                   Training Layer                        │
│  ┌─────────────────┬─────────────────┬────────────────┐ │
│  │ Bandwidth       │ Compression     │ Reliability    │ │
│  │ Predictor       │ Selector        │ Detector       │ │
│  │ (LSTM)          │ (Policy Net)    │ (Isolation F.) │ │
│  │ train_lstm.py   │ train_comp*.py  │ train_iso*.py  │ │
│  └─────────────────┴─────────────────┴────────────────┘ │
│  ┌─────────────────────────────────────────────────────┐│
│  │ Consensus Latency (LSTM Autoencoder)                ││
│  │ train_lstm_autoencoder.py                           ││
│  └─────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                  Evaluation Layer                       │
│  - eval_bandwidth.py (Correlation, MAPE)               │
│  - eval_compression.py (Accuracy, Throughput Gain)     │
│  - eval_reliability.py (Recall, PR-AUC)                │
│  - eval_consensus_latency.py (Detection Acc, MAE)      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│                 Integration Layer                       │
│  - model_export.py (SavedModel, ONNX)                  │
│  - go_integration_test.go (API Compatibility)          │
│  - api_compatibility_check.py (Contract Validation)    │
└─────────────────────────────────────────────────────────┘
```

---

## Model Specifications

### Model 1: Bandwidth Predictor (LSTM)

**Purpose:** Predict future network throughput based on historical metrics

**Input Features (7):**
- `rtt_ms` - Round-trip time in milliseconds
- `jitter_ms` - Jitter in milliseconds
- `throughput_mbps` - Current throughput
- `packet_loss` - Packet loss percentage
- `link_type_encoded` - Link type (DC/Metro/WAN)
- `network_tier_encoded` - Network tier (1/2/3)
- `congestion_window` - TCP congestion window size

**Output:**
- Regression: `next_throughput_mbps` (continuous)
- OR Classification: `bandwidth_class` (good/degraded/bad)

**Architecture:**
```python
LSTM(64, return_sequences=True, dropout=0.2)
LSTM(32, dropout=0.2)
Dense(16, activation='relu')
Dense(1)  # OR Dense(3, activation='softmax') for classification
```

**Success Criteria:**
- **Regression:** Correlation ≥ 0.98 AND MAPE < 5%
- **Classification:** Accuracy ≥ 98%

**Training Config:**
- Sequence Length: 10 time steps
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Early Stopping: Patience 10
- Data Split: 70/15/15 (temporal)

---

### Model 2: Compression Selector (Policy Network)

**Purpose:** Select optimal compression level based on network conditions

**Input Features (6):**
- `throughput_mbps` - Current throughput
- `rtt_ms` - Round-trip time
- `link_type_encoded` - Link type
- `hde_compression_ratio` - Historical compression ratio
- `hde_delta_hit_rate` - Delta encoding hit rate
- `amst_transfer_rate` - AMST transfer rate

**Output:**
- `compression_level` (0-9) - Optimal compression level
- OR `compression_type` (none/hde/zstd)

**Architecture:**
```python
Dense(64, activation='relu', dropout=0.3)
Dense(32, activation='relu', dropout=0.3)
Dense(16, activation='relu')
Dense(10, activation='softmax')
```

**Oracle Generation:**
Offline optimal compression computed from historical throughput gain:
```python
optimal_compression = argmax(throughput_gain[compression_level])
```

**Success Criteria:**
- Decision Accuracy ≥ 98% vs offline oracle
- Measurable Throughput Gain > 0%

**Training Config:**
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Data Split: 70/15/15 (stratified)

---

### Model 3: Node Reliability (Isolation Forest)

**Purpose:** Detect anomalous node behavior indicating reliability issues

**Input Features (6):**
- `uptime_pct` - Node uptime percentage
- `failure_rate` - Failure rate
- `packet_loss` - Packet loss percentage
- `retransmits` - Retransmission count
- `error_budget_burn_rate` - Error budget consumption rate
- `rtt_ms` - Round-trip time

**Output:**
- `anomaly_score` (0-1) - Continuous anomaly score
- OR `is_anomaly` (binary) - Anomaly flag

**Algorithm:**
```python
IsolationForest(
    n_estimators=100,
    contamination='auto',
    max_samples='auto',
    random_state=42
)
```

**Labeled Incidents:**
Ground truth from production incident timestamps matched to metrics

**Success Criteria:**
- Recall ≥ 98% on labeled incidents
- PR-AUC ≥ 0.90 (Precision-Recall Area Under Curve)

**Training Config:**
- Contamination: Auto-detected
- Data Split: 85/15 (temporal)
- Feature Scaling: StandardScaler

---

### Model 4: Consensus Latency (LSTM Autoencoder)

**Purpose:** Detect high-latency consensus episodes via reconstruction error

**Input Features (1 time series):**
- `consensus_latency_ms` - Consensus latency over time window

**Output:**
- `reconstruction_error` - Reconstruction MSE
- Threshold-based binary classification

**Architecture:**
```python
# Encoder
LSTM(64, return_sequences=True, dropout=0.2)
LSTM(32, return_sequences=False)

# Decoder
RepeatVector(sequence_length)
LSTM(32, return_sequences=True, dropout=0.2)
LSTM(64, return_sequences=True)
TimeDistributed(Dense(1))
```

**Training Strategy:**
1. Train on normal latency patterns only
2. Compute reconstruction error on validation set
3. Set threshold at 98th percentile
4. Classify test samples exceeding threshold as anomalies

**Success Criteria:**
- Detection Accuracy ≥ 98% on high-latency episodes
- Reconstruction MAE < reasonable threshold (model-dependent)

**Training Config:**
- Sequence Length: 20 time steps
- Batch Size: 32
- Optimizer: Adam (lr=0.001)
- Train on Normal Data Only
- Data Split: 70/15/15 (temporal)

---

## CLI Interface Design

### Master Orchestrator
```bash
python backend/ml/train_dwcp_models.py \
  --data-path data/dwcp_metrics.csv \
  --output-dir checkpoints/dwcp_v1 \
  --models bandwidth,compression,reliability,latency \
  --target-accuracy 0.98 \
  --epochs 100 \
  --validation-split 0.2 \
  --parallel
```

**Options:**
- `--models`: Comma-separated list of models to train
- `--target-accuracy`: Minimum accuracy threshold (default: 0.98)
- `--parallel`: Train models in parallel (faster)
- `--export-format`: Model export format (keras,onnx,savedmodel)

### Individual Model Training
```bash
# Bandwidth Predictor
python backend/core/network/dwcp/prediction/training/train_lstm.py \
  --data-path data/dwcp_metrics.csv \
  --output checkpoints/bandwidth_predictor.keras \
  --target-correlation 0.98 \
  --target-mape 5.0 \
  --epochs 100

# Compression Selector
python backend/core/network/dwcp/compression/training/train_compression_selector.py \
  --data-path data/dwcp_metrics.csv \
  --output checkpoints/compression_selector.keras \
  --target-accuracy 0.98 \
  --epochs 50

# Reliability Detector
python backend/core/network/dwcp/monitoring/training/train_isolation_forest.py \
  --data-path data/dwcp_metrics.csv \
  --incidents-path data/labeled_incidents.json \
  --output checkpoints/reliability_model.pkl \
  --target-recall 0.98 \
  --target-pr-auc 0.90

# Consensus Latency
python backend/core/network/dwcp/monitoring/training/train_lstm_autoencoder.py \
  --data-path data/dwcp_metrics.csv \
  --output checkpoints/consensus_latency.keras \
  --target-accuracy 0.98 \
  --window-size 20 \
  --epochs 50
```

---

## Evaluation Reports

### Report Structure (JSON)
```json
{
  "model_name": "bandwidth_predictor_lstm",
  "version": "1.0.0",
  "training_date": "2025-11-14T12:00:00Z",
  "target_metrics": {
    "correlation": 0.98,
    "mape": 5.0
  },
  "achieved_metrics": {
    "correlation": 0.985,
    "mape": 4.2,
    "mae": 12.3,
    "rmse": 18.5
  },
  "test_set_size": 2000,
  "training_time_seconds": 847,
  "model_size_mb": 2.4,
  "success": true,
  "hyperparameters": {
    "lstm_units": [64, 32],
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs_trained": 45
  },
  "deployment_ready": true
}
```

### Aggregated Report (Master)
```json
{
  "training_session": "dwcp_neural_v1_20251114",
  "overall_success": true,
  "models_trained": 4,
  "models_passed": 4,
  "total_training_time_seconds": 3245,
  "models": {
    "bandwidth_predictor": { /* detailed report */ },
    "compression_selector": { /* detailed report */ },
    "reliability_detector": { /* detailed report */ },
    "consensus_latency": { /* detailed report */ }
  },
  "production_readiness": {
    "all_targets_met": true,
    "integration_tests_passed": true,
    "go_api_compatibility": true,
    "deployment_recommendation": "APPROVED"
  }
}
```

---

## Integration with Go (Zero API Changes)

### Model Export Formats
1. **Keras SavedModel** - Primary format for TensorFlow Serving
2. **ONNX** - Cross-platform inference
3. **Pickle (scikit-learn)** - For Isolation Forest

### Go Integration Pattern
```go
// NO CHANGES to existing DWCP Go interfaces
// Models integrate via inference endpoints

type ModelInferenceClient interface {
    PredictBandwidth(ctx context.Context, features []float64) (float64, error)
    SelectCompression(ctx context.Context, features []float64) (int, error)
    DetectAnomaly(ctx context.Context, features []float64) (bool, float64, error)
    DetectHighLatency(ctx context.Context, sequence []float64) (bool, error)
}

// gRPC/HTTP inference endpoints (separate service)
// OR ONNX runtime embedded in Go
```

### API Compatibility Test
```go
// backend/ml/go_integration_test.go
func TestBandwidthPredictorCompatibility(t *testing.T) {
    client := NewModelInferenceClient("localhost:8501")

    features := []float64{10.5, 2.3, 150.0, 0.01, 1, 2, 65536}
    prediction, err := client.PredictBandwidth(ctx, features)

    assert.NoError(t, err)
    assert.InRange(t, prediction, 0.0, 1000.0)
}
```

---

## Data Versioning & Reproducibility

### DVC Integration (Optional)
```yaml
# .dvc/config
remote:
  storage: s3://dwcp-ml-artifacts/datasets

# data/dwcp_metrics.csv.dvc
outs:
- md5: a1b2c3d4e5f6
  size: 52428800
  path: dwcp_metrics.csv
```

### Training Reproducibility
- **Random Seeds:** Fixed across all models (42)
- **Data Splits:** Deterministic temporal splits
- **Hyperparameters:** Version-controlled YAML configs
- **Environment:** Docker container with pinned dependencies

---

## Deployment Pipeline

### Phase 1: Training
1. Validate data schema compliance
2. Train all 4 models in parallel
3. Evaluate against 98% targets
4. Generate evaluation reports

### Phase 2: Validation
1. Cross-validation on held-out test set
2. Go API compatibility tests
3. Performance benchmarks (latency, throughput)
4. A/B test readiness check

### Phase 3: Deployment
1. Export models to production format
2. Deploy to inference service (TensorFlow Serving / TorchServe)
3. Canary deployment to 1% traffic
4. Monitor metrics and error rates
5. Gradual rollout to 100%

---

## Success Criteria Checklist

### Per Model
- [ ] Data schema finalized and documented
- [ ] Training script implemented with CLI interface
- [ ] Hyperparameters documented (YAML)
- [ ] Model achieves 98% target on test set
- [ ] Evaluation report generated (JSON + Markdown)
- [ ] Model exported in Go-compatible format
- [ ] Training reproducible with fixed seed
- [ ] Training time < 2 hours
- [ ] Model size < 10 MB

### Overall
- [ ] All 4 models meet 98% targets
- [ ] Master orchestrator implemented
- [ ] Documentation complete
- [ ] Integration path to Go defined
- [ ] Production deployment plan approved
- [ ] No Go API changes required
- [ ] Inference latency < 10ms (p99)

---

## Next Steps

1. **Implement Master Orchestrator** (`train_dwcp_models.py`)
2. **Update Individual Training Scripts** to align with 98% targets
3. **Create Evaluation Framework** (`evaluate_dwcp_models.py`)
4. **Generate Synthetic Training Data** for testing
5. **Run Full Training Pipeline** and validate reports
6. **Document Deployment Procedure** for production

---

**Document Owner:** ML Engineering Team
**Review Cycle:** Quarterly
**Last Reviewed:** 2025-11-14
