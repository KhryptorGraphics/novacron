# DWCP Compression Selector - ML Architecture Design

## Executive Summary

**Objective**: Design and implement a production-ready ML training pipeline for DWCP Compression Selector that achieves ≥98% decision accuracy vs offline oracle, with measurable throughput gains in distributed WAN environments.

**Target Metrics**:
- Decision Accuracy: ≥98% vs offline oracle
- Throughput Gain: >10% over baseline
- Inference Latency: <10ms (real-time requirement)
- Model Size: <50MB (deployment constraint)

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DWCP Production System                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ HDE Engine   │    │ AMST Engine  │    │  Baseline    │     │
│  │ (Delta+Dict) │    │ (Multi-Stream)│   │  (Zstd)      │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                    │                    │              │
│         └────────────────────┴────────────────────┘              │
│                              │                                   │
│                    ┌─────────▼──────────┐                       │
│                    │  Compression       │                       │
│                    │  Selector (ML)     │  ◄── Real-time       │
│                    │  - Policy Network  │      Metrics         │
│                    │  - Feature Eng.    │                       │
│                    │  - Inference <10ms │                       │
│                    └─────────┬──────────┘                       │
│                              │                                   │
│                    ┌─────────▼──────────┐                       │
│                    │  Selected Algorithm│                       │
│                    │  (HDE/AMST/None)   │                       │
│                    └────────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                 Offline Training Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │ Data         │───▶│ Oracle       │───▶│ Supervised   │     │
│  │ Collection   │    │ Computation  │    │ Learning     │     │
│  │ (Metrics)    │    │ (Optimal)    │    │ (Policy Net) │     │
│  └──────────────┘    └──────────────┘    └──────┬───────┘     │
│                                                   │              │
│                                          ┌────────▼───────┐     │
│                                          │ Model Export   │     │
│                                          │ (TFLite/ONNX)  │     │
│                                          └────────┬───────┘     │
│                                                   │              │
│                                          ┌────────▼───────┐     │
│                                          │ Go Integration │     │
│                                          │ (Inference)    │     │
│                                          └────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Compression Algorithms Analysis

### 2.1 Available Algorithms

| Algorithm | Use Case | Compression Ratio | Speed | CPU Cost | Best For |
|-----------|----------|-------------------|-------|----------|----------|
| **HDE** (Hierarchical Delta Encoding) | Delta encoding with dictionary | 10-50x | Medium | Medium | Repetitive data, VM state |
| **AMST** (Adaptive Multi-Stream) | Parallel stream transfer | 1-5x | Fast | Low | Large transfers, high bandwidth |
| **Baseline** (Zstd) | Standard compression | 2-10x | Fast | Low | General purpose, fallback |
| **None** | No compression | 1x | Fastest | None | Small data, incompressible |

### 2.2 Compression Selection Criteria

**Decision Factors**:
1. **Network Characteristics**: RTT, bandwidth, jitter, packet loss
2. **Data Characteristics**: Size, entropy, compressibility, delta potential
3. **System State**: CPU usage, memory availability
4. **Historical Performance**: Per-algorithm success rates

---

## 3. Offline Oracle Design

### 3.1 Oracle Definition

**Optimal Compression** = `argmin(transfer_time + cpu_overhead)`

Where:
```
transfer_time = data_size_compressed / effective_bandwidth
cpu_overhead = compression_time + decompression_time
effective_bandwidth = available_bandwidth * (1 - packet_loss_rate)
```

### 3.2 Oracle Computation Algorithm

```python
def compute_oracle_compression(metrics):
    """
    Compute optimal compression for each sample based on:
    1. Actual transfer time measurements
    2. CPU overhead measurements
    3. Network efficiency

    Returns: Optimal compression choice (hde/amst/none)
    """
    candidates = ['hde', 'amst', 'none']
    costs = {}

    for algo in candidates:
        # Transfer time (network cost)
        compressed_size = metrics[f'{algo}_compressed_size']
        bandwidth = metrics['available_bandwidth_mbps']
        transfer_time = compressed_size / bandwidth

        # CPU overhead (processing cost)
        cpu_time = metrics[f'{algo}_compression_time_ms']
        cpu_overhead = cpu_time * metrics['cpu_usage']

        # Total cost (weighted sum)
        costs[algo] = transfer_time + cpu_overhead

    # Select minimum cost
    optimal = min(costs, key=costs.get)
    return optimal
```

### 3.3 Oracle Validation

- **Ground Truth**: Based on actual production measurements
- **Multi-Objective**: Balances throughput, latency, and resource usage
- **Context-Aware**: Considers network mode (datacenter/wan/hybrid)

---

## 4. ML Model Architecture

### 4.1 Policy Network Design

**Architecture**: Gradient Boosted Decision Trees (XGBoost) + Neural Network Ensemble

**Rationale**:
- **XGBoost**: Excellent for tabular data with mixed features
- **Neural Network**: Captures non-linear interactions
- **Ensemble**: Combines strengths of both approaches

#### 4.1.1 XGBoost Model

```python
XGBoostClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=3  # hde, amst, none
)
```

**Features** (18 input features):
1. Network metrics (6): rtt_ms, jitter_ms, bandwidth_mbps, packet_loss_rate, link_type, region
2. Data characteristics (5): size_bytes, entropy, compressibility_score, delta_potential, data_type
3. System state (4): cpu_usage, memory_available, active_transfers, time_of_day
4. Historical performance (3): recent_hde_ratio, recent_amst_ratio, baseline_ratio

**Output**: Probability distribution over 3 classes [P(hde), P(amst), P(none)]

#### 4.1.2 Neural Network Model

```python
Sequential([
    Dense(128, activation='relu', input_dim=18),
    BatchNormalization(),
    Dropout(0.3),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')
])
```

#### 4.1.3 Ensemble Strategy

```python
final_prediction = 0.7 * xgboost_prob + 0.3 * neural_net_prob
selected_compression = argmax(final_prediction)
confidence = max(final_prediction)
```

### 4.2 Feature Engineering

#### 4.2.1 Network Features
- **RTT Bins**: Categorize into datacenter (<1ms), metro (1-10ms), wan (>10ms)
- **Bandwidth Ratio**: Current bandwidth / historical average
- **Network Stability**: Moving average of jitter

#### 4.2.2 Data Features
- **Entropy Score**: Shannon entropy normalized to [0, 1]
- **Delta Potential**: Estimated delta compression ratio based on recent baselines
- **Size Category**: Small (<1KB), medium (1KB-1MB), large (>1MB)

#### 4.2.3 Temporal Features
- **Time of Day**: Hour of day (traffic patterns)
- **Recent Performance**: Exponential moving average of compression ratios

### 4.3 Inference Optimization

**Requirements**: <10ms latency for real-time selection

**Strategies**:
1. **Model Quantization**: Convert to int8 for faster inference
2. **Feature Caching**: Pre-compute static features
3. **Batch Prediction**: Predict next N transfers in advance
4. **Compiled Models**: Use TFLite or ONNX runtime

---

## 5. Training Pipeline Architecture

### 5.1 Data Collection Pipeline

```
Production DWCP Nodes
        │
        ▼
┌─────────────────┐
│ Metrics Logger  │  ◄── Logs every compression operation
│ (Go)            │      - Network state
│                 │      - Algorithm used
│                 │      - Performance metrics
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Time-Series DB  │  ◄── InfluxDB / Prometheus
│ (Metrics Store) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ETL Pipeline    │  ◄── Extract, Transform, Load
│ (Python)        │      - Aggregation
│                 │      - Feature extraction
│                 │      - Oracle computation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Training Data   │  ◄── CSV / Parquet
│ (Labeled)       │      - Features + Labels
└─────────────────┘
```

### 5.2 Training Workflow

```
Step 1: Data Preparation
    ├── Load metrics from production
    ├── Compute oracle labels (optimal compression)
    ├── Extract features
    ├── Split: Train (70%), Validation (15%), Test (15%)
    └── Standardize features

Step 2: Model Training
    ├── XGBoost: Train on tabular features
    ├── Neural Net: Train on normalized features
    ├── Ensemble: Combine predictions
    └── Hyperparameter tuning (Optuna)

Step 3: Evaluation
    ├── Test accuracy vs oracle (target ≥98%)
    ├── Per-class metrics (precision, recall, F1)
    ├── Throughput gain estimation
    └── Inference latency benchmarking

Step 4: Model Export
    ├── Export to TFLite (optimized)
    ├── Export to ONNX (cross-platform)
    └── Generate Go integration code

Step 5: Deployment
    ├── Integration testing
    ├── Canary deployment (5% traffic)
    ├── A/B testing (monitor metrics)
    └── Full rollout
```

### 5.3 Continuous Learning

**Feedback Loop**:
1. Production model makes predictions
2. Actual performance is measured
3. New training data is collected
4. Model is retrained weekly
5. Performance regression testing
6. Automatic rollback if accuracy drops

---

## 6. Data Schema

### 6.1 Training Data Schema

```sql
CREATE TABLE compression_training_data (
    -- Metadata
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    region VARCHAR(50),
    availability_zone VARCHAR(50),

    -- Network Characteristics
    link_type VARCHAR(20),  -- dc, metro, wan
    node_id VARCHAR(100),
    peer_id VARCHAR(100),
    rtt_ms FLOAT,
    jitter_ms FLOAT,
    available_bandwidth_mbps FLOAT,
    packet_loss_rate FLOAT,

    -- Data Characteristics
    data_size_bytes BIGINT,
    data_type VARCHAR(50),  -- vm_memory, vm_disk, database, etc.
    entropy FLOAT,
    compressibility_score FLOAT,

    -- System State
    cpu_usage FLOAT,
    memory_available_mb BIGINT,
    active_transfers INT,

    -- HDE Metrics
    hde_compression_ratio FLOAT,
    hde_delta_hit_rate FLOAT,
    hde_compression_time_ms FLOAT,
    hde_compressed_size_bytes BIGINT,
    hde_throughput_mbps FLOAT,

    -- AMST Metrics
    amst_streams INT,
    amst_transfer_rate_mbps FLOAT,
    amst_compression_time_ms FLOAT,
    amst_compressed_size_bytes BIGINT,

    -- Baseline Metrics
    baseline_compression_ratio FLOAT,
    baseline_compression_time_ms FLOAT,
    baseline_compressed_size_bytes BIGINT,

    -- Ground Truth
    current_compression VARCHAR(20),  -- Algorithm actually used
    optimal_compression VARCHAR(20),  -- Oracle-computed optimal

    -- Performance Outcomes
    actual_transfer_time_ms FLOAT,
    actual_throughput_mbps FLOAT,
    total_cpu_overhead_ms FLOAT
);

CREATE INDEX idx_timestamp ON compression_training_data(timestamp);
CREATE INDEX idx_link_type ON compression_training_data(link_type);
CREATE INDEX idx_optimal ON compression_training_data(optimal_compression);
```

---

## 7. Integration with Go

### 7.1 Go Inference Interface

```go
// CompressionSelectorML provides ML-based compression selection
type CompressionSelectorML struct {
    modelPath string
    interpreter *tflite.Interpreter
    featureExtractor *FeatureExtractor
    cache *InferenceCache
}

// SelectCompression performs real-time ML inference
func (cs *CompressionSelectorML) SelectCompression(
    ctx context.Context,
    networkState *NetworkState,
    dataChars *DataCharacteristics,
) (CompressionChoice, float64, error) {

    // Extract features
    features := cs.featureExtractor.Extract(networkState, dataChars)

    // Run inference (< 10ms)
    probabilities, err := cs.interpreter.Predict(features)
    if err != nil {
        return FallbackCompression, 0.0, err
    }

    // Select highest probability
    choice := argmax(probabilities)
    confidence := max(probabilities)

    return choice, confidence, nil
}
```

### 7.2 Fallback Strategy

**Edge Cases**:
1. Model inference fails → Use rule-based selection
2. Low confidence (<0.6) → Use conservative baseline
3. Network mode changes → Invalidate cache, re-predict

---

## 8. Performance Targets

### 8.1 Model Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Decision Accuracy | ≥98% | Test set accuracy vs oracle |
| Precision (HDE) | ≥95% | TP / (TP + FP) |
| Recall (HDE) | ≥95% | TP / (TP + FN) |
| F1 Score | ≥0.96 | Harmonic mean |

### 8.2 System Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Throughput Gain | >10% | vs baseline compression |
| Inference Latency | <10ms | 99th percentile |
| Model Size | <50MB | Deployed model binary |
| Memory Usage | <100MB | Runtime memory footprint |

### 8.3 Production Metrics

| Metric | Target | Monitoring |
|--------|--------|------------|
| Prediction Accuracy | >95% | Daily validation |
| False Positive Rate | <5% | Suboptimal selections |
| System Overhead | <2% | CPU/memory impact |
| Model Drift | <3% accuracy drop | Weekly retraining trigger |

---

## 9. Deployment Strategy

### 9.1 Staged Rollout

**Phase 1: Shadow Mode** (Week 1-2)
- Model runs alongside rule-based selector
- Predictions logged but not used
- Accuracy measured vs production outcomes

**Phase 2: Canary Deployment** (Week 3-4)
- 5% of traffic uses ML selector
- Monitor for regressions
- Automatic rollback if accuracy <95%

**Phase 3: Gradual Rollout** (Week 5-8)
- 25% → 50% → 75% → 100%
- Per-region rollout
- Continuous monitoring

**Phase 4: Full Production** (Week 9+)
- 100% ML-based selection
- Weekly model retraining
- Continuous improvement

### 9.2 Monitoring & Alerting

**Key Metrics**:
- Prediction accuracy (daily)
- Throughput gain (hourly)
- Inference latency (p50, p95, p99)
- Model staleness (days since last training)

**Alerts**:
- Accuracy drop >3% → Trigger retraining
- Latency p99 >15ms → Investigate inference optimization
- Throughput regression >5% → Rollback to previous model

---

## 10. Risk Mitigation

### 10.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model overfitting | High | Medium | Cross-validation, regularization |
| Inference latency | High | Low | Model quantization, caching |
| Data drift | Medium | High | Weekly retraining, monitoring |
| Integration bugs | Medium | Medium | Extensive testing, canary deployment |

### 10.2 Operational Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Training data quality | High | Data validation pipeline |
| Model deployment failure | High | Automated rollback |
| Production performance regression | High | Shadow mode + gradual rollout |

---

## 11. Future Enhancements

### 11.1 Short-term (3-6 months)
- Online learning: Continuous model updates
- Per-region models: Geographic optimization
- Cost-aware selection: Include pricing signals

### 11.2 Long-term (6-12 months)
- Reinforcement learning: Direct throughput optimization
- Multi-task learning: Joint compression + routing optimization
- Federated learning: Distributed training across regions

---

## 12. Success Criteria

**Project Success** = ALL of the following:
1. ✅ Model achieves ≥98% accuracy vs offline oracle
2. ✅ Measurable throughput gain >10% over baseline
3. ✅ Inference latency <10ms (p99)
4. ✅ No production incidents during rollout
5. ✅ Model deployed to 100% of traffic within 8 weeks

**Acceptance Testing**:
- Offline: Test set accuracy ≥98%
- Shadow: 7 days with accuracy ≥95% on production data
- Canary: 14 days with throughput gain ≥10%
- Full: 30 days with no regressions

---

## 13. Documentation Deliverables

1. **Architecture Design** (this document)
2. **Data Pipeline Specification** (compression_data_pipeline.md)
3. **Model Training Guide** (compression_selector_training.md)
4. **Evaluation Report** (compression_selector_eval.md)
5. **Go Integration Guide** (compression_selector_integration.md)
6. **Operations Runbook** (compression_selector_ops.md)

---

## Appendix A: Technology Stack

**Training**:
- Python 3.9+
- TensorFlow 2.13 / PyTorch 2.0
- XGBoost 1.7+
- Scikit-learn 1.3
- Pandas, NumPy

**Deployment**:
- TensorFlow Lite (Go bindings)
- ONNX Runtime (cross-platform)
- Go 1.21+

**Infrastructure**:
- InfluxDB (metrics storage)
- Prometheus (monitoring)
- Grafana (dashboards)
- Airflow (pipeline orchestration)

---

## Appendix B: References

1. DWCP Protocol Specification
2. HDE Algorithm Documentation
3. AMST Performance Analysis
4. Production Metrics Schema
5. ML Model Deployment Best Practices
