# DWCP Neural Training Pipeline

Production-grade neural model training infrastructure for the Distributed Wide-area Consensus Protocol (DWCP).

## Overview

This pipeline trains 4 specialized neural models with **98% accuracy targets** using SPARC methodology:

1. **Bandwidth Predictor (LSTM)** - Predict network throughput
2. **Compression Selector (Policy Network)** - Optimize compression levels
3. **Node Reliability Detector (Isolation Forest)** - Detect anomalous nodes
4. **Consensus Latency Detector (LSTM Autoencoder)** - Identify high-latency episodes

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install tensorflow scikit-learn pandas numpy joblib

# Verify installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

### Training All Models

```bash
# Train all 4 models with default settings
python backend/ml/train_dwcp_models.py \
  --data-path data/dwcp_metrics.csv \
  --output-dir checkpoints/dwcp_v1 \
  --target-accuracy 0.98 \
  --parallel

# Train specific models only
python backend/ml/train_dwcp_models.py \
  --data-path data/dwcp_metrics.csv \
  --models bandwidth,compression \
  --epochs 50
```

### Evaluation

```bash
# Comprehensive evaluation on held-out test set
python backend/ml/evaluate_dwcp_models.py \
  --checkpoints-dir checkpoints/dwcp_v1 \
  --test-data-path data/dwcp_test.csv \
  --output-dir reports/dwcp_neural_v1
```

## Data Schema

All models use the unified schema defined in `schemas/dwcp_training_schema.json`.

See full documentation in `docs/neural_training_architecture.md`.

## Model Specifications

### 1. Bandwidth Predictor (LSTM)
- **Target:** Correlation ≥ 0.98 AND MAPE < 5%
- **Input:** 7 features × 10 time steps

### 2. Compression Selector (Policy Network)
- **Target:** Accuracy ≥ 98% vs offline oracle
- **Input:** 6 features

### 3. Reliability Detector (Isolation Forest)
- **Target:** Recall ≥ 98% + PR-AUC ≥ 0.90
- **Input:** 6 features

### 4. Consensus Latency (LSTM Autoencoder)
- **Target:** Detection accuracy ≥ 98%
- **Input:** Latency time series (20 steps)

## Success Criteria

- ✅ All models achieve 98% targets
- ✅ Training reproducible
- ✅ Zero Go API changes required
- ✅ Production deployment ready

---

**Version:** 1.0.0
**Status:** Production-Ready
