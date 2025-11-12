# ML Model Training for DWCP Anomaly Detection

This directory contains training scripts for ML-based anomaly detection models used in DWCP monitoring.

## Models

### 1. Isolation Forest
- **File**: `train_isolation_forest.py`
- **Purpose**: Detects outliers in multivariate metrics
- **Algorithm**: Ensemble of isolation trees
- **Best for**: General anomaly detection, works well with limited data

### 2. LSTM Autoencoder
- **File**: `train_lstm_autoencoder.py`
- **Purpose**: Learns normal time-series patterns
- **Algorithm**: Sequence-to-sequence LSTM neural network
- **Best for**: Time-series anomalies, detecting pattern deviations

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Training

### Isolation Forest

```bash
# With synthetic data (for testing)
python train_isolation_forest.py --synthetic --output ../models

# With real data
python train_isolation_forest.py --data /path/to/normal_data.csv --output ../models

# Custom parameters
python train_isolation_forest.py \
    --data training_data.csv \
    --output ../models \
    --contamination 0.01 \
    --n-estimators 200
```

### LSTM Autoencoder

```bash
# With synthetic data (for testing)
python train_lstm_autoencoder.py --synthetic --output ../models

# With real data
python train_lstm_autoencoder.py --data /path/to/timeseries_data.csv --output ../models

# Custom parameters
python train_lstm_autoencoder.py \
    --data training_data.csv \
    --output ../models \
    --window-size 10 \
    --epochs 100 \
    --batch-size 64 \
    --encoding-dim 32
```

## Data Format

### For Isolation Forest

CSV file with columns:
```
timestamp,bandwidth,latency,packet_loss,jitter,cpu_usage,memory_usage,error_rate
2024-01-01T00:00:00Z,105.2,10.5,0.01,1.2,45.3,58.7,0.001
2024-01-01T00:01:00Z,103.8,11.2,0.02,1.1,46.1,59.2,0.001
...
```

### For LSTM Autoencoder

Same CSV format as above. The script will automatically create sliding windows from the time series.

## Metrics

Models are trained on 7 features:

1. **bandwidth** (Mbps) - Network bandwidth utilization
2. **latency** (ms) - Network latency
3. **packet_loss** (%) - Packet loss rate
4. **jitter** (ms) - Network jitter
5. **cpu_usage** (%) - CPU utilization
6. **memory_usage** (%) - Memory utilization
7. **error_rate** (%) - Error rate

## Output

After training, the following files are generated in the output directory:

### Isolation Forest
- `isolation_forest.pkl` - Scikit-learn model
- `scaler.pkl` - Feature scaler
- `isolation_forest.onnx` - ONNX model (if conversion succeeds)
- `model_metadata.json` - Model metadata

### LSTM Autoencoder
- `lstm_autoencoder.h5` - Keras model
- `lstm_scaler.pkl` - Feature scaler
- `lstm_autoencoder.onnx` - ONNX model (if conversion succeeds)
- `lstm_metadata.json` - Model metadata and threshold

## Using Trained Models

The Go implementation in `isolation_forest.go` and `lstm_autoencoder.go` will load these models using:
- ONNX Runtime (for production inference)
- Or simplified implementations (for testing without ONNX)

## Retraining

Models should be retrained periodically with fresh normal data:

1. Collect normal operating data (no anomalies) for at least 1 week
2. Export to CSV format
3. Run training scripts
4. Replace model files in deployment
5. Restart monitoring service

## Performance Tips

1. **Data Quality**: Use only confirmed normal data for training
2. **Data Quantity**: Minimum 10,000 samples recommended
3. **Feature Scaling**: Always use the same scaler for inference
4. **Threshold Tuning**: Adjust contamination/percentile based on false positive rate
5. **Regular Retraining**: Retrain monthly to adapt to changing patterns

## Monitoring Model Performance

Track these metrics in production:

- **False Positive Rate**: Anomalies detected that are actually normal
- **False Negative Rate**: Real anomalies missed by the model
- **Detection Latency**: Time from anomaly occurrence to detection
- **Model Drift**: Changes in data distribution over time

## Troubleshooting

### ONNX Export Fails
- Ensure `tf2onnx` and `skl2onnx` are installed
- Check TensorFlow/sklearn versions are compatible
- ONNX export is optional - simplified implementations work without it

### Poor Detection Accuracy
- Increase training data size
- Adjust contamination parameter
- Retrain with more recent data
- Check for data quality issues

### High False Positives
- Increase contamination parameter (Isolation Forest)
- Increase threshold percentile (LSTM)
- Ensure training data represents all normal patterns

### High False Negatives
- Decrease contamination parameter
- Decrease threshold percentile
- Add more anomaly types to evaluation
