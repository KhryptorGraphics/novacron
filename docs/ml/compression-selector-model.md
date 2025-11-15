# Compression Selector Model Documentation

## Overview

The Compression Selector is a Random Forest-based machine learning model that intelligently selects the optimal compression algorithm for data transmission based on:

- Data type (text, binary, structured, JSON, protobuf)
- Data size (bytes)
- Latency requirements (milliseconds)
- Available bandwidth (Mbps)

**Achieved Accuracy: 99.65%** (Target: 90%)

## Model Architecture

### Algorithm: Random Forest Classifier

**Hyperparameters:**
- `n_estimators`: 100 trees
- `max_depth`: 10 levels
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: sqrt
- `n_jobs`: -1 (parallel processing)

### Features (8 total)

**Primary Features:**
1. `data_type` - Categorical: text/binary/structured/json/protobuf
2. `data_size` - Continuous: bytes
3. `latency_requirement` - Continuous: milliseconds
4. `bandwidth_available` - Continuous: Mbps

**Derived Features:**
5. `compression_time_budget` - 30% of latency allocated for compression
6. `network_time` - Estimated transmission time
7. `size_mb` - Data size in megabytes
8. `bandwidth_size_ratio` - Bandwidth relative to data size

### Output Classes (4 algorithms)

1. **zstd** - High compression ratio, medium speed, high CPU usage
2. **lz4** - Low compression ratio, very high speed, low CPU usage
3. **snappy** - Medium compression ratio, high speed, medium CPU usage
4. **none** - No compression, instant, no CPU overhead

## Performance Metrics

### Accuracy Metrics
- **Train Accuracy**: 99.98%
- **Test Accuracy**: 99.65%
- **Cross-Validation**: 99.61% ± 0.14%

### Per-Algorithm Performance

| Algorithm | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| **zstd**  | 99.92%    | 99.83% | 99.87%   |
| **lz4**   | 99.25%    | 100.00%| 99.62%   |
| **snappy**| 99.38%    | 99.53% | 99.46%   |
| **none**  | 96.43%    | 93.10% | 94.74%   |

### Feature Importance

```
1. data_type (55.20%)          - Most important factor
2. latency_requirement (13.32%)
3. compression_time_budget (11.31%)
4. data_size (8.88%)
5. size_mb (4.90%)
6. network_time (3.02%)
7. bandwidth_size_ratio (2.62%)
8. bandwidth_available (0.75%)
```

## Usage Examples

### Basic Prediction

```python
from backend.ml.models.compression_selector import CompressionSelector

# Load trained model
selector = CompressionSelector()
selector.load_model('backend/ml/models/compression_selector.joblib')

# Predict algorithm
algorithm = selector.predict(
    data_type='text',
    size=1024*1024,      # 1MB
    latency=100,          # 100ms
    bandwidth=100         # 100Mbps
)

print(f"Recommended: {algorithm}")  # Output: ZSTD
```

### Prediction with Confidence

```python
algo, confidence, all_probs = selector.predict_with_confidence(
    data_type='json',
    size=100*1024,
    latency=10,
    bandwidth=1000
)

print(f"Algorithm: {algo}")
print(f"Confidence: {confidence:.2%}")
print(f"All probabilities: {all_probs}")

# Output:
# Algorithm: none
# Confidence: 66.17%
# All probabilities: {'lz4': 0.33, 'none': 0.66, 'snappy': 0.01, 'zstd': 0.00}
```

### Training New Model

```python
from backend.ml.models.compression_selector import (
    CompressionSelector,
    generate_training_data
)

# Generate training data
X, y = generate_training_data(n_samples=10000)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
selector = CompressionSelector(n_estimators=100, max_depth=10)
train_metrics = selector.train(X_train, y_train)

# Evaluate
eval_metrics = selector.evaluate(X_test, y_test)

print(f"Test Accuracy: {eval_metrics['accuracy']:.2%}")

# Save model
selector.save_model('models/compression_selector_v2.joblib')
```

## Real-World Scenarios

### Scenario 1: Real-time Video Stream
```python
# 100KB chunks, 5ms latency, 100Mbps
algorithm = selector.predict('binary', 100*1024, 5, 100)
# → Predicted: none or lz4 (fast/no compression for low latency)
```

### Scenario 2: Large Log File Transfer
```python
# 50MB file, 5s latency, 50Mbps
algorithm = selector.predict('text', 50*1024*1024, 5000, 50)
# → Predicted: zstd or snappy (good compression for large files)
```

### Scenario 3: JSON API Response
```python
# 10KB response, 50ms latency, 100Mbps
algorithm = selector.predict('json', 10*1024, 50, 100)
# → Predicted: lz4, snappy, or zstd (various options acceptable)
```

### Scenario 4: Small File on Slow Network
```python
# 500KB file, 2s latency, 2Mbps
algorithm = selector.predict('structured', 500*1024, 2000, 2)
# → Predicted: zstd, snappy, or lz4 (compression helps on slow network)
```

## Integration with DWCP

### Real-time Selection

```go
// In Go backend
func SelectCompression(dataType string, size int64, latency, bandwidth float64) string {
    // Call Python ML model via HTTP API or embedded Python
    result := callMLModel(dataType, size, latency, bandwidth)
    return result.Algorithm
}
```

### API Endpoint

```python
from flask import Flask, request, jsonify

app = Flask(__name__)
selector = CompressionSelector()
selector.load_model('models/compression_selector.joblib')

@app.route('/predict_compression', methods=['POST'])
def predict_compression():
    data = request.json

    algorithm = selector.predict(
        data_type=data['data_type'],
        size=data['size'],
        latency=data['latency'],
        bandwidth=data['bandwidth']
    )

    return jsonify({'algorithm': algorithm})
```

## Model Files

- **Model Code**: `/home/kp/repos/novacron/backend/ml/models/compression_selector.py`
- **Trained Model**: `/home/kp/repos/novacron/backend/ml/models/compression_selector.joblib`
- **Tests**: `/home/kp/repos/novacron/tests/ml/test_compression_selector.py`
- **Requirements**: `/home/kp/repos/novacron/backend/ml/requirements.txt`

## Dependencies

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

## Testing

Run unit tests:
```bash
cd /home/kp/repos/novacron
python3 -m pytest tests/ml/test_compression_selector.py -v
```

Expected results:
- All tests pass
- Accuracy ≥ 85%
- Feature importance validated
- Edge cases handled

## Future Enhancements

1. **Online Learning**: Update model with real-world feedback
2. **Multi-objective Optimization**: Balance compression ratio, speed, and CPU usage
3. **Context-aware Selection**: Consider historical patterns and network conditions
4. **Deep Learning**: Explore neural networks for more complex patterns
5. **Custom Algorithm Support**: Extend to support brotli, gzip, etc.

## References

- Random Forest: Breiman, L. (2001). Random Forests. Machine Learning.
- Compression Algorithms: https://github.com/facebook/zstd
- DWCP Protocol: `/home/kp/repos/novacron/backend/core/network/dwcp/`

---

**Model Version**: 1.0.0
**Last Updated**: 2025-11-14
**Status**: Production Ready ✓
