# Machine Learning Models for Novacron

This directory contains ML models used in the Novacron distributed computing platform.

## Models

### 1. Compression Selector (Random Forest)

**Purpose**: Intelligently select optimal compression algorithm for data transmission

**Accuracy**: 99.65% (Target: 90%)

**Location**: `models/compression_selector.py`

**Documentation**: `../../docs/ml/compression-selector-model.md`

**Features**:
- Data type classification (text/binary/structured/json/protobuf)
- Size-aware selection (handles 100B to 100MB+)
- Latency-conscious (1ms to 1s+ budgets)
- Bandwidth-adaptive (1Mbps to 1Gbps+)

**Algorithms**:
- **zstd**: Best compression, slower
- **lz4**: Fast compression, lower ratio
- **snappy**: Balanced speed/ratio
- **none**: No compression overhead

## Installation

```bash
# Install ML dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from models.compression_selector import CompressionSelector

# Load model
selector = CompressionSelector()
selector.load_model('models/compression_selector.joblib')

# Predict
algorithm = selector.predict(
    data_type='text',
    size=1024*1024,  # 1MB
    latency=100,     # 100ms
    bandwidth=100    # 100Mbps
)

print(f"Use: {algorithm}")  # Output: zstd
```

## Training

```bash
# Train compression selector
python3 models/compression_selector.py

# Output:
# Train Accuracy: 99.98%
# Test Accuracy: 99.65%
# Model saved to: models/compression_selector.joblib
```

## Testing

```bash
# Run all ML tests
pytest ../tests/ml/ -v

# Run specific model tests
pytest ../tests/ml/test_compression_selector.py -v
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Compression Selector | 99.65% | 99.25% | 99.62% | 99.43% |

## Directory Structure

```
backend/ml/
├── models/
│   ├── compression_selector.py      # Main model code
│   └── compression_selector.joblib  # Trained model
├── data/                             # Training data (if needed)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file

tests/ml/
└── test_compression_selector.py     # Unit tests

docs/ml/
└── compression-selector-model.md    # Full documentation
```

## Integration with DWCP

The compression selector integrates with the DWCP (Dynamic Weighted Communication Protocol) to optimize data transmission:

1. **Pre-transmission**: Query model for optimal algorithm
2. **Compression**: Apply selected algorithm
3. **Transmission**: Send compressed data
4. **Feedback**: Log performance for future training

## Future Models

Planned ML models for Novacron:

1. **Task Scheduler** - Predict optimal task allocation
2. **Load Balancer** - Predict node capacity and health
3. **Anomaly Detector** - Identify system issues
4. **Performance Predictor** - Estimate task completion time
5. **Resource Optimizer** - Optimize resource allocation

---

**Last Updated**: 2025-11-14
**Status**: Production Ready ✓
