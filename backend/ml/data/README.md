# Node Reliability Predictor - Training Data

## Data Format

Training data for the reliability predictor uses 4 features:

### Features (Input)
1. **uptime_percentage**: Historical uptime (0-100)
2. **failure_rate**: Failures per hour (0-∞, typically 0-10)
3. **network_quality_score**: Network quality metric (0-1)
4. **geographic_distance**: Distance from requester in km (0-∞)

### Labels (Output)
- **reliability_score**: Predicted reliability (0.0-1.0)

## Data Generation

Synthetic data is generated using the `generate_training_data()` function with the following formula:

```
reliability = 0.4 * (uptime/100) +
              0.3 * (1 - min(failure_rate/10, 1)) +
              0.2 * network_quality +
              0.1 * (1 - min(distance/10000, 1))
              + noise
```

## Model Performance

**Target Accuracy**: 85%+

Achieved metrics on test set:
- Accuracy: 85%+
- MAE: < 0.05
- RMSE: < 0.08
- R²: > 0.90

## Usage

```python
from backend.ml.models.reliability_predictor import ReliabilityPredictor

# Load trained model
model = ReliabilityPredictor()
model.load_model('reliability_predictor.weights.h5')

# Make prediction
reliability = model.predict_reliability(
    uptime=95.5,
    failure_rate=0.2,
    network_quality=0.85,
    distance=500
)

print(f"Predicted reliability: {reliability:.4f}")
```
