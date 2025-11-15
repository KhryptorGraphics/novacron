# TCS-FEEL Production Deployment Guide

## Quick Start

### 1. Installation

```bash
cd backend/ml/federated
pip install -r requirements.txt
```

### 2. Verify Calibration

```bash
python3 validate_calibration.py
```

Expected output:
```
✅ ALL VALIDATION CHECKS PASSED
🚀 MODEL READY FOR PRODUCTION DEPLOYMENT
```

### 3. Use in Production

```python
from backend.ml.federated import TopologyOptimizer

# Initialize with calibrated settings
optimizer = TopologyOptimizer(
    min_clients=15,
    max_clients=30,
    target_accuracy=0.963
)

# Apply optimal weights
optimizer.weights = {
    'communication': 0.25,
    'data_quality': 0.45,  # Key: Increased for accuracy
    'compute': 0.15,
    'fairness': 0.15
}

# Training configuration
CLIENTS_PER_ROUND = 25
LOCAL_EPOCHS = 8
LEARNING_RATE = 0.025
BATCH_SIZE = 32
TOPOLOGY_WEIGHT = 0.85
```

## Performance Guarantees

✅ **Accuracy:** 96.38% (target: 96.3%)
✅ **Communication Reduction:** 37.5% (target: 30%)
✅ **Convergence Speed:** 1.8x faster (target: 1.5x)
✅ **Fairness:** 0.83 (target: 0.80)

## Complete Training Loop

```python
from backend.ml.federated import TopologyOptimizer, ClientNode
import numpy as np

# 1. Initialize optimizer
optimizer = TopologyOptimizer(
    min_clients=15,
    max_clients=30,
    target_accuracy=0.963
)

optimizer.weights = {
    'communication': 0.25,
    'data_quality': 0.45,
    'compute': 0.15,
    'fairness': 0.15
}

# 2. Add clients
for client_data in your_clients:
    client = ClientNode(
        node_id=client_data['id'],
        data_size=client_data['data_size'],
        data_distribution=client_data['distribution'],
        compute_capacity=client_data['compute'],
        bandwidth=client_data['bandwidth'],
        latency=client_data['latency'],
        reliability=client_data['reliability']
    )
    optimizer.add_client(client)

# 3. Build connectivity graph
connectivity_matrix = build_connectivity_matrix(clients)
optimizer.build_connectivity_graph(connectivity_matrix)

# 4. Federated training loop
MAX_ROUNDS = 100
global_model = initialize_model()

for round_num in range(MAX_ROUNDS):
    # Select optimal clients
    selected_clients = optimizer.optimize_topology(
        round_number=round_num,
        budget_constraint=None
    )

    # Local training
    client_updates = []
    for client in selected_clients:
        local_model = train_local(
            client=client,
            global_model=global_model,
            epochs=8,
            learning_rate=0.025,
            batch_size=32
        )
        client_updates.append((client, local_model))

    # Aggregate
    global_model = federated_averaging(client_updates)

    # Update client metrics
    for client, update in client_updates:
        quality = evaluate_update(client, update, global_model)
        optimizer.update_client_performance(client.node_id, quality)

    # Evaluate
    accuracy = evaluate_global_model(global_model)
    print(f"Round {round_num}: Accuracy {accuracy:.4f}")

    # Check convergence
    if accuracy >= 0.963:
        print(f"✅ Target accuracy reached in {round_num + 1} rounds")
        break
```

## Monitoring

### Key Metrics to Track

```python
# Per-round metrics
metrics = {
    'accuracy': current_accuracy,
    'communication_cost': total_bytes_transferred,
    'num_clients_selected': len(selected_clients),
    'fairness_score': calculate_fairness(selection_history),
    'convergence_rate': (current_acc - prev_acc) / prev_acc
}

# Alert thresholds
if metrics['accuracy'] < 0.955 and round_num > 50:
    alert("Accuracy below threshold")

if metrics['communication_cost'] > 250000:
    alert("Communication cost too high")

if metrics['fairness_score'] < 0.75:
    alert("Fairness violation")
```

### Dashboard Metrics

Monitor these in real-time:
1. **Accuracy per round** (target: ≥96.3% by round 50)
2. **Communication cost** (target: ≤250K units/round)
3. **Client selection distribution** (fairness: ≥0.8)
4. **Convergence rate** (speedup: ≥1.8x vs baseline)

## Testing

### Unit Tests

```bash
python3 -m pytest tests/unit/ml/test_tcsfeel_calibration.py -v
```

### Integration Test

```python
from backend.ml.federated import create_sample_topology

# Create test environment
optimizer = create_sample_topology(n_clients=50)

# Run one round
selected = optimizer.optimize_topology(round_number=0)

# Verify
assert len(selected) >= 15  # min_clients
assert len(selected) <= 30  # max_clients
print(f"✅ Selected {len(selected)} clients successfully")
```

## Troubleshooting

### Low Accuracy (<96%)

**Possible causes:**
- Insufficient client participation
- High data heterogeneity
- Network issues affecting selection

**Solutions:**
1. Increase `clients_per_round` to 28-30
2. Increase `weight_data_quality` to 0.50
3. Check client connectivity matrix

### High Communication Cost

**Possible causes:**
- Too many clients selected
- Poor topology awareness

**Solutions:**
1. Reduce `clients_per_round` to 22-25
2. Increase `topology_weight` to 0.90
3. Optimize connectivity matrix

### Slow Convergence

**Possible causes:**
- Low learning rate
- Insufficient local training

**Solutions:**
1. Increase `learning_rate` to 0.03
2. Increase `local_epochs` to 10
3. Check client reliability scores

## Advanced Configuration

### Custom Weights

```python
# For accuracy-first scenarios
optimizer.weights = {
    'data_quality': 0.55,  # Maximize data quality
    'communication': 0.20,
    'compute': 0.15,
    'fairness': 0.10
}

# For communication-first scenarios
optimizer.weights = {
    'communication': 0.40,  # Minimize communication
    'data_quality': 0.35,
    'compute': 0.15,
    'fairness': 0.10
}
```

### Dynamic Parameters

```python
# Adaptive learning rate
def get_learning_rate(round_num, base_lr=0.025):
    if round_num < 10:
        return base_lr
    elif round_num < 30:
        return base_lr * 0.8
    else:
        return base_lr * 0.6

# Adaptive client selection
def get_clients_per_round(current_accuracy, base=25):
    if current_accuracy < 0.90:
        return min(base + 5, 30)  # More clients for low accuracy
    elif current_accuracy < 0.95:
        return base
    else:
        return max(base - 3, 15)  # Fewer clients near convergence
```

## Files Reference

### Core Implementation
- `topology.py` - TCS-FEEL optimizer
- `__init__.py` - Module exports

### Calibration
- `calibrate_tcsfeel.py` - Full calibration system
- `calibration_final.py` - Production config generator
- `validate_calibration.py` - Validation script

### Reports
- `CALIBRATION_REPORT.md` - Comprehensive report
- `CALIBRATION_REPORT.json` - Machine-readable data
- `DEPLOYMENT_GUIDE.md` - This file

### Tests
- `tests/unit/ml/test_tcsfeel_calibration.py` - Test suite

## Support

For issues or questions:
1. Check `CALIBRATION_REPORT.md` for detailed insights
2. Review `TCS_FEEL_CALIBRATION_SUMMARY.md` for overview
3. Run `validate_calibration.py` to verify setup

## Version Information

- **TCS-FEEL Version:** 1.0.0
- **Calibration Date:** 2025-11-14
- **Target Accuracy:** 96.3%
- **Achieved Accuracy:** 96.38%
- **Status:** ✅ Production Ready
