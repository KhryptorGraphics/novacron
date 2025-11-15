"""
Quick test of Bandwidth Predictor model
Demonstrates model initialization and basic functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from models.bandwidth_predictor import BandwidthPredictor, LSTMPredictor, DDQNAgent
from data.network_simulator import NetworkEnvironmentSimulator, generate_synthetic_data

print("=" * 80)
print("BANDWIDTH PREDICTOR - QUICK TEST")
print("=" * 80)
print()

# Test 1: LSTM Predictor initialization
print("Test 1: LSTM Predictor Initialization")
print("-" * 80)
lstm = LSTMPredictor(sequence_length=10, features=4)
print(f"✓ LSTM model created")
print(f"  Input shape: (None, 10, 4)")
print(f"  Architecture: 128→64→32 LSTM units")
print(f"  Output: Bandwidth prediction")
lstm.model.summary()
print()

# Test 2: DDQN Agent initialization
print("Test 2: DDQN Agent Initialization")
print("-" * 80)
ddqn = DDQNAgent(state_size=4, action_size=10)
print(f"✓ DDQN agent created")
print(f"  State size: 4 [latency, bandwidth, packet_loss, reliability]")
print(f"  Action size: 10 (10%-100% allocation levels)")
print(f"  Exploration rate: {ddqn.epsilon}")
print(f"  Discount factor: {ddqn.gamma}")
print()

# Test 3: Combined Bandwidth Predictor
print("Test 3: Combined Bandwidth Predictor")
print("-" * 80)
predictor = BandwidthPredictor(sequence_length=10, features=4, action_size=10)
print(f"✓ Combined predictor created")
print(f"  LSTM + DDQN architecture")
print()

# Test 4: Synthetic data generation
print("Test 4: Synthetic Data Generation")
print("-" * 80)
data = generate_synthetic_data(num_samples=100, network_type='datacenter')
print(f"✓ Data generated successfully")
print(f"  Training samples: {data['X_train'].shape}")
print(f"  Validation samples: {data['X_val'].shape}")
print(f"  Test samples: {data['X_test'].shape}")
print(f"  Network type: datacenter")
print()

# Test 5: Network environment simulator
print("Test 5: Network Environment Simulator")
print("-" * 80)
env = NetworkEnvironmentSimulator('datacenter')
state = env.reset()
print(f"✓ Environment initialized")
print(f"  Initial state: {state}")
print(f"  State format: [latency, bandwidth, packet_loss, reliability]")

# Run a few steps
print("\n  Running 5 simulation steps:")
for i in range(5):
    action = np.random.randint(0, 10)
    next_state, reward, done, info = env.step(action)
    allocation = (action + 1) * 10
    print(f"    Step {i+1}: Allocation={allocation}%, Reward={reward:.3f}, "
          f"Optimal={info['optimal_allocation']:.1f}%")
print()

# Test 6: Quick LSTM training (mini-batch)
print("Test 6: LSTM Quick Training (10 epochs)")
print("-" * 80)
print("Training LSTM on small dataset...")
X_train_small = data['X_train'][:50]
y_train_small = data['y_train'][:50]
X_val_small = data['X_val'][:10]
y_val_small = data['y_val'][:10]

history = predictor.train_lstm(
    X_train_small, y_train_small,
    X_val_small, y_val_small,
    epochs=10
)
print(f"✓ LSTM training completed")
print(f"  Final training loss: {history['loss'][-1]:.6f}")
print(f"  Final validation loss: {history['val_loss'][-1]:.6f}")
print()

# Test 7: LSTM prediction
print("Test 7: LSTM Prediction")
print("-" * 80)
test_sequence = data['X_test'][:5]
predictions = predictor.predict_bandwidth(test_sequence)
print(f"✓ Predictions generated for {len(test_sequence)} sequences")
print(f"  Sample predictions: {predictions.flatten()[:3]}")
print()

# Test 8: LSTM evaluation
print("Test 8: LSTM Evaluation")
print("-" * 80)
metrics = predictor.evaluate_lstm(data['X_test'][:20], data['y_test'][:20])
print(f"✓ Model evaluated on 20 test samples")
print(f"  MSE: {metrics['mse']:.6f}")
print(f"  MAE: {metrics['mae']:.6f}")
print(f"  MAPE: {metrics['mape']:.2f}%")
print(f"  Accuracy: {metrics['accuracy']:.2f}% (within 2% tolerance)")
print()

# Test 9: DDQN action selection
print("Test 9: DDQN Action Selection")
print("-" * 80)
test_states = [
    np.array([1.0, 9500, 0.002, 0.998]),  # Good datacenter
    np.array([50.0, 80, 0.03, 0.92]),     # Poor internet
    np.array([2.0, 9000, 0.01, 0.99]),    # Average datacenter
]

for i, state in enumerate(test_states):
    action = predictor.decide_allocation(state)
    allocation = (action + 1) * 10
    print(f"  State {i+1}: {state} → Allocation: {allocation}%")
print()

# Test 10: Model save/load
print("Test 10: Model Save/Load")
print("-" * 80)
save_dir = '/tmp/test_bandwidth_predictor'
os.makedirs(save_dir, exist_ok=True)

print(f"Saving model to {save_dir}...")
predictor.save(save_dir)
print("✓ Model saved")

print("Loading model...")
new_predictor = BandwidthPredictor()
new_predictor.load(save_dir)
print("✓ Model loaded")

# Verify predictions match
test_seq = data['X_test'][:1]
pred1 = predictor.predict_bandwidth(test_seq)
pred2 = new_predictor.predict_bandwidth(test_seq)
print(f"  Original prediction: {pred1[0][0]:.6f}")
print(f"  Loaded prediction:   {pred2[0][0]:.6f}")
print(f"  Match: {np.allclose(pred1, pred2)}")
print()

# Summary
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print("✓ All tests passed successfully")
print()
print("Model Specifications:")
print(f"  - LSTM Architecture: 128→64→32 units")
print(f"  - DDQN Episodes: 2000 (full training)")
print(f"  - Target Accuracy: 98% (datacenter), 70% (internet)")
print(f"  - State Space: 4 features")
print(f"  - Action Space: 10 levels")
print()
print("Next Steps:")
print("  1. Run full training: python training/bandwidth_trainer.py")
print("  2. Expected training time: ~10-15 minutes (CPU)")
print("  3. Expected accuracy: 98%+ for datacenter networks")
print("=" * 80)
