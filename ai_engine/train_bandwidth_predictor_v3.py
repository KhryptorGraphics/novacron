#!/usr/bin/env python3
"""
Training script for DWCP v3 Bandwidth Predictor
Trains both datacenter and internet models with proper validation.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from bandwidth_predictor_v3 import (
    BandwidthPredictorV3,
    NetworkMetrics,
    generate_synthetic_data
)


def train_and_evaluate(mode: str, data_size: int = 2000, epochs: int = 50):
    """Train and evaluate a model for specific mode"""
    print(f"\n{'='*70}")
    print(f"Training {mode.upper()} Mode Predictor")
    print(f"{'='*70}")

    # Generate training data
    print(f"\n1. Generating {data_size} synthetic {mode} samples...")
    training_data = generate_synthetic_data(mode, data_size)
    print(f"   Generated {len(training_data)} samples")

    # Split into train/validation/test
    train_size = int(0.7 * len(training_data))
    val_size = int(0.15 * len(training_data))

    train_data = training_data[:train_size]
    val_data = training_data[train_size:train_size + val_size]
    test_data = training_data[train_size + val_size:]

    print(f"   Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Create predictor
    print(f"\n2. Creating {mode} predictor...")
    predictor = BandwidthPredictorV3(mode=mode)

    # Train model
    print(f"\n3. Training model...")
    training_results = predictor.train(
        train_data + val_data,  # Use both for training with internal validation
        validation_split=0.2,
        epochs=epochs,
        batch_size=32
    )

    print(f"\n   Training Results:")
    print(f"   - Final Val Loss: {training_results['final_val_loss']:.4f}")
    print(f"   - Final Val MAE: {training_results['final_val_mae']:.2f} Mbps")
    print(f"   - Epochs Trained: {training_results['epochs_trained']}")

    # Evaluate on test set
    print(f"\n4. Evaluating on test set...")
    test_predictions = []
    test_actuals = []
    test_confidences = []

    sequence_length = predictor.config.sequence_length
    for i in range(sequence_length, len(test_data)):
        history = test_data[i - sequence_length:i]
        actual = test_data[i].bandwidth_mbps

        try:
            prediction, confidence = predictor.predict(history, return_confidence=True)
            test_predictions.append(prediction)
            test_actuals.append(actual)
            test_confidences.append(confidence)
        except Exception as e:
            print(f"   Warning: Prediction failed at step {i}: {e}")

    # Calculate test metrics
    if len(test_predictions) > 0:
        predictions_array = np.array(test_predictions)
        actuals_array = np.array(test_actuals)
        confidences_array = np.array(test_confidences)

        mae = np.mean(np.abs(predictions_array - actuals_array))
        mse = np.mean((predictions_array - actuals_array) ** 2)
        rmse = np.sqrt(mse)

        # Calculate accuracy (percentage within ±20%)
        errors = np.abs(predictions_array - actuals_array) / actuals_array
        accuracy_20 = np.mean(errors < 0.20) * 100  # Within 20%
        accuracy_10 = np.mean(errors < 0.10) * 100  # Within 10%

        avg_confidence = np.mean(confidences_array)

        print(f"\n   Test Set Metrics:")
        print(f"   - MAE: {mae:.2f} Mbps")
        print(f"   - RMSE: {rmse:.2f} Mbps")
        print(f"   - Accuracy (±20%): {accuracy_20:.1f}%")
        print(f"   - Accuracy (±10%): {accuracy_10:.1f}%")
        print(f"   - Average Confidence: {avg_confidence:.2%}")

        # Check if target met
        if mode == 'datacenter':
            target = 85.0
            status = "✓" if accuracy_20 >= target else "✗"
            print(f"\n   Target (85%+ accuracy): {status} ({accuracy_20:.1f}%)")
        else:  # internet
            target = 70.0
            status = "✓" if accuracy_20 >= target else "✗"
            print(f"\n   Target (70%+ accuracy): {status} ({accuracy_20:.1f}%)")

    # Save model
    model_dir = f"models/{mode}"
    print(f"\n5. Saving model to {model_dir}...")
    predictor.save_model(model_dir)
    print(f"   Model saved successfully")

    # Try to export to ONNX
    try:
        print(f"\n6. Exporting to ONNX format...")
        predictor.export_to_onnx(model_dir)
        print(f"   ONNX export successful")
    except Exception as e:
        print(f"   ONNX export failed: {e}")
        print(f"   Install tf2onnx: pip install tf2onnx")

    return {
        'mode': mode,
        'training_results': training_results,
        'test_mae': mae if len(test_predictions) > 0 else None,
        'test_rmse': rmse if len(test_predictions) > 0 else None,
        'test_accuracy_20': accuracy_20 if len(test_predictions) > 0 else None,
        'test_accuracy_10': accuracy_10 if len(test_predictions) > 0 else None,
        'avg_confidence': avg_confidence if len(test_predictions) > 0 else None,
        'predictor': predictor
    }


def plot_training_history(results_dc, results_inet):
    """Plot training history for both models"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DWCP v3 Bandwidth Predictor Training Results', fontsize=16)

    # Datacenter loss
    history_dc = results_dc['training_results']['history']
    axes[0, 0].plot(history_dc['loss'], label='Train Loss')
    axes[0, 0].plot(history_dc['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Datacenter Mode - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Datacenter MAE
    axes[0, 1].plot(history_dc['mae'], label='Train MAE')
    axes[0, 1].plot(history_dc['val_mae'], label='Val MAE')
    axes[0, 1].set_title('Datacenter Mode - MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (Mbps)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Internet loss
    history_inet = results_inet['training_results']['history']
    axes[1, 0].plot(history_inet['loss'], label='Train Loss')
    axes[1, 0].plot(history_inet['val_loss'], label='Val Loss')
    axes[1, 0].set_title('Internet Mode - Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss (MSE)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Internet MAE
    axes[1, 1].plot(history_inet['mae'], label='Train MAE')
    axes[1, 1].plot(history_inet['val_mae'], label='Val MAE')
    axes[1, 1].set_title('Internet Mode - MAE')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MAE (Mbps)')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print("\nTraining history plot saved to: models/training_history.png")


def main():
    parser = argparse.ArgumentParser(
        description='Train DWCP v3 Bandwidth Predictor'
    )
    parser.add_argument(
        '--mode',
        choices=['datacenter', 'internet', 'both'],
        default='both',
        help='Which mode to train (default: both)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=2000,
        help='Number of training samples (default: 2000)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate training history plots'
    )

    args = parser.parse_args()

    print("\n" + "="*70)
    print("DWCP v3 Bandwidth Predictor Training")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Mode: {args.mode}")
    print(f"  - Samples: {args.samples}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Plot: {args.plot}")

    results = {}

    # Train datacenter model
    if args.mode in ['datacenter', 'both']:
        results['datacenter'] = train_and_evaluate('datacenter', args.samples, args.epochs)

    # Train internet model
    if args.mode in ['internet', 'both']:
        results['internet'] = train_and_evaluate('internet', args.samples, args.epochs)

    # Summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")

    for mode, result in results.items():
        print(f"\n{mode.upper()} Mode:")
        if result['test_mae'] is not None:
            print(f"  - Test MAE: {result['test_mae']:.2f} Mbps")
            print(f"  - Test RMSE: {result['test_rmse']:.2f} Mbps")
            print(f"  - Accuracy (±20%): {result['test_accuracy_20']:.1f}%")
            print(f"  - Confidence: {result['avg_confidence']:.2%}")

    # Plot if requested
    if args.plot and 'datacenter' in results and 'internet' in results:
        try:
            plot_training_history(results['datacenter'], results['internet'])
        except Exception as e:
            print(f"\nWarning: Failed to generate plots: {e}")

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Deploy ONNX models to: /var/lib/dwcp/models/")
    print("2. Run integration tests: go test -v ./v3/prediction/")
    print("3. Enable v3 prediction: update feature flags")


if __name__ == "__main__":
    main()
