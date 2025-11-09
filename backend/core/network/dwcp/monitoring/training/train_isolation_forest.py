#!/usr/bin/env python3
"""
Train Isolation Forest model for anomaly detection in DWCP metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import json
from pathlib import Path


def load_training_data(filepath):
    """Load training data from CSV file."""
    df = pd.read_csv(filepath)

    # Expected columns: timestamp, bandwidth, latency, packet_loss, jitter,
    #                   cpu_usage, memory_usage, error_rate

    feature_columns = [
        'bandwidth', 'latency', 'packet_loss', 'jitter',
        'cpu_usage', 'memory_usage', 'error_rate'
    ]

    X = df[feature_columns].values

    return X, df


def train_isolation_forest(X, contamination=0.01, n_estimators=100):
    """
    Train Isolation Forest model.

    Args:
        X: Training data (normal data only)
        contamination: Expected proportion of anomalies (default: 0.01 = 1%)
        n_estimators: Number of isolation trees

    Returns:
        Trained model and scaler
    """
    print(f"Training Isolation Forest with {len(X)} samples...")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=n_estimators,
        max_samples='auto',
        contamination=contamination,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        verbose=1
    )

    model.fit(X_scaled)

    # Evaluate on training data
    predictions = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)

    n_anomalies = np.sum(predictions == -1)
    print(f"Detected {n_anomalies} anomalies in training data ({n_anomalies/len(X)*100:.2f}%)")
    print(f"Anomaly score range: [{scores.min():.4f}, {scores.max():.4f}]")

    return model, scaler


def export_to_onnx(model, scaler, output_path):
    """Export model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        print("Exporting to ONNX...")

        # Define input type
        initial_type = [('float_input', FloatTensorType([None, 7]))]

        # Convert model
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"ONNX model saved to {output_path}")

    except ImportError:
        print("Warning: skl2onnx not available, skipping ONNX export")


def save_model(model, scaler, output_dir):
    """Save model and scaler."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save scikit-learn model
    model_path = output_dir / "isolation_forest.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save scaler
    scaler_path = output_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Export to ONNX
    onnx_path = output_dir / "isolation_forest.onnx"
    export_to_onnx(model, scaler, onnx_path)

    # Save metadata
    metadata = {
        "model_type": "isolation_forest",
        "n_estimators": model.n_estimators,
        "contamination": model.contamination,
        "n_features": 7,
        "feature_names": [
            "bandwidth", "latency", "packet_loss", "jitter",
            "cpu_usage", "memory_usage", "error_rate"
        ]
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")


def generate_synthetic_data(n_samples=10000):
    """Generate synthetic training data for testing."""
    print(f"Generating {n_samples} synthetic samples...")

    np.random.seed(42)

    # Normal operating ranges
    bandwidth = np.random.normal(100, 15, n_samples)  # Mbps
    latency = np.random.normal(10, 2, n_samples)      # ms
    packet_loss = np.random.gamma(2, 0.005, n_samples) # %
    jitter = np.random.gamma(2, 0.5, n_samples)       # ms
    cpu_usage = np.random.beta(2, 5, n_samples) * 100 # %
    memory_usage = np.random.beta(3, 4, n_samples) * 100 # %
    error_rate = np.random.gamma(2, 0.0005, n_samples) # %

    # Clip to reasonable ranges
    bandwidth = np.clip(bandwidth, 0, 200)
    latency = np.clip(latency, 0, 50)
    packet_loss = np.clip(packet_loss, 0, 5)
    jitter = np.clip(jitter, 0, 10)
    cpu_usage = np.clip(cpu_usage, 0, 100)
    memory_usage = np.clip(memory_usage, 0, 100)
    error_rate = np.clip(error_rate, 0, 1)

    X = np.column_stack([
        bandwidth, latency, packet_loss, jitter,
        cpu_usage, memory_usage, error_rate
    ])

    return X


def main():
    parser = argparse.ArgumentParser(description='Train Isolation Forest for DWCP anomaly detection')
    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--output', type=str, default='../models', help='Output directory')
    parser.add_argument('--contamination', type=float, default=0.01, help='Expected contamination')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of trees')

    args = parser.parse_args()

    # Load or generate data
    if args.synthetic or not args.data:
        X = generate_synthetic_data()
    else:
        X, _ = load_training_data(args.data)

    # Train model
    model, scaler = train_isolation_forest(
        X,
        contamination=args.contamination,
        n_estimators=args.n_estimators
    )

    # Save model
    save_model(model, scaler, args.output)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
