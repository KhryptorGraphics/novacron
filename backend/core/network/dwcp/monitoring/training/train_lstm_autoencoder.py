#!/usr/bin/env python3
"""
Train LSTM Autoencoder for time-series anomaly detection in DWCP metrics.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import argparse
import json
from pathlib import Path


def load_training_data(filepath, window_size=10):
    """Load and prepare time-series training data."""
    df = pd.read_csv(filepath)

    feature_columns = [
        'bandwidth', 'latency', 'packet_loss', 'jitter',
        'cpu_usage', 'memory_usage', 'error_rate'
    ]

    X = df[feature_columns].values

    # Create sliding windows
    X_windows = []
    for i in range(len(X) - window_size + 1):
        X_windows.append(X[i:i+window_size])

    X_windows = np.array(X_windows)

    return X_windows


def build_lstm_autoencoder(timesteps=10, n_features=7, encoding_dim=32):
    """
    Build LSTM Autoencoder model.

    Architecture:
        Encoder: Input -> LSTM(64) -> LSTM(32) -> Encoding
        Decoder: Encoding -> RepeatVector -> LSTM(32) -> LSTM(64) -> Output
    """
    print(f"Building LSTM Autoencoder (timesteps={timesteps}, features={n_features})...")

    # Encoder
    encoder_inputs = keras.Input(shape=(timesteps, n_features), name='encoder_input')
    encoded = keras.layers.LSTM(64, return_sequences=True, name='encoder_lstm_1')(encoder_inputs)
    encoded = keras.layers.Dropout(0.2)(encoded)
    encoded = keras.layers.LSTM(encoding_dim, return_sequences=False, name='encoder_lstm_2')(encoded)

    # Decoder
    decoded = keras.layers.RepeatVector(timesteps, name='decoder_repeat')(encoded)
    decoded = keras.layers.LSTM(encoding_dim, return_sequences=True, name='decoder_lstm_1')(decoded)
    decoded = keras.layers.Dropout(0.2)(decoded)
    decoded = keras.layers.LSTM(64, return_sequences=True, name='decoder_lstm_2')(decoded)
    decoder_outputs = keras.layers.TimeDistributed(
        keras.layers.Dense(n_features),
        name='decoder_output'
    )(decoded)

    # Autoencoder model
    autoencoder = keras.Model(encoder_inputs, decoder_outputs, name='lstm_autoencoder')

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )

    print(autoencoder.summary())

    return autoencoder


def train_autoencoder(model, X_train, epochs=50, batch_size=32):
    """Train the LSTM Autoencoder."""
    print(f"Training LSTM Autoencoder with {len(X_train)} samples...")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    return history


def calculate_threshold(model, X_train, percentile=95):
    """Calculate reconstruction error threshold."""
    print("Calculating reconstruction error threshold...")

    # Get reconstructions
    reconstructions = model.predict(X_train, verbose=0)

    # Calculate MSE for each sample
    mse = np.mean(np.power(X_train - reconstructions, 2), axis=(1, 2))

    # Set threshold at percentile
    threshold = np.percentile(mse, percentile)

    print(f"Reconstruction error statistics:")
    print(f"  Mean: {mse.mean():.6f}")
    print(f"  Std:  {mse.std():.6f}")
    print(f"  Min:  {mse.min():.6f}")
    print(f"  Max:  {mse.max():.6f}")
    print(f"  {percentile}th percentile (threshold): {threshold:.6f}")

    return threshold


def export_to_onnx(model, output_path):
    """Export model to ONNX format."""
    try:
        import tf2onnx

        print("Exporting to ONNX...")

        # Convert to ONNX
        onnx_model, _ = tf2onnx.convert.from_keras(model)

        # Save ONNX model
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        print(f"ONNX model saved to {output_path}")

    except ImportError:
        print("Warning: tf2onnx not available, skipping ONNX export")


def save_model(model, scaler, threshold, output_dir, timesteps, n_features):
    """Save model, scaler, and threshold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save Keras model
    model_path = output_dir / "lstm_autoencoder.h5"
    model.save(model_path)
    print(f"Keras model saved to {model_path}")

    # Save scaler
    import joblib
    scaler_path = output_dir / "lstm_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Export to ONNX
    onnx_path = output_dir / "lstm_autoencoder.onnx"
    export_to_onnx(model, onnx_path)

    # Save metadata and threshold
    metadata = {
        "model_type": "lstm_autoencoder",
        "timesteps": timesteps,
        "n_features": n_features,
        "threshold": float(threshold),
        "feature_names": [
            "bandwidth", "latency", "packet_loss", "jitter",
            "cpu_usage", "memory_usage", "error_rate"
        ]
    }

    metadata_path = output_dir / "lstm_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")


def generate_synthetic_data(n_samples=10000, window_size=10):
    """Generate synthetic time-series data."""
    print(f"Generating {n_samples} synthetic time-series samples...")

    np.random.seed(42)

    # Generate time-series with trends and seasonality
    t = np.arange(n_samples + window_size)

    # Add daily seasonality (assuming hourly data)
    seasonal = 10 * np.sin(2 * np.pi * t / 24)

    # Normal operating ranges with trend and seasonality
    bandwidth = 100 + seasonal + np.random.normal(0, 5, len(t))
    latency = 10 - seasonal * 0.5 + np.random.normal(0, 1, len(t))
    packet_loss = np.abs(0.01 + np.random.normal(0, 0.005, len(t)))
    jitter = np.abs(1.0 + seasonal * 0.1 + np.random.normal(0, 0.2, len(t)))
    cpu_usage = 50 + seasonal * 2 + np.random.normal(0, 5, len(t))
    memory_usage = 60 + np.random.normal(0, 5, len(t))
    error_rate = np.abs(0.001 + np.random.normal(0, 0.0002, len(t)))

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

    # Create sliding windows
    X_windows = []
    for i in range(n_samples):
        X_windows.append(X[i:i+window_size])

    X_windows = np.array(X_windows)

    return X_windows


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Autoencoder for DWCP anomaly detection')
    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
    parser.add_argument('--output', type=str, default='../models', help='Output directory')
    parser.add_argument('--window-size', type=int, default=10, help='Window size for time series')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--encoding-dim', type=int, default=32, help='Encoding dimension')

    args = parser.parse_args()

    # Load or generate data
    if args.synthetic or not args.data:
        X = generate_synthetic_data(window_size=args.window_size)
    else:
        X = load_training_data(args.data, window_size=args.window_size)

    # Standardize features
    scaler = StandardScaler()
    n_samples, timesteps, n_features = X.shape
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, timesteps, n_features)

    # Build model
    model = build_lstm_autoencoder(
        timesteps=args.window_size,
        n_features=7,
        encoding_dim=args.encoding_dim
    )

    # Train model
    history = train_autoencoder(
        model, X_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    # Calculate threshold
    threshold = calculate_threshold(model, X_scaled)

    # Save model
    save_model(model, scaler, threshold, args.output, args.window_size, 7)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
