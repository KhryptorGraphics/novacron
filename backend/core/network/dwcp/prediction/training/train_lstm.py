#!/usr/bin/env python3
"""
LSTM Bandwidth Predictor Training Script
Trains a neural network to predict network bandwidth, latency, packet loss, and jitter
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tf2onnx


class BandwidthLSTMTrainer:
    """Trainer for LSTM bandwidth prediction model"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None

    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = tf.keras.Sequential([
            # First LSTM layer with dropout
            tf.keras.layers.LSTM(
                128,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2,
                input_shape=input_shape,
                name='lstm_layer_1'
            ),

            # Second LSTM layer with dropout
            tf.keras.layers.LSTM(
                64,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_layer_2'
            ),

            # Dense layer with ReLU activation
            tf.keras.layers.Dense(32, activation='relu', name='dense_layer'),

            # Batch normalization
            tf.keras.layers.BatchNormalization(),

            # Output layer: 4 predictions
            # (bandwidth, latency, packet_loss, jitter)
            tf.keras.layers.Dense(4, name='output_layer')
        ])

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['learning_rate']
            ),
            loss='mse',
            metrics=['mae', 'mse']
        )

        return model

    def load_data(self, data_path):
        """Load and prepare training data"""
        print(f"Loading data from {data_path}")

        # Read CSV file
        df = pd.read_csv(data_path)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Sort by timestamp
        df = df.sort_values('timestamp')

        print(f"Loaded {len(df)} samples")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def prepare_sequences(self, df, window_size=10):
        """Prepare sequences for LSTM training"""
        print(f"Preparing sequences with window size {window_size}")

        # Features: bandwidth, latency, packet_loss, jitter, time_of_day, day_of_week
        feature_cols = [
            'bandwidth_mbps',
            'latency_ms',
            'packet_loss',
            'jitter_ms',
            'time_of_day',
            'day_of_week'
        ]

        # Targets: bandwidth, latency, packet_loss, jitter (future values)
        target_cols = [
            'bandwidth_mbps',
            'latency_ms',
            'packet_loss',
            'jitter_ms'
        ]

        X = []
        y = []

        # Create sequences
        for i in range(len(df) - window_size):
            # Input: window_size previous samples
            sequence = df.iloc[i:i+window_size][feature_cols].values
            X.append(sequence)

            # Output: next sample's values
            target = df.iloc[i+window_size][target_cols].values
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        return X, y

    def normalize_data(self, X, y, fit=True):
        """Normalize input and output data"""
        print("Normalizing data")

        # Reshape for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])

        if fit:
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            y_scaled = self.scaler_y.fit_transform(y)
        else:
            X_scaled = self.scaler_X.transform(X_reshaped)
            y_scaled = self.scaler_y.transform(y)

        # Reshape back
        X_scaled = X_scaled.reshape(X.shape)

        return X_scaled, y_scaled

    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model"""
        print("Training model")

        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))

        print(self.model.summary())

        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                self.config['checkpoint_path'],
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config['tensorboard_dir'],
                histogram_freq=1
            )
        ]

        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model")

        # Get predictions
        y_pred = self.model.predict(X_test)

        # Denormalize
        y_test_denorm = self.scaler_y.inverse_transform(y_test)
        y_pred_denorm = self.scaler_y.inverse_transform(y_pred)

        # Calculate metrics
        metrics = {}
        target_names = ['bandwidth_mbps', 'latency_ms', 'packet_loss', 'jitter_ms']

        for i, name in enumerate(target_names):
            actual = y_test_denorm[:, i]
            predicted = y_pred_denorm[:, i]

            mae = np.mean(np.abs(actual - predicted))
            mape = np.mean(np.abs((actual - predicted) / actual)) * 100
            rmse = np.sqrt(np.mean((actual - predicted) ** 2))

            metrics[name] = {
                'mae': float(mae),
                'mape': float(mape),
                'rmse': float(rmse)
            }

            print(f"{name}:")
            print(f"  MAE: {mae:.4f}")
            print(f"  MAPE: {mape:.2f}%")
            print(f"  RMSE: {rmse:.4f}")

        return metrics

    def export_to_onnx(self, output_path):
        """Export model to ONNX format"""
        print(f"Exporting model to ONNX: {output_path}")

        # Convert to ONNX
        spec = (tf.TensorSpec((None, 10, 6), tf.float32, name="input"),)
        output_path_str = str(output_path)

        model_proto, _ = tf2onnx.convert.from_keras(
            self.model,
            input_signature=spec,
            opset=13
        )

        # Save to file
        with open(output_path_str, "wb") as f:
            f.write(model_proto.SerializeToString())

        print(f"Model exported to {output_path}")

    def save_metadata(self, output_path, metrics):
        """Save model metadata"""
        metadata = {
            'version': self.config['model_version'],
            'created_at': datetime.now().isoformat(),
            'training_config': self.config,
            'metrics': metrics,
            'model_architecture': {
                'input_shape': [10, 6],
                'output_shape': [4],
                'layers': [
                    {'type': 'LSTM', 'units': 128, 'dropout': 0.2},
                    {'type': 'LSTM', 'units': 64, 'dropout': 0.2},
                    {'type': 'Dense', 'units': 32, 'activation': 'relu'},
                    {'type': 'Dense', 'units': 4}
                ]
            },
            'features': [
                'bandwidth_mbps',
                'latency_ms',
                'packet_loss',
                'jitter_ms',
                'time_of_day',
                'day_of_week'
            ],
            'targets': [
                'bandwidth_mbps',
                'latency_ms',
                'packet_loss',
                'jitter_ms'
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train LSTM bandwidth predictor')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--output', default='./models', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--window-size', type=int, default=10, help='Sequence window size')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = {
        'model_version': 'v' + datetime.now().strftime('%Y%m%d_%H%M%S'),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'window_size': args.window_size,
        'validation_split': args.validation_split,
        'early_stopping_patience': 5,
        'checkpoint_path': str(output_dir / 'best_model.h5'),
        'tensorboard_dir': str(output_dir / 'tensorboard')
    }

    print("=" * 80)
    print("LSTM Bandwidth Predictor Training")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print("=" * 80)

    # Initialize trainer
    trainer = BandwidthLSTMTrainer(config)

    # Load data
    df = trainer.load_data(args.data)

    # Prepare sequences
    X, y = trainer.prepare_sequences(df, window_size=args.window_size)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.validation_split,
        shuffle=False  # Keep temporal order
    )

    # Further split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        shuffle=False
    )

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")

    # Normalize data
    X_train, y_train = trainer.normalize_data(X_train, y_train, fit=True)
    X_val, y_val = trainer.normalize_data(X_val, y_val, fit=False)
    X_test, y_test = trainer.normalize_data(X_test, y_test, fit=False)

    # Train model
    history = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate model
    metrics = trainer.evaluate(X_test, y_test)

    # Export to ONNX
    onnx_path = output_dir / f'bandwidth_lstm_{config["model_version"]}.onnx'
    trainer.export_to_onnx(onnx_path)

    # Save metadata
    metadata_path = output_dir / f'model_metadata_{config["model_version"]}.json'
    trainer.save_metadata(metadata_path, metrics)

    # Save training history
    history_path = output_dir / f'training_history_{config["model_version"]}.json'
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']],
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print("=" * 80)
    print(f"ONNX model: {onnx_path}")
    print(f"Metadata: {metadata_path}")
    print(f"History: {history_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
