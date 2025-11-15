#!/usr/bin/env python3
"""
Optimized LSTM Bandwidth Predictor Training Script - Target: ≥98% Accuracy
Trains a neural network to predict network bandwidth, latency, packet loss, and jitter

Target Metrics:
- Correlation coefficient ≥ 0.98 for bandwidth prediction
- MAPE < 5% for bandwidth
- Overall regression accuracy ≥ 98%
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tf2onnx

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class OptimizedBandwidthLSTMTrainer:
    """Optimized trainer for LSTM bandwidth prediction model targeting ≥98% accuracy"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.history = None
        self.feature_cols = None
        self.target_cols = None

    def build_optimized_model(self, input_shape):
        """Build optimized LSTM model architecture for ≥98% accuracy"""
        print("\nBuilding optimized LSTM architecture...")

        model = tf.keras.Sequential([
            # First LSTM layer - increased units for better capacity
            tf.keras.layers.LSTM(
                256,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.2,
                input_shape=input_shape,
                name='lstm_layer_1'
            ),

            # Batch normalization for stability
            tf.keras.layers.BatchNormalization(),

            # Second LSTM layer with return sequences
            tf.keras.layers.LSTM(
                128,
                return_sequences=True,
                dropout=0.3,
                recurrent_dropout=0.2,
                name='lstm_layer_2'
            ),

            # Batch normalization
            tf.keras.layers.BatchNormalization(),

            # Third LSTM layer without return sequences
            tf.keras.layers.LSTM(
                64,
                dropout=0.2,
                recurrent_dropout=0.2,
                name='lstm_layer_3'
            ),

            # Dense layers with progressive reduction
            tf.keras.layers.Dense(128, activation='relu', name='dense_layer_1'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Dense(64, activation='relu', name='dense_layer_2'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(32, activation='relu', name='dense_layer_3'),

            # Output layer: 4 predictions (bandwidth, latency, packet_loss, jitter)
            tf.keras.layers.Dense(4, name='output_layer')
        ])

        # Use Adam optimizer with custom learning rate schedule
        initial_lr = self.config['learning_rate']
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # Compile with multiple metrics
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[
                'mae',
                'mse',
                tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            ]
        )

        return model

    def load_data(self, data_path):
        """Load and prepare training data"""
        print(f"\nLoading data from {data_path}")

        df = pd.read_csv(data_path)

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')

        print(f"Loaded {len(df)} samples")
        if 'timestamp' in df.columns:
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def prepare_features(self, df):
        """Prepare feature columns"""
        # Primary features for prediction
        feature_cols = [
            'throughput_mbps',  # or 'bandwidth_mbps'
            'rtt_ms',          # or 'latency_ms'
            'packet_loss',
            'jitter_ms',
        ]

        # Add temporal features if available
        if 'time_of_day' in df.columns:
            feature_cols.append('time_of_day')
        if 'day_of_week' in df.columns:
            feature_cols.append('day_of_week')

        # Add network context features
        if 'congestion_window' in df.columns:
            feature_cols.append('congestion_window')
        if 'queue_depth' in df.columns:
            feature_cols.append('queue_depth')
        if 'retransmits' in df.columns:
            feature_cols.append('retransmits')

        # Normalize column names (handle both throughput_mbps and bandwidth_mbps)
        if 'bandwidth_mbps' in df.columns and 'throughput_mbps' not in df.columns:
            df['throughput_mbps'] = df['bandwidth_mbps']
        if 'latency_ms' in df.columns and 'rtt_ms' not in df.columns:
            df['rtt_ms'] = df['latency_ms']

        # Filter to only available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        print(f"\nUsing {len(feature_cols)} features: {feature_cols}")

        return feature_cols

    def prepare_sequences(self, df, window_size=20):
        """Prepare sequences for LSTM training with sliding window"""
        print(f"\nPreparing sequences with window size {window_size}")

        # Determine feature columns
        self.feature_cols = self.prepare_features(df)

        # Target columns (what we're predicting)
        self.target_cols = [
            'throughput_mbps',
            'rtt_ms',
            'packet_loss',
            'jitter_ms'
        ]

        X = []
        y = []

        # Create sequences with sliding window
        for i in range(len(df) - window_size):
            # Input: window_size previous samples
            sequence = df.iloc[i:i+window_size][self.feature_cols].values
            X.append(sequence)

            # Output: next sample's values
            target = df.iloc[i+window_size][self.target_cols].values
            y.append(target)

        X = np.array(X)
        y = np.array(y)

        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape} (samples, timesteps, features)")
        print(f"y shape: {y.shape} (samples, targets)")

        return X, y

    def normalize_data(self, X, y, fit=True):
        """Normalize input and output data"""
        print("\nNormalizing data...")

        # Reshape for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])

        if fit:
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            y_scaled = self.scaler_y.fit_transform(y)
            print(f"Fitted scalers - X mean: {self.scaler_X.mean_[:3]}, y mean: {self.scaler_y.mean_}")
        else:
            X_scaled = self.scaler_X.transform(X_reshaped)
            y_scaled = self.scaler_y.transform(y)

        # Reshape back
        X_scaled = X_scaled.reshape(X.shape)

        return X_scaled, y_scaled

    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model"""
        print("\n" + "="*80)
        print("TRAINING LSTM MODEL")
        print("="*80)

        # Build model
        self.model = self.build_optimized_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )

        print("\nModel Architecture:")
        self.model.summary()

        # Calculate total parameters
        total_params = self.model.count_params()
        print(f"\nTotal parameters: {total_params:,}")

        # Callbacks
        checkpoint_dir = Path(self.config['checkpoint_path']).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

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
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.config['tensorboard_dir'],
                histogram_freq=1,
                write_graph=True
            ),
            tf.keras.callbacks.CSVLogger(
                str(checkpoint_dir / 'training_log.csv')
            )
        ]

        # Train model
        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Initial learning rate: {self.config['learning_rate']}")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1
        )

        print("\nTraining completed!")
        return self.history

    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation of model performance"""
        print("\n" + "="*80)
        print("EVALUATING MODEL")
        print("="*80)

        # Get predictions
        y_pred_scaled = self.model.predict(X_test, verbose=0)

        # Denormalize
        y_test_denorm = self.scaler_y.inverse_transform(y_test)
        y_pred_denorm = self.scaler_y.inverse_transform(y_pred_scaled)

        # Calculate metrics for each target
        metrics = {}
        target_names = self.target_cols

        print("\nPer-Target Metrics:")
        print("-" * 80)

        for i, name in enumerate(target_names):
            actual = y_test_denorm[:, i]
            predicted = y_pred_denorm[:, i]

            # Regression metrics
            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)

            # MAPE (handle division by zero)
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100

            # Correlation coefficient
            correlation = np.corrcoef(actual, predicted)[0, 1]

            metrics[name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2_score': float(r2),
                'correlation': float(correlation)
            }

            print(f"\n{name}:")
            print(f"  MAE:         {mae:.4f}")
            print(f"  RMSE:        {rmse:.4f}")
            print(f"  MAPE:        {mape:.2f}%")
            print(f"  R² Score:    {r2:.4f}")
            print(f"  Correlation: {correlation:.4f}")

        # Overall accuracy (average correlation across all targets)
        avg_correlation = np.mean([m['correlation'] for m in metrics.values()])
        avg_mape = np.mean([m['mape'] for m in metrics.values()])

        print("\n" + "="*80)
        print("OVERALL PERFORMANCE")
        print("="*80)
        print(f"Average Correlation: {avg_correlation:.4f}")
        print(f"Average MAPE:        {avg_mape:.2f}%")

        # Check if we met the ≥98% target
        target_met = avg_correlation >= 0.98 and avg_mape <= 5.0

        print("\n" + "="*80)
        if target_met:
            print("✅ TARGET MET: ≥98% Accuracy Achieved!")
        else:
            print("❌ TARGET NOT MET")
        print("="*80)

        # Detailed accuracy metrics
        accuracy_metrics = {
            'overall': {
                'average_correlation': float(avg_correlation),
                'average_mape': float(avg_mape),
                'target_met': target_met,
                'accuracy_percent': float(avg_correlation * 100)
            },
            'per_target': metrics
        }

        return accuracy_metrics, y_test_denorm, y_pred_denorm

    def plot_training_history(self, output_dir):
        """Plot training history"""
        if self.history is None:
            return

        print("\nGenerating training plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)

        # Loss
        axes[0, 0].plot(self.history.history['loss'], label='Train Loss')
        axes[0, 0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # MAE
        axes[0, 1].plot(self.history.history['mae'], label='Train MAE')
        axes[0, 1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # RMSE
        axes[1, 0].plot(self.history.history['rmse'], label='Train RMSE')
        axes[1, 0].plot(self.history.history['val_rmse'], label='Val RMSE')
        axes[1, 0].set_title('Root Mean Squared Error')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate (if available)
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr_history = [self.config['learning_rate'] * (0.96 ** (i // 1000)) for i in range(len(self.history.history['loss']))]
            axes[1, 1].plot(lr_history)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)

        plt.tight_layout()
        plot_path = output_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved training history plot to {plot_path}")

    def plot_predictions(self, y_test, y_pred, output_dir):
        """Plot actual vs predicted values"""
        print("\nGenerating prediction plots...")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Actual vs Predicted Values', fontsize=16)

        for i, name in enumerate(self.target_cols):
            row = i // 2
            col = i % 2

            actual = y_test[:, i]
            predicted = y_pred[:, i]

            # Scatter plot
            axes[row, col].scatter(actual, predicted, alpha=0.5, s=10)

            # Perfect prediction line
            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            axes[row, col].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

            # Calculate R²
            r2 = r2_score(actual, predicted)
            correlation = np.corrcoef(actual, predicted)[0, 1]

            axes[row, col].set_title(f'{name}\nR²={r2:.4f}, Corr={correlation:.4f}')
            axes[row, col].set_xlabel('Actual')
            axes[row, col].set_ylabel('Predicted')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'predictions_scatter.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved prediction scatter plot to {plot_path}")

        # Time series plot for first 500 samples
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('Time Series: Actual vs Predicted (First 500 Samples)', fontsize=16)

        samples_to_plot = min(500, len(y_test))

        for i, name in enumerate(self.target_cols):
            axes[i].plot(y_test[:samples_to_plot, i], label='Actual', linewidth=2, alpha=0.7)
            axes[i].plot(y_pred[:samples_to_plot, i], label='Predicted', linewidth=2, alpha=0.7)
            axes[i].set_title(name)
            axes[i].set_xlabel('Sample')
            axes[i].set_ylabel('Value')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = output_dir / 'predictions_timeseries.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved time series plot to {plot_path}")

    def export_to_onnx(self, output_path):
        """Export model to ONNX format for Go inference"""
        print(f"\nExporting model to ONNX: {output_path}")

        # Get input shape from model
        input_shape = self.model.input_shape
        timesteps = input_shape[1]
        features = input_shape[2]

        # Create TensorSpec
        spec = (tf.TensorSpec((None, timesteps, features), tf.float32, name="input"),)

        # Convert to ONNX
        model_proto, _ = tf2onnx.convert.from_keras(
            self.model,
            input_signature=spec,
            opset=13
        )

        # Save to file
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())

        # Get file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model exported successfully!")
        print(f"File size: {file_size_mb:.2f} MB")

        return file_size_mb

    def save_metadata(self, output_path, metrics, model_size_mb, training_time):
        """Save comprehensive model metadata"""
        metadata = {
            'version': self.config['model_version'],
            'created_at': datetime.now().isoformat(),
            'training_duration_seconds': training_time,
            'model_size_mb': model_size_mb,
            'target_achieved': metrics['overall']['target_met'],
            'training_config': self.config,
            'performance_metrics': metrics,
            'model_architecture': {
                'type': 'LSTM',
                'input_shape': [self.config['window_size'], len(self.feature_cols)],
                'output_shape': [len(self.target_cols)],
                'total_parameters': int(self.model.count_params()),
                'layers': [
                    {'type': 'LSTM', 'units': 256, 'dropout': 0.3},
                    {'type': 'BatchNormalization'},
                    {'type': 'LSTM', 'units': 128, 'dropout': 0.3},
                    {'type': 'BatchNormalization'},
                    {'type': 'LSTM', 'units': 64, 'dropout': 0.2},
                    {'type': 'Dense', 'units': 128, 'activation': 'relu'},
                    {'type': 'Dropout', 'rate': 0.3},
                    {'type': 'Dense', 'units': 64, 'activation': 'relu'},
                    {'type': 'Dropout', 'rate': 0.2},
                    {'type': 'Dense', 'units': 32, 'activation': 'relu'},
                    {'type': 'Dense', 'units': 4, 'activation': 'linear'}
                ]
            },
            'features': self.feature_cols,
            'targets': self.target_cols,
            'scaler_params': {
                'X_mean': self.scaler_X.mean_.tolist(),
                'X_scale': self.scaler_X.scale_.tolist(),
                'y_mean': self.scaler_y.mean_.tolist(),
                'y_scale': self.scaler_y.scale_.tolist()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train optimized LSTM bandwidth predictor (Target: ≥98% accuracy)'
    )
    parser.add_argument('--data-path', required=True, help='Path to training data CSV')
    parser.add_argument('--output-dir', default='./checkpoints/bandwidth_predictor',
                       help='Output directory for model and artifacts')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--window-size', type=int, default=20,
                       help='Sequence window size (timesteps)')
    parser.add_argument('--validation-split', type=float, default=0.15,
                       help='Validation set size')
    parser.add_argument('--test-split', type=float, default=0.15, help='Test set size')
    parser.add_argument('--target-correlation', type=float, default=0.98,
                       help='Target correlation coefficient')
    parser.add_argument('--target-mape', type=float, default=5.0, help='Target MAPE (%)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    model_version = 'v' + datetime.now().strftime('%Y%m%d_%H%M%S')
    config = {
        'model_version': model_version,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'window_size': args.window_size,
        'validation_split': args.validation_split,
        'test_split': args.test_split,
        'early_stopping_patience': 15,
        'target_correlation': args.target_correlation,
        'target_mape': args.target_mape,
        'checkpoint_path': str(output_dir / 'best_model.keras'),
        'tensorboard_dir': str(output_dir / 'tensorboard'),
        'seed': args.seed
    }

    print("=" * 80)
    print("OPTIMIZED LSTM BANDWIDTH PREDICTOR TRAINING")
    print("Target: ≥98% Accuracy (Correlation ≥ 0.98, MAPE < 5%)")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print("=" * 80)

    import time
    start_time = time.time()

    # Initialize trainer
    trainer = OptimizedBandwidthLSTMTrainer(config)

    # Load data
    df = trainer.load_data(args.data_path)

    # Prepare sequences
    X, y = trainer.prepare_sequences(df, window_size=args.window_size)

    # Temporal split (no shuffling to preserve time order)
    # Split: 70% train, 15% val, 15% test
    test_size = int(len(X) * args.test_split)
    val_size = int(len(X) * args.validation_split)
    train_size = len(X) - test_size - val_size

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]

    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"\nTemporal data split (preserving time order):")
    print(f"  Train:      {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Normalize data
    X_train, y_train = trainer.normalize_data(X_train, y_train, fit=True)
    X_val, y_val = trainer.normalize_data(X_val, y_val, fit=False)
    X_test, y_test = trainer.normalize_data(X_test, y_test, fit=False)

    # Train model
    history = trainer.train(X_train, y_train, X_val, y_val)

    # Evaluate model
    metrics, y_test_denorm, y_pred_denorm = trainer.evaluate(X_test, y_test)

    training_time = time.time() - start_time

    # Generate plots
    trainer.plot_training_history(output_dir)
    trainer.plot_predictions(y_test_denorm, y_pred_denorm, output_dir)

    # Export to ONNX
    onnx_path = output_dir / f'bandwidth_lstm_{model_version}.onnx'
    model_size_mb = trainer.export_to_onnx(onnx_path)

    # Save metadata
    metadata_path = output_dir / f'model_metadata_{model_version}.json'
    trainer.save_metadata(metadata_path, metrics, model_size_mb, training_time)

    # Save training history
    history_path = output_dir / f'training_history_{model_version}.json'
    with open(history_path, 'w') as f:
        json.dump({
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']],
        }, f, indent=2)

    # Save evaluation report
    report_path = output_dir / 'bandwidth_predictor_report.json'
    with open(report_path, 'w') as f:
        json.dump({
            'model': 'bandwidth_predictor',
            'version': model_version,
            'success': metrics['overall']['target_met'],
            'achieved_metrics': {
                'correlation': metrics['overall']['average_correlation'],
                'mape': metrics['overall']['average_mape'],
                'accuracy_percent': metrics['overall']['accuracy_percent']
            },
            'target_metrics': {
                'correlation': config['target_correlation'],
                'mape': config['target_mape']
            },
            'training_time_seconds': training_time,
            'model_size_mb': model_size_mb,
            'detailed_metrics': metrics
        }, f, indent=2)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Training time:    {training_time:.2f}s")
    print(f"Model size:       {model_size_mb:.2f} MB")
    print(f"Correlation:      {metrics['overall']['average_correlation']:.4f}")
    print(f"MAPE:             {metrics['overall']['average_mape']:.2f}%")
    print(f"Accuracy:         {metrics['overall']['accuracy_percent']:.2f}%")
    print(f"\nTarget Met:       {'✅ YES' if metrics['overall']['target_met'] else '❌ NO'}")
    print("=" * 80)
    print(f"\nArtifacts saved to: {output_dir}")
    print(f"  - ONNX model:     {onnx_path.name}")
    print(f"  - Metadata:       {metadata_path.name}")
    print(f"  - Report:         {report_path.name}")
    print(f"  - History:        {history_path.name}")
    print("=" * 80)

    # Return exit code based on target achievement
    return 0 if metrics['overall']['target_met'] else 1


if __name__ == '__main__':
    sys.exit(main())
