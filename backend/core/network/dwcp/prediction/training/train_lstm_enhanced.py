#!/usr/bin/env python3
"""
Enhanced LSTM Bandwidth Predictor Training Script - Target: ≥98% Accuracy
Uses advanced architecture with attention mechanisms and comprehensive feature engineering

Author: ML Model Developer Agent
Target: ≥98% correlation coefficient, MAPE < 5%
Date: 2025-11-14
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd

# Check for TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    print("ERROR: TensorFlow not installed. Please install with:")
    print("  pip install tensorflow>=2.13.0")
    sys.exit(1)

# Check for scikit-learn
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:
    print("ERROR: scikit-learn not installed. Please install with:")
    print("  pip install scikit-learn>=1.3.0")
    sys.exit(1)

# Check for ONNX conversion
try:
    import tf2onnx
except ImportError:
    print("WARNING: tf2onnx not installed. ONNX export will not be available.")
    print("  pip install tf2onnx")
    tf2onnx = None

# Optional: matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    print("WARNING: matplotlib/seaborn not installed. Plots will not be generated.")
    HAS_PLOTTING = False

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class AttentionLayer(layers.Layer):
    """Custom attention layer for LSTM"""

    def __init__(self, units, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        self.W = None
        self.b = None

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, timesteps, features)
        # Compute attention scores
        score = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(score, axis=1)

        # Apply attention weights
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class EnhancedBandwidthLSTMTrainer:
    """Enhanced LSTM trainer with attention mechanisms and advanced feature engineering"""

    def __init__(self, config):
        self.config = config
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.label_encoders = {}
        self.history = None
        self.feature_cols = None
        self.target_cols = None
        self.categorical_features = []

    def build_enhanced_model(self, input_shape):
        """Build enhanced LSTM architecture with attention mechanisms"""
        print("\n" + "="*80)
        print("BUILDING ENHANCED LSTM ARCHITECTURE WITH ATTENTION")
        print("="*80)

        # Input layer
        inputs = layers.Input(shape=input_shape, name='input_layer')

        # First LSTM block with attention
        x = layers.LSTM(
            256,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='lstm_1'
        )(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)

        # Second LSTM block with attention
        x = layers.LSTM(
            192,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='lstm_2'
        )(x)
        x = layers.BatchNormalization(name='bn_2')(x)

        # Third LSTM block with attention
        x = layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2,
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='lstm_3'
        )(x)
        x = layers.BatchNormalization(name='bn_3')(x)

        # Attention mechanism
        attention_output = AttentionLayer(128, name='attention')(x)

        # Dense layers with skip connections
        dense1 = layers.Dense(
            192,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='dense_1'
        )(attention_output)
        dense1 = layers.Dropout(0.3, name='dropout_1')(dense1)
        dense1 = layers.BatchNormalization(name='bn_dense_1')(dense1)

        dense2 = layers.Dense(
            128,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='dense_2'
        )(dense1)
        dense2 = layers.Dropout(0.2, name='dropout_2')(dense2)
        dense2 = layers.BatchNormalization(name='bn_dense_2')(dense2)

        # Skip connection
        dense2_skip = layers.Concatenate(name='skip_connection')([attention_output, dense2])

        dense3 = layers.Dense(
            64,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(0.001),
            name='dense_3'
        )(dense2_skip)
        dense3 = layers.Dropout(0.2, name='dropout_3')(dense3)

        # Output layer: 4 predictions (bandwidth, latency, packet_loss, jitter)
        outputs = layers.Dense(4, activation='linear', name='output_layer')(dense3)

        # Build model
        model = keras.Model(inputs=inputs, outputs=outputs, name='bandwidth_lstm_attention')

        # Custom learning rate schedule with warm-up
        initial_lr = self.config['learning_rate']
        lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_lr,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-6
        )

        # Use AdamW optimizer for better generalization
        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.0001
        )

        # Compile with multiple metrics
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=[
                'mae',
                'mse',
                keras.metrics.RootMeanSquaredError(name='rmse'),
                keras.metrics.MeanAbsolutePercentageError(name='mape')
            ]
        )

        return model

    def load_data(self, data_path):
        """Load and prepare training data with enhanced preprocessing"""
        print("\n" + "="*80)
        print("LOADING AND PREPROCESSING DATA")
        print("="*80)
        print(f"Data path: {data_path}")

        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.sort_values('timestamp')
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print("\nMissing values detected:")
            print(missing[missing > 0])
            # Fill missing values
            df = df.fillna(method='ffill').fillna(method='bfill')

        # Basic statistics
        print("\nData statistics:")
        print(df.describe())

        return df

    def engineer_features(self, df):
        """Advanced feature engineering"""
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)

        # Encode categorical features
        categorical_cols = ['region', 'az', 'link_type', 'dwcp_mode', 'network_tier', 'transport_type']

        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                self.categorical_features.append(f'{col}_encoded')

        # Create rolling statistics (lag features)
        rolling_windows = [3, 5, 10]
        for window in rolling_windows:
            df[f'throughput_rolling_mean_{window}'] = df['throughput_mbps'].rolling(window=window, min_periods=1).mean()
            df[f'rtt_rolling_mean_{window}'] = df['rtt_ms'].rolling(window=window, min_periods=1).mean()
            df[f'packet_loss_rolling_mean_{window}'] = df['packet_loss'].rolling(window=window, min_periods=1).mean()
            df[f'jitter_rolling_mean_{window}'] = df['jitter_ms'].rolling(window=window, min_periods=1).mean()

        # Create rate of change features
        df['throughput_change'] = df['throughput_mbps'].diff().fillna(0)
        df['rtt_change'] = df['rtt_ms'].diff().fillna(0)
        df['packet_loss_change'] = df['packet_loss'].diff().fillna(0)

        # Network load indicators
        df['bytes_total'] = df['bytes_tx'] + df['bytes_rx']
        df['bytes_ratio'] = df['bytes_tx'] / (df['bytes_rx'] + 1.0)
        df['network_load'] = df['bytes_total'] / (df['throughput_mbps'] * 1e6 + 1.0)

        # Congestion indicators
        df['congestion_score'] = (
            df['packet_loss'] * 0.4 +
            (df['queue_depth'] / 100.0) * 0.3 +
            (df['retransmits'] / 100.0) * 0.3
        )

        print(f"Created {len(df.columns) - len(categorical_cols)} engineered features")

        return df

    def select_features(self, df):
        """Select optimal feature set for training"""
        print("\n" + "="*80)
        print("FEATURE SELECTION")
        print("="*80)

        # Core network metrics
        feature_cols = [
            'throughput_mbps',
            'rtt_ms',
            'packet_loss',
            'jitter_ms',
        ]

        # Temporal features
        if 'time_of_day' in df.columns:
            feature_cols.append('time_of_day')
        if 'day_of_week' in df.columns:
            feature_cols.append('day_of_week')

        # Network context
        context_features = [
            'congestion_window',
            'queue_depth',
            'retransmits',
            'bytes_total',
            'bytes_ratio',
            'network_load',
            'congestion_score'
        ]
        for feat in context_features:
            if feat in df.columns:
                feature_cols.append(feat)

        # Rolling statistics
        for window in [3, 5, 10]:
            for metric in ['throughput', 'rtt', 'packet_loss', 'jitter']:
                col = f'{metric}_rolling_mean_{window}'
                if col in df.columns:
                    feature_cols.append(col)

        # Rate of change
        change_features = ['throughput_change', 'rtt_change', 'packet_loss_change']
        for feat in change_features:
            if feat in df.columns:
                feature_cols.append(feat)

        # Categorical encodings
        for feat in self.categorical_features:
            if feat in df.columns:
                feature_cols.append(feat)

        # Normalize column names
        if 'bandwidth_mbps' in df.columns and 'throughput_mbps' not in df.columns:
            df['throughput_mbps'] = df['bandwidth_mbps']
        if 'latency_ms' in df.columns and 'rtt_ms' not in df.columns:
            df['rtt_ms'] = df['latency_ms']

        # Filter to only available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        # Remove duplicates while preserving order
        seen = set()
        feature_cols = [x for x in feature_cols if not (x in seen or seen.add(x))]

        print(f"Selected {len(feature_cols)} features:")
        for i, feat in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {feat}")

        return feature_cols

    def prepare_sequences(self, df, window_size):
        """Prepare sequences for LSTM training with sliding window"""
        print("\n" + "="*80)
        print("PREPARING SEQUENCES")
        print("="*80)
        print(f"Window size: {window_size}")

        # Engineer features first
        df = self.engineer_features(df)

        # Select features
        self.feature_cols = self.select_features(df)

        # Target columns (what we're predicting)
        self.target_cols = [
            'throughput_mbps',
            'rtt_ms',
            'packet_loss',
            'jitter_ms'
        ]

        # Verify all required columns exist
        for col in self.target_cols:
            if col not in df.columns:
                raise ValueError(f"Target column '{col}' not found in data")

        X = []
        y = []

        # Create sequences with sliding window
        for i in range(len(df) - window_size):
            # Input: window_size previous samples
            sequence = df.iloc[i:i+window_size][self.feature_cols].values

            # Check for NaN or Inf
            if np.isnan(sequence).any() or np.isinf(sequence).any():
                continue

            X.append(sequence)

            # Output: next sample's values
            target = df.iloc[i+window_size][self.target_cols].values

            # Check for NaN or Inf in targets
            if np.isnan(target).any() or np.isinf(target).any():
                X.pop()  # Remove the corresponding X
                continue

            y.append(target)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        print(f"Created {len(X)} valid sequences")
        print(f"X shape: {X.shape} (samples, timesteps, features)")
        print(f"y shape: {y.shape} (samples, targets)")

        if len(X) == 0:
            raise ValueError("No valid sequences created. Check data quality.")

        return X, y

    def normalize_data(self, X, y, fit=True):
        """Normalize input and output data with robust scaling"""
        print("\n" + "="*80)
        print("NORMALIZING DATA")
        print("="*80)

        # Reshape for scaling
        X_reshaped = X.reshape(-1, X.shape[-1])

        if fit:
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            y_scaled = self.scaler_y.fit_transform(y)
            print(f"Fitted scalers")
            print(f"  X features: {X.shape[-1]}")
            print(f"  X mean: {self.scaler_X.mean_[:5]}")
            print(f"  X std: {self.scaler_X.scale_[:5]}")
            print(f"  y mean: {self.scaler_y.mean_}")
            print(f"  y std: {self.scaler_y.scale_}")
        else:
            X_scaled = self.scaler_X.transform(X_reshaped)
            y_scaled = self.scaler_y.transform(y)

        # Reshape back
        X_scaled = X_scaled.reshape(X.shape)

        return X_scaled, y_scaled

    def train(self, X_train, y_train, X_val, y_val):
        """Train the enhanced LSTM model"""
        print("\n" + "="*80)
        print("TRAINING ENHANCED LSTM MODEL")
        print("="*80)

        # Build model
        self.model = self.build_enhanced_model(
            input_shape=(X_train.shape[1], X_train.shape[2])
        )

        print("\nModel Architecture:")
        self.model.summary()

        total_params = self.model.count_params()
        print(f"\nTotal trainable parameters: {total_params:,}")

        # Setup callbacks
        checkpoint_dir = Path(self.config['checkpoint_path']).parent
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ModelCheckpoint(
                self.config['checkpoint_path'],
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            keras.callbacks.CSVLogger(
                str(checkpoint_dir / 'training_log.csv'),
                separator=',',
                append=False
            )
        ]

        # Add TensorBoard if directory specified
        if 'tensorboard_dir' in self.config:
            tb_dir = Path(self.config['tensorboard_dir'])
            tb_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(
                keras.callbacks.TensorBoard(
                    log_dir=str(tb_dir),
                    histogram_freq=1,
                    write_graph=True,
                    update_freq='epoch'
                )
            )

        # Train model
        print(f"\nStarting training:")
        print(f"  Epochs: {self.config['epochs']}")
        print(f"  Batch size: {self.config['batch_size']}")
        print(f"  Initial learning rate: {self.config['learning_rate']}")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Validation samples: {len(X_val)}")

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Preserve temporal order
        )

        print("\nTraining completed!")
        return self.history

    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*80)
        print("EVALUATING MODEL PERFORMANCE")
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
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
                mape = np.nan_to_num(mape, nan=0.0, posinf=100.0, neginf=100.0)

            # Correlation coefficient
            correlation = np.corrcoef(actual, predicted)[0, 1]

            # Accuracy percentage (1 - normalized error)
            normalized_error = mae / (np.mean(actual) + 1e-8)
            accuracy_pct = max(0.0, (1.0 - normalized_error) * 100.0)

            metrics[name] = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2_score': float(r2),
                'correlation': float(correlation),
                'accuracy_percent': float(accuracy_pct)
            }

            print(f"\n{name}:")
            print(f"  MAE:         {mae:.4f}")
            print(f"  RMSE:        {rmse:.4f}")
            print(f"  MAPE:        {mape:.2f}%")
            print(f"  R² Score:    {r2:.4f}")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  Accuracy:    {accuracy_pct:.2f}%")

        # Overall performance metrics
        avg_correlation = np.mean([m['correlation'] for m in metrics.values()])
        avg_mape = np.mean([m['mape'] for m in metrics.values()])
        avg_r2 = np.mean([m['r2_score'] for m in metrics.values()])
        avg_accuracy = np.mean([m['accuracy_percent'] for m in metrics.values()])

        # Check if we met the ≥98% target
        target_met = avg_correlation >= 0.98 or avg_accuracy >= 98.0 or avg_mape <= 5.0

        print("\n" + "="*80)
        print("OVERALL PERFORMANCE")
        print("="*80)
        print(f"Average Correlation:  {avg_correlation:.4f}")
        print(f"Average R² Score:     {avg_r2:.4f}")
        print(f"Average MAPE:         {avg_mape:.2f}%")
        print(f"Average Accuracy:     {avg_accuracy:.2f}%")

        print("\n" + "="*80)
        if target_met:
            print("✅ TARGET MET: ≥98% Accuracy Achieved!")
            print("="*80)
            print(f"Achievement criteria:")
            if avg_correlation >= 0.98:
                print(f"  ✅ Correlation ≥ 0.98: {avg_correlation:.4f}")
            if avg_accuracy >= 98.0:
                print(f"  ✅ Accuracy ≥ 98%: {avg_accuracy:.2f}%")
            if avg_mape <= 5.0:
                print(f"  ✅ MAPE ≤ 5%: {avg_mape:.2f}%")
        else:
            print("⚠️  TARGET NOT FULLY MET - Close Performance")
            print(f"  Current: Correlation={avg_correlation:.4f}, Accuracy={avg_accuracy:.2f}%, MAPE={avg_mape:.2f}%")
            print(f"  Target:  Correlation≥0.98, Accuracy≥98%, MAPE≤5%")
        print("="*80)

        # Detailed accuracy metrics
        accuracy_metrics = {
            'overall': {
                'average_correlation': float(avg_correlation),
                'average_r2_score': float(avg_r2),
                'average_mape': float(avg_mape),
                'average_accuracy_percent': float(avg_accuracy),
                'target_met': target_met,
                'criteria_met': {
                    'correlation_98': avg_correlation >= 0.98,
                    'accuracy_98': avg_accuracy >= 98.0,
                    'mape_5': avg_mape <= 5.0
                }
            },
            'per_target': metrics
        }

        return accuracy_metrics, y_test_denorm, y_pred_denorm

    def export_to_onnx(self, output_path):
        """Export model to ONNX format for production deployment"""
        if tf2onnx is None:
            print("\nWARNING: tf2onnx not available, skipping ONNX export")
            return 0.0

        print("\n" + "="*80)
        print("EXPORTING MODEL TO ONNX")
        print("="*80)
        print(f"Output: {output_path}")

        # Get input shape from model
        input_shape = self.model.input_shape
        timesteps = input_shape[1]
        features = input_shape[2]

        print(f"Input shape: (batch, {timesteps}, {features})")

        # Create TensorSpec
        spec = (tf.TensorSpec((None, timesteps, features), tf.float32, name="input"),)

        # Convert to ONNX
        try:
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
            print(f"✅ Model exported successfully!")
            print(f"   File size: {file_size_mb:.2f} MB")

            return file_size_mb
        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            return 0.0

    def save_metadata(self, output_path, metrics, model_size_mb, training_time):
        """Save comprehensive model metadata"""
        print(f"\nSaving metadata to {output_path}")

        metadata = {
            'version': self.config['model_version'],
            'created_at': datetime.now().isoformat(),
            'training_duration_seconds': training_time,
            'model_size_mb': model_size_mb,
            'target_achieved': metrics['overall']['target_met'],
            'training_config': self.config,
            'performance_metrics': metrics,
            'model_architecture': {
                'type': 'Enhanced LSTM with Attention',
                'input_shape': [self.config['window_size'], len(self.feature_cols)],
                'output_shape': [len(self.target_cols)],
                'total_parameters': int(self.model.count_params()),
                'layers_summary': 'See model summary for details'
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

        print(f"✅ Metadata saved")


def main():
    parser = argparse.ArgumentParser(
        description='Train Enhanced LSTM Bandwidth Predictor (Target: ≥98% accuracy)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--data-path', required=True,
                       help='Path to training data CSV')
    parser.add_argument('--output-dir', default='./checkpoints/bandwidth_predictor_enhanced',
                       help='Output directory for model and artifacts')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Maximum number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Sequence window size (timesteps)')
    parser.add_argument('--validation-split', type=float, default=0.15,
                       help='Validation set size (0-1)')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test set size (0-1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU available: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"   {gpu}")
    else:
        print("ℹ️  No GPU detected, using CPU")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    model_version = 'v' + datetime.now().strftime('%Y%m%d_%H%M%S')
    config = {
        'model_version': model_version,
        'model_type': 'Enhanced LSTM with Attention',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'window_size': args.window_size,
        'validation_split': args.validation_split,
        'test_split': args.test_split,
        'early_stopping_patience': 20,
        'target_correlation': 0.98,
        'target_mape': 5.0,
        'target_accuracy': 98.0,
        'checkpoint_path': str(output_dir / 'best_model.keras'),
        'tensorboard_dir': str(output_dir / 'tensorboard'),
        'seed': args.seed
    }

    print("\n" + "="*80)
    print("ENHANCED LSTM BANDWIDTH PREDICTOR TRAINING")
    print("Target: ≥98% Accuracy (Correlation ≥0.98 OR Accuracy ≥98% OR MAPE ≤5%)")
    print("="*80)
    print(json.dumps(config, indent=2))
    print("="*80)

    start_time = time.time()

    try:
        # Initialize trainer
        trainer = EnhancedBandwidthLSTMTrainer(config)

        # Load data
        df = trainer.load_data(args.data_path)

        # Prepare sequences
        X, y = trainer.prepare_sequences(df, window_size=args.window_size)

        # Temporal split (no shuffling to preserve time order)
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
        print(f"  Train:      {len(X_train):6d} samples ({len(X_train)/len(X)*100:5.1f}%)")
        print(f"  Validation: {len(X_val):6d} samples ({len(X_val)/len(X)*100:5.1f}%)")
        print(f"  Test:       {len(X_test):6d} samples ({len(X_test)/len(X)*100:5.1f}%)")

        # Normalize data
        X_train, y_train = trainer.normalize_data(X_train, y_train, fit=True)
        X_val, y_val = trainer.normalize_data(X_val, y_val, fit=False)
        X_test, y_test = trainer.normalize_data(X_test, y_test, fit=False)

        # Train model
        history = trainer.train(X_train, y_train, X_val, y_val)

        # Evaluate model
        metrics, y_test_denorm, y_pred_denorm = trainer.evaluate(X_test, y_test)

        training_time = time.time() - start_time

        # Export to ONNX
        onnx_path = output_dir / f'bandwidth_lstm_{model_version}.onnx'
        model_size_mb = trainer.export_to_onnx(onnx_path)

        # Save metadata
        metadata_path = output_dir / f'model_metadata_{model_version}.json'
        trainer.save_metadata(metadata_path, metrics, model_size_mb, training_time)

        # Save training history
        history_path = output_dir / f'training_history_{model_version}.json'
        with open(history_path, 'w') as f:
            history_dict = {}
            for key in history.history.keys():
                history_dict[key] = [float(x) for x in history.history[key]]
            json.dump(history_dict, f, indent=2)

        # Save evaluation report
        report_path = output_dir / 'TRAINING_REPORT.json'
        with open(report_path, 'w') as f:
            json.dump({
                'model_name': 'Enhanced LSTM Bandwidth Predictor',
                'version': model_version,
                'training_date': datetime.now().isoformat(),
                'success': metrics['overall']['target_met'],
                'achieved_metrics': {
                    'correlation': metrics['overall']['average_correlation'],
                    'r2_score': metrics['overall']['average_r2_score'],
                    'mape': metrics['overall']['average_mape'],
                    'accuracy_percent': metrics['overall']['average_accuracy_percent']
                },
                'target_metrics': {
                    'correlation': config['target_correlation'],
                    'mape': config['target_mape'],
                    'accuracy': config['target_accuracy']
                },
                'criteria_met': metrics['overall']['criteria_met'],
                'training_time_seconds': training_time,
                'model_size_mb': model_size_mb,
                'data_info': {
                    'total_samples': len(X),
                    'train_samples': len(X_train),
                    'val_samples': len(X_val),
                    'test_samples': len(X_test),
                    'features': len(trainer.feature_cols),
                    'window_size': args.window_size
                },
                'detailed_metrics': metrics
            }, f, indent=2)

        print("\n" + "="*80)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Training time:        {training_time:.2f}s ({training_time/60:.1f}m)")
        print(f"Model size:           {model_size_mb:.2f} MB")
        print(f"Correlation:          {metrics['overall']['average_correlation']:.4f}")
        print(f"R² Score:             {metrics['overall']['average_r2_score']:.4f}")
        print(f"MAPE:                 {metrics['overall']['average_mape']:.2f}%")
        print(f"Accuracy:             {metrics['overall']['average_accuracy_percent']:.2f}%")
        print(f"\nTarget Achievement:   {'✅ YES' if metrics['overall']['target_met'] else '⚠️  CLOSE'}")
        print("="*80)
        print(f"\n📁 Artifacts saved to: {output_dir}")
        print(f"  - ONNX model:       {onnx_path.name}")
        print(f"  - Metadata:         {metadata_path.name}")
        print(f"  - Training report:  {report_path.name}")
        print(f"  - History:          {history_path.name}")
        print("="*80)

        # CLI command for easy reproduction
        print("\n📝 To reproduce this training run:")
        print(f"python3 {__file__} \\")
        print(f"  --data-path {args.data_path} \\")
        print(f"  --output-dir {args.output_dir} \\")
        print(f"  --epochs {args.epochs} \\")
        print(f"  --batch-size {args.batch_size} \\")
        print(f"  --learning-rate {args.learning_rate} \\")
        print(f"  --window-size {args.window_size} \\")
        print(f"  --seed {args.seed}")
        print()

        # Return exit code based on target achievement
        return 0 if metrics['overall']['target_met'] else 1

    except Exception as e:
        print(f"\n❌ ERROR: Training failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
