#!/usr/bin/env python3
"""
PyTorch LSTM Bandwidth Predictor Training Script - Target: ≥98% Accuracy
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
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series sequences"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BandwidthLSTM(nn.Module):
    """Optimized LSTM model for bandwidth prediction targeting ≥98% accuracy"""

    def __init__(self, input_size, hidden_sizes=[256, 128, 64], output_size=4, dropout=0.3):
        super(BandwidthLSTM, self).__init__()

        self.hidden_sizes = hidden_sizes
        self.num_layers = len(hidden_sizes)

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size,
            hidden_sizes[0],
            batch_first=True,
            dropout=dropout if len(hidden_sizes) > 1 else 0
        )
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])

        if len(hidden_sizes) > 1:
            self.lstm2 = nn.LSTM(
                hidden_sizes[0],
                hidden_sizes[1],
                batch_first=True,
                dropout=dropout if len(hidden_sizes) > 2 else 0
            )
            self.bn2 = nn.BatchNorm1d(hidden_sizes[1])

        if len(hidden_sizes) > 2:
            self.lstm3 = nn.LSTM(
                hidden_sizes[1],
                hidden_sizes[2],
                batch_first=True,
                dropout=0
            )

        # Dense layers
        last_hidden = hidden_sizes[-1]
        self.fc1 = nn.Linear(last_hidden, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout * 0.7)

        self.fc3 = nn.Linear(64, 32)

        # Output layer
        self.fc_out = nn.Linear(32, output_size)

        # Activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM 1
        out, _ = self.lstm1(x)
        out = out[:, -1, :]  # Take last timestep
        out = self.bn1(out)

        # LSTM 2
        if hasattr(self, 'lstm2'):
            # Reshape for LSTM input
            out = out.unsqueeze(1).repeat(1, x.size(1), 1)
            out, _ = self.lstm2(out)
            out = out[:, -1, :]
            out = self.bn2(out)

        # LSTM 3
        if hasattr(self, 'lstm3'):
            out = out.unsqueeze(1).repeat(1, x.size(1), 1)
            out, _ = self.lstm3(out)
            out = out[:, -1, :]

        # Dense layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.bn3(out)

        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu(out)

        # Output
        out = self.fc_out(out)

        return out


class BandwidthLSTMTrainer:
    """Trainer for LSTM bandwidth prediction model"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")

        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.train_losses = []
        self.val_losses = []
        self.feature_cols = None
        self.target_cols = None

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
            'throughput_mbps',
            'rtt_ms',
            'packet_loss',
            'jitter_ms',
        ]

        # Add temporal features
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

        # Normalize column names
        if 'bandwidth_mbps' in df.columns and 'throughput_mbps' not in df.columns:
            df['throughput_mbps'] = df['bandwidth_mbps']
        if 'latency_ms' in df.columns and 'rtt_ms' not in df.columns:
            df['rtt_ms'] = df['latency_ms']

        # Filter to only available columns
        feature_cols = [col for col in feature_cols if col in df.columns]

        print(f"\nUsing {len(feature_cols)} features: {feature_cols}")

        return feature_cols

    def prepare_sequences(self, df, window_size=20):
        """Prepare sequences for LSTM training"""
        print(f"\nPreparing sequences with window size {window_size}")

        self.feature_cols = self.prepare_features(df)
        self.target_cols = [
            'throughput_mbps',
            'rtt_ms',
            'packet_loss',
            'jitter_ms'
        ]

        X = []
        y = []

        for i in range(len(df) - window_size):
            sequence = df.iloc[i:i+window_size][self.feature_cols].values
            X.append(sequence)

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

        X_reshaped = X.reshape(-1, X.shape[-1])

        if fit:
            X_scaled = self.scaler_X.fit_transform(X_reshaped)
            y_scaled = self.scaler_y.fit_transform(y)
        else:
            X_scaled = self.scaler_X.transform(X_reshaped)
            y_scaled = self.scaler_y.transform(y)

        X_scaled = X_scaled.reshape(X.shape)

        return X_scaled, y_scaled

    def train(self, X_train, y_train, X_val, y_val):
        """Train the LSTM model"""
        print("\n" + "="*80)
        print("TRAINING PYTORCH LSTM MODEL")
        print("="*80)

        # Create datasets and dataloaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        # Build model
        input_size = X_train.shape[2]
        self.model = BandwidthLSTM(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            output_size=4,
            dropout=0.3
        ).to(self.device)

        print("\nModel Architecture:")
        print(self.model)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['learning_rate']
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['early_stopping_patience']

        print(f"\nStarting training for {self.config['epochs']} epochs...")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Initial learning rate: {self.config['learning_rate']}")

        for epoch in range(self.config['epochs']):
            # Training
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()

            val_loss /= len(val_loader)
            self.val_losses.append(val_loss)

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.config['epochs']}] - "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                checkpoint_path = self.config['checkpoint_path']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        # Load best model
        checkpoint = torch.load(self.config['checkpoint_path'])
        self.model.load_state_dict(checkpoint['model_state_dict'])

        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.6f}")

    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation of model performance"""
        print("\n" + "="*80)
        print("EVALUATING MODEL")
        print("="*80)

        self.model.eval()

        # Get predictions
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred_scaled = self.model(X_test_tensor).cpu().numpy()

        # Denormalize
        y_test_denorm = self.scaler_y.inverse_transform(y_test)
        y_pred_denorm = self.scaler_y.inverse_transform(y_pred_scaled)

        # Calculate metrics
        metrics = {}

        print("\nPer-Target Metrics:")
        print("-" * 80)

        for i, name in enumerate(self.target_cols):
            actual = y_test_denorm[:, i]
            predicted = y_pred_denorm[:, i]

            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)
            mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
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

        # Overall metrics
        avg_correlation = np.mean([m['correlation'] for m in metrics.values()])
        avg_mape = np.mean([m['mape'] for m in metrics.values()])

        print("\n" + "="*80)
        print("OVERALL PERFORMANCE")
        print("="*80)
        print(f"Average Correlation: {avg_correlation:.4f}")
        print(f"Average MAPE:        {avg_mape:.2f}%")

        target_met = avg_correlation >= 0.98 and avg_mape <= 5.0

        print("\n" + "="*80)
        if target_met:
            print("✅ TARGET MET: ≥98% Accuracy Achieved!")
        else:
            print("❌ TARGET NOT MET")
        print("="*80)

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
        print("\nGenerating training plots...")

        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, label='Train Loss', linewidth=2)
        ax.plot(epochs, self.val_losses, label='Val Loss', linewidth=2)
        ax.set_title('Training History', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

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

            axes[row, col].scatter(actual, predicted, alpha=0.5, s=10)

            min_val = min(actual.min(), predicted.min())
            max_val = max(actual.max(), predicted.max())
            axes[row, col].plot([min_val, max_val], [min_val, max_val],
                               'r--', linewidth=2, label='Perfect Prediction')

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

    def export_to_onnx(self, output_path, input_shape):
        """Export model to ONNX format"""
        print(f"\nExporting model to ONNX: {output_path}")

        self.model.eval()

        # Create dummy input
        dummy_input = torch.randn(1, input_shape[0], input_shape[1]).to(self.device)

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Model exported successfully!")
        print(f"File size: {file_size_mb:.2f} MB")

        return file_size_mb

    def save_metadata(self, output_path, metrics, model_size_mb, training_time):
        """Save model metadata"""
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
                'framework': 'PyTorch',
                'input_shape': [self.config['window_size'], len(self.feature_cols)],
                'output_shape': [len(self.target_cols)],
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'layers': [
                    {'type': 'LSTM', 'units': 256, 'dropout': 0.3},
                    {'type': 'BatchNorm1d', 'features': 256},
                    {'type': 'LSTM', 'units': 128, 'dropout': 0.3},
                    {'type': 'BatchNorm1d', 'features': 128},
                    {'type': 'LSTM', 'units': 64},
                    {'type': 'Linear', 'units': 128, 'activation': 'relu'},
                    {'type': 'Dropout', 'rate': 0.3},
                    {'type': 'BatchNorm1d', 'features': 128},
                    {'type': 'Linear', 'units': 64, 'activation': 'relu'},
                    {'type': 'Dropout', 'rate': 0.21},
                    {'type': 'Linear', 'units': 32, 'activation': 'relu'},
                    {'type': 'Linear', 'units': 4, 'activation': 'linear'}
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
        description='Train PyTorch LSTM bandwidth predictor (Target: ≥98% accuracy)'
    )
    parser.add_argument('--data-path', required=True, help='Path to training data CSV')
    parser.add_argument('--output-dir', default='./checkpoints/bandwidth_predictor',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--window-size', type=int, default=20, help='Sequence window size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_version = 'v' + datetime.now().strftime('%Y%m%d_%H%M%S')
    config = {
        'model_version': model_version,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'window_size': args.window_size,
        'early_stopping_patience': 15,
        'checkpoint_path': str(output_dir / 'best_model.pth'),
        'seed': args.seed
    }

    print("=" * 80)
    print("PYTORCH LSTM BANDWIDTH PREDICTOR TRAINING")
    print("Target: ≥98% Accuracy")
    print("=" * 80)
    print(json.dumps(config, indent=2))
    print("=" * 80)

    start_time = time.time()

    trainer = BandwidthLSTMTrainer(config)
    df = trainer.load_data(args.data_path)
    X, y = trainer.prepare_sequences(df, window_size=args.window_size)

    # Temporal split
    test_size = int(len(X) * 0.15)
    val_size = int(len(X) * 0.15)
    train_size = len(X) - test_size - val_size

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    print(f"\nData split:")
    print(f"  Train:      {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")

    X_train, y_train = trainer.normalize_data(X_train, y_train, fit=True)
    X_val, y_val = trainer.normalize_data(X_val, y_val, fit=False)
    X_test, y_test = trainer.normalize_data(X_test, y_test, fit=False)

    trainer.train(X_train, y_train, X_val, y_val)
    metrics, y_test_denorm, y_pred_denorm = trainer.evaluate(X_test, y_test)

    training_time = time.time() - start_time

    trainer.plot_training_history(output_dir)
    trainer.plot_predictions(y_test_denorm, y_pred_denorm, output_dir)

    onnx_path = output_dir / f'bandwidth_lstm_{model_version}.onnx'
    model_size_mb = trainer.export_to_onnx(onnx_path, (args.window_size, len(trainer.feature_cols)))

    metadata_path = output_dir / f'model_metadata_{model_version}.json'
    trainer.save_metadata(metadata_path, metrics, model_size_mb, training_time)

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
                'correlation': 0.98,
                'mape': 5.0
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
    print("=" * 80)

    return 0 if metrics['overall']['target_met'] else 1


if __name__ == '__main__':
    sys.exit(main())
