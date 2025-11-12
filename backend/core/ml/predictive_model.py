#!/usr/bin/env python3
"""
LSTM-based Predictive Performance Model for DWCP v3
Predicts performance degradation and optimal resource allocation
"""

import numpy as np
import pandas as pd
import json
import logging
import pickle
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """Custom dataset for time series data"""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMPredictor(nn.Module):
    """LSTM neural network for performance prediction"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(LSTMPredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.Tanh(),
            nn.Linear(lstm_output_size // 2, 1),
            nn.Softmax(dim=1)
        )

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, output_size)
        )

    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)

        # Apply attention
        attention_weights = self.attention(lstm_out)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layers
        output = self.fc(context_vector)

        return output, attention_weights


class PerformancePredictorModel:
    """Main class for performance prediction"""

    def __init__(
        self,
        config: Optional[Dict] = None,
        model_dir: str = "/tmp/ml_models"
    ):
        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }

        logger.info(f"Initialized PerformancePredictorModel on {self.device}")

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'sequence_length': 60,  # 60 time steps for prediction
            'prediction_horizon': 5,  # Predict 5 steps ahead
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.2,
            'bidirectional': True,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'early_stopping_patience': 10,
            'feature_columns': [
                'latency_mean', 'latency_std', 'latency_p95', 'latency_p99',
                'throughput_mean', 'throughput_std',
                'error_rate', 'error_count',
                'cpu_usage', 'memory_usage', 'network_io', 'disk_io',
                'compression_ratio', 'prediction_accuracy', 'consensus_time',
                'hour_of_day', 'day_of_week',
                'lag_1', 'lag_5', 'lag_10',
                'rolling_mean_5', 'rolling_mean_15', 'rolling_mean_30',
                'rolling_std_5', 'rolling_std_15', 'rolling_std_30'
            ],
            'target_column': 'latency_mean'
        }

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load production data for training"""
        logger.info(f"Loading data from {data_path}")

        if data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data['features'])
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} records")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for training"""
        logger.info("Engineering features...")

        # Ensure timestamp is datetime
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found")
        else:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Temporal features
        if 'timestamp' in df.columns:
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Statistical features for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['hour_of_day', 'day_of_week', 'is_weekend']:
                # Lagged features
                for lag in [1, 5, 10, 30]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)

                # Rolling statistics
                for window in [5, 15, 30, 60]:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()

                # Rate of change
                df[f'{col}_diff'] = df[col].diff()
                df[f'{col}_pct_change'] = df[col].pct_change()

        # Drop NaN values created by feature engineering
        df = df.dropna()

        logger.info(f"Engineered features, shape: {df.shape}")
        return df

    def create_sequences(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
        prediction_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        sequences = []
        sequence_targets = []

        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            seq = data[i:i + sequence_length]
            target = targets[i + sequence_length + prediction_horizon - 1]
            sequences.append(seq)
            sequence_targets.append(target)

        return np.array(sequences), np.array(sequence_targets)

    def prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training"""
        logger.info("Preparing data for training...")

        # Select features
        feature_cols = [col for col in self.config['feature_columns'] if col in df.columns]
        if not feature_cols:
            # Use all numeric columns if specified features not found
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if 'timestamp' in feature_cols:
                feature_cols.remove('timestamp')

        logger.info(f"Using {len(feature_cols)} features")

        # Prepare features and targets
        X = df[feature_cols].values
        y = df[self.config['target_column']].values if self.config['target_column'] in df.columns else X[:, 0]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        # Create sequences
        X_seq, y_seq = self.create_sequences(
            X_scaled,
            y_scaled,
            self.config['sequence_length'],
            self.config['prediction_horizon']
        )

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_seq, y_seq, test_size=test_size, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size / (1 - test_size), shuffle=False
        )

        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)

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
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False
        )

        logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

        return train_loader, val_loader, test_loader

    def build_model(self, input_size: int) -> None:
        """Build LSTM model"""
        logger.info("Building LSTM model...")

        self.model = LSTMPredictor(
            input_size=input_size,
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=1,
            dropout=self.config['dropout'],
            bidirectional=self.config['bidirectional']
        ).to(self.device)

        logger.info(f"Model architecture:\n{self.model}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters())}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_best: bool = True
    ) -> Dict:
        """Train the model"""
        logger.info("Starting training...")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0

            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                outputs, _ = self.model(sequences)
                loss = criterion(outputs.squeeze(), targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs.squeeze() - targets)).item()

            train_loss /= len(train_loader)
            train_mae /= len(train_loader)

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0

            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    outputs, _ = self.model(sequences)
                    loss = criterion(outputs.squeeze(), targets)

                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs.squeeze() - targets)).item()

            val_loss /= len(val_loader)
            val_mae /= len(val_loader)

            # Update learning rate
            scheduler.step(val_loss)

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)

            logger.info(
                f"Epoch {epoch + 1}/{self.config['epochs']} - "
                f"Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, "
                f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_best:
                    self.save_model('best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break

        logger.info("Training complete")
        return self.history

    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate the model"""
        logger.info("Evaluating model...")

        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                outputs, _ = self.model(sequences)
                predictions.extend(outputs.squeeze().cpu().numpy())
                actuals.extend(targets.numpy())

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Inverse transform predictions
        predictions_original = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_original = self.target_scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

        # Calculate metrics
        mse = mean_squared_error(actuals_original, predictions_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals_original, predictions_original)
        r2 = r2_score(actuals_original, predictions_original)

        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actuals_original - predictions_original) / (actuals_original + 1e-10))) * 100

        # Calculate prediction accuracy (within 5% threshold)
        accuracy_5pct = np.mean(np.abs((actuals_original - predictions_original) / (actuals_original + 1e-10)) < 0.05) * 100

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'accuracy_5pct': float(accuracy_5pct),
            'num_predictions': len(predictions)
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def predict(self, sequence: np.ndarray) -> Tuple[float, np.ndarray]:
        """Make a single prediction"""
        self.model.eval()

        # Scale input
        sequence_scaled = self.scaler.transform(sequence)
        sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output, attention_weights = self.model(sequence_tensor)

        # Inverse transform prediction
        prediction_scaled = output.squeeze().cpu().numpy()
        prediction = self.target_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

        attention = attention_weights.squeeze().cpu().numpy()

        return prediction, attention

    def predict_degradation(
        self,
        current_metrics: Dict,
        forecast_horizon: int = 30
    ) -> Dict:
        """Predict performance degradation"""
        logger.info(f"Predicting degradation for {forecast_horizon} steps ahead")

        # This would use the trained model to predict future performance
        # For now, implementing simplified logic

        degradation_forecast = {
            'timestamp': datetime.now().isoformat(),
            'forecast_horizon': forecast_horizon,
            'degradation_probability': 0.0,
            'predicted_metrics': {},
            'recommendations': []
        }

        # Calculate degradation probability based on current trends
        # This is a simplified version - production would use the full LSTM model

        return degradation_forecast

    def optimize_allocation(self, constraints: Dict) -> Dict:
        """Predict optimal resource allocation"""
        logger.info("Optimizing resource allocation")

        optimization_result = {
            'timestamp': datetime.now().isoformat(),
            'optimal_allocation': {},
            'expected_performance': {},
            'cost_estimate': 0.0,
            'confidence': 0.0
        }

        return optimization_result

    def save_model(self, filename: str) -> None:
        """Save model and scalers"""
        model_path = self.model_dir / filename
        scaler_path = self.model_dir / f"{filename}_scaler.pkl"
        target_scaler_path = self.model_dir / f"{filename}_target_scaler.pkl"
        config_path = self.model_dir / f"{filename}_config.json"

        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'history': self.history
        }, model_path)

        # Save scalers
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        with open(target_scaler_path, 'wb') as f:
            pickle.dump(self.target_scaler, f)

        # Save config
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename: str) -> None:
        """Load model and scalers"""
        model_path = self.model_dir / filename
        scaler_path = self.model_dir / f"{filename}_scaler.pkl"
        target_scaler_path = self.model_dir / f"{filename}_target_scaler.pkl"

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.config = checkpoint['config']
        self.history = checkpoint['history']

        # Build model architecture
        self.build_model(input_size=len(self.config['feature_columns']))
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load scalers
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        with open(target_scaler_path, 'rb') as f:
            self.target_scaler = pickle.load(f)

        logger.info(f"Model loaded from {model_path}")


def main():
    """Main training pipeline"""
    logger.info("Starting LSTM Predictive Model Training")

    # Initialize model
    model = PerformancePredictorModel()

    # Load and prepare data (example)
    # In production, this would load real production data
    logger.info("Training pipeline ready")
    logger.info("Use model.load_data() to load production metrics")
    logger.info("Use model.prepare_data() to prepare training data")
    logger.info("Use model.train() to train the model")
    logger.info("Use model.evaluate() to evaluate performance")

    return model


if __name__ == "__main__":
    model = main()
