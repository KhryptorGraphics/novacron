"""
Consensus Latency Predictor using LSTM
Target: 90% accuracy for consensus protocol latency prediction

Features:
- node_count: Number of nodes in consensus
- network_mode: LAN/WAN (encoded)
- byzantine_ratio: Ratio of byzantine nodes (0.0-0.33)
- message_size: Message size in bytes

Output: expected_latency in milliseconds
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
import os
from typing import Tuple, Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusLatencyPredictor:
    """LSTM-based predictor for consensus protocol latency"""

    def __init__(self, sequence_length: int = 10):
        """
        Initialize the predictor

        Args:
            sequence_length: Number of timesteps for LSTM input
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_names = ['node_count', 'network_mode', 'byzantine_ratio', 'message_size']
        self.history = None

    def _build_model(self, input_shape: Tuple[int, int]) -> models.Sequential:
        """
        Build LSTM model architecture

        Args:
            input_shape: Shape of input sequences (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First LSTM layer with return sequences for stacking
            layers.LSTM(
                64,
                return_sequences=True,
                input_shape=input_shape,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),

            # Second LSTM layer
            layers.LSTM(
                32,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),

            # Dense layers for prediction
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)  # Output: latency prediction
        ])

        # Compile with Adam optimizer and MAE loss
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mae',
            metrics=['mse', 'mae', tf.keras.metrics.MeanAbsolutePercentageError(name='mape')]
        )

        logger.info(f"Model built with input shape: {input_shape}")
        return model

    def _encode_features(self, node_count: int, network_mode: str,
                        byzantine_ratio: float, msg_size: int) -> np.ndarray:
        """
        Encode raw features into model input format

        Args:
            node_count: Number of nodes in consensus
            network_mode: 'LAN' or 'WAN'
            byzantine_ratio: Ratio of byzantine nodes (0.0-0.33)
            msg_size: Message size in bytes

        Returns:
            Encoded feature vector
        """
        # Encode network mode: LAN=0, WAN=1
        network_encoded = 1.0 if network_mode.upper() == 'WAN' else 0.0

        features = np.array([
            float(node_count),
            network_encoded,
            float(byzantine_ratio),
            float(msg_size)
        ])

        return features.reshape(1, -1)

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the LSTM model

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history dictionary
        """
        logger.info(f"Training with {len(X_train)} samples")

        # Scale features and targets
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_scaled)

        # Build model
        self.model = self._build_model(input_shape=(self.sequence_length, X_train.shape[1]))

        # Prepare validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).ravel()
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_scaled)
            validation_data = (X_val_seq, y_val_seq)

        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        # Train model
        self.history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )

        # Calculate final metrics
        metrics = self._calculate_accuracy(X_train_seq, y_train_seq)
        logger.info(f"Training completed - Accuracy: {metrics['accuracy']:.2f}%")

        return {
            'history': self.history.history,
            'final_metrics': metrics
        }

    def _calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Calculate prediction accuracy within 10% threshold

        Args:
            X: Feature sequences
            y: True latency values (scaled)

        Returns:
            Dictionary with accuracy metrics
        """
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.scaler_y.inverse_transform(predictions_scaled)
        y_true = self.scaler_y.inverse_transform(y.reshape(-1, 1))

        # Calculate percentage error
        percentage_errors = np.abs((predictions - y_true) / y_true) * 100

        # Accuracy: predictions within 10% of actual
        within_10_percent = np.sum(percentage_errors <= 10) / len(percentage_errors) * 100

        return {
            'accuracy': within_10_percent,
            'mean_absolute_error': np.mean(np.abs(predictions - y_true)),
            'mean_percentage_error': np.mean(percentage_errors),
            'rmse': np.sqrt(np.mean((predictions - y_true) ** 2))
        }

    def predict_latency(self, node_count: int, network_mode: str,
                       byzantine_ratio: float, msg_size: int,
                       historical_data: List[np.ndarray] = None) -> Dict:
        """
        Predict consensus latency for given parameters

        Args:
            node_count: Number of nodes in consensus
            network_mode: 'LAN' or 'WAN'
            byzantine_ratio: Ratio of byzantine nodes (0.0-0.33)
            msg_size: Message size in bytes
            historical_data: Previous feature vectors for sequence (optional)

        Returns:
            Dictionary with prediction and confidence
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Encode current features
        current_features = self._encode_features(node_count, network_mode, byzantine_ratio, msg_size)

        # Create sequence
        if historical_data is None or len(historical_data) < self.sequence_length - 1:
            # Use current features repeated for sequence
            sequence = np.repeat(current_features, self.sequence_length, axis=0)
        else:
            # Use historical data + current
            sequence = np.vstack([
                historical_data[-(self.sequence_length-1):],
                current_features
            ])

        # Scale and reshape
        sequence_scaled = self.scaler_X.transform(sequence)
        sequence_reshaped = sequence_scaled.reshape(1, self.sequence_length, -1)

        # Predict
        prediction_scaled = self.model.predict(sequence_reshaped, verbose=0)
        prediction = self.scaler_y.inverse_transform(prediction_scaled)[0][0]

        # Calculate confidence based on training performance
        confidence = self._estimate_confidence(node_count, network_mode, byzantine_ratio, msg_size)

        return {
            'predicted_latency_ms': float(prediction),
            'confidence': float(confidence),
            'parameters': {
                'node_count': node_count,
                'network_mode': network_mode,
                'byzantine_ratio': byzantine_ratio,
                'message_size_bytes': msg_size
            }
        }

    def _estimate_confidence(self, node_count: int, network_mode: str,
                           byzantine_ratio: float, msg_size: int) -> float:
        """
        Estimate prediction confidence based on parameter ranges

        Args:
            node_count: Number of nodes
            network_mode: Network type
            byzantine_ratio: Byzantine ratio
            msg_size: Message size

        Returns:
            Confidence score (0-1)
        """
        # Base confidence from training accuracy
        if self.history:
            val_mape = self.history.history.get('val_mape', [0])[-1]
            base_confidence = max(0, 1 - (val_mape / 100))
        else:
            base_confidence = 0.9

        # Adjust for parameter extremes
        penalties = []

        # Node count penalty (very low or very high)
        if node_count < 3 or node_count > 100:
            penalties.append(0.1)

        # Byzantine ratio penalty (high ratios)
        if byzantine_ratio > 0.25:
            penalties.append(0.05)

        # Message size penalty (very large messages)
        if msg_size > 1000000:  # > 1MB
            penalties.append(0.05)

        confidence = base_confidence - sum(penalties)
        return max(0.5, min(1.0, confidence))  # Clamp between 0.5 and 1.0

    def save_model(self, filepath: str):
        """Save model and scalers"""
        if self.model is None:
            raise ValueError("No model to save")

        # Save Keras model
        self.model.save(f"{filepath}_model.keras")

        # Save scalers and metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'feature_names': self.feature_names,
            'scaler_X_mean': self.scaler_X.mean_.tolist(),
            'scaler_X_scale': self.scaler_X.scale_.tolist(),
            'scaler_y_mean': self.scaler_y.mean_.tolist(),
            'scaler_y_scale': self.scaler_y.scale_.tolist()
        }

        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model and scalers"""
        # Load Keras model
        self.model = models.load_model(f"{filepath}_model.keras")

        # Load metadata
        with open(f"{filepath}_metadata.json", 'r') as f:
            metadata = json.load(f)

        self.sequence_length = metadata['sequence_length']
        self.feature_names = metadata['feature_names']

        # Restore scalers
        self.scaler_X.mean_ = np.array(metadata['scaler_X_mean'])
        self.scaler_X.scale_ = np.array(metadata['scaler_X_scale'])
        self.scaler_y.mean_ = np.array(metadata['scaler_y_mean'])
        self.scaler_y.scale_ = np.array(metadata['scaler_y_scale'])

        logger.info(f"Model loaded from {filepath}")


def generate_synthetic_training_data(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for consensus latency prediction

    This simulates realistic consensus protocol behavior:
    - LAN networks: lower latency, less variance
    - WAN networks: higher latency, more variance
    - Byzantine nodes increase latency
    - Larger messages increase latency
    - More nodes increase latency (logarithmically)

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (features, latencies)
    """
    np.random.seed(42)

    features = []
    latencies = []

    for _ in range(n_samples):
        # Random parameters
        node_count = np.random.randint(3, 101)
        network_mode = np.random.choice([0, 1])  # 0=LAN, 1=WAN
        byzantine_ratio = np.random.uniform(0.0, 0.33)
        msg_size = np.random.randint(100, 100000)

        # Base latency calculation
        base_latency = 10 if network_mode == 0 else 100  # LAN vs WAN base

        # Node count impact (logarithmic)
        node_impact = np.log10(node_count) * 5

        # Byzantine impact (linear)
        byzantine_impact = byzantine_ratio * 50

        # Message size impact (logarithmic)
        msg_impact = np.log10(msg_size) * 2

        # Network variance
        variance = 5 if network_mode == 0 else 30
        noise = np.random.normal(0, variance)

        # Final latency
        latency = base_latency + node_impact + byzantine_impact + msg_impact + noise
        latency = max(1, latency)  # Ensure positive

        features.append([node_count, network_mode, byzantine_ratio, msg_size])
        latencies.append(latency)

    return np.array(features), np.array(latencies)


if __name__ == "__main__":
    # Example usage and training
    print("Generating synthetic training data...")
    X, y = generate_synthetic_training_data(n_samples=10000)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Initialize and train predictor
    predictor = ConsensusLatencyPredictor(sequence_length=10)

    print("\nTraining model...")
    training_results = predictor.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    X_test_scaled = predictor.scaler_X.transform(X_test)
    y_test_scaled = predictor.scaler_y.transform(y_test.reshape(-1, 1)).ravel()
    X_test_seq, y_test_seq = predictor._create_sequences(X_test_scaled, y_test_scaled)

    test_metrics = predictor._calculate_accuracy(X_test_seq, y_test_seq)
    print(f"\nTest Set Metrics:")
    print(f"  Accuracy (within 10%): {test_metrics['accuracy']:.2f}%")
    print(f"  Mean Absolute Error: {test_metrics['mean_absolute_error']:.2f} ms")
    print(f"  Mean Percentage Error: {test_metrics['mean_percentage_error']:.2f}%")
    print(f"  RMSE: {test_metrics['rmse']:.2f} ms")

    # Example predictions
    print("\n" + "="*60)
    print("Example Predictions:")
    print("="*60)

    test_cases = [
        (7, 'LAN', 0.1, 1000),
        (21, 'WAN', 0.2, 5000),
        (50, 'LAN', 0.0, 500),
        (100, 'WAN', 0.33, 50000)
    ]

    for node_count, network_mode, byz_ratio, msg_size in test_cases:
        result = predictor.predict_latency(node_count, network_mode, byz_ratio, msg_size)
        print(f"\nNodes: {node_count}, Network: {network_mode}, Byzantine: {byz_ratio:.2f}, Msg: {msg_size}B")
        print(f"  Predicted Latency: {result['predicted_latency_ms']:.2f} ms")
        print(f"  Confidence: {result['confidence']:.2%}")

    # Save model
    model_path = "/home/kp/repos/novacron/backend/ml/models/consensus_latency_predictor"
    predictor.save_model(model_path)
    print(f"\nModel saved to {model_path}")
