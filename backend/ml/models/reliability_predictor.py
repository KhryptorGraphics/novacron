"""
Node Reliability Predictor - DQN-based ML model for predicting node reliability
Target Accuracy: 85%+

Features:
- uptime_percentage: Historical uptime (0-100)
- failure_rate: Failures per hour
- network_quality_score: Network quality metric (0-1)
- geographic_distance: Distance from requester (km)

Output:
- reliability_score: Predicted reliability (0.0-1.0)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from typing import Tuple, List, Dict
import json
import os
from datetime import datetime


class ReliabilityPredictor:
    """DQN-based node reliability prediction model"""

    def __init__(self, state_size: int = 4, learning_rate: float = 0.001):
        """
        Initialize the reliability predictor

        Args:
            state_size: Number of input features (default: 4)
            learning_rate: Learning rate for optimizer
        """
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        # Training parameters
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.memory = []
        self.max_memory_size = 10000

        # Performance tracking
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'mae': [],
            'epochs': []
        }

    def _build_model(self) -> keras.Model:
        """
        Build the DQN neural network architecture

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.state_size,),
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu',
                        kernel_regularizer=keras.regularizers.l2(0.01)),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'accuracy']
        )

        return model

    def update_target_model(self):
        """Update target model weights from main model"""
        self.target_model.set_weights(self.model.get_weights())

    def predict_reliability(
        self,
        uptime: float,
        failure_rate: float,
        network_quality: float,
        distance: float
    ) -> float:
        """
        Predict node reliability score

        Args:
            uptime: Uptime percentage (0-100)
            failure_rate: Failures per hour
            network_quality: Network quality score (0-1)
            distance: Geographic distance in km

        Returns:
            Reliability score (0.0-1.0)
        """
        # Normalize inputs
        state = self._normalize_state(uptime, failure_rate, network_quality, distance)

        # Make prediction
        reliability = self.model.predict(state, verbose=0)[0][0]

        return float(reliability)

    def predict_batch(self, states: np.ndarray) -> np.ndarray:
        """
        Predict reliability for multiple nodes

        Args:
            states: Batch of normalized states (n, 4)

        Returns:
            Array of reliability scores
        """
        return self.model.predict(states, verbose=0).flatten()

    def _normalize_state(
        self,
        uptime: float,
        failure_rate: float,
        network_quality: float,
        distance: float
    ) -> np.ndarray:
        """
        Normalize input features

        Args:
            uptime: Uptime percentage (0-100)
            failure_rate: Failures per hour
            network_quality: Network quality score (0-1)
            distance: Geographic distance in km

        Returns:
            Normalized state array
        """
        # Normalize uptime to 0-1
        norm_uptime = uptime / 100.0

        # Normalize failure rate (cap at 10 failures/hour)
        norm_failure_rate = min(failure_rate / 10.0, 1.0)

        # Network quality already 0-1
        norm_network = network_quality

        # Normalize distance (cap at 10000 km)
        norm_distance = min(distance / 10000.0, 1.0)

        return np.array([[norm_uptime, norm_failure_rate, norm_network, norm_distance]])

    def remember(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store experience in replay memory

        Args:
            state: Current state
            action: Action taken (predicted reliability)
            reward: Reward received
            next_state: Next state
            done: Whether episode is complete
        """
        self.memory.append((state, action, reward, next_state, done))

        # Limit memory size
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def replay(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Train model on batch from replay memory

        Args:
            batch_size: Size of training batch

        Returns:
            Training metrics
        """
        if len(self.memory) < batch_size:
            return {'loss': 0.0, 'accuracy': 0.0}

        # Sample random batch
        batch = np.random.choice(len(self.memory), batch_size, replace=False)

        states = []
        targets = []

        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]

            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state, verbose=0)[0]
                )

            states.append(state[0])
            targets.append([target])

        states = np.array(states)
        targets = np.array(targets)

        # Train model
        history = self.model.fit(
            states,
            targets,
            epochs=1,
            verbose=0,
            batch_size=batch_size
        )

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return {
            'loss': float(history.history['loss'][0]),
            'accuracy': float(history.history['accuracy'][0])
        }

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 100,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Train the model on labeled data

        Args:
            X_train: Training features (n, 4)
            y_train: Training labels (n,)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )

        # Update training history
        self.training_history['loss'].extend(history.history['loss'])
        self.training_history['accuracy'].extend(history.history['accuracy'])
        self.training_history['mae'].extend(history.history['mae'])
        self.training_history['epochs'].extend(range(len(history.history['loss'])))

        # Update target model
        self.update_target_model()

        return {
            'loss': history.history['loss'],
            'accuracy': history.history['accuracy'],
            'mae': history.history['mae'],
            'val_loss': history.history.get('val_loss', []),
            'val_accuracy': history.history.get('val_accuracy', []),
            'val_mae': history.history.get('val_mae', [])
        }

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            X_test: Test features (n, 4)
            y_test: Test labels (n,)

        Returns:
            Evaluation metrics
        """
        loss, mae, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        # Make predictions
        predictions = self.model.predict(X_test, verbose=0).flatten()

        # Calculate additional metrics
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        r2 = 1 - (np.sum((y_test - predictions) ** 2) /
                  np.sum((y_test - np.mean(y_test)) ** 2))

        return {
            'loss': float(loss),
            'mae': float(mae),
            'accuracy': float(accuracy),
            'rmse': float(rmse),
            'r2': float(r2)
        }

    def save_model(self, filepath: str):
        """
        Save model and training history

        Args:
            filepath: Path to save model
        """
        # Save model weights
        self.model.save_weights(filepath)

        # Save training history
        history_path = filepath.replace('.weights.h5', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"Model saved to {filepath}")
        print(f"History saved to {history_path}")

    def load_model(self, filepath: str):
        """
        Load model and training history

        Args:
            filepath: Path to model weights
        """
        # Load model weights
        self.model.load_weights(filepath)
        self.update_target_model()

        # Load training history
        history_path = filepath.replace('.weights.h5', '_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)

        print(f"Model loaded from {filepath}")

    def get_summary(self) -> str:
        """
        Get model architecture summary

        Returns:
            Model summary as string
        """
        from io import StringIO
        import sys

        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()

        self.model.summary()

        sys.stdout = old_stdout
        return buffer.getvalue()


def generate_training_data(n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for reliability prediction

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (features, labels)
    """
    np.random.seed(42)

    # Generate features
    uptime = np.random.uniform(50, 100, n_samples)
    failure_rate = np.random.exponential(0.5, n_samples)
    network_quality = np.random.beta(8, 2, n_samples)
    distance = np.random.exponential(1000, n_samples)

    # Calculate reliability score (ground truth)
    # Higher uptime -> higher reliability
    # Lower failure rate -> higher reliability
    # Higher network quality -> higher reliability
    # Lower distance -> higher reliability

    reliability = (
        0.4 * (uptime / 100.0) +
        0.3 * (1 - np.minimum(failure_rate / 10.0, 1.0)) +
        0.2 * network_quality +
        0.1 * (1 - np.minimum(distance / 10000.0, 1.0))
    )

    # Add noise
    reliability += np.random.normal(0, 0.05, n_samples)
    reliability = np.clip(reliability, 0, 1)

    # Normalize features
    norm_uptime = uptime / 100.0
    norm_failure_rate = np.minimum(failure_rate / 10.0, 1.0)
    norm_distance = np.minimum(distance / 10000.0, 1.0)

    features = np.column_stack([
        norm_uptime,
        norm_failure_rate,
        network_quality,
        norm_distance
    ])

    return features, reliability


if __name__ == "__main__":
    # Generate training data
    print("Generating training data...")
    X, y = generate_training_data(10000)

    # Split into train/val/test
    from sklearn.model_selection import train_test_split

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Initialize model
    print("\nInitializing model...")
    model = ReliabilityPredictor(state_size=4, learning_rate=0.001)
    print(model.get_summary())

    # Train model
    print("\nTraining model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=32
    )

    # Evaluate
    print("\nEvaluating model...")
    metrics = model.evaluate(X_test, y_test)

    print("\nFinal Metrics:")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")

    # Target: 85% accuracy
    if metrics['accuracy'] >= 0.85:
        print(f"\n✅ TARGET ACHIEVED: {metrics['accuracy']*100:.2f}% accuracy (target: 85%)")
    else:
        print(f"\n❌ TARGET NOT MET: {metrics['accuracy']*100:.2f}% accuracy (target: 85%)")

    # Save model
    model_path = "/home/kp/repos/novacron/backend/ml/models/reliability_predictor.weights.h5"
    model.save_model(model_path)

    # Test prediction
    print("\nTest Prediction:")
    reliability = model.predict_reliability(
        uptime=95.5,
        failure_rate=0.2,
        network_quality=0.85,
        distance=500
    )
    print(f"Node with 95.5% uptime, 0.2 failures/hr, 0.85 network quality, 500km distance:")
    print(f"Predicted reliability: {reliability:.4f}")
