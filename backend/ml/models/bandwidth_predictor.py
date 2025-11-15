"""
Bandwidth Predictor Model - LSTM+DDQN Architecture
Target Accuracy: 98% for datacenter, 70% for internet

Author: Novacron ML Team
Date: 2025-11-14
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Dict
import pickle
import os


class LSTMPredictor:
    """LSTM-based time series predictor for bandwidth patterns"""

    def __init__(self, sequence_length: int = 10, features: int = 4):
        """
        Initialize LSTM predictor

        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features (latency, bandwidth, packet_loss, reliability)
        """
        self.sequence_length = sequence_length
        self.features = features
        self.model = self._build_model()
        self.history = None

    def _build_model(self) -> Model:
        """Build LSTM architecture: 128→64→32 units"""
        model = tf.keras.Sequential([
            # First LSTM layer with return sequences
            layers.LSTM(
                128,
                return_sequences=True,
                input_shape=(self.sequence_length, self.features),
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),

            # Second LSTM layer
            layers.LSTM(
                64,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),

            # Third LSTM layer
            layers.LSTM(
                32,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.BatchNormalization(),

            # Dense layers
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(8, activation='relu'),
            layers.Dense(1)  # Single output: predicted bandwidth
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the LSTM model

        Args:
            X_train: Training sequences (samples, sequence_length, features)
            y_train: Training targets (samples, 1)
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size

        Returns:
            Training history
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]

        validation_data = (X_val, y_val) if X_val is not None else None

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict bandwidth for given sequences"""
        return self.model.predict(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        mse = np.mean((predictions - y_test) ** 2)
        mae = np.mean(np.abs(predictions - y_test))
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100

        # Calculate accuracy (within 2% tolerance)
        tolerance = 0.02
        accurate_predictions = np.abs((predictions - y_test) / y_test) <= tolerance
        accuracy = np.mean(accurate_predictions) * 100

        return {
            'mse': float(mse),
            'mae': float(mae),
            'mape': float(mape),
            'accuracy': float(accuracy)
        }


class DDQNAgent:
    """Double Deep Q-Network for bandwidth allocation decisions"""

    def __init__(self, state_size: int = 4, action_size: int = 10):
        """
        Initialize DDQN agent

        Args:
            state_size: Size of state vector [latency, bandwidth, packet_loss, reliability]
            action_size: Number of discrete bandwidth allocation levels
        """
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_frequency = 10

        # Networks
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

        self.training_step = 0

    def _build_model(self) -> Model:
        """Build DQN neural network"""
        model = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),

            layers.Dense(self.action_size, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def update_target_model(self):
        """Copy weights from model to target_model"""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        state = np.reshape(state, [1, self.state_size])
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train on batch from replay memory using Double DQN"""
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Double DQN: use model to select action, target_model to evaluate
        next_q_values_model = self.model.predict(next_states, verbose=0)
        next_q_values_target = self.target_model.predict(next_states, verbose=0)

        # Get current Q values
        current_q_values = self.model.predict(states, verbose=0)

        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                # Double DQN update
                best_action = np.argmax(next_q_values_model[i])
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * next_q_values_target[i][best_action]

        # Train the model
        self.model.fit(states, current_q_values, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.update_target_frequency == 0:
            self.update_target_model()

    def train(self, env_simulator, episodes: int = 2000) -> Dict:
        """
        Train DDQN agent

        Args:
            env_simulator: Environment simulator with step() and reset() methods
            episodes: Number of training episodes

        Returns:
            Training statistics
        """
        episode_rewards = []
        episode_losses = []

        for episode in range(episodes):
            state = env_simulator.reset()
            total_reward = 0
            done = False
            steps = 0

            while not done and steps < 100:
                action = self.act(state)
                next_state, reward, done, _ = env_simulator.step(action)

                self.remember(state, action, reward, next_state, done)
                self.replay()

                state = next_state
                total_reward += reward
                steps += 1

            episode_rewards.append(total_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{episodes}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        return {
            'episode_rewards': episode_rewards,
            'final_epsilon': self.epsilon,
            'avg_reward': np.mean(episode_rewards[-100:])
        }


class BandwidthPredictor:
    """Combined LSTM+DDQN Bandwidth Predictor"""

    def __init__(self, sequence_length: int = 10, features: int = 4, action_size: int = 10):
        """
        Initialize hybrid bandwidth predictor

        Args:
            sequence_length: LSTM sequence length
            features: Number of input features
            action_size: Number of bandwidth allocation actions
        """
        self.lstm_predictor = LSTMPredictor(sequence_length, features)
        self.ddqn_agent = DDQNAgent(state_size=features, action_size=action_size)
        self.sequence_length = sequence_length
        self.features = features

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray = None, y_val: np.ndarray = None,
                   epochs: int = 100) -> Dict:
        """Train LSTM component"""
        return self.lstm_predictor.train(X_train, y_train, X_val, y_val, epochs)

    def train_ddqn(self, env_simulator, episodes: int = 2000) -> Dict:
        """Train DDQN component"""
        return self.ddqn_agent.train(env_simulator, episodes)

    def predict_bandwidth(self, sequence: np.ndarray) -> float:
        """Predict bandwidth using LSTM"""
        if sequence.ndim == 2:
            sequence = np.expand_dims(sequence, axis=0)
        return float(self.lstm_predictor.predict(sequence)[0][0])

    def decide_allocation(self, state: np.ndarray) -> int:
        """Decide bandwidth allocation using DDQN"""
        return self.ddqn_agent.act(state)

    def evaluate_lstm(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate LSTM performance"""
        return self.lstm_predictor.evaluate(X_test, y_test)

    def save(self, directory: str):
        """Save both models"""
        os.makedirs(directory, exist_ok=True)

        # Save LSTM
        self.lstm_predictor.model.save(os.path.join(directory, 'lstm_model.h5'))

        # Save DDQN
        self.ddqn_agent.model.save(os.path.join(directory, 'ddqn_model.h5'))
        self.ddqn_agent.target_model.save(os.path.join(directory, 'ddqn_target_model.h5'))

        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'features': self.features,
            'action_size': self.ddqn_agent.action_size,
            'epsilon': self.ddqn_agent.epsilon
        }
        with open(os.path.join(directory, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, directory: str):
        """Load both models"""
        # Load LSTM
        self.lstm_predictor.model = tf.keras.models.load_model(
            os.path.join(directory, 'lstm_model.h5')
        )

        # Load DDQN
        self.ddqn_agent.model = tf.keras.models.load_model(
            os.path.join(directory, 'ddqn_model.h5')
        )
        self.ddqn_agent.target_model = tf.keras.models.load_model(
            os.path.join(directory, 'ddqn_target_model.h5')
        )

        # Load metadata
        with open(os.path.join(directory, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        self.ddqn_agent.epsilon = metadata['epsilon']


if __name__ == "__main__":
    print("Bandwidth Predictor Model Initialized")
    print("Architecture: LSTM (128→64→32) + DDQN")
    print("Target Accuracy: 98% (datacenter), 70% (internet)")
