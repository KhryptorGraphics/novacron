#!/usr/bin/env python3
"""
Deep Q-Network (DQN) Training for DWCP Task Partitioning
Implements DQN with prioritized experience replay and Double DQN
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import time
from typing import Tuple, List, Dict, Any
import logging
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for DQN training"""

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, td_error: float = None):
        """Add experience with priority based on TD error"""
        if td_error is None:
            # Default high priority for new experiences
            priority = 1.0
        else:
            priority = (abs(td_error) + 1e-5) ** self.alpha

        experience = (state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)

        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        scaled_priorities = priorities ** beta
        sample_probs = scaled_priorities / np.sum(scaled_priorities)

        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probs, replace=False)

        # Get samples
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        for idx in indices:
            s, a, r, ns, d = self.buffer[idx]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * sample_probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), weights, indices)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on new TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-5) ** self.alpha
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network Agent with Double DQN and Prioritized Replay"""

    def __init__(self, state_size: int = 20, action_size: int = 15,
                 learning_rate: float = 0.001, gamma: float = 0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=10000, alpha=0.6)
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate

        # Neural networks
        self.model = self._build_model()  # Online network
        self.target_model = self._build_model()  # Target network
        self.update_target_model()

        # Training metrics
        self.loss_history = []
        self.reward_history = []
        self.epsilon_history = []

    def _build_model(self) -> keras.Model:
        """Build the DQN neural network"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),

            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),

            keras.layers.Dense(self.action_size, activation='linear')
        ])

        # Compile with Huber loss (more robust to outliers)
        model.compile(
            loss=keras.losses.Huber(),
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )

        return model

    def update_target_model(self):
        """Copy weights from online model to target model"""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Reshape state for prediction
        state = state.reshape(1, -1)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        # Calculate TD error for prioritization
        state_batch = state.reshape(1, -1)
        next_state_batch = next_state.reshape(1, -1)

        current_q = self.model.predict(state_batch, verbose=0)[0][action]

        if done:
            target_q = reward
        else:
            # Double DQN: use online network to select action, target network to evaluate
            next_action = np.argmax(self.model.predict(next_state_batch, verbose=0)[0])
            target_q = reward + self.gamma * self.target_model.predict(next_state_batch, verbose=0)[0][next_action]

        td_error = target_q - current_q
        self.memory.add(state, action, reward, next_state, done, td_error)

    def replay(self, batch_size: int = 32, beta: float = 0.4):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return

        # Sample batch with importance weights
        states, actions, rewards, next_states, dones, weights, indices = \
            self.memory.sample(batch_size, beta)

        # Prepare training data
        target = self.model.predict(states, verbose=0)
        target_next = self.model.predict(next_states, verbose=0)
        target_val = self.target_model.predict(next_states, verbose=0)

        td_errors = []

        for i in range(batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                # Double DQN update
                next_action = np.argmax(target_next[i])
                target[i][actions[i]] = rewards[i] + self.gamma * target_val[i][next_action]

            # Calculate TD error for priority update
            td_error = target[i][actions[i]] - self.model.predict(states[i:i+1], verbose=0)[0][actions[i]]
            td_errors.append(td_error)

        # Apply importance sampling weights
        sample_weights = weights.reshape(-1, 1)

        # Train the model
        history = self.model.fit(states, target, sample_weight=weights,
                                 epochs=1, verbose=0, batch_size=batch_size)

        # Update priorities in replay buffer
        self.memory.update_priorities(indices, np.array(td_errors))

        # Store loss
        if history.history['loss']:
            self.loss_history.append(history.history['loss'][0])

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath: str):
        """Save the model and training metadata"""
        # Save model weights
        self.model.save(filepath + '.h5')

        # Export to ONNX format for Go inference
        self.export_onnx(filepath + '.onnx')

        # Save metadata
        metadata = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'loss_history': self.loss_history[-100:],  # Last 100 losses
            'reward_history': self.reward_history[-100:],  # Last 100 episodes
            'training_episodes': len(self.reward_history)
        }

        with open(filepath + '.metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def export_onnx(self, filepath: str):
        """Export model to ONNX format for Go inference"""
        try:
            import tf2onnx

            # Define input signature
            spec = (tf.TensorSpec((None, self.state_size), tf.float32, name="input"),)

            # Convert to ONNX
            model_proto, _ = tf2onnx.convert.from_keras(
                self.model,
                input_signature=spec,
                opset=13,
                output_path=filepath
            )

            logger.info(f"Model exported to ONNX format: {filepath}")
        except ImportError:
            logger.warning("tf2onnx not installed, skipping ONNX export")

    def load(self, filepath: str):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath + '.h5')
        self.update_target_model()

        # Load metadata
        with open(filepath + '.metadata.json', 'r') as f:
            metadata = json.load(f)
            self.epsilon = metadata.get('epsilon', self.epsilon_min)
            self.reward_history = metadata.get('reward_history', [])
            self.loss_history = metadata.get('loss_history', [])


class DWCPEnvironment:
    """Simulated DWCP network environment for training"""

    def __init__(self):
        self.state_size = 20
        self.action_size = 15
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        # Initialize stream metrics
        self.stream_bandwidth = np.random.uniform(50, 150, 4)  # 50-150 Mbps
        self.stream_latency = np.random.uniform(5, 20, 4)  # 5-20ms
        self.stream_congestion = np.random.uniform(0, 0.3, 4)  # 0-30% congestion
        self.stream_success_rate = np.random.uniform(0.85, 0.99, 4)  # 85-99% success

        # Initialize task properties
        self.task_queue_depth = np.random.randint(1, 50)
        self.task_size = np.random.uniform(1e6, 1e9)  # 1MB to 1GB
        self.task_priority = np.random.uniform(0, 1)
        self.time_of_day = np.random.uniform(0, 1)

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Get current state as numpy array"""
        state = np.zeros(self.state_size)

        # Stream metrics (normalized)
        state[0:4] = self.stream_bandwidth / 1000  # Normalize to 0-1
        state[4:8] = self.stream_latency / 100
        state[8:12] = self.stream_congestion
        state[12:16] = self.stream_success_rate

        # Task properties
        state[16] = min(self.task_queue_depth / 100, 1.0)
        state[17] = min(self.task_size / 1e9, 1.0)
        state[18] = self.task_priority
        state[19] = self.time_of_day

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Execute action and return next state, reward, done"""
        # Calculate reward based on action
        reward = self._calculate_reward(action)

        # Update environment state
        self._update_state(action)

        # Check if episode is done
        self.task_queue_depth = max(0, self.task_queue_depth - 1)
        done = self.task_queue_depth == 0

        # Generate new task if not done
        if not done:
            self.task_size = np.random.uniform(1e6, 1e9)
            self.task_priority = np.random.uniform(0, 1)

        next_state = self._get_state()

        return next_state, reward, done

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward for the given action"""
        # Determine which streams are used
        if action < 4:
            # Single stream
            streams = [action]
        elif action < 10:
            # Two streams
            stream_map = {4: [0,1], 5: [0,2], 6: [0,3], 7: [1,2], 8: [1,3], 9: [2,3]}
            streams = stream_map[action]
        elif action < 14:
            # Three streams
            stream_map = {10: [0,1,2], 11: [0,1,3], 12: [0,2,3], 13: [1,2,3]}
            streams = stream_map[action]
        else:
            # All streams
            streams = [0, 1, 2, 3]

        # Calculate throughput
        throughput = sum(self.stream_bandwidth[s] * self.stream_success_rate[s] / len(streams)
                        for s in streams)

        # Calculate latency (max of used streams)
        latency = max(self.stream_latency[s] * (1 + self.stream_congestion[s])
                     for s in streams)

        # Calculate load imbalance
        if len(streams) > 1:
            loads = [self.stream_congestion[s] for s in streams]
            imbalance = np.std(loads)
        else:
            imbalance = 0

        # Composite reward
        reward = 1.0 * (throughput / 100)  # Normalize throughput
        reward -= 0.5 * (latency / 20)  # Normalize latency
        reward -= 0.3 * imbalance

        # Bonus for completing high-priority tasks quickly
        if self.task_priority > 0.7 and latency < 10:
            reward += 1.0

        # Penalty for using congested streams
        congestion_penalty = sum(self.stream_congestion[s] for s in streams) / len(streams)
        reward -= 0.5 * congestion_penalty

        return reward

    def _update_state(self, action: int):
        """Update environment state after action"""
        # Determine used streams
        if action < 4:
            used_streams = [action]
        elif action < 10:
            stream_map = {4: [0,1], 5: [0,2], 6: [0,3], 7: [1,2], 8: [1,3], 9: [2,3]}
            used_streams = stream_map[action]
        elif action < 14:
            stream_map = {10: [0,1,2], 11: [0,1,3], 12: [0,2,3], 13: [1,2,3]}
            used_streams = stream_map[action]
        else:
            used_streams = [0, 1, 2, 3]

        # Increase congestion on used streams
        for s in used_streams:
            self.stream_congestion[s] = min(1.0, self.stream_congestion[s] + 0.05)

        # Decrease congestion on unused streams
        for s in range(4):
            if s not in used_streams:
                self.stream_congestion[s] = max(0, self.stream_congestion[s] - 0.02)

        # Randomly fluctuate bandwidth and latency
        self.stream_bandwidth += np.random.uniform(-5, 5, 4)
        self.stream_bandwidth = np.clip(self.stream_bandwidth, 10, 200)

        self.stream_latency += np.random.uniform(-1, 1, 4)
        self.stream_latency = np.clip(self.stream_latency, 1, 50)

        # Update time of day
        self.time_of_day = (self.time_of_day + 0.001) % 1.0


def train_dqn(episodes: int = 1000, batch_size: int = 32,
              update_target_freq: int = 10, save_freq: int = 100):
    """Main training loop for DQN agent"""

    # Initialize environment and agent
    env = DWCPEnvironment()
    agent = DQNAgent(state_size=env.state_size, action_size=env.action_size)

    # Training metrics
    episode_rewards = []
    avg_rewards = []

    # Beta schedule for prioritized replay (anneal from 0.4 to 1.0)
    beta_schedule = np.linspace(0.4, 1.0, episodes)

    logger.info(f"Starting DQN training for {episodes} episodes...")

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            # Choose action
            action = agent.act(state)

            # Execute action
            next_state, reward, done = env.step(action)

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Update state
            state = next_state
            total_reward += reward
            steps += 1

            # Train on batch
            if len(agent.memory) > batch_size:
                beta = beta_schedule[episode]
                agent.replay(batch_size, beta)

            if done or steps > 200:
                break

        # Store episode reward
        episode_rewards.append(total_reward)
        agent.reward_history.append(total_reward)

        # Calculate moving average
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_rewards.append(avg_reward)
        else:
            avg_reward = np.mean(episode_rewards)
            avg_rewards.append(avg_reward)

        # Update target network
        if episode % update_target_freq == 0:
            agent.update_target_model()

        # Save model
        if episode % save_freq == 0 and episode > 0:
            model_path = f'models/dqn_checkpoint_{episode}'
            agent.save(model_path)
            logger.info(f"Model saved: {model_path}")

        # Log progress
        if episode % 10 == 0:
            logger.info(f"Episode {episode}/{episodes} - "
                       f"Reward: {total_reward:.2f} - "
                       f"Avg Reward: {avg_reward:.2f} - "
                       f"Epsilon: {agent.epsilon:.3f} - "
                       f"Loss: {agent.loss_history[-1] if agent.loss_history else 0:.4f}")

    # Save final model
    final_path = 'models/dqn_final'
    agent.save(final_path)
    logger.info(f"Training complete! Final model saved: {final_path}")

    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'avg_rewards': avg_rewards,
        'loss_history': agent.loss_history,
        'epsilon_history': agent.epsilon_history
    }

    with open('models/training_history.json', 'w') as f:
        json.dump(history, f)

    return agent, history


def evaluate_agent(agent: DQNAgent, episodes: int = 100):
    """Evaluate trained agent performance"""
    env = DWCPEnvironment()

    # Disable exploration
    original_epsilon = agent.epsilon
    agent.epsilon = 0

    rewards = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)

            state = next_state
            total_reward += reward
            steps += 1

            if done or steps > 200:
                break

        rewards.append(total_reward)

    # Restore epsilon
    agent.epsilon = original_epsilon

    # Calculate statistics
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)

    logger.info(f"Evaluation Results ({episodes} episodes):")
    logger.info(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    logger.info(f"  Min Reward: {min_reward:.2f}")
    logger.info(f"  Max Reward: {max_reward:.2f}")

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'rewards': rewards
    }


def main():
    parser = argparse.ArgumentParser(description='Train DQN for DWCP Task Partitioning')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--update-freq', type=int, default=10, help='Target network update frequency')
    parser.add_argument('--save-freq', type=int, default=100, help='Model save frequency')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--load-model', type=str, help='Load pre-trained model')

    args = parser.parse_args()

    # Create models directory
    os.makedirs('models', exist_ok=True)

    if args.load_model:
        # Load and evaluate existing model
        agent = DQNAgent()
        agent.load(args.load_model)
        logger.info(f"Model loaded from {args.load_model}")

        if args.evaluate:
            evaluate_agent(agent)
    else:
        # Train new model
        agent, history = train_dqn(
            episodes=args.episodes,
            batch_size=args.batch_size,
            update_target_freq=args.update_freq,
            save_freq=args.save_freq
        )

        if args.evaluate:
            evaluate_agent(agent)

        # Plot training curves if matplotlib available
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Plot rewards
            ax1.plot(history['episode_rewards'], alpha=0.3, label='Episode Rewards')
            ax1.plot(history['avg_rewards'], label='Average Rewards (100 episodes)')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.set_title('Training Rewards')
            ax1.legend()
            ax1.grid(True)

            # Plot loss
            ax2.plot(history['loss_history'])
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.set_title('Training Loss')
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig('models/training_curves.png')
            logger.info("Training curves saved to models/training_curves.png")

        except ImportError:
            logger.info("Matplotlib not available, skipping plot generation")


if __name__ == '__main__':
    main()