"""
Network Environment Simulator for Bandwidth Predictor Training
Generates synthetic network data and provides RL environment
"""

import numpy as np
from typing import Dict, Tuple


class NetworkEnvironmentSimulator:
    """Simulates network environment for DDQN training"""

    def __init__(self, network_type: str = 'datacenter'):
        """
        Initialize network simulator

        Args:
            network_type: 'datacenter' or 'internet'
        """
        self.network_type = network_type
        self.state = None
        self.time_step = 0
        self.max_steps = 100

        # Network characteristics
        if network_type == 'datacenter':
            self.base_bandwidth = 10000  # 10 Gbps
            self.latency_range = (0.1, 2.0)  # ms
            self.packet_loss_range = (0.0, 0.01)  # 0-1%
            self.reliability_range = (0.99, 1.0)
        else:  # internet
            self.base_bandwidth = 100  # 100 Mbps
            self.latency_range = (10, 200)  # ms
            self.packet_loss_range = (0.0, 0.05)  # 0-5%
            self.reliability_range = (0.90, 0.99)

    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        self.time_step = 0

        # Generate random initial state [latency, bandwidth, packet_loss, reliability]
        latency = np.random.uniform(*self.latency_range)
        bandwidth = np.random.uniform(
            self.base_bandwidth * 0.8,
            self.base_bandwidth * 1.2
        )
        packet_loss = np.random.uniform(*self.packet_loss_range)
        reliability = np.random.uniform(*self.reliability_range)

        self.state = np.array([latency, bandwidth, packet_loss, reliability])
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return new state, reward, done, info

        Args:
            action: Bandwidth allocation level (0-9)

        Returns:
            next_state, reward, done, info
        """
        self.time_step += 1

        # Map action to bandwidth allocation (0-100%)
        allocation_percent = (action + 1) * 10

        # Current state
        latency, bandwidth, packet_loss, reliability = self.state

        # Calculate reward based on allocation quality
        optimal_allocation = self._calculate_optimal_allocation(
            latency, bandwidth, packet_loss, reliability
        )

        allocation_error = abs(allocation_percent - optimal_allocation) / 100.0
        reward = 1.0 - allocation_error

        # Penalize extreme allocations
        if allocation_percent < 20 or allocation_percent > 90:
            reward -= 0.2

        # Bonus for high reliability allocations
        if reliability > 0.95 and 40 <= allocation_percent <= 70:
            reward += 0.1

        # Update state with some noise
        latency = np.clip(
            latency + np.random.normal(0, latency * 0.1),
            *self.latency_range
        )
        bandwidth = np.clip(
            bandwidth + np.random.normal(0, bandwidth * 0.05),
            self.base_bandwidth * 0.5,
            self.base_bandwidth * 1.5
        )
        packet_loss = np.clip(
            packet_loss + np.random.normal(0, 0.001),
            *self.packet_loss_range
        )
        reliability = np.clip(
            reliability + np.random.normal(0, 0.01),
            *self.reliability_range
        )

        self.state = np.array([latency, bandwidth, packet_loss, reliability])

        done = self.time_step >= self.max_steps

        info = {
            'optimal_allocation': optimal_allocation,
            'actual_allocation': allocation_percent,
            'allocation_error': allocation_error
        }

        return self.state, reward, done, info

    def _calculate_optimal_allocation(self, latency: float, bandwidth: float,
                                     packet_loss: float, reliability: float) -> float:
        """Calculate optimal bandwidth allocation percentage"""
        # Simple heuristic based on network conditions
        base_allocation = 50.0

        # Adjust for latency
        if latency < 1.0:
            base_allocation += 10
        elif latency > 50:
            base_allocation -= 10

        # Adjust for packet loss
        if packet_loss > 0.02:
            base_allocation -= 15
        elif packet_loss < 0.005:
            base_allocation += 5

        # Adjust for reliability
        if reliability > 0.98:
            base_allocation += 10
        elif reliability < 0.95:
            base_allocation -= 10

        return np.clip(base_allocation, 10, 90)


def generate_synthetic_data(num_samples: int = 10000,
                           network_type: str = 'datacenter',
                           sequence_length: int = 10,
                           noise_level: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Generate synthetic network data for LSTM training

    Args:
        num_samples: Number of samples to generate
        network_type: 'datacenter' or 'internet'
        sequence_length: Length of each sequence
        noise_level: Amount of noise to add (0-1)

    Returns:
        Dictionary with train/val/test splits
    """
    # Network characteristics
    if network_type == 'datacenter':
        base_bandwidth = 10000  # 10 Gbps
        latency_range = (0.1, 2.0)
        packet_loss_range = (0.0, 0.01)
        reliability_range = (0.99, 1.0)
    else:  # internet
        base_bandwidth = 100  # 100 Mbps
        latency_range = (10, 200)
        packet_loss_range = (0.0, 0.05)
        reliability_range = (0.90, 0.99)

    # Generate time series data
    sequences = []
    targets = []

    for _ in range(num_samples):
        # Generate a sequence
        sequence = []
        for t in range(sequence_length + 1):  # +1 for target
            # Generate features with temporal correlation
            if t == 0:
                latency = np.random.uniform(*latency_range)
                bandwidth = np.random.uniform(
                    base_bandwidth * 0.8,
                    base_bandwidth * 1.2
                )
                packet_loss = np.random.uniform(*packet_loss_range)
                reliability = np.random.uniform(*reliability_range)
            else:
                # Add temporal correlation
                latency = np.clip(
                    sequence[-1][0] + np.random.normal(0, latency_range[1] * 0.1),
                    *latency_range
                )
                bandwidth = np.clip(
                    sequence[-1][1] + np.random.normal(0, base_bandwidth * 0.05),
                    base_bandwidth * 0.5,
                    base_bandwidth * 1.5
                )
                packet_loss = np.clip(
                    sequence[-1][2] + np.random.normal(0, 0.001),
                    *packet_loss_range
                )
                reliability = np.clip(
                    sequence[-1][3] + np.random.normal(0, 0.01),
                    *reliability_range
                )

            # Add noise
            latency *= (1 + np.random.normal(0, noise_level))
            bandwidth *= (1 + np.random.normal(0, noise_level))
            packet_loss *= (1 + np.random.normal(0, noise_level))
            reliability *= (1 + np.random.normal(0, noise_level * 0.5))

            sequence.append([latency, bandwidth, packet_loss, reliability])

        # Split into input sequence and target
        sequences.append(sequence[:sequence_length])
        targets.append(sequence[sequence_length][1])  # Target is next bandwidth

    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets).reshape(-1, 1)

    # Normalize features
    X_mean = X.mean(axis=(0, 1))
    X_std = X.std(axis=(0, 1))
    X = (X - X_mean) / (X_std + 1e-8)

    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / (y_std + 1e-8)

    # Split into train/val/test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)

    X_train = X[:train_size]
    y_train = y[:train_size]

    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]

    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'normalization': {
            'X_mean': X_mean.tolist(),
            'X_std': X_std.tolist(),
            'y_mean': float(y_mean),
            'y_std': float(y_std)
        }
    }


if __name__ == "__main__":
    print("Generating synthetic network data...")
    data = generate_synthetic_data(num_samples=1000, network_type='datacenter')

    print(f"Training samples: {data['X_train'].shape}")
    print(f"Validation samples: {data['X_val'].shape}")
    print(f"Test samples: {data['X_test'].shape}")

    print("\nTesting environment simulator...")
    env = NetworkEnvironmentSimulator('datacenter')
    state = env.reset()
    print(f"Initial state: {state}")

    for i in range(5):
        action = np.random.randint(0, 10)
        next_state, reward, done, info = env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward:.3f}, State={next_state}")
