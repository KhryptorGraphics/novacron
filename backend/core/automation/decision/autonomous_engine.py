#!/usr/bin/env python3
"""
Autonomous Decision Engine
Implements RL-based autonomous decision-making for infrastructure management
"""

import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DecisionType(Enum):
    """Decision types"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MIGRATE_VM = "migrate_vm"
    REBALANCE_LOAD = "rebalance_load"
    ENABLE_FEATURE = "enable_feature"
    DISABLE_FEATURE = "disable_feature"
    ADJUST_POLICY = "adjust_policy"
    NO_ACTION = "no_action"


@dataclass
class InfrastructureState:
    """Current infrastructure state"""
    cpu_usage: float
    memory_usage: float
    network_latency: float
    active_vms: int
    request_rate: float
    error_rate: float
    cost_per_hour: float
    time_of_day: int  # 0-23
    day_of_week: int  # 0-6


@dataclass
class Decision:
    """Autonomous decision"""
    decision_type: DecisionType
    confidence: float
    parameters: Dict[str, Any]
    expected_impact: Dict[str, float]
    reasoning: str
    timestamp: datetime
    priority: int


@dataclass
class DecisionOutcome:
    """Outcome of a decision"""
    decision_id: str
    success: bool
    actual_impact: Dict[str, float]
    reward: float
    timestamp: datetime


class DQNetwork(nn.Module):
    """Deep Q-Network for decision making"""

    def __init__(self, state_dim: int, action_dim: int):
        super(DQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_dim)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]

        states = torch.FloatTensor([b[0] for b in batch])
        actions = torch.LongTensor([b[1] for b in batch])
        rewards = torch.FloatTensor([b[2] for b in batch])
        next_states = torch.FloatTensor([b[3] for b in batch])
        dones = torch.FloatTensor([b[4] for b in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class AutonomousDecisionEngine:
    """
    Autonomous decision engine using Deep Reinforcement Learning
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # State and action dimensions
        self.state_dim = 9  # Number of state features
        self.action_dim = len(DecisionType)

        # RL parameters
        self.gamma = self.config.get('gamma', 0.99)  # Discount factor
        self.epsilon = self.config.get('epsilon_start', 1.0)  # Exploration rate
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.batch_size = self.config.get('batch_size', 32)
        self.target_update_freq = self.config.get('target_update_freq', 100)

        # Initialize networks
        self.policy_net = DQNetwork(self.state_dim, self.action_dim)
        self.target_net = DQNetwork(self.state_dim, self.action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        # Experience replay
        self.replay_buffer = ReplayBuffer(capacity=self.config.get('buffer_size', 10000))

        # Decision history
        self.decision_history: List[Decision] = []
        self.outcome_history: List[DecisionOutcome] = []

        # Training state
        self.training_step = 0
        self.learning_enabled = self.config.get('learning_enabled', True)

        # Safety constraints
        self.min_confidence = self.config.get('min_confidence', 0.75)
        self.require_approval = self.config.get('require_approval', False)

        self.logger.info("Autonomous decision engine initialized")

    def state_to_vector(self, state: InfrastructureState) -> np.ndarray:
        """Convert infrastructure state to feature vector"""
        return np.array([
            state.cpu_usage / 100.0,  # Normalize to 0-1
            state.memory_usage / 100.0,
            state.network_latency / 1000.0,
            state.active_vms / 100.0,
            state.request_rate / 10000.0,
            state.error_rate,
            state.cost_per_hour / 1000.0,
            state.time_of_day / 24.0,
            state.day_of_week / 7.0
        ], dtype=np.float32)

    def select_action(self, state: InfrastructureState,
                     explore: bool = True) -> Tuple[DecisionType, float]:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current infrastructure state
            explore: Whether to explore (for training)

        Returns:
            Tuple of (action, confidence)
        """

        state_vector = self.state_to_vector(state)

        # Exploration vs exploitation
        if explore and np.random.random() < self.epsilon:
            # Random action (exploration)
            action_idx = np.random.randint(0, self.action_dim)
            confidence = 0.5
        else:
            # Greedy action (exploitation)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()

                # Calculate confidence from Q-values
                q_values_np = q_values.numpy()[0]
                max_q = q_values_np[action_idx]
                mean_q = q_values_np.mean()
                confidence = min(1.0, (max_q - mean_q) / (np.abs(max_q) + 1e-6) + 0.5)

        action = list(DecisionType)[action_idx]
        return action, confidence

    def make_decision(self, state: InfrastructureState) -> Optional[Decision]:
        """
        Make autonomous decision based on current state

        Args:
            state: Current infrastructure state

        Returns:
            Decision or None if no action needed
        """

        # Select action
        action, confidence = self.select_action(state, explore=False)

        # Check confidence threshold
        if confidence < self.min_confidence:
            self.logger.info(f"Decision confidence {confidence:.2f} below threshold {self.min_confidence}")
            return None

        # Create decision with parameters and reasoning
        parameters, expected_impact, reasoning = self._generate_decision_details(
            action, state, confidence
        )

        decision = Decision(
            decision_type=action,
            confidence=confidence,
            parameters=parameters,
            expected_impact=expected_impact,
            reasoning=reasoning,
            timestamp=datetime.now(),
            priority=self._calculate_priority(action, state, expected_impact)
        )

        self.decision_history.append(decision)

        self.logger.info(
            f"Decision made: {action.value} with confidence {confidence:.2f}"
        )

        return decision

    def _generate_decision_details(self, action: DecisionType,
                                   state: InfrastructureState,
                                   confidence: float) -> Tuple[Dict, Dict, str]:
        """Generate detailed decision parameters and reasoning"""

        parameters = {}
        expected_impact = {}
        reasoning = ""

        if action == DecisionType.SCALE_UP:
            parameters = {
                "additional_vms": int(state.active_vms * 0.2) + 1,
                "vm_type": "standard",
                "region": "auto"
            }
            expected_impact = {
                "cpu_reduction": 15.0,
                "latency_reduction": 10.0,
                "cost_increase": 20.0
            }
            reasoning = f"High CPU usage ({state.cpu_usage:.1f}%) and request rate ({state.request_rate:.0f}/s) indicate need for scaling"

        elif action == DecisionType.SCALE_DOWN:
            parameters = {
                "remove_vms": max(1, int(state.active_vms * 0.15)),
                "drain_timeout": 300
            }
            expected_impact = {
                "cost_reduction": 15.0,
                "cpu_increase": 5.0
            }
            reasoning = f"Low resource utilization (CPU: {state.cpu_usage:.1f}%, Memory: {state.memory_usage:.1f}%) allows for cost optimization"

        elif action == DecisionType.MIGRATE_VM:
            parameters = {
                "source_host": "auto-detect",
                "target_host": "auto-select",
                "migration_type": "live"
            }
            expected_impact = {
                "balance_improvement": 20.0,
                "latency_reduction": 5.0
            }
            reasoning = "Resource imbalance detected, migration will improve distribution"

        elif action == DecisionType.REBALANCE_LOAD:
            parameters = {
                "strategy": "weighted_round_robin",
                "rebalance_threshold": 0.7
            }
            expected_impact = {
                "latency_reduction": 15.0,
                "error_rate_reduction": 10.0
            }
            reasoning = f"High latency ({state.network_latency:.1f}ms) and error rate ({state.error_rate:.3f}) require load rebalancing"

        elif action == DecisionType.ADJUST_POLICY:
            parameters = {
                "policy_type": "auto_scaling",
                "adjustment": "more_aggressive"
            }
            expected_impact = {
                "responsiveness_improvement": 25.0
            }
            reasoning = "Request patterns suggest need for policy adjustment"

        else:  # NO_ACTION
            reasoning = "Current state is within optimal parameters"

        return parameters, expected_impact, reasoning

    def _calculate_priority(self, action: DecisionType, state: InfrastructureState,
                           expected_impact: Dict[str, float]) -> int:
        """Calculate decision priority (1-10, 10 highest)"""

        priority = 5  # Default medium priority

        # Increase priority for critical situations
        if state.error_rate > 0.05:  # 5% error rate
            priority += 3
        if state.cpu_usage > 90:
            priority += 2
        if state.network_latency > 500:  # 500ms
            priority += 2

        # Adjust based on expected impact
        if expected_impact.get("error_rate_reduction", 0) > 20:
            priority += 1
        if expected_impact.get("cost_reduction", 0) > 30:
            priority += 1

        return min(10, max(1, priority))

    def record_outcome(self, decision: Decision, outcome: DecisionOutcome):
        """
        Record decision outcome for learning

        Args:
            decision: Original decision
            outcome: Observed outcome
        """

        self.outcome_history.append(outcome)

        if not self.learning_enabled:
            return

        # Calculate reward based on outcome
        reward = self._calculate_reward(decision, outcome)

        # Store experience in replay buffer
        # In real implementation, would track state transitions
        # Here's a simplified version
        state_vector = np.random.randn(self.state_dim)  # Placeholder
        next_state_vector = np.random.randn(self.state_dim)  # Placeholder
        action_idx = list(DecisionType).index(decision.decision_type)

        self.replay_buffer.push(
            state_vector,
            action_idx,
            reward,
            next_state_vector,
            outcome.success
        )

        # Train if buffer has enough samples
        if len(self.replay_buffer) >= self.batch_size:
            self.train_step()

        self.logger.info(
            f"Recorded outcome: success={outcome.success}, reward={reward:.2f}"
        )

    def _calculate_reward(self, decision: Decision, outcome: DecisionOutcome) -> float:
        """Calculate reward from decision outcome"""

        if not outcome.success:
            return -10.0  # Penalty for failed actions

        reward = 0.0

        # Reward for positive impacts
        actual = outcome.actual_impact
        expected = decision.expected_impact

        for metric, expected_value in expected.items():
            actual_value = actual.get(metric, 0)

            if "reduction" in metric or "improvement" in metric:
                # Higher is better
                reward += actual_value * 0.5
            elif "increase" in metric and "cost" in metric:
                # Cost increase is bad
                reward -= actual_value * 0.3

        # Bonus for high confidence correct decisions
        if outcome.success and decision.confidence > 0.9:
            reward += 5.0

        return reward

    def train_step(self):
        """Perform one training step"""

        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Compute Q values
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = self.criterion(q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.training_step += 1

        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.training_step % 100 == 0:
            self.logger.info(
                f"Training step {self.training_step}, "
                f"loss: {loss.item():.4f}, epsilon: {self.epsilon:.4f}"
            )

    def export_metrics(self) -> Dict[str, Any]:
        """Export engine metrics"""

        total_decisions = len(self.decision_history)
        total_outcomes = len(self.outcome_history)

        successful_outcomes = sum(1 for o in self.outcome_history if o.success)
        success_rate = successful_outcomes / max(1, total_outcomes)

        avg_confidence = 0.0
        if total_decisions > 0:
            avg_confidence = sum(d.confidence for d in self.decision_history) / total_decisions

        avg_reward = 0.0
        if total_outcomes > 0:
            avg_reward = sum(o.reward for o in self.outcome_history) / total_outcomes

        decision_distribution = {}
        for d in self.decision_history:
            decision_distribution[d.decision_type.value] = \
                decision_distribution.get(d.decision_type.value, 0) + 1

        return {
            "total_decisions": total_decisions,
            "total_outcomes": total_outcomes,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "average_reward": avg_reward,
            "training_steps": self.training_step,
            "epsilon": self.epsilon,
            "buffer_size": len(self.replay_buffer),
            "decision_distribution": decision_distribution,
            "learning_enabled": self.learning_enabled
        }

    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon
        }, path)
        self.logger.info(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
        self.logger.info(f"Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize decision engine
    engine = AutonomousDecisionEngine({
        'learning_enabled': True,
        'min_confidence': 0.70,
        'epsilon_start': 0.5
    })

    # Simulate decision making over time
    for i in range(500):
        # Simulate infrastructure state
        state = InfrastructureState(
            cpu_usage=60 + np.random.randn() * 20,
            memory_usage=50 + np.random.randn() * 15,
            network_latency=100 + np.random.randn() * 30,
            active_vms=10 + int(np.random.randn() * 3),
            request_rate=1000 + np.random.randn() * 200,
            error_rate=0.01 + np.random.randn() * 0.005,
            cost_per_hour=50 + np.random.randn() * 10,
            time_of_day=i % 24,
            day_of_week=(i // 24) % 7
        )

        # Make decision
        decision = engine.make_decision(state)

        if decision:
            # Simulate outcome
            success = np.random.random() > 0.2  # 80% success rate
            actual_impact = {
                k: v * (0.8 + np.random.random() * 0.4)
                for k, v in decision.expected_impact.items()
            }

            outcome = DecisionOutcome(
                decision_id=f"decision-{i}",
                success=success,
                actual_impact=actual_impact,
                reward=0.0,  # Will be calculated
                timestamp=datetime.now()
            )

            # Record outcome for learning
            engine.record_outcome(decision, outcome)

    # Export and print metrics
    metrics = engine.export_metrics()
    print("\nAutonomous Decision Engine Metrics:")
    print(json.dumps(metrics, indent=2))

    print(f"\nDecision Distribution:")
    for decision_type, count in metrics['decision_distribution'].items():
        print(f"  {decision_type}: {count}")
