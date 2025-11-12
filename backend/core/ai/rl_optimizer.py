"""
Reinforcement Learning Optimizer for DWCP v3
Multi-agent RL with Proximal Policy Optimization for distributed optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from collections import deque, namedtuple
import gym
from gym import spaces
import random
from datetime import datetime
import json
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Experience tuple for replay buffer
Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'info'])

@dataclass
class RLConfig:
    """Configuration for RL optimizer"""
    n_agents: int = 4
    state_dim: int = 128
    action_dim: int = 32
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    ppo_epochs: int = 10
    batch_size: int = 64
    buffer_size: int = 10000
    update_frequency: int = 2048
    self_play_enabled: bool = True
    exploration_noise: float = 0.1

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int,
                 continuous: bool = False):
        super().__init__()
        self.continuous = continuous

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # For continuous actions
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities and value"""
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)

        return action_logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy"""
        action_logits, value = self.forward(state)

        if self.continuous:
            mean = action_logits
            std = self.log_std.exp()
            dist = Normal(mean, std)
            action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            dist = Categorical(logits=action_logits)
            action = action_logits.argmax(dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor):
        """Evaluate actions for PPO update"""
        action_logits, values = self.forward(states)

        if self.continuous:
            mean = action_logits
            std = self.log_std.exp()
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        else:
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

        return values.squeeze(-1), log_probs, entropy

class ReplayBuffer:
    """Experience replay buffer for RL training"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Add experience to buffer"""
        self.buffer.append(Experience(*args))

    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class PPOTrainer:
    """Proximal Policy Optimization trainer"""

    def __init__(self, config: RLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy = ActorCritic(
            config.state_dim,
            config.action_dim,
            config.hidden_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate
        )

        self.memory = []
        self.training_stats = []

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        last_value = values[-1]

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]

        returns = advantages + values
        return advantages, returns

    def update(self, states: torch.Tensor, actions: torch.Tensor,
               old_log_probs: torch.Tensor, advantages: torch.Tensor,
               returns: torch.Tensor):
        """PPO update step"""
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0

        for _ in range(self.config.ppo_epochs):
            # Evaluate current policy
            values, log_probs, entropy = self.policy.evaluate_actions(states, actions)

            # Calculate ratio for PPO
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon,
                              1 + self.config.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss
            value_loss = F.mse_loss(values, returns)

            # Total loss
            loss = (policy_loss +
                   self.config.value_loss_coef * value_loss -
                   self.config.entropy_coef * entropy.mean())

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()

        # Store training statistics
        self.training_stats.append({
            'loss': total_loss / self.config.ppo_epochs,
            'policy_loss': total_policy_loss / self.config.ppo_epochs,
            'value_loss': total_value_loss / self.config.ppo_epochs,
            'entropy': total_entropy / self.config.ppo_epochs
        })

    def train_step(self, experiences: List[Experience]):
        """Train on batch of experiences"""
        states = torch.FloatTensor([e.state for e in experiences]).to(self.device)
        actions = torch.LongTensor([e.action for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e.done for e in experiences]).to(self.device)

        # Get values and log probs
        with torch.no_grad():
            _, old_log_probs, old_values = self.policy.get_action(states)
            _, _, next_values = self.policy.get_action(next_states)

        # Compute advantages
        advantages, returns = self.compute_gae(rewards, old_values, dones)

        # PPO update
        self.update(states, actions, old_log_probs, advantages, returns)

class MultiAgentEnvironment(gym.Env):
    """Multi-agent environment for distributed optimization"""

    def __init__(self, n_agents: int, state_dim: int, action_dim: int):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define action and observation spaces
        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,), dtype=np.float32
        )

        self.current_step = 0
        self.max_steps = 1000
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = 0
        self.agent_states = np.random.randn(self.n_agents, self.state_dim)
        self.global_state = self._compute_global_state()
        return self.global_state

    def _compute_global_state(self) -> np.ndarray:
        """Compute global state from agent states"""
        return np.concatenate([
            self.agent_states.mean(axis=0),
            self.agent_states.std(axis=0),
            np.array([self.current_step / self.max_steps])
        ])[:self.state_dim]

    def step(self, actions: Union[int, List[int]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute actions in environment"""
        if isinstance(actions, int):
            actions = [actions]

        rewards = []
        for i, action in enumerate(actions):
            # Simulate resource allocation optimization
            reward = self._compute_reward(i, action)
            rewards.append(reward)

            # Update agent state based on action
            self._update_agent_state(i, action)

        self.current_step += 1
        self.global_state = self._compute_global_state()

        done = self.current_step >= self.max_steps
        info = {
            'agent_rewards': rewards,
            'step': self.current_step
        }

        return self.global_state, np.mean(rewards), done, info

    def _compute_reward(self, agent_id: int, action: int) -> float:
        """Compute reward for agent action"""
        # Reward based on resource utilization and efficiency
        base_reward = -0.1  # Small negative reward to encourage efficiency

        # Reward for balanced resource allocation
        utilization = self.agent_states[agent_id].mean()
        if 0.6 <= utilization <= 0.8:  # Optimal utilization range
            base_reward += 1.0

        # Penalty for extreme actions
        if action < 5 or action > self.action_dim - 5:
            base_reward -= 0.5

        # Cooperation bonus
        agent_variance = self.agent_states.var(axis=0).mean()
        if agent_variance < 0.1:  # Agents are well coordinated
            base_reward += 0.5

        return base_reward

    def _update_agent_state(self, agent_id: int, action: int):
        """Update agent state based on action"""
        # Map action to state change
        action_effect = (action - self.action_dim / 2) / self.action_dim
        self.agent_states[agent_id] += action_effect * 0.1

        # Add some noise for realism
        self.agent_states[agent_id] += np.random.randn(self.state_dim) * 0.01

        # Clip to reasonable range
        self.agent_states[agent_id] = np.clip(self.agent_states[agent_id], -2, 2)

class SelfPlayManager:
    """Manager for self-play training"""

    def __init__(self, config: RLConfig):
        self.config = config
        self.agent_pool = []
        self.current_generation = 0
        self.match_history = []

    def add_agent(self, agent: ActorCritic):
        """Add agent to pool"""
        self.agent_pool.append({
            'agent': agent,
            'generation': self.current_generation,
            'elo_rating': 1500,
            'games_played': 0
        })

    def select_opponent(self) -> ActorCritic:
        """Select opponent for self-play"""
        if len(self.agent_pool) < 2:
            return None

        # Select based on Elo rating with some randomness
        weights = np.array([agent['elo_rating'] for agent in self.agent_pool])
        weights = np.exp(weights / 100)  # Softmax-like selection
        weights /= weights.sum()

        opponent_idx = np.random.choice(len(self.agent_pool), p=weights)
        return self.agent_pool[opponent_idx]['agent']

    def update_ratings(self, agent_idx: int, opponent_idx: int, result: float):
        """Update Elo ratings after match"""
        K = 32  # Elo K-factor

        agent_rating = self.agent_pool[agent_idx]['elo_rating']
        opponent_rating = self.agent_pool[opponent_idx]['elo_rating']

        # Expected scores
        expected_agent = 1 / (1 + 10 ** ((opponent_rating - agent_rating) / 400))
        expected_opponent = 1 - expected_agent

        # Update ratings
        self.agent_pool[agent_idx]['elo_rating'] += K * (result - expected_agent)
        self.agent_pool[opponent_idx]['elo_rating'] += K * ((1 - result) - expected_opponent)

        # Update game counts
        self.agent_pool[agent_idx]['games_played'] += 1
        self.agent_pool[opponent_idx]['games_played'] += 1

    def run_match(self, agent1: ActorCritic, agent2: ActorCritic,
                  env: MultiAgentEnvironment) -> float:
        """Run match between two agents"""
        state = env.reset()
        total_reward_1 = 0
        total_reward_2 = 0

        for _ in range(env.max_steps):
            # Get actions from both agents
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action1, _, _ = agent1.get_action(state_tensor, deterministic=True)
                action2, _, _ = agent2.get_action(state_tensor, deterministic=True)

            # Execute actions
            actions = [action1.item(), action2.item()]
            next_state, reward, done, info = env.step(actions)

            total_reward_1 += info['agent_rewards'][0]
            total_reward_2 += info['agent_rewards'][1]

            state = next_state
            if done:
                break

        # Determine winner (1 for agent1 win, 0 for agent2 win, 0.5 for draw)
        if total_reward_1 > total_reward_2:
            return 1.0
        elif total_reward_2 > total_reward_1:
            return 0.0
        else:
            return 0.5

class ContinuousOptimizer:
    """Continuous parameter optimization using RL"""

    def __init__(self, param_ranges: Dict[str, Tuple[float, float]],
                 objective_func: callable):
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.n_params = len(self.param_names)
        self.objective_func = objective_func

        # Initialize continuous actor-critic
        self.policy = ActorCritic(
            state_dim=self.n_params * 2,  # Current params + gradients
            action_dim=self.n_params,
            hidden_dim=256,
            continuous=True
        )

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-4)
        self.best_params = None
        self.best_score = -np.inf

    def normalize_params(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [-1, 1] range"""
        normalized = np.zeros_like(params)
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[name]
            normalized[i] = 2 * (params[i] - min_val) / (max_val - min_val) - 1
        return normalized

    def denormalize_params(self, normalized: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [-1, 1] range"""
        params = np.zeros_like(normalized)
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[name]
            params[i] = min_val + (normalized[i] + 1) * (max_val - min_val) / 2
        return params

    def optimize(self, n_iterations: int = 1000) -> Dict[str, float]:
        """Optimize parameters using RL"""
        current_params = np.random.randn(self.n_params)
        gradients = np.zeros(self.n_params)

        for iteration in range(n_iterations):
            # Create state
            state = np.concatenate([
                self.normalize_params(current_params),
                gradients
            ])

            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Get action from policy
            action, log_prob, value = self.policy.get_action(state_tensor)
            action = action.squeeze().cpu().numpy()

            # Update parameters
            new_params = current_params + action * 0.1
            new_params = np.clip(new_params, -1, 1)

            # Evaluate objective
            denorm_params = self.denormalize_params(new_params)
            score = self.objective_func(
                {name: denorm_params[i] for i, name in enumerate(self.param_names)}
            )

            # Compute reward
            reward = score - self.objective_func(
                {name: self.denormalize_params(current_params)[i]
                 for i, name in enumerate(self.param_names)}
            )

            # Update gradients estimate
            gradients = 0.9 * gradients + 0.1 * (new_params - current_params)

            # Store if best
            if score > self.best_score:
                self.best_score = score
                self.best_params = denorm_params

            # Update policy
            if iteration % 32 == 31:
                self._update_policy()

            current_params = new_params

            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}, Best score: {self.best_score:.4f}")

        return {name: self.best_params[i] for i, name in enumerate(self.param_names)}

    def _update_policy(self):
        """Update policy using collected experiences"""
        # Simplified update for demonstration
        pass

class GameTheoreticAllocator:
    """Game-theoretic resource allocation"""

    def __init__(self, n_players: int, resources: int):
        self.n_players = n_players
        self.resources = resources
        self.allocation_history = []

    def compute_nash_equilibrium(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """Compute Nash equilibrium for resource allocation"""
        # Simplified Nash equilibrium computation
        # In practice, would use more sophisticated algorithms

        n_strategies = payoff_matrix.shape[1]
        equilibrium = np.ones(n_strategies) / n_strategies

        for _ in range(100):  # Iterative refinement
            best_responses = payoff_matrix @ equilibrium
            best_strategy = np.argmax(best_responses)
            equilibrium *= 0.9
            equilibrium[best_strategy] += 0.1
            equilibrium /= equilibrium.sum()

        return equilibrium

    def allocate_resources(self, demands: List[float]) -> List[float]:
        """Allocate resources using game theory"""
        total_demand = sum(demands)

        if total_demand <= self.resources:
            # All demands can be satisfied
            return demands

        # Create payoff matrix
        n = len(demands)
        payoff_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    payoff_matrix[i, j] = demands[i] / total_demand
                else:
                    payoff_matrix[i, j] = -demands[j] / (total_demand * (n - 1))

        # Find equilibrium
        equilibrium = self.compute_nash_equilibrium(payoff_matrix)

        # Allocate based on equilibrium
        allocation = equilibrium * self.resources

        self.allocation_history.append({
            'demands': demands,
            'allocation': allocation.tolist(),
            'efficiency': min(1.0, self.resources / total_demand)
        })

        return allocation.tolist()

class RLOptimizer:
    """Main RL optimizer orchestrator"""

    def __init__(self, config: RLConfig = None):
        self.config = config or RLConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize components
        self.ppo_trainer = PPOTrainer(self.config)
        self.environment = MultiAgentEnvironment(
            self.config.n_agents,
            self.config.state_dim,
            self.config.action_dim
        )
        self.self_play_manager = SelfPlayManager(self.config) if config.self_play_enabled else None
        self.replay_buffer = ReplayBuffer(self.config.buffer_size)

        # Game theoretic allocator
        self.allocator = GameTheoreticAllocator(self.config.n_agents, 100)

        # Metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'optimization_improvements': [],
            'resource_utilization': [],
            'convergence_rate': []
        }

        self.model_path = Path("/home/kp/novacron/models/rl_optimizer")
        self.model_path.mkdir(parents=True, exist_ok=True)

    def train(self, n_episodes: int = 1000):
        """Train the RL optimizer"""
        logger.info(f"Starting RL training for {n_episodes} episodes")

        for episode in range(n_episodes):
            state = self.environment.reset()
            episode_reward = 0
            experiences = []

            for step in range(self.environment.max_steps):
                # Get action from policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob, value = self.ppo_trainer.policy.get_action(state_tensor)
                action_item = action.item() if isinstance(action, torch.Tensor) else action

                # Step environment
                next_state, reward, done, info = self.environment.step(action_item)

                # Store experience
                experiences.append(Experience(
                    state, action_item, reward, next_state, done, info
                ))
                self.replay_buffer.push(state, action_item, reward, next_state, done, info)

                episode_reward += reward
                state = next_state

                if done:
                    break

            # Update policy
            if len(self.replay_buffer) >= self.config.batch_size:
                batch = self.replay_buffer.sample(self.config.batch_size)
                self.ppo_trainer.train_step(batch)

            # Self-play if enabled
            if self.self_play_manager and episode % 100 == 0:
                self._run_self_play()

            # Track metrics
            self.metrics['episode_rewards'].append(episode_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(self.metrics['episode_rewards'][-100:])
                logger.info(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        self.save_model()

    def _run_self_play(self):
        """Run self-play training round"""
        if len(self.self_play_manager.agent_pool) == 0:
            self.self_play_manager.add_agent(self.ppo_trainer.policy)

        opponent = self.self_play_manager.select_opponent()
        if opponent:
            result = self.self_play_manager.run_match(
                self.ppo_trainer.policy,
                opponent,
                self.environment
            )
            logger.info(f"Self-play result: {result}")

    def optimize_parameters(self, current_params: Dict[str, float],
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Optimize system parameters using trained policy"""
        # Convert parameters to state
        state = self._params_to_state(current_params)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # Get optimal action
        with torch.no_grad():
            action, _, _ = self.ppo_trainer.policy.get_action(
                state_tensor, deterministic=True
            )

        # Convert action to parameter updates
        optimized_params = self._apply_action_to_params(
            current_params, action, constraints
        )

        # Track improvement
        improvement = self._calculate_improvement(current_params, optimized_params)
        self.metrics['optimization_improvements'].append(improvement)

        return optimized_params

    def _params_to_state(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameters to state vector"""
        # Normalize and concatenate parameters
        state_vector = []
        for key, value in params.items():
            normalized = (value - 50) / 50  # Assuming params in [0, 100] range
            state_vector.append(normalized)

        # Pad or truncate to match state dimension
        state_vector = np.array(state_vector)
        if len(state_vector) < self.config.state_dim:
            state_vector = np.pad(state_vector,
                                 (0, self.config.state_dim - len(state_vector)))
        else:
            state_vector = state_vector[:self.config.state_dim]

        return state_vector

    def _apply_action_to_params(self, params: Dict[str, float],
                                action: torch.Tensor,
                                constraints: Optional[Dict[str, Any]]) -> Dict[str, float]:
        """Apply RL action to parameters"""
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action

        # Map action to parameter changes
        param_keys = list(params.keys())
        optimized = params.copy()

        for i, key in enumerate(param_keys):
            if i < len(action_np):
                # Scale action to parameter range
                change = action_np if isinstance(action_np, (int, float)) else action_np[i]
                change = (change - self.config.action_dim / 2) / self.config.action_dim * 10

                new_value = params[key] + change

                # Apply constraints if provided
                if constraints and key in constraints:
                    min_val = constraints[key].get('min', 0)
                    max_val = constraints[key].get('max', 100)
                    new_value = np.clip(new_value, min_val, max_val)

                optimized[key] = new_value

        return optimized

    def _calculate_improvement(self, original: Dict[str, float],
                              optimized: Dict[str, float]) -> float:
        """Calculate improvement metric"""
        # Simplified: calculate mean relative change
        improvements = []
        for key in original:
            if key in optimized and original[key] != 0:
                improvement = (optimized[key] - original[key]) / original[key]
                improvements.append(improvement)

        return np.mean(improvements) if improvements else 0.0

    def allocate_resources(self, resource_demands: List[float]) -> List[float]:
        """Allocate resources using game-theoretic approach"""
        allocation = self.allocator.allocate_resources(resource_demands)

        # Track utilization
        total_allocated = sum(allocation)
        total_demanded = sum(resource_demands)
        utilization = min(1.0, total_allocated / total_demanded)
        self.metrics['resource_utilization'].append(utilization)

        return allocation

    def save_model(self):
        """Save trained model"""
        torch.save({
            'policy_state_dict': self.ppo_trainer.policy.state_dict(),
            'optimizer_state_dict': self.ppo_trainer.optimizer.state_dict(),
            'config': self.config.__dict__,
            'metrics': self.metrics
        }, self.model_path / "rl_optimizer.pt")

        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load trained model"""
        checkpoint = torch.load(self.model_path / "rl_optimizer.pt")
        self.ppo_trainer.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.ppo_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.metrics = checkpoint['metrics']

        logger.info("Model loaded successfully")

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.metrics['episode_rewards']:
            return {}

        return {
            'average_reward': np.mean(self.metrics['episode_rewards'][-100:]),
            'best_reward': max(self.metrics['episode_rewards']),
            'improvement_rate': np.mean(self.metrics['optimization_improvements'][-100:])
                               if self.metrics['optimization_improvements'] else 0,
            'resource_utilization': np.mean(self.metrics['resource_utilization'][-100:])
                                   if self.metrics['resource_utilization'] else 0,
            'total_episodes': len(self.metrics['episode_rewards']),
            'convergence_trend': self._calculate_convergence_trend()
        }

    def _calculate_convergence_trend(self) -> str:
        """Calculate convergence trend"""
        if len(self.metrics['episode_rewards']) < 200:
            return "insufficient_data"

        recent = np.mean(self.metrics['episode_rewards'][-50:])
        older = np.mean(self.metrics['episode_rewards'][-100:-50])

        if recent > older * 1.05:
            return "improving"
        elif recent < older * 0.95:
            return "degrading"
        else:
            return "converged"

if __name__ == "__main__":
    # Example usage
    config = RLConfig(
        n_agents=4,
        state_dim=128,
        action_dim=32,
        self_play_enabled=True
    )

    optimizer = RLOptimizer(config)

    # Train the optimizer
    optimizer.train(n_episodes=100)

    # Example optimization
    current_params = {
        'cpu_allocation': 50.0,
        'memory_allocation': 60.0,
        'network_bandwidth': 40.0,
        'disk_io_priority': 30.0
    }

    optimized = optimizer.optimize_parameters(current_params)
    print(f"Optimized parameters: {optimized}")

    # Example resource allocation
    demands = [25.0, 30.0, 35.0, 40.0]
    allocation = optimizer.allocate_resources(demands)
    print(f"Resource allocation: {allocation}")

    # Get stats
    stats = optimizer.get_optimization_stats()
    print(f"Optimization stats: {stats}")