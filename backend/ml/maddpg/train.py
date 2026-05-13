"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) Training
Implements centralized training with decentralized execution
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque
import random
import os
import json
import math
from environment import DistributedResourceEnv


def set_random_seeds(seed: int):
    """Seed Python, NumPy, and Torch RNGs for reproducible training runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Actor(nn.Module):
    """Actor network for MADDPG (policy network)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256,
                 action_prior: float = 0.3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.action_prior = min(max(action_prior, 1e-3), 1.0 - 1e-3)

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self._initialize_action_prior()

    def _initialize_action_prior(self):
        prior_logit = math.log(self.action_prior / (1.0 - self.action_prior))
        nn.init.zeros_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, prior_logit)

    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        # Sigmoid for [0,1] action space
        action = torch.sigmoid(self.fc3(x))
        return action


class Critic(nn.Module):
    """Critic network for MADDPG (Q-value network)"""

    def __init__(self, total_state_dim: int, total_action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        # Centralized critic sees all states and actions
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, states, actions):
        # Concatenate all states and actions
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        return q_value


class OUNoise:
    """Ornstein-Uhlenbeck noise for exploration"""

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * self.mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


class ReplayBuffer:
    """Experience replay buffer for MADDPG"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, states, actions, rewards, next_states, dones):
        """Store transition"""
        self.buffer.append((states, actions, rewards, next_states, dones))

    def sample(self, batch_size: int):
        """Sample batch of transitions"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = [torch.FloatTensor(np.array(s)) for s in zip(*states)]
        actions = [torch.FloatTensor(np.array(a)) for a in zip(*actions)]
        rewards = [torch.FloatTensor(r).unsqueeze(1) for r in zip(*rewards)]
        next_states = [torch.FloatTensor(np.array(s)) for s in zip(*next_states)]
        dones = [torch.FloatTensor(d).unsqueeze(1) for d in zip(*dones)]

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class MADDPGAgent:
    """Single agent in MADDPG framework"""

    def __init__(self,
                 agent_id: int,
                 num_agents: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 action_prior: float = 0.3):

        self.agent_id = agent_id
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.action_prior = action_prior

        # Actor networks (decentralized - only sees own state)
        self.actor = Actor(state_dim, action_dim, hidden_dim, action_prior=action_prior)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, action_prior=action_prior)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic networks (centralized - sees all states and actions)
        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents
        self.critic = Critic(total_state_dim, total_action_dim, hidden_dim)
        self.critic_target = Critic(total_state_dim, total_action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Exploration noise
        self.noise = OUNoise(action_dim)

    def select_action(self, state, add_noise=True, noise_scale=1.0):
        """Select action using actor network"""
        state = torch.FloatTensor(state).unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze(0).cpu().numpy()
        self.actor.train()

        if add_noise:
            noise = self.noise.sample() * noise_scale
            action = np.clip(action + noise, 0.0, 1.0)

        return action

    def update(self,
               agents: List['MADDPGAgent'],
               states: List[torch.Tensor],
               actions: List[torch.Tensor],
               rewards: List[torch.Tensor],
               next_states: List[torch.Tensor],
               dones: List[torch.Tensor]):
        """Update actor and critic networks"""

        # Concatenate all states and actions for centralized critic
        all_states = torch.cat(states, dim=1)
        all_actions = torch.cat(actions, dim=1)
        all_next_states = torch.cat(next_states, dim=1)

        # Update Critic
        # Compute target Q-value
        with torch.no_grad():
            # Get next actions from all target actors
            next_actions = [agent.actor_target(next_states[i])
                           for i, agent in enumerate(agents)]
            all_next_actions = torch.cat(next_actions, dim=1)

            # Compute target Q-value
            target_q = self.critic_target(all_next_states, all_next_actions)
            target_q = rewards[self.agent_id] + self.gamma * target_q * (1 - dones[self.agent_id])

        # Compute current Q-value
        current_q = self.critic(all_states, all_actions)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_q)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update Actor
        # Compute actions from current policy for all agents
        current_actions = []
        for i, agent in enumerate(agents):
            if i == self.agent_id:
                current_actions.append(self.actor(states[i]))
            else:
                current_actions.append(actions[i].detach())

        all_current_actions = torch.cat(current_actions, dim=1)

        # Actor loss (maximize Q-value)
        actor_loss = -self.critic(all_states, all_current_actions).mean()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'q_value': current_q.mean().item(),
            'target_update': True,
        }

    def soft_update(self):
        """Soft update of target networks"""
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str):
        """Save agent models"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        """Load agent models"""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])


class MATD3Agent(MADDPGAgent):
    """Multi-agent TD3 agent with twin critics and delayed policy updates."""

    def __init__(self,
                 agent_id: int,
                 num_agents: int,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 action_prior: float = 0.3):
        super().__init__(
            agent_id=agent_id,
            num_agents=num_agents,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            action_prior=action_prior
        )

        total_state_dim = state_dim * num_agents
        total_action_dim = action_dim * num_agents
        self.critic2 = Critic(total_state_dim, total_action_dim, hidden_dim)
        self.critic2_target = Critic(total_state_dim, total_action_dim, hidden_dim)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = max(1, policy_delay)
        self.update_count = 0

    def update(self,
               agents: List['MADDPGAgent'],
               states: List[torch.Tensor],
               actions: List[torch.Tensor],
               rewards: List[torch.Tensor],
               next_states: List[torch.Tensor],
               dones: List[torch.Tensor]):
        """Update twin critics every step and actor on the delayed TD3 cadence."""

        self.update_count += 1

        all_states = torch.cat(states, dim=1)
        all_actions = torch.cat(actions, dim=1)
        all_next_states = torch.cat(next_states, dim=1)

        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(agents):
                next_action = agent.actor_target(next_states[i])
                noise = torch.randn_like(next_action) * self.policy_noise
                noise = noise.clamp(-self.noise_clip, self.noise_clip)
                next_actions.append((next_action + noise).clamp(0.0, 1.0))

            all_next_actions = torch.cat(next_actions, dim=1)
            target_q1 = self.critic_target(all_next_states, all_next_actions)
            target_q2 = self.critic2_target(all_next_states, all_next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards[self.agent_id] + self.gamma * target_q * (1 - dones[self.agent_id])

        current_q1 = self.critic(all_states, all_actions)
        current_q2 = self.critic2(all_states, all_actions)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)
        self.critic2_optimizer.step()

        actor_updated = self.update_count % self.policy_delay == 0
        actor_loss_value = 0.0
        if actor_updated:
            current_actions = []
            for i, agent in enumerate(agents):
                if i == self.agent_id:
                    current_actions.append(self.actor(states[i]))
                else:
                    current_actions.append(actions[i].detach())

            all_current_actions = torch.cat(current_actions, dim=1)
            actor_loss = -self.critic(all_states, all_current_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            actor_loss_value = actor_loss.item()

        return {
            'critic_loss': critic1_loss.item(),
            'critic2_loss': critic2_loss.item(),
            'actor_loss': actor_loss_value,
            'q_value': current_q1.mean().item(),
            'target_update': actor_updated,
        }

    def soft_update(self):
        """Soft update actor and both critic target networks."""
        super().soft_update()

        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str):
        """Save MATD3 agent models."""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'critic2_target': self.critic2_target.state_dict(),
        }, path)

    def load(self, path: str):
        """Load MATD3 agent models, accepting MADDPG checkpoints for actor-only evaluation."""
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint.get('actor_target', checkpoint['actor']))

        if 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint.get('critic_target', checkpoint['critic']))
        if 'critic2' in checkpoint:
            self.critic2.load_state_dict(checkpoint['critic2'])
            self.critic2_target.load_state_dict(checkpoint.get('critic2_target', checkpoint['critic2']))


class MADDPGTrainer:
    """MADDPG multi-agent trainer"""

    def __init__(self,
                 env: DistributedResourceEnv,
                 hidden_dim: int = 256,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 buffer_capacity: int = 100000,
                 batch_size: int = 256,
                 action_prior: float = 0.3,
                 agent_cls=MADDPGAgent,
                 agent_kwargs=None,
                 seed: int = None):

        self.env = env
        self.seed = seed
        if seed is not None:
            set_random_seeds(seed)

        self.num_agents = env.num_agents
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.hidden_dim = hidden_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.action_prior = action_prior
        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs or {}

        # Create agents
        self.agents = [
            self.agent_cls(
                agent_id=i,
                num_agents=self.num_agents,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau,
                action_prior=action_prior,
                **self.agent_kwargs
            )
            for i in range(self.num_agents)
        ]

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        # Training metrics
        self.episode_rewards = []
        self.episode_sla_violations = []
        self.episode_completion_rates = []

    def train(self,
              num_episodes: int = 10000,
              max_steps: int = 1000,
              warmup_episodes: int = 100,
              update_interval: int = 1,
              save_interval: int = 100,
              log_interval: int = 10,
              save_dir: str = './models'):
        """Train MADDPG agents"""

        os.makedirs(save_dir, exist_ok=True)

        algorithm_name = self.agent_cls.__name__.replace("Agent", "")
        print(f"Starting {algorithm_name} training for {num_episodes} episodes...")
        print(f"Agents: {self.num_agents}, State dim: {self.state_dim}, Action dim: {self.action_dim}")

        best_reward = -float('inf')

        for episode in range(num_episodes):
            states, _ = self.env.reset()
            episode_reward = 0.0
            losses = {i: {'critic': 0.0, 'actor': 0.0} for i in range(self.num_agents)}

            # Decay exploration noise
            noise_scale = max(0.1, 1.0 - episode / (num_episodes * 0.5))

            for step in range(max_steps):
                # Select actions
                add_noise = episode < warmup_episodes
                actions = [
                    agent.select_action(states[i], add_noise=add_noise, noise_scale=noise_scale)
                    for i, agent in enumerate(self.agents)
                ]

                # Environment step
                next_states, rewards, terminated, truncated, info = self.env.step(actions)

                # Store transition
                dones = [float(terminated or truncated)] * self.num_agents
                self.replay_buffer.push(states, actions, rewards, next_states, dones)

                # Update agents
                if len(self.replay_buffer) >= self.batch_size and episode >= warmup_episodes:
                    if step % update_interval == 0:
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
                            self.replay_buffer.sample(self.batch_size)

                        for i, agent in enumerate(self.agents):
                            metrics = agent.update(
                                self.agents,
                                batch_states,
                                batch_actions,
                                batch_rewards,
                                batch_next_states,
                                batch_dones
                            )
                            losses[i]['critic'] += metrics['critic_loss']
                            losses[i]['actor'] += metrics['actor_loss']

                            if metrics.get('target_update', True):
                                agent.soft_update()

                episode_reward += sum(rewards)
                states = next_states

                if terminated or truncated:
                    break

            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_sla_violations.append(info['sla_violation_rate'])
            self.episode_completion_rates.append(info['completion_rate'])

            reward_window = min(log_interval, len(self.episode_rewards))
            avg_reward = np.mean(self.episode_rewards[-reward_window:])
            saved_best = False
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_models(os.path.join(save_dir, 'best'))
                saved_best = True

            # Logging
            if log_interval > 0 and (episode + 1) % log_interval == 0:
                avg_sla = np.mean(self.episode_sla_violations[-reward_window:])
                avg_completion = np.mean(self.episode_completion_rates[-reward_window:])

                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  SLA Violations: {avg_sla:.2%}")
                print(f"  Completion Rate: {avg_completion:.2%}")
                print(f"  Noise Scale: {noise_scale:.3f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")

                if saved_best:
                    print(f"  ✓ New best model saved (reward: {best_reward:.2f})")

            # Periodic save
            if save_interval > 0 and (episode + 1) % save_interval == 0:
                self.save_models(os.path.join(save_dir, f'checkpoint_{episode + 1}'))

        # Save final models
        self.save_models(os.path.join(save_dir, 'final'))

        # Save training metrics
        self.save_metrics(os.path.join(save_dir, 'metrics.json'), metadata={
            'algorithm': algorithm_name.lower(),
            'num_agents': self.num_agents,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'lr_actor': self.lr_actor,
            'lr_critic': self.lr_critic,
            'gamma': self.gamma,
            'tau': self.tau,
            'buffer_capacity': self.buffer_capacity,
            'batch_size': self.batch_size,
            'action_prior': self.action_prior,
            'workload_arrival_rate': getattr(self.env, 'workload_arrival_rate', None),
            'episode_length': getattr(self.env, 'episode_length', None),
            'agent_cls': self.agent_cls.__name__,
            'agent_kwargs': self.agent_kwargs,
            'num_episodes': num_episodes,
            'max_steps': max_steps,
            'warmup_episodes': warmup_episodes,
            'update_interval': update_interval,
            'save_interval': save_interval,
            'log_interval': log_interval,
            'seed': self.seed,
            'best_reward': float(best_reward),
        })

        print("\n✓ Training complete!")
        print(f"Best average reward: {best_reward:.2f}")

        return self.episode_rewards, self.episode_sla_violations, self.episode_completion_rates

    def save_models(self, path: str):
        """Save all agent models"""
        os.makedirs(path, exist_ok=True)
        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(path, f'agent_{i}.pt'))

    def load_models(self, path: str):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(path, f'agent_{i}.pt'))

    def save_metrics(self, path: str, metadata: Dict[str, Any] = None):
        """Save training metrics"""
        metrics = {
            'metadata': metadata or {},
            'episode_rewards': self.episode_rewards,
            'episode_sla_violations': self.episode_sla_violations,
            'episode_completion_rates': self.episode_completion_rates,
        }
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)

    def evaluate(self, num_episodes: int = 100, render: bool = False):
        """Evaluate trained agents"""
        print(f"\nEvaluating for {num_episodes} episodes...")

        eval_rewards = []
        eval_sla_violations = []
        eval_completion_rates = []

        for episode in range(num_episodes):
            states, _ = self.env.reset()
            episode_reward = 0.0

            while True:
                # Select actions (no exploration)
                actions = [
                    agent.select_action(states[i], add_noise=False)
                    for i, agent in enumerate(self.agents)
                ]

                next_states, rewards, terminated, truncated, info = self.env.step(actions)

                if render:
                    self.env.render()

                episode_reward += sum(rewards)
                states = next_states

                if terminated or truncated:
                    eval_rewards.append(episode_reward)
                    eval_sla_violations.append(info['sla_violation_rate'])
                    eval_completion_rates.append(info['completion_rate'])
                    break

        print(f"\nEvaluation Results ({num_episodes} episodes):")
        print(f"  Average Reward: {np.mean(eval_rewards):.2f} ± {np.std(eval_rewards):.2f}")
        print(f"  SLA Violations: {np.mean(eval_sla_violations):.2%} ± {np.std(eval_sla_violations):.2%}")
        print(f"  Completion Rate: {np.mean(eval_completion_rates):.2%} ± {np.std(eval_completion_rates):.2%}")

        return eval_rewards, eval_sla_violations, eval_completion_rates


class MATD3Trainer(MADDPGTrainer):
    """MATD3 trainer using twin critics, policy smoothing, and delayed actor updates."""

    def __init__(self,
                 env: DistributedResourceEnv,
                 hidden_dim: int = 256,
                 lr_actor: float = 1e-4,
                 lr_critic: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 0.01,
                 buffer_capacity: int = 100000,
                 batch_size: int = 256,
                 action_prior: float = 0.3,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 seed: int = None):
        super().__init__(
            env=env,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            action_prior=action_prior,
            seed=seed,
            agent_cls=MATD3Agent,
            agent_kwargs={
                'policy_noise': policy_noise,
                'noise_clip': noise_clip,
                'policy_delay': policy_delay,
            }
        )


# Main training script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MADDPG or MATD3 resource allocation agents.")
    parser.add_argument("--algorithm", choices=["maddpg", "matd3"], default="maddpg")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--warmup-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=100000)
    parser.add_argument("--lr-actor", type=float, default=1e-4)
    parser.add_argument("--lr-critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--action-prior", type=float, default=0.3)
    parser.add_argument("--workload-arrival-rate", type=float, default=5.0)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--update-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-dir", default=None)
    args = parser.parse_args()

    set_random_seeds(args.seed)

    # Create environment
    env = DistributedResourceEnv(
        num_agents=args.num_agents,
        workload_arrival_rate=args.workload_arrival_rate,
        episode_length=args.max_steps,
        seed=args.seed
    )

    # Create trainer
    trainer_cls = MATD3Trainer if args.algorithm == "matd3" else MADDPGTrainer
    trainer = trainer_cls(
        env=env,
        hidden_dim=args.hidden_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        tau=args.tau,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        action_prior=args.action_prior,
        **({
            'policy_noise': args.policy_noise,
            'noise_clip': args.noise_clip,
            'policy_delay': args.policy_delay,
        } if args.algorithm == "matd3" else {}),
        seed=args.seed
    )

    # Train
    rewards, sla_violations, completion_rates = trainer.train(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        warmup_episodes=args.warmup_episodes,
        update_interval=args.update_interval,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        save_dir=args.save_dir or f'./models/{args.algorithm}'
    )

    # Evaluate
    if args.eval_episodes > 0:
        trainer.evaluate(num_episodes=args.eval_episodes, render=False)
