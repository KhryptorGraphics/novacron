"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) Training
Implements centralized training with decentralized execution
"""
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
from environment import DistributedResourceEnv


class Actor(nn.Module):
    """Actor network for MADDPG (policy network)"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

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
                 tau: float = 0.01):

        self.agent_id = agent_id
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Actor networks (decentralized - only sees own state)
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim)
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
            'q_value': current_q.mean().item()
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
                 batch_size: int = 256):

        self.env = env
        self.num_agents = env.num_agents
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.batch_size = batch_size

        # Create agents
        self.agents = [
            MADDPGAgent(
                agent_id=i,
                num_agents=self.num_agents,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=hidden_dim,
                lr_actor=lr_actor,
                lr_critic=lr_critic,
                gamma=gamma,
                tau=tau
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

        print(f"Starting MADDPG training for {num_episodes} episodes...")
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

                            # Soft update target networks
                            agent.soft_update()

                episode_reward += sum(rewards)
                states = next_states

                if terminated or truncated:
                    break

            # Record metrics
            self.episode_rewards.append(episode_reward)
            self.episode_sla_violations.append(info['sla_violation_rate'])
            self.episode_completion_rates.append(info['completion_rate'])

            # Logging
            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-log_interval:])
                avg_sla = np.mean(self.episode_sla_violations[-log_interval:])
                avg_completion = np.mean(self.episode_completion_rates[-log_interval:])

                print(f"\nEpisode {episode + 1}/{num_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  SLA Violations: {avg_sla:.2%}")
                print(f"  Completion Rate: {avg_completion:.2%}")
                print(f"  Noise Scale: {noise_scale:.3f}")
                print(f"  Buffer Size: {len(self.replay_buffer)}")

                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    self.save_models(os.path.join(save_dir, 'best'))
                    print(f"  ✓ New best model saved (reward: {best_reward:.2f})")

            # Periodic save
            if (episode + 1) % save_interval == 0:
                self.save_models(os.path.join(save_dir, f'checkpoint_{episode + 1}'))

        # Save final models
        self.save_models(os.path.join(save_dir, 'final'))

        # Save training metrics
        self.save_metrics(os.path.join(save_dir, 'metrics.json'))

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

    def save_metrics(self, path: str):
        """Save training metrics"""
        metrics = {
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


# Main training script
if __name__ == "__main__":
    # Create environment
    env = DistributedResourceEnv(
        num_agents=10,
        workload_arrival_rate=5.0,
        episode_length=1000
    )

    # Create trainer
    trainer = MADDPGTrainer(
        env=env,
        hidden_dim=256,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.01,
        buffer_capacity=100000,
        batch_size=256
    )

    # Train
    rewards, sla_violations, completion_rates = trainer.train(
        num_episodes=10000,
        max_steps=1000,
        warmup_episodes=100,
        save_interval=100,
        log_interval=10,
        save_dir='./models/maddpg'
    )

    # Evaluate
    trainer.evaluate(num_episodes=100, render=False)
