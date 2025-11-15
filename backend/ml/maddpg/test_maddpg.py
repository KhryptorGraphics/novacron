#!/usr/bin/env python3
"""
Unit tests for MADDPG training components
"""
import unittest
import torch
import numpy as np
from train import Actor, Critic, OUNoise, ReplayBuffer, MADDPGAgent
from environment import DistributedResourceEnv


class TestActor(unittest.TestCase):
    """Test Actor network"""

    def test_actor_initialization(self):
        """Test actor is initialized correctly"""
        actor = Actor(state_dim=8, action_dim=4, hidden_dim=128)

        self.assertIsNotNone(actor.fc1)
        self.assertIsNotNone(actor.fc2)
        self.assertIsNotNone(actor.fc3)

    def test_actor_forward(self):
        """Test actor forward pass"""
        actor = Actor(state_dim=8, action_dim=4, hidden_dim=128)
        state = torch.randn(1, 8)

        action = actor(state)

        self.assertEqual(action.shape, (1, 4))
        # Actions should be in [0, 1] due to sigmoid
        self.assertTrue(torch.all(action >= 0))
        self.assertTrue(torch.all(action <= 1))

    def test_actor_batch(self):
        """Test actor with batch input"""
        actor = Actor(state_dim=8, action_dim=4, hidden_dim=128)
        states = torch.randn(32, 8)

        actions = actor(states)

        self.assertEqual(actions.shape, (32, 4))


class TestCritic(unittest.TestCase):
    """Test Critic network"""

    def test_critic_initialization(self):
        """Test critic is initialized correctly"""
        critic = Critic(total_state_dim=40, total_action_dim=20, hidden_dim=128)

        self.assertIsNotNone(critic.fc1)
        self.assertIsNotNone(critic.fc2)
        self.assertIsNotNone(critic.fc3)

    def test_critic_forward(self):
        """Test critic forward pass"""
        num_agents = 5
        critic = Critic(
            total_state_dim=8 * num_agents,
            total_action_dim=4 * num_agents,
            hidden_dim=128
        )

        states = torch.randn(1, 8 * num_agents)
        actions = torch.randn(1, 4 * num_agents)

        q_value = critic(states, actions)

        self.assertEqual(q_value.shape, (1, 1))

    def test_critic_batch(self):
        """Test critic with batch input"""
        num_agents = 5
        critic = Critic(
            total_state_dim=8 * num_agents,
            total_action_dim=4 * num_agents,
            hidden_dim=128
        )

        states = torch.randn(32, 8 * num_agents)
        actions = torch.randn(32, 4 * num_agents)

        q_values = critic(states, actions)

        self.assertEqual(q_values.shape, (32, 1))


class TestOUNoise(unittest.TestCase):
    """Test Ornstein-Uhlenbeck noise"""

    def test_noise_initialization(self):
        """Test noise is initialized correctly"""
        noise = OUNoise(action_dim=4)

        self.assertEqual(len(noise.state), 4)
        self.assertEqual(noise.mu, 0.0)

    def test_noise_sampling(self):
        """Test noise sampling"""
        noise = OUNoise(action_dim=4, sigma=0.1)
        sample = noise.sample()

        self.assertEqual(len(sample), 4)

    def test_noise_reset(self):
        """Test noise reset"""
        noise = OUNoise(action_dim=4)

        # Sample some noise
        for _ in range(10):
            noise.sample()

        # Reset
        noise.reset()

        self.assertTrue(np.allclose(noise.state, noise.mu))

    def test_noise_properties(self):
        """Test noise has correct statistical properties"""
        noise = OUNoise(action_dim=4, mu=0.0, sigma=0.2)

        samples = [noise.sample() for _ in range(1000)]
        mean = np.mean(samples, axis=0)

        # Mean should be close to mu
        self.assertTrue(np.allclose(mean, 0.0, atol=0.1))


class TestReplayBuffer(unittest.TestCase):
    """Test ReplayBuffer"""

    def test_buffer_initialization(self):
        """Test buffer is initialized correctly"""
        buffer = ReplayBuffer(capacity=1000)

        self.assertEqual(len(buffer), 0)

    def test_buffer_push(self):
        """Test pushing to buffer"""
        buffer = ReplayBuffer(capacity=1000)

        states = [np.random.rand(8) for _ in range(5)]
        actions = [np.random.rand(4) for _ in range(5)]
        rewards = [1.0] * 5
        next_states = [np.random.rand(8) for _ in range(5)]
        dones = [False] * 5

        buffer.push(states, actions, rewards, next_states, dones)

        self.assertEqual(len(buffer), 1)

    def test_buffer_sample(self):
        """Test sampling from buffer"""
        buffer = ReplayBuffer(capacity=1000)

        # Add some transitions
        for _ in range(100):
            states = [np.random.rand(8) for _ in range(5)]
            actions = [np.random.rand(4) for _ in range(5)]
            rewards = [np.random.rand()] * 5
            next_states = [np.random.rand(8) for _ in range(5)]
            dones = [False] * 5
            buffer.push(states, actions, rewards, next_states, dones)

        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        self.assertEqual(len(states), 5)  # 5 agents
        self.assertEqual(states[0].shape, (batch_size, 8))
        self.assertEqual(actions[0].shape, (batch_size, 4))

    def test_buffer_capacity(self):
        """Test buffer respects capacity"""
        buffer = ReplayBuffer(capacity=10)

        # Add more than capacity
        for _ in range(20):
            states = [np.random.rand(8) for _ in range(5)]
            actions = [np.random.rand(4) for _ in range(5)]
            rewards = [1.0] * 5
            next_states = [np.random.rand(8) for _ in range(5)]
            dones = [False] * 5
            buffer.push(states, actions, rewards, next_states, dones)

        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)


class TestMADDPGAgent(unittest.TestCase):
    """Test MADDPGAgent"""

    def setUp(self):
        """Set up test agent"""
        self.agent = MADDPGAgent(
            agent_id=0,
            num_agents=5,
            state_dim=8,
            action_dim=4,
            hidden_dim=128
        )

    def test_agent_initialization(self):
        """Test agent is initialized correctly"""
        self.assertEqual(self.agent.agent_id, 0)
        self.assertEqual(self.agent.num_agents, 5)
        self.assertIsNotNone(self.agent.actor)
        self.assertIsNotNone(self.agent.critic)

    def test_agent_select_action(self):
        """Test action selection"""
        state = np.random.rand(8)
        action = self.agent.select_action(state, add_noise=False)

        self.assertEqual(len(action), 4)
        self.assertTrue(np.all(action >= 0))
        self.assertTrue(np.all(action <= 1))

    def test_agent_select_action_with_noise(self):
        """Test action selection with exploration noise"""
        state = np.random.rand(8)

        # Get action without noise
        action_no_noise = self.agent.select_action(state, add_noise=False)

        # Get action with noise
        action_with_noise = self.agent.select_action(state, add_noise=True)

        # Actions should be different
        self.assertFalse(np.allclose(action_no_noise, action_with_noise))

    def test_agent_soft_update(self):
        """Test soft update of target networks"""
        # Get initial target parameters
        initial_actor_param = next(self.agent.actor_target.parameters()).clone()

        # Update main network
        for param in self.agent.actor.parameters():
            param.data.fill_(1.0)

        # Soft update
        self.agent.soft_update()

        # Target should change slightly
        updated_param = next(self.agent.actor_target.parameters())
        self.assertFalse(torch.allclose(initial_actor_param, updated_param))

    def test_agent_update(self):
        """Test agent update"""
        # Create dummy agents
        agents = [
            MADDPGAgent(i, 5, 8, 4, 128)
            for i in range(5)
        ]

        # Create dummy batch
        batch_size = 32
        states = [torch.randn(batch_size, 8) for _ in range(5)]
        actions = [torch.randn(batch_size, 4) for _ in range(5)]
        rewards = [torch.randn(batch_size, 1) for _ in range(5)]
        next_states = [torch.randn(batch_size, 8) for _ in range(5)]
        dones = [torch.zeros(batch_size, 1) for _ in range(5)]

        # Update agent
        metrics = self.agent.update(agents, states, actions, rewards, next_states, dones)

        # Should return loss metrics
        self.assertIn('critic_loss', metrics)
        self.assertIn('actor_loss', metrics)
        self.assertIn('q_value', metrics)


class TestMADDPGIntegration(unittest.TestCase):
    """Integration tests with environment"""

    def test_agent_environment_interaction(self):
        """Test agent can interact with environment"""
        env = DistributedResourceEnv(num_agents=3, seed=42)
        observations, _ = env.reset()

        agents = [
            MADDPGAgent(i, 3, 8, 4, 128)
            for i in range(3)
        ]

        # Select actions
        actions = [
            agent.select_action(obs, add_noise=False)
            for agent, obs in zip(agents, observations)
        ]

        # Step environment
        next_observations, rewards, _, _, _ = env.step(actions)

        self.assertEqual(len(next_observations), 3)
        self.assertEqual(len(rewards), 3)

    def test_training_step(self):
        """Test a single training step"""
        env = DistributedResourceEnv(num_agents=3, seed=42)
        buffer = ReplayBuffer(capacity=1000)

        agents = [
            MADDPGAgent(i, 3, 8, 4, 128)
            for i in range(3)
        ]

        # Collect some experience
        observations, _ = env.reset()
        for _ in range(10):
            actions = [
                agent.select_action(obs, add_noise=True)
                for agent, obs in zip(agents, observations)
            ]

            next_observations, rewards, terminated, truncated, _ = env.step(actions)
            dones = [float(terminated or truncated)] * 3

            buffer.push(observations, actions, rewards, next_observations, dones)

            observations = next_observations

            if terminated or truncated:
                break

        # Sample and update
        if len(buffer) >= 8:
            states, actions, rewards, next_states, dones = buffer.sample(8)

            for agent in agents:
                agent.update(agents, states, actions, rewards, next_states, dones)
                agent.soft_update()


if __name__ == '__main__':
    unittest.main(verbosity=2)
