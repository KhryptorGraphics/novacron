#!/usr/bin/env python3
"""
Unit tests for MADDPG training components
"""
import json
import os
import tempfile
import unittest
import torch
import numpy as np
from benchmark import evaluate_acceptance, run_benchmark
from train import Actor, Critic, OUNoise, ReplayBuffer, MADDPGAgent, MATD3Agent, MATD3Trainer
from environment import DistributedResourceEnv
from validate_artifact import validate_performance_artifact


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

    def test_actor_action_prior_initialization(self):
        """Test actor output layer bias is initialized from action prior"""
        actor = Actor(state_dim=8, action_dim=4, hidden_dim=128, action_prior=0.25)
        expected_bias = torch.full((4,), np.log(0.25 / 0.75), dtype=actor.fc3.bias.dtype)
        states = torch.randn(3, 8)
        actions = actor(states)

        self.assertTrue(torch.allclose(actor.fc3.bias.detach(), expected_bias))
        self.assertTrue(torch.allclose(actions, torch.full_like(actions, 0.25), atol=1e-6))

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
        """Test noise is mean-reverting toward mu"""
        noise = OUNoise(action_dim=4, mu=0.0, theta=0.15, sigma=0.0)
        noise.state = np.ones(4)

        for _ in range(10):
            noise.sample()

        self.assertTrue(np.all(noise.state < 1.0))
        self.assertTrue(np.all(noise.state > 0.0))


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


class TestMATD3Agent(unittest.TestCase):
    """Test MATD3Agent"""

    def setUp(self):
        self.agent = MATD3Agent(
            agent_id=0,
            num_agents=5,
            state_dim=8,
            action_dim=4,
            hidden_dim=128,
            policy_delay=2
        )

    def test_agent_initialization(self):
        """Test twin critic structures are initialized"""
        self.assertIsNotNone(self.agent.critic)
        self.assertIsNotNone(self.agent.critic2)
        self.assertIsNotNone(self.agent.critic_target)
        self.assertIsNotNone(self.agent.critic2_target)
        self.assertEqual(self.agent.policy_delay, 2)

    def test_agent_update_delays_actor(self):
        """Test MATD3 delays actor updates and reports twin critic loss"""
        agents = [
            MATD3Agent(i, 5, 8, 4, 128, policy_delay=2)
            for i in range(5)
        ]

        batch_size = 32
        states = [torch.randn(batch_size, 8) for _ in range(5)]
        actions = [torch.rand(batch_size, 4) for _ in range(5)]
        rewards = [torch.randn(batch_size, 1) for _ in range(5)]
        next_states = [torch.randn(batch_size, 8) for _ in range(5)]
        dones = [torch.zeros(batch_size, 1) for _ in range(5)]

        first_metrics = agents[0].update(agents, states, actions, rewards, next_states, dones)
        second_metrics = agents[0].update(agents, states, actions, rewards, next_states, dones)

        self.assertIn('critic2_loss', first_metrics)
        self.assertFalse(first_metrics['target_update'])
        self.assertTrue(second_metrics['target_update'])


class TestMATD3Trainer(unittest.TestCase):
    """Test MATD3Trainer wiring"""

    def test_trainer_uses_matd3_agents(self):
        env = DistributedResourceEnv(num_agents=3, seed=42)
        trainer = MATD3Trainer(env, hidden_dim=128, batch_size=8)

        self.assertEqual(len(trainer.agents), 3)
        self.assertTrue(all(isinstance(agent, MATD3Agent) for agent in trainer.agents))

    def test_train_save_and_benchmark_smoke(self):
        """Test MATD3 can train, save, load, and benchmark with matching architecture"""
        with tempfile.TemporaryDirectory() as temp_dir:
            env = DistributedResourceEnv(num_agents=2, episode_length=1, seed=42)
            trainer = MATD3Trainer(env, hidden_dim=16, batch_size=2)

            trainer.train(
                num_episodes=1,
                max_steps=1,
                warmup_episodes=0,
                save_interval=10,
                log_interval=10,
                save_dir=temp_dir
            )

            final_dir = os.path.join(temp_dir, 'final')
            best_dir = os.path.join(temp_dir, 'best')
            self.assertTrue(os.path.exists(os.path.join(final_dir, 'agent_0.pt')))
            self.assertTrue(os.path.exists(os.path.join(final_dir, 'agent_1.pt')))
            self.assertTrue(os.path.exists(os.path.join(best_dir, 'agent_0.pt')))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, 'metrics.json')))
            with open(os.path.join(temp_dir, 'metrics.json'), 'r') as f:
                metrics = json.load(f)
            self.assertEqual(metrics['metadata']['algorithm'], 'matd3')
            self.assertEqual(metrics['metadata']['num_agents'], 2)
            self.assertEqual(metrics['metadata']['action_prior'], 0.3)
            self.assertEqual(metrics['metadata']['workload_arrival_rate'], env.workload_arrival_rate)
            self.assertEqual(metrics['metadata']['episode_length'], env.episode_length)
            self.assertEqual(metrics['metadata']['agent_cls'], 'MATD3Agent')
            self.assertEqual(metrics['metadata']['agent_kwargs']['policy_delay'], 2)

            output_path = os.path.join(temp_dir, 'benchmark.json')
            results = run_benchmark(
                model_path=final_dir,
                model_algorithm='matd3',
                num_episodes=1,
                max_steps=1,
                num_agents=2,
                hidden_dim=16,
                seed=42,
                output_path=output_path
            )

            self.assertIn('random', results)
            self.assertIn('greedy', results)
            self.assertIn('matd3', results)
            self.assertIn('improvements', results)
            self.assertIn('acceptance', results)
            self.assertIn('greedy', results['improvements'])
            with open(output_path, 'r') as f:
                saved = json.load(f)
            self.assertIn('matd3', saved)
            self.assertIn('improvements', saved)
            self.assertIn('acceptance', saved)
            self.assertEqual(saved['metadata']['model_algorithm'], 'matd3')
            self.assertEqual(saved['metadata']['num_agents'], 2)
            self.assertEqual(saved['metadata']['seed'], 42)

    def test_prior_initialized_model_passes_short_acceptance_profile(self):
        """Test prior-initialized MATD3 model can clear a short deterministic benchmark gate"""
        with tempfile.TemporaryDirectory() as temp_dir:
            seed = 42
            model_dir = os.path.join(temp_dir, 'model')
            env = DistributedResourceEnv(
                num_agents=5,
                workload_arrival_rate=5.0,
                episode_length=100,
                seed=seed
            )
            trainer = MATD3Trainer(
                env,
                hidden_dim=64,
                batch_size=64,
                action_prior=0.25,
                seed=seed
            )
            trainer.save_models(model_dir)

            results = run_benchmark(
                model_path=model_dir,
                model_algorithm='matd3',
                num_episodes=8,
                max_steps=100,
                num_agents=5,
                hidden_dim=64,
                workload_arrival_rate=5.0,
                seed=seed,
                target_reward_improvement_pct=20.0,
                output_path=os.path.join(temp_dir, 'benchmark.json')
            )

            self.assertTrue(results['acceptance']['passed'])
            self.assertGreaterEqual(
                results['improvements']['random']['reward_improvement_pct'],
                20.0
            )
            self.assertGreaterEqual(
                results['improvements']['greedy']['reward_improvement_pct'],
                20.0
            )

    def test_acceptance_gate_evaluates_all_baselines(self):
        """Test target gate requires every baseline to meet reward improvement target"""
        passing = evaluate_acceptance({
            'random': {'reward_improvement_pct': 25.0},
            'greedy': {'reward_improvement_pct': 21.0},
        }, target_reward_improvement_pct=20.0)
        failing = evaluate_acceptance({
            'random': {'reward_improvement_pct': 25.0},
            'greedy': {'reward_improvement_pct': 19.0},
        }, target_reward_improvement_pct=20.0)

        self.assertTrue(passing['passed'])
        self.assertFalse(failing['passed'])
        self.assertFalse(failing['baselines']['greedy']['passed'])

    def test_performance_artifact_validation(self):
        """Test artifact validator checks metadata consistency and gate result"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_root = os.path.join(temp_dir, 'model')
            best_dir = os.path.join(model_root, 'best')
            os.makedirs(best_dir)
            for agent_id in range(2):
                open(os.path.join(best_dir, f'agent_{agent_id}.pt'), 'w').close()

            metrics = {
                'metadata': {
                    'algorithm': 'matd3',
                    'num_agents': 2,
                    'hidden_dim': 16,
                    'num_episodes': 10,
                    'max_steps': 100,
                    'seed': 42,
                    'workload_arrival_rate': 5.0,
                }
            }
            benchmark = {
                'metadata': {
                    'model_algorithm': 'matd3',
                    'model_path': best_dir,
                    'algorithms': ['random', 'greedy', 'matd3'],
                    'num_agents': 2,
                    'hidden_dim': 16,
                    'num_episodes': 5,
                    'max_steps': 100,
                    'seed': 42,
                    'workload_arrival_rate': 5.0,
                    'target_reward_improvement_pct': 20.0,
                },
                'acceptance': {
                    'passed': True,
                    'target_reward_improvement_pct': 20.0,
                    'baselines': {
                        'random': {'passed': True, 'reward_improvement_pct': 25.0},
                        'greedy': {'passed': True, 'reward_improvement_pct': 30.0},
                    },
                },
                'improvements': {
                    'random': {'reward_improvement_pct': 25.0},
                    'greedy': {'reward_improvement_pct': 30.0},
                },
                'random': {
                    'avg_reward': 10.0,
                    'avg_sla_violation': 1.0,
                    'avg_completion_rate': 0.8,
                },
                'greedy': {
                    'avg_reward': 8.0,
                    'avg_sla_violation': 1.2,
                    'avg_completion_rate': 0.7,
                },
                'matd3': {
                    'avg_reward': 13.0,
                    'avg_sla_violation': 0.5,
                    'avg_completion_rate': 0.9,
                },
            }
            metrics_path = os.path.join(model_root, 'metrics.json')
            benchmark_path = os.path.join(model_root, 'benchmark.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark, f)

            result = validate_performance_artifact(
                model_root,
                benchmark_path,
                min_training_episodes=10,
                min_benchmark_episodes=5,
                min_max_steps=100,
                min_num_agents=2
            )
            self.assertTrue(result['passed'])

            result = validate_performance_artifact(
                model_root,
                benchmark_path,
                min_training_episodes=11
            )
            self.assertFalse(result['passed'])
            self.assertTrue(any('num_episodes' in error for error in result['errors']))

            benchmark['acceptance']['baselines']['greedy']['reward_improvement_pct'] = 19.0
            benchmark['improvements']['greedy']['reward_improvement_pct'] = 19.0
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark, f)

            result = validate_performance_artifact(model_root, benchmark_path)
            self.assertFalse(result['passed'])
            self.assertTrue(any('below target' in error for error in result['errors']))

            benchmark['acceptance']['baselines']['greedy']['reward_improvement_pct'] = 30.0
            benchmark['improvements']['greedy']['reward_improvement_pct'] = 29.0
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark, f)

            result = validate_performance_artifact(model_root, benchmark_path)
            self.assertFalse(result['passed'])
            self.assertTrue(any('does not match measured' in error for error in result['errors']))

            benchmark['acceptance']['baselines']['greedy']['reward_improvement_pct'] = 30.0
            benchmark['improvements']['greedy']['reward_improvement_pct'] = 30.0
            benchmark['acceptance']['baselines']['greedy']['passed'] = False
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark, f)

            result = validate_performance_artifact(model_root, benchmark_path)
            self.assertFalse(result['passed'])
            self.assertTrue(any('greedy' in error for error in result['errors']))

            benchmark['acceptance']['baselines']['greedy']['passed'] = True
            benchmark['metadata']['model_path'] = ''
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark, f)

            result = validate_performance_artifact(model_root, benchmark_path)
            self.assertFalse(result['passed'])
            self.assertTrue(any('model_path' in error for error in result['errors']))

            benchmark['metadata']['model_path'] = best_dir
            del benchmark['matd3']
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark, f)

            result = validate_performance_artifact(model_root, benchmark_path)
            self.assertFalse(result['passed'])
            self.assertTrue(any('missing algorithm: matd3' in error for error in result['errors']))

            benchmark['matd3'] = {
                'avg_reward': 13.0,
                'avg_sla_violation': 0.5,
                'avg_completion_rate': 0.9,
            }
            benchmark['metadata']['algorithms'] = ['random', 'greedy']
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark, f)

            result = validate_performance_artifact(model_root, benchmark_path)
            self.assertFalse(result['passed'])
            self.assertTrue(any('algorithms missing: matd3' in error for error in result['errors']))

            benchmark['metadata']['algorithms'] = ['random', 'greedy', 'matd3']
            del benchmark['matd3']['avg_completion_rate']
            with open(benchmark_path, 'w') as f:
                json.dump(benchmark, f)

            result = validate_performance_artifact(model_root, benchmark_path)
            self.assertFalse(result['passed'])
            self.assertTrue(any('avg_completion_rate' in error for error in result['errors']))

    def test_benchmark_is_seed_reproducible(self):
        """Test benchmark baselines are reproducible with a fixed seed"""
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = run_benchmark(
                model_path=None,
                num_episodes=2,
                max_steps=2,
                num_agents=2,
                hidden_dim=16,
                seed=7,
                output_path=os.path.join(first_dir, 'benchmark.json')
            )
            second = run_benchmark(
                model_path=None,
                num_episodes=2,
                max_steps=2,
                num_agents=2,
                hidden_dim=16,
                seed=7,
                output_path=os.path.join(second_dir, 'benchmark.json')
            )

            for algorithm in ['random', 'greedy']:
                for metric in [
                    'avg_reward',
                    'avg_sla_violation',
                    'avg_completion_rate',
                ]:
                    self.assertAlmostEqual(first[algorithm][metric], second[algorithm][metric])


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
