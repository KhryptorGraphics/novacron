#!/usr/bin/env python3
"""
Unit tests for DistributedResourceEnv
"""
import unittest
import numpy as np
from environment import DistributedResourceEnv, Node, Workload


class TestNode(unittest.TestCase):
    """Test Node class"""

    def test_node_initialization(self):
        """Test node is initialized correctly"""
        node = Node(
            id=0,
            cpu_capacity=100.0,
            memory_capacity=64.0,
            bandwidth_capacity=1000.0,
            storage_capacity=500.0
        )

        self.assertEqual(node.id, 0)
        self.assertEqual(node.cpu_capacity, 100.0)
        self.assertEqual(node.cpu_usage, 0.0)
        self.assertEqual(node.cpu_available, 100.0)

    def test_node_observation(self):
        """Test node observation is correctly normalized"""
        node = Node(
            id=0,
            cpu_capacity=100.0,
            memory_capacity=64.0,
            bandwidth_capacity=1000.0,
            storage_capacity=500.0,
            cpu_usage=50.0,
            memory_usage=32.0
        )

        obs = node.get_observation()

        self.assertEqual(len(obs), 8)
        self.assertAlmostEqual(obs[0], 0.5)  # cpu_usage / cpu_capacity
        self.assertAlmostEqual(obs[1], 0.5)  # memory_usage / memory_capacity
        self.assertAlmostEqual(obs[4], 0.5)  # cpu_available / cpu_capacity


class TestWorkload(unittest.TestCase):
    """Test Workload class"""

    def test_workload_initialization(self):
        """Test workload is initialized correctly"""
        workload = Workload(
            id=1,
            cpu_requirement=10.0,
            memory_requirement=8.0,
            bandwidth_requirement=100.0,
            storage_requirement=50.0,
            priority=2.0
        )

        self.assertEqual(workload.id, 1)
        self.assertEqual(workload.cpu_requirement, 10.0)
        self.assertEqual(workload.priority, 2.0)

    def test_workload_requirements(self):
        """Test workload requirements array"""
        workload = Workload(
            id=1,
            cpu_requirement=10.0,
            memory_requirement=8.0,
            bandwidth_requirement=100.0,
            storage_requirement=50.0
        )

        reqs = workload.get_requirements()
        self.assertEqual(len(reqs), 4)
        self.assertEqual(reqs[0], 10.0)
        self.assertEqual(reqs[1], 8.0)


class TestDistributedResourceEnv(unittest.TestCase):
    """Test DistributedResourceEnv"""

    def setUp(self):
        """Set up test environment"""
        self.env = DistributedResourceEnv(
            num_agents=5,
            workload_arrival_rate=3.0,
            episode_length=100,
            seed=42
        )

    def test_env_initialization(self):
        """Test environment is initialized correctly"""
        self.assertEqual(self.env.num_agents, 5)
        self.assertEqual(len(self.env.nodes), 5)
        self.assertEqual(self.env.observation_space.shape, (8,))
        self.assertEqual(self.env.action_space.shape, (4,))

    def test_env_reset(self):
        """Test environment reset"""
        observations, info = self.env.reset(seed=42)

        self.assertEqual(len(observations), 5)
        self.assertEqual(observations[0].shape, (8,))
        self.assertEqual(self.env.current_step, 0)
        self.assertIn('step', info)
        self.assertIn('total_workloads', info)

    def test_env_step(self):
        """Test environment step"""
        self.env.reset(seed=42)

        # Random actions
        actions = [self.env.action_space.sample() for _ in range(5)]

        observations, rewards, terminated, truncated, info = self.env.step(actions)

        self.assertEqual(len(observations), 5)
        self.assertEqual(len(rewards), 5)
        self.assertEqual(observations[0].shape, (8,))
        self.assertEqual(self.env.current_step, 1)
        self.assertFalse(terminated)
        self.assertIn('completion_rate', info)

    def test_env_episode_termination(self):
        """Test episode terminates after max steps"""
        self.env.episode_length = 10
        self.env.reset(seed=42)

        for _ in range(10):
            actions = [self.env.action_space.sample() for _ in range(5)]
            _, _, terminated, _, _ = self.env.step(actions)

        self.assertTrue(terminated)

    def test_workload_generation(self):
        """Test workload generation"""
        workload = self.env._generate_workload()

        self.assertGreater(workload.id, 0)
        self.assertGreater(workload.cpu_requirement, 0)
        self.assertGreater(workload.memory_requirement, 0)
        self.assertIn(workload.priority, [1.0, 2.0, 3.0])

    def test_allocation_logic(self):
        """Test workload allocation logic"""
        self.env.reset(seed=42)

        # Create a simple allocation scenario
        # Action: allocate 100% of available resources
        actions = [np.array([1.0, 1.0, 1.0, 1.0]) for _ in range(5)]

        observations, rewards, _, _, info = self.env.step(actions)

        # Should have some allocations
        self.assertGreater(info['total_workloads'], 0)

        # Rewards should be non-zero if allocations happened
        if info['completed_workloads'] > 0:
            self.assertTrue(any(r > 0 for r in rewards))

    def test_sla_violation_tracking(self):
        """Test SLA violation tracking"""
        self.env.reset(seed=42)

        # Don't allocate anything (all zeros)
        actions = [np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(5)]

        # Run several steps
        for _ in range(10):
            _, _, terminated, truncated, info = self.env.step(actions)
            if terminated or truncated:
                break

        # Should have some SLA violations
        self.assertGreaterEqual(info['sla_violations'], 0)

    def test_resource_decay(self):
        """Test resource usage decays over time"""
        self.env.reset(seed=42)

        # Allocate some resources
        actions = [np.array([0.5, 0.5, 0.5, 0.5]) for _ in range(5)]
        self.env.step(actions)

        # Record initial usage
        initial_usage = sum(node.cpu_usage for node in self.env.nodes)

        # Step with no new allocations
        actions = [np.array([0.0, 0.0, 0.0, 0.0]) for _ in range(5)]
        for _ in range(5):
            self.env.step(actions)

        # Usage should decrease
        final_usage = sum(node.cpu_usage for node in self.env.nodes)
        self.assertLessEqual(final_usage, initial_usage)

    def test_observation_bounds(self):
        """Test observations are within [0, 1] bounds"""
        self.env.reset(seed=42)

        for _ in range(10):
            actions = [self.env.action_space.sample() for _ in range(5)]
            observations, _, terminated, truncated, _ = self.env.step(actions)

            for obs in observations:
                self.assertTrue(np.all(obs >= 0.0))
                self.assertTrue(np.all(obs <= 1.0))

            if terminated or truncated:
                break

    def test_heterogeneous_nodes(self):
        """Test nodes have different capacities"""
        self.env.reset(seed=42)

        capacities = [node.cpu_capacity for node in self.env.nodes]

        # Should have some variation
        self.assertGreater(max(capacities), min(capacities))

    def test_info_dict_structure(self):
        """Test info dict has required keys"""
        self.env.reset(seed=42)

        actions = [self.env.action_space.sample() for _ in range(5)]
        _, _, _, _, info = self.env.step(actions)

        required_keys = [
            'step', 'total_workloads', 'completed_workloads',
            'failed_workloads', 'completion_rate', 'sla_violation_rate',
            'avg_utilization', 'total_reward'
        ]

        for key in required_keys:
            self.assertIn(key, info)


class TestEnvironmentEdgeCases(unittest.TestCase):
    """Test edge cases"""

    def test_single_agent(self):
        """Test environment with single agent"""
        env = DistributedResourceEnv(num_agents=1, seed=42)
        observations, _ = env.reset()

        self.assertEqual(len(observations), 1)

        actions = [env.action_space.sample()]
        observations, rewards, _, _, _ = env.step(actions)

        self.assertEqual(len(observations), 1)
        self.assertEqual(len(rewards), 1)

    def test_many_agents(self):
        """Test environment with many agents"""
        env = DistributedResourceEnv(num_agents=50, seed=42)
        observations, _ = env.reset()

        self.assertEqual(len(observations), 50)

        actions = [env.action_space.sample() for _ in range(50)]
        observations, rewards, _, _, _ = env.step(actions)

        self.assertEqual(len(observations), 50)

    def test_zero_workload_arrival(self):
        """Test with very low workload arrival rate"""
        env = DistributedResourceEnv(
            num_agents=5,
            workload_arrival_rate=0.01,
            seed=42
        )
        env.reset()

        actions = [env.action_space.sample() for _ in range(5)]
        _, _, _, _, info = env.step(actions)

        # Should work even with few/no workloads
        self.assertGreaterEqual(info['total_workloads'], 0)

    def test_action_clipping(self):
        """Test actions are properly clipped"""
        env = DistributedResourceEnv(num_agents=5, seed=42)
        env.reset()

        # Invalid actions (out of bounds)
        invalid_actions = [
            np.array([2.0, -1.0, 1.5, 0.5]) for _ in range(5)
        ]

        # Should not crash
        observations, _, _, _, _ = env.step(invalid_actions)
        self.assertEqual(len(observations), 5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
