"""
Multi-Agent Distributed Resource Environment
Gymnasium-compatible environment for distributed resource allocation
"""
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class ResourceType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    BANDWIDTH = "bandwidth"
    STORAGE = "storage"


@dataclass
class Node:
    """Represents a compute node in the distributed system"""
    id: int
    cpu_capacity: float
    memory_capacity: float
    bandwidth_capacity: float
    storage_capacity: float
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    bandwidth_usage: float = 0.0
    storage_usage: float = 0.0

    @property
    def cpu_available(self) -> float:
        return max(0, self.cpu_capacity - self.cpu_usage)

    @property
    def memory_available(self) -> float:
        return max(0, self.memory_capacity - self.memory_usage)

    @property
    def bandwidth_available(self) -> float:
        return max(0, self.bandwidth_capacity - self.bandwidth_usage)

    @property
    def storage_available(self) -> float:
        return max(0, self.storage_capacity - self.storage_usage)

    def get_observation(self) -> np.ndarray:
        """Get normalized observation vector for this node"""
        return np.array([
            self.cpu_usage / max(self.cpu_capacity, 1e-6),
            self.memory_usage / max(self.memory_capacity, 1e-6),
            self.bandwidth_usage / max(self.bandwidth_capacity, 1e-6),
            self.storage_usage / max(self.storage_capacity, 1e-6),
            self.cpu_available / max(self.cpu_capacity, 1e-6),
            self.memory_available / max(self.memory_capacity, 1e-6),
            self.bandwidth_available / max(self.bandwidth_capacity, 1e-6),
            self.storage_available / max(self.storage_capacity, 1e-6),
        ], dtype=np.float32)


@dataclass
class Workload:
    """Represents a workload to be allocated"""
    id: int
    cpu_requirement: float
    memory_requirement: float
    bandwidth_requirement: float
    storage_requirement: float
    priority: float = 1.0
    sla_deadline: float = 1.0  # in seconds

    def get_requirements(self) -> np.ndarray:
        return np.array([
            self.cpu_requirement,
            self.memory_requirement,
            self.bandwidth_requirement,
            self.storage_requirement
        ], dtype=np.float32)


class DistributedResourceEnv(gym.Env):
    """
    Multi-Agent Distributed Resource Allocation Environment

    Each agent controls one compute node and decides how much of its resources
    to allocate to incoming workloads. Agents must cooperate to optimize
    global system performance.

    Observation Space (per agent): Box(8,) - node resource state
    Action Space (per agent): Box(4,) - resource allocation percentages [0,1]

    Rewards based on:
    - Workload completion (meeting SLA deadlines)
    - Resource utilization efficiency
    - Load balancing across nodes
    - SLA violation penalties
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self,
                 num_agents: int = 10,
                 workload_arrival_rate: float = 5.0,
                 episode_length: int = 1000,
                 seed: int = None):
        super().__init__()

        self.num_agents = num_agents
        self.workload_arrival_rate = workload_arrival_rate
        self.episode_length = episode_length
        self.current_step = 0

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Initialize nodes
        self.nodes: List[Node] = []
        self._initialize_nodes()

        # Workload queue
        self.workload_queue: List[Workload] = []
        self.completed_workloads: List[Workload] = []
        self.failed_workloads: List[Workload] = []
        self.workload_counter = 0

        # Performance metrics
        self.total_reward = 0.0
        self.sla_violations = 0
        self.total_workloads = 0

    def _initialize_nodes(self):
        """Initialize compute nodes with heterogeneous capacities"""
        for i in range(self.num_agents):
            # Heterogeneous node capacities (some nodes more powerful than others)
            capacity_multiplier = np.random.uniform(0.5, 1.5)
            self.nodes.append(Node(
                id=i,
                cpu_capacity=100.0 * capacity_multiplier,
                memory_capacity=64.0 * capacity_multiplier,
                bandwidth_capacity=1000.0 * capacity_multiplier,
                storage_capacity=500.0 * capacity_multiplier
            ))

    def _generate_workload(self) -> Workload:
        """Generate a new workload with random requirements"""
        self.workload_counter += 1
        return Workload(
            id=self.workload_counter,
            cpu_requirement=np.random.uniform(5.0, 30.0),
            memory_requirement=np.random.uniform(2.0, 16.0),
            bandwidth_requirement=np.random.uniform(50.0, 200.0),
            storage_requirement=np.random.uniform(10.0, 100.0),
            priority=np.random.choice([1.0, 2.0, 3.0]),  # Low, medium, high
            sla_deadline=np.random.uniform(0.5, 2.0)
        )

    def reset(self, seed=None, options=None) -> Tuple[List[np.ndarray], Dict]:
        """Reset the environment"""
        super().reset(seed=seed)

        # Reset nodes
        self._initialize_nodes()

        # Clear workloads
        self.workload_queue = []
        self.completed_workloads = []
        self.failed_workloads = []
        self.workload_counter = 0

        # Reset metrics
        self.current_step = 0
        self.total_reward = 0.0
        self.sla_violations = 0
        self.total_workloads = 0

        # Generate initial workloads
        num_initial_workloads = np.random.poisson(self.workload_arrival_rate)
        for _ in range(num_initial_workloads):
            self.workload_queue.append(self._generate_workload())

        # Return observations for all agents
        observations = [node.get_observation() for node in self.nodes]
        info = self._get_info()

        return observations, info

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """
        Execute one step in the environment

        Args:
            actions: List of action arrays, one per agent
                     Each action is [cpu_alloc, mem_alloc, bw_alloc, storage_alloc]

        Returns:
            observations: List of observations, one per agent
            rewards: List of rewards, one per agent
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        self.current_step += 1

        # Clip actions to valid range
        actions = [np.clip(action, 0.0, 1.0) for action in actions]

        # Process workload allocation based on actions
        rewards, allocations = self._allocate_workloads(actions)

        # Update resource usage based on allocations
        self._update_resource_usage(allocations)

        # Generate new workloads (Poisson arrival)
        num_new_workloads = np.random.poisson(self.workload_arrival_rate / 100.0)
        for _ in range(num_new_workloads):
            self.workload_queue.append(self._generate_workload())

        # Get new observations
        observations = [node.get_observation() for node in self.nodes]

        # Check if episode is done
        terminated = self.current_step >= self.episode_length
        truncated = False

        info = self._get_info()

        return observations, rewards, terminated, truncated, info

    def _allocate_workloads(self, actions: List[np.ndarray]) -> Tuple[List[float], List[Dict]]:
        """
        Allocate workloads to nodes based on agent actions

        Returns:
            rewards: Reward for each agent
            allocations: Resource allocations for each node
        """
        rewards = [0.0] * self.num_agents
        allocations = [{} for _ in range(self.num_agents)]

        # Process each workload in the queue
        remaining_workloads = []
        for workload in self.workload_queue:
            allocated = False

            # Find best node for this workload based on actions
            allocation_scores = []
            for agent_id, (node, action) in enumerate(zip(self.nodes, actions)):
                # Calculate allocation percentage for this workload
                cpu_alloc = action[0] * node.cpu_available
                mem_alloc = action[1] * node.memory_available
                bw_alloc = action[2] * node.bandwidth_available
                storage_alloc = action[3] * node.storage_available

                # Check if node can satisfy workload requirements
                can_allocate = (
                    cpu_alloc >= workload.cpu_requirement and
                    mem_alloc >= workload.memory_requirement and
                    bw_alloc >= workload.bandwidth_requirement and
                    storage_alloc >= workload.storage_requirement
                )

                if can_allocate:
                    # Score based on resource efficiency
                    efficiency = (
                        (cpu_alloc - workload.cpu_requirement) / max(node.cpu_capacity, 1e-6) +
                        (mem_alloc - workload.memory_requirement) / max(node.memory_capacity, 1e-6) +
                        (bw_alloc - workload.bandwidth_requirement) / max(node.bandwidth_capacity, 1e-6) +
                        (storage_alloc - workload.storage_requirement) / max(node.storage_capacity, 1e-6)
                    ) / 4.0
                    allocation_scores.append((agent_id, efficiency))

            # Allocate to best node
            if allocation_scores:
                allocation_scores.sort(key=lambda x: x[1])  # Lower waste is better
                best_agent = allocation_scores[0][0]
                node = self.nodes[best_agent]

                # Allocate resources
                node.cpu_usage += workload.cpu_requirement
                node.memory_usage += workload.memory_requirement
                node.bandwidth_usage += workload.bandwidth_requirement
                node.storage_usage += workload.storage_requirement

                # Reward for successful allocation
                base_reward = 10.0 * workload.priority
                efficiency_bonus = (1.0 - allocation_scores[0][1]) * 5.0
                rewards[best_agent] += base_reward + efficiency_bonus

                self.completed_workloads.append(workload)
                self.total_workloads += 1
                allocated = True

            if not allocated:
                # Penalty for failed allocation (SLA violation)
                penalty = -5.0 * workload.priority
                for i in range(self.num_agents):
                    rewards[i] += penalty / self.num_agents  # Shared penalty

                self.failed_workloads.append(workload)
                self.sla_violations += 1
                self.total_workloads += 1

        # Add load balancing reward
        if self.num_agents > 1:
            load_variance = np.var([node.cpu_usage / max(node.cpu_capacity, 1e-6)
                                   for node in self.nodes])
            balance_reward = -2.0 * load_variance  # Penalty for imbalance
            for i in range(self.num_agents):
                rewards[i] += balance_reward

        self.total_reward += sum(rewards)

        return rewards, allocations

    def _update_resource_usage(self, allocations: List[Dict]):
        """Update resource usage based on workload completion"""
        # Decay resource usage over time (workloads complete)
        decay_rate = 0.1
        for node in self.nodes:
            node.cpu_usage = max(0, node.cpu_usage - decay_rate * node.cpu_capacity / 100.0)
            node.memory_usage = max(0, node.memory_usage - decay_rate * node.memory_capacity / 100.0)
            node.bandwidth_usage = max(0, node.bandwidth_usage - decay_rate * node.bandwidth_capacity / 100.0)
            node.storage_usage = max(0, node.storage_usage - decay_rate * node.storage_capacity / 100.0)

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info"""
        completion_rate = (
            len(self.completed_workloads) / max(self.total_workloads, 1)
        )
        sla_violation_rate = (
            self.sla_violations / max(self.total_workloads, 1)
        )
        avg_utilization = np.mean([
            node.cpu_usage / max(node.cpu_capacity, 1e-6)
            for node in self.nodes
        ])

        return {
            'step': self.current_step,
            'total_workloads': self.total_workloads,
            'completed_workloads': len(self.completed_workloads),
            'failed_workloads': len(self.failed_workloads),
            'completion_rate': completion_rate,
            'sla_violation_rate': sla_violation_rate,
            'avg_utilization': avg_utilization,
            'total_reward': self.total_reward,
        }

    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Workloads: {self.total_workloads} total, "
                  f"{len(self.completed_workloads)} completed, "
                  f"{len(self.failed_workloads)} failed")
            print(f"SLA Violations: {self.sla_violations}")
            print(f"Total Reward: {self.total_reward:.2f}")
            print("\nNode States:")
            for node in self.nodes:
                print(f"  Node {node.id}: "
                      f"CPU {node.cpu_usage:.1f}/{node.cpu_capacity:.1f}, "
                      f"Mem {node.memory_usage:.1f}/{node.memory_capacity:.1f}")

    def close(self):
        """Clean up resources"""
        pass


# Test the environment
if __name__ == "__main__":
    env = DistributedResourceEnv(num_agents=5)
    observations, info = env.reset()

    print(f"Environment initialized with {env.num_agents} agents")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run a few test steps
    for step in range(10):
        # Random actions for testing
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        observations, rewards, terminated, truncated, info = env.step(actions)

        print(f"\nStep {step + 1}:")
        print(f"  Rewards: {[f'{r:.2f}' for r in rewards]}")
        print(f"  Completion rate: {info['completion_rate']:.2%}")
        print(f"  SLA violation rate: {info['sla_violation_rate']:.2%}")

        if terminated or truncated:
            break

    env.close()
