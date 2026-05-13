#!/usr/bin/env python3
"""
MADDPG Performance Benchmarking
Compares MADDPG vs Greedy vs Random allocation
"""
import argparse
import numpy as np
import json
import os
import sys
import time
from environment import DistributedResourceEnv
from train import MADDPGTrainer, MATD3Trainer


def greedy_allocate(env, workloads):
    """Greedy allocation: allocate to node with most available resources"""
    allocations = []

    for workload in workloads:
        best_node = -1
        best_available = -1.0

        for i, node in enumerate(env.nodes):
            # Check if node can satisfy workload
            can_allocate = (
                node.cpu_available >= workload.cpu_requirement and
                node.memory_available >= workload.memory_requirement and
                node.bandwidth_available >= workload.bandwidth_requirement and
                node.storage_available >= workload.storage_requirement
            )

            if can_allocate:
                available = (node.cpu_available + node.memory_available +
                           node.bandwidth_available + node.storage_available)
                if available > best_available:
                    best_available = available
                    best_node = i

        if best_node != -1:
            node = env.nodes[best_node]
            node.cpu_usage += workload.cpu_requirement
            node.memory_usage += workload.memory_requirement
            node.bandwidth_usage += workload.bandwidth_requirement
            node.storage_usage += workload.storage_requirement
            allocations.append(best_node)
        else:
            allocations.append(-1)  # Failed

    return allocations


def random_allocate(env, workloads):
    """Random allocation: allocate to random node that can fit"""
    allocations = []

    for workload in workloads:
        candidates = []

        for i, node in enumerate(env.nodes):
            can_allocate = (
                node.cpu_available >= workload.cpu_requirement and
                node.memory_available >= workload.memory_requirement and
                node.bandwidth_available >= workload.bandwidth_requirement and
                node.storage_available >= workload.storage_requirement
            )
            if can_allocate:
                candidates.append(i)

        if candidates:
            best_node = np.random.choice(candidates)
            node = env.nodes[best_node]
            node.cpu_usage += workload.cpu_requirement
            node.memory_usage += workload.memory_requirement
            node.bandwidth_usage += workload.bandwidth_requirement
            node.storage_usage += workload.storage_requirement
            allocations.append(best_node)
        else:
            allocations.append(-1)

    return allocations


def benchmark_algorithm(env, algorithm, trainer=None, num_episodes=100, max_steps=1000, seed=42):
    """Benchmark a single algorithm"""
    if algorithm in ('maddpg', 'matd3') and trainer is None:
        raise ValueError(f"{algorithm.upper()} benchmark requires a trainer with loaded model weights")

    rewards = []
    sla_violations = []
    completion_rates = []
    execution_times = []

    for episode in range(num_episodes):
        episode_seed = seed + episode
        states, _ = env.reset(seed=episode_seed)
        env.action_space.seed(episode_seed)
        episode_reward = 0.0

        start_time = time.time()

        for _ in range(max_steps):
            if algorithm in ('maddpg', 'matd3'):
                actions = [
                    agent.select_action(states[i], add_noise=False)
                    for i, agent in enumerate(trainer.agents)
                ]
            elif algorithm == 'greedy':
                # Greedy actions
                actions = []
                for node in env.nodes:
                    # Greedy: try to use maximum available resources
                    action = np.array([1.0, 1.0, 1.0, 1.0])
                    actions.append(action)
            elif algorithm == 'random':
                # Random actions
                actions = [env.action_space.sample() for _ in range(env.num_agents)]
            else:
                raise ValueError(f"unknown algorithm: {algorithm}")

            states, rewards_step, terminated, truncated, info = env.step(actions)
            episode_reward += sum(rewards_step)

            if terminated or truncated:
                break

        execution_time = time.time() - start_time

        rewards.append(episode_reward)
        sla_violations.append(info['sla_violation_rate'])
        completion_rates.append(info['completion_rate'])
        execution_times.append(execution_time)

    return {
        'avg_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'avg_sla_violation': float(np.mean(sla_violations)),
        'std_sla_violation': float(np.std(sla_violations)),
        'avg_completion_rate': float(np.mean(completion_rates)),
        'std_completion_rate': float(np.std(completion_rates)),
        'avg_execution_time': float(np.mean(execution_times)),
        'std_execution_time': float(np.std(execution_times)),
    }


def calculate_improvements(model_result, baseline_results):
    """Calculate model improvements over each baseline."""
    improvements = {}

    for baseline, baseline_result in baseline_results.items():
        improvements[baseline] = {
            'reward_improvement_pct': (
                (model_result['avg_reward'] - baseline_result['avg_reward']) /
                max(abs(baseline_result['avg_reward']), 1e-6)
            ) * 100,
            'sla_violation_reduction_pct': (
                (baseline_result['avg_sla_violation'] - model_result['avg_sla_violation']) /
                max(baseline_result['avg_sla_violation'], 1e-6)
            ) * 100,
            'completion_improvement_pct': (
                (model_result['avg_completion_rate'] - baseline_result['avg_completion_rate']) /
                max(baseline_result['avg_completion_rate'], 1e-6)
            ) * 100,
        }

    return improvements


def evaluate_acceptance(improvements, target_reward_improvement_pct=20.0):
    """Evaluate whether model improvements meet the configured benchmark target."""
    baselines = {}
    passed = bool(improvements)

    for baseline, improvement in improvements.items():
        reward_improvement = improvement['reward_improvement_pct']
        baseline_passed = reward_improvement >= target_reward_improvement_pct
        baselines[baseline] = {
            'passed': baseline_passed,
            'reward_improvement_pct': reward_improvement,
            'target_reward_improvement_pct': target_reward_improvement_pct,
        }
        passed = passed and baseline_passed

    return {
        'passed': passed,
        'target_reward_improvement_pct': target_reward_improvement_pct,
        'baselines': baselines,
    }


def run_benchmark(model_path=None, model_algorithm='maddpg', num_episodes=100, max_steps=1000,
                  num_agents=10, hidden_dim=256, workload_arrival_rate=5.0, seed=42,
                  target_reward_improvement_pct=20.0,
                  output_path='./models/maddpg/benchmark_results.json'):
    """Run full benchmark comparing all algorithms"""
    print("=" * 80)
    print("MADDPG Performance Benchmark")
    print("=" * 80)

    # Create environment
    env = DistributedResourceEnv(
        num_agents=num_agents,
        workload_arrival_rate=workload_arrival_rate,
        seed=seed
    )

    trainer = None
    if model_path:
        trainer_cls = MATD3Trainer if model_algorithm == 'matd3' else MADDPGTrainer
        trainer = trainer_cls(env, hidden_dim=hidden_dim)
        trainer.load_models(model_path)
        print(f"Loaded {model_algorithm.upper()} model weights from {model_path}")
    else:
        print("No --model-path provided; benchmarking baselines only.")

    # Benchmark algorithms
    algorithms = ['random', 'greedy']
    if trainer is not None:
        algorithms.append(model_algorithm)
    results = {}

    for algo in algorithms:
        print(f"\nBenchmarking {algo.upper()}...")
        results[algo] = benchmark_algorithm(
            env,
            algo,
            trainer=trainer,
            num_episodes=num_episodes,
            max_steps=max_steps,
            seed=seed
        )

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    for algo in algorithms:
        result = results[algo]
        print(f"\n{algo.upper()}:")
        print(f"  Avg Reward:         {result['avg_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"  SLA Violations:     {result['avg_sla_violation']:.2%} ± {result['std_sla_violation']:.2%}")
        print(f"  Completion Rate:    {result['avg_completion_rate']:.2%} ± {result['std_completion_rate']:.2%}")
        print(f"  Execution Time:     {result['avg_execution_time']:.3f}s ± {result['std_execution_time']:.3f}s")

    # Calculate improvements
    print("\n" + "=" * 80)
    print("DRL IMPROVEMENTS OVER BASELINES")
    print("=" * 80)

    model_result = results.get(model_algorithm)
    if model_result is None:
        print("\nDRL model not benchmarked; provide --model-path to compute improvements.")
    else:
        baseline_results = {baseline: results[baseline] for baseline in ['random', 'greedy']}
        improvements = calculate_improvements(model_result, baseline_results)
        results['improvements'] = improvements
        results['acceptance'] = evaluate_acceptance(
            improvements,
            target_reward_improvement_pct=target_reward_improvement_pct
        )

        for baseline, improvement in improvements.items():
            reward_improvement = improvement['reward_improvement_pct']
            sla_improvement = improvement['sla_violation_reduction_pct']
            completion_improvement = improvement['completion_improvement_pct']

            print(f"\n{model_algorithm.upper()} vs {baseline.upper()}:")
            print(f"  Reward Improvement:      {reward_improvement:+.1f}%")
            print(f"  SLA Violation Reduction: {sla_improvement:+.1f}%")
            print(f"  Completion Improvement:  {completion_improvement:+.1f}%")

        status = "PASS" if results['acceptance']['passed'] else "FAIL"
        print(f"\nAcceptance Gate ({target_reward_improvement_pct:.1f}% reward improvement): {status}")

    results['metadata'] = {
        'model_algorithm': model_algorithm,
        'model_path': model_path,
        'num_episodes': num_episodes,
        'max_steps': max_steps,
        'num_agents': num_agents,
        'hidden_dim': hidden_dim,
        'workload_arrival_rate': workload_arrival_rate,
        'seed': seed,
        'target_reward_improvement_pct': target_reward_improvement_pct,
        'algorithms': algorithms,
    }

    # Save results
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✓ Benchmark complete! Results saved to {output_path}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MADDPG resource allocation.")
    parser.add_argument("--model-path", help="Directory containing trained agent_*.pt weights")
    parser.add_argument("--algorithm", choices=["maddpg", "matd3"], default="maddpg")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--workload-arrival-rate", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-reward-improvement", type=float, default=20.0)
    parser.add_argument("--fail-on-target-miss", action="store_true")
    parser.add_argument("--output", default="./models/maddpg/benchmark_results.json")
    args = parser.parse_args()

    results = run_benchmark(
        model_path=args.model_path,
        model_algorithm=args.algorithm,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        hidden_dim=args.hidden_dim,
        workload_arrival_rate=args.workload_arrival_rate,
        seed=args.seed,
        target_reward_improvement_pct=args.target_reward_improvement,
        output_path=args.output
    )

    if args.fail_on_target_miss and not results.get('acceptance', {}).get('passed', False):
        sys.exit(1)
