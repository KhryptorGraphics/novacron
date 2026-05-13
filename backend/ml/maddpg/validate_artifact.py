#!/usr/bin/env python3
"""Validate a trained MADDPG/MATD3 performance artifact."""
import argparse
import json
import os
import sys
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return json.load(f)


def _same_float(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _at_least(value: Any, minimum: int) -> bool:
    try:
        return int(value) >= minimum
    except (TypeError, ValueError):
        return False


def _at_least_float(value: Any, minimum: float) -> bool:
    try:
        return float(value) >= minimum
    except (TypeError, ValueError):
        return False


def validate_performance_artifact(model_root: str,
                                  benchmark_path: str,
                                  target_reward_improvement_pct: float = 20.0,
                                  min_training_episodes: int = 1,
                                  min_benchmark_episodes: int = 1,
                                  min_max_steps: int = 1,
                                  min_num_agents: int = 1) -> Dict[str, Any]:
    """Validate model files, training metadata, benchmark metadata, and gate status."""
    errors: List[str] = []
    model_root = os.path.abspath(model_root)
    benchmark_path = os.path.abspath(benchmark_path)
    metrics_path = os.path.join(model_root, 'metrics.json')

    if not os.path.isdir(model_root):
        errors.append(f"model_root missing: {model_root}")
    if not os.path.exists(metrics_path):
        errors.append(f"metrics.json missing: {metrics_path}")
    if not os.path.exists(benchmark_path):
        errors.append(f"benchmark JSON missing: {benchmark_path}")

    if errors:
        return {'passed': False, 'errors': errors}

    metrics = _load_json(metrics_path)
    benchmark = _load_json(benchmark_path)
    training_meta = metrics.get('metadata', {})
    benchmark_meta = benchmark.get('metadata', {})
    acceptance = benchmark.get('acceptance', {})
    improvements = benchmark.get('improvements', {})

    if not acceptance.get('passed', False):
        errors.append('benchmark acceptance.passed is not true')

    for baseline in ('random', 'greedy'):
        baseline_gate = acceptance.get('baselines', {}).get(baseline)
        if baseline_gate is None:
            errors.append(f"acceptance missing baseline: {baseline}")
            continue
        if not baseline_gate.get('passed', False):
            errors.append(f"{baseline} baseline did not pass reward improvement target")

        gate_improvement = baseline_gate.get('reward_improvement_pct')
        measured_improvement = improvements.get(baseline, {}).get('reward_improvement_pct')
        if not _same_float(gate_improvement, measured_improvement):
            errors.append(
                f"{baseline} acceptance reward improvement {gate_improvement!r} "
                f"does not match measured improvement {measured_improvement!r}"
            )
        if not _at_least_float(gate_improvement, target_reward_improvement_pct):
            errors.append(
                f"{baseline} reward improvement {gate_improvement!r} is below "
                f"target {target_reward_improvement_pct}"
            )

    expected_model_path = os.path.abspath(os.path.join(model_root, 'best'))
    actual_model_path = benchmark_meta.get('model_path')
    if not actual_model_path:
        errors.append('benchmark metadata missing model_path')
    elif os.path.abspath(actual_model_path) != expected_model_path:
        errors.append(
            f"benchmark model_path {actual_model_path} does not match {expected_model_path}"
        )

    model_algorithm = benchmark_meta.get('model_algorithm')
    algorithms = benchmark_meta.get('algorithms', [])
    for algorithm in ('random', 'greedy', model_algorithm):
        if not algorithm:
            continue
        if algorithm not in benchmark:
            errors.append(f"benchmark results missing algorithm: {algorithm}")
        else:
            for metric in ('avg_reward', 'avg_sla_violation', 'avg_completion_rate'):
                try:
                    float(benchmark[algorithm][metric])
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"benchmark result for {algorithm} missing numeric metric: {metric}"
                    )
        if algorithm not in algorithms:
            errors.append(f"benchmark metadata algorithms missing: {algorithm}")

    comparisons = {
        'algorithm': ('algorithm', 'model_algorithm'),
        'num_agents': ('num_agents', 'num_agents'),
        'hidden_dim': ('hidden_dim', 'hidden_dim'),
        'max_steps': ('max_steps', 'max_steps'),
        'seed': ('seed', 'seed'),
        'workload_arrival_rate': ('workload_arrival_rate', 'workload_arrival_rate'),
    }
    for label, (training_key, benchmark_key) in comparisons.items():
        training_value = training_meta.get(training_key)
        benchmark_value = benchmark_meta.get(benchmark_key)
        if isinstance(training_value, float) or isinstance(benchmark_value, float):
            matches = _same_float(training_value, benchmark_value)
        else:
            matches = training_value == benchmark_value
        if not matches:
            errors.append(
                f"metadata mismatch for {label}: training={training_value!r}, "
                f"benchmark={benchmark_value!r}"
            )

    benchmark_target = benchmark_meta.get('target_reward_improvement_pct')
    acceptance_target = acceptance.get('target_reward_improvement_pct')
    if not _same_float(benchmark_target, target_reward_improvement_pct):
        errors.append(
            f"benchmark target {benchmark_target!r} does not match "
            f"{target_reward_improvement_pct}"
        )
    if not _same_float(acceptance_target, target_reward_improvement_pct):
        errors.append(
            f"acceptance target {acceptance_target!r} does not match "
            f"{target_reward_improvement_pct}"
        )

    minimums = {
        'training num_episodes': (training_meta.get('num_episodes'), min_training_episodes),
        'benchmark num_episodes': (benchmark_meta.get('num_episodes'), min_benchmark_episodes),
        'training max_steps': (training_meta.get('max_steps'), min_max_steps),
        'benchmark max_steps': (benchmark_meta.get('max_steps'), min_max_steps),
        'training num_agents': (training_meta.get('num_agents'), min_num_agents),
        'benchmark num_agents': (benchmark_meta.get('num_agents'), min_num_agents),
    }
    for label, (value, minimum) in minimums.items():
        if not _at_least(value, minimum):
            errors.append(f"{label} {value!r} is below required minimum {minimum}")

    num_agents = training_meta.get('num_agents')
    if isinstance(num_agents, int):
        for agent_id in range(num_agents):
            agent_path = os.path.join(expected_model_path, f'agent_{agent_id}.pt')
            if not os.path.exists(agent_path):
                errors.append(f"missing model weight: {agent_path}")
    else:
        errors.append(f"invalid num_agents metadata: {num_agents!r}")

    return {
        'passed': not errors,
        'errors': errors,
        'model_root': model_root,
        'benchmark_path': benchmark_path,
        'training_metadata': training_meta,
        'benchmark_metadata': benchmark_meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MADDPG/MATD3 performance artifacts.")
    parser.add_argument("--model-root", required=True, help="Training output directory")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON path")
    parser.add_argument("--target-reward-improvement", type=float, default=20.0)
    parser.add_argument("--min-training-episodes", type=int, default=1)
    parser.add_argument("--min-benchmark-episodes", type=int, default=1)
    parser.add_argument("--min-max-steps", type=int, default=1)
    parser.add_argument("--min-num-agents", type=int, default=1)
    args = parser.parse_args()

    result = validate_performance_artifact(
        model_root=args.model_root,
        benchmark_path=args.benchmark,
        target_reward_improvement_pct=args.target_reward_improvement,
        min_training_episodes=args.min_training_episodes,
        min_benchmark_episodes=args.min_benchmark_episodes,
        min_max_steps=args.min_max_steps,
        min_num_agents=args.min_num_agents
    )
    print(json.dumps(result, indent=2))
    return 0 if result['passed'] else 1


if __name__ == "__main__":
    sys.exit(main())
