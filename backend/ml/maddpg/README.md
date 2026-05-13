# MADDPG Multi-Agent Reinforcement Learning for Resource Allocation

## Overview

This module implements **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)** and **MATD3 (Multi-Agent TD3)** for intelligent distributed resource allocation in the Novacron system. The performance target is a **20-40% reward improvement** over greedy allocation, enforced by the reproducible benchmark gate below.

## Architecture

### Components

1. **Environment (`environment.py`)**: Gymnasium-compatible multi-agent environment
   - Simulates distributed compute nodes
   - Handles workload arrivals (Poisson process)
   - Tracks SLA violations and resource utilization
   - Provides normalized observations to agents

2. **Training (`train.py`)**: MADDPG and MATD3 implementation
   - Actor-Critic architecture with centralized training, decentralized execution
   - MATD3 twin critics, target policy smoothing, and delayed actor updates
   - Experience replay buffer
   - Ornstein-Uhlenbeck noise for exploration
   - Soft target network updates

3. **Resource Allocator (`allocator.go`)**: Go integration
   - Loads trained PyTorch models
   - Real-time resource allocation using trained agents
   - Performance metrics tracking
   - Thread-safe allocation management

4. **Inference (`inference.py`)**: Model serving
   - Lightweight inference script for Go integration
   - JSON-based state/action communication
   - CPU-optimized inference

## Installation

```bash
cd backend/ml/maddpg
pip install -r requirements.txt
```

## Training

### Quick Start

```bash
# Train MATD3 agents (10,000 episodes)
python train.py --algorithm matd3
```

### Configuration

Use CLI flags to adjust the main training parameters:

```bash
python train.py \
  --algorithm matd3 \
  --episodes 10000 \
  --max-steps 1000 \
  --warmup-episodes 100 \
  --eval-episodes 100 \
  --num-agents 10 \
  --hidden-dim 256 \
  --batch-size 256 \
  --buffer-capacity 100000 \
  --action-prior 0.3 \
  --workload-arrival-rate 5.0 \
  --seed 42 \
  --update-interval 1 \
  --save-dir ./models/matd3
```

### Training Output

```
Episode 1000/10000
  Avg Reward: 1247.32
  SLA Violations: 3.2%
  Completion Rate: 96.8%
  Noise Scale: 0.800
  Buffer Size: 100000
  ✓ New best model saved (reward: 1247.32)
```

Models saved to:
- `./models/<algorithm>/best/` - Best performing model
- `./models/<algorithm>/checkpoint_*/` - Periodic checkpoints
- `./models/<algorithm>/final/` - Final trained model
- `./models/<algorithm>/metrics.json` - Training metrics and run metadata

## Evaluation

### Benchmark Gate

```bash
# Compare a trained model vs greedy and random; exit non-zero if the target is missed.
python benchmark.py \
  --model-path ./models/matd3/best \
  --algorithm matd3 \
  --episodes 100 \
  --max-steps 1000 \
  --num-agents 10 \
  --hidden-dim 256 \
  --workload-arrival-rate 5.0 \
  --seed 42 \
  --target-reward-improvement 20.0 \
  --fail-on-target-miss \
  --output ./models/matd3/benchmark_results.json
```

For a full train-and-gate run:

```bash
./verify_performance.sh
```

This runs training, benchmarks `models/<algorithm>/best`, and validates the resulting
`metrics.json` plus `benchmark_results.json` with `validate_artifact.py`, including
minimum episodes, steps, and agent count for the selected profile.

For a shorter staging profile, override the shell variables instead of editing code:

```bash
EPISODES=1000 MAX_STEPS=300 BATCH_SIZE=64 UPDATE_INTERVAL=4 BENCHMARK_EPISODES=30 ./verify_performance.sh
```

### Testing Trained Model

```python
from train import MADDPGTrainer
from environment import DistributedResourceEnv

# Load environment
env = DistributedResourceEnv(num_agents=10)

# Create trainer and load model
trainer = MADDPGTrainer(env)
trainer.load_models('./models/maddpg/best')

# Evaluate
trainer.evaluate(num_episodes=100, render=False)
```

## Go Integration

### Usage in Go

```go
package main

import (
    "fmt"
    "github.com/novacron/backend/ml/maddpg"
)

func main() {
    // Create nodes
    nodes := []*maddpg.Node{
        {ID: 0, CPUCapacity: 100, MemoryCapacity: 64, ...},
        {ID: 1, CPUCapacity: 150, MemoryCapacity: 96, ...},
        // ... more nodes
    }

    // Initialize allocator with trained model
    allocator, err := maddpg.NewResourceAllocator(
        "./models/maddpg/best",
        nodes,
    )
    if err != nil {
        panic(err)
    }

    // Create workloads
    workloads := []maddpg.Workload{
        {ID: 1, CPURequirement: 20, MemoryRequirement: 8, ...},
        {ID: 2, CPURequirement: 35, MemoryRequirement: 16, ...},
    }

    // Allocate resources
    allocations, err := allocator.AllocateResources(workloads)
    if err != nil {
        panic(err)
    }

    // Print allocations
    for _, alloc := range allocations {
        fmt.Printf("Workload %d -> Node %d\n", alloc.WorkloadID, alloc.NodeID)
    }

    // Get performance metrics
    metrics := allocator.GetMetrics()
    fmt.Printf("Success Rate: %.2f%%\n", metrics.SuccessRate * 100)
    fmt.Printf("SLA Violations: %d\n", metrics.SLAViolations)
}
```

### Performance Report

```go
report := allocator.PerformanceReport()
fmt.Printf("%+v\n", report)
// Output:
// {
//   "total_allocations": 1523,
//   "successful_allocs": 1487,
//   "success_rate": 0.976,
//   "sla_violations": 36,
//   "avg_utilization": 0.847,
//   ...
// }
```

## Algorithm Details

### MADDPG Overview

**Multi-Agent Deep Deterministic Policy Gradient (MADDPG)** extends DDPG to multi-agent settings:

1. **Centralized Training**: Critic sees all agents' states and actions
2. **Decentralized Execution**: Each actor only sees local state
3. **Cooperation**: Agents learn to cooperate through shared rewards

### State Space (per agent)

8-dimensional observation:
```
[cpu_util, mem_util, bw_util, storage_util,
 cpu_avail, mem_avail, bw_avail, storage_avail]
```

All values normalized to [0, 1].

### Action Space (per agent)

4-dimensional continuous actions in [0, 1]:
```
[cpu_allocation_pct, mem_allocation_pct,
 bw_allocation_pct, storage_allocation_pct]
```

### Reward Function

```
reward = base_reward + efficiency_bonus + balance_reward - penalties

where:
  base_reward = 10.0 * workload_priority (successful allocation)
  efficiency_bonus = (1 - resource_waste) * 5.0
  balance_reward = -2.0 * load_variance (across nodes)
  penalties = -5.0 * workload_priority (failed allocation, SLA violation)
```

### Network Architecture

**Actor (Policy Network)**:
```
Input (8) -> FC(256) -> LayerNorm -> ReLU
          -> FC(256) -> LayerNorm -> ReLU
          -> FC(4) -> Sigmoid
```

**Critic (Q-Network)**:
```
Input (8*N + 4*N) -> FC(256) -> LayerNorm -> ReLU
                  -> FC(256) -> LayerNorm -> ReLU
                  -> FC(1)
```

where N = number of agents.

## Performance Target

| Metric | Baseline | Target | Gate |
|--------|----------|--------|------|
| Reward improvement | Greedy | >= 20% | `benchmark.py --fail-on-target-miss` |
| SLA violations | Greedy | Lower is better | Recorded in benchmark JSON |
| Completion rate | Greedy | Higher is better | Recorded in benchmark JSON |
| Utilization | Greedy | Higher is better | Recorded in benchmark JSON |

## Key Features

- **Cooperative Learning**: Agents learn to balance load across the system
- **SLA-Aware**: Prioritizes workloads based on deadlines and priority
- **Scalable**: Handles heterogeneous node capacities
- **Go Integration**: Thread-safe allocator with metrics and injectable predictors
- **Efficient**: Centralized training, decentralized execution
- **Verifiable**: Deterministic seeds, model-backed benchmark, and acceptance gate

## Troubleshooting

### Training Issues

**Problem**: Slow convergence
- Increase `warmup_episodes` for more exploration
- Adjust learning rates (`lr_actor`, `lr_critic`)
- Increase `buffer_capacity` for better sample diversity

**Problem**: Unstable training
- Decrease learning rates
- Increase `batch_size`
- Adjust `tau` for slower target network updates

### Go Integration Issues

**Problem**: Python not found
- Ensure Python 3.7+ is installed
- Add to PATH: `export PATH=$PATH:/usr/bin/python3`

**Problem**: Model loading fails
- Verify model path exists
- Check all agent files present: `agent_0.pt`, `agent_1.pt`, ...

## Future Enhancements

1. **Prioritized Replay**: Sample important transitions more frequently
2. **Hindsight Experience Replay**: Learn from failed allocations
3. **Communication Protocol**: Inter-agent message passing
4. **Transfer Learning**: Pre-train on simpler environments
5. **Production Trace Training**: Train and gate on captured workload traces

## References

- [MADDPG Paper](https://arxiv.org/abs/1706.02275)
- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)

## License

Copyright 2025 Novacron. All rights reserved.
