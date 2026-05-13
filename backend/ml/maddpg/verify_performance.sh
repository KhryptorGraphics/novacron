#!/usr/bin/env bash
# Train a model-backed allocator and fail unless the benchmark target is met.

set -euo pipefail

ALGORITHM="${ALGORITHM:-matd3}"
EPISODES="${EPISODES:-10000}"
MAX_STEPS="${MAX_STEPS:-1000}"
WARMUP_EPISODES="${WARMUP_EPISODES:-100}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
BENCHMARK_EPISODES="${BENCHMARK_EPISODES:-100}"
NUM_AGENTS="${NUM_AGENTS:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
BUFFER_CAPACITY="${BUFFER_CAPACITY:-100000}"
LR_ACTOR="${LR_ACTOR:-1e-4}"
LR_CRITIC="${LR_CRITIC:-1e-3}"
GAMMA="${GAMMA:-0.99}"
TAU="${TAU:-0.01}"
ACTION_PRIOR="${ACTION_PRIOR:-0.3}"
WORKLOAD_ARRIVAL_RATE="${WORKLOAD_ARRIVAL_RATE:-5.0}"
POLICY_NOISE="${POLICY_NOISE:-0.2}"
NOISE_CLIP="${NOISE_CLIP:-0.5}"
POLICY_DELAY="${POLICY_DELAY:-2}"
UPDATE_INTERVAL="${UPDATE_INTERVAL:-1}"
SEED="${SEED:-42}"
TARGET_REWARD_IMPROVEMENT="${TARGET_REWARD_IMPROVEMENT:-20.0}"
MODEL_ROOT="${MODEL_ROOT:-./models/${ALGORITHM}}"
BENCHMARK_OUTPUT="${BENCHMARK_OUTPUT:-${MODEL_ROOT}/benchmark_results.json}"

echo "Training ${ALGORITHM} model into ${MODEL_ROOT}"
python3 train.py \
  --algorithm "${ALGORITHM}" \
  --episodes "${EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --warmup-episodes "${WARMUP_EPISODES}" \
  --eval-episodes "${EVAL_EPISODES}" \
  --num-agents "${NUM_AGENTS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --batch-size "${BATCH_SIZE}" \
  --buffer-capacity "${BUFFER_CAPACITY}" \
  --lr-actor "${LR_ACTOR}" \
  --lr-critic "${LR_CRITIC}" \
  --gamma "${GAMMA}" \
  --tau "${TAU}" \
  --action-prior "${ACTION_PRIOR}" \
  --workload-arrival-rate "${WORKLOAD_ARRIVAL_RATE}" \
  --policy-noise "${POLICY_NOISE}" \
  --noise-clip "${NOISE_CLIP}" \
  --policy-delay "${POLICY_DELAY}" \
  --update-interval "${UPDATE_INTERVAL}" \
  --seed "${SEED}" \
  --save-dir "${MODEL_ROOT}"

echo "Benchmarking ${MODEL_ROOT}/best against greedy and random baselines"
python3 benchmark.py \
  --model-path "${MODEL_ROOT}/best" \
  --algorithm "${ALGORITHM}" \
  --episodes "${BENCHMARK_EPISODES}" \
  --max-steps "${MAX_STEPS}" \
  --num-agents "${NUM_AGENTS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --workload-arrival-rate "${WORKLOAD_ARRIVAL_RATE}" \
  --seed "${SEED}" \
  --target-reward-improvement "${TARGET_REWARD_IMPROVEMENT}" \
  --fail-on-target-miss \
  --output "${BENCHMARK_OUTPUT}"

python3 validate_artifact.py \
  --model-root "${MODEL_ROOT}" \
  --benchmark "${BENCHMARK_OUTPUT}" \
  --target-reward-improvement "${TARGET_REWARD_IMPROVEMENT}" \
  --min-training-episodes "${EPISODES}" \
  --min-benchmark-episodes "${BENCHMARK_EPISODES}" \
  --min-max-steps "${MAX_STEPS}" \
  --min-num-agents "${NUM_AGENTS}"

echo "Performance target met. Results: ${BENCHMARK_OUTPUT}"
