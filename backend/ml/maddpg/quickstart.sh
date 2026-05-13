#!/bin/bash
# MADDPG Quick Start Script

set -e

echo "========================================="
echo "MADDPG Multi-Agent RL Quick Start"
echo "========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.7+"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Run environment tests
echo "🧪 Testing environment..."
python3 test_environment.py --quiet 2>&1 | grep -E "(OK|FAILED|ERROR)" || true
echo "✓ Environment tests complete"
echo ""

# Run MADDPG tests
echo "🧪 Testing MADDPG components..."
python3 test_maddpg.py --quiet 2>&1 | grep -E "(OK|FAILED|ERROR)" || true
echo "✓ MADDPG tests complete"
echo ""

# Quick training demo
echo "🚀 Running quick MATD3 training demo..."
python3 train.py \
  --algorithm matd3 \
  --episodes 100 \
  --max-steps 200 \
  --warmup-episodes 10 \
  --eval-episodes 20 \
  --num-agents 5 \
  --hidden-dim 128 \
  --batch-size 64 \
  --buffer-capacity 10000 \
  --action-prior 0.3 \
  --workload-arrival-rate 3.0 \
  --seed 42 \
  --update-interval 2 \
  --save-interval 50 \
  --log-interval 20 \
  --save-dir ./models/matd3_demo

echo "📊 Running demo benchmark without acceptance failure..."
python3 benchmark.py \
  --model-path ./models/matd3_demo/best \
  --algorithm matd3 \
  --episodes 10 \
  --max-steps 200 \
  --num-agents 5 \
  --hidden-dim 128 \
  --workload-arrival-rate 3.0 \
  --seed 42 \
  --target-reward-improvement 20.0 \
  --output ./models/matd3_demo/benchmark_results.json

echo ""
echo "========================================="
echo "✓ MADDPG Quick Start Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Train full model: python3 train.py --algorithm matd3"
echo "  2. Run acceptance gate: ./verify_performance.sh"
echo "  3. Integrate with Go: see README.md"
echo ""
