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

# Quick training demo (100 episodes)
echo "🚀 Running quick training demo (100 episodes)..."
cat > /tmp/maddpg_demo.py << 'EOF'
from environment import DistributedResourceEnv
from train import MADDPGTrainer

# Create environment
env = DistributedResourceEnv(num_agents=5, workload_arrival_rate=3.0, episode_length=200)

# Create trainer
trainer = MADDPGTrainer(env, hidden_dim=128, buffer_capacity=10000, batch_size=64)

# Quick training
print("\nTraining MADDPG agents...")
trainer.train(
    num_episodes=100,
    max_steps=200,
    warmup_episodes=10,
    save_interval=50,
    log_interval=20,
    save_dir='./models/maddpg_demo'
)

# Evaluate
print("\nEvaluating trained model...")
trainer.evaluate(num_episodes=20, render=False)

print("\n✓ Demo complete! Model saved to ./models/maddpg_demo/")
EOF

python3 /tmp/maddpg_demo.py
rm /tmp/maddpg_demo.py

echo ""
echo "========================================="
echo "✓ MADDPG Quick Start Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Train full model: python3 train.py"
echo "  2. Run benchmarks: python3 benchmark.py"
echo "  3. Integrate with Go: see README.md"
echo ""
