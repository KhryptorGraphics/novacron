#!/bin/bash
#
# ML Model Training Execution Script
# Trains all ML models using conda environment
#

set -e

# Add conda to PATH
export PATH="/home/kp/anaconda3/bin:$PATH"

# Activate conda environment
echo "Activating conda environment: ml-training"
source /home/kp/anaconda3/etc/profile.d/conda.sh
conda activate ml-training

# Verify environment
echo "Python: $(python --version)"
echo "TensorFlow: $(python -c 'import tensorflow as tf; print(tf.__version__)' 2>/dev/null || echo 'Not installed')"

# Change to ML directory
cd /home/kp/repos/novacron/backend/ml

# Create checkpoints directory
mkdir -p checkpoints

# Run training orchestrator
echo ""
echo "=========================================="
echo "Starting ML Model Training Suite"
echo "=========================================="
echo ""

python train_all_models.py

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
