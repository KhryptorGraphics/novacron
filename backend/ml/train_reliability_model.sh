#!/bin/bash

# Node Reliability Predictor Training Script
# Target: 85% accuracy

echo "🚀 Starting Node Reliability Predictor Training..."
echo "Target: 85% accuracy"
echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r backend/ml/requirements.txt

# Run training
echo ""
echo "Training model..."
python backend/ml/models/reliability_predictor.py

# Run tests
echo ""
echo "Running unit tests..."
python -m pytest tests/ml/test_reliability_predictor.py -v

# Deactivate virtual environment
deactivate

echo ""
echo "✅ Training complete!"
echo "Model saved to: backend/ml/models/reliability_predictor.weights.h5"
