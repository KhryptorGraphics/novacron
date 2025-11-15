#!/bin/bash
# Setup script for LSTM training environment
# Installs required Python dependencies

set -e

echo "========================================"
echo "Setting up LSTM Training Environment"
echo "========================================"

# Check Python version
python3 --version

# Upgrade pip
echo ""
echo "Upgrading pip..."
python3 -m pip install --upgrade pip --quiet

# Install core dependencies
echo ""
echo "Installing TensorFlow..."
python3 -m pip install tensorflow==2.15.0 --quiet

echo "Installing NumPy and Pandas..."
python3 -m pip install numpy==1.24.3 pandas==2.1.4 --quiet

echo "Installing scikit-learn..."
python3 -m pip install scikit-learn==1.3.2 --quiet

echo "Installing ONNX conversion..."
python3 -m pip install tf2onnx==1.16.1 onnx==1.15.0 --quiet

# Optional plotting dependencies
echo "Installing visualization libraries..."
python3 -m pip install matplotlib==3.8.2 seaborn==0.13.0 --quiet || echo "Warning: Plotting libraries failed"

echo ""
echo "========================================"
echo "✅ Environment setup complete!"
echo "========================================"
echo ""
echo "Installed packages:"
python3 -m pip list | grep -E "(tensorflow|numpy|pandas|scikit|onnx|matplotlib|seaborn)"
echo ""
