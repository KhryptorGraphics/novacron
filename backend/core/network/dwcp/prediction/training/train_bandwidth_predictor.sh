#!/bin/bash
# DWCP Bandwidth Predictor Training Script
# Automates the complete training workflow
# Target: ≥98% Accuracy

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================================================"
echo " DWCP BANDWIDTH PREDICTOR LSTM TRAINING"
echo " Target: ≥98% Accuracy (Correlation ≥0.98, MAPE ≤5%, or Accuracy ≥98%)"
echo "========================================================================"
echo ""

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Default configuration
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/dwcp_training.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/bandwidth_predictor}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
WINDOW_SIZE="${WINDOW_SIZE:-30}"
SEED="${SEED:-42}"

# Virtual environment
VENV_DIR="$SCRIPT_DIR/training_venv"

echo "Configuration:"
echo "  Data path:       $DATA_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Epochs:          $EPOCHS"
echo "  Batch size:      $BATCH_SIZE"
echo "  Learning rate:   $LEARNING_RATE"
echo "  Window size:     $WINDOW_SIZE"
echo "  Random seed:     $SEED"
echo ""

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}ERROR: Training data not found: $DATA_PATH${NC}"
    echo ""
    echo "Please ensure the training data is available."
    echo "Expected format: CSV with columns:"
    echo "  timestamp, region, az, link_type, node_id, peer_id,"
    echo "  rtt_ms, jitter_ms, throughput_mbps, bytes_tx, bytes_rx,"
    echo "  packet_loss, retransmits, congestion_window, queue_depth,"
    echo "  dwcp_mode, network_tier, transport_type,"
    echo "  time_of_day, day_of_week, bandwidth_mbps, latency_ms"
    echo ""
    exit 1
fi

# Count data samples
SAMPLE_COUNT=$(wc -l < "$DATA_PATH")
echo -e "${BLUE}Data file found: $SAMPLE_COUNT samples${NC}"
echo ""

# Minimum samples check
if [ "$SAMPLE_COUNT" -lt 1000 ]; then
    echo -e "${YELLOW}WARNING: Less than 1000 samples. Accuracy may be limited.${NC}"
    echo "Recommended: 15,000+ samples for ≥98% accuracy"
    echo ""
fi

# Step 1: Check Python environment
echo "========================================================================"
echo " STEP 1: Checking Python Environment"
echo "========================================================================"
echo ""

PYTHON_BIN="/usr/bin/python3"

if [ ! -x "$PYTHON_BIN" ]; then
    echo -e "${RED}ERROR: Python 3 not found at $PYTHON_BIN${NC}"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_BIN --version 2>&1)
echo -e "${GREEN}Python found: $PYTHON_VERSION${NC}"
echo ""

# Step 2: Setup virtual environment
echo "========================================================================"
echo " STEP 2: Setting Up Virtual Environment"
echo "========================================================================"
echo ""

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at: $VENV_DIR"
    echo "To recreate, run: rm -rf $VENV_DIR"
    echo ""
else
    echo "Creating virtual environment..."

    # Check if python3-venv is available
    if ! $PYTHON_BIN -m venv --help >/dev/null 2>&1; then
        echo -e "${YELLOW}WARNING: python3-venv not available${NC}"
        echo ""
        echo "To install on Ubuntu/Debian:"
        echo "  sudo apt install python3-venv"
        echo ""
        echo "Attempting to continue without venv..."
        echo "Dependencies will need to be installed system-wide or with --break-system-packages"
        echo ""
        VENV_DIR=""
    else
        $PYTHON_BIN -m venv --system-site-packages "$VENV_DIR"
        echo -e "${GREEN}Virtual environment created${NC}"
        echo ""
    fi
fi

# Activate virtual environment if available
if [ -n "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
    PYTHON_BIN="$VENV_DIR/bin/python"
    PIP_BIN="$VENV_DIR/bin/pip"
    echo -e "${GREEN}Virtual environment activated${NC}"
    echo ""
else
    PYTHON_BIN="/usr/bin/python3"
    PIP_BIN="$PYTHON_BIN -m pip"
    echo -e "${YELLOW}Using system Python${NC}"
    echo ""
fi

# Step 3: Install dependencies
echo "========================================================================"
echo " STEP 3: Installing Dependencies"
echo "========================================================================"
echo ""

echo "Checking required packages..."
echo ""

# Check TensorFlow
if $PYTHON_BIN -c "import tensorflow" 2>/dev/null; then
    TF_VERSION=$($PYTHON_BIN -c "import tensorflow; print(tensorflow.__version__)")
    echo -e "${GREEN}✓ TensorFlow $TF_VERSION${NC}"
else
    echo -e "${YELLOW}✗ TensorFlow not found${NC}"
    echo "  Installing TensorFlow 2.15.0..."
    $PIP_BIN install tensorflow==2.15.0 || {
        echo -e "${RED}ERROR: Failed to install TensorFlow${NC}"
        echo ""
        echo "Manual installation options:"
        echo "  1. System packages: sudo apt install python3-tensorflow"
        echo "  2. Break system: $PIP_BIN install --break-system-packages tensorflow"
        echo "  3. Docker: Use TensorFlow Docker image"
        echo ""
        exit 1
    }
fi

# Check NumPy
if $PYTHON_BIN -c "import numpy" 2>/dev/null; then
    echo -e "${GREEN}✓ NumPy${NC}"
else
    echo -e "${YELLOW}✗ NumPy not found${NC}"
    $PIP_BIN install numpy
fi

# Check Pandas
if $PYTHON_BIN -c "import pandas" 2>/dev/null; then
    echo -e "${GREEN}✓ Pandas${NC}"
else
    echo -e "${YELLOW}✗ Pandas not found${NC}"
    $PIP_BIN install pandas
fi

# Check scikit-learn
if $PYTHON_BIN -c "import sklearn" 2>/dev/null; then
    echo -e "${GREEN}✓ scikit-learn${NC}"
else
    echo -e "${YELLOW}✗ scikit-learn not found${NC}"
    $PIP_BIN install scikit-learn
fi

# Check tf2onnx (optional)
if $PYTHON_BIN -c "import tf2onnx" 2>/dev/null; then
    echo -e "${GREEN}✓ tf2onnx${NC}"
else
    echo -e "${YELLOW}✗ tf2onnx not found (optional for ONNX export)${NC}"
fi

echo ""
echo -e "${GREEN}Dependencies ready${NC}"
echo ""

# Step 4: Run training
echo "========================================================================"
echo " STEP 4: Training LSTM Model"
echo "========================================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

TRAINING_SCRIPT="$SCRIPT_DIR/train_lstm_enhanced.py"

if [ ! -f "$TRAINING_SCRIPT" ]; then
    echo -e "${RED}ERROR: Training script not found: $TRAINING_SCRIPT${NC}"
    exit 1
fi

echo "Starting training with enhanced LSTM architecture..."
echo "Output will be saved to: $OUTPUT_DIR"
echo ""
echo "Training command:"
echo "$PYTHON_BIN $TRAINING_SCRIPT \\"
echo "  --data-path $DATA_PATH \\"
echo "  --output-dir $OUTPUT_DIR \\"
echo "  --epochs $EPOCHS \\"
echo "  --batch-size $BATCH_SIZE \\"
echo "  --learning-rate $LEARNING_RATE \\"
echo "  --window-size $WINDOW_SIZE \\"
echo "  --seed $SEED"
echo ""
echo "This may take 10-60 minutes depending on hardware..."
echo ""

$PYTHON_BIN "$TRAINING_SCRIPT" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --window-size "$WINDOW_SIZE" \
    --seed "$SEED"

TRAINING_EXIT_CODE=$?

echo ""
echo "========================================================================"
echo " TRAINING COMPLETE"
echo "========================================================================"
echo ""

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo ""
    echo "Generated artifacts:"
    ls -lh "$OUTPUT_DIR" 2>/dev/null | grep -E "\.(onnx|json|keras|csv)$" || echo "  (artifacts not found)"
    echo ""

    # Check if report exists
    REPORT_FILE="$OUTPUT_DIR/TRAINING_REPORT.json"
    if [ -f "$REPORT_FILE" ]; then
        echo "Training report:"
        $PYTHON_BIN -c "
import json
with open('$REPORT_FILE') as f:
    report = json.load(f)
    print(f\"  Model: {report['model_name']}\")
    print(f\"  Version: {report['version']}\")
    print(f\"  Success: {report['success']}\")
    metrics = report['achieved_metrics']
    print(f\"  Correlation: {metrics['correlation']:.4f}\")
    print(f\"  Accuracy: {metrics['accuracy_percent']:.2f}%\")
    print(f\"  MAPE: {metrics['mape']:.2f}%\")
" 2>/dev/null || cat "$REPORT_FILE"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Review training report: $OUTPUT_DIR/TRAINING_REPORT.json"
    echo "  2. Deploy ONNX model: $OUTPUT_DIR/*.onnx"
    echo "  3. Integrate with Go predictor"
    echo ""
else
    echo -e "${RED}❌ Training failed with exit code $TRAINING_EXIT_CODE${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Check data quality and format"
    echo "  2. Verify dependencies are installed"
    echo "  3. Review error messages above"
    echo "  4. See docs/ml/BANDWIDTH_PREDICTOR_TRAINING_GUIDE.md"
    echo ""
    exit $TRAINING_EXIT_CODE
fi

echo "========================================================================"
echo " Training script completed"
echo "========================================================================"
echo ""
