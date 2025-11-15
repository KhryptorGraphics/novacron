#!/bin/bash
# TCS-FEEL Calibration Runner

set -e

echo "========================================"
echo "TCS-FEEL Model Calibration"
echo "========================================"

# Setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

echo "✓ Python 3 found: $(python3 --version)"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Run calibration
echo ""
echo "Starting calibration..."
echo "Target: 96.3% accuracy"
echo "Baseline: 86.8% accuracy"
echo ""

python3 calibrate_tcsfeel.py

# Check results
if [ -f "CALIBRATION_REPORT.md" ]; then
    echo ""
    echo "========================================"
    echo "✅ Calibration Complete"
    echo "========================================"
    echo ""
    echo "Report: backend/ml/federated/CALIBRATION_REPORT.md"
    echo "Data: backend/ml/federated/CALIBRATION_REPORT.json"
    echo ""

    # Show summary
    if command -v jq &> /dev/null; then
        echo "Summary:"
        jq '.summary' CALIBRATION_REPORT.json
    fi
else
    echo ""
    echo "❌ Calibration failed - no report generated"
    exit 1
fi
