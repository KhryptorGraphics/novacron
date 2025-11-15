#!/bin/bash
# Monitor LSTM Autoencoder training status

echo "Checking training status..."
echo "================================"

# Check if metadata file exists (indicates completion)
METADATA_FILE="/home/kp/repos/novacron/backend/ml/models/consensus/consensus_metadata.json"
REPORT_FILE="/home/kp/repos/novacron/docs/models/consensus_latency_eval.md"

if [ -f "$METADATA_FILE" ]; then
    echo "✓ Training COMPLETED!"
    echo ""
    echo "Model Metadata:"
    cat "$METADATA_FILE"
    echo ""
    echo "================================"

    if [ -f "$REPORT_FILE" ]; then
        echo ""
        echo "Evaluation Report Preview:"
        head -80 "$REPORT_FILE"
    fi
else
    echo "Training in progress..."
    echo ""
    echo "Last 30 lines of training log:"
    tail -30 /tmp/training_output.log 2>/dev/null || echo "Log file not found"
fi
