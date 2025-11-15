# DWCP Bandwidth Predictor LSTM Training

Train a high-accuracy LSTM neural network to predict network bandwidth, latency, packet loss, and jitter.

**Target Accuracy:** ≥98% (Correlation ≥0.98, MAPE ≤5%, or Accuracy ≥98%)

## Quick Start

### 1. One-Command Training

```bash
./train_bandwidth_predictor.sh
```

This automated script will:
- ✅ Check Python environment
- ✅ Setup virtual environment
- ✅ Install dependencies
- ✅ Validate training data
- ✅ Train the model
- ✅ Export to ONNX format
- ✅ Generate comprehensive reports

### 2. Custom Configuration

```bash
# Set environment variables
export DATA_PATH="/path/to/your/data.csv"
export OUTPUT_DIR="./my_model"
export EPOCHS=300
export BATCH_SIZE=128

# Run training
./train_bandwidth_predictor.sh
```

### 3. Direct Python Execution

```bash
# Install dependencies first
pip install tensorflow numpy pandas scikit-learn tf2onnx

# Run training
python3 train_lstm_enhanced.py \
    --data-path ../../../../../../data/dwcp_training.csv \
    --output-dir ./checkpoints/bandwidth_predictor \
    --epochs 200 \
    --batch-size 64 \
    --window-size 30
```

## Files in This Directory

### Training Scripts

- **`train_lstm_enhanced.py`** - Enhanced LSTM training with attention mechanisms (742 lines)
- **`train_bandwidth_predictor.sh`** - Automated training workflow (309 lines)
- **`train_lstm_optimized.py`** - Alternative optimized training script
- **`train_lstm.py`** - Original baseline training script

### Other Files

- **`README.md`** - This file
- **`setup_training_env.sh`** - Environment setup helper
- **`training_venv/`** - Virtual environment (created automatically)

## Model Architecture

**Enhanced LSTM with Attention:**

```
Input (30 timesteps, 40-50 features)
    ↓
LSTM (256 units) + BatchNorm
    ↓
LSTM (192 units) + BatchNorm
    ↓
LSTM (128 units) + BatchNorm
    ↓
Attention Layer (128 units)
    ↓
Dense (192) → Dense (128) → Skip Connection → Dense (64)
    ↓
Output (4 predictions: bandwidth, latency, packet_loss, jitter)
```

**Key Features:**
- 🎯 Custom attention mechanism for temporal focus
- 🔄 Skip connections for better gradient flow
- 📊 Batch normalization for training stability
- 🛡️ L2 regularization + dropout for generalization
- 🚀 AdamW optimizer with cosine decay
- 📈 40-50 engineered features from 22 raw columns

## Data Format

### Required CSV Columns

```csv
timestamp,region,az,link_type,node_id,peer_id,
rtt_ms,jitter_ms,throughput_mbps,bytes_tx,bytes_rx,
packet_loss,retransmits,congestion_window,queue_depth,
dwcp_mode,network_tier,transport_type,
time_of_day,day_of_week,bandwidth_mbps,latency_ms
```

### Data Requirements

- **Minimum:** 10,000 samples
- **Recommended:** 15,000+ samples (current dataset)
- **Optimal:** 50,000+ samples
- **Format:** Temporal sequence (sorted by timestamp)
- **Quality:** No missing critical values

## Training Configuration

### Default Parameters (Optimized for ≥98% Accuracy)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-path` | `data/dwcp_training.csv` | Training data CSV file |
| `--output-dir` | `./checkpoints/bandwidth_predictor` | Output directory |
| `--epochs` | 200 | Maximum training epochs |
| `--batch-size` | 64 | Training batch size |
| `--learning-rate` | 0.001 | Initial learning rate |
| `--window-size` | 30 | Sequence length (timesteps) |
| `--validation-split` | 0.15 | Validation set (15%) |
| `--test-split` | 0.15 | Test set (15%) |
| `--seed` | 42 | Random seed for reproducibility |

### Environment Variables

```bash
DATA_PATH="/path/to/data.csv"          # Training data location
OUTPUT_DIR="./my_training_run"         # Output directory
EPOCHS=200                             # Training epochs
BATCH_SIZE=64                          # Batch size
LEARNING_RATE=0.001                    # Learning rate
WINDOW_SIZE=30                         # Sequence length
SEED=42                                # Random seed
```

## Output Artifacts

### Generated Files

```
checkpoints/bandwidth_predictor_enhanced/
├── best_model.keras                        # Best model (Keras format)
├── bandwidth_lstm_vYYYYMMDD_HHMMSS.onnx   # ONNX model for production
├── model_metadata_vYYYYMMDD_HHMMSS.json   # Model config & scalers
├── training_history_vYYYYMMDD_HHMMSS.json # Training metrics
├── TRAINING_REPORT.json                    # Comprehensive report
├── training_log.csv                        # Per-epoch metrics
└── tensorboard/                            # TensorBoard logs
```

### Key Artifacts

1. **ONNX Model** (`.onnx`) - Deploy this to production
2. **Metadata** (`.json`) - Contains scaler parameters for normalization
3. **Training Report** (`TRAINING_REPORT.json`) - Evaluation results

## Target Metrics

### Success Criteria (ANY of the following)

✅ **Correlation ≥ 0.98** - Strong linear relationship
✅ **Accuracy ≥ 98%** - High prediction accuracy
✅ **MAPE ≤ 5%** - Low percentage error

### Evaluation Metrics

**Per-Target Metrics:**
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- R² Score
- Correlation Coefficient
- Accuracy Percentage

**Overall Performance:**
- Average across 4 targets (bandwidth, latency, packet_loss, jitter)

## Training Time

| Hardware | Expected Time | Notes |
|----------|---------------|-------|
| CPU (Intel/AMD) | 30-60 min | Baseline performance |
| GPU (NVIDIA CUDA) | 10-20 min | 3-5x speedup |
| GPU (Apple Metal) | 15-25 min | M1/M2 chips |

*Times based on 15,000 samples, 200 epochs*

## Requirements

### Python Version
- Python 3.8 or higher
- Tested with Python 3.12.3

### Dependencies

**Required:**
- `tensorflow>=2.13.0` - Deep learning framework
- `numpy>=1.20.0` - Numerical operations
- `pandas>=1.3.0` - Data manipulation
- `scikit-learn>=1.0.0` - Preprocessing & metrics

**Optional:**
- `tf2onnx>=1.15.0` - ONNX export
- `matplotlib>=3.5.0` - Plotting (if available)
- `seaborn>=0.12.0` - Enhanced plotting

### System Packages (Ubuntu/Debian)

```bash
# Option 1: Use virtual environment (recommended)
sudo apt install python3-venv

# Option 2: Install system packages
sudo apt install python3-tensorflow python3-numpy python3-pandas python3-sklearn
```

## Troubleshooting

### TensorFlow Not Available

**Problem:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solutions:**

1. **Use Docker (Recommended):**
```bash
docker run -it --rm \
  -v $(pwd):/workspace \
  tensorflow/tensorflow:2.15.0 \
  python /workspace/train_lstm_enhanced.py --data-path /workspace/data.csv
```

2. **Install with pip:**
```bash
pip install tensorflow==2.15.0
```

3. **Use system packages:**
```bash
sudo apt install python3-tensorflow
```

### Out of Memory

**Problem:** Training crashes with OOM error

**Solutions:**
- Reduce batch size: `export BATCH_SIZE=32`
- Reduce window size: `export WINDOW_SIZE=20`
- Use CPU instead of GPU: `export CUDA_VISIBLE_DEVICES=""`

### Low Accuracy (<95%)

**Possible Causes:**
1. Insufficient training data
2. Poor data quality
3. Inadequate training time
4. Suboptimal hyperparameters

**Solutions:**
- Collect more data (target 50K+ samples)
- Increase epochs: `export EPOCHS=300`
- Reduce learning rate: `export LEARNING_RATE=0.0005`
- Check data for outliers and missing values

### Training Too Slow

**Solutions:**
- Use GPU if available
- Increase batch size: `export BATCH_SIZE=128`
- Reduce epochs: `export EPOCHS=100`
- Use mixed precision training (automatic with TF 2.15)

## Monitoring Training

### Console Output

Training progress shown in real-time:
```
Epoch 1/200
234/234 [==============================] - 45s 192ms/step
  loss: 0.0234 - mae: 0.1123 - val_loss: 0.0189 - val_mae: 0.0987
```

### TensorBoard (Optional)

```bash
# Start TensorBoard
tensorboard --logdir checkpoints/bandwidth_predictor_enhanced/tensorboard

# Open browser to http://localhost:6006
```

### Training Log

CSV file with per-epoch metrics:
```bash
cat checkpoints/bandwidth_predictor_enhanced/training_log.csv
```

## Integration with Go Predictor

### 1. Deploy ONNX Model

```bash
# Copy trained model to Go project
cp checkpoints/bandwidth_predictor_enhanced/bandwidth_lstm_*.onnx \
   ../../models/bandwidth_predictor.onnx
```

### 2. Update Go Code

```go
// Load model
predictor, err := NewLSTMPredictor("models/bandwidth_predictor.onnx")

// Load metadata for scaler parameters
// See: model_metadata_*.json
```

### 3. Feature Extraction

Update Go code to extract same 40-50 features used in training:
- Rolling statistics
- Rate of change
- Network load metrics
- Categorical encodings

See `train_lstm_enhanced.py` for exact feature engineering logic.

## Advanced Usage

### Custom Feature Engineering

Edit `train_lstm_enhanced.py` → `engineer_features()` method:

```python
def engineer_features(self, df):
    # Add your custom features here
    df['my_custom_feature'] = ...
    return df
```

### Hyperparameter Tuning

Use environment variables or command-line arguments:

```bash
python3 train_lstm_enhanced.py \
    --data-path data.csv \
    --epochs 300 \
    --batch-size 128 \
    --learning-rate 0.0005 \
    --window-size 40
```

### Ensemble Training

Train multiple models with different seeds:

```bash
for seed in 42 43 44 45 46; do
    python3 train_lstm_enhanced.py \
        --data-path data.csv \
        --output-dir ./ensemble/model_$seed \
        --seed $seed
done
```

Then average predictions from all models for higher accuracy.

## Documentation

### Comprehensive Guides

1. **Training Guide:** `docs/ml/BANDWIDTH_PREDICTOR_TRAINING_GUIDE.md`
   - Complete architecture details
   - Training procedures
   - Troubleshooting
   - Integration guide

2. **Training Summary:** `docs/ml/BANDWIDTH_PREDICTOR_TRAINING_SUMMARY.md`
   - Implementation summary
   - Key achievements
   - Next steps

### Code Documentation

All scripts are fully documented with:
- Module docstrings
- Function docstrings
- Inline comments
- Type hints

## Support

### Getting Help

1. Check this README first
2. Review training guide: `docs/ml/BANDWIDTH_PREDICTOR_TRAINING_GUIDE.md`
3. Check training summary: `docs/ml/BANDWIDTH_PREDICTOR_TRAINING_SUMMARY.md`
4. Review error messages and logs
5. File an issue in the project repository

### Common Questions

**Q: Can I use my own data?**
A: Yes! Just ensure it has the required columns and set `DATA_PATH`.

**Q: How long does training take?**
A: 10-60 minutes depending on hardware (see table above).

**Q: What if I don't have a GPU?**
A: Training works fine on CPU, just takes longer (30-60 minutes).

**Q: Can I interrupt training?**
A: Yes, the best model is saved automatically. Resume by continuing training.

**Q: How do I know if training succeeded?**
A: Check `TRAINING_REPORT.json` for `"success": true` and target metrics.

## Examples

### Basic Training

```bash
./train_bandwidth_predictor.sh
```

### Training with Custom Data

```bash
export DATA_PATH="/my/custom/network_data.csv"
./train_bandwidth_predictor.sh
```

### Fast Training (for testing)

```bash
export EPOCHS=50
export BATCH_SIZE=128
./train_bandwidth_predictor.sh
```

### High-Accuracy Training (more epochs)

```bash
export EPOCHS=300
export LEARNING_RATE=0.0005
./train_bandwidth_predictor.sh
```

## License

See project LICENSE file.

## Contributors

- ML Model Developer Agent
- DWCP Team

---

**Last Updated:** 2025-11-14
**Version:** 1.0
**Target:** ≥98% Accuracy

**Ready to Train!** 🚀
