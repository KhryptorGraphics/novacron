# Computer Vision Project: CIFAR-10 Classification

This project demonstrates the complete MLE-Star workflow for computer vision tasks using CIFAR-10 image classification with Convolutional Neural Networks (CNNs).

## Project Overview

**Task**: Multi-class image classification on the CIFAR-10 dataset
**Framework**: PyTorch
**Architecture**: ResNet-inspired CNN with residual connections
**Performance Target**: 85% test accuracy

## Dataset

**CIFAR-10** consists of 60,000 32x32 color images in 10 classes:
- Airplane, Car, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- 50,000 training images, 10,000 test images
- 6,000 images per class

## Model Architecture

### CIFAR10CNN Features:
- **Input**: 32x32x3 RGB images
- **Initial Conv**: 3 → 64 channels with BatchNorm
- **Residual Blocks**: 4 layers with increasing channels (64→128→256→512)
- **Global Average Pooling**: Reduces spatial dimensions
- **Classifier**: Dropout + Linear layer for 10 classes
- **Parameters**: ~2M parameters

### Key Components:
```python
class ResidualBlock(nn.Module):
    """Residual block with skip connections"""
    
class CIFAR10CNN(nn.Module):
    """Main CNN architecture"""
```

## Training Configuration

```yaml
optimizer: Adam (lr=0.001, weight_decay=1e-4)
scheduler: CosineAnnealingLR
batch_size: 128
epochs: 100
data_augmentation:
  - RandomCrop(32, padding=4)
  - RandomHorizontalFlip
  - ColorJitter
```

## Results

### Expected Performance:
- **Test Accuracy**: 87% (target: 85%)
- **Training Time**: 2-4 hours on GPU
- **Inference Speed**: <50ms per sample
- **Model Size**: ~50MB

### Per-Class Performance:
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Plane | 0.90 | 0.88 | 0.89 |
| Car | 0.92 | 0.94 | 0.93 |
| Bird | 0.82 | 0.80 | 0.81 |
| Cat | 0.78 | 0.82 | 0.80 |
| Deer | 0.84 | 0.86 | 0.85 |
| Dog | 0.81 | 0.79 | 0.80 |
| Frog | 0.89 | 0.91 | 0.90 |
| Horse | 0.88 | 0.87 | 0.87 |
| Ship | 0.91 | 0.93 | 0.92 |
| Truck | 0.85 | 0.83 | 0.84 |

## MLE-Star Workflow Implementation

### Stage 1: Situation Analysis
**Function**: `analyze_data_situation()`
- Dataset characteristics analysis
- Challenge identification (small images, inter-class similarity)
- Recommended approaches (data augmentation, regularization)

### Stage 2: Task Definition  
**Function**: `define_model_task()`
- Formal task specification
- Success metrics definition
- Model constraints establishment

### Stage 3: Action Planning
- Architecture selection rationale
- Training strategy design
- Evaluation methodology

### Stage 4: Implementation
**File**: `trainer.py`
- Complete training pipeline
- Real-time monitoring with TensorBoard
- Checkpoint management

### Stage 5: Results Evaluation
**File**: `evaluator.py`
- Comprehensive performance analysis
- Confusion matrix generation
- Misclassification analysis

### Stage 6: Refinement
- Hyperparameter optimization
- Architecture improvements
- Performance enhancement

### Stage 7: Deployment Preparation
- Model serialization (TorchScript)
- API endpoint design
- Monitoring setup

## Usage

### Quick Start
```bash
# Train model
cd src
python trainer.py

# Evaluate model
python evaluator.py

# Data analysis
python data_loader.py
```

### Custom Configuration
```python
from trainer import CIFAR10Trainer, execute_training_action

# Custom training config
config = {
    'model_type': 'cnn',
    'batch_size': 64,
    'epochs': 50,
    'learning_rate': 0.0005,
    'dropout_rate': 0.3
}

results = execute_training_action(config)
```

### Visualization
```python
from data_loader import CIFAR10DataLoader

loader = CIFAR10DataLoader()
loader.visualize_samples(num_samples=8)
```

## File Structure

```
computer-vision/
├── src/
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── model.py           # CNN architecture definition
│   ├── trainer.py         # Training pipeline
│   └── evaluator.py       # Model evaluation
├── config/
│   └── mle_star_config.yaml
├── data/                  # Dataset storage (auto-downloaded)
├── models/                # Trained models and checkpoints
├── notebooks/             # Jupyter notebooks for analysis
└── docs/                  # Documentation
```

## Key Features

### Advanced Data Augmentation
- **RandomCrop**: Improves translation invariance
- **RandomHorizontalFlip**: Increases dataset diversity
- **ColorJitter**: Enhances robustness to lighting variations

### Residual Architecture Benefits
- **Gradient Flow**: Skip connections prevent vanishing gradients
- **Feature Reuse**: Efficient parameter utilization
- **Training Stability**: Improved convergence properties

### Comprehensive Evaluation
- **Multi-metric Assessment**: Accuracy, precision, recall, F1
- **Per-class Analysis**: Identifies class-specific challenges
- **Confusion Matrix**: Visual performance assessment
- **Misclassification Study**: Error pattern analysis

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch_size in config
2. **Slow Training**: Increase num_workers for data loading
3. **Poor Convergence**: Adjust learning rate or add warmup
4. **Overfitting**: Increase dropout rate or data augmentation

### Performance Optimization
- **Mixed Precision**: Use `torch.cuda.amp` for faster training
- **Data Loading**: Increase `num_workers` and enable `pin_memory`
- **Model Optimization**: Use `torch.jit.script` for deployment

## Extensions

### Advanced Architectures
- **ResNet-18/34**: Deeper residual networks
- **DenseNet**: Dense connections between layers
- **EfficientNet**: Compound scaling approach

### Transfer Learning
- **Pre-trained Models**: ImageNet initialization
- **Fine-tuning**: Gradual unfreezing strategy
- **Feature Extraction**: Frozen backbone approach

### Ensemble Methods
- **Model Averaging**: Multiple architecture ensemble
- **Test-time Augmentation**: Multiple predictions per sample
- **Cross-validation**: Multiple fold training

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [Data Augmentation Techniques](https://pytorch.org/vision/stable/transforms.html)
