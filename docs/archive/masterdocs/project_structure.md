# {{experimentName}}

A machine learning project following the MLE-Star methodology (Model design, Learning pipeline, Evaluation, Systematic testing, Training optimization, Analysis validation, Refinement deployment).

## Project Structure

```
{{experimentName}}/
├── data/                    # Data storage
│   ├── raw/                # Original, immutable data
│   ├── processed/          # Cleaned and processed data
│   └── external/           # Third-party datasets
├── models/                 # Trained models and artifacts
├── notebooks/              # Jupyter notebooks for exploration
│   ├── 01_model_design.ipynb
│   ├── 02_training_pipeline.ipynb
│   ├── 03_model_evaluation.ipynb
│   ├── 04_hyperparameter_tuning.ipynb
│   ├── 05_model_analysis.ipynb
│   └── 06_deployment.ipynb
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── features/          # Feature engineering
│   ├── models/            # Model definitions and training
│   ├── visualization/     # Visualization utilities
│   └── api/               # API endpoints for serving
├── tests/                  # Unit tests
├── configs/                # Configuration files
├── outputs/                # Generated outputs
│   ├── models/            # Saved models
│   ├── figures/           # Generated plots
│   └── reports/           # Analysis reports
├── requirements.txt        # Python dependencies
├── config.yaml            # Main configuration
└── README.md              # This file
```

## MLE-Star Methodology

### 1. Model Design (M)
- Define problem statement and success metrics
- Select appropriate ML algorithms and architectures
- Design model architecture and components

### 2. Learning Pipeline (L)
- Implement data preprocessing and feature engineering
- Create training and validation pipelines
- Set up data loaders and transformation pipelines

### 3. Evaluation (E)
- Define evaluation metrics and validation strategies
- Implement comprehensive model evaluation
- Create performance monitoring and reporting

### 4. Systematic Testing (S)
- Implement unit tests for all components
- Create integration tests for pipelines
- Add data validation and model testing

### 5. Training Optimization (T)
- Implement hyperparameter tuning
- Optimize training procedures and schedules
- Add model selection and ensemble methods

### 6. Analysis Validation (A)
- Perform model interpretability analysis
- Validate model assumptions and behavior
- Generate comprehensive analysis reports

### 7. Refinement Deployment (R)
- Refine model based on analysis results
- Prepare model for deployment
- Create deployment infrastructure and monitoring

## Getting Started

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Project**
   Edit `configs/config.yaml` with your specific settings.

3. **Run MLE-Star Workflow**
   ```bash
   # Initialize project (already done)
   claude-flow automation mle-star status
   
   # Run individual stages
   claude-flow automation mle-star stage model_design
   claude-flow automation mle-star stage learning_pipeline
   
   # Or run complete workflow
   claude-flow automation mle-star run
   ```

## ML Framework: {{mlFramework}}

This project is configured to use **{{mlFramework}}** as the primary ML framework.

### Framework-Specific Setup

{{#if (eq mlFramework "pytorch")}}
**PyTorch Configuration:**
- Version: 1.9.0+
- GPU Support: Available if CUDA is installed
- Key Components: torch, torchvision, torch.nn, torch.optim

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
```
{{/if}}

{{#if (eq mlFramework "tensorflow")}}
**TensorFlow Configuration:**
- Version: 2.6.0+
- GPU Support: Available if CUDA is installed
- Key Components: tf.keras, tf.data, tf.nn

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```
{{/if}}

{{#if (eq mlFramework "scikit-learn")}}
**Scikit-Learn Configuration:**
- Version: 1.0.0+
- CPU-optimized machine learning
- Key Components: sklearn.model_selection, sklearn.metrics

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
```
{{/if}}

## Usage Examples

### Running Individual Stages
```bash
# Model design stage
claude-flow automation mle-star stage model_design --framework {{mlFramework}}

# Training pipeline stage
claude-flow automation mle-star stage learning_pipeline

# Model evaluation
claude-flow automation mle-star stage evaluation_setup
```

### Complete Workflow
```bash
# Run all stages
claude-flow automation mle-star run --continue-on-error

# Check status
claude-flow automation mle-star status

# Validate environment
claude-flow automation mle-star validate
```

### Deployment
```bash
# Deploy as API
claude-flow automation mle-star deploy --service api

# Deploy with Docker
claude-flow automation mle-star deploy --service docker
```

## Configuration

The main configuration is in `configs/config.yaml`. Key settings include:

- **Data paths**: Input and output data directories
- **Model parameters**: Architecture and hyperparameters
- **Training settings**: Batch size, epochs, learning rate
- **Evaluation metrics**: Performance measurement criteria

## Development Workflow

1. **Data Exploration** → `notebooks/01_model_design.ipynb`
2. **Pipeline Development** → `notebooks/02_training_pipeline.ipynb`
3. **Model Training** → `src/models/train.py`
4. **Evaluation** → `notebooks/03_model_evaluation.ipynb`
5. **Optimization** → `notebooks/04_hyperparameter_tuning.ipynb`
6. **Analysis** → `notebooks/05_model_analysis.ipynb`
7. **Deployment** → `notebooks/06_deployment.ipynb`

## Testing

Run tests with:
```bash
pytest tests/
```

Generate coverage report:
```bash
pytest --cov=src tests/
```

## Contributing

1. Follow MLE-Star methodology for all changes
2. Add tests for new functionality
3. Update documentation
4. Run validation before submitting

## License

[Specify your license here]

---

Generated with MLE-Star methodology for systematic ML development.