# MLE-Star Sample Projects

This directory contains comprehensive sample ML projects that demonstrate the complete 7-stage MLE-Star workflow across three different machine learning domains:

- **Computer Vision**: CIFAR-10 image classification with CNN (PyTorch)
- **NLP**: Sentiment analysis with BERT (Transformers/PyTorch) 
- **Tabular Data**: Ensemble learning for classification/regression (Scikit-learn/XGBoost)

## 🎯 Overview

Each project implements all 7 stages of the MLE-Star methodology:

1. **Situation Analysis**: Problem understanding and data characteristics
2. **Task Definition**: Specific ML objectives and success metrics
3. **Action Planning**: Architecture design and training strategy
4. **Implementation**: Model training and development
5. **Results Evaluation**: Performance assessment and analysis
6. **Refinement**: Model optimization and improvement
7. **Deployment Preparation**: Production readiness and documentation

## 📁 Project Structure

```
mle-star-samples/
├── computer-vision/           # CIFAR-10 CNN Project
│   ├── src/
│   │   ├── data_loader.py     # Data loading and preprocessing
│   │   ├── model.py           # CNN model architecture
│   │   ├── trainer.py         # Training pipeline
│   │   └── evaluator.py       # Model evaluation
│   ├── config/
│   │   └── mle_star_config.yaml
│   ├── data/                  # Dataset storage
│   ├── models/                # Model artifacts
│   ├── notebooks/             # Jupyter notebooks
│   └── docs/                  # Documentation
│
├── nlp/                       # BERT Sentiment Analysis
│   ├── src/
│   │   ├── data_processor.py  # Text preprocessing
│   │   ├── bert_model.py      # BERT model implementation
│   │   └── trainer.py         # Training pipeline
│   ├── config/
│   │   └── mle_star_config.yaml
│   └── ...
│
├── tabular-data/              # Ensemble Learning
│   ├── src/
│   │   ├── data_preprocessor.py  # Feature engineering
│   │   └── model_ensemble.py     # Ensemble models
│   ├── config/
│   │   └── mle_star_config.yaml
│   └── ...
│
├── mle_star_workflow_runner.py   # Main orchestration script
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install core ML dependencies
pip install torch torchvision transformers scikit-learn xgboost lightgbm
pip install pandas numpy matplotlib seaborn jupyter
pip install tensorboard mlflow optuna shap

# Install NLP dependencies
pip install nltk datasets tokenizers

# Or install everything from requirements file
pip install -r requirements.txt
```

### 2. Run Complete MLE-Star Workflow

```bash
# Run all projects
python mle_star_workflow_runner.py

# Run specific projects
python mle_star_workflow_runner.py --projects computer-vision nlp

# Run with custom base directory
python mle_star_workflow_runner.py --base-dir /path/to/samples
```

### 3. Run Individual Projects

```bash
# Computer Vision
cd computer-vision/src
python trainer.py

# NLP
cd nlp/src  
python trainer.py

# Tabular Data
cd tabular-data/src
python model_ensemble.py
```

## 📊 Project Details

### Computer Vision (CIFAR-10 Classification)

**Objective**: Multi-class image classification on CIFAR-10 dataset

**Key Features**:
- ResNet-inspired CNN architecture with residual blocks
- Advanced data augmentation pipeline
- Comprehensive evaluation with confusion matrices
- Performance target: 85% accuracy
- Training time: ~2-4 hours on GPU

**MLE-Star Highlights**:
- **Situation**: Analysis of small 32x32 images and class imbalance
- **Task**: 10-class classification with interpretability requirements
- **Action**: Residual architecture with data augmentation strategy
- **Implementation**: PyTorch training with TensorBoard logging
- **Results**: Detailed per-class analysis and error investigation
- **Refinement**: Hyperparameter tuning and architecture optimization
- **Deployment**: TorchScript export and serving preparation

### NLP (BERT Sentiment Analysis)

**Objective**: Binary sentiment classification using pre-trained BERT

**Key Features**:
- Fine-tuned BERT-base-uncased model
- Advanced text preprocessing pipeline
- Attention visualization capabilities
- Performance target: 90% accuracy, 95% ROC-AUC
- Training time: ~1-3 hours on GPU

**MLE-Star Highlights**:
- **Situation**: Text length analysis and sentiment distribution
- **Task**: Binary classification with explainability needs
- **Action**: BERT fine-tuning with optimal hyperparameters
- **Implementation**: Transformers library with gradient clipping
- **Results**: Classification report with attention analysis
- **Refinement**: Learning rate scheduling and sequence length optimization
- **Deployment**: HuggingFace model hub integration

### Tabular Data (Ensemble Learning)

**Objective**: Classification/regression on structured data using ensemble methods

**Key Features**:
- Multiple base models (Random Forest, XGBoost, LightGBM)
- Comprehensive preprocessing pipeline
- Voting and stacking ensemble strategies
- Feature importance analysis and interpretability
- Performance targets: 85% accuracy (classification), 80% R² (regression)

**MLE-Star Highlights**:
- **Situation**: Data quality assessment and feature type analysis
- **Task**: Ensemble learning with interpretability requirements
- **Action**: Multi-algorithm ensemble with feature engineering
- **Implementation**: Scikit-learn pipeline with cross-validation
- **Results**: Model comparison and feature importance analysis
- **Refinement**: Hyperparameter tuning across multiple models
- **Deployment**: Model serialization and API preparation

## 🔧 Configuration

Each project includes a comprehensive YAML configuration file (`mle_star_config.yaml`) that defines:

- **7-Stage Workflow**: Dependencies, duration estimates, and outputs
- **Data Configuration**: Preprocessing, augmentation, and validation strategies
- **Model Architecture**: Parameters, components, and alternatives
- **Training Setup**: Optimizers, schedulers, and hyperparameters
- **Evaluation Metrics**: Primary/secondary metrics and targets
- **Quality Gates**: Success criteria for each stage
- **Deployment Settings**: Export formats and serving configuration

## 📈 Expected Outputs

After running the complete workflow, each project generates:

### Stage Outputs
1. **Situation Analysis**: Data statistics and problem assessment (`data/situation_analysis.json`)
2. **Task Definition**: Formal task specification (`config/task_definition.json`)
3. **Action Planning**: Architecture and strategy documentation (`config/action_plan.json`)
4. **Implementation**: Trained models and checkpoints (`models/implementation_results.json`)
5. **Results Evaluation**: Performance metrics and analysis (`models/evaluation_results.json`)
6. **Refinement**: Optimization results (`models/refinement_results.json`)
7. **Deployment Prep**: Production artifacts (`models/deployment_info.json`)

### Artifacts
- **Models**: Trained model weights and checkpoints
- **Logs**: Training curves and TensorBoard logs
- **Visualizations**: Confusion matrices, learning curves, feature importance
- **Reports**: Comprehensive evaluation and analysis reports
- **Documentation**: API documentation and deployment guides

## 🎯 Performance Benchmarks

| Project | Metric | Target | Expected |
|---------|--------|--------|----------|
| Computer Vision | Accuracy | 85% | 87% |
| NLP | Accuracy | 90% | 91% |
| NLP | ROC-AUC | 95% | 95% |
| Tabular (Classification) | Accuracy | 85% | 88% |
| Tabular (Regression) | R² Score | 80% | 84% |

## 🔍 Key Learning Points

### MLE-Star Methodology Benefits
1. **Systematic Approach**: Structured progression through all ML stages
2. **Quality Gates**: Clear success criteria prevent downstream issues
3. **Documentation**: Comprehensive tracking of decisions and results
4. **Reproducibility**: Detailed configurations enable experiment reproduction
5. **Deployment Readiness**: Built-in preparation for production deployment

### Domain-Specific Insights
1. **Computer Vision**: Importance of data augmentation and residual architectures
2. **NLP**: Benefits of pre-trained models and careful hyperparameter tuning
3. **Tabular Data**: Power of ensemble methods and feature engineering

### Cross-Domain Patterns
1. **Preprocessing**: Critical importance of domain-appropriate data preparation
2. **Evaluation**: Multi-metric assessment provides comprehensive understanding
3. **Optimization**: Systematic refinement yields consistent improvements
4. **Production**: Early deployment consideration improves final model quality

## 🛠 Customization

### Adding New Projects
1. Create project directory with standard structure
2. Implement required source files following existing patterns
3. Create `mle_star_config.yaml` with 7-stage workflow definition
4. Add project to `mle_star_workflow_runner.py`

### Modifying Existing Projects
1. Update configuration files to change parameters
2. Modify source code for different architectures or approaches
3. Adjust quality gates and performance targets
4. Extend evaluation metrics and analysis

### Integration with External Tools
- **MLflow**: Experiment tracking and model registry
- **Weights & Biases**: Advanced experiment visualization
- **TensorBoard**: Training monitoring and visualization
- **Docker**: Containerized deployment preparation
- **Kubernetes**: Scalable serving infrastructure

## 🐛 Troubleshooting

### Common Issues

1. **ImportError**: Install missing dependencies with `pip install -r requirements.txt`
2. **CUDA Issues**: Ensure PyTorch CUDA version matches your GPU drivers
3. **Memory Errors**: Reduce batch sizes in configuration files
4. **Permission Errors**: Ensure write permissions for data/ and models/ directories

### Debug Mode
```bash
# Run with verbose logging
python mle_star_workflow_runner.py --projects computer-vision 2>&1 | tee debug.log
```

### Individual Stage Testing
```python
# Test specific stages
from mle_star_workflow_runner import MLEStarWorkflowRunner

runner = MLEStarWorkflowRunner()
result = runner.execute_stage('computer-vision', '1_situation_analysis', {})
print(result)
```

## 📚 Additional Resources

- [MLE-Star Research Paper](../docs/mle-star-research.md)
- [Integration Analysis](../docs/mle-integration-analysis.md)
- [Performance Analysis](../docs/performance-analysis-report.md)
- [Original NovaCron Documentation](../../README.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements following the MLE-Star methodology
4. Update documentation and tests
5. Submit a pull request

## 📄 License

This project follows the same license as the parent NovaCron project.

---

**Note**: These sample projects are designed for educational and demonstration purposes. For production use, additional considerations around security, scalability, and monitoring should be implemented.
