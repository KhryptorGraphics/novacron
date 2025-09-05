# Hands-on Lab 1: MLE-Star Fundamentals
## Complete 7-Stage ML Development Workflow

**Duration**: 3-4 hours  
**Difficulty**: Beginner  
**Prerequisites**: Basic Python and ML knowledge  

---

## Lab Overview

In this hands-on lab, you'll implement the complete MLE-Star methodology using a real computer vision project. You'll work through all 7 stages systematically, experiencing the structured approach to ML development that ensures quality and reproducibility.

### Learning Objectives

By the end of this lab, you will be able to:
- Execute all 7 stages of the MLE-Star methodology
- Configure and validate quality gates at each stage
- Use NovaCron's automated workflow orchestration
- Generate comprehensive documentation and artifacts
- Implement proper model validation and deployment preparation

### Lab Environment

- **Platform**: NovaCron Training Cluster
- **Dataset**: CIFAR-10 (32x32 color images, 10 classes)
- **Framework**: PyTorch with NovaCron orchestration
- **Compute**: GPU-enabled training environment

---

## Pre-lab Setup (15 minutes)

### Environment Access
1. Access your assigned NovaCron training environment:
   ```bash
   # Connect to training cluster
   ssh trainee@novacron-lab-[YOUR_ID].training.com
   
   # Activate environment
   conda activate mle-star-lab
   ```

2. Clone the lab repository:
   ```bash
   git clone https://github.com/novacron/mle-star-training-labs.git
   cd mle-star-training-labs/lab-01-fundamentals
   ```

3. Verify environment setup:
   ```bash
   python verify_environment.py
   ```
   **Expected output**: ✅ All dependencies verified

### Initial Project Structure
```
lab-01-fundamentals/
├── src/
│   ├── data_loader.py          # Data handling utilities
│   ├── model.py               # Model architecture
│   ├── trainer.py             # Training pipeline
│   └── evaluator.py           # Evaluation utilities
├── config/
│   └── mle_star_config.yaml   # 7-stage configuration
├── data/                      # Dataset storage
├── models/                    # Model artifacts
├── notebooks/
│   └── exploration.ipynb      # Data exploration
└── docs/                      # Documentation
```

---

## Stage 1: Situation Analysis (30 minutes)

### Objective
Understand the CIFAR-10 dataset characteristics and identify key challenges that will influence your ML approach.

### Tasks

#### 1.1 Dataset Exploration
Execute the situation analysis script:
```bash
python src/situation_analysis.py --output data/situation_analysis.json
```

**Expected Activities**:
- Dataset loading and basic statistics
- Class distribution analysis
- Image characteristics examination
- Data quality assessment
- Challenge identification

#### 1.2 Analysis Review
Open and examine the generated analysis:
```bash
cat data/situation_analysis.json | jq '.summary'
```

**Key Insights to Identify**:
- Are classes balanced? (Should find slight imbalance)
- What's the image resolution challenge? (32x32 is quite small)
- Are there quality issues? (Some images are difficult even for humans)
- What preprocessing might be needed?

#### 1.3 Interactive Exploration
Open the Jupyter notebook for visual exploration:
```bash
jupyter notebook notebooks/exploration.ipynb
```

**Notebook Tasks**:
- Visualize sample images from each class
- Create class distribution charts
- Analyze pixel value distributions
- Identify potential data augmentation needs

### Validation Checkpoint ✅
**Quality Gate**: Complete situation analysis with the following artifacts:
- [ ] `data/situation_analysis.json` generated
- [ ] Identified at least 3 key challenges
- [ ] Documented preprocessing recommendations
- [ ] Visual exploration completed

**Validation Command**:
```bash
python validate_stage.py --stage 1 --artifacts data/situation_analysis.json
```

---

## Stage 2: Task Definition (20 minutes)

### Objective
Transform your understanding into specific ML objectives, success metrics, and constraints.

### Tasks

#### 2.1 Define ML Task Specification
Edit the task definition file:
```bash
nano config/task_definition.yaml
```

**Required Specifications**:
```yaml
task_definition:
  problem_type: "multiclass_classification"
  target_variable: "class_label"
  success_metrics:
    primary: "accuracy"
    target_value: 0.85
    secondary:
      - precision_weighted: 0.83
      - recall_weighted: 0.83
      - f1_score_weighted: 0.83
  
  constraints:
    training_time: "< 4 hours"
    model_size: "< 50MB"
    inference_time: "< 100ms"
    memory_usage: "< 8GB"
  
  quality_requirements:
    min_validation_accuracy: 0.75
    max_overfitting_gap: 0.05
    interpretability_level: "medium"
```

#### 2.2 Generate Task Specification
Run the task definition generator:
```bash
python src/task_definition.py --config config/task_definition.yaml --output config/task_spec.json
```

#### 2.3 Validate Task Feasibility
Execute feasibility analysis:
```bash
python src/feasibility_analysis.py --task config/task_spec.json
```

**Expected Output**: Feasibility report with recommendations

### Validation Checkpoint ✅
**Quality Gate**: Valid task specification meeting requirements:
- [ ] Clear problem definition with measurable objectives
- [ ] Realistic success metrics based on dataset analysis
- [ ] Feasible constraints considering available resources
- [ ] Quality requirements aligned with business needs

---

## Stage 3: Action Planning (45 minutes)

### Objective
Design your model architecture, training strategy, and evaluation approach based on the previous stages.

### Tasks

#### 3.1 Architecture Design
Implement your CNN architecture:
```python
# Edit src/model.py
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CIFAR10CNN, self).__init__()
        
        # TODO: Design your architecture
        # Hint: Consider residual connections for this small image size
        # Recommended: 
        # - Initial conv layer (3 -> 64 channels)
        # - 3-4 residual blocks with increasing channels
        # - Global average pooling
        # - Dropout + final linear layer
        
        pass  # Replace with your implementation
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass
```

**Architecture Requirements**:
- Handle 32x32x3 input images
- Output 10 classes
- Include dropout for regularization
- Use modern architectural patterns (residual connections recommended)
- Target ~1-5M parameters

#### 3.2 Training Strategy Configuration
Configure your training approach:
```yaml
# Edit config/mle_star_config.yaml - training section
training:
  optimizer:
    type: "Adam"
    learning_rate: 0.001
    weight_decay: 0.0001
  
  scheduler:
    type: "CosineAnnealingLR"
    T_max: 100
  
  hyperparameters:
    batch_size: 128
    epochs: 100
    early_stopping: true
    patience: 10
  
  data_augmentation:
    - RandomCrop: {size: 32, padding: 4}
    - RandomHorizontalFlip: {p: 0.5}
    - ColorJitter: {brightness: 0.2, contrast: 0.2}
```

#### 3.3 Generate Action Plan
Create comprehensive action plan:
```bash
python src/action_planning.py --config config/mle_star_config.yaml --output config/action_plan.json
```

#### 3.4 Architecture Validation
Test your architecture design:
```bash
python src/validate_architecture.py --config config/action_plan.json
```

**Validation Checks**:
- Model loads correctly
- Forward pass works with sample data
- Parameter count within limits
- Memory requirements acceptable

### Validation Checkpoint ✅
**Quality Gate**: Complete and validated action plan:
- [ ] Architecture implements required functionality
- [ ] Training strategy addresses identified challenges
- [ ] Resource requirements meet constraints
- [ ] Plan is technically feasible

---

## Stage 4: Implementation (90 minutes)

### Objective
Implement and execute your training pipeline, producing a trained model that meets your success criteria.

### Tasks

#### 4.1 Complete Model Implementation
Finish implementing your model architecture from Stage 3. Here's a reference implementation:

```python
# Complete implementation in src/model.py
class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(CIFAR10CNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.res_block1 = self._make_res_block(64, 64, 2)
        self.res_block2 = self._make_res_block(64, 128, 2) 
        self.res_block3 = self._make_res_block(128, 256, 2)
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
        
    def _make_res_block(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Classification head
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
```

#### 4.2 Execute Training
Start the training process with monitoring:
```bash
# Launch training with real-time monitoring
python src/trainer.py --config config/mle_star_config.yaml --monitor
```

**Training Monitoring**:
- TensorBoard visualization: `tensorboard --logdir models/experiments`
- Real-time metrics in terminal
- Automatic checkpointing every 10 epochs
- Early stopping based on validation accuracy

#### 4.3 Monitor Training Progress
While training runs, monitor key metrics:
```bash
# Check training progress
python src/monitor_training.py --run-id [generated_run_id]

# View TensorBoard (in browser)
# Navigate to: http://localhost:6006
```

**Key Metrics to Watch**:
- Training/validation accuracy curves
- Loss convergence
- Learning rate schedule
- GPU utilization
- Memory usage

#### 4.4 Training Completion Analysis
After training completes (or if you need to stop early):
```bash
# Generate training summary
python src/analyze_training.py --checkpoint models/experiments/latest/
```

### Troubleshooting Common Issues

**Issue: Out of Memory**
```bash
# Reduce batch size and restart
sed -i 's/batch_size: 128/batch_size: 64/' config/mle_star_config.yaml
python src/trainer.py --config config/mle_star_config.yaml --resume
```

**Issue: Slow Convergence**
```bash
# Check if learning rate is too low/high
python src/learning_rate_finder.py --config config/mle_star_config.yaml
```

### Validation Checkpoint ✅
**Quality Gate**: Successfully trained model meeting minimum requirements:
- [ ] Training completed without critical errors
- [ ] Validation accuracy > 75% (minimum threshold)
- [ ] Model checkpoints saved properly
- [ ] Training metrics logged and accessible

---

## Stage 5: Results Evaluation (30 minutes)

### Objective
Comprehensively evaluate your trained model's performance and understand its behavior.

### Tasks

#### 5.1 Comprehensive Model Evaluation
Execute full evaluation suite:
```bash
python src/evaluator.py --checkpoint models/experiments/latest/best_model.pt --output models/evaluation_results.json
```

**Evaluation Components**:
- Overall accuracy and per-class metrics
- Confusion matrix analysis
- ROC curves and AUC scores
- Precision-recall analysis
- Error case examination

#### 5.2 Generate Evaluation Report
Create detailed evaluation report:
```bash
python src/generate_report.py --results models/evaluation_results.json --output docs/evaluation_report.html
```

#### 5.3 Visual Analysis
Generate visualization suite:
```bash
python src/visualize_results.py --results models/evaluation_results.json --output models/visualizations/
```

**Generated Visualizations**:
- Confusion matrix heatmap
- Per-class performance charts
- Sample predictions with confidence scores
- Misclassification analysis
- Learning curves

#### 5.4 Error Analysis
Analyze model failures:
```bash
jupyter notebook notebooks/error_analysis.ipynb
```

**Error Analysis Tasks**:
- Identify most confused classes
- Analyze failure patterns
- Examine difficult examples
- Suggest improvement strategies

### Expected Results Analysis

Review your results against targets:
```python
# Example results interpretation
{
    "overall_accuracy": 0.87,  # ✅ Exceeds 0.85 target
    "per_class_accuracy": {
        "airplane": 0.89,
        "automobile": 0.91,
        "bird": 0.82,     # ⚠️  Lower performance class
        "cat": 0.78,      # ⚠️  Challenging class
        "deer": 0.85,
        "dog": 0.81,      # ⚠️  Often confused with cat
        "frog": 0.88,
        "horse": 0.89,
        "ship": 0.92,
        "truck": 0.90
    },
    "training_time": "2.3 hours",  # ✅ Under 4-hour limit
    "model_size": "42MB"           # ✅ Under 50MB limit
}
```

### Validation Checkpoint ✅
**Quality Gate**: Comprehensive evaluation meeting success criteria:
- [ ] Overall accuracy meets or exceeds target (85%)
- [ ] All per-class accuracies > 70%
- [ ] Comprehensive analysis completed
- [ ] Evaluation artifacts generated

---

## Stage 6: Refinement (45 minutes)

### Objective
Systematically improve your model based on evaluation insights.

### Tasks

#### 6.1 Identify Improvement Opportunities
Based on your evaluation, identify specific areas for improvement:
```bash
python src/improvement_analysis.py --results models/evaluation_results.json
```

**Common Improvement Areas**:
- Classes with low accuracy (bird, cat, dog)
- Overfitting indicators
- Training efficiency
- Hyperparameter optimization opportunities

#### 6.2 Hyperparameter Optimization
Run systematic hyperparameter tuning:
```bash
# Launch hyperparameter optimization
python src/hyperparameter_tuning.py --config config/tuning_config.yaml --trials 20
```

**Tuning Configuration** (`config/tuning_config.yaml`):
```yaml
hyperparameter_tuning:
  method: "optuna"
  objective: "validation_accuracy"
  trials: 20
  
  parameters:
    learning_rate:
      type: "float"
      low: 0.0001
      high: 0.01
      log: true
    
    batch_size:
      type: "categorical" 
      choices: [64, 128, 256]
    
    dropout_rate:
      type: "float"
      low: 0.3
      high: 0.7
    
    weight_decay:
      type: "float"
      low: 0.00001
      high: 0.001
      log: true
```

#### 6.3 Data Augmentation Refinement
Optimize data augmentation based on error analysis:
```bash
# Test different augmentation strategies
python src/augmentation_optimization.py --base-config config/mle_star_config.yaml
```

#### 6.4 Model Architecture Refinement
If needed, make architectural improvements:
```python
# Example: Add attention mechanism for better feature learning
class ImprovedCIFAR10CNN(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(ImprovedCIFAR10CNN, self).__init__()
        # Add spatial attention
        self.attention = SpatialAttention()
        # ... rest of architecture
```

#### 6.5 Retrain with Best Configuration
Train final model with optimized hyperparameters:
```bash
python src/trainer.py --config config/optimized_config.yaml --final-training
```

### Validation Checkpoint ✅
**Quality Gate**: Improved model with documented enhancements:
- [ ] Hyperparameter optimization completed
- [ ] Model performance improved over baseline
- [ ] Refinement strategies documented
- [ ] Final model meets or exceeds all targets

---

## Stage 7: Deployment Preparation (30 minutes)

### Objective
Prepare your model for production deployment with proper optimization and documentation.

### Tasks

#### 7.1 Model Optimization for Deployment
Optimize model for production inference:
```bash
# Convert to TorchScript for deployment
python src/model_optimization.py --checkpoint models/final_model.pt --output models/production/
```

**Optimization Steps**:
- TorchScript compilation
- Model quantization (optional)
- ONNX export for interoperability
- Performance benchmarking

#### 7.2 Create Inference Pipeline
Implement production inference code:
```python
# src/inference_pipeline.py
class CIFAR10InferenceService:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        self.transforms = self._get_transforms()
    
    def predict(self, image):
        """Single image prediction"""
        tensor = self.transforms(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)
        return probabilities.cpu().numpy()[0]
    
    def predict_batch(self, images):
        """Batch prediction for efficiency"""
        # Implementation here
        pass
```

#### 7.3 Generate Deployment Documentation
Create comprehensive deployment package:
```bash
python src/generate_deployment_docs.py --model models/production/ --output docs/deployment/
```

**Generated Documentation**:
- API specification
- Model performance characteristics
- Deployment requirements
- Monitoring recommendations
- Troubleshooting guide

#### 7.4 Performance Benchmarking
Benchmark production model:
```bash
python src/benchmark_model.py --model models/production/cifar10_optimized.pt
```

**Benchmark Metrics**:
- Inference latency (per sample)
- Throughput (samples/second)
- Memory usage
- CPU utilization

### Validation Checkpoint ✅
**Quality Gate**: Production-ready deployment package:
- [ ] Model optimized and exported correctly
- [ ] Inference pipeline implemented and tested
- [ ] Complete deployment documentation generated
- [ ] Performance benchmarks meet requirements

---

## Lab Completion and Assessment (15 minutes)

### Final Validation
Run comprehensive validation across all stages:
```bash
python validate_complete_workflow.py --project-dir . --output validation_report.json
```

### Results Summary
Generate final project summary:
```bash
python generate_lab_summary.py --validation validation_report.json --output lab_summary.html
```

### Expected Learning Outcomes Assessment

**Technical Skills Acquired**:
- [ ] Complete MLE-Star methodology implementation
- [ ] Systematic approach to ML project development  
- [ ] Quality gate validation and documentation
- [ ] Production deployment preparation

**Deliverables Checklist**:
- [ ] All 7 stages completed with artifacts
- [ ] Model exceeds performance targets
- [ ] Complete documentation generated
- [ ] Production deployment package ready

### Next Steps

1. **Review Your Results**: Examine the generated summary and identify areas for further improvement
2. **Advanced Labs**: Proceed to Lab 2 (VM Management) or Lab 3 (Advanced Orchestration)
3. **Real Projects**: Apply MLE-Star methodology to your own ML projects
4. **Community**: Join the NovaCron community to share experiences and learn from others

---

## Troubleshooting Guide

### Common Issues and Solutions

**Problem**: Training takes too long
**Solution**: Reduce dataset size for lab purposes:
```bash
python src/create_subset.py --size 10000 --output data/cifar10_subset/
```

**Problem**: GPU out of memory
**Solution**: Reduce batch size and enable gradient accumulation:
```yaml
training:
  hyperparameters:
    batch_size: 32
    gradient_accumulation_steps: 4
```

**Problem**: Model not converging
**Solution**: Check learning rate and try different optimizers:
```bash
python src/learning_rate_finder.py --config config/mle_star_config.yaml
```

**Problem**: Poor performance on specific classes
**Solution**: Implement class-balanced training:
```yaml
training:
  class_weights: "balanced"
  focal_loss: true
```

### Getting Help

- **Lab Support**: Use `help` command for stage-specific guidance
- **Community Forum**: https://community.novacron.ai/labs
- **Office Hours**: Weekly sessions for live support
- **Documentation**: Comprehensive guides at docs.novacron.ai

---

## Lab Resources

### Code Templates
- Model architecture templates
- Training loop implementations
- Evaluation script templates
- Deployment pipeline examples

### Datasets
- CIFAR-10 (included)
- CIFAR-100 (advanced exercises)
- Custom dataset integration examples

### Reference Solutions
- Complete working implementation
- Alternative approaches and architectures
- Performance optimization examples
- Advanced feature implementations

**Congratulations on completing Lab 1! You've successfully implemented the complete MLE-Star methodology and are ready for more advanced topics.**