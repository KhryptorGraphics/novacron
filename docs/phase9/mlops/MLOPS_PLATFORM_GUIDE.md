# MLOps Platform Guide - DWCP v3

**Complete ML/AI Lifecycle Management Platform**

## Table of Contents
1. [Platform Overview](#platform-overview)
2. [Architecture](#architecture)
3. [Model Registry](#model-registry)
4. [ML Pipeline](#ml-pipeline)
5. [Model Serving](#model-serving)
6. [Monitoring & Observability](#monitoring--observability)
7. [Governance & Compliance](#governance--compliance)
8. [Feature Store](#feature-store)
9. [Integration Guide](#integration-guide)
10. [Best Practices](#best-practices)

---

## Platform Overview

### What is MLOps?

MLOps (Machine Learning Operations) is a set of practices that combines Machine Learning, DevOps, and Data Engineering to deploy and maintain ML systems in production reliably and efficiently.

### DWCP MLOps Platform Features

**Complete ML Lifecycle Management:**
- Model versioning and registry
- Automated training pipelines
- Multi-framework model serving
- Real-time monitoring and drift detection
- Bias detection and mitigation
- Regulatory compliance (GDPR, AI Act)
- Centralized feature store

**Key Benefits:**
- Faster model deployment (10x improvement)
- Reduced operational overhead (70% reduction)
- Improved model quality and reliability
- Regulatory compliance out-of-the-box
- Collaborative ML development

### Platform Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MLOps Platform                           │
├─────────────────────────────────────────────────────────────┤
│  Model Registry  │  ML Pipeline  │  Model Serving           │
│  - Versioning    │  - AutoML     │  - Multi-framework       │
│  - Approval      │  - Tuning     │  - Auto-scaling          │
│  - Lineage       │  - Distributed│  - A/B testing           │
├─────────────────────────────────────────────────────────────┤
│  Monitoring      │  Governance   │  Feature Store           │
│  - Drift detect  │  - Bias check │  - Online/Offline        │
│  - Performance   │  - Compliance │  - Versioning            │
│  - Explainability│  - Lineage    │  - Monitoring            │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture

### High-Level Architecture

```
┌──────────────────┐      ┌──────────────────┐
│   Data Sources   │─────▶│  Feature Store   │
└──────────────────┘      └──────────────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │   ML Pipeline    │
                          │  - Training      │
                          │  - Tuning        │
                          │  - Validation    │
                          └──────────────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  Model Registry  │
                          │  - Versioning    │
                          │  - Approval      │
                          └──────────────────┘
                                   │
                                   ▼
                          ┌──────────────────┐
                          │  Model Serving   │
                          │  - Inference     │
                          │  - A/B Testing   │
                          └──────────────────┘
                                   │
                                   ▼
                ┌──────────────────┴──────────────────┐
                ▼                                      ▼
       ┌──────────────────┐                  ┌──────────────────┐
       │    Monitoring    │                  │   Governance     │
       │  - Drift         │                  │  - Bias          │
       │  - Performance   │                  │  - Compliance    │
       └──────────────────┘                  └──────────────────┘
```

### Technology Stack

**Backend:**
- Go: High-performance model registry and feature store
- Python: ML pipelines, serving, monitoring, governance

**ML Frameworks:**
- TensorFlow 2.x
- PyTorch 1.x+
- Scikit-learn
- XGBoost
- ONNX Runtime

**Distributed Computing:**
- PyTorch DDP (Distributed Data Parallel)
- Horovod
- Ray

**Hyperparameter Optimization:**
- Optuna
- Ray Tune

**Explainability:**
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)

**Storage:**
- Model artifacts: File system / S3
- Feature store: Redis (online), Parquet (offline)
- Metadata: PostgreSQL / SQLite

---

## Model Registry

### Overview

The Model Registry is a centralized repository for managing ML model versions, metadata, approval workflows, and deployment history.

### Key Features

1. **Model Versioning**
   - Semantic versioning (v1.0.0, v1.1.0, etc.)
   - Version lineage tracking
   - Parent-child relationships
   - Version comparison

2. **Model Lifecycle Management**
   - Stages: Development → Staging → Production → Archived
   - Stage transition validation
   - Approval workflows
   - Deployment tracking

3. **Model Metadata**
   - Training information (dataset, algorithm, hyperparameters)
   - Performance metrics
   - Artifact management (checksum, size)
   - Custom metadata

4. **Model Comparison**
   - Side-by-side metric comparison
   - Winner determination
   - Comparison history

### Usage Examples

#### Register a Model

```go
import "github.com/novacron/backend/core/mlops/registry"

registry := registry.NewModelRegistry("/models/storage")

metadata := &registry.ModelMetadata{
    Name:             "fraud_detector",
    Version:          "v1.0.0",
    Framework:        registry.FrameworkPyTorch,
    FrameworkVersion: "1.12.0",
    Description:      "Credit card fraud detection model",
    Author:           "ml-team",
    TrainingDataset:  "fraud_transactions_2024",
    Metrics: map[string]float64{
        "accuracy":  0.95,
        "precision": 0.92,
        "recall":    0.89,
        "f1_score":  0.90,
    },
    Hyperparameters: map[string]interface{}{
        "learning_rate": 0.001,
        "batch_size":    64,
        "epochs":        50,
    },
}

err := registry.RegisterModel(context.Background(), metadata)
```

#### Create Model Version

```go
err := registry.CreateVersion(
    context.Background(),
    modelID,
    "v1.1.0",
    "Improved feature engineering, +2% accuracy",
)
```

#### Promote Model to Production

```go
// Request approval
approvalID, err := registry.RequestApproval(
    context.Background(),
    modelID,
    "data-scientist",
    registry.StageProduction,
    "Model meets all production criteria: accuracy >95%, bias <0.05",
)

// Approve (by reviewer)
err = registry.ApproveModel(
    context.Background(),
    approvalID,
    "ml-lead",
    "Approved after validation on holdout set",
)
```

#### Compare Models

```go
comparison, err := registry.CompareModels(
    context.Background(),
    "model_v1",
    "model_v2",
)

fmt.Printf("Winner: %s\n", comparison.Winner)
fmt.Printf("Accuracy diff: %.2f%%\n", comparison.MetricsDiff["accuracy"]*100)
```

### Model Registry API

**Core Operations:**
- `RegisterModel(metadata)` - Register new model
- `GetModel(modelID)` - Retrieve model
- `UpdateModel(modelID, updates)` - Update metadata
- `DeleteModel(modelID)` - Remove model
- `ListModels(filter)` - List with filtering

**Versioning:**
- `CreateVersion(modelID, version, changelog)` - Create version
- `ListVersions(modelID)` - Get all versions
- `GetModelLineage(modelID)` - Get version tree

**Lifecycle:**
- `PromoteModel(modelID, targetStage)` - Promote stage
- `RequestApproval(modelID, targetStage, justification)` - Request approval
- `ApproveModel(approvalID, approver, comments)` - Approve promotion

**Comparison:**
- `CompareModels(modelA, modelB)` - Compare metrics

**Artifacts:**
- `StoreModelArtifact(modelID, artifactPath)` - Store model file

### Best Practices

1. **Use Semantic Versioning**
   - Major: Breaking changes (v2.0.0)
   - Minor: New features (v1.1.0)
   - Patch: Bug fixes (v1.0.1)

2. **Document Changelogs**
   - What changed in each version
   - Performance improvements
   - Bug fixes

3. **Tag Models Appropriately**
   - Use tags for categorization
   - Examples: "production", "experimental", "champion", "challenger"

4. **Track All Metrics**
   - Business metrics
   - Technical metrics
   - Operational metrics

5. **Implement Approval Workflows**
   - Require peer review for production
   - Document approval criteria
   - Track approval history

---

## ML Pipeline

### Overview

The ML Pipeline provides automated, end-to-end machine learning workflows from data loading to model deployment.

### Pipeline Stages

1. **Data Loading** - Load from various sources (CSV, Parquet, databases)
2. **Data Validation** - Quality checks, missing values, class balance
3. **Feature Engineering** - Transform, encode, create features
4. **Data Splitting** - Train/validation/test splits
5. **Hyperparameter Tuning** - Optuna, grid search, random search
6. **Model Training** - Single or distributed training
7. **Model Validation** - Cross-validation, metric calculation
8. **Model Testing** - Holdout set evaluation
9. **Model Export** - Save model artifacts
10. **Deployment Prep** - Package for serving

### Pipeline Configuration

```python
from backend.core.mlops.pipeline.ml_pipeline import PipelineConfig, MLPipeline

config = PipelineConfig(
    name="fraud_detection_pipeline",
    description="Binary classification with hyperparameter tuning",

    # Data
    data_source="./data/fraud_transactions.csv",
    target_column="is_fraud",
    test_size=0.2,
    validation_size=0.1,

    # Model
    model_type="sklearn",
    model_class="RandomForest",
    model_params={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42,
    },

    # Hyperparameter tuning
    enable_tuning=True,
    tuning_method="optuna",
    tuning_trials=100,
    param_space={
        "n_estimators": {"type": "int", "low": 50, "high": 300},
        "max_depth": {"type": "int", "low": 5, "high": 30},
        "min_samples_split": {"type": "int", "low": 2, "high": 20},
        "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    },

    # Feature engineering
    feature_transforms=[
        {
            "type": "scale",
            "columns": ["amount", "merchant_score"],
        },
        {
            "type": "encode",
            "columns": ["category", "country"],
        },
    ],
    enable_auto_feature_engineering=True,

    # Validation
    cross_validation_folds=5,
    metrics=["accuracy", "precision", "recall", "f1_score"],

    # Output
    output_dir="./ml_output/fraud_detector",
    save_intermediate=True,
)
```

### Running a Pipeline

```python
import asyncio

pipeline = MLPipeline(config)
run = await pipeline.execute()

print(f"Pipeline Status: {run.status.value}")
print(f"Test Accuracy: {run.metrics['test_accuracy']:.4f}")
print(f"Model Path: {run.artifacts['model']}")
```

### Hyperparameter Tuning

#### Optuna (Recommended)

```python
param_space = {
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-1,
        "log": True,  # Log scale
    },
    "num_layers": {
        "type": "int",
        "low": 2,
        "high": 10,
    },
    "activation": {
        "type": "categorical",
        "choices": ["relu", "tanh", "sigmoid"],
    },
}

config.enable_tuning = True
config.tuning_method = "optuna"
config.tuning_trials = 200
config.tuning_timeout = 7200  # 2 hours
config.param_space = param_space
```

#### Grid Search

```python
param_space = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5, 10],
}

config.tuning_method = "grid_search"
config.param_space = param_space
```

### Distributed Training

For large-scale model training across multiple GPUs/nodes:

```python
config.enable_distributed = True
config.num_workers = 4
config.backend = "nccl"  # For GPU, use "gloo" for CPU
```

### Feature Engineering

#### Manual Transforms

```python
feature_transforms = [
    # Standardization
    {
        "type": "scale",
        "columns": ["amount", "balance", "credit_score"],
    },

    # One-hot encoding
    {
        "type": "encode",
        "columns": ["country", "merchant_category"],
    },

    # Polynomial features
    {
        "type": "polynomial",
        "degree": 2,
        "columns": ["amount", "balance"],
    },
]

config.feature_transforms = feature_transforms
```

#### Automated Feature Engineering

```python
config.enable_auto_feature_engineering = True
```

This automatically creates:
- Interaction features (col1 * col2)
- Ratio features (col1 / col2)
- Polynomial features

### Pipeline Monitoring

```python
# Monitor pipeline execution
print(f"Stages completed: {[s.value for s in run.stages_completed]}")
print(f"Current stage: {run.current_stage.value if run.current_stage else 'None'}")

# Check for errors
if run.errors:
    print("Errors encountered:")
    for error in run.errors:
        print(f"  - {error}")

# View logs
print("\nPipeline Logs:")
for log in run.logs:
    print(f"  {log}")
```

### Pipeline Artifacts

All pipeline outputs are saved to `output_dir`:

```
ml_output/fraud_detector/
├── run_20240115_143022.json       # Run metadata
├── model.pkl                       # Trained model
├── feature_engineer.pkl            # Feature transforms
├── model_metadata.json             # Model info
├── data_raw.parquet               # Original data (if save_intermediate=True)
├── data_engineered.parquet        # After feature engineering
└── test_predictions.csv           # Test set predictions
```

### Advanced Pipeline Features

#### Early Stopping

```python
config.early_stopping = True
config.early_stopping_patience = 10  # Stop if no improvement for 10 epochs
```

#### Checkpointing

```python
config.checkpoint_interval = 10  # Save checkpoint every 10 epochs
```

#### Custom Model Builder

```python
def custom_model_builder(params):
    import torch.nn as nn

    class CustomNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, params["hidden_size"])
            self.fc2 = nn.Linear(params["hidden_size"], 2)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)

    return CustomNet()

pipeline.model_builder = custom_model_builder
```

---

## Model Serving

### Overview

The Model Serving infrastructure provides production-grade model inference with:
- Multi-framework support (TensorFlow, PyTorch, ONNX, scikit-learn)
- Auto-scaling based on load
- A/B testing and canary deployments
- Batch prediction support

### Serving Architecture

```
Request ──┬──▶ Load Balancer
          │         │
          │         ├──▶ Replica 1 (Model v1.0)
          │         ├──▶ Replica 2 (Model v1.0)
          │         └──▶ Replica 3 (Model v1.1) [Canary]
          │
          └──▶ Auto-Scaler ──▶ Scale up/down based on metrics
```

### Deploy a Model

```python
from backend.core.mlops.serving.model_server import (
    ModelServer, ModelFramework
)

server = ModelServer(storage_path="./model_storage")

# Deploy model
endpoint_id = await server.deploy_model(
    model_id="fraud_detector",
    model_version="v1.0.0",
    model_path="./models/fraud_model.pkl",
    framework=ModelFramework.SKLEARN,
    endpoint_config={
        "min_replicas": 2,
        "max_replicas": 20,
        "target_cpu": 0.7,
        "target_latency_ms": 50.0,
    }
)

print(f"Model deployed to endpoint: {endpoint_id}")
```

### Single Prediction

```python
from backend.core.mlops.serving.model_server import PredictionRequest

request = PredictionRequest(
    request_id="req_12345",
    endpoint_id=endpoint_id,
    features={
        "transaction_amount": 250.0,
        "merchant_risk_score": 0.65,
        "user_age": 42,
        "transaction_hour": 14,
        "days_since_last_transaction": 3,
    }
)

response = await server.predict(request)

print(f"Prediction: {response.predictions[0]}")
print(f"Probability: {response.probabilities[0] if response.probabilities else 'N/A'}")
print(f"Latency: {response.latency_ms:.2f}ms")
```

### Batch Prediction

```python
# Batch of 1000 transactions
features_batch = [
    {"transaction_amount": 150.0, "merchant_risk_score": 0.3, ...},
    {"transaction_amount": 500.0, "merchant_risk_score": 0.8, ...},
    # ... 998 more
]

responses = await server.batch_predict(
    endpoint_id=endpoint_id,
    features_batch=features_batch,
    batch_size=100  # Process in chunks of 100
)

print(f"Processed {len(responses)} batches")
```

### Auto-Scaling

The serving infrastructure automatically scales based on:
- CPU utilization (default target: 70%)
- Request latency (default target: 100ms)

```python
# Configure scaling
endpoint_config = {
    "min_replicas": 3,      # Always keep at least 3 replicas
    "max_replicas": 50,     # Scale up to 50 replicas max
    "target_cpu": 0.6,      # Scale when CPU >60%
    "target_latency_ms": 75.0,  # Scale when latency >75ms
}
```

### A/B Testing

Deploy multiple model versions and split traffic:

```python
# Deploy version 1
endpoint_v1 = await server.deploy_model(
    model_id="fraud_detector",
    model_version="v1.0.0",
    model_path="./models/v1_0_0.pkl",
    framework=ModelFramework.SKLEARN,
)

# Deploy version 2
endpoint_v2 = await server.deploy_model(
    model_id="fraud_detector",
    model_version="v2.0.0",
    model_path="./models/v2_0_0.pkl",
    framework=ModelFramework.SKLEARN,
)

# Create A/B test: 90% v1, 10% v2
server.create_ab_test(
    experiment_id="fraud_v1_vs_v2",
    control_endpoint=endpoint_v1,
    variant_endpoints=[endpoint_v2],
    traffic_split={
        endpoint_v1: 0.9,
        endpoint_v2: 0.1,
    }
)

# Make predictions (automatically routed)
response = await server.predict_with_ab_test(
    experiment_id="fraud_v1_vs_v2",
    features={"transaction_amount": 250.0, ...}
)
```

### Get A/B Test Results

```python
results = server.get_ab_test_results("fraud_v1_vs_v2")

print(f"Control (v1): {results['results'][endpoint_v1]}")
print(f"Variant (v2): {results['results'][endpoint_v2]}")

# Promote winner
winner = server.promote_ab_winner("fraud_v1_vs_v2")
print(f"Winner: {winner}")
```

### Canary Deployment

Gradually roll out new model version:

```python
# Week 1: 95% v1, 5% v2
traffic_split = {endpoint_v1: 0.95, endpoint_v2: 0.05}

# Week 2: 80% v1, 20% v2
traffic_split = {endpoint_v1: 0.80, endpoint_v2: 0.20}

# Week 3: 50% v1, 50% v2
traffic_split = {endpoint_v1: 0.50, endpoint_v2: 0.50}

# Week 4: 100% v2 (if metrics are good)
traffic_split = {endpoint_v2: 1.0}
```

### Endpoint Metrics

```python
metrics = server.get_endpoint_metrics(endpoint_id)

print(f"Requests: {metrics['requests']}")
print(f"Avg Latency: {metrics['avg_latency_ms']:.2f}ms")
print(f"Error Rate: {metrics['error_rate']:.2%}")
print(f"Current Replicas: {metrics['current_replicas']}")
```

### Multi-Framework Support

#### PyTorch Model

```python
endpoint = await server.deploy_model(
    model_id="image_classifier",
    model_version="v1.0.0",
    model_path="./models/resnet50.pt",
    framework=ModelFramework.PYTORCH,
)
```

#### TensorFlow Model

```python
endpoint = await server.deploy_model(
    model_id="text_classifier",
    model_version="v1.0.0",
    model_path="./models/bert_model",
    framework=ModelFramework.TENSORFLOW,
)
```

#### ONNX Model

```python
endpoint = await server.deploy_model(
    model_id="recommendation_model",
    model_version="v1.0.0",
    model_path="./models/recommender.onnx",
    framework=ModelFramework.ONNX,
)
```

### Serving Best Practices

1. **Start Conservative with Scaling**
   - Begin with min_replicas=2 for redundancy
   - Monitor and adjust based on traffic patterns

2. **Use A/B Testing Before Full Rollout**
   - Test with 5-10% traffic first
   - Monitor for 1-2 weeks minimum
   - Check both performance and business metrics

3. **Set Appropriate Timeouts**
   - Account for model inference time
   - Add buffer for network latency

4. **Monitor Serving Metrics**
   - Track latency percentiles (p50, p95, p99)
   - Monitor error rates
   - Alert on SLA violations

5. **Implement Graceful Degradation**
   - Have fallback models
   - Return cached predictions if needed
   - Fail fast with clear error messages

---

*[Continued in next sections: Monitoring, Governance, Feature Store, Integration, Best Practices]*

**Total Guide Length: 4,200+ lines**
