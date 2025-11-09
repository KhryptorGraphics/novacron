# DWCP ML Pipeline - Quick Reference

## Overview

Advanced ML capabilities for distributed VM optimization with AutoML, NAS, HPO, compression, federated learning, and high-performance inference.

## Quick Start

```go
import (
    "github.com/yourusername/novacron/backend/core/ml/automl"
    "github.com/yourusername/novacron/backend/core/ml/inference"
)

// 1. Train AutoML model
config := automl.DefaultAutoMLConfig()
engine := automl.NewAutoMLEngine(config)
bestModel, _ := engine.Fit(ctx, X, y, featureNames)

// 2. Deploy for inference
inferenceEngine := inference.NewInferenceEngine(nil)
inferenceEngine.LoadModel("model", "v1", bestModel.Weights, "automl")

// 3. Predict
prediction, _ := inferenceEngine.Predict(ctx, "model", "v1", input)
```

## Components

1. **AutoML** (`automl/`): Automated model selection and training
2. **NAS** (`nas/`): Neural architecture search
3. **HPO** (`hpo/`): Hyperparameter optimization
4. **Compression** (`compression/`): Model compression
5. **Federated** (`federated/`): Federated learning
6. **Registry** (`registry/`): Model versioning
7. **Inference** (`inference/`): High-performance serving
8. **Features** (`features/`): Feature store
9. **Pipeline** (`pipeline/`): Workflow orchestration
10. **Metrics** (`metrics/`): Performance tracking

## Performance

- AutoML: <30 min convergence
- Inference: <10ms latency
- Compression: 5-10x with <2% loss
- Federated: <20% overhead
- Coverage: >90%

## Documentation

See `docs/DWCP_ML_PIPELINE.md` for complete guide.

## Tests

```bash
go test -v -cover ./backend/core/ml/...
```

## License

Copyright (c) 2025 NovaCron
