"""Resource optimizer must scale numeric features only."""

import pandas as pd

from ai_engine.core.resource_optimizer import ResourceOptimizationModel
from ai_engine.models.base import ModelMetadata, ModelType


def _frame(rows: int) -> pd.DataFrame:
    data = []
    for i in range(rows):
        data.append({
            "cpu_cores": 2.0 + i,
            "memory_gb": 4.0,
            "storage_gb": 100.0,
            "cpu_utilization": 0.2 + (i % 5) * 0.15,
            "memory_utilization": 0.4,
            "storage_utilization": 0.3,
            "workload_type": "web_server",
        })
    return pd.DataFrame(data)


def test_one_row_optimize_ignores_workload_type_label():
    model = ResourceOptimizationModel(ModelMetadata(
        model_id="resource-test",
        model_type=ModelType.RESOURCE_OPTIMIZATION,
        version="1.0.0",
    ))
    features = _frame(10)
    targets = pd.DataFrame({
        "cost": [float(i) for i in range(10)],
        "performance": [0.5] * 10,
    })
    model.train(features, targets)

    recommendations = model.get_optimization_recommendations(_frame(1))

    assert len(recommendations) == 1
    assert "workload_type" not in model._feature_names
