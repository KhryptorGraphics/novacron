"""
Framework Extensions
===================

Extended framework support for MLE-Star including:
- JAX/Flax integration for advanced neural networks
- MLflow for experiment tracking and model registry
- Weights & Biases integration for experiment monitoring
- Ray Tune for distributed hyperparameter optimization
- Kubeflow for Kubernetes-based ML workflows
"""

from .jax_flax import JAXFlaxIntegration
from .mlflow_integration import MLflowIntegration
from .wandb_integration import WeightsBiasesIntegration
from .ray_tune_integration import RayTuneIntegration
from .kubeflow_integration import KubeflowIntegration

__all__ = [
    'JAXFlaxIntegration',
    'MLflowIntegration', 
    'WeightsBiasesIntegration',
    'RayTuneIntegration',
    'KubeflowIntegration'
]