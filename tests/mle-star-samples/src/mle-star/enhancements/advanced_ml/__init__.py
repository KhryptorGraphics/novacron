"""
Advanced ML Features
===================

Advanced machine learning capabilities for MLE-Star including:
- Automated hyperparameter optimization with Bayesian methods
- Neural architecture search (NAS) capabilities
- AutoML pipelines with feature selection
- Model interpretability with SHAP and LIME
- Advanced ensemble methods and model stacking
"""

from .bayesian_hpo import BayesianHPO
from .neural_architecture_search import NeuralArchitectureSearch
from .automl_pipeline import AutoMLPipeline
from .model_interpretability import ModelInterpretability
from .ensemble_methods import EnsembleMethods

__all__ = [
    'BayesianHPO',
    'NeuralArchitectureSearch',
    'AutoMLPipeline',
    'ModelInterpretability',
    'EnsembleMethods'
]