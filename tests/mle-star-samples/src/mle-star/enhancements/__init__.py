"""
MLE-Star Enhanced Framework
==========================

Comprehensive machine learning enhancement suite for the MLE-Star methodology.
Provides extended framework support, advanced ML features, production operations,
and an expanded template library.

Components:
-----------
- framework_extensions: JAX/Flax, MLflow, W&B, Ray Tune, Kubeflow integrations
- advanced_ml: HPO, NAS, AutoML, interpretability, ensemble methods
- production_ops: Model versioning, A/B testing, drift detection, monitoring
- templates: Domain-specific ML templates and workflows

Usage:
------
```python
from mle_star.enhancements import EnhancementRegistry
from mle_star.enhancements.framework_extensions import JAXFlaxIntegration
from mle_star.enhancements.advanced_ml import BayesianHPO

# Register enhancements
registry = EnhancementRegistry()
registry.register('jax_flax', JAXFlaxIntegration())
registry.register('bayesian_hpo', BayesianHPO())

# Use in MLE-Star workflow
enhanced_workflow = registry.enhance_workflow(base_workflow)
```

Architecture:
-------------
- Plugin-based architecture for easy extension
- Configuration-driven workflows
- Backwards compatible with existing MLE-Star
- Comprehensive logging and monitoring
- Production-ready components
"""

__version__ = "2.0.0"
__author__ = "MLE-Star Enhancement Team"

from .core import EnhancementRegistry, BaseEnhancement
from .framework_extensions import *
from .advanced_ml import *
from .production_ops import *
from .templates import *

__all__ = [
    'EnhancementRegistry',
    'BaseEnhancement',
]