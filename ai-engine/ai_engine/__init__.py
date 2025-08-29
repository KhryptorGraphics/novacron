"""
NovaCron AI Operations Engine

A comprehensive AI-powered engine for predictive VM management, intelligent workload
placement, anomaly detection, and resource optimization.
"""

__version__ = "0.1.0"
__author__ = "NovaCron Team"
__email__ = "team@novacron.dev"

from .config import Settings
from .core.failure_predictor import FailurePredictionService
from .core.workload_optimizer import WorkloadPlacementService
from .core.anomaly_detector import AnomalyDetectionService
from .core.resource_optimizer import ResourceOptimizationService

__all__ = [
    "Settings",
    "FailurePredictionService",
    "WorkloadPlacementService", 
    "AnomalyDetectionService",
    "ResourceOptimizationService",
]