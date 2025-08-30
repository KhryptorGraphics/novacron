"""
NovaCron AI Engine - Machine Learning for VM Management
"""

__version__ = "0.1.0"
__author__ = "NovaCron Team"

from .app import app
from .models import MigrationPredictor, ResourcePredictor, AnomalyDetector

__all__ = [
    "app",
    "MigrationPredictor",
    "ResourcePredictor",
    "AnomalyDetector",
]