"""
NovaCron AI Engine - Machine Learning for VM Management
"""

__version__ = "0.1.0"
__author__ = "NovaCron Team"

# Conditional imports to handle missing dependencies gracefully
try:
    from .app import app
    _app_available = True
except ImportError as e:
    print(f"Warning: FastAPI app not available due to missing dependencies: {e}")
    app = None
    _app_available = False

try:
    from .models import (
        MigrationPredictor, ResourcePredictor, AnomalyDetector, WorkloadPredictor,
        EnhancedResourcePredictor, AdvancedAnomalyDetector,
        SophisticatedMigrationPredictor, EnhancedWorkloadPredictor
    )
    _models_available = True
except ImportError as e:
    print(f"Warning: Some models not available due to missing dependencies: {e}")
    MigrationPredictor = None
    ResourcePredictor = None
    AnomalyDetector = None
    WorkloadPredictor = None
    EnhancedResourcePredictor = None
    AdvancedAnomalyDetector = None
    SophisticatedMigrationPredictor = None
    EnhancedWorkloadPredictor = None
    _models_available = False

# Performance optimizer should always be available as it has minimal dependencies
try:
    from .performance_optimizer import PerformancePredictor, BandwidthOptimizationEngine
    _performance_available = True
except ImportError as e:
    print(f"Warning: Performance optimizer not available: {e}")
    PerformancePredictor = None
    BandwidthOptimizationEngine = None
    _performance_available = False

__all__ = [
    "app",
    "MigrationPredictor",
    "ResourcePredictor",
    "AnomalyDetector",
    "WorkloadPredictor",
    "EnhancedResourcePredictor",
    "AdvancedAnomalyDetector",
    "SophisticatedMigrationPredictor",
    "EnhancedWorkloadPredictor",
    "PerformancePredictor",
    "BandwidthOptimizationEngine"
]