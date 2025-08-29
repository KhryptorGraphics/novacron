"""
Test configuration and fixtures for the AI Operations Engine.
"""

import pytest
import asyncio
from unittest.mock import Mock
import pandas as pd
import numpy as np
from typing import Generator

from ai_engine.config import Settings
from ai_engine.core.failure_predictor import FailurePredictionService
from ai_engine.core.workload_optimizer import WorkloadPlacementService  
from ai_engine.core.anomaly_detector import AnomalyDetectionService
from ai_engine.core.resource_optimizer import ResourceOptimizationService


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    settings = Settings()
    settings.database.url = "sqlite:///:memory:"
    settings.redis.url = "redis://localhost:6379/15"  # Use test DB
    settings.ml.model_storage_path = "/tmp/test-models"
    settings.testing = True
    return settings


@pytest.fixture
def sample_metrics_data() -> pd.DataFrame:
    """Create sample metrics data for testing."""
    np.random.seed(42)
    
    data = []
    for i in range(100):
        data.append({
            'timestamp': f'2024-01-01T{i%24:02d}:00:00Z',
            'node_id': f'node-{i%5}',
            'cpu_utilization': np.random.normal(0.5, 0.2),
            'memory_utilization': np.random.normal(0.6, 0.15),
            'disk_utilization': np.random.normal(0.3, 0.1),
            'network_utilization': np.random.normal(0.2, 0.1),
            'temperature': np.random.normal(45, 5),
            'failure_label': 0 if i < 90 else 1  # 10% failures
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_workload_data() -> pd.DataFrame:
    """Create sample workload data for testing."""
    np.random.seed(42)
    
    data = []
    for i in range(50):
        data.append({
            'workload_id': f'workload-{i}',
            'cpu_cores': np.random.randint(1, 8),
            'memory_gb': np.random.randint(2, 16),
            'storage_gb': np.random.randint(50, 500),
            'workload_type': np.random.choice(['web_server', 'database', 'ml_training']),
            'cpu_utilization': np.random.normal(0.6, 0.2),
            'memory_utilization': np.random.normal(0.7, 0.2),
            'node_id': f'node-{i%3}',
            'performance': np.random.normal(0.8, 0.1),
            'cost': np.random.normal(100, 30),
            'efficiency': np.random.normal(0.75, 0.15),
            'sustainability': np.random.normal(0.6, 0.2)
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def mock_failure_service(mock_settings) -> FailurePredictionService:
    """Create mock failure prediction service."""
    service = FailurePredictionService(mock_settings)
    service.active_model = Mock()
    service.active_model.is_trained = True
    service.active_model.metadata.model_id = "test_failure_model"
    return service


@pytest.fixture
def mock_placement_service(mock_settings) -> WorkloadPlacementService:
    """Create mock workload placement service."""
    service = WorkloadPlacementService(mock_settings)
    service.active_model = Mock()
    service.active_model.is_trained = True
    service.active_model.metadata.model_id = "test_placement_model"
    return service


@pytest.fixture
def mock_anomaly_service(mock_settings) -> AnomalyDetectionService:
    """Create mock anomaly detection service."""
    service = AnomalyDetectionService(mock_settings)
    service.active_model = Mock()
    service.active_model.is_trained = True
    service.active_model.metadata.model_id = "test_anomaly_model"
    return service


@pytest.fixture
def mock_resource_service(mock_settings) -> ResourceOptimizationService:
    """Create mock resource optimization service."""
    service = ResourceOptimizationService(mock_settings)
    service.active_model = Mock()
    service.active_model.is_trained = True
    service.active_model.metadata.model_id = "test_resource_model"
    return service


@pytest.fixture
def sample_prediction_request() -> dict:
    """Create sample prediction request."""
    return {
        "request_id": "test_request_123",
        "features": {
            "cpu_utilization": 0.85,
            "memory_utilization": 0.92,
            "disk_utilization": 0.78,
            "network_utilization": 0.45,
            "temperature": 67.5,
            "node_id": "node-001"
        }
    }


@pytest.fixture
def sample_workload_request() -> dict:
    """Create sample workload placement request."""
    return {
        "workload_id": "web-app-01",
        "cpu_cores": 4,
        "memory_gb": 8,
        "storage_gb": 100,
        "workload_type": "web_server",
        "sla_requirements": 0.99
    }


@pytest.fixture
def sample_available_nodes() -> list:
    """Create sample available nodes."""
    return [
        {
            "node_id": "node-001",
            "cpu_cores_available": 8,
            "memory_available": 16,
            "storage_available": 500,
            "cpu_utilization": 0.3,
            "memory_utilization": 0.4,
            "network_latency": 2.5,
            "datacenter_location": "us-east-1"
        },
        {
            "node_id": "node-002", 
            "cpu_cores_available": 4,
            "memory_available": 8,
            "storage_available": 200,
            "cpu_utilization": 0.7,
            "memory_utilization": 0.6,
            "network_latency": 5.0,
            "datacenter_location": "us-west-2"
        }
    ]