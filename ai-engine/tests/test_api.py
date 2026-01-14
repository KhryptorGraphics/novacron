"""
Tests for the AI Engine API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock, patch

from ai_engine.api.main import app
from ai_engine.models.base import PredictionResponse


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Mock all AI services."""
    with patch('ai_engine.api.main.failure_service') as failure_mock, \
         patch('ai_engine.api.main.placement_service') as placement_mock, \
         patch('ai_engine.api.main.anomaly_service') as anomaly_mock, \
         patch('ai_engine.api.main.resource_service') as resource_mock:
        
        # Configure failure service mock
        failure_mock.predict_failures = AsyncMock(return_value=PredictionResponse(
            request_id="test_123",
            model_id="failure_model",
            prediction=0,
            confidence=0.8,
            response_time=0.05
        ))
        failure_mock.models = {"failure_v1": Mock()}
        failure_mock.active_model = Mock()
        failure_mock.active_model.metadata.model_id = "failure_v1"
        
        # Configure placement service mock
        placement_mock.optimize_placement = AsyncMock(return_value=[])
        placement_mock.batch_optimize = AsyncMock(return_value={})
        placement_mock.get_placement_factors = Mock(return_value={})
        placement_mock.active_model = Mock()
        placement_mock.active_model.metadata.model_id = "placement_v1"
        
        # Configure anomaly service mock  
        anomaly_mock.detect_anomalies = AsyncMock(return_value=PredictionResponse(
            request_id="test_123",
            model_id="anomaly_model", 
            prediction=0,
            confidence=0.9,
            response_time=0.03
        ))
        anomaly_mock.batch_detect = AsyncMock(return_value=[])
        anomaly_mock.get_anomaly_trends = AsyncMock(return_value={})
        anomaly_mock.active_model = Mock()
        anomaly_mock.active_model.metadata.model_id = "anomaly_v1"
        
        # Configure resource service mock
        resource_mock.get_optimization_recommendations = AsyncMock(return_value=[])
        resource_mock.optimize_single_workload = AsyncMock(return_value=None)
        resource_mock.get_optimization_summary = AsyncMock(return_value={})
        resource_mock.active_model = Mock()
        resource_mock.active_model.metadata.model_id = "resource_v1"
        
        yield {
            'failure': failure_mock,
            'placement': placement_mock,
            'anomaly': anomaly_mock,
            'resource': resource_mock
        }


class TestRootEndpoints:
    """Test root API endpoints."""
    
    def test_root_endpoint(self, client, mock_services):
        """Test root endpoint returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "NovaCron AI Operations Engine"
        assert data["version"] == "0.1.0"
        assert "services" in data
        assert "endpoints" in data
    
    def test_health_check(self, client, mock_services):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert len(data["services"]) == 4  # All AI services


class TestFailurePredictionAPI:
    """Test failure prediction API endpoints."""
    
    def test_predict_failure_success(self, client, mock_services):
        """Test successful failure prediction."""
        request_data = {
            "request_id": "test_123",
            "features": {
                "cpu_utilization": 0.85,
                "memory_utilization": 0.92,
                "temperature": 67.5
            }
        }
        
        response = client.post("/api/v1/failure/predict", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["request_id"] == "test_123"
        assert data["model_id"] == "failure_model"
        assert "prediction" in data
        assert "confidence" in data
    
    def test_predict_failure_invalid_request(self, client, mock_services):
        """Test failure prediction with invalid request."""
        request_data = {
            "request_id": "test_123"
            # Missing features
        }
        
        response = client.post("/api/v1/failure/predict", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_list_failure_models(self, client, mock_services):
        """Test listing failure prediction models."""
        response = client.get("/api/v1/failure/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "models" in data


class TestWorkloadPlacementAPI:
    """Test workload placement API endpoints."""
    
    def test_optimize_placement_success(self, client, mock_services):
        """Test successful workload placement optimization."""
        request_data = {
            "workload_id": "web-app-01",
            "cpu_cores": 4,
            "memory_gb": 8
        }
        available_nodes = [
            {"node_id": "node-001", "cpu_cores_available": 8}
        ]
        
        # Mock the placement service response
        from ai_engine.core.workload_optimizer import PlacementCandidate
        candidate = PlacementCandidate(
            node_id="node-001",
            score=0.85,
            resource_utilization={"cpu": 0.5},
            estimated_performance={"throughput": 1000},
            constraints_satisfied=True,
            reasoning=["Good resource match"]
        )
        mock_services['placement'].optimize_placement.return_value = [candidate]
        
        response = client.post("/api/v1/placement/optimize", json={
            "workload_request": request_data,
            "available_nodes": available_nodes
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "candidates" in data
        assert "recommendation" in data
    
    def test_batch_optimize_placement(self, client, mock_services):
        """Test batch workload placement optimization."""
        workload_requests = [
            {"workload_id": "app-01", "cpu_cores": 2},
            {"workload_id": "app-02", "cpu_cores": 4}
        ]
        available_nodes = [{"node_id": "node-001"}]
        
        response = client.post("/api/v1/placement/batch", json={
            "workload_requests": workload_requests,
            "available_nodes": available_nodes
        })
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
    
    def test_get_placement_factors(self, client, mock_services):
        """Test getting placement factors."""
        mock_services['placement'].get_placement_factors.return_value = {
            "cpu_cores": {"weight": 0.15, "type": "continuous"}
        }
        
        response = client.get("/api/v1/placement/factors")
        assert response.status_code == 200
        
        data = response.json()
        assert "placement_factors" in data
        assert "total_factors" in data


class TestAnomalyDetectionAPI:
    """Test anomaly detection API endpoints."""
    
    def test_detect_anomaly_success(self, client, mock_services):
        """Test successful anomaly detection."""
        request_data = {
            "request_id": "anomaly_123",
            "features": {
                "cpu_utilization": 0.95,
                "network_errors": 25
            }
        }
        
        response = client.post("/api/v1/anomaly/detect", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["request_id"] == "anomaly_123" 
        assert "prediction" in data
        assert "confidence" in data
    
    def test_batch_detect_anomalies(self, client, mock_services):
        """Test batch anomaly detection."""
        data_samples = [
            {"cpu_utilization": 0.5, "memory_utilization": 0.6},
            {"cpu_utilization": 0.9, "memory_utilization": 0.95}
        ]
        
        mock_services['anomaly'].batch_detect.return_value = [
            {"sample_index": 0, "is_anomaly": False},
            {"sample_index": 1, "is_anomaly": True}
        ]
        
        response = client.post("/api/v1/anomaly/batch", json=data_samples)
        assert response.status_code == 200
        
        data = response.json()
        assert "results" in data
        assert "total_samples" in data
        assert data["total_samples"] == 2
    
    def test_get_anomaly_trends(self, client, mock_services):
        """Test getting anomaly trends."""
        mock_services['anomaly'].get_anomaly_trends.return_value = {
            "total_anomalies": 5,
            "anomaly_rate": 2.5,
            "trend": "stable"
        }
        
        response = client.get("/api/v1/anomaly/trends?hours=24")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_anomalies" in data
        assert "anomaly_rate" in data


class TestResourceOptimizationAPI:
    """Test resource optimization API endpoints."""
    
    def test_optimize_resources(self, client, mock_services):
        """Test resource optimization."""
        resource_data = [
            {
                "workload_id": "app-01",
                "cpu_cores": 4,
                "memory_gb": 8,
                "cpu_utilization": 0.3
            }
        ]
        
        from ai_engine.core.resource_optimizer import ResourceRecommendation
        recommendation = ResourceRecommendation(
            action="scale_down",
            target_resources={"cpu_cores": 2, "memory_gb": 4},
            expected_impact={"cost_change": -0.3},
            confidence=0.8,
            reasoning=["Low utilization detected"],
            priority=2
        )
        mock_services['resource'].get_optimization_recommendations.return_value = [recommendation]
        
        response = client.post("/api/v1/resource/optimize", json=resource_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "recommendations" in data
        assert "total_recommendations" in data
    
    def test_optimize_single_workload(self, client, mock_services):
        """Test single workload resource optimization."""
        workload_data = {
            "workload_id": "app-01",
            "cpu_cores": 4,
            "cpu_utilization": 0.2
        }
        
        response = client.post("/api/v1/resource/optimize-single", json=workload_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "recommendation" in data
        assert "workload_id" in data
    
    def test_get_optimization_summary(self, client, mock_services):
        """Test getting optimization summary."""
        mock_services['resource'].get_optimization_summary.return_value = {
            "total_recommendations": 10,
            "potential_savings": 500.0,
            "average_confidence": 0.85
        }
        
        response = client.get("/api/v1/resource/summary?hours=24")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_recommendations" in data
        assert "potential_savings" in data


class TestModelManagementAPI:
    """Test model management API endpoints."""
    
    def test_activate_model_success(self, client, mock_services):
        """Test successful model activation."""
        response = client.post("/api/v1/models/failure/activate?model_id=failure_v2")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "failure_v2" in data["message"]
    
    def test_activate_model_invalid_service(self, client, mock_services):
        """Test model activation with invalid service type."""
        response = client.post("/api/v1/models/invalid/activate?model_id=model_v1")
        assert response.status_code == 400
        
        data = response.json()
        assert "Invalid service type" in data["detail"]
    
    def test_get_model_performance(self, client, mock_services):
        """Test getting model performance metrics."""
        mock_services['failure'].get_model_metrics.return_value = {
            "accuracy": 0.94,
            "precision": 0.89,
            "f1_score": 0.90
        }
        
        response = client.get("/api/v1/models/failure/performance?model_id=failure_v1")
        assert response.status_code == 200
        
        data = response.json()
        assert "metrics" in data
        assert "model_id" in data