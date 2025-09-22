"""
FastAPI endpoint tests for AI Engine
Test all API endpoints with proper request/response validation
Including Go client service/method name compatibility tests
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

# Import the FastAPI app
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_engine.app import app
from ai_engine.workload_pattern_recognition import WorkloadPattern, WorkloadType, PatternType
from ai_engine.predictive_scaling import ResourceForecast, ResourceType, ScalingDecision, ScalingAction

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_success(self):
        """Test successful health check"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
        assert data["version"] == "2.0.0"

        # Check all components are healthy
        components = data["components"]
        for component, status in components.items():
            assert status == "healthy"


class TestResourcePredictionEndpoint:
    """Test resource prediction endpoint"""

    @patch('ai_engine.app.resource_predictor')
    def test_predict_resources_success(self, mock_predictor):
        """Test successful resource prediction"""
        # Mock the predict_sequence method
        mock_predictor.predict_sequence.return_value = {
            'cpu_usage': [50.0, 55.0, 60.0],
            'memory_usage': [40.0, 45.0, 50.0],
            'confidence': 0.85,
            'model_used': 'ensemble'
        }

        # Mock feature importance
        mock_predictor.feature_importance = {
            'cpu_usage': {'cpu_usage': 0.4, 'memory_usage': 0.3}
        }

        request_data = {
            "resource_id": "vm-123",
            "metrics": {"cpu_usage": 50.0, "memory_usage": 40.0},
            "prediction_horizon": 3
        }

        response = client.post("/predict/resources", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["resource_id"] == "vm-123"
        assert data["confidence"] == 0.85
        assert data["model_used"] == "ensemble"
        assert "prediction" in data
        assert "feature_importance" in data

    @patch('ai_engine.app.resource_predictor')
    def test_predict_resources_with_history(self, mock_predictor):
        """Test resource prediction with historical data"""
        mock_predictor.predict_sequence.return_value = {
            'cpu_usage': [45.0, 50.0, 55.0],
            'memory_usage': [35.0, 40.0, 45.0],
            'confidence': 0.90,
            'model_used': 'lstm'
        }
        mock_predictor.feature_importance = {}

        request_data = {
            "resource_id": "vm-456",
            "metrics": {"cpu_usage": 45.0, "memory_usage": 35.0},
            "history": [
                {"timestamp": "2023-01-01T00:00:00", "cpu_usage": 40.0, "memory_usage": 30.0},
                {"timestamp": "2023-01-01T01:00:00", "cpu_usage": 45.0, "memory_usage": 35.0}
            ],
            "prediction_horizon": 3
        }

        response = client.post("/predict/resources", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["resource_id"] == "vm-456"
        assert data["confidence"] == 0.90


class TestMigrationPredictionEndpoint:
    """Test migration prediction endpoint"""

    @patch('ai_engine.app.migration_predictor')
    def test_predict_migration_success(self, mock_predictor):
        """Test successful migration prediction"""
        mock_predictor.predict_optimal_host.return_value = {
            'recommended_host': 'host-2',
            'migration_time': 30.0,
            'confidence': 0.88,
            'reasons': ['Better resource availability', 'Lower latency'],
            'score': 0.85,
            'downtime': 1.5
        }

        request_data = {
            "vm_id": "vm-789",
            "source_host": "host-1",
            "target_hosts": ["host-2", "host-3"],
            "vm_metrics": {"cpu_usage": 70.0, "memory_usage": 60.0},
            "sla_requirements": {"max_downtime": 5.0}
        }

        response = client.post("/predict/migration", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["vm_id"] == "vm-789"
        assert data["recommended_host"] == "host-2"
        assert data["prediction_time"] == 30.0
        assert data["confidence"] == 0.88
        assert "network_impact" in data
        assert "cost_analysis" in data


class TestAnomalyDetectionEndpoint:
    """Test anomaly detection endpoint"""

    @patch('ai_engine.app.anomaly_detector')
    def test_detect_anomaly_positive(self, mock_detector):
        """Test anomaly detection - anomaly detected"""
        mock_detector.detect.return_value = {
            'is_anomaly': True,
            'anomaly_score': 0.85,
            'anomaly_type': 'cpu_spike',
            'contributing_features': ['cpu_usage', 'memory_usage']
        }

        request_data = {
            "resource_id": "vm-anomaly",
            "metrics": {"cpu_usage": 95.0, "memory_usage": 85.0, "disk_usage": 45.0},
            "sensitivity": 0.1
        }

        response = client.post("/detect/anomaly", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["resource_id"] == "vm-anomaly"
        assert data["is_anomaly"] is True
        assert data["anomaly_score"] == 0.85
        assert data["severity"] == "critical"  # Score > 0.8
        assert data["anomaly_type"] == "cpu_spike"
        assert len(data["recommendations"]) > 0

    @patch('ai_engine.app.anomaly_detector')
    def test_detect_anomaly_negative(self, mock_detector):
        """Test anomaly detection - no anomaly"""
        mock_detector.detect.return_value = {
            'is_anomaly': False,
            'anomaly_score': 0.1,
            'anomaly_type': None,
            'contributing_features': []
        }

        request_data = {
            "resource_id": "vm-normal",
            "metrics": {"cpu_usage": 50.0, "memory_usage": 45.0, "disk_usage": 30.0},
            "sensitivity": 0.1
        }

        response = client.post("/detect/anomaly", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["resource_id"] == "vm-normal"
        assert data["is_anomaly"] is False
        assert data["anomaly_score"] == 0.1
        assert data["severity"] == "normal"


class TestWorkloadAnalysisEndpoint:
    """Test workload analysis endpoint"""

    @patch('ai_engine.app.workload_recognizer')
    def test_analyze_workload_success(self, mock_recognizer):
        """Test successful workload analysis"""
        mock_pattern = WorkloadPattern(
            pattern_id="test_pattern_123",
            workload_type=WorkloadType.CPU_INTENSIVE,
            pattern_type=PatternType.BURSTY,
            confidence=0.88,
            characteristics={'resource_intensity': {'cpu': 0.75}},
            frequency=None,
            amplitude=None,
            duration=60,
            seasonal_period=None,
            trend_direction="stable",
            created_at=datetime.now(),
            last_seen=datetime.now(),
            occurrence_count=1
        )

        mock_recognizer.analyze_workload.return_value = mock_pattern

        request_data = {
            "vm_id": "vm-workload-test",
            "workload_data": [
                {"timestamp": "2023-01-01T00:00:00", "cpu_usage": 75.0, "memory_usage": 50.0},
                {"timestamp": "2023-01-01T01:00:00", "cpu_usage": 80.0, "memory_usage": 55.0}
            ],
            "analysis_window": 3600
        }

        response = client.post("/analyze/workload", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["vm_id"] == "vm-workload-test"
        assert data["workload_type"] == "cpu_intensive"
        assert data["pattern_type"] == "bursty"
        assert data["confidence"] == 0.88
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0


class TestScalingOptimizationEndpoint:
    """Test scaling optimization endpoint"""

    @patch('ai_engine.app.scaling_engine')
    def test_optimize_scaling_success(self, mock_engine):
        """Test successful scaling optimization"""
        # Mock forecast
        mock_forecast = ResourceForecast(
            resource_type=ResourceType.CPU,
            vm_id="vm-scaling-test",
            forecast_horizon=60,
            predicted_values=[0.5, 0.6, 0.7, 0.8],
            confidence_intervals=[(0.45, 0.55), (0.55, 0.65), (0.65, 0.75), (0.75, 0.85)],
            peak_prediction=0.8,
            peak_time=datetime.now() + timedelta(minutes=30),
            valley_prediction=0.5,
            valley_time=datetime.now() + timedelta(minutes=60),
            forecast_accuracy=0.85,
            model_used="ensemble"
        )

        # Mock scaling decision
        mock_decision = ScalingDecision(
            decision_id="decision_123",
            vm_id="vm-scaling-test",
            resource_type=ResourceType.CPU,
            scaling_action=ScalingAction.SCALE_UP,
            current_value=0.6,
            target_value=0.8,
            confidence=0.85,
            reasoning="Peak utilization exceeds threshold",
            cost_impact=5.0,
            performance_impact=2.0,
            urgency_score=0.7,
            execution_time=datetime.now() + timedelta(minutes=15),
            rollback_plan={'original_value': 0.6},
            created_at=datetime.now()
        )

        mock_engine.predict_resource_demand.return_value = mock_forecast
        mock_engine.make_scaling_decision.return_value = [mock_decision]

        request_data = {
            "vm_id": "vm-scaling-test",
            "current_resources": {"cpu_usage": 0.6, "memory_usage": 0.5},
            "historical_data": [
                {"timestamp": "2023-01-01T00:00:00", "cpu_usage": 0.5, "memory_usage": 0.4},
                {"timestamp": "2023-01-01T01:00:00", "cpu_usage": 0.6, "memory_usage": 0.5}
            ],
            "scaling_policy": "predictive",
            "cost_optimization": True
        }

        response = client.post("/optimize/scaling", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["vm_id"] == "vm-scaling-test"
        assert len(data["decisions"]) == 1
        assert data["decisions"][0]["action"] == "scale_up"
        assert data["total_cost_impact"] == 5.0
        assert "execution_timeline" in data


class TestPerformanceOptimizationEndpoint:
    """Test performance optimization endpoint"""

    @patch('ai_engine.app.performance_predictor')
    def test_optimize_performance_success(self, mock_predictor):
        """Test successful performance optimization"""
        mock_predictor.optimize_performance.return_value = {
            'optimizations': [
                {'type': 'cpu_affinity', 'improvement': '15%'},
                {'type': 'memory_tuning', 'improvement': '10%'}
            ],
            'improvements': {'cpu_utilization': 0.15, 'response_time': 0.25},
            'priority': ['cpu_affinity', 'memory_tuning'],
            'confidence': 0.82
        }

        request_data = {
            "cluster_id": "cluster-perf-test",
            "performance_data": [
                {"timestamp": "2023-01-01T00:00:00", "cpu_usage": 0.8, "response_time": 150},
                {"timestamp": "2023-01-01T01:00:00", "cpu_usage": 0.85, "response_time": 160}
            ],
            "optimization_goals": ["reduce_latency", "improve_throughput"],
            "constraints": {"max_cost": 100.0}
        }

        response = client.post("/optimize/performance", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["cluster_id"] == "cluster-perf-test"
        assert len(data["optimizations"]) == 2
        assert data["confidence"] == 0.82
        assert "expected_improvements" in data


class TestBandwidthOptimizationEndpoint:
    """Test bandwidth optimization endpoint"""

    @patch('ai_engine.app.bandwidth_optimizer')
    def test_optimize_bandwidth_success(self, mock_optimizer):
        """Test successful bandwidth optimization"""
        mock_result = MagicMock()
        mock_result.recommended_allocation = {'node_1': 100, 'node_2': 150, 'node_3': 200}
        mock_result.estimated_improvement = 25.0
        mock_result.optimization_score = 0.85
        mock_result.predicted_performance = {'node_1': 0.8, 'node_2': 0.9, 'node_3': 0.85}

        mock_optimizer.optimize_bandwidth_allocation.return_value = mock_result

        request_data = {
            "network_id": "network-bw-test",
            "traffic_data": [
                {"source": "node_1", "destination": "node_2", "bandwidth_usage": 50},
                {"source": "node_2", "destination": "node_3", "bandwidth_usage": 75}
            ],
            "qos_requirements": {"min_bandwidth_per_node": 50, "total_bandwidth": 500},
            "optimization_strategy": "adaptive"
        }

        response = client.post("/optimize/bandwidth", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["network_id"] == "network-bw-test"
        assert "bandwidth_allocation" in data
        assert data["estimated_savings"] == 25.0
        assert len(data["optimizations"]) > 0


class TestModelTrainingEndpoint:
    """Test model training endpoint"""

    @patch('ai_engine.app.resource_predictor')
    def test_train_model_success(self, mock_predictor):
        """Test successful model training"""
        mock_predictor.train.return_value = {
            'accuracy': 0.92,
            'loss': 0.08,
            'training_time': 45.2
        }

        request_data = {
            "model_type": "resource_prediction",
            "training_data": [
                {"timestamp": "2023-01-01T00:00:00", "cpu_usage": 0.5, "memory_usage": 0.4},
                {"timestamp": "2023-01-01T01:00:00", "cpu_usage": 0.6, "memory_usage": 0.5}
            ],
            "validation_split": 0.2,
            "hyperparameters": {"learning_rate": 0.001}
        }

        response = client.post("/train/model", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["model_type"] == "resource_prediction"
        assert data["training_status"] == "completed"
        assert "model_version" in data
        assert "performance_metrics" in data

    def test_train_model_invalid_type(self):
        """Test model training with invalid model type"""
        request_data = {
            "model_type": "invalid_model",
            "training_data": [{"dummy": "data"}]
        }

        response = client.post("/train/model", json=request_data)
        assert response.status_code == 400
        assert "Unknown model type" in response.json()["detail"]


class TestModelsInfoEndpoint:
    """Test models info endpoint"""

    def test_get_models_info_success(self):
        """Test successful models info retrieval"""
        response = client.get("/models/info")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert "total_models" in data
        assert "system_info" in data
        assert data["total_models"] == 7
        assert len(data["models"]) == 7

        # Check first model structure
        first_model = data["models"][0]
        required_fields = ["name", "version", "type", "accuracy", "algorithms",
                          "last_trained", "training_samples", "feature_count"]
        for field in required_fields:
            assert field in first_model


class TestSystemMetricsEndpoint:
    """Test system metrics endpoint"""

    def test_get_system_metrics_success(self):
        """Test successful system metrics retrieval"""
        response = client.get("/metrics/system")
        assert response.status_code == 200

        data = response.json()
        required_metrics = [
            "cpu_utilization", "memory_utilization", "disk_utilization",
            "network_utilization", "active_predictions", "completed_optimizations",
            "anomalies_detected", "model_accuracy_avg", "prediction_latency_ms",
            "system_load", "timestamp"
        ]

        for metric in required_metrics:
            assert metric in data

        # Validate ranges for utilization metrics
        assert 0 <= data["cpu_utilization"] <= 1
        assert 0 <= data["memory_utilization"] <= 1
        assert 0 <= data["disk_utilization"] <= 1
        assert 0 <= data["network_utilization"] <= 1


class TestPredictionStatisticsEndpoint:
    """Test prediction statistics endpoint"""

    def test_get_prediction_statistics_success(self):
        """Test successful prediction statistics retrieval"""
        response = client.get("/statistics/predictions")
        assert response.status_code == 200

        data = response.json()
        required_sections = [
            "total_predictions", "accuracy_by_model", "prediction_latency",
            "model_utilization", "error_rates", "timestamp"
        ]

        for section in required_sections:
            assert section in data

        # Validate accuracy_by_model structure
        accuracy_models = data["accuracy_by_model"]
        expected_models = [
            "resource_predictor", "migration_predictor", "anomaly_detector",
            "workload_predictor", "performance_optimizer"
        ]

        for model in expected_models:
            assert model in accuracy_models
            assert 0 <= accuracy_models[model] <= 1


class TestProcessRequestEndpoint:
    """Test process request dispatcher endpoint"""

    @patch('ai_engine.app.resource_predictor')
    def test_process_resource_prediction(self, mock_predictor):
        """Test process request for resource prediction"""
        mock_predictor.predict_sequence.return_value = {
            'cpu_usage': [55.0, 60.0, 65.0],
            'memory_usage': [45.0, 50.0, 55.0],
            'confidence': 0.87,
            'model_used': 'ensemble'
        }
        mock_predictor.feature_importance = {}

        request_data = {
            "service": "resource",
            "method": "predict",
            "data": {
                "resource_id": "vm-process-test",
                "metrics": {"cpu_usage": 55.0, "memory_usage": 45.0},
                "prediction_horizon": 3
            }
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["confidence"] == 0.87

    @patch('ai_engine.app.anomaly_detector')
    def test_process_anomaly_detection(self, mock_detector):
        """Test process request for anomaly detection"""
        mock_detector.detect.return_value = {
            'is_anomaly': True,
            'anomaly_score': 0.75,
            'anomaly_type': 'memory_spike',
            'contributing_features': ['memory_usage']
        }

        request_data = {
            "service": "anomaly",
            "method": "detect",
            "data": {
                "resource_id": "vm-anomaly-process",
                "metrics": {"cpu_usage": 50.0, "memory_usage": 90.0},
                "sensitivity": 0.1
            }
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["confidence"] == 0.75

    def test_process_unknown_service(self):
        """Test process request with unknown service"""
        request_data = {
            "service": "unknown_service",
            "method": "unknown_method",
            "data": {}
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False
        assert "Unknown service" in data["error"]
        assert "supported_services" in data


class TestErrorHandling:
    """Test error handling scenarios"""

    @patch('ai_engine.app.resource_predictor')
    def test_resource_prediction_error(self, mock_predictor):
        """Test resource prediction with exception"""
        mock_predictor.predict_sequence.side_effect = Exception("Model failure")

        request_data = {
            "resource_id": "vm-error-test",
            "metrics": {"cpu_usage": 50.0, "memory_usage": 40.0}
        }

        response = client.post("/predict/resources", json=request_data)
        assert response.status_code == 500
        assert "Model failure" in response.json()["detail"]

    def test_invalid_request_data(self):
        """Test endpoints with invalid request data"""
        # Missing required fields
        invalid_request = {"invalid_field": "value"}

        response = client.post("/predict/resources", json=invalid_request)
        assert response.status_code == 422  # Pydantic validation error

    @patch('ai_engine.app.migration_predictor')
    def test_migration_prediction_error(self, mock_predictor):
        """Test migration prediction with exception"""
        mock_predictor.predict_optimal_host.side_effect = Exception("Migration analysis failed")

        request_data = {
            "vm_id": "vm-migration-error",
            "source_host": "host-1",
            "target_hosts": ["host-2"],
            "vm_metrics": {"cpu_usage": 70.0}
        }

        response = client.post("/predict/migration", json=request_data)
        assert response.status_code == 500
        assert "Migration analysis failed" in response.json()["detail"]


# Integration test configuration
class TestGoClientCompatibility:
    """Test Go client service/method name compatibility with dispatcher"""

    @patch('ai_engine.app.resource_predictor')
    def test_go_resource_prediction_predict_demand(self, mock_predictor):
        """Test Go client resource_prediction/predict_demand mapping"""
        mock_predictor.predict_sequence.return_value = {
            'cpu_usage': [60.0, 65.0, 70.0],
            'memory_usage': [45.0, 50.0, 55.0],
            'confidence': 0.88,
            'model_used': 'lstm'
        }
        mock_predictor.feature_importance = {}

        request_data = {
            "service": "resource_prediction",
            "method": "predict_demand",
            "data": {
                "resource_id": "node-789",
                "metrics": {"cpu_usage": 55.0, "memory_usage": 42.0},
                "prediction_horizon": 5
            }
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["confidence"] == 0.88
        assert data["data"]["resource_id"] == "node-789"

    @patch('ai_engine.app.performance_predictor')
    def test_go_performance_optimization_optimize_cluster(self, mock_predictor):
        """Test Go client performance_optimization/optimize_cluster mapping"""
        mock_predictor.optimize_performance.return_value = {
            'optimizations': [{'type': 'cpu', 'action': 'scale-up'}],
            'improvements': {'cpu': 0.25, 'memory': 0.15},
            'priority': ['cpu', 'memory'],
            'confidence': 0.82
        }

        request_data = {
            "service": "performance_optimization",
            "method": "optimize_cluster",
            "data": {
                "cluster_id": "cluster-456",
                "performance_data": [{"timestamp": "2023-01-01", "cpu": 80.0}],
                "optimization_goals": ["improve_performance"]
            }
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["confidence"] == 0.82

    @patch('ai_engine.app.anomaly_detector')
    def test_go_anomaly_detection_detect(self, mock_detector):
        """Test Go client anomaly_detection/detect mapping"""
        mock_detector.detect.return_value = {
            'is_anomaly': True,
            'anomaly_score': 0.75,
            'anomaly_type': 'cpu_spike',
            'contributing_features': ['cpu_usage']
        }

        request_data = {
            "service": "anomaly_detection",
            "method": "detect",
            "data": {
                "resource_id": "vm-abc",
                "metrics": {"cpu_usage": 95.0, "memory_usage": 60.0},
                "sensitivity": 0.2
            }
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["confidence"] == 0.75
        assert data["data"]["is_anomaly"] is True

    @patch('ai_engine.app.workload_recognizer')
    def test_go_workload_pattern_recognition_analyze_patterns(self, mock_recognizer):
        """Test Go client workload_pattern_recognition/analyze_patterns mapping"""
        # Create a mock pattern object with all required attributes
        mock_pattern = Mock()
        mock_pattern.workload_type = WorkloadType.CPU_INTENSIVE
        mock_pattern.pattern_type = PatternType.STEADY_STATE
        mock_pattern.confidence = 0.91
        mock_pattern.characteristics = {"avg_cpu": 75.0}
        mock_recognizer.analyze_workload.return_value = mock_pattern

        request_data = {
            "service": "workload_pattern_recognition",
            "method": "analyze_patterns",
            "data": {
                "vm_id": "vm-xyz",
                "workload_data": [{"timestamp": "2023-01-01", "cpu": 75.0}],
                "analysis_window": 3600
            }
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["confidence"] == 0.91
        assert data["data"]["workload_type"] == "cpu_intensive"

    @patch('ai_engine.app.scaling_engine')
    def test_go_predictive_scaling_predict_scaling(self, mock_engine):
        """Test Go client predictive_scaling/predict_scaling mapping"""
        # Create a mock decision object with all required attributes
        mock_decision = Mock()
        mock_decision.resource_type = ResourceType.CPU
        mock_decision.scaling_action = ScalingAction.SCALE_UP
        mock_decision.current_value = 2
        mock_decision.target_value = 4
        mock_decision.confidence = 0.85
        mock_decision.reasoning = "High demand predicted"
        mock_decision.execution_time = datetime.now()
        mock_decision.urgency_score = 0.8
        mock_decision.cost_impact = 25.0
        mock_decision.performance_impact = 0.9

        # Create a mock forecast object
        mock_forecast = Mock()
        mock_forecast.resource_type = ResourceType.CPU
        mock_forecast.predictions = [80.0, 85.0, 90.0]
        mock_forecast.confidence_intervals = [(75, 85), (80, 90), (85, 95)]
        mock_forecast.confidence = 0.85
        mock_forecast.trend = "increasing"
        mock_forecast.seasonality_detected = False
        mock_forecast.anomaly_risk = 0.1

        # Mock the predict_resource_demand method
        mock_engine.predict_resource_demand.return_value = mock_forecast

        # Mock the make_scaling_decision method
        mock_engine.make_scaling_decision.return_value = [mock_decision]

        request_data = {
            "service": "predictive_scaling",
            "method": "predict_scaling",
            "data": {
                "vm_id": "vm-scaling",
                "current_resources": {"cpu_usage": 75.0},
                "historical_data": [{"timestamp": "2023-01-01", "cpu": 70.0}]
            }
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert "data" in data
        assert data["confidence"] > 0

    @patch('ai_engine.app.resource_predictor')
    def test_go_model_training_train(self, mock_predictor):
        """Test Go client model_training/train mapping"""
        mock_predictor.train.return_value = {"accuracy": 0.92, "loss": 0.08}

        request_data = {
            "service": "model_training",
            "method": "train",
            "data": {
                "model_type": "resource_prediction",
                "training_data": [{"cpu": 50, "memory": 40}],
                "validation_split": 0.2
            }
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["confidence"] == 0.9
        assert data["data"]["training_status"] == "completed"

    def test_go_model_management_get_info(self):
        """Test Go client model_management/get_info mapping"""
        request_data = {
            "service": "model_management",
            "method": "get_info",
            "data": {}
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["confidence"] == 1.0
        assert "data" in data
        assert "models" in data["data"]
        assert len(data["data"]["models"]) > 0

    def test_go_health_check(self):
        """Test Go client health/check mapping"""
        request_data = {
            "service": "health",
            "method": "check",
            "data": {}
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["confidence"] == 1.0
        assert "data" in data
        assert data["data"]["status"] == "healthy"

    def test_go_unknown_service_method(self):
        """Test unknown Go service/method returns appropriate error"""
        request_data = {
            "service": "unknown_service",
            "method": "unknown_method",
            "data": {}
        }

        response = client.post("/api/v1/process", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is False
        assert "error" in data
        assert "supported_services" in data
        # Verify that Go service names are in supported services
        assert "resource_prediction" in data["supported_services"]
        assert "performance_optimization" in data["supported_services"]
        assert "anomaly_detection" in data["supported_services"]

    def test_case_insensitive_service_method(self):
        """Test that service/method names are case-insensitive"""
        request_data = {
            "service": "RESOURCE_PREDICTION",
            "method": "PREDICT_DEMAND",
            "data": {
                "resource_id": "test-node",
                "metrics": {"cpu_usage": 50.0}
            }
        }

        with patch('ai_engine.app.resource_predictor') as mock_predictor:
            mock_predictor.predict_sequence.return_value = {
                'cpu_usage': [55.0],
                'memory_usage': [45.0],
                'confidence': 0.8,
                'model_used': 'test'
            }
            mock_predictor.feature_importance = {}

            response = client.post("/api/v1/process", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True

    def test_service_method_with_whitespace(self):
        """Test that service/method names handle whitespace correctly"""
        request_data = {
            "service": "  resource_prediction  ",
            "method": "  predict_demand  ",
            "data": {
                "resource_id": "test-node",
                "metrics": {"cpu_usage": 50.0}
            }
        }

        with patch('ai_engine.app.resource_predictor') as mock_predictor:
            mock_predictor.predict_sequence.return_value = {
                'cpu_usage': [55.0],
                'memory_usage': [45.0],
                'confidence': 0.8,
                'model_used': 'test'
            }
            mock_predictor.feature_importance = {}

            response = client.post("/api/v1/process", json=request_data)
            assert response.status_code == 200

            data = response.json()
            assert data["success"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])