"""
Multi-framework model serving infrastructure with auto-scaling and A/B testing.
Supports TensorFlow, PyTorch, ONNX, and scikit-learn models.
"""

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    import onnxruntime as ort
    import tensorflow as tf
    ADVANCED_FRAMEWORKS = True
except ImportError:
    ADVANCED_FRAMEWORKS = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFramework(Enum):
    """Supported ML frameworks"""
    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    XGBOOST = "xgboost"


class DeploymentStrategy(Enum):
    """Model deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    AB_TEST = "ab_test"
    SHADOW = "shadow"
    ROLLING = "rolling"


@dataclass
class ModelEndpoint:
    """Model serving endpoint configuration"""
    endpoint_id: str
    model_id: str
    model_version: str
    framework: ModelFramework
    model_path: str

    # Serving configuration
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu: float = 0.7
    target_latency_ms: float = 100.0

    # Deployment
    strategy: DeploymentStrategy = DeploymentStrategy.ROLLING
    traffic_split: Dict[str, float] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: str = "active"

    # Metrics
    request_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0


@dataclass
class PredictionRequest:
    """Prediction request payload"""
    request_id: str
    endpoint_id: str
    features: Union[Dict[str, Any], List[Dict[str, Any]]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PredictionResponse:
    """Prediction response payload"""
    request_id: str
    predictions: Union[List[Any], Any]
    probabilities: Optional[List[List[float]]] = None
    latency_ms: float = 0.0
    model_version: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ModelLoader:
    """Universal model loader for multiple frameworks"""

    @staticmethod
    def load_model(model_path: str, framework: ModelFramework) -> Any:
        """Load model from disk based on framework"""
        logger.info(f"Loading {framework.value} model from {model_path}")

        if framework == ModelFramework.SKLEARN or framework == ModelFramework.XGBOOST:
            with open(model_path, 'rb') as f:
                return pickle.load(f)

        elif framework == ModelFramework.PYTORCH:
            if not ADVANCED_FRAMEWORKS:
                raise RuntimeError("PyTorch not available")
            model = torch.load(model_path)
            model.eval()
            return model

        elif framework == ModelFramework.TENSORFLOW:
            if not ADVANCED_FRAMEWORKS:
                raise RuntimeError("TensorFlow not available")
            return tf.keras.models.load_model(model_path)

        elif framework == ModelFramework.ONNX:
            if not ADVANCED_FRAMEWORKS:
                raise RuntimeError("ONNX Runtime not available")
            return ort.InferenceSession(model_path)

        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def predict(model: Any, features: Union[pd.DataFrame, np.ndarray], framework: ModelFramework) -> np.ndarray:
        """Universal prediction interface"""

        if framework == ModelFramework.SKLEARN or framework == ModelFramework.XGBOOST:
            return model.predict(features)

        elif framework == ModelFramework.PYTORCH:
            if isinstance(features, pd.DataFrame):
                features = features.values
            features_tensor = torch.FloatTensor(features)
            with torch.no_grad():
                output = model(features_tensor)
                return output.numpy()

        elif framework == ModelFramework.TENSORFLOW:
            if isinstance(features, pd.DataFrame):
                features = features.values
            return model.predict(features)

        elif framework == ModelFramework.ONNX:
            if isinstance(features, pd.DataFrame):
                features = features.values
            input_name = model.get_inputs()[0].name
            return model.run(None, {input_name: features.astype(np.float32)})[0]

        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def predict_proba(model: Any, features: Union[pd.DataFrame, np.ndarray], framework: ModelFramework) -> Optional[np.ndarray]:
        """Predict probabilities if available"""

        try:
            if framework == ModelFramework.SKLEARN or framework == ModelFramework.XGBOOST:
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(features)

            elif framework == ModelFramework.PYTORCH:
                if isinstance(features, pd.DataFrame):
                    features = features.values
                features_tensor = torch.FloatTensor(features)
                with torch.no_grad():
                    output = model(features_tensor)
                    proba = torch.softmax(output, dim=1)
                    return proba.numpy()

            elif framework == ModelFramework.TENSORFLOW:
                if isinstance(features, pd.DataFrame):
                    features = features.values
                output = model.predict(features)
                if len(output.shape) > 1 and output.shape[1] > 1:
                    return output

        except Exception as e:
            logger.warning(f"Failed to get probabilities: {e}")

        return None


class AutoScaler:
    """Automatic endpoint scaling based on metrics"""

    def __init__(self, endpoint: ModelEndpoint):
        self.endpoint = endpoint
        self.current_replicas = endpoint.min_replicas
        self.last_scale_time = time.time()
        self.scale_cooldown = 60  # seconds

    def should_scale_up(self, current_cpu: float, current_latency: float) -> bool:
        """Check if scaling up is needed"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False

        if self.current_replicas >= self.endpoint.max_replicas:
            return False

        if current_cpu > self.endpoint.target_cpu:
            return True

        if current_latency > self.endpoint.target_latency_ms:
            return True

        return False

    def should_scale_down(self, current_cpu: float, current_latency: float) -> bool:
        """Check if scaling down is needed"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False

        if self.current_replicas <= self.endpoint.min_replicas:
            return False

        if current_cpu < self.endpoint.target_cpu * 0.5 and current_latency < self.endpoint.target_latency_ms * 0.5:
            return True

        return False

    def scale_up(self) -> int:
        """Scale up replicas"""
        new_replicas = min(self.current_replicas + 1, self.endpoint.max_replicas)
        logger.info(f"Scaling up from {self.current_replicas} to {new_replicas} replicas")
        self.current_replicas = new_replicas
        self.last_scale_time = time.time()
        return new_replicas

    def scale_down(self) -> int:
        """Scale down replicas"""
        new_replicas = max(self.current_replicas - 1, self.endpoint.min_replicas)
        logger.info(f"Scaling down from {self.current_replicas} to {new_replicas} replicas")
        self.current_replicas = new_replicas
        self.last_scale_time = time.time()
        return new_replicas


class ABTestManager:
    """A/B testing and canary deployment manager"""

    def __init__(self):
        self.experiments = {}
        self.traffic_router = {}

    def create_experiment(
        self,
        experiment_id: str,
        control_endpoint: str,
        variant_endpoints: List[str],
        traffic_split: Dict[str, float]
    ):
        """Create A/B test experiment"""
        total_traffic = sum(traffic_split.values())
        if abs(total_traffic - 1.0) > 0.001:
            raise ValueError(f"Traffic split must sum to 1.0, got {total_traffic}")

        self.experiments[experiment_id] = {
            "control": control_endpoint,
            "variants": variant_endpoints,
            "traffic_split": traffic_split,
            "metrics": {endpoint: {"requests": 0, "errors": 0, "latency_sum": 0.0}
                       for endpoint in [control_endpoint] + variant_endpoints},
            "start_time": datetime.now(),
        }

        logger.info(f"Created experiment {experiment_id} with traffic split: {traffic_split}")

    def route_request(self, experiment_id: str) -> str:
        """Route request to appropriate endpoint based on traffic split"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")

        experiment = self.experiments[experiment_id]
        traffic_split = experiment["traffic_split"]

        # Weighted random selection
        endpoints = list(traffic_split.keys())
        weights = list(traffic_split.values())

        rand = np.random.random()
        cumsum = 0.0
        for endpoint, weight in zip(endpoints, weights):
            cumsum += weight
            if rand <= cumsum:
                return endpoint

        return endpoints[0]

    def record_metrics(self, experiment_id: str, endpoint: str, latency_ms: float, error: bool = False):
        """Record metrics for experiment"""
        if experiment_id not in self.experiments:
            return

        metrics = self.experiments[experiment_id]["metrics"][endpoint]
        metrics["requests"] += 1
        metrics["latency_sum"] += latency_ms
        if error:
            metrics["errors"] += 1

    def get_experiment_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment results and statistics"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment not found: {experiment_id}")

        experiment = self.experiments[experiment_id]
        results = {}

        for endpoint, metrics in experiment["metrics"].items():
            requests = metrics["requests"]
            if requests > 0:
                avg_latency = metrics["latency_sum"] / requests
                error_rate = metrics["errors"] / requests
            else:
                avg_latency = 0.0
                error_rate = 0.0

            results[endpoint] = {
                "requests": requests,
                "avg_latency_ms": avg_latency,
                "error_rate": error_rate,
                "total_errors": metrics["errors"],
            }

        return {
            "experiment_id": experiment_id,
            "control": experiment["control"],
            "variants": experiment["variants"],
            "traffic_split": experiment["traffic_split"],
            "results": results,
            "duration_hours": (datetime.now() - experiment["start_time"]).total_seconds() / 3600,
        }

    def determine_winner(self, experiment_id: str, metric: str = "avg_latency_ms") -> str:
        """Determine winning variant based on metric"""
        results = self.get_experiment_results(experiment_id)

        best_endpoint = None
        best_value = float('inf') if metric.endswith('latency') or metric.endswith('error_rate') else float('-inf')

        for endpoint, metrics in results["results"].items():
            value = metrics.get(metric, 0)

            if metric.endswith('latency') or metric.endswith('error_rate'):
                # Lower is better
                if value < best_value:
                    best_value = value
                    best_endpoint = endpoint
            else:
                # Higher is better
                if value > best_value:
                    best_value = value
                    best_endpoint = endpoint

        return best_endpoint


class ModelServer:
    """Multi-framework model serving server"""

    def __init__(self, storage_path: str = "./models"):
        self.storage_path = Path(storage_path)
        self.endpoints = {}
        self.models = {}
        self.scalers = {}
        self.ab_manager = ABTestManager()
        self.executor = ThreadPoolExecutor(max_workers=10)

        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def deploy_model(
        self,
        model_id: str,
        model_version: str,
        model_path: str,
        framework: ModelFramework,
        endpoint_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Deploy a model to serving endpoint"""

        endpoint_id = f"{model_id}_{model_version}_{int(time.time())}"

        config = endpoint_config or {}
        endpoint = ModelEndpoint(
            endpoint_id=endpoint_id,
            model_id=model_id,
            model_version=model_version,
            framework=framework,
            model_path=model_path,
            min_replicas=config.get("min_replicas", 1),
            max_replicas=config.get("max_replicas", 10),
            target_cpu=config.get("target_cpu", 0.7),
            target_latency_ms=config.get("target_latency_ms", 100.0),
        )

        # Load model
        try:
            model = await asyncio.to_thread(ModelLoader.load_model, model_path, framework)
            self.models[endpoint_id] = model
            self.endpoints[endpoint_id] = endpoint
            self.scalers[endpoint_id] = AutoScaler(endpoint)

            logger.info(f"Deployed model {model_id} v{model_version} to endpoint {endpoint_id}")
            return endpoint_id

        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise

    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        """Serve prediction request"""

        start_time = time.time()

        try:
            endpoint = self.endpoints.get(request.endpoint_id)
            if not endpoint:
                raise ValueError(f"Endpoint not found: {request.endpoint_id}")

            model = self.models.get(request.endpoint_id)
            if not model:
                raise ValueError(f"Model not loaded for endpoint: {request.endpoint_id}")

            # Convert features to DataFrame
            if isinstance(request.features, dict):
                features = pd.DataFrame([request.features])
            else:
                features = pd.DataFrame(request.features)

            # Get predictions
            predictions = await asyncio.to_thread(
                ModelLoader.predict,
                model,
                features,
                endpoint.framework
            )

            # Get probabilities if available
            probabilities = await asyncio.to_thread(
                ModelLoader.predict_proba,
                model,
                features,
                endpoint.framework
            )

            latency_ms = (time.time() - start_time) * 1000

            # Update metrics
            endpoint.request_count += 1
            endpoint.total_latency_ms += latency_ms

            # Check auto-scaling
            avg_latency = endpoint.total_latency_ms / endpoint.request_count
            scaler = self.scalers[request.endpoint_id]

            if scaler.should_scale_up(0.8, avg_latency):
                scaler.scale_up()
            elif scaler.should_scale_down(0.3, avg_latency):
                scaler.scale_down()

            return PredictionResponse(
                request_id=request.request_id,
                predictions=predictions.tolist(),
                probabilities=probabilities.tolist() if probabilities is not None else None,
                latency_ms=latency_ms,
                model_version=endpoint.model_version,
                metadata={"endpoint_id": endpoint.endpoint_id}
            )

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            endpoint.error_count += 1
            raise

    async def batch_predict(
        self,
        endpoint_id: str,
        features_batch: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> List[PredictionResponse]:
        """Batch prediction processing"""

        responses = []

        for i in range(0, len(features_batch), batch_size):
            batch = features_batch[i:i+batch_size]

            request = PredictionRequest(
                request_id=f"batch_{i}_{int(time.time())}",
                endpoint_id=endpoint_id,
                features=batch
            )

            response = await self.predict(request)
            responses.append(response)

        return responses

    def create_ab_test(
        self,
        experiment_id: str,
        control_endpoint: str,
        variant_endpoints: List[str],
        traffic_split: Dict[str, float]
    ):
        """Create A/B test between model versions"""

        # Validate endpoints exist
        all_endpoints = [control_endpoint] + variant_endpoints
        for endpoint in all_endpoints:
            if endpoint not in self.endpoints:
                raise ValueError(f"Endpoint not found: {endpoint}")

        self.ab_manager.create_experiment(
            experiment_id,
            control_endpoint,
            variant_endpoints,
            traffic_split
        )

    async def predict_with_ab_test(
        self,
        experiment_id: str,
        features: Union[Dict[str, Any], List[Dict[str, Any]]]
    ) -> PredictionResponse:
        """Predict with A/B test routing"""

        # Route to endpoint based on traffic split
        endpoint_id = self.ab_manager.route_request(experiment_id)

        request = PredictionRequest(
            request_id=f"ab_{experiment_id}_{int(time.time())}",
            endpoint_id=endpoint_id,
            features=features
        )

        start_time = time.time()

        try:
            response = await self.predict(request)
            latency_ms = (time.time() - start_time) * 1000

            # Record metrics for experiment
            self.ab_manager.record_metrics(experiment_id, endpoint_id, latency_ms, error=False)

            return response

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self.ab_manager.record_metrics(experiment_id, endpoint_id, latency_ms, error=True)
            raise

    def get_ab_test_results(self, experiment_id: str) -> Dict[str, Any]:
        """Get A/B test results"""
        return self.ab_manager.get_experiment_results(experiment_id)

    def promote_ab_winner(self, experiment_id: str) -> str:
        """Promote winning variant to 100% traffic"""
        winner = self.ab_manager.determine_winner(experiment_id)

        # Update traffic split to 100% for winner
        experiment = self.ab_manager.experiments[experiment_id]
        experiment["traffic_split"] = {winner: 1.0}

        logger.info(f"Promoted {winner} to 100% traffic for experiment {experiment_id}")
        return winner

    async def undeploy_model(self, endpoint_id: str):
        """Remove model from serving"""

        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint not found: {endpoint_id}")

        endpoint = self.endpoints[endpoint_id]
        endpoint.status = "inactive"

        # Cleanup
        if endpoint_id in self.models:
            del self.models[endpoint_id]

        if endpoint_id in self.scalers:
            del self.scalers[endpoint_id]

        logger.info(f"Undeployed endpoint: {endpoint_id}")

    def get_endpoint_metrics(self, endpoint_id: str) -> Dict[str, Any]:
        """Get endpoint metrics"""

        if endpoint_id not in self.endpoints:
            raise ValueError(f"Endpoint not found: {endpoint_id}")

        endpoint = self.endpoints[endpoint_id]
        scaler = self.scalers.get(endpoint_id)

        avg_latency = (endpoint.total_latency_ms / endpoint.request_count
                      if endpoint.request_count > 0 else 0.0)

        error_rate = (endpoint.error_count / endpoint.request_count
                     if endpoint.request_count > 0 else 0.0)

        return {
            "endpoint_id": endpoint_id,
            "model_id": endpoint.model_id,
            "model_version": endpoint.model_version,
            "status": endpoint.status,
            "requests": endpoint.request_count,
            "avg_latency_ms": avg_latency,
            "error_rate": error_rate,
            "current_replicas": scaler.current_replicas if scaler else 0,
            "min_replicas": endpoint.min_replicas,
            "max_replicas": endpoint.max_replicas,
        }

    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all serving endpoints"""
        return [
            {
                "endpoint_id": endpoint_id,
                "model_id": endpoint.model_id,
                "model_version": endpoint.model_version,
                "framework": endpoint.framework.value,
                "status": endpoint.status,
                "requests": endpoint.request_count,
            }
            for endpoint_id, endpoint in self.endpoints.items()
        ]


# Example usage
async def example_serving():
    """Example model serving workflow"""

    server = ModelServer(storage_path="./model_storage")

    # Deploy model
    endpoint_id = await server.deploy_model(
        model_id="fraud_detector",
        model_version="v1.0.0",
        model_path="./models/fraud_model.pkl",
        framework=ModelFramework.SKLEARN,
        endpoint_config={
            "min_replicas": 2,
            "max_replicas": 10,
            "target_latency_ms": 50.0,
        }
    )

    # Single prediction
    request = PredictionRequest(
        request_id="req_001",
        endpoint_id=endpoint_id,
        features={
            "transaction_amount": 150.0,
            "merchant_risk_score": 0.3,
            "user_age": 35,
        }
    )

    response = await server.predict(request)
    print(f"Prediction: {response.predictions}")
    print(f"Latency: {response.latency_ms:.2f}ms")

    # A/B test
    server.create_ab_test(
        experiment_id="fraud_model_test",
        control_endpoint=endpoint_id,
        variant_endpoints=["fraud_detector_v2_123"],
        traffic_split={endpoint_id: 0.9, "fraud_detector_v2_123": 0.1}
    )


if __name__ == "__main__":
    asyncio.run(example_serving())
