"""
ML model monitoring and observability with drift detection and explainability.
Tracks model performance, data drift, and provides prediction explanations.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

try:
    import shap
    from lime import lime_tabular
    EXPLAINABILITY_AVAILABLE = True
except ImportError:
    EXPLAINABILITY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detection"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    LABEL_DRIFT = "label_drift"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    model_id: str
    version: str
    timestamp: datetime

    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None

    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None

    # Operational metrics
    latency_p50: Optional[float] = None
    latency_p95: Optional[float] = None
    latency_p99: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None

    # Business metrics
    prediction_count: int = 0
    positive_predictions: int = 0
    negative_predictions: int = 0


@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    model_id: str
    drift_type: DriftType
    severity: AlertSeverity
    feature_name: Optional[str] = None
    drift_score: float = 0.0
    threshold: float = 0.0
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureStats:
    """Statistical metrics for a feature"""
    feature_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    missing_count: int
    unique_count: int
    timestamp: datetime = field(default_factory=datetime.now)


class DataDriftDetector:
    """Detect data drift using statistical tests"""

    def __init__(self, reference_data: pd.DataFrame, sensitivity: float = 0.05):
        self.reference_data = reference_data
        self.sensitivity = sensitivity
        self.reference_stats = self._calculate_stats(reference_data)

    def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, FeatureStats]:
        """Calculate feature statistics"""
        stats_dict = {}

        for col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                stats_dict[col] = FeatureStats(
                    feature_name=col,
                    mean=float(data[col].mean()),
                    std=float(data[col].std()),
                    min=float(data[col].min()),
                    max=float(data[col].max()),
                    median=float(data[col].median()),
                    q25=float(data[col].quantile(0.25)),
                    q75=float(data[col].quantile(0.75)),
                    missing_count=int(data[col].isna().sum()),
                    unique_count=int(data[col].nunique()),
                )

        return stats_dict

    def detect_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """Detect drift between reference and current data"""
        alerts = []

        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue

            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                # Kolmogorov-Smirnov test for continuous features
                statistic, p_value = stats.ks_2samp(
                    self.reference_data[col].dropna(),
                    current_data[col].dropna()
                )

                if p_value < self.sensitivity:
                    alerts.append(DriftAlert(
                        alert_id=f"drift_{col}_{int(time.time())}",
                        model_id="",  # Set by caller
                        drift_type=DriftType.DATA_DRIFT,
                        severity=AlertSeverity.WARNING if p_value < 0.01 else AlertSeverity.INFO,
                        feature_name=col,
                        drift_score=float(statistic),
                        threshold=self.sensitivity,
                        description=f"Data drift detected in feature '{col}' (KS statistic: {statistic:.4f}, p-value: {p_value:.4f})",
                        metadata={"test": "ks_test", "p_value": p_value}
                    ))

            else:
                # Chi-square test for categorical features
                ref_counts = self.reference_data[col].value_counts()
                curr_counts = current_data[col].value_counts()

                # Align categories
                all_categories = set(ref_counts.index) | set(curr_counts.index)
                ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]

                if sum(ref_freq) > 0 and sum(curr_freq) > 0:
                    statistic, p_value = stats.chisquare(curr_freq, ref_freq)

                    if p_value < self.sensitivity:
                        alerts.append(DriftAlert(
                            alert_id=f"drift_{col}_{int(time.time())}",
                            model_id="",
                            drift_type=DriftType.DATA_DRIFT,
                            severity=AlertSeverity.WARNING if p_value < 0.01 else AlertSeverity.INFO,
                            feature_name=col,
                            drift_score=float(statistic),
                            threshold=self.sensitivity,
                            description=f"Data drift detected in categorical feature '{col}' (Chi-square: {statistic:.4f}, p-value: {p_value:.4f})",
                            metadata={"test": "chi_square", "p_value": p_value}
                        ))

        return alerts

    def get_feature_drift_score(self, feature_name: str, current_data: pd.DataFrame) -> float:
        """Get drift score for a specific feature"""
        if feature_name not in self.reference_data.columns or feature_name not in current_data.columns:
            return 0.0

        if pd.api.types.is_numeric_dtype(self.reference_data[feature_name]):
            statistic, _ = stats.ks_2samp(
                self.reference_data[feature_name].dropna(),
                current_data[feature_name].dropna()
            )
            return float(statistic)

        return 0.0


class ConceptDriftDetector:
    """Detect concept drift using prediction performance"""

    def __init__(self, window_size: int = 1000, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None

    def update(self, accuracy: float):
        """Update with new performance metric"""
        self.performance_history.append(accuracy)

        if self.baseline_performance is None and len(self.performance_history) >= self.window_size:
            self.baseline_performance = np.mean(self.performance_history)

    def detect_drift(self) -> Optional[DriftAlert]:
        """Detect concept drift"""
        if self.baseline_performance is None or len(self.performance_history) < self.window_size:
            return None

        current_performance = np.mean(list(self.performance_history)[-100:])  # Last 100 predictions
        performance_drop = self.baseline_performance - current_performance

        if performance_drop > self.threshold:
            return DriftAlert(
                alert_id=f"concept_drift_{int(time.time())}",
                model_id="",
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=AlertSeverity.CRITICAL if performance_drop > 0.1 else AlertSeverity.WARNING,
                drift_score=performance_drop,
                threshold=self.threshold,
                description=f"Concept drift detected: performance dropped by {performance_drop:.2%}",
                metadata={"baseline": self.baseline_performance, "current": current_performance}
            )

        return None


class ModelExplainer:
    """Model prediction explainability"""

    def __init__(self, model: Any, feature_names: List[str], model_type: str = "tree"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.explainer = None

        if EXPLAINABILITY_AVAILABLE:
            if model_type == "tree":
                self.explainer = shap.TreeExplainer(model)
            else:
                # Use LIME for other models
                pass

    def explain_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Explain a single prediction"""
        if not EXPLAINABILITY_AVAILABLE:
            return {"error": "Explainability libraries not available"}

        if self.explainer is None:
            return {"error": "Explainer not initialized"}

        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(features)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification

            # Calculate feature importance
            if len(features.shape) == 1:
                features = features.reshape(1, -1)

            feature_importance = {
                name: float(value)
                for name, value in zip(self.feature_names, shap_values[0])
            }

            # Sort by absolute importance
            sorted_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ))

            return {
                "feature_importance": sorted_importance,
                "top_features": list(sorted_importance.keys())[:5],
                "method": "shap",
            }

        except Exception as e:
            logger.error(f"Explainability failed: {e}")
            return {"error": str(e)}

    def get_global_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get global feature importance across dataset"""
        if not EXPLAINABILITY_AVAILABLE or self.explainer is None:
            return {}

        try:
            shap_values = self.explainer.shap_values(X)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)

            return {
                name: float(value)
                for name, value in zip(self.feature_names, mean_shap)
            }

        except Exception as e:
            logger.error(f"Global importance calculation failed: {e}")
            return {}


class MLMonitor:
    """Comprehensive ML model monitoring system"""

    def __init__(self, model_id: str, model_version: str):
        self.model_id = model_id
        self.model_version = model_version

        # Metrics storage
        self.metrics_history = deque(maxlen=10000)
        self.latency_buffer = deque(maxlen=1000)
        self.prediction_buffer = deque(maxlen=1000)

        # Drift detection
        self.data_drift_detector = None
        self.concept_drift_detector = ConceptDriftDetector()

        # Alerts
        self.active_alerts = []
        self.alert_history = deque(maxlen=1000)

        # Explainability
        self.explainer = None

        # Feature tracking
        self.feature_importance_cache = {}
        self.feature_stats_history = defaultdict(list)

    def set_reference_data(self, reference_data: pd.DataFrame, sensitivity: float = 0.05):
        """Set reference data for drift detection"""
        self.data_drift_detector = DataDriftDetector(reference_data, sensitivity)
        logger.info(f"Reference data set with {len(reference_data)} samples")

    def set_explainer(self, model: Any, feature_names: List[str], model_type: str = "tree"):
        """Configure model explainer"""
        self.explainer = ModelExplainer(model, feature_names, model_type)
        logger.info("Model explainer configured")

    async def log_prediction(
        self,
        features: Dict[str, Any],
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency_ms: float = 0.0,
        metadata: Dict[str, Any] = None
    ):
        """Log a prediction for monitoring"""

        # Log latency
        self.latency_buffer.append(latency_ms)

        # Log prediction
        self.prediction_buffer.append({
            "features": features,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "timestamp": datetime.now(),
            "metadata": metadata or {},
        })

        # Update concept drift detector
        if ground_truth is not None:
            is_correct = (prediction == ground_truth)
            self.concept_drift_detector.update(float(is_correct))

    async def check_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """Check for data drift"""
        if self.data_drift_detector is None:
            return []

        alerts = self.data_drift_detector.detect_drift(current_data)

        # Set model_id for alerts
        for alert in alerts:
            alert.model_id = self.model_id

        # Check concept drift
        concept_alert = self.concept_drift_detector.detect_drift()
        if concept_alert:
            concept_alert.model_id = self.model_id
            alerts.append(concept_alert)

        # Store alerts
        self.active_alerts.extend(alerts)
        self.alert_history.extend(alerts)

        return alerts

    def get_performance_metrics(self, window_minutes: int = 60) -> ModelMetrics:
        """Calculate performance metrics over time window"""

        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)

        # Filter recent predictions
        recent_predictions = [
            p for p in self.prediction_buffer
            if p["timestamp"] >= cutoff_time
        ]

        if not recent_predictions:
            return ModelMetrics(
                model_id=self.model_id,
                version=self.model_version,
                timestamp=datetime.now()
            )

        # Calculate metrics
        correct_predictions = sum(
            1 for p in recent_predictions
            if p["ground_truth"] is not None and p["prediction"] == p["ground_truth"]
        )

        total_with_truth = sum(
            1 for p in recent_predictions
            if p["ground_truth"] is not None
        )

        accuracy = correct_predictions / total_with_truth if total_with_truth > 0 else None

        # Latency metrics
        recent_latencies = [l for l in self.latency_buffer]
        latency_p50 = float(np.percentile(recent_latencies, 50)) if recent_latencies else None
        latency_p95 = float(np.percentile(recent_latencies, 95)) if recent_latencies else None
        latency_p99 = float(np.percentile(recent_latencies, 99)) if recent_latencies else None

        # Prediction distribution
        positive_preds = sum(1 for p in recent_predictions if p["prediction"] == 1)
        negative_preds = len(recent_predictions) - positive_preds

        return ModelMetrics(
            model_id=self.model_id,
            version=self.model_version,
            timestamp=datetime.now(),
            accuracy=accuracy,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=len(recent_predictions) / window_minutes,
            prediction_count=len(recent_predictions),
            positive_predictions=positive_preds,
            negative_predictions=negative_preds,
        )

    def explain_prediction(self, features: np.ndarray) -> Dict[str, Any]:
        """Explain a prediction"""
        if self.explainer is None:
            return {"error": "Explainer not configured"}

        return self.explainer.explain_prediction(features)

    def get_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get global feature importance"""
        if self.explainer is None:
            return {}

        return self.explainer.get_global_feature_importance(X)

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[DriftAlert]:
        """Get active alerts"""
        if severity:
            return [a for a in self.active_alerts if a.severity == severity]
        return self.active_alerts

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge and remove an alert"""
        self.active_alerts = [a for a in self.active_alerts if a.alert_id != alert_id]

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""

        metrics = self.get_performance_metrics(window_minutes=60)
        critical_alerts = self.get_active_alerts(severity=AlertSeverity.CRITICAL)
        warning_alerts = self.get_active_alerts(severity=AlertSeverity.WARNING)

        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "performance": {
                "accuracy": metrics.accuracy,
                "throughput": metrics.throughput,
                "latency_p95": metrics.latency_p95,
                "prediction_count": metrics.prediction_count,
            },
            "alerts": {
                "critical": len(critical_alerts),
                "warning": len(warning_alerts),
                "total": len(self.active_alerts),
                "recent": [
                    {
                        "severity": a.severity.value,
                        "type": a.drift_type.value,
                        "description": a.description,
                        "timestamp": a.timestamp.isoformat(),
                    }
                    for a in list(self.alert_history)[-10:]
                ],
            },
            "health_score": self._calculate_health_score(metrics, critical_alerts, warning_alerts),
        }

    def _calculate_health_score(
        self,
        metrics: ModelMetrics,
        critical_alerts: List[DriftAlert],
        warning_alerts: List[DriftAlert]
    ) -> float:
        """Calculate overall model health score (0-100)"""

        score = 100.0

        # Deduct for alerts
        score -= len(critical_alerts) * 20
        score -= len(warning_alerts) * 5

        # Deduct for poor performance
        if metrics.accuracy is not None and metrics.accuracy < 0.7:
            score -= (0.7 - metrics.accuracy) * 100

        # Deduct for high latency
        if metrics.latency_p95 is not None and metrics.latency_p95 > 1000:
            score -= min(20, (metrics.latency_p95 - 1000) / 100)

        return max(0.0, min(100.0, score))


# Example usage
async def example_monitoring():
    """Example monitoring workflow"""

    # Create monitor
    monitor = MLMonitor(model_id="fraud_detector", model_version="v1.0.0")

    # Set reference data
    reference_data = pd.DataFrame({
        "transaction_amount": np.random.normal(100, 50, 1000),
        "merchant_risk": np.random.uniform(0, 1, 1000),
        "user_age": np.random.randint(18, 80, 1000),
    })
    monitor.set_reference_data(reference_data)

    # Log predictions
    await monitor.log_prediction(
        features={"transaction_amount": 150, "merchant_risk": 0.3, "user_age": 35},
        prediction=0,
        ground_truth=0,
        latency_ms=45.2
    )

    # Check drift
    current_data = pd.DataFrame({
        "transaction_amount": np.random.normal(120, 60, 100),  # Drift!
        "merchant_risk": np.random.uniform(0, 1, 100),
        "user_age": np.random.randint(18, 80, 100),
    })

    alerts = await monitor.check_drift(current_data)
    print(f"Drift alerts: {len(alerts)}")

    # Get dashboard
    dashboard = monitor.get_monitoring_dashboard()
    print(f"Model health score: {dashboard['health_score']:.1f}")


if __name__ == "__main__":
    asyncio.run(example_monitoring())
