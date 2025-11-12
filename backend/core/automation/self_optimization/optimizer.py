#!/usr/bin/env python3
"""
Self-Optimizing Infrastructure Module
Implements ML-based infrastructure optimization with autonomous tuning
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib


class OptimizationTarget(Enum):
    """Optimization target types"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    NETWORK_LATENCY = "network_latency"
    DISK_IO = "disk_io"
    COST = "cost"
    PERFORMANCE = "performance"


@dataclass
class MetricSample:
    """Single metric sample"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    network_latency: float
    disk_io: float
    active_vms: int
    request_rate: float
    error_rate: float
    cost_per_hour: float


@dataclass
class OptimizationRecommendation:
    """Infrastructure optimization recommendation"""
    target: OptimizationTarget
    current_value: float
    recommended_value: float
    estimated_improvement: float
    confidence: float
    actions: List[str]
    estimated_impact: Dict[str, float]
    priority: str
    timestamp: datetime


@dataclass
class TuningParameter:
    """System tuning parameter"""
    name: str
    current_value: Any
    recommended_value: Any
    min_value: Any
    max_value: Any
    parameter_type: str
    impact_score: float
    validation_required: bool


class InfrastructureOptimizer:
    """
    Self-optimizing infrastructure manager using ML-based optimization
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # ML models for different optimization targets
        self.models: Dict[OptimizationTarget, Any] = {}
        self.scalers: Dict[OptimizationTarget, StandardScaler] = {}

        # Historical data storage
        self.metric_history: List[MetricSample] = []
        self.recommendation_history: List[OptimizationRecommendation] = []

        # Optimization state
        self.learning_enabled = self.config.get('learning_enabled', True)
        self.auto_apply_enabled = self.config.get('auto_apply_enabled', False)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.85)

        # Initialize models
        self._initialize_models()

        self.logger.info("Infrastructure optimizer initialized")

    def _initialize_models(self):
        """Initialize ML models for optimization"""
        targets = [
            OptimizationTarget.CPU_USAGE,
            OptimizationTarget.MEMORY_USAGE,
            OptimizationTarget.NETWORK_LATENCY,
            OptimizationTarget.COST
        ]

        for target in targets:
            # Use Gradient Boosting for better performance prediction
            self.models[target] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            self.scalers[target] = StandardScaler()

    def collect_metrics(self, metrics: Dict[str, Any]) -> MetricSample:
        """
        Collect and store infrastructure metrics

        Args:
            metrics: Current infrastructure metrics

        Returns:
            MetricSample: Processed metric sample
        """
        sample = MetricSample(
            timestamp=datetime.now(),
            cpu_usage=metrics.get('cpu_usage', 0.0),
            memory_usage=metrics.get('memory_usage', 0.0),
            network_latency=metrics.get('network_latency', 0.0),
            disk_io=metrics.get('disk_io', 0.0),
            active_vms=metrics.get('active_vms', 0),
            request_rate=metrics.get('request_rate', 0.0),
            error_rate=metrics.get('error_rate', 0.0),
            cost_per_hour=metrics.get('cost_per_hour', 0.0)
        )

        self.metric_history.append(sample)

        # Keep only last 7 days of data
        cutoff = datetime.now() - timedelta(days=7)
        self.metric_history = [
            s for s in self.metric_history if s.timestamp > cutoff
        ]

        self.logger.debug(f"Collected metrics: {len(self.metric_history)} samples")
        return sample

    def analyze_infrastructure(self) -> List[OptimizationRecommendation]:
        """
        Analyze infrastructure and generate optimization recommendations

        Returns:
            List of optimization recommendations
        """
        if len(self.metric_history) < 100:
            self.logger.warning("Insufficient data for optimization (need 100+ samples)")
            return []

        recommendations = []

        # Analyze each optimization target
        for target in OptimizationTarget:
            try:
                rec = self._analyze_target(target)
                if rec and rec.confidence >= self.confidence_threshold:
                    recommendations.append(rec)
            except Exception as e:
                self.logger.error(f"Error analyzing {target}: {e}")

        # Sort by priority and estimated improvement
        recommendations.sort(
            key=lambda x: (x.priority == 'critical', x.estimated_improvement),
            reverse=True
        )

        self.recommendation_history.extend(recommendations)

        self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
        return recommendations

    def _analyze_target(self, target: OptimizationTarget) -> Optional[OptimizationRecommendation]:
        """Analyze specific optimization target"""

        # Prepare training data
        X, y = self._prepare_training_data(target)

        if len(X) < 50:
            return None

        # Train model
        X_scaled = self.scalers[target].fit_transform(X)
        self.models[target].fit(X_scaled, y)

        # Predict optimal configuration
        current_metrics = self._get_current_metrics()
        current_value = self._get_target_value(target, current_metrics)

        # Generate recommendations
        recommended_value, confidence = self._predict_optimal_value(target, current_metrics)

        if abs(recommended_value - current_value) / current_value < 0.05:
            # Less than 5% improvement
            return None

        improvement = (current_value - recommended_value) / current_value * 100

        actions = self._generate_actions(target, current_value, recommended_value)
        estimated_impact = self._estimate_impact(target, current_value, recommended_value)
        priority = self._calculate_priority(improvement, confidence)

        return OptimizationRecommendation(
            target=target,
            current_value=current_value,
            recommended_value=recommended_value,
            estimated_improvement=improvement,
            confidence=confidence,
            actions=actions,
            estimated_impact=estimated_impact,
            priority=priority,
            timestamp=datetime.now()
        )

    def _prepare_training_data(self, target: OptimizationTarget) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML model"""

        X = []
        y = []

        for sample in self.metric_history:
            features = [
                sample.cpu_usage,
                sample.memory_usage,
                sample.network_latency,
                sample.disk_io,
                sample.active_vms,
                sample.request_rate,
                sample.error_rate,
                sample.cost_per_hour
            ]

            target_value = self._get_target_value(target, sample)

            X.append(features)
            y.append(target_value)

        return np.array(X), np.array(y)

    def _get_target_value(self, target: OptimizationTarget, sample: Any) -> float:
        """Extract target value from sample"""
        if target == OptimizationTarget.CPU_USAGE:
            return sample.cpu_usage
        elif target == OptimizationTarget.MEMORY_USAGE:
            return sample.memory_usage
        elif target == OptimizationTarget.NETWORK_LATENCY:
            return sample.network_latency
        elif target == OptimizationTarget.DISK_IO:
            return sample.disk_io
        elif target == OptimizationTarget.COST:
            return sample.cost_per_hour
        else:
            return 0.0

    def _get_current_metrics(self) -> MetricSample:
        """Get most recent metrics"""
        if not self.metric_history:
            return MetricSample(
                timestamp=datetime.now(),
                cpu_usage=0.0, memory_usage=0.0, network_latency=0.0,
                disk_io=0.0, active_vms=0, request_rate=0.0,
                error_rate=0.0, cost_per_hour=0.0
            )
        return self.metric_history[-1]

    def _predict_optimal_value(self, target: OptimizationTarget,
                              current_metrics: MetricSample) -> Tuple[float, float]:
        """Predict optimal value for target"""

        features = np.array([[
            current_metrics.cpu_usage,
            current_metrics.memory_usage,
            current_metrics.network_latency,
            current_metrics.disk_io,
            current_metrics.active_vms,
            current_metrics.request_rate,
            current_metrics.error_rate,
            current_metrics.cost_per_hour
        ]])

        features_scaled = self.scalers[target].transform(features)
        prediction = self.models[target].predict(features_scaled)[0]

        # Calculate confidence based on model score
        X, y = self._prepare_training_data(target)
        X_scaled = self.scalers[target].transform(X)
        score = self.models[target].score(X_scaled, y)
        confidence = max(0.0, min(1.0, score))

        return prediction, confidence

    def _generate_actions(self, target: OptimizationTarget,
                         current: float, recommended: float) -> List[str]:
        """Generate specific actions for optimization"""

        actions = []

        if target == OptimizationTarget.CPU_USAGE:
            if recommended < current:
                actions.append(f"Reduce CPU allocation from {current:.1f}% to {recommended:.1f}%")
                actions.append("Consider VM consolidation")
                actions.append("Implement CPU throttling for non-critical workloads")

        elif target == OptimizationTarget.MEMORY_USAGE:
            if recommended < current:
                actions.append(f"Reduce memory allocation from {current:.1f}GB to {recommended:.1f}GB")
                actions.append("Enable memory overcommit")
                actions.append("Implement memory ballooning")

        elif target == OptimizationTarget.NETWORK_LATENCY:
            if recommended < current:
                actions.append(f"Target latency reduction from {current:.1f}ms to {recommended:.1f}ms")
                actions.append("Optimize network routing")
                actions.append("Enable network compression")

        elif target == OptimizationTarget.COST:
            if recommended < current:
                actions.append(f"Reduce hourly cost from ${current:.2f} to ${recommended:.2f}")
                actions.append("Right-size VM instances")
                actions.append("Implement auto-scaling policies")

        return actions

    def _estimate_impact(self, target: OptimizationTarget,
                        current: float, recommended: float) -> Dict[str, float]:
        """Estimate impact of optimization"""

        improvement_pct = (current - recommended) / current * 100

        impact = {
            "performance_improvement": improvement_pct,
            "cost_reduction": 0.0,
            "resource_savings": 0.0
        }

        if target == OptimizationTarget.COST:
            impact["cost_reduction"] = improvement_pct
            impact["annual_savings"] = (current - recommended) * 24 * 365

        if target in [OptimizationTarget.CPU_USAGE, OptimizationTarget.MEMORY_USAGE]:
            impact["resource_savings"] = improvement_pct
            impact["capacity_freed"] = improvement_pct

        return impact

    def _calculate_priority(self, improvement: float, confidence: float) -> str:
        """Calculate recommendation priority"""

        score = improvement * confidence

        if score > 30:
            return "critical"
        elif score > 15:
            return "high"
        elif score > 5:
            return "medium"
        else:
            return "low"

    def generate_tuning_parameters(self, target: OptimizationTarget) -> List[TuningParameter]:
        """
        Generate specific tuning parameters for optimization

        Args:
            target: Optimization target

        Returns:
            List of tuning parameters
        """

        parameters = []

        if target == OptimizationTarget.CPU_USAGE:
            parameters.extend([
                TuningParameter(
                    name="vm.cpu.cores",
                    current_value=4,
                    recommended_value=3,
                    min_value=1,
                    max_value=16,
                    parameter_type="integer",
                    impact_score=0.85,
                    validation_required=True
                ),
                TuningParameter(
                    name="cpu.shares",
                    current_value=1024,
                    recommended_value=768,
                    min_value=256,
                    max_value=2048,
                    parameter_type="integer",
                    impact_score=0.65,
                    validation_required=False
                )
            ])

        elif target == OptimizationTarget.MEMORY_USAGE:
            parameters.extend([
                TuningParameter(
                    name="vm.memory.size_gb",
                    current_value=16,
                    recommended_value=12,
                    min_value=4,
                    max_value=64,
                    parameter_type="integer",
                    impact_score=0.90,
                    validation_required=True
                ),
                TuningParameter(
                    name="memory.balloon.enabled",
                    current_value=False,
                    recommended_value=True,
                    min_value=False,
                    max_value=True,
                    parameter_type="boolean",
                    impact_score=0.70,
                    validation_required=False
                )
            ])

        return parameters

    def apply_optimization(self, recommendation: OptimizationRecommendation,
                          dry_run: bool = True) -> Dict[str, Any]:
        """
        Apply optimization recommendation

        Args:
            recommendation: Recommendation to apply
            dry_run: If True, simulate without applying

        Returns:
            Application result
        """

        result = {
            "success": False,
            "target": recommendation.target.value,
            "actions_taken": [],
            "dry_run": dry_run,
            "timestamp": datetime.now().isoformat()
        }

        if dry_run:
            result["success"] = True
            result["actions_taken"] = recommendation.actions
            result["message"] = "Dry run completed successfully"
            self.logger.info(f"Dry run optimization for {recommendation.target.value}")
            return result

        if not self.auto_apply_enabled:
            result["message"] = "Auto-apply disabled, manual approval required"
            self.logger.warning("Optimization requires manual approval")
            return result

        if recommendation.confidence < self.confidence_threshold:
            result["message"] = f"Confidence {recommendation.confidence:.2f} below threshold"
            self.logger.warning("Optimization confidence too low")
            return result

        # Apply optimization actions
        try:
            for action in recommendation.actions:
                # In real implementation, execute actual optimization
                self.logger.info(f"Applying action: {action}")
                result["actions_taken"].append(action)

            result["success"] = True
            result["message"] = "Optimization applied successfully"

        except Exception as e:
            result["message"] = f"Optimization failed: {str(e)}"
            self.logger.error(f"Optimization error: {e}")

        return result

    def export_metrics(self) -> Dict[str, Any]:
        """Export optimizer metrics"""

        total_recommendations = len(self.recommendation_history)
        high_confidence = len([r for r in self.recommendation_history
                              if r.confidence >= 0.9])

        avg_improvement = 0.0
        if total_recommendations > 0:
            avg_improvement = sum(r.estimated_improvement
                                 for r in self.recommendation_history) / total_recommendations

        return {
            "total_samples": len(self.metric_history),
            "total_recommendations": total_recommendations,
            "high_confidence_recommendations": high_confidence,
            "average_improvement": avg_improvement,
            "learning_enabled": self.learning_enabled,
            "auto_apply_enabled": self.auto_apply_enabled,
            "confidence_threshold": self.confidence_threshold
        }

    def save_models(self, directory: str):
        """Save trained models to disk"""
        import os
        os.makedirs(directory, exist_ok=True)

        for target, model in self.models.items():
            model_path = os.path.join(directory, f"{target.value}_model.pkl")
            scaler_path = os.path.join(directory, f"{target.value}_scaler.pkl")

            joblib.dump(model, model_path)
            joblib.dump(self.scalers[target], scaler_path)

        self.logger.info(f"Models saved to {directory}")

    def load_models(self, directory: str):
        """Load trained models from disk"""
        import os

        for target in OptimizationTarget:
            model_path = os.path.join(directory, f"{target.value}_model.pkl")
            scaler_path = os.path.join(directory, f"{target.value}_scaler.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.models[target] = joblib.load(model_path)
                self.scalers[target] = joblib.load(scaler_path)

        self.logger.info(f"Models loaded from {directory}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize optimizer
    optimizer = InfrastructureOptimizer({
        'learning_enabled': True,
        'auto_apply_enabled': False,
        'confidence_threshold': 0.85
    })

    # Simulate metric collection
    for i in range(200):
        metrics = {
            'cpu_usage': 70 + np.random.randn() * 10,
            'memory_usage': 8 + np.random.randn() * 2,
            'network_latency': 50 + np.random.randn() * 5,
            'disk_io': 100 + np.random.randn() * 20,
            'active_vms': 10,
            'request_rate': 1000 + np.random.randn() * 100,
            'error_rate': 0.01,
            'cost_per_hour': 50 + np.random.randn() * 5
        }
        optimizer.collect_metrics(metrics)

    # Analyze and generate recommendations
    recommendations = optimizer.analyze_infrastructure()

    print(f"\nGenerated {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"\n{rec.target.value}:")
        print(f"  Current: {rec.current_value:.2f}")
        print(f"  Recommended: {rec.recommended_value:.2f}")
        print(f"  Improvement: {rec.estimated_improvement:.1f}%")
        print(f"  Confidence: {rec.confidence:.2f}")
        print(f"  Priority: {rec.priority}")
        print(f"  Actions: {', '.join(rec.actions)}")

    # Export metrics
    metrics = optimizer.export_metrics()
    print(f"\nOptimizer metrics: {json.dumps(metrics, indent=2)}")
