"""
Metrics calculation utilities for the AI Operations Engine.

Provides comprehensive metrics calculation for model performance,
business impact assessment, and system monitoring.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)


logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Comprehensive metrics calculator for ML models and system performance."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels  
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of classification metrics
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'specificity': self._calculate_specificity(y_true, y_pred),
                'balanced_accuracy': self._calculate_balanced_accuracy(y_true, y_pred)
            }
            
            # Add probability-based metrics if available
            if y_proba is not None:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                    metrics['pr_auc'] = average_precision_score(y_true, y_proba)
                except ValueError as e:
                    logger.warning(f"Could not calculate AUC metrics: {str(e)}")
                    metrics['roc_auc'] = 0.0
                    metrics['pr_auc'] = 0.0
            
            # Confusion matrix metrics
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):  # Binary classification
                tn, fp, fn, tp = cm.ravel()
                metrics.update({
                    'true_positives': int(tp),
                    'true_negatives': int(tn), 
                    'false_positives': int(fp),
                    'false_negatives': int(fn),
                    'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
                    'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0.0
                })
            
            logger.debug(f"Calculated classification metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {str(e)}")
            return {}
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True continuous values
            y_pred: Predicted continuous values
            
        Returns:
            Dictionary of regression metrics
        """
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2_score': r2_score(y_true, y_pred),
                'mape': self._calculate_mape(y_true, y_pred),
                'median_absolute_error': self._calculate_median_absolute_error(y_true, y_pred),
                'explained_variance': self._calculate_explained_variance(y_true, y_pred)
            }
            
            logger.debug(f"Calculated regression metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {str(e)}")
            return {}
    
    def calculate_anomaly_detection_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                          y_scores: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics specific to anomaly detection.
        
        Args:
            y_true: True anomaly labels (1 = anomaly, 0 = normal)
            y_pred: Predicted anomaly labels
            y_scores: Anomaly scores
            
        Returns:
            Dictionary of anomaly detection metrics
        """
        try:
            # Basic classification metrics
            metrics = self.calculate_classification_metrics(y_true, y_pred, y_scores)
            
            # Anomaly-specific metrics
            metrics.update({
                'detection_rate': recall_score(y_true, y_pred, zero_division=0),  # Same as recall
                'false_alarm_rate': self._calculate_false_alarm_rate(y_true, y_pred),
                'precision_at_k': self._calculate_precision_at_k(y_true, y_scores, k=100),
                'coverage': self._calculate_coverage(y_true, y_pred),
                'lift': self._calculate_lift(y_true, y_pred)
            })
            
            logger.debug(f"Calculated anomaly detection metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating anomaly detection metrics: {str(e)}")
            return {}
    
    def calculate_placement_metrics(self, actual_performance: np.ndarray, 
                                  predicted_performance: np.ndarray,
                                  actual_costs: np.ndarray,
                                  predicted_costs: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics specific to workload placement optimization.
        
        Args:
            actual_performance: Actual performance outcomes
            predicted_performance: Predicted performance scores
            actual_costs: Actual costs incurred
            predicted_costs: Predicted costs
            
        Returns:
            Dictionary of placement optimization metrics
        """
        try:
            metrics = {}
            
            # Performance prediction accuracy
            if len(actual_performance) > 0 and len(predicted_performance) > 0:
                perf_metrics = self.calculate_regression_metrics(actual_performance, predicted_performance)
                metrics.update({f'performance_{k}': v for k, v in perf_metrics.items()})
            
            # Cost prediction accuracy  
            if len(actual_costs) > 0 and len(predicted_costs) > 0:
                cost_metrics = self.calculate_regression_metrics(actual_costs, predicted_costs)
                metrics.update({f'cost_{k}': v for k, v in cost_metrics.items()})
            
            # Placement-specific metrics
            if len(actual_performance) > 0 and len(actual_costs) > 0:
                metrics.update({
                    'cost_performance_ratio': np.mean(actual_costs) / (np.mean(actual_performance) + 1e-6),
                    'efficiency_score': np.mean(actual_performance) / (np.mean(actual_costs) + 1e-6),
                    'placement_satisfaction': self._calculate_placement_satisfaction(
                        actual_performance, actual_costs
                    )
                })
            
            logger.debug(f"Calculated placement metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating placement metrics: {str(e)}")
            return {}
    
    def calculate_resource_optimization_metrics(self, 
                                              baseline_utilization: np.ndarray,
                                              optimized_utilization: np.ndarray,
                                              baseline_costs: np.ndarray,
                                              optimized_costs: np.ndarray,
                                              sla_violations_baseline: int = 0,
                                              sla_violations_optimized: int = 0) -> Dict[str, float]:
        """
        Calculate metrics for resource optimization effectiveness.
        
        Args:
            baseline_utilization: Resource utilization before optimization
            optimized_utilization: Resource utilization after optimization
            baseline_costs: Costs before optimization
            optimized_costs: Costs after optimization
            sla_violations_baseline: SLA violations before optimization
            sla_violations_optimized: SLA violations after optimization
            
        Returns:
            Dictionary of optimization metrics
        """
        try:
            metrics = {}
            
            # Utilization improvements
            if len(baseline_utilization) > 0 and len(optimized_utilization) > 0:
                metrics.update({
                    'utilization_improvement': (np.mean(optimized_utilization) - 
                                              np.mean(baseline_utilization)),
                    'utilization_efficiency_gain': ((np.mean(optimized_utilization) - 
                                                   np.mean(baseline_utilization)) / 
                                                  (np.mean(baseline_utilization) + 1e-6)),
                    'utilization_variance_reduction': (np.var(baseline_utilization) - 
                                                     np.var(optimized_utilization))
                })
            
            # Cost improvements
            if len(baseline_costs) > 0 and len(optimized_costs) > 0:
                cost_savings = np.sum(baseline_costs) - np.sum(optimized_costs)
                metrics.update({
                    'cost_savings_absolute': cost_savings,
                    'cost_savings_percentage': cost_savings / (np.sum(baseline_costs) + 1e-6),
                    'cost_efficiency_improvement': (np.mean(baseline_costs) - np.mean(optimized_costs)) / 
                                                 (np.mean(baseline_costs) + 1e-6)
                })
            
            # SLA and performance metrics
            metrics.update({
                'sla_violation_reduction': sla_violations_baseline - sla_violations_optimized,
                'sla_improvement_rate': ((sla_violations_baseline - sla_violations_optimized) / 
                                       (sla_violations_baseline + 1)) if sla_violations_baseline > 0 else 0.0
            })
            
            # Overall optimization score
            utilization_score = metrics.get('utilization_efficiency_gain', 0)
            cost_score = metrics.get('cost_savings_percentage', 0)
            sla_score = metrics.get('sla_improvement_rate', 0)
            
            metrics['optimization_score'] = (utilization_score * 0.4 + 
                                           cost_score * 0.4 + 
                                           sla_score * 0.2)
            
            logger.debug(f"Calculated resource optimization metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating resource optimization metrics: {str(e)}")
            return {}
    
    def calculate_business_impact_metrics(self, 
                                        downtime_hours: float = 0.0,
                                        performance_degradation: float = 0.0,
                                        cost_savings: float = 0.0,
                                        revenue_per_hour: float = 1000.0,
                                        user_satisfaction_change: float = 0.0) -> Dict[str, float]:
        """
        Calculate business impact metrics for AI operations.
        
        Args:
            downtime_hours: Hours of downtime prevented/caused
            performance_degradation: Performance impact (0-1 scale)
            cost_savings: Direct cost savings in dollars
            revenue_per_hour: Revenue impact per hour
            user_satisfaction_change: Change in user satisfaction (-1 to 1)
            
        Returns:
            Dictionary of business impact metrics
        """
        try:
            # Direct financial impact
            downtime_cost_avoided = downtime_hours * revenue_per_hour
            performance_revenue_impact = performance_degradation * revenue_per_hour * 24  # Daily impact
            
            metrics = {
                'downtime_cost_avoided': downtime_cost_avoided,
                'performance_revenue_impact': performance_revenue_impact,
                'direct_cost_savings': cost_savings,
                'total_financial_impact': downtime_cost_avoided + cost_savings - performance_revenue_impact,
                'roi': ((cost_savings + downtime_cost_avoided) / (cost_savings + 1)) - 1,
                'user_satisfaction_impact': user_satisfaction_change,
                'business_value_score': self._calculate_business_value_score(
                    downtime_cost_avoided, cost_savings, user_satisfaction_change
                )
            }
            
            logger.debug(f"Calculated business impact metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating business impact metrics: {str(e)}")
            return {}
    
    def calculate_model_drift_metrics(self, reference_predictions: np.ndarray,
                                    current_predictions: np.ndarray,
                                    reference_features: np.ndarray,
                                    current_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate model drift detection metrics.
        
        Args:
            reference_predictions: Predictions from reference period
            current_predictions: Predictions from current period
            reference_features: Features from reference period
            current_features: Features from current period
            
        Returns:
            Dictionary of drift metrics
        """
        try:
            metrics = {}
            
            # Prediction drift
            if len(reference_predictions) > 0 and len(current_predictions) > 0:
                metrics.update({
                    'prediction_drift_psi': self._calculate_psi(reference_predictions, current_predictions),
                    'prediction_drift_js': self._calculate_js_divergence(reference_predictions, current_predictions),
                    'prediction_mean_shift': np.mean(current_predictions) - np.mean(reference_predictions),
                    'prediction_variance_ratio': np.var(current_predictions) / (np.var(reference_predictions) + 1e-6)
                })
            
            # Feature drift
            if reference_features.shape[0] > 0 and current_features.shape[0] > 0:
                feature_drift_scores = []
                for i in range(min(reference_features.shape[1], current_features.shape[1])):
                    ref_feature = reference_features[:, i]
                    cur_feature = current_features[:, i]
                    
                    psi = self._calculate_psi(ref_feature, cur_feature)
                    feature_drift_scores.append(psi)
                
                if feature_drift_scores:
                    metrics.update({
                        'feature_drift_max': max(feature_drift_scores),
                        'feature_drift_mean': np.mean(feature_drift_scores),
                        'feature_drift_count': sum(1 for score in feature_drift_scores if score > 0.1)
                    })
            
            # Overall drift score
            drift_components = []
            if 'prediction_drift_psi' in metrics:
                drift_components.append(metrics['prediction_drift_psi'])
            if 'feature_drift_mean' in metrics:
                drift_components.append(metrics['feature_drift_mean'])
            
            if drift_components:
                metrics['overall_drift_score'] = np.mean(drift_components)
            
            logger.debug(f"Calculated model drift metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating model drift metrics: {str(e)}")
            return {}
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return 0.0
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy."""
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (sensitivity + specificity) / 2.0
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    
    def _calculate_median_absolute_error(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Median Absolute Error."""
        return np.median(np.abs(y_true - y_pred))
    
    def _calculate_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate explained variance."""
        var_y = np.var(y_true)
        var_residual = np.var(y_true - y_pred)
        return 1 - (var_residual / (var_y + 1e-6))
    
    def _calculate_false_alarm_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate false alarm rate for anomaly detection."""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return fp / (fp + tn) if (fp + tn) > 0 else 0.0
        return 0.0
    
    def _calculate_precision_at_k(self, y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
        """Calculate precision at k for anomaly detection."""
        # Sort by scores (descending)
        sorted_indices = np.argsort(y_scores)[::-1]
        top_k_indices = sorted_indices[:k]
        
        if len(top_k_indices) == 0:
            return 0.0
        
        return np.sum(y_true[top_k_indices]) / len(top_k_indices)
    
    def _calculate_coverage(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate coverage (percentage of anomalies detected)."""
        total_anomalies = np.sum(y_true)
        detected_anomalies = np.sum(y_true * y_pred)
        return detected_anomalies / total_anomalies if total_anomalies > 0 else 0.0
    
    def _calculate_lift(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate lift for anomaly detection."""
        baseline_rate = np.mean(y_true)
        detected_rate = np.sum(y_true * y_pred) / np.sum(y_pred) if np.sum(y_pred) > 0 else 0
        return detected_rate / baseline_rate if baseline_rate > 0 else 0.0
    
    def _calculate_placement_satisfaction(self, performance: np.ndarray, costs: np.ndarray) -> float:
        """Calculate placement satisfaction score."""
        # Normalize performance and costs
        perf_norm = (performance - np.min(performance)) / (np.max(performance) - np.min(performance) + 1e-6)
        cost_norm = (costs - np.min(costs)) / (np.max(costs) - np.min(costs) + 1e-6)
        
        # Satisfaction is high performance with low cost
        satisfaction = np.mean(perf_norm - cost_norm)
        return np.clip(satisfaction, -1, 1)
    
    def _calculate_business_value_score(self, downtime_avoided: float, 
                                      cost_savings: float, user_satisfaction: float) -> float:
        """Calculate overall business value score."""
        # Normalize components to 0-1 scale
        downtime_score = np.clip(downtime_avoided / 10000, 0, 1)  # Assume $10k max
        cost_score = np.clip(cost_savings / 5000, 0, 1)  # Assume $5k max
        satisfaction_score = (user_satisfaction + 1) / 2  # Convert -1,1 to 0,1
        
        # Weighted combination
        return 0.4 * downtime_score + 0.4 * cost_score + 0.2 * satisfaction_score
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, 
                      buckets: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) for drift detection.
        
        Args:
            reference: Reference distribution
            current: Current distribution  
            buckets: Number of buckets for discretization
            
        Returns:
            PSI value (>0.1 indicates drift)
        """
        try:
            # Create bins based on reference distribution
            bin_edges = np.percentile(reference, np.linspace(0, 100, buckets + 1))
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
            
            # Calculate distributions
            ref_counts = np.histogram(reference, bins=bin_edges)[0]
            cur_counts = np.histogram(current, bins=bin_edges)[0]
            
            # Calculate proportions (add small constant to avoid log(0))
            ref_props = (ref_counts + 1e-6) / (len(reference) + buckets * 1e-6)
            cur_props = (cur_counts + 1e-6) / (len(current) + buckets * 1e-6)
            
            # Calculate PSI
            psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
            
            return float(psi)
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_js_divergence(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence for drift detection."""
        try:
            from scipy.spatial.distance import jensenshannon
            
            # Create histograms
            combined_range = (min(np.min(reference), np.min(current)),
                            max(np.max(reference), np.max(current)))
            bins = np.linspace(combined_range[0], combined_range[1], 50)
            
            ref_hist = np.histogram(reference, bins=bins)[0]
            cur_hist = np.histogram(current, bins=bins)[0]
            
            # Normalize to probability distributions
            ref_dist = (ref_hist + 1e-6) / (np.sum(ref_hist) + 50 * 1e-6)
            cur_dist = (cur_hist + 1e-6) / (np.sum(cur_hist) + 50 * 1e-6)
            
            # Calculate JS divergence
            js_div = jensenshannon(ref_dist, cur_dist)
            
            return float(js_div ** 2)  # Square to get JS divergence
            
        except Exception as e:
            logger.warning(f"JS divergence calculation failed: {str(e)}")
            return 0.0


class SystemMetricsCollector:
    """Collect and aggregate system performance metrics."""
    
    def __init__(self):
        """Initialize system metrics collector."""
        self.metrics_history: List[Dict[str, Any]] = []
        self._max_history = 10000
    
    def collect_prediction_metrics(self, service_type: str, model_id: str,
                                 response_time: float, prediction: Any,
                                 confidence: Optional[float] = None) -> None:
        """
        Collect metrics from prediction requests.
        
        Args:
            service_type: Type of AI service
            model_id: Model identifier
            response_time: Response time in seconds
            prediction: Prediction result
            confidence: Prediction confidence
        """
        metric_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'service_type': service_type,
            'model_id': model_id,
            'response_time': response_time,
            'prediction_type': type(prediction).__name__,
            'confidence': confidence,
            'has_prediction': prediction is not None
        }
        
        self.metrics_history.append(metric_record)
        
        # Maintain history size
        if len(self.metrics_history) > self._max_history:
            self.metrics_history = self.metrics_history[-self._max_history:]
    
    def get_service_performance_summary(self, service_type: str,
                                      time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """
        Get performance summary for a specific service.
        
        Args:
            service_type: Service to analyze
            time_window: Time window for analysis
            
        Returns:
            Service performance summary
        """
        cutoff_time = datetime.utcnow() - time_window
        
        # Filter metrics for service and time window
        relevant_metrics = [
            m for m in self.metrics_history
            if (m['service_type'] == service_type and
                datetime.fromisoformat(m['timestamp']) > cutoff_time)
        ]
        
        if not relevant_metrics:
            return {
                'service_type': service_type,
                'total_requests': 0,
                'avg_response_time': 0.0,
                'success_rate': 0.0,
                'avg_confidence': 0.0
            }
        
        # Calculate summary statistics
        response_times = [m['response_time'] for m in relevant_metrics]
        confidences = [m['confidence'] for m in relevant_metrics if m['confidence'] is not None]
        success_count = sum(1 for m in relevant_metrics if m['has_prediction'])
        
        summary = {
            'service_type': service_type,
            'total_requests': len(relevant_metrics),
            'avg_response_time': np.mean(response_times),
            'min_response_time': np.min(response_times),
            'max_response_time': np.max(response_times),
            'p95_response_time': np.percentile(response_times, 95),
            'success_rate': success_count / len(relevant_metrics),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'time_window_hours': time_window.total_seconds() / 3600
        }
        
        return summary
    
    def get_overall_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        if not self.metrics_history:
            return {'status': 'no_data', 'services': []}
        
        # Get unique services
        services = list(set(m['service_type'] for m in self.metrics_history))
        
        service_summaries = []
        for service in services:
            summary = self.get_service_performance_summary(service)
            service_summaries.append(summary)
        
        # Calculate overall metrics
        all_response_times = [m['response_time'] for m in self.metrics_history[-1000:]]  # Last 1000 requests
        overall_success_rate = sum(1 for m in self.metrics_history[-1000:] if m['has_prediction']) / min(1000, len(self.metrics_history))
        
        health = {
            'status': 'healthy' if overall_success_rate > 0.95 else 'degraded' if overall_success_rate > 0.8 else 'unhealthy',
            'overall_success_rate': overall_success_rate,
            'overall_avg_response_time': np.mean(all_response_times) if all_response_times else 0.0,
            'active_services': len(services),
            'total_requests_recent': len(self.metrics_history[-1000:]),
            'services': service_summaries
        }
        
        return health