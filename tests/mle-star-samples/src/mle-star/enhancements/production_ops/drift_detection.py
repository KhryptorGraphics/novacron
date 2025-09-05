"""
Data Drift Detection
==================

Data drift detection and monitoring for production ML models:
- Statistical drift detection methods
- Distribution comparison techniques
- Feature-wise drift analysis
- Concept drift detection
- Real-time monitoring and alerting
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path

try:
    from scipy import stats
    from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift"""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftSeverity(Enum):
    """Severity levels for drift detection"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Drift detection alert"""
    alert_id: str
    drift_type: DriftType
    severity: DriftSeverity
    feature_name: str
    drift_score: float
    p_value: float
    threshold: float
    detected_at: datetime
    description: str
    recommendation: str


@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection"""
    # Detection methods
    statistical_tests: List[str] = field(default_factory=lambda: ["ks_test", "wasserstein", "psi"])
    
    # Thresholds
    drift_threshold: float = 0.05  # P-value threshold
    psi_threshold: float = 0.2     # Population Stability Index threshold
    wasserstein_threshold: float = 0.1
    
    # Monitoring configuration
    monitoring_window_hours: int = 24
    reference_window_days: int = 7
    min_samples_for_test: int = 100
    
    # Feature analysis
    categorical_features: List[str] = field(default_factory=list)
    numerical_features: List[str] = field(default_factory=list)
    ignore_features: List[str] = field(default_factory=list)
    
    # Concept drift
    concept_drift_window: int = 1000  # Number of samples for concept drift detection
    concept_drift_threshold: float = 0.05
    
    # Alerting
    enable_alerts: bool = True
    alert_cooldown_hours: int = 6
    
    # Storage
    store_drift_reports: bool = True
    drift_reports_path: str = "./drift_reports"
    max_reference_samples: int = 10000


class StatisticalDriftDetector:
    """Statistical methods for drift detection"""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
    
    def detect_numerical_drift(self, reference_data: np.ndarray, 
                              current_data: np.ndarray, 
                              feature_name: str) -> Dict[str, float]:
        """Detect drift in numerical features"""
        results = {}
        
        # Kolmogorov-Smirnov test
        if "ks_test" in self.config.statistical_tests and SCIPY_AVAILABLE:
            ks_stat, ks_p_value = ks_2samp(reference_data, current_data)
            results["ks_statistic"] = ks_stat
            results["ks_p_value"] = ks_p_value
        
        # Wasserstein distance
        if "wasserstein" in self.config.statistical_tests and SCIPY_AVAILABLE:
            wasserstein_dist = wasserstein_distance(reference_data, current_data)
            results["wasserstein_distance"] = wasserstein_dist
        
        # Population Stability Index (PSI)
        if "psi" in self.config.statistical_tests:
            psi_score = self._calculate_psi(reference_data, current_data)
            results["psi_score"] = psi_score
        
        # Mean and variance shift detection
        ref_mean, ref_std = np.mean(reference_data), np.std(reference_data)
        curr_mean, curr_std = np.mean(current_data), np.std(current_data)
        
        # Z-test for mean difference (if scipy available)
        if SCIPY_AVAILABLE:
            pooled_se = np.sqrt(ref_std**2/len(reference_data) + curr_std**2/len(current_data))
            if pooled_se > 0:
                z_stat = abs(curr_mean - ref_mean) / pooled_se
                z_p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                results["mean_shift_z_score"] = z_stat
                results["mean_shift_p_value"] = z_p_value
        
        # Simple ratio-based checks
        results["mean_ratio"] = curr_mean / ref_mean if ref_mean != 0 else float('inf')
        results["std_ratio"] = curr_std / ref_std if ref_std != 0 else float('inf')
        
        return results
    
    def detect_categorical_drift(self, reference_data: np.ndarray,
                               current_data: np.ndarray,
                               feature_name: str) -> Dict[str, float]:
        """Detect drift in categorical features"""
        results = {}
        
        # Get value counts
        ref_values, ref_counts = np.unique(reference_data, return_counts=True)
        curr_values, curr_counts = np.unique(current_data, return_counts=True)
        
        # Create combined set of categories
        all_categories = np.union1d(ref_values, curr_values)
        
        # Build frequency tables
        ref_freq = np.zeros(len(all_categories))
        curr_freq = np.zeros(len(all_categories))
        
        for i, cat in enumerate(all_categories):
            ref_idx = np.where(ref_values == cat)[0]
            curr_idx = np.where(curr_values == cat)[0]
            
            ref_freq[i] = ref_counts[ref_idx[0]] if len(ref_idx) > 0 else 0
            curr_freq[i] = curr_counts[curr_idx[0]] if len(curr_idx) > 0 else 0
        
        # Chi-square test
        if SCIPY_AVAILABLE and np.sum(ref_freq) > 0 and np.sum(curr_freq) > 0:
            try:
                # Create contingency table
                contingency_table = np.array([ref_freq, curr_freq])
                chi2_stat, chi2_p_value, _, _ = chi2_contingency(contingency_table)
                results["chi2_statistic"] = chi2_stat
                results["chi2_p_value"] = chi2_p_value
            except Exception as e:
                logger.warning(f"Chi-square test failed for {feature_name}: {e}")
        
        # PSI for categorical data
        psi_score = self._calculate_categorical_psi(ref_freq, curr_freq)
        results["psi_score"] = psi_score
        
        # Jensen-Shannon divergence
        js_divergence = self._calculate_js_divergence(ref_freq, curr_freq)
        results["js_divergence"] = js_divergence
        
        return results
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index for numerical data"""
        try:
            # Create bins based on reference data
            bin_edges = np.histogram_bin_edges(reference, bins=bins)
            
            # Calculate histograms
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to proportions
            ref_props = ref_counts / np.sum(ref_counts)
            curr_props = curr_counts / np.sum(curr_counts)
            
            # Calculate PSI
            psi = 0
            for ref_prop, curr_prop in zip(ref_props, curr_props):
                # Avoid division by zero
                if ref_prop == 0:
                    ref_prop = 0.0001
                if curr_prop == 0:
                    curr_prop = 0.0001
                
                psi += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
            
            return psi
            
        except Exception as e:
            logger.warning(f"PSI calculation failed: {e}")
            return 0.0
    
    def _calculate_categorical_psi(self, ref_freq: np.ndarray, curr_freq: np.ndarray) -> float:
        """Calculate PSI for categorical data"""
        try:
            # Convert to proportions
            ref_props = ref_freq / np.sum(ref_freq) if np.sum(ref_freq) > 0 else ref_freq
            curr_props = curr_freq / np.sum(curr_freq) if np.sum(curr_freq) > 0 else curr_freq
            
            psi = 0
            for ref_prop, curr_prop in zip(ref_props, curr_props):
                # Avoid division by zero
                if ref_prop == 0:
                    ref_prop = 0.0001
                if curr_prop == 0:
                    curr_prop = 0.0001
                
                psi += (curr_prop - ref_prop) * np.log(curr_prop / ref_prop)
            
            return psi
            
        except Exception as e:
            logger.warning(f"Categorical PSI calculation failed: {e}")
            return 0.0
    
    def _calculate_js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Calculate Jensen-Shannon divergence"""
        try:
            # Normalize to probabilities
            p = p / np.sum(p) if np.sum(p) > 0 else p + 1e-10
            q = q / np.sum(q) if np.sum(q) > 0 else q + 1e-10
            
            # Add small constant to avoid log(0)
            p = p + 1e-10
            q = q + 1e-10
            
            # Calculate KL divergences
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * np.log(p / m))
            kl_qm = np.sum(q * np.log(q / m))
            
            # Jensen-Shannon divergence
            js_div = 0.5 * kl_pm + 0.5 * kl_qm
            
            return js_div
            
        except Exception as e:
            logger.warning(f"JS divergence calculation failed: {e}")
            return 0.0


class ConceptDriftDetector:
    """Concept drift detection using model performance"""
    
    def __init__(self, config: DriftDetectionConfig):
        self.config = config
        self.performance_history = []
    
    def detect_concept_drift(self, model: Any, X: np.ndarray, y: np.ndarray,
                           performance_metric: str = "accuracy") -> Dict[str, Any]:
        """Detect concept drift using sliding window performance"""
        try:
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(X)
            else:
                raise ValueError("Model does not have predict method")
            
            # Calculate performance
            if performance_metric == "accuracy":
                current_performance = accuracy_score(y, predictions)
            elif performance_metric == "precision":
                current_performance = precision_score(y, predictions, average='weighted', zero_division=0)
            elif performance_metric == "recall":
                current_performance = recall_score(y, predictions, average='weighted', zero_division=0)
            else:
                raise ValueError(f"Unsupported metric: {performance_metric}")
            
            # Add to history
            self.performance_history.append({
                'timestamp': datetime.now(),
                'performance': current_performance,
                'sample_size': len(y)
            })
            
            # Keep only recent history
            window_size = self.config.concept_drift_window
            if len(self.performance_history) > window_size:
                self.performance_history = self.performance_history[-window_size:]
            
            # Detect drift if we have enough history
            drift_detected = False
            drift_score = 0.0
            
            if len(self.performance_history) >= 2:
                # Simple approach: compare recent performance with baseline
                recent_performances = [h['performance'] for h in self.performance_history[-10:]]
                baseline_performances = [h['performance'] for h in self.performance_history[:10]]
                
                if len(baseline_performances) > 0:
                    baseline_mean = np.mean(baseline_performances)
                    recent_mean = np.mean(recent_performances)
                    
                    # Calculate relative drop
                    if baseline_mean > 0:
                        relative_drop = (baseline_mean - recent_mean) / baseline_mean
                        drift_score = relative_drop
                        
                        if relative_drop > self.config.concept_drift_threshold:
                            drift_detected = True
            
            return {
                'drift_detected': drift_detected,
                'drift_score': drift_score,
                'current_performance': current_performance,
                'baseline_performance': np.mean([h['performance'] for h in self.performance_history[:10]]) if len(self.performance_history) >= 10 else current_performance,
                'performance_trend': self._calculate_trend()
            }
            
        except Exception as e:
            logger.error(f"Concept drift detection failed: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'current_performance': 0.0,
                'error': str(e)
            }
    
    def _calculate_trend(self) -> str:
        """Calculate performance trend"""
        if len(self.performance_history) < 3:
            return "insufficient_data"
        
        recent_performances = [h['performance'] for h in self.performance_history[-5:]]
        
        # Simple trend calculation
        if len(recent_performances) >= 2:
            slope = np.polyfit(range(len(recent_performances)), recent_performances, 1)[0]
            
            if slope > 0.001:
                return "improving"
            elif slope < -0.001:
                return "degrading"
            else:
                return "stable"
        
        return "unknown"


class DriftDetector(BaseEnhancement):
    """Data Drift Detection enhancement for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="drift_detector",
            version="1.0.0",
            enabled=True,
            priority=5,
            parameters={
                "statistical_tests": ["ks_test", "wasserstein", "psi"],
                "drift_threshold": 0.05,
                "psi_threshold": 0.2,
                "wasserstein_threshold": 0.1,
                "monitoring_window_hours": 24,
                "reference_window_days": 7,
                "min_samples_for_test": 100,
                "categorical_features": [],
                "numerical_features": [],
                "ignore_features": [],
                "concept_drift_window": 1000,
                "concept_drift_threshold": 0.05,
                "enable_alerts": True,
                "alert_cooldown_hours": 6,
                "store_drift_reports": True,
                "drift_reports_path": "./drift_reports",
                "max_reference_samples": 10000
            }
        )
    
    def initialize(self) -> bool:
        """Initialize Drift Detector"""
        try:
            # Create configuration
            self.drift_config = DriftDetectionConfig(**self.config.parameters)
            
            # Initialize detectors
            self.statistical_detector = StatisticalDriftDetector(self.drift_config)
            self.concept_detector = ConceptDriftDetector(self.drift_config)
            
            # Storage for reference data and alerts
            self.reference_data: Optional[pd.DataFrame] = None
            self.reference_timestamp: Optional[datetime] = None
            self.active_alerts: List[DriftAlert] = []
            
            # Create reports directory
            Path(self.drift_config.drift_reports_path).mkdir(parents=True, exist_ok=True)
            
            self._logger.info("Drift Detector initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Drift Detector: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with drift detection capabilities"""
        enhanced = workflow.copy()
        
        # Add drift detection configuration
        if 'drift_detection' not in enhanced:
            enhanced['drift_detection'] = {}
        
        enhanced['drift_detection'] = {
            'enabled': True,
            'statistical_tests': self.drift_config.statistical_tests,
            'thresholds': {
                'drift_threshold': self.drift_config.drift_threshold,
                'psi_threshold': self.drift_config.psi_threshold,
                'wasserstein_threshold': self.drift_config.wasserstein_threshold
            },
            'monitoring': {
                'window_hours': self.drift_config.monitoring_window_hours,
                'reference_window_days': self.drift_config.reference_window_days,
                'min_samples': self.drift_config.min_samples_for_test
            },
            'concept_drift': {
                'window_size': self.drift_config.concept_drift_window,
                'threshold': self.drift_config.concept_drift_threshold
            },
            'alerting': {
                'enabled': self.drift_config.enable_alerts,
                'cooldown_hours': self.drift_config.alert_cooldown_hours
            }
        }
        
        # Enhance MLE-Star stages with drift detection
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 4: Implementation - Add baseline establishment
            if '4_implementation' in stages:
                if 'drift_baseline_establishment' not in stages['4_implementation']:
                    stages['4_implementation']['drift_baseline_establishment'] = [
                        'reference_data_collection',
                        'feature_distribution_profiling',
                        'baseline_performance_measurement',
                        'drift_monitoring_setup'
                    ]
            
            # Stage 7: Deployment Prep - Add drift monitoring setup
            if '7_deployment_prep' in stages:
                if 'production_drift_monitoring' not in stages['7_deployment_prep']:
                    stages['7_deployment_prep']['production_drift_monitoring'] = [
                        'drift_detection_pipeline_setup',
                        'alert_configuration',
                        'monitoring_dashboard_integration',
                        'automated_response_triggers'
                    ]
        
        self._logger.debug("Enhanced workflow with drift detection capabilities")
        return enhanced
    
    def set_reference_data(self, reference_data: pd.DataFrame) -> bool:
        """Set reference data for drift detection"""
        try:
            # Limit reference data size if too large
            if len(reference_data) > self.drift_config.max_reference_samples:
                reference_data = reference_data.sample(
                    n=self.drift_config.max_reference_samples, 
                    random_state=42
                )
            
            self.reference_data = reference_data.copy()
            self.reference_timestamp = datetime.now()
            
            # Auto-detect feature types if not specified
            if not self.drift_config.numerical_features and not self.drift_config.categorical_features:
                self._auto_detect_feature_types()
            
            self._logger.info(f"Reference data set with {len(reference_data)} samples and {len(reference_data.columns)} features")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to set reference data: {e}")
            return False
    
    def detect_drift(self, current_data: pd.DataFrame,
                    model: Optional[Any] = None,
                    target_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect drift in current data compared to reference"""
        try:
            if self.reference_data is None:
                raise ValueError("Reference data not set. Call set_reference_data() first.")
            
            self._logger.info("Starting drift detection analysis")
            
            # Ensure we have enough samples
            if len(current_data) < self.drift_config.min_samples_for_test:
                self._logger.warning(f"Insufficient samples for drift detection: {len(current_data)} < {self.drift_config.min_samples_for_test}")
                return {'insufficient_samples': True}
            
            drift_results = {
                'detection_timestamp': datetime.now().isoformat(),
                'samples_analyzed': len(current_data),
                'features_analyzed': [],
                'drift_detected': False,
                'feature_drift_results': {},
                'concept_drift_results': {},
                'alerts': []
            }
            
            # Align features between reference and current data
            common_features = set(self.reference_data.columns) & set(current_data.columns)
            features_to_analyze = list(common_features - set(self.drift_config.ignore_features))
            
            drift_results['features_analyzed'] = features_to_analyze
            
            # Analyze each feature
            for feature in features_to_analyze:
                feature_result = self._analyze_feature_drift(
                    self.reference_data[feature].values,
                    current_data[feature].values,
                    feature
                )
                
                drift_results['feature_drift_results'][feature] = feature_result
                
                # Check if drift detected for this feature
                if feature_result.get('drift_detected', False):
                    drift_results['drift_detected'] = True
                    
                    # Create alert
                    alert = self._create_drift_alert(feature, feature_result)
                    drift_results['alerts'].append(alert)
                    self.active_alerts.append(alert)
            
            # Concept drift detection if model and target provided
            if model is not None and target_data is not None:
                concept_result = self.concept_detector.detect_concept_drift(
                    model, current_data.values, target_data
                )
                drift_results['concept_drift_results'] = concept_result
                
                if concept_result.get('drift_detected', False):
                    drift_results['drift_detected'] = True
                    
                    # Create concept drift alert
                    concept_alert = self._create_concept_drift_alert(concept_result)
                    drift_results['alerts'].append(concept_alert)
                    self.active_alerts.append(concept_alert)
            
            # Save drift report if configured
            if self.drift_config.store_drift_reports:
                self._save_drift_report(drift_results)
            
            self._logger.info(f"Drift detection completed. Drift detected: {drift_results['drift_detected']}")
            
            return drift_results
            
        except Exception as e:
            self._logger.error(f"Drift detection failed: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_drift(self, reference_values: np.ndarray,
                             current_values: np.ndarray,
                             feature_name: str) -> Dict[str, Any]:
        """Analyze drift for a single feature"""
        try:
            # Determine feature type
            is_categorical = (feature_name in self.drift_config.categorical_features or
                            self._is_categorical_feature(reference_values))
            
            if is_categorical:
                results = self.statistical_detector.detect_categorical_drift(
                    reference_values, current_values, feature_name
                )
            else:
                results = self.statistical_detector.detect_numerical_drift(
                    reference_values, current_values, feature_name
                )
            
            # Determine if drift is detected based on configured thresholds
            drift_detected = False
            drift_reasons = []
            
            # Check statistical test results
            if 'ks_p_value' in results and results['ks_p_value'] < self.drift_config.drift_threshold:
                drift_detected = True
                drift_reasons.append(f"KS test p-value: {results['ks_p_value']:.4f}")
            
            if 'chi2_p_value' in results and results['chi2_p_value'] < self.drift_config.drift_threshold:
                drift_detected = True
                drift_reasons.append(f"Chi-square test p-value: {results['chi2_p_value']:.4f}")
            
            if 'psi_score' in results and results['psi_score'] > self.drift_config.psi_threshold:
                drift_detected = True
                drift_reasons.append(f"PSI score: {results['psi_score']:.4f}")
            
            if 'wasserstein_distance' in results and results['wasserstein_distance'] > self.drift_config.wasserstein_threshold:
                drift_detected = True
                drift_reasons.append(f"Wasserstein distance: {results['wasserstein_distance']:.4f}")
            
            # Add detection summary
            results['drift_detected'] = drift_detected
            results['drift_reasons'] = drift_reasons
            results['feature_type'] = 'categorical' if is_categorical else 'numerical'
            results['severity'] = self._assess_drift_severity(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Feature drift analysis failed for {feature_name}: {e}")
            return {
                'drift_detected': False,
                'error': str(e),
                'feature_type': 'unknown'
            }
    
    def _is_categorical_feature(self, values: np.ndarray) -> bool:
        """Heuristically determine if feature is categorical"""
        # Consider categorical if: string type, or numeric with few unique values
        if np.issubdtype(values.dtype, np.str_) or np.issubdtype(values.dtype, np.object_):
            return True
        
        # For numeric data, consider categorical if unique values <= 20 and < 50% of total
        unique_values = len(np.unique(values))
        return unique_values <= 20 and unique_values < len(values) * 0.5
    
    def _assess_drift_severity(self, results: Dict[str, Any]) -> DriftSeverity:
        """Assess severity of detected drift"""
        if not results.get('drift_detected', False):
            return DriftSeverity.LOW
        
        # Check PSI score
        psi_score = results.get('psi_score', 0)
        if psi_score > 0.5:
            return DriftSeverity.CRITICAL
        elif psi_score > 0.3:
            return DriftSeverity.HIGH
        elif psi_score > 0.2:
            return DriftSeverity.MEDIUM
        
        # Check p-values
        min_p_value = min([
            results.get('ks_p_value', 1.0),
            results.get('chi2_p_value', 1.0),
            results.get('mean_shift_p_value', 1.0)
        ])
        
        if min_p_value < 0.001:
            return DriftSeverity.HIGH
        elif min_p_value < 0.01:
            return DriftSeverity.MEDIUM
        
        return DriftSeverity.LOW
    
    def _create_drift_alert(self, feature_name: str, feature_result: Dict[str, Any]) -> DriftAlert:
        """Create drift alert from feature analysis"""
        severity = feature_result.get('severity', DriftSeverity.LOW)
        drift_score = feature_result.get('psi_score', 0.0)
        p_value = min([
            feature_result.get('ks_p_value', 1.0),
            feature_result.get('chi2_p_value', 1.0),
            feature_result.get('mean_shift_p_value', 1.0)
        ])
        
        description = f"Data drift detected in feature '{feature_name}'. "
        description += f"Severity: {severity.value}. "
        description += f"Reasons: {', '.join(feature_result.get('drift_reasons', []))}"
        
        recommendation = self._generate_drift_recommendation(severity, feature_result)
        
        alert_id = f"drift_{feature_name}_{int(datetime.now().timestamp())}"
        
        return DriftAlert(
            alert_id=alert_id,
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            feature_name=feature_name,
            drift_score=drift_score,
            p_value=p_value,
            threshold=self.drift_config.psi_threshold,
            detected_at=datetime.now(),
            description=description,
            recommendation=recommendation
        )
    
    def _create_concept_drift_alert(self, concept_result: Dict[str, Any]) -> DriftAlert:
        """Create concept drift alert"""
        drift_score = concept_result.get('drift_score', 0.0)
        current_perf = concept_result.get('current_performance', 0.0)
        baseline_perf = concept_result.get('baseline_performance', 0.0)
        
        # Assess severity based on performance drop
        if drift_score > 0.2:
            severity = DriftSeverity.CRITICAL
        elif drift_score > 0.1:
            severity = DriftSeverity.HIGH
        elif drift_score > 0.05:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        description = f"Concept drift detected. Model performance dropped by {drift_score:.2%}. "
        description += f"Current performance: {current_perf:.4f}, Baseline: {baseline_perf:.4f}"
        
        recommendation = "Consider model retraining or updating with recent data. "
        recommendation += "Investigate changes in data patterns or business context."
        
        alert_id = f"concept_drift_{int(datetime.now().timestamp())}"
        
        return DriftAlert(
            alert_id=alert_id,
            drift_type=DriftType.CONCEPT_DRIFT,
            severity=severity,
            feature_name="model_performance",
            drift_score=drift_score,
            p_value=0.0,  # Not applicable for concept drift
            threshold=self.drift_config.concept_drift_threshold,
            detected_at=datetime.now(),
            description=description,
            recommendation=recommendation
        )
    
    def _generate_drift_recommendation(self, severity: DriftSeverity, 
                                     feature_result: Dict[str, Any]) -> str:
        """Generate recommendation based on drift severity"""
        base_recommendations = {
            DriftSeverity.LOW: "Monitor closely. Consider investigating data collection process.",
            DriftSeverity.MEDIUM: "Investigate data source changes. Consider model retraining.",
            DriftSeverity.HIGH: "Immediate investigation required. Model retraining recommended.",
            DriftSeverity.CRITICAL: "Critical drift detected. Stop model serving and retrain immediately."
        }
        
        recommendation = base_recommendations.get(severity, "Monitor the situation.")
        
        # Add specific recommendations based on drift type
        if feature_result.get('feature_type') == 'categorical':
            recommendation += " Check for new categories or changes in category distribution."
        else:
            recommendation += " Analyze changes in feature statistics and distributions."
        
        return recommendation
    
    def _auto_detect_feature_types(self):
        """Automatically detect numerical and categorical features"""
        if self.reference_data is None:
            return
        
        numerical_features = []
        categorical_features = []
        
        for column in self.reference_data.columns:
            values = self.reference_data[column].values
            if self._is_categorical_feature(values):
                categorical_features.append(column)
            else:
                numerical_features.append(column)
        
        self.drift_config.numerical_features = numerical_features
        self.drift_config.categorical_features = categorical_features
        
        self._logger.info(f"Auto-detected {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
    
    def _save_drift_report(self, drift_results: Dict[str, Any]):
        """Save drift detection report to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(self.drift_config.drift_reports_path) / f"drift_report_{timestamp}.json"
            
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj
            
            serializable_results = convert_types(drift_results)
            
            with open(report_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self._logger.info(f"Drift report saved to: {report_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to save drift report: {e}")
    
    def get_active_alerts(self, severity_filter: Optional[DriftSeverity] = None) -> List[DriftAlert]:
        """Get active drift alerts"""
        if severity_filter is None:
            return self.active_alerts.copy()
        
        return [alert for alert in self.active_alerts if alert.severity == severity_filter]
    
    def clear_alerts(self, older_than_hours: Optional[int] = None):
        """Clear old alerts"""
        if older_than_hours is None:
            older_than_hours = self.drift_config.alert_cooldown_hours
        
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        self.active_alerts = [
            alert for alert in self.active_alerts 
            if alert.detected_at > cutoff_time
        ]
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection status"""
        return {
            'reference_data_set': self.reference_data is not None,
            'reference_timestamp': self.reference_timestamp.isoformat() if self.reference_timestamp else None,
            'reference_samples': len(self.reference_data) if self.reference_data is not None else 0,
            'active_alerts': len(self.active_alerts),
            'alerts_by_severity': {
                severity.value: len([a for a in self.active_alerts if a.severity == severity])
                for severity in DriftSeverity
            },
            'features_monitored': {
                'numerical': len(self.drift_config.numerical_features),
                'categorical': len(self.drift_config.categorical_features)
            },
            'configuration': {
                'drift_threshold': self.drift_config.drift_threshold,
                'psi_threshold': self.drift_config.psi_threshold,
                'monitoring_window_hours': self.drift_config.monitoring_window_hours
            }
        }