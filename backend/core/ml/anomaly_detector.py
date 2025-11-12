#!/usr/bin/env python3
"""
Isolation Forest Anomaly Detection System for DWCP v3
Detects unusual patterns in production metrics with automatic alerting
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnomalyType:
    """Anomaly type enumeration"""
    LATENCY_SPIKE = "latency_spike"
    THROUGHPUT_DROP = "throughput_drop"
    ERROR_BURST = "error_burst"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    COMPRESSION_FAILURE = "compression_failure"
    PREDICTION_ERROR = "prediction_error"
    CONSENSUS_DELAY = "consensus_delay"
    NETWORK_ANOMALY = "network_anomaly"
    UNKNOWN = "unknown"


class AnomalyDetector:
    """Isolation Forest-based anomaly detection system"""

    def __init__(
        self,
        config: Optional[Dict] = None,
        model_dir: str = "/tmp/ml_models"
    ):
        self.config = config or self._default_config()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Primary isolation forest model
        self.isolation_forest = IsolationForest(
            contamination=self.config['contamination'],
            max_samples=self.config['max_samples'],
            n_estimators=self.config['n_estimators'],
            random_state=42
        )

        # Secondary models for different metric types
        self.metric_models = {}

        # Scalers for normalization
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance

        # Anomaly history and statistics
        self.anomaly_history = []
        self.baseline_statistics = {}
        self.alert_thresholds = self._initialize_thresholds()

        # Clustering for anomaly grouping
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)

        logger.info("Initialized AnomalyDetector")

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'contamination': 0.05,  # 5% expected anomaly rate
            'max_samples': 256,
            'n_estimators': 100,
            'threshold_multiplier': 3.0,  # For statistical anomaly detection
            'min_samples_for_training': 1000,
            'anomaly_score_threshold': -0.5,
            'feature_columns': [
                'latency_mean', 'latency_std', 'latency_p95', 'latency_p99',
                'throughput_mean', 'throughput_std',
                'error_rate', 'cpu_usage', 'memory_usage',
                'compression_ratio', 'prediction_accuracy', 'consensus_time'
            ],
            'alert_severity': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            }
        }

    def _initialize_thresholds(self) -> Dict:
        """Initialize alert thresholds"""
        return {
            AnomalyType.LATENCY_SPIKE: {
                'critical': 1000.0,  # ms
                'high': 500.0,
                'medium': 200.0,
                'low': 100.0
            },
            AnomalyType.THROUGHPUT_DROP: {
                'critical': 0.5,  # 50% drop
                'high': 0.3,
                'medium': 0.2,
                'low': 0.1
            },
            AnomalyType.ERROR_BURST: {
                'critical': 100,  # errors per minute
                'high': 50,
                'medium': 25,
                'low': 10
            },
            AnomalyType.RESOURCE_EXHAUSTION: {
                'critical': 0.95,  # 95% usage
                'high': 0.85,
                'medium': 0.75,
                'low': 0.65
            }
        }

    def train(self, data: pd.DataFrame, save_model: bool = True) -> Dict:
        """Train anomaly detection models"""
        logger.info(f"Training anomaly detector on {len(data)} samples")

        if len(data) < self.config['min_samples_for_training']:
            logger.warning(f"Insufficient training data: {len(data)} < {self.config['min_samples_for_training']}")
            return {'status': 'insufficient_data', 'samples': len(data)}

        # Extract features
        feature_cols = [col for col in self.config['feature_columns'] if col in data.columns]
        X = data[feature_cols].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Compute baseline statistics
        self._compute_baseline_statistics(data, feature_cols)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Apply PCA for dimensionality reduction
        X_pca = self.pca.fit_transform(X_scaled)

        # Train primary isolation forest
        logger.info("Training Isolation Forest...")
        self.isolation_forest.fit(X_pca)

        # Train metric-specific models
        self._train_metric_models(data, feature_cols)

        # Evaluate on training data
        anomaly_scores = self.isolation_forest.score_samples(X_pca)
        predictions = self.isolation_forest.predict(X_pca)

        # Calculate metrics
        n_anomalies = np.sum(predictions == -1)
        anomaly_rate = n_anomalies / len(predictions)

        training_metrics = {
            'status': 'success',
            'samples': len(data),
            'features': len(feature_cols),
            'pca_components': self.pca.n_components_,
            'anomalies_detected': int(n_anomalies),
            'anomaly_rate': float(anomaly_rate),
            'mean_anomaly_score': float(np.mean(anomaly_scores)),
            'std_anomaly_score': float(np.std(anomaly_scores))
        }

        logger.info(f"Training complete: {training_metrics}")

        if save_model:
            self.save_model('anomaly_detector.pkl')

        return training_metrics

    def _compute_baseline_statistics(self, data: pd.DataFrame, feature_cols: List[str]) -> None:
        """Compute baseline statistics for each feature"""
        logger.info("Computing baseline statistics...")

        for col in feature_cols:
            if col in data.columns:
                values = data[col].values
                self.baseline_statistics[col] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'p50': float(np.percentile(values, 50)),
                    'p95': float(np.percentile(values, 95)),
                    'p99': float(np.percentile(values, 99)),
                    'threshold_upper': float(np.mean(values) + self.config['threshold_multiplier'] * np.std(values)),
                    'threshold_lower': float(np.mean(values) - self.config['threshold_multiplier'] * np.std(values))
                }

        logger.info(f"Computed baseline statistics for {len(self.baseline_statistics)} features")

    def _train_metric_models(self, data: pd.DataFrame, feature_cols: List[str]) -> None:
        """Train specialized models for different metric types"""
        logger.info("Training metric-specific models...")

        # Group features by metric type
        metric_groups = {
            'latency': [col for col in feature_cols if 'latency' in col],
            'throughput': [col for col in feature_cols if 'throughput' in col],
            'errors': [col for col in feature_cols if 'error' in col],
            'resources': [col for col in feature_cols if any(x in col for x in ['cpu', 'memory', 'disk', 'network'])],
            'dwcp': [col for col in feature_cols if any(x in col for x in ['compression', 'prediction', 'consensus'])]
        }

        for metric_type, cols in metric_groups.items():
            if not cols:
                continue

            # Extract relevant features
            X = data[cols].values
            X = np.nan_to_num(X, nan=0.0)

            # Train specialized isolation forest
            model = IsolationForest(
                contamination=self.config['contamination'],
                max_samples=min(256, len(X)),
                n_estimators=50,
                random_state=42
            )
            model.fit(X)

            self.metric_models[metric_type] = {
                'model': model,
                'features': cols,
                'scaler': StandardScaler().fit(X)
            }

        logger.info(f"Trained {len(self.metric_models)} metric-specific models")

    def detect(self, data: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in new data"""
        logger.info(f"Detecting anomalies in {len(data)} samples")

        # Extract features
        feature_cols = [col for col in self.config['feature_columns'] if col in data.columns]
        X = data[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)

        # Scale and transform
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)

        # Primary detection with isolation forest
        anomaly_scores = self.isolation_forest.score_samples(X_pca)
        predictions = self.isolation_forest.predict(X_pca)

        anomalies = []

        for idx, (score, pred) in enumerate(zip(anomaly_scores, predictions)):
            if pred == -1 or score < self.config['anomaly_score_threshold']:
                # Detected an anomaly
                anomaly_info = self._analyze_anomaly(data.iloc[idx], score, feature_cols)

                anomalies.append(anomaly_info)

        # Apply metric-specific detection
        metric_anomalies = self._detect_metric_anomalies(data, feature_cols)
        anomalies.extend(metric_anomalies)

        # Apply statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(data, feature_cols)
        anomalies.extend(statistical_anomalies)

        # Deduplicate and rank anomalies
        anomalies = self._deduplicate_anomalies(anomalies)
        anomalies = sorted(anomalies, key=lambda x: x['severity_score'], reverse=True)

        # Store in history
        self.anomaly_history.extend(anomalies)

        logger.info(f"Detected {len(anomalies)} anomalies")

        return anomalies

    def _analyze_anomaly(self, sample: pd.Series, anomaly_score: float, feature_cols: List[str]) -> Dict:
        """Analyze detected anomaly to determine type and severity"""

        # Calculate feature contributions
        feature_deviations = {}
        for col in feature_cols:
            if col in sample and col in self.baseline_statistics:
                value = sample[col]
                baseline = self.baseline_statistics[col]
                deviation = abs(value - baseline['mean']) / (baseline['std'] + 1e-10)
                feature_deviations[col] = float(deviation)

        # Determine anomaly type
        anomaly_type = self._classify_anomaly_type(sample, feature_deviations)

        # Calculate severity
        severity, severity_score = self._calculate_severity(anomaly_type, sample, feature_deviations)

        anomaly_info = {
            'timestamp': sample.get('timestamp', datetime.now().isoformat()),
            'anomaly_type': anomaly_type,
            'severity': severity,
            'severity_score': severity_score,
            'anomaly_score': float(anomaly_score),
            'feature_deviations': feature_deviations,
            'top_contributing_features': sorted(
                feature_deviations.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'metrics': {col: float(sample[col]) for col in feature_cols if col in sample},
            'baseline_comparison': self._compare_to_baseline(sample, feature_cols)
        }

        return anomaly_info

    def _classify_anomaly_type(self, sample: pd.Series, deviations: Dict) -> str:
        """Classify the type of anomaly"""

        # Check for specific anomaly patterns
        if 'latency_mean' in deviations and deviations['latency_mean'] > 3:
            return AnomalyType.LATENCY_SPIKE

        if 'throughput_mean' in deviations and deviations['throughput_mean'] > 3:
            # Check if throughput is abnormally low
            if 'throughput_mean' in sample:
                baseline_throughput = self.baseline_statistics.get('throughput_mean', {}).get('mean', 0)
                if sample['throughput_mean'] < baseline_throughput * 0.7:
                    return AnomalyType.THROUGHPUT_DROP

        if 'error_rate' in deviations and deviations['error_rate'] > 2:
            return AnomalyType.ERROR_BURST

        if any('cpu' in k or 'memory' in k for k in deviations.keys()):
            for k, v in deviations.items():
                if ('cpu' in k or 'memory' in k) and v > 2.5:
                    if k in sample and sample[k] > 0.85:
                        return AnomalyType.RESOURCE_EXHAUSTION

        if 'compression_ratio' in deviations and deviations['compression_ratio'] > 2:
            return AnomalyType.COMPRESSION_FAILURE

        if 'prediction_accuracy' in deviations and deviations['prediction_accuracy'] > 2:
            return AnomalyType.PREDICTION_ERROR

        if 'consensus_time' in deviations and deviations['consensus_time'] > 2:
            return AnomalyType.CONSENSUS_DELAY

        return AnomalyType.UNKNOWN

    def _calculate_severity(
        self,
        anomaly_type: str,
        sample: pd.Series,
        deviations: Dict
    ) -> Tuple[str, float]:
        """Calculate anomaly severity"""

        # Get maximum deviation
        max_deviation = max(deviations.values()) if deviations else 0

        # Calculate severity score (0-1)
        severity_score = min(max_deviation / 10.0, 1.0)

        # Determine severity level
        if severity_score >= self.config['alert_severity']['critical']:
            severity = 'critical'
        elif severity_score >= self.config['alert_severity']['high']:
            severity = 'high'
        elif severity_score >= self.config['alert_severity']['medium']:
            severity = 'medium'
        else:
            severity = 'low'

        # Adjust based on anomaly type-specific thresholds
        if anomaly_type in self.alert_thresholds:
            thresholds = self.alert_thresholds[anomaly_type]

            # Check metric value against thresholds
            if anomaly_type == AnomalyType.LATENCY_SPIKE and 'latency_mean' in sample:
                latency = sample['latency_mean']
                if latency >= thresholds['critical']:
                    severity = 'critical'
                    severity_score = 1.0

            elif anomaly_type == AnomalyType.ERROR_BURST and 'error_rate' in sample:
                error_rate = sample['error_rate']
                if error_rate >= thresholds['critical']:
                    severity = 'critical'
                    severity_score = 1.0

        return severity, severity_score

    def _compare_to_baseline(self, sample: pd.Series, feature_cols: List[str]) -> Dict:
        """Compare sample to baseline statistics"""
        comparison = {}

        for col in feature_cols:
            if col in sample and col in self.baseline_statistics:
                value = sample[col]
                baseline = self.baseline_statistics[col]

                comparison[col] = {
                    'current': float(value),
                    'baseline_mean': baseline['mean'],
                    'baseline_p95': baseline['p95'],
                    'deviation_pct': float((value - baseline['mean']) / (baseline['mean'] + 1e-10) * 100),
                    'z_score': float((value - baseline['mean']) / (baseline['std'] + 1e-10))
                }

        return comparison

    def _detect_metric_anomalies(self, data: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
        """Detect anomalies using metric-specific models"""
        anomalies = []

        for metric_type, model_info in self.metric_models.items():
            model = model_info['model']
            metric_features = model_info['features']
            scaler = model_info['scaler']

            if not all(f in data.columns for f in metric_features):
                continue

            X = data[metric_features].values
            X = np.nan_to_num(X, nan=0.0)
            X_scaled = scaler.transform(X)

            predictions = model.predict(X_scaled)
            scores = model.score_samples(X_scaled)

            for idx, (pred, score) in enumerate(zip(predictions, scores)):
                if pred == -1:
                    anomaly = {
                        'timestamp': data.iloc[idx].get('timestamp', datetime.now().isoformat()),
                        'anomaly_type': f"{metric_type}_anomaly",
                        'severity': 'medium',
                        'severity_score': 0.6,
                        'anomaly_score': float(score),
                        'detection_method': f'metric_specific_{metric_type}',
                        'metrics': {col: float(data.iloc[idx][col]) for col in metric_features}
                    }
                    anomalies.append(anomaly)

        return anomalies

    def _detect_statistical_anomalies(self, data: pd.DataFrame, feature_cols: List[str]) -> List[Dict]:
        """Detect anomalies using statistical methods"""
        anomalies = []

        for col in feature_cols:
            if col not in data.columns or col not in self.baseline_statistics:
                continue

            baseline = self.baseline_statistics[col]
            values = data[col].values

            # Detect values outside threshold
            upper_violations = values > baseline['threshold_upper']
            lower_violations = values < baseline['threshold_lower']

            for idx in np.where(upper_violations | lower_violations)[0]:
                value = values[idx]
                z_score = (value - baseline['mean']) / (baseline['std'] + 1e-10)

                anomaly = {
                    'timestamp': data.iloc[idx].get('timestamp', datetime.now().isoformat()),
                    'anomaly_type': f'statistical_{col}',
                    'severity': 'low' if abs(z_score) < 4 else 'medium',
                    'severity_score': min(abs(z_score) / 10.0, 1.0),
                    'detection_method': 'statistical',
                    'feature': col,
                    'value': float(value),
                    'baseline_mean': baseline['mean'],
                    'z_score': float(z_score)
                }
                anomalies.append(anomaly)

        return anomalies

    def _deduplicate_anomalies(self, anomalies: List[Dict]) -> List[Dict]:
        """Remove duplicate anomaly detections"""
        if not anomalies:
            return []

        # Group by timestamp
        timestamp_groups = defaultdict(list)
        for anomaly in anomalies:
            timestamp = anomaly['timestamp']
            timestamp_groups[timestamp].append(anomaly)

        deduplicated = []
        for timestamp, group in timestamp_groups.items():
            # Keep the anomaly with highest severity score
            best = max(group, key=lambda x: x['severity_score'])
            deduplicated.append(best)

        return deduplicated

    def generate_alert(self, anomaly: Dict) -> Dict:
        """Generate alert for anomaly"""
        alert = {
            'alert_id': f"alert_{datetime.now().timestamp()}",
            'timestamp': anomaly['timestamp'],
            'anomaly_type': anomaly['anomaly_type'],
            'severity': anomaly['severity'],
            'severity_score': anomaly['severity_score'],
            'title': self._generate_alert_title(anomaly),
            'description': self._generate_alert_description(anomaly),
            'metrics': anomaly.get('metrics', {}),
            'recommended_actions': self._generate_recommendations(anomaly),
            'alert_channels': self._determine_alert_channels(anomaly['severity'])
        }

        logger.info(f"Generated alert: {alert['alert_id']} - {alert['title']}")

        return alert

    def _generate_alert_title(self, anomaly: Dict) -> str:
        """Generate alert title"""
        anomaly_type = anomaly['anomaly_type']
        severity = anomaly['severity'].upper()

        titles = {
            AnomalyType.LATENCY_SPIKE: f"[{severity}] Latency Spike Detected",
            AnomalyType.THROUGHPUT_DROP: f"[{severity}] Throughput Drop Detected",
            AnomalyType.ERROR_BURST: f"[{severity}] Error Burst Detected",
            AnomalyType.RESOURCE_EXHAUSTION: f"[{severity}] Resource Exhaustion Detected",
            AnomalyType.COMPRESSION_FAILURE: f"[{severity}] Compression Failure Detected",
            AnomalyType.PREDICTION_ERROR: f"[{severity}] Prediction Error Detected",
            AnomalyType.CONSENSUS_DELAY: f"[{severity}] Consensus Delay Detected"
        }

        return titles.get(anomaly_type, f"[{severity}] Anomaly Detected")

    def _generate_alert_description(self, anomaly: Dict) -> str:
        """Generate alert description"""
        description = f"Anomaly detected at {anomaly['timestamp']}\n"
        description += f"Type: {anomaly['anomaly_type']}\n"
        description += f"Severity Score: {anomaly['severity_score']:.2f}\n\n"

        if 'feature_deviations' in anomaly:
            description += "Top Contributing Factors:\n"
            for feature, deviation in anomaly.get('top_contributing_features', [])[:3]:
                description += f"  - {feature}: {deviation:.2f}σ deviation\n"

        return description

    def _generate_recommendations(self, anomaly: Dict) -> List[str]:
        """Generate recommended actions"""
        recommendations = []

        anomaly_type = anomaly['anomaly_type']

        if anomaly_type == AnomalyType.LATENCY_SPIKE:
            recommendations.extend([
                "Check for network congestion",
                "Review recent configuration changes",
                "Verify database query performance",
                "Scale up compute resources if needed"
            ])
        elif anomaly_type == AnomalyType.THROUGHPUT_DROP:
            recommendations.extend([
                "Check upstream service health",
                "Verify load balancer configuration",
                "Review rate limiting settings",
                "Scale out worker nodes"
            ])
        elif anomaly_type == AnomalyType.ERROR_BURST:
            recommendations.extend([
                "Review error logs for patterns",
                "Check service dependencies",
                "Verify data validation logic",
                "Rollback recent deployments if necessary"
            ])
        elif anomaly_type == AnomalyType.RESOURCE_EXHAUSTION:
            recommendations.extend([
                "Immediate: Scale up resources",
                "Review resource allocation policies",
                "Identify memory leaks or inefficiencies",
                "Implement resource limits and quotas"
            ])

        return recommendations

    def _determine_alert_channels(self, severity: str) -> List[str]:
        """Determine which alert channels to use"""
        channels = {
            'critical': ['pagerduty', 'slack', 'email', 'sms'],
            'high': ['slack', 'email'],
            'medium': ['slack', 'email'],
            'low': ['email']
        }

        return channels.get(severity, ['email'])

    def get_anomaly_report(self, time_range: timedelta = timedelta(hours=24)) -> Dict:
        """Generate anomaly analysis report"""
        cutoff_time = datetime.now() - time_range

        recent_anomalies = [
            a for a in self.anomaly_history
            if datetime.fromisoformat(a['timestamp']) >= cutoff_time
        ]

        # Calculate statistics
        total_anomalies = len(recent_anomalies)
        by_type = defaultdict(int)
        by_severity = defaultdict(int)

        for anomaly in recent_anomalies:
            by_type[anomaly['anomaly_type']] += 1
            by_severity[anomaly['severity']] += 1

        report = {
            'time_range': str(time_range),
            'cutoff_time': cutoff_time.isoformat(),
            'total_anomalies': total_anomalies,
            'by_type': dict(by_type),
            'by_severity': dict(by_severity),
            'critical_anomalies': [a for a in recent_anomalies if a['severity'] == 'critical'],
            'false_positive_rate': self._estimate_false_positive_rate(recent_anomalies)
        }

        logger.info(f"Generated anomaly report: {total_anomalies} anomalies in {time_range}")

        return report

    def _estimate_false_positive_rate(self, anomalies: List[Dict]) -> float:
        """Estimate false positive rate (simplified)"""
        # In production, this would track operator feedback
        # For now, return a placeholder value
        return 0.05  # 5% estimated false positive rate

    def save_model(self, filename: str) -> None:
        """Save anomaly detector models"""
        model_path = self.model_dir / filename

        model_data = {
            'isolation_forest': self.isolation_forest,
            'metric_models': self.metric_models,
            'scaler': self.scaler,
            'pca': self.pca,
            'baseline_statistics': self.baseline_statistics,
            'config': self.config
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {model_path}")

    def load_model(self, filename: str) -> None:
        """Load anomaly detector models"""
        model_path = self.model_dir / filename

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.isolation_forest = model_data['isolation_forest']
        self.metric_models = model_data['metric_models']
        self.scaler = model_data['scaler']
        self.pca = model_data['pca']
        self.baseline_statistics = model_data['baseline_statistics']
        self.config = model_data['config']

        logger.info(f"Model loaded from {model_path}")


def main():
    """Main anomaly detection pipeline"""
    logger.info("Starting Anomaly Detection System")

    # Initialize detector
    detector = AnomalyDetector()

    logger.info("Anomaly detector ready")
    logger.info("Use detector.train() to train on historical data")
    logger.info("Use detector.detect() to detect anomalies in new data")
    logger.info("Use detector.generate_alert() to create alerts")

    return detector


if __name__ == "__main__":
    detector = main()
