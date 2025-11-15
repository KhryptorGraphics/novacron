#!/usr/bin/env python3
"""
Train Isolation Forest model for Node Reliability Anomaly Detection in DWCP.

Target: ≥98% recall on labeled incidents with FP rate <5%

Data Schema (node-level aggregates):
- timestamp, node_id, region, az
- error_rate, timeout_rate, latency_p50, latency_p99
- sla_violations, connection_failures, packet_loss_rate
- cpu_usage, memory_usage, disk_io
- dwcp_mode, network_tier
- label (0=normal, 1=incident)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split, StratifiedKFold
import joblib
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Advanced feature engineering for node reliability metrics."""

    def __init__(self, rolling_windows=[5, 15, 30]):
        self.rolling_windows = rolling_windows
        self.feature_names = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features from raw node metrics.

        Features include:
        - Rolling window statistics (mean, std, min, max)
        - Rate of change features
        - Interaction features
        - Threshold violation indicators
        """
        logger.info("Engineering features from raw metrics...")
        df_engineered = df.copy()

        # Sort by timestamp for rolling windows
        if 'timestamp' in df.columns:
            df_engineered['timestamp'] = pd.to_datetime(df_engineered['timestamp'])
            df_engineered = df_engineered.sort_values('timestamp')

        # Base features
        base_features = [
            'error_rate', 'timeout_rate', 'latency_p50', 'latency_p99',
            'sla_violations', 'connection_failures', 'packet_loss_rate',
            'cpu_usage', 'memory_usage', 'disk_io'
        ]

        # Rolling window statistics
        for window in self.rolling_windows:
            for feature in base_features:
                if feature in df.columns:
                    # Rolling mean
                    col_name = f'{feature}_rolling_mean_{window}'
                    df_engineered[col_name] = df[feature].rolling(window=window, min_periods=1).mean()

                    # Rolling std
                    col_name = f'{feature}_rolling_std_{window}'
                    df_engineered[col_name] = df[feature].rolling(window=window, min_periods=1).std().fillna(0)

                    # Rolling min/max
                    df_engineered[f'{feature}_rolling_min_{window}'] = df[feature].rolling(window=window, min_periods=1).min()
                    df_engineered[f'{feature}_rolling_max_{window}'] = df[feature].rolling(window=window, min_periods=1).max()

        # Rate of change features
        for feature in base_features:
            if feature in df.columns:
                df_engineered[f'{feature}_rate_of_change'] = df[feature].diff().fillna(0)
                df_engineered[f'{feature}_acceleration'] = df[feature].diff().diff().fillna(0)

        # Interaction features
        if 'error_rate' in df.columns and 'timeout_rate' in df.columns:
            df_engineered['error_timeout_product'] = df['error_rate'] * df['timeout_rate']

        if 'latency_p99' in df.columns and 'latency_p50' in df.columns:
            df_engineered['latency_spread'] = df['latency_p99'] - df['latency_p50']
            df_engineered['latency_ratio'] = df['latency_p99'] / (df['latency_p50'] + 1e-6)

        if 'cpu_usage' in df.columns and 'memory_usage' in df.columns:
            df_engineered['resource_pressure'] = (df['cpu_usage'] + df['memory_usage']) / 2

        # Threshold violation indicators
        if 'error_rate' in df.columns:
            df_engineered['high_error_rate'] = (df['error_rate'] > 0.01).astype(int)

        if 'latency_p99' in df.columns:
            df_engineered['high_latency'] = (df['latency_p99'] > 100).astype(int)

        if 'packet_loss_rate' in df.columns:
            df_engineered['high_packet_loss'] = (df['packet_loss_rate'] > 0.05).astype(int)

        # Categorical encoding
        if 'dwcp_mode' in df.columns:
            df_engineered = pd.get_dummies(df_engineered, columns=['dwcp_mode'], prefix='mode')

        if 'network_tier' in df.columns:
            df_engineered = pd.get_dummies(df_engineered, columns=['network_tier'], prefix='tier')

        # Store feature names
        exclude_cols = ['timestamp', 'node_id', 'region', 'az', 'label']
        self.feature_names = [col for col in df_engineered.columns if col not in exclude_cols]

        logger.info(f"Engineered {len(self.feature_names)} features from {len(base_features)} base features")

        return df_engineered


class IsolationForestTuner:
    """Tune Isolation Forest for high recall with acceptable FP rate."""

    def __init__(self, target_recall=0.98, max_fp_rate=0.05):
        self.target_recall = target_recall
        self.max_fp_rate = max_fp_rate
        self.best_model = None
        self.best_scaler = None
        self.best_threshold = None
        self.best_params = None
        self.evaluation_results = {}

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters to achieve target recall.

        Hyperparameters:
        - n_estimators: [100, 200, 300, 500]
        - max_samples: [128, 256, 512, 'auto']
        - max_features: [0.5, 0.75, 1.0]
        - contamination: Based on actual incident rate
        """
        logger.info("Starting hyperparameter tuning...")

        # Calculate contamination from labeled data
        incident_rate = y_train.sum() / len(y_train)
        contamination_values = [
            max(0.001, incident_rate * 0.5),
            incident_rate,
            min(0.5, incident_rate * 1.5),
            min(0.5, incident_rate * 2.0)
        ]

        logger.info(f"Incident rate in training data: {incident_rate:.4f}")
        logger.info(f"Testing contamination values: {contamination_values}")

        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_samples': [128, 256, 512, 'auto'],
            'max_features': [0.5, 0.75, 1.0],
            'contamination': contamination_values
        }

        best_recall = 0
        best_config = None
        results = []

        total_configs = (len(param_grid['n_estimators']) *
                        len(param_grid['max_samples']) *
                        len(param_grid['max_features']) *
                        len(param_grid['contamination']))

        logger.info(f"Evaluating {total_configs} configurations...")

        config_idx = 0
        for n_est in param_grid['n_estimators']:
            for max_samp in param_grid['max_samples']:
                for max_feat in param_grid['max_features']:
                    for contam in param_grid['contamination']:
                        config_idx += 1

                        if config_idx % 10 == 0:
                            logger.info(f"Progress: {config_idx}/{total_configs} configurations tested")

                        # Train model
                        scaler = RobustScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)

                        model = IsolationForest(
                            n_estimators=n_est,
                            max_samples=max_samp,
                            max_features=max_feat,
                            contamination=contam,
                            random_state=42,
                            n_jobs=-1
                        )

                        model.fit(X_train_scaled)

                        # Get anomaly scores
                        scores = model.score_samples(X_val_scaled)

                        # Tune threshold for target recall
                        threshold, metrics = self._tune_threshold(scores, y_val)

                        recall = metrics['recall']
                        fp_rate = metrics['fp_rate']

                        results.append({
                            'n_estimators': n_est,
                            'max_samples': max_samp,
                            'max_features': max_feat,
                            'contamination': contam,
                            'threshold': threshold,
                            'recall': recall,
                            'precision': metrics['precision'],
                            'f1': metrics['f1'],
                            'fp_rate': fp_rate
                        })

                        # Check if this is the best configuration
                        if recall >= self.target_recall and fp_rate <= self.max_fp_rate:
                            if recall > best_recall:
                                best_recall = recall
                                best_config = {
                                    'model': model,
                                    'scaler': scaler,
                                    'threshold': threshold,
                                    'params': {
                                        'n_estimators': n_est,
                                        'max_samples': max_samp,
                                        'max_features': max_feat,
                                        'contamination': contam
                                    },
                                    'metrics': metrics
                                }

        if best_config:
            self.best_model = best_config['model']
            self.best_scaler = best_config['scaler']
            self.best_threshold = best_config['threshold']
            self.best_params = best_config['params']
            logger.info(f"Best configuration found: Recall={best_recall:.4f}")
            logger.info(f"Best params: {self.best_params}")
        else:
            # Fallback: select config with highest recall
            results_df = pd.DataFrame(results)
            best_idx = results_df['recall'].idxmax()
            best_result = results_df.iloc[best_idx]

            logger.warning(f"Target recall/FP rate not achieved. Using best recall configuration.")
            logger.warning(f"Best recall: {best_result['recall']:.4f}, FP rate: {best_result['fp_rate']:.4f}")

            # Retrain with best params
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            model = IsolationForest(
                n_estimators=int(best_result['n_estimators']),
                max_samples=best_result['max_samples'],
                max_features=best_result['max_features'],
                contamination=best_result['contamination'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_scaled)

            self.best_model = model
            self.best_scaler = scaler
            self.best_threshold = best_result['threshold']
            self.best_params = {
                'n_estimators': int(best_result['n_estimators']),
                'max_samples': best_result['max_samples'],
                'max_features': best_result['max_features'],
                'contamination': best_result['contamination']
            }

        return {
            'best_params': self.best_params,
            'best_threshold': self.best_threshold,
            'all_results': results
        }

    def _tune_threshold(
        self,
        scores: np.ndarray,
        y_true: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Tune threshold to achieve target recall.

        In Isolation Forest:
        - Lower (more negative) scores = more anomalous
        - Higher (less negative) scores = more normal
        - We want: scores <= threshold → anomaly (label=1)
        """
        if y_true.sum() == 0:
            return scores.min(), {'recall': 0, 'precision': 0, 'f1': 0, 'fp_rate': 1.0}

        best_threshold = scores.max()  # Start with most conservative
        best_metrics = {'recall': 0, 'precision': 0, 'f1': 0, 'fp_rate': 1.0}

        # Try different percentile thresholds (from low to high)
        # Lower percentiles = lower thresholds = more predictions as anomalies
        for percentile in np.linspace(1, 99, 200):
            threshold = np.percentile(scores, percentile)
            predictions = (scores <= threshold).astype(int)  # <= threshold = anomaly

            tp = np.sum((predictions == 1) & (y_true == 1))
            fp = np.sum((predictions == 1) & (y_true == 0))
            tn = np.sum((predictions == 0) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # Prioritize recall >= target, then minimize FP rate
            if recall >= self.target_recall:
                if best_metrics['recall'] < self.target_recall or \
                   (recall >= best_metrics['recall'] and fp_rate < best_metrics['fp_rate']):
                    best_threshold = threshold
                    best_metrics = {
                        'recall': recall,
                        'precision': precision,
                        'f1': f1,
                        'fp_rate': fp_rate
                    }
            # If we haven't achieved target recall yet, take highest recall
            elif recall > best_metrics['recall']:
                best_threshold = threshold
                best_metrics = {
                    'recall': recall,
                    'precision': precision,
                    'f1': f1,
                    'fp_rate': fp_rate
                }

        return best_threshold, best_metrics

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        """
        logger.info("Evaluating model on test set...")

        X_test_scaled = self.best_scaler.transform(X_test)
        scores = self.best_model.score_samples(X_test_scaled)
        predictions = (scores <= self.best_threshold).astype(int)

        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions, average='binary', zero_division=0
        )

        cm = confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = cm.ravel()

        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        # ROC-AUC (using negative scores as positive class scores)
        try:
            roc_auc = roc_auc_score(y_test, -scores)
        except:
            roc_auc = 0.0

        results = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'fp_rate': float(fp_rate),
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            },
            'threshold': float(self.best_threshold),
            'n_test_samples': len(y_test),
            'n_incidents': int(y_test.sum())
        }

        # Feature importance (approximate using permutation)
        if feature_names and len(feature_names) > 0:
            logger.info("Calculating feature importance...")
            importance = self._calculate_feature_importance(X_test_scaled, y_test, feature_names)
            results['feature_importance'] = importance

        self.evaluation_results = results

        logger.info(f"Test Results:")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  FP Rate: {fp_rate:.4f}")
        logger.info(f"  ROC-AUC: {roc_auc:.4f}")
        logger.info(f"  Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        return results

    def _calculate_feature_importance(
        self,
        X_scaled: np.ndarray,
        y_true: np.ndarray,
        feature_names: list,
        n_top: int = 20
    ) -> Dict[str, float]:
        """
        Calculate approximate feature importance using score degradation.
        """
        baseline_scores = self.best_model.score_samples(X_scaled)
        baseline_predictions = (baseline_scores <= self.best_threshold).astype(int)
        baseline_f1 = precision_recall_fscore_support(
            y_true, baseline_predictions, average='binary', zero_division=0
        )[2]

        importance_dict = {}

        # Sample features to speed up computation
        n_features = X_scaled.shape[1]
        features_to_test = min(n_features, 50)
        feature_indices = np.random.choice(n_features, features_to_test, replace=False)

        for idx in feature_indices:
            X_permuted = X_scaled.copy()
            np.random.shuffle(X_permuted[:, idx])

            permuted_scores = self.best_model.score_samples(X_permuted)
            permuted_predictions = (permuted_scores <= self.best_threshold).astype(int)
            permuted_f1 = precision_recall_fscore_support(
                y_true, permuted_predictions, average='binary', zero_division=0
            )[2]

            importance = baseline_f1 - permuted_f1
            importance_dict[feature_names[idx]] = float(importance)

        # Sort and return top N
        sorted_importance = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:n_top]
        )

        return sorted_importance


def load_training_data(filepath: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load and prepare training data."""
    logger.info(f"Loading training data from {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} samples")

    if 'label' not in df.columns:
        raise ValueError("Data must contain 'label' column (0=normal, 1=incident)")

    # Separate features and labels
    y = df['label'].values

    logger.info(f"Label distribution: Normal={np.sum(y==0)}, Incidents={np.sum(y==1)}")

    return df, y


def generate_synthetic_data(n_samples: int = 10000, incident_rate: float = 0.02) -> pd.DataFrame:
    """
    Generate synthetic training data for testing.
    """
    logger.info(f"Generating {n_samples} synthetic samples with {incident_rate*100:.1f}% incident rate")

    np.random.seed(42)

    # Normal operating ranges
    n_normal = int(n_samples * (1 - incident_rate))
    n_incident = n_samples - n_normal

    # Normal samples
    normal_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_normal, freq='1min'),
        'node_id': np.random.choice([f'node-{i:03d}' for i in range(100)], n_normal),
        'region': np.random.choice(['us-east', 'us-west', 'eu-west'], n_normal),
        'az': np.random.choice(['az1', 'az2', 'az3'], n_normal),
        'error_rate': np.random.gamma(2, 0.0005, n_normal),
        'timeout_rate': np.random.gamma(2, 0.0003, n_normal),
        'latency_p50': np.random.normal(10, 2, n_normal),
        'latency_p99': np.random.normal(30, 5, n_normal),
        'sla_violations': np.random.poisson(0.1, n_normal),
        'connection_failures': np.random.poisson(0.05, n_normal),
        'packet_loss_rate': np.random.gamma(2, 0.005, n_normal),
        'cpu_usage': np.random.beta(2, 5, n_normal) * 100,
        'memory_usage': np.random.beta(3, 4, n_normal) * 100,
        'disk_io': np.random.gamma(5, 10, n_normal),
        'dwcp_mode': np.random.choice(['standard', 'optimized', 'fallback'], n_normal),
        'network_tier': np.random.choice(['tier1', 'tier2', 'tier3'], n_normal),
        'label': np.zeros(n_normal, dtype=int)
    }

    # Incident samples (anomalous patterns)
    incident_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_incident, freq='1min'),
        'node_id': np.random.choice([f'node-{i:03d}' for i in range(100)], n_incident),
        'region': np.random.choice(['us-east', 'us-west', 'eu-west'], n_incident),
        'az': np.random.choice(['az1', 'az2', 'az3'], n_incident),
        'error_rate': np.random.gamma(5, 0.005, n_incident),  # Higher error rate
        'timeout_rate': np.random.gamma(5, 0.003, n_incident),  # Higher timeout rate
        'latency_p50': np.random.normal(50, 20, n_incident),  # Higher latency
        'latency_p99': np.random.normal(150, 50, n_incident),  # Much higher p99
        'sla_violations': np.random.poisson(2, n_incident),  # More violations
        'connection_failures': np.random.poisson(1, n_incident),  # More failures
        'packet_loss_rate': np.random.gamma(5, 0.02, n_incident),  # Higher packet loss
        'cpu_usage': np.random.beta(5, 2, n_incident) * 100,  # Higher CPU
        'memory_usage': np.random.beta(5, 2, n_incident) * 100,  # Higher memory
        'disk_io': np.random.gamma(10, 20, n_incident),  # Higher disk I/O
        'dwcp_mode': np.random.choice(['standard', 'optimized', 'fallback'], n_incident),
        'network_tier': np.random.choice(['tier1', 'tier2', 'tier3'], n_incident),
        'label': np.ones(n_incident, dtype=int)
    }

    df_normal = pd.DataFrame(normal_data)
    df_incident = pd.DataFrame(incident_data)

    df = pd.concat([df_normal, df_incident], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    logger.info(f"Generated {len(df)} samples: {n_normal} normal, {n_incident} incidents")

    return df


def save_model(
    model: IsolationForest,
    scaler: RobustScaler,
    threshold: float,
    params: Dict[str, Any],
    feature_names: list,
    output_dir: str
):
    """Save model artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving model artifacts to {output_dir}")

    # Save model
    model_path = output_dir / "isolation_forest_node_reliability.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save scaler
    scaler_path = output_dir / "scaler_node_reliability.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")

    # Save threshold and metadata
    metadata = {
        'model_type': 'isolation_forest',
        'task': 'node_reliability_anomaly_detection',
        'threshold': float(threshold),
        'hyperparameters': params,
        'n_features': len(feature_names),
        'feature_names': feature_names,
        'training_date': datetime.now().isoformat(),
        'target_recall': 0.98,
        'max_fp_rate': 0.05
    }

    metadata_path = output_dir / "model_metadata_node_reliability.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    # Save hyperparameters separately
    config_path = output_dir / "hyperparameters_node_reliability.json"
    with open(config_path, 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Hyperparameters saved to {config_path}")


def generate_evaluation_report(
    results: Dict[str, Any],
    params: Dict[str, Any],
    output_path: str
):
    """Generate markdown evaluation report."""
    logger.info(f"Generating evaluation report: {output_path}")

    cm = results['confusion_matrix']

    report = f"""# Node Reliability Isolation Forest - Evaluation Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Configuration

**Hyperparameters**:
- n_estimators: {params['n_estimators']}
- max_samples: {params['max_samples']}
- max_features: {params['max_features']}
- contamination: {params['contamination']:.4f}

**Decision Threshold**: {results['threshold']:.6f}

## Performance Metrics

### Classification Metrics
- **Recall**: {results['recall']:.4f} (Target: ≥0.98)
- **Precision**: {results['precision']:.4f}
- **F1 Score**: {results['f1_score']:.4f}
- **ROC-AUC**: {results['roc_auc']:.4f}
- **False Positive Rate**: {results['fp_rate']:.4f} (Target: <0.05)

### Confusion Matrix

|              | Predicted Normal | Predicted Incident |
|--------------|------------------|-------------------|
| **Actual Normal**   | {cm['tn']}         | {cm['fp']}          |
| **Actual Incident** | {cm['fn']}         | {cm['tp']}          |

- True Negatives (TN): {cm['tn']}
- False Positives (FP): {cm['fp']}
- False Negatives (FN): {cm['fn']}
- True Positives (TP): {cm['tp']}

### Test Set Statistics
- Total Samples: {results['n_test_samples']}
- Incident Samples: {results['n_incidents']}
- Incident Rate: {results['n_incidents']/results['n_test_samples']*100:.2f}%

## Target Achievement

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Recall | ≥98% | {results['recall']*100:.2f}% | {'✓ PASS' if results['recall'] >= 0.98 else '✗ FAIL'} |
| FP Rate | <5% | {results['fp_rate']*100:.2f}% | {'✓ PASS' if results['fp_rate'] < 0.05 else '✗ FAIL'} |

"""

    # Add feature importance if available
    if 'feature_importance' in results and results['feature_importance']:
        report += "\n## Feature Importance (Top 20)\n\n"
        report += "Features ranked by impact on model performance:\n\n"
        report += "| Rank | Feature | Importance |\n"
        report += "|------|---------|------------|\n"

        for rank, (feature, importance) in enumerate(results['feature_importance'].items(), 1):
            report += f"| {rank} | {feature} | {importance:.6f} |\n"

    report += f"""
## Recommendations

"""

    if results['recall'] >= 0.98 and results['fp_rate'] < 0.05:
        report += "✓ Model meets all target requirements and is ready for production deployment.\n\n"
    else:
        if results['recall'] < 0.98:
            report += f"⚠ Recall ({results['recall']:.4f}) is below target (0.98). Consider:\n"
            report += "  - Lowering decision threshold\n"
            report += "  - Increasing training data with more incident examples\n"
            report += "  - Engineering additional features\n\n"

        if results['fp_rate'] >= 0.05:
            report += f"⚠ False positive rate ({results['fp_rate']:.4f}) exceeds target (0.05). Consider:\n"
            report += "  - Raising decision threshold\n"
            report += "  - Reducing contamination parameter\n"
            report += "  - Improving feature engineering\n\n"

    report += """
## Model Deployment

**Model Files**:
- isolation_forest_node_reliability.pkl - Trained model
- scaler_node_reliability.pkl - Feature scaler
- model_metadata_node_reliability.json - Model configuration
- hyperparameters_node_reliability.json - Optimal hyperparameters

**CLI Reproduction**:
```bash
python train_isolation_forest.py \\
  --data /path/to/data.csv \\
  --output ../models \\
  --target-recall 0.98 \\
  --max-fp-rate 0.05
```

## Next Steps

1. Validate model on production data stream
2. Monitor false positive/negative rates in production
3. Retrain monthly with updated incident data
4. A/B test against current alerting system
5. Integrate with DWCP monitoring dashboard
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report)

    logger.info(f"Evaluation report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train Isolation Forest for Node Reliability Anomaly Detection'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to training data CSV with labels'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic data for testing'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=10000,
        help='Number of synthetic samples to generate'
    )
    parser.add_argument(
        '--incident-rate',
        type=float,
        default=0.02,
        help='Incident rate for synthetic data (default: 0.02 = 2%%)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='../models',
        help='Output directory for model artifacts'
    )
    parser.add_argument(
        '--report',
        type=str,
        default='../../../../../../docs/models/node_reliability_eval.md',
        help='Path for evaluation report'
    )
    parser.add_argument(
        '--target-recall',
        type=float,
        default=0.98,
        help='Target recall rate (default: 0.98)'
    )
    parser.add_argument(
        '--max-fp-rate',
        type=float,
        default=0.05,
        help='Maximum acceptable false positive rate (default: 0.05)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Node Reliability Isolation Forest Training")
    logger.info("=" * 80)

    # Load or generate data
    if args.synthetic or not args.data:
        df = generate_synthetic_data(n_samples=args.n_samples, incident_rate=args.incident_rate)
        y = df['label'].values
    else:
        df, y = load_training_data(args.data)

    # Feature engineering
    feature_engineer = FeatureEngineer(rolling_windows=[5, 15, 30])
    df_engineered = feature_engineer.engineer_features(df)

    # Extract features
    X = df_engineered[feature_engineer.feature_names].values

    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {len(feature_engineer.feature_names)}")

    # Temporal train/test split
    split_idx = int(len(X) * (1 - args.test_size))
    X_train_full, X_test = X[:split_idx], X[split_idx:]
    y_train_full, y_test = y[:split_idx], y[split_idx:]

    # Further split training into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    # Tune model
    tuner = IsolationForestTuner(
        target_recall=args.target_recall,
        max_fp_rate=args.max_fp_rate
    )

    tuning_results = tuner.tune_hyperparameters(X_train, y_train, X_val, y_val)

    logger.info("\n" + "=" * 80)
    logger.info("Hyperparameter Tuning Complete")
    logger.info("=" * 80)
    logger.info(f"Best parameters: {tuning_results['best_params']}")
    logger.info(f"Best threshold: {tuning_results['best_threshold']:.6f}")

    # Evaluate on test set
    evaluation_results = tuner.evaluate(X_test, y_test, feature_engineer.feature_names)

    logger.info("\n" + "=" * 80)
    logger.info("Final Test Set Evaluation")
    logger.info("=" * 80)

    # Save model
    save_model(
        tuner.best_model,
        tuner.best_scaler,
        tuner.best_threshold,
        tuner.best_params,
        feature_engineer.feature_names,
        args.output
    )

    # Generate report
    generate_evaluation_report(
        evaluation_results,
        tuner.best_params,
        args.report
    )

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)

    # Print summary
    recall_status = "✓ PASS" if evaluation_results['recall'] >= args.target_recall else "✗ FAIL"
    fp_status = "✓ PASS" if evaluation_results['fp_rate'] < args.max_fp_rate else "✗ FAIL"

    print("\n" + "=" * 80)
    print("FINAL RESULTS SUMMARY")
    print("=" * 80)
    print(f"Recall:          {evaluation_results['recall']*100:.2f}% (Target: {args.target_recall*100:.0f}%) {recall_status}")
    print(f"Precision:       {evaluation_results['precision']*100:.2f}%")
    print(f"F1 Score:        {evaluation_results['f1_score']:.4f}")
    print(f"FP Rate:         {evaluation_results['fp_rate']*100:.2f}% (Target: <{args.max_fp_rate*100:.0f}%) {fp_status}")
    print(f"ROC-AUC:         {evaluation_results['roc_auc']:.4f}")
    print("=" * 80)
    print(f"\nModel artifacts saved to: {Path(args.output).absolute()}")
    print(f"Evaluation report: {Path(args.report).absolute()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
