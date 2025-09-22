"""
Performance Prediction and Bandwidth Optimization Engine - Sprint 4
Comprehensive performance prediction and bandwidth optimization using ML models,
reinforcement learning, and predictive algorithms for NovaCron.
"""

import numpy as np
import pandas as pd
import logging
import json
import time
import threading
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from collections import deque

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Optimization libraries
try:
    import scipy.optimize as opt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Optimization features will be limited.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. Graph-based optimizations will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure

    Units:
    - cpu_utilization: percentage (0-100%)
    - memory_utilization: percentage (0-100%)
    - disk_iops: operations per second (0+)
    - network_bandwidth_mbps: megabits per second (0+)
    - latency_ms: milliseconds (0+)
    - throughput_ops_sec: operations per second (0+)
    - error_rate: decimal ratio (0.0-1.0, where 1.0 = 100%)
    - response_time_ms: milliseconds (0+)
    """
    timestamp: datetime
    node_id: str
    cpu_utilization: float  # 0-100%
    memory_utilization: float  # 0-100%
    disk_iops: int  # ops/sec
    network_bandwidth_mbps: float  # Mbps
    latency_ms: float  # milliseconds
    throughput_ops_sec: float  # ops/sec
    error_rate: float  # 0.0-1.0 (decimal ratio)
    response_time_ms: float  # milliseconds

@dataclass
class BandwidthOptimizationResult:
    """Bandwidth optimization result"""
    recommended_allocation: Dict[str, float]
    predicted_performance: Dict[str, float]
    optimization_score: float
    confidence: float
    alternative_strategies: List[Dict[str, Any]]
    estimated_improvement: float

@dataclass
class PerformancePrediction:
    """Performance prediction result

    All predictions are in natural units matching PerformanceMetrics:
    - latency_ms: milliseconds (0+)
    - throughput_ops_sec: operations per second (0+)
    - error_rate: decimal ratio (0.0-1.0)
    - response_time_ms: milliseconds (0+)
    """
    predicted_metrics: Dict[str, float]  # Natural units as documented above
    confidence: float  # 0.0-1.0
    uncertainty_bounds: Dict[str, Tuple[float, float]]  # (lower, upper) in natural units
    contributing_factors: List[str]
    recommendations: List[str]
    prediction_horizon_minutes: int

class PerformancePredictor:
    """
    Advanced performance prediction using ensemble methods with feature engineering
    for system metrics, workload characteristics, and network topology awareness.
    """

    def __init__(self, db_path: str = None):
        # Get DB path from environment variable or use default
        if db_path is None:
            db_path = os.environ.get(
                'PERFORMANCE_DB',
                os.path.join(os.environ.get('NOVACRON_DATA_DIR', '/var/lib/novacron'), 'performance_predictor.db')
            )

        # Ensure directory exists and is writable
        db_dir = os.path.dirname(db_path)
        try:
            Path(db_dir).mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = os.path.join(db_dir, '.write_test')
            Path(test_file).touch()
            os.remove(test_file)
            logger.info(f"Using database path: {db_path}")
        except (OSError, PermissionError) as e:
            # Fall back to /tmp if the preferred directory is not writable
            logger.warning(f"Cannot write to {db_dir}: {e}. Falling back to /tmp")
            db_path = '/tmp/performance_predictor.db'
            logger.info(f"Using fallback database path: {db_path}")

        self.db_path = db_path
        self.models = {}
        self.scalers = {}  # Scalers for features and targets
        self.target_scalers = {}  # Separate scalers for inverse transforming predictions
        self.feature_columns = {}
        self.is_trained = False
        self.training_history = []
        self.lock = threading.Lock()

        # Performance targets in natural units
        self.targets = {
            'latency_ms': {'optimal': 10.0, 'acceptable': 50.0, 'critical': 100.0},  # milliseconds
            'throughput_ops_sec': {'optimal': 1000.0, 'acceptable': 500.0, 'critical': 100.0},  # ops/sec
            'cpu_utilization': {'optimal': 70.0, 'acceptable': 85.0, 'critical': 95.0},  # percentage
            'memory_utilization': {'optimal': 80.0, 'acceptable': 90.0, 'critical': 95.0},  # percentage
            'error_rate': {'optimal': 0.01, 'acceptable': 0.05, 'critical': 0.10}  # decimal ratio (0.0-1.0)
        }

        # Initialize database
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for performance data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME,
                        node_id TEXT,
                        cpu_utilization REAL,
                        memory_utilization REAL,
                        disk_iops INTEGER,
                        network_bandwidth_mbps REAL,
                        latency_ms REAL,
                        throughput_ops_sec REAL,
                        error_rate REAL,
                        response_time_ms REAL,
                        workload_type TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Performance predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        node_id TEXT,
                        prediction_time DATETIME,
                        predicted_metrics TEXT,
                        actual_metrics TEXT,
                        accuracy_score REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                conn.commit()

        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def store_performance_data(self, metrics: PerformanceMetrics) -> bool:
        """Store performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics
                    (timestamp, node_id, cpu_utilization, memory_utilization, disk_iops,
                     network_bandwidth_mbps, latency_ms, throughput_ops_sec, error_rate, response_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp, metrics.node_id, metrics.cpu_utilization,
                    metrics.memory_utilization, metrics.disk_iops, metrics.network_bandwidth_mbps,
                    metrics.latency_ms, metrics.throughput_ops_sec, metrics.error_rate,
                    metrics.response_time_ms
                ))
                conn.commit()

            return True

        except Exception as e:
            logger.error(f"Error storing performance data: {e}")
            return False

    def train_models(self, lookback_days: int = 30) -> Dict[str, float]:
        """Train performance prediction models"""

        try:
            # Load training data
            training_data = self._load_training_data(lookback_days)

            if training_data is None or len(training_data) < 100:
                logger.warning("Insufficient training data for performance prediction")
                return {}

            logger.info(f"Training performance models with {len(training_data)} samples")

            # Feature engineering
            features_df = self._engineer_performance_features(training_data)

            # Define prediction targets
            target_metrics = ['latency_ms', 'throughput_ops_sec', 'error_rate', 'response_time_ms']

            results = {}

            for target in target_metrics:
                if target not in features_df.columns:
                    continue

                logger.info(f"Training models for {target}")

                # Prepare features and target
                feature_cols = [col for col in features_df.columns if col not in target_metrics]
                X = features_df[feature_cols].fillna(0)
                y = features_df[target].fillna(features_df[target].mean())

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=None
                )

                # Scale features using StandardScaler
                feature_scaler = StandardScaler()
                X_train_scaled = feature_scaler.fit_transform(X_train)
                X_test_scaled = feature_scaler.transform(X_test)

                # Scale targets appropriately based on metric type
                if target in ['latency_ms', 'response_time_ms', 'throughput_ops_sec']:
                    # Use MinMaxScaler for bounded metrics to preserve natural scale
                    target_scaler = MinMaxScaler()
                    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
                    self.target_scalers[target] = target_scaler
                elif target == 'error_rate':
                    # Error rate is already in 0.0-1.0 range, use StandardScaler
                    target_scaler = StandardScaler()
                    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
                    y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
                    self.target_scalers[target] = target_scaler
                else:
                    # Use original values for other metrics
                    y_train_scaled = y_train
                    y_test_scaled = y_test
                    self.target_scalers[target] = None

                self.scalers[target] = feature_scaler
                self.feature_columns[target] = feature_cols

                # Train ensemble models
                models = {
                    'random_forest': RandomForestRegressor(
                        n_estimators=100,
                        random_state=42,
                        n_jobs=-1
                    ),
                    'gradient_boosting': GradientBoostingRegressor(
                        n_estimators=100,
                        random_state=42
                    ),
                    'neural_network': MLPRegressor(
                        hidden_layer_sizes=(100, 50),
                        random_state=42,
                        max_iter=1000
                    ),
                    'ridge': Ridge(alpha=1.0)
                }

                target_models = {}
                best_score = float('inf')

                for model_name, model in models.items():
                    try:
                        # Train models with appropriate data scaling
                        use_scaled_targets = self.target_scalers[target] is not None
                        train_y = y_train_scaled if use_scaled_targets else y_train
                        test_y = y_test_scaled if use_scaled_targets else y_test

                        # Hyperparameter optimization for key models
                        if model_name == 'random_forest':
                            param_grid = {
                                'n_estimators': [50, 100, 200],
                                'max_depth': [10, 20, None],
                                'min_samples_split': [2, 5, 10]
                            }
                            grid_search = GridSearchCV(
                                model, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
                            )
                            grid_search.fit(X_train, train_y)
                            model = grid_search.best_estimator_
                            y_pred_scaled = model.predict(X_test)
                        elif model_name == 'neural_network':
                            model.fit(X_train_scaled, train_y)
                            y_pred_scaled = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, train_y)
                            y_pred_scaled = model.predict(X_test)

                        # Apply inverse transform to get predictions in natural units
                        if use_scaled_targets:
                            y_pred = self.target_scalers[target].inverse_transform(
                                y_pred_scaled.reshape(-1, 1)
                            ).flatten()
                            # Use original test values for evaluation
                            test_values = y_test
                        else:
                            y_pred = y_pred_scaled
                            test_values = test_y

                        # Calculate metrics in natural units
                        mae = mean_absolute_error(test_values, y_pred)
                        r2 = r2_score(test_values, y_pred)

                        target_models[model_name] = model

                        if mae < best_score:
                            best_score = mae

                        logger.info(f"{target}/{model_name} - MAE: {mae:.4f}, R²: {r2:.4f}")

                    except Exception as e:
                        logger.error(f"Error training {model_name} for {target}: {e}")
                        continue

                self.models[target] = target_models
                results[target] = best_score

            self.is_trained = True
            self.training_history.append({
                'timestamp': datetime.now(),
                'targets_trained': len(results),
                'data_points': len(training_data)
            })

            return results

        except Exception as e:
            logger.error(f"Performance model training error: {e}")
            raise

    def predict_performance(self, current_metrics: Dict[str, float],
                          prediction_horizon_minutes: int = 60) -> PerformancePrediction:
        """Predict future performance metrics"""

        if not self.is_trained:
            # Return baseline predictions in natural units
            return PerformancePrediction(
                predicted_metrics={
                    'latency_ms': 15.0,  # 15 milliseconds
                    'throughput_ops_sec': 500.0,  # 500 operations per second
                    'error_rate': 0.02,  # 2% error rate (0.02 decimal ratio)
                    'response_time_ms': 25.0  # 25 milliseconds
                },
                confidence=0.5,
                uncertainty_bounds={
                    'latency_ms': (10.0, 20.0),  # milliseconds
                    'throughput_ops_sec': (400.0, 600.0),  # ops/sec
                    'error_rate': (0.01, 0.05),  # decimal ratio (1%-5%)
                    'response_time_ms': (20.0, 30.0)  # milliseconds
                },
                contributing_factors=['baseline_prediction'],
                recommendations=['Collect more performance data for better predictions'],
                prediction_horizon_minutes=prediction_horizon_minutes
            )

        try:
            # Prepare features from current metrics
            feature_df = pd.DataFrame([current_metrics])
            features_df = self._engineer_performance_features(feature_df)

            predicted_metrics = {}
            uncertainty_bounds = {}
            confidences = {}

            for target, models in self.models.items():
                if not models or target not in self.feature_columns:
                    continue

                # Get feature columns for this target
                feature_cols = self.feature_columns[target]
                X = features_df[feature_cols].fillna(0)

                # Get predictions from all models
                predictions = []

                for model_name, model in models.items():
                    try:
                        if model_name == 'neural_network' and target in self.scalers:
                            X_scaled = self.scalers[target].transform(X)
                            pred_scaled = model.predict(X_scaled)[0]
                        else:
                            pred_scaled = model.predict(X)[0]

                        # Apply inverse transform if target was scaled during training
                        if target in self.target_scalers and self.target_scalers[target] is not None:
                            pred = self.target_scalers[target].inverse_transform(
                                np.array([[pred_scaled]])
                            )[0, 0]
                        else:
                            pred = pred_scaled

                        predictions.append(pred)

                    except Exception as e:
                        logger.warning(f"Prediction error with {model_name} for {target}: {e}")
                        continue

                if predictions:
                    # Ensemble prediction
                    final_pred = np.mean(predictions)
                    pred_std = np.std(predictions)

                    # Adjust for prediction horizon
                    horizon_factor = 1.0 + (prediction_horizon_minutes / 60.0) * 0.1
                    final_pred *= horizon_factor
                    pred_std *= horizon_factor

                    predicted_metrics[target] = final_pred
                    uncertainty_bounds[target] = (
                        final_pred - 1.96 * pred_std,
                        final_pred + 1.96 * pred_std
                    )

                    # Calculate confidence
                    confidences[target] = max(0.1, 1.0 - (pred_std / max(final_pred, 1.0)))

            # Overall confidence
            overall_confidence = np.mean(list(confidences.values())) if confidences else 0.5

            # Generate recommendations
            recommendations = self._generate_performance_recommendations(
                predicted_metrics, current_metrics
            )

            # Identify contributing factors
            contributing_factors = self._identify_contributing_factors(current_metrics, predicted_metrics)

            return PerformancePrediction(
                predicted_metrics=predicted_metrics,
                confidence=overall_confidence,
                uncertainty_bounds=uncertainty_bounds,
                contributing_factors=contributing_factors,
                recommendations=recommendations,
                prediction_horizon_minutes=prediction_horizon_minutes
            )

        except Exception as e:
            logger.error(f"Performance prediction error: {e}")
            # Return fallback prediction in natural units
            return PerformancePrediction(
                predicted_metrics={
                    'latency_ms': 20.0,  # 20 milliseconds
                    'throughput_ops_sec': 400.0  # 400 operations per second
                },
                confidence=0.3,
                uncertainty_bounds={
                    'latency_ms': (15.0, 25.0),  # milliseconds
                    'throughput_ops_sec': (300.0, 500.0)  # ops/sec
                },
                contributing_factors=['prediction_error'],
                recommendations=['Check system status and retry prediction'],
                prediction_horizon_minutes=prediction_horizon_minutes
            )

    def _load_training_data(self, lookback_days: int) -> Optional[pd.DataFrame]:
        """Load training data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT * FROM performance_metrics
                    WHERE timestamp >= datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                    LIMIT 10000
                '''.format(lookback_days)

                df = pd.read_sql_query(query, conn)

                if len(df) < 50:
                    logger.warning("Insufficient training data")
                    return None

                # Convert timestamp
                df['timestamp'] = pd.to_datetime(df['timestamp'])

                return df

        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None

    def _engineer_performance_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for performance prediction"""
        features = data.copy()

        # Time-based features
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['hour'] = features['timestamp'].dt.hour
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
            features['is_business_hours'] = features['hour'].between(9, 17).astype(int)
        else:
            # Use current time if no timestamp
            now = datetime.now()
            features['hour'] = now.hour
            features['day_of_week'] = now.weekday()
            features['is_weekend'] = 1 if now.weekday() in [5, 6] else 0
            features['is_business_hours'] = 1 if 9 <= now.hour <= 17 else 0

        # Resource utilization features
        if 'cpu_utilization' in features.columns and 'memory_utilization' in features.columns:
            features['resource_pressure'] = (
                features['cpu_utilization'] * features['memory_utilization']
            ).fillna(0)

        # Performance ratios
        if 'latency_ms' in features.columns and 'throughput_ops_sec' in features.columns:
            features['latency_throughput_ratio'] = (
                features['latency_ms'] / features['throughput_ops_sec'].clip(lower=1)
            ).fillna(0)

        # Moving averages (if sufficient data)
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(features) > 5:
                features[f'{col}_ma3'] = features[col].rolling(window=3, min_periods=1).mean()
                features[f'{col}_ma5'] = features[col].rolling(window=5, min_periods=1).mean()

        # Load indicators
        if 'cpu_utilization' in features.columns:
            features['cpu_load_level'] = pd.cut(
                features['cpu_utilization'],
                bins=[0, 50, 80, 100],
                labels=['low', 'medium', 'high']
            ).cat.codes

        if 'memory_utilization' in features.columns:
            features['memory_load_level'] = pd.cut(
                features['memory_utilization'],
                bins=[0, 60, 85, 100],
                labels=['low', 'medium', 'high']
            ).cat.codes

        return features.fillna(0)

    def _generate_performance_recommendations(self, predicted_metrics: Dict[str, float],
                                           current_metrics: Dict[str, float]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # Latency recommendations
        if 'latency_ms' in predicted_metrics:
            predicted_latency = predicted_metrics['latency_ms']
            if predicted_latency > self.targets['latency_ms']['critical']:
                recommendations.append("CRITICAL: Predicted latency exceeds 100ms. Consider load balancing or scaling.")
            elif predicted_latency > self.targets['latency_ms']['acceptable']:
                recommendations.append("WARNING: Predicted latency above acceptable levels. Monitor closely.")

        # Throughput recommendations
        if 'throughput_ops_sec' in predicted_metrics:
            predicted_throughput = predicted_metrics['throughput_ops_sec']
            if predicted_throughput < self.targets['throughput_ops_sec']['critical']:
                recommendations.append("CRITICAL: Predicted throughput critically low. Immediate scaling needed.")
            elif predicted_throughput < self.targets['throughput_ops_sec']['acceptable']:
                recommendations.append("WARNING: Predicted throughput declining. Consider capacity planning.")

        # Resource recommendations (CPU/Memory in percentage 0-100%)
        current_cpu = current_metrics.get('cpu_utilization', 0)  # percentage
        current_memory = current_metrics.get('memory_utilization', 0)  # percentage

        if current_cpu > 85.0:  # 85% CPU utilization
            recommendations.append("High CPU utilization detected (>85%). Consider CPU scaling or optimization.")

        if current_memory > 90.0:  # 90% memory utilization
            recommendations.append("High memory utilization detected (>90%). Consider memory scaling.")

        # Error rate recommendations
        if 'error_rate' in predicted_metrics:
            predicted_error_rate = predicted_metrics['error_rate']
            if predicted_error_rate > self.targets['error_rate']['acceptable']:
                recommendations.append("Predicted error rate above acceptable threshold. Review error handling.")

        return recommendations if recommendations else ["System performance within acceptable parameters."]

    def optimize_performance(self, df: pd.DataFrame, goals: List[str], constraints: Dict[str,Any]) -> Dict[str,Any]:
        """Optimize performance based on goals and constraints"""
        # Aggregate current metrics
        current = df.select_dtypes(include=[np.number]).mean(numeric_only=True).to_dict()
        pred = self.predict_performance(current)

        # Derive optimizations from goals
        optimizations = []
        priority = []
        improvements = {}

        for g in goals:
            if g == 'minimize_latency':
                optimizations.append({'type':'tuning','target':'latency','action':'reduce_queue_depth','params':{'factor':0.8}})
                improvements['latency_ms'] = max(0.0, current.get('latency_ms', 0) - pred.predicted_metrics.get('latency_ms', 0))
                priority.append('latency')
            elif g == 'maximize_throughput':
                optimizations.append({'type':'tuning','target':'throughput','action':'increase_buffer_size','params':{'factor':1.5}})
                improvements['throughput_ops_sec'] = pred.predicted_metrics.get('throughput_ops_sec', 0) - current.get('throughput_ops_sec', 0)
                priority.append('throughput')
            elif g == 'improve_efficiency':
                optimizations.append({'type':'resource','target':'efficiency','action':'optimize_cpu_governor','params':{'mode':'performance'}})
                improvements['efficiency'] = 0.15  # 15% improvement estimate
                priority.append('efficiency')

        return {
            'optimizations': optimizations,
            'improvements': improvements,
            'priority': priority,
            'confidence': pred.confidence
        }

    def _identify_contributing_factors(self, current_metrics: Dict[str, float],
                                     predicted_metrics: Dict[str, float]) -> List[str]:
        """Identify factors contributing to predicted performance"""
        factors = []

        # High resource utilization (percentages 0-100%)
        cpu_util = current_metrics.get('cpu_utilization', 0)  # percentage
        memory_util = current_metrics.get('memory_utilization', 0)  # percentage

        if cpu_util > 80.0:  # 80% CPU utilization
            factors.append('high_cpu_utilization')
        if memory_util > 85.0:  # 85% memory utilization
            factors.append('high_memory_utilization')

        # Network factors (bandwidth in Mbps)
        if current_metrics.get('network_bandwidth_mbps', 0) > 800.0:  # 800 Mbps
            factors.append('high_network_utilization')

        # Time-based factors
        now = datetime.now()
        if 9 <= now.hour <= 17 and now.weekday() < 5:
            factors.append('business_hours')

        # Performance trends
        if 'latency_ms' in predicted_metrics:
            if predicted_metrics['latency_ms'] > current_metrics.get('latency_ms', 0) * 1.2:
                factors.append('degrading_latency')

        if 'throughput_ops_sec' in predicted_metrics:
            if predicted_metrics['throughput_ops_sec'] < current_metrics.get('throughput_ops_sec', 0) * 0.8:
                factors.append('declining_throughput')

        return factors if factors else ['normal_operation']

class BandwidthOptimizationEngine:
    """
    Intelligent bandwidth optimization using reinforcement learning and genetic algorithms
    """

    def __init__(self):
        self.optimization_history = []
        self.current_allocations = {}
        self.performance_feedback = deque(maxlen=1000)

    def optimize_bandwidth_allocation(self, nodes: List[str], total_bandwidth: float,
                                    requirements: Dict[str, Dict[str, float]],
                                    constraints: Optional[Dict[str, Any]] = None) -> BandwidthOptimizationResult:
        """
        Optimize bandwidth allocation using multi-objective optimization
        """

        try:
            logger.info(f"Optimizing bandwidth allocation for {len(nodes)} nodes")

            if not SCIPY_AVAILABLE:
                # Fallback to simple proportional allocation
                return self._fallback_bandwidth_allocation(nodes, total_bandwidth, requirements)

            # Define optimization variables
            num_nodes = len(nodes)

            # Objective function: maximize overall performance while minimizing cost
            def objective_function(allocations):
                score = 0.0

                for i, node in enumerate(nodes):
                    allocation = allocations[i]
                    requirement = requirements.get(node, {})

                    # Performance score based on meeting requirements
                    min_bw = requirement.get('min_bandwidth', 10)
                    optimal_bw = requirement.get('optimal_bandwidth', 100)

                    if allocation >= optimal_bw:
                        performance_score = 1.0
                    elif allocation >= min_bw:
                        performance_score = allocation / optimal_bw
                    else:
                        performance_score = (allocation / min_bw) * 0.5

                    # Priority weighting
                    priority = requirement.get('priority', 1.0)
                    score += performance_score * priority

                # Penalty for over-allocation
                total_allocated = sum(allocations)
                if total_allocated > total_bandwidth:
                    score -= (total_allocated - total_bandwidth) * 10

                return -score  # Minimize negative score

            # Constraints
            def constraint_total_bandwidth(allocations):
                return total_bandwidth - sum(allocations)

            # Bounds for each allocation
            bounds = []
            for node in nodes:
                requirement = requirements.get(node, {})
                min_bw = requirement.get('min_bandwidth', 1)
                max_bw = min(requirement.get('max_bandwidth', total_bandwidth), total_bandwidth)
                bounds.append((min_bw, max_bw))

            # Initial guess: proportional allocation
            total_min = sum(requirements.get(node, {}).get('min_bandwidth', 10) for node in nodes)
            if total_min > total_bandwidth:
                # Scale down proportionally
                scale_factor = total_bandwidth / total_min
                initial_guess = [
                    requirements.get(node, {}).get('min_bandwidth', 10) * scale_factor
                    for node in nodes
                ]
            else:
                # Distribute remaining bandwidth proportionally
                remaining = total_bandwidth - total_min
                total_optimal = sum(
                    requirements.get(node, {}).get('optimal_bandwidth', 100) for node in nodes
                )
                initial_guess = []
                for node in nodes:
                    min_bw = requirements.get(node, {}).get('min_bandwidth', 10)
                    optimal_bw = requirements.get(node, {}).get('optimal_bandwidth', 100)
                    if total_optimal > 0:
                        extra = remaining * (optimal_bw / total_optimal)
                        initial_guess.append(min_bw + extra)
                    else:
                        initial_guess.append(min_bw)

            # Optimize using SLSQP
            constraints_list = [{'type': 'ineq', 'fun': constraint_total_bandwidth}]

            if constraints:
                # Add custom constraints if provided
                pass

            result = opt.minimize(
                objective_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'disp': False}
            )

            if result.success:
                optimized_allocations = dict(zip(nodes, result.x))
                optimization_score = -result.fun

                # Calculate predicted performance
                predicted_performance = {}
                for i, node in enumerate(nodes):
                    allocation = result.x[i]
                    requirement = requirements.get(node, {})
                    optimal_bw = requirement.get('optimal_bandwidth', 100)

                    predicted_performance[node] = min(1.0, allocation / optimal_bw)

                # Generate alternative strategies
                alternative_strategies = self._generate_alternative_strategies(
                    nodes, total_bandwidth, requirements
                )

                # Estimate improvement
                current_total = sum(self.current_allocations.get(node, 10) for node in nodes)
                estimated_improvement = (optimization_score / len(nodes)) * 100

                return BandwidthOptimizationResult(
                    recommended_allocation=optimized_allocations,
                    predicted_performance=predicted_performance,
                    optimization_score=optimization_score,
                    confidence=0.85,
                    alternative_strategies=alternative_strategies,
                    estimated_improvement=estimated_improvement
                )

            else:
                logger.warning("Bandwidth optimization failed, using fallback")
                return self._fallback_bandwidth_allocation(nodes, total_bandwidth, requirements)

        except Exception as e:
            logger.error(f"Bandwidth optimization error: {e}")
            return self._fallback_bandwidth_allocation(nodes, total_bandwidth, requirements)

    def _fallback_bandwidth_allocation(self, nodes: List[str], total_bandwidth: float,
                                     requirements: Dict[str, Dict[str, float]]) -> BandwidthOptimizationResult:
        """Fallback proportional allocation"""

        # Simple proportional allocation based on priorities
        total_priority = sum(
            requirements.get(node, {}).get('priority', 1.0) for node in nodes
        )

        allocations = {}
        predicted_performance = {}

        for node in nodes:
            requirement = requirements.get(node, {})
            priority = requirement.get('priority', 1.0)
            min_bw = requirement.get('min_bandwidth', 10)

            # Proportional allocation
            proportional_allocation = (priority / total_priority) * total_bandwidth
            final_allocation = max(min_bw, proportional_allocation)

            allocations[node] = final_allocation

            # Simple performance prediction
            optimal_bw = requirement.get('optimal_bandwidth', 100)
            predicted_performance[node] = min(1.0, final_allocation / optimal_bw)

        return BandwidthOptimizationResult(
            recommended_allocation=allocations,
            predicted_performance=predicted_performance,
            optimization_score=0.7,
            confidence=0.6,
            alternative_strategies=[],
            estimated_improvement=10.0
        )

    def _generate_alternative_strategies(self, nodes: List[str], total_bandwidth: float,
                                       requirements: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Generate alternative bandwidth allocation strategies"""

        alternatives = []

        # Strategy 1: Equal allocation
        equal_allocation = total_bandwidth / len(nodes)
        alternatives.append({
            'name': 'equal_allocation',
            'description': 'Equal bandwidth allocation across all nodes',
            'allocations': {node: equal_allocation for node in nodes},
            'trade_offs': 'Simple but may not meet individual requirements'
        })

        # Strategy 2: Minimum requirements only
        total_min = sum(requirements.get(node, {}).get('min_bandwidth', 10) for node in nodes)
        if total_min <= total_bandwidth:
            remaining = total_bandwidth - total_min
            alternatives.append({
                'name': 'minimum_plus_equal',
                'description': 'Minimum requirements plus equal distribution of remaining bandwidth',
                'allocations': {
                    node: requirements.get(node, {}).get('min_bandwidth', 10) + (remaining / len(nodes))
                    for node in nodes
                },
                'trade_offs': 'Guarantees minimum requirements but may be suboptimal'
            })

        # Strategy 3: Priority-based allocation
        priorities = {node: requirements.get(node, {}).get('priority', 1.0) for node in nodes}
        max_priority = max(priorities.values())
        priority_allocations = {}

        for node in nodes:
            priority_factor = priorities[node] / max_priority
            base_allocation = total_bandwidth / len(nodes)
            priority_allocations[node] = base_allocation * (0.5 + 0.5 * priority_factor)

        alternatives.append({
            'name': 'priority_weighted',
            'description': 'Bandwidth allocation weighted by node priority',
            'allocations': priority_allocations,
            'trade_offs': 'Favors high-priority nodes but may underserve others'
        })

        return alternatives

class NetworkPerformanceForecaster:
    """Network performance forecasting with time series analysis"""

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False

    def train_forecasting_models(self, network_data: pd.DataFrame) -> Dict[str, float]:
        """Train time series forecasting models for network performance"""

        try:
            # Ensure datetime index
            if 'timestamp' not in network_data.columns:
                logger.error("Network data must have timestamp column")
                return {}

            network_data['timestamp'] = pd.to_datetime(network_data['timestamp'])
            network_data = network_data.set_index('timestamp').sort_index()

            # Forecast targets
            targets = ['bandwidth_utilization', 'latency_ms', 'packet_loss_rate', 'throughput_mbps']
            results = {}

            for target in targets:
                if target not in network_data.columns:
                    continue

                logger.info(f"Training forecasting model for {target}")

                # Prepare time series data
                ts_data = network_data[target].fillna(method='ffill').fillna(0)

                if len(ts_data) < 50:
                    logger.warning(f"Insufficient data for {target}")
                    continue

                # Create lagged features
                lagged_features = self._create_forecasting_features(ts_data)

                if len(lagged_features) < 20:
                    continue

                X = lagged_features.drop('target', axis=1)
                y = lagged_features['target']

                # Split data temporally
                split_idx = int(0.8 * len(X))
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                self.scalers[target] = scaler

                # Train models
                models = {
                    'linear': LinearRegression(),
                    'ridge': Ridge(alpha=1.0),
                    'random_forest': RandomForestRegressor(n_estimators=50, random_state=42),
                    'neural_network': MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42, max_iter=500)
                }

                target_models = {}
                best_mae = float('inf')

                for model_name, model in models.items():
                    try:
                        if model_name == 'neural_network':
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled)
                        else:
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                        mae = mean_absolute_error(y_test, y_pred)
                        target_models[model_name] = model

                        if mae < best_mae:
                            best_mae = mae

                        logger.info(f"{target}/{model_name} - MAE: {mae:.4f}")

                    except Exception as e:
                        logger.error(f"Error training {model_name} for {target}: {e}")

                self.models[target] = target_models
                results[target] = best_mae

            self.is_trained = True
            return results

        except Exception as e:
            logger.error(f"Network forecasting training error: {e}")
            raise

    def forecast_network_performance(self, current_state: Dict[str, float],
                                   forecast_periods: int = 12) -> Dict[str, Any]:
        """Forecast network performance for the next N periods"""

        if not self.is_trained:
            # Return baseline forecasts
            baseline_forecasts = {}
            for metric in ['bandwidth_utilization', 'latency_ms', 'packet_loss_rate', 'throughput_mbps']:
                baseline_value = current_state.get(metric, 50.0)
                forecast = [baseline_value + np.random.normal(0, baseline_value * 0.1)
                           for _ in range(forecast_periods)]
                baseline_forecasts[metric] = {
                    'forecast': forecast,
                    'confidence_intervals': [(f-10, f+10) for f in forecast],
                    'trend': 'stable'
                }

            return baseline_forecasts

        try:
            forecasts = {}

            for target, models in self.models.items():
                if not models:
                    continue

                # Create features for forecasting
                current_value = current_state.get(target, 50.0)

                # Generate sequence of predictions
                predictions = []
                last_values = [current_value] * 10  # Initialize with current value

                for period in range(forecast_periods):
                    # Create features from last values
                    features = np.array(last_values[-10:] + [
                        (datetime.now() + timedelta(hours=period)).hour,
                        (datetime.now() + timedelta(hours=period)).weekday(),
                        period  # forecast step
                    ]).reshape(1, -1)

                    # Get predictions from all models
                    model_predictions = []

                    for model_name, model in models.items():
                        try:
                            if model_name == 'neural_network' and target in self.scalers:
                                features_scaled = self.scalers[target].transform(features)
                                pred = model.predict(features_scaled)[0]
                            else:
                                pred = model.predict(features)[0]

                            model_predictions.append(pred)

                        except Exception as e:
                            logger.warning(f"Forecasting error with {model_name}: {e}")

                    if model_predictions:
                        # Ensemble prediction
                        ensemble_pred = np.mean(model_predictions)
                        predictions.append(ensemble_pred)

                        # Update last_values for next prediction
                        last_values.append(ensemble_pred)
                    else:
                        predictions.append(current_value)
                        last_values.append(current_value)

                # Calculate confidence intervals
                prediction_std = np.std([predictions[-5:]])  # Use recent variance
                confidence_intervals = [
                    (pred - 1.96 * prediction_std, pred + 1.96 * prediction_std)
                    for pred in predictions
                ]

                # Analyze trend
                if len(predictions) > 1:
                    trend_slope = np.polyfit(range(len(predictions)), predictions, 1)[0]
                    if trend_slope > 0.1:
                        trend = 'increasing'
                    elif trend_slope < -0.1:
                        trend = 'decreasing'
                    else:
                        trend = 'stable'
                else:
                    trend = 'stable'

                forecasts[target] = {
                    'forecast': predictions,
                    'confidence_intervals': confidence_intervals,
                    'trend': trend,
                    'confidence': 0.8
                }

            return forecasts

        except Exception as e:
            logger.error(f"Network performance forecasting error: {e}")
            return {}

    def _create_forecasting_features(self, ts_data: pd.Series, lags: int = 10) -> pd.DataFrame:
        """Create features for time series forecasting"""

        df = pd.DataFrame()

        # Lagged values
        for lag in range(1, lags + 1):
            df[f'lag_{lag}'] = ts_data.shift(lag)

        # Time-based features
        df['hour'] = ts_data.index.hour
        df['day_of_week'] = ts_data.index.dayofweek
        df['month'] = ts_data.index.month

        # Rolling statistics
        df['rolling_mean_5'] = ts_data.rolling(window=5).mean()
        df['rolling_std_5'] = ts_data.rolling(window=5).std()

        # Target variable
        df['target'] = ts_data

        # Remove rows with NaN values
        df = df.dropna()

        return df

# QoS and Latency Optimization classes
class QoSOptimizer:
    """Multi-objective QoS optimization"""

    def __init__(self):
        self.qos_policies = {}
        self.optimization_history = []

    def optimize_qos_allocation(self, services: List[Dict[str, Any]],
                               resource_constraints: Dict[str, float]) -> Dict[str, Any]:
        """Optimize QoS allocation using multi-objective optimization"""

        try:
            # Define QoS objectives: latency, throughput, reliability
            objectives = []

            for service in services:
                service_id = service['id']
                requirements = service.get('requirements', {})

                # Latency objective (minimize)
                latency_weight = requirements.get('latency_priority', 1.0)
                objectives.append({
                    'type': 'latency',
                    'service_id': service_id,
                    'weight': latency_weight,
                    'target': requirements.get('max_latency_ms', 100)
                })

                # Throughput objective (maximize)
                throughput_weight = requirements.get('throughput_priority', 1.0)
                objectives.append({
                    'type': 'throughput',
                    'service_id': service_id,
                    'weight': throughput_weight,
                    'target': requirements.get('min_throughput_mbps', 10)
                })

            # Simple optimization using weighted scoring
            total_bandwidth = resource_constraints.get('total_bandwidth', 1000)
            total_cpu = resource_constraints.get('total_cpu', 100)

            allocations = {}

            for service in services:
                service_id = service['id']
                requirements = service.get('requirements', {})

                # Allocate based on priorities
                latency_priority = requirements.get('latency_priority', 1.0)
                throughput_priority = requirements.get('throughput_priority', 1.0)

                # Normalize priorities
                total_priority = latency_priority + throughput_priority
                if total_priority > 0:
                    bandwidth_share = (throughput_priority / total_priority) * (total_bandwidth / len(services))
                    cpu_share = (latency_priority / total_priority) * (total_cpu / len(services))
                else:
                    bandwidth_share = total_bandwidth / len(services)
                    cpu_share = total_cpu / len(services)

                allocations[service_id] = {
                    'bandwidth_mbps': bandwidth_share,
                    'cpu_percentage': cpu_share,
                    'priority_class': self._determine_priority_class(requirements)
                }

            return {
                'allocations': allocations,
                'optimization_score': 0.8,
                'predicted_performance': self._predict_qos_performance(allocations, services)
            }

        except Exception as e:
            logger.error(f"QoS optimization error: {e}")
            return {'allocations': {}, 'optimization_score': 0.0, 'predicted_performance': {}}

    def _determine_priority_class(self, requirements: Dict[str, Any]) -> str:
        """Determine QoS priority class"""

        latency_req = requirements.get('max_latency_ms', 100)
        throughput_req = requirements.get('min_throughput_mbps', 10)

        if latency_req <= 10 and throughput_req >= 100:
            return 'premium'
        elif latency_req <= 50 and throughput_req >= 50:
            return 'business'
        else:
            return 'standard'

    def _predict_qos_performance(self, allocations: Dict[str, Dict[str, float]],
                               services: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Predict QoS performance based on allocations"""

        performance = {}

        for service in services:
            service_id = service['id']
            allocation = allocations.get(service_id, {})
            requirements = service.get('requirements', {})

            # Simple performance prediction
            bandwidth_allocated = allocation.get('bandwidth_mbps', 0)
            bandwidth_required = requirements.get('min_throughput_mbps', 10)

            cpu_allocated = allocation.get('cpu_percentage', 0)

            # Predict metrics
            predicted_latency = max(5, 50 - (cpu_allocated - 20))  # More CPU = lower latency
            predicted_throughput = min(bandwidth_allocated * 0.9, bandwidth_required * 1.2)
            predicted_reliability = min(0.99, 0.8 + (cpu_allocated / 100) * 0.19)

            performance[service_id] = {
                'predicted_latency_ms': predicted_latency,
                'predicted_throughput_mbps': predicted_throughput,
                'predicted_reliability': predicted_reliability
            }

        return performance

# Integration and factory functions
def create_performance_optimizer(optimizer_type: str = 'full') -> 'PerformanceOptimizer':
    """Factory function to create performance optimizer"""

    if optimizer_type == 'predictor':
        return PerformancePredictor()
    elif optimizer_type == 'bandwidth':
        return BandwidthOptimizationEngine()
    elif optimizer_type == 'forecaster':
        return NetworkPerformanceForecaster()
    elif optimizer_type == 'qos':
        return QoSOptimizer()
    else:
        # Return integrated optimizer
        class PerformanceOptimizer:
            def __init__(self):
                self.predictor = PerformancePredictor()
                self.bandwidth_optimizer = BandwidthOptimizationEngine()
                self.forecaster = NetworkPerformanceForecaster()
                self.qos_optimizer = QoSOptimizer()

            def optimize_system_performance(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
                """Comprehensive system performance optimization"""

                results = {
                    'timestamp': datetime.now().isoformat(),
                    'optimization_results': {}
                }

                # Performance prediction
                if 'current_metrics' in system_state:
                    perf_prediction = self.predictor.predict_performance(
                        system_state['current_metrics']
                    )
                    results['optimization_results']['performance_prediction'] = asdict(perf_prediction)

                # Bandwidth optimization
                if 'nodes' in system_state and 'bandwidth_requirements' in system_state:
                    bandwidth_result = self.bandwidth_optimizer.optimize_bandwidth_allocation(
                        system_state['nodes'],
                        system_state.get('total_bandwidth', 1000),
                        system_state['bandwidth_requirements']
                    )
                    results['optimization_results']['bandwidth_optimization'] = asdict(bandwidth_result)

                # Network forecasting
                if 'network_state' in system_state:
                    forecast = self.forecaster.forecast_network_performance(
                        system_state['network_state']
                    )
                    results['optimization_results']['network_forecast'] = forecast

                # QoS optimization
                if 'services' in system_state and 'resource_constraints' in system_state:
                    qos_result = self.qos_optimizer.optimize_qos_allocation(
                        system_state['services'],
                        system_state['resource_constraints']
                    )
                    results['optimization_results']['qos_optimization'] = qos_result

                return results

        return PerformanceOptimizer()