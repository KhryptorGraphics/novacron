"""
Predictive Scaling Engine for NovaCron
Implements ML-driven auto-scaling with advanced prediction algorithms and cost optimization
"""

import sqlite3
import json
import logging
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Guard TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Conv1D, MaxPooling1D
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
    tf = None

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from scipy.optimize import minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    """Scaling action types"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"
    MIGRATE = "migrate"
    CONSOLIDATE = "consolidate"

class ResourceType(Enum):
    """Resource types for scaling"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    VM_COUNT = "vm_count"

class ScalingPolicy(Enum):
    """Scaling policy types"""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    PROACTIVE = "proactive"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    HYBRID = "hybrid"

@dataclass
class ScalingDecision:
    """Data structure for scaling decisions"""
    decision_id: str
    vm_id: str
    resource_type: ResourceType
    scaling_action: ScalingAction
    current_value: float
    target_value: float
    confidence: float
    reasoning: str
    cost_impact: float
    performance_impact: float
    urgency_score: float
    execution_time: datetime
    rollback_plan: Optional[Dict[str, Any]]
    created_at: datetime
    # New fields for migration clarity
    target_node_id: Optional[str] = None
    migration_plan: Optional[Dict[str, Any]] = None

@dataclass
class ResourceForecast:
    """Resource demand forecast"""
    resource_type: ResourceType
    vm_id: str
    forecast_horizon: int  # minutes
    predicted_values: List[float]
    confidence_intervals: List[Tuple[float, float]]
    peak_prediction: float
    peak_time: Optional[datetime]
    valley_prediction: float
    valley_time: Optional[datetime]
    forecast_accuracy: float
    model_used: str

@dataclass
class CostModel:
    """Cost modeling for scaling decisions"""
    cpu_cost_per_unit: float = 0.05
    memory_cost_per_gb: float = 0.02
    storage_cost_per_gb: float = 0.001
    network_cost_per_gb: float = 0.01
    vm_startup_cost: float = 0.10
    migration_cost: float = 0.25
    sla_penalty_cost: float = 10.0

class PredictiveScalingEngine:
    """Advanced predictive scaling engine with ML-driven optimization"""

    def __init__(self, db_path: str = None):
        # Get DB path from environment variable or use default
        if db_path is None:
            db_path = os.environ.get(
                'PREDICTIVE_SCALING_DB',
                os.path.join(os.environ.get('NOVACRON_DATA_DIR', '/var/lib/novacron'), 'predictive_scaling.db')
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
            db_path = '/tmp/predictive_scaling.db'
            logger.info(f"Using fallback database path: {db_path}")

        self.db_path = db_path
        self.scalers = {}
        self.models = {}
        self.cost_model = CostModel()

        # LSTM training status guards
        self.lstm_trained = False

        # Model parameters
        self.prediction_horizon = 60  # minutes
        self.confidence_threshold = 0.7
        self.scale_up_threshold = 0.8
        self.scale_down_threshold = 0.3
        self.cost_optimization_weight = 0.4
        self.performance_weight = 0.6

        # Historical data for model training
        self.training_data = {}
        self.feature_history = {}

        self._init_database()
        self._init_models()

    def _init_database(self):
        """Initialize SQLite database for scaling data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS scaling_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    decision_id TEXT,
                    vm_id TEXT,
                    resource_type TEXT,
                    scaling_action TEXT,
                    current_value REAL,
                    target_value REAL,
                    confidence REAL,
                    reasoning TEXT,
                    cost_impact REAL,
                    performance_impact REAL,
                    urgency_score REAL,
                    execution_time TEXT,
                    rollback_plan TEXT,
                    created_at TEXT,
                    success BOOLEAN,
                    actual_outcome REAL,
                    target_node_id TEXT,
                    migration_plan TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS resource_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    resource_type TEXT,
                    vm_id TEXT,
                    forecast_horizon INTEGER,
                    predicted_values TEXT,
                    confidence_intervals TEXT,
                    peak_prediction REAL,
                    peak_time TEXT,
                    valley_prediction REAL,
                    valley_time TEXT,
                    forecast_accuracy REAL,
                    model_used TEXT,
                    created_at TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS cost_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vm_id TEXT,
                    period_start TEXT,
                    period_end TEXT,
                    total_cost REAL,
                    cpu_cost REAL,
                    memory_cost REAL,
                    storage_cost REAL,
                    network_cost REAL,
                    scaling_cost REAL,
                    sla_penalties REAL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    resource_type TEXT,
                    accuracy_score REAL,
                    mse REAL,
                    mae REAL,
                    r2_score REAL,
                    last_updated TEXT,
                    training_samples INTEGER,
                    UNIQUE(model_name, resource_type)
                )
            """)

    def _init_models(self):
        """Initialize ML models for different resource types with proper scaling

        Resource units:
        - CPU: percentage (0-100%)
        - Memory: percentage (0-100%)
        - Storage: GB or percentage (0-100%)
        - Network: Mbps
        - VM_COUNT: count (integer)
        """
        for resource_type in ResourceType:
            # Use MinMaxScaler for bounded resources, StandardScaler for unbounded
            if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                # CPU/Memory are percentage-based (0-100%)
                self.scalers[resource_type.value] = MinMaxScaler(feature_range=(0, 100))
            elif resource_type == ResourceType.STORAGE:
                # Storage can be percentage or GB - use MinMaxScaler
                self.scalers[resource_type.value] = MinMaxScaler(feature_range=(0, 1000))  # 0-1TB
            elif resource_type == ResourceType.NETWORK:
                # Network bandwidth in Mbps - use MinMaxScaler
                self.scalers[resource_type.value] = MinMaxScaler(feature_range=(0, 10000))  # 0-10Gbps
            else:
                # VM_COUNT and others - use StandardScaler
                self.scalers[resource_type.value] = StandardScaler()

            # Ensemble of models for robust predictions
            models_dict = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gbr': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0)
            }

            # Only add LSTM if TensorFlow is available
            if TF_AVAILABLE:
                models_dict['lstm'] = self._build_lstm_model()
            else:
                logger.warning(f"TensorFlow not available. LSTM model disabled for {resource_type.value}.")

            self.models[resource_type.value] = models_dict

    def _build_lstm_model(self):
        """Build LSTM model for time series prediction"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot build LSTM model.")
            return None

        model = Sequential([
            Input(shape=(60, 5)),  # 60 time steps, 5 features
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, return_sequences=False, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def predict_resource_demand(self, vm_id: str, resource_type: ResourceType,
                              historical_data: pd.DataFrame) -> ResourceForecast:
        """Predict future resource demand using ensemble models

        Args:
            vm_id: Virtual machine identifier
            resource_type: Type of resource (CPU, Memory, etc.)
            historical_data: Historical metrics with columns like 'cpu_usage', 'memory_usage', etc.

        Returns:
            ResourceForecast with predictions in natural units:
            - CPU/Memory: percentage (0-100%)
            - Network: Mbps
            - Storage: GB or percentage
            - VM_COUNT: integer count
        """
        if len(historical_data) < 60:
            logger.warning(f"Insufficient historical data for {vm_id}:{resource_type.value}")
            return self._generate_fallback_forecast(vm_id, resource_type, historical_data)

        # Get the last timestamp from historical data to avoid feature leakage
        last_timestamp = pd.to_datetime(historical_data['timestamp']).iloc[-1]

        # Prepare features
        features = self._prepare_features(historical_data, resource_type)

        # Generate predictions from all models
        predictions = {}
        confidences = {}

        for model_name, model in self.models[resource_type.value].items():
            try:
                if model_name == 'lstm':
                    pred, conf = self._predict_lstm(model, features, last_timestamp)
                else:
                    pred, conf = self._predict_sklearn(model, features)

                predictions[model_name] = pred
                confidences[model_name] = conf

            except Exception as e:
                logger.error(f"Prediction failed for {model_name}: {str(e)}")
                continue

        if not predictions:
            return self._generate_fallback_forecast(vm_id, resource_type, historical_data)

        # Ensemble predictions with confidence weighting
        ensemble_pred = self._ensemble_predictions(predictions, confidences)

        # Apply inverse scaling to get predictions in natural units
        if resource_type.value in self.scalers:
            scaler = self.scalers[resource_type.value]
            # Reshape for inverse transform
            scaled_pred = ensemble_pred.reshape(-1, 1)
            try:
                # Only apply inverse transform if scaler has been fitted
                if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                    ensemble_pred = scaler.inverse_transform(scaled_pred).flatten()
            except (ValueError, AttributeError):
                # If inverse transform fails, keep original predictions
                pass

        confidence_intervals = self._calculate_confidence_intervals(ensemble_pred, confidences, resource_type)

        # Identify peaks and valleys
        peak_idx = int(np.argmax(ensemble_pred))
        valley_idx = int(np.argmin(ensemble_pred))

        forecast = ResourceForecast(
            resource_type=resource_type,
            vm_id=vm_id,
            forecast_horizon=self.prediction_horizon,
            predicted_values=ensemble_pred.tolist(),
            confidence_intervals=confidence_intervals,
            peak_prediction=float(ensemble_pred[peak_idx]),
            peak_time=last_timestamp + timedelta(minutes=peak_idx),
            valley_prediction=float(ensemble_pred[valley_idx]),
            valley_time=last_timestamp + timedelta(minutes=valley_idx),
            forecast_accuracy=np.mean(list(confidences.values())),
            model_used="ensemble"
        )

        # Store forecast
        self._store_forecast(forecast)
        return forecast

    def _prepare_features(self, data: pd.DataFrame, resource_type: ResourceType) -> np.ndarray:
        """Prepare feature matrix for ML models"""
        # Resource utilization
        resource_col = f"{resource_type.value}_usage"
        if resource_col not in data.columns:
            resource_col = "cpu_usage"  # fallback

        # Create features
        features = []

        # Historical values (lookback window)
        lookback = 60
        resource_values = data[resource_col].fillna(0).values

        if len(resource_values) < lookback:
            # Pad with mean if insufficient data
            mean_val = np.mean(resource_values) if len(resource_values) > 0 else 0
            padded = np.full(lookback, mean_val)
            padded[-len(resource_values):] = resource_values
            resource_values = padded

        # Time-based features
        timestamps = pd.to_datetime(data['timestamp'])
        hours = timestamps.dt.hour.values
        days_of_week = timestamps.dt.dayofweek.values

        # Pad time features if necessary
        if len(hours) < lookback:
            mean_hour = np.mean(hours) if len(hours) > 0 else 12
            hours = np.full(lookback, mean_hour)
        if len(days_of_week) < lookback:
            mean_dow = np.mean(days_of_week) if len(days_of_week) > 0 else 1
            days_of_week = np.full(lookback, mean_dow)

        # Create feature matrix for each time step
        for i in range(lookback):
            step_features = [
                resource_values[i],
                np.sin(2 * np.pi * hours[i] / 24),  # Hour cyclical
                np.cos(2 * np.pi * hours[i] / 24),
                np.sin(2 * np.pi * days_of_week[i] / 7),  # Day cyclical
                np.cos(2 * np.pi * days_of_week[i] / 7)
            ]
            features.append(step_features)

        return np.array(features).reshape(1, lookback, 5)

    def _predict_lstm(self, model, features: np.ndarray, last_timestamp: datetime) -> Tuple[np.ndarray, float]:
        """Generate LSTM predictions"""
        if not TF_AVAILABLE or model is None:
            logger.warning("TensorFlow not available or LSTM model is None. Using fallback prediction.")
            # Return a simple fallback prediction
            baseline = 0.5
            predictions = [baseline + i * 0.001 for i in range(self.prediction_horizon)]
            return np.array(predictions), 0.3

        # Check if LSTM model has been trained
        if not self.lstm_trained:
            logger.warning("LSTM model not trained yet. Using fallback prediction.")
            # Return a simple baseline prediction when untrained
            baseline = np.mean(features.flatten()) if len(features.flatten()) > 0 else 0.5
            predictions = [baseline + i * 0.001 for i in range(self.prediction_horizon)]
            return np.array(predictions), 0.2  # Lower confidence for untrained model

        # Predict next values
        predictions = []
        current_input = features.copy()

        for step in range(self.prediction_horizon):
            pred = model.predict(current_input, verbose=0)[0, 0]
            predictions.append(pred)

            # Update input for next prediction (sliding window) with proper cyclical features
            # Step forward from last known timestamp to avoid feature leakage
            ts = last_timestamp + timedelta(minutes=step+1)
            hour = ts.hour
            dow = ts.weekday()
            cyc = [np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24),
                   np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)]
            new_step = np.hstack([[pred], cyc])
            new_input = np.vstack([current_input[0,1:,:], new_step])
            current_input = new_input.reshape(1, -1, 5)

        # Calculate confidence based on prediction variance
        pred_variance = np.var(predictions)
        confidence = max(0.1, 1.0 - min(pred_variance, 1.0))

        return np.array(predictions), confidence

    def _predict_sklearn(self, model, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """Generate sklearn model predictions

        Returns predictions in the same scale as training data.
        Units depend on resource type:
        - CPU/Memory: percentage (0-100%)
        - Network: Mbps
        - IOPS: operations per second
        """
        # Flatten features for sklearn models
        X = features.reshape(features.shape[0], -1)

        # Simple approach: predict one step ahead and repeat
        # In practice, this would be more sophisticated
        try:
            single_pred = model.predict(X)[0]
        except:
            # Model not trained, use conservative fallback based on resource type
            single_pred = 0.5  # Will be scaled appropriately by caller

        # Generate predictions with slight variation
        predictions = []
        base_pred = single_pred

        for i in range(self.prediction_horizon):
            # Add slight trend and noise
            trend = i * 0.001
            noise = np.random.normal(0, 0.02)
            # Remove hard clipping - let natural scaling handle bounds
            pred = base_pred + trend + noise
            predictions.append(pred)

        confidence = 0.6  # Moderate confidence for simple models
        return np.array(predictions), confidence

    def _ensemble_predictions(self, predictions: Dict[str, np.ndarray],
                            confidences: Dict[str, float]) -> np.ndarray:
        """Combine predictions using confidence weighting"""
        if not predictions:
            return np.zeros(self.prediction_horizon)

        total_weight = sum(confidences.values())
        if total_weight == 0:
            # Equal weighting if no confidence information
            weights = {name: 1.0/len(predictions) for name in predictions.keys()}
        else:
            weights = {name: conf/total_weight for name, conf in confidences.items()}

        # Weighted ensemble
        ensemble = np.zeros(self.prediction_horizon)
        for name, pred in predictions.items():
            ensemble += weights[name] * pred

        return ensemble

    def _calculate_confidence_intervals(self, predictions: np.ndarray,
                                     confidences: Dict[str, float], resource_type: ResourceType) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions in natural units

        Args:
            predictions: Predictions in natural units (%, Mbps, GB, count)
            confidences: Model confidence scores
            resource_type: Type of resource for appropriate bounds
        """
        avg_confidence = np.mean(list(confidences.values()))

        # Calculate standard deviation based on prediction range and confidence
        pred_range = np.max(predictions) - np.min(predictions)
        std_dev = max(0.01, pred_range * 0.1 * (1 - avg_confidence))

        intervals = []
        for pred in predictions:
            lower = pred - 2 * std_dev
            upper = pred + 2 * std_dev

            # Apply resource-specific bounds
            if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                # Percentage-based resources: 0-100%
                lower = max(0, lower)
                upper = min(100, upper)
            elif resource_type == ResourceType.NETWORK:
                # Network bandwidth: non-negative Mbps
                lower = max(0, lower)
                # No upper bound for network - let it be unconstrained
            elif resource_type == ResourceType.VM_COUNT:
                # VM count: non-negative integers
                lower = max(0, int(lower))
                upper = int(upper)
            else:
                # Other resources: non-negative
                lower = max(0, lower)

            intervals.append((lower, upper))

        return intervals

    def _generate_fallback_forecast(self, vm_id: str, resource_type: ResourceType,
                                  data: pd.DataFrame) -> ResourceForecast:
        """Generate simple fallback forecast when ML models fail

        Returns predictions in natural units based on resource type.
        """
        # Get the last timestamp from historical data, or use current time if no data
        if len(data) > 0 and 'timestamp' in data.columns:
            last_timestamp = pd.to_datetime(data['timestamp']).iloc[-1]
        else:
            last_timestamp = datetime.now()

        resource_col = f"{resource_type.value}_usage"
        if resource_col not in data.columns or len(data) == 0:
            # No data available, use conservative estimates in natural units
            if resource_type in [ResourceType.CPU, ResourceType.MEMORY]:
                baseline = 50.0  # 50% utilization
            elif resource_type == ResourceType.NETWORK:
                baseline = 100.0  # 100 Mbps
            elif resource_type == ResourceType.VM_COUNT:
                baseline = 1.0  # 1 VM
            else:
                baseline = 50.0  # Generic baseline
        else:
            baseline = data[resource_col].fillna(50.0).mean()

        # Simple flat prediction with slight upward trend in natural units
        trend_factor = 0.001 * baseline  # Trend proportional to baseline
        predictions = [baseline + i * trend_factor for i in range(self.prediction_horizon)]

        # Calculate confidence intervals in natural units
        interval_width = baseline * 0.2  # 20% of baseline value
        confidence_intervals = [(p - interval_width, p + interval_width) for p in predictions]

        return ResourceForecast(
            resource_type=resource_type,
            vm_id=vm_id,
            forecast_horizon=self.prediction_horizon,
            predicted_values=predictions,
            confidence_intervals=confidence_intervals,
            peak_prediction=max(predictions),
            peak_time=last_timestamp + timedelta(minutes=int(predictions.index(max(predictions)))),
            valley_prediction=min(predictions),
            valley_time=last_timestamp + timedelta(minutes=int(predictions.index(min(predictions)))),
            forecast_accuracy=0.3,  # Low confidence for fallback
            model_used="fallback"
        )

    def make_scaling_decision(self, vm_id: str, forecasts: Dict[ResourceType, ResourceForecast],
                            current_resources: Dict[ResourceType, float]) -> List[ScalingDecision]:
        """Make intelligent scaling decisions based on forecasts"""
        decisions = []

        for resource_type, forecast in forecasts.items():
            current_value = current_resources.get(resource_type, 0.5)

            # Analyze forecast for scaling opportunities
            peak_util = forecast.peak_prediction
            avg_util = np.mean(forecast.predicted_values)

            # Determine scaling action
            action, target, reasoning, target_node_id, migration_plan = self._determine_scaling_action(
                resource_type, current_value, peak_util, avg_util, forecast
            )

            if action == ScalingAction.NO_ACTION:
                continue

            # Calculate impacts
            cost_impact = self._calculate_cost_impact(resource_type, current_value, target, action)
            perf_impact = self._calculate_performance_impact(resource_type, current_value, target)
            urgency = self._calculate_urgency_score(forecast, current_value)

            # Create decision
            decision = ScalingDecision(
                decision_id=f"{vm_id}_{resource_type.value}_{datetime.now().timestamp()}",
                vm_id=vm_id,
                resource_type=resource_type,
                scaling_action=action,
                current_value=current_value,
                target_value=target,
                confidence=forecast.forecast_accuracy,
                reasoning=reasoning,
                cost_impact=cost_impact,
                performance_impact=perf_impact,
                urgency_score=urgency,
                execution_time=self._calculate_optimal_execution_time(forecast),
                rollback_plan=self._create_rollback_plan(resource_type, current_value),
                created_at=datetime.now(),
                target_node_id=target_node_id,
                migration_plan=migration_plan
            )

            decisions.append(decision)

        # Optimize decision set for cost-performance balance
        optimized_decisions = self._optimize_decision_set(decisions)

        # Store decisions
        for decision in optimized_decisions:
            self._store_scaling_decision(decision)

        return optimized_decisions

    def _determine_scaling_action(self, resource_type: ResourceType, current: float,
                                peak: float, avg: float, forecast: ResourceForecast) -> Tuple[ScalingAction, float, str, Optional[str], Optional[Dict[str, Any]]]:
        """Determine appropriate scaling action

        Returns:
            Tuple of (action, target_value, reasoning, target_node_id, migration_plan)
        """
        # Migration consideration for CPU optimization (check first to prefer migration over scaling)
        if (resource_type == ResourceType.CPU and
            avg > 0.7 and  # High average utilization
            current > 0.75 and  # High current utilization
            (forecast is None or forecast.forecast_accuracy > 0.8)):  # High confidence forecast
            # Calculate expected utilization after migration (assume 30% improvement)
            expected_util_after_migration = current * 0.7
            target_node_id = self._select_migration_target()
            # Use VM ID from forecast if available, otherwise use generic ID
            vm_id = forecast.vm_id if forecast else "unknown-vm"
            migration_plan = self._create_migration_plan(vm_id, target_node_id, current, expected_util_after_migration)
            return (ScalingAction.MIGRATE, expected_util_after_migration,
                   f"High CPU utilization (avg: {avg:.2f}, current: {current:.2f}) suggests migration for resource optimization",
                   target_node_id, migration_plan)

        # Scale up conditions
        if peak > self.scale_up_threshold:
            target = min(1.0, peak * 1.2)  # 20% buffer above peak
            return ScalingAction.SCALE_UP, target, f"Peak utilization {peak:.2f} exceeds threshold {self.scale_up_threshold}", None, None

        # Scale down conditions
        if avg < self.scale_down_threshold and peak < self.scale_down_threshold * 1.5:
            target = max(0.1, avg * 1.3)  # 30% buffer above average
            return ScalingAction.SCALE_DOWN, target, f"Average utilization {avg:.2f} below threshold {self.scale_down_threshold}", None, None

        return ScalingAction.NO_ACTION, current, "No scaling action required", None, None

    def _calculate_cost_impact(self, resource_type: ResourceType, current: float,
                             target: float, action: ScalingAction) -> float:
        """Calculate cost impact of scaling decision"""
        if action == ScalingAction.NO_ACTION:
            return 0.0

        resource_delta = target - current

        cost_per_unit = {
            ResourceType.CPU: self.cost_model.cpu_cost_per_unit,
            ResourceType.MEMORY: self.cost_model.memory_cost_per_gb,
            ResourceType.STORAGE: self.cost_model.storage_cost_per_gb,
            ResourceType.NETWORK: self.cost_model.network_cost_per_gb
        }

        base_cost = resource_delta * cost_per_unit.get(resource_type, 0.05)

        # Add action-specific costs
        if action == ScalingAction.MIGRATE:
            base_cost += self.cost_model.migration_cost
        elif action in [ScalingAction.SCALE_OUT]:
            base_cost += self.cost_model.vm_startup_cost

        return base_cost

    def _calculate_performance_impact(self, resource_type: ResourceType,
                                    current: float, target: float) -> float:
        """Calculate performance impact (positive = improvement)"""
        if target > current:
            # Scaling up improves performance
            improvement = (target - current) * 10  # Normalized performance gain
            return min(improvement, 5.0)  # Cap at 5x improvement
        else:
            # Scaling down may hurt performance
            degradation = (current - target) * -5  # Negative impact
            return max(degradation, -2.0)  # Cap degradation

    def _calculate_urgency_score(self, forecast: ResourceForecast, current: float) -> float:
        """Calculate urgency score based on forecast and current state"""
        # Time to threshold breach
        threshold = 0.9
        time_to_breach = None

        for i, pred in enumerate(forecast.predicted_values):
            if pred > threshold:
                time_to_breach = i
                break

        if time_to_breach is None:
            return 0.1  # Low urgency

        # Urgency inversely related to time
        max_urgency_time = 15  # minutes
        urgency = max(0.1, 1.0 - (time_to_breach / max_urgency_time))

        # Boost urgency if current utilization is already high
        if current > 0.8:
            urgency *= 1.5

        return min(urgency, 1.0)

    def _calculate_optimal_execution_time(self, forecast: ResourceForecast) -> datetime:
        """Calculate optimal time to execute scaling action"""
        # Find the best time to scale based on predicted utilization
        min_util_idx = 0
        min_util = float('inf')

        for i, pred in enumerate(forecast.predicted_values[:30]):  # Check next 30 minutes
            if pred < min_util:
                min_util = pred
                min_util_idx = i

        # Use the forecast's valley_time if available, otherwise calculate from current time
        if forecast.valley_time and min_util_idx == forecast.predicted_values.index(forecast.valley_prediction):
            return forecast.valley_time
        else:
            # Calculate based on current time - this is acceptable as it's for execution scheduling
            return datetime.now() + timedelta(minutes=min_util_idx)

    def _create_rollback_plan(self, resource_type: ResourceType, original_value: float) -> Dict[str, Any]:
        """Create rollback plan for scaling decision"""
        return {
            'original_value': original_value,
            'rollback_action': 'restore_original',
            'rollback_timeout_minutes': 30,
            'rollback_conditions': [
                'performance_degradation > 20%',
                'error_rate > 5%',
                'user_initiated'
            ]
        }

    def _optimize_decision_set(self, decisions: List[ScalingDecision]) -> List[ScalingDecision]:
        """Optimize set of scaling decisions for cost-performance balance"""
        if not decisions:
            return decisions

        # Simple optimization: prioritize by urgency and cost-performance ratio
        def optimization_score(decision):
            cost_perf_ratio = decision.performance_impact / (abs(decision.cost_impact) + 0.01)
            return decision.urgency_score * 0.4 + cost_perf_ratio * 0.6

        # Sort by optimization score
        optimized = sorted(decisions, key=optimization_score, reverse=True)

        # Remove conflicting decisions
        resource_types_seen = set()
        final_decisions = []

        for decision in optimized:
            if decision.resource_type not in resource_types_seen:
                final_decisions.append(decision)
                resource_types_seen.add(decision.resource_type)

        return final_decisions

    def _store_forecast(self, forecast: ResourceForecast):
        """Store resource forecast in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO resource_forecasts
                (resource_type, vm_id, forecast_horizon, predicted_values, confidence_intervals,
                 peak_prediction, peak_time, valley_prediction, valley_time, forecast_accuracy,
                 model_used, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                forecast.resource_type.value, forecast.vm_id, forecast.forecast_horizon,
                json.dumps(forecast.predicted_values), json.dumps(forecast.confidence_intervals),
                forecast.peak_prediction,
                forecast.peak_time.isoformat() if forecast.peak_time else None,
                forecast.valley_prediction,
                forecast.valley_time.isoformat() if forecast.valley_time else None,
                forecast.forecast_accuracy, forecast.model_used,
                datetime.now().isoformat()
            ))

    def _store_scaling_decision(self, decision: ScalingDecision):
        """Store scaling decision in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO scaling_history
                (decision_id, vm_id, resource_type, scaling_action, current_value, target_value,
                 confidence, reasoning, cost_impact, performance_impact, urgency_score,
                 execution_time, rollback_plan, created_at, target_node_id, migration_plan)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.decision_id, decision.vm_id, decision.resource_type.value,
                decision.scaling_action.value, decision.current_value, decision.target_value,
                decision.confidence, decision.reasoning, decision.cost_impact,
                decision.performance_impact, decision.urgency_score,
                decision.execution_time.isoformat(),
                json.dumps(decision.rollback_plan) if decision.rollback_plan else None,
                decision.created_at.isoformat(),
                decision.target_node_id,
                json.dumps(decision.migration_plan) if decision.migration_plan else None
            ))

    def train_models(self, historical_data: Dict[str, pd.DataFrame], retrain: bool = False):
        """Train predictive models on historical data"""
        logger.info("Training predictive scaling models...")

        for vm_id, data in historical_data.items():
            if len(data) < 100:  # Need sufficient data for training
                logger.warning(f"Insufficient data for training VM {vm_id}")
                continue

            for resource_type in ResourceType:
                if resource_type == ResourceType.VM_COUNT:
                    continue  # Skip VM count for individual VM training

                resource_col = f"{resource_type.value}_usage"
                if resource_col not in data.columns:
                    continue

                try:
                    # Prepare training data
                    X, y = self._prepare_training_data(data, resource_type)

                    if len(X) < 50:
                        continue

                    # Train sklearn models
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )

                    # Scale features
                    scaler = self.scalers[resource_type.value]
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Train each model
                    for model_name, model in self.models[resource_type.value].items():
                        if model_name == 'lstm':
                            # Train LSTM model with different data format
                            if TF_AVAILABLE and model is not None:
                                try:
                                    # Prepare LSTM training data (3D format)
                                    X_lstm = self._prepare_lstm_training_data(data, resource_type)
                                    if X_lstm is not None and len(X_lstm) > 50:
                                        # Simple training for LSTM (this should be enhanced)
                                        y_lstm = X_lstm[:, -1, 0]  # Use last time step as target
                                        X_lstm_input = X_lstm[:, :-1, :]  # Use all but last time step as input

                                        if len(X_lstm_input) > 10:
                                            X_lstm_train, X_lstm_test, y_lstm_train, y_lstm_test = train_test_split(
                                                X_lstm_input, y_lstm, test_size=0.2, random_state=42
                                            )

                                            model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32,
                                                     validation_data=(X_lstm_test, y_lstm_test), verbose=0)

                                            # Set LSTM as trained
                                            self.lstm_trained = True
                                            logger.info(f"LSTM model trained successfully for {resource_type.value}")
                                except Exception as e:
                                    logger.error(f"LSTM training failed for {resource_type.value}: {str(e)}")
                            continue

                        model.fit(X_train_scaled, y_train)

                        # Evaluate model
                        y_pred = model.predict(X_test_scaled)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)

                        # Store performance metrics
                        self._store_model_performance(
                            model_name, resource_type, mse, mae, r2, len(X_train)
                        )

                        logger.info(f"Trained {model_name} for {resource_type.value}: "
                                  f"MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

                except Exception as e:
                    logger.error(f"Training failed for {vm_id}:{resource_type.value}: {str(e)}")

    def _prepare_training_data(self, data: pd.DataFrame, resource_type: ResourceType) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        resource_col = f"{resource_type.value}_usage"

        # Create feature windows and targets
        window_size = 60
        X, y = [], []

        for i in range(window_size, len(data)):
            # Features: historical values and time features
            window_data = data.iloc[i-window_size:i]

            # Resource values
            resource_values = window_data[resource_col].fillna(0).values

            # Time features
            timestamps = pd.to_datetime(window_data['timestamp'])
            hour_sin = np.sin(2 * np.pi * timestamps.dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * timestamps.dt.hour / 24)
            dow_sin = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
            dow_cos = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)

            # Combine features (flattened for sklearn)
            features = np.concatenate([
                resource_values,
                hour_sin.values,
                hour_cos.values,
                dow_sin.values,
                dow_cos.values
            ])

            X.append(features)
            y.append(data.iloc[i][resource_col])

        return np.array(X), np.array(y)

    def _prepare_lstm_training_data(self, data: pd.DataFrame, resource_type: ResourceType) -> Optional[np.ndarray]:
        """Prepare 3D training data for LSTM models"""
        resource_col = f"{resource_type.value}_usage"

        if resource_col not in data.columns or len(data) < 120:  # Need sufficient data
            return None

        # Create sequences for LSTM
        sequence_length = 60
        X = []

        for i in range(sequence_length, len(data) - 1):
            # Get sequence window
            window_data = data.iloc[i-sequence_length:i+1]

            # Resource values
            resource_values = window_data[resource_col].fillna(0).values

            # Time features
            timestamps = pd.to_datetime(window_data['timestamp'])
            hour_sin = np.sin(2 * np.pi * timestamps.dt.hour / 24)
            hour_cos = np.cos(2 * np.pi * timestamps.dt.hour / 24)
            dow_sin = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
            dow_cos = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)

            # Create feature matrix for this sequence
            sequence_features = []
            for j in range(len(resource_values)):
                step_features = [
                    resource_values[j],
                    hour_sin.iloc[j],
                    hour_cos.iloc[j],
                    dow_sin.iloc[j],
                    dow_cos.iloc[j]
                ]
                sequence_features.append(step_features)

            X.append(sequence_features)

        if len(X) == 0:
            return None

        return np.array(X)

    def _select_migration_target(self) -> str:
        """Select optimal target node for VM migration

        Returns:
            Target node ID with sufficient capacity and optimal characteristics
        """
        # This is a simplified implementation - in practice, this would:
        # 1. Query cluster state for available nodes
        # 2. Analyze resource capacity and utilization patterns
        # 3. Consider network topology and latency
        # 4. Apply placement policies and constraints

        # For now, return a placeholder that indicates the selection logic
        available_nodes = ["node-01", "node-02", "node-03", "node-04"]

        # In reality, this would analyze:
        # - Current resource utilization per node
        # - Available capacity
        # - Network proximity
        # - Workload affinity rules
        # - Power consumption and efficiency

        # Return the "best" node (simplified selection)
        import random
        return random.choice(available_nodes)

    def _create_migration_plan(self, vm_id: str, target_node_id: str,
                              current_util: float, expected_util: float) -> Dict[str, Any]:
        """Create detailed migration execution plan

        Args:
            vm_id: VM to be migrated
            target_node_id: Destination node
            current_util: Current resource utilization
            expected_util: Expected utilization after migration

        Returns:
            Detailed migration plan with execution steps and validation criteria
        """
        return {
            "migration_type": "live_migration",
            "source_vm_id": vm_id,
            "target_node_id": target_node_id,
            "expected_utilization_improvement": {
                "current_utilization": current_util,
                "expected_utilization": expected_util,
                "improvement_percentage": ((current_util - expected_util) / current_util) * 100
            },
            "execution_steps": [
                {
                    "step": 1,
                    "action": "pre_migration_validation",
                    "description": "Validate target node capacity and network connectivity",
                    "timeout_seconds": 30
                },
                {
                    "step": 2,
                    "action": "initiate_live_migration",
                    "description": f"Start live migration of {vm_id} to {target_node_id}",
                    "timeout_seconds": 300
                },
                {
                    "step": 3,
                    "action": "monitor_migration_progress",
                    "description": "Monitor memory transfer and VM state synchronization",
                    "timeout_seconds": 600
                },
                {
                    "step": 4,
                    "action": "finalize_migration",
                    "description": "Complete migration and update cluster state",
                    "timeout_seconds": 60
                },
                {
                    "step": 5,
                    "action": "post_migration_validation",
                    "description": "Verify VM functionality and performance improvement",
                    "timeout_seconds": 120
                }
            ],
            "success_criteria": [
                "VM remains responsive throughout migration",
                "Migration completes within 10 minutes",
                "Resource utilization improves by at least 20%",
                "No data loss or corruption detected"
            ],
            "rollback_conditions": [
                "Migration fails to complete within timeout",
                "VM becomes unresponsive",
                "Target node resources become insufficient",
                "Network connectivity issues detected"
            ],
            "resource_requirements": {
                "target_node_cpu_available": f">{current_util * 1.1}%",
                "target_node_memory_available": f">2GB + VM_size",
                "network_bandwidth_required": "100Mbps minimum"
            },
            "estimated_downtime_seconds": 2,
            "estimated_completion_minutes": 8
        }

    def _store_model_performance(self, model_name: str, resource_type: ResourceType,
                               mse: float, mae: float, r2: float, samples: int):
        """Store model performance metrics"""
        with sqlite3.connect(self.db_path) as conn:
            # Calculate accuracy score (1 - normalized MAE)
            accuracy = max(0, 1 - mae)

            conn.execute("""
                INSERT INTO model_performance
                (model_name, resource_type, accuracy_score, mse, mae, r2_score, last_updated, training_samples)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(model_name, resource_type) DO UPDATE SET
                    accuracy_score=excluded.accuracy_score,
                    mse=excluded.mse,
                    mae=excluded.mae,
                    r2_score=excluded.r2_score,
                    last_updated=excluded.last_updated,
                    training_samples=excluded.training_samples
            """, (
                model_name, resource_type.value, accuracy, mse, mae, r2,
                datetime.now().isoformat(), samples
            ))

    def get_scaling_recommendations(self, vm_id: str,
                                  lookback_hours: int = 24) -> Dict[str, Any]:
        """Get scaling recommendations for a VM"""
        # This would integrate with the workload pattern recognition
        # and other NovaCron components for comprehensive recommendations

        recommendations = {
            'vm_id': vm_id,
            'timestamp': datetime.now().isoformat(),
            'recommendations': [],
            'cost_analysis': {},
            'performance_projection': {},
            'confidence_score': 0.0
        }

        return recommendations


# Legacy wrapper for backward compatibility
class AutoScaler(PredictiveScalingEngine):
    """Legacy wrapper maintaining API compatibility"""

    def __init__(self, db_path: str = None):
        # Initialize parent with proper db path handling
        super().__init__(db_path=db_path)

    def scale_decision(self, vm_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Legacy scaling decision method"""
        # Convert metrics to forecasts (simplified)
        forecasts = {}
        current_resources = {}

        for resource, value in metrics.items():
            if resource.endswith('_usage'):
                resource_type = ResourceType(resource.replace('_usage', ''))

                # Use current time as the base for legacy mode - this is acceptable for legacy compatibility
                base_time = datetime.now()

                # Create simple forecast
                forecast = ResourceForecast(
                    resource_type=resource_type,
                    vm_id=vm_id,
                    forecast_horizon=60,
                    predicted_values=[value] * 60,  # Flat prediction
                    confidence_intervals=[(value-0.1, value+0.1)] * 60,
                    peak_prediction=value,
                    peak_time=base_time + timedelta(minutes=30),
                    valley_prediction=value,
                    valley_time=base_time + timedelta(minutes=60),
                    forecast_accuracy=0.5,
                    model_used="legacy"
                )

                forecasts[resource_type] = forecast
                current_resources[resource_type] = value

        decisions = self.make_scaling_decision(vm_id, forecasts, current_resources)

        if decisions:
            decision = decisions[0]
            result = {
                'action': decision.scaling_action.value,
                'resource': decision.resource_type.value,
                'current': decision.current_value,
                'target': decision.target_value,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning
            }

            # Add migration-specific information if this is a migration decision
            if decision.scaling_action == ScalingAction.MIGRATE and decision.migration_plan:
                result.update({
                    'target_node_id': decision.target_node_id,
                    'migration_details': {
                        'migration_type': decision.migration_plan.get('migration_type'),
                        'expected_improvement': decision.migration_plan.get('expected_utilization_improvement'),
                        'execution_steps': decision.migration_plan.get('execution_steps'),
                        'success_criteria': decision.migration_plan.get('success_criteria'),
                        'estimated_completion_minutes': decision.migration_plan.get('estimated_completion_minutes')
                    }
                })

            return result

        return {'action': 'no_action', 'reasoning': 'No scaling required'}