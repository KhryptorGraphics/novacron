"""
Advanced Workload Pattern Recognition Engine for NovaCron
Implements sophisticated pattern analysis, anomaly detection, and workload classification
"""

import sqlite3
import json
import logging
import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkloadType(Enum):
    """Workload type classifications"""
    CPU_INTENSIVE = "cpu_intensive"
    MEMORY_INTENSIVE = "memory_intensive"
    IO_INTENSIVE = "io_intensive"
    NETWORK_INTENSIVE = "network_intensive"
    BATCH_PROCESSING = "batch_processing"
    REAL_TIME = "real_time"
    INTERACTIVE = "interactive"
    BACKGROUND = "background"
    PERIODIC = "periodic"
    BURSTY = "bursty"
    STEADY_STATE = "steady_state"
    UNKNOWN = "unknown"

class PatternType(Enum):
    """Pattern type classifications"""
    SEASONAL = "seasonal"
    TRENDING = "trending"
    CYCLIC = "cyclic"
    IRREGULAR = "irregular"
    SPIKE = "spike"
    VALLEY = "valley"
    PLATEAU = "plateau"
    EXPONENTIAL_GROWTH = "exponential_growth"
    EXPONENTIAL_DECAY = "exponential_decay"
    BURSTY = "bursty"
    STEADY_STATE = "steady_state"

@dataclass
class WorkloadPattern:
    """Data structure for workload patterns

    Units and Semantics:
    - frequency: cycles per hour (float, e.g., 0.5 = once every 2 hours)
    - seasonal_period: period in hours (int, e.g., 24 = daily cycle, 168 = weekly cycle)
    - duration: total duration in minutes (int)
    - amplitude: standard deviation of primary metric (float)

    Assumptions:
    - Input data sampling rate: 1 sample per hour
    - Time-based periods expressed in hours
    """
    pattern_id: str
    workload_type: WorkloadType
    pattern_type: PatternType
    confidence: float
    characteristics: Dict[str, Any]
    frequency: Optional[float]  # cycles per hour
    amplitude: Optional[float]  # std deviation of primary metric
    duration: Optional[int]     # duration in minutes
    seasonal_period: Optional[int]  # period in hours
    trend_direction: Optional[str]
    created_at: datetime
    last_seen: datetime
    occurrence_count: int

@dataclass
class WorkloadFeatures:
    """Feature vector for workload analysis

    Units and Semantics:
    - seasonality_score: confidence of seasonal pattern (0-1 float)
    - seasonal_period_samples: dominant period in data samples (int, e.g., 24 samples = 24 hours if 1 sample/hour)
    - duration_minutes: total observation duration in minutes
    - trend_score: trend strength (-1 to 1, negative=decreasing, positive=increasing)
    - burstiness: variability measure (-1 to 1, higher=more bursty)
    """
    cpu_mean: float
    cpu_std: float
    cpu_max: float
    memory_mean: float
    memory_std: float
    memory_max: float
    io_mean: float
    io_std: float
    io_max: float
    network_mean: float
    network_std: float
    network_max: float
    duration_minutes: int
    time_of_day: int
    day_of_week: int
    burstiness: float
    seasonality_score: float        # confidence of seasonal pattern (0-1)
    seasonal_period_samples: Optional[int]  # dominant period in data samples
    trend_score: float

class WorkloadPatternRecognizer:
    """Advanced workload pattern recognition using ML and statistical analysis"""

    def __init__(self, db_path: str = None):
        # Get DB path from environment variable or use default
        if db_path is None:
            db_path = os.environ.get(
                'WORKLOAD_PATTERNS_DB',
                os.path.join(os.environ.get('NOVACRON_DATA_DIR', '/var/lib/novacron'), 'workload_patterns.db')
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
            db_path = '/tmp/workload_patterns.db'
            logger.info(f"Using fallback database path: {db_path}")

        self.db_path = db_path
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)
        self.clustering_model = None
        self.classification_model = None
        self.lstm_model = None
        self.pattern_cache = {}
        self.feature_importance = {}

        # LSTM training status guard
        self.lstm_trained = False

        # Pattern detection parameters
        self.seasonality_threshold = 0.7
        self.trend_threshold = 0.6
        self.spike_threshold = 3.0
        self.burstiness_threshold = 2.0

        self._init_database()
        self._init_models()

    def _init_database(self):
        """Initialize SQLite database for pattern storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workload_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    workload_type TEXT,
                    pattern_type TEXT,
                    confidence REAL,
                    characteristics TEXT,
                    frequency REAL,
                    amplitude REAL,
                    duration INTEGER,
                    seasonal_period INTEGER,
                    trend_direction TEXT,
                    created_at TEXT,
                    last_seen TEXT,
                    occurrence_count INTEGER
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS workload_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    vm_id TEXT,
                    features TEXT,
                    pattern_id TEXT,
                    classification_confidence REAL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS pattern_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_id TEXT,
                    timestamp TEXT,
                    evolution_data TEXT,
                    drift_score REAL
                )
            """)

    def _init_models(self):
        """Initialize ML models for pattern recognition"""
        # Clustering model for pattern discovery
        self.clustering_model = KMeans(n_clusters=8, random_state=42)

        # Classification model for pattern matching
        self.classification_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )

        # LSTM model for temporal pattern recognition (lazy loading)
        # Check environment flag for LSTM training
        self.enable_wpr_lstm = os.getenv('ENABLE_WPR_LSTM', 'false').lower() == 'true'

        if TF_AVAILABLE and self.enable_wpr_lstm:
            self._build_lstm_model()
        else:
            self.lstm_model = None
            if not TF_AVAILABLE:
                logger.warning("TensorFlow not available. LSTM model disabled.")
            else:
                logger.info("LSTM model disabled by configuration. Set ENABLE_WPR_LSTM=true to enable.")

    def _build_lstm_model(self):
        """Build LSTM model for temporal pattern analysis"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot build LSTM model.")
            return

        model = Sequential([
            Input(shape=(60, 8)),  # 60 time steps, 8 features
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, return_sequences=False, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.3),
            Dense(len(WorkloadType), activation='softmax')
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        self.lstm_model = model

    def train_lstm(self, training_data: List[pd.DataFrame]):
        """Train LSTM model explicitly when needed"""
        if not TF_AVAILABLE:
            logger.warning("TensorFlow not available. Cannot train LSTM model.")
            return

        if self.lstm_model is None:
            self._build_lstm_model()

        if self.lstm_model is None:
            logger.error("Failed to build LSTM model for training.")
            return

        logger.info("Training LSTM model for workload pattern recognition...")

        try:
            # Prepare training data for LSTM
            X_train, y_train = self._prepare_lstm_training_data(training_data)

            if X_train is not None and len(X_train) > 50:
                # Split training and validation data
                from sklearn.model_selection import train_test_split
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )

                # Train the LSTM model
                history = self.lstm_model.fit(
                    X_train_split, y_train_split,
                    batch_size=32,
                    epochs=100,
                    validation_data=(X_val_split, y_val_split),
                    verbose=1,
                    callbacks=[
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(patience=5, factor=0.5)
                    ]
                )

                # Mark LSTM as trained
                self.lstm_trained = True
                logger.info("LSTM model training completed successfully.")
            else:
                logger.warning("Insufficient training data for LSTM model.")

        except Exception as e:
            logger.error(f"LSTM training failed: {str(e)}")
            self.lstm_trained = False

    def _prepare_lstm_training_data(self, training_data: List[pd.DataFrame]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for LSTM model"""
        if not training_data:
            return None, None

        X, y = [], []
        sequence_length = 60

        for df in training_data:
            if len(df) < sequence_length + 1:
                continue

            # Extract features for each sequence
            for i in range(sequence_length, len(df)):
                # Get sequence window
                window = df.iloc[i-sequence_length:i]

                # Create feature sequence (8 features: cpu, mem, io, net + time features)
                sequence_features = []
                for _, row in window.iterrows():
                    features = [
                        row.get('cpu_usage', 0),
                        row.get('memory_usage', 0),
                        row.get('io_usage', 0),
                        row.get('network_usage', 0),
                        np.sin(2 * np.pi * pd.to_datetime(row['timestamp']).hour / 24),
                        np.cos(2 * np.pi * pd.to_datetime(row['timestamp']).hour / 24),
                        np.sin(2 * np.pi * pd.to_datetime(row['timestamp']).weekday() / 7),
                        np.cos(2 * np.pi * pd.to_datetime(row['timestamp']).weekday() / 7)
                    ]
                    sequence_features.append(features)

                X.append(sequence_features)

                # Create target (workload type as one-hot)
                target = np.zeros(len(WorkloadType))
                # For now, use a simple heuristic to determine workload type
                current_row = df.iloc[i]
                if current_row.get('cpu_usage', 0) > 0.7:
                    target[0] = 1  # CPU_INTENSIVE
                elif current_row.get('memory_usage', 0) > 0.7:
                    target[1] = 1  # MEMORY_INTENSIVE
                else:
                    target[-1] = 1  # UNKNOWN

                y.append(target)

        if len(X) == 0:
            return None, None

        return np.array(X), np.array(y)

    def extract_features(self, workload_data: pd.DataFrame) -> WorkloadFeatures:
        """Extract comprehensive features from workload data"""
        # Resource utilization statistics
        cpu_stats = workload_data['cpu_usage'].agg(['mean', 'std', 'max'])
        memory_stats = workload_data['memory_usage'].agg(['mean', 'std', 'max'])
        io_stats = workload_data['io_usage'].agg(['mean', 'std', 'max'])
        network_stats = workload_data['network_usage'].agg(['mean', 'std', 'max'])

        # Temporal features
        duration_minutes = len(workload_data)
        time_features = pd.to_datetime(workload_data['timestamp'])
        avg_hour = time_features.dt.hour.mean()
        avg_dow = time_features.dt.dayofweek.mean()

        # Advanced pattern features
        burstiness = self._calculate_burstiness(workload_data['cpu_usage'])
        seasonality_score, seasonal_period_samples = self._calculate_seasonality_score(workload_data)
        trend = self._calculate_trend_score(workload_data)

        return WorkloadFeatures(
            cpu_mean=cpu_stats['mean'],
            cpu_std=cpu_stats['std'],
            cpu_max=cpu_stats['max'],
            memory_mean=memory_stats['mean'],
            memory_std=memory_stats['std'],
            memory_max=memory_stats['max'],
            io_mean=io_stats['mean'],
            io_std=io_stats['std'],
            io_max=io_stats['max'],
            network_mean=network_stats['mean'],
            network_std=network_stats['std'],
            network_max=network_stats['max'],
            duration_minutes=duration_minutes,
            time_of_day=int(avg_hour),
            day_of_week=int(avg_dow),
            burstiness=burstiness,
            seasonality_score=seasonality_score,
            seasonal_period_samples=seasonal_period_samples,
            trend_score=trend
        )

    def _calculate_burstiness(self, series: pd.Series) -> float:
        """Calculate burstiness score for time series"""
        if len(series) < 2:
            return 0.0

        mean_val = series.mean()
        std_val = series.std()

        if mean_val == 0:
            return 0.0

        # Burstiness coefficient
        return (std_val - mean_val) / (std_val + mean_val)

    def _calculate_seasonality_score(self, data: pd.DataFrame) -> Tuple[float, Optional[int]]:
        """Calculate seasonality score and dominant period using FFT

        Returns:
            tuple: (seasonality_score, dominant_period_samples)
            - seasonality_score: confidence of seasonal pattern (0-1)
            - dominant_period_samples: period in data samples, None if no strong pattern

        Assumes:
            - Data sampling rate: 1 sample per hour
            - Minimum 24 samples needed for meaningful analysis
        """
        if len(data) < 24:  # Need at least 24 points for meaningful seasonality
            return 0.0, None

        # Use CPU usage as primary signal
        signal = data['cpu_usage'].fillna(0)
        n_samples = len(signal)

        # Apply FFT
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(n_samples)

        # Find dominant frequencies (skip DC component at index 0)
        power = np.abs(fft) ** 2
        # Only consider positive frequencies in the first half
        positive_power = power[1:n_samples//2]

        if len(positive_power) == 0:
            return 0.0, None

        dominant_freq_idx = np.argmax(positive_power) + 1  # +1 because we skipped index 0

        # Calculate dominant period in samples
        # period = 1 / frequency, but freqs are normalized by sample rate
        # For fftfreq, period_samples = n_samples / freq_index
        if dominant_freq_idx > 0:
            dominant_period_samples = n_samples // dominant_freq_idx
            # Clamp to reasonable range (minimum 2 samples, maximum half the data)
            dominant_period_samples = max(2, min(dominant_period_samples, n_samples // 2))
        else:
            dominant_period_samples = None

        # Calculate seasonality confidence based on power concentration
        total_power = np.sum(positive_power)
        dominant_power = power[dominant_freq_idx]

        if total_power == 0:
            return 0.0, dominant_period_samples

        seasonality_score = min(dominant_power / total_power, 1.0)

        # Only return period if seasonality is strong enough
        if seasonality_score < 0.3:  # Minimum threshold for meaningful seasonality
            dominant_period_samples = None

        return seasonality_score, dominant_period_samples

    def _calculate_trend_score(self, data: pd.DataFrame) -> float:
        """Calculate trend score using linear regression"""
        if len(data) < 5:
            return 0.0

        x = np.arange(len(data))
        y = data['cpu_usage'].fillna(0)

        # Simple linear regression
        n = len(x)
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xy = np.sum(x * y)
        sum_x2 = np.sum(x * x)

        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        # Normalize slope to get trend score
        return np.tanh(slope)

    def classify_workload_type(self, features: WorkloadFeatures) -> Tuple[WorkloadType, float]:
        """Classify workload type based on features"""
        # Rule-based classification with confidence scoring
        scores = {}

        # CPU-intensive detection
        cpu_intensity = (features.cpu_mean + features.cpu_max) / 2
        scores[WorkloadType.CPU_INTENSIVE] = cpu_intensity

        # Memory-intensive detection
        memory_intensity = (features.memory_mean + features.memory_max) / 2
        scores[WorkloadType.MEMORY_INTENSIVE] = memory_intensity

        # IO-intensive detection
        io_intensity = (features.io_mean + features.io_max) / 2
        scores[WorkloadType.IO_INTENSIVE] = io_intensity

        # Network-intensive detection
        network_intensity = (features.network_mean + features.network_max) / 2
        scores[WorkloadType.NETWORK_INTENSIVE] = network_intensity

        # Batch processing detection
        batch_score = 0.5 if features.duration_minutes > 60 else 0.2
        if features.burstiness < 0.3:  # Low burstiness indicates steady processing
            batch_score += 0.3
        scores[WorkloadType.BATCH_PROCESSING] = batch_score

        # Real-time detection
        realtime_score = 0.8 if features.burstiness > self.burstiness_threshold else 0.2
        scores[WorkloadType.REAL_TIME] = realtime_score

        # Interactive workload detection
        interactive_score = 0.6 if 9 <= features.time_of_day <= 17 else 0.3
        if features.burstiness > 1.0:
            interactive_score += 0.2
        scores[WorkloadType.INTERACTIVE] = interactive_score

        # Background workload detection
        background_score = 0.7 if features.time_of_day < 6 or features.time_of_day > 22 else 0.2
        scores[WorkloadType.BACKGROUND] = background_score

        # Find the highest scoring type
        best_type = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_type]

        return best_type, confidence

    def detect_pattern_type(self, features: WorkloadFeatures, data: pd.DataFrame) -> Tuple[PatternType, float]:
        """Detect pattern type from features and time series data"""
        scores = {}

        # Seasonal pattern detection
        scores[PatternType.SEASONAL] = features.seasonality_score

        # Trending pattern detection
        trend_abs = abs(features.trend_score)
        scores[PatternType.TRENDING] = trend_abs if trend_abs > self.trend_threshold else 0.0

        # Spike detection
        cpu_series = data['cpu_usage']
        spike_score = 0.0
        if len(cpu_series) > 1:
            z_scores = np.abs((cpu_series - cpu_series.mean()) / cpu_series.std())
            spike_count = np.sum(z_scores > self.spike_threshold)
            spike_score = min(spike_count / len(cpu_series) * 5, 1.0)
        scores[PatternType.SPIKE] = spike_score

        # Cyclic pattern detection (different from seasonal)
        cyclic_score = 0.5 if features.seasonality_score > 0.4 and features.duration_minutes > 120 else 0.0
        scores[PatternType.CYCLIC] = cyclic_score

        # Bursty pattern detection
        bursty_score = min(features.burstiness, 1.0) if features.burstiness > self.burstiness_threshold else 0.0
        scores[PatternType.BURSTY] = bursty_score

        # Steady state detection
        steady_score = 1.0 - features.burstiness if features.burstiness < 0.5 else 0.0
        scores[PatternType.STEADY_STATE] = steady_score

        # Irregular pattern detection
        irregular_score = 0.7 if features.seasonality_score < 0.3 and features.burstiness > 1.5 else 0.0
        scores[PatternType.IRREGULAR] = irregular_score

        best_pattern = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_pattern]

        return best_pattern, confidence

    def analyze_workload(self, vm_id: str, workload_data: pd.DataFrame) -> WorkloadPattern:
        """Comprehensive workload analysis"""
        # Extract features
        features = self.extract_features(workload_data)

        # Use ML models in priority order: LSTM -> ML classifier -> rule-based
        workload_type, workload_confidence = None, 0.0

        # Try LSTM first if trained
        if self.lstm_trained and self.lstm_model is not None and TF_AVAILABLE:
            try:
                workload_type, workload_confidence = self._classify_with_lstm(workload_data)
            except Exception as e:
                logger.warning(f"LSTM classification failed: {str(e)}")

        # Try ML classification model if LSTM failed or isn't available
        if workload_type is None and self.classification_model is not None:
            try:
                workload_type, workload_confidence = self.predict_workload_type_ml(features)
            except Exception as e:
                logger.warning(f"ML classification failed: {str(e)}")

        # Fall back to rule-based classification
        if workload_type is None:
            workload_type, workload_confidence = self.classify_workload_type(features)

        pattern_type, pattern_confidence = self.detect_pattern_type(features, workload_data)

        # Calculate additional pattern characteristics
        characteristics = {
            'vm_id': vm_id,
            'resource_intensity': {
                'cpu': features.cpu_mean,
                'memory': features.memory_mean,
                'io': features.io_mean,
                'network': features.network_mean
            },
            'variability': {
                'cpu_cv': features.cpu_std / features.cpu_mean if features.cpu_mean > 0 else 0,
                'memory_cv': features.memory_std / features.memory_mean if features.memory_mean > 0 else 0
            },
            'temporal_profile': {
                'duration': features.duration_minutes,
                'time_of_day': features.time_of_day,
                'day_of_week': features.day_of_week
            }
        }

        # Generate stable, unique pattern ID using UUID5
        key = json.dumps(characteristics, sort_keys=True)
        pattern_uuid = uuid.uuid5(uuid.NAMESPACE_URL, key)
        pattern_id = f"{workload_type.value}_{pattern_type.value}_{pattern_uuid}"

        # Create pattern object
        pattern = WorkloadPattern(
            pattern_id=pattern_id,
            workload_type=workload_type,
            pattern_type=pattern_type,
            confidence=min(workload_confidence, pattern_confidence),
            characteristics=characteristics,
            frequency=self._calculate_frequency(features.seasonal_period_samples),
            amplitude=features.cpu_std,
            duration=features.duration_minutes,
            seasonal_period=self._calculate_seasonal_period_hours(features.seasonal_period_samples),
            trend_direction="up" if features.trend_score > 0.1 else "down" if features.trend_score < -0.1 else "stable",
            created_at=datetime.now(),
            last_seen=datetime.now(),
            occurrence_count=1
        )

        # Store pattern
        self._store_pattern(pattern)
        self._store_workload_history(vm_id, features, pattern)

        return pattern

    def _calculate_frequency(self, period_samples: Optional[int]) -> Optional[float]:
        """Calculate frequency in cycles per hour

        Args:
            period_samples: Period in data samples (assuming 1 sample = 1 hour)

        Returns:
            Frequency in cycles per hour, or None if no valid period
        """
        if period_samples is None or period_samples <= 0:
            return None

        # Assuming 1 sample = 1 hour, frequency = cycles per hour
        return 1.0 / period_samples

    def _calculate_seasonal_period_hours(self, period_samples: Optional[int]) -> Optional[int]:
        """Calculate seasonal period in hours

        Args:
            period_samples: Period in data samples (assuming 1 sample = 1 hour)

        Returns:
            Period in hours, or None if no valid period
        """
        if period_samples is None or period_samples <= 0:
            return None

        # Assuming 1 sample = 1 hour, period in hours = period in samples
        return period_samples

    def _store_pattern(self, pattern: WorkloadPattern):
        """Store pattern in database"""
        with sqlite3.connect(self.db_path) as conn:
            # Check if pattern exists
            existing = conn.execute(
                "SELECT occurrence_count FROM workload_patterns WHERE pattern_id = ?",
                (pattern.pattern_id,)
            ).fetchone()

            if existing:
                # Update existing pattern
                conn.execute("""
                    UPDATE workload_patterns
                    SET last_seen = ?, occurrence_count = occurrence_count + 1
                    WHERE pattern_id = ?
                """, (pattern.last_seen.isoformat(), pattern.pattern_id))
            else:
                # Insert new pattern
                conn.execute("""
                    INSERT INTO workload_patterns
                    (pattern_id, workload_type, pattern_type, confidence, characteristics,
                     frequency, amplitude, duration, seasonal_period, trend_direction,
                     created_at, last_seen, occurrence_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_id, pattern.workload_type.value, pattern.pattern_type.value,
                    pattern.confidence, json.dumps(pattern.characteristics),
                    pattern.frequency, pattern.amplitude, pattern.duration,
                    pattern.seasonal_period, pattern.trend_direction,
                    pattern.created_at.isoformat(), pattern.last_seen.isoformat(),
                    pattern.occurrence_count
                ))

    def _store_workload_history(self, vm_id: str, features: WorkloadFeatures, pattern: WorkloadPattern):
        """Store workload classification history"""
        with sqlite3.connect(self.db_path) as conn:
            # Convert features to dict and handle numpy types
            features_dict = asdict(features)
            # Convert numpy types to native Python types for JSON serialization
            for key, value in features_dict.items():
                if isinstance(value, (np.int64, np.int32)):
                    features_dict[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    features_dict[key] = float(value)
                elif value is None:
                    features_dict[key] = None

            conn.execute("""
                INSERT INTO workload_history
                (timestamp, vm_id, features, pattern_id, classification_confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                vm_id,
                json.dumps(features_dict),
                pattern.pattern_id,
                pattern.confidence
            ))

    def get_patterns_by_type(self, workload_type: WorkloadType = None,
                           pattern_type: PatternType = None) -> List[WorkloadPattern]:
        """Retrieve patterns by type"""
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM workload_patterns WHERE 1=1"
            params = []

            if workload_type:
                query += " AND workload_type = ?"
                params.append(workload_type.value)

            if pattern_type:
                query += " AND pattern_type = ?"
                params.append(pattern_type.value)

            query += " ORDER BY occurrence_count DESC"

            rows = conn.execute(query, params).fetchall()

            patterns = []
            for row in rows:
                pattern = WorkloadPattern(
                    pattern_id=row[0],
                    workload_type=WorkloadType(row[1]),
                    pattern_type=PatternType(row[2]),
                    confidence=row[3],
                    characteristics=json.loads(row[4]),
                    frequency=row[5],
                    amplitude=row[6],
                    duration=row[7],
                    seasonal_period=row[8],
                    trend_direction=row[9],
                    created_at=datetime.fromisoformat(row[10]),
                    last_seen=datetime.fromisoformat(row[11]),
                    occurrence_count=row[12]
                )
                patterns.append(pattern)

            return patterns

    def predict_future_pattern(self, vm_id: str, hours_ahead: int = 24) -> Dict[str, Any]:
        """Predict future workload patterns"""
        # Get recent workload history
        with sqlite3.connect(self.db_path) as conn:
            recent_data = conn.execute("""
                SELECT features, pattern_id, classification_confidence, timestamp
                FROM workload_history
                WHERE vm_id = ? AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
            """, (vm_id,)).fetchall()

        if not recent_data:
            return {"prediction": "insufficient_data", "confidence": 0.0}

        # Analyze pattern evolution
        pattern_sequence = [row[1] for row in recent_data[-24:]]  # Last 24 observations

        if not pattern_sequence:
            return {"prediction": "no_patterns", "confidence": 0.0}

        # Simple pattern-based prediction (can be enhanced with ML)
        pattern_counts = {}
        for pattern_id in pattern_sequence:
            pattern_counts[pattern_id] = pattern_counts.get(pattern_id, 0) + 1

        # Most likely next pattern
        most_likely_pattern = max(pattern_counts.keys(), key=lambda k: pattern_counts[k])
        confidence = pattern_counts[most_likely_pattern] / len(pattern_sequence)

        return {
            "prediction": most_likely_pattern,
            "confidence": confidence,
            "alternative_patterns": sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "prediction_horizon_hours": hours_ahead
        }

    def train_classification_model(self, retrain: bool = False):
        """Train ML models on historical data"""
        if not retrain and self.classification_model is not None:
            logger.info("Models already trained, skipping...")
            return

        # Load training data - JOIN with workload_patterns to get stable workload_type labels
        with sqlite3.connect(self.db_path) as conn:
            data = conn.execute("""
                SELECT wh.features, wp.workload_type
                FROM workload_history wh
                JOIN workload_patterns wp ON wh.pattern_id = wp.pattern_id
                WHERE wh.classification_confidence > 0.6
                AND wp.workload_type IS NOT NULL
            """).fetchall()

        if len(data) < 50:
            logger.warning("Insufficient training data, using pre-configured rules")
            return

        # Prepare training data
        X = []
        y = []

        for row in data:
            features_dict = json.loads(row[0])
            workload_type = row[1]  # Use workload_type instead of pattern_id

            # Convert features to vector
            feature_vector = [
                features_dict['cpu_mean'], features_dict['cpu_std'], features_dict['cpu_max'],
                features_dict['memory_mean'], features_dict['memory_std'], features_dict['memory_max'],
                features_dict['io_mean'], features_dict['io_std'], features_dict['io_max'],
                features_dict['network_mean'], features_dict['network_std'], features_dict['network_max'],
                features_dict['burstiness'], features_dict['seasonality_score'], features_dict['trend_score'],
                features_dict['time_of_day'], features_dict['day_of_week'],
                features_dict.get('seasonal_period_samples', 0) or 0  # Handle None/missing values
            ]

            X.append(feature_vector)
            y.append(workload_type)  # Use stable workload type label

        X = np.array(X)
        y = np.array(y)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train classification model
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.classification_model.fit(X_train, y_train)

        # Evaluate model
        train_score = self.classification_model.score(X_train, y_train)
        test_score = self.classification_model.score(X_test, y_test)

        logger.info(f"Classification model trained - Train: {train_score:.3f}, Test: {test_score:.3f}")

        # Store feature importance
        if hasattr(self.classification_model, 'feature_importances_'):
            feature_names = [
                'cpu_mean', 'cpu_std', 'cpu_max',
                'memory_mean', 'memory_std', 'memory_max',
                'io_mean', 'io_std', 'io_max',
                'network_mean', 'network_std', 'network_max',
                'burstiness', 'seasonality_score', 'trend_score',
                'time_of_day', 'day_of_week', 'seasonal_period_samples'
            ]

            self.feature_importance = dict(zip(
                feature_names,
                self.classification_model.feature_importances_
            ))

    def predict_workload_type_ml(self, features: WorkloadFeatures) -> Tuple[WorkloadType, float]:
        """Predict workload type using trained ML model"""
        if self.classification_model is None:
            logger.warning("Classification model not trained, falling back to rule-based classification")
            return self.classify_workload_type(features)

        # Check if scaler is fitted
        if not hasattr(self.scaler, 'scale_'):
            logger.warning("Model scaler not fitted, falling back to rule-based classification")
            return self.classify_workload_type(features)

        try:
            # Convert features to vector
            feature_vector = np.array([[
                features.cpu_mean, features.cpu_std, features.cpu_max,
                features.memory_mean, features.memory_std, features.memory_max,
                features.io_mean, features.io_std, features.io_max,
                features.network_mean, features.network_std, features.network_max,
                features.burstiness, features.seasonality_score, features.trend_score,
                features.time_of_day, features.day_of_week,
                features.seasonal_period_samples or 0  # Handle None values
            ]])

            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)

            # Predict workload type
            predicted_type = self.classification_model.predict(feature_vector_scaled)[0]

            # Get prediction probabilities for confidence score
            if hasattr(self.classification_model, 'predict_proba'):
                probabilities = self.classification_model.predict_proba(feature_vector_scaled)[0]
                confidence = np.max(probabilities)
            else:
                # Fallback confidence for models without probability support
                confidence = 0.7

            # Convert string back to WorkloadType enum
            try:
                workload_type = WorkloadType(predicted_type)
                return workload_type, float(confidence)
            except ValueError:
                logger.warning(f"Unknown workload type predicted: {predicted_type}")
                return WorkloadType.UNKNOWN, 0.5

        except Exception as e:
            logger.warning(f"ML prediction failed: {str(e)}, falling back to rule-based")
            return self.classify_workload_type(features)

    def _classify_with_lstm(self, workload_data: pd.DataFrame) -> Tuple[WorkloadType, float]:
        """Classify workload using trained LSTM model"""
        if not self.lstm_trained or self.lstm_model is None:
            raise ValueError("LSTM model not trained")

        sequence_length = 60
        if len(workload_data) < sequence_length:
            raise ValueError("Insufficient data for LSTM classification")

        # Prepare input sequence
        sequence_features = []
        for _, row in workload_data.iloc[-sequence_length:].iterrows():
            features = [
                row.get('cpu_usage', 0),
                row.get('memory_usage', 0),
                row.get('io_usage', 0),
                row.get('network_usage', 0),
                np.sin(2 * np.pi * pd.to_datetime(row['timestamp']).hour / 24),
                np.cos(2 * np.pi * pd.to_datetime(row['timestamp']).hour / 24),
                np.sin(2 * np.pi * pd.to_datetime(row['timestamp']).weekday() / 7),
                np.cos(2 * np.pi * pd.to_datetime(row['timestamp']).weekday() / 7)
            ]
            sequence_features.append(features)

        # Predict with LSTM
        X = np.array([sequence_features])
        prediction = self.lstm_model.predict(X, verbose=0)[0]

        # Get most likely workload type
        predicted_idx = np.argmax(prediction)
        confidence = prediction[predicted_idx]

        workload_types = list(WorkloadType)
        if predicted_idx < len(workload_types):
            return workload_types[predicted_idx], float(confidence)
        else:
            return WorkloadType.UNKNOWN, 0.5


class WorkloadAnomalyDetector:
    """Specialized anomaly detection for workload patterns"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, normal_patterns: List[WorkloadFeatures]):
        """Fit anomaly detector on normal patterns"""
        if not normal_patterns:
            return

        # Convert to feature matrix
        X = self._features_to_matrix(normal_patterns)
        X_scaled = self.scaler.fit_transform(X)

        # Fit isolation forest
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True

    def detect_anomalies(self, patterns: List[WorkloadFeatures]) -> List[bool]:
        """Detect anomalies in workload patterns"""
        if not self.is_fitted or not patterns:
            return [False] * len(patterns)

        X = self._features_to_matrix(patterns)
        X_scaled = self.scaler.transform(X)

        predictions = self.isolation_forest.predict(X_scaled)
        return [pred == -1 for pred in predictions]

    def _features_to_matrix(self, features_list: List[WorkloadFeatures]) -> np.ndarray:
        """Convert WorkloadFeatures list to matrix"""
        matrix = []
        for features in features_list:
            row = [
                features.cpu_mean, features.cpu_std, features.cpu_max,
                features.memory_mean, features.memory_std, features.memory_max,
                features.io_mean, features.io_std, features.io_max,
                features.network_mean, features.network_std, features.network_max,
                features.burstiness, features.seasonality_score, features.trend_score,
                features.seasonal_period_samples or 0  # Handle None values
            ]
            matrix.append(row)

        return np.array(matrix)


# Legacy wrapper for backward compatibility
class WorkloadClassifier(WorkloadPatternRecognizer):
    """Legacy wrapper maintaining API compatibility"""

    def __init__(self, db_path: str = None):
        # Initialize parent with proper db path handling
        super().__init__(db_path=db_path)

    def classify(self, workload_data: pd.DataFrame) -> Dict[str, Any]:
        """Legacy classification method"""
        pattern = self.analyze_workload("legacy_vm", workload_data)

        return {
            'workload_type': pattern.workload_type.value,
            'pattern_type': pattern.pattern_type.value,
            'confidence': pattern.confidence,
            'characteristics': pattern.characteristics
        }