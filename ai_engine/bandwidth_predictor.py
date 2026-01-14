#!/usr/bin/env python3
"""
AI-powered bandwidth prediction engine for NovaCron network fabric.
Uses machine learning models to predict network performance, bandwidth requirements,
and optimal routing decisions.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
import pickle
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import sqlite3
import threading
import requests
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """Network performance metrics"""
    timestamp: datetime
    source_node: str
    target_node: str
    bandwidth_mbps: float
    latency_ms: float
    packet_loss: float
    jitter_ms: float
    throughput_mbps: float
    connection_quality: float
    route_hops: int
    congestion_level: float

@dataclass
class WorkloadCharacteristics:
    """VM workload characteristics for bandwidth prediction"""
    vm_id: str
    workload_type: str  # interactive, batch, streaming, compute
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    network_intensive: bool
    expected_connections: int
    data_transfer_pattern: str  # burst, steady, periodic
    peak_hours: List[int]
    historical_bandwidth: float

@dataclass
class PredictionRequest:
    """Request for bandwidth prediction"""
    source_node: str
    target_node: str
    workload_chars: WorkloadCharacteristics
    time_horizon_hours: int
    confidence_level: float
    include_uncertainty: bool

@dataclass
class BandwidthPrediction:
    """Bandwidth prediction result"""
    predicted_bandwidth: float
    confidence_interval: Tuple[float, float]
    prediction_confidence: float
    optimal_time_window: Optional[Tuple[datetime, datetime]]
    alternative_routes: List[Dict[str, Any]]
    congestion_forecast: Dict[str, float]
    recommendation: str

class BandwidthPredictor:
    """AI-powered bandwidth prediction engine"""
    
    def __init__(self, db_path: str = "/tmp/bandwidth_predictor.db", 
                 model_path: str = "/tmp/bandwidth_models/"):
        self.db_path = db_path
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
        # Model hyperparameters
        self.model_configs = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'mlp': {
                'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128, 64)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01],
                'max_iter': [500, 1000, 1500]
            }
        }
        
        self.lock = threading.Lock()
        self.prediction_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Initialize database
        self._init_database()
        
        # Start background training
        self.training_thread = None
        self.should_retrain = False
        self.last_training_time = None
        
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Network metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS network_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    source_node TEXT,
                    target_node TEXT,
                    bandwidth_mbps REAL,
                    latency_ms REAL,
                    packet_loss REAL,
                    jitter_ms REAL,
                    throughput_mbps REAL,
                    connection_quality REAL,
                    route_hops INTEGER,
                    congestion_level REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Workload characteristics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workload_characteristics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vm_id TEXT,
                    workload_type TEXT,
                    cpu_cores INTEGER,
                    memory_gb REAL,
                    storage_gb REAL,
                    network_intensive BOOLEAN,
                    expected_connections INTEGER,
                    data_transfer_pattern TEXT,
                    peak_hours TEXT,
                    historical_bandwidth REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Prediction cache table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE,
                    prediction_json TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expires_at DATETIME
                )
            ''')
            
            # Training statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    mae REAL,
                    mse REAL,
                    r2_score REAL,
                    training_samples INTEGER,
                    training_time REAL,
                    hyperparameters TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def store_network_metrics(self, metrics: NetworkMetrics) -> bool:
        """Store network performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO network_metrics 
                    (timestamp, source_node, target_node, bandwidth_mbps, latency_ms,
                     packet_loss, jitter_ms, throughput_mbps, connection_quality,
                     route_hops, congestion_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.timestamp, metrics.source_node, metrics.target_node,
                    metrics.bandwidth_mbps, metrics.latency_ms, metrics.packet_loss,
                    metrics.jitter_ms, metrics.throughput_mbps, metrics.connection_quality,
                    metrics.route_hops, metrics.congestion_level
                ))
                conn.commit()
                
            # Mark for retraining if we have enough new data
            self._check_retraining_needed()
            return True
            
        except Exception as e:
            logger.error(f"Error storing network metrics: {e}")
            return False
    
    def store_workload_characteristics(self, workload: WorkloadCharacteristics) -> bool:
        """Store VM workload characteristics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO workload_characteristics 
                    (vm_id, workload_type, cpu_cores, memory_gb, storage_gb,
                     network_intensive, expected_connections, data_transfer_pattern,
                     peak_hours, historical_bandwidth)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    workload.vm_id, workload.workload_type, workload.cpu_cores,
                    workload.memory_gb, workload.storage_gb, workload.network_intensive,
                    workload.expected_connections, workload.data_transfer_pattern,
                    json.dumps(workload.peak_hours), workload.historical_bandwidth
                ))
                conn.commit()
                
            return True
            
        except Exception as e:
            logger.error(f"Error storing workload characteristics: {e}")
            return False
    
    def _load_training_data(self) -> Optional[pd.DataFrame]:
        """Load training data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Join network metrics with workload characteristics
                query = '''
                    SELECT 
                        nm.timestamp,
                        nm.source_node,
                        nm.target_node,
                        nm.bandwidth_mbps,
                        nm.latency_ms,
                        nm.packet_loss,
                        nm.jitter_ms,
                        nm.throughput_mbps,
                        nm.connection_quality,
                        nm.route_hops,
                        nm.congestion_level,
                        wc.workload_type,
                        wc.cpu_cores,
                        wc.memory_gb,
                        wc.storage_gb,
                        wc.network_intensive,
                        wc.expected_connections,
                        wc.data_transfer_pattern,
                        wc.historical_bandwidth
                    FROM network_metrics nm
                    LEFT JOIN workload_characteristics wc ON 1=1
                    WHERE nm.timestamp >= datetime('now', '-30 days')
                    ORDER BY nm.timestamp DESC
                    LIMIT 100000
                '''
                
                df = pd.read_sql_query(query, conn)
                
                if len(df) < 100:
                    logger.warning("Insufficient training data available")
                    return None
                
                # Feature engineering
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['is_peak_hour'] = df['hour'].isin([8, 9, 10, 16, 17, 18]).astype(int)
                
                # Handle missing values
                df.fillna(0, inplace=True)
                
                return df
                
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for training"""
        # Define features
        numerical_features = [
            'latency_ms', 'packet_loss', 'jitter_ms', 'throughput_mbps',
            'connection_quality', 'route_hops', 'congestion_level',
            'cpu_cores', 'memory_gb', 'storage_gb', 'expected_connections',
            'historical_bandwidth', 'hour', 'day_of_week', 'is_weekend', 'is_peak_hour'
        ]
        
        categorical_features = [
            'source_node', 'target_node', 'workload_type', 'data_transfer_pattern'
        ]
        
        # Handle categorical variables
        for feature in categorical_features:
            if feature not in self.encoders:
                self.encoders[feature] = LabelEncoder()
                df[feature] = self.encoders[feature].fit_transform(df[feature].astype(str))
            else:
                # Handle new categories
                df[feature] = df[feature].astype(str)
                unique_values = set(df[feature].unique())
                known_values = set(self.encoders[feature].classes_)
                new_values = unique_values - known_values
                
                if new_values:
                    # Extend encoder classes
                    all_classes = list(self.encoders[feature].classes_) + list(new_values)
                    self.encoders[feature].classes_ = np.array(all_classes)
                
                df[feature] = self.encoders[feature].transform(df[feature])
        
        # Handle boolean features
        df['network_intensive'] = df['network_intensive'].astype(int)
        
        # Prepare feature matrix
        feature_columns = numerical_features + categorical_features + ['network_intensive']
        X = df[feature_columns].fillna(0)
        y = df['bandwidth_mbps']
        
        self.feature_columns = feature_columns
        
        return X, y
    
    def train_models(self) -> bool:
        """Train bandwidth prediction models"""
        logger.info("Starting model training...")
        
        try:
            # Load training data
            df = self._load_training_data()
            if df is None:
                logger.error("No training data available")
                return False
            
            # Prepare features
            X, y = self._prepare_features(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            if 'scaler' not in self.scalers:
                self.scalers['scaler'] = StandardScaler()
            
            X_train_scaled = self.scalers['scaler'].fit_transform(X_train)
            X_test_scaled = self.scalers['scaler'].transform(X_test)
            
            # Train models with hyperparameter tuning
            models_to_train = {
                'random_forest': RandomForestRegressor(random_state=42),
                'gradient_boost': GradientBoostingRegressor(random_state=42),
                'mlp': MLPRegressor(random_state=42, max_iter=1000)
            }
            
            best_scores = {}
            training_start = time.time()
            
            for model_name, model in models_to_train.items():
                logger.info(f"Training {model_name}...")
                
                # Hyperparameter tuning
                grid_search = GridSearchCV(
                    model, 
                    self.model_configs[model_name],
                    cv=3, 
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1
                )
                
                if model_name == 'mlp':
                    grid_search.fit(X_train_scaled, y_train)
                    y_pred = grid_search.predict(X_test_scaled)
                else:
                    grid_search.fit(X_train, y_train)
                    y_pred = grid_search.predict(X_test)
                
                # Store best model
                self.models[model_name] = grid_search.best_estimator_
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                best_scores[model_name] = {'mae': mae, 'mse': mse, 'r2': r2}
                
                # Store training statistics
                self._store_training_stats(
                    model_name, mae, mse, r2, len(X_train),
                    time.time() - training_start,
                    json.dumps(grid_search.best_params_)
                )
                
                logger.info(f"{model_name} - MAE: {mae:.3f}, MSE: {mse:.3f}, R²: {r2:.3f}")
            
            # Select best model
            best_model = min(best_scores.keys(), key=lambda k: best_scores[k]['mae'])
            logger.info(f"Best model: {best_model}")
            
            # Save models
            self._save_models()
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            self.should_retrain = False
            
            logger.info(f"Model training completed in {time.time() - training_start:.2f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
    
    def _store_training_stats(self, model_name: str, mae: float, mse: float, 
                            r2: float, samples: int, training_time: float,
                            hyperparameters: str):
        """Store training statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO training_stats 
                    (model_name, mae, mse, r2_score, training_samples, 
                     training_time, hyperparameters)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (model_name, mae, mse, r2, samples, training_time, hyperparameters))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing training stats: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        try:
            import os
            os.makedirs(self.model_path, exist_ok=True)
            
            # Save models
            with open(f"{self.model_path}/models.pkl", 'wb') as f:
                pickle.dump(self.models, f)
            
            # Save scalers
            with open(f"{self.model_path}/scalers.pkl", 'wb') as f:
                pickle.dump(self.scalers, f)
            
            # Save encoders
            with open(f"{self.model_path}/encoders.pkl", 'wb') as f:
                pickle.dump(self.encoders, f)
            
            # Save feature columns
            with open(f"{self.model_path}/features.pkl", 'wb') as f:
                pickle.dump(self.feature_columns, f)
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_models(self) -> bool:
        """Load trained models from disk"""
        try:
            # Load models
            with open(f"{self.model_path}/models.pkl", 'rb') as f:
                self.models = pickle.load(f)
            
            # Load scalers
            with open(f"{self.model_path}/scalers.pkl", 'rb') as f:
                self.scalers = pickle.load(f)
            
            # Load encoders
            with open(f"{self.model_path}/encoders.pkl", 'rb') as f:
                self.encoders = pickle.load(f)
            
            # Load feature columns
            with open(f"{self.model_path}/features.pkl", 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.warning(f"Could not load existing models: {e}")
            return False
    
    def predict_bandwidth(self, request: PredictionRequest) -> BandwidthPrediction:
        """Predict bandwidth for a given request"""
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_result = self._get_cached_prediction(cache_key)
        if cached_result:
            return cached_result
        
        if not self.is_trained:
            if not self._load_models():
                # Train models if none exist
                if not self.train_models():
                    # Return fallback prediction
                    return self._fallback_prediction(request)
        
        try:
            # Prepare input features
            features = self._prepare_prediction_features(request)
            
            # Get predictions from all models
            predictions = {}
            for model_name, model in self.models.items():
                if model_name == 'mlp':
                    # Scale features for MLP
                    features_scaled = self.scalers['scaler'].transform([features])[0]
                    pred = model.predict([features_scaled])[0]
                else:
                    pred = model.predict([features])[0]
                predictions[model_name] = max(0, pred)  # Ensure non-negative
            
            # Ensemble prediction (weighted average)
            weights = {'random_forest': 0.4, 'gradient_boost': 0.4, 'mlp': 0.2}
            ensemble_pred = sum(predictions[model] * weights.get(model, 1/len(predictions)) 
                              for model in predictions)
            
            # Calculate confidence interval (simplified)
            pred_std = np.std(list(predictions.values()))
            z_score = 1.96 if request.confidence_level >= 0.95 else 1.645
            confidence_interval = (
                max(0, ensemble_pred - z_score * pred_std),
                ensemble_pred + z_score * pred_std
            )
            
            # Calculate prediction confidence
            pred_confidence = max(0, 1 - pred_std / max(ensemble_pred, 1))
            
            # Generate recommendations
            recommendation = self._generate_recommendation(
                ensemble_pred, pred_confidence, request
            )
            
            # Get congestion forecast
            congestion_forecast = self._forecast_congestion(request)
            
            # Find optimal time window
            optimal_window = self._find_optimal_time_window(request)
            
            # Get alternative routes
            alternative_routes = self._get_alternative_routes(request)
            
            result = BandwidthPrediction(
                predicted_bandwidth=ensemble_pred,
                confidence_interval=confidence_interval,
                prediction_confidence=pred_confidence,
                optimal_time_window=optimal_window,
                alternative_routes=alternative_routes,
                congestion_forecast=congestion_forecast,
                recommendation=recommendation
            )
            
            # Cache result
            self._cache_prediction(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return self._fallback_prediction(request)
    
    def _prepare_prediction_features(self, request: PredictionRequest) -> List[float]:
        """Prepare features for prediction"""
        # Get current time features
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        is_weekend = 1 if day_of_week in [5, 6] else 0
        is_peak_hour = 1 if hour in [8, 9, 10, 16, 17, 18] else 0
        
        # Get historical network metrics for this route
        historical_metrics = self._get_historical_metrics(
            request.source_node, request.target_node
        )
        
        # Prepare feature vector based on training feature columns
        features = []
        for feature in self.feature_columns:
            if feature == 'latency_ms':
                features.append(historical_metrics.get('avg_latency', 10.0))
            elif feature == 'packet_loss':
                features.append(historical_metrics.get('avg_packet_loss', 0.001))
            elif feature == 'jitter_ms':
                features.append(historical_metrics.get('avg_jitter', 2.0))
            elif feature == 'throughput_mbps':
                features.append(historical_metrics.get('avg_throughput', 100.0))
            elif feature == 'connection_quality':
                features.append(historical_metrics.get('avg_quality', 0.8))
            elif feature == 'route_hops':
                features.append(historical_metrics.get('avg_hops', 5))
            elif feature == 'congestion_level':
                features.append(historical_metrics.get('avg_congestion', 0.3))
            elif feature == 'cpu_cores':
                features.append(request.workload_chars.cpu_cores)
            elif feature == 'memory_gb':
                features.append(request.workload_chars.memory_gb)
            elif feature == 'storage_gb':
                features.append(request.workload_chars.storage_gb)
            elif feature == 'expected_connections':
                features.append(request.workload_chars.expected_connections)
            elif feature == 'historical_bandwidth':
                features.append(request.workload_chars.historical_bandwidth)
            elif feature == 'hour':
                features.append(hour)
            elif feature == 'day_of_week':
                features.append(day_of_week)
            elif feature == 'is_weekend':
                features.append(is_weekend)
            elif feature == 'is_peak_hour':
                features.append(is_peak_hour)
            elif feature == 'source_node':
                # Encode using stored encoder
                try:
                    encoded = self.encoders[feature].transform([request.source_node])[0]
                except ValueError:
                    encoded = 0  # Unknown node
                features.append(encoded)
            elif feature == 'target_node':
                try:
                    encoded = self.encoders[feature].transform([request.target_node])[0]
                except ValueError:
                    encoded = 0  # Unknown node
                features.append(encoded)
            elif feature == 'workload_type':
                try:
                    encoded = self.encoders[feature].transform([request.workload_chars.workload_type])[0]
                except ValueError:
                    encoded = 0  # Unknown workload type
                features.append(encoded)
            elif feature == 'data_transfer_pattern':
                try:
                    encoded = self.encoders[feature].transform([request.workload_chars.data_transfer_pattern])[0]
                except ValueError:
                    encoded = 0  # Unknown pattern
                features.append(encoded)
            elif feature == 'network_intensive':
                features.append(1 if request.workload_chars.network_intensive else 0)
            else:
                features.append(0.0)  # Default value
        
        return features
    
    def _get_historical_metrics(self, source_node: str, target_node: str) -> Dict[str, float]:
        """Get historical network metrics for a route"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        AVG(latency_ms) as avg_latency,
                        AVG(packet_loss) as avg_packet_loss,
                        AVG(jitter_ms) as avg_jitter,
                        AVG(throughput_mbps) as avg_throughput,
                        AVG(connection_quality) as avg_quality,
                        AVG(route_hops) as avg_hops,
                        AVG(congestion_level) as avg_congestion
                    FROM network_metrics 
                    WHERE source_node = ? AND target_node = ?
                    AND timestamp >= datetime('now', '-7 days')
                ''', (source_node, target_node))
                
                result = cursor.fetchone()
                if result and result[0] is not None:
                    return {
                        'avg_latency': result[0],
                        'avg_packet_loss': result[1],
                        'avg_jitter': result[2],
                        'avg_throughput': result[3],
                        'avg_quality': result[4],
                        'avg_hops': result[5],
                        'avg_congestion': result[6]
                    }
                    
        except Exception as e:
            logger.error(f"Error getting historical metrics: {e}")
        
        # Return defaults if no data
        return {
            'avg_latency': 10.0,
            'avg_packet_loss': 0.001,
            'avg_jitter': 2.0,
            'avg_throughput': 100.0,
            'avg_quality': 0.8,
            'avg_hops': 5,
            'avg_congestion': 0.3
        }
    
    def _generate_recommendation(self, predicted_bandwidth: float, 
                               confidence: float, request: PredictionRequest) -> str:
        """Generate actionable recommendation"""
        if confidence < 0.6:
            return "LOW_CONFIDENCE: Consider collecting more network metrics for this route"
        elif predicted_bandwidth < 10:
            return "INSUFFICIENT_BANDWIDTH: Consider alternative routes or scheduling"
        elif predicted_bandwidth > 1000:
            return "EXCELLENT_BANDWIDTH: Optimal conditions for data transfer"
        elif confidence > 0.8 and predicted_bandwidth > 100:
            return "GOOD_CONDITIONS: Proceed with current routing plan"
        else:
            return "MODERATE_CONDITIONS: Monitor network performance during transfer"
    
    def _forecast_congestion(self, request: PredictionRequest) -> Dict[str, float]:
        """Forecast network congestion for the next few hours"""
        # Simplified congestion forecast based on historical patterns
        current_hour = datetime.now().hour
        congestion_forecast = {}
        
        # Peak hours typically have higher congestion
        peak_hours = [8, 9, 10, 16, 17, 18]
        
        for i in range(request.time_horizon_hours):
            hour = (current_hour + i) % 24
            if hour in peak_hours:
                congestion = 0.7 + 0.2 * np.random.random()  # 70-90% congestion
            elif hour in [0, 1, 2, 3, 4, 5]:
                congestion = 0.1 + 0.2 * np.random.random()  # 10-30% congestion
            else:
                congestion = 0.3 + 0.3 * np.random.random()  # 30-60% congestion
            
            congestion_forecast[f"hour_{i+1}"] = congestion
        
        return congestion_forecast
    
    def _find_optimal_time_window(self, request: PredictionRequest) -> Optional[Tuple[datetime, datetime]]:
        """Find optimal time window for data transfer"""
        # Find the hour with lowest predicted congestion
        congestion = self._forecast_congestion(request)
        min_congestion_hour = min(congestion.keys(), key=lambda k: congestion[k])
        
        # Calculate optimal start time
        hour_offset = int(min_congestion_hour.split('_')[1])
        optimal_start = datetime.now() + timedelta(hours=hour_offset)
        optimal_end = optimal_start + timedelta(hours=1)  # 1-hour window
        
        return (optimal_start, optimal_end)
    
    def _get_alternative_routes(self, request: PredictionRequest) -> List[Dict[str, Any]]:
        """Get alternative routes with predictions"""
        # Simplified alternative route suggestion
        # In a real implementation, this would query the network topology
        alternatives = []
        
        # Generate some dummy alternative routes
        for i in range(2):
            alt_route = {
                'route_id': f"alt_route_{i+1}",
                'intermediate_nodes': [f"intermediate_{i+1}"],
                'predicted_bandwidth': np.random.uniform(50, 200),
                'estimated_latency': np.random.uniform(15, 50),
                'reliability_score': np.random.uniform(0.7, 0.95)
            }
            alternatives.append(alt_route)
        
        return alternatives
    
    def _fallback_prediction(self, request: PredictionRequest) -> BandwidthPrediction:
        """Provide fallback prediction when models are unavailable"""
        # Simple heuristic-based prediction
        base_bandwidth = 100.0  # Base 100 Mbps
        
        # Adjust based on workload characteristics
        if request.workload_chars.network_intensive:
            base_bandwidth *= 1.5
        
        if request.workload_chars.workload_type == 'streaming':
            base_bandwidth *= 1.2
        elif request.workload_chars.workload_type == 'batch':
            base_bandwidth *= 0.8
        
        return BandwidthPrediction(
            predicted_bandwidth=base_bandwidth,
            confidence_interval=(base_bandwidth * 0.7, base_bandwidth * 1.3),
            prediction_confidence=0.5,
            optimal_time_window=None,
            alternative_routes=[],
            congestion_forecast={},
            recommendation="FALLBACK_PREDICTION: Using heuristic-based estimate"
        )
    
    def _generate_cache_key(self, request: PredictionRequest) -> str:
        """Generate cache key for prediction request"""
        import hashlib
        
        key_data = f"{request.source_node}_{request.target_node}_{request.workload_chars.vm_id}_{request.time_horizon_hours}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _cache_prediction(self, cache_key: str, prediction: BandwidthPrediction):
        """Cache prediction result"""
        try:
            expires_at = datetime.now() + timedelta(seconds=self.cache_ttl)
            prediction_json = json.dumps({
                'predicted_bandwidth': prediction.predicted_bandwidth,
                'confidence_interval': prediction.confidence_interval,
                'prediction_confidence': prediction.prediction_confidence,
                'recommendation': prediction.recommendation,
                'congestion_forecast': prediction.congestion_forecast
            })
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO prediction_cache 
                    (cache_key, prediction_json, expires_at)
                    VALUES (?, ?, ?)
                ''', (cache_key, prediction_json, expires_at))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error caching prediction: {e}")
    
    def _get_cached_prediction(self, cache_key: str) -> Optional[BandwidthPrediction]:
        """Get cached prediction if available and not expired"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT prediction_json FROM prediction_cache 
                    WHERE cache_key = ? AND expires_at > datetime('now')
                ''', (cache_key,))
                
                result = cursor.fetchone()
                if result:
                    data = json.loads(result[0])
                    return BandwidthPrediction(
                        predicted_bandwidth=data['predicted_bandwidth'],
                        confidence_interval=tuple(data['confidence_interval']),
                        prediction_confidence=data['prediction_confidence'],
                        optimal_time_window=None,
                        alternative_routes=[],
                        congestion_forecast=data['congestion_forecast'],
                        recommendation=data['recommendation']
                    )
                    
        except Exception as e:
            logger.error(f"Error getting cached prediction: {e}")
        
        return None
    
    def _check_retraining_needed(self):
        """Check if model retraining is needed"""
        if self.last_training_time is None:
            self.should_retrain = True
            return
        
        # Retrain every 24 hours or if significant new data
        if (datetime.now() - self.last_training_time).total_seconds() > 86400:
            self.should_retrain = True
    
    def start_background_training(self):
        """Start background training thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_thread = threading.Thread(target=self._background_training)
            self.training_thread.daemon = True
            self.training_thread.start()
    
    def _background_training(self):
        """Background training loop"""
        while True:
            try:
                if self.should_retrain:
                    logger.info("Starting scheduled model retraining...")
                    self.train_models()
                
                # Sleep for 1 hour before checking again
                time.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in background training: {e}")
                time.sleep(3600)
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get current model performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT model_name, mae, mse, r2_score, training_samples
                    FROM training_stats
                    WHERE created_at = (
                        SELECT MAX(created_at) FROM training_stats 
                        WHERE model_name = training_stats.model_name
                    )
                    ORDER BY model_name
                ''')
                
                results = cursor.fetchall()
                performance = {}
                
                for row in results:
                    performance[row[0]] = {
                        'mae': row[1],
                        'mse': row[2],
                        'r2_score': row[3],
                        'training_samples': row[4]
                    }
                
                return performance
                
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}

# HTTP API Server
class BandwidthPredictorAPI:
    """HTTP API for bandwidth prediction service"""
    
    def __init__(self, predictor: BandwidthPredictor, host: str = "0.0.0.0", port: int = 8080):
        self.predictor = predictor
        self.host = host
        self.port = port
        self.app = None
    
    def setup_routes(self):
        """Setup HTTP routes for the API"""
        from aiohttp import web
        
        self.app = web.Application()
        self.app.router.add_post('/predict', self.handle_predict)
        self.app.router.add_post('/metrics', self.handle_metrics)
        self.app.router.add_post('/workload', self.handle_workload)
        self.app.router.add_get('/performance', self.handle_performance)
        self.app.router.add_get('/health', self.handle_health)
        
        return self.app
    
    async def handle_predict(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction request"""
        try:
            # Parse request
            workload_chars = WorkloadCharacteristics(
                vm_id=request_data['workload']['vm_id'],
                workload_type=request_data['workload']['workload_type'],
                cpu_cores=request_data['workload']['cpu_cores'],
                memory_gb=request_data['workload']['memory_gb'],
                storage_gb=request_data['workload']['storage_gb'],
                network_intensive=request_data['workload']['network_intensive'],
                expected_connections=request_data['workload']['expected_connections'],
                data_transfer_pattern=request_data['workload']['data_transfer_pattern'],
                peak_hours=request_data['workload'].get('peak_hours', []),
                historical_bandwidth=request_data['workload']['historical_bandwidth']
            )
            
            pred_request = PredictionRequest(
                source_node=request_data['source_node'],
                target_node=request_data['target_node'],
                workload_chars=workload_chars,
                time_horizon_hours=request_data.get('time_horizon_hours', 24),
                confidence_level=request_data.get('confidence_level', 0.95),
                include_uncertainty=request_data.get('include_uncertainty', True)
            )
            
            # Get prediction
            prediction = self.predictor.predict_bandwidth(pred_request)
            
            # Format response
            return {
                'success': True,
                'prediction': {
                    'predicted_bandwidth': prediction.predicted_bandwidth,
                    'confidence_interval': prediction.confidence_interval,
                    'prediction_confidence': prediction.prediction_confidence,
                    'optimal_time_window': [
                        prediction.optimal_time_window[0].isoformat() if prediction.optimal_time_window else None,
                        prediction.optimal_time_window[1].isoformat() if prediction.optimal_time_window else None
                    ],
                    'alternative_routes': prediction.alternative_routes,
                    'congestion_forecast': prediction.congestion_forecast,
                    'recommendation': prediction.recommendation
                }
            }
            
        except Exception as e:
            logger.error(f"Error handling prediction request: {e}")
            return {'success': False, 'error': str(e)}
    
    async def handle_metrics(self, request):
        """Handle metrics storage request"""
        try:
            data = await request.json()
            metrics = NetworkMetrics(
                timestamp=datetime.fromisoformat(data['timestamp']) if isinstance(data['timestamp'], str) else data['timestamp'],
                source_node=data['source_node'],
                target_node=data['target_node'],
                bandwidth_mbps=data['bandwidth_mbps'],
                latency_ms=data['latency_ms'],
                packet_loss=data['packet_loss'],
                jitter_ms=data['jitter_ms'],
                throughput_mbps=data['throughput_mbps'],
                connection_quality=data['connection_quality'],
                route_hops=data['route_hops'],
                congestion_level=data['congestion_level']
            )
            
            success = self.predictor.store_network_metrics(metrics)
            
            from aiohttp import web
            return web.json_response({'success': success})
            
        except Exception as e:
            logger.error(f"Error handling metrics request: {e}")
            from aiohttp import web
            return web.json_response({'success': False, 'error': str(e)}, status=400)
    
    async def handle_workload(self, request):
        """Handle workload characteristics storage"""
        try:
            data = await request.json()
            workload = WorkloadCharacteristics(
                vm_id=data['vm_id'],
                workload_type=data['workload_type'],
                cpu_cores=data['cpu_cores'],
                memory_gb=data['memory_gb'],
                storage_gb=data['storage_gb'],
                network_intensive=data['network_intensive'],
                expected_connections=data['expected_connections'],
                data_transfer_pattern=data['data_transfer_pattern'],
                peak_hours=data.get('peak_hours', []),
                historical_bandwidth=data['historical_bandwidth']
            )
            
            success = self.predictor.store_workload_characteristics(workload)
            
            from aiohttp import web
            return web.json_response({'success': success})
            
        except Exception as e:
            logger.error(f"Error handling workload request: {e}")
            from aiohttp import web
            return web.json_response({'success': False, 'error': str(e)}, status=400)
    
    async def handle_performance(self, request):
        """Handle model performance request"""
        try:
            performance = self.predictor.get_model_performance()
            
            from aiohttp import web
            return web.json_response({'models': performance})
            
        except Exception as e:
            logger.error(f"Error handling performance request: {e}")
            from aiohttp import web
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_health(self, request):
        """Handle health check request"""
        from aiohttp import web
        return web.json_response({'status': 'healthy', 'is_trained': self.predictor.is_trained})
    
    def run(self):
        """Run the HTTP API server"""
        from aiohttp import web
        
        logger.info(f"Starting Bandwidth Predictor API on {self.host}:{self.port}")
        
        app = self.setup_routes()
        web.run_app(app, host=self.host, port=self.port)

# Example usage
def main():
    """Main function for testing"""
    logger.info("Initializing Bandwidth Predictor...")
    
    # Initialize predictor
    predictor = BandwidthPredictor()
    
    # Generate some sample training data
    for i in range(1000):
        metrics = NetworkMetrics(
            timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 43200)),  # Last 30 days
            source_node=f"node_{np.random.randint(1, 10)}",
            target_node=f"node_{np.random.randint(1, 10)}",
            bandwidth_mbps=np.random.uniform(10, 500),
            latency_ms=np.random.uniform(1, 100),
            packet_loss=np.random.uniform(0, 0.05),
            jitter_ms=np.random.uniform(0, 10),
            throughput_mbps=np.random.uniform(10, 450),
            connection_quality=np.random.uniform(0.5, 1.0),
            route_hops=np.random.randint(1, 10),
            congestion_level=np.random.uniform(0, 1)
        )
        predictor.store_network_metrics(metrics)
        
        if i % 5 == 0:  # Store workload characteristics every 5 metrics
            workload = WorkloadCharacteristics(
                vm_id=f"vm_{i}",
                workload_type=np.random.choice(['interactive', 'batch', 'streaming', 'compute']),
                cpu_cores=np.random.randint(1, 16),
                memory_gb=np.random.uniform(1, 32),
                storage_gb=np.random.uniform(10, 500),
                network_intensive=np.random.choice([True, False]),
                expected_connections=np.random.randint(1, 50),
                data_transfer_pattern=np.random.choice(['burst', 'steady', 'periodic']),
                peak_hours=[8, 9, 10, 16, 17, 18],
                historical_bandwidth=np.random.uniform(10, 200)
            )
            predictor.store_workload_characteristics(workload)
    
    # Train models
    logger.info("Training models...")
    predictor.train_models()
    
    # Start background training
    predictor.start_background_training()
    
    # Test prediction
    test_workload = WorkloadCharacteristics(
        vm_id="test_vm",
        workload_type="interactive",
        cpu_cores=4,
        memory_gb=8.0,
        storage_gb=100.0,
        network_intensive=True,
        expected_connections=10,
        data_transfer_pattern="burst",
        peak_hours=[8, 9, 10, 16, 17, 18],
        historical_bandwidth=150.0
    )
    
    test_request = PredictionRequest(
        source_node="node_1",
        target_node="node_5",
        workload_chars=test_workload,
        time_horizon_hours=24,
        confidence_level=0.95,
        include_uncertainty=True
    )
    
    logger.info("Making test prediction...")
    result = predictor.predict_bandwidth(test_request)
    
    logger.info(f"Prediction result:")
    logger.info(f"  Predicted bandwidth: {result.predicted_bandwidth:.2f} Mbps")
    logger.info(f"  Confidence interval: {result.confidence_interval}")
    logger.info(f"  Prediction confidence: {result.prediction_confidence:.3f}")
    logger.info(f"  Recommendation: {result.recommendation}")
    
    # Get model performance
    performance = predictor.get_model_performance()
    logger.info(f"Model performance: {performance}")

if __name__ == "__main__":
    main()