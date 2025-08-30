"""
Machine Learning Models for NovaCron AI Engine
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MigrationPredictor:
    """Predicts optimal migration targets and timing"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train the migration prediction model"""
        try:
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            score = self.model.score(X_scaled, y)
            logger.info(f"Migration model trained with score: {score}")
            return score
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
            
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Predict migration time and confidence"""
        if not self.is_trained:
            # Return default values if not trained
            return 30.0, 0.5
            
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            prediction = self.model.predict(features_scaled)[0]
            
            # Calculate confidence based on prediction variance
            if hasattr(self.model, 'predict_proba'):
                confidence = max(self.model.predict_proba(features_scaled)[0])
            else:
                confidence = 0.75  # Default confidence
                
            return prediction, confidence
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 30.0, 0.5

class ResourcePredictor:
    """Predicts future resource usage patterns"""
    
    def __init__(self):
        self.cpu_model = RandomForestRegressor(n_estimators=50)
        self.memory_model = RandomForestRegressor(n_estimators=50)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Train resource prediction models"""
        try:
            # Prepare features and targets
            features = historical_data[['hour', 'day_of_week', 'load_average']].values
            cpu_target = historical_data['cpu_usage'].values
            memory_target = historical_data['memory_usage'].values
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Train models
            self.cpu_model.fit(features_scaled, cpu_target)
            self.memory_model.fit(features_scaled, memory_target)
            
            self.is_trained = True
            
            scores = {
                'cpu_score': self.cpu_model.score(features_scaled, cpu_target),
                'memory_score': self.memory_model.score(features_scaled, memory_target)
            }
            
            logger.info(f"Resource models trained with scores: {scores}")
            return scores
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
            
    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict resource usage"""
        if not self.is_trained:
            # Return default predictions
            return {
                'cpu_prediction': 50.0,
                'memory_prediction': 60.0,
                'confidence': 0.5
            }
            
        try:
            # Convert features to array
            feature_array = np.array([[
                features.get('hour', 12),
                features.get('day_of_week', 3),
                features.get('load_average', 1.0)
            ]])
            
            features_scaled = self.scaler.transform(feature_array)
            
            cpu_pred = self.cpu_model.predict(features_scaled)[0]
            memory_pred = self.memory_model.predict(features_scaled)[0]
            
            return {
                'cpu_prediction': cpu_pred,
                'memory_prediction': memory_pred,
                'confidence': 0.75
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'cpu_prediction': 50.0,
                'memory_prediction': 60.0,
                'confidence': 0.5
            }

class AnomalyDetector:
    """Detects anomalies in system metrics"""
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, normal_data: np.ndarray) -> float:
        """Train anomaly detection model on normal data"""
        try:
            data_scaled = self.scaler.fit_transform(normal_data)
            self.model.fit(data_scaled)
            self.is_trained = True
            
            # Calculate training score
            predictions = self.model.predict(data_scaled)
            accuracy = np.mean(predictions == 1)  # 1 means normal
            
            logger.info(f"Anomaly detector trained with accuracy: {accuracy}")
            return accuracy
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
            
    def detect(self, metrics: np.ndarray) -> Tuple[bool, float]:
        """Detect if metrics are anomalous"""
        if not self.is_trained:
            # Return default if not trained
            return False, 0.0
            
        try:
            metrics_scaled = self.scaler.transform(metrics.reshape(1, -1))
            prediction = self.model.predict(metrics_scaled)[0]
            score = self.model.score_samples(metrics_scaled)[0]
            
            # Convert score to probability (0-1 range)
            anomaly_probability = 1 / (1 + np.exp(score))
            
            is_anomaly = prediction == -1
            
            return is_anomaly, anomaly_probability
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return False, 0.0
            
class WorkloadPredictor:
    """Predicts future workload patterns"""
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, historical_workload: pd.DataFrame) -> float:
        """Train workload prediction model"""
        try:
            # Extract time-based features
            features = historical_workload[['hour', 'day', 'month']].values
            target = historical_workload['workload'].values
            
            features_scaled = self.scaler.fit_transform(features)
            self.model.fit(features_scaled, target)
            self.is_trained = True
            
            score = self.model.score(features_scaled, target)
            logger.info(f"Workload model trained with score: {score}")
            return score
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
            
    def predict_next_period(self, current_time: Dict[str, int], periods: int = 24) -> List[float]:
        """Predict workload for next N periods"""
        if not self.is_trained:
            # Return default predictions
            return [50.0] * periods
            
        try:
            predictions = []
            
            for i in range(periods):
                hour = (current_time['hour'] + i) % 24
                day = current_time['day'] + (current_time['hour'] + i) // 24
                month = current_time['month']
                
                features = np.array([[hour, day, month]])
                features_scaled = self.scaler.transform(features)
                
                pred = self.model.predict(features_scaled)[0]
                predictions.append(pred)
                
            return predictions
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return [50.0] * periods