"""
Predictive failure detection system for NovaCron.

Implements ML models to predict hardware failures 15-30 minutes in advance
using time-series analysis, anomaly detection, and ensemble methods.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

from ..models.base import BaseMLModel, ModelMetadata, ModelType, PredictionRequest, PredictionResponse
from ..utils.metrics import MetricsCalculator
from ..utils.feature_engineering import TimeSeriesFeatureExtractor
from ..database.models import FailurePredictionRecord


logger = logging.getLogger(__name__)


class FailurePredictionModel(BaseMLModel):
    """Advanced failure prediction model using ensemble methods and time-series features."""
    
    def __init__(self, model_metadata: ModelMetadata):
        """Initialize failure prediction model."""
        super().__init__(model_metadata)
        
        # Ensemble components
        self._xgb_model: Optional[xgb.XGBClassifier] = None
        self._rf_model: Optional[RandomForestClassifier] = None
        self._isolation_forest: Optional[IsolationForest] = None
        
        # Feature preprocessing
        self._scaler: Optional[StandardScaler] = None
        self._feature_selector: Optional[Any] = None
        
        # Time-series feature extraction
        self._ts_extractor = TimeSeriesFeatureExtractor()
        
        # Model configuration
        self._ensemble_weights = {'xgb': 0.5, 'rf': 0.3, 'isolation': 0.2}
        self._prediction_horizon = 30  # minutes
        self._lookback_window = 24  # hours
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, float]:
        """
        Train the failure prediction model.
        
        Args:
            X: Training features with time-series data
            y: Binary failure labels (1 = failure within prediction horizon, 0 = normal)
            validation_data: Optional validation dataset
            
        Returns:
            Training metrics
        """
        logger.info(f"Training failure prediction model with {len(X)} samples")
        start_time = datetime.utcnow()
        
        try:
            # Extract time-series features
            logger.info("Extracting time-series features...")
            X_features = self._extract_ts_features(X)
            
            # Split data if validation not provided
            if validation_data is None:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_features, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, y_train = X_features, y
                X_val, y_val = self._extract_ts_features(validation_data[0]), validation_data[1]
            
            # Scale features
            self._scaler = RobustScaler()
            X_train_scaled = self._scaler.fit_transform(X_train)
            X_val_scaled = self._scaler.transform(X_val)
            
            # Feature selection based on mutual information
            logger.info("Performing feature selection...")
            X_train_selected, selected_features = self._select_features(
                pd.DataFrame(X_train_scaled, columns=X_train.columns), y_train
            )
            X_val_selected = X_val_scaled[:, selected_features]
            
            # Store feature names
            self._feature_names = X_train.columns[selected_features].tolist()
            
            # Train ensemble models
            metrics = self._train_ensemble(
                X_train_selected, y_train, X_val_selected, y_val
            )
            
            # Update metadata
            training_duration = (datetime.utcnow() - start_time).total_seconds()
            self.update_metadata(
                training_status="completed",
                trained_at=datetime.utcnow(),
                training_duration=training_duration,
                training_samples=len(X_train),
                validation_samples=len(X_val),
                feature_count=len(self._feature_names),
                **metrics
            )
            
            self._is_trained = True
            logger.info(f"Model training completed in {training_duration:.2f} seconds")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            self.update_metadata(training_status="failed")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict failure probability.
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (1 = failure predicted, 0 = normal)
        """
        probabilities = self.predict_proba(X)
        threshold = 0.5  # Could be tuned based on business requirements
        return (probabilities[:, 1] > threshold).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get failure probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability matrix [normal_prob, failure_prob]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features and preprocess
        X_features = self._extract_ts_features(X)
        X_scaled = self._scaler.transform(X_features)
        X_selected = X_scaled[:, [X_features.columns.get_loc(col) for col in self._feature_names]]
        
        # Get ensemble predictions
        xgb_proba = self._xgb_model.predict_proba(X_selected)
        rf_proba = self._rf_model.predict_proba(X_selected)
        isolation_scores = self._isolation_forest.decision_function(X_selected)
        
        # Convert isolation forest scores to probabilities
        isolation_proba = self._scores_to_proba(isolation_scores)
        
        # Weighted ensemble
        ensemble_proba = (
            self._ensemble_weights['xgb'] * xgb_proba +
            self._ensemble_weights['rf'] * rf_proba +
            self._ensemble_weights['isolation'] * isolation_proba
        )
        
        return ensemble_proba
    
    def _extract_ts_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Extract time-series features from raw metrics data."""
        # Assume X contains columns: timestamp, node_id, cpu_usage, memory_usage, 
        # disk_usage, network_usage, temperature, etc.
        
        if 'timestamp' not in X.columns:
            # Add synthetic timestamps if not present
            X = X.copy()
            X['timestamp'] = pd.date_range(
                start=datetime.now() - timedelta(hours=self._lookback_window),
                periods=len(X),
                freq='1T'  # 1 minute intervals
            )
        
        # Use tsfresh for automatic feature extraction
        ts_features = self._ts_extractor.extract_features(
            X, column_id='node_id', column_sort='timestamp'
        )
        
        return ts_features
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Select most informative features."""
        # Use tsfresh feature selection
        features_filtered = select_features(X, y)
        selected_indices = [X.columns.get_loc(col) for col in features_filtered.columns]
        
        return features_filtered.values, np.array(selected_indices)
    
    def _train_ensemble(self, X_train: np.ndarray, y_train: pd.Series, 
                       X_val: np.ndarray, y_val: pd.Series) -> Dict[str, float]:
        """Train ensemble of models."""
        
        # XGBoost Classifier
        logger.info("Training XGBoost model...")
        self._xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='auc',
            early_stopping_rounds=20
        )
        
        self._xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Random Forest Classifier
        logger.info("Training Random Forest model...")
        self._rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        self._rf_model.fit(X_train, y_train)
        
        # Isolation Forest for anomaly detection
        logger.info("Training Isolation Forest...")
        self._isolation_forest = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        self._isolation_forest.fit(X_train)
        
        # Calculate validation metrics
        metrics = self._calculate_metrics(X_val, y_val)
        
        return metrics
    
    def _calculate_metrics(self, X_val: np.ndarray, y_val: pd.Series) -> Dict[str, float]:
        """Calculate validation metrics."""
        y_pred = self.predict(pd.DataFrame(X_val))
        y_proba = self.predict_proba(pd.DataFrame(X_val))[:, 1]
        
        # Classification metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1_score': f1_score(y_val, y_pred, zero_division=0),
            'auc_score': roc_auc_score(y_val, y_proba)
        }
        
        return metrics
    
    def _scores_to_proba(self, scores: np.ndarray) -> np.ndarray:
        """Convert isolation forest scores to probabilities."""
        # Normalize scores to [0, 1] range
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Convert to probability format [normal_prob, failure_prob]
        failure_prob = 1 - scores_norm
        normal_prob = scores_norm
        
        return np.column_stack([normal_prob, failure_prob])
    
    def save_model(self, filepath: str) -> None:
        """Save the ensemble model."""
        import joblib
        
        model_data = {
            'xgb_model': self._xgb_model,
            'rf_model': self._rf_model,
            'isolation_forest': self._isolation_forest,
            'scaler': self._scaler,
            'feature_names': self._feature_names,
            'ensemble_weights': self._ensemble_weights,
            'metadata': self.metadata.dict()
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load the ensemble model."""
        import joblib
        
        model_data = joblib.load(filepath)
        
        self._xgb_model = model_data['xgb_model']
        self._rf_model = model_data['rf_model']
        self._isolation_forest = model_data['isolation_forest']
        self._scaler = model_data['scaler']
        self._feature_names = model_data['feature_names']
        self._ensemble_weights = model_data['ensemble_weights']
        
        self._is_trained = True
        logger.info(f"Model loaded from {filepath}")


class FailurePredictionService:
    """Service for managing failure prediction models and real-time monitoring."""
    
    def __init__(self, settings):
        """Initialize failure prediction service."""
        self.settings = settings
        self.models: Dict[str, FailurePredictionModel] = {}
        self.active_model: Optional[FailurePredictionModel] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator()
    
    async def initialize(self) -> None:
        """Initialize the service and load models."""
        logger.info("Initializing failure prediction service...")
        
        # Load existing models
        await self._load_models()
        
        # Start monitoring if we have an active model
        if self.active_model:
            await self.start_monitoring()
    
    async def train_model(self, training_data: pd.DataFrame, 
                         model_id: Optional[str] = None) -> FailurePredictionModel:
        """
        Train a new failure prediction model.
        
        Args:
            training_data: Historical data with features and failure labels
            model_id: Optional model identifier
            
        Returns:
            Trained model instance
        """
        if model_id is None:
            model_id = f"failure_pred_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        # Create model metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_type=ModelType.FAILURE_PREDICTION,
            version="1.0.0"
        )
        
        # Create and train model
        model = FailurePredictionModel(metadata)
        
        # Prepare training data
        feature_columns = [col for col in training_data.columns if col != 'failure_label']
        X = training_data[feature_columns]
        y = training_data['failure_label']
        
        # Train the model
        metrics = model.train(X, y)
        
        # Store the model
        self.models[model_id] = model
        
        # Save model to disk
        model_path = f"{self.settings.ml.model_storage_path}/{model_id}.joblib"
        model.save_model(model_path)
        
        logger.info(f"Trained failure prediction model {model_id} with metrics: {metrics}")
        
        return model
    
    async def predict_failures(self, request: PredictionRequest) -> PredictionResponse:
        """
        Predict potential failures for given system metrics.
        
        Args:
            request: Prediction request with system metrics
            
        Returns:
            Failure prediction response
        """
        if not self.active_model or not self.active_model.is_trained:
            raise ValueError("No active trained model available for prediction")
        
        start_time = datetime.utcnow()
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Get prediction probabilities
        probabilities = self.active_model.predict_proba(features_df)
        failure_probability = probabilities[0, 1]
        
        # Determine if failure is predicted
        threshold = self.settings.ml.failure_prediction_threshold
        failure_predicted = failure_probability > threshold
        
        # Calculate confidence
        confidence = max(probabilities[0])
        
        # Prepare response
        response_time = (datetime.utcnow() - start_time).total_seconds()
        
        response = PredictionResponse(
            request_id=request.request_id,
            model_id=self.active_model.metadata.model_id,
            prediction=int(failure_predicted),
            confidence=confidence,
            probabilities={
                'normal': float(probabilities[0, 0]),
                'failure': float(probabilities[0, 1])
            },
            response_time=response_time,
            metadata={
                'failure_probability': failure_probability,
                'threshold': threshold,
                'prediction_horizon_minutes': self.settings.ml.prediction_horizon
            }
        )
        
        # Log high-risk predictions
        if failure_predicted:
            logger.warning(
                f"Failure predicted for request {request.request_id} "
                f"with probability {failure_probability:.3f}"
            )
        
        return response
    
    async def start_monitoring(self) -> None:
        """Start real-time failure monitoring."""
        if self._monitoring_task and not self._monitoring_task.done():
            logger.warning("Monitoring task is already running")
            return
        
        logger.info("Starting failure monitoring...")
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop real-time failure monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            logger.info("Failure monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for continuous failure prediction."""
        while True:
            try:
                # Get current system metrics from NovaCron API
                metrics_data = await self._fetch_current_metrics()
                
                if not metrics_data:
                    await asyncio.sleep(60)  # Wait before retry
                    continue
                
                # Make predictions for each node
                for node_data in metrics_data:
                    request = PredictionRequest(
                        request_id=f"monitor_{node_data['node_id']}_{datetime.utcnow().isoformat()}",
                        features=node_data
                    )
                    
                    try:
                        response = await self.predict_failures(request)
                        
                        # Store prediction result
                        await self._store_prediction(request, response)
                        
                        # Send alerts if failure is predicted
                        if response.prediction == 1:
                            await self._send_failure_alert(node_data, response)
                            
                    except Exception as e:
                        logger.error(f"Prediction failed for node {node_data['node_id']}: {str(e)}")
                
                # Wait before next monitoring cycle
                await asyncio.sleep(self.settings.monitoring.health_check_interval)
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(60)
    
    async def _fetch_current_metrics(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch current system metrics from NovaCron API."""
        # This would integrate with the NovaCron monitoring API
        # For now, return mock data structure
        return None
    
    async def _store_prediction(self, request: PredictionRequest, response: PredictionResponse) -> None:
        """Store prediction result in database."""
        # Implementation would store in PostgreSQL via SQLAlchemy
        pass
    
    async def _send_failure_alert(self, node_data: Dict[str, Any], response: PredictionResponse) -> None:
        """Send failure alert to NovaCron alerting system."""
        logger.critical(
            f"FAILURE ALERT: Node {node_data.get('node_id')} predicted to fail "
            f"within {self.settings.ml.prediction_horizon} minutes "
            f"(confidence: {response.confidence:.3f})"
        )
        # Integration with alerting system would go here
    
    async def _load_models(self) -> None:
        """Load existing models from storage."""
        # Implementation would scan model storage directory and load models
        pass
    
    def set_active_model(self, model_id: str) -> None:
        """Set the active model for predictions."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        self.active_model = self.models[model_id]
        logger.info(f"Active model set to {model_id}")
    
    def get_model_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get metrics for a specific model."""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        return self.models[model_id].metadata.dict()