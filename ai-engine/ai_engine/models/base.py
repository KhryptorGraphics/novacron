"""Base model classes and interfaces for the AI engine."""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Supported model types."""
    
    FAILURE_PREDICTION = "failure_prediction"
    WORKLOAD_PLACEMENT = "workload_placement"
    ANOMALY_DETECTION = "anomaly_detection"
    RESOURCE_OPTIMIZATION = "resource_optimization"


class TrainingStatus(str, Enum):
    """Model training status."""
    
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATING = "validating"


class ModelMetadata(BaseModel):
    """Model metadata and versioning information."""
    
    model_id: str = Field(..., description="Unique model identifier")
    model_type: ModelType = Field(..., description="Type of ML model")
    version: str = Field(..., description="Model version")
    
    # Training information
    training_status: TrainingStatus = TrainingStatus.PENDING
    trained_at: Optional[datetime] = None
    training_duration: Optional[float] = None  # seconds
    
    # Performance metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_score: Optional[float] = None
    
    # Data information
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    feature_count: Optional[int] = None
    
    # Deployment information
    deployed_at: Optional[datetime] = None
    is_active: bool = False
    
    # Additional metadata
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    feature_names: List[str] = Field(default_factory=list)
    target_names: List[str] = Field(default_factory=list)
    
    class Config:
        use_enum_values = True


class PredictionRequest(BaseModel):
    """Base prediction request."""
    
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    """Base prediction response."""
    
    request_id: str = Field(..., description="Request identifier")
    model_id: str = Field(..., description="Model used for prediction")
    prediction: Union[float, int, str, List[float]] = Field(..., description="Model prediction")
    confidence: Optional[float] = Field(None, description="Prediction confidence score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Class probabilities")
    
    # Response metadata
    response_time: float = Field(..., description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseMLModel(ABC):
    """Abstract base class for all ML models in the AI engine."""
    
    def __init__(self, model_metadata: ModelMetadata):
        """Initialize base model with metadata."""
        self.metadata = model_metadata
        self._model: Optional[Any] = None
        self._feature_names: List[str] = []
        self._is_trained: bool = False
    
    @property
    def is_trained(self) -> bool:
        """Check if model is trained and ready for predictions."""
        return self._is_trained and self._model is not None
    
    @property
    def feature_names(self) -> List[str]:
        """Get feature names used by the model."""
        return self._feature_names
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, 
              validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None) -> Dict[str, float]:
        """
        Train the model with provided data.
        
        Args:
            X: Training features
            y: Training targets
            validation_data: Optional validation dataset
            
        Returns:
            Training metrics dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability matrix
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        pass
    
    def validate_input(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and preprocess input data.
        
        Args:
            X: Input features
            
        Returns:
            Validated and preprocessed features
            
        Raises:
            ValueError: If input validation fails
        """
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        
        if not self._feature_names:
            return X
        
        # Check for required features
        missing_features = set(self._feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and reorder features
        X_validated = X[self._feature_names].copy()
        
        # Handle missing values
        if X_validated.isnull().any().any():
            X_validated = X_validated.fillna(X_validated.median())
        
        return X_validated
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            return None
        
        # Default implementation - should be overridden by specific models
        return None
    
    def update_metadata(self, **kwargs: Any) -> None:
        """Update model metadata with provided values."""
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)


class ModelRegistry(ABC):
    """Abstract base class for model registry implementations."""
    
    @abstractmethod
    async def register_model(self, model: BaseMLModel) -> None:
        """Register a new model in the registry."""
        pass
    
    @abstractmethod
    async def get_model(self, model_id: str) -> Optional[BaseMLModel]:
        """Retrieve a model from the registry."""
        pass
    
    @abstractmethod
    async def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """List all registered models."""
        pass
    
    @abstractmethod
    async def update_model(self, model_id: str, metadata: ModelMetadata) -> None:
        """Update model metadata in the registry."""
        pass
    
    @abstractmethod
    async def delete_model(self, model_id: str) -> None:
        """Delete a model from the registry."""
        pass