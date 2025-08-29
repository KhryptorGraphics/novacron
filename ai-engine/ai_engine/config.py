"""Configuration management for the AI Operations Engine.

Optimized for Python 3.12 with improved type annotations and validation.
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator, Field
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseSettings(PydanticBaseSettings):  # Python 3.12 compatible
    """Database configuration."""
    
    url: str = "postgresql://postgres:postgres@localhost:11432/novacron"
    pool_size: int = 20
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 3600
    
    class Config:
        env_prefix = "DB_"


class RedisSettings(BaseSettings):
    """Redis configuration for caching and queuing."""
    
    url: str = "redis://localhost:6379/0"
    max_connections: int = 100
    retry_on_timeout: bool = True
    socket_timeout: int = 5
    
    class Config:
        env_prefix = "REDIS_"


class MLSettings(BaseSettings):
    """Machine learning model configuration."""
    
    # Model paths
    model_storage_path: str = "/var/lib/novacron/ai-models"
    model_registry_url: Optional[str] = None
    
    # Training parameters
    train_batch_size: int = 1000
    inference_batch_size: int = 100
    model_update_interval: int = 3600  # seconds
    
    # Feature engineering
    feature_window_size: int = 24  # hours
    prediction_horizon: int = 30  # minutes
    min_samples_for_training: int = 1000
    
    # Performance thresholds
    failure_prediction_threshold: float = 0.7
    anomaly_detection_threshold: float = 0.95
    drift_detection_threshold: float = 0.1
    
    class Config:
        env_prefix = "ML_"


class NovaCronSettings(BaseSettings):
    """NovaCron API integration settings."""
    
    api_url: str = "http://localhost:8090"
    ws_url: str = "ws://localhost:8091"
    api_timeout: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.0
    
    # Authentication
    username: str = "ai-engine"
    password: str = "changeme"
    jwt_secret: Optional[str] = None
    
    class Config:
        env_prefix = "NOVACRON_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    # Prometheus metrics
    metrics_port: int = 9100
    metrics_path: str = "/metrics"
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: Optional[str] = None
    
    # Health checks
    health_check_interval: int = 60
    health_check_timeout: int = 10
    
    class Config:
        env_prefix = "MONITORING_"


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""
    
    secret_key: str = "changeme_in_production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # API security
    enable_cors: bool = True
    allowed_origins: List[str] = ["http://localhost:8092", "http://localhost:3001"]
    api_key_header: str = "X-API-Key"
    
    @validator("secret_key")
    def validate_secret_key(cls, v: str) -> str:
        """Validate secret key in production."""
        if os.getenv("ENVIRONMENT", "development") == "production" and v == "changeme_in_production":
            raise ValueError("Secret key must be set in production")
        return v
    
    class Config:
        env_prefix = "SECURITY_"


class Settings(PydanticBaseSettings):  # Python 3.12 optimized
    """Main application settings."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    testing: bool = False
    
    # API server
    host: str = "0.0.0.0"
    port: int = 8093
    workers: int = 1
    
    # Sub-configurations
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    ml: MLSettings = MLSettings()
    novacron: NovaCronSettings = NovaCronSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    security: SecuritySettings = SecuritySettings()
    
    @validator("debug")
    def set_debug_mode(cls, v: bool, values: dict) -> bool:
        """Set debug mode based on environment."""
        if values.get("environment") == "development":
            return True
        return v
    
    @validator("workers")
    def set_workers_count(cls, v: int, values: dict) -> int:
        """Set workers based on environment."""
        if values.get("environment") == "development":
            return 1
        return max(1, min(v, os.cpu_count() or 1))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings singleton."""
    return Settings()