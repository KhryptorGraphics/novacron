"""
Model Registry
=============

Model versioning and registry management for MLE-Star:
- Model artifact storage and versioning
- Metadata tracking and lineage
- Model deployment lifecycle management
- Performance tracking across versions
- Model governance and compliance
"""

import logging
import os
import json
import pickle
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import shutil

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except ImportError:
    SQLITE_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Model metadata structure"""
    model_id: str
    name: str
    version: str
    created_at: datetime
    framework: str
    task_type: str
    performance_metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_data_info: Dict[str, Any]
    model_size: int  # in bytes
    checksum: str
    tags: List[str]
    description: str
    author: str
    stage: str = "staging"  # staging, production, archived
    parent_model_id: Optional[str] = None


@dataclass
class ModelRegistryConfig:
    """Configuration for model registry"""
    # Storage configuration
    registry_path: str = "./model_registry"
    database_path: str = "registry.db"
    
    # Model storage
    model_storage_backend: str = "filesystem"  # filesystem, s3, gcs
    compress_models: bool = True
    model_format: str = "joblib"  # joblib, pickle
    
    # Versioning
    versioning_strategy: str = "semantic"  # semantic, timestamp, incremental
    auto_increment_version: bool = True
    
    # Metadata tracking
    track_lineage: bool = True
    track_experiments: bool = True
    store_training_code: bool = False
    
    # Performance tracking
    track_performance_history: bool = True
    performance_degradation_threshold: float = 0.05
    
    # Governance
    require_approval_for_production: bool = True
    max_models_per_stage: int = 5
    retention_policy_days: int = 365
    
    # Integration
    mlflow_tracking_uri: Optional[str] = None
    wandb_project: Optional[str] = None


class ModelStorage:
    """Model artifact storage backend"""
    
    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self.base_path = Path(config.registry_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def store_model(self, model: Any, model_id: str, version: str) -> Tuple[str, str]:
        """Store model artifact and return path and checksum"""
        try:
            # Create version directory
            model_dir = self.base_path / model_id / version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine file path
            if self.config.model_format == "joblib" and JOBLIB_AVAILABLE:
                model_path = model_dir / "model.joblib"
                joblib.dump(model, model_path, compress=self.config.compress_models)
            else:
                model_path = model_dir / "model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Calculate checksum
            checksum = self._calculate_checksum(model_path)
            
            # Store checksum
            with open(model_dir / "checksum.txt", 'w') as f:
                f.write(checksum)
            
            return str(model_path), checksum
            
        except Exception as e:
            logger.error(f"Failed to store model {model_id}:{version}: {e}")
            raise
    
    def load_model(self, model_id: str, version: str) -> Any:
        """Load model artifact"""
        try:
            model_dir = self.base_path / model_id / version
            
            # Try joblib first, then pickle
            joblib_path = model_dir / "model.joblib"
            pickle_path = model_dir / "model.pkl"
            
            if joblib_path.exists() and JOBLIB_AVAILABLE:
                return joblib.load(joblib_path)
            elif pickle_path.exists():
                with open(pickle_path, 'rb') as f:
                    return pickle.load(f)
            else:
                raise FileNotFoundError(f"Model not found: {model_id}:{version}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_id}:{version}: {e}")
            raise
    
    def delete_model(self, model_id: str, version: str) -> bool:
        """Delete model artifact"""
        try:
            model_dir = self.base_path / model_id / version
            if model_dir.exists():
                shutil.rmtree(model_dir)
                
                # Remove parent directory if empty
                parent_dir = model_dir.parent
                if parent_dir.exists() and not list(parent_dir.iterdir()):
                    parent_dir.rmdir()
                
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}:{version}: {e}")
            return False
    
    def model_exists(self, model_id: str, version: str) -> bool:
        """Check if model exists"""
        model_dir = self.base_path / model_id / version
        joblib_path = model_dir / "model.joblib"
        pickle_path = model_dir / "model.pkl"
        
        return joblib_path.exists() or pickle_path.exists()
    
    def get_model_size(self, model_id: str, version: str) -> int:
        """Get model size in bytes"""
        try:
            model_dir = self.base_path / model_id / version
            total_size = 0
            
            for file_path in model_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size
            
        except Exception as e:
            logger.error(f"Failed to get model size {model_id}:{version}: {e}")
            return 0
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


class ModelMetadataStore:
    """Model metadata storage and retrieval"""
    
    def __init__(self, config: ModelRegistryConfig):
        self.config = config
        self.db_path = Path(config.registry_path) / config.database_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        if not SQLITE_AVAILABLE:
            raise RuntimeError("SQLite not available for metadata storage")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS models (
                        model_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        version TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        framework TEXT,
                        task_type TEXT,
                        performance_metrics TEXT,
                        hyperparameters TEXT,
                        training_data_info TEXT,
                        model_size INTEGER,
                        checksum TEXT,
                        tags TEXT,
                        description TEXT,
                        author TEXT,
                        stage TEXT DEFAULT 'staging',
                        parent_model_id TEXT,
                        PRIMARY KEY (model_id, version)
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS model_lineage (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        version TEXT NOT NULL,
                        parent_model_id TEXT,
                        parent_version TEXT,
                        relationship_type TEXT,
                        created_at TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_id TEXT NOT NULL,
                        version TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        recorded_at TEXT NOT NULL,
                        environment TEXT
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def store_metadata(self, metadata: ModelMetadata) -> bool:
        """Store model metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO models (
                        model_id, name, version, created_at, framework, task_type,
                        performance_metrics, hyperparameters, training_data_info,
                        model_size, checksum, tags, description, author, stage,
                        parent_model_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.name,
                    metadata.version,
                    metadata.created_at.isoformat(),
                    metadata.framework,
                    metadata.task_type,
                    json.dumps(metadata.performance_metrics),
                    json.dumps(metadata.hyperparameters),
                    json.dumps(metadata.training_data_info),
                    metadata.model_size,
                    metadata.checksum,
                    json.dumps(metadata.tags),
                    metadata.description,
                    metadata.author,
                    metadata.stage,
                    metadata.parent_model_id
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to store metadata for {metadata.model_id}:{metadata.version}: {e}")
            return False
    
    def get_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Retrieve model metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM models WHERE model_id = ? AND version = ?
                """, (model_id, version))
                
                row = cursor.fetchone()
                if row:
                    return ModelMetadata(
                        model_id=row['model_id'],
                        name=row['name'],
                        version=row['version'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        framework=row['framework'],
                        task_type=row['task_type'],
                        performance_metrics=json.loads(row['performance_metrics'] or '{}'),
                        hyperparameters=json.loads(row['hyperparameters'] or '{}'),
                        training_data_info=json.loads(row['training_data_info'] or '{}'),
                        model_size=row['model_size'],
                        checksum=row['checksum'],
                        tags=json.loads(row['tags'] or '[]'),
                        description=row['description'],
                        author=row['author'],
                        stage=row['stage'],
                        parent_model_id=row['parent_model_id']
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get metadata for {model_id}:{version}: {e}")
            return None
    
    def list_models(self, stage: Optional[str] = None, 
                   name_pattern: Optional[str] = None) -> List[ModelMetadata]:
        """List models with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM models"
                params = []
                conditions = []
                
                if stage:
                    conditions.append("stage = ?")
                    params.append(stage)
                
                if name_pattern:
                    conditions.append("name LIKE ?")
                    params.append(f"%{name_pattern}%")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY created_at DESC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                models = []
                for row in rows:
                    metadata = ModelMetadata(
                        model_id=row['model_id'],
                        name=row['name'],
                        version=row['version'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        framework=row['framework'],
                        task_type=row['task_type'],
                        performance_metrics=json.loads(row['performance_metrics'] or '{}'),
                        hyperparameters=json.loads(row['hyperparameters'] or '{}'),
                        training_data_info=json.loads(row['training_data_info'] or '{}'),
                        model_size=row['model_size'],
                        checksum=row['checksum'],
                        tags=json.loads(row['tags'] or '[]'),
                        description=row['description'],
                        author=row['author'],
                        stage=row['stage'],
                        parent_model_id=row['parent_model_id']
                    )
                    models.append(metadata)
                
                return models
                
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def delete_metadata(self, model_id: str, version: str) -> bool:
        """Delete model metadata"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM models WHERE model_id = ? AND version = ?
                """, (model_id, version))
                
                deleted = cursor.rowcount > 0
                conn.commit()
                return deleted
                
        except Exception as e:
            logger.error(f"Failed to delete metadata for {model_id}:{version}: {e}")
            return False
    
    def update_stage(self, model_id: str, version: str, stage: str) -> bool:
        """Update model stage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE models SET stage = ? WHERE model_id = ? AND version = ?
                """, (stage, model_id, version))
                
                updated = cursor.rowcount > 0
                conn.commit()
                return updated
                
        except Exception as e:
            logger.error(f"Failed to update stage for {model_id}:{version}: {e}")
            return False


class ModelRegistry(BaseEnhancement):
    """Model Registry enhancement for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="model_registry",
            version="1.0.0",
            enabled=SQLITE_AVAILABLE,
            priority=15,
            parameters={
                "registry_path": "./model_registry",
                "database_path": "registry.db",
                "model_storage_backend": "filesystem",
                "compress_models": True,
                "model_format": "joblib" if JOBLIB_AVAILABLE else "pickle",
                "versioning_strategy": "semantic",
                "auto_increment_version": True,
                "track_lineage": True,
                "track_experiments": True,
                "store_training_code": False,
                "track_performance_history": True,
                "performance_degradation_threshold": 0.05,
                "require_approval_for_production": True,
                "max_models_per_stage": 5,
                "retention_policy_days": 365,
                "mlflow_tracking_uri": None,
                "wandb_project": None
            }
        )
    
    def initialize(self) -> bool:
        """Initialize Model Registry"""
        if not SQLITE_AVAILABLE:
            self._logger.error("SQLite not available for model registry")
            return False
        
        try:
            # Create configuration
            self.registry_config = ModelRegistryConfig(**self.config.parameters)
            
            # Initialize storage and metadata store
            self.storage = ModelStorage(self.registry_config)
            self.metadata_store = ModelMetadataStore(self.registry_config)
            
            self._logger.info("Model Registry initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Model Registry: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with model registry capabilities"""
        enhanced = workflow.copy()
        
        # Add model registry configuration
        if 'model_registry' not in enhanced:
            enhanced['model_registry'] = {}
        
        enhanced['model_registry'] = {
            'enabled': True,
            'registry_path': self.registry_config.registry_path,
            'versioning': {
                'strategy': self.registry_config.versioning_strategy,
                'auto_increment': self.registry_config.auto_increment_version
            },
            'storage': {
                'backend': self.registry_config.model_storage_backend,
                'format': self.registry_config.model_format,
                'compress': self.registry_config.compress_models
            },
            'governance': {
                'require_approval': self.registry_config.require_approval_for_production,
                'max_models_per_stage': self.registry_config.max_models_per_stage,
                'retention_days': self.registry_config.retention_policy_days
            },
            'tracking': {
                'lineage': self.registry_config.track_lineage,
                'experiments': self.registry_config.track_experiments,
                'performance_history': self.registry_config.track_performance_history
            }
        }
        
        # Enhance MLE-Star stages with registry integration
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 4: Implementation - Add model registration
            if '4_implementation' in stages:
                if 'model_registration' not in stages['4_implementation']:
                    stages['4_implementation']['model_registration'] = [
                        'model_artifact_storage',
                        'metadata_capture',
                        'version_assignment',
                        'lineage_tracking'
                    ]
            
            # Stage 5: Results Evaluation - Add performance tracking
            if '5_results_evaluation' in stages:
                if 'registry_performance_tracking' not in stages['5_results_evaluation']:
                    stages['5_results_evaluation']['registry_performance_tracking'] = [
                        'performance_metrics_storage',
                        'baseline_comparison',
                        'version_performance_analysis'
                    ]
            
            # Stage 7: Deployment Prep - Add deployment management
            if '7_deployment_prep' in stages:
                if 'deployment_registry_management' not in stages['7_deployment_prep']:
                    stages['7_deployment_prep']['deployment_registry_management'] = [
                        'production_approval_workflow',
                        'staging_to_production_promotion',
                        'model_serving_preparation',
                        'rollback_strategy_setup'
                    ]
        
        self._logger.debug("Enhanced workflow with model registry capabilities")
        return enhanced
    
    def register_model(self, model: Any, name: str, 
                      performance_metrics: Dict[str, float],
                      hyperparameters: Dict[str, Any] = None,
                      training_data_info: Dict[str, Any] = None,
                      description: str = "",
                      tags: List[str] = None,
                      author: str = "",
                      framework: str = "sklearn",
                      task_type: str = "classification",
                      version: Optional[str] = None,
                      parent_model_id: Optional[str] = None) -> Optional[str]:
        """Register a model in the registry"""
        try:
            self._logger.info(f"Registering model: {name}")
            
            # Generate model ID and version
            model_id = self._generate_model_id(name)
            if version is None:
                version = self._generate_version(model_id)
            
            # Store model artifact
            model_path, checksum = self.storage.store_model(model, model_id, version)
            model_size = self.storage.get_model_size(model_id, version)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                created_at=datetime.now(),
                framework=framework,
                task_type=task_type,
                performance_metrics=performance_metrics or {},
                hyperparameters=hyperparameters or {},
                training_data_info=training_data_info or {},
                model_size=model_size,
                checksum=checksum,
                tags=tags or [],
                description=description,
                author=author,
                parent_model_id=parent_model_id
            )
            
            # Store metadata
            success = self.metadata_store.store_metadata(metadata)
            
            if success:
                self._logger.info(f"Successfully registered model {model_id}:{version}")
                return f"{model_id}:{version}"
            else:
                # Clean up stored model if metadata storage failed
                self.storage.delete_model(model_id, version)
                return None
                
        except Exception as e:
            self._logger.error(f"Failed to register model: {e}")
            return None
    
    def load_model(self, model_id: str, version: str) -> Optional[Any]:
        """Load a model from the registry"""
        try:
            # Verify model exists in metadata
            metadata = self.metadata_store.get_metadata(model_id, version)
            if not metadata:
                self._logger.error(f"Model not found in registry: {model_id}:{version}")
                return None
            
            # Load model artifact
            model = self.storage.load_model(model_id, version)
            
            self._logger.info(f"Successfully loaded model {model_id}:{version}")
            return model
            
        except Exception as e:
            self._logger.error(f"Failed to load model {model_id}:{version}: {e}")
            return None
    
    def get_model_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Get model metadata"""
        return self.metadata_store.get_metadata(model_id, version)
    
    def list_models(self, stage: Optional[str] = None, 
                   name_pattern: Optional[str] = None) -> List[ModelMetadata]:
        """List models in the registry"""
        return self.metadata_store.list_models(stage, name_pattern)
    
    def promote_model(self, model_id: str, version: str, target_stage: str) -> bool:
        """Promote model to a different stage"""
        try:
            # Check if model exists
            metadata = self.metadata_store.get_metadata(model_id, version)
            if not metadata:
                self._logger.error(f"Model not found: {model_id}:{version}")
                return False
            
            # Apply governance rules
            if target_stage == "production" and self.registry_config.require_approval_for_production:
                # In a real implementation, this would trigger an approval workflow
                self._logger.info(f"Production promotion requires approval for {model_id}:{version}")
            
            # Check stage limits
            if target_stage != "archived":
                current_stage_models = self.list_models(stage=target_stage)
                if len(current_stage_models) >= self.registry_config.max_models_per_stage:
                    self._logger.warning(f"Stage '{target_stage}' has reached maximum capacity")
            
            # Update stage
            success = self.metadata_store.update_stage(model_id, version, target_stage)
            
            if success:
                self._logger.info(f"Successfully promoted {model_id}:{version} to {target_stage}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to promote model {model_id}:{version}: {e}")
            return False
    
    def delete_model(self, model_id: str, version: str) -> bool:
        """Delete model from registry"""
        try:
            # Delete metadata first
            metadata_deleted = self.metadata_store.delete_metadata(model_id, version)
            
            # Delete model artifact
            artifact_deleted = self.storage.delete_model(model_id, version)
            
            success = metadata_deleted or artifact_deleted
            
            if success:
                self._logger.info(f"Successfully deleted model {model_id}:{version}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to delete model {model_id}:{version}: {e}")
            return False
    
    def _generate_model_id(self, name: str) -> str:
        """Generate unique model ID from name"""
        # Simple implementation - in production, this could be more sophisticated
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name.lower())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{clean_name}_{timestamp}"
    
    def _generate_version(self, model_id: str) -> str:
        """Generate version number for model"""
        if self.registry_config.versioning_strategy == "timestamp":
            return datetime.now().strftime("%Y%m%d_%H%M%S")
        elif self.registry_config.versioning_strategy == "incremental":
            # Find existing versions and increment
            existing_models = self.metadata_store.list_models()
            existing_versions = [m.version for m in existing_models if m.model_id == model_id]
            
            if not existing_versions:
                return "v1"
            
            # Extract numeric part and increment
            numeric_versions = []
            for v in existing_versions:
                if v.startswith('v') and v[1:].isdigit():
                    numeric_versions.append(int(v[1:]))
            
            if numeric_versions:
                return f"v{max(numeric_versions) + 1}"
            else:
                return f"v{len(existing_versions) + 1}"
        else:
            # Default semantic versioning
            return "1.0.0"
    
    def get_registry_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        try:
            all_models = self.list_models()
            
            stats = {
                'total_models': len(all_models),
                'models_by_stage': {},
                'models_by_framework': {},
                'models_by_task_type': {},
                'total_size_mb': 0,
                'latest_registration': None
            }
            
            for model in all_models:
                # Count by stage
                stage = model.stage
                stats['models_by_stage'][stage] = stats['models_by_stage'].get(stage, 0) + 1
                
                # Count by framework
                framework = model.framework
                stats['models_by_framework'][framework] = stats['models_by_framework'].get(framework, 0) + 1
                
                # Count by task type
                task_type = model.task_type
                stats['models_by_task_type'][task_type] = stats['models_by_task_type'].get(task_type, 0) + 1
                
                # Sum up sizes
                stats['total_size_mb'] += model.model_size / (1024 * 1024)
                
                # Track latest registration
                if stats['latest_registration'] is None or model.created_at > stats['latest_registration']:
                    stats['latest_registration'] = model.created_at
            
            # Round size
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            
            return stats
            
        except Exception as e:
            self._logger.error(f"Failed to get registry statistics: {e}")
            return {}