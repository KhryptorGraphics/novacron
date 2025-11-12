"""
ML Pipeline orchestration for automated training, tuning, and validation.
Supports distributed training and advanced hyperparameter optimization.
"""

import asyncio
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import optuna
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    ADVANCED_FEATURES = True
except ImportError:
    ADVANCED_FEATURES = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Pipeline execution stages"""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    FEATURE_ENGINEERING = "feature_engineering"
    DATA_SPLITTING = "data_splitting"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    MODEL_TRAINING = "model_training"
    MODEL_VALIDATION = "model_validation"
    MODEL_TESTING = "model_testing"
    MODEL_EXPORT = "model_export"
    DEPLOYMENT_PREP = "deployment_prep"


class PipelineStatus(Enum):
    """Pipeline execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline"""
    name: str
    description: str = ""

    # Data configuration
    data_source: str = ""
    target_column: str = ""
    feature_columns: List[str] = field(default_factory=list)

    # Training configuration
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42

    # Model configuration
    model_type: str = "sklearn"  # sklearn, pytorch, tensorflow, xgboost
    model_class: str = ""
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Hyperparameter tuning
    enable_tuning: bool = False
    tuning_method: str = "optuna"  # optuna, grid_search, random_search
    tuning_trials: int = 100
    tuning_timeout: int = 3600  # seconds
    param_space: Dict[str, Any] = field(default_factory=dict)

    # Distributed training
    enable_distributed: bool = False
    num_workers: int = 1
    backend: str = "gloo"  # gloo, nccl, mpi

    # Feature engineering
    feature_transforms: List[Dict[str, Any]] = field(default_factory=list)
    enable_auto_feature_engineering: bool = False

    # Validation
    cross_validation_folds: int = 5
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    # Output
    output_dir: str = "./ml_pipeline_output"
    save_intermediate: bool = True

    # Advanced options
    early_stopping: bool = True
    early_stopping_patience: int = 10
    checkpoint_interval: int = 10
    enable_logging: bool = True


@dataclass
class PipelineRun:
    """Represents a pipeline execution run"""
    run_id: str
    pipeline_name: str
    config: PipelineConfig
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    stages_completed: List[PipelineStage] = field(default_factory=list)
    current_stage: Optional[PipelineStage] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)


class FeatureEngineer:
    """Automated feature engineering"""

    def __init__(self):
        self.transforms = []
        self.feature_stats = {}

    def fit_transform(self, df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
        """Apply feature engineering transforms"""
        logger.info("Starting feature engineering...")

        # Calculate feature statistics
        self.feature_stats = {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "missing": df.isnull().sum().to_dict(),
        }

        # Apply configured transforms
        for transform in config.feature_transforms:
            df = self._apply_transform(df, transform)

        # Auto feature engineering if enabled
        if config.enable_auto_feature_engineering:
            df = self._auto_feature_engineering(df)

        return df

    def _apply_transform(self, df: pd.DataFrame, transform: Dict[str, Any]) -> pd.DataFrame:
        """Apply a single transform"""
        transform_type = transform.get("type")

        if transform_type == "scale":
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            columns = transform.get("columns", df.select_dtypes(include=[np.number]).columns)
            df[columns] = scaler.fit_transform(df[columns])

        elif transform_type == "encode":
            from sklearn.preprocessing import LabelEncoder
            columns = transform.get("columns", df.select_dtypes(include=["object"]).columns)
            for col in columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        elif transform_type == "polynomial":
            from sklearn.preprocessing import PolynomialFeatures
            degree = transform.get("degree", 2)
            columns = transform.get("columns")
            poly = PolynomialFeatures(degree=degree)
            poly_features = poly.fit_transform(df[columns])
            feature_names = [f"{col}_poly_{i}" for i, col in enumerate(poly.get_feature_names_out())]
            df = pd.concat([df, pd.DataFrame(poly_features, columns=feature_names)], axis=1)

        return df

    def _auto_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatic feature engineering"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Create interaction features
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

        # Create ratio features
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + 1e-8)

        return df


class HyperparameterTuner:
    """Hyperparameter tuning using Optuna or other methods"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.best_params = {}
        self.study = None

    def tune(self, X_train, y_train, X_val, y_val, model_builder: Callable) -> Dict[str, Any]:
        """Run hyperparameter tuning"""
        if not ADVANCED_FEATURES:
            logger.warning("Advanced features not available, skipping tuning")
            return self.config.model_params

        logger.info(f"Starting hyperparameter tuning with {self.config.tuning_method}...")

        if self.config.tuning_method == "optuna":
            return self._optuna_tuning(X_train, y_train, X_val, y_val, model_builder)
        elif self.config.tuning_method == "grid_search":
            return self._grid_search_tuning(X_train, y_train, X_val, y_val, model_builder)
        else:
            return self._random_search_tuning(X_train, y_train, X_val, y_val, model_builder)

    def _optuna_tuning(self, X_train, y_train, X_val, y_val, model_builder: Callable) -> Dict[str, Any]:
        """Optuna-based hyperparameter tuning"""
        def objective(trial):
            # Build parameter suggestions from param space
            params = {}
            for param_name, param_config in self.config.param_space.items():
                if param_config["type"] == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )

            # Build and train model
            model = model_builder(params)
            model.fit(X_train, y_train)

            # Evaluate on validation set
            score = model.score(X_val, y_val)
            return score

        # Create and optimize study
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(
            objective,
            n_trials=self.config.tuning_trials,
            timeout=self.config.tuning_timeout,
            show_progress_bar=True
        )

        self.best_params = self.study.best_params
        logger.info(f"Best parameters: {self.best_params}")
        logger.info(f"Best score: {self.study.best_value}")

        return self.best_params

    def _grid_search_tuning(self, X_train, y_train, X_val, y_val, model_builder: Callable) -> Dict[str, Any]:
        """Grid search hyperparameter tuning"""
        from sklearn.model_selection import GridSearchCV

        model = model_builder(self.config.model_params)
        grid_search = GridSearchCV(
            model,
            self.config.param_space,
            cv=self.config.cross_validation_folds,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.best_params = grid_search.best_params_
        return self.best_params

    def _random_search_tuning(self, X_train, y_train, X_val, y_val, model_builder: Callable) -> Dict[str, Any]:
        """Random search hyperparameter tuning"""
        from sklearn.model_selection import RandomizedSearchCV

        model = model_builder(self.config.model_params)
        random_search = RandomizedSearchCV(
            model,
            self.config.param_space,
            n_iter=self.config.tuning_trials,
            cv=self.config.cross_validation_folds,
            n_jobs=-1
        )
        random_search.fit(X_train, y_train)

        self.best_params = random_search.best_params_
        return self.best_params


class DistributedTrainer:
    """Distributed training support for PyTorch models"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.rank = 0
        self.world_size = config.num_workers

    def setup(self, rank: int, world_size: int):
        """Initialize distributed training"""
        if not ADVANCED_FEATURES:
            logger.warning("PyTorch not available, distributed training disabled")
            return

        self.rank = rank
        self.world_size = world_size

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        dist.init_process_group(
            self.config.backend,
            rank=rank,
            world_size=world_size
        )

    def cleanup(self):
        """Cleanup distributed training"""
        if ADVANCED_FEATURES and dist.is_initialized():
            dist.destroy_process_group()

    def train(self, model, train_loader, optimizer, criterion, epochs: int):
        """Distributed training loop"""
        if not ADVANCED_FEATURES:
            return

        # Wrap model in DDP
        model = DDP(model, device_ids=[self.rank])

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.rank), target.to(self.rank)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if batch_idx % 100 == 0 and self.rank == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            if self.rank == 0:
                avg_loss = running_loss / len(train_loader)
                logger.info(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")


class MLPipeline:
    """Main ML pipeline orchestrator"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run = None
        self.feature_engineer = FeatureEngineer()
        self.tuner = HyperparameterTuner(config)
        self.distributed_trainer = DistributedTrainer(config) if config.enable_distributed else None

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    async def execute(self) -> PipelineRun:
        """Execute the complete ML pipeline"""
        run_id = f"run_{int(time.time())}_{hash(self.config.name) % 10000}"
        self.run = PipelineRun(
            run_id=run_id,
            pipeline_name=self.config.name,
            config=self.config,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Execute pipeline stages
            await self._execute_stage(PipelineStage.DATA_LOADING, self._load_data)
            await self._execute_stage(PipelineStage.DATA_VALIDATION, self._validate_data)
            await self._execute_stage(PipelineStage.FEATURE_ENGINEERING, self._engineer_features)
            await self._execute_stage(PipelineStage.DATA_SPLITTING, self._split_data)

            if self.config.enable_tuning:
                await self._execute_stage(PipelineStage.HYPERPARAMETER_TUNING, self._tune_hyperparameters)

            await self._execute_stage(PipelineStage.MODEL_TRAINING, self._train_model)
            await self._execute_stage(PipelineStage.MODEL_VALIDATION, self._validate_model)
            await self._execute_stage(PipelineStage.MODEL_TESTING, self._test_model)
            await self._execute_stage(PipelineStage.MODEL_EXPORT, self._export_model)
            await self._execute_stage(PipelineStage.DEPLOYMENT_PREP, self._prepare_deployment)

            self.run.status = PipelineStatus.COMPLETED
            self.run.end_time = datetime.now()

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self.run.status = PipelineStatus.FAILED
            self.run.errors.append(str(e))
            self.run.end_time = datetime.now()

        finally:
            await self._save_run_metadata()

        return self.run

    async def _execute_stage(self, stage: PipelineStage, func: Callable):
        """Execute a pipeline stage"""
        logger.info(f"Executing stage: {stage.value}")
        self.run.current_stage = stage

        try:
            await asyncio.to_thread(func)
            self.run.stages_completed.append(stage)
            self.run.logs.append(f"Completed stage: {stage.value}")
        except Exception as e:
            error_msg = f"Stage {stage.value} failed: {str(e)}"
            self.run.errors.append(error_msg)
            raise RuntimeError(error_msg)

    def _load_data(self):
        """Load training data"""
        logger.info(f"Loading data from: {self.config.data_source}")

        # Support multiple data sources
        if self.config.data_source.endswith('.csv'):
            self.data = pd.read_csv(self.config.data_source)
        elif self.config.data_source.endswith('.parquet'):
            self.data = pd.read_parquet(self.config.data_source)
        elif self.config.data_source.endswith('.json'):
            self.data = pd.read_json(self.config.data_source)
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")

        logger.info(f"Loaded {len(self.data)} samples with {len(self.data.columns)} features")

        # Save artifact
        if self.config.save_intermediate:
            artifact_path = Path(self.config.output_dir) / "data_raw.parquet"
            self.data.to_parquet(artifact_path)
            self.run.artifacts["data_raw"] = str(artifact_path)

    def _validate_data(self):
        """Validate data quality"""
        logger.info("Validating data quality...")

        # Check for missing target
        if self.config.target_column not in self.data.columns:
            raise ValueError(f"Target column not found: {self.config.target_column}")

        # Check for missing values
        missing_pct = self.data.isnull().sum() / len(self.data) * 100
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            logger.warning(f"Columns with >50% missing: {high_missing.to_dict()}")

        # Check for class imbalance
        if self.data[self.config.target_column].dtype in ['object', 'category']:
            class_dist = self.data[self.config.target_column].value_counts(normalize=True)
            logger.info(f"Class distribution: {class_dist.to_dict()}")

        self.run.metrics["data_quality_score"] = 100 - missing_pct.mean()

    def _engineer_features(self):
        """Apply feature engineering"""
        logger.info("Engineering features...")

        self.data = self.feature_engineer.fit_transform(self.data, self.config)

        # Save artifact
        if self.config.save_intermediate:
            artifact_path = Path(self.config.output_dir) / "data_engineered.parquet"
            self.data.to_parquet(artifact_path)
            self.run.artifacts["data_engineered"] = str(artifact_path)

    def _split_data(self):
        """Split data into train/val/test sets"""
        logger.info("Splitting data...")

        # Separate features and target
        if self.config.feature_columns:
            X = self.data[self.config.feature_columns]
        else:
            X = self.data.drop(columns=[self.config.target_column])

        y = self.data[self.config.target_column]

        # Split into train and temp (val + test)
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(
            X, y,
            test_size=self.config.test_size + self.config.validation_size,
            random_state=self.config.random_seed
        )

        # Split temp into val and test
        val_ratio = self.config.validation_size / (self.config.test_size + self.config.validation_size)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(
            X_temp, y_temp,
            test_size=1-val_ratio,
            random_state=self.config.random_seed
        )

        logger.info(f"Train: {len(self.X_train)}, Val: {len(self.X_val)}, Test: {len(self.X_test)}")

    def _tune_hyperparameters(self):
        """Tune model hyperparameters"""
        logger.info("Tuning hyperparameters...")

        def model_builder(params):
            return self._build_model(params)

        self.best_params = self.tuner.tune(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            model_builder
        )

        self.run.metrics["tuning_trials"] = self.config.tuning_trials
        self.run.artifacts["best_params"] = json.dumps(self.best_params)

    def _train_model(self):
        """Train the ML model"""
        logger.info("Training model...")

        # Use best params if tuning was done
        params = self.best_params if hasattr(self, 'best_params') else self.config.model_params

        # Build and train model
        self.model = self._build_model(params)

        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time

        self.run.metrics["training_time_seconds"] = training_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

    def _validate_model(self):
        """Validate model performance"""
        logger.info("Validating model...")

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        y_pred = self.model.predict(self.X_val)

        metrics = {
            "val_accuracy": accuracy_score(self.y_val, y_pred),
            "val_precision": precision_score(self.y_val, y_pred, average='weighted', zero_division=0),
            "val_recall": recall_score(self.y_val, y_pred, average='weighted', zero_division=0),
            "val_f1": f1_score(self.y_val, y_pred, average='weighted', zero_division=0),
        }

        self.run.metrics.update(metrics)
        logger.info(f"Validation metrics: {metrics}")

    def _test_model(self):
        """Test model on holdout set"""
        logger.info("Testing model...")

        from sklearn.metrics import accuracy_score, classification_report

        y_pred = self.model.predict(self.X_test)

        test_accuracy = accuracy_score(self.y_test, y_pred)
        self.run.metrics["test_accuracy"] = test_accuracy

        report = classification_report(self.y_test, y_pred)
        logger.info(f"Test accuracy: {test_accuracy:.4f}")
        logger.info(f"Classification report:\n{report}")

        # Save predictions
        if self.config.save_intermediate:
            predictions_path = Path(self.config.output_dir) / "test_predictions.csv"
            pd.DataFrame({
                "actual": self.y_test,
                "predicted": y_pred
            }).to_csv(predictions_path, index=False)
            self.run.artifacts["test_predictions"] = str(predictions_path)

    def _export_model(self):
        """Export trained model"""
        logger.info("Exporting model...")

        model_path = Path(self.config.output_dir) / "model.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        self.run.artifacts["model"] = str(model_path)
        logger.info(f"Model saved to: {model_path}")

    def _prepare_deployment(self):
        """Prepare model for deployment"""
        logger.info("Preparing deployment artifacts...")

        # Save feature engineer
        fe_path = Path(self.config.output_dir) / "feature_engineer.pkl"
        with open(fe_path, 'wb') as f:
            pickle.dump(self.feature_engineer, f)

        # Save model metadata
        metadata = {
            "model_type": self.config.model_type,
            "model_class": self.config.model_class,
            "features": self.config.feature_columns if self.config.feature_columns else list(self.X_train.columns),
            "target": self.config.target_column,
            "metrics": self.run.metrics,
            "training_date": datetime.now().isoformat(),
        }

        metadata_path = Path(self.config.output_dir) / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        self.run.artifacts["feature_engineer"] = str(fe_path)
        self.run.artifacts["metadata"] = str(metadata_path)

    def _build_model(self, params: Dict[str, Any]):
        """Build model from configuration"""
        if self.config.model_type == "sklearn":
            if self.config.model_class == "RandomForest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**params)
            elif self.config.model_class == "LogisticRegression":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(**params)
            elif self.config.model_class == "GradientBoosting":
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(**params)

        elif self.config.model_type == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(**params)

        raise ValueError(f"Unsupported model type: {self.config.model_type}")

    async def _save_run_metadata(self):
        """Save pipeline run metadata"""
        run_path = Path(self.config.output_dir) / f"run_{self.run.run_id}.json"

        run_data = {
            "run_id": self.run.run_id,
            "pipeline_name": self.run.pipeline_name,
            "status": self.run.status.value,
            "start_time": self.run.start_time.isoformat() if self.run.start_time else None,
            "end_time": self.run.end_time.isoformat() if self.run.end_time else None,
            "stages_completed": [s.value for s in self.run.stages_completed],
            "metrics": self.run.metrics,
            "artifacts": self.run.artifacts,
            "errors": self.run.errors,
            "logs": self.run.logs,
        }

        with open(run_path, 'w') as f:
            json.dump(run_data, f, indent=2)

        logger.info(f"Run metadata saved to: {run_path}")


# Example usage and testing
async def example_pipeline():
    """Example ML pipeline execution"""

    config = PipelineConfig(
        name="classification_pipeline",
        description="Binary classification with hyperparameter tuning",
        data_source="./data/train.csv",
        target_column="target",
        model_type="sklearn",
        model_class="RandomForest",
        model_params={"n_estimators": 100, "max_depth": 10},
        enable_tuning=True,
        tuning_trials=50,
        param_space={
            "n_estimators": {"type": "int", "low": 50, "high": 200},
            "max_depth": {"type": "int", "low": 5, "high": 20},
            "min_samples_split": {"type": "int", "low": 2, "high": 10},
        },
        cross_validation_folds=5,
        output_dir="./ml_output",
    )

    pipeline = MLPipeline(config)
    run = await pipeline.execute()

    print(f"Pipeline completed with status: {run.status.value}")
    print(f"Test accuracy: {run.metrics.get('test_accuracy', 0):.4f}")
    print(f"Artifacts: {run.artifacts}")


if __name__ == "__main__":
    asyncio.run(example_pipeline())
