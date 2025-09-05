"""
MLflow Integration
=================

Experiment tracking and model registry capabilities for MLE-Star using MLflow:
- Experiment tracking and logging
- Model registry and versioning
- Artifact management
- Model serving and deployment
- Collaborative ML workflows
"""

import logging
import os
import json
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import pickle
import tempfile

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.pytorch
    import mlflow.tensorflow
    from mlflow.tracking import MlflowClient
    from mlflow.entities import ViewType
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class MLflowConfig:
    """Configuration for MLflow integration"""
    # Tracking configuration
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "MLE-Star-Experiments"
    run_name_template: str = "{project_name}_{timestamp}"
    
    # Model registry
    registered_model_name_template: str = "{project_name}_model"
    model_stage: str = "Staging"  # None, Staging, Production, Archived
    
    # Artifact logging
    artifact_location: Optional[str] = None
    log_model_artifacts: bool = True
    log_code: bool = True
    log_data_samples: bool = False
    
    # Auto-logging
    auto_log_frameworks: List[str] = None  # ["sklearn", "pytorch", "tensorflow"]
    auto_log_every_n_iter: int = 1
    
    # Tagging and organization
    default_tags: Dict[str, str] = None
    track_system_metrics: bool = True


class MLflowLogger:
    """MLflow logging utilities for MLE-Star"""
    
    def __init__(self, config: MLflowConfig):
        self.config = config
        self.client = None
        self.experiment_id = None
        self.active_run = None
        
    def initialize(self) -> bool:
        """Initialize MLflow client and experiment"""
        try:
            # Set tracking URI
            mlflow.set_tracking_uri(self.config.tracking_uri)
            
            # Create MLflow client
            self.client = MlflowClient()
            
            # Create or get experiment
            try:
                experiment = mlflow.get_experiment_by_name(self.config.experiment_name)
                if experiment is None:
                    self.experiment_id = mlflow.create_experiment(
                        self.config.experiment_name,
                        artifact_location=self.config.artifact_location
                    )
                else:
                    self.experiment_id = experiment.experiment_id
                    
            except Exception as e:
                logger.warning(f"Failed to create experiment: {e}")
                self.experiment_id = "0"  # Default experiment
            
            # Set experiment
            mlflow.set_experiment(experiment_id=self.experiment_id)
            
            # Enable auto-logging if configured
            if self.config.auto_log_frameworks:
                self._setup_auto_logging()
            
            logger.info(f"MLflow initialized with experiment: {self.config.experiment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MLflow: {e}")
            return False
    
    def _setup_auto_logging(self):
        """Setup automatic logging for supported frameworks"""
        for framework in self.config.auto_log_frameworks:
            try:
                if framework == "sklearn":
                    mlflow.sklearn.autolog(
                        log_input_examples=self.config.log_data_samples,
                        log_model_signatures=True,
                        log_models=self.config.log_model_artifacts
                    )
                elif framework == "pytorch":
                    mlflow.pytorch.autolog(
                        log_every_n_epoch=self.config.auto_log_every_n_iter,
                        log_models=self.config.log_model_artifacts
                    )
                elif framework == "tensorflow":
                    mlflow.tensorflow.autolog(
                        every_n_iter=self.config.auto_log_every_n_iter,
                        log_models=self.config.log_model_artifacts
                    )
                    
                logger.info(f"Enabled auto-logging for {framework}")
                
            except Exception as e:
                logger.warning(f"Failed to enable auto-logging for {framework}: {e}")
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> str:
        """Start a new MLflow run"""
        try:
            # Combine default tags with provided tags
            all_tags = {}
            if self.config.default_tags:
                all_tags.update(self.config.default_tags)
            if tags:
                all_tags.update(tags)
            
            # Start run
            run = mlflow.start_run(
                run_name=run_name,
                tags=all_tags,
                experiment_id=self.experiment_id
            )
            
            self.active_run = run
            
            # Log system info if enabled
            if self.config.track_system_metrics:
                self._log_system_info()
            
            logger.info(f"Started MLflow run: {run.info.run_id}")
            return run.info.run_id
            
        except Exception as e:
            logger.error(f"Failed to start MLflow run: {e}")
            return None
    
    def end_run(self, status: str = "FINISHED"):
        """End the current MLflow run"""
        try:
            if self.active_run:
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
                self.active_run = None
        except Exception as e:
            logger.error(f"Failed to end MLflow run: {e}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow"""
        try:
            # Convert non-string values to strings
            str_params = {k: str(v) for k, v in params.items()}
            mlflow.log_params(str_params)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """Log metrics to MLflow"""
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_artifact(self, artifact_path: Union[str, Path], artifact_name: Optional[str] = None):
        """Log artifact to MLflow"""
        try:
            if artifact_name:
                # Create temporary file with desired name
                with tempfile.NamedTemporaryFile(suffix=f"_{artifact_name}", delete=False) as tmp:
                    with open(artifact_path, 'rb') as src:
                        tmp.write(src.read())
                    mlflow.log_artifact(tmp.name, artifact_name)
                    os.unlink(tmp.name)
            else:
                mlflow.log_artifact(str(artifact_path))
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_model(self, model: Any, artifact_path: str = "model", 
                  signature=None, input_example=None, **kwargs):
        """Log model to MLflow"""
        try:
            # Detect model type and log appropriately
            model_type = type(model).__module__
            
            if "sklearn" in model_type:
                mlflow.sklearn.log_model(
                    model, artifact_path, signature=signature, 
                    input_example=input_example, **kwargs
                )
            elif "torch" in model_type:
                mlflow.pytorch.log_model(
                    model, artifact_path, signature=signature,
                    input_example=input_example, **kwargs
                )
            elif "tensorflow" in model_type or "keras" in model_type:
                mlflow.tensorflow.log_model(
                    model, artifact_path, signature=signature,
                    input_example=input_example, **kwargs
                )
            else:
                # Generic pickle logging
                with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
                    pickle.dump(model, tmp)
                    tmp.flush()
                    mlflow.log_artifact(tmp.name, f"{artifact_path}/model.pkl")
                    os.unlink(tmp.name)
                    
            logger.info(f"Logged model to {artifact_path}")
            
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def register_model(self, model_uri: str, model_name: Optional[str] = None) -> Optional[str]:
        """Register model in MLflow Model Registry"""
        try:
            if model_name is None:
                model_name = self.config.registered_model_name_template.format(
                    project_name="MLE_Star"
                )
            
            result = mlflow.register_model(model_uri, model_name)
            
            # Optionally transition to stage
            if self.config.model_stage and self.config.model_stage != "None":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=result.version,
                    stage=self.config.model_stage
                )
                
            logger.info(f"Registered model: {model_name} version {result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
    
    def _log_system_info(self):
        """Log system information"""
        try:
            import platform
            import psutil
            import sys
            
            system_info = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "cpu_count": psutil.cpu_count(),
                "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2)
            }
            
            mlflow.set_tags(system_info)
            
        except Exception as e:
            logger.warning(f"Failed to log system info: {e}")
    
    def search_runs(self, filter_string: str = "", max_results: int = 100) -> List[Dict[str, Any]]:
        """Search MLflow runs"""
        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=filter_string,
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=max_results
            )
            
            run_data = []
            for run in runs:
                run_data.append({
                    'run_id': run.info.run_id,
                    'start_time': run.info.start_time,
                    'end_time': run.info.end_time,
                    'status': run.info.status,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                })
            
            return run_data
            
        except Exception as e:
            logger.error(f"Failed to search runs: {e}")
            return []


class MLflowIntegration(BaseEnhancement):
    """MLflow integration for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="mlflow_integration",
            version="1.0.0",
            enabled=MLFLOW_AVAILABLE,
            priority=30,
            parameters={
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "MLE-Star-Experiments", 
                "run_name_template": "{project_name}_{timestamp}",
                "registered_model_name_template": "{project_name}_model",
                "model_stage": "Staging",
                "artifact_location": None,
                "log_model_artifacts": True,
                "log_code": True,
                "log_data_samples": False,
                "auto_log_frameworks": ["sklearn", "pytorch"],
                "auto_log_every_n_iter": 1,
                "default_tags": {"framework": "mle-star"},
                "track_system_metrics": True
            }
        )
    
    def initialize(self) -> bool:
        """Initialize MLflow integration"""
        if not MLFLOW_AVAILABLE:
            self._logger.error("MLflow not available. Install with: pip install mlflow")
            return False
        
        try:
            # Create MLflow configuration
            self.mlflow_config = MLflowConfig(**self.config.parameters)
            
            # Initialize logger
            self.mlflow_logger = MLflowLogger(self.mlflow_config)
            
            # Initialize MLflow
            success = self.mlflow_logger.initialize()
            
            if success:
                self._logger.info("MLflow integration initialized successfully")
            else:
                self._logger.error("Failed to initialize MLflow integration")
                
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to initialize MLflow: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with MLflow capabilities"""
        enhanced = workflow.copy()
        
        # Add MLflow logging configuration
        if 'logging' not in enhanced:
            enhanced['logging'] = {}
        
        enhanced['logging']['mlflow'] = {
            'enabled': True,
            'tracking_uri': self.mlflow_config.tracking_uri,
            'experiment_name': self.mlflow_config.experiment_name,
            'auto_log_frameworks': self.mlflow_config.auto_log_frameworks,
            'log_model_artifacts': self.mlflow_config.log_model_artifacts,
            'track_system_metrics': self.mlflow_config.track_system_metrics
        }
        
        # Add model registry configuration
        if 'model_registry' not in enhanced:
            enhanced['model_registry'] = {}
        
        enhanced['model_registry']['mlflow'] = {
            'enabled': True,
            'registered_model_name_template': self.mlflow_config.registered_model_name_template,
            'model_stage': self.mlflow_config.model_stage
        }
        
        # Enhance MLE-Star stages with MLflow logging
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 1: Situation Analysis - Log data analysis
            if '1_situation_analysis' in stages:
                if 'mlflow_logging' not in stages['1_situation_analysis']:
                    stages['1_situation_analysis']['mlflow_logging'] = [
                        'data_statistics',
                        'dataset_schema',
                        'data_quality_metrics'
                    ]
            
            # Stage 2: Task Definition - Log objectives and metrics
            if '2_task_definition' in stages:
                if 'mlflow_logging' not in stages['2_task_definition']:
                    stages['2_task_definition']['mlflow_logging'] = [
                        'task_parameters',
                        'success_metrics_definition',
                        'model_constraints'
                    ]
            
            # Stage 3: Action Planning - Log model architecture decisions
            if '3_action_planning' in stages:
                if 'mlflow_logging' not in stages['3_action_planning']:
                    stages['3_action_planning']['mlflow_logging'] = [
                        'model_architecture_params',
                        'training_strategy',
                        'hyperparameter_space'
                    ]
            
            # Stage 4: Implementation - Log training metrics and model
            if '4_implementation' in stages:
                if 'mlflow_logging' not in stages['4_implementation']:
                    stages['4_implementation']['mlflow_logging'] = [
                        'training_metrics',
                        'model_checkpoints',
                        'training_artifacts',
                        'model_registration'
                    ]
            
            # Stage 5: Results Evaluation - Log evaluation metrics
            if '5_results_evaluation' in stages:
                if 'mlflow_logging' not in stages['5_results_evaluation']:
                    stages['5_results_evaluation']['mlflow_logging'] = [
                        'evaluation_metrics',
                        'confusion_matrix',
                        'performance_plots',
                        'model_comparison'
                    ]
            
            # Stage 6: Refinement - Log optimization results
            if '6_refinement' in stages:
                if 'mlflow_logging' not in stages['6_refinement']:
                    stages['6_refinement']['mlflow_logging'] = [
                        'hyperparameter_tuning_results',
                        'model_versions',
                        'optimization_metrics',
                        'ablation_study_results'
                    ]
            
            # Stage 7: Deployment Prep - Log deployment artifacts
            if '7_deployment_prep' in stages:
                if 'mlflow_logging' not in stages['7_deployment_prep']:
                    stages['7_deployment_prep']['mlflow_logging'] = [
                        'deployment_artifacts',
                        'model_serving_config',
                        'inference_examples',
                        'production_model_registration'
                    ]
        
        # Add MLflow-specific evaluation metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'mlflow_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['mlflow_metrics'] = [
                'run_duration',
                'artifact_count',
                'model_size',
                'registration_success'
            ]
        
        self._logger.debug("Enhanced workflow with MLflow capabilities")
        return enhanced
    
    def create_run_context(self, project_name: str, stage: str, **kwargs):
        """Create MLflow run context for MLE-Star stage"""
        try:
            import time
            
            # Generate run name
            run_name = f"{project_name}_{stage}_{int(time.time())}"
            
            # Prepare tags
            tags = {
                "mle_star_stage": stage,
                "project_name": project_name,
                **kwargs
            }
            
            # Start run
            run_id = self.mlflow_logger.start_run(run_name=run_name, tags=tags)
            
            return run_id
            
        except Exception as e:
            self._logger.error(f"Failed to create run context: {e}")
            return None
    
    def log_stage_results(self, stage: str, results: Dict[str, Any]):
        """Log results for a specific MLE-Star stage"""
        try:
            # Separate parameters and metrics
            params = {}
            metrics = {}
            
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    metrics[f"{stage}_{key}"] = value
                else:
                    params[f"{stage}_{key}"] = str(value)
            
            # Log to MLflow
            if params:
                self.mlflow_logger.log_params(params)
            if metrics:
                self.mlflow_logger.log_metrics(metrics)
                
            self._logger.debug(f"Logged results for stage: {stage}")
            
        except Exception as e:
            self._logger.error(f"Failed to log stage results: {e}")
    
    def compare_experiments(self, metric_name: str, ascending: bool = False) -> List[Dict[str, Any]]:
        """Compare experiments by a specific metric"""
        try:
            runs = self.mlflow_logger.search_runs()
            
            # Filter runs with the specified metric
            valid_runs = [run for run in runs if metric_name in run['metrics']]
            
            # Sort by metric
            valid_runs.sort(
                key=lambda x: x['metrics'][metric_name], 
                reverse=not ascending
            )
            
            return valid_runs
            
        except Exception as e:
            self._logger.error(f"Failed to compare experiments: {e}")
            return []