"""
Weights & Biases Integration
===========================

Advanced experiment monitoring and collaboration for MLE-Star using W&B:
- Real-time experiment tracking and visualization
- Hyperparameter optimization with Sweeps
- Model and dataset versioning
- Collaborative experiment management
- Advanced visualizations and reporting
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass
import json
import tempfile

try:
    import wandb
    from wandb.integration.sklearn import plot_learning_curve, plot_confusion_matrix
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass 
class WandbConfig:
    """Configuration for Weights & Biases integration"""
    # Project configuration
    project_name: str = "mle-star-experiments"
    entity: Optional[str] = None  # W&B username or team name
    group: Optional[str] = None   # Experiment group
    job_type: Optional[str] = "train"
    
    # Run configuration
    run_name: Optional[str] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None
    
    # Logging configuration
    log_frequency: int = 100  # Steps between logs
    log_gradients: bool = False
    log_parameters: bool = True
    save_code: bool = True
    
    # Visualization configuration
    log_confusion_matrix: bool = True
    log_feature_importance: bool = True
    log_model_architecture: bool = True
    log_learning_curves: bool = True
    
    # Model management
    save_model: bool = True
    model_name: Optional[str] = None
    
    # Sweep configuration
    sweep_config: Optional[Dict[str, Any]] = None
    sweep_count: int = 10


class WandbLogger:
    """Weights & Biases logging utilities for MLE-Star"""
    
    def __init__(self, config: WandbConfig):
        self.config = config
        self.run = None
        self.sweep_id = None
        
    def initialize(self, **kwargs) -> bool:
        """Initialize W&B run"""
        try:
            # Combine config with kwargs
            init_kwargs = {
                'project': self.config.project_name,
                'entity': self.config.entity,
                'group': self.config.group,
                'job_type': self.config.job_type,
                'name': self.config.run_name,
                'notes': self.config.notes,
                'tags': self.config.tags,
                'save_code': self.config.save_code,
                **kwargs
            }
            
            # Remove None values
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
            
            # Initialize W&B
            self.run = wandb.init(**init_kwargs)
            
            logger.info(f"W&B initialized: {self.run.name} ({self.run.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            return False
    
    def log(self, data: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """Log data to W&B"""
        try:
            if self.run:
                wandb.log(data, step=step, commit=commit)
        except Exception as e:
            logger.error(f"Failed to log data to W&B: {e}")
    
    def log_metrics(self, metrics: Dict[str, Union[int, float]], step: Optional[int] = None):
        """Log metrics to W&B"""
        try:
            if self.run:
                wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to W&B config"""
        try:
            if self.run:
                wandb.config.update(params)
        except Exception as e:
            logger.error(f"Failed to log parameters: {e}")
    
    def log_artifact(self, artifact_path: str, name: str, artifact_type: str = "dataset"):
        """Log artifact to W&B"""
        try:
            if self.run:
                artifact = wandb.Artifact(name, type=artifact_type)
                artifact.add_file(artifact_path)
                wandb.log_artifact(artifact)
        except Exception as e:
            logger.error(f"Failed to log artifact: {e}")
    
    def log_model(self, model_path: str, name: Optional[str] = None, aliases: List[str] = None):
        """Log model to W&B"""
        try:
            if self.run and self.config.save_model:
                model_name = name or self.config.model_name or "model"
                aliases = aliases or ["latest"]
                
                # Create model artifact
                model_artifact = wandb.Artifact(
                    name=f"{model_name}",
                    type="model"
                )
                model_artifact.add_file(model_path)
                
                # Log with aliases
                wandb.log_artifact(model_artifact, aliases=aliases)
                
                logger.info(f"Logged model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to log model: {e}")
    
    def log_confusion_matrix(self, y_true, y_pred, class_names: Optional[List[str]] = None):
        """Log confusion matrix visualization"""
        try:
            if self.run and self.config.log_confusion_matrix:
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=y_true,
                        preds=y_pred,
                        class_names=class_names
                    )
                })
        except Exception as e:
            logger.error(f"Failed to log confusion matrix: {e}")
    
    def log_feature_importance(self, feature_names: List[str], importance_values: List[float]):
        """Log feature importance visualization"""
        try:
            if self.run and self.config.log_feature_importance:
                data = [[name, importance] for name, importance in zip(feature_names, importance_values)]
                table = wandb.Table(data=data, columns=["feature", "importance"])
                
                wandb.log({
                    "feature_importance": wandb.plot.bar(
                        table, "feature", "importance", 
                        title="Feature Importance"
                    )
                })
        except Exception as e:
            logger.error(f"Failed to log feature importance: {e}")
    
    def log_learning_curves(self, train_sizes, train_scores, validation_scores):
        """Log learning curves"""
        try:
            if self.run and self.config.log_learning_curves:
                import numpy as np
                
                # Calculate mean and std
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                val_mean = np.mean(validation_scores, axis=1)
                val_std = np.std(validation_scores, axis=1)
                
                # Create data for plotting
                data = []
                for i, size in enumerate(train_sizes):
                    data.append([
                        size, train_mean[i], train_std[i], 
                        val_mean[i], val_std[i]
                    ])
                
                table = wandb.Table(
                    data=data,
                    columns=["train_size", "train_score", "train_std", 
                            "val_score", "val_std"]
                )
                
                wandb.log({
                    "learning_curves": wandb.plot.line_series(
                        xs=train_sizes,
                        ys=[train_mean, val_mean],
                        keys=["train", "validation"],
                        title="Learning Curves",
                        xname="Training Set Size"
                    )
                })
                
        except Exception as e:
            logger.error(f"Failed to log learning curves: {e}")
    
    def log_model_architecture(self, model, input_shape: Optional[tuple] = None):
        """Log model architecture visualization"""
        try:
            if self.run and self.config.log_model_architecture:
                # Try different ways to log model architecture
                if hasattr(model, 'summary'):
                    # Keras/TensorFlow model
                    import io
                    import contextlib
                    
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        model.summary()
                    
                    wandb.log({"model_architecture": wandb.Html(
                        f"<pre>{buffer.getvalue()}</pre>"
                    )})
                    
                elif hasattr(model, '__str__'):
                    # PyTorch or other models
                    wandb.log({"model_architecture": wandb.Html(
                        f"<pre>{str(model)}</pre>"
                    )})
                    
        except Exception as e:
            logger.error(f"Failed to log model architecture: {e}")
    
    def create_sweep(self, sweep_config: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create hyperparameter sweep"""
        try:
            config = sweep_config or self.config.sweep_config
            if not config:
                logger.warning("No sweep configuration provided")
                return None
                
            self.sweep_id = wandb.sweep(
                sweep=config,
                project=self.config.project_name,
                entity=self.config.entity
            )
            
            logger.info(f"Created W&B sweep: {self.sweep_id}")
            return self.sweep_id
            
        except Exception as e:
            logger.error(f"Failed to create sweep: {e}")
            return None
    
    def run_sweep_agent(self, train_function: Callable, count: Optional[int] = None):
        """Run sweep agent"""
        try:
            if not self.sweep_id:
                logger.error("No sweep ID available. Create sweep first.")
                return
                
            count = count or self.config.sweep_count
            
            wandb.agent(
                sweep_id=self.sweep_id,
                function=train_function,
                count=count
            )
            
        except Exception as e:
            logger.error(f"Failed to run sweep agent: {e}")
    
    def finish(self):
        """Finish W&B run"""
        try:
            if self.run:
                wandb.finish()
                logger.info("W&B run finished")
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")


class WeightsBiasesIntegration(BaseEnhancement):
    """Weights & Biases integration for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="wandb_integration",
            version="1.0.0", 
            enabled=WANDB_AVAILABLE,
            priority=25,
            parameters={
                "project_name": "mle-star-experiments",
                "entity": None,
                "group": None,
                "job_type": "train",
                "run_name": None,
                "notes": None,
                "tags": ["mle-star"],
                "log_frequency": 100,
                "log_gradients": False,
                "log_parameters": True,
                "save_code": True,
                "log_confusion_matrix": True,
                "log_feature_importance": True,
                "log_model_architecture": True,
                "log_learning_curves": True,
                "save_model": True,
                "model_name": None,
                "sweep_config": None,
                "sweep_count": 10
            }
        )
    
    def initialize(self) -> bool:
        """Initialize W&B integration"""
        if not WANDB_AVAILABLE:
            self._logger.error("Weights & Biases not available. Install with: pip install wandb")
            return False
        
        try:
            # Create W&B configuration
            self.wandb_config = WandbConfig(**self.config.parameters)
            
            # Initialize logger
            self.wandb_logger = WandbLogger(self.wandb_config)
            
            # Check if W&B is logged in
            if not wandb.api.api_key:
                self._logger.warning("W&B not logged in. Run 'wandb login' or set WANDB_API_KEY")
            
            self._logger.info("W&B integration initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize W&B: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with W&B capabilities"""
        enhanced = workflow.copy()
        
        # Add W&B logging configuration
        if 'logging' not in enhanced:
            enhanced['logging'] = {}
        
        enhanced['logging']['wandb'] = {
            'enabled': True,
            'project_name': self.wandb_config.project_name,
            'entity': self.wandb_config.entity,
            'log_frequency': self.wandb_config.log_frequency,
            'log_gradients': self.wandb_config.log_gradients,
            'save_code': self.wandb_config.save_code,
            'visualizations': {
                'confusion_matrix': self.wandb_config.log_confusion_matrix,
                'feature_importance': self.wandb_config.log_feature_importance,
                'model_architecture': self.wandb_config.log_model_architecture,
                'learning_curves': self.wandb_config.log_learning_curves
            }
        }
        
        # Add hyperparameter optimization
        if self.wandb_config.sweep_config:
            if 'hyperparameter_optimization' not in enhanced:
                enhanced['hyperparameter_optimization'] = {}
            
            enhanced['hyperparameter_optimization']['wandb_sweeps'] = {
                'enabled': True,
                'sweep_config': self.wandb_config.sweep_config,
                'sweep_count': self.wandb_config.sweep_count
            }
        
        # Enhance MLE-Star stages with W&B logging
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 1: Situation Analysis
            if '1_situation_analysis' in stages:
                if 'wandb_logging' not in stages['1_situation_analysis']:
                    stages['1_situation_analysis']['wandb_logging'] = [
                        'data_exploration_plots',
                        'data_distribution_charts',
                        'correlation_matrices'
                    ]
            
            # Stage 2: Task Definition
            if '2_task_definition' in stages:
                if 'wandb_logging' not in stages['2_task_definition']:
                    stages['2_task_definition']['wandb_logging'] = [
                        'task_configuration',
                        'success_criteria',
                        'baseline_metrics'
                    ]
            
            # Stage 3: Action Planning
            if '3_action_planning' in stages:
                if 'wandb_logging' not in stages['3_action_planning']:
                    stages['3_action_planning']['wandb_logging'] = [
                        'model_architecture_diagram',
                        'hyperparameter_search_space',
                        'training_strategy_config'
                    ]
            
            # Stage 4: Implementation
            if '4_implementation' in stages:
                if 'wandb_logging' not in stages['4_implementation']:
                    stages['4_implementation']['wandb_logging'] = [
                        'real_time_training_metrics',
                        'loss_curves',
                        'gradient_histograms',
                        'model_checkpoints'
                    ]
            
            # Stage 5: Results Evaluation
            if '5_results_evaluation' in stages:
                if 'wandb_logging' not in stages['5_results_evaluation']:
                    stages['5_results_evaluation']['wandb_logging'] = [
                        'confusion_matrix_visualization',
                        'roc_curves',
                        'precision_recall_curves',
                        'feature_importance_plots'
                    ]
            
            # Stage 6: Refinement
            if '6_refinement' in stages:
                if 'wandb_logging' not in stages['6_refinement']:
                    stages['6_refinement']['wandb_logging'] = [
                        'hyperparameter_sweep_results',
                        'model_comparison_charts',
                        'optimization_progress',
                        'ablation_study_results'
                    ]
            
            # Stage 7: Deployment Prep
            if '7_deployment_prep' in stages:
                if 'wandb_logging' not in stages['7_deployment_prep']:
                    stages['7_deployment_prep']['wandb_logging'] = [
                        'final_model_artifacts',
                        'inference_examples',
                        'deployment_metrics',
                        'model_performance_summary'
                    ]
        
        # Add W&B specific evaluation metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'wandb_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['wandb_metrics'] = [
                'run_duration',
                'gpu_utilization',
                'memory_usage',
                'artifact_count',
                'visualization_count'
            ]
        
        self._logger.debug("Enhanced workflow with W&B capabilities")
        return enhanced
    
    def create_run_context(self, project_name: str, stage: str, config: Optional[Dict[str, Any]] = None):
        """Create W&B run context for MLE-Star stage"""
        try:
            run_config = {
                'name': f"{project_name}_{stage}",
                'job_type': stage,
                'tags': [stage, 'mle-star'],
                'notes': f"MLE-Star {stage} stage for {project_name}",
                'config': config or {}
            }
            
            success = self.wandb_logger.initialize(**run_config)
            
            if success:
                self._logger.info(f"Created W&B run for stage: {stage}")
                return self.wandb_logger.run
            else:
                return None
                
        except Exception as e:
            self._logger.error(f"Failed to create run context: {e}")
            return None
    
    def log_stage_results(self, stage: str, results: Dict[str, Any], step: Optional[int] = None):
        """Log results for a specific MLE-Star stage"""
        try:
            # Separate different types of data
            metrics = {}
            config_updates = {}
            
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    metrics[f"{stage}_{key}"] = value
                else:
                    config_updates[f"{stage}_{key}"] = value
            
            # Log metrics
            if metrics:
                self.wandb_logger.log_metrics(metrics, step=step)
            
            # Update config
            if config_updates:
                self.wandb_logger.log_parameters(config_updates)
                
            self._logger.debug(f"Logged results for stage: {stage}")
            
        except Exception as e:
            self._logger.error(f"Failed to log stage results: {e}")
    
    def create_default_sweep_config(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Create default sweep configuration"""
        sweep_config = {
            'method': 'bayes',  # or 'grid', 'random'
            'metric': {
                'name': 'val_accuracy',
                'goal': 'maximize'
            },
            'parameters': {}
        }
        
        # Convert parameters to W&B sweep format
        for key, value in parameters.items():
            if isinstance(value, list):
                sweep_config['parameters'][key] = {'values': value}
            elif isinstance(value, dict) and 'min' in value and 'max' in value:
                sweep_config['parameters'][key] = {
                    'min': value['min'],
                    'max': value['max']
                }
                if 'distribution' in value:
                    sweep_config['parameters'][key]['distribution'] = value['distribution']
            else:
                sweep_config['parameters'][key] = {'value': value}
        
        return sweep_config
    
    def run_hyperparameter_sweep(self, train_function: Callable, 
                                parameters: Dict[str, Any], count: int = 10):
        """Run hyperparameter sweep"""
        try:
            # Create sweep configuration
            sweep_config = self.create_default_sweep_config(parameters)
            
            # Create sweep
            sweep_id = self.wandb_logger.create_sweep(sweep_config)
            
            if sweep_id:
                # Run sweep agent
                self.wandb_logger.run_sweep_agent(train_function, count)
                return sweep_id
            else:
                return None
                
        except Exception as e:
            self._logger.error(f"Failed to run hyperparameter sweep: {e}")
            return None