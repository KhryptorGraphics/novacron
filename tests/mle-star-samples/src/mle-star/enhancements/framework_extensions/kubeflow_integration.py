"""
Kubeflow Integration
===================

Kubernetes-based ML workflows for MLE-Star using Kubeflow:
- ML pipeline orchestration on Kubernetes
- Distributed training and hyperparameter tuning
- Model serving and deployment automation
- Experiment tracking and metadata management
- Resource management and scaling
"""

import logging
import os
import yaml
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pathlib import Path
import tempfile
import json

try:
    from kfp import dsl, compiler, Client
    from kfp.components import create_component_from_func
    import kfp.components as comp
    KUBEFLOW_AVAILABLE = True
except ImportError:
    KUBEFLOW_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class KubeflowConfig:
    """Configuration for Kubeflow integration"""
    # Kubeflow Pipelines configuration
    host: str = "http://localhost:8080"
    namespace: str = "kubeflow"
    client_id: Optional[str] = None
    other_client_id: Optional[str] = None
    other_client_secret: Optional[str] = None
    
    # Pipeline configuration
    pipeline_name: str = "mle-star-pipeline"
    experiment_name: str = "mle-star-experiments"
    run_name_template: str = "{project_name}_{timestamp}"
    
    # Resource configuration
    default_cpu_request: str = "100m"
    default_cpu_limit: str = "1"
    default_memory_request: str = "128Mi"
    default_memory_limit: str = "1Gi"
    default_gpu_limit: Optional[str] = None
    
    # Docker configuration
    base_image: str = "python:3.8"
    requirements_file: Optional[str] = None
    custom_dockerfile: Optional[str] = None
    
    # Storage configuration
    persistent_volume_claim: Optional[str] = None
    volume_mount_path: str = "/mnt/data"
    
    # Katib (HPO) configuration
    katib_enabled: bool = True
    katib_algorithm: str = "bayesianoptimization"  # random, grid, bayesianoptimization, tpe
    katib_objective_metric: str = "val_accuracy"
    katib_objective_type: str = "maximize"  # maximize, minimize
    
    # KFServing configuration
    kfserving_enabled: bool = True
    serving_runtime: str = "sklearn"  # sklearn, pytorch, tensorflow, custom


class KubeflowPipelineBuilder:
    """Utility for building Kubeflow pipelines for MLE-Star"""
    
    def __init__(self, config: KubeflowConfig):
        self.config = config
        self.client = None
        
    def initialize(self) -> bool:
        """Initialize Kubeflow client"""
        try:
            self.client = Client(
                host=self.config.host,
                client_id=self.config.client_id,
                namespace=self.config.namespace,
                other_client_id=self.config.other_client_id,
                other_client_secret=self.config.other_client_secret
            )
            
            # Test connection
            experiments = self.client.list_experiments()
            logger.info(f"Connected to Kubeflow Pipelines. Found {len(experiments.experiments or [])} experiments")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Kubeflow client: {e}")
            return False
    
    def create_base_component(self, func: callable, packages: List[str] = None,
                            base_image: Optional[str] = None) -> Any:
        """Create Kubeflow component from function"""
        try:
            packages = packages or []
            base_image = base_image or self.config.base_image
            
            # Add common packages
            common_packages = [
                "pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"
            ]
            packages.extend(common_packages)
            
            component = create_component_from_func(
                func,
                base_image=base_image,
                packages_to_install=packages
            )
            
            return component
            
        except Exception as e:
            logger.error(f"Failed to create component: {e}")
            return None
    
    def create_data_preparation_component(self) -> Any:
        """Create data preparation component"""
        def prepare_data(
            dataset_path: str,
            output_path: str,
            test_size: float = 0.2,
            random_state: int = 42
        ) -> str:
            """Prepare data for ML training"""
            import pandas as pd
            import numpy as np
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            import pickle
            import os
            
            # Load data
            if dataset_path.endswith('.csv'):
                data = pd.read_csv(dataset_path)
            elif dataset_path.endswith('.parquet'):
                data = pd.read_parquet(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path}")
            
            # Assume last column is target
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save prepared data
            os.makedirs(output_path, exist_ok=True)
            
            np.save(f"{output_path}/X_train.npy", X_train_scaled)
            np.save(f"{output_path}/X_test.npy", X_test_scaled)
            np.save(f"{output_path}/y_train.npy", y_train.values)
            np.save(f"{output_path}/y_test.npy", y_test.values)
            
            # Save scaler
            with open(f"{output_path}/scaler.pkl", 'wb') as f:
                pickle.dump(scaler, f)
            
            # Save metadata
            metadata = {
                'n_samples': len(data),
                'n_features': X.shape[1],
                'n_classes': len(np.unique(y)) if y.dtype in ['object', 'category'] else 1,
                'feature_names': X.columns.tolist(),
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
            
            with open(f"{output_path}/metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            return output_path
        
        return self.create_base_component(
            prepare_data,
            packages=["scikit-learn", "pandas", "numpy"]
        )
    
    def create_training_component(self) -> Any:
        """Create model training component"""
        def train_model(
            data_path: str,
            output_path: str,
            model_type: str = "random_forest",
            hyperparameters: str = "{}"
        ) -> str:
            """Train ML model"""
            import numpy as np
            import json
            import pickle
            import os
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.linear_model import LogisticRegression, LinearRegression
            from sklearn.svm import SVC, SVR
            from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
            
            # Load data
            X_train = np.load(f"{data_path}/X_train.npy")
            X_test = np.load(f"{data_path}/X_test.npy")
            y_train = np.load(f"{data_path}/y_train.npy")
            y_test = np.load(f"{data_path}/y_test.npy")
            
            # Load metadata
            with open(f"{data_path}/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Parse hyperparameters
            params = json.loads(hyperparameters)
            
            # Create model
            is_classification = metadata['n_classes'] > 1
            
            if model_type == "random_forest":
                if is_classification:
                    model = RandomForestClassifier(**params)
                else:
                    model = RandomForestRegressor(**params)
            elif model_type == "logistic_regression" and is_classification:
                model = LogisticRegression(**params)
            elif model_type == "linear_regression" and not is_classification:
                model = LinearRegression(**params)
            elif model_type == "svm":
                if is_classification:
                    model = SVC(**params)
                else:
                    model = SVR(**params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            if is_classification:
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)
                metric_name = "accuracy"
            else:
                train_score = mean_squared_error(y_train, train_pred)
                test_score = mean_squared_error(y_test, test_pred)
                metric_name = "mse"
            
            # Save model and results
            os.makedirs(output_path, exist_ok=True)
            
            with open(f"{output_path}/model.pkl", 'wb') as f:
                pickle.dump(model, f)
            
            results = {
                'model_type': model_type,
                'hyperparameters': params,
                f'train_{metric_name}': float(train_score),
                f'val_{metric_name}': float(test_score),
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test)
            }
            
            with open(f"{output_path}/results.json", 'w') as f:
                json.dump(results, f)
            
            # Log metrics for Kubeflow
            from kfp import dsl
            dsl.get_pipeline_conf().add_op_transformer(
                lambda op: op.add_env_variable('KFP_V2_RUNTIME_INFO', 'true')
            )
            
            return output_path
        
        return self.create_base_component(
            train_model,
            packages=["scikit-learn", "numpy"]
        )
    
    def create_evaluation_component(self) -> Any:
        """Create model evaluation component"""
        def evaluate_model(
            model_path: str,
            data_path: str,
            output_path: str
        ) -> str:
            """Evaluate trained model"""
            import numpy as np
            import json
            import pickle
            import os
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                mean_squared_error, mean_absolute_error, r2_score,
                classification_report, confusion_matrix
            )
            
            # Load model
            with open(f"{model_path}/model.pkl", 'rb') as f:
                model = pickle.load(f)
            
            # Load data
            X_test = np.load(f"{data_path}/X_test.npy")
            y_test = np.load(f"{data_path}/y_test.npy")
            
            # Load metadata
            with open(f"{data_path}/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Calculate metrics
            is_classification = metadata['n_classes'] > 1
            
            if is_classification:
                metrics = {
                    'accuracy': float(accuracy_score(y_test, predictions)),
                    'precision': float(precision_score(y_test, predictions, average='weighted')),
                    'recall': float(recall_score(y_test, predictions, average='weighted')),
                    'f1_score': float(f1_score(y_test, predictions, average='weighted'))
                }
                
                # Confusion matrix
                cm = confusion_matrix(y_test, predictions)
                metrics['confusion_matrix'] = cm.tolist()
                
            else:
                metrics = {
                    'mse': float(mean_squared_error(y_test, predictions)),
                    'mae': float(mean_absolute_error(y_test, predictions)),
                    'r2_score': float(r2_score(y_test, predictions))
                }
            
            # Save results
            os.makedirs(output_path, exist_ok=True)
            
            with open(f"{output_path}/evaluation_metrics.json", 'w') as f:
                json.dump(metrics, f)
            
            # Save predictions
            np.save(f"{output_path}/predictions.npy", predictions)
            
            return output_path
        
        return self.create_base_component(
            evaluate_model,
            packages=["scikit-learn", "numpy"]
        )
    
    @dsl.pipeline(
        name="MLE-Star Pipeline",
        description="Complete MLE-Star workflow on Kubeflow"
    )
    def create_mle_star_pipeline(
        dataset_path: str,
        model_type: str = "random_forest",
        hyperparameters: str = "{}",
        test_size: float = 0.2
    ):
        """Create complete MLE-Star pipeline"""
        
        # Component definitions
        prep_data_op = self.create_data_preparation_component()
        train_model_op = self.create_training_component()
        evaluate_model_op = self.create_evaluation_component()
        
        # Pipeline steps
        # Step 1: Data preparation
        data_prep_task = prep_data_op(
            dataset_path=dataset_path,
            output_path="/tmp/prepared_data",
            test_size=test_size
        )
        
        # Step 2: Model training
        training_task = train_model_op(
            data_path=data_prep_task.output,
            output_path="/tmp/trained_model",
            model_type=model_type,
            hyperparameters=hyperparameters
        )
        
        # Step 3: Model evaluation
        evaluation_task = evaluate_model_op(
            model_path=training_task.output,
            data_path=data_prep_task.output,
            output_path="/tmp/evaluation_results"
        )
        
        # Add resource requests
        for task in [data_prep_task, training_task, evaluation_task]:
            task.set_cpu_request(self.config.default_cpu_request)
            task.set_cpu_limit(self.config.default_cpu_limit)
            task.set_memory_request(self.config.default_memory_request)
            task.set_memory_limit(self.config.default_memory_limit)
            
            if self.config.default_gpu_limit:
                task.set_gpu_limit(self.config.default_gpu_limit)
        
        return evaluation_task.output
    
    def compile_pipeline(self, output_path: str = "mle_star_pipeline.yaml"):
        """Compile pipeline to YAML"""
        try:
            compiler.Compiler().compile(
                self.create_mle_star_pipeline,
                output_path
            )
            
            logger.info(f"Pipeline compiled to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to compile pipeline: {e}")
            return None
    
    def create_experiment(self, experiment_name: Optional[str] = None) -> Optional[str]:
        """Create experiment"""
        try:
            experiment_name = experiment_name or self.config.experiment_name
            
            experiment = self.client.create_experiment(
                name=experiment_name,
                namespace=self.config.namespace
            )
            
            logger.info(f"Created experiment: {experiment_name}")
            return experiment.id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return None
    
    def run_pipeline(self, pipeline_path: str, experiment_id: str,
                    parameters: Optional[Dict[str, Any]] = None,
                    run_name: Optional[str] = None) -> Optional[str]:
        """Run pipeline"""
        try:
            run = self.client.run_pipeline(
                experiment_id=experiment_id,
                job_name=run_name or self.config.run_name_template.format(
                    project_name="mle_star", timestamp=int(time.time())
                ),
                pipeline_package_path=pipeline_path,
                params=parameters or {}
            )
            
            logger.info(f"Started pipeline run: {run.id}")
            return run.id
            
        except Exception as e:
            logger.error(f"Failed to run pipeline: {e}")
            return None
    
    def create_katib_experiment(self, search_space: Dict[str, Any],
                               objective_metric: Optional[str] = None,
                               max_trials: int = 10) -> Dict[str, Any]:
        """Create Katib hyperparameter tuning experiment"""
        objective_metric = objective_metric or self.config.katib_objective_metric
        
        katib_experiment = {
            "apiVersion": "kubeflow.org/v1beta1",
            "kind": "Experiment",
            "metadata": {
                "name": f"mle-star-katib-{int(time.time())}",
                "namespace": self.config.namespace
            },
            "spec": {
                "algorithm": {
                    "algorithmName": self.config.katib_algorithm
                },
                "objective": {
                    "type": self.config.katib_objective_type,
                    "objectiveMetricName": objective_metric
                },
                "parameters": [],
                "trialTemplate": {
                    "primaryContainerName": "training-container",
                    "trialSpec": {
                        "apiVersion": "batch/v1",
                        "kind": "Job",
                        "spec": {
                            "template": {
                                "spec": {
                                    "containers": [
                                        {
                                            "name": "training-container",
                                            "image": self.config.base_image,
                                            "command": [
                                                "python", 
                                                "/opt/mle-star/train.py"
                                            ]
                                        }
                                    ],
                                    "restartPolicy": "Never"
                                }
                            }
                        }
                    }
                },
                "parallelTrialCount": min(max_trials, 4),
                "maxTrialCount": max_trials,
                "maxFailedTrialCount": max_trials // 2
            }
        }
        
        # Add search space parameters
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                if param_config.get('type') == 'double':
                    katib_experiment["spec"]["parameters"].append({
                        "name": param_name,
                        "parameterType": "double",
                        "feasibleSpace": {
                            "min": str(param_config['min']),
                            "max": str(param_config['max'])
                        }
                    })
                elif param_config.get('type') == 'categorical':
                    katib_experiment["spec"]["parameters"].append({
                        "name": param_name,
                        "parameterType": "categorical",
                        "feasibleSpace": {
                            "list": param_config['choices']
                        }
                    })
                elif param_config.get('type') == 'discrete':
                    katib_experiment["spec"]["parameters"].append({
                        "name": param_name,
                        "parameterType": "discrete",
                        "feasibleSpace": {
                            "list": [str(x) for x in param_config['values']]
                        }
                    })
        
        return katib_experiment


class KubeflowIntegration(BaseEnhancement):
    """Kubeflow integration for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="kubeflow_integration",
            version="1.0.0",
            enabled=KUBEFLOW_AVAILABLE,
            priority=15,
            parameters={
                "host": "http://localhost:8080",
                "namespace": "kubeflow",
                "client_id": None,
                "other_client_id": None,
                "other_client_secret": None,
                "pipeline_name": "mle-star-pipeline",
                "experiment_name": "mle-star-experiments",
                "run_name_template": "{project_name}_{timestamp}",
                "default_cpu_request": "100m",
                "default_cpu_limit": "1",
                "default_memory_request": "128Mi",
                "default_memory_limit": "1Gi",
                "default_gpu_limit": None,
                "base_image": "python:3.8",
                "requirements_file": None,
                "custom_dockerfile": None,
                "persistent_volume_claim": None,
                "volume_mount_path": "/mnt/data",
                "katib_enabled": True,
                "katib_algorithm": "bayesianoptimization",
                "katib_objective_metric": "val_accuracy",
                "katib_objective_type": "maximize",
                "kfserving_enabled": True,
                "serving_runtime": "sklearn"
            }
        )
    
    def initialize(self) -> bool:
        """Initialize Kubeflow integration"""
        if not KUBEFLOW_AVAILABLE:
            self._logger.error("Kubeflow Pipelines not available. Install with: pip install kfp")
            return False
        
        try:
            # Create Kubeflow configuration
            self.kubeflow_config = KubeflowConfig(**self.config.parameters)
            
            # Initialize pipeline builder
            self.pipeline_builder = KubeflowPipelineBuilder(self.kubeflow_config)
            
            # Initialize client
            success = self.pipeline_builder.initialize()
            
            if success:
                self._logger.info("Kubeflow integration initialized successfully")
            else:
                self._logger.warning("Kubeflow client initialization failed. Pipeline compilation still available.")
                success = True  # Allow compilation even without cluster access
                
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Kubeflow: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with Kubeflow capabilities"""
        enhanced = workflow.copy()
        
        # Add Kubeflow pipeline configuration
        if 'pipeline_orchestration' not in enhanced:
            enhanced['pipeline_orchestration'] = {}
        
        enhanced['pipeline_orchestration']['kubeflow'] = {
            'enabled': True,
            'host': self.kubeflow_config.host,
            'namespace': self.kubeflow_config.namespace,
            'pipeline_name': self.kubeflow_config.pipeline_name,
            'experiment_name': self.kubeflow_config.experiment_name,
            'base_image': self.kubeflow_config.base_image,
            'resource_config': {
                'cpu_request': self.kubeflow_config.default_cpu_request,
                'cpu_limit': self.kubeflow_config.default_cpu_limit,
                'memory_request': self.kubeflow_config.default_memory_request,
                'memory_limit': self.kubeflow_config.default_memory_limit,
                'gpu_limit': self.kubeflow_config.default_gpu_limit
            }
        }
        
        # Add Katib hyperparameter tuning
        if self.kubeflow_config.katib_enabled:
            if 'hyperparameter_optimization' not in enhanced:
                enhanced['hyperparameter_optimization'] = {}
            
            enhanced['hyperparameter_optimization']['katib'] = {
                'enabled': True,
                'algorithm': self.kubeflow_config.katib_algorithm,
                'objective_metric': self.kubeflow_config.katib_objective_metric,
                'objective_type': self.kubeflow_config.katib_objective_type
            }
        
        # Add KFServing deployment
        if self.kubeflow_config.kfserving_enabled:
            if 'deployment' not in enhanced:
                enhanced['deployment'] = {}
            
            enhanced['deployment']['kfserving'] = {
                'enabled': True,
                'serving_runtime': self.kubeflow_config.serving_runtime,
                'auto_scaling': True,
                'canary_deployment': True
            }
        
        # Enhance MLE-Star stages with Kubeflow components
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Map each stage to Kubeflow components
            stage_components = {
                '1_situation_analysis': 'data_exploration_component',
                '2_task_definition': 'task_definition_component', 
                '3_action_planning': 'planning_component',
                '4_implementation': 'training_component',
                '5_results_evaluation': 'evaluation_component',
                '6_refinement': 'optimization_component',
                '7_deployment_prep': 'deployment_component'
            }
            
            for stage_name, component_name in stage_components.items():
                if stage_name in stages:
                    if 'kubeflow_component' not in stages[stage_name]:
                        stages[stage_name]['kubeflow_component'] = {
                            'name': component_name,
                            'containerized': True,
                            'resource_requirements': {
                                'cpu': self.kubeflow_config.default_cpu_request,
                                'memory': self.kubeflow_config.default_memory_request
                            }
                        }
        
        self._logger.debug("Enhanced workflow with Kubeflow capabilities")
        return enhanced
    
    def create_and_run_pipeline(self, project_name: str, dataset_path: str,
                               model_config: Dict[str, Any]) -> Optional[str]:
        """Create and run complete MLE-Star pipeline"""
        try:
            import time
            
            # Compile pipeline
            pipeline_path = self.pipeline_builder.compile_pipeline(
                f"mle_star_pipeline_{project_name}.yaml"
            )
            
            if not pipeline_path:
                return None
            
            # Create experiment
            experiment_id = self.pipeline_builder.create_experiment(
                f"mle-star-{project_name}"
            )
            
            if not experiment_id:
                return None
            
            # Prepare parameters
            parameters = {
                'dataset_path': dataset_path,
                'model_type': model_config.get('type', 'random_forest'),
                'hyperparameters': json.dumps(model_config.get('parameters', {})),
                'test_size': model_config.get('test_size', 0.2)
            }
            
            # Run pipeline
            run_id = self.pipeline_builder.run_pipeline(
                pipeline_path=pipeline_path,
                experiment_id=experiment_id,
                parameters=parameters,
                run_name=f"mle-star-{project_name}-{int(time.time())}"
            )
            
            if run_id:
                self._logger.info(f"Started MLE-Star pipeline run: {run_id}")
            
            return run_id
            
        except Exception as e:
            self._logger.error(f"Failed to create and run pipeline: {e}")
            return None
    
    def create_katib_hpo_experiment(self, search_space: Dict[str, Any],
                                   max_trials: int = 10) -> Optional[Dict[str, Any]]:
        """Create Katib hyperparameter optimization experiment"""
        try:
            katib_experiment = self.pipeline_builder.create_katib_experiment(
                search_space=search_space,
                max_trials=max_trials
            )
            
            # Save experiment YAML
            experiment_file = f"katib_experiment_{int(time.time())}.yaml"
            with open(experiment_file, 'w') as f:
                yaml.dump(katib_experiment, f, default_flow_style=False)
            
            self._logger.info(f"Created Katib experiment YAML: {experiment_file}")
            return katib_experiment
            
        except Exception as e:
            self._logger.error(f"Failed to create Katib experiment: {e}")
            return None