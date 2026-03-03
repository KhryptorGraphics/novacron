"""
Ray Tune Integration
===================

Distributed hyperparameter optimization for MLE-Star using Ray Tune:
- Scalable hyperparameter search across multiple nodes
- Advanced search algorithms (Bayesian, Population-based, etc.)
- Early stopping and pruning strategies
- Integration with popular ML frameworks
- Resource-aware scheduling and optimization
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union, Callable, Tuple
from dataclasses import dataclass
import json
import tempfile

try:
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, MedianStoppingRule
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.bayesopt import BayesOptSearch
    from ray.tune.search.hyperopt import HyperOptSearch
    from ray.tune.search.optuna import OptunaSearch
    from ray.tune.integration.wandb import WandbLoggerCallback
    from ray.tune.integration.mlflow import MLflowLoggerCallback
    RAY_TUNE_AVAILABLE = True
except ImportError:
    RAY_TUNE_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class RayTuneConfig:
    """Configuration for Ray Tune integration"""
    # Ray cluster configuration
    ray_address: Optional[str] = None  # Ray cluster address or "auto"
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    memory: Optional[int] = None  # Memory in bytes
    
    # Search space configuration
    search_algorithm: str = "bayesian"  # bayesian, hyperopt, optuna, random, grid
    search_algorithm_config: Dict[str, Any] = None
    
    # Scheduler configuration
    scheduler: str = "asha"  # asha, pbt, median, fifo
    scheduler_config: Dict[str, Any] = None
    
    # Stopping criteria
    max_trials: int = 100
    max_concurrent_trials: int = 4
    time_budget_s: Optional[int] = None  # Time budget in seconds
    
    # Early stopping
    early_stopping_metric: str = "val_accuracy"
    early_stopping_mode: str = "max"  # max or min
    early_stopping_patience: int = 10
    
    # Resource allocation
    resources_per_trial: Dict[str, Union[int, float]] = None
    
    # Logging and callbacks
    local_dir: str = "./ray_results"
    experiment_name: str = "mle_star_tune"
    log_to_file: bool = True
    verbose: int = 1
    
    # Integration with other tools
    wandb_integration: bool = False
    mlflow_integration: bool = False


class RayTuneSearchSpace:
    """Utilities for defining Ray Tune search spaces"""
    
    @staticmethod
    def uniform(low: float, high: float) -> Dict[str, Any]:
        """Uniform distribution"""
        return tune.uniform(low, high)
    
    @staticmethod
    def loguniform(low: float, high: float) -> Dict[str, Any]:
        """Log-uniform distribution"""
        return tune.loguniform(low, high)
    
    @staticmethod
    def choice(choices: List[Any]) -> Dict[str, Any]:
        """Choice from discrete values"""
        return tune.choice(choices)
    
    @staticmethod
    def grid_search(values: List[Any]) -> Dict[str, Any]:
        """Grid search over values"""
        return tune.grid_search(values)
    
    @staticmethod
    def randint(lower: int, upper: int) -> Dict[str, Any]:
        """Random integer"""
        return tune.randint(lower, upper)
    
    @staticmethod
    def randn(mean: float = 0.0, sd: float = 1.0) -> Dict[str, Any]:
        """Random normal distribution"""
        return tune.randn(mean, sd)


class RayTuneTrainer:
    """Ray Tune training utilities for MLE-Star"""
    
    def __init__(self, config: RayTuneConfig):
        self.config = config
        self.search_alg = None
        self.scheduler = None
        self.callbacks = []
        
    def initialize(self) -> bool:
        """Initialize Ray Tune"""
        try:
            # Initialize Ray
            if not ray.is_initialized():
                ray_init_config = {}
                
                if self.config.ray_address:
                    ray_init_config['address'] = self.config.ray_address
                if self.config.num_cpus:
                    ray_init_config['num_cpus'] = self.config.num_cpus
                if self.config.num_gpus:
                    ray_init_config['num_gpus'] = self.config.num_gpus
                if self.config.memory:
                    ray_init_config['object_store_memory'] = self.config.memory
                
                ray.init(**ray_init_config)
            
            # Setup search algorithm
            self._setup_search_algorithm()
            
            # Setup scheduler
            self._setup_scheduler()
            
            # Setup callbacks
            self._setup_callbacks()
            
            logger.info("Ray Tune initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ray Tune: {e}")
            return False
    
    def _setup_search_algorithm(self):
        """Setup search algorithm"""
        try:
            config = self.config.search_algorithm_config or {}
            
            if self.config.search_algorithm == "bayesian":
                self.search_alg = BayesOptSearch(
                    metric=self.config.early_stopping_metric,
                    mode=self.config.early_stopping_mode,
                    **config
                )
                
            elif self.config.search_algorithm == "hyperopt":
                self.search_alg = HyperOptSearch(
                    metric=self.config.early_stopping_metric,
                    mode=self.config.early_stopping_mode,
                    **config
                )
                
            elif self.config.search_algorithm == "optuna":
                self.search_alg = OptunaSearch(
                    metric=self.config.early_stopping_metric,
                    mode=self.config.early_stopping_mode,
                    **config
                )
            
            # Apply concurrency limiter
            if self.search_alg and self.config.max_concurrent_trials > 1:
                self.search_alg = ConcurrencyLimiter(
                    self.search_alg, 
                    max_concurrent=self.config.max_concurrent_trials
                )
                
        except Exception as e:
            logger.warning(f"Failed to setup search algorithm: {e}")
            self.search_alg = None
    
    def _setup_scheduler(self):
        """Setup scheduler"""
        try:
            config = self.config.scheduler_config or {}
            
            if self.config.scheduler == "asha":
                self.scheduler = ASHAScheduler(
                    metric=self.config.early_stopping_metric,
                    mode=self.config.early_stopping_mode,
                    max_t=100,  # Maximum iterations per trial
                    grace_period=10,  # Minimum iterations before stopping
                    reduction_factor=2,
                    **config
                )
                
            elif self.config.scheduler == "pbt":
                self.scheduler = PopulationBasedTraining(
                    time_attr="training_iteration",
                    perturbation_interval=20,
                    hyperparam_mutations={
                        "lr": tune.loguniform(1e-5, 1e-1),
                        "batch_size": tune.choice([16, 32, 64, 128])
                    },
                    **config
                )
                
            elif self.config.scheduler == "median":
                self.scheduler = MedianStoppingRule(
                    time_attr="training_iteration",
                    metric=self.config.early_stopping_metric,
                    mode=self.config.early_stopping_mode,
                    grace_period=self.config.early_stopping_patience,
                    **config
                )
                
        except Exception as e:
            logger.warning(f"Failed to setup scheduler: {e}")
            self.scheduler = None
    
    def _setup_callbacks(self):
        """Setup logging callbacks"""
        try:
            # W&B integration
            if self.config.wandb_integration:
                try:
                    wandb_callback = WandbLoggerCallback(
                        project=self.config.experiment_name,
                        save_code=True
                    )
                    self.callbacks.append(wandb_callback)
                except Exception as e:
                    logger.warning(f"Failed to setup W&B callback: {e}")
            
            # MLflow integration
            if self.config.mlflow_integration:
                try:
                    mlflow_callback = MLflowLoggerCallback(
                        experiment_name=self.config.experiment_name,
                        save_artifact=True
                    )
                    self.callbacks.append(mlflow_callback)
                except Exception as e:
                    logger.warning(f"Failed to setup MLflow callback: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to setup callbacks: {e}")
    
    def run_hyperparameter_search(self, trainable: Callable, search_space: Dict[str, Any],
                                 num_samples: Optional[int] = None) -> Any:
        """Run hyperparameter search"""
        try:
            num_samples = num_samples or self.config.max_trials
            
            # Setup resources per trial
            resources = self.config.resources_per_trial or {"cpu": 1}
            
            # Setup stopping criteria
            stop_criteria = {}
            if self.config.time_budget_s:
                stop_criteria["time_total_s"] = self.config.time_budget_s
            
            # Setup reporter
            reporter = CLIReporter(
                parameter_columns=list(search_space.keys())[:4],  # Show first 4 params
                metric_columns=[self.config.early_stopping_metric, "training_iteration"]
            )
            
            # Run tuning
            analysis = tune.run(
                trainable,
                config=search_space,
                num_samples=num_samples,
                search_alg=self.search_alg,
                scheduler=self.scheduler,
                stop=stop_criteria,
                resources_per_trial=resources,
                local_dir=self.config.local_dir,
                name=self.config.experiment_name,
                progress_reporter=reporter,
                callbacks=self.callbacks,
                verbose=self.config.verbose,
                log_to_file=self.config.log_to_file
            )
            
            logger.info(f"Hyperparameter search completed with {len(analysis.trials)} trials")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to run hyperparameter search: {e}")
            return None
    
    def get_best_config(self, analysis, metric: Optional[str] = None, 
                       mode: Optional[str] = None) -> Dict[str, Any]:
        """Get best configuration from analysis"""
        try:
            metric = metric or self.config.early_stopping_metric
            mode = mode or self.config.early_stopping_mode
            
            best_config = analysis.get_best_config(metric=metric, mode=mode)
            best_trial = analysis.get_best_trial(metric=metric, mode=mode)
            
            result = {
                'config': best_config,
                'trial_id': best_trial.trial_id,
                'metrics': best_trial.last_result
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get best config: {e}")
            return {}
    
    def shutdown(self):
        """Shutdown Ray"""
        try:
            if ray.is_initialized():
                ray.shutdown()
                logger.info("Ray shutdown completed")
        except Exception as e:
            logger.warning(f"Error during Ray shutdown: {e}")


class RayTuneIntegration(BaseEnhancement):
    """Ray Tune integration for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="ray_tune_integration",
            version="1.0.0",
            enabled=RAY_TUNE_AVAILABLE,
            priority=20,
            dependencies=["mlflow_integration", "wandb_integration"],
            parameters={
                "ray_address": None,
                "num_cpus": None,
                "num_gpus": None,
                "memory": None,
                "search_algorithm": "bayesian",
                "search_algorithm_config": {},
                "scheduler": "asha",
                "scheduler_config": {},
                "max_trials": 100,
                "max_concurrent_trials": 4,
                "time_budget_s": None,
                "early_stopping_metric": "val_accuracy",
                "early_stopping_mode": "max",
                "early_stopping_patience": 10,
                "resources_per_trial": {"cpu": 1, "gpu": 0},
                "local_dir": "./ray_results",
                "experiment_name": "mle_star_tune",
                "log_to_file": True,
                "verbose": 1,
                "wandb_integration": True,
                "mlflow_integration": True
            }
        )
    
    def initialize(self) -> bool:
        """Initialize Ray Tune integration"""
        if not RAY_TUNE_AVAILABLE:
            self._logger.error("Ray Tune not available. Install with: pip install ray[tune] bayesian-optimization hyperopt optuna")
            return False
        
        try:
            # Create Ray Tune configuration
            self.ray_config = RayTuneConfig(**self.config.parameters)
            
            # Initialize trainer
            self.ray_trainer = RayTuneTrainer(self.ray_config)
            
            # Initialize Ray Tune
            success = self.ray_trainer.initialize()
            
            if success:
                self._logger.info("Ray Tune integration initialized successfully")
            else:
                self._logger.error("Failed to initialize Ray Tune integration")
                
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Ray Tune: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with Ray Tune capabilities"""
        enhanced = workflow.copy()
        
        # Add hyperparameter optimization configuration
        if 'hyperparameter_optimization' not in enhanced:
            enhanced['hyperparameter_optimization'] = {}
        
        enhanced['hyperparameter_optimization']['ray_tune'] = {
            'enabled': True,
            'search_algorithm': self.ray_config.search_algorithm,
            'scheduler': self.ray_config.scheduler,
            'max_trials': self.ray_config.max_trials,
            'max_concurrent_trials': self.ray_config.max_concurrent_trials,
            'early_stopping': {
                'metric': self.ray_config.early_stopping_metric,
                'mode': self.ray_config.early_stopping_mode,
                'patience': self.ray_config.early_stopping_patience
            },
            'resources_per_trial': self.ray_config.resources_per_trial,
            'integrations': {
                'wandb': self.ray_config.wandb_integration,
                'mlflow': self.ray_config.mlflow_integration
            }
        }
        
        # Add distributed training configuration
        if 'training' not in enhanced:
            enhanced['training'] = {}
        
        enhanced['training']['ray_tune_options'] = {
            'distributed': True if self.ray_config.ray_address else False,
            'num_cpus': self.ray_config.num_cpus,
            'num_gpus': self.ray_config.num_gpus,
            'memory': self.ray_config.memory,
            'parallel_trials': self.ray_config.max_concurrent_trials
        }
        
        # Enhance MLE-Star stages with Ray Tune capabilities
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 3: Action Planning - Add HPO strategy
            if '3_action_planning' in stages:
                if 'ray_tune_planning' not in stages['3_action_planning']:
                    stages['3_action_planning']['ray_tune_planning'] = [
                        'hyperparameter_search_space_definition',
                        'search_algorithm_selection',
                        'early_stopping_strategy',
                        'resource_allocation_planning'
                    ]
            
            # Stage 4: Implementation - Add distributed training
            if '4_implementation' in stages:
                if 'ray_tune_implementation' not in stages['4_implementation']:
                    stages['4_implementation']['ray_tune_implementation'] = [
                        'distributed_training_setup',
                        'parallel_hyperparameter_search',
                        'resource_efficient_training',
                        'real_time_optimization'
                    ]
            
            # Stage 6: Refinement - Add advanced optimization
            if '6_refinement' in stages:
                if 'ray_tune_optimization' not in stages['6_refinement']:
                    stages['6_refinement']['ray_tune_optimization'] = [
                        'population_based_training',
                        'bayesian_optimization',
                        'early_stopping_analysis',
                        'multi_objective_optimization'
                    ]
        
        # Add Ray Tune specific evaluation metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'ray_tune_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['ray_tune_metrics'] = [
                'total_trials_run',
                'best_trial_performance',
                'search_efficiency',
                'resource_utilization',
                'convergence_speed'
            ]
        
        self._logger.debug("Enhanced workflow with Ray Tune capabilities")
        return enhanced
    
    def create_search_space_from_config(self, param_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create Ray Tune search space from parameter configuration"""
        search_space = {}
        
        for param_name, param_def in param_config.items():
            if isinstance(param_def, dict):
                if param_def.get('type') == 'uniform':
                    search_space[param_name] = tune.uniform(
                        param_def['low'], param_def['high']
                    )
                elif param_def.get('type') == 'loguniform':
                    search_space[param_name] = tune.loguniform(
                        param_def['low'], param_def['high']
                    )
                elif param_def.get('type') == 'choice':
                    search_space[param_name] = tune.choice(param_def['choices'])
                elif param_def.get('type') == 'randint':
                    search_space[param_name] = tune.randint(
                        param_def['low'], param_def['high']
                    )
                elif param_def.get('type') == 'grid_search':
                    search_space[param_name] = tune.grid_search(param_def['values'])
            elif isinstance(param_def, list):
                # Treat as choices
                search_space[param_name] = tune.choice(param_def)
            else:
                # Fixed value
                search_space[param_name] = param_def
        
        return search_space
    
    def create_trainable_function(self, train_func: Callable, 
                                 data_loader_func: Optional[Callable] = None) -> Callable:
        """Create Ray Tune compatible trainable function"""
        def trainable(config):
            """Ray Tune trainable function"""
            try:
                # Load data if data loader provided
                if data_loader_func:
                    data = data_loader_func()
                else:
                    data = None
                
                # Run training with current config
                results = train_func(config, data)
                
                # Report metrics to Ray Tune
                for epoch, metrics in enumerate(results.get('history', [])):
                    tune.report(training_iteration=epoch, **metrics)
                
                # Report final metrics
                final_metrics = results.get('final_metrics', {})
                tune.report(**final_metrics)
                
            except Exception as e:
                logger.error(f"Error in trainable function: {e}")
                # Report error to Ray Tune
                tune.report(error=str(e))
        
        return trainable
    
    def run_mle_star_optimization(self, trainable_func: Callable, 
                                 search_space: Dict[str, Any],
                                 num_trials: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Run MLE-Star hyperparameter optimization"""
        try:
            # Run hyperparameter search
            analysis = self.ray_trainer.run_hyperparameter_search(
                trainable_func, search_space, num_trials
            )
            
            if analysis is None:
                return None
            
            # Get best configuration
            best_result = self.ray_trainer.get_best_config(analysis)
            
            # Add analysis summary
            best_result['analysis_summary'] = {
                'total_trials': len(analysis.trials),
                'completed_trials': len([t for t in analysis.trials if t.status == 'TERMINATED']),
                'failed_trials': len([t for t in analysis.trials if t.status == 'ERROR']),
                'best_trial_id': best_result.get('trial_id'),
                'experiment_path': analysis.get_best_logdir()
            }
            
            self._logger.info(f"MLE-Star optimization completed. Best {self.ray_config.early_stopping_metric}: "
                            f"{best_result['metrics'].get(self.ray_config.early_stopping_metric)}")
            
            return best_result
            
        except Exception as e:
            self._logger.error(f"Failed to run MLE-Star optimization: {e}")
            return None
    
    def cleanup(self):
        """Cleanup Ray resources"""
        try:
            self.ray_trainer.shutdown()
        except Exception as e:
            self._logger.warning(f"Error during cleanup: {e}")