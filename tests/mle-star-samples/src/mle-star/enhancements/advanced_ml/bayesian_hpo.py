"""
Bayesian Hyperparameter Optimization
====================================

Advanced hyperparameter optimization using Bayesian methods for MLE-Star:
- Gaussian Process-based optimization
- Tree-structured Parzen Estimator (TPE)
- Multi-objective optimization
- Early stopping and pruning
- Acquisition function strategies
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass
import json
from pathlib import Path

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
    from optuna.integration import SklearnIntegration
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from skopt import gp_minimize, forest_minimize, gbrt_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.acquisition import gaussian_ei, gaussian_lcb, gaussian_pi
    from skopt.utils import use_named_args
    SCIKIT_OPTIMIZE_AVAILABLE = True
except ImportError:
    SCIKIT_OPTIMIZE_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
    from hyperopt.mongoexp import MongoTrials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class BayesianHPOConfig:
    """Configuration for Bayesian HPO"""
    # Optimization method
    method: str = "optuna"  # optuna, scikit-optimize, hyperopt
    
    # Optuna configuration
    optuna_sampler: str = "tpe"  # tpe, cmaes, random
    optuna_pruner: str = "median"  # median, successive_halving, hyperband, none
    optuna_direction: str = "maximize"  # maximize, minimize
    optuna_storage: Optional[str] = None  # Database URL for persistence
    
    # Scikit-optimize configuration
    skopt_base_estimator: str = "gp"  # gp, rf, et, gbrt
    skopt_acquisition: str = "EI"  # EI, LCB, PI
    skopt_n_initial_points: int = 10
    
    # HyperOpt configuration
    hyperopt_algo: str = "tpe"  # tpe, random, adaptive_tpe
    hyperopt_trials_storage: Optional[str] = None  # MongoDB URL
    
    # General optimization settings
    n_trials: int = 100
    timeout: Optional[int] = None  # Seconds
    n_jobs: int = 1  # Parallel optimization
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 20
    min_trials_for_stopping: int = 30
    
    # Multi-objective optimization
    multi_objective: bool = False
    objectives: List[str] = None  # List of metric names
    
    # Results and logging
    save_history: bool = True
    history_path: str = "./hpo_history"
    log_level: str = "INFO"


class OptunaBayesianOptimizer:
    """Optuna-based Bayesian optimization"""
    
    def __init__(self, config: BayesianHPOConfig):
        self.config = config
        self.study = None
        
    def initialize(self) -> bool:
        """Initialize Optuna study"""
        try:
            # Setup sampler
            if self.config.optuna_sampler == "tpe":
                sampler = TPESampler()
            elif self.config.optuna_sampler == "cmaes":
                sampler = CmaEsSampler()
            else:
                sampler = RandomSampler()
            
            # Setup pruner
            if self.config.optuna_pruner == "median":
                pruner = MedianPruner(
                    n_startup_trials=self.config.min_trials_for_stopping,
                    n_warmup_steps=self.config.early_stopping_patience
                )
            elif self.config.optuna_pruner == "successive_halving":
                pruner = SuccessiveHalvingPruner()
            elif self.config.optuna_pruner == "hyperband":
                pruner = HyperbandPruner()
            else:
                pruner = None
            
            # Create study
            study_name = f"mle_star_hpo_{hash(str(self.config))}"
            
            if self.config.multi_objective:
                directions = ['maximize' if self.config.optuna_direction == 'maximize' 
                            else 'minimize' for _ in self.config.objectives]
                self.study = optuna.create_study(
                    study_name=study_name,
                    directions=directions,
                    sampler=sampler,
                    pruner=pruner,
                    storage=self.config.optuna_storage,
                    load_if_exists=True
                )
            else:
                self.study = optuna.create_study(
                    study_name=study_name,
                    direction=self.config.optuna_direction,
                    sampler=sampler,
                    pruner=pruner,
                    storage=self.config.optuna_storage,
                    load_if_exists=True
                )
            
            logger.info(f"Initialized Optuna study: {study_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Optuna: {e}")
            return False
    
    def optimize(self, objective_func: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run Bayesian optimization"""
        try:
            def optuna_objective(trial):
                """Optuna objective function wrapper"""
                params = {}
                
                # Sample parameters from search space
                for param_name, param_config in search_space.items():
                    if isinstance(param_config, dict):
                        param_type = param_config.get('type')
                        
                        if param_type == 'uniform':
                            params[param_name] = trial.suggest_float(
                                param_name, param_config['low'], param_config['high']
                            )
                        elif param_type == 'loguniform':
                            params[param_name] = trial.suggest_float(
                                param_name, param_config['low'], param_config['high'], log=True
                            )
                        elif param_type == 'int':
                            params[param_name] = trial.suggest_int(
                                param_name, param_config['low'], param_config['high']
                            )
                        elif param_type == 'categorical':
                            params[param_name] = trial.suggest_categorical(
                                param_name, param_config['choices']
                            )
                        elif param_type == 'discrete':
                            params[param_name] = trial.suggest_discrete_uniform(
                                param_name, param_config['low'], param_config['high'], param_config['q']
                            )
                    else:
                        # Fixed parameter
                        params[param_name] = param_config
                
                # Evaluate objective function
                result = objective_func(params)
                
                # Handle different result formats
                if isinstance(result, dict):
                    if self.config.multi_objective:
                        return [result[obj] for obj in self.config.objectives]
                    else:
                        # Use first metric as primary objective
                        primary_metric = list(result.keys())[0]
                        
                        # Log additional metrics
                        for metric_name, value in result.items():
                            if metric_name != primary_metric:
                                trial.set_user_attr(metric_name, value)
                        
                        return result[primary_metric]
                else:
                    return result
            
            # Run optimization
            self.study.optimize(
                optuna_objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.n_jobs
            )
            
            # Get results
            if self.config.multi_objective:
                best_trials = self.study.best_trials
                results = {
                    'best_trials': [
                        {
                            'params': trial.params,
                            'values': trial.values,
                            'number': trial.number
                        }
                        for trial in best_trials
                    ],
                    'n_trials': len(self.study.trials),
                    'optimization_history': [
                        {
                            'trial': trial.number,
                            'values': trial.values,
                            'params': trial.params
                        }
                        for trial in self.study.trials
                    ]
                }
            else:
                best_trial = self.study.best_trial
                results = {
                    'best_params': best_trial.params,
                    'best_value': best_trial.value,
                    'best_trial_number': best_trial.number,
                    'n_trials': len(self.study.trials),
                    'optimization_history': [
                        {
                            'trial': trial.number,
                            'value': trial.value,
                            'params': trial.params
                        }
                        for trial in self.study.trials if trial.value is not None
                    ]
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Optuna optimization failed: {e}")
            return {}


class ScikitOptimizeBayesianOptimizer:
    """Scikit-optimize based Bayesian optimization"""
    
    def __init__(self, config: BayesianHPOConfig):
        self.config = config
        
    def optimize(self, objective_func: Callable, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """Run Bayesian optimization using scikit-optimize"""
        try:
            # Convert search space to scikit-optimize format
            dimensions = []
            param_names = []
            
            for param_name, param_config in search_space.items():
                param_names.append(param_name)
                
                if isinstance(param_config, dict):
                    param_type = param_config.get('type')
                    
                    if param_type in ['uniform', 'loguniform']:
                        prior = 'log-uniform' if param_type == 'loguniform' else 'uniform'
                        dimensions.append(Real(
                            param_config['low'], 
                            param_config['high'], 
                            prior=prior,
                            name=param_name
                        ))
                    elif param_type == 'int':
                        dimensions.append(Integer(
                            param_config['low'],
                            param_config['high'],
                            name=param_name
                        ))
                    elif param_type == 'categorical':
                        dimensions.append(Categorical(
                            param_config['choices'],
                            name=param_name
                        ))
                else:
                    # Fixed parameter - skip in optimization
                    continue
            
            # Objective function wrapper
            @use_named_args(dimensions)
            def skopt_objective(**params):
                # Add fixed parameters
                full_params = params.copy()
                for param_name, param_config in search_space.items():
                    if not isinstance(param_config, dict):
                        full_params[param_name] = param_config
                
                result = objective_func(full_params)
                
                # Convert to single value (scikit-optimize doesn't support multi-objective)
                if isinstance(result, dict):
                    primary_metric = list(result.keys())[0]
                    value = result[primary_metric]
                else:
                    value = result
                
                # Convert to minimization problem if needed
                if self.config.optuna_direction == 'maximize':
                    value = -value
                
                return value
            
            # Choose optimization method
            if self.config.skopt_base_estimator == "gp":
                optimizer_func = gp_minimize
            elif self.config.skopt_base_estimator == "rf":
                optimizer_func = forest_minimize
            else:
                optimizer_func = gbrt_minimize
            
            # Set acquisition function
            acq_func = self.config.skopt_acquisition.lower()
            
            # Run optimization
            result = optimizer_func(
                func=skopt_objective,
                dimensions=dimensions,
                n_calls=self.config.n_trials,
                n_initial_points=self.config.skopt_n_initial_points,
                acq_func=acq_func,
                n_jobs=self.config.n_jobs
            )
            
            # Extract results
            best_params = dict(zip(param_names, result.x))
            best_value = result.fun
            
            # Convert back from minimization if needed
            if self.config.optuna_direction == 'maximize':
                best_value = -best_value
            
            # Prepare optimization history
            history = []
            for i, (params, value) in enumerate(zip(result.x_iters, result.func_vals)):
                param_dict = dict(zip(param_names, params))
                if self.config.optuna_direction == 'maximize':
                    value = -value
                
                history.append({
                    'trial': i,
                    'value': value,
                    'params': param_dict
                })
            
            results = {
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(result.x_iters),
                'optimization_history': history,
                'convergence': result.func_vals.tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Scikit-optimize optimization failed: {e}")
            return {}


class BayesianHPO(BaseEnhancement):
    """Bayesian Hyperparameter Optimization enhancement for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="bayesian_hpo",
            version="1.0.0",
            enabled=OPTUNA_AVAILABLE or SCIKIT_OPTIMIZE_AVAILABLE or HYPEROPT_AVAILABLE,
            priority=40,
            parameters={
                "method": "optuna" if OPTUNA_AVAILABLE else ("scikit-optimize" if SCIKIT_OPTIMIZE_AVAILABLE else "hyperopt"),
                "optuna_sampler": "tpe",
                "optuna_pruner": "median",
                "optuna_direction": "maximize",
                "optuna_storage": None,
                "skopt_base_estimator": "gp",
                "skopt_acquisition": "EI",
                "skopt_n_initial_points": 10,
                "hyperopt_algo": "tpe",
                "hyperopt_trials_storage": None,
                "n_trials": 100,
                "timeout": None,
                "n_jobs": 1,
                "early_stopping_enabled": True,
                "early_stopping_patience": 20,
                "min_trials_for_stopping": 30,
                "multi_objective": False,
                "objectives": ["val_accuracy", "val_loss"],
                "save_history": True,
                "history_path": "./hpo_history",
                "log_level": "INFO"
            }
        )
    
    def initialize(self) -> bool:
        """Initialize Bayesian HPO"""
        available_methods = []
        
        if OPTUNA_AVAILABLE:
            available_methods.append("optuna")
        if SCIKIT_OPTIMIZE_AVAILABLE:
            available_methods.append("scikit-optimize")
        if HYPEROPT_AVAILABLE:
            available_methods.append("hyperopt")
        
        if not available_methods:
            self._logger.error("No Bayesian optimization libraries available. Install optuna, scikit-optimize, or hyperopt")
            return False
        
        try:
            # Create configuration
            self.hpo_config = BayesianHPOConfig(**self.config.parameters)
            
            # Validate method
            if self.hpo_config.method not in available_methods:
                self._logger.warning(f"Method {self.hpo_config.method} not available. Using {available_methods[0]}")
                self.hpo_config.method = available_methods[0]
            
            # Initialize optimizer based on method
            if self.hpo_config.method == "optuna":
                self.optimizer = OptunaBayesianOptimizer(self.hpo_config)
                success = self.optimizer.initialize()
            elif self.hpo_config.method == "scikit-optimize":
                self.optimizer = ScikitOptimizeBayesianOptimizer(self.hpo_config)
                success = True
            else:
                # HyperOpt implementation would go here
                self._logger.warning("HyperOpt implementation not yet available")
                success = False
            
            if success:
                self._logger.info(f"Bayesian HPO initialized with method: {self.hpo_config.method}")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Bayesian HPO: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with Bayesian HPO capabilities"""
        enhanced = workflow.copy()
        
        # Add Bayesian HPO configuration
        if 'hyperparameter_optimization' not in enhanced:
            enhanced['hyperparameter_optimization'] = {}
        
        enhanced['hyperparameter_optimization']['bayesian'] = {
            'enabled': True,
            'method': self.hpo_config.method,
            'n_trials': self.hpo_config.n_trials,
            'timeout': self.hpo_config.timeout,
            'multi_objective': self.hpo_config.multi_objective,
            'objectives': self.hpo_config.objectives,
            'early_stopping': {
                'enabled': self.hpo_config.early_stopping_enabled,
                'patience': self.hpo_config.early_stopping_patience,
                'min_trials': self.hpo_config.min_trials_for_stopping
            },
            'parallel_jobs': self.hpo_config.n_jobs
        }
        
        # Enhance MLE-Star stages with Bayesian HPO
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 3: Action Planning - Add HPO strategy
            if '3_action_planning' in stages:
                if 'bayesian_hpo_planning' not in stages['3_action_planning']:
                    stages['3_action_planning']['bayesian_hpo_planning'] = [
                        'hyperparameter_space_design',
                        'acquisition_function_selection',
                        'surrogate_model_configuration',
                        'multi_objective_strategy'
                    ]
            
            # Stage 6: Refinement - Add Bayesian optimization
            if '6_refinement' in stages:
                if 'bayesian_optimization' not in stages['6_refinement']:
                    stages['6_refinement']['bayesian_optimization'] = [
                        'gaussian_process_modeling',
                        'acquisition_function_optimization',
                        'uncertainty_quantification',
                        'pareto_frontier_analysis'
                    ]
        
        # Add Bayesian HPO specific metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'bayesian_hpo_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['bayesian_hpo_metrics'] = [
                'convergence_rate',
                'acquisition_function_values',
                'uncertainty_estimates',
                'exploration_exploitation_balance',
                'hyperparameter_importance'
            ]
        
        self._logger.debug("Enhanced workflow with Bayesian HPO capabilities")
        return enhanced
    
    def create_search_space_template(self, model_type: str) -> Dict[str, Any]:
        """Create search space template for common model types"""
        templates = {
            'random_forest': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]}
            },
            'xgboost': {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 1000},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'learning_rate': {'type': 'loguniform', 'low': 0.01, 'high': 0.3},
                'subsample': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'uniform', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'loguniform', 'low': 1e-8, 'high': 1.0},
                'reg_lambda': {'type': 'loguniform', 'low': 1e-8, 'high': 1.0}
            },
            'neural_network': {
                'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-1},
                'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128, 256]},
                'hidden_layers': {'type': 'int', 'low': 1, 'high': 5},
                'hidden_units': {'type': 'categorical', 'choices': [64, 128, 256, 512]},
                'dropout_rate': {'type': 'uniform', 'low': 0.0, 'high': 0.5},
                'optimizer': {'type': 'categorical', 'choices': ['adam', 'sgd', 'rmsprop']}
            },
            'svm': {
                'C': {'type': 'loguniform', 'low': 1e-3, 'high': 1e3},
                'gamma': {'type': 'loguniform', 'low': 1e-6, 'high': 1e1},
                'kernel': {'type': 'categorical', 'choices': ['rbf', 'poly', 'sigmoid']},
                'degree': {'type': 'int', 'low': 2, 'high': 5}  # Only for poly kernel
            }
        }
        
        return templates.get(model_type, {})
    
    def optimize_hyperparameters(self, objective_func: Callable, 
                                search_space: Dict[str, Any],
                                study_name: Optional[str] = None) -> Dict[str, Any]:
        """Run Bayesian hyperparameter optimization"""
        try:
            # Save current search space and configuration
            if self.hpo_config.save_history:
                history_dir = Path(self.hpo_config.history_path)
                history_dir.mkdir(parents=True, exist_ok=True)
                
                config_file = history_dir / f"{study_name or 'hpo_study'}_config.json"
                with open(config_file, 'w') as f:
                    json.dump({
                        'search_space': search_space,
                        'config': self.hpo_config.__dict__
                    }, f, indent=2)
            
            # Run optimization
            results = self.optimizer.optimize(objective_func, search_space)
            
            # Save results
            if self.hpo_config.save_history and results:
                results_file = history_dir / f"{study_name or 'hpo_study'}_results.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                
                self._logger.info(f"HPO results saved to: {results_file}")
            
            return results
            
        except Exception as e:
            self._logger.error(f"Hyperparameter optimization failed: {e}")
            return {}
    
    def analyze_optimization_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze optimization results and provide insights"""
        try:
            analysis = {}
            
            if 'optimization_history' in results:
                history = results['optimization_history']
                
                # Convergence analysis
                values = [trial['value'] for trial in history if 'value' in trial]
                if values:
                    analysis['convergence'] = {
                        'total_trials': len(values),
                        'best_value': max(values) if self.hpo_config.optuna_direction == 'maximize' else min(values),
                        'improvement_over_time': np.diff(np.maximum.accumulate(values) if self.hpo_config.optuna_direction == 'maximize' 
                                                       else np.minimum.accumulate(values)).tolist(),
                        'convergence_trial': len(values) - np.argmax(values[::-1]) - 1 if self.hpo_config.optuna_direction == 'maximize'
                                           else len(values) - np.argmin(values[::-1]) - 1
                    }
                
                # Parameter importance (simple correlation analysis)
                if len(history) > 10:
                    param_importance = {}
                    all_params = set()
                    
                    for trial in history:
                        if 'params' in trial:
                            all_params.update(trial['params'].keys())
                    
                    for param in all_params:
                        param_values = []
                        objective_values = []
                        
                        for trial in history:
                            if 'params' in trial and param in trial['params'] and 'value' in trial:
                                param_values.append(trial['params'][param])
                                objective_values.append(trial['value'])
                        
                        if len(param_values) > 1 and len(set(param_values)) > 1:
                            # Calculate correlation
                            try:
                                correlation = np.corrcoef(param_values, objective_values)[0, 1]
                                param_importance[param] = abs(correlation) if not np.isnan(correlation) else 0
                            except:
                                param_importance[param] = 0
                    
                    analysis['parameter_importance'] = dict(sorted(
                        param_importance.items(), key=lambda x: x[1], reverse=True
                    ))
            
            # Best configuration analysis
            if 'best_params' in results:
                analysis['best_configuration'] = {
                    'parameters': results['best_params'],
                    'performance': results.get('best_value'),
                    'trial_number': results.get('best_trial_number')
                }
            
            return analysis
            
        except Exception as e:
            self._logger.error(f"Failed to analyze optimization results: {e}")
            return {}