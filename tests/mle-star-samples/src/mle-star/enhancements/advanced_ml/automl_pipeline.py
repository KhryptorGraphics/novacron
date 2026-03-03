"""
AutoML Pipeline
===============

Automated machine learning pipeline for MLE-Star:
- Automated feature engineering and selection
- Algorithm selection and hyperparameter optimization
- Ensemble model construction
- Pipeline optimization and validation
- Meta-learning for improved efficiency
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json
from abc import ABC, abstractmethod

try:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder
    from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, RFECV
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVC, SVR
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML Pipeline"""
    # General settings
    time_limit: Optional[int] = None  # Time limit in seconds
    task_type: str = "auto"  # auto, classification, regression
    
    # Data preprocessing
    handle_missing: str = "auto"  # auto, drop, impute_mean, impute_median, impute_mode
    handle_categorical: str = "auto"  # auto, drop, encode, one_hot
    feature_scaling: str = "auto"  # auto, none, standard, robust, minmax
    
    # Feature engineering
    feature_generation: bool = True
    polynomial_features: bool = True
    interaction_features: bool = True
    statistical_features: bool = True
    
    # Feature selection
    feature_selection: bool = True
    selection_methods: List[str] = None  # univariate, rfe, lasso, rf_importance
    max_features: Optional[int] = None
    feature_selection_cv: int = 5
    
    # Algorithm selection
    algorithms: Optional[List[str]] = None  # If None, use all available
    exclude_algorithms: List[str] = None
    
    # Hyperparameter optimization
    hpo_method: str = "random"  # random, bayesian, evolutionary
    hpo_iterations: int = 100
    hpo_cv_folds: int = 3
    
    # Ensemble methods
    ensemble_methods: List[str] = None  # voting, stacking, blending
    ensemble_size: int = 5
    
    # Validation
    validation_strategy: str = "holdout"  # holdout, cv, stratified_cv
    validation_split: float = 0.2
    cv_folds: int = 5
    
    # Meta-learning
    use_meta_learning: bool = True
    meta_features: bool = True
    warm_start: bool = True
    
    # Output
    save_pipeline: bool = True
    save_models: bool = True
    output_path: str = "./automl_output"


class FeatureEngineer:
    """Automated feature engineering"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.feature_generators = []
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply feature engineering transformations"""
        X_engineered = X.copy()
        
        if self.config.feature_generation:
            # Polynomial features
            if self.config.polynomial_features:
                X_engineered = self._generate_polynomial_features(X_engineered)
            
            # Interaction features
            if self.config.interaction_features:
                X_engineered = self._generate_interaction_features(X_engineered)
            
            # Statistical features
            if self.config.statistical_features:
                X_engineered = self._generate_statistical_features(X_engineered)
        
        return X_engineered
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply transformations to new data"""
        X_engineered = X.copy()
        
        # Apply stored transformations
        for generator in self.feature_generators:
            X_engineered = generator.transform(X_engineered)
        
        return X_engineered
    
    def _generate_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features"""
        from sklearn.preprocessing import PolynomialFeatures
        
        # Only use numerical columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) == 0:
            return X
        
        # Limit to prevent explosion of features
        max_cols = min(5, len(numerical_cols))
        selected_cols = numerical_cols[:max_cols]
        
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X_poly = poly.fit_transform(X[selected_cols])
        
        # Create column names
        feature_names = poly.get_feature_names_out(selected_cols)
        X_poly_df = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
        
        # Remove original columns to avoid duplication
        X_new = X.drop(columns=selected_cols)
        X_new = pd.concat([X_new, X_poly_df], axis=1)
        
        return X_new
    
    def _generate_interaction_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate interaction features"""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return X
        
        X_new = X.copy()
        
        # Generate pairwise interactions (limited to prevent feature explosion)
        max_interactions = min(10, len(numerical_cols) * (len(numerical_cols) - 1) // 2)
        interactions_added = 0
        
        for i, col1 in enumerate(numerical_cols):
            for j, col2 in enumerate(numerical_cols[i+1:], i+1):
                if interactions_added >= max_interactions:
                    break
                
                # Multiplication
                X_new[f"{col1}_x_{col2}"] = X[col1] * X[col2]
                
                # Division (avoid division by zero)
                X_new[f"{col1}_div_{col2}"] = X[col1] / (X[col2] + 1e-8)
                
                interactions_added += 2
        
        return X_new
    
    def _generate_statistical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical features"""
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) < 2:
            return X
        
        X_new = X.copy()
        
        # Row-wise statistics
        X_new['row_mean'] = X[numerical_cols].mean(axis=1)
        X_new['row_std'] = X[numerical_cols].std(axis=1)
        X_new['row_min'] = X[numerical_cols].min(axis=1)
        X_new['row_max'] = X[numerical_cols].max(axis=1)
        X_new['row_median'] = X[numerical_cols].median(axis=1)
        
        return X_new


class AlgorithmSelector:
    """Automated algorithm selection"""
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.available_algorithms = self._get_available_algorithms()
        
    def _get_available_algorithms(self) -> Dict[str, Any]:
        """Get available algorithms based on installed packages"""
        algorithms = {}
        
        if SKLEARN_AVAILABLE:
            algorithms.update({
                'random_forest': {
                    'classifier': RandomForestClassifier,
                    'regressor': RandomForestRegressor,
                    'params': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, 20, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    }
                },
                'logistic_regression': {
                    'classifier': LogisticRegression,
                    'regressor': Ridge,
                    'params': {
                        'C': [0.1, 1.0, 10.0, 100.0],
                        'solver': ['liblinear', 'lbfgs'],
                        'max_iter': [1000]
                    }
                },
                'svm': {
                    'classifier': SVC,
                    'regressor': SVR,
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'kernel': ['rbf', 'linear'],
                        'gamma': ['scale', 'auto']
                    }
                },
                'knn': {
                    'classifier': KNeighborsClassifier,
                    'regressor': KNeighborsRegressor,
                    'params': {
                        'n_neighbors': [3, 5, 7, 10],
                        'weights': ['uniform', 'distance']
                    }
                },
                'naive_bayes': {
                    'classifier': GaussianNB,
                    'regressor': None,
                    'params': {}
                }
            })
        
        if XGBOOST_AVAILABLE:
            algorithms['xgboost'] = {
                'classifier': xgb.XGBClassifier,
                'regressor': xgb.XGBRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
            }
        
        if LIGHTGBM_AVAILABLE:
            algorithms['lightgbm'] = {
                'classifier': lgb.LGBMClassifier,
                'regressor': lgb.LGBMRegressor,
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 6, 10],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'num_leaves': [31, 50, 100],
                    'subsample': [0.8, 1.0]
                }
            }
        
        return algorithms
    
    def select_algorithms(self, task_type: str) -> List[str]:
        """Select appropriate algorithms for the task"""
        algorithms = self.config.algorithms or list(self.available_algorithms.keys())
        
        # Filter by task type and availability
        filtered_algorithms = []
        for alg_name in algorithms:
            if alg_name in self.available_algorithms:
                alg_config = self.available_algorithms[alg_name]
                
                if task_type == 'classification' and alg_config['classifier'] is not None:
                    filtered_algorithms.append(alg_name)
                elif task_type == 'regression' and alg_config['regressor'] is not None:
                    filtered_algorithms.append(alg_name)
        
        # Remove excluded algorithms
        if self.config.exclude_algorithms:
            filtered_algorithms = [alg for alg in filtered_algorithms 
                                 if alg not in self.config.exclude_algorithms]
        
        return filtered_algorithms
    
    def get_algorithm_instance(self, algorithm_name: str, task_type: str, 
                             hyperparameters: Dict[str, Any] = None):
        """Get algorithm instance with hyperparameters"""
        if algorithm_name not in self.available_algorithms:
            raise ValueError(f"Algorithm {algorithm_name} not available")
        
        alg_config = self.available_algorithms[algorithm_name]
        
        if task_type == 'classification':
            algorithm_class = alg_config['classifier']
        else:
            algorithm_class = alg_config['regressor']
        
        if algorithm_class is None:
            raise ValueError(f"Algorithm {algorithm_name} not available for {task_type}")
        
        # Apply hyperparameters
        params = hyperparameters or {}
        return algorithm_class(**params)


class AutoMLPipeline(BaseEnhancement):
    """AutoML Pipeline enhancement for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="automl_pipeline",
            version="1.0.0",
            enabled=SKLEARN_AVAILABLE,
            priority=30,
            parameters={
                "time_limit": None,
                "task_type": "auto",
                "handle_missing": "auto",
                "handle_categorical": "auto",
                "feature_scaling": "auto",
                "feature_generation": True,
                "polynomial_features": True,
                "interaction_features": True,
                "statistical_features": True,
                "feature_selection": True,
                "selection_methods": ["univariate", "rfe"],
                "max_features": None,
                "feature_selection_cv": 5,
                "algorithms": None,
                "exclude_algorithms": [],
                "hpo_method": "random",
                "hpo_iterations": 100,
                "hpo_cv_folds": 3,
                "ensemble_methods": ["voting"],
                "ensemble_size": 5,
                "validation_strategy": "holdout",
                "validation_split": 0.2,
                "cv_folds": 5,
                "use_meta_learning": True,
                "meta_features": True,
                "warm_start": True,
                "save_pipeline": True,
                "save_models": True,
                "output_path": "./automl_output"
            }
        )
    
    def initialize(self) -> bool:
        """Initialize AutoML Pipeline"""
        if not SKLEARN_AVAILABLE:
            self._logger.error("Scikit-learn not available for AutoML")
            return False
        
        try:
            # Create configuration
            self.automl_config = AutoMLConfig(**self.config.parameters)
            
            # Initialize components
            self.feature_engineer = FeatureEngineer(self.automl_config)
            self.algorithm_selector = AlgorithmSelector(self.automl_config)
            
            # Initialize pipeline storage
            self.best_pipeline = None
            self.pipeline_history = []
            
            self._logger.info("AutoML Pipeline initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize AutoML Pipeline: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with AutoML capabilities"""
        enhanced = workflow.copy()
        
        # Add AutoML configuration
        if 'automl' not in enhanced:
            enhanced['automl'] = {}
        
        enhanced['automl'] = {
            'enabled': True,
            'task_type': self.automl_config.task_type,
            'time_limit': self.automl_config.time_limit,
            'feature_engineering': {
                'enabled': self.automl_config.feature_generation,
                'polynomial_features': self.automl_config.polynomial_features,
                'interaction_features': self.automl_config.interaction_features,
                'statistical_features': self.automl_config.statistical_features
            },
            'feature_selection': {
                'enabled': self.automl_config.feature_selection,
                'methods': self.automl_config.selection_methods,
                'max_features': self.automl_config.max_features
            },
            'hyperparameter_optimization': {
                'method': self.automl_config.hpo_method,
                'iterations': self.automl_config.hpo_iterations,
                'cv_folds': self.automl_config.hpo_cv_folds
            },
            'ensemble': {
                'enabled': bool(self.automl_config.ensemble_methods),
                'methods': self.automl_config.ensemble_methods,
                'size': self.automl_config.ensemble_size
            },
            'validation': {
                'strategy': self.automl_config.validation_strategy,
                'split': self.automl_config.validation_split,
                'cv_folds': self.automl_config.cv_folds
            }
        }
        
        # Enhance MLE-Star stages with AutoML
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 1: Situation Analysis - Add automated data analysis
            if '1_situation_analysis' in stages:
                if 'automl_analysis' not in stages['1_situation_analysis']:
                    stages['1_situation_analysis']['automl_analysis'] = [
                        'automated_data_profiling',
                        'task_type_detection',
                        'data_quality_assessment',
                        'meta_feature_extraction'
                    ]
            
            # Stage 2: Task Definition - Add automated objective setting
            if '2_task_definition' in stages:
                if 'automl_task_definition' not in stages['2_task_definition']:
                    stages['2_task_definition']['automl_task_definition'] = [
                        'metric_selection',
                        'validation_strategy_selection',
                        'resource_allocation',
                        'success_criteria_definition'
                    ]
            
            # Stage 3: Action Planning - Add automated planning
            if '3_action_planning' in stages:
                if 'automl_planning' not in stages['3_action_planning']:
                    stages['3_action_planning']['automl_planning'] = [
                        'preprocessing_strategy',
                        'feature_engineering_plan',
                        'algorithm_selection_strategy',
                        'ensemble_strategy'
                    ]
            
            # Stage 4: Implementation - Add AutoML execution
            if '4_implementation' in stages:
                if 'automl_implementation' not in stages['4_implementation']:
                    stages['4_implementation']['automl_implementation'] = [
                        'automated_preprocessing',
                        'feature_engineering_execution',
                        'algorithm_optimization',
                        'model_training_pipeline'
                    ]
            
            # Stage 5: Results Evaluation - Add automated evaluation
            if '5_results_evaluation' in stages:
                if 'automl_evaluation' not in stages['5_results_evaluation']:
                    stages['5_results_evaluation']['automl_evaluation'] = [
                        'model_comparison',
                        'ensemble_evaluation',
                        'feature_importance_analysis',
                        'performance_profiling'
                    ]
            
            # Stage 6: Refinement - Add automated optimization
            if '6_refinement' in stages:
                if 'automl_refinement' not in stages['6_refinement']:
                    stages['6_refinement']['automl_refinement'] = [
                        'pipeline_optimization',
                        'ensemble_refinement',
                        'hyperparameter_fine_tuning',
                        'feature_selection_optimization'
                    ]
        
        # Add AutoML specific evaluation metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'automl_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['automl_metrics'] = [
                'pipeline_efficiency',
                'automation_degree',
                'feature_engineering_impact',
                'ensemble_improvement',
                'search_efficiency'
            ]
        
        self._logger.debug("Enhanced workflow with AutoML capabilities")
        return enhanced
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
           task_type: Optional[str] = None) -> Dict[str, Any]:
        """Fit AutoML pipeline"""
        try:
            import time
            start_time = time.time()
            
            self._logger.info("Starting AutoML pipeline fitting")
            
            # Detect task type if not provided
            if task_type is None:
                task_type = self._detect_task_type(y)
            
            # Data preprocessing
            X_processed, preprocessor = self._preprocess_data(X, y)
            
            # Feature engineering
            X_engineered = self.feature_engineer.fit_transform(X_processed, y)
            
            # Feature selection
            X_selected, feature_selector = self._select_features(X_engineered, y, task_type)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_selected, y, test_size=self.automl_config.validation_split, 
                random_state=42, stratify=y if task_type == 'classification' else None
            )
            
            # Algorithm selection and optimization
            best_models = self._optimize_algorithms(X_train, y_train, X_val, y_val, task_type)
            
            # Ensemble creation
            ensemble_model = self._create_ensemble(best_models, X_val, y_val, task_type)
            
            # Build final pipeline
            pipeline = {
                'preprocessor': preprocessor,
                'feature_engineer': self.feature_engineer,
                'feature_selector': feature_selector,
                'model': ensemble_model,
                'task_type': task_type,
                'feature_names': X_selected.columns.tolist(),
                'target_classes': y.unique().tolist() if task_type == 'classification' else None
            }
            
            # Evaluate final pipeline
            final_score = self._evaluate_pipeline(pipeline, X_val, y_val, task_type)
            
            # Store best pipeline
            self.best_pipeline = pipeline
            
            elapsed_time = time.time() - start_time
            
            results = {
                'pipeline': pipeline,
                'final_score': final_score,
                'task_type': task_type,
                'elapsed_time': elapsed_time,
                'feature_count': len(X_selected.columns),
                'original_feature_count': len(X.columns),
                'best_models': best_models[:5]  # Top 5 models
            }
            
            # Save pipeline if configured
            if self.automl_config.save_pipeline:
                self._save_pipeline(results)
            
            self._logger.info(f"AutoML pipeline completed in {elapsed_time:.2f}s. Final score: {final_score:.4f}")
            
            return results
            
        except Exception as e:
            self._logger.error(f"AutoML pipeline fitting failed: {e}")
            return {}
    
    def _detect_task_type(self, y: pd.Series) -> str:
        """Detect task type from target variable"""
        if self.automl_config.task_type != "auto":
            return self.automl_config.task_type
        
        # Check if classification or regression
        if y.dtype == 'object' or len(y.unique()) < 20:
            return 'classification'
        else:
            return 'regression'
    
    def _preprocess_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess data"""
        X_processed = X.copy()
        preprocessor = {'transformations': []}
        
        # Handle missing values
        if X_processed.isnull().any().any():
            if self.automl_config.handle_missing == "auto":
                # Use median for numerical, mode for categorical
                for col in X_processed.columns:
                    if X_processed[col].isnull().any():
                        if X_processed[col].dtype in ['int64', 'float64']:
                            fill_value = X_processed[col].median()
                        else:
                            fill_value = X_processed[col].mode()[0] if not X_processed[col].mode().empty else "missing"
                        
                        X_processed[col].fillna(fill_value, inplace=True)
                        preprocessor['transformations'].append({
                            'type': 'fillna',
                            'column': col,
                            'value': fill_value
                        })
        
        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            if self.automl_config.handle_categorical == "auto":
                # One-hot encode categorical variables
                X_processed = pd.get_dummies(X_processed, columns=categorical_cols, drop_first=True)
                preprocessor['transformations'].append({
                    'type': 'one_hot',
                    'columns': categorical_cols.tolist()
                })
        
        # Feature scaling
        numerical_cols = X_processed.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0 and self.automl_config.feature_scaling == "auto":
            scaler = StandardScaler()
            X_processed[numerical_cols] = scaler.fit_transform(X_processed[numerical_cols])
            preprocessor['scaler'] = scaler
            preprocessor['numerical_columns'] = numerical_cols.tolist()
        
        return X_processed, preprocessor
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, 
                        task_type: str) -> Tuple[pd.DataFrame, Any]:
        """Select features"""
        if not self.automl_config.feature_selection:
            return X, None
        
        # Use univariate selection as default
        if task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k='all')
        else:
            selector = SelectKBest(score_func=f_regression, k='all')
        
        X_selected = selector.fit_transform(X, y)
        
        # Get selected features
        selected_features = X.columns[selector.get_support()]
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        # Limit features if specified
        if self.automl_config.max_features and len(selected_features) > self.automl_config.max_features:
            top_features = selected_features[:self.automl_config.max_features]
            X_selected_df = X_selected_df[top_features]
        
        return X_selected_df, selector
    
    def _optimize_algorithms(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           task_type: str) -> List[Dict[str, Any]]:
        """Optimize algorithms"""
        from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
        
        selected_algorithms = self.algorithm_selector.select_algorithms(task_type)
        results = []
        
        for alg_name in selected_algorithms[:5]:  # Limit to top 5 algorithms
            try:
                self._logger.info(f"Optimizing {alg_name}")
                
                # Get algorithm and parameter space
                alg_config = self.algorithm_selector.available_algorithms[alg_name]
                base_model = self.algorithm_selector.get_algorithm_instance(alg_name, task_type)
                param_grid = alg_config['params']
                
                # Hyperparameter optimization
                if self.automl_config.hpo_method == "random":
                    search = RandomizedSearchCV(
                        base_model, 
                        param_grid,
                        n_iter=min(self.automl_config.hpo_iterations, 50),
                        cv=self.automl_config.hpo_cv_folds,
                        scoring='accuracy' if task_type == 'classification' else 'r2',
                        random_state=42,
                        n_jobs=-1
                    )
                else:
                    # Grid search for smaller parameter spaces
                    search = GridSearchCV(
                        base_model,
                        param_grid,
                        cv=self.automl_config.hpo_cv_folds,
                        scoring='accuracy' if task_type == 'classification' else 'r2',
                        n_jobs=-1
                    )
                
                # Fit and evaluate
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
                # Evaluate on validation set
                val_pred = best_model.predict(X_val)
                
                if task_type == 'classification':
                    val_score = accuracy_score(y_val, val_pred)
                else:
                    val_score = r2_score(y_val, val_pred)
                
                results.append({
                    'algorithm': alg_name,
                    'model': best_model,
                    'best_params': search.best_params_,
                    'cv_score': search.best_score_,
                    'val_score': val_score,
                    'search_time': getattr(search, 'refit_time_', 0)
                })
                
                self._logger.info(f"{alg_name} - CV: {search.best_score_:.4f}, Val: {val_score:.4f}")
                
            except Exception as e:
                self._logger.warning(f"Failed to optimize {alg_name}: {e}")
        
        # Sort by validation score
        results.sort(key=lambda x: x['val_score'], reverse=True)
        return results
    
    def _create_ensemble(self, models: List[Dict[str, Any]], X_val: pd.DataFrame,
                        y_val: pd.Series, task_type: str) -> Any:
        """Create ensemble model"""
        if len(models) < 2:
            return models[0]['model'] if models else None
        
        # Take top models for ensemble
        top_models = models[:self.automl_config.ensemble_size]
        
        if 'voting' in self.automl_config.ensemble_methods:
            estimators = [(model['algorithm'], model['model']) for model in top_models]
            
            if task_type == 'classification':
                ensemble = VotingClassifier(estimators, voting='soft')
            else:
                ensemble = VotingRegressor(estimators)
            
            # Fit ensemble
            X_train_full = X_val  # Use validation set as we already have trained models
            ensemble.fit(X_train_full, y_val)
            
            return ensemble
        
        # If no ensemble method specified, return best model
        return models[0]['model']
    
    def _evaluate_pipeline(self, pipeline: Dict[str, Any], X_val: pd.DataFrame,
                          y_val: pd.Series, task_type: str) -> float:
        """Evaluate final pipeline"""
        try:
            predictions = pipeline['model'].predict(X_val)
            
            if task_type == 'classification':
                return accuracy_score(y_val, predictions)
            else:
                return r2_score(y_val, predictions)
                
        except Exception as e:
            self._logger.error(f"Pipeline evaluation failed: {e}")
            return 0.0
    
    def _save_pipeline(self, results: Dict[str, Any]):
        """Save AutoML pipeline"""
        try:
            output_path = Path(self.automl_config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_path / "automl_results.json", 'w') as f:
                # Create JSON-serializable copy
                json_results = {
                    'final_score': results['final_score'],
                    'task_type': results['task_type'],
                    'elapsed_time': results['elapsed_time'],
                    'feature_count': results['feature_count'],
                    'original_feature_count': results['original_feature_count']
                }
                json.dump(json_results, f, indent=2)
            
            # Save pipeline (using pickle for sklearn models)
            import pickle
            with open(output_path / "pipeline.pkl", 'wb') as f:
                pickle.dump(results['pipeline'], f)
            
            self._logger.info(f"AutoML pipeline saved to: {output_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to save pipeline: {e}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the fitted pipeline"""
        if self.best_pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit() first.")
        
        try:
            # Apply preprocessing
            X_processed = X.copy()
            
            # Apply stored transformations
            preprocessor = self.best_pipeline['preprocessor']
            for transformation in preprocessor['transformations']:
                if transformation['type'] == 'fillna':
                    X_processed[transformation['column']].fillna(
                        transformation['value'], inplace=True
                    )
                elif transformation['type'] == 'one_hot':
                    # This would need more sophisticated handling in production
                    pass
            
            # Apply scaling if used
            if 'scaler' in preprocessor:
                numerical_cols = preprocessor['numerical_columns']
                X_processed[numerical_cols] = preprocessor['scaler'].transform(
                    X_processed[numerical_cols]
                )
            
            # Apply feature engineering
            X_engineered = self.best_pipeline['feature_engineer'].transform(X_processed)
            
            # Apply feature selection
            if self.best_pipeline['feature_selector'] is not None:
                X_selected = self.best_pipeline['feature_selector'].transform(X_engineered)
                X_selected = pd.DataFrame(
                    X_selected, 
                    columns=self.best_pipeline['feature_names'],
                    index=X.index
                )
            else:
                X_selected = X_engineered[self.best_pipeline['feature_names']]
            
            # Make predictions
            predictions = self.best_pipeline['model'].predict(X_selected)
            
            return predictions
            
        except Exception as e:
            self._logger.error(f"Prediction failed: {e}")
            return np.array([])