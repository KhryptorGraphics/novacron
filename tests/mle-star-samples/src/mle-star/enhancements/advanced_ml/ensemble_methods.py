"""
Advanced Ensemble Methods
========================

Advanced ensemble learning techniques for MLE-Star:
- Stacking and meta-learning
- Blending and weighted averaging
- Dynamic ensemble selection
- Multi-level ensembles
- Ensemble diversity optimization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from abc import ABC, abstractmethod

try:
    from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
    from sklearn.ensemble import VotingClassifier, VotingRegressor, BaggingClassifier, BaggingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, mean_squared_error, log_loss
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    # Ensemble types
    use_voting: bool = True
    use_stacking: bool = True
    use_blending: bool = True
    use_dynamic_selection: bool = False
    
    # Stacking configuration
    stacking_cv_folds: int = 5
    stacking_meta_learner: str = "logistic"  # logistic, ridge, lasso, rf
    stacking_use_probabilities: bool = True
    
    # Blending configuration
    blending_holdout_ratio: float = 0.2
    blending_optimization_method: str = "nelder_mead"  # nelder_mead, scipy_minimize
    
    # Dynamic selection
    dynamic_selection_method: str = "ola"  # ola, lca, mcb, knora_e
    dynamic_selection_k: int = 5
    
    # Diversity measures
    diversity_measures: List[str] = None  # disagreement, correlation, kappa
    min_diversity_threshold: float = 0.1
    
    # Model selection
    max_models_in_ensemble: int = 10
    selection_metric: str = "accuracy"  # accuracy, f1, rmse, r2
    
    # Optimization
    ensemble_pruning: bool = True
    performance_threshold: float = 0.95  # Keep models with >95% of best performance
    
    # Output
    save_ensemble_models: bool = True
    output_path: str = "./ensemble_models"


class StackingEnsemble(BaseEstimator):
    """Multi-level stacking ensemble"""
    
    def __init__(self, base_models: List[Any], meta_learner: Any, 
                 cv_folds: int = 5, use_probabilities: bool = True):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.cv_folds = cv_folds
        self.use_probabilities = use_probabilities
        self.fitted_base_models = None
        self.fitted_meta_learner = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit stacking ensemble"""
        try:
            # Create cross-validation folds
            if len(np.unique(y)) > 2:  # Multi-class or regression
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            else:
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            # Generate meta-features using cross-validation
            meta_features = np.zeros((X.shape[0], len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                if self.use_probabilities and hasattr(model, 'predict_proba'):
                    # Use probabilities for classification
                    cv_predictions = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
                    if cv_predictions.shape[1] == 2:
                        # Binary classification - use positive class probability
                        meta_features[:, i] = cv_predictions[:, 1]
                    else:
                        # Multi-class - use max probability
                        meta_features[:, i] = np.max(cv_predictions, axis=1)
                else:
                    # Use predictions
                    cv_predictions = cross_val_predict(model, X, y, cv=cv)
                    meta_features[:, i] = cv_predictions
            
            # Fit base models on full training data
            self.fitted_base_models = []
            for model in self.base_models:
                fitted_model = clone(model)
                fitted_model.fit(X, y)
                self.fitted_base_models.append(fitted_model)
            
            # Fit meta-learner on meta-features
            self.fitted_meta_learner = clone(self.meta_learner)
            self.fitted_meta_learner.fit(meta_features, y)
            
            return self
            
        except Exception as e:
            logger.error(f"Stacking ensemble fitting failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with stacking ensemble"""
        if self.fitted_base_models is None or self.fitted_meta_learner is None:
            raise ValueError("Ensemble not fitted")
        
        # Generate meta-features
        meta_features = np.zeros((X.shape[0], len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            if self.use_probabilities and hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)
                if predictions.shape[1] == 2:
                    meta_features[:, i] = predictions[:, 1]
                else:
                    meta_features[:, i] = np.max(predictions, axis=1)
            else:
                meta_features[:, i] = model.predict(X)
        
        # Make final prediction with meta-learner
        return self.fitted_meta_learner.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with stacking ensemble"""
        if not hasattr(self.fitted_meta_learner, 'predict_proba'):
            raise ValueError("Meta-learner does not support probability prediction")
        
        # Generate meta-features
        meta_features = np.zeros((X.shape[0], len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X)
                if predictions.shape[1] == 2:
                    meta_features[:, i] = predictions[:, 1]
                else:
                    meta_features[:, i] = np.max(predictions, axis=1)
            else:
                meta_features[:, i] = model.predict(X)
        
        return self.fitted_meta_learner.predict_proba(meta_features)


class BlendingEnsemble(BaseEstimator):
    """Blending ensemble with optimized weights"""
    
    def __init__(self, base_models: List[Any], holdout_ratio: float = 0.2,
                 optimization_method: str = "nelder_mead"):
        self.base_models = base_models
        self.holdout_ratio = holdout_ratio
        self.optimization_method = optimization_method
        self.fitted_base_models = None
        self.weights = None
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit blending ensemble"""
        try:
            from sklearn.model_selection import train_test_split
            
            # Split data for blending
            X_train, X_blend, y_train, y_blend = train_test_split(
                X, y, test_size=self.holdout_ratio, random_state=42,
                stratify=y if len(np.unique(y)) < 20 else None
            )
            
            # Fit base models on training set
            self.fitted_base_models = []
            blend_predictions = np.zeros((len(X_blend), len(self.base_models)))
            
            for i, model in enumerate(self.base_models):
                fitted_model = clone(model)
                fitted_model.fit(X_train, y_train)
                self.fitted_base_models.append(fitted_model)
                
                # Get predictions on blend set
                if hasattr(fitted_model, 'predict_proba') and len(np.unique(y)) < 20:
                    proba = fitted_model.predict_proba(X_blend)
                    if proba.shape[1] == 2:
                        blend_predictions[:, i] = proba[:, 1]
                    else:
                        blend_predictions[:, i] = np.max(proba, axis=1)
                else:
                    blend_predictions[:, i] = fitted_model.predict(X_blend)
            
            # Optimize weights
            self.weights = self._optimize_weights(blend_predictions, y_blend)
            
            return self
            
        except Exception as e:
            logger.error(f"Blending ensemble fitting failed: {e}")
            raise
    
    def _optimize_weights(self, predictions: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Optimize blending weights"""
        try:
            from scipy.optimize import minimize
            
            n_models = predictions.shape[1]
            
            def objective(weights):
                # Ensure weights are non-negative and sum to 1
                weights = np.abs(weights)
                if np.sum(weights) == 0:
                    weights = np.ones(n_models) / n_models
                else:
                    weights = weights / np.sum(weights)
                
                # Calculate weighted prediction
                weighted_pred = np.dot(predictions, weights)
                
                # Calculate loss (use appropriate loss for task type)
                if len(np.unique(y_true)) < 20:  # Classification
                    # Convert to probabilities if needed
                    weighted_pred = np.clip(weighted_pred, 1e-15, 1 - 1e-15)
                    if len(np.unique(y_true)) == 2:
                        # Binary classification - use log loss
                        return log_loss(y_true, weighted_pred)
                    else:
                        # Multi-class - use accuracy (negative for minimization)
                        pred_classes = (weighted_pred > 0.5).astype(int)
                        return -accuracy_score(y_true, pred_classes)
                else:
                    # Regression - use MSE
                    return mean_squared_error(y_true, weighted_pred)
            
            # Initial weights (uniform)
            initial_weights = np.ones(n_models) / n_models
            
            # Optimization constraints
            constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(n_models)]
            
            # Optimize
            if self.optimization_method == "nelder_mead":
                result = minimize(objective, initial_weights, method='Nelder-Mead')
            else:
                result = minimize(objective, initial_weights, method='SLSQP',
                                bounds=bounds, constraints=constraints)
            
            weights = result.x
            weights = np.abs(weights) / np.sum(np.abs(weights))  # Normalize
            
            return weights
            
        except Exception as e:
            logger.warning(f"Weight optimization failed, using uniform weights: {e}")
            return np.ones(predictions.shape[1]) / predictions.shape[1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with blending ensemble"""
        if self.fitted_base_models is None or self.weights is None:
            raise ValueError("Ensemble not fitted")
        
        # Get predictions from all models
        predictions = np.zeros((X.shape[0], len(self.fitted_base_models)))
        
        for i, model in enumerate(self.fitted_base_models):
            if hasattr(model, 'predict_proba') and len(self.fitted_base_models[0].classes_) < 20:
                proba = model.predict_proba(X)
                if proba.shape[1] == 2:
                    predictions[:, i] = proba[:, 1]
                else:
                    predictions[:, i] = np.max(proba, axis=1)
            else:
                predictions[:, i] = model.predict(X)
        
        # Return weighted average
        return np.dot(predictions, self.weights)


class DiversityAnalyzer:
    """Analyze ensemble diversity"""
    
    @staticmethod
    def disagreement_measure(predictions: np.ndarray) -> float:
        """Calculate disagreement diversity measure"""
        n_models = predictions.shape[1]
        n_samples = predictions.shape[0]
        
        total_disagreement = 0
        for i in range(n_samples):
            sample_predictions = predictions[i, :]
            # Count how many pairs disagree
            disagreements = 0
            total_pairs = 0
            
            for j in range(n_models):
                for k in range(j + 1, n_models):
                    total_pairs += 1
                    if sample_predictions[j] != sample_predictions[k]:
                        disagreements += 1
            
            if total_pairs > 0:
                total_disagreement += disagreements / total_pairs
        
        return total_disagreement / n_samples if n_samples > 0 else 0
    
    @staticmethod
    def correlation_diversity(predictions: np.ndarray) -> float:
        """Calculate correlation-based diversity measure"""
        correlations = []
        n_models = predictions.shape[1]
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                corr = np.corrcoef(predictions[:, i], predictions[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        # Return average absolute correlation (lower is more diverse)
        return 1 - np.mean(correlations) if correlations else 0
    
    @staticmethod
    def kappa_diversity(predictions: np.ndarray, y_true: np.ndarray) -> float:
        """Calculate Kappa-based diversity measure"""
        from sklearn.metrics import cohen_kappa_score
        
        kappa_scores = []
        n_models = predictions.shape[1]
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                try:
                    kappa = cohen_kappa_score(predictions[:, i], predictions[:, j])
                    if not np.isnan(kappa):
                        kappa_scores.append(abs(kappa))
                except:
                    continue
        
        # Return diversity (1 - average kappa)
        return 1 - np.mean(kappa_scores) if kappa_scores else 0


class EnsembleMethods(BaseEnhancement):
    """Advanced Ensemble Methods enhancement for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="ensemble_methods",
            version="1.0.0",
            enabled=SKLEARN_AVAILABLE,
            priority=20,
            parameters={
                "use_voting": True,
                "use_stacking": True,
                "use_blending": True,
                "use_dynamic_selection": False,
                "stacking_cv_folds": 5,
                "stacking_meta_learner": "logistic",
                "stacking_use_probabilities": True,
                "blending_holdout_ratio": 0.2,
                "blending_optimization_method": "nelder_mead",
                "dynamic_selection_method": "ola",
                "dynamic_selection_k": 5,
                "diversity_measures": ["disagreement", "correlation", "kappa"],
                "min_diversity_threshold": 0.1,
                "max_models_in_ensemble": 10,
                "selection_metric": "accuracy",
                "ensemble_pruning": True,
                "performance_threshold": 0.95,
                "save_ensemble_models": True,
                "output_path": "./ensemble_models"
            }
        )
    
    def initialize(self) -> bool:
        """Initialize Ensemble Methods"""
        if not SKLEARN_AVAILABLE:
            self._logger.error("Scikit-learn not available for ensemble methods")
            return False
        
        try:
            # Create configuration
            self.ensemble_config = EnsembleConfig(**self.config.parameters)
            
            # Initialize diversity analyzer
            self.diversity_analyzer = DiversityAnalyzer()
            
            # Storage for fitted ensembles
            self.fitted_ensembles = {}
            
            self._logger.info("Advanced Ensemble Methods initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Ensemble Methods: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with ensemble capabilities"""
        enhanced = workflow.copy()
        
        # Add ensemble configuration
        if 'ensemble_learning' not in enhanced:
            enhanced['ensemble_learning'] = {}
        
        enhanced['ensemble_learning'] = {
            'enabled': True,
            'methods': {
                'voting': self.ensemble_config.use_voting,
                'stacking': self.ensemble_config.use_stacking,
                'blending': self.ensemble_config.use_blending,
                'dynamic_selection': self.ensemble_config.use_dynamic_selection
            },
            'stacking': {
                'cv_folds': self.ensemble_config.stacking_cv_folds,
                'meta_learner': self.ensemble_config.stacking_meta_learner,
                'use_probabilities': self.ensemble_config.stacking_use_probabilities
            },
            'blending': {
                'holdout_ratio': self.ensemble_config.blending_holdout_ratio,
                'optimization_method': self.ensemble_config.blending_optimization_method
            },
            'diversity': {
                'measures': self.ensemble_config.diversity_measures,
                'min_threshold': self.ensemble_config.min_diversity_threshold
            },
            'selection': {
                'max_models': self.ensemble_config.max_models_in_ensemble,
                'metric': self.ensemble_config.selection_metric,
                'pruning': self.ensemble_config.ensemble_pruning,
                'performance_threshold': self.ensemble_config.performance_threshold
            }
        }
        
        # Enhance MLE-Star stages with ensemble learning
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 4: Implementation - Add ensemble training
            if '4_implementation' in stages:
                if 'ensemble_training' not in stages['4_implementation']:
                    stages['4_implementation']['ensemble_training'] = [
                        'base_model_training',
                        'diversity_analysis',
                        'ensemble_construction',
                        'meta_learner_training'
                    ]
            
            # Stage 5: Results Evaluation - Add ensemble evaluation
            if '5_results_evaluation' in stages:
                if 'ensemble_evaluation' not in stages['5_results_evaluation']:
                    stages['5_results_evaluation']['ensemble_evaluation'] = [
                        'individual_model_performance',
                        'ensemble_performance',
                        'diversity_metrics',
                        'ensemble_comparison'
                    ]
            
            # Stage 6: Refinement - Add ensemble optimization
            if '6_refinement' in stages:
                if 'ensemble_optimization' not in stages['6_refinement']:
                    stages['6_refinement']['ensemble_optimization'] = [
                        'model_selection_optimization',
                        'weight_optimization',
                        'diversity_enhancement',
                        'ensemble_pruning'
                    ]
        
        # Add ensemble specific evaluation metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'ensemble_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['ensemble_metrics'] = [
                'ensemble_accuracy',
                'individual_model_performance',
                'diversity_measures',
                'ensemble_improvement',
                'model_agreement'
            ]
        
        self._logger.debug("Enhanced workflow with ensemble capabilities")
        return enhanced
    
    def create_ensemble(self, base_models: List[Any], X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray, 
                       task_type: str = "classification") -> Dict[str, Any]:
        """Create and train ensemble models"""
        try:
            self._logger.info("Creating ensemble models")
            
            ensembles = {}
            results = {}
            
            # Evaluate base models individually
            base_model_results = self._evaluate_base_models(
                base_models, X_train, y_train, X_val, y_val, task_type
            )
            results['base_models'] = base_model_results
            
            # Select best models for ensemble
            selected_models = self._select_models_for_ensemble(
                base_models, base_model_results
            )
            
            # Voting ensemble
            if self.ensemble_config.use_voting:
                voting_ensemble = self._create_voting_ensemble(
                    selected_models, task_type
                )
                voting_ensemble.fit(X_train, y_train)
                ensembles['voting'] = voting_ensemble
                
                # Evaluate voting ensemble
                voting_score = self._evaluate_ensemble(
                    voting_ensemble, X_val, y_val, task_type
                )
                results['voting'] = voting_score
            
            # Stacking ensemble
            if self.ensemble_config.use_stacking:
                meta_learner = self._create_meta_learner(task_type)
                stacking_ensemble = StackingEnsemble(
                    selected_models, meta_learner,
                    cv_folds=self.ensemble_config.stacking_cv_folds,
                    use_probabilities=self.ensemble_config.stacking_use_probabilities
                )
                stacking_ensemble.fit(X_train, y_train)
                ensembles['stacking'] = stacking_ensemble
                
                # Evaluate stacking ensemble
                stacking_score = self._evaluate_ensemble(
                    stacking_ensemble, X_val, y_val, task_type
                )
                results['stacking'] = stacking_score
            
            # Blending ensemble
            if self.ensemble_config.use_blending:
                blending_ensemble = BlendingEnsemble(
                    selected_models,
                    holdout_ratio=self.ensemble_config.blending_holdout_ratio,
                    optimization_method=self.ensemble_config.blending_optimization_method
                )
                blending_ensemble.fit(X_train, y_train)
                ensembles['blending'] = blending_ensemble
                
                # Evaluate blending ensemble
                blending_score = self._evaluate_ensemble(
                    blending_ensemble, X_val, y_val, task_type
                )
                results['blending'] = blending_score
            
            # Analyze diversity
            diversity_results = self._analyze_ensemble_diversity(
                selected_models, X_val, y_val
            )
            results['diversity'] = diversity_results
            
            # Store fitted ensembles
            self.fitted_ensembles = ensembles
            
            # Save ensembles if configured
            if self.ensemble_config.save_ensemble_models:
                self._save_ensembles(ensembles, results)
            
            self._logger.info("Ensemble creation completed")
            return {'ensembles': ensembles, 'results': results}
            
        except Exception as e:
            self._logger.error(f"Ensemble creation failed: {e}")
            return {}
    
    def _evaluate_base_models(self, models: List[Any], X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray, task_type: str) -> List[Dict[str, Any]]:
        """Evaluate individual base models"""
        results = []
        
        for i, model in enumerate(models):
            try:
                # Train model
                fitted_model = clone(model)
                fitted_model.fit(X_train, y_train)
                
                # Make predictions
                if task_type == "classification":
                    predictions = fitted_model.predict(X_val)
                    score = accuracy_score(y_val, predictions)
                    metric_name = "accuracy"
                else:
                    predictions = fitted_model.predict(X_val)
                    score = -mean_squared_error(y_val, predictions)  # Negative MSE for maximization
                    metric_name = "neg_mse"
                
                results.append({
                    'model_index': i,
                    'model_name': model.__class__.__name__,
                    metric_name: score,
                    'predictions': predictions
                })
                
            except Exception as e:
                logger.warning(f"Failed to evaluate model {i}: {e}")
                results.append({
                    'model_index': i,
                    'model_name': model.__class__.__name__,
                    'error': str(e)
                })
        
        return results
    
    def _select_models_for_ensemble(self, models: List[Any], 
                                   results: List[Dict[str, Any]]) -> List[Any]:
        """Select best models for ensemble based on performance and diversity"""
        # Filter successful models
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return models  # Return all if none succeeded
        
        # Sort by performance
        metric_key = 'accuracy' if 'accuracy' in successful_results[0] else 'neg_mse'
        successful_results.sort(key=lambda x: x[metric_key], reverse=True)
        
        # Apply performance threshold
        if self.ensemble_config.ensemble_pruning:
            best_score = successful_results[0][metric_key]
            threshold = best_score * self.ensemble_config.performance_threshold
            
            filtered_results = [r for r in successful_results if r[metric_key] >= threshold]
        else:
            filtered_results = successful_results
        
        # Limit number of models
        max_models = min(self.ensemble_config.max_models_in_ensemble, len(filtered_results))
        selected_results = filtered_results[:max_models]
        
        # Return selected models
        selected_indices = [r['model_index'] for r in selected_results]
        return [models[i] for i in selected_indices]
    
    def _create_voting_ensemble(self, models: List[Any], task_type: str):
        """Create voting ensemble"""
        estimators = [(f"model_{i}", model) for i, model in enumerate(models)]
        
        if task_type == "classification":
            return VotingClassifier(estimators, voting='soft')
        else:
            return VotingRegressor(estimators)
    
    def _create_meta_learner(self, task_type: str):
        """Create meta-learner for stacking"""
        if self.ensemble_config.stacking_meta_learner == "logistic" and task_type == "classification":
            return LogisticRegression(random_state=42)
        elif self.ensemble_config.stacking_meta_learner == "ridge":
            return Ridge(random_state=42)
        elif self.ensemble_config.stacking_meta_learner == "lasso":
            return Lasso(random_state=42)
        elif self.ensemble_config.stacking_meta_learner == "rf":
            if task_type == "classification":
                return RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                return RandomForestRegressor(n_estimators=50, random_state=42)
        else:
            # Default
            if task_type == "classification":
                return LogisticRegression(random_state=42)
            else:
                return Ridge(random_state=42)
    
    def _evaluate_ensemble(self, ensemble, X_val: np.ndarray, y_val: np.ndarray, 
                         task_type: str) -> Dict[str, float]:
        """Evaluate ensemble performance"""
        try:
            predictions = ensemble.predict(X_val)
            
            if task_type == "classification":
                accuracy = accuracy_score(y_val, predictions)
                
                result = {'accuracy': accuracy}
                
                # Add probability-based metrics if available
                if hasattr(ensemble, 'predict_proba'):
                    try:
                        probabilities = ensemble.predict_proba(X_val)
                        if probabilities.shape[1] == 2:
                            result['log_loss'] = log_loss(y_val, probabilities)
                    except:
                        pass
                
                return result
            else:
                mse = mean_squared_error(y_val, predictions)
                return {'mse': mse, 'neg_mse': -mse}
                
        except Exception as e:
            logger.error(f"Ensemble evaluation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_ensemble_diversity(self, models: List[Any], X: np.ndarray, 
                                  y: np.ndarray) -> Dict[str, float]:
        """Analyze diversity of ensemble models"""
        try:
            # Get predictions from all models
            predictions = np.zeros((len(X), len(models)))
            
            for i, model in enumerate(models):
                try:
                    if hasattr(model, 'predict'):
                        predictions[:, i] = model.predict(X)
                except:
                    predictions[:, i] = np.zeros(len(X))  # Fallback
            
            diversity_metrics = {}
            
            # Calculate diversity measures
            if 'disagreement' in self.ensemble_config.diversity_measures:
                diversity_metrics['disagreement'] = self.diversity_analyzer.disagreement_measure(predictions)
            
            if 'correlation' in self.ensemble_config.diversity_measures:
                diversity_metrics['correlation'] = self.diversity_analyzer.correlation_diversity(predictions)
            
            if 'kappa' in self.ensemble_config.diversity_measures:
                diversity_metrics['kappa'] = self.diversity_analyzer.kappa_diversity(predictions, y)
            
            return diversity_metrics
            
        except Exception as e:
            logger.error(f"Diversity analysis failed: {e}")
            return {}
    
    def _save_ensembles(self, ensembles: Dict[str, Any], results: Dict[str, Any]):
        """Save ensemble models and results"""
        try:
            output_path = Path(self.ensemble_config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_path / "ensemble_results.json", 'w') as f:
                # Create JSON-serializable copy
                json_results = {}
                for key, value in results.items():
                    if key != 'base_models':
                        json_results[key] = value
                    else:
                        # Handle base models separately
                        json_results[key] = [
                            {k: v for k, v in model.items() if k != 'predictions'}
                            for model in value
                        ]
                
                json.dump(json_results, f, indent=2)
            
            # Save ensemble models using pickle
            import pickle
            for name, ensemble in ensembles.items():
                with open(output_path / f"{name}_ensemble.pkl", 'wb') as f:
                    pickle.dump(ensemble, f)
            
            self._logger.info(f"Ensembles saved to: {output_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to save ensembles: {e}")
    
    def get_best_ensemble(self, task_type: str = "classification") -> Tuple[Any, Dict[str, Any]]:
        """Get the best performing ensemble"""
        if not self.fitted_ensembles:
            raise ValueError("No ensembles fitted")
        
        best_ensemble = None
        best_score = float('-inf')
        best_results = None
        
        # Find best ensemble based on performance
        for name, ensemble in self.fitted_ensembles.items():
            # This would require storing evaluation results
            # For now, return the first available ensemble
            if best_ensemble is None:
                best_ensemble = ensemble
                best_results = {'ensemble_type': name}
        
        return best_ensemble, best_results