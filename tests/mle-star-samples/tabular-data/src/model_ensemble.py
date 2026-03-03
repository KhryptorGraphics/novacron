"""Ensemble models for tabular data classification and regression."""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    VotingClassifier, VotingRegressor,
    BaggingClassifier, BaggingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

class TabularEnsemble:
    """Advanced ensemble model for tabular data with multiple algorithms."""
    
    def __init__(self, task_type: str = 'classification', random_state: int = 42):
        self.task_type = task_type
        self.random_state = random_state
        
        # Model components
        self.base_models = {}
        self.ensemble_model = None
        self.meta_learner = None
        
        # Training results
        self.training_results = {}
        self.feature_importance = {}
        self.cv_scores = {}
        
        # Initialize base models
        self._initialize_base_models()
    
    def _initialize_base_models(self):
        """Initialize base models for ensemble."""
        
        if self.task_type == 'classification':
            self.base_models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=self.random_state
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1, 
                    random_state=self.random_state, eval_metric='logloss'
                ),
                'lightgbm': lgb.LGBMClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, verbose=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state
                ),
                'logistic_regression': LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                'svm': SVC(
                    probability=True, random_state=self.random_state
                )
            }
        else:  # regression
            self.base_models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, max_depth=10, random_state=self.random_state
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state
                ),
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, verbose=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state
                ),
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0, random_state=self.random_state),
                'lasso': Lasso(alpha=1.0, random_state=self.random_state)
            }
    
    def evaluate_base_models(self, X: pd.DataFrame, y: pd.Series, 
                           cv_folds: int = 5) -> Dict[str, Dict[str, float]]:
        """Evaluate all base models using cross-validation."""
        
        print("Evaluating base models...")
        results = {}
        
        # Setup cross-validation
        if self.task_type == 'classification':
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        for model_name, model in self.base_models.items():
            print(f"Evaluating {model_name}...")
            model_results = {}
            
            try:
                for metric in scoring:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                    model_results[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                
                results[model_name] = model_results
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        self.cv_scores = results
        return results
    
    def create_voting_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Union[VotingClassifier, VotingRegressor]:
        """Create voting ensemble from best performing models."""
        
        # Select top 3 models based on primary metric
        if self.task_type == 'classification':
            primary_metric = 'accuracy'
            ensemble_class = VotingClassifier
        else:
            primary_metric = 'r2'
            ensemble_class = VotingRegressor
        
        # Rank models by performance
        model_scores = {}
        for model_name, results in self.cv_scores.items():
            if primary_metric in results:
                model_scores[model_name] = results[primary_metric]['mean']
        
        # Select top models
        top_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"Selected top models for ensemble: {[name for name, _ in top_models]}")
        
        # Create ensemble
        estimators = [(name, self.base_models[name]) for name, _ in top_models]
        
        if self.task_type == 'classification':
            ensemble = VotingClassifier(
                estimators=estimators, 
                voting='soft'  # Use probability voting
            )
        else:
            ensemble = VotingRegressor(estimators=estimators)
        
        return ensemble
    
    def create_stacking_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Union[Any, Any]:
        """Create stacking ensemble with meta-learner."""
        
        from sklearn.ensemble import StackingClassifier, StackingRegressor
        
        # Select base models (exclude simple linear models for stacking)
        base_estimators = [
            ('rf', self.base_models['random_forest']),
            ('xgb', self.base_models['xgboost']),
            ('lgb', self.base_models['lightgbm'])
        ]
        
        # Define meta-learner
        if self.task_type == 'classification':
            meta_learner = LogisticRegression(random_state=self.random_state)
            ensemble = StackingClassifier(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5,
                stack_method='predict_proba'
            )
        else:
            meta_learner = Ridge(alpha=1.0, random_state=self.random_state)
            ensemble = StackingRegressor(
                estimators=base_estimators,
                final_estimator=meta_learner,
                cv=5
            )
        
        return ensemble
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                      ensemble_type: str = 'voting') -> Any:
        """Train ensemble model."""
        
        print(f"Training {ensemble_type} ensemble...")
        
        # First evaluate base models
        self.evaluate_base_models(X, y)
        
        # Create ensemble based on type
        if ensemble_type == 'voting':
            self.ensemble_model = self.create_voting_ensemble(X, y)
        elif ensemble_type == 'stacking':
            self.ensemble_model = self.create_stacking_ensemble(X, y)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        # Train ensemble
        self.ensemble_model.fit(X, y)
        
        # Extract feature importance where available
        self._extract_feature_importance(X.columns)
        
        print(f"Ensemble training completed.")
        return self.ensemble_model
    
    def _extract_feature_importance(self, feature_names: List[str]):
        """Extract feature importance from base models."""
        
        for model_name, model in self.base_models.items():
            if hasattr(model, 'feature_importances_'):
                # Train individual model to get feature importance
                self.feature_importance[model_name] = dict(
                    zip(feature_names, model.feature_importances_)
                )
        
        # Ensemble feature importance (average of base models)
        if self.feature_importance:
            ensemble_importance = {}
            for feature in feature_names:
                importance_values = [
                    importance.get(feature, 0) 
                    for importance in self.feature_importance.values()
                ]
                ensemble_importance[feature] = np.mean(importance_values)
            
            self.feature_importance['ensemble'] = ensemble_importance
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble model."""
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained. Call train_ensemble first.")
        
        return self.ensemble_model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification tasks")
        
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained. Call train_ensemble first.")
        
        return self.ensemble_model.predict_proba(X)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate trained ensemble model."""
        
        predictions = self.predict(X_test)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, predictions, average='weighted', zero_division=0)
            }
            
            # Add ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    probabilities = self.predict_proba(X_test)[:, 1]
                    metrics['roc_auc'] = roc_auc_score(y_test, probabilities)
                except:
                    pass
            
        else:  # regression
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2_score': r2_score(y_test, predictions)
            }
        
        return metrics
    
    def plot_model_comparison(self, save_path: Optional[str] = None):
        """Plot comparison of base model performances."""
        
        if not self.cv_scores:
            print("No cross-validation results available. Run evaluate_base_models first.")
            return
        
        # Prepare data for plotting
        if self.task_type == 'classification':
            metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
            metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        else:
            metrics = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
            metric_labels = ['R² Score', 'MSE (neg)', 'MAE (neg)']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Base Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, (metric, label) in enumerate(zip(metrics[:4], metric_labels[:4])):
            row = idx // 2
            col = idx % 2
            
            model_names = []
            mean_scores = []
            std_scores = []
            
            for model_name, results in self.cv_scores.items():
                if metric in results:
                    model_names.append(model_name.replace('_', ' ').title())
                    mean_scores.append(results[metric]['mean'])
                    std_scores.append(results[metric]['std'])
            
            # Create bar plot with error bars
            bars = axes[row, col].bar(model_names, mean_scores, yerr=std_scores, 
                                     capsize=5, alpha=0.8, color='skyblue')
            axes[row, col].set_title(f'{label} Comparison')
            axes[row, col].set_ylabel(label)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, mean_score in zip(bars, mean_scores):
                axes[row, col].text(bar.get_x() + bar.get_width()/2., 
                                   bar.get_height() + bar.get_height() * 0.01,
                                   f'{mean_score:.3f}', 
                                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, top_n: int = 15, save_path: Optional[str] = None):
        """Plot feature importance from ensemble."""
        
        if 'ensemble' not in self.feature_importance:
            print("No feature importance available. Train ensemble first.")
            return
        
        # Get top features
        importance_dict = self.feature_importance['ensemble']
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, importances = zip(*top_features)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), importances, color='lightgreen', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Features by Importance (Ensemble Average)')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_model(self, file_path: str):
        """Save trained ensemble model."""
        if self.ensemble_model is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'base_models': self.base_models,
            'task_type': self.task_type,
            'feature_importance': self.feature_importance,
            'cv_scores': self.cv_scores,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, file_path)
        print(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str):
        """Load trained ensemble model."""
        model_data = joblib.load(file_path)
        
        self.ensemble_model = model_data['ensemble_model']
        self.base_models = model_data['base_models']
        self.task_type = model_data['task_type']
        self.feature_importance = model_data['feature_importance']
        self.cv_scores = model_data['cv_scores']
        self.random_state = model_data['random_state']
        
        print(f"Model loaded from {file_path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        
        info = {
            'task_type': self.task_type,
            'base_models': list(self.base_models.keys()),
            'ensemble_type': type(self.ensemble_model).__name__ if self.ensemble_model else None,
            'is_trained': self.ensemble_model is not None,
            'feature_importance_available': bool(self.feature_importance),
            'cv_results_available': bool(self.cv_scores)
        }
        
        if self.cv_scores:
            # Add best model information
            if self.task_type == 'classification':
                primary_metric = 'accuracy'
            else:
                primary_metric = 'r2'
            
            best_model = None
            best_score = -np.inf
            
            for model_name, results in self.cv_scores.items():
                if primary_metric in results:
                    score = results[primary_metric]['mean']
                    if score > best_score:
                        best_score = score
                        best_model = model_name
            
            info['best_base_model'] = best_model
            info['best_base_model_score'] = best_score
        
        return info

# MLE-Star Stage 2: Task Definition for Tabular Data
def define_tabular_task(task_type: str = 'classification') -> Dict[str, Any]:
    """Define the specific tabular ML task and model requirements."""
    
    if task_type == 'classification':
        task_definition = {
            'task_type': 'binary/multiclass classification on tabular data',
            'input_specification': {
                'format': 'structured tabular data (CSV/DataFrame)',
                'features': 'mixed numerical and categorical features',
                'preprocessing': 'comprehensive pipeline with encoding and scaling',
                'missing_values': 'handled via imputation strategies'
            },
            'output_specification': {
                'format': 'class predictions and probabilities',
                'classes': 'variable (2+ classes)',
                'output_types': ['predictions', 'probabilities', 'confidence_scores']
            },
            'success_metrics': {
                'primary': 'accuracy',
                'secondary': ['precision', 'recall', 'f1-score', 'roc-auc'],
                'target_accuracy': 0.85
            }
        }
    else:  # regression
        task_definition = {
            'task_type': 'regression on tabular data',
            'input_specification': {
                'format': 'structured tabular data (CSV/DataFrame)',
                'features': 'mixed numerical and categorical features',
                'preprocessing': 'comprehensive pipeline with encoding and scaling',
                'missing_values': 'handled via imputation strategies'
            },
            'output_specification': {
                'format': 'continuous numerical predictions',
                'range': 'problem-dependent',
                'output_types': ['predictions', 'prediction_intervals']
            },
            'success_metrics': {
                'primary': 'r2_score',
                'secondary': ['mse', 'rmse', 'mae'],
                'target_r2': 0.80
            }
        }
    
    # Common elements
    task_definition.update({
        'model_constraints': {
            'interpretability': 'high (ensemble of interpretable models)',
            'training_time': '<30 minutes on standard hardware',
            'inference_time': '<100ms per sample',
            'memory_usage': '<4GB during training'
        },
        'training_strategy': {
            'approach': 'ensemble learning with multiple algorithms',
            'base_models': ['Random Forest', 'XGBoost', 'LightGBM', 'Gradient Boosting'],
            'ensemble_method': 'voting or stacking',
            'cross_validation': '5-fold stratified (classification) or k-fold (regression)',
            'hyperparameter_tuning': 'grid search or random search'
        },
        'data_requirements': {
            'minimum_samples': 100,
            'recommended_samples': 1000,
            'feature_engineering': 'polynomial features and interactions',
            'feature_selection': 'importance-based selection',
            'data_quality': 'automated quality assessment and cleaning'
        }
    })
    
    return task_definition

if __name__ == '__main__':
    # Test ensemble model
    print("Testing Tabular Ensemble Model...")
    
    # Create synthetic data
    from data_preprocessor import TabularDataPreprocessor
    
    preprocessor = TabularDataPreprocessor('approved', 'classification')
    df = preprocessor.create_synthetic_dataset(n_samples=1000, task_type='classification')
    
    # Preprocess data
    X, y = preprocessor.preprocess_data(df, fit=True)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create and train ensemble
    ensemble = TabularEnsemble(task_type='classification')
    
    # Train voting ensemble
    model = ensemble.train_ensemble(X_train, y_train, ensemble_type='voting')
    
    # Evaluate model
    metrics = ensemble.evaluate_model(X_test, y_test)
    print("\nEnsemble Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Get model info
    model_info = ensemble.get_model_info()
    print("\nModel Information:")
    for key, value in model_info.items():
        print(f"{key}: {value}")
    
    # Task definition
    task = define_tabular_task('classification')
    print("\n=== Tabular Task Definition ===")
    for key, value in task.items():
        print(f"{key}: {value}")
