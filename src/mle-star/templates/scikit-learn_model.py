#!/usr/bin/env python3
"""
Scikit-Learn Model Implementation for {{experimentName}}
MLE-Star Framework - Classical Machine Learning Models
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import yaml
import logging
from datetime import datetime

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Classification models
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, VotingClassifier, BaggingClassifier
)
from sklearn.linear_model import (
    LogisticRegression, RidgeClassifier, SGDClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

# Regression models
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    HuberRegressor, RANSACRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# Clustering models
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering
)

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

# Model selection and validation
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.inspection import permutation_importance

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory for creating scikit-learn models"""
    
    # Model catalogs
    CLASSIFICATION_MODELS = {
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'extra_trees': ExtraTreesClassifier,
        'logistic_regression': LogisticRegression,
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'naive_bayes': GaussianNB,
        'decision_tree': DecisionTreeClassifier,
        'ridge': RidgeClassifier,
        'sgd': SGDClassifier,
        'lda': LinearDiscriminantAnalysis
    }
    
    REGRESSION_MODELS = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'extra_trees': ExtraTreesRegressor,
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'svm': SVR,
        'knn': KNeighborsRegressor,
        'decision_tree': DecisionTreeRegressor,
        'huber': HuberRegressor,
        'ransac': RANSACRegressor
    }
    
    CLUSTERING_MODELS = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'agglomerative': AgglomerativeClustering,
        'spectral': SpectralClustering
    }
    
    @classmethod
    def create_model(cls, model_name, task_type='classification', **kwargs):
        """
        Create a scikit-learn model
        
        Args:
            model_name: Name of the model
            task_type: Type of task ('classification', 'regression', 'clustering')
            **kwargs: Model parameters
        
        Returns:
            Scikit-learn model instance
        """
        if task_type == 'classification':
            model_dict = cls.CLASSIFICATION_MODELS
        elif task_type == 'regression':
            model_dict = cls.REGRESSION_MODELS
        elif task_type == 'clustering':
            model_dict = cls.CLUSTERING_MODELS
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        if model_name not in model_dict:
            available_models = list(model_dict.keys())
            raise ValueError(f"Unknown model: {model_name}. Available models: {available_models}")
        
        model_class = model_dict[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def get_default_params(cls, model_name, task_type='classification'):
        """Get default parameters for a model"""
        model = cls.create_model(model_name, task_type)
        return model.get_params()
    
    @classmethod
    def get_param_grid(cls, model_name, task_type='classification'):
        """Get hyperparameter grid for model tuning"""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            },
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        }
        
        return param_grids.get(model_name, {})

class MLPipeline:
    """
    Complete ML pipeline with preprocessing, model training, and evaluation
    """
    
    def __init__(self, config):
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        
        self.pipeline = None
        self.best_model = None
        self.feature_names = None
        self.target_name = None
        self.task_type = self.model_config.get('task_type', 'classification')
        
        # Results storage
        self.results = {
            'training_scores': {},
            'validation_scores': {},
            'test_scores': {},
            'feature_importance': None,
            'cross_validation_scores': None
        }
    
    def create_preprocessing_pipeline(self, X, categorical_features=None, numerical_features=None):
        """
        Create preprocessing pipeline
        
        Args:
            X: Feature data
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
        
        Returns:
            Preprocessing pipeline
        """
        if categorical_features is None:
            if hasattr(X, 'dtypes'):
                categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            else:
                categorical_features = []
        
        if numerical_features is None:
            if hasattr(X, 'dtypes'):
                numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
            else:
                numerical_features = list(range(X.shape[1]))
        
        # Preprocessing steps
        preprocessing_config = self.config.get('preprocessing', {})
        
        # Numerical preprocessing
        scaling_method = preprocessing_config.get('scaling', 'standard')
        if scaling_method == 'standard':
            numerical_transformer = StandardScaler()
        elif scaling_method == 'minmax':
            numerical_transformer = MinMaxScaler()
        elif scaling_method == 'robust':
            numerical_transformer = RobustScaler()
        else:
            numerical_transformer = StandardScaler()
        
        # Categorical preprocessing
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Combine preprocessing steps
        if len(categorical_features) > 0 and len(numerical_features) > 0:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
        elif len(numerical_features) > 0:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features)
                ]
            )
        elif len(categorical_features) > 0:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
        else:
            # No preprocessing needed
            preprocessor = 'passthrough'
        
        return preprocessor
    
    def create_model(self, model_name=None, **model_params):
        """Create a model with given parameters"""
        model_name = model_name or self.model_config.get('algorithm', 'random_forest')
        
        # Get random state for reproducibility
        random_state = self.config.get('data', {}).get('random_seed', 42)
        
        # Add random_state to model parameters if the model supports it
        if 'random_state' not in model_params:
            model_params['random_state'] = random_state
        
        return ModelFactory.create_model(model_name, self.task_type, **model_params)
    
    def create_pipeline(self, X, y=None, model_name=None, **model_params):
        """
        Create complete ML pipeline
        
        Args:
            X: Feature data
            y: Target data (for determining preprocessing)
            model_name: Name of the model to use
            **model_params: Model parameters
        
        Returns:
            Complete ML pipeline
        """
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(X)
        
        # Create model
        model = self.create_model(model_name, **model_params)
        
        # Create complete pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier' if self.task_type == 'classification' else 'regressor', model)
        ])
        
        return pipeline
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
        
        Returns:
            Trained pipeline
        """
        logger.info("Starting model training...")
        
        # Store feature names
        if hasattr(X_train, 'columns'):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Create pipeline if not exists
        if self.pipeline is None:
            self.pipeline = self.create_pipeline(X_train, y_train)
        
        # Train the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Calculate training scores
        train_pred = self.pipeline.predict(X_train)
        self.results['training_scores'] = self._calculate_scores(y_train, train_pred)
        
        # Calculate validation scores if validation data provided
        if X_val is not None and y_val is not None:
            val_pred = self.pipeline.predict(X_val)
            self.results['validation_scores'] = self._calculate_scores(y_val, val_pred)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.pipeline, X_train, y_train,
            cv=self.training_config.get('cv_folds', 5),
            scoring=self._get_primary_metric()
        )
        self.results['cross_validation_scores'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        # Calculate feature importance if possible
        self._calculate_feature_importance(X_train)
        
        logger.info(f"Training completed. Training {self._get_primary_metric()}: "
                   f"{self.results['training_scores'][self._get_primary_metric()]:.4f}")
        
        return self.pipeline
    
    def hyperparameter_tuning(self, X_train, y_train, model_name=None, param_grid=None, 
                            search_type='grid', n_iter=50):
        """
        Perform hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model
            param_grid: Parameter grid for tuning
            search_type: Type of search ('grid' or 'random')
            n_iter: Number of iterations for random search
        
        Returns:
            Best model pipeline
        """
        logger.info(f"Starting hyperparameter tuning using {search_type} search...")
        
        # Create base pipeline
        base_pipeline = self.create_pipeline(X_train, y_train, model_name)
        
        # Get parameter grid
        if param_grid is None:
            model_name = model_name or self.model_config.get('algorithm', 'random_forest')
            base_param_grid = ModelFactory.get_param_grid(model_name, self.task_type)
            
            # Prefix parameters for pipeline
            model_key = 'classifier' if self.task_type == 'classification' else 'regressor'
            param_grid = {f'{model_key}__{k}': v for k, v in base_param_grid.items()}
        
        # Setup search
        cv_folds = self.training_config.get('cv_folds', 5)
        scoring = self._get_primary_metric()
        
        if search_type == 'grid':
            search = GridSearchCV(
                base_pipeline, param_grid, cv=cv_folds, scoring=scoring,
                n_jobs=-1, verbose=1
            )
        elif search_type == 'random':
            search = RandomizedSearchCV(
                base_pipeline, param_grid, n_iter=n_iter, cv=cv_folds,
                scoring=scoring, n_jobs=-1, verbose=1,
                random_state=self.config.get('data', {}).get('random_seed', 42)
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
        
        # Perform search
        search.fit(X_train, y_train)
        
        # Store best model
        self.pipeline = search.best_estimator_
        self.best_model = search
        
        logger.info(f"Hyperparameter tuning completed. Best {scoring}: {search.best_score_:.4f}")
        logger.info(f"Best parameters: {search.best_params_}")
        
        return self.pipeline
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Test scores dictionary
        """
        if self.pipeline is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Make predictions
        y_pred = self.pipeline.predict(X_test)
        
        # Calculate scores
        test_scores = self._calculate_scores(y_test, y_pred)
        self.results['test_scores'] = test_scores
        
        # Calculate probabilities for classification
        if self.task_type == 'classification' and hasattr(self.pipeline, 'predict_proba'):
            y_proba = self.pipeline.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # Binary classification
                test_scores['auc'] = roc_auc_score(y_test, y_proba[:, 1])
        
        logger.info(f"Test evaluation completed. Test {self._get_primary_metric()}: "
                   f"{test_scores[self._get_primary_metric()]:.4f}")
        
        return test_scores
    
    def _calculate_scores(self, y_true, y_pred):
        """Calculate appropriate scores based on task type"""
        scores = {}
        
        if self.task_type == 'classification':
            scores['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Multi-class vs binary classification
            average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
            
            scores['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
            scores['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
            scores['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
            
        elif self.task_type == 'regression':
            scores['mse'] = mean_squared_error(y_true, y_pred)
            scores['rmse'] = np.sqrt(scores['mse'])
            scores['mae'] = mean_absolute_error(y_true, y_pred)
            scores['r2'] = r2_score(y_true, y_pred)
        
        return scores
    
    def _get_primary_metric(self):
        """Get primary metric for the task"""
        if self.task_type == 'classification':
            return 'accuracy'
        elif self.task_type == 'regression':
            return 'r2'
        else:
            return 'accuracy'
    
    def _calculate_feature_importance(self, X_train):
        """Calculate feature importance if model supports it"""
        try:
            model_step_name = 'classifier' if self.task_type == 'classification' else 'regressor'
            model = self.pipeline.named_steps[model_step_name]
            
            if hasattr(model, 'feature_importances_'):
                # Get feature names after preprocessing
                preprocessor = self.pipeline.named_steps['preprocessor']
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    feature_names = self.feature_names
                
                importance_scores = model.feature_importances_
                
                # Create feature importance dictionary
                self.results['feature_importance'] = dict(zip(feature_names, importance_scores))
                
                # Sort by importance
                sorted_importance = sorted(
                    self.results['feature_importance'].items(),
                    key=lambda x: x[1], reverse=True
                )
                
                logger.info("Top 10 most important features:")
                for feature, importance in sorted_importance[:10]:
                    logger.info(f"  {feature}: {importance:.4f}")
            
            else:
                logger.info("Model doesn't support feature importance")
                
        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
    
    def get_classification_report(self, y_true, y_pred):
        """Get detailed classification report"""
        if self.task_type != 'classification':
            raise ValueError("Classification report only available for classification tasks")
        
        return classification_report(y_true, y_pred, output_dict=True)
    
    def get_confusion_matrix(self, y_true, y_pred):
        """Get confusion matrix"""
        if self.task_type != 'classification':
            raise ValueError("Confusion matrix only available for classification tasks")
        
        return confusion_matrix(y_true, y_pred)
    
    def save_model(self, filepath):
        """Save the trained pipeline"""
        if self.pipeline is None:
            raise ValueError("No trained model to save")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline and results
        save_data = {
            'pipeline': self.pipeline,
            'config': self.config,
            'results': self.results,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'task_type': self.task_type
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a saved pipeline"""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.pipeline = save_data['pipeline']
        self.config = save_data['config']
        self.results = save_data['results']
        self.feature_names = save_data['feature_names']
        self.target_name = save_data['target_name']
        self.task_type = save_data['task_type']
        
        logger.info(f"Model loaded from {filepath}")
        return self.pipeline
    
    def create_ensemble(self, models_configs, X_train, y_train, method='voting'):
        """
        Create ensemble of models
        
        Args:
            models_configs: List of model configurations
            X_train: Training features
            y_train: Training labels
            method: Ensemble method ('voting', 'bagging')
        
        Returns:
            Ensemble pipeline
        """
        # Create individual models
        estimators = []
        for i, config in enumerate(models_configs):
            model_name = config.get('algorithm', 'random_forest')
            model_params = config.get('params', {})
            model = self.create_model(model_name, **model_params)
            estimators.append((f'model_{i}', model))
        
        # Create ensemble
        if method == 'voting':
            if self.task_type == 'classification':
                ensemble = VotingClassifier(
                    estimators=estimators,
                    voting=models_configs[0].get('voting', 'hard')
                )
            else:
                # For regression, use simple averaging
                from sklearn.ensemble import VotingRegressor
                ensemble = VotingRegressor(estimators=estimators)
                
        elif method == 'bagging':
            # Use first model as base estimator
            base_model = self.create_model(models_configs[0].get('algorithm', 'random_forest'))
            ensemble = BaggingClassifier(
                base_estimator=base_model,
                n_estimators=len(models_configs)
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # Create preprocessing pipeline
        preprocessor = self.create_preprocessing_pipeline(X_train)
        
        # Create ensemble pipeline
        ensemble_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ensemble', ensemble)
        ])
        
        self.pipeline = ensemble_pipeline
        return ensemble_pipeline

if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'algorithm': 'random_forest',
            'task_type': 'classification'
        },
        'training': {
            'cv_folds': 5
        },
        'data': {
            'random_seed': 42
        },
        'preprocessing': {
            'scaling': 'standard'
        }
    }
    
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train pipeline
    ml_pipeline = MLPipeline(config)
    pipeline = ml_pipeline.train(X_train, y_train)
    
    # Evaluate
    test_scores = ml_pipeline.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_scores['accuracy']:.4f}")
    
    # Save model
    ml_pipeline.save_model('./outputs/models/sklearn_model.pkl')
    print("Model saved successfully")