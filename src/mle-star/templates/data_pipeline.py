#!/usr/bin/env python3
"""
Data Pipeline Module for {{experimentName}}
MLE-Star Framework - Data Processing and Feature Engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import yaml
import logging

logger = logging.getLogger(__name__)

class DataPipeline:
    """
    Comprehensive data processing pipeline for ML projects
    
    Features:
    - Data loading from multiple formats
    - Missing value imputation
    - Feature scaling and normalization
    - Categorical encoding
    - Feature selection
    - Data splitting and validation
    """
    
    def __init__(self, config):
        self.config = config
        self.data_config = config.get('data', {})
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.label_encoders = {}
        self.feature_names = []
        
        # Setup paths
        self.raw_data_path = Path(self.data_config.get('raw_data_path', './data/raw/'))
        self.processed_data_path = Path(self.data_config.get('processed_data_path', './data/processed/'))
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, file_path=None, file_format='csv'):
        """
        Load data from various file formats
        
        Args:
            file_path: Path to data file
            file_format: Format of data file ('csv', 'json', 'parquet', 'excel')
        
        Returns:
            pandas.DataFrame: Loaded data
        """
        if file_path is None:
            # Generate synthetic data for demonstration
            return self._generate_synthetic_data()
        
        file_path = Path(file_path)
        
        try:
            if file_format == 'csv':
                df = pd.read_csv(file_path)
            elif file_format == 'json':
                df = pd.read_json(file_path)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_format == 'excel':
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            logger.info(f"Data loaded successfully: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for demonstration purposes"""
        np.random.seed(self.data_config.get('random_seed', 42))
        
        n_samples = 5000
        n_features = 10
        
        # Generate features with different distributions
        X = np.random.randn(n_samples, n_features)
        
        # Add some correlation and noise
        X[:, 1] = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.3
        X[:, 2] = X[:, 0] * 0.3 + X[:, 1] * 0.4 + np.random.randn(n_samples) * 0.2
        
        # Create target with complex relationship
        y = ((X[:, 0] * 0.5 + X[:, 1] * 0.3 + X[:, 2] * 0.2 + 
              np.random.randn(n_samples) * 0.1) > 0).astype(int)
        
        # Add some categorical features
        categorical_1 = np.random.choice(['A', 'B', 'C'], n_samples)
        categorical_2 = np.random.choice(['High', 'Medium', 'Low'], n_samples)
        
        # Create DataFrame
        feature_names = [f'numerical_feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['categorical_feature_1'] = categorical_1
        df['categorical_feature_2'] = categorical_2
        df['target'] = y
        
        # Add some missing values
        missing_mask = np.random.random(df.shape) < 0.05  # 5% missing
        df = df.mask(missing_mask)
        
        logger.info(f"Synthetic data generated: {df.shape}")
        return df
    
    def analyze_data(self, df):
        """
        Perform exploratory data analysis
        
        Args:
            df: Input DataFrame
        
        Returns:
            dict: Data analysis summary
        """
        analysis = {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numerical_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'unique_values': {col: df[col].nunique() for col in df.columns},
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Statistical summary for numerical features
        if analysis['numerical_features']:
            analysis['numerical_summary'] = df[analysis['numerical_features']].describe().to_dict()
        
        # Value counts for categorical features (top 5)
        analysis['categorical_summary'] = {}
        for col in analysis['categorical_features']:
            analysis['categorical_summary'][col] = df[col].value_counts().head().to_dict()
        
        logger.info(f"Data analysis completed: {analysis['shape']} shape, "
                   f"{len(analysis['numerical_features'])} numerical, "
                   f"{len(analysis['categorical_features'])} categorical features")
        
        return analysis
    
    def handle_missing_values(self, df, strategy='auto'):
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'mode', 'knn', 'auto')
        
        Returns:
            pandas.DataFrame: DataFrame with imputed values
        """
        df_imputed = df.copy()
        
        # Separate numerical and categorical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if strategy == 'auto':
            # Use different strategies for different column types
            num_strategy = 'median'
            cat_strategy = 'most_frequent'
        elif strategy == 'knn':
            num_strategy = 'knn'
            cat_strategy = 'most_frequent'
        else:
            num_strategy = strategy
            cat_strategy = 'most_frequent' if strategy in ['mean', 'median'] else strategy
        
        # Handle numerical features
        if len(numerical_cols) > 0 and df[numerical_cols].isnull().any().any():
            if num_strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=num_strategy)
            
            df_imputed[numerical_cols] = imputer.fit_transform(df[numerical_cols])
            self.numerical_imputer = imputer
            
            logger.info(f"Numerical missing values imputed using {num_strategy} strategy")
        
        # Handle categorical features
        if len(categorical_cols) > 0 and df[categorical_cols].isnull().any().any():
            cat_imputer = SimpleImputer(strategy=cat_strategy)
            df_imputed[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.categorical_imputer = cat_imputer
            
            logger.info(f"Categorical missing values imputed using {cat_strategy} strategy")
        
        return df_imputed
    
    def encode_categorical_features(self, df, encoding_strategy='auto'):
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            encoding_strategy: Encoding strategy ('onehot', 'label', 'auto')
        
        Returns:
            pandas.DataFrame: DataFrame with encoded features
        """
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) == 0:
            return df_encoded
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            
            if encoding_strategy == 'auto':
                # Use one-hot for low cardinality, label encoding for high cardinality
                use_onehot = n_unique <= 10
            elif encoding_strategy == 'onehot':
                use_onehot = True
            else:
                use_onehot = False
            
            if use_onehot and n_unique <= 20:  # Prevent too many dummy columns
                # One-hot encoding
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = df_encoded.drop(col, axis=1)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                logger.info(f"One-hot encoded feature: {col} ({n_unique} categories)")
            else:
                # Label encoding
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Label encoded feature: {col} ({n_unique} categories)")
        
        return df_encoded
    
    def scale_features(self, X_train, X_val=None, X_test=None, method='standard'):
        """
        Scale numerical features
        
        Args:
            X_train: Training features
            X_val: Validation features (optional)
            X_test: Test features (optional)
            method: Scaling method ('standard', 'minmax', 'robust')
        
        Returns:
            Scaled feature arrays
        """
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit on training data
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler = scaler
        
        results = [X_train_scaled]
        
        # Transform validation and test data
        if X_val is not None:
            X_val_scaled = scaler.transform(X_val)
            results.append(X_val_scaled)
        
        if X_test is not None:
            X_test_scaled = scaler.transform(X_test)
            results.append(X_test_scaled)
        
        logger.info(f"Features scaled using {method} scaling")
        
        return results if len(results) > 1 else results[0]
    
    def select_features(self, X, y, method='univariate', k=10):
        """
        Perform feature selection
        
        Args:
            X: Feature matrix
            y: Target vector
            method: Feature selection method ('univariate', 'correlation')
            k: Number of features to select
        
        Returns:
            Selected features and selector object
        """
        if method == 'univariate':
            selector = SelectKBest(score_func=f_classif, k=k)
            X_selected = selector.fit_transform(X, y)
            self.feature_selector = selector
            
            # Get selected feature names if available
            if hasattr(selector, 'get_feature_names_out'):
                selected_features = selector.get_feature_names_out()
            else:
                selected_features = [f"feature_{i}" for i in range(X_selected.shape[1])]
            
            logger.info(f"Selected {k} features using univariate selection")
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        return X_selected, selected_features
    
    def split_data(self, X, y, test_size=None, val_size=None, random_state=None):
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Target
            test_size: Test set proportion
            val_size: Validation set proportion
            random_state: Random seed
        
        Returns:
            Split data arrays
        """
        # Get split ratios from config if not provided
        test_size = test_size or self.data_config.get('test_split', 0.1)
        val_size = val_size or self.data_config.get('validation_split', 0.2)
        random_state = random_state or self.data_config.get('random_seed', 42)
        stratify = self.data_config.get('stratify', True)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if stratify else None
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=random_state,
            stratify=y_temp if stratify else None
        )
        
        logger.info(f"Data split completed - Train: {len(X_train)}, "
                   f"Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def prepare_data(self, target_column='target', file_path=None):
        """
        Complete data preparation pipeline
        
        Args:
            target_column: Name of target column
            file_path: Path to data file (optional)
        
        Returns:
            Prepared data splits
        """
        logger.info("Starting data preparation pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        
        # Analyze data
        analysis = self.analyze_data(df)
        
        # Save analysis report
        analysis_path = self.processed_data_path / 'data_analysis.yaml'
        with open(analysis_path, 'w') as f:
            yaml.dump(analysis, f, default_flow_style=False)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # Separate features and target
        if target_column not in df_clean.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        y = df_clean[target_column].values
        X_df = df_clean.drop(target_column, axis=1)
        
        # Encode categorical features
        X_encoded = self.encode_categorical_features(X_df)
        X = X_encoded.values
        self.feature_names = X_encoded.columns.tolist()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)
        
        # Scale features
        scaling_method = self.config.get('preprocessing', {}).get('scaling', 'standard')
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test, method=scaling_method
        )
        
        # Feature selection (optional)
        if self.config.get('preprocessing', {}).get('feature_selection', False):
            k_features = self.config['preprocessing'].get('n_features', 10)
            X_train_selected, selected_features = self.select_features(
                X_train_scaled, y_train, k=k_features
            )
            X_val_selected = self.feature_selector.transform(X_val_scaled)
            X_test_selected = self.feature_selector.transform(X_test_scaled)
            
            X_train_scaled = X_train_selected
            X_val_scaled = X_val_selected
            X_test_scaled = X_test_selected
            self.feature_names = selected_features
        
        # Save processed data and preprocessing objects
        self._save_preprocessing_objects()
        
        logger.info("Data preparation pipeline completed successfully")
        
        # Return data in different formats based on configuration
        return self._format_data_output(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test
        )
    
    def _save_preprocessing_objects(self):
        """Save preprocessing objects for later use"""
        objects_to_save = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders
        }
        
        if hasattr(self, 'numerical_imputer'):
            objects_to_save['numerical_imputer'] = self.numerical_imputer
        
        if hasattr(self, 'categorical_imputer'):
            objects_to_save['categorical_imputer'] = self.categorical_imputer
        
        if self.feature_selector:
            objects_to_save['feature_selector'] = self.feature_selector
        
        # Save each object separately
        for name, obj in objects_to_save.items():
            if obj is not None:
                save_path = self.processed_data_path / f'{name}.pkl'
                with open(save_path, 'wb') as f:
                    pickle.dump(obj, f)
                logger.info(f"Saved {name} to {save_path}")
    
    def _format_data_output(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Format data output based on framework"""
        framework = self.config.get('model', {}).get('framework', 'scikit-learn')
        
        if framework == 'pytorch':
            # Return torch tensors if PyTorch is available
            try:
                import torch
                return (
                    (torch.FloatTensor(X_train), torch.LongTensor(y_train)),
                    (torch.FloatTensor(X_val), torch.LongTensor(y_val)),
                    (torch.FloatTensor(X_test), torch.LongTensor(y_test))
                )
            except ImportError:
                logger.warning("PyTorch not available, returning numpy arrays")
        
        # Default: return numpy arrays
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def load_preprocessing_objects(self):
        """Load saved preprocessing objects"""
        objects_to_load = [
            'scaler', 'feature_names', 'label_encoders',
            'numerical_imputer', 'categorical_imputer', 'feature_selector'
        ]
        
        for obj_name in objects_to_load:
            obj_path = self.processed_data_path / f'{obj_name}.pkl'
            if obj_path.exists():
                with open(obj_path, 'rb') as f:
                    setattr(self, obj_name, pickle.load(f))
                logger.info(f"Loaded {obj_name} from {obj_path}")
    
    def transform_new_data(self, X_new):
        """
        Transform new data using fitted preprocessing objects
        
        Args:
            X_new: New data to transform
        
        Returns:
            Transformed data
        """
        if self.scaler is None:
            raise ValueError("Preprocessing objects not fitted. Run prepare_data first.")
        
        X_transformed = X_new.copy()
        
        # Apply same preprocessing steps
        if hasattr(self, 'numerical_imputer') and self.numerical_imputer:
            numerical_cols = X_transformed.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                X_transformed[numerical_cols] = self.numerical_imputer.transform(
                    X_transformed[numerical_cols]
                )
        
        if hasattr(self, 'categorical_imputer') and self.categorical_imputer:
            categorical_cols = X_transformed.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                X_transformed[categorical_cols] = self.categorical_imputer.transform(
                    X_transformed[categorical_cols]
                )
        
        # Encode categorical features
        for col, encoder in self.label_encoders.items():
            if col in X_transformed.columns:
                X_transformed[col] = encoder.transform(X_transformed[col].astype(str))
        
        # Scale features
        X_scaled = self.scaler.transform(X_transformed.values)
        
        # Feature selection
        if self.feature_selector:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        return X_scaled

if __name__ == "__main__":
    # Example usage
    config = {
        'data': {
            'random_seed': 42,
            'train_split': 0.7,
            'validation_split': 0.2,
            'test_split': 0.1,
            'stratify': True
        },
        'model': {
            'framework': 'scikit-learn'
        },
        'preprocessing': {
            'scaling': 'standard',
            'feature_selection': False,
            'n_features': 10
        }
    }
    
    pipeline = DataPipeline(config)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = pipeline.prepare_data()
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")