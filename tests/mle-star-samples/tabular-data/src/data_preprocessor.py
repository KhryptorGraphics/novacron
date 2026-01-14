"""Data preprocessing and feature engineering for tabular data."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class TabularDataPreprocessor:
    """Comprehensive preprocessing pipeline for tabular data."""
    
    def __init__(self, target_column: str, task_type: str = 'classification'):
        self.target_column = target_column
        self.task_type = task_type  # 'classification' or 'regression'
        
        # Preprocessing components
        self.label_encoders = {}
        self.scaler = None
        self.feature_selector = None
        self.polynomial_features = None
        self.column_transformer = None
        self.imputer = None
        
        # Data information
        self.numerical_features = []
        self.categorical_features = []
        self.feature_names = []
        self.n_features_original = 0
        self.n_features_final = 0
        
        # Preprocessing history
        self.preprocessing_steps = []
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze dataset characteristics and quality."""
        
        analysis = {
            'basic_info': {
                'n_samples': len(df),
                'n_features': len(df.columns) - 1,  # Excluding target
                'memory_usage': df.memory_usage(deep=True).sum() / (1024**2),  # MB
                'dtypes_count': df.dtypes.value_counts().to_dict()
            },
            'missing_values': {
                'total_missing': df.isnull().sum().sum(),
                'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
                'columns_with_missing': df.columns[df.isnull().any()].tolist()
            },
            'target_analysis': self._analyze_target(df),
            'feature_types': self._identify_feature_types(df),
            'duplicates': {
                'duplicate_rows': df.duplicated().sum(),
                'unique_samples': df.drop_duplicates().shape[0]
            },
            'outliers': self._detect_outliers(df)
        }
        
        return analysis
    
    def _analyze_target(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze target variable characteristics."""
        
        target = df[self.target_column]
        
        if self.task_type == 'classification':
            value_counts = target.value_counts()
            return {
                'type': 'classification',
                'n_classes': len(value_counts),
                'class_distribution': value_counts.to_dict(),
                'class_balance': (value_counts / len(df)).to_dict(),
                'is_balanced': (value_counts.max() / value_counts.min()) < 3
            }
        else:  # regression
            return {
                'type': 'regression',
                'mean': target.mean(),
                'std': target.std(),
                'min': target.min(),
                'max': target.max(),
                'skewness': target.skew(),
                'kurtosis': target.kurtosis(),
                'distribution': 'normal' if abs(target.skew()) < 0.5 else 'skewed'
            }
    
    def _identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify numerical and categorical features."""
        
        features = df.drop(columns=[self.target_column])
        
        numerical = features.select_dtypes(include=[np.number]).columns.tolist()
        categorical = features.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Check for numerical features that should be treated as categorical
        for col in numerical.copy():
            unique_count = features[col].nunique()
            if unique_count < 10 and unique_count / len(features) < 0.05:
                numerical.remove(col)
                categorical.append(col)
        
        self.numerical_features = numerical
        self.categorical_features = categorical
        
        return {
            'numerical': numerical,
            'categorical': categorical,
            'numerical_count': len(numerical),
            'categorical_count': len(categorical)
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numerical features using IQR method."""
        
        outliers_info = {}
        
        for col in self.numerical_features:
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            outliers_info[col] = {
                'count': len(outliers),
                'percentage': len(outliers) / len(data) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return outliers_info
    
    def create_synthetic_dataset(self, n_samples: int = 1000, 
                               task_type: str = 'classification') -> pd.DataFrame:
        """Create synthetic dataset for demonstration."""
        
        np.random.seed(42)
        
        if task_type == 'classification':
            # Binary classification dataset
            data = {
                'age': np.random.normal(35, 10, n_samples),
                'income': np.random.lognormal(10, 1, n_samples),
                'education_years': np.random.normal(14, 3, n_samples),
                'experience': np.random.exponential(5, n_samples),
                'credit_score': np.random.normal(650, 100, n_samples),
                'debt_ratio': np.random.beta(2, 5, n_samples),
                'city': np.random.choice(['New York', 'Chicago', 'LA', 'Houston', 'Phoenix'], n_samples),
                'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
                'employment_type': np.random.choice(['Full-time', 'Part-time', 'Contract', 'Unemployed'], n_samples)
            }
            
            df = pd.DataFrame(data)
            
            # Ensure positive values where appropriate
            df['age'] = np.abs(df['age'])
            df['education_years'] = np.clip(df['education_years'], 8, 25)
            df['credit_score'] = np.clip(df['credit_score'], 300, 850)
            
            # Create target with some logical relationship
            target_prob = (
                0.1 +
                0.3 * (df['income'] > df['income'].median()).astype(int) +
                0.2 * (df['credit_score'] > 650).astype(int) +
                0.2 * (df['education_years'] > 16).astype(int) +
                0.2 * (df['debt_ratio'] < 0.3).astype(int)
            )
            
            df['approved'] = np.random.binomial(1, target_prob, n_samples)
            self.target_column = 'approved'
            
        else:  # regression
            # House price prediction dataset
            data = {
                'square_feet': np.random.normal(2000, 500, n_samples),
                'bedrooms': np.random.poisson(3, n_samples),
                'bathrooms': np.random.gamma(2, 1, n_samples),
                'age': np.random.exponential(10, n_samples),
                'garage_size': np.random.choice([0, 1, 2, 3], n_samples),
                'lot_size': np.random.lognormal(8, 0.5, n_samples),
                'neighborhood': np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples),
                'house_type': np.random.choice(['Single Family', 'Condo', 'Townhouse'], n_samples),
                'heating_type': np.random.choice(['Gas', 'Electric', 'Oil'], n_samples)
            }
            
            df = pd.DataFrame(data)
            
            # Ensure reasonable values
            df['square_feet'] = np.clip(df['square_feet'], 500, 5000)
            df['bedrooms'] = np.clip(df['bedrooms'], 1, 6)
            df['bathrooms'] = np.clip(df['bathrooms'], 1, 5)
            df['age'] = np.clip(df['age'], 0, 50)
            
            # Create target with logical relationship
            base_price = (
                100 * df['square_feet'] +
                10000 * df['bedrooms'] +
                15000 * df['bathrooms'] +
                500 * df['lot_size'] -
                1000 * df['age'] +
                5000 * (df['neighborhood'] == 'Downtown').astype(int) +
                np.random.normal(0, 20000, n_samples)  # noise
            )
            
            df['price'] = np.clip(base_price, 50000, 1000000)
            self.target_column = 'price'
        
        # Add some missing values randomly
        missing_cols = np.random.choice(df.columns[:-1], size=3, replace=False)
        for col in missing_cols:
            missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
            df.loc[missing_indices, col] = np.nan
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply comprehensive preprocessing pipeline."""
        
        if fit:
            self.preprocessing_steps = []
            self.n_features_original = len(df.columns) - 1
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        # Step 1: Handle missing values
        X = self._handle_missing_values(X, fit)
        
        # Step 2: Encode categorical variables
        X = self._encode_categorical_features(X, fit)
        
        # Step 3: Scale numerical features
        X = self._scale_numerical_features(X, fit)
        
        # Step 4: Feature engineering
        X = self._engineer_features(X, fit)
        
        # Step 5: Feature selection
        X = self._select_features(X, y, fit)
        
        if fit:
            self.feature_names = X.columns.tolist()
            self.n_features_final = len(X.columns)
        
        return X, y
    
    def _handle_missing_values(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Handle missing values using appropriate strategies."""
        
        if fit:
            # Choose imputation strategy
            numerical_strategy = 'median' if len(self.numerical_features) > 0 else None
            categorical_strategy = 'most_frequent' if len(self.categorical_features) > 0 else None
            
            # Create imputers
            transformers = []
            
            if numerical_strategy:
                transformers.append((
                    'numerical_imputer',
                    SimpleImputer(strategy=numerical_strategy),
                    self.numerical_features
                ))
            
            if categorical_strategy:
                transformers.append((
                    'categorical_imputer',
                    SimpleImputer(strategy=categorical_strategy),
                    self.categorical_features
                ))
            
            if transformers:
                self.imputer = ColumnTransformer(
                    transformers=transformers,
                    remainder='passthrough',
                    verbose_feature_names_out=False
                )
                
                self.imputer.fit(X)
                self.preprocessing_steps.append('Missing value imputation')
        
        if self.imputer:
            X_imputed = self.imputer.transform(X)
            X = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        
        return X
    
    def _encode_categorical_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Encode categorical features."""
        
        if not self.categorical_features:
            return X
        
        if fit:
            # Choose encoding strategy based on cardinality
            high_cardinality_threshold = 10
            
            transformers = []
            
            for col in self.categorical_features:
                cardinality = X[col].nunique()
                
                if cardinality > high_cardinality_threshold:
                    # Use target encoding for high cardinality
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                else:
                    # Use one-hot encoding for low cardinality
                    encoder = OneHotEncoder(drop='if_binary', sparse=False, handle_unknown='ignore')
                
                transformers.append((f'{col}_encoder', encoder, [col]))
            
            # Create column transformer for categorical features
            self.categorical_transformer = ColumnTransformer(
                transformers=transformers,
                remainder='passthrough',
                verbose_feature_names_out=False
            )
            
            self.categorical_transformer.fit(X)
            self.preprocessing_steps.append('Categorical encoding')
        
        if hasattr(self, 'categorical_transformer'):
            # Transform data
            X_encoded = self.categorical_transformer.transform(X)
            
            # Get feature names
            feature_names = []
            for i, (name, transformer, columns) in enumerate(self.categorical_transformer.transformers_):
                if name != 'remainder':
                    if hasattr(transformer, 'get_feature_names_out'):
                        names = transformer.get_feature_names_out(columns)
                    else:
                        names = [f"{columns[0]}_{j}" for j in range(transformer.transform(X[columns]).shape[1])]
                    feature_names.extend(names)
                else:
                    # Add names for passthrough features
                    remaining_cols = [col for col in X.columns if col not in self.categorical_features]
                    feature_names.extend(remaining_cols)
            
            X = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)
        
        return X
    
    def _scale_numerical_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Scale numerical features."""
        
        # Update numerical features list after encoding
        current_numerical = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if not current_numerical:
            return X
        
        if fit:
            # Choose scaler based on data distribution
            self.scaler = RobustScaler()  # Robust to outliers
            self.scaler.fit(X[current_numerical])
            self.preprocessing_steps.append('Feature scaling')
        
        if self.scaler:
            X_scaled = X.copy()
            X_scaled[current_numerical] = self.scaler.transform(X[current_numerical])
            X = X_scaled
        
        return X
    
    def _engineer_features(self, X: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """Create engineered features."""
        
        if fit:
            # Get current numerical features
            numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numerical_cols) >= 2:
                # Create interaction features for top features
                top_features = numerical_cols[:min(3, len(numerical_cols))]
                self.polynomial_features = PolynomialFeatures(
                    degree=2, 
                    interaction_only=True, 
                    include_bias=False
                )
                self.polynomial_features.fit(X[top_features])
                self.preprocessing_steps.append('Feature engineering (interactions)')
        
        if self.polynomial_features:
            top_features = X.select_dtypes(include=[np.number]).columns.tolist()[:3]
            if len(top_features) >= 2:
                poly_features = self.polynomial_features.transform(X[top_features])
                poly_names = self.polynomial_features.get_feature_names_out(top_features)
                
                # Add only new features (interactions)
                new_features = poly_features[:, len(top_features):]
                new_names = poly_names[len(top_features):]
                
                poly_df = pd.DataFrame(new_features, columns=new_names, index=X.index)
                X = pd.concat([X, poly_df], axis=1)
        
        return X
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, fit: bool) -> pd.DataFrame:
        """Select most important features."""
        
        if fit:
            # Choose feature selection method based on task type and number of features
            if len(X.columns) > 50:
                if self.task_type == 'classification':
                    # Use Random Forest for feature importance
                    rf = RandomForestClassifier(n_estimators=50, random_state=42)
                    self.feature_selector = SelectFromModel(rf, threshold='median')
                else:
                    rf = RandomForestRegressor(n_estimators=50, random_state=42)
                    self.feature_selector = SelectFromModel(rf, threshold='median')
                
                self.feature_selector.fit(X, y)
                self.preprocessing_steps.append('Feature selection')
        
        if self.feature_selector:
            X_selected = self.feature_selector.transform(X)
            selected_features = X.columns[self.feature_selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X
    
    def get_preprocessing_report(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing report."""
        
        return {
            'preprocessing_steps': self.preprocessing_steps,
            'original_features': self.n_features_original,
            'final_features': self.n_features_final,
            'feature_reduction': (self.n_features_original - self.n_features_final) / self.n_features_original * 100,
            'feature_types': {
                'numerical': len(self.numerical_features),
                'categorical': len(self.categorical_features)
            },
            'final_feature_names': self.feature_names[:10] if self.feature_names else []  # Show first 10
        }
    
    def visualize_data_analysis(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive data visualization."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tabular Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Missing values heatmap
        if df.isnull().sum().sum() > 0:
            sns.heatmap(df.isnull(), cbar=True, ax=axes[0, 0], cmap='viridis')
            axes[0, 0].set_title('Missing Values Pattern')
        else:
            axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('Missing Values Pattern')
        
        # 2. Target distribution
        if self.task_type == 'classification':
            df[self.target_column].value_counts().plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Target Class Distribution')
        else:
            df[self.target_column].hist(bins=30, ax=axes[0, 1])
            axes[0, 1].set_title('Target Distribution')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Feature types
        feature_types = self._identify_feature_types(df)
        type_counts = [feature_types['numerical_count'], feature_types['categorical_count']]
        axes[0, 2].pie(type_counts, labels=['Numerical', 'Categorical'], autopct='%1.1f%%')
        axes[0, 2].set_title('Feature Types Distribution')
        
        # 4. Correlation heatmap (numerical features only)
        numerical_data = df.select_dtypes(include=[np.number])
        if len(numerical_data.columns) > 1:
            corr_matrix = numerical_data.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       fmt='.2f', ax=axes[1, 0], square=True)
            axes[1, 0].set_title('Feature Correlation Matrix')
        else:
            axes[1, 0].text(0.5, 0.5, 'Insufficient Numerical Features', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Correlation Matrix')
        
        # 5. Data quality overview
        quality_metrics = {
            'Complete': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'Unique': (df.drop_duplicates().shape[0] / len(df)) * 100
        }
        
        bars = axes[1, 1].bar(quality_metrics.keys(), quality_metrics.values())
        axes[1, 1].set_title('Data Quality Metrics')
        axes[1, 1].set_ylabel('Percentage')
        axes[1, 1].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{height:.1f}%', ha='center', va='bottom')
        
        # 6. Sample size and feature count
        dataset_info = {
            'Samples': len(df),
            'Features': len(df.columns) - 1
        }
        
        bars = axes[1, 2].bar(dataset_info.keys(), dataset_info.values())
        axes[1, 2].set_title('Dataset Size')
        axes[1, 2].set_ylabel('Count')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

# MLE-Star Stage 1: Situation Analysis for Tabular Data
def analyze_tabular_situation(df: Optional[pd.DataFrame] = None, 
                             task_type: str = 'classification') -> Dict[str, Any]:
    """Analyze the tabular data situation for ML task."""
    
    # Create or use provided dataset
    if df is None:
        preprocessor = TabularDataPreprocessor('target', task_type)
        df = preprocessor.create_synthetic_dataset(task_type=task_type)
    else:
        # Assume target is the last column if not specified
        target_col = df.columns[-1]
        preprocessor = TabularDataPreprocessor(target_col, task_type)
    
    # Analyze data
    analysis = preprocessor.analyze_data(df)
    
    situation_analysis = {
        'problem_type': f'{task_type} on tabular data',
        'dataset_characteristics': analysis,
        'challenges': [
            'Mixed data types (numerical and categorical)',
            'Potential missing values and outliers',
            'Feature scaling requirements',
            'Categorical encoding complexity',
            'Feature selection for high-dimensional data',
            'Class imbalance (if classification)'
        ],
        'recommended_approaches': [
            'Comprehensive preprocessing pipeline',
            'Robust feature engineering and selection',
            'Ensemble methods (Random Forest, XGBoost)',
            'Cross-validation for reliable evaluation',
            'Hyperparameter optimization',
            'Model interpretation and explainability'
        ],
        'preprocessing_requirements': {
            'missing_value_handling': len(analysis['missing_values']['columns_with_missing']) > 0,
            'categorical_encoding': len(analysis['feature_types']['categorical']) > 0,
            'feature_scaling': len(analysis['feature_types']['numerical']) > 0,
            'outlier_treatment': any(info['percentage'] > 5 for info in analysis['outliers'].values()),
            'feature_selection': analysis['basic_info']['n_features'] > 20
        }
    }
    
    return situation_analysis

if __name__ == '__main__':
    # Test tabular data preprocessing
    print("Testing Tabular Data Preprocessing...")
    
    # Create preprocessor and synthetic data
    preprocessor = TabularDataPreprocessor('approved', 'classification')
    df = preprocessor.create_synthetic_dataset(n_samples=1000, task_type='classification')
    
    print(f"Created dataset with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Analyze data
    analysis = preprocessor.analyze_data(df)
    print("\nData Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=['approved']), df['approved'], 
        test_size=0.2, random_state=42, stratify=df['approved']
    )
    
    # Preprocess data
    X_train_processed, y_train = preprocessor.preprocess_data(
        pd.concat([X_train, y_train], axis=1), fit=True
    )
    X_test_processed, y_test = preprocessor.preprocess_data(
        pd.concat([X_test, y_test], axis=1), fit=False
    )
    
    print(f"\nProcessing Results:")
    print(f"Original features: {X_train.shape[1]}")
    print(f"Processed features: {X_train_processed.shape[1]}")
    
    # Get preprocessing report
    report = preprocessor.get_preprocessing_report()
    print("\nPreprocessing Report:")
    for key, value in report.items():
        print(f"{key}: {value}")
    
    # Situation analysis
    situation = analyze_tabular_situation(df, 'classification')
    print("\n=== Tabular Data Situation Analysis ===")
    for key, value in situation.items():
        print(f"{key}: {value}")
