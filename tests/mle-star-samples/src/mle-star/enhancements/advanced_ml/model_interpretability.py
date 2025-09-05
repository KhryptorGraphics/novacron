"""
Model Interpretability
=====================

Model interpretability and explainability for MLE-Star using SHAP and LIME:
- Global feature importance analysis
- Local prediction explanations
- Model behavior visualization
- Bias detection and fairness analysis
- Decision boundary visualization
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    import lime.lime_text
    import lime.lime_image
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance, partial_dependence
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class InterpretabilityConfig:
    """Configuration for model interpretability"""
    # General settings
    explain_global: bool = True
    explain_local: bool = True
    explain_samples: int = 100  # Number of samples for local explanations
    
    # SHAP configuration
    shap_explainer: str = "auto"  # auto, tree, linear, kernel, deep, gradient
    shap_background_samples: int = 100
    shap_max_evals: int = 1000
    
    # LIME configuration
    lime_mode: str = "tabular"  # tabular, text, image
    lime_num_features: int = 10
    lime_num_samples: int = 1000
    
    # Feature importance
    permutation_importance: bool = True
    partial_dependence: bool = True
    feature_interaction: bool = True
    
    # Visualization
    create_plots: bool = True
    save_plots: bool = True
    plot_format: str = "png"  # png, pdf, svg
    plot_dpi: int = 300
    
    # Bias and fairness
    fairness_analysis: bool = True
    sensitive_features: Optional[List[str]] = None
    fairness_metrics: List[str] = None
    
    # Output configuration
    output_path: str = "./interpretability_output"
    save_explanations: bool = True


class SHAPExplainer:
    """SHAP-based model explanations"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.explainer = None
        self.shap_values = None
        
    def initialize(self, model, X_background: pd.DataFrame, 
                  feature_names: Optional[List[str]] = None) -> bool:
        """Initialize SHAP explainer"""
        try:
            # Select appropriate explainer
            if self.config.shap_explainer == "auto":
                explainer_type = self._detect_explainer_type(model)
            else:
                explainer_type = self.config.shap_explainer
            
            # Sample background data if too large
            if len(X_background) > self.config.shap_background_samples:
                background_sample = X_background.sample(
                    n=self.config.shap_background_samples, 
                    random_state=42
                )
            else:
                background_sample = X_background
            
            # Create explainer
            if explainer_type == "tree":
                self.explainer = shap.TreeExplainer(model, background_sample)
            elif explainer_type == "linear":
                self.explainer = shap.LinearExplainer(model, background_sample)
            elif explainer_type == "kernel":
                self.explainer = shap.KernelExplainer(model.predict, background_sample)
            elif explainer_type == "deep":
                self.explainer = shap.DeepExplainer(model, background_sample.values)
            elif explainer_type == "gradient":
                self.explainer = shap.GradientExplainer(model, background_sample.values)
            else:
                # Default to KernelExplainer
                self.explainer = shap.KernelExplainer(model.predict, background_sample)
            
            self.feature_names = feature_names or list(X_background.columns)
            
            logger.info(f"SHAP explainer initialized: {explainer_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {e}")
            return False
    
    def _detect_explainer_type(self, model) -> str:
        """Detect appropriate SHAP explainer for model"""
        model_name = model.__class__.__name__.lower()
        
        # Tree-based models
        if any(tree_name in model_name for tree_name in 
               ['randomforest', 'xgb', 'lgb', 'decisiontree', 'gradientboosting']):
            return "tree"
        
        # Linear models
        if any(linear_name in model_name for linear_name in
               ['linear', 'logistic', 'ridge', 'lasso', 'elasticnet']):
            return "linear"
        
        # Default to kernel explainer
        return "kernel"
    
    def explain_global(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Generate global explanations"""
        try:
            if self.explainer is None:
                raise ValueError("SHAP explainer not initialized")
            
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # Use the first class for global importance
                shap_values_global = shap_values[0]
            else:
                shap_values_global = shap_values
            
            # Calculate feature importance
            feature_importance = np.abs(shap_values_global).mean(axis=0)
            
            # Create feature importance ranking
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            global_explanation = {
                'feature_importance': importance_df.to_dict('records'),
                'shap_values': shap_values_global.tolist() if isinstance(shap_values_global, np.ndarray) else shap_values_global,
                'expected_value': getattr(self.explainer, 'expected_value', None),
                'feature_names': self.feature_names
            }
            
            # Store for plotting
            self.shap_values = shap_values
            
            return global_explanation
            
        except Exception as e:
            logger.error(f"Global SHAP explanation failed: {e}")
            return {}
    
    def explain_local(self, X: pd.DataFrame, indices: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """Generate local explanations for specific instances"""
        try:
            if self.explainer is None:
                raise ValueError("SHAP explainer not initialized")
            
            if indices is None:
                # Select random samples
                n_samples = min(self.config.explain_samples, len(X))
                indices = np.random.choice(len(X), n_samples, replace=False)
            
            X_explain = X.iloc[indices]
            shap_values = self.explainer.shap_values(X_explain)
            
            explanations = []
            
            for i, idx in enumerate(indices):
                if isinstance(shap_values, list):
                    # Multi-class case - use first class
                    instance_shap = shap_values[0][i]
                else:
                    instance_shap = shap_values[i]
                
                # Create explanation for this instance
                explanation = {
                    'instance_index': int(idx),
                    'feature_contributions': dict(zip(self.feature_names, instance_shap)),
                    'instance_values': X.iloc[idx].to_dict(),
                    'prediction_impact': float(np.sum(instance_shap))
                }
                
                explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Local SHAP explanation failed: {e}")
            return []
    
    def create_plots(self, X: pd.DataFrame, save_path: Optional[Path] = None):
        """Create SHAP visualization plots"""
        try:
            if self.shap_values is None:
                logger.warning("No SHAP values available for plotting")
                return
            
            if save_path:
                save_path.mkdir(parents=True, exist_ok=True)
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
            if save_path:
                plt.savefig(save_path / f"shap_summary.{self.config.plot_format}", 
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(self.shap_values, X, plot_type="bar", 
                            feature_names=self.feature_names, show=False)
            if save_path:
                plt.savefig(save_path / f"shap_importance.{self.config.plot_format}",
                           dpi=self.config.plot_dpi, bbox_inches='tight')
            plt.close()
            
            # Partial dependence plots for top features
            if hasattr(self, 'feature_names') and len(self.feature_names) > 0:
                top_features = self.feature_names[:5]  # Top 5 features
                
                for feature in top_features:
                    try:
                        plt.figure(figsize=(8, 6))
                        shap.partial_dependence_plot(
                            feature, self.explainer.model.predict, X, ice=False, 
                            model_expected_value=True, feature_expected_value=True, show=False
                        )
                        if save_path:
                            plt.savefig(save_path / f"shap_pdp_{feature}.{self.config.plot_format}",
                                      dpi=self.config.plot_dpi, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        logger.warning(f"Failed to create PDP plot for {feature}: {e}")
                        plt.close()
                        
        except Exception as e:
            logger.error(f"SHAP plotting failed: {e}")


class LIMEExplainer:
    """LIME-based model explanations"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        self.explainer = None
        
    def initialize(self, X_train: pd.DataFrame, feature_names: Optional[List[str]] = None,
                  class_names: Optional[List[str]] = None) -> bool:
        """Initialize LIME explainer"""
        try:
            self.feature_names = feature_names or list(X_train.columns)
            self.class_names = class_names
            
            if self.config.lime_mode == "tabular":
                self.explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train.values,
                    feature_names=self.feature_names,
                    class_names=self.class_names,
                    mode='classification' if self.class_names else 'regression',
                    discretize_continuous=True
                )
            
            logger.info(f"LIME explainer initialized: {self.config.lime_mode}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {e}")
            return False
    
    def explain_instance(self, model, instance: pd.Series, 
                        predict_proba: bool = True) -> Dict[str, Any]:
        """Explain a single instance"""
        try:
            if self.explainer is None:
                raise ValueError("LIME explainer not initialized")
            
            # Get prediction function
            if predict_proba and hasattr(model, 'predict_proba'):
                predict_fn = model.predict_proba
            else:
                predict_fn = model.predict
            
            # Generate explanation
            explanation = self.explainer.explain_instance(
                instance.values,
                predict_fn,
                num_features=self.config.lime_num_features,
                num_samples=self.config.lime_num_samples
            )
            
            # Extract feature contributions
            feature_contributions = {}
            for feature_idx, contribution in explanation.as_list():
                if isinstance(feature_idx, str):
                    feature_name = feature_idx
                else:
                    feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"feature_{feature_idx}"
                
                feature_contributions[feature_name] = contribution
            
            result = {
                'instance_values': instance.to_dict(),
                'feature_contributions': feature_contributions,
                'intercept': explanation.intercept[0] if hasattr(explanation, 'intercept') else 0,
                'prediction_probability': getattr(explanation, 'predict_proba', None),
                'local_pred': explanation.local_pred[0] if hasattr(explanation, 'local_pred') else None
            }
            
            return result
            
        except Exception as e:
            logger.error(f"LIME instance explanation failed: {e}")
            return {}


class BiasAnalyzer:
    """Bias and fairness analysis"""
    
    def __init__(self, config: InterpretabilityConfig):
        self.config = config
        
    def analyze_fairness(self, model, X: pd.DataFrame, y: pd.Series,
                        sensitive_features: List[str]) -> Dict[str, Any]:
        """Analyze model fairness across sensitive features"""
        try:
            fairness_results = {}
            
            for sensitive_feature in sensitive_features:
                if sensitive_feature not in X.columns:
                    logger.warning(f"Sensitive feature {sensitive_feature} not found in data")
                    continue
                
                # Get unique values of sensitive feature
                sensitive_values = X[sensitive_feature].unique()
                
                feature_analysis = {
                    'feature_name': sensitive_feature,
                    'groups': {},
                    'fairness_metrics': {}
                }
                
                # Analyze each group
                for value in sensitive_values:
                    group_mask = X[sensitive_feature] == value
                    X_group = X[group_mask]
                    y_group = y[group_mask]
                    
                    if len(X_group) == 0:
                        continue
                    
                    # Make predictions
                    y_pred = model.predict(X_group)
                    
                    # Calculate metrics
                    if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
                        # Classification metrics
                        from sklearn.metrics import accuracy_score, precision_score, recall_score
                        
                        accuracy = accuracy_score(y_group, y_pred)
                        precision = precision_score(y_group, y_pred, average='weighted', zero_division=0)
                        recall = recall_score(y_group, y_pred, average='weighted', zero_division=0)
                        
                        group_metrics = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'sample_count': len(X_group),
                            'positive_rate': (y_pred == 1).mean() if len(np.unique(y_pred)) > 1 else 0
                        }
                    else:
                        # Regression metrics
                        from sklearn.metrics import mean_squared_error, mean_absolute_error
                        
                        mse = mean_squared_error(y_group, y_pred)
                        mae = mean_absolute_error(y_group, y_pred)
                        
                        group_metrics = {
                            'mse': mse,
                            'mae': mae,
                            'sample_count': len(X_group),
                            'prediction_mean': y_pred.mean()
                        }
                    
                    feature_analysis['groups'][str(value)] = group_metrics
                
                # Calculate fairness metrics
                if len(feature_analysis['groups']) > 1:
                    feature_analysis['fairness_metrics'] = self._calculate_fairness_metrics(
                        feature_analysis['groups']
                    )
                
                fairness_results[sensitive_feature] = feature_analysis
            
            return fairness_results
            
        except Exception as e:
            logger.error(f"Fairness analysis failed: {e}")
            return {}
    
    def _calculate_fairness_metrics(self, groups: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate fairness metrics across groups"""
        fairness_metrics = {}
        
        # Get all metric names
        metric_names = set()
        for group_metrics in groups.values():
            metric_names.update(group_metrics.keys())
        
        metric_names.discard('sample_count')  # Don't calculate fairness for sample count
        
        for metric_name in metric_names:
            values = []
            for group_metrics in groups.values():
                if metric_name in group_metrics:
                    values.append(group_metrics[metric_name])
            
            if len(values) > 1:
                # Statistical parity (difference between max and min)
                fairness_metrics[f"{metric_name}_parity"] = max(values) - min(values)
                
                # Ratio of min to max
                if max(values) > 0:
                    fairness_metrics[f"{metric_name}_ratio"] = min(values) / max(values)
                
                # Coefficient of variation
                if np.mean(values) > 0:
                    fairness_metrics[f"{metric_name}_cv"] = np.std(values) / np.mean(values)
        
        return fairness_metrics


class ModelInterpretability(BaseEnhancement):
    """Model Interpretability enhancement for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="model_interpretability",
            version="1.0.0",
            enabled=SHAP_AVAILABLE or LIME_AVAILABLE,
            priority=25,
            parameters={
                "explain_global": True,
                "explain_local": True,
                "explain_samples": 100,
                "shap_explainer": "auto",
                "shap_background_samples": 100,
                "shap_max_evals": 1000,
                "lime_mode": "tabular",
                "lime_num_features": 10,
                "lime_num_samples": 1000,
                "permutation_importance": True,
                "partial_dependence": True,
                "feature_interaction": True,
                "create_plots": True,
                "save_plots": True,
                "plot_format": "png",
                "plot_dpi": 300,
                "fairness_analysis": True,
                "sensitive_features": None,
                "fairness_metrics": ["statistical_parity", "equalized_odds"],
                "output_path": "./interpretability_output",
                "save_explanations": True
            }
        )
    
    def initialize(self) -> bool:
        """Initialize Model Interpretability"""
        available_methods = []
        
        if SHAP_AVAILABLE:
            available_methods.append("SHAP")
        if LIME_AVAILABLE:
            available_methods.append("LIME")
        if SKLEARN_AVAILABLE:
            available_methods.append("Permutation Importance")
        
        if not available_methods:
            self._logger.error("No interpretability libraries available. Install SHAP, LIME, or use sklearn")
            return False
        
        try:
            # Create configuration
            self.interp_config = InterpretabilityConfig(**self.config.parameters)
            
            # Initialize explainers (will be done when needed)
            self.shap_explainer = None
            self.lime_explainer = None
            self.bias_analyzer = None
            
            if SHAP_AVAILABLE:
                self.shap_explainer = SHAPExplainer(self.interp_config)
            if LIME_AVAILABLE:
                self.lime_explainer = LIMEExplainer(self.interp_config)
            
            self.bias_analyzer = BiasAnalyzer(self.interp_config)
            
            self._logger.info(f"Model Interpretability initialized with: {', '.join(available_methods)}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Model Interpretability: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with interpretability capabilities"""
        enhanced = workflow.copy()
        
        # Add interpretability configuration
        if 'interpretability' not in enhanced:
            enhanced['interpretability'] = {}
        
        enhanced['interpretability'] = {
            'enabled': True,
            'global_explanations': self.interp_config.explain_global,
            'local_explanations': self.interp_config.explain_local,
            'methods': {
                'shap': SHAP_AVAILABLE,
                'lime': LIME_AVAILABLE,
                'permutation_importance': SKLEARN_AVAILABLE
            },
            'visualization': {
                'create_plots': self.interp_config.create_plots,
                'save_plots': self.interp_config.save_plots,
                'plot_format': self.interp_config.plot_format
            },
            'fairness_analysis': {
                'enabled': self.interp_config.fairness_analysis,
                'sensitive_features': self.interp_config.sensitive_features,
                'metrics': self.interp_config.fairness_metrics
            }
        }
        
        # Enhance MLE-Star stages with interpretability
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 5: Results Evaluation - Add interpretability analysis
            if '5_results_evaluation' in stages:
                if 'interpretability_analysis' not in stages['5_results_evaluation']:
                    stages['5_results_evaluation']['interpretability_analysis'] = [
                        'global_feature_importance',
                        'local_prediction_explanations',
                        'model_behavior_visualization',
                        'bias_and_fairness_assessment'
                    ]
            
            # Stage 6: Refinement - Add interpretability-driven improvements
            if '6_refinement' in stages:
                if 'interpretability_refinement' not in stages['6_refinement']:
                    stages['6_refinement']['interpretability_refinement'] = [
                        'feature_selection_based_on_importance',
                        'bias_mitigation_strategies',
                        'model_simplification',
                        'explanation_quality_improvement'
                    ]
            
            # Stage 7: Deployment Prep - Add explainability documentation
            if '7_deployment_prep' in stages:
                if 'explainability_documentation' not in stages['7_deployment_prep']:
                    stages['7_deployment_prep']['explainability_documentation'] = [
                        'model_explanation_reports',
                        'feature_importance_documentation',
                        'bias_assessment_reports',
                        'interpretability_guidelines'
                    ]
        
        # Add interpretability specific evaluation metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'interpretability_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['interpretability_metrics'] = [
                'explanation_consistency',
                'explanation_stability',
                'feature_importance_agreement',
                'bias_metrics',
                'explanation_coverage'
            ]
        
        self._logger.debug("Enhanced workflow with interpretability capabilities")
        return enhanced
    
    def explain_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                     y_test: pd.Series, feature_names: Optional[List[str]] = None,
                     class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive model explanations"""
        try:
            self._logger.info("Starting comprehensive model explanation")
            
            explanations = {}
            feature_names = feature_names or list(X_train.columns)
            
            # SHAP explanations
            if self.shap_explainer and SHAP_AVAILABLE:
                self._logger.info("Generating SHAP explanations")
                
                # Initialize SHAP explainer
                success = self.shap_explainer.initialize(model, X_train, feature_names)
                
                if success:
                    # Global explanations
                    if self.interp_config.explain_global:
                        global_explanation = self.shap_explainer.explain_global(X_test)
                        explanations['shap_global'] = global_explanation
                    
                    # Local explanations
                    if self.interp_config.explain_local:
                        local_explanations = self.shap_explainer.explain_local(X_test)
                        explanations['shap_local'] = local_explanations
                    
                    # Create plots
                    if self.interp_config.create_plots:
                        output_path = Path(self.interp_config.output_path) / "shap_plots"
                        self.shap_explainer.create_plots(X_test, output_path)
            
            # LIME explanations
            if self.lime_explainer and LIME_AVAILABLE:
                self._logger.info("Generating LIME explanations")
                
                # Initialize LIME explainer
                success = self.lime_explainer.initialize(X_train, feature_names, class_names)
                
                if success and self.interp_config.explain_local:
                    # Generate explanations for sample instances
                    lime_explanations = []
                    n_samples = min(self.interp_config.explain_samples, len(X_test))
                    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)
                    
                    for idx in sample_indices[:10]:  # Limit to 10 for performance
                        explanation = self.lime_explainer.explain_instance(
                            model, X_test.iloc[idx]
                        )
                        explanation['instance_index'] = int(idx)
                        lime_explanations.append(explanation)
                    
                    explanations['lime_local'] = lime_explanations
            
            # Permutation importance (sklearn)
            if SKLEARN_AVAILABLE and self.interp_config.permutation_importance:
                self._logger.info("Calculating permutation importance")
                
                try:
                    perm_importance = permutation_importance(
                        model, X_test, y_test, n_repeats=5, random_state=42
                    )
                    
                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance_mean': perm_importance.importances_mean,
                        'importance_std': perm_importance.importances_std
                    }).sort_values('importance_mean', ascending=False)
                    
                    explanations['permutation_importance'] = importance_df.to_dict('records')
                    
                except Exception as e:
                    self._logger.warning(f"Permutation importance failed: {e}")
            
            # Bias analysis
            if self.interp_config.fairness_analysis and self.interp_config.sensitive_features:
                self._logger.info("Conducting fairness analysis")
                
                sensitive_features = [f for f in self.interp_config.sensitive_features 
                                   if f in X_test.columns]
                
                if sensitive_features:
                    fairness_analysis = self.bias_analyzer.analyze_fairness(
                        model, X_test, y_test, sensitive_features
                    )
                    explanations['fairness_analysis'] = fairness_analysis
            
            # Save explanations
            if self.interp_config.save_explanations:
                self._save_explanations(explanations)
            
            self._logger.info("Model explanation completed")
            return explanations
            
        except Exception as e:
            self._logger.error(f"Model explanation failed: {e}")
            return {}
    
    def _save_explanations(self, explanations: Dict[str, Any]):
        """Save explanations to file"""
        try:
            output_path = Path(self.interp_config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            json_explanations = convert_for_json(explanations)
            
            with open(output_path / "explanations.json", 'w') as f:
                json.dump(json_explanations, f, indent=2)
            
            self._logger.info(f"Explanations saved to: {output_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to save explanations: {e}")
    
    def create_interpretability_report(self, explanations: Dict[str, Any],
                                     model_name: str = "Model") -> str:
        """Create interpretability report"""
        try:
            report = f"# {model_name} Interpretability Report\n\n"
            
            # Global feature importance
            if 'shap_global' in explanations:
                report += "## Global Feature Importance (SHAP)\n\n"
                
                importance_data = explanations['shap_global'].get('feature_importance', [])
                if importance_data:
                    report += "| Feature | Importance |\n|---------|------------|\n"
                    for item in importance_data[:10]:  # Top 10 features
                        report += f"| {item['feature']} | {item['importance']:.4f} |\n"
                    report += "\n"
            
            if 'permutation_importance' in explanations:
                report += "## Permutation Importance\n\n"
                
                report += "| Feature | Importance | Std Dev |\n|---------|------------|----------|\n"
                for item in explanations['permutation_importance'][:10]:
                    report += f"| {item['feature']} | {item['importance_mean']:.4f} | {item['importance_std']:.4f} |\n"
                report += "\n"
            
            # Fairness analysis
            if 'fairness_analysis' in explanations:
                report += "## Fairness Analysis\n\n"
                
                for feature, analysis in explanations['fairness_analysis'].items():
                    report += f"### {feature}\n\n"
                    
                    # Group performance
                    report += "| Group | Sample Count | Accuracy | Precision | Recall |\n"
                    report += "|-------|--------------|----------|-----------|--------|\n"
                    
                    for group, metrics in analysis['groups'].items():
                        acc = metrics.get('accuracy', 'N/A')
                        prec = metrics.get('precision', 'N/A')
                        rec = metrics.get('recall', 'N/A')
                        count = metrics.get('sample_count', 'N/A')
                        
                        report += f"| {group} | {count} | {acc:.4f if acc != 'N/A' else 'N/A'} | "
                        report += f"{prec:.4f if prec != 'N/A' else 'N/A'} | {rec:.4f if rec != 'N/A' else 'N/A'} |\n"
                    
                    report += "\n"
                    
                    # Fairness metrics
                    if 'fairness_metrics' in analysis:
                        report += "**Fairness Metrics:**\n\n"
                        for metric, value in analysis['fairness_metrics'].items():
                            report += f"- {metric}: {value:.4f}\n"
                        report += "\n"
            
            # Key insights
            report += "## Key Insights\n\n"
            
            if 'shap_global' in explanations:
                importance_data = explanations['shap_global'].get('feature_importance', [])
                if importance_data:
                    top_feature = importance_data[0]['feature']
                    report += f"- Most important feature: **{top_feature}**\n"
            
            if 'fairness_analysis' in explanations:
                report += "- Fairness analysis completed for sensitive features\n"
                
                # Check for potential bias
                bias_detected = False
                for feature, analysis in explanations['fairness_analysis'].items():
                    fairness_metrics = analysis.get('fairness_metrics', {})
                    for metric, value in fairness_metrics.items():
                        if 'parity' in metric and abs(value) > 0.1:  # Threshold for concern
                            report += f"- Potential bias detected in {feature} ({metric}: {value:.4f})\n"
                            bias_detected = True
                
                if not bias_detected:
                    report += "- No significant bias detected in the analyzed features\n"
            
            return report
            
        except Exception as e:
            self._logger.error(f"Failed to create interpretability report: {e}")
            return "Report generation failed"