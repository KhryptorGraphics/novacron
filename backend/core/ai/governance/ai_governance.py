"""
AI Model Governance & Ethics Framework for NovaCron
Implements responsible AI with bias detection, explainability, and regulatory compliance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import asyncio
from datetime import datetime, timedelta
import logging
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque

# ML and AI Ethics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import shap
import lime
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import AnchorTabular, CounterFactual
from interpretML import show
from interpret.glassbox import ExplainableBoostingRegressor

# Fairness and Bias
import fairlearn
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

# Model Versioning and Lineage
import mlflow
from mlflow.tracking import MlflowClient
import dvc.api
import git
from dagshub import dagshub_logger

# Model Monitoring
from evidently.model_profile import Profile
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, CatTargetDriftTab
import whylogs
from whylogs import get_or_create_session
import great_expectations as ge

# Privacy
from differential_privacy import LaplaceMechanism, GaussianMechanism
import tenseal as ts
from opacus import PrivacyEngine
from opacus.utils import module_modification

# Documentation
from dataclasses_jsonschema import JsonSchemaMixin
import yaml
from jinja2 import Template

# Monitoring
from prometheus_client import Counter, Gauge, Histogram, Summary

# Security
from cryptography.fernet import Fernet
import jwt

warnings = [...]
logger = logging.getLogger(__name__)

# Prometheus metrics
models_evaluated = Counter('ai_models_evaluated_total', 'Total models evaluated')
bias_detected = Counter('bias_detections_total', 'Bias detection events')
fairness_score = Gauge('model_fairness_score', 'Model fairness score', ['model'])
explainability_score = Gauge('model_explainability_score', 'Explainability score', ['model'])
privacy_violations = Counter('privacy_violations_total', 'Privacy violations detected')
compliance_checks = Counter('compliance_checks_total', 'Compliance checks performed')
drift_detected = Counter('model_drift_detected_total', 'Model drift detections')
governance_score = Gauge('overall_governance_score', 'Overall AI governance score')

class BiasType(Enum):
    """Types of bias to detect"""
    DEMOGRAPHIC = "demographic"
    SELECTION = "selection"
    MEASUREMENT = "measurement"
    AGGREGATION = "aggregation"
    REPRESENTATION = "representation"
    HISTORICAL = "historical"
    EVALUATION = "evaluation"

class ComplianceFramework(Enum):
    """Regulatory compliance frameworks"""
    EU_AI_ACT = "eu_ai_act"
    GDPR = "gdpr"
    CCPA = "ccpa"
    ISO_23053 = "iso_23053"
    IEEE_7000 = "ieee_7000"
    NIST_AI_RMF = "nist_ai_rmf"

class ExplainabilityMethod(Enum):
    """Model explainability methods"""
    SHAP = "shap"
    LIME = "lime"
    ANCHOR = "anchor"
    COUNTERFACTUAL = "counterfactual"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    ATTENTION_WEIGHTS = "attention_weights"

class PrivacyTechnique(Enum):
    """Privacy preservation techniques"""
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    FEDERATED_LEARNING = "federated_learning"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    SECURE_MULTIPARTY = "secure_multiparty"
    DATA_ANONYMIZATION = "data_anonymization"

@dataclass
class ModelCard:
    """Model documentation card"""
    model_id: str
    model_name: str
    version: str
    description: str
    intended_use: str
    training_data: Dict[str, Any]
    evaluation_metrics: Dict[str, float]
    limitations: List[str]
    ethical_considerations: List[str]
    fairness_metrics: Dict[str, float]
    explainability_methods: List[ExplainabilityMethod]
    privacy_measures: List[PrivacyTechnique]
    created_date: datetime
    last_updated: datetime
    owner: str
    contact: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BiasAssessment:
    """Bias assessment results"""
    model_id: str
    bias_detected: bool
    bias_types: List[BiasType]
    affected_groups: List[str]
    disparity_metrics: Dict[str, float]
    recommendations: List[str]
    mitigation_applied: bool
    assessment_date: datetime

@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    model_id: str
    frameworks: List[ComplianceFramework]
    compliant: bool
    violations: List[str]
    requirements_met: Dict[str, bool]
    recommendations: List[str]
    certification_status: str
    report_date: datetime

@dataclass
class ExplainabilityReport:
    """Model explainability report"""
    model_id: str
    global_explanations: Dict[str, Any]
    feature_importance: Dict[str, float]
    sample_explanations: List[Dict[str, Any]]
    explanation_methods: List[ExplainabilityMethod]
    interpretability_score: float
    report_date: datetime

class BiasDetector:
    """Detect and mitigate bias in ML models"""

    def __init__(self):
        self.fairness_metrics = {}
        self.mitigation_strategies = {}
        self.thresholds = {
            'demographic_parity': 0.1,
            'equal_opportunity': 0.1,
            'disparate_impact': 0.8
        }

    async def assess_bias(self, model, X: np.ndarray, y: np.ndarray,
                         sensitive_features: pd.DataFrame) -> BiasAssessment:
        """Assess model for various types of bias"""
        logger.info(f"Assessing bias for model {model.__class__.__name__}")

        # Get predictions
        y_pred = model.predict(X)

        # Calculate fairness metrics
        metrics = self._calculate_fairness_metrics(y, y_pred, sensitive_features)

        # Detect bias types
        bias_types = self._detect_bias_types(metrics)

        # Identify affected groups
        affected_groups = self._identify_affected_groups(metrics, sensitive_features)

        # Generate recommendations
        recommendations = self._generate_bias_recommendations(bias_types, metrics)

        # Create assessment
        assessment = BiasAssessment(
            model_id=hashlib.md5(str(model).encode()).hexdigest(),
            bias_detected=len(bias_types) > 0,
            bias_types=bias_types,
            affected_groups=affected_groups,
            disparity_metrics=metrics,
            recommendations=recommendations,
            mitigation_applied=False,
            assessment_date=datetime.now()
        )

        # Update metrics
        if assessment.bias_detected:
            bias_detected.inc()

        fairness_score.labels(model=model.__class__.__name__).set(
            self._calculate_fairness_score(metrics)
        )

        return assessment

    def _calculate_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   sensitive_features: pd.DataFrame) -> Dict[str, float]:
        """Calculate various fairness metrics"""
        metrics = {}

        # Using fairlearn's MetricFrame
        for feature_name in sensitive_features.columns:
            feature_values = sensitive_features[feature_name]

            # Selection rate
            selection_rates = MetricFrame(
                metrics=selection_rate,
                y_true=y_true,
                y_pred=y_pred,
                sensitive_features=feature_values
            )

            # Demographic parity difference
            dp_diff = demographic_parity_difference(
                y_true, y_pred, sensitive_features=feature_values
            )

            metrics[f'{feature_name}_selection_rate'] = selection_rates.overall
            metrics[f'{feature_name}_dp_difference'] = dp_diff

            # Disparate impact ratio
            group_rates = selection_rates.by_group
            if len(group_rates) == 2:
                rates = list(group_rates.values())
                if rates[1] > 0:
                    metrics[f'{feature_name}_disparate_impact'] = rates[0] / rates[1]

        return metrics

    def _detect_bias_types(self, metrics: Dict[str, float]) -> List[BiasType]:
        """Detect types of bias present"""
        bias_types = []

        # Check demographic parity
        for key, value in metrics.items():
            if 'dp_difference' in key and abs(value) > self.thresholds['demographic_parity']:
                bias_types.append(BiasType.DEMOGRAPHIC)

            if 'disparate_impact' in key and value < self.thresholds['disparate_impact']:
                bias_types.append(BiasType.SELECTION)

        # Remove duplicates
        return list(set(bias_types))

    def _identify_affected_groups(self, metrics: Dict[str, float],
                                 sensitive_features: pd.DataFrame) -> List[str]:
        """Identify groups affected by bias"""
        affected = []

        for feature in sensitive_features.columns:
            if f'{feature}_dp_difference' in metrics:
                if abs(metrics[f'{feature}_dp_difference']) > self.thresholds['demographic_parity']:
                    affected.append(feature)

        return affected

    def _generate_bias_recommendations(self, bias_types: List[BiasType],
                                      metrics: Dict[str, float]) -> List[str]:
        """Generate recommendations to address bias"""
        recommendations = []

        if BiasType.DEMOGRAPHIC in bias_types:
            recommendations.append("Apply demographic parity constraints during training")
            recommendations.append("Use reweighing to balance training data")

        if BiasType.SELECTION in bias_types:
            recommendations.append("Review and adjust decision thresholds")
            recommendations.append("Implement equalized odds post-processing")

        if any('disparate_impact' in k and v < 0.8 for k, v in metrics.items()):
            recommendations.append("Consider adversarial debiasing techniques")

        recommendations.append("Collect more diverse training data")
        recommendations.append("Implement continuous bias monitoring")

        return recommendations

    def _calculate_fairness_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall fairness score (0-100)"""
        score = 100.0

        # Penalize for demographic parity violations
        for key, value in metrics.items():
            if 'dp_difference' in key:
                score -= abs(value) * 100

            if 'disparate_impact' in key:
                score -= max(0, (0.8 - value)) * 50

        return max(0, score)

    async def mitigate_bias(self, model, X: np.ndarray, y: np.ndarray,
                           sensitive_features: pd.DataFrame) -> Any:
        """Apply bias mitigation techniques"""
        logger.info("Applying bias mitigation...")

        # Reweighing technique
        reweighing = Reweighing()

        # Create AIF360 dataset
        dataset = self._create_aif360_dataset(X, y, sensitive_features)

        # Apply reweighing
        dataset_transformed = reweighing.fit_transform(dataset)

        # Retrain model with reweighted data
        X_transformed = dataset_transformed.features
        y_transformed = dataset_transformed.labels.ravel()
        sample_weights = dataset_transformed.instance_weights

        # Return transformed data for retraining
        return X_transformed, y_transformed, sample_weights

    def _create_aif360_dataset(self, X: np.ndarray, y: np.ndarray,
                              sensitive_features: pd.DataFrame) -> BinaryLabelDataset:
        """Create AIF360 dataset for bias mitigation"""
        # Simplified - would need proper dataset creation
        df = pd.DataFrame(X)
        df['label'] = y

        for col in sensitive_features.columns:
            df[col] = sensitive_features[col]

        return BinaryLabelDataset(
            df=df,
            label_names=['label'],
            protected_attribute_names=list(sensitive_features.columns)
        )

class ModelExplainer:
    """Generate model explanations"""

    def __init__(self):
        self.explainers = {}
        self.explanation_cache = {}

    async def explain_model(self, model, X: np.ndarray,
                           feature_names: List[str],
                           method: ExplainabilityMethod = ExplainabilityMethod.SHAP) -> ExplainabilityReport:
        """Generate model explanations"""
        logger.info(f"Generating {method.value} explanations")

        global_explanations = {}
        sample_explanations = []
        feature_importance = {}

        if method == ExplainabilityMethod.SHAP:
            explanations = await self._shap_explain(model, X, feature_names)
            global_explanations = explanations['global']
            feature_importance = explanations['importance']
            sample_explanations = explanations['samples']

        elif method == ExplainabilityMethod.LIME:
            explanations = await self._lime_explain(model, X, feature_names)
            feature_importance = explanations['importance']
            sample_explanations = explanations['samples']

        elif method == ExplainabilityMethod.COUNTERFACTUAL:
            explanations = await self._counterfactual_explain(model, X, feature_names)
            sample_explanations = explanations['counterfactuals']

        # Calculate interpretability score
        interp_score = self._calculate_interpretability_score(
            feature_importance, sample_explanations
        )

        report = ExplainabilityReport(
            model_id=hashlib.md5(str(model).encode()).hexdigest(),
            global_explanations=global_explanations,
            feature_importance=feature_importance,
            sample_explanations=sample_explanations,
            explanation_methods=[method],
            interpretability_score=interp_score,
            report_date=datetime.now()
        )

        # Update metrics
        explainability_score.labels(model=model.__class__.__name__).set(interp_score)

        return report

    async def _shap_explain(self, model, X: np.ndarray,
                          feature_names: List[str]) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        # Create SHAP explainer
        if len(X) > 100:
            background = shap.sample(X, 100)
        else:
            background = X

        explainer = shap.KernelExplainer(model.predict, background)

        # Calculate SHAP values
        shap_values = explainer.shap_values(X[:10])  # Explain first 10 samples

        # Global feature importance
        importance = {}
        if isinstance(shap_values, list):
            mean_shap = np.abs(shap_values[0]).mean(axis=0)
        else:
            mean_shap = np.abs(shap_values).mean(axis=0)

        for i, name in enumerate(feature_names[:len(mean_shap)]):
            importance[name] = float(mean_shap[i])

        # Sample explanations
        samples = []
        for i in range(min(5, len(X))):
            sample_exp = {}
            if isinstance(shap_values, list):
                values = shap_values[0][i]
            else:
                values = shap_values[i]

            for j, name in enumerate(feature_names[:len(values)]):
                sample_exp[name] = float(values[j])

            samples.append(sample_exp)

        return {
            'global': {'base_value': float(explainer.expected_value) if not isinstance(explainer.expected_value, list) else float(explainer.expected_value[0])},
            'importance': importance,
            'samples': samples
        }

    async def _lime_explain(self, model, X: np.ndarray,
                          feature_names: List[str]) -> Dict[str, Any]:
        """Generate LIME explanations"""
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            X,
            feature_names=feature_names,
            mode='classification' if hasattr(model, 'predict_proba') else 'regression'
        )

        # Explain samples
        samples = []
        importance_scores = defaultdict(list)

        for i in range(min(5, len(X))):
            if hasattr(model, 'predict_proba'):
                exp = explainer.explain_instance(X[i], model.predict_proba, num_features=len(feature_names))
            else:
                exp = explainer.explain_instance(X[i], model.predict, num_features=len(feature_names))

            # Extract explanation
            sample_exp = {}
            for feat, val in exp.as_list():
                # Parse feature name from LIME format
                feat_name = feat.split(' ')[0] if ' ' in feat else feat
                sample_exp[feat_name] = val
                importance_scores[feat_name].append(abs(val))

            samples.append(sample_exp)

        # Average importance
        importance = {
            feat: np.mean(scores) for feat, scores in importance_scores.items()
        }

        return {
            'importance': importance,
            'samples': samples
        }

    async def _counterfactual_explain(self, model, X: np.ndarray,
                                     feature_names: List[str]) -> Dict[str, Any]:
        """Generate counterfactual explanations"""
        counterfactuals = []

        # Simplified counterfactual generation
        for i in range(min(5, len(X))):
            original = X[i]
            original_pred = model.predict([original])[0]

            # Find minimal change to flip prediction
            cf = original.copy()

            # Try changing each feature slightly
            for j in range(len(cf)):
                cf_temp = cf.copy()
                cf_temp[j] += np.std(X[:, j]) * 0.5

                new_pred = model.predict([cf_temp])[0]
                if new_pred != original_pred:
                    cf = cf_temp
                    break

            counterfactuals.append({
                'original': original.tolist(),
                'counterfactual': cf.tolist(),
                'changed_features': [feature_names[j] for j in range(len(original)) if original[j] != cf[j]]
            })

        return {'counterfactuals': counterfactuals}

    def _calculate_interpretability_score(self, feature_importance: Dict[str, float],
                                         sample_explanations: List[Dict]) -> float:
        """Calculate interpretability score (0-100)"""
        score = 0.0

        # Score based on feature importance concentration
        if feature_importance:
            values = list(feature_importance.values())
            # Higher score if importance is concentrated in fewer features
            top_5_importance = sum(sorted(values, reverse=True)[:5])
            total_importance = sum(values)
            if total_importance > 0:
                concentration = top_5_importance / total_importance
                score += concentration * 50

        # Score based on explanation consistency
        if sample_explanations:
            # Check if similar features are important across samples
            all_features = set()
            for exp in sample_explanations:
                all_features.update(exp.keys())

            if all_features:
                consistency = len(all_features) / (len(sample_explanations) * len(all_features))
                score += (1 - consistency) * 50

        return min(100, score)

class PrivacyGuard:
    """Ensure model privacy and data protection"""

    def __init__(self):
        self.privacy_engine = None
        self.epsilon = 1.0  # Differential privacy parameter
        self.delta = 1e-5

    async def apply_differential_privacy(self, model, optimizer,
                                        data_loader, epochs: int = 10) -> Dict[str, Any]:
        """Apply differential privacy to model training"""
        logger.info("Applying differential privacy...")

        # Initialize Opacus privacy engine
        self.privacy_engine = PrivacyEngine()

        # Make model private
        model, optimizer, data_loader = self.privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0
        )

        # Track privacy budget
        privacy_spent = []

        for epoch in range(epochs):
            # Training loop would go here
            epsilon = self.privacy_engine.get_epsilon(delta=self.delta)
            privacy_spent.append(epsilon)

            logger.info(f"Epoch {epoch}: ε = {epsilon:.2f}")

        return {
            'final_epsilon': privacy_spent[-1] if privacy_spent else 0,
            'delta': self.delta,
            'privacy_guarantee': f'(ε={privacy_spent[-1]:.2f}, δ={self.delta})'
        }

    async def anonymize_data(self, data: pd.DataFrame,
                           sensitive_columns: List[str]) -> pd.DataFrame:
        """Anonymize sensitive data columns"""
        logger.info("Anonymizing sensitive data...")

        anonymized = data.copy()

        for column in sensitive_columns:
            if column in anonymized.columns:
                # Apply different techniques based on data type
                if anonymized[column].dtype == 'object':
                    # Hash string values
                    anonymized[column] = anonymized[column].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8]
                    )
                else:
                    # Add Laplace noise to numeric values
                    noise = np.random.laplace(0, 1/self.epsilon, len(anonymized))
                    anonymized[column] += noise

        return anonymized

    async def encrypt_model(self, model) -> bytes:
        """Encrypt model for secure storage"""
        # Generate encryption key
        key = Fernet.generate_key()
        cipher = Fernet(key)

        # Serialize model
        import pickle
        model_bytes = pickle.dumps(model)

        # Encrypt
        encrypted = cipher.encrypt(model_bytes)

        logger.info("Model encrypted successfully")
        return encrypted

class ComplianceChecker:
    """Check regulatory compliance"""

    def __init__(self):
        self.requirements = self._load_requirements()

    def _load_requirements(self) -> Dict[ComplianceFramework, List[str]]:
        """Load compliance requirements"""
        return {
            ComplianceFramework.EU_AI_ACT: [
                "Human oversight capability",
                "Transparency and explainability",
                "Data governance and quality",
                "Accuracy and robustness testing",
                "Cybersecurity measures",
                "Bias detection and mitigation"
            ],
            ComplianceFramework.GDPR: [
                "Right to explanation",
                "Data minimization",
                "Privacy by design",
                "Consent management",
                "Data portability"
            ],
            ComplianceFramework.ISO_23053: [
                "Risk assessment",
                "Performance metrics",
                "Documentation requirements",
                "Testing procedures",
                "Monitoring capabilities"
            ]
        }

    async def check_compliance(self, model_card: ModelCard,
                              frameworks: List[ComplianceFramework]) -> ComplianceReport:
        """Check model compliance with frameworks"""
        logger.info(f"Checking compliance for model {model_card.model_id}")

        violations = []
        requirements_met = {}

        for framework in frameworks:
            framework_requirements = self.requirements.get(framework, [])

            for requirement in framework_requirements:
                met = self._check_requirement(model_card, requirement)
                requirements_met[f"{framework.value}_{requirement}"] = met

                if not met:
                    violations.append(f"{framework.value}: {requirement} not met")

        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(violations)

        report = ComplianceReport(
            model_id=model_card.model_id,
            frameworks=frameworks,
            compliant=len(violations) == 0,
            violations=violations,
            requirements_met=requirements_met,
            recommendations=recommendations,
            certification_status="Compliant" if len(violations) == 0 else "Non-compliant",
            report_date=datetime.now()
        )

        # Update metrics
        compliance_checks.inc()

        return report

    def _check_requirement(self, model_card: ModelCard, requirement: str) -> bool:
        """Check if specific requirement is met"""
        requirement_lower = requirement.lower()

        if "explainability" in requirement_lower:
            return len(model_card.explainability_methods) > 0

        if "bias" in requirement_lower:
            return 'bias_score' in model_card.fairness_metrics

        if "privacy" in requirement_lower:
            return len(model_card.privacy_measures) > 0

        if "documentation" in requirement_lower:
            return bool(model_card.description and model_card.intended_use)

        if "testing" in requirement_lower:
            return len(model_card.evaluation_metrics) > 0

        # Default to True for demo
        return True

    def _generate_compliance_recommendations(self, violations: List[str]) -> List[str]:
        """Generate recommendations for compliance"""
        recommendations = []

        for violation in violations:
            if "explainability" in violation.lower():
                recommendations.append("Implement SHAP or LIME explanations")

            if "bias" in violation.lower():
                recommendations.append("Conduct bias assessment using fairness metrics")

            if "privacy" in violation.lower():
                recommendations.append("Apply differential privacy or data anonymization")

            if "documentation" in violation.lower():
                recommendations.append("Complete model card documentation")

        recommendations.append("Schedule regular compliance audits")
        recommendations.append("Implement continuous monitoring")

        return recommendations

class ModelVersionControl:
    """Manage model versions and lineage"""

    def __init__(self):
        self.mlflow_client = MlflowClient()
        self.model_registry = {}

    async def register_model(self, model, model_card: ModelCard) -> str:
        """Register model with version control"""
        logger.info(f"Registering model {model_card.model_name} version {model_card.version}")

        # Start MLflow run
        with mlflow.start_run():
            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Log metrics
            for metric_name, value in model_card.evaluation_metrics.items():
                mlflow.log_metric(metric_name, value)

            # Log parameters
            mlflow.log_param("model_name", model_card.model_name)
            mlflow.log_param("version", model_card.version)

            # Log model card as artifact
            with open(f"{model_card.model_id}_card.json", 'w') as f:
                json.dump({
                    'model_name': model_card.model_name,
                    'version': model_card.version,
                    'description': model_card.description,
                    'intended_use': model_card.intended_use,
                    'limitations': model_card.limitations,
                    'ethical_considerations': model_card.ethical_considerations
                }, f)

            mlflow.log_artifact(f"{model_card.model_id}_card.json")

            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mlflow.register_model(model_uri, model_card.model_name)

        # Store in registry
        self.model_registry[model_card.model_id] = {
            'card': model_card,
            'uri': model_uri,
            'registered_date': datetime.now()
        }

        return model_uri

    async def track_lineage(self, model_id: str, parent_models: List[str],
                           training_data: str, transformations: List[str]) -> Dict[str, Any]:
        """Track model lineage"""
        lineage = {
            'model_id': model_id,
            'parent_models': parent_models,
            'training_data': training_data,
            'transformations': transformations,
            'timestamp': datetime.now().isoformat()
        }

        # Store lineage in MLflow
        with mlflow.start_run():
            mlflow.log_dict(lineage, "lineage.json")

        return lineage

class DriftMonitor:
    """Monitor model and data drift"""

    def __init__(self):
        self.baseline_data = None
        self.drift_thresholds = {
            'data_drift': 0.1,
            'prediction_drift': 0.05,
            'performance_drift': 0.1
        }

    async def detect_drift(self, current_data: np.ndarray,
                          baseline_data: np.ndarray) -> Dict[str, Any]:
        """Detect data and model drift"""
        logger.info("Detecting drift...")

        # Calculate drift metrics
        drift_metrics = {}

        # KS test for each feature
        for i in range(current_data.shape[1]):
            from scipy.stats import ks_2samp
            statistic, p_value = ks_2samp(baseline_data[:, i], current_data[:, i])
            drift_metrics[f'feature_{i}_ks'] = statistic
            drift_metrics[f'feature_{i}_pvalue'] = p_value

        # Overall drift score
        drift_score = np.mean([v for k, v in drift_metrics.items() if '_ks' in k])

        # Check if drift detected
        is_drift = drift_score > self.drift_thresholds['data_drift']

        if is_drift:
            drift_detected.inc()
            logger.warning(f"Data drift detected: score = {drift_score:.3f}")

        return {
            'drift_detected': is_drift,
            'drift_score': drift_score,
            'feature_drifts': drift_metrics,
            'recommendation': "Retrain model" if is_drift else "Continue monitoring"
        }

class AIGovernanceFramework:
    """
    Main AI Governance and Ethics Framework
    Ensures responsible AI with comprehensive oversight
    """

    def __init__(self):
        self.bias_detector = BiasDetector()
        self.explainer = ModelExplainer()
        self.privacy_guard = PrivacyGuard()
        self.compliance_checker = ComplianceChecker()
        self.version_control = ModelVersionControl()
        self.drift_monitor = DriftMonitor()
        self.governance_history = deque(maxlen=1000)

        logger.info("AI Governance Framework initialized")

    async def evaluate_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            sensitive_features: pd.DataFrame,
                            model_card: ModelCard) -> Dict[str, Any]:
        """
        Comprehensive model evaluation for governance

        Args:
            model: Trained model
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            sensitive_features: Sensitive attributes
            model_card: Model documentation

        Returns:
            Comprehensive governance evaluation
        """
        logger.info(f"Evaluating model {model_card.model_id} for governance")
        evaluation_results = {}

        # 1. Bias Assessment
        bias_assessment = await self.bias_detector.assess_bias(
            model, X_test, y_test, sensitive_features
        )
        evaluation_results['bias_assessment'] = bias_assessment

        # 2. Explainability Analysis
        explainability_report = await self.explainer.explain_model(
            model, X_test,
            feature_names=[f'feature_{i}' for i in range(X_test.shape[1])],
            method=ExplainabilityMethod.SHAP
        )
        evaluation_results['explainability'] = explainability_report

        # 3. Privacy Assessment
        privacy_metrics = {
            'differential_privacy_applied': PrivacyTechnique.DIFFERENTIAL_PRIVACY in model_card.privacy_measures,
            'data_anonymized': PrivacyTechnique.DATA_ANONYMIZATION in model_card.privacy_measures,
            'encryption_enabled': True  # Assume encryption is enabled
        }
        evaluation_results['privacy'] = privacy_metrics

        # 4. Compliance Check
        compliance_report = await self.compliance_checker.check_compliance(
            model_card,
            [ComplianceFramework.EU_AI_ACT, ComplianceFramework.GDPR]
        )
        evaluation_results['compliance'] = compliance_report

        # 5. Drift Detection
        if len(X_train) > 100:
            drift_results = await self.drift_monitor.detect_drift(
                X_test, X_train[:100]
            )
            evaluation_results['drift'] = drift_results

        # 6. Model Registration
        model_uri = await self.version_control.register_model(model, model_card)
        evaluation_results['model_uri'] = model_uri

        # Calculate overall governance score
        governance_score_value = self._calculate_governance_score(evaluation_results)
        evaluation_results['governance_score'] = governance_score_value

        # Update metrics
        models_evaluated.inc()
        governance_score.set(governance_score_value)

        # Store in history
        self.governance_history.append({
            'model_id': model_card.model_id,
            'timestamp': datetime.now(),
            'results': evaluation_results
        })

        return evaluation_results

    def _calculate_governance_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall governance score (0-100)"""
        score = 0.0
        weights = {
            'bias': 25,
            'explainability': 25,
            'privacy': 20,
            'compliance': 20,
            'drift': 10
        }

        # Bias score
        if 'bias_assessment' in results:
            bias_score = 100 if not results['bias_assessment'].bias_detected else 50
            score += bias_score * weights['bias'] / 100

        # Explainability score
        if 'explainability' in results:
            score += results['explainability'].interpretability_score * weights['explainability'] / 100

        # Privacy score
        if 'privacy' in results:
            privacy_score = sum(results['privacy'].values()) / len(results['privacy']) * 100
            score += privacy_score * weights['privacy'] / 100

        # Compliance score
        if 'compliance' in results:
            compliance_score = 100 if results['compliance'].compliant else 50
            score += compliance_score * weights['compliance'] / 100

        # Drift score
        if 'drift' in results:
            drift_score = 100 if not results['drift'].get('drift_detected', False) else 50
            score += drift_score * weights['drift'] / 100

        return min(100, score)

    async def generate_governance_report(self, model_id: str) -> str:
        """Generate comprehensive governance report"""
        # Find evaluation in history
        evaluation = None
        for entry in self.governance_history:
            if entry['model_id'] == model_id:
                evaluation = entry
                break

        if not evaluation:
            return "No evaluation found for model"

        report = f"""
AI Governance Report
====================
Model ID: {model_id}
Date: {evaluation['timestamp']}

Governance Score: {evaluation['results'].get('governance_score', 0):.1f}/100

Bias Assessment
---------------
Bias Detected: {evaluation['results']['bias_assessment'].bias_detected}
Affected Groups: {', '.join(evaluation['results']['bias_assessment'].affected_groups)}

Explainability
--------------
Interpretability Score: {evaluation['results']['explainability'].interpretability_score:.1f}
Methods Used: {', '.join([m.value for m in evaluation['results']['explainability'].explanation_methods])}

Privacy
-------
Differential Privacy: {evaluation['results']['privacy']['differential_privacy_applied']}
Data Anonymization: {evaluation['results']['privacy']['data_anonymized']}

Compliance
----------
Compliant: {evaluation['results']['compliance'].compliant}
Frameworks: {', '.join([f.value for f in evaluation['results']['compliance'].frameworks])}

Recommendations
---------------
"""
        # Add all recommendations
        all_recommendations = set()

        if 'bias_assessment' in evaluation['results']:
            all_recommendations.update(evaluation['results']['bias_assessment'].recommendations)

        if 'compliance' in evaluation['results']:
            all_recommendations.update(evaluation['results']['compliance'].recommendations)

        for rec in all_recommendations:
            report += f"- {rec}\n"

        return report

# Example usage
async def test_governance_framework():
    """Test the AI governance framework"""

    # Create framework
    governance = AIGovernanceFramework()

    # Create sample model and data
    from sklearn.ensemble import RandomForestClassifier

    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.choice([0, 1], 1000)
    X_test = np.random.randn(200, 20)
    y_test = np.random.choice([0, 1], 200)

    # Sensitive features
    sensitive_features = pd.DataFrame({
        'gender': np.random.choice(['M', 'F'], 200),
        'age_group': np.random.choice(['young', 'middle', 'senior'], 200)
    })

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Create model card
    model_card = ModelCard(
        model_id="test_model_001",
        model_name="Risk Predictor",
        version="1.0.0",
        description="Predicts customer risk level",
        intended_use="Credit risk assessment",
        training_data={'samples': 1000, 'features': 20},
        evaluation_metrics={'accuracy': 0.85, 'auc': 0.82},
        limitations=["Limited to structured data", "Requires minimum 1000 samples"],
        ethical_considerations=["Ensure fair treatment across demographics"],
        fairness_metrics={'demographic_parity': 0.05},
        explainability_methods=[ExplainabilityMethod.SHAP, ExplainabilityMethod.LIME],
        privacy_measures=[PrivacyTechnique.DIFFERENTIAL_PRIVACY],
        created_date=datetime.now(),
        last_updated=datetime.now(),
        owner="AI Team",
        contact="ai@company.com"
    )

    # Evaluate model
    evaluation = await governance.evaluate_model(
        model, X_train, y_train, X_test, y_test,
        sensitive_features, model_card
    )

    print("Governance Evaluation Results:")
    print(f"Governance Score: {evaluation['governance_score']:.1f}/100")
    print(f"Bias Detected: {evaluation['bias_assessment'].bias_detected}")
    print(f"Compliance: {evaluation['compliance'].compliant}")
    print(f"Interpretability: {evaluation['explainability'].interpretability_score:.1f}")

    # Generate report
    report = await governance.generate_governance_report("test_model_001")
    print("\nGovernance Report:")
    print(report)

    return governance

if __name__ == "__main__":
    # Run test
    asyncio.run(test_governance_framework())