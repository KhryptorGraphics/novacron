"""
ML Governance & Compliance framework for bias detection, explainability enforcement,
data lineage, and regulatory compliance (GDPR, CCPA, AI Act).
"""

import asyncio
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Regulatory compliance frameworks"""
    GDPR = "gdpr"  # General Data Protection Regulation (EU)
    CCPA = "ccpa"  # California Consumer Privacy Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    AI_ACT = "ai_act"  # EU AI Act
    SOC2 = "soc2"  # Service Organization Control 2
    ISO27001 = "iso27001"


class RiskLevel(Enum):
    """AI system risk levels (EU AI Act)"""
    UNACCEPTABLE = "unacceptable"
    HIGH = "high"
    LIMITED = "limited"
    MINIMAL = "minimal"


class BiasType(Enum):
    """Types of bias to detect"""
    STATISTICAL_PARITY = "statistical_parity"
    EQUAL_OPPORTUNITY = "equal_opportunity"
    EQUALIZED_ODDS = "equalized_odds"
    DEMOGRAPHIC_PARITY = "demographic_parity"
    DISPARATE_IMPACT = "disparate_impact"


@dataclass
class DataLineage:
    """Track data provenance and transformations"""
    lineage_id: str
    dataset_id: str
    dataset_name: str
    source: str
    created_at: datetime

    # Transformations applied
    transformations: List[Dict[str, Any]] = field(default_factory=list)

    # Parent datasets
    parent_datasets: List[str] = field(default_factory=list)

    # Models trained on this data
    trained_models: List[str] = field(default_factory=list)

    # Access log
    access_log: List[Dict[str, Any]] = field(default_factory=list)

    # Data quality
    quality_score: float = 0.0
    validation_results: Dict[str, Any] = field(default_factory=dict)

    # Compliance
    contains_pii: bool = False
    data_retention_days: int = 365
    anonymized: bool = False


@dataclass
class ModelLineage:
    """Track model provenance and lifecycle"""
    lineage_id: str
    model_id: str
    model_version: str
    created_at: datetime

    # Training information
    training_dataset: str
    training_algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_duration_seconds: float = 0.0

    # Dependencies
    parent_model_id: Optional[str] = None
    dependent_features: List[str] = field(default_factory=list)

    # Deployment history
    deployments: List[Dict[str, Any]] = field(default_factory=list)

    # Performance history
    performance_metrics: List[Dict[str, Any]] = field(default_factory=list)

    # Approvals and reviews
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    audits: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class BiasReport:
    """Bias analysis report"""
    report_id: str
    model_id: str
    protected_attribute: str
    bias_type: BiasType
    bias_score: float
    threshold: float
    is_biased: bool
    mitigation_recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    model_id: str
    framework: ComplianceFramework
    risk_level: RiskLevel
    compliance_score: float  # 0-100
    requirements_met: List[str] = field(default_factory=list)
    requirements_failed: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    auditor: str = ""


class BiasDete ctor:
    """Detect and measure algorithmic bias"""

    def __init__(self, protected_attributes: List[str]):
        self.protected_attributes = protected_attributes

    def detect_statistical_parity(
        self,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        threshold: float = 0.1
    ) -> BiasReport:
        """
        Statistical parity: P(Y_pred=1|A=0) ≈ P(Y_pred=1|A=1)
        """
        groups = np.unique(protected_attr)

        if len(groups) < 2:
            return self._create_report("statistical_parity", protected_attr, 0.0, threshold, False)

        # Calculate positive prediction rates for each group
        rates = {}
        for group in groups:
            mask = protected_attr == group
            rates[group] = np.mean(y_pred[mask])

        # Calculate maximum difference
        max_diff = max(rates.values()) - min(rates.values())
        is_biased = max_diff > threshold

        recommendations = []
        if is_biased:
            recommendations.append("Apply reweighting to balance prediction rates across groups")
            recommendations.append("Consider using fairness-aware learning algorithms")
            recommendations.append("Collect more balanced training data")

        return self._create_report(
            "statistical_parity",
            protected_attr,
            max_diff,
            threshold,
            is_biased,
            recommendations,
            metadata={"group_rates": {str(k): float(v) for k, v in rates.items()}}
        )

    def detect_equal_opportunity(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        threshold: float = 0.1
    ) -> BiasReport:
        """
        Equal opportunity: TPR should be similar across groups
        """
        groups = np.unique(protected_attr)

        if len(groups) < 2:
            return self._create_report("equal_opportunity", protected_attr, 0.0, threshold, False)

        # Calculate TPR for each group
        tpr_rates = {}
        for group in groups:
            mask = protected_attr == group
            positives = y_true[mask] == 1

            if np.sum(positives) == 0:
                tpr_rates[group] = 0.0
            else:
                true_positives = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1))
                tpr_rates[group] = true_positives / np.sum(positives)

        # Calculate maximum difference
        max_diff = max(tpr_rates.values()) - min(tpr_rates.values())
        is_biased = max_diff > threshold

        recommendations = []
        if is_biased:
            recommendations.append("Apply post-processing calibration to equalize TPR across groups")
            recommendations.append("Use threshold optimization per group")
            recommendations.append("Balance positive examples in training data")

        return self._create_report(
            "equal_opportunity",
            protected_attr,
            max_diff,
            threshold,
            is_biased,
            recommendations,
            metadata={"group_tpr": {str(k): float(v) for k, v in tpr_rates.items()}}
        )

    def detect_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        threshold: float = 0.1
    ) -> BiasReport:
        """
        Equalized odds: Both TPR and FPR should be similar across groups
        """
        groups = np.unique(protected_attr)

        if len(groups) < 2:
            return self._create_report("equalized_odds", protected_attr, 0.0, threshold, False)

        # Calculate TPR and FPR for each group
        metrics = {}
        for group in groups:
            mask = protected_attr == group

            positives = y_true[mask] == 1
            negatives = y_true[mask] == 0

            if np.sum(positives) > 0:
                tpr = np.sum((y_true[mask] == 1) & (y_pred[mask] == 1)) / np.sum(positives)
            else:
                tpr = 0.0

            if np.sum(negatives) > 0:
                fpr = np.sum((y_true[mask] == 0) & (y_pred[mask] == 1)) / np.sum(negatives)
            else:
                fpr = 0.0

            metrics[group] = {"tpr": tpr, "fpr": fpr}

        # Calculate maximum differences
        tpr_diff = max(m["tpr"] for m in metrics.values()) - min(m["tpr"] for m in metrics.values())
        fpr_diff = max(m["fpr"] for m in metrics.values()) - min(m["fpr"] for m in metrics.values())
        max_diff = max(tpr_diff, fpr_diff)

        is_biased = max_diff > threshold

        recommendations = []
        if is_biased:
            recommendations.append("Apply equalized odds post-processing")
            recommendations.append("Use fairness constraints during training")
            recommendations.append("Balance both positive and negative examples across groups")

        return self._create_report(
            "equalized_odds",
            protected_attr,
            max_diff,
            threshold,
            is_biased,
            recommendations,
            metadata={"group_metrics": {str(k): v for k, v in metrics.items()}}
        )

    def detect_disparate_impact(
        self,
        y_pred: np.ndarray,
        protected_attr: np.ndarray,
        threshold: float = 0.8  # 80% rule
    ) -> BiasReport:
        """
        Disparate impact: ratio of selection rates should be >= 80%
        """
        groups = np.unique(protected_attr)

        if len(groups) < 2:
            return self._create_report("disparate_impact", protected_attr, 1.0, threshold, False)

        # Calculate selection rates
        rates = {}
        for group in groups:
            mask = protected_attr == group
            rates[group] = np.mean(y_pred[mask])

        # Calculate disparate impact ratio
        min_rate = min(rates.values())
        max_rate = max(rates.values())

        if max_rate == 0:
            ratio = 1.0
        else:
            ratio = min_rate / max_rate

        is_biased = ratio < threshold

        recommendations = []
        if is_biased:
            recommendations.append(f"Disparate impact ratio is {ratio:.2f}, below 80% threshold")
            recommendations.append("Review feature selection for potential bias sources")
            recommendations.append("Consider applying fairness-aware preprocessing")

        return self._create_report(
            "disparate_impact",
            protected_attr,
            ratio,
            threshold,
            is_biased,
            recommendations,
            metadata={"group_rates": {str(k): float(v) for k, v in rates.items()}, "ratio": ratio}
        )

    def _create_report(
        self,
        bias_type: str,
        protected_attr: np.ndarray,
        score: float,
        threshold: float,
        is_biased: bool,
        recommendations: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> BiasReport:
        """Create bias report"""
        attr_name = f"protected_attr_{hash(str(protected_attr[:10]))}"

        return BiasReport(
            report_id=f"bias_{bias_type}_{int(datetime.now().timestamp())}",
            model_id="",  # Set by caller
            protected_attribute=attr_name,
            bias_type=BiasType[bias_type.upper()],
            bias_score=float(score),
            threshold=threshold,
            is_biased=is_biased,
            mitigation_recommendations=recommendations or [],
            metadata=metadata or {}
        )


class ComplianceChecker:
    """Check compliance with regulatory frameworks"""

    def __init__(self):
        self.requirements = self._load_requirements()

    def _load_requirements(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Load compliance requirements"""
        return {
            ComplianceFramework.GDPR: {
                "data_minimization": "Collect only necessary data",
                "right_to_explanation": "Provide explanations for automated decisions",
                "data_retention": "Define and enforce retention policies",
                "consent_management": "Obtain and track user consent",
                "data_portability": "Enable data export",
                "right_to_erasure": "Enable data deletion",
            },
            ComplianceFramework.AI_ACT: {
                "risk_assessment": "Conduct AI risk assessment",
                "human_oversight": "Ensure human oversight capability",
                "transparency": "Provide transparency documentation",
                "accuracy_requirements": "Meet accuracy requirements",
                "robustness": "Ensure model robustness",
                "bias_mitigation": "Implement bias detection and mitigation",
            },
            ComplianceFramework.CCPA: {
                "data_disclosure": "Disclose data collection practices",
                "opt_out_rights": "Enable opt-out of data sales",
                "data_deletion": "Enable data deletion requests",
                "non_discrimination": "No discrimination for privacy rights exercise",
            },
        }

    async def assess_compliance(
        self,
        model_id: str,
        framework: ComplianceFramework,
        model_metadata: Dict[str, Any],
        checks: Dict[str, bool]
    ) -> ComplianceReport:
        """Assess compliance with framework"""

        requirements = self.requirements.get(framework, {})
        met = []
        failed = []

        for req_id, req_desc in requirements.items():
            if checks.get(req_id, False):
                met.append(req_desc)
            else:
                failed.append(req_desc)

        compliance_score = (len(met) / len(requirements)) * 100 if requirements else 100.0

        # Determine risk level
        risk_level = self._assess_risk_level(model_metadata, compliance_score)

        # Generate recommendations
        recommendations = []
        for req_desc in failed:
            recommendations.append(f"Implement: {req_desc}")

        return ComplianceReport(
            report_id=f"compliance_{framework.value}_{int(datetime.now().timestamp())}",
            model_id=model_id,
            framework=framework,
            risk_level=risk_level,
            compliance_score=compliance_score,
            requirements_met=met,
            requirements_failed=failed,
            recommendations=recommendations,
        )

    def _assess_risk_level(self, model_metadata: Dict[str, Any], compliance_score: float) -> RiskLevel:
        """Assess AI system risk level per EU AI Act"""

        # High-risk indicators
        use_case = model_metadata.get("use_case", "").lower()
        domain = model_metadata.get("domain", "").lower()

        high_risk_domains = ["healthcare", "finance", "law enforcement", "employment", "education"]
        high_risk_uses = ["credit scoring", "hiring", "medical diagnosis", "criminal justice"]

        if compliance_score < 50:
            return RiskLevel.HIGH

        for keyword in high_risk_domains + high_risk_uses:
            if keyword in use_case or keyword in domain:
                return RiskLevel.HIGH

        if compliance_score < 75:
            return RiskLevel.LIMITED

        return RiskLevel.MINIMAL


class GovernanceManager:
    """Central ML governance and compliance manager"""

    def __init__(self, storage_path: str = "./governance"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Lineage tracking
        self.data_lineage = {}
        self.model_lineage = {}

        # Bias and compliance
        self.bias_detector = None
        self.compliance_checker = ComplianceChecker()

        # Reports
        self.bias_reports = []
        self.compliance_reports = []
        self.audit_log = []

    def register_dataset(
        self,
        dataset_id: str,
        dataset_name: str,
        source: str,
        contains_pii: bool = False,
        parent_datasets: List[str] = None
    ) -> DataLineage:
        """Register a dataset in lineage system"""

        lineage = DataLineage(
            lineage_id=f"data_{dataset_id}_{int(datetime.now().timestamp())}",
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            source=source,
            created_at=datetime.now(),
            contains_pii=contains_pii,
            parent_datasets=parent_datasets or [],
        )

        self.data_lineage[dataset_id] = lineage
        self._audit("dataset_registered", {"dataset_id": dataset_id})

        return lineage

    def register_model(
        self,
        model_id: str,
        model_version: str,
        training_dataset: str,
        training_algorithm: str,
        hyperparameters: Dict[str, Any]
    ) -> ModelLineage:
        """Register a model in lineage system"""

        lineage = ModelLineage(
            lineage_id=f"model_{model_id}_{int(datetime.now().timestamp())}",
            model_id=model_id,
            model_version=model_version,
            created_at=datetime.now(),
            training_dataset=training_dataset,
            training_algorithm=training_algorithm,
            hyperparameters=hyperparameters,
        )

        self.model_lineage[model_id] = lineage

        # Link to dataset
        if training_dataset in self.data_lineage:
            self.data_lineage[training_dataset].trained_models.append(model_id)

        self._audit("model_registered", {"model_id": model_id, "version": model_version})

        return lineage

    async def assess_bias(
        self,
        model_id: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected_attributes: Dict[str, np.ndarray]
    ) -> List[BiasReport]:
        """Comprehensive bias assessment"""

        if not protected_attributes:
            logger.warning("No protected attributes provided for bias assessment")
            return []

        self.bias_detector = BiasDetector(list(protected_attributes.keys()))
        reports = []

        for attr_name, attr_values in protected_attributes.items():
            # Run all bias detection methods
            reports.append(self.bias_detector.detect_statistical_parity(y_pred, attr_values))
            reports.append(self.bias_detector.detect_equal_opportunity(y_true, y_pred, attr_values))
            reports.append(self.bias_detector.detect_equalized_odds(y_true, y_pred, attr_values))
            reports.append(self.bias_detector.detect_disparate_impact(y_pred, attr_values))

        # Set model_id for all reports
        for report in reports:
            report.model_id = model_id

        self.bias_reports.extend(reports)
        self._audit("bias_assessment", {"model_id": model_id, "reports_generated": len(reports)})

        return reports

    async def assess_compliance(
        self,
        model_id: str,
        framework: ComplianceFramework,
        model_metadata: Dict[str, Any],
        compliance_checks: Dict[str, bool]
    ) -> ComplianceReport:
        """Assess regulatory compliance"""

        report = await self.compliance_checker.assess_compliance(
            model_id,
            framework,
            model_metadata,
            compliance_checks
        )

        self.compliance_reports.append(report)
        self._audit("compliance_assessment", {
            "model_id": model_id,
            "framework": framework.value,
            "score": report.compliance_score
        })

        return report

    def get_data_lineage(self, dataset_id: str) -> Optional[DataLineage]:
        """Get complete data lineage"""
        return self.data_lineage.get(dataset_id)

    def get_model_lineage(self, model_id: str) -> Optional[ModelLineage]:
        """Get complete model lineage"""
        return self.model_lineage.get(model_id)

    def get_full_lineage_tree(self, model_id: str) -> Dict[str, Any]:
        """Get full lineage tree (model + data)"""

        model_lin = self.model_lineage.get(model_id)
        if not model_lin:
            return {}

        data_lin = self.data_lineage.get(model_lin.training_dataset)

        return {
            "model": {
                "id": model_lin.model_id,
                "version": model_lin.model_version,
                "created": model_lin.created_at.isoformat(),
                "algorithm": model_lin.training_algorithm,
            },
            "training_data": {
                "id": data_lin.dataset_id if data_lin else None,
                "name": data_lin.dataset_name if data_lin else None,
                "source": data_lin.source if data_lin else None,
                "contains_pii": data_lin.contains_pii if data_lin else None,
            },
            "parent_datasets": data_lin.parent_datasets if data_lin else [],
            "deployments": model_lin.deployments,
        }

    def get_bias_summary(self, model_id: str) -> Dict[str, Any]:
        """Get bias assessment summary for model"""

        model_reports = [r for r in self.bias_reports if r.model_id == model_id]

        if not model_reports:
            return {"model_id": model_id, "assessments": 0, "biased": 0}

        biased_reports = [r for r in model_reports if r.is_biased]

        return {
            "model_id": model_id,
            "total_assessments": len(model_reports),
            "biased_assessments": len(biased_reports),
            "bias_types": list(set(r.bias_type.value for r in biased_reports)),
            "max_bias_score": max(r.bias_score for r in model_reports),
            "recommendations": list(set(
                rec for r in biased_reports for rec in r.mitigation_recommendations
            ))[:5],
        }

    def get_compliance_summary(self, model_id: str) -> Dict[str, Any]:
        """Get compliance summary for model"""

        model_reports = [r for r in self.compliance_reports if r.model_id == model_id]

        if not model_reports:
            return {"model_id": model_id, "assessments": 0}

        return {
            "model_id": model_id,
            "frameworks_assessed": [r.framework.value for r in model_reports],
            "compliance_scores": {r.framework.value: r.compliance_score for r in model_reports},
            "risk_level": model_reports[0].risk_level.value,
            "overall_compliance": sum(r.compliance_score for r in model_reports) / len(model_reports),
        }

    def _audit(self, event_type: str, details: Dict[str, Any]):
        """Log governance event"""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
        })

    async def export_governance_report(self, model_id: str, output_path: str):
        """Export comprehensive governance report"""

        report = {
            "model_id": model_id,
            "generated_at": datetime.now().isoformat(),
            "lineage": self.get_full_lineage_tree(model_id),
            "bias_assessment": self.get_bias_summary(model_id),
            "compliance": self.get_compliance_summary(model_id),
            "audit_trail": [
                event for event in self.audit_log
                if model_id in str(event.get("details", {}))
            ],
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Governance report exported to {output_path}")


# Example usage
async def example_governance():
    """Example governance workflow"""

    gov = GovernanceManager()

    # Register dataset
    dataset_lineage = gov.register_dataset(
        dataset_id="fraud_data_v1",
        dataset_name="Fraud Detection Dataset",
        source="internal_transactions",
        contains_pii=True
    )

    # Register model
    model_lineage = gov.register_model(
        model_id="fraud_model_v1",
        model_version="1.0.0",
        training_dataset="fraud_data_v1",
        training_algorithm="RandomForest",
        hyperparameters={"n_estimators": 100, "max_depth": 10}
    )

    # Assess bias
    y_true = np.random.randint(0, 2, 1000)
    y_pred = np.random.randint(0, 2, 1000)
    protected_attrs = {"age_group": np.random.randint(0, 3, 1000)}

    bias_reports = await gov.assess_bias("fraud_model_v1", y_true, y_pred, protected_attrs)
    print(f"Bias assessments: {len(bias_reports)}, Biased: {sum(r.is_biased for r in bias_reports)}")

    # Assess compliance
    compliance_report = await gov.assess_compliance(
        model_id="fraud_model_v1",
        framework=ComplianceFramework.GDPR,
        model_metadata={"domain": "finance", "use_case": "fraud detection"},
        compliance_checks={
            "data_minimization": True,
            "right_to_explanation": True,
            "data_retention": True,
            "consent_management": False,
        }
    )
    print(f"GDPR compliance score: {compliance_report.compliance_score:.1f}%")

    # Export report
    await gov.export_governance_report("fraud_model_v1", "./governance_report.json")


if __name__ == "__main__":
    asyncio.run(example_governance())
