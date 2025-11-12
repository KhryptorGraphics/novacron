#!/usr/bin/env python3
"""
Configuration Drift Detection and Remediation
Detects configuration drift and automatically remediates with approval workflows
"""

import json
import logging
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class DriftSeverity(Enum):
    """Drift severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RemediationStrategy(Enum):
    """Remediation strategy types"""
    AUTO = "auto"
    MANUAL = "manual"
    APPROVAL_REQUIRED = "approval_required"
    NOTIFY_ONLY = "notify_only"


@dataclass
class ConfigurationBaseline:
    """Configuration baseline definition"""
    resource_id: str
    resource_type: str
    config: Dict[str, Any]
    config_hash: str
    timestamp: datetime
    tags: Dict[str, str]


@dataclass
class DriftDetection:
    """Detected configuration drift"""
    detection_id: str
    resource_id: str
    resource_type: str
    severity: DriftSeverity
    fields_changed: List[str]
    expected_values: Dict[str, Any]
    actual_values: Dict[str, Any]
    drift_percentage: float
    detected_at: datetime
    remediation_strategy: RemediationStrategy


@dataclass
class RemediationPlan:
    """Remediation plan"""
    plan_id: str
    drift_detection: DriftDetection
    actions: List[Dict[str, Any]]
    estimated_duration: int
    requires_approval: bool
    approval_status: Optional[str]
    approved_by: Optional[str]
    created_at: datetime


class DriftDetector:
    """
    Configuration drift detector with automatic remediation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Storage
        self.baselines: Dict[str, ConfigurationBaseline] = {}
        self.detections: List[DriftDetection] = []
        self.remediation_plans: Dict[str, RemediationPlan] = {}

        # Configuration
        self.scan_interval = self.config.get('scan_interval', 300)  # 5 minutes
        self.auto_remediate = self.config.get('auto_remediate', False)
        self.drift_threshold = self.config.get('drift_threshold', 0.05)  # 5%

        # Severity thresholds
        self.severity_thresholds = {
            DriftSeverity.CRITICAL: 0.50,  # 50% drift
            DriftSeverity.HIGH: 0.25,      # 25% drift
            DriftSeverity.MEDIUM: 0.10,    # 10% drift
            DriftSeverity.LOW: 0.05,       # 5% drift
        }

        self.logger.info("Drift detector initialized")

    def register_baseline(self, resource_id: str, resource_type: str,
                         config: Dict[str, Any], tags: Optional[Dict[str, str]] = None):
        """
        Register configuration baseline for a resource

        Args:
            resource_id: Unique resource identifier
            resource_type: Type of resource (vm, network, policy, etc.)
            config: Current configuration
            tags: Optional tags for categorization
        """

        config_hash = self._compute_config_hash(config)

        baseline = ConfigurationBaseline(
            resource_id=resource_id,
            resource_type=resource_type,
            config=config,
            config_hash=config_hash,
            timestamp=datetime.now(),
            tags=tags or {}
        )

        self.baselines[resource_id] = baseline

        self.logger.info(
            f"Registered baseline for {resource_type} {resource_id}"
        )

    def scan_for_drift(self, current_configs: Dict[str, Dict[str, Any]]) -> List[DriftDetection]:
        """
        Scan for configuration drift

        Args:
            current_configs: Current configurations keyed by resource_id

        Returns:
            List of detected drifts
        """

        detected_drifts = []

        for resource_id, current_config in current_configs.items():
            if resource_id not in self.baselines:
                self.logger.warning(f"No baseline found for {resource_id}")
                continue

            baseline = self.baselines[resource_id]

            # Detect drift
            drift = self._detect_drift(baseline, current_config)
            if drift:
                detected_drifts.append(drift)
                self.detections.append(drift)

        self.logger.info(f"Scan complete: {len(detected_drifts)} drifts detected")
        return detected_drifts

    def _detect_drift(self, baseline: ConfigurationBaseline,
                     current_config: Dict[str, Any]) -> Optional[DriftDetection]:
        """Detect drift between baseline and current configuration"""

        # Compute current hash
        current_hash = self._compute_config_hash(current_config)

        # No drift if hashes match
        if current_hash == baseline.config_hash:
            return None

        # Compare configurations
        fields_changed = []
        expected_values = {}
        actual_values = {}

        self._compare_configs(
            baseline.config,
            current_config,
            "",
            fields_changed,
            expected_values,
            actual_values
        )

        if not fields_changed:
            return None

        # Calculate drift percentage
        total_fields = len(self._flatten_config(baseline.config))
        drift_percentage = len(fields_changed) / max(1, total_fields)

        # Determine severity
        severity = self._calculate_severity(drift_percentage, fields_changed, baseline)

        # Determine remediation strategy
        remediation_strategy = self._determine_remediation_strategy(severity)

        detection = DriftDetection(
            detection_id=f"drift-{datetime.now().timestamp()}",
            resource_id=baseline.resource_id,
            resource_type=baseline.resource_type,
            severity=severity,
            fields_changed=fields_changed,
            expected_values=expected_values,
            actual_values=actual_values,
            drift_percentage=drift_percentage,
            detected_at=datetime.now(),
            remediation_strategy=remediation_strategy
        )

        self.logger.warning(
            f"Drift detected in {baseline.resource_type} {baseline.resource_id}: "
            f"{len(fields_changed)} fields changed ({drift_percentage*100:.1f}% drift)"
        )

        return detection

    def _compare_configs(self, expected: Any, actual: Any, prefix: str,
                        changed_fields: List[str], expected_values: Dict,
                        actual_values: Dict):
        """Recursively compare configurations"""

        if isinstance(expected, dict) and isinstance(actual, dict):
            all_keys = set(expected.keys()) | set(actual.keys())

            for key in all_keys:
                new_prefix = f"{prefix}.{key}" if prefix else key

                if key not in expected:
                    changed_fields.append(new_prefix)
                    actual_values[new_prefix] = actual[key]
                    expected_values[new_prefix] = None
                elif key not in actual:
                    changed_fields.append(new_prefix)
                    expected_values[new_prefix] = expected[key]
                    actual_values[new_prefix] = None
                elif expected[key] != actual[key]:
                    if isinstance(expected[key], (dict, list)):
                        self._compare_configs(
                            expected[key], actual[key], new_prefix,
                            changed_fields, expected_values, actual_values
                        )
                    else:
                        changed_fields.append(new_prefix)
                        expected_values[new_prefix] = expected[key]
                        actual_values[new_prefix] = actual[key]

        elif isinstance(expected, list) and isinstance(actual, list):
            if expected != actual:
                changed_fields.append(prefix)
                expected_values[prefix] = expected
                actual_values[prefix] = actual

        elif expected != actual:
            changed_fields.append(prefix)
            expected_values[prefix] = expected
            actual_values[prefix] = actual

    def _flatten_config(self, config: Any, prefix: str = "") -> List[str]:
        """Flatten configuration to list of field paths"""

        fields = []

        if isinstance(config, dict):
            for key, value in config.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    fields.extend(self._flatten_config(value, new_prefix))
                else:
                    fields.append(new_prefix)
        elif isinstance(config, list):
            for i, item in enumerate(config):
                new_prefix = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    fields.extend(self._flatten_config(item, new_prefix))
                else:
                    fields.append(new_prefix)
        else:
            fields.append(prefix)

        return fields

    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute hash of configuration"""

        config_json = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()

    def _calculate_severity(self, drift_percentage: float, fields_changed: List[str],
                           baseline: ConfigurationBaseline) -> DriftSeverity:
        """Calculate drift severity"""

        # Check for critical fields
        critical_fields = ['security', 'authentication', 'encryption', 'firewall']
        for field in fields_changed:
            for critical in critical_fields:
                if critical in field.lower():
                    return DriftSeverity.CRITICAL

        # Based on percentage thresholds
        if drift_percentage >= self.severity_thresholds[DriftSeverity.CRITICAL]:
            return DriftSeverity.CRITICAL
        elif drift_percentage >= self.severity_thresholds[DriftSeverity.HIGH]:
            return DriftSeverity.HIGH
        elif drift_percentage >= self.severity_thresholds[DriftSeverity.MEDIUM]:
            return DriftSeverity.MEDIUM
        elif drift_percentage >= self.severity_thresholds[DriftSeverity.LOW]:
            return DriftSeverity.LOW
        else:
            return DriftSeverity.INFO

    def _determine_remediation_strategy(self, severity: DriftSeverity) -> RemediationStrategy:
        """Determine remediation strategy based on severity"""

        if severity == DriftSeverity.CRITICAL:
            return RemediationStrategy.APPROVAL_REQUIRED
        elif severity == DriftSeverity.HIGH:
            return RemediationStrategy.APPROVAL_REQUIRED
        elif severity == DriftSeverity.MEDIUM:
            if self.auto_remediate:
                return RemediationStrategy.AUTO
            else:
                return RemediationStrategy.APPROVAL_REQUIRED
        elif severity == DriftSeverity.LOW:
            if self.auto_remediate:
                return RemediationStrategy.AUTO
            else:
                return RemediationStrategy.MANUAL
        else:
            return RemediationStrategy.NOTIFY_ONLY

    def create_remediation_plan(self, drift: DriftDetection) -> RemediationPlan:
        """
        Create remediation plan for detected drift

        Args:
            drift: Detected drift

        Returns:
            Remediation plan
        """

        actions = []

        baseline = self.baselines.get(drift.resource_id)
        if not baseline:
            raise ValueError(f"No baseline found for {drift.resource_id}")

        # Generate remediation actions
        for field in drift.fields_changed:
            expected_value = drift.expected_values.get(field)
            actual_value = drift.actual_values.get(field)

            action = {
                "type": "update",
                "field": field,
                "from": actual_value,
                "to": expected_value,
                "method": self._get_remediation_method(drift.resource_type, field)
            }
            actions.append(action)

        # Estimate duration (1 second per field + overhead)
        estimated_duration = len(actions) * 1 + 10

        # Check if approval required
        requires_approval = drift.remediation_strategy in [
            RemediationStrategy.APPROVAL_REQUIRED,
            RemediationStrategy.MANUAL
        ]

        plan = RemediationPlan(
            plan_id=f"plan-{datetime.now().timestamp()}",
            drift_detection=drift,
            actions=actions,
            estimated_duration=estimated_duration,
            requires_approval=requires_approval,
            approval_status="pending" if requires_approval else "auto_approved",
            approved_by=None,
            created_at=datetime.now()
        )

        self.remediation_plans[plan.plan_id] = plan

        self.logger.info(
            f"Created remediation plan {plan.plan_id} with {len(actions)} actions"
        )

        return plan

    def _get_remediation_method(self, resource_type: str, field: str) -> str:
        """Determine remediation method for field"""

        method_map = {
            "vm": "api_update",
            "network": "api_update",
            "policy": "api_update",
            "storage": "api_update"
        }

        return method_map.get(resource_type, "manual")

    def execute_remediation(self, plan_id: str, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute remediation plan

        Args:
            plan_id: Remediation plan ID
            dry_run: If True, simulate without applying

        Returns:
            Execution result
        """

        plan = self.remediation_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        # Check approval
        if plan.requires_approval and plan.approval_status != "approved":
            return {
                "success": False,
                "message": "Remediation requires approval",
                "approval_status": plan.approval_status
            }

        result = {
            "success": True,
            "plan_id": plan_id,
            "actions_executed": [],
            "actions_failed": [],
            "dry_run": dry_run,
            "timestamp": datetime.now().isoformat()
        }

        self.logger.info(
            f"{'Simulating' if dry_run else 'Executing'} remediation plan {plan_id}"
        )

        for action in plan.actions:
            try:
                if dry_run:
                    self.logger.info(f"Would execute: {action}")
                    result["actions_executed"].append(action)
                else:
                    # Execute actual remediation
                    self._execute_action(plan.drift_detection.resource_id, action)
                    result["actions_executed"].append(action)

            except Exception as e:
                self.logger.error(f"Action failed: {action}, error: {e}")
                result["actions_failed"].append({
                    "action": action,
                    "error": str(e)
                })
                result["success"] = False

        if result["success"] and not dry_run:
            # Update baseline
            baseline = self.baselines[plan.drift_detection.resource_id]
            baseline.timestamp = datetime.now()
            baseline.config_hash = self._compute_config_hash(baseline.config)

        return result

    def _execute_action(self, resource_id: str, action: Dict[str, Any]):
        """Execute single remediation action"""

        # Placeholder - would call actual infrastructure API
        self.logger.info(
            f"Executing action on {resource_id}: "
            f"Update {action['field']} from {action['from']} to {action['to']}"
        )

    def approve_remediation(self, plan_id: str, approved_by: str):
        """Approve remediation plan"""

        plan = self.remediation_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan not found: {plan_id}")

        plan.approval_status = "approved"
        plan.approved_by = approved_by

        self.logger.info(f"Remediation plan {plan_id} approved by {approved_by}")

    def export_metrics(self) -> Dict[str, Any]:
        """Export drift detection metrics"""

        total_detections = len(self.detections)
        total_baselines = len(self.baselines)

        severity_counts = {}
        for severity in DriftSeverity:
            severity_counts[severity.value] = sum(
                1 for d in self.detections if d.severity == severity
            )

        remediation_stats = {
            "total_plans": len(self.remediation_plans),
            "pending_approval": sum(
                1 for p in self.remediation_plans.values()
                if p.approval_status == "pending"
            ),
            "approved": sum(
                1 for p in self.remediation_plans.values()
                if p.approval_status == "approved"
            )
        }

        return {
            "total_baselines": total_baselines,
            "total_detections": total_detections,
            "severity_distribution": severity_counts,
            "remediation_stats": remediation_stats,
            "auto_remediate_enabled": self.auto_remediate,
            "drift_threshold": self.drift_threshold
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize detector
    detector = DriftDetector({
        'auto_remediate': False,
        'drift_threshold': 0.05
    })

    # Register baselines
    detector.register_baseline(
        resource_id="vm-001",
        resource_type="vm",
        config={
            "name": "web-server-01",
            "cpu_cores": 4,
            "memory_gb": 8,
            "disk_gb": 100,
            "security_group": "sg-web",
            "firewall_rules": ["allow:80", "allow:443"]
        },
        tags={"environment": "production", "tier": "web"}
    )

    # Simulate drift
    current_configs = {
        "vm-001": {
            "name": "web-server-01",
            "cpu_cores": 6,  # Drifted!
            "memory_gb": 8,
            "disk_gb": 150,  # Drifted!
            "security_group": "sg-web",
            "firewall_rules": ["allow:80", "allow:443", "allow:22"]  # Drifted!
        }
    }

    # Scan for drift
    drifts = detector.scan_for_drift(current_configs)

    print(f"\nDetected {len(drifts)} configuration drifts:\n")

    for drift in drifts:
        print(f"Resource: {drift.resource_id}")
        print(f"Severity: {drift.severity.value}")
        print(f"Drift: {drift.drift_percentage*100:.1f}%")
        print(f"Changed fields: {', '.join(drift.fields_changed)}")
        print(f"Remediation strategy: {drift.remediation_strategy.value}")

        # Create remediation plan
        plan = detector.create_remediation_plan(drift)
        print(f"\nRemediation Plan: {plan.plan_id}")
        print(f"Actions: {len(plan.actions)}")
        print(f"Requires approval: {plan.requires_approval}")

        # Simulate remediation
        result = detector.execute_remediation(plan.plan_id, dry_run=True)
        print(f"Dry run result: {result['success']}")
        print(f"Actions would execute: {len(result['actions_executed'])}")

    # Export metrics
    metrics = detector.export_metrics()
    print(f"\nDrift Detector Metrics:")
    print(json.dumps(metrics, indent=2))
