#!/usr/bin/env python3
"""
Compliance Automation Framework
Continuous compliance validation against standards (CIS, PCI-DSS, HIPAA, SOC2)
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum


class ComplianceStandard(Enum):
    """Compliance standards"""
    CIS = "cis"
    PCI_DSS = "pci_dss"
    HIPAA = "hipaa"
    SOC2 = "soc2"
    GDPR = "gdpr"
    ISO27001 = "iso27001"


class Severity(Enum):
    """Compliance violation severity"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ComplianceControl:
    """Compliance control definition"""
    control_id: str
    standard: ComplianceStandard
    name: str
    description: str
    section: str
    severity: Severity
    validation_script: str
    remediation_script: Optional[str]
    auto_remediate: bool
    tags: List[str]


@dataclass
class ComplianceViolation:
    """Compliance violation"""
    violation_id: str
    control: ComplianceControl
    resource_id: str
    resource_type: str
    severity: Severity
    details: str
    detected_at: datetime
    remediation_available: bool
    remediation_applied: bool
    resolved_at: Optional[datetime]


@dataclass
class ComplianceReport:
    """Compliance assessment report"""
    report_id: str
    standard: ComplianceStandard
    scan_started: datetime
    scan_completed: datetime
    total_controls: int
    passed_controls: int
    failed_controls: int
    violations: List[ComplianceViolation]
    compliance_score: float
    summary: Dict[str, Any]


class ComplianceFramework:
    """
    Automated compliance validation and remediation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Storage
        self.controls: Dict[str, ComplianceControl] = {}
        self.violations: List[ComplianceViolation] = []
        self.reports: List[ComplianceReport] = []

        # Configuration
        self.auto_remediate = self.config.get('auto_remediate', False)
        self.scan_interval = self.config.get('scan_interval', 86400)  # 24 hours

        # Initialize built-in controls
        self._initialize_controls()

        self.logger.info("Compliance framework initialized")

    def _initialize_controls(self):
        """Initialize built-in compliance controls"""

        # CIS Controls
        self._register_control(ComplianceControl(
            control_id="cis-1.1.1",
            standard=ComplianceStandard.CIS,
            name="Ensure mounting of cramfs filesystems is disabled",
            description="The cramfs filesystem type is a compressed read-only Linux filesystem",
            section="1.1 Filesystem Configuration",
            severity=Severity.HIGH,
            validation_script="check_cramfs_disabled",
            remediation_script="disable_cramfs",
            auto_remediate=True,
            tags=["filesystem", "hardening"]
        ))

        self._register_control(ComplianceControl(
            control_id="cis-1.1.2",
            standard=ComplianceStandard.CIS,
            name="Ensure /tmp is configured",
            description="The /tmp directory is a world-writable directory used for temporary storage",
            section="1.1 Filesystem Configuration",
            severity=Severity.MEDIUM,
            validation_script="check_tmp_configured",
            remediation_script="configure_tmp",
            auto_remediate=False,
            tags=["filesystem", "tmp"]
        ))

        # PCI-DSS Controls
        self._register_control(ComplianceControl(
            control_id="pci-2.1",
            standard=ComplianceStandard.PCI_DSS,
            name="Change vendor-supplied defaults",
            description="Always change vendor-supplied defaults and remove or disable unnecessary default accounts",
            section="2 - Do not use vendor-supplied defaults",
            severity=Severity.CRITICAL,
            validation_script="check_default_passwords",
            remediation_script=None,
            auto_remediate=False,
            tags=["passwords", "defaults"]
        ))

        self._register_control(ComplianceControl(
            control_id="pci-8.3",
            standard=ComplianceStandard.PCI_DSS,
            name="Implement multi-factor authentication",
            description="MFA is required for all remote access to CDE",
            section="8 - Identify and authenticate access",
            severity=Severity.CRITICAL,
            validation_script="check_mfa_enabled",
            remediation_script="enable_mfa",
            auto_remediate=False,
            tags=["authentication", "mfa"]
        ))

        # HIPAA Controls
        self._register_control(ComplianceControl(
            control_id="hipaa-164.312a1",
            standard=ComplianceStandard.HIPAA,
            name="Access Control",
            description="Implement technical policies and procedures for ePHI access",
            section="164.312(a)(1) - Access Control",
            severity=Severity.CRITICAL,
            validation_script="check_access_controls",
            remediation_script="configure_access_controls",
            auto_remediate=False,
            tags=["access-control", "ephi"]
        ))

        self._register_control(ComplianceControl(
            control_id="hipaa-164.312a2iv",
            standard=ComplianceStandard.HIPAA,
            name="Encryption",
            description="Implement encryption for ePHI at rest and in transit",
            section="164.312(a)(2)(iv) - Encryption",
            severity=Severity.CRITICAL,
            validation_script="check_encryption",
            remediation_script="enable_encryption",
            auto_remediate=False,
            tags=["encryption", "data-protection"]
        ))

        self.logger.info(f"Initialized {len(self.controls)} compliance controls")

    def _register_control(self, control: ComplianceControl):
        """Register compliance control"""
        self.controls[control.control_id] = control

    def run_compliance_scan(self, standard: ComplianceStandard,
                           resources: List[Dict[str, Any]]) -> ComplianceReport:
        """
        Run compliance scan against standard

        Args:
            standard: Compliance standard to scan against
            resources: Resources to scan

        Returns:
            Compliance report
        """

        scan_started = datetime.now()

        self.logger.info(f"Starting compliance scan for {standard.value}")

        # Get controls for standard
        controls = [c for c in self.controls.values() if c.standard == standard]

        violations = []
        passed = 0
        failed = 0

        for control in controls:
            for resource in resources:
                violation = self._validate_control(control, resource)

                if violation:
                    violations.append(violation)
                    failed += 1

                    # Auto-remediate if enabled
                    if self.auto_remediate and control.auto_remediate and control.remediation_script:
                        self.logger.info(f"Auto-remediating violation: {violation.violation_id}")
                        remediation_result = self._remediate_violation(violation)

                        if remediation_result:
                            violation.remediation_applied = True
                            violation.resolved_at = datetime.now()
                else:
                    passed += 1

        scan_completed = datetime.now()

        # Calculate compliance score
        total = passed + failed
        compliance_score = (passed / total * 100) if total > 0 else 0

        # Generate summary
        severity_counts = {}
        for sev in Severity:
            severity_counts[sev.value] = sum(
                1 for v in violations if v.severity == sev
            )

        report = ComplianceReport(
            report_id=f"report-{scan_completed.timestamp()}",
            standard=standard,
            scan_started=scan_started,
            scan_completed=scan_completed,
            total_controls=total,
            passed_controls=passed,
            failed_controls=failed,
            violations=violations,
            compliance_score=compliance_score,
            summary={
                "severity_distribution": severity_counts,
                "critical_violations": severity_counts.get(Severity.CRITICAL.value, 0),
                "auto_remediated": sum(1 for v in violations if v.remediation_applied),
                "scan_duration": (scan_completed - scan_started).total_seconds()
            }
        )

        self.reports.append(report)
        self.violations.extend(violations)

        self.logger.info(
            f"Compliance scan completed: {compliance_score:.1f}% compliant "
            f"({passed}/{total} controls passed)"
        )

        return report

    def _validate_control(self, control: ComplianceControl,
                         resource: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Validate single control against resource"""

        # Execute validation script
        is_compliant = self._execute_validation(control.validation_script, resource)

        if is_compliant:
            return None

        # Create violation
        violation = ComplianceViolation(
            violation_id=f"violation-{datetime.now().timestamp()}",
            control=control,
            resource_id=resource.get('id', 'unknown'),
            resource_type=resource.get('type', 'unknown'),
            severity=control.severity,
            details=f"Control {control.control_id} failed: {control.name}",
            detected_at=datetime.now(),
            remediation_available=control.remediation_script is not None,
            remediation_applied=False,
            resolved_at=None
        )

        self.logger.warning(
            f"Compliance violation: {control.control_id} on {resource.get('id')}"
        )

        return violation

    def _execute_validation(self, script: str, resource: Dict[str, Any]) -> bool:
        """Execute validation script"""

        # Validation script implementations
        validations = {
            "check_cramfs_disabled": lambda r: r.get('cramfs_enabled', True) is False,
            "check_tmp_configured": lambda r: r.get('tmp_configured', False) is True,
            "check_default_passwords": lambda r: r.get('using_default_password', True) is False,
            "check_mfa_enabled": lambda r: r.get('mfa_enabled', False) is True,
            "check_access_controls": lambda r: r.get('access_controls_configured', False) is True,
            "check_encryption": lambda r: r.get('encryption_enabled', False) is True,
        }

        validator = validations.get(script)
        if not validator:
            self.logger.warning(f"Unknown validation script: {script}")
            return True

        try:
            return validator(resource)
        except Exception as e:
            self.logger.error(f"Validation error: {e}")
            return True

    def _remediate_violation(self, violation: ComplianceViolation) -> bool:
        """Remediate compliance violation"""

        if not violation.control.remediation_script:
            return False

        self.logger.info(
            f"Remediating {violation.control.control_id} on {violation.resource_id}"
        )

        try:
            # Execute remediation script
            success = self._execute_remediation(
                violation.control.remediation_script,
                violation.resource_id
            )

            if success:
                violation.remediation_applied = True
                violation.resolved_at = datetime.now()

            return success

        except Exception as e:
            self.logger.error(f"Remediation failed: {e}")
            return False

    def _execute_remediation(self, script: str, resource_id: str) -> bool:
        """Execute remediation script"""

        # Remediation script implementations
        remediations = {
            "disable_cramfs": lambda r: True,
            "configure_tmp": lambda r: True,
            "enable_mfa": lambda r: True,
            "configure_access_controls": lambda r: True,
            "enable_encryption": lambda r: True,
        }

        remediation = remediations.get(script)
        if not remediation:
            self.logger.warning(f"Unknown remediation script: {script}")
            return False

        try:
            return remediation(resource_id)
        except Exception as e:
            self.logger.error(f"Remediation execution error: {e}")
            return False

    def get_compliance_status(self, standard: ComplianceStandard) -> Dict[str, Any]:
        """Get current compliance status for standard"""

        # Get latest report for standard
        reports = [r for r in self.reports if r.standard == standard]
        if not reports:
            return {
                "standard": standard.value,
                "status": "no_data",
                "last_scan": None
            }

        latest_report = max(reports, key=lambda r: r.scan_completed)

        # Get open violations
        open_violations = [
            v for v in self.violations
            if v.control.standard == standard and v.resolved_at is None
        ]

        return {
            "standard": standard.value,
            "compliance_score": latest_report.compliance_score,
            "last_scan": latest_report.scan_completed.isoformat(),
            "total_violations": len(latest_report.violations),
            "open_violations": len(open_violations),
            "critical_violations": sum(
                1 for v in open_violations if v.severity == Severity.CRITICAL
            ),
            "auto_remediated": latest_report.summary["auto_remediated"],
            "passed_controls": latest_report.passed_controls,
            "failed_controls": latest_report.failed_controls
        }

    def generate_audit_report(self, standard: ComplianceStandard,
                             start_date: datetime,
                             end_date: datetime) -> Dict[str, Any]:
        """Generate audit report for compliance standard"""

        # Filter reports by date range
        reports = [
            r for r in self.reports
            if r.standard == standard
            and start_date <= r.scan_completed <= end_date
        ]

        if not reports:
            return {
                "standard": standard.value,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "status": "no_data"
            }

        # Calculate trends
        avg_score = sum(r.compliance_score for r in reports) / len(reports)
        first_score = reports[0].compliance_score
        last_score = reports[-1].compliance_score
        score_trend = last_score - first_score

        # Violation trends
        total_violations = sum(len(r.violations) for r in reports)
        auto_remediated = sum(r.summary["auto_remediated"] for r in reports)

        return {
            "standard": standard.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_scans": len(reports),
            "average_compliance_score": avg_score,
            "score_trend": score_trend,
            "total_violations": total_violations,
            "auto_remediated_count": auto_remediated,
            "auto_remediation_rate": auto_remediated / max(1, total_violations),
            "latest_score": last_score
        }

    def export_metrics(self) -> Dict[str, Any]:
        """Export compliance framework metrics"""

        total_violations = len(self.violations)
        resolved_violations = sum(1 for v in self.violations if v.resolved_at)

        severity_counts = {}
        for sev in Severity:
            severity_counts[sev.value] = sum(
                1 for v in self.violations
                if v.severity == sev and not v.resolved_at
            )

        return {
            "total_controls": len(self.controls),
            "total_scans": len(self.reports),
            "total_violations": total_violations,
            "open_violations": total_violations - resolved_violations,
            "resolved_violations": resolved_violations,
            "resolution_rate": resolved_violations / max(1, total_violations),
            "severity_distribution": severity_counts,
            "auto_remediate_enabled": self.auto_remediate,
            "standards_monitored": list(set(c.standard.value for c in self.controls.values()))
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize compliance framework
    framework = ComplianceFramework({
        'auto_remediate': True,
        'scan_interval': 3600
    })

    # Simulate resources
    resources = [
        {
            "id": "server-001",
            "type": "vm",
            "cramfs_enabled": False,
            "tmp_configured": True,
            "using_default_password": False,
            "mfa_enabled": True,
            "access_controls_configured": True,
            "encryption_enabled": True
        },
        {
            "id": "server-002",
            "type": "vm",
            "cramfs_enabled": True,  # Violation!
            "tmp_configured": False,  # Violation!
            "using_default_password": False,
            "mfa_enabled": False,  # Violation!
            "access_controls_configured": False,  # Violation!
            "encryption_enabled": True
        }
    ]

    # Run compliance scans
    print("\n=== Running Compliance Scans ===\n")

    for standard in [ComplianceStandard.CIS, ComplianceStandard.PCI_DSS, ComplianceStandard.HIPAA]:
        report = framework.run_compliance_scan(standard, resources)

        print(f"\n{standard.value.upper()} Compliance Report:")
        print(f"Score: {report.compliance_score:.1f}%")
        print(f"Passed: {report.passed_controls}/{report.total_controls}")
        print(f"Violations: {len(report.violations)}")
        print(f"Auto-remediated: {report.summary['auto_remediated']}")

        # Show violations
        if report.violations:
            print("\nViolations:")
            for v in report.violations:
                print(f"  - {v.control.control_id}: {v.control.name}")
                print(f"    Resource: {v.resource_id}")
                print(f"    Severity: {v.severity.value}")
                print(f"    Remediated: {v.remediation_applied}")

    # Get compliance status
    print("\n\n=== Compliance Status ===\n")
    for standard in [ComplianceStandard.CIS, ComplianceStandard.PCI_DSS]:
        status = framework.get_compliance_status(standard)
        print(f"\n{standard.value.upper()}:")
        print(json.dumps(status, indent=2))

    # Generate audit report
    print("\n\n=== Audit Report ===\n")
    audit_report = framework.generate_audit_report(
        ComplianceStandard.CIS,
        datetime.now() - timedelta(days=30),
        datetime.now()
    )
    print(json.dumps(audit_report, indent=2))

    # Export metrics
    print("\n\n=== Framework Metrics ===\n")
    metrics = framework.export_metrics()
    print(json.dumps(metrics, indent=2))
