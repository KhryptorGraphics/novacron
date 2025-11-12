#!/usr/bin/env python3
"""
Runbook Automation Engine 2.0
Automatically generates, executes, and validates runbooks from system behavior
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


class RunbookType(Enum):
    """Types of runbooks"""
    DEPLOYMENT = "deployment"
    INCIDENT = "incident_response"
    MAINTENANCE = "maintenance"
    DISASTER_RECOVERY = "disaster_recovery"
    SCALING = "scaling"
    BACKUP = "backup"
    SECURITY = "security"


class StepStatus(Enum):
    """Status of runbook steps"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RunbookStep:
    """Represents a single runbook step"""
    id: str
    name: str
    description: str
    command: str
    expected_output: Optional[str] = None
    timeout: int = 300
    retries: int = 3
    on_failure: str = "abort"  # abort, continue, retry
    validation: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)

    status: StepStatus = StepStatus.PENDING
    output: str = ""
    error: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    attempts: int = 0


@dataclass
class Runbook:
    """Represents a complete runbook"""
    id: str
    name: str
    description: str
    type: RunbookType
    version: str
    steps: List[RunbookStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    author: str = "system-generated"
    created_at: datetime = field(default_factory=datetime.now)

    # Execution tracking
    execution_id: Optional[str] = None
    execution_status: str = "not_started"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class RunbookGenerator:
    """Generates runbooks from system behavior and patterns"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.patterns = self._load_patterns()

    def _load_patterns(self) -> Dict[str, Any]:
        """Load runbook generation patterns"""
        return {
            "deployment": {
                "steps": [
                    "pre_deployment_checks",
                    "backup_current_state",
                    "deploy_new_version",
                    "health_checks",
                    "rollback_on_failure"
                ]
            },
            "incident_response": {
                "steps": [
                    "identify_issue",
                    "isolate_affected_resources",
                    "gather_diagnostics",
                    "apply_fix",
                    "verify_resolution",
                    "post_incident_review"
                ]
            },
            "scaling": {
                "steps": [
                    "check_current_capacity",
                    "calculate_required_resources",
                    "provision_resources",
                    "configure_load_balancer",
                    "verify_scaling"
                ]
            }
        }

    def generate_from_incident(self, incident_data: Dict[str, Any]) -> Runbook:
        """Generate runbook from incident data"""
        self.logger.info(f"Generating runbook from incident: {incident_data.get('id')}")

        steps = []

        # Step 1: Diagnostic collection
        steps.append(RunbookStep(
            id="step-1",
            name="Collect diagnostics",
            description="Gather system diagnostics and logs",
            command=f"kubectl logs {incident_data.get('pod_name')} --tail=500",
            timeout=60
        ))

        # Step 2: Check resource status
        steps.append(RunbookStep(
            id="step-2",
            name="Check resource status",
            description="Verify resource availability and health",
            command="kubectl get pods,services,deployments -o wide",
            dependencies=["step-1"]
        ))

        # Step 3: Apply fix based on incident type
        fix_command = self._generate_fix_command(incident_data)
        steps.append(RunbookStep(
            id="step-3",
            name="Apply fix",
            description=f"Apply fix for {incident_data.get('type')}",
            command=fix_command,
            dependencies=["step-2"],
            on_failure="abort"
        ))

        # Step 4: Verification
        steps.append(RunbookStep(
            id="step-4",
            name="Verify resolution",
            description="Verify the issue is resolved",
            command="kubectl get pods | grep Running | wc -l",
            expected_output="3",  # Expected number of running pods
            dependencies=["step-3"]
        ))

        runbook = Runbook(
            id=f"rb-incident-{int(time.time())}",
            name=f"Incident Response: {incident_data.get('title')}",
            description=f"Auto-generated runbook for {incident_data.get('type')} incident",
            type=RunbookType.INCIDENT,
            version="1.0.0",
            steps=steps,
            variables=incident_data,
            tags=["auto-generated", "incident-response", incident_data.get('severity', 'medium')]
        )

        return runbook

    def generate_from_pattern(self, pattern_name: str, context: Dict[str, Any]) -> Runbook:
        """Generate runbook from a known pattern"""
        self.logger.info(f"Generating runbook from pattern: {pattern_name}")

        pattern = self.patterns.get(pattern_name)
        if not pattern:
            raise ValueError(f"Unknown pattern: {pattern_name}")

        steps = []
        for i, step_template in enumerate(pattern["steps"]):
            step = RunbookStep(
                id=f"step-{i+1}",
                name=step_template.replace("_", " ").title(),
                description=f"Execute {step_template}",
                command=self._generate_command_for_step(step_template, context),
                dependencies=[f"step-{i}"] if i > 0 else []
            )
            steps.append(step)

        runbook = Runbook(
            id=f"rb-pattern-{int(time.time())}",
            name=f"{pattern_name.title()} Runbook",
            description=f"Auto-generated {pattern_name} runbook",
            type=RunbookType[pattern_name.upper()],
            version="1.0.0",
            steps=steps,
            variables=context,
            tags=["auto-generated", pattern_name]
        )

        return runbook

    def _generate_fix_command(self, incident_data: Dict[str, Any]) -> str:
        """Generate fix command based on incident type"""
        incident_type = incident_data.get("type", "unknown")

        fix_commands = {
            "pod_crash": "kubectl rollout restart deployment/{deployment_name}",
            "high_cpu": "kubectl scale deployment {deployment_name} --replicas={new_replicas}",
            "oom": "kubectl set resources deployment {deployment_name} --limits=memory=2Gi",
            "disk_full": "kubectl exec {pod_name} -- rm -rf /tmp/*"
        }

        command_template = fix_commands.get(incident_type, "echo 'Manual intervention required'")
        return command_template.format(**incident_data)

    def _generate_command_for_step(self, step_name: str, context: Dict[str, Any]) -> str:
        """Generate command for a specific step"""
        commands = {
            "pre_deployment_checks": "kubectl get nodes && kubectl get pods --all-namespaces",
            "backup_current_state": "kubectl get all -o yaml > backup-$(date +%Y%m%d-%H%M%S).yaml",
            "deploy_new_version": f"kubectl apply -f {context.get('manifest_file', 'deployment.yaml')}",
            "health_checks": "kubectl wait --for=condition=ready pod -l app={app_name} --timeout=300s",
            "rollback_on_failure": "kubectl rollout undo deployment/{deployment_name}",
            "check_current_capacity": "kubectl top nodes && kubectl top pods",
            "calculate_required_resources": "echo 'Analyzing workload...'",
            "provision_resources": "kubectl scale deployment {deployment_name} --replicas={replicas}",
            "configure_load_balancer": "kubectl patch svc {service_name} -p '{patch_json}'",
            "verify_scaling": "kubectl get pods -l app={app_name}"
        }

        command_template = commands.get(step_name, f"echo 'Execute {step_name}'")
        try:
            return command_template.format(**context)
        except KeyError:
            return command_template


class RunbookExecutor:
    """Executes runbooks with proper error handling and validation"""

    def __init__(self, logger: logging.Logger, dry_run: bool = False):
        self.logger = logger
        self.dry_run = dry_run

    def execute(self, runbook: Runbook) -> bool:
        """Execute a complete runbook"""
        runbook.execution_id = f"exec-{int(time.time())}"
        runbook.execution_status = "running"
        runbook.start_time = datetime.now()

        self.logger.info(f"Executing runbook: {runbook.name} (ID: {runbook.execution_id})")

        try:
            # Execute steps in order, respecting dependencies
            for step in runbook.steps:
                if not self._execute_step(step, runbook):
                    if step.on_failure == "abort":
                        runbook.execution_status = "failed"
                        return False
                    elif step.on_failure == "continue":
                        step.status = StepStatus.SKIPPED
                        continue

            runbook.execution_status = "completed"
            return True

        except Exception as e:
            self.logger.error(f"Runbook execution failed: {str(e)}")
            runbook.execution_status = "failed"
            return False

        finally:
            runbook.end_time = datetime.now()
            self._generate_execution_report(runbook)

    def _execute_step(self, step: RunbookStep, runbook: Runbook) -> bool:
        """Execute a single runbook step"""
        # Check dependencies
        for dep_id in step.dependencies:
            dep_step = next((s for s in runbook.steps if s.id == dep_id), None)
            if dep_step and dep_step.status != StepStatus.COMPLETED:
                self.logger.warning(f"Step {step.id} skipped due to failed dependency {dep_id}")
                step.status = StepStatus.SKIPPED
                return True

        self.logger.info(f"Executing step: {step.name}")
        step.status = StepStatus.RUNNING
        step.start_time = datetime.now()

        # Retry loop
        for attempt in range(step.retries):
            step.attempts = attempt + 1

            try:
                if self.dry_run:
                    self.logger.info(f"[DRY RUN] Would execute: {step.command}")
                    step.output = "[DRY RUN] Command not executed"
                    success = True
                else:
                    success = self._execute_command(step)

                if success:
                    step.status = StepStatus.COMPLETED
                    step.end_time = datetime.now()
                    self.logger.info(f"Step {step.id} completed successfully")
                    return True

                if attempt < step.retries - 1:
                    self.logger.warning(f"Step {step.id} failed, retrying ({attempt + 1}/{step.retries})")
                    time.sleep(2 ** attempt)  # Exponential backoff

            except Exception as e:
                self.logger.error(f"Step {step.id} execution error: {str(e)}")
                step.error = str(e)

        step.status = StepStatus.FAILED
        step.end_time = datetime.now()
        return False

    def _execute_command(self, step: RunbookStep) -> bool:
        """Execute a single command"""
        try:
            result = subprocess.run(
                step.command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=step.timeout
            )

            step.output = result.stdout
            step.error = result.stderr

            # Validate output if expected output is specified
            if step.expected_output:
                if step.expected_output not in result.stdout:
                    self.logger.warning(f"Output validation failed for step {step.id}")
                    return False

            # Validate using custom validation if specified
            if step.validation:
                if not self._validate_output(result.stdout, step.validation):
                    self.logger.warning(f"Custom validation failed for step {step.id}")
                    return False

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            self.logger.error(f"Step {step.id} timed out after {step.timeout} seconds")
            step.error = f"Timeout after {step.timeout} seconds"
            return False

        except Exception as e:
            self.logger.error(f"Command execution failed: {str(e)}")
            step.error = str(e)
            return False

    def _validate_output(self, output: str, validation: Dict[str, Any]) -> bool:
        """Validate command output using custom validation rules"""
        validation_type = validation.get("type")

        if validation_type == "contains":
            return validation["value"] in output

        elif validation_type == "regex":
            import re
            pattern = validation["pattern"]
            return re.search(pattern, output) is not None

        elif validation_type == "json":
            try:
                data = json.loads(output)
                key = validation.get("key")
                expected = validation.get("expected")
                return data.get(key) == expected
            except:
                return False

        return True

    def _generate_execution_report(self, runbook: Runbook):
        """Generate execution report"""
        duration = (runbook.end_time - runbook.start_time).total_seconds() if runbook.end_time else 0

        report = {
            "runbook_id": runbook.id,
            "execution_id": runbook.execution_id,
            "name": runbook.name,
            "status": runbook.execution_status,
            "duration_seconds": duration,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "status": step.status.value,
                    "attempts": step.attempts,
                    "output": step.output[:500],  # Truncate
                    "error": step.error
                }
                for step in runbook.steps
            ]
        }

        report_file = f"/tmp/runbook-report-{runbook.execution_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Execution report saved to: {report_file}")


class RunbookValidator:
    """Validates runbooks for correctness and safety"""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def validate(self, runbook: Runbook) -> tuple[bool, List[str]]:
        """Validate a runbook"""
        errors = []

        # Check basic structure
        if not runbook.steps:
            errors.append("Runbook must have at least one step")

        # Check step dependencies
        step_ids = {step.id for step in runbook.steps}
        for step in runbook.steps:
            for dep_id in step.dependencies:
                if dep_id not in step_ids:
                    errors.append(f"Step {step.id} has invalid dependency: {dep_id}")

        # Check for dangerous commands
        dangerous_patterns = ['rm -rf /', 'dd if=', 'mkfs.', '> /dev/']
        for step in runbook.steps:
            for pattern in dangerous_patterns:
                if pattern in step.command:
                    errors.append(f"Step {step.id} contains potentially dangerous command: {pattern}")

        # Check timeout values
        for step in runbook.steps:
            if step.timeout <= 0 or step.timeout > 3600:
                errors.append(f"Step {step.id} has invalid timeout: {step.timeout}")

        is_valid = len(errors) == 0

        if is_valid:
            self.logger.info(f"Runbook validation passed: {runbook.name}")
        else:
            self.logger.error(f"Runbook validation failed: {', '.join(errors)}")

        return is_valid, errors


class RunbookLibrary:
    """Manages a library of runbooks"""

    def __init__(self, storage_path: str, logger: logging.Logger):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def save(self, runbook: Runbook) -> str:
        """Save runbook to library"""
        filename = f"{runbook.id}.yaml"
        filepath = self.storage_path / filename

        # Convert to dict
        runbook_dict = {
            "id": runbook.id,
            "name": runbook.name,
            "description": runbook.description,
            "type": runbook.type.value,
            "version": runbook.version,
            "author": runbook.author,
            "created_at": runbook.created_at.isoformat(),
            "variables": runbook.variables,
            "tags": runbook.tags,
            "steps": [
                {
                    "id": step.id,
                    "name": step.name,
                    "description": step.description,
                    "command": step.command,
                    "expected_output": step.expected_output,
                    "timeout": step.timeout,
                    "retries": step.retries,
                    "on_failure": step.on_failure,
                    "validation": step.validation,
                    "dependencies": step.dependencies
                }
                for step in runbook.steps
            ]
        }

        with open(filepath, 'w') as f:
            yaml.dump(runbook_dict, f, default_flow_style=False)

        self.logger.info(f"Runbook saved: {filepath}")
        return str(filepath)

    def load(self, runbook_id: str) -> Runbook:
        """Load runbook from library"""
        filename = f"{runbook_id}.yaml"
        filepath = self.storage_path / filename

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        # Reconstruct runbook
        steps = [
            RunbookStep(
                id=step["id"],
                name=step["name"],
                description=step["description"],
                command=step["command"],
                expected_output=step.get("expected_output"),
                timeout=step.get("timeout", 300),
                retries=step.get("retries", 3),
                on_failure=step.get("on_failure", "abort"),
                validation=step.get("validation"),
                dependencies=step.get("dependencies", [])
            )
            for step in data["steps"]
        ]

        runbook = Runbook(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            type=RunbookType(data["type"]),
            version=data["version"],
            steps=steps,
            variables=data.get("variables", {}),
            tags=data.get("tags", []),
            author=data.get("author", "unknown"),
            created_at=datetime.fromisoformat(data["created_at"])
        )

        self.logger.info(f"Runbook loaded: {runbook_id}")
        return runbook

    def list(self, tag: Optional[str] = None) -> List[str]:
        """List runbooks in library"""
        runbooks = []
        for filepath in self.storage_path.glob("*.yaml"):
            if tag:
                # Load and check tags
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                    if tag in data.get("tags", []):
                        runbooks.append(filepath.stem)
            else:
                runbooks.append(filepath.stem)

        return runbooks


def main():
    """Main entry point for runbook automation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("RunbookEngine")

    # Example usage
    generator = RunbookGenerator(logger)
    validator = RunbookValidator(logger)
    executor = RunbookExecutor(logger, dry_run=True)
    library = RunbookLibrary("/tmp/runbooks", logger)

    # Generate example incident runbook
    incident = {
        "id": "inc-12345",
        "title": "Pod CrashLoopBackOff",
        "type": "pod_crash",
        "severity": "high",
        "pod_name": "api-server-abc123",
        "deployment_name": "api-server"
    }

    runbook = generator.generate_from_incident(incident)

    # Validate
    is_valid, errors = validator.validate(runbook)
    if is_valid:
        # Save to library
        library.save(runbook)

        # Execute
        success = executor.execute(runbook)
        logger.info(f"Runbook execution {'succeeded' if success else 'failed'}")
    else:
        logger.error(f"Validation errors: {errors}")


if __name__ == "__main__":
    main()
