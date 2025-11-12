#!/usr/bin/env python3
"""
Runbook Automation System
Automatically generates and executes runbooks from incident patterns
Integrates with PagerDuty, OpsGenie, and Slack
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import aiohttp
import yaml
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RunbookStatus(Enum):
    """Runbook execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class StepType(Enum):
    """Types of runbook steps"""
    COMMAND = "command"
    SCRIPT = "script"
    API_CALL = "api_call"
    MANUAL = "manual"
    DECISION = "decision"
    VALIDATION = "validation"
    ROLLBACK = "rollback"


@dataclass
class RunbookStep:
    """Represents a single step in a runbook"""
    id: str
    name: str
    description: str
    type: StepType
    command: Optional[str] = None
    script: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_method: Optional[str] = "GET"
    api_payload: Optional[Dict] = None
    validation: Optional[Callable] = None
    rollback: Optional['RunbookStep'] = None
    timeout: int = 300  # seconds
    retry_count: int = 3
    retry_delay: int = 5
    required: bool = True
    condition: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Runbook:
    """Represents a complete runbook"""
    id: str
    name: str
    description: str
    category: str
    severity: str
    tags: List[str]
    steps: List[RunbookStep]
    variables: Dict[str, Any] = field(default_factory=dict)
    prerequisites: List[str] = field(default_factory=list)
    estimated_duration: int = 0  # seconds
    approval_required: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class RunbookExecution:
    """Tracks runbook execution"""
    id: str
    runbook: Runbook
    status: RunbookStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    executed_steps: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    executor: str = "automated"


class RunbookGenerator:
    """Generates runbooks from incident patterns"""

    def __init__(self, pattern_db_path: str):
        self.pattern_db_path = Path(pattern_db_path)
        self.patterns = self._load_patterns()
        self.templates = self._load_templates()

    def _load_patterns(self) -> Dict[str, Any]:
        """Load incident patterns from database"""
        if not self.pattern_db_path.exists():
            return {}

        with open(self.pattern_db_path) as f:
            return json.load(f)

    def _load_templates(self) -> Dict[str, Template]:
        """Load runbook templates"""
        templates = {}
        template_dir = Path(__file__).parent / "templates"

        if template_dir.exists():
            for template_file in template_dir.glob("*.j2"):
                with open(template_file) as f:
                    templates[template_file.stem] = Template(f.read())

        return templates

    def generate_from_incident(self, incident: Dict[str, Any]) -> Runbook:
        """Generate runbook from incident data"""
        # Find similar patterns
        pattern = self._find_similar_pattern(incident)

        if pattern:
            return self._generate_from_pattern(incident, pattern)
        else:
            return self._generate_generic(incident)

    def _find_similar_pattern(self, incident: Dict[str, Any]) -> Optional[Dict]:
        """Find similar incident pattern"""
        incident_type = incident.get("type", "")
        incident_service = incident.get("service", "")

        for pattern_id, pattern in self.patterns.items():
            if (pattern.get("type") == incident_type and
                pattern.get("service") == incident_service):
                return pattern

        return None

    def _generate_from_pattern(
        self,
        incident: Dict[str, Any],
        pattern: Dict[str, Any]
    ) -> Runbook:
        """Generate runbook from known pattern"""
        steps = []

        # Add detection steps
        steps.append(RunbookStep(
            id="detect",
            name="Verify Incident",
            description=f"Verify {incident['type']} incident on {incident['service']}",
            type=StepType.VALIDATION,
            command=pattern.get("detection_command"),
        ))

        # Add remediation steps from pattern
        for i, action in enumerate(pattern.get("remediation_actions", [])):
            steps.append(RunbookStep(
                id=f"remediate_{i}",
                name=action["name"],
                description=action["description"],
                type=StepType(action["type"]),
                command=action.get("command"),
                script=action.get("script"),
                rollback=self._create_rollback_step(action),
            ))

        # Add validation step
        steps.append(RunbookStep(
            id="validate",
            name="Validate Recovery",
            description="Validate system has recovered",
            type=StepType.VALIDATION,
            command=pattern.get("validation_command"),
        ))

        return Runbook(
            id=f"runbook-{incident['id']}",
            name=f"Runbook for {incident['type']}",
            description=f"Automated runbook for {incident['type']} on {incident['service']}",
            category=incident.get("category", "general"),
            severity=incident.get("severity", "medium"),
            tags=incident.get("tags", []),
            steps=steps,
            estimated_duration=pattern.get("avg_duration", 300),
        )

    def _generate_generic(self, incident: Dict[str, Any]) -> Runbook:
        """Generate generic runbook for unknown incident"""
        steps = [
            RunbookStep(
                id="investigate",
                name="Investigate Incident",
                description="Gather information about the incident",
                type=StepType.MANUAL,
            ),
            RunbookStep(
                id="diagnose",
                name="Diagnose Root Cause",
                description="Identify the root cause of the incident",
                type=StepType.MANUAL,
            ),
            RunbookStep(
                id="remediate",
                name="Apply Remediation",
                description="Apply appropriate remediation steps",
                type=StepType.MANUAL,
            ),
            RunbookStep(
                id="validate",
                name="Validate Recovery",
                description="Confirm system has recovered",
                type=StepType.VALIDATION,
            ),
        ]

        return Runbook(
            id=f"runbook-{incident['id']}",
            name=f"Generic Runbook for {incident['type']}",
            description=f"Manual runbook for {incident['type']}",
            category=incident.get("category", "general"),
            severity=incident.get("severity", "medium"),
            tags=["manual", "generic"],
            steps=steps,
            approval_required=True,
        )

    def _create_rollback_step(self, action: Dict) -> Optional[RunbookStep]:
        """Create rollback step for an action"""
        if "rollback" not in action:
            return None

        return RunbookStep(
            id=f"rollback_{action['id']}",
            name=f"Rollback {action['name']}",
            description=action["rollback"]["description"],
            type=StepType.ROLLBACK,
            command=action["rollback"].get("command"),
            script=action["rollback"].get("script"),
        )


class RunbookExecutor:
    """Executes runbooks with automated validation"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dry_run = config.get("dry_run", False)
        self.execution_history = []

    async def execute(self, runbook: Runbook, context: Dict[str, Any]) -> RunbookExecution:
        """Execute a runbook"""
        execution = RunbookExecution(
            id=f"exec-{int(time.time())}",
            runbook=runbook,
            status=RunbookStatus.RUNNING,
            started_at=datetime.now(),
            context=context,
        )

        logger.info(f"Starting runbook execution: {runbook.name}")

        if self.dry_run:
            logger.info("DRY RUN MODE - No actual changes will be made")

        try:
            # Check prerequisites
            if not await self._check_prerequisites(runbook):
                execution.status = RunbookStatus.FAILED
                execution.errors.append("Prerequisites not met")
                return execution

            # Execute steps
            for step in runbook.steps:
                # Check condition
                if step.condition and not step.condition(context):
                    logger.info(f"Skipping step {step.name} - condition not met")
                    execution.executed_steps.append({
                        "step": step.name,
                        "status": "skipped",
                        "reason": "condition not met",
                    })
                    continue

                # Execute step
                result = await self._execute_step(step, context)

                execution.executed_steps.append({
                    "step": step.name,
                    "status": result["status"],
                    "output": result.get("output"),
                    "error": result.get("error"),
                    "duration": result.get("duration"),
                })

                if result["status"] == "failed":
                    if step.required:
                        # Attempt rollback
                        await self._rollback(execution)
                        execution.status = RunbookStatus.FAILED
                        return execution
                    else:
                        logger.warning(f"Optional step failed: {step.name}")

            execution.status = RunbookStatus.SUCCESS
            logger.info(f"Runbook completed successfully: {runbook.name}")

        except Exception as e:
            logger.error(f"Runbook execution failed: {e}")
            execution.status = RunbookStatus.FAILED
            execution.errors.append(str(e))
            await self._rollback(execution)

        finally:
            execution.ended_at = datetime.now()
            self.execution_history.append(execution)

        return execution

    async def _check_prerequisites(self, runbook: Runbook) -> bool:
        """Check runbook prerequisites"""
        for prereq in runbook.prerequisites:
            # Check prerequisites (simplified)
            logger.info(f"Checking prerequisite: {prereq}")

        return True

    async def _execute_step(
        self,
        step: RunbookStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single runbook step"""
        logger.info(f"Executing step: {step.name}")
        start_time = time.time()

        result = {
            "status": "success",
            "output": None,
            "error": None,
            "duration": 0,
        }

        try:
            if self.dry_run and step.type != StepType.VALIDATION:
                logger.info(f"DRY RUN: Would execute {step.type.value}: {step.name}")
                result["output"] = "DRY RUN - No action taken"
                return result

            # Execute based on step type
            if step.type == StepType.COMMAND:
                output = await self._execute_command(step, context)
                result["output"] = output

            elif step.type == StepType.SCRIPT:
                output = await self._execute_script(step, context)
                result["output"] = output

            elif step.type == StepType.API_CALL:
                output = await self._execute_api_call(step, context)
                result["output"] = output

            elif step.type == StepType.VALIDATION:
                is_valid = await self._execute_validation(step, context)
                if not is_valid:
                    result["status"] = "failed"
                    result["error"] = "Validation failed"

            elif step.type == StepType.MANUAL:
                logger.info(f"Manual step: {step.description}")
                result["output"] = "Manual step - requires human intervention"

            # Run validation if provided
            if step.validation:
                if not step.validation(result["output"], context):
                    result["status"] = "failed"
                    result["error"] = "Step validation failed"

        except subprocess.TimeoutExpired:
            result["status"] = "failed"
            result["error"] = f"Step timeout after {step.timeout}s"
            logger.error(f"Step timeout: {step.name}")

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Step execution failed: {step.name} - {e}")

            # Retry if configured
            if step.retry_count > 0:
                logger.info(f"Retrying step {step.name} ({step.retry_count} attempts remaining)")
                await asyncio.sleep(step.retry_delay)
                step.retry_count -= 1
                return await self._execute_step(step, context)

        finally:
            result["duration"] = time.time() - start_time

        return result

    async def _execute_command(
        self,
        step: RunbookStep,
        context: Dict[str, Any]
    ) -> str:
        """Execute shell command"""
        command = self._substitute_variables(step.command, context)

        logger.info(f"Executing command: {command}")

        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=step.timeout
            )

            if process.returncode != 0:
                raise RuntimeError(f"Command failed: {stderr.decode()}")

            return stdout.decode()

        except asyncio.TimeoutError:
            process.kill()
            raise subprocess.TimeoutExpired(command, step.timeout)

    async def _execute_script(
        self,
        step: RunbookStep,
        context: Dict[str, Any]
    ) -> str:
        """Execute script"""
        script_path = Path(step.script)

        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Make script executable
        os.chmod(script_path, 0o755)

        command = str(script_path)
        return await self._execute_command(
            RunbookStep(
                id=step.id,
                name=step.name,
                description=step.description,
                type=StepType.COMMAND,
                command=command,
                timeout=step.timeout,
            ),
            context
        )

    async def _execute_api_call(
        self,
        step: RunbookStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute API call"""
        url = self._substitute_variables(step.api_endpoint, context)
        method = step.api_method.upper()
        payload = step.api_payload or {}

        logger.info(f"Making API call: {method} {url}")

        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=step.timeout)
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def _execute_validation(
        self,
        step: RunbookStep,
        context: Dict[str, Any]
    ) -> bool:
        """Execute validation"""
        if step.validation:
            return step.validation(context)

        if step.command:
            try:
                output = await self._execute_command(step, context)
                return True
            except Exception:
                return False

        return True

    async def _rollback(self, execution: RunbookExecution):
        """Rollback executed steps"""
        logger.warning("Initiating rollback")

        # Execute rollback steps in reverse order
        for step_result in reversed(execution.executed_steps):
            if step_result["status"] == "success":
                step = next(
                    (s for s in execution.runbook.steps if s.name == step_result["step"]),
                    None
                )

                if step and step.rollback:
                    logger.info(f"Rolling back step: {step.name}")
                    try:
                        await self._execute_step(step.rollback, execution.context)
                    except Exception as e:
                        logger.error(f"Rollback failed for {step.name}: {e}")

    def _substitute_variables(self, text: str, context: Dict[str, Any]) -> str:
        """Substitute variables in text"""
        if not text:
            return text

        for key, value in context.items():
            text = text.replace(f"${{{key}}}", str(value))

        return text


class IntegrationManager:
    """Manages integrations with external services"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pagerduty_key = config.get("pagerduty_api_key")
        self.opsgenie_key = config.get("opsgenie_api_key")
        self.slack_webhook = config.get("slack_webhook_url")

    async def notify_pagerduty(
        self,
        event_type: str,
        summary: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Send notification to PagerDuty"""
        if not self.pagerduty_key:
            logger.warning("PagerDuty API key not configured")
            return

        url = "https://events.pagerduty.com/v2/enqueue"
        payload = {
            "routing_key": self.pagerduty_key,
            "event_action": event_type,
            "payload": {
                "summary": summary,
                "severity": severity,
                "source": "runbook-automation",
                "custom_details": details,
            }
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 202:
                    logger.info("PagerDuty notification sent")
                else:
                    logger.error(f"PagerDuty notification failed: {await response.text()}")

    async def notify_opsgenie(
        self,
        message: str,
        priority: str,
        details: Dict[str, Any]
    ):
        """Send notification to OpsGenie"""
        if not self.opsgenie_key:
            logger.warning("OpsGenie API key not configured")
            return

        url = "https://api.opsgenie.com/v2/alerts"
        headers = {"Authorization": f"GenieKey {self.opsgenie_key}"}
        payload = {
            "message": message,
            "priority": priority,
            "details": details,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 202:
                    logger.info("OpsGenie notification sent")
                else:
                    logger.error(f"OpsGenie notification failed: {await response.text()}")

    async def notify_slack(self, message: str, attachments: Optional[List[Dict]] = None):
        """Send notification to Slack"""
        if not self.slack_webhook:
            logger.warning("Slack webhook URL not configured")
            return

        payload = {
            "text": message,
            "attachments": attachments or [],
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.slack_webhook, json=payload) as response:
                if response.status == 200:
                    logger.info("Slack notification sent")
                else:
                    logger.error(f"Slack notification failed: {await response.text()}")


class RunbookAutomation:
    """Main runbook automation system"""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.generator = RunbookGenerator(
            self.config.get("pattern_db_path", "patterns.json")
        )
        self.executor = RunbookExecutor(self.config)
        self.integrations = IntegrationManager(self.config)

    async def handle_incident(self, incident: Dict[str, Any]):
        """Handle an incident with automated runbook"""
        logger.info(f"Handling incident: {incident['id']}")

        # Generate runbook
        runbook = self.generator.generate_from_incident(incident)

        # Send notifications
        await self.integrations.notify_slack(
            f"🚨 Incident detected: {incident['type']}\n"
            f"Generated runbook: {runbook.name}"
        )

        # Execute runbook if no approval required
        if not runbook.approval_required:
            execution = await self.executor.execute(runbook, incident)

            # Send completion notification
            if execution.status == RunbookStatus.SUCCESS:
                await self.integrations.notify_slack(
                    f"✅ Runbook completed successfully for incident {incident['id']}\n"
                    f"Duration: {(execution.ended_at - execution.started_at).seconds}s"
                )
            else:
                await self.integrations.notify_pagerduty(
                    "trigger",
                    f"Runbook execution failed for {incident['id']}",
                    "critical",
                    {"execution_id": execution.id, "errors": execution.errors}
                )
        else:
            logger.info("Runbook requires approval - waiting for manual execution")
            await self.integrations.notify_slack(
                f"⚠️ Runbook requires approval: {runbook.name}\n"
                f"Please review and execute manually"
            )


async def main():
    """Main entry point"""
    config_path = os.environ.get("RUNBOOK_CONFIG", "config/runbook.yaml")

    automation = RunbookAutomation(config_path)

    # Example incident
    incident = {
        "id": "INC-001",
        "type": "high_latency",
        "service": "api-gateway",
        "severity": "high",
        "category": "performance",
        "tags": ["latency", "api"],
        "context": {
            "region": "us-west-2",
            "environment": "production",
        }
    }

    await automation.handle_incident(incident)


if __name__ == "__main__":
    asyncio.run(main())
