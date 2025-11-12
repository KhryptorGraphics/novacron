#!/usr/bin/env python3
"""
Runbook Automation 2.0 with AI Enhancements
Automated execution of operational runbooks with intelligent decision-making
"""

import json
import logging
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class StepStatus(Enum):
    """Runbook step status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class RunbookType(Enum):
    """Runbook types"""
    INCIDENT_RESPONSE = "incident_response"
    MAINTENANCE = "maintenance"
    DEPLOYMENT = "deployment"
    RECOVERY = "recovery"
    DIAGNOSTIC = "diagnostic"


@dataclass
class RunbookStep:
    """Runbook step definition"""
    step_id: str
    name: str
    description: str
    action_type: str  # command, api, manual, decision
    action: str
    parameters: Dict[str, Any]
    expected_result: Optional[str]
    timeout: int
    retry_count: int
    dependencies: List[str]
    condition: Optional[str]


@dataclass
class RunbookExecution:
    """Runbook execution instance"""
    execution_id: str
    runbook_id: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime]
    steps_completed: int
    steps_failed: int
    triggered_by: str
    context: Dict[str, Any]


class RunbookExecutor:
    """
    AI-enhanced runbook executor
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Storage
        self.runbooks: Dict[str, Dict[str, Any]] = {}
        self.executions: Dict[str, RunbookExecution] = {}

        # Action handlers
        self.action_handlers: Dict[str, Callable] = {
            "command": self._execute_command,
            "api": self._execute_api_call,
            "script": self._execute_script,
            "manual": self._execute_manual_step,
            "decision": self._execute_decision,
            "notification": self._send_notification
        }

        # AI-enhanced features
        self.enable_ai_suggestions = self.config.get('enable_ai', True)
        self.enable_auto_recovery = self.config.get('auto_recovery', True)

        self.logger.info("Runbook executor initialized")

    def load_runbook(self, runbook_path: str) -> Dict[str, Any]:
        """
        Load runbook from YAML file

        Args:
            runbook_path: Path to runbook YAML

        Returns:
            Loaded runbook
        """

        try:
            with open(runbook_path, 'r') as f:
                runbook = yaml.safe_load(f)

            runbook_id = runbook['metadata']['id']
            self.runbooks[runbook_id] = runbook

            self.logger.info(f"Loaded runbook {runbook_id}: {runbook['metadata']['name']}")
            return runbook

        except Exception as e:
            self.logger.error(f"Failed to load runbook: {e}")
            raise

    def execute_runbook(self, runbook_id: str, context: Optional[Dict[str, Any]] = None,
                       triggered_by: str = "system") -> RunbookExecution:
        """
        Execute runbook

        Args:
            runbook_id: Runbook identifier
            context: Execution context
            triggered_by: Who triggered the execution

        Returns:
            Runbook execution
        """

        runbook = self.runbooks.get(runbook_id)
        if not runbook:
            raise ValueError(f"Runbook not found: {runbook_id}")

        execution = RunbookExecution(
            execution_id=f"exec-{datetime.now().timestamp()}",
            runbook_id=runbook_id,
            status="running",
            started_at=datetime.now(),
            completed_at=None,
            steps_completed=0,
            steps_failed=0,
            triggered_by=triggered_by,
            context=context or {}
        )

        self.executions[execution.execution_id] = execution

        self.logger.info(
            f"Executing runbook {runbook_id} ({runbook['metadata']['name']})"
        )

        try:
            # Execute steps in order
            steps = runbook['steps']
            step_results = {}

            for step in steps:
                step_obj = RunbookStep(
                    step_id=step['id'],
                    name=step['name'],
                    description=step['description'],
                    action_type=step['action_type'],
                    action=step['action'],
                    parameters=step.get('parameters', {}),
                    expected_result=step.get('expected_result'),
                    timeout=step.get('timeout', 300),
                    retry_count=step.get('retry_count', 0),
                    dependencies=step.get('dependencies', []),
                    condition=step.get('condition')
                )

                # Check dependencies
                if not self._check_dependencies(step_obj, step_results):
                    self.logger.warning(f"Skipping step {step_obj.step_id} due to failed dependencies")
                    step_results[step_obj.step_id] = {"status": StepStatus.SKIPPED}
                    continue

                # Evaluate condition
                if step_obj.condition and not self._evaluate_condition(step_obj.condition, execution.context):
                    self.logger.info(f"Skipping step {step_obj.step_id} due to condition")
                    step_results[step_obj.step_id] = {"status": StepStatus.SKIPPED}
                    continue

                # Execute step
                result = self._execute_step(step_obj, execution)
                step_results[step_obj.step_id] = result

                if result['status'] == StepStatus.SUCCESS:
                    execution.steps_completed += 1
                    # Update context with step outputs
                    execution.context.update(result.get('outputs', {}))
                elif result['status'] == StepStatus.FAILED:
                    execution.steps_failed += 1

                    # Check if critical step
                    if step.get('critical', False):
                        self.logger.error(f"Critical step {step_obj.step_id} failed, halting execution")
                        execution.status = "failed"
                        break

                    # AI-enhanced recovery
                    if self.enable_auto_recovery:
                        recovery_action = self._suggest_recovery(step_obj, result)
                        if recovery_action:
                            self.logger.info(f"Attempting auto-recovery: {recovery_action}")
                            # Execute recovery action
                            # Placeholder for recovery logic

            # Complete execution
            if execution.status == "running":
                execution.status = "success" if execution.steps_failed == 0 else "partial_success"

            execution.completed_at = datetime.now()

            self.logger.info(
                f"Runbook execution {execution.execution_id} completed: {execution.status}"
            )

        except Exception as e:
            execution.status = "failed"
            execution.completed_at = datetime.now()
            self.logger.error(f"Runbook execution failed: {e}")

        return execution

    def _execute_step(self, step: RunbookStep, execution: RunbookExecution) -> Dict[str, Any]:
        """Execute a single runbook step"""

        self.logger.info(f"Executing step: {step.name}")

        result = {
            "step_id": step.step_id,
            "status": StepStatus.PENDING,
            "started_at": datetime.now(),
            "outputs": {},
            "error": None
        }

        try:
            # Get action handler
            handler = self.action_handlers.get(step.action_type)
            if not handler:
                raise ValueError(f"Unknown action type: {step.action_type}")

            # Execute with retry logic
            for attempt in range(step.retry_count + 1):
                try:
                    output = handler(step, execution.context)
                    result['outputs'] = output
                    result['status'] = StepStatus.SUCCESS
                    break

                except Exception as e:
                    if attempt < step.retry_count:
                        self.logger.warning(f"Step failed, retrying... (attempt {attempt + 1})")
                        continue
                    else:
                        raise

        except Exception as e:
            result['status'] = StepStatus.FAILED
            result['error'] = str(e)
            self.logger.error(f"Step {step.step_id} failed: {e}")

        result['completed_at'] = datetime.now()

        return result

    def _execute_command(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute shell command"""

        import subprocess

        command = step.action
        # Substitute variables from context
        for key, value in context.items():
            command = command.replace(f"${{{key}}}", str(value))

        self.logger.info(f"Executing command: {command}")

        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=step.timeout
        )

        if result.returncode != 0:
            raise Exception(f"Command failed: {result.stderr}")

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }

    def _execute_api_call(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute API call"""

        import requests

        url = step.action
        method = step.parameters.get('method', 'GET')
        headers = step.parameters.get('headers', {})
        data = step.parameters.get('data')

        self.logger.info(f"Executing API call: {method} {url}")

        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            timeout=step.timeout
        )

        response.raise_for_status()

        return {
            "status_code": response.status_code,
            "body": response.json() if response.text else None
        }

    def _execute_script(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute script"""

        import subprocess

        script_path = step.action
        args = step.parameters.get('args', [])

        self.logger.info(f"Executing script: {script_path}")

        result = subprocess.run(
            [script_path] + args,
            capture_output=True,
            text=True,
            timeout=step.timeout
        )

        if result.returncode != 0:
            raise Exception(f"Script failed: {result.stderr}")

        return {
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    def _execute_manual_step(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute manual step (requires human intervention)"""

        self.logger.info(f"Manual step: {step.description}")

        # In real implementation, would create ticket/notification
        # and wait for confirmation

        return {
            "message": "Manual step completed",
            "action_required": step.description
        }

    def _execute_decision(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decision step"""

        condition = step.parameters.get('condition')
        decision = self._evaluate_condition(condition, context)

        self.logger.info(f"Decision step result: {decision}")

        return {
            "decision": decision,
            "condition": condition
        }

    def _send_notification(self, step: RunbookStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification"""

        message = step.action
        # Substitute variables
        for key, value in context.items():
            message = message.replace(f"${{{key}}}", str(value))

        channel = step.parameters.get('channel', 'email')
        recipients = step.parameters.get('recipients', [])

        self.logger.info(f"Sending notification to {channel}: {recipients}")

        # Placeholder for actual notification
        return {
            "sent": True,
            "channel": channel,
            "recipients": recipients
        }

    def _check_dependencies(self, step: RunbookStep, step_results: Dict[str, Any]) -> bool:
        """Check if step dependencies are satisfied"""

        for dep_id in step.dependencies:
            if dep_id not in step_results:
                return False

            if step_results[dep_id].get('status') != StepStatus.SUCCESS:
                return False

        return True

    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate condition expression"""

        # Simple condition evaluation
        # Real implementation would use expression parser

        # Replace variables in condition
        for key, value in context.items():
            condition = condition.replace(f"${{{key}}}", str(value))

        try:
            # Safely evaluate condition
            return eval(condition)
        except:
            return False

    def _suggest_recovery(self, step: RunbookStep, result: Dict[str, Any]) -> Optional[str]:
        """AI-enhanced recovery suggestion"""

        error = result.get('error', '')

        # Simple heuristics - could be enhanced with ML
        recovery_patterns = {
            "connection refused": "restart_service",
            "timeout": "retry_with_backoff",
            "permission denied": "escalate_privileges",
            "not found": "recreate_resource"
        }

        for pattern, action in recovery_patterns.items():
            if pattern in error.lower():
                return action

        return None

    def generate_runbook_from_incident(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI-assisted runbook generation from incident

        Args:
            incident: Incident details

        Returns:
            Generated runbook
        """

        self.logger.info(f"Generating runbook for incident: {incident.get('title')}")

        # AI-enhanced runbook generation
        # This would use LLM or knowledge base in production

        runbook = {
            "metadata": {
                "id": f"incident-{incident['id']}-runbook",
                "name": f"Runbook for {incident['title']}",
                "type": "incident_response",
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "generated_from_incident": incident['id']
            },
            "steps": [
                {
                    "id": "step-1",
                    "name": "Identify affected systems",
                    "description": "Identify all systems affected by the incident",
                    "action_type": "command",
                    "action": "kubectl get pods -l app=${service_name} --all-namespaces",
                    "parameters": {}
                },
                {
                    "id": "step-2",
                    "name": "Check service health",
                    "description": "Verify health status of affected services",
                    "action_type": "api",
                    "action": "https://api.example.com/health",
                    "parameters": {"method": "GET"},
                    "dependencies": ["step-1"]
                },
                {
                    "id": "step-3",
                    "name": "Collect logs",
                    "description": "Collect logs from affected systems",
                    "action_type": "command",
                    "action": "kubectl logs -l app=${service_name} --tail=1000",
                    "parameters": {}
                },
                {
                    "id": "step-4",
                    "name": "Attempt automatic recovery",
                    "description": "Restart affected services",
                    "action_type": "command",
                    "action": "kubectl rollout restart deployment/${service_name}",
                    "parameters": {},
                    "dependencies": ["step-2"]
                },
                {
                    "id": "step-5",
                    "name": "Verify recovery",
                    "description": "Verify that services are healthy after recovery",
                    "action_type": "api",
                    "action": "https://api.example.com/health",
                    "parameters": {"method": "GET"},
                    "dependencies": ["step-4"]
                },
                {
                    "id": "step-6",
                    "name": "Notify stakeholders",
                    "description": "Notify stakeholders of resolution",
                    "action_type": "notification",
                    "action": "Incident ${incident_id} has been resolved",
                    "parameters": {
                        "channel": "slack",
                        "recipients": ["#incidents"]
                    },
                    "dependencies": ["step-5"]
                }
            ]
        }

        # Save generated runbook
        runbook_id = runbook['metadata']['id']
        self.runbooks[runbook_id] = runbook

        self.logger.info(f"Generated runbook {runbook_id}")

        return runbook

    def export_metrics(self) -> Dict[str, Any]:
        """Export runbook execution metrics"""

        total_executions = len(self.executions)
        successful_executions = sum(
            1 for e in self.executions.values()
            if e.status == "success"
        )

        total_steps = sum(
            e.steps_completed + e.steps_failed
            for e in self.executions.values()
        )

        return {
            "total_runbooks": len(self.runbooks),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": total_executions - successful_executions,
            "success_rate": successful_executions / max(1, total_executions),
            "total_steps_executed": total_steps,
            "ai_suggestions_enabled": self.enable_ai_suggestions,
            "auto_recovery_enabled": self.enable_auto_recovery
        }


# Example runbook YAML
EXAMPLE_RUNBOOK = """
metadata:
  id: db-failover
  name: Database Failover
  type: incident_response
  version: 1.0

steps:
  - id: step-1
    name: Detect primary database failure
    description: Check if primary database is responding
    action_type: command
    action: pg_isready -h ${primary_host} -p 5432
    timeout: 10
    retry_count: 3

  - id: step-2
    name: Promote standby to primary
    description: Promote standby database to primary role
    action_type: command
    action: pg_ctl promote -D /var/lib/postgresql/data
    critical: true
    dependencies: [step-1]

  - id: step-3
    name: Update DNS records
    description: Update DNS to point to new primary
    action_type: api
    action: https://api.dns.com/records/${db_record_id}
    parameters:
      method: PUT
      data:
        target: ${new_primary_ip}
    dependencies: [step-2]

  - id: step-4
    name: Notify team
    description: Notify on-call team of failover
    action_type: notification
    action: "Database failover completed: ${primary_host} -> ${new_primary_host}"
    parameters:
      channel: slack
      recipients: ["#database-team"]
    dependencies: [step-3]
"""

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize executor
    executor = RunbookExecutor({
        'enable_ai': True,
        'auto_recovery': True
    })

    # Save example runbook
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(EXAMPLE_RUNBOOK)
        runbook_path = f.name

    # Load runbook
    runbook = executor.load_runbook(runbook_path)
    print(f"\nLoaded runbook: {runbook['metadata']['name']}")

    # Execute runbook
    execution = executor.execute_runbook(
        runbook_id="db-failover",
        context={
            "primary_host": "db-primary.example.com",
            "new_primary_host": "db-standby.example.com",
            "new_primary_ip": "10.0.1.2",
            "db_record_id": "12345"
        },
        triggered_by="monitoring-system"
    )

    print(f"\nExecution: {execution.execution_id}")
    print(f"Status: {execution.status}")
    print(f"Steps completed: {execution.steps_completed}")
    print(f"Steps failed: {execution.steps_failed}")

    # Generate runbook from incident
    incident = {
        "id": "INC-001",
        "title": "Service Unavailable",
        "description": "API service returning 503 errors",
        "service_name": "api-service"
    }

    generated_runbook = executor.generate_runbook_from_incident(incident)
    print(f"\nGenerated runbook: {generated_runbook['metadata']['name']}")
    print(f"Steps: {len(generated_runbook['steps'])}")

    # Export metrics
    metrics = executor.export_metrics()
    print(f"\nRunbook Executor Metrics:")
    print(json.dumps(metrics, indent=2))

    # Cleanup
    import os
    os.unlink(runbook_path)
