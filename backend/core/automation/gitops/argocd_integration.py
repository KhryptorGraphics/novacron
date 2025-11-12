#!/usr/bin/env python3
"""
GitOps Integration with ArgoCD
Implements declarative infrastructure management with Git-based deployments
"""

import json
import logging
import yaml
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class SyncStatus(Enum):
    """Synchronization status"""
    SYNCED = "synced"
    OUT_OF_SYNC = "out_of_sync"
    UNKNOWN = "unknown"


class HealthStatus(Enum):
    """Health status"""
    HEALTHY = "healthy"
    PROGRESSING = "progressing"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass
class Application:
    """ArgoCD application definition"""
    name: str
    namespace: str
    project: str
    source_repo_url: str
    source_path: str
    source_target_revision: str
    destination_server: str
    destination_namespace: str
    sync_policy: Dict[str, Any]
    health_status: HealthStatus
    sync_status: SyncStatus
    created_at: datetime


@dataclass
class SyncOperation:
    """Synchronization operation"""
    operation_id: str
    application_name: str
    revision: str
    initiated_by: str
    initiated_at: datetime
    status: str
    message: Optional[str] = None
    resources_synced: int = 0
    resources_failed: int = 0


@dataclass
class ApplicationResource:
    """Application resource"""
    group: str
    version: str
    kind: str
    namespace: str
    name: str
    status: str
    health: HealthStatus
    sync_status: SyncStatus


class ArgoCDIntegrator:
    """
    ArgoCD integration for GitOps-based infrastructure management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.argocd_server = self.config.get('argocd_server', 'localhost:8080')
        self.argocd_token = self.config.get('argocd_token', '')
        self.git_repo_url = self.config.get('git_repo_url', '')
        self.default_project = self.config.get('default_project', 'default')

        # Storage
        self.applications: Dict[str, Application] = {}
        self.sync_operations: List[SyncOperation] = []

        # Sync policies
        self.default_sync_policy = {
            "automated": {
                "prune": True,
                "selfHeal": True,
                "allowEmpty": False
            },
            "syncOptions": [
                "Validate=true",
                "CreateNamespace=true",
                "PrunePropagationPolicy=foreground"
            ],
            "retry": {
                "limit": 5,
                "backoff": {
                    "duration": "5s",
                    "factor": 2,
                    "maxDuration": "3m"
                }
            }
        }

        self.logger.info("ArgoCD integrator initialized")

    def create_application(self, name: str, repo_url: str, path: str,
                          target_revision: str = "HEAD",
                          destination_namespace: str = "default",
                          sync_policy: Optional[Dict[str, Any]] = None) -> Application:
        """
        Create ArgoCD application

        Args:
            name: Application name
            repo_url: Git repository URL
            path: Path in repository
            target_revision: Git revision (branch/tag/commit)
            destination_namespace: Target Kubernetes namespace
            sync_policy: Custom sync policy

        Returns:
            Created application
        """

        app = Application(
            name=name,
            namespace="argocd",
            project=self.default_project,
            source_repo_url=repo_url,
            source_path=path,
            source_target_revision=target_revision,
            destination_server="https://kubernetes.default.svc",
            destination_namespace=destination_namespace,
            sync_policy=sync_policy or self.default_sync_policy,
            health_status=HealthStatus.UNKNOWN,
            sync_status=SyncStatus.UNKNOWN,
            created_at=datetime.now()
        )

        self.applications[name] = app

        self.logger.info(
            f"Created application {name} targeting {repo_url}/{path}@{target_revision}"
        )

        # Generate ArgoCD manifest
        manifest = self._generate_application_manifest(app)
        self.logger.debug(f"Application manifest:\n{yaml.dump(manifest)}")

        return app

    def _generate_application_manifest(self, app: Application) -> Dict[str, Any]:
        """Generate ArgoCD application manifest"""

        return {
            "apiVersion": "argoproj.io/v1alpha1",
            "kind": "Application",
            "metadata": {
                "name": app.name,
                "namespace": app.namespace
            },
            "spec": {
                "project": app.project,
                "source": {
                    "repoURL": app.source_repo_url,
                    "path": app.source_path,
                    "targetRevision": app.source_target_revision
                },
                "destination": {
                    "server": app.destination_server,
                    "namespace": app.destination_namespace
                },
                "syncPolicy": app.sync_policy
            }
        }

    def sync_application(self, app_name: str, revision: Optional[str] = None,
                        prune: bool = True, dry_run: bool = False,
                        initiated_by: str = "system") -> SyncOperation:
        """
        Synchronize application with Git repository

        Args:
            app_name: Application name
            revision: Specific revision to sync (optional)
            prune: Remove resources not in Git
            dry_run: Preview changes without applying
            initiated_by: User/system initiating sync

        Returns:
            Sync operation
        """

        app = self.applications.get(app_name)
        if not app:
            raise ValueError(f"Application not found: {app_name}")

        operation = SyncOperation(
            operation_id=f"sync-{datetime.now().timestamp()}",
            application_name=app_name,
            revision=revision or app.source_target_revision,
            initiated_by=initiated_by,
            initiated_at=datetime.now(),
            status="in_progress"
        )

        self.sync_operations.append(operation)

        self.logger.info(
            f"{'Dry run' if dry_run else 'Syncing'} application {app_name} "
            f"to revision {operation.revision}"
        )

        try:
            # Get desired state from Git
            desired_resources = self._fetch_resources_from_git(
                app.source_repo_url, app.source_path, operation.revision
            )

            # Get current state from cluster
            current_resources = self._fetch_current_resources(app)

            # Calculate diff
            diff = self._calculate_diff(desired_resources, current_resources, prune)

            operation.message = f"Found {len(diff['to_create'])} resources to create, " \
                              f"{len(diff['to_update'])} to update, " \
                              f"{len(diff['to_delete'])} to delete"

            if dry_run:
                operation.status = "succeeded"
                self.logger.info(f"Dry run complete: {operation.message}")
                return operation

            # Apply changes
            results = self._apply_sync(app, diff)

            operation.resources_synced = results["synced"]
            operation.resources_failed = results["failed"]

            if results["failed"] == 0:
                operation.status = "succeeded"
                app.sync_status = SyncStatus.SYNCED
                app.health_status = HealthStatus.HEALTHY
            else:
                operation.status = "failed"
                operation.message = f"Sync completed with {results['failed']} failures"

        except Exception as e:
            operation.status = "failed"
            operation.message = str(e)
            self.logger.error(f"Sync failed: {e}")

        self.logger.info(
            f"Sync operation {operation.operation_id} completed: {operation.status}"
        )

        return operation

    def _fetch_resources_from_git(self, repo_url: str, path: str,
                                  revision: str) -> List[Dict[str, Any]]:
        """Fetch resource manifests from Git repository"""

        # Placeholder - would use git clone/fetch
        self.logger.debug(f"Fetching resources from {repo_url}/{path}@{revision}")

        # Simulated resources
        return [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "app-config", "namespace": "default"},
                "data": {"key": "value"}
            },
            {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {"name": "app", "namespace": "default"},
                "spec": {"replicas": 3}
            }
        ]

    def _fetch_current_resources(self, app: Application) -> List[Dict[str, Any]]:
        """Fetch current resources from cluster"""

        # Placeholder - would use kubectl/k8s client
        self.logger.debug(f"Fetching current resources for {app.name}")

        return [
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "app-config", "namespace": "default"},
                "data": {"key": "old_value"}  # Drift!
            }
        ]

    def _calculate_diff(self, desired: List[Dict[str, Any]],
                       current: List[Dict[str, Any]],
                       prune: bool) -> Dict[str, List]:
        """Calculate diff between desired and current state"""

        diff = {
            "to_create": [],
            "to_update": [],
            "to_delete": []
        }

        # Build resource index
        current_index = {
            self._resource_key(r): r for r in current
        }

        desired_index = {
            self._resource_key(r): r for r in desired
        }

        # Find resources to create or update
        for key, desired_resource in desired_index.items():
            if key not in current_index:
                diff["to_create"].append(desired_resource)
            else:
                current_resource = current_index[key]
                if self._resources_differ(desired_resource, current_resource):
                    diff["to_update"].append(desired_resource)

        # Find resources to delete (if pruning enabled)
        if prune:
            for key, current_resource in current_index.items():
                if key not in desired_index:
                    diff["to_delete"].append(current_resource)

        return diff

    def _resource_key(self, resource: Dict[str, Any]) -> str:
        """Generate unique key for resource"""

        metadata = resource.get("metadata", {})
        return f"{resource['kind']}/{metadata.get('namespace', 'default')}/{metadata['name']}"

    def _resources_differ(self, desired: Dict[str, Any],
                         current: Dict[str, Any]) -> bool:
        """Check if resources differ"""

        # Simple comparison - real implementation would use strategic merge
        return desired != current

    def _apply_sync(self, app: Application, diff: Dict[str, List]) -> Dict[str, int]:
        """Apply synchronization changes"""

        results = {"synced": 0, "failed": 0}

        # Create new resources
        for resource in diff["to_create"]:
            try:
                self._create_resource(resource)
                results["synced"] += 1
                self.logger.info(f"Created {self._resource_key(resource)}")
            except Exception as e:
                results["failed"] += 1
                self.logger.error(f"Failed to create resource: {e}")

        # Update existing resources
        for resource in diff["to_update"]:
            try:
                self._update_resource(resource)
                results["synced"] += 1
                self.logger.info(f"Updated {self._resource_key(resource)}")
            except Exception as e:
                results["failed"] += 1
                self.logger.error(f"Failed to update resource: {e}")

        # Delete pruned resources
        for resource in diff["to_delete"]:
            try:
                self._delete_resource(resource)
                results["synced"] += 1
                self.logger.info(f"Deleted {self._resource_key(resource)}")
            except Exception as e:
                results["failed"] += 1
                self.logger.error(f"Failed to delete resource: {e}")

        return results

    def _create_resource(self, resource: Dict[str, Any]):
        """Create Kubernetes resource"""
        # Placeholder - would use kubectl apply or k8s client
        pass

    def _update_resource(self, resource: Dict[str, Any]):
        """Update Kubernetes resource"""
        # Placeholder - would use kubectl apply or k8s client
        pass

    def _delete_resource(self, resource: Dict[str, Any]):
        """Delete Kubernetes resource"""
        # Placeholder - would use kubectl delete or k8s client
        pass

    def get_application_status(self, app_name: str) -> Dict[str, Any]:
        """Get application synchronization and health status"""

        app = self.applications.get(app_name)
        if not app:
            raise ValueError(f"Application not found: {app_name}")

        # Get resources
        resources = self._get_application_resources(app)

        # Calculate overall status
        sync_status = self._calculate_sync_status(resources)
        health_status = self._calculate_health_status(resources)

        return {
            "name": app.name,
            "sync_status": sync_status.value,
            "health_status": health_status.value,
            "source": {
                "repo_url": app.source_repo_url,
                "path": app.source_path,
                "revision": app.source_target_revision
            },
            "resources": [
                {
                    "kind": r.kind,
                    "name": r.name,
                    "sync_status": r.sync_status.value,
                    "health": r.health.value
                }
                for r in resources
            ]
        }

    def _get_application_resources(self, app: Application) -> List[ApplicationResource]:
        """Get application resources"""

        # Placeholder - would fetch from ArgoCD API
        return [
            ApplicationResource(
                group="",
                version="v1",
                kind="ConfigMap",
                namespace=app.destination_namespace,
                name="app-config",
                status="Synced",
                health=HealthStatus.HEALTHY,
                sync_status=SyncStatus.SYNCED
            ),
            ApplicationResource(
                group="apps",
                version="v1",
                kind="Deployment",
                namespace=app.destination_namespace,
                name="app",
                status="Synced",
                health=HealthStatus.HEALTHY,
                sync_status=SyncStatus.SYNCED
            )
        ]

    def _calculate_sync_status(self, resources: List[ApplicationResource]) -> SyncStatus:
        """Calculate overall sync status"""

        if all(r.sync_status == SyncStatus.SYNCED for r in resources):
            return SyncStatus.SYNCED
        elif any(r.sync_status == SyncStatus.OUT_OF_SYNC for r in resources):
            return SyncStatus.OUT_OF_SYNC
        else:
            return SyncStatus.UNKNOWN

    def _calculate_health_status(self, resources: List[ApplicationResource]) -> HealthStatus:
        """Calculate overall health status"""

        if all(r.health == HealthStatus.HEALTHY for r in resources):
            return HealthStatus.HEALTHY
        elif any(r.health == HealthStatus.DEGRADED for r in resources):
            return HealthStatus.DEGRADED
        elif any(r.health == HealthStatus.PROGRESSING for r in resources):
            return HealthStatus.PROGRESSING
        else:
            return HealthStatus.UNKNOWN

    def rollback_application(self, app_name: str, target_revision: str,
                           initiated_by: str = "system") -> SyncOperation:
        """Rollback application to previous revision"""

        self.logger.info(f"Rolling back {app_name} to {target_revision}")

        return self.sync_application(
            app_name,
            revision=target_revision,
            prune=True,
            initiated_by=initiated_by
        )

    def export_metrics(self) -> Dict[str, Any]:
        """Export GitOps metrics"""

        total_apps = len(self.applications)
        synced_apps = sum(
            1 for app in self.applications.values()
            if app.sync_status == SyncStatus.SYNCED
        )
        healthy_apps = sum(
            1 for app in self.applications.values()
            if app.health_status == HealthStatus.HEALTHY
        )

        total_syncs = len(self.sync_operations)
        successful_syncs = sum(
            1 for op in self.sync_operations
            if op.status == "succeeded"
        )

        return {
            "total_applications": total_apps,
            "synced_applications": synced_apps,
            "healthy_applications": healthy_apps,
            "out_of_sync_applications": total_apps - synced_apps,
            "total_sync_operations": total_syncs,
            "successful_syncs": successful_syncs,
            "failed_syncs": total_syncs - successful_syncs,
            "sync_success_rate": successful_syncs / max(1, total_syncs)
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize GitOps integrator
    integrator = ArgoCDIntegrator({
        'argocd_server': 'argocd.example.com',
        'git_repo_url': 'https://github.com/org/infra-configs',
        'default_project': 'novacron'
    })

    # Create application
    app = integrator.create_application(
        name="novacron-app",
        repo_url="https://github.com/org/novacron-configs",
        path="k8s/production",
        target_revision="main",
        destination_namespace="novacron-prod"
    )

    print(f"\nCreated application: {app.name}")

    # Perform sync
    sync_op = integrator.sync_application(
        app_name="novacron-app",
        initiated_by="admin",
        dry_run=False
    )

    print(f"\nSync operation: {sync_op.operation_id}")
    print(f"Status: {sync_op.status}")
    print(f"Message: {sync_op.message}")
    print(f"Resources synced: {sync_op.resources_synced}")

    # Get application status
    status = integrator.get_application_status("novacron-app")
    print(f"\nApplication status:")
    print(f"Sync: {status['sync_status']}")
    print(f"Health: {status['health_status']}")
    print(f"Resources: {len(status['resources'])}")

    # Export metrics
    metrics = integrator.export_metrics()
    print(f"\nGitOps metrics:")
    print(json.dumps(metrics, indent=2))
