"""
NovaCron Python SDK Client
"""

import asyncio
import json
import time
import redis.asyncio as redis
from typing import Dict, List, Optional, Union, AsyncIterator, Callable
from urllib.parse import urljoin
import aiohttp
import websockets
from datetime import datetime, timedelta
import logging
import uuid
from enum import Enum
import backoff

from .models import (
    VM,
    VMMetrics,
    VMTemplate,
    Migration,
    Node,
    CreateVMRequest,
    UpdateVMRequest,
    MigrationRequest,
)
from .exceptions import (
    NovaCronException,
    AuthenticationError,
    ResourceNotFoundError,
    ValidationError,
    APIError,
)


class NovaCronClient:
    """
    Asynchronous client for NovaCron API
    """

    def __init__(
        self,
        base_url: str,
        api_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        redis_url: Optional[str] = None,
        cache_ttl: int = 300,
        enable_ai_features: bool = False,
        cloud_provider: str = "local",
        region: Optional[str] = None,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60,
    ):
        """
        Initialize NovaCron client
        
        Args:
            base_url: Base URL for NovaCron API (e.g., "http://localhost:8090")
            api_token: JWT token for authentication
            username: Username for basic auth
            password: Password for basic auth
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay between retries in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.username = username
        self.password = password
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.redis_url = redis_url
        self.cache_ttl = cache_ttl
        self.enable_ai_features = enable_ai_features
        self.cloud_provider = cloud_provider
        self.region = region
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._redis: Optional[redis.Redis] = None
        self._token_refresh_task: Optional[asyncio.Task] = None
        self._token_expires_at: Optional[datetime] = None
        self._circuit_breaker_failures: Dict[str, int] = {}
        self._circuit_breaker_last_failure: Dict[str, datetime] = {}
        self.logger = logging.getLogger(f"novacron.{self.__class__.__name__}")

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def _ensure_session(self):
        """Ensure HTTP session exists"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)

    async def close(self):
        """Close HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests"""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """
        Make HTTP request with retry logic
        """
        await self._ensure_session()
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        headers = self._get_headers()

        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.request(
                    method,
                    url,
                    headers=headers,
                    json=data,
                    params=params,
                    auth=aiohttp.BasicAuth(self.username, self.password) if self.username else None,
                ) as response:
                    if response.status == 401:
                        raise AuthenticationError("Invalid credentials")
                    elif response.status == 404:
                        raise ResourceNotFoundError("Resource not found")
                    elif response.status == 422:
                        error_data = await response.json()
                        raise ValidationError(error_data.get("message", "Validation error"))
                    elif not (200 <= response.status < 300):
                        error_text = await response.text()
                        raise APIError(f"HTTP {response.status}: {error_text}")

                    if response.content_length == 0:
                        return {}
                    
                    return await response.json()

            except aiohttp.ClientError as e:
                if attempt == self.max_retries:
                    raise APIError(f"Request failed: {e}")
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    # VM Management Methods

    async def create_vm(self, request: CreateVMRequest) -> VM:
        """
        Create a new VM
        
        Args:
            request: VM creation request
            
        Returns:
            Created VM object
        """
        data = await self._request("POST", "/api/vms", request.dict())
        return VM.from_dict(data)

    async def get_vm(self, vm_id: str) -> VM:
        """
        Get VM by ID
        
        Args:
            vm_id: VM identifier
            
        Returns:
            VM object
        """
        data = await self._request("GET", f"/api/vms/{vm_id}")
        return VM.from_dict(data)

    async def list_vms(
        self,
        tenant_id: Optional[str] = None,
        state: Optional[str] = None,
        node_id: Optional[str] = None,
    ) -> List[VM]:
        """
        List VMs with optional filtering
        
        Args:
            tenant_id: Filter by tenant ID
            state: Filter by VM state
            node_id: Filter by node ID
            
        Returns:
            List of VM objects
        """
        params = {}
        if tenant_id:
            params["tenant_id"] = tenant_id
        if state:
            params["state"] = state
        if node_id:
            params["node_id"] = node_id

        data = await self._request("GET", "/api/vms", params=params)
        return [VM.from_dict(vm_data) for vm_data in data]

    async def update_vm(self, vm_id: str, updates: UpdateVMRequest) -> VM:
        """
        Update VM configuration
        
        Args:
            vm_id: VM identifier
            updates: Update request
            
        Returns:
            Updated VM object
        """
        data = await self._request("PUT", f"/api/vms/{vm_id}", updates.dict())
        return VM.from_dict(data)

    async def delete_vm(self, vm_id: str) -> None:
        """
        Delete VM
        
        Args:
            vm_id: VM identifier
        """
        await self._request("DELETE", f"/api/vms/{vm_id}")

    # VM Lifecycle Methods

    async def start_vm(self, vm_id: str) -> None:
        """Start VM"""
        await self._request("POST", f"/api/vms/{vm_id}/start")

    async def stop_vm(self, vm_id: str, force: bool = False) -> None:
        """
        Stop VM
        
        Args:
            vm_id: VM identifier
            force: Force stop if graceful stop fails
        """
        data = {"force": force} if force else None
        await self._request("POST", f"/api/vms/{vm_id}/stop", data)

    async def restart_vm(self, vm_id: str) -> None:
        """Restart VM"""
        await self._request("POST", f"/api/vms/{vm_id}/restart")

    async def pause_vm(self, vm_id: str) -> None:
        """Pause VM"""
        await self._request("POST", f"/api/vms/{vm_id}/pause")

    async def resume_vm(self, vm_id: str) -> None:
        """Resume paused VM"""
        await self._request("POST", f"/api/vms/{vm_id}/resume")

    # Metrics and Monitoring

    async def get_vm_metrics(
        self,
        vm_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> VMMetrics:
        """
        Get VM metrics
        
        Args:
            vm_id: VM identifier
            start_time: Start time for metrics range
            end_time: End time for metrics range
            
        Returns:
            VM metrics
        """
        params = {}
        if start_time:
            params["start"] = start_time.isoformat()
        if end_time:
            params["end"] = end_time.isoformat()

        data = await self._request("GET", f"/api/vms/{vm_id}/metrics", params=params)
        return VMMetrics.from_dict(data)

    async def get_system_metrics(
        self,
        node_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """
        Get system-wide metrics
        
        Args:
            node_id: Node identifier
            start_time: Start time for metrics range
            end_time: End time for metrics range
            
        Returns:
            System metrics
        """
        params = {}
        if node_id:
            params["node_id"] = node_id
        if start_time:
            params["start"] = start_time.isoformat()
        if end_time:
            params["end"] = end_time.isoformat()

        return await self._request("GET", "/api/metrics/system", params=params)

    # Migration Methods

    async def migrate_vm(self, vm_id: str, request: MigrationRequest) -> Migration:
        """
        Migrate VM to another node
        
        Args:
            vm_id: VM identifier
            request: Migration request
            
        Returns:
            Migration object
        """
        data = await self._request("POST", f"/api/vms/{vm_id}/migrate", request.dict())
        return Migration.from_dict(data)

    async def get_migration(self, migration_id: str) -> Migration:
        """Get migration status"""
        data = await self._request("GET", f"/api/migrations/{migration_id}")
        return Migration.from_dict(data)

    async def list_migrations(
        self,
        vm_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Migration]:
        """
        List migrations
        
        Args:
            vm_id: Filter by VM ID
            status: Filter by migration status
            
        Returns:
            List of migrations
        """
        params = {}
        if vm_id:
            params["vm_id"] = vm_id
        if status:
            params["status"] = status

        data = await self._request("GET", "/api/migrations", params=params)
        return [Migration.from_dict(migration_data) for migration_data in data]

    async def cancel_migration(self, migration_id: str) -> None:
        """Cancel ongoing migration"""
        await self._request("POST", f"/api/migrations/{migration_id}/cancel")

    # Template Methods

    async def create_vm_template(self, template: VMTemplate) -> VMTemplate:
        """Create VM template"""
        data = await self._request("POST", "/api/templates", template.dict())
        return VMTemplate.from_dict(data)

    async def get_vm_template(self, template_id: str) -> VMTemplate:
        """Get VM template"""
        data = await self._request("GET", f"/api/templates/{template_id}")
        return VMTemplate.from_dict(data)

    async def list_vm_templates(self) -> List[VMTemplate]:
        """List VM templates"""
        data = await self._request("GET", "/api/templates")
        return [VMTemplate.from_dict(template_data) for template_data in data]

    async def update_vm_template(self, template_id: str, template: VMTemplate) -> VMTemplate:
        """Update VM template"""
        data = await self._request("PUT", f"/api/templates/{template_id}", template.dict())
        return VMTemplate.from_dict(data)

    async def delete_vm_template(self, template_id: str) -> None:
        """Delete VM template"""
        await self._request("DELETE", f"/api/templates/{template_id}")

    # Node Management

    async def list_nodes(self) -> List[Node]:
        """List cluster nodes"""
        data = await self._request("GET", "/api/nodes")
        return [Node.from_dict(node_data) for node_data in data]

    async def get_node(self, node_id: str) -> Node:
        """Get node information"""
        data = await self._request("GET", f"/api/nodes/{node_id}")
        return Node.from_dict(data)

    async def get_node_metrics(
        self,
        node_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict:
        """Get node metrics"""
        params = {}
        if start_time:
            params["start"] = start_time.isoformat()
        if end_time:
            params["end"] = end_time.isoformat()

        return await self._request("GET", f"/api/nodes/{node_id}/metrics", params=params)

    # WebSocket Methods

    async def stream_vm_events(
        self, vm_id: Optional[str] = None
    ) -> AsyncIterator[Dict]:
        """
        Stream VM events via WebSocket
        
        Args:
            vm_id: VM ID to filter events (optional)
            
        Yields:
            VM event dictionaries
        """
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url += "/ws/events"
        
        if vm_id:
            ws_url += f"?vm_id={vm_id}"

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        async with websockets.connect(ws_url, extra_headers=headers) as websocket:
            async for message in websocket:
                try:
                    event = json.loads(message)
                    yield event
                except json.JSONDecodeError:
                    continue

    async def stream_metrics(
        self, vm_id: Optional[str] = None, interval: int = 5
    ) -> AsyncIterator[Dict]:
        """
        Stream real-time metrics
        
        Args:
            vm_id: VM ID for specific VM metrics
            interval: Update interval in seconds
            
        Yields:
            Metrics data dictionaries
        """
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_url += f"/ws/metrics?interval={interval}"
        
        if vm_id:
            ws_url += f"&vm_id={vm_id}"

        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        async with websockets.connect(ws_url, extra_headers=headers) as websocket:
            async for message in websocket:
                try:
                    metrics = json.loads(message)
                    yield metrics
                except json.JSONDecodeError:
                    continue

    # Health and Status

    async def health_check(self) -> Dict:
        """Check API health status"""
        return await self._request("GET", "/health")

    async def get_version(self) -> Dict:
        """Get API version information"""
        return await self._request("GET", "/version")

    # Batch Operations

    async def batch_start_vms(self, vm_ids: List[str]) -> Dict[str, bool]:
        """
        Start multiple VMs
        
        Args:
            vm_ids: List of VM identifiers
            
        Returns:
            Dictionary mapping VM ID to success status
        """
        results = {}
        tasks = [self.start_vm(vm_id) for vm_id in vm_ids]
        
        for vm_id, task in zip(vm_ids, tasks):
            try:
                await task
                results[vm_id] = True
            except Exception:
                results[vm_id] = False
        
        return results

    async def batch_stop_vms(self, vm_ids: List[str], force: bool = False) -> Dict[str, bool]:
        """
        Stop multiple VMs
        
        Args:
            vm_ids: List of VM identifiers
            force: Force stop if graceful stop fails
            
        Returns:
            Dictionary mapping VM ID to success status
        """
        results = {}
        tasks = [self.stop_vm(vm_id, force) for vm_id in vm_ids]
        
        for vm_id, task in zip(vm_ids, tasks):
            try:
                await task
                results[vm_id] = True
            except Exception:
                results[vm_id] = False
        
        return results

    # Authentication Helper

    async def authenticate(self, username: str, password: str) -> str:
        """
        Authenticate and get JWT token
        
        Args:
            username: Username
            password: Password
            
        Returns:
            JWT token
        """
        auth_data = {
            "username": username,
            "password": password,
        }
        
        response = await self._request("POST", "/api/auth/login", auth_data)
        token = response.get("token")
        
        if token:
            self.api_token = token
            return token
        else:
            raise AuthenticationError("Failed to get authentication token")

    async def refresh_token(self) -> str:
        """
        Refresh JWT token
        
        Returns:
            New JWT token
        """
        if not self.api_token:
            raise AuthenticationError("No token to refresh")

        response = await self._request("POST", "/api/auth/refresh")
        token = response.get("token")
        
        if token:
            self.api_token = token
            return token
        else:
            raise AuthenticationError("Failed to refresh authentication token")