"""
VM Management Client

Provides high-level VM management operations for DWCP.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

from .client import Client, MessageType, VMOperation
from .exceptions import VMNotFoundError


class VMState(str, Enum):
    """VM state enumeration"""
    CREATING = "creating"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    MIGRATING = "migrating"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class NetworkInterface:
    """Network interface configuration"""
    name: str
    type: str = "virtio"
    mac: Optional[str] = None
    bridge: Optional[str] = None
    vlan: int = 0
    ip_address: Optional[str] = None
    netmask: Optional[str] = None
    bandwidth: int = 0


@dataclass
class NetworkConfig:
    """Network configuration"""
    mode: str = "bridge"
    interfaces: List[NetworkInterface] = field(default_factory=list)
    dns: List[str] = field(default_factory=list)
    gateway: Optional[str] = None
    mtu: int = 1500


@dataclass
class Affinity:
    """Node affinity configuration"""
    node_selector: Dict[str, str] = field(default_factory=dict)
    required_nodes: List[str] = field(default_factory=list)
    preferred_nodes: List[str] = field(default_factory=list)
    anti_affinity_vms: List[str] = field(default_factory=list)
    require_same_host: List[str] = field(default_factory=list)


@dataclass
class VMConfig:
    """VM configuration"""
    name: str
    memory: int  # bytes
    cpus: int
    disk: int  # bytes
    image: str
    network: NetworkConfig = field(default_factory=NetworkConfig)
    cloud_init: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    priority: int = 0
    affinity: Optional[Affinity] = None

    # Advanced features
    enable_gpu: bool = False
    gpu_type: str = ""
    enable_sr_iov: bool = False
    enable_tpm: bool = False
    enable_secure_boot: bool = False
    host_devices: List[str] = field(default_factory=list)

    # Performance tuning
    cpu_pinning: List[int] = field(default_factory=list)
    numa_nodes: List[int] = field(default_factory=list)
    huge_pages: bool = False
    io_threads: int = 0

    # Resource limits
    memory_max: int = 0
    cpu_quota: int = 0
    disk_iops_limit: int = 0
    network_bandwidth: int = 0


@dataclass
class VMMetrics:
    """VM runtime metrics"""
    cpu_usage: float
    memory_used: int
    memory_available: int
    disk_read: int
    disk_write: int
    network_rx: int
    network_tx: int
    timestamp: datetime


@dataclass
class VM:
    """Virtual machine representation"""
    id: str
    name: str
    state: VMState
    config: VMConfig
    node: str
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    metrics: Optional[VMMetrics] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class VMEvent:
    """VM state change event"""
    type: str
    vm: VM
    timestamp: datetime
    message: str


@dataclass
class MigrationOptions:
    """VM migration options"""
    live: bool = True
    offline: bool = False
    max_downtime: int = 500  # milliseconds
    bandwidth: int = 0  # bytes/sec, 0 = unlimited
    compression: bool = True
    auto_converge: bool = True
    post_copy: bool = False
    parallel: int = 4
    verify_checksum: bool = True
    encrypt_transport: bool = True


class MigrationState(str, Enum):
    """Migration state enumeration"""
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETING = "completing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class MigrationStatus:
    """Migration status"""
    id: str
    vm_id: str
    source_node: str
    target_node: str
    state: MigrationState
    progress: float  # 0-100
    bytes_total: int
    bytes_sent: int
    throughput: int  # bytes/sec
    downtime: int  # milliseconds
    started_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


@dataclass
class SnapshotOptions:
    """Snapshot options"""
    include_memory: bool = True
    description: str = ""
    quiesce: bool = True


@dataclass
class Snapshot:
    """VM snapshot"""
    id: str
    vm_id: str
    name: str
    description: str
    size: int
    created_at: datetime
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


class VMClient:
    """Client for VM management operations"""

    def __init__(self, client: Client):
        """Initialize VM client"""
        self.client = client

    async def create(self, config: VMConfig) -> VM:
        """Create a new VM"""
        req = {
            "operation": VMOperation.CREATE,
            "request": {
                "config": self._config_to_dict(config),
            },
        }

        resp = await self.client._send_request(MessageType.VM, req)
        data = json.loads(resp)

        return self._dict_to_vm(data["vm"])

    async def start(self, vm_id: str) -> None:
        """Start a VM"""
        req = {
            "operation": VMOperation.START,
            "vm_id": vm_id,
        }

        await self.client._send_request(MessageType.VM, req)

    async def stop(self, vm_id: str, force: bool = False) -> None:
        """Stop a VM"""
        req = {
            "operation": VMOperation.STOP,
            "vm_id": vm_id,
            "force": force,
        }

        await self.client._send_request(MessageType.VM, req)

    async def destroy(self, vm_id: str) -> None:
        """Destroy a VM"""
        req = {
            "operation": VMOperation.DESTROY,
            "vm_id": vm_id,
        }

        await self.client._send_request(MessageType.VM, req)

    async def get(self, vm_id: str) -> VM:
        """Get VM information"""
        req = {
            "operation": VMOperation.STATUS,
            "vm_id": vm_id,
        }

        resp = await self.client._send_request(MessageType.VM, req)
        data = json.loads(resp)

        return self._dict_to_vm(data)

    async def list(self, filters: Optional[Dict[str, str]] = None) -> List[VM]:
        """List all VMs with optional filters"""
        req = {
            "operation": VMOperation.STATUS,
            "filters": filters or {},
        }

        resp = await self.client._send_request(MessageType.VM, req)
        data = json.loads(resp)

        return [self._dict_to_vm(vm) for vm in data]

    async def watch(self, vm_id: str) -> AsyncIterator[VMEvent]:
        """Watch VM state changes"""
        stream = await self.client.new_stream()

        # Send watch request
        req = {
            "operation": "watch",
            "vm_id": vm_id,
        }
        req_bytes = json.dumps(req).encode("utf-8")
        await stream.send(req_bytes)

        try:
            while True:
                data = await stream.receive()
                event_data = json.loads(data)

                event = VMEvent(
                    type=event_data["type"],
                    vm=self._dict_to_vm(event_data["vm"]),
                    timestamp=datetime.fromisoformat(event_data["timestamp"]),
                    message=event_data["message"],
                )

                yield event
        finally:
            await stream.close()

    async def migrate(
        self,
        vm_id: str,
        target_node: str,
        options: Optional[MigrationOptions] = None,
    ) -> MigrationStatus:
        """Initiate VM migration"""
        if options is None:
            options = MigrationOptions()

        req = {
            "vm_id": vm_id,
            "target_node": target_node,
            "options": self._migration_options_to_dict(options),
        }

        resp = await self.client._send_request(MessageType.MIGRATION, req)
        data = json.loads(resp)

        return self._dict_to_migration_status(data)

    async def get_migration_status(self, migration_id: str) -> MigrationStatus:
        """Get migration status"""
        req = {
            "operation": "status",
            "migration_id": migration_id,
        }

        resp = await self.client._send_request(MessageType.MIGRATION, req)
        data = json.loads(resp)

        return self._dict_to_migration_status(data)

    async def snapshot(
        self,
        vm_id: str,
        snapshot_name: str,
        options: Optional[SnapshotOptions] = None,
    ) -> Snapshot:
        """Create VM snapshot"""
        if options is None:
            options = SnapshotOptions()

        req = {
            "vm_id": vm_id,
            "name": snapshot_name,
            "options": {
                "include_memory": options.include_memory,
                "description": options.description,
                "quiesce": options.quiesce,
            },
        }

        resp = await self.client._send_request(MessageType.SNAPSHOT, req)
        data = json.loads(resp)

        return self._dict_to_snapshot(data)

    async def list_snapshots(self, vm_id: str) -> List[Snapshot]:
        """List all snapshots for a VM"""
        req = {
            "operation": "list",
            "vm_id": vm_id,
        }

        resp = await self.client._send_request(MessageType.SNAPSHOT, req)
        data = json.loads(resp)

        return [self._dict_to_snapshot(s) for s in data]

    async def restore_snapshot(self, vm_id: str, snapshot_id: str) -> None:
        """Restore VM from snapshot"""
        req = {
            "operation": VMOperation.RESTORE,
            "vm_id": vm_id,
            "snapshot_id": snapshot_id,
        }

        await self.client._send_request(MessageType.SNAPSHOT, req)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete a snapshot"""
        req = {
            "operation": "delete",
            "snapshot_id": snapshot_id,
        }

        await self.client._send_request(MessageType.SNAPSHOT, req)

    async def get_metrics(self, vm_id: str, duration: str = "5m") -> VMMetrics:
        """Get VM metrics"""
        req = {
            "vm_id": vm_id,
            "duration": duration,
        }

        resp = await self.client._send_request(MessageType.METRICS, req)
        data = json.loads(resp)

        return self._dict_to_metrics(data)

    async def stream_metrics(
        self,
        vm_id: str,
        interval: str = "1s",
    ) -> AsyncIterator[VMMetrics]:
        """Stream real-time VM metrics"""
        stream = await self.client.new_stream()

        req = {
            "operation": "stream_metrics",
            "vm_id": vm_id,
            "interval": interval,
        }
        req_bytes = json.dumps(req).encode("utf-8")
        await stream.send(req_bytes)

        try:
            while True:
                data = await stream.receive()
                metrics_data = json.loads(data)
                yield self._dict_to_metrics(metrics_data)
        finally:
            await stream.close()

    # Helper methods for data conversion

    def _config_to_dict(self, config: VMConfig) -> Dict[str, Any]:
        """Convert VMConfig to dictionary"""
        return {
            "name": config.name,
            "memory": config.memory,
            "cpus": config.cpus,
            "disk": config.disk,
            "image": config.image,
            "network": {
                "mode": config.network.mode,
                "interfaces": [
                    {
                        "name": iface.name,
                        "type": iface.type,
                        "mac": iface.mac,
                        "bridge": iface.bridge,
                        "vlan": iface.vlan,
                        "ip_address": iface.ip_address,
                        "netmask": iface.netmask,
                        "bandwidth": iface.bandwidth,
                    }
                    for iface in config.network.interfaces
                ],
                "dns": config.network.dns,
                "gateway": config.network.gateway,
                "mtu": config.network.mtu,
            },
            "cloud_init": config.cloud_init,
            "labels": config.labels,
            "annotations": config.annotations,
            "priority": config.priority,
            "enable_gpu": config.enable_gpu,
            "gpu_type": config.gpu_type,
            "enable_sr_iov": config.enable_sr_iov,
            "enable_tpm": config.enable_tpm,
            "enable_secure_boot": config.enable_secure_boot,
        }

    def _dict_to_vm(self, data: Dict[str, Any]) -> VM:
        """Convert dictionary to VM"""
        config_data = data.get("config", {})
        network_data = config_data.get("network", {})

        return VM(
            id=data["id"],
            name=data["name"],
            state=VMState(data["state"]),
            config=VMConfig(
                name=config_data["name"],
                memory=config_data["memory"],
                cpus=config_data["cpus"],
                disk=config_data["disk"],
                image=config_data["image"],
                network=NetworkConfig(
                    mode=network_data.get("mode", "bridge"),
                    interfaces=[],
                    dns=network_data.get("dns", []),
                    gateway=network_data.get("gateway"),
                    mtu=network_data.get("mtu", 1500),
                ),
            ),
            node=data["node"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            labels=data.get("labels", {}),
            annotations=data.get("annotations", {}),
        )

    def _dict_to_metrics(self, data: Dict[str, Any]) -> VMMetrics:
        """Convert dictionary to VMMetrics"""
        return VMMetrics(
            cpu_usage=data["cpu_usage"],
            memory_used=data["memory_used"],
            memory_available=data["memory_available"],
            disk_read=data["disk_read"],
            disk_write=data["disk_write"],
            network_rx=data["network_rx"],
            network_tx=data["network_tx"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    def _dict_to_migration_status(self, data: Dict[str, Any]) -> MigrationStatus:
        """Convert dictionary to MigrationStatus"""
        return MigrationStatus(
            id=data["id"],
            vm_id=data["vm_id"],
            source_node=data["source_node"],
            target_node=data["target_node"],
            state=MigrationState(data["state"]),
            progress=data["progress"],
            bytes_total=data["bytes_total"],
            bytes_sent=data["bytes_sent"],
            throughput=data["throughput"],
            downtime=data["downtime"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            error=data.get("error"),
        )

    def _dict_to_snapshot(self, data: Dict[str, Any]) -> Snapshot:
        """Convert dictionary to Snapshot"""
        return Snapshot(
            id=data["id"],
            vm_id=data["vm_id"],
            name=data["name"],
            description=data["description"],
            size=data["size"],
            created_at=datetime.fromisoformat(data["created_at"]),
            parent=data.get("parent"),
            children=data.get("children", []),
        )

    def _migration_options_to_dict(self, options: MigrationOptions) -> Dict[str, Any]:
        """Convert MigrationOptions to dictionary"""
        return {
            "live": options.live,
            "offline": options.offline,
            "max_downtime": options.max_downtime,
            "bandwidth": options.bandwidth,
            "compression": options.compression,
            "auto_converge": options.auto_converge,
            "post_copy": options.post_copy,
            "parallel": options.parallel,
            "verify_checksum": options.verify_checksum,
            "encrypt_transport": options.encrypt_transport,
        }
