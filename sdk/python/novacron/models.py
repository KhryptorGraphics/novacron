"""
NovaCron SDK Data Models
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum


class VMState(str, Enum):
    """VM state enumeration"""
    UNKNOWN = "unknown"
    CREATED = "created"
    CREATING = "creating"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    STOPPED = "stopped"
    PAUSED = "paused"
    PAUSING = "pausing"
    RESUMING = "resuming"
    RESTARTING = "restarting"
    DELETING = "deleting"
    MIGRATING = "migrating"
    FAILED = "failed"


class MigrationType(str, Enum):
    """Migration type enumeration"""
    COLD = "cold"
    WARM = "warm"
    LIVE = "live"


class MigrationStatus(str, Enum):
    """Migration status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VMConfig:
    """VM configuration"""
    cpu_shares: int = 1024
    memory_mb: int = 512
    disk_size_gb: int = 10
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    network_id: Optional[str] = None
    work_dir: Optional[str] = None
    rootfs: Optional[str] = None
    mounts: Optional[List[Dict]] = None
    tags: Optional[Dict[str, str]] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VMConfig':
        """Create from dictionary"""
        return cls(
            cpu_shares=data.get('cpu_shares', 1024),
            memory_mb=data.get('memory_mb', 512),
            disk_size_gb=data.get('disk_size_gb', 10),
            command=data.get('command'),
            args=data.get('args'),
            env=data.get('env'),
            network_id=data.get('network_id'),
            work_dir=data.get('work_dir'),
            rootfs=data.get('rootfs'),
            mounts=data.get('mounts'),
            tags=data.get('tags'),
        )


@dataclass
class VM:
    """Virtual Machine representation"""
    id: str
    name: str
    state: VMState
    node_id: Optional[str] = None
    owner_id: Optional[int] = None
    tenant_id: str = "default"
    config: Optional[VMConfig] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.config:
            result['config'] = self.config.dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VM':
        """Create from dictionary"""
        config = None
        if data.get('config'):
            config = VMConfig.from_dict(data['config'])

        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))

        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))

        return cls(
            id=data['id'],
            name=data['name'],
            state=VMState(data['state']),
            node_id=data.get('node_id'),
            owner_id=data.get('owner_id'),
            tenant_id=data.get('tenant_id', 'default'),
            config=config,
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class VMMetrics:
    """VM performance metrics"""
    vm_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_sent: int = 0
    network_recv: int = 0
    iops: int = 0
    timestamp: Optional[datetime] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VMMetrics':
        """Create from dictionary"""
        timestamp = None
        if data.get('timestamp') or data.get('last_updated'):
            timestamp_str = data.get('timestamp') or data.get('last_updated')
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        return cls(
            vm_id=data['vm_id'],
            cpu_usage=data.get('cpu_usage', 0.0),
            memory_usage=data.get('memory_usage', 0.0),
            disk_usage=data.get('disk_usage', 0.0),
            network_sent=data.get('network_sent', 0),
            network_recv=data.get('network_recv', 0),
            iops=data.get('iops', 0),
            timestamp=timestamp,
        )


@dataclass
class VMTemplate:
    """VM template for creating VMs"""
    id: Optional[str] = None
    name: str = ""
    description: Optional[str] = None
    config: Optional[VMConfig] = None
    default_node_selector: Optional[Dict[str, str]] = None
    parameters: Optional[List[Dict]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        if self.config:
            result['config'] = self.config.dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VMTemplate':
        """Create from dictionary"""
        config = None
        if data.get('config'):
            config = VMConfig.from_dict(data['config'])

        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))

        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))

        return cls(
            id=data.get('id'),
            name=data['name'],
            description=data.get('description'),
            config=config,
            default_node_selector=data.get('default_node_selector'),
            parameters=data.get('parameters'),
            created_at=created_at,
            updated_at=updated_at,
        )


@dataclass
class Migration:
    """VM migration information"""
    id: str
    vm_id: str
    source_node_id: str
    target_node_id: str
    type: MigrationType
    status: MigrationStatus
    progress: float = 0.0
    bytes_total: int = 0
    bytes_transferred: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Migration':
        """Create from dictionary"""
        def parse_datetime(date_str):
            if date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return None

        return cls(
            id=data['id'],
            vm_id=data['vm_id'],
            source_node_id=data['source_node_id'],
            target_node_id=data['target_node_id'],
            type=MigrationType(data['type']),
            status=MigrationStatus(data['status']),
            progress=data.get('progress', 0.0),
            bytes_total=data.get('bytes_total', 0),
            bytes_transferred=data.get('bytes_transferred', 0),
            started_at=parse_datetime(data.get('started_at')),
            completed_at=parse_datetime(data.get('completed_at')),
            error_message=data.get('error_message'),
            created_at=parse_datetime(data.get('created_at')),
            updated_at=parse_datetime(data.get('updated_at')),
        )


@dataclass
class Node:
    """Cluster node information"""
    id: str
    name: str
    address: str
    status: str = "unknown"
    capabilities: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None
    labels: Optional[Dict[str, str]] = None
    last_seen: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Node':
        """Create from dictionary"""
        def parse_datetime(date_str):
            if date_str:
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return None

        return cls(
            id=data['id'],
            name=data['name'],
            address=data['address'],
            status=data.get('status', 'unknown'),
            capabilities=data.get('capabilities'),
            resources=data.get('resources'),
            labels=data.get('labels'),
            last_seen=parse_datetime(data.get('last_seen')),
            created_at=parse_datetime(data.get('created_at')),
            updated_at=parse_datetime(data.get('updated_at')),
        )


# Request Models

@dataclass
class CreateVMRequest:
    """Request to create a VM"""
    name: str
    command: Optional[str] = None
    args: Optional[List[str]] = None
    cpu_shares: int = 1024
    memory_mb: int = 512
    disk_size_gb: int = 10
    tags: Optional[Dict[str, str]] = None
    tenant_id: str = "default"

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class UpdateVMRequest:
    """Request to update a VM"""
    name: Optional[str] = None
    cpu_shares: Optional[int] = None
    memory_mb: Optional[int] = None
    disk_size_gb: Optional[int] = None
    tags: Optional[Dict[str, str]] = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        return result


@dataclass
class MigrationRequest:
    """Request to migrate a VM"""
    target_node_id: str
    type: MigrationType = MigrationType.LIVE
    force: bool = False
    bandwidth_limit: Optional[int] = None
    compression: bool = True

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)