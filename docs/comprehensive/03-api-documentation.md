# NovaCron API Documentation

## API Overview

The NovaCron API provides comprehensive REST, GraphQL, and WebSocket endpoints for virtual machine management, monitoring, and orchestration. All APIs follow OpenAPI 3.0 standards with consistent authentication, error handling, and response formatting.

**Base URL**: `https://api.novacron.com/api/v1`  
**Authentication**: JWT Bearer Token or API Key  
**Content-Type**: `application/json`

## Authentication

### JWT Authentication

**Login Endpoint:**
```http
POST /auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "secure_password"
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "123",
    "username": "user@example.com",
    "email": "user@example.com",
    "role": "admin",
    "tenant_id": "tenant-001"
  },
  "expires_in": 3600
}
```

**Authentication Header:**
```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### API Key Authentication

For service-to-service communication:
```http
X-API-Key: your-api-key-here
```

### Token Validation

```http
GET /auth/validate
Authorization: Bearer <token>
```

**Response:**
```json
{
  "valid": true,
  "user": {
    "id": "123",
    "username": "user@example.com",
    "role": "admin",
    "tenant_id": "tenant-001"
  },
  "expires_at": "2025-09-02T15:30:00Z"
}
```

## Virtual Machine Management API

### List Virtual Machines

```http
GET /vms
Authorization: Bearer <token>
```

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `limit` (integer): Items per page (default: 20, max: 100)
- `state` (string): Filter by VM state (running, stopped, pending, error)
- `node_id` (string): Filter by node ID
- `tenant_id` (string): Filter by tenant (admin only)
- `search` (string): Search by VM name or ID

**Response:**
```json
{
  "data": [
    {
      "id": "vm-001",
      "name": "web-server-01",
      "state": "running",
      "node_id": "node-01",
      "owner_id": "123",
      "tenant_id": "tenant-001",
      "config": {
        "cpu": 2,
        "memory": 4096,
        "disk": 50,
        "network": "default"
      },
      "created_at": "2025-09-01T10:00:00Z",
      "updated_at": "2025-09-01T10:00:00Z",
      "metrics": {
        "cpu_usage": 45.2,
        "memory_usage": 72.1,
        "disk_usage": 35.8,
        "network_rx": 1048576,
        "network_tx": 2097152
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 150,
    "pages": 8
  }
}
```

### Create Virtual Machine

```http
POST /vms
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "new-vm",
  "config": {
    "cpu": 2,
    "memory": 4096,
    "disk": 50,
    "image": "ubuntu-22.04",
    "network": "default",
    "ssh_key": "ssh-rsa AAAAB3..."
  },
  "node_id": "node-01",
  "tags": ["web", "production"]
}
```

**Response:**
```json
{
  "id": "vm-new-001",
  "name": "new-vm",
  "state": "creating",
  "node_id": "node-01",
  "owner_id": "123",
  "tenant_id": "tenant-001",
  "config": {
    "cpu": 2,
    "memory": 4096,
    "disk": 50,
    "image": "ubuntu-22.04",
    "network": "default"
  },
  "created_at": "2025-09-02T10:00:00Z",
  "updated_at": "2025-09-02T10:00:00Z",
  "job_id": "job-create-001"
}
```

### Get Virtual Machine Details

```http
GET /vms/{vm_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "vm-001",
  "name": "web-server-01",
  "state": "running",
  "node_id": "node-01",
  "owner_id": "123",
  "tenant_id": "tenant-001",
  "config": {
    "cpu": 2,
    "memory": 4096,
    "disk": 50,
    "network": "default"
  },
  "metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 72.1,
    "disk_usage": 35.8,
    "network_rx": 1048576,
    "network_tx": 2097152,
    "uptime": 86400
  },
  "network": {
    "interfaces": [
      {
        "name": "eth0",
        "ip": "192.168.1.100",
        "mac": "52:54:00:12:34:56"
      }
    ]
  },
  "storage": {
    "volumes": [
      {
        "id": "vol-001",
        "device": "/dev/vda",
        "size": 50,
        "type": "ssd"
      }
    ]
  },
  "created_at": "2025-09-01T10:00:00Z",
  "updated_at": "2025-09-01T10:00:00Z"
}
```

### VM Lifecycle Operations

#### Start VM
```http
POST /vms/{vm_id}/start
Authorization: Bearer <token>
```

#### Stop VM
```http
POST /vms/{vm_id}/stop
Authorization: Bearer <token>

{
  "force": false,
  "timeout": 60
}
```

#### Restart VM
```http
POST /vms/{vm_id}/restart
Authorization: Bearer <token>

{
  "timeout": 60
}
```

#### Delete VM
```http
DELETE /vms/{vm_id}
Authorization: Bearer <token>

{
  "force": false,
  "delete_volumes": true
}
```

**Lifecycle Response:**
```json
{
  "job_id": "job-operation-001",
  "status": "accepted",
  "message": "VM operation initiated successfully"
}
```

### VM Migration

```http
POST /vms/{vm_id}/migrate
Authorization: Bearer <token>
Content-Type: application/json

{
  "target_node": "node-02",
  "live_migration": true,
  "timeout": 300
}
```

**Response:**
```json
{
  "job_id": "job-migrate-001",
  "status": "accepted",
  "estimated_duration": 120,
  "migration_id": "migration-001"
}
```

### VM Snapshots

#### Create Snapshot
```http
POST /vms/{vm_id}/snapshots
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "pre-update-snapshot",
  "description": "Snapshot before system update",
  "include_memory": true
}
```

#### List Snapshots
```http
GET /vms/{vm_id}/snapshots
Authorization: Bearer <token>
```

#### Restore Snapshot
```http
POST /vms/{vm_id}/snapshots/{snapshot_id}/restore
Authorization: Bearer <token>
```

## Monitoring API

### System Metrics

```http
GET /monitoring/metrics
Authorization: Bearer <token>
```

**Query Parameters:**
- `from` (ISO 8601): Start time for metrics
- `to` (ISO 8601): End time for metrics  
- `resolution` (string): Metric resolution (1m, 5m, 1h, 1d)
- `metrics` (array): Specific metrics to retrieve

**Response:**
```json
{
  "system": {
    "cpu_usage": 45.2,
    "memory_usage": 72.1,
    "disk_usage": 35.8,
    "network_usage": 125.7,
    "load_average": [1.5, 1.3, 1.1]
  },
  "timeseries": {
    "cpu_usage": [
      {"timestamp": "2025-09-02T10:00:00Z", "value": 42.1},
      {"timestamp": "2025-09-02T10:01:00Z", "value": 45.2}
    ],
    "memory_usage": [
      {"timestamp": "2025-09-02T10:00:00Z", "value": 70.5},
      {"timestamp": "2025-09-02T10:01:00Z", "value": 72.1}
    ]
  }
}
```

### VM Metrics

```http
GET /monitoring/vms/{vm_id}/metrics
Authorization: Bearer <token>
```

**Response:**
```json
{
  "vm_id": "vm-001",
  "current": {
    "cpu_usage": 78.5,
    "memory_usage": 65.2,
    "disk_usage": 45.8,
    "network_rx": 1048576,
    "network_tx": 2097152,
    "iops": 150
  },
  "timeseries": {
    "cpu_usage": [...],
    "memory_usage": [...],
    "disk_io": [...],
    "network_io": [...]
  }
}
```

### Alerts

#### List Alerts
```http
GET /monitoring/alerts
Authorization: Bearer <token>
```

**Query Parameters:**
- `status` (string): Alert status (firing, resolved, silenced)
- `severity` (string): Alert severity (critical, warning, info)
- `resource` (string): Filter by resource type or ID

**Response:**
```json
{
  "data": [
    {
      "id": "alert-001",
      "name": "High CPU Usage",
      "description": "VM database-01 CPU usage exceeds 90%",
      "severity": "warning",
      "status": "firing",
      "resource": "vm-database-01",
      "started_at": "2025-09-02T14:30:00Z",
      "labels": {
        "vm": "database-01",
        "metric": "cpu",
        "threshold": "90"
      },
      "value": 92.1,
      "annotations": {
        "runbook": "https://docs.novacron.com/runbooks/high-cpu",
        "dashboard": "https://grafana.novacron.com/d/vm-dashboard"
      }
    }
  ]
}
```

#### Acknowledge Alert
```http
POST /monitoring/alerts/{alert_id}/acknowledge
Authorization: Bearer <token>
Content-Type: application/json

{
  "comment": "Investigating root cause",
  "expires_at": "2025-09-02T18:00:00Z"
}
```

## Orchestration API

### Job Management

#### List Jobs
```http
GET /orchestration/jobs
Authorization: Bearer <token>
```

**Response:**
```json
{
  "data": [
    {
      "id": "job-001",
      "type": "vm_migration",
      "status": "running",
      "progress": 65,
      "resource_id": "vm-001",
      "created_at": "2025-09-02T10:00:00Z",
      "started_at": "2025-09-02T10:01:00Z",
      "estimated_completion": "2025-09-02T10:05:00Z"
    }
  ]
}
```

#### Get Job Status
```http
GET /orchestration/jobs/{job_id}
Authorization: Bearer <token>
```

**Response:**
```json
{
  "id": "job-001",
  "type": "vm_migration",
  "status": "running",
  "progress": 65,
  "resource_id": "vm-001",
  "details": {
    "source_node": "node-01",
    "target_node": "node-02",
    "migration_type": "live",
    "data_transferred": "2.5GB",
    "remaining_data": "1.2GB"
  },
  "created_at": "2025-09-02T10:00:00Z",
  "started_at": "2025-09-02T10:01:00Z",
  "estimated_completion": "2025-09-02T10:05:00Z",
  "logs": [
    {
      "timestamp": "2025-09-02T10:01:30Z",
      "level": "info",
      "message": "Migration started successfully"
    }
  ]
}
```

#### Cancel Job
```http
POST /orchestration/jobs/{job_id}/cancel
Authorization: Bearer <token>
```

### Scheduling

#### Create Scheduled Task
```http
POST /orchestration/schedule
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "nightly-backup",
  "description": "Daily backup of all VMs",
  "schedule": "0 2 * * *",
  "enabled": true,
  "task": {
    "type": "backup",
    "config": {
      "vm_filter": "state:running",
      "retention_days": 7
    }
  }
}
```

## Storage API

### Volume Management

#### List Volumes
```http
GET /storage/volumes
Authorization: Bearer <token>
```

#### Create Volume
```http
POST /storage/volumes
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "data-volume-01",
  "size": 100,
  "type": "ssd",
  "encrypted": true
}
```

#### Attach Volume
```http
POST /storage/volumes/{volume_id}/attach
Authorization: Bearer <token>
Content-Type: application/json

{
  "vm_id": "vm-001",
  "device": "/dev/vdb"
}
```

### Backup Management

#### Create Backup
```http
POST /backups
Authorization: Bearer <token>
Content-Type: application/json

{
  "vm_id": "vm-001",
  "name": "pre-maintenance-backup",
  "type": "full",
  "compression": true,
  "encryption": true
}
```

#### List Backups
```http
GET /backups
Authorization: Bearer <token>
```

#### Restore Backup
```http
POST /backups/{backup_id}/restore
Authorization: Bearer <token>
Content-Type: application/json

{
  "target_vm": "vm-002",
  "restore_type": "full"
}
```

## Network API

### Network Management

#### List Networks
```http
GET /networks
Authorization: Bearer <token>
```

#### Create Network
```http
POST /networks
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "production-network",
  "cidr": "192.168.100.0/24",
  "gateway": "192.168.100.1",
  "dns_servers": ["8.8.8.8", "8.8.4.4"],
  "vlan_id": 100
}
```

### Security Groups

#### Create Security Group
```http
POST /security-groups
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "web-servers",
  "description": "Security group for web servers",
  "rules": [
    {
      "direction": "ingress",
      "protocol": "tcp",
      "port_range": "80-80",
      "source": "0.0.0.0/0"
    },
    {
      "direction": "ingress",
      "protocol": "tcp", 
      "port_range": "443-443",
      "source": "0.0.0.0/0"
    }
  ]
}
```

## WebSocket API

### Real-time Event Streaming

**Connection URL**: `wss://api.novacron.com/ws/events`

**Authentication**: Include JWT token in query parameter
```
wss://api.novacron.com/ws/events?token=<jwt_token>
```

### Event Types

#### VM Events
```json
{
  "type": "vm_state_change",
  "timestamp": "2025-09-02T10:00:00Z",
  "data": {
    "vm_id": "vm-001",
    "old_state": "stopped",
    "new_state": "running",
    "node_id": "node-01"
  }
}
```

#### Metric Updates
```json
{
  "type": "metrics_update",
  "timestamp": "2025-09-02T10:00:00Z",
  "data": {
    "vm_id": "vm-001",
    "metrics": {
      "cpu_usage": 75.2,
      "memory_usage": 68.1
    }
  }
}
```

#### Alert Notifications
```json
{
  "type": "alert",
  "timestamp": "2025-09-02T10:00:00Z",
  "data": {
    "alert_id": "alert-001",
    "severity": "warning",
    "message": "High CPU usage detected",
    "resource": "vm-001"
  }
}
```

#### Job Progress Updates
```json
{
  "type": "job_progress",
  "timestamp": "2025-09-02T10:00:00Z",
  "data": {
    "job_id": "job-001",
    "progress": 75,
    "status": "running",
    "estimated_completion": "2025-09-02T10:05:00Z"
  }
}
```

## GraphQL API

### Endpoint
```
POST /graphql
Authorization: Bearer <token>
```

### Schema Overview

```graphql
type Query {
  vms(filter: VMFilter, pagination: Pagination): VMConnection!
  vm(id: ID!): VM
  metrics(filter: MetricFilter): [Metric!]!
  alerts(filter: AlertFilter): [Alert!]!
  jobs(filter: JobFilter): [Job!]!
}

type Mutation {
  createVM(input: CreateVMInput!): VM!
  updateVM(id: ID!, input: UpdateVMInput!): VM!
  deleteVM(id: ID!, force: Boolean): Boolean!
  startVM(id: ID!): Job!
  stopVM(id: ID!, force: Boolean): Job!
  migrateVM(id: ID!, targetNode: String!): Job!
}

type Subscription {
  vmStateChanged(vmId: ID): VM!
  metricsUpdated(resourceId: ID): Metric!
  alertCreated: Alert!
  jobProgressUpdated(jobId: ID): Job!
}
```

### Example Queries

#### Get VM with Metrics
```graphql
query GetVMWithMetrics($vmId: ID!) {
  vm(id: $vmId) {
    id
    name
    state
    node_id
    config {
      cpu
      memory
      disk
    }
    metrics {
      cpu_usage
      memory_usage
      disk_usage
      network_rx
      network_tx
    }
    alerts {
      id
      severity
      message
    }
  }
}
```

#### Create VM Mutation
```graphql
mutation CreateVM($input: CreateVMInput!) {
  createVM(input: $input) {
    id
    name
    state
    job_id
  }
}
```

## Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "VM_NOT_FOUND",
    "message": "Virtual machine with ID 'vm-001' not found",
    "details": {
      "vm_id": "vm-001",
      "tenant_id": "tenant-001"
    },
    "timestamp": "2025-09-02T10:00:00Z",
    "request_id": "req-12345"
  }
}
```

### HTTP Status Codes

- `200 OK`: Successful request
- `201 Created`: Resource created successfully
- `400 Bad Request`: Invalid request data
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Access denied
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource conflict
- `422 Unprocessable Entity`: Validation errors
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request validation failed |
| `UNAUTHORIZED` | Authentication required |
| `FORBIDDEN` | Access denied |
| `NOT_FOUND` | Resource not found |
| `CONFLICT` | Resource already exists |
| `RATE_LIMITED` | Too many requests |
| `VM_NOT_FOUND` | Virtual machine not found |
| `VM_STATE_INVALID` | Invalid VM state for operation |
| `INSUFFICIENT_RESOURCES` | Not enough resources available |
| `MIGRATION_FAILED` | VM migration failed |
| `BACKUP_FAILED` | Backup operation failed |

## Rate Limiting

### Limits
- **Authenticated Users**: 1000 requests per hour
- **API Key**: 5000 requests per hour
- **Admin Users**: 10000 requests per hour

### Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1693660800
```

## API Versioning

### Version Header
```http
Accept: application/vnd.novacron.v1+json
```

### URL Versioning
```
https://api.novacron.com/api/v1/vms
https://api.novacron.com/api/v2/vms
```

## SDK Examples

### Go SDK
```go
import "github.com/novacron/go-sdk"

client := novacron.NewClient("your-api-key")
vms, err := client.VMs.List(ctx, novacron.VMListOptions{
    State: "running",
    Limit: 50,
})
```

### Python SDK
```python
from novacron import Client

client = Client(api_key="your-api-key")
vms = client.vms.list(state="running", limit=50)
```

### JavaScript SDK
```javascript
import { NovaCronClient } from '@novacron/js-sdk';

const client = new NovaCronClient({ apiKey: 'your-api-key' });
const vms = await client.vms.list({ state: 'running', limit: 50 });
```

---

**Document Classification**: Public API Documentation  
**Last Updated**: September 2, 2025  
**API Version**: v1.0  
**Next Review**: October 1, 2025