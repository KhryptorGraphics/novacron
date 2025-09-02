# API Specification - NovaCron VM Management Platform

## Overview

**Version**: 1.0.0  
**Base URL**: `https://api.novacron.io/v1`  
**Protocol**: HTTPS with TLS 1.3  
**Authentication**: JWT Bearer tokens with RSA-256 signing

## Authentication

### POST /auth/login
Authenticate user and receive JWT token.

**Request:**
```json
{
  "email": "admin@example.com",
  "password": "SecurePassword123!",
  "mfa_code": "123456"  // Optional if MFA enabled
}
```

**Response (200 OK):**
```json
{
  "token": "eyJhbGciOiJSUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJSUzI1NiIs...",
  "expires_in": 3600,
  "user": {
    "id": "usr_abc123",
    "email": "admin@example.com",
    "role": "admin",
    "permissions": ["vm:*", "network:*", "storage:*"]
  }
}
```

### POST /auth/refresh
Refresh expired access token.

**Headers:**
```
Authorization: Bearer {refresh_token}
```

**Response (200 OK):**
```json
{
  "token": "eyJhbGciOiJSUzI1NiIs...",
  "expires_in": 3600
}
```

## Virtual Machines

### GET /vms
List all virtual machines with filtering and pagination.

**Query Parameters:**
- `page` (integer): Page number (default: 1)
- `limit` (integer): Items per page (default: 20, max: 100)
- `status` (string): Filter by status (running|stopped|suspended|error)
- `hypervisor` (string): Filter by hypervisor (kvm|vmware|hyperv|xen)
- `cloud` (string): Filter by cloud provider (aws|azure|gcp|onprem)
- `search` (string): Search by name or ID

**Response (200 OK):**
```json
{
  "data": [
    {
      "id": "vm_xyz789",
      "name": "production-web-01",
      "status": "running",
      "hypervisor": "kvm",
      "cloud_provider": "aws",
      "region": "us-east-1",
      "resources": {
        "cpu": 4,
        "memory_gb": 16,
        "storage_gb": 100
      },
      "network": {
        "private_ip": "10.0.1.10",
        "public_ip": "54.123.45.67",
        "vpc_id": "vpc-abc123"
      },
      "created_at": "2025-01-15T10:30:00Z",
      "updated_at": "2025-01-30T14:22:00Z"
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 145,
    "pages": 8
  }
}
```

### POST /vms
Create a new virtual machine.

**Request:**
```json
{
  "name": "staging-api-02",
  "template_id": "tpl_ubuntu2204",
  "hypervisor": "kvm",
  "cloud_provider": "aws",
  "region": "us-west-2",
  "resources": {
    "cpu": 2,
    "memory_gb": 8,
    "storage_gb": 50
  },
  "network": {
    "vpc_id": "vpc-def456",
    "subnet_id": "subnet-789xyz",
    "security_groups": ["sg-web", "sg-ssh"]
  },
  "metadata": {
    "environment": "staging",
    "team": "backend",
    "cost_center": "engineering"
  }
}
```

**Response (201 Created):**
```json
{
  "id": "vm_newid123",
  "name": "staging-api-02",
  "status": "provisioning",
  "provisioning_job_id": "job_prov456",
  "estimated_completion": "2025-01-30T15:35:00Z"
}
```

### GET /vms/{vm_id}
Get detailed information about a specific VM.

**Response (200 OK):**
```json
{
  "id": "vm_xyz789",
  "name": "production-web-01",
  "status": "running",
  "hypervisor": "kvm",
  "hypervisor_details": {
    "host": "hv-node-03.datacenter.local",
    "version": "7.0.0",
    "uuid": "550e8400-e29b-41d4-a716-446655440000"
  },
  "resources": {
    "cpu": 4,
    "cpu_usage_percent": 45.2,
    "memory_gb": 16,
    "memory_usage_percent": 72.8,
    "storage_gb": 100,
    "storage_usage_percent": 34.5
  },
  "network": {
    "interfaces": [
      {
        "name": "eth0",
        "mac": "52:54:00:12:34:56",
        "private_ip": "10.0.1.10",
        "public_ip": "54.123.45.67",
        "bandwidth_mbps": 1000
      }
    ]
  },
  "monitoring": {
    "metrics_endpoint": "/vms/vm_xyz789/metrics",
    "logs_endpoint": "/vms/vm_xyz789/logs",
    "alerts_active": 0
  }
}
```

### PUT /vms/{vm_id}
Update VM configuration.

**Request:**
```json
{
  "name": "production-web-01-updated",
  "resources": {
    "cpu": 8,
    "memory_gb": 32
  },
  "metadata": {
    "environment": "production",
    "scaling_enabled": true
  }
}
```

**Response (200 OK):**
```json
{
  "id": "vm_xyz789",
  "status": "updating",
  "update_job_id": "job_upd789",
  "changes_pending": ["cpu", "memory", "metadata"]
}
```

### DELETE /vms/{vm_id}
Delete a virtual machine.

**Query Parameters:**
- `force` (boolean): Force deletion even if VM is running
- `preserve_volumes` (boolean): Keep storage volumes after deletion

**Response (202 Accepted):**
```json
{
  "id": "vm_xyz789",
  "status": "deleting",
  "deletion_job_id": "job_del012",
  "estimated_completion": "2025-01-30T16:00:00Z"
}
```

## VM Operations

### POST /vms/{vm_id}/start
Start a stopped VM.

**Response (200 OK):**
```json
{
  "id": "vm_xyz789",
  "status": "starting",
  "operation_id": "op_start345"
}
```

### POST /vms/{vm_id}/stop
Stop a running VM.

**Request:**
```json
{
  "graceful": true,
  "timeout_seconds": 60
}
```

**Response (200 OK):**
```json
{
  "id": "vm_xyz789",
  "status": "stopping",
  "operation_id": "op_stop678"
}
```

### POST /vms/{vm_id}/restart
Restart a VM.

**Response (200 OK):**
```json
{
  "id": "vm_xyz789",
  "status": "restarting",
  "operation_id": "op_restart901"
}
```

### POST /vms/{vm_id}/migrate
Live migrate VM to different host.

**Request:**
```json
{
  "target_host": "hv-node-05.datacenter.local",
  "live_migration": true,
  "preserve_state": true,
  "bandwidth_limit_mbps": 100
}
```

**Response (202 Accepted):**
```json
{
  "id": "vm_xyz789",
  "status": "migrating",
  "migration_job_id": "job_mig234",
  "source_host": "hv-node-03.datacenter.local",
  "target_host": "hv-node-05.datacenter.local",
  "progress_percent": 0,
  "estimated_completion": "2025-01-30T16:30:00Z"
}
```

## Monitoring & Metrics

### GET /vms/{vm_id}/metrics
Get real-time metrics for a VM.

**Query Parameters:**
- `interval` (string): Time interval (1m|5m|1h|1d)
- `metrics` (string): Comma-separated metrics to retrieve

**Response (200 OK):**
```json
{
  "vm_id": "vm_xyz789",
  "timestamp": "2025-01-30T15:00:00Z",
  "interval": "5m",
  "metrics": {
    "cpu": {
      "usage_percent": 45.2,
      "user_percent": 35.1,
      "system_percent": 10.1,
      "iowait_percent": 2.3
    },
    "memory": {
      "total_gb": 16,
      "used_gb": 11.65,
      "free_gb": 4.35,
      "cache_gb": 2.1
    },
    "disk": {
      "read_iops": 145,
      "write_iops": 89,
      "read_mbps": 12.4,
      "write_mbps": 8.7
    },
    "network": {
      "rx_mbps": 25.6,
      "tx_mbps": 18.3,
      "rx_packets": 15234,
      "tx_packets": 12890
    }
  }
}
```

### GET /vms/{vm_id}/logs
Stream VM logs.

**Query Parameters:**
- `lines` (integer): Number of lines to retrieve
- `follow` (boolean): Stream logs in real-time
- `since` (string): ISO timestamp to start from

**Response (200 OK):**
```json
{
  "vm_id": "vm_xyz789",
  "logs": [
    {
      "timestamp": "2025-01-30T14:59:45Z",
      "level": "INFO",
      "source": "system",
      "message": "VM health check passed"
    },
    {
      "timestamp": "2025-01-30T15:00:00Z",
      "level": "WARNING",
      "source": "application",
      "message": "High memory usage detected: 85%"
    }
  ]
}
```

## Batch Operations

### POST /batch/vms
Perform operations on multiple VMs.

**Request:**
```json
{
  "operation": "stop",
  "vm_ids": ["vm_001", "vm_002", "vm_003"],
  "options": {
    "graceful": true,
    "parallel": true
  }
}
```

**Response (202 Accepted):**
```json
{
  "batch_job_id": "batch_job_567",
  "operation": "stop",
  "total_vms": 3,
  "status": "processing",
  "results": [
    {"vm_id": "vm_001", "status": "stopping"},
    {"vm_id": "vm_002", "status": "stopping"},
    {"vm_id": "vm_003", "status": "queued"}
  ]
}
```

## WebSocket Events

### WS /ws/events
Real-time event stream for VM operations.

**Connection:**
```javascript
const ws = new WebSocket('wss://api.novacron.io/v1/ws/events');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'authenticate',
    token: 'Bearer eyJhbGciOiJSUzI1NiIs...'
  }));
};
```

**Event Types:**
```json
{
  "type": "vm.status.changed",
  "timestamp": "2025-01-30T15:15:00Z",
  "data": {
    "vm_id": "vm_xyz789",
    "old_status": "running",
    "new_status": "stopped",
    "reason": "user_initiated"
  }
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Validation failed",
    "details": [
      {
        "field": "resources.cpu",
        "message": "CPU count must be between 1 and 64"
      }
    ]
  }
}
```

### 401 Unauthorized
```json
{
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired token"
  }
}
```

### 403 Forbidden
```json
{
  "error": {
    "code": "FORBIDDEN",
    "message": "Insufficient permissions for operation",
    "required_permission": "vm:delete"
  }
}
```

### 404 Not Found
```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "VM not found",
    "resource": "vm_xyz789"
  }
}
```

### 429 Too Many Requests
```json
{
  "error": {
    "code": "RATE_LIMITED",
    "message": "Rate limit exceeded",
    "retry_after": 60,
    "limit": "100 requests per minute"
  }
}
```

### 500 Internal Server Error
```json
{
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred",
    "request_id": "req_abc123xyz",
    "support_url": "https://support.novacron.io"
  }
}
```

## Rate Limiting

- **Default**: 100 requests per minute per API key
- **Burst**: 20 requests per second allowed
- **Headers**: 
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Requests remaining
  - `X-RateLimit-Reset`: Unix timestamp when limit resets

## Versioning

API versions are specified in the URL path. Breaking changes require new versions.

- **Current**: `/v1` (stable)
- **Beta**: `/v2-beta` (experimental features)
- **Deprecated**: Versions supported for 12 months after deprecation

## SDK Support

Official SDKs available:
- **Go**: `github.com/novacron/go-sdk`
- **Python**: `pip install novacron`
- **JavaScript/TypeScript**: `npm install @novacron/sdk`
- **Java**: `com.novacron:sdk:1.0.0`

---
*API Specification generated using BMad Create Doc Task*
*Date: 2025-01-30*
*Status: Draft specification for review*