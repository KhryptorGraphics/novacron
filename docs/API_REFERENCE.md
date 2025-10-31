# NovaCron API Reference

**Version**: 2.0.0  
**Base URL**: `https://api.novacron.com/v1`  
**Authentication**: Bearer Token (JWT)

---

## 🔐 Authentication

### Obtain Access Token

```http
POST /auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password",
  "mfa_code": "123456"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600,
  "token_type": "Bearer"
}
```

### Use Token

```http
GET /vms
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

---

## 🖥️ VM Management API

### List VMs

```http
GET /vms?page=1&limit=50&provider=aws&status=running
Authorization: Bearer <token>
```

**Response**:
```json
{
  "vms": [
    {
      "id": "vm-123456",
      "name": "web-server-1",
      "provider": "aws",
      "region": "us-east-1",
      "status": "running",
      "cpu": 4,
      "memory": 8589934592,
      "disk": 107374182400,
      "public_ip": "54.123.45.67",
      "private_ip": "10.0.1.100",
      "created_at": "2025-10-31T10:00:00Z",
      "updated_at": "2025-10-31T12:00:00Z"
    }
  ],
  "total": 150,
  "page": 1,
  "limit": 50
}
```

### Create VM

```http
POST /vms
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "web-server-2",
  "provider": "aws",
  "region": "us-east-1",
  "cpu": 4,
  "memory": 8589934592,
  "disk": 107374182400,
  "image": "ubuntu-22.04",
  "network": "vpc-123456",
  "tags": {
    "environment": "production",
    "team": "backend"
  }
}
```

**Response**:
```json
{
  "id": "vm-789012",
  "name": "web-server-2",
  "status": "pending",
  "created_at": "2025-10-31T14:00:00Z"
}
```

### Get VM Details

```http
GET /vms/{vm_id}
Authorization: Bearer <token>
```

### Update VM

```http
PATCH /vms/{vm_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "cpu": 8,
  "memory": 17179869184,
  "tags": {
    "environment": "production",
    "team": "backend",
    "updated": "true"
  }
}
```

### Delete VM

```http
DELETE /vms/{vm_id}
Authorization: Bearer <token>
```

### Start/Stop VM

```http
POST /vms/{vm_id}/start
POST /vms/{vm_id}/stop
POST /vms/{vm_id}/restart
Authorization: Bearer <token>
```

---

## 🔄 Migration API

### Start Live Migration

```http
POST /vms/{vm_id}/migrate
Authorization: Bearer <token>
Content-Type: application/json

{
  "target_provider": "azure",
  "target_region": "eastus",
  "migration_type": "live",
  "enable_wan_optimization": true,
  "max_downtime": "1s"
}
```

**Response**:
```json
{
  "migration_id": "mig-456789",
  "vm_id": "vm-123456",
  "status": "in_progress",
  "phase": "pre-copy",
  "started_at": "2025-10-31T15:00:00Z",
  "estimated_completion": "2025-10-31T15:30:00Z"
}
```

### Get Migration Status

```http
GET /migrations/{migration_id}
Authorization: Bearer <token>
```

**Response**:
```json
{
  "migration_id": "mig-456789",
  "vm_id": "vm-123456",
  "status": "completed",
  "phase": "complete",
  "source_provider": "aws",
  "target_provider": "azure",
  "started_at": "2025-10-31T15:00:00Z",
  "completed_at": "2025-10-31T15:25:00Z",
  "downtime": "450ms",
  "data_transferred": 85899345920,
  "compression_ratio": 0.6
}
```

### List Migrations

```http
GET /migrations?status=completed&page=1&limit=20
Authorization: Bearer <token>
```

---

## ☁️ Multi-Cloud API

### List Cloud Providers

```http
GET /providers
Authorization: Bearer <token>
```

**Response**:
```json
{
  "providers": [
    {
      "name": "aws",
      "regions": ["us-east-1", "us-west-2", "eu-west-1"],
      "available": true,
      "resources": {
        "total_cpu": 1000,
        "available_cpu": 450,
        "total_memory": 2199023255552,
        "available_memory": 1099511627776
      },
      "pricing": {
        "cpu_per_hour": 0.05,
        "memory_per_gb": 0.01,
        "currency": "USD"
      }
    }
  ]
}
```

### Get Optimal Provider

```http
POST /providers/optimal
Authorization: Bearer <token>
Content-Type: application/json

{
  "cpu": 4,
  "memory": 8589934592,
  "disk": 107374182400,
  "optimize_for": "cost"
}
```

**Response**:
```json
{
  "provider": "aws",
  "region": "us-east-1",
  "score": 0.92,
  "estimated_cost": 0.25,
  "reasoning": [
    "Lowest cost option",
    "High resource availability",
    "Low latency to target region"
  ]
}
```

---

## 🌐 Edge Computing API

### List Edge Nodes

```http
GET /edge/nodes
Authorization: Bearer <token>
```

**Response**:
```json
{
  "nodes": [
    {
      "id": "edge-001",
      "name": "edge-us-east-1",
      "location": {
        "latitude": 40.7128,
        "longitude": -74.0060,
        "city": "New York",
        "country": "USA"
      },
      "status": "online",
      "resources": {
        "cpu": 16,
        "memory": 34359738368,
        "disk": 1099511627776,
        "gpu": 1
      },
      "workloads": 5,
      "last_heartbeat": "2025-10-31T16:00:00Z"
    }
  ]
}
```

### Deploy Workload to Edge

```http
POST /edge/workloads
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "ai-inference",
  "type": "ai-inference",
  "image": "novacron/ai-model:latest",
  "resources": {
    "cpu": 2,
    "memory": 4294967296,
    "gpu": 1
  },
  "target_location": {
    "latitude": 40.7128,
    "longitude": -74.0060
  }
}
```

---

## 🔒 Security API

### List Users

```http
GET /users
Authorization: Bearer <token>
```

### Create User

```http
POST /users
Authorization: Bearer <token>
Content-Type: application/json

{
  "username": "newuser",
  "email": "newuser@example.com",
  "roles": ["operator"],
  "mfa_enabled": true
}
```

### Assign Role

```http
POST /users/{user_id}/roles
Authorization: Bearer <token>
Content-Type: application/json

{
  "role_id": "role-operator"
}
```

### Get Audit Logs

```http
GET /audit/logs?user_id=user-123&action=vm.create&start_time=2025-10-01&end_time=2025-10-31
Authorization: Bearer <token>
```

**Response**:
```json
{
  "events": [
    {
      "id": "audit-123456",
      "action": "vm.create",
      "user_id": "user-123",
      "resource": "vm",
      "resource_id": "vm-789012",
      "timestamp": "2025-10-31T14:00:00Z",
      "success": true,
      "ip_address": "203.0.113.42"
    }
  ],
  "total": 1250,
  "page": 1
}
```

---

## 📊 Metrics API

### Get System Metrics

```http
GET /metrics/system
Authorization: Bearer <token>
```

**Response**:
```json
{
  "timestamp": "2025-10-31T16:00:00Z",
  "cpu_usage": 0.65,
  "memory_usage": 0.72,
  "disk_usage": 0.45,
  "network_in": 1073741824,
  "network_out": 536870912,
  "active_vms": 150,
  "active_migrations": 3
}
```

### Get VM Metrics

```http
GET /vms/{vm_id}/metrics?start_time=2025-10-31T00:00:00Z&end_time=2025-10-31T23:59:59Z
Authorization: Bearer <token>
```

---

## 🤖 Smart Agent API

### Analyze Task

```http
POST /agents/analyze
Authorization: Bearer <token>
Content-Type: application/json

{
  "task": "Implement OAuth authentication with Google",
  "files": ["backend/auth/oauth.go", "frontend/components/Login.tsx"]
}
```

**Response**:
```json
{
  "complexity": "very-complex",
  "score": 4,
  "confidence": 0.95,
  "recommended_agents": ["coordinator", "architect", "coder", "tester", "security-auditor"],
  "topology": "adaptive",
  "estimated_duration": 60,
  "reasoning": [
    "High complexity keywords detected",
    "Security considerations needed",
    "Multiple technologies: go, react"
  ]
}
```

### Start Auto-Spawning

```http
POST /agents/spawn
Authorization: Bearer <token>
Content-Type: application/json

{
  "task": "Implement OAuth authentication",
  "files": ["backend/auth/oauth.go"],
  "max_agents": 8
}
```

---

## 📝 Error Responses

### Error Format

```json
{
  "error": {
    "code": "RESOURCE_NOT_FOUND",
    "message": "VM with ID vm-123456 not found",
    "details": {
      "vm_id": "vm-123456"
    },
    "timestamp": "2025-10-31T16:00:00Z"
  }
}
```

### Common Error Codes

- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `409` - Conflict
- `429` - Too Many Requests
- `500` - Internal Server Error
- `503` - Service Unavailable

---

## 🔄 Rate Limiting

- **Default**: 1000 requests per hour
- **Burst**: 100 requests per minute
- **Headers**:
  - `X-RateLimit-Limit`: Total limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

---

**API Version**: 2.0.0  
**Last Updated**: 2025-10-31

