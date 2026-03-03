# DWCP v3.0 Developer Training Manual
## Building Applications on Internet-Scale Distributed Hypervisor

**Version:** 3.0.0
**Last Updated:** 2025-01-10
**Training Duration:** 3-5 Days
**Target Audience:** Software Engineers, Backend Developers, Integration Engineers
**Prerequisites:** Go/Python proficiency, REST API experience, Distributed Systems knowledge

---

## Table of Contents

1. [Training Overview](#training-overview)
2. [DWCP v3 API Deep Dive](#dwcp-v3-api-deep-dive)
3. [Integration Patterns](#integration-patterns)
4. [Component Architecture](#component-architecture)
5. [Code Examples](#code-examples)
6. [Testing Strategies](#testing-strategies)
7. [Debugging Techniques](#debugging-techniques)
8. [Best Practices](#best-practices)
9. [Advanced Topics](#advanced-topics)
10. [Hands-On Exercises](#hands-on-exercises)
11. [Certification Assessment](#certification-assessment)

---

## 1. Training Overview

### 1.1 Learning Objectives

By the end of this training, you will be able to:

- **Integrate** applications with DWCP v3.0 REST and gRPC APIs
- **Implement** custom VM migration workflows
- **Extend** DWCP v3 with custom components
- **Optimize** applications for distributed hypervisor architecture
- **Debug** complex distributed systems issues
- **Contribute** to DWCP v3 codebase
- **Design** fault-tolerant distributed applications

### 1.2 Training Schedule

#### Day 1: API Fundamentals
- **Morning:** DWCP v3 architecture review (2 hours)
- **Morning:** REST API deep dive (2 hours)
- **Afternoon:** gRPC API and protobuf (2 hours)
- **Afternoon:** Authentication and authorization (2 hours)
- **Evening:** Lab: Build simple VM manager (2 hours)

#### Day 2: Component Architecture
- **Morning:** AMST transport layer (2 hours)
- **Morning:** HDE encoding layer (2 hours)
- **Afternoon:** PBA prediction engine (2 hours)
- **Afternoon:** ASS state synchronization (2 hours)
- **Evening:** Lab: Implement custom transport plugin (2 hours)

#### Day 3: Advanced Integration
- **Morning:** ITP placement algorithms (2 hours)
- **Morning:** ACP consensus protocols (2 hours)
- **Afternoon:** Testing and debugging (2 hours)
- **Afternoon:** Performance profiling (2 hours)
- **Evening:** Lab: Build placement scheduler (2 hours)

#### Day 4: Production Development
- **Morning:** Error handling and resilience (2 hours)
- **Morning:** Observability and tracing (2 hours)
- **Afternoon:** Security best practices (2 hours)
- **Afternoon:** Code review and style guide (2 hours)
- **Evening:** Lab: Add observability to app (2 hours)

#### Day 5: Advanced Topics and Certification
- **Morning:** Contributing to DWCP v3 (2 hours)
- **Morning:** Advanced use cases (2 hours)
- **Afternoon:** Final project work (3 hours)
- **Evening:** Certification exam (2 hours)

### 1.3 Training Environment

**Required Software:**
- Go 1.21+ (primary language)
- Python 3.10+ (client libraries)
- Docker and Docker Compose
- VS Code or GoLand IDE
- Postman or curl (API testing)
- Git and GitHub account

**Training Cluster:**
- Access to shared DWCP v3 test cluster
- 3 controller nodes, 20 worker nodes
- Grafana/Jaeger dashboards
- API endpoint: `https://dwcp-training.example.com`

---

## 2. DWCP v3 API Deep Dive

### 2.1 API Architecture

DWCP v3 provides two API interfaces:

1. **REST API** (HTTP/JSON) - Management operations, queries
2. **gRPC API** (Protocol Buffers) - High-performance data plane

#### 2.1.1 REST API Overview

**Base URL:** `https://dwcp-api.example.com/v3`

**Authentication:** Bearer token (JWT)

**Key Endpoints:**
```
/cluster              # Cluster management
/nodes                # Node management
/vms                  # VM lifecycle
/migrations           # Migration operations
/placements           # Placement decisions
/metrics              # Metrics and monitoring
```

#### 2.1.2 gRPC API Overview

**Service Definition:** `api/proto/dwcp/v3/dwcp.proto`

**Key Services:**
```protobuf
service ClusterService {
  rpc GetClusterStatus(ClusterStatusRequest) returns (ClusterStatusResponse);
  rpc ListNodes(ListNodesRequest) returns (ListNodesResponse);
}

service VMService {
  rpc CreateVM(CreateVMRequest) returns (CreateVMResponse);
  rpc MigrateVM(MigrateVMRequest) returns (stream MigrationProgress);
}

service PlacementService {
  rpc OptimizePlacement(PlacementRequest) returns (PlacementResponse);
}
```

### 2.2 REST API Examples

#### 2.2.1 Authentication

```bash
# Login and get token
curl -X POST https://dwcp-api.example.com/v3/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "secret"
  }'

# Response:
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_at": "2025-01-11T12:00:00Z"
}

# Use token in subsequent requests
curl -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIs..." \
  https://dwcp-api.example.com/v3/cluster/status
```

#### 2.2.2 Cluster Operations

**Get Cluster Status:**
```bash
curl -X GET https://dwcp-api.example.com/v3/cluster/status \
  -H "Authorization: Bearer $TOKEN"

# Response:
{
  "cluster_id": "prod-cluster-1",
  "status": "healthy",
  "controllers": {
    "total": 3,
    "healthy": 3,
    "leader": "controller-1"
  },
  "workers": {
    "total": 1000,
    "ready": 985,
    "unreachable": 15
  },
  "consensus": {
    "protocol": "raft",
    "latency_ms": 50
  }
}
```

**List Nodes:**
```bash
curl -X GET https://dwcp-api.example.com/v3/nodes?status=ready&limit=10 \
  -H "Authorization: Bearer $TOKEN"

# Response:
{
  "nodes": [
    {
      "id": "worker-001",
      "type": "worker",
      "status": "ready",
      "region": "us-east-1",
      "resources": {
        "cpu": 32,
        "memory_gb": 128,
        "disk_gb": 1024
      },
      "vms": 42
    },
    ...
  ],
  "total": 985,
  "page": 1
}
```

#### 2.2.3 VM Lifecycle

**Create VM:**
```bash
curl -X POST https://dwcp-api.example.com/v3/vms \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "web-server-01",
    "image": "ubuntu-22.04",
    "resources": {
      "cpu": 4,
      "memory_gb": 8,
      "disk_gb": 50
    },
    "placement": {
      "region": "us-east-1",
      "affinity": {
        "type": "anti",
        "vms": ["web-server-02"]
      }
    }
  }'

# Response:
{
  "vm_id": "vm-12345",
  "status": "creating",
  "node": "worker-042",
  "created_at": "2025-01-10T12:00:00Z"
}
```

**Get VM Details:**
```bash
curl -X GET https://dwcp-api.example.com/v3/vms/vm-12345 \
  -H "Authorization: Bearer $TOKEN"

# Response:
{
  "vm_id": "vm-12345",
  "name": "web-server-01",
  "status": "running",
  "node": "worker-042",
  "resources": {
    "cpu": 4,
    "memory_gb": 8,
    "disk_gb": 50
  },
  "network": {
    "ip": "10.0.1.42",
    "mac": "52:54:00:12:34:56"
  },
  "uptime_seconds": 3600
}
```

**Delete VM:**
```bash
curl -X DELETE https://dwcp-api.example.com/v3/vms/vm-12345 \
  -H "Authorization: Bearer $TOKEN"

# Response:
{
  "status": "deleted",
  "deleted_at": "2025-01-10T13:00:00Z"
}
```

#### 2.2.4 VM Migration

**Start Migration:**
```bash
curl -X POST https://dwcp-api.example.com/v3/migrations \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "vm_id": "vm-12345",
    "target_node": "worker-156",
    "strategy": "pre-copy",
    "options": {
      "max_downtime_ms": 20000,
      "compression_level": 5,
      "amst_streams": 8
    }
  }'

# Response:
{
  "migration_id": "mig-67890",
  "status": "in_progress",
  "started_at": "2025-01-10T14:00:00Z"
}
```

**Get Migration Status:**
```bash
curl -X GET https://dwcp-api.example.com/v3/migrations/mig-67890 \
  -H "Authorization: Bearer $TOKEN"

# Response:
{
  "migration_id": "mig-67890",
  "vm_id": "vm-12345",
  "status": "in_progress",
  "progress": 65,
  "source_node": "worker-042",
  "target_node": "worker-156",
  "stats": {
    "bytes_transferred": 1400000000,
    "bytes_total": 2147483648,
    "bandwidth_mbps": 450,
    "elapsed_seconds": 35
  }
}
```

**List Migrations:**
```bash
curl -X GET https://dwcp-api.example.com/v3/migrations?status=in_progress \
  -H "Authorization: Bearer $TOKEN"

# Response:
{
  "migrations": [
    {
      "migration_id": "mig-67890",
      "vm_id": "vm-12345",
      "status": "in_progress",
      "progress": 75
    },
    ...
  ]
}
```

#### 2.2.5 Placement Operations

**Optimize Placement:**
```bash
curl -X POST https://dwcp-api.example.com/v3/placements/optimize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "genetic",
    "target": "resource_utilization",
    "constraints": {
      "max_vms_per_node": 100,
      "min_free_memory_pct": 10
    }
  }'

# Response:
{
  "optimization_id": "opt-11111",
  "status": "completed",
  "result": {
    "migrations_required": 47,
    "resource_utilization": 0.82,
    "estimated_time_minutes": 25
  },
  "migrations": [
    {
      "vm_id": "vm-12345",
      "from": "worker-042",
      "to": "worker-156"
    },
    ...
  ]
}
```

#### 2.2.6 Metrics and Monitoring

**Get Metrics:**
```bash
curl -X GET https://dwcp-api.example.com/v3/metrics?name=migration_duration_seconds&duration=1h \
  -H "Authorization: Bearer $TOKEN"

# Response:
{
  "metric": "migration_duration_seconds",
  "data_points": [
    {
      "timestamp": "2025-01-10T14:00:00Z",
      "value": 85.2,
      "labels": {
        "vm_size": "2gb",
        "source_region": "us-east-1",
        "target_region": "us-east-1"
      }
    },
    ...
  ]
}
```

### 2.3 gRPC API Examples

#### 2.3.1 Proto Definition

**File:** `api/proto/dwcp/v3/vm.proto`

```protobuf
syntax = "proto3";

package dwcp.v3;

option go_package = "github.com/your-org/dwcp-v3/api/proto/v3;dwcp";

service VMService {
  // Create a new VM
  rpc CreateVM(CreateVMRequest) returns (CreateVMResponse);

  // Get VM details
  rpc GetVM(GetVMRequest) returns (GetVMResponse);

  // Migrate VM to another node
  rpc MigrateVM(MigrateVMRequest) returns (stream MigrationProgress);

  // Delete VM
  rpc DeleteVM(DeleteVMRequest) returns (DeleteVMResponse);
}

message CreateVMRequest {
  string name = 1;
  string image = 2;
  Resources resources = 3;
  PlacementConstraints placement = 4;
}

message Resources {
  int32 cpu = 1;
  int64 memory_bytes = 2;
  int64 disk_bytes = 3;
  repeated GPU gpus = 4;
}

message GPU {
  string type = 1;  // e.g., "nvidia-a100"
  int32 count = 2;
}

message PlacementConstraints {
  string region = 1;
  AffinityRule affinity = 2;
}

message AffinityRule {
  string type = 1;  // "affinity" or "anti-affinity"
  repeated string vm_ids = 2;
}

message CreateVMResponse {
  string vm_id = 1;
  string node_id = 2;
  string status = 3;
}

message MigrateVMRequest {
  string vm_id = 1;
  string target_node = 2;
  MigrationOptions options = 3;
}

message MigrationOptions {
  int32 max_downtime_ms = 1;
  int32 compression_level = 2;
  int32 amst_streams = 3;
}

message MigrationProgress {
  string migration_id = 1;
  string status = 2;  // "in_progress", "completed", "failed"
  int32 progress_percent = 3;
  int64 bytes_transferred = 4;
  int64 bytes_total = 5;
  double bandwidth_mbps = 6;
  int32 elapsed_seconds = 7;
}
```

#### 2.3.2 Go Client Example

```go
package main

import (
    "context"
    "fmt"
    "io"
    "log"
    "time"

    "google.golang.org/grpc"
    "google.golang.org/grpc/credentials"
    pb "github.com/your-org/dwcp-v3/api/proto/v3"
)

func main() {
    // Setup TLS credentials
    creds, err := credentials.NewClientTLSFromFile("ca.crt", "")
    if err != nil {
        log.Fatalf("Failed to load credentials: %v", err)
    }

    // Connect to DWCP gRPC server
    conn, err := grpc.Dial(
        "dwcp-grpc.example.com:8081",
        grpc.WithTransportCredentials(creds),
        grpc.WithPerRPCCredentials(&authToken{token: "your-jwt-token"}),
    )
    if err != nil {
        log.Fatalf("Failed to connect: %v", err)
    }
    defer conn.Close()

    client := pb.NewVMServiceClient(conn)

    // Create VM
    vmID, err := createVM(client)
    if err != nil {
        log.Fatalf("CreateVM failed: %v", err)
    }
    fmt.Printf("Created VM: %s\n", vmID)

    // Wait for VM to be ready
    time.Sleep(30 * time.Second)

    // Migrate VM
    err = migrateVM(client, vmID, "worker-156")
    if err != nil {
        log.Fatalf("MigrateVM failed: %v", err)
    }
}

func createVM(client pb.VMServiceClient) (string, error) {
    ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
    defer cancel()

    req := &pb.CreateVMRequest{
        Name:  "test-vm",
        Image: "ubuntu-22.04",
        Resources: &pb.Resources{
            Cpu:         4,
            MemoryBytes: 8 * 1024 * 1024 * 1024,  // 8 GB
            DiskBytes:   50 * 1024 * 1024 * 1024, // 50 GB
        },
        Placement: &pb.PlacementConstraints{
            Region: "us-east-1",
        },
    }

    resp, err := client.CreateVM(ctx, req)
    if err != nil {
        return "", err
    }

    return resp.VmId, nil
}

func migrateVM(client pb.VMServiceClient, vmID, targetNode string) error {
    ctx := context.Background()

    req := &pb.MigrateVMRequest{
        VmId:       vmID,
        TargetNode: targetNode,
        Options: &pb.MigrationOptions{
            MaxDowntimeMs:    20000,
            CompressionLevel: 5,
            AmstStreams:      8,
        },
    }

    stream, err := client.MigrateVM(ctx, req)
    if err != nil {
        return err
    }

    // Stream migration progress
    for {
        progress, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }

        fmt.Printf("Migration %s: %d%% (%d/%d bytes, %.2f Mbps)\n",
            progress.Status,
            progress.ProgressPercent,
            progress.BytesTransferred,
            progress.BytesTotal,
            progress.BandwidthMbps,
        )

        if progress.Status == "completed" {
            fmt.Println("Migration completed successfully!")
            break
        } else if progress.Status == "failed" {
            return fmt.Errorf("migration failed")
        }
    }

    return nil
}

// authToken implements credentials.PerRPCCredentials
type authToken struct {
    token string
}

func (t *authToken) GetRequestMetadata(ctx context.Context, uri ...string) (map[string]string, error) {
    return map[string]string{
        "authorization": "Bearer " + t.token,
    }, nil
}

func (t *authToken) RequireTransportSecurity() bool {
    return true
}
```

#### 2.3.3 Python Client Example

```python
import grpc
import dwcp.v3.vm_pb2 as vm_pb2
import dwcp.v3.vm_pb2_grpc as vm_pb2_grpc

def main():
    # Setup TLS credentials
    with open('ca.crt', 'rb') as f:
        creds = grpc.ssl_channel_credentials(f.read())

    # Setup auth token
    auth_creds = grpc.access_token_call_credentials('your-jwt-token')
    composite_creds = grpc.composite_channel_credentials(creds, auth_creds)

    # Connect to DWCP gRPC server
    channel = grpc.secure_channel(
        'dwcp-grpc.example.com:8081',
        composite_creds
    )
    client = vm_pb2_grpc.VMServiceStub(channel)

    # Create VM
    vm_id = create_vm(client)
    print(f"Created VM: {vm_id}")

    # Migrate VM
    migrate_vm(client, vm_id, "worker-156")

def create_vm(client):
    request = vm_pb2.CreateVMRequest(
        name="test-vm",
        image="ubuntu-22.04",
        resources=vm_pb2.Resources(
            cpu=4,
            memory_bytes=8 * 1024**3,  # 8 GB
            disk_bytes=50 * 1024**3    # 50 GB
        ),
        placement=vm_pb2.PlacementConstraints(
            region="us-east-1"
        )
    )

    response = client.CreateVM(request, timeout=10)
    return response.vm_id

def migrate_vm(client, vm_id, target_node):
    request = vm_pb2.MigrateVMRequest(
        vm_id=vm_id,
        target_node=target_node,
        options=vm_pb2.MigrationOptions(
            max_downtime_ms=20000,
            compression_level=5,
            amst_streams=8
        )
    )

    # Stream migration progress
    for progress in client.MigrateVM(request):
        print(f"Migration {progress.status}: {progress.progress_percent}% "
              f"({progress.bytes_transferred}/{progress.bytes_total} bytes, "
              f"{progress.bandwidth_mbps:.2f} Mbps)")

        if progress.status == "completed":
            print("Migration completed successfully!")
            break
        elif progress.status == "failed":
            raise Exception("Migration failed")

if __name__ == '__main__':
    main()
```

### 2.4 API Error Handling

#### 2.4.1 REST API Error Format

```json
{
  "error": {
    "code": "VM_NOT_FOUND",
    "message": "VM with ID vm-12345 not found",
    "details": {
      "vm_id": "vm-12345",
      "cluster": "prod-cluster-1"
    },
    "request_id": "req-abc123"
  }
}
```

#### 2.4.2 gRPC Error Codes

| gRPC Code | HTTP Code | Description |
|-----------|-----------|-------------|
| `OK` | 200 | Success |
| `INVALID_ARGUMENT` | 400 | Invalid request parameters |
| `UNAUTHENTICATED` | 401 | Missing or invalid token |
| `PERMISSION_DENIED` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `ALREADY_EXISTS` | 409 | Resource already exists |
| `RESOURCE_EXHAUSTED` | 429 | Rate limit exceeded |
| `INTERNAL` | 500 | Internal server error |
| `UNAVAILABLE` | 503 | Service unavailable |

#### 2.4.3 Error Handling Example (Go)

```go
func handleMigrationError(err error) {
    st, ok := status.FromError(err)
    if !ok {
        log.Printf("Unknown error: %v", err)
        return
    }

    switch st.Code() {
    case codes.NotFound:
        log.Printf("VM not found: %s", st.Message())
    case codes.ResourceExhausted:
        log.Printf("Migration quota exceeded, retrying in 60s...")
        time.Sleep(60 * time.Second)
        // Retry migration
    case codes.FailedPrecondition:
        log.Printf("Migration precondition failed: %s", st.Message())
        // Check VM state before retrying
    case codes.Internal:
        log.Printf("Internal server error: %s", st.Message())
        // Report to ops team
    default:
        log.Printf("Error: %s (%s)", st.Message(), st.Code())
    }
}
```

---

## 3. Integration Patterns

### 3.1 Common Use Cases

#### 3.1.1 VM Orchestration Platform

**Scenario:** Build Kubernetes-like orchestrator on DWCP

**Components:**
- VM lifecycle manager (create, delete, restart)
- Scheduler (placement decisions)
- Load balancer (distribute traffic)
- Auto-scaler (scale based on metrics)

**Architecture:**
```
┌─────────────────────────────────────┐
│      Your Orchestrator              │
│  ┌────────────┐  ┌───────────────┐  │
│  │ Scheduler  │  │ Load Balancer │  │
│  └────────────┘  └───────────────┘  │
│         ↓                ↓           │
└─────────────────────────────────────┘
                 ↓
         DWCP v3 API
                 ↓
┌─────────────────────────────────────┐
│        DWCP v3 Cluster              │
│   [worker-1] [worker-2] [worker-3]  │
│      VM-1        VM-2      VM-3     │
└─────────────────────────────────────┘
```

**Example Code (Go):**

```go
package orchestrator

import (
    "context"
    "fmt"
    "time"

    pb "github.com/your-org/dwcp-v3/api/proto/v3"
)

type Orchestrator struct {
    vmClient        pb.VMServiceClient
    placementClient pb.PlacementServiceClient
    scheduler       *Scheduler
}

// DeployApplication deploys a multi-VM application
func (o *Orchestrator) DeployApplication(ctx context.Context, app *Application) error {
    // 1. Get optimal placement for all VMs
    placement, err := o.scheduler.ScheduleApplication(ctx, app)
    if err != nil {
        return fmt.Errorf("scheduling failed: %w", err)
    }

    // 2. Create VMs in parallel
    vmIDs := make([]string, len(app.VMs))
    errChan := make(chan error, len(app.VMs))

    for i, vmSpec := range app.VMs {
        go func(idx int, spec VMSpec, node string) {
            req := &pb.CreateVMRequest{
                Name:      spec.Name,
                Image:     spec.Image,
                Resources: toProtoResources(spec.Resources),
                Placement: &pb.PlacementConstraints{
                    Region: node,
                },
            }

            resp, err := o.vmClient.CreateVM(ctx, req)
            if err != nil {
                errChan <- err
                return
            }

            vmIDs[idx] = resp.VmId
            errChan <- nil
        }(i, vmSpec, placement.Nodes[i])
    }

    // 3. Wait for all VMs to be created
    for range app.VMs {
        if err := <-errChan; err != nil {
            return fmt.Errorf("VM creation failed: %w", err)
        }
    }

    // 4. Configure networking and load balancing
    err = o.configureNetworking(ctx, vmIDs, app.NetworkPolicy)
    if err != nil {
        return fmt.Errorf("networking configuration failed: %w", err)
    }

    return nil
}

// AutoScale scales application based on metrics
func (o *Orchestrator) AutoScale(ctx context.Context, appID string) {
    ticker := time.NewTicker(30 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            app, err := o.getApplication(appID)
            if err != nil {
                continue
            }

            metrics, err := o.getApplicationMetrics(appID)
            if err != nil {
                continue
            }

            // Scale up if CPU > 80%
            if metrics.CPUUtilization > 0.80 && len(app.VMs) < app.MaxVMs {
                err := o.scaleUp(ctx, app)
                if err != nil {
                    fmt.Printf("Scale up failed: %v\n", err)
                }
            }

            // Scale down if CPU < 20%
            if metrics.CPUUtilization < 0.20 && len(app.VMs) > app.MinVMs {
                err := o.scaleDown(ctx, app)
                if err != nil {
                    fmt.Printf("Scale down failed: %v\n", err)
                }
            }
        }
    }
}

func (o *Orchestrator) scaleUp(ctx context.Context, app *Application) error {
    // Add new VM to application
    vmSpec := app.VMs[0] // Use template from first VM
    placement, err := o.scheduler.ScheduleVM(ctx, vmSpec)
    if err != nil {
        return err
    }

    req := &pb.CreateVMRequest{
        Name:      fmt.Sprintf("%s-vm-%d", app.Name, len(app.VMs)+1),
        Image:     vmSpec.Image,
        Resources: toProtoResources(vmSpec.Resources),
        Placement: &pb.PlacementConstraints{
            Region: placement.Node,
        },
    }

    resp, err := o.vmClient.CreateVM(ctx, req)
    if err != nil {
        return err
    }

    fmt.Printf("Scaled up: Created VM %s\n", resp.VmId)
    return nil
}
```

#### 3.1.2 Live Migration Service

**Scenario:** Automated VM migration based on policies

**Triggers:**
- Node maintenance (drain VMs before update)
- Resource balancing (move VMs to optimize utilization)
- Network optimization (reduce inter-VM latency)
- Cost optimization (move to cheaper nodes)

**Example Code (Python):**

```python
import dwcp.v3 as dwcp
import asyncio
from typing import List, Dict

class MigrationService:
    def __init__(self, api_client: dwcp.Client):
        self.client = api_client
        self.policies: List[MigrationPolicy] = []

    async def run(self):
        """Main loop: evaluate policies and trigger migrations"""
        while True:
            for policy in self.policies:
                migrations = await policy.evaluate(self.client)
                for migration in migrations:
                    await self.execute_migration(migration)

            await asyncio.sleep(60)  # Check every minute

    async def execute_migration(self, migration: Migration):
        """Execute migration with retry and rollback"""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                print(f"Migrating VM {migration.vm_id} to {migration.target_node} "
                      f"(attempt {attempt + 1}/{max_retries})")

                # Start migration
                response = await self.client.migrations.create(
                    vm_id=migration.vm_id,
                    target_node=migration.target_node,
                    options={
                        'max_downtime_ms': 20000,
                        'compression_level': 5
                    }
                )

                # Wait for completion
                async for progress in self.client.migrations.stream(response.migration_id):
                    print(f"  Progress: {progress.progress_percent}%")

                    if progress.status == 'completed':
                        print(f"  ✓ Migration completed")
                        return
                    elif progress.status == 'failed':
                        raise Exception("Migration failed")

            except Exception as e:
                print(f"  ✗ Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    await self.alert_ops_team(migration, e)
                else:
                    await asyncio.sleep(30)  # Wait before retry

class MaintenancePolicy(MigrationPolicy):
    """Migrate VMs from nodes scheduled for maintenance"""

    async def evaluate(self, client: dwcp.Client) -> List[Migration]:
        migrations = []

        # Get nodes scheduled for maintenance
        nodes = await client.nodes.list(tags=['maintenance=scheduled'])

        for node in nodes:
            # Get VMs on this node
            vms = await client.vms.list(node_id=node.id)

            for vm in vms:
                # Find alternative node
                target = await client.placements.optimize(
                    vm_id=vm.id,
                    exclude_nodes=[node.id]
                )

                migrations.append(Migration(
                    vm_id=vm.id,
                    source_node=node.id,
                    target_node=target.node_id,
                    reason='node_maintenance'
                ))

        return migrations

class LoadBalancingPolicy(MigrationPolicy):
    """Migrate VMs to balance resource utilization"""

    async def evaluate(self, client: dwcp.Client) -> List[Migration]:
        migrations = []

        # Get nodes with high utilization (>85%)
        overloaded = await client.nodes.list(cpu_utilization_gt=0.85)

        # Get nodes with low utilization (<40%)
        underutilized = await client.nodes.list(cpu_utilization_lt=0.40)

        if not overloaded or not underutilized:
            return migrations

        for source_node in overloaded:
            # Get VMs sorted by resource usage
            vms = await client.vms.list(
                node_id=source_node.id,
                sort='cpu_utilization',
                order='asc'
            )

            # Migrate smallest VMs first
            for vm in vms[:3]:  # Move top 3 smallest VMs
                target = underutilized[0]

                migrations.append(Migration(
                    vm_id=vm.id,
                    source_node=source_node.id,
                    target_node=target.id,
                    reason='load_balancing'
                ))

        return migrations
```

#### 3.1.3 Disaster Recovery System

**Scenario:** Automated DR with cross-region replication

**Features:**
- Continuous VM snapshots
- Cross-region replication
- Automated failover
- Recovery time objective (RTO): <5 minutes
- Recovery point objective (RPO): <1 minute

**Example Code (Go):**

```go
package dr

import (
    "context"
    "fmt"
    "time"

    pb "github.com/your-org/dwcp-v3/api/proto/v3"
)

type DRService struct {
    primaryClient   pb.VMServiceClient
    secondaryClient pb.VMServiceClient
    config          DRConfig
}

type DRConfig struct {
    PrimaryRegion    string
    SecondaryRegion  string
    SnapshotInterval time.Duration
    MaxRPO           time.Duration
}

// Run continuous disaster recovery loop
func (dr *DRService) Run(ctx context.Context) error {
    ticker := time.NewTicker(dr.config.SnapshotInterval)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-ticker.C:
            err := dr.replicateVMs(ctx)
            if err != nil {
                fmt.Printf("Replication failed: %v\n", err)
            }
        }
    }
}

// replicateVMs creates snapshots and replicates to secondary region
func (dr *DRService) replicateVMs(ctx context.Context) error {
    // 1. Get all protected VMs in primary region
    vms, err := dr.getProtectedVMs(ctx)
    if err != nil {
        return err
    }

    // 2. Snapshot and replicate each VM
    for _, vm := range vms {
        go dr.replicateVM(ctx, vm)
    }

    return nil
}

func (dr *DRService) replicateVM(ctx context.Context, vm *VM) error {
    // 1. Create snapshot of primary VM
    snapshot, err := dr.createSnapshot(ctx, vm.ID)
    if err != nil {
        return fmt.Errorf("snapshot failed: %w", err)
    }

    // 2. Check if secondary VM exists
    secondaryVM, err := dr.getSecondaryVM(ctx, vm.ID)
    if err != nil {
        // Create secondary VM if doesn't exist
        secondaryVM, err = dr.createSecondaryVM(ctx, vm)
        if err != nil {
            return fmt.Errorf("create secondary VM failed: %w", err)
        }
    }

    // 3. Replicate snapshot to secondary region
    err = dr.replicateSnapshot(ctx, snapshot, secondaryVM)
    if err != nil {
        return fmt.Errorf("snapshot replication failed: %w", err)
    }

    fmt.Printf("Replicated VM %s to secondary region (RPO: %s)\n",
        vm.ID, time.Since(snapshot.Timestamp))

    return nil
}

// Failover switches traffic to secondary region
func (dr *DRService) Failover(ctx context.Context) error {
    fmt.Println("Initiating DR failover to secondary region...")

    // 1. Stop replication (prevent data corruption)
    dr.stopReplication()

    // 2. Activate all VMs in secondary region
    vms, err := dr.getSecondaryVMs(ctx)
    if err != nil {
        return err
    }

    for _, vm := range vms {
        err := dr.activateVM(ctx, vm.ID)
        if err != nil {
            fmt.Printf("Failed to activate VM %s: %v\n", vm.ID, err)
            continue
        }
        fmt.Printf("Activated VM %s in secondary region\n", vm.ID)
    }

    // 3. Update DNS to point to secondary region
    err = dr.updateDNS(dr.config.SecondaryRegion)
    if err != nil {
        return fmt.Errorf("DNS update failed: %w", err)
    }

    // 4. Verify all VMs are accessible
    err = dr.verifyFailover(ctx)
    if err != nil {
        return fmt.Errorf("failover verification failed: %w", err)
    }

    fmt.Println("DR failover completed successfully")
    return nil
}
```

### 3.2 SDK and Client Libraries

#### 3.2.1 Official Go SDK

**Installation:**
```bash
go get github.com/your-org/dwcp-v3-go-sdk
```

**Usage:**
```go
import "github.com/your-org/dwcp-v3-go-sdk/dwcp"

client, err := dwcp.NewClient(dwcp.Config{
    APIEndpoint: "https://dwcp-api.example.com",
    Token:       "your-jwt-token",
})

vm, err := client.VMs().Create(ctx, &dwcp.VMSpec{
    Name:  "web-server",
    Image: "ubuntu-22.04",
    Resources: dwcp.Resources{
        CPU:    4,
        Memory: 8 * dwcp.GB,
        Disk:   50 * dwcp.GB,
    },
})
```

#### 3.2.2 Official Python SDK

**Installation:**
```bash
pip install dwcp-v3-python-sdk
```

**Usage:**
```python
from dwcp import Client, VMSpec, Resources, GB

client = Client(
    api_endpoint='https://dwcp-api.example.com',
    token='your-jwt-token'
)

vm = client.vms.create(VMSpec(
    name='web-server',
    image='ubuntu-22.04',
    resources=Resources(
        cpu=4,
        memory=8*GB,
        disk=50*GB
    )
))
```

---

## 4. Component Architecture

### 4.1 AMST (Adaptive Multi-Stream Transport)

#### 4.1.1 Architecture

**Code Location:** `backend/core/network/dwcp_v3/transport/`

**Key Files:**
- `amst.go` - Main AMST implementation
- `stream.go` - Individual stream management
- `scheduler.go` - Data distribution across streams
- `congestion.go` - BBR v2 congestion control

**How it Works:**

```
┌─────────────────────────────────────────────────────────┐
│                    AMST Transport                        │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Stream 1   │  │  Stream 2   │  │  Stream N   │     │
│  │  (TCP)      │  │  (TCP)      │  │  (TCP)      │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         ↓                ↓                ↓             │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Scheduler (Weighted RR)                │   │
│  └──────────────────────────────────────────────────┘   │
│         ↑                                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Congestion Monitor (BBR v2)            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

#### 4.1.2 Code Example: AMST Stream

```go
// backend/core/network/dwcp_v3/transport/stream.go
package transport

import (
    "context"
    "net"
    "time"
)

type Stream struct {
    id         int
    conn       net.Conn
    bandwidth  float64  // Current bandwidth (Mbps)
    rtt        time.Duration
    cwnd       int      // Congestion window (packets)
    state      StreamState
    metrics    *StreamMetrics
}

type StreamState int

const (
    StateIdle StreamState = iota
    StateActive
    StateDegraded
    StateFailed
)

// Send sends data chunk over this stream
func (s *Stream) Send(ctx context.Context, chunk []byte) error {
    // 1. Check stream health
    if s.state == StateFailed {
        return ErrStreamFailed
    }

    // 2. Apply congestion control
    delay := s.calculatePacingDelay(len(chunk))
    if delay > 0 {
        select {
        case <-time.After(delay):
        case <-ctx.Done():
            return ctx.Err()
        }
    }

    // 3. Send data
    start := time.Now()
    n, err := s.conn.Write(chunk)
    if err != nil {
        s.state = StateFailed
        return err
    }

    // 4. Update metrics
    duration := time.Since(start)
    s.updateMetrics(n, duration)

    return nil
}

// calculatePacingDelay implements BBR v2 pacing
func (s *Stream) calculatePacingDelay(bytes int) time.Duration {
    // BBR pacing: delay = bytes / (pacing_gain * bandwidth)
    const pacingGain = 2.77
    targetBandwidth := s.bandwidth * pacingGain

    if targetBandwidth == 0 {
        return 0
    }

    // Convert Mbps to bytes/second
    bytesPerSecond := targetBandwidth * 1_000_000 / 8

    // Calculate delay for this chunk
    delay := time.Duration(float64(bytes)/bytesPerSecond) * time.Second

    return delay
}

// updateMetrics updates stream metrics based on send result
func (s *Stream) updateMetrics(bytes int, duration time.Duration) {
    // Calculate instantaneous bandwidth
    instantBW := float64(bytes) * 8 / duration.Seconds() / 1_000_000

    // Exponential moving average
    const alpha = 0.2
    s.bandwidth = alpha*instantBW + (1-alpha)*s.bandwidth

    // Update metrics
    s.metrics.BytesSent += int64(bytes)
    s.metrics.PacketsSent++
    s.metrics.AvgBandwidthMbps = s.bandwidth
}
```

#### 4.1.3 Code Example: AMST Scheduler

```go
// backend/core/network/dwcp_v3/transport/scheduler.go
package transport

import (
    "context"
    "sync"
)

type Scheduler struct {
    streams    []*Stream
    currentIdx int
    mu         sync.Mutex
}

// Distribute distributes data across streams using weighted round-robin
func (sch *Scheduler) Distribute(ctx context.Context, data []byte) error {
    // 1. Split data into chunks
    chunks := sch.splitData(data)

    // 2. Distribute chunks across streams
    errChan := make(chan error, len(chunks))
    for i, chunk := range chunks {
        stream := sch.selectStream()

        go func(s *Stream, c []byte, idx int) {
            err := s.Send(ctx, c)
            errChan <- err
        }(stream, chunk, i)
    }

    // 3. Wait for all sends to complete
    for range chunks {
        if err := <-errChan; err != nil {
            return err
        }
    }

    return nil
}

// selectStream selects next stream using weighted round-robin
func (sch *Scheduler) selectStream() *Stream {
    sch.mu.Lock()
    defer sch.mu.Unlock()

    // Filter healthy streams
    healthy := make([]*Stream, 0)
    for _, s := range sch.streams {
        if s.state == StateActive {
            healthy = append(healthy, s)
        }
    }

    if len(healthy) == 0 {
        // Fallback to any non-failed stream
        for _, s := range sch.streams {
            if s.state != StateFailed {
                return s
            }
        }
        return sch.streams[0]  // Last resort
    }

    // Weighted selection based on bandwidth
    totalBW := 0.0
    for _, s := range healthy {
        totalBW += s.bandwidth
    }

    // Simple round-robin (can be improved with actual weighted selection)
    sch.currentIdx = (sch.currentIdx + 1) % len(healthy)
    return healthy[sch.currentIdx]
}

func (sch *Scheduler) splitData(data []byte) [][]byte {
    const chunkSize = 64 * 1024  // 64 KB chunks

    chunks := make([][]byte, 0)
    for i := 0; i < len(data); i += chunkSize {
        end := i + chunkSize
        if end > len(data) {
            end = len(data)
        }
        chunks = append(chunks, data[i:end])
    }

    return chunks
}
```

### 4.2 HDE (Hierarchical Data Encoding)

#### 4.2.1 Architecture

**Code Location:** `backend/core/network/dwcp_v3/encoding/`

**Key Files:**
- `hde.go` - Main HDE implementation
- `compressor.go` - Zstandard compression
- `deduplicator.go` - Content-aware deduplication
- `dictionary.go` - Compression dictionary training

**Pipeline:**

```
Raw VM Memory
     ↓
┌─────────────────────┐
│   Block Splitter    │  (4KB fixed blocks)
└─────────────────────┘
     ↓
┌─────────────────────┐
│   Deduplicator      │  (SHA-256 hash → DB lookup)
└─────────────────────┘
     ↓
┌─────────────────────┐
│   Compressor        │  (Zstandard level 5)
└─────────────────────┘
     ↓
Compressed Unique Blocks
```

#### 4.2.2 Code Example: Deduplication

```go
// backend/core/network/dwcp_v3/encoding/deduplicator.go
package encoding

import (
    "crypto/sha256"
    "fmt"

    "github.com/dgraph-io/badger/v3"
)

type Deduplicator struct {
    db        *badger.DB
    blockSize int
    stats     *DedupStats
}

type DedupStats struct {
    TotalBlocks      int64
    UniqueBlocks     int64
    DuplicateBlocks  int64
    BytesSaved       int64
}

// Deduplicate processes data and returns only unique blocks
func (d *Deduplicator) Deduplicate(data []byte) (*DedupResult, error) {
    blocks := d.splitBlocks(data)
    result := &DedupResult{
        Blocks: make([]*Block, 0, len(blocks)),
    }

    for i, blockData := range blocks {
        // 1. Compute hash
        hash := sha256.Sum256(blockData)
        hashStr := fmt.Sprintf("%x", hash)

        // 2. Check if block exists in DB
        exists, err := d.blockExists(hashStr)
        if err != nil {
            return nil, err
        }

        if exists {
            // Block is duplicate, store reference only
            result.Blocks = append(result.Blocks, &Block{
                Index:       i,
                Hash:        hashStr,
                IsDuplicate: true,
            })

            d.stats.DuplicateBlocks++
            d.stats.BytesSaved += int64(len(blockData))
        } else {
            // Block is unique, store full data
            result.Blocks = append(result.Blocks, &Block{
                Index:       i,
                Hash:        hashStr,
                Data:        blockData,
                IsDuplicate: false,
            })

            // Save to DB for future lookups
            err = d.saveBlock(hashStr, blockData)
            if err != nil {
                return nil, err
            }

            d.stats.UniqueBlocks++
        }

        d.stats.TotalBlocks++
    }

    return result, nil
}

func (d *Deduplicator) blockExists(hash string) (bool, error) {
    err := d.db.View(func(txn *badger.Txn) error {
        _, err := txn.Get([]byte(hash))
        return err
    })

    if err == badger.ErrKeyNotFound {
        return false, nil
    }
    if err != nil {
        return false, err
    }
    return true, nil
}

func (d *Deduplicator) saveBlock(hash string, data []byte) error {
    return d.db.Update(func(txn *badger.Txn) error {
        return txn.Set([]byte(hash), data)
    })
}

func (d *Deduplicator) splitBlocks(data []byte) [][]byte {
    blocks := make([][]byte, 0)
    for i := 0; i < len(data); i += d.blockSize {
        end := i + d.blockSize
        if end > len(data) {
            end = len(data)
        }
        blocks = append(blocks, data[i:end])
    }
    return blocks
}

// GetStats returns deduplication statistics
func (d *Deduplicator) GetStats() DedupStats {
    return *d.stats
}

// ComputeEfficiency returns deduplication efficiency (0-1)
func (d *Deduplicator) ComputeEfficiency() float64 {
    if d.stats.TotalBlocks == 0 {
        return 0
    }
    return float64(d.stats.DuplicateBlocks) / float64(d.stats.TotalBlocks)
}
```

#### 4.2.3 Code Example: Compression

```go
// backend/core/network/dwcp_v3/encoding/compressor.go
package encoding

import (
    "bytes"

    "github.com/klauspost/compress/zstd"
)

type Compressor struct {
    level      zstd.EncoderLevel
    encoder    *zstd.Encoder
    decoder    *zstd.Decoder
    dictionary []byte
}

// NewCompressor creates a new Zstandard compressor
func NewCompressor(level int, dictPath string) (*Compressor, error) {
    // Map level to zstd encoder level
    var encLevel zstd.EncoderLevel
    switch level {
    case 1:
        encLevel = zstd.SpeedFastest
    case 3:
        encLevel = zstd.SpeedDefault
    case 5:
        encLevel = zstd.SpeedBetterCompression
    case 9:
        encLevel = zstd.SpeedBestCompression
    default:
        encLevel = zstd.SpeedDefault
    }

    // Load dictionary if provided
    var dict []byte
    if dictPath != "" {
        var err error
        dict, err = loadDictionary(dictPath)
        if err != nil {
            return nil, err
        }
    }

    // Create encoder
    encoder, err := zstd.NewWriter(nil,
        zstd.WithEncoderLevel(encLevel),
        zstd.WithEncoderDict(dict),
    )
    if err != nil {
        return nil, err
    }

    // Create decoder
    decoder, err := zstd.NewReader(nil,
        zstd.WithDecoderDicts(dict),
    )
    if err != nil {
        return nil, err
    }

    return &Compressor{
        level:      encLevel,
        encoder:    encoder,
        decoder:    decoder,
        dictionary: dict,
    }, nil
}

// Compress compresses data using Zstandard
func (c *Compressor) Compress(data []byte) ([]byte, error) {
    return c.encoder.EncodeAll(data, nil), nil
}

// Decompress decompresses Zstandard data
func (c *Compressor) Decompress(compressed []byte) ([]byte, error) {
    return c.decoder.DecodeAll(compressed, nil)
}

// ComputeRatio returns compression ratio (0-1, lower is better)
func (c *Compressor) ComputeRatio(original, compressed []byte) float64 {
    return float64(len(compressed)) / float64(len(original))
}
```

### 4.3 PBA (Predictive Bandwidth Allocator)

#### 4.3.1 Architecture

**Code Location:** `backend/core/network/dwcp_v3/prediction/`

**Key Files:**
- `pba.go` - Main PBA implementation
- `lstm.go` - LSTM neural network
- `trainer.go` - Model training pipeline
- `predictor.go` - Real-time prediction

**LSTM Model:**

```
Input: [bandwidth, latency, loss] (last 60 samples, 5 minutes)
    ↓
┌──────────────────┐
│  LSTM Layer 1    │  (128 units)
└──────────────────┘
    ↓
┌──────────────────┐
│  LSTM Layer 2    │  (64 units)
└──────────────────┘
    ↓
┌──────────────────┐
│  LSTM Layer 3    │  (32 units)
└──────────────────┘
    ↓
┌──────────────────┐
│  Dense Layer     │  (3 units)
└──────────────────┘
    ↓
Output: [predicted_bandwidth, predicted_latency, predicted_loss] (next 30s)
```

#### 4.3.2 Code Example: LSTM Prediction

```go
// backend/core/network/dwcp_v3/prediction/predictor.go
package prediction

import (
    "context"
    "time"

    "gorgonia.org/gorgonia"
    "gorgonia.org/tensor"
)

type Predictor struct {
    model          *LSTMModel
    sequenceLength int
    history        *CircularBuffer
}

type NetworkMetrics struct {
    BandwidthMbps float64
    LatencyMs     float64
    PacketLoss    float64
    Timestamp     time.Time
}

// Predict forecasts network conditions for next N seconds
func (p *Predictor) Predict(ctx context.Context, horizon time.Duration) (*Prediction, error) {
    // 1. Get recent metrics (last 60 samples = 5 minutes)
    metrics := p.history.GetRecent(p.sequenceLength)
    if len(metrics) < p.sequenceLength {
        return nil, ErrInsufficientData
    }

    // 2. Prepare input tensor (shape: [1, sequenceLength, 3])
    input := make([]float64, p.sequenceLength*3)
    for i, m := range metrics {
        input[i*3+0] = m.BandwidthMbps / 1000.0  // Normalize to [0, 1]
        input[i*3+1] = m.LatencyMs / 500.0       // Normalize to [0, 1]
        input[i*3+2] = m.PacketLoss               // Already [0, 1]
    }

    inputTensor := tensor.New(
        tensor.WithShape(1, p.sequenceLength, 3),
        tensor.WithBacking(input),
    )

    // 3. Run LSTM inference
    output, err := p.model.Forward(inputTensor)
    if err != nil {
        return nil, err
    }

    // 4. Extract predictions
    outputData := output.Data().([]float64)

    prediction := &Prediction{
        BandwidthMbps: outputData[0] * 1000.0,  // Denormalize
        LatencyMs:     outputData[1] * 500.0,
        PacketLoss:    outputData[2],
        Confidence:    p.computeConfidence(metrics),
        ValidUntil:    time.Now().Add(horizon),
    }

    return prediction, nil
}

// computeConfidence estimates prediction confidence based on historical accuracy
func (p *Predictor) computeConfidence(recentMetrics []*NetworkMetrics) float64 {
    // Use variance of recent metrics as confidence proxy
    // Low variance = high confidence, high variance = low confidence

    var bwVariance, latVariance, lossVariance float64
    var bwMean, latMean, lossMean float64

    n := float64(len(recentMetrics))
    for _, m := range recentMetrics {
        bwMean += m.BandwidthMbps
        latMean += m.LatencyMs
        lossMean += m.PacketLoss
    }
    bwMean /= n
    latMean /= n
    lossMean /= n

    for _, m := range recentMetrics {
        bwVariance += (m.BandwidthMbps - bwMean) * (m.BandwidthMbps - bwMean)
        latVariance += (m.LatencyMs - latMean) * (m.LatencyMs - latMean)
        lossVariance += (m.PacketLoss - lossMean) * (m.PacketLoss - lossMean)
    }
    bwVariance /= n
    latVariance /= n
    lossVariance /= n

    // Normalize variances and compute confidence
    // Lower variance = higher confidence
    confidence := 1.0 - (bwVariance/1000.0 + latVariance/100.0 + lossVariance)
    if confidence < 0 {
        confidence = 0
    }
    if confidence > 1 {
        confidence = 1
    }

    return confidence
}
```

### 4.4 Additional Components

Due to length constraints, complete code examples for **ASS**, **ITP**, and **ACP** are available in separate documentation:

- **ASS (State Sync):** `docs/deployment/DWCP_V3_ASS_IMPLEMENTATION.md`
- **ITP (Placement):** `docs/deployment/DWCP_V3_ITP_IMPLEMENTATION.md`
- **ACP (Consensus):** `docs/deployment/DWCP_V3_ACP_IMPLEMENTATION.md`

---

## 5. Code Examples

### 5.1 Full Application: VM Load Balancer

**Scenario:** Distribute incoming traffic across multiple VMs

**Complete Code:** `examples/vm-load-balancer/`

**Key Features:**
- Health checking (detect failed VMs)
- Weighted round-robin distribution
- Auto-scaling (add/remove VMs based on load)
- Metrics export (Prometheus)

**main.go:**

```go
package main

import (
    "context"
    "fmt"
    "log"
    "net/http"
    "time"

    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promhttp"
    pb "github.com/your-org/dwcp-v3/api/proto/v3"
)

type LoadBalancer struct {
    vmClient pb.VMServiceClient
    backends []*Backend
    current  int
    metrics  *Metrics
}

type Backend struct {
    VMID      string
    IP        string
    Healthy   bool
    Weight    int
    ActiveReqs int
}

type Metrics struct {
    requestsTotal   prometheus.Counter
    requestDuration prometheus.Histogram
    backendHealth   *prometheus.GaugeVec
}

func main() {
    // Setup DWCP client
    vmClient := setupDWCPClient()

    // Create load balancer
    lb := &LoadBalancer{
        vmClient: vmClient,
        backends: make([]*Backend, 0),
        metrics:  setupMetrics(),
    }

    // Discover backend VMs
    ctx := context.Background()
    err := lb.discoverBackends(ctx)
    if err != nil {
        log.Fatalf("Failed to discover backends: %v", err)
    }

    // Start health checking
    go lb.healthCheckLoop(ctx)

    // Start auto-scaler
    go lb.autoScaleLoop(ctx)

    // Start HTTP server
    http.HandleFunc("/", lb.handleRequest)
    http.Handle("/metrics", promhttp.Handler())

    log.Println("Load balancer listening on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}

func (lb *LoadBalancer) handleRequest(w http.ResponseWriter, r *http.Request) {
    start := time.Now()
    defer func() {
        lb.metrics.requestDuration.Observe(time.Since(start).Seconds())
        lb.metrics.requestsTotal.Inc()
    }()

    // 1. Select backend using weighted round-robin
    backend := lb.selectBackend()
    if backend == nil {
        http.Error(w, "No healthy backends available", http.StatusServiceUnavailable)
        return
    }

    backend.ActiveReqs++
    defer func() { backend.ActiveReqs-- }()

    // 2. Proxy request to backend
    err := lb.proxyRequest(w, r, backend)
    if err != nil {
        log.Printf("Proxy error: %v", err)
        http.Error(w, "Backend error", http.StatusBadGateway)
        return
    }
}

func (lb *LoadBalancer) selectBackend() *Backend {
    // Weighted round-robin selection
    healthy := make([]*Backend, 0)
    for _, b := range lb.backends {
        if b.Healthy {
            healthy = append(healthy, b)
        }
    }

    if len(healthy) == 0 {
        return nil
    }

    // Simple round-robin (can be enhanced with actual weighting)
    lb.current = (lb.current + 1) % len(healthy)
    return healthy[lb.current]
}

func (lb *LoadBalancer) healthCheckLoop(ctx context.Context) {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            for _, backend := range lb.backends {
                healthy := lb.checkHealth(backend)
                backend.Healthy = healthy

                // Update Prometheus metric
                if healthy {
                    lb.metrics.backendHealth.WithLabelValues(backend.VMID).Set(1)
                } else {
                    lb.metrics.backendHealth.WithLabelValues(backend.VMID).Set(0)
                }
            }
        }
    }
}

func (lb *LoadBalancer) autoScaleLoop(ctx context.Context) {
    ticker := time.NewTicker(60 * time.Second)
    defer ticker.Stop()

    for {
        select {
        case <-ctx.Done():
            return
        case <-ticker.C:
            avgLoad := lb.computeAverageLoad()

            // Scale up if average load > 80%
            if avgLoad > 0.80 && len(lb.backends) < 10 {
                err := lb.scaleUp(ctx)
                if err != nil {
                    log.Printf("Scale up failed: %v", err)
                }
            }

            // Scale down if average load < 20%
            if avgLoad < 0.20 && len(lb.backends) > 2 {
                err := lb.scaleDown(ctx)
                if err != nil {
                    log.Printf("Scale down failed: %v", err)
                }
            }
        }
    }
}
```

### 5.2 Full Application: Automated VM Backup

**Complete Code:** `examples/vm-backup-service/`

**See:** `examples/vm-backup-service/README.md` for full implementation

---

## 6. Testing Strategies

### 6.1 Unit Testing

**Example: Testing AMST Stream**

```go
// backend/core/network/dwcp_v3/transport/stream_test.go
package transport

import (
    "context"
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
)

func TestStreamSend(t *testing.T) {
    // Setup mock connection
    mockConn := &mockTCPConn{
        writeDelay: 10 * time.Millisecond,
    }

    stream := &Stream{
        id:        1,
        conn:      mockConn,
        bandwidth: 100.0,  // 100 Mbps
        state:     StateActive,
        metrics:   &StreamMetrics{},
    }

    // Test successful send
    ctx := context.Background()
    data := make([]byte, 64*1024)  // 64 KB
    err := stream.Send(ctx, data)

    require.NoError(t, err)
    assert.Equal(t, int64(64*1024), stream.metrics.BytesSent)
    assert.Equal(t, int64(1), stream.metrics.PacketsSent)
}

func TestStreamSendWithCongestion(t *testing.T) {
    mockConn := &mockTCPConn{
        writeDelay: 100 * time.Millisecond,  // Simulate slow network
    }

    stream := &Stream{
        id:        1,
        conn:      mockConn,
        bandwidth: 10.0,   // Low bandwidth (10 Mbps)
        state:     StateActive,
        metrics:   &StreamMetrics{},
    }

    ctx := context.Background()
    data := make([]byte, 1024*1024)  // 1 MB

    start := time.Now()
    err := stream.Send(ctx, data)
    duration := time.Since(start)

    require.NoError(t, err)

    // Should take longer due to pacing
    assert.Greater(t, duration, 500*time.Millisecond)
}
```

### 6.2 Integration Testing

**Example: End-to-End VM Migration Test**

```go
// tests/integration/migration_test.go
package integration

import (
    "context"
    "testing"
    "time"

    "github.com/stretchr/testify/require"
    pb "github.com/your-org/dwcp-v3/api/proto/v3"
)

func TestVMMigration(t *testing.T) {
    if testing.Short() {
        t.Skip("Skipping integration test")
    }

    // Setup test cluster (3 controllers, 10 workers)
    cluster := setupTestCluster(t)
    defer cluster.Teardown()

    client := cluster.Client()

    // 1. Create test VM
    ctx := context.Background()
    vm, err := client.CreateVM(ctx, &pb.CreateVMRequest{
        Name:  "test-vm",
        Image: "ubuntu-22.04",
        Resources: &pb.Resources{
            Cpu:         2,
            MemoryBytes: 2 * 1024 * 1024 * 1024,  // 2 GB
            DiskBytes:   20 * 1024 * 1024 * 1024, // 20 GB
        },
    })
    require.NoError(t, err)
    require.NotEmpty(t, vm.VmId)

    // 2. Wait for VM to be ready
    waitForVMReady(t, client, vm.VmId, 60*time.Second)

    // 3. Start migration
    sourceNode := vm.NodeId
    targetNode := cluster.GetAlternativeNode(sourceNode)

    stream, err := client.MigrateVM(ctx, &pb.MigrateVMRequest{
        VmId:       vm.VmId,
        TargetNode: targetNode,
        Options: &pb.MigrationOptions{
            MaxDowntimeMs:    20000,
            CompressionLevel: 5,
            AmstStreams:      8,
        },
    })
    require.NoError(t, err)

    // 4. Monitor migration progress
    var finalStatus string
    var totalBytes int64
    migrationStart := time.Now()

    for {
        progress, err := stream.Recv()
        if err == io.EOF {
            break
        }
        require.NoError(t, err)

        t.Logf("Migration progress: %d%% (%d/%d bytes, %.2f Mbps)",
            progress.ProgressPercent,
            progress.BytesTransferred,
            progress.BytesTotal,
            progress.BandwidthMbps)

        finalStatus = progress.Status
        totalBytes = progress.BytesTotal

        if progress.Status == "completed" || progress.Status == "failed" {
            break
        }
    }

    migrationDuration := time.Since(migrationStart)

    // 5. Assert migration success
    require.Equal(t, "completed", finalStatus)
    require.Greater(t, totalBytes, int64(0))

    // 6. Verify VM is on target node
    vmInfo, err := client.GetVM(ctx, &pb.GetVMRequest{VmId: vm.VmId})
    require.NoError(t, err)
    require.Equal(t, targetNode, vmInfo.NodeId)

    // 7. Verify performance metrics
    assert.Less(t, migrationDuration, 120*time.Second, "Migration took too long")
    assert.Greater(t, totalBytes, int64(1*1024*1024*1024), "Expected ~2GB transferred")

    // 8. Cleanup
    _, err = client.DeleteVM(ctx, &pb.DeleteVMRequest{VmId: vm.VmId})
    require.NoError(t, err)
}
```

### 6.3 Performance Testing

**Example: Benchmark AMST Throughput**

```go
// backend/core/network/dwcp_v3/transport/amst_bench_test.go
package transport

import (
    "context"
    "testing"
)

func BenchmarkAMSTThroughput(b *testing.B) {
    amst := setupAMST(8)  // 8 streams
    ctx := context.Background()
    data := make([]byte, 1024*1024)  // 1 MB

    b.ResetTimer()
    b.SetBytes(int64(len(data)))

    for i := 0; i < b.N; i++ {
        err := amst.Send(ctx, data)
        if err != nil {
            b.Fatal(err)
        }
    }
}

func BenchmarkAMSTStreamScaling(b *testing.B) {
    testCases := []int{1, 2, 4, 8, 16}

    for _, numStreams := range testCases {
        b.Run(fmt.Sprintf("streams=%d", numStreams), func(b *testing.B) {
            amst := setupAMST(numStreams)
            ctx := context.Background()
            data := make([]byte, 1024*1024)

            b.ResetTimer()
            b.SetBytes(int64(len(data)))

            for i := 0; i < b.N; i++ {
                err := amst.Send(ctx, data)
                if err != nil {
                    b.Fatal(err)
                }
            }
        })
    }
}
```

---

## 7. Debugging Techniques

### 7.1 Distributed Tracing

**Add Tracing to Your Application:**

```go
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

func migrateVMWithTracing(ctx context.Context, vmID, targetNode string) error {
    tracer := otel.Tracer("vm-migration")
    ctx, span := tracer.Start(ctx, "migrate_vm")
    defer span.End()

    span.SetAttributes(
        attribute.String("vm.id", vmID),
        attribute.String("target.node", targetNode),
    )

    // Your migration logic here
    err := performMigration(ctx, vmID, targetNode)
    if err != nil {
        span.RecordError(err)
        return err
    }

    return nil
}
```

### 7.2 Logging Best Practices

**Structured Logging Example:**

```go
import "go.uber.org/zap"

logger, _ := zap.NewProduction()
defer logger.Sync()

logger.Info("Starting VM migration",
    zap.String("vm_id", vmID),
    zap.String("source_node", sourceNode),
    zap.String("target_node", targetNode),
    zap.Int("compression_level", 5),
)

// During migration
logger.Debug("Migration progress",
    zap.String("vm_id", vmID),
    zap.Int("progress_percent", 45),
    zap.Float64("bandwidth_mbps", 450.2),
)

// On error
logger.Error("Migration failed",
    zap.String("vm_id", vmID),
    zap.Error(err),
    zap.Duration("elapsed", duration),
)
```

### 7.3 Debugging Tools

**1. DWCP CLI Debug Mode:**
```bash
dwcp-cli migration status --vm vm-12345 --debug
```

**2. gRPC Reflection:**
```bash
grpcurl -plaintext localhost:8081 list
grpcurl -plaintext localhost:8081 describe dwcp.v3.VMService
```

**3. Prometheus Queries:**
```promql
# Check migration duration trend
rate(dwcp_migration_duration_seconds_sum[5m]) / rate(dwcp_migration_duration_seconds_count[5m])

# Check AMST stream health
avg(dwcp_amst_streams_active) by (node)
```

---

## 8. Best Practices

### 8.1 Error Handling

**Always handle errors explicitly:**

```go
// ❌ Bad: Ignoring errors
vm, _ := client.CreateVM(ctx, req)

// ✅ Good: Proper error handling
vm, err := client.CreateVM(ctx, req)
if err != nil {
    if status.Code(err) == codes.ResourceExhausted {
        // Retry with backoff
        time.Sleep(30 * time.Second)
        vm, err = client.CreateVM(ctx, req)
    }
    if err != nil {
        return fmt.Errorf("failed to create VM: %w", err)
    }
}
```

### 8.2 Resource Management

**Always cleanup resources:**

```go
// ✅ Good: Use defer for cleanup
conn, err := grpc.Dial(endpoint, opts...)
if err != nil {
    return err
}
defer conn.Close()

// ✅ Good: Cancel contexts
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()
```

### 8.3 Concurrency

**Use goroutines wisely:**

```go
// ✅ Good: Limit concurrency
const maxConcurrent = 10
semaphore := make(chan struct{}, maxConcurrent)

for _, vm := range vms {
    semaphore <- struct{}{}  // Acquire
    go func(v VM) {
        defer func() { <-semaphore }()  // Release
        migrateVM(ctx, v.ID)
    }(vm)
}
```

---

## 9. Advanced Topics

### 9.1 Custom Transport Plugin

**See:** `examples/custom-transport-plugin/`

### 9.2 Custom Placement Algorithm

**See:** `examples/custom-placement-algorithm/`

### 9.3 Extending DWCP with Webhooks

**See:** `examples/webhook-integration/`

---

## 10. Hands-On Exercises

See `docs/training/labs/developer/` for complete lab guides.

**Lab 1:** Build Simple VM Manager (CLI tool)
**Lab 2:** Implement Custom Transport Plugin
**Lab 3:** Build Placement Scheduler
**Lab 4:** Add Distributed Tracing
**Lab 5:** Performance Profiling

---

## 11. Certification Assessment

**Written Exam:** 40 questions, 90 minutes, 80% pass
**Practical:** Build complete application (4 hours)

---

## 12. Additional Resources

- **API Reference:** `docs/deployment/DWCP_V3_API_REFERENCE.md`
- **Code Examples:** `examples/`
- **SDK Documentation:** `https://pkg.go.dev/github.com/your-org/dwcp-v3-go-sdk`

---

**End of Developer Training Manual**

**Next Steps:**
1. Complete hands-on labs
2. Build sample application
3. Take certification exam
4. Join developer community (Slack: #dwcp-v3-dev)

**Questions?** Contact: dev-support@example.com
