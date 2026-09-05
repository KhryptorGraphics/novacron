# NovaCron Microservices Decomposition Blueprint

## Overview

This blueprint provides detailed guidance for decomposing the current NovaCron monolithic API server into a scalable microservices architecture while maintaining system reliability and performance.

## Current State Analysis

### Existing Monolithic Structure
```
main.go (722 lines) handles:
├── Authentication & Authorization (auth.AuthManager)
├── VM Management (core_vm.VMManager)
├── KVM Hypervisor Management (hypervisor.KVMManager)
├── Federation Management (federation.FederationManager)
├── Multi-Cloud Orchestration (multicloud.UnifiedOrchestrator)
├── Database Management (PostgreSQL connections)
├── Monitoring & Metrics (mock handlers)
├── HTTP Routing & Middleware
└── Health Checks & API Info
```

### Problems with Current Architecture
1. **Single Point of Failure**: All functionality in one process
2. **Scaling Inefficiency**: Cannot scale individual components independently
3. **Development Complexity**: Changes require full system redeployment
4. **Resource Waste**: Over-provisioning for peak loads across all functions
5. **Technology Lock-in**: All components must use same technology stack

## Target Microservices Architecture

### Service Boundary Design

#### 1. **Authentication Service** (`novacron-auth-service`)
```go
// Responsibilities
- User authentication & authorization
- JWT token management & validation
- Role-based access control (RBAC)
- Multi-tenancy support
- API key management for service-to-service communication

// Technology Stack
- Go with Gin framework
- PostgreSQL for user data
- Redis for session management
- JWT with RS256 signing

// Database Schema
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    tenant_id UUID NOT NULL,
    mfa_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id),
    permissions JSONB,
    expires_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

#### 2. **VM Service** (`novacron-vm-service`)
```go
// Responsibilities  
- VM lifecycle management (create, start, stop, delete)
- VM configuration and state management
- Direct hypervisor integration (KVM, containers)
- Resource allocation and validation
- VM metrics collection

// Technology Stack
- Go with gRPC for internal APIs, HTTP for external
- PostgreSQL for VM state persistence
- Redis for real-time VM status caching
- Direct libvirt integration for KVM

// Database Schema
CREATE TABLE vms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    state vm_state_enum NOT NULL,
    hypervisor_type VARCHAR(50) NOT NULL,
    node_id UUID REFERENCES nodes(id),
    owner_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    resource_config JSONB NOT NULL,
    network_config JSONB,
    storage_config JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TYPE vm_state_enum AS ENUM ('creating', 'running', 'stopped', 'paused', 'migrating', 'error');
```

#### 3. **Federation Service** (`novacron-federation-service`)
```go
// Responsibilities
- Multi-cloud provider management
- Cross-cloud VM migration orchestration
- Provider health monitoring & failover
- Compliance policy enforcement
- Cost optimization recommendations

// Technology Stack
- Go with gRPC + HTTP REST
- PostgreSQL for federation state
- Redis for provider health caching
- Message queue (NATS) for async operations

// Database Schema
CREATE TABLE cloud_providers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    provider_type VARCHAR(50) NOT NULL,
    regions JSONB NOT NULL,
    credentials_encrypted TEXT NOT NULL,
    config JSONB,
    health_status VARCHAR(20) DEFAULT 'unknown',
    last_health_check TIMESTAMP,
    enabled BOOLEAN DEFAULT true
);

CREATE TABLE migrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID NOT NULL,
    source_provider UUID REFERENCES cloud_providers(id),
    destination_provider UUID REFERENCES cloud_providers(id),
    migration_plan JSONB NOT NULL,
    status migration_status_enum NOT NULL,
    progress_percentage INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    error_details JSONB
);
```

#### 4. **Monitoring Service** (`novacron-monitoring-service`)
```go
// Responsibilities
- Metrics collection and aggregation
- Real-time alerting and notifications
- Performance analytics and reporting
- System health monitoring
- Log aggregation and analysis

// Technology Stack
- Go with Prometheus client libraries
- TimescaleDB for time-series metrics
- Redis for real-time alerting
- Grafana for visualization
- Elasticsearch for log aggregation

// Database Schema (TimescaleDB)
CREATE TABLE vm_metrics (
    time TIMESTAMPTZ NOT NULL,
    vm_id UUID NOT NULL,
    cpu_usage DOUBLE PRECISION,
    memory_usage DOUBLE PRECISION,
    network_bytes_sent BIGINT,
    network_bytes_recv BIGINT,
    disk_read_bytes BIGINT,
    disk_write_bytes BIGINT,
    disk_usage_percentage DOUBLE PRECISION
);

SELECT create_hypertable('vm_metrics', 'time');
```

#### 5. **Scheduler Service** (`novacron-scheduler-service`)
```go
// Responsibilities
- Resource allocation optimization
- VM placement decisions
- Load balancing across nodes
- Predictive scaling recommendations
- Policy-based scheduling rules

// Technology Stack
- Go with machine learning libraries
- PostgreSQL for scheduling policies
- Redis for real-time resource state
- Message queue for scheduling events

// Database Schema
CREATE TABLE scheduling_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    tenant_id UUID NOT NULL,
    rules JSONB NOT NULL,
    priority INTEGER DEFAULT 0,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE resource_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id UUID NOT NULL,
    vm_id UUID NOT NULL,
    cpu_allocated INTEGER NOT NULL,
    memory_allocated BIGINT NOT NULL,
    storage_allocated BIGINT NOT NULL,
    allocated_at TIMESTAMP DEFAULT NOW()
);
```

#### 6. **API Gateway Service** (`novacron-api-gateway`)
```go
// Responsibilities
- External API routing and rate limiting
- Request authentication and authorization
- Response caching and compression
- API versioning and backward compatibility
- Circuit breaking and retry logic

// Technology Stack
- Go with Gin framework
- Redis for rate limiting and caching
- Consul for service discovery
- Prometheus for metrics

// Configuration Example
type GatewayConfig struct {
    Services map[string]ServiceConfig `yaml:"services"`
    RateLimit RateLimitConfig `yaml:"rate_limit"`
    Auth AuthConfig `yaml:"auth"`
}

type ServiceConfig struct {
    URL string `yaml:"url"`
    Timeout time.Duration `yaml:"timeout"`
    Retries int `yaml:"retries"`
    CircuitBreaker CircuitBreakerConfig `yaml:"circuit_breaker"`
}
```

## Service Communication Design

### 1. **Internal Communication (Service-to-Service)**
```go
// gRPC Protocol Definitions

// auth-service/proto/auth.proto
service AuthService {
  rpc ValidateToken(ValidateTokenRequest) returns (ValidateTokenResponse);
  rpc GetUserPermissions(GetUserPermissionsRequest) returns (GetUserPermissionsResponse);
  rpc CreateAPIKey(CreateAPIKeyRequest) returns (CreateAPIKeyResponse);
}

// vm-service/proto/vm.proto  
service VMService {
  rpc CreateVM(CreateVMRequest) returns (CreateVMResponse);
  rpc GetVMStatus(GetVMStatusRequest) returns (GetVMStatusResponse);
  rpc MigrateVM(MigrateVMRequest) returns (MigrateVMResponse);
}

// federation-service/proto/federation.proto
service FederationService {
  rpc ListProviders(ListProvidersRequest) returns (ListProvidersResponse);
  rpc InitiateMigration(InitiateMigrationRequest) returns (InitiateMigrationResponse);
  rpc GetMigrationStatus(GetMigrationStatusRequest) returns (GetMigrationStatusResponse);
}
```

### 2. **External Communication (Client-to-Service)**
```yaml
# API Gateway Routes Configuration
routes:
  # Authentication routes
  - path: /auth/*
    service: auth-service
    methods: [POST]
    rate_limit: 100/minute
    
  # VM management routes  
  - path: /api/vms/*
    service: vm-service
    methods: [GET, POST, PUT, DELETE]
    auth_required: true
    rate_limit: 1000/minute
    
  # Multi-cloud routes
  - path: /api/multicloud/*
    service: federation-service
    methods: [GET, POST, PUT, DELETE] 
    auth_required: true
    permissions: [multicloud:read, multicloud:write]
    
  # Monitoring routes
  - path: /api/monitoring/*
    service: monitoring-service
    methods: [GET]
    auth_required: true
    cache_ttl: 30s
```

### 3. **Asynchronous Communication (Message Queues)**
```go
// Event-driven architecture using NATS

// VM lifecycle events
type VMEvent struct {
    Type      string    `json:"type"`      // created, started, stopped, deleted
    VMID      string    `json:"vm_id"`
    TenantID  string    `json:"tenant_id"`
    Timestamp time.Time `json:"timestamp"`
    Metadata  map[string]interface{} `json:"metadata"`
}

// Migration events
type MigrationEvent struct {
    Type        string    `json:"type"`    // started, progress, completed, failed
    MigrationID string    `json:"migration_id"`
    VMID        string    `json:"vm_id"`
    Progress    int       `json:"progress"`
    Timestamp   time.Time `json:"timestamp"`
    Error       string    `json:"error,omitempty"`
}

// Publishers and Subscribers
publishers := map[string][]string{
    "vm-service":        {"vm.events", "vm.metrics"},
    "federation-service": {"migration.events", "provider.health"},
    "monitoring-service": {"alerts", "system.health"},
}

subscribers := map[string][]string{
    "monitoring-service": {"vm.events", "migration.events"},
    "scheduler-service":  {"vm.events", "vm.metrics"},
    "federation-service": {"vm.events"},
}
```

## Data Management Strategy

### 1. **Database-per-Service Pattern**
```yaml
services:
  auth-service:
    database: novacron_auth
    tables: [users, api_keys, sessions, permissions]
    
  vm-service:
    database: novacron_vms  
    tables: [vms, vm_templates, vm_snapshots, vm_networks]
    
  federation-service:
    database: novacron_federation
    tables: [cloud_providers, migrations, compliance_policies]
    
  monitoring-service:
    database: novacron_monitoring (TimescaleDB)
    tables: [vm_metrics, system_metrics, alerts, events]
    
  scheduler-service:
    database: novacron_scheduler
    tables: [scheduling_policies, resource_allocations, nodes]
```

### 2. **Shared Data Access Patterns**
```go
// Cross-service data access via APIs, not direct database access

// Example: VM Service needs user information
func (vs *VMService) CreateVM(ctx context.Context, req *CreateVMRequest) (*CreateVMResponse, error) {
    // Validate user permissions via auth service
    authReq := &auth.ValidateTokenRequest{Token: req.AuthToken}
    authResp, err := vs.authClient.ValidateToken(ctx, authReq)
    if err != nil {
        return nil, fmt.Errorf("authentication failed: %w", err)
    }
    
    // Create VM with validated user context
    vm := &VM{
        Name: req.Name,
        OwnerID: authResp.User.ID,
        TenantID: authResp.User.TenantID,
        Config: req.Config,
    }
    
    return vs.createVMInternal(ctx, vm)
}
```

## Deployment Strategy

### 1. **Kubernetes Deployment Configuration**
```yaml
# Example: VM Service Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-vm-service
  namespace: novacron
spec:
  replicas: 3
  selector:
    matchLabels:
      app: novacron-vm-service
  template:
    metadata:
      labels:
        app: novacron-vm-service
    spec:
      containers:
      - name: vm-service
        image: novacron/vm-service:v1.0.0
        ports:
        - containerPort: 8080  # HTTP
        - containerPort: 9090  # gRPC
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: vm-service-db-secret
              key: url
        - name: AUTH_SERVICE_URL
          value: "novacron-auth-service.novacron.svc.cluster.local:9090"
        resources:
          requests:
            cpu: 100m
            memory: 128Mi
          limits:
            cpu: 500m
            memory: 512Mi
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### 2. **Service Discovery Configuration**
```yaml
# Service Discovery with Consul
apiVersion: v1
kind: ConfigMap
metadata:
  name: consul-config
data:
  consul.hcl: |
    datacenter = "novacron-dc1"
    data_dir = "/consul/data"
    log_level = "INFO"
    server = true
    ui_config {
      enabled = true
    }
    connect {
      enabled = true
    }
    ports {
      grpc = 8502
    }
```

## Migration Plan

### Phase 1: Extract Authentication Service (Week 1)
1. Create `auth-service` with user management functionality
2. Update existing `main.go` to use auth service via gRPC
3. Migrate user data to auth service database
4. Deploy both services side-by-side with feature flags

### Phase 2: Extract VM Service (Week 2)  
1. Create `vm-service` with VM lifecycle management
2. Move KVM manager and VM manager code to vm-service
3. Update API routes to proxy to vm-service
4. Test VM operations end-to-end

### Phase 3: Extract Federation Service (Week 3)
1. Create `federation-service` with multi-cloud functionality  
2. Move federation manager and multi-cloud orchestrator
3. Update migration workflows to use federation service
4. Test cross-cloud operations

### Phase 4: Extract Monitoring Service (Week 4)
1. Create `monitoring-service` with metrics and alerting
2. Migrate mock handlers to real monitoring implementation
3. Set up TimescaleDB for time-series data
4. Configure Grafana dashboards

### Phase 5: Deploy API Gateway (Week 5)
1. Create `api-gateway` with routing and rate limiting
2. Configure service discovery and load balancing
3. Remove direct service access, route through gateway
4. Performance testing and optimization

### Phase 6: Extract Scheduler Service (Week 6)
1. Create `scheduler-service` with resource allocation logic
2. Implement predictive scaling algorithms  
3. Add machine learning capabilities for optimization
4. Full system integration testing

## Monitoring & Observability

### 1. **Service-Level Metrics**
```go
// Prometheus metrics for each service
var (
    requestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "http_request_duration_seconds",
            Help: "HTTP request duration in seconds",
        },
        []string{"service", "method", "endpoint", "status_code"},
    )
    
    activeConnections = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "active_connections",
            Help: "Number of active connections",
        },
        []string{"service", "type"}, // type: http, grpc, database
    )
    
    errorRate = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "errors_total",
            Help: "Total number of errors",
        },
        []string{"service", "type", "error_code"},
    )
)
```

### 2. **Distributed Tracing**
```go
// OpenTelemetry integration
import (
    "go.opentelemetry.io/otel"
    "go.opentelemetry.io/otel/trace"
)

func (vs *VMService) CreateVM(ctx context.Context, req *CreateVMRequest) (*CreateVMResponse, error) {
    tracer := otel.Tracer("vm-service")
    ctx, span := tracer.Start(ctx, "create-vm")
    defer span.End()
    
    // Add trace attributes
    span.SetAttributes(
        attribute.String("vm.name", req.Name),
        attribute.String("vm.type", req.Type),
        attribute.String("tenant.id", req.TenantID),
    )
    
    // Continue with business logic...
}
```

## Success Criteria

### Performance Targets
- **Service Response Time**: p95 < 100ms for individual services
- **End-to-End Latency**: p95 < 500ms for complete workflows
- **Throughput**: Handle 10,000+ concurrent requests across all services
- **Availability**: 99.9% uptime per service (99.99% overall system)

### Scalability Targets  
- **Horizontal Scaling**: Auto-scale from 2 to 50 replicas per service
- **Resource Efficiency**: 70%+ CPU/memory utilization during normal operations
- **Data Growth**: Support 1M+ VMs across all cloud providers
- **Geographic Distribution**: Deploy across 3+ regions with <100ms inter-region latency

### Operational Targets
- **Deployment Time**: < 5 minutes for individual service updates
- **Rollback Time**: < 2 minutes for emergency rollbacks  
- **Mean Time to Recovery**: < 15 minutes for service failures
- **Development Velocity**: Enable independent team development per service

This microservices blueprint provides a comprehensive roadmap for transforming NovaCron from a monolithic application into a highly scalable, maintainable, and operationally efficient distributed system.