# NovaCron Initialization Requirements Specification

**Version:** 1.0
**Date:** 2025-11-10
**Architect:** System Architecture Designer
**Status:** Complete - Ready for Implementation

---

## Executive Summary

This document specifies the complete functional and non-functional requirements for the NovaCron initialization system. These requirements are derived from the Architecture State Assessment and v2.0 design specification.

---

## 1. Functional Requirements

### FR-001: Environment Detection
**Priority:** CRITICAL
**Description:** System must automatically detect deployment environment

**Requirements:**
- MUST detect datacenter/internet/hybrid mode automatically
- MUST complete detection within 2 seconds
- MUST support manual override via configuration flag
- MUST measure latency (ICMP ping or TCP RTT)
- MUST measure bandwidth (historical throughput)
- MUST apply thresholds: datacenter (<10ms, >1Gbps), internet (>50ms or <1Gbps)

**Acceptance Criteria:**
- ✅ Detection accuracy: >95%
- ✅ Detection time: <2 seconds
- ✅ Override flag works correctly
- ✅ Mode transitions logged

### FR-002: Configuration Loading
**Priority:** CRITICAL
**Description:** System must load and validate configuration from multiple sources

**Requirements:**
- MUST load configuration from YAML/JSON files
- MUST support environment variable overrides
- MUST support command-line flag overrides
- MUST validate configuration schema before proceeding
- MUST fail fast on invalid configuration
- MUST apply precedence: CLI flags > Env vars > Config files > Defaults
- MUST log configuration source and values (excluding secrets)

**Acceptance Criteria:**
- ✅ All configuration formats supported (YAML, JSON, env, flags)
- ✅ Schema validation passes
- ✅ Precedence order respected
- ✅ Secrets not logged
- ✅ Invalid config fails fast with clear error

### FR-003: Component Initialization
**Priority:** CRITICAL
**Description:** System must initialize components in correct order with parallelization

**Requirements:**
- MUST initialize components in dependency order
- MUST parallelize independent components within same level
- MUST complete initialization in 15-25 seconds (target), max 30 seconds
- MUST provide progress logging for each phase and component
- MUST track initialization metrics (time per component, phase)
- MUST support retry for retriable errors (max 3 attempts)

**Acceptance Criteria:**
- ✅ All components initialized successfully
- ✅ Dependency order respected
- ✅ Parallel execution working (2.8-4.4x speedup)
- ✅ Boot time: 15-25s (95th percentile), max 30s
- ✅ Progress logging clear and actionable

### FR-004: Error Handling
**Priority:** CRITICAL
**Description:** System must handle errors appropriately based on criticality

**Requirements:**
- MUST halt initialization on critical component failures
- MUST retry retriable errors (max 3 attempts, exponential backoff 1s/2s/4s)
- MUST gracefully degrade for non-critical component failures
- MUST log all errors with structured context (component, phase, cause)
- MUST distinguish error categories: critical, degraded, warning
- MUST provide actionable error messages

**Critical Components (fail-fast):**
- SecurityComponent
- DatabaseComponent
- NetworkComponent
- DWCPComponent

**Degraded Components (retry + warn):**
- CacheComponent (fallback to in-memory)
- MLComponent (fallback to simple algorithms)

**Warning Components (log + continue):**
- MonitoringComponent
- MLComponent model loading

**Acceptance Criteria:**
- ✅ Critical failures halt initialization
- ✅ Retriable errors retried correctly
- ✅ Non-critical failures allow continued operation
- ✅ Error context complete and actionable
- ✅ Recovery strategies documented

### FR-005: Health Checks
**Priority:** HIGH
**Description:** System must implement health checks for all components

**Requirements:**
- MUST implement health check interface for all components
- MUST expose health endpoint (/health) returning 200 OK or 503 Unavailable
- MUST support readiness checks (/ready) indicating system ready for traffic
- MUST support liveness checks (/live) indicating system responsive
- MUST cache health check results (TTL: 5 seconds)
- MUST return component-level health details

**Health Check Response:**
```json
{
  "status": "healthy" | "unhealthy",
  "components": {
    "Security": {
      "state": "healthy",
      "message": "OK",
      "last_check": "2025-11-10T23:00:00Z"
    },
    "Database": {
      "state": "healthy",
      "message": "OK",
      "last_check": "2025-11-10T23:00:00Z"
    }
  }
}
```

**Acceptance Criteria:**
- ✅ All components implement HealthCheck()
- ✅ Health endpoint returns correct status
- ✅ Readiness endpoint accurate
- ✅ Liveness endpoint responsive
- ✅ Caching reduces load

### FR-006: DWCP v3 Integration
**Priority:** CRITICAL
**Description:** System must initialize all DWCP v3 components correctly

**Requirements:**
- MUST initialize all 6 DWCP v3 components: AMST, HDE, PBA, ASS, ACP, ITP
- MUST configure DWCP based on detected mode (datacenter/internet/hybrid)
- MUST support mode switching at runtime (with component reconfiguration)
- MUST collect DWCP metrics (transport throughput, compression ratio, prediction accuracy, sync latency, consensus time, placement quality)
- MUST handle DWCP component failures appropriately

**Component-Specific Requirements:**

**AMST (Transport):**
- Initialize RDMA/TCP based on mode
- Configure stream count (datacenter: 32-512, internet: 4-16)
- Set up congestion control (datacenter: CUBIC, internet: BBR)

**HDE (Encoding):**
- Initialize compression algorithms (LZ4, zstd, zstd-max)
- Set up delta encoding baseline cache
- Configure ML compression selector

**PBA (Prediction):**
- Load ONNX model for bandwidth prediction
- Initialize datacenter/internet/ensemble predictors
- Set up prediction metrics collection

**ASS (Synchronization):**
- Initialize Raft/CRDT based on mode
- Set up vector clocks
- Configure replication parameters

**ACP (Consensus):**
- Initialize Raft/PBFT based on mode
- Configure quorum sizes
- Set up gossip protocol

**ITP (Placement):**
- Load ONNX model for task placement
- Initialize DQN/geographic placement
- Configure placement policies

**Acceptance Criteria:**
- ✅ All 6 components initialized
- ✅ Mode-aware configuration applied
- ✅ Mode switching works correctly
- ✅ Metrics collected and exported
- ✅ Component failures handled gracefully

### FR-007: ML Model Integration
**Priority:** HIGH
**Description:** System must integrate ML models for PBA and ITP

**Requirements:**
- MUST load ONNX models for PBA (bandwidth prediction) and ITP (task placement)
- MUST initialize ONNX runtime correctly
- MUST handle model loading failures gracefully (fallback to simple algorithms)
- MUST collect model inference metrics (latency, accuracy)
- MUST support model hot-reload (for updates)

**Model Paths:**
- PBA datacenter: `models/pba_lstm_datacenter.onnx`
- PBA internet: `models/pba_lstm_internet.onnx`
- ITP datacenter: `models/itp_dqn_datacenter.onnx`
- ITP internet: `models/itp_geographic.onnx`

**Fallback Algorithms:**
- PBA: Simple moving average (last 10 samples)
- ITP: Round-robin or geographic placement

**Acceptance Criteria:**
- ✅ ONNX models loaded successfully
- ✅ Inference working (<1ms per prediction)
- ✅ Fallback algorithms work correctly
- ✅ Metrics collected
- ✅ Hot-reload supported

---

## 2. Non-Functional Requirements

### NFR-001: Performance
**Priority:** CRITICAL

**Requirements:**
- Boot time: 15-25 seconds (95th percentile), max 30 seconds
- Memory usage: <2GB during initialization
- CPU usage: <80% during initialization (averaged)
- Parallel speedup: 2.8-4.4x vs sequential initialization
- Health check latency: <10ms (cached), <100ms (uncached)
- Configuration loading: <500ms
- Environment detection: <2 seconds

**Acceptance Criteria:**
- ✅ Boot time meets target (95% of boots in 15-25s)
- ✅ Memory usage stays below 2GB
- ✅ CPU usage reasonable (<80% average)
- ✅ Parallel speedup achieved
- ✅ Latency targets met

### NFR-002: Reliability
**Priority:** CRITICAL

**Requirements:**
- Initialization success rate: >99.9%
- Retry success rate: >95% (for retriable errors)
- No memory leaks during initialization
- No goroutine leaks during initialization
- Graceful shutdown: all components shut down cleanly
- Component failure isolation: one component failure does not cascade

**Acceptance Criteria:**
- ✅ Success rate >99.9% over 1000 boots
- ✅ Retry mechanism effective
- ✅ Memory leak tests pass (valgrind, pprof)
- ✅ Goroutine leak tests pass (goleak)
- ✅ Shutdown graceful (max 10s)
- ✅ Failures isolated

### NFR-003: Observability
**Priority:** HIGH

**Requirements:**
- Structured logging (JSON format) for all initialization events
- Metrics for each initialization phase (duration, success/failure)
- Metrics for each component (init time, health status)
- Tracing support (OpenTelemetry) for initialization flow
- Error context complete (component, phase, cause, stack trace)
- Metrics exported to Prometheus

**Key Metrics:**
- `novacron_init_phase_duration_seconds{phase="pre_init|core|services|post_init"}`
- `novacron_init_component_duration_seconds{component="security|database|..."}`
- `novacron_init_success_total{phase="..."}`
- `novacron_init_failures_total{phase="...",component="...",error_type="..."}`
- `novacron_component_health{component="...",state="healthy|degraded|unhealthy"}`

**Acceptance Criteria:**
- ✅ All events logged (JSON format)
- ✅ Metrics collected and exported
- ✅ Tracing working (OpenTelemetry)
- ✅ Error context actionable
- ✅ Dashboards available (Grafana)

### NFR-004: Maintainability
**Priority:** HIGH

**Requirements:**
- Component interface compliance (all components implement Component interface)
- Unit test coverage: >80%
- Integration test coverage: >70%
- Documentation for each component (initialization, configuration, dependencies)
- Code style compliance (gofmt, golint)
- Dependency management (go.mod up to date)

**Acceptance Criteria:**
- ✅ All components comply with interface
- ✅ Test coverage targets met
- ✅ Documentation complete
- ✅ Code style consistent
- ✅ Dependencies managed

### NFR-005: Security
**Priority:** CRITICAL

**Requirements:**
- TLS for all network connections (min TLS 1.2)
- Encrypted credentials at rest (AES-256)
- No secrets in logs (redacted)
- Audit trail for initialization events (who, what, when, result)
- Least privilege: components run with minimal required permissions
- Security component initialized before any network operations

**Acceptance Criteria:**
- ✅ TLS enforced
- ✅ Credentials encrypted
- ✅ Secrets redacted from logs
- ✅ Audit trail complete
- ✅ Least privilege enforced
- ✅ Security-first initialization

---

## 3. Component-Specific Requirements

### 3.1 SecurityComponent

**Responsibilities:**
- Initialize TLS certificates and keys
- Configure authentication providers (OIDC, LDAP, local)
- Set up encryption for credentials at rest
- Initialize audit logging

**Configuration:**
```yaml
security:
  tls:
    cert_file: /etc/novacron/certs/server.crt
    key_file: /etc/novacron/certs/server.key
    ca_file: /etc/novacron/certs/ca.crt
    min_version: "1.2"
  auth:
    providers:
      - type: oidc
        issuer: https://auth.example.com
        client_id: novacron
        client_secret: ${OIDC_CLIENT_SECRET}
      - type: local
        users_file: /etc/novacron/users.yaml
  encryption:
    algorithm: aes-256-gcm
    key_file: /etc/novacron/secrets/encryption.key
  audit:
    enabled: true
    log_file: /var/log/novacron/audit.log
```

**Dependencies:** None (Level 0)

**Initialization Steps:**
1. Load TLS certificates (validate expiry, chain)
2. Initialize encryption keys (or generate if missing)
3. Configure authentication providers (validate connectivity)
4. Set up audit logging

**Error Handling:** CRITICAL (fail-fast)

### 3.2 DatabaseComponent

**Responsibilities:**
- Establish database connection (PostgreSQL or distributed DB)
- Run migrations to latest schema
- Verify connection pool health
- Set up query logging

**Configuration:**
```yaml
database:
  driver: postgres
  host: localhost
  port: 5432
  database: novacron
  user: novacron
  password: ${DB_PASSWORD}
  sslmode: require
  max_connections: 100
  max_idle: 10
  max_lifetime: 1h
  migrations:
    enabled: true
    path: /etc/novacron/migrations
```

**Dependencies:** SecurityComponent (for TLS)

**Initialization Steps:**
1. Establish database connection with TLS
2. Verify connection (ping)
3. Run migrations (if enabled)
4. Configure connection pool
5. Set up query logging

**Error Handling:** CRITICAL (fail-fast)

### 3.3 CacheComponent

**Responsibilities:**
- Establish cache connection (Redis or distributed cache)
- Verify cache health
- Configure eviction policies
- Set up metrics collection

**Configuration:**
```yaml
cache:
  driver: redis
  host: localhost
  port: 6379
  password: ${REDIS_PASSWORD}
  database: 0
  tls: true
  max_retries: 3
  pool_size: 100
  eviction_policy: lru
  max_memory: 2GB
```

**Dependencies:** SecurityComponent (for TLS)

**Initialization Steps:**
1. Establish cache connection with TLS
2. Verify connection (ping)
3. Configure eviction policies
4. Set up metrics collection

**Error Handling:** DEGRADED (retry, then warn, fallback to in-memory cache)

### 3.4 NetworkComponent

**Responsibilities:**
- Initialize network stack
- Configure listeners (HTTP, gRPC, metrics)
- Set up connection pools
- Configure firewalls/security groups

**Configuration:**
```yaml
network:
  listeners:
    - protocol: https
      address: 0.0.0.0:8443
      tls: true
    - protocol: grpc
      address: 0.0.0.0:9090
      tls: true
    - protocol: http
      address: 127.0.0.1:9091
      tls: false  # Metrics endpoint (internal)
  connection_pools:
    max_connections: 1000
    max_idle: 100
    max_lifetime: 1h
  firewalls:
    enabled: true
    rules:
      - allow: 0.0.0.0/0:8443  # Public API
      - allow: 10.0.0.0/8:9090  # Internal gRPC
      - allow: 127.0.0.1:9091  # Metrics
```

**Dependencies:** SecurityComponent (for TLS)

**Initialization Steps:**
1. Initialize network stack
2. Configure listeners with TLS
3. Set up connection pools
4. Configure firewalls/security groups

**Error Handling:** CRITICAL (fail-fast)

### 3.5 DWCPComponent

**Responsibilities:**
- Detect network mode (datacenter/internet/hybrid)
- Initialize all 6 DWCP v3 components (AMST, HDE, PBA, ASS, ACP, ITP)
- Configure components based on mode
- Collect DWCP metrics

**Configuration:** See `config/dwcp-v3-*.yaml`

**Dependencies:** NetworkComponent

**Initialization Steps:**
1. Detect network mode
2. Initialize AMST v3 (transport layer)
3. Initialize HDE v3 (encoding layer)
4. Initialize PBA v3 (prediction layer, load ONNX model)
5. Initialize ASS v3 (sync layer)
6. Initialize ACP v3 (consensus layer)
7. Initialize ITP v3 (placement layer, load ONNX model)
8. Set up metrics collection

**Error Handling:**
- DWCP core (AMST, HDE, ASS, ACP): CRITICAL (fail-fast)
- ML components (PBA, ITP): DEGRADED (warn, use fallback algorithms)

### 3.6 OrchestratorComponent

**Responsibilities:**
- Initialize VM scheduler (using DWCP ITP for placement)
- Load existing VMs from database
- Reconcile VM states
- Start scheduler loop

**Configuration:**
```yaml
orchestrator:
  scheduler:
    interval: 5s
    max_concurrent: 100
  vm_reconciliation:
    enabled: true
    interval: 30s
  placement:
    strategy: dwcp-itp  # or round-robin, geographic
```

**Dependencies:** DWCPComponent, DatabaseComponent

**Initialization Steps:**
1. Initialize VM scheduler (using DWCP ITP)
2. Load existing VMs from database
3. Reconcile VM states (ensure actual matches desired)
4. Start scheduler loop

**Error Handling:** CRITICAL (fail-fast)

### 3.7 APIComponent

**Responsibilities:**
- Set up API routes (REST, gRPC)
- Start HTTP/gRPC servers
- Configure middleware (auth, logging, rate limiting)
- Set up API documentation (Swagger/OpenAPI)

**Configuration:**
```yaml
api:
  rest:
    enabled: true
    address: 0.0.0.0:8443
    tls: true
  grpc:
    enabled: true
    address: 0.0.0.0:9090
    tls: true
  middleware:
    auth: true
    logging: true
    rate_limiting:
      enabled: true
      requests_per_minute: 1000
  documentation:
    enabled: true
    path: /api/docs
```

**Dependencies:** DatabaseComponent, NetworkComponent, OrchestratorComponent

**Initialization Steps:**
1. Set up API routes
2. Configure middleware
3. Start HTTP/gRPC servers
4. Set up API documentation

**Error Handling:** CRITICAL (fail-fast)

### 3.8 MonitoringComponent

**Responsibilities:**
- Initialize Prometheus registry
- Register component metrics
- Start metrics exporter
- Set up health check endpoints

**Configuration:**
```yaml
monitoring:
  prometheus:
    enabled: true
    address: 127.0.0.1:9091
    path: /metrics
  health_checks:
    enabled: true
    cache_ttl: 5s
  tracing:
    enabled: true
    exporter: otlp
    endpoint: localhost:4317
```

**Dependencies:** NetworkComponent

**Initialization Steps:**
1. Initialize Prometheus registry
2. Register component metrics
3. Start metrics exporter
4. Set up health check endpoints
5. Initialize tracing (OpenTelemetry)

**Error Handling:** WARNING (log, continue)

### 3.9 MLComponent

**Responsibilities:**
- Load ONNX models (PBA, ITP)
- Initialize ONNX runtime
- Set up model inference endpoints
- Collect model metrics

**Configuration:**
```yaml
ml:
  enabled: true
  models:
    pba:
      datacenter: models/pba_lstm_datacenter.onnx
      internet: models/pba_lstm_internet.onnx
    itp:
      datacenter: models/itp_dqn_datacenter.onnx
      internet: models/itp_geographic.onnx
  onnx_runtime:
    threads: 4
    device: cpu  # or gpu
  fallback:
    enabled: true
    algorithms:
      pba: moving_average
      itp: geographic
```

**Dependencies:** DatabaseComponent, DWCPComponent

**Initialization Steps:**
1. Initialize ONNX runtime
2. Load ONNX models (PBA, ITP)
3. Verify model inference (test prediction)
4. Set up model metrics collection

**Error Handling:** DEGRADED (warn, use fallback algorithms)

---

## 4. Testing Requirements

### 4.1 Unit Tests

**Coverage Target:** >80%

**Required Tests:**
- Component initialization (happy path, error cases)
- Dependency resolution
- Retry logic
- Health checks
- Configuration loading and validation
- Error handling and recovery

### 4.2 Integration Tests

**Coverage Target:** >70%

**Required Tests:**
- Full initialization flow (all phases)
- Parallel component initialization
- DWCP v3 integration
- ML model integration (ONNX)
- Health check endpoints
- Error scenarios (component failures)

### 4.3 Performance Tests

**Required Benchmarks:**
- Boot time (p50, p95, p99)
- Memory usage during initialization
- CPU usage during initialization
- Parallel speedup (vs sequential)
- Health check latency

### 4.4 E2E Tests

**Required Scenarios:**
- Fresh installation (no existing data)
- Upgrade (existing data, migrations)
- Configuration changes (hot reload)
- Failure recovery (component crashes)
- Mode switching (datacenter ↔ internet)

---

## 5. Documentation Requirements

**Required Documentation:**
- Architecture overview (this document)
- Component initialization guide (per component)
- Configuration reference (all parameters)
- Troubleshooting guide (common issues, solutions)
- API documentation (REST, gRPC)
- Metrics reference (all metrics, descriptions)

---

## 6. Acceptance Criteria Summary

### Phase 1: Pre-Initialization (2-5s)
- ✅ Environment detected correctly
- ✅ Configuration loaded and validated
- ✅ Logger operational
- ✅ Minimum resources available

### Phase 2: Core Initialization (5-10s)
- ✅ SecurityComponent initialized
- ✅ DatabaseComponent initialized
- ✅ CacheComponent initialized (or fallback)
- ✅ NetworkComponent initialized
- ✅ DWCPComponent initialized (all 6 subcomponents)

### Phase 3: Services Initialization (10-20s)
- ✅ OrchestratorComponent initialized
- ✅ APIComponent initialized
- ✅ MonitoringComponent initialized
- ✅ MLComponent initialized (or fallback)

### Phase 4: Post-Initialization (20-25s)
- ✅ Health checks passing
- ✅ Readiness probe active
- ✅ Metrics exported
- ✅ System ready for traffic

### Overall System
- ✅ Boot time: 15-25s (95th percentile)
- ✅ Initialization success rate: >99.9%
- ✅ Test coverage: >80% unit, >70% integration
- ✅ Documentation complete
- ✅ All requirements met

---

**Document Status:** Complete - Ready for Implementation
**Next Steps:** Begin Phase 2 Week 1 (Core Components Implementation)
**Maintained By:** System Architecture Designer
