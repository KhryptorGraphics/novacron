# NovaCron Initialization Architecture

**Architecture Decision Record (ADR-INIT-001)**

**Date:** 2025-11-14
**Status:** APPROVED
**Architect:** System Architecture Designer
**Version:** 2.0

---

## Executive Summary

This document defines the **comprehensive initialization architecture** for NovaCron's distributed VM management platform. The architecture coordinates initialization across both Node.js (frontend/orchestration) and Go (backend/DWCP) runtimes, ensuring proper dependency management, health validation, and graceful degradation.

### Key Architectural Principles

1. **Phased Initialization**: Sequential phase execution with dependency validation
2. **Dual-Runtime Coordination**: Node.js and Go runtime orchestration via health checks
3. **Component Lifecycle Management**: Standardized lifecycle for all components
4. **Configuration Hierarchy**: Environment-specific with secure override capabilities
5. **Health-First Design**: Continuous validation at each initialization phase
6. **Graceful Degradation**: Optional services fail independently without blocking critical path
7. **Event-Driven Lifecycle**: Observable initialization and shutdown sequences
8. **Security by Default**: Zero-trust initialization with credential validation

---

## Table of Contents

1. [System Context (C4 Level 1)](#1-system-context-c4-level-1)
2. [Architecture Overview (C4 Level 2)](#2-architecture-overview-c4-level-2)
3. [Component Design (C4 Level 3)](#3-component-design-c4-level-3)
4. [Initialization Phases](#4-initialization-phases)
5. [Dependency Graph](#5-dependency-graph)
6. [Configuration Schema](#6-configuration-schema)
7. [Security Architecture](#7-security-architecture)
8. [Error Handling & Recovery](#8-error-handling--recovery)
9. [Health Checks & Monitoring](#9-health-checks--monitoring)
10. [Deployment Considerations](#10-deployment-considerations)
11. [Architecture Decision Records](#11-architecture-decision-records)

---

## 1. System Context (C4 Level 1)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NovaCron Platform Bootstrap                          │
│                                                                              │
│  External Dependencies           Platform Initialization          Outputs   │
│  ┌──────────────┐              ┌──────────────────────┐        ┌─────────┐ │
│  │ Environment  │─────────────→│  Initialization      │───────→│ Ready   │ │
│  │ Variables    │              │  Orchestrator        │        │ Services│ │
│  └──────────────┘              └──────────────────────┘        └─────────┘ │
│                                          │                                   │
│  ┌──────────────┐                        │                     ┌─────────┐ │
│  │ Config Files │────────────────────────┤                     │ Health  │ │
│  │ (JSON/YAML)  │                        │                     │ Checks  │ │
│  └──────────────┘                        │                     └─────────┘ │
│                                          │                                   │
│  ┌──────────────┐                        │                     ┌─────────┐ │
│  │ PostgreSQL   │────────────────────────┤                     │ Metrics │ │
│  │ Database     │                        │                     │ Export  │ │
│  └──────────────┘                        │                     └─────────┘ │
│                                          │                                   │
│  ┌──────────────┐                        │                                  │
│  │ Redis Cache  │────────────────────────┘                                  │
│  └──────────────┘                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### System Boundaries

- **Input Boundary**: Configuration files, environment variables, external services
- **Processing Boundary**: Initialization orchestrator coordinates all components
- **Output Boundary**: Ready services, health check endpoints, metrics export

---

## 2. Architecture Overview (C4 Level 2)

### 2.1 Dual-Runtime Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     NovaCron Initialization Flow                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌────────────────────────────────┐  ┌──────────────────────────────┐   │
│  │   Node.js Initialization       │  │   Go Initialization          │   │
│  │   (Frontend/Orchestration)     │  │   (Backend/DWCP)             │   │
│  │                                │  │                              │   │
│  │  Phase 0: Pre-flight Checks    │  │  Phase 0: Environment Setup  │   │
│  │    ├─ Runtime version          │  │    ├─ Runtime version        │   │
│  │    ├─ Directories              │  │    ├─ Directories            │   │
│  │    └─ Memory availability      │  │    └─ Kernel capabilities    │   │
│  │                                │  │                              │   │
│  │  Phase 1: Configuration        │  │  Phase 1: Configuration      │   │
│  │    ├─ Load default config      │  │    ├─ Load config            │   │
│  │    ├─ Environment overrides    │  │    ├─ Validate config        │   │
│  │    └─ Validate schema          │  │    └─ Security check         │   │
│  │                                │  │                              │   │
│  │  Phase 2: Logging              │  │  Phase 2: Logging            │   │
│  │    ├─ Initialize logger        │  │    ├─ Zap logger init        │   │
│  │    ├─ Setup rotation           │  │    ├─ Structured logging     │   │
│  │    └─ Configure levels         │  │    └─ Audit logging          │   │
│  │                                │  │                              │   │
│  │  Phase 3: Database             │  │  Phase 3: Database           │   │
│  │    ├─ PostgreSQL pool          │←─┼──→├─ SQL connection         │   │
│  │    ├─ Redis client             │←─┼──→├─ Connection pool        │   │
│  │    └─ Health checks            │  │    └─ Migration check        │   │
│  │                                │  │                              │   │
│  │  Phase 4: Core Services        │  │  Phase 4: Core Components    │   │
│  │    ├─ Workload monitor         │←─┼──→├─ DWCP Manager           │   │
│  │    ├─ MCP integration          │  │    ├─ Network layer          │   │
│  │    └─ Cache manager            │  │    └─ Security manager       │   │
│  │                                │  │                              │   │
│  │  Phase 5: Optional Services    │  │  Phase 5: DWCP Components    │   │
│  │    ├─ Agent spawner            │  │    ├─ AMST transport         │   │
│  │    ├─ Auto-orchestrator        │  │    ├─ HDE compression        │   │
│  │    └─ ML services              │  │    ├─ Prediction engine      │   │
│  │                                │  │    ├─ Consensus layer        │   │
│  │                                │  │    └─ Resilience manager     │   │
│  │  Phase 6: Health & Ready       │  │  Phase 6: Health & Ready     │   │
│  │    ├─ Component validation     │←─┼──→├─ Component health       │   │
│  │    ├─ Health endpoint          │  │    ├─ Metrics collection     │   │
│  │    └─ Ready signal             │  │    └─ Ready signal           │   │
│  └────────────────────────────────┘  └──────────────────────────────┘   │
│                    │                              │                       │
│                    └──────────────────────────────┘                       │
│                               │                                           │
│                    ┌──────────▼──────────┐                                │
│                    │   Platform Ready    │                                │
│                    │  - API Available    │                                │
│                    │  - DWCP Running     │                                │
│                    │  - Health OK        │                                │
│                    └─────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Coordination Mechanism

**Health Check Synchronization**: Node.js and Go runtimes coordinate via:
- Shared database health status table
- HTTP health check endpoints
- Shared Redis cache for status updates
- Event-driven status notifications

---

## 3. Component Design (C4 Level 3)

### 3.1 Go Backend Initialization (DWCP Focus)

```go
// Component Lifecycle Interface
type Component interface {
    Name() string
    Initialize(ctx context.Context) error
    Shutdown(ctx context.Context) error
    HealthCheck(ctx context.Context) error
    Dependencies() []string
}

// Initialization Orchestrator
type Orchestrator struct {
    components map[string]*ComponentInfo
    order      []string
    logger     *zap.Logger
    metrics    *MetricsCollector
}

// Component Registration & Initialization
func (o *Orchestrator) Initialize(ctx context.Context) error {
    // 1. Build dependency graph
    order, err := o.buildInitOrder()

    // 2. Initialize in dependency order
    for _, name := range order {
        if err := o.initializeComponent(ctx, name); err != nil {
            // Rollback on failure
            o.shutdownInitialized(ctx)
            return err
        }
    }

    // 3. Verify all health checks
    return o.healthCheckAll(ctx)
}
```

### 3.2 Key Components

#### DWCP Manager Component

```go
type DWCPComponent struct {
    manager *dwcp.Manager
    config  *dwcp.Config
    logger  *zap.Logger
}

func (d *DWCPComponent) Initialize(ctx context.Context) error {
    // Phase 0: Core Infrastructure (Transport, Compression)
    if err := d.manager.startPhase0Components(ctx); err != nil {
        return err
    }

    // Phase 1: Intelligence Layer (Prediction)
    if err := d.manager.startPhase1Components(ctx); err != nil {
        return err
    }

    // Phase 2: Coordination Layer (Sync, Consensus)
    if err := d.manager.startPhase2Components(ctx); err != nil {
        return err
    }

    // Phase 3: Resilience Layer (Circuit breaker, Fallback)
    if err := d.manager.startPhase3Components(ctx); err != nil {
        return err
    }

    return nil
}

func (d *DWCPComponent) Dependencies() []string {
    return []string{"database", "logging", "security"}
}
```

#### Database Component

```go
type DatabaseComponent struct {
    pool   *sql.DB
    config *DatabaseConfig
    logger *zap.Logger
}

func (db *DatabaseComponent) Initialize(ctx context.Context) error {
    // 1. Connect to PostgreSQL
    pool, err := sql.Open("postgres", db.config.ConnectionString())
    if err != nil {
        return fmt.Errorf("failed to open database: %w", err)
    }

    // 2. Configure connection pool
    pool.SetMaxOpenConns(db.config.MaxConnections)
    pool.SetMaxIdleConns(db.config.MaxIdleConns)
    pool.SetConnMaxLifetime(db.config.ConnMaxLifetime)

    // 3. Test connection
    if err := pool.PingContext(ctx); err != nil {
        return fmt.Errorf("database ping failed: %w", err)
    }

    db.pool = pool
    return nil
}

func (db *DatabaseComponent) Dependencies() []string {
    return []string{"logging", "config"}
}
```

#### Security Component

```go
type SecurityComponent struct {
    manager     *security.SecurityManager
    encryption  *security.EncryptionManager
    audit       *security.AuditLogger
    zeroTrust   *security.ZeroTrustManager
    logger      *zap.Logger
}

func (s *SecurityComponent) Initialize(ctx context.Context) error {
    // 1. Initialize encryption
    if err := s.encryption.Initialize(); err != nil {
        return err
    }

    // 2. Initialize audit logging
    if err := s.audit.Start(); err != nil {
        return err
    }

    // 3. Initialize zero-trust network
    if s.zeroTrust.Enabled {
        if err := s.zeroTrust.Initialize(ctx); err != nil {
            return err
        }
    }

    return nil
}

func (s *SecurityComponent) Dependencies() []string {
    return []string{"database", "logging"}
}
```

### 3.3 Node.js Frontend Initialization

```javascript
class PlatformInitializer extends EventEmitter {
    async initialize() {
        // Phase 0: Pre-flight checks
        await this.performPreflightChecks();

        // Phase 1: Configuration
        await this.loadConfiguration();
        this.emit('init:config-loaded');

        // Phase 2: Logging
        await this.setupLogging();
        this.emit('init:logging-setup');

        // Phase 3: Database connections
        await this.connectDatabases();
        this.emit('init:databases-connected');

        // Phase 4: Core services
        await this.initializeCoreServices();
        this.emit('init:core-services-ready');

        // Phase 5: Optional services
        await this.initializeOptionalServices();
        this.emit('init:optional-services-ready');

        // Phase 6: Health validation
        await this.validateHealth();
        this.emit('init:complete');
    }

    async connectDatabases() {
        // PostgreSQL connection pool
        const { Pool } = require('pg');
        this.pgPool = new Pool({
            host: this.config.database.postgres.host,
            port: this.config.database.postgres.port,
            database: this.config.database.postgres.database,
            user: this.config.database.postgres.user,
            password: this.config.database.postgres.password,
            max: this.config.database.postgres.poolSize
        });

        // Test connection
        await this.pgPool.query('SELECT NOW()');

        // Redis connection
        const redis = require('redis');
        this.redisClient = redis.createClient({
            host: this.config.database.redis.host,
            port: this.config.database.redis.port
        });

        await this.redisClient.connect();
        await this.redisClient.ping();
    }
}
```

---

## 4. Initialization Phases

### 4.1 Phase Timeline

| Phase | Node.js | Go Backend | Duration | Critical |
|-------|---------|------------|----------|----------|
| **Phase 0** | Pre-flight Checks | Environment Setup | 0-2s | Yes |
| **Phase 1** | Configuration | Configuration | 2-5s | Yes |
| **Phase 2** | Logging | Logging | 1-2s | Yes |
| **Phase 3** | Database | Database | 3-8s | Yes |
| **Phase 4** | Core Services | Core Components | 5-10s | Yes |
| **Phase 5** | Optional Services | DWCP Components | 5-15s | No |
| **Phase 6** | Health Validation | Health Validation | 2-5s | Yes |
| **Total** | | | **18-47s** | |

### 4.2 Phase Descriptions

#### Phase 0: Pre-flight Checks
**Purpose**: Validate environment before any initialization

**Node.js**:
- Runtime version >= 18.0.0
- Required directories exist (`src/`, `config/`, `logs/`)
- File permissions validated
- Memory availability check (minimum 512MB free)

**Go**:
- Runtime version >= 1.21
- Kernel capabilities check (for KVM/networking)
- cgroup limits validation
- System resource availability

**Failure Handling**: Exit immediately with error code 1

#### Phase 1: Configuration Loading
**Purpose**: Load and validate all configuration

**Node.js**:
```javascript
// 1. Load default configuration
const defaultConfig = JSON.parse(
    fs.readFileSync('./src/config/config.default.json')
);

// 2. Load environment-specific config
const envConfig = JSON.parse(
    fs.readFileSync(`./src/config/config.${NODE_ENV}.json`)
);

// 3. Merge configurations
const config = mergeDeep(defaultConfig, envConfig);

// 4. Apply environment variable overrides
applyEnvOverrides(config, 'NOVACRON_');

// 5. Validate schema
validateConfigSchema(config);
```

**Go**:
```go
// 1. Load configuration file
config, err := config.Load("/etc/novacron/config.yaml")

// 2. Apply environment overrides
if err := config.LoadFromEnv(); err != nil {
    return err
}

// 3. Validate configuration
if err := config.Validate(); err != nil {
    return err
}

// 4. Security validation
if err := security.ValidateConfig(config); err != nil {
    return err
}
```

**Failure Handling**: Log error, exit with code 2

#### Phase 2: Logging System
**Purpose**: Initialize structured logging

**Node.js**: Console logger with JSON format
**Go**: Zap logger with structured fields

**Failure Handling**: Fall back to console logging

#### Phase 3: Database Connections
**Purpose**: Establish all database connections

**Critical Dependencies**:
- PostgreSQL (required)
- Redis (optional for caching)

**Connection Strategy**:
```go
// PostgreSQL with retry logic
func connectWithRetry(config *DatabaseConfig) (*sql.DB, error) {
    maxRetries := 5
    retryDelay := 2 * time.Second

    for i := 0; i < maxRetries; i++ {
        db, err := sql.Open("postgres", config.DSN())
        if err == nil {
            if err := db.Ping(); err == nil {
                return db, nil
            }
        }

        log.Warn("Database connection failed, retrying...",
            "attempt", i+1,
            "retry_in", retryDelay)

        time.Sleep(retryDelay)
        retryDelay *= 2  // Exponential backoff
    }

    return nil, fmt.Errorf("failed to connect after %d attempts", maxRetries)
}
```

**Failure Handling**: Retry with exponential backoff, fail after 5 attempts

#### Phase 4: Core Services/Components
**Purpose**: Initialize critical platform services

**Node.js Core Services**:
- Cache manager
- Workload monitor
- MCP integration layer

**Go Core Components**:
- DWCP Manager (Phase 0 only: Transport + Compression)
- Network layer initialization
- Security manager
- VM manager

**Initialization Order**:
1. Security manager (first for audit logging)
2. Network layer
3. VM manager
4. DWCP Manager (Phase 0 components only)

**Failure Handling**: Rollback all initialized components, exit

#### Phase 5: Optional Services
**Purpose**: Initialize non-critical services

**Node.js**:
- Smart agent spawner
- Auto-spawning orchestrator
- ML task classifier

**Go (DWCP Phases 1-3)**:
- Phase 1: Prediction engine (ML-based bandwidth prediction)
- Phase 2: Consensus layer (Raft/Gossip/Byzantine)
- Phase 3: Resilience manager (circuit breaker, fallback)

**Failure Handling**: Log warning, continue with degraded functionality

#### Phase 6: Health Validation
**Purpose**: Verify all components are healthy

**Health Checks**:
```go
type HealthCheck struct {
    Component string
    Status    string  // "healthy", "degraded", "unhealthy"
    Message   string
    Timestamp time.Time
}

func (o *Orchestrator) HealthCheckAll(ctx context.Context) error {
    results := make([]HealthCheck, 0)

    for name, info := range o.components {
        if err := info.Component.HealthCheck(ctx); err != nil {
            results = append(results, HealthCheck{
                Component: name,
                Status:    "unhealthy",
                Message:   err.Error(),
            })
        } else {
            results = append(results, HealthCheck{
                Component: name,
                Status:    "healthy",
            })
        }
    }

    // Check if any critical component is unhealthy
    for _, result := range results {
        if result.Status == "unhealthy" && isCritical(result.Component) {
            return fmt.Errorf("critical component unhealthy: %s", result.Component)
        }
    }

    return nil
}
```

**Failure Handling**: Exit if critical component fails health check

---

## 5. Dependency Graph

### 5.1 Visual Dependency Graph

```
┌────────────────────────────────────────────────────────────────────┐
│                      Component Dependencies                         │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Level 0: Foundation                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                         │
│  │  Config  │  │  Logger  │  │  Metrics │                         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                         │
│       │             │             │                                 │
│       └─────────────┴─────────────┘                                 │
│                     │                                               │
│  Level 1: Data Layer                                                │
│              ┌──────▼───────┐                                       │
│              │   Database   │                                       │
│              └──────┬───────┘                                       │
│                     │                                               │
│  Level 2: Security                                                  │
│              ┌──────▼───────┐                                       │
│              │   Security   │                                       │
│              └──────┬───────┘                                       │
│                     │                                               │
│  Level 3: Core Infrastructure                                       │
│       ┌─────────────┼─────────────┐                                │
│       │             │             │                                 │
│  ┌────▼────┐  ┌────▼────┐  ┌─────▼────┐                           │
│  │ Network │  │   VM    │  │  DWCP    │                           │
│  │  Layer  │  │ Manager │  │ Manager  │                           │
│  └────┬────┘  └────┬────┘  └─────┬────┘                           │
│       │            │             │                                 │
│       └────────────┴─────────────┘                                 │
│                    │                                                │
│  Level 4: Application Services                                      │
│             ┌──────▼──────┐                                        │
│             │     API     │                                        │
│             │   Gateway   │                                        │
│             └─────────────┘                                        │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 Dependency Matrix

| Component | Dependencies | Type | Phase |
|-----------|--------------|------|-------|
| Config | None | Critical | 1 |
| Logger | Config | Critical | 2 |
| Metrics | Config, Logger | Critical | 2 |
| Database | Config, Logger | Critical | 3 |
| Security | Database, Logger | Critical | 4 |
| Network Layer | Security, Config | Critical | 4 |
| VM Manager | Security, Database | Critical | 4 |
| DWCP Manager | Security, Network | Critical | 4-5 |
| API Gateway | All Core | Critical | 6 |
| Agent Spawner | API, DWCP | Optional | 5 |
| ML Services | Database, API | Optional | 5 |

### 5.3 Parallel Initialization Groups

**Group 1 (Sequential - Foundation)**:
1. Config
2. Logger
3. Metrics

**Group 2 (Parallel - Data)**:
1. Database (PostgreSQL)
2. Database (Redis)

**Group 3 (Sequential - Security)**:
1. Security Manager

**Group 4 (Parallel - Infrastructure)**:
1. Network Layer
2. VM Manager
3. DWCP Manager (Phase 0)

**Group 5 (Parallel - Optional)**:
1. DWCP Phases 1-3
2. ML Services
3. Agent Spawner

---

## 6. Configuration Schema

### 6.1 Configuration File Structure

```json
{
  "environment": "production",
  "platform": {
    "name": "NovaCron",
    "version": "1.0.0",
    "nodeId": "node-001"
  },
  "system": {
    "dataDir": "/var/lib/novacron",
    "logLevel": "info",
    "maxConcurrency": 4,
    "shutdownTimeout": "30s"
  },
  "database": {
    "postgres": {
      "host": "localhost",
      "port": 5432,
      "database": "novacron",
      "user": "novacron",
      "password": "${POSTGRES_PASSWORD}",
      "poolSize": 20,
      "maxIdleConns": 5,
      "connMaxLifetime": "1h",
      "sslMode": "require"
    },
    "redis": {
      "host": "localhost",
      "port": 6379,
      "password": "${REDIS_PASSWORD}",
      "database": 0,
      "poolSize": 10
    }
  },
  "dwcp": {
    "enabled": true,
    "transport": {
      "minStreams": 16,
      "maxStreams": 256,
      "initialStreams": 32,
      "congestionAlgorithm": "bbr",
      "enableRDMA": false
    },
    "compression": {
      "enabled": true,
      "algorithm": "zstd",
      "level": "balanced",
      "enableDeltaEncoding": true,
      "enableDictionary": true
    },
    "prediction": {
      "enabled": true,
      "modelType": "lstm",
      "predictionHorizon": "5m"
    },
    "consensus": {
      "enabled": true,
      "algorithm": "raft",
      "quorumSize": 3
    }
  },
  "security": {
    "zeroTrust": {
      "enabled": true,
      "continuousAuth": true,
      "maxTrustDuration": "4h"
    },
    "encryption": {
      "algorithm": "AES-256-GCM",
      "keyRotationInterval": "24h"
    },
    "audit": {
      "enabled": true,
      "retentionDays": 90
    }
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8080,
    "cors": {
      "enabled": true,
      "allowedOrigins": ["https://app.novacron.io"]
    },
    "rateLimit": {
      "enabled": true,
      "requestsPerMinute": 1000
    }
  },
  "monitoring": {
    "metricsPort": 9090,
    "healthCheckPort": 8081,
    "prometheusEnabled": true
  }
}
```

### 6.2 Environment Variable Overrides

**Naming Convention**: `NOVACRON_<SECTION>_<KEY>`

**Examples**:
```bash
# Database overrides
export NOVACRON_DATABASE_POSTGRES_PASSWORD="secure_password"
export NOVACRON_DATABASE_POSTGRES_HOST="db.example.com"

# DWCP overrides
export NOVACRON_DWCP_ENABLED="true"
export NOVACRON_DWCP_TRANSPORT_MAXSTREAMS="512"

# Security overrides
export NOVACRON_SECURITY_ZEROTRUST_ENABLED="true"
```

### 6.3 Configuration Validation

```go
type ConfigValidator struct {
    required []string
    validators map[string]func(interface{}) error
}

func (v *ConfigValidator) Validate(config *Config) error {
    // 1. Check required fields
    for _, field := range v.required {
        if !hasField(config, field) {
            return fmt.Errorf("required field missing: %s", field)
        }
    }

    // 2. Validate field values
    for field, validator := range v.validators {
        value := getField(config, field)
        if err := validator(value); err != nil {
            return fmt.Errorf("validation failed for %s: %w", field, err)
        }
    }

    // 3. Cross-field validation
    if err := v.validateCrossFields(config); err != nil {
        return err
    }

    return nil
}
```

---

## 7. Security Architecture

### 7.1 Security Initialization Flow

```
┌─────────────────────────────────────────────────────────────┐
│              Security Component Initialization               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Credential Validation                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Validate database credentials                     │   │
│  │ • Check API keys/tokens                             │   │
│  │ • Verify SSL/TLS certificates                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  Step 2: Encryption Initialization                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Load/generate master encryption key               │   │
│  │ • Initialize AES-256-GCM cipher                     │   │
│  │ • Setup key rotation schedule                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  Step 3: Audit Logging                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Initialize audit log database                     │   │
│  │ • Configure log retention policy                    │   │
│  │ • Setup log signing for integrity                   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  Step 4: Zero-Trust Network                                 │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Initialize continuous authentication              │   │
│  │ • Setup micro-segmentation rules                    │   │
│  │ • Configure least-privilege policies                │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                   │
│  Step 5: Security Validation                                │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ • Verify all security components healthy            │   │
│  │ • Test encryption/decryption                        │   │
│  │ • Validate audit logging operational                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Secrets Management

**Strategy**: Never store secrets in configuration files

**Supported Sources**:
1. **Environment Variables** (development/testing)
2. **HashiCorp Vault** (production)
3. **AWS Secrets Manager** (AWS deployments)
4. **Azure Key Vault** (Azure deployments)

```go
type SecretsProvider interface {
    GetSecret(key string) (string, error)
    RotateSecret(key string) error
}

func loadSecrets(config *Config) error {
    provider := getSecretsProvider(config.SecretsBackend)

    // Load database password
    dbPassword, err := provider.GetSecret("database/postgres/password")
    if err != nil {
        return err
    }
    config.Database.Postgres.Password = dbPassword

    // Load API keys
    apiKey, err := provider.GetSecret("api/master-key")
    if err != nil {
        return err
    }
    config.API.MasterKey = apiKey

    return nil
}
```

---

## 8. Error Handling & Recovery

### 8.1 Error Classification

| Error Type | Severity | Action | Example |
|------------|----------|--------|---------|
| **Critical** | Fatal | Exit immediately | Database connection failed |
| **Recoverable** | High | Retry with backoff | Temporary network issue |
| **Degraded** | Medium | Continue with reduced functionality | Optional service unavailable |
| **Warning** | Low | Log and continue | Cache miss |

### 8.2 Recovery Strategies

#### Automatic Retry

```go
type RetryConfig struct {
    MaxAttempts   int
    InitialDelay  time.Duration
    MaxDelay      time.Duration
    Multiplier    float64
}

func WithRetry(ctx context.Context, config RetryConfig, fn func() error) error {
    delay := config.InitialDelay

    for attempt := 1; attempt <= config.MaxAttempts; attempt++ {
        err := fn()
        if err == nil {
            return nil
        }

        if !isRetriable(err) {
            return err
        }

        if attempt == config.MaxAttempts {
            return fmt.Errorf("failed after %d attempts: %w", attempt, err)
        }

        log.Warn("Retrying after error",
            "attempt", attempt,
            "delay", delay,
            "error", err)

        select {
        case <-ctx.Done():
            return ctx.Err()
        case <-time.After(delay):
            delay = time.Duration(float64(delay) * config.Multiplier)
            if delay > config.MaxDelay {
                delay = config.MaxDelay
            }
        }
    }

    return fmt.Errorf("unreachable")
}
```

#### Rollback on Failure

```go
func (o *Orchestrator) Initialize(ctx context.Context) error {
    initialized := make([]string, 0)

    for _, name := range o.order {
        if err := o.initializeComponent(ctx, name); err != nil {
            // Rollback in reverse order
            o.logger.Error("Component failed, rolling back",
                "component", name,
                "error", err)

            for i := len(initialized) - 1; i >= 0; i-- {
                if err := o.shutdownComponent(ctx, initialized[i]); err != nil {
                    o.logger.Error("Rollback failed",
                        "component", initialized[i],
                        "error", err)
                }
            }

            return fmt.Errorf("initialization failed: %w", err)
        }

        initialized = append(initialized, name)
    }

    return nil
}
```

#### Checkpoint & Recovery

```go
type RecoveryManager struct {
    checkpoints map[string]interface{}
    db          *sql.DB
}

func (r *RecoveryManager) SaveCheckpoint(name string, state interface{}) error {
    data, err := json.Marshal(state)
    if err != nil {
        return err
    }

    _, err = r.db.Exec(`
        INSERT INTO checkpoints (name, state, created_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (name) DO UPDATE SET
            state = $2,
            created_at = NOW()
    `, name, data)

    return err
}

func (r *RecoveryManager) RestoreFromCheckpoint(name string) (interface{}, error) {
    var data []byte
    err := r.db.QueryRow(`
        SELECT state FROM checkpoints
        WHERE name = $1
        ORDER BY created_at DESC
        LIMIT 1
    `, name).Scan(&data)

    if err != nil {
        return nil, err
    }

    var state interface{}
    if err := json.Unmarshal(data, &state); err != nil {
        return nil, err
    }

    return state, nil
}
```

---

## 9. Health Checks & Monitoring

### 9.1 Health Check Endpoints

#### Liveness Probe
**Endpoint**: `GET /health/live`
**Purpose**: Determine if application should be restarted

```go
func (h *HealthHandler) Liveness(w http.ResponseWriter, r *http.Request) {
    // Simple check: is the process running?
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]string{
        "status": "alive",
        "timestamp": time.Now().Format(time.RFC3339),
    })
}
```

#### Readiness Probe
**Endpoint**: `GET /health/ready`
**Purpose**: Determine if application can serve traffic

```go
func (h *HealthHandler) Readiness(w http.ResponseWriter, r *http.Request) {
    ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
    defer cancel()

    checks := []HealthCheck{
        {Name: "database", Check: h.checkDatabase},
        {Name: "cache", Check: h.checkCache},
        {Name: "dwcp", Check: h.checkDWCP},
    }

    results := make(map[string]interface{})
    allHealthy := true

    for _, check := range checks {
        err := check.Check(ctx)
        if err != nil {
            results[check.Name] = map[string]string{
                "status": "unhealthy",
                "error": err.Error(),
            }
            allHealthy = false
        } else {
            results[check.Name] = map[string]string{
                "status": "healthy",
            }
        }
    }

    status := http.StatusOK
    if !allHealthy {
        status = http.StatusServiceUnavailable
    }

    w.WriteHeader(status)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status": ternary(allHealthy, "ready", "not ready"),
        "checks": results,
        "timestamp": time.Now().Format(time.RFC3339),
    })
}
```

#### Detailed Health Check
**Endpoint**: `GET /health/status`
**Purpose**: Detailed component status for monitoring

```go
func (h *HealthHandler) DetailedStatus(w http.ResponseWriter, r *http.Request) {
    status := h.orchestrator.GetStatus()

    response := map[string]interface{}{
        "platform": map[string]string{
            "name": "NovaCron",
            "version": "1.0.0",
            "uptime": time.Since(h.startTime).String(),
        },
        "components": status,
        "metrics": h.metrics.Export(),
        "timestamp": time.Now().Format(time.RFC3339),
    }

    w.Header().Set("Content-Type", "application/json")
    json.NewEncoder(w).Encode(response)
}
```

### 9.2 Metrics Collection

```go
type Metrics struct {
    // Initialization metrics
    InitDuration       prometheus.Histogram
    ComponentInitTime  *prometheus.HistogramVec
    ComponentStatus    *prometheus.GaugeVec

    // Runtime metrics
    ComponentHealth    *prometheus.GaugeVec
    ErrorCount         *prometheus.CounterVec

    // DWCP metrics
    DWCPBandwidth      prometheus.Gauge
    DWCPLatency        prometheus.Histogram
    DWCPCompressionRatio prometheus.Gauge
}

func (m *Metrics) RecordComponentInit(name string, duration time.Duration, success bool) {
    m.ComponentInitTime.WithLabelValues(name).Observe(duration.Seconds())

    status := 1.0
    if !success {
        status = 0.0
    }
    m.ComponentStatus.WithLabelValues(name).Set(status)
}
```

---

## 10. Deployment Considerations

### 10.1 Container Deployment (Docker/Kubernetes)

#### Dockerfile Best Practices

```dockerfile
# Multi-stage build for Go backend
FROM golang:1.21-alpine AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o novacron-backend ./cmd/api-server

# Runtime image
FROM alpine:latest
RUN apk --no-cache add ca-certificates
WORKDIR /app

# Create non-root user
RUN addgroup -g 1000 novacron && \
    adduser -D -u 1000 -G novacron novacron

# Copy binary and config
COPY --from=builder /build/novacron-backend .
COPY --chown=novacron:novacron config/ ./config/

USER novacron
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8081/health/live || exit 1

CMD ["./novacron-backend"]
```

#### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: novacron-backend
  template:
    metadata:
      labels:
        app: novacron-backend
    spec:
      initContainers:
      - name: wait-for-db
        image: busybox:1.36
        command: ['sh', '-c', 'until nc -z postgres 5432; do sleep 2; done']

      containers:
      - name: backend
        image: novacron/backend:1.0.0
        ports:
        - containerPort: 8080
          name: api
        - containerPort: 9090
          name: metrics

        env:
        - name: NOVACRON_DATABASE_POSTGRES_HOST
          value: "postgres"
        - name: NOVACRON_DATABASE_POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: password

        livenessProbe:
          httpGet:
            path: /health/live
            port: 8081
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8081
          initialDelaySeconds: 40
          periodSeconds: 5

        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### 10.2 VM Deployment

#### Systemd Service

```ini
[Unit]
Description=NovaCron Backend Service
After=network.target postgresql.service redis.service
Requires=postgresql.service

[Service]
Type=notify
User=novacron
Group=novacron
WorkingDirectory=/opt/novacron
ExecStart=/opt/novacron/bin/novacron-backend
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
KillSignal=SIGTERM
TimeoutStopSec=30
Restart=on-failure
RestartSec=5s

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/novacron /var/log/novacron

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

---

## 11. Architecture Decision Records

### ADR-001: Phased Initialization Strategy

**Status**: Accepted
**Date**: 2025-11-14

**Context**: System has complex dependencies requiring ordered initialization

**Decision**: Implement sequential phased initialization with 6 distinct phases

**Rationale**:
- Ensures dependencies are met before component initialization
- Provides clear failure points for debugging
- Allows for health validation at each phase
- Enables parallel initialization within phases

**Consequences**:
- **Positive**: Predictable startup, clear error diagnosis
- **Negative**: Slightly longer startup time vs. full parallelization
- **Mitigation**: Use parallel initialization within dependency groups

### ADR-002: Dual-Runtime Architecture

**Status**: Accepted
**Date**: 2025-11-14

**Context**: Platform uses both Node.js and Go runtimes

**Decision**: Independent initialization per runtime with health-check coordination

**Rationale**:
- Each runtime has language-appropriate initialization patterns
- Allows independent scaling and deployment
- Reduces coupling between frontend and backend
- Enables graceful degradation

**Consequences**:
- **Positive**: Language-native patterns, independent deployment
- **Negative**: Requires coordination mechanism
- **Mitigation**: Shared database health status, HTTP health checks

### ADR-003: Configuration Hierarchy

**Status**: Accepted
**Date**: 2025-11-14

**Context**: Need flexible configuration across environments

**Decision**: Three-tier configuration: default → environment → runtime

**Rationale**:
- Provides safe defaults
- Allows environment-specific overrides (dev/staging/prod)
- Supports runtime secrets via environment variables
- Prevents accidental secret commits

**Consequences**:
- **Positive**: Flexible, secure, version-controlled defaults
- **Negative**: Multiple configuration sources to manage
- **Mitigation**: Clear documentation, validation at load time

### ADR-004: Component Lifecycle Interface

**Status**: Accepted
**Date**: 2025-11-14

**Context**: Need standardized component management

**Decision**: Define Component interface with Initialize/Shutdown/HealthCheck

**Rationale**:
- Provides consistent lifecycle management
- Enables automatic orchestration
- Simplifies testing and mocking
- Supports dependency declaration

**Consequences**:
- **Positive**: Standardized patterns, testable components
- **Negative**: All components must implement interface
- **Mitigation**: Provide base implementation for simple components

### ADR-005: Zero-Trust Security Initialization

**Status**: Accepted
**Date**: 2025-11-14

**Context**: Platform handles sensitive VM operations

**Decision**: Initialize security components before any other services

**Rationale**:
- Ensures audit logging captures all operations
- Validates credentials before use
- Enables continuous authentication from start
- Prevents security gaps during initialization

**Consequences**:
- **Positive**: Security-first approach, complete audit trail
- **Negative**: Slightly longer initialization time
- **Mitigation**: Optimize security component initialization

---

## Appendix A: Initialization Checklist

### Pre-Deployment Checklist

- [ ] Configuration files validated
- [ ] Database migrations prepared
- [ ] Secrets configured in secrets manager
- [ ] SSL/TLS certificates installed
- [ ] Monitoring configured (Prometheus/Grafana)
- [ ] Log aggregation setup (ELK/Loki)
- [ ] Health check endpoints tested
- [ ] Backup and recovery tested
- [ ] Security scan completed
- [ ] Load testing completed

### First-Time Initialization Checklist

- [ ] Create database and user
- [ ] Apply database migrations
- [ ] Generate encryption keys
- [ ] Configure secrets
- [ ] Test database connectivity
- [ ] Verify network connectivity
- [ ] Check KVM/hypervisor access
- [ ] Validate configuration
- [ ] Run health checks
- [ ] Verify metrics collection

---

## Appendix B: Troubleshooting Guide

### Common Issues

#### Database Connection Failures

**Symptom**: "failed to connect to database"

**Diagnosis**:
```bash
# Check database is running
systemctl status postgresql

# Test connection
psql -h localhost -U novacron -d novacron

# Check connection limits
SELECT count(*) FROM pg_stat_activity;
```

**Solution**:
- Verify credentials in configuration
- Check network connectivity
- Verify database exists and user has permissions
- Check connection pool limits

#### DWCP Initialization Failures

**Symptom**: "DWCP component failed to initialize"

**Diagnosis**:
```bash
# Check DWCP logs
journalctl -u novacron -f | grep DWCP

# Verify network capabilities
ip link show

# Check kernel modules
lsmod | grep kvm
```

**Solution**:
- Verify network interface configuration
- Check for required kernel modules
- Validate DWCP configuration
- Review security policies

#### Out of Memory During Initialization

**Symptom**: Process killed during initialization

**Diagnosis**:
```bash
# Check available memory
free -h

# Check OOM killer logs
dmesg | grep -i kill

# Check cgroup limits
cat /sys/fs/cgroup/memory/memory.limit_in_bytes
```

**Solution**:
- Increase memory limits
- Reduce concurrent initialization (maxConcurrency)
- Optimize component memory usage
- Enable swap if appropriate

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.0 | 2025-11-14 | System Architecture Designer | Complete architecture design |
| 1.0 | 2025-11-14 | System Architect | Initial comprehensive design |

**Next Review Date**: After Phase 1 Implementation

**Approval**: System Architecture Designer
**Date**: 2025-11-14
