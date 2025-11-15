# NovaCron Initialization Architecture - Comprehensive Design

**Architecture Decision Record (ADR)**

**Date:** 2025-11-14
**Status:** APPROVED
**Architect:** SystemArchitect Agent (Swarm Coordinator)
**Swarm:** swarm-fkhx8lyef

---

## Executive Summary

This document defines the **complete initialization architecture** for NovaCron, covering both frontend (Node.js) and backend (Go) initialization sequences. The architecture ensures proper startup ordering, dependency management, health validation, and graceful degradation across distributed components.

### Key Architectural Principles

1. **Phased Initialization**: Sequential phase execution with dependency validation
2. **Dual-Runtime Coordination**: Node.js and Go runtime orchestration
3. **Configuration Hierarchy**: Environment-specific with override capabilities
4. **Health-First Design**: Continuous validation at each initialization phase
5. **Graceful Degradation**: Optional services fail independently
6. **Event-Driven Lifecycle**: Observable initialization and shutdown sequences

---

## 1. System-Wide Initialization Overview (C4 Level 1)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    NovaCron Platform Bootstrap                       │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                  Configuration Layer                        │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐  │    │
│  │  │   Default    │  │  Environment │  │   Runtime Env   │  │    │
│  │  │ config.json  │→ │ Overrides    │→ │   Variables     │  │    │
│  │  └──────────────┘  └──────────────┘  └─────────────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                             ↓                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              Initialization Orchestrator                    │    │
│  │                                                             │    │
│  │  ┌─────────────────┐           ┌─────────────────┐        │    │
│  │  │   Node.js Init  │           │    Go Init      │        │    │
│  │  │  (Frontend/API) │           │   (Backend)     │        │    │
│  │  │                 │           │                 │        │    │
│  │  │  Phase 1: Env   │           │  Phase 1: Env   │        │    │
│  │  │  Phase 2: Config│           │  Phase 2: Config│        │    │
│  │  │  Phase 3: DB    │←─────────→│  Phase 3: DB    │        │    │
│  │  │  Phase 4: Core  │   Sync    │  Phase 4: Core  │        │    │
│  │  │  Phase 5: Opt   │           │  Phase 5: Opt   │        │    │
│  │  │  Phase 6: Health│           │  Phase 6: Health│        │    │
│  │  └─────────────────┘           └─────────────────┘        │    │
│  └────────────────────────────────────────────────────────────┘    │
│                             ↓                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                 Service Mesh Ready                          │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │    │
│  │  │   API    │  │   DWCP   │  │   ML     │  │  Consensus│  │    │
│  │  │ Gateway  │  │ Manager  │  │  Engine  │  │  Layer    │  │    │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘  │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Initialization Phase Architecture (C4 Level 2)

### 2.1 Node.js Frontend/API Initialization

```javascript
// Phase Sequence
Phase 0: Pre-Flight Checks (0-2s)
  ├─► Node.js version >= 18.0.0
  ├─► Required directories exist (src/, config/, logs/)
  ├─► File permissions validated
  └─► Memory availability check

Phase 1: Configuration Loading (2-4s)
  ├─► Load config.default.json
  ├─► Merge environment-specific config
  ├─► Apply NOVACRON_* environment variables
  ├─► Validate configuration schema
  └─► Emit 'init:config-loaded' event

Phase 2: Logging & Observability (4-5s)
  ├─► Initialize structured logger
  ├─► Setup log rotation
  ├─► Configure log levels per environment
  └─► Emit 'init:logging-setup' event

Phase 3: Database Connections (5-10s)
  ├─► PostgreSQL connection pool (max 10)
  ├─► Redis client connection
  ├─► Health check queries
  ├─► Connection timeout handling
  └─► Emit 'init:databases-connected' event

Phase 4: Core Services Initialization (10-15s)
  ├─► Cache Manager (required)
  ├─► Workload Monitor (required)
  ├─► MCP Integration (required)
  ├─► Error on core service failure
  └─► Emit 'init:core-services-ready' event

Phase 5: Optional Services (15-20s)
  ├─► Smart Agent Spawner (optional)
  ├─► Auto-Spawning Orchestrator (optional)
  ├─► Continue on optional service failure
  └─► Emit 'init:optional-services-ready' event

Phase 6: Error Handlers & Health (20-22s)
  ├─► Unhandled rejection handler
  ├─► Uncaught exception handler
  ├─► SIGTERM/SIGINT handlers
  ├─► Health check endpoint ready
  └─► Emit 'init:complete' event

Total: 22 seconds (development), 15 seconds (production optimized)
```

### 2.2 Go Backend Initialization

```go
// Phase Sequence
Phase 0: Environment Validation (0-1s)
  ├─► Go version >= 1.21
  ├─► System dependencies check
  ├─► Resource availability validation
  └─► Logger initialization

Phase 1: Configuration Loading (1-3s)
  ├─► Load YAML/JSON configuration
  ├─► Environment variable overrides
  ├─► Configuration validation
  ├─► Security configuration
  └─► Log configuration loaded

Phase 2: Database & Cache (3-8s)
  ├─► PostgreSQL connection (pq driver)
  ├─► Redis connection (go-redis)
  ├─► Connection pool configuration
  ├─► Migration execution (if needed)
  └─► Health checks

Phase 3: Core Components (8-15s)
  ├─► Authentication Manager
  ├─► Encryption Manager
  ├─► Audit Logger
  ├─► Security Coordinator
  ├─► DWCP Manager (if enabled)
  └─► Transport layer initialization

Phase 4: Service Layers (15-25s)
  ├─► VM Service
  ├─► Hypervisor Interface
  ├─► Orchestration Engine
  ├─► ML Prediction Service (optional)
  ├─► Consensus Layer (if enabled)
  └─► Federation Manager

Phase 5: API Gateway (25-30s)
  ├─► REST API routes (Gorilla Mux)
  ├─► gRPC server initialization
  ├─► WebSocket handlers
  ├─► GraphQL resolvers (optional)
  ├─► Middleware chain
  └─► CORS configuration

Phase 6: Health & Readiness (30-32s)
  ├─► Health check endpoints
  ├─► Liveness probe
  ├─► Readiness probe
  ├─► Metrics collection start
  └─► Register with load balancer

Total: 32 seconds (cold start), 20 seconds (warm start)
```

---

## 3. Component-Level Design

### 3.1 Configuration Management

```go
// Go Configuration Structure
type PlatformConfig struct {
    Environment  string         `yaml:"environment"`
    Server       ServerConfig   `yaml:"server"`
    Database     DatabaseConfig `yaml:"database"`
    Security     SecurityConfig `yaml:"security"`
    Services     ServicesConfig `yaml:"services"`
    Logging      LoggingConfig  `yaml:"logging"`
    Features     FeatureFlags   `yaml:"features"`
}

type ServerConfig struct {
    APIPort        int           `yaml:"api_port"`
    WSPort         int           `yaml:"ws_port"`
    GRPCPort       int           `yaml:"grpc_port"`
    ReadTimeout    time.Duration `yaml:"read_timeout"`
    WriteTimeout   time.Duration `yaml:"write_timeout"`
    MaxHeaderBytes int           `yaml:"max_header_bytes"`
}

type DatabaseConfig struct {
    Postgres PostgresConfig `yaml:"postgres"`
    Redis    RedisConfig    `yaml:"redis"`
}

type PostgresConfig struct {
    Host              string        `yaml:"host"`
    Port              int           `yaml:"port"`
    Database          string        `yaml:"database"`
    User              string        `yaml:"user"`
    Password          string        `yaml:"password"` // From env var
    PoolSize          int           `yaml:"pool_size"`
    IdleTimeout       time.Duration `yaml:"idle_timeout"`
    ConnectionTimeout time.Duration `yaml:"connection_timeout"`
}
```

```javascript
// Node.js Configuration Structure
{
  "environment": "production",
  "platform": {
    "name": "NovaCron",
    "version": "1.0.0"
  },
  "database": {
    "postgres": {
      "host": "localhost",
      "port": 5432,
      "database": "novacron",
      "poolSize": 10,
      "idleTimeout": 30000
    },
    "redis": {
      "host": "localhost",
      "port": 6379,
      "database": 0
    }
  },
  "services": {
    "cache": { "enabled": true, "ttl": 3600 },
    "workload-monitor": { "enabled": true, "interval": 5000 },
    "mcp-integration": { "enabled": true, "timeout": 10000 }
  }
}
```

### 3.2 Service Dependency Graph

```
┌──────────────────────────────────────────────────────────────┐
│                   Service Dependencies                        │
│                                                               │
│  Level 0: Foundation                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │  Config  │  │  Logger  │  │   Env    │                   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       └─────────────┼─────────────┘                          │
│                     │                                          │
│  Level 1: Storage                                             │
│  ┌──────────────────▼───────────────────┐                    │
│  │  ┌──────────┐  ┌──────────┐         │                    │
│  │  │PostgreSQL│  │  Redis   │         │                    │
│  │  └────┬─────┘  └────┬─────┘         │                    │
│  └───────┼─────────────┼────────────────┘                    │
│          │             │                                      │
│  Level 2: Core Services                                      │
│  ┌───────▼─────────────▼────────────────┐                    │
│  │  ┌──────────┐  ┌──────────┐         │                    │
│  │  │   Auth   │  │ Security │         │                    │
│  │  │ Manager  │  │Coordinator│         │                    │
│  │  └────┬─────┘  └────┬─────┘         │                    │
│  └───────┼─────────────┼────────────────┘                    │
│          │             │                                      │
│  Level 3: Business Logic                                     │
│  ┌───────▼─────────────▼────────────────┐                    │
│  │  ┌──────────┐  ┌──────────┐         │                    │
│  │  │VM Service│  │   DWCP   │         │                    │
│  │  │          │  │ Manager  │         │                    │
│  │  └────┬─────┘  └────┬─────┘         │                    │
│  └───────┼─────────────┼────────────────┘                    │
│          │             │                                      │
│  Level 4: API Gateway                                        │
│  ┌───────▼─────────────▼────────────────┐                    │
│  │  ┌──────────┐  ┌──────────┐         │                    │
│  │  │   REST   │  │  gRPC    │         │                    │
│  │  │   API    │  │  Server  │         │                    │
│  │  └──────────┘  └──────────┘         │                    │
│  └────────────────────────────────────────┘                    │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Initialization Interfaces

### 4.1 Go Initializer Interface

```go
// Initializer represents a component that can be initialized
type Initializer interface {
    // Initialize starts the component with given context
    Initialize(ctx context.Context) error

    // Name returns the component name
    Name() string

    // Dependencies returns required dependencies
    Dependencies() []string

    // HealthCheck validates component health
    HealthCheck() error
}

// Shutdowner represents a component that can be gracefully shut down
type Shutdowner interface {
    // Shutdown gracefully stops the component
    Shutdown(ctx context.Context) error
}

// InitializationManager orchestrates component initialization
type InitializationManager struct {
    components    map[string]Initializer
    initialized   map[string]bool
    shutdowners   []Shutdowner
    config        *PlatformConfig
    logger        *zap.Logger
    metrics       *InitMetrics
    mu            sync.RWMutex
}

// RegisterComponent registers a component for initialization
func (m *InitializationManager) RegisterComponent(comp Initializer) error {
    m.mu.Lock()
    defer m.mu.Unlock()

    if _, exists := m.components[comp.Name()]; exists {
        return fmt.Errorf("component %s already registered", comp.Name())
    }

    m.components[comp.Name()] = comp
    return nil
}

// InitializeAll initializes all components respecting dependencies
func (m *InitializationManager) InitializeAll(ctx context.Context) error {
    startTime := time.Now()

    // Build dependency graph
    graph, err := m.buildDependencyGraph()
    if err != nil {
        return fmt.Errorf("failed to build dependency graph: %w", err)
    }

    // Topological sort for initialization order
    order, err := graph.TopologicalSort()
    if err != nil {
        return fmt.Errorf("cyclic dependency detected: %w", err)
    }

    // Initialize components in order
    for _, name := range order {
        comp := m.components[name]

        m.logger.Info("Initializing component",
            zap.String("component", name))

        if err := comp.Initialize(ctx); err != nil {
            return fmt.Errorf("failed to initialize %s: %w", name, err)
        }

        m.initialized[name] = true

        // Track shutdowner
        if shutdowner, ok := comp.(Shutdowner); ok {
            m.shutdowners = append([]Shutdowner{shutdowner}, m.shutdowners...)
        }

        m.logger.Info("Component initialized successfully",
            zap.String("component", name))
    }

    duration := time.Since(startTime)
    m.metrics.InitializationDuration = duration
    m.metrics.ComponentsInitialized = len(order)

    m.logger.Info("All components initialized",
        zap.Duration("duration", duration),
        zap.Int("components", len(order)))

    return nil
}
```

### 4.2 Node.js Initializer Interface

```javascript
/**
 * Base class for initializable services
 */
class InitializableService extends EventEmitter {
    constructor(name, config) {
        super();
        this.name = name;
        this.config = config;
        this.state = 'pending';
        this.dependencies = [];
    }

    /**
     * Initialize the service
     * @returns {Promise<void>}
     */
    async initialize() {
        throw new Error('initialize() must be implemented');
    }

    /**
     * Get service name
     * @returns {string}
     */
    getName() {
        return this.name;
    }

    /**
     * Get required dependencies
     * @returns {string[]}
     */
    getDependencies() {
        return this.dependencies;
    }

    /**
     * Health check
     * @returns {Promise<boolean>}
     */
    async healthCheck() {
        return this.state === 'initialized';
    }

    /**
     * Graceful shutdown
     * @returns {Promise<void>}
     */
    async shutdown() {
        this.state = 'stopped';
        this.emit('shutdown');
    }
}

/**
 * Service initialization orchestrator
 */
class ServiceOrchestrator extends EventEmitter {
    constructor() {
        super();
        this.services = new Map();
        this.initialized = new Map();
        this.startTime = null;
    }

    /**
     * Register a service for initialization
     */
    register(service) {
        if (!(service instanceof InitializableService)) {
            throw new Error('Service must extend InitializableService');
        }

        this.services.set(service.getName(), service);
    }

    /**
     * Initialize all services in dependency order
     */
    async initializeAll() {
        this.startTime = Date.now();
        this.emit('orchestration:start');

        // Build dependency graph
        const graph = this.buildDependencyGraph();

        // Topological sort
        const order = this.topologicalSort(graph);

        // Initialize in order
        for (const serviceName of order) {
            const service = this.services.get(serviceName);

            this.emit('service:initializing', { service: serviceName });

            try {
                await service.initialize();
                this.initialized.set(serviceName, true);
                this.emit('service:initialized', { service: serviceName });
            } catch (error) {
                this.emit('service:failed', { service: serviceName, error });
                throw new ServiceInitializationError(
                    `Failed to initialize ${serviceName}`,
                    error
                );
            }
        }

        const duration = Date.now() - this.startTime;
        this.emit('orchestration:complete', { duration, count: order.length });
    }
}
```

---

## 5. Health Check Architecture

### 5.1 Health Check Levels

```
┌──────────────────────────────────────────────────────────┐
│                  Health Check Pyramid                     │
│                                                           │
│  Level 4: Business Logic                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │  - VM provisioning can complete                 │    │
│  │  - ML predictions are accurate                  │    │
│  │  - Consensus can reach quorum                   │    │
│  └─────────────────────────────────────────────────┘    │
│                       ▲                                   │
│  Level 3: Service Health                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │  - All core services responding                 │    │
│  │  - Optional services degraded gracefully        │    │
│  │  - API endpoints return 200 OK                  │    │
│  └─────────────────────────────────────────────────┘    │
│                       ▲                                   │
│  Level 2: Dependencies                                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │  - Database connections alive                   │    │
│  │  - Redis cache responding                       │    │
│  │  - External services reachable                  │    │
│  └─────────────────────────────────────────────────┘    │
│                       ▲                                   │
│  Level 1: Process Health                                 │
│  ┌─────────────────────────────────────────────────┐    │
│  │  - Process running                              │    │
│  │  - Memory within limits                         │    │
│  │  - CPU not saturated                            │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### 5.2 Health Endpoint Implementation

```go
// Go Health Check Handler
type HealthChecker struct {
    db               *sql.DB
    redis            *redis.Client
    dwcpManager      *dwcp.Manager
    services         map[string]Initializer
    startTime        time.Time
}

func (h *HealthChecker) HandleLiveness(w http.ResponseWriter, r *http.Request) {
    // Simple: Is the process alive?
    w.WriteHeader(http.StatusOK)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status": "alive",
        "uptime": time.Since(h.startTime).Seconds(),
    })
}

func (h *HealthChecker) HandleReadiness(w http.ResponseWriter, r *http.Request) {
    // Complex: Can we serve traffic?
    ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
    defer cancel()

    checks := []HealthCheck{
        {Name: "database", Check: h.checkDatabase},
        {Name: "redis", Check: h.checkRedis},
        {Name: "dwcp", Check: h.checkDWCP},
    }

    results := make(map[string]interface{})
    allHealthy := true

    for _, check := range checks {
        err := check.Check(ctx)
        results[check.Name] = map[string]interface{}{
            "healthy": err == nil,
            "error":   errToString(err),
        }
        if err != nil {
            allHealthy = false
        }
    }

    status := http.StatusOK
    if !allHealthy {
        status = http.StatusServiceUnavailable
    }

    w.WriteHeader(status)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "status": statusString(allHealthy),
        "checks": results,
        "uptime": time.Since(h.startTime).Seconds(),
    })
}

func (h *HealthChecker) checkDatabase(ctx context.Context) error {
    return h.db.PingContext(ctx)
}

func (h *HealthChecker) checkRedis(ctx context.Context) error {
    return h.redis.Ping(ctx).Err()
}

func (h *HealthChecker) checkDWCP(ctx context.Context) error {
    if h.dwcpManager == nil || !h.dwcpManager.IsEnabled() {
        return nil // Not enabled = healthy
    }
    return h.dwcpManager.HealthCheck()
}
```

---

## 6. Error Handling Strategy

### 6.1 Error Classification

```go
// Error types with recovery strategies
type InitErrorType int

const (
    ErrorTypeFatal InitErrorType = iota      // Cannot continue
    ErrorTypeRecoverable                     // Retry possible
    ErrorTypeDegradable                      // Continue with degraded service
)

type InitError struct {
    Type      InitErrorType
    Component string
    Message   string
    Cause     error
    Timestamp time.Time
    Retries   int
}

// Error handling policy
func (m *InitializationManager) handleInitError(err *InitError) error {
    switch err.Type {
    case ErrorTypeFatal:
        // Log and abort
        m.logger.Fatal("Fatal initialization error",
            zap.String("component", err.Component),
            zap.Error(err.Cause))
        return err

    case ErrorTypeRecoverable:
        // Retry with exponential backoff
        if err.Retries < 3 {
            backoff := time.Duration(math.Pow(2, float64(err.Retries))) * time.Second
            m.logger.Warn("Retrying component initialization",
                zap.String("component", err.Component),
                zap.Duration("backoff", backoff),
                zap.Int("attempt", err.Retries+1))
            time.Sleep(backoff)
            return nil // Signal retry
        }
        // Exceeded retries, escalate to fatal
        err.Type = ErrorTypeFatal
        return m.handleInitError(err)

    case ErrorTypeDegradable:
        // Log warning and continue
        m.logger.Warn("Optional component initialization failed",
            zap.String("component", err.Component),
            zap.Error(err.Cause))
        m.metrics.DegradedServices++
        return nil
    }

    return err
}
```

---

## 7. Graceful Shutdown Sequence

```go
// Shutdown orchestration
func (m *InitializationManager) Shutdown(ctx context.Context) error {
    m.logger.Info("Initiating graceful shutdown")

    shutdownCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()

    // Shutdown in reverse initialization order
    for i := len(m.shutdowners) - 1; i >= 0; i-- {
        shutdowner := m.shutdowners[i]

        // Determine component name
        name := "unknown"
        if named, ok := shutdowner.(interface{ Name() string }); ok {
            name = named.Name()
        }

        m.logger.Info("Shutting down component", zap.String("component", name))

        if err := shutdowner.Shutdown(shutdownCtx); err != nil {
            m.logger.Error("Component shutdown error",
                zap.String("component", name),
                zap.Error(err))
            // Continue shutting down other components
        }
    }

    m.logger.Info("Graceful shutdown complete")
    return nil
}

// Signal handling
func setupSignalHandlers(manager *InitializationManager) {
    sigChan := make(chan os.Signal, 1)
    signal.Notify(sigChan, syscall.SIGTERM, syscall.SIGINT)

    go func() {
        sig := <-sigChan
        logger.Info("Received shutdown signal", zap.String("signal", sig.String()))

        ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
        defer cancel()

        if err := manager.Shutdown(ctx); err != nil {
            logger.Error("Shutdown error", zap.Error(err))
            os.Exit(1)
        }

        os.Exit(0)
    }()
}
```

---

## 8. Initialization Metrics

```go
type InitMetrics struct {
    // Timing metrics
    InitializationDuration time.Duration
    PhaseTimings           map[string]time.Duration

    // Component metrics
    ComponentsInitialized  int
    ComponentsFailed       int
    DegradedServices       int

    // Dependency metrics
    DatabaseConnectTime    time.Duration
    RedisConnectTime       time.Duration

    // Health metrics
    LastHealthCheck        time.Time
    HealthChecksPassed     int
    HealthChecksFailed     int

    // Resource metrics
    MemoryUsedMB          float64
    GoroutinesActive      int
}

func (m *InitializationManager) ExportMetrics() *InitMetrics {
    m.mu.RLock()
    defer m.mu.RUnlock()

    // Calculate memory usage
    var mem runtime.MemStats
    runtime.ReadMemStats(&mem)

    m.metrics.MemoryUsedMB = float64(mem.Alloc) / 1024 / 1024
    m.metrics.GoroutinesActive = runtime.NumGoroutine()

    return m.metrics
}
```

---

## 9. Integration with Existing Systems

### 9.1 DWCP Manager Integration

```go
// DWCP-aware initialization
func (m *InitializationManager) initializeDWCP(ctx context.Context) error {
    dwcpConfig := m.config.Services.DWCP

    // Create DWCP manager
    dwcpManager, err := dwcp.NewManager(dwcpConfig, m.logger)
    if err != nil {
        return &InitError{
            Type:      ErrorTypeDegradable, // DWCP is optional
            Component: "dwcp",
            Message:   "Failed to create DWCP manager",
            Cause:     err,
        }
    }

    // Start DWCP manager
    if err := dwcpManager.Start(); err != nil {
        return &InitError{
            Type:      ErrorTypeDegradable,
            Component: "dwcp",
            Message:   "Failed to start DWCP manager",
            Cause:     err,
        }
    }

    m.dwcpManager = dwcpManager

    m.logger.Info("DWCP manager initialized",
        zap.Bool("enabled", dwcpConfig.Enabled),
        zap.String("version", dwcp.DWCPVersion))

    return nil
}
```

### 9.2 Frontend-Backend Coordination

```javascript
// Node.js coordination with Go backend
class BackendHealthChecker {
    constructor(backendURL) {
        this.backendURL = backendURL;
        this.maxRetries = 10;
        this.retryDelay = 2000;
    }

    async waitForBackendReady() {
        for (let i = 0; i < this.maxRetries; i++) {
            try {
                const response = await fetch(`${this.backendURL}/health/ready`);
                if (response.ok) {
                    const data = await response.json();
                    if (data.status === 'ready') {
                        return true;
                    }
                }
            } catch (error) {
                // Backend not ready, retry
            }

            await this.sleep(this.retryDelay);
        }

        throw new Error('Backend failed to become ready');
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Integration into PlatformInitializer
async initializeFrontend() {
    // ... existing initialization ...

    // Wait for backend before completing
    const backendChecker = new BackendHealthChecker(this.config.backend.url);
    await backendChecker.waitForBackendReady();

    this.log('info', 'Backend is ready, completing frontend initialization');
}
```

---

## 10. Architecture Decision Records

### ADR-001: Phased Initialization

**Context**: Need deterministic startup with clear failure points

**Decision**: Use sequential phased initialization with dependency validation

**Rationale**:
- Clear debugging when failures occur
- Dependency graph ensures correct ordering
- Health checks at each phase validate state

**Consequences**:
- (+) Predictable startup behavior
- (+) Easy to diagnose failures
- (-) Longer startup time vs parallel init
- (-) Blocked on slow dependencies

**Status**: APPROVED

---

### ADR-002: Dual Runtime Architecture

**Context**: Frontend (Node.js) and backend (Go) must coordinate startup

**Decision**: Independent initialization with health-check coordination

**Rationale**:
- Each runtime optimized for its use case
- Health endpoints provide coordination points
- Graceful degradation if one runtime fails

**Consequences**:
- (+) Language-appropriate initialization
- (+) Independent scaling
- (+) Clear separation of concerns
- (-) Additional coordination complexity
- (-) Must handle partial failures

**Status**: APPROVED

---

### ADR-003: Configuration Hierarchy

**Context**: Need environment-specific config with runtime overrides

**Decision**: Three-tier configuration: Default → Environment → Runtime Env Vars

**Rationale**:
- Default config provides safe fallbacks
- Environment-specific config for dev/staging/prod
- Runtime env vars for secrets and deployment-specific values

**Consequences**:
- (+) Clear configuration precedence
- (+) Secrets not in version control
- (+) Easy environment switching
- (-) Must maintain multiple config files
- (-) Potential for override confusion

**Status**: APPROVED

---

## 11. Testing Strategy

### 11.1 Initialization Testing

```go
func TestInitializationManager_InitializeAll(t *testing.T) {
    tests := []struct {
        name          string
        components    []Initializer
        config        *PlatformConfig
        expectError   bool
        expectPartial bool
    }{
        {
            name: "all components succeed",
            components: []Initializer{
                &MockComponent{name: "comp1", deps: []string{}},
                &MockComponent{name: "comp2", deps: []string{"comp1"}},
            },
            config:      testConfig(),
            expectError: false,
        },
        {
            name: "cyclic dependency detected",
            components: []Initializer{
                &MockComponent{name: "comp1", deps: []string{"comp2"}},
                &MockComponent{name: "comp2", deps: []string{"comp1"}},
            },
            config:      testConfig(),
            expectError: true,
        },
        {
            name: "optional component fails",
            components: []Initializer{
                &MockComponent{name: "core", deps: []string{}, required: true},
                &MockComponent{name: "optional", deps: []string{}, failInit: true},
            },
            config:        testConfig(),
            expectError:   false,
            expectPartial: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            manager := NewInitializationManager(tt.config, zaptest.NewLogger(t))

            for _, comp := range tt.components {
                manager.RegisterComponent(comp)
            }

            ctx := context.Background()
            err := manager.InitializeAll(ctx)

            if tt.expectError && err == nil {
                t.Error("expected error, got nil")
            }
            if !tt.expectError && err != nil {
                t.Errorf("unexpected error: %v", err)
            }
            if tt.expectPartial {
                metrics := manager.ExportMetrics()
                if metrics.DegradedServices == 0 {
                    t.Error("expected degraded services")
                }
            }
        })
    }
}
```

---

## 12. Deployment Considerations

### 12.1 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-backend
spec:
  replicas: 3
  template:
    spec:
      initContainers:
      - name: wait-for-postgres
        image: busybox:1.35
        command:
        - sh
        - -c
        - |
          until nc -z postgres 5432; do
            echo "Waiting for PostgreSQL..."
            sleep 2
          done

      containers:
      - name: backend
        image: novacron/backend:latest
        env:
        - name: NOVACRON_DATABASE_POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: postgres-password

        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 35
          periodSeconds: 5
          failureThreshold: 3

        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## 13. Monitoring & Observability

### 13.1 Initialization Metrics (Prometheus)

```yaml
# Initialization duration histogram
novacron_init_duration_seconds:
  type: histogram
  help: "Time taken for complete initialization"
  buckets: [10, 20, 30, 40, 50, 60]

# Component initialization status
novacron_component_init_status:
  type: gauge
  help: "Component initialization status (0=failed, 1=success, 2=degraded)"
  labels: [component]

# Health check results
novacron_health_check_status:
  type: gauge
  help: "Health check status (0=unhealthy, 1=healthy)"
  labels: [check_type]
```

---

## 14. Next Steps

### Immediate Actions

1. **Implement Go InitializationManager**
   - Create initialization orchestrator
   - Implement dependency graph resolution
   - Add component registration

2. **Enhance Node.js Initializer**
   - Add ServiceOrchestrator class
   - Implement backend health coordination
   - Add initialization metrics

3. **Create Integration Tests**
   - End-to-end initialization tests
   - Failure scenario testing
   - Performance benchmarks

4. **Document Runbooks**
   - Initialization failure recovery
   - Configuration troubleshooting
   - Health check interpretation

---

## 15. Conclusion

This comprehensive initialization architecture provides:

- **Robust Startup**: Phased initialization with validation
- **Clear Dependencies**: Dependency graph prevents initialization order issues
- **Health Validation**: Multi-level health checks ensure system readiness
- **Graceful Degradation**: Optional services fail independently
- **Observable Lifecycle**: Events and metrics throughout initialization
- **Production Ready**: Kubernetes integration, metrics, error handling

The architecture supports both Node.js and Go runtimes while maintaining clear separation of concerns and enabling independent scaling and deployment.

**Status**: READY FOR IMPLEMENTATION
**Next Review**: After Phase 1 implementation completion
