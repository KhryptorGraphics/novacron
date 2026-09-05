# NovaCron Data Architecture Strategy

## Executive Summary

This document outlines a comprehensive data architecture strategy for NovaCron's evolution from a monolithic to microservices architecture, focusing on data consistency, performance optimization, and scalability across distributed VM management operations.

## Current Data Architecture Analysis

### Existing Database Schema
Based on analysis of the current system, we have the following tables:

```sql
-- Core user management
users (id, username, email, password_hash, role, tenant_id, created_at, updated_at)

-- VM lifecycle management  
vms (id, name, state, node_id, owner_id, tenant_id, config, created_at, updated_at)
vm_metrics (id, vm_id, cpu_usage, memory_usage, network_sent, network_recv, timestamp)

-- Multi-cloud federation
cloud_providers (id, name, type, config, enabled, created_at, updated_at)
multicloud_vms (id, provider_id, vm_id, name, region, instance_type, state, tags, created_at, updated_at)
migrations (id, vm_id, source_provider, destination_provider, status, progress, started_at, completed_at, error_message)

-- Indexing strategy
idx_vm_metrics_vm_id, idx_vm_metrics_timestamp
idx_multicloud_vms_provider, idx_migrations_status
```

### Current Limitations
1. **Single Database Bottleneck**: All services share one PostgreSQL instance
2. **Limited Scalability**: No read/write splitting or sharding strategy
3. **Mixed Concerns**: User data and VM metrics in the same database
4. **No Caching Strategy**: Direct database access without caching layers
5. **Lack of Data Versioning**: No audit trails or change tracking

## Target Data Architecture

### 1. **Database-per-Service Pattern**

#### Authentication Service Database
```sql
-- Database: novacron_auth
CREATE DATABASE novacron_auth;

-- Core user management with enhanced security
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role user_role_enum NOT NULL DEFAULT 'user',
    tenant_id UUID NOT NULL,
    mfa_enabled BOOLEAN DEFAULT false,
    mfa_secret VARCHAR(255),
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP,
    last_login TIMESTAMP,
    password_changed_at TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- API key management for service-to-service authentication
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    permissions JSONB NOT NULL DEFAULT '{}',
    scopes TEXT[] NOT NULL DEFAULT '{}',
    rate_limit_per_minute INTEGER DEFAULT 1000,
    expires_at TIMESTAMP,
    last_used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Session management for web clients
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Role-based access control
CREATE TABLE roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE user_roles (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID REFERENCES roles(id) ON DELETE CASCADE,
    granted_by UUID REFERENCES users(id),
    granted_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (user_id, role_id)
);

-- Audit logging for authentication events
CREATE TABLE auth_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TYPE user_role_enum AS ENUM ('admin', 'operator', 'user', 'viewer');

-- Indexes for performance
CREATE INDEX idx_users_tenant_id ON users(tenant_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX idx_audit_log_user_id ON auth_audit_log(user_id);
CREATE INDEX idx_audit_log_created_at ON auth_audit_log(created_at);
```

#### VM Service Database
```sql
-- Database: novacron_vms
CREATE DATABASE novacron_vms;

-- Enhanced VM management with full lifecycle support
CREATE TABLE vms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    display_name VARCHAR(255),
    description TEXT,
    state vm_state_enum NOT NULL DEFAULT 'creating',
    hypervisor_type hypervisor_enum NOT NULL,
    node_id UUID,
    owner_id UUID NOT NULL, -- Reference to auth service
    tenant_id UUID NOT NULL,
    template_id UUID REFERENCES vm_templates(id),
    resource_config JSONB NOT NULL,
    network_config JSONB,
    storage_config JSONB,
    security_config JSONB,
    backup_config JSONB,
    tags JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    deleted_at TIMESTAMP -- Soft delete
);

-- VM templates for standardized deployments
CREATE TABLE vm_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(50) NOT NULL,
    owner_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    base_image VARCHAR(255) NOT NULL,
    resource_config JSONB NOT NULL,
    network_template JSONB,
    storage_template JSONB,
    security_template JSONB,
    tags JSONB DEFAULT '{}',
    is_public BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- VM snapshots for backup and cloning
CREATE TABLE vm_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID REFERENCES vms(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    snapshot_type snapshot_type_enum NOT NULL,
    storage_path TEXT NOT NULL,
    size_bytes BIGINT,
    compression_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- Network management
CREATE TABLE vm_networks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID REFERENCES vms(id) ON DELETE CASCADE,
    network_name VARCHAR(255) NOT NULL,
    interface_name VARCHAR(100) NOT NULL,
    ip_address INET,
    mac_address MACADDR,
    network_type VARCHAR(50) NOT NULL,
    vlan_id INTEGER,
    bandwidth_limit_mbps INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- VM state transitions for audit
CREATE TABLE vm_state_transitions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID REFERENCES vms(id) ON DELETE CASCADE,
    from_state vm_state_enum,
    to_state vm_state_enum NOT NULL,
    reason VARCHAR(255),
    initiated_by UUID NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TYPE vm_state_enum AS ENUM (
    'creating', 'running', 'stopped', 'paused', 'suspended',
    'migrating', 'backing_up', 'restoring', 'error', 'deleted'
);

CREATE TYPE hypervisor_enum AS ENUM ('kvm', 'xen', 'vmware', 'hyper-v', 'container');
CREATE TYPE snapshot_type_enum AS ENUM ('manual', 'scheduled', 'pre-migration', 'pre-update');

-- Performance indexes
CREATE INDEX idx_vms_owner_id ON vms(owner_id);
CREATE INDEX idx_vms_tenant_id ON vms(tenant_id);
CREATE INDEX idx_vms_state ON vms(state);
CREATE INDEX idx_vms_node_id ON vms(node_id);
CREATE INDEX idx_vms_created_at ON vms(created_at);
CREATE INDEX idx_vm_snapshots_vm_id ON vm_snapshots(vm_id);
CREATE INDEX idx_vm_networks_vm_id ON vm_networks(vm_id);
CREATE INDEX idx_vm_state_transitions_vm_id ON vm_state_transitions(vm_id);
```

#### Monitoring Service Database (TimescaleDB)
```sql
-- Database: novacron_monitoring (TimescaleDB extension)
CREATE DATABASE novacron_monitoring;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Time-series VM metrics with hypertables for performance
CREATE TABLE vm_metrics (
    time TIMESTAMPTZ NOT NULL,
    vm_id UUID NOT NULL,
    node_id UUID,
    tenant_id UUID NOT NULL,
    cpu_usage_percent DOUBLE PRECISION,
    memory_usage_bytes BIGINT,
    memory_available_bytes BIGINT,
    network_bytes_sent BIGINT,
    network_bytes_recv BIGINT,
    network_packets_sent BIGINT,
    network_packets_recv BIGINT,
    disk_read_bytes BIGINT,
    disk_write_bytes BIGINT,
    disk_read_ops BIGINT,
    disk_write_ops BIGINT,
    disk_usage_bytes BIGINT,
    disk_available_bytes BIGINT,
    load_average DOUBLE PRECISION,
    process_count INTEGER,
    temperature_celsius DOUBLE PRECISION
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('vm_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- System-level metrics
CREATE TABLE system_metrics (
    time TIMESTAMPTZ NOT NULL,
    node_id UUID NOT NULL,
    cpu_usage_percent DOUBLE PRECISION,
    memory_usage_bytes BIGINT,
    memory_total_bytes BIGINT,
    disk_usage_bytes BIGINT,
    disk_total_bytes BIGINT,
    network_bytes_sent BIGINT,
    network_bytes_recv BIGINT,
    load_average_1m DOUBLE PRECISION,
    load_average_5m DOUBLE PRECISION,
    load_average_15m DOUBLE PRECISION,
    vm_count INTEGER,
    running_vm_count INTEGER
);

SELECT create_hypertable('system_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Alert definitions and states
CREATE TABLE alert_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tenant_id UUID NOT NULL,
    metric_name VARCHAR(255) NOT NULL,
    threshold_value DOUBLE PRECISION NOT NULL,
    comparison_operator VARCHAR(10) NOT NULL, -- >, <, >=, <=, ==, !=
    evaluation_window INTERVAL NOT NULL,
    severity alert_severity_enum NOT NULL,
    enabled BOOLEAN DEFAULT true,
    notification_channels JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE alert_instances (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id UUID REFERENCES alert_rules(id) ON DELETE CASCADE,
    vm_id UUID,
    node_id UUID,
    state alert_state_enum NOT NULL,
    value DOUBLE PRECISION,
    message TEXT,
    labels JSONB,
    started_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP,
    acknowledged_at TIMESTAMP,
    acknowledged_by UUID
);

CREATE TYPE alert_severity_enum AS ENUM ('info', 'warning', 'critical', 'emergency');
CREATE TYPE alert_state_enum AS ENUM ('pending', 'firing', 'resolved', 'acknowledged');

-- Retention policies for efficient storage
SELECT add_retention_policy('vm_metrics', INTERVAL '90 days');
SELECT add_retention_policy('system_metrics', INTERVAL '180 days');

-- Indexes optimized for time-series queries
CREATE INDEX idx_vm_metrics_vm_id_time ON vm_metrics(vm_id, time DESC);
CREATE INDEX idx_vm_metrics_tenant_id_time ON vm_metrics(tenant_id, time DESC);
CREATE INDEX idx_system_metrics_node_id_time ON system_metrics(node_id, time DESC);
CREATE INDEX idx_alert_instances_rule_id ON alert_instances(rule_id);
CREATE INDEX idx_alert_instances_state ON alert_instances(state);
```

#### Federation Service Database
```sql
-- Database: novacron_federation
CREATE DATABASE novacron_federation;

-- Enhanced cloud provider management
CREATE TABLE cloud_providers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    provider_type provider_type_enum NOT NULL,
    regions JSONB NOT NULL DEFAULT '[]',
    credentials_encrypted TEXT NOT NULL,
    config JSONB NOT NULL DEFAULT '{}',
    capabilities JSONB NOT NULL DEFAULT '{}',
    pricing_model JSONB,
    health_status provider_health_enum DEFAULT 'unknown',
    health_check_url VARCHAR(500),
    last_health_check TIMESTAMP,
    health_check_interval INTERVAL DEFAULT '5 minutes',
    enabled BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Multi-cloud VM tracking
CREATE TABLE multicloud_vms (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    local_vm_id UUID NOT NULL, -- Reference to VM service
    provider_id UUID REFERENCES cloud_providers(id),
    provider_vm_id VARCHAR(255) NOT NULL,
    name VARCHAR(255),
    region VARCHAR(100),
    availability_zone VARCHAR(100),
    instance_type VARCHAR(100),
    instance_family VARCHAR(100),
    state provider_vm_state_enum NOT NULL,
    public_ip INET,
    private_ip INET,
    cost_per_hour DECIMAL(10,4),
    tags JSONB DEFAULT '{}',
    provider_metadata JSONB DEFAULT '{}',
    last_sync TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Cross-cloud migration tracking with detailed workflow
CREATE TABLE migrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID NOT NULL,
    source_provider UUID REFERENCES cloud_providers(id),
    destination_provider UUID REFERENCES cloud_providers(id),
    migration_type migration_type_enum NOT NULL,
    migration_plan JSONB NOT NULL,
    status migration_status_enum NOT NULL DEFAULT 'planned',
    progress_percentage INTEGER DEFAULT 0,
    current_phase VARCHAR(100),
    estimated_duration INTERVAL,
    estimated_completion_time TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    rollback_plan JSONB,
    cost_estimate DECIMAL(10,2),
    actual_cost DECIMAL(10,2),
    error_details JSONB,
    initiated_by UUID NOT NULL,
    approved_by UUID,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Migration steps for detailed tracking
CREATE TABLE migration_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    migration_id UUID REFERENCES migrations(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    step_name VARCHAR(255) NOT NULL,
    description TEXT,
    status step_status_enum NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    duration INTERVAL,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    output JSONB
);

-- Compliance policy management
CREATE TABLE compliance_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    framework VARCHAR(100) NOT NULL, -- GDPR, HIPAA, SOC2, PCI-DSS
    description TEXT,
    rules JSONB NOT NULL,
    data_residency_requirements JSONB,
    encryption_requirements JSONB,
    audit_requirements JSONB,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Compliance violations tracking
CREATE TABLE compliance_violations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_id UUID REFERENCES compliance_policies(id),
    vm_id UUID,
    provider_id UUID REFERENCES cloud_providers(id),
    violation_type VARCHAR(100) NOT NULL,
    severity violation_severity_enum NOT NULL,
    description TEXT NOT NULL,
    detected_at TIMESTAMP DEFAULT NOW(),
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    auto_remediation_applied BOOLEAN DEFAULT false
);

-- Cost optimization tracking
CREATE TABLE cost_optimizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID NOT NULL,
    current_provider UUID REFERENCES cloud_providers(id),
    recommended_provider UUID REFERENCES cloud_providers(id),
    current_cost_per_hour DECIMAL(10,4),
    projected_cost_per_hour DECIMAL(10,4),
    potential_monthly_savings DECIMAL(10,2),
    optimization_type VARCHAR(100) NOT NULL,
    confidence_score DOUBLE PRECISION,
    recommendation JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    applied_at TIMESTAMP
);

CREATE TYPE provider_type_enum AS ENUM ('aws', 'azure', 'gcp', 'on_premise', 'hybrid');
CREATE TYPE provider_health_enum AS ENUM ('healthy', 'degraded', 'unhealthy', 'unknown');
CREATE TYPE provider_vm_state_enum AS ENUM ('pending', 'running', 'stopping', 'stopped', 'terminated');
CREATE TYPE migration_type_enum AS ENUM ('cold', 'warm', 'live', 'clone');
CREATE TYPE migration_status_enum AS ENUM (
    'planned', 'approved', 'in_progress', 'completed', 
    'failed', 'rolled_back', 'cancelled'
);
CREATE TYPE step_status_enum AS ENUM ('pending', 'in_progress', 'completed', 'failed', 'skipped');
CREATE TYPE violation_severity_enum AS ENUM ('low', 'medium', 'high', 'critical');

-- Performance indexes
CREATE INDEX idx_cloud_providers_type ON cloud_providers(provider_type);
CREATE INDEX idx_cloud_providers_enabled ON cloud_providers(enabled);
CREATE INDEX idx_multicloud_vms_provider_id ON multicloud_vms(provider_id);
CREATE INDEX idx_multicloud_vms_local_vm_id ON multicloud_vms(local_vm_id);
CREATE INDEX idx_migrations_status ON migrations(status);
CREATE INDEX idx_migrations_vm_id ON migrations(vm_id);
CREATE INDEX idx_migration_steps_migration_id ON migration_steps(migration_id);
CREATE INDEX idx_compliance_violations_policy_id ON compliance_violations(policy_id);
CREATE INDEX idx_cost_optimizations_vm_id ON cost_optimizations(vm_id);
```

#### Scheduler Service Database
```sql
-- Database: novacron_scheduler
CREATE DATABASE novacron_scheduler;

-- Node and cluster management
CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    hostname VARCHAR(255) NOT NULL,
    ip_address INET NOT NULL,
    node_type node_type_enum NOT NULL,
    cluster_id UUID REFERENCES clusters(id),
    region VARCHAR(100),
    availability_zone VARCHAR(100),
    capacity JSONB NOT NULL, -- CPU, memory, storage, network
    allocatable JSONB NOT NULL,
    allocated JSONB NOT NULL,
    labels JSONB DEFAULT '{}',
    taints JSONB DEFAULT '[]',
    status node_status_enum DEFAULT 'unknown',
    last_heartbeat TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    cluster_type VARCHAR(50) NOT NULL,
    version VARCHAR(50),
    config JSONB,
    status cluster_status_enum DEFAULT 'unknown',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Advanced scheduling policies
CREATE TABLE scheduling_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    tenant_id UUID NOT NULL,
    priority INTEGER DEFAULT 0,
    policy_type policy_type_enum NOT NULL,
    rules JSONB NOT NULL,
    constraints JSONB DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    anti_affinity_rules JSONB DEFAULT '[]',
    resource_requirements JSONB,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Resource allocation tracking
CREATE TABLE resource_allocations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID NOT NULL,
    node_id UUID REFERENCES nodes(id),
    allocation_type allocation_type_enum NOT NULL,
    cpu_cores INTEGER NOT NULL,
    cpu_mhz INTEGER,
    memory_mb BIGINT NOT NULL,
    storage_gb BIGINT NOT NULL,
    network_bandwidth_mbps INTEGER,
    gpu_count INTEGER DEFAULT 0,
    gpu_type VARCHAR(100),
    allocated_at TIMESTAMP DEFAULT NOW(),
    deallocated_at TIMESTAMP,
    status allocation_status_enum DEFAULT 'active'
);

-- Scheduling decisions and history
CREATE TABLE scheduling_decisions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID NOT NULL,
    requested_resources JSONB NOT NULL,
    constraints JSONB DEFAULT '{}',
    candidate_nodes JSONB NOT NULL,
    selected_node_id UUID REFERENCES nodes(id),
    decision_algorithm VARCHAR(100) NOT NULL,
    decision_score DOUBLE PRECISION,
    decision_factors JSONB,
    decision_time TIMESTAMP DEFAULT NOW(),
    scheduler_version VARCHAR(50)
);

-- Predictive scaling recommendations
CREATE TABLE scaling_recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vm_id UUID NOT NULL,
    current_resources JSONB NOT NULL,
    recommended_resources JSONB NOT NULL,
    prediction_confidence DOUBLE PRECISION,
    prediction_horizon INTERVAL NOT NULL,
    cost_impact DECIMAL(10,2),
    performance_impact JSONB,
    recommendation_reason TEXT,
    ml_model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW(),
    applied_at TIMESTAMP,
    feedback_score INTEGER -- User feedback on recommendation quality
);

CREATE TYPE node_type_enum AS ENUM ('compute', 'storage', 'network', 'gpu', 'hybrid');
CREATE TYPE node_status_enum AS ENUM ('ready', 'not_ready', 'unknown', 'maintenance');
CREATE TYPE cluster_status_enum AS ENUM ('active', 'inactive', 'maintenance', 'error');
CREATE TYPE policy_type_enum AS ENUM ('placement', 'scaling', 'migration', 'cost_optimization');
CREATE TYPE allocation_type_enum AS ENUM ('guaranteed', 'burstable', 'best_effort');
CREATE TYPE allocation_status_enum AS ENUM ('active', 'deallocated', 'migrating');

-- Performance indexes
CREATE INDEX idx_nodes_cluster_id ON nodes(cluster_id);
CREATE INDEX idx_nodes_status ON nodes(status);
CREATE INDEX idx_scheduling_policies_tenant_id ON scheduling_policies(tenant_id);
CREATE INDEX idx_resource_allocations_vm_id ON resource_allocations(vm_id);
CREATE INDEX idx_resource_allocations_node_id ON resource_allocations(node_id);
CREATE INDEX idx_scheduling_decisions_vm_id ON scheduling_decisions(vm_id);
CREATE INDEX idx_scaling_recommendations_vm_id ON scaling_recommendations(vm_id);
```

### 2. **Cross-Service Data Access Patterns**

#### Service-to-Service Communication
```go
// Example: VM Service needs user validation from Auth Service
type AuthServiceClient interface {
    ValidateToken(ctx context.Context, token string) (*UserInfo, error)
    GetUserPermissions(ctx context.Context, userID string) ([]string, error)
    CheckPermission(ctx context.Context, userID, resource, action string) (bool, error)
}

// VM Service implementation
func (vs *VMService) CreateVM(ctx context.Context, req *CreateVMRequest) (*CreateVMResponse, error) {
    // Cross-service call to validate authentication
    user, err := vs.authClient.ValidateToken(ctx, req.AuthToken)
    if err != nil {
        return nil, status.Errorf(codes.Unauthenticated, "invalid token: %v", err)
    }
    
    // Check authorization
    hasPermission, err := vs.authClient.CheckPermission(ctx, user.ID, "vm", "create")
    if err != nil || !hasPermission {
        return nil, status.Errorf(codes.PermissionDenied, "insufficient permissions")
    }
    
    // Create VM with authenticated context
    vm := &VM{
        Name: req.Name,
        OwnerID: user.ID,
        TenantID: user.TenantID,
        Config: req.Config,
    }
    
    return vs.createVMInternal(ctx, vm)
}
```

#### Data Consistency Patterns
```go
// Saga pattern for cross-service transactions
type VMCreationSaga struct {
    AuthService       AuthServiceClient
    VMService         VMServiceClient  
    SchedulerService  SchedulerServiceClient
    MonitoringService MonitoringServiceClient
}

func (saga *VMCreationSaga) CreateVM(ctx context.Context, req *CreateVMRequest) error {
    // Step 1: Validate authentication
    user, err := saga.AuthService.ValidateToken(ctx, req.AuthToken)
    if err != nil {
        return err
    }
    
    // Step 2: Reserve resources
    reservation, err := saga.SchedulerService.ReserveResources(ctx, &ReserveResourcesRequest{
        Resources: req.Resources,
        UserID: user.ID,
    })
    if err != nil {
        return err
    }
    
    // Step 3: Create VM
    vm, err := saga.VMService.CreateVM(ctx, &VMServiceCreateRequest{
        Name: req.Name,
        OwnerID: user.ID,
        Resources: req.Resources,
        ReservationID: reservation.ID,
    })
    if err != nil {
        // Compensating action: Release reservation
        saga.SchedulerService.ReleaseReservation(ctx, reservation.ID)
        return err
    }
    
    // Step 4: Setup monitoring
    _, err = saga.MonitoringService.CreateVMMonitoring(ctx, &CreateVMMonitoringRequest{
        VMID: vm.ID,
        MetricConfigs: req.MonitoringConfig,
    })
    if err != nil {
        // Compensating actions: Delete VM and release reservation
        saga.VMService.DeleteVM(ctx, vm.ID)
        saga.SchedulerService.ReleaseReservation(ctx, reservation.ID)
        return err
    }
    
    return nil
}
```

### 3. **Caching Strategy**

#### Multi-Layer Caching Architecture
```go
// L1: Application-level caching
type VMCache struct {
    vmStatusCache    *sync.Map           // In-memory cache for VM status
    templateCache    *lru.Cache          // LRU cache for VM templates
    permissionCache  *expirable.LRU      // Expirable cache for user permissions
}

// L2: Redis cluster for cross-service caching
type RedisClusterCache struct {
    vmMetricsCache   redis.Client        // VM metrics with 5-minute TTL
    userSessionCache redis.Client        // User sessions with 24-hour TTL  
    providerCache    redis.Client        // Cloud provider status with 1-minute TTL
}

// L3: Database query result caching
type QueryCache struct {
    preparedStatements map[string]*sql.Stmt
    queryResultCache  *groupcache.Group
}

// Example caching implementation
func (vs *VMService) GetVMStatus(ctx context.Context, vmID string) (*VMStatus, error) {
    // L1: Check application cache
    if status, ok := vs.cache.vmStatusCache.Load(vmID); ok {
        return status.(*VMStatus), nil
    }
    
    // L2: Check Redis cache
    if status, err := vs.redisCache.Get(ctx, "vm:status:"+vmID).Result(); err == nil {
        vmStatus := &VMStatus{}
        json.Unmarshal([]byte(status), vmStatus)
        vs.cache.vmStatusCache.Store(vmID, vmStatus)
        return vmStatus, nil
    }
    
    // L3: Query database
    vmStatus, err := vs.db.GetVMStatus(ctx, vmID)
    if err != nil {
        return nil, err
    }
    
    // Cache the result
    statusJSON, _ := json.Marshal(vmStatus)
    vs.redisCache.SetEX(ctx, "vm:status:"+vmID, statusJSON, 30*time.Second)
    vs.cache.vmStatusCache.Store(vmID, vmStatus)
    
    return vmStatus, nil
}
```

### 4. **Data Migration Strategy**

#### Phase 1: Database Separation (Week 1-2)
```sql
-- Migration script template
-- 1. Create new service-specific databases
CREATE DATABASE novacron_auth;
CREATE DATABASE novacron_vms;
CREATE DATABASE novacron_federation;
CREATE DATABASE novacron_monitoring;
CREATE DATABASE novacron_scheduler;

-- 2. Migrate data with referential integrity preservation
-- Auth service migration
INSERT INTO novacron_auth.users 
SELECT id, username, email, password_hash, role, tenant_id, created_at, updated_at 
FROM novacron.users;

-- VM service migration  
INSERT INTO novacron_vms.vms
SELECT id, name, state, node_id, owner_id, tenant_id, config, created_at, updated_at
FROM novacron.vms;

-- Update foreign key references to use service-to-service lookups
-- (This requires application-level changes, not database FKs)
```

#### Phase 2: Gradual Service Migration
```go
// Dual-write pattern during migration
func (vs *VMService) CreateVM(ctx context.Context, req *CreateVMRequest) (*CreateVMResponse, error) {
    // Write to new service database
    vm, err := vs.createVMInNewDB(ctx, req)
    if err != nil {
        return nil, err
    }
    
    // Also write to old database for backwards compatibility
    if vs.config.DualWriteEnabled {
        go func() {
            ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
            defer cancel()
            vs.createVMInOldDB(ctx, vm)
        }()
    }
    
    return vm, nil
}
```

### 5. **Performance Optimization**

#### Connection Pool Configuration
```go
// Database connection pools per service
type DatabaseConfig struct {
    MaxOpenConns    int           `yaml:"max_open_conns"`     // 100
    MaxIdleConns    int           `yaml:"max_idle_conns"`     // 25
    ConnMaxLifetime time.Duration `yaml:"conn_max_lifetime"`  // 1 hour
    ConnMaxIdleTime time.Duration `yaml:"conn_max_idle_time"` // 15 minutes
}

// Redis cluster configuration
type RedisConfig struct {
    ClusterNodes    []string      `yaml:"cluster_nodes"`
    PoolSize        int           `yaml:"pool_size"`          // 100
    MinIdleConns    int           `yaml:"min_idle_conns"`     // 20
    DialTimeout     time.Duration `yaml:"dial_timeout"`       // 5 seconds
    ReadTimeout     time.Duration `yaml:"read_timeout"`       // 3 seconds
    WriteTimeout    time.Duration `yaml:"write_timeout"`      // 3 seconds
}
```

#### Query Optimization
```sql
-- Partition large tables by time for better performance
CREATE TABLE vm_metrics_y2025m01 PARTITION OF vm_metrics
FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

-- Create materialized views for common aggregations
CREATE MATERIALIZED VIEW vm_metrics_hourly AS
SELECT 
    vm_id,
    date_trunc('hour', time) as hour,
    AVG(cpu_usage_percent) as avg_cpu,
    MAX(memory_usage_bytes) as max_memory,
    AVG(network_bytes_sent) as avg_network_sent
FROM vm_metrics
GROUP BY vm_id, date_trunc('hour', time);

-- Refresh materialized view periodically
CREATE OR REPLACE FUNCTION refresh_vm_metrics_hourly()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY vm_metrics_hourly;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh every hour
SELECT cron.schedule('refresh-vm-metrics', '0 * * * *', 'SELECT refresh_vm_metrics_hourly();');
```

### 6. **Backup and Disaster Recovery**

#### Backup Strategy
```yaml
# Backup configuration per service
services:
  auth-service:
    backup_schedule: "0 2 * * *"  # Daily at 2 AM
    retention_days: 90
    encryption: true
    
  vm-service:
    backup_schedule: "0 1 * * *"  # Daily at 1 AM  
    retention_days: 30
    incremental: true
    
  monitoring-service:
    backup_schedule: "0 3 * * 0"  # Weekly on Sunday at 3 AM
    retention_weeks: 12
    compression: true
    
  federation-service:
    backup_schedule: "0 1 * * *"  # Daily at 1 AM
    retention_days: 60
    cross_region: true
```

This comprehensive data architecture strategy provides a scalable foundation for NovaCron's evolution into a distributed microservices platform while maintaining data consistency, performance, and reliability across all components.