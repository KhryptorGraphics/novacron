-- NovaCron Initial Database Schema
-- Version: 1.0.0
-- Description: Complete database schema with security, performance, and audit capabilities

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create custom enum types
DO $$ BEGIN
    CREATE TYPE user_role_type AS ENUM ('admin', 'operator', 'user', 'viewer');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE vm_state_type AS ENUM ('creating', 'running', 'stopped', 'suspended', 'error', 'migrating', 'deleting');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE alert_severity_type AS ENUM ('info', 'warning', 'error', 'critical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE audit_action_type AS ENUM ('create', 'read', 'update', 'delete', 'login', 'logout', 'start', 'stop', 'migrate');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- Users table with enhanced security
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role user_role_type NOT NULL DEFAULT 'user',
    tenant_id VARCHAR(255) NOT NULL DEFAULT 'default',
    last_login_at TIMESTAMP WITH TIME ZONE,
    password_changed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    failed_login_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMP WITH TIME ZONE,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    api_key_hash VARCHAR(255),
    session_token_hash VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_email CHECK (email ~* '^[A-Za-z0-9._%-]+@[A-Za-z0-9.-]+[.][A-Za-z]+$'),
    CONSTRAINT username_length CHECK (length(username) >= 3 AND length(username) <= 255),
    CONSTRAINT password_requirements CHECK (length(password_hash) >= 60) -- bcrypt hash length
);

-- User sessions for enhanced security tracking
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token_hash VARCHAR(255) NOT NULL UNIQUE,
    ip_address INET NOT NULL,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Roles and permissions system
CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    permissions JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_roles (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    assigned_by UUID REFERENCES users(id),
    PRIMARY KEY (user_id, role_id)
);

-- Compute nodes table
CREATE TABLE IF NOT EXISTS compute_nodes (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    hostname VARCHAR(255) NOT NULL,
    ip_address INET NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'unknown',
    hypervisor_type VARCHAR(50) NOT NULL DEFAULT 'kvm',
    cpu_cores INTEGER NOT NULL DEFAULT 0,
    cpu_threads INTEGER NOT NULL DEFAULT 0,
    memory_total BIGINT NOT NULL DEFAULT 0,
    memory_available BIGINT NOT NULL DEFAULT 0,
    storage_total BIGINT NOT NULL DEFAULT 0,
    storage_available BIGINT NOT NULL DEFAULT 0,
    vm_count INTEGER NOT NULL DEFAULT 0,
    load_average DECIMAL(5,2) DEFAULT 0,
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enhanced VMs table
CREATE TABLE IF NOT EXISTS vms (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    state vm_state_type NOT NULL DEFAULT 'creating',
    node_id VARCHAR(255) REFERENCES compute_nodes(id),
    owner_id UUID NOT NULL REFERENCES users(id),
    tenant_id VARCHAR(255) NOT NULL DEFAULT 'default',
    
    -- VM Configuration
    cpu_cores INTEGER NOT NULL DEFAULT 1,
    memory_mb BIGINT NOT NULL DEFAULT 1024,
    disk_size_gb BIGINT NOT NULL DEFAULT 20,
    
    -- Network Configuration
    ip_address INET,
    mac_address MACADDR,
    network_id VARCHAR(255),
    
    -- Advanced Configuration
    config JSONB DEFAULT '{}',
    template_id VARCHAR(255),
    snapshot_policy JSONB DEFAULT '{}',
    backup_policy JSONB DEFAULT '{}',
    
    -- Lifecycle Management
    scheduled_start TIMESTAMP WITH TIME ZONE,
    scheduled_stop TIMESTAMP WITH TIME ZONE,
    auto_start BOOLEAN DEFAULT FALSE,
    
    -- Metadata and Tags
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_vm_name CHECK (length(name) >= 1 AND length(name) <= 255),
    CONSTRAINT positive_resources CHECK (cpu_cores > 0 AND memory_mb > 0 AND disk_size_gb > 0)
);

-- VM metrics with time-series data
CREATE TABLE IF NOT EXISTS vm_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id VARCHAR(255) NOT NULL REFERENCES vms(id) ON DELETE CASCADE,
    node_id VARCHAR(255) REFERENCES compute_nodes(id),
    
    -- Resource Utilization
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_bytes BIGINT,
    memory_usage_percent DECIMAL(5,2),
    disk_usage_bytes BIGINT,
    disk_usage_percent DECIMAL(5,2),
    
    -- Network Statistics
    network_rx_bytes BIGINT DEFAULT 0,
    network_tx_bytes BIGINT DEFAULT 0,
    network_rx_packets BIGINT DEFAULT 0,
    network_tx_packets BIGINT DEFAULT 0,
    network_rx_errors BIGINT DEFAULT 0,
    network_tx_errors BIGINT DEFAULT 0,
    
    -- Disk I/O Statistics
    disk_read_bytes BIGINT DEFAULT 0,
    disk_write_bytes BIGINT DEFAULT 0,
    disk_read_ops BIGINT DEFAULT 0,
    disk_write_ops BIGINT DEFAULT 0,
    disk_iops INTEGER DEFAULT 0,
    
    -- Additional Metrics
    load_average DECIMAL(5,2),
    uptime_seconds BIGINT,
    
    collected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT valid_percentages CHECK (
        (cpu_usage_percent IS NULL OR (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100)) AND
        (memory_usage_percent IS NULL OR (memory_usage_percent >= 0 AND memory_usage_percent <= 100)) AND
        (disk_usage_percent IS NULL OR (disk_usage_percent >= 0 AND disk_usage_percent <= 100))
    )
);

-- Node metrics
CREATE TABLE IF NOT EXISTS node_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id VARCHAR(255) NOT NULL REFERENCES compute_nodes(id) ON DELETE CASCADE,
    
    -- Resource Utilization
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_bytes BIGINT,
    memory_total_bytes BIGINT,
    memory_usage_percent DECIMAL(5,2),
    storage_usage_bytes BIGINT,
    storage_total_bytes BIGINT,
    storage_usage_percent DECIMAL(5,2),
    
    -- System Load
    load_average_1m DECIMAL(5,2),
    load_average_5m DECIMAL(5,2),
    load_average_15m DECIMAL(5,2),
    
    -- Network Statistics (aggregate)
    network_rx_bytes BIGINT DEFAULT 0,
    network_tx_bytes BIGINT DEFAULT 0,
    
    collected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- VM snapshots
CREATE TABLE IF NOT EXISTS vm_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id VARCHAR(255) NOT NULL REFERENCES vms(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    size_bytes BIGINT,
    created_by UUID NOT NULL REFERENCES users(id),
    parent_snapshot_id UUID REFERENCES vm_snapshots(id),
    storage_path TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(vm_id, name)
);

-- VM backups
CREATE TABLE IF NOT EXISTS vm_backups (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id VARCHAR(255) NOT NULL REFERENCES vms(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    backup_type VARCHAR(50) NOT NULL DEFAULT 'full',
    size_bytes BIGINT,
    storage_location TEXT NOT NULL,
    backup_status VARCHAR(50) NOT NULL DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID NOT NULL REFERENCES users(id),
    metadata JSONB DEFAULT '{}',
    retention_until TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_backup_type CHECK (backup_type IN ('full', 'incremental', 'differential'))
);

-- Alerts and monitoring
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    severity alert_severity_type NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'firing',
    
    -- Alert Source
    resource_type VARCHAR(100), -- 'vm', 'node', 'system'
    resource_id VARCHAR(255),
    metric_name VARCHAR(100),
    metric_value DECIMAL(10,2),
    threshold_value DECIMAL(10,2),
    
    -- Alert Management
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    acknowledged_by UUID REFERENCES users(id),
    resolved_at TIMESTAMP WITH TIME ZONE,
    resolved_by UUID REFERENCES users(id),
    
    -- Alert Rules
    alert_rule_id UUID,
    labels JSONB DEFAULT '{}',
    annotations JSONB DEFAULT '{}',
    
    fired_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_status CHECK (status IN ('firing', 'acknowledged', 'resolved', 'silenced'))
);

-- Alert rules for monitoring automation
CREATE TABLE IF NOT EXISTS alert_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT,
    query TEXT NOT NULL,
    condition_operator VARCHAR(10) NOT NULL CHECK (condition_operator IN ('>', '<', '>=', '<=', '==', '!=')),
    threshold_value DECIMAL(10,2) NOT NULL,
    severity alert_severity_type NOT NULL,
    evaluation_interval INTEGER NOT NULL DEFAULT 60, -- seconds
    labels JSONB DEFAULT '{}',
    annotations JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT TRUE,
    created_by UUID NOT NULL REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Comprehensive audit log
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    session_id UUID REFERENCES user_sessions(id),
    action audit_action_type NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    resource_name VARCHAR(255),
    
    -- Request Details
    ip_address INET,
    user_agent TEXT,
    request_method VARCHAR(10),
    request_path TEXT,
    request_body JSONB,
    
    -- Response Details
    status_code INTEGER,
    response_body JSONB,
    
    -- Change Tracking
    old_values JSONB,
    new_values JSONB,
    
    -- Additional Context
    tenant_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_status_code CHECK (status_code >= 100 AND status_code < 600)
);

-- VM migrations tracking
CREATE TABLE IF NOT EXISTS vm_migrations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id VARCHAR(255) NOT NULL REFERENCES vms(id) ON DELETE CASCADE,
    source_node_id VARCHAR(255) NOT NULL REFERENCES compute_nodes(id),
    target_node_id VARCHAR(255) NOT NULL REFERENCES compute_nodes(id),
    migration_type VARCHAR(50) NOT NULL DEFAULT 'live',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    progress_percent INTEGER DEFAULT 0,
    initiated_by UUID NOT NULL REFERENCES users(id),
    reason TEXT,
    downtime_seconds INTEGER,
    data_transferred_bytes BIGINT DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT valid_migration_type CHECK (migration_type IN ('live', 'offline', 'evacuation')),
    CONSTRAINT valid_migration_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT valid_progress CHECK (progress_percent >= 0 AND progress_percent <= 100),
    CONSTRAINT different_nodes CHECK (source_node_id != target_node_id)
);

-- System configuration
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(255) PRIMARY KEY,
    value TEXT NOT NULL,
    value_type VARCHAR(50) NOT NULL DEFAULT 'string',
    description TEXT,
    category VARCHAR(100),
    updated_by UUID REFERENCES users(id),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_value_type CHECK (value_type IN ('string', 'integer', 'boolean', 'json'))
);

-- Insert default configuration values
INSERT INTO system_config (key, value, value_type, description, category) VALUES
    ('max_vms_per_node', '50', 'integer', 'Maximum VMs allowed per compute node', 'capacity'),
    ('default_vm_memory_mb', '1024', 'integer', 'Default memory allocation for new VMs', 'defaults'),
    ('default_vm_disk_gb', '20', 'integer', 'Default disk size for new VMs', 'defaults'),
    ('backup_retention_days', '30', 'integer', 'Default backup retention period', 'backup'),
    ('snapshot_retention_days', '7', 'integer', 'Default snapshot retention period', 'backup'),
    ('alert_evaluation_interval', '60', 'integer', 'Alert rule evaluation interval in seconds', 'monitoring'),
    ('max_failed_login_attempts', '5', 'integer', 'Maximum failed login attempts before lockout', 'security'),
    ('session_timeout_minutes', '480', 'integer', 'Session timeout in minutes', 'security'),
    ('enable_audit_logging', 'true', 'boolean', 'Enable comprehensive audit logging', 'security')
ON CONFLICT (key) DO NOTHING;

-- Insert default roles
INSERT INTO roles (name, description, permissions) VALUES
    ('admin', 'Full system administrator', '["*"]'),
    ('operator', 'VM operations and monitoring', '["vm:*", "monitoring:read", "alerts:*"]'),
    ('user', 'Basic VM management', '["vm:create", "vm:read", "vm:update", "vm:delete", "monitoring:read"]'),
    ('viewer', 'Read-only access', '["vm:read", "monitoring:read", "alerts:read"]')
ON CONFLICT (name) DO NOTHING;