-- Database schema for NovaCron
-- Execute this file to create the complete database schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user' NOT NULL CHECK (role IN ('admin', 'operator', 'user')),
    tenant_id VARCHAR(255) DEFAULT 'default' NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- VMs table
CREATE TABLE IF NOT EXISTS vms (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    state VARCHAR(50) NOT NULL CHECK (state IN ('creating', 'created', 'starting', 'running', 'stopping', 'stopped', 'paused', 'migrating', 'failed', 'deleting')),
    node_id VARCHAR(255),
    owner_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    tenant_id VARCHAR(255) DEFAULT 'default' NOT NULL,
    config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- VM metrics table
CREATE TABLE IF NOT EXISTS vm_metrics (
    id SERIAL PRIMARY KEY,
    vm_id VARCHAR(255) REFERENCES vms(id) ON DELETE CASCADE,
    cpu_usage FLOAT NOT NULL DEFAULT 0,
    memory_usage FLOAT NOT NULL DEFAULT 0,
    disk_usage FLOAT NOT NULL DEFAULT 0,
    network_sent BIGINT NOT NULL DEFAULT 0,
    network_recv BIGINT NOT NULL DEFAULT 0,
    iops INTEGER NOT NULL DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    node_id VARCHAR(255) NOT NULL,
    cpu_usage FLOAT NOT NULL DEFAULT 0,
    memory_usage FLOAT NOT NULL DEFAULT 0,
    memory_total BIGINT NOT NULL DEFAULT 0,
    memory_available BIGINT NOT NULL DEFAULT 0,
    disk_usage FLOAT NOT NULL DEFAULT 0,
    disk_total BIGINT NOT NULL DEFAULT 0,
    disk_available BIGINT NOT NULL DEFAULT 0,
    network_sent BIGINT NOT NULL DEFAULT 0,
    network_recv BIGINT NOT NULL DEFAULT 0,
    load_average_1 FLOAT NOT NULL DEFAULT 0,
    load_average_5 FLOAT NOT NULL DEFAULT 0,
    load_average_15 FLOAT NOT NULL DEFAULT 0,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(50) NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    status VARCHAR(50) NOT NULL CHECK (status IN ('firing', 'resolved', 'acknowledged', 'suppressed')),
    resource VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    metric_name VARCHAR(255) NOT NULL,
    threshold FLOAT NOT NULL,
    current_value FLOAT NOT NULL,
    labels JSONB,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    acknowledged_by VARCHAR(255),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE IF NOT EXISTS sessions (
    id VARCHAR(255) PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    ip_address INET,
    user_agent TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Migrations table
CREATE TABLE IF NOT EXISTS migrations (
    id VARCHAR(255) PRIMARY KEY,
    vm_id VARCHAR(255) REFERENCES vms(id) ON DELETE CASCADE,
    source_node_id VARCHAR(255) NOT NULL,
    target_node_id VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('cold', 'warm', 'live')),
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    progress FLOAT DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    bytes_total BIGINT DEFAULT 0,
    bytes_transferred BIGINT DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Nodes table
CREATE TABLE IF NOT EXISTS nodes (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'offline' CHECK (status IN ('online', 'offline', 'maintenance')),
    capabilities JSONB,
    resources JSONB,
    labels JSONB,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance

-- VM metrics indexes
CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id ON vm_metrics(vm_id);
CREATE INDEX IF NOT EXISTS idx_vm_metrics_timestamp ON vm_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_timestamp ON vm_metrics(vm_id, timestamp DESC);

-- System metrics indexes
CREATE INDEX IF NOT EXISTS idx_system_metrics_node_id ON system_metrics(node_id);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_metrics_node_timestamp ON system_metrics(node_id, timestamp DESC);

-- VM indexes
CREATE INDEX IF NOT EXISTS idx_vms_state ON vms(state);
CREATE INDEX IF NOT EXISTS idx_vms_node_id ON vms(node_id);
CREATE INDEX IF NOT EXISTS idx_vms_owner_id ON vms(owner_id);
CREATE INDEX IF NOT EXISTS idx_vms_tenant_id ON vms(tenant_id);

-- Alert indexes
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_resource ON alerts(resource);
CREATE INDEX IF NOT EXISTS idx_alerts_start_time ON alerts(start_time DESC);

-- Session indexes
CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

-- Audit log indexes
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource);

-- Migration indexes
CREATE INDEX IF NOT EXISTS idx_migrations_vm_id ON migrations(vm_id);
CREATE INDEX IF NOT EXISTS idx_migrations_status ON migrations(status);
CREATE INDEX IF NOT EXISTS idx_migrations_created_at ON migrations(created_at DESC);

-- Node indexes
CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_nodes_last_seen ON nodes(last_seen DESC);

-- Create default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, role, tenant_id) 
VALUES ('admin', 'admin@novacron.local', '$2a$10$rDjR8Z8Z8Z8Z8Z8Z8Z8Z8uQ8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8Z8', 'admin', 'default')
ON CONFLICT (username) DO NOTHING;

-- Create default node
INSERT INTO nodes (id, name, address, status, capabilities, resources, labels)
VALUES (
    'default-node-01',
    'Default Node 1',
    'localhost:9000',
    'online',
    '{"kvm": true, "containers": true}',
    '{"cpu_cores": 8, "memory_gb": 32, "disk_gb": 1000}',
    '{"environment": "development", "zone": "default"}'
) ON CONFLICT (id) DO NOTHING;