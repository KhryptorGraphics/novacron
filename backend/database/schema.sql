-- NovaCron Database Schema
-- PostgreSQL 14+

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Enum types
CREATE TYPE user_status AS ENUM ('active', 'inactive', 'suspended', 'pending');
CREATE TYPE user_role AS ENUM ('admin', 'operator', 'viewer');
CREATE TYPE vm_state AS ENUM ('running', 'stopped', 'paused', 'migrating', 'error');
CREATE TYPE migration_type AS ENUM ('cold', 'warm', 'live');
CREATE TYPE migration_status AS ENUM ('pending', 'in_progress', 'completed', 'failed', 'cancelled');
CREATE TYPE alert_severity AS ENUM ('critical', 'error', 'warning', 'info');
CREATE TYPE resource_type AS ENUM ('cpu', 'memory', 'disk', 'network');

-- Organizations table
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    size VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    role user_role DEFAULT 'viewer',
    status user_status DEFAULT 'pending',
    email_verified BOOLEAN DEFAULT FALSE,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(255),
    last_login TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sessions table
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    token TEXT UNIQUE NOT NULL,
    refresh_token TEXT UNIQUE,
    ip_address INET,
    user_agent TEXT,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_accessed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    revoked_at TIMESTAMP WITH TIME ZONE,
    selected_cluster_id TEXT,
    metadata JSONB DEFAULT '{}'
);

-- Runtime cluster catalog
CREATE TABLE runtime_clusters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    tier TEXT NOT NULL,
    interconnect_latency_ms DOUBLE PRECISION NOT NULL,
    interconnect_bandwidth_mbps DOUBLE PRECISION NOT NULL,
    growth_latency_penalty_ms DOUBLE PRECISION NOT NULL,
    growth_bandwidth_penalty_mbps DOUBLE PRECISION NOT NULL,
    current_node_count INTEGER NOT NULL DEFAULT 1,
    max_supported_node_count INTEGER NOT NULL DEFAULT 1,
    performance_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    growth_state TEXT NOT NULL DEFAULT 'expandable',
    federation_state TEXT NOT NULL DEFAULT 'cluster-local',
    degraded BOOLEAN NOT NULL DEFAULT FALSE,
    last_evaluated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- Runtime user memberships in logical clusters
CREATE TABLE runtime_cluster_memberships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    cluster_id TEXT NOT NULL REFERENCES runtime_clusters(id) ON DELETE CASCADE,
    state TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    source TEXT NOT NULL DEFAULT 'runtime',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, cluster_id)
);

-- Runtime user-to-cluster edge performance samples
CREATE TABLE runtime_user_cluster_edges (
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    cluster_id TEXT NOT NULL REFERENCES runtime_clusters(id) ON DELETE CASCADE,
    latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
    bandwidth_mbps DOUBLE PRECISION NOT NULL DEFAULT 0,
    recorded_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, cluster_id)
);

-- Audit logs table
CREATE TABLE audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Nodes table
CREATE TABLE nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    hostname VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET NOT NULL,
    port INTEGER DEFAULT 9000,
    status VARCHAR(50) DEFAULT 'offline',
    cpu_cores INTEGER,
    memory_mb BIGINT,
    disk_gb BIGINT,
    hypervisor_type VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- VMs table
CREATE TABLE vms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    node_id UUID REFERENCES nodes(id) ON DELETE SET NULL,
    state vm_state DEFAULT 'stopped',
    cpu_cores INTEGER NOT NULL,
    memory_mb INTEGER NOT NULL,
    disk_gb INTEGER NOT NULL,
    os_type VARCHAR(50),
    os_version VARCHAR(50),
    network_config JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- VM Metrics table (time-series data)
CREATE TABLE vm_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID NOT NULL REFERENCES vms(id) ON DELETE CASCADE,
    cpu_usage DECIMAL(5,2),
    memory_usage BIGINT,
    memory_percent DECIMAL(5,2),
    disk_read_bytes BIGINT,
    disk_write_bytes BIGINT,
    network_rx_bytes BIGINT,
    network_tx_bytes BIGINT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for time-series queries
CREATE INDEX idx_vm_metrics_vm_timestamp ON vm_metrics(vm_id, timestamp DESC);

-- Node Metrics table
CREATE TABLE node_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id UUID NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    cpu_usage DECIMAL(5,2),
    memory_usage BIGINT,
    memory_percent DECIMAL(5,2),
    disk_usage BIGINT,
    disk_percent DECIMAL(5,2),
    network_rx_bytes BIGINT,
    network_tx_bytes BIGINT,
    load_average DECIMAL(5,2)[3],
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_node_metrics_node_timestamp ON node_metrics(node_id, timestamp DESC);

-- Migrations table
CREATE TABLE migrations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID NOT NULL REFERENCES vms(id) ON DELETE CASCADE,
    source_node_id UUID REFERENCES nodes(id),
    target_node_id UUID REFERENCES nodes(id),
    type migration_type NOT NULL,
    status migration_status DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Alerts table
CREATE TABLE alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    message TEXT,
    severity alert_severity NOT NULL,
    resource_type resource_type,
    resource_id UUID,
    node_id UUID REFERENCES nodes(id) ON DELETE SET NULL,
    vm_id UUID REFERENCES vms(id) ON DELETE SET NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by UUID REFERENCES users(id),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Storage volumes table
CREATE TABLE storage_volumes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    node_id UUID REFERENCES nodes(id) ON DELETE SET NULL,
    vm_id UUID REFERENCES vms(id) ON DELETE SET NULL,
    size_gb INTEGER NOT NULL,
    used_gb INTEGER DEFAULT 0,
    path VARCHAR(500),
    type VARCHAR(50),
    encrypted BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Network interfaces table
CREATE TABLE network_interfaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID REFERENCES vms(id) ON DELETE CASCADE,
    name VARCHAR(50) NOT NULL,
    mac_address VARCHAR(17),
    ip_address INET,
    subnet_mask INET,
    gateway INET,
    vlan_id INTEGER,
    bridge_name VARCHAR(50),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Snapshots table
CREATE TABLE snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID NOT NULL REFERENCES vms(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    size_bytes BIGINT,
    state VARCHAR(50),
    parent_id UUID REFERENCES snapshots(id),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Jobs table (for async operations)
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    payload JSONB DEFAULT '{}',
    result JSONB,
    error TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API keys table
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    scopes TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_vms_node ON vms(node_id);
CREATE INDEX idx_vms_owner ON vms(owner_id);
CREATE INDEX idx_vms_state ON vms(state);
CREATE INDEX idx_migrations_vm ON migrations(vm_id);
CREATE INDEX idx_migrations_status ON migrations(status);
CREATE INDEX idx_alerts_severity ON alerts(severity);
CREATE INDEX idx_alerts_acknowledged ON alerts(acknowledged);
CREATE INDEX idx_audit_logs_user ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_created ON audit_logs(created_at DESC);
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_refresh_token ON sessions(refresh_token);
CREATE INDEX idx_runtime_cluster_memberships_user ON runtime_cluster_memberships(user_id);
CREATE INDEX idx_runtime_cluster_memberships_cluster ON runtime_cluster_memberships(cluster_id);
CREATE INDEX idx_runtime_clusters_performance ON runtime_clusters(performance_score DESC, tier);
CREATE INDEX idx_runtime_user_cluster_edges_user ON runtime_user_cluster_edges(user_id);
CREATE INDEX idx_sessions_token ON sessions(token);

-- Functions
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_nodes_updated_at BEFORE UPDATE ON nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vms_updated_at BEFORE UPDATE ON vms
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_migrations_updated_at BEFORE UPDATE ON migrations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Initial data
INSERT INTO organizations (name, slug, size) VALUES
    ('Default Organization', 'default', '1-10');

-- Create default admin user (password: admin123)
INSERT INTO users (email, username, password_hash, first_name, last_name, role, status, email_verified) VALUES
    ('admin@novacron.io', 'admin', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Admin', 'User', 'admin', 'active', true);
