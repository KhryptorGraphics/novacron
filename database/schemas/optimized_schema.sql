-- NovaCron Optimized Database Schema
-- Version: 2.0.0
-- Description: Performance-optimized schema with sharding, partitioning, and normalization
-- Database: PostgreSQL 15+ with TimescaleDB extension

-- ============================================================================
-- EXTENSIONS AND SETTINGS
-- ============================================================================

-- Required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "btree_gist";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_partman";
CREATE EXTENSION IF NOT EXISTS "postgres_fdw";

-- Performance settings
SET max_parallel_workers_per_gather = 4;
SET max_parallel_workers = 8;
SET max_parallel_maintenance_workers = 4;
SET effective_cache_size = '8GB';
SET shared_buffers = '2GB';
SET work_mem = '16MB';
SET maintenance_work_mem = '512MB';

-- ============================================================================
-- ENUM TYPES
-- ============================================================================

CREATE TYPE vm_state AS ENUM (
    'provisioning',
    'starting',
    'running',
    'stopping',
    'stopped',
    'suspended',
    'migrating',
    'error',
    'deleted'
);

CREATE TYPE node_status AS ENUM (
    'online',
    'offline',
    'maintenance',
    'draining',
    'error'
);

CREATE TYPE resource_category AS ENUM (
    'general',
    'compute',
    'memory',
    'storage',
    'gpu'
);

CREATE TYPE migration_type AS ENUM (
    'live',
    'offline',
    'evacuation'
);

CREATE TYPE alert_severity AS ENUM (
    'info',
    'warning',
    'error',
    'critical'
);

-- ============================================================================
-- LOOKUP TABLES (Normalized)
-- ============================================================================

-- VM States lookup table for better performance
CREATE TABLE vm_states (
    id SMALLINT PRIMARY KEY,
    name VARCHAR(20) UNIQUE NOT NULL,
    description TEXT,
    is_billable BOOLEAN DEFAULT TRUE,
    allows_operations BOOLEAN DEFAULT TRUE
);

INSERT INTO vm_states (id, name, description, is_billable, allows_operations) VALUES
    (1, 'provisioning', 'VM is being created', FALSE, FALSE),
    (2, 'starting', 'VM is starting up', TRUE, FALSE),
    (3, 'running', 'VM is operational', TRUE, TRUE),
    (4, 'stopping', 'VM is shutting down', TRUE, FALSE),
    (5, 'stopped', 'VM is stopped', FALSE, TRUE),
    (6, 'suspended', 'VM is suspended', FALSE, TRUE),
    (7, 'migrating', 'VM is being migrated', TRUE, FALSE),
    (8, 'error', 'VM encountered an error', FALSE, TRUE),
    (9, 'deleted', 'VM is marked for deletion', FALSE, FALSE);

-- Resource profiles for standardized configurations
CREATE TABLE resource_profiles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    cpu_cores SMALLINT NOT NULL CHECK (cpu_cores > 0),
    memory_mb INTEGER NOT NULL CHECK (memory_mb > 0),
    disk_gb INTEGER NOT NULL CHECK (disk_gb > 0),
    network_mbps INTEGER,
    iops_limit INTEGER,
    gpu_count SMALLINT DEFAULT 0,
    category resource_category NOT NULL,
    monthly_cost DECIMAL(10,2),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_resource_profiles_category (category),
    INDEX idx_resource_profiles_active (is_active) WHERE is_active = TRUE
);

-- ============================================================================
-- SHARDING INFRASTRUCTURE
-- ============================================================================

-- Shard configuration table
CREATE TABLE shard_config (
    shard_id SERIAL PRIMARY KEY,
    shard_name VARCHAR(100) UNIQUE NOT NULL,
    host VARCHAR(255) NOT NULL,
    port INTEGER DEFAULT 5432,
    database_name VARCHAR(100) NOT NULL,
    username VARCHAR(100) NOT NULL,
    password_encrypted TEXT NOT NULL,
    connection_pool_size INTEGER DEFAULT 20,
    weight INTEGER DEFAULT 100,
    status VARCHAR(20) DEFAULT 'active',
    region VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    CHECK (status IN ('active', 'readonly', 'maintenance', 'offline')),
    INDEX idx_shard_status (status)
);

-- Tenant to shard mapping
CREATE TABLE tenant_shards (
    tenant_id VARCHAR(100) PRIMARY KEY,
    shard_id INTEGER NOT NULL REFERENCES shard_config(shard_id),
    vm_count INTEGER DEFAULT 0,
    storage_bytes BIGINT DEFAULT 0,
    assigned_at TIMESTAMPTZ DEFAULT NOW(),
    last_rebalanced TIMESTAMPTZ,
    
    INDEX idx_tenant_shard (shard_id)
);

-- ============================================================================
-- CORE TABLES (Optimized)
-- ============================================================================

-- Organizations table with better indexing
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    tier VARCHAR(20) DEFAULT 'standard',
    max_vms INTEGER DEFAULT 100,
    max_storage_gb INTEGER DEFAULT 1000,
    billing_email VARCHAR(255),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_org_tenant (tenant_id),
    INDEX idx_org_active (is_active) WHERE is_active = TRUE
);

-- Users table with optimized indexes
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID NOT NULL REFERENCES organizations(id),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    is_active BOOLEAN DEFAULT TRUE,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret VARCHAR(255),
    last_login_at TIMESTAMPTZ,
    failed_attempts INTEGER DEFAULT 0,
    locked_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_users_org (organization_id),
    INDEX idx_users_email (email),
    INDEX idx_users_active (is_active) WHERE is_active = TRUE,
    INDEX idx_users_role (role)
);

-- Compute nodes table with status tracking
CREATE TABLE compute_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) UNIQUE NOT NULL,
    hostname VARCHAR(255) NOT NULL,
    ip_address INET NOT NULL,
    region VARCHAR(50) NOT NULL,
    availability_zone VARCHAR(50),
    status node_status NOT NULL DEFAULT 'offline',
    
    -- Resource capacity
    cpu_cores INTEGER NOT NULL,
    memory_total_mb BIGINT NOT NULL,
    storage_total_gb BIGINT NOT NULL,
    gpu_count INTEGER DEFAULT 0,
    
    -- Current usage (denormalized for performance)
    cpu_allocated INTEGER DEFAULT 0,
    memory_allocated_mb BIGINT DEFAULT 0,
    storage_allocated_gb BIGINT DEFAULT 0,
    vm_count INTEGER DEFAULT 0,
    
    -- Health tracking
    last_heartbeat TIMESTAMPTZ,
    health_score DECIMAL(3,2) DEFAULT 1.0,
    
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    INDEX idx_nodes_status (status),
    INDEX idx_nodes_region (region, availability_zone),
    INDEX idx_nodes_heartbeat (last_heartbeat DESC),
    INDEX idx_nodes_capacity (cpu_cores, memory_total_mb) WHERE status = 'online'
);

-- VMs table - Partitioned by creation date
CREATE TABLE vms (
    id UUID DEFAULT gen_random_uuid(),
    tenant_id VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    state_id SMALLINT NOT NULL REFERENCES vm_states(id),
    node_id UUID REFERENCES compute_nodes(id),
    owner_id UUID NOT NULL REFERENCES users(id),
    resource_profile_id INTEGER REFERENCES resource_profiles(id),
    
    -- Network configuration
    ip_address INET,
    mac_address MACADDR,
    hostname VARCHAR(255),
    
    -- Lifecycle timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    stopped_at TIMESTAMPTZ,
    deleted_at TIMESTAMPTZ,
    
    -- Billing tracking
    billable_hours DECIMAL(10,2) DEFAULT 0,
    last_billed_at TIMESTAMPTZ,
    
    PRIMARY KEY (id, created_at)
) PARTITION BY RANGE (created_at);

-- Create indexes on parent table (inherited by partitions)
CREATE INDEX idx_vms_tenant ON vms (tenant_id, state_id);
CREATE INDEX idx_vms_node ON vms (node_id) WHERE node_id IS NOT NULL;
CREATE INDEX idx_vms_owner ON vms (owner_id);
CREATE INDEX idx_vms_state ON vms (state_id);
CREATE INDEX idx_vms_created ON vms (created_at DESC);

-- Create initial partitions
CREATE TABLE vms_2024_q1 PARTITION OF vms
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE vms_2024_q2 PARTITION OF vms
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

CREATE TABLE vms_2024_q3 PARTITION OF vms
    FOR VALUES FROM ('2024-07-01') TO ('2024-10-01');

CREATE TABLE vms_2024_q4 PARTITION OF vms
    FOR VALUES FROM ('2024-10-01') TO ('2025-01-01');

-- ============================================================================
-- TIME-SERIES DATA (TimescaleDB)
-- ============================================================================

-- VM metrics hypertable
CREATE TABLE vm_metrics (
    time TIMESTAMPTZ NOT NULL,
    vm_id UUID NOT NULL,
    
    -- CPU metrics
    cpu_usage_percent REAL CHECK (cpu_usage_percent >= 0 AND cpu_usage_percent <= 100),
    cpu_throttle_percent REAL,
    
    -- Memory metrics
    memory_used_mb INTEGER,
    memory_percent REAL CHECK (memory_percent >= 0 AND memory_percent <= 100),
    memory_swap_mb INTEGER,
    
    -- Disk I/O metrics
    disk_read_bytes BIGINT,
    disk_write_bytes BIGINT,
    disk_iops INTEGER,
    
    -- Network metrics
    network_rx_bytes BIGINT,
    network_tx_bytes BIGINT,
    network_rx_packets BIGINT,
    network_tx_packets BIGINT,
    
    -- Additional metrics
    processes INTEGER,
    threads INTEGER,
    
    PRIMARY KEY (vm_id, time)
);

-- Convert to hypertable
SELECT create_hypertable('vm_metrics', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Add compression
ALTER TABLE vm_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'vm_id',
    timescaledb.compress_orderby = 'time DESC'
);

-- Compression policy (compress after 7 days)
SELECT add_compression_policy('vm_metrics', INTERVAL '7 days', 
    if_not_exists => TRUE);

-- Retention policy (keep 90 days)
SELECT add_retention_policy('vm_metrics', INTERVAL '90 days',
    if_not_exists => TRUE);

-- Node metrics hypertable
CREATE TABLE node_metrics (
    time TIMESTAMPTZ NOT NULL,
    node_id UUID NOT NULL,
    
    -- Resource utilization
    cpu_usage_percent REAL,
    memory_used_mb BIGINT,
    memory_percent REAL,
    storage_used_gb BIGINT,
    storage_percent REAL,
    
    -- System load
    load_1m REAL,
    load_5m REAL,
    load_15m REAL,
    
    -- Network totals
    network_rx_bytes BIGINT,
    network_tx_bytes BIGINT,
    
    -- Temperature (if available)
    cpu_temp_celsius REAL,
    
    PRIMARY KEY (node_id, time)
);

-- Convert to hypertable
SELECT create_hypertable('node_metrics', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Add compression
ALTER TABLE node_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'node_id',
    timescaledb.compress_orderby = 'time DESC'
);

-- ============================================================================
-- CONTINUOUS AGGREGATES (TimescaleDB)
-- ============================================================================

-- Hourly VM metrics aggregate
CREATE MATERIALIZED VIEW vm_metrics_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    vm_id,
    AVG(cpu_usage_percent) AS avg_cpu,
    MAX(cpu_usage_percent) AS max_cpu,
    MIN(cpu_usage_percent) AS min_cpu,
    AVG(memory_percent) AS avg_memory,
    MAX(memory_percent) AS max_memory,
    SUM(disk_read_bytes) AS total_disk_read,
    SUM(disk_write_bytes) AS total_disk_write,
    SUM(network_rx_bytes) AS total_network_rx,
    SUM(network_tx_bytes) AS total_network_tx,
    COUNT(*) AS sample_count
FROM vm_metrics
GROUP BY hour, vm_id
WITH NO DATA;

-- Refresh policy
SELECT add_continuous_aggregate_policy('vm_metrics_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Daily VM metrics aggregate
CREATE MATERIALIZED VIEW vm_metrics_daily
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', time) AS day,
    vm_id,
    AVG(cpu_usage_percent) AS avg_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage_percent) AS p95_cpu,
    MAX(cpu_usage_percent) AS max_cpu,
    AVG(memory_percent) AS avg_memory,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_percent) AS p95_memory,
    MAX(memory_percent) AS max_memory,
    SUM(disk_read_bytes) AS total_disk_read,
    SUM(disk_write_bytes) AS total_disk_write,
    SUM(network_rx_bytes) AS total_network_rx,
    SUM(network_tx_bytes) AS total_network_tx
FROM vm_metrics
GROUP BY day, vm_id
WITH NO DATA;

-- ============================================================================
-- AUDIT AND COMPLIANCE
-- ============================================================================

-- Audit log table - partitioned by month
CREATE TABLE audit_log (
    id UUID DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id UUID REFERENCES users(id),
    tenant_id VARCHAR(100),
    action VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    changes JSONB,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,
    
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- Create indexes
CREATE INDEX idx_audit_user ON audit_log (user_id, timestamp DESC);
CREATE INDEX idx_audit_resource ON audit_log (resource_type, resource_id, timestamp DESC);
CREATE INDEX idx_audit_action ON audit_log (action, timestamp DESC);

-- Create initial partitions
CREATE TABLE audit_log_2024_01 PARTITION OF audit_log
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

CREATE TABLE audit_log_2024_02 PARTITION OF audit_log
    FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');

-- ============================================================================
-- MATERIALIZED VIEWS FOR REPORTING
-- ============================================================================

-- Resource utilization summary
CREATE MATERIALIZED VIEW mv_resource_utilization AS
SELECT
    n.id AS node_id,
    n.name AS node_name,
    n.region,
    n.cpu_cores AS total_cpu,
    n.cpu_allocated AS used_cpu,
    ROUND((n.cpu_allocated::DECIMAL / NULLIF(n.cpu_cores, 0)) * 100, 2) AS cpu_percent,
    n.memory_total_mb AS total_memory,
    n.memory_allocated_mb AS used_memory,
    ROUND((n.memory_allocated_mb::DECIMAL / NULLIF(n.memory_total_mb, 0)) * 100, 2) AS memory_percent,
    n.vm_count,
    n.status,
    n.last_heartbeat
FROM compute_nodes n
WITH DATA;

CREATE UNIQUE INDEX idx_mv_resource_util_node ON mv_resource_utilization (node_id);

-- Tenant resource summary
CREATE MATERIALIZED VIEW mv_tenant_summary AS
SELECT
    v.tenant_id,
    COUNT(*) AS total_vms,
    COUNT(*) FILTER (WHERE s.name = 'running') AS running_vms,
    COUNT(*) FILTER (WHERE s.name = 'stopped') AS stopped_vms,
    SUM(rp.cpu_cores) AS total_cpu,
    SUM(rp.memory_mb) AS total_memory_mb,
    SUM(rp.disk_gb) AS total_disk_gb,
    SUM(v.billable_hours) AS total_billable_hours
FROM vms v
JOIN vm_states s ON v.state_id = s.id
LEFT JOIN resource_profiles rp ON v.resource_profile_id = rp.id
WHERE v.deleted_at IS NULL
GROUP BY v.tenant_id
WITH DATA;

CREATE UNIQUE INDEX idx_mv_tenant_summary ON mv_tenant_summary (tenant_id);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get optimal shard for new tenant
CREATE OR REPLACE FUNCTION get_optimal_shard(p_tenant_id VARCHAR)
RETURNS INTEGER AS $$
DECLARE
    v_shard_id INTEGER;
BEGIN
    -- Get existing assignment
    SELECT shard_id INTO v_shard_id
    FROM tenant_shards
    WHERE tenant_id = p_tenant_id;
    
    IF v_shard_id IS NOT NULL THEN
        RETURN v_shard_id;
    END IF;
    
    -- Find shard with least load
    WITH shard_loads AS (
        SELECT 
            sc.shard_id,
            sc.weight,
            COALESCE(SUM(ts.vm_count), 0) AS total_vms,
            COALESCE(SUM(ts.storage_bytes), 0) AS total_storage
        FROM shard_config sc
        LEFT JOIN tenant_shards ts ON sc.shard_id = ts.shard_id
        WHERE sc.status = 'active'
        GROUP BY sc.shard_id, sc.weight
    )
    SELECT shard_id INTO v_shard_id
    FROM shard_loads
    ORDER BY (total_vms::DECIMAL / weight) ASC
    LIMIT 1;
    
    -- Assign tenant to shard
    INSERT INTO tenant_shards (tenant_id, shard_id)
    VALUES (p_tenant_id, v_shard_id);
    
    RETURN v_shard_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update node capacity on VM changes
CREATE OR REPLACE FUNCTION update_node_capacity()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE compute_nodes
        SET 
            cpu_allocated = cpu_allocated + (SELECT cpu_cores FROM resource_profiles WHERE id = NEW.resource_profile_id),
            memory_allocated_mb = memory_allocated_mb + (SELECT memory_mb FROM resource_profiles WHERE id = NEW.resource_profile_id),
            storage_allocated_gb = storage_allocated_gb + (SELECT disk_gb FROM resource_profiles WHERE id = NEW.resource_profile_id),
            vm_count = vm_count + 1,
            updated_at = NOW()
        WHERE id = NEW.node_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE compute_nodes
        SET 
            cpu_allocated = cpu_allocated - (SELECT cpu_cores FROM resource_profiles WHERE id = OLD.resource_profile_id),
            memory_allocated_mb = memory_allocated_mb - (SELECT memory_mb FROM resource_profiles WHERE id = OLD.resource_profile_id),
            storage_allocated_gb = storage_allocated_gb - (SELECT disk_gb FROM resource_profiles WHERE id = OLD.resource_profile_id),
            vm_count = vm_count - 1,
            updated_at = NOW()
        WHERE id = OLD.node_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic capacity updates
CREATE TRIGGER trg_update_node_capacity_insert
    AFTER INSERT ON vms
    FOR EACH ROW
    EXECUTE FUNCTION update_node_capacity();

CREATE TRIGGER trg_update_node_capacity_delete
    AFTER DELETE ON vms
    FOR EACH ROW
    EXECUTE FUNCTION update_node_capacity();

-- ============================================================================
-- PARTITION MANAGEMENT
-- ============================================================================

-- Automated partition creation for time-series tables
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS void AS $$
DECLARE
    tables text[] := ARRAY['audit_log'];
    table_name text;
    start_date date;
    end_date date;
    partition_name text;
BEGIN
    FOREACH table_name IN ARRAY tables
    LOOP
        FOR i IN 0..2 LOOP
            start_date := date_trunc('month', CURRENT_DATE + (i || ' months')::interval);
            end_date := start_date + interval '1 month';
            partition_name := table_name || '_' || to_char(start_date, 'YYYY_MM');
            
            IF NOT EXISTS (
                SELECT 1 FROM pg_class WHERE relname = partition_name
            ) THEN
                EXECUTE format(
                    'CREATE TABLE IF NOT EXISTS %I PARTITION OF %I FOR VALUES FROM (%L) TO (%L)',
                    partition_name, table_name, start_date, end_date
                );
            END IF;
        END LOOP;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

-- Schedule monthly partition creation (requires pg_cron)
-- SELECT cron.schedule('create-partitions', '0 0 1 * *', 'SELECT create_monthly_partitions()');

-- ============================================================================
-- REFRESH MATERIALIZED VIEWS
-- ============================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_resource_utilization;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_tenant_summary;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh (requires pg_cron)
-- SELECT cron.schedule('refresh-views', '*/5 * * * *', 'SELECT refresh_materialized_views()');

-- ============================================================================
-- INITIAL DATA
-- ============================================================================

-- Insert standard resource profiles
INSERT INTO resource_profiles (name, cpu_cores, memory_mb, disk_gb, network_mbps, iops_limit, category, monthly_cost) VALUES
    ('nano.1', 1, 512, 10, 100, 100, 'general', 5.00),
    ('micro.1', 1, 1024, 20, 100, 200, 'general', 10.00),
    ('small.2', 2, 2048, 40, 250, 400, 'general', 20.00),
    ('medium.2', 2, 4096, 80, 500, 800, 'general', 40.00),
    ('large.4', 4, 8192, 160, 1000, 1600, 'general', 80.00),
    ('xlarge.8', 8, 16384, 320, 2000, 3200, 'general', 160.00),
    ('compute.4', 4, 4096, 80, 1000, 2000, 'compute', 100.00),
    ('compute.8', 8, 8192, 160, 2000, 4000, 'compute', 200.00),
    ('memory.2', 2, 16384, 80, 500, 1000, 'memory', 120.00),
    ('memory.4', 4, 32768, 160, 1000, 2000, 'memory', 240.00),
    ('storage.2', 2, 4096, 1000, 500, 10000, 'storage', 150.00),
    ('gpu.4', 4, 16384, 160, 2000, 3200, 'gpu', 500.00)
ON CONFLICT (name) DO NOTHING;

-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Create roles
CREATE ROLE novacron_app;
CREATE ROLE novacron_readonly;
CREATE ROLE novacron_admin;

-- Grant permissions
GRANT USAGE ON SCHEMA public TO novacron_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO novacron_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO novacron_app;

GRANT USAGE ON SCHEMA public TO novacron_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO novacron_readonly;

GRANT ALL PRIVILEGES ON SCHEMA public TO novacron_admin;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO novacron_admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO novacron_admin;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE vms IS 'Virtual machines table - partitioned by creation date for scalability';
COMMENT ON TABLE vm_metrics IS 'Time-series metrics for VMs - TimescaleDB hypertable with compression';
COMMENT ON TABLE resource_profiles IS 'Standardized VM configurations for consistent resource allocation';
COMMENT ON TABLE tenant_shards IS 'Tenant to database shard mapping for horizontal scaling';
COMMENT ON FUNCTION get_optimal_shard IS 'Returns the optimal shard for a tenant based on current load distribution';