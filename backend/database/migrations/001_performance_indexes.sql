-- Performance Optimization Indexes for NovaCron
-- Target: <500ms dashboard queries, <200ms VM listing, <200ms monitoring queries

-- =====================================================
-- BTREE INDEXES FOR EXACT MATCH AND RANGE QUERIES
-- =====================================================

-- VMs table optimization
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_state_organization 
    ON vms(state, organization_id) 
    WHERE state IN ('running', 'stopped', 'paused');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_owner_state 
    ON vms(owner_id, state) 
    INCLUDE (name, node_id, created_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_node_state 
    ON vms(node_id, state) 
    WHERE node_id IS NOT NULL;

-- Composite index for dashboard queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_dashboard 
    ON vms(organization_id, state, created_at DESC) 
    INCLUDE (name, cpu_cores, memory_mb, disk_gb);

-- =====================================================
-- JSONB INDEXES USING GIN
-- =====================================================

-- GIN index for JSONB metadata searches
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_metadata_gin 
    ON vms USING gin (metadata);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_network_config_gin 
    ON vms USING gin (network_config);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nodes_metadata_gin 
    ON nodes USING gin (metadata);

-- Optimized JSONB path indexes for specific queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_metadata_tags 
    ON vms USING gin ((metadata -> 'tags'));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_details_gin 
    ON audit_logs USING gin (details);

-- =====================================================
-- TIME-SERIES OPTIMIZATION WITH BRIN INDEXES
-- =====================================================

-- BRIN indexes for time-series data (very space-efficient)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_timestamp_brin 
    ON vm_metrics USING brin (timestamp) 
    WITH (pages_per_range = 128);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_metrics_timestamp_brin 
    ON node_metrics USING brin (timestamp) 
    WITH (pages_per_range = 128);

-- Composite BRIN for correlated columns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_vm_timestamp_brin 
    ON vm_metrics USING brin (vm_id, timestamp) 
    WITH (pages_per_range = 64);

-- =====================================================
-- PARTIAL INDEXES FOR SPECIFIC QUERY PATTERNS
-- =====================================================

-- Active sessions index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_active 
    ON sessions(user_id, expires_at) 
    WHERE expires_at > NOW();

-- Unacknowledged alerts
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_unacknowledged 
    ON alerts(severity, created_at DESC) 
    WHERE acknowledged = false AND resolved = false;

-- Active migrations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_migrations_active 
    ON migrations(vm_id, status, started_at) 
    WHERE status IN ('pending', 'in_progress');

-- Recent audit logs
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_recent 
    ON audit_logs(user_id, created_at DESC) 
    WHERE created_at > NOW() - INTERVAL '30 days';

-- =====================================================
-- COVERING INDEXES TO AVOID TABLE LOOKUPS
-- =====================================================

-- Users table covering index for authentication
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_auth 
    ON users(email) 
    INCLUDE (id, username, password_hash, role, status, organization_id);

-- Nodes health monitoring
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_nodes_health 
    ON nodes(status, last_heartbeat DESC) 
    INCLUDE (name, hostname, ip_address, cpu_cores, memory_mb);

-- VM metrics latest reading
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_latest 
    ON vm_metrics(vm_id, timestamp DESC) 
    INCLUDE (cpu_usage, memory_percent, disk_read_bytes, disk_write_bytes);

-- =====================================================
-- EXPRESSION INDEXES FOR COMPUTED COLUMNS
-- =====================================================

-- Case-insensitive username/email search
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_lower 
    ON users(LOWER(email));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_username_lower 
    ON users(LOWER(username));

-- Date-based indexes for reporting
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_created_date 
    ON vms(DATE(created_at));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_logs_date 
    ON audit_logs(DATE(created_at));

-- =====================================================
-- HASH INDEXES FOR EQUALITY SEARCHES
-- =====================================================

-- Token lookups (hash is faster for exact match)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_token_hash 
    ON sessions USING hash (token);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_api_keys_hash 
    ON api_keys USING hash (key_hash);

-- =====================================================
-- UNIQUE INDEXES FOR CONSTRAINTS AND PERFORMANCE
-- =====================================================

-- Ensure uniqueness while improving lookup speed
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_unique_name_org 
    ON vms(LOWER(name), organization_id);

-- =====================================================
-- STATISTICS AND MAINTENANCE
-- =====================================================

-- Update statistics for query planner
ANALYZE vms;
ANALYZE vm_metrics;
ANALYZE nodes;
ANALYZE node_metrics;
ANALYZE users;
ANALYZE sessions;
ANALYZE audit_logs;
ANALYZE alerts;
ANALYZE migrations;

-- Set appropriate statistics targets for heavily queried columns
ALTER TABLE vm_metrics ALTER COLUMN timestamp SET STATISTICS 1000;
ALTER TABLE vm_metrics ALTER COLUMN vm_id SET STATISTICS 500;
ALTER TABLE vms ALTER COLUMN state SET STATISTICS 100;
ALTER TABLE vms ALTER COLUMN organization_id SET STATISTICS 100;

-- =====================================================
-- PERFORMANCE MONITORING VIEWS
-- =====================================================

-- View for monitoring index usage
CREATE OR REPLACE VIEW index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
    CASE WHEN idx_scan = 0 THEN 'UNUSED' ELSE 'ACTIVE' END as status
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- View for identifying missing indexes
CREATE OR REPLACE VIEW missing_indexes AS
SELECT 
    schemaname,
    tablename,
    seq_scan - idx_scan as seq_scan_excess,
    CASE 
        WHEN seq_scan - idx_scan > 0 THEN 
            'Table has ' || (seq_scan - idx_scan) || ' more sequential scans than index scans'
        ELSE 'Well indexed'
    END as assessment
FROM pg_stat_user_tables
WHERE seq_scan > idx_scan
ORDER BY seq_scan - idx_scan DESC;

COMMENT ON VIEW index_usage IS 'Monitor index usage and identify unused indexes';
COMMENT ON VIEW missing_indexes IS 'Identify tables that might benefit from additional indexes';