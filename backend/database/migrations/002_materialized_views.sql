-- Materialized Views for Complex Query Optimization
-- Target: Pre-computed aggregates for dashboard and reporting

-- =====================================================
-- DASHBOARD MATERIALIZED VIEW
-- =====================================================

-- Real-time dashboard statistics (refreshed every minute)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_dashboard_stats AS
WITH vm_summary AS (
    SELECT 
        organization_id,
        state,
        COUNT(*) as vm_count,
        SUM(cpu_cores) as total_cpu,
        SUM(memory_mb) as total_memory_mb,
        SUM(disk_gb) as total_disk_gb,
        AVG(cpu_cores) as avg_cpu,
        AVG(memory_mb) as avg_memory_mb
    FROM vms
    WHERE state != 'error'
    GROUP BY organization_id, state
),
node_summary AS (
    SELECT 
        COUNT(*) as total_nodes,
        COUNT(*) FILTER (WHERE status = 'online') as online_nodes,
        SUM(cpu_cores) as total_node_cpu,
        SUM(memory_mb) as total_node_memory,
        SUM(disk_gb) as total_node_disk,
        AVG(EXTRACT(EPOCH FROM (NOW() - last_heartbeat))) as avg_heartbeat_age
    FROM nodes
),
recent_alerts AS (
    SELECT 
        severity,
        COUNT(*) as alert_count
    FROM alerts
    WHERE created_at > NOW() - INTERVAL '24 hours'
        AND resolved = false
    GROUP BY severity
),
resource_utilization AS (
    SELECT 
        AVG(cpu_usage) as avg_cpu_usage,
        AVG(memory_percent) as avg_memory_usage,
        MAX(cpu_usage) as max_cpu_usage,
        MAX(memory_percent) as max_memory_usage,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage) as p95_cpu,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_percent) as p95_memory
    FROM (
        SELECT DISTINCT ON (vm_id)
            vm_id,
            cpu_usage,
            memory_percent
        FROM vm_metrics
        WHERE timestamp > NOW() - INTERVAL '5 minutes'
        ORDER BY vm_id, timestamp DESC
    ) latest_metrics
)
SELECT 
    NOW() as calculated_at,
    vm_summary.*,
    node_summary.*,
    recent_alerts.*,
    resource_utilization.*
FROM vm_summary
CROSS JOIN node_summary
CROSS JOIN recent_alerts
CROSS JOIN resource_utilization;

CREATE UNIQUE INDEX ON mv_dashboard_stats (organization_id, state);
CREATE INDEX ON mv_dashboard_stats (calculated_at DESC);

-- =====================================================
-- VM METRICS AGGREGATION
-- =====================================================

-- Hourly VM metrics rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_metrics_hourly AS
SELECT 
    vm_id,
    date_trunc('hour', timestamp) as hour,
    COUNT(*) as sample_count,
    AVG(cpu_usage) as avg_cpu,
    MIN(cpu_usage) as min_cpu,
    MAX(cpu_usage) as max_cpu,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cpu_usage) as median_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage) as p95_cpu,
    AVG(memory_percent) as avg_memory,
    MAX(memory_percent) as max_memory,
    SUM(disk_read_bytes) as total_disk_read,
    SUM(disk_write_bytes) as total_disk_write,
    SUM(network_rx_bytes) as total_network_rx,
    SUM(network_tx_bytes) as total_network_tx
FROM vm_metrics
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY vm_id, date_trunc('hour', timestamp);

CREATE UNIQUE INDEX ON mv_vm_metrics_hourly (vm_id, hour);
CREATE INDEX ON mv_vm_metrics_hourly (hour DESC);

-- Daily VM metrics rollup
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_metrics_daily AS
SELECT 
    vm_id,
    date_trunc('day', timestamp) as day,
    COUNT(*) as sample_count,
    AVG(cpu_usage) as avg_cpu,
    MAX(cpu_usage) as max_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage) as p95_cpu,
    AVG(memory_percent) as avg_memory,
    MAX(memory_percent) as max_memory,
    SUM(disk_read_bytes) as total_disk_read,
    SUM(disk_write_bytes) as total_disk_write,
    SUM(network_rx_bytes) as total_network_rx,
    SUM(network_tx_bytes) as total_network_tx
FROM vm_metrics
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY vm_id, date_trunc('day', timestamp);

CREATE UNIQUE INDEX ON mv_vm_metrics_daily (vm_id, day);
CREATE INDEX ON mv_vm_metrics_daily (day DESC);

-- =====================================================
-- VM LISTING OPTIMIZATION
-- =====================================================

-- Pre-computed VM listing with latest metrics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_listing AS
WITH latest_metrics AS (
    SELECT DISTINCT ON (vm_id)
        vm_id,
        cpu_usage,
        memory_percent,
        timestamp as last_metric_time
    FROM vm_metrics
    WHERE timestamp > NOW() - INTERVAL '10 minutes'
    ORDER BY vm_id, timestamp DESC
),
vm_tags AS (
    SELECT 
        id,
        metadata->>'tags' as tags,
        metadata->>'environment' as environment,
        metadata->>'project' as project
    FROM vms
)
SELECT 
    v.id,
    v.name,
    v.state,
    v.node_id,
    v.cpu_cores,
    v.memory_mb,
    v.disk_gb,
    v.owner_id,
    v.organization_id,
    v.created_at,
    v.updated_at,
    n.hostname as node_hostname,
    n.status as node_status,
    u.username as owner_username,
    u.email as owner_email,
    o.name as organization_name,
    lm.cpu_usage as current_cpu_usage,
    lm.memory_percent as current_memory_percent,
    lm.last_metric_time,
    vt.tags,
    vt.environment,
    vt.project,
    CASE 
        WHEN v.state = 'running' AND lm.last_metric_time IS NULL THEN 'no_metrics'
        WHEN v.state = 'running' AND lm.last_metric_time < NOW() - INTERVAL '5 minutes' THEN 'stale_metrics'
        WHEN v.state = 'running' THEN 'healthy'
        ELSE v.state::text
    END as health_status
FROM vms v
LEFT JOIN nodes n ON v.node_id = n.id
LEFT JOIN users u ON v.owner_id = u.id
LEFT JOIN organizations o ON v.organization_id = o.id
LEFT JOIN latest_metrics lm ON v.id = lm.vm_id
LEFT JOIN vm_tags vt ON v.id = vt.id;

CREATE UNIQUE INDEX ON mv_vm_listing (id);
CREATE INDEX ON mv_vm_listing (organization_id, state);
CREATE INDEX ON mv_vm_listing (owner_id);
CREATE INDEX ON mv_vm_listing (node_id);
CREATE INDEX ON mv_vm_listing (created_at DESC);
CREATE INDEX ON mv_vm_listing (health_status);

-- =====================================================
-- ALERT SUMMARY VIEW
-- =====================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_alert_summary AS
WITH alert_stats AS (
    SELECT 
        resource_id,
        resource_type,
        severity,
        COUNT(*) as alert_count,
        MAX(created_at) as latest_alert,
        ARRAY_AGG(title ORDER BY created_at DESC) as recent_titles
    FROM alerts
    WHERE created_at > NOW() - INTERVAL '7 days'
        AND resolved = false
    GROUP BY resource_id, resource_type, severity
)
SELECT 
    resource_id,
    resource_type,
    severity,
    alert_count,
    latest_alert,
    recent_titles[1:5] as top_5_alerts
FROM alert_stats;

CREATE INDEX ON mv_alert_summary (resource_id);
CREATE INDEX ON mv_alert_summary (severity, latest_alert DESC);

-- =====================================================
-- NODE CAPACITY VIEW
-- =====================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_node_capacity AS
WITH vm_allocation AS (
    SELECT 
        node_id,
        COUNT(*) as vm_count,
        SUM(cpu_cores) as allocated_cpu,
        SUM(memory_mb) as allocated_memory_mb,
        SUM(disk_gb) as allocated_disk_gb
    FROM vms
    WHERE state IN ('running', 'paused')
    GROUP BY node_id
),
latest_node_metrics AS (
    SELECT DISTINCT ON (node_id)
        node_id,
        cpu_usage,
        memory_percent,
        disk_percent,
        load_average
    FROM node_metrics
    WHERE timestamp > NOW() - INTERVAL '5 minutes'
    ORDER BY node_id, timestamp DESC
)
SELECT 
    n.id,
    n.name,
    n.hostname,
    n.status,
    n.cpu_cores as total_cpu,
    n.memory_mb as total_memory_mb,
    n.disk_gb as total_disk_gb,
    COALESCE(va.vm_count, 0) as vm_count,
    COALESCE(va.allocated_cpu, 0) as allocated_cpu,
    COALESCE(va.allocated_memory_mb, 0) as allocated_memory_mb,
    COALESCE(va.allocated_disk_gb, 0) as allocated_disk_gb,
    n.cpu_cores - COALESCE(va.allocated_cpu, 0) as available_cpu,
    n.memory_mb - COALESCE(va.allocated_memory_mb, 0) as available_memory_mb,
    n.disk_gb - COALESCE(va.allocated_disk_gb, 0) as available_disk_gb,
    ROUND(COALESCE(va.allocated_cpu::numeric / NULLIF(n.cpu_cores, 0) * 100, 0), 2) as cpu_allocation_percent,
    ROUND(COALESCE(va.allocated_memory_mb::numeric / NULLIF(n.memory_mb, 0) * 100, 0), 2) as memory_allocation_percent,
    lnm.cpu_usage as current_cpu_usage,
    lnm.memory_percent as current_memory_usage,
    lnm.disk_percent as current_disk_usage,
    lnm.load_average
FROM nodes n
LEFT JOIN vm_allocation va ON n.id = va.node_id
LEFT JOIN latest_node_metrics lnm ON n.id = lnm.node_id
WHERE n.status = 'online';

CREATE UNIQUE INDEX ON mv_node_capacity (id);
CREATE INDEX ON mv_node_capacity (available_cpu DESC);
CREATE INDEX ON mv_node_capacity (available_memory_mb DESC);

-- =====================================================
-- USER ACTIVITY SUMMARY
-- =====================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_activity AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.role,
    u.organization_id,
    COUNT(DISTINCT al.id) as total_actions_30d,
    COUNT(DISTINCT al.id) FILTER (WHERE al.created_at > NOW() - INTERVAL '24 hours') as actions_24h,
    COUNT(DISTINCT al.id) FILTER (WHERE al.created_at > NOW() - INTERVAL '7 days') as actions_7d,
    MAX(al.created_at) as last_activity,
    COUNT(DISTINCT v.id) as owned_vms,
    SUM(v.cpu_cores) as total_cpu_owned,
    SUM(v.memory_mb) as total_memory_owned
FROM users u
LEFT JOIN audit_logs al ON u.id = al.user_id AND al.created_at > NOW() - INTERVAL '30 days'
LEFT JOIN vms v ON u.id = v.owner_id
GROUP BY u.id, u.username, u.email, u.role, u.organization_id;

CREATE UNIQUE INDEX ON mv_user_activity (id);
CREATE INDEX ON mv_user_activity (organization_id);
CREATE INDEX ON mv_user_activity (last_activity DESC NULLS LAST);

-- =====================================================
-- REFRESH POLICIES
-- =====================================================

-- Create refresh function
CREATE OR REPLACE FUNCTION refresh_materialized_views()
RETURNS void AS $$
BEGIN
    -- Refresh frequently accessed views
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_listing;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_node_capacity;
    
    -- Refresh less frequently accessed views
    IF EXTRACT(MINUTE FROM NOW()) % 5 = 0 THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_metrics_hourly;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_alert_summary;
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_user_activity;
    END IF;
    
    -- Refresh daily rollups once per hour
    IF EXTRACT(MINUTE FROM NOW()) = 0 THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_metrics_daily;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Schedule automatic refresh (requires pg_cron extension)
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule('refresh-materialized-views', '* * * * *', 'SELECT refresh_materialized_views();');

COMMENT ON MATERIALIZED VIEW mv_dashboard_stats IS 'Pre-computed dashboard statistics for fast loading';
COMMENT ON MATERIALIZED VIEW mv_vm_listing IS 'Optimized VM listing with joined data';
COMMENT ON MATERIALIZED VIEW mv_vm_metrics_hourly IS 'Hourly rollup of VM metrics';
COMMENT ON MATERIALIZED VIEW mv_vm_metrics_daily IS 'Daily rollup of VM metrics';
COMMENT ON MATERIALIZED VIEW mv_alert_summary IS 'Alert aggregation for quick access';
COMMENT ON MATERIALIZED VIEW mv_node_capacity IS 'Node capacity and allocation summary';
COMMENT ON MATERIALIZED VIEW mv_user_activity IS 'User activity tracking and resource usage';