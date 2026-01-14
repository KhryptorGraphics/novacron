-- NovaCron Materialized Views for Reporting and Analytics
-- Version: 1.0.0
-- Description: Performance-optimized materialized views for dashboards and reporting

-- VM Summary Statistics (refreshed every 5 minutes)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_summary AS
SELECT 
    date_trunc('hour', NOW()) as snapshot_time,
    tenant_id,
    COUNT(*) as total_vms,
    COUNT(*) FILTER (WHERE state = 'running') as running_vms,
    COUNT(*) FILTER (WHERE state = 'stopped') as stopped_vms,
    COUNT(*) FILTER (WHERE state = 'error') as error_vms,
    COUNT(*) FILTER (WHERE state = 'creating') as creating_vms,
    COUNT(*) FILTER (WHERE state = 'migrating') as migrating_vms,
    SUM(cpu_cores) as total_cpu_cores,
    SUM(memory_mb) as total_memory_mb,
    SUM(disk_size_gb) as total_disk_gb,
    AVG(cpu_cores) as avg_cpu_cores,
    AVG(memory_mb) as avg_memory_mb,
    AVG(disk_size_gb) as avg_disk_gb
FROM vms
GROUP BY tenant_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_vm_summary_tenant ON mv_vm_summary(tenant_id);

-- Node Resource Utilization Summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_node_utilization AS
WITH latest_metrics AS (
    SELECT DISTINCT ON (node_id) 
        node_id,
        cpu_usage_percent,
        memory_usage_percent,
        storage_usage_percent,
        collected_at
    FROM node_metrics
    ORDER BY node_id, collected_at DESC
)
SELECT 
    cn.id as node_id,
    cn.name as node_name,
    cn.status,
    cn.cpu_cores,
    cn.cpu_threads,
    cn.memory_total,
    cn.storage_total,
    cn.vm_count,
    COALESCE(lm.cpu_usage_percent, 0) as current_cpu_usage,
    COALESCE(lm.memory_usage_percent, 0) as current_memory_usage,
    COALESCE(lm.storage_usage_percent, 0) as current_storage_usage,
    cn.load_average,
    CASE 
        WHEN cn.vm_count = 0 THEN 0
        ELSE ROUND((cn.vm_count::decimal / 50) * 100, 2) -- Assuming max 50 VMs per node
    END as vm_density_percent,
    cn.last_heartbeat,
    lm.collected_at as metrics_updated_at
FROM compute_nodes cn
LEFT JOIN latest_metrics lm ON cn.id = lm.node_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_node_utilization_id ON mv_node_utilization(node_id);

-- VM Performance Aggregates (hourly)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_performance_hourly AS
SELECT 
    vm_id,
    date_trunc('hour', collected_at) as hour,
    COUNT(*) as sample_count,
    AVG(cpu_usage_percent) as avg_cpu_usage,
    MAX(cpu_usage_percent) as max_cpu_usage,
    MIN(cpu_usage_percent) as min_cpu_usage,
    AVG(memory_usage_percent) as avg_memory_usage,
    MAX(memory_usage_percent) as max_memory_usage,
    MIN(memory_usage_percent) as min_memory_usage,
    AVG(disk_usage_percent) as avg_disk_usage,
    MAX(disk_usage_percent) as max_disk_usage,
    SUM(network_rx_bytes) as total_network_rx,
    SUM(network_tx_bytes) as total_network_tx,
    SUM(disk_read_bytes) as total_disk_read,
    SUM(disk_write_bytes) as total_disk_write,
    AVG(disk_iops) as avg_iops,
    MAX(disk_iops) as max_iops
FROM vm_metrics
WHERE collected_at >= NOW() - INTERVAL '7 days'
GROUP BY vm_id, date_trunc('hour', collected_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_vm_performance_hourly_vm_hour ON mv_vm_performance_hourly(vm_id, hour);

-- Daily VM Performance Summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_performance_daily AS
SELECT 
    vm_id,
    date_trunc('day', collected_at) as day,
    COUNT(*) as sample_count,
    AVG(cpu_usage_percent) as avg_cpu_usage,
    MAX(cpu_usage_percent) as max_cpu_usage,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage_percent) as p95_cpu_usage,
    AVG(memory_usage_percent) as avg_memory_usage,
    MAX(memory_usage_percent) as max_memory_usage,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_usage_percent) as p95_memory_usage,
    AVG(disk_usage_percent) as avg_disk_usage,
    MAX(disk_usage_percent) as max_disk_usage,
    SUM(network_rx_bytes) as total_network_rx,
    SUM(network_tx_bytes) as total_network_tx,
    SUM(disk_read_bytes) as total_disk_read,
    SUM(disk_write_bytes) as total_disk_write,
    AVG(disk_iops) as avg_iops,
    MAX(disk_iops) as max_iops,
    -- Calculate efficiency metrics
    CASE 
        WHEN AVG(cpu_usage_percent) > 0 THEN 
            ROUND((SUM(disk_read_bytes + disk_write_bytes) / AVG(cpu_usage_percent)) / 1024 / 1024, 2)
        ELSE 0 
    END as io_efficiency_mb_per_cpu_percent
FROM vm_metrics
WHERE collected_at >= NOW() - INTERVAL '30 days'
GROUP BY vm_id, date_trunc('day', collected_at);

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_vm_performance_daily_vm_day ON mv_vm_performance_daily(vm_id, day);

-- Alert Summary Statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_alert_summary AS
SELECT 
    date_trunc('day', fired_at) as alert_date,
    severity,
    resource_type,
    COUNT(*) as total_alerts,
    COUNT(*) FILTER (WHERE status = 'firing') as firing_alerts,
    COUNT(*) FILTER (WHERE status = 'acknowledged') as acknowledged_alerts,
    COUNT(*) FILTER (WHERE status = 'resolved') as resolved_alerts,
    COUNT(*) FILTER (WHERE acknowledged_at IS NOT NULL) as total_acknowledged,
    COUNT(*) FILTER (WHERE resolved_at IS NOT NULL) as total_resolved,
    AVG(EXTRACT(EPOCH FROM (COALESCE(acknowledged_at, NOW()) - fired_at))) as avg_time_to_ack_seconds,
    AVG(EXTRACT(EPOCH FROM (COALESCE(resolved_at, NOW()) - fired_at))) as avg_time_to_resolution_seconds
FROM alerts
WHERE fired_at >= NOW() - INTERVAL '30 days'
GROUP BY date_trunc('day', fired_at), severity, resource_type;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_alert_summary_date_sev_res ON mv_alert_summary(alert_date, severity, resource_type);

-- VM Migration Statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_migration_summary AS
SELECT 
    date_trunc('day', started_at) as migration_date,
    migration_type,
    COUNT(*) as total_migrations,
    COUNT(*) FILTER (WHERE status = 'completed') as successful_migrations,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_migrations,
    COUNT(*) FILTER (WHERE status = 'pending') as pending_migrations,
    COUNT(*) FILTER (WHERE status = 'running') as running_migrations,
    AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) FILTER (WHERE status = 'completed') as avg_migration_time_seconds,
    AVG(downtime_seconds) FILTER (WHERE status = 'completed' AND downtime_seconds IS NOT NULL) as avg_downtime_seconds,
    AVG(data_transferred_bytes) FILTER (WHERE status = 'completed') as avg_data_transferred,
    ROUND(
        (COUNT(*) FILTER (WHERE status = 'completed')::decimal / COUNT(*) * 100), 2
    ) as success_rate_percent
FROM vm_migrations
WHERE started_at >= NOW() - INTERVAL '30 days'
GROUP BY date_trunc('day', started_at), migration_type;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_migration_summary_date_type ON mv_migration_summary(migration_date, migration_type);

-- User Activity Summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_user_activity AS
SELECT 
    date_trunc('day', created_at) as activity_date,
    user_id,
    action,
    resource_type,
    COUNT(*) as activity_count,
    COUNT(DISTINCT session_id) as unique_sessions,
    COUNT(DISTINCT ip_address) as unique_ips
FROM audit_log
WHERE created_at >= NOW() - INTERVAL '30 days'
GROUP BY date_trunc('day', created_at), user_id, action, resource_type;

CREATE INDEX IF NOT EXISTS idx_mv_user_activity_date_user ON mv_user_activity(activity_date, user_id);

-- Tenant Resource Usage Summary
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_tenant_usage AS
WITH tenant_metrics AS (
    SELECT 
        v.tenant_id,
        COUNT(v.id) as vm_count,
        SUM(v.cpu_cores) as total_cpu_cores,
        SUM(v.memory_mb) as total_memory_mb,
        SUM(v.disk_size_gb) as total_disk_gb,
        COUNT(*) FILTER (WHERE v.state = 'running') as running_vms
    FROM vms v
    GROUP BY v.tenant_id
),
tenant_performance AS (
    SELECT 
        v.tenant_id,
        AVG(vm.cpu_usage_percent) as avg_cpu_usage,
        AVG(vm.memory_usage_percent) as avg_memory_usage,
        SUM(vm.network_rx_bytes + vm.network_tx_bytes) as total_network_bytes
    FROM vms v
    INNER JOIN vm_metrics vm ON v.id = vm.vm_id
    WHERE vm.collected_at >= NOW() - INTERVAL '24 hours'
    GROUP BY v.tenant_id
),
tenant_costs AS (
    SELECT 
        tenant_id,
        -- Simple cost calculation (can be enhanced with actual pricing)
        (total_cpu_cores * 10 + total_memory_mb * 0.01 + total_disk_gb * 0.5) as estimated_monthly_cost
    FROM tenant_metrics
)
SELECT 
    tm.tenant_id,
    tm.vm_count,
    tm.running_vms,
    tm.total_cpu_cores,
    tm.total_memory_mb,
    tm.total_disk_gb,
    COALESCE(tp.avg_cpu_usage, 0) as avg_cpu_usage_24h,
    COALESCE(tp.avg_memory_usage, 0) as avg_memory_usage_24h,
    COALESCE(tp.total_network_bytes, 0) as network_bytes_24h,
    tc.estimated_monthly_cost,
    NOW() as last_updated
FROM tenant_metrics tm
LEFT JOIN tenant_performance tp ON tm.tenant_id = tp.tenant_id
LEFT JOIN tenant_costs tc ON tm.tenant_id = tc.tenant_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_tenant_usage_tenant ON mv_tenant_usage(tenant_id);

-- System Health Overview
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_system_health AS
WITH node_health AS (
    SELECT 
        COUNT(*) as total_nodes,
        COUNT(*) FILTER (WHERE status = 'online') as healthy_nodes,
        COUNT(*) FILTER (WHERE status = 'offline') as offline_nodes,
        COUNT(*) FILTER (WHERE last_heartbeat < NOW() - INTERVAL '5 minutes') as stale_nodes
    FROM compute_nodes
),
vm_health AS (
    SELECT 
        COUNT(*) as total_vms,
        COUNT(*) FILTER (WHERE state = 'running') as running_vms,
        COUNT(*) FILTER (WHERE state = 'error') as error_vms
    FROM vms
),
alert_health AS (
    SELECT 
        COUNT(*) FILTER (WHERE status = 'firing' AND severity = 'critical') as critical_alerts,
        COUNT(*) FILTER (WHERE status = 'firing' AND severity = 'warning') as warning_alerts
    FROM alerts
    WHERE fired_at >= NOW() - INTERVAL '24 hours'
)
SELECT 
    nh.total_nodes,
    nh.healthy_nodes,
    nh.offline_nodes,
    nh.stale_nodes,
    ROUND((nh.healthy_nodes::decimal / nh.total_nodes * 100), 2) as node_health_percent,
    
    vh.total_vms,
    vh.running_vms,
    vh.error_vms,
    ROUND(((vh.total_vms - vh.error_vms)::decimal / vh.total_vms * 100), 2) as vm_health_percent,
    
    ah.critical_alerts,
    ah.warning_alerts,
    
    -- Overall system health score (0-100)
    ROUND((
        (nh.healthy_nodes::decimal / nh.total_nodes * 40) +
        ((vh.total_vms - vh.error_vms)::decimal / vh.total_vms * 40) +
        (CASE 
            WHEN ah.critical_alerts = 0 THEN 20 
            WHEN ah.critical_alerts <= 5 THEN 10
            ELSE 0
        END)
    ), 2) as overall_health_score,
    
    NOW() as snapshot_time
FROM node_health nh, vm_health vh, alert_health ah;

-- Create refresh functions for materialized views
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW mv_vm_summary;
    REFRESH MATERIALIZED VIEW mv_node_utilization;
    REFRESH MATERIALIZED VIEW mv_vm_performance_hourly;
    REFRESH MATERIALIZED VIEW mv_vm_performance_daily;
    REFRESH MATERIALIZED VIEW mv_alert_summary;
    REFRESH MATERIALIZED VIEW mv_migration_summary;
    REFRESH MATERIALIZED VIEW mv_user_activity;
    REFRESH MATERIALIZED VIEW mv_tenant_usage;
    REFRESH MATERIALIZED VIEW mv_system_health;
END;
$$ LANGUAGE plpgsql;

-- Create function to refresh critical views only (for frequent updates)
CREATE OR REPLACE FUNCTION refresh_critical_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW mv_vm_summary;
    REFRESH MATERIALIZED VIEW mv_node_utilization;
    REFRESH MATERIALIZED VIEW mv_system_health;
END;
$$ LANGUAGE plpgsql;

-- Create function to refresh performance views (for reporting)
CREATE OR REPLACE FUNCTION refresh_performance_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW mv_vm_performance_hourly;
    REFRESH MATERIALIZED VIEW mv_vm_performance_daily;
    REFRESH MATERIALIZED VIEW mv_tenant_usage;
END;
$$ LANGUAGE plpgsql;

-- Initial refresh of all materialized views
SELECT refresh_all_materialized_views();