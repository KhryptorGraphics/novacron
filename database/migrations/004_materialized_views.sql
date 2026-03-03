-- Materialized Views for Dashboard Performance Optimization
-- Creates pre-computed views to eliminate N+1 queries and achieve sub-50ms load times

-- ============================================================================
-- VM Listing Materialized View
-- Eliminates N+1 queries for VM list pages
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_listing AS
SELECT 
    v.id,
    v.name,
    v.state,
    v.node_id,
    COALESCE(CAST(v.config->>'cpu_cores' AS INTEGER), 1) as cpu_cores,
    COALESCE(CAST(v.config->>'memory_mb' AS INTEGER), 512) as memory_mb,
    COALESCE(CAST(v.config->>'disk_gb' AS INTEGER), 10) as disk_gb,
    v.owner_id,
    v.tenant_id,
    v.created_at,
    v.updated_at,
    
    -- Preload user information
    u.username as owner_username,
    u.email as owner_email,
    
    -- Preload tenant information  
    t.name as tenant_name,
    t.organization_id,
    
    -- Preload node information
    n.name as node_name,
    n.hostname as node_hostname,
    n.status as node_status,
    
    -- Latest metrics to avoid N+1 queries
    COALESCE(latest_metrics.cpu_usage, 0) as current_cpu_usage,
    COALESCE(latest_metrics.memory_percent, 0) as current_memory_percent,
    COALESCE(latest_metrics.disk_usage, 0) as current_disk_usage,
    COALESCE((latest_metrics.network_rx_bytes + latest_metrics.network_tx_bytes) / 1024 / 1024, 0) as current_network_mb,
    
    -- Health status computation
    CASE 
        WHEN latest_metrics.cpu_usage > 90 OR latest_metrics.memory_percent > 90 THEN 'critical'
        WHEN latest_metrics.cpu_usage > 80 OR latest_metrics.memory_percent > 80 THEN 'warning'
        WHEN latest_metrics.cpu_usage IS NULL THEN 'unknown'
        ELSE 'healthy'
    END as health_status,
    
    -- Resource utilization percentages
    CASE 
        WHEN CAST(v.config->>'cpu_cores' AS INTEGER) > 0 
        THEN ROUND((latest_metrics.cpu_usage / CAST(v.config->>'cpu_cores' AS INTEGER))::numeric, 2)
        ELSE 0 
    END as cpu_utilization_percent,
    
    COALESCE(latest_metrics.memory_percent, 0) as memory_utilization_percent,
    
    -- Last activity timestamp
    COALESCE(latest_metrics.timestamp, v.updated_at) as last_activity
    
FROM vms v
LEFT JOIN users u ON v.owner_id = u.id
LEFT JOIN tenants t ON v.tenant_id = t.id  
LEFT JOIN nodes n ON v.node_id = n.id
LEFT JOIN LATERAL (
    SELECT 
        cpu_usage,
        memory_usage,
        memory_percent,
        disk_usage,
        network_rx_bytes,
        network_tx_bytes,
        timestamp
    FROM vm_metrics 
    WHERE vm_id = v.id 
    ORDER BY timestamp DESC 
    LIMIT 1
) latest_metrics ON true;

-- Create indexes for fast filtering and sorting
CREATE INDEX IF NOT EXISTS idx_mv_vm_listing_tenant_id ON mv_vm_listing(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mv_vm_listing_owner_id ON mv_vm_listing(owner_id);  
CREATE INDEX IF NOT EXISTS idx_mv_vm_listing_node_id ON mv_vm_listing(node_id);
CREATE INDEX IF NOT EXISTS idx_mv_vm_listing_state ON mv_vm_listing(state);
CREATE INDEX IF NOT EXISTS idx_mv_vm_listing_health ON mv_vm_listing(health_status);
CREATE INDEX IF NOT EXISTS idx_mv_vm_listing_created ON mv_vm_listing(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_mv_vm_listing_org_id ON mv_vm_listing(organization_id);

-- Composite index for common filter combinations
CREATE INDEX IF NOT EXISTS idx_mv_vm_listing_org_state_created 
ON mv_vm_listing(organization_id, state, created_at DESC);

-- ============================================================================
-- Dashboard Statistics Materialized View  
-- Pre-computes expensive aggregate statistics for dashboards
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_dashboard_stats AS
WITH vm_stats AS (
    SELECT 
        t.organization_id,
        COUNT(*) as vm_count,
        COUNT(*) FILTER (WHERE v.state = 'running') as running_vms,
        COUNT(*) FILTER (WHERE v.state = 'stopped') as stopped_vms, 
        COUNT(*) FILTER (WHERE v.state = 'paused') as paused_vms,
        COUNT(*) FILTER (WHERE v.state = 'error') as error_vms,
        COALESCE(SUM(CAST(v.config->>'cpu_cores' AS INTEGER)), 0) as total_cpu,
        COALESCE(SUM(CAST(v.config->>'memory_mb' AS BIGINT)), 0) as total_memory_mb,
        COALESCE(SUM(CAST(v.config->>'disk_gb' AS BIGINT)), 0) as total_disk_gb
    FROM vms v
    LEFT JOIN tenants t ON v.tenant_id = t.id
    WHERE t.organization_id IS NOT NULL
    GROUP BY t.organization_id
),
node_stats AS (
    SELECT 
        t.organization_id,
        COUNT(*) as total_nodes,
        COUNT(*) FILTER (WHERE n.status = 'online') as online_nodes,
        COUNT(*) FILTER (WHERE n.status = 'offline') as offline_nodes,
        COUNT(*) FILTER (WHERE n.status = 'maintenance') as maintenance_nodes,
        COALESCE(SUM(n.total_cpu), 0) as total_node_cpu,
        COALESCE(SUM(n.total_memory_mb), 0) as total_node_memory_mb,
        COALESCE(SUM(n.total_disk_gb), 0) as total_node_disk_gb
    FROM nodes n
    LEFT JOIN tenants t ON n.tenant_id = t.id  
    WHERE t.organization_id IS NOT NULL
    GROUP BY t.organization_id
),
recent_metrics AS (
    SELECT 
        t.organization_id,
        AVG(vm.cpu_usage) as avg_cpu_usage,
        AVG(vm.memory_usage) as avg_memory_usage,
        AVG(vm.memory_percent) as avg_memory_percent,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY vm.cpu_usage) as p50_cpu_usage,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY vm.cpu_usage) as p95_cpu_usage,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY vm.cpu_usage) as p99_cpu_usage,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY vm.memory_percent) as p50_memory_usage,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY vm.memory_percent) as p95_memory_usage,
        PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY vm.memory_percent) as p99_memory_usage,
        MAX(vm.timestamp) as latest_metric_time
    FROM vm_metrics vm
    JOIN vms v ON vm.vm_id = v.id
    LEFT JOIN tenants t ON v.tenant_id = t.id
    WHERE vm.timestamp > NOW() - INTERVAL '1 hour'
      AND t.organization_id IS NOT NULL
    GROUP BY t.organization_id
),
alert_stats AS (
    SELECT 
        t.organization_id,
        COUNT(*) as alert_count,
        COUNT(*) FILTER (WHERE a.severity = 'critical') as critical_alerts,
        COUNT(*) FILTER (WHERE a.severity = 'warning') as warning_alerts,
        COUNT(*) FILTER (WHERE a.severity = 'info') as info_alerts,
        COUNT(*) FILTER (WHERE a.status = 'firing') as active_alerts,
        COUNT(*) FILTER (WHERE a.status = 'acknowledged') as acknowledged_alerts
    FROM alerts a
    JOIN vms v ON a.resource_id = v.id AND a.resource = 'vm'
    LEFT JOIN tenants t ON v.tenant_id = t.id
    WHERE a.status IN ('firing', 'acknowledged') 
      AND t.organization_id IS NOT NULL
    GROUP BY t.organization_id
),
storage_stats AS (
    SELECT 
        t.organization_id,
        COALESCE(SUM(CAST(vm.disk_read_bytes AS BIGINT)), 0) as total_disk_read_bytes,
        COALESCE(SUM(CAST(vm.disk_write_bytes AS BIGINT)), 0) as total_disk_write_bytes,
        COALESCE(SUM(CAST(vm.network_rx_bytes AS BIGINT)), 0) as total_network_rx_bytes,
        COALESCE(SUM(CAST(vm.network_tx_bytes AS BIGINT)), 0) as total_network_tx_bytes
    FROM vm_metrics vm
    JOIN vms v ON vm.vm_id = v.id
    LEFT JOIN tenants t ON v.tenant_id = t.id
    WHERE vm.timestamp > NOW() - INTERVAL '24 hours'
      AND t.organization_id IS NOT NULL
    GROUP BY t.organization_id
)
SELECT 
    COALESCE(vs.organization_id, ns.organization_id, rm.organization_id, als.organization_id, ss.organization_id) as organization_id,
    
    -- VM statistics
    COALESCE(vs.vm_count, 0) as vm_count,
    COALESCE(vs.running_vms, 0) as running_vms,
    COALESCE(vs.stopped_vms, 0) as stopped_vms,
    COALESCE(vs.paused_vms, 0) as paused_vms,
    COALESCE(vs.error_vms, 0) as error_vms,
    COALESCE(vs.total_cpu, 0) as total_cpu,
    COALESCE(vs.total_memory_mb, 0) as total_memory_mb,
    COALESCE(vs.total_disk_gb, 0) as total_disk_gb,
    
    -- Node statistics
    COALESCE(ns.total_nodes, 0) as total_nodes,
    COALESCE(ns.online_nodes, 0) as online_nodes,
    COALESCE(ns.offline_nodes, 0) as offline_nodes,
    COALESCE(ns.maintenance_nodes, 0) as maintenance_nodes,
    COALESCE(ns.total_node_cpu, 0) as total_node_cpu,
    COALESCE(ns.total_node_memory_mb, 0) as total_node_memory_mb,
    COALESCE(ns.total_node_disk_gb, 0) as total_node_disk_gb,
    
    -- Performance metrics
    COALESCE(ROUND(rm.avg_cpu_usage::numeric, 2), 0) as avg_cpu_usage,
    COALESCE(ROUND(rm.avg_memory_usage::numeric, 2), 0) as avg_memory_usage,
    COALESCE(ROUND(rm.avg_memory_percent::numeric, 2), 0) as avg_memory_percent,
    COALESCE(ROUND(rm.p50_cpu_usage::numeric, 2), 0) as p50_cpu_usage,
    COALESCE(ROUND(rm.p95_cpu_usage::numeric, 2), 0) as p95_cpu_usage,
    COALESCE(ROUND(rm.p99_cpu_usage::numeric, 2), 0) as p99_cpu_usage,
    COALESCE(ROUND(rm.p50_memory_usage::numeric, 2), 0) as p50_memory_usage,
    COALESCE(ROUND(rm.p95_memory_usage::numeric, 2), 0) as p95_memory_usage,
    COALESCE(ROUND(rm.p99_memory_usage::numeric, 2), 0) as p99_memory_usage,
    
    -- Alert statistics
    COALESCE(als.alert_count, 0) as alert_count,
    COALESCE(als.critical_alerts, 0) as critical_alerts,
    COALESCE(als.warning_alerts, 0) as warning_alerts,
    COALESCE(als.info_alerts, 0) as info_alerts,
    COALESCE(als.active_alerts, 0) as active_alerts,
    COALESCE(als.acknowledged_alerts, 0) as acknowledged_alerts,
    
    -- Storage and network I/O (last 24 hours)
    COALESCE(ss.total_disk_read_bytes, 0) as total_disk_read_bytes_24h,
    COALESCE(ss.total_disk_write_bytes, 0) as total_disk_write_bytes_24h,  
    COALESCE(ss.total_network_rx_bytes, 0) as total_network_rx_bytes_24h,
    COALESCE(ss.total_network_tx_bytes, 0) as total_network_tx_bytes_24h,
    
    -- Resource utilization percentages
    CASE 
        WHEN COALESCE(ns.total_node_cpu, 0) > 0 
        THEN ROUND((COALESCE(vs.total_cpu, 0)::float / ns.total_node_cpu * 100)::numeric, 2)
        ELSE 0 
    END as cpu_allocation_percent,
    
    CASE 
        WHEN COALESCE(ns.total_node_memory_mb, 0) > 0 
        THEN ROUND((COALESCE(vs.total_memory_mb, 0)::float / ns.total_node_memory_mb * 100)::numeric, 2)
        ELSE 0 
    END as memory_allocation_percent,
    
    CASE 
        WHEN COALESCE(ns.total_node_disk_gb, 0) > 0 
        THEN ROUND((COALESCE(vs.total_disk_gb, 0)::float / ns.total_node_disk_gb * 100)::numeric, 2)
        ELSE 0 
    END as disk_allocation_percent,
    
    -- Timestamps
    rm.latest_metric_time,
    NOW() as calculated_at

FROM vm_stats vs
FULL OUTER JOIN node_stats ns ON vs.organization_id = ns.organization_id
FULL OUTER JOIN recent_metrics rm ON COALESCE(vs.organization_id, ns.organization_id) = rm.organization_id
FULL OUTER JOIN alert_stats als ON COALESCE(vs.organization_id, ns.organization_id, rm.organization_id) = als.organization_id
FULL OUTER JOIN storage_stats ss ON COALESCE(vs.organization_id, ns.organization_id, rm.organization_id, als.organization_id) = ss.organization_id
WHERE COALESCE(vs.organization_id, ns.organization_id, rm.organization_id, als.organization_id, ss.organization_id) IS NOT NULL;

-- Index for fast dashboard queries
CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_dashboard_stats_org ON mv_dashboard_stats(organization_id);
CREATE INDEX IF NOT EXISTS idx_mv_dashboard_stats_calculated ON mv_dashboard_stats(calculated_at DESC);

-- ============================================================================
-- Node Capacity Materialized View
-- Optimizes node capacity and allocation queries
-- ============================================================================
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_node_capacity AS
SELECT 
    n.id,
    n.name,
    n.hostname,
    n.status,
    n.total_cpu,
    n.total_memory_mb,
    n.total_disk_gb,
    n.tenant_id,
    t.name as tenant_name,
    t.organization_id,
    
    -- VM allocation statistics
    COALESCE(vm_alloc.vm_count, 0) as vm_count,
    COALESCE(vm_alloc.allocated_cpu, 0) as allocated_cpu,
    COALESCE(vm_alloc.allocated_memory_mb, 0) as allocated_memory_mb,
    COALESCE(vm_alloc.allocated_disk_gb, 0) as allocated_disk_gb,
    
    -- Available resources
    n.total_cpu - COALESCE(vm_alloc.allocated_cpu, 0) as available_cpu,
    n.total_memory_mb - COALESCE(vm_alloc.allocated_memory_mb, 0) as available_memory_mb,
    n.total_disk_gb - COALESCE(vm_alloc.allocated_disk_gb, 0) as available_disk_gb,
    
    -- Allocation percentages
    CASE 
        WHEN n.total_cpu > 0 
        THEN ROUND((COALESCE(vm_alloc.allocated_cpu, 0)::float / n.total_cpu * 100)::numeric, 2)
        ELSE 0 
    END as cpu_allocation_percent,
    
    CASE 
        WHEN n.total_memory_mb > 0 
        THEN ROUND((COALESCE(vm_alloc.allocated_memory_mb, 0)::float / n.total_memory_mb * 100)::numeric, 2)
        ELSE 0 
    END as memory_allocation_percent,
    
    CASE 
        WHEN n.total_disk_gb > 0 
        THEN ROUND((COALESCE(vm_alloc.allocated_disk_gb, 0)::float / n.total_disk_gb * 100)::numeric, 2)
        ELSE 0 
    END as disk_allocation_percent,
    
    -- Current resource usage (from recent metrics)
    COALESCE(recent_usage.current_cpu_usage, 0) as current_cpu_usage,
    COALESCE(recent_usage.current_memory_usage, 0) as current_memory_usage,
    COALESCE(recent_usage.current_load_1, 0) as current_load_1,
    COALESCE(recent_usage.current_load_5, 0) as current_load_5,
    COALESCE(recent_usage.current_load_15, 0) as current_load_15,
    
    -- Health indicators
    CASE 
        WHEN n.status != 'online' THEN 'offline'
        WHEN recent_usage.current_cpu_usage > 90 OR recent_usage.current_memory_usage > 90 THEN 'critical'
        WHEN recent_usage.current_cpu_usage > 80 OR recent_usage.current_memory_usage > 80 THEN 'warning'
        ELSE 'healthy'
    END as health_status,
    
    recent_usage.last_metric_time,
    n.updated_at,
    NOW() as calculated_at
    
FROM nodes n
LEFT JOIN tenants t ON n.tenant_id = t.id
LEFT JOIN (
    SELECT 
        node_id,
        COUNT(*) as vm_count,
        COALESCE(SUM(CAST(config->>'cpu_cores' AS INTEGER)), 0) as allocated_cpu,
        COALESCE(SUM(CAST(config->>'memory_mb' AS BIGINT)), 0) as allocated_memory_mb,
        COALESCE(SUM(CAST(config->>'disk_gb' AS BIGINT)), 0) as allocated_disk_gb
    FROM vms 
    WHERE state IN ('running', 'paused')
    GROUP BY node_id
) vm_alloc ON n.id = vm_alloc.node_id
LEFT JOIN (
    SELECT 
        node_id,
        AVG(cpu_usage) as current_cpu_usage,
        AVG(memory_usage) as current_memory_usage, 
        AVG(load_average_1) as current_load_1,
        AVG(load_average_5) as current_load_5,
        AVG(load_average_15) as current_load_15,
        MAX(timestamp) as last_metric_time
    FROM system_metrics 
    WHERE timestamp > NOW() - INTERVAL '10 minutes'
    GROUP BY node_id
) recent_usage ON n.id = recent_usage.node_id;

-- Indexes for node capacity queries
CREATE INDEX IF NOT EXISTS idx_mv_node_capacity_status ON mv_node_capacity(status);
CREATE INDEX IF NOT EXISTS idx_mv_node_capacity_tenant ON mv_node_capacity(tenant_id);
CREATE INDEX IF NOT EXISTS idx_mv_node_capacity_org ON mv_node_capacity(organization_id);
CREATE INDEX IF NOT EXISTS idx_mv_node_capacity_health ON mv_node_capacity(health_status);
CREATE INDEX IF NOT EXISTS idx_mv_node_capacity_available_memory ON mv_node_capacity(available_memory_mb DESC);
CREATE INDEX IF NOT EXISTS idx_mv_node_capacity_available_cpu ON mv_node_capacity(available_cpu DESC);

-- Composite index for capacity planning queries
CREATE INDEX IF NOT EXISTS idx_mv_node_capacity_planning 
ON mv_node_capacity(status, available_cpu DESC, available_memory_mb DESC) 
WHERE status = 'online';

-- ============================================================================
-- VM Metrics Aggregation Views (Hourly and Daily)
-- Pre-aggregates metrics for fast historical queries
-- ============================================================================

-- Hourly aggregation view
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_metrics_hourly AS
SELECT 
    vm_id,
    date_trunc('hour', timestamp) as hour,
    COUNT(*) as sample_count,
    
    -- CPU metrics
    ROUND(AVG(cpu_usage)::numeric, 2) as avg_cpu,
    ROUND(MIN(cpu_usage)::numeric, 2) as min_cpu,
    ROUND(MAX(cpu_usage)::numeric, 2) as max_cpu,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage)::numeric, 2) as p95_cpu,
    
    -- Memory metrics  
    ROUND(AVG(memory_usage)::numeric, 2) as avg_memory,
    ROUND(MIN(memory_usage)::numeric, 2) as min_memory,
    ROUND(MAX(memory_usage)::numeric, 2) as max_memory,
    ROUND(AVG(memory_percent)::numeric, 2) as avg_memory_percent,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_percent)::numeric, 2) as p95_memory_percent,
    
    -- Disk I/O metrics
    COALESCE(SUM(disk_read_bytes), 0) as total_disk_read,
    COALESCE(SUM(disk_write_bytes), 0) as total_disk_write,
    COALESCE(AVG(disk_read_bytes), 0) as avg_disk_read,
    COALESCE(AVG(disk_write_bytes), 0) as avg_disk_write,
    COALESCE(MAX(disk_read_bytes), 0) as max_disk_read,
    COALESCE(MAX(disk_write_bytes), 0) as max_disk_write,
    
    -- Network I/O metrics
    COALESCE(SUM(network_rx_bytes), 0) as total_network_rx,
    COALESCE(SUM(network_tx_bytes), 0) as total_network_tx,
    COALESCE(AVG(network_rx_bytes), 0) as avg_network_rx,
    COALESCE(AVG(network_tx_bytes), 0) as avg_network_tx,
    COALESCE(MAX(network_rx_bytes), 0) as max_network_rx,
    COALESCE(MAX(network_tx_bytes), 0) as max_network_tx

FROM vm_metrics
GROUP BY vm_id, date_trunc('hour', timestamp);

-- Indexes for hourly metrics
CREATE INDEX IF NOT EXISTS idx_mv_vm_metrics_hourly_vm_hour ON mv_vm_metrics_hourly(vm_id, hour DESC);
CREATE INDEX IF NOT EXISTS idx_mv_vm_metrics_hourly_hour ON mv_vm_metrics_hourly(hour DESC);

-- Daily aggregation view  
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_vm_metrics_daily AS
SELECT 
    vm_id,
    date_trunc('day', timestamp) as day,
    COUNT(*) as sample_count,
    
    -- CPU metrics
    ROUND(AVG(cpu_usage)::numeric, 2) as avg_cpu,
    ROUND(MIN(cpu_usage)::numeric, 2) as min_cpu,
    ROUND(MAX(cpu_usage)::numeric, 2) as max_cpu,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage)::numeric, 2) as p95_cpu,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY cpu_usage)::numeric, 2) as p99_cpu,
    
    -- Memory metrics
    ROUND(AVG(memory_usage)::numeric, 2) as avg_memory,
    ROUND(MIN(memory_usage)::numeric, 2) as min_memory, 
    ROUND(MAX(memory_usage)::numeric, 2) as max_memory,
    ROUND(AVG(memory_percent)::numeric, 2) as avg_memory_percent,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_percent)::numeric, 2) as p95_memory_percent,
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY memory_percent)::numeric, 2) as p99_memory_percent,
    
    -- Disk I/O metrics
    COALESCE(SUM(disk_read_bytes), 0) as total_disk_read,
    COALESCE(SUM(disk_write_bytes), 0) as total_disk_write,
    COALESCE(AVG(disk_read_bytes), 0) as avg_disk_read,
    COALESCE(AVG(disk_write_bytes), 0) as avg_disk_write,
    
    -- Network I/O metrics
    COALESCE(SUM(network_rx_bytes), 0) as total_network_rx,
    COALESCE(SUM(network_tx_bytes), 0) as total_network_tx,
    COALESCE(AVG(network_rx_bytes), 0) as avg_network_rx,
    COALESCE(AVG(network_tx_bytes), 0) as avg_network_tx

FROM vm_metrics
GROUP BY vm_id, date_trunc('day', timestamp);

-- Indexes for daily metrics
CREATE INDEX IF NOT EXISTS idx_mv_vm_metrics_daily_vm_day ON mv_vm_metrics_daily(vm_id, day DESC);
CREATE INDEX IF NOT EXISTS idx_mv_vm_metrics_daily_day ON mv_vm_metrics_daily(day DESC);

-- ============================================================================
-- Refresh Functions and Triggers
-- Automatic refresh logic for materialized views
-- ============================================================================

-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_performance_views() RETURNS void AS $$
BEGIN
    -- Refresh in dependency order
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_listing;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_stats;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_node_capacity;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_metrics_hourly;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_metrics_daily;
    
    -- Log refresh
    INSERT INTO materialized_view_refresh_log (view_name, refreshed_at) VALUES
    ('mv_vm_listing', NOW()),
    ('mv_dashboard_stats', NOW()), 
    ('mv_node_capacity', NOW()),
    ('mv_vm_metrics_hourly', NOW()),
    ('mv_vm_metrics_daily', NOW());
END;
$$ LANGUAGE plpgsql;

-- Create refresh log table if it doesn't exist
CREATE TABLE IF NOT EXISTS materialized_view_refresh_log (
    id SERIAL PRIMARY KEY,
    view_name TEXT NOT NULL,
    refreshed_at TIMESTAMP DEFAULT NOW(),
    duration_ms INTEGER,
    row_count BIGINT
);

CREATE INDEX IF NOT EXISTS idx_mv_refresh_log_view_time ON materialized_view_refresh_log(view_name, refreshed_at DESC);

-- Function to refresh views with performance tracking
CREATE OR REPLACE FUNCTION refresh_performance_views_with_metrics() RETURNS void AS $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    duration_ms INTEGER;
    row_count BIGINT;
BEGIN
    -- VM Listing
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_vm_listing;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    GET DIAGNOSTICS row_count = ROW_COUNT;
    INSERT INTO materialized_view_refresh_log (view_name, refreshed_at, duration_ms, row_count) 
    VALUES ('mv_vm_listing', end_time, duration_ms, (SELECT COUNT(*) FROM mv_vm_listing));
    
    -- Dashboard Stats
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_dashboard_stats;
    end_time := clock_timestamp(); 
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    INSERT INTO materialized_view_refresh_log (view_name, refreshed_at, duration_ms, row_count) 
    VALUES ('mv_dashboard_stats', end_time, duration_ms, (SELECT COUNT(*) FROM mv_dashboard_stats));
    
    -- Node Capacity
    start_time := clock_timestamp();
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_node_capacity;
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    INSERT INTO materialized_view_refresh_log (view_name, refreshed_at, duration_ms, row_count) 
    VALUES ('mv_node_capacity', end_time, duration_ms, (SELECT COUNT(*) FROM mv_node_capacity));
    
    -- Metrics Hourly (only refresh recent data)
    start_time := clock_timestamp();
    DELETE FROM mv_vm_metrics_hourly WHERE hour >= date_trunc('hour', NOW() - INTERVAL '2 hours');
    INSERT INTO mv_vm_metrics_hourly 
    SELECT vm_id, date_trunc('hour', timestamp) as hour, COUNT(*) as sample_count,
           ROUND(AVG(cpu_usage)::numeric, 2) as avg_cpu,
           ROUND(MIN(cpu_usage)::numeric, 2) as min_cpu,
           ROUND(MAX(cpu_usage)::numeric, 2) as max_cpu,
           ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage)::numeric, 2) as p95_cpu,
           ROUND(AVG(memory_usage)::numeric, 2) as avg_memory,
           ROUND(MIN(memory_usage)::numeric, 2) as min_memory,
           ROUND(MAX(memory_usage)::numeric, 2) as max_memory,
           ROUND(AVG(memory_percent)::numeric, 2) as avg_memory_percent,
           ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_percent)::numeric, 2) as p95_memory_percent,
           COALESCE(SUM(disk_read_bytes), 0) as total_disk_read,
           COALESCE(SUM(disk_write_bytes), 0) as total_disk_write,
           COALESCE(AVG(disk_read_bytes), 0) as avg_disk_read,
           COALESCE(AVG(disk_write_bytes), 0) as avg_disk_write,
           COALESCE(MAX(disk_read_bytes), 0) as max_disk_read,
           COALESCE(MAX(disk_write_bytes), 0) as max_disk_write,
           COALESCE(SUM(network_rx_bytes), 0) as total_network_rx,
           COALESCE(SUM(network_tx_bytes), 0) as total_network_tx,
           COALESCE(AVG(network_rx_bytes), 0) as avg_network_rx,
           COALESCE(AVG(network_tx_bytes), 0) as avg_network_tx,
           COALESCE(MAX(network_rx_bytes), 0) as max_network_rx,
           COALESCE(MAX(network_tx_bytes), 0) as max_network_tx
    FROM vm_metrics
    WHERE timestamp >= date_trunc('hour', NOW() - INTERVAL '2 hours')
    GROUP BY vm_id, date_trunc('hour', timestamp);
    end_time := clock_timestamp();
    duration_ms := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
    INSERT INTO materialized_view_refresh_log (view_name, refreshed_at, duration_ms, row_count) 
    VALUES ('mv_vm_metrics_hourly', end_time, duration_ms, (SELECT COUNT(*) FROM mv_vm_metrics_hourly));
END;
$$ LANGUAGE plpgsql;

-- Schedule automatic refresh (requires pg_cron extension)
-- SELECT cron.schedule('refresh-performance-views', '*/2 * * * *', 'SELECT refresh_performance_views_with_metrics()');

-- ============================================================================ 
-- Performance Monitoring Functions
-- Functions to monitor materialized view performance and health
-- ============================================================================

-- Function to get materialized view statistics
CREATE OR REPLACE FUNCTION get_materialized_view_stats() 
RETURNS TABLE (
    view_name text,
    row_count bigint,
    size_mb numeric,
    last_refreshed timestamp,
    refresh_duration_ms integer,
    avg_refresh_duration_ms numeric
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname || '.' || matviewname as view_name,
        n_tup_ins + n_tup_upd + n_tup_del as row_count,
        ROUND((pg_total_relation_size(schemaname||'.'||matviewname) / 1024 / 1024)::numeric, 2) as size_mb,
        rl.refreshed_at as last_refreshed,
        rl.duration_ms as refresh_duration_ms,
        ROUND(AVG(rl.duration_ms) OVER (PARTITION BY rl.view_name ORDER BY rl.refreshed_at ROWS BETWEEN 4 PRECEDING AND CURRENT ROW)::numeric, 0) as avg_refresh_duration_ms
    FROM pg_stat_user_tables pgt
    RIGHT JOIN pg_matviews pmv ON pgt.relname = pmv.matviewname
    LEFT JOIN LATERAL (
        SELECT * FROM materialized_view_refresh_log 
        WHERE view_name = pmv.matviewname 
        ORDER BY refreshed_at DESC 
        LIMIT 1
    ) rl ON true
    WHERE pmv.matviewname LIKE 'mv_%'
    ORDER BY size_mb DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to analyze query performance on materialized views
CREATE OR REPLACE FUNCTION analyze_mv_query_performance() RETURNS void AS $$
BEGIN
    -- Update statistics for all materialized views
    ANALYZE mv_vm_listing;
    ANALYZE mv_dashboard_stats; 
    ANALYZE mv_node_capacity;
    ANALYZE mv_vm_metrics_hourly;
    ANALYZE mv_vm_metrics_daily;
END;
$$ LANGUAGE plpgsql;