-- TimescaleDB Optimization for Time-Series Data
-- Target: <200ms for real-time monitoring queries

-- =====================================================
-- ENABLE TIMESCALEDB EXTENSION
-- =====================================================

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================
-- CONVERT METRICS TABLES TO HYPERTABLES
-- =====================================================

-- Convert vm_metrics to hypertable
SELECT create_hypertable('vm_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Convert node_metrics to hypertable
SELECT create_hypertable('node_metrics', 'timestamp',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- =====================================================
-- COMPRESSION POLICIES FOR OLDER DATA
-- =====================================================

-- Enable compression on vm_metrics
ALTER TABLE vm_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'vm_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy (compress chunks older than 7 days)
SELECT add_compression_policy('vm_metrics', INTERVAL '7 days');

-- Enable compression on node_metrics
ALTER TABLE node_metrics SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'node_id',
    timescaledb.compress_orderby = 'timestamp DESC'
);

-- Add compression policy
SELECT add_compression_policy('node_metrics', INTERVAL '7 days');

-- =====================================================
-- CONTINUOUS AGGREGATES FOR REAL-TIME ROLLUPS
-- =====================================================

-- 5-minute rollup for real-time monitoring
CREATE MATERIALIZED VIEW IF NOT EXISTS vm_metrics_5min
WITH (timescaledb.continuous) AS
SELECT 
    vm_id,
    time_bucket('5 minutes', timestamp) AS bucket,
    COUNT(*) as sample_count,
    AVG(cpu_usage) as avg_cpu,
    MAX(cpu_usage) as max_cpu,
    MIN(cpu_usage) as min_cpu,
    AVG(memory_percent) as avg_memory,
    MAX(memory_percent) as max_memory,
    SUM(disk_read_bytes) as total_disk_read,
    SUM(disk_write_bytes) as total_disk_write,
    SUM(network_rx_bytes) as total_network_rx,
    SUM(network_tx_bytes) as total_network_tx
FROM vm_metrics
GROUP BY vm_id, bucket
WITH NO DATA;

-- Create refresh policy for 5-minute aggregate
SELECT add_continuous_aggregate_policy('vm_metrics_5min',
    start_offset => INTERVAL '10 minutes',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute'
);

-- 1-hour rollup for dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS vm_metrics_hourly_ts
WITH (timescaledb.continuous) AS
SELECT 
    vm_id,
    time_bucket('1 hour', timestamp) AS hour,
    COUNT(*) as sample_count,
    AVG(cpu_usage) as avg_cpu,
    MAX(cpu_usage) as max_cpu,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cpu_usage) as median_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage) as p95_cpu,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY cpu_usage) as p99_cpu,
    AVG(memory_percent) as avg_memory,
    MAX(memory_percent) as max_memory,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_percent) as p95_memory,
    SUM(disk_read_bytes) as total_disk_read,
    SUM(disk_write_bytes) as total_disk_write,
    SUM(network_rx_bytes) as total_network_rx,
    SUM(network_tx_bytes) as total_network_tx
FROM vm_metrics
GROUP BY vm_id, hour
WITH NO DATA;

-- Create refresh policy for hourly aggregate
SELECT add_continuous_aggregate_policy('vm_metrics_hourly_ts',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour'
);

-- Daily rollup for historical analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS vm_metrics_daily_ts
WITH (timescaledb.continuous) AS
SELECT 
    vm_id,
    time_bucket('1 day', timestamp) AS day,
    COUNT(*) as sample_count,
    AVG(cpu_usage) as avg_cpu,
    MAX(cpu_usage) as max_cpu,
    MIN(cpu_usage) as min_cpu,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cpu_usage) as median_cpu,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY cpu_usage) as p95_cpu,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY cpu_usage) as p99_cpu,
    AVG(memory_percent) as avg_memory,
    MAX(memory_percent) as max_memory,
    MIN(memory_percent) as min_memory,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY memory_percent) as p95_memory,
    SUM(disk_read_bytes) as total_disk_read,
    SUM(disk_write_bytes) as total_disk_write,
    SUM(network_rx_bytes) as total_network_rx,
    SUM(network_tx_bytes) as total_network_tx
FROM vm_metrics
GROUP BY vm_id, day
WITH NO DATA;

-- Create refresh policy for daily aggregate
SELECT add_continuous_aggregate_policy('vm_metrics_daily_ts',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day'
);

-- =====================================================
-- NODE METRICS CONTINUOUS AGGREGATES
-- =====================================================

-- 5-minute rollup for nodes
CREATE MATERIALIZED VIEW IF NOT EXISTS node_metrics_5min
WITH (timescaledb.continuous) AS
SELECT 
    node_id,
    time_bucket('5 minutes', timestamp) AS bucket,
    AVG(cpu_usage) as avg_cpu,
    MAX(cpu_usage) as max_cpu,
    AVG(memory_percent) as avg_memory,
    MAX(memory_percent) as max_memory,
    AVG(disk_percent) as avg_disk,
    MAX(disk_percent) as max_disk,
    AVG(load_average[1]) as avg_load_1,
    AVG(load_average[5]) as avg_load_5,
    AVG(load_average[15]) as avg_load_15,
    SUM(network_rx_bytes) as total_network_rx,
    SUM(network_tx_bytes) as total_network_tx
FROM node_metrics
GROUP BY node_id, bucket
WITH NO DATA;

-- Create refresh policy
SELECT add_continuous_aggregate_policy('node_metrics_5min',
    start_offset => INTERVAL '10 minutes',
    end_offset => INTERVAL '1 minute',
    schedule_interval => INTERVAL '1 minute'
);

-- =====================================================
-- RETENTION POLICIES
-- =====================================================

-- Drop raw data older than 30 days
SELECT add_retention_policy('vm_metrics', INTERVAL '30 days');
SELECT add_retention_policy('node_metrics', INTERVAL '30 days');

-- Keep 5-minute aggregates for 90 days
SELECT add_retention_policy('vm_metrics_5min', INTERVAL '90 days');
SELECT add_retention_policy('node_metrics_5min', INTERVAL '90 days');

-- Keep hourly aggregates for 1 year
SELECT add_retention_policy('vm_metrics_hourly_ts', INTERVAL '365 days');

-- Keep daily aggregates forever (no retention policy)

-- =====================================================
-- OPTIMIZED TIME-SERIES FUNCTIONS
-- =====================================================

-- Function to get latest metrics for all VMs (optimized)
CREATE OR REPLACE FUNCTION get_latest_vm_metrics()
RETURNS TABLE (
    vm_id UUID,
    cpu_usage DECIMAL(5,2),
    memory_percent DECIMAL(5,2),
    timestamp TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT ON (m.vm_id)
        m.vm_id,
        m.cpu_usage,
        m.memory_percent,
        m.timestamp
    FROM vm_metrics m
    WHERE m.timestamp > NOW() - INTERVAL '5 minutes'
    ORDER BY m.vm_id, m.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to get time-weighted averages
CREATE OR REPLACE FUNCTION time_weighted_average(
    p_vm_id UUID,
    p_metric TEXT,
    p_start TIMESTAMP WITH TIME ZONE,
    p_end TIMESTAMP WITH TIME ZONE
) RETURNS DECIMAL AS $$
DECLARE
    result DECIMAL;
BEGIN
    EXECUTE format('
        SELECT time_weight(''Linear'', %I, timestamp)
        FROM vm_metrics
        WHERE vm_id = $1 
        AND timestamp BETWEEN $2 AND $3',
        p_metric
    ) INTO result USING p_vm_id, p_start, p_end;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- REAL-TIME MONITORING VIEWS
-- =====================================================

-- Real-time VM status (last 5 minutes)
CREATE OR REPLACE VIEW realtime_vm_status AS
SELECT 
    v.id,
    v.name,
    v.state,
    COALESCE(lm.cpu_usage, 0) as current_cpu,
    COALESCE(lm.memory_percent, 0) as current_memory,
    CASE 
        WHEN v.state = 'running' AND lm.timestamp IS NULL THEN 'no_data'
        WHEN v.state = 'running' AND lm.timestamp < NOW() - INTERVAL '2 minutes' THEN 'stale'
        WHEN v.state = 'running' AND lm.cpu_usage > 90 THEN 'high_cpu'
        WHEN v.state = 'running' AND lm.memory_percent > 90 THEN 'high_memory'
        WHEN v.state = 'running' THEN 'healthy'
        ELSE v.state::text
    END as health_status,
    lm.timestamp as last_update
FROM vms v
LEFT JOIN LATERAL (
    SELECT cpu_usage, memory_percent, timestamp
    FROM vm_metrics
    WHERE vm_id = v.id
    ORDER BY timestamp DESC
    LIMIT 1
) lm ON true
WHERE v.state = 'running';

-- Alert conditions based on metrics
CREATE OR REPLACE VIEW metric_alerts AS
WITH recent_metrics AS (
    SELECT 
        vm_id,
        AVG(cpu_usage) as avg_cpu,
        MAX(cpu_usage) as max_cpu,
        AVG(memory_percent) as avg_memory,
        MAX(memory_percent) as max_memory
    FROM vm_metrics_5min
    WHERE bucket > NOW() - INTERVAL '15 minutes'
    GROUP BY vm_id
)
SELECT 
    v.id as vm_id,
    v.name as vm_name,
    CASE
        WHEN rm.avg_cpu > 80 THEN 'critical'
        WHEN rm.avg_cpu > 70 THEN 'warning'
        ELSE 'ok'
    END as cpu_alert_level,
    CASE
        WHEN rm.avg_memory > 90 THEN 'critical'
        WHEN rm.avg_memory > 80 THEN 'warning'
        ELSE 'ok'
    END as memory_alert_level,
    rm.avg_cpu,
    rm.max_cpu,
    rm.avg_memory,
    rm.max_memory
FROM vms v
JOIN recent_metrics rm ON v.id = rm.vm_id
WHERE rm.avg_cpu > 70 OR rm.avg_memory > 80;

-- =====================================================
-- INDEXES FOR TIMESCALEDB
-- =====================================================

-- Additional indexes for time-series queries
CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id_time 
    ON vm_metrics (vm_id, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_node_metrics_node_id_time 
    ON node_metrics (node_id, timestamp DESC);

-- Indexes on continuous aggregates
CREATE INDEX IF NOT EXISTS idx_vm_metrics_5min_vm_bucket 
    ON vm_metrics_5min (vm_id, bucket DESC);

CREATE INDEX IF NOT EXISTS idx_vm_metrics_hourly_vm_hour 
    ON vm_metrics_hourly_ts (vm_id, hour DESC);

-- =====================================================
-- PERFORMANCE MONITORING
-- =====================================================

-- View to monitor chunk sizes and compression
CREATE OR REPLACE VIEW chunk_compression_stats AS
SELECT 
    hypertable_name,
    chunk_name,
    pg_size_pretty(before_compression_total_bytes) as before_compression,
    pg_size_pretty(after_compression_total_bytes) as after_compression,
    compression_ratio
FROM timescaledb_information.compressed_chunk_stats
ORDER BY hypertable_name, chunk_name;

-- View to monitor continuous aggregate refresh lag
CREATE OR REPLACE VIEW continuous_aggregate_stats AS
SELECT 
    view_name,
    completed_threshold,
    invalidation_threshold,
    NOW() - completed_threshold as refresh_lag
FROM timescaledb_information.continuous_aggregate_stats
ORDER BY view_name;

COMMENT ON VIEW chunk_compression_stats IS 'Monitor TimescaleDB chunk compression effectiveness';
COMMENT ON VIEW continuous_aggregate_stats IS 'Monitor continuous aggregate refresh lag';