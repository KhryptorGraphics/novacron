-- Seed Metrics
-- Sample metrics data for development and testing

-- Generate VM metrics for the last 24 hours
INSERT INTO vm_metrics (vm_id, cpu_usage, memory_usage, memory_percent, disk_read_bytes, disk_write_bytes, network_rx_bytes, network_tx_bytes, timestamp)
SELECT
    vm.id,
    20 + (RANDOM() * 60)::NUMERIC(5,2),  -- CPU usage between 20-80%
    (vm.memory_mb * 0.3 + RANDOM() * vm.memory_mb * 0.5)::BIGINT,  -- Memory usage 30-80% of allocated
    30 + (RANDOM() * 50)::NUMERIC(5,2),  -- Memory percent between 30-80%
    (RANDOM() * 1000000)::BIGINT,  -- Disk read bytes
    (RANDOM() * 500000)::BIGINT,   -- Disk write bytes
    (RANDOM() * 10000000)::BIGINT, -- Network RX bytes
    (RANDOM() * 5000000)::BIGINT,  -- Network TX bytes
    NOW() - (interval '1 hour' * generate_series(0, 23))
FROM vms vm
WHERE vm.state = 'running'
ORDER BY vm.id, timestamp;

-- Generate more granular metrics for the last hour (every 5 minutes)
INSERT INTO vm_metrics (vm_id, cpu_usage, memory_usage, memory_percent, disk_read_bytes, disk_write_bytes, network_rx_bytes, network_tx_bytes, timestamp)
SELECT
    vm.id,
    30 + (RANDOM() * 40)::NUMERIC(5,2),  -- CPU usage between 30-70%
    (vm.memory_mb * 0.4 + RANDOM() * vm.memory_mb * 0.3)::BIGINT,  -- Memory usage 40-70% of allocated
    40 + (RANDOM() * 30)::NUMERIC(5,2),  -- Memory percent between 40-70%
    (RANDOM() * 500000)::BIGINT,   -- Disk read bytes
    (RANDOM() * 250000)::BIGINT,   -- Disk write bytes
    (RANDOM() * 5000000)::BIGINT,  -- Network RX bytes
    (RANDOM() * 2500000)::BIGINT,  -- Network TX bytes
    NOW() - (interval '5 minutes' * generate_series(0, 11))
FROM vms vm
WHERE vm.state = 'running' AND vm.name LIKE 'web-server%'
ORDER BY vm.id, timestamp;

-- Generate node metrics for the last 24 hours
INSERT INTO node_metrics (node_id, cpu_usage, memory_usage, memory_percent, disk_usage, disk_percent, network_rx_bytes, network_tx_bytes, load_average, timestamp)
SELECT
    n.id,
    15 + (RANDOM() * 50)::NUMERIC(5,2),  -- CPU usage between 15-65%
    (n.memory_mb * 0.2 + RANDOM() * n.memory_mb * 0.4)::BIGINT,  -- Memory usage 20-60% of total
    20 + (RANDOM() * 40)::NUMERIC(5,2),  -- Memory percent between 20-60%
    (n.disk_gb * 0.3 + RANDOM() * n.disk_gb * 0.3)::BIGINT,  -- Disk usage 30-60% of total
    30 + (RANDOM() * 30)::NUMERIC(5,2),  -- Disk percent between 30-60%
    (RANDOM() * 100000000)::BIGINT,  -- Network RX bytes
    (RANDOM() * 50000000)::BIGINT,   -- Network TX bytes
    ARRAY[
        (RANDOM() * 4)::NUMERIC(5,2),
        (RANDOM() * 3)::NUMERIC(5,2),
        (RANDOM() * 2)::NUMERIC(5,2)
    ],  -- Load average [1min, 5min, 15min]
    NOW() - (interval '1 hour' * generate_series(0, 23))
FROM nodes n
WHERE n.status = 'online'
ORDER BY n.id, timestamp;

-- Generate some high-load metrics for alert testing
INSERT INTO node_metrics (node_id, cpu_usage, memory_usage, memory_percent, disk_usage, disk_percent, network_rx_bytes, network_tx_bytes, load_average, timestamp)
VALUES
    ('n1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 85.5, 120000, 91.5, 1800, 90.0, 900000000, 450000000, ARRAY[8.5, 7.2, 6.8], NOW() - INTERVAL '10 minutes'),
    ('n1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 92.3, 125000, 95.3, 1900, 95.0, 950000000, 475000000, ARRAY[9.2, 8.5, 7.5], NOW() - INTERVAL '5 minutes'),
    ('n3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 88.7, 245000, 93.5, 3700, 92.5, 800000000, 400000000, ARRAY[12.5, 10.2, 9.8], NOW() - INTERVAL '15 minutes');