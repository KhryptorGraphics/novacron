-- Seed Nodes
-- Development and test compute nodes

INSERT INTO nodes (id, name, hostname, ip_address, port, status, cpu_cores, memory_mb, disk_gb, hypervisor_type, metadata, last_heartbeat) VALUES
    -- Production nodes
    ('n1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'prod-node-01', 'prod-node-01.novacron.local', '10.0.1.10', 9000, 'online', 32, 131072, 2000, 'kvm', '{"datacenter": "us-east-1", "rack": "A1", "gpu": false}', NOW()),
    ('n2eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'prod-node-02', 'prod-node-02.novacron.local', '10.0.1.11', 9000, 'online', 32, 131072, 2000, 'kvm', '{"datacenter": "us-east-1", "rack": "A2", "gpu": false}', NOW()),
    ('n3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'prod-node-03', 'prod-node-03.novacron.local', '10.0.1.12', 9000, 'online', 64, 262144, 4000, 'kvm', '{"datacenter": "us-east-1", "rack": "B1", "gpu": true, "gpu_model": "NVIDIA A100"}', NOW()),
    
    -- Development nodes
    ('n4eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'dev-node-01', 'dev-node-01.novacron.local', '10.0.2.10', 9000, 'online', 16, 65536, 1000, 'kvm', '{"datacenter": "us-west-1", "rack": "D1", "gpu": false}', NOW()),
    ('n5eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'dev-node-02', 'dev-node-02.novacron.local', '10.0.2.11', 9000, 'online', 16, 65536, 1000, 'containers', '{"datacenter": "us-west-1", "rack": "D2", "gpu": false}', NOW()),
    
    -- Test nodes with various states
    ('n6eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'test-node-01', 'test-node-01.novacron.local', '10.0.3.10', 9000, 'offline', 8, 32768, 500, 'kvm', '{"datacenter": "eu-central-1", "rack": "T1", "gpu": false}', NOW() - INTERVAL '1 hour'),
    ('n7eebc99-9c0b-4ef8-bb6d-6bb9bd380a07', 'maint-node-01', 'maint-node-01.novacron.local', '10.0.3.11', 9000, 'maintenance', 16, 65536, 1000, 'kvm', '{"datacenter": "eu-central-1", "rack": "T2", "gpu": false, "maintenance_reason": "Hardware upgrade"}', NOW() - INTERVAL '30 minutes'),
    
    -- Edge nodes
    ('n8eebc99-9c0b-4ef8-bb6d-6bb9bd380a08', 'edge-node-01', 'edge-node-01.novacron.local', '10.0.4.10', 9000, 'online', 4, 16384, 250, 'containers', '{"datacenter": "edge-location-1", "type": "edge", "gpu": false}', NOW()),
    ('n9eebc99-9c0b-4ef8-bb6d-6bb9bd380a09', 'edge-node-02', 'edge-node-02.novacron.local', '10.0.4.11', 9000, 'online', 4, 16384, 250, 'containers', '{"datacenter": "edge-location-2", "type": "edge", "gpu": false}', NOW())
ON CONFLICT (id) DO NOTHING;