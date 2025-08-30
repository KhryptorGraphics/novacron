-- Seed VMs
-- Development and test virtual machines

INSERT INTO vms (id, name, node_id, state, cpu_cores, memory_mb, disk_gb, os_type, os_version, owner_id, organization_id, network_config, metadata) VALUES
    -- Production VMs
    ('v1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'web-server-01', 'n1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'running', 4, 8192, 100, 'linux', 'Ubuntu 22.04', 'a1eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{"ip": "192.168.1.10", "gateway": "192.168.1.1", "dns": ["8.8.8.8", "8.8.4.4"]}', '{"purpose": "web server", "environment": "production"}'),
    ('v2eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'web-server-02', 'n2eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'running', 4, 8192, 100, 'linux', 'Ubuntu 22.04', 'a1eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{"ip": "192.168.1.11", "gateway": "192.168.1.1", "dns": ["8.8.8.8", "8.8.4.4"]}', '{"purpose": "web server", "environment": "production"}'),
    ('v3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'database-01', 'n3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'running', 8, 32768, 500, 'linux', 'Ubuntu 22.04', 'b1eebc99-9c0b-4ef8-bb6d-6bb9bd380a13', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{"ip": "192.168.1.20", "gateway": "192.168.1.1", "dns": ["8.8.8.8", "8.8.4.4"]}', '{"purpose": "database", "environment": "production", "database": "postgresql"}'),
    
    -- Development VMs
    ('v4eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'dev-app-01', 'n4eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'running', 2, 4096, 50, 'linux', 'Ubuntu 20.04', 'b2eebc99-9c0b-4ef8-bb6d-6bb9bd380a14', 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a12', '{"ip": "192.168.2.10", "gateway": "192.168.2.1", "dns": ["8.8.8.8"]}', '{"purpose": "development", "environment": "dev"}'),
    ('v5eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'dev-app-02', 'n4eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'stopped', 2, 4096, 50, 'linux', 'CentOS 8', 'b2eebc99-9c0b-4ef8-bb6d-6bb9bd380a14', 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a12', '{"ip": "192.168.2.11", "gateway": "192.168.2.1", "dns": ["8.8.8.8"]}', '{"purpose": "development", "environment": "dev"}'),
    
    -- Test VMs with various states
    ('v6eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'test-vm-01', 'n5eebc99-9c0b-4ef8-bb6d-6bb9bd380a05', 'paused', 1, 2048, 25, 'linux', 'Alpine Linux', 'c1eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', 'c0eebc99-9c0b-4ef8-bb6d-6bb9bd380a13', '{"ip": "192.168.3.10", "gateway": "192.168.3.1"}', '{"purpose": "testing", "environment": "test"}'),
    ('v7eebc99-9c0b-4ef8-bb6d-6bb9bd380a07', 'migration-test', 'n1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'migrating', 2, 4096, 50, 'windows', 'Windows Server 2019', 'c2eebc99-9c0b-4ef8-bb6d-6bb9bd380a16', 'd0eebc99-9c0b-4ef8-bb6d-6bb9bd380a14', '{"ip": "192.168.1.30", "gateway": "192.168.1.1"}', '{"purpose": "migration test", "environment": "test"}'),
    ('v8eebc99-9c0b-4ef8-bb6d-6bb9bd380a08', 'error-vm', 'n6eebc99-9c0b-4ef8-bb6d-6bb9bd380a06', 'error', 1, 1024, 20, 'linux', 'Debian 11', 'd1eebc99-9c0b-4ef8-bb6d-6bb9bd380a17', 'e0eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', '{"ip": "192.168.4.10"}', '{"purpose": "error testing", "error": "disk failure"}'),
    
    -- Edge VMs
    ('v9eebc99-9c0b-4ef8-bb6d-6bb9bd380a09', 'edge-app-01', 'n8eebc99-9c0b-4ef8-bb6d-6bb9bd380a08', 'running', 1, 1024, 10, 'linux', 'Alpine Linux', 'a1eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{"ip": "192.168.5.10"}', '{"purpose": "edge computing", "environment": "edge"}'),
    ('vaeebc99-9c0b-4ef8-bb6d-6bb9bd380a10', 'edge-app-02', 'n9eebc99-9c0b-4ef8-bb6d-6bb9bd380a09', 'running', 1, 1024, 10, 'linux', 'Alpine Linux', 'a1eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', '{"ip": "192.168.5.11"}', '{"purpose": "edge computing", "environment": "edge"}')
ON CONFLICT (id) DO NOTHING;

-- Create some snapshots
INSERT INTO snapshots (id, vm_id, name, description, size_bytes, state) VALUES
    ('s1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'v1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'web-server-01-backup', 'Weekly backup', 10737418240, 'available'),
    ('s2eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'v3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'database-01-before-upgrade', 'Before PostgreSQL upgrade', 53687091200, 'available'),
    ('s3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'v4eebc99-9c0b-4ef8-bb6d-6bb9bd380a04', 'dev-app-01-snapshot', 'Development snapshot', 5368709120, 'creating')
ON CONFLICT (id) DO NOTHING;

-- Create some storage volumes
INSERT INTO storage_volumes (id, name, node_id, vm_id, size_gb, used_gb, path, type, encrypted) VALUES
    ('sv1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'web-data', 'n1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'v1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 100, 45, '/storage/volumes/web-data', 'ssd', false),
    ('sv2eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'db-data', 'n3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'v3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 500, 250, '/storage/volumes/db-data', 'ssd', true),
    ('sv3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'backup-volume', 'n2eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', NULL, 1000, 600, '/storage/volumes/backups', 'hdd', true)
ON CONFLICT (id) DO NOTHING;

-- Create some network interfaces
INSERT INTO network_interfaces (id, vm_id, name, mac_address, ip_address, subnet_mask, gateway, vlan_id, bridge_name) VALUES
    ('ni1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'v1eebc99-9c0b-4ef8-bb6d-6bb9bd380a01', 'eth0', '52:54:00:12:34:56', '192.168.1.10', '255.255.255.0', '192.168.1.1', 100, 'br0'),
    ('ni2eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'v2eebc99-9c0b-4ef8-bb6d-6bb9bd380a02', 'eth0', '52:54:00:12:34:57', '192.168.1.11', '255.255.255.0', '192.168.1.1', 100, 'br0'),
    ('ni3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'v3eebc99-9c0b-4ef8-bb6d-6bb9bd380a03', 'eth0', '52:54:00:12:34:58', '192.168.1.20', '255.255.255.0', '192.168.1.1', 100, 'br0')
ON CONFLICT (id) DO NOTHING;