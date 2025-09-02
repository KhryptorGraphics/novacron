-- NovaCron Database Schema Initialization
-- Create tables for VM management, users, metrics, and federation

-- Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'user' CHECK (role IN ('admin', 'operator', 'user')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB DEFAULT '{}'
);

-- Virtual Machines table
CREATE TABLE IF NOT EXISTS virtual_machines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'stopped' CHECK (status IN ('running', 'stopped', 'paused', 'error', 'creating', 'migrating')),
    cpu_cores INTEGER NOT NULL DEFAULT 1,
    memory_mb INTEGER NOT NULL DEFAULT 1024,
    disk_gb INTEGER NOT NULL DEFAULT 20,
    os_type VARCHAR(50) NOT NULL,
    ip_address INET,
    host_node VARCHAR(100),
    owner_id UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT ARRAY[]::TEXT[]
);

-- VM Metrics table
CREATE TABLE IF NOT EXISTS vm_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID REFERENCES virtual_machines(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    disk_usage DECIMAL(5,2),
    network_in_bytes BIGINT DEFAULT 0,
    network_out_bytes BIGINT DEFAULT 0,
    disk_read_bytes BIGINT DEFAULT 0,
    disk_write_bytes BIGINT DEFAULT 0
);

-- Hypervisor Nodes table
CREATE TABLE IF NOT EXISTS hypervisor_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    hostname VARCHAR(255) NOT NULL,
    ip_address INET NOT NULL,
    status VARCHAR(20) DEFAULT 'offline' CHECK (status IN ('online', 'offline', 'maintenance', 'error')),
    cpu_cores INTEGER NOT NULL,
    memory_mb INTEGER NOT NULL,
    disk_gb INTEGER NOT NULL,
    hypervisor_type VARCHAR(50) NOT NULL DEFAULT 'kvm',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Federation Clusters table
CREATE TABLE IF NOT EXISTS federation_clusters (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    endpoint_url VARCHAR(255) NOT NULL,
    auth_token VARCHAR(255),
    status VARCHAR(20) DEFAULT 'disconnected' CHECK (status IN ('connected', 'disconnected', 'error')),
    last_sync TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Audit Logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Backup Jobs table
CREATE TABLE IF NOT EXISTS backup_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID REFERENCES virtual_machines(id) ON DELETE CASCADE,
    name VARCHAR(100) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    backup_type VARCHAR(20) DEFAULT 'full' CHECK (backup_type IN ('full', 'incremental', 'differential')),
    size_bytes BIGINT,
    storage_path VARCHAR(500),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_vms_owner ON virtual_machines(owner_id);
CREATE INDEX IF NOT EXISTS idx_vms_status ON virtual_machines(status);
CREATE INDEX IF NOT EXISTS idx_vms_host ON virtual_machines(host_node);
CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id ON vm_metrics(vm_id);
CREATE INDEX IF NOT EXISTS idx_vm_metrics_timestamp ON vm_metrics(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_backup_jobs_vm_id ON backup_jobs(vm_id);
CREATE INDEX IF NOT EXISTS idx_backup_jobs_status ON backup_jobs(status);

-- Insert demo data
INSERT INTO users (username, email, password_hash, role) VALUES 
    ('admin', 'admin@novacron.local', '$2b$10$rOlrKMULCXLYPdlG4F3p2uXhLJ9F7E5BVfQUMNc.N.yF8V0X3q8Ra', 'admin'),
    ('operator1', 'op1@novacron.local', '$2b$10$rOlrKMULCXLYPdlG4F3p2uXhLJ9F7E5BVfQUMNc.N.yF8V0X3q8Ra', 'operator'),
    ('user1', 'user1@novacron.local', '$2b$10$rOlrKMULCXLYPdlG4F3p2uXhLJ9F7E5BVfQUMNc.N.yF8V0X3q8Ra', 'user')
ON CONFLICT (email) DO NOTHING;

INSERT INTO hypervisor_nodes (name, hostname, ip_address, status, cpu_cores, memory_mb, disk_gb) VALUES 
    ('hypervisor-01', 'hv01.novacron.local', '192.168.1.101', 'online', 16, 65536, 2048),
    ('hypervisor-02', 'hv02.novacron.local', '192.168.1.102', 'online', 24, 131072, 4096),
    ('hypervisor-03', 'hv03.novacron.local', '192.168.1.103', 'maintenance', 32, 262144, 8192)
ON CONFLICT (name) DO NOTHING;

-- Create some demo VMs
DO $$
DECLARE
    user_id UUID;
BEGIN
    SELECT id INTO user_id FROM users WHERE username = 'user1' LIMIT 1;
    
    INSERT INTO virtual_machines (name, status, cpu_cores, memory_mb, disk_gb, os_type, ip_address, host_node, owner_id) VALUES
        ('web-server-1', 'running', 2, 2048, 40, 'Ubuntu 22.04 LTS', '192.168.1.201', 'hypervisor-01', user_id),
        ('web-server-2', 'stopped', 4, 4096, 60, 'Ubuntu 22.04 LTS', '192.168.1.202', 'hypervisor-02', user_id),
        ('web-server-3', 'paused', 2, 2048, 80, 'Ubuntu 22.04 LTS', '192.168.1.203', 'hypervisor-03', user_id);
END $$;

DO $$
DECLARE
    operator_id UUID;
BEGIN
    SELECT id INTO operator_id FROM users WHERE username = 'operator1' LIMIT 1;
    
    INSERT INTO virtual_machines (name, status, cpu_cores, memory_mb, disk_gb, os_type, ip_address, host_node, owner_id) VALUES
        ('db-server-1', 'running', 4, 8192, 100, 'CentOS 8', '192.168.1.211', 'hypervisor-01', operator_id),
        ('db-server-2', 'running', 4, 8192, 100, 'CentOS 8', '192.168.1.212', 'hypervisor-02', operator_id);
END $$;

-- Generate some metrics data
DO $$
DECLARE
    vm_record RECORD;
    i INTEGER;
BEGIN
    FOR vm_record IN SELECT id FROM virtual_machines LOOP
        FOR i IN 0..23 LOOP
            INSERT INTO vm_metrics (vm_id, timestamp, cpu_usage, memory_usage, disk_usage, network_in_bytes, network_out_bytes)
            VALUES (
                vm_record.id,
                NOW() - INTERVAL '1 hour' * i,
                RANDOM() * 100,
                RANDOM() * 100,
                RANDOM() * 100,
                (RANDOM() * 1000000000)::BIGINT,
                (RANDOM() * 1000000000)::BIGINT
            );
        END LOOP;
    END LOOP;
END $$;

-- Add some audit logs
DO $$
DECLARE
    u_record RECORD;
    vm_record RECORD;
    actions TEXT[] := ARRAY['vm_create', 'vm_start', 'vm_stop', 'vm_delete'];
    counter INTEGER := 0;
BEGIN
    FOR u_record IN SELECT id FROM users LOOP
        FOR vm_record IN SELECT id FROM virtual_machines LOOP
            EXIT WHEN counter >= 50;
            
            INSERT INTO audit_logs (user_id, action, resource_type, resource_id, details, ip_address)
            VALUES (
                u_record.id,
                actions[(RANDOM() * 4)::INTEGER + 1],
                'virtual_machine',
                vm_record.id,
                '{"source": "demo_data"}',
                ('192.168.1.' || (50 + (RANDOM() * 50)::INTEGER)::TEXT)::INET
            );
            
            counter := counter + 1;
        END LOOP;
    END LOOP;
END $$;