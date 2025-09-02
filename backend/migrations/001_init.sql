-- NovaCron Database Initialization
-- Initial schema for production deployment

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

-- VMs table
CREATE TABLE IF NOT EXISTS vms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'stopped',
    cpu_cores INTEGER DEFAULT 1,
    memory_mb INTEGER DEFAULT 1024,
    disk_gb INTEGER DEFAULT 20,
    os_type VARCHAR(100),
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    stopped_at TIMESTAMP WITH TIME ZONE
);

-- VM Metrics table
CREATE TABLE IF NOT EXISTS vm_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID REFERENCES vms(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    cpu_usage_percent DECIMAL(5,2),
    memory_usage_mb INTEGER,
    disk_usage_gb DECIMAL(10,2),
    network_in_mb DECIMAL(10,2),
    network_out_mb DECIMAL(10,2),
    uptime_seconds BIGINT
);

-- Storage Volumes table
CREATE TABLE IF NOT EXISTS storage_volumes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    vm_id UUID REFERENCES vms(id) ON DELETE SET NULL,
    size_gb INTEGER NOT NULL,
    tier VARCHAR(20) DEFAULT 'hot',
    status VARCHAR(50) DEFAULT 'available',
    mount_point VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System Events table
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id UUID,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    message TEXT,
    severity VARCHAR(20) DEFAULT 'info',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Sessions table for authentication
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_agent TEXT,
    ip_address INET
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_vms_user_id ON vms(user_id);
CREATE INDEX IF NOT EXISTS idx_vms_status ON vms(status);
CREATE INDEX IF NOT EXISTS idx_vm_metrics_vm_id ON vm_metrics(vm_id);
CREATE INDEX IF NOT EXISTS idx_vm_metrics_timestamp ON vm_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_storage_volumes_user_id ON storage_volumes(user_id);
CREATE INDEX IF NOT EXISTS idx_system_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_created_at ON system_events(created_at);
CREATE INDEX IF NOT EXISTS idx_user_sessions_token_hash ON user_sessions(token_hash);
CREATE INDEX IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);

-- Insert sample data for testing
INSERT INTO users (email, username, password_hash, role) VALUES
('admin@novacron.com', 'admin', crypt('admin123', gen_salt('bf')), 'admin'),
('user@novacron.com', 'user', crypt('user123', gen_salt('bf')), 'user'),
('test@novacron.com', 'test', crypt('test123', gen_salt('bf')), 'user')
ON CONFLICT (email) DO NOTHING;

-- Insert sample VMs
INSERT INTO vms (name, user_id, status, cpu_cores, memory_mb, disk_gb, os_type) 
SELECT 
    'Demo VM ' || generate_series,
    (SELECT id FROM users WHERE username = 'admin'),
    CASE WHEN generate_series % 3 = 0 THEN 'running' ELSE 'stopped' END,
    CASE WHEN generate_series % 4 = 0 THEN 4 ELSE 2 END,
    CASE WHEN generate_series % 3 = 0 THEN 4096 ELSE 2048 END,
    CASE WHEN generate_series % 2 = 0 THEN 50 ELSE 20 END,
    CASE WHEN generate_series % 2 = 0 THEN 'Ubuntu 22.04' ELSE 'CentOS 8' END
FROM generate_series(1, 5)
ON CONFLICT DO NOTHING;

-- Insert sample metrics
INSERT INTO vm_metrics (vm_id, cpu_usage_percent, memory_usage_mb, disk_usage_gb, network_in_mb, network_out_mb, uptime_seconds)
SELECT 
    vm.id,
    RANDOM() * 100,
    RANDOM() * vm.memory_mb,
    RANDOM() * vm.disk_gb,
    RANDOM() * 10,
    RANDOM() * 5,
    EXTRACT(EPOCH FROM NOW() - vm.created_at)::BIGINT
FROM vms vm
WHERE vm.status = 'running';

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_vms_updated_at BEFORE UPDATE ON vms FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_storage_volumes_updated_at BEFORE UPDATE ON storage_volumes FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();