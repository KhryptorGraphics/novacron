-- Initial schema migration
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Hypervisors table
CREATE TABLE IF NOT EXISTS hypervisors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hostname VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('KVM', 'VMware', 'Hyper-V', 'Xen', 'AWS', 'Azure', 'GCP')),
    capacity JSONB NOT NULL DEFAULT '{}',
    available JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'maintenance', 'error')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Virtual machines table
CREATE TABLE IF NOT EXISTS virtual_machines (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL CHECK (status IN ('running', 'stopped', 'suspended', 'migrating', 'error')),
    hypervisor_id UUID REFERENCES hypervisors(id) ON DELETE SET NULL,
    resources JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    ip_address INET,
    mac_address MACADDR,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Migrations table
CREATE TABLE IF NOT EXISTS vm_migrations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID REFERENCES virtual_machines(id) ON DELETE CASCADE,
    source_host UUID REFERENCES hypervisors(id),
    target_host UUID REFERENCES hypervisors(id),
    status VARCHAR(50) NOT NULL CHECK (status IN ('pending', 'preparing', 'transferring', 'completing', 'completed', 'failed', 'cancelled')),
    progress INTEGER DEFAULT 0 CHECK (progress >= 0 AND progress <= 100),
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Storage volumes table
CREATE TABLE IF NOT EXISTS storage_volumes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    size_gb INTEGER NOT NULL CHECK (size_gb > 0),
    type VARCHAR(50) NOT NULL CHECK (type IN ('ssd', 'hdd', 'network')),
    status VARCHAR(50) DEFAULT 'available' CHECK (status IN ('available', 'attached', 'error')),
    vm_id UUID REFERENCES virtual_machines(id) ON DELETE SET NULL,
    hypervisor_id UUID REFERENCES hypervisors(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Network interfaces table
CREATE TABLE IF NOT EXISTS network_interfaces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID REFERENCES virtual_machines(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    mac_address MACADDR,
    ip_address INET,
    vlan_id INTEGER,
    type VARCHAR(50) CHECK (type IN ('bridge', 'nat', 'host', 'internal')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Snapshots table
CREATE TABLE IF NOT EXISTS snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    vm_id UUID REFERENCES virtual_machines(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    size_bytes BIGINT,
    status VARCHAR(50) DEFAULT 'creating' CHECK (status IN ('creating', 'available', 'deleting', 'error')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Metrics table (for time-series data)
CREATE TABLE IF NOT EXISTS metrics (
    id BIGSERIAL PRIMARY KEY,
    resource_type VARCHAR(50) NOT NULL,
    resource_id UUID NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DOUBLE PRECISION NOT NULL,
    labels JSONB DEFAULT '{}',
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_hypervisors_status ON hypervisors(status);
CREATE INDEX idx_hypervisors_type ON hypervisors(type);

CREATE INDEX idx_vms_status ON virtual_machines(status);
CREATE INDEX idx_vms_hypervisor ON virtual_machines(hypervisor_id);
CREATE INDEX idx_vms_name ON virtual_machines(name);

CREATE INDEX idx_migrations_vm ON vm_migrations(vm_id);
CREATE INDEX idx_migrations_status ON vm_migrations(status);
CREATE INDEX idx_migrations_dates ON vm_migrations(started_at, completed_at);

CREATE INDEX idx_storage_vm ON storage_volumes(vm_id);
CREATE INDEX idx_storage_status ON storage_volumes(status);

CREATE INDEX idx_network_vm ON network_interfaces(vm_id);

CREATE INDEX idx_snapshots_vm ON snapshots(vm_id);
CREATE INDEX idx_snapshots_status ON snapshots(status);

CREATE INDEX idx_audit_user ON audit_logs(user_id);
CREATE INDEX idx_audit_resource ON audit_logs(resource_type, resource_id);
CREATE INDEX idx_audit_created ON audit_logs(created_at);

CREATE INDEX idx_metrics_resource ON metrics(resource_type, resource_id);
CREATE INDEX idx_metrics_name ON metrics(metric_name);
CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_hypervisors_updated_at BEFORE UPDATE ON hypervisors
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_vms_updated_at BEFORE UPDATE ON virtual_machines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_storage_updated_at BEFORE UPDATE ON storage_volumes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();