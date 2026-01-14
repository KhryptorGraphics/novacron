-- Migration: add_resource_quotas
-- Created: 2025-01-30
-- Direction: UP
-- Description: Add resource quota management for organizations and users

-- Create resource quota table for organizations
CREATE TABLE organization_quotas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    max_vms INTEGER DEFAULT 100,
    max_cpu_cores INTEGER DEFAULT 1000,
    max_memory_gb INTEGER DEFAULT 4000,
    max_storage_gb INTEGER DEFAULT 10000,
    max_snapshots INTEGER DEFAULT 500,
    used_vms INTEGER DEFAULT 0,
    used_cpu_cores INTEGER DEFAULT 0,
    used_memory_gb INTEGER DEFAULT 0,
    used_storage_gb INTEGER DEFAULT 0,
    used_snapshots INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_org_quota UNIQUE(organization_id)
);

-- Create resource quota table for users
CREATE TABLE user_quotas (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    max_vms INTEGER DEFAULT 10,
    max_cpu_cores INTEGER DEFAULT 100,
    max_memory_gb INTEGER DEFAULT 400,
    max_storage_gb INTEGER DEFAULT 1000,
    max_snapshots INTEGER DEFAULT 50,
    used_vms INTEGER DEFAULT 0,
    used_cpu_cores INTEGER DEFAULT 0,
    used_memory_gb INTEGER DEFAULT 0,
    used_storage_gb INTEGER DEFAULT 0,
    used_snapshots INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_user_quota UNIQUE(user_id)
);

-- Add quota tracking columns to vms table
ALTER TABLE vms 
    ADD COLUMN IF NOT EXISTS resource_locked BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS quota_override JSONB;

-- Create indexes for quota tables
CREATE INDEX idx_org_quotas_organization ON organization_quotas(organization_id);
CREATE INDEX idx_user_quotas_user ON user_quotas(user_id);

-- Create triggers for updated_at
CREATE TRIGGER update_organization_quotas_updated_at BEFORE UPDATE ON organization_quotas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_quotas_updated_at BEFORE UPDATE ON user_quotas
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to check quota availability
CREATE OR REPLACE FUNCTION check_quota_available(
    p_user_id UUID,
    p_cpu_cores INTEGER,
    p_memory_gb INTEGER,
    p_storage_gb INTEGER
) RETURNS BOOLEAN AS $$
DECLARE
    v_user_quota RECORD;
    v_org_quota RECORD;
    v_organization_id UUID;
BEGIN
    -- Get user's organization
    SELECT organization_id INTO v_organization_id
    FROM users WHERE id = p_user_id;
    
    -- Check user quota
    SELECT * INTO v_user_quota
    FROM user_quotas WHERE user_id = p_user_id;
    
    IF v_user_quota IS NOT NULL THEN
        IF (v_user_quota.used_cpu_cores + p_cpu_cores) > v_user_quota.max_cpu_cores OR
           (v_user_quota.used_memory_gb + p_memory_gb) > v_user_quota.max_memory_gb OR
           (v_user_quota.used_storage_gb + p_storage_gb) > v_user_quota.max_storage_gb THEN
            RETURN FALSE;
        END IF;
    END IF;
    
    -- Check organization quota
    IF v_organization_id IS NOT NULL THEN
        SELECT * INTO v_org_quota
        FROM organization_quotas WHERE organization_id = v_organization_id;
        
        IF v_org_quota IS NOT NULL THEN
            IF (v_org_quota.used_cpu_cores + p_cpu_cores) > v_org_quota.max_cpu_cores OR
               (v_org_quota.used_memory_gb + p_memory_gb) > v_org_quota.max_memory_gb OR
               (v_org_quota.used_storage_gb + p_storage_gb) > v_org_quota.max_storage_gb THEN
                RETURN FALSE;
            END IF;
        END IF;
    END IF;
    
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- Insert default quotas for existing organizations
INSERT INTO organization_quotas (organization_id, max_vms, max_cpu_cores, max_memory_gb, max_storage_gb, max_snapshots)
SELECT id, 100, 1000, 4000, 10000, 500
FROM organizations
WHERE NOT EXISTS (
    SELECT 1 FROM organization_quotas WHERE organization_id = organizations.id
);

-- Insert default quotas for existing users
INSERT INTO user_quotas (user_id, max_vms, max_cpu_cores, max_memory_gb, max_storage_gb, max_snapshots)
SELECT id, 10, 100, 400, 1000, 50
FROM users
WHERE NOT EXISTS (
    SELECT 1 FROM user_quotas WHERE user_id = users.id
);

-- Update quota usage based on existing VMs
UPDATE user_quotas uq
SET 
    used_vms = COALESCE(vm_counts.count, 0),
    used_cpu_cores = COALESCE(vm_counts.total_cpu, 0),
    used_memory_gb = COALESCE(vm_counts.total_memory, 0),
    used_storage_gb = COALESCE(vm_counts.total_storage, 0)
FROM (
    SELECT 
        owner_id,
        COUNT(*) as count,
        SUM(cpu_cores) as total_cpu,
        SUM(memory_mb / 1024) as total_memory,
        SUM(disk_gb) as total_storage
    FROM vms
    WHERE state NOT IN ('error', 'stopped')
    GROUP BY owner_id
) vm_counts
WHERE uq.user_id = vm_counts.owner_id;

-- Update organization quota usage
UPDATE organization_quotas oq
SET 
    used_vms = COALESCE(vm_counts.count, 0),
    used_cpu_cores = COALESCE(vm_counts.total_cpu, 0),
    used_memory_gb = COALESCE(vm_counts.total_memory, 0),
    used_storage_gb = COALESCE(vm_counts.total_storage, 0)
FROM (
    SELECT 
        organization_id,
        COUNT(*) as count,
        SUM(cpu_cores) as total_cpu,
        SUM(memory_mb / 1024) as total_memory,
        SUM(disk_gb) as total_storage
    FROM vms
    WHERE state NOT IN ('error', 'stopped')
    GROUP BY organization_id
) vm_counts
WHERE oq.organization_id = vm_counts.organization_id;