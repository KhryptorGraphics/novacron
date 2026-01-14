-- Auth tables migration
-- Adds roles, role_permissions, tenants tables

-- Roles table
CREATE TABLE IF NOT EXISTS roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    is_system BOOLEAN DEFAULT FALSE,
    tenant_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_by VARCHAR(255)
);

-- Role permissions table (stores permissions as JSON)
CREATE TABLE IF NOT EXISTS role_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
    resource VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    effect VARCHAR(20) DEFAULT 'allow' CHECK (effect IN ('allow', 'deny')),
    conditions JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(role_id, resource, action)
);

-- Tenants/Organizations extended table (if not exists)
-- Note: organizations table already exists in schema

-- Insert default system roles
INSERT INTO roles (id, name, description, is_system) VALUES
    ('00000000-0000-0000-0000-000000000001', 'admin', 'Full system administrator', TRUE),
    ('00000000-0000-0000-0000-000000000002', 'operator', 'System operator with limited admin rights', TRUE),
    ('00000000-0000-0000-0000-000000000003', 'user', 'Standard user', TRUE),
    ('00000000-0000-0000-0000-000000000004', 'viewer', 'Read-only access', TRUE)
ON CONFLICT (id) DO NOTHING;

-- Insert default admin permissions
INSERT INTO role_permissions (role_id, resource, action, effect) VALUES
    ('00000000-0000-0000-0000-000000000001', '*', '*', 'allow'),
    ('00000000-0000-0000-0000-000000000002', 'vms', '*', 'allow'),
    ('00000000-0000-0000-0000-000000000002', 'nodes', 'read', 'allow'),
    ('00000000-0000-0000-0000-000000000002', 'metrics', '*', 'allow'),
    ('00000000-0000-0000-0000-000000000003', 'vms', 'read', 'allow'),
    ('00000000-0000-0000-0000-000000000003', 'vms', 'create', 'allow'),
    ('00000000-0000-0000-0000-000000000003', 'vms', 'update', 'allow'),
    ('00000000-0000-0000-0000-000000000004', 'vms', 'read', 'allow'),
    ('00000000-0000-0000-0000-000000000004', 'metrics', 'read', 'allow')
ON CONFLICT (role_id, resource, action) DO NOTHING;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_roles_name ON roles(name);
CREATE INDEX IF NOT EXISTS idx_roles_tenant ON roles(tenant_id);
CREATE INDEX IF NOT EXISTS idx_roles_system ON roles(is_system) WHERE is_system = TRUE;
CREATE INDEX IF NOT EXISTS idx_role_permissions_role ON role_permissions(role_id);

-- Create updated_at trigger for roles
CREATE OR REPLACE FUNCTION update_roles_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS roles_updated_at_trigger ON roles;
CREATE TRIGGER roles_updated_at_trigger
    BEFORE UPDATE ON roles
    FOR EACH ROW
    EXECUTE FUNCTION update_roles_updated_at();
