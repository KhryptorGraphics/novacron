-- Migration: add_rbac_roles
-- Created: 2026-07-11
-- Direction: UP
-- Description: DB-backed RBAC role/permission catalogs (replaces the hardcoded
-- roleCatalog/permissionCatalog Go vars in backend/api/security/rbac_store.go).
-- Seeds the same 6 roles and 11 permissions the hardcoded catalogs shipped, so
-- existing authorization behavior is unchanged after migrating.

CREATE TABLE roles (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    permissions JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE permissions (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT ''
);

INSERT INTO roles (id, name, description, permissions) VALUES
    ('super-admin', 'Super Admin', 'Full platform administration', '["*"]'::jsonb),
    ('admin', 'Administrator', 'Administrative access to platform and security surfaces',
        '["read","write","delete","admin","audit.read","monitoring.read","rbac.manage","security.read","security.write","vm.console","vm.manage"]'::jsonb),
    ('operator', 'Operator', 'Operational access to consoles and runtime controls',
        '["read","monitoring.read","security.read","vm.console","vm.manage"]'::jsonb),
    ('viewer', 'Viewer', 'Read-only access to monitoring and security telemetry',
        '["read","monitoring.read","security.read"]'::jsonb),
    ('readonly', 'Read Only', 'Read-only access to platform data', '["read"]'::jsonb),
    ('user', 'User', 'Standard user access', '["read","write"]'::jsonb)
ON CONFLICT (id) DO NOTHING;

INSERT INTO permissions (id, name, description) VALUES
    ('read', 'Read', 'Read access to resources'),
    ('write', 'Write', 'Write access to resources'),
    ('delete', 'Delete', 'Delete access to resources'),
    ('admin', 'Admin', 'Administrative access'),
    ('audit.read', 'Audit Read', 'Read access to audit events and exports'),
    ('monitoring.read', 'Monitoring Read', 'Read access to monitoring and metrics streams'),
    ('rbac.manage', 'RBAC Manage', 'Manage role assignments'),
    ('security.read', 'Security Read', 'Read access to security dashboards and alerts'),
    ('security.write', 'Security Write', 'Start scans and mutate security settings'),
    ('vm.console', 'VM Console', 'Access VM console websocket sessions'),
    ('vm.manage', 'VM Manage', 'Manage VM lifecycle operations')
ON CONFLICT (id) DO NOTHING;
