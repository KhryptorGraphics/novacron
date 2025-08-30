-- Migration: init_schema
-- Created: 2025-01-30
-- Direction: DOWN

-- Drop triggers
DROP TRIGGER IF EXISTS update_storage_volumes_updated_at ON storage_volumes;
DROP TRIGGER IF EXISTS update_migrations_updated_at ON migrations;
DROP TRIGGER IF EXISTS update_vms_updated_at ON vms;
DROP TRIGGER IF EXISTS update_nodes_updated_at ON nodes;
DROP TRIGGER IF EXISTS update_organizations_updated_at ON organizations;
DROP TRIGGER IF EXISTS update_users_updated_at ON users;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop tables (in reverse order of creation to handle foreign keys)
DROP TABLE IF EXISTS api_keys CASCADE;
DROP TABLE IF EXISTS jobs CASCADE;
DROP TABLE IF EXISTS snapshots CASCADE;
DROP TABLE IF EXISTS network_interfaces CASCADE;
DROP TABLE IF EXISTS storage_volumes CASCADE;
DROP TABLE IF EXISTS alerts CASCADE;
DROP TABLE IF EXISTS migrations CASCADE;
DROP TABLE IF EXISTS node_metrics CASCADE;
DROP TABLE IF EXISTS vm_metrics CASCADE;
DROP TABLE IF EXISTS vms CASCADE;
DROP TABLE IF EXISTS nodes CASCADE;
DROP TABLE IF EXISTS audit_logs CASCADE;
DROP TABLE IF EXISTS sessions CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS organizations CASCADE;

-- Drop enum types
DROP TYPE IF EXISTS resource_type;
DROP TYPE IF EXISTS alert_severity;
DROP TYPE IF EXISTS migration_status;
DROP TYPE IF EXISTS migration_type;
DROP TYPE IF EXISTS vm_state;
DROP TYPE IF EXISTS user_role;
DROP TYPE IF EXISTS user_status;