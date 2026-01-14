-- Rollback initial schema migration
DROP TRIGGER IF EXISTS update_storage_updated_at ON storage_volumes;
DROP TRIGGER IF EXISTS update_vms_updated_at ON virtual_machines;
DROP TRIGGER IF EXISTS update_hypervisors_updated_at ON hypervisors;

DROP FUNCTION IF EXISTS update_updated_at_column();

DROP TABLE IF EXISTS metrics CASCADE;
DROP TABLE IF EXISTS audit_logs CASCADE;
DROP TABLE IF EXISTS snapshots CASCADE;
DROP TABLE IF EXISTS network_interfaces CASCADE;
DROP TABLE IF EXISTS storage_volumes CASCADE;
DROP TABLE IF EXISTS vm_migrations CASCADE;
DROP TABLE IF EXISTS virtual_machines CASCADE;
DROP TABLE IF EXISTS hypervisors CASCADE;

DROP EXTENSION IF EXISTS "uuid-ossp";