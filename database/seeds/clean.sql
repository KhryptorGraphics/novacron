-- Clean seed data
-- Removes all seed data while preserving schema

-- Delete in reverse order of foreign key dependencies

-- Metrics
DELETE FROM vm_metrics WHERE vm_id IN (
    SELECT id FROM vms WHERE id LIKE '_eebc99-%'
);
DELETE FROM node_metrics WHERE node_id IN (
    SELECT id FROM nodes WHERE id LIKE '_eebc99-%'
);

-- VM-related data
DELETE FROM network_interfaces WHERE vm_id IN (
    SELECT id FROM vms WHERE id LIKE '_eebc99-%'
);
DELETE FROM snapshots WHERE vm_id IN (
    SELECT id FROM vms WHERE id LIKE '_eebc99-%'
);
DELETE FROM storage_volumes WHERE id LIKE 'sv_eebc99-%';
DELETE FROM migrations WHERE vm_id IN (
    SELECT id FROM vms WHERE id LIKE '_eebc99-%'
);
DELETE FROM alerts WHERE id LIKE '_eebc99-%';

-- VMs and nodes
DELETE FROM vms WHERE id LIKE '_eebc99-%';
DELETE FROM nodes WHERE id LIKE '_eebc99-%';

-- User-related data
DELETE FROM api_keys WHERE user_id IN (
    SELECT id FROM users WHERE id LIKE '_eebc99-%' AND username != 'admin'
);
DELETE FROM sessions WHERE user_id IN (
    SELECT id FROM users WHERE id LIKE '_eebc99-%' AND username != 'admin'
);
DELETE FROM audit_logs WHERE user_id IN (
    SELECT id FROM users WHERE id LIKE '_eebc99-%' AND username != 'admin'
);
DELETE FROM jobs WHERE created_by IN (
    SELECT id FROM users WHERE id LIKE '_eebc99-%' AND username != 'admin'
);

-- Users and organizations
DELETE FROM users WHERE id LIKE '_eebc99-%' AND username != 'admin';
DELETE FROM organizations WHERE id LIKE '_eebc99-%' AND slug != 'novacron';

-- Reset sequences if needed (PostgreSQL specific)
-- This ensures new IDs start from appropriate values
SELECT setval(pg_get_serial_sequence('audit_logs', 'id'), COALESCE(MAX(id), 1)) FROM audit_logs;
SELECT setval(pg_get_serial_sequence('jobs', 'id'), COALESCE(MAX(id), 1)) FROM jobs;