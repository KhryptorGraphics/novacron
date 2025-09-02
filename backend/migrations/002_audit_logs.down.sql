-- Drop triggers
DROP TRIGGER IF EXISTS update_rotation_schedules_updated_at ON secret_rotation_schedules;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop tables in reverse order
DROP TABLE IF EXISTS secret_rotation_schedules;
DROP TABLE IF EXISTS secret_rotation_history;
DROP TABLE IF EXISTS audit_logs_archive;
DROP TABLE IF EXISTS audit_logs;