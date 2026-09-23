DROP INDEX IF EXISTS fabric_jobs_organization_created_idx;
ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS organization_id;
