-- Migration: migration_job_organization
-- Created: 2026-09-23
-- Direction: UP
-- Description: Persist migration-job organization ownership so status remains
-- readable by the tenant after the source VM row is removed.

ALTER TABLE migration_jobs
    ADD COLUMN organization_id UUID;

UPDATE migration_jobs j
SET organization_id = v.organization_id
FROM vms v
WHERE v.id::text = j.vm_id
  AND j.organization_id IS NULL;
