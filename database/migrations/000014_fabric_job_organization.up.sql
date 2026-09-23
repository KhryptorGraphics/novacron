-- Migration: fabric_job_organization
-- Created: 2026-09-23
-- Direction: UP
-- Description: Persist fabric-job organization ownership independently of the
-- executor VM, which may run on a peer with a separate database.

ALTER TABLE fabric_jobs
    ADD COLUMN organization_id UUID;

UPDATE fabric_jobs j
SET organization_id = v.organization_id
FROM vms v
WHERE v.id::text = j.vm_id
  AND j.organization_id IS NULL;

CREATE INDEX fabric_jobs_organization_created_idx
    ON fabric_jobs (organization_id, created_at DESC);
