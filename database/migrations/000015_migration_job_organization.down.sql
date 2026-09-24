-- Migration: migration_job_organization
-- Created: 2026-09-23
-- Direction: DOWN
-- Description: Remove migration-job organization ownership column.

ALTER TABLE migration_jobs DROP COLUMN IF EXISTS organization_id;