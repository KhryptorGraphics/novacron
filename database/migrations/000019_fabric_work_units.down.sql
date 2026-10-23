-- Migration: fabric_work_units
-- Created: 2026-10-23
-- Direction: DOWN
-- Description: Reverts the work-unit lifecycle columns, their value CHECKs and
-- the partial index. Columns that predate this migration are left untouched.

DROP INDEX IF EXISTS fabric_jobs_work_unit_created_idx;

ALTER TABLE fabric_jobs DROP CONSTRAINT IF EXISTS fabric_jobs_attempts_nonnegative;
ALTER TABLE fabric_jobs DROP CONSTRAINT IF EXISTS fabric_jobs_attempt_limit_positive;
ALTER TABLE fabric_jobs DROP CONSTRAINT IF EXISTS fabric_jobs_input_bytes_nonnegative;

ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS checkpoint_state;
ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS checkpoint_path;
ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS attempt_limit;
ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS attempts;
ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS input_bytes;
ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS artifact_hashes;
ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS work_unit;
