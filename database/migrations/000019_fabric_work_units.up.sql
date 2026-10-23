-- Migration: fabric_work_units
-- Created: 2026-10-23
-- Direction: UP
-- Description: Fabric work-unit lifecycle. A work unit is a Process-driver VM
-- job with content-addressed input artifacts and a checkpoint/retry budget:
-- `artifact_hashes` is the JSONB list of input artifact digests, `input_bytes`
-- the total input size, `attempts`/`attempt_limit` the retry budget, and
-- `checkpoint_path`/`checkpoint_state` the executor's resume point
-- ('none' | 'available' | 'consumed'). Jobs that are not work units keep
-- work_unit = FALSE and the defaults below, so existing rows stay valid.

ALTER TABLE fabric_jobs
    ADD COLUMN IF NOT EXISTS work_unit BOOLEAN NOT NULL DEFAULT FALSE;

ALTER TABLE fabric_jobs
    ADD COLUMN IF NOT EXISTS artifact_hashes JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE fabric_jobs
    ADD COLUMN IF NOT EXISTS input_bytes BIGINT NOT NULL DEFAULT 0;

ALTER TABLE fabric_jobs
    ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0;

ALTER TABLE fabric_jobs
    ADD COLUMN IF NOT EXISTS attempt_limit INTEGER NOT NULL DEFAULT 1;

ALTER TABLE fabric_jobs
    ADD COLUMN IF NOT EXISTS checkpoint_path TEXT;

ALTER TABLE fabric_jobs
    ADD COLUMN IF NOT EXISTS checkpoint_state TEXT;

ALTER TABLE fabric_jobs DROP CONSTRAINT IF EXISTS fabric_jobs_attempts_nonnegative;
ALTER TABLE fabric_jobs
    ADD CONSTRAINT fabric_jobs_attempts_nonnegative CHECK (attempts >= 0);

ALTER TABLE fabric_jobs DROP CONSTRAINT IF EXISTS fabric_jobs_attempt_limit_positive;
ALTER TABLE fabric_jobs
    ADD CONSTRAINT fabric_jobs_attempt_limit_positive CHECK (attempt_limit >= 1);

ALTER TABLE fabric_jobs DROP CONSTRAINT IF EXISTS fabric_jobs_input_bytes_nonnegative;
ALTER TABLE fabric_jobs
    ADD CONSTRAINT fabric_jobs_input_bytes_nonnegative CHECK (input_bytes >= 0);

-- Only work-unit jobs are ever scanned; the partial predicate keeps the index
-- off the ordinary fabric-job workload.
CREATE INDEX IF NOT EXISTS fabric_jobs_work_unit_created_idx
    ON fabric_jobs (work_unit, created_at DESC)
    WHERE work_unit;
