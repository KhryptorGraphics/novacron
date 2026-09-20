-- Migration: fabric_jobs_name
-- Created: 2026-09-20
-- Direction: UP
-- Description: The fabric jobs API accepts a job name and echoes it in
-- responses, but fabric_jobs had no column for it, so a submitted name was
-- silently dropped (list/detail always returned ""). Persist it.

ALTER TABLE fabric_jobs ADD COLUMN IF NOT EXISTS name TEXT;
