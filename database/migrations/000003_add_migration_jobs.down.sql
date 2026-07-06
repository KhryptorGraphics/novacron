-- Migration: add_migration_jobs
-- Created: 2026-07-05
-- Direction: DOWN
-- Description: Remove async VM migration job status persistence

DROP TABLE IF EXISTS migration_jobs;
