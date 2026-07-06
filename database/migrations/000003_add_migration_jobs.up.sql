-- Migration: add_migration_jobs
-- Created: 2026-07-05
-- Direction: UP
-- Description: Persist async VM migration job status so it survives an api-server restart

-- Async migration jobs (POST /api/vms/{id}/migrate/async). id and vm_id are the
-- free-form path values used by the async handler, not UUIDs, so they are TEXT
-- and vm_id carries no FK to vms.
CREATE TABLE migration_jobs (
    id TEXT PRIMARY KEY,
    vm_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    finished_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
