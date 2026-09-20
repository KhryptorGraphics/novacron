-- Migration: fabric_jobs
-- Created: 2026-09-20
-- Direction: UP
-- Description: Persisted fabric compute-job records (P2/G2). A fabric job is a
-- Process VM (clusterCreateSpec.Command) placed on a fabric node by the
-- bandwidth-aware placement cost; this table maps the job id to the VM that
-- runs it and the node it was placed on, so job status survives a restart.

CREATE TABLE fabric_jobs (
    id          TEXT PRIMARY KEY,
    vm_id       TEXT NOT NULL,
    node_id     TEXT NOT NULL,
    command     TEXT NOT NULL,
    status      TEXT NOT NULL,
    error       TEXT,
    placed_by   TEXT,
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
