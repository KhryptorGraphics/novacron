-- Migration: fabric_jobs_name
-- Created: 2026-09-20
-- Direction: DOWN

ALTER TABLE fabric_jobs DROP COLUMN IF EXISTS name;
