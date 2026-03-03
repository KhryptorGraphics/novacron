-- Migration: add_resource_quotas
-- Created: 2025-01-30
-- Direction: DOWN
-- Description: Remove resource quota management

-- Drop function
DROP FUNCTION IF EXISTS check_quota_available(UUID, INTEGER, INTEGER, INTEGER);

-- Drop triggers
DROP TRIGGER IF EXISTS update_user_quotas_updated_at ON user_quotas;
DROP TRIGGER IF EXISTS update_organization_quotas_updated_at ON organization_quotas;

-- Remove quota tracking columns from vms table
ALTER TABLE vms 
    DROP COLUMN IF EXISTS resource_locked,
    DROP COLUMN IF EXISTS quota_override;

-- Drop tables
DROP TABLE IF EXISTS user_quotas CASCADE;
DROP TABLE IF EXISTS organization_quotas CASCADE;