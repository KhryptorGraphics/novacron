-- Migration: add_rbac_roles
-- Created: 2026-07-11
-- Direction: DOWN
-- Description: Drop the DB-backed RBAC role/permission catalogs

DROP TABLE IF EXISTS permissions;
DROP TABLE IF EXISTS roles;
