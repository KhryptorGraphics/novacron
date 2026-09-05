-- Migration: auth_tokens
-- Created: 2026-09-04
-- Direction: DOWN
-- Description: Drops the auth_tokens table (email verification + password
-- reset token storage; novacron-8ba).

DROP TABLE IF EXISTS auth_tokens;