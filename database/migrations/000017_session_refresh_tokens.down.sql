-- Migration: session_refresh_tokens
-- Direction: DOWN
-- Reverts the refresh-token/revocation columns. `token`'s NOT NULL is
-- intentionally not restored: rows created after the UP migration hold a NULL
-- token and there is no honest value to backfill.

DROP INDEX IF EXISTS idx_sessions_expires_at;
DROP INDEX IF EXISTS idx_sessions_user_active;
DROP INDEX IF EXISTS idx_sessions_previous_refresh_token_hash;
DROP INDEX IF EXISTS idx_sessions_refresh_token_hash;
ALTER TABLE sessions DROP COLUMN IF EXISTS last_accessed_at;
ALTER TABLE sessions DROP COLUMN IF EXISTS revoked_at;
ALTER TABLE sessions DROP COLUMN IF EXISTS previous_refresh_token_hash;
ALTER TABLE sessions DROP COLUMN IF EXISTS refresh_token_hash;
