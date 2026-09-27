-- Migration: session_refresh_tokens
-- Direction: UP
-- DB-backed refresh-token rotation and server-side session revocation for
-- api-server (novacron-czy). The access JWT carries a "sid" claim naming the
-- sessions row; requireAuth rejects it once revoked_at is set (logout,
-- refresh-token reuse) or the row is gone. Only the sha256 hex of the opaque
-- refresh token is stored. `token` was meant to hold the raw access token,
-- which must never be persisted (and does not fit VARCHAR(255) for a real
-- JWT); it becomes NULL-only.

ALTER TABLE sessions ALTER COLUMN token DROP NOT NULL;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS refresh_token_hash CHAR(64);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS previous_refresh_token_hash CHAR(64);
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS revoked_at TIMESTAMP WITH TIME ZONE;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMP WITH TIME ZONE;

CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_refresh_token_hash
    ON sessions(refresh_token_hash) WHERE refresh_token_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_previous_refresh_token_hash
    ON sessions(previous_refresh_token_hash) WHERE previous_refresh_token_hash IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_user_active
    ON sessions(user_id) WHERE revoked_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);

COMMENT ON COLUMN sessions.refresh_token_hash IS 'sha256 hex of the current opaque refresh token; rotated on every POST /api/auth/refresh.';
COMMENT ON COLUMN sessions.previous_refresh_token_hash IS 'sha256 hex of the immediately-prior refresh token, kept one generation so presenting it again (reuse of a rotated token) is detected and revokes the session.';
COMMENT ON COLUMN sessions.revoked_at IS 'Set on logout or refresh-token reuse detection; access tokens carrying this session id and its refresh token are rejected once non-null.';
COMMENT ON COLUMN sessions.last_accessed_at IS 'Updated on session creation, refresh rotation and GET /api/auth/me.';
COMMENT ON COLUMN sessions.expires_at IS 'Refresh-token expiry (7 days past the last rotation); rows past it are deleted by the api-server sweeper.';
