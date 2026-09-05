-- Migration: auth_tokens
-- Created: 2026-09-04
-- Direction: UP
-- Description: Single-use auth token storage for email verification and
-- password reset flows (novacron-8ba). Only the sha256 hex of the raw token
-- is persisted; the raw token is delivered exclusively by email.

CREATE TABLE auth_tokens (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash CHAR(64) UNIQUE NOT NULL,
    purpose    VARCHAR(32) NOT NULL CHECK (purpose IN ('email_verification','password_reset')),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    used_at    TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_auth_tokens_user_purpose ON auth_tokens(user_id, purpose);