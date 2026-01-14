-- Audit logs table for security audit trail
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type VARCHAR(50) NOT NULL,
    actor VARCHAR(255) NOT NULL,
    resource VARCHAR(500),
    action VARCHAR(50) NOT NULL,
    result VARCHAR(50) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(100),
    details JSONB,
    error_msg TEXT,
    sensitivity VARCHAR(50) NOT NULL DEFAULT 'INTERNAL',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX idx_audit_logs_actor ON audit_logs(actor);
CREATE INDEX idx_audit_logs_event_type ON audit_logs(event_type);
CREATE INDEX idx_audit_logs_resource ON audit_logs(resource);
CREATE INDEX idx_audit_logs_result ON audit_logs(result);
CREATE INDEX idx_audit_logs_request_id ON audit_logs(request_id);
CREATE INDEX idx_audit_logs_sensitivity ON audit_logs(sensitivity);

-- Archive table for old audit logs
CREATE TABLE IF NOT EXISTS audit_logs_archive (
    id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    actor VARCHAR(255) NOT NULL,
    resource VARCHAR(500),
    action VARCHAR(50) NOT NULL,
    result VARCHAR(50) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(100),
    details JSONB,
    error_msg TEXT,
    sensitivity VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    archived_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for archive table
CREATE INDEX idx_audit_logs_archive_timestamp ON audit_logs_archive(timestamp DESC);
CREATE INDEX idx_audit_logs_archive_archived_at ON audit_logs_archive(archived_at DESC);

-- Secret rotation tracking table
CREATE TABLE IF NOT EXISTS secret_rotation_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    secret_key VARCHAR(500) NOT NULL,
    version VARCHAR(50) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    rotated_from VARCHAR(50),
    rotated_by VARCHAR(255),
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    metadata JSONB,
    UNIQUE(secret_key, version)
);

-- Indexes for rotation history
CREATE INDEX idx_rotation_history_secret_key ON secret_rotation_history(secret_key);
CREATE INDEX idx_rotation_history_status ON secret_rotation_history(status);
CREATE INDEX idx_rotation_history_expires_at ON secret_rotation_history(expires_at);

-- Rotation schedules table
CREATE TABLE IF NOT EXISTS secret_rotation_schedules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    secret_key VARCHAR(500) NOT NULL UNIQUE,
    next_rotation TIMESTAMPTZ NOT NULL,
    last_rotation TIMESTAMPTZ,
    policy JSONB NOT NULL,
    notification_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for rotation schedules
CREATE INDEX idx_rotation_schedules_next_rotation ON secret_rotation_schedules(next_rotation);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_rotation_schedules_updated_at 
    BEFORE UPDATE ON secret_rotation_schedules 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE audit_logs IS 'Security audit trail for all secret operations';
COMMENT ON TABLE audit_logs_archive IS 'Archive of old audit logs for compliance';
COMMENT ON TABLE secret_rotation_history IS 'History of all secret rotations';
COMMENT ON TABLE secret_rotation_schedules IS 'Scheduled rotation policies for secrets';