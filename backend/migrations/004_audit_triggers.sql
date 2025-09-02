-- NovaCron Audit and Security Triggers
-- Version: 1.0.0
-- Description: Comprehensive audit logging and security enforcement triggers

-- Enable audit logging function
CREATE OR REPLACE FUNCTION audit_trigger_function()
RETURNS TRIGGER AS $$
DECLARE
    audit_user_id UUID;
    audit_session_id UUID;
    audit_ip_address INET;
BEGIN
    -- Get current user context (set by application)
    audit_user_id := current_setting('audit.user_id', true)::UUID;
    audit_session_id := current_setting('audit.session_id', true)::UUID;
    audit_ip_address := current_setting('audit.ip_address', true)::INET;

    -- Insert audit record
    IF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (
            user_id, session_id, action, resource_type, resource_id, resource_name,
            ip_address, old_values, tenant_id, created_at
        ) VALUES (
            audit_user_id, audit_session_id, 'delete'::audit_action_type, TG_TABLE_NAME,
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN OLD.id::text
                WHEN TG_TABLE_NAME = 'vms' THEN OLD.id
                WHEN TG_TABLE_NAME = 'compute_nodes' THEN OLD.id
                ELSE OLD.id::text
            END,
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN OLD.username
                WHEN TG_TABLE_NAME = 'vms' THEN OLD.name
                WHEN TG_TABLE_NAME = 'compute_nodes' THEN OLD.name
                ELSE NULL
            END,
            audit_ip_address,
            row_to_json(OLD),
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN OLD.tenant_id
                WHEN TG_TABLE_NAME = 'vms' THEN OLD.tenant_id
                ELSE 'system'
            END,
            NOW()
        );
        RETURN OLD;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (
            user_id, session_id, action, resource_type, resource_id, resource_name,
            ip_address, old_values, new_values, tenant_id, created_at
        ) VALUES (
            audit_user_id, audit_session_id, 'update'::audit_action_type, TG_TABLE_NAME,
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN NEW.id::text
                WHEN TG_TABLE_NAME = 'vms' THEN NEW.id
                WHEN TG_TABLE_NAME = 'compute_nodes' THEN NEW.id
                ELSE NEW.id::text
            END,
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN NEW.username
                WHEN TG_TABLE_NAME = 'vms' THEN NEW.name
                WHEN TG_TABLE_NAME = 'compute_nodes' THEN NEW.name
                ELSE NULL
            END,
            audit_ip_address,
            row_to_json(OLD),
            row_to_json(NEW),
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN NEW.tenant_id
                WHEN TG_TABLE_NAME = 'vms' THEN NEW.tenant_id
                ELSE 'system'
            END,
            NOW()
        );
        RETURN NEW;
    ELSIF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (
            user_id, session_id, action, resource_type, resource_id, resource_name,
            ip_address, new_values, tenant_id, created_at
        ) VALUES (
            audit_user_id, audit_session_id, 'create'::audit_action_type, TG_TABLE_NAME,
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN NEW.id::text
                WHEN TG_TABLE_NAME = 'vms' THEN NEW.id
                WHEN TG_TABLE_NAME = 'compute_nodes' THEN NEW.id
                ELSE NEW.id::text
            END,
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN NEW.username
                WHEN TG_TABLE_NAME = 'vms' THEN NEW.name
                WHEN TG_TABLE_NAME = 'compute_nodes' THEN NEW.name
                ELSE NULL
            END,
            audit_ip_address,
            row_to_json(NEW),
            CASE 
                WHEN TG_TABLE_NAME = 'users' THEN NEW.tenant_id
                WHEN TG_TABLE_NAME = 'vms' THEN NEW.tenant_id
                ELSE 'system'
            END,
            NOW()
        );
        RETURN NEW;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Security trigger to update timestamps and enforce business rules
CREATE OR REPLACE FUNCTION security_trigger_function()
RETURNS TRIGGER AS $$
BEGIN
    -- Update timestamp for all tables
    NEW.updated_at = NOW();

    -- Specific security checks based on table
    IF TG_TABLE_NAME = 'users' THEN
        -- Prevent role escalation without proper authorization
        IF TG_OP = 'UPDATE' AND OLD.role != NEW.role THEN
            -- Check if the current user has admin privileges
            IF current_setting('audit.user_role', true) != 'admin' THEN
                RAISE EXCEPTION 'Insufficient privileges to change user role';
            END IF;
        END IF;

        -- Reset failed login attempts on successful password change
        IF TG_OP = 'UPDATE' AND OLD.password_hash != NEW.password_hash THEN
            NEW.failed_login_attempts = 0;
            NEW.locked_until = NULL;
            NEW.password_changed_at = NOW();
        END IF;

        -- Enforce password complexity (basic check - should be done in application)
        IF TG_OP = 'INSERT' OR (TG_OP = 'UPDATE' AND OLD.password_hash != NEW.password_hash) THEN
            IF LENGTH(NEW.password_hash) < 60 THEN
                RAISE EXCEPTION 'Password hash too short - must be properly hashed';
            END IF;
        END IF;

    ELSIF TG_TABLE_NAME = 'vms' THEN
        -- Validate resource constraints
        IF NEW.cpu_cores < 1 OR NEW.cpu_cores > 64 THEN
            RAISE EXCEPTION 'CPU cores must be between 1 and 64';
        END IF;

        IF NEW.memory_mb < 512 OR NEW.memory_mb > 1048576 THEN -- 512MB to 1TB
            RAISE EXCEPTION 'Memory must be between 512MB and 1TB';
        END IF;

        IF NEW.disk_size_gb < 1 OR NEW.disk_size_gb > 10240 THEN -- 1GB to 10TB
            RAISE EXCEPTION 'Disk size must be between 1GB and 10TB';
        END IF;

        -- Validate state transitions
        IF TG_OP = 'UPDATE' AND OLD.state != NEW.state THEN
            -- Check valid state transitions
            CASE OLD.state
                WHEN 'creating' THEN
                    IF NEW.state NOT IN ('running', 'error', 'stopped') THEN
                        RAISE EXCEPTION 'Invalid state transition from creating to %', NEW.state;
                    END IF;
                WHEN 'running' THEN
                    IF NEW.state NOT IN ('stopped', 'suspended', 'error', 'migrating') THEN
                        RAISE EXCEPTION 'Invalid state transition from running to %', NEW.state;
                    END IF;
                WHEN 'stopped' THEN
                    IF NEW.state NOT IN ('running', 'deleting', 'error') THEN
                        RAISE EXCEPTION 'Invalid state transition from stopped to %', NEW.state;
                    END IF;
                WHEN 'migrating' THEN
                    IF NEW.state NOT IN ('running', 'error') THEN
                        RAISE EXCEPTION 'Invalid state transition from migrating to %', NEW.state;
                    END IF;
            END CASE;
        END IF;

    ELSIF TG_TABLE_NAME = 'vm_migrations' THEN
        -- Validate migration progress
        IF NEW.progress_percent < 0 OR NEW.progress_percent > 100 THEN
            RAISE EXCEPTION 'Migration progress must be between 0 and 100';
        END IF;

        -- Auto-complete migration when progress reaches 100%
        IF TG_OP = 'UPDATE' AND NEW.progress_percent = 100 AND OLD.status = 'running' THEN
            NEW.status = 'completed';
            NEW.completed_at = NOW();
        END IF;

    ELSIF TG_TABLE_NAME = 'alerts' THEN
        -- Auto-set resolved timestamp when status changes to resolved
        IF TG_OP = 'UPDATE' AND OLD.status != 'resolved' AND NEW.status = 'resolved' THEN
            NEW.resolved_at = NOW();
            IF NEW.resolved_by IS NULL THEN
                NEW.resolved_by = current_setting('audit.user_id', true)::UUID;
            END IF;
        END IF;

        -- Auto-set acknowledged timestamp when status changes to acknowledged
        IF TG_OP = 'UPDATE' AND OLD.status != 'acknowledged' AND NEW.status = 'acknowledged' THEN
            NEW.acknowledged_at = NOW();
            IF NEW.acknowledged_by IS NULL THEN
                NEW.acknowledged_by = current_setting('audit.user_id', true)::UUID;
            END IF;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Function to automatically clean up old sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to automatically clean up old audit logs (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    retention_days INTEGER;
BEGIN
    -- Get retention policy from system config
    SELECT value::INTEGER INTO retention_days 
    FROM system_config 
    WHERE key = 'audit_log_retention_days'
    LIMIT 1;
    
    -- Default to 365 days if not configured
    IF retention_days IS NULL THEN
        retention_days := 365;
    END IF;
    
    DELETE FROM audit_log 
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to automatically clean up old metrics (retention policy)
CREATE OR REPLACE FUNCTION cleanup_old_metrics()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
    vm_metrics_retention_days INTEGER := 30;
    node_metrics_retention_days INTEGER := 30;
BEGIN
    -- Clean up old VM metrics
    DELETE FROM vm_metrics 
    WHERE collected_at < NOW() - (vm_metrics_retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean up old node metrics
    DELETE FROM node_metrics 
    WHERE collected_at < NOW() - (node_metrics_retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS deleted_count = deleted_count + ROW_COUNT;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for all critical tables
DROP TRIGGER IF EXISTS audit_users_trigger ON users;
CREATE TRIGGER audit_users_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_vms_trigger ON vms;
CREATE TRIGGER audit_vms_trigger
    AFTER INSERT OR UPDATE OR DELETE ON vms
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_compute_nodes_trigger ON compute_nodes;
CREATE TRIGGER audit_compute_nodes_trigger
    AFTER INSERT OR UPDATE OR DELETE ON compute_nodes
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_vm_snapshots_trigger ON vm_snapshots;
CREATE TRIGGER audit_vm_snapshots_trigger
    AFTER INSERT OR UPDATE OR DELETE ON vm_snapshots
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_vm_backups_trigger ON vm_backups;
CREATE TRIGGER audit_vm_backups_trigger
    AFTER INSERT OR UPDATE OR DELETE ON vm_backups
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_vm_migrations_trigger ON vm_migrations;
CREATE TRIGGER audit_vm_migrations_trigger
    AFTER INSERT OR UPDATE OR DELETE ON vm_migrations
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_alerts_trigger ON alerts;
CREATE TRIGGER audit_alerts_trigger
    AFTER INSERT OR UPDATE OR DELETE ON alerts
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

DROP TRIGGER IF EXISTS audit_user_roles_trigger ON user_roles;
CREATE TRIGGER audit_user_roles_trigger
    AFTER INSERT OR DELETE ON user_roles
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create security triggers for business rule enforcement
DROP TRIGGER IF EXISTS security_users_trigger ON users;
CREATE TRIGGER security_users_trigger
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION security_trigger_function();

DROP TRIGGER IF EXISTS security_vms_trigger ON vms;
CREATE TRIGGER security_vms_trigger
    BEFORE INSERT OR UPDATE ON vms
    FOR EACH ROW EXECUTE FUNCTION security_trigger_function();

DROP TRIGGER IF EXISTS security_vm_migrations_trigger ON vm_migrations;
CREATE TRIGGER security_vm_migrations_trigger
    BEFORE UPDATE ON vm_migrations
    FOR EACH ROW EXECUTE FUNCTION security_trigger_function();

DROP TRIGGER IF EXISTS security_alerts_trigger ON alerts;
CREATE TRIGGER security_alerts_trigger
    BEFORE UPDATE ON alerts
    FOR EACH ROW EXECUTE FUNCTION security_trigger_function();

-- Create trigger to update updated_at timestamp for other tables
CREATE OR REPLACE FUNCTION update_timestamp_trigger()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply timestamp triggers to tables that need them
DROP TRIGGER IF EXISTS update_timestamp_compute_nodes ON compute_nodes;
CREATE TRIGGER update_timestamp_compute_nodes
    BEFORE UPDATE ON compute_nodes
    FOR EACH ROW EXECUTE FUNCTION update_timestamp_trigger();

DROP TRIGGER IF EXISTS update_timestamp_alert_rules ON alert_rules;
CREATE TRIGGER update_timestamp_alert_rules
    BEFORE UPDATE ON alert_rules
    FOR EACH ROW EXECUTE FUNCTION update_timestamp_trigger();

DROP TRIGGER IF EXISTS update_timestamp_system_config ON system_config;
CREATE TRIGGER update_timestamp_system_config
    BEFORE UPDATE ON system_config
    FOR EACH ROW EXECUTE FUNCTION update_timestamp_trigger();

DROP TRIGGER IF EXISTS update_timestamp_roles ON roles;
CREATE TRIGGER update_timestamp_roles
    BEFORE UPDATE ON roles
    FOR EACH ROW EXECUTE FUNCTION update_timestamp_trigger();

-- Add system configuration for audit log retention
INSERT INTO system_config (key, value, value_type, description, category) VALUES
    ('audit_log_retention_days', '365', 'integer', 'Number of days to retain audit logs', 'security'),
    ('vm_metrics_retention_days', '30', 'integer', 'Number of days to retain VM metrics', 'monitoring'),
    ('node_metrics_retention_days', '30', 'integer', 'Number of days to retain node metrics', 'monitoring'),
    ('session_cleanup_interval_hours', '1', 'integer', 'Interval in hours for session cleanup', 'security')
ON CONFLICT (key) DO NOTHING;

-- Create function to get current audit context (used by applications)
CREATE OR REPLACE FUNCTION set_audit_context(
    p_user_id UUID,
    p_session_id UUID DEFAULT NULL,
    p_ip_address INET DEFAULT NULL,
    p_user_role TEXT DEFAULT NULL
)
RETURNS void AS $$
BEGIN
    PERFORM set_config('audit.user_id', p_user_id::text, true);
    PERFORM set_config('audit.session_id', COALESCE(p_session_id::text, ''), true);
    PERFORM set_config('audit.ip_address', COALESCE(p_ip_address::text, ''), true);
    PERFORM set_config('audit.user_role', COALESCE(p_user_role, ''), true);
END;
$$ LANGUAGE plpgsql;

-- Create function to clear audit context
CREATE OR REPLACE FUNCTION clear_audit_context()
RETURNS void AS $$
BEGIN
    PERFORM set_config('audit.user_id', '', true);
    PERFORM set_config('audit.session_id', '', true);
    PERFORM set_config('audit.ip_address', '', true);
    PERFORM set_config('audit.user_role', '', true);
END;
$$ LANGUAGE plpgsql;