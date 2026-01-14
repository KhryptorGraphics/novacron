-- NovaCron Performance Indexes
-- Version: 1.0.0
-- Description: Comprehensive index strategy for optimal query performance

-- User table indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_last_login_at ON users(last_login_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_locked_until ON users(locked_until) WHERE locked_until IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON users(created_at);

-- User sessions indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_token_hash ON user_sessions(session_token_hash);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_expires_at ON user_sessions(expires_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_ip_address ON user_sessions(ip_address);

-- User roles composite index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_roles_composite ON user_roles(user_id, role_id);

-- Compute nodes indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compute_nodes_status ON compute_nodes(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compute_nodes_hypervisor_type ON compute_nodes(hypervisor_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compute_nodes_last_heartbeat ON compute_nodes(last_heartbeat);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compute_nodes_load_average ON compute_nodes(load_average);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compute_nodes_vm_count ON compute_nodes(vm_count);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compute_nodes_tags ON compute_nodes USING GIN(tags);

-- VMs table indexes - critical for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_state ON vms(state);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_node_id ON vms(node_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_owner_id ON vms(owner_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_tenant_id ON vms(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_name ON vms(name);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_created_at ON vms(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_updated_at ON vms(updated_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_ip_address ON vms(ip_address) WHERE ip_address IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_template_id ON vms(template_id) WHERE template_id IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_scheduled_start ON vms(scheduled_start) WHERE scheduled_start IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_scheduled_stop ON vms(scheduled_stop) WHERE scheduled_stop IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_auto_start ON vms(auto_start) WHERE auto_start = true;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_tags ON vms USING GIN(tags);

-- Composite indexes for common query patterns
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_tenant_owner ON vms(tenant_id, owner_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_node_state ON vms(node_id, state);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_state_created ON vms(state, created_at);

-- VM metrics indexes - time-series optimized
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_vm_id_collected ON vm_metrics(vm_id, collected_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_collected_at ON vm_metrics(collected_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_node_id ON vm_metrics(node_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_cpu_usage ON vm_metrics(cpu_usage_percent) WHERE cpu_usage_percent IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_memory_usage ON vm_metrics(memory_usage_percent) WHERE memory_usage_percent IS NOT NULL;

-- Node metrics indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_metrics_node_collected ON node_metrics(node_id, collected_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_metrics_collected_at ON node_metrics(collected_at DESC);

-- VM snapshots indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_snapshots_vm_id ON vm_snapshots(vm_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_snapshots_created_at ON vm_snapshots(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_snapshots_created_by ON vm_snapshots(created_by);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_snapshots_parent ON vm_snapshots(parent_snapshot_id) WHERE parent_snapshot_id IS NOT NULL;

-- VM backups indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_backups_vm_id ON vm_backups(vm_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_backups_status ON vm_backups(backup_status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_backups_created_by ON vm_backups(created_by);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_backups_started_at ON vm_backups(started_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_backups_retention ON vm_backups(retention_until) WHERE retention_until IS NOT NULL;

-- Alerts indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_resource ON alerts(resource_type, resource_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_fired_at ON alerts(fired_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged_at) WHERE acknowledged_at IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_resolved ON alerts(resolved_at) WHERE resolved_at IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_rule_id ON alerts(alert_rule_id) WHERE alert_rule_id IS NOT NULL;

-- Alert rules indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_rules_enabled ON alert_rules(enabled) WHERE enabled = true;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_rules_severity ON alert_rules(severity);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alert_rules_created_by ON alert_rules(created_by);

-- Audit log indexes - critical for security and compliance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_session_id ON audit_log(session_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_action ON audit_log(action);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_resource ON audit_log(resource_type, resource_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_tenant_id ON audit_log(tenant_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_ip_address ON audit_log(ip_address);

-- Composite audit indexes for common security queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_user_action_time ON audit_log(user_id, action, created_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_log_resource_action_time ON audit_log(resource_type, resource_id, action, created_at DESC);

-- VM migrations indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_migrations_vm_id ON vm_migrations(vm_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_migrations_source_node ON vm_migrations(source_node_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_migrations_target_node ON vm_migrations(target_node_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_migrations_status ON vm_migrations(status);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_migrations_started_at ON vm_migrations(started_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_migrations_initiated_by ON vm_migrations(initiated_by);

-- System config index
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_system_config_category ON system_config(category);

-- Partial indexes for specific use cases
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_running_by_node ON vms(node_id) WHERE state = 'running';
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_error_state ON vms(id, updated_at) WHERE state = 'error';
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_firing ON alerts(fired_at DESC, severity) WHERE status = 'firing';
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_active ON user_sessions(user_id, last_activity) WHERE expires_at > NOW();

-- Expression indexes for complex queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_daily_avg ON vm_metrics(vm_id, date_trunc('day', collected_at));
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_node_metrics_hourly_avg ON node_metrics(node_id, date_trunc('hour', collected_at));

-- Full-text search indexes (if needed)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_name_trgm ON vms USING gin(name gin_trgm_ops);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_compute_nodes_name_trgm ON compute_nodes USING gin(name gin_trgm_ops);

-- Enable pg_trgm extension for fuzzy search if not already enabled
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Create function-based indexes for JSON queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_config_template ON vms((config->>'template')) WHERE config ? 'template';
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_alerts_labels_vm ON alerts USING gin((labels->>'vm')) WHERE labels ? 'vm';

-- Statistics targets for better query planning
ALTER TABLE vm_metrics ALTER COLUMN collected_at SET STATISTICS 1000;
ALTER TABLE audit_log ALTER COLUMN created_at SET STATISTICS 1000;
ALTER TABLE alerts ALTER COLUMN fired_at SET STATISTICS 1000;
ALTER TABLE vms ALTER COLUMN created_at SET STATISTICS 500;
ALTER TABLE vms ALTER COLUMN state SET STATISTICS 500;