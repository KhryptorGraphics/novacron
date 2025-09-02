# NovaCron Secrets Management Security Guide

## Overview

NovaCron implements a comprehensive secrets management system with audit logging, automatic rotation, and multi-provider support. This guide covers configuration, operation, and security best practices.

## Architecture

### Components

1. **Secrets Manager** (`secrets_manager_enhanced.go`)
   - Multi-provider abstraction (Vault, AWS, Environment)
   - Caching layer with TTL
   - Audit integration
   - Configuration-driven setup

2. **Audit Logger** (`audit_logger.go`)
   - Complete audit trail for all secret operations
   - Database persistence
   - Security alerting
   - Compliance reporting

3. **Rotation Manager** (`secret_rotation.go`)
   - Automatic and manual rotation
   - Version tracking
   - Notification system
   - Policy-based scheduling

## Configuration

### Environment Variables

**REQUIRED** (No defaults for security):
- `VAULT_ADDR`: Vault server address (e.g., `https://vault.example.com:8200`)
- `SECRETS_PROVIDER`: Provider selection (`vault`, `aws`, `env`)

**Optional with Defaults**:
```bash
# Cache Configuration
SECRETS_CACHE_ENABLED=true
SECRETS_CACHE_TTL=300  # seconds
SECRETS_CACHE_MAX_ENTRIES=1000

# Audit Configuration
AUDIT_ENABLED=true
AUDIT_STORAGE_TYPE=database  # database, file, syslog

# Rotation Configuration
ROTATION_ENABLED=true
ROTATION_MAX_AGE_HOURS=2160  # 90 days
ROTATION_INTERVAL_HOURS=1440  # 60 days
ROTATION_NOTIFY_HOURS=168  # 7 days before rotation
ROTATION_AUTO_ROTATE=false
ROTATION_REQUIRE_APPROVAL=true
```

### Configuration File

Location: `/etc/novacron/secrets.yaml` or `$SECRETS_CONFIG_PATH`

```yaml
secrets:
  provider: ${SECRETS_PROVIDER:vault}
  
  vault:
    address: ${VAULT_ADDR}  # REQUIRED - no default
    token: ${VAULT_TOKEN}
    path_prefix: ${VAULT_PATH_PREFIX:secret/data/novacron}
    
    tls:
      enabled: true
      ca_cert: /path/to/ca.pem
      
    auth:
      method: approle  # token, approle, kubernetes
      approle:
        role_id: ${VAULT_ROLE_ID}
        secret_id: ${VAULT_SECRET_ID}

audit:
  enabled: true
  storage:
    type: database
    
  alerting:
    events:
      - SECRET_MODIFY
      - SECRET_DELETE
      - AUTH_FAILURE
    
rotation:
  enabled: true
  default_policy:
    max_age_hours: 2160
    rotation_interval_hours: 1440
    auto_rotate: false
```

## Security Operations

### 1. Initial Setup

```bash
# 1. Set required environment variables
export VAULT_ADDR="https://vault.example.com:8200"
export VAULT_TOKEN="your-token"
export SECRETS_PROVIDER="vault"

# 2. Initialize database migrations
./novacron migrate up

# 3. Configure rotation policies
./novacron secrets policy set --key jwt_secret --max-age 30d --interval 20d

# 4. Enable audit logging
export AUDIT_ENABLED=true
export AUDIT_STORAGE_TYPE=database
```

### 2. Secret Rotation Procedures

#### Manual Rotation

```bash
# Rotate a specific secret
./novacron secrets rotate --key database_password --approve

# View rotation history
./novacron secrets history --key database_password

# View pending rotations
./novacron secrets pending
```

#### Automatic Rotation Setup

```yaml
# In secrets.yaml
rotation:
  policies:
    jwt_secret:
      max_age_hours: 720  # 30 days
      auto_rotate: true
      require_approval: false
```

### 3. Audit Log Management

#### Viewing Audit Logs

```sql
-- Recent secret access
SELECT timestamp, actor, resource, action, result, ip_address
FROM audit_logs
WHERE event_type = 'SECRET_ACCESS'
  AND timestamp > NOW() - INTERVAL '24 hours'
ORDER BY timestamp DESC;

-- Failed access attempts
SELECT timestamp, actor, resource, error_msg
FROM audit_logs
WHERE result = 'FAILURE'
ORDER BY timestamp DESC;

-- Secret modifications
SELECT timestamp, actor, resource, details
FROM audit_logs
WHERE event_type IN ('SECRET_MODIFY', 'SECRET_DELETE')
ORDER BY timestamp DESC;
```

#### Archiving Old Logs

```sql
-- Archive logs older than 90 days (automated)
INSERT INTO audit_logs_archive
SELECT * FROM audit_logs WHERE timestamp < NOW() - INTERVAL '90 days';

DELETE FROM audit_logs WHERE timestamp < NOW() - INTERVAL '90 days';
```

### 4. Emergency Procedures

#### Secret Compromise Response

1. **Immediate Rotation**:
```bash
./novacron secrets rotate --key compromised_key --force --no-approval
```

2. **Audit Investigation**:
```sql
SELECT * FROM audit_logs
WHERE resource = 'compromised_key'
  AND timestamp > NOW() - INTERVAL '7 days'
ORDER BY timestamp;
```

3. **Revoke Old Versions**:
```bash
./novacron secrets revoke --key compromised_key --version v1234567
```

#### System Recovery

1. **Restore from Backup**:
```bash
./novacron secrets restore --backup-id backup-20250901
```

2. **Verify Integrity**:
```bash
./novacron secrets verify --all
```

## Security Best Practices

### 1. Provider Configuration

- **Never hardcode Vault address** - Always use environment variables
- **Use TLS for all connections** - Enable TLS in configuration
- **Implement least privilege** - Use AppRole or Kubernetes auth, not root tokens
- **Rotate authentication tokens** - Set token TTL and renewal policies

### 2. Rotation Policies

- **JWT Secrets**: Rotate every 30 days
- **Database Passwords**: Rotate every 90 days
- **API Keys**: Rotate annually or on compromise
- **Encryption Keys**: Rotate every 180 days with versioning

### 3. Audit Requirements

- **Enable audit logging in production** - Required for compliance
- **Monitor failed access attempts** - Set up alerting
- **Archive logs regularly** - Maintain 90-day hot storage
- **Protect audit logs** - Use separate database/permissions

### 4. Access Control

```yaml
# Example RBAC configuration
roles:
  developer:
    secrets:
      - read: "database_url"
      - read: "api_keys/*"
    
  operator:
    secrets:
      - read: "*"
      - write: "api_keys/*"
      - rotate: "*"
    
  security_admin:
    secrets:
      - "*": "*"
    audit:
      - read: "*"
```

## Monitoring and Alerting

### Key Metrics

1. **Secret Access Rate**: Monitor for anomalies
2. **Rotation Success Rate**: Should be 100%
3. **Cache Hit Rate**: Optimize TTL if low
4. **Audit Log Volume**: Watch for spikes

### Alert Conditions

```yaml
alerts:
  - name: high_failure_rate
    condition: failure_rate > 0.1
    severity: critical
    
  - name: rotation_overdue
    condition: days_until_rotation < 0
    severity: warning
    
  - name: unauthorized_access
    condition: result = "DENIED"
    severity: high
```

## Compliance

### SOC2 Requirements

- ✅ Audit logging for all secret access
- ✅ Automatic rotation policies
- ✅ Encryption at rest and in transit
- ✅ Access control and authentication
- ✅ Monitoring and alerting

### HIPAA Requirements

- ✅ PHI encryption key management
- ✅ Audit trail for 6 years
- ✅ Access logging with user identification
- ✅ Automatic logoff/timeout

### PCI-DSS Requirements

- ✅ Cryptographic key rotation
- ✅ Split knowledge and dual control
- ✅ Key versioning and archival
- ✅ Secure key storage

## Troubleshooting

### Common Issues

1. **"VAULT_ADDR is required"**
   - Solution: Set `VAULT_ADDR` environment variable
   - Never use localhost:8200 in production

2. **"Secret rotation failed"**
   - Check rotation policy configuration
   - Verify provider connectivity
   - Review audit logs for errors

3. **"Cache not updating"**
   - Check TTL configuration
   - Verify cache invalidation on updates
   - Monitor cache metrics

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=debug
export SECRETS_VERBOSE=true

# Test configuration
./novacron secrets test-config

# Dry run rotation
./novacron secrets rotate --key test --dry-run
```

## Migration from Old System

### Step 1: Export Existing Secrets

```bash
# From old system
./old-system export-secrets > secrets.json
```

### Step 2: Import to New System

```bash
# Import with audit trail
./novacron secrets import --file secrets.json --audit
```

### Step 3: Set Rotation Policies

```bash
# Apply policies to imported secrets
./novacron secrets policy apply --all --policy default
```

### Step 4: Verify

```bash
# Test secret access
./novacron secrets test --all

# Verify audit logging
./novacron audit verify --recent
```

## API Reference

### Secrets Manager API

```go
// Get secret with audit logging
func (m *EnhancedSecretsManager) GetSecret(ctx context.Context, key string) (string, error)

// Set secret with audit logging
func (m *EnhancedSecretsManager) SetSecret(ctx context.Context, key string, value string) error

// Rotate secret
func (m *EnhancedSecretsManager) RotateSecret(ctx context.Context, key string) error

// Register rotation policy
func (m *EnhancedSecretsManager) RegisterRotationPolicy(key string, policy SecretRotationPolicy) error
```

### Audit Logger API

```go
// Log secret access
func (a *AuditLogger) LogSecretAccess(ctx context.Context, actor, resource string, action AuditAction, result AuditResult, details map[string]interface{}) error

// Query audit logs
func (a *AuditLogger) Query(ctx context.Context, filter AuditFilter) ([]AuditEvent, error)
```

## Support

For security issues, contact: security@novacron.io

For general support: support@novacron.io

---

**Last Updated**: 2025-09-01  
**Version**: 1.0.0  
**Classification**: CONFIDENTIAL