# NovaCron Security Migration Guide

## Overview

This guide provides step-by-step instructions for implementing the three critical security fixes identified in the codebase analysis:

1. **SQL Injection Prevention** - Migrate to parameterized queries
2. **Secrets Management** - Remove hardcoded secrets and implement Vault
3. **HTTPS/TLS** - Enable encrypted communication

## 1. SQL Injection Prevention

### Current Issues
- Direct SQL string concatenation in authentication queries
- User input not properly sanitized before database operations
- No query builder or ORM layer for safe SQL construction

### Migration Steps

#### Step 1: Update Database Queries

Replace all direct SQL queries with parameterized versions:

**Before (Vulnerable):**
```go
query := fmt.Sprintf("SELECT * FROM users WHERE email = '%s'", userEmail)
rows, err := db.Query(query)
```

**After (Secure):**
```go
query := "SELECT * FROM users WHERE email = $1"
rows, err := db.Query(query, userEmail)
```

#### Step 2: Implement Repository Pattern

Use the new secure database wrapper:

```go
import "github.com/novacron/backend/core/security"

// Initialize secure database
db, _ := sql.Open("postgres", dbURL)
secureDB := security.NewSecureDB(db)

// Create repositories
vmRepo := security.NewVMRepository(secureDB)
userRepo := security.NewUserRepository(secureDB)

// Use safe methods
vms, err := vmRepo.GetVMs(ctx, "running")
user, err := userRepo.GetUserByEmail(ctx, email)
```

#### Step 3: Use Query Builder for Complex Queries

```go
qb := security.NewQueryBuilder()
query, args := qb.Select("id", "name", "state").
    From("vms").
    Where("state = ?", "running").
    And("node_id = ?", nodeID).
    OrderBy("created_at", "DESC").
    Build()

rows, err := db.Query(query, args...)
```

#### Step 4: Input Validation

Always validate user input before using in queries:

```go
validator := security.NewInputValidator()

// Validate VM name
if err := validator.ValidateVMName(vmName); err != nil {
    return fmt.Errorf("invalid VM name: %w", err)
}

// Validate email
if !security.ValidateEmail(email) {
    return fmt.Errorf("invalid email format")
}
```

## 2. Secrets Management with Vault

### Current Issues
- Hardcoded default secrets in code
- Secrets stored in environment variables without encryption
- No secret rotation mechanism

### Migration Steps

#### Step 1: Set Up HashiCorp Vault

```bash
# Install Vault
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# Start Vault in development mode (for testing)
vault server -dev

# In production, use proper configuration
vault server -config=/etc/vault/config.hcl
```

#### Step 2: Initialize Vault Secrets

```go
// Initialize Vault for NovaCron
initializer, _ := security.NewVaultInitializer(
    "http://localhost:8200",
    rootToken,
)

// Set up all required secrets
err := initializer.SetupNovaCronSecrets(ctx)

// Create application token
appToken, err := initializer.CreateAppToken(ctx)
```

#### Step 3: Update Application Configuration

Replace hardcoded secrets with Vault integration:

**Before:**
```go
authSecret := getEnvOrDefault("AUTH_SECRET", "changeme_in_production")
```

**After:**
```go
vault, _ := security.NewVaultManager(vaultAddr, vaultToken)
secrets, _ := vault.LoadSecrets(ctx)
authSecret := secrets.AuthSecret
```

#### Step 4: Environment Variables for Development

Create `.env.development`:
```bash
NOVACRON_ENV=development
VAULT_ADDR=http://localhost:8200
VAULT_TOKEN=dev-token
AUTH_SECRET=dev-secret-change-in-production
DB_PASSWORD=dev-db-password
NOVACRON_API_KEY=dev-api-key
JWT_SECRET=dev-jwt-secret
```

#### Step 5: Production Deployment

```bash
# Set Vault address and token
export VAULT_ADDR=https://vault.production.com:8200
export VAULT_TOKEN=$(cat /etc/novacron/vault-token)

# Application will automatically load secrets from Vault
```

## 3. HTTPS/TLS Configuration

### Current Issues
- All communication over unencrypted HTTP
- No TLS configuration for API endpoints
- Missing security headers

### Migration Steps

#### Step 1: Generate TLS Certificates

For development:
```go
hosts := []string{"localhost", "127.0.0.1", "novacron.local"}
err := security.GenerateSelfSignedCert(
    hosts,
    "/etc/novacron/tls/cert.pem",
    "/etc/novacron/tls/key.pem",
)
```

For production (using Let's Encrypt):
```bash
# Install certbot
sudo apt-get install certbot

# Generate certificate
sudo certbot certonly --standalone -d novacron.example.com

# Certificates will be in:
# /etc/letsencrypt/live/novacron.example.com/fullchain.pem
# /etc/letsencrypt/live/novacron.example.com/privkey.pem
```

#### Step 2: Update Server Configuration

```go
// Initialize TLS configuration
tlsConfig := security.NewTLSConfig(certPath, keyPath)

// Create HTTPS server
tlsServer, _ := security.NewTLSServer(":8443", handler, tlsConfig)

// Start HTTP to HTTPS redirect
go func() {
    redirectHandler := &security.HTTPSRedirectHandler{HTTPSPort: "8443"}
    http.ListenAndServe(":8080", redirectHandler)
}()

// Start HTTPS server
tlsServer.Start()
```

#### Step 3: Add Security Headers

```go
// Wrap handlers with security headers middleware
handler := security.SecurityHeaders(mux)
```

This adds:
- `Strict-Transport-Security` (HSTS)
- `X-Content-Type-Options`
- `X-Frame-Options`
- `Content-Security-Policy`
- `X-XSS-Protection`

#### Step 4: Update Frontend Configuration

Update frontend to use HTTPS endpoints:

```javascript
// frontend/src/config/api.ts
const API_BASE = process.env.NODE_ENV === 'production' 
    ? 'https://api.novacron.com:8443'
    : 'https://localhost:8443';

const WS_BASE = process.env.NODE_ENV === 'production'
    ? 'wss://api.novacron.com:8443'
    : 'wss://localhost:8443';
```

## Testing the Security Fixes

### Test SQL Injection Prevention

```bash
# Try SQL injection (should fail)
curl -X GET "https://localhost:8443/api/vms?state='; DROP TABLE vms; --"

# Response should be 400 Bad Request or filtered
```

### Test Vault Integration

```bash
# Check if secrets are loaded from Vault
curl -X GET https://localhost:8443/api/health

# Should return success without exposing secrets
```

### Test HTTPS/TLS

```bash
# Test HTTPS endpoint
curl -k https://localhost:8443/api/health

# Test HTTP redirect
curl -I http://localhost:8080/api/health
# Should return 301 redirect to HTTPS

# Check TLS version
openssl s_client -connect localhost:8443 -tls1_2
```

## Deployment Checklist

- [ ] All SQL queries use parameterized statements
- [ ] Input validation implemented for all user inputs
- [ ] Vault configured and secrets migrated
- [ ] No hardcoded secrets in codebase
- [ ] TLS certificates generated and installed
- [ ] HTTPS enabled on all endpoints
- [ ] HTTP to HTTPS redirect configured
- [ ] Security headers added to all responses
- [ ] Rate limiting implemented for authentication endpoints
- [ ] Audit logging enabled for security events
- [ ] Penetration testing performed
- [ ] Security monitoring configured

## Rollback Plan

If issues occur during migration:

1. **Database**: Queries are backward compatible, no rollback needed
2. **Vault**: Fall back to environment variables by setting `NOVACRON_ENV=development`
3. **TLS**: Can temporarily disable by reverting to HTTP (not recommended)

## Security Monitoring

Set up monitoring for:
- Failed authentication attempts
- SQL injection attempts (400 errors on query endpoints)
- TLS handshake failures
- Vault access failures
- Rate limit violations

## Additional Resources

- [OWASP SQL Injection Prevention](https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html)
- [HashiCorp Vault Documentation](https://www.vaultproject.io/docs)
- [Mozilla SSL Configuration Generator](https://ssl-config.mozilla.org/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)