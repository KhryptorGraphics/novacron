# NovaCron Project Completion - Ralph Orchestration Prompt

## Mission
Complete NovaCron distributed VM management platform to production-ready state using iterative development with automated testing and verification.

## Tools & Integration

### Required MCP Tools
- **Serena**: LSP-based code intelligence for Go/TypeScript editing
- **Cipher**: Memory storage for cross-session context
- **Beads**: Task tracking with NC- prefix (bd commands)
- **Playwright/Skyvern**: Browser automation for E2E testing

### Environment
- **PostgreSQL (NovaCron)**: Port 15432 (non-standard to avoid conflicts)
- **Redis (NovaCron)**: Port 16379 (non-standard to avoid conflicts)
- **Qdrant (NovaCron)**: Port 16333 (non-standard to avoid conflicts)
- **API (REST)**: Port 8090
- **API (WebSocket)**: Port 8091
- **Frontend**: Port 8092
- **MySQL root**: teamrsi123teamrsi123teamrsi123
- **NGC API Key**: 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
- **Email Server**:
  - From: notifications@giggahost.com
  - Password: BL12925VVdd!!
  - Use for: Password reset, 2FA backup codes, account verification, login notifications

---

## Phase 0: Infrastructure Verification

### Execution Steps
1. Check port availability:
   ```bash
   lsof -i :8090 -i :8091 -i :8092 -i :15432 -i :16379 2>/dev/null || echo "Checking ports..."
   ```

2. Verify PostgreSQL connectivity (non-standard port 15432):
   ```bash
   pg_isready -h localhost -p 15432 || echo "PostgreSQL not running on 15432"
   ```

3. Verify Redis connectivity (non-standard port 16379):
   ```bash
   docker exec novacron-redis redis-cli ping || redis-cli -p 16379 ping
   ```

4. Check Go environment:
   ```bash
   go version && cd backend && go mod verify
   ```

5. Check Node.js environment:
   ```bash
   node --version && cd frontend && npm ci
   ```

### Success Criteria
- All ports either available or running NovaCron services
- PostgreSQL responds to pg_isready
- Redis responds PONG
- Go 1.21+ installed
- Node.js 18+ installed

### On Failure
- If port conflict: Check what's using the port, reconfigure if non-NovaCron service
- If PostgreSQL down: Start with:
  ```bash
  docker run -d --restart=always --name novacron-postgres \
    -p 15432:5432 -e POSTGRES_PASSWORD=novacron_secure_pwd postgres:15
  ```
- If Redis down: Start with:
  ```bash
  docker run -d --restart=always --name novacron-redis \
    -p 16379:6379 redis:7-alpine
  ```

---

## Phase 1: Database Persistence Implementation

### Context
Current auth stores (User, Role, Tenant, Session) are in-memory only. Need PostgreSQL and Redis implementations.

### Beads Tasks
```bash
bd update NC-e4g --status=in_progress  # PostgresUserStore (already in progress)
bd update NC-fes --status=in_progress  # PostgresRoleStore
bd update NC-szz --status=in_progress  # PostgresTenantStore
bd update NC-hss --status=in_progress  # RedisSessionStore
bd update NC-00n --status=in_progress  # Token revocation
```

### Execution Steps

#### 1.1 Create PostgreSQL User Store
File: `backend/core/auth/postgres_user_store.go`

```go
// Implement UserService interface with PostgreSQL backend
// Use database/sql with lib/pq driver
// Connection pooling already configured in main.go
```

Key methods:
- `Create(user *User, password string) error`
- `Get(id string) (*User, error)`
- `GetByUsername(username string) (*User, error)`
- `GetByEmail(email string) (*User, error)`
- `List(filter *UserFilter) ([]*User, error)`
- `Update(user *User) error`
- `Delete(id string) error`
- `SetPassword(id, password string) error`
- `VerifyPassword(id, password string) (bool, error)`

#### 1.2 Create PostgreSQL Role Store
File: `backend/core/auth/postgres_role_store.go`

#### 1.3 Create PostgreSQL Tenant Store
File: `backend/core/auth/postgres_tenant_store.go`

#### 1.4 Create Redis Session Store
File: `backend/core/auth/redis_session_store.go`

```go
// Use redis/go-redis/v9
// Store sessions with TTL matching session expiration
// Key format: "session:{session_id}"
```

#### 1.5 Implement Token Revocation
File: `backend/core/auth/token_revocation.go`

```go
// Redis-backed token blacklist
// Key format: "revoked:{jti}"
// TTL matches token expiration
```

### Verification
```bash
make core-test
go test -v ./backend/core/auth/... -run "Postgres|Redis"
```

### Memory Sync
After completing each store:
```
cipher: Store "Completed PostgresUserStore with connection pooling, transaction support"
```

### On Completion
```bash
bd close NC-e4g NC-fes NC-szz NC-hss NC-00n
```

---

## Phase 2: Security Hardening

### Beads Tasks
```bash
bd update NC-rle --status=in_progress  # JWT validation middleware
bd update NC-bx3 --status=in_progress  # Rate limiting middleware
bd update NC-0ey --status=in_progress  # CORS and security headers
```

### Execution Steps

#### 2.1 Complete JWT Validation Middleware
File: `backend/pkg/middleware/auth.go`

Current status: Has `//go:build experimental` tag. Need to:
1. Remove experimental build tag
2. Add token revocation check
3. Add RS256 support (currently HMAC only)
4. Wire into main router

```go
// Check token against revocation list
if revoked, _ := tokenRevocation.IsRevoked(claims["jti"].(string)); revoked {
    http.Error(w, "Token has been revoked", http.StatusUnauthorized)
    return
}
```

#### 2.2 Add Rate Limiting Middleware
File: `backend/pkg/middleware/rate_limit.go`

```go
// Use golang.org/x/time/rate
// Configure per-endpoint limits
// Store in Redis for distributed deployments
```

Recommended limits:
- `/auth/login`: 5 requests/minute per IP
- `/auth/register`: 3 requests/minute per IP
- `/api/*`: 100 requests/minute per user
- `/ws/*`: 10 connections per user

#### 2.3 Configure Security Headers
File: `backend/pkg/middleware/security_headers.go`

Headers to add:
- `Content-Security-Policy`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security` (when TLS enabled)

#### 2.4 Update Default Credentials
File: `.env.production`

- Change AUTH_SECRET
- Change POSTGRES_PASSWORD
- Change REDIS_PASSWORD
- Change GRAFANA_PASSWORD

### Verification
```bash
# Test rate limiting
for i in {1..10}; do curl -s -o /dev/null -w "%{http_code}\n" localhost:8090/api/auth/login -X POST; done

# Test security headers
curl -I localhost:8090/api/health | grep -E "X-|Content-Security|Strict"
```

### On Completion
```bash
bd close NC-rle NC-bx3 NC-0ey
```

---

## Phase 3: Integration Wiring

### Beads Tasks
```bash
bd update NC-1z7 --status=in_progress  # Email service for password reset
bd update NC-9ue --status=in_progress  # Monitoring dashboards real metrics
bd update NC-8tt --status=in_progress  # WebSocket event flow test
```

### Execution Steps

#### 3.1 Wire Email Service
File: `backend/core/notification/email_service.go`

```go
// SMTP-based email service
// Support for templates
// Queue emails via Redis for reliability
```

Required templates:
- Password reset
- 2FA backup codes
- Account verification
- Login notification

#### 3.2 Connect Monitoring Dashboards
Files in `frontend/src/components/monitoring/`

Current: Some components use mock data fallbacks
Action: Verify all components fetch from real `/api/v1/metrics` endpoints

Test each dashboard:
```bash
curl localhost:8090/api/v1/metrics/system
curl localhost:8090/api/v1/metrics/vms
curl localhost:8090/api/v1/alerts
```

#### 3.3 Test WebSocket Event Flow
```javascript
// Browser console test
const ws = new WebSocket('ws://localhost:8091/ws/events/v1');
ws.onopen = () => ws.send(JSON.stringify({type: 'auth', token: 'YOUR_TOKEN'}));
ws.onmessage = (e) => console.log('Event:', JSON.parse(e.data));
```

Verify events flow:
1. VM state changes
2. Alert notifications
3. Metric updates

### Verification
```bash
cd frontend && npm test
# E2E test with Playwright
npx playwright test monitoring.spec.ts
```

### On Completion
```bash
bd close NC-1z7 NC-9ue NC-8tt
```

---

## Phase 4: E2E Testing & Verification

### Beads Tasks
```bash
bd update NC-a2k --status=in_progress  # E2E auth flow testing
bd update NC-m7v --status=in_progress  # Security audit
```

### Execution Steps

#### 4.1 E2E Auth Flow Testing
Use Skyvern or Playwright for browser automation:

```yaml
# Skyvern task
task: "Test NovaCron authentication flow"
url: "http://localhost:8092/auth/login"
steps:
  - action: fill
    selector: "input[name='username']"
    value: "testuser@example.com"
  - action: fill
    selector: "input[name='password']"
    value: "TestPassword123!"
  - action: click
    selector: "button[type='submit']"
  - action: wait
    selector: ".dashboard"
    timeout: 10000
  - action: verify
    selector: ".user-menu"
    contains: "testuser"
```

Test flows:
1. Register new user
2. Login with credentials
3. Setup 2FA
4. Login with 2FA
5. Password reset flow
6. Logout

#### 4.2 Security Audit
Run security checks:

```bash
# Check for exposed secrets
grep -r "password\|secret\|key" --include="*.go" --include="*.ts" | grep -v "_test\|example\|mock"

# Check TLS configuration
openssl s_client -connect localhost:8090 -tls1_2

# Check rate limiting
ab -n 100 -c 10 http://localhost:8090/api/auth/login

# Check CORS
curl -H "Origin: http://evil.com" -I localhost:8090/api/health
```

### Verification Checklist
- [ ] User can register
- [ ] User can login
- [ ] 2FA works correctly
- [ ] Password reset sends email
- [ ] JWT tokens are validated
- [ ] Rate limiting prevents abuse
- [ ] CORS blocks unauthorized origins
- [ ] WebSocket events flow correctly
- [ ] Dashboards show real data
- [ ] VM lifecycle works end-to-end

### On Completion
```bash
bd close NC-a2k NC-m7v
bd stats  # Verify all tasks closed
```

---

## Phase 5: Jetson Thor Deployment Scripts

### Execution Steps

#### 5.1 Create Tegra Setup Script
File: `scripts/jetson-thor-setup.sh`

```bash
#!/bin/bash
# NovaCron Jetson Thor Setup Script

# Check Tegra platform
if ! grep -q "tegra" /proc/device-tree/compatible; then
    echo "Error: Not a Tegra platform"
    exit 1
fi

# Check CUDA availability
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA not installed"
    exit 1
fi

# Install dependencies
apt-get update
apt-get install -y \
    golang-go \
    nodejs \
    npm \
    postgresql-client \
    redis-tools

# Setup Go environment
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Build with CUDA support
cd backend
CGO_ENABLED=1 GOOS=linux GOARCH=arm64 go build -tags cuda -o novacron-api ./cmd/api-server

# Build frontend
cd ../frontend
npm ci
npm run build

echo "NovaCron Jetson Thor setup complete"
```

#### 5.2 CUDA/TensorRT Optimization
- Verify all ML components use GPU
- Check TensorRT availability for inference
- Disable CPU fallbacks for GPU-capable code

```bash
# Check TensorRT
dpkg -l | grep tensorrt

# Verify GPU usage
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

### Success Criteria
- All components build for ARM64
- CUDA acceleration enabled
- TensorRT optimizations applied
- No CPU fallbacks for GPU code

---

## Iteration Protocol

### After Each Iteration
1. Run tests: `make core-test && cd frontend && npm test`
2. Check Beads: `bd ready` for next task
3. Store context: `cipher: Store "<summary of work done>"`
4. Git checkpoint: `git add -A && git commit -m "Phase X.Y: <description>"`

### On Error
1. Read error message carefully
2. Search codebase for similar patterns
3. Use scholarly research MCP for unknown implementations
4. Fix and retry

### Memory Sync Pattern
After every significant change:
```
cipher: Store "NovaCron: Completed <component> with <key details>"
```

### Port Conflict Resolution
If port in use by non-NovaCron service:
1. Check what's using: `lsof -i :<port>`
2. If safe, stop the service
3. If not safe, reconfigure NovaCron to use alternate port
4. Update .env files with new port

### Dependency Resolution
If pip wheel not available:
1. Check conda-forge: `conda search <package>`
2. Build from source if needed
3. Copy to shared location for other envs

---

## Completion Criteria

Project is complete when:
1. All Beads tasks closed: `bd stats` shows 0 open
2. All tests pass: `make test` succeeds
3. E2E flows work in browser
4. Security audit passes
5. Jetson Thor deployment scripts verified

### Final Verification Command
```bash
bd stats && make test && cd frontend && npm test && echo "NovaCron Complete!"
```

---

## DO NOT
- Use claude-flow
- Be destructive to other services on the server
- Use CPU implementations when GPU/CUDA available
- Skip test verification
- Ignore port conflicts

## ALWAYS
- Use Serena for code editing
- Store context in Cipher after major changes
- Update Beads task status
- Run tests after changes
- Check port availability before starting services
