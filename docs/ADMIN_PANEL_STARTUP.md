# NovaCron Admin Panel - Quick Start Guide

## 🚀 Starting the Admin Panel

### Prerequisites

Before starting, ensure you have:
- ✅ PostgreSQL database running
- ✅ Go 1.21+ installed
- ✅ Node.js 18+ and npm installed
- ✅ Database connection configured

---

## Backend Setup

### 1. Configure Environment

Create or update your configuration file `/backend/configs/config.yaml`:

```yaml
server:
  api_port: "8090"
  ws_port: "8091"
  read_timeout: 30s
  write_timeout: 30s
  idle_timeout: 120s
  shutdown_timeout: 30s

database:
  url: "postgres://username:password@localhost:5432/novacron?sslmode=disable"
  max_connections: 50
  conn_max_lifetime: 30m
  conn_max_idle_time: 10m

auth:
  secret: "your-secret-key-change-this-in-production"

logging:
  level: "info"
  format: "json"
  output: "stdout"
  structured: true

vm:
  storage_path: "/var/lib/novacron/vms"
```

Or set environment variables:
```bash
export DATABASE_URL="postgres://username:password@localhost:5432/novacron?sslmode=disable"
export AUTH_SECRET="your-secret-key-change-this-in-production"
export API_PORT="8090"
```

### 2. Start Backend Server

```bash
cd /home/kp/repos/novacron
go mod tidy
go run ./backend/cmd/api-server/main.go
```

The canonical development server will:
- ✅ Connect to PostgreSQL
- ✅ Run database migrations automatically
- ✅ Provide the working local auth, VM, and monitoring API surface
- ✅ Start listening on port 8090

**Expected Output:**
```
INFO Starting NovaCron API Server... version=1.0.0 api_port=8090
INFO Database migrations completed successfully
INFO API Server starting port=8090
```

### 3. Verify Backend is Running

Test the health endpoint:
```bash
curl http://localhost:8090/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-07T...",
  "version": "1.0.0",
  "service": "novacron-api",
  "checks": {
    "database": "ok",
    "storage": "ok"
  }
}
```

### 4. Smoke Test the Canonical API

The `main.go` entrypoint is the canonical local backend. It exposes the auth, VM, and monitoring surfaces needed for general frontend work, but it still returns explicit `501 Not Implemented` responses for `/api/security/*`, `/api/admin/security/*`, `/api/ws/*`, and `/graphql`.

Use these checks instead:
```bash
# Login via canonical auth route
curl -X POST http://localhost:8090/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"user@example.com","password":"secret"}'

# List VMs
curl http://localhost:8090/api/vms

# List VMs via canonical v1 route
curl http://localhost:8090/api/v1/vms

# Fetch monitoring metrics
curl http://localhost:8090/api/monitoring/metrics

# Fetch VM monitoring summary
curl http://localhost:8090/api/monitoring/vms
```

---

## Frontend Setup

### 1. Install Dependencies

```bash
cd /home/kp/repos/novacron/frontend
npm install
```

### 2. Configure API Connection

Update `/frontend/.env.local` (create if doesn't exist):

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8090
NEXT_PUBLIC_WS_URL=ws://localhost:8091

# Auth Configuration
NEXT_PUBLIC_AUTH_ENABLED=true

# Feature Flags
NEXT_PUBLIC_ENABLE_ADMIN=true
```

### 3. Start Frontend Development Server

```bash
cd /home/kp/repos/novacron/frontend
npm run dev
```

The frontend will start on `http://localhost:8092`

**Expected Output:**
```
   ▲ Next.js 14.x.x
   - Local:        http://localhost:8092
   - Network:      http://192.168.x.x:3000

 ✓ Ready in 2.5s
```

### 4. Access Admin Panel

Open your browser and navigate to:
```
http://localhost:8092/admin
```

**Available Admin Routes:**
- **Dashboard**: http://localhost:8092/admin
- **Users**: http://localhost:8092/admin/users
- **Security**: http://localhost:8092/admin/security
- **Analytics**: http://localhost:8092/admin/analytics
- **VMs**: http://localhost:8092/admin/vms
- **Config**: http://localhost:8092/admin/config

---

## Canonical API Surface

The `main.go` entrypoint is the supported local backend for this stabilization phase. It exposes the canonical auth and `/api/v1` routes, while keeping compatibility aliases for the older `/auth/*` and `/api/*` paths.

Use the surface below when validating the local stack:

```
GET    /health                            # API health
POST   /api/auth/login                    # Canonical authentication
POST   /api/auth/register                 # Canonical authentication
GET    /api/auth/check-email              # Canonical auth helper
POST   /auth/login                        # Authentication
POST   /auth/register                     # Authentication
GET    /api/v1/vms                        # Canonical VM list
POST   /api/v1/vms                        # Canonical VM create
GET    /api/v1/vms/{id}                   # Canonical VM details
POST   /api/v1/vms/{id}/start             # Canonical VM start
POST   /api/v1/vms/{id}/stop              # Canonical VM stop
GET    /api/v1/vms/{id}/metrics           # Canonical VM metrics
GET    /api/vms                           # VM list
POST   /api/vms                           # VM create
GET    /api/monitoring/metrics            # Monitoring summary
GET    /api/monitoring/vms                # VM monitoring view
GET    /api/security/*                    # Explicit 501 not implemented
GET    /api/admin/security/*              # Explicit 501 not implemented
GET    /graphql                           # Explicit 501 not implemented
```

The `/admin` frontend route is still available for UI development, but richer admin and security workflows depend on the broader backend rehabilitation track rather than the canonical server documented here.

---

## Database Schema

The following tables are automatically created:

### Core Tables
- `users` - User accounts with roles and 2FA
- `vms` - Virtual machine records
- `vm_metrics` - VM performance metrics

### Additional Tables
- `vm_templates` - VM template definitions
- `security_alerts` - Security alerts and incidents
- `audit_logs` - Comprehensive audit trail
- `security_policies` - Security policy configuration

These tables may exist after migrations, but the standalone server documented here does not expose the full admin/security CRUD surface for them.

---

## Troubleshooting

### Backend Issues

**Database Connection Failed:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection manually
psql -h localhost -U username -d novacron
```

**Port Already in Use:**
```bash
# Check what's using port 8090
lsof -i :8090

# Kill the process or change the port in config
```

**Migrations Failed:**
```bash
# Manually run migrations
psql -h localhost -U username -d novacron -f backend/migrations/schema.sql
```

### Frontend Issues

**API Connection Failed:**
- Verify backend is running on port 8090
- Check CORS settings in backend
- Verify `NEXT_PUBLIC_API_URL` in `.env.local`

**Authentication Issues:**
- Ensure you have a valid JWT token
- Check token hasn't expired (default: 24 hours)
- Verify user has `admin` role

**Build Errors:**
```bash
# Clear Next.js cache
rm -rf .next
npm run dev
```

---

## Development Tips

### 1. Hot Reload

Both frontend and backend support hot reload:
- **Frontend**: Changes auto-reload in browser
- **Backend**: Restart server manually or use `air` for hot reload

### 2. Database Reset

To reset the database:
```sql
DROP DATABASE novacron;
CREATE DATABASE novacron;
```

Then restart the backend to run migrations.

### 3. Sample Data

Create sample admin user:
```sql
INSERT INTO users (username, email, password_hash, role, active)
VALUES ('admin', 'admin@novacron.local', '$2a$10$...', 'admin', true);
```

Create sample VM template:
```sql
INSERT INTO vm_templates (id, name, os, os_version, cpu_cores, memory_mb, disk_gb, created_by)
VALUES ('tmpl-001', 'Ubuntu 24.04 LTS', 'ubuntu', '24.04', 2, 4096, 40, 'admin@novacron.local');
```

---

## Production Deployment

This guide covers the supported local development path only:

```bash
cd /home/kp/repos/novacron
npm run dev
```

That starts the standalone Go API on `:8090` and the Next.js frontend on `:8092`.

The full production backend entrypoint (`./backend/cmd/api-server/main.go`) is part of the broader backend rehabilitation track and should not be treated as a drop-in replacement for the standalone server until that work is completed.

---

## Security Considerations

### Required for Production

1. **Change Default Secrets**
   - Generate strong JWT secret
   - Use environment variables, not config files

2. **Enable HTTPS**
   - Configure TLS certificates
   - Update CORS origins

3. **Database Security**
   - Use strong passwords
   - Enable SSL mode
   - Restrict network access

4. **Admin Access Control**
   - Enforce strong passwords
   - Enable 2FA for admin users
   - Monitor audit logs

5. **Rate Limiting**
   - Configure API rate limits
   - Protect against brute force

---

## Support

For issues or questions:
- **Documentation**: `/docs/ADMIN_PANEL_COMPLETION.md`
- **Local Backend Entry Point**: `/backend/cmd/api-server/main.go`
- **Frontend Components**: Check `/frontend/src/components/admin/`

---

## Next Steps

1. ✅ Create admin user account
2. ✅ Verify authentication and role-based access on `/admin`
3. ✅ Create VM templates for your environment
4. ✅ Validate VM and monitoring endpoints against the standalone API surface
5. ✅ Track any `/api/admin/*` or `/api/security/*` work against the backend rehabilitation backlog

---

**Admin Panel Status:** ✅ **Local Standalone Development Path Operational**

Backend API is listening on `http://localhost:8090`
Frontend is accessible at `http://localhost:8092/admin`
