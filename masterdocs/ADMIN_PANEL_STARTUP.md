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
  api_port: "8080"
  ws_port: "8081"
  config_path: "/etc/novacron/config.yaml"  # For admin config API
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
export API_PORT="8080"
```

### 2. Start Backend Server

```bash
cd /home/kp/novacron/backend

# Start the API server
go run cmd/api-server/main.go
```

The server will:
- ✅ Connect to PostgreSQL
- ✅ Run database migrations automatically
- ✅ Create all admin tables (users, templates, security_alerts, audit_logs, etc.)
- ✅ Register admin API routes at `/api/admin/*`
- ✅ Start listening on port 8080

**Expected Output:**
```
INFO Starting NovaCron API Server... version=1.0.0 api_port=8080
INFO Database migrations completed successfully
INFO Admin API routes registered
INFO API Server starting port=8080
```

### 3. Verify Backend is Running

Test the health endpoint:
```bash
curl http://localhost:8080/health
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
    "kvm": "not configured",
    "storage": "ok"
  }
}
```

### 4. Test Admin Endpoints

Create a test user:
```bash
curl -X POST http://localhost:8080/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "email": "admin@novacron.local",
    "password": "Admin123!@#",
    "tenant_id": "default"
  }'
```

Login and get token:
```bash
curl -X POST http://localhost:8080/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "Admin123!@#"
  }'
```

Save the token from the response.

Test admin API (replace TOKEN with your actual token):
```bash
# List users
curl http://localhost:8080/api/admin/users \
  -H "Authorization: Bearer TOKEN"

# Get security metrics
curl http://localhost:8080/api/admin/security/metrics \
  -H "Authorization: Bearer TOKEN"

# List VM templates
curl http://localhost:8080/api/admin/templates \
  -H "Authorization: Bearer TOKEN"
```

---

## Frontend Setup

### 1. Install Dependencies

```bash
cd /home/kp/novacron/frontend
npm install
```

### 2. Configure API Connection

Update `/frontend/.env.local` (create if doesn't exist):

```env
# API Configuration
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8081

# Auth Configuration
NEXT_PUBLIC_AUTH_ENABLED=true

# Feature Flags
NEXT_PUBLIC_ENABLE_ADMIN=true
```

### 3. Start Frontend Development Server

```bash
cd /home/kp/novacron/frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

**Expected Output:**
```
   ▲ Next.js 14.x.x
   - Local:        http://localhost:3000
   - Network:      http://192.168.x.x:3000

 ✓ Ready in 2.5s
```

### 4. Access Admin Panel

Open your browser and navigate to:
```
http://localhost:3000/admin
```

**Available Admin Routes:**
- **Dashboard**: http://localhost:3000/admin
- **Users**: http://localhost:3000/admin/users
- **Security**: http://localhost:3000/admin/security
- **Analytics**: http://localhost:3000/admin/analytics
- **VMs**: http://localhost:3000/admin/vms
- **Config**: http://localhost:3000/admin/config

---

## Admin API Endpoints

### User Management
```
GET    /api/admin/users                    # List users
POST   /api/admin/users                    # Create user
GET    /api/admin/users/{id}               # Get user
PUT    /api/admin/users/{id}               # Update user
DELETE /api/admin/users/{id}               # Delete user
POST   /api/admin/users/{id}/roles         # Assign roles
POST   /api/admin/users/bulk               # Bulk operations
```

### Security
```
GET    /api/admin/security/metrics         # Security overview
GET    /api/admin/security/alerts          # List alerts
GET    /api/admin/security/alerts/{id}     # Get alert
PUT    /api/admin/security/alerts/{id}     # Update alert
GET    /api/admin/security/audit           # Audit logs
GET    /api/admin/security/policies        # Security policies
PUT    /api/admin/security/policies/{id}   # Update policy
```

### VM Templates
```
GET    /api/admin/templates                # List templates
POST   /api/admin/templates                # Create template
GET    /api/admin/templates/{id}           # Get template
PUT    /api/admin/templates/{id}           # Update template
DELETE /api/admin/templates/{id}           # Delete template
```

### System Configuration
```
GET    /api/admin/config                   # Get configuration
PUT    /api/admin/config                   # Update configuration
POST   /api/admin/config/validate          # Validate config
POST   /api/admin/config/backup            # Create backup
GET    /api/admin/config/backups           # List backups
POST   /api/admin/config/restore/{id}      # Restore backup
```

### Database Administration
```
GET    /api/admin/database/tables          # List tables
GET    /api/admin/database/tables/{table}  # Table details
POST   /api/admin/database/query           # Execute query
POST   /api/admin/database/execute         # Execute statement
```

---

## Database Schema

The following tables are automatically created:

### Core Tables
- `users` - User accounts with roles and 2FA
- `vms` - Virtual machine records
- `vm_metrics` - VM performance metrics

### Admin Tables
- `vm_templates` - VM template definitions
- `security_alerts` - Security alerts and incidents
- `audit_logs` - Comprehensive audit trail
- `security_policies` - Security policy configuration

All tables include proper indexes for performance.

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
# Check what's using port 8080
lsof -i :8080

# Kill the process or change the port in config
```

**Migrations Failed:**
```bash
# Manually run migrations
psql -h localhost -U username -d novacron -f backend/migrations/schema.sql
```

### Frontend Issues

**API Connection Failed:**
- Verify backend is running on port 8080
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

### Backend

```bash
# Build the binary
cd backend
go build -o novacron-server cmd/api-server/main.go

# Run with systemd
sudo systemctl start novacron-api
```

### Frontend

```bash
# Build production bundle
cd frontend
npm run build

# Start production server
npm start
```

Or use a reverse proxy (nginx/caddy) for production.

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
- **API Reference**: Check individual handler files in `/backend/api/admin/`
- **Frontend Components**: Check `/frontend/src/components/admin/`

---

## Next Steps

1. ✅ Create admin user account
2. ✅ Configure security policies
3. ✅ Create VM templates for your environment
4. ✅ Set up monitoring and alerts
5. ✅ Configure backup schedules
6. ✅ Review audit logs regularly

---

**Admin Panel Status:** ✅ **FULLY CONNECTED AND OPERATIONAL**

Backend API is listening on `http://localhost:8080/api/admin/*`
Frontend is accessible at `http://localhost:3000/admin`
