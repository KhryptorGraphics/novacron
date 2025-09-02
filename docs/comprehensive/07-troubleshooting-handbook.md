# NovaCron Troubleshooting Handbook

## Overview

This handbook provides comprehensive troubleshooting procedures for common issues, systematic debugging approaches, and recovery procedures for the NovaCron platform.

## Troubleshooting Framework

### 1. Systematic Approach

#### Problem Identification Process
```
1. Symptom Identification
   ├── What is not working as expected?
   ├── When did the issue start?
   ├── What changed recently?
   └── Who is affected?

2. Impact Assessment
   ├── Severity: Critical/High/Medium/Low
   ├── Scope: System-wide/Service/Feature/User
   ├── Business Impact: Revenue/Users/Operations
   └── SLA Impact: Response time/Availability/Error rate

3. Initial Diagnosis
   ├── Check system status and health endpoints
   ├── Review recent logs and error messages
   ├── Verify configuration and environment
   └── Identify potential root causes

4. Resolution & Verification
   ├── Apply fixes based on diagnosis
   ├── Verify fix effectiveness
   ├── Monitor for regression
   └── Document resolution steps
```

#### Triage Priority Matrix
| Impact | Urgency | Priority | Response Time |
|--------|---------|----------|---------------|
| High | High | P0 | 15 minutes |
| High | Medium | P1 | 1 hour |
| Medium | High | P1 | 1 hour |
| Medium | Medium | P2 | 4 hours |
| Low | Any | P3 | 24 hours |

### 2. Diagnostic Tools

#### System Health Check Script
```bash
#!/bin/bash
# NovaCron System Health Check Script

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== NovaCron System Health Check ==="
echo "Timestamp: $(date)"
echo

# Function to check service status
check_service() {
    local service=$1
    local port=$2
    
    if systemctl is-active --quiet $service; then
        echo -e "${GREEN}✓${NC} $service is running"
        
        if [ ! -z "$port" ]; then
            if nc -z localhost $port; then
                echo -e "${GREEN}✓${NC} $service port $port is accessible"
            else
                echo -e "${RED}✗${NC} $service port $port is not accessible"
                return 1
            fi
        fi
    else
        echo -e "${RED}✗${NC} $service is not running"
        return 1
    fi
    return 0
}

# Function to check disk space
check_disk_space() {
    local threshold=80
    df -H | awk '
    BEGIN { print "Disk Space Check:" }
    NR>1 {
        gsub(/%/, "", $5)
        if($5 > '$threshold') {
            print "⚠️  " $1 " is " $5 "% full (>" '$threshold' "%)"
        } else {
            print "✓ " $1 " is " $5 "% full"
        }
    }'
}

# Function to check memory usage
check_memory() {
    local mem_usage=$(free | awk 'FNR==2{printf "%.0f", $3/($3+$4)*100}')
    if [ $mem_usage -gt 80 ]; then
        echo -e "${RED}⚠️${NC} Memory usage is ${mem_usage}%"
    else
        echo -e "${GREEN}✓${NC} Memory usage is ${mem_usage}%"
    fi
}

# Function to check load average
check_load() {
    local load=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    local cores=$(nproc)
    local load_percent=$(echo "$load * 100 / $cores" | bc -l | cut -d. -f1)
    
    if [ $load_percent -gt 100 ]; then
        echo -e "${RED}⚠️${NC} Load average is ${load} (${load_percent}% of ${cores} cores)"
    else
        echo -e "${GREEN}✓${NC} Load average is ${load} (${load_percent}% of ${cores} cores)"
    fi
}

# Main health checks
echo "### Service Status ###"
check_service "novacron-api" 8080
check_service "novacron-frontend" 3000
check_service "postgresql" 5432
check_service "redis-server" 6379
echo

echo "### System Resources ###"
check_disk_space
check_memory
check_load
echo

echo "### Application Health ###"
# API health check
if curl -f -s http://localhost:8080/health > /dev/null; then
    echo -e "${GREEN}✓${NC} API health endpoint responding"
else
    echo -e "${RED}✗${NC} API health endpoint not responding"
fi

# Frontend health check
if curl -f -s http://localhost:3000 > /dev/null; then
    echo -e "${GREEN}✓${NC} Frontend is accessible"
else
    echo -e "${RED}✗${NC} Frontend is not accessible"
fi

# Database connectivity
if PGPASSWORD=$DB_PASSWORD psql -h localhost -U novacron -d novacron -c "SELECT 1;" > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Database connection successful"
else
    echo -e "${RED}✗${NC} Database connection failed"
fi

echo
echo "### Recent Errors ###"
# Check recent errors in application logs
if [ -f "/var/log/novacron/app.log" ]; then
    echo "Recent application errors (last 10):"
    tail -1000 /var/log/novacron/app.log | grep -i error | tail -10
fi

echo
echo "Health check completed at $(date)"
```

#### Log Analysis Tool
```python
#!/usr/bin/env python3
"""
NovaCron Log Analysis Tool
Analyzes application logs for patterns, errors, and performance issues
"""

import re
import json
import sys
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

class LogAnalyzer:
    def __init__(self, log_file: str):
        self.log_file = log_file
        self.patterns = {
            'error': re.compile(r'"level":"error"', re.IGNORECASE),
            'warning': re.compile(r'"level":"warn"', re.IGNORECASE),
            'slow_query': re.compile(r'duration["\s]*:["\s]*([0-9.]+)', re.IGNORECASE),
            'http_error': re.compile(r'"status":([45][0-9]{2})'),
            'database_error': re.compile(r'database|postgres|sql', re.IGNORECASE),
            'auth_failure': re.compile(r'authentication|unauthorized|forbidden', re.IGNORECASE),
        }
    
    def analyze_timeframe(self, hours: int = 24) -> Dict:
        """Analyze logs from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        results = {
            'errors': [],
            'warnings': [],
            'slow_queries': [],
            'http_errors': Counter(),
            'error_patterns': Counter(),
            'timeline': defaultdict(int),
        }
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())
                    log_time = datetime.fromisoformat(log_entry.get('timestamp', '').replace('Z', '+00:00'))
                    
                    if log_time < cutoff_time:
                        continue
                    
                    # Count timeline
                    hour_key = log_time.strftime('%Y-%m-%d %H:00')
                    results['timeline'][hour_key] += 1
                    
                    # Analyze patterns
                    self._analyze_entry(log_entry, results)
                    
                except (json.JSONDecodeError, ValueError):
                    continue
        
        return results
    
    def _analyze_entry(self, entry: Dict, results: Dict):
        """Analyze individual log entry"""
        message = entry.get('message', '')
        level = entry.get('level', '')
        
        # Categorize by level
        if level.lower() == 'error':
            results['errors'].append(entry)
            results['error_patterns'][self._categorize_error(message)] += 1
        elif level.lower() in ['warn', 'warning']:
            results['warnings'].append(entry)
        
        # Check for HTTP errors
        if 'status' in entry:
            status = entry.get('status')
            if isinstance(status, int) and status >= 400:
                results['http_errors'][status] += 1
        
        # Check for slow queries
        if 'duration' in entry:
            duration = float(entry.get('duration', 0))
            if duration > 1.0:  # Queries slower than 1 second
                results['slow_queries'].append({
                    'duration': duration,
                    'query': entry.get('query', 'Unknown'),
                    'timestamp': entry.get('timestamp'),
                })
    
    def _categorize_error(self, message: str) -> str:
        """Categorize error message"""
        if re.search(r'database|postgres|sql', message, re.IGNORECASE):
            return 'Database Error'
        elif re.search(r'network|connection|timeout', message, re.IGNORECASE):
            return 'Network Error'
        elif re.search(r'auth|unauthorized|forbidden', message, re.IGNORECASE):
            return 'Authentication Error'
        elif re.search(r'vm|virtual machine|kvm', message, re.IGNORECASE):
            return 'VM Management Error'
        else:
            return 'General Error'
    
    def generate_report(self, hours: int = 24) -> str:
        """Generate analysis report"""
        results = self.analyze_timeframe(hours)
        
        report = f"""
=== NovaCron Log Analysis Report ===
Analysis Period: Last {hours} hours
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Log Entries: {sum(results['timeline'].values())}
- Errors: {len(results['errors'])}
- Warnings: {len(results['warnings'])}
- Slow Queries: {len(results['slow_queries'])}
- HTTP Errors: {sum(results['http_errors'].values())}

## Error Breakdown
"""
        
        for error_type, count in results['error_patterns'].most_common():
            report += f"- {error_type}: {count}\n"
        
        report += "\n## HTTP Error Status Codes\n"
        for status, count in results['http_errors'].most_common():
            report += f"- {status}: {count}\n"
        
        if results['slow_queries']:
            report += "\n## Slowest Queries\n"
            sorted_queries = sorted(results['slow_queries'], 
                                  key=lambda x: x['duration'], reverse=True)[:5]
            for query in sorted_queries:
                report += f"- {query['duration']:.2f}s: {query['query'][:100]}...\n"
        
        report += "\n## Recent Critical Errors\n"
        for error in results['errors'][-5:]:  # Last 5 errors
            timestamp = error.get('timestamp', 'Unknown')
            message = error.get('message', 'No message')[:200]
            report += f"- {timestamp}: {message}\n"
        
        return report

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: log_analyzer.py <log_file> [hours]")
        sys.exit(1)
    
    log_file = sys.argv[1]
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    
    analyzer = LogAnalyzer(log_file)
    print(analyzer.generate_report(hours))
```

## Common Issues & Solutions

### 1. API Performance Issues

#### Symptom: High Response Times
```
Problem: API responses taking longer than 1 second
Status Code: 200 but slow response
Logs: No obvious errors, but high latency
```

**Diagnosis Steps:**
```bash
# 1. Check system resources
top -p $(pidof novacron-api)
iostat -x 1 10

# 2. Check database performance
sudo -u postgres psql -d novacron << 'EOF'
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
WHERE mean_time > 100 
ORDER BY total_time DESC 
LIMIT 10;
EOF

# 3. Check connection pools
curl -s http://localhost:8080/debug/pprof/goroutine?debug=1 | head -20
```

**Solutions:**

1. **Database Query Optimization:**
```sql
-- Add missing indexes
CREATE INDEX CONCURRENTLY idx_vms_tenant_state ON vms(tenant_id, state);
CREATE INDEX CONCURRENTLY idx_vm_metrics_recent ON vm_metrics(vm_id, timestamp) 
WHERE timestamp > NOW() - INTERVAL '24 hours';

-- Update table statistics
ANALYZE vms;
ANALYZE vm_metrics;
```

2. **Connection Pool Tuning:**
```go
// Increase connection pool size
db.SetMaxOpenConns(50)
db.SetMaxIdleConns(25)
db.SetConnMaxLifetime(300 * time.Second)
```

3. **Add Caching:**
```go
// Cache frequent queries
func (h *Handler) ListVMsWithCache(c *gin.Context) {
    cacheKey := fmt.Sprintf("vms:list:%s", c.Query("tenant_id"))
    
    if cached, found := h.cache.Get(cacheKey); found {
        c.JSON(200, cached)
        return
    }
    
    vms, err := h.vmManager.ListVMs(c.Query("tenant_id"))
    if err != nil {
        c.JSON(500, gin.H{"error": err.Error()})
        return
    }
    
    h.cache.Set(cacheKey, vms, 5*time.Minute)
    c.JSON(200, vms)
}
```

#### Symptom: API Gateway Timeout
```
Problem: 504 Gateway Timeout errors
Client receiving: 504 status code
Nginx logs: upstream timed out
```

**Diagnosis:**
```bash
# Check nginx status
systemctl status nginx

# Check upstream health
curl -I http://localhost:8080/health

# Check nginx configuration
nginx -t
```

**Solutions:**
```nginx
# Increase timeout values in nginx.conf
location /api/ {
    proxy_connect_timeout 30s;
    proxy_send_timeout 60s;
    proxy_read_timeout 60s;
    proxy_pass http://novacron_api;
}
```

### 2. Database Connectivity Issues

#### Symptom: Connection Pool Exhaustion
```
Problem: "too many clients already" errors
Logs: connection pool full
Database: max_connections reached
```

**Diagnosis:**
```sql
-- Check active connections
SELECT count(*) as total_connections FROM pg_stat_activity;

-- Check connections by state
SELECT state, count(*) 
FROM pg_stat_activity 
GROUP BY state;

-- Check long-running queries
SELECT pid, now() - pg_stat_activity.query_start AS duration, query 
FROM pg_stat_activity 
WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';
```

**Solutions:**

1. **Increase connection limits:**
```sql
-- In postgresql.conf
max_connections = 200
shared_buffers = 256MB

-- Restart PostgreSQL
sudo systemctl restart postgresql
```

2. **Fix connection leaks:**
```go
func (db *Database) QueryWithTimeout(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
    ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
    defer cancel()
    
    rows, err := db.db.QueryContext(ctx, query, args...)
    if err != nil {
        return nil, err
    }
    
    // Ensure rows are closed
    return &RowsWithCleanup{Rows: rows}, nil
}

type RowsWithCleanup struct {
    *sql.Rows
}

func (r *RowsWithCleanup) Close() error {
    if r.Rows != nil {
        return r.Rows.Close()
    }
    return nil
}
```

3. **Implement connection pool monitoring:**
```go
func (db *Database) MonitorConnectionPool() {
    ticker := time.NewTicker(30 * time.Second)
    go func() {
        for range ticker.C {
            stats := db.db.Stats()
            if stats.InUse > stats.MaxOpenConnections*0.8 {
                logger.Warn("Connection pool usage high",
                    "in_use", stats.InUse,
                    "max", stats.MaxOpenConnections,
                    "wait_count", stats.WaitCount,
                )
            }
        }
    }()
}
```

#### Symptom: Database Lock Contention
```
Problem: Queries hanging or timing out
Logs: deadlock detected, process canceled
Performance: Significant slowdown
```

**Diagnosis:**
```sql
-- Check for locks
SELECT 
    blocked_locks.pid AS blocked_pid,
    blocked_activity.usename AS blocked_user,
    blocking_locks.pid AS blocking_pid,
    blocking_activity.usename AS blocking_user,
    blocked_activity.query AS blocked_statement,
    blocking_activity.query AS current_statement_in_blocking_process
FROM pg_catalog.pg_locks blocked_locks
JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid
JOIN pg_catalog.pg_locks blocking_locks 
    ON blocking_locks.locktype = blocked_locks.locktype
    AND blocking_locks.transactionid = blocked_locks.transactionid
JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid
WHERE NOT blocked_locks.granted;
```

**Solutions:**

1. **Identify and fix problematic queries:**
```sql
-- Kill long-running blocking query (carefully!)
SELECT pg_terminate_backend(12345); -- Replace with actual PID
```

2. **Implement proper transaction management:**
```go
func (db *Database) ExecuteInTransaction(fn func(tx *sql.Tx) error) error {
    tx, err := db.db.Begin()
    if err != nil {
        return err
    }
    
    defer func() {
        if p := recover(); p != nil {
            tx.Rollback()
            panic(p)
        } else if err != nil {
            tx.Rollback()
        } else {
            err = tx.Commit()
        }
    }()
    
    err = fn(tx)
    return err
}
```

### 3. VM Management Issues

#### Symptom: VM Creation Failures
```
Problem: VMs failing to create
Status: CREATE_FAILED state
Logs: KVM/libvirt errors
```

**Diagnosis:**
```bash
# Check KVM functionality
lsmod | grep kvm
ls -la /dev/kvm

# Check libvirt status
systemctl status libvirtd
virsh list --all

# Check available resources
virsh nodeinfo
virsh capabilities
```

**Solutions:**

1. **Fix KVM permissions:**
```bash
# Add user to kvm group
sudo usermod -a -G kvm novacron
sudo usermod -a -G libvirt novacron

# Restart services
sudo systemctl restart libvirtd
sudo systemctl restart novacron-api
```

2. **Check storage availability:**
```bash
# Check storage pool
virsh pool-list
virsh pool-info default

# Create storage pool if missing
virsh pool-define-as default dir --target /var/lib/libvirt/images
virsh pool-autostart default
virsh pool-start default
```

3. **Implement better error handling:**
```go
func (vm *VMManager) CreateVM(config VMConfig) (*VM, error) {
    // Pre-flight checks
    if err := vm.validateResources(config); err != nil {
        return nil, fmt.Errorf("resource validation failed: %w", err)
    }
    
    if err := vm.checkStorageSpace(config.DiskSize); err != nil {
        return nil, fmt.Errorf("insufficient storage: %w", err)
    }
    
    // Create VM with retry logic
    var lastErr error
    for attempt := 0; attempt < 3; attempt++ {
        vm, err := vm.attemptCreateVM(config)
        if err == nil {
            return vm, nil
        }
        
        lastErr = err
        logger.Warn("VM creation attempt failed",
            "attempt", attempt+1,
            "error", err,
        )
        
        time.Sleep(time.Duration(attempt+1) * time.Second)
    }
    
    return nil, fmt.Errorf("VM creation failed after 3 attempts: %w", lastErr)
}
```

#### Symptom: VM Migration Failures
```
Problem: Live migration failing
Status: MIGRATION_FAILED
Logs: Network connectivity issues
```

**Diagnosis:**
```bash
# Test network connectivity between nodes
ping target-node.internal
ssh target-node.internal 'virsh list'

# Check migration compatibility
virsh domcapabilities | grep -A5 migration

# Test migration manually
virsh migrate --live vm-name qemu+ssh://target-node/system
```

**Solutions:**

1. **Fix network configuration:**
```bash
# Ensure consistent network setup
virsh net-list
virsh net-dumpxml default

# Create bridge if missing
sudo brctl addbr virbr0
sudo brctl addif virbr0 eth0
```

2. **Implement migration pre-checks:**
```go
func (vm *VMManager) MigrateVM(vmID, targetNode string) error {
    // Pre-migration checks
    if err := vm.checkMigrationCompatibility(vmID, targetNode); err != nil {
        return fmt.Errorf("migration compatibility check failed: %w", err)
    }
    
    if err := vm.checkTargetNodeCapacity(targetNode); err != nil {
        return fmt.Errorf("target node capacity check failed: %w", err)
    }
    
    if err := vm.checkNetworkConnectivity(targetNode); err != nil {
        return fmt.Errorf("network connectivity check failed: %w", err)
    }
    
    return vm.performMigration(vmID, targetNode)
}
```

### 4. Authentication Issues

#### Symptom: JWT Token Validation Failures
```
Problem: Users getting 401 Unauthorized
Logs: "invalid token signature"
Frontend: Redirect to login page
```

**Diagnosis:**
```bash
# Check JWT secret configuration
grep JWT_SECRET /opt/novacron/config/app.yaml

# Validate token manually
echo "eyJhbGc..." | base64 -d | jq .

# Check token expiration
date -d @<expiration_timestamp>
```

**Solutions:**

1. **Fix JWT configuration:**
```yaml
# Ensure consistent JWT secret across all instances
auth:
  jwt_secret: "your-consistent-secret-key-32-chars-min"
  token_expiry: "24h"
  refresh_token_expiry: "168h"
```

2. **Implement token refresh:**
```go
func (auth *AuthManager) RefreshToken(refreshToken string) (*TokenPair, error) {
    claims, err := auth.validateRefreshToken(refreshToken)
    if err != nil {
        return nil, fmt.Errorf("invalid refresh token: %w", err)
    }
    
    // Generate new token pair
    accessToken, err := auth.generateAccessToken(claims.UserID)
    if err != nil {
        return nil, err
    }
    
    newRefreshToken, err := auth.generateRefreshToken(claims.UserID)
    if err != nil {
        return nil, err
    }
    
    return &TokenPair{
        AccessToken:  accessToken,
        RefreshToken: newRefreshToken,
    }, nil
}
```

#### Symptom: Session Management Issues
```
Problem: Users logged out unexpectedly
Logs: Session expired or not found
Database: Session table cleanup issues
```

**Solutions:**

1. **Implement proper session cleanup:**
```sql
-- Clean up expired sessions
DELETE FROM user_sessions 
WHERE expires_at < NOW() - INTERVAL '1 hour';

-- Create index for cleanup performance
CREATE INDEX CONCURRENTLY idx_sessions_expires 
ON user_sessions(expires_at) 
WHERE expires_at < NOW();
```

2. **Add session monitoring:**
```go
func (sm *SessionManager) CleanupExpiredSessions() {
    ticker := time.NewTicker(1 * time.Hour)
    go func() {
        for range ticker.C {
            deleted, err := sm.deleteExpiredSessions()
            if err != nil {
                logger.Error("Session cleanup failed", "error", err)
            } else if deleted > 0 {
                logger.Info("Cleaned up expired sessions", "count", deleted)
            }
        }
    }()
}
```

### 5. Frontend Issues

#### Symptom: Slow Page Loading
```
Problem: Pages taking >5 seconds to load
Browser: Network tab shows slow requests
User Experience: Poor performance
```

**Diagnosis:**
```javascript
// Check bundle sizes
npm run analyze

// Monitor performance
console.time('page-load');
// ... page load logic
console.timeEnd('page-load');
```

**Solutions:**

1. **Implement code splitting:**
```typescript
// Lazy load non-critical components
const HeavyComponent = lazy(() => 
  import('./HeavyComponent').then(module => ({
    default: module.HeavyComponent
  }))
);

// Use Suspense with loading fallback
<Suspense fallback={<LoadingSpinner />}>
  <HeavyComponent />
</Suspense>
```

2. **Optimize API calls:**
```typescript
// Batch API requests
const useBatchedQueries = (queries: string[]) => {
  return useQuery(['batched', queries], async () => {
    const results = await Promise.all(
      queries.map(query => fetch(query).then(r => r.json()))
    );
    return results;
  });
};
```

#### Symptom: WebSocket Connection Issues
```
Problem: Real-time updates not working
Console: WebSocket connection failed
Network: Connection dropping frequently
```

**Solutions:**

1. **Implement connection recovery:**
```typescript
class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  
  connect() {
    try {
      this.ws = new WebSocket(WS_URL);
      this.setupEventHandlers();
    } catch (error) {
      console.error('WebSocket connection failed:', error);
      this.handleReconnect();
    }
  }
  
  private handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = Math.pow(2, this.reconnectAttempts) * 1000;
      
      setTimeout(() => {
        console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
        this.connect();
      }, delay);
    }
  }
  
  private setupEventHandlers() {
    if (!this.ws) return;
    
    this.ws.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    };
    
    this.ws.onclose = (event) => {
      console.log('WebSocket closed:', event.reason);
      this.handleReconnect();
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }
}
```

## System Recovery Procedures

### 1. Service Recovery

#### API Server Recovery
```bash
#!/bin/bash
# API Server Recovery Script

echo "Starting API server recovery..."

# Stop current service
sudo systemctl stop novacron-api

# Check for stuck processes
pkill -f novacron-api

# Clear any lock files
rm -f /var/run/novacron-api.lock

# Check database connectivity
if ! PGPASSWORD=$DB_PASSWORD psql -h localhost -U novacron -d novacron -c "SELECT 1;" > /dev/null 2>&1; then
    echo "Database connection failed, attempting recovery..."
    sudo systemctl restart postgresql
    sleep 10
fi

# Start service
sudo systemctl start novacron-api

# Verify recovery
sleep 5
if curl -f -s http://localhost:8080/health > /dev/null; then
    echo "API server recovery successful"
else
    echo "API server recovery failed"
    exit 1
fi
```

#### Database Recovery
```bash
#!/bin/bash
# Database Recovery Script

echo "Starting database recovery..."

# Check if PostgreSQL is running
if ! systemctl is-active --quiet postgresql; then
    echo "PostgreSQL is not running, starting..."
    sudo systemctl start postgresql
fi

# Check for corruption
sudo -u postgres pg_dump novacron > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Database corruption detected, attempting repair..."
    
    # Stop applications
    sudo systemctl stop novacron-api
    
    # Run integrity check
    sudo -u postgres postgres --single -D /var/lib/postgresql/15/main novacron
    
    # Restart PostgreSQL
    sudo systemctl restart postgresql
    
    # Start applications
    sudo systemctl start novacron-api
fi

echo "Database recovery completed"
```

### 2. Data Recovery

#### VM Data Recovery
```bash
#!/bin/bash
# VM Data Recovery Script

VM_STORAGE="/var/lib/libvirt/images"
BACKUP_LOCATION="/opt/novacron/backups/vm-storage"
VM_ID=$1

if [ -z "$VM_ID" ]; then
    echo "Usage: $0 <vm_id>"
    exit 1
fi

echo "Recovering VM data for: $VM_ID"

# Stop VM if running
virsh destroy $VM_ID 2>/dev/null || true

# Find latest backup
LATEST_BACKUP=$(find $BACKUP_LOCATION -name "*${VM_ID}*" -type f | sort | tail -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "No backup found for VM: $VM_ID"
    exit 1
fi

echo "Using backup: $LATEST_BACKUP"

# Restore VM disk
cp "$LATEST_BACKUP" "$VM_STORAGE/${VM_ID}.qcow2"

# Fix permissions
chown libvirt-qemu:libvirt-qemu "$VM_STORAGE/${VM_ID}.qcow2"

# Restart VM
virsh start $VM_ID

echo "VM recovery completed for: $VM_ID"
```

### 3. Configuration Recovery

#### Configuration Backup and Restore
```bash
#!/bin/bash
# Configuration Recovery Script

CONFIG_DIR="/opt/novacron/config"
BACKUP_DIR="/opt/novacron/backups/config"
ACTION=$1

backup_config() {
    echo "Backing up configuration..."
    timestamp=$(date +"%Y%m%d_%H%M%S")
    backup_path="$BACKUP_DIR/config_backup_$timestamp.tar.gz"
    
    tar -czf "$backup_path" -C "$CONFIG_DIR" .
    echo "Configuration backed up to: $backup_path"
}

restore_config() {
    local backup_file=$2
    if [ -z "$backup_file" ]; then
        # Use latest backup
        backup_file=$(ls -t $BACKUP_DIR/config_backup_*.tar.gz | head -1)
    fi
    
    if [ ! -f "$backup_file" ]; then
        echo "Backup file not found: $backup_file"
        exit 1
    fi
    
    echo "Restoring configuration from: $backup_file"
    
    # Stop services
    sudo systemctl stop novacron-api
    sudo systemctl stop novacron-frontend
    
    # Backup current config
    backup_config
    
    # Restore from backup
    rm -rf "$CONFIG_DIR"/*
    tar -xzf "$backup_file" -C "$CONFIG_DIR"
    
    # Fix permissions
    chown -R novacron:novacron "$CONFIG_DIR"
    chmod 600 "$CONFIG_DIR"/*.yaml
    
    # Start services
    sudo systemctl start novacron-api
    sudo systemctl start novacron-frontend
    
    echo "Configuration restored successfully"
}

case $ACTION in
    "backup")
        backup_config
        ;;
    "restore")
        restore_config $@
        ;;
    *)
        echo "Usage: $0 {backup|restore} [backup_file]"
        exit 1
        ;;
esac
```

## Emergency Procedures

### 1. Critical System Failure

#### Emergency Response Checklist
```
□ Assess impact and scope
□ Activate incident response team
□ Communicate status to stakeholders
□ Implement immediate containment
□ Begin recovery procedures
□ Monitor recovery progress
□ Verify system functionality
□ Document incident details
□ Conduct post-incident review
```

#### Emergency Contact Script
```bash
#!/bin/bash
# Emergency Notification Script

SEVERITY=$1
MESSAGE=$2
INCIDENT_ID=$(date +"%Y%m%d_%H%M%S")

# Notification channels
SLACK_WEBHOOK="https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"
EMAIL_LIST="ops-team@company.com,management@company.com"
SMS_SERVICE="https://api.sms-service.com/send"

# Send Slack notification
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"🚨 CRITICAL ALERT [$INCIDENT_ID]\n**Severity:** $SEVERITY\n**Message:** $MESSAGE\n**Time:** $(date)\"}" \
  $SLACK_WEBHOOK

# Send email notification
echo "Subject: [CRITICAL] NovaCron System Alert - $INCIDENT_ID
From: noreply@novacron.com
To: $EMAIL_LIST

CRITICAL SYSTEM ALERT

Incident ID: $INCIDENT_ID
Severity: $SEVERITY
Message: $MESSAGE
Timestamp: $(date)

Please check the system immediately and follow emergency procedures.

Monitoring Dashboard: https://grafana.novacron.com
Runbooks: https://docs.novacron.com/runbooks
" | sendmail $EMAIL_LIST

echo "Emergency notifications sent for incident: $INCIDENT_ID"
```

### 2. Security Incident Response

#### Security Incident Containment
```bash
#!/bin/bash
# Security Incident Response Script

INCIDENT_TYPE=$1
AFFECTED_RESOURCE=$2

echo "Security incident detected: $INCIDENT_TYPE"
echo "Affected resource: $AFFECTED_RESOURCE"

# Immediate containment actions
case $INCIDENT_TYPE in
    "unauthorized_access")
        echo "Implementing access controls..."
        # Block suspicious IP addresses
        iptables -I INPUT -s $AFFECTED_RESOURCE -j DROP
        
        # Invalidate all sessions
        redis-cli FLUSHDB
        
        # Force password reset for affected accounts
        psql -d novacron -c "UPDATE users SET password_reset_required = true WHERE last_login_ip = '$AFFECTED_RESOURCE';"
        ;;
        
    "data_breach")
        echo "Implementing data protection measures..."
        # Stop data processing
        systemctl stop novacron-api
        
        # Enable audit logging
        sed -i 's/log_statement = .*/log_statement = all/' /etc/postgresql/15/main/postgresql.conf
        systemctl restart postgresql
        
        # Notify relevant parties
        ./notify_data_breach.sh "$AFFECTED_RESOURCE"
        ;;
        
    "malware_detected")
        echo "Implementing malware containment..."
        # Isolate affected system
        iptables -P INPUT DROP
        iptables -P OUTPUT DROP
        iptables -P FORWARD DROP
        
        # Take memory dump for analysis
        dd if=/proc/kcore of=/tmp/memory_dump_$(date +%Y%m%d_%H%M%S).img
        ;;
esac

echo "Containment measures implemented"
echo "Incident ID: SEC_$(date +%Y%m%d_%H%M%S)"
```

## Performance Troubleshooting

### 1. CPU Performance Issues

#### High CPU Usage Analysis
```bash
#!/bin/bash
# CPU Performance Analysis Script

echo "=== CPU Performance Analysis ==="
echo "Current CPU usage:"
top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1

echo -e "\nTop CPU consuming processes:"
ps aux --sort=-%cpu | head -10

echo -e "\nNovaCron API CPU usage:"
top -bn1 -p $(pidof novacron-api) | tail -1

echo -e "\nSystem load averages:"
uptime

echo -e "\nCPU core information:"
lscpu | grep -E "^CPU\(s\)|^Core\(s\)|^Model name"

# Check for CPU throttling
if [ -f /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count ]; then
    echo -e "\nThermal throttling count:"
    cat /sys/devices/system/cpu/cpu*/thermal_throttle/package_throttle_count
fi

# Application-specific analysis
echo -e "\nGo runtime analysis (if pprof available):"
curl -s http://localhost:6060/debug/pprof/goroutine?debug=1 | head -20
```

### 2. Memory Performance Issues

#### Memory Usage Analysis
```bash
#!/bin/bash
# Memory Performance Analysis Script

echo "=== Memory Performance Analysis ==="
echo "Memory usage overview:"
free -h

echo -e "\nTop memory consuming processes:"
ps aux --sort=-%mem | head -10

echo -e "\nNovaCron API memory usage:"
ps -p $(pidof novacron-api) -o pid,ppid,%mem,rss,vsz,comm

echo -e "\nMemory fragmentation:"
grep -E "MemTotal|MemFree|MemAvailable|Buffers|Cached|Slab" /proc/meminfo

echo -e "\nSwap usage:"
swapon -s

# Check for memory leaks in Go applications
echo -e "\nGo heap analysis (if pprof available):"
curl -s http://localhost:6060/debug/pprof/heap?debug=1 | head -20
```

## Monitoring & Alerting Setup

### 1. Custom Alert Rules

#### Prometheus Alert Rules
```yaml
# /etc/prometheus/rules/novacron.yml
groups:
- name: novacron.rules
  rules:
  - alert: APIHighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 2m
    labels:
      severity: warning
      service: novacron-api
    annotations:
      summary: "API response time is high"
      description: "95th percentile response time is {{ $value }}s for 2 minutes"
      runbook_url: "https://docs.novacron.com/runbooks/api-performance"
  
  - alert: DatabaseConnectionHigh
    expr: database_connections_active / database_connections_max > 0.8
    for: 1m
    labels:
      severity: warning
      service: postgresql
    annotations:
      summary: "Database connection usage is high"
      description: "Database connection usage is {{ $value | humanizePercentage }}"
      runbook_url: "https://docs.novacron.com/runbooks/database-connections"
  
  - alert: VMOperationFailure
    expr: increase(vm_operations_total{status="failed"}[5m]) > 5
    for: 0m
    labels:
      severity: critical
      service: vm-manager
    annotations:
      summary: "Multiple VM operations are failing"
      description: "{{ $value }} VM operations have failed in the last 5 minutes"
      runbook_url: "https://docs.novacron.com/runbooks/vm-operations"
```

### 2. Log Monitoring Setup

#### Log-based Alerts
```yaml
# /etc/promtail/config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
- job_name: novacron-logs
  static_configs:
  - targets:
      - localhost
    labels:
      job: novacron-api
      __path__: /var/log/novacron/*.log
  
  pipeline_stages:
  - json:
      expressions:
        level: level
        message: message
        timestamp: timestamp
  
  - labels:
      level:
  
  - match:
      selector: '{level="error"}'
      action: keep
      stages:
      - metrics:
          error_total:
            type: Counter
            description: "Total number of errors"
            config:
              action: inc
```

## Documentation & Knowledge Base

### 1. Runbook Template

#### Standard Runbook Format
```markdown
# [Issue Type] Troubleshooting Runbook

## Overview
Brief description of the issue and its impact.

## Symptoms
- What users/operators experience
- Error messages or indicators
- Performance characteristics

## Impact Assessment
- Business impact level (Critical/High/Medium/Low)
- Affected systems/users
- SLA implications

## Diagnosis Steps
1. Initial checks and data gathering
2. Root cause analysis procedures
3. Verification steps

## Resolution Procedures
### Immediate Actions
- Emergency containment steps
- Temporary workarounds

### Permanent Fix
- Root cause resolution
- Preventive measures
- Configuration changes

## Verification
- How to confirm the fix worked
- Monitoring points to watch
- Success criteria

## Prevention
- Monitoring improvements
- Process changes
- Documentation updates

## Escalation
- When to escalate
- Who to contact
- Required information for escalation

## Related Links
- Monitoring dashboards
- Log locations
- Documentation references
```

### 2. Troubleshooting Decision Tree

```
System Issue Detected
├── Service Down?
│   ├── YES → Check system resources → Restart service → Monitor
│   └── NO → Continue to performance check
├── Performance Issue?
│   ├── YES → Identify bottleneck (CPU/Memory/IO/Network)
│   │   ├── CPU → Check processes → Optimize/Scale
│   │   ├── Memory → Check for leaks → Restart/Optimize
│   │   ├── IO → Check storage → Optimize queries/Add storage
│   │   └── Network → Check connectivity → Fix network issues
│   └── NO → Continue to application check
├── Application Error?
│   ├── YES → Check logs → Identify error type
│   │   ├── Database → Check connections → Fix queries/Scale DB
│   │   ├── Authentication → Check config → Fix auth issues
│   │   ├── VM Operations → Check hypervisor → Fix VM issues
│   │   └── Other → Follow specific troubleshooting guide
│   └── NO → Check monitoring for anomalies
└── No Clear Issue → Gather more data → Escalate if needed
```

---

**Document Classification**: Operational - Support Team  
**Last Updated**: September 2, 2025  
**Version**: 1.0  
**Review Schedule**: Monthly  
**Emergency Contact**: ops-team@novacron.com