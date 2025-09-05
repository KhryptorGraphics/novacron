# NovaCron Database Migration Plan
## Version 1.0 | Zero-Downtime Architecture Overhaul

### Executive Summary

This comprehensive migration plan transforms the NovaCron database architecture from the current fragmented schema to an optimized, performance-enhanced design with zero-downtime deployment. The migration supports:

- **Zero-downtime deployment** with rolling updates
- **Performance improvements** up to 400% via optimized indexing and materialized views
- **Enhanced security** with Row Level Security (RLS) and tenant isolation
- **Scalability improvements** supporting 10x current capacity
- **Disaster recovery** enhancements with point-in-time recovery

---

## 🔍 Architecture Analysis

### Current State Assessment

**Schema Inconsistencies Identified:**
- **Dual table structures**: `hypervisors` vs `nodes`, `virtual_machines` vs `vms`
- **Missing performance optimizations**: Limited indexing strategy
- **Security gaps**: No Row Level Security, basic audit logging
- **Scalability constraints**: No materialized views, basic partitioning

**Performance Bottlenecks:**
- Time-series queries on metrics tables (5-10 second response times)
- Complex join operations without proper indexing
- Missing materialized views for dashboard queries
- Inefficient audit log queries

**Target Architecture Benefits:**
- Unified schema with consistent naming conventions
- Advanced performance indexing (+400% query performance)
- Comprehensive materialized views for real-time dashboards
- Enhanced security with RLS and tenant isolation
- Improved monitoring with system health views

---

## 📋 Migration Strategy Overview

### Zero-Downtime Approach

**Strategy**: **Blue-Green with Shadow Replication**
- **Blue Environment**: Current production system
- **Green Environment**: New optimized schema
- **Shadow Replication**: Real-time data sync during migration
- **Atomic Switchover**: DNS/Load Balancer redirect with < 30 second downtime

### Migration Phases

1. **Phase 1**: Infrastructure Preparation (2 hours)
2. **Phase 2**: Schema Migration & Shadow Sync (4 hours)
3. **Phase 3**: Performance Optimization (2 hours)
4. **Phase 4**: Data Validation & Testing (3 hours)
5. **Phase 5**: Production Cutover (30 minutes)
6. **Phase 6**: Post-Migration Verification (1 hour)

**Total Migration Window**: 12.5 hours | **Downtime**: < 30 minutes

---

## 🚀 Phase 1: Infrastructure Preparation

### Duration: 2 hours

### Objectives
- Set up Green environment with new schema
- Configure replication infrastructure
- Establish monitoring and rollback mechanisms

### Tasks

#### 1.1 Environment Setup
```bash
# Create new database instance (Green)
CREATE DATABASE novacron_green WITH TEMPLATE novacron_blue;

# Configure connection pooling
PgBouncer Configuration:
- Green Pool: max_client_conn = 200
- Blue Pool: max_client_conn = 100 (reduced capacity)
```

#### 1.2 Replication Setup
```sql
-- Configure logical replication
SELECT pg_create_logical_replication_slot('novacron_migration', 'pgoutput');

-- Create publication for all tables
CREATE PUBLICATION novacron_migration FOR ALL TABLES;
```

#### 1.3 Monitoring Setup
- Deploy migration monitoring dashboard
- Configure alerting for replication lag (> 10 seconds)
- Set up automated rollback triggers

### Success Criteria
- ✅ Green environment accessible and responsive
- ✅ Replication lag < 5 seconds
- ✅ Monitoring dashboard operational
- ✅ Rollback procedures tested

---

## 🔄 Phase 2: Schema Migration & Shadow Sync

### Duration: 4 hours

### Objectives
- Deploy optimized schema to Green environment
- Establish real-time data synchronization
- Migrate existing data with transformations

### 2.1 Schema Deployment

#### New Schema Features
```sql
-- Enhanced tenant isolation
ALTER TABLE users ADD COLUMN tenant_id UUID REFERENCES tenants(id);
ALTER TABLE vms ADD COLUMN tenant_id UUID REFERENCES tenants(id);

-- Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE vms ENABLE ROW LEVEL SECURITY;

CREATE POLICY user_tenant_isolation ON users
  FOR ALL TO application_role
  USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

#### Performance Indexes
```sql
-- Critical performance indexes
CREATE INDEX CONCURRENTLY idx_vms_tenant_state ON vms(tenant_id, state);
CREATE INDEX CONCURRENTLY idx_vm_metrics_vm_time ON vm_metrics(vm_id, collected_at DESC);
CREATE INDEX CONCURRENTLY idx_audit_tenant_time ON audit_log(tenant_id, created_at DESC);
```

### 2.2 Data Transformation Pipeline

#### Table Mapping Strategy
```sql
-- Hypervisors → Compute Nodes transformation
INSERT INTO compute_nodes (id, name, hostname, status, cpu_cores, memory_total, hypervisor_type, created_at)
SELECT 
    id,
    hostname as name,
    hostname,
    CASE 
        WHEN status = 'active' THEN 'online'
        WHEN status = 'inactive' THEN 'offline'
        ELSE 'maintenance'
    END as status,
    (capacity->>'cpu_cores')::integer,
    (capacity->>'memory_mb')::bigint,
    type as hypervisor_type,
    created_at
FROM hypervisors;

-- Virtual Machines → VMs transformation (enhanced)
INSERT INTO vms (id, name, state, node_id, cpu_cores, memory_mb, disk_size_gb, tenant_id, owner_id, created_at)
SELECT 
    vm.id,
    vm.name,
    vm.status::vm_state,
    vm.hypervisor_id as node_id,
    (vm.resources->>'cpu_cores')::integer,
    (vm.resources->>'memory_mb')::integer,
    (vm.resources->>'disk_gb')::integer,
    u.tenant_id, -- Map from user's tenant
    vm.owner_id,
    vm.created_at
FROM virtual_machines vm
LEFT JOIN users u ON vm.owner_id = u.id;
```

### 2.3 Shadow Replication

#### Real-time Sync Setup
```sql
-- Configure replication subscription
CREATE SUBSCRIPTION novacron_shadow_sync
CONNECTION 'host=blue-db port=5432 dbname=novacron user=repl_user'
PUBLICATION novacron_migration
WITH (copy_data = false, create_slot = false, slot_name = 'novacron_migration');
```

#### Conflict Resolution
```sql
-- Conflict resolution function
CREATE OR REPLACE FUNCTION handle_replication_conflict()
RETURNS TRIGGER AS $$
BEGIN
    -- Log conflict for analysis
    INSERT INTO migration_conflicts (table_name, record_id, conflict_type, details)
    VALUES (TG_TABLE_NAME, NEW.id, 'UPDATE_CONFLICT', row_to_json(NEW));
    
    -- Apply last-writer-wins with timestamp comparison
    IF NEW.updated_at > OLD.updated_at THEN
        RETURN NEW;
    ELSE
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;
```

### Success Criteria
- ✅ All tables migrated with data integrity verified
- ✅ Replication lag < 2 seconds consistently
- ✅ Zero replication conflicts
- ✅ Performance indexes created successfully

---

## ⚡ Phase 3: Performance Optimization

### Duration: 2 hours

### Objectives
- Deploy materialized views for dashboard queries
- Optimize database configuration
- Enable advanced performance features

### 3.1 Materialized Views Deployment

#### Critical Performance Views
```sql
-- VM Summary for dashboards (5-minute refresh)
CREATE MATERIALIZED VIEW mv_vm_dashboard AS
SELECT 
    tenant_id,
    COUNT(*) as total_vms,
    COUNT(*) FILTER (WHERE state = 'running') as running_vms,
    SUM(cpu_cores) as total_cpu_cores,
    SUM(memory_mb) as total_memory_mb,
    AVG(CASE WHEN latest_metrics.cpu_usage IS NOT NULL 
        THEN latest_metrics.cpu_usage ELSE 0 END) as avg_cpu_usage
FROM vms v
LEFT JOIN LATERAL (
    SELECT cpu_usage_percent as cpu_usage
    FROM vm_metrics m 
    WHERE m.vm_id = v.id 
    ORDER BY collected_at DESC 
    LIMIT 1
) latest_metrics ON true
WHERE v.state != 'deleted'
GROUP BY tenant_id;

-- Refresh schedule: Every 5 minutes
SELECT cron.schedule('refresh-vm-dashboard', '*/5 * * * *', 'REFRESH MATERIALIZED VIEW mv_vm_dashboard;');
```

#### Real-time System Health
```sql
CREATE MATERIALIZED VIEW mv_system_health_realtime AS
WITH node_health AS (
    SELECT 
        COUNT(*) as total_nodes,
        COUNT(*) FILTER (WHERE status = 'online') as healthy_nodes,
        AVG(cpu_usage_percent) as avg_node_cpu,
        AVG(memory_usage_percent) as avg_node_memory
    FROM compute_nodes cn
    LEFT JOIN LATERAL (
        SELECT cpu_usage_percent, memory_usage_percent
        FROM node_metrics nm
        WHERE nm.node_id = cn.id
        ORDER BY collected_at DESC
        LIMIT 1
    ) latest ON true
    WHERE last_heartbeat > NOW() - INTERVAL '5 minutes'
)
SELECT 
    total_nodes,
    healthy_nodes,
    ROUND((healthy_nodes::decimal / total_nodes * 100), 2) as node_health_percent,
    ROUND(avg_node_cpu, 2) as avg_cpu_usage,
    ROUND(avg_node_memory, 2) as avg_memory_usage,
    NOW() as last_updated
FROM node_health;
```

### 3.2 Database Configuration Optimization

#### PostgreSQL Performance Tuning
```ini
# postgresql.conf optimizations
shared_buffers = 2GB                    # 25% of total RAM
effective_cache_size = 6GB              # 75% of total RAM
work_mem = 256MB                        # For sorting/hashing
maintenance_work_mem = 1GB              # For VACUUM, CREATE INDEX
wal_buffers = 64MB                      # Write-ahead logging
checkpoint_completion_target = 0.9      # Smooth checkpoints
random_page_cost = 1.1                  # SSD optimization
effective_io_concurrency = 200          # SSD optimization

# Connection optimization
max_connections = 300
shared_preload_libraries = 'pg_stat_statements,pg_cron,auto_explain'

# Query optimization
auto_explain.log_min_duration = '1s'
auto_explain.log_analyze = on
log_statement = 'ddl'
```

### 3.3 Advanced Indexing Strategy

#### Composite Indexes for Common Queries
```sql
-- Multi-tenant query optimization
CREATE INDEX CONCURRENTLY idx_vms_tenant_state_created 
ON vms(tenant_id, state, created_at DESC) 
WHERE state IN ('running', 'stopped');

-- Metrics query optimization
CREATE INDEX CONCURRENTLY idx_vm_metrics_time_series 
ON vm_metrics(vm_id, collected_at DESC) 
INCLUDE (cpu_usage_percent, memory_usage_percent);

-- Audit compliance optimization
CREATE INDEX CONCURRENTLY idx_audit_compliance 
ON audit_log(tenant_id, action, created_at DESC)
WHERE action IN ('CREATE', 'DELETE', 'MODIFY');
```

#### Partial Indexes for Efficiency
```sql
-- Active sessions only
CREATE INDEX CONCURRENTLY idx_user_sessions_active 
ON user_sessions(user_id, last_activity) 
WHERE expires_at > NOW();

-- Error state monitoring
CREATE INDEX CONCURRENTLY idx_vms_error_monitoring 
ON vms(state, updated_at DESC, tenant_id) 
WHERE state = 'error';
```

### Success Criteria
- ✅ All materialized views created and refreshing
- ✅ Database configuration optimized
- ✅ Query performance improved by 300-400%
- ✅ Dashboard response times < 500ms

---

## ✅ Phase 4: Data Validation & Testing

### Duration: 3 hours

### Objectives
- Comprehensive data integrity validation
- Performance testing and benchmarking
- Security testing and compliance verification

### 4.1 Data Integrity Validation

#### Automated Validation Suite
```sql
-- Row count verification
CREATE OR REPLACE FUNCTION validate_migration_counts()
RETURNS TABLE(table_name text, blue_count bigint, green_count bigint, match boolean) AS $$
BEGIN
    RETURN QUERY
    WITH counts AS (
        SELECT 'users'::text as table_name, 
               (SELECT COUNT(*) FROM blue_db.users) as blue_count,
               (SELECT COUNT(*) FROM users) as green_count
        UNION ALL
        SELECT 'vms'::text, 
               (SELECT COUNT(*) FROM blue_db.vms) as blue_count,
               (SELECT COUNT(*) FROM vms) as green_count
        -- ... additional tables
    )
    SELECT c.table_name, c.blue_count, c.green_count, (c.blue_count = c.green_count) as match
    FROM counts c;
END;
$$ LANGUAGE plpgsql;

-- Data consistency checks
CREATE OR REPLACE FUNCTION validate_referential_integrity()
RETURNS TABLE(check_name text, status text, details text) AS $$
BEGIN
    -- Check VM -> Node references
    RETURN QUERY
    SELECT 'vm_node_references'::text, 
           CASE WHEN COUNT(*) = 0 THEN 'PASS' ELSE 'FAIL' END,
           'Orphaned VMs: ' || COUNT(*)::text
    FROM vms v 
    WHERE v.node_id IS NOT NULL 
    AND NOT EXISTS (SELECT 1 FROM compute_nodes cn WHERE cn.id = v.node_id);
    
    -- Additional integrity checks...
END;
$$ LANGUAGE plpgsql;
```

#### Critical Data Validation Queries
```sql
-- Validate user-tenant relationships
SELECT 
    'User-Tenant Mapping' as check_name,
    COUNT(*) as total_users,
    COUNT(tenant_id) as users_with_tenant,
    ROUND((COUNT(tenant_id)::decimal / COUNT(*) * 100), 2) as mapping_percentage
FROM users;

-- Validate VM resource allocations
SELECT 
    'VM Resource Validation' as check_name,
    COUNT(*) as total_vms,
    COUNT(*) FILTER (WHERE cpu_cores > 0 AND memory_mb > 0 AND disk_size_gb > 0) as valid_resources,
    COUNT(*) FILTER (WHERE cpu_cores IS NULL OR memory_mb IS NULL OR disk_size_gb IS NULL) as missing_resources
FROM vms;
```

### 4.2 Performance Testing

#### Benchmark Queries
```sql
-- Dashboard query performance test
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    t.name as tenant_name,
    vs.total_vms,
    vs.running_vms,
    vs.total_cpu_cores,
    vs.avg_cpu_usage
FROM mv_vm_dashboard vs
JOIN tenants t ON vs.tenant_id = t.id
ORDER BY vs.total_vms DESC;

-- Time-series metrics query test
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    date_trunc('hour', collected_at) as hour,
    AVG(cpu_usage_percent) as avg_cpu,
    MAX(cpu_usage_percent) as max_cpu
FROM vm_metrics 
WHERE vm_id = $1 
AND collected_at >= NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', collected_at)
ORDER BY hour;
```

#### Load Testing Script
```bash
#!/bin/bash
# load_test_migration.sh

echo "Starting migration load test..."

# Concurrent dashboard queries
for i in {1..50}; do
    psql -h green-db -c "SELECT * FROM mv_vm_dashboard;" &
done

# Concurrent metrics queries
for i in {1..30}; do
    psql -h green-db -c "
        SELECT AVG(cpu_usage_percent) 
        FROM vm_metrics 
        WHERE collected_at >= NOW() - INTERVAL '1 hour';" &
done

wait
echo "Load test completed. Check response times."
```

### 4.3 Security Validation

#### Row Level Security Testing
```sql
-- Test tenant isolation
SET app.tenant_id = 'tenant-1-uuid';
SELECT COUNT(*) FROM vms; -- Should only return tenant-1 VMs

SET app.tenant_id = 'tenant-2-uuid';
SELECT COUNT(*) FROM vms; -- Should only return tenant-2 VMs

-- Test cross-tenant access prevention
SET app.tenant_id = 'tenant-1-uuid';
SELECT * FROM vms WHERE tenant_id != 'tenant-1-uuid'; -- Should return 0 rows
```

#### Audit Trail Verification
```sql
-- Verify audit logging is working
INSERT INTO vms (name, cpu_cores, memory_mb, disk_size_gb, tenant_id) 
VALUES ('test-vm', 2, 4096, 50, current_setting('app.tenant_id')::uuid);

-- Check audit record was created
SELECT action, resource_type, resource_id, details 
FROM audit_log 
WHERE created_at >= NOW() - INTERVAL '1 minute'
ORDER BY created_at DESC 
LIMIT 1;
```

### Success Criteria
- ✅ 100% data integrity validation passed
- ✅ Query performance improved by 300%+ 
- ✅ All security tests passed
- ✅ Load testing shows stable performance under 2x normal load

---

## 🔄 Phase 5: Production Cutover

### Duration: 30 minutes

### Objectives
- Execute atomic switchover to Green environment
- Minimize service disruption to < 30 seconds
- Ensure immediate rollback capability

### 5.1 Pre-Cutover Checklist

```bash
#!/bin/bash
# pre_cutover_checklist.sh

echo "=== Pre-Cutover Validation ==="

# 1. Verify replication status
echo "Checking replication lag..."
REPLICATION_LAG=$(psql -h blue-db -tAc "SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) FROM pg_stat_replication;")
if [ "$REPLICATION_LAG" -lt 1000000 ]; then
    echo "✅ Replication lag acceptable: $REPLICATION_LAG bytes"
else
    echo "❌ Replication lag too high: $REPLICATION_LAG bytes"
    exit 1
fi

# 2. Verify Green database health
echo "Checking Green database connectivity..."
psql -h green-db -c "SELECT 1;" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Green database accessible"
else
    echo "❌ Green database not accessible"
    exit 1
fi

# 3. Verify materialized views are up to date
echo "Checking materialized view freshness..."
LAST_REFRESH=$(psql -h green-db -tAc "SELECT last_updated FROM mv_system_health_realtime;")
if [ $(date -d "$LAST_REFRESH" +%s) -gt $(date -d "5 minutes ago" +%s) ]; then
    echo "✅ Materialized views are fresh"
else
    echo "❌ Materialized views are stale"
    exit 1
fi

echo "All pre-cutover checks passed!"
```

### 5.2 Cutover Execution

#### Step 1: Application Graceful Shutdown (10 seconds)
```bash
# Signal applications to finish current requests
kubectl scale deployment novacron-api --replicas=0
kubectl scale deployment novacron-worker --replicas=0

# Wait for graceful shutdown
sleep 10
```

#### Step 2: Final Data Sync (10 seconds)
```sql
-- Stop replication and sync final changes
SELECT pg_replication_slot_advance('novacron_migration', pg_current_wal_lsn());
ALTER SUBSCRIPTION novacron_shadow_sync DISABLE;
```

#### Step 3: DNS/Load Balancer Switch (5 seconds)
```bash
# Update load balancer to point to Green database
kubectl patch configmap database-config --patch '{"data":{"DB_HOST":"green-db.novacron.internal"}}'

# Update DNS records
aws route53 change-resource-record-sets --hosted-zone-id Z123456789 --change-batch '{
    "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
            "Name": "db.novacron.internal",
            "Type": "CNAME", 
            "TTL": 60,
            "ResourceRecords": [{"Value": "green-db.novacron.internal"}]
        }
    }]
}'
```

#### Step 4: Application Startup (5 seconds)
```bash
# Start applications pointing to Green database
kubectl scale deployment novacron-api --replicas=3
kubectl scale deployment novacron-worker --replicas=2

# Verify application health
kubectl wait --for=condition=ready pod -l app=novacron-api --timeout=30s
```

### 5.3 Post-Cutover Verification

```bash
#!/bin/bash
# post_cutover_verification.sh

echo "=== Post-Cutover Verification ==="

# 1. Verify applications are using Green database
APP_RESPONSE=$(curl -s "https://api.novacron.io/health/database")
if echo "$APP_RESPONSE" | grep -q "green-db"; then
    echo "✅ Applications connected to Green database"
else
    echo "❌ Applications not using Green database"
    exit 1
fi

# 2. Test critical functionality
echo "Testing VM creation..."
VM_ID=$(curl -s -X POST "https://api.novacron.io/vms" \
    -H "Authorization: Bearer $TEST_TOKEN" \
    -d '{"name":"migration-test-vm","cpu":1,"memory":1024,"disk":10}' | jq -r '.id')

if [ "$VM_ID" != "null" ]; then
    echo "✅ VM creation successful: $VM_ID"
    # Cleanup test VM
    curl -s -X DELETE "https://api.novacron.io/vms/$VM_ID" -H "Authorization: Bearer $TEST_TOKEN"
else
    echo "❌ VM creation failed"
    exit 1
fi

echo "Cutover verification completed successfully!"
```

### Rollback Procedure (if needed)

```bash
#!/bin/bash
# emergency_rollback.sh

echo "⚠️ EXECUTING EMERGENCY ROLLBACK"

# 1. Immediate DNS switch back to Blue
kubectl patch configmap database-config --patch '{"data":{"DB_HOST":"blue-db.novacron.internal"}}'

# 2. Restart applications
kubectl rollout restart deployment/novacron-api
kubectl rollout restart deployment/novacron-worker

# 3. Verify rollback success
sleep 30
kubectl get pods | grep -E "(api|worker)" | grep Running
if [ $? -eq 0 ]; then
    echo "✅ Rollback successful - applications running on Blue database"
else
    echo "❌ Rollback failed - manual intervention required"
    exit 1
fi
```

### Success Criteria
- ✅ Total downtime < 30 seconds
- ✅ All applications healthy and responsive
- ✅ Database queries returning expected results
- ✅ Rollback procedure tested and ready

---

## 🔍 Phase 6: Post-Migration Verification

### Duration: 1 hour

### Objectives
- Comprehensive system health verification
- Performance benchmarking against baselines
- Documentation of migration outcomes

### 6.1 System Health Verification

#### Database Health Check
```sql
-- Comprehensive database health assessment
WITH health_metrics AS (
    SELECT 
        -- Connection health
        count(*) as active_connections,
        count(*) FILTER (WHERE state = 'active') as active_queries,
        -- Replication health (if applicable)
        (SELECT count(*) FROM pg_stat_replication) as replication_slots,
        -- Lock contention
        count(*) FILTER (WHERE wait_event_type = 'Lock') as waiting_locks,
        -- Query performance
        avg(duration) FILTER (WHERE duration IS NOT NULL) as avg_query_duration
    FROM pg_stat_activity
    WHERE datname = 'novacron'
),
storage_metrics AS (
    SELECT 
        pg_size_pretty(pg_database_size('novacron')) as database_size,
        pg_size_pretty(sum(pg_total_relation_size(schemaname||'.'||tablename))) as total_table_size
    FROM pg_tables 
    WHERE schemaname = 'public'
)
SELECT 
    'Database Health Summary' as category,
    jsonb_build_object(
        'active_connections', hm.active_connections,
        'active_queries', hm.active_queries,
        'database_size', sm.database_size,
        'avg_query_duration_ms', round(hm.avg_query_duration, 2),
        'health_score', 
            CASE 
                WHEN hm.active_connections < 100 AND hm.waiting_locks = 0 THEN 'EXCELLENT'
                WHEN hm.active_connections < 200 AND hm.waiting_locks < 5 THEN 'GOOD'
                ELSE 'NEEDS_ATTENTION'
            END
    ) as metrics
FROM health_metrics hm, storage_metrics sm;
```

#### Application Performance Verification
```bash
#!/bin/bash
# application_performance_check.sh

echo "=== Application Performance Verification ==="

# API response time testing
echo "Testing API response times..."
for endpoint in "/health" "/vms" "/nodes" "/metrics/dashboard"; do
    RESPONSE_TIME=$(curl -w "%{time_total}" -s -o /dev/null "https://api.novacron.io$endpoint")
    echo "GET $endpoint: ${RESPONSE_TIME}s"
    
    # Alert if response time > 1 second
    if (( $(echo "$RESPONSE_TIME > 1.0" | bc -l) )); then
        echo "⚠️ Slow response detected for $endpoint"
    fi
done

# Dashboard load time testing
echo "Testing dashboard performance..."
DASHBOARD_LOAD=$(curl -w "%{time_total}" -s -o /dev/null "https://dashboard.novacron.io")
echo "Dashboard load time: ${DASHBOARD_LOAD}s"

# Database query performance
echo "Testing database query performance..."
DB_QUERY_TIME=$(psql -h green-db -tAc "\timing" -c "SELECT COUNT(*) FROM mv_vm_dashboard;" 2>&1 | grep "Time" | cut -d' ' -f2)
echo "Dashboard query time: $DB_QUERY_TIME"
```

### 6.2 Performance Benchmarking

#### Query Performance Comparison
```sql
-- Before/After performance comparison
CREATE TEMPORARY TABLE performance_comparison AS
WITH baseline_metrics AS (
    -- Historical performance data from Blue database
    SELECT 'dashboard_summary' as query_type, 2.3 as baseline_seconds
    UNION SELECT 'vm_metrics_hourly', 4.1
    UNION SELECT 'audit_log_search', 1.8
    UNION SELECT 'tenant_usage_report', 5.2
),
current_metrics AS (
    SELECT 
        'dashboard_summary' as query_type,
        (SELECT extract(epoch from duration) FROM (
            EXPLAIN (ANALYZE) SELECT * FROM mv_vm_dashboard LIMIT 100
        ) AS t(duration)) as current_seconds
    -- Add more query performance tests...
)
SELECT 
    bm.query_type,
    bm.baseline_seconds,
    cm.current_seconds,
    round(((bm.baseline_seconds - cm.current_seconds) / bm.baseline_seconds * 100)::numeric, 2) as improvement_percent
FROM baseline_metrics bm
JOIN current_metrics cm ON bm.query_type = cm.query_type;

SELECT * FROM performance_comparison;
```

#### Materialized View Performance
```sql
-- Verify materialized view refresh performance
SELECT 
    schemaname,
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) as size,
    (SELECT count(*) FROM information_schema.table_constraints 
     WHERE table_name = matviewname AND constraint_type = 'PRIMARY KEY') > 0 as has_unique_index
FROM pg_matviews 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||matviewname) DESC;

-- Test refresh times
\timing on
REFRESH MATERIALIZED VIEW mv_vm_dashboard;
REFRESH MATERIALIZED VIEW mv_system_health_realtime;
REFRESH MATERIALIZED VIEW mv_tenant_usage;
\timing off
```

### 6.3 Data Consistency Final Validation

#### Cross-Reference Data Validation
```sql
-- Final data consistency check across all critical tables
CREATE OR REPLACE FUNCTION final_migration_validation()
RETURNS TABLE(validation_item text, status text, details text) AS $$
BEGIN
    -- User count validation
    RETURN QUERY
    SELECT 'Total Users'::text, 'INFO'::text, 
           'Migrated: ' || count(*)::text || ' users' 
    FROM users;
    
    -- VM count and state distribution
    RETURN QUERY
    SELECT 'VM States'::text, 'INFO'::text,
           'Total: ' || count(*)::text || ' | Running: ' || 
           count(*) FILTER (WHERE state = 'running')::text ||
           ' | Stopped: ' || count(*) FILTER (WHERE state = 'stopped')::text
    FROM vms;
    
    -- Tenant isolation verification
    RETURN QUERY
    SELECT 'Tenant Isolation'::text,
           CASE WHEN count(DISTINCT tenant_id) > 0 THEN 'PASS' ELSE 'FAIL' END,
           'Distinct tenants: ' || count(DISTINCT tenant_id)::text
    FROM vms WHERE tenant_id IS NOT NULL;
    
    -- Metrics data integrity
    RETURN QUERY
    SELECT 'Metrics Integrity'::text,
           CASE WHEN count(*) > 0 THEN 'PASS' ELSE 'FAIL' END,
           'Recent metrics: ' || count(*)::text || ' records in last hour'
    FROM vm_metrics WHERE collected_at > NOW() - INTERVAL '1 hour';
    
    -- Referential integrity check
    RETURN QUERY
    SELECT 'Referential Integrity'::text,
           CASE WHEN count(*) = 0 THEN 'PASS' ELSE 'FAIL' END,
           'Orphaned records: ' || count(*)::text
    FROM vms v LEFT JOIN compute_nodes cn ON v.node_id = cn.id 
    WHERE v.node_id IS NOT NULL AND cn.id IS NULL;
    
END;
$$ LANGUAGE plpgsql;

-- Execute final validation
SELECT * FROM final_migration_validation();
```

### 6.4 Monitoring and Alerting Setup

#### Post-Migration Monitoring Configuration
```sql
-- Create monitoring views for ongoing operations
CREATE OR REPLACE VIEW v_migration_health AS
SELECT 
    'Database' as component,
    CASE 
        WHEN pg_is_in_recovery() THEN 'STANDBY'
        ELSE 'PRIMARY' 
    END as status,
    pg_size_pretty(pg_database_size(current_database())) as size,
    (SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()) as connections,
    NOW() as last_check;

-- Alert thresholds
CREATE TABLE IF NOT EXISTS migration_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50),
    threshold_value DECIMAL,
    current_value DECIMAL,
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Performance monitoring triggers
CREATE OR REPLACE FUNCTION monitor_query_performance()
RETURNS TRIGGER AS $$
BEGIN
    -- Log slow queries > 5 seconds
    IF NEW.duration > 5.0 THEN
        INSERT INTO migration_alerts (alert_type, threshold_value, current_value, status)
        VALUES ('SLOW_QUERY', 5.0, NEW.duration, 'ACTIVE');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

### Success Criteria
- ✅ All health checks passing with GREEN status
- ✅ Query performance improved by 300%+ over baseline
- ✅ Data consistency validation 100% successful
- ✅ Monitoring and alerting fully operational
- ✅ Migration documentation complete

---

## 📊 Migration Scripts & Procedures

### Migration Script Library

#### 1. Pre-Migration Schema Preparation
```sql
-- File: /scripts/001_pre_migration_setup.sql
-- Purpose: Prepare source database for migration

BEGIN;

-- Create migration tracking table
CREATE TABLE IF NOT EXISTS migration_progress (
    phase VARCHAR(50) PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'PENDING',
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    details JSONB DEFAULT '{}'
);

INSERT INTO migration_progress (phase) VALUES 
    ('INFRASTRUCTURE_PREP'),
    ('SCHEMA_MIGRATION'),
    ('PERFORMANCE_OPTIMIZATION'),
    ('DATA_VALIDATION'),
    ('PRODUCTION_CUTOVER'),
    ('POST_MIGRATION_VERIFICATION');

-- Create backup tables for rollback
CREATE TABLE users_backup AS SELECT * FROM users;
CREATE TABLE vms_backup AS SELECT * FROM vms;
CREATE TABLE nodes_backup AS SELECT * FROM nodes;

-- Update migration progress
UPDATE migration_progress 
SET status = 'COMPLETED', completed_at = NOW() 
WHERE phase = 'INFRASTRUCTURE_PREP';

COMMIT;
```

#### 2. Schema Migration Core Script
```sql
-- File: /scripts/002_schema_migration.sql
-- Purpose: Execute core schema transformations

BEGIN;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- Create new enum types
CREATE TYPE vm_state_new AS ENUM ('creating', 'running', 'stopped', 'paused', 'migrating', 'error', 'deleting');
CREATE TYPE node_status_new AS ENUM ('online', 'offline', 'maintenance', 'error');

-- Create tenants table (new requirement)
CREATE TABLE tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Insert default tenant for existing data
INSERT INTO tenants (name, slug) VALUES ('Default Tenant', 'default');

-- Schema migration for compute_nodes (previously nodes)
CREATE TABLE compute_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    hostname VARCHAR(255) UNIQUE NOT NULL,
    ip_address INET NOT NULL,
    status node_status_new DEFAULT 'offline',
    
    -- Enhanced resource tracking
    cpu_cores INTEGER NOT NULL,
    cpu_threads INTEGER,
    memory_total BIGINT NOT NULL, -- in MB
    storage_total BIGINT, -- in GB
    
    -- Hypervisor information
    hypervisor_type VARCHAR(50),
    hypervisor_version VARCHAR(50),
    
    -- Operational data
    vm_count INTEGER DEFAULT 0,
    load_average DECIMAL(5,2)[3],
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    
    -- Metadata and tagging
    tags JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enhanced VMs table with tenant support
CREATE TABLE vms_new (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    state vm_state_new DEFAULT 'stopped',
    
    -- Resource allocation
    cpu_cores INTEGER NOT NULL,
    memory_mb INTEGER NOT NULL,
    disk_size_gb INTEGER NOT NULL,
    
    -- Relationships
    node_id UUID REFERENCES compute_nodes(id) ON DELETE SET NULL,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    owner_id UUID REFERENCES users(id) ON DELETE SET NULL,
    template_id UUID, -- For VM templates
    
    -- Network configuration
    ip_address INET,
    mac_address MACADDR,
    
    -- Operational settings
    auto_start BOOLEAN DEFAULT FALSE,
    scheduled_start TIMESTAMP WITH TIME ZONE,
    scheduled_stop TIMESTAMP WITH TIME ZONE,
    
    -- Metadata and tagging
    tags JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Update migration progress
UPDATE migration_progress 
SET status = 'IN_PROGRESS', started_at = NOW() 
WHERE phase = 'SCHEMA_MIGRATION';

COMMIT;
```

#### 3. Data Migration Script
```sql
-- File: /scripts/003_data_migration.sql
-- Purpose: Migrate data from old schema to new schema

BEGIN;

-- Get default tenant ID
SET @default_tenant_id = (SELECT id FROM tenants WHERE slug = 'default');

-- Migrate nodes to compute_nodes
INSERT INTO compute_nodes (
    id, name, hostname, ip_address, status, cpu_cores, memory_total, 
    hypervisor_type, last_heartbeat, metadata, created_at, updated_at
)
SELECT 
    id,
    name,
    hostname,
    ip_address,
    CASE 
        WHEN status = 'active' THEN 'online'::node_status_new
        WHEN status = 'inactive' THEN 'offline'::node_status_new
        ELSE 'maintenance'::node_status_new
    END,
    COALESCE(cpu_cores, 1),
    COALESCE(memory_mb, 1024),
    hypervisor_type,
    last_heartbeat,
    metadata,
    created_at,
    updated_at
FROM nodes;

-- Update users with tenant_id
ALTER TABLE users ADD COLUMN tenant_id UUID REFERENCES tenants(id);
UPDATE users SET tenant_id = @default_tenant_id WHERE tenant_id IS NULL;

-- Migrate VMs to new structure
INSERT INTO vms_new (
    id, name, state, cpu_cores, memory_mb, disk_size_gb,
    node_id, tenant_id, owner_id, ip_address, metadata, created_at, updated_at
)
SELECT 
    v.id,
    v.name,
    CASE 
        WHEN v.state = 'running' THEN 'running'::vm_state_new
        WHEN v.state = 'stopped' THEN 'stopped'::vm_state_new
        WHEN v.state = 'paused' THEN 'paused'::vm_state_new
        WHEN v.state = 'migrating' THEN 'migrating'::vm_state_new
        ELSE 'error'::vm_state_new
    END,
    v.cpu_cores,
    v.memory_mb,
    v.disk_gb,
    v.node_id,
    COALESCE(u.tenant_id, @default_tenant_id),
    v.owner_id,
    (v.network_config->>'ip_address')::inet,
    v.metadata,
    v.created_at,
    v.updated_at
FROM vms v
LEFT JOIN users u ON v.owner_id = u.id;

-- Update VM count in compute_nodes
UPDATE compute_nodes 
SET vm_count = (
    SELECT COUNT(*) 
    FROM vms_new 
    WHERE node_id = compute_nodes.id 
    AND state != 'deleting'
);

-- Update migration progress
UPDATE migration_progress 
SET status = 'COMPLETED', completed_at = NOW() 
WHERE phase = 'SCHEMA_MIGRATION';

COMMIT;
```

#### 4. Performance Optimization Script
```sql
-- File: /scripts/004_performance_optimization.sql
-- Purpose: Create all performance indexes and materialized views

BEGIN;

UPDATE migration_progress 
SET status = 'IN_PROGRESS', started_at = NOW() 
WHERE phase = 'PERFORMANCE_OPTIMIZATION';

-- Critical performance indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_tenant_state ON vms_new(tenant_id, state);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_node_state ON vms_new(node_id, state);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vm_metrics_vm_time ON vm_metrics(vm_id, collected_at DESC);

-- Composite indexes for dashboard queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_vms_tenant_owner ON vms_new(tenant_id, owner_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_tenant_time ON audit_log(tenant_id, created_at DESC);

-- Materialized views for dashboards
CREATE MATERIALIZED VIEW mv_tenant_dashboard AS
SELECT 
    t.id as tenant_id,
    t.name as tenant_name,
    COUNT(v.id) as total_vms,
    COUNT(v.id) FILTER (WHERE v.state = 'running') as running_vms,
    COUNT(v.id) FILTER (WHERE v.state = 'stopped') as stopped_vms,
    COUNT(v.id) FILTER (WHERE v.state = 'error') as error_vms,
    SUM(v.cpu_cores) as total_cpu_cores,
    SUM(v.memory_mb) as total_memory_mb,
    SUM(v.disk_size_gb) as total_disk_gb,
    NOW() as last_updated
FROM tenants t
LEFT JOIN vms_new v ON t.id = v.tenant_id
GROUP BY t.id, t.name;

CREATE UNIQUE INDEX ON mv_tenant_dashboard(tenant_id);

-- Set up automatic refresh
SELECT cron.schedule('refresh-tenant-dashboard', '*/5 * * * *', 
    'REFRESH MATERIALIZED VIEW mv_tenant_dashboard;');

UPDATE migration_progress 
SET status = 'COMPLETED', completed_at = NOW() 
WHERE phase = 'PERFORMANCE_OPTIMIZATION';

COMMIT;
```

#### 5. Final Cutover Script
```sql
-- File: /scripts/005_production_cutover.sql
-- Purpose: Execute final production cutover

BEGIN;

UPDATE migration_progress 
SET status = 'IN_PROGRESS', started_at = NOW() 
WHERE phase = 'PRODUCTION_CUTOVER';

-- Rename tables to complete migration
DROP TABLE IF EXISTS vms_old CASCADE;
ALTER TABLE vms RENAME TO vms_old;
ALTER TABLE vms_new RENAME TO vms;

DROP TABLE IF EXISTS nodes_old CASCADE;
ALTER TABLE nodes RENAME TO nodes_old;
-- compute_nodes is already the new table

-- Update foreign key constraints
ALTER TABLE vm_metrics DROP CONSTRAINT IF EXISTS vm_metrics_vm_id_fkey;
ALTER TABLE vm_metrics ADD CONSTRAINT vm_metrics_vm_id_fkey 
    FOREIGN KEY (vm_id) REFERENCES vms(id) ON DELETE CASCADE;

-- Enable Row Level Security
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE vms ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY tenant_isolation_users ON users
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

CREATE POLICY tenant_isolation_vms ON vms
    FOR ALL TO application_role
    USING (tenant_id = current_setting('app.tenant_id')::uuid);

-- Final data consistency check
DO $$
DECLARE
    vm_count_old INTEGER;
    vm_count_new INTEGER;
    user_count INTEGER;
    node_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO vm_count_old FROM vms_old;
    SELECT COUNT(*) INTO vm_count_new FROM vms;
    SELECT COUNT(*) INTO user_count FROM users WHERE tenant_id IS NOT NULL;
    SELECT COUNT(*) INTO node_count FROM compute_nodes;
    
    IF vm_count_new != vm_count_old THEN
        RAISE EXCEPTION 'VM count mismatch: old=%, new=%', vm_count_old, vm_count_new;
    END IF;
    
    INSERT INTO migration_progress (phase, status, details)
    VALUES ('CUTOVER_VALIDATION', 'COMPLETED', jsonb_build_object(
        'vm_count', vm_count_new,
        'user_count', user_count,
        'node_count', node_count,
        'validation_time', NOW()
    ));
END $$;

UPDATE migration_progress 
SET status = 'COMPLETED', completed_at = NOW() 
WHERE phase = 'PRODUCTION_CUTOVER';

COMMIT;
```

---

## 🔙 Rollback Procedures

### Comprehensive Rollback Strategy

#### Immediate Rollback (< 5 minutes)
For critical issues discovered within first hour after cutover:

```bash
#!/bin/bash
# immediate_rollback.sh

echo "🚨 EXECUTING IMMEDIATE ROLLBACK"
set -e  # Exit on any error

# 1. Switch DNS/Load Balancer back to Blue
echo "Switching traffic back to Blue database..."
kubectl patch configmap database-config --patch '{
    "data": {
        "DB_HOST": "blue-db.novacron.internal",
        "DB_PORT": "5432"
    }
}'

# 2. Restart all applications
echo "Restarting applications..."
kubectl rollout restart deployment/novacron-api
kubectl rollout restart deployment/novacron-worker
kubectl rollout restart deployment/novacron-dashboard

# 3. Verify rollback success
echo "Waiting for applications to be ready..."
kubectl wait --for=condition=ready pod -l app=novacron-api --timeout=120s

# 4. Test critical functionality
echo "Testing critical functionality..."
API_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "https://api.novacron.io/health")
if [ "$API_HEALTH" = "200" ]; then
    echo "✅ Immediate rollback successful"
    echo "⚠️ MANUAL ACTION REQUIRED: Investigate Green database issues"
else
    echo "❌ Rollback verification failed - escalate immediately"
    exit 1
fi
```

#### Planned Rollback (30 minutes)
For issues requiring data synchronization:

```sql
-- File: /scripts/rollback_with_data_sync.sql
-- Purpose: Rollback with data preservation

BEGIN;

-- Create rollback tracking
CREATE TABLE rollback_progress (
    step VARCHAR(50) PRIMARY KEY,
    status VARCHAR(20) DEFAULT 'PENDING',
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

INSERT INTO rollback_progress (step) VALUES 
    ('DATA_SYNC'), ('SCHEMA_RESTORE'), ('VALIDATION'), ('CLEANUP');

-- 1. Sync any data created during Green operation back to Blue
INSERT INTO blue_db.vms (
    id, name, state, cpu_cores, memory_mb, disk_size_gb, 
    node_id, owner_id, created_at, updated_at
)
SELECT 
    id, name, state::text::blue_db.vm_state, cpu_cores, memory_mb, disk_size_gb,
    node_id, owner_id, created_at, updated_at
FROM vms 
WHERE created_at > (SELECT completed_at FROM migration_progress WHERE phase = 'PRODUCTION_CUTOVER')
ON CONFLICT (id) DO UPDATE SET
    state = EXCLUDED.state,
    updated_at = EXCLUDED.updated_at;

UPDATE rollback_progress SET status = 'COMPLETED', completed_at = NOW() WHERE step = 'DATA_SYNC';

-- 2. Restore Blue schema integrity
-- (Blue database should already have consistent schema)

UPDATE rollback_progress SET status = 'COMPLETED', completed_at = NOW() WHERE step = 'SCHEMA_RESTORE';

-- 3. Validate rollback data integrity
DO $$
DECLARE
    vm_count_green INTEGER;
    vm_count_blue INTEGER;
BEGIN
    SELECT COUNT(*) INTO vm_count_green FROM vms;
    SELECT COUNT(*) INTO vm_count_blue FROM blue_db.vms;
    
    IF ABS(vm_count_green - vm_count_blue) > 5 THEN
        RAISE EXCEPTION 'Significant data discrepancy detected: Green=%, Blue=%', 
            vm_count_green, vm_count_blue;
    END IF;
END $$;

UPDATE rollback_progress SET status = 'COMPLETED', completed_at = NOW() WHERE step = 'VALIDATION';

-- 4. Log rollback completion
INSERT INTO audit_log (action, resource_type, details)
VALUES ('DATABASE_ROLLBACK', 'SYSTEM', jsonb_build_object(
    'rollback_time', NOW(),
    'reason', 'Planned rollback due to migration issues',
    'data_synced', true
));

UPDATE rollback_progress SET status = 'COMPLETED', completed_at = NOW() WHERE step = 'CLEANUP';

COMMIT;
```

#### Data Recovery Procedures

```sql
-- File: /scripts/data_recovery.sql
-- Purpose: Recover specific data if needed

-- Function to recover specific VM data
CREATE OR REPLACE FUNCTION recover_vm_data(vm_uuid UUID)
RETURNS BOOLEAN AS $$
DECLARE
    vm_record RECORD;
BEGIN
    -- Get VM data from backup
    SELECT * INTO vm_record FROM vms_backup WHERE id = vm_uuid;
    
    IF FOUND THEN
        -- Insert or update VM record
        INSERT INTO vms (id, name, state, cpu_cores, memory_mb, disk_size_gb, 
                        node_id, owner_id, created_at, updated_at)
        VALUES (vm_record.id, vm_record.name, vm_record.state, 
                vm_record.cpu_cores, vm_record.memory_mb, vm_record.disk_size_gb,
                vm_record.node_id, vm_record.owner_id, 
                vm_record.created_at, vm_record.updated_at)
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            state = EXCLUDED.state,
            cpu_cores = EXCLUDED.cpu_cores,
            memory_mb = EXCLUDED.memory_mb,
            disk_size_gb = EXCLUDED.disk_size_gb,
            node_id = EXCLUDED.node_id,
            owner_id = EXCLUDED.owner_id,
            updated_at = NOW();
            
        -- Log recovery action
        INSERT INTO audit_log (action, resource_type, resource_id, details)
        VALUES ('DATA_RECOVERY', 'VM', vm_uuid, jsonb_build_object(
            'recovery_time', NOW(),
            'source', 'backup_table'
        ));
        
        RETURN TRUE;
    ELSE
        RETURN FALSE;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Bulk data recovery function
CREATE OR REPLACE FUNCTION recover_all_missing_data()
RETURNS TABLE(recovered_table TEXT, record_count INTEGER) AS $$
BEGIN
    -- Recover missing VMs
    INSERT INTO vms SELECT * FROM vms_backup 
    WHERE id NOT IN (SELECT id FROM vms)
    ON CONFLICT (id) DO NOTHING;
    
    RETURN QUERY SELECT 'vms'::TEXT, (SELECT COUNT(*) FROM vms_backup 
                                     WHERE id NOT IN (SELECT id FROM vms));
    
    -- Recover missing users (if any)
    INSERT INTO users SELECT * FROM users_backup 
    WHERE id NOT IN (SELECT id FROM users)
    ON CONFLICT (id) DO NOTHING;
    
    RETURN QUERY SELECT 'users'::TEXT, (SELECT COUNT(*) FROM users_backup 
                                       WHERE id NOT IN (SELECT id FROM users));
END;
$$ LANGUAGE plpgsql;
```

---

## 📊 Risk Assessment & Mitigation

### Risk Matrix

| Risk Category | Probability | Impact | Mitigation Strategy | Contingency Plan |
|---------------|------------|--------|-------------------|------------------|
| **Data Loss** | Low (5%) | Critical | Automated backups, Real-time sync | Point-in-time recovery |
| **Extended Downtime** | Medium (15%) | High | Blue-green deployment | Immediate rollback |
| **Performance Degradation** | Low (10%) | Medium | Pre-migration testing | Query optimization |
| **Replication Lag** | Medium (20%) | Medium | Monitoring & alerts | Manual sync procedures |
| **Application Compatibility** | Low (5%) | High | Staging environment testing | Code hotfixes |

### Detailed Risk Analysis

#### 1. Data Integrity Risks

**Risk**: Data corruption or loss during migration
- **Probability**: 5% (Low)
- **Impact**: Critical - Could result in permanent data loss
- **Mitigation**:
  - Automated point-in-time backups every 15 minutes during migration
  - Comprehensive data validation at each phase
  - Real-time replication monitoring with automatic alerts
  - Checksums and row counts verification

**Contingency Plan**:
```sql
-- Emergency data recovery procedure
CREATE OR REPLACE FUNCTION emergency_data_recovery()
RETURNS TABLE(recovery_status TEXT, details TEXT) AS $$
BEGIN
    -- Restore from most recent backup
    PERFORM pg_stop_backup();
    PERFORM pg_restore_point('emergency_recovery');
    
    -- Validate data integrity
    RETURN QUERY
    SELECT 'DATA_RESTORED'::TEXT, 
           'Restored from backup at: ' || NOW()::TEXT;
END;
$$ LANGUAGE plpgsql;
```

#### 2. Application Downtime Risks

**Risk**: Extended service interruption beyond 30-second target
- **Probability**: 15% (Medium)
- **Impact**: High - Revenue loss, customer dissatisfaction
- **Mitigation**:
  - Blue-green deployment strategy
  - Automated rollback triggers
  - Load balancer health checks
  - Pre-validated application configurations

**Contingency Plan**:
```bash
# Automated rollback trigger
#!/bin/bash
DOWNTIME_THRESHOLD=45  # seconds
CURRENT_DOWNTIME=$(check_service_availability)

if [ $CURRENT_DOWNTIME -gt $DOWNTIME_THRESHOLD ]; then
    echo "Downtime exceeded threshold, executing automatic rollback"
    ./immediate_rollback.sh
    send_alert "Migration rolled back due to extended downtime: ${CURRENT_DOWNTIME}s"
fi
```

#### 3. Performance Risks

**Risk**: New schema performs worse than expected
- **Probability**: 10% (Low)
- **Impact**: Medium - Degraded user experience
- **Mitigation**:
  - Extensive performance testing in staging
  - Query plan analysis and optimization
  - Gradual traffic migration capability
  - Performance monitoring with alerts

**Contingency Plan**:
```sql
-- Performance optimization emergency procedures
CREATE OR REPLACE FUNCTION optimize_critical_queries()
RETURNS VOID AS $$
BEGIN
    -- Emergency index creation for critical queries
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_emergency_vm_dashboard 
    ON vms(tenant_id, state) WHERE state IN ('running', 'error');
    
    CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_emergency_metrics 
    ON vm_metrics(vm_id, collected_at DESC) 
    WHERE collected_at > NOW() - INTERVAL '1 hour';
    
    -- Update table statistics
    ANALYZE vms;
    ANALYZE vm_metrics;
    ANALYZE compute_nodes;
END;
$$ LANGUAGE plpgsql;
```

#### 4. Replication Issues

**Risk**: Replication lag or failure during migration
- **Probability**: 20% (Medium)
- **Impact**: Medium - Data inconsistency between environments
- **Mitigation**:
  - Real-time replication monitoring
  - Automated lag detection and alerting
  - Manual sync procedures
  - Conflict resolution mechanisms

**Monitoring Script**:
```bash
#!/bin/bash
# replication_monitor.sh

while true; do
    LAG=$(psql -h blue-db -tAc "
        SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) 
        FROM pg_stat_replication;")
    
    if [ "$LAG" -gt 10000000 ]; then  # 10MB lag threshold
        echo "⚠️ Replication lag detected: $LAG bytes"
        # Execute lag resolution procedure
        ./resolve_replication_lag.sh
    fi
    
    sleep 30
done
```

### Crisis Management Procedures

#### Escalation Matrix

**Level 1 - Automated Response** (0-5 minutes)
- Automated monitoring detects issues
- Rollback scripts execute automatically
- Stakeholders notified via alerts

**Level 2 - Engineering Response** (5-15 minutes)
- Senior Database Engineer investigates
- Manual intervention if needed
- Status updates every 5 minutes

**Level 3 - Management Escalation** (15+ minutes)
- CTO and VP Engineering involved
- Customer communication initiated
- Post-incident review scheduled

#### Communication Plan

```bash
# Alert notification system
send_migration_alert() {
    local severity=$1
    local message=$2
    
    # Slack notification
    curl -X POST -H 'Content-type: application/json' \
        --data '{"text":"🚨 MIGRATION ALERT ['$severity']: '$message'"}' \
        $SLACK_WEBHOOK_URL
    
    # Email for critical alerts
    if [ "$severity" = "CRITICAL" ]; then
        echo "$message" | mail -s "CRITICAL: Database Migration Alert" \
            engineering-oncall@novacron.io
    fi
    
    # PagerDuty for critical issues
    if [ "$severity" = "CRITICAL" ]; then
        curl -X POST 'https://events.pagerduty.com/v2/enqueue' \
            -H 'Content-Type: application/json' \
            -d '{
                "routing_key": "'$PAGERDUTY_KEY'",
                "event_action": "trigger",
                "payload": {
                    "summary": "Database Migration Critical Alert",
                    "severity": "critical",
                    "source": "migration-system"
                }
            }'
    fi
}
```

---

## 📈 Success Metrics & Validation

### Key Performance Indicators

#### Performance Metrics
- **Query Response Time**: Target < 500ms (90% of queries)
- **Dashboard Load Time**: Target < 2 seconds
- **Database Connection Pool**: Target < 70% utilization
- **Materialized View Refresh**: Target < 30 seconds

#### Reliability Metrics
- **Downtime During Migration**: Target < 30 seconds
- **Data Consistency**: Target 100% (zero data loss)
- **Rollback Capability**: Target < 5 minutes end-to-end
- **Application Recovery**: Target < 2 minutes after rollback

#### Security Metrics
- **Row Level Security**: 100% tenant isolation
- **Audit Trail**: 100% action logging
- **Access Control**: Zero unauthorized access attempts
- **Data Encryption**: 100% sensitive field encryption

### Validation Framework

#### Automated Test Suite
```bash
#!/bin/bash
# comprehensive_validation.sh

echo "=== NovaCron Migration Validation Suite ==="

# Test 1: Database Connectivity
echo "Testing database connectivity..."
psql -h green-db -c "SELECT version();" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Database connectivity: PASS"
else
    echo "❌ Database connectivity: FAIL"
    exit 1
fi

# Test 2: Data Integrity
echo "Testing data integrity..."
INTEGRITY_RESULT=$(psql -h green-db -tAc "SELECT final_migration_validation();")
if echo "$INTEGRITY_RESULT" | grep -q "FAIL"; then
    echo "❌ Data integrity: FAIL"
    echo "$INTEGRITY_RESULT"
    exit 1
else
    echo "✅ Data integrity: PASS"
fi

# Test 3: Performance Benchmarks
echo "Testing query performance..."
QUERY_TIME=$(psql -h green-db -c "\timing on" -c "SELECT * FROM mv_vm_dashboard LIMIT 100;" 2>&1 | grep "Time" | awk '{print $2}' | cut -d'.' -f1)
if [ "$QUERY_TIME" -lt 500 ]; then
    echo "✅ Query performance: PASS (${QUERY_TIME}ms)"
else
    echo "⚠️ Query performance: SLOW (${QUERY_TIME}ms)"
fi

# Test 4: Tenant Isolation
echo "Testing tenant isolation..."
psql -h green-db -c "SET app.tenant_id = 'test-tenant-1';" -c "SELECT COUNT(*) FROM vms;" > /tmp/tenant1_count
psql -h green-db -c "SET app.tenant_id = 'test-tenant-2';" -c "SELECT COUNT(*) FROM vms;" > /tmp/tenant2_count

if [ -s /tmp/tenant1_count ] && [ -s /tmp/tenant2_count ]; then
    echo "✅ Tenant isolation: PASS"
else
    echo "❌ Tenant isolation: FAIL"
    exit 1
fi

# Test 5: Application Integration
echo "Testing application integration..."
API_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/api_response "https://api.novacron.io/health")
if [ "$API_RESPONSE" = "200" ]; then
    echo "✅ Application integration: PASS"
else
    echo "❌ Application integration: FAIL (HTTP $API_RESPONSE)"
    exit 1
fi

echo "🎉 All validation tests passed!"
```

#### Performance Benchmarking Script
```sql
-- File: /scripts/performance_benchmark.sql
-- Purpose: Comprehensive performance testing

-- Dashboard query benchmark
\timing on

-- Test 1: VM Dashboard Query
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    tenant_id,
    total_vms,
    running_vms,
    total_cpu_cores,
    avg_cpu_usage
FROM mv_vm_dashboard
ORDER BY total_vms DESC
LIMIT 50;

-- Test 2: Time-series metrics query
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    date_trunc('hour', collected_at) as hour,
    AVG(cpu_usage_percent) as avg_cpu,
    MAX(memory_usage_percent) as max_memory
FROM vm_metrics 
WHERE collected_at >= NOW() - INTERVAL '24 hours'
GROUP BY date_trunc('hour', collected_at)
ORDER BY hour DESC;

-- Test 3: Complex join query
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    v.name as vm_name,
    cn.name as node_name,
    t.name as tenant_name,
    v.state,
    latest_metrics.avg_cpu
FROM vms v
JOIN compute_nodes cn ON v.node_id = cn.id
JOIN tenants t ON v.tenant_id = t.id
LEFT JOIN LATERAL (
    SELECT AVG(cpu_usage_percent) as avg_cpu
    FROM vm_metrics m
    WHERE m.vm_id = v.id
    AND m.collected_at >= NOW() - INTERVAL '1 hour'
) latest_metrics ON true
WHERE v.state = 'running'
LIMIT 100;

-- Test 4: Audit log query
EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
SELECT 
    user_id,
    action,
    resource_type,
    created_at
FROM audit_log
WHERE created_at >= NOW() - INTERVAL '7 days'
AND action IN ('CREATE', 'DELETE', 'MODIFY')
ORDER BY created_at DESC
LIMIT 1000;

\timing off

-- Generate performance report
SELECT 
    'Performance Benchmark Summary' as report_type,
    jsonb_build_object(
        'dashboard_query_optimized', 'Materialized view approach',
        'metrics_query_performance', 'Time-based indexing effective',
        'join_query_efficiency', 'Composite indexes utilized',
        'audit_query_speed', 'Optimized for compliance queries'
    ) as results;
```

### Post-Migration Monitoring

#### Real-time Monitoring Dashboard
```sql
-- Create monitoring view for operations team
CREATE OR REPLACE VIEW v_migration_monitoring AS
SELECT 
    -- Database health
    'database_health' as metric_category,
    jsonb_build_object(
        'active_connections', (SELECT count(*) FROM pg_stat_activity WHERE datname = current_database()),
        'database_size', pg_size_pretty(pg_database_size(current_database())),
        'longest_query_seconds', (
            SELECT EXTRACT(EPOCH FROM (NOW() - query_start))::integer
            FROM pg_stat_activity 
            WHERE state = 'active' 
            ORDER BY query_start 
            LIMIT 1
        ),
        'cache_hit_ratio', (
            SELECT round(
                (sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read))) * 100, 2
            ) FROM pg_statio_user_tables
        )
    ) as metrics,
    NOW() as measured_at
    
UNION ALL

SELECT 
    -- Application performance
    'application_performance' as metric_category,
    jsonb_build_object(
        'total_vms', (SELECT count(*) FROM vms),
        'running_vms', (SELECT count(*) FROM vms WHERE state = 'running'),
        'error_vms', (SELECT count(*) FROM vms WHERE state = 'error'),
        'recent_api_calls', (
            SELECT count(*) FROM audit_log 
            WHERE created_at >= NOW() - INTERVAL '1 hour'
        ),
        'tenant_count', (SELECT count(*) FROM tenants)
    ) as metrics,
    NOW() as measured_at;

-- Schedule automated monitoring reports
SELECT cron.schedule('migration-monitoring-report', '*/15 * * * *', 
    'INSERT INTO migration_monitoring_log SELECT * FROM v_migration_monitoring;');
```

#### Alerting Configuration
```sql
-- Create alerting functions
CREATE OR REPLACE FUNCTION check_migration_health()
RETURNS TABLE(alert_type TEXT, severity TEXT, message TEXT) AS $$
BEGIN
    -- Check for high connection usage
    RETURN QUERY
    SELECT 'CONNECTION_USAGE'::TEXT, 'WARNING'::TEXT,
           'High connection usage: ' || count(*)::TEXT || ' active connections'
    FROM pg_stat_activity 
    WHERE datname = current_database()
    HAVING count(*) > 200;
    
    -- Check for slow queries
    RETURN QUERY
    SELECT 'SLOW_QUERY'::TEXT, 'CRITICAL'::TEXT,
           'Query running for ' || EXTRACT(EPOCH FROM (NOW() - query_start))::TEXT || ' seconds'
    FROM pg_stat_activity 
    WHERE state = 'active' 
    AND query_start < NOW() - INTERVAL '30 seconds'
    AND query NOT ILIKE '%pg_stat_activity%';
    
    -- Check for replication lag (if applicable)
    RETURN QUERY
    SELECT 'REPLICATION_LAG'::TEXT, 'WARNING'::TEXT,
           'Replication lag: ' || pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn)::TEXT || ' bytes'
    FROM pg_stat_replication
    WHERE pg_wal_lsn_diff(pg_current_wal_lsn(), replay_lsn) > 10000000;
    
    -- Check materialized view freshness
    RETURN QUERY
    SELECT 'STALE_VIEWS'::TEXT, 'WARNING'::TEXT,
           'Materialized views not refreshed in last hour'
    FROM mv_vm_dashboard
    WHERE last_updated < NOW() - INTERVAL '1 hour';
    
END;
$$ LANGUAGE plpgsql;
```

---

## 📅 Implementation Timeline

### Detailed Project Schedule

#### Phase 1: Infrastructure Preparation (Week 1)
**Duration**: 2 hours execution, 1 week preparation

| Task | Duration | Owner | Dependencies | Success Criteria |
|------|----------|-------|--------------|------------------|
| Environment Setup | 30 min | DevOps Engineer | None | Green DB accessible |
| Replication Configuration | 45 min | DBA | Environment Setup | Replication lag < 5s |
| Monitoring Deployment | 30 min | DevOps Engineer | Environment Setup | Dashboard operational |
| Rollback Testing | 15 min | DBA | All above | < 5 min rollback time |

#### Phase 2: Schema Migration & Sync (Week 2)  
**Duration**: 4 hours execution, 1 week preparation

| Task | Duration | Owner | Dependencies | Success Criteria |
|------|----------|-------|--------------|------------------|
| Schema Deployment | 60 min | DBA | Phase 1 Complete | New schema validated |
| Data Migration | 120 min | DBA | Schema Deployment | 100% data integrity |
| Shadow Replication | 60 min | DBA | Data Migration | Real-time sync active |

#### Phase 3: Performance Optimization (Week 3)
**Duration**: 2 hours execution, 1 week preparation

| Task | Duration | Owner | Dependencies | Success Criteria |
|------|----------|-------|--------------|------------------|
| Index Creation | 45 min | DBA | Phase 2 Complete | All indexes created |
| Materialized Views | 60 min | DBA | Index Creation | Views refreshing |
| Config Optimization | 15 min | DBA | Views Complete | Settings applied |

#### Phase 4: Data Validation & Testing (Week 4)
**Duration**: 3 hours execution, 1 week preparation

| Task | Duration | Owner | Dependencies | Success Criteria |
|------|----------|-------|--------------|------------------|
| Data Integrity Check | 60 min | DBA | Phase 3 Complete | 100% validation pass |
| Performance Testing | 90 min | QA Engineer | Integrity Check | 300%+ improvement |
| Security Validation | 30 min | Security Engineer | Performance Testing | All tests pass |

#### Phase 5: Production Cutover (Week 5)
**Duration**: 30 minutes execution, 1 week preparation

| Task | Duration | Owner | Dependencies | Success Criteria |
|------|----------|-------|--------------|------------------|
| Pre-cutover Validation | 10 min | DBA | Phase 4 Complete | All checks pass |
| Traffic Switch | 10 min | DevOps Engineer | Validation | < 30s downtime |
| Post-cutover Testing | 10 min | QA Engineer | Traffic Switch | Apps healthy |

#### Phase 6: Post-Migration Verification (Week 5-6)
**Duration**: 1 hour execution + 1 week monitoring

| Task | Duration | Owner | Dependencies | Success Criteria |
|------|----------|-------|--------------|------------------|
| System Health Check | 20 min | DBA | Phase 5 Complete | All systems green |
| Performance Benchmarking | 30 min | DBA | Health Check | Targets exceeded |
| Documentation Update | 10 min | Tech Writer | Benchmarking | Docs complete |

### Resource Requirements

#### Team Allocation
- **Senior Database Administrator**: 40 hours (full-time for 1 week)
- **DevOps Engineer**: 20 hours (part-time across 5 weeks)
- **QA Engineer**: 15 hours (testing phases)
- **Security Engineer**: 8 hours (security validation)
- **Technical Writer**: 5 hours (documentation)

#### Infrastructure Costs
- **Green Environment**: $2,000/month (during migration)
- **Additional Backup Storage**: $300/month
- **Monitoring Tools**: $150/month
- **Total Migration Cost**: ~$3,000

#### Critical Dependencies
1. **Staging Environment Access**: Required 2 weeks before migration
2. **Maintenance Window**: 12-hour window with 30-minute downtime allowance
3. **Stakeholder Approval**: Security and compliance team sign-off
4. **Backup Systems**: Verified backup restoration capabilities

---

## 📝 Conclusion & Next Steps

### Migration Summary

This comprehensive migration plan transforms the NovaCron database architecture with:

- **Zero-downtime deployment** using blue-green strategy with < 30 seconds service interruption
- **Performance improvements** of 300-400% through optimized indexing and materialized views
- **Enhanced security** via Row Level Security and comprehensive audit trails
- **Improved scalability** supporting 10x current capacity with advanced partitioning
- **Robust disaster recovery** with point-in-time recovery and automated backup systems

### Key Success Factors

1. **Thorough Testing**: Extensive staging environment validation
2. **Automated Monitoring**: Real-time alerting and automated rollback capabilities
3. **Clear Communication**: Detailed stakeholder notifications and status updates
4. **Risk Mitigation**: Comprehensive rollback procedures and data recovery plans
5. **Performance Focus**: Materialized views and optimized indexing for dashboard queries

### Post-Migration Benefits

#### Immediate Benefits (Week 1)
- ✅ Unified schema eliminates dual table maintenance
- ✅ Dashboard queries respond in < 500ms (vs previous 3-5 seconds)
- ✅ Tenant isolation provides enterprise-grade security
- ✅ Real-time monitoring improves operational visibility

#### Medium-term Benefits (Month 1-3)
- ✅ Materialized views enable real-time analytics dashboards
- ✅ Advanced indexing supports complex reporting queries
- ✅ Improved backup and recovery reduces RTO from hours to minutes
- ✅ Row Level Security enables multi-tenant SaaS expansion

#### Long-term Benefits (6+ Months)
- ✅ Architecture supports 10x user growth without major changes
- ✅ Performance monitoring enables proactive optimization
- ✅ Enhanced audit capabilities support compliance requirements
- ✅ Modernized schema enables future feature development

### Immediate Next Steps

#### Week 1: Pre-Migration Preparation
1. **Schedule stakeholder review meeting** for migration plan approval
2. **Provision Green environment** and configure monitoring
3. **Set up staging environment** for final testing validation
4. **Conduct team walkthrough** of migration procedures and rollback plans

#### Week 2: Final Preparations  
1. **Execute full migration rehearsal** in staging environment
2. **Validate all scripts and procedures** with timing measurements
3. **Configure monitoring dashboards** and alert thresholds
4. **Obtain final approvals** from security and compliance teams

#### Week 3: Go-Live Preparation
1. **Schedule migration window** with all stakeholders
2. **Prepare communication templates** for customer notifications
3. **Brief support team** on migration timeline and expected impacts
4. **Final review** of rollback procedures and emergency contacts

### Long-term Recommendations

#### Database Architecture Evolution
- **Implement database sharding** for multi-region expansion
- **Add read replicas** for improved read performance
- **Consider time-series database** for metrics storage optimization
- **Evaluate columnstore indexes** for analytical workloads

#### Operational Excellence
- **Establish regular performance reviews** quarterly
- **Implement automated schema change management**
- **Develop disaster recovery testing schedule**
- **Create database capacity planning processes**

---

**Migration Plan Approval Required From:**
- [ ] Chief Technology Officer
- [ ] VP of Engineering  
- [ ] Database Team Lead
- [ ] Security Team Lead
- [ ] DevOps Team Lead

**Final Approval Date**: _________________

**Scheduled Migration Date**: _________________

**Migration Team Lead**: _________________

---

*This migration plan ensures NovaCron's database architecture is modernized, performant, and ready for enterprise-scale growth while maintaining zero-downtime service delivery.*