# Maintenance Runbook
## NovaCron v10 Extended - Routine Maintenance & System Updates

### Document Information
- **Version**: 1.0.0
- **Last Updated**: 2025-01-05
- **Classification**: OPERATIONAL
- **Review Frequency**: Monthly

---

## 1. Scheduled Maintenance Windows

### Maintenance Schedule

```yaml
maintenance_windows:
  production:
    weekly:
      window: "Sunday 02:00-04:00 UTC"
      operations:
        - database_optimization
        - cache_cleanup
        - log_rotation
        - metric_aggregation
      notification: "24 hours advance"
      
    monthly:
      window: "First Sunday 01:00-06:00 UTC"
      operations:
        - system_updates
        - security_patches
        - certificate_renewal
        - backup_verification
        - index_rebuild
      notification: "1 week advance"
      
    quarterly:
      window: "First Saturday 00:00-08:00 UTC"
      operations:
        - major_upgrades
        - infrastructure_updates
        - disaster_recovery_test
        - compliance_audit
      notification: "2 weeks advance"
      
  staging:
    daily:
      window: "Every day 04:00-05:00 UTC"
      operations:
        - automated_testing
        - performance_benchmarks
        - security_scans
      notification: "none"
      
  development:
    continuous:
      operations:
        - continuous_deployment
        - automated_testing
      notification: "none"
```

### Pre-Maintenance Checklist

```bash
#!/bin/bash
# pre-maintenance-checklist.sh

echo "=== Pre-Maintenance Checklist ==="

MAINTENANCE_TYPE=$1  # weekly|monthly|quarterly
ENVIRONMENT=$2       # production|staging|development

run_pre_maintenance_checks() {
    local checks_passed=true
    
    # 1. Verify backup completion
    echo "[1/10] Verifying recent backups..."
    if ! verify_backups; then
        echo "❌ Backup verification failed"
        checks_passed=false
    else
        echo "✅ Backups verified"
    fi
    
    # 2. Check system health
    echo "[2/10] Checking system health..."
    if ! check_system_health; then
        echo "❌ System health check failed"
        checks_passed=false
    else
        echo "✅ System healthy"
    fi
    
    # 3. Verify rollback capability
    echo "[3/10] Verifying rollback procedures..."
    if ! test_rollback_capability; then
        echo "❌ Rollback verification failed"
        checks_passed=false
    else
        echo "✅ Rollback ready"
    fi
    
    # 4. Check maintenance scripts
    echo "[4/10] Validating maintenance scripts..."
    if ! validate_maintenance_scripts; then
        echo "❌ Script validation failed"
        checks_passed=false
    else
        echo "✅ Scripts validated"
    fi
    
    # 5. Verify team availability
    echo "[5/10] Checking team availability..."
    if ! check_team_availability; then
        echo "⚠️  Limited team availability"
    else
        echo "✅ Team available"
    fi
    
    # 6. Review change requests
    echo "[6/10] Reviewing change requests..."
    pending_changes=$(get_pending_changes)
    echo "   Pending changes: $pending_changes"
    
    # 7. Check dependencies
    echo "[7/10] Checking external dependencies..."
    if ! check_external_dependencies; then
        echo "⚠️  Some dependencies unavailable"
    else
        echo "✅ Dependencies available"
    fi
    
    # 8. Verify monitoring
    echo "[8/10] Verifying monitoring systems..."
    if ! verify_monitoring_systems; then
        echo "❌ Monitoring issues detected"
        checks_passed=false
    else
        echo "✅ Monitoring operational"
    fi
    
    # 9. Test notification systems
    echo "[9/10] Testing notification systems..."
    if ! test_notification_systems; then
        echo "⚠️  Notification system issues"
    else
        echo "✅ Notifications working"
    fi
    
    # 10. Create maintenance snapshot
    echo "[10/10] Creating pre-maintenance snapshot..."
    if ! create_maintenance_snapshot; then
        echo "❌ Snapshot creation failed"
        checks_passed=false
    else
        echo "✅ Snapshot created"
    fi
    
    # Final decision
    if [ "$checks_passed" = true ]; then
        echo ""
        echo "✅ PRE-MAINTENANCE CHECKS PASSED"
        echo "Maintenance window can proceed"
        return 0
    else
        echo ""
        echo "❌ PRE-MAINTENANCE CHECKS FAILED"
        echo "Please resolve issues before proceeding"
        return 1
    fi
}

verify_backups() {
    # Check last backup time
    last_backup=$(aws s3 ls s3://novacron-backups/ --recursive | tail -1 | awk '{print $1, $2}')
    backup_age=$(date -d "$last_backup" +%s)
    current_time=$(date +%s)
    age_hours=$(( (current_time - backup_age) / 3600 ))
    
    if [ $age_hours -gt 24 ]; then
        return 1
    fi
    
    # Verify backup integrity
    aws s3api head-object --bucket novacron-backups --key latest/backup.tar.gz.sha256
    return $?
}

check_system_health() {
    # Check all services
    services=("novacron-api" "novacron-scheduler" "novacron-worker" "postgresql" "redis")
    
    for service in "${services[@]}"; do
        if ! systemctl is-active --quiet $service; then
            echo "   Service $service is not running"
            return 1
        fi
    done
    
    # Check resource usage
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        echo "   High CPU usage: ${cpu_usage}%"
        return 1
    fi
    
    return 0
}

# Run checks
run_pre_maintenance_checks
```

---

## 2. Database Maintenance

### Database Optimization Procedures

```sql
-- database-maintenance.sql

-- 1. Vacuum and Analyze
VACUUM (VERBOSE, ANALYZE);

-- 2. Reindex critical tables
REINDEX TABLE CONCURRENTLY users;
REINDEX TABLE CONCURRENTLY tasks;
REINDEX TABLE CONCURRENTLY schedules;
REINDEX TABLE CONCURRENTLY audit_logs;

-- 3. Update statistics
ANALYZE users;
ANALYZE tasks;
ANALYZE schedules;
ANALYZE audit_logs;

-- 4. Clean up old partitions
DO $$
DECLARE
    partition_name TEXT;
    cutoff_date DATE := CURRENT_DATE - INTERVAL '90 days';
BEGIN
    FOR partition_name IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE tablename LIKE 'audit_logs_%' 
        AND tablename < 'audit_logs_' || to_char(cutoff_date, 'YYYY_MM')
    LOOP
        EXECUTE format('DROP TABLE IF EXISTS %I CASCADE', partition_name);
        RAISE NOTICE 'Dropped partition: %', partition_name;
    END LOOP;
END $$;

-- 5. Check for bloat
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - pg_relation_size(schemaname||'.'||tablename)) AS index_size,
    ROUND(100 * pg_total_relation_size(schemaname||'.'||tablename) / 
          NULLIF(SUM(pg_total_relation_size(schemaname||'.'||tablename)) OVER (), 0), 2) AS percentage
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 20;

-- 6. Fix sequence gaps
DO $$
DECLARE
    seq RECORD;
    max_id BIGINT;
BEGIN
    FOR seq IN 
        SELECT sequence_name, 
               REPLACE(sequence_name, '_id_seq', '') AS table_name,
               REPLACE(REPLACE(sequence_name, '_seq', ''), REPLACE(sequence_name, '_id_seq', '') || '_', '') AS column_name
        FROM information_schema.sequences 
        WHERE sequence_schema = 'public'
    LOOP
        EXECUTE format('SELECT COALESCE(MAX(%I), 0) FROM %I', seq.column_name, seq.table_name) INTO max_id;
        EXECUTE format('SELECT setval(''%I'', %s)', seq.sequence_name, max_id + 1);
    END LOOP;
END $$;

-- 7. Connection pool optimization
ALTER SYSTEM SET max_connections = 400;
ALTER SYSTEM SET shared_buffers = '8GB';
ALTER SYSTEM SET effective_cache_size = '24GB';
ALTER SYSTEM SET maintenance_work_mem = '2GB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
SELECT pg_reload_conf();
```

### Automated Database Maintenance

```python
#!/usr/bin/env python3
# database_maintenance.py

import psycopg2
import logging
from datetime import datetime, timedelta
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseMaintenance:
    def __init__(self, connection_string):
        self.conn_string = connection_string
        self.conn = None
        self.maintenance_log = []
        
    def connect(self):
        """Establish database connection"""
        self.conn = psycopg2.connect(self.conn_string)
        self.conn.autocommit = True
        
    def run_maintenance(self, maintenance_type='weekly'):
        """Run maintenance based on schedule type"""
        logger.info(f"Starting {maintenance_type} database maintenance")
        
        if maintenance_type == 'daily':
            self.daily_maintenance()
        elif maintenance_type == 'weekly':
            self.weekly_maintenance()
        elif maintenance_type == 'monthly':
            self.monthly_maintenance()
        else:
            raise ValueError(f"Unknown maintenance type: {maintenance_type}")
        
        self.generate_maintenance_report()
        
    def daily_maintenance(self):
        """Daily maintenance tasks"""
        tasks = [
            ('Update statistics', self.update_statistics),
            ('Clean temporary tables', self.clean_temp_tables),
            ('Archive old logs', self.archive_old_logs),
            ('Check replication lag', self.check_replication_lag),
        ]
        
        self.execute_maintenance_tasks(tasks)
        
    def weekly_maintenance(self):
        """Weekly maintenance tasks"""
        tasks = [
            ('Vacuum analyze', self.vacuum_analyze),
            ('Reindex tables', self.reindex_tables),
            ('Clean up orphaned records', self.cleanup_orphans),
            ('Optimize queries', self.optimize_slow_queries),
            ('Partition maintenance', self.maintain_partitions),
        ]
        
        self.execute_maintenance_tasks(tasks)
        
    def monthly_maintenance(self):
        """Monthly maintenance tasks"""
        tasks = [
            ('Full vacuum', self.full_vacuum),
            ('Rebuild indexes', self.rebuild_indexes),
            ('Analyze table bloat', self.analyze_bloat),
            ('Update table statistics', self.update_all_statistics),
            ('Archive old data', self.archive_old_data),
        ]
        
        self.execute_maintenance_tasks(tasks)
        
    def execute_maintenance_tasks(self, tasks):
        """Execute maintenance tasks and log results"""
        for task_name, task_func in tasks:
            try:
                logger.info(f"Executing: {task_name}")
                start_time = datetime.now()
                result = task_func()
                duration = (datetime.now() - start_time).total_seconds()
                
                self.maintenance_log.append({
                    'task': task_name,
                    'status': 'success',
                    'duration': duration,
                    'result': result
                })
                
                logger.info(f"Completed: {task_name} ({duration:.2f}s)")
                
            except Exception as e:
                logger.error(f"Failed: {task_name} - {str(e)}")
                self.maintenance_log.append({
                    'task': task_name,
                    'status': 'failed',
                    'error': str(e)
                })
    
    def vacuum_analyze(self):
        """Run VACUUM ANALYZE on all tables"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT schemaname, tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            tables = cur.fetchall()
            
            for schema, table in tables:
                logger.info(f"  Vacuuming {schema}.{table}")
                cur.execute(f"VACUUM ANALYZE {schema}.{table}")
                
        return f"Vacuumed {len(tables)} tables"
    
    def reindex_tables(self):
        """Reindex tables with high bloat"""
        with self.conn.cursor() as cur:
            # Find tables with high index bloat
            cur.execute("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
                FROM pg_stat_user_indexes
                JOIN pg_index ON pg_stat_user_indexes.indexrelid = pg_index.indexrelid
                WHERE pg_relation_size(indexrelid) > 100000000  -- Indexes > 100MB
                ORDER BY pg_relation_size(indexrelid) DESC
                LIMIT 10
            """)
            
            indexes = cur.fetchall()
            
            for schema, table, index, size in indexes:
                logger.info(f"  Reindexing {index} ({size})")
                try:
                    cur.execute(f"REINDEX INDEX CONCURRENTLY {index}")
                except:
                    # Fall back to non-concurrent if needed
                    cur.execute(f"REINDEX INDEX {index}")
                    
        return f"Reindexed {len(indexes)} indexes"
    
    def cleanup_orphans(self):
        """Clean up orphaned records"""
        with self.conn.cursor() as cur:
            # Example: Clean orphaned task executions
            cur.execute("""
                DELETE FROM task_executions 
                WHERE task_id NOT IN (SELECT id FROM tasks)
            """)
            orphaned_executions = cur.rowcount
            
            # Clean orphaned audit logs
            cur.execute("""
                DELETE FROM audit_logs 
                WHERE user_id NOT IN (SELECT id FROM users)
                AND created_at < NOW() - INTERVAL '30 days'
            """)
            orphaned_logs = cur.rowcount
            
        return f"Cleaned {orphaned_executions} executions, {orphaned_logs} logs"
    
    def maintain_partitions(self):
        """Maintain table partitions"""
        with self.conn.cursor() as cur:
            # Create new partitions for next month
            next_month = datetime.now() + timedelta(days=30)
            partition_name = f"audit_logs_{next_month.strftime('%Y_%m')}"
            
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {partition_name} 
                PARTITION OF audit_logs 
                FOR VALUES FROM ('{next_month.strftime('%Y-%m-01')}') 
                TO ('{(next_month + timedelta(days=32)).strftime('%Y-%m-01')}')
            """)
            
            # Drop old partitions
            cutoff_date = datetime.now() - timedelta(days=90)
            cur.execute("""
                SELECT tablename 
                FROM pg_tables 
                WHERE tablename LIKE 'audit_logs_%' 
                AND tablename < %s
            """, (f"audit_logs_{cutoff_date.strftime('%Y_%m')}",))
            
            old_partitions = cur.fetchall()
            for (partition,) in old_partitions:
                logger.info(f"  Dropping old partition: {partition}")
                cur.execute(f"DROP TABLE IF EXISTS {partition}")
                
        return f"Created new partition, dropped {len(old_partitions)} old"
    
    def optimize_slow_queries(self):
        """Identify and optimize slow queries"""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    query,
                    calls,
                    total_exec_time,
                    mean_exec_time,
                    max_exec_time
                FROM pg_stat_statements
                WHERE mean_exec_time > 1000  -- Queries taking > 1 second
                ORDER BY mean_exec_time DESC
                LIMIT 10
            """)
            
            slow_queries = cur.fetchall()
            
            recommendations = []
            for query, calls, total, mean, max_time in slow_queries:
                # Simple optimization recommendations
                if 'JOIN' in query and 'index' not in query.lower():
                    recommendations.append("Consider adding indexes for JOIN conditions")
                if 'LIKE' in query and query.count('%') > 2:
                    recommendations.append("Consider full-text search instead of multiple LIKE")
                if 'SELECT *' in query:
                    recommendations.append("Select only needed columns instead of *")
                    
        return f"Found {len(slow_queries)} slow queries, {len(recommendations)} recommendations"
    
    def generate_maintenance_report(self):
        """Generate maintenance report"""
        report = f"""
=== Database Maintenance Report ===
Date: {datetime.now().isoformat()}
Total Tasks: {len(self.maintenance_log)}
Successful: {sum(1 for log in self.maintenance_log if log['status'] == 'success')}
Failed: {sum(1 for log in self.maintenance_log if log['status'] == 'failed')}

Task Details:
"""
        
        for log in self.maintenance_log:
            if log['status'] == 'success':
                report += f"✅ {log['task']}: {log.get('result', 'Completed')} ({log.get('duration', 0):.2f}s)\n"
            else:
                report += f"❌ {log['task']}: {log.get('error', 'Failed')}\n"
                
        logger.info(report)
        
        # Save report
        with open(f"/logs/db-maintenance-{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
            f.write(report)

# Run maintenance
if __name__ == "__main__":
    import sys
    
    maintenance_type = sys.argv[1] if len(sys.argv) > 1 else 'weekly'
    
    db = DatabaseMaintenance("postgresql://user:pass@localhost/novacron")
    db.connect()
    db.run_maintenance(maintenance_type)
```

---

## 3. System Updates

### Update Deployment Procedures

```bash
#!/bin/bash
# system-update-procedure.sh

UPDATE_TYPE=$1  # security|feature|major
ENVIRONMENT=$2  # production|staging|development

perform_system_update() {
    echo "=== System Update Procedure ==="
    echo "Type: $UPDATE_TYPE"
    echo "Environment: $ENVIRONMENT"
    echo "Start Time: $(date)"
    
    # Phase 1: Preparation
    prepare_for_update
    
    # Phase 2: Backup
    create_update_backup
    
    # Phase 3: Update deployment
    deploy_updates
    
    # Phase 4: Verification
    verify_update_success
    
    # Phase 5: Cleanup
    post_update_cleanup
    
    echo "Update completed at: $(date)"
}

prepare_for_update() {
    echo ""
    echo "=== Phase 1: Preparation ==="
    
    # 1. Enable maintenance mode
    echo "Enabling maintenance mode..."
    kubectl set env deployment/novacron-api MAINTENANCE_MODE=true
    
    # 2. Scale down non-critical services
    echo "Scaling down non-critical services..."
    kubectl scale deployment novacron-batch-processor --replicas=0
    kubectl scale deployment novacron-report-generator --replicas=0
    
    # 3. Drain queue
    echo "Draining job queue..."
    redis-cli SET queue_drain_mode true
    
    # Wait for queue to empty
    while [ $(redis-cli LLEN job_queue) -gt 0 ]; do
        echo "  Waiting for queue to drain: $(redis-cli LLEN job_queue) jobs remaining"
        sleep 5
    done
    
    # 4. Create update manifest
    cat > /tmp/update-manifest.yaml << EOF
update:
  type: $UPDATE_TYPE
  environment: $ENVIRONMENT
  timestamp: $(date -Iseconds)
  components:
    - api: ${API_VERSION}
    - scheduler: ${SCHEDULER_VERSION}
    - worker: ${WORKER_VERSION}
  rollback_point: $(git rev-parse HEAD)
EOF
}

create_update_backup() {
    echo ""
    echo "=== Phase 2: Backup ==="
    
    BACKUP_ID="update-backup-$(date +%Y%m%d-%H%M%S)"
    
    # 1. Database backup
    echo "Creating database backup..."
    pg_dump -h $DB_HOST -U $DB_USER novacron | gzip > /backups/$BACKUP_ID-db.sql.gz
    
    # 2. Configuration backup
    echo "Backing up configurations..."
    kubectl get configmap --all-namespaces -o yaml > /backups/$BACKUP_ID-configmaps.yaml
    kubectl get secret --all-namespaces -o yaml > /backups/$BACKUP_ID-secrets.yaml
    
    # 3. Persistent volume backup
    echo "Backing up persistent volumes..."
    for pv in $(kubectl get pv -o name); do
        kubectl get $pv -o yaml > /backups/$BACKUP_ID-$pv.yaml
    done
    
    # 4. Application state backup
    echo "Backing up application state..."
    redis-cli --rdb /backups/$BACKUP_ID-redis.rdb
    
    # 5. Upload to S3
    echo "Uploading backups to S3..."
    aws s3 sync /backups/ s3://novacron-backups/$BACKUP_ID/
    
    echo "Backup completed: $BACKUP_ID"
}

deploy_updates() {
    echo ""
    echo "=== Phase 3: Update Deployment ==="
    
    case $UPDATE_TYPE in
        "security")
            deploy_security_updates
            ;;
        "feature")
            deploy_feature_updates
            ;;
        "major")
            deploy_major_updates
            ;;
    esac
}

deploy_security_updates() {
    echo "Deploying security updates..."
    
    # 1. Update base images
    docker pull novacron/api:latest-security
    docker pull novacron/scheduler:latest-security
    docker pull novacron/worker:latest-security
    
    # 2. Rolling update
    kubectl set image deployment/novacron-api api=novacron/api:latest-security
    kubectl rollout status deployment/novacron-api --timeout=10m
    
    kubectl set image deployment/novacron-scheduler scheduler=novacron/scheduler:latest-security
    kubectl rollout status deployment/novacron-scheduler --timeout=10m
    
    kubectl set image deployment/novacron-worker worker=novacron/worker:latest-security
    kubectl rollout status deployment/novacron-worker --timeout=10m
    
    # 3. Update security policies
    kubectl apply -f /updates/security-policies.yaml
}

deploy_feature_updates() {
    echo "Deploying feature updates..."
    
    # 1. Database migrations
    echo "Running database migrations..."
    kubectl run migration --rm -i --restart=Never \
        --image=novacron/migrator:latest \
        -- migrate up
    
    # 2. Blue-green deployment
    echo "Starting blue-green deployment..."
    
    # Deploy to green environment
    kubectl apply -f /updates/green-deployment.yaml
    kubectl wait --for=condition=available --timeout=10m deployment/novacron-api-green
    
    # Test green environment
    if test_deployment "green"; then
        # Switch traffic to green
        kubectl patch service novacron-api -p '{"spec":{"selector":{"version":"green"}}}'
        
        # Wait for traffic drain
        sleep 30
        
        # Remove blue deployment
        kubectl delete deployment novacron-api-blue
        
        # Rename green to blue for next update
        kubectl patch deployment novacron-api-green \
            -p '{"metadata":{"name":"novacron-api-blue"}}'
    else
        echo "Green deployment failed tests, rolling back"
        kubectl delete deployment novacron-api-green
        return 1
    fi
}

deploy_major_updates() {
    echo "Deploying major version update..."
    
    # 1. Check compatibility
    echo "Checking compatibility..."
    if ! check_compatibility; then
        echo "Compatibility check failed"
        return 1
    fi
    
    # 2. Multi-stage deployment
    stages=("database" "backend" "frontend" "workers")
    
    for stage in "${stages[@]}"; do
        echo "Updating stage: $stage"
        
        # Deploy stage
        kubectl apply -f /updates/$stage-update.yaml
        
        # Wait for stage completion
        kubectl wait --for=condition=available --timeout=15m deployment/$stage
        
        # Verify stage
        if ! verify_stage $stage; then
            echo "Stage $stage verification failed"
            initiate_rollback
            return 1
        fi
        
        echo "Stage $stage completed successfully"
    done
}

verify_update_success() {
    echo ""
    echo "=== Phase 4: Verification ==="
    
    local all_checks_passed=true
    
    # 1. Health checks
    echo "Running health checks..."
    for service in api scheduler worker; do
        if ! curl -f https://novacron.io/health/$service; then
            echo "  ❌ $service health check failed"
            all_checks_passed=false
        else
            echo "  ✅ $service healthy"
        fi
    done
    
    # 2. Smoke tests
    echo "Running smoke tests..."
    if ! run_smoke_tests; then
        echo "  ❌ Smoke tests failed"
        all_checks_passed=false
    else
        echo "  ✅ Smoke tests passed"
    fi
    
    # 3. Performance validation
    echo "Validating performance..."
    response_time=$(curl -w "%{time_total}" -o /dev/null -s https://novacron.io/api/status)
    if (( $(echo "$response_time > 1" | bc -l) )); then
        echo "  ⚠️  High response time: ${response_time}s"
    else
        echo "  ✅ Response time: ${response_time}s"
    fi
    
    # 4. Error rate check
    echo "Checking error rates..."
    error_rate=$(curl -s https://metrics.novacron.io/error-rate | jq -r .rate)
    if (( $(echo "$error_rate > 1" | bc -l) )); then
        echo "  ❌ High error rate: ${error_rate}%"
        all_checks_passed=false
    else
        echo "  ✅ Error rate: ${error_rate}%"
    fi
    
    if [ "$all_checks_passed" = false ]; then
        echo ""
        echo "❌ Verification failed, initiating rollback"
        initiate_rollback
        return 1
    fi
    
    echo ""
    echo "✅ All verification checks passed"
    return 0
}

post_update_cleanup() {
    echo ""
    echo "=== Phase 5: Cleanup ==="
    
    # 1. Disable maintenance mode
    echo "Disabling maintenance mode..."
    kubectl set env deployment/novacron-api MAINTENANCE_MODE=false
    
    # 2. Scale up services
    echo "Scaling up services..."
    kubectl scale deployment novacron-batch-processor --replicas=3
    kubectl scale deployment novacron-report-generator --replicas=2
    
    # 3. Re-enable queue processing
    echo "Re-enabling queue processing..."
    redis-cli DEL queue_drain_mode
    
    # 4. Clear caches
    echo "Clearing caches..."
    redis-cli FLUSHDB
    
    # 5. Update documentation
    echo "Updating documentation..."
    update_version_documentation
    
    # 6. Send notifications
    echo "Sending update notifications..."
    send_update_notifications
}

initiate_rollback() {
    echo ""
    echo "=== INITIATING ROLLBACK ==="
    
    # Get rollback point from manifest
    ROLLBACK_POINT=$(cat /tmp/update-manifest.yaml | grep rollback_point | cut -d: -f2 | tr -d ' ')
    
    echo "Rolling back to: $ROLLBACK_POINT"
    
    # Rollback deployments
    kubectl rollout undo deployment/novacron-api
    kubectl rollout undo deployment/novacron-scheduler
    kubectl rollout undo deployment/novacron-worker
    
    # Wait for rollback
    kubectl rollout status deployment/novacron-api --timeout=10m
    kubectl rollout status deployment/novacron-scheduler --timeout=10m
    kubectl rollout status deployment/novacron-worker --timeout=10m
    
    # Restore database if needed
    if [ "$UPDATE_TYPE" == "major" ]; then
        echo "Restoring database backup..."
        restore_database_backup
    fi
    
    echo "Rollback completed"
}

# Execute update
perform_system_update
```

---

## 4. Certificate Management

### SSL/TLS Certificate Renewal

```python
#!/usr/bin/env python3
# certificate_management.py

import ssl
import socket
import datetime
import subprocess
import json
from typing import Dict, List

class CertificateManager:
    def __init__(self):
        self.domains = [
            'novacron.io',
            'api.novacron.io',
            'app.novacron.io',
            '*.novacron.io'
        ]
        self.certificates = {}
        
    def check_all_certificates(self) -> Dict:
        """Check expiration for all certificates"""
        results = {}
        
        for domain in self.domains:
            cert_info = self.check_certificate(domain)
            results[domain] = cert_info
            
            # Alert if expiring soon
            if cert_info['days_remaining'] < 30:
                self.alert_expiring_certificate(domain, cert_info)
                
        return results
    
    def check_certificate(self, domain: str) -> Dict:
        """Check certificate expiration for a domain"""
        try:
            # Get certificate
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert = ssock.getpeercert()
                    
            # Parse expiration
            not_after = datetime.datetime.strptime(
                cert['notAfter'],
                '%b %d %H:%M:%S %Y %Z'
            )
            
            days_remaining = (not_after - datetime.datetime.now()).days
            
            return {
                'domain': domain,
                'expires': not_after.isoformat(),
                'days_remaining': days_remaining,
                'issuer': dict(x[0] for x in cert['issuer']),
                'subject': dict(x[0] for x in cert['subject']),
                'status': 'valid' if days_remaining > 30 else 'expiring'
            }
            
        except Exception as e:
            return {
                'domain': domain,
                'error': str(e),
                'status': 'error'
            }
    
    def renew_certificate(self, domain: str):
        """Renew certificate using Let's Encrypt"""
        
        print(f"Renewing certificate for {domain}")
        
        # Use certbot for renewal
        cmd = [
            'certbot', 'renew',
            '--domain', domain,
            '--non-interactive',
            '--agree-tos',
            '--email', 'admin@novacron.io',
            '--webroot',
            '--webroot-path', '/var/www/certbot'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Certificate renewed successfully for {domain}")
            self.deploy_certificate(domain)
            return True
        else:
            print(f"Certificate renewal failed: {result.stderr}")
            return False
    
    def deploy_certificate(self, domain: str):
        """Deploy renewed certificate to services"""
        
        # Copy certificates to Kubernetes secrets
        subprocess.run([
            'kubectl', 'create', 'secret', 'tls',
            f'{domain}-tls',
            '--cert', f'/etc/letsencrypt/live/{domain}/fullchain.pem',
            '--key', f'/etc/letsencrypt/live/{domain}/privkey.pem',
            '--dry-run=client', '-o', 'yaml', '|',
            'kubectl', 'apply', '-f', '-'
        ])
        
        # Reload services
        services = ['nginx', 'haproxy']
        for service in services:
            subprocess.run(['systemctl', 'reload', service])
        
        # Update CDN
        self.update_cdn_certificate(domain)
    
    def update_cdn_certificate(self, domain: str):
        """Update certificate in CDN"""
        
        # Read certificate
        with open(f'/etc/letsencrypt/live/{domain}/fullchain.pem', 'r') as f:
            cert_body = f.read()
            
        with open(f'/etc/letsencrypt/live/{domain}/privkey.pem', 'r') as f:
            private_key = f.read()
            
        # Update CloudFront
        import boto3
        cf = boto3.client('cloudfront')
        
        response = cf.update_distribution(
            Id='E1234567890',
            DistributionConfig={
                'ViewerCertificate': {
                    'ACMCertificateArn': self.upload_to_acm(domain, cert_body, private_key),
                    'SSLSupportMethod': 'sni-only'
                }
            }
        )
    
    def automated_renewal_check(self):
        """Automated certificate renewal check"""
        
        print("=== Certificate Renewal Check ===")
        print(f"Check time: {datetime.datetime.now()}")
        
        renewal_needed = []
        
        for domain in self.domains:
            cert_info = self.check_certificate(domain)
            
            print(f"\n{domain}:")
            print(f"  Expires: {cert_info.get('expires', 'Unknown')}")
            print(f"  Days remaining: {cert_info.get('days_remaining', 'Unknown')}")
            
            # Auto-renew if less than 30 days
            if cert_info.get('days_remaining', 0) < 30:
                renewal_needed.append(domain)
                
        if renewal_needed:
            print(f"\n⚠️  Certificates needing renewal: {renewal_needed}")
            
            for domain in renewal_needed:
                if self.renew_certificate(domain):
                    print(f"✅ Renewed: {domain}")
                else:
                    print(f"❌ Failed to renew: {domain}")
                    self.alert_renewal_failure(domain)
        else:
            print("\n✅ All certificates are valid")
    
    def generate_certificate_report(self) -> str:
        """Generate certificate status report"""
        
        results = self.check_all_certificates()
        
        report = """
=== Certificate Status Report ===
Generated: {}

Certificate Status:
""".format(datetime.datetime.now().isoformat())
        
        for domain, info in results.items():
            if info.get('status') == 'valid':
                status_icon = "✅"
            elif info.get('status') == 'expiring':
                status_icon = "⚠️"
            else:
                status_icon = "❌"
                
            report += f"\n{status_icon} {domain}"
            
            if 'days_remaining' in info:
                report += f"\n   Expires: {info['expires']}"
                report += f"\n   Days remaining: {info['days_remaining']}"
            else:
                report += f"\n   Error: {info.get('error', 'Unknown')}"
                
        # Add recommendations
        report += "\n\nRecommendations:"
        
        expiring_soon = [d for d, i in results.items() 
                        if i.get('days_remaining', 0) < 30]
        
        if expiring_soon:
            report += f"\n- Renew certificates for: {', '.join(expiring_soon)}"
        else:
            report += "\n- No immediate action required"
            
        return report

# Certificate automation script
if __name__ == "__main__":
    manager = CertificateManager()
    
    # Check and renew certificates
    manager.automated_renewal_check()
    
    # Generate report
    report = manager.generate_certificate_report()
    print(report)
    
    # Save report
    with open(f"/reports/certificates-{datetime.datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
        f.write(report)
```

---

## 5. Health Check Procedures

### System Health Monitoring

```go
// health_check.go
package maintenance

import (
    "context"
    "database/sql"
    "fmt"
    "net/http"
    "time"
)

type HealthChecker struct {
    Services []ServiceHealth
    Database *sql.DB
    Redis    RedisClient
}

type ServiceHealth struct {
    Name     string
    URL      string
    Timeout  time.Duration
    Critical bool
}

type HealthStatus struct {
    Service   string
    Status    string
    Latency   time.Duration
    Message   string
    Timestamp time.Time
}

func (hc *HealthChecker) RunHealthChecks(ctx context.Context) []HealthStatus {
    results := []HealthStatus{}
    
    // Check services
    for _, service := range hc.Services {
        status := hc.checkService(ctx, service)
        results = append(results, status)
    }
    
    // Check database
    dbStatus := hc.checkDatabase(ctx)
    results = append(results, dbStatus)
    
    // Check cache
    cacheStatus := hc.checkCache(ctx)
    results = append(results, cacheStatus)
    
    // Check disk space
    diskStatus := hc.checkDiskSpace()
    results = append(results, diskStatus)
    
    // Check memory
    memStatus := hc.checkMemory()
    results = append(results, memStatus)
    
    return results
}

func (hc *HealthChecker) checkService(ctx context.Context, service ServiceHealth) HealthStatus {
    start := time.Now()
    status := HealthStatus{
        Service:   service.Name,
        Timestamp: time.Now(),
    }
    
    // Create HTTP client with timeout
    client := &http.Client{
        Timeout: service.Timeout,
    }
    
    // Make health check request
    req, err := http.NewRequestWithContext(ctx, "GET", service.URL, nil)
    if err != nil {
        status.Status = "error"
        status.Message = fmt.Sprintf("Failed to create request: %v", err)
        return status
    }
    
    resp, err := client.Do(req)
    status.Latency = time.Since(start)
    
    if err != nil {
        status.Status = "unhealthy"
        status.Message = fmt.Sprintf("Request failed: %v", err)
        return status
    }
    defer resp.Body.Close()
    
    if resp.StatusCode == http.StatusOK {
        status.Status = "healthy"
        status.Message = "Service responding normally"
    } else {
        status.Status = "degraded"
        status.Message = fmt.Sprintf("Unexpected status code: %d", resp.StatusCode)
    }
    
    return status
}

func (hc *HealthChecker) checkDatabase(ctx context.Context) HealthStatus {
    start := time.Now()
    status := HealthStatus{
        Service:   "database",
        Timestamp: time.Now(),
    }
    
    // Check connection
    err := hc.Database.PingContext(ctx)
    status.Latency = time.Since(start)
    
    if err != nil {
        status.Status = "unhealthy"
        status.Message = fmt.Sprintf("Database ping failed: %v", err)
        return status
    }
    
    // Check replication lag
    var lag time.Duration
    row := hc.Database.QueryRowContext(ctx, 
        "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()))::int")
    
    if err := row.Scan(&lag); err == nil && lag > 5 {
        status.Status = "degraded"
        status.Message = fmt.Sprintf("Replication lag: %v", lag)
    } else {
        status.Status = "healthy"
        status.Message = "Database healthy"
    }
    
    return status
}

func (hc *HealthChecker) checkCache(ctx context.Context) HealthStatus {
    start := time.Now()
    status := HealthStatus{
        Service:   "cache",
        Timestamp: time.Now(),
    }
    
    // Ping Redis
    err := hc.Redis.Ping(ctx).Err()
    status.Latency = time.Since(start)
    
    if err != nil {
        status.Status = "unhealthy"
        status.Message = fmt.Sprintf("Redis ping failed: %v", err)
        return status
    }
    
    // Check memory usage
    info, _ := hc.Redis.Info(ctx, "memory").Result()
    // Parse and check memory usage
    
    status.Status = "healthy"
    status.Message = "Cache healthy"
    
    return status
}

func (hc *HealthChecker) checkDiskSpace() HealthStatus {
    status := HealthStatus{
        Service:   "disk",
        Timestamp: time.Now(),
    }
    
    // Get disk usage
    usage := getDiskUsage("/")
    
    if usage > 90 {
        status.Status = "critical"
        status.Message = fmt.Sprintf("Disk usage critical: %d%%", usage)
    } else if usage > 80 {
        status.Status = "warning"
        status.Message = fmt.Sprintf("Disk usage high: %d%%", usage)
    } else {
        status.Status = "healthy"
        status.Message = fmt.Sprintf("Disk usage normal: %d%%", usage)
    }
    
    return status
}

func (hc *HealthChecker) checkMemory() HealthStatus {
    status := HealthStatus{
        Service:   "memory",
        Timestamp: time.Now(),
    }
    
    // Get memory usage
    usage := getMemoryUsage()
    
    if usage > 90 {
        status.Status = "critical"
        status.Message = fmt.Sprintf("Memory usage critical: %d%%", usage)
    } else if usage > 80 {
        status.Status = "warning"
        status.Message = fmt.Sprintf("Memory usage high: %d%%", usage)
    } else {
        status.Status = "healthy"
        status.Message = fmt.Sprintf("Memory usage normal: %d%%", usage)
    }
    
    return status
}

// Automated remediation
func (hc *HealthChecker) AutoRemediate(status HealthStatus) error {
    switch status.Service {
    case "database":
        if status.Status == "unhealthy" {
            return hc.restartDatabase()
        }
    case "cache":
        if status.Status == "unhealthy" {
            return hc.clearCache()
        }
    case "disk":
        if status.Status == "critical" {
            return hc.cleanupDisk()
        }
    case "memory":
        if status.Status == "critical" {
            return hc.freeMemory()
        }
    default:
        if status.Status == "unhealthy" {
            return hc.restartService(status.Service)
        }
    }
    
    return nil
}
```

---

## 6. Log Management

### Log Rotation and Archival

```bash
#!/bin/bash
# log-management.sh

manage_logs() {
    echo "=== Log Management ==="
    
    # Rotate logs
    rotate_application_logs
    
    # Archive old logs
    archive_old_logs
    
    # Clean up disk space
    cleanup_log_space
    
    # Verify log shipping
    verify_log_shipping
}

rotate_application_logs() {
    echo "Rotating application logs..."
    
    # Application logs
    for logfile in /var/log/novacron/*.log; do
        if [ -f "$logfile" ]; then
            # Check size
            size=$(stat -f%z "$logfile" 2>/dev/null || stat -c%s "$logfile")
            
            if [ $size -gt 104857600 ]; then  # 100MB
                # Rotate log
                mv "$logfile" "${logfile}.$(date +%Y%m%d-%H%M%S)"
                touch "$logfile"
                
                # Signal application to reopen log files
                pkill -USR1 -f novacron
            fi
        fi
    done
    
    # Nginx logs
    nginx -s reopen
    
    # System logs
    logrotate -f /etc/logrotate.conf
}

archive_old_logs() {
    echo "Archiving old logs..."
    
    ARCHIVE_DIR="/archives/logs/$(date +%Y/%m)"
    mkdir -p $ARCHIVE_DIR
    
    # Find logs older than 7 days
    find /var/log -name "*.log.*" -mtime +7 -type f | while read logfile; do
        # Compress and archive
        gzip -9 "$logfile"
        mv "${logfile}.gz" $ARCHIVE_DIR/
    done
    
    # Upload to S3
    aws s3 sync $ARCHIVE_DIR s3://novacron-logs/archives/$(date +%Y/%m)/ \
        --storage-class GLACIER
    
    # Remove local archives older than 30 days
    find /archives/logs -mtime +30 -type f -delete
}

cleanup_log_space() {
    echo "Cleaning up log disk space..."
    
    # Check disk usage
    usage=$(df /var/log | tail -1 | awk '{print $5}' | sed 's/%//')
    
    if [ $usage -gt 80 ]; then
        echo "  High disk usage detected: ${usage}%"
        
        # Remove old compressed logs
        find /var/log -name "*.gz" -mtime +3 -delete
        
        # Truncate large active logs
        find /var/log -name "*.log" -size +1G -exec truncate -s 100M {} \;
        
        # Clean journal logs
        journalctl --vacuum-time=3d
        journalctl --vacuum-size=500M
    fi
}

verify_log_shipping() {
    echo "Verifying log shipping..."
    
    # Check if log shipper is running
    if ! systemctl is-active --quiet filebeat; then
        echo "  ⚠️  Filebeat not running, restarting..."
        systemctl restart filebeat
    fi
    
    # Check if logs are being shipped
    last_shipped=$(curl -s http://localhost:5066/stats | jq -r .publish_events.total)
    sleep 5
    current_shipped=$(curl -s http://localhost:5066/stats | jq -r .publish_events.total)
    
    if [ "$last_shipped" == "$current_shipped" ]; then
        echo "  ⚠️  No logs shipped in last 5 seconds"
        
        # Check and fix common issues
        systemctl restart filebeat
        
        # Check ElasticSearch connectivity
        curl -s http://elasticsearch:9200/_cluster/health
    else
        shipped=$((current_shipped - last_shipped))
        echo "  ✅ Shipped $shipped events in last 5 seconds"
    fi
}

# Run log management
manage_logs
```

---

## 7. Backup Verification

### Automated Backup Testing

```python
#!/usr/bin/env python3
# backup_verification.py

import os
import subprocess
import hashlib
import tempfile
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class BackupVerifier:
    def __init__(self):
        self.backup_location = "s3://novacron-backups/"
        self.test_results = []
        
    def verify_all_backups(self) -> Dict:
        """Verify all backup types"""
        
        print("=== Backup Verification ===")
        print(f"Verification time: {datetime.now()}")
        
        results = {
            'database': self.verify_database_backup(),
            'files': self.verify_file_backup(),
            'configuration': self.verify_config_backup(),
            'snapshots': self.verify_snapshots()
        }
        
        return results
    
    def verify_database_backup(self) -> Dict:
        """Verify database backup integrity"""
        
        print("\nVerifying database backups...")
        
        # Get latest backup
        latest_backup = self.get_latest_backup('database')
        
        if not latest_backup:
            return {'status': 'failed', 'error': 'No backup found'}
        
        # Download backup
        with tempfile.TemporaryDirectory() as tmpdir:
            local_file = f"{tmpdir}/db_backup.sql.gz"
            subprocess.run([
                'aws', 's3', 'cp',
                latest_backup,
                local_file
            ])
            
            # Verify checksum
            if not self.verify_checksum(local_file):
                return {'status': 'failed', 'error': 'Checksum mismatch'}
            
            # Test restore to temporary database
            if not self.test_database_restore(local_file):
                return {'status': 'failed', 'error': 'Restore test failed'}
                
        return {
            'status': 'success',
            'backup_file': latest_backup,
            'size': self.get_file_size(latest_backup),
            'age_hours': self.get_backup_age(latest_backup)
        }
    
    def test_database_restore(self, backup_file: str) -> bool:
        """Test database restore"""
        
        try:
            # Create test database
            subprocess.run([
                'createdb', 'novacron_test'
            ], check=True)
            
            # Restore backup
            with subprocess.Popen(['gunzip', '-c', backup_file], stdout=subprocess.PIPE) as gz:
                subprocess.run([
                    'psql', 'novacron_test'
                ], stdin=gz.stdout, check=True)
            
            # Verify data
            result = subprocess.run([
                'psql', 'novacron_test', '-c',
                'SELECT COUNT(*) FROM users'
            ], capture_output=True, text=True)
            
            # Cleanup
            subprocess.run(['dropdb', 'novacron_test'])
            
            return result.returncode == 0
            
        except Exception as e:
            print(f"  Restore test failed: {e}")
            return False
    
    def verify_file_backup(self) -> Dict:
        """Verify file backup integrity"""
        
        print("\nVerifying file backups...")
        
        # List recent file backups
        result = subprocess.run([
            'aws', 's3', 'ls',
            f'{self.backup_location}files/',
            '--recursive'
        ], capture_output=True, text=True)
        
        if not result.stdout:
            return {'status': 'failed', 'error': 'No file backups found'}
        
        # Parse backup files
        backups = []
        for line in result.stdout.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 4:
                backups.append({
                    'date': parts[0],
                    'time': parts[1],
                    'size': int(parts[2]),
                    'file': parts[3]
                })
        
        # Check backup age
        latest = backups[-1] if backups else None
        if latest:
            backup_time = datetime.strptime(f"{latest['date']} {latest['time']}", "%Y-%m-%d %H:%M:%S")
            age = datetime.now() - backup_time
            
            if age > timedelta(days=1):
                return {'status': 'warning', 'message': 'Backup older than 24 hours'}
                
        return {
            'status': 'success',
            'total_backups': len(backups),
            'latest_backup': latest,
            'total_size': sum(b['size'] for b in backups)
        }
    
    def verify_snapshots(self) -> Dict:
        """Verify EBS snapshots"""
        
        print("\nVerifying EBS snapshots...")
        
        import boto3
        ec2 = boto3.client('ec2')
        
        # List snapshots
        response = ec2.describe_snapshots(
            OwnerIds=['self'],
            Filters=[
                {'Name': 'tag:Application', 'Values': ['NovaCron']},
                {'Name': 'status', 'Values': ['completed']}
            ]
        )
        
        snapshots = response['Snapshots']
        
        if not snapshots:
            return {'status': 'failed', 'error': 'No snapshots found'}
        
        # Check snapshot age
        latest = max(snapshots, key=lambda x: x['StartTime'])
        age = datetime.now(latest['StartTime'].tzinfo) - latest['StartTime']
        
        status = 'success'
        if age > timedelta(days=7):
            status = 'warning'
        elif age > timedelta(days=30):
            status = 'critical'
            
        return {
            'status': status,
            'total_snapshots': len(snapshots),
            'latest_snapshot': latest['SnapshotId'],
            'age_days': age.days
        }
    
    def perform_restore_drill(self) -> bool:
        """Perform full restore drill"""
        
        print("\n=== Restore Drill ===")
        
        # Create test environment
        test_env = self.create_test_environment()
        
        try:
            # Restore database
            if not self.restore_database_drill(test_env):
                return False
                
            # Restore files
            if not self.restore_files_drill(test_env):
                return False
                
            # Restore configuration
            if not self.restore_config_drill(test_env):
                return False
                
            # Verify application starts
            if not self.verify_application_start(test_env):
                return False
                
            print("✅ Restore drill completed successfully")
            return True
            
        finally:
            # Cleanup test environment
            self.cleanup_test_environment(test_env)
    
    def generate_backup_report(self) -> str:
        """Generate backup verification report"""
        
        results = self.verify_all_backups()
        
        report = f"""
=== Backup Verification Report ===
Date: {datetime.now().isoformat()}

Backup Status:
"""
        
        for backup_type, result in results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            report += f"\n{status_icon} {backup_type}: {result['status']}"
            
            if result['status'] != 'success':
                report += f"\n   Issue: {result.get('error', result.get('message', 'Unknown'))}"
                
        # Add recommendations
        report += "\n\nRecommendations:"
        
        failed = [k for k, v in results.items() if v['status'] == 'failed']
        if failed:
            report += f"\n- CRITICAL: Fix failed backups immediately: {', '.join(failed)}"
            
        warnings = [k for k, v in results.items() if v['status'] == 'warning']
        if warnings:
            report += f"\n- WARNING: Review backup age for: {', '.join(warnings)}"
            
        if not failed and not warnings:
            report += "\n- All backups verified successfully"
            
        return report

# Run verification
if __name__ == "__main__":
    verifier = BackupVerifier()
    
    # Verify all backups
    report = verifier.generate_backup_report()
    print(report)
    
    # Save report
    with open(f"/reports/backup-verification-{datetime.now().strftime('%Y%m%d')}.txt", 'w') as f:
        f.write(report)
    
    # Optionally run restore drill
    if datetime.now().day == 1:  # First day of month
        verifier.perform_restore_drill()
```

---

## 8. Appendix

### Maintenance Commands Quick Reference

```bash
# Database maintenance
psql -c "VACUUM ANALYZE;"
psql -c "REINDEX DATABASE novacron;"
pg_repack -d novacron

# Cache maintenance
redis-cli BGREWRITEAOF
redis-cli BGSAVE
redis-cli --bigkeys
redis-cli MEMORY DOCTOR

# Disk cleanup
du -sh /var/log/*
find /tmp -mtime +7 -delete
docker system prune -a -f
journalctl --vacuum-time=7d

# Certificate management
certbot renew --dry-run
certbot certificates
openssl x509 -in cert.pem -noout -dates

# Backup operations
pg_dump novacron | gzip > backup.sql.gz
mongodump --archive --gzip
rsync -avz /data/ /backup/

# System updates
apt update && apt upgrade
yum update
npm update
pip install --upgrade pip

# Health checks
curl http://localhost/health
systemctl status novacron
docker ps --filter health=unhealthy

# Log analysis
tail -f /var/log/novacron/app.log
grep ERROR /var/log/novacron/*.log
journalctl -u novacron -f
```

### Maintenance Schedule Template

| Task | Frequency | Duration | Impact | Notification |
|------|-----------|----------|--------|--------------|
| Log rotation | Daily | 5 min | None | None |
| Cache cleanup | Daily | 10 min | Minor | None |
| Database vacuum | Weekly | 30 min | Minor | 24h advance |
| Index rebuild | Weekly | 1 hour | Moderate | 24h advance |
| Security updates | Weekly | 2 hours | Moderate | 48h advance |
| Full backup | Daily | 1 hour | Minor | None |
| Backup verification | Weekly | 30 min | None | None |
| Certificate renewal | Monthly | 15 min | None | 1 week advance |
| Major updates | Quarterly | 4 hours | Major | 2 weeks advance |
| DR drill | Quarterly | 8 hours | None | 1 month advance |

---

**Document Review Schedule**: Monthly
**Last Review**: 2025-01-05
**Next Review**: 2025-02-05
**Owner**: Platform Operations Team