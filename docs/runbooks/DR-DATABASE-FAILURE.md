# Disaster Recovery Runbook: Database Failure

**Severity:** CRITICAL
**RTO:** <1 hour
**RPO:** <15 minutes
**Owner:** Database Team
**Last Updated:** 2025-11-12

---

## Overview

This runbook covers procedures for recovering from complete database failure, including primary server crashes, data corruption, or infrastructure failures.

**When to use this runbook:**
- Primary database server is unresponsive
- Database corruption detected
- Complete data center failure affecting database
- Storage failure on database server

---

## Detection

### Automated Alerts

**Prometheus alerts that trigger:**
- `ServiceDown{service="postgresql"}` - Database unreachable for 2+ minutes
- `DatabaseConnectionPoolExhaustion` - Cannot establish new connections
- `HighDatabaseErrorRate` - >1% error rate for 5+ minutes

**Monitoring dashboards:**
- Grafana: http://grafana.novacron.io:3000/d/database-performance
- PagerDuty alert will fire to on-call DBA

### Manual Detection

1. **Check database connectivity:**
   ```bash
   psql -h db.novacron.io -U novacron -d novacron -c "SELECT current_timestamp;"
   ```
   Expected: Current timestamp returned
   Failure: Connection refused, timeout, or authentication error

2. **Check database service status:**
   ```bash
   kubectl get pods -n novacron | grep postgres
   ```
   Expected: Running (1/1)
   Failure: CrashLoopBackOff, Error, Pending

3. **Check database logs:**
   ```bash
   kubectl logs -n novacron postgres-primary-0 --tail=100
   ```
   Look for: "FATAL", "PANIC", "ERROR" messages

---

## Diagnosis

### Step 1: Determine Failure Type

**Database process crashed:**
```bash
# Check if PostgreSQL process is running
kubectl exec -n novacron postgres-primary-0 -- ps aux | grep postgres
```

**Storage failure:**
```bash
# Check disk usage and health
kubectl exec -n novacron postgres-primary-0 -- df -h
kubectl exec -n novacron postgres-primary-0 -- smartctl -H /dev/sda
```

**Data corruption:**
```bash
# Check PostgreSQL logs for corruption messages
kubectl logs -n novacron postgres-primary-0 | grep -i "corruption\|invalid\|damaged"
```

**Network partition:**
```bash
# Test network connectivity
kubectl exec -n novacron postgres-primary-0 -- ping -c 3 8.8.8.8
```

### Step 2: Check Standby Status

```bash
# Verify standby is healthy and up-to-date
psql -h standby.novacron.io -U postgres -c "SELECT pg_is_in_recovery(), pg_last_wal_replay_lsn();"
```

### Step 3: Assess Data Loss

```bash
# Check replication lag
psql -h standby.novacron.io -U postgres -c "SELECT EXTRACT(EPOCH FROM (now() - pg_last_xact_replay_timestamp()));"
```
- <15 minutes: Within RPO, proceed with failover
- >15 minutes: Escalate to engineering leadership

---

## Recovery Procedure

### Option A: Failover to Standby (Preferred)

**Prerequisites:**
- Standby database is healthy
- Replication lag <15 minutes
- DBA approval obtained

**Estimated Time:** 10-20 minutes

**Steps:**

1. **Notify stakeholders** (via Slack #incidents):
   ```
   INCIDENT: Database failover in progress
   ETA: 15 minutes
   Impact: Brief service interruption (30-60 seconds)
   ```

2. **Drain traffic from primary:**
   ```bash
   # Update load balancer to stop sending traffic
   kubectl patch service postgres-primary -n novacron \
     -p '{"spec":{"selector":{"role":"none"}}}'
   ```

3. **Wait for active connections to drain:**
   ```bash
   # Wait up to 60 seconds
   for i in {1..60}; do
     CONN_COUNT=$(psql -h db.novacron.io -U postgres -t -c "SELECT count(*) FROM pg_stat_activity WHERE datname='novacron';")
     if [ "$CONN_COUNT" -eq "0" ]; then
       break
     fi
     echo "Waiting for connections to drain... ($CONN_COUNT active)"
     sleep 1
   done
   ```

4. **Promote standby to primary:**
   ```bash
   cd /home/kp/novacron/scripts/failover
   ./promote-standby.sh
   ```

5. **Update DNS/Load Balancer:**
   ```bash
   # Point primary endpoint to new server
   kubectl patch service postgres-primary -n novacron \
     -p '{"spec":{"selector":{"statefulset.kubernetes.io/pod-name":"postgres-standby-0"}}}'
   ```

6. **Verify new primary is accepting connections:**
   ```bash
   psql -h db.novacron.io -U novacron -d novacron -c "SELECT current_timestamp, version();"
   ```

7. **Resume application traffic:**
   ```bash
   # Restart API servers to establish new connections
   kubectl rollout restart deployment/novacron-api -n novacron
   ```

8. **Monitor for 15 minutes:**
   - Watch Grafana dashboard for error rates
   - Check logs for connection errors
   - Verify application functionality

**Rollback:** If issues arise, can switch back to original primary if it recovers

### Option B: Restore from Backup

**Prerequisites:**
- Standby is also unavailable or too far behind
- Recent backup available (<24 hours)

**Estimated Time:** 45-60 minutes

**Steps:**

1. **Provision new database server:**
   ```bash
   kubectl apply -f k8s/postgres-restore.yaml
   ```

2. **Download latest backup:**
   ```bash
   aws s3 cp s3://novacron-backups/database/latest.sql.gz /tmp/
   ```

3. **Verify backup integrity:**
   ```bash
   gunzip -t /tmp/latest.sql.gz
   md5sum /tmp/latest.sql.gz
   ```

4. **Restore database:**
   ```bash
   cd /home/kp/novacron/scripts/restore
   export DB_HOST=new-db.novacron.io
   export DB_PASSWORD=$NEW_DB_PASSWORD
   ./database-restore.sh /tmp/latest.sql.gz
   ```

5. **Apply WAL archives (if available):**
   ```bash
   # Replay Write-Ahead Logs to minimize data loss
   pg_wal_replay /backup/wal_archive/
   ```

6. **Update application configuration:**
   ```bash
   kubectl set env deployment/novacron-api -n novacron \
     DATABASE_URL=postgresql://novacron:$DB_PASSWORD@new-db.novacron.io:5432/novacron
   ```

7. **Restart applications:**
   ```bash
   kubectl rollout restart deployment/novacron-api -n novacron
   kubectl rollout restart deployment/novacron-core -n novacron
   ```

---

## Verification

### Step 1: Connection Test
```bash
psql -h db.novacron.io -U novacron -d novacron << EOF
SELECT current_timestamp;
SELECT count(*) FROM vms;
SELECT count(*) FROM users;
EOF
```

### Step 2: Application Health Check
```bash
curl -f https://novacron.io/health
# Expected: {"status":"ok","database":"connected"}
```

### Step 3: Monitor Metrics
- Check Grafana dashboard: Database performance should be normal
- API error rate should be <1%
- API latency should be <100ms p95

### Step 4: Smoke Tests
```bash
# Create test VM
curl -X POST https://novacron.io/api/v1/vms \
  -H "Authorization: Bearer $API_TOKEN" \
  -d '{"name":"test-recovery","size":"small"}'

# Verify creation
curl https://novacron.io/api/v1/vms | jq '.[] | select(.name=="test-recovery")'

# Clean up
curl -X DELETE https://novacron.io/api/v1/vms/test-recovery
```

---

## Post-Recovery Tasks

1. **Document timeline** in incident postmortem
2. **Investigate root cause** of database failure
3. **Update monitoring** if gaps were identified
4. **Test failback** when original primary is restored
5. **Review backup procedures** and retention policies
6. **Schedule DR drill** to validate procedures

---

## Escalation

**Escalation path:**
1. On-call DBA (via PagerDuty)
2. Database Team Lead (if >30 minutes)
3. VP Engineering (if >1 hour or data loss)
4. CTO (if customer impact severe)

**Contact Information:**
- DBA Team: #dba-team Slack channel
- PagerDuty: https://novacron.pagerduty.com
- Emergency Hotline: 1-800-NOVACRON

---

## Related Runbooks

- [DR-SERVICE-FAILURE.md](DR-SERVICE-FAILURE.md) - Application recovery
- [DR-COMPLETE-RECOVERY.md](DR-COMPLETE-RECOVERY.md) - Full system recovery
- [RUNBOOK-BACKUP.md](RUNBOOK-BACKUP.md) - Backup procedures
- [RUNBOOK-RESTORE.md](RUNBOOK-RESTORE.md) - Restore procedures

---

## Testing History

| Date | Tester | Scenario | Result | Notes |
|------|--------|----------|--------|-------|
| 2025-11-12 | SRE Team | Failover to standby | Success | 12 minutes total, <1s downtime |
| 2025-11-12 | SRE Team | Restore from backup | Success | 52 minutes total, 3-minute data loss |

---

**Document Version:** 1.0
**Last Tested:** 2025-11-12
**Next Review:** 2025-12-12
