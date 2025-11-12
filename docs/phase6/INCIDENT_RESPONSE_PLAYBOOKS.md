# DWCP v3 Incident Response Playbooks

**Version**: 1.0.0
**Last Updated**: 2025-11-10
**Audience**: On-Call Engineers, SREs, Incident Response Team
**Classification**: Internal Use - Production Critical

---

## Table of Contents

1. [Overview](#overview)
2. [Incident Classification](#incident-classification)
3. [General Response Framework](#general-response-framework)
4. [Playbook: High Error Rate](#playbook-high-error-rate)
5. [Playbook: Service Outage](#playbook-service-outage)
6. [Playbook: Database Performance Degradation](#playbook-database-performance-degradation)
7. [Playbook: Memory Leak](#playbook-memory-leak)
8. [Playbook: Network Connectivity Issues](#playbook-network-connectivity-issues)
9. [Playbook: Security Breach](#playbook-security-breach)
10. [Playbook: Data Corruption](#playbook-data-corruption)
11. [Playbook: Capacity Exhaustion](#playbook-capacity-exhaustion)
12. [Playbook: Certificate Expiration](#playbook-certificate-expiration)
13. [Playbook: DDoS Attack](#playbook-ddos-attack)
14. [Playbook: Message Queue Backlog](#playbook-message-queue-backlog)
15. [Playbook: Cascading Failures](#playbook-cascading-failures)
16. [Post-Incident Review Process](#post-incident-review-process)

---

## Overview

### Purpose

This document provides step-by-step playbooks for responding to common incidents in the DWCP v3 distributed system. Each playbook includes detection methods, investigation steps, resolution procedures, and post-incident actions.

### Using This Document

1. **Identify the incident type** using symptoms and monitoring alerts
2. **Follow the appropriate playbook** step-by-step
3. **Document all actions** in the incident timeline
4. **Escalate when necessary** based on severity and time elapsed
5. **Complete post-incident review** after resolution

### Incident Response Principles

1. **Safety First**: Never make changes that could worsen the situation
2. **Minimize Impact**: Prioritize actions that reduce user impact
3. **Document Everything**: Record all observations and actions
4. **Communicate Clearly**: Keep stakeholders informed
5. **Learn and Improve**: Every incident is a learning opportunity

---

## Incident Classification

### Severity Levels

| Severity | Description | Response Time | Examples |
|----------|-------------|---------------|----------|
| SEV-1 | Critical - Complete service outage | Immediate | Total system down, data breach |
| SEV-2 | High - Major service degradation | <15 minutes | High error rate, regional outage |
| SEV-3 | Medium - Partial service degradation | <1 hour | Slow performance, minor features down |
| SEV-4 | Low - Minor issues | <4 hours | Non-critical warnings, cosmetic issues |

### Impact Assessment Matrix

| User Impact | Data Impact | Duration | Severity |
|-------------|-------------|----------|----------|
| >50% users | Data loss | Any | SEV-1 |
| >25% users | Integrity risk | >15 min | SEV-1 |
| >10% users | None | >30 min | SEV-2 |
| <10% users | None | >1 hour | SEV-3 |
| Minimal | None | Any | SEV-4 |

---

## General Response Framework

### Initial Response Checklist

```bash
#!/bin/bash
# General incident response initialization

INCIDENT_ID="inc-$(date +%Y%m%d-%H%M%S)"
SEVERITY=$1  # SEV-1, SEV-2, SEV-3, SEV-4

echo "=== INCIDENT RESPONSE ==="
echo "Incident ID: $INCIDENT_ID"
echo "Severity: $SEVERITY"
echo "Start Time: $(date -u)"

# 1. Create incident ticket
./scripts/create-incident.sh \
    --id $INCIDENT_ID \
    --severity $SEVERITY \
    --created-by "$(whoami)"

# 2. Start incident timer
echo $(date -u +%s) > /tmp/${INCIDENT_ID}-start

# 3. Create incident workspace
mkdir -p /incidents/${INCIDENT_ID}
cd /incidents/${INCIDENT_ID}

# 4. Initialize incident log
cat > incident.log << EOF
Incident ID: $INCIDENT_ID
Severity: $SEVERITY
Start Time: $(date -u)
Responder: $(whoami)

=== TIMELINE ===
EOF

# 5. Gather initial information
echo "$(date -u +%T) - Gathering system snapshot..." | tee -a incident.log
./scripts/system-snapshot.sh --output snapshot-initial.json

# 6. Check recent changes
echo "$(date -u +%T) - Checking recent changes..." | tee -a incident.log
./scripts/recent-changes.sh --hours 2 --output recent-changes.json

# 7. Collect error logs
echo "$(date -u +%T) - Collecting error logs..." | tee -a incident.log
./scripts/collect-errors.sh --since 30m --output errors.log

# 8. Set up monitoring
echo "$(date -u +%T) - Setting up enhanced monitoring..." | tee -a incident.log
./scripts/enable-monitoring.sh --incident $INCIDENT_ID --level enhanced

# 9. Notify team (based on severity)
if [[ "$SEVERITY" == "SEV-1" || "$SEVERITY" == "SEV-2" ]]; then
    echo "$(date -u +%T) - Paging response team..." | tee -a incident.log
    ./scripts/page-team.sh --severity $SEVERITY --incident $INCIDENT_ID
fi

# 10. Create incident channel
echo "$(date -u +%T) - Creating incident channel..." | tee -a incident.log
./scripts/create-incident-channel.sh --id $INCIDENT_ID

echo "=== INITIAL RESPONSE COMPLETE ==="
echo "Incident Channel: #incident-${INCIDENT_ID}"
echo "Incident Workspace: /incidents/${INCIDENT_ID}"
echo ""
echo "Next Steps:"
echo "1. Identify incident type and follow appropriate playbook"
echo "2. Update incident channel every 5 minutes"
echo "3. Document all actions in incident.log"
```

### Communication Template

**Initial Alert**:
```
🚨 INCIDENT ALERT 🚨

Incident ID: [INCIDENT_ID]
Severity: [SEV-X]
Status: Investigating
Start Time: [TIME]

Description: [Brief description of the issue]
Impact: [User impact description]
Current Actions: [What is being done right now]

Updates will be provided every 5 minutes.
Incident Channel: #incident-[ID]
```

**Status Update** (every 5 minutes):
```
⏱️ UPDATE - [TIME]

Incident ID: [INCIDENT_ID]
Status: [Investigating / Mitigating / Resolved]

Progress: [What has been discovered/done]
Next Steps: [What will be done next]
ETA: [Estimated time to resolution, if known]
```

**Resolution Notice**:
```
✅ RESOLVED

Incident ID: [INCIDENT_ID]
Resolution Time: [TIME]
Duration: [DURATION]

Summary: [What happened and how it was resolved]
Root Cause: [If known]
Impact: [Final impact assessment]

Post-Incident Review: Scheduled for [DATE/TIME]
```

---

## Playbook: High Error Rate

### Detection

**Alert**: Error rate > 1% for 2+ minutes
**Symptoms**:
- Increased 5xx HTTP responses
- User complaints about errors
- Spike in error logs

### Investigation

```bash
#!/bin/bash
# High error rate investigation

INCIDENT_ID=$1

echo "=== INVESTIGATING HIGH ERROR RATE ==="
echo "Incident: $INCIDENT_ID"

# 1. Confirm current error rate
echo "Step 1: Checking current error rate..."
CURRENT_ERROR_RATE=$(curl -s "http://prometheus.dwcp.io/api/v1/query?query=sum(rate(http_requests_total{status=~\"5..\"}[5m]))/sum(rate(http_requests_total[5m]))*100" | jq -r '.data.result[0].value[1]')
echo "Current error rate: ${CURRENT_ERROR_RATE}%"

# 2. Identify which services are failing
echo "Step 2: Identifying failing services..."
kubectl get pods --all-namespaces -o json | \
    jq -r '.items[] | select(.status.containerStatuses[]?.restartCount > 0) | "\(.metadata.namespace)/\(.metadata.name) - Restarts: \(.status.containerStatuses[0].restartCount)"'

# 3. Check for recent deployments
echo "Step 3: Checking recent deployments..."
./scripts/recent-deployments.sh --hours 2

# 4. Analyze error patterns
echo "Step 4: Analyzing error patterns..."
kubectl logs --all-namespaces --selector=app=dwcp-api --since=10m | \
    grep -i "error\|exception\|failed" | \
    awk '{print $5}' | sort | uniq -c | sort -rn | head -20

# 5. Check dependencies
echo "Step 5: Checking external dependencies..."
./scripts/check-dependencies.sh --timeout 5

# 6. Resource utilization
echo "Step 6: Checking resource utilization..."
kubectl top pods --all-namespaces --sort-by=cpu | head -20

# 7. Database health
echo "Step 7: Checking database health..."
./scripts/check-database.sh --comprehensive

# 8. Generate hypothesis
echo "Step 8: Generating hypothesis..."
echo "=== HYPOTHESIS ==="
if [ "$(./scripts/recent-deployments.sh --hours 1 | wc -l)" -gt "0" ]; then
    echo "Likely cause: Recent deployment"
    echo "Recommendation: Rollback deployment"
elif [ "$(./scripts/check-dependencies.sh --timeout 5 | grep -c "FAILED")" -gt "0" ]; then
    echo "Likely cause: Dependency failure"
    echo "Recommendation: Enable degraded mode or failover"
else
    echo "Likely cause: Resource exhaustion or database issue"
    echo "Recommendation: Scale up or optimize queries"
fi
```

### Resolution Steps

#### Scenario A: Recent Deployment Causing Errors

```bash
#!/bin/bash
# Rollback recent deployment

INCIDENT_ID=$1
SERVICE_NAME=$2

echo "=== ROLLING BACK DEPLOYMENT ==="
echo "Incident: $INCIDENT_ID"
echo "Service: $SERVICE_NAME"

# 1. Confirm decision to rollback
echo "$(date -u +%T) - Decision: Rollback deployment" | tee -a /incidents/${INCIDENT_ID}/incident.log

# 2. Perform rollback
echo "$(date -u +%T) - Executing rollback..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl rollout undo deployment/$SERVICE_NAME -n production

# 3. Monitor rollback progress
kubectl rollout status deployment/$SERVICE_NAME -n production

# 4. Verify error rate decrease
for i in {1..10}; do
    sleep 30
    ERROR_RATE=$(curl -s "http://prometheus.dwcp.io/api/v1/query?query=sum(rate(http_requests_total{status=~\"5..\"}[5m]))/sum(rate(http_requests_total[5m]))*100" | jq -r '.data.result[0].value[1]')
    echo "$(date -u +%T) - Error rate after ${i*30}s: ${ERROR_RATE}%" | tee -a /incidents/${INCIDENT_ID}/incident.log

    if (( $(echo "$ERROR_RATE < 0.5" | bc -l) )); then
        echo "$(date -u +%T) - Error rate normalized" | tee -a /incidents/${INCIDENT_ID}/incident.log
        break
    fi
done

# 5. Document resolution
cat >> /incidents/${INCIDENT_ID}/incident.log << EOF
$(date -u +%T) - Resolution: Deployment rollback
Service: $SERVICE_NAME
Final Error Rate: ${ERROR_RATE}%
EOF
```

#### Scenario B: Dependency Failure

```bash
#!/bin/bash
# Enable degraded mode due to dependency failure

INCIDENT_ID=$1
FAILED_DEPENDENCY=$2

echo "=== ENABLING DEGRADED MODE ==="
echo "Incident: $INCIDENT_ID"
echo "Failed Dependency: $FAILED_DEPENDENCY"

# 1. Enable degraded mode configuration
echo "$(date -u +%T) - Enabling degraded mode..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl set env deployment/dwcp-api \
    DEGRADED_MODE=true \
    DISABLE_${FAILED_DEPENDENCY}=true

# 2. Update circuit breaker thresholds
kubectl patch configmap circuit-breaker-config -p \
    '{"data":{"'${FAILED_DEPENDENCY}'_threshold":"0.1"}}'

# 3. Monitor error rate
sleep 60
ERROR_RATE=$(curl -s "http://prometheus.dwcp.io/api/v1/query?query=sum(rate(http_requests_total{status=~\"5..\"}[5m]))/sum(rate(http_requests_total[5m]))*100" | jq -r '.data.result[0].value[1]')
echo "$(date -u +%T) - Error rate after degraded mode: ${ERROR_RATE}%" | tee -a /incidents/${INCIDENT_ID}/incident.log

# 4. Notify users
./scripts/notify-users.sh --message "Some features temporarily unavailable due to maintenance"
```

#### Scenario C: Resource Exhaustion

```bash
#!/bin/bash
# Scale up due to resource exhaustion

INCIDENT_ID=$1
SERVICE_NAME=$2

echo "=== SCALING UP SERVICE ==="
echo "Incident: $INCIDENT_ID"
echo "Service: $SERVICE_NAME"

# 1. Get current replica count
CURRENT_REPLICAS=$(kubectl get deployment/$SERVICE_NAME -n production -o jsonpath='{.spec.replicas}')
TARGET_REPLICAS=$((CURRENT_REPLICAS * 2))

echo "$(date -u +%T) - Current replicas: $CURRENT_REPLICAS" | tee -a /incidents/${INCIDENT_ID}/incident.log
echo "$(date -u +%T) - Target replicas: $TARGET_REPLICAS" | tee -a /incidents/${INCIDENT_ID}/incident.log

# 2. Scale up
kubectl scale deployment/$SERVICE_NAME -n production --replicas=$TARGET_REPLICAS

# 3. Wait for new pods to be ready
kubectl rollout status deployment/$SERVICE_NAME -n production

# 4. Monitor error rate
for i in {1..5}; do
    sleep 30
    ERROR_RATE=$(curl -s "http://prometheus.dwcp.io/api/v1/query?query=sum(rate(http_requests_total{status=~\"5..\"}[5m]))/sum(rate(http_requests_total[5m]))*100" | jq -r '.data.result[0].value[1]')
    echo "$(date -u +%T) - Error rate after ${i*30}s: ${ERROR_RATE}%" | tee -a /incidents/${INCIDENT_ID}/incident.log
done
```

### Post-Incident Actions

```bash
#!/bin/bash
# Post-incident actions for high error rate

INCIDENT_ID=$1

# 1. Collect final metrics
./scripts/collect-metrics.sh --incident $INCIDENT_ID --period incident

# 2. Generate incident report
./scripts/generate-report.sh --incident $INCIDENT_ID --template high-error-rate

# 3. Create action items
cat > /incidents/${INCIDENT_ID}/action-items.md << 'EOF'
# Action Items

## Immediate
- [ ] Review deployment process to prevent similar incidents
- [ ] Add additional monitoring for early detection
- [ ] Update runbook based on lessons learned

## Short-term
- [ ] Implement better circuit breakers
- [ ] Add automated rollback on error spike
- [ ] Improve error logging and categorization

## Long-term
- [ ] Implement canary deployments
- [ ] Add chaos testing for dependency failures
- [ ] Create automated remediation for common scenarios
EOF

# 4. Schedule post-incident review
./scripts/schedule-pir.sh --incident $INCIDENT_ID --date "+2 days"
```

---

## Playbook: Service Outage

### Detection

**Alert**: Service availability < 50% for 1+ minute
**Symptoms**:
- Multiple health check failures
- Load balancer marking backends as unhealthy
- Complete service unavailability

### Investigation

```bash
#!/bin/bash
# Service outage investigation

INCIDENT_ID=$1
SERVICE_NAME=$2

echo "=== INVESTIGATING SERVICE OUTAGE ==="
echo "Incident: $INCIDENT_ID"
echo "Service: $SERVICE_NAME"

# 1. Verify outage
echo "Step 1: Verifying outage..."
HEALTHY_PODS=$(kubectl get pods -n production -l app=$SERVICE_NAME -o json | jq -r '.items[] | select(.status.phase=="Running" and .status.containerStatuses[].ready==true) | .metadata.name' | wc -l)
TOTAL_PODS=$(kubectl get pods -n production -l app=$SERVICE_NAME -o json | jq -r '.items | length')
echo "Healthy pods: $HEALTHY_PODS / $TOTAL_PODS"

# 2. Check pod status
echo "Step 2: Checking pod status..."
kubectl get pods -n production -l app=$SERVICE_NAME -o wide

# 3. Check pod logs for all replicas
echo "Step 3: Collecting pod logs..."
for pod in $(kubectl get pods -n production -l app=$SERVICE_NAME -o jsonpath='{.items[*].metadata.name}'); do
    echo "=== Logs for $pod ==="
    kubectl logs -n production $pod --tail=50 | tail -20
done

# 4. Check events
echo "Step 4: Checking recent events..."
kubectl get events -n production --sort-by='.lastTimestamp' | grep $SERVICE_NAME | tail -20

# 5. Check node health
echo "Step 5: Checking node health..."
kubectl get nodes -o wide

# 6. Check resource quotas
echo "Step 6: Checking resource quotas..."
kubectl describe resourcequota -n production

# 7. Check network policies
echo "Step 7: Checking network policies..."
kubectl get networkpolicies -n production -o yaml

# 8. Test service connectivity
echo "Step 8: Testing service connectivity..."
kubectl run test-pod --image=curlimages/curl --rm -it --restart=Never -- curl -v http://${SERVICE_NAME}.production.svc.cluster.local/health
```

### Resolution Steps

#### Scenario A: Pods CrashLooping

```bash
#!/bin/bash
# Fix CrashLooping pods

INCIDENT_ID=$1
SERVICE_NAME=$2

echo "=== RESOLVING CRASHLOOP ==="

# 1. Get crash reason
echo "$(date -u +%T) - Analyzing crash reason..." | tee -a /incidents/${INCIDENT_ID}/incident.log
CRASH_REASON=$(kubectl get pods -n production -l app=$SERVICE_NAME -o json | jq -r '.items[0].status.containerStatuses[0].lastState.terminated.reason')
echo "Crash reason: $CRASH_REASON"

# 2. Check if it's a configuration issue
if kubectl get pods -n production -l app=$SERVICE_NAME -o json | jq -r '.items[0].status.containerStatuses[0].lastState.terminated.exitCode' | grep -q "1"; then
    echo "$(date -u +%T) - Configuration error detected" | tee -a /incidents/${INCIDENT_ID}/incident.log

    # Rollback to last known good configuration
    ./scripts/rollback-config.sh --service $SERVICE_NAME

elif kubectl get pods -n production -l app=$SERVICE_NAME -o json | jq -r '.items[0].status.containerStatuses[0].lastState.terminated.reason' | grep -q "OOMKilled"; then
    echo "$(date -u +%T) - OOM detected, increasing memory" | tee -a /incidents/${INCIDENT_ID}/incident.log

    # Increase memory limits
    kubectl set resources deployment/$SERVICE_NAME -n production \
        --limits=memory=4Gi \
        --requests=memory=2Gi

else
    echo "$(date -u +%T) - Unknown crash, rolling back deployment" | tee -a /incidents/${INCIDENT_ID}/incident.log

    # Rollback deployment
    kubectl rollout undo deployment/$SERVICE_NAME -n production
fi

# 3. Wait for pods to stabilize
kubectl rollout status deployment/$SERVICE_NAME -n production

# 4. Verify health
./scripts/health-check.sh --service $SERVICE_NAME
```

#### Scenario B: Node Failure

```bash
#!/bin/bash
# Recover from node failure

INCIDENT_ID=$1
FAILED_NODE=$2

echo "=== RECOVERING FROM NODE FAILURE ==="

# 1. Cordon failed node
echo "$(date -u +%T) - Cordoning failed node: $FAILED_NODE" | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl cordon $FAILED_NODE

# 2. Drain node
echo "$(date -u +%T) - Draining node..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl drain $FAILED_NODE --ignore-daemonsets --delete-emptydir-data --timeout=5m

# 3. Verify pods rescheduled
echo "$(date -u +%T) - Verifying pod rescheduling..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl get pods --all-namespaces -o wide | grep -v $FAILED_NODE | grep Running | wc -l

# 4. Check cluster capacity
AVAILABLE_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.spec.unschedulable!=true) | .metadata.name' | wc -l)
echo "$(date -u +%T) - Available nodes: $AVAILABLE_NODES" | tee -a /incidents/${INCIDENT_ID}/incident.log

# 5. If capacity is low, add nodes
if [ $AVAILABLE_NODES -lt 10 ]; then
    echo "$(date -u +%T) - Low capacity, adding nodes..." | tee -a /incidents/${INCIDENT_ID}/incident.log
    ./scripts/add-nodes.sh --count 3
fi

# 6. Attempt to recover failed node
echo "$(date -u +%T) - Attempting node recovery..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/recover-node.sh --node $FAILED_NODE
```

#### Scenario C: Network Partition

```bash
#!/bin/bash
# Recover from network partition

INCIDENT_ID=$1

echo "=== RECOVERING FROM NETWORK PARTITION ==="

# 1. Identify partitioned nodes
echo "$(date -u +%T) - Identifying partitioned nodes..." | tee -a /incidents/${INCIDENT_ID}/incident.log
PARTITIONED_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.status.conditions[] | select(.type=="Ready" and .status=="Unknown")) | .metadata.name')

# 2. Verify network connectivity
echo "$(date -u +%T) - Testing network connectivity..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/network-test.sh --nodes "$PARTITIONED_NODES"

# 3. Restart network components
echo "$(date -u +%T) - Restarting network components..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl rollout restart daemonset/kube-proxy -n kube-system
kubectl rollout restart daemonset/calico-node -n kube-system

# 4. Wait for network recovery
sleep 60

# 5. Verify pod networking
echo "$(date -u +%T) - Verifying pod networking..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/verify-pod-networking.sh

# 6. If still partitioned, failover to backup region
if [ ! -z "$PARTITIONED_NODES" ]; then
    echo "$(date -u +%T) - Network partition persists, initiating regional failover..." | tee -a /incidents/${INCIDENT_ID}/incident.log
    ./scripts/regional-failover.sh --from us-east-1 --to us-west-2
fi
```

### Post-Incident Actions

```bash
#!/bin/bash
# Post-incident actions for service outage

INCIDENT_ID=$1

# 1. Calculate impact
DOWNTIME=$(./scripts/calculate-downtime.sh --incident $INCIDENT_ID)
AFFECTED_USERS=$(./scripts/calculate-affected-users.sh --incident $INCIDENT_ID)

# 2. Document incident
cat >> /incidents/${INCIDENT_ID}/incident.log << EOF

=== FINAL SUMMARY ===
Downtime: $DOWNTIME
Affected Users: $AFFECTED_USERS
Root Cause: [To be determined in PIR]
EOF

# 3. Create action items
cat > /incidents/${INCIDENT_ID}/action-items.md << 'EOF'
# Action Items

## Critical
- [ ] Add redundancy to prevent single point of failure
- [ ] Implement automated failover
- [ ] Increase monitoring sensitivity for early detection

## High Priority
- [ ] Add health checks for all critical dependencies
- [ ] Implement circuit breakers
- [ ] Create automated recovery procedures

## Medium Priority
- [ ] Review and update capacity planning
- [ ] Improve documentation for similar incidents
- [ ] Add chaos testing scenarios
EOF
```

---

## Playbook: Database Performance Degradation

### Detection

**Alert**: Database query latency > 1 second (p95) for 5+ minutes
**Symptoms**:
- Slow API responses
- Database connection pool exhaustion
- High database CPU/memory usage

### Investigation

```bash
#!/bin/bash
# Database performance investigation

INCIDENT_ID=$1
DB_TYPE=${2:-"postgres"}  # postgres, mysql, redis

echo "=== INVESTIGATING DATABASE PERFORMANCE ==="
echo "Incident: $INCIDENT_ID"
echo "Database Type: $DB_TYPE"

# 1. Check database metrics
echo "Step 1: Checking database metrics..."
if [ "$DB_TYPE" == "postgres" ]; then
    kubectl exec -n production deployment/postgres-primary -- \
        psql -c "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"

    kubectl exec -n production deployment/postgres-primary -- \
        psql -c "SELECT query, state, wait_event_type, wait_event, now() - state_change as duration FROM pg_stat_activity WHERE state != 'idle' ORDER BY duration DESC LIMIT 10;"
fi

# 2. Identify slow queries
echo "Step 2: Identifying slow queries..."
if [ "$DB_TYPE" == "postgres" ]; then
    kubectl exec -n production deployment/postgres-primary -- \
        psql -c "SELECT query, calls, total_time, mean_time, max_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 20;"
fi

# 3. Check for locks
echo "Step 3: Checking for locks..."
if [ "$DB_TYPE" == "postgres" ]; then
    kubectl exec -n production deployment/postgres-primary -- \
        psql -c "SELECT blocked_locks.pid AS blocked_pid, blocked_activity.usename AS blocked_user, blocking_locks.pid AS blocking_pid, blocking_activity.usename AS blocking_user, blocked_activity.query AS blocked_statement, blocking_activity.query AS current_statement_in_blocking_process FROM pg_catalog.pg_locks blocked_locks JOIN pg_catalog.pg_stat_activity blocked_activity ON blocked_activity.pid = blocked_locks.pid JOIN pg_catalog.pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype AND blocking_locks.relation = blocked_locks.relation AND blocking_locks.page = blocked_locks.page AND blocking_locks.tuple = blocked_locks.tuple AND blocking_locks.virtualxid = blocked_locks.virtualxid AND blocking_locks.transactionid = blocked_locks.transactionid AND blocking_locks.classid = blocked_locks.classid AND blocking_locks.objid = blocked_locks.objid AND blocking_locks.objsubid = blocked_locks.objsubid AND blocking_locks.pid != blocked_locks.pid JOIN pg_catalog.pg_stat_activity blocking_activity ON blocking_activity.pid = blocking_locks.pid WHERE NOT blocked_locks.granted;"
fi

# 4. Check database size and bloat
echo "Step 4: Checking database size and bloat..."
if [ "$DB_TYPE" == "postgres" ]; then
    kubectl exec -n production deployment/postgres-primary -- \
        psql -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size, n_dead_tup, n_live_tup FROM pg_stat_user_tables ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC LIMIT 20;"
fi

# 5. Check connection pool
echo "Step 5: Checking connection pool..."
./scripts/check-connection-pool.sh --db $DB_TYPE

# 6. Check for missing indexes
echo "Step 6: Checking for missing indexes..."
if [ "$DB_TYPE" == "postgres" ]; then
    kubectl exec -n production deployment/postgres-primary -- \
        psql -c "SELECT schemaname, tablename, seq_scan, seq_tup_read, idx_scan, seq_tup_read / NULLIF(seq_scan, 0) as avg_seq_tup FROM pg_stat_user_tables WHERE seq_scan > 0 ORDER BY seq_tup_read DESC LIMIT 20;"
fi

# 7. Check database resource usage
echo "Step 7: Checking resource usage..."
kubectl top pods -n production -l app=postgres

# 8. Generate hypothesis
echo "=== HYPOTHESIS ==="
./scripts/analyze-db-issue.sh --incident $INCIDENT_ID --db $DB_TYPE
```

### Resolution Steps

#### Scenario A: Slow Query

```bash
#!/bin/bash
# Resolve slow query issue

INCIDENT_ID=$1
SLOW_QUERY_HASH=$2

echo "=== RESOLVING SLOW QUERY ==="

# 1. Kill long-running query
echo "$(date -u +%T) - Killing slow query..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl exec -n production deployment/postgres-primary -- \
    psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE query_start < now() - interval '5 minutes' AND state = 'active';"

# 2. Analyze query plan
echo "$(date -u +%T) - Analyzing query plan..." | tee -a /incidents/${INCIDENT_ID}/incident.log
QUERY=$(kubectl exec -n production deployment/postgres-primary -- \
    psql -t -c "SELECT query FROM pg_stat_statements WHERE queryid = $SLOW_QUERY_HASH;")

kubectl exec -n production deployment/postgres-primary -- \
    psql -c "EXPLAIN ANALYZE $QUERY;" > /incidents/${INCIDENT_ID}/query-plan.txt

# 3. Create missing index if needed
if grep -q "Seq Scan" /incidents/${INCIDENT_ID}/query-plan.txt; then
    echo "$(date -u +%T) - Creating missing index..." | tee -a /incidents/${INCIDENT_ID}/incident.log

    # Extract table and column from query (simplified)
    TABLE=$(echo "$QUERY" | grep -oP "FROM \K\w+")
    COLUMN=$(echo "$QUERY" | grep -oP "WHERE \K\w+")

    kubectl exec -n production deployment/postgres-primary -- \
        psql -c "CREATE INDEX CONCURRENTLY idx_${TABLE}_${COLUMN} ON ${TABLE}(${COLUMN});"
fi

# 4. Monitor query performance
sleep 60
NEW_MEAN_TIME=$(kubectl exec -n production deployment/postgres-primary -- \
    psql -t -c "SELECT mean_time FROM pg_stat_statements WHERE queryid = $SLOW_QUERY_HASH;")

echo "$(date -u +%T) - New mean query time: ${NEW_MEAN_TIME}ms" | tee -a /incidents/${INCIDENT_ID}/incident.log
```

#### Scenario B: Connection Pool Exhaustion

```bash
#!/bin/bash
# Resolve connection pool exhaustion

INCIDENT_ID=$1

echo "=== RESOLVING CONNECTION POOL EXHAUSTION ==="

# 1. Kill idle connections
echo "$(date -u +%T) - Killing idle connections..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl exec -n production deployment/postgres-primary -- \
    psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < now() - interval '5 minutes';"

# 2. Increase connection pool size
echo "$(date -u +%T) - Increasing connection pool size..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl set env deployment/dwcp-api \
    DB_POOL_SIZE=100 \
    DB_POOL_MAX=150

# 3. Add read replica
echo "$(date -u +%T) - Adding read replica..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl scale statefulset/postgres-replica -n production --replicas=3

# 4. Configure connection pooler (PgBouncer)
echo "$(date -u +%T) - Deploying PgBouncer..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl apply -f /manifests/pgbouncer.yaml

# 5. Update applications to use pooler
kubectl set env deployment/dwcp-api \
    DB_HOST=pgbouncer.production.svc.cluster.local \
    DB_PORT=6432

# 6. Monitor connection count
for i in {1..5}; do
    sleep 30
    CONNECTIONS=$(kubectl exec -n production deployment/postgres-primary -- \
        psql -t -c "SELECT count(*) FROM pg_stat_activity;")
    echo "$(date -u +%T) - Active connections: $CONNECTIONS" | tee -a /incidents/${INCIDENT_ID}/incident.log
done
```

#### Scenario C: Table Bloat

```bash
#!/bin/bash
# Resolve table bloat

INCIDENT_ID=$1

echo "=== RESOLVING TABLE BLOAT ==="

# 1. Identify bloated tables
echo "$(date -u +%T) - Identifying bloated tables..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl exec -n production deployment/postgres-primary -- \
    psql -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size, n_dead_tup FROM pg_stat_user_tables WHERE n_dead_tup > 10000 ORDER BY n_dead_tup DESC LIMIT 10;" \
    > /incidents/${INCIDENT_ID}/bloated-tables.txt

# 2. Run VACUUM ANALYZE on bloated tables
while IFS= read -r line; do
    TABLE=$(echo "$line" | awk '{print $2}')
    echo "$(date -u +%T) - Vacuuming table: $TABLE..." | tee -a /incidents/${INCIDENT_ID}/incident.log

    kubectl exec -n production deployment/postgres-primary -- \
        psql -c "VACUUM ANALYZE $TABLE;"
done < /incidents/${INCIDENT_ID}/bloated-tables.txt

# 3. Enable auto-vacuum if not enabled
kubectl exec -n production deployment/postgres-primary -- \
    psql -c "ALTER SYSTEM SET autovacuum = on; SELECT pg_reload_conf();"

# 4. Monitor bloat reduction
sleep 300
kubectl exec -n production deployment/postgres-primary -- \
    psql -c "SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size, n_dead_tup FROM pg_stat_user_tables ORDER BY n_dead_tup DESC LIMIT 10;"
```

### Post-Incident Actions

```bash
#!/bin/bash
# Post-incident actions for database performance

INCIDENT_ID=$1

# Action items
cat > /incidents/${INCIDENT_ID}/action-items.md << 'EOF'
# Action Items

## Immediate
- [ ] Review and optimize identified slow queries
- [ ] Add missing indexes
- [ ] Configure auto-vacuum properly
- [ ] Implement connection pooling

## Short-term
- [ ] Set up query performance monitoring
- [ ] Create alerts for slow queries
- [ ] Implement query timeout limits
- [ ] Add database read replicas

## Long-term
- [ ] Implement database sharding strategy
- [ ] Set up automated index recommendation
- [ ] Create database performance testing suite
- [ ] Implement query caching layer
EOF
```

---

## Playbook: Memory Leak

### Detection

**Alert**: Memory utilization increasing steadily over 24 hours
**Symptoms**:
- Gradual memory increase
- OOMKilled pods
- Performance degradation over time

### Investigation

```bash
#!/bin/bash
# Memory leak investigation

INCIDENT_ID=$1
SERVICE_NAME=$2

echo "=== INVESTIGATING MEMORY LEAK ==="
echo "Incident: $INCIDENT_ID"
echo "Service: $SERVICE_NAME"

# 1. Check memory trends
echo "Step 1: Analyzing memory trends..."
./scripts/analyze-memory-trend.sh --service $SERVICE_NAME --hours 24

# 2. Get heap dump from running pod
echo "Step 2: Capturing heap dump..."
POD=$(kubectl get pods -n production -l app=$SERVICE_NAME -o jsonpath='{.items[0].metadata.name}')
kubectl exec -n production $POD -- kill -USR1 1  # Trigger heap dump (if supported)

# 3. Download heap dump
kubectl cp production/$POD:/tmp/heapdump.hprof /incidents/${INCIDENT_ID}/heapdump.hprof

# 4. Check for goroutine leaks (if Go application)
kubectl exec -n production $POD -- curl http://localhost:6060/debug/pprof/goroutine?debug=1 \
    > /incidents/${INCIDENT_ID}/goroutines.txt

# 5. Analyze object allocation
kubectl exec -n production $POD -- curl http://localhost:6060/debug/pprof/heap \
    > /incidents/${INCIDENT_ID}/heap.pprof

# 6. Check for file descriptor leaks
kubectl exec -n production $POD -- ls -la /proc/1/fd | wc -l

# 7. Review recent code changes
echo "Step 7: Reviewing recent code changes..."
./scripts/recent-changes.sh --service $SERVICE_NAME --days 7 --type code
```

### Resolution Steps

#### Immediate Mitigation

```bash
#!/bin/bash
# Immediate memory leak mitigation

INCIDENT_ID=$1
SERVICE_NAME=$2

echo "=== MITIGATING MEMORY LEAK ==="

# 1. Restart affected pods (temporary fix)
echo "$(date -u +%T) - Restarting affected pods..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl rollout restart deployment/$SERVICE_NAME -n production

# 2. Implement pod restart policy based on memory usage
echo "$(date -u +%T) - Implementing automated restart policy..." | tee -a /incidents/${INCIDENT_ID}/incident.log
cat > /tmp/memory-restart-policy.yaml << 'EOF'
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: SERVICE_NAME-pdb
spec:
  maxUnavailable: 1
  selector:
    matchLabels:
      app: SERVICE_NAME
EOF

sed "s/SERVICE_NAME/$SERVICE_NAME/g" /tmp/memory-restart-policy.yaml | kubectl apply -f -

# 3. Set up automated pod cycling based on memory
kubectl patch deployment $SERVICE_NAME -n production -p \
    '{"spec":{"template":{"metadata":{"annotations":{"auto-restart-on-memory":"true","memory-threshold":"2Gi"}}}}}'

# 4. Monitor memory after restart
for i in {1..10}; do
    sleep 60
    MEMORY=$(kubectl top pods -n production -l app=$SERVICE_NAME --no-headers | awk '{sum+=$3} END {print sum}')
    echo "$(date -u +%T) - Total memory usage after ${i} minutes: ${MEMORY}Mi" | tee -a /incidents/${INCIDENT_ID}/incident.log
done
```

#### Root Cause Fix

```bash
#!/bin/bash
# Fix memory leak root cause

INCIDENT_ID=$1
SERVICE_NAME=$2

echo "=== FIXING MEMORY LEAK ROOT CAUSE ==="

# 1. Analyze heap dump
echo "$(date -u +%T) - Analyzing heap dump..." | tee -a /incidents/${INCIDENT_ID}/incident.log
# Use appropriate tool based on language (jhat for Java, go tool pprof for Go, etc.)
go tool pprof -http=:8080 /incidents/${INCIDENT_ID}/heap.pprof &

# 2. Identify leak pattern from analysis
# (Manual step - examine heap dump in browser)

# 3. Apply fix (example: fix goroutine leak)
echo "$(date -u +%T) - Applying code fix..." | tee -a /incidents/${INCIDENT_ID}/incident.log

# Create patch to fix leak (example)
cat > /tmp/leak-fix.patch << 'EOF'
diff --git a/pkg/worker/pool.go b/pkg/worker/pool.go
index abc1234..def5678 100644
--- a/pkg/worker/pool.go
+++ b/pkg/worker/pool.go
@@ -45,6 +45,7 @@ func (p *Pool) Start(ctx context.Context) {
        for {
                select {
                case <-ctx.Done():
+                       close(p.workCh)
                        return
                case work := <-p.workCh:
                        p.process(work)
EOF

# 4. Deploy fix
./scripts/deploy-hotfix.sh --service $SERVICE_NAME --patch /tmp/leak-fix.patch

# 5. Monitor memory for 24 hours
./scripts/monitor-memory.sh --service $SERVICE_NAME --duration 24h --alert-on-trend
```

### Post-Incident Actions

```bash
# Action items
cat > /incidents/${INCIDENT_ID}/action-items.md << 'EOF'
# Action Items

## Critical
- [ ] Add memory profiling to CI/CD pipeline
- [ ] Implement automated memory leak detection
- [ ] Add resource limits to all pods

## High Priority
- [ ] Create memory leak testing scenarios
- [ ] Add memory usage alerts
- [ ] Document common memory leak patterns

## Medium Priority
- [ ] Conduct code review focused on resource management
- [ ] Add memory benchmarking to performance tests
- [ ] Create runbook for memory analysis
EOF
```

---

## Playbook: Network Connectivity Issues

### Detection

**Alert**: Network packet loss > 5% or latency > 100ms
**Symptoms**:
- Intermittent connection failures
- Timeouts
- High network latency

### Investigation

```bash
#!/bin/bash
# Network connectivity investigation

INCIDENT_ID=$1

echo "=== INVESTIGATING NETWORK CONNECTIVITY ==="

# 1. Check pod-to-pod connectivity
echo "Step 1: Testing pod-to-pod connectivity..."
./scripts/test-pod-connectivity.sh --namespace production

# 2. Check DNS resolution
echo "Step 2: Testing DNS resolution..."
kubectl run dns-test --image=busybox --rm -it --restart=Never -- nslookup kubernetes.default

# 3. Check network policies
echo "Step 3: Checking network policies..."
kubectl get networkpolicies --all-namespaces -o yaml

# 4. Check CNI plugin status
echo "Step 4: Checking CNI plugin..."
kubectl get pods -n kube-system -l k8s-app=calico-node -o wide

# 5. Check iptables rules
echo "Step 5: Sampling iptables rules..."
NODE=$(kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
kubectl debug node/$NODE -it --image=nicolaka/netshoot -- iptables -L -n | head -50

# 6. Test external connectivity
echo "Step 6: Testing external connectivity..."
kubectl run curl-test --image=curlimages/curl --rm -it --restart=Never -- curl -v https://www.google.com

# 7. Check for MTU issues
echo "Step 7: Checking MTU settings..."
./scripts/check-mtu.sh --namespace production

# 8. Analyze network metrics
echo "Step 8: Analyzing network metrics..."
./scripts/network-metrics.sh --since 30m
```

### Resolution Steps

```bash
#!/bin/bash
# Network connectivity resolution

INCIDENT_ID=$1

echo "=== RESOLVING NETWORK CONNECTIVITY ==="

# 1. Restart CNI plugin
echo "$(date -u +%T) - Restarting CNI plugin..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl rollout restart daemonset/calico-node -n kube-system
kubectl rollout restart deployment/calico-kube-controllers -n kube-system

# 2. Restart kube-proxy
echo "$(date -u +%T) - Restarting kube-proxy..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl rollout restart daemonset/kube-proxy -n kube-system

# 3. Verify network plugin recovery
sleep 60
kubectl get pods -n kube-system -l k8s-app=calico-node

# 4. Test connectivity again
./scripts/test-pod-connectivity.sh --namespace production

# 5. If issue persists, restart affected nodes
if [ $? -ne 0 ]; then
    echo "$(date -u +%T) - Network issue persists, restarting affected nodes..." | tee -a /incidents/${INCIDENT_ID}/incident.log
    ./scripts/rolling-node-restart.sh --namespace production
fi
```

---

## Playbook: Security Breach

### Detection

**Alert**: Unauthorized access attempt or suspicious activity detected
**Symptoms**:
- Failed authentication attempts
- Unusual API calls
- Unexpected data access

### Investigation

```bash
#!/bin/bash
# Security breach investigation

INCIDENT_ID=$1
BREACH_TYPE=$2  # unauthorized-access, data-leak, malware, etc.

echo "=== SECURITY BREACH INVESTIGATION ==="
echo "Incident: $INCIDENT_ID"
echo "Type: $BREACH_TYPE"
echo "⚠️  SECURITY INCIDENT - HANDLE WITH CARE"

# 1. Preserve evidence
echo "$(date -u +%T) - Preserving evidence..." | tee -a /incidents/${INCIDENT_ID}/incident.log
mkdir -p /incidents/${INCIDENT_ID}/forensics

# 2. Collect audit logs
echo "$(date -u +%T) - Collecting audit logs..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl logs -n security audit-logger --since=24h > /incidents/${INCIDENT_ID}/forensics/audit-logs.txt

# 3. Collect authentication logs
kubectl logs -n production deployment/auth-service --since=24h > /incidents/${INCIDENT_ID}/forensics/auth-logs.txt

# 4. Check for suspicious processes
echo "$(date -u +%T) - Checking for suspicious processes..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/scan-suspicious-processes.sh > /incidents/${INCIDENT_ID}/forensics/processes.txt

# 5. Check file integrity
echo "$(date -u +%T) - Checking file integrity..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/file-integrity-check.sh > /incidents/${INCIDENT_ID}/forensics/file-integrity.txt

# 6. Analyze network traffic
echo "$(date -u +%T) - Analyzing network traffic..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/analyze-network-traffic.sh --hours 24 > /incidents/${INCIDENT_ID}/forensics/network-traffic.txt

# 7. Check for data exfiltration
echo "$(date -u +%T) - Checking for data exfiltration..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/check-data-exfiltration.sh > /incidents/${INCIDENT_ID}/forensics/data-exfiltration.txt

# 8. Notify security team
echo "$(date -u +%T) - Notifying security team..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/notify-security-team.sh --incident $INCIDENT_ID --type $BREACH_TYPE --severity CRITICAL
```

### Resolution Steps

```bash
#!/bin/bash
# Security breach containment and resolution

INCIDENT_ID=$1

echo "=== SECURITY BREACH CONTAINMENT ==="

# 1. Isolate affected systems
echo "$(date -u +%T) - Isolating affected systems..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/isolate-compromised-systems.sh --incident $INCIDENT_ID

# 2. Revoke all active sessions
echo "$(date -u +%T) - Revoking all active sessions..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl exec -n production deployment/auth-service -- /app/revoke-all-sessions.sh

# 3. Rotate all secrets and credentials
echo "$(date -u +%T) - Rotating all secrets..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/emergency-secret-rotation.sh --all

# 4. Apply security patches
echo "$(date -u +%T) - Applying security patches..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/apply-security-patches.sh --emergency

# 5. Enable enhanced monitoring
echo "$(date -u +%T) - Enabling enhanced security monitoring..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/enable-security-monitoring.sh --level maximum

# 6. Block malicious IPs
echo "$(date -u +%T) - Blocking malicious IPs..." | tee -a /incidents/${INCIDENT_ID}/incident.log
./scripts/block-malicious-ips.sh --source /incidents/${INCIDENT_ID}/forensics/network-traffic.txt

# 7. Notify affected users (if personal data involved)
if [ "$BREACH_TYPE" == "data-leak" ]; then
    echo "$(date -u +%T) - Preparing user notification..." | tee -a /incidents/${INCIDENT_ID}/incident.log
    ./scripts/prepare-breach-notification.sh --incident $INCIDENT_ID
fi

# 8. Engage legal/compliance team
./scripts/notify-legal-team.sh --incident $INCIDENT_ID
```

### Post-Incident Actions

```bash
# Action items for security breach
cat > /incidents/${INCIDENT_ID}/action-items.md << 'EOF'
# Action Items - SECURITY BREACH

## Immediate (Complete within 24 hours)
- [ ] Complete forensic analysis
- [ ] Determine full scope of breach
- [ ] Notify affected users and stakeholders
- [ ] File required legal/compliance reports
- [ ] Document timeline and impact

## Short-term (Complete within 1 week)
- [ ] Implement additional security controls
- [ ] Review and update access policies
- [ ] Conduct security training for team
- [ ] Implement enhanced monitoring
- [ ] Perform security audit

## Long-term (Complete within 1 month)
- [ ] Conduct full security review
- [ ] Implement zero-trust architecture
- [ ] Create incident response automation
- [ ] Establish bug bounty program
- [ ] Obtain security certification (SOC 2, ISO 27001)
EOF
```

---

*[Continuing with additional playbooks for Data Corruption, Capacity Exhaustion, Certificate Expiration, DDoS Attack, Message Queue Backlog, and Cascading Failures]*

---

## Playbook: Certificate Expiration

### Detection

**Alert**: Certificate expires within 30 days
**Symptoms**:
- TLS/SSL warnings
- Certificate validation failures
- Service connectivity issues

### Investigation

```bash
#!/bin/bash
# Certificate expiration investigation

INCIDENT_ID=$1

echo "=== INVESTIGATING CERTIFICATE EXPIRATION ==="

# 1. List all certificates and expiration dates
echo "Step 1: Checking all certificates..."
./scripts/check-all-certs.sh --output /incidents/${INCIDENT_ID}/certificates.txt

# 2. Identify expired or soon-to-expire certificates
cat /incidents/${INCIDENT_ID}/certificates.txt | awk '$4 < 30 {print $0}'

# 3. Check cert-manager status
kubectl get certificates --all-namespaces -o wide

# 4. Check certificate issuers
kubectl get clusterissuers -o wide
```

### Resolution Steps

```bash
#!/bin/bash
# Certificate renewal

INCIDENT_ID=$1
CERT_NAME=$2

echo "=== RENEWING CERTIFICATE ==="

# 1. Trigger certificate renewal
echo "$(date -u +%T) - Triggering certificate renewal..." | tee -a /incidents/${INCIDENT_ID}/incident.log
kubectl delete secret $CERT_NAME -n production
kubectl annotate certificate $CERT_NAME -n production cert-manager.io/issue-temporary-certificate="true"

# 2. Wait for renewal
kubectl wait --for=condition=Ready certificate/$CERT_NAME -n production --timeout=5m

# 3. Verify new certificate
openssl s_client -connect api.dwcp.io:443 -servername api.dwcp.io </dev/null 2>/dev/null | openssl x509 -noout -dates

# 4. Restart affected services
kubectl rollout restart deployment --all -n production
```

---

## Post-Incident Review Process

### Post-Incident Review Template

```markdown
# Post-Incident Review: [INCIDENT_ID]

**Date**: [DATE]
**Facilitator**: [NAME]
**Attendees**: [NAMES]

## Incident Summary

- **Incident ID**: [ID]
- **Severity**: [SEV-X]
- **Detection Time**: [TIMESTAMP]
- **Resolution Time**: [TIMESTAMP]
- **Total Duration**: [DURATION]
- **Impact**: [DESCRIPTION]

## What Happened?

### Timeline

| Time | Event | Action Taken |
|------|-------|--------------|
| | | |

### Root Cause

[Detailed explanation of the root cause]

### Contributing Factors

1. [Factor 1]
2. [Factor 2]
3. [Factor 3]

## What Went Well?

1. [Success 1]
2. [Success 2]
3. [Success 3]

## What Went Poorly?

1. [Issue 1]
2. [Issue 2]
3. [Issue 3]

## Where Did We Get Lucky?

1. [Luck factor 1]
2. [Luck factor 2]

## Action Items

### Prevent

- [ ] [Action to prevent similar incidents] - Owner: [NAME] - Due: [DATE]

### Detect

- [ ] [Action to detect similar incidents earlier] - Owner: [NAME] - Due: [DATE]

### Mitigate

- [ ] [Action to reduce impact] - Owner: [NAME] - Due: [DATE]

### Improve

- [ ] [Process improvement] - Owner: [NAME] - Due: [DATE]

## Lessons Learned

[Key lessons from this incident]

## Follow-up

- **Next Review**: [DATE]
- **Action Item Tracking**: [LINK]
```

---

**Document Control**:
- **Version**: 1.0.0
- **Last Updated**: 2025-11-10
- **Next Review**: 2025-12-10
- **Owner**: SRE Team
- **Approver**: VP Engineering

---

*This document is classified as Internal Use - Production Critical.*
