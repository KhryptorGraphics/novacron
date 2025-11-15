# High Availability and Fault Tolerance Operations Runbook

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Failure Scenarios](#failure-scenarios)
3. [Recovery Procedures](#recovery-procedures)
4. [Monitoring and Alerts](#monitoring-and-alerts)
5. [Performance Tuning](#performance-tuning)
6. [Disaster Recovery](#disaster-recovery)

## System Architecture

### Components

#### Consensus Layer (Raft)
- **Location**: `backend/core/consensus/`
- **Quorum**: Requires (N/2)+1 nodes for consensus
- **Leader Election**: 150-300ms randomized timeout
- **Heartbeat Interval**: 50ms
- **Log Replication**: Synchronous to majority

#### Resilience Patterns
- **Circuit Breakers**: Prevent cascade failures
- **Rate Limiters**: Control request flow
- **Bulkheads**: Isolate failures
- **Retry Policies**: Automatic recovery
- **Error Budgets**: SLO tracking

#### Failure Detection
- **Phi Accrual Detector**: Adaptive failure detection
- **Default Threshold**: φ = 8.0 (P_later ~ 0.0003)
- **Suspected Threshold**: φ ≥ 8.0
- **Failed Threshold**: φ ≥ 16.0

## Failure Scenarios

### 1. Single Node Failure

#### Symptoms
- Node unreachable
- Increased latency on affected routes
- Phi detector shows increasing φ value

#### Detection
```bash
# Check node health
curl -X GET http://node-X:8080/health

# Check Phi value
curl -X GET http://monitoring:9090/metrics | grep phi_accrual

# Check Raft state
curl -X GET http://node-X:8080/raft/state
```

#### Recovery Procedure
1. **Automatic Recovery** (< 30 seconds)
   - Raft automatically handles single node failures
   - New leader elected if leader fails
   - Traffic rerouted by load balancer

2. **Manual Intervention** (if needed)
   ```bash
   # Restart failed node
   systemctl restart novacron-node@X

   # Verify node rejoined cluster
   novacron-cli cluster status

   # Check log replication caught up
   novacron-cli raft status --node=X
   ```

### 2. Network Partition (Split Brain)

#### Symptoms
- Cluster split into multiple partitions
- Inconsistent state between partitions
- Multiple leaders reported

#### Detection
```bash
# Check partition status
novacron-cli cluster check-partition

# Verify quorum in each partition
for node in $(novacron-cli node list); do
  echo "Node $node:"
  novacron-cli raft quorum-status --node=$node
done
```

#### Recovery Procedure

1. **Identify Majority Partition**
   ```bash
   # Find partition with quorum
   novacron-cli partition identify-majority
   ```

2. **Fence Minority Partition**
   ```bash
   # STONITH (Shoot The Other Node In The Head)
   for node in $(novacron-cli partition list-minority); do
     novacron-cli node fence --node=$node --force
   done
   ```

3. **Restore Network Connectivity**
   ```bash
   # Fix network issues
   # Check routing tables, firewall rules, etc.
   ```

4. **Rejoin Nodes**
   ```bash
   # Rejoin fenced nodes
   for node in $(novacron-cli partition list-minority); do
     novacron-cli node rejoin --node=$node --force-resync
   done
   ```

5. **Verify Consistency**
   ```bash
   novacron-cli cluster verify-consistency
   ```

### 3. Cascading Failures

#### Symptoms
- Multiple services failing in sequence
- Circuit breakers opening across services
- System-wide degradation

#### Detection
```bash
# Check circuit breaker states
curl -X GET http://monitoring:9090/metrics | grep circuit_breaker_state

# Check service dependencies
novacron-cli service dependency-map

# Monitor error rates
novacron-cli metrics error-rate --window=5m
```

#### Recovery Procedure

1. **Activate Degradation Mode**
   ```bash
   # Enable graceful degradation
   novacron-cli degradation set --level=emergency

   # Disable non-critical features
   novacron-cli feature disable --category=non-critical
   ```

2. **Isolate Root Cause**
   ```bash
   # Find failing service
   novacron-cli service find-root-failure

   # Check service logs
   journalctl -u novacron-service-X -n 100
   ```

3. **Reset Circuit Breakers** (after fix)
   ```bash
   # Reset specific circuit breaker
   novacron-cli circuit-breaker reset --service=X

   # Reset all circuit breakers
   novacron-cli circuit-breaker reset-all
   ```

4. **Gradual Recovery**
   ```bash
   # Slowly increase traffic
   for pct in 10 25 50 75 100; do
     novacron-cli traffic set --percent=$pct
     sleep 60
     novacron-cli metrics check-health || break
   done
   ```

### 4. Resource Exhaustion

#### Symptoms
- High memory/CPU usage
- Increased latency
- Request timeouts

#### Detection
```bash
# Check resource usage
novacron-cli resource status

# Check bulkhead saturation
novacron-cli bulkhead status

# Memory analysis
novacron-cli memory profile --output=mem.prof
go tool pprof mem.prof
```

#### Recovery Procedure

1. **Activate Resource Limits**
   ```bash
   # Enable bulkheads
   novacron-cli bulkhead enable --service=all

   # Adjust rate limits
   novacron-cli rate-limit set --rps=100 --burst=200
   ```

2. **Shed Load**
   ```bash
   # Enable load shedding
   novacron-cli load-shed enable --threshold=80

   # Prioritize critical traffic
   novacron-cli traffic prioritize --class=critical
   ```

3. **Scale Resources**
   ```bash
   # Horizontal scaling
   novacron-cli cluster scale --add-nodes=2

   # Vertical scaling (requires restart)
   novacron-cli node resize --memory=16G --cpu=8
   ```

## Recovery Procedures

### Emergency Failover

**RTO Target**: < 30 seconds
**RPO Target**: < 5 seconds

```bash
#!/bin/bash
# emergency_failover.sh

STANDBY_SITE="${1:-site-b}"

echo "Initiating emergency failover to $STANDBY_SITE"

# 1. Stop writes to primary
novacron-cli site set-readonly --site=primary

# 2. Wait for replication to catch up
novacron-cli replication wait-sync --timeout=10s

# 3. Promote standby
novacron-cli site promote --site=$STANDBY_SITE

# 4. Update DNS
novacron-cli dns update --primary=$STANDBY_SITE

# 5. Verify health
novacron-cli site verify --site=$STANDBY_SITE

echo "Failover complete. New primary: $STANDBY_SITE"
```

### Disaster Recovery from Backup

```bash
#!/bin/bash
# disaster_recovery.sh

BACKUP_ID="${1}"
TARGET_SITE="${2:-recovery}"

echo "Starting disaster recovery from backup $BACKUP_ID"

# 1. Verify backup integrity
novacron-cli backup verify --id=$BACKUP_ID

# 2. Prepare recovery environment
novacron-cli site prepare --site=$TARGET_SITE

# 3. Restore data
novacron-cli backup restore --id=$BACKUP_ID --target=$TARGET_SITE

# 4. Apply transaction logs
novacron-cli wal replay --since=$BACKUP_ID --target=$TARGET_SITE

# 5. Verify consistency
novacron-cli data verify --site=$TARGET_SITE

# 6. Activate site
novacron-cli site activate --site=$TARGET_SITE

echo "Recovery complete. Site $TARGET_SITE is active"
```

## Monitoring and Alerts

### Key Metrics

| Metric | Warning Threshold | Critical Threshold | Action |
|--------|------------------|-------------------|---------|
| Node Phi Value | φ > 6 | φ > 8 | Check node health |
| Circuit Breaker State | Half-Open | Open | Investigate failures |
| Error Rate | > 1% | > 5% | Check error budget |
| P95 Latency | > 200ms | > 500ms | Scale/optimize |
| Replication Lag | > RPO/2 | > RPO | Check network/load |
| Consensus Latency | > 100ms | > 500ms | Check Raft health |

### Alert Response

#### Critical Alerts (Page immediately)
- Cluster lost quorum
- Primary site down
- Data corruption detected
- Security breach

#### Warning Alerts (Business hours)
- Single node failure
- High error rate
- Approaching error budget
- High replication lag

### Health Check Endpoints

```bash
# Overall health
curl http://localhost:8080/health

# Detailed health
curl http://localhost:8080/health/detailed

# Raft consensus health
curl http://localhost:8080/raft/health

# Replication status
curl http://localhost:8080/replication/status

# Circuit breaker states
curl http://localhost:8080/resilience/circuit-breakers
```

## Performance Tuning

### Raft Consensus Tuning

```yaml
# raft_config.yaml
consensus:
  election_timeout_min: 150ms
  election_timeout_max: 300ms
  heartbeat_interval: 50ms
  snapshot_threshold: 10000  # Entries before snapshot
  max_append_entries: 100     # Max entries per RPC
```

### Resilience Pattern Tuning

```yaml
# resilience_config.yaml
circuit_breaker:
  failure_threshold: 5
  success_threshold: 3
  timeout: 30s
  half_open_max_requests: 3

rate_limiter:
  requests_per_second: 1000
  burst: 2000

bulkhead:
  max_concurrent: 100
  max_queue: 500
  timeout: 500ms

retry:
  max_attempts: 3
  initial_delay: 100ms
  max_delay: 1s
  multiplier: 2.0
  jitter: true
```

### Phi Accrual Detector Tuning

```yaml
# phi_detector_config.yaml
failure_detector:
  threshold: 8.0              # Base threshold
  adaptive: true              # Enable adaptive thresholds
  window_size: 200            # Samples for statistics
  network_jitter_factor: 0.3  # Expected jitter (30%)
```

## Disaster Recovery

### Backup Strategy

#### Full Backups
- **Schedule**: Daily at 02:00 UTC
- **Retention**: 7 days
- **Storage**: Geo-redundant S3

#### Incremental Backups
- **Schedule**: Every 6 hours
- **Retention**: 48 hours
- **Storage**: Regional S3

#### Point-in-Time Recovery
- **WAL Archival**: Continuous
- **Retention**: 7 days
- **RPO**: < 5 seconds

### Recovery Testing

#### Monthly DR Drill
```bash
#!/bin/bash
# monthly_dr_drill.sh

echo "Starting monthly DR drill"

# 1. Create test environment
novacron-cli test-env create --name=dr-test

# 2. Restore latest backup
LATEST_BACKUP=$(novacron-cli backup list --limit=1 --type=full)
novacron-cli backup restore --id=$LATEST_BACKUP --target=dr-test

# 3. Run validation tests
novacron-cli test run --suite=dr-validation --env=dr-test

# 4. Measure recovery metrics
RTO=$(novacron-cli metrics get --name=recovery_time)
RPO=$(novacron-cli metrics get --name=data_loss)

echo "DR Drill Complete"
echo "RTO: $RTO (Target: 30s)"
echo "RPO: $RPO (Target: 5s)"

# 5. Clean up
novacron-cli test-env destroy --name=dr-test
```

### Rollback Procedures

```bash
#!/bin/bash
# rollback.sh

VERSION="${1}"

echo "Rolling back to version $VERSION"

# 1. Create backup of current state
novacron-cli backup create --type=full --tag=pre-rollback

# 2. Stop traffic
novacron-cli traffic stop

# 3. Rollback application
novacron-cli deploy rollback --version=$VERSION

# 4. Verify health
novacron-cli health verify --strict

# 5. Resume traffic gradually
novacron-cli traffic resume --gradual

echo "Rollback complete to version $VERSION"
```

## Automation Scripts

### Auto-Healing

```bash
#!/bin/bash
# auto_heal.sh

while true; do
  # Check cluster health
  HEALTH=$(novacron-cli cluster health-score)

  if [ "$HEALTH" -lt 80 ]; then
    echo "Cluster health degraded: $HEALTH"

    # Find unhealthy nodes
    UNHEALTHY=$(novacron-cli node list --unhealthy)

    for node in $UNHEALTHY; do
      echo "Attempting to heal node $node"

      # Try restart
      novacron-cli node restart --node=$node

      sleep 10

      # Check if recovered
      if ! novacron-cli node is-healthy --node=$node; then
        # Replace node
        echo "Node $node failed to recover, replacing"
        novacron-cli node replace --node=$node
      fi
    done
  fi

  sleep 30
done
```

### Capacity Planning

```bash
#!/bin/bash
# capacity_check.sh

# Get current metrics
CPU_USAGE=$(novacron-cli metrics get --name=cpu_usage --aggregate=p95)
MEM_USAGE=$(novacron-cli metrics get --name=memory_usage --aggregate=p95)
DISK_USAGE=$(novacron-cli metrics get --name=disk_usage --aggregate=max)

# Check thresholds
if [ "$CPU_USAGE" -gt 70 ]; then
  echo "WARNING: High CPU usage: ${CPU_USAGE}%"
  echo "Consider scaling horizontally"
fi

if [ "$MEM_USAGE" -gt 80 ]; then
  echo "WARNING: High memory usage: ${MEM_USAGE}%"
  echo "Consider increasing memory allocation"
fi

if [ "$DISK_USAGE" -gt 75 ]; then
  echo "WARNING: High disk usage: ${DISK_USAGE}%"
  echo "Consider expanding storage or archiving old data"
fi

# Predict future capacity needs
GROWTH_RATE=$(novacron-cli metrics growth-rate --window=30d)
DAYS_TO_CAPACITY=$(novacron-cli capacity predict --growth=$GROWTH_RATE)

echo "At current growth rate (${GROWTH_RATE}%/day):"
echo "Days until capacity limit: $DAYS_TO_CAPACITY"
```

## Contact Information

### Escalation Path
1. **L1 Support**: Check runbook, restart services
2. **L2 Support**: Investigate logs, perform failover
3. **L3 Support**: Root cause analysis, code fixes
4. **Engineering**: Architecture changes, disaster recovery

### On-Call Rotation
- **Primary**: Check PagerDuty
- **Secondary**: Check PagerDuty
- **Manager**: Check escalation policy

### Communication Channels
- **Incidents**: #incidents (Slack)
- **Discussions**: #platform-reliability (Slack)
- **War Room**: Zoom link in incident template

## Appendix

### Common Commands Reference

```bash
# Cluster Management
novacron-cli cluster status
novacron-cli cluster health
novacron-cli cluster verify-consistency

# Node Management
novacron-cli node list
novacron-cli node restart --node=X
novacron-cli node drain --node=X
novacron-cli node replace --node=X

# Raft Consensus
novacron-cli raft status
novacron-cli raft leader
novacron-cli raft snapshot create
novacron-cli raft log compact

# Resilience Controls
novacron-cli circuit-breaker status
novacron-cli rate-limit status
novacron-cli bulkhead status
novacron-cli error-budget status

# Monitoring
novacron-cli metrics list
novacron-cli alerts list --active
novacron-cli logs tail --service=X

# Disaster Recovery
novacron-cli backup list
novacron-cli backup create --type=full
novacron-cli backup restore --id=X
novacron-cli site failover --to=X
```

### Useful One-Liners

```bash
# Find nodes with high phi values
novacron-cli node list --format=json | jq '.[] | select(.phi > 6)'

# Check all circuit breaker states
for svc in $(novacron-cli service list); do
  echo "$svc: $(novacron-cli circuit-breaker state --service=$svc)"
done

# Monitor replication lag
watch -n 5 'novacron-cli replication lag --format=table'

# Quick health dashboard
watch -n 2 'novacron-cli cluster health --format=summary'
```

---

**Last Updated**: November 2024
**Version**: 1.0.0
**Review Cycle**: Monthly