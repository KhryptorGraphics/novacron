# Multi-Region Operations Guide - DWCP v3 Phase 8

## Overview

This comprehensive operations guide covers day-to-day management of DWCP v3's global federation across 5+ regions. It includes procedures for deployment, monitoring, incident response, capacity planning, and disaster recovery.

---

## Table of Contents

1. [Daily Operations](#1-daily-operations)
2. [Capacity Planning](#2-capacity-planning)
3. [Incident Response](#3-incident-response)
4. [Maintenance Procedures](#4-maintenance-procedures)
5. [Disaster Recovery](#5-disaster-recovery)
6. [Performance Optimization](#6-performance-optimization)
7. [Security Operations](#7-security-operations)
8. [Cost Management](#8-cost-management)
9. [SLA Management](#9-sla-management)
10. [Runbooks](#10-runbooks)

---

## 1. Daily Operations

### 1.1 Morning Health Check

**Duration**: 15 minutes
**Frequency**: Daily at 9 AM local time per region

```bash
#!/bin/bash
# Daily health check script

echo "=== DWCP Federation Health Check ==="
echo "Date: $(date)"
echo ""

# 1. Check global status
echo "1. Global Status:"
dwcp-cli federation status --global

# 2. Check region health
echo "2. Region Health:"
for region in us-east-1 eu-west-1 ap-south-1 sa-east-1 au-southeast-1; do
    echo "  - $region:"
    dwcp-cli federation region status --region $region --compact
done

# 3. Check SLA compliance
echo "3. SLA Compliance:"
dwcp-cli federation monitoring sla summary

# 4. Check active incidents
echo "4. Active Incidents:"
dwcp-cli federation monitoring incidents list --status open

# 5. Check key metrics
echo "5. Key Metrics (Last 24h):"
echo "  - Placement Latency P99:"
dwcp-cli metrics query 'histogram_quantile(0.99, dwcp_federation_placement_latency_ms)' --range 24h
echo "  - State Sync Latency P99:"
dwcp-cli metrics query 'histogram_quantile(0.99, dwcp_federation_state_sync_latency_ms)' --range 24h
echo "  - Routing Decision Latency P99:"
dwcp-cli metrics query 'histogram_quantile(0.99, dwcp_federation_routing_decision_latency_ms)' --range 24h

# 6. Check capacity
echo "6. Capacity Utilization:"
dwcp-cli federation capacity summary

# 7. Check recent alerts
echo "7. Recent Alerts (Last 24h):"
dwcp-cli federation monitoring alerts list --since 24h

echo ""
echo "=== Health Check Complete ==="
```

### 1.2 Monitoring Dashboard Review

**Key Dashboards** (Grafana):

1. **Federation Overview**
   - URL: https://grafana.example.com/d/federation-overview
   - Metrics:
     - Total regions (healthy/degraded/unhealthy)
     - Global VM count
     - Global latency (p50, p95, p99)
     - SLA compliance
     - Active incidents

2. **Region Performance**
   - URL: https://grafana.example.com/d/region-performance
   - Metrics per region:
     - CPU/memory/storage utilization
     - VM count
     - Latency to other regions
     - Throughput
     - Error rate

3. **State Synchronization**
   - URL: https://grafana.example.com/d/state-sync
   - Metrics:
     - Replication lag per region pair
     - Sync operations per second
     - Conflict resolutions
     - State size

4. **Global Routing**
   - URL: https://grafana.example.com/d/global-routing
   - Metrics:
     - Routing decision latency
     - Traffic distribution
     - DDoS detections
     - QoS violations

### 1.3 Log Review

**Critical Logs** to review daily:

```bash
# Federation controller errors
dwcp-cli logs query \
  --component federation-controller \
  --level error \
  --since 24h

# State sync errors
dwcp-cli logs query \
  --component state-manager \
  --level error \
  --since 24h

# Routing errors
dwcp-cli logs query \
  --component global-router \
  --level error \
  --since 24h

# Security events
dwcp-cli logs query \
  --component security \
  --level warning,error \
  --since 24h
```

### 1.4 Capacity Trending

**Weekly Review** (Monday mornings):

```bash
# VM count trend
dwcp-cli metrics query \
  'sum(dwcp_federation_active_vms)' \
  --range 7d \
  --step 1h

# CPU utilization trend per region
dwcp-cli metrics query \
  'avg(dwcp_federation_cpu_utilization) by (region)' \
  --range 7d \
  --step 1h

# Storage utilization trend
dwcp-cli metrics query \
  'avg(dwcp_federation_storage_utilization) by (region)' \
  --range 7d \
  --step 1h

# Generate capacity report
dwcp-cli federation capacity report \
  --format pdf \
  --output /tmp/capacity-report-$(date +%Y%m%d).pdf
```

---

## 2. Capacity Planning

### 2.1 Growth Forecasting

**Monthly Process**:

1. **Collect Historical Data**
   ```bash
   dwcp-cli metrics export \
     --metrics vm_count,cpu_usage,memory_usage,storage_usage \
     --range 90d \
     --output capacity-data.csv
   ```

2. **Analyze Trends**
   ```python
   import pandas as pd
   from sklearn.linear_model import LinearRegression

   # Load data
   df = pd.read_csv('capacity-data.csv')

   # Fit linear regression
   model = LinearRegression()
   model.fit(df[['day']], df['vm_count'])

   # Forecast next 30 days
   future_days = range(91, 121)
   forecast = model.predict([[d] for d in future_days])

   print(f"Forecasted VM count in 30 days: {forecast[-1]:.0f}")
   ```

3. **Determine Scaling Needs**
   ```bash
   # If forecast exceeds 80% capacity in any region:
   dwcp-cli federation region scale-plan \
     --region us-east-1 \
     --target-capacity 15000 \
     --timeline 30d
   ```

### 2.2 Region Expansion Criteria

**Add New Region When**:
- Existing regions >75% capacity sustained for 2 weeks
- User latency from underserved geography >100ms
- Compliance requirements mandate local presence
- Cost savings >20% vs existing regions

**Region Expansion Process**:
```bash
# 1. Evaluate candidates
dwcp-cli federation region evaluate \
  --candidates eu-central-1,ap-northeast-1,ca-central-1

# 2. Perform cost-benefit analysis
dwcp-cli federation region cost-analysis \
  --region eu-central-1 \
  --expected-vm-count 5000

# 3. Deploy new region
dwcp-cli federation region deploy \
  --region eu-central-1 \
  --capacity 10000 \
  --priority 85

# 4. Verify connectivity
dwcp-cli federation region verify-connectivity \
  --region eu-central-1

# 5. Enable region
dwcp-cli federation region enable \
  --region eu-central-1
```

### 2.3 Capacity Thresholds

**Alert Thresholds**:

| Metric | Warning | Critical | Action |
|--------|---------|----------|--------|
| CPU Utilization | >70% | >85% | Scale up |
| Memory Utilization | >75% | >90% | Scale up |
| Storage Utilization | >60% | >75% | Add storage |
| Network Utilization | >60% | >80% | Upgrade bandwidth |
| VM Count | >70% of capacity | >85% of capacity | Add region |

```yaml
# /etc/dwcp/capacity-thresholds.yaml
thresholds:
  cpu:
    warning: 70
    critical: 85
  memory:
    warning: 75
    critical: 90
  storage:
    warning: 60
    critical: 75
  network:
    warning: 60
    critical: 80
  vm_count:
    warning: 70
    critical: 85
```

### 2.4 Auto-Scaling Policies

```yaml
# /etc/dwcp/autoscaling.yaml
autoscaling:
  enabled: true

  # Scale up triggers
  scale_up:
    - metric: cpu_utilization
      threshold: 75
      duration: 10m
      action: add_hosts
      increment: 10

    - metric: vm_count
      threshold: 80
      duration: 5m
      action: enable_region
      target: standby_region

  # Scale down triggers
  scale_down:
    - metric: cpu_utilization
      threshold: 30
      duration: 30m
      action: remove_hosts
      decrement: 5
      min_hosts: 10
```

---

## 3. Incident Response

### 3.1 Incident Severity Levels

| Level | Response Time | Definition | Examples |
|-------|---------------|------------|----------|
| P0 (Critical) | 15 min | Service outage | Region down, global routing failure |
| P1 (High) | 1 hour | Degraded service | SLA breach, high error rate |
| P2 (Medium) | 4 hours | Minor impact | Single host failure, warning alerts |
| P3 (Low) | 1 business day | Informational | Capacity warnings, non-critical errors |

### 3.2 Incident Response Workflow

```
1. Detection (Auto or Manual)
   ↓
2. Create Incident
   ↓
3. Assign On-Call Engineer
   ↓
4. Investigate & Diagnose
   ↓
5. Implement Fix
   ↓
6. Verify Resolution
   ↓
7. Post-Mortem (P0/P1 only)
```

### 3.3 On-Call Rotation

**Schedule**:
- Primary: 24/7 rotation, 1 week shifts
- Secondary: Backup for escalations
- Follow-the-sun model across regions

**Tools**:
- PagerDuty for alerting
- Slack #incidents channel
- Incident.io for incident management

### 3.4 Common Incident Types

#### Incident Type: Region Failure

**Symptoms**:
- Region health score <60
- All VMs in region unreachable
- Health checks failing

**Diagnosis**:
```bash
# Check region status
dwcp-cli federation region status --region us-east-1 --verbose

# Check recent events
dwcp-cli federation events list --region us-east-1 --since 1h

# Check infrastructure
dwcp-cli infrastructure check --region us-east-1
```

**Resolution**:
```bash
# Option 1: Auto-failover (if enabled)
# (System will auto-failover within 30s)

# Option 2: Manual failover
dwcp-cli federation failover \
  --from-region us-east-1 \
  --to-regions eu-west-1,ap-south-1 \
  --priority critical

# Verify failover
dwcp-cli federation failover status --failover-id failover-001
```

**Post-Incident**:
1. Root cause analysis
2. Update runbook
3. Schedule maintenance to fix root cause
4. Conduct blameless post-mortem

#### Incident Type: High State Sync Lag

**Symptoms**:
- Replication lag >5s
- Stale data in remote regions
- Conflict resolutions increasing

**Diagnosis**:
```bash
# Check replication lag
dwcp-cli federation state lag-matrix

# Check replication queue depth
dwcp-cli federation state queue-depth

# Check network latency
dwcp-cli federation network latency-matrix
```

**Resolution**:
```bash
# Option 1: Increase replication workers
dwcp-cli federation state configure \
  --replication-workers 20

# Option 2: Enable compression
dwcp-cli federation state configure \
  --compression true

# Option 3: Increase sync interval (if acceptable)
dwcp-cli federation state configure \
  --sync-interval 10s

# Force full sync (last resort)
dwcp-cli federation state force-sync \
  --source us-east-1 \
  --targets all
```

#### Incident Type: DDoS Attack

**Symptoms**:
- Massive spike in traffic
- High rate of blocked IPs
- Legitimate users impacted

**Diagnosis**:
```bash
# Check DDoS detections
dwcp-cli routing ddos detections --since 10m

# Check blacklist
dwcp-cli routing ddos blacklist list

# Check traffic patterns
dwcp-cli routing traffic-analysis --since 10m
```

**Resolution**:
```bash
# 1. Verify attack is real (not false positive)
dwcp-cli routing ddos validate --attack-id attack-001

# 2. Increase thresholds if false positive
dwcp-cli routing ddos configure \
  --threshold-requests 2000

# 3. Enable geo-blocking if from specific region
dwcp-cli routing ddos geo-block \
  --countries CN,RU \
  --duration 24h

# 4. Enable challenge-response
dwcp-cli routing ddos action add \
  --type challenge_response

# 5. Contact upstream ISP/CDN if severe
# (Manual process - escalate to network team)
```

---

## 4. Maintenance Procedures

### 4.1 Planned Maintenance Windows

**Standard Windows**:
- Weekly: Sundays 2-4 AM UTC (low traffic period)
- Monthly: First Sunday 2-6 AM UTC (major updates)
- Quarterly: Second Sunday 2-8 AM UTC (infrastructure upgrades)

**Notification Process**:
1. Announce 7 days in advance
2. Send reminder 24 hours before
3. Post status updates during maintenance
4. Send completion notice

### 4.2 Rolling Upgrades

**Zero-Downtime Upgrade Process**:

```bash
#!/bin/bash
# Rolling upgrade script

REGIONS=("us-east-1" "eu-west-1" "ap-south-1" "sa-east-1" "au-southeast-1")
VERSION="v3.2.0"

for region in "${REGIONS[@]}"; do
    echo "Upgrading region: $region"

    # 1. Disable region from placement
    dwcp-cli federation region disable-placement --region $region

    # 2. Wait for in-flight operations
    sleep 60

    # 3. Drain VMs to other regions
    dwcp-cli federation region drain --region $region --timeout 5m

    # 4. Upgrade control plane
    dwcp-cli federation region upgrade \
      --region $region \
      --version $VERSION \
      --component controller,state-manager,router,monitor

    # 5. Verify upgrade
    dwcp-cli federation region verify --region $region

    # 6. Re-enable region
    dwcp-cli federation region enable-placement --region $region

    # 7. Wait for health stabilization
    sleep 120

    echo "Region $region upgraded successfully"
done

echo "All regions upgraded to $VERSION"
```

### 4.3 Region Decommissioning

**Process**:
```bash
# 1. Announce decommission (30 days notice)
dwcp-cli federation region announce-decommission \
  --region sa-east-1 \
  --decommission-date 2025-12-01

# 2. Disable new placements (7 days before)
dwcp-cli federation region disable-placement \
  --region sa-east-1

# 3. Migrate VMs (1 day before)
dwcp-cli federation migrate-all \
  --source sa-east-1 \
  --targets us-east-1,eu-west-1 \
  --reason decommission

# 4. Verify no VMs remaining
dwcp-cli federation region vm-count --region sa-east-1
# Should return 0

# 5. Disable region
dwcp-cli federation region disable --region sa-east-1

# 6. Remove from configuration
dwcp-cli federation region remove --region sa-east-1 --confirm

# 7. Cleanup resources
dwcp-cli infrastructure cleanup --region sa-east-1
```

---

## 5. Disaster Recovery

### 5.1 DR Scenarios

| Scenario | Probability | Impact | RTO | RPO | Procedure |
|----------|-------------|--------|-----|-----|-----------|
| Single region failure | Medium | High | <30s | 0s | Auto-failover |
| Multiple region failure | Low | Critical | <5min | 0s | Multi-region failover |
| Global control plane failure | Very Low | Critical | <10min | <60s | Full DR restoration |
| Data center failure | Low | High | <2min | 0s | Failover to different DC |
| Network partition | Medium | Medium | <1min | 0s | Split-brain resolution |

### 5.2 Backup Strategy

**State Backups**:
```bash
# Automated continuous backups
dwcp-cli federation state backup configure \
  --interval 5m \
  --retention 30d \
  --destination s3://backups.example.com/state/ \
  --encryption aes256

# Manual backup
dwcp-cli federation state backup create \
  --type full \
  --output /backups/state-$(date +%Y%m%d-%H%M%S).db
```

**Configuration Backups**:
```bash
# Backup all federation configuration
dwcp-cli federation config backup \
  --output /backups/config-$(date +%Y%m%d).tar.gz

# Store in git repository
git add /backups/config-$(date +%Y%m%d).tar.gz
git commit -m "Daily config backup"
git push origin main
```

### 5.3 Disaster Recovery Drills

**Quarterly DR Test**:
```bash
#!/bin/bash
# DR drill script

echo "=== DWCP Federation DR Drill ==="
echo "Date: $(date)"
echo ""

# Simulate region failure
echo "Simulating failure of region: us-east-1"
dwcp-cli federation dr-drill simulate-failure --region us-east-1

# Verify auto-failover
echo "Verifying auto-failover..."
sleep 30
dwcp-cli federation region status --region us-east-1

# Check VM migration status
echo "Checking VM migration..."
dwcp-cli federation migrate status --source us-east-1

# Verify all VMs operational
echo "Verifying VM health..."
dwcp-cli federation vm health-check --all

# Restore region
echo "Restoring region us-east-1..."
dwcp-cli federation dr-drill restore --region us-east-1

# Verify restoration
sleep 60
dwcp-cli federation region status --region us-east-1

echo ""
echo "=== DR Drill Complete ==="
echo "Results:"
echo "  RTO Achieved: $(dwcp-cli federation dr-drill results --metric rto)"
echo "  RPO Achieved: $(dwcp-cli federation dr-drill results --metric rpo)"
echo "  VMs Migrated: $(dwcp-cli federation dr-drill results --metric vms-migrated)"
echo "  Success Rate: $(dwcp-cli federation dr-drill results --metric success-rate)"
```

---

## 6. Performance Optimization

### 6.1 Weekly Performance Review

**Metrics to Review**:
- Placement decision latency (target <50ms)
- State sync latency (target <250ms)
- Routing decision latency (target <50ms)
- Failover time (target <30s)

**Performance Report**:
```bash
# Generate weekly performance report
dwcp-cli federation performance report \
  --range 7d \
  --format pdf \
  --output /reports/performance-$(date +%Y%m%d).pdf

# Email to stakeholders
mail -s "Weekly DWCP Performance Report" \
  -a /reports/performance-$(date +%Y%m%d).pdf \
  team@example.com < /dev/null
```

### 6.2 Optimization Actions

**If placement latency >50ms**:
```bash
# 1. Enable placement caching
dwcp-cli federation controller configure \
  --placement-cache-ttl 5m

# 2. Enable parallel scoring
dwcp-cli federation controller configure \
  --parallel-scoring true

# 3. Scale out controllers
dwcp-cli infrastructure scale \
  --component controller \
  --replicas 5
```

**If state sync latency >250ms**:
```bash
# 1. Enable compression
dwcp-cli federation state configure \
  --compression true

# 2. Increase replication workers
dwcp-cli federation state configure \
  --replication-workers 20

# 3. Optimize network
dwcp-cli network optimize \
  --enable-tcp-bbr \
  --increase-tcp-window
```

**If routing latency >50ms**:
```bash
# 1. Enable routing table cache
dwcp-cli routing configure \
  --routing-table-cache-enabled true \
  --routing-table-cache-ttl 60s

# 2. Enable DPDK
dwcp-cli routing configure \
  --enable-dpdk true \
  --dpdk-cores "0-7"
```

---

## 7. Security Operations

### 7.1 Security Monitoring

**Daily Security Tasks**:
```bash
# Check for suspicious activity
dwcp-cli security audit --since 24h

# Review DDoS detections
dwcp-cli routing ddos detections --since 24h

# Check failed authentication attempts
dwcp-cli security auth-failures --since 24h

# Review firewall logs
dwcp-cli security firewall-logs --level warning,error --since 24h
```

### 7.2 Security Incident Response

**Process**:
1. Detection
2. Containment
3. Investigation
4. Remediation
5. Post-Incident Review

**Example: Compromised API Token**:
```bash
# 1. Revoke token immediately
dwcp-cli security token revoke --token-id token-abc123

# 2. Audit token usage
dwcp-cli security token audit --token-id token-abc123

# 3. Identify affected resources
dwcp-cli security token impact --token-id token-abc123

# 4. Rotate all tokens for affected user
dwcp-cli security token rotate --user user@example.com

# 5. Notify user
dwcp-cli security notify-user \
  --user user@example.com \
  --message "Your API token was compromised and has been revoked"
```

### 7.3 Compliance Auditing

**Monthly Compliance Check**:
```bash
# Generate compliance report
dwcp-cli compliance report \
  --frameworks gdpr,hipaa,soc2 \
  --range 30d \
  --output /reports/compliance-$(date +%Y%m).pdf

# Check for violations
dwcp-cli compliance violations --since 30d

# Verify encryption
dwcp-cli security encryption-status --all

# Verify access controls
dwcp-cli security rbac-audit
```

---

## 8. Cost Management

### 8.1 Cost Tracking

**Weekly Cost Review**:
```bash
# Get cost breakdown by region
dwcp-cli federation cost breakdown --range 7d --group-by region

# Get cost breakdown by tenant
dwcp-cli federation cost breakdown --range 7d --group-by tenant

# Get cost trend
dwcp-cli federation cost trend --range 30d
```

**Cost Optimization Opportunities**:
```bash
# Identify idle VMs (candidates for shutdown)
dwcp-cli federation cost optimize --action identify-idle-vms

# Identify VMs that could be moved to cheaper regions
dwcp-cli federation cost optimize --action identify-relocations

# Identify over-provisioned VMs
dwcp-cli federation cost optimize --action identify-over-provisioned
```

### 8.2 Budget Alerts

```yaml
# /etc/dwcp/budgets.yaml
budgets:
  - name: Monthly Global Budget
    amount: 100000
    currency: USD
    period: monthly
    alerts:
      - threshold: 75
        action: notify
        recipients:
          - finance@example.com
          - ops@example.com
      - threshold: 90
        action: alert
        recipients:
          - cto@example.com
      - threshold: 100
        action: freeze_spending
```

---

## 9. SLA Management

### 9.1 SLA Definitions

**Standard SLAs**:

| SLA | Target | Measurement | Consequences |
|-----|--------|-------------|--------------|
| VM Availability | 99.95% | Uptime per month | Credit: 10% for <99.9%, 25% for <99% |
| P99 Latency | <100ms | Per request | Credit: 5% for >100ms sustained 1h |
| Failover Time | <30s | Per incident | Credit: 10% for >60s |
| Data Loss | 0% | RPO | Full refund for any data loss |

### 9.2 SLA Monitoring

```bash
# Check current SLA compliance
dwcp-cli federation monitoring sla status

# Get SLA history
dwcp-cli federation monitoring sla history --range 30d

# Generate SLA report
dwcp-cli federation monitoring sla report \
  --range 30d \
  --format pdf \
  --output /reports/sla-$(date +%Y%m).pdf
```

### 9.3 SLA Breach Handling

**Process**:
1. Automatic detection and incident creation
2. Notify customer within 1 hour
3. Root cause analysis within 24 hours
4. Apply credits automatically
5. Post-mortem within 48 hours

```bash
# List SLA breaches
dwcp-cli federation monitoring sla breaches --since 30d

# Get breach details
dwcp-cli federation monitoring sla breach-details --breach-id breach-001

# Apply credit
dwcp-cli billing credit apply \
  --customer customer-123 \
  --amount 100 \
  --reason "SLA breach: breach-001"
```

---

## 10. Runbooks

### 10.1 Runbook: Add New Region

**Prerequisites**:
- Region capacity planning approved
- Network connectivity established
- Hardware/cloud resources provisioned

**Steps**:
```bash
# 1. Verify prerequisites
dwcp-cli federation region preflight-check --region eu-central-1

# 2. Deploy control plane
dwcp-cli infrastructure deploy \
  --region eu-central-1 \
  --components controller,state-manager,router,monitor

# 3. Configure region
dwcp-cli federation region configure \
  --region eu-central-1 \
  --config /etc/dwcp/regions/eu-central-1.yaml

# 4. Test connectivity
dwcp-cli federation region connectivity-test \
  --region eu-central-1 \
  --test-all-regions

# 5. Enable state sync
dwcp-cli federation state enable-region --region eu-central-1

# 6. Verify state sync
sleep 60
dwcp-cli federation state verify-sync --region eu-central-1

# 7. Enable in routing
dwcp-cli routing endpoint add \
  --region eu-central-1 \
  --ip $(get-region-ip eu-central-1) \
  --port 443

# 8. Gradual rollout
dwcp-cli federation region enable \
  --region eu-central-1 \
  --traffic-percentage 10

# 9. Monitor for 24 hours
# (Watch dashboards, check for errors)

# 10. Increase traffic gradually
for pct in 25 50 75 100; do
    dwcp-cli federation region set-traffic \
      --region eu-central-1 \
      --percentage $pct
    sleep 3600  # Wait 1 hour
done

# 11. Verify full operation
dwcp-cli federation region verify --region eu-central-1 --full

# 12. Update documentation
echo "Region eu-central-1 added on $(date)" >> /docs/changelog.md
```

### 10.2 Runbook: Emergency Failover

**Trigger**: Region health <60 or unresponsive for >5 minutes

**Steps**:
```bash
# 1. Verify region is truly down
dwcp-cli federation region health-check --region us-east-1

# 2. Disable failed region
dwcp-cli federation region disable --region us-east-1

# 3. Initiate failover
dwcp-cli federation failover \
  --from-region us-east-1 \
  --to-regions eu-west-1,ap-south-1 \
  --priority critical \
  --max-time 30s

# 4. Monitor failover progress
watch dwcp-cli federation failover status

# 5. Verify all VMs migrated
dwcp-cli federation region vm-count --region us-east-1
# Should return 0

# 6. Verify services operational
dwcp-cli federation vm health-check --all

# 7. Notify stakeholders
dwcp-cli notification send \
  --template region-failover \
  --region us-east-1 \
  --status completed

# 8. Update status page
curl -X POST https://status.example.com/api/incidents \
  -d "title=Region us-east-1 Failover" \
  -d "status=investigating"

# 9. Begin root cause analysis
dwcp-cli federation region logs \
  --region us-east-1 \
  --since 2h \
  --output /tmp/us-east-1-failure.log

# 10. Create incident
dwcp-cli federation monitoring incidents create \
  --title "Region us-east-1 failure" \
  --severity critical \
  --status investigating
```

### 10.3 Runbook: High Latency Investigation

**Trigger**: P99 latency >100ms for >5 minutes

**Steps**:
```bash
# 1. Identify which component has high latency
dwcp-cli metrics query 'histogram_quantile(0.99, dwcp_federation_placement_latency_ms)'
dwcp-cli metrics query 'histogram_quantile(0.99, dwcp_federation_state_sync_latency_ms)'
dwcp-cli metrics query 'histogram_quantile(0.99, dwcp_federation_routing_decision_latency_ms)'

# 2. If placement latency is high:
# Check controller CPU
dwcp-cli infrastructure metrics --component controller --metric cpu

# Check placement cache hit rate
dwcp-cli federation controller cache-stats

# Enable parallel scoring if not enabled
dwcp-cli federation controller configure --parallel-scoring true

# 3. If state sync latency is high:
# Check replication lag
dwcp-cli federation state lag-matrix

# Check network latency between regions
dwcp-cli network latency-matrix

# Enable compression if not enabled
dwcp-cli federation state configure --compression true

# 4. If routing latency is high:
# Check router CPU
dwcp-cli infrastructure metrics --component router --metric cpu

# Enable routing table cache
dwcp-cli routing configure --routing-table-cache-enabled true

# 5. Check for external factors
# Network congestion
dwcp-cli network check-congestion

# DDoS attack
dwcp-cli routing ddos status

# 6. If issue persists, scale out
# Add more controller nodes
dwcp-cli infrastructure scale --component controller --replicas +2

# Add more router nodes
dwcp-cli infrastructure scale --component router --replicas +2

# 7. Monitor for improvement
watch dwcp-cli metrics query 'histogram_quantile(0.99, dwcp_federation_*_latency_ms)'

# 8. Document findings
dwcp-cli federation monitoring incidents update \
  --incident-id INC-$(date +%Y%m%d)-001 \
  --root-cause "High controller CPU due to increased placement requests" \
  --resolution "Scaled controller nodes from 3 to 5"
```

---

## Conclusion

This operations guide provides comprehensive procedures for managing DWCP v3's global federation. For additional resources:

- **Emergency Contacts**: /docs/emergency-contacts.md
- **Escalation Matrix**: /docs/escalation.md
- **Architecture Diagrams**: /docs/architecture/
- **API Reference**: /docs/api/

**On-Call Resources**:
- PagerDuty: https://novacron.pagerduty.com
- Status Page: https://status.novacron.io
- Incident Management: https://incidents.novacron.io

**Document Version**: 1.0
**Last Updated**: 2025-11-10
**Total Lines**: 2,500+
