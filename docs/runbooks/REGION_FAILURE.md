# DR Runbook: Region Failure

## Overview

**Scenario**: Complete failure of a primary or secondary region
**RTO**: 30 minutes
**RPO**: 5 minutes
**Automation**: Fully automated (optional manual approval)

## Automatic Execution

This runbook executes automatically when:
- Region health score < 0.3 for 2+ minutes
- Region error rate > 50% for 1+ minute
- Complete loss of region connectivity
- Critical infrastructure failure

## Manual Execution

To manually trigger this runbook:

```bash
# Using CLI
novacron dr failover --region us-east-1 --reason "Planned maintenance"

# Using API
curl -X POST https://api.novacron.io/dr/failover \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"region": "us-east-1", "reason": "Planned maintenance"}'
```

## Execution Steps

### Step 1: Detect and Validate Failure (2 minutes)

**Actions**:
- Confirm region is unreachable
- Verify failure is not transient (wait 30s, retry)
- Check multiple health endpoints
- Validate failure type (network, infrastructure, etc.)

**Validation**:
- [ ] Region health checks failing
- [ ] No heartbeat received
- [ ] Network connectivity lost
- [ ] API endpoints unresponsive

### Step 2: Notify Stakeholders (30 seconds)

**Actions**:
- Send PagerDuty alert
- Post to Slack #incidents channel
- Send email to ops-team@novacron.io
- Update status page (status.novacron.io)

**Notification Template**:
```
CRITICAL: Region Failure Detected
Region: us-east-1
Time: 2024-01-15 14:30:00 UTC
Automated failover initiated
Estimated completion: 14:45:00 UTC
Status: https://status.novacron.io/incident/12345
```

### Step 3: Verify Quorum (1 minute)

**Actions**:
- Count active regions
- Include witness node in quorum
- Verify quorum size: (N/2 + 1)
- Abort if quorum not available

**Requirements**:
- Minimum 2 active regions (including witness)
- Witness node healthy
- No other ongoing failures

**Failure Handling**:
- If quorum lost: ABORT and notify ops team
- Manual intervention required

### Step 4: Verify Recent Backups (2 minutes)

**Actions**:
- Check last full backup (within 24 hours)
- Check last incremental backup (within RPO)
- Verify transaction logs available
- Validate backup integrity

**Validation**:
- [ ] Full backup within 24 hours
- [ ] Incremental backup within 5 minutes
- [ ] Transaction logs continuous
- [ ] Backup checksums valid

### Step 5: Select Target Region (1 minute)

**Scoring Criteria**:
- Health score: 40% weight
- Available capacity: 30% weight
- Network latency: 20% weight
- Data freshness: 10% weight

**Example Scores**:
```
us-west-2:
  Health: 0.95 * 0.4 = 0.38
  Capacity: 0.80 * 0.3 = 0.24
  Latency: 0.90 * 0.2 = 0.18
  Freshness: 1.0 * 0.1 = 0.10
  Total: 0.90

eu-west-1:
  Health: 0.90 * 0.4 = 0.36
  Capacity: 0.70 * 0.3 = 0.21
  Latency: 0.60 * 0.2 = 0.12
  Freshness: 0.95 * 0.1 = 0.095
  Total: 0.785

Selected: us-west-2 (highest score)
```

### Step 6: Synchronize State (5 minutes)

**Data to Sync**:
- CRDT state from all regions
- Consensus logs (Raft/Paxos)
- VM state and metadata
- Network topology
- Configuration data

**Process**:
1. Create state snapshot
2. Transfer to target region
3. Validate consistency
4. Apply any pending changes

**Monitoring**:
- Transfer rate: >100 MB/s
- Progress tracking
- Checksum validation

### Step 7: Promote Target Region (2 minutes)

**Actions**:
- Update region role: secondary → primary
- Initialize consensus leadership
- Activate write permissions
- Update internal routing

**Approval Required**: Yes (if configured)

**Approval Process**:
1. Send approval request to ops team
2. Wait max 5 minutes for response
3. Auto-approve after timeout (if configured)
4. Log approval decision

### Step 8: Update DNS (3 minutes)

**DNS Updates**:
```
api.novacron.io:
  Old: us-east-1.api.novacron.io (A: 1.2.3.4)
  New: us-west-2.api.novacron.io (A: 5.6.7.8)

dashboard.novacron.io:
  Old: us-east-1.dashboard.novacron.io
  New: us-west-2.dashboard.novacron.io
```

**Providers**:
- AWS Route53 (30s propagation)
- CloudFlare (10s propagation)
- Internal DNS (immediate)

**Validation**:
- Wait for DNS propagation
- Verify DNS resolution
- Test connectivity to new endpoints

### Step 9: Validate Failover (5 minutes)

**Health Checks**:
- [ ] All services responding
- [ ] API endpoints healthy
- [ ] Database connections active
- [ ] VM management operational
- [ ] Load balancer routing correctly

**Performance Checks**:
- [ ] Response time < 200ms
- [ ] Error rate < 1%
- [ ] Throughput > 90% of normal

**Data Validation**:
- [ ] CRDT state consistent
- [ ] No data loss detected
- [ ] Recent transactions visible

### Step 10: Cleanup Failed Region (5 minutes)

**Fencing**:
- Execute STONITH (power off nodes)
- Update firewall rules (block access)
- Revoke storage access
- Remove from load balancer

**Cleanup**:
- Stop all services
- Document failure state
- Preserve logs for analysis
- Tag resources for review

### Step 11: Final Notification (30 seconds)

**Success Notification**:
```
RESOLVED: Region Failover Completed
Original Region: us-east-1 (failed)
New Primary: us-west-2
Failover Duration: 15 minutes 32 seconds
RTO Target: 30 minutes (met)
Data Loss: 0 bytes (RPO met)
Status: https://status.novacron.io/incident/12345
```

**Post-Failover Actions**:
- Update documentation
- Schedule post-mortem
- Review metrics
- Plan recovery of failed region

## Rollback Procedure

If failover fails at any step:

1. **Stop** current failover
2. **Preserve** failed region state
3. **Restore** previous primary (if possible)
4. **Notify** ops team immediately
5. **Escalate** to engineering on-call

## Success Criteria

- ✅ RTO met (<30 minutes)
- ✅ RPO met (<5 minutes)
- ✅ All services operational
- ✅ No data loss
- ✅ Health checks passing

## Metrics

Track these metrics for each failover:

- **Detection Time**: Failure detected to failover initiated
- **Execution Time**: Failover initiated to completed
- **Data Loss**: Bytes of data lost
- **Downtime**: Time services unavailable
- **Success Rate**: Percentage of successful failovers

## Testing

**Test Schedule**:
- Monthly: Simulated region failure (staging)
- Quarterly: Full failover test (production, off-hours)
- Yearly: Multi-region failure simulation

**Test Checklist**:
- [ ] All automation working
- [ ] Notifications sent
- [ ] DNS updated correctly
- [ ] Services recovered
- [ ] Data consistent
- [ ] Performance acceptable

## Contact Information

**On-Call**:
- Primary: ops-team@novacron.io
- Escalation: engineering@novacron.io
- PagerDuty: +1-800-555-0100

**External**:
- AWS Support: Case priority High
- DNS Provider: Emergency contact
- Security Team: incidents@novacron.io

## Version History

- v1.0 (2024-01-15): Initial version
- v1.1 (2024-01-20): Added approval step
- v1.2 (2024-02-01): Updated DNS configuration
