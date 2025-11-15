# DWCP Systems - Operations Playbook

**Version:** 1.0
**Date:** 2025-11-14
**Scope:** Day-to-day operations for 6 production DWCP systems

## Day 1 Operations Checklist

### Morning Startup (00:00-08:00 UTC)

**System Health Verification (30 minutes)**
```bash
# Run comprehensive health check
./scripts/morning-health-check.sh

# Check all services running
systemctl status dwcp-manager
systemctl status compression-api
systemctl status probft-node
systemctl status bullshark-node
systemctl status tpbft-node
systemctl status maddpg-allocator

# Verify cluster health
curl http://monitoring/api/clusters/all/health

# Review overnight alerts
./scripts/review-alerts.sh --since "24h ago"
```

**Expected Results:**
- ✅ All services: Active (running)
- ✅ All health checks: Passing
- ✅ Alert count: 0 critical, <5 warnings
- ✅ Resource usage: Within normal ranges

### Daily Monitoring (08:00-17:00 UTC)

**Hourly Checks (automated)**
- System metrics collection
- Error rate monitoring
- Performance baseline comparison
- Capacity utilization tracking

**Manual Reviews (every 4 hours)**
```bash
# Check consensus health
./scripts/check-consensus-health.sh

# Review error logs
./scripts/review-error-logs.sh --last 4h

# Verify ML model performance
curl http://compression-api:5000/api/v1/model/info
curl http://maddpg:8080/metrics | grep optimization

# Check resource allocation efficiency
./scripts/resource-allocation-report.sh
```

### Evening Shutdown (17:00-00:00 UTC)

**End-of-Day Reports (17:00)**
```bash
# Generate daily performance report
./scripts/daily-performance-report.sh

# Review capacity trends
./scripts/capacity-trend-analysis.sh

# Check for any anomalies
./scripts/anomaly-detection.sh --date today
```

**Nightly Maintenance (22:00)**
```bash
# Rotate logs
./scripts/rotate-logs.sh

# Backup configurations
./scripts/backup-configs.sh

# Run database maintenance
./scripts/db-maintenance.sh

# Clear old metrics (>90 days)
./scripts/cleanup-old-metrics.sh --age 90
```

## Weekly Operations

### Monday: Capacity Planning Review
```bash
# Analyze last week's capacity
./scripts/weekly-capacity-report.sh

# Forecast next week's needs
./scripts/capacity-forecast.sh --weeks 1

# Review and adjust auto-scaling thresholds
./scripts/review-autoscaling.sh
```

### Wednesday: Performance Tuning
```bash
# Analyze consensus performance
./scripts/consensus-performance-analysis.sh

# Review ML model accuracy
./scripts/ml-model-validation.sh

# Optimize resource allocation
./scripts/resource-optimization-review.sh
```

### Friday: Health & Security Review
```bash
# Weekly security scan
./scripts/security-scan.sh

# Review access logs
./scripts/access-log-analysis.sh --last-week

# Update dependencies (if needed)
./scripts/dependency-update-check.sh

# Generate weekly ops report
./scripts/weekly-ops-report.sh
```

## Monthly Operations

### First Monday: Configuration Audit
- Review all system configurations
- Verify backup integrity
- Test disaster recovery procedures
- Update documentation

### Mid-Month: Performance Review
- Analyze monthly performance trends
- Identify optimization opportunities
- Review and update SLAs
- Capacity planning for next month

### End-of-Month: Compliance & Reporting
- Generate monthly compliance report
- Review security posture
- Update runbooks based on incidents
- Team retrospective and process improvements

## Incident Response Procedures

### P0: Critical Incident (Immediate Response)

**Detection:**
- Automated alert: Service down
- Consensus failure
- Data corruption
- Security breach

**Response Steps:**
1. Acknowledge alert (< 2 minutes)
2. Assess impact and scope (< 5 minutes)
3. Initiate incident response protocol
4. Execute recovery procedure (see runbooks)
5. Communicate to stakeholders
6. Post-incident review within 24 hours

**Communication:**
```
Subject: [P0] DWCP System Incident - [Brief Description]

Status: Investigating/Mitigating/Resolved
Impact: [User impact description]
ETA: [Time to resolution]
Updates: [Frequency of updates]

Team: [Incident commander, responders]
```

### P1: High Priority (Response within 15 minutes)

**Detection:**
- High error rate (>0.5%)
- Performance degradation (>50%)
- Resource exhaustion warning (>90%)

**Response Steps:**
1. Acknowledge alert (< 15 minutes)
2. Investigate root cause
3. Implement mitigation if needed
4. Monitor for escalation
5. Document in incident log

### P2: Medium Priority (Response within 1 hour)

**Detection:**
- Warning thresholds exceeded
- Non-critical errors
- Capacity planning triggers

**Response Steps:**
1. Review alert context
2. Schedule maintenance if needed
3. Document for weekly review
4. Update monitoring if necessary

## Escalation Paths

### Level 1: On-Call Engineer
- Initial response
- Standard troubleshooting
- Execute runbook procedures
- **Escalate if:** Issue persists >30 minutes or requires architectural decision

### Level 2: Senior Platform Engineer
- Complex troubleshooting
- Architecture-level decisions
- Cross-system coordination
- **Escalate if:** Multi-system impact or data integrity concern

### Level 3: Engineering Leadership
- Business impact decisions
- Major architectural changes
- Security incidents
- External communication approval

## Common Operational Tasks

### Adding a New Node to Consensus Cluster

**ProBFT:**
```bash
# 1. Prepare new node
./scripts/prepare-probft-node.sh node-08

# 2. Generate VRF keys
ssh node-08 "./probft-node --generate-vrf-keys"

# 3. Update cluster configuration
./scripts/add-node-to-cluster.sh probft node-08

# 4. Rolling restart existing nodes
./scripts/rolling-restart.sh probft

# 5. Verify cluster health
./scripts/verify-cluster.sh probft
```

### Updating ML Model

**Compression Selector:**
```bash
# 1. Train new model
cd /home/kp/repos/novacron/backend/ml
python3 models/compression_selector.py

# 2. Validate model performance
python3 -m tests.test_compression_selector

# 3. Deploy to staging
./scripts/deploy-model.sh compression-selector staging

# 4. Canary deployment (10%)
./scripts/canary-deploy.sh compression-api 10%

# 5. Full deployment if successful
./scripts/canary-deploy.sh compression-api 100%
```

### Database Maintenance

```bash
# Weekly vacuum
psql -U dwcp -d dwcp_db -c "VACUUM ANALYZE;"

# Reindex if needed
psql -U dwcp -d dwcp_db -c "REINDEX DATABASE dwcp_db;"

# Check for bloat
./scripts/check-db-bloat.sh

# Backup before major maintenance
pg_dump dwcp_db > /backup/dwcp_db_$(date +%Y%m%d).sql
```

## Performance Tuning Guide

### Identifying Performance Issues

**Step 1: Metrics Review**
```bash
# Check key metrics
curl http://prometheus:9090/api/v1/query?query=dwcp_request_duration_seconds{quantile="0.99"}

# Identify slow endpoints
./scripts/slow-endpoint-analysis.sh

# Resource bottlenecks
./scripts/resource-bottleneck-detection.sh
```

**Step 2: Profiling**
```bash
# Go services
curl http://dwcp-manager:8080/debug/pprof/profile > cpu.prof
go tool pprof cpu.prof

# Python services
py-spy record -o compression-api.svg --pid $(pgrep -f compression-api)
```

**Step 3: Optimization**
- Adjust configuration parameters (see runbooks)
- Scale horizontally if CPU/memory bound
- Optimize queries if database bound
- Add caching layer if appropriate

---
**Document Version:** 1.0
**Last Updated:** 2025-11-14
**Owner:** Operations Team
