# DWCP v3 Production Go-Live Runbook

**Document Version:** 1.0
**Date:** 2025-11-10
**Project:** NovaCron DWCP v3
**Environment:** Production

---

## Executive Summary

This runbook provides detailed procedures for deploying DWCP v3 to production. Follow all steps sequentially and validate each phase before proceeding.

**Estimated Total Time:** 4-6 hours
**Rollback Time:** 30 minutes
**Team Size:** 5-8 personnel

---

## Pre-Deployment Phase (T-7 Days to T-1 Day)

### T-7 Days: Final Preparation Begins

#### Checklist Review
- [ ] Complete production readiness checklist (156 items)
- [ ] Validate all Phase 1-3 implementations
- [ ] Confirm Phase 4 optimizations tested
- [ ] Review performance benchmarks
- [ ] Validate security configurations

**Responsible:** Technical Lead
**Duration:** 4 hours

#### Stakeholder Communication
- [ ] Send go-live notification to stakeholders
- [ ] Schedule deployment window
- [ ] Confirm maintenance window approvals
- [ ] Brief executive team
- [ ] Notify customer success team

**Responsible:** Project Manager
**Duration:** 2 hours

---

### T-5 Days: Infrastructure Preparation

#### Production Environment Setup
```bash
# 1. Provision infrastructure
cd infrastructure/terraform
terraform plan -out=production.tfplan
terraform apply production.tfplan

# 2. Verify infrastructure
./scripts/verify-infrastructure.sh production

# 3. Configure networking
./scripts/configure-networking.sh production
```

**Responsible:** Infrastructure Team
**Duration:** 4 hours

#### Container Images
```bash
# 1. Build production images
docker build -t dwcp-v3:prod-${VERSION} .

# 2. Security scan
trivy image dwcp-v3:prod-${VERSION}

# 3. Push to registry
docker push registry.novacron.io/dwcp-v3:prod-${VERSION}

# 4. Sign images
cosign sign registry.novacron.io/dwcp-v3:prod-${VERSION}
```

**Responsible:** DevOps Team
**Duration:** 2 hours

---

### T-3 Days: Pre-Deployment Validation

#### Staging Environment Testing
```bash
# 1. Deploy to staging
kubectl apply -f k8s/staging/ --namespace=dwcp-staging

# 2. Run smoke tests
./scripts/smoke-tests.sh staging

# 3. Run load tests
./scripts/load-tests.sh staging 1000

# 4. Validate performance
./scripts/validate-performance.sh staging
```

**Expected Results:**
- Throughput: >100 MB/s
- Latency: <50ms
- Error Rate: <0.1%
- CPU Usage: <70%
- Memory Usage: <2GB

**Responsible:** QA Team
**Duration:** 6 hours

#### Security Validation
```bash
# 1. Security scan
./scripts/security-scan.sh staging

# 2. Penetration testing
./scripts/pentest.sh staging

# 3. Compliance check
./scripts/compliance-check.sh staging
```

**Responsible:** Security Team
**Duration:** 4 hours

---

### T-1 Day: Final Preparation

#### Pre-Deployment Checklist
- [ ] All tests passing in staging
- [ ] Performance benchmarks met
- [ ] Security scans clean
- [ ] Backup procedures validated
- [ ] Rollback plan tested
- [ ] Monitoring dashboards configured
- [ ] Alert rules active
- [ ] On-call team briefed
- [ ] Communication plan ready

**Responsible:** All Teams
**Duration:** 4 hours

#### Team Readiness
- [ ] Technical team on standby
- [ ] Operations team ready
- [ ] Security team available
- [ ] Management informed
- [ ] Customer support prepared

**Responsible:** Project Manager
**Duration:** 1 hour

---

## Deployment Day (T-0)

### Phase 1: Pre-Deployment (T-2 hours)

#### 1.1 Team Assembly (T-2:00)
```
Assemble Go-Live Team:
- Technical Lead
- DevOps Engineers (2)
- Operations Manager
- Security Officer
- QA Lead
- Project Manager
```

**Actions:**
- [ ] Open war room (virtual/physical)
- [ ] Start screen sharing
- [ ] Begin recording session
- [ ] Confirm communication channels
- [ ] Review deployment plan

**Duration:** 15 minutes

#### 1.2 System Health Check (T-1:45)
```bash
# Check current production status
./scripts/health-check.sh production

# Verify baseline metrics
./scripts/baseline-metrics.sh production

# Check dependencies
./scripts/check-dependencies.sh production
```

**Expected Output:**
```
✓ All services healthy
✓ Database connections: OK
✓ Network latency: <10ms
✓ CPU: 45% average
✓ Memory: 1.2GB / 4GB
✓ Disk: 30% used
✓ Active migrations: 12
```

**Duration:** 15 minutes

#### 1.3 Create Backup (T-1:30)
```bash
# Full system backup
./scripts/backup-production.sh full

# Verify backup
./scripts/verify-backup.sh latest

# Document backup ID
BACKUP_ID=$(cat /tmp/latest-backup-id)
echo "Backup ID: $BACKUP_ID" >> deployment-log.txt
```

**Validation:**
- [ ] Backup completed successfully
- [ ] Backup size matches expectations
- [ ] Backup integrity verified
- [ ] Backup ID recorded

**Duration:** 30 minutes

#### 1.4 Enable Maintenance Mode (T-1:00)
```bash
# Enable maintenance mode
kubectl patch deployment frontend \
  -p '{"spec":{"replicas":0}}'

# Redirect traffic to maintenance page
./scripts/enable-maintenance.sh

# Verify maintenance mode
curl https://novacron.io/health
# Expected: {"status": "maintenance"}
```

**Duration:** 10 minutes

---

### Phase 2: Deployment (T-0 to T+30)

#### 2.1 Deploy DWCP v3 Components (T-0:00)

**2.1.1 Deploy Core Services**
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/production/namespace.yaml
kubectl apply -f k8s/production/configmaps.yaml
kubectl apply -f k8s/production/secrets.yaml
kubectl apply -f k8s/production/dwcp-core.yaml

# Wait for pods to be ready
kubectl wait --for=condition=ready pod \
  -l app=dwcp-core \
  --timeout=300s
```

**Validation:**
```bash
kubectl get pods -l app=dwcp-core
# Expected: All pods Running (3/3)
```

**Duration:** 10 minutes

**2.1.2 Deploy Federation Services**
```bash
kubectl apply -f k8s/production/federation.yaml

kubectl wait --for=condition=ready pod \
  -l app=dwcp-federation \
  --timeout=300s
```

**Duration:** 5 minutes

**2.1.3 Deploy Migration Services**
```bash
kubectl apply -f k8s/production/migration.yaml

kubectl wait --for=condition=ready pod \
  -l app=dwcp-migration \
  --timeout=300s
```

**Duration:** 5 minutes

#### 2.2 Database Migration (T+10)
```bash
# Run database migrations
kubectl exec -it dwcp-core-0 -- \
  /app/migrate --config=/etc/dwcp/config.yaml

# Verify migration
kubectl exec -it dwcp-core-0 -- \
  /app/migrate --verify
```

**Expected Output:**
```
✓ Migration 001: Core schema
✓ Migration 002: Baseline tables
✓ Migration 003: Metrics tables
✓ All migrations applied successfully
```

**Duration:** 5 minutes

#### 2.3 Configuration Validation (T+15)
```bash
# Validate configurations
./scripts/validate-config.sh production

# Test internal connectivity
./scripts/test-connectivity.sh production

# Verify service mesh
./scripts/verify-mesh.sh production
```

**Duration:** 5 minutes

---

### Phase 3: Validation & Testing (T+20 to T+60)

#### 3.1 Smoke Tests (T+20)
```bash
# Run automated smoke tests
./scripts/smoke-tests.sh production

# Expected results
Test Suite: Smoke Tests
  ✓ Health endpoint responding
  ✓ Metrics endpoint available
  ✓ Authentication working
  ✓ Core services reachable
  ✓ Database connections active

All tests passed (5/5)
```

**Duration:** 10 minutes

#### 3.2 Functional Validation (T+30)
```bash
# Test VM migration
./scripts/test-migration.sh production test-vm-001

# Test federation sync
./scripts/test-federation.sh production cluster-a cluster-b

# Test monitoring
./scripts/test-monitoring.sh production
```

**Expected Output:**
```
Migration Test:
  VM: test-vm-001
  Size: 4GB
  Duration: 12.5s
  Throughput: 320 MB/s
  Speedup: 2.8x
  Status: ✓ PASSED

Federation Test:
  Clusters: cluster-a <-> cluster-b
  Sync time: 0.8s
  Data loss: 0 bytes
  Status: ✓ PASSED
```

**Duration:** 15 minutes

#### 3.3 Performance Validation (T+45)
```bash
# Run performance benchmarks
./scripts/benchmark-production.sh

# Compare with baseline
./scripts/compare-metrics.sh baseline current
```

**Expected Metrics:**
```
Performance Comparison:
Metric              Baseline    Current    Delta
--------------------------------------------------
Throughput         100 MB/s    125 MB/s   +25%
Latency            50ms        42ms       -16%
Compression Ratio  3.0x        3.2x       +6.7%
CPU Usage          65%         58%        -10.8%
Memory Usage       1.8GB       1.6GB      -11.1%
Error Rate         0.10%       0.05%      -50%

Status: ✓ ALL METRICS IMPROVED
```

**Duration:** 15 minutes

---

### Phase 4: Traffic Migration (T+60 to T+90)

#### 4.1 Enable Canary Deployment (T+60)
```bash
# Route 10% traffic to DWCP v3
kubectl patch virtualservice dwcp \
  --type=merge \
  -p '{"spec":{"http":[{"route":[
    {"destination":{"host":"dwcp-v3"},"weight":10},
    {"destination":{"host":"dwcp-v2"},"weight":90}
  ]}]}}'

# Monitor for 10 minutes
watch -n 5 './scripts/monitor-canary.sh'
```

**Validation Criteria:**
- Error rate <0.1%
- Latency p99 <100ms
- No service disruptions

**Duration:** 15 minutes

#### 4.2 Gradual Traffic Increase (T+75)
```bash
# Increase to 50% traffic
kubectl patch virtualservice dwcp \
  --type=merge \
  -p '{"spec":{"http":[{"route":[
    {"destination":{"host":"dwcp-v3"},"weight":50},
    {"destination":{"host":"dwcp-v2"},"weight":50}
  ]}]}}'

# Monitor for 10 minutes
watch -n 5 './scripts/monitor-traffic.sh'
```

**Duration:** 15 minutes

---

### Phase 5: Full Cutover (T+90 to T+120)

#### 5.1 Complete Traffic Migration (T+90)
```bash
# Route 100% traffic to DWCP v3
kubectl patch virtualservice dwcp \
  --type=merge \
  -p '{"spec":{"http":[{"route":[
    {"destination":{"host":"dwcp-v3"},"weight":100}
  ]}]}}'

# Disable v2 services
kubectl scale deployment dwcp-v2 --replicas=0
```

**Duration:** 5 minutes

#### 5.2 Disable Maintenance Mode (T+95)
```bash
# Restore frontend
kubectl patch deployment frontend \
  -p '{"spec":{"replicas":3}}'

# Disable maintenance page
./scripts/disable-maintenance.sh

# Verify service
curl https://novacron.io/health
# Expected: {"status": "ok", "version": "v3"}
```

**Duration:** 5 minutes

#### 5.3 Final Validation (T+100)
```bash
# Run full test suite
./scripts/integration-tests.sh production

# Validate all features
./scripts/validate-features.sh production

# Check error rates
./scripts/check-errors.sh production
```

**Expected Output:**
```
Integration Tests:
  ✓ VM Migration (50/50)
  ✓ Federation Sync (25/25)
  ✓ Security (20/20)
  ✓ Monitoring (15/15)

Total: 110/110 tests passed
Status: ✓ ALL SYSTEMS GO
```

**Duration:** 20 minutes

---

## Post-Deployment Phase (T+120 to T+24h)

### Immediate Post-Deployment (T+2h to T+4h)

#### Monitoring & Observation
```bash
# Start intensive monitoring
./scripts/monitor-production.sh --intensive

# Key metrics to watch:
# - Throughput
# - Latency
# - Error rates
# - CPU/Memory usage
# - Active migrations
```

**Actions:**
- [ ] Monitor dashboards continuously
- [ ] Check error logs every 15 minutes
- [ ] Validate performance metrics hourly
- [ ] Verify no degradation

**Duration:** 2 hours

---

### Extended Monitoring (T+4h to T+24h)

#### Stability Verification
```bash
# Generate hourly reports
while true; do
  ./scripts/hourly-report.sh production
  sleep 3600
done
```

**Checkpoints:**
- T+4h: First stability checkpoint
- T+8h: Second stability checkpoint
- T+12h: Third stability checkpoint
- T+24h: Final stability checkpoint

**Actions at Each Checkpoint:**
- [ ] Review performance metrics
- [ ] Check error logs
- [ ] Validate data integrity
- [ ] Confirm no issues reported
- [ ] Update stakeholders

---

### T+24 Hours: Final Sign-Off

#### Success Criteria Validation
```
✓ Throughput: 125 MB/s (Target: 100 MB/s)
✓ Latency: 42ms (Target: <50ms)
✓ Compression: 3.2x (Target: >3x)
✓ Uptime: 100% (Target: >99.9%)
✓ Error Rate: 0.05% (Target: <0.1%)
✓ Zero rollbacks required
✓ Zero critical incidents
✓ All features operational
```

#### Team Debrief
- [ ] Conduct post-deployment review
- [ ] Document lessons learned
- [ ] Update runbook with improvements
- [ ] Celebrate success

#### Final Communication
```
Subject: DWCP v3 Go-Live - SUCCESS

Team,

DWCP v3 has been successfully deployed to production.

Key Achievements:
- Zero downtime deployment
- 25% throughput improvement
- 16% latency reduction
- All performance targets exceeded
- No incidents or issues

The system is now stable and operating normally.

Thank you for your excellent work!
```

---

## Rollback Procedures

### When to Rollback

Trigger rollback if ANY of the following occur:
- Error rate exceeds 1%
- Critical service outage
- Data corruption detected
- Performance degradation >20%
- Security breach identified

### Rollback Steps (30 minutes)

#### 1. Immediate Actions (0-5 min)
```bash
# Stop DWCP v3 traffic
kubectl patch virtualservice dwcp \
  --type=merge \
  -p '{"spec":{"http":[{"route":[
    {"destination":{"host":"dwcp-v2"},"weight":100}
  ]}]}}'

# Scale down v3
kubectl scale deployment dwcp-v3 --replicas=0
```

#### 2. Restore v2 (5-15 min)
```bash
# Scale up v2
kubectl scale deployment dwcp-v2 --replicas=5

# Wait for readiness
kubectl wait --for=condition=ready pod \
  -l app=dwcp-v2 \
  --timeout=300s
```

#### 3. Rollback Database (15-25 min)
```bash
# Restore from backup
./scripts/restore-backup.sh $BACKUP_ID

# Verify restoration
./scripts/verify-restore.sh
```

#### 4. Validation (25-30 min)
```bash
# Test v2 functionality
./scripts/smoke-tests.sh production

# Verify metrics
./scripts/check-metrics.sh production
```

---

## Emergency Contacts

### On-Call Team
- **Technical Lead:** [REDACTED]
- **DevOps Lead:** [REDACTED]
- **Operations Manager:** [REDACTED]
- **Security Officer:** [REDACTED]

### Escalation
- **Level 1:** Team Leads (0-15 min)
- **Level 2:** Senior Management (15-30 min)
- **Level 3:** Executive Team (30-60 min)

### Support Channels
- **Slack:** #dwcp-deployment
- **Email:** deployment@novacron.io
- **Phone:** [REDACTED]
- **Emergency:** [REDACTED]

---

## Appendices

### A. Command Reference
```bash
# Health check
curl https://api.novacron.io/v3/health

# Metrics
curl https://api.novacron.io/v3/metrics

# Logs
kubectl logs -f deployment/dwcp-v3 --tail=100

# Events
kubectl get events --sort-by='.lastTimestamp'
```

### B. Troubleshooting Guide
See: docs/DWCP_V3_TROUBLESHOOTING.md

### C. Configuration Reference
See: docs/DWCP_V3_CONFIGURATION.md

---

**Document Control:**
- **Version:** 1.0
- **Last Modified:** 2025-11-10
- **Next Review:** Post-Deployment
- **Classification:** Internal - Confidential
