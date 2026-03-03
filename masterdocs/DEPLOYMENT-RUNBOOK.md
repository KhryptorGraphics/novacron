# NovaCron Deployment Runbook

## Overview

This runbook provides step-by-step procedures for deploying NovaCron to production, handling incidents, and performing operational tasks.

## Table of Contents

1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Standard Deployment Procedure](#standard-deployment-procedure)
3. [Emergency Rollback Procedure](#emergency-rollback-procedure)
4. [Incident Response](#incident-response)
5. [Health Check Procedures](#health-check-procedures)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Maintenance Procedures](#maintenance-procedures)

## Pre-Deployment Checklist

### 1. Code Review

- [ ] All pull requests reviewed and approved
- [ ] Code follows project style guidelines
- [ ] No unresolved comments or concerns
- [ ] All required approvals obtained
- [ ] Security review completed (if required)

### 2. Testing Verification

- [ ] All unit tests passing (backend: 70%+ coverage)
- [ ] All integration tests passing
- [ ] Frontend E2E tests passing
- [ ] DWCP protocol tests passing
- [ ] Performance benchmarks meet baseline
- [ ] Security scans show no critical issues

### 3. Environment Preparation

- [ ] Staging deployment successful
- [ ] Staging smoke tests passed
- [ ] Production secrets updated (if required)
- [ ] Database migrations reviewed
- [ ] Configuration changes validated
- [ ] Monitoring dashboards ready
- [ ] Alert rules configured
- [ ] On-call engineer notified

### 4. Communication

- [ ] Deployment scheduled in team calendar
- [ ] Stakeholders notified
- [ ] Maintenance window communicated (if required)
- [ ] Rollback plan documented
- [ ] Post-deployment verification plan ready

### 5. Backup Verification

- [ ] Latest database backup confirmed
- [ ] Backup restoration tested (within last 7 days)
- [ ] Configuration backup taken
- [ ] Previous deployment artifacts available

## Standard Deployment Procedure

### Step 1: Pre-Deployment Backup (Duration: 5-10 minutes)

**Trigger automated backup:**
```bash
# SSH to production Kubernetes cluster
kubectl config use-context production

# Trigger backup job
kubectl create job --from=cronjob/novacron-backup \
  backup-pre-deploy-$(date +%s) -n novacron

# Wait for backup completion
kubectl wait --for=condition=complete \
  --timeout=600s \
  job/backup-pre-deploy-* -n novacron

# Verify backup
kubectl logs job/backup-pre-deploy-* -n novacron
```

**Expected Output:**
```
Backup started at 2025-11-10 15:30:00
Database backup: SUCCESS (size: 2.3 GB)
Configuration backup: SUCCESS
Backup uploaded to: gs://novacron-backups/2025-11-10/...
Backup completed successfully
```

### Step 2: Deploy Green Environment (Duration: 10-15 minutes)

**Trigger GitHub Actions workflow:**
```bash
# Tag the release
git tag -a v1.6.0 -m "Release v1.6.0: DWCP v3 integration"
git push origin v1.6.0

# Monitor workflow
gh workflow view "NovaCron Production CI/CD Pipeline"
gh run watch
```

**Or manual deployment:**
```bash
# Set environment variables
export VERSION=v1.6.0
export IMAGE_TAG=v1.6.0

# Deploy green environment
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/secrets.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml

# Deploy green version (parallel to blue)
kubectl apply -f deployment/kubernetes/deployment-green.yaml

# Wait for green deployment readiness
kubectl rollout status deployment/novacron-api-green -n novacron --timeout=600s
```

### Step 3: Green Environment Validation (Duration: 5-10 minutes)

**Run health checks:**
```bash
# Get green pod IP
GREEN_POD=$(kubectl get pod -n novacron \
  -l app=novacron,version=green \
  -o jsonpath='{.items[0].metadata.name}')

# Test health endpoint
kubectl exec -n novacron $GREEN_POD -- \
  curl -f http://localhost:8090/health

# Test database connectivity
kubectl exec -n novacron $GREEN_POD -- \
  curl -f http://localhost:8090/health/db

# Test DWCP endpoints
kubectl exec -n novacron $GREEN_POD -- \
  curl -f http://localhost:8090/api/v1/dwcp/status
```

**Run smoke tests:**
```bash
# Port forward to green deployment
kubectl port-forward -n novacron \
  deployment/novacron-api-green 8091:8090 &

# Run smoke test suite
./scripts/smoke-tests.sh http://localhost:8091

# Expected output:
# ✓ Health check: PASS
# ✓ API info: PASS
# ✓ DWCP status: PASS
# ✓ Database connectivity: PASS
# ✓ Cache connectivity: PASS
```

### Step 4: Traffic Switch (Duration: 2-5 minutes)

**Gradual traffic shift (recommended):**
```bash
# 5% traffic to green
kubectl patch service novacron-api -n novacron -p '
{
  "spec": {
    "selector": {
      "app": "novacron",
      "version": "green"
    },
    "sessionAffinity": "ClientIP"
  }
}'

# Monitor metrics for 5 minutes
kubectl top pods -n novacron
watch -n 5 'curl -s https://novacron.local/metrics | grep http_requests_total'

# If metrics look good, increase to 25%
# Continue monitoring...

# If metrics look good, switch 100% traffic
kubectl patch service novacron-api -n novacron -p '
{
  "spec": {
    "selector": {
      "app": "novacron",
      "version": "green"
    }
  }
}'
```

**Immediate switch (use with caution):**
```bash
# Switch all traffic to green
kubectl patch service novacron-api -n novacron -p '
{
  "spec": {
    "selector": {
      "app": "novacron",
      "version": "green"
    }
  }
}'
```

### Step 5: Post-Deployment Verification (Duration: 10-15 minutes)

**Verify production endpoints:**
```bash
# Health check
curl -f https://novacron.local/health

# API info
curl -f https://novacron.local/api/v1/info

# DWCP status
curl -f https://novacron.local/api/v1/dwcp/status

# Metrics endpoint
curl -f https://novacron.local/metrics | grep "dwcp_"
```

**Run comprehensive tests:**
```bash
# Run integration test suite against production
export API_ENDPOINT=https://novacron.local
go test -v -tags=integration ./test/integration/...

# Run performance smoke test
ab -n 1000 -c 10 https://novacron.local/health
```

**Monitor metrics:**
```bash
# Watch key metrics for 15 minutes
# - Request rate
# - Error rate
# - Response time
# - DWCP protocol metrics

# Grafana dashboard:
# https://grafana.novacron.local/d/novacron-production
```

### Step 6: Blue Environment Cleanup (Duration: 2-5 minutes)

**After 24 hours of successful operation:**
```bash
# Scale down blue deployment
kubectl scale deployment novacron-api-blue \
  -n novacron --replicas=0

# Wait 1 hour, then delete blue deployment
kubectl delete deployment novacron-api-blue -n novacron

# Rename green to blue for next deployment
kubectl patch deployment novacron-api-green -n novacron -p '
{
  "metadata": {
    "name": "novacron-api-blue"
  }
}'
```

### Step 7: Post-Deployment Tasks

**Update documentation:**
```bash
# Update deployment history
echo "$(date): v1.6.0 deployed successfully" >> docs/deployment-history.md

# Commit deployment notes
git add docs/deployment-history.md
git commit -m "docs: Record v1.6.0 deployment"
git push origin main
```

**Notify stakeholders:**
```bash
# Slack notification (automated via GitHub Actions)
# Manual notification if needed:
curl -X POST $SLACK_WEBHOOK_URL \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "✅ NovaCron v1.6.0 deployed to production successfully!",
    "attachments": [{
      "color": "good",
      "fields": [
        {"title": "Version", "value": "v1.6.0", "short": true},
        {"title": "Deployment Time", "value": "15:45 UTC", "short": true},
        {"title": "Deployed By", "value": "DevOps Team", "short": true}
      ]
    }]
  }'
```

## Emergency Rollback Procedure

### When to Rollback

**Immediate rollback triggers:**
- Health checks failing (> 3 consecutive failures)
- Error rate > 1% of requests
- 5xx errors > 10 per minute
- Response time > 2x baseline
- DWCP protocol failures
- Database connection issues
- Security breach detected

### Quick Rollback (< 2 minutes)

**Step 1: Switch traffic back to blue:**
```bash
# Immediate traffic switch
kubectl patch service novacron-api -n novacron -p '
{
  "spec": {
    "selector": {
      "app": "novacron",
      "version": "blue"
    }
  }
}'

# Verify rollback
curl -f https://novacron.local/health
kubectl get endpoints novacron-api -n novacron
```

**Step 2: Verify blue environment:**
```bash
# Check blue pods
kubectl get pods -n novacron -l version=blue
kubectl top pods -n novacron -l version=blue

# Monitor metrics
watch -n 5 'curl -s https://novacron.local/metrics | grep http_requests_total'
```

**Step 3: Scale down green environment:**
```bash
# Scale down failed green deployment
kubectl scale deployment novacron-api-green \
  -n novacron --replicas=0

# Collect logs for post-mortem
kubectl logs deployment/novacron-api-green -n novacron \
  --tail=1000 > /tmp/green-deployment-failure-$(date +%s).log
```

### Database Rollback

**If database migration caused issues:**
```bash
# Check current migration version
kubectl exec -n novacron deployment/novacron-api-blue -- \
  ./scripts/migration-status.sh

# Rollback migration
kubectl exec -n novacron deployment/novacron-api-blue -- \
  ./scripts/migration-rollback.sh --version 20250110

# Or restore from backup
kubectl create job --from=cronjob/novacron-restore \
  restore-emergency-$(date +%s) -n novacron

# Wait for restoration
kubectl wait --for=condition=complete \
  --timeout=1800s \
  job/restore-emergency-* -n novacron
```

### Full Environment Rollback

**If partial rollback insufficient:**
```bash
# Restore entire environment to previous version
./scripts/rollback-environment.sh --version v1.5.0

# This will:
# 1. Restore database from backup
# 2. Restore configuration
# 3. Deploy previous container images
# 4. Verify health checks
# 5. Update monitoring
```

## Incident Response

### Severity Levels

**P0 (Critical) - Full outage:**
- Service completely unavailable
- Response time: Immediate
- Actions: Page on-call engineer, initiate rollback

**P1 (High) - Partial outage:**
- Significant feature degradation
- Response time: Within 15 minutes
- Actions: Alert on-call, investigate and resolve

**P2 (Medium) - Performance degradation:**
- Slow response times, high error rate
- Response time: Within 1 hour
- Actions: Investigate during business hours

**P3 (Low) - Minor issues:**
- Non-critical feature issues
- Response time: Next business day
- Actions: Create ticket, fix in next release

### Incident Response Procedure

**Step 1: Acknowledge and assess (5 minutes):**
```bash
# Check overall system status
kubectl get pods -n novacron
kubectl top nodes
kubectl top pods -n novacron

# Check recent events
kubectl get events -n novacron --sort-by='.lastTimestamp' | head -20

# Check service endpoints
kubectl get endpoints -n novacron

# Review metrics
# Open Grafana dashboard: https://grafana.novacron.local
```

**Step 2: Investigate (10-15 minutes):**
```bash
# Check application logs
kubectl logs -n novacron deployment/novacron-api --tail=100

# Check for errors
kubectl logs -n novacron deployment/novacron-api \
  --tail=1000 | grep -i "error\|fatal\|panic"

# Check database connectivity
kubectl exec -n novacron deployment/novacron-api -- \
  curl -f http://localhost:8090/health/db

# Check cache connectivity
kubectl exec -n novacron deployment/novacron-api -- \
  curl -f http://localhost:8090/health/cache

# Review Prometheus alerts
curl -s http://prometheus:9090/api/v1/alerts | jq .
```

**Step 3: Mitigate (varies):**
```bash
# Option 1: Restart pods (if memory leak or deadlock)
kubectl rollout restart deployment/novacron-api -n novacron

# Option 2: Scale up (if resource exhaustion)
kubectl scale deployment/novacron-api -n novacron --replicas=5

# Option 3: Rollback (if recent deployment caused issue)
# See "Emergency Rollback Procedure" above

# Option 4: Clear cache (if cache corruption)
kubectl exec -n novacron deployment/novacron-api -- \
  redis-cli -h redis FLUSHDB
```

**Step 4: Verify resolution:**
```bash
# Run health checks
./scripts/health-check.sh production

# Monitor metrics for 15 minutes
# - Error rate should decrease
# - Response time should normalize
# - Resource usage should stabilize

# Run smoke tests
./scripts/smoke-tests.sh https://novacron.local
```

**Step 5: Document incident:**
```markdown
# Incident Report: [Brief Description]

## Summary
- Date: 2025-11-10
- Duration: 15:30 - 15:45 UTC (15 minutes)
- Severity: P1
- Impact: 5% of requests failing

## Timeline
- 15:30: Alert triggered (high error rate)
- 15:32: Engineer acknowledged
- 15:35: Root cause identified (database connection pool exhausted)
- 15:40: Mitigation applied (increased pool size)
- 15:45: Service restored

## Root Cause
Database connection pool size insufficient for traffic spike

## Resolution
- Increased max connections from 100 to 200
- Added connection pool monitoring
- Adjusted autoscaling thresholds

## Action Items
- [ ] Update connection pool defaults
- [ ] Add connection pool alerts
- [ ] Review autoscaling configuration
- [ ] Document incident in knowledge base
```

## Health Check Procedures

### Manual Health Checks

**Application health:**
```bash
# Basic health check
curl -f https://novacron.local/health

# Expected response:
# {"status": "healthy", "timestamp": "2025-11-10T15:30:00Z"}

# Detailed health check
curl -f https://novacron.local/health/detailed

# Expected response:
# {
#   "status": "healthy",
#   "components": {
#     "database": "healthy",
#     "cache": "healthy",
#     "dwcp": "healthy"
#   }
# }
```

**Database health:**
```bash
# Connection test
kubectl exec -n novacron deployment/novacron-api -- \
  psql -h postgres -U novacron -d novacron -c "SELECT 1"

# Expected output: 1

# Check connections
kubectl exec -n novacron deployment/novacron-api -- \
  psql -h postgres -U novacron -d novacron -c \
  "SELECT count(*) FROM pg_stat_activity"
```

**Cache health:**
```bash
# Redis ping
kubectl exec -n novacron deployment/novacron-api -- \
  redis-cli -h redis PING

# Expected output: PONG

# Check memory usage
kubectl exec -n novacron deployment/novacron-api -- \
  redis-cli -h redis INFO memory | grep used_memory_human
```

**DWCP protocol health:**
```bash
# Check DWCP status
curl -f https://novacron.local/api/v1/dwcp/status

# Expected response:
# {
#   "status": "active",
#   "active_streams": 15,
#   "bandwidth_allocated": 1000000000,
#   "compression_ratio": 2.5
# }

# Check DWCP metrics
curl -s https://novacron.local/metrics | grep "dwcp_"
```

### Automated Health Check Script

```bash
#!/bin/bash
# scripts/health-check.sh

ENVIRONMENT=$1
BASE_URL="https://novacron.local"

if [ "$ENVIRONMENT" = "staging" ]; then
  BASE_URL="https://staging.novacron.local"
fi

echo "Running health checks against $BASE_URL..."

# Test health endpoint
if curl -sf "$BASE_URL/health" > /dev/null; then
  echo "✓ Health check: PASS"
else
  echo "✗ Health check: FAIL"
  exit 1
fi

# Test API info
if curl -sf "$BASE_URL/api/v1/info" > /dev/null; then
  echo "✓ API info: PASS"
else
  echo "✗ API info: FAIL"
  exit 1
fi

# Test DWCP status
if curl -sf "$BASE_URL/api/v1/dwcp/status" > /dev/null; then
  echo "✓ DWCP status: PASS"
else
  echo "✗ DWCP status: FAIL"
  exit 1
fi

echo "All health checks passed!"
```

## Troubleshooting Guide

### Common Issues

#### Issue: Pods in CrashLoopBackOff

**Symptoms:**
```bash
kubectl get pods -n novacron
# NAME                           READY   STATUS             RESTARTS
# novacron-api-xxx               0/1     CrashLoopBackOff   5
```

**Diagnosis:**
```bash
# Check pod logs
kubectl logs -n novacron novacron-api-xxx

# Check previous logs
kubectl logs -n novacron novacron-api-xxx --previous

# Describe pod
kubectl describe pod -n novacron novacron-api-xxx
```

**Common causes and solutions:**
1. Configuration error → Check ConfigMap and Secrets
2. Database unavailable → Verify database connectivity
3. Insufficient resources → Increase CPU/memory limits
4. Failed health checks → Review liveness/readiness probes

#### Issue: High Memory Usage

**Symptoms:**
```bash
kubectl top pods -n novacron
# NAME                           CPU     MEMORY
# novacron-api-xxx               500m    1800Mi/2048Mi
```

**Diagnosis:**
```bash
# Check memory metrics
kubectl exec -n novacron deployment/novacron-api -- \
  curl -s http://localhost:9090/metrics | grep "go_memstats"

# Check for memory leaks
kubectl exec -n novacron deployment/novacron-api -- \
  curl -s http://localhost:9090/debug/pprof/heap > heap.profile
```

**Solutions:**
1. Restart pod (temporary)
2. Increase memory limits
3. Investigate memory leaks (profile analysis)
4. Implement memory optimization

#### Issue: Database Connection Pool Exhausted

**Symptoms:**
```
Error: database connection pool exhausted
Error: could not acquire connection
```

**Diagnosis:**
```bash
# Check active connections
kubectl exec -n novacron deployment/novacron-api -- \
  psql -h postgres -c "SELECT count(*) FROM pg_stat_activity"

# Check pool configuration
kubectl get configmap -n novacron novacron-config -o yaml | grep pool
```

**Solutions:**
1. Increase max connections in ConfigMap
2. Review connection leak (application logs)
3. Scale up application replicas
4. Optimize database queries

#### Issue: High Latency

**Symptoms:**
- Response time > 1 second
- Prometheus alert: HighResponseTime

**Diagnosis:**
```bash
# Check response time percentiles
curl -s https://novacron.local/metrics | \
  grep "http_request_duration_seconds"

# Check database query performance
kubectl exec -n novacron deployment/novacron-api -- \
  psql -h postgres -c "SELECT * FROM pg_stat_statements \
  ORDER BY mean_exec_time DESC LIMIT 10"

# Check DWCP latency
curl -s https://novacron.local/metrics | grep "dwcp_latency"
```

**Solutions:**
1. Database optimization (add indexes, query tuning)
2. Cache implementation (Redis)
3. DWCP parameter tuning
4. Scale up resources

## Maintenance Procedures

### Database Maintenance

**Vacuum database (monthly):**
```bash
kubectl exec -n novacron deployment/novacron-api -- \
  psql -h postgres -c "VACUUM ANALYZE"
```

**Reindex tables (quarterly):**
```bash
kubectl exec -n novacron deployment/novacron-api -- \
  psql -h postgres -c "REINDEX DATABASE novacron"
```

### Certificate Renewal

**TLS certificates (automatic via cert-manager):**
```bash
# Check certificate expiration
kubectl get certificate -n novacron

# Manually trigger renewal if needed
kubectl delete certificate novacron-tls -n novacron
# cert-manager will automatically recreate
```

### Log Rotation

**Application logs:**
```bash
# Logs automatically rotated by Kubernetes
# Check log retention
kubectl get pods -n novacron -o yaml | grep "maxLogFiles"
```

### Image Cleanup

**Remove old container images (monthly):**
```bash
# GitHub Container Registry cleanup
# Automated via GitHub Actions workflow
gh api repos/{owner}/{repo}/actions/artifacts --paginate \
  | jq '.artifacts[] | select(.expired == false) | .id' \
  | xargs -I {} gh api -X DELETE repos/{owner}/{repo}/actions/artifacts/{}
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-10
**Owner:** DevOps Team
**Review Cycle:** Quarterly
