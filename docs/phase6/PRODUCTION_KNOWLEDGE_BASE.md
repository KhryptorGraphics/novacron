# DWCP v3 Production Knowledge Base and FAQ

**Version**: 1.0.0  
**Last Updated**: 2025-11-10  
**Audience**: All Engineering Teams  
**Classification**: Internal Use

---

## Table of Contents

1. [Frequently Asked Questions](#frequently-asked-questions)
2. [Known Issues and Workarounds](#known-issues-and-workarounds)
3. [Tips and Tricks](#tips-and-tricks)
4. [Troubleshooting Common Problems](#troubleshooting-common-problems)
5. [Performance Optimization Guide](#performance-optimization-guide)
6. [Deployment FAQ](#deployment-faq)
7. [Monitoring and Alerting FAQ](#monitoring-and-alerting-faq)
8. [Database FAQ](#database-faq)
9. [Security FAQ](#security-faq)
10. [Incident Response FAQ](#incident-response-faq)

---

## Frequently Asked Questions

### General Questions

**Q: Where can I find the system architecture documentation?**
A: System architecture documentation is located at:
- `/docs/architecture/SYSTEM_ARCHITECTURE.md`
- `/docs/architecture/COMPONENT_DESIGN.md`
- Confluence: https://wiki.dwcp.io/architecture

**Q: Who do I contact for production issues?**
A: 
- **Severity 1 (Critical)**: Page on-call via PagerDuty
- **Severity 2 (High)**: Post in #incident-response Slack channel
- **Severity 3-4**: Create ticket in JIRA (PROD project)
- **General questions**: #ask-sre Slack channel

**Q: How do I get access to production systems?**
A:
1. Submit access request via ServiceNow
2. Manager approval required
3. Complete production safety training
4. Security team approval
5. Access granted (reviewed quarterly)

**Q: What are our SLAs?**
A:
- **Availability**: 99.95% (21.6 minutes downtime/month)
- **Latency (p95)**: <200ms
- **Error Rate**: <0.1%
- **Support Response**: Critical issues within 15 minutes

**Q: How often do we deploy?**
A: 
- Average: 15 deployments per day
- Peak: Up to 30 deployments per day
- Deployment windows: Anytime (24/7 with canary deployment)
- Freeze periods: During major events, holidays

**Q: What's the process for emergency deployments?**
A:
1. Declare emergency in #incident-response
2. Get approval from on-call SRE + engineering manager
3. Follow emergency deployment runbook
4. Deploy with enhanced monitoring
5. Post-incident review required within 48h

**Q: How do I check system health?**
A:
- **Primary Dashboard**: https://grafana.dwcp.io/d/system-health
- **Service Status**: https://status.dwcp.io
- **CLI**: `./scripts/health-check.sh --all-regions`

**Q: Where are the logs?**
A:
- **Centralized Logs**: https://logs.dwcp.io (Kibana)
- **CLI Access**: `kubectl logs <pod>` or `stern <service>`
- **Log Retention**: 90 days hot, 1 year archived

**Q: How do I create a runbook?**
A:
1. Use template at `/docs/templates/RUNBOOK_TEMPLATE.md`
2. Include: Purpose, Prerequisites, Steps, Rollback, Troubleshooting
3. Test all commands
4. Peer review required
5. Add to runbook registry

**Q: What's the incident response process?**
A:
See `/docs/phase6/INCIDENT_RESPONSE_PLAYBOOKS.md` for detailed playbooks
Quick summary:
1. Detect & Alert
2. Assess & Triage
3. Respond & Mitigate
4. Resolve & Verify
5. Post-Incident Review

---

## Known Issues and Workarounds

### Issue KI-001: Kafka Consumer Lag Spikes

**Symptom**: Consumer lag spikes to 10k+ messages during deployments

**Root Cause**: Consumer rebalancing during rolling deployment

**Workaround**:
```bash
# Temporarily increase partition count
kafka-topics.sh --bootstrap-server kafka.dwcp.io:9092 \
    --alter --topic dwcp-tasks --partitions 30

# After deployment, reduce back to normal
kafka-topics.sh --bootstrap-server kafka.dwcp.io:9092 \
    --alter --topic dwcp-tasks --partitions 20
```

**Permanent Fix**: In progress (IMP-2025-045)
- Implement graceful consumer shutdown
- ETA: Q2 2025

---

### Issue KI-002: Redis Connection Timeout Under Load

**Symptom**: Intermittent Redis connection timeouts when RPS >12k

**Root Cause**: Redis single-threaded, max connections reached

**Workaround**:
```yaml
# Increase connection pool
redis:
  pool_size: 500  # from 200
  connection_timeout: 5s  # from 3s
  
# Or enable Redis Cluster mode
redis:
  mode: cluster
  nodes:
    - redis-0:6379
    - redis-1:6379
    - redis-2:6379
```

**Permanent Fix**: Migrating to Redis Cluster (Q1 2025)

---

### Issue KI-003: Database Connection Pool Exhaustion During Peak

**Symptom**: "Connection pool exhausted" errors 5-7pm daily

**Root Cause**: Fixed pool size unable to handle evening traffic spike

**Workaround**:
```bash
# Scale up database connection pool during peak hours
kubectl set env deployment/dwcp-api \
    DB_POOL_SIZE=150 \
    DB_POOL_MAX=200

# Scale back after peak
kubectl set env deployment/dwcp-api \
    DB_POOL_SIZE=100 \
    DB_POOL_MAX=150
```

**Permanent Fix**: Implemented (IMP-2025-001)
- Dynamic connection pooling based on load
- Auto-scaling connection pool
- Deployed: 2025-01-15

---

### Issue KI-004: Prometheus Query Timeout on Large Time Ranges

**Symptom**: Grafana dashboards timeout when viewing >7 days

**Root Cause**: Large metric cardinality, inefficient queries

**Workaround**:
```bash
# Use aggregated metrics for large time ranges
# Instead of:
rate(http_requests_total[30d])

# Use:
rate(http_requests_total:1h[30d])  # Pre-aggregated hourly
```

**Permanent Fix**: In planning
- Implement Thanos for long-term storage
- Downsampling for old metrics
- ETA: Q2 2025

---

### Issue KI-005: Slow Startup Time for Large Pods

**Symptom**: Pods take 3-5 minutes to become ready after deployment

**Root Cause**: Large Docker images (2.5GB), slow dependency initialization

**Workaround**:
```bash
# Use readiness gates to prevent traffic before fully ready
kubectl patch deployment dwcp-api -p '
spec:
  template:
    spec:
      readinessGates:
      - conditionType: "app.kubernetes.io/pod-ready"
'

# Pre-pull images on nodes
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: image-prepuller
spec:
  selector:
    matchLabels:
      app: image-prepuller
  template:
    metadata:
      labels:
        app: image-prepuller
    spec:
      containers:
      - name: prepuller
        image: dwcp/api:latest
        command: ['sh', '-c', 'sleep infinity']
EOF
```

**Permanent Fix**: In progress
- Optimize Docker image (multi-stage builds)
- Lazy load dependencies
- ETA: Q1 2025

---

## Tips and Tricks

### Tip 1: Fast Log Search Across All Pods

```bash
# Search all pods in namespace for error
kubectl logs -n production --all-containers=true --prefix=true \
    -l app=dwcp-api --since=1h | grep -i error

# Use stern for real-time tailing
stern dwcp-api --since 1h --color always | grep ERROR

# Search with context (5 lines before/after)
kubectl logs -n production deployment/dwcp-api --since=1h | \
    grep -B 5 -A 5 "error"
```

### Tip 2: Quick Performance Profiling

```bash
# Get top CPU/Memory consuming pods
kubectl top pods --all-namespaces --sort-by=cpu | head -20
kubectl top pods --all-namespaces --sort-by=memory | head -20

# Profile specific pod
kubectl exec -it <pod> -- /app/profile --duration 30s

# Get goroutine dump (Go apps)
kubectl exec <pod> -- curl http://localhost:6060/debug/pprof/goroutine?debug=1
```

### Tip 3: Database Query Analysis

```sql
-- Find slow queries
SELECT
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 20;

-- Find missing indexes
SELECT
    schemaname,
    tablename,
    seq_scan,
    seq_tup_read,
    idx_scan
FROM pg_stat_user_tables
WHERE seq_scan > 0
ORDER BY seq_tup_read DESC
LIMIT 20;

-- Find bloated tables
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
    n_dead_tup
FROM pg_stat_user_tables
WHERE n_dead_tup > 1000
ORDER BY n_dead_tup DESC;
```

### Tip 4: Kubernetes Debugging

```bash
# Get events for a pod
kubectl get events --field-selector involvedObject.name=<pod-name>

# Check why pod is pending
kubectl describe pod <pod> | grep -A 10 "Events:"

# Interactive debugging
kubectl run debug --rm -it --image=nicolaka/netshoot -- bash

# Check resource quotas
kubectl describe resourcequota -n production

# Force pod recreation
kubectl delete pod <pod> --grace-period=0 --force
```

### Tip 5: Effective Monitoring Queries

```promql
# Request rate by endpoint (top 10)
topk(10, sum by (path) (rate(http_requests_total[5m])))

# Error rate percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) 
/ 
sum(rate(http_requests_total[5m])) 
* 100

# Latency heatmap
histogram_quantile(0.95, 
  sum by (le) (rate(http_request_duration_seconds_bucket[5m]))
)

# Top memory consumers
topk(10, 
  sum by (pod) (container_memory_working_set_bytes)
)

# Pods restarting frequently
count by (pod) (kube_pod_container_status_restarts_total > 5)
```

### Tip 6: Security Audit Commands

```bash
# Check for secrets in code
git-secrets --scan

# Scan Docker image for vulnerabilities
trivy image dwcp/api:latest --severity HIGH,CRITICAL

# Check for exposed secrets in Kubernetes
kubectl get secrets --all-namespaces -o json | \
    jq -r '.items[] | select(.type=="Opaque") | .metadata.name'

# Audit RBAC permissions
kubectl auth can-i --list --as=system:serviceaccount:production:api

# Check network policies
kubectl get networkpolicies --all-namespaces
```

### Tip 7: Cost Optimization

```bash
# Find over-provisioned pods
kubectl get pods --all-namespaces -o json | \
    jq -r '.items[] | 
    select(.spec.containers[].resources.requests.cpu != null) | 
    "\(.metadata.name) Request:\(.spec.containers[0].resources.requests.cpu) Limit:\(.spec.containers[0].resources.limits.cpu)"'

# Check for unused PVs
kubectl get pv | grep Available

# Identify idle resources
./scripts/find-idle-resources.sh --threshold 20%
```

---

## Troubleshooting Common Problems

### Problem 1: High API Latency

**Symptoms**:
- Latency p95 >1000ms
- User complaints about slow responses
- Timeout errors

**Diagnostic Steps**:
```bash
# 1. Check if issue is widespread
curl -s "http://prometheus.dwcp.io/api/v1/query?query=histogram_quantile(0.95,rate(http_request_duration_seconds_bucket[5m]))"

# 2. Identify slow endpoints
kubectl logs deployment/dwcp-api --since=10m | \
    grep -oP 'path=\K[^ ]+.*duration=\K[0-9]+' | \
    sort -n -t= -k2 | tail -20

# 3. Check database performance
kubectl exec deployment/postgres-primary -- \
    psql -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# 4. Check resource utilization
kubectl top pods -n production --sort-by=cpu

# 5. Check for network issues
./scripts/network-latency-test.sh
```

**Common Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| Slow database queries | Add indexes, optimize queries |
| Resource exhaustion | Scale up pods |
| External dependency slow | Enable circuit breaker, add timeout |
| Memory leak | Restart pods, fix leak |
| Network latency | Check service mesh, DNS |

---

### Problem 2: Pods Stuck in CrashLoopBackOff

**Symptoms**:
- Pods continuously restarting
- Status: CrashLoopBackOff
- Service unavailable

**Diagnostic Steps**:
```bash
# 1. Get pod status
kubectl get pods -n production -l app=dwcp-api

# 2. Check pod logs
kubectl logs <pod> --previous

# 3. Describe pod for events
kubectl describe pod <pod>

# 4. Check for resource limits
kubectl describe pod <pod> | grep -A 5 "Limits:"

# 5. Check configmaps/secrets
kubectl get configmap -n production
kubectl get secrets -n production
```

**Common Causes & Solutions**:

| Exit Code | Cause | Solution |
|-----------|-------|----------|
| 0 | Successful exit (shouldn't restart) | Check restart policy |
| 1 | Application error | Check logs, fix bug |
| 137 | OOMKilled | Increase memory limits |
| 139 | Segmentation fault | Debug application |
| 143 | SIGTERM | Graceful shutdown issue |

---

### Problem 3: Database Connection Issues

**Symptoms**:
- "Connection refused" errors
- "Too many connections" errors
- Database timeouts

**Diagnostic Steps**:
```bash
# 1. Check database pod status
kubectl get pods -n production -l app=postgres

# 2. Check connection count
kubectl exec deployment/postgres-primary -- \
    psql -c "SELECT count(*) FROM pg_stat_activity;"

# 3. Check for long-running queries
kubectl exec deployment/postgres-primary -- \
    psql -c "SELECT pid, now() - query_start as duration, query FROM pg_stat_activity WHERE state = 'active' ORDER BY duration DESC LIMIT 10;"

# 4. Check connection pool config
kubectl get configmap db-config -o yaml

# 5. Test connectivity
kubectl run db-test --rm -it --image=postgres:14 -- \
    psql -h postgres-primary -U app_user -d app_db
```

**Solutions**:
```bash
# Kill idle connections
kubectl exec deployment/postgres-primary -- \
    psql -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND now() - state_change > interval '5 minutes';"

# Increase max connections (requires restart)
kubectl exec deployment/postgres-primary -- \
    psql -c "ALTER SYSTEM SET max_connections = 200;"

# Scale connection pool
kubectl set env deployment/dwcp-api DB_POOL_SIZE=100
```

---

### Problem 4: High Memory Usage

**Symptoms**:
- Memory utilization >85%
- OOMKills
- Slow performance

**Diagnostic Steps**:
```bash
# 1. Check memory usage
kubectl top pods --sort-by=memory | head -20

# 2. Check for memory leaks
kubectl exec <pod> -- curl http://localhost:6060/debug/pprof/heap > heap.prof

# 3. Analyze heap dump
go tool pprof heap.prof
(pprof) top
(pprof) list <function>

# 4. Check memory limits
kubectl get pods -o json | \
    jq -r '.items[] | "\(.metadata.name) Memory: \(.spec.containers[0].resources.limits.memory)"'
```

**Solutions**:
```bash
# Increase memory limits
kubectl set resources deployment/dwcp-api \
    --limits=memory=4Gi \
    --requests=memory=2Gi

# Restart pods to clear leaked memory
kubectl rollout restart deployment/dwcp-api

# Enable GC tuning (Go)
kubectl set env deployment/dwcp-api GOGC=50
```

---

### Problem 5: SSL Certificate Errors

**Symptoms**:
- "Certificate expired" errors
- "Certificate not trusted" errors
- TLS handshake failures

**Diagnostic Steps**:
```bash
# 1. Check certificate expiration
echo | openssl s_client -connect api.dwcp.io:443 -servername api.dwcp.io 2>/dev/null | \
    openssl x509 -noout -dates

# 2. Check cert-manager status
kubectl get certificates --all-namespaces
kubectl get certificaterequests --all-namespaces

# 3. Check cert-manager logs
kubectl logs -n cert-manager deployment/cert-manager

# 4. Verify certificate chain
echo | openssl s_client -connect api.dwcp.io:443 -showcerts
```

**Solutions**:
```bash
# Force certificate renewal
kubectl delete secret tls-cert -n production
kubectl annotate certificate api-tls -n production \
    cert-manager.io/issue-temporary-certificate="true"

# Check renewal
kubectl describe certificate api-tls -n production

# Verify new certificate
curl -vI https://api.dwcp.io
```

---

## Performance Optimization Guide

### Optimization Checklist

**Application Level**:
- [ ] Remove n+1 query problems
- [ ] Implement caching strategy
- [ ] Use connection pooling
- [ ] Enable compression
- [ ] Optimize JSON serialization
- [ ] Use async processing for slow operations
- [ ] Implement batch processing
- [ ] Add indexes to frequently queried fields

**Database Level**:
- [ ] Create appropriate indexes
- [ ] Optimize slow queries
- [ ] Use read replicas
- [ ] Implement query result caching
- [ ] Partition large tables
- [ ] Archive old data
- [ ] Tune database parameters
- [ ] Regular VACUUM ANALYZE

**Infrastructure Level**:
- [ ] Use CDN for static assets
- [ ] Enable HTTP/2
- [ ] Optimize Docker images
- [ ] Use HPA for auto-scaling
- [ ] Implement request/response compression
- [ ] Use persistent connections
- [ ] Optimize network topology

### Performance Testing

```bash
#!/bin/bash
# Performance test script

# 1. Load test with k6
k6 run --vus 100 --duration 5m performance-test.js

# 2. Database performance test
pgbench -h postgres-primary -U app_user -d app_db -c 50 -j 4 -T 300

# 3. Cache hit rate test
redis-cli --stat

# 4. API endpoint latency test
for i in {1..1000}; do
    curl -w "@curl-format.txt" -o /dev/null -s https://api.dwcp.io/health
done | awk '{sum+=$1; count++} END {print "Average:", sum/count}'
```

---

## Deployment FAQ

**Q: How long does a typical deployment take?**
A: 
- Canary (5% → 100%): 15-20 minutes
- Blue-Green: 10-15 minutes
- Rolling update: 5-10 minutes

**Q: Can I rollback a deployment?**
A: Yes, always:
```bash
kubectl rollout undo deployment/dwcp-api
```

**Q: How do I deploy to a specific environment?**
A:
```bash
# Development
./deploy.sh --env dev --version v1.2.3

# Staging
./deploy.sh --env staging --version v1.2.3

# Production (requires approval)
./deploy.sh --env production --version v1.2.3
```

**Q: What happens if deployment fails?**
A:
- Automatic rollback if error rate >5%
- Manual rollback available anytime
- No user impact due to rolling deployment

**Q: How do I check deployment status?**
A:
```bash
kubectl rollout status deployment/dwcp-api
kubectl get deployments
kubectl describe deployment dwcp-api
```

---

## Monitoring and Alerting FAQ

**Q: Where are the main dashboards?**
A:
- System Health: https://grafana.dwcp.io/d/system-health
- Application Metrics: https://grafana.dwcp.io/d/app-metrics
- Database: https://grafana.dwcp.io/d/database
- Infrastructure: https://grafana.dwcp.io/d/infrastructure

**Q: How do I create a new dashboard?**
A:
1. Use dashboard template in Grafana
2. Add panels with PromQL queries
3. Set appropriate thresholds
4. Export as JSON
5. Commit to `/dashboards/` directory
6. Apply via GitOps

**Q: How do I set up alerts?**
A:
```yaml
# Create alert rule
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: my-alert
spec:
  groups:
  - name: my-alerts
    rules:
    - alert: HighLatency
      expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
      for: 5m
      annotations:
        summary: "High API latency detected"
```

**Q: Why am I not receiving alerts?**
A: Check:
1. Alert rule syntax
2. AlertManager configuration
3. Notification channels (Slack, PagerDuty)
4. Alert silence/inhibition rules

---

## Database FAQ

**Q: How do I run a migration?**
A:
```bash
# 1. Create migration file
./migrate create -ext sql -dir migrations -seq add_users_table

# 2. Test on development
./migrate -path migrations -database "postgresql://localhost/dev" up

# 3. Apply to production (during maintenance window)
./migrate -path migrations -database "postgresql://prod/app" up
```

**Q: How do I create a backup?**
A:
```bash
# Automated daily backups run at 2am UTC
# Manual backup:
pg_dump -h postgres-primary -U app_user app_db | \
    gzip > backup-$(date +%Y%m%d).sql.gz
```

**Q: How do I restore from backup?**
A:
```bash
# 1. Download backup from S3
aws s3 cp s3://backups/20250110.sql.gz .

# 2. Restore
gunzip < 20250110.sql.gz | \
    psql -h postgres-primary -U app_user app_db
```

---

## Security FAQ

**Q: How do I rotate secrets?**
A: See `/docs/runbooks/SECRET_ROTATION.md`
```bash
./scripts/rotate-secret.sh --secret <name>
```

**Q: How do I report a security vulnerability?**
A:
1. Email: security@dwcp.io
2. PGP Key: https://keybase.io/dwcp_security
3. DO NOT create public GitHub issues
4. Response SLA: 24 hours

**Q: How do I check for vulnerabilities?**
A:
```bash
# Scan Docker images
trivy image dwcp/api:latest

# Scan dependencies
npm audit
snyk test

# Scan Kubernetes manifests
kube-score deployment.yaml
```

---

## Incident Response FAQ

**Q: How do I declare an incident?**
A:
```bash
./scripts/declare-incident.sh --severity SEV-1 --title "API outage"
```
Or use Slack: `/incident declare`

**Q: Who should I notify?**
A:
- **SEV-1**: Page on-call + notify in #incident-response
- **SEV-2**: Post in #incident-response
- **SEV-3/4**: Create ticket

**Q: How do I update stakeholders?**
A: Post updates every 5-15 minutes in incident channel:
```
UPDATE [HH:MM]
Status: Investigating/Mitigating/Resolved
Progress: [what's been done]
Next: [what will be done]
ETA: [if known]
```

**Q: When do I need a post-incident review?**
A: Required for all SEV-1 and SEV-2 incidents, recommended for SEV-3

---

**Document Control**:
- **Version**: 1.0.0
- **Last Updated**: 2025-11-10
- **Next Review**: Monthly
- **Owner**: SRE Team

*This is a living document. Contribute via PR to `/docs/phase6/PRODUCTION_KNOWLEDGE_BASE.md`*
