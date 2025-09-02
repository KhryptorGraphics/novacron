# BMad Task 9: Troubleshooting Guide - NovaCron Platform Operations

## Operational Troubleshooting Guide: NovaCron Platform
**Target Audience**: Operations teams, SRE engineers, Support staff  
**System Coverage**: 600+ Go services, 38K+ TypeScript files  
**SLA Context**: 99.95% uptime, <1s response time requirements  
**Last Updated**: September 2025  

---

## Quick Reference Emergency Contacts

### Escalation Matrix
| Severity | Response Time | Primary Contact | Secondary Contact |
|----------|--------------|-----------------|-------------------|
| **Critical** | 15 minutes | On-call engineer | Engineering lead |
| **High** | 2 hours | Platform team | DevOps engineer |
| **Medium** | 8 hours | Support team | Product owner |
| **Low** | 24 hours | Support queue | Documentation team |

### Service Status Dashboard
- **Primary**: https://status.novacron.io
- **Internal**: https://monitoring.novacron.internal/grafana
- **Alerting**: https://alerts.novacron.internal/alertmanager

---

## System Architecture Quick Reference

### Core Services and Ports
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  VM Management  │    │  Orchestration  │
│   Port 8080     │◄──►│   Port 8081     │◄──►│   Port 8082     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Federation    │    │   ML Service    │    │  Backup Service │
│   Port 8084     │    │   Port 8083     │    │   Port 8085     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Monitoring    │
│   Port 9090     │
└─────────────────┘
```

### Health Check Endpoints
```bash
# Core service health checks
curl -f http://localhost:8080/health      # API Gateway
curl -f http://localhost:8081/health      # VM Management
curl -f http://localhost:8082/health      # Orchestration
curl -f http://localhost:8083/health      # ML Service
curl -f http://localhost:8084/health      # Federation
curl -f http://localhost:8085/health      # Backup Service
curl -f http://localhost:9090/health      # Monitoring
```

---

## Common Issues and Solutions

### 1. Response Time SLA Violations (P95 > 1s)

#### Symptoms
- Dashboard showing response times above 1000ms
- User complaints about slow dashboard loading
- Alert: `ResponseTimeSLAViolation` firing

#### Immediate Actions (5 minutes)
```bash
# Check current system load
kubectl top nodes
kubectl top pods -n novacron

# Identify bottleneck services
curl -s http://localhost:9090/api/v1/query?query='histogram_quantile(0.95,sum(rate(http_request_duration_seconds_bucket[5m]))by(service,le))'

# Check database connection pools
psql -h db.novacron.internal -c "SELECT pid, state, query_start, query FROM pg_stat_activity WHERE state != 'idle';"

# Redis cache performance
redis-cli --latency-history -i 1
```

#### Root Cause Analysis
```bash
# Database query performance
psql -h db.novacron.internal -c "SELECT query, mean_exec_time, calls FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# Memory pressure indicators
free -h
kubectl describe nodes | grep -A 5 "Allocated resources"

# Network latency between services
for service in vm-service orchestration-service ml-service; do
  kubectl exec -n novacron deployment/api-gateway -- wget -qO- --timeout=1 http://$service:8080/health
done
```

#### Resolution Steps
```bash
# Scale up high-load services
kubectl scale deployment/vm-service --replicas=6 -n novacron
kubectl scale deployment/api-gateway --replicas=4 -n novacron

# Clear application caches if stale
redis-cli FLUSHDB
kubectl rollout restart deployment/vm-service -n novacron

# Database optimization (if query performance is poor)
psql -h db.novacron.internal -c "ANALYZE;"
psql -h db.novacron.internal -c "VACUUM ANALYZE;"
```

### 2. VM Operation Failures

#### Symptoms
- VM creation/deletion operations failing
- Error rate above 0.1% threshold
- Alert: `VMOperationFailureHigh` firing

#### Immediate Diagnostics
```bash
# Check VM service logs
kubectl logs -f deployment/vm-service -n novacron --tail=50

# Provider-specific health checks
curl -s http://localhost:8081/api/v1/providers/aws/health
curl -s http://localhost:8081/api/v1/providers/azure/health  
curl -s http://localhost:8081/api/v1/providers/gcp/health

# Check cloud provider credentials
kubectl get secret aws-credentials -n novacron -o yaml
kubectl get secret azure-credentials -n novacron -o yaml
kubectl get secret gcp-credentials -n novacron -o yaml
```

#### Common Resolution Patterns
```bash
# Credential refresh (if authentication errors)
./scripts/rotate-cloud-credentials.sh

# Provider-specific troubleshooting
case $PROVIDER in
  "aws")
    aws ec2 describe-instances --region us-east-1 --max-items 1
    aws sts get-caller-identity
    ;;
  "azure")
    az vm list --resource-group novacron-rg --query "[0]"
    az account show
    ;;
  "gcp")
    gcloud compute instances list --limit=1
    gcloud auth list
    ;;
esac

# Reset connection pools
kubectl rollout restart deployment/vm-service -n novacron
```

### 3. Federation Consensus Failures

#### Symptoms  
- Cross-cluster operations failing
- Split-brain scenarios detected
- Alert: `FederationConsensusFailure` firing

#### Immediate Assessment
```bash
# Check cluster health across all regions
for cluster in us-east us-west eu-central; do
  echo "=== $cluster ==="
  kubectl --context=$cluster get nodes
  kubectl --context=$cluster get pods -n novacron | grep federation
done

# Raft consensus status
kubectl exec -n novacron deployment/federation-service -- \
  curl -s http://localhost:8084/api/v1/raft/status

# Network connectivity between clusters
kubectl exec -n novacron deployment/federation-service -- \
  nc -zv federation-service.eu-central.novacron.internal 8084
```

#### Resolution Procedures
```bash
# Force new leader election (CAUTION)
kubectl exec -n novacron deployment/federation-service -- \
  curl -X POST http://localhost:8084/api/v1/raft/leadership-transfer

# Restart federation service in specific order
kubectl rollout restart deployment/federation-service -n novacron --context=us-east
sleep 30
kubectl rollout restart deployment/federation-service -n novacron --context=us-west  
sleep 30
kubectl rollout restart deployment/federation-service -n novacron --context=eu-central

# Validate consensus restoration
kubectl exec -n novacron deployment/federation-service -- \
  curl -s http://localhost:8084/api/v1/raft/peers
```

### 4. Database Connection Issues

#### Symptoms
- `database connection failed` errors in logs
- Connection pool exhaustion alerts
- Unable to query VM status

#### Immediate Actions
```bash
# Check PostgreSQL status
systemctl status postgresql
pg_isready -h db.novacron.internal

# Connection pool status
psql -h db.novacron.internal -c "SELECT count(*) FROM pg_stat_activity;"
psql -h db.novacron.internal -c "SELECT state, count(*) FROM pg_stat_activity GROUP BY state;"

# Check connection limits
psql -h db.novacron.internal -c "SHOW max_connections;"
psql -h db.novacron.internal -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"
```

#### Resolution Steps
```bash
# Restart services with high connection usage
kubectl rollout restart deployment/vm-service -n novacron
kubectl rollout restart deployment/orchestration-service -n novacron

# Database maintenance (if needed)
psql -h db.novacron.internal -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'idle' AND state_change < now() - interval '5 minutes';"

# Scale connection pooling
kubectl scale deployment/pgbouncer --replicas=3 -n database
```

### 5. Frontend Dashboard Issues

#### Symptoms
- Dashboard not loading or showing errors
- WebSocket connection failures
- Real-time updates not working

#### Diagnostics
```bash
# Check frontend service status
kubectl get pods -n novacron -l app=novacron-frontend
kubectl logs -f deployment/novacron-frontend -n novacron

# API Gateway connectivity
curl -s http://api-gateway.novacron.internal:8080/api/v1/health

# WebSocket endpoint testing
wscat -c ws://api-gateway.novacron.internal:8080/ws/vms/tenant-123
```

#### Resolution
```bash
# Restart frontend services
kubectl rollout restart deployment/novacron-frontend -n novacron

# Clear CDN cache (if using external CDN)
curl -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/purge_cache" \
  -H "Authorization: Bearer $CLOUDFLARE_TOKEN" \
  -H "Content-Type: application/json" \
  --data '{"purge_everything":true}'

# Check browser console for client-side errors
echo "Ask user to check browser developer tools console"
```

---

## Performance Troubleshooting

### Memory Leak Detection

#### Monitoring Memory Usage
```bash
# Monitor memory consumption trends
kubectl top pods -n novacron --sort-by=memory
kubectl describe nodes | grep -A 10 "Allocated resources"

# Memory profiling for Go services
kubectl exec -n novacron deployment/vm-service -- \
  curl -s http://localhost:6060/debug/pprof/heap > heap.prof
  
go tool pprof heap.prof
```

#### Memory Leak Resolution
```bash
# Force garbage collection
kubectl exec -n novacron deployment/vm-service -- \
  curl -X POST http://localhost:6060/debug/gc

# Rolling restart to clear memory
kubectl rollout restart deployment/vm-service -n novacron
kubectl rollout status deployment/vm-service -n novacron
```

### CPU Performance Issues

#### CPU Utilization Analysis
```bash
# CPU hotspot identification
kubectl exec -n novacron deployment/vm-service -- \
  curl -s http://localhost:6060/debug/pprof/profile?seconds=30 > cpu.prof

go tool pprof cpu.prof

# Check CPU throttling
kubectl describe pods -n novacron | grep -A 5 "Limits\|Requests"
```

### Network Latency Issues

#### Network Diagnostics
```bash
# Inter-service latency testing
kubectl exec -n novacron deployment/api-gateway -- \
  time curl -s http://vm-service:8081/health

# DNS resolution timing
kubectl exec -n novacron deployment/api-gateway -- \
  nslookup vm-service.novacron.svc.cluster.local

# Packet loss detection
kubectl exec -n novacron deployment/api-gateway -- \
  ping -c 10 vm-service.novacron.svc.cluster.local
```

---

## Security Incident Response

### Authentication Failures

#### Symptoms
- High number of authentication failures
- JWT validation errors
- Alert: `AuthenticationFailureSpike` firing

#### Investigation Steps
```bash
# Check authentication logs
kubectl logs -f deployment/api-gateway -n novacron | grep "auth_failure"

# JWT token validation
kubectl exec -n novacron deployment/api-gateway -- \
  curl -s http://localhost:8080/debug/jwt/validate -H "Authorization: Bearer $SUSPECT_TOKEN"

# Rate limiting status
redis-cli GET "rate_limit:user:$USER_ID"
```

#### Response Actions
```bash
# Block suspicious IP addresses
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: block-malicious-ips
  namespace: novacron
spec:
  podSelector:
    matchLabels:
      app: api-gateway
  policyTypes:
  - Ingress
  ingress:
  - from:
    - namespaceSelector: {}
    except:
    - namespaceSelector:
        matchLabels:
          name: blocked-ips
EOF

# Rotate JWT signing keys (if compromise suspected)
./scripts/rotate-jwt-keys.sh

# Force user session invalidation
redis-cli DEL "session:user:$COMPROMISED_USER_ID"
```

### Data Breach Detection

#### Investigation Protocol
```bash
# Audit log analysis
kubectl exec -n novacron deployment/api-gateway -- \
  grep -E "(large_data_export|bulk_vm_access|privilege_escalation)" /var/log/audit.log

# Database access patterns
psql -h db.novacron.internal -c "
  SELECT query, calls, mean_exec_time, rows 
  FROM pg_stat_statements 
  WHERE query LIKE '%SELECT%' AND rows > 1000 
  ORDER BY calls DESC;"

# Unusual API access patterns  
curl -s "http://localhost:9090/api/v1/query?query=rate(http_requests_total[5m]) > 100"
```

---

## Monitoring and Alerting Troubleshooting

### Prometheus Issues

#### Metrics Collection Problems
```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'

# Storage usage
du -sh /var/lib/prometheus/data/
df -h /var/lib/prometheus/

# Query performance
curl -s "http://localhost:9090/api/v1/query?query=prometheus_tsdb_symbol_table_size_bytes"
```

#### Grafana Dashboard Issues
```bash
# Check Grafana connectivity to Prometheus
curl -f http://grafana.novacron.internal:3000/api/datasources/proxy/1/api/v1/query?query=up

# Dashboard rendering issues
tail -f /var/log/grafana/grafana.log | grep ERROR

# Reset Grafana dashboard cache
systemctl restart grafana-server
```

### Alert Manager Configuration

#### Alert Routing Problems
```bash
# Check alert manager configuration
curl -s http://localhost:9093/api/v1/status | jq .

# Verify alert routing
curl -s http://localhost:9093/api/v1/alerts | jq '.data[] | select(.status.state == "active")'

# Test notification channels
amtool config routes test --config.file=/etc/alertmanager/alertmanager.yml
```

---

## Disaster Recovery Procedures

### Complete System Failure

#### Recovery Priority Order
1. **Database restoration** (RTO: 15 minutes)
2. **Core services** (VM, Orchestration, API Gateway)
3. **Supporting services** (ML, Federation, Backup)
4. **Frontend and monitoring**

#### Database Recovery
```bash
# Restore from latest backup
pg_restore -h db.novacron.internal -U postgres -d novacron /backups/novacron-$(date +%Y%m%d).sql

# Verify data integrity
psql -h db.novacron.internal -c "SELECT COUNT(*) FROM vms WHERE status IN ('running', 'stopped');"

# Re-create indexes if needed
psql -h db.novacron.internal -c "REINDEX DATABASE novacron;"
```

#### Service Recovery
```bash
# Deploy in dependency order
kubectl apply -f k8s/database/
kubectl wait --for=condition=ready pod -l app=postgresql -n database --timeout=300s

kubectl apply -f k8s/core-services/
kubectl wait --for=condition=ready pod -l app=vm-service -n novacron --timeout=300s

kubectl apply -f k8s/supporting-services/
kubectl apply -f k8s/frontend/
```

### Cross-Region Failover

#### Failover Checklist
```bash
# 1. Assess primary region status
kubectl --context=us-east get nodes
kubectl --context=us-east get pods -n novacron

# 2. Activate secondary region
kubectl --context=us-west scale deployment/api-gateway --replicas=6 -n novacron
kubectl --context=us-west scale deployment/vm-service --replicas=4 -n novacron

# 3. Update DNS routing
# Point novacron.com to us-west load balancer IP
dig novacron.com

# 4. Database failover (if needed)
pg_ctl promote -D /var/lib/postgresql/data/

# 5. Verify service health
curl -f https://novacron.com/api/v1/health
```

---

## Capacity Planning and Scaling

### Auto-scaling Troubleshooting

#### HPA Issues
```bash
# Check HPA status
kubectl get hpa -n novacron
kubectl describe hpa novacron-vm-service-hpa -n novacron

# Metrics server connectivity
kubectl top nodes
kubectl top pods -n novacron

# Custom metrics availability
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/novacron/pods/*/requests_per_second"
```

#### Manual Scaling Decisions
```bash
# Current resource utilization
kubectl top pods -n novacron --sort-by=cpu
kubectl top pods -n novacron --sort-by=memory

# Scaling recommendation based on current load
CURRENT_RPS=$(curl -s "http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total[5m]))" | jq -r '.data.result[0].value[1]')
echo "Current RPS: $CURRENT_RPS"
echo "Recommended replicas: $(echo "$CURRENT_RPS / 100" | bc)"
```

---

## Runbook Templates

### Incident Response Template

#### 1. Initial Assessment (0-5 minutes)
```bash
# Severity assessment
SEVERITY="[P1/P2/P3/P4]"
IMPACT="[Customer impact description]"
AFFECTED_SERVICES="[Service list]"

# Initial status update
echo "INCIDENT: $SEVERITY - $IMPACT affecting $AFFECTED_SERVICES at $(date)" | tee incident-$(date +%s).log
```

#### 2. Investigation (5-15 minutes)
```bash
# System health check
./scripts/health-check-all.sh | tee -a incident-$(date +%s).log

# Identify root cause
# - Check recent deployments
# - Review error logs
# - Analyze metrics
```

#### 3. Resolution (15-45 minutes)
```bash
# Document all actions taken
echo "ACTIONS TAKEN:" | tee -a incident-$(date +%s).log
echo "- [timestamp] [action description]" | tee -a incident-$(date +%s).log

# Verify resolution
./scripts/smoke-test.sh | tee -a incident-$(date +%s).log
```

#### 4. Post-Incident (24 hours)
```bash
# Generate incident report
./scripts/generate-incident-report.sh incident-$(date +%s).log

# Schedule post-mortem
echo "Post-mortem scheduled for: [date/time]"
echo "Attendees: [list]"
echo "Root cause: [description]"
echo "Action items: [list]"
```

---

## Useful Scripts and Commands

### Health Check Script
```bash
#!/bin/bash
# health-check-all.sh

echo "=== NovaCron System Health Check ==="
echo "Timestamp: $(date)"
echo

# Service health checks
services=("api-gateway:8080" "vm-service:8081" "orchestration-service:8082" 
          "ml-service:8083" "federation-service:8084" "backup-service:8085")

for service in "${services[@]}"; do
  name=${service%:*}
  port=${service#*:}
  
  if kubectl exec -n novacron deployment/$name -- curl -f http://localhost:$port/health >/dev/null 2>&1; then
    echo "✅ $name is healthy"
  else
    echo "❌ $name is unhealthy"
  fi
done

# Database connectivity
if pg_isready -h db.novacron.internal >/dev/null 2>&1; then
  echo "✅ Database is accessible"
else
  echo "❌ Database is inaccessible"
fi

# Redis connectivity  
if redis-cli -h redis.novacron.internal ping >/dev/null 2>&1; then
  echo "✅ Redis is responsive"
else
  echo "❌ Redis is unresponsive"
fi

echo
echo "=== Resource Utilization ==="
kubectl top nodes | head -5
kubectl top pods -n novacron --sort-by=cpu | head -10
```

### Performance Monitoring Script
```bash
#!/bin/bash
# performance-monitor.sh

echo "=== Performance Metrics ==="
echo "Timestamp: $(date)"
echo

# Response time metrics
echo "Response Times (P95):"
curl -s "http://localhost:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(http_request_duration_seconds_bucket[5m]))by(service,le))" | \
jq -r '.data.result[] | "\(.metric.service): \(.value[1])s"'

echo
echo "Error Rates:"
curl -s "http://localhost:9090/api/v1/query?query=sum(rate(http_requests_total{status=~\"5..\"}[5m]))by(service)" | \
jq -r '.data.result[] | "\(.metric.service): \(.value[1])"'

echo
echo "Database Connections:"
psql -h db.novacron.internal -t -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"
```

---

## Contact Information and Escalation

### Emergency Contacts
- **On-Call Engineer**: +1-555-ONCALL (24/7)
- **Engineering Manager**: engineering-manager@novacron.com
- **Site Reliability**: sre-team@novacron.com
- **Security Team**: security@novacron.com

### Vendor Support Contacts
- **AWS Support**: Case creation via AWS Console
- **Azure Support**: Case creation via Azure Portal
- **GCP Support**: Case creation via Google Cloud Console
- **Kubernetes**: Community Slack #kubernetes-users

### Internal Tools
- **Status Page**: https://status.novacron.io
- **Runbook Repository**: https://github.com/novacron/runbooks
- **Monitoring Dashboard**: https://monitoring.novacron.internal
- **Log Aggregation**: https://logs.novacron.internal

---

*Troubleshooting guide maintained by NovaCron SRE Team - Last updated September 2025*