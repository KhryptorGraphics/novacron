# NovaCron Operations Manual

## Overview

This operations manual provides comprehensive procedures for managing, monitoring, and troubleshooting the NovaCron platform in production environments.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Monitoring & Alerting](#monitoring--alerting)
3. [Incident Response](#incident-response)
4. [Maintenance Procedures](#maintenance-procedures)
5. [Backup & Recovery](#backup--recovery)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)
8. [Security Operations](#security-operations)

## Daily Operations

### Morning Health Checks

**Automated Health Check:**
```bash
# Run comprehensive health validation
./scripts/health-check.sh

# Check service status
docker-compose ps
kubectl get pods -n novacron-prod

# Review overnight alerts
curl -s http://prometheus:9090/api/v1/alerts | jq '.data.alerts[] | select(.state=="firing")'
```

**Manual Checks:**
- [ ] All services running and healthy
- [ ] No critical alerts firing
- [ ] System resources within normal ranges
- [ ] Backup completion status
- [ ] SSL certificate validity
- [ ] Log aggregation functioning

### Service Status Monitoring

**Key Metrics to Monitor:**
- API response time (<200ms average)
- Error rate (<0.1%)
- System uptime (>99.9%)
- Database connections (<80% pool utilization)
- Memory usage (<85%)
- CPU usage (<70% average)
- Disk space (<80% utilization)

**Service Endpoints:**
```bash
# Health endpoints
curl http://localhost:8090/health          # API health
curl http://localhost:8092/health          # Frontend health
curl http://localhost:9090/-/healthy       # Prometheus health
curl http://localhost:3000/api/health      # Grafana health
```

## Monitoring & Alerting

### Prometheus Metrics

**Critical Metrics:**
```promql
# Service availability
up{job="novacron-api"}

# API performance
rate(http_requests_total[5m])
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rates
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# System resources
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100
rate(cpu_seconds_total{mode="idle"}[5m]) * 100
```

### Grafana Dashboards

**System Overview Dashboard:**
- Service health status matrix
- Resource utilization trends
- Alert status panel
- Performance metrics

**Application Dashboard:**
- API endpoint performance
- Database query analysis
- User activity metrics
- VM operation statistics

### Alert Configuration

**Critical Alerts (Immediate Response):**
- Service downtime
- High error rate (>5%)
- Database connectivity issues
- SSL certificate expiration
- Disk space critical (>90%)

**Warning Alerts (Response within 1 hour):**
- High response times (>2s)
- Elevated resource usage (>80%)
- Authentication failures
- Backup issues

### Alert Escalation

**Level 1 - On-call Engineer:**
- Initial response within 5 minutes
- Triage and basic troubleshooting
- Escalate if needed

**Level 2 - Senior Engineer:**
- Complex technical issues
- System architecture problems
- Performance optimization

**Level 3 - Management:**
- Business impact assessment
- External communication
- Vendor escalation

## Incident Response

### Incident Classification

**P1 - Critical (Response: Immediate)**
- Complete service outage
- Data loss or corruption
- Security breach
- Customer-facing functionality down

**P2 - High (Response: 30 minutes)**
- Partial service degradation
- Performance issues affecting users
- Non-critical security vulnerabilities

**P3 - Medium (Response: 2 hours)**
- Minor service issues
- Individual user problems
- Cosmetic issues

**P4 - Low (Response: Next business day)**
- Documentation updates
- Enhancement requests
- Non-urgent maintenance

### Incident Response Procedure

**1. Detection and Alerting**
```bash
# Check system status
./scripts/health-check.sh

# Review active alerts
kubectl get events -n novacron-prod --sort-by='.lastTimestamp'
docker-compose logs --tail=100
```

**2. Initial Assessment**
- Determine scope and impact
- Classify incident priority
- Identify affected services/users
- Estimate resolution time

**3. Response Actions**
```bash
# Gather diagnostic information
kubectl describe pods -n novacron-prod
docker stats
journalctl -u docker.service --since "1 hour ago"

# Check resource usage
df -h
free -h
top
```

**4. Communication**
- Notify stakeholders immediately
- Update status page
- Provide regular updates (every 30 minutes)
- Post-incident summary

**5. Resolution and Recovery**
```bash
# Service restart
docker-compose restart api
kubectl rollout restart deployment/novacron-api -n novacron-prod

# Configuration rollback
git checkout HEAD~1 deployment/
./scripts/deploy-prod.sh
```

### Emergency Procedures

**Complete System Outage:**
```bash
# Emergency restart
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d

# Kubernetes emergency restart
kubectl delete pods --all -n novacron-prod
```

**Database Emergency:**
```bash
# Switch to read replica
kubectl patch service postgres-service -p '{"spec":{"selector":{"role":"replica"}}}'

# Emergency backup
pg_dump $DATABASE_URL > emergency-backup-$(date +%Y%m%d-%H%M).sql
```

**Security Incident:**
```bash
# Isolate compromised services
docker network disconnect novacron-prod compromised-container

# Revoke authentication tokens
kubectl delete secret novacron-secrets -n novacron-prod
kubectl create secret generic novacron-secrets --from-literal=auth-secret="$(openssl rand -base64 64)"
```

## Maintenance Procedures

### Scheduled Maintenance

**Weekly Maintenance Window (Sunday 02:00-04:00 UTC):**
- System updates and patches
- Database maintenance
- Log rotation and cleanup
- Performance optimization
- Backup validation

**Monthly Maintenance:**
- Security audit
- Capacity planning review
- Disaster recovery testing
- Documentation updates

### Update Procedures

**Application Updates:**
```bash
# 1. Backup current state
./scripts/backup-deployment.sh

# 2. Update images
docker pull novacron/api:latest
docker pull novacron/frontend:latest

# 3. Rolling update
docker-compose -f docker-compose.prod.yml up -d --no-deps api
docker-compose -f docker-compose.prod.yml up -d --no-deps frontend

# 4. Verify deployment
./scripts/health-check.sh
```

**System Updates:**
```bash
# 1. Update base system
apt update && apt upgrade -y

# 2. Update Docker
apt install docker-ce docker-ce-cli containerd.io

# 3. Update Kubernetes
kubectl drain $NODE_NAME --ignore-daemonsets
apt update && apt install -y kubelet kubeadm kubectl
kubectl uncordon $NODE_NAME
```

### Database Maintenance

**Regular Maintenance:**
```sql
-- Analyze table statistics
ANALYZE;

-- Reindex if needed
REINDEX INDEX CONCURRENTLY idx_vms_owner_id;

-- Clean up old data
DELETE FROM vm_metrics WHERE timestamp < NOW() - INTERVAL '90 days';

-- Vacuum tables
VACUUM ANALYZE vm_metrics;
```

**Performance Tuning:**
```bash
# Monitor slow queries
docker exec postgres psql -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Connection monitoring
docker exec postgres psql -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
```

## Backup & Recovery

### Backup Strategy

**Automated Daily Backups:**
```bash
#!/bin/bash
# Database backup
pg_dump $DATABASE_URL | gzip > backup-$(date +%Y%m%d).sql.gz

# Configuration backup
tar -czf config-backup-$(date +%Y%m%d).tar.gz deployment/

# Upload to cloud storage
aws s3 cp backup-$(date +%Y%m%d).sql.gz s3://backups/novacron/database/
aws s3 cp config-backup-$(date +%Y%m%d).tar.gz s3://backups/novacron/config/
```

**Recovery Procedures:**
```bash
# Database recovery
gunzip -c backup-20250101.sql.gz | psql $DATABASE_URL

# Configuration recovery
tar -xzf config-backup-20250101.tar.gz
./scripts/deploy-prod.sh
```

### Disaster Recovery

**RTO (Recovery Time Objective): 4 hours**
**RPO (Recovery Point Objective): 1 hour**

**Full System Recovery:**
```bash
# 1. Provision new infrastructure
terraform apply -var="environment=disaster-recovery"

# 2. Restore database
pg_restore -h new-db-host -U postgres -d novacron latest-backup.sql

# 3. Deploy application
./scripts/deploy-prod.sh kubernetes

# 4. Update DNS
# Point domain to new infrastructure

# 5. Verify functionality
./scripts/health-check.sh
```

## Troubleshooting

### Common Issues

**High Response Times:**
```bash
# Check system resources
htop
iostat -x 1

# Analyze slow queries
docker exec postgres psql -c "SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Check connection pools
docker exec postgres psql -c "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"
```

**Memory Issues:**
```bash
# Check memory usage
free -h
docker stats

# Identify memory-consuming processes
ps aux --sort=-%mem | head -10

# Check for memory leaks
valgrind --tool=memcheck --leak-check=full ./api-server
```

**Network Connectivity:**
```bash
# Check service connectivity
telnet api-service 8090
nslookup api-service

# Check network policies (Kubernetes)
kubectl describe networkpolicy -n novacron-prod

# Test DNS resolution
dig api.yourdomain.com
```

**Database Connection Issues:**
```bash
# Test direct connection
psql $DATABASE_URL -c "SELECT 1;"

# Check connection limits
docker exec postgres psql -c "SHOW max_connections;"
docker exec postgres psql -c "SELECT count(*) FROM pg_stat_activity;"

# Monitor connection pool
docker logs pgbouncer | grep -i "pool"
```

### Log Analysis

**Application Logs:**
```bash
# API logs
docker-compose logs -f api
kubectl logs -f deployment/novacron-api -n novacron-prod

# Database logs
docker-compose logs postgres
tail -f /var/log/postgresql/postgresql.log
```

**System Logs:**
```bash
# System events
journalctl -f
dmesg -w

# Docker events
docker events

# Kubernetes events
kubectl get events -n novacron-prod --sort-by='.lastTimestamp'
```

### Performance Diagnostics

**API Performance:**
```bash
# Load testing
k6 run load-test.js

# Profile API endpoints
curl -w "@curl-format.txt" -o /dev/null -s "http://api:8090/api/vm/vms"

# Memory profiling
curl http://api:8090/debug/pprof/heap > heap.prof
go tool pprof heap.prof
```

**Database Performance:**
```sql
-- Slow query analysis
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 20;

-- Index usage
SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch 
FROM pg_stat_user_indexes 
ORDER BY idx_tup_read DESC;

-- Table statistics
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del, n_live_tup, n_dead_tup 
FROM pg_stat_user_tables 
ORDER BY n_live_tup DESC;
```

## Performance Tuning

### Application Optimization

**Go Application Tuning:**
```bash
# Environment variables
export GOMAXPROCS=4
export GOMEMLIMIT=1536MiB
export GOGC=100

# Connection pool tuning
DB_MAX_OPEN_CONNS=25
DB_MAX_IDLE_CONNS=5
DB_CONN_MAX_LIFETIME=5m
```

**Database Optimization:**
```postgresql
-- PostgreSQL tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET work_mem = '4MB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
SELECT pg_reload_conf();
```

**Redis Optimization:**
```bash
# Redis tuning
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save "900 1 300 10"
redis-cli CONFIG SET appendfsync everysec
```

### Scaling Procedures

**Horizontal Scaling:**
```bash
# Docker Compose scaling
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Kubernetes scaling
kubectl scale deployment novacron-api --replicas=5 -n novacron-prod

# Auto-scaling configuration
kubectl autoscale deployment novacron-api --cpu-percent=70 --min=3 --max=10 -n novacron-prod
```

**Vertical Scaling:**
```yaml
# Update resource limits
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"
```

## Security Operations

### Security Monitoring

**Log Analysis:**
```bash
# Authentication failures
grep "authentication failed" /var/log/auth.log

# Suspicious API activity
docker logs api 2>&1 | grep -E "(401|403|429)"

# Network intrusion detection
fail2ban-client status
```

**Vulnerability Scanning:**
```bash
# Container scanning
docker scan novacron/api:latest

# Dependency scanning
npm audit
go mod audit
```

### Incident Response

**Security Breach Response:**
```bash
# 1. Isolate affected systems
docker network disconnect novacron-prod compromised-container

# 2. Preserve evidence
docker logs compromised-container > security-incident-logs.txt
docker export compromised-container > compromised-container.tar

# 3. Reset credentials
kubectl delete secret novacron-secrets
# Generate new secrets

# 4. Audit system access
grep "sudo" /var/log/auth.log
grep "ssh" /var/log/auth.log

# 5. Apply security patches
apt update && apt upgrade -y
```

### Compliance

**Audit Logging:**
- All API requests logged
- Database access logging enabled
- System access tracking
- Configuration changes tracked

**Data Protection:**
- Encryption at rest and in transit
- Access controls implemented
- Data retention policies
- Regular security assessments

## Contact Information

### On-Call Escalation

**Primary On-Call:**
- Phone: +1-555-ON-CALL
- Email: oncall@company.com
- Slack: #oncall

**Secondary On-Call:**
- Phone: +1-555-BACKUP
- Email: backup@company.com

**Management Escalation:**
- CTO: cto@company.com
- VP Engineering: vpeng@company.com

### Vendor Contacts

**Cloud Provider Support:**
- AWS: enterprise-support@aws.com
- GCP: support@google.com

**Third-Party Services:**
- Monitoring: support@monitoring-vendor.com
- Database: support@database-vendor.com

---

**Document Version:** 1.0  
**Last Updated:** 2025-09-02  
**Next Review:** Monthly  
**Owner:** DevOps Team