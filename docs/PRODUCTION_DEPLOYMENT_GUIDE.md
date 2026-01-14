# NovaCron Production Deployment Guide

**Version**: 2.0.0  
**Status**: Production Ready  
**Date**: 2025-10-31

---

## 🎯 Overview

This guide provides step-by-step instructions for deploying NovaCron to production environments. Follow these procedures to ensure a smooth, secure, and reliable deployment.

---

## 📋 Pre-Deployment Checklist

### Infrastructure Requirements
- [ ] Kubernetes cluster (v1.28+) or Docker Swarm
- [ ] PostgreSQL 15+ (with replication)
- [ ] Redis 7+ (with clustering)
- [ ] Load balancer (AWS ALB, Azure LB, or NGINX)
- [ ] SSL/TLS certificates
- [ ] DNS configuration
- [ ] Monitoring infrastructure (Prometheus, Grafana)
- [ ] Log aggregation (ELK or similar)

### Security Requirements
- [ ] Firewall rules configured
- [ ] VPC/Network isolation
- [ ] Secrets management (Vault, AWS Secrets Manager)
- [ ] SSL/TLS certificates installed
- [ ] Security groups configured
- [ ] IAM roles and policies
- [ ] Backup encryption keys

### Capacity Planning
- [ ] CPU: 16+ cores per node
- [ ] Memory: 32GB+ per node
- [ ] Storage: 500GB+ SSD
- [ ] Network: 10Gbps+ bandwidth
- [ ] Database: 100GB+ with IOPS 3000+

---

## 🚀 Deployment Steps

### Step 1: Infrastructure Setup

#### 1.1 Kubernetes Cluster Setup

```bash
# Create namespace
kubectl create namespace novacron

# Create secrets
kubectl create secret generic novacron-secrets \
  --from-literal=db-password=<DB_PASSWORD> \
  --from-literal=redis-password=<REDIS_PASSWORD> \
  --from-literal=jwt-secret=<JWT_SECRET> \
  --from-literal=encryption-key=<ENCRYPTION_KEY> \
  -n novacron

# Create ConfigMap
kubectl create configmap novacron-config \
  --from-file=config/production.yaml \
  -n novacron
```

#### 1.2 Database Setup

```sql
-- Create database
CREATE DATABASE novacron_production;

-- Create user
CREATE USER novacron WITH ENCRYPTED PASSWORD '<PASSWORD>';

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE novacron_production TO novacron;

-- Enable extensions
\c novacron_production
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Run migrations
\i migrations/001_initial_schema.sql
\i migrations/002_add_multicloud.sql
\i migrations/003_add_edge.sql
\i migrations/004_add_security.sql
```

#### 1.3 Redis Setup

```bash
# Redis cluster configuration
redis-cli --cluster create \
  redis-1:6379 \
  redis-2:6379 \
  redis-3:6379 \
  --cluster-replicas 1

# Set password
redis-cli CONFIG SET requirepass <REDIS_PASSWORD>
```

---

### Step 2: Application Deployment

#### 2.1 Deploy Backend Services

```bash
# Apply Kubernetes manifests
kubectl apply -f deployment/kubernetes/backend-deployment.yaml
kubectl apply -f deployment/kubernetes/backend-service.yaml

# Verify deployment
kubectl get pods -n novacron
kubectl logs -f deployment/novacron-backend -n novacron
```

**backend-deployment.yaml**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-backend
  namespace: novacron
spec:
  replicas: 3
  selector:
    matchLabels:
      app: novacron-backend
  template:
    metadata:
      labels:
        app: novacron-backend
    spec:
      containers:
      - name: backend
        image: novacron/backend:2.0.0
        ports:
        - containerPort: 8080
        - containerPort: 9090
        env:
        - name: DB_HOST
          value: "postgres.novacron.svc.cluster.local"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: db-password
        - name: REDIS_HOST
          value: "redis.novacron.svc.cluster.local"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
```

#### 2.2 Deploy Frontend

```bash
# Build frontend
cd frontend
npm run build

# Deploy to CDN or Kubernetes
kubectl apply -f deployment/kubernetes/frontend-deployment.yaml
kubectl apply -f deployment/kubernetes/frontend-service.yaml
```

#### 2.3 Deploy Edge Agents

```bash
# Deploy edge agent to edge nodes
./scripts/deploy-edge-agent.sh \
  --node-id edge-node-1 \
  --location "us-east-1" \
  --cloud-endpoint "https://api.novacron.com"
```

---

### Step 3: Configure Load Balancer

#### 3.1 NGINX Configuration

```nginx
upstream novacron_backend {
    least_conn;
    server backend-1:8080 max_fails=3 fail_timeout=30s;
    server backend-2:8080 max_fails=3 fail_timeout=30s;
    server backend-3:8080 max_fails=3 fail_timeout=30s;
}

server {
    listen 443 ssl http2;
    server_name api.novacron.com;

    ssl_certificate /etc/ssl/certs/novacron.crt;
    ssl_certificate_key /etc/ssl/private/novacron.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://novacron_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /metrics {
        proxy_pass http://novacron_backend;
        allow 10.0.0.0/8;
        deny all;
    }
}
```

---

### Step 4: Monitoring Setup

#### 4.1 Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'novacron-backend'
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names:
            - novacron
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: novacron-backend
      - source_labels: [__meta_kubernetes_pod_ip]
        target_label: __address__
        replacement: ${1}:9090

  - job_name: 'novacron-edge'
    static_configs:
      - targets: ['edge-1:9090', 'edge-2:9090']
```

#### 4.2 Grafana Dashboards

```bash
# Import dashboards
curl -X POST http://grafana:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @dashboards/novacron-overview.json

curl -X POST http://grafana:3000/api/dashboards/import \
  -H "Content-Type: application/json" \
  -d @dashboards/vm-metrics.json
```

---

### Step 5: Security Hardening

#### 5.1 Enable RBAC

```bash
# Create admin user
novacron user create \
  --username admin \
  --email admin@company.com \
  --role admin

# Create operator users
novacron user create \
  --username operator1 \
  --email operator1@company.com \
  --role operator
```

#### 5.2 Configure Audit Logging

```yaml
# config/production.yaml
security:
  rbac:
    enabled: true
    auditLog:
      enabled: true
      destination: "/var/log/novacron/audit.log"
      maxSize: "100MB"
      maxBackups: 10
      compress: true
```

#### 5.3 Enable Encryption

```bash
# Generate encryption keys
openssl rand -base64 32 > /etc/novacron/encryption.key

# Configure encryption
kubectl create secret generic encryption-key \
  --from-file=/etc/novacron/encryption.key \
  -n novacron
```

---

### Step 6: Backup Configuration

#### 6.1 Database Backup

```bash
# Create backup script
cat > /usr/local/bin/backup-novacron-db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/backups/novacron"
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h postgres -U novacron novacron_production | \
  gzip > $BACKUP_DIR/novacron_$DATE.sql.gz
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
EOF

chmod +x /usr/local/bin/backup-novacron-db.sh

# Add to crontab
echo "0 2 * * * /usr/local/bin/backup-novacron-db.sh" | crontab -
```

#### 6.2 Configuration Backup

```bash
# Backup configurations
kubectl get configmap -n novacron -o yaml > backup/configmaps.yaml
kubectl get secret -n novacron -o yaml > backup/secrets.yaml
```

---

## 🔍 Post-Deployment Verification

### Health Checks

```bash
# Check backend health
curl https://api.novacron.com/health

# Check database connectivity
novacron db ping

# Check Redis connectivity
novacron redis ping

# Check all services
kubectl get all -n novacron
```

### Performance Testing

```bash
# Run load test
k6 run tests/load/production-load-test.js

# Expected results:
# - Response time p95 < 100ms
# - Success rate > 99.5%
# - Throughput > 1000 req/s
```

### Security Audit

```bash
# Run security scan
trivy image novacron/backend:2.0.0

# Check SSL configuration
sslscan api.novacron.com

# Verify RBAC
novacron security audit
```

---

## 📊 Monitoring & Alerts

### Key Metrics to Monitor

1. **System Metrics**
   - CPU usage < 80%
   - Memory usage < 85%
   - Disk usage < 80%
   - Network throughput

2. **Application Metrics**
   - Request rate
   - Response time (p50, p95, p99)
   - Error rate < 0.5%
   - Active VMs

3. **Business Metrics**
   - VM creation rate
   - Migration success rate > 99%
   - Cost per VM
   - User satisfaction

### Alert Configuration

```yaml
# alerts.yml
groups:
  - name: novacron
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.01
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 0.1
        for: 5m
        annotations:
          summary: "High response time detected"
```

---

## 🔄 Rollback Procedure

### Quick Rollback

```bash
# Rollback to previous version
kubectl rollout undo deployment/novacron-backend -n novacron

# Verify rollback
kubectl rollout status deployment/novacron-backend -n novacron
```

### Database Rollback

```bash
# Restore from backup
gunzip < /backups/novacron/novacron_YYYYMMDD_HHMMSS.sql.gz | \
  psql -h postgres -U novacron novacron_production
```

---

## 📞 Support & Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check for memory leaks
   - Increase pod resources
   - Review caching configuration

2. **Database Connection Issues**
   - Verify connection pool settings
   - Check database health
   - Review network policies

3. **Migration Failures**
   - Check WAN connectivity
   - Verify bandwidth limits
   - Review migration logs

### Getting Help

- **Documentation**: `docs/`
- **Logs**: `kubectl logs -f deployment/novacron-backend -n novacron`
- **Metrics**: `https://grafana.novacron.com`
- **Support**: support@novacron.com

---

**Deployment Status**: ✅ Ready for Production  
**Last Updated**: 2025-10-31  
**Version**: 2.0.0

