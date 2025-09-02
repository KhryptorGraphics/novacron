# NovaCron Production Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying NovaCron in a production environment with enterprise-grade security, monitoring, and high availability features.

## Prerequisites

### System Requirements

**Minimum Hardware:**
- CPU: 8 cores (16 cores recommended)
- RAM: 16GB (32GB recommended)
- Storage: 100GB SSD (500GB recommended)
- Network: 1Gbps connection

**Software Requirements:**
- Docker 24.0+ and Docker Compose 2.20+
- Kubernetes 1.28+ (for K8s deployment)
- kubectl and Helm 3.12+
- OpenSSL for certificate generation
- jq for JSON processing

### Network Requirements

**Ports:**
- 80/443: HTTP/HTTPS traffic
- 8090: API server
- 8091: WebSocket connections
- 5432: PostgreSQL (internal)
- 6379: Redis (internal)
- 9090: Prometheus (monitoring)
- 3000: Grafana (dashboards)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd novacron
cp .env.example .env.production
```

### 2. Configure Environment

Edit `.env.production`:

```bash
# Domain Configuration
DOMAIN=your-domain.com
ACME_EMAIL=admin@your-domain.com

# Database Configuration
DB_USER=novacron_user
DB_NAME=novacron_prod
DB_PASSWORD=<strong-password>

# Redis Configuration  
REDIS_PASSWORD=<strong-password>

# Authentication
AUTH_SECRET=<64-char-jwt-secret>

# Monitoring
GRAFANA_ADMIN_USER=admin
GRAFANA_PASSWORD=<strong-password>
```

### 3. Deploy

```bash
# Docker Compose deployment (recommended for single-server)
./scripts/deploy-prod.sh docker

# Kubernetes deployment (recommended for clusters)
./scripts/deploy-prod.sh kubernetes
```

## Detailed Deployment Steps

### Docker Compose Deployment

#### Step 1: Environment Setup

```bash
# Create production environment
cd deployment/docker
cp ../../.env.production .env

# Generate secure passwords
openssl rand -base64 32  # Use for DB_PASSWORD
openssl rand -base64 32  # Use for REDIS_PASSWORD  
openssl rand -base64 64  # Use for AUTH_SECRET
openssl rand -base64 32  # Use for GRAFANA_PASSWORD
```

#### Step 2: Create Docker Secrets

```bash
# Create secure secrets
echo "your-db-password" | docker secret create novacron_db_password -
echo "your-grafana-password" | docker secret create novacron_grafana_password -
```

#### Step 3: Build Images

```bash
# Build production images
docker build -f deployment/docker/Dockerfile.prod -t novacron/api:latest .
docker build -f deployment/docker/Dockerfile.frontend.prod -t novacron/frontend:latest ./frontend
```

#### Step 4: Deploy Services

```bash
# Deploy all services
docker-compose -f docker-compose.prod.yml up -d

# Check service status
docker-compose -f docker-compose.prod.yml ps
```

#### Step 5: Verify Deployment

```bash
# Check health endpoints
curl -f http://localhost:8090/health
curl -f http://localhost:8092/health

# View logs
docker-compose -f docker-compose.prod.yml logs -f api
```

### Kubernetes Deployment

#### Step 1: Prepare Cluster

```bash
# Create namespace
kubectl apply -f deployment/kubernetes/namespace.yaml

# Verify cluster resources
kubectl get nodes
kubectl get ns novacron-prod
```

#### Step 2: Create Secrets

```bash
# Create application secrets
kubectl create secret generic novacron-secrets \
  --namespace=novacron-prod \
  --from-literal=database-url="postgresql://user:pass@postgres:5432/db" \
  --from-literal=redis-url="redis://:pass@redis:6379" \
  --from-literal=auth-secret="your-jwt-secret"

# Create TLS certificates
kubectl create secret tls novacron-tls \
  --namespace=novacron-prod \
  --cert=path/to/cert.crt \
  --key=path/to/cert.key
```

#### Step 3: Deploy Applications

```bash
# Deploy PostgreSQL
kubectl apply -f deployment/kubernetes/postgres.yaml

# Deploy Redis
kubectl apply -f deployment/kubernetes/redis.yaml

# Deploy API
kubectl apply -f deployment/kubernetes/api-deployment.yaml

# Deploy Frontend
kubectl apply -f deployment/kubernetes/frontend.yaml

# Deploy Ingress
kubectl apply -f deployment/kubernetes/ingress.yaml
```

#### Step 4: Configure Monitoring

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace novacron-prod \
  --values deployment/kubernetes/prometheus-values.yaml

# Deploy Grafana dashboards
kubectl apply -f deployment/kubernetes/grafana-dashboards.yaml
```

## Security Hardening

### SSL/TLS Configuration

**Automatic Certificates (Let's Encrypt):**
```bash
# Configure in docker-compose.prod.yml
certificatesresolvers.letsencrypt.acme.email=admin@your-domain.com
```

**Manual Certificates:**
```bash
# Generate self-signed (development only)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Use production certificates
cp /path/to/your/cert.pem deployment/docker/certs/
cp /path/to/your/key.pem deployment/docker/certs/
```

### Database Security

**PostgreSQL Hardening:**
```sql
-- Create dedicated user
CREATE USER novacron_prod WITH PASSWORD 'strong_password';
CREATE DATABASE novacron_prod OWNER novacron_prod;

-- Restrict permissions
GRANT CONNECT ON DATABASE novacron_prod TO novacron_prod;
GRANT USAGE ON SCHEMA public TO novacron_prod;
GRANT CREATE ON SCHEMA public TO novacron_prod;
```

**Connection Security:**
```bash
# Use connection pooling
POSTGRES_HOST_AUTH_METHOD=scram-sha-256
POSTGRES_INITDB_ARGS=--auth-host=scram-sha-256
```

### Application Security

**JWT Configuration:**
```bash
# Generate secure JWT secret (64+ characters)
AUTH_SECRET=$(openssl rand -base64 64)

# Configure token expiration
JWT_EXPIRY=15m
JWT_REFRESH_EXPIRY=7d
```

**API Rate Limiting:**
```yaml
# Configure in Traefik
http:
  middlewares:
    rate-limit:
      rateLimit:
        burst: 100
        average: 50
```

## Monitoring Setup

### Prometheus Configuration

**Metrics Collection:**
- API response times and error rates
- Database connection pools and query performance
- System resources (CPU, memory, disk)
- Custom business metrics

**Alert Rules:**
- Service downtime (critical)
- High error rates >5% (critical)
- Response time >2s (warning)
- Resource usage >80% (warning)

### Grafana Dashboards

**System Overview:**
- Service health status
- Resource utilization trends
- Error rate monitoring
- Performance metrics

**Application Metrics:**
- API endpoint performance
- Database query analysis
- User activity patterns
- VM operation statistics

### Log Management

**Centralized Logging:**
```bash
# Deploy ELK stack (optional)
docker-compose -f docker-compose.logging.yml up -d

# Configure log forwarding
LOG_DRIVER=fluentd
LOG_OPTIONS="fluentd-address=localhost:24224"
```

**Log Retention:**
- Application logs: 30 days
- Audit logs: 90 days
- System logs: 7 days
- Error logs: 60 days

## High Availability

### Database High Availability

**Master-Replica Setup:**
```yaml
# PostgreSQL replica configuration
postgres-replica:
  image: postgres:15-alpine
  environment:
    POSTGRES_USER: replica
    POSTGRES_REPLICATION_USER: replica
    POSTGRES_REPLICATION_PASSWORD: replica_pass
  command: |
    postgres 
    -c wal_level=replica
    -c max_wal_senders=3
    -c max_replication_slots=3
```

**Automatic Failover:**
```bash
# Use Patroni for automatic failover
helm install patroni bitnami/postgresql-ha \
  --namespace novacron-prod \
  --values patroni-values.yaml
```

### Application Scaling

**Horizontal Pod Autoscaling:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
spec:
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Load Balancing:**
```yaml
# Configure in Traefik
services:
  novacron-api:
    loadBalancer:
      healthCheck:
        path: /health
        interval: 30s
      servers:
      - url: http://api1:8090
      - url: http://api2:8090
      - url: http://api3:8090
```

## Backup and Recovery

### Database Backups

**Automated Backups:**
```bash
#!/bin/bash
# Backup script
pg_dump $DATABASE_URL > backup-$(date +%Y%m%d).sql
aws s3 cp backup-$(date +%Y%m%d).sql s3://backups/novacron/

# Schedule with cron
0 2 * * * /path/to/backup.sh
```

**Point-in-Time Recovery:**
```bash
# Enable WAL archiving
archive_mode = on
archive_command = 'aws s3 cp %p s3://wal-backups/%f'

# Recovery procedure
pg_basebackup -h master -D /var/lib/postgresql/recovery -U replica -W
```

### Application State Backup

**Configuration Backup:**
```bash
# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz deployment/
aws s3 cp config-backup-$(date +%Y%m%d).tar.gz s3://backups/novacron/config/
```

**Volume Snapshots:**
```bash
# Docker volumes
docker run --rm -v novacron_postgres_data:/data -v $(pwd):/backup alpine \
  tar czf /backup/postgres-$(date +%Y%m%d).tar.gz /data

# Kubernetes PV snapshots
kubectl create -f volume-snapshot.yaml
```

## Performance Optimization

### Database Optimization

**PostgreSQL Tuning:**
```sql
-- Connection settings
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB

-- Performance settings
random_page_cost = 1.1
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

**Query Optimization:**
```sql
-- Add indexes for common queries
CREATE INDEX idx_vms_owner_id ON vms(owner_id);
CREATE INDEX idx_vms_state ON vms(state);
CREATE INDEX idx_vm_metrics_timestamp ON vm_metrics(timestamp);
```

### Redis Optimization

**Memory Configuration:**
```bash
# Redis tuning
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
appendonly yes
```

### Application Optimization

**Go Runtime Tuning:**
```bash
# Environment variables
GOMAXPROCS=4
GOMEMLIMIT=1536MiB
GOGC=100
```

**Connection Pooling:**
```go
// Database connection pool
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(5)
db.SetConnMaxLifetime(5 * time.Minute)
```

## Troubleshooting

### Common Issues

**Service Won't Start:**
```bash
# Check logs
docker-compose logs api
kubectl logs -f deployment/novacron-api

# Check configuration
docker-compose config
kubectl describe pod <pod-name>
```

**Database Connection Issues:**
```bash
# Test connectivity
docker exec -it postgres psql -U novacron_user -d novacron_prod -c '\l'

# Check network policies
kubectl describe networkpolicy
```

**High Response Times:**
```bash
# Check system resources
htop
iostat -x 1

# Analyze slow queries
docker exec postgres psql -c "SELECT * FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
```

### Recovery Procedures

**Database Recovery:**
```bash
# Restore from backup
pg_restore -h localhost -U novacron_user -d novacron_prod backup.sql

# Point-in-time recovery
pg_ctl stop
rm -rf /var/lib/postgresql/data/*
pg_basebackup -h replica -D /var/lib/postgresql/data -U replica
```

**Application Recovery:**
```bash
# Rollback deployment
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up -d --force-recreate

# Kubernetes rollback
kubectl rollout undo deployment/novacron-api --namespace novacron-prod
```

## Maintenance

### Regular Tasks

**Daily:**
- Check service health
- Review error logs
- Monitor resource usage
- Verify backups

**Weekly:**
- Update security patches
- Review performance metrics
- Test backup restoration
- Clean up old logs

**Monthly:**
- Security audit
- Capacity planning review
- Disaster recovery testing
- Documentation updates

### Updates and Patches

**Application Updates:**
```bash
# Build new image
docker build -t novacron/api:v1.1.0 .

# Update docker-compose
sed -i 's/novacron\/api:latest/novacron\/api:v1.1.0/' docker-compose.prod.yml

# Rolling update
docker-compose -f docker-compose.prod.yml up -d api
```

**System Updates:**
```bash
# Update base images
docker pull postgres:15-alpine
docker pull redis:7-alpine
docker pull traefik:v3.0

# Security updates
apt update && apt upgrade -y
```

## Support and Contact

**Documentation:**
- Production Readiness Checklist: `/docs/PRODUCTION_READINESS_CHECKLIST.md`
- API Documentation: `/docs/api/`
- Architecture Guide: `/docs/ARCHITECTURE.md`

**Monitoring:**
- Grafana Dashboards: `https://grafana.your-domain.com`
- Prometheus Metrics: `https://metrics.your-domain.com`
- Log Analysis: `https://logs.your-domain.com`

**Emergency Contacts:**
- DevOps Team: devops@your-company.com
- On-call Engineer: +1-555-SUPPORT
- Escalation: cto@your-company.com

---

**Version:** 1.0  
**Last Updated:** 2025-09-02  
**Next Review:** Quarterly