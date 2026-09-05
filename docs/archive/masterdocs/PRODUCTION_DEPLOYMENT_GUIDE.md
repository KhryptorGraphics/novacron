# NovaCron Production Deployment Guide

## Overview

This comprehensive guide walks you through deploying NovaCron to production with enterprise-grade security, monitoring, and backup capabilities.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Security Configuration](#security-configuration)
4. [Database Setup](#database-setup)
5. [Application Deployment](#application-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Backup Configuration](#backup-configuration)
8. [Post-Deployment Verification](#post-deployment-verification)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)

## Prerequisites

### Infrastructure Requirements

- **Kubernetes Cluster**: v1.24+ with ingress controller
- **Storage**: SSD-backed storage classes for database and monitoring
- **Network**: Load balancer with SSL termination capability
- **DNS**: Wildcard DNS for `*.novacron.local` or your domain
- **Resources**: Minimum 16 CPU cores, 32GB RAM across cluster

### Software Dependencies

- `kubectl` v1.24+
- `helm` v3.8+
- `docker` v20.10+
- `openssl` for certificate generation
- `psql` PostgreSQL client
- `aws` CLI (if using S3 backups)

### External Services (Optional)

- **SMTP Server**: For email notifications
- **S3 Bucket**: For backup storage
- **Slack/Webhook**: For alert notifications
- **Certificate Authority**: For TLS certificates

## Infrastructure Setup

### 1. Cluster Preparation

```bash
# Verify cluster connectivity
kubectl cluster-info

# Check available storage classes
kubectl get storageclass

# Verify ingress controller
kubectl get pods -n ingress-nginx
```

### 2. Create Storage Classes (if needed)

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs  # Adjust for your provider
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
allowVolumeExpansion: true
reclaimPolicy: Retain
```

### 3. Configure DNS

Ensure DNS records point to your ingress load balancer:

```
novacron.local              A/CNAME  <ingress-ip>
*.novacron.local           A/CNAME  <ingress-ip>
api.novacron.local         A/CNAME  <ingress-ip>
ws.novacron.local          A/CNAME  <ingress-ip>
grafana.novacron.local     A/CNAME  <ingress-ip>
prometheus.novacron.local  A/CNAME  <ingress-ip>
```

## Security Configuration

### 1. Generate Secrets

```bash
cd scripts
./generate-secrets.sh --apply-k8s
```

This generates:
- JWT secrets
- Database credentials
- TLS certificates
- Encryption keys
- Kubernetes secrets YAML

### 2. Review Generated Configuration

```bash
# Review the generated environment file
cat deployment/secrets/.env.production

# Update external service configuration
vim deployment/secrets/.env.production
```

Update these sections:
- SMTP credentials
- AWS/S3 configuration
- Webhook URLs
- Domain names

### 3. Apply Secrets to Kubernetes

```bash
kubectl apply -f deployment/secrets/secrets-generated.yaml
```

### 4. Verify Secrets

```bash
kubectl get secrets -n novacron
kubectl describe secret novacron-app-secrets -n novacron
```

## Database Setup

### 1. Deploy PostgreSQL

```bash
# Apply namespace and RBAC
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/rbac.yaml

# Apply ConfigMaps
kubectl apply -f deployment/kubernetes/configmap.yaml

# Deploy PostgreSQL StatefulSet
kubectl apply -f deployment/kubernetes/deployments.yaml
```

### 2. Wait for Database to be Ready

```bash
kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgres -n novacron --timeout=300s
```

### 3. Run Database Migrations

```bash
# Port forward to database for initial setup
kubectl port-forward svc/novacron-postgres 5432:5432 -n novacron &

# Run migrations
cd backend/migrations
DATABASE_URL="postgres://username:password@localhost:5432/novacron?sslmode=require" ./run_migrations.sh migrate

# Kill port forward
kill %1
```

### 4. Verify Database Setup

```bash
kubectl exec -it statefulset/novacron-postgres -n novacron -- \
  psql -U novacron -d novacron -c "\dt"
```

## Application Deployment

### 1. Deploy Application Services

```bash
# Deploy API and Frontend
kubectl apply -f deployment/kubernetes/deployments.yaml
kubectl apply -f deployment/kubernetes/services.yaml
```

### 2. Configure Ingress

```bash
# Apply ingress configuration
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### 3. Wait for Deployment

```bash
kubectl wait --for=condition=available deployment/novacron-api -n novacron --timeout=600s
kubectl wait --for=condition=available deployment/novacron-frontend -n novacron --timeout=600s
```

### 4. Verify Application Deployment

```bash
# Check pod status
kubectl get pods -n novacron

# Check service endpoints
kubectl get endpoints -n novacron

# Test health endpoints
curl -k https://novacron.local/health
curl -k https://api.novacron.local/health
```

## Monitoring Setup

### 1. Deploy Monitoring Stack

```bash
cd scripts/monitoring
./setup-monitoring.sh -n monitoring -p mySecureGrafanaPassword
```

### 2. Verify Monitoring

```bash
# Check monitoring pods
kubectl get pods -n monitoring

# Test Prometheus
curl -k https://prometheus.novacron.local/-/ready

# Test Grafana
curl -k https://admin:mySecureGrafanaPassword@grafana.novacron.local/api/health
```

### 3. Import Dashboards

Access Grafana at https://grafana.novacron.local and import dashboards:
- NovaCron Overview (ID: auto-generated)
- PostgreSQL Dashboard (ID: 9628)
- Node Exporter Dashboard (ID: 1860)
- Kubernetes Cluster Dashboard (ID: 315)

## Backup Configuration

### 1. Configure Backup Service

```bash
# Review backup configuration
vim backend/configs/production/app.yaml

# Apply backup CronJob
kubectl apply -f deployment/kubernetes/cronjobs.yaml
```

### 2. Test Backup

```bash
# Manual backup test
kubectl create job --from=cronjob/novacron-backup manual-test-backup -n novacron

# Check backup job
kubectl logs job/manual-test-backup -n novacron
```

### 3. Verify Backup Storage

```bash
# Check local backup storage
kubectl exec -it deployment/novacron-api -n novacron -- ls -la /var/backups/novacron/

# Check S3 backups (if configured)
aws s3 ls s3://your-backup-bucket/novacron/database/
```

## Post-Deployment Verification

### 1. Comprehensive Health Check

```bash
cd scripts
./health-check.sh production
```

### 2. Security Scan

```bash
# Run security validation
kubectl get networkpolicies -n novacron
kubectl get podsecuritypolicy

# Check RBAC
kubectl auth can-i --list --as=system:serviceaccount:novacron:novacron-api
```

### 3. Performance Testing

```bash
# Basic load test
ab -n 1000 -c 10 https://api.novacron.local/health

# Database performance
kubectl exec -it statefulset/novacron-postgres -n novacron -- \
  pgbench -i -s 10 novacron
```

### 4. Create Initial Admin User

```bash
kubectl exec -it deployment/novacron-api -n novacron -- \
  /app/create-admin-user --username admin --email admin@novacron.local
```

## Troubleshooting

### Common Issues

#### 1. Database Connection Issues

```bash
# Check database status
kubectl get pods -l app.kubernetes.io/name=postgres -n novacron

# Check database logs
kubectl logs statefulset/novacron-postgres -n novacron

# Test connection
kubectl exec -it deployment/novacron-api -n novacron -- \
  psql $DATABASE_URL -c "SELECT version();"
```

#### 2. API Server Issues

```bash
# Check API logs
kubectl logs deployment/novacron-api -n novacron

# Check service endpoints
kubectl describe endpoints novacron-api -n novacron

# Check ingress
kubectl describe ingress novacron-ingress -n novacron
```

#### 3. Storage Issues

```bash
# Check PVC status
kubectl get pvc -n novacron

# Check storage class
kubectl describe storageclass fast-ssd

# Check volume mounts
kubectl describe pod <pod-name> -n novacron
```

#### 4. Certificate Issues

```bash
# Check TLS secrets
kubectl describe secret novacron-tls-certs -n novacron

# Test certificate
openssl s_client -connect novacron.local:443 -servername novacron.local
```

### Debug Commands

```bash
# Get all resources in novacron namespace
kubectl get all -n novacron

# Check events
kubectl get events -n novacron --sort-by='.lastTimestamp'

# Describe problematic resources
kubectl describe <resource-type> <resource-name> -n novacron

# Check resource usage
kubectl top pods -n novacron
kubectl top nodes
```

## Maintenance

### Regular Tasks

#### Daily
- [ ] Check application health endpoints
- [ ] Review monitoring alerts
- [ ] Verify backup completion

#### Weekly
- [ ] Review security alerts
- [ ] Check disk usage and cleanup
- [ ] Update monitoring dashboards

#### Monthly
- [ ] Update application dependencies
- [ ] Rotate secrets and certificates
- [ ] Performance review and optimization
- [ ] Security audit

### Update Procedures

#### Application Updates

```bash
# Update image tags in deployment
sed -i 's/image: novacron\/api-server:.*/image: novacron\/api-server:v1.1.0/g' \
  deployment/kubernetes/deployments.yaml

# Apply updates
kubectl apply -f deployment/kubernetes/deployments.yaml

# Monitor rollout
kubectl rollout status deployment/novacron-api -n novacron
```

#### Database Migrations

```bash
# Create backup before migration
kubectl create job --from=cronjob/novacron-backup pre-migration-backup -n novacron

# Run migrations
kubectl exec -it deployment/novacron-api -n novacron -- /app/migrate up
```

#### Certificate Renewal

```bash
# Renew certificates
cd scripts
./renew-certificates.sh

# Update secrets
kubectl apply -f deployment/secrets/secrets-generated.yaml
```

### Scaling

#### Horizontal Scaling

```bash
# Scale API servers
kubectl scale deployment novacron-api --replicas=5 -n novacron

# Scale frontend
kubectl scale deployment novacron-frontend --replicas=3 -n novacron
```

#### Vertical Scaling

```bash
# Update resource limits in deployment YAML
vim deployment/kubernetes/deployments.yaml

# Apply changes
kubectl apply -f deployment/kubernetes/deployments.yaml
```

### Backup and Recovery

#### Manual Backup

```bash
cd scripts/backup
./backup-database.sh
```

#### Disaster Recovery

```bash
# Stop application
kubectl scale deployment novacron-api --replicas=0 -n novacron

# Restore database
kubectl exec -it statefulset/novacron-postgres -n novacron -- \
  pg_restore -U novacron -d novacron -v /path/to/backup.sql

# Start application
kubectl scale deployment novacron-api --replicas=3 -n novacron
```

## Security Best Practices

### 1. Network Security
- Enable network policies
- Use internal service mesh
- Regular security scans
- Monitor network traffic

### 2. Access Control
- Implement RBAC properly
- Use service accounts
- Regular access reviews
- Multi-factor authentication

### 3. Data Protection
- Encrypt data at rest
- Secure data in transit
- Regular backup testing
- Data retention policies

### 4. Monitoring
- Security event monitoring
- Anomaly detection
- Regular security audits
- Incident response procedures

## Support and Documentation

### Additional Resources
- [Configuration Reference](CONFIG_REFERENCE.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Security Guide](SECURITY_GUIDE.md)
- [Scaling Guide](SCALING_GUIDE.md)

### Getting Help
- Check logs first
- Review monitoring dashboards
- Consult troubleshooting guide
- Contact operations team

---

**Document Version**: 1.0.0  
**Last Updated**: $(date)  
**Deployment Target**: Production  
**Maintainer**: DevOps Team