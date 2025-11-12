# DWCP v3 CI/CD Guide

Comprehensive guide for CI/CD pipelines, deployment automation, and operational procedures for DWCP v3.

## Table of Contents

1. [Overview](#overview)
2. [CI/CD Architecture](#cicd-architecture)
3. [GitHub Actions Workflows](#github-actions-workflows)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Terraform Infrastructure](#terraform-infrastructure)
7. [Deployment Procedures](#deployment-procedures)
8. [Rollback Procedures](#rollback-procedures)
9. [Monitoring and Alerting](#monitoring-and-alerting)
10. [Troubleshooting](#troubleshooting)

## Overview

DWCP v3 uses a comprehensive CI/CD pipeline built with GitHub Actions, containerized with Docker, orchestrated with Kubernetes, and provisioned with Terraform.

### Key Features

- **Automated Testing**: Unit, integration, and performance tests
- **Multi-Stage Build**: Optimized Docker images
- **Gradual Rollout**: Feature flag-based deployment (10% → 50% → 100%)
- **Automatic Rollback**: On failure detection
- **Infrastructure as Code**: Terraform-managed cloud resources
- **Comprehensive Monitoring**: Prometheus + Grafana stack

## CI/CD Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     GitHub Repository                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│               GitHub Actions CI Pipeline                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Quality  │→ │  Tests   │→ │  Build   │→ │  Docker  │   │
│  │  Check   │  │          │  │          │  │  Build   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│              GitHub Actions CD Pipeline                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Staging  │→ │Production│→ │  Feature │→ │  Health  │   │
│  │  Deploy  │  │  Deploy  │  │  Flags   │  │  Checks  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                  Kubernetes Cluster                          │
│  ┌───────────────────┐      ┌───────────────────┐          │
│  │   DWCP v3 Pods    │      │   Redis Cache     │          │
│  │   (Auto-scaled)   │      │                   │          │
│  └───────────────────┘      └───────────────────┘          │
│  ┌───────────────────┐      ┌───────────────────┐          │
│  │   Prometheus      │      │   Grafana         │          │
│  │   Monitoring      │      │   Dashboards      │          │
│  └───────────────────┘      └───────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## GitHub Actions Workflows

### CI Pipeline (.github/workflows/dwcp-v3-ci.yml)

**Triggers:**
- Push to `main`, `develop`, `feature/**`, `dwcp-v3/**` branches
- Pull requests to `main`, `develop`
- Manual workflow dispatch

**Jobs:**

1. **Code Quality & Linting**
   - ESLint
   - Prettier format check
   - TypeScript type checking
   - Security vulnerability scanning

2. **Unit Tests**
   - Matrix testing (Node 18, 20)
   - Coverage threshold: 90%
   - Upload coverage to Codecov

3. **Integration Tests**
   - Redis service container
   - Full integration test suite
   - Test result artifacts

4. **DWCP v3 Component Tests**
   - Matrix testing per component:
     - Codec
     - Framing
     - Connection
     - Multiplexing
     - Flow Control
     - Reliability
   - Component-specific coverage

5. **Performance Tests**
   - Benchmark suite
   - Performance regression checks
   - Results archived for 30 days

6. **Security Scanning**
   - Trivy vulnerability scanner
   - SARIF upload to GitHub Security
   - npm audit

7. **Build Verification**
   - Production build
   - Artifact verification
   - Build artifact upload

8. **Docker Build Test**
   - Multi-stage build validation
   - Build cache optimization

### CD Pipeline (.github/workflows/dwcp-v3-cd.yml)

**Triggers:**
- Push to `main`
- Git tags (`v*`)
- Manual workflow dispatch

**Jobs:**

1. **Build and Push Docker Image**
   - Multi-stage optimized build
   - Push to GitHub Container Registry
   - Generate SBOM (Software Bill of Materials)
   - Image metadata tagging

2. **Deploy to Staging**
   - Automatic on `main` branch
   - Smoke tests
   - Environment: `staging`

3. **Deploy to Production**
   - Requires staging success
   - Manual approval required
   - Environment: `production`
   - Deployment backup
   - Health checks
   - Metrics verification

4. **Feature Flag Rollout**
   - Gradual rollout: 10% → 50% → 100%
   - Monitoring at each stage
   - Automatic progression

5. **Automatic Rollback**
   - Triggers on deployment failure
   - Restores previous version
   - Disables feature flags
   - Creates incident issue

6. **Post-Deployment Validation**
   - End-to-end tests
   - Deployment report generation
   - Metrics update

## Docker Deployment

### Multi-Stage Dockerfile

**Location:** `deployments/docker/Dockerfile.dwcp-v3`

**Stages:**

1. **Dependencies Stage**
   - Production dependencies only
   - Optimized for caching

2. **Builder Stage**
   - Full dependencies
   - Application build
   - Production pruning

3. **Runtime Stage**
   - Minimal Alpine base
   - Non-root user (nodejs:1001)
   - dumb-init for signal handling
   - Health checks included

**Features:**
- Multi-stage optimization
- Security best practices
- Health check endpoints
- Minimal attack surface
- Proper signal handling

### Docker Compose Stack

**Location:** `deployments/docker-compose-dwcp-v3.yml`

**Services:**

1. **dwcp-v3** - Main application
   - Ports: 8080 (app), 9090 (metrics)
   - Resource limits: 2 CPU, 2GB RAM
   - Health checks
   - Auto-restart

2. **redis** - State management
   - Persistent volume
   - Memory limit: 512MB
   - Health checks

3. **prometheus** - Metrics collection
   - Scrapes metrics every 30s
   - Alert rules configured
   - Persistent storage

4. **grafana** - Dashboards
   - Pre-configured dashboards
   - Redis datasource
   - Admin credentials configurable

5. **node-exporter** - System metrics
   - Host metrics collection

6. **redis-exporter** - Redis metrics
   - Redis-specific metrics

7. **alertmanager** - Alert handling
   - Alert routing
   - Notification configuration

**Usage:**

```bash
# Start full stack
docker-compose -f deployments/docker-compose-dwcp-v3.yml up -d

# View logs
docker-compose -f deployments/docker-compose-dwcp-v3.yml logs -f

# Stop stack
docker-compose -f deployments/docker-compose-dwcp-v3.yml down

# Stop and remove volumes
docker-compose -f deployments/docker-compose-dwcp-v3.yml down -v
```

## Kubernetes Deployment

### Deployment Manifests

**Location:** `deployments/k8s/dwcp-v3-deployment.yaml`

**Resources:**

1. **Namespace:** `dwcp-v3`
2. **ConfigMap:** Application configuration
3. **Secret:** Sensitive data (Redis password, API keys)
4. **Deployment:** 3 replicas, rolling update strategy
5. **Service:** ClusterIP with load balancing
6. **ServiceAccount:** RBAC configuration
7. **PersistentVolumeClaim:** Data storage (10Gi)
8. **Redis Deployment:** State management
9. **HorizontalPodAutoscaler:** Auto-scaling (3-10 replicas)
10. **Ingress:** External access with TLS

**Resource Limits:**

```yaml
requests:
  cpu: 500m
  memory: 512Mi
limits:
  cpu: 2000m
  memory: 2Gi
```

**Health Checks:**

- **Liveness Probe:** `/health` endpoint, 30s initial delay
- **Readiness Probe:** `/ready` endpoint, 10s initial delay

### Monitoring Stack

**Location:** `deployments/k8s/dwcp-v3-monitoring.yaml`

**Components:**

1. **ServiceMonitor:** Prometheus scraping configuration
2. **Grafana Dashboard:** Pre-configured DWCP v3 metrics
3. **Alert Rules:**
   - High error rate (>5%)
   - High latency (>1s P95)
   - Low throughput (<1MB/s)
   - Pod restarts
   - High memory/CPU usage
   - Deployment issues
   - Redis down
   - Low disk space

4. **Prometheus Config:**
   - 30s scrape interval
   - Kubernetes service discovery
   - Alert manager integration

**Deploy Monitoring:**

```bash
kubectl apply -f deployments/k8s/dwcp-v3-monitoring.yaml
```

## Terraform Infrastructure

### Infrastructure as Code

**Location:** `deployments/terraform/dwcp-v3/`

**Modules:**

1. **main.tf** - Core infrastructure
   - VPC with public/private subnets
   - Internet Gateway
   - NAT Gateway
   - Security Groups
   - Application Load Balancer
   - Target Groups
   - CloudWatch Logs
   - S3 Artifact Bucket

2. **variables.tf** - Configuration variables
   - Environment settings
   - Network configuration
   - Application parameters
   - Auto-scaling settings
   - DWCP v3-specific config

**Key Resources:**

- **VPC:** 10.0.0.0/16 CIDR
- **Subnets:** 3 public + 3 private across AZs
- **Security Groups:**
  - HTTP/HTTPS (80, 443)
  - Application (8080)
  - Metrics (9090)
  - Redis (6379)
- **ALB:** Layer 7 load balancing with health checks
- **CloudWatch:** 30-day log retention

### Terraform Usage

```bash
# Initialize Terraform
cd deployments/terraform/dwcp-v3
terraform init

# Plan deployment
terraform plan -var="environment=production"

# Apply infrastructure
terraform apply -var="environment=production"

# Show outputs
terraform output

# Destroy infrastructure (careful!)
terraform destroy -var="environment=production"
```

### State Management

- **Backend:** S3 with DynamoDB locking
- **Bucket:** `novacron-terraform-state`
- **Key:** `dwcp-v3/terraform.tfstate`
- **Encryption:** Enabled

## Deployment Procedures

### Manual Deployment

**Script:** `scripts/deploy-dwcp-v3.sh`

**Usage:**

```bash
# Deploy to staging
./scripts/deploy-dwcp-v3.sh --environment staging --tag v3.0.0

# Deploy to production
./scripts/deploy-dwcp-v3.sh --environment production --tag v3.0.0

# Dry run
./scripts/deploy-dwcp-v3.sh --dry-run --environment production --tag v3.0.0

# Skip tests
./scripts/deploy-dwcp-v3.sh --environment staging --skip-tests

# Disable auto-rollback
./scripts/deploy-dwcp-v3.sh --environment production --no-rollback
```

**Features:**

- ✅ Pre-deployment checks
- ✅ Backup current deployment
- ✅ Run tests
- ✅ Validate manifests
- ✅ Deploy with health checks
- ✅ Automatic rollback on failure
- ✅ Post-deployment report

### Gradual Rollout

**Script:** `scripts/rollout-dwcp-v3.sh`

**Usage:**

```bash
# Full gradual rollout (10% → 50% → 100%)
./scripts/rollout-dwcp-v3.sh --image ghcr.io/novacron/dwcp-v3:v3.0.0

# Deploy to specific percentage
./scripts/rollout-dwcp-v3.sh --image ghcr.io/novacron/dwcp-v3:v3.0.0 --percentage 50

# Custom wait time between stages
./scripts/rollout-dwcp-v3.sh --image ghcr.io/novacron/dwcp-v3:v3.0.0 --wait-time 600

# Custom error threshold
./scripts/rollout-dwcp-v3.sh --image ghcr.io/novacron/dwcp-v3:v3.0.0 --error-threshold 3
```

**Rollout Process:**

1. Deploy new version at 0% traffic
2. Set feature flag to 10%
3. Monitor for 5 minutes
4. If healthy, increase to 50%
5. Monitor for 5 minutes
6. If healthy, increase to 100%
7. Automatic rollback on failures

## Rollback Procedures

### Automatic Rollback

Automatic rollback triggers on:
- Deployment failure
- Health check failures
- High error rates during gradual rollout

### Manual Rollback

**Using kubectl:**

```bash
# Rollback to previous version
kubectl rollout undo deployment/dwcp-v3 -n production

# Rollback to specific revision
kubectl rollout undo deployment/dwcp-v3 -n production --to-revision=2

# Check rollout status
kubectl rollout status deployment/dwcp-v3 -n production
```

**Using deployment script:**

```bash
# The script automatically creates backups
# Restore from backup
kubectl apply -f backups/deployments/dwcp-v3-production-TIMESTAMP.yaml
```

### Feature Flag Rollback

```bash
# Disable feature flags
kubectl create configmap dwcp-v3-rollout \
  --from-literal=percentage=0 \
  -n production \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart deployment
kubectl rollout restart deployment/dwcp-v3 -n production
```

## Monitoring and Alerting

### Prometheus Metrics

**DWCP v3 Metrics:**

- `dwcp_requests_total` - Total requests
- `dwcp_errors_total` - Total errors
- `dwcp_latency_seconds` - Request latency histogram
- `dwcp_active_connections` - Active connection count
- `dwcp_bytes_transferred_total` - Total bytes transferred
- `process_resident_memory_bytes` - Memory usage

**System Metrics:**

- CPU usage
- Memory usage
- Disk usage
- Network I/O

### Grafana Dashboards

**Pre-configured Panels:**

1. Request Rate (5m rate)
2. Error Rate (5m rate)
3. Latency P95
4. Active Connections
5. Throughput (MB/s)
6. Memory Usage

**Access:**

```
URL: http://localhost:3000
Default User: admin
Default Password: admin (change on first login)
```

### Alert Rules

**Critical Alerts:**

- Pod restarts
- Deployment replica mismatch
- Redis down

**Warning Alerts:**

- High error rate (>5%)
- High latency (>1s P95)
- Low throughput (<1MB/s)
- High memory usage (>90%)
- High CPU usage (>1.8 cores)
- Low disk space (<10%)

**Alert Configuration:**

Edit `deployments/k8s/dwcp-v3-monitoring.yaml` to customize alerts.

## Troubleshooting

### Common Issues

#### 1. Deployment Stuck in Pending

**Symptoms:**
```bash
$ kubectl get pods -n dwcp-v3
NAME                      READY   STATUS    RESTARTS   AGE
dwcp-v3-xxx-yyy          0/1     Pending   0          5m
```

**Solutions:**

```bash
# Check node resources
kubectl describe nodes

# Check pod events
kubectl describe pod <pod-name> -n dwcp-v3

# Check persistent volume claims
kubectl get pvc -n dwcp-v3
```

#### 2. High Error Rate

**Symptoms:**
- Prometheus alert: HighErrorRate
- Application returning errors

**Solutions:**

```bash
# Check application logs
kubectl logs -n dwcp-v3 -l app=dwcp-v3 --tail=100

# Check Redis connectivity
kubectl exec -n dwcp-v3 <pod-name> -- redis-cli -h dwcp-v3-redis ping

# Rollback if necessary
kubectl rollout undo deployment/dwcp-v3 -n dwcp-v3
```

#### 3. Image Pull Errors

**Symptoms:**
```
Error: ErrImagePull
```

**Solutions:**

```bash
# Check image exists
docker pull ghcr.io/novacron/dwcp-v3:v3.0.0

# Verify image pull secret
kubectl get secrets -n dwcp-v3

# Create image pull secret if needed
kubectl create secret docker-registry regcred \
  --docker-server=ghcr.io \
  --docker-username=<username> \
  --docker-password=<token> \
  -n dwcp-v3
```

#### 4. Health Check Failures

**Symptoms:**
- Pods in CrashLoopBackOff
- Liveness/Readiness probe failures

**Solutions:**

```bash
# Test health endpoint manually
kubectl exec -n dwcp-v3 <pod-name> -- curl http://localhost:8080/health

# Check application startup
kubectl logs -n dwcp-v3 <pod-name> --previous

# Increase probe initial delay if needed
kubectl edit deployment/dwcp-v3 -n dwcp-v3
# Modify initialDelaySeconds
```

#### 5. Performance Issues

**Symptoms:**
- High latency
- Low throughput
- Resource exhaustion

**Solutions:**

```bash
# Check resource usage
kubectl top pods -n dwcp-v3
kubectl top nodes

# Increase resources
kubectl edit deployment/dwcp-v3 -n dwcp-v3
# Modify resources.limits and resources.requests

# Scale up replicas
kubectl scale deployment/dwcp-v3 --replicas=5 -n dwcp-v3
```

### Debug Commands

```bash
# Get all resources
kubectl get all -n dwcp-v3

# Check events
kubectl get events -n dwcp-v3 --sort-by='.lastTimestamp'

# Exec into pod
kubectl exec -it -n dwcp-v3 <pod-name> -- /bin/sh

# Port forward for local testing
kubectl port-forward -n dwcp-v3 svc/dwcp-v3 8080:80

# View logs with follow
kubectl logs -n dwcp-v3 -l app=dwcp-v3 -f

# Check deployment status
kubectl rollout status deployment/dwcp-v3 -n dwcp-v3

# View deployment history
kubectl rollout history deployment/dwcp-v3 -n dwcp-v3
```

## Best Practices

### Development

1. **Always run tests locally** before pushing
2. **Use feature branches** for new features
3. **Write tests** for new functionality
4. **Update documentation** with code changes
5. **Follow semantic versioning** for tags

### Deployment

1. **Deploy to staging first** always
2. **Monitor metrics** during deployment
3. **Use gradual rollouts** for production
4. **Keep backups** before major changes
5. **Document incidents** and resolutions

### Security

1. **Never commit secrets** to repository
2. **Use Kubernetes secrets** for sensitive data
3. **Rotate credentials** regularly
4. **Keep dependencies updated** (npm audit)
5. **Review security scans** in CI

### Monitoring

1. **Set up alerts** for critical metrics
2. **Review dashboards** regularly
3. **Establish baselines** for normal operation
4. **Investigate anomalies** promptly
5. **Keep logs** for troubleshooting

## Support and Resources

- **GitHub Repository:** https://github.com/novacron/novacron
- **Documentation:** `/home/kp/novacron/docs/`
- **Issues:** GitHub Issues
- **Monitoring:** Grafana dashboards
- **Logs:** CloudWatch / Kubernetes logs

## Appendix

### Environment Variables

**Application:**
- `NODE_ENV` - Environment (production, staging)
- `PORT` - Application port (default: 8080)
- `LOG_LEVEL` - Logging level (info, debug, error)
- `REDIS_URL` - Redis connection URL
- `PROMETHEUS_PORT` - Metrics port (default: 9090)

**DWCP v3:**
- `DWCP_VERSION` - Protocol version
- `MAX_CONNECTIONS` - Maximum concurrent connections
- `BUFFER_SIZE` - Buffer size in bytes

### File Locations

- CI/CD Workflows: `.github/workflows/`
- Docker files: `deployments/docker/`
- Kubernetes manifests: `deployments/k8s/`
- Terraform modules: `deployments/terraform/dwcp-v3/`
- Deployment scripts: `scripts/`
- Documentation: `docs/`
- Backups: `backups/deployments/`

---

**Last Updated:** 2025-11-10
**Version:** 1.0.0
**Maintainer:** NovaCron Team
