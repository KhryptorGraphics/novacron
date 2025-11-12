# DWCP v3 Phase 5: Deployment Scripts Guide

**Phase:** 5 - Production Deployment and Validation
**Version:** 1.0.0
**Last Updated:** 2025-11-10

## Overview

This directory contains all scripts necessary for deploying DWCP v3 to staging, validating the deployment, and executing a phased production rollout with automatic health monitoring and rollback capabilities.

## Directory Structure

```
scripts/
├── staging-deploy/
│   ├── 01-provision-infrastructure.sh    # Terraform infrastructure provisioning
│   └── 02-deploy-application.sh          # Kubernetes application deployment
├── validation/
│   └── run-validation-suite.sh           # Comprehensive validation tests
├── production-rollout/
│   └── phase1-10pct-rollout.sh          # Production Phase 1 (10% rollout)
└── README-PHASE5-DEPLOYMENT.md           # This file
```

## Quick Start

### Prerequisites

1. **Required Tools:**
   ```bash
   # Verify installations
   terraform --version   # >= 1.5.0
   aws --version        # AWS CLI v2
   docker --version     # >= 20.10
   kubectl version      # >= 1.28.0
   curl --version       # Any recent version
   bc --version         # For calculations
   jq --version         # For JSON parsing
   ```

2. **AWS Credentials:**
   ```bash
   aws configure
   # Verify
   aws sts get-caller-identity
   ```

3. **Kubernetes Context:**
   ```bash
   # For staging deployment
   kubectl config use-context staging-cluster

   # For production rollout
   kubectl config use-context production-cluster
   ```

4. **Docker Registry Login:**
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
   ```

### Staging Deployment (Complete Workflow)

```bash
# Step 1: Provision Infrastructure (5-10 minutes)
cd /home/kp/novacron/scripts/staging-deploy
./01-provision-infrastructure.sh

# Expected output:
# ===== DWCP v3 Phase 5: Staging Infrastructure Provisioning =====
# [INFO] Checking prerequisites...
# [SUCCESS] Prerequisites check passed
# [INFO] Initializing Terraform...
# [SUCCESS] Terraform initialized successfully
# ...
# [SUCCESS] ===== Infrastructure Provisioning Complete =====

# Step 2: Source environment variables
source /home/kp/novacron/deployments/staging-env.sh

# Verify
echo $DWCP_VPC_ID
echo $DWCP_ALB_DNS

# Step 3: Deploy Application (10-15 minutes)
./02-deploy-application.sh

# Expected output:
# ===== DWCP v3 Phase 5: Staging Application Deployment =====
# [INFO] Building Docker image...
# [SUCCESS] Docker image built: ghcr.io/novacron/dwcp-v3:staging-20251110-120000
# [INFO] Pushing Docker image to registry...
# [SUCCESS] Docker image pushed successfully
# ...
# [SUCCESS] ===== Application Deployment Complete =====

# Step 4: Run Validation Suite (5-10 minutes)
cd /home/kp/novacron/scripts/validation
./run-validation-suite.sh

# Expected output:
# ===== DWCP v3 Phase 5: Comprehensive Validation Suite =====
# ═══════════════════════════════════════════════════════
#  1. Infrastructure Validation
# ═══════════════════════════════════════════════════════
# ✓ Namespace exists
# ✓ Deployments created
# ✓ Services created
# ...
# [SUCCESS] ✅ ALL VALIDATIONS PASSED - GO FOR PRODUCTION

# Step 5: Review Validation Report
cat /home/kp/novacron/test-results/staging-validation-*/validation-report.md
```

### Production Rollout (Phased Approach)

```bash
# Prerequisites:
# 1. Staging validation passed
# 2. Manual approvals obtained
# 3. On-call team notified
# 4. Runbooks reviewed

# Switch to production context
kubectl config use-context production-cluster

# Phase 1: 10% Traffic Rollout (Day 1)
cd /home/kp/novacron/scripts/production-rollout
./phase1-10pct-rollout.sh

# Expected output:
# ===== DWCP v3 Phase 5: Production Rollout Phase 1 (10%) =====
# [WARNING] You are about to rollout DWCP v3 to 10% of production traffic
# Type 'ROLLOUT' to confirm: ROLLOUT
# [INFO] Checking prerequisites...
# [SUCCESS] Pre-rollout checks passed
# [INFO] Capturing Baseline Metrics...
# [SUCCESS] Baseline metrics captured
# [INFO] Updating Feature Flag to 10%...
# [SUCCESS] Feature flag updated to 10%
# [INFO] Monitoring Health (600s)...
# [INFO] Health check #1 (30s elapsed, 570s remaining)
#   Error Rate: 0.12% (threshold: 1%)
#   P99 Latency: 47ms (threshold: 100ms)
#   Pods: 3/3 ready
# ...
# [SUCCESS] ✅ Phase 1 Rollout Complete

# Monitor for 24 hours before proceeding to Phase 2

# Phase 2: 50% Traffic Rollout (Day 2)
# Edit phase1-10pct-rollout.sh and change ROLLOUT_PERCENTAGE=50
# Then re-run the script

# Phase 3: 100% Traffic Rollout (Day 3)
# Edit phase1-10pct-rollout.sh and change ROLLOUT_PERCENTAGE=100
# Then re-run the script
```

## Script Details

### 1. Infrastructure Provisioning Script

**File:** `staging-deploy/01-provision-infrastructure.sh`

**Purpose:** Provisions AWS infrastructure using Terraform

**What it does:**
1. Checks prerequisites (Terraform, AWS CLI, credentials)
2. Initializes Terraform with S3 backend
3. Creates staging.tfvars configuration
4. Validates Terraform configuration
5. Plans infrastructure changes
6. Applies infrastructure (with confirmation)
7. Saves outputs (VPC ID, ALB DNS, etc.)
8. Verifies infrastructure
9. Notifies completion via Claude Flow hooks

**Terraform Resources Created:**
- VPC with 3 public and 3 private subnets
- Internet Gateway and NAT Gateways
- Security Groups (ports 80, 443, 8080, 9090, 6379)
- Application Load Balancer
- Target Group with health checks
- CloudWatch Log Group
- S3 bucket for artifacts

**Configuration:**
```bash
# Environment variables
export AWS_REGION=us-east-1
export STATE_BUCKET=novacron-terraform-state

# Execution
./01-provision-infrastructure.sh
```

**Outputs:**
- Terraform state: `deployments/terraform/dwcp-v3/staging.tfstate`
- Outputs JSON: `deployments/terraform/dwcp-v3/staging-outputs.json`
- Environment file: `deployments/staging-env.sh`

**Troubleshooting:**
```bash
# If Terraform fails
cd /home/kp/novacron/deployments/terraform/dwcp-v3
terraform validate
terraform plan -var-file=staging.tfvars

# If state is locked
terraform force-unlock <lock-id>

# Clean up (DANGER - destroys all infrastructure)
terraform destroy -var-file=staging.tfvars
```

### 2. Application Deployment Script

**File:** `staging-deploy/02-deploy-application.sh`

**Purpose:** Builds Docker image and deploys to Kubernetes

**What it does:**
1. Checks prerequisites (Docker, kubectl)
2. Builds DWCP v3 Docker image
3. Pushes image to container registry
4. Creates Kubernetes namespace
5. Deploys secrets (Redis password, API key)
6. Deploys ConfigMap (environment configuration)
7. Deploys Redis cache
8. Deploys DWCP v3 application
9. Waits for deployment to be ready
10. Deploys Service and Ingress
11. Runs smoke tests
12. Saves deployment information

**Configuration:**
```bash
# Environment variables
export NAMESPACE=staging
export IMAGE_TAG=staging-$(date +%Y%m%d-%H%M%S)
export REGISTRY=ghcr.io/novacron

# Execution
./02-deploy-application.sh
```

**Kubernetes Resources Created:**
- Namespace: `staging`
- Deployment: `dwcp-v3` (2 replicas)
- Deployment: `dwcp-v3-redis` (1 replica)
- Service: `dwcp-v3` (ClusterIP)
- Service: `dwcp-v3-redis` (ClusterIP)
- ConfigMap: `dwcp-v3-config`
- Secret: `dwcp-v3-secrets`
- PersistentVolumeClaim: `dwcp-v3-data` (10Gi)
- HorizontalPodAutoscaler: `dwcp-v3-hpa` (2-10 pods)

**Verification:**
```bash
# Check pods
kubectl get pods -n staging

# Check services
kubectl get svc -n staging

# View logs
kubectl logs -n staging -l app=dwcp-v3 -f

# Test health endpoint
kubectl port-forward -n staging svc/dwcp-v3 8080:80
curl http://localhost:8080/health
```

### 3. Validation Suite Script

**File:** `validation/run-validation-suite.sh`

**Purpose:** Comprehensive validation of staging deployment

**What it validates:**
1. **Infrastructure (5 tests)**
   - Namespace existence
   - Deployments created
   - Services created
   - ConfigMaps deployed
   - Secrets deployed

2. **Component Health (5 tests)**
   - DWCP v3 pods ready
   - Redis operational
   - Health endpoint responding
   - Readiness endpoint responding
   - Metrics endpoint responding

3. **DWCP v3 Components (6 tests)**
   - AMST operational
   - HDE operational
   - PBA operational
   - ASS operational
   - ACP operational
   - ITP operational

4. **Performance Baselines (4 metrics)**
   - Datacenter throughput captured
   - Internet compression captured
   - P99 latency captured
   - Error rate captured

5. **Security (5 tests)**
   - Non-root containers
   - Secrets not in env vars
   - CPU limits defined
   - Memory limits defined
   - Resource quotas configured

6. **Monitoring (4 tests)**
   - Prometheus scraping enabled
   - DWCP metrics exposed
   - Application logging working
   - Log aggregation operational

7. **Feature Flags (2 tests)**
   - V3 rollout at 0%
   - DWCP mode configured

8. **Integration (2 tests)**
   - App can reach Redis
   - Service DNS resolution

**Total Tests:** 52+ automated tests

**Configuration:**
```bash
# Environment variables
export NAMESPACE=staging
export DWCP_ENDPOINT=http://localhost:8080

# Execution
./run-validation-suite.sh
```

**Outputs:**
- Test results log: `test-results/staging-validation-<timestamp>/test-results.log`
- Validation report: `test-results/staging-validation-<timestamp>/validation-report.md`
- Exit code: 0 (success) or 1 (failure)

**Success Criteria:**
- All 52 tests passing (100%)
- 0 critical security vulnerabilities
- All DWCP components operational
- Performance baselines captured
- GO/NO-GO decision: **GO FOR PRODUCTION**

### 4. Production Rollout Script (Phase 1)

**File:** `production-rollout/phase1-10pct-rollout.sh`

**Purpose:** Automated 10% production rollout with health monitoring

**What it does:**
1. Pre-rollout validation
2. Captures baseline metrics
3. Updates feature flag to 10%
4. Rolling restart of pods
5. Continuous health monitoring (10 minutes)
6. Validates rollout success
7. Automatic rollback if thresholds violated
8. Generates rollout report

**Health Monitoring:**
- **Check Interval:** 30 seconds
- **Duration:** 10 minutes (20 checks)
- **Metrics Monitored:**
  - Error rate (threshold: 1%)
  - P99 latency (threshold: 100ms)
  - Pod readiness (100% ready)
  - Pod crash count (0 crashes)

**Automatic Rollback Triggers:**
- Error rate > 1%
- P99 latency > 100ms
- < 100% pods ready
- Any pod crashes
- 3+ health check failures

**Configuration:**
```bash
# Environment variables
export ENVIRONMENT=production
export NAMESPACE=production
export ROLLOUT_PERCENTAGE=10
export MONITORING_DURATION=600
export AUTO_ROLLBACK=true

# Execution
./phase1-10pct-rollout.sh
```

**Safety Features:**
- Production context verification
- Manual confirmation required (type "ROLLOUT")
- Real-time health monitoring
- Automatic rollback < 2 minutes
- Incident report generation
- Claude Flow notifications

**Outputs:**
- Rollout report: `docs/DWCP_V3_PHASE5_ROLLOUT_PHASE1_REPORT.md`
- Incident report (if rollback): `/tmp/dwcp-v3-rollback-<timestamp>.md`

## Monitoring and Troubleshooting

### Real-Time Monitoring

```bash
# Watch pods
watch kubectl get pods -n staging

# Stream logs
kubectl logs -n staging -l app=dwcp-v3 -f

# Check metrics
kubectl port-forward -n staging svc/dwcp-v3 9090:9090
curl http://localhost:9090/metrics | grep dwcp_

# Grafana dashboard
kubectl port-forward -n monitoring svc/grafana 3000:80
# Navigate to: http://localhost:3000/d/dwcp-v3-deployment
```

### Common Issues

**Issue 1: Terraform Fails to Initialize**
```bash
# Solution: Check AWS credentials
aws sts get-caller-identity

# Verify S3 bucket exists
aws s3 ls s3://novacron-terraform-state

# Check DynamoDB table
aws dynamodb describe-table --table-name terraform-lock
```

**Issue 2: Docker Build Fails**
```bash
# Solution: Check Dockerfile
cd /home/kp/novacron/deployments/docker
docker build -f Dockerfile.dwcp-v3 -t test .

# Check for syntax errors
hadolint Dockerfile.dwcp-v3
```

**Issue 3: Pods Not Ready**
```bash
# Solution: Check pod events
kubectl describe pod <pod-name> -n staging

# Check logs
kubectl logs <pod-name> -n staging

# Check resource limits
kubectl top pods -n staging
```

**Issue 4: Health Checks Failing**
```bash
# Solution: Check health endpoint
kubectl port-forward -n staging svc/dwcp-v3 8080:80
curl -v http://localhost:8080/health

# Check application logs
kubectl logs -n staging -l app=dwcp-v3 --tail=100
```

**Issue 5: Validation Tests Failing**
```bash
# Solution: Review test results
cat /home/kp/novacron/test-results/staging-validation-*/test-results.log

# Re-run specific category
# Edit run-validation-suite.sh to run only failing tests

# Check individual components
kubectl exec -n staging deployment/dwcp-v3 -- curl localhost:8080/health
```

## Manual Rollback Procedure

If automatic rollback fails or manual intervention is needed:

```bash
# 1. Revert feature flag
kubectl patch configmap dwcp-v3-config -n production \
  --type merge -p '{"data":{"FEATURE_FLAG_V3_ROLLOUT":"0"}}'

# 2. Restart pods
kubectl rollout restart deployment/dwcp-v3 -n production

# 3. Verify rollout
kubectl rollout status deployment/dwcp-v3 -n production

# 4. Check health
kubectl port-forward -n production svc/dwcp-v3 8080:80
curl http://localhost:8080/health

# 5. Verify metrics
kubectl port-forward -n production svc/dwcp-v3 9090:9090
curl http://localhost:9090/metrics | grep dwcp_error_rate
```

## Next Steps

After successful staging validation:

1. **Review validation report:**
   ```bash
   cat /home/kp/novacron/test-results/staging-validation-*/validation-report.md
   ```

2. **Obtain approvals:**
   - Tech Lead sign-off
   - Security team review
   - Product owner approval

3. **Schedule production deployment:**
   - Communicate to stakeholders
   - Notify on-call team
   - Schedule maintenance window (if needed)

4. **Execute Phase 1 rollout:**
   ```bash
   ./production-rollout/phase1-10pct-rollout.sh
   ```

5. **Monitor for 24 hours:**
   - Review metrics hourly
   - Check error logs
   - Monitor user feedback

6. **Proceed to Phase 2 (50%):**
   - Update ROLLOUT_PERCENTAGE=50
   - Re-run rollout script

7. **Proceed to Phase 3 (100%):**
   - Update ROLLOUT_PERCENTAGE=100
   - Re-run rollout script

8. **Deprecate V1 (Week 4):**
   - Verify 0% traffic to V1
   - Remove V1 infrastructure
   - Update documentation

## Additional Resources

- **Phase 5 Documentation:** `/home/kp/novacron/docs/DWCP_V3_PHASE5_STAGING_VALIDATION.md`
- **CI/CD Workflows:** `/home/kp/novacron/.github/workflows/dwcp-v3-cd.yml`
- **Terraform Modules:** `/home/kp/novacron/deployments/terraform/dwcp-v3/`
- **Kubernetes Manifests:** `/home/kp/novacron/deployments/k8s/`
- **Monitoring Configs:** `/home/kp/novacron/deployments/monitoring/`

## Support

- **Slack:** #dwcp-v3-deployment
- **PagerDuty:** On-call rotation
- **Email:** devops@novacron.io

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-10
**Author:** CI/CD Engineer (Phase 5)
