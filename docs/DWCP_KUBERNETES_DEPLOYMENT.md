# NovaCron DWCP Phase 3: Global Kubernetes Deployment

## Executive Summary

This document provides comprehensive guidance for deploying NovaCron's distributed VM management platform globally using Kubernetes with DWCP (Distributed WAN Communication Protocol) Phase 3 enhancements.

**Key Features:**
- Multi-region Kubernetes deployment automation
- GitOps workflows with ArgoCD
- Zero-downtime rolling updates
- Auto-scaling (HPA + Cluster Autoscaler)
- Service mesh integration (Istio)
- Disaster recovery with Velero
- Comprehensive observability (Prometheus, Grafana, Jaeger)

## Architecture Overview

### Deployment Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                        Global Load Balancer                      │
│                    (Route53 / CloudFlare / GCP)                  │
└────────────┬────────────────────────────┬─────────────────┬──────┘
             │                            │                 │
    ┌────────▼────────┐         ┌─────────▼──────┐  ┌──────▼──────┐
    │  US-EAST-1      │         │  EU-WEST-1     │  │ AP-SE-1     │
    │  K8s Cluster    │◄────────┤  K8s Cluster   │◄─┤ K8s Cluster │
    │                 │  DWCP   │                │  │             │
    │  20 API Pods    │         │  15 API Pods   │  │ 10 API Pods │
    │  7 Consensus    │         │  5 Consensus   │  │ 5 Consensus │
    │  DaemonSet DWCP │         │  DaemonSet DWCP│  │ DaemonSet   │
    └─────────────────┘         └────────────────┘  └─────────────┘
```

### Component Architecture

**Control Plane:**
- Kubernetes Operator (CRD-based cluster management)
- ArgoCD (GitOps controller)
- Cluster Autoscaler
- External Secrets Operator

**Data Plane:**
- API Server Deployment (HPA: 3-100 replicas)
- Consensus StatefulSet (5-11 nodes, always odd)
- DWCP Network DaemonSet (on every node)
- PostgreSQL StatefulSet (primary + 3 replicas)
- Redis Cluster (3 masters, 3 replicas)

**Service Mesh:**
- Istio control plane
- Envoy sidecars
- mTLS enforcement
- Traffic management

**Observability:**
- Prometheus (metrics collection)
- Grafana (visualization)
- Jaeger (distributed tracing)
- OpenTelemetry Collector

## Quick Start

### Prerequisites

1. **Kubernetes Clusters:**
   - 3+ clusters (multi-region recommended)
   - Kubernetes 1.27+
   - CNI plugin installed (Calico/Cilium/Flannel)
   - StorageClass with dynamic provisioning

2. **Tools:**
   - kubectl 1.27+
   - helm 3.12+
   - argocd CLI
   - velero CLI

3. **Cloud Resources:**
   - S3 buckets (or equivalent) for backups
   - Container registry (ECR/GCR/Harbor)
   - DNS zone for domain management

### Installation Steps

#### 1. Install CRDs and Operator

```bash
# Install CRDs
kubectl apply -f backend/deployments/k8s/operator/crd_novacroncluster.yaml
kubectl apply -f backend/deployments/k8s/operator/crd_novacronregion.yaml
kubectl apply -f backend/deployments/k8s/operator/crd_dwcpfederation.yaml

# Deploy operator
kubectl create namespace novacron-system
kubectl apply -f backend/deployments/k8s/operator/operator-deployment.yaml
```

#### 2. Install Helm Chart (Single Region)

```bash
# Add Helm dependencies
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install to staging
helm install novacron ./backend/deployments/k8s/charts/novacron \
  --namespace novacron \
  --create-namespace \
  --values ./backend/deployments/k8s/charts/novacron/values-staging.yaml

# Install to production
helm install novacron ./backend/deployments/k8s/charts/novacron \
  --namespace novacron \
  --create-namespace \
  --values ./backend/deployments/k8s/charts/novacron/values-production.yaml
```

#### 3. Deploy Using CRD (Multi-Region)

```bash
# Create NovaCronCluster resource
cat <<EOF | kubectl apply -f -
apiVersion: novacron.io/v1
kind: NovaCronCluster
metadata:
  name: production
  namespace: novacron
spec:
  version: "3.0.0"
  regions:
    - name: us-east-1
      replicas: 20
      resources:
        requests:
          cpu: "16"
          memory: "32Gi"
    - name: eu-west-1
      replicas: 15
      resources:
        requests:
          cpu: "12"
          memory: "24Gi"
    - name: ap-southeast-1
      replicas: 10
      resources:
        requests:
          cpu: "8"
          memory: "16Gi"
  dwcp:
    enabled: true
    amst:
      minStreams: 32
      maxStreams: 512
    hde:
      compressionLevel: 9
    acp:
      nodes: 7
      quorumSize: 4
      consensusEngine: raft
  monitoring:
    prometheus: true
    grafana: true
    jaeger: true
  serviceMesh:
    enabled: true
EOF

# Wait for cluster to be ready
kubectl wait --for=condition=Ready novacroncluster/production -n novacron --timeout=30m
```

#### 4. Setup GitOps with ArgoCD

```bash
# Install ArgoCD
kubectl create namespace argocd
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Apply ArgoCD applications
kubectl apply -f backend/deployments/k8s/argocd/novacron-core.yaml
kubectl apply -f backend/deployments/k8s/argocd/novacron-dwcp.yaml
kubectl apply -f backend/deployments/k8s/argocd/novacron-monitoring.yaml

# Sync applications
argocd app sync novacron-core
argocd app sync novacron-dwcp
argocd app sync novacron-monitoring
```

## Scaling Guide

### Horizontal Pod Autoscaling

**Automatic scaling based on metrics:**

```yaml
# HPA configuration (already deployed via Helm)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: novacron-api-hpa
spec:
  minReplicas: 3
  maxReplicas: 100
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: "1000"
```

**Manual scaling:**

```bash
# Scale API deployment
kubectl scale deployment novacron-api --replicas=50 -n novacron

# Scale consensus StatefulSet (must be odd number)
kubectl scale statefulset novacron-consensus --replicas=9 -n novacron
```

### Cluster Autoscaling

**Node pool auto-scaling:**

```bash
# AWS EKS example
aws autoscaling update-auto-scaling-group \
  --auto-scaling-group-name novacron-node-group \
  --min-size 10 \
  --max-size 100 \
  --desired-capacity 20

# GKE example
gcloud container clusters update novacron-cluster \
  --enable-autoscaling \
  --min-nodes 10 \
  --max-nodes 100 \
  --zone us-east1-b
```

### Multi-Region Scaling

**Using KubeFed for federated scaling:**

```bash
# Update replica scheduling
kubectl patch replicaschedulingpreference novacron-api-scheduling \
  -n novacron \
  --type merge \
  -p '{"spec": {"totalReplicas": 100}}'
```

## Deployment Strategies

### Zero-Downtime Rolling Update

**Deployment configuration:**

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxUnavailable: 0  # Never reduce capacity
    maxSurge: 1        # Add one pod at a time
```

**Update process:**

```bash
# Update image
kubectl set image deployment/novacron-api \
  api=novacron/api:3.1.0 -n novacron

# Monitor rollout
kubectl rollout status deployment/novacron-api -n novacron

# Rollback if needed
kubectl rollout undo deployment/novacron-api -n novacron
```

### Canary Deployment

**Using Istio for traffic splitting:**

```bash
# Deploy canary version
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-api-canary
  labels:
    version: v3.1.0-canary
spec:
  replicas: 2
  # ... rest of deployment
EOF

# Configure traffic split (90% stable, 10% canary)
kubectl apply -f backend/deployments/k8s/service-mesh/istio-virtualservice.yaml

# Monitor canary metrics
# If successful, promote canary to stable
kubectl set image deployment/novacron-api api=novacron/api:3.1.0-canary
```

### Blue-Green Deployment

```bash
# Deploy green environment
kubectl apply -f novacron-green-deployment.yaml

# Test green environment
kubectl port-forward svc/novacron-api-green 8080:80

# Switch traffic to green
kubectl patch service novacron-api \
  -p '{"spec":{"selector":{"version":"green"}}}'

# Delete blue environment after validation
kubectl delete deployment novacron-api-blue
```

## Disaster Recovery

### Backup Strategy

**Automated backups with Velero:**

```bash
# Hourly incremental backups
Schedule: "0 * * * *"
Retention: 7 days

# Daily full backups
Schedule: "0 2 * * *"
Retention: 30 days

# Cross-region backups
Schedule: "0 3 * * *"
Retention: 90 days
```

**Manual backup:**

```bash
# Create on-demand backup
velero backup create novacron-manual-backup \
  --include-namespaces novacron \
  --snapshot-volumes \
  --ttl 720h

# Verify backup
velero backup describe novacron-manual-backup
```

### Recovery Procedures

#### Scenario 1: Complete Cluster Loss

```bash
# 1. Provision new cluster
eksctl create cluster -f cluster-config.yaml

# 2. Install Velero with same configuration
velero install \
  --provider aws \
  --bucket novacron-backups-us-east-1 \
  --backup-location-config region=us-east-1

# 3. Restore from backup
velero restore create --from-backup novacron-daily-full-backup-latest

# 4. Wait for restore
velero restore describe novacron-restore-20231115

# 5. Verify services
kubectl get all -n novacron
kubectl wait --for=condition=ready pod -l app=novacron-api -n novacron

# 6. Update DNS to point to new cluster
# (Manual step or automated via external-dns)
```

#### Scenario 2: Namespace Corruption

```bash
# Delete corrupted namespace
kubectl delete namespace novacron --force --grace-period=0

# Restore namespace from backup
velero restore create namespace-restore \
  --from-backup novacron-hourly-backup-latest \
  --include-namespaces novacron

# Verify restoration
kubectl get all -n novacron
```

#### Scenario 3: Data Corruption

```bash
# Restore volume snapshots
velero restore create data-restore \
  --from-backup novacron-daily-full-backup-<timestamp> \
  --include-resources persistentvolumeclaims,persistentvolumes

# Restart affected pods
kubectl rollout restart statefulset/novacron-consensus -n novacron
```

### DR Testing

**Monthly DR drill:**

```bash
# Create test namespace
kubectl create namespace dr-test

# Restore to test namespace
velero restore create dr-drill-$(date +%Y%m%d) \
  --from-backup novacron-daily-full-backup-latest \
  --namespace-mappings novacron:dr-test

# Validate restore
kubectl get all -n dr-test
kubectl exec -it -n dr-test deploy/novacron-api -- /health-check.sh

# Cleanup
kubectl delete namespace dr-test
```

## Troubleshooting

### Common Issues

#### 1. Pods Stuck in Pending

```bash
# Check resource availability
kubectl describe nodes | grep -A5 "Allocated resources"

# Check PVC status
kubectl get pvc -n novacron

# Check events
kubectl get events -n novacron --sort-by='.lastTimestamp'
```

#### 2. Consensus Quorum Lost

```bash
# Check consensus pod status
kubectl get pods -l app=novacron-consensus -n novacron

# View logs
kubectl logs -l app=novacron-consensus -n novacron --tail=100

# Force restart if needed
kubectl delete pod novacron-consensus-0 -n novacron
```

#### 3. High Latency

```bash
# Check metrics
kubectl port-forward -n monitoring svc/prometheus 9090:9090
# Open http://localhost:9090 and query:
# histogram_quantile(0.95, http_request_duration_seconds_bucket)

# Check resource usage
kubectl top pods -n novacron

# Scale up if needed
kubectl scale deployment novacron-api --replicas=20 -n novacron
```

#### 4. Network Connectivity Issues

```bash
# Check DWCP DaemonSet
kubectl get daemonset novacron-dwcp-network -n novacron

# Check network policies
kubectl get networkpolicies -n novacron

# Test connectivity
kubectl exec -it -n novacron deploy/novacron-api -- \
  curl -v http://novacron-consensus-0.novacron-consensus-headless:9090/health
```

### Debug Commands

```bash
# Get all resources
kubectl get all -n novacron

# Describe deployment
kubectl describe deployment novacron-api -n novacron

# View logs
kubectl logs -f -l app=novacron-api -n novacron --tail=100

# Execute shell in pod
kubectl exec -it -n novacron deploy/novacron-api -- /bin/bash

# Port forward for local access
kubectl port-forward -n novacron svc/novacron-api 8080:80

# Get events
kubectl get events -n novacron --watch

# Check resource usage
kubectl top nodes
kubectl top pods -n novacron
```

## Cost Optimization

### Resource Right-Sizing

**Review resource usage:**

```bash
# Install VPA (Vertical Pod Autoscaler)
kubectl apply -f https://github.com/kubernetes/autoscaler/releases/download/vertical-pod-autoscaler-0.14.0/vpa-v0.14.0.yaml

# Create VPA resource
kubectl apply -f - <<EOF
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: novacron-api-vpa
  namespace: novacron
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: novacron-api
  updatePolicy:
    updateMode: "Recommend"
EOF

# Get recommendations
kubectl get vpa novacron-api-vpa -n novacron -o yaml
```

### Spot Instances

**Use spot instances for non-critical workloads:**

```yaml
# Node affinity for spot instances
affinity:
  nodeAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 50
        preference:
          matchExpressions:
            - key: node.kubernetes.io/instance-type
              operator: In
              values:
                - spot
```

### Storage Optimization

```bash
# Review PVC usage
kubectl get pvc -A -o custom-columns=NAME:.metadata.name,NAMESPACE:.metadata.namespace,SIZE:.spec.resources.requests.storage,USED:.status.capacity.storage

# Resize PVCs
kubectl patch pvc data-novacron-consensus-0 -n novacron \
  -p '{"spec":{"resources":{"requests":{"storage":"200Gi"}}}}'
```

## Security Best Practices

1. **Enable mTLS in service mesh**
2. **Use External Secrets Operator for secret management**
3. **Implement NetworkPolicies for pod-to-pod communication**
4. **Enable Pod Security Standards**
5. **Regular security scans with Trivy/Snyk**
6. **RBAC with least privilege principle**
7. **Audit logging enabled**

## Performance Tuning

### Database Optimization

```yaml
postgresql:
  primary:
    resources:
      requests:
        cpu: "8"
        memory: "16Gi"
    configuration: |
      max_connections = 1000
      shared_buffers = 4GB
      effective_cache_size = 12GB
      work_mem = 16MB
      maintenance_work_mem = 2GB
```

### Redis Tuning

```yaml
redis:
  master:
    configuration: |
      maxmemory 8gb
      maxmemory-policy allkeys-lru
      save ""  # Disable persistence for cache
```

### DWCP Optimization

```yaml
dwcp:
  amst:
    minStreams: 32
    maxStreams: 512
  hde:
    compressionLevel: 9
    algorithm: zstd
  networking:
    mtu: 9000  # Jumbo frames
    tcp_nodelay: true
```

## Monitoring and Alerting

### Key Metrics

- **API Server:** Request rate, error rate, latency (P50, P95, P99)
- **Consensus:** Commit latency, leader elections, quorum status
- **DWCP:** Throughput, compression ratio, packet loss
- **Resources:** CPU, memory, disk usage

### Alert Rules

See `backend/deployments/k8s/monitoring/prometheus-servicemonitor.yaml` for complete alert definitions.

## Support

For issues or questions:
- GitHub Issues: https://github.com/novacron/novacron/issues
- Documentation: https://docs.novacron.io
- Slack: #novacron-kubernetes

## Appendix

### File Structure

```
backend/deployments/k8s/
├── operator/
│   ├── novacron_operator.go
│   ├── crd_novacroncluster.yaml
│   ├── crd_novacronregion.yaml
│   └── crd_dwcpfederation.yaml
├── charts/novacron/
│   ├── Chart.yaml
│   ├── values.yaml
│   ├── values-production.yaml
│   ├── values-staging.yaml
│   └── templates/
│       └── deployment.yaml
├── manifests/
│   ├── consensus-statefulset.yaml
│   └── network-daemonset.yaml
├── autoscaling/
│   ├── hpa.yaml
│   └── cluster-autoscaler.yaml
├── service-mesh/
│   └── istio-virtualservice.yaml
├── argocd/
│   ├── novacron-core.yaml
│   ├── novacron-dwcp.yaml
│   └── novacron-monitoring.yaml
├── strategies/
│   └── rolling-update.yaml
├── federation/
│   └── kubefed-config.yaml
├── storage/
│   └── storage-classes.yaml
├── secrets/
│   └── external-secrets.yaml
├── network/
│   └── network-policies.yaml
├── ci/
│   └── github-workflows-deploy.yml
├── dr/
│   └── velero-backup.yaml
├── monitoring/
│   ├── prometheus-servicemonitor.yaml
│   └── jaeger-deployment.yaml
└── tests/
    └── integration-tests.yaml
```

### Glossary

- **CRD:** Custom Resource Definition
- **HPA:** Horizontal Pod Autoscaler
- **VPA:** Vertical Pod Autoscaler
- **PDB:** Pod Disruption Budget
- **PVC:** Persistent Volume Claim
- **GitOps:** Git-based operations and deployment
- **DWCP:** Distributed WAN Communication Protocol
- **AMST:** Adaptive Multi-Stream Transport
- **HDE:** Hierarchical Data Encoding
- **ACP:** Adaptive Consensus Protocol

---

**Version:** 3.0.0
**Last Updated:** 2025-11-09
**Authors:** NovaCron Platform Team
