# NovaCron Kubernetes Deployment - DWCP Phase 3

Production-ready Kubernetes deployment automation for global multi-region NovaCron deployments.

## Quick Start

### Deploy to Staging

\`\`\`bash
helm install novacron ./charts/novacron \
  --namespace novacron \
  --create-namespace \
  --values ./charts/novacron/values-staging.yaml
\`\`\`

### Deploy to Production (Multi-Region)

\`\`\`bash
# Install CRDs first
kubectl apply -f operator/crd_*.yaml

# Deploy operator
kubectl apply -f operator/operator-deployment.yaml

# Create cluster
kubectl apply -f - <<YAML
apiVersion: novacron.io/v1
kind: NovaCronCluster
metadata:
  name: production
spec:
  version: "3.0.0"
  regions:
    - name: us-east-1
      replicas: 20
    - name: eu-west-1
      replicas: 15
    - name: ap-southeast-1
      replicas: 10
  dwcp:
    enabled: true
YAML
\`\`\`

## Directory Structure

- **operator/** - Kubernetes operator and CRDs
- **charts/** - Helm charts
- **manifests/** - Kubernetes manifests
- **autoscaling/** - HPA and Cluster Autoscaler
- **service-mesh/** - Istio configuration
- **argocd/** - GitOps applications
- **strategies/** - Deployment strategies
- **federation/** - Multi-region federation
- **storage/** - Persistent storage
- **secrets/** - Secret management
- **network/** - Network policies
- **ci/** - CI/CD pipelines
- **dr/** - Disaster recovery
- **monitoring/** - Observability stack
- **tests/** - Integration tests

## Documentation

See [DWCP_KUBERNETES_DEPLOYMENT.md](../../docs/DWCP_KUBERNETES_DEPLOYMENT.md) for comprehensive guide.

## Features

✅ Multi-region Kubernetes deployment
✅ GitOps with ArgoCD
✅ Zero-downtime rolling updates
✅ Auto-scaling (HPA + Cluster Autoscaler)
✅ Service mesh (Istio)
✅ Disaster recovery (Velero)
✅ Observability (Prometheus, Grafana, Jaeger)
✅ CI/CD pipeline
✅ Security (mTLS, NetworkPolicies, External Secrets)

## Support

- Documentation: https://docs.novacron.io
- GitHub: https://github.com/novacron/novacron
- Issues: https://github.com/novacron/novacron/issues
