# NovaCron Enhanced Kubernetes Operator

The enhanced NovaCron Kubernetes Operator provides advanced multi-cloud federation and AI-powered operations for managing virtualized workloads across distributed environments.

## Features

### Core Capabilities
- **VirtualMachine Management**: Create, manage, and monitor virtual machines
- **VM Templates**: Reusable VM configuration templates
- **VM Clusters**: Scalable clusters of VMs with load balancing

### Enhanced Multi-Cloud Features
- **Multi-Cloud VMs**: Deploy VMs across multiple cloud providers (AWS, Azure, GCP)
- **Federated Clusters**: Manage clusters spanning multiple cloud environments
- **Cost Optimization**: AI-powered cost optimization with spot instance support
- **Cross-Cloud Migration**: Live, warm, and cold migration across providers
- **Disaster Recovery**: Automated backup and recovery across regions

### AI-Powered Operations
- **AI Scheduling Policies**: Machine learning-based placement decisions
- **Performance Optimization**: Predictive scaling and resource allocation
- **Anomaly Detection**: AI-powered monitoring and alerting
- **Cost Prediction**: ML models for cost forecasting and optimization

### Cache Integration
- **Redis Cluster Support**: High-performance caching with Redis
- **Multi-Level Caching**: L1/L2 cache hierarchies
- **Cache Warming**: Intelligent pre-loading strategies
- **Performance Monitoring**: Real-time cache metrics and optimization

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Kubernetes    │    │  NovaCron        │    │  Multi-Cloud    │
│   API Server    │◄──►│  Enhanced        │◄──►│  Providers      │
│                 │    │  Operator        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────┐
                       │ AI Engine   │
                       │ & Cache     │
                       │ Manager     │
                       └─────────────┘
```

## Quick Start

### Prerequisites
- Kubernetes cluster (v1.20+)
- kubectl configured
- Docker (for building images)

### Installation

1. **Deploy the operator**:
   ```bash
   ./deploy/deploy-enhanced-operator.sh
   ```

2. **Deploy with examples**:
   ```bash
   ./deploy/deploy-enhanced-operator.sh --examples
   ```

3. **Verify installation**:
   ```bash
   kubectl get pods -n novacron-system
   kubectl get crds | grep novacron.io
   ```

### Quick Example

Create a multi-cloud VM:

```yaml
apiVersion: novacron.io/v1
kind: MultiCloudVM
metadata:
  name: my-app
spec:
  deploymentStrategy:
    type: "active-passive"
    primary: "aws-us-east"
    secondary: ["azure-eastus"]
  providers:
  - name: "aws-us-east"
    region: "us-east-1"
    credentialsSecret: "aws-credentials"
  vmTemplate:
    name: "web-server-template"
```

## Custom Resource Definitions

### VirtualMachine
Standard single-cloud VM management with NovaCron backend integration.

### MultiCloudVM  
Deploy and manage VMs across multiple cloud providers with:
- Active-active, active-passive, burst, and cost-optimized strategies
- Automatic failover and disaster recovery
- Cost optimization with spot instances
- Cross-cloud migration capabilities

### FederatedCluster
Manage Kubernetes clusters across multiple cloud providers:
- Cross-cluster networking and service mesh
- Federated data replication
- Load balancing across clusters
- Unified management interface

### AISchedulingPolicy
AI-powered scheduling and placement decisions:
- Neural networks, decision trees, reinforcement learning
- Multi-objective optimization (cost, performance, availability)
- Real-time data integration (Prometheus, CloudWatch, etc.)
- Online learning and model retraining

### CacheIntegration
Redis-based caching with advanced features:
- Multi-level cache hierarchies  
- Intelligent warming strategies
- Performance monitoring and optimization
- High availability with sentinel support

## Configuration

### Cloud Provider Credentials

Create secrets for each cloud provider:

```bash
# AWS
kubectl create secret generic aws-credentials \
  --from-literal=access_key_id=YOUR_ACCESS_KEY \
  --from-literal=secret_access_key=YOUR_SECRET_KEY \
  -n novacron-system

# Azure  
kubectl create secret generic azure-credentials \
  --from-literal=client_id=YOUR_CLIENT_ID \
  --from-literal=client_secret=YOUR_CLIENT_SECRET \
  --from-literal=tenant_id=YOUR_TENANT_ID \
  --from-literal=subscription_id=YOUR_SUBSCRIPTION_ID \
  -n novacron-system

# GCP
kubectl create secret generic gcp-credentials \
  --from-literal=project_id=YOUR_PROJECT_ID \
  --from-literal=service_account_key=YOUR_SERVICE_ACCOUNT_JSON \
  -n novacron-system
```

### NovaCron API Integration

Update the API token secret:

```bash
kubectl patch secret novacron-api-token -n novacron-system \
  -p '{"data":{"token":"'$(echo -n "your-api-token" | base64)'"}}'
```

## Monitoring and Observability

### Prometheus Integration

The operator exposes metrics on port 8080:
- `novacron_vm_total` - Total VMs by status
- `novacron_multicloud_vm_deployments` - Multi-cloud deployments by provider
- `novacron_ai_scheduling_accuracy` - AI model accuracy
- `novacron_cache_hit_rate` - Cache performance metrics

### Grafana Dashboard

Import the provided dashboard:
```bash
kubectl create configmap grafana-dashboard-novacron \
  --from-file=deploy/monitoring/grafana-dashboard.json \
  -n monitoring
```

### Alerts

Deploy Prometheus alerts:
```bash
kubectl apply -f deploy/monitoring/servicemonitor.yaml
```

## Development

### Building

```bash
# Build binary
make build

# Build Docker image  
make docker-build

# Run tests
make test

# Deploy for development
make deploy
```

### Testing

```bash
# Unit tests
make test-unit

# Integration tests (requires cluster)
make test-integration

# Create test cluster
make kind-cluster
make kind-load-image
make deploy
```

## Examples

### Multi-Cloud Deployment

```yaml
apiVersion: novacron.io/v1
kind: MultiCloudVM
metadata:
  name: web-app
spec:
  deploymentStrategy:
    type: "cost-optimized"
    primary: "aws-us-east"
    secondary: ["gcp-us-central", "azure-eastus"]
  costOptimization:
    useSpotInstances: true
    maxCostPerHour: "1.00"
    costBasedScaling: true
```

### AI Scheduling

```yaml
apiVersion: novacron.io/v1  
kind: AISchedulingPolicy
metadata:
  name: performance-optimizer
spec:
  modelConfig:
    modelType: "neural-network"
  objectives:
  - type: "performance"
    weight: 0.6
  - type: "cost"
    weight: 0.4
  dataSources:
  - type: "prometheus"
    connection:
      url: "http://prometheus:9090"
```

### Cache Integration

```yaml
apiVersion: novacron.io/v1
kind: CacheIntegration  
metadata:
  name: redis-cache
spec:
  redisConfig:
    endpoints: ["redis:6379"]
  strategy:
    type: "write-through"
  warmingConfig:
    enabled: true
```

## Troubleshooting

### Common Issues

1. **Operator not starting**:
   ```bash
   kubectl logs -n novacron-system -l app=novacron-operator
   ```

2. **CRDs not found**:
   ```bash
   kubectl apply -f deploy/crds/
   ```

3. **RBAC permissions**:
   ```bash
   kubectl apply -f deploy/rbac/
   ```

4. **Cloud provider authentication**:
   ```bash
   kubectl get secrets -n novacron-system
   kubectl describe secret aws-credentials -n novacron-system
   ```

### Debugging

Enable debug logging:
```yaml
spec:
  containers:
  - name: manager
    args:
    - --zap-log-level=debug
```

View detailed status:
```bash
kubectl describe multicloudvm my-app
kubectl get events --sort-by=.lastTimestamp
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make changes and add tests
4. Run tests: `make test`
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Create a Pull Request

### Development Environment

```bash
# Install tools
make install-tools

# Format and lint
make fmt lint

# Test changes
make test-unit test-integration
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Security**: security@novacron.io

## Roadmap

- [ ] Support for additional cloud providers (DigitalOcean, Linode)
- [ ] Advanced AI models (reinforcement learning, GANs)
- [ ] Service mesh integration
- [ ] GitOps workflow integration
- [ ] Cost allocation and chargeback
- [ ] Compliance and governance policies

---

**NovaCron Enhanced Kubernetes Operator v2.0.0** - Transforming multi-cloud VM management with AI-powered operations.