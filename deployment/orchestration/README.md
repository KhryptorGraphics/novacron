# NovaCron Orchestration Production Deployment

This directory contains everything needed to deploy the NovaCron advanced orchestration system to production Kubernetes environments.

## Overview

The NovaCron orchestration system provides:

- **VM Placement Intelligence**: ML-powered placement decisions with multi-criteria optimization
- **Auto-Scaling**: Predictive scaling based on resource utilization and performance patterns
- **Self-Healing**: Automated failure detection and recovery mechanisms
- **Policy Management**: Flexible policy engine with constraint-based rules
- **Performance Tuning**: ML model optimization with A/B testing and continuous learning
- **Comprehensive Monitoring**: Real-time metrics, alerting, and observability

## Architecture

```
┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Frontend UI       │    │  Orchestration   │    │   ML Training       │
│   (React/Next.js)   │────│     Engine       │────│    Service          │
└─────────────────────┘    └──────────────────┘    └─────────────────────┘
                                      │
                           ┌──────────┼──────────┐
                           │          │          │
                    ┌─────────┐ ┌─────────┐ ┌─────────┐
                    │ Storage │ │  Event  │ │   ML    │
                    │  (PVC)  │ │  Bus    │ │ Models  │
                    └─────────┘ └─────────┘ └─────────┘
```

## Quick Start

### Prerequisites

- Kubernetes 1.20+
- kubectl configured for target cluster
- Helm 3.x (optional but recommended)
- Storage class supporting ReadWriteMany (for ML models)
- Prometheus Operator (for monitoring)

### Basic Deployment

```bash
# Clone the repository
git clone https://github.com/your-org/novacron.git
cd novacron/deployment/orchestration

# Configure secrets (REQUIRED - see Security section)
cp kubernetes/secrets.yaml.example kubernetes/secrets.yaml
# Edit kubernetes/secrets.yaml with your actual secrets

# Deploy to production
./scripts/deploy.sh

# Or use environment variables for customization
DEPLOYMENT_ENV=staging MONITORING_ENABLED=true ./scripts/deploy.sh
```

### Verification

```bash
# Check deployment status
kubectl get pods -n novacron-orchestration

# View service endpoints
kubectl get services -n novacron-orchestration

# Check orchestration engine health
kubectl port-forward -n novacron-orchestration svc/orchestration-engine 8080:80
curl http://localhost:8080/health
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEPLOYMENT_ENV` | `production` | Deployment environment (production/staging/development) |
| `DRY_RUN` | `false` | Enable dry-run mode for testing |
| `SKIP_SECRETS` | `false` | Skip secrets deployment (for CI/CD) |
| `MONITORING_ENABLED` | `true` | Enable Prometheus monitoring |
| `PRESERVE_DATA` | `true` | Preserve data during cleanup |

### Orchestration Configuration

The system is configured via ConfigMaps in `kubernetes/configmap.yaml`:

```yaml
orchestration:
  engine_type: default
  placement_strategy: balanced
  max_concurrent_decisions: 100
  decision_timeout: 30s

ml_models:
  training_enabled: true
  benchmark_enabled: true
  model_retention_days: 90
  
monitoring:
  metrics_enabled: true
  tracing_enabled: true
  prometheus_metrics: true
```

### ML Model Configuration

ML models are configured in the `ml-config` ConfigMap:

```yaml
performance_thresholds:
  min_accuracy: 0.85
  max_latency_ms: 100
  min_throughput_rps: 100

retraining_triggers:
  accuracy_drop: 0.05
  data_drift_threshold: 0.1
  time_since_last_training: "168h"
```

## Security

### Secrets Management

**CRITICAL**: Never commit actual secrets to version control.

1. Copy the secrets template:
   ```bash
   cp kubernetes/secrets.yaml.example kubernetes/secrets.yaml
   ```

2. Update with real values:
   ```bash
   # Edit secrets.yaml and replace all CHANGE_ME values
   vim kubernetes/secrets.yaml
   ```

3. Use external secret management (recommended):
   ```bash
   # Using External Secrets Operator with Vault
   kubectl apply -f kubernetes/external-secrets.yaml
   ```

### RBAC

The deployment creates minimal RBAC permissions:

- Service account: `orchestration-service`
- Cluster role: `orchestration-operator`
- Permissions: Read-only cluster access, limited write access to managed resources

### Network Policies

For additional security, deploy network policies:

```bash
kubectl apply -f security/network-policies.yaml
```

### Pod Security Standards

All pods run with security context:
- Non-root user (UID 1000)
- Read-only root filesystem
- No privilege escalation
- Dropped capabilities

## Storage

### Persistent Volumes

The system requires three persistent volumes:

1. **ML Models PVC** (100Gi, ReadWriteMany)
   - Stores trained ML models
   - Shared across orchestration engines

2. **Orchestration Data PVC** (50Gi, ReadWriteOnce)
   - Decision history and state
   - Per-instance storage

3. **Training Data PVC** (200Gi, ReadWriteMany)
   - Training datasets
   - Shared with ML training service

### Storage Classes

Configure appropriate storage classes:

```yaml
# Fast SSD for performance-critical data
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/aws-ebs
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
```

## Monitoring and Alerting

### Prometheus Metrics

Key metrics exposed:

- `orchestration_decisions_total` - Total decisions made
- `orchestration_decision_duration_seconds` - Decision latency
- `ml_model_accuracy` - Model accuracy scores
- `orchestration_policy_violations_total` - Policy violations

### Grafana Dashboard

Import the dashboard from `monitoring/grafana-dashboard.json`:

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload `monitoring/grafana-dashboard.json`
4. Configure data source as your Prometheus instance

### Alerting Rules

Critical alerts configured:

- **OrchestrationEngineDown**: Engine unavailable
- **OrchestrationHighErrorRate**: >5% error rate
- **MLModelAccuracyDrop**: Model accuracy <80%
- **OrchestrationPolicyViolation**: Policy violations detected

## Scaling

### Horizontal Scaling

Scale orchestration engines:

```bash
kubectl scale deployment orchestration-engine -n novacron-orchestration --replicas=5
```

### Resource Limits

Configure appropriate resource limits:

```yaml
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi
```

### ML Training Resources

ML training requires GPU resources:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
    cpu: 8000m
    memory: 16Gi
```

## Troubleshooting

### Common Issues

1. **Pods stuck in Pending**
   ```bash
   kubectl describe pod <pod-name> -n novacron-orchestration
   # Check resource constraints and node selectors
   ```

2. **Health checks failing**
   ```bash
   kubectl logs <pod-name> -n novacron-orchestration
   # Check application logs for errors
   ```

3. **ML models not loading**
   ```bash
   kubectl exec -it <pod-name> -n novacron-orchestration -- ls -la /var/lib/novacron/ml-models
   # Verify PVC mounts and permissions
   ```

### Debug Mode

Enable debug logging:

```bash
kubectl set env deployment/orchestration-engine LOG_LEVEL=debug -n novacron-orchestration
```

### Port Forwarding

Access services locally:

```bash
# Orchestration API
kubectl port-forward -n novacron-orchestration svc/orchestration-engine 8080:80

# Metrics
kubectl port-forward -n novacron-orchestration svc/orchestration-engine 9090:9090
```

## Backup and Recovery

### Data Backup

Automated backup during cleanup:

```bash
# Backup is created automatically
./scripts/cleanup.sh
```

Manual backup:

```bash
# Backup ML models
kubectl exec -it <pod-name> -n novacron-orchestration -- tar czf - -C /var/lib/novacron/ml-models . > ml-models-backup.tar.gz

# Backup configuration
kubectl get configmap -n novacron-orchestration -o yaml > config-backup.yaml
```

### Disaster Recovery

1. Restore storage volumes
2. Apply configuration backups
3. Redeploy application

```bash
kubectl apply -f config-backup.yaml
./scripts/deploy.sh
```

## Upgrading

### Rolling Upgrades

Update image version:

```bash
kubectl set image deployment/orchestration-engine orchestration-engine=novacron/orchestration:1.1.0 -n novacron-orchestration
```

### Blue-Green Deployment

For zero-downtime upgrades:

1. Deploy new version with different selector
2. Update service to point to new version
3. Remove old version after verification

### ML Model Updates

Models support A/B testing:

```bash
# Enable A/B testing in configuration
kubectl patch configmap ml-config -n novacron-orchestration --patch '{"data":{"ab_testing_enabled":"true"}}'
```

## Production Checklist

- [ ] Secrets configured with real values
- [ ] Storage classes configured
- [ ] Resource limits set appropriately
- [ ] Monitoring and alerting configured
- [ ] RBAC policies reviewed
- [ ] Network policies applied
- [ ] Backup strategy implemented
- [ ] Disaster recovery plan documented
- [ ] Performance benchmarks established
- [ ] Load testing completed

## Support

### Logs

```bash
# All orchestration logs
kubectl logs -n novacron-orchestration -l app.kubernetes.io/name=novacron-orchestration -f

# Specific component logs
kubectl logs -n novacron-orchestration -l app.kubernetes.io/component=orchestration-engine -f
kubectl logs -n novacron-orchestration -l app.kubernetes.io/component=ml-training -f
```

### Runbooks

Detailed runbooks available in `runbooks/`:
- [Engine Down](runbooks/engine-down.md)
- [High Error Rate](runbooks/high-error-rate.md)  
- [ML Model Issues](runbooks/ml-model-issues.md)
- [Performance Problems](runbooks/performance-problems.md)

### Emergency Contacts

- **Platform Team**: platform-team@company.com
- **On-Call**: +1-555-ON-CALL-1
- **Slack**: #novacron-support

## Cleanup

Remove the entire deployment:

```bash
# Preserve data (default)
./scripts/cleanup.sh

# Delete everything including data
PRESERVE_DATA=false ./scripts/cleanup.sh

# Force delete without confirmation
FORCE_DELETE=true ./scripts/cleanup.sh
```

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for development guidelines.

## License

See [LICENSE](../../LICENSE) for license information.