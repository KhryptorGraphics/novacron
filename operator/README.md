# NovaCron Kubernetes Operator

The NovaCron Kubernetes Operator enables management of virtual machines as native Kubernetes resources, providing a cloud-native approach to VM orchestration.

## Features

- **Native Kubernetes Integration**: Manage VMs using kubectl and Kubernetes APIs
- **Declarative VM Management**: Define VM specifications using YAML manifests
- **Live Migration Support**: Seamlessly migrate VMs between nodes
- **Automated Snapshots**: Schedule and manage VM snapshots
- **Multi-Cloud Support**: Deploy VMs across different cloud providers
- **Resource Management**: CPU, memory, disk, and GPU resource allocation
- **Network Configuration**: Flexible networking with multiple interface support
- **Lifecycle Hooks**: Pre-start and pre-stop hooks for custom logic

## Installation

### Prerequisites

- Kubernetes cluster (v1.25+)
- kubectl configured to access your cluster
- cert-manager (for webhook certificates)

### Install the Operator

1. Install the CRDs:
```bash
kubectl apply -f config/crd/bases/
```

2. Deploy the operator:
```bash
kubectl apply -f config/manager/
```

3. Verify installation:
```bash
kubectl get pods -n novacron-system
kubectl get crd virtualmachines.novacron.io
```

## Usage

### Create a Virtual Machine

```yaml
apiVersion: novacron.io/v1alpha1
kind: VirtualMachine
metadata:
  name: my-vm
  namespace: default
spec:
  running: true
  template:
    spec:
      resources:
        cpu: 2
        memory: 4Gi
        disk: 50Gi
      image:
        source: "marketplace://ubuntu-22.04-lts"
      networks:
        - name: default
          type: bridge
          ipv4:
            method: dhcp
```

Apply the manifest:
```bash
kubectl apply -f my-vm.yaml
```

### Manage VMs

```bash
# List VMs
kubectl get virtualmachines
kubectl get vm  # short name

# Get VM details
kubectl describe vm my-vm

# Watch VM status
kubectl get vm -w

# Delete VM
kubectl delete vm my-vm
```

### VM Operations

#### Start/Stop VM

```bash
# Stop VM
kubectl patch vm my-vm --type merge -p '{"spec":{"running":false}}'

# Start VM
kubectl patch vm my-vm --type merge -p '{"spec":{"running":true}}'
```

#### Live Migration

```yaml
apiVersion: novacron.io/v1alpha1
kind: VirtualMachine
metadata:
  name: my-vm
spec:
  migrationPolicy:
    allowLiveMigration: true
    compression: true
    encrypted: true
    bandwidth: "10Gbps"
```

#### Snapshots

```yaml
apiVersion: novacron.io/v1alpha1
kind: VirtualMachine
metadata:
  name: my-vm
spec:
  snapshotPolicy:
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention: 7  # Keep 7 snapshots
```

## Advanced Features

### GPU Support

```yaml
spec:
  template:
    spec:
      devices:
        - type: gpu
          vendor: nvidia
          model: tesla-t4
          count: 1
```

### Multiple Network Interfaces

```yaml
spec:
  template:
    spec:
      networks:
        - name: management
          type: bridge
          ipv4:
            method: static
            address: 10.0.1.10/24
            gateway: 10.0.1.1
        - name: data
          type: sr-iov
          ipv4:
            method: dhcp
```

### Cloud-Init Configuration

```yaml
spec:
  template:
    spec:
      userData: |
        #cloud-config
        users:
          - name: admin
            ssh_authorized_keys:
              - ssh-rsa AAAAB3...
        packages:
          - docker.io
          - nginx
        runcmd:
          - docker run -d nginx
```

### Node Affinity

```yaml
spec:
  template:
    spec:
      nodeSelector:
        node.novacron.io/type: compute
        node.novacron.io/gpu: "true"
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: node.novacron.io/region
                operator: In
                values:
                - us-east-1
                - us-west-2
```

## Architecture

The operator follows the standard Kubernetes controller pattern:

1. **CRD Definition**: Defines the VirtualMachine custom resource
2. **Controller**: Watches for VirtualMachine resources and reconciles state
3. **Provider Interface**: Abstracts underlying VM management (KVM, VMware, Cloud)
4. **Webhook**: Validates and mutates VirtualMachine resources

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   kubectl    │────▶│  Kubernetes  │────▶│   NovaCron   │
│              │     │     API      │     │   Operator   │
└──────────────┘     └──────────────┘     └──────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  VM Providers   │
                                          ├─────────────────┤
                                          │ KVM │ VMware │  │
                                          │ AWS │ Azure  │  │
                                          └─────────────────┘
```

## Development

### Prerequisites

- Go 1.21+
- Kubebuilder 3.0+
- Docker
- Kind or Minikube for local testing

### Building

```bash
# Generate code
make generate

# Build operator
make build

# Build Docker image
make docker-build IMG=novacron/operator:latest

# Push Docker image
make docker-push IMG=novacron/operator:latest
```

### Testing

```bash
# Run unit tests
make test

# Run integration tests
make test-integration

# Run e2e tests
make test-e2e
```

### Local Development

```bash
# Install CRDs
make install

# Run operator locally
make run

# Deploy to cluster
make deploy
```

## Monitoring

The operator exposes Prometheus metrics:

- `novacron_vm_total`: Total number of VMs by phase
- `novacron_vm_cpu_usage_cores`: CPU usage per VM
- `novacron_vm_memory_usage_bytes`: Memory usage per VM
- `novacron_reconcile_total`: Total reconciliations
- `novacron_reconcile_duration_seconds`: Reconciliation duration

## Troubleshooting

### Check Operator Logs

```bash
kubectl logs -n novacron-system deployment/novacron-controller-manager
```

### Check VM Events

```bash
kubectl describe vm my-vm
kubectl get events --field-selector involvedObject.name=my-vm
```

### Common Issues

1. **VM Stuck in Pending**
   - Check node capacity: `kubectl describe nodes`
   - Check operator logs for errors
   - Verify provider connectivity

2. **Migration Failed**
   - Check network connectivity between nodes
   - Verify migration policy settings
   - Check source and target node resources

3. **VM Won't Start**
   - Check image availability
   - Verify resource requests
   - Check node selector constraints

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## Support

- Documentation: https://docs.novacron.io
- Issues: https://github.com/novacron/operator/issues
- Slack: https://novacron.slack.com
- Email: support@novacron.io