# NovaCron Kubernetes Operator Specification

## Overview

The NovaCron Kubernetes Operator extends Kubernetes to manage virtual machines as native resources, enabling declarative VM lifecycle management alongside container workloads.

## Custom Resource Definitions (CRDs)

### 1. VirtualMachine CRD

```yaml
apiVersion: novacron.io/v1alpha1
kind: VirtualMachine
metadata:
  name: example-vm
  namespace: default
spec:
  # VM Specification
  template:
    metadata:
      labels:
        app: web-server
    spec:
      # Compute Resources
      resources:
        cpu: 4
        memory: 8Gi
        disk: 100Gi
      
      # VM Image
      image:
        source: "marketplace://ubuntu-22.04-lts"
        # Alternative sources:
        # source: "http://images.example.com/ubuntu.qcow2"
        # source: "s3://bucket/path/to/image.vmdk"
        # source: "pvc://namespace/pvc-name"
      
      # Network Configuration
      networks:
        - name: default
          type: bridge
          ipv4:
            method: dhcp
        - name: management
          type: ovs
          ipv4:
            method: static
            address: 10.0.1.10/24
            gateway: 10.0.1.1
      
      # Storage Volumes
      volumes:
        - name: data
          size: 500Gi
          storageClass: fast-ssd
          accessMode: ReadWriteOnce
      
      # GPU/Accelerator Support
      devices:
        - type: gpu
          vendor: nvidia
          model: "tesla-t4"
          count: 1
      
      # Cloud-init Configuration
      userData: |
        #cloud-config
        users:
          - name: admin
            ssh_authorized_keys:
              - ssh-rsa AAAAB3...
      
      # Placement Constraints
      nodeSelector:
        node.novacron.io/type: compute
        node.novacron.io/gpu: "true"
      
      # Affinity Rules
      affinity:
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            - labelSelector:
                matchLabels:
                  app: database
      
      # Lifecycle Hooks
      lifecycle:
        preStart:
          exec:
            command: ["/hooks/pre-start.sh"]
        preStop:
          exec:
            command: ["/hooks/graceful-shutdown.sh"]
            timeoutSeconds: 300

  # VM Behavior Configuration
  running: true  # Desired state
  
  # Migration Policy
  migrationPolicy:
    allowLiveMigration: true
    compression: true
    encrypted: true
    bandwidth: "1Gbps"
  
  # Snapshot Policy
  snapshotPolicy:
    schedule: "0 2 * * *"  # Daily at 2 AM
    retention: 7
  
  # Update Strategy
  updateStrategy:
    type: RollingUpdate  # or Recreate
    rollingUpdate:
      maxUnavailable: 1

status:
  # Current State
  phase: Running  # Pending, Running, Migrating, Stopped, Failed
  
  # Runtime Information
  nodeName: node-01
  ipAddresses:
    - 10.0.0.5
    - 192.168.1.100
  
  # Resource Usage
  resources:
    cpu:
      used: "2.3"
      available: "4"
    memory:
      used: "5.2Gi"
      available: "8Gi"
    disk:
      used: "45Gi"
      available: "100Gi"
  
  # Conditions
  conditions:
    - type: Ready
      status: "True"
      lastProbeTime: "2024-01-20T10:00:00Z"
      lastTransitionTime: "2024-01-20T09:00:00Z"
      reason: VMReady
      message: "VM is running and ready"
    - type: LiveMigratable
      status: "True"
      reason: MigrationCapable
      message: "VM can be live migrated"
  
  # Migration Status
  migration:
    targetNode: ""
    startTime: null
    progress: 0
    
  # Snapshots
  snapshots:
    - name: "snapshot-20240120-020000"
      createdAt: "2024-01-20T02:00:00Z"
      size: "42Gi"
```

### 2. VMPool CRD (Auto-scaling Group)

```yaml
apiVersion: novacron.io/v1alpha1
kind: VMPool
metadata:
  name: web-server-pool
spec:
  replicas: 3
  minReplicas: 2
  maxReplicas: 10
  
  # VM Template
  template:
    metadata:
      labels:
        app: web-server
        tier: frontend
    spec:
      # Same as VirtualMachine spec
      resources:
        cpu: 2
        memory: 4Gi
  
  # Auto-scaling Configuration
  autoscaling:
    enabled: true
    metrics:
      - type: cpu
        targetAverageUtilization: 70
      - type: memory
        targetAverageUtilization: 80
      - type: custom
        metric:
          name: requests_per_second
          selector:
            matchLabels:
              app: web-server
        targetValue: "1000"
    
    # Predictive Scaling
    predictive:
      enabled: true
      model: arima  # or lstm, prophet
      lookbackWindow: 7d
      forecastHorizon: 1h
    
    # Scaling Behavior
    behavior:
      scaleUp:
        stabilizationWindowSeconds: 60
        policies:
          - type: Pods
            value: 2
            periodSeconds: 60
      scaleDown:
        stabilizationWindowSeconds: 300
        policies:
          - type: Percent
            value: 10
            periodSeconds: 60
  
  # Update Strategy
  updateStrategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
      partition: 0
  
status:
  replicas: 3
  readyReplicas: 3
  availableReplicas: 3
  unavailableReplicas: 0
  updatedReplicas: 3
  
  # Individual VM Status
  vms:
    - name: web-server-pool-abc123
      phase: Running
      node: node-01
    - name: web-server-pool-def456
      phase: Running
      node: node-02
    - name: web-server-pool-ghi789
      phase: Running
      node: node-03
```

### 3. VMNetwork CRD

```yaml
apiVersion: novacron.io/v1alpha1
kind: VMNetwork
metadata:
  name: production-network
spec:
  # Network Type
  type: overlay  # bridge, ovs, sr-iov, macvlan
  
  # Network Configuration
  config:
    driver: vxlan
    vni: 100
    mtu: 1450
  
  # Subnet Configuration
  subnets:
    - name: default
      cidr: 10.0.0.0/24
      gateway: 10.0.0.1
      dhcp:
        enabled: true
        range:
          start: 10.0.0.100
          end: 10.0.0.200
        options:
          - name: dns-server
            value: "8.8.8.8,8.8.4.4"
          - name: domain-name
            value: "novacron.local"
  
  # Security Policies
  policies:
    - name: allow-web
      priority: 100
      direction: ingress
      protocol: tcp
      ports: [80, 443]
      source: 0.0.0.0/0
    - name: allow-ssh
      priority: 200
      direction: ingress
      protocol: tcp
      ports: [22]
      source: 10.0.0.0/8
  
  # QoS Configuration
  qos:
    bandwidth:
      ingress: "10Gbps"
      egress: "10Gbps"
    priority: high
  
status:
  phase: Active
  allocatedIPs: 45
  availableIPs: 211
  connectedVMs: 45
```

### 4. VMStorage CRD

```yaml
apiVersion: novacron.io/v1alpha1
kind: VMStorage
metadata:
  name: shared-storage
spec:
  # Storage Type
  type: distributed  # local, nfs, ceph, glusterfs
  
  # Storage Configuration
  config:
    backend: ceph
    pool: vms
    replication: 3
  
  # Capacity
  capacity: 10Ti
  
  # Access Modes
  accessModes:
    - ReadWriteMany
  
  # Performance Tier
  performanceTier: ssd  # hdd, ssd, nvme
  
  # Encryption
  encryption:
    enabled: true
    provider: vault
    keyRotation: 30d
  
  # Snapshot Configuration
  snapshots:
    enabled: true
    schedule: "0 */6 * * *"  # Every 6 hours
    retention:
      daily: 7
      weekly: 4
      monthly: 3
  
  # Replication
  replication:
    enabled: true
    targets:
      - name: dr-site
        endpoint: "https://dr.novacron.io"
        schedule: "*/15 * * * *"  # Every 15 minutes
  
status:
  phase: Ready
  used: "3.2Ti"
  available: "6.8Ti"
  allocations: 127
  health: Healthy
  lastSnapshot: "2024-01-20T12:00:00Z"
  lastReplication: "2024-01-20T12:15:00Z"
```

## Operator Implementation

### Architecture

```go
// pkg/controller/virtualmachine_controller.go
package controller

import (
    "context"
    "fmt"
    
    novacronv1alpha1 "github.com/novacron/operator/api/v1alpha1"
    corev1 "k8s.io/api/core/v1"
    "k8s.io/apimachinery/pkg/runtime"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/client"
    "sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

type VirtualMachineReconciler struct {
    client.Client
    Scheme *runtime.Scheme
    VMManager VMManagerInterface
}

// Reconcile handles VirtualMachine resource changes
func (r *VirtualMachineReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
    log := ctrl.LoggerFrom(ctx)
    
    // Fetch the VirtualMachine instance
    vm := &novacronv1alpha1.VirtualMachine{}
    if err := r.Get(ctx, req.NamespacedName, vm); err != nil {
        return ctrl.Result{}, client.IgnoreNotFound(err)
    }
    
    // Handle deletion
    if !vm.DeletionTimestamp.IsZero() {
        return r.handleDeletion(ctx, vm)
    }
    
    // Add finalizer
    if !controllerutil.ContainsFinalizer(vm, vmFinalizer) {
        controllerutil.AddFinalizer(vm, vmFinalizer)
        if err := r.Update(ctx, vm); err != nil {
            return ctrl.Result{}, err
        }
    }
    
    // Reconcile VM state
    switch vm.Status.Phase {
    case "":
        return r.handleCreate(ctx, vm)
    case novacronv1alpha1.VMPending:
        return r.handlePending(ctx, vm)
    case novacronv1alpha1.VMRunning:
        return r.handleRunning(ctx, vm)
    case novacronv1alpha1.VMMigrating:
        return r.handleMigrating(ctx, vm)
    default:
        return ctrl.Result{}, fmt.Errorf("unknown phase: %s", vm.Status.Phase)
    }
}

func (r *VirtualMachineReconciler) handleCreate(ctx context.Context, vm *novacronv1alpha1.VirtualMachine) (ctrl.Result, error) {
    log := ctrl.LoggerFrom(ctx)
    log.Info("Creating VM", "name", vm.Name)
    
    // Select node for VM placement
    node, err := r.selectNode(ctx, vm)
    if err != nil {
        return ctrl.Result{}, err
    }
    
    // Create VM on selected node
    vmID, err := r.VMManager.CreateVM(ctx, node, vm.Spec)
    if err != nil {
        return ctrl.Result{}, err
    }
    
    // Update status
    vm.Status.Phase = novacronv1alpha1.VMPending
    vm.Status.NodeName = node
    vm.Status.VMID = vmID
    
    if err := r.Status().Update(ctx, vm); err != nil {
        return ctrl.Result{}, err
    }
    
    // Requeue to check status
    return ctrl.Result{RequeueAfter: 10 * time.Second}, nil
}

func (r *VirtualMachineReconciler) selectNode(ctx context.Context, vm *novacronv1alpha1.VirtualMachine) (string, error) {
    // Implement placement algorithm
    // Consider:
    // - Node selector
    // - Affinity/Anti-affinity rules
    // - Resource availability
    // - Load balancing
    
    nodes := &corev1.NodeList{}
    if err := r.List(ctx, nodes); err != nil {
        return "", err
    }
    
    for _, node := range nodes.Items {
        // Check if node matches selector
        if !matchesSelector(node, vm.Spec.NodeSelector) {
            continue
        }
        
        // Check resource availability
        if hasCapacity(node, vm.Spec.Resources) {
            return node.Name, nil
        }
    }
    
    return "", fmt.Errorf("no suitable node found")
}
```

### Webhook Validation

```go
// pkg/webhook/virtualmachine_webhook.go
package webhook

import (
    "context"
    "fmt"
    
    "k8s.io/apimachinery/pkg/runtime"
    ctrl "sigs.k8s.io/controller-runtime"
    "sigs.k8s.io/controller-runtime/pkg/webhook"
    
    novacronv1alpha1 "github.com/novacron/operator/api/v1alpha1"
)

type VirtualMachineWebhook struct{}

// ValidateCreate validates VM creation
func (w *VirtualMachineWebhook) ValidateCreate(ctx context.Context, obj runtime.Object) error {
    vm := obj.(*novacronv1alpha1.VirtualMachine)
    
    // Validate resources
    if vm.Spec.Resources.CPU < 1 {
        return fmt.Errorf("CPU must be at least 1")
    }
    
    if vm.Spec.Resources.Memory.Value() < 512*1024*1024 {
        return fmt.Errorf("Memory must be at least 512Mi")
    }
    
    // Validate image source
    if vm.Spec.Image.Source == "" {
        return fmt.Errorf("Image source is required")
    }
    
    // Validate network configuration
    for _, net := range vm.Spec.Networks {
        if err := validateNetwork(net); err != nil {
            return err
        }
    }
    
    return nil
}

// Default sets default values
func (w *VirtualMachineWebhook) Default(ctx context.Context, obj runtime.Object) error {
    vm := obj.(*novacronv1alpha1.VirtualMachine)
    
    // Set default update strategy
    if vm.Spec.UpdateStrategy.Type == "" {
        vm.Spec.UpdateStrategy.Type = "RollingUpdate"
    }
    
    // Set default migration policy
    if vm.Spec.MigrationPolicy == nil {
        vm.Spec.MigrationPolicy = &novacronv1alpha1.MigrationPolicy{
            AllowLiveMigration: true,
            Compression: true,
            Encrypted: false,
        }
    }
    
    return nil
}
```

## Deployment

### Operator Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-operator
  namespace: novacron-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: novacron-operator
  template:
    metadata:
      labels:
        app: novacron-operator
    spec:
      serviceAccountName: novacron-operator
      containers:
      - name: operator
        image: novacron/operator:v1.0.0
        command:
        - /manager
        args:
        - --leader-elect
        - --metrics-bind-addr=:8080
        - --health-probe-bind-addr=:8081
        env:
        - name: WATCH_NAMESPACE
          value: ""  # Watch all namespaces
        resources:
          limits:
            cpu: 500m
            memory: 512Mi
          requests:
            cpu: 100m
            memory: 128Mi
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8081
          initialDelaySeconds: 15
          periodSeconds: 20
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8081
          initialDelaySeconds: 5
          periodSeconds: 10
```

### RBAC Configuration

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: novacron-operator
rules:
- apiGroups: [""]
  resources: ["nodes", "pods", "services", "persistentvolumeclaims"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "daemonsets", "statefulsets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["novacron.io"]
  resources: ["virtualmachines", "vmpools", "vmnetworks", "vmstorages"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["novacron.io"]
  resources: ["virtualmachines/status", "vmpools/status"]
  verbs: ["get", "update", "patch"]
```

## Monitoring & Observability

### Prometheus Metrics

```go
// pkg/metrics/metrics.go
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
)

var (
    // VM Metrics
    VMTotal = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "novacron_vm_total",
            Help: "Total number of VMs",
        },
        []string{"namespace", "phase"},
    )
    
    VMCPUUsage = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "novacron_vm_cpu_usage_cores",
            Help: "CPU usage in cores",
        },
        []string{"namespace", "vm", "node"},
    )
    
    VMMemoryUsage = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "novacron_vm_memory_usage_bytes",
            Help: "Memory usage in bytes",
        },
        []string{"namespace", "vm", "node"},
    )
    
    // Operation Metrics
    ReconcileTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "novacron_reconcile_total",
            Help: "Total number of reconciliations",
        },
        []string{"controller", "result"},
    )
    
    ReconcileDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "novacron_reconcile_duration_seconds",
            Help: "Reconciliation duration in seconds",
            Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
        },
        []string{"controller"},
    )
)

func init() {
    prometheus.MustRegister(
        VMTotal,
        VMCPUUsage,
        VMMemoryUsage,
        ReconcileTotal,
        ReconcileDuration,
    )
}
```

## Testing

### Unit Tests

```go
// pkg/controller/virtualmachine_controller_test.go
package controller

import (
    "context"
    "testing"
    
    . "github.com/onsi/ginkgo/v2"
    . "github.com/onsi/gomega"
    
    novacronv1alpha1 "github.com/novacron/operator/api/v1alpha1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("VirtualMachine Controller", func() {
    Context("When creating a VirtualMachine", func() {
        It("Should create VM on appropriate node", func() {
            vm := &novacronv1alpha1.VirtualMachine{
                ObjectMeta: metav1.ObjectMeta{
                    Name:      "test-vm",
                    Namespace: "default",
                },
                Spec: novacronv1alpha1.VirtualMachineSpec{
                    Resources: novacronv1alpha1.Resources{
                        CPU:    2,
                        Memory: "4Gi",
                    },
                },
            }
            
            Expect(k8sClient.Create(ctx, vm)).Should(Succeed())
            
            Eventually(func() string {
                vmCreated := &novacronv1alpha1.VirtualMachine{}
                k8sClient.Get(ctx, client.ObjectKeyFromObject(vm), vmCreated)
                return vmCreated.Status.Phase
            }, timeout, interval).Should(Equal("Running"))
        })
    })
})
```

## GitOps Integration

### ArgoCD Application

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: novacron-vms
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/novacron/vm-configs
    targetRevision: HEAD
    path: environments/production
  destination:
    server: https://kubernetes.default.svc
    namespace: production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```