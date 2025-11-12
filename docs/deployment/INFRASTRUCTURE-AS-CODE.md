# NovaCron Infrastructure as Code

## Overview

This document provides Infrastructure as Code (IaC) templates and configurations for deploying NovaCron production infrastructure using Terraform, Kubernetes, and related tools.

## Table of Contents

1. [Terraform Configuration](#terraform-configuration)
2. [Kubernetes Manifests](#kubernetes-manifests)
3. [Helm Charts](#helm-charts)
4. [Configuration Management](#configuration-management)
5. [Network Configuration](#network-configuration)
6. [Storage Configuration](#storage-configuration)

## Terraform Configuration

### Directory Structure

```
infrastructure/
├── terraform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── versions.tf
│   ├── modules/
│   │   ├── kubernetes/
│   │   ├── database/
│   │   ├── cache/
│   │   ├── networking/
│   │   └── monitoring/
│   └── environments/
│       ├── staging/
│       └── production/
```

### Main Configuration (main.tf)

```hcl
terraform {
  required_version = ">= 1.5.0"

  backend "gcs" {
    bucket = "novacron-terraform-state"
    prefix = "production"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "kubernetes" {
  host                   = module.kubernetes.endpoint
  token                  = module.kubernetes.access_token
  cluster_ca_certificate = base64decode(module.kubernetes.ca_certificate)
}

# Kubernetes Cluster
module "kubernetes" {
  source = "./modules/kubernetes"

  project_id        = var.project_id
  region            = var.region
  cluster_name      = "novacron-${var.environment}"
  node_pool_config  = var.node_pool_config
  network_config    = var.network_config
}

# Database
module "database" {
  source = "./modules/database"

  project_id       = var.project_id
  region           = var.region
  instance_name    = "novacron-db-${var.environment}"
  database_version = "POSTGRES_15"
  tier             = var.database_tier
  backup_config    = var.backup_config
}

# Redis Cache
module "cache" {
  source = "./modules/cache"

  project_id    = var.project_id
  region        = var.region
  instance_name = "novacron-cache-${var.environment}"
  memory_size   = var.cache_memory_size
  redis_version = "REDIS_7_0"
}

# Networking
module "networking" {
  source = "./modules/networking"

  project_id   = var.project_id
  region       = var.region
  vpc_name     = "novacron-vpc-${var.environment}"
  subnet_cidr  = var.subnet_cidr
}

# Monitoring
module "monitoring" {
  source = "./modules/monitoring"

  project_id   = var.project_id
  cluster_name = module.kubernetes.cluster_name
}
```

### Variables (variables.tf)

```hcl
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be staging or production"
  }
}

variable "node_pool_config" {
  description = "Kubernetes node pool configuration"
  type = object({
    machine_type   = string
    min_node_count = number
    max_node_count = number
    disk_size_gb   = number
  })
  default = {
    machine_type   = "n2-standard-4"
    min_node_count = 3
    max_node_count = 10
    disk_size_gb   = 100
  }
}

variable "database_tier" {
  description = "Database instance tier"
  type        = string
  default     = "db-custom-4-16384"
}

variable "cache_memory_size" {
  description = "Redis cache memory size in GB"
  type        = number
  default     = 8
}

variable "backup_config" {
  description = "Database backup configuration"
  type = object({
    enabled                        = bool
    start_time                     = string
    point_in_time_recovery_enabled = bool
    retention_days                 = number
  })
  default = {
    enabled                        = true
    start_time                     = "03:00"
    point_in_time_recovery_enabled = true
    retention_days                 = 30
  }
}
```

### Kubernetes Module (modules/kubernetes/main.tf)

```hcl
resource "google_container_cluster" "primary" {
  name     = var.cluster_name
  location = var.region

  # We can't create a cluster with no node pool, so we create the smallest possible default node pool
  # and immediately delete it
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = var.network
  subnetwork = var.subnetwork

  # Network policy
  network_policy {
    enabled  = true
    provider = "CALICO"
  }

  # Workload identity
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Binary authorization
  binary_authorization {
    evaluation_mode = "PROJECT_SINGLETON_POLICY_ENFORCE"
  }

  # Maintenance window
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }

  # Monitoring
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]

    managed_prometheus {
      enabled = true
    }
  }

  # Release channel
  release_channel {
    channel = "REGULAR"
  }

  # Cluster autoscaling
  cluster_autoscaling {
    enabled = true

    resource_limits {
      resource_type = "cpu"
      minimum       = 4
      maximum       = 64
    }

    resource_limits {
      resource_type = "memory"
      minimum       = 16
      maximum       = 256
    }

    auto_provisioning_defaults {
      disk_size = 100
      disk_type = "pd-standard"
    }
  }
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "${var.cluster_name}-node-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  node_count = var.min_node_count

  autoscaling {
    min_node_count = var.min_node_count
    max_node_count = var.max_node_count
  }

  node_config {
    preemptible  = false
    machine_type = var.machine_type

    # Google recommends custom service accounts with minimal permissions
    service_account = google_service_account.kubernetes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]

    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }

    disk_size_gb = var.disk_size_gb
    disk_type    = "pd-ssd"

    # Enable workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }

    # Security
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }

  management {
    auto_repair  = true
    auto_upgrade = true
  }
}

resource "google_service_account" "kubernetes" {
  account_id   = "${var.cluster_name}-sa"
  display_name = "Service Account for ${var.cluster_name}"
}
```

### Database Module (modules/database/main.tf)

```hcl
resource "google_sql_database_instance" "main" {
  name             = var.instance_name
  database_version = var.database_version
  region           = var.region

  settings {
    tier              = var.tier
    availability_type = "REGIONAL"
    disk_type         = "PD_SSD"
    disk_size         = 100
    disk_autoresize   = true

    backup_configuration {
      enabled                        = var.backup_config.enabled
      start_time                     = var.backup_config.start_time
      point_in_time_recovery_enabled = var.backup_config.point_in_time_recovery_enabled
      transaction_log_retention_days = 7
      backup_retention_settings {
        retained_backups = var.backup_config.retention_days
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.network_id
      require_ssl     = true

      ssl_mode = "ENCRYPTED_ONLY"
    }

    insights_config {
      query_insights_enabled  = true
      query_string_length     = 1024
      record_application_tags = true
    }

    database_flags {
      name  = "max_connections"
      value = "200"
    }

    database_flags {
      name  = "shared_buffers"
      value = "4194304"  # 4GB in 8kB pages
    }

    database_flags {
      name  = "work_mem"
      value = "16384"    # 16MB in kB
    }
  }

  deletion_protection = true
}

resource "google_sql_database" "database" {
  name     = "novacron"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "user" {
  name     = "novacron"
  instance = google_sql_database_instance.main.name
  password = var.database_password
}

# Read replica for reporting queries
resource "google_sql_database_instance" "read_replica" {
  name                 = "${var.instance_name}-replica"
  master_instance_name = google_sql_database_instance.main.name
  region               = var.region
  database_version     = var.database_version

  replica_configuration {
    failover_target = false
  }

  settings {
    tier              = var.tier
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_autoresize   = true

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.network_id
    }
  }
}
```

### Cache Module (modules/cache/main.tf)

```hcl
resource "google_redis_instance" "cache" {
  name               = var.instance_name
  tier               = "STANDARD_HA"
  memory_size_gb     = var.memory_size
  region             = var.region
  redis_version      = var.redis_version
  display_name       = "NovaCron Cache ${var.environment}"
  reserved_ip_range  = var.redis_ip_range
  authorized_network = var.network_id

  # High availability configuration
  replica_count      = 1
  read_replicas_mode = "READ_REPLICAS_ENABLED"

  # Persistence configuration
  persistence_config {
    persistence_mode    = "RDB"
    rdb_snapshot_period = "TWELVE_HOURS"
  }

  # Maintenance window
  maintenance_policy {
    weekly_maintenance_window {
      day = "SUNDAY"
      start_time {
        hours   = 3
        minutes = 0
      }
    }
  }

  # Redis configuration
  redis_configs = {
    maxmemory-policy = "allkeys-lru"
    timeout          = "300"
  }

  labels = {
    environment = var.environment
    managed-by  = "terraform"
  }
}
```

## Kubernetes Manifests

### Namespace

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: novacron
  labels:
    name: novacron
    environment: production
```

### ConfigMap

```yaml
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: novacron-config
  namespace: novacron
data:
  # Application configuration
  LOG_LEVEL: "info"
  API_PORT: "8090"
  METRICS_PORT: "9090"

  # Database configuration
  DB_MAX_CONNECTIONS: "200"
  DB_MAX_IDLE_CONNECTIONS: "50"
  DB_CONNECTION_TIMEOUT: "30s"

  # Cache configuration
  REDIS_DB: "0"
  REDIS_MAX_RETRIES: "3"
  REDIS_TIMEOUT: "5s"

  # DWCP configuration
  DWCP_MAX_STREAMS: "1000"
  DWCP_BUFFER_SIZE: "65536"
  DWCP_COMPRESSION_LEVEL: "6"

  # Performance tuning
  GOMAXPROCS: "4"
  GOMEMLIMIT: "1800MiB"
```

### Secrets

```yaml
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: novacron-secrets
  namespace: novacron
type: Opaque
stringData:
  database-url: "postgresql://novacron:PASSWORD@postgres:5432/novacron?sslmode=require"
  redis-url: "redis://:PASSWORD@redis:6379/0"
  jwt-secret: "GENERATED_JWT_SECRET"
```

### Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-api
  namespace: novacron
  labels:
    app: novacron
    component: api
    version: v1.6.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: novacron
      component: api
  template:
    metadata:
      labels:
        app: novacron
        component: api
        version: v1.6.0
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: novacron-api
      securityContext:
        fsGroup: 1000
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault

      containers:
      - name: api
        image: ghcr.io/novacron/api-server:v1.6.0
        imagePullPolicy: IfNotPresent

        ports:
        - name: http
          containerPort: 8090
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP

        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: novacron-config
              key: LOG_LEVEL
        - name: API_PORT
          valueFrom:
            configMapKeyRef:
              name: novacron-config
              key: API_PORT
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: redis-url
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: novacron-secrets
              key: jwt-secret

        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
            ephemeral-storage: "1Gi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
            ephemeral-storage: "2Gi"

        livenessProbe:
          httpGet:
            path: /health
            port: 8090
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3

        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8090
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 3

        startupProbe:
          httpGet:
            path: /health
            port: 8090
            scheme: HTTP
          initialDelaySeconds: 0
          periodSeconds: 5
          timeoutSeconds: 3
          successThreshold: 1
          failureThreshold: 30

        securityContext:
          runAsUser: 1000
          runAsGroup: 1000
          readOnlyRootFilesystem: true
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL

        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache

      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}

      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - novacron
                - key: component
                  operator: In
                  values:
                  - api
              topologyKey: kubernetes.io/hostname

      topologySpreadConstraints:
      - maxSkew: 1
        topologyKey: topology.kubernetes.io/zone
        whenUnsatisfiable: ScheduleAnyway
        labelSelector:
          matchLabels:
            app: novacron
            component: api
```

### Service

```yaml
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: novacron-api
  namespace: novacron
  labels:
    app: novacron
    component: api
spec:
  type: ClusterIP
  selector:
    app: novacron
    component: api
  ports:
  - name: http
    port: 8090
    targetPort: 8090
    protocol: TCP
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800
```

### Horizontal Pod Autoscaler

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: novacron-api-hpa
  namespace: novacron
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: novacron-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 1
        periodSeconds: 60
      selectPolicy: Min
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30
      - type: Pods
        value: 2
        periodSeconds: 30
      selectPolicy: Max
```

### Pod Disruption Budget

```yaml
# kubernetes/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: novacron-api-pdb
  namespace: novacron
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: novacron
      component: api
```

### Network Policy

```yaml
# kubernetes/networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: novacron-api-netpol
  namespace: novacron
spec:
  podSelector:
    matchLabels:
      app: novacron
      component: api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8090
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - namespaceSelector: {}
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
```

### Ingress

```yaml
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: novacron-ingress
  namespace: novacron
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/limit-rps: "20"
    nginx.ingress.kubernetes.io/limit-connections: "10"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-connect-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "30"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "30"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - novacron.local
    secretName: novacron-tls
  rules:
  - host: novacron.local
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: novacron-api
            port:
              number: 8090
```

## Deployment Commands

### Terraform Deployment

```bash
# Initialize Terraform
cd infrastructure/terraform/environments/production
terraform init

# Plan changes
terraform plan -out=tfplan

# Apply changes
terraform apply tfplan

# Destroy (use with extreme caution!)
terraform destroy
```

### Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -f kubernetes/

# Or apply in order
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/secrets.yaml
kubectl apply -f kubernetes/rbac.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl apply -f kubernetes/hpa.yaml
kubectl apply -f kubernetes/pdb.yaml
kubectl apply -f kubernetes/networkpolicy.yaml
kubectl apply -f kubernetes/ingress.yaml

# Verify deployment
kubectl get all -n novacron
kubectl rollout status deployment/novacron-api -n novacron
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-10
**Owner:** DevOps Team
