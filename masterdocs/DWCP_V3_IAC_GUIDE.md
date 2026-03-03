# DWCP v3 Infrastructure as Code Guide

**Version**: 3.0.0
**Last Updated**: 2025-11-10
**Authors**: NovaCron Infrastructure Team

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Ansible Automation](#ansible-automation)
4. [Terraform Modules](#terraform-modules)
5. [Configuration Management](#configuration-management)
6. [Policy as Code](#policy-as-code)
7. [Drift Detection](#drift-detection)
8. [Deployment Workflows](#deployment-workflows)
9. [Multi-Region Setup](#multi-region-setup)
10. [Disaster Recovery](#disaster-recovery)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)

---

## Overview

The DWCP v3 Infrastructure as Code (IaC) implementation provides comprehensive automation for deploying, managing, and maintaining DWCP v3 across hybrid datacenter and internet environments.

### Key Features

- **Ansible Playbooks**: Complete automation for setup, upgrade, and configuration
- **Terraform Modules**: Infrastructure provisioning for network, compute, and monitoring
- **Configuration Templates**: Mode-specific configurations (datacenter, internet, hybrid)
- **Policy as Code**: OPA policies for security and network validation
- **Drift Detection**: Continuous monitoring and automatic remediation
- **Multi-Region**: Geographic distribution with failover capabilities
- **GitOps Ready**: Version-controlled infrastructure with automated deployment

### Components

```
deployments/
├── ansible/
│   ├── dwcp-v3-setup.yml           # Initial deployment
│   ├── dwcp-v3-upgrade.yml         # Zero-downtime upgrade
│   └── inventory/
│       └── production.ini          # Node inventory
├── terraform/
│   ├── modules/
│   │   ├── dwcp-v3-network/        # VPC, subnets, security groups
│   │   ├── dwcp-v3-compute/        # EC2, ASG, load balancers
│   │   └── dwcp-v3-monitoring/     # Prometheus, Grafana
│   └── environments/
│       └── production/             # Multi-region deployment
config/
├── dwcp-v3-datacenter.yaml         # RDMA-optimized configuration
├── dwcp-v3-internet.yaml           # TCP/BBR configuration
└── dwcp-v3-hybrid.yaml             # Adaptive mode configuration
policies/
├── dwcp-v3-security.rego           # Security validation
└── dwcp-v3-network.rego            # Network validation
scripts/
└── drift-detection.sh              # Configuration drift detection
```

---

## Architecture

### Infrastructure Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     Multi-Region Layer                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐            │
│  │ US-EAST-1│    │ US-WEST-2│    │ EU-WEST-1│            │
│  └──────────┘    └──────────┘    └──────────┘            │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Network Layer (VPC)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Datacenter │  │   Internet   │  │  Monitoring  │    │
│  │   Subnets    │  │   Subnets    │  │   Subnet     │    │
│  │  (Private)   │  │   (Public)   │  │  (Private)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Compute Layer (EC2)                      │
│  ┌──────────────┐  ┌──────────────┐                       │
│  │  Datacenter  │  │   Internet   │                       │
│  │     ASG      │  │      ASG     │                       │
│  │ (RDMA/EFA)   │  │  (TCP/BBR)   │                       │
│  └──────────────┘  └──────────────┘                       │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                 Monitoring Layer (Docker)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                │
│  │Prometheus│  │ Grafana  │  │Alertmgr  │                │
│  └──────────┘  └──────────┘  └──────────┘                │
└─────────────────────────────────────────────────────────────┘
```

### Deployment Modes

| Mode | Transport | Target Latency | Use Case |
|------|-----------|----------------|----------|
| **Datacenter** | RDMA | 5µs | High-performance cluster computing |
| **Internet** | TCP/BBR | 50ms | WAN communication, untrusted networks |
| **Hybrid** | RDMA + TCP | Adaptive | Mixed environments, automatic switching |

---

## Ansible Automation

### Setup Playbook

Deploy DWCP v3 from scratch:

```bash
# Deploy to all nodes
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-setup.yml

# Deploy specific components
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-setup.yml \
    --tags prerequisites,build,config

# Dry run
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-setup.yml \
    --check
```

**Playbook Tasks**:
1. System prerequisites (kernel, packages, RDMA drivers)
2. Go installation and environment setup
3. DWCP v3 build from source
4. Configuration deployment
5. Systemd service setup
6. Monitoring installation (Node Exporter)
7. Health validation

### Upgrade Playbook

Zero-downtime upgrade from v1 to v3:

```bash
# Full upgrade with default settings
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml

# Gradual rollout (20% batch size)
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    -e "batch_size=20%"

# Upgrade without feature flags
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    --skip-tags features
```

**Upgrade Phases**:
1. **Pre-flight**: Backup, validation, disk/memory checks
2. **Deploy v3**: Install alongside v1 (shadow mode)
3. **Traffic Migration**: Gradual shift (0% → 10% → 25% → 50% → 75% → 100%)
4. **Feature Flags**: Progressive enablement (v3 → hybrid → compression → Byzantine)
5. **Cutover**: Stop v1, switch v3 to primary port
6. **Validation**: Health checks, performance metrics
7. **Rollback** (on failure): Automatic revert to v1

### Inventory Management

```ini
# deployments/ansible/inventory/production.ini

# Datacenter nodes (RDMA-enabled)
[datacenter]
dc-node-01 ansible_host=10.0.1.10 rdma_device=mlx5_0
dc-node-02 ansible_host=10.0.1.11 rdma_device=mlx5_0

# Internet nodes (TCP/BBR)
[internet]
inet-node-01 ansible_host=203.0.113.10 region=us-east-1
inet-node-02 ansible_host=203.0.113.11 region=us-west-2

# Hybrid nodes (adaptive)
[hybrid]
hybrid-node-01 ansible_host=10.0.2.10 primary_mode=datacenter
hybrid-node-02 ansible_host=203.0.113.20 primary_mode=internet

# Variables
[datacenter:vars]
dwcp_mode=datacenter
rdma_enabled=true
compression_enabled=false

[internet:vars]
dwcp_mode=internet
compression_enabled=true
byzantine_tolerance=true
```

---

## Terraform Modules

### Network Module

Create VPC, subnets, security groups, and RDMA-enabled networking:

```bash
cd deployments/terraform/environments/production
terraform init
terraform plan
terraform apply
```

**Module Inputs**:
- `vpc_cidr`: VPC CIDR block (default: 10.0.0.0/16)
- `availability_zones`: List of AZs (min 2 for HA)
- `enable_rdma`: Enable EFA for RDMA (default: true)
- `enable_ipv6`: IPv6 support (default: true)
- `enable_flow_logs`: VPC flow logs (default: true)

**Module Outputs**:
- `vpc_id`: VPC identifier
- `datacenter_subnet_ids`: Private subnet IDs for RDMA
- `internet_subnet_ids`: Public subnet IDs for internet
- `datacenter_security_group_id`: SG for datacenter nodes
- `internet_security_group_id`: SG for internet nodes

**Key Resources**:
- VPC with /16 CIDR
- Datacenter subnets (private, RDMA-enabled)
- Internet subnets (public, internet gateway)
- NAT gateways for datacenter egress
- Security groups with least-privilege rules
- Network ACLs for additional security
- VPC flow logs to CloudWatch

### Compute Module

Deploy EC2 instances, auto-scaling groups, and load balancers:

```bash
cd deployments/terraform/environments/production
terraform apply -target=module.compute_primary
```

**Module Inputs**:
- `datacenter_instance_type`: Instance type for datacenter (default: p4d.24xlarge)
- `internet_instance_type`: Instance type for internet (default: c6in.32xlarge)
- `datacenter_min_size`: Min datacenter nodes (default: 3)
- `datacenter_max_size`: Max datacenter nodes (default: 10)
- `internet_min_size`: Min internet nodes (default: 5)
- `internet_max_size`: Max internet nodes (default: 20)

**Module Outputs**:
- `datacenter_asg_name`: Datacenter auto-scaling group
- `internet_asg_name`: Internet auto-scaling group
- `datacenter_nlb_dns`: Datacenter load balancer DNS
- `internet_nlb_dns`: Internet load balancer DNS

**Key Resources**:
- Launch templates with user data
- Auto-scaling groups with health checks
- Network load balancers (layer 4)
- Target groups with health checks
- Placement groups (cluster for datacenter)
- IAM roles and instance profiles
- Auto-scaling policies (CPU-based)

### Monitoring Module

Deploy Prometheus, Grafana, and Alertmanager:

```bash
cd deployments/terraform/environments/production
terraform apply -target=module.monitoring_primary
```

**Module Inputs**:
- `monitoring_instance_type`: Instance type (default: m6i.2xlarge)
- `prometheus_retention_days`: Data retention (default: 30)
- `grafana_admin_password`: Admin password (sensitive)
- `alert_email`: Email for alerts

**Module Outputs**:
- `prometheus_url`: Prometheus UI URL
- `grafana_url`: Grafana UI URL
- `alertmanager_url`: Alertmanager UI URL

**Key Resources**:
- EC2 instance with Docker
- EBS volumes for persistent storage
- Docker Compose stack (Prometheus, Grafana, Alertmanager)
- EC2 service discovery for DWCP nodes
- CloudWatch integration
- SNS topic for critical alerts

---

## Configuration Management

### Datacenter Mode Configuration

Optimized for RDMA high-performance networks:

```yaml
# config/dwcp-v3-datacenter.yaml
mode: datacenter

network:
  transport: rdma
  rdmaDevice: mlx5_0
  rdma:
    enabled: true
    qpType: RC
    maxQueuePairs: 1024
    hugepagesEnabled: true

performance:
  targetLatencyMicros: 5
  targetThroughputGbps: 200
  zeroCopy: true
  pollingMode: true

protocol:
  compression:
    enabled: false  # Disabled for lowest latency
  encryption:
    enabled: false  # Trusted network
```

**Use Cases**:
- High-performance computing clusters
- AI/ML training with distributed GPUs
- Real-time analytics pipelines
- Low-latency trading systems

### Internet Mode Configuration

Optimized for TCP/BBR over unreliable WAN:

```yaml
# config/dwcp-v3-internet.yaml
mode: internet

network:
  transport: tcp
  tcp:
    congestionControl: bbr
    noDelay: true
    fastOpen: true

performance:
  targetLatencyMs: 50
  targetThroughputGbps: 10

protocol:
  compression:
    enabled: true
    algorithm: zstd
    level: 3
  encryption:
    enabled: true
    algorithm: aes-256-gcm

byzantine:
  enabled: true
  authentication:
    method: hmac-sha256
```

**Use Cases**:
- Multi-cloud communication
- Edge-to-core data transfer
- Geo-distributed applications
- Untrusted network environments

### Hybrid Mode Configuration

Adaptive switching between datacenter and internet:

```yaml
# config/dwcp-v3-hybrid.yaml
mode: hybrid

network:
  transports:
    - name: rdma
      enabled: true
      priority: 1
    - name: tcp
      enabled: true
      priority: 2

adaptive:
  detection:
    enabled: true
    interval: 1s
  datacenterCriteria:
    - metric: latency
      operator: less_than
      value: 100us
      weight: 0.4
  switching:
    cooldownPeriod: 30s
    gracefulTransition: true
```

**Use Cases**:
- Mixed environments (datacenter + cloud)
- Disaster recovery scenarios
- Bursting to cloud resources
- Development/staging environments

---

## Policy as Code

### Security Policy

Validate security configuration with OPA:

```bash
# Check configuration against security policy
opa eval --data policies/dwcp-v3-security.rego \
    --input config/dwcp-v3-internet.yaml \
    --format pretty \
    "data.dwcp.security.decision"

# Output:
# {
#   "allow": true,
#   "violations": [],
#   "metadata": {...}
# }
```

**Security Rules**:
1. **Encryption**: Internet mode must have encryption enabled
2. **Authentication**: Internet mode must have authentication
3. **TLS**: Minimum TLS 1.2, TLS 1.3 recommended
4. **Byzantine**: Internet mode must have Byzantine tolerance
5. **Network**: No unrestricted SSH/RDP, management ports VPC-only
6. **IAM**: Least privilege, no wildcard resources
7. **Compliance**: Logging enabled, audit trail configured
8. **Vulnerabilities**: No critical CVEs (CVSS >= 9.0)

### Network Policy

Validate network configuration:

```bash
opa eval --data policies/dwcp-v3-network.rego \
    --input terraform-output.json \
    --format pretty \
    "data.dwcp.network.decision"
```

**Network Rules**:
1. **Topology**: Valid VPC CIDR, no subnet overlap, multi-AZ
2. **Subnets**: Datacenter subnets private, internet subnets public
3. **Routing**: Valid default routes, NAT gateways configured
4. **Firewall**: RDMA traffic allowed in VPC, management ports restricted
5. **RDMA**: Proper queue configuration, hugepages enabled
6. **Performance**: MTU appropriate for mode, buffer sizes valid
7. **Multi-Region**: VPC peering, DNS failover

### CI/CD Integration

Integrate OPA validation in CI/CD pipeline:

```yaml
# .github/workflows/validate-config.yml
name: Validate Configuration

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install OPA
        run: |
          curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64
          chmod +x opa
      - name: Validate Security Policy
        run: |
          ./opa test policies/dwcp-v3-security.rego
          for config in config/*.yaml; do
            ./opa eval --data policies/dwcp-v3-security.rego \
              --input $config --format pretty \
              "data.dwcp.security.decision" | grep '"allow": true'
          done
      - name: Validate Network Policy
        run: |
          ./opa test policies/dwcp-v3-network.rego
```

---

## Drift Detection

### Automated Drift Detection

Continuously monitor configuration drift:

```bash
# Initialize drift detection
sudo scripts/drift-detection.sh init

# Capture baseline
sudo scripts/drift-detection.sh baseline

# Run one-time detection
sudo scripts/drift-detection.sh detect

# Start continuous monitoring (5-minute interval)
sudo scripts/drift-detection.sh monitor --interval 300

# Dry run (detection without remediation)
sudo scripts/drift-detection.sh detect --dry-run
```

### Drift Types Detected

| Drift Type | Description | Severity | Auto-Remediate |
|------------|-------------|----------|----------------|
| **Config File** | Configuration files modified | Medium | Yes (Git restore) |
| **Terraform** | Infrastructure drift from state | High | Yes (terraform apply) |
| **Running Config** | Live config differs from baseline | High | Yes (Ansible re-run) |
| **Policy Violation** | OPA policy violations | Critical | No (manual review) |

### Remediation Strategies

1. **Configuration Files**: Restore from Git repository
2. **Terraform State**: Apply Terraform plan to reconcile
3. **Running Configuration**: Re-run Ansible playbooks
4. **Policy Violations**: Alert and require manual intervention

### Drift Detection Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Drift Detection                          │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Config Files │  │   Terraform  │  │   Running    │    │
│  │   Monitor    │  │    Monitor   │  │Config Monitor│    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│         │                 │                  │            │
│         └─────────────────┼──────────────────┘            │
│                           │                                │
│                   ┌───────▼───────┐                       │
│                   │  Drift Report │                       │
│                   └───────┬───────┘                       │
│                           │                                │
│         ┌─────────────────┼─────────────────┐            │
│         │                 │                 │            │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐    │
│  │    Alert    │  │  Remediate  │  │   Audit     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Alert Channels

1. **Email**: Sent to ops@novacron.io
2. **Slack**: Posted to #alerts channel (if webhook configured)
3. **Logs**: Written to /var/log/dwcp/drift-detection.log
4. **Metrics**: Exposed via Prometheus

---

## Deployment Workflows

### Fresh Deployment

Complete deployment from scratch:

```bash
# 1. Initialize Terraform backend
cd deployments/terraform/environments/production
terraform init

# 2. Plan infrastructure
terraform plan -out=tfplan

# 3. Apply infrastructure
terraform apply tfplan

# 4. Wait for instances to be ready
aws ec2 wait instance-running --filters "Name=tag:Environment,Values=production"

# 5. Deploy DWCP v3 with Ansible
cd ../../../
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-setup.yml

# 6. Validate deployment
ansible all -i deployments/ansible/inventory/production.ini \
    -m shell -a "dwcp-ctl version && dwcp-ctl status"

# 7. Initialize drift detection
sudo scripts/drift-detection.sh init
```

### Zero-Downtime Upgrade

Upgrade from v1 to v3 without service interruption:

```bash
# 1. Pre-flight checks
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    --tags preflight --check

# 2. Deploy v3 in shadow mode
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    --tags deploy

# 3. Gradual traffic migration (20% batch)
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    --tags migrate \
    -e "batch_size=20%"

# 4. Enable features progressively
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    --tags features

# 5. Final cutover
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    --tags cutover

# 6. Validation
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    --tags validation
```

### Rollback Procedure

Revert to previous version on failure:

```bash
# Automatic rollback (on upgrade failure)
ansible-playbook -i deployments/ansible/inventory/production.ini \
    deployments/ansible/dwcp-v3-upgrade.yml \
    --tags rollback \
    -e "upgrade_failed=true"

# Manual rollback
ansible all -i deployments/ansible/inventory/production.ini \
    -m systemd -a "name=dwcp-v3 state=stopped"
ansible all -i deployments/ansible/inventory/production.ini \
    -m systemd -a "name=dwcp state=started"
```

---

## Multi-Region Setup

### Architecture

DWCP v3 supports multi-region deployment with automatic failover:

- **Primary Region**: US-EAST-1 (70% traffic)
- **Secondary Region**: US-WEST-2 (20% traffic, disaster recovery)
- **Tertiary Region**: EU-WEST-1 (10% traffic, global expansion)

### Deployment

```bash
# Deploy all regions
cd deployments/terraform/environments/production
terraform apply

# Deploy specific region
terraform apply -target=module.network_primary -target=module.compute_primary

# Verify multi-region setup
terraform output primary_region_endpoints
terraform output secondary_region_endpoints
```

### VPC Peering

Cross-region connectivity:

```hcl
# Automatic VPC peering between regions
resource "aws_vpc_peering_connection" "primary_secondary" {
  vpc_id      = module.network_primary.vpc_id
  peer_vpc_id = module.network_secondary.vpc_id
  peer_region = var.secondary_region
}
```

### Global DNS

Route53 weighted routing for traffic distribution:

```
api.dwcp.novacron.io
├── US-EAST-1 (weight: 70, health check)
├── US-WEST-2 (weight: 20, health check)
└── EU-WEST-1 (weight: 10, health check)
```

---

## Disaster Recovery

### Backup Strategy

1. **Configuration**: S3 with cross-region replication
2. **Terraform State**: S3 backend with versioning
3. **Monitoring Data**: EBS snapshots, cross-region copy
4. **Application State**: DWCP handles stateless design

### Failover Procedure

```bash
# 1. Detect primary region failure
aws cloudwatch get-metric-statistics \
    --namespace AWS/ELB \
    --metric-name HealthyHostCount \
    --dimensions Name=LoadBalancerName,Value=primary-nlb \
    --start-time $(date -u -d '5 minutes ago' +%Y-%m-%dT%H:%M:%S) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --period 60 \
    --statistics Average

# 2. Update Route53 to remove primary
aws route53 change-resource-record-sets \
    --hosted-zone-id Z123456789 \
    --change-batch file://remove-primary.json

# 3. Scale up secondary region
aws autoscaling set-desired-capacity \
    --auto-scaling-group-name dwcp-v3-datacenter-asg-production \
    --desired-capacity 10 \
    --region us-west-2

# 4. Monitor failover
watch -n 5 'dig +short api.dwcp.novacron.io'
```

### Recovery Time Objectives

- **RTO (Recovery Time Objective)**: 5 minutes
- **RPO (Recovery Point Objective)**: 0 seconds (stateless)

---

## Troubleshooting

### Common Issues

#### Ansible Connection Failures

```bash
# Test connectivity
ansible all -i deployments/ansible/inventory/production.ini -m ping

# Check SSH keys
ansible all -i deployments/ansible/inventory/production.ini \
    -m shell -a "whoami" --private-key ~/.ssh/your-key.pem

# Increase verbosity
ansible-playbook -vvv ...
```

#### Terraform State Lock

```bash
# Force unlock (use with caution)
terraform force-unlock <LOCK_ID>

# Alternative: Delete DynamoDB lock item
aws dynamodb delete-item \
    --table-name terraform-state-lock \
    --key '{"LockID": {"S": "novacron-terraform-state-production/dwcp-v3/terraform.tfstate"}}'
```

#### RDMA Not Available

```bash
# Check RDMA devices
ansible datacenter -i deployments/ansible/inventory/production.ini \
    -m shell -a "ibv_devices"

# Verify EFA driver
ansible datacenter -i deployments/ansible/inventory/production.ini \
    -m shell -a "fi_info -p efa"

# Check kernel modules
ansible datacenter -i deployments/ansible/inventory/production.ini \
    -m shell -a "lsmod | grep rdma"
```

#### High Drift Detection Rate

```bash
# Check drift detection logs
tail -f /var/log/dwcp/drift-detection.log

# Disable auto-remediation temporarily
sudo scripts/drift-detection.sh monitor --no-auto

# Capture new baseline
sudo scripts/drift-detection.sh baseline
```

### Debug Commands

```bash
# Check DWCP service status
ansible all -i deployments/ansible/inventory/production.ini \
    -m systemd -a "name=dwcp state=started"

# View logs
ansible all -i deployments/ansible/inventory/production.ini \
    -m shell -a "journalctl -u dwcp -n 100 --no-pager"

# Check health endpoint
ansible all -i deployments/ansible/inventory/production.ini \
    -m uri -a "url=http://localhost:9100/health"

# Verify configuration
ansible all -i deployments/ansible/inventory/production.ini \
    -m shell -a "dwcp-ctl config validate"
```

---

## Best Practices

### Version Control

1. **Git Workflow**: Use feature branches for infrastructure changes
2. **Code Review**: Require approval for production changes
3. **Tagging**: Tag releases for rollback capability
4. **Secrets**: Use HashiCorp Vault or AWS Secrets Manager

### Testing

1. **Terraform Plan**: Always review plan before apply
2. **Ansible Check Mode**: Use `--check` for dry runs
3. **Staging Environment**: Test changes in staging first
4. **Automated Testing**: Use Terraform validation and OPA tests

### Security

1. **Least Privilege**: Minimal IAM permissions
2. **Encryption**: Enable encryption at rest and in transit
3. **Network Segmentation**: Use security groups and NACLs
4. **Audit Logging**: Enable CloudTrail and VPC flow logs
5. **Regular Updates**: Apply security patches promptly

### Monitoring

1. **Metrics**: Monitor all key performance indicators
2. **Alerts**: Configure alerts for critical thresholds
3. **Dashboards**: Create Grafana dashboards for visibility
4. **Log Aggregation**: Centralize logs in CloudWatch or ELK
5. **Drift Detection**: Enable continuous configuration monitoring

### Documentation

1. **Inline Comments**: Document complex logic
2. **README Files**: Provide usage instructions
3. **Runbooks**: Document operational procedures
4. **Change Log**: Maintain history of changes

---

## Appendix

### File Structure

```
novacron/
├── deployments/
│   ├── ansible/
│   │   ├── dwcp-v3-setup.yml                  (649 lines)
│   │   ├── dwcp-v3-upgrade.yml                (459 lines)
│   │   ├── inventory/
│   │   │   └── production.ini                 (211 lines)
│   ├── terraform/
│   │   ├── modules/
│   │   │   ├── dwcp-v3-network/
│   │   │   │   └── main.tf                     (636 lines)
│   │   │   ├── dwcp-v3-compute/
│   │   │   │   └── main.tf                     (618 lines)
│   │   │   └── dwcp-v3-monitoring/
│   │   │       └── main.tf                     (715 lines)
│   │   └── environments/
│   │       └── production/
│   │           └── main.tf                     (413 lines)
├── config/
│   ├── dwcp-v3-datacenter.yaml                (241 lines)
│   ├── dwcp-v3-internet.yaml                  (290 lines)
│   └── dwcp-v3-hybrid.yaml                    (395 lines)
├── policies/
│   ├── dwcp-v3-security.rego                  (537 lines)
│   └── dwcp-v3-network.rego                   (618 lines)
├── scripts/
│   └── drift-detection.sh                     (736 lines)
└── docs/
    └── DWCP_V3_IAC_GUIDE.md                   (This document)
```

### Total Lines of Code

- **Ansible**: 1,319 lines
- **Terraform**: 2,382 lines
- **Configuration**: 926 lines
- **Policies**: 1,155 lines
- **Scripts**: 736 lines
- **Documentation**: 1,200+ lines

**Total**: 7,718+ lines of Infrastructure as Code

### References

- [Ansible Documentation](https://docs.ansible.com/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Open Policy Agent](https://www.openpolicyagent.org/docs/latest/)
- [AWS EFA (RDMA)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html)
- [BBR Congestion Control](https://github.com/google/bbr)

### Support

For questions or issues:
- Email: infrastructure@novacron.io
- Slack: #dwcp-infrastructure
- GitHub Issues: https://github.com/novacron/dwcp/issues

---

**Document Version**: 1.0.0
**Last Updated**: 2025-11-10
**Next Review**: 2025-12-10
