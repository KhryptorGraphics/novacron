# Ubuntu 24.04 LTS Features in NovaCron

This document provides a comprehensive overview of the Ubuntu 24.04 LTS (Noble Numbat) support in NovaCron's distributed hypervisor.

## Table of Contents

1. [Introduction](#introduction)
2. [Snapshot and Backup Support](#snapshot-and-backup-support)
3. [Performance Optimization](#performance-optimization)
4. [Migration Support](#migration-support)
5. [Enhanced Cloud-Init Integration](#enhanced-cloud-init-integration)
6. [API Reference](#api-reference)
7. [CLI Reference](#cli-reference)
8. [Troubleshooting](#troubleshooting)

## Introduction

Ubuntu 24.04 LTS (Noble Numbat) is the latest Long Term Support release from Canonical, offering enhanced performance, security, and stability. NovaCron's distributed hypervisor provides comprehensive support for Ubuntu 24.04 LTS, including advanced features such as snapshot/backup capabilities, performance optimization, migration support, and enhanced cloud-init integration.

### Key Benefits

- **Enhanced Performance**: Optimized for modern workloads with improved resource utilization
- **Advanced Snapshot System**: Point-in-time recovery with minimal downtime
- **Seamless Migration**: Live migration with minimal service interruption
- **Flexible Cloud-Init Integration**: Customizable VM provisioning with templates

## Snapshot and Backup Support

NovaCron provides comprehensive snapshot and backup capabilities for Ubuntu 24.04 VMs, allowing for point-in-time recovery and data protection.

### Features

- **Live Snapshots**: Create snapshots of running VMs without downtime
- **Offline Snapshots**: Create consistent snapshots of stopped VMs
- **Incremental Backups**: Space-efficient backup storage
- **Quick Restore**: Rapidly restore VMs from snapshots or backups
- **Snapshot Management**: List, delete, and manage snapshots through API and CLI

### Usage Examples

#### Creating a Snapshot

```bash
# Using the CLI
novacron vm snapshot create --vm-id <vm-id> --name "my-snapshot" --description "My first snapshot"

# Using the API
curl -X POST "http://localhost:8080/api/v1/vms/<vm-id>/snapshots" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-snapshot",
    "description": "My first snapshot",
    "tags": {
      "purpose": "backup",
      "environment": "production"
    }
  }'
```

#### Restoring from a Snapshot

```bash
# Using the CLI
novacron vm snapshot restore --vm-id <vm-id> --snapshot-id <snapshot-id>

# Using the API
curl -X POST "http://localhost:8080/api/v1/vms/<vm-id>/snapshots/<snapshot-id>/restore"
```

#### Creating a Full Backup

```bash
# Using the CLI
novacron vm backup create --vm-id <vm-id> --name "full-backup"

# Using the API
curl -X POST "http://localhost:8080/api/v1/vms/<vm-id>/backups" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "full-backup",
    "type": "full"
  }'
```

## Performance Optimization

NovaCron provides advanced performance optimization capabilities for Ubuntu 24.04 VMs, allowing for fine-tuned resource allocation and utilization.

### Performance Profiles

NovaCron includes several pre-defined performance profiles for Ubuntu 24.04 VMs:

1. **Balanced**: General-purpose profile with balanced resource allocation
2. **High-Performance**: Optimized for compute-intensive workloads
3. **Low-Latency**: Optimized for real-time and latency-sensitive applications
4. **Energy-Efficient**: Optimized for low power consumption
5. **Ubuntu-24-04-Optimized**: Specifically optimized for Ubuntu 24.04 workloads

### Optimization Categories

- **CPU Optimization**: CPU pinning, NUMA awareness, CPU model selection
- **Memory Optimization**: Huge pages, memory ballooning, KSM
- **I/O Optimization**: Disk caching, I/O scheduling, AIO backends
- **Network Optimization**: Multi-queue, TCP tuning, vhost-net

### Usage Examples

#### Applying a Performance Profile

```bash
# Using the CLI
novacron vm performance apply-profile --vm-id <vm-id> --profile "high-performance"

# Using the API
curl -X POST "http://localhost:8080/api/v1/vms/<vm-id>/performance/profile" \
  -H "Content-Type: application/json" \
  -d '{
    "profile": "high-performance"
  }'
```

#### Getting Performance Metrics

```bash
# Using the CLI
novacron vm performance metrics --vm-id <vm-id>

# Using the API
curl -X GET "http://localhost:8080/api/v1/vms/<vm-id>/performance/metrics"
```

## Migration Support

NovaCron provides comprehensive migration support for Ubuntu 24.04 VMs, allowing for seamless movement of VMs between nodes.

### Migration Features

- **Live Migration**: Migrate running VMs with minimal downtime
- **Offline Migration**: Migrate stopped VMs between nodes
- **Post-copy Migration**: Optimize migration performance with post-copy mode
- **Incremental Migration**: Efficient migration with incremental transfers
- **Migration Verification**: Ensure data integrity after migration

### Usage Examples

#### Migrating a VM

```bash
# Using the CLI
novacron vm migrate --vm-id <vm-id> --target-node "node-2" --live

# Using the API
curl -X POST "http://localhost:8080/api/v1/vms/<vm-id>/migrate" \
  -H "Content-Type: application/json" \
  -d '{
    "target_node_id": "node-2",
    "options": {
      "live_migration": true,
      "auto_start": true,
      "delete_source": false,
      "verify_after_migration": true
    }
  }'
```

#### Getting Migration Status

```bash
# Using the CLI
novacron migration status --migration-id <migration-id>

# Using the API
curl -X GET "http://localhost:8080/api/v1/migrations/<migration-id>"
```

## Enhanced Cloud-Init Integration

NovaCron provides enhanced cloud-init integration for Ubuntu 24.04 VMs, allowing for flexible and customizable VM provisioning.

### Cloud-Init Features

- **Template-Based Provisioning**: Use pre-defined templates for common VM types
- **Custom User Data**: Provide custom cloud-init user data
- **Network Configuration**: Configure VM networking through cloud-init
- **SSH Key Injection**: Automatically inject SSH keys into VMs
- **Package Installation**: Install packages during VM provisioning
- **Script Execution**: Run custom scripts during VM provisioning

### Available Templates

1. **Default**: Basic Ubuntu 24.04 VM with minimal configuration
2. **Web Server**: Ubuntu 24.04 VM configured as a web server with Nginx
3. **Database**: Ubuntu 24.04 VM configured as a database server with PostgreSQL
4. **Development**: Ubuntu 24.04 VM configured for development with various tools
5. **Minimal**: Minimal Ubuntu 24.04 VM with only essential packages

### Usage Examples

#### Creating a VM with a Cloud-Init Template

```bash
# Using the CLI
novacron vm create --name "web-server" --image "ubuntu-24.04" \
  --cloud-init-template "web-server" \
  --template-vars '{"ssh_authorized_keys": ["ssh-rsa AAAA..."]}'

# Using the API
curl -X POST "http://localhost:8080/api/v1/vms" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "web-server",
    "vcpus": 2,
    "memory_mb": 2048,
    "disk_mb": 20480,
    "image": "ubuntu-24.04",
    "cloud_init": {
      "user_data_template": "web-server",
      "template_vars": {
        "ssh_authorized_keys": ["ssh-rsa AAAA..."]
      }
    }
  }'
```

#### Getting Cloud-Init Status

```bash
# Using the CLI
novacron vm cloudinit status --vm-id <vm-id>

# Using the API
curl -X GET "http://localhost:8080/api/v1/vms/<vm-id>/cloudinit/status"
```

## API Reference

### VM Management

- `GET /api/v1/vms`: List all VMs
- `POST /api/v1/vms`: Create a new VM
- `GET /api/v1/vms/{vm_id}`: Get VM details
- `DELETE /api/v1/vms/{vm_id}`: Delete a VM
- `POST /api/v1/vms/{vm_id}/start`: Start a VM
- `POST /api/v1/vms/{vm_id}/stop`: Stop a VM
- `POST /api/v1/vms/{vm_id}/restart`: Restart a VM

### Snapshot and Backup

- `GET /api/v1/vms/{vm_id}/snapshots`: List snapshots for a VM
- `POST /api/v1/vms/{vm_id}/snapshots`: Create a snapshot
- `GET /api/v1/snapshots/{snapshot_id}`: Get snapshot details
- `DELETE /api/v1/snapshots/{snapshot_id}`: Delete a snapshot
- `POST /api/v1/vms/{vm_id}/snapshots/{snapshot_id}/restore`: Restore from a snapshot
- `GET /api/v1/vms/{vm_id}/backups`: List backups for a VM
- `POST /api/v1/vms/{vm_id}/backups`: Create a backup
- `DELETE /api/v1/backups/{backup_id}`: Delete a backup
- `POST /api/v1/vms/{vm_id}/backups/{backup_id}/restore`: Restore from a backup

### Performance Optimization

- `GET /api/v1/performance/profiles`: List available performance profiles
- `POST /api/v1/vms/{vm_id}/performance/profile`: Apply a performance profile
- `GET /api/v1/vms/{vm_id}/performance/metrics`: Get performance metrics

### Migration

- `POST /api/v1/vms/{vm_id}/migrate`: Migrate a VM
- `GET /api/v1/migrations/{migration_id}`: Get migration status
- `POST /api/v1/migrations/{migration_id}/cancel`: Cancel a migration

### Cloud-Init

- `GET /api/v1/cloudinit/templates`: List available cloud-init templates
- `GET /api/v1/vms/{vm_id}/cloudinit/status`: Get cloud-init status
- `GET /api/v1/vms/{vm_id}/cloudinit/logs`: Get cloud-init logs

## CLI Reference

### VM Management

```bash
novacron vm list
novacron vm create --name <name> --image "ubuntu-24.04" --vcpus <vcpus> --memory <memory_mb> --disk <disk_mb>
novacron vm show --vm-id <vm-id>
novacron vm delete --vm-id <vm-id>
novacron vm start --vm-id <vm-id>
novacron vm stop --vm-id <vm-id>
novacron vm restart --vm-id <vm-id>
```

### Snapshot and Backup

```bash
novacron vm snapshot list --vm-id <vm-id>
novacron vm snapshot create --vm-id <vm-id> --name <name> --description <description>
novacron vm snapshot show --snapshot-id <snapshot-id>
novacron vm snapshot delete --snapshot-id <snapshot-id>
novacron vm snapshot restore --vm-id <vm-id> --snapshot-id <snapshot-id>
novacron vm backup list --vm-id <vm-id>
novacron vm backup create --vm-id <vm-id> --name <name>
novacron vm backup delete --backup-id <backup-id>
novacron vm backup restore --vm-id <vm-id> --backup-id <backup-id>
```

### Performance Optimization

```bash
novacron performance profile list
novacron vm performance apply-profile --vm-id <vm-id> --profile <profile>
novacron vm performance metrics --vm-id <vm-id>
```

### Migration

```bash
novacron vm migrate --vm-id <vm-id> --target-node <target-node> --live
novacron migration status --migration-id <migration-id>
novacron migration cancel --migration-id <migration-id>
```

### Cloud-Init

```bash
novacron cloudinit template list
novacron vm cloudinit status --vm-id <vm-id>
novacron vm cloudinit logs --vm-id <vm-id>
```

## Troubleshooting

### Common Issues

#### VM Creation Fails

**Symptoms**: VM creation fails with an error message.

**Possible Causes**:
- Insufficient resources on the host
- Invalid cloud-init configuration
- Image not found or corrupted

**Solutions**:
1. Check host resources with `novacron node status`
2. Validate cloud-init configuration
3. Verify image exists with `novacron image list`

#### Snapshot Creation Fails

**Symptoms**: Snapshot creation fails with an error message.

**Possible Causes**:
- Insufficient disk space
- VM is in an inconsistent state
- Permissions issues

**Solutions**:
1. Check disk space with `df -h`
2. Restart the VM and try again
3. Check permissions on the snapshot directory

#### Migration Fails

**Symptoms**: VM migration fails with an error message.

**Possible Causes**:
- Network connectivity issues
- Insufficient resources on target node
- VM is using non-migratable features

**Solutions**:
1. Check network connectivity between nodes
2. Verify target node has sufficient resources
3. Check if VM is using features that prevent migration

#### Performance Issues

**Symptoms**: VM performance is poor or inconsistent.

**Possible Causes**:
- Resource contention on the host
- Suboptimal performance profile
- VM overload

**Solutions**:
1. Check host load with `novacron node status`
2. Apply a different performance profile
3. Increase VM resources or migrate to a less loaded node

### Logs and Diagnostics

- **API Logs**: `/var/log/novacron/api.log`
- **Hypervisor Logs**: `/var/log/novacron/hypervisor.log`
- **VM Logs**: `/var/log/novacron/vms/<vm-id>.log`
- **Migration Logs**: `/var/log/novacron/migrations/<migration-id>.log`
- **Cloud-Init Logs**: Access via `novacron vm cloudinit logs --vm-id <vm-id>`

### Getting Help

For additional support with Ubuntu 24.04 in NovaCron, please contact:

- **Documentation**: [https://docs.novacron.example.com](https://docs.novacron.example.com)
- **GitHub Issues**: [https://github.com/novacron/novacron/issues](https://github.com/novacron/novacron/issues)
- **Support Email**: support@novacron.example.com
