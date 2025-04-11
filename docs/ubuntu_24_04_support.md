# Ubuntu 24.04 LTS Support in NovaCron

This document describes how to use Ubuntu 24.04 LTS (Noble Numbat) with NovaCron's distributed hypervisor.

## Overview

NovaCron now supports Ubuntu 24.04 LTS as a guest operating system for virtual machines. This support includes:

1. Cloud provider integration (AWS, Azure, GCP)
2. KVM-based VMs with Ubuntu 24.04 images
3. VM templates for Ubuntu 24.04
4. Cloud-init integration for VM customization

## Prerequisites

To use Ubuntu 24.04 with NovaCron, you need:

1. NovaCron hypervisor with KVM support enabled
2. At least 20GB of free disk space for VM images
3. Internet access to download Ubuntu cloud images (or a local mirror)

## Setting Up Ubuntu 24.04 Images

### Automatic Setup

The easiest way to set up Ubuntu 24.04 support is to use the provided script:

```bash
sudo ./scripts/prepare_ubuntu_24_04.sh
```

This script will:
1. Download the official Ubuntu 24.04 cloud image
2. Convert it to qcow2 format if needed
3. Resize it to 20GB
4. Store it in the NovaCron image directory

### Manual Setup

If you prefer to set up the image manually:

1. Download the Ubuntu 24.04 cloud image:
   ```bash
   wget https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-amd64.img
   ```

2. Convert to qcow2 format if needed:
   ```bash
   qemu-img convert -f qcow2 -O qcow2 noble-server-cloudimg-amd64.img ubuntu-24.04-server-cloudimg-amd64.qcow2
   ```

3. Resize the image:
   ```bash
   qemu-img resize ubuntu-24.04-server-cloudimg-amd64.qcow2 20G
   ```

4. Move to NovaCron image directory:
   ```bash
   sudo mv ubuntu-24.04-server-cloudimg-amd64.qcow2 /var/lib/novacron/images/
   sudo chmod 644 /var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2
   ```

## Creating Ubuntu 24.04 VMs

### Using the API

To create an Ubuntu 24.04 VM using the NovaCron API:

```json
POST /api/v1/vms
{
  "name": "ubuntu-24-04-vm",
  "spec": {
    "vcpu": 2,
    "memory_mb": 2048,
    "disk_mb": 20480,
    "type": "kvm",
    "image": "/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2",
    "networks": [
      {
        "network_id": "default"
      }
    ],
    "env": {
      "OS_VERSION": "24.04",
      "OS_NAME": "Ubuntu",
      "OS_CODENAME": "Noble Numbat"
    },
    "labels": {
      "os": "ubuntu",
      "version": "24.04",
      "lts": "true"
    }
  },
  "tags": {
    "purpose": "web-server",
    "environment": "production"
  }
}
```

### Using the CLI

To create an Ubuntu 24.04 VM using the NovaCron CLI:

```bash
novacron vm create --name ubuntu-24-04-vm \
  --vcpu 2 \
  --memory 2048 \
  --disk 20480 \
  --type kvm \
  --image /var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2 \
  --network default \
  --label os=ubuntu \
  --label version=24.04 \
  --label lts=true \
  --tag purpose=web-server \
  --tag environment=production
```

### Using Templates

NovaCron provides a pre-configured template for Ubuntu 24.04:

```bash
novacron template list
# Find the ID of the Ubuntu 24.04 template

novacron vm create-from-template --template-id ubuntu-server-24.04 \
  --name my-ubuntu-vm \
  --param hostname=web-server-1 \
  --param ssh_key="ssh-rsa AAAAB3NzaC1..."
```

## Cloud-Init Integration

Ubuntu 24.04 VMs support cloud-init for initial configuration. You can provide user data when creating a VM:

```bash
novacron vm create --name ubuntu-24-04-vm \
  --image /var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2 \
  --cloud-init-user-data /path/to/user-data.yaml
```

Example user-data.yaml:
```yaml
#cloud-config
hostname: web-server-1
manage_etc_hosts: true
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - ssh-rsa AAAAB3NzaC1...
package_update: true
package_upgrade: true
packages:
  - nginx
  - docker.io
  - fail2ban
```

## Known Issues and Limitations

1. GPU passthrough requires additional configuration
2. Nested virtualization may not work in all environments
3. Some cloud-specific features may not be available in the local hypervisor

## Troubleshooting

### VM Fails to Start

If the VM fails to start, check:

1. KVM is properly enabled on the host:
   ```bash
   ls -l /dev/kvm
   ```

2. The image file exists and has correct permissions:
   ```bash
   ls -l /var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2
   ```

3. The hypervisor logs:
   ```bash
   journalctl -u novacron-hypervisor
   ```

### VM Starts But Cannot Connect

If the VM starts but you cannot connect:

1. Check the network configuration:
   ```bash
   novacron vm show-network <vm-id>
   ```

2. Verify cloud-init completed successfully:
   ```bash
   novacron vm get-console-log <vm-id> | grep cloud-init
   ```

## Support

For additional support with Ubuntu 24.04 on NovaCron, please contact the NovaCron support team or open an issue on the GitHub repository.
