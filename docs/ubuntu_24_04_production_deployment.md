# Ubuntu 24.04 Production Deployment Guide

This guide provides step-by-step instructions for deploying Ubuntu 24.04 support in a NovaCron production environment.

## Prerequisites

Before deploying Ubuntu 24.04 support in production, ensure you have:

1. A server with the following specifications:
   - At least 4 CPU cores
   - At least 8GB RAM
   - At least 100GB free disk space
   - Ubuntu 22.04 LTS or later installed

2. The following software installed:
   - QEMU/KVM
   - libvirt
   - Python 3.10 or later
   - curl, wget, jq

3. Network connectivity:
   - Outbound internet access for downloading images
   - Firewall rules allowing access to ports 8080 (API) and 9000 (Hypervisor)

4. User permissions:
   - Root or sudo access for installation

## Deployment Steps

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/novacron.git
cd novacron
```

### Step 2: Configure Environment Variables

Create a `.env` file with the necessary environment variables:

```bash
cat > .env << EOF
# Database Configuration
DB_HOST=localhost
DB_NAME=novacron
DB_USER=novacron
DB_PASSWORD=your_secure_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PASSWORD=your_secure_redis_password

# Node Configuration
NODE_ID=node-$(hostname)
NODE_NAME=$(hostname)
CLUSTER_NAME=production
REGION=us-east
ZONE=us-east-1a

# JWT Secret
JWT_SECRET=$(openssl rand -hex 32)
EOF
```

Load the environment variables:

```bash
source .env
```

### Step 3: Run the Production Deployment Script

The production deployment script will:
- Install necessary dependencies
- Prepare the Ubuntu 24.04 image
- Configure the services
- Start the services

```bash
sudo ./scripts/production_deploy.sh
```

This script may take several minutes to complete, especially when downloading the Ubuntu 24.04 image.

### Step 4: Verify the Deployment

After the deployment script completes, verify that everything is working correctly:

```bash
./scripts/production_test.sh
```

This script will:
- Check if all services are running
- Verify that the Ubuntu 24.04 image is available
- Create a test VM with Ubuntu 24.04
- Test the VM lifecycle (start, stop, delete)
- Generate a test report

### Step 5: Monitor the Deployment

Start the monitoring dashboard to monitor Ubuntu 24.04 VMs:

```bash
./scripts/ubuntu_24_04_dashboard.sh
```

This dashboard provides real-time information about:
- Service status
- VM count and states
- Resource usage
- Recent events

For continuous monitoring, set up the monitoring service:

```bash
sudo systemctl enable novacron-ubuntu-24-04-monitor.service
sudo systemctl start novacron-ubuntu-24-04-monitor.service
```

## Post-Deployment Tasks

### Security Hardening

1. Configure firewall rules:
   ```bash
   sudo ufw allow 8080/tcp  # API
   sudo ufw allow 9000/tcp  # Hypervisor
   sudo ufw enable
   ```

2. Set up SSL/TLS:
   ```bash
   # Install certbot
   sudo apt-get install certbot
   
   # Get certificates
   sudo certbot certonly --standalone -d api.novacron.example.com
   
   # Configure NovaCron to use SSL
   # Edit /etc/novacron/api.yaml and update the server section
   ```

3. Configure authentication:
   ```bash
   # Edit /etc/novacron/api.yaml and ensure auth.enabled is set to true
   ```

### Backup Configuration

1. Set up regular backups of the configuration:
   ```bash
   # Create a backup script
   cat > /opt/novacron/scripts/backup.sh << EOF
   #!/bin/bash
   BACKUP_DIR="/var/backups/novacron"
   mkdir -p \$BACKUP_DIR
   cp -r /etc/novacron \$BACKUP_DIR/config-\$(date +%Y%m%d)
   cp -r /var/lib/novacron/images \$BACKUP_DIR/images-\$(date +%Y%m%d)
   EOF
   
   # Make it executable
   chmod +x /opt/novacron/scripts/backup.sh
   
   # Add to crontab
   echo "0 2 * * * /opt/novacron/scripts/backup.sh" | sudo tee -a /etc/crontab
   ```

### Performance Tuning

1. Adjust KVM settings for better performance:
   ```bash
   # Edit /etc/modprobe.d/kvm.conf
   echo "options kvm_intel nested=1" | sudo tee -a /etc/modprobe.d/kvm.conf
   ```

2. Optimize libvirt settings:
   ```bash
   # Edit /etc/libvirt/qemu.conf
   # Increase memory and CPU allocation
   ```

## Troubleshooting

### Common Issues

#### Services Fail to Start

If services fail to start, check the logs:

```bash
journalctl -u novacron-hypervisor.service -n 100
journalctl -u novacron-api.service -n 100
```

#### VM Creation Fails

If VM creation fails, check:

1. The Ubuntu 24.04 image exists and has correct permissions:
   ```bash
   ls -la /var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2
   ```

2. The API logs for errors:
   ```bash
   tail -n 100 /var/log/novacron/api.log
   ```

3. The hypervisor logs for errors:
   ```bash
   tail -n 100 /var/log/novacron/hypervisor.log
   ```

#### Network Issues

If VMs have network issues:

1. Check the bridge configuration:
   ```bash
   ip link show
   brctl show
   ```

2. Verify that the default network is defined:
   ```bash
   virsh net-list --all
   ```

## Support

For additional support with Ubuntu 24.04 deployment in NovaCron, please contact:

- Email: support@novacron.example.com
- GitHub Issues: https://github.com/yourusername/novacron/issues
- Documentation: https://docs.novacron.example.com

## References

- [Ubuntu 24.04 LTS Documentation](https://help.ubuntu.com/24.04/)
- [NovaCron API Documentation](https://docs.novacron.example.com/api)
- [KVM Documentation](https://www.linux-kvm.org/page/Documents)
