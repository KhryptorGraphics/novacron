# Ubuntu 24.04 Rollback Plan

This document outlines the steps to roll back the Ubuntu 24.04 support in NovaCron if issues are encountered in production.

## When to Roll Back

Consider rolling back the Ubuntu 24.04 support if:

1. Critical services fail to start or become unstable after deployment
2. Ubuntu 24.04 VMs consistently fail to create or start
3. Performance issues are observed with Ubuntu 24.04 VMs
4. Security vulnerabilities are discovered in the Ubuntu 24.04 image

## Pre-Rollback Checklist

Before initiating a rollback, perform the following checks:

1. Verify that the issues are specifically related to Ubuntu 24.04 support
2. Check if the issues can be resolved with configuration changes
3. Ensure you have backups of all configuration files
4. Notify users of the planned rollback
5. Schedule the rollback during a maintenance window if possible

## Rollback Procedure

### Step 1: Stop All Ubuntu 24.04 VMs

```bash
#!/bin/bash
# Script to stop all Ubuntu 24.04 VMs

API_ENDPOINT="http://localhost:8080"

# Get all VMs
VM_LIST=$(curl -s "$API_ENDPOINT/api/v1/vms")

# Filter Ubuntu 24.04 VMs
UBUNTU_24_04_VMS=$(echo "$VM_LIST" | jq '[.[] | select(.spec.image | contains("ubuntu-24.04"))]')

# Stop each VM
echo "$UBUNTU_24_04_VMS" | jq -r '.[] | .id' | while read -r VM_ID; do
    echo "Stopping VM $VM_ID..."
    curl -s -X POST "$API_ENDPOINT/api/v1/vms/$VM_ID/stop"
    sleep 5
done

echo "All Ubuntu 24.04 VMs have been stopped"
```

### Step 2: Disable Ubuntu 24.04 in Configuration Files

Edit the API configuration file:

```bash
sudo cp /etc/novacron/api.yaml /etc/novacron/api.yaml.bak
sudo sed -i '/Ubuntu 24.04 LTS/,+5d' /etc/novacron/api.yaml
```

Edit the hypervisor configuration file:

```bash
sudo cp /etc/novacron/hypervisor.yaml /etc/novacron/hypervisor.yaml.bak
sudo sed -i '/Ubuntu 24.04 LTS/,+5d' /etc/novacron/hypervisor.yaml
```

### Step 3: Restart Services

```bash
sudo systemctl restart novacron-api.service
sudo systemctl restart novacron-hypervisor.service
```

### Step 4: Stop and Disable the Monitoring Service

```bash
sudo systemctl stop novacron-ubuntu-24-04-monitor.service
sudo systemctl disable novacron-ubuntu-24-04-monitor.service
```

### Step 5: Verify Rollback

```bash
# Check if services are running
systemctl status novacron-api.service
systemctl status novacron-hypervisor.service

# Check if Ubuntu 24.04 is no longer in the templates
curl -s http://localhost:8080/api/v1/templates | grep -i "ubuntu 24.04"
```

### Step 6: Notify Users

Notify users that Ubuntu 24.04 support has been temporarily disabled and provide alternative options:

```
Subject: Important Notice: Temporary Suspension of Ubuntu 24.04 Support

Dear NovaCron Users,

We have temporarily suspended support for Ubuntu 24.04 VMs due to [specific issue]. 
Our team is working to resolve the issue as quickly as possible.

In the meantime, please use Ubuntu 22.04 LTS for your VMs, which remains fully supported.

We apologize for any inconvenience this may cause and will notify you as soon as 
Ubuntu 24.04 support is restored.

Thank you for your understanding.

The NovaCron Team
```

## Post-Rollback Actions

### Investigate the Issue

1. Collect logs from affected systems:
   ```bash
   journalctl -u novacron-api.service -n 1000 > api-logs.txt
   journalctl -u novacron-hypervisor.service -n 1000 > hypervisor-logs.txt
   cat /var/log/novacron/ubuntu_24_04_monitoring.log > monitoring-logs.txt
   ```

2. Analyze the logs to identify the root cause

3. Test potential fixes in a development environment

### Develop a Fix

1. Create a fix for the identified issue

2. Test the fix thoroughly in a development environment

3. Document the fix and update the deployment scripts

### Re-deploy with Fixes

1. Update the deployment scripts with the fixes

2. Follow the deployment guide to re-deploy Ubuntu 24.04 support

3. Monitor the deployment closely for any issues

## Emergency Contact Information

In case of critical issues during rollback:

- Primary Contact: John Doe, john.doe@novacron.example.com, +1-555-123-4567
- Secondary Contact: Jane Smith, jane.smith@novacron.example.com, +1-555-987-6543
- Operations Team: ops@novacron.example.com, +1-555-789-0123

## Rollback Completion Checklist

- [ ] All Ubuntu 24.04 VMs have been stopped
- [ ] Ubuntu 24.04 has been removed from configuration files
- [ ] Services have been restarted
- [ ] Monitoring service has been disabled
- [ ] Users have been notified
- [ ] Logs have been collected for investigation
- [ ] Rollback has been verified
