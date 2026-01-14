# NovaCron UFW Firewall Configuration

This document provides comprehensive guidance for configuring Ubuntu Firewall (UFW) for NovaCron deployment security.

## Overview

NovaCron includes comprehensive UFW application profiles and automated configuration scripts for secure firewall deployment across development, staging, and production environments.

## Quick Start

### 1. Install UFW Profiles

```bash
# Install NovaCron UFW profiles
sudo ./scripts/install-ufw-profiles.sh

# Verify profiles are available
ufw app list | grep NovaCron
```

### 2. Configure Firewall Rules

```bash
# Development environment (permissive)
sudo ./scripts/setup-ufw-rules.sh development

# Production environment (restrictive)
sudo ./scripts/setup-ufw-rules.sh production

# Staging environment (balanced)
sudo ./scripts/setup-ufw-rules.sh staging
```

### 3. Enable UFW

```bash
# Enable the firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

## Service Ports

### Core Services

| Service | Port | Protocol | Profile | Description |
|---------|------|----------|---------|-------------|
| API Server | 8090 | TCP | `NovaCron API` | REST API endpoints |
| WebSocket | 8091 | TCP | `NovaCron WebSocket` | Real-time communication |
| Frontend | 8092 | TCP | `NovaCron Frontend` | Web interface |
| AI Engine | 8093 | TCP | `NovaCron AI Engine` | AI/ML processing |

### Infrastructure Services

| Service | Port | Protocol | Profile | Description |
|---------|------|----------|---------|-------------|
| PostgreSQL | 11432 | TCP | `NovaCron Database` | Database access |
| Redis | 6379 | TCP | `NovaCron Redis` | Caching system |
| Prometheus | 9090 | TCP | `NovaCron Prometheus` | Metrics collection |
| Grafana | 3001 | TCP | `NovaCron Grafana` | Monitoring dashboard |
| Node Exporter | 9100 | TCP | `NovaCron Node Exporter` | System metrics |

### Virtualization Services

| Service | Port | Protocol | Profile | Description |
|---------|------|----------|---------|-------------|
| Hypervisor | 9000 | TCP | `NovaCron Hypervisor` | VM management |
| VNC Console | 5900-5999 | TCP | `NovaCron VNC` | VM console access |
| Migration | 49152-49215 | TCP | `NovaCron Migration` | VM migration |
| Cluster Comm | 7946 | TCP/UDP | `NovaCron Cluster` | Node coordination |

### System Services

| Service | Port | Protocol | Profile | Description |
|---------|------|----------|---------|-------------|
| SSH | 22 | TCP | `NovaCron SSH` | System administration |
| HTTP | 80 | TCP | `NovaCron HTTP` | Web access (redirect) |
| HTTPS | 443 | TCP | `NovaCron HTTPS` | Secure web access |
| DNS | 53 | TCP/UDP | `NovaCron DNS` | Service discovery |
| SNMP | 161 | UDP | `NovaCron SNMP` | Network monitoring |
| Syslog | 514 | UDP | `NovaCron Syslog` | Centralized logging |

## UFW Profile Bundles

### Service Groups

```bash
# Core services (API, WebSocket, Frontend)
ufw allow "NovaCron Core Services"

# Complete monitoring stack
ufw allow "NovaCron Monitoring Stack"

# Data services (Database + Redis)
ufw allow "NovaCron Data Layer"

# Web services (HTTP + HTTPS)
ufw allow "NovaCron Security Web"

# VM infrastructure
ufw allow "NovaCron Virtualization"

# Complete system (all services)
ufw allow "NovaCron Full Stack"
```

## Environment-Specific Configurations

### Development Environment

**Characteristics:**
- Maximum accessibility for testing
- All services accessible from any IP
- VNC console access enabled
- Open monitoring access

**Configuration:**
```bash
sudo NOVACRON_ENV=development ./scripts/setup-ufw-rules.sh development
```

**Rules Applied:**
- SSH: Open access
- Web services: Direct service access + HTTP/HTTPS
- Core services: Full access (8090-8093)
- Monitoring: Open access
- VNC: Enabled
- Data services: Open access

### Staging Environment

**Characteristics:**
- Balanced security for testing
- Monitoring restricted to private networks
- VNC access disabled
- SSH limited to private networks

**Configuration:**
```bash
sudo NOVACRON_ENV=staging ./scripts/setup-ufw-rules.sh staging
```

**Rules Applied:**
- SSH: Private networks only (RFC1918)
- Web services: HTTP/HTTPS + direct access
- Core services: Full access
- Monitoring: Private networks only
- VNC: Disabled
- Data services: Open access

### Production Environment

**Characteristics:**
- Maximum security
- Monitoring access restricted
- VNC access disabled
- Database access restricted
- SSH limited to management networks

**Configuration:**
```bash
sudo NOVACRON_ENV=production ./scripts/setup-ufw-rules.sh production
```

**Rules Applied:**
- SSH: Private networks only (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
- Web services: HTTPS (443) + HTTP redirect (80)
- Core services: Full access (API endpoints)
- Monitoring: Private networks only
- VNC: Disabled
- Data services: Localhost + private networks only

## Manual UFW Commands

### Basic Service Access

```bash
# Allow individual services
sudo ufw allow "NovaCron API"
sudo ufw allow "NovaCron WebSocket"
sudo ufw allow "NovaCron Frontend"
sudo ufw allow "NovaCron AI Engine"

# Allow monitoring
sudo ufw allow "NovaCron Prometheus"
sudo ufw allow "NovaCron Grafana"

# Allow hypervisor services
sudo ufw allow "NovaCron Hypervisor"
sudo ufw allow "NovaCron Migration"
```

### Restricted Access

```bash
# Allow service from specific IP/network
sudo ufw allow from 192.168.1.0/24 to any app "NovaCron Prometheus"

# Allow database access from specific hosts
sudo ufw allow from 10.0.1.100 to any port 11432

# Allow SSH from management network
sudo ufw allow from 192.168.100.0/24 to any port 22
```

### Port-Based Rules

```bash
# Allow specific ports
sudo ufw allow 8090/tcp  # API
sudo ufw allow 8091/tcp  # WebSocket
sudo ufw allow 9090/tcp  # Prometheus

# Allow port ranges
sudo ufw allow 5900:5999/tcp  # VNC console range
sudo ufw allow 49152:49215/tcp  # Migration ports
```

## Security Best Practices

### 1. Principle of Least Privilege

```bash
# Start with deny-all policy
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Add only required services
sudo ufw allow "NovaCron Core Services"
sudo ufw allow ssh  # For management
```

### 2. Network Segmentation

```bash
# Management network access only
sudo ufw allow from 192.168.100.0/24 to any app "NovaCron SSH"
sudo ufw allow from 192.168.100.0/24 to any app "NovaCron Prometheus"

# Application network access
sudo ufw allow from 10.0.1.0/24 to any app "NovaCron API"
```

### 3. Rate Limiting

```bash
# Limit connection attempts
sudo ufw limit ssh
sudo ufw limit 80/tcp
sudo ufw limit 443/tcp
```

### 4. Logging and Monitoring

```bash
# Enable UFW logging
sudo ufw logging on

# View UFW logs
sudo tail -f /var/log/ufw.log

# Check denied connections
sudo grep -i deny /var/log/ufw.log
```

## Troubleshooting

### Common Issues

#### 1. Service Not Accessible

```bash
# Check UFW status
sudo ufw status verbose

# Check specific profile
sudo ufw app info "NovaCron API"

# Test connectivity
curl -v http://localhost:8090/health
```

#### 2. Profile Not Found

```bash
# Reinstall profiles
sudo ./scripts/install-ufw-profiles.sh

# Reload application profiles
sudo ufw app update all

# List available profiles
sudo ufw app list | grep NovaCron
```

#### 3. SSH Access Lost

```bash
# Disable UFW temporarily (from console)
sudo ufw disable

# Allow SSH from your IP
sudo ufw allow from YOUR_IP to any port 22

# Re-enable UFW
sudo ufw enable
```

### Diagnostic Commands

```bash
# Show detailed UFW status
sudo ufw status verbose

# Show UFW logs
sudo journalctl -u ufw -f

# List all UFW rules by number
sudo ufw status numbered

# Show raw iptables rules
sudo iptables -L -n -v
```

## Advanced Configuration

### Custom Profile Creation

Create custom application profiles in `/etc/ufw/applications.d/`:

```bash
# Custom profile for additional service
sudo tee /etc/ufw/applications.d/custom-service <<EOF
[Custom Service]
title=Custom Application
description=My custom application
ports=8080/tcp
EOF

# Reload profiles
sudo ufw app update all
```

### Environment Variables

```bash
# Configure behavior via environment
export NOVACRON_ENV=production      # Environment type
export ALLOW_ALL_SSH=false         # SSH access control
export RESTRICT_MONITORING=true    # Monitoring access
export ENABLE_VNC_ACCESS=false     # VNC console access

# Run with custom settings
sudo -E ./scripts/setup-ufw-rules.sh production
```

### Integration with Configuration Management

#### Ansible Integration

```yaml
- name: Install NovaCron UFW profiles
  script: scripts/install-ufw-profiles.sh
  become: yes

- name: Configure UFW rules for production
  script: scripts/setup-ufw-rules.sh production
  become: yes
  environment:
    NOVACRON_ENV: production
    RESTRICT_MONITORING: true
```

#### Docker Integration

```dockerfile
# Add to Dockerfile for containerized deployments
RUN ./scripts/install-ufw-profiles.sh && \
    ./scripts/setup-ufw-rules.sh production && \
    ufw enable
```

## Maintenance

### Regular Tasks

```bash
# Check UFW status weekly
sudo ufw status verbose

# Review logs for unusual activity
sudo grep -i "BLOCK\|DENY" /var/log/ufw.log | tail -20

# Update profiles after configuration changes
sudo ufw app update all
```

### Profile Updates

When service ports change, update the profiles:

1. Edit `/home/kp/novacron/configs/ufw/applications.d/novacron`
2. Run: `sudo ./scripts/install-ufw-profiles.sh`
3. Update rules: `sudo ./scripts/setup-ufw-rules.sh [environment]`

## References

- [Ubuntu UFW Documentation](https://help.ubuntu.com/community/UFW)
- [UFW Manual Pages](https://manpages.ubuntu.com/manpages/focal/man8/ufw.8.html)
- [NovaCron Security Documentation](./SECURITY.md)
- [NovaCron Deployment Guide](./DEPLOYMENT.md)