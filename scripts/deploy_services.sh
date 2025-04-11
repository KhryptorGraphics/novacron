#!/bin/bash
# Script to deploy NovaCron services with Ubuntu 24.04 support

set -e

# Configuration
CONFIG_DIR="/etc/novacron"
SYSTEMD_DIR="/etc/systemd/system"
LOG_DIR="/var/log/novacron"
USER="novacron"
GROUP="novacron"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
}

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    print_error "This script must be run as root"
    exit 1
fi

# Create directories
print_status "Creating directories..."
mkdir -p "$CONFIG_DIR"
mkdir -p "$LOG_DIR"

# Copy configuration files
print_status "Copying configuration files..."
cp config/production/api.yaml "$CONFIG_DIR/api.yaml"
cp config/production/hypervisor.yaml "$CONFIG_DIR/hypervisor.yaml"

# Copy systemd service files
print_status "Installing systemd service files..."
cp config/systemd/novacron-api.service "$SYSTEMD_DIR/"
cp config/systemd/novacron-hypervisor.service "$SYSTEMD_DIR/"
cp config/systemd/novacron-ubuntu-24-04-monitor.service "$SYSTEMD_DIR/"

# Set permissions
print_status "Setting permissions..."
chown -R "$USER:$GROUP" "$CONFIG_DIR" || true
chown -R "$USER:$GROUP" "$LOG_DIR" || true
chmod 750 "$CONFIG_DIR"
chmod 640 "$CONFIG_DIR"/*.yaml

# Reload systemd
print_status "Reloading systemd..."
systemctl daemon-reload

# Restart services
print_status "Restarting services..."
systemctl restart novacron-hypervisor.service || systemctl start novacron-hypervisor.service
systemctl restart novacron-api.service || systemctl start novacron-api.service
systemctl restart novacron-ubuntu-24-04-monitor.service || systemctl start novacron-ubuntu-24-04-monitor.service

# Enable services
print_status "Enabling services to start at boot..."
systemctl enable novacron-hypervisor.service
systemctl enable novacron-api.service
systemctl enable novacron-ubuntu-24-04-monitor.service

# Check service status
print_status "Checking service status..."
systemctl status novacron-hypervisor.service --no-pager
systemctl status novacron-api.service --no-pager
systemctl status novacron-ubuntu-24-04-monitor.service --no-pager

print_status "Services deployed successfully!"
print_status "Ubuntu 24.04 support is now available in NovaCron"

exit 0
