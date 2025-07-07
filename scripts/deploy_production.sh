#!/bin/bash
# Script to deploy NovaCron to production with Ubuntu 24.04 support

set -e

# Configuration
NOVACRON_HOME="/opt/novacron"
CONFIG_DIR="/etc/novacron"
SYSTEMD_DIR="/etc/systemd/system"
IMAGES_DIR="/var/lib/novacron/images"
LOGS_DIR="/var/log/novacron"
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
mkdir -p $NOVACRON_HOME
mkdir -p $CONFIG_DIR
mkdir -p $IMAGES_DIR
mkdir -p $LOGS_DIR
mkdir -p $IMAGES_DIR/templates

# Create user and group if they don't exist
if ! getent group $GROUP > /dev/null; then
    print_status "Creating group $GROUP..."
    groupadd $GROUP
fi

if ! getent passwd $USER > /dev/null; then
    print_status "Creating user $USER..."
    useradd -m -g $GROUP -s /bin/bash $USER
fi

# Copy configuration files
print_status "Copying configuration files..."
cp config/production/api.yaml $CONFIG_DIR/api.yaml
cp config/production/hypervisor.yaml $CONFIG_DIR/hypervisor.yaml

# Copy systemd service files
print_status "Installing systemd service files..."
cp config/systemd/novacron-api.service $SYSTEMD_DIR/
cp config/systemd/novacron-hypervisor.service $SYSTEMD_DIR/

# Reload systemd
systemctl daemon-reload

# Download Ubuntu 24.04 image if it doesn't exist
if [ ! -f "$IMAGES_DIR/ubuntu-24.04-server-cloudimg-amd64.qcow2" ]; then
    print_status "Downloading Ubuntu 24.04 image..."
    ./scripts/prepare_ubuntu_24_04.sh
else
    print_warning "Ubuntu 24.04 image already exists, skipping download"
fi

# Set permissions
print_status "Setting permissions..."
chown -R $USER:$GROUP $NOVACRON_HOME
chown -R $USER:$GROUP $CONFIG_DIR
chown -R $USER:$GROUP $IMAGES_DIR
chown -R $USER:$GROUP $LOGS_DIR
chmod 750 $CONFIG_DIR
chmod 640 $CONFIG_DIR/*.yaml

# Start services
print_status "Starting services..."
systemctl enable novacron-hypervisor.service
systemctl enable novacron-api.service
systemctl start novacron-hypervisor.service
systemctl start novacron-api.service

# Check service status
print_status "Checking service status..."
systemctl status novacron-hypervisor.service --no-pager
systemctl status novacron-api.service --no-pager

print_status "Deployment complete!"
print_status "Ubuntu 24.04 support is now available in NovaCron"
print_status "You can access the API at http://localhost:8090"
print_status "You can access the hypervisor at http://localhost:9000"

exit 0
