#!/bin/bash
# Comprehensive script to deploy Ubuntu 24.04 support in NovaCron production environment

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
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> deployment.log
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> deployment.log
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> deployment.log
}

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    print_error "This script must be run as root"
    exit 1
fi

# Start logging
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting NovaCron production deployment with Ubuntu 24.04 support" > deployment.log

# Step 1: Create necessary directories and users
print_status "Step 1: Creating necessary directories and users..."

# Create directories
mkdir -p $NOVACRON_HOME
mkdir -p $CONFIG_DIR
mkdir -p $IMAGES_DIR
mkdir -p $LOGS_DIR
mkdir -p $IMAGES_DIR/templates
mkdir -p /var/lib/novacron/vms
mkdir -p /var/lib/novacron/volumes
mkdir -p /var/lib/novacron/backups
mkdir -p /var/lib/novacron/temp

# Create user and group if they don't exist
if ! getent group $GROUP > /dev/null; then
    print_status "Creating group $GROUP..."
    groupadd $GROUP
fi

if ! getent passwd $USER > /dev/null; then
    print_status "Creating user $USER..."
    useradd -m -g $GROUP -s /bin/bash $USER
fi

# Step 2: Install dependencies
print_status "Step 2: Installing dependencies..."
apt-get update
apt-get install -y qemu-kvm libvirt-daemon-system virtinst bridge-utils \
    libvirt-clients python3-pip python3-venv jq curl wget \
    qemu-utils cloud-image-utils

# Step 3: Prepare Ubuntu 24.04 image
print_status "Step 3: Preparing Ubuntu 24.04 image..."
if [ ! -f "$IMAGES_DIR/ubuntu-24.04-server-cloudimg-amd64.qcow2" ]; then
    print_status "Downloading and preparing Ubuntu 24.04 image..."
    ./scripts/prepare_ubuntu_24_04.sh
else
    print_status "Ubuntu 24.04 image already exists, checking integrity..."
    qemu-img check $IMAGES_DIR/ubuntu-24.04-server-cloudimg-amd64.qcow2
    if [ $? -ne 0 ]; then
        print_warning "Image integrity check failed, re-downloading..."
        rm -f $IMAGES_DIR/ubuntu-24.04-server-cloudimg-amd64.qcow2
        ./scripts/prepare_ubuntu_24_04.sh
    else
        print_status "Image integrity check passed"
    fi
fi

# Step 4: Copy application files
print_status "Step 4: Copying application files..."
# Copy backend files
cp -r backend/* $NOVACRON_HOME/
# Copy scripts
mkdir -p $NOVACRON_HOME/scripts
cp scripts/monitor_ubuntu_24_04_vms.sh $NOVACRON_HOME/scripts/
chmod +x $NOVACRON_HOME/scripts/monitor_ubuntu_24_04_vms.sh

# Step 5: Copy configuration files
print_status "Step 5: Copying configuration files..."
cp config/production/api.yaml $CONFIG_DIR/api.yaml
cp config/production/hypervisor.yaml $CONFIG_DIR/hypervisor.yaml

# Replace environment variables in configuration files
print_status "Configuring environment variables in configuration files..."
# Generate a random JWT secret if not provided
JWT_SECRET=${JWT_SECRET:-$(openssl rand -hex 32)}
# Set default values for other variables
DB_HOST=${DB_HOST:-"localhost"}
DB_NAME=${DB_NAME:-"novacron"}
DB_USER=${DB_USER:-"novacron"}
DB_PASSWORD=${DB_PASSWORD:-"$(openssl rand -hex 16)"}
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PASSWORD=${REDIS_PASSWORD:-"$(openssl rand -hex 16)"}
NODE_ID=${NODE_ID:-"node-$(hostname)"}
NODE_NAME=${NODE_NAME:-"$(hostname)"}
CLUSTER_NAME=${CLUSTER_NAME:-"production"}
REGION=${REGION:-"default"}
ZONE=${ZONE:-"default"}

# Replace variables in configuration files
sed -i "s/\${JWT_SECRET}/$JWT_SECRET/g" $CONFIG_DIR/api.yaml
sed -i "s/\${DB_HOST}/$DB_HOST/g" $CONFIG_DIR/api.yaml
sed -i "s/\${DB_NAME}/$DB_NAME/g" $CONFIG_DIR/api.yaml
sed -i "s/\${DB_USER}/$DB_USER/g" $CONFIG_DIR/api.yaml
sed -i "s/\${DB_PASSWORD}/$DB_PASSWORD/g" $CONFIG_DIR/api.yaml
sed -i "s/\${REDIS_HOST}/$REDIS_HOST/g" $CONFIG_DIR/api.yaml
sed -i "s/\${REDIS_PASSWORD}/$REDIS_PASSWORD/g" $CONFIG_DIR/api.yaml

sed -i "s/\${NODE_ID}/$NODE_ID/g" $CONFIG_DIR/hypervisor.yaml
sed -i "s/\${NODE_NAME}/$NODE_NAME/g" $CONFIG_DIR/hypervisor.yaml
sed -i "s/\${CLUSTER_NAME}/$CLUSTER_NAME/g" $CONFIG_DIR/hypervisor.yaml
sed -i "s/\${REGION}/$REGION/g" $CONFIG_DIR/hypervisor.yaml
sed -i "s/\${ZONE}/$ZONE/g" $CONFIG_DIR/hypervisor.yaml

# Step 6: Install systemd service files
print_status "Step 6: Installing systemd service files..."
cp config/systemd/novacron-api.service $SYSTEMD_DIR/
cp config/systemd/novacron-hypervisor.service $SYSTEMD_DIR/
cp config/systemd/novacron-ubuntu-24-04-monitor.service $SYSTEMD_DIR/

# Update paths in service files
sed -i "s|/opt/novacron|$NOVACRON_HOME|g" $SYSTEMD_DIR/novacron-api.service
sed -i "s|/opt/novacron|$NOVACRON_HOME|g" $SYSTEMD_DIR/novacron-hypervisor.service
sed -i "s|/opt/novacron|$NOVACRON_HOME|g" $SYSTEMD_DIR/novacron-ubuntu-24-04-monitor.service

# Step 7: Set permissions
print_status "Step 7: Setting permissions..."
chown -R $USER:$GROUP $NOVACRON_HOME
chown -R $USER:$GROUP $CONFIG_DIR
chown -R $USER:$GROUP $IMAGES_DIR
chown -R $USER:$GROUP $LOGS_DIR
chown -R $USER:$GROUP /var/lib/novacron
chmod 750 $CONFIG_DIR
chmod 640 $CONFIG_DIR/*.yaml
chmod 755 $NOVACRON_HOME/scripts/*.sh

# Step 8: Reload systemd and start services
print_status "Step 8: Reloading systemd and starting services..."
systemctl daemon-reload
systemctl enable novacron-hypervisor.service
systemctl enable novacron-api.service
systemctl enable novacron-ubuntu-24-04-monitor.service

systemctl start novacron-hypervisor.service
sleep 5  # Wait for hypervisor to start
systemctl start novacron-api.service
sleep 3  # Wait for API to start
systemctl start novacron-ubuntu-24-04-monitor.service

# Step 9: Verify services are running
print_status "Step 9: Verifying services are running..."
hypervisor_status=$(systemctl is-active novacron-hypervisor.service)
api_status=$(systemctl is-active novacron-api.service)
monitor_status=$(systemctl is-active novacron-ubuntu-24-04-monitor.service)

if [ "$hypervisor_status" = "active" ] && [ "$api_status" = "active" ] && [ "$monitor_status" = "active" ]; then
    print_status "All services are running"
else
    print_warning "Some services are not running:"
    print_warning "Hypervisor: $hypervisor_status"
    print_warning "API: $api_status"
    print_warning "Monitor: $monitor_status"
    
    print_status "Checking service logs..."
    journalctl -u novacron-hypervisor.service -n 20 --no-pager
    journalctl -u novacron-api.service -n 20 --no-pager
    journalctl -u novacron-ubuntu-24-04-monitor.service -n 20 --no-pager
fi

# Step 10: Save deployment information
print_status "Step 10: Saving deployment information..."
cat > $NOVACRON_HOME/deployment-info.txt << EOF
NovaCron Deployment Information
==============================
Deployment Date: $(date)
Deployment User: $(whoami)
Hostname: $(hostname)

Services:
- Hypervisor: $hypervisor_status
- API: $api_status
- Monitor: $monitor_status

Configuration:
- Config Directory: $CONFIG_DIR
- Images Directory: $IMAGES_DIR
- Logs Directory: $LOGS_DIR

Ubuntu 24.04 Support:
- Image Path: $IMAGES_DIR/ubuntu-24.04-server-cloudimg-amd64.qcow2
- Image Size: $(du -h $IMAGES_DIR/ubuntu-24.04-server-cloudimg-amd64.qcow2 | cut -f1)

Environment Variables:
- NODE_ID: $NODE_ID
- NODE_NAME: $NODE_NAME
- CLUSTER_NAME: $CLUSTER_NAME
- REGION: $REGION
- ZONE: $ZONE

Database:
- Host: $DB_HOST
- Name: $DB_NAME
- User: $DB_USER

Redis:
- Host: $REDIS_HOST

For support, contact the NovaCron team.
EOF

chown $USER:$GROUP $NOVACRON_HOME/deployment-info.txt
chmod 640 $NOVACRON_HOME/deployment-info.txt

print_status "Deployment completed successfully!"
print_status "Ubuntu 24.04 support is now available in NovaCron"
print_status "Deployment information saved to $NOVACRON_HOME/deployment-info.txt"
print_status "Deployment log saved to $(pwd)/deployment.log"

exit 0
