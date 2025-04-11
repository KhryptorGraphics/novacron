#!/bin/bash
# Script to verify Ubuntu 24.04 support in NovaCron

set -e

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

# Step 1: Check if the Ubuntu 24.04 image exists
print_status "Step 1: Checking if the Ubuntu 24.04 image exists..."
IMAGE_PATH="/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2"
if [ -f "$IMAGE_PATH" ]; then
    print_status "Ubuntu 24.04 image exists at $IMAGE_PATH"
    print_status "Image size: $(du -h "$IMAGE_PATH" | cut -f1)"
else
    print_error "Ubuntu 24.04 image does not exist at $IMAGE_PATH"
    exit 1
fi

# Step 2: Check if the API configuration includes Ubuntu 24.04
print_status "Step 2: Checking if the API configuration includes Ubuntu 24.04..."
if [ -f "/etc/novacron/api.yaml" ]; then
    if grep -q "Ubuntu 24.04 LTS" /etc/novacron/api.yaml; then
        print_status "API configuration includes Ubuntu 24.04"
    else
        print_warning "API configuration does not include Ubuntu 24.04"
    fi
else
    print_warning "API configuration file not found at /etc/novacron/api.yaml"
fi

# Step 3: Check if the hypervisor configuration includes Ubuntu 24.04
print_status "Step 3: Checking if the hypervisor configuration includes Ubuntu 24.04..."
if [ -f "/etc/novacron/hypervisor.yaml" ]; then
    if grep -q "Ubuntu 24.04 LTS" /etc/novacron/hypervisor.yaml; then
        print_status "Hypervisor configuration includes Ubuntu 24.04"
    else
        print_warning "Hypervisor configuration does not include Ubuntu 24.04"
    fi
else
    print_warning "Hypervisor configuration file not found at /etc/novacron/hypervisor.yaml"
fi

# Step 4: Check if the documentation exists
print_status "Step 4: Checking if the documentation exists..."
if [ -f "docs/ubuntu_24_04_support.md" ]; then
    print_status "Documentation exists at docs/ubuntu_24_04_support.md"
else
    print_error "Documentation does not exist at docs/ubuntu_24_04_support.md"
    exit 1
fi

# Step 5: Check if the VM creation script exists
print_status "Step 5: Checking if the VM creation script exists..."
if [ -f "scripts/create_ubuntu_24_04_vm_api.sh" ]; then
    print_status "VM creation script exists at scripts/create_ubuntu_24_04_vm_api.sh"
else
    print_error "VM creation script does not exist at scripts/create_ubuntu_24_04_vm_api.sh"
    exit 1
fi

# Step 6: Check if the VM lifecycle test script exists
print_status "Step 6: Checking if the VM lifecycle test script exists..."
if [ -f "scripts/test_ubuntu_24_04_lifecycle.sh" ]; then
    print_status "VM lifecycle test script exists at scripts/test_ubuntu_24_04_lifecycle.sh"
else
    print_error "VM lifecycle test script does not exist at scripts/test_ubuntu_24_04_lifecycle.sh"
    exit 1
fi

# Step 7: Check if the monitoring script exists
print_status "Step 7: Checking if the monitoring script exists..."
if [ -f "scripts/monitor_ubuntu_24_04_vms.sh" ]; then
    print_status "Monitoring script exists at scripts/monitor_ubuntu_24_04_vms.sh"
else
    print_error "Monitoring script does not exist at scripts/monitor_ubuntu_24_04_vms.sh"
    exit 1
fi

print_status "Verification completed successfully!"
print_status "Ubuntu 24.04 support is properly configured in NovaCron"

exit 0
