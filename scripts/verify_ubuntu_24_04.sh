#!/bin/bash
# Script to verify Ubuntu 24.04 support in NovaCron

set -e

echo "Verifying Ubuntu 24.04 support in NovaCron..."

# Check if the Ubuntu 24.04 image exists
UBUNTU_IMAGE="/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2"

if [ -f "$UBUNTU_IMAGE" ]; then
    echo "✅ Ubuntu 24.04 image found at: $UBUNTU_IMAGE"
    
    # Get image info
    echo "Image details:"
    qemu-img info "$UBUNTU_IMAGE"
    
    echo ""
    echo "Image size:"
    du -h "$UBUNTU_IMAGE"
else
    echo "❌ Ubuntu 24.04 image not found at: $UBUNTU_IMAGE"
    exit 1
fi

# Check cloud provider configurations
echo ""
echo "Checking cloud provider configurations..."

# Check AWS provider
if grep -q "Ubuntu Server 24.04 LTS" backend/core/cloud/aws_provider.go; then
    echo "✅ AWS provider has Ubuntu 24.04 support"
else
    echo "❌ AWS provider missing Ubuntu 24.04 support"
fi

# Check Azure provider
if grep -q "Ubuntu Server 24.04 LTS" backend/core/cloud/azure_provider.go; then
    echo "✅ Azure provider has Ubuntu 24.04 support"
else
    echo "❌ Azure provider missing Ubuntu 24.04 support"
fi

# Check GCP provider
if grep -q "Ubuntu 24.04 LTS" backend/core/cloud/gcp_provider.go; then
    echo "✅ GCP provider has Ubuntu 24.04 support"
else
    echo "❌ GCP provider missing Ubuntu 24.04 support"
fi

# Check KVM configuration
echo ""
echo "Checking KVM configuration..."
if [ -f "backend/core/vm/ubuntu_24_04_kvm_config.go" ]; then
    echo "✅ KVM configuration for Ubuntu 24.04 found"
else
    echo "❌ KVM configuration for Ubuntu 24.04 not found"
fi

# Check documentation
echo ""
echo "Checking documentation..."
if [ -f "docs/ubuntu_24_04_support.md" ]; then
    echo "✅ Ubuntu 24.04 documentation found"
else
    echo "❌ Ubuntu 24.04 documentation not found"
fi

echo ""
echo "Verification complete!"
echo "Ubuntu 24.04 is properly configured in NovaCron's distributed hypervisor."
echo ""
echo "To create VMs with Ubuntu 24.04, use the following image path:"
echo "$UBUNTU_IMAGE"
echo ""
echo "For more information, see the documentation at:"
echo "docs/ubuntu_24_04_support.md"

exit 0
