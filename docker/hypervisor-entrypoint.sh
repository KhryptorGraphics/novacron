#!/bin/sh
# NovaCron Hypervisor Entrypoint Script
set -e

# Check if we're running with sufficient privileges
if [ ! -e /dev/kvm ] && [ ! -e /dev/null/kvm ]; then
    echo "WARNING: KVM device not found. Hardware virtualization may not be available."
    echo "         Container may need to be run with --privileged flag."
fi

# Create config file if it doesn't exist
if [ ! -f /etc/novacron/config.yaml ]; then
    echo "Creating default configuration file..."
    mkdir -p /etc/novacron
    cat > /etc/novacron/config.yaml << EOF
# NovaCron Hypervisor Configuration
nodeId: ${NODE_ID:-node1}
logLevel: ${LOG_LEVEL:-info}
storagePath: ${STORAGE_PATH:-/var/lib/novacron/vms}
clusterAddr: ${CLUSTER_ADDR:-novacron-api:8080}
api:
  host: 0.0.0.0
  port: 9000
  tlsEnabled: false
vm:
  defaultMemory: 2048
  defaultCpus: 2
  defaultDiskSize: 20
  supportedDrivers:
    - kvm
    - containerd
  memoryOvercommitRatio: 1.2
migration:
  storagePath: ${STORAGE_PATH:-/var/lib/novacron/vms}/migrations
  defaultType: cold
  liveIterations: 5
  compressionLevel: 6
  defaultBandwidthLimit: 0
EOF
    echo "Configuration file created at /etc/novacron/config.yaml"
fi

# Create storage directory if it doesn't exist
if [ ! -d ${STORAGE_PATH:-/var/lib/novacron/vms} ]; then
    echo "Creating VM storage directory..."
    mkdir -p ${STORAGE_PATH:-/var/lib/novacron/vms}
    mkdir -p ${STORAGE_PATH:-/var/lib/novacron/vms}/migrations
fi

# Check libvirt connectivity
if which virsh >/dev/null 2>&1; then
    echo "Checking libvirt connectivity..."
    if ! virsh -c qemu:///system list >/dev/null 2>&1; then
        echo "WARNING: Cannot connect to libvirt. KVM-based VMs will not be available."
        echo "         Please ensure libvirt is properly configured."
    else
        echo "Libvirt connection successful."
    fi
else
    echo "WARNING: virsh not found. KVM management will not be available."
fi

# Check containerd connectivity
if which ctr >/dev/null 2>&1; then
    echo "Checking containerd connectivity..."
    if ! ctr version >/dev/null 2>&1; then
        echo "WARNING: Cannot connect to containerd. Container-based VMs will not be available."
    else
        echo "Containerd connection successful."
    fi
else
    echo "INFO: ctr not found. Container-based VMs may not be available."
fi

# Print startup message
echo "Starting NovaCron Hypervisor..."
echo "Node ID: ${NODE_ID:-node1}"
echo "Log Level: ${LOG_LEVEL:-info}"
echo "Storage Path: ${STORAGE_PATH:-/var/lib/novacron/vms}"
echo "Cluster Address: ${CLUSTER_ADDR:-novacron-api:8080}"

# Start the application
exec "$@"
