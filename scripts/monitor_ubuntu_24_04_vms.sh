#!/bin/bash
# Script to monitor Ubuntu 24.04 VMs and make performance adjustments

set -e

# Configuration
API_ENDPOINT="http://localhost:8090"
MONITORING_INTERVAL=60  # seconds
LOG_FILE="/var/log/novacron/ubuntu_24_04_monitoring.log"
ALERT_THRESHOLD_CPU=80  # percentage
ALERT_THRESHOLD_MEMORY=80  # percentage
ALERT_THRESHOLD_DISK=80  # percentage

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+] $1${NC}"
    echo "[+] $(date +"%Y-%m-%d %H:%M:%S") - $1" >> "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
    echo "[!] $(date +"%Y-%m-%d %H:%M:%S") - $1" >> "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
    echo "[-] $(date +"%Y-%m-%d %H:%M:%S") - $1" >> "$LOG_FILE"
}

# Function to make API requests
make_request() {
    local method=$1
    local endpoint=$2
    local data=$3
    
    if [ -n "$data" ]; then
        curl -s -X "$method" -H "Content-Type: application/json" -d "$data" "$API_ENDPOINT$endpoint"
    else
        curl -s -X "$method" "$API_ENDPOINT$endpoint"
    fi
}

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"

print_status "Starting Ubuntu 24.04 VM monitoring..."
print_status "Monitoring interval: $MONITORING_INTERVAL seconds"
print_status "Log file: $LOG_FILE"

# Main monitoring loop
while true; do
    print_status "Fetching VM list..."
    VM_LIST=$(make_request "GET" "/api/v1/vms")
    
    # Filter for Ubuntu 24.04 VMs
    UBUNTU_24_04_VMS=$(echo "$VM_LIST" | jq '[.[] | select(.spec.image | contains("ubuntu-24.04"))]')
    VM_COUNT=$(echo "$UBUNTU_24_04_VMS" | jq 'length')
    
    print_status "Found $VM_COUNT Ubuntu 24.04 VMs"
    
    # Process each VM
    echo "$UBUNTU_24_04_VMS" | jq -c '.[]' | while read -r VM; do
        VM_ID=$(echo "$VM" | jq -r '.id')
        VM_NAME=$(echo "$VM" | jq -r '.name')
        VM_STATE=$(echo "$VM" | jq -r '.state')
        
        print_status "Monitoring VM: $VM_NAME ($VM_ID) - State: $VM_STATE"
        
        # Skip VMs that are not running
        if [ "$VM_STATE" != "running" ]; then
            print_warning "VM $VM_NAME is not running, skipping..."
            continue
        }
        
        # Get VM metrics
        VM_METRICS=$(make_request "GET" "/api/v1/vms/$VM_ID/metrics")
        
        # Extract metrics
        CPU_USAGE=$(echo "$VM_METRICS" | jq -r '.cpu_usage_percent')
        MEMORY_USAGE=$(echo "$VM_METRICS" | jq -r '.memory_usage_percent')
        DISK_USAGE=$(echo "$VM_METRICS" | jq -r '.disk_usage_percent')
        
        print_status "VM $VM_NAME metrics:"
        print_status "  CPU: $CPU_USAGE%"
        print_status "  Memory: $MEMORY_USAGE%"
        print_status "  Disk: $DISK_USAGE%"
        
        # Check for high CPU usage
        if (( $(echo "$CPU_USAGE > $ALERT_THRESHOLD_CPU" | bc -l) )); then
            print_warning "High CPU usage detected for VM $VM_NAME: $CPU_USAGE%"
            
            # Check if VM has less than 4 vCPUs
            VM_VCPU=$(echo "$VM" | jq -r '.spec.vcpu')
            if [ "$VM_VCPU" -lt 4 ]; then
                print_warning "VM $VM_NAME has only $VM_VCPU vCPUs, considering upgrade"
                
                # In a real environment, you might want to:
                # 1. Notify administrators
                # 2. Automatically scale up resources
                # 3. Migrate to a more powerful node
            }
        }
        
        # Check for high memory usage
        if (( $(echo "$MEMORY_USAGE > $ALERT_THRESHOLD_MEMORY" | bc -l) )); then
            print_warning "High memory usage detected for VM $VM_NAME: $MEMORY_USAGE%"
            
            # Check if VM has less than 4GB memory
            VM_MEMORY=$(echo "$VM" | jq -r '.spec.memory_mb')
            if [ "$VM_MEMORY" -lt 4096 ]; then
                print_warning "VM $VM_NAME has only $VM_MEMORY MB memory, considering upgrade"
            }
        }
        
        # Check for high disk usage
        if (( $(echo "$DISK_USAGE > $ALERT_THRESHOLD_DISK" | bc -l) )); then
            print_warning "High disk usage detected for VM $VM_NAME: $DISK_USAGE%"
            
            # Check if VM has less than 50GB disk
            VM_DISK=$(echo "$VM" | jq -r '.spec.disk_mb')
            if [ "$VM_DISK" -lt 51200 ]; then
                print_warning "VM $VM_NAME has only $VM_DISK MB disk, considering upgrade"
            }
        }
    done
    
    print_status "Monitoring cycle completed, sleeping for $MONITORING_INTERVAL seconds..."
    sleep "$MONITORING_INTERVAL"
done
