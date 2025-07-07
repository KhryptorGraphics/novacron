#!/bin/bash
# Comprehensive script to test Ubuntu 24.04 support in NovaCron production environment

set -e

# Configuration
API_ENDPOINT="http://localhost:8090"
IMAGE_PATH="/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2"
VM_NAME="ubuntu-24-04-test-$(date +%s)"
LOG_FILE="test-results-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print status messages
print_status() {
    echo -e "${GREEN}[+] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> $LOG_FILE
}

print_warning() {
    echo -e "${YELLOW}[!] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> $LOG_FILE
}

print_error() {
    echo -e "${RED}[-] $1${NC}"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> $LOG_FILE
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

# Start logging
echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] Starting NovaCron production test with Ubuntu 24.04 support" > $LOG_FILE

# Test 1: Check if services are running
print_status "Test 1: Checking if services are running..."

# Check hypervisor service
hypervisor_status=$(systemctl is-active novacron-hypervisor.service)
if [ "$hypervisor_status" = "active" ]; then
    print_status "Hypervisor service is running"
else
    print_error "Hypervisor service is not running"
    systemctl status novacron-hypervisor.service --no-pager >> $LOG_FILE
fi

# Check API service
api_status=$(systemctl is-active novacron-api.service)
if [ "$api_status" = "active" ]; then
    print_status "API service is running"
else
    print_error "API service is not running"
    systemctl status novacron-api.service --no-pager >> $LOG_FILE
fi

# Check monitor service
monitor_status=$(systemctl is-active novacron-ubuntu-24-04-monitor.service)
if [ "$monitor_status" = "active" ]; then
    print_status "Monitor service is running"
else
    print_error "Monitor service is not running"
    systemctl status novacron-ubuntu-24-04-monitor.service --no-pager >> $LOG_FILE
fi

# Test 2: Check API health
print_status "Test 2: Checking API health..."
API_HEALTH=$(make_request "GET" "/health")
if [[ "$API_HEALTH" == *"ok"* ]]; then
    print_status "API health check passed"
else
    print_error "API health check failed"
    echo "$API_HEALTH" >> $LOG_FILE
fi

# Test 3: Check hypervisor health
print_status "Test 3: Checking hypervisor health..."
HYPERVISOR_HEALTH=$(curl -s "http://localhost:9000/health")
if [[ "$HYPERVISOR_HEALTH" == *"ok"* ]]; then
    print_status "Hypervisor health check passed"
else
    print_error "Hypervisor health check failed"
    echo "$HYPERVISOR_HEALTH" >> $LOG_FILE
fi

# Test 4: Check if Ubuntu 24.04 image exists
print_status "Test 4: Checking if Ubuntu 24.04 image exists..."
if [ -f "$IMAGE_PATH" ]; then
    print_status "Ubuntu 24.04 image exists"
    print_status "Image size: $(du -h "$IMAGE_PATH" | cut -f1)"
    print_status "Image details: $(qemu-img info --output=json "$IMAGE_PATH" | jq -c .)"
else
    print_error "Ubuntu 24.04 image does not exist"
    exit 1
fi

# Test 5: Get available OS templates
print_status "Test 5: Getting available OS templates..."
OS_TEMPLATES=$(make_request "GET" "/api/v1/templates")
echo "$OS_TEMPLATES" | jq . >> $LOG_FILE

# Check if Ubuntu 24.04 is in the templates
if [[ "$OS_TEMPLATES" == *"Ubuntu 24.04"* ]]; then
    print_status "Ubuntu 24.04 is available in the templates"
else
    print_warning "Ubuntu 24.04 is not available in the templates"
fi

# Test 6: Create a VM with Ubuntu 24.04
print_status "Test 6: Creating a VM with Ubuntu 24.04..."
CREATE_PAYLOAD=$(cat <<EOF
{
  "name": "$VM_NAME",
  "vcpus": 2,
  "memory_mb": 2048,
  "disk_mb": 20480,
  "image": "$IMAGE_PATH",
  "network_config": {
    "interfaces": [
      {
        "network_id": "default"
      }
    ]
  },
  "tags": ["ubuntu", "24.04", "lts", "test"]
}
EOF
)

CREATE_RESPONSE=$(make_request "POST" "/api/v1/vms" "$CREATE_PAYLOAD")
echo "$CREATE_RESPONSE" | jq . >> $LOG_FILE

# Extract VM ID from response
VM_ID=$(echo "$CREATE_RESPONSE" | jq -r '.id')

if [ "$VM_ID" == "null" ] || [ -z "$VM_ID" ]; then
    print_error "Failed to create VM"
    exit 1
fi

print_status "VM created with ID: $VM_ID"

# Wait for VM to be created
print_status "Waiting for VM to be created..."
sleep 10

# Test 7: Get VM details
print_status "Test 7: Getting VM details..."
VM_DETAILS=$(make_request "GET" "/api/v1/vms/$VM_ID")
echo "$VM_DETAILS" | jq . >> $LOG_FILE

# Check if VM has Ubuntu 24.04 image
VM_IMAGE=$(echo "$VM_DETAILS" | jq -r '.spec.image')
if [[ "$VM_IMAGE" == *"ubuntu-24.04"* ]]; then
    print_status "VM has Ubuntu 24.04 image"
else
    print_warning "VM does not have Ubuntu 24.04 image"
    print_warning "VM image: $VM_IMAGE"
fi

# Test 8: Start VM
print_status "Test 8: Starting VM..."
START_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/start")
echo "$START_RESPONSE" | jq . >> $LOG_FILE

# Wait for VM to start
print_status "Waiting for VM to start..."
sleep 20

# Test 9: Get VM status
print_status "Test 9: Getting VM status..."
VM_STATUS=$(make_request "GET" "/api/v1/vms/$VM_ID")
echo "$VM_STATUS" | jq . >> $LOG_FILE

# Check if VM is running
VM_STATE=$(echo "$VM_STATUS" | jq -r '.state')
if [ "$VM_STATE" == "running" ]; then
    print_status "VM is running"
else
    print_warning "VM is not running"
    print_warning "VM state: $VM_STATE"
fi

# Test 10: Get VM console log
print_status "Test 10: Getting VM console log..."
CONSOLE_LOG=$(make_request "GET" "/api/v1/vms/$VM_ID/console")
echo "$CONSOLE_LOG" | head -n 50 >> $LOG_FILE

# Check if Ubuntu 24.04 is mentioned in the console log
if [[ "$CONSOLE_LOG" == *"Ubuntu 24.04"* ]] || [[ "$CONSOLE_LOG" == *"noble"* ]]; then
    print_status "Ubuntu 24.04 is mentioned in the console log"
else
    print_warning "Ubuntu 24.04 is not mentioned in the console log"
fi

# Test 11: Stop VM
print_status "Test 11: Stopping VM..."
STOP_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/stop")
echo "$STOP_RESPONSE" | jq . >> $LOG_FILE

# Wait for VM to stop
print_status "Waiting for VM to stop..."
sleep 20

# Test 12: Get VM status after stop
print_status "Test 12: Getting VM status after stop..."
VM_STATUS=$(make_request "GET" "/api/v1/vms/$VM_ID")
echo "$VM_STATUS" | jq . >> $LOG_FILE

# Check if VM is stopped
VM_STATE=$(echo "$VM_STATUS" | jq -r '.state')
if [ "$VM_STATE" == "stopped" ]; then
    print_status "VM is stopped"
else
    print_warning "VM is not stopped"
    print_warning "VM state: $VM_STATE"
fi

# Test 13: Delete VM
print_status "Test 13: Deleting VM..."
DELETE_RESPONSE=$(make_request "DELETE" "/api/v1/vms/$VM_ID")
echo "$DELETE_RESPONSE" | jq . >> $LOG_FILE

# Wait for VM to be deleted
print_status "Waiting for VM to be deleted..."
sleep 10

# Test 14: Verify VM is deleted
print_status "Test 14: Verifying VM is deleted..."
VM_GET_RESPONSE=$(make_request "GET" "/api/v1/vms/$VM_ID")
if [[ "$VM_GET_RESPONSE" == *"not found"* ]]; then
    print_status "VM is deleted"
else
    print_warning "VM is not deleted"
    echo "$VM_GET_RESPONSE" | jq . >> $LOG_FILE
fi

# Test 15: Check monitoring service
print_status "Test 15: Checking monitoring service..."
MONITOR_LOG=$(journalctl -u novacron-ubuntu-24-04-monitor.service -n 50 --no-pager)
echo "$MONITOR_LOG" | tail -n 20 >> $LOG_FILE

if [[ "$MONITOR_LOG" == *"monitoring"* ]]; then
    print_status "Monitoring service is working"
else
    print_warning "Monitoring service may not be working properly"
fi

# Summarize test results
print_status "Test completed!"
print_status "Summary:"
print_status "- Services: $([ "$hypervisor_status" = "active" ] && [ "$api_status" = "active" ] && [ "$monitor_status" = "active" ] && echo "All running" || echo "Some not running")"
print_status "- API Health: $([ "$API_HEALTH" == *"ok"* ] && echo "OK" || echo "Failed")"
print_status "- Hypervisor Health: $([ "$HYPERVISOR_HEALTH" == *"ok"* ] && echo "OK" || echo "Failed")"
print_status "- Ubuntu 24.04 Image: $([ -f "$IMAGE_PATH" ] && echo "Available" || echo "Not available")"
print_status "- VM Creation: $([ "$VM_ID" != "null" ] && [ -n "$VM_ID" ] && echo "Successful" || echo "Failed")"
print_status "- VM Lifecycle: $([ "$VM_STATE" == "stopped" ] && echo "Successful" || echo "Issues detected")"
print_status "- VM Deletion: $([ "$VM_GET_RESPONSE" == *"not found"* ] && echo "Successful" || echo "Failed")"

print_status "Test results saved to $LOG_FILE"

exit 0
