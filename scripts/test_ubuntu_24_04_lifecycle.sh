#!/bin/bash
# Script to test the full lifecycle of an Ubuntu 24.04 VM

set -e

# Configuration
API_ENDPOINT="http://localhost:8090"
IMAGE_PATH="/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2"
VM_NAME="ubuntu-24-04-lifecycle-test"

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

# Step 1: Create VM
print_status "Step 1: Creating Ubuntu 24.04 VM..."
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
echo "$CREATE_RESPONSE" | jq .

# Extract VM ID from response
VM_ID=$(echo "$CREATE_RESPONSE" | jq -r '.id')

if [ "$VM_ID" == "null" ] || [ -z "$VM_ID" ]; then
    print_error "Failed to create VM"
    exit 1
fi

print_status "VM created with ID: $VM_ID"

# Wait for VM to be created
print_status "Waiting for VM to be created..."
sleep 5

# Step 2: Get VM details
print_status "Step 2: Getting VM details..."
VM_DETAILS=$(make_request "GET" "/api/v1/vms/$VM_ID")
echo "$VM_DETAILS" | jq .

# Step 3: Start VM
print_status "Step 3: Starting VM..."
START_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/start")
echo "$START_RESPONSE" | jq .

# Wait for VM to start
print_status "Waiting for VM to start..."
sleep 10

# Step 4: Get VM status
print_status "Step 4: Getting VM status..."
VM_STATUS=$(make_request "GET" "/api/v1/vms/$VM_ID")
echo "$VM_STATUS" | jq .

# Step 5: Stop VM
print_status "Step 5: Stopping VM..."
STOP_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/stop")
echo "$STOP_RESPONSE" | jq .

# Wait for VM to stop
print_status "Waiting for VM to stop..."
sleep 10

# Step 6: Get VM status
print_status "Step 6: Getting VM status after stop..."
VM_STATUS=$(make_request "GET" "/api/v1/vms/$VM_ID")
echo "$VM_STATUS" | jq .

# Step 7: Restart VM
print_status "Step 7: Restarting VM..."
RESTART_RESPONSE=$(make_request "POST" "/api/v1/vms/$VM_ID/restart")
echo "$RESTART_RESPONSE" | jq .

# Wait for VM to restart
print_status "Waiting for VM to restart..."
sleep 15

# Step 8: Get VM status
print_status "Step 8: Getting VM status after restart..."
VM_STATUS=$(make_request "GET" "/api/v1/vms/$VM_ID")
echo "$VM_STATUS" | jq .

# Step 9: Delete VM
print_status "Step 9: Deleting VM..."
DELETE_RESPONSE=$(make_request "DELETE" "/api/v1/vms/$VM_ID")
echo "$DELETE_RESPONSE" | jq .

print_status "Lifecycle test completed successfully!"
print_status "Ubuntu 24.04 VM lifecycle test passed"

exit 0
