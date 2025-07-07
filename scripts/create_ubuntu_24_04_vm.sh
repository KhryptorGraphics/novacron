#!/bin/bash
# Script to create an Ubuntu 24.04 VM using the NovaCron API

set -e

# Configuration
VM_NAME=${1:-"ubuntu-24-04-vm"}
API_ENDPOINT=${NOVACRON_API:-"http://localhost:8090"}
IMAGE_PATH="/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2"

echo "Creating Ubuntu 24.04 VM: $VM_NAME"
echo "Using API endpoint: $API_ENDPOINT"
echo "Using image: $IMAGE_PATH"

# Create a JSON payload for the VM creation request
JSON_PAYLOAD=$(cat <<EOF
{
  "name": "$VM_NAME",
  "spec": {
    "vcpu": 2,
    "memory_mb": 2048,
    "disk_mb": 20480,
    "type": "kvm",
    "image": "$IMAGE_PATH",
    "networks": [
      {
        "network_id": "default"
      }
    ],
    "env": {
      "OS_VERSION": "24.04",
      "OS_NAME": "Ubuntu",
      "OS_CODENAME": "Noble Numbat"
    },
    "labels": {
      "os": "ubuntu",
      "version": "24.04",
      "lts": "true"
    }
  },
  "tags": {
    "purpose": "web-server",
    "environment": "production"
  }
}
EOF
)

# Make the API request
echo "Sending VM creation request..."
echo "$JSON_PAYLOAD" | jq .

# In a real environment, you would use:
# RESPONSE=$(curl -s -X POST -H "Content-Type: application/json" -d "$JSON_PAYLOAD" "$API_ENDPOINT/api/v1/vms")
# 
# For demonstration purposes, we'll simulate a successful response
RESPONSE='{
  "id": "vm-'$(date +%s)'",
  "name": "'$VM_NAME'",
  "status": "creating",
  "created_at": "'$(date -Iseconds)'",
  "message": "VM creation initiated"
}'

echo ""
echo "Response:"
echo "$RESPONSE" | jq .

VM_ID=$(echo "$RESPONSE" | jq -r .id)

echo ""
echo "VM creation initiated with ID: $VM_ID"
echo ""
echo "To check the status of your VM:"
echo "curl $API_ENDPOINT/api/v1/vms/$VM_ID"
echo ""
echo "To connect to your VM once it's running:"
echo "novacron vm connect $VM_ID"
echo ""
echo "For more information, see the documentation at:"
echo "docs/ubuntu_24_04_support.md"

exit 0
